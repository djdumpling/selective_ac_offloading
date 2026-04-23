# Observations from Simulator Development

## 1. Korthikanti Selective AC Is Obsolete with FlashAttention

Korthikanti et al. (MLSys 2023) proposed recomputing the attention core (QK^T, softmax,
dropout) — the cheapest operations that consume the most memory (the 5as²b quadratic term).
This was the right strategy pre-FlashAttention.

With FlashAttention active, the quadratic tensors are never materialized. FA's backward
kernel recomputes attention blocks on-the-fly from Q, K, V. The Korthikanti selective
strategy therefore has **nothing to recompute** — it degenerates to No AC.

Our simulator confirms this: Selective AC produces identical activation memory and step
latency to No AC for every FA-enabled model (Llama-7B/13B/70B).

**This observation is not formalized in any published paper.** Korthikanti's paper predates
widespread FA adoption, and no subsequent work has re-derived the optimal selective strategy
for the FA era.

## 2. The MLP Block Is the New Bottleneck

With FA eliminating the attention quadratic term, per-layer activation memory becomes:

```
Attention (with FA+RoPE): ~14 sbh / tp   (Q, K, V, rotary Q/K, FA output, o_proj input, FA logsumexp)
MLP (SwiGLU):              ~4 × ffn × sb × bpe / tp + sbh × bpe / sp  (gate, up, silu_out, linear2, input)
LayerNorm:                 2 × sbh × 4 / sp   (fp32 inputs)
```

For Llama-7B (MHA, h=4096, ffn=11008, tp=1): attention is ~14 × 4096 × 2 = 112K elements
per token, MLP is ~4 × 11008 × 2 + 4096 × 2 = 96K elements per token. They're roughly
balanced — but with **GQA** (e.g., Qwen3-8B: 8 KV heads), K/V tensors shrink 4×, making
attention even cheaper and MLP even more dominant (~46% of per-layer activation memory).

The MLP is the primary target for cheap recomputation — specifically `mlp_silu_output`, the
SiLU activation output saved by the elementwise multiply backward (see Bug Fix 5 below).
This holds for both MHA and GQA architectures.

## 3. FA-Era Selective AC: The New Practical Default

The FA-era analogue of Korthikanti's strategy: checkpoint the pointwise activation function,
which eliminates `mlp_silu_output` (the SiLU output saved by the elementwise multiply
backward), while keeping the matmul outputs (gate_output, up_output) which are expensive
to recompute.  See Bug Fix 6 below for the corrected modeling.

**Simulator predictions and GPU measurements:**

| Model                      | Memory Saved | Step Overhead (sim) | Step Overhead (GPU) |
|----------------------------|-------------:|--------------------:|--------------------:|
| Llama-7B (H100, single GPU)|       -11.9% |              +0.00% |            **+0.9%** |
| Qwen3-8B (H100, single GPU)|       -11.4% |              +0.00% |            **+1.1%** |
| Llama-7B (A100, FSDP dp=8) |       -14.5% |              +0.00% |                  — |
| Llama-13B (A100, FSDP dp=8)|       -14.5% |              +0.00% |                  — |
| Llama-70B (H100, TP=8+dp=8)|       -18.2% |              +0.00% |                  — |
| GPT-3 175B (A100, TP=8+PP=8)|      -11.6% |              +0.00% |                  — |

This is a genuine free lunch: the recomputed operation (SiLU + elementwise multiply for
SwiGLU) is so cheap relative to the surrounding matmuls that the measured overhead is <2%.
The simulator predicts 0% because the recompute FLOPs are negligible; the measured ~1%
comes from `torch.utils.checkpoint` bookkeeping (extra autograd nodes), not from the
recomputed FLOPs themselves.

This mirrors Korthikanti's principle exactly — **recompute the cheapest operation that frees
the most memory** — but applied to the post-FA bottleneck.

## 4. 3-Resource Joint Optimization: Real but Costly

With honest cost modeling (compression compute + PCIe contention), the 3-Resource strategy
(offload + compress + recompute) costs **+1.6–3.2% step latency**:

- **Compression is not free.** Low-rank projection of an [sb, d] tensor at rank r costs
  4 × sb × d × r FLOPs (two matmuls for compress + decompress). For Llama-7B with
  r = h/8 = 512 and ffn = 11008, this is ~180 GFLOPs per layer, or ~5.8 TFLOPs total.
- **PCIe bandwidth is contested.** FSDP NCCL collectives (all-gather, reduce-scatter)
  consume ~40% of PCIe bandwidth on multi-node setups, reducing effective offload bandwidth
  from 32 GB/s to ~19 GB/s on A100.

The 3-Resource strategy saves ~30% more activation memory beyond FA-Selective, but at a
real latency cost. It's a Pareto extension, not a free improvement.

## 5. Pipeline-Position-Aware AC: Orthogonal and Composable

Under 1F1B pipeline parallelism, stage s (from the end) stashes (s−1) in-flight
microbatch activations. The first stage pays (PP−1)× the stashing memory of the last stage.

This means:
- Early pipeline stages should use aggressive AC (every byte saved is multiplied)
- Late pipeline stages should use minimal AC (no stashing penalty)

This observation is **orthogonal to the FA-era analysis** — it composes with any per-layer
strategy. No existing system (Megatron-LM, Alpa, NEST) accounts for this.

## 6. Pipeline-Aware AC: Two Regimes

The simulator implementation revealed two distinct regimes for pipeline-aware AC:

### Regime 1: Memory-constrained (Llama-7B PP=4, large batch, A100-80GB)

Activation stashing creates real differential pressure across stages:
- Stage 0 (stashes 3 microbatches): 75 GB with FA-Selective → OOM → forced into Full AC
- Stages 1-3 (stash 0-2): can use No AC and fit comfortably

Pipeline-aware correctly assigns Full AC to stage 0 and No AC to stages 1-3. However,
**throughput does not improve** — the pipeline runs at the speed of its slowest stage, and
stage 0 (Full AC, 24.6% overhead) is the bottleneck. Late stages finishing faster is wasted.

The value in this regime is **feasibility, not throughput**: pipeline-aware ensures all stages
fit in memory without forcing unnecessary overhead on stages that don't need it. If a
better-than-Full-AC option existed for stage 0 (e.g., FA-Selective with partial offloading),
the throughput bottleneck would relax.

### Regime 2: Memory-comfortable (GPT-3 175B PP=8, TP=8, A100-80GB)

With high PP and TP, activations per stage are so small (~2.4 GB) that No AC fits on all
stages, even with 7 stashed microbatches. Pipeline-aware correctly selects No AC everywhere.

The "gain" vs. uniform Full AC (+20.28% faster) is real but trivial — it's just the gain
from not using Full AC when it's unnecessary. Any reasonable practitioner would already
choose No AC in this setting.

### Regime 3: The Sweet Spot (found via systematic search)

A systematic search over (model, PP, seq_len, micro_batch_size) revealed the sweet spot:
**medium-sized models with high PP depth** where per-stage activations are small enough that
FA-Selective fits on the bottleneck stage, but total stash makes No AC overflow.

**Llama-7B, PP=8, dp=4, A100-80GB (the breakthrough cases):**

| Config               | Uniform Full AC       | Pipeline-Aware        | Throughput Gain |
|----------------------|----------------------:|----------------------:|----------------:|
| seq=2048, mbs=8      | 684.6 ms (23.9% ovhd) | 552.4 ms (0.0% ovhd) | **+23.9%**      |
| seq=4096, mbs=4      | 741.0 ms (24.6% ovhd) | 594.7 ms (0.0% ovhd) | **+24.6%**      |
| seq=8192, mbs=2      | 853.7 ms (25.7% ovhd) | 679.3 ms (0.0% ovhd) | **+25.7%**      |

Per-stage assignment: `['FA-Selective', 'No AC', 'No AC', 'No AC', 'No AC', 'No AC', 'No AC', 'No AC']`

Why PP=8 works when PP=4 didn't: with PP=8, each stage holds only 4 layers (32/8) instead
of 8 (32/4). Each stashed microbatch is half the size. So even though stage 0 stashes 7
microbatches (more than PP=4's 3), each is smaller. FA-Selective's 14.5% savings squeezes
the peak to 69.7 GB / 72.0 GB budget — just barely fits. No AC would be ~81 GB → OOM.

The throughput gain is nearly the full elimination of Full AC overhead (~24%), because
FA-Selective's recompute cost (pointwise SiLU + multiply) is negligible.

**Why smaller models (4B) don't help:** they're too comfortable — No AC fits everywhere,
so pipeline-aware trivially picks No AC on all stages. The framework doesn't add insight
because there's no memory tension. Pipeline-aware AC is most valuable for medium-sized
models on memory-constrained hardware with moderate pipeline depth — exactly the regime
where practitioners actually struggle with the AC/memory tradeoff.

### Summary Table (updated)

```
                        Uniform Full AC    Uniform FA-Sel    Pipeline-Aware
Llama-7B PP=4 mbs=4:
  All stages fit?       YES                NO (stage 0)      YES
  Bottleneck strategy:  Full AC            —                 Full AC
  Bottleneck overhead:  24.60%             —                 24.60%
  Overall step (ms):    1481.93            —                 1481.93
  Gain vs Uniform Full: —                  —                 +0.0%

Llama-7B PP=8 mbs=4:
  All stages fit?       YES                NO (stage 0)      YES
  Bottleneck strategy:  Full AC            —                 FA-Selective
  Bottleneck overhead:  24.60%             —                 0.00%
  Overall step (ms):    741.0              —                 594.7
  Gain vs Uniform Full: —                  —                 +24.6%  ← THE SWEET SPOT

GPT-3 175B PP=8:
  All stages fit?       YES                YES               YES
  Bottleneck strategy:  Full AC            No AC             No AC
  Bottleneck overhead:  20.28%             0.00%             0.00%
  Overall step (ms):    264.67             220.04            220.03
  Gain vs Uniform Full: —                  +20.3%            +20.3%
```

## 7. Multi-Schedule Pipeline Analysis

We implemented a multi-schedule pipeline model covering six schedules: 1F1B, 1F1B
Interleaved, ZB-H1, ZB-H2, ZB-V, and DualPipe. Each schedule has a different stash
profile, bubble fraction, and extra memory overhead. Pipeline-aware AC generalizes
across all of them.

### Key findings per schedule

**1F1B / 1F1B Interleaved / ZB-H1 / ZB-V:** These all share the same asymmetric stash
profile — stage p stashes (PP-1-p) microbatches. Pipeline-aware AC works identically
across them. The differences are in bubble fraction (1F1B > Interleaved > ZB = 0) and
communication patterns, but these don't affect the AC strategy selection.

**ZB-H2 (Zero Bubble with deferred W):** ZB-H2 defers weight gradient computations to
eliminate bubble, but this stores extra gradient memory per stage (e.g., 12.1 GB/stage
for Llama-7B PP=4, 40.5 GB/stage for GPT-3 175B PP=8). This extra memory eats into
activation headroom, sometimes pushing stages from No AC to FA-Selective. In the
Llama-7B PP=4 case, ZB-H2 was the only schedule that produced three different strategy
levels (Full AC / FA-Selective / No AC). For GPT-3 175B PP=8, the 40.5 GB/stage overhead
pushed ALL stages over budget, forcing uniform Full AC — worse than every other schedule.

**DualPipe (bidirectional):** DualPipe processes two micro-batch streams in opposite
directions. Each stage holds stash from both streams: (PP-1-p) from stream A + p from
stream B = PP-1 for all stages. The stash is **symmetric** — every stage faces identical
memory pressure. This eliminates the asymmetry that pipeline-aware AC exploits. With
DualPipe, pipeline-aware AC degenerates to uniform AC (same strategy everywhere).

### Schedule interaction summary

| Schedule | Stash Profile | Bubble | Pipeline-Aware Benefit |
|----------|---------------|--------|------------------------|
| 1F1B | Asymmetric (PP-1-p) | (PP-1)/M | Full: exploits asymmetry |
| 1F1B Interleaved | Asymmetric (same) | (PP-1)/(M×v) | Full: same stash, less bubble |
| ZB-H1 | Asymmetric (same) | ~0% | Full: same stash, zero bubble |
| ZB-H2 | Asymmetric (same) | 0% | Amplified: extra deferred-W memory creates more pressure |
| ZB-V | Asymmetric (same) | 0% | Full: same stash, zero bubble |
| DualPipe | Symmetric (all=PP-1) | ~0% | None: no asymmetry to exploit |

### Implication for the paper

Pipeline-aware AC is **schedule-agnostic in implementation but schedule-dependent in
impact.** This is a strength: the framework generalizes cleanly. The paper should present
it as a general technique and show results across schedules, noting that DualPipe's
symmetric design eliminates the opportunity while ZB-H2's deferred-W overhead creates
new opportunities.

## 8. Bug Fixes and Corrected Results

Seven bugs were identified and fixed. Fixes 1-4 were found via adversarial review. Fixes
5-7 were found via GPU validation (comparing simulator predictions against measured
activation memory on an H100 with PyTorch 2.10 and Transformers 5.5.4). The corrections
changed some numerical results but **strengthened rather than weakened** the core claims.

### Fix 1: Bubble fraction now affects pipeline throughput (High severity)

**Bug:** `bubble_fraction` was computed per schedule but never used — overall latency was
always `max(stage_latency)` regardless of schedule. All schedules showed identical step
times.

**Fix:** Overall latency = `bottleneck_per_microbatch × (1 + bubble_fraction)`.

**Impact on results:**

| Config (Llama-7B PP=4) | Before Fix | After Fix |
|-------------------------|------------|-----------|
| 1F1B step time | 1481.9 ms | **1759.8 ms** (+18.8% bubble) |
| 1F1B Interleaved step time | 1481.9 ms | **1620.9 ms** (+9.4% bubble) |
| ZB-H1 step time | 1481.9 ms | **1481.9 ms** (zero bubble, unchanged) |

| Config (GPT-3 175B PP=8) | Before Fix | After Fix |
|---------------------------|------------|-----------|
| 1F1B step time | 264.7 ms | **380.5 ms** (+43.8% bubble!) |
| ZB-H1 step time | 264.7 ms | **264.7 ms** (unchanged) |

**Implication for paper:** This fix makes the schedule comparison meaningful. The 1F1B → ZB
throughput improvement (44% for GPT-3 PP=8) was invisible before. It also changes the
pipeline-aware throughput gain calculation — when comparing pipeline-aware under 1F1B vs.
uniform Full AC under 1F1B, both include the same bubble overhead, so the relative gain
from AC strategy is unchanged. But comparing pipeline-aware across schedules now correctly
shows ZB's advantage.

### Fix 2: Korthikanti selective added to pipeline strategy search (High severity)

**Bug:** `STRATEGY_LEVELS` only had No AC, FA-Selective, and Full AC. For non-FA models
(GPT-3 175B), FA-Selective only recomputes `mlp_linear2_input` which doesn't address the
quadratic attention term. The pipeline-aware search jumped directly from "doesn't fit" to
Full AC (20%+ overhead), skipping Korthikanti selective (~2.7% overhead).

**Fix:** Added "Korthikanti Selective" between FA-Selective and Full AC. It recomputes
the attention core (QK^T, softmax, dropout output) — only effective without FA.

**Impact:** For non-FA models under memory pressure, pipeline-aware can now assign
Korthikanti Selective (~2.7% overhead) instead of Full AC (~20% overhead). This is a
large throughput improvement for pre-FA architectures.

### Fix 3: Sweet-spot search filter corrected (Medium severity)

**Bug:** Filter `len(set(strategies)) >= 1` is always true. Dozens of trivial "all No AC"
cases were reported as "breakthroughs" with +23% gains.

**Fix:** Require `len(set(strategies)) >= 2` (genuinely differentiated) AND bottleneck
avoids Full AC.

**Impact:** Search now correctly reports only 3 genuine sweet-spot cases (Llama-7B PP=8
with FA-Selective on stage 0, No AC on stages 1-7). The +24.6% gain in these cases was
always real — the fix just removed the noise around it.

### Fix 4: ZB-H2 per-stage extra memory (Medium severity)

**Bug:** ZB-H2 deferred-W memory was computed from stage 0's layer count and applied
uniformly. Stages with fewer layers (uneven PP division) were overcharged.

**Fix:** Compute per-stage from each stage's actual layer count.

**Impact:** Small — only affects uneven PP divisions (e.g., 32 layers / 3 stages). The
last stage now correctly gets less deferred-W overhead.

### Fix 5: Missing SiLU activation output tensor (High severity)

**Bug:** The MLP memory model accounted for four tensors per SwiGLU layer: `mlp_input`,
`mlp_gate_output`, `mlp_up_output`, and `mlp_linear2_input`. But PyTorch's autograd also
retains the SiLU activation output — the elementwise multiply `silu(gate) * up` saves
**both** its operands for backward: `up` (already modeled) and `silu(gate)` (missing).
This is a distinct tensor from `gate` because SiLU is non-linear.

**Fix:** Added `mlp_silu_output` (size: `s × b × ffn × bpe / tp`) and its GeLU equivalent
`mlp_gelu_output` to the MLP memory model.

**Impact:** ~1.34 GiB underestimate for Llama-7B (s=2048, b=1). This was the single
largest source of error in the GPU validation.

**Validated on GPU:** Confirmed via `saved_tensors_hooks` — exactly 4 ffn-sized (43 MiB)
bf16 tensors per layer for Llama-7B, 4 × 48 MiB for Qwen3-8B.

### Fix 6: FA-Selective recomputes the wrong tensor (High severity)

**Bug:** The FA-Selective strategy was modeled as recomputing `mlp_linear2_input` (=
`silu(gate) * up`). But in the real `torch.utils.checkpoint` implementation, the
checkpoint wraps the pointwise activation (`_silu_mul`), so what gets eliminated is the
**intermediate** `silu(gate)` inside the checkpoint — NOT `mlp_linear2_input`, which exits
the checkpoint and is saved by `down_proj`'s backward.

The old model got the **savings** right by coincidence (`silu(gate)` and `linear2_input`
are the same size), but the absolute memory levels were wrong for both No AC and
FA-Selective.

**Fix:** FA-Selective now recomputes `mlp_silu_output` / `mlp_gelu_output` instead of
`mlp_linear2_input`.

**Impact:** Combined with Fix 5, reduces No AC prediction error from -21.8% to -5.8% and
FA-Selective from -24.4% to -6.3% (before rotary fix).

**Validated on GPU:** FA-Selective correctly eliminates `mlp_silu_output` while retaining
`mlp_linear2_input`.  Measured savings (11.9% for Llama-7B, 11.4% for Qwen3-8B) match
simulator predictions.

### Fix 7: Missing rotary position embedding tensors (Medium severity)

**Bug:** For models using RoPE (Llama family), `apply_rotary_pos_emb` produces a
post-rotation Q and K that are saved by SDPA. These are distinct allocations from the
pre-rotation Q and K (saved by the rotary backward), but the simulator modeled only one Q
and one K tensor each.

**Fix:** Added `use_rotary_embeddings` config flag (True for Llama, False for GPT-3). When
active, adds `attn_rotary_q` and `attn_rotary_k` tensors.

**Impact:** ~1.0 GiB underestimate for Llama-7B (MHA). For GQA models (Llama-70B with
8 KV heads), the K rotary tensor is 8× smaller.

**Validated on GPU:** Using `torch.autograd.graph.saved_tensors_hooks`, we confirmed exactly
8 hidden-sized (16 MiB) bf16 tensors per layer on an H100 with PyTorch 2.10 + Transformers
5.5.4.  This matches the simulator's prediction (qkv_input, q, k, v, rotary_q, rotary_k,
out_proj_input, mlp_input).  Note: SDPA does NOT save its output O as a separate tensor in
PyTorch 2.10 — it either recomputes O in backward or shares the allocation with
`attn_out_proj_input`.  An earlier version of this fix included `attn_fa_output` as a
separate tensor but this was disproved by the GPU measurement (+2.8% overshoot → removed).

## 9. Qwen3-8B Validation: GQA and QK-Norm

Extending validation from Llama-7B (MHA) to Qwen3-8B (GQA + QK-norm) revealed two
architecture-specific features the simulator needed to model.

### Per-layer tensor comparison (from `saved_tensors_hooks` on H100)

| Size | Qwen3-8B (GQA) | Llama-7B (MHA) | What's different |
|---|---|---|---|
| 48 MiB × 4/layer | ffn activations | 43 MiB × 4 | Same count, larger ffn (12288 vs 11008) |
| 32 MiB × 5/layer | 2 LN fp32 + 2 params + **1 Q QK-norm fp32** | 32 MiB × 6/layer | **+1 QK-norm** `(1, 2048, 32, 128)` fp32 |
| 16 MiB × 7/layer | qkv_in, Q×2, **K_expanded**, **V_expanded**, o_proj, mlp_in | 16 MiB × 8/layer | GQA: K/V expanded 8→32 heads by `repeat_kv()` |
| 8 MiB × 3/layer | 2 params (k/v_proj) + **1 K QK-norm fp32** | — | **NEW**: K norm fp32 `(1, 2048, 8, 128)` |
| 4 MiB × 3/layer | K_pre, K_post, V_original | — | **NEW**: GQA small K/V (8 heads) |
| 0.2 MiB × 2/layer | Q norm rstd + FA logsumexp | 0.25 MiB × 1 | **NEW**: QK-norm rstd |
| 0.1 MiB × 1/layer | K norm rstd | — | **NEW**: QK-norm rstd |

### Feature 1: QK-Norm (RMSNorm on Q and K)

Qwen3 applies RMSNorm to Q and K after projection (`q_norm`, `k_norm`).  Each stores an
fp32 copy for backward (same as LayerNorm):
- Q norm: `[b, s, n_heads, head_dim]` fp32 = 32 MiB/layer
- K norm: `[b, s, kv_heads, head_dim]` fp32 = 8 MiB/layer (GQA-sized)

**Fix:** Added `use_qk_norm` config flag.  When active, adds `attn_q_norm_input` and
`attn_k_norm_input` tensors.

### Feature 2: GQA KV expansion (`repeat_kv`)

With GQA (8 KV heads, 32 Q heads), HuggingFace calls `repeat_kv()` to expand K/V from
`[b, 8, s, d_k]` → `[b, 32, s, d_k]` before SDPA.  SDPA saves the **expanded** versions
(16 MiB each), while the original small K/V (4 MiB each) also persist.

**Fix:** When `is_gqa` is True, adds `attn_k_expanded` and `attn_v_expanded` at full
head count size.

### Validation results

| Strategy | Llama-7B (MHA) | | Qwen3-8B (GQA) | |
|---|---|---|---|---|
| | Predicted | Error | Predicted | Error |
| No AC | 11.383 GiB | **-1.5%** | 14.774 GiB | **-0.4%** |
| FA-Selective | 10.039 GiB | **-1.4%** | 13.087 GiB | **-0.4%** |
| Full AC | 0.500 GiB | -8.8% | 0.562 GiB | -7.8% |

### Do the core claims still hold?

**Yes.** Summary of impact on each claim:

| Claim | Effect of Fixes |
|-------|-----------------|
| Korthikanti selective = no-op with FA | Unchanged — this is a structural property, not a numerical result |
| FA-Selective saves 12-18% at ~0% overhead | Unchanged — single-stage result, not affected by pipeline fixes |
| Pipeline-aware +24.6% in sweet spot | **Unchanged** — the per-microbatch bottleneck is the same; bubble affects absolute time but not relative gain |
| Multi-schedule comparison | **Corrected** — now shows meaningful differences between 1F1B (43.8% bubble for PP=8) and ZB (zero bubble) |
| 3-Resource costs 1.6-3.2% | Unchanged — single-stage result |
| DualPipe nullifies pipeline-aware | Unchanged — structural property |
| Non-FA pipeline-aware | **Improved** — Korthikanti selective now available, reducing bottleneck overhead from ~20% to ~2.7% |
| Memory formulas match GPU within 5% | **Validated** — Llama-7B: -1.5%, Qwen3-8B: -0.4% (H100, PyTorch 2.10) |

The fixes made the simulator more accurate without invalidating any core contribution.
The main numerical change is that 1F1B step times are now 19-44% higher (due to bubble),
which makes the ZB schedule comparison genuine and strengthens the argument for considering
pipeline schedules jointly with AC strategy.

## 10. GPU Validation Observations

### FA-Selective overhead is real but tiny

The simulator predicts 0.0% compute overhead for FA-Selective (the recomputed SiLU is
negligible relative to surrounding matmuls).  GPU measurements show:
- Llama-7B: **+0.9%** overhead (190.3 ms → 192.0 ms fwd+bwd)
- Qwen3-8B: **+1.1%** overhead (247.3 ms → 249.5 ms fwd+bwd)

The small measured overhead likely comes from `torch.utils.checkpoint` bookkeeping (extra
autograd nodes, re-execution of the checkpoint region), not from the recomputed FLOPs.
This is still a "free lunch" for practical purposes — 1% overhead for 11-12% memory savings.

### Full AC overhead is higher than predicted

The simulator predicts ~24% compute overhead for Full AC (recomputing one full forward pass
per layer).  GPU measurements show:
- Llama-7B: **+28.6%** overhead
- Qwen3-8B: **+48.5%** overhead (!)

The discrepancy grows with GQA.  Possible causes: (1) `repeat_kv` is recomputed inside the
checkpoint region (expanding K/V from 8→32 heads), adding work not modeled as "recompute";
(2) QK-norm adds recompute cost; (3) kernel launch overhead from re-executing many small
ops inside the checkpoint.  The Qwen3-8B case is striking — Full AC costs nearly 50% more
compute, making the case for FA-Selective even stronger.

### GQA amplifies MLP dominance

With GQA (8 KV heads), attention activation memory drops dramatically:
- Llama-7B (MHA): K=16 MiB, V=16 MiB per layer
- Qwen3-8B (GQA): K=4 MiB, V=4 MiB per layer (4× smaller)

But MLP activations are unchanged (or larger: ffn=12288 vs 11008).  For Qwen3-8B, the MLP
block accounts for **~46%** of per-layer activation memory vs ~37% for attention.  This
means FA-Selective (which targets MLP) is the right strategy regardless of whether the
model uses MHA or GQA.

### Full AC residual error (~8%) is from non-per-layer costs

Both models show ~8% error for Full AC (predicting 0.50-0.56 GiB vs measured 0.55-0.61 GiB).
The ~50-60 MiB gap is from tensors outside the per-layer model:
- Final RMSNorm fp32 copy: 32 MiB
- Embedding layer output: 16 MiB
- RMSNorm rstd tensors and other small buffers

This is not worth fixing — Full AC memory is so small (0.5-0.6 GiB) that the absolute
error is negligible for any practical memory budget calculation.

---

## Paper Formulation

### Title (working)

"Selective Activation Checkpointing in the FlashAttention Era"

or

"Rethinking Activation Checkpointing for Modern Transformer Training"

### Core Thesis

FlashAttention fundamentally changed the activation memory landscape. The dominant selective
recomputation strategy (Korthikanti et al.) targets the wrong tensors. We re-derive the
optimal strategy for FA-era transformers and show it extends naturally to pipeline-aware
non-uniform checkpointing.

### Contributions

1. **(Observation)** Formalize the FA-era bottleneck shift: with FlashAttention, the MLP
   block dominates activation memory and the Korthikanti selective strategy is a no-op.
   Re-derive per-tensor memory formulas under FA.

2. **(Algorithm)** FA-era selective AC: recompute activation functions (GeLU/SiLU), keep
   matmul outputs. 12–18% activation memory reduction at ~0% overhead. The same principle
   as Korthikanti — recompute cheap ops that free lots of memory — applied to the correct
   post-FA targets.

3. **(Algorithm)** Pipeline-position-aware AC: adapt checkpointing aggressiveness per
   pipeline stage based on the schedule-specific stash count. Generalizes across 1F1B,
   1F1B Interleaved, ZB-H1/H2/V, and DualPipe. In the sweet-spot regime (Llama-7B PP=8),
   achieves **+24.6% throughput** over uniform Full AC by assigning FA-Selective to the
   bottleneck stage and No AC to all others.

4. **(Analysis)** Multi-schedule interaction: show that pipeline-aware AC composes
   differently with each schedule — full benefit on asymmetric schedules (1F1B family, ZB),
   amplified by ZB-H2's deferred-W overhead, and nullified by DualPipe's symmetric stash.

5. **(System)** Analytical simulator with per-tensor granularity, realistic compression
   compute cost, PCIe contention modeling, and multi-schedule pipeline support. 64 tests
   validated against Korthikanti formulas and GPU measurements.

### Optional Extension (if time permits)

6. **(Algorithm)** Three-resource DP jointly optimizing recomputation, offloading, and
   compression. Show it extends the Pareto frontier at 1.6–3.2% latency cost.

### Venue Fit

- **MLSys**: Primary target. Algorithms + systems, empirical evaluation.
- **ASPLOS/OSDI**: If system implementation is strong enough.
- **Workshop paper (MLSys/NeurIPS)**: Contributions 1–3 alone are a solid workshop paper.

### Evaluation Plan

**Baselines:**
1. No AC (keep everything)
2. Full AC (recompute everything, ~24% overhead)
3. Korthikanti Selective AC (recompute attention core — show it's a no-op with FA)
4. PyTorch SAC Memory Budget API (min-cut/knapsack)

**Our methods:**
5. FA-era Selective AC (recompute activation functions)
6. Pipeline-position-aware AC (non-uniform across stages, multi-schedule)
7. FA-Selective + Pipeline-aware (composed — the main result)
8. (Optional) 3-Resource DP

**Key comparison for pipeline-aware:** Llama-7B PP=8 with varying (seq_len, mbs) to show
the sweet-spot regime where pipeline-aware + FA-Selective achieves +24% throughput over
uniform Full AC.

**Metrics:** Peak activation memory, training throughput (tokens/sec), recompute overhead (%),
Pareto frontier of memory vs. throughput.

**Models:** Llama-7B, Llama-13B, Llama-70B (FA-enabled, SwiGLU, MHA/GQA), Qwen3-8B
(FA + GQA + QK-norm), GPT-3 175B (no FA, GELU).

**Hardware:** A100-40GB, A100-80GB, H100-80GB. Varying TP, PP, DP configurations.

**Schedule comparison:** 1F1B, 1F1B Interleaved, ZB-H1, ZB-H2, ZB-V, DualPipe — show how
pipeline-aware AC interacts with each.

## 11. Realistic Training Configs: Sweet-Spot Validation

The initial sweet-spot analysis (Section 6) used Llama-7B PP=8 — a config no practitioner
would choose (PP=8 for a 7B model is 4 layers/stage, absurdly fragmented). To validate
that the sweet spot exists for **realistic training configurations**, we added verified
model architectures and parallelism configs from published technical reports, then re-ran
the pipeline-aware search.

### Model configs added (sources verified from technical reports and HuggingFace)

| Model | Layers | Hidden | Heads | KV Heads | FFN | FA | Act | Source |
|-------|--------|--------|-------|----------|-----|-----|-----|--------|
| Llama-3 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | Yes | SwiGLU | arxiv 2407.21783 |
| Llama-3.1 405B | 126 | 16384 | 128 | 8 (GQA) | 53248 | Yes | SwiGLU | arxiv 2407.21783 |
| GPT-NeoX-20B | 44 | 6144 | 64 | 64 (MHA) | 24576 | No | GeLU | arxiv 2204.06745 |
| BLOOM-176B | 70 | 14336 | 112 | 112 (MHA) | 57344 | No | GeLU | arxiv 2211.05100 |
| Falcon-180B | 80 | 14848 | 232 | 8 (GQA) | 59392 | Yes | GeLU | HuggingFace config |

### Results: Sweet spot on published training configs

**Llama-3 70B (Meta's config: TP=8, PP=4, H100-80GB):**

| seq_len | mbs | Bottleneck | Stage 0 → Stages 1-3 | Gain vs Full AC |
|---------|-----|------------|----------------------|-----------------|
| 4096 | 4 | Stage 0 | FA-Selective → No AC | **+22.4%** |
| **8192** | **2** | **Stage 0** | **FA-Selective → No AC** | **+23.2%** |
| 16384 | 1 | Stage 0 | FA-Selective → No AC | **+24.4%** |

**seq=8192, mbs=2 is very close to Meta's actual pretraining setup** (pretraining at
seq=8192, global batch ~16M tokens). This is the headline result: a +23% throughput gain
on the real Llama-3 70B training configuration.

The same gains hold on A100-80GB (fine-tuning / smaller cluster scenario).

**Llama-3.1 405B (PP=8 variant, TP=8, H100-80GB):**

| seq_len | mbs | Bottleneck | Stages | Gain |
|---------|-----|------------|--------|------|
| 4096 | 1 | Stage 0 | FA-Selective → No AC (×7) | **+21.9%** |

At Meta's actual PP=16, per-stage activations are too small (only ~8 layers/stage) and
stage 0 is forced into Full AC due to 15 stashed microbatches. At PP=8 (~16 layers/stage),
the sweet spot appears.

**GPT-NeoX-20B (EleutherAI config: TP=2, PP=4, A100-80GB):**

| seq_len | mbs | Bottleneck | Strategy | Gain |
|---------|-----|------------|----------|------|
| 2048 | 4 | Stage 0 | Korthikanti Sel → No AC | **+20.0%** |
| 4096 | 2 | Stage 0 | Korthikanti Sel → No AC | **+20.5%** |

For pre-FA models, the pipeline-aware framework uses Korthikanti selective (which IS
effective without FA) instead of FA-Selective. This validates the framework's generality.
Note: tested on A100-80GB (upgraded from the original A100-40GB, where memory is too tight
even for selective AC on stage 0).

**BLOOM-176B (BigScience config: TP=4, PP=12, A100-80GB):**

| seq_len | mbs | Bottleneck | Strategy gradient across 12 stages | Gain |
|---------|-----|------------|-----------------------------------|------|
| 2048 | 2 | Stage 0 | Korthikanti (0-2) → FA-Sel (3) → No AC (4-11) | **+19.7%** |

**mbs=2 was BLOOM's actual training micro-batch size.** The 12-stage pipeline shows a
beautiful 3-strategy gradient: aggressive checkpointing on the first 3 stages (high stash),
FA-Selective on stage 3, and No AC on stages 4-11 (low/zero stash).

### Why the sweet spot appears at realistic configs

The sweet-spot condition is: `activation_per_layer × layers_per_stage × (1 + stash_count)`
exceeds the memory budget with No AC, but selective AC's ~12-14% savings brings it under.

For 70B models with PP=4:
- 20 layers/stage (80 layers ÷ 4)
- Stage 0 stashes 3 microbatches (PP-1 = 3)
- At seq=8192, mbs=2: each microbatch's per-stage activations are ~15 GB
- 3 stashed × 15 GB = 45 GB stash alone → No AC at ~66 GB total → over 72 GB budget
- FA-Selective saves ~12% → ~58 GB → fits with headroom

This is exactly the regime practitioners operate in: moderate PP for multi-node training,
sequence lengths of 4K-16K, and micro-batch sizes of 1-4.

### Key insight: the sweet spot is NOT about small models with high PP

The original Llama-7B PP=8 result was a proof of concept in an artificial config. The
realistic sweet spot is **large models (70B+) with moderate PP (4-8) at standard training
seq_lens (4K-16K)**. These are configs practitioners actually use.

### GPU validation: the scale problem

The realistic configs where the sweet spot appears (Llama-3 70B TP=8 PP=4, BLOOM-176B
TP=4 PP=12) require 32-384 GPUs. On 8 H100s (single node), the only models that hit the
sweet spot are small models with artificially high PP:

| 8-GPU config | Sweet spot? | Realistic? |
|---|---|---|
| Qwen3-8B TP=2 PP=4 | Yes (+23-27%) | **No** — 8B model doesn't need PP=4 |
| GPT-NeoX-20B TP=2 PP=4 | Yes (+20%) | **Marginal** — 20B can use PP but TP=2 PP=4 is unusual |
| Llama-7B TP=1 PP=8 | No (No AC fits) | No |
| Llama-13B TP=2 PP=4 | No (No AC fits or Full AC bottleneck) | No |
| Llama-3 70B TP=4 PP=2 | No (OOM everywhere) | N/A — doesn't fit |

**The 8-GPU validation strategy** is to use Qwen3-8B TP=2 PP=4 as a **simulator accuracy
test**, not a production-worthy demo:
1. Validate that the simulator's pipeline memory predictions are correct (stash model, per-stage peaks)
2. Validate that the throughput gain from pipeline-aware AC matches the prediction (~23%)
3. Use simulator accuracy on 8 GPUs to argue credibility of predictions at 32+ GPUs

This is a standard systems paper approach (validate model on available hardware, extrapolate
to larger scale), but reviewers may push back on lacking an end-to-end result at realistic
scale. **32 H100s (4 nodes × 8 GPUs) running Llama-3 70B TP=8 PP=4 would be the ideal
validation** — it uses Meta's actual published training config and shows the sweet spot
at seq=8192, mbs=2.

### What Still Needs Validation (next steps)

1. **Simulator memory validation (DONE).** Validated on H100 with PyTorch 2.10 and
   Transformers 5.5.4.  Two models tested:
   - Llama-7B (MHA, SwiGLU, RoPE): **-1.5%** error (No AC), **-1.4%** (FA-Selective)
   - Qwen3-8B (GQA, SwiGLU, RoPE, QK-norm): **-0.4%** error (No AC), **-0.5%** (FA-Selective)
   Per-layer tensor counts confirmed via `saved_tensors_hooks`.

2. **FA-Selective compute overhead (DONE — single GPU).** Measured on H100:
   - Llama-7B: +0.9% overhead for 11.9% memory savings
   - Qwen3-8B: +1.1% overhead for 11.4% memory savings
   Confirms the "free lunch" claim.  The small overhead is from checkpoint bookkeeping,
   not from the recomputed FLOPs.

3. **Pipeline-aware throughput — 8 GPU validation (TODO).**
   Qwen3-8B, TP=2, PP=4, DP=1 on 8 H100s. Three runs:
   - (a) Uniform Full AC — baseline
   - (b) Uniform No AC — should OOM on stage 0 (confirms memory pressure)
   - (c) Pipeline-aware: FA-Selective on stage 0, No AC on stages 1-3
   Test at seq=4096/mbs=4 or seq=8192/mbs=2. Simulator predicts (c) is +23-25% faster
   than (a). Requires modifying Megatron-LM's `--recompute-granularity` to accept
   per-stage configs. The FA-Selective patch is the same `_silu_mul` checkpoint from
   `validate_on_gpu.py`, applied only to stage 0's layers.
   **This validates the simulator but uses an artificial config.**

4. **Pipeline-aware throughput — 32 GPU validation (TODO, ideal).**
   Llama-3 70B, TP=8, PP=4, DP=1 on 32 H100s (4 nodes). Same three-way comparison.
   Simulator predicts +23.2% at seq=8192, mbs=2. This is Meta's published training
   config and would be the strongest possible evidence.
   **This validates both the simulator AND the practical relevance.**

5. **Multi-schedule validation (TODO — stretch goal).** Test with ZB-H1/H2 (available
   in the zero-bubble codebase) to confirm the schedule-interaction predictions.

### Current state of evidence

| Claim | Status | Evidence |
|-------|--------|----------|
| Korthikanti selective = no-op with FA | Analytical ✓ | Simulator, 64 tests |
| FA-Selective saves 11-12% at <2% overhead | **Validated ✓** | H100: Llama-7B 11.9% savings / +0.9% overhead, Qwen3-8B 11.4% / +1.1% |
| Memory formulas match GPU within 2% | **Validated ✓** | H100: Llama-7B -1.5%, Qwen3-8B -0.4% (No AC and FA-Selective) |
| Simulator generalizes across architectures | **Validated ✓** | MHA (Llama) and GQA+QK-norm (Qwen3) both <2% error |
| Full AC overhead higher than modeled | **Observed** | Llama-7B: +28.6% (sim: +24%), Qwen3-8B: +48.5% (sim: +24%) |
| GQA amplifies MLP dominance | **Observed** | Qwen3-8B: MLP = 46% of activation memory vs 37% for attention |
| Pipeline-aware +23% on Llama-3 70B | Analytical ✓ | Simulator: TP=8, PP=4, seq=8192, mbs=2, H100. Needs multi-GPU validation |
| Pipeline-aware +20% on BLOOM-176B | Analytical ✓ | Simulator: TP=4, PP=12, seq=2048, mbs=2, A100. Needs multi-GPU validation |
| Pipeline-aware +20% on GPT-NeoX-20B | Analytical ✓ | Simulator: TP=2, PP=4, A100-80GB. Korthikanti selective on bottleneck |
| Sweet spot exists at realistic configs | Analytical ✓ | Llama-3 70B, BLOOM-176B, GPT-NeoX-20B — all at published training configs |
| 1F1B has 19-44% bubble overhead vs ZB | Analytical ✓ | Post-fix comparison |
| 3-Resource costs 1.6-3.2% overhead | Analytical ✓ | Needs GPU validation |
| DualPipe nullifies pipeline-aware AC | Analytical ✓ | Structural argument |
| ZB-H2 amplifies pipeline-aware AC | Analytical ✓ | Needs GPU validation |
| PCIe offload memory cost is exact | **Validated ✓** | H200: Llama-7B mlp_linear2_input offload matches simulator to 0.0% at seq=2048 and seq=4096 |
| PCIe offload overlap is stream-dependent | **Validated ✓ (v5)** | Dedicated CUDA stream (sync_mode="overlap"): +1.5%/+0.4% at seq=2048/4096, matches simulator to 0%. Default stream (sync_mode="serial"): +28.6%/+24.9% measured, simulator predicts +42ms/+84ms (within 5-15%). Selectable via `offload_sync_mode` parameter on `simulate()` / `simulate_pipeline_*()` and `--offload-sync-mode` on the runner. |
| Selective offload beats Full AC on modern H200 NVLink | **Validated ✓** | 4× H200, Llama-7B, PP=4, seq=32K: offload-all-mlp +12.5% throughput vs Full AC at μb=4, +10.4% at μb=8 |
| Pipeline offload has real PCIe bus contention | **Observed** | Per-tensor independent stall model is 16% optimistic at PP=4 seq=32K. `schedule_offloads` (half-duplex) is the right model to wire in |
| `schedule_offloads` wired into `simulate()` | **Implemented v4** | Multi-tensor offload now accounts for shared PCIe bus. Overcorrects to 44% pessimistic at PP=4 seq=32K because `pcie_busy_until = recv_end` overcounts the ALAP wait interval — needs interval-based bus model as followup |
| Interval-based bus scheduler replaces `pcie_busy_until` | **Implemented v4a** | Sends + recvs now pack into a sorted list of busy intervals. At PP=4 seq=32K, bus work (88ms) fits comfortably in the per-layer gap (~180ms) → predicted stall = 0, step returns to 11,980ms. The 14% optimism vs measured (13,927ms) is **not** explained by bus contention at this config; something else (stream dispatch, DMA setup, first-mb effects) is responsible |
| Pipeline-aware sweet spot at 8-GPU PP=8 scale | **Validated ✓** | 8× H200, Llama-7B, PP=8, seq=32768, mbs=1, μb=8, 1F1B. Uniform No AC OOMs on ranks 0-2 (sim: 196/171/149 GB > 141 GB). Uniform Full AC fits at 24.7 GB, 25,090 tok/s. Pipeline-aware (offload-all-mlp × 3 + no-ac × 5, dispatched via `--per-stage`) fits at 125.8 GB peak, **32,076 tok/s = +27.8% vs Full AC**. Simulator predicted +29.0% — relative error within 1.2 pp. Does not substitute for literal Llama-3 70B TP=8 PP=4 on 32 GPUs, but confirms the mechanism on the largest config reachable on a single H200 node. |

---

## Finding: Selective CPU offload beats Full AC on modern H200 NVLink clusters at long context

### What we measured (2026-04-23, v3)

**Config:** Llama-2-7B, PP=4 on 4× H200 NVLink (NV18), seq_len=32768, mbs=1,
1F1B schedule, SDPA attention, bf16. Each rank owns 8 decoder layers. Results
at two microbatch counts:

**μb=4** (bubble fraction 75%, warmup-heavy):

| Strategy | Fits? | Step (ms) | Throughput | Peak HBM (rank 0) | Offload/step |
|----------|-------|-----------|------------|-------------------|--------------|
| No AC | **OOM** | — | — | — | — |
| Offload linear2 | **OOM** | — | — | — | — |
| Full AC | ✓ | 9,743 | 13,452 tok/s | 24.7 GB | 0 GB |
| **Offload all MLP** | **✓** | **8,658** | **15,139 tok/s** | 101.1 GB | 85 GB |

**μb=8** (bubble fraction 37.5%, steady-state):

| Strategy | Fits? | Step (ms) | Throughput | Peak HBM (rank 0) | Offload/step |
|----------|-------|-----------|------------|-------------------|--------------|
| Full AC | ✓ | 15,384 | 17,040 tok/s | 26.7 GB | 0 GB |
| **Offload all MLP** | **✓** | **13,927** | **18,821 tok/s** | 103.1 GB | 306 GB |

Offload beats Full AC by **+12.5% throughput at μb=4** and **+10.4% at μb=8**.
The headline claim the repo is named for — selective CPU offloading is useful
on modern clusters — is validated on an 8× H200 node at a realistic long-context
training configuration.

### Why this config matters

Three configs were necessary for the experiment to be meaningful. None of them
are consumer-GPU regimes:

1. **Long context (seq=32768)**: at shorter sequences, activation memory fits
   trivially and offloading never pays. Long-context training is where modern
   H200 clusters actually spend their time.
2. **Pipeline parallelism (PP=4)**: the 1F1B schedule stashes (PP−1−p)=3
   microbatches on stage 0. Activation pressure is amplified exactly on the
   stage that would otherwise OOM under No AC.
3. **Full-GPU-scale mbs**: with mbs=1 seq=32K, No AC peak is 208 GB per stage
   — well over the 141 GB H200 HBM. There is no amount of TP/DP within 4
   GPUs that rescues this without either recomputation or offload.

### Simulator validation

| Strategy | Measured step | Simulator no-bubble | Simulator w/ bubble | Error (no bubble) |
|----------|---------------|---------------------|---------------------|-------------------|
| Full AC (μb=8) | 15,384 ms | 15,449 ms | 21,243 ms | **−0.4%** |
| Offload all MLP (μb=8) | 13,927 ms | 11,980 ms | 16,473 ms | +16.3% |

The simulator's compute model is essentially exact for Full AC at seq=32K
(matches measured to 0.4% on the no-bubble projection). For Offload all MLP,
measured is 16% slower than the no-bubble prediction — the simulator's
per-tensor independent stall model undercounts bus contention. Under 1F1B,
three microbatches' worth of MLP activations (≈ 66 GB per stage) queue
through one PCIe lane during the warmup phase, and some transfers fail to
overlap with compute despite the dedicated stream.

The right model for this regime is `simulator/offload_model.py::schedule_offloads`,
which does account for half-duplex bus serialization — but `simulate()` does
not invoke it. Wiring `schedule_offloads` into the pipeline-aware path is the
natural next improvement to the cost model.

### What's next

- Adapt `simulator/environment.py` to call `schedule_offloads` when a pipeline
  stage accumulates multiple OFFLOAD_CPU decisions across microbatches.
- Push seq_len higher (seq=65536, seq=131072) to see whether offload continues
  to beat Full AC as activation/compute ratio shifts.
- Combine offload with heterogeneous per-stage strategies (pipeline-aware
  offload): offload-all-mlp on stage 0 where pressure is worst, lighter
  strategies on later stages.

## Finding: Selective CPU offloading — memory model is exact; overlap depends on stream discipline

### What we measured (2026-04-23)

We validated `simulator/offload_model.py`'s peak-HBM and stall-time predictions
against real H200 runs. Setup: Llama-2-7B decoder, bfloat16, SDPA attention,
offloading the `mlp_linear2_input` tensor on every one of the 32 decoder layers
via `torch.autograd.graph.saved_tensors_hooks` applied only around `down_proj`.
Hooks filter out parameter-like tensors (leaves with `requires_grad` *and* views
whose `_base` is such a leaf — this is required to skip `W.T` saved by
`F.linear`, which otherwise triples the PCIe traffic).

| seq | HBM saved (measured / simulator) | Overhead (default stream) | Overhead (dedicated stream) |
|-----|----------------------------------|---------------------------|------------------------------|
| 2048 | 1.344 GB / 1.344 GB (+0.0%) | +49.7 ms (+28.6%) | +2.6 ms (+1.5%) |
| 4096 | 2.688 GB / 2.688 GB (+0.0%) | +88.6 ms (+24.9%) | +1.5 ms (+0.4%) |

Outputs are bit-identical between the baseline and each offload mode
(rel_err = 0 in bf16), confirming the pack/unpack round-trip is lossless.

### What the simulator gets right

The formula `saved = size(linear2_input) × num_layers` in `memory_model.py` and
`environment.py`'s OFFLOAD_CPU branch predicts the retained post-forward
activation delta to the byte — no surprises, and the prediction scales
correctly with seq_len.

### What the simulator does NOT model

`simulate()` assumes transfers overlap with compute as long as
`liveness_gap > round_trip`, and reports zero stall for this config. The
default-stream run shows ~28% step-time overhead — because the naive
`cpu.copy_(tensor, non_blocking=True)` DMA still serializes with subsequent
compute on the same CUDA stream. The "transfers are free" assumption requires
a dedicated CUDA stream with explicit stream-event synchronization, which is
what the `dedicated` mode in `offload/hooks.py::CPUOffloadHook` installs.

With the dedicated stream, overhead collapses to the noise floor (≤1.5%), and
the simulator's "full overlap" prediction holds. Without it, a naive offload
implementation costs 25-30% throughput — which the simulator currently does
not reflect. Either the cost model should gain a "sync mode" input, or
implementations using `saved_tensors_hooks` should document that a dedicated
stream is non-optional.

---

## Finding: Wiring `schedule_offloads` into `simulate()` overcorrects the +16% optimism

### Change (2026-04-23, v4)

`simulator/environment.py` now collects all OFFLOAD_CPU decisions per layer
into a pending list and calls `schedule_offloads(pairs, gpu, par)` once per
layer, instead of summing `compute_offload_result(tensor, gap)` per tensor
as if each transfer had its own PCIe lane. This is the wiring Andrew named as
"the natural next cost-model improvement."

Regression tests (`tests/test_offload_strategy_config.py::TestOffloadBusContention`)
pin the new behavior:
- `test_single_tensor_matches_independent_model` — with one OFFLOAD_CPU tensor
  per layer, `schedule_offloads`'s result is identical to the independent
  formula, so the validated seq=2048/4096 measurements do not move.
- `test_stall_monotonic_in_offload_count` — offloading more tensors per layer
  produces more stall under the bus model.
- `test_layer_stall_matches_direct_schedule_offloads` — the per-layer stall in
  `SimulatorResult.per_layer[i].offload_stall_s` equals a direct
  `schedule_offloads` call on the same tensors+gaps.

### Predicted-vs-measured sign flip

At the validated PP=4, seq=32K, μb=8 offload-all-mlp config:

| Model | Predicted step (ms) | Measured (ms) | Error |
|---|---|---|---|
| Independent per-tensor stall (before) | 11,980 | 13,927 | **−14% (optimistic)** |
| `schedule_offloads` wired in (after) | **20,043** | 13,927 | **+44% (pessimistic)** |

The wiring is directionally correct (it now accounts for multiple tensors
competing for one PCIe lane), but the swing to pessimism is larger than the
original optimism. The +16% gap the wiring was meant to close is now a
−28% gap in the other direction.

### Root cause: `pcie_busy_until = recv_end` overcounts the bus

Reading `simulator/offload_model.py::schedule_offloads` lines 182–197:

```python
send_start = max(0.0, pcie_busy_until)
send_end = send_start + send
recv_start = max(send_end, gap - recv)   # ALAP: schedule recv close to deadline
recv_end = recv_start + recv
pcie_busy_until = recv_end                # ← overcount
```

The recv is scheduled ALAP (as late as possible so it finishes at the
deadline). That means the interval `[send_end, recv_start]` is *bus idle* —
the send has finished and the recv is waiting for its deadline. Other
tensors' sends *could* fire in that window, but `pcie_busy_until = recv_end`
claims the bus is occupied all the way through. Subsequent tensors are
forced to queue behind a gap that isn't really there.

At the seq=32K offload-all-mlp config, each of the 4 offloaded MLP tensors
per layer has transfer time T ≈ 11 ms (721 MB at 64 GB/s PCIe Gen5). Under
the current scheduler:

| Tensor | send | recv (ALAP) | pcie_busy_until after |
|---|---|---|---|
| 1 | [0, 11] | [gap−11, gap] | gap |
| 2 | [gap, gap+11] | [gap+11, gap+22] | gap+22 |
| 3 | [gap+22, gap+33] | [gap+33, gap+44] | gap+44 |
| 4 | [gap+44, gap+55] | [gap+55, gap+66] | gap+66 |

Stalls cascade linearly: 0, 22, 44, 66 ms per tensor. But physically, bus
time for all 4 tensors is 88 ms (4 × 22 ms round trip); under a proper
interval-based scheduler, excess stall is `max(0, 88 − overlap_window)`,
not `0 + 22 + 44 + 66 = 132 ms`.

### Followup: interval-based bus model

The fix is to replace the scalar `pcie_busy_until` with a list of busy
intervals and place each send in the earliest free slot (and each recv in
the latest free slot ≤ deadline, after its send). That should land
predicted step latency between the old optimistic and new pessimistic
numbers — ideally within ±5% of measured.

Until that lands, the simulator's offload stall is a known overestimate at
configs with ≥2 OFFLOAD_CPU tensors per layer. Single-tensor offload
configs (the validated seq=2048/4096 path) are unaffected.

| Claim | Status | Evidence |
|---|---|---|
| Wiring accounts for multi-tensor bus contention | **Implemented** | `environment.py` per-layer schedule_offloads call + 4 new tests |
| Wiring reduces 16% optimism at PP=4 seq=32K | **No** — overcorrects to 44% pessimism | Delta driven by `pcie_busy_until = recv_end` overcount |
| schedule_offloads needs interval-based scheduling | **Proposed** | See task #1a |

---

## Finding: Interval-based bus scheduler replaces `pcie_busy_until`; 14% PP=4 seq=32K gap is not bus contention

### Change (2026-04-23, v4a)

`simulator/offload_model.py::schedule_offloads` now maintains a sorted list
of non-overlapping busy intervals instead of a single `pcie_busy_until`
cursor. Each tensor's send goes into the earliest free slot starting at
t ≥ 0 (drain the queue fast, free memory early); each recv goes into the
latest free slot whose end ≤ deadline (ALAP — don't block earlier-deadline
recvs). When no deadline-meeting slot exists, the recv is pushed past the
deadline and the excess is stall.

Two helpers, `_earliest_free_slot` and `_latest_free_slot_by_deadline`,
walk the busy list in forward / reverse order respectively. Insertion is
binary-searched into the sorted list. All sends and recvs from all tensors
end up as non-overlapping intervals, so subsequent tensors correctly see
the real bus availability (including the window between an earlier
tensor's send and its ALAP recv).

### Predicted-vs-measured at PP=4 seq=32K μb=8

| Model | Predicted step (ms) | Measured (ms) | Error |
|---|---|---|---|
| Independent per-tensor stall (pre-v4) | 11,980 | 13,927 | −14% |
| `schedule_offloads` with `pcie_busy_until` (v4) | 20,043 | 13,927 | +44% |
| **Interval-based (v4a, current)** | **11,980** | 13,927 | **−14%** |

The interval-based scheduler predicts **zero stall** at this config, because
per-layer bus work (4 × 22 ms = 88 ms for the four MLP tensors) fits
comfortably inside the per-layer compute window (fwd + bwd ≈ 180 ms). That
prediction is physically correct — the bus genuinely has headroom here —
and the 14% gap vs. measured is therefore **not** explained by multi-tensor
bus contention. Likely remaining sources:

- `saved_tensors_hooks` dispatch overhead (per-tensor Python hook cost ×
  32 layers × 8 microbatches).
- CUDA DMA launch latency per copy that doesn't amortize into bandwidth.
- Per-microbatch first-touch effects (allocator warm-up, pinned-buffer
  reuse) the analytical model treats as zero.
- The simulator's MFU constant (0.5) is slightly off for this specific
  kernel mix at long context.

### Where the interval-based scheduler actually matters

The fix is correct and necessary for regimes where bus work approaches or
exceeds the liveness gap. A sweep of `llama_7b` offload-all-mlp over
seq_len:

| seq_len | Offload tensors | Predicted stall / mb (ms) |
|---|---|---|
| 512 | 4 | 5.25 |
| 1024 | 4 | 10.50 |
| 2048 | 4 | 0.00 |
| 4096 | 4 | 0.00 |
| 8192+ | 4 | 0.00 |

At seq=512 and seq=1024 the compute window shrinks enough that bus time
matters; the interval scheduler correctly reports that. Above seq=2048 the
window dominates and stall is zero regardless.

### What v4a did and didn't do

- ✓ The bus model is now physically correct: single-tensor path unchanged
  (verified by regression test), multi-tensor path accounts for the
  half-duplex shared bus without the ALAP-wait overcount.
- ✓ Previously failing tests that asserted the old broken behavior have
  been replaced with tests that pin the correct behavior (both the
  zero-stall-when-window-fits case and the nonzero-stall-when-overcommitted
  case).
- ✗ Did **not** close the 14% PP=4 seq=32K gap vs measured. Bus contention
  was not the right diagnosis at that config.

### What's next

The 14% gap warrants a separate investigation. Candidates:

1. **Measure the per-offload hook overhead** on GPU: wrap `CPUOffloadHook._pack`
   with a profiler and see whether Python dispatch × 32 × 8 accounts for
   ~200 ms per microbatch.
2. **Add a per-transfer fixed overhead** to the PCIe cost model. Currently
   `transfer_time = size / eff_bw`; adding a `+ launch_latency` term (e.g.
   10-50 µs per copy) would model DMA setup cost.
3. **Sweep the MFU constant** in `get_layer_compute_profile` — if 0.45
   better matches long-context H200 measurements, that's a two-line fix.

None of these are within the scope of the "wire `schedule_offloads`" task.
For the simulator's purposes, the bus model is now right for the physics
it was intended to capture.

---

## Finding: `offload_sync_mode` flag models the default-stream penalty

### Change (2026-04-23, v5)

`simulator/offload_model.py` gained a `SyncMode = Literal["overlap", "serial"]`
parameter, threaded through `compute_offload_result`, `schedule_offloads`,
`simulate()`, `simulate_pipeline_aware_ac()`, and `simulate_pipeline_uniform_ac()`.
Exposed as `--offload-sync-mode` on the throughput runner.

- `sync_mode="overlap"` (default): current behavior. DMAs run on a dedicated
  CUDA stream; stall is computed by the interval-based bus scheduler. Matches
  what `offload/hooks.py::CPUOffloadHook` does when given an `offload_stream`.
- `sync_mode="serial"`: each transfer incurs stall = round_trip regardless of
  liveness gap, because on the default stream DMAs block the next compute op.
  The interval-based scheduler is skipped entirely.

### Validation against the single-GPU offload measurements

| seq | Predicted stall (serial) | Measured default-stream overhead | Error |
|---|---|---|---|
| 2048 | 41.99 ms | 49.7 ms | −15.5% |
| 4096 | 83.98 ms | 88.6 ms | −5.2% |

Both overlap and serial modes now have simulator predictions that agree with
measurement to within ~5-15%. The remaining gap at seq=2048 is likely the
same Python-hook dispatch overhead that accounts for the 14% discrepancy at
PP=4 seq=32K.

### Why this matters

Without `sync_mode="serial"`, any naive user who calls `saved_tensors_hooks`
without a dedicated stream would run into a 25-30% throughput cliff that the
simulator claimed wouldn't happen. Now the simulator tells both stories, and
the runner's `--offload-sync-mode` CLI makes it explicit which prediction to
compare measurement against.

This does not change the headline "offload beats Full AC" claim, which was
always measured with a dedicated stream. It does mean future users can
predict the cost of getting the stream discipline wrong before they burn a
GPU-hour on it.

---

## Finding: Pipeline-aware sweet spot validated end-to-end at PP=8 on 8× H200

### What we measured (2026-04-23, v6)

**Config:** Llama-2-7B, single-node 8× H200 NVLink, PP=8 (TP=1, DP=1),
seq_len=32768, mbs=1, μb=8, 1F1B, SDPA attention, bf16. Each rank owns 4
decoder layers; stage 0 additionally holds the embedding, stage 7 the final
RMSNorm. 2 warmup + 3 timed steps, CUDA event timing. Optimizer step
intentionally omitted so measured step latency compares apples-to-apples
against the simulator's fwd+bwd+recompute prediction.

This is the natural 8-GPU proxy for OBSERVATIONS.md #4 (Llama-3 70B
TP=8 PP=4 on 32 H100s). The literal config is not reachable on this
hardware — the runner is PP-only and we only have one node — but the
mechanism (stash pressure concentrated on early stages, heterogeneous
strategies across the pipeline) is the same.

### Three-way comparison

| Strategy | Fits? | Bottleneck step | Throughput | Peak HBM (max stage) | Offload/step |
|----------|-------|-----------------|------------|----------------------|--------------|
| **No AC uniform** | **OOM** | — | — | ranks 0,1,2 OOM | — |
| Full AC uniform | ✓ | 10,448 ms | 25,090 tok/s | 24.7 GB (stage 0) | 0 GB |
| **Pipeline-aware** | ✓ | **8,173 ms** | **32,076 tok/s** | 125.8 GB (stage 3) | 170 GB (stages 0-2) |

**Pipeline-aware beats Full AC by +27.8%** at this config. The No AC OOM
happens exactly where the simulator says it will: ranks 0, 1, 2 (predicted
peaks 196 / 171 / 149 GB vs 141 GB H200 HBM). The per-stage strategies
dispatched — `offload-all-mlp` × 3 on the high-stash early stages,
`no-ac` × 5 on the low-stash late stages — are the same ones
`simulate_pipeline_aware_ac` independently chose.

### Simulator accuracy

| Claim | Predicted | Measured | Error |
|---|---|---|---|
| Pipeline-aware speedup over Full AC | +29.0% | **+27.8%** | **−1.2 pp** |
| Full AC per-μb at bottleneck | 966 ms | 1,306 ms | +35.3% |
| Pipeline-aware per-μb at bottleneck | 749 ms | 1,021 ms | +36.4% |
| Full AC peak HBM stage 0 | 22.0 GB | 24.7 GB | +12.3% |

The **relative** prediction — which is the one the simulator is actually
selected on — lands within 1.2 percentage points of measured. On absolute
per-microbatch latency the simulator is uniformly ~35% optimistic (same
magnitude for both Full AC and the offload path), consistent with the PP=4
seq=32K gap noted in the v4a finding above: the root cause is *not* bus
contention, since both strategies undercount by the same amount and only
one of them transfers anything. Best guesses: Python-side hook dispatch
overhead, first-microbatch NCCL warmup effects, or MFU<0.5 at seq=32K.

The bubble-inflated prediction (bubble fraction = (PP−1)/M = 87.5% at
PP=M=8) overshoots in the other direction: Full AC predicted-with-bubble
14,484 ms vs measured 10,448 ms = −27.9%. Measured sits symmetrically
between the no-bubble and bubble-adjusted predictions. This suggests
`torch.distributed.pipelining.Schedule1F1B` achieves meaningfully better
overlap than the simulator's bubble model assumes at this M=PP corner.

### What this validates

- **Simulator's pipeline-aware recommendation is measurable** — not just
  an analytical artifact. The same per-stage strategy picker that produced
  "+23% on Llama-3 70B TP=8 PP=4" (listed in the evidence table as
  Analytical ✓, needs multi-GPU validation) is now validated on an 8-GPU
  proxy using exactly the same heuristic.
- **No AC OOMs precisely where predicted** — ranks 0, 1, 2 at PP=8
  seq=32K. The per-stage stash model (stage p stashes PP−1−p microbatches
  under 1F1B) holds at the boundary.
- **Offload scales with pipeline depth**. Prior PP=4 seq=32K result was
  +10.4% (offload-all-mlp uniform vs Full AC uniform). At PP=8 the sweet
  spot grows to +27.8% because offload is selective — applied only where
  it pays, not uniformly.

### Runner changes required

The existing `--ac pipeline-aware` mode maps to `full-ac × half + no-ac × half`
(see `throughput/strategies.py`), which is **not** what the simulator's
`simulate_pipeline_aware_ac` picks at this config. Running that mode
instead would leave the +27.8% on the table. A `--per-stage` CLI option
was added so the runner can dispatch any explicit strategy list and
compare to the simulator's heterogeneous recommendation directly:

```bash
./throughput/launch.sh 8 pipeline-aware --seq 32768 --mbs 1 --microbatches 8 \
  --per-stage offload-all-mlp,offload-all-mlp,offload-all-mlp,no-ac,no-ac,no-ac,no-ac,no-ac
```

### Caveats

- **Not the literal OBSERVATIONS.md #4 config.** That one calls for
  Llama-3-70B TP=8 PP=4 on 32 H100s. It requires 4 nodes and TP support
  in the runner (currently PP-only). The PP=8 Llama-2-7B config here is
  the biggest config that shows a genuine pipeline-aware sweet spot
  on 8× H200 with the existing runner.
- **Optimizer step omitted.** Measured step time is fwd + bwd + recompute
  only, per the repo's standing policy. Real training throughput would
  include an optimizer step that's identical across strategies, so the
  relative speedup is the meaningful comparison either way.
- **Three timed steps per run.** Run-to-run noise was not characterized;
  the 1.2 pp gap between predicted and measured speedup is within the
  plausible noise floor at this step count.
