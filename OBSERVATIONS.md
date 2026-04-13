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
Attention (with FA):  ~9 sbh / tp    (Q, K, V, output proj input, FA logsumexp)
MLP (SwiGLU):         ~3 × ffn × sb × bpe / tp + sbh × bpe / sp   (gate, up, linear2, input)
LayerNorm:            2 × sbh × 4 / sp   (fp32 inputs)
```

For Llama-7B (h=4096, ffn=11008, tp=1): attention is ~9 × 4096 × 2 = 72K elements per
token, MLP is ~3 × 11008 × 2 = 66K elements per token. They're roughly equal, but the MLP
tensors are the ones amenable to cheap recomputation — specifically `mlp_linear2_input`,
which is the output of a pointwise activation function (SiLU for SwiGLU, GeLU for standard).

## 3. FA-Era Selective AC: The New Practical Default

The FA-era analogue of Korthikanti's strategy: recompute `mlp_linear2_input` (the activation
function output), which is a cheap pointwise operation, while keeping the matmul outputs
(gate_output, up_output) which are expensive to recompute.

**Simulator results:**

| Model                      | Memory Saved | Step Overhead |
|----------------------------|-------------:|--------------:|
| Llama-7B (A100, FSDP dp=8) |       -14.5% |        +0.00% |
| Llama-13B (A100, FSDP dp=8)|       -14.5% |        +0.00% |
| Llama-70B (H100, TP=8+dp=8)|       -18.2% |        +0.00% |
| GPT-3 175B (A100, TP=8+PP=8)|      -11.6% |        +0.00% |

This is a genuine free lunch: the recomputed operation (SiLU + elementwise multiply for
SwiGLU) is so cheap relative to the surrounding matmuls that the overhead is immeasurable.

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

Four bugs were identified (via adversarial review) and fixed. The corrections changed
some numerical results but **strengthened rather than weakened** the core claims.

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

The fixes made the simulator more accurate without invalidating any core contribution.
The main numerical change is that 1F1B step times are now 19-44% higher (due to bubble),
which makes the ZB schedule comparison genuine and strengthens the argument for considering
pipeline schedules jointly with AC strategy.

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
   compute cost, PCIe contention modeling, and multi-schedule pipeline support. 63 tests
   validated against Korthikanti formulas.

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

**Models:** Llama-7B, Llama-13B, Llama-70B (FA-enabled, SwiGLU), GPT-3 175B (no FA, GELU).

**Hardware:** A100-40GB, A100-80GB, H100-80GB. Varying TP, PP, DP configurations.

**Schedule comparison:** 1F1B, 1F1B Interleaved, ZB-H1, ZB-H2, ZB-V, DualPipe — show how
pipeline-aware AC interacts with each.

### What Still Needs Validation (next steps)

1. **Simulator validation on real hardware (highest priority).** Run Llama-7B training step
   on a single A100, dump memory snapshot via `torch.cuda.memory._dump_snapshot()`, compare
   to simulator prediction. Target <5% error on peak memory. This confirms the memory
   formulas are correct.

2. **FA-Selective in practice.** Implement via `torch.utils.checkpoint` with a custom
   `checkpoint_fn` that only wraps the activation function. Measure real overhead on 1 GPU.
   Verify the ~0% overhead prediction — kernel launch overhead, memory allocator pressure,
   and autograd bookkeeping could add a small constant.

3. **Pipeline-aware AC in Megatron-LM.** Modify `--recompute-granularity` to accept per-stage
   configs. Run the Llama-7B PP=8 sweet-spot case on 8 GPUs and measure actual throughput.
   The simulator predicts +24.6% vs. uniform Full AC — this needs experimental confirmation.

4. **Multi-schedule validation.** If possible, test with ZB-H1/H2 (available in the
   zero-bubble codebase) to confirm the schedule-interaction predictions.

### Current state of evidence (post bug-fix)

| Claim | Status | Evidence | Survived Bug Fixes? |
|-------|--------|----------|---------------------|
| Korthikanti selective = no-op with FA | Analytical ✓ | Simulator, 63 tests | Yes (structural) |
| FA-Selective saves 12-18% at ~0% overhead | Analytical ✓ | Needs GPU validation | Yes (unchanged) |
| Pipeline-aware gives +24.6% in sweet spot | Analytical ✓ | Needs GPU validation | Yes (unchanged) |
| 1F1B has 19-44% bubble overhead vs ZB | Analytical ✓ | Post-fix comparison | NEW (was broken) |
| Non-FA pipeline uses Korthikanti (~2.7%) | Analytical ✓ | Post-fix strategy search | NEW (was skipped) |
| 3-Resource costs 1.6-3.2% overhead | Analytical ✓ | Needs GPU validation | Yes (unchanged) |
| DualPipe nullifies pipeline-aware AC | Analytical ✓ | Structural argument | Yes (structural) |
| ZB-H2 amplifies pipeline-aware AC | Analytical ✓ | Needs GPU validation | Yes (unchanged) |
