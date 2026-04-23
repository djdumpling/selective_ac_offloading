# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Analytical simulator for activation checkpointing (AC), CPU offloading, and low-rank compression strategies on transformer training workloads. The simulator predicts peak HBM, recompute overhead, PCIe stalls, and step latency for a given (model, GPU, parallelism, per-tensor strategy) tuple, and is validated against real GPU measurements.

Not a training system — there is no model training code here. The repo produces analytical predictions and cross-checks them against `LlamaModel` / `Qwen3Model` forward+backward on a single GPU.

## Commands

Simulator + tests have no GPU dependency; the validate scripts do.

The repo uses a uv-managed venv at `.venv/` (Python 3.11, torch 2.11 + cu128, transformers 5.6). Create with `uv venv --python 3.11 && uv pip install torch --torch-backend=cu128 transformers pytest`. Do not use system Python on the shared cluster.

```bash
.venv/bin/python -m pytest                           # run everything
.venv/bin/python -m pytest tests/test_simulator.py::TestPipelineAwareAC   # one class
.venv/bin/python -m pytest tests/test_simulator.py -k stash               # by keyword
.venv/bin/python -m pytest -x -v                     # stop on first failure, verbose

.venv/bin/python demo.py                        # strategy comparison across Llama/GPT-3 configs
.venv/bin/python demo_pipeline_schedules.py     # 1F1B / ZB-H1/H2 / DualPipe comparison
.venv/bin/python demo_sweet_spot.py             # configs where pipeline-aware AC beats uniform
.venv/bin/python demo_realistic_sweet_spot.py   # same, with real published training configs
.venv/bin/python demo_8gpu_search.py            # sweep single-node H100 parallelism combos

# Single-GPU validation (activation memory comparison):
.venv/bin/python validate_on_gpu.py             # Llama-2-7B, simulator vs real activation memory
.venv/bin/python validate_qwen3_8b.py           # Qwen3-8B (GQA + QK-norm paths)
.venv/bin/python snapshot_activations.py        # saved_tensors_hooks dump for Llama
.venv/bin/python snapshot_qwen3.py              # same for Qwen3
.venv/bin/python analyze_snapshot.py memory_snapshot_no_ac.pickle  # parse a CUDA memory snapshot

# Multi-GPU throughput validation (torch.distributed.pipelining, 1F1B):
./throughput/launch.sh 2 full-ac        --seq 2048 --mbs 1 --microbatches 4
./throughput/launch.sh 4 no-ac          --seq 4096 --mbs 1 --microbatches 8
./throughput/launch.sh 4 pipeline-aware --seq 4096 --mbs 1 --microbatches 8
# First arg = pp_size (= nproc_per_node); second = {no-ac, full-ac, pipeline-aware}.
# Rank 0 prints measured vs. simulator-predicted step latency.
```

`conftest.py` puts the repo root on `sys.path`, so tests and scripts both `from simulator import ...` without install.

## Architecture

Everything flows through `simulator.environment.simulate(cfg, gpu, strategies, par)`. The inputs are:

- `ModelConfig` — architecture + workload shape (`simulator/config.py`, with prebuilt `llama_7b`, `llama3_70b`, `qwen3_8b`, `gpt3_175b`, etc.). Feature flags (`use_flash_attention`, `use_rotary_embeddings`, `use_qk_norm`, `num_kv_heads`, `activation_fn`) gate which tensors the memory model emits.
- `GPUConfig` — HBM capacity/bandwidth, peak TFLOPS, PCIe bandwidth. Prebuilt `A100_80GB`, `H100_80GB`, etc.
- `ParallelismConfig` — `tp_size`, `pp_size`, `dp_size`, `sp_size`. Used to shard tensor sizes, parameter counts, and bandwidths.
- `strategies: list[LayerStrategy]` — per-layer dict of `TensorDecision` (KEEP / RECOMPUTE / OFFLOAD_CPU / COMPRESS, plus optional `compress_rank`, `stored_size_bytes`, `allow_nonrecomputable`). Missing layers / tensors default to KEEP.

### The per-tensor model (`simulator/memory_model.py`)

The model's fundamental unit is `TensorInfo` — one entry per activation tensor retained for backward, with `size_bytes`, `recompute_flops`, `recompute_from`, and a `recomputable` flag (False for opaque things like FA logsumexp and dropout masks). Tensors are emitted conditionally based on the `ModelConfig`:

- `use_flash_attention=True` ⇒ no quadratic `attn_softmax`/dropout tensors; instead a small `attn_fa_logsumexp` at fp32.
- `use_rotary_embeddings=True` ⇒ extra `attn_rotary_q`/`attn_rotary_k` post-rotation copies that SDPA saves separately from the pre-rotation Q/K.
- `is_gqa` (`num_kv_heads < n_heads`) ⇒ extra `attn_k_expanded`/`attn_v_expanded` saved by SDPA after `repeat_kv`.
- `use_qk_norm=True` ⇒ extra fp32 copies from per-head RMSNorm (Qwen3).
- `is_swiglu` ⇒ gate/up/silu tensors instead of the single gelu input/output pair.

If you add an architecture feature, it is almost always a new branch in `get_attention_tensors` / `get_mlp_tensors`, plus a matching assertion in `tests/test_simulator.py`.

### Strategy levels and pipeline-aware AC (`simulator/environment.py`)

Named strategies are declared in `STRATEGY_LEVELS`, ordered from least to most aggressive:

```
No AC → FA-Selective → Korthikanti Selective → Full AC
```

`simulate_pipeline_aware_ac` asks each pipeline stage independently for the **least aggressive** strategy that fits in HBM given its stash count. The interesting cases are when stages disagree — that's the "sweet spot" the demos search for. When they all agree, pipeline-aware reduces to uniform.

Stash counts and bubble fractions come from `simulator/pipeline_schedules.py`:

- 1F1B, 1F1B Interleaved, ZB-H1, ZB-H2, ZB-V, DualPipe
- `_stash_dualpipe` is symmetric (all stages stash `PP-1`) — hence the "pipeline-aware provides no benefit on DualPipe" invariant asserted in tests.
- ZB-H2 adds per-stage extra memory for deferred weight gradients; this flows into `_run_pipeline_simulation` as `extra_memory_per_stage`.

### Cost models

- `simulator/compute_model.py` — forward FLOPs per block, `flops_to_latency` using a configurable MFU (`efficiency`, default 0.5). Backward ≈ 2× forward.
- `simulator/offload_model.py` — PCIe transfer times with NCCL contention. `estimate_nccl_pcie_utilization` decides whether collectives touch PCIe (intra-node up to 8 GPUs ⇒ NVLink ⇒ no contention; multi-node DP ⇒ 40% of PCIe consumed). `schedule_offloads` is a half-duplex bus model that serializes transfers in decreasing-liveness-gap order (agreeable deadlines).
- `simulator/compression_model.py` — low-rank decomposition memory + **non-zero** FLOPs (`4·sb·d·r` round-trip). Tests assert compression increases step latency; don't regress this to free.

### GPU validation (`validate_on_gpu.py`, `validate_qwen3_8b.py`, `snapshot_activations.py`)

The validators build `LlamaModel`/`AutoModel` with random weights, measure retained forward activations via CUDA memory reset + peak tracking and via `torch.autograd.graph.saved_tensors_hooks`, then compare to `simulate_no_ac` / `simulate_fa_selective_ac` / `simulate_full_ac`. Target is <5% error on retained forward activations. If a new architecture feature changes memory predictions, re-run these on GPU before trusting the simulator. `LlamaModel` (not `LlamaForCausalLM`) is deliberate so measurements track transformer activations, not LM-head logits.

### Multi-GPU throughput validation (`throughput/run_pipeline.py`)

Pipeline-parallel runner on `torch.distributed.pipelining.Schedule1F1B`. Builds a hand-constructed `LlamaStage` per rank (embed on stage 0, `LlamaDecoderLayer` slice in the middle, final RMSNorm on stage N-1) — not `LlamaModel`, because HF's forward can't be trivially sharded. RoPE is recomputed per stage (weight-free). Activation checkpointing uses `apply_activation_checkpointing` with `checkpoint_wrapper` targeting `LlamaDecoderLayer`; per-stage policy comes from `throughput/strategies.py::stage_strategies` (kept torch-free so it's unit-testable). `pipeline-aware` puts Full AC on the first `pp_size // 2` ranks and No AC on the rest — mirrors `STRATEGY_LEVELS`'s least-aggressive-that-fits heuristic.

Optimizer step is intentionally omitted — measured step time is fwd + bwd + recompute only, directly comparable to the simulator's `step_latency_s`. Only 1F1B is wired up in v1; `torch.distributed.pipelining` ships `ScheduleInterleaved1F1B` / `ScheduleGPipe` that are easy to add. ZB-H1/H2/DualPipe would be larger lifts since they don't exist upstream.

**Measured vs. simulator reference points on 4× H200 (Llama-7B, seq=4096, mbs=1, μb=8):** No AC 1004 ms (sim 1032 ms, −2.7%), Full AC 1309 ms (sim 1286 ms, +1.8%). The bubble-adjusted simulator prediction (`overall_step_latency_s × num_microbatches`) is the one to compare against, not `bottleneck_step_latency_s`. Note that for Llama-7B on 141 GB HBM, the simulator correctly recommends "No AC everywhere" — there's no sweet spot to validate heterogeneous AC until model size / seq_len / mbs push early stages into OOM under No AC.

## Reference material in repo

- `SELECTIVE_AC_KNOWLEDGE_BASE.md` — prior-work synthesis and derivations (Korthikanti, Chen-Xu-Zhang-Guestrin, LoRAct, pipeline schedules). Cite this when explaining *why* a formula has the shape it does.
- `OBSERVATIONS.md` — running log of findings from simulator development, including "Korthikanti selective AC is a no-op with FlashAttention" and the bug-fix history for per-tensor accounting (rotary copies, GQA repeat_kv, SwiGLU silu output, qk-norm fp32 copies). Check this before adjusting the memory model — several surprising behaviors are already documented.
- `results_llama_7b.txt`, `results_qwen3_8b.txt` — last known-good validation numbers.
- `references/llm-analysis/` — placeholder directory for the upstream llm-analysis reference (currently empty).
