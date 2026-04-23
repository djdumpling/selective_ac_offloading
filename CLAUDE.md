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

# Single-GPU selective offload validation (saved_tensors_hooks to pinned CPU):
.venv/bin/python offload/validate_offload.py --seq 2048 --mbs 1            # default: compare both streams
.venv/bin/python offload/validate_offload.py --seq 4096 --stream dedicated # measure overlap-enabled path only

# Multi-GPU throughput validation (torch.distributed.pipelining, 1F1B):
./throughput/launch.sh 2 full-ac        --seq 2048 --mbs 1 --microbatches 4
./throughput/launch.sh 4 no-ac          --seq 4096 --mbs 1 --microbatches 8
./throughput/launch.sh 4 pipeline-aware --seq 4096 --mbs 1 --microbatches 8
./throughput/launch.sh 4 offload-all-mlp --seq 32768 --mbs 1 --microbatches 8   # long-context sweet spot
# First arg = pp_size (= nproc_per_node); second = {no-ac, full-ac, pipeline-aware,
#   offload-linear2, offload-all-mlp}.
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

### Selective offload validation (`offload/validate_offload.py`)

Offloads `mlp_linear2_input` on every Llama-2-7B decoder layer to pinned CPU via `torch.autograd.graph.saved_tensors_hooks`, then compares measured peak HBM and step-time overhead to `simulate(..., OFFLOAD_CPU)`. Reuses the `LlamaMLP.forward` monkeypatch pattern from `validate_on_gpu.py` so only tensors saved by `down_proj` get captured — gate/up/silu outputs remain on GPU. The hook filters parameter-like tensors (including views like `W.T` that `F.linear` saves) via `_is_parameter_like`, otherwise the PCIe bus fills with weight transposes and stalls compute.

Two transfer modes: `--stream default` (DMA on the default CUDA stream → serializes with compute) and `--stream dedicated` (shared offload stream with cross-stream events → overlaps with compute). The simulator's `offload_model.py` assumes dedicated-stream behavior; the validator reports both so the gap is visible.

```bash
.venv/bin/python offload/validate_offload.py --seq 2048 --mbs 1
.venv/bin/python offload/validate_offload.py --seq 4096 --mbs 1
```

**Measured on 1× H200, Llama-2-7B, mbs=1:** HBM savings match simulator to 0% at both seq=2048 (1.34 GB) and seq=4096 (2.69 GB). With a dedicated stream, step-time overhead is +1.5% / +0.4% respectively (vs. simulator's 0% prediction). With the default stream, overhead is +28.6% / +24.9% — useful as a measurement of the cost of naive synchronous offload.

### Multi-GPU throughput validation (`throughput/run_pipeline.py`)

Pipeline-parallel runner on `torch.distributed.pipelining.Schedule1F1B`. Builds a hand-constructed `LlamaStage` per rank (embed on stage 0, `LlamaDecoderLayer` slice in the middle, final RMSNorm on stage N-1) — not `LlamaModel`, because HF's forward can't be trivially sharded. RoPE is recomputed per stage (weight-free). Activation checkpointing uses `apply_activation_checkpointing` with `checkpoint_wrapper` targeting `LlamaDecoderLayer`; per-stage policy comes from `throughput/strategies.py`. Uniform modes (`no-ac`, `full-ac`, `offload-linear2`, `offload-all-mlp`) use `stage_strategies()` which returns the same strategy for every stage.

`--ac pipeline-aware` uses `pipeline_aware_stage_strategies(cfg, gpu, par, schedule, num_microbatches)` instead: it delegates to `simulator.environment.simulate_pipeline_aware_ac`, which walks `STRATEGY_LEVELS` least-aggressive-first and picks the first that fits each stage's HBM budget given the 1F1B stash. Simulator strategy names (`"No AC"`, `"Offload all MLP"`, `"Full AC"`, etc.) are mapped back to runner names via `SIM_TO_RUNNER_STRATEGY`. If the simulator picks `FA-Selective` or `Korthikanti Selective` — which the runner can't execute (no Megatron-style per-module wrapping yet) — `pipeline_aware_stage_strategies` raises `ValueError` with a pointer to fall back to a supported uniform mode. The simulator is deterministic and torch-free, so every rank computes the same assignment without coordination.

CPU-offload modes (`offload-linear2`, `offload-all-mlp`) install the `LlamaMLP.forward` monkeypatch from `offload/hooks.py::CPUOffloadHook`. Each rank creates one shared CUDA stream so GPU→CPU DMAs overlap with compute on the default stream. Mapped to simulator strategy names via `throughput/strategies.py::RUNNER_TO_SIM_STRATEGY` so the runner can print simulator predictions for the same strategy.

`--offload-sync-mode {overlap, serial}` controls which cost model the simulator uses for the comparison printout. `overlap` (default) matches the dedicated-stream behavior the runner actually installs; `serial` models the default-stream penalty (stall = round_trip per tensor, ignoring liveness gap) and is useful for predicting what a naive `saved_tensors_hooks` implementation without a dedicated stream would cost. At Llama-7B seq=2048 single-tensor offload, serial predicts 42 ms stall/mb vs measured +49.7 ms on default stream.

Optimizer step is intentionally omitted — measured step time is fwd + bwd + recompute only, directly comparable to the simulator's `step_latency_s`. Three schedules are wired up via `--schedule {1f1b, gpipe, interleaved-1f1b}`: `1f1b` is the default Narayanan et al. 2021 schedule; `gpipe` runs all forwards before any backward (worst-case memory, smaller bubble at low M); `interleaved-1f1b` splits each rank into `--num-chunks` virtual stages for a proportionally reduced bubble. ZB-H1/H2/ZB-V/DualPipe stay simulator-only — they don't exist upstream in `torch.distributed.pipelining`.

Model support spans `--model {llama7b, llama13b, qwen3_8b}`. Qwen3-8B uses a `Qwen3Stage` class that mirrors `LlamaStage` but pulls in `Qwen3DecoderLayer` (GQA with 8 KV heads, per-head QK-norm). The offload-hook monkeypatch and full-AC wrapper both accept either decoder type via `isinstance` on a tuple.

**Short-context reference points on 4× H200 (Llama-7B, seq=4096, mbs=1, μb=8):** No AC 1004 ms (sim 1032 ms, −2.7%), Full AC 1309 ms (sim 1286 ms, +1.8%). The bubble-adjusted simulator prediction (`overall_step_latency_s × num_microbatches`) is the one to compare against, not `bottleneck_step_latency_s`. At this scale all strategies fit on 141 GB HBM and pipeline-aware reduces to uniform No AC everywhere.

**Long-context sweet spot on 4× H200 (Llama-7B, seq=32768, mbs=1, μb=8):** No AC OOMs (sim predicts 208 GB/stage > 141 GB HBM). Full AC 15,384 ms / 17,040 tok/s at 27 GB peak. Uniform Offload all MLP 13,927 ms / 18,821 tok/s at 103 GB peak (+10.4% throughput vs Full AC). At this config, `pipeline-aware` picks `[offload-all-mlp, offload-all-mlp, no-ac, no-ac]` — offload only on stages 0-1 which carry 1F1B stash pressure (3 and 2 microbatches), no-ac on stages 2-3 (stash 1 and 0). That heterogeneous assignment is what the repo's thesis is really about, and needs a GPU measurement to compare against uniform offload-all-mlp. The simulator's bus model (v4a interval-based) correctly predicts zero stall here because per-layer bus work (88 ms for 4 MLP tensors) fits inside the per-layer compute window (~180 ms); the 14% gap vs. measured step time is not bus contention but likely stream-dispatch / DMA-launch overhead.

## Reference material in repo

- `SELECTIVE_AC_KNOWLEDGE_BASE.md` — prior-work synthesis and derivations (Korthikanti, Chen-Xu-Zhang-Guestrin, LoRAct, pipeline schedules). Cite this when explaining *why* a formula has the shape it does.
- `OBSERVATIONS.md` — running log of findings from simulator development, including "Korthikanti selective AC is a no-op with FlashAttention" and the bug-fix history for per-tensor accounting (rotary copies, GQA repeat_kv, SwiGLU silu output, qk-norm fp32 copies). Check this before adjusting the memory model — several surprising behaviors are already documented.
- `results_llama_7b.txt`, `results_qwen3_8b.txt` — last known-good validation numbers.
- `references/llm-analysis/` — placeholder directory for the upstream llm-analysis reference (currently empty).
