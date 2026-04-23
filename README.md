# Selective AC Offloading

Analytical simulator and validation scripts for activation checkpointing,
offloading, and compression strategies on transformer training workloads.

## What Is Here

- `simulator/`: core memory, compute, offload, and pipeline models
- `tests/`: unit and integration tests for the simulator
- `demo.py`: strategy comparison across several representative configs
- `demo_pipeline_schedules.py`: schedule-aware pipeline simulations
- `demo_sweet_spot.py`: search for pipeline-aware sweet spots
- `validate_on_gpu.py`: apples-to-apples GPU validation against PyTorch on A100/H100/H200
- `throughput/`: multi-GPU pipeline-parallel throughput runner (torch.distributed.pipelining)

## Local Simulator Testing

The simulator and tests do not require a GPU.

1. Install Python 3.11+ and `pytest`.
2. Run:

```bash
python3 -m pytest
```

Useful smoke checks:

```bash
python3 demo.py
python3 demo_pipeline_schedules.py
python3 demo_sweet_spot.py
```

## GPU Environment

Create a venv with [uv](https://docs.astral.sh/uv/) (recommended on shared clusters) and install the GPU stack:

```bash
uv venv --python 3.11
uv pip install torch --torch-backend=cu128        # match your host CUDA
uv pip install -r requirements.txt
```

The venv is placed at `.venv/` and is gitignored. Use `.venv/bin/python` for everything below.

## Single-GPU Validation

`validate_on_gpu.py` is intended for a CUDA machine with an A100/H100/H200 and a working
FlashAttention-2 backend (or PyTorch SDPA, which dispatches to FA on H100/H200).

```bash
.venv/bin/python validate_on_gpu.py        # Llama-2-7B
.venv/bin/python validate_qwen3_8b.py      # Qwen3-8B (GQA + QK-norm)
```

Notes:

- The scripts use `LlamaModel`/`AutoModel`, not `ForCausalLM`, so measured memory
  tracks transformer activations rather than LM-head logits.
- `validate_on_gpu.py` writes a CUDA memory snapshot named
  `memory_snapshot_no_ac.pickle` for inspection with <https://pytorch.org/memory_viz>.

## Multi-GPU Throughput Validation

`throughput/run_pipeline.py` runs a hand-split Llama stack under
`torch.distributed.pipelining.Schedule1F1B` and compares measured step latency
and per-rank peak HBM against the simulator's predictions.

```bash
# 2 GPUs, full activation checkpointing everywhere:
./throughput/launch.sh 2 full-ac --seq 2048 --mbs 1 --microbatches 4

# 4 GPUs, no checkpointing:
./throughput/launch.sh 4 no-ac --seq 4096 --mbs 1 --microbatches 8

# 4 GPUs, pipeline-aware (full-ac on first half of stages, no-ac on the rest):
./throughput/launch.sh 4 pipeline-aware --seq 4096 --mbs 1 --microbatches 8
```

The first argument is `pp_size` (also `--nproc_per_node`); the second is one of
`no-ac`, `full-ac`, `pipeline-aware`. Rank 0 prints a measured-vs-predicted
comparison at the end of each run.

Scope: v1 supports the 1F1B schedule only; optimizer step is intentionally
skipped so measured wall time is directly comparable to the simulator's
`step_latency_s` (fwd + bwd + recompute, no optimizer). ZB-H1/H2/DualPipe are
simulator-only — they don't exist in torch upstream.

**Reference measurements on 4× H200 (Llama-7B, seq=4096, mbs=1, μb=8):**

| Strategy | Measured step | Simulator (w/ bubble) | Error |
|---|---|---|---|
| No AC | 1004 ms | 1032 ms | −2.7% |
| Full AC | 1309 ms | 1286 ms | +1.8% |

Per-rank peak HBM under No AC decreases 30 → 24 → 18 → 13 GB across stages 0-3,
matching the 1F1B `(PP − 1 − p)` stash formula.
