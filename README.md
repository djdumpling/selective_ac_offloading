# Selective AC Offloading

Analytical simulator and validation scripts for activation checkpointing,
offloading, and compression strategies on transformer training workloads.

## What Is Here

- `simulator/`: core memory, compute, offload, and pipeline models
- `tests/`: unit and integration tests for the simulator
- `demo.py`: strategy comparison across several representative configs
- `demo_pipeline_schedules.py`: schedule-aware pipeline simulations
- `demo_sweet_spot.py`: search for pipeline-aware sweet spots
- `validate_on_gpu.py`: apples-to-apples GPU validation against PyTorch on A100/H100

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

## GPU Validation

`validate_on_gpu.py` is intended for a CUDA machine with an A100 or H100 and a
working FlashAttention-2 backend.

Install the required runtime packages for your CUDA / PyTorch stack:

```bash
python3 -m pip install torch transformers flash-attn
```

Then run:

```bash
python3 validate_on_gpu.py
```

Notes:

- The script uses `LlamaModel`, not `LlamaForCausalLM`, so measured memory tracks
  transformer activations rather than LM-head logits.
- The script writes a CUDA memory snapshot named `memory_snapshot_no_ac.pickle`
  for inspection with https://pytorch.org/memory_viz.
- If FlashAttention-2 is not available for the installed PyTorch / CUDA stack,
  model creation will fail early with a setup error.
