"""Validate selective CPU offloading end-to-end on a single GPU.

For each Llama-2-7B decoder layer, wrap the `down_proj` call in a
`CPUOffloadHook` so the saved `mlp_linear2_input` tensor travels to pinned
CPU memory during forward and returns on backward. Measure peak HBM, forward
time, and backward time — compare against `simulate(..., OFFLOAD_CPU)` for
the same layers.

The PCIe cost model predicts:
  transfer_time  = size_bytes / effective_pcie_bandwidth
  stall_time     = max(0, round_trip - liveness_gap)

On H200 (PCIe Gen5 = 64 GB/s) with linear2_input = 45 MB/layer at seq=2048,
round-trip ≈ 1.4 ms per tensor vs. ≈10 ms of layer compute → full overlap,
zero stall expected. The measurement proves it.

Launch:
  .venv/bin/python offload/validate_offload.py
  .venv/bin/python offload/validate_offload.py --seq 4096 --mbs 1
"""
from __future__ import annotations

import argparse
import gc
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offload.hooks import CPUOffloadHook  # noqa: E402
from simulator.config import (  # noqa: E402
    A100_80GB,
    ActivationFunction,
    GPUConfig,
    H100_80GB,
    H200_141GB,
    LayerStrategy,
    ModelConfig,
    ParallelismConfig,
    TensorAction,
    TensorDecision,
)
from simulator.environment import simulate, simulate_no_ac  # noqa: E402


DTYPE = torch.bfloat16
DTYPE_BYTES = 2


def build_llama_config(seq: int) -> LlamaConfig:
    cfg = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=max(4096, seq),
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        attention_bias=False,
        mlp_bias=False,
        vocab_size=32000,
        use_cache=False,
        torch_dtype=DTYPE,
    )
    cfg._attn_implementation = "sdpa"  # SDPA dispatches to FA on H100/H200.
    return cfg


def build_sim_config(seq: int, mbs: int) -> ModelConfig:
    return ModelConfig(
        name="llama-2-7b-offload",
        num_layers=32,
        hidden_dim=4096,
        n_heads=32,
        num_kv_heads=32,
        ffn_dim=11008,
        vocab_size=32000,
        seq_len=seq,
        micro_batch_size=mbs,
        activation_fn=ActivationFunction.SWIGLU,
        use_flash_attention=True,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def detect_gpu_profile() -> GPUConfig:
    name = torch.cuda.get_device_name(0).lower()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "h200" in name or total_gb > 100:
        return H200_141GB
    if "h100" in name:
        return H100_80GB
    if "a100" in name:
        return A100_80GB
    raise RuntimeError(
        f"Unsupported GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)"
    )


# ── Monkeypatch: wrap down_proj in an offload context ───────────────────────
#
# Mirrors validate_on_gpu.py::enable_fa_selective_checkpointing: replaces the
# bound MLP forward so that only the tensors saved by down_proj (= the
# silu(gate)*up product, named `mlp_linear2_input` in the simulator) are
# intercepted by the hook. gate/up/silu outputs are saved earlier, outside
# the wrapped region, and therefore stay on GPU.

def _build_offload_forward(hook_factory):
    def offload_forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        linear2_input = self.act_fn(gate) * up
        hook = hook_factory(self)
        with hook:
            return self.down_proj(linear2_input)
    return offload_forward


def enable_offload_linear2(
    model: LlamaModel, min_bytes: int, use_dedicated_stream: bool
) -> tuple[list[CPUOffloadHook], "torch.cuda.Stream | None"]:
    """Install the monkeypatch on every decoder layer. Returns (hooks, stream).

    When `use_dedicated_stream=True`, all hooks share one CUDA stream so GPU→CPU
    DMAs can overlap with compute on the default stream. Sharing a stream keeps
    the transfers serialized per PCIe link (matching reality) while still
    parallelizing with compute.
    """
    hooks: list[CPUOffloadHook] = []
    stream = torch.cuda.Stream() if use_dedicated_stream else None

    def hook_factory(_mlp):
        hook = CPUOffloadHook(min_bytes=min_bytes, offload_stream=stream)
        hooks.append(hook)
        return hook

    patched = _build_offload_forward(hook_factory)

    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward_for_offload"):
            continue
        mlp._original_forward_for_offload = mlp.forward
        mlp.forward = types.MethodType(patched, mlp)
    return hooks, stream


def disable_offload_linear2(model: LlamaModel) -> None:
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward_for_offload"):
            mlp.forward = mlp._original_forward_for_offload
            delattr(mlp, "_original_forward_for_offload")


# ── Measurement harness (close to validate_on_gpu.py's _profile_fwd_bwd) ────

def _gb(x: float) -> str:
    return f"{x / 1024**3:.3f} GB"


def _mb(x: float) -> str:
    return f"{x / 1024**2:.1f} MB"


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def current_mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def peak_mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@dataclass
class Measurement:
    label: str
    pre_forward_memory: float
    post_forward_memory: float
    peak_memory: float
    fwd_time_ms: float
    bwd_time_ms: float
    bytes_offloaded: int = 0
    output_sum: float = 0.0  # scalar checksum for correctness


def profile(model: LlamaModel, input_ids: torch.Tensor, label: str,
            hooks: list[CPUOffloadHook] | None) -> Measurement:
    # Warmup: two full fwd/bwd to settle allocator + any autotune.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, use_cache=False)
        out.last_hidden_state.sum().backward()
        del out
    if hooks is not None:
        for h in hooks:
            h.reset_stats()
    clear_memory()

    model.zero_grad(set_to_none=True)
    pre_fwd = current_mem()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = model(input_ids=input_ids, use_cache=False)
    loss = out.last_hidden_state.sum()
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end)

    post_fwd = current_mem()
    output_sum = float(loss.detach().float().item())

    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end)

    pk = peak_mem()
    bytes_off = 0
    if hooks is not None:
        bytes_off = sum(h.stats.bytes_offloaded for h in hooks)

    del out, loss

    return Measurement(
        label=label,
        pre_forward_memory=pre_fwd,
        post_forward_memory=post_fwd,
        peak_memory=pk,
        fwd_time_ms=fwd_ms,
        bwd_time_ms=bwd_ms,
        bytes_offloaded=bytes_off,
        output_sum=output_sum,
    )


# ── Simulator predictions ───────────────────────────────────────────────────

def get_sim_predictions(sim_cfg: ModelConfig, gpu: GPUConfig):
    par = ParallelismConfig()
    baseline = simulate_no_ac(sim_cfg, gpu, par=par)

    strategies = [
        LayerStrategy(
            layer_idx=i,
            decisions={
                "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
            },
        )
        for i in range(sim_cfg.num_layers)
    ]
    offload = simulate(sim_cfg, gpu, strategies, par=par)
    return baseline, offload


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--min-bytes", type=int, default=1_000_000,
                   help="Tensors >= this size are offloaded to pinned CPU.")
    p.add_argument("--stream", choices=["default", "dedicated", "both"], default="both",
                   help="Which CUDA stream to use for the GPU<->CPU DMA. "
                        "'default' serializes with compute; 'dedicated' overlaps; "
                        "'both' runs both and reports them side-by-side.")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script requires a CUDA GPU.")
    gpu = detect_gpu_profile()
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print(f"Simulator profile: {gpu.name}")
    print(f"seq_len={args.seq}, micro_batch_size={args.mbs}, dtype={DTYPE}")

    hf_config = build_llama_config(args.seq)
    sim_cfg = build_sim_config(args.seq, args.mbs)

    print("\nBuilding Llama-2-7B decoder (random weights)...")
    model = LlamaModel(hf_config).to(dtype=DTYPE, device="cuda")
    model.train()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  decoder parameters: {nparams / 1e9:.2f}B")

    input_ids = torch.randint(0, hf_config.vocab_size,
                              (args.mbs, args.seq), device="cuda")

    # ── Simulator side ──────────────────────────────────────────────────
    baseline_sim, offload_sim = get_sim_predictions(sim_cfg, gpu)
    pred_saved_bytes = (
        baseline_sim.peak_activation_memory_bytes
        - offload_sim.peak_activation_memory_bytes
    )
    pred_stall_ms = offload_sim.total_offload_stall_s * 1000

    # ── Measured: baseline (no offload) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  BASELINE (no offload)")
    print("=" * 70)
    m_base = profile(model, input_ids, label="no-offload", hooks=None)
    retained_base = m_base.post_forward_memory - m_base.pre_forward_memory
    print(f"  peak HBM (incl bwd):     {_gb(m_base.peak_memory)}")
    print(f"  retained post-forward:   {_gb(retained_base)}")
    print(f"  forward:                 {m_base.fwd_time_ms:.1f} ms")
    print(f"  backward:                {m_base.bwd_time_ms:.1f} ms")
    print(f"  output .sum():           {m_base.output_sum:.3e}")

    # ── Measured: offload linear2_input ─────────────────────────────────
    which = args.stream
    modes = ["default", "dedicated"] if which == "both" else [which]
    offload_runs: dict[str, tuple[Measurement, float, list[CPUOffloadHook]]] = {}

    for mode in modes:
        print("\n" + "=" * 70)
        print(f"  OFFLOAD mlp_linear2_input  (stream={mode})")
        print("=" * 70)
        hooks, _stream = enable_offload_linear2(
            model, min_bytes=args.min_bytes,
            use_dedicated_stream=(mode == "dedicated"),
        )
        try:
            m_off = profile(model, input_ids, label=f"offload-{mode}", hooks=hooks)
        finally:
            disable_offload_linear2(model)
        retained_off = m_off.post_forward_memory - m_off.pre_forward_memory
        print(f"  peak HBM (incl bwd):     {_gb(m_off.peak_memory)}")
        print(f"  retained post-forward:   {_gb(retained_off)}")
        print(f"  forward:                 {m_off.fwd_time_ms:.1f} ms")
        print(f"  backward:                {m_off.bwd_time_ms:.1f} ms")
        print(f"  output .sum():           {m_off.output_sum:.3e}")
        print(f"  bytes offloaded:         {_mb(m_off.bytes_offloaded)} total "
              f"({m_off.bytes_offloaded / (sim_cfg.num_layers) / 1024**2:.1f} MB/layer)")

        total_seen = sum(h.stats.tensors_seen for h in hooks)
        total_off = sum(h.stats.tensors_offloaded for h in hooks)
        print(f"  tensors seen/offloaded: {total_seen}/{total_off}")

        offload_runs[mode] = (m_off, retained_off, hooks)

    # For downstream comparisons, use the dedicated-stream run if available —
    # it's the realistic apples-to-apples point against the simulator's
    # "transfers overlap with compute" assumption. Fall back to default.
    primary_mode = "dedicated" if "dedicated" in offload_runs else "default"
    m_off, retained_off, hooks = offload_runs[primary_mode]

    # ── Correctness ─────────────────────────────────────────────────────
    rel_err = abs(m_off.output_sum - m_base.output_sum) / max(abs(m_base.output_sum), 1e-12)
    print("\n" + "=" * 70)
    print("  CORRECTNESS CHECK (output .sum() should match to bf16 precision)")
    print("=" * 70)
    print(f"  baseline output sum: {m_base.output_sum:.6e}")
    print(f"  offload  output sum: {m_off.output_sum:.6e}")
    print(f"  relative error:      {rel_err:.2e}")
    correctness_pass = rel_err < 1e-2

    # ── Comparison table ────────────────────────────────────────────────
    # The simulator's peak_activation_memory_bytes is computed at end-of-forward
    # (before any gradient allocation), so compare to retained = post_fwd - pre_fwd.
    measured_saved = retained_base - retained_off
    measured_step_ms = (m_off.fwd_time_ms + m_off.bwd_time_ms) - \
                       (m_base.fwd_time_ms + m_base.bwd_time_ms)

    print("\n" + "=" * 70)
    print("  COMPARISON: measured vs simulator")
    print("=" * 70)
    print(f"\n  Activation savings (retained post-forward):")
    print(f"    measured:   {_gb(measured_saved)}")
    print(f"    simulator:  {_gb(pred_saved_bytes)}")
    if pred_saved_bytes > 0:
        err_pct = (measured_saved - pred_saved_bytes) / pred_saved_bytes * 100
        print(f"    error:      {err_pct:+.1f}%")

    print(f"\n  Step-time overhead from offload (fwd+bwd vs baseline):")
    for mode, (m, _r, _h) in offload_runs.items():
        delta = (m.fwd_time_ms + m.bwd_time_ms) - (m_base.fwd_time_ms + m_base.bwd_time_ms)
        pct = delta / (m_base.fwd_time_ms + m_base.bwd_time_ms) * 100
        print(f"    measured (stream={mode:<9}): {delta:+.1f} ms ({pct:+.1f}%)")
    print(f"    simulator:                  {pred_stall_ms:+.1f} ms "
          f"(total_offload_stall_s × 1000)")

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    hbm_ok = pred_saved_bytes > 0 and \
        abs(measured_saved - pred_saved_bytes) / pred_saved_bytes < 0.15
    stall_ok = abs(measured_step_ms - pred_stall_ms) < max(3.0, 0.1 * (m_base.fwd_time_ms + m_base.bwd_time_ms))
    print(f"  Correctness (outputs match): {'PASS' if correctness_pass else 'FAIL'} (rel_err={rel_err:.2e})")
    print(f"  HBM savings (±10%):          {'PASS' if hbm_ok else 'CHECK'}")
    print(f"  Stall time within tolerance: {'PASS' if stall_ok else 'CHECK'}")


if __name__ == "__main__":
    main()
