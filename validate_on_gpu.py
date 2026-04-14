"""Validate simulator predictions against real GPU activation measurements.

Run this on a machine with an H100 or A100 GPU and a working FlashAttention-2
backend:
    pip install torch transformers flash-attn
    python validate_on_gpu.py

What this does:
1. Builds the Llama-2-7B decoder stack from config (random weights, no download)
2. Measures retained forward activations under No AC, FA-Selective, and Full AC
3. Compares those measurements against the simulator's analytical predictions
4. Reports observed step peaks and timings for context
5. Dumps a CUDA memory snapshot for detailed inspection

This intentionally uses `LlamaModel` plus a tiny scalar loss so the
measurements reflect transformer activations rather than LM-head logits.

Target: <5% error on retained forward activation memory.
"""

import gc
import sys
import types
from dataclasses import dataclass
from pathlib import Path


# ── Dependency check ─────────────────────────────────────────────────────────

def check_deps():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Run this on a GPU machine.")
            sys.exit(1)
        print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"HBM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers {transformers.__version__}")
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    try:
        import flash_attn
        print(f"flash-attn {flash_attn.__version__}")
    except ImportError:
        print("flash-attn not installed — will use PyTorch SDPA (same FA kernel on H100)")


check_deps()

import torch
from torch.utils.checkpoint import checkpoint
from transformers import LlamaConfig, LlamaModel


# ── Add simulator to path ────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from simulator.config import (
    A100_80GB,
    ActivationFunction,
    GPUConfig,
    H100_80GB,
    ModelConfig,
    ParallelismConfig,
)
from simulator.environment import (
    simulate_fa_selective_ac,
    simulate_full_ac,
    simulate_no_ac,
)


# ── Config ───────────────────────────────────────────────────────────────────

SEQ_LEN = 2048
MICRO_BATCH_SIZE = 1
DTYPE = torch.bfloat16
DTYPE_BYTES = torch.tensor(0, dtype=DTYPE).element_size()

# Llama-2-7B decoder architecture.
# Reference config values:
#   hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
#   num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=4096,
#   hidden_act=silu, rms_norm_eps=1e-5, attention_dropout=0.0.
LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    attention_dropout=0.0,
    attention_bias=False,
    mlp_bias=False,
    vocab_size=32000,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    use_cache=False,
    torch_dtype=DTYPE,
)
try:
    import flash_attn  # noqa: F401
    LLAMA_CONFIG._attn_implementation = "flash_attention_2"
    print("Using: flash_attention_2")
except ImportError:
    LLAMA_CONFIG._attn_implementation = "sdpa"
    print("Using: SDPA (PyTorch built-in FlashAttention kernel)")

# Matching simulator config.
SIM_CONFIG = ModelConfig(
    name="llama-2-7b-validate",
    num_layers=32,
    hidden_dim=4096,
    n_heads=32,
    num_kv_heads=32,
    ffn_dim=11008,
    vocab_size=32000,
    seq_len=SEQ_LEN,
    micro_batch_size=MICRO_BATCH_SIZE,
    activation_fn=ActivationFunction.SWIGLU,
    use_flash_attention=True,
    use_attn_dropout=False,
)

MODEL_OUTPUT_BYTES = SEQ_LEN * MICRO_BATCH_SIZE * LLAMA_CONFIG.hidden_size * DTYPE_BYTES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gb(x):
    return f"{x / 1024**3:.3f} GB"


def clear_memory():
    """Aggressively clear GPU memory and reset peak stats."""
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


def detect_gpu_profile() -> GPUConfig:
    """Map the current CUDA device to the closest simulator hardware profile."""
    name = torch.cuda.get_device_name(0).lower()
    total_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3

    if "h100" in name:
        return H100_80GB
    if "a100" in name:
        return A100_80GB

    raise RuntimeError(
        f"Unsupported GPU for this validator: {torch.cuda.get_device_name(0)} "
        f"({total_gb:.1f} GB). Expected an H100 or A100."
    )


def configure_no_ac(model: LlamaModel) -> None:
    disable_fa_selective_checkpointing(model)
    model.gradient_checkpointing_disable()


def configure_full_ac(model: LlamaModel) -> None:
    disable_fa_selective_checkpointing(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def configure_fa_selective(model: LlamaModel) -> None:
    model.gradient_checkpointing_disable()
    enable_fa_selective_checkpointing(model)


def _silu_mul(act_fn, gate_out, up_out):
    """Pointwise SwiGLU activation: silu(gate) * up.

    Defined at module scope so checkpoint() receives a plain function
    with no closure over `self`, avoiding reference cycles.
    """
    return act_fn(gate_out) * up_out


def enable_fa_selective_checkpointing(model: LlamaModel) -> None:
    """Checkpoint only the pointwise SwiGLU activation inside each MLP."""
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward_for_validation"):
            continue

        mlp._original_forward_for_validation = mlp.forward

        def selective_forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            # Recompute only the cheap pointwise activation product in backward.
            linear2_input = checkpoint(
                _silu_mul, self.act_fn, gate, up,
                use_reentrant=False,
            )
            return self.down_proj(linear2_input)

        mlp.forward = types.MethodType(selective_forward, mlp)


def disable_fa_selective_checkpointing(model: LlamaModel) -> None:
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward_for_validation"):
            mlp.forward = mlp._original_forward_for_validation
            delattr(mlp, "_original_forward_for_validation")


# ── Profiling functions ──────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    strategy: str
    param_memory: float
    pre_forward_memory: float
    post_forward_memory: float
    peak_memory: float
    retained_activation_memory: float            # post_forward - pre_forward
    retained_activation_no_output: float         # subtract final model output
    observed_peak_activation_memory: float       # peak - pre_forward
    fwd_time_ms: float
    bwd_time_ms: float


def create_model() -> LlamaModel:
    """Create the Llama-2-7B decoder stack with random weights on GPU."""
    print(f"\nCreating Llama-2-7B decoder stack (random weights, {DTYPE})...")
    try:
        model = LlamaModel(LLAMA_CONFIG).to(dtype=DTYPE, device="cuda")
    except Exception as exc:
        print("ERROR: Failed to create the model with FlashAttention-2 enabled.")
        print("Install a compatible FlashAttention backend for your PyTorch/CUDA stack.")
        raise SystemExit(1) from exc

    model.train()
    configure_no_ac(model)
    print(f"  Decoder parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    return model


def create_input():
    """Create a random input batch."""
    return torch.randint(0, LLAMA_CONFIG.vocab_size, (MICRO_BATCH_SIZE, SEQ_LEN), device="cuda")


def forward_loss(model: LlamaModel, input_ids: torch.Tensor):
    """Run the decoder stack and build a tiny scalar loss for backward."""
    outputs = model(input_ids=input_ids, use_cache=False)
    loss = outputs.last_hidden_state.sum()
    return outputs, loss


def profile_no_ac(model: LlamaModel, input_ids: torch.Tensor) -> ProfileResult:
    """Profile with no activation checkpointing."""
    configure_no_ac(model)
    return _profile_fwd_bwd(model, input_ids, "No AC")


def profile_full_ac(model: LlamaModel, input_ids: torch.Tensor) -> ProfileResult:
    """Profile with full-layer activation checkpointing."""
    configure_full_ac(model)
    return _profile_fwd_bwd(model, input_ids, "Full AC")


def profile_fa_selective(model: LlamaModel, input_ids: torch.Tensor) -> ProfileResult:
    """Profile with FA-era selective AC: checkpoint only the SwiGLU pointwise op."""
    configure_fa_selective(model)
    return _profile_fwd_bwd(model, input_ids, "FA-Selective")


def _profile_fwd_bwd(model: LlamaModel, input_ids: torch.Tensor, strategy_name: str) -> ProfileResult:
    """Core profiling: measure retained activation memory and observed step peak."""
    # Warm up the allocator and kernels under the selected strategy.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        outputs, loss = forward_loss(model, input_ids)
        loss.backward()
        del outputs, loss
    clear_memory()

    # Parameter memory with no gradients / activations.
    model.zero_grad(set_to_none=True)
    clear_memory()
    param_mem = current_mem()

    clear_memory()
    pre_fwd = current_mem()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = model(input_ids=input_ids, use_cache=False)
    loss = outputs.last_hidden_state.sum()  # tiny scalar; include in fwd timing
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end)

    post_fwd = current_mem()
    retained_activation = post_fwd - pre_fwd
    retained_no_output = max(0.0, retained_activation - MODEL_OUTPUT_BYTES)

    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end)

    peak = peak_mem()
    peak_activation = peak - pre_fwd

    del outputs, loss

    return ProfileResult(
        strategy=strategy_name,
        param_memory=param_mem,
        pre_forward_memory=pre_fwd,
        post_forward_memory=post_fwd,
        peak_memory=peak,
        retained_activation_memory=retained_activation,
        retained_activation_no_output=retained_no_output,
        observed_peak_activation_memory=peak_activation,
        fwd_time_ms=fwd_ms,
        bwd_time_ms=bwd_ms,
    )


def dump_memory_snapshot(model: LlamaModel, input_ids: torch.Tensor, filename="memory_snapshot_no_ac.pickle"):
    """Dump a full CUDA memory snapshot for detailed analysis."""
    configure_no_ac(model)
    model.zero_grad(set_to_none=True)
    clear_memory()

    print("\n  Recording memory history for snapshot...")
    torch.cuda.memory._record_memory_history(max_entries=100000)

    outputs, loss = forward_loss(model, input_ids)
    loss.backward()
    torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(filename)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"  Snapshot saved to {filename}")
    print("  View at: https://pytorch.org/memory_viz")

    del outputs, loss


# ── Simulator predictions ───────────────────────────────────────────────────

def get_simulator_predictions(gpu: GPUConfig):
    """Get simulator predictions for each strategy."""
    par = ParallelismConfig()

    return {
        "No AC": simulate_no_ac(SIM_CONFIG, gpu, par=par),
        "FA-Selective": simulate_fa_selective_ac(SIM_CONFIG, gpu, par=par),
        "Full AC": simulate_full_ac(SIM_CONFIG, gpu, par=par),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    gpu_profile = detect_gpu_profile()

    print("=" * 80)
    print("  SIMULATOR VALIDATION: Llama-2-7B decoder activations on GPU")
    print(f"  seq_len={SEQ_LEN}, micro_batch_size={MICRO_BATCH_SIZE}, dtype={DTYPE}")
    print(f"  simulator_gpu_profile={gpu_profile.name}")
    print("=" * 80)

    sim_results = get_simulator_predictions(gpu_profile)
    print("\nSimulator predictions (retained forward activations):")
    for strategy in ["No AC", "FA-Selective", "Full AC"]:
        pred = sim_results[strategy].peak_activation_memory_bytes
        print(f"  {strategy:<20} {_gb(pred)}")

    model = create_model()
    input_ids = create_input()

    print("\n" + "=" * 80)
    print("  PROFILING")
    print("=" * 80)

    results = {}

    for strategy, fn in [
        ("No AC", profile_no_ac),
        ("FA-Selective", profile_fa_selective),
        ("Full AC", profile_full_ac),
    ]:
        print(f"\n--- {strategy} ---")
        result = fn(model, input_ids)
        results[strategy] = result
        print(f"  Param memory:                {_gb(result.param_memory)}")
        print(f"  Retained activations:        {_gb(result.retained_activation_memory)}")
        print(f"  Retained activations (-out): {_gb(result.retained_activation_no_output)}")
        print(f"  Observed step peak:          {_gb(result.observed_peak_activation_memory)}")
        print(f"  Total peak:                  {_gb(result.peak_memory)}")
        print(f"  Forward:                     {result.fwd_time_ms:.1f} ms")
        print(f"  Backward:                    {result.bwd_time_ms:.1f} ms")

    print("\n" + "=" * 80)
    print("  COMPARISON: Simulator vs Measured")
    print("=" * 80)
    print("\n  Retained forward activations, excluding the final model output tensor.")
    print(f"\n  {'Strategy':<20} {'Sim Predicted':>14} {'GPU Measured':>14} {'Error':>8}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*8}")

    errors = {}
    for strategy in ["No AC", "FA-Selective", "Full AC"]:
        pred = sim_results[strategy].peak_activation_memory_bytes
        measured = results[strategy].retained_activation_no_output
        error = (pred - measured) / measured * 100 if measured > 0 else float("inf")
        errors[strategy] = abs(error)
        print(f"  {strategy:<20} {_gb(pred):>14} {_gb(measured):>14} {error:>+7.1f}%")

    print("\n  Measured compute overhead vs No AC:")
    no_ac_total_ms = results["No AC"].fwd_time_ms + results["No AC"].bwd_time_ms
    for strategy in ["FA-Selective", "Full AC"]:
        total_ms = results[strategy].fwd_time_ms + results[strategy].bwd_time_ms
        measured_overhead = total_ms / no_ac_total_ms - 1.0
        sim_overhead = sim_results[strategy].recompute_overhead_pct / 100.0
        print(
            f"  {strategy:<20} measured={measured_overhead:+.1%} "
            f"simulator={sim_overhead:+.1%}"
        )

    print("\n  Memory savings vs No AC:")
    no_ac_mem = results["No AC"].retained_activation_no_output
    for strategy in ["FA-Selective", "Full AC"]:
        saved = 1.0 - results[strategy].retained_activation_no_output / no_ac_mem
        print(f"  {strategy:<20} {saved:.1%}")

    print("\n" + "=" * 80)
    print("  MEMORY SNAPSHOT")
    print("=" * 80)
    dump_memory_snapshot(model, input_ids)

    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    max_error = max(errors.values())
    if max_error < 5:
        print(f"  PASS: Max error {max_error:.1f}% < 5% target")
    elif max_error < 15:
        print(f"  MARGINAL: Max error {max_error:.1f}% — inspect the memory snapshot")
    else:
        print(f"  FAIL: Max error {max_error:.1f}% > 15% — formulas or measurement path still mismatch")

    for strategy in ["No AC", "FA-Selective", "Full AC"]:
        print(f"  {strategy:<20} error={errors[strategy]:.1f}%")


if __name__ == "__main__":
    main()
