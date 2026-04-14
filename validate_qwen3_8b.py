"""Validate simulator predictions against Qwen3-8B on GPU.

Tests GQA (8 KV heads) which makes K/V 4× smaller than MHA.
Run: python validate_qwen3_8b.py
"""

import gc
import sys
import types
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from simulator.config import ActivationFunction, H100_80GB, ModelConfig, ParallelismConfig
from simulator.environment import simulate_no_ac, simulate_fa_selective_ac, simulate_full_ac

# ── Config ───────────────────────────────────────────────────────────────────

SEQ_LEN = 2048
MICRO_BATCH_SIZE = 1
DTYPE = torch.bfloat16
DTYPE_BYTES = 2

MODEL_NAME = "Qwen/Qwen3-8B"

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"HBM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load the HF config
hf_config = AutoConfig.from_pretrained(MODEL_NAME)
hf_config.use_cache = False
hf_config.torch_dtype = DTYPE

# Force SDPA
hf_config._attn_implementation = "sdpa"
print(f"Using: SDPA (PyTorch built-in FlashAttention kernel)")

# Simulator config
SIM_CONFIG = ModelConfig(
    name="qwen3-8b-validate",
    num_layers=hf_config.num_hidden_layers,
    hidden_dim=hf_config.hidden_size,
    n_heads=hf_config.num_attention_heads,
    num_kv_heads=hf_config.num_key_value_heads,
    ffn_dim=hf_config.intermediate_size,
    vocab_size=hf_config.vocab_size,
    seq_len=SEQ_LEN,
    micro_batch_size=MICRO_BATCH_SIZE,
    activation_fn=ActivationFunction.SWIGLU,
    use_flash_attention=True,
    use_attn_dropout=False,
    use_rotary_embeddings=True,
    use_qk_norm=True,
)

MODEL_OUTPUT_BYTES = SEQ_LEN * MICRO_BATCH_SIZE * hf_config.hidden_size * DTYPE_BYTES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gb(x):
    return f"{x / 1024**3:.3f} GB"


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


# ── SwiGLU selective checkpoint ──────────────────────────────────────────────

def _silu_mul(act_fn, gate_out, up_out):
    return act_fn(gate_out) * up_out


def enable_fa_selective_checkpointing(model):
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward"):
            continue
        mlp._original_forward = mlp.forward

        def selective_forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            linear2_input = checkpoint(
                _silu_mul, self.act_fn, gate, up, use_reentrant=False,
            )
            return self.down_proj(linear2_input)

        mlp.forward = types.MethodType(selective_forward, mlp)


def disable_fa_selective_checkpointing(model):
    for layer in model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "_original_forward"):
            mlp.forward = mlp._original_forward
            delattr(mlp, "_original_forward")


# ── Profiling ────────────────────────────────────────────────────────────────

def profile(model, input_ids, strategy_name, configure_fn):
    configure_fn(model)

    # Warmup
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, use_cache=False)
        out.last_hidden_state.sum().backward()
        del out
    clear_memory()

    model.zero_grad(set_to_none=True)
    clear_memory()
    param_mem = current_mem()
    pre_fwd = current_mem()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    outputs = model(input_ids=input_ids, use_cache=False)
    loss = outputs.last_hidden_state.sum()
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end)

    post_fwd = current_mem()
    retained = post_fwd - pre_fwd
    retained_no_out = max(0.0, retained - MODEL_OUTPUT_BYTES)

    start.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end)

    peak = peak_mem()
    del outputs, loss

    print(f"\n--- {strategy_name} ---")
    print(f"  Param memory:                {_gb(param_mem)}")
    print(f"  Retained activations:        {_gb(retained)}")
    print(f"  Retained activations (-out): {_gb(retained_no_out)}")
    print(f"  Forward:                     {fwd_ms:.1f} ms")
    print(f"  Backward:                    {bwd_ms:.1f} ms")

    return retained_no_out, fwd_ms, bwd_ms


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    gpu = H100_80GB

    print("=" * 80)
    print(f"  QWEN3-8B VALIDATION (GQA: {hf_config.num_key_value_heads} KV heads)")
    print(f"  seq_len={SEQ_LEN}, micro_batch_size={MICRO_BATCH_SIZE}, dtype={DTYPE}")
    print(f"  hidden={hf_config.hidden_size}, ffn={hf_config.intermediate_size}, "
          f"layers={hf_config.num_hidden_layers}")
    print("=" * 80)

    # Simulator predictions
    par = ParallelismConfig()
    r_no = simulate_no_ac(SIM_CONFIG, gpu, par=par)
    r_fa = simulate_fa_selective_ac(SIM_CONFIG, gpu, par=par)
    r_full = simulate_full_ac(SIM_CONFIG, gpu, par=par)

    print("\nSimulator predictions (retained forward activations):")
    print(f"  No AC          {_gb(r_no.peak_activation_memory_bytes)}")
    print(f"  FA-Selective   {_gb(r_fa.peak_activation_memory_bytes)}")
    print(f"  Full AC        {_gb(r_full.peak_activation_memory_bytes)}")

    # Build model (random weights, no download)
    print(f"\nCreating Qwen3-8B decoder stack (random weights, {DTYPE})...")
    # Use the base model class (no LM head) — Qwen3 uses Qwen2 architecture
    from transformers import Qwen3Model
    model = Qwen3Model(hf_config).to(dtype=DTYPE, device="cuda")
    model.train()
    print(f"  Decoder parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    input_ids = torch.randint(0, hf_config.vocab_size, (MICRO_BATCH_SIZE, SEQ_LEN), device="cuda")

    print("\n" + "=" * 80)
    print("  PROFILING")
    print("=" * 80)

    def cfg_no_ac(m):
        disable_fa_selective_checkpointing(m)
        m.gradient_checkpointing_disable()

    def cfg_fa_sel(m):
        m.gradient_checkpointing_disable()
        enable_fa_selective_checkpointing(m)

    def cfg_full_ac(m):
        disable_fa_selective_checkpointing(m)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    results = {}
    for name, fn in [("No AC", cfg_no_ac), ("FA-Selective", cfg_fa_sel), ("Full AC", cfg_full_ac)]:
        measured, fwd, bwd = profile(model, input_ids, name, fn)
        results[name] = (measured, fwd, bwd)

    # Comparison
    print("\n" + "=" * 80)
    print("  COMPARISON: Simulator vs Measured")
    print("=" * 80)
    print(f"\n  {'Strategy':<20} {'Sim Predicted':>14} {'GPU Measured':>14} {'Error':>8}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*8}")

    sim = {"No AC": r_no, "FA-Selective": r_fa, "Full AC": r_full}
    errors = {}
    for name in ["No AC", "FA-Selective", "Full AC"]:
        pred = sim[name].peak_activation_memory_bytes
        measured = results[name][0]
        err = (pred - measured) / measured * 100 if measured > 0 else float("inf")
        errors[name] = abs(err)
        print(f"  {name:<20} {_gb(pred):>14} {_gb(measured):>14} {err:>+7.1f}%")

    # Memory savings
    print(f"\n  Memory savings vs No AC:")
    no_ac_mem = results["No AC"][0]
    for name in ["FA-Selective", "Full AC"]:
        saved = 1.0 - results[name][0] / no_ac_mem
        print(f"  {name:<20} {saved:.1%}")

    # Verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    max_error = max(errors.values())
    if max_error < 5:
        print(f"  PASS: Max error {max_error:.1f}% < 5% target")
    elif max_error < 15:
        print(f"  MARGINAL: Max error {max_error:.1f}%")
    else:
        print(f"  FAIL: Max error {max_error:.1f}% > 15%")
    for name in ["No AC", "FA-Selective", "Full AC"]:
        print(f"  {name:<20} error={errors[name]:.1f}%")


if __name__ == "__main__":
    main()
