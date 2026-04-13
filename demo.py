"""Demo: compare AC strategies on Llama-7B and GPT-3 175B."""

from simulator.config import (
    A100_80GB,
    H100_80GB,
    ParallelismConfig,
    TensorAction,
    TensorDecision,
    LayerStrategy,
    llama_7b,
    llama_70b,
    gpt3_175b,
)
from simulator.memory_model import (
    get_all_tensors_per_layer,
    get_korthikanti_reference,
)
from simulator.environment import (
    simulate,
    simulate_no_ac,
    simulate_full_ac,
    simulate_selective_ac,
    print_result,
)


def _gb(x: float) -> str:
    return f"{x / 1024**3:.2f} GB"


def compare_strategies(name, cfg, gpu, par=ParallelismConfig()):
    print(f"\n{'='*70}")
    print(f"  {name} on {gpu.name}  (seq={cfg.seq_len}, mbs={cfg.micro_batch_size})")
    print(f"{'='*70}")

    results = {
        "No AC": simulate_no_ac(cfg, gpu, par=par),
        "Full AC": simulate_full_ac(cfg, gpu, par=par),
        "Selective AC": simulate_selective_ac(cfg, gpu, par=par),
    }

    # Custom "FA-aware" strategy: keep attention (FA handles it),
    # offload large MLP tensors, recompute dropout masks
    tensors = get_all_tensors_per_layer(cfg, par)
    strategies = []
    for i in range(cfg.num_layers):
        decisions = {}
        for t in tensors:
            if "gate_output" in t.name or "up_output" in t.name or "gelu_input" in t.name:
                decisions[t.name] = TensorDecision(action=TensorAction.OFFLOAD_CPU)
            elif "linear2_input" in t.name:
                decisions[t.name] = TensorDecision(action=TensorAction.COMPRESS, compress_rank=cfg.hidden_dim // 8)
            elif "dropout_mask" in t.name:
                decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
            # Everything else: KEEP (default)
        strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))

    results["3-Resource (offload+compress+recompute)"] = simulate(
        cfg, gpu, strategies=strategies, par=par
    )

    print(f"\n  {'Strategy':<42} {'Activation':>12} {'Peak Total':>12} {'Recomp%':>8} {'Fits?':>6}")
    print(f"  {'-'*42} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")
    for sname, r in results.items():
        fits = "YES" if r.fits_in_memory else "NO"
        print(f"  {sname:<42} {_gb(r.peak_activation_memory_bytes):>12} "
              f"{_gb(r.total_peak_memory_bytes):>12} "
              f"{r.recompute_overhead_pct:>7.2f}% {fits:>6}")

    # Show per-tensor breakdown for 3-resource strategy (layer 0)
    r3 = results["3-Resource (offload+compress+recompute)"]
    print(f"\n  Layer 0 tensor decisions (3-Resource):")
    for tname, action in r3.per_layer[0].tensor_details.items():
        print(f"    {tname:<30} {action}")


if __name__ == "__main__":
    # ── Llama-7B on A100-80GB with FSDP ─────────────────────────────────
    compare_strategies(
        "Llama-7B",
        llama_7b(seq_len=4096, micro_batch_size=2),
        A100_80GB,
        par=ParallelismConfig(dp_size=8),
    )

    # ── Llama-70B on H100 with TP=8 ─────────────────────────────────────
    compare_strategies(
        "Llama-70B",
        llama_70b(seq_len=4096, micro_batch_size=1),
        H100_80GB,
        par=ParallelismConfig(tp_size=8, dp_size=8),
    )

    # ── Korthikanti reference check ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Korthikanti Reference Formulas (GPT-3 175B, no FA)")
    print(f"{'='*70}")
    cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
    ref = get_korthikanti_reference(cfg)
    print(f"  No AC per layer:        {_gb(ref['no_ac'])}")
    print(f"  Selective AC per layer: {_gb(ref['selective_ac'])}")
    print(f"  Full AC per layer:      {_gb(ref['full_ac'])}")
    print(f"  No AC total (96 layers): {_gb(ref['no_ac'] * 96)}")
