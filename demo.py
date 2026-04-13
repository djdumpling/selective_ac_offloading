"""Demo: compare activation strategies with realistic cost modeling.

Now includes:
- FA-era selective AC (recompute activation functions, keep matmul outputs)
- Compression compute cost (no longer modeled as free)
- PCIe contention from NCCL (offload bandwidth reduced under FSDP)
"""

from simulator.config import (
    A100_40GB,
    A100_80GB,
    H100_80GB,
    LayerStrategy,
    ParallelismConfig,
    TensorAction,
    TensorDecision,
    gpt3_175b,
    llama_7b,
    llama_13b,
    llama_70b,
)
from simulator.environment import (
    simulate,
    simulate_fa_selective_ac,
    simulate_full_ac,
    simulate_no_ac,
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
    simulate_selective_ac,
    print_pipeline_result,
)
from simulator.memory_model import get_all_tensors_per_layer
from simulator.offload_model import effective_pcie_bandwidth


def _gb(x: float) -> str:
    return f"{x / 1024**3:.2f} GB"


def _ms(x: float) -> str:
    return f"{x * 1000:.2f}"


def _compression_rank(cfg, par: ParallelismConfig) -> int:
    rows = cfg.seq_len * cfg.micro_batch_size
    cols = cfg.ffn_dim // par.tp_size
    target = max(1, cfg.hidden_dim // 8)
    return min(target, rows, cols)


def build_three_resource_strategy(cfg, par=ParallelismConfig()):
    """Hybrid: offload large MLP tensors + compress linear2 + recompute attn core (non-FA)."""
    tensors = get_all_tensors_per_layer(cfg, par)
    rank = _compression_rank(cfg, par)
    strategies = []

    for layer_idx in range(cfg.num_layers):
        decisions = {}
        for tensor in tensors:
            if not cfg.use_flash_attention and tensor.name in {
                "attn_softmax",
                "attn_softmax_dropout_output",
            }:
                decisions[tensor.name] = TensorDecision(action=TensorAction.RECOMPUTE)
            elif tensor.name in {
                "mlp_gate_output",
                "mlp_up_output",
                "mlp_gelu_input",
            }:
                decisions[tensor.name] = TensorDecision(action=TensorAction.OFFLOAD_CPU)
            elif tensor.name == "mlp_linear2_input":
                decisions[tensor.name] = TensorDecision(
                    action=TensorAction.COMPRESS,
                    compress_rank=rank,
                )
        strategies.append(LayerStrategy(layer_idx=layer_idx, decisions=decisions))

    return strategies


def compare_strategies(name, cfg, gpu, par=ParallelismConfig(), note=""):
    print(f"\n{'=' * 100}")
    print(
        f"  {name} on {gpu.name}"
        f"  (seq={cfg.seq_len}, mbs={cfg.micro_batch_size}, "
        f"tp={par.tp_size}, pp={par.pp_size}, dp={par.dp_size})"
    )
    if note:
        print(f"  {note}")

    # Show effective PCIe bandwidth
    eff_bw = effective_pcie_bandwidth(gpu, par) / (1024 ** 3)
    raw_bw = gpu.pcie_bandwidth_gb_s
    if eff_bw < raw_bw:
        print(f"  PCIe: {raw_bw:.0f} GB/s raw → {eff_bw:.1f} GB/s effective (NCCL contention)")
    else:
        print(f"  PCIe: {raw_bw:.0f} GB/s (no contention)")
    print(f"{'=' * 100}")

    results = {
        "No AC": simulate_no_ac(cfg, gpu, par=par),
        "Full AC": simulate_full_ac(cfg, gpu, par=par),
        "Selective AC": simulate_selective_ac(cfg, gpu, par=par),
        "FA-Selective": simulate_fa_selective_ac(cfg, gpu, par=par),
        "3-Resource": simulate(
            cfg, gpu,
            strategies=build_three_resource_strategy(cfg, par),
            par=par,
        ),
    }

    print(
        f"\n  {'Strategy':<16} {'Activation':>12} {'Peak Total':>12} "
        f"{'Step (ms)':>10} {'Overhead%':>10} {'Fits?':>6}"
    )
    print(
        f"  {'-' * 16} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 6}"
    )

    baseline_step = results["No AC"].step_latency_s

    for sname, result in results.items():
        fits = "YES" if result.fits_in_memory else "NO"
        # Show overhead as % increase in step latency over No AC
        step_overhead = (
            100.0 * (result.step_latency_s / baseline_step - 1.0)
            if baseline_step > 0 else 0.0
        )
        print(
            f"  {sname:<16} {_gb(result.peak_activation_memory_bytes):>12} "
            f"{_gb(result.total_peak_memory_bytes):>12} "
            f"{_ms(result.step_latency_s):>10} "
            f"{step_overhead:>+9.2f}% {fits:>6}"
        )

    # Compare FA-Selective vs No AC (the meaningful comparison for FA models)
    fa_sel = results["FA-Selective"]
    no_ac = results["No AC"]
    mem_saved = 100.0 * (1.0 - fa_sel.peak_activation_memory_bytes / no_ac.peak_activation_memory_bytes)
    step_cost = 100.0 * (fa_sel.step_latency_s / no_ac.step_latency_s - 1.0)
    print(
        f"\n  FA-Selective vs No AC: "
        f"activation memory -{mem_saved:.1f}%, step latency {step_cost:+.2f}%"
    )

    # Compare 3-Resource vs FA-Selective
    hybrid = results["3-Resource"]
    mem_saved_3r = 100.0 * (1.0 - hybrid.peak_activation_memory_bytes / fa_sel.peak_activation_memory_bytes)
    step_cost_3r = 100.0 * (hybrid.step_latency_s / fa_sel.step_latency_s - 1.0)
    print(
        f"  3-Resource vs FA-Selective: "
        f"activation memory {mem_saved_3r:+.1f}%, step latency {step_cost_3r:+.2f}%"
    )


if __name__ == "__main__":
    print("=" * 100)
    print("  REALISTIC COST MODEL DEMO")
    print("  - Compression compute cost: 4×sb×d×r FLOPs per compressed tensor")
    print("  - PCIe contention: FSDP NCCL traffic reduces available offload bandwidth")
    print("  - FA-era selective: recompute activation functions, keep matmul outputs")
    print("=" * 100)

    cases = [
        (
            "Llama-7B",
            llama_7b(seq_len=4096, micro_batch_size=2),
            A100_80GB,
            ParallelismConfig(dp_size=8),
            "FSDP dp=8 (NCCL contention on PCIe)",
        ),
        (
            "Llama-7B",
            llama_7b(seq_len=4096, micro_batch_size=2),
            A100_80GB,
            ParallelismConfig(tp_size=8),
            "TP=8 on NVLink (no PCIe contention)",
        ),
        (
            "Llama-13B",
            llama_13b(seq_len=4096, micro_batch_size=1),
            A100_80GB,
            ParallelismConfig(dp_size=8),
            "FSDP dp=8",
        ),
        (
            "Llama-70B",
            llama_70b(seq_len=4096, micro_batch_size=1),
            H100_80GB,
            ParallelismConfig(tp_size=8, dp_size=8),
            "TP=8 + FSDP dp=8 (multi-node)",
        ),
        (
            "GPT-3 175B",
            gpt3_175b(seq_len=2048, micro_batch_size=1),
            A100_80GB,
            ParallelismConfig(tp_size=8, pp_size=8),
            "Korthikanti Table 3 (no FA)",
        ),
    ]

    for name, cfg, gpu, par, note in cases:
        compare_strategies(name, cfg, gpu, par, note=note)

    # ── Pipeline-position-aware AC comparison ────────────────────────────
    print("\n\n")
    print("=" * 100)
    print("  PIPELINE-POSITION-AWARE AC")
    print("  Under 1F1B, stage p stashes (PP-1-p) microbatch activations.")
    print("  Early stages need aggressive AC; late stages can afford less.")
    print("  Pipeline-aware selects the least aggressive strategy that fits per stage.")
    print("=" * 100)

    pipeline_cases = [
        (
            "Llama-7B PP=4",
            llama_7b(seq_len=4096, micro_batch_size=4),
            A100_80GB,
            ParallelismConfig(pp_size=4, dp_size=4),
        ),
        (
            "GPT-3 175B PP=8",
            gpt3_175b(seq_len=2048, micro_batch_size=1),
            A100_80GB,
            ParallelismConfig(tp_size=8, pp_size=8),
        ),
    ]

    for name, cfg, gpu, par in pipeline_cases:
        print(f"\n{'=' * 100}")
        print(f"  {name} on {gpu.name}  (tp={par.tp_size}, pp={par.pp_size}, dp={par.dp_size})")
        print(f"{'=' * 100}")

        print("\n  --- Uniform Full AC (current practice) ---")
        pr_uniform = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="Full AC")
        print_pipeline_result(pr_uniform)

        print("\n  --- Uniform FA-Selective ---")
        pr_fa_uniform = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="FA-Selective")
        print_pipeline_result(pr_fa_uniform)

        print("\n  --- Pipeline-Aware (adaptive per stage) ---")
        pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)
        print_pipeline_result(pr_aware)

        # Summary comparison
        lat_uniform = pr_uniform.overall_step_latency_s
        lat_aware = pr_aware.overall_step_latency_s
        speedup = (lat_uniform / lat_aware - 1.0) * 100 if lat_aware > 0 else 0
        print(f"\n  Pipeline-aware vs Uniform Full AC: "
              f"step latency {speedup:+.2f}% "
              f"({'faster' if speedup > 0 else 'same'})")

        all_uniform_fit = pr_uniform.all_fit
        all_aware_fit = pr_aware.all_fit
        if all_aware_fit and not all_uniform_fit:
            print("  Pipeline-aware enables fitting when uniform does not!")
        elif all_aware_fit and all_uniform_fit:
            aware_strategies = [sr.strategy_name for sr in pr_aware.stages]
            print(f"  Per-stage strategies: {aware_strategies}")
