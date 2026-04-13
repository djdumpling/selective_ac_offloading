"""Find configs where pipeline-aware AC actually improves throughput.

The "sweet spot" is where:
- No AC + stash on stage 0 → OOM
- FA-Selective + stash on stage 0 → fits
- Pipeline-aware assigns FA-Selective to early stages, No AC to late stages
- Bottleneck overhead drops from 24% (Full AC) to ~0% (FA-Selective)
"""

from simulator.config import (
    A100_80GB,
    H100_80GB,
    ModelConfig,
    ActivationFunction,
    ParallelismConfig,
    llama_7b,
)
from simulator.environment import (
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
    print_pipeline_result,
)
from simulator.pipeline_schedules import PipelineSchedule


def small_model(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    """~4B parameter model (Qwen3-4B-like)."""
    return ModelConfig(
        name="small-4b",
        num_layers=40,
        hidden_dim=2560,
        n_heads=20,
        num_kv_heads=4,
        ffn_dim=6912,
        vocab_size=151936,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
    )


def medium_model(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    """~8B parameter model (Llama-3.1-8B-like)."""
    return ModelConfig(
        name="medium-8b",
        num_layers=32,
        hidden_dim=4096,
        n_heads=32,
        num_kv_heads=8,
        ffn_dim=14336,
        vocab_size=128256,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
    )


def _gb(x: float) -> str:
    return f"{x / 1024**3:.1f}"


def search_sweet_spot(name, model_fn, gpu, pp_values, dp, seq_lens, mbs_values):
    """Search over configs to find where pipeline-aware AC improves throughput."""
    print(f"\n{'='*100}")
    print(f"  SEARCHING: {name}")
    print(f"{'='*100}")

    found_any = False

    for pp in pp_values:
        for seq in seq_lens:
            for mbs in mbs_values:
                cfg = model_fn(seq_len=seq, micro_batch_size=mbs)
                par = ParallelismConfig(pp_size=pp, dp_size=dp)

                pr_uniform_full = simulate_pipeline_uniform_ac(
                    cfg, gpu, par, strategy_name="Full AC")
                pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)

                if not pr_aware.all_fit:
                    continue

                aware_strategies = [sr.strategy_name for sr in pr_aware.stages]
                bottleneck_strategy = pr_aware.bottleneck.strategy_name

                # The sweet spot: pipeline-aware bottleneck is LESS aggressive than Full AC
                throughput_gain = (
                    pr_uniform_full.overall_step_latency_s / pr_aware.overall_step_latency_s - 1
                ) * 100 if pr_aware.overall_step_latency_s > 0 else 0

                # Only show interesting cases
                if bottleneck_strategy != "Full AC" and len(set(aware_strategies)) >= 1:
                    found_any = True
                    print(f"\n  CONFIG: pp={pp}, seq={seq}, mbs={mbs}, dp={dp}")
                    print(f"  Uniform Full AC: step={pr_uniform_full.overall_step_latency_s*1000:.1f}ms, "
                          f"overhead={pr_uniform_full.total_recompute_overhead_pct:.1f}%, "
                          f"fit={'YES' if pr_uniform_full.all_fit else 'NO'}")
                    print(f"  Pipeline-Aware:  step={pr_aware.overall_step_latency_s*1000:.1f}ms, "
                          f"overhead={pr_aware.total_recompute_overhead_pct:.1f}%, "
                          f"fit=YES")
                    print(f"  THROUGHPUT GAIN: {throughput_gain:+.1f}%")
                    print(f"  Per-stage: {aware_strategies}")
                    print(f"  Bottleneck: stage {pr_aware.bottleneck_stage} ({bottleneck_strategy})")

                    # Show the memory breakdown for the bottleneck stage
                    bn = pr_aware.bottleneck
                    print(f"    Stage {bn.stage_idx}: stash={bn.num_stashed_microbatches}, "
                          f"act={_gb(bn.sim.peak_activation_memory_bytes)}GB, "
                          f"peak={_gb(bn.sim.total_peak_memory_bytes)}GB / "
                          f"{_gb(bn.sim.hbm_capacity_bytes)}GB budget")

    if not found_any:
        print("  No sweet-spot configs found in search range.")


if __name__ == "__main__":
    print("="*100)
    print("  PIPELINE-AWARE AC: SWEET SPOT SEARCH")
    print("  Looking for configs where the bottleneck stage avoids Full AC")
    print("="*100)

    # Small ~4B model: less param/optimizer overhead → more room for activations
    search_sweet_spot(
        "Small 4B model on A100-80GB",
        small_model, A100_80GB,
        pp_values=[2, 4, 8],
        dp=4,
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )

    # Medium ~8B model
    search_sweet_spot(
        "Medium 8B model on A100-80GB",
        medium_model, A100_80GB,
        pp_values=[2, 4, 8],
        dp=4,
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )

    # Llama-7B: the original case, search wider
    search_sweet_spot(
        "Llama-7B on A100-80GB (wider search)",
        llama_7b, A100_80GB,
        pp_values=[2, 4, 8],
        dp=4,
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )

    # Try H100 (more headroom)
    search_sweet_spot(
        "Small 4B model on H100-80GB",
        small_model, H100_80GB,
        pp_values=[2, 4, 8],
        dp=4,
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )
