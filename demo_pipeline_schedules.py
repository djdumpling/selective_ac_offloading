"""Demo: Pipeline-aware AC across different pipeline schedules.

Compares 1F1B, 1F1B Interleaved, ZB-H1, ZB-H2, ZB-V, and DualPipe
to show how stash profiles and bubble fractions interact with
pipeline-position-aware activation checkpointing.
"""

from simulator.config import (
    A100_80GB,
    H100_80GB,
    ParallelismConfig,
    llama_7b,
    llama_70b,
    gpt3_175b,
)
from simulator.environment import (
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
    print_pipeline_result,
)
from simulator.pipeline_schedules import PipelineSchedule, get_schedule_profile


def _gb(x: float) -> str:
    return f"{x / 1024**3:.2f}"


def compare_schedules(name, cfg, gpu, par):
    print(f"\n{'=' * 110}")
    print(f"  {name} on {gpu.name}")
    print(f"  tp={par.tp_size}, pp={par.pp_size}, dp={par.dp_size}, "
          f"seq={cfg.seq_len}, mbs={cfg.micro_batch_size}")
    print(f"{'=' * 110}")

    schedules = [
        PipelineSchedule.ONE_F_ONE_B,
        PipelineSchedule.ONE_F_ONE_B_INTERLEAVED,
        PipelineSchedule.ZB_H1,
        PipelineSchedule.ZB_H2,
        PipelineSchedule.ZB_V,
        PipelineSchedule.DUALPIPE,
    ]

    # Collect results for summary table
    summary_rows = []

    for sched in schedules:
        profile = get_schedule_profile(sched, cfg, par, num_microbatches=16)

        pr_uniform_full = simulate_pipeline_uniform_ac(
            cfg, gpu, par, strategy_name="Full AC", schedule=sched)
        pr_uniform_fa = simulate_pipeline_uniform_ac(
            cfg, gpu, par, strategy_name="FA-Selective", schedule=sched)
        pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par, schedule=sched)

        strategies_used = [sr.strategy_name for sr in pr_aware.stages]
        unique_strategies = set(strategies_used)
        is_differentiated = len(unique_strategies) > 1

        summary_rows.append({
            "schedule": sched.value,
            "stash_first": profile.stash_counts[0],
            "stash_last": profile.stash_counts[-1],
            "symmetric": profile.stash_counts[0] == profile.stash_counts[-1],
            "bubble": profile.bubble_fraction,
            "extra_mem_gb": profile.extra_memory_per_stage / 1024**3,
            "uniform_full_fit": pr_uniform_full.all_fit,
            "uniform_full_step": pr_uniform_full.overall_step_latency_s,
            "uniform_full_ovhd": pr_uniform_full.total_recompute_overhead_pct,
            "uniform_fa_fit": pr_uniform_fa.all_fit,
            "uniform_fa_step": pr_uniform_fa.overall_step_latency_s,
            "aware_fit": pr_aware.all_fit,
            "aware_step": pr_aware.overall_step_latency_s,
            "aware_ovhd": pr_aware.total_recompute_overhead_pct,
            "strategies": strategies_used,
            "differentiated": is_differentiated,
        })

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n  {'Schedule':<20} {'Stash':>10} {'Bubble':>7} {'Extra':>8}"
          f"  {'Uniform Full':>16} {'Uniform FA-Sel':>16} {'Pipeline-Aware':>16} {'Strategies'}")
    print(f"  {'':20} {'(first/last)':>10} {'':>7} {'Mem':>8}"
          f"  {'Step(ms) Fit':>16} {'Step(ms) Fit':>16} {'Step(ms) Fit':>16}")
    print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*8}"
          f"  {'-'*16} {'-'*16} {'-'*16} {'-'*30}")

    for r in summary_rows:
        stash_str = f"{r['stash_first']}/{r['stash_last']}"
        if r["symmetric"]:
            stash_str += " (sym)"

        uf_fit = "YES" if r["uniform_full_fit"] else "NO"
        ufa_fit = "YES" if r["uniform_fa_fit"] else "NO"
        a_fit = "YES" if r["aware_fit"] else "NO"

        extra_str = f"{r['extra_mem_gb']:.1f}GB" if r["extra_mem_gb"] > 0.01 else "—"

        # Compact strategy summary
        strats = r["strategies"]
        if not r["differentiated"]:
            strat_str = f"all {strats[0]}"
        else:
            counts = {}
            for s in strats:
                counts[s] = counts.get(s, 0) + 1
            strat_str = ", ".join(f"{v}×{k}" for k, v in counts.items())

        print(
            f"  {r['schedule']:<20} {stash_str:>10} {r['bubble']:>6.1%} {extra_str:>8}"
            f"  {r['uniform_full_step']*1000:>9.1f} {uf_fit:>4}"
            f"  {r['uniform_fa_step']*1000:>9.1f} {ufa_fit:>4}"
            f"  {r['aware_step']*1000:>9.1f} {a_fit:>4}"
            f"  {strat_str}"
        )

    # ── Detailed per-stage view for the most interesting case ────────────
    # Find schedule where pipeline-aware uses different strategies
    interesting = [r for r in summary_rows if r["differentiated"]]
    if interesting:
        r = interesting[0]
        sched = PipelineSchedule(r["schedule"])
        print(f"\n  Detailed view: {r['schedule']} (pipeline-aware uses mixed strategies)")
        pr = simulate_pipeline_aware_ac(cfg, gpu, par, schedule=sched)
        print_pipeline_result(pr)
    else:
        print(f"\n  No schedule produced differentiated strategies for this config.")
        print(f"  (Memory is either too tight or too comfortable for all stages.)")


if __name__ == "__main__":
    print("=" * 110)
    print("  PIPELINE SCHEDULE COMPARISON")
    print("  How do different pipeline schedules interact with pipeline-aware AC?")
    print("=" * 110)

    # Case 1: Memory-constrained — large batch Llama-7B
    # This is where stash pressure creates real differential
    compare_schedules(
        "Llama-7B (memory-constrained, large batch)",
        llama_7b(seq_len=4096, micro_batch_size=4),
        A100_80GB,
        ParallelismConfig(pp_size=4, dp_size=4),
    )

    # Case 2: Llama-70B with TP+PP
    compare_schedules(
        "Llama-70B (TP=8, PP=4)",
        llama_70b(seq_len=4096, micro_batch_size=1),
        H100_80GB,
        ParallelismConfig(tp_size=8, pp_size=4, dp_size=4),
    )

    # Case 3: GPT-3 175B — the Korthikanti reference case
    compare_schedules(
        "GPT-3 175B (Korthikanti config)",
        gpt3_175b(seq_len=2048, micro_batch_size=1),
        A100_80GB,
        ParallelismConfig(tp_size=8, pp_size=8),
    )
