"""Search for sweet-spot configs that fit on 8 H100 GPUs (single node).

With 8 GPUs on one NVLink node, feasible parallelism combos:
  TP=1, PP=8, DP=1  — max pipeline depth, minimal TP
  TP=2, PP=4, DP=1  — moderate pipeline, some TP
  TP=4, PP=2, DP=1  — minimal pipeline, high TP
  TP=1, PP=4, DP=2  — moderate pipeline with some DP
  TP=2, PP=2, DP=2  — balanced

We need PP >= 2 for pipeline-aware AC to matter, and enough memory
pressure that stage 0's stash causes trouble.
"""

from simulator.config import (
    H100_80GB,
    A100_80GB,
    ParallelismConfig,
    ModelConfig,
    ActivationFunction,
    llama_7b,
    llama_13b,
    llama_70b,
    llama3_70b,
    qwen3_8b,
    gpt_neox_20b,
)
from simulator.environment import (
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
)


def _gb(x: float) -> str:
    return f"{x / 1024**3:.1f}"


def test_config(name, model_fn, gpu, par, seq_lens, mbs_values, source=""):
    """Test a specific (model, gpu, parallelism) config."""
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"  GPU: {gpu.name}, TP={par.tp_size}, PP={par.pp_size}, DP={par.dp_size}")
    print(f"  Total GPUs: {par.tp_size * par.pp_size * par.dp_size}")
    if source:
        print(f"  {source}")
    print(f"{'='*100}")

    for seq in seq_lens:
        for mbs in mbs_values:
            cfg = model_fn(seq_len=seq, micro_batch_size=mbs)
            try:
                pr_full = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="Full AC")
                pr_no = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="No AC")
                pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)
            except Exception as e:
                print(f"  [seq={seq:>6}, mbs={mbs}] ERROR: {e}")
                continue

            strategies = [sr.strategy_name for sr in pr_aware.stages]
            unique = set(strategies)
            bn = pr_aware.bottleneck

            gain = 0
            if pr_full.all_fit and pr_full.overall_step_latency_s > 0:
                gain = (pr_full.overall_step_latency_s / pr_aware.overall_step_latency_s - 1) * 100

            # Mark sweet spots
            is_sweet = len(unique) >= 2 and bn.strategy_name != "Full AC" and pr_aware.all_fit
            marker = " <<<< SWEET SPOT" if is_sweet else ""

            no_ac_status = "fits" if pr_no.all_fit else "OOM"
            aware_status = "fits" if pr_aware.all_fit else "OOM"

            print(f"  [seq={seq:>6}, mbs={mbs}] "
                  f"NoAC={no_ac_status:<4} Aware={aware_status:<4} "
                  f"gain={gain:>+5.1f}% "
                  f"bn={bn.strategy_name:<22} "
                  f"strategies={strategies}{marker}")

            if is_sweet:
                print(f"    Stage 0: stash={bn.num_stashed_microbatches}, "
                      f"peak={_gb(bn.sim.total_peak_memory_bytes)}GB / "
                      f"{_gb(bn.sim.hbm_capacity_bytes)}GB")


if __name__ == "__main__":
    print("=" * 100)
    print("  8-GPU SWEET-SPOT SEARCH (single node: 8x H100-80GB)")
    print("=" * 100)

    # ── TP=1, PP=8 configs (most pipeline stages) ──────────────────────────

    test_config(
        "Llama-7B: TP=1, PP=8, DP=1",
        llama_7b, H100_80GB,
        par=ParallelismConfig(tp_size=1, pp_size=8, dp_size=1),
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )

    test_config(
        "Qwen3-8B: TP=1, PP=8, DP=1",
        qwen3_8b, H100_80GB,
        par=ParallelismConfig(tp_size=1, pp_size=8, dp_size=1),
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4, 8],
    )

    test_config(
        "Llama-13B: TP=1, PP=8, DP=1",
        llama_13b, H100_80GB,
        par=ParallelismConfig(tp_size=1, pp_size=8, dp_size=1),
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2, 4],
    )

    # ── TP=2, PP=4 configs (more realistic TP) ─────────────────────────────

    test_config(
        "Llama-7B: TP=2, PP=4, DP=1",
        llama_7b, H100_80GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=1),
        seq_lens=[2048, 4096, 8192, 16384],
        mbs_values=[1, 2, 4, 8],
    )

    test_config(
        "Qwen3-8B: TP=2, PP=4, DP=1",
        qwen3_8b, H100_80GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=1),
        seq_lens=[2048, 4096, 8192, 16384],
        mbs_values=[1, 2, 4, 8],
    )

    test_config(
        "Llama-13B: TP=2, PP=4, DP=1",
        llama_13b, H100_80GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=1),
        seq_lens=[2048, 4096, 8192, 16384],
        mbs_values=[1, 2, 4, 8],
    )

    # ── TP=4, PP=2 configs (minimal pipeline) ──────────────────────────────

    test_config(
        "Llama-13B: TP=4, PP=2, DP=1",
        llama_13b, H100_80GB,
        par=ParallelismConfig(tp_size=4, pp_size=2, dp_size=1),
        seq_lens=[4096, 8192, 16384],
        mbs_values=[1, 2, 4, 8],
    )

    test_config(
        "Llama-3 70B: TP=4, PP=2, DP=1 (tight fit)",
        llama3_70b, H100_80GB,
        par=ParallelismConfig(tp_size=4, pp_size=2, dp_size=1),
        seq_lens=[2048, 4096, 8192],
        mbs_values=[1, 2],
    )

    # ── GPT-NeoX-20B (no FA, Korthikanti applies) ──────────────────────────

    test_config(
        "GPT-NeoX-20B: TP=2, PP=4, DP=1 (no FA)",
        gpt_neox_20b, H100_80GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=1),
        seq_lens=[2048, 4096],
        mbs_values=[2, 4, 8],
    )

    print("\n\n" + "=" * 100)
    print("  SEARCH COMPLETE")
    print("=" * 100)
