"""Sweet-spot search using REALISTIC model + parallelism configurations.

Each (model, GPU, parallelism) config is taken from published training setups.
We sweep over seq_len and micro_batch_size around the published values to find
configs where pipeline-aware AC improves throughput over uniform Full AC.

Sources for each config are documented in simulator/config.py.
"""

from simulator.config import (
    A100_40GB,
    A100_80GB,
    H100_80GB,
    ParallelismConfig,
    llama3_70b,
    llama3_405b,
    gpt_neox_20b,
    bloom_176b,
    falcon_180b,
    llama_70b,
)
from simulator.environment import (
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
    print_pipeline_result,
)


def _gb(x: float) -> str:
    return f"{x / 1024**3:.1f}"


def search_sweet_spot(name, model_fn, gpu, par, seq_lens, mbs_values, source=""):
    """Search over (seq_len, mbs) for a fixed (model, gpu, parallelism) config."""
    print(f"\n{'='*100}")
    print(f"  {name}")
    print(f"  GPU: {gpu.name}, TP={par.tp_size}, PP={par.pp_size}, DP={par.dp_size}")
    if source:
        print(f"  Source: {source}")
    print(f"{'='*100}")

    found_any = False

    for seq in seq_lens:
        for mbs in mbs_values:
            cfg = model_fn(seq_len=seq, micro_batch_size=mbs)
            try:
                pr_uniform_full = simulate_pipeline_uniform_ac(
                    cfg, gpu, par, strategy_name="Full AC")
                pr_uniform_no = simulate_pipeline_uniform_ac(
                    cfg, gpu, par, strategy_name="No AC")
                pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)
            except Exception as e:
                print(f"  [seq={seq}, mbs={mbs}] ERROR: {e}")
                continue

            if not pr_aware.all_fit:
                # Even pipeline-aware can't fit — skip
                continue

            aware_strategies = [sr.strategy_name for sr in pr_aware.stages]
            bottleneck_strategy = pr_aware.bottleneck.strategy_name
            unique_strategies = set(aware_strategies)

            # Compute throughput gain vs uniform Full AC
            if pr_uniform_full.all_fit and pr_uniform_full.overall_step_latency_s > 0:
                gain_vs_full = (
                    pr_uniform_full.overall_step_latency_s / pr_aware.overall_step_latency_s - 1
                ) * 100
            else:
                gain_vs_full = float('inf')  # Full AC doesn't fit but aware does

            # Check if uniform No AC fits everywhere
            no_ac_fits = pr_uniform_no.all_fit

            # Classify the result
            is_differentiated = len(unique_strategies) >= 2
            bottleneck_avoids_full = bottleneck_strategy != "Full AC"

            # Report interesting cases:
            # 1. Differentiated strategies with bottleneck avoiding Full AC (best case)
            # 2. Pipeline-aware fits but uniform No AC doesn't (feasibility gain)
            # 3. Any case with non-trivial strategy differentiation
            interesting = (
                (is_differentiated and bottleneck_avoids_full) or
                (pr_aware.all_fit and not no_ac_fits and is_differentiated)
            )

            if interesting:
                found_any = True
                print(f"\n  --- seq={seq}, mbs={mbs} ---")

                # Stage memory details
                bn = pr_aware.bottleneck
                print(f"  Bottleneck: stage {pr_aware.bottleneck_stage} "
                      f"({bottleneck_strategy}), "
                      f"stash={bn.num_stashed_microbatches} microbatches, "
                      f"peak={_gb(bn.sim.total_peak_memory_bytes)}GB / "
                      f"{_gb(bn.sim.hbm_capacity_bytes)}GB")

                print(f"  Per-stage strategies: {aware_strategies}")

                if pr_uniform_full.all_fit:
                    print(f"  Uniform Full AC:  {pr_uniform_full.overall_step_latency_s*1000:.1f} ms "
                          f"(overhead: {pr_uniform_full.total_recompute_overhead_pct:.1f}%)")
                else:
                    print(f"  Uniform Full AC:  DOES NOT FIT")

                print(f"  Pipeline-Aware:   {pr_aware.overall_step_latency_s*1000:.1f} ms "
                      f"(overhead: {pr_aware.total_recompute_overhead_pct:.1f}%)")

                if gain_vs_full != float('inf'):
                    print(f"  THROUGHPUT GAIN vs Full AC: {gain_vs_full:+.1f}%")
                else:
                    print(f"  FEASIBILITY GAIN: Pipeline-aware fits, uniform Full AC does not!")

                if not no_ac_fits:
                    print(f"  Uniform No AC: DOES NOT FIT (confirms memory pressure)")

            # Also report "comfortable" cases to understand the landscape
            elif not found_any and seq == seq_lens[0] and mbs == mbs_values[0]:
                print(f"\n  [seq={seq}, mbs={mbs}] No AC fits={no_ac_fits}, "
                      f"strategies={list(unique_strategies)}, "
                      f"bottleneck={bottleneck_strategy}")

    if not found_any:
        print("  No sweet-spot configs found in search range.")
        # Show why — report the first config's details
        cfg = model_fn(seq_len=seq_lens[0], micro_batch_size=mbs_values[0])
        pr_no = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="No AC")
        pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)
        strategies = [sr.strategy_name for sr in pr_aware.stages]
        print(f"  Diagnostic (seq={seq_lens[0]}, mbs={mbs_values[0]}):")
        print(f"    No AC fits: {pr_no.all_fit}")
        print(f"    Pipeline-aware strategies: {strategies}")
        for i, sr in enumerate(pr_aware.stages):
            print(f"    Stage {i}: {sr.strategy_name}, "
                  f"stash={sr.num_stashed_microbatches}, "
                  f"peak={_gb(sr.sim.total_peak_memory_bytes)}GB / "
                  f"{_gb(sr.sim.hbm_capacity_bytes)}GB")


if __name__ == "__main__":
    print("=" * 100)
    print("  REALISTIC SWEET-SPOT SEARCH")
    print("  Using published training configs from Llama-3, GPT-NeoX, BLOOM, Falcon")
    print("=" * 100)

    # ── 1. Llama-3 70B: THE most promising case ────────────────────────────
    # Meta's actual training config: TP=8, PP=4, H100-80GB
    # GQA (8 KV heads) + SwiGLU + RoPE + FA
    # Pretraining seq=8192, extended to 128K
    search_sweet_spot(
        "Llama-3 70B — Meta's training config (pretraining)",
        llama3_70b, H100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=4, dp_size=64),
        seq_lens=[4096, 8192, 16384, 32768],
        mbs_values=[1, 2, 4],
        source="arxiv 2407.21783 — TP=8, PP=4, H100",
    )

    # Llama-3 70B on A100 (common for fine-tuning / smaller clusters)
    search_sweet_spot(
        "Llama-3 70B — A100-80GB (fine-tuning / smaller cluster)",
        llama3_70b, A100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=4, dp_size=8),
        seq_lens=[4096, 8192, 16384, 32768],
        mbs_values=[1, 2, 4],
        source="Common fine-tuning setup — TP=8, PP=4, A100-80GB",
    )

    # Llama-3 70B long-context on H100 (the context extension phase)
    search_sweet_spot(
        "Llama-3 70B — Long-context extension (H100)",
        llama3_70b, H100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=4, dp_size=16),
        seq_lens=[32768, 65536, 131072],
        mbs_values=[1],
        source="Llama-3.1 long-context extension phase",
    )

    # ── 2. Llama-2 70B with PP (alternative config) ────────────────────────
    # Original Llama-2 70B was trained with FSDP, but practitioners often
    # deploy with PP when crossing node boundaries
    search_sweet_spot(
        "Llama-2 70B — Multi-node with PP=4",
        llama_70b, A100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=4, dp_size=8),
        seq_lens=[4096, 8192, 16384],
        mbs_values=[1, 2, 4],
        source="Practitioner config — TP=8, PP=4, A100-80GB",
    )

    # ── 3. Llama-3.1 405B ──────────────────────────────────────────────────
    # Meta's training config: TP=8, PP=16, DP=128, H100-80GB
    search_sweet_spot(
        "Llama-3.1 405B — Meta's training config",
        llama3_405b, H100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=16, dp_size=128),
        seq_lens=[4096, 8192],
        mbs_values=[1, 2],
        source="arxiv 2407.21783 — TP=8, PP=16, 16384 H100s",
    )

    # 405B with PP=8 (smaller cluster / different partitioning)
    search_sweet_spot(
        "Llama-3.1 405B — PP=8 variant",
        llama3_405b, H100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=8, dp_size=64),
        seq_lens=[4096, 8192],
        mbs_values=[1, 2],
        source="Alternative config — TP=8, PP=8, H100",
    )

    # ── 4. GPT-NeoX-20B ────────────────────────────────────────────────────
    # EleutherAI's actual config: TP=2, PP=4, DP=12, A100-40GB
    # No FlashAttention — Korthikanti selective applies here
    search_sweet_spot(
        "GPT-NeoX-20B — EleutherAI's training config",
        gpt_neox_20b, A100_40GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=12),
        seq_lens=[2048, 4096],
        mbs_values=[2, 4, 8],
        source="arxiv 2204.06745 — TP=2, PP=4, 96 A100-40GB",
    )

    # GPT-NeoX-20B on A100-80GB (more headroom)
    search_sweet_spot(
        "GPT-NeoX-20B — A100-80GB (upgraded hardware)",
        gpt_neox_20b, A100_80GB,
        par=ParallelismConfig(tp_size=2, pp_size=4, dp_size=12),
        seq_lens=[2048, 4096, 8192],
        mbs_values=[2, 4, 8],
        source="Same parallelism, upgraded to A100-80GB",
    )

    # ── 5. BLOOM-176B ──────────────────────────────────────────────────────
    # BigScience's actual config: TP=4, PP=12, DP=8, A100-80GB
    # No FlashAttention, ALiBi, MHA
    search_sweet_spot(
        "BLOOM-176B — BigScience training config",
        bloom_176b, A100_80GB,
        par=ParallelismConfig(tp_size=4, pp_size=12, dp_size=8),
        seq_lens=[2048],
        mbs_values=[1, 2, 4],
        source="arxiv 2211.05100 — TP=4, PP=12, 384 A100-80GB",
    )

    # ── 6. Falcon-180B ─────────────────────────────────────────────────────
    # TII's config: TP=8, PP=8, DP=64, A100-80GB
    # Note: parallel attn+MLP is approximated by our sequential model
    search_sweet_spot(
        "Falcon-180B — TII training config (approximate)",
        falcon_180b, A100_80GB,
        par=ParallelismConfig(tp_size=8, pp_size=8, dp_size=64),
        seq_lens=[2048, 4096],
        mbs_values=[1, 2, 4],
        source="HuggingFace — TP=8, PP=8, 4096 A100-80GB (parallel attn+MLP approximated)",
    )

    print("\n\n" + "=" * 100)
    print("  SEARCH COMPLETE")
    print("=" * 100)
