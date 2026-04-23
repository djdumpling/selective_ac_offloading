"""Unit tests for the per-stage AC strategy selector used by the pipeline
throughput runner. These tests must stay importable without torch so they
run in the simulator-only CI path."""

import pytest

from throughput.strategies import (
    RUNNER_TO_SIM_STRATEGY,
    SIM_TO_RUNNER_STRATEGY,
    VALID_MODES,
    VALID_STAGE_STRATEGIES,
    interleaved_chunk_layer_spans,
    parse_per_stage_override,
    pipeline_aware_stage_strategies,
    stage_strategies,
)


class TestStageStrategies:
    def test_no_ac_uniform(self):
        assert stage_strategies("no-ac", 4) == ["no-ac"] * 4

    def test_full_ac_uniform(self):
        assert stage_strategies("full-ac", 4) == ["full-ac"] * 4

    def test_pipeline_aware_pp2(self):
        # First half gets full-ac, rest gets no-ac. pp=2 → one of each.
        assert stage_strategies("pipeline-aware", 2) == ["full-ac", "no-ac"]

    def test_pipeline_aware_pp4(self):
        assert stage_strategies("pipeline-aware", 4) == [
            "full-ac", "full-ac", "no-ac", "no-ac"
        ]

    def test_pipeline_aware_pp8(self):
        result = stage_strategies("pipeline-aware", 8)
        assert result[:4] == ["full-ac"] * 4
        assert result[4:] == ["no-ac"] * 4

    def test_pipeline_aware_odd_pp_rounds_down(self):
        # pp=3: half=1 → full-ac on stage 0, no-ac on 1,2.
        # Matches 1F1B's (pp-1-p) stash shape where late stages need less memory.
        assert stage_strategies("pipeline-aware", 3) == ["full-ac", "no-ac", "no-ac"]

    def test_length_matches_pp_size(self):
        for pp in (1, 2, 3, 4, 8, 16):
            for mode in VALID_MODES:
                assert len(stage_strategies(mode, pp)) == pp

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown ac mode"):
            stage_strategies("selective-ac", 4)

    def test_zero_pp_raises(self):
        with pytest.raises(ValueError):
            stage_strategies("no-ac", 0)

    def test_aware_is_monotonic_non_increasing_aggressiveness(self):
        """Early stages must be at least as aggressive as later stages — the
        core pipeline-aware invariant that mirrors STRATEGY_LEVELS."""
        order = {"full-ac": 1, "no-ac": 0}
        for pp in (2, 4, 6, 8):
            strategies = stage_strategies("pipeline-aware", pp)
            levels = [order[s] for s in strategies]
            assert all(a >= b for a, b in zip(levels, levels[1:])), (
                f"non-monotonic at pp={pp}: {strategies}"
            )

    def test_offload_linear2_uniform(self):
        assert stage_strategies("offload-linear2", 4) == ["offload-linear2"] * 4

    def test_offload_all_mlp_uniform(self):
        assert stage_strategies("offload-all-mlp", 4) == ["offload-all-mlp"] * 4


class TestRunnerToSimMapping:
    """Every runner mode that maps 1:1 to a simulator strategy must be present
    in `RUNNER_TO_SIM_STRATEGY`, and every target name must actually exist in
    the simulator's `STRATEGY_LEVELS`."""

    def test_all_uniform_modes_have_mapping(self):
        for mode in VALID_STAGE_STRATEGIES:
            assert mode in RUNNER_TO_SIM_STRATEGY, f"missing mapping for {mode}"

    def test_mapped_names_exist_in_simulator(self):
        from simulator.environment import STRATEGY_LEVELS
        sim_names = {name for name, _ in STRATEGY_LEVELS}
        for runner_name, sim_name in RUNNER_TO_SIM_STRATEGY.items():
            assert sim_name in sim_names, (
                f"runner mode {runner_name} → {sim_name!r} "
                f"not in simulator STRATEGY_LEVELS {sim_names}"
            )

    def test_sim_to_runner_inverse(self):
        """SIM_TO_RUNNER_STRATEGY must be the exact inverse of RUNNER_TO_SIM_STRATEGY."""
        for runner_name, sim_name in RUNNER_TO_SIM_STRATEGY.items():
            assert SIM_TO_RUNNER_STRATEGY[sim_name] == runner_name

    def test_unsupported_sim_strategies_absent(self):
        """FA-Selective and Korthikanti don't have runner implementations — their
        absence from SIM_TO_RUNNER_STRATEGY is what triggers the clear error in
        pipeline_aware_stage_strategies()."""
        assert "FA-Selective" not in SIM_TO_RUNNER_STRATEGY
        assert "Korthikanti Selective" not in SIM_TO_RUNNER_STRATEGY


class TestPipelineAwareStageStrategies:
    """Simulator-driven heterogeneous strategy assignment. Covers the path that
    the throughput runner actually uses for --ac pipeline-aware."""

    def test_uniform_no_ac_when_plenty_of_hbm(self):
        """At small seq_len, No AC fits every stage — pick should be uniform."""
        from simulator.config import H200_141GB, ParallelismConfig, llama_7b
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        strategies = pipeline_aware_stage_strategies(
            cfg, H200_141GB, ParallelismConfig(pp_size=4),
            num_microbatches=8,
        )
        assert strategies == ["no-ac"] * 4

    def test_heterogeneous_pick_at_seq32k_pp4(self):
        """The validated config: PP=4 Llama-7B seq=32K. The simulator should
        pick [offload-all-mlp, offload-all-mlp, no-ac, no-ac] — early stages
        carry 1F1B stash pressure (PP-1-p = 3, 2 microbatches) and need
        offload; late stages (stash=0,1) fit No AC."""
        from simulator.config import H200_141GB, ParallelismConfig, llama_7b
        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        strategies = pipeline_aware_stage_strategies(
            cfg, H200_141GB, ParallelismConfig(pp_size=4),
            num_microbatches=8,
        )
        assert strategies == [
            "offload-all-mlp", "offload-all-mlp", "no-ac", "no-ac",
        ]

    def test_length_matches_pp_size(self):
        from simulator.config import H200_141GB, ParallelismConfig, llama_7b
        for pp in (2, 4, 8):
            cfg = llama_7b(seq_len=16384, micro_batch_size=1)
            strategies = pipeline_aware_stage_strategies(
                cfg, H200_141GB, ParallelismConfig(pp_size=pp),
                num_microbatches=8,
            )
            assert len(strategies) == pp

    def test_monotonic_aggressiveness(self):
        """Early stages must be at least as aggressive as later stages — the
        simulator picks this way by construction (more stash = need to fit
        tighter budget = more aggressive strategy)."""
        from simulator.config import H200_141GB, ParallelismConfig, llama_7b
        # Ordered least→most aggressive, matching STRATEGY_LEVELS order.
        order = {
            "no-ac": 0,
            "offload-linear2": 1,
            "offload-all-mlp": 2,
            "full-ac": 3,
        }
        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        strategies = pipeline_aware_stage_strategies(
            cfg, H200_141GB, ParallelismConfig(pp_size=4),
            num_microbatches=8,
        )
        levels = [order[s] for s in strategies]
        assert all(a >= b for a, b in zip(levels, levels[1:])), (
            f"non-monotonic: {strategies}"
        )

    def test_unsupported_strategy_raises(self):
        """If the simulator picks FA-Selective or Korthikanti, the runner can't
        execute it — we must raise a clear error rather than silently degrading
        to a different strategy."""
        import unittest.mock as mock
        from types import SimpleNamespace

        from simulator.config import H200_141GB, ParallelismConfig, llama_7b
        fake_pr = SimpleNamespace(stages=[
            SimpleNamespace(strategy_name="No AC"),
            SimpleNamespace(strategy_name="FA-Selective"),  # unsupported
            SimpleNamespace(strategy_name="No AC"),
            SimpleNamespace(strategy_name="No AC"),
        ])
        with mock.patch(
            "simulator.environment.simulate_pipeline_aware_ac",
            return_value=fake_pr,
        ):
            cfg = llama_7b(seq_len=2048, micro_batch_size=1)
            with pytest.raises(ValueError, match="FA-Selective"):
                pipeline_aware_stage_strategies(
                    cfg, H200_141GB, ParallelismConfig(pp_size=4),
                    num_microbatches=8,
                )


class TestInterleavedChunkLayerSpans:
    """Virtual-stage assignment for interleaved 1F1B. Each rank owns
    `num_chunks` non-contiguous chunks at positions rank, rank+pp, rank+2pp, ..."""

    def test_round_robin_assignment_pp4_chunks2(self):
        """pp=4, num_chunks=2, 32 layers → 8 virtual stages, 4 layers each.
        Rank 0: virtual stages 0, 4 → [0-4), [16-20).
        Rank 1: virtual stages 1, 5 → [4-8), [20-24)."""
        assert interleaved_chunk_layer_spans(32, 4, 2, 0) == [
            (0, 0, 4),
            (4, 16, 20),
        ]
        assert interleaved_chunk_layer_spans(32, 4, 2, 1) == [
            (1, 4, 8),
            (5, 20, 24),
        ]
        assert interleaved_chunk_layer_spans(32, 4, 2, 3) == [
            (3, 12, 16),
            (7, 28, 32),
        ]

    def test_num_chunks_1_matches_stage_layer_span(self):
        """num_chunks=1 reduces to contiguous assignment (same as 1F1B)."""
        from simulator.environment import _stage_layer_span
        for rank in range(4):
            spans = interleaved_chunk_layer_spans(32, 4, 1, rank)
            assert len(spans) == 1
            virtual, start, end = spans[0]
            assert virtual == rank
            assert (start, end) == _stage_layer_span(32, 4, rank)

    def test_uneven_division_raises(self):
        """32 layers can't split evenly into 3 pp × 2 chunks = 6 virtual stages."""
        with pytest.raises(ValueError, match="divisible"):
            interleaved_chunk_layer_spans(32, 3, 2, 0)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            interleaved_chunk_layer_spans(32, 0, 2, 0)
        with pytest.raises(ValueError):
            interleaved_chunk_layer_spans(32, 4, 0, 0)

    def test_each_rank_gets_num_chunks_spans(self):
        for num_chunks in (1, 2, 4):
            for rank in range(4):
                spans = interleaved_chunk_layer_spans(32, 4, num_chunks, rank)
                assert len(spans) == num_chunks

    def test_virtual_stage_ordering_within_rank(self):
        """Chunks on a single rank appear in increasing virtual-stage order,
        which matches the order torch.distributed.pipelining expects."""
        spans = interleaved_chunk_layer_spans(32, 4, 4, 0)
        virtuals = [v for v, _, _ in spans]
        assert virtuals == sorted(virtuals)


class TestParsePerStageOverride:
    """--per-stage is the escape hatch for manual experiments: specify an
    exact per-stage strategy list from the command line."""

    def test_basic_parse(self):
        assert parse_per_stage_override(
            "offload-all-mlp,offload-all-mlp,no-ac,no-ac", 4,
        ) == ["offload-all-mlp", "offload-all-mlp", "no-ac", "no-ac"]

    def test_strips_whitespace(self):
        assert parse_per_stage_override(
            " no-ac , full-ac , no-ac , no-ac ", 4,
        ) == ["no-ac", "full-ac", "no-ac", "no-ac"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="entries but --pp is"):
            parse_per_stage_override("no-ac,no-ac,no-ac", 4)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="not in"):
            parse_per_stage_override("no-ac,selective,no-ac,no-ac", 4)

    def test_pipeline_aware_not_valid_per_stage_entry(self):
        """pipeline-aware is a picking mode, not a per-stage strategy."""
        with pytest.raises(ValueError, match="not in"):
            parse_per_stage_override("pipeline-aware,no-ac,no-ac,no-ac", 4)

    def test_all_valid_stage_strategies_accepted(self):
        for s in VALID_STAGE_STRATEGIES:
            result = parse_per_stage_override(",".join([s] * 3), 3)
            assert result == [s] * 3
