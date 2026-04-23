"""Unit tests for the per-stage AC strategy selector used by the pipeline
throughput runner. These tests must stay importable without torch so they
run in the simulator-only CI path."""

import pytest

from throughput.strategies import VALID_MODES, stage_strategies


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
