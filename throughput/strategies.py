"""Per-stage AC strategy selection for the pipeline throughput runner.

Kept free of torch imports so it can be unit-tested without GPU deps.
The runner (`throughput/run_pipeline.py`) consumes the list returned by
`stage_strategies()` and applies the corresponding wrapper to each rank.
"""
from __future__ import annotations


VALID_MODES = ("no-ac", "full-ac", "pipeline-aware")
VALID_STAGE_STRATEGIES = ("no-ac", "full-ac")


def stage_strategies(ac_mode: str, pp_size: int) -> list[str]:
    """Return a per-stage strategy list of length `pp_size`.

    Mirrors `simulator.environment.STRATEGY_LEVELS` ordering: high-stash early
    stages get the aggressive strategy (full-ac), lax late stages get no-ac.
    For `pipeline-aware`, the first `pp_size // 2` stages use full-ac and the
    rest use no-ac — the same shape the simulator's least-aggressive-that-fits
    heuristic picks on sweet-spot configs.
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")

    if ac_mode == "no-ac":
        return ["no-ac"] * pp_size
    if ac_mode == "full-ac":
        return ["full-ac"] * pp_size
    if ac_mode == "pipeline-aware":
        half = pp_size // 2
        return ["full-ac" if s < half else "no-ac" for s in range(pp_size)]
    raise ValueError(
        f"unknown ac mode: {ac_mode!r} (expected one of {VALID_MODES})"
    )
