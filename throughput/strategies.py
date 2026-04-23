"""Per-stage AC strategy selection for the pipeline throughput runner.

Kept free of torch imports so it can be unit-tested without GPU deps.
The runner (`throughput/run_pipeline.py`) consumes the list returned by
`stage_strategies()` and applies the corresponding wrapper to each rank.
"""
from __future__ import annotations


VALID_MODES = (
    "no-ac",
    "full-ac",
    "pipeline-aware",
    "offload-linear2",
    "offload-all-mlp",
)
VALID_STAGE_STRATEGIES = ("no-ac", "full-ac", "offload-linear2", "offload-all-mlp")

# Each runner-side per-stage strategy maps to a simulator strategy name
# declared in `simulator.environment.STRATEGY_LEVELS`.
RUNNER_TO_SIM_STRATEGY = {
    "no-ac": "No AC",
    "full-ac": "Full AC",
    "offload-linear2": "Offload linear2",
    "offload-all-mlp": "Offload all MLP",
}


def stage_strategies(ac_mode: str, pp_size: int) -> list[str]:
    """Return a per-stage strategy list of length `pp_size`.

    For uniform modes every stage gets the same strategy. For `pipeline-aware`,
    the first `pp_size // 2` stages use full-ac and the rest use no-ac —
    mirrors the simulator's least-aggressive-that-fits heuristic.
    """
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")

    if ac_mode in ("no-ac", "full-ac", "offload-linear2", "offload-all-mlp"):
        return [ac_mode] * pp_size
    if ac_mode == "pipeline-aware":
        half = pp_size // 2
        return ["full-ac" if s < half else "no-ac" for s in range(pp_size)]
    raise ValueError(
        f"unknown ac mode: {ac_mode!r} (expected one of {VALID_MODES})"
    )
