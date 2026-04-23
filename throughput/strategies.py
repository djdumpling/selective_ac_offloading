"""Per-stage AC strategy selection for the pipeline throughput runner.

Kept free of torch imports so it can be unit-tested without GPU deps.
The runner (`throughput/run_pipeline.py`) consumes the list returned by
`stage_strategies()` or `pipeline_aware_stage_strategies()` and applies
the corresponding wrapper to each rank.
"""
from __future__ import annotations

from typing import TYPE_CHECKING


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

# Inverse mapping. Simulator strategies absent from this dict have no runner
# implementation — we'd need to wire Megatron-style selective checkpointing
# into LlamaDecoderLayer to support FA-Selective / Korthikanti Selective.
SIM_TO_RUNNER_STRATEGY = {v: k for k, v in RUNNER_TO_SIM_STRATEGY.items()}


if TYPE_CHECKING:  # pragma: no cover - typing only, avoid importing torch
    from simulator.config import GPUConfig, ModelConfig, ParallelismConfig
    from simulator.pipeline_schedules import PipelineSchedule


def stage_strategies(ac_mode: str, pp_size: int) -> list[str]:
    """Return a per-stage strategy list of length `pp_size` for uniform modes.

    For uniform modes every stage gets the same strategy. `pipeline-aware` is
    handled separately by `pipeline_aware_stage_strategies()` which needs the
    model/GPU config to consult the simulator; the static half-full/half-none
    fallback below is only used when the dynamic picker is unavailable.
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


def interleaved_chunk_layer_spans(
    num_layers: int, pp_size: int, num_chunks: int, rank: int,
) -> list[tuple[int, int, int]]:
    """Return (virtual_stage_index, layer_start, layer_end) for each chunk this rank owns.

    Interleaved 1F1B splits the model into `pp_size * num_chunks` virtual stages
    and assigns them round-robin to ranks: virtual stage v goes to rank v % pp_size,
    at position v // pp_size among that rank's chunks. E.g., pp=4 num_chunks=2,
    rank 0 owns virtual stages 0 and 4. With num_layers=32, that's layers [0-4)
    and [16-20).

    Raises ValueError if layers don't divide evenly.
    """
    if pp_size < 1 or num_chunks < 1:
        raise ValueError(f"pp_size/num_chunks must be >= 1, got {pp_size}/{num_chunks}")
    total_virtual = pp_size * num_chunks
    if num_layers % total_virtual != 0:
        raise ValueError(
            f"num_layers ({num_layers}) must be divisible by pp_size*num_chunks "
            f"({pp_size}*{num_chunks}={total_virtual}) for interleaved scheduling"
        )
    layers_per_chunk = num_layers // total_virtual
    out: list[tuple[int, int, int]] = []
    for chunk_idx in range(num_chunks):
        virtual_stage = rank + chunk_idx * pp_size
        start = virtual_stage * layers_per_chunk
        end = start + layers_per_chunk
        out.append((virtual_stage, start, end))
    return out


def pipeline_aware_stage_strategies(
    cfg: "ModelConfig",
    gpu: "GPUConfig",
    par: "ParallelismConfig",
    schedule: "PipelineSchedule | None" = None,
    num_microbatches: int = 8,
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
) -> list[str]:
    """Ask the simulator which strategy fits each stage, then map to runner names.

    Delegates to `simulator.environment.simulate_pipeline_aware_ac`, which
    walks `STRATEGY_LEVELS` least-aggressive-first per stage and returns the
    first that fits the stage's HBM budget given its 1F1B stash. We convert
    the chosen simulator strategy names to the runner names via
    `SIM_TO_RUNNER_STRATEGY`.

    Raises `ValueError` if the simulator picks a strategy the runner doesn't
    implement (FA-Selective or Korthikanti Selective — those require
    Megatron-style per-module wrapping, not yet wired into `LlamaStage`).
    """
    from simulator.environment import simulate_pipeline_aware_ac  # local: torch-free

    pr = simulate_pipeline_aware_ac(
        cfg, gpu, par,
        schedule=schedule,
        efficiency=efficiency,
        memory_budget_frac=memory_budget_frac,
        num_microbatches=num_microbatches,
    )
    runner_strats: list[str] = []
    unsupported: list[tuple[int, str]] = []
    for stage_idx, stage in enumerate(pr.stages):
        sim_name = stage.strategy_name
        runner_name = SIM_TO_RUNNER_STRATEGY.get(sim_name)
        if runner_name is None:
            unsupported.append((stage_idx, sim_name))
        else:
            runner_strats.append(runner_name)

    if unsupported:
        details = ", ".join(f"stage {i} → {name!r}" for i, name in unsupported)
        raise ValueError(
            f"Simulator picked strategies with no runner implementation: "
            f"{details}. Supported runner strategies: {sorted(SIM_TO_RUNNER_STRATEGY)}. "
            f"Either rerun with --ac=full-ac / --ac=offload-all-mlp as a uniform "
            f"fallback, or implement the missing wrapper in throughput/run_pipeline.py."
        )
    return runner_strats
