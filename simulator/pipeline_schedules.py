"""Pipeline schedule models: stash profiles, bubble fractions, extra memory.

Each pipeline schedule determines:
1. How many microbatch activations each stage must stash (memory pressure)
2. What fraction of time is wasted in pipeline bubbles (throughput)
3. Extra memory overhead (e.g., deferred weight gradients in ZB-H2)

Supported schedules:
- 1F1B:              Standard Megatron-LM schedule (Narayanan et al., 2021)
- 1F1B Interleaved:  Virtual stages, reduced bubble (Narayanan et al., 2021)
- ZB-H1:             Zero Bubble, 1 pending W (Qi et al., 2024)
- ZB-H2:             Zero Bubble, 2 pending W — trades memory for zero bubble
- ZB-V:              Zero Bubble with virtual stages
- DualPipe:          Bidirectional pipeline (DeepSeek, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import ModelConfig, ParallelismConfig


class PipelineSchedule(Enum):
    ONE_F_ONE_B = "1f1b"
    ONE_F_ONE_B_INTERLEAVED = "1f1b_interleaved"
    ZB_H1 = "zb_h1"
    ZB_H2 = "zb_h2"
    ZB_V = "zb_v"
    DUALPIPE = "dualpipe"


@dataclass(frozen=True)
class ScheduleProfile:
    """Derived properties of a pipeline schedule for a given config."""
    schedule: PipelineSchedule
    pp_size: int
    num_chunks: int                  # Virtual stages per device (1 for non-interleaved)
    stash_counts: list[int]          # Per-stage stash count
    bubble_fraction: float           # Fraction of step time wasted in bubble
    extra_memory_per_stage: list[float]  # Extra bytes per stage (e.g., deferred W)
    description: str


# ── Stash count functions ────────────────────────────────────────────────────

def _stash_1f1b(stage: int, pp: int) -> int:
    """1F1B: stage p stashes (PP-1-p) microbatches during warmup."""
    return max(0, pp - 1 - stage)


def _stash_1f1b_interleaved(stage: int, pp: int) -> int:
    """1F1B Interleaved: same total activation stash per device as 1F1B.

    With v virtual stages per device, each chunk has L/(PP*v) layers.
    A microbatch visiting this device activates all v chunks, so
    the stashed activation per microbatch covers all local layers (L/PP),
    same as non-interleaved.  Stash count remains (PP-1-p).

    The benefit of interleaving is reduced bubble, not reduced memory.
    """
    return max(0, pp - 1 - stage)


def _stash_zb_h1(stage: int, pp: int) -> int:
    """ZB-H1: same activation stash as 1F1B.

    ZB-H1 splits backward into B (input gradients) and W (weight gradients).
    W fills bubble time, so the bubble approaches zero.  But the number of
    in-flight microbatches is the same as 1F1B, so activation stash is identical.
    The only extra memory is 1 deferred W per stage (handled in extra_memory).
    """
    return max(0, pp - 1 - stage)


def _stash_zb_h2(stage: int, pp: int) -> int:
    """ZB-H2: same activation stash as 1F1B.

    ZB-H2 achieves truly zero bubble by deferring up to PP weight gradient
    computations.  The activation stash is the same as 1F1B — the extra
    memory comes from deferred weight gradients, modeled separately.
    """
    return max(0, pp - 1 - stage)


def _stash_zb_v(stage: int, pp: int, num_chunks: int = 2) -> int:
    """ZB-V: zero bubble with virtual stages.

    Combines virtual stages (like 1F1B interleaved) with the B/W split.
    Each device holds num_chunks virtual stages.  The stash profile is
    similar to 1F1B interleaved — same total per device.
    """
    return max(0, pp - 1 - stage)


def _stash_dualpipe(stage: int, pp: int) -> int:
    """DualPipe: bidirectional pipeline with symmetric stash.

    Two micro-batch streams flow in opposite directions:
    - Stream A (forward): stage p at position p
    - Stream B (reverse): stage p at position (PP-1-p)

    Each stage holds stash from both streams:
    - From stream A: (PP-1-p) microbatches
    - From stream B: p microbatches (reverse direction)
    - Total: (PP-1-p) + p = PP-1 for ALL stages

    This makes stash pressure SYMMETRIC — every stage holds the same
    number of microbatch activations.  Pipeline-aware AC provides no
    benefit because there's no differential pressure to exploit.
    """
    forward_stash = max(0, pp - 1 - stage)
    reverse_stash = stage
    return forward_stash + reverse_stash  # = PP - 1 for all stages


# ── Bubble fraction ──────────────────────────────────────────────────────────

def _bubble_1f1b(pp: int, num_microbatches: int) -> float:
    """1F1B bubble fraction: (PP-1) / num_microbatches."""
    if num_microbatches == 0:
        return 1.0
    return (pp - 1) / num_microbatches


def _bubble_1f1b_interleaved(pp: int, num_microbatches: int, num_chunks: int = 2) -> float:
    """Interleaved 1F1B: bubble reduced by 1/num_chunks."""
    if num_microbatches == 0:
        return 1.0
    return (pp - 1) / (num_microbatches * num_chunks)


def _bubble_zero() -> float:
    """ZB schedules and DualPipe: zero or near-zero bubble."""
    return 0.0


# ── Extra memory ─────────────────────────────────────────────────────────────

def _extra_memory_zb_h2(
    cfg: ModelConfig,
    par: ParallelismConfig,
) -> list[float]:
    """ZB-H2 extra memory: deferred weight gradients, computed per stage.

    ZB-H2 defers up to PP weight gradient computations to fill the bubble.
    Each deferred W stores one full set of weight gradients for the stage's
    local layers.  Stages with fewer layers (uneven division) have less
    deferred-W overhead.

    extra_stage_i = PP × (local_layers_i × params_per_layer × bytes_per_param / tp)
    """
    from .environment import _layer_param_count, _stage_layer_span

    pp = par.pp_size
    extras = []
    for stage_idx in range(pp):
        start, end = _stage_layer_span(cfg.num_layers, pp, stage_idx)
        local_layers = end - start
        params_per_stage = local_layers * _layer_param_count(cfg)
        gradient_bytes = params_per_stage * cfg.dtype_bytes / par.tp_size
        extras.append(pp * gradient_bytes)
    return extras


# ── Main interface ───────────────────────────────────────────────────────────

def get_schedule_profile(
    schedule: PipelineSchedule,
    cfg: ModelConfig,
    par: ParallelismConfig,
    num_microbatches: int = 16,
    num_chunks: int = 2,
) -> ScheduleProfile:
    """Compute the full profile for a pipeline schedule.

    Args:
        schedule: Which pipeline schedule to use.
        cfg: Model config.
        par: Parallelism config (pp_size must be >= 2).
        num_microbatches: Total microbatches per training step.
        num_chunks: Virtual stages per device (for interleaved/ZB-V).
    """
    pp = par.pp_size

    zero_extras = [0.0] * pp

    if schedule == PipelineSchedule.ONE_F_ONE_B:
        stash_counts = [_stash_1f1b(s, pp) for s in range(pp)]
        bubble = _bubble_1f1b(pp, num_microbatches)
        extras = zero_extras
        desc = f"1F1B: asymmetric stash (first={pp-1}, last=0), bubble={bubble:.1%}"

    elif schedule == PipelineSchedule.ONE_F_ONE_B_INTERLEAVED:
        stash_counts = [_stash_1f1b_interleaved(s, pp) for s in range(pp)]
        bubble = _bubble_1f1b_interleaved(pp, num_microbatches, num_chunks)
        extras = zero_extras
        desc = (f"1F1B Interleaved (v={num_chunks}): "
                f"same stash as 1F1B, bubble={bubble:.1%}")

    elif schedule == PipelineSchedule.ZB_H1:
        stash_counts = [_stash_zb_h1(s, pp) for s in range(pp)]
        bubble = _bubble_zero()
        extras = zero_extras
        desc = f"ZB-H1: same stash as 1F1B, near-zero bubble, 1 deferred W"

    elif schedule == PipelineSchedule.ZB_H2:
        stash_counts = [_stash_zb_h2(s, pp) for s in range(pp)]
        bubble = _bubble_zero()
        extras = _extra_memory_zb_h2(cfg, par)
        desc = (f"ZB-H2: same stash as 1F1B, zero bubble, "
                f"PP deferred W (+{max(extras) / 1024**3:.2f} GB/stage max)")

    elif schedule == PipelineSchedule.ZB_V:
        stash_counts = [_stash_zb_v(s, pp, num_chunks) for s in range(pp)]
        bubble = _bubble_zero()
        extras = zero_extras
        desc = (f"ZB-V (v={num_chunks}): "
                f"same stash as 1F1B, zero bubble with virtual stages")

    elif schedule == PipelineSchedule.DUALPIPE:
        stash_counts = [_stash_dualpipe(s, pp) for s in range(pp)]
        bubble = _bubble_zero()
        extras = zero_extras
        desc = (f"DualPipe: SYMMETRIC stash (all stages={pp-1}), "
                f"zero bubble, bidirectional")

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return ScheduleProfile(
        schedule=schedule,
        pp_size=pp,
        num_chunks=num_chunks,
        stash_counts=stash_counts,
        bubble_fraction=bubble,
        extra_memory_per_stage=extras,
        description=desc,
    )
