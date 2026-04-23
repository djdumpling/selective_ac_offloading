"""Simulator environment: the main entry point.

Takes (model_spec, hardware_spec, per-layer AC decisions) and returns
a full analysis: peak memory, recompute overhead, offload stalls,
compression error, and per-layer breakdown.

This models the memory timeline across the full forward-backward pass
for a single training step, tracking:
1. Parameter memory (constant)
2. Optimizer state memory (constant)
3. Gradient memory (peaks during backward)
4. Activation memory (varies per-layer based on AC decisions)
5. Offload transfers in-flight
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compression_model import CompressionResult, compress_tensor
from .compute_model import (
    LayerComputeProfile,
    flops_to_latency,
    get_fwd_flops_per_layer,
    get_layer_compute_profile,
    get_recompute_overhead_ratio,
)
from .config import (
    BYTES_BF16,
    BYTES_FP32,
    GPUConfig,
    LayerStrategy,
    ModelConfig,
    ParallelismConfig,
    TensorAction,
    TensorDecision,
)
from .memory_model import TensorInfo, get_all_tensors_per_layer
from .offload_model import SyncMode, schedule_offloads


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class LayerMemoryBreakdown:
    """Memory analysis for a single layer."""
    layer_idx: int
    kept_bytes: float = 0.0
    recomputed_bytes: float = 0.0
    offloaded_bytes: float = 0.0
    compressed_original_bytes: float = 0.0
    compressed_stored_bytes: float = 0.0
    recompute_flops: float = 0.0
    offload_stall_s: float = 0.0
    compression_flops: float = 0.0
    compression_error: float = 0.0
    tensor_details: dict[str, str] = field(default_factory=dict)

    @property
    def total_hbm_bytes(self) -> float:
        """HBM consumed by this layer's activations (kept + compressed)."""
        return self.kept_bytes + self.compressed_stored_bytes


@dataclass
class SimulatorResult:
    """Full simulation output."""
    # Memory
    param_memory_bytes: float
    optimizer_memory_bytes: float
    gradient_memory_bytes: float
    peak_activation_memory_bytes: float
    total_peak_memory_bytes: float
    hbm_capacity_bytes: float
    fits_in_memory: bool

    # Overhead
    total_recompute_flops: float
    recompute_overhead_pct: float
    total_offload_stall_s: float
    total_compression_flops: float
    total_compression_error: float

    # Per-layer breakdown
    per_layer: list[LayerMemoryBreakdown]

    # Compute profile
    fwd_latency_s: float
    bwd_latency_s: float
    step_latency_s: float

    # Pipeline info
    pipeline_stage: int = 0
    stashed_microbatch_bytes: float = 0.0


# ── Helper: parameter / optimizer / gradient memory ──────────────────────────

def _layer_param_count(cfg: ModelConfig) -> int:
    """Approximate parameter count for one transformer layer."""
    h = cfg.hidden_dim
    ffn = cfg.ffn_dim
    kv_groups = cfg.num_kv_groups

    # Per-layer params:
    # Attention: W_q(h×h) + W_k(h×h/kv_groups) + W_v(h×h/kv_groups) + W_o(h×h)
    attn_params = h * h + 2 * h * (h // kv_groups) + h * h
    # MLP: depends on activation
    if cfg.is_swiglu:
        mlp_params = 3 * h * ffn  # gate, up, down
    else:
        mlp_params = 2 * h * ffn  # up, down
    # LayerNorm: 2 × (weight + bias) = 2 × 2h = 4h
    ln_params = 4 * h

    return attn_params + mlp_params + ln_params


def _stage_layer_span(
    num_layers: int,
    pp_size: int,
    pipeline_stage: int,
) -> tuple[int, int]:
    """Inclusive-exclusive layer range handled by one pipeline stage."""
    if pp_size <= 1:
        return 0, num_layers
    if pipeline_stage < 0 or pipeline_stage >= pp_size:
        raise ValueError(
            f"pipeline_stage={pipeline_stage} must be in [0, {pp_size - 1}]"
        )

    base = num_layers // pp_size
    extra = num_layers % pp_size
    start = pipeline_stage * base + min(pipeline_stage, extra)
    count = base + (1 if pipeline_stage < extra else 0)
    return start, start + count


def _param_count(
    cfg: ModelConfig,
    par: ParallelismConfig,
    pipeline_stage: int,
) -> int:
    """Approximate parameter count resident on one pipeline stage."""
    start, end = _stage_layer_span(cfg.num_layers, par.pp_size, pipeline_stage)
    local_layers = end - start
    count = local_layers * _layer_param_count(cfg)

    # Embeddings typically live on the first stage; final LN/head on the last.
    if pipeline_stage == 0:
        count += cfg.vocab_size * cfg.hidden_dim
    if pipeline_stage == par.pp_size - 1:
        count += 2 * cfg.hidden_dim

    return count


def _param_memory(
    cfg: ModelConfig,
    par: ParallelismConfig,
    pipeline_stage: int,
) -> float:
    """Parameter memory in bytes (bf16 weights)."""
    return _param_count(cfg, par, pipeline_stage) * BYTES_BF16 / par.tp_size


def _optimizer_memory(
    cfg: ModelConfig,
    par: ParallelismConfig,
    pipeline_stage: int,
) -> float:
    """Optimizer state memory (Adam: 12 bytes/param for fp32 copies + m + v)."""
    return _param_count(cfg, par, pipeline_stage) * 12 / par.tp_size / par.dp_size


def _gradient_memory(
    cfg: ModelConfig,
    par: ParallelismConfig,
    pipeline_stage: int,
) -> float:
    """Gradient memory (same dtype as params)."""
    return _param_count(cfg, par, pipeline_stage) * BYTES_BF16 / par.tp_size


# ── Liveness gap estimation ─────────────────────────────────────────────────

def _estimate_liveness_gap(
    layer_idx: int,
    num_layers: int,
    layer_compute: LayerComputeProfile,
) -> float:
    """Estimate liveness gap for block-boundary activations.

    A tensor created at layer i in the forward pass is consumed at layer i
    in the backward pass.  The gap spans:
    - Remaining forward layers: (num_layers - 1 - layer_idx) × fwd_latency
    - All backward layers up to layer i: (num_layers - 1 - layer_idx) × bwd_latency
    Plus the current layer's backward.
    """
    remaining = num_layers - 1 - layer_idx
    gap = (remaining * layer_compute.fwd_total_latency_s +
           remaining * layer_compute.bwd_total_latency_s +
           layer_compute.bwd_total_latency_s)
    return gap


def _estimate_intra_block_liveness_gap(
    layer_compute: LayerComputeProfile,
) -> float:
    """Liveness gap for intra-block intermediates (short-lived).

    These are created during a block's forward and consumed during the
    same block's backward.  Gap ≈ remaining forward of this layer +
    backward of this layer ≈ 1× fwd + 1× bwd ≈ 3× fwd.
    """
    return layer_compute.fwd_total_latency_s + layer_compute.bwd_total_latency_s


def _checkpoint_boundary_bytes(
    cfg: ModelConfig,
    par: ParallelismConfig,
) -> float:
    """Bytes retained per layer input under full-layer activation checkpointing."""
    return cfg.seq_len * cfg.micro_batch_size * cfg.hidden_dim * cfg.dtype_bytes / par.sp_size


# ── Main simulation ─────────────────────────────────────────────────────────

def simulate(
    cfg: ModelConfig,
    gpu: GPUConfig,
    strategies: list[LayerStrategy],
    par: ParallelismConfig = ParallelismConfig(),
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
    pipeline_stage: int = 0,
    num_microbatches_in_flight: int = 0,
    offload_sync_mode: SyncMode = "overlap",
) -> SimulatorResult:
    """Run the full simulation.

    Args:
        cfg: Model architecture config.
        gpu: GPU hardware config.
        strategies: Per-layer decisions (one LayerStrategy per layer).
            If empty/shorter than num_layers, defaults to KEEP for all tensors.
        par: Parallelism config.
        efficiency: Achieved MFU (fraction of peak TFLOPS).
        memory_budget_frac: Fraction of HBM to use (e.g., 0.90 for 90%).
        pipeline_stage: Pipeline stage index (0 = first stage).
        num_microbatches_in_flight: For pipeline parallelism, how many
            microbatch activations are stashed at this stage.
    """
    start_layer, end_layer = _stage_layer_span(
        cfg.num_layers, par.pp_size, pipeline_stage)
    layer_indices = list(range(start_layer, end_layer))
    num_local_layers = len(layer_indices)
    layer_compute = get_layer_compute_profile(cfg, gpu, par, efficiency)

    # Pad strategies if needed
    strategy_map: dict[int, LayerStrategy] = {s.layer_idx: s for s in strategies}

    # Fixed memory components
    param_mem = _param_memory(cfg, par, pipeline_stage)
    optim_mem = _optimizer_memory(cfg, par, pipeline_stage)
    grad_mem = _gradient_memory(cfg, par, pipeline_stage)

    # Process each layer
    per_layer_results: list[LayerMemoryBreakdown] = []
    total_recompute_flops = 0.0
    total_offload_stall = 0.0
    total_compression_flops = 0.0
    total_compression_error = 0.0

    for local_idx, layer_idx in enumerate(layer_indices):
        tensors = get_all_tensors_per_layer(cfg, par)
        strategy = strategy_map.get(layer_idx)
        breakdown = LayerMemoryBreakdown(layer_idx=layer_idx)

        liveness_gap_boundary = _estimate_liveness_gap(
            local_idx, num_local_layers, layer_compute)
        liveness_gap_intra = _estimate_intra_block_liveness_gap(layer_compute)

        # Collect OFFLOAD_CPU decisions for this layer. Stall is computed after
        # the per-tensor loop via schedule_offloads(), which models the
        # half-duplex PCIe bus: when multiple tensors offload within one layer,
        # their transfers queue on the shared bus instead of each hitting its
        # own independent deadline. Summing per-tensor stalls as if independent
        # is ~16% optimistic at PP=4 seq=32K on real H200 measurements.
        offload_pending: list[tuple[TensorInfo, float]] = []
        offload_stored_bytes: dict[str, float] = {}

        for tensor in tensors:
            # Look up decision for this tensor
            if strategy and tensor.name in strategy.decisions:
                decision = strategy.decisions[tensor.name]
            else:
                decision = TensorDecision(action=TensorAction.KEEP)

            # Determine liveness gap for this tensor
            is_boundary = tensor.block in ("layernorm", "residual")
            gap = liveness_gap_boundary if is_boundary else liveness_gap_intra
            stored_bytes = (
                decision.stored_size_bytes
                if decision.stored_size_bytes is not None
                else tensor.size_bytes
            )

            if decision.action == TensorAction.KEEP:
                breakdown.kept_bytes += stored_bytes
                breakdown.tensor_details[tensor.name] = "KEEP"

            elif decision.action == TensorAction.RECOMPUTE:
                if not tensor.recomputable and not decision.allow_nonrecomputable:
                    raise ValueError(
                        f"Tensor {tensor.name} cannot be recomputed directly."
                    )
                breakdown.recomputed_bytes += tensor.size_bytes
                if tensor.recomputable:
                    breakdown.recompute_flops += tensor.recompute_flops
                    total_recompute_flops += tensor.recompute_flops
                    breakdown.tensor_details[tensor.name] = "RECOMPUTE"
                else:
                    breakdown.tensor_details[tensor.name] = "RECOMPUTE (enclosing checkpoint)"

            elif decision.action == TensorAction.OFFLOAD_CPU:
                offload_pending.append((tensor, gap))
                offload_stored_bytes[tensor.name] = stored_bytes

            elif decision.action == TensorAction.COMPRESS:
                sb = cfg.seq_len * cfg.micro_batch_size
                # Determine feature dim based on tensor
                if "gelu" in tensor.name or "gate" in tensor.name or "up" in tensor.name or "linear2" in tensor.name:
                    feat_dim = cfg.ffn_dim // par.tp_size
                else:
                    feat_dim = cfg.hidden_dim // par.tp_size

                comp = compress_tensor(
                    tensor, sb, feat_dim,
                    rank=decision.compress_rank,
                    bytes_per_element=cfg.dtype_bytes,
                )
                breakdown.compressed_original_bytes += comp.original_bytes
                breakdown.compressed_stored_bytes += comp.compressed_bytes
                breakdown.compression_flops += comp.total_flops
                total_compression_flops += comp.total_flops
                breakdown.compression_error = max(
                    breakdown.compression_error, comp.estimated_relative_error)
                total_compression_error = max(
                    total_compression_error, comp.estimated_relative_error)
                breakdown.tensor_details[tensor.name] = (
                    f"COMPRESS r={decision.compress_rank} "
                    f"({comp.compression_ratio:.2f}x, err={comp.estimated_relative_error:.4f}, "
                    f"flops={comp.total_flops:.2e})"
                )

        # Schedule this layer's offloads on a shared half-duplex PCIe bus.
        # schedule_offloads sorts by decreasing liveness gap (agreeable
        # deadlines) and serializes transfers — so stall on a later tensor
        # reflects the bus being occupied by earlier ones. `sync_mode="serial"`
        # switches to the no-overlap model: stall = round_trip per tensor.
        if offload_pending:
            offload_results = schedule_offloads(
                offload_pending, gpu, par, sync_mode=offload_sync_mode,
            )
            for r in offload_results:
                breakdown.offloaded_bytes += offload_stored_bytes[r.tensor_name]
                breakdown.offload_stall_s += r.stall_time_s
                total_offload_stall += r.stall_time_s
                breakdown.tensor_details[r.tensor_name] = (
                    f"OFFLOAD (stall={r.stall_time_s * 1000:.2f}ms, "
                    f"eff_bw={r.effective_bw_gb_s:.1f}GB/s)"
                )

        per_layer_results.append(breakdown)

    # Peak activation memory = max across all layers of their HBM usage
    # In the simple model, all layers are alive simultaneously (no pipeline overlap)
    # Total activation memory = sum of all layers' HBM usage
    activation_memory_per_layer = [r.total_hbm_bytes for r in per_layer_results]
    total_activation_memory = sum(activation_memory_per_layer)

    # Pipeline stashing: early stages hold (s-1) microbatch activations
    # Each stashed microbatch has the same activation footprint
    stashed_bytes = num_microbatches_in_flight * total_activation_memory

    # Peak memory across the training step
    # During backward: activations + gradients + params + optimizer
    peak_activation = total_activation_memory + stashed_bytes
    total_peak = param_mem + optim_mem + grad_mem + peak_activation

    # Compute overhead
    fwd_flops = get_fwd_flops_per_layer(cfg, par) * num_local_layers
    training_flops = 3 * fwd_flops
    extra_flops = total_recompute_flops + total_compression_flops
    recompute_pct = (extra_flops / training_flops * 100
                     if training_flops > 0 else 0.0)

    # Latency
    fwd_lat = layer_compute.fwd_total_latency_s * num_local_layers
    bwd_lat = layer_compute.bwd_total_latency_s * num_local_layers
    recompute_lat = flops_to_latency(total_recompute_flops, gpu, efficiency)
    compression_lat = flops_to_latency(total_compression_flops, gpu, efficiency)
    step_lat = fwd_lat + bwd_lat + recompute_lat + compression_lat + total_offload_stall

    hbm_capacity = gpu.hbm_capacity_bytes * memory_budget_frac

    return SimulatorResult(
        param_memory_bytes=param_mem,
        optimizer_memory_bytes=optim_mem,
        gradient_memory_bytes=grad_mem,
        peak_activation_memory_bytes=peak_activation,
        total_peak_memory_bytes=total_peak,
        hbm_capacity_bytes=hbm_capacity,
        fits_in_memory=total_peak <= hbm_capacity,
        total_recompute_flops=total_recompute_flops,
        recompute_overhead_pct=recompute_pct,
        total_offload_stall_s=total_offload_stall,
        total_compression_flops=total_compression_flops,
        total_compression_error=total_compression_error,
        per_layer=per_layer_results,
        fwd_latency_s=fwd_lat,
        bwd_latency_s=bwd_lat,
        step_latency_s=step_lat,
        pipeline_stage=pipeline_stage,
        stashed_microbatch_bytes=stashed_bytes,
    )


# ── Convenience: simulate common strategies ──────────────────────────────────

def simulate_no_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    **kwargs,
) -> SimulatorResult:
    """Simulate with no activation checkpointing (keep everything)."""
    return simulate(cfg, gpu, strategies=[], par=par, **kwargs)


def simulate_full_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    **kwargs,
) -> SimulatorResult:
    """Simulate with full-layer activation checkpointing."""
    tensors = get_all_tensors_per_layer(cfg, par)
    strategies = []
    for i in range(cfg.num_layers):
        decisions = _build_full_ac_decisions(tensors, cfg, par)
        strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))
    return simulate(cfg, gpu, strategies=strategies, par=par, **kwargs)


def simulate_selective_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    **kwargs,
) -> SimulatorResult:
    """Simulate Korthikanti-style selective AC: recompute attention core, keep MLP.

    This is the Megatron-LM default: recompute QK^T, softmax, attention
    dropout while keeping all MLP activations.
    """
    tensors = get_all_tensors_per_layer(cfg, par)
    recompute_names = {"attn_softmax"}
    if cfg.use_softmax_dropout:
        recompute_names.add("attn_softmax_dropout_output")
    strategies = []
    for i in range(cfg.num_layers):
        decisions = {}
        for t in tensors:
            if not cfg.use_flash_attention and t.name in recompute_names:
                decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
            else:
                decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
        strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))
    return simulate(cfg, gpu, strategies=strategies, par=par, **kwargs)


# ── Strategy builders for pipeline-aware AC ──────────────────────────────────

def _build_no_ac_decisions(
    tensors: list[TensorInfo],
) -> dict[str, TensorDecision]:
    """All tensors kept — maximum memory, zero overhead."""
    return {t.name: TensorDecision(action=TensorAction.KEEP) for t in tensors}


def _build_fa_selective_decisions(
    tensors: list[TensorInfo],
    cfg: ModelConfig,
) -> dict[str, TensorDecision]:
    """FA-era selective: recompute the activation function output only.

    The checkpoint wraps the pointwise activation (SiLU or GeLU), so the
    intermediate activation output (mlp_silu_output / mlp_gelu_output) is
    eliminated.  The second-linear input (mlp_linear2_input) is still
    retained because it exits the checkpoint region and is saved by
    down_proj's backward.
    """
    recompute_names = {"mlp_silu_output", "mlp_gelu_output"}
    decisions = {}
    for t in tensors:
        if t.name in recompute_names:
            decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
        else:
            decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
    return decisions


def _build_korthikanti_selective_decisions(
    tensors: list[TensorInfo],
    cfg: ModelConfig,
) -> dict[str, TensorDecision]:
    """Korthikanti selective: recompute attention core (QK^T, softmax, dropout output).

    Only effective without FlashAttention — eliminates the quadratic 5as²b term
    at ~2.7% overhead.  With FA active, this degenerates to No AC (the quadratic
    tensors don't exist).
    """
    recompute_names = {"attn_softmax", "attn_softmax_dropout_output"}
    decisions = {}
    for t in tensors:
        if not cfg.use_flash_attention and t.name in recompute_names and t.recomputable:
            decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
        else:
            decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
    return decisions


def _build_offload_linear2_decisions(
    tensors: list[TensorInfo],
) -> dict[str, TensorDecision]:
    """Offload just `mlp_linear2_input` (saved by down_proj's backward).

    Keeps every other activation on-GPU. Smallest PCIe footprint that still
    frees the biggest single MLP tensor per layer.
    """
    decisions = {}
    for t in tensors:
        if t.name == "mlp_linear2_input":
            decisions[t.name] = TensorDecision(action=TensorAction.OFFLOAD_CPU)
        else:
            decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
    return decisions


def _build_offload_all_mlp_decisions(
    tensors: list[TensorInfo],
) -> dict[str, TensorDecision]:
    """Offload every MLP activation that's worth more than a few MB: gate_output,
    up_output, silu_output (SwiGLU) or gelu_output (GeLU), and linear2_input.

    Attention tensors stay on-GPU — they're already small when FA is active."""
    offload_names = {
        "mlp_gate_output",
        "mlp_up_output",
        "mlp_silu_output",
        "mlp_gelu_output",
        "mlp_linear2_input",
    }
    decisions = {}
    for t in tensors:
        if t.name in offload_names:
            decisions[t.name] = TensorDecision(action=TensorAction.OFFLOAD_CPU)
        else:
            decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
    return decisions


def _build_full_ac_decisions(
    tensors: list[TensorInfo],
    cfg: ModelConfig,
    par: ParallelismConfig,
) -> dict[str, TensorDecision]:
    """Full-layer activation checkpointing.

    This matches the common HF/PyTorch behavior: retain the layer input at the
    checkpoint boundary and recompute the layer interior during backward.
    """
    checkpoint_bytes = _checkpoint_boundary_bytes(cfg, par)
    decisions = {}
    for t in tensors:
        if t.name == "ln1_input":
            decisions[t.name] = TensorDecision(
                action=TensorAction.KEEP,
                stored_size_bytes=checkpoint_bytes,
            )
        else:
            decisions[t.name] = TensorDecision(
                action=TensorAction.RECOMPUTE,
                allow_nonrecomputable=not t.recomputable,
            )
    return decisions


# Named strategy levels in order of increasing aggressiveness.
# Pipeline-aware AC tries these in order and picks the first that fits.
STRATEGY_LEVELS = [
    ("No AC", _build_no_ac_decisions),
    ("Offload linear2", _build_offload_linear2_decisions),
    ("Offload all MLP", _build_offload_all_mlp_decisions),
    ("FA-Selective", _build_fa_selective_decisions),
    ("Korthikanti Selective", _build_korthikanti_selective_decisions),
    ("Full AC", _build_full_ac_decisions),
]


# ── Pipeline-aware multi-stage simulation ────────────────────────────────────

@dataclass
class PipelineStageResult:
    """Result for one pipeline stage."""
    stage_idx: int
    strategy_name: str
    num_stashed_microbatches: int
    sim: SimulatorResult


@dataclass
class PipelineResult:
    """Aggregate result across all pipeline stages."""
    stages: list[PipelineStageResult]
    bottleneck_stage: int               # Stage with highest step latency
    bottleneck_step_latency_s: float    # Per-microbatch time of slowest stage
    overall_step_latency_s: float       # Bubble-adjusted effective per-microbatch latency
    total_recompute_overhead_pct: float  # Bottleneck stage's overhead
    all_fit: bool                       # All stages fit in memory
    schedule_name: str = ""
    bubble_fraction: float = 0.0
    num_microbatches: int = 16

    @property
    def bottleneck(self) -> PipelineStageResult:
        return self.stages[self.bottleneck_stage]


def _stash_count_1f1b(stage_idx: int, pp_size: int) -> int:
    """Number of in-flight microbatch activations stashed at a 1F1B stage."""
    return max(0, pp_size - 1 - stage_idx)


def _build_decisions(build_fn, tensors, cfg, par):
    """Call a strategy builder, passing through any required config arguments.

    Offload builders (`_build_offload_*`) take only `tensors` — same arity
    as `_build_no_ac_decisions`, so they fall through to the default branch."""
    if build_fn == _build_fa_selective_decisions:
        return build_fn(tensors, cfg)
    if build_fn == _build_korthikanti_selective_decisions:
        return build_fn(tensors, cfg)
    if build_fn == _build_full_ac_decisions:
        return build_fn(tensors, cfg, par)
    return build_fn(tensors)


def _run_pipeline_simulation(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig,
    stash_counts: list[int],
    strategy_assignments: list[str],
    extra_memory_per_stage: list[float],
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
    schedule_name: str = "",
    bubble_fraction: float = 0.0,
    num_microbatches: int = 16,
    offload_sync_mode: SyncMode = "overlap",
) -> PipelineResult:
    """Core pipeline simulation: run each stage with its assigned strategy.

    Args:
        stash_counts: Per-stage microbatch stash count.
        strategy_assignments: Per-stage strategy name from STRATEGY_LEVELS.
        extra_memory_per_stage: Per-stage extra bytes (e.g., deferred W in ZB-H2).
        bubble_fraction: Fraction of total step time wasted in pipeline bubble.
        num_microbatches: Total microbatches per training step (affects bubble cost).
    """
    pp_size = par.pp_size
    tensors = get_all_tensors_per_layer(cfg, par)

    # Build strategy lookup
    strategy_lookup = {name: fn for name, fn in STRATEGY_LEVELS}

    stage_results: list[PipelineStageResult] = []

    for stage_idx in range(pp_size):
        stash_count = stash_counts[stage_idx]
        sname = strategy_assignments[stage_idx]
        build_fn = strategy_lookup[sname]

        start, end = _stage_layer_span(cfg.num_layers, pp_size, stage_idx)
        decisions = _build_decisions(build_fn, tensors, cfg, par)

        strategies = [
            LayerStrategy(layer_idx=i, decisions=decisions)
            for i in range(start, end)
        ]

        result = simulate(
            cfg, gpu, strategies,
            par=par,
            efficiency=efficiency,
            memory_budget_frac=memory_budget_frac,
            pipeline_stage=stage_idx,
            num_microbatches_in_flight=stash_count,
            offload_sync_mode=offload_sync_mode,
        )

        # Account for per-stage extra memory (e.g., ZB-H2 deferred gradients)
        stage_extra = extra_memory_per_stage[stage_idx]
        if stage_extra > 0:
            adjusted_peak = result.total_peak_memory_bytes + stage_extra
            result = SimulatorResult(
                param_memory_bytes=result.param_memory_bytes,
                optimizer_memory_bytes=result.optimizer_memory_bytes,
                gradient_memory_bytes=result.gradient_memory_bytes + stage_extra,
                peak_activation_memory_bytes=result.peak_activation_memory_bytes,
                total_peak_memory_bytes=adjusted_peak,
                hbm_capacity_bytes=result.hbm_capacity_bytes,
                fits_in_memory=adjusted_peak <= result.hbm_capacity_bytes,
                total_recompute_flops=result.total_recompute_flops,
                recompute_overhead_pct=result.recompute_overhead_pct,
                total_offload_stall_s=result.total_offload_stall_s,
                total_compression_flops=result.total_compression_flops,
                total_compression_error=result.total_compression_error,
                per_layer=result.per_layer,
                fwd_latency_s=result.fwd_latency_s,
                bwd_latency_s=result.bwd_latency_s,
                step_latency_s=result.step_latency_s,
                pipeline_stage=result.pipeline_stage,
                stashed_microbatch_bytes=result.stashed_microbatch_bytes,
            )

        stage_results.append(PipelineStageResult(
            stage_idx=stage_idx,
            strategy_name=sname,
            num_stashed_microbatches=stash_count,
            sim=result,
        ))

    bottleneck_idx = max(range(pp_size), key=lambda i: stage_results[i].sim.step_latency_s)
    bottleneck_latency = stage_results[bottleneck_idx].sim.step_latency_s
    all_fit = all(sr.sim.fits_in_memory for sr in stage_results)

    # Bubble-adjusted effective per-microbatch latency.
    # For M microbatches, total pipeline time is:
    #   M × bottleneck_per_microbatch × (1 + bubble_fraction)
    # Dividing by M gives the effective per-microbatch latency reported here.
    overall_latency = bottleneck_latency * (1.0 + bubble_fraction)

    return PipelineResult(
        stages=stage_results,
        bottleneck_stage=bottleneck_idx,
        bottleneck_step_latency_s=bottleneck_latency,
        overall_step_latency_s=overall_latency,
        total_recompute_overhead_pct=stage_results[bottleneck_idx].sim.recompute_overhead_pct,
        all_fit=all_fit,
        schedule_name=schedule_name,
        bubble_fraction=bubble_fraction,
        num_microbatches=num_microbatches,
    )


def simulate_pipeline_aware_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig,
    schedule=None,
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
    num_microbatches: int = 16,
    num_chunks: int = 2,
    offload_sync_mode: SyncMode = "overlap",
) -> PipelineResult:
    """Simulate non-uniform AC across pipeline stages under any schedule.

    For each stage, selects the **least aggressive** AC strategy that
    fits within the HBM budget after accounting for the schedule-specific
    stashed microbatch activations.

    Args:
        schedule: A PipelineSchedule enum value, or None for 1F1B default.
        offload_sync_mode: See `simulator.offload_model.SyncMode`.
    """
    from .pipeline_schedules import PipelineSchedule, get_schedule_profile

    if schedule is None:
        schedule = PipelineSchedule.ONE_F_ONE_B

    pp_size = par.pp_size
    assert pp_size >= 2, "Pipeline-aware AC requires pp_size >= 2"

    profile = get_schedule_profile(schedule, cfg, par, num_microbatches, num_chunks)
    tensors = get_all_tensors_per_layer(cfg, par)

    # For each stage, find the least aggressive strategy that fits
    strategy_assignments: list[str] = []
    for stage_idx in range(pp_size):
        stash_count = profile.stash_counts[stage_idx]
        stage_extra = profile.extra_memory_per_stage[stage_idx]
        chosen = None

        for level_name, build_fn in STRATEGY_LEVELS:
            start, end = _stage_layer_span(cfg.num_layers, pp_size, stage_idx)
            decisions = _build_decisions(build_fn, tensors, cfg, par)

            strategies = [
                LayerStrategy(layer_idx=i, decisions=decisions)
                for i in range(start, end)
            ]

            result = simulate(
                cfg, gpu, strategies,
                par=par,
                efficiency=efficiency,
                memory_budget_frac=memory_budget_frac,
                pipeline_stage=stage_idx,
                num_microbatches_in_flight=stash_count,
                offload_sync_mode=offload_sync_mode,
            )

            # Check fit including per-stage extra memory
            peak_with_extra = result.total_peak_memory_bytes + stage_extra
            if peak_with_extra <= result.hbm_capacity_bytes:
                chosen = level_name
                break

        strategy_assignments.append(chosen or "Full AC")

    return _run_pipeline_simulation(
        cfg, gpu, par,
        stash_counts=profile.stash_counts,
        strategy_assignments=strategy_assignments,
        extra_memory_per_stage=profile.extra_memory_per_stage,
        efficiency=efficiency,
        memory_budget_frac=memory_budget_frac,
        schedule_name=profile.description,
        bubble_fraction=profile.bubble_fraction,
        num_microbatches=num_microbatches,
        offload_sync_mode=offload_sync_mode,
    )


def simulate_pipeline_uniform_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig,
    strategy_name: str = "Full AC",
    schedule=None,
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
    num_microbatches: int = 16,
    num_chunks: int = 2,
    offload_sync_mode: SyncMode = "overlap",
) -> PipelineResult:
    """Simulate uniform AC across all pipeline stages under any schedule."""
    from .pipeline_schedules import PipelineSchedule, get_schedule_profile

    if schedule is None:
        schedule = PipelineSchedule.ONE_F_ONE_B

    pp_size = par.pp_size
    assert pp_size >= 2, "Pipeline AC requires pp_size >= 2"

    profile = get_schedule_profile(schedule, cfg, par, num_microbatches, num_chunks)
    assignments = [strategy_name] * pp_size

    return _run_pipeline_simulation(
        cfg, gpu, par,
        stash_counts=profile.stash_counts,
        strategy_assignments=assignments,
        extra_memory_per_stage=profile.extra_memory_per_stage,
        efficiency=efficiency,
        memory_budget_frac=memory_budget_frac,
        schedule_name=profile.description,
        bubble_fraction=profile.bubble_fraction,
        num_microbatches=num_microbatches,
        offload_sync_mode=offload_sync_mode,
    )


def simulate_pipeline_custom_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig,
    strategy_assignments: list[str],
    schedule=None,
    efficiency: float = 0.5,
    memory_budget_frac: float = 0.90,
    num_microbatches: int = 16,
    num_chunks: int = 2,
    offload_sync_mode: SyncMode = "overlap",
) -> PipelineResult:
    """Simulate a user-specified per-stage AC assignment under any schedule.

    Unlike `simulate_pipeline_aware_ac` (which walks `STRATEGY_LEVELS`
    least-aggressive-first per stage) or `simulate_pipeline_uniform_ac`
    (which applies one strategy everywhere), this takes the caller's exact
    per-stage list. Used by the throughput runner's `--per-stage` override
    so a manual experiment still gets a simulator prediction that matches
    the strategies actually running on GPU.
    """
    from .pipeline_schedules import PipelineSchedule, get_schedule_profile

    if schedule is None:
        schedule = PipelineSchedule.ONE_F_ONE_B

    pp_size = par.pp_size
    assert pp_size >= 2, "Pipeline AC requires pp_size >= 2"
    if len(strategy_assignments) != pp_size:
        raise ValueError(
            f"strategy_assignments has {len(strategy_assignments)} entries, "
            f"expected pp_size={pp_size}"
        )
    known = {name for name, _ in STRATEGY_LEVELS}
    unknown = [s for s in strategy_assignments if s not in known]
    if unknown:
        raise ValueError(
            f"unknown strategy names {unknown}; expected any of {sorted(known)}"
        )

    profile = get_schedule_profile(schedule, cfg, par, num_microbatches, num_chunks)

    return _run_pipeline_simulation(
        cfg, gpu, par,
        stash_counts=profile.stash_counts,
        strategy_assignments=list(strategy_assignments),
        extra_memory_per_stage=profile.extra_memory_per_stage,
        efficiency=efficiency,
        memory_budget_frac=memory_budget_frac,
        schedule_name=profile.description,
        bubble_fraction=profile.bubble_fraction,
        num_microbatches=num_microbatches,
        offload_sync_mode=offload_sync_mode,
    )


def print_pipeline_result(pr: PipelineResult) -> None:
    """Pretty-print a pipeline-aware simulation result."""
    if pr.schedule_name:
        print(f"  Schedule: {pr.schedule_name}")

    print(f"\n  {'Stage':>5} {'Strategy':<16} {'Stash':>5} "
          f"{'Activation':>12} {'Peak':>12} {'Step(ms)':>10} {'Ovhd%':>7} {'Fit':>4}")
    print(f"  {'-'*5} {'-'*16} {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*7} {'-'*4}")

    for sr in pr.stages:
        fit = "YES" if sr.sim.fits_in_memory else "NO"
        marker = " <-- bottleneck" if sr.stage_idx == pr.bottleneck_stage else ""
        print(
            f"  {sr.stage_idx:>5} {sr.strategy_name:<16} {sr.num_stashed_microbatches:>5} "
            f"{_fmt_bytes(sr.sim.peak_activation_memory_bytes):>12} "
            f"{_fmt_bytes(sr.sim.total_peak_memory_bytes):>12} "
            f"{sr.sim.step_latency_s * 1000:>10.2f} "
            f"{sr.sim.recompute_overhead_pct:>6.2f}% {fit:>4}{marker}"
        )

    if pr.bubble_fraction > 0:
        print(f"\n  Bottleneck per-microbatch: {pr.bottleneck_step_latency_s*1000:.2f}ms")
        print(f"  Bubble overhead: +{pr.bubble_fraction:.1%}")
        print(f"  Effective per-microbatch: {pr.overall_step_latency_s*1000:.2f}ms "
              f"(= {pr.bottleneck_step_latency_s*1000:.2f} × {1+pr.bubble_fraction:.3f})")
    else:
        print(f"\n  Per-microbatch: {pr.overall_step_latency_s*1000:.2f}ms (zero bubble)")
    print(f"  Overhead: {pr.total_recompute_overhead_pct:.2f}%, "
          f"all_fit={'YES' if pr.all_fit else 'NO'}")


# ── Single-stage convenience strategies ──────────────────────────────────────

def simulate_fa_selective_ac(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    **kwargs,
) -> SimulatorResult:
    """Simulate FA-era selective AC: recompute activation functions, keep matmul outputs.

    This is the FA-era analogue of Korthikanti's strategy.  With FA active,
    the attention core is already memory-efficient.  The new bottleneck is
    the MLP block.  This strategy:

    - Keeps all attention tensors (FA handles them)
    - Keeps MLP matmul outputs (gate_output, up_output, gelu_input)
      because they're expensive to recompute (full matmul)
    - Recomputes the activation function output (mlp_silu_output or
      mlp_gelu_output): the cheap pointwise op wrapped in a checkpoint

    The checkpoint wraps the pointwise activation, so the intermediate
    (silu(gate) for SwiGLU, gelu(x) for GeLU) is not saved.  The
    second-linear input (mlp_linear2_input = silu(gate)*up or gelu(x))
    still exits the checkpoint and is saved by down_proj's backward.

    For GELU MLP:  saves ffn*sb*bpe/tp per layer (mlp_gelu_output)
                   at cost of GeLU recompute (~8 FLOPs per element)
    For SwiGLU MLP: saves ffn*sb*bpe/tp per layer (mlp_silu_output)
                    at cost of SiLU recompute
    """
    tensors = get_all_tensors_per_layer(cfg, par)
    recompute_names = {"mlp_silu_output", "mlp_gelu_output"}
    strategies = []
    for i in range(cfg.num_layers):
        decisions = {}
        for t in tensors:
            if t.name in recompute_names:
                decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
            else:
                decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
        strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))
    return simulate(cfg, gpu, strategies=strategies, par=par, **kwargs)


# ── Pretty printing ─────────────────────────────────────────────────────────

def _fmt_bytes(b: float) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024:.2f} KB"


def print_result(result: SimulatorResult) -> None:
    """Pretty-print a simulation result."""
    print("=" * 70)
    print("SIMULATION RESULT")
    print("=" * 70)
    print(f"  Parameter memory:          {_fmt_bytes(result.param_memory_bytes)}")
    print(f"  Optimizer memory:          {_fmt_bytes(result.optimizer_memory_bytes)}")
    print(f"  Gradient memory:           {_fmt_bytes(result.gradient_memory_bytes)}")
    print(f"  Peak activation memory:    {_fmt_bytes(result.peak_activation_memory_bytes)}")
    if result.stashed_microbatch_bytes > 0:
        print(f"    (includes stashed):      {_fmt_bytes(result.stashed_microbatch_bytes)}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total peak memory:         {_fmt_bytes(result.total_peak_memory_bytes)}")
    print(f"  HBM capacity (budget):     {_fmt_bytes(result.hbm_capacity_bytes)}")
    print(f"  Fits in memory:            {'YES' if result.fits_in_memory else 'NO — OOM!'}")
    print()
    print(f"  Recompute overhead:        {result.recompute_overhead_pct:.2f}%")
    print(f"  Offload stall:             {result.total_offload_stall_s * 1000:.2f} ms")
    print(f"  Max compression error:     {result.total_compression_error:.6f}")
    print()
    print(f"  Forward latency:           {result.fwd_latency_s * 1000:.2f} ms")
    print(f"  Backward latency:          {result.bwd_latency_s * 1000:.2f} ms")
    print(f"  Step latency:              {result.step_latency_s * 1000:.2f} ms")
    print("=" * 70)

    # Per-layer summary
    print("\nPER-LAYER BREAKDOWN (first 5 layers):")
    for lr in result.per_layer[:5]:
        print(f"  Layer {lr.layer_idx}: "
              f"kept={_fmt_bytes(lr.kept_bytes)}, "
              f"recomp={_fmt_bytes(lr.recomputed_bytes)}, "
              f"offload={_fmt_bytes(lr.offloaded_bytes)}, "
              f"compress={_fmt_bytes(lr.compressed_stored_bytes)}")
        for tname, action in lr.tensor_details.items():
            print(f"    {tname}: {action}")
    if len(result.per_layer) > 5:
        print(f"  ... ({len(result.per_layer) - 5} more layers)")
