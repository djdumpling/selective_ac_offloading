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
from .offload_model import OffloadResult, compute_offload_result


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

def _param_count(cfg: ModelConfig) -> int:
    """Approximate parameter count for a transformer model."""
    h = cfg.hidden_dim
    ffn = cfg.ffn_dim
    L = cfg.num_layers
    V = cfg.vocab_size
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

    per_layer = attn_params + mlp_params + ln_params
    # Embedding + final LN + output head
    embedding = V * h
    final_ln = 2 * h
    # Output head often shares weights with embedding
    output_head = 0  # weight-tied

    return L * per_layer + embedding + final_ln + output_head


def _param_memory(cfg: ModelConfig, par: ParallelismConfig) -> float:
    """Parameter memory in bytes (bf16 weights)."""
    return _param_count(cfg) * BYTES_BF16 / par.tp_size


def _optimizer_memory(cfg: ModelConfig, par: ParallelismConfig) -> float:
    """Optimizer state memory (Adam: 12 bytes/param for fp32 copies + m + v)."""
    return _param_count(cfg) * 12 / par.tp_size / par.dp_size


def _gradient_memory(cfg: ModelConfig, par: ParallelismConfig) -> float:
    """Gradient memory (same dtype as params)."""
    return _param_count(cfg) * BYTES_BF16 / par.tp_size


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
    num_layers = cfg.num_layers
    layer_compute = get_layer_compute_profile(cfg, gpu, par, efficiency)

    # Pad strategies if needed
    strategy_map: dict[int, LayerStrategy] = {s.layer_idx: s for s in strategies}

    # Fixed memory components
    param_mem = _param_memory(cfg, par)
    optim_mem = _optimizer_memory(cfg, par)
    grad_mem = _gradient_memory(cfg, par)

    # Process each layer
    per_layer_results: list[LayerMemoryBreakdown] = []
    total_recompute_flops = 0.0
    total_offload_stall = 0.0
    total_compression_error = 0.0

    for layer_idx in range(num_layers):
        tensors = get_all_tensors_per_layer(cfg, par)
        strategy = strategy_map.get(layer_idx)
        breakdown = LayerMemoryBreakdown(layer_idx=layer_idx)

        liveness_gap_boundary = _estimate_liveness_gap(
            layer_idx, num_layers, layer_compute)
        liveness_gap_intra = _estimate_intra_block_liveness_gap(layer_compute)

        for tensor in tensors:
            # Look up decision for this tensor
            if strategy and tensor.name in strategy.decisions:
                decision = strategy.decisions[tensor.name]
            else:
                decision = TensorDecision(action=TensorAction.KEEP)

            # Determine liveness gap for this tensor
            is_boundary = tensor.block in ("layernorm", "residual")
            gap = liveness_gap_boundary if is_boundary else liveness_gap_intra

            if decision.action == TensorAction.KEEP:
                breakdown.kept_bytes += tensor.size_bytes
                breakdown.tensor_details[tensor.name] = "KEEP"

            elif decision.action == TensorAction.RECOMPUTE:
                breakdown.recomputed_bytes += tensor.size_bytes
                breakdown.recompute_flops += tensor.recompute_flops
                total_recompute_flops += tensor.recompute_flops
                breakdown.tensor_details[tensor.name] = "RECOMPUTE"

            elif decision.action == TensorAction.OFFLOAD_CPU:
                result = compute_offload_result(tensor, gap, gpu)
                breakdown.offloaded_bytes += tensor.size_bytes
                breakdown.offload_stall_s += result.stall_time_s
                total_offload_stall += result.stall_time_s
                breakdown.tensor_details[tensor.name] = (
                    f"OFFLOAD (stall={result.stall_time_s * 1000:.2f}ms)"
                )

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
                breakdown.compression_error = max(
                    breakdown.compression_error, comp.estimated_relative_error)
                total_compression_error = max(
                    total_compression_error, comp.estimated_relative_error)
                breakdown.tensor_details[tensor.name] = (
                    f"COMPRESS r={decision.compress_rank} "
                    f"({comp.compression_ratio:.2f}x, err={comp.estimated_relative_error:.4f})"
                )

        per_layer_results.append(breakdown)

    # Peak activation memory = max across all layers of their HBM usage
    # In the simple model, all layers are alive simultaneously (no pipeline overlap)
    # Total activation memory = sum of all layers' HBM usage
    activation_memory_per_layer = [r.total_hbm_bytes for r in per_layer_results]
    total_activation_memory = sum(activation_memory_per_layer)

    # Pipeline stashing: early stages hold (s-1) microbatch activations
    # Each stashed microbatch has the same activation footprint
    activation_per_microbatch = total_activation_memory / num_layers  # per-layer avg
    stashed_bytes = num_microbatches_in_flight * activation_per_microbatch

    # Peak memory across the training step
    # During backward: activations + gradients + params + optimizer
    peak_activation = total_activation_memory + stashed_bytes
    total_peak = param_mem + optim_mem + grad_mem + peak_activation

    # Compute overhead
    fwd_flops = get_fwd_flops_per_layer(cfg, par) * num_layers
    training_flops = 3 * fwd_flops
    recompute_pct = (total_recompute_flops / training_flops * 100
                     if training_flops > 0 else 0.0)

    # Latency
    fwd_lat = layer_compute.fwd_total_latency_s * num_layers
    bwd_lat = layer_compute.bwd_total_latency_s * num_layers
    recompute_lat = flops_to_latency(total_recompute_flops, gpu, efficiency)
    step_lat = fwd_lat + bwd_lat + recompute_lat + total_offload_stall

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
    """Simulate with full activation checkpointing (recompute everything)."""
    tensors = get_all_tensors_per_layer(cfg, par)
    strategies = []
    for i in range(cfg.num_layers):
        decisions = {}
        for t in tensors:
            # Dropout masks can't be recomputed (need same RNG state)
            if "dropout_mask" in t.name or "logsumexp" in t.name:
                decisions[t.name] = TensorDecision(action=TensorAction.KEEP)
            else:
                decisions[t.name] = TensorDecision(action=TensorAction.RECOMPUTE)
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
    recompute_names = {
        "attn_softmax", "attn_softmax_dropout_mask",
        "attn_softmax_dropout_output", "attn_out_proj_input",
    }
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
