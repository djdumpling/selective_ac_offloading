"""Per-tensor and per-layer compute cost model.

Estimates FLOPs and wall-clock latency for forward and backward passes,
and specifically for recomputing individual activation tensors.

Reference: llm-analysis (cli99) for per-layer FLOPs formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GPUConfig, ModelConfig, ParallelismConfig
from .memory_model import TensorInfo


@dataclass(frozen=True)
class LayerComputeProfile:
    """Compute costs for a single transformer layer."""
    fwd_attn_flops: float
    fwd_mlp_flops: float
    fwd_total_flops: float
    bwd_total_flops: float          # ≈ 2× forward
    fwd_attn_latency_s: float
    fwd_mlp_latency_s: float
    fwd_total_latency_s: float
    bwd_total_latency_s: float


def get_fwd_flops_attn(cfg: ModelConfig, par: ParallelismConfig = ParallelismConfig()) -> float:
    """Forward FLOPs for the attention block of one layer.

    Includes: Q, K, V projections + QK^T matmul + softmax·V + output projection.
    Reference: 4·b·s·h² (Q proj + out proj) + 4·b·s·h²/num_kv_groups (K+V proj)
               + 4·b·s²·h (attention scores + weighted sum)
    """
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim

    # Q and output projections: each 2·b·s·h² FLOPs
    qo_flops = 4 * b * s * h * h

    # K and V projections (with GQA): each 2·b·s·(h/num_kv_groups)·h
    kv_flops = 4 * b * s * h * h / cfg.num_kv_groups

    # Attention score computation: QK^T + softmax·V
    # QK^T: 2·b·a·s²·d_k = 2·b·s²·h  (since a·d_k = h)
    # Attn·V: 2·b·a·s²·d_k = 2·b·s²·h
    attn_flops = 4 * b * s * s * h

    return qo_flops + kv_flops + attn_flops


def get_fwd_flops_mlp(cfg: ModelConfig, par: ParallelismConfig = ParallelismConfig()) -> float:
    """Forward FLOPs for the MLP block of one layer.

    Dense MLP (GELU):   2 linear layers: 2 × 2·b·s·h·ffn = 4·b·s·h·ffn
    SwiGLU MLP:         3 linear layers: 3 × 2·b·s·h·ffn = 6·b·s·h·ffn
    """
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim
    ffn = cfg.ffn_dim

    multiplier = 6 if cfg.is_swiglu else 4
    return multiplier * b * s * h * ffn


def get_fwd_flops_per_layer(cfg: ModelConfig, par: ParallelismConfig = ParallelismConfig()) -> float:
    """Total forward FLOPs for one transformer layer."""
    return get_fwd_flops_attn(cfg, par) + get_fwd_flops_mlp(cfg, par)


def flops_to_latency(flops: float, gpu: GPUConfig, efficiency: float = 0.5) -> float:
    """Convert FLOPs to wall-clock latency in seconds.

    Args:
        flops: Number of floating point operations.
        gpu: GPU hardware profile.
        efficiency: Achieved fraction of peak TFLOPS (MFU). Typical: 0.3-0.6.
    """
    achieved_flops_per_sec = gpu.peak_flops_per_sec * efficiency
    if achieved_flops_per_sec == 0:
        return float("inf")
    return flops / achieved_flops_per_sec


def get_layer_compute_profile(
    cfg: ModelConfig,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    efficiency: float = 0.5,
) -> LayerComputeProfile:
    """Full compute profile for one transformer layer."""
    fwd_attn = get_fwd_flops_attn(cfg, par)
    fwd_mlp = get_fwd_flops_mlp(cfg, par)
    fwd_total = fwd_attn + fwd_mlp
    bwd_total = 2 * fwd_total  # Standard approximation

    return LayerComputeProfile(
        fwd_attn_flops=fwd_attn,
        fwd_mlp_flops=fwd_mlp,
        fwd_total_flops=fwd_total,
        bwd_total_flops=bwd_total,
        fwd_attn_latency_s=flops_to_latency(fwd_attn, gpu, efficiency),
        fwd_mlp_latency_s=flops_to_latency(fwd_mlp, gpu, efficiency),
        fwd_total_latency_s=flops_to_latency(fwd_total, gpu, efficiency),
        bwd_total_latency_s=flops_to_latency(bwd_total, gpu, efficiency),
    )


def get_tensor_recompute_latency(
    tensor: TensorInfo,
    gpu: GPUConfig,
    efficiency: float = 0.5,
) -> float:
    """Wall-clock time to recompute a single tensor."""
    return flops_to_latency(tensor.recompute_flops, gpu, efficiency)


def get_recompute_overhead_ratio(
    tensors_to_recompute: list[TensorInfo],
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> float:
    """Fraction of extra FLOPs from recomputing given tensors vs. full forward.

    The Korthikanti selective strategy (recompute attention core only)
    achieves ~2.7% overhead on GPT-3.  We compute the actual ratio for
    any arbitrary set of tensors.
    """
    recompute_flops = sum(t.recompute_flops for t in tensors_to_recompute)
    # Total training FLOPs = 3 × forward (1 fwd + 2 bwd)
    fwd_flops = get_fwd_flops_per_layer(cfg, par)
    training_flops = 3 * fwd_flops
    if training_flops == 0:
        return 0.0
    return recompute_flops / training_flops
