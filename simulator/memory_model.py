"""Per-tensor activation memory model for transformer layers.

Derives from Korthikanti et al. (MLSys 2023) but at individual tensor
granularity rather than per-block aggregates.  FA-aware: when FlashAttention
is active the quadratic attention tensors are never materialized.

Key reference formulas (no FA, no AC, bf16):
    Attention block:  11·s·b·h + 5·a·s²·b  bytes
    MLP block:        19·s·b·h              bytes
    LayerNorm (×2):    4·s·b·h              bytes
    Total per layer:  s·b·h·(34 + 5·a·s/h)  bytes

With FlashAttention:
    Attention block:   9·s·b·h  (no quadratic term)
    MLP block:        19·s·b·h  (unchanged)
    Total per layer:  ~32·s·b·h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (
    BYTES_FP32,
    ActivationFunction,
    ModelConfig,
    ParallelismConfig,
)


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for a single activation tensor stored for backward."""
    name: str                 # e.g. "qkv_input", "gelu_input"
    block: str                # "attention", "mlp", "layernorm", "residual"
    size_bytes: float         # Memory footprint in bytes
    recompute_flops: float    # FLOPs to recompute this tensor from kept inputs
    recompute_from: list[str] # Names of tensors required to recompute this one
    recomputable: bool = True # False for tensors like RNG masks / FA stats
    description: str = ""     # Human-readable description


def _sbh(cfg: ModelConfig) -> float:
    """Convenience: seq_len * micro_batch_size * hidden_dim."""
    return cfg.seq_len * cfg.micro_batch_size * cfg.hidden_dim


def get_attention_tensors(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> list[TensorInfo]:
    """Return per-tensor activation memory for one attention block.

    Tensors are listed in forward-pass order.  When FlashAttention is
    active, the quadratic softmax/dropout tensors are omitted (FA
    recomputes them on the fly in the backward kernel).
    """
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim
    a = cfg.n_heads
    d_k = cfg.head_dim
    bpe = cfg.dtype_bytes           # bytes per element (usually 2 for bf16)
    tp = par.tp_size
    sp = par.sp_size
    kv_heads = cfg.num_kv_heads

    tensors: list[TensorInfo] = []

    # ── 1. Input to QKV projection (LayerNorm output) ────────────────────
    # Shape: [s, b, h], divided by sp for sequence parallelism
    qkv_input_bytes = s * b * h * bpe / sp
    # Recomputing this = re-running layernorm: negligible FLOPs (5*s*b*h)
    tensors.append(TensorInfo(
        name="attn_qkv_input",
        block="attention",
        size_bytes=qkv_input_bytes,
        recompute_flops=5 * s * b * h / sp,
        recompute_from=["ln1_input"],
        description="Input to QKV projection (post-LayerNorm)",
    ))

    # ── 2. Q tensor ──────────────────────────────────────────────────────
    # Shape: [s, b, h], divided by tp
    q_bytes = s * b * h * bpe / tp
    # Recompute: Q = input @ W_q → 2·s·b·h·h / tp FLOPs (matmul)
    tensors.append(TensorInfo(
        name="attn_q",
        block="attention",
        size_bytes=q_bytes,
        recompute_flops=2 * s * b * h * h / tp,
        recompute_from=["attn_qkv_input"],
        description="Q projection output",
    ))

    # ── 3. K tensor ──────────────────────────────────────────────────────
    # With GQA: shape [s, b, h * kv_heads/n_heads], divided by tp
    k_size = s * b * (h * kv_heads // a) * bpe / tp
    k_flops = 2 * s * b * (h * kv_heads // a) * h / tp
    tensors.append(TensorInfo(
        name="attn_k",
        block="attention",
        size_bytes=k_size,
        recompute_flops=k_flops,
        recompute_from=["attn_qkv_input"],
        description="K projection output",
    ))

    # ── 4. V tensor ──────────────────────────────────────────────────────
    v_size = k_size  # Same shape as K for GQA
    v_flops = k_flops
    tensors.append(TensorInfo(
        name="attn_v",
        block="attention",
        size_bytes=v_size,
        recompute_flops=v_flops,
        recompute_from=["attn_qkv_input"],
        description="V projection output",
    ))

    # ── 5–7. Quadratic attention tensors (ONLY without FlashAttention) ───
    if not cfg.use_flash_attention:
        # Softmax output: [b, a, s, s], divided by tp
        softmax_bytes = b * a * s * s * bpe / tp
        # QK^T matmul + softmax: 2·b·a·s²·d_k + 5·b·a·s² FLOPs
        softmax_flops = (2 * b * a * s * s * d_k + 5 * b * a * s * s) / tp
        tensors.append(TensorInfo(
            name="attn_softmax",
            block="attention",
            size_bytes=softmax_bytes,
            recompute_flops=softmax_flops,
            recompute_from=["attn_q", "attn_k"],
            description="Softmax(QK^T/sqrt(d_k)) attention weights",
        ))

        # Softmax dropout mask: [b, a, s, s], 1 byte per element
        if cfg.use_softmax_dropout:
            dropout_mask_bytes = b * a * s * s * 1 / tp  # 1 byte per bool
            tensors.append(TensorInfo(
                name="attn_softmax_dropout_mask",
                block="attention",
                size_bytes=dropout_mask_bytes,
                recompute_flops=0,  # Masks can't be recomputed (need same RNG)
                recompute_from=[],
                recomputable=False,
                description="Softmax dropout mask (boolean)",
            ))

        # Dropout output (post-dropout attention weights): [b, a, s, s]
        if cfg.use_softmax_dropout:
            dropout_out_bytes = b * a * s * s * bpe / tp
            tensors.append(TensorInfo(
                name="attn_softmax_dropout_output",
                block="attention",
                size_bytes=dropout_out_bytes,
                recompute_flops=softmax_flops + b * a * s * s / tp,
                recompute_from=["attn_q", "attn_k"],
                description="Post-dropout attention weights",
            ))
    else:
        # FlashAttention: stores only the logsumexp for backward
        # Shape: [b, a, s], stored as fp32 (4 bytes)
        fa_lse_bytes = b * a * s * BYTES_FP32 / tp
        tensors.append(TensorInfo(
            name="attn_fa_logsumexp",
            block="attention",
            size_bytes=fa_lse_bytes,
            recompute_flops=0,  # Part of FA kernel; not independently recomputable
            recompute_from=[],
            recomputable=False,
            description="FlashAttention logsumexp statistics",
        ))

    # ── 8. Attention output projection input ─────────────────────────────
    # Shape: [s, b, h], divided by tp
    attn_out_proj_bytes = s * b * h * bpe / tp
    # Recompute = full attention (QK^T·V): 2·b·a·s²·d_k (for QK^T) + 2·b·a·s²·d_k (for ·V)
    attn_out_recompute = 4 * b * a * s * s * d_k / tp
    tensors.append(TensorInfo(
        name="attn_out_proj_input",
        block="attention",
        size_bytes=attn_out_proj_bytes,
        recompute_flops=attn_out_recompute,
        recompute_from=["attn_q", "attn_k", "attn_v"],
        description="Input to attention output projection",
    ))

    # ── 9. Attention output dropout mask ─────────────────────────────────
    if cfg.use_attn_dropout:
        attn_dropout_bytes = s * b * h * 1 / sp  # 1 byte per bool
        tensors.append(TensorInfo(
            name="attn_dropout_mask",
            block="attention",
            size_bytes=attn_dropout_bytes,
            recompute_flops=0,
            recompute_from=[],
            recomputable=False,
            description="Output dropout mask after attention (boolean)",
        ))

    return tensors


def get_mlp_tensors(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> list[TensorInfo]:
    """Return per-tensor activation memory for one MLP block."""
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim
    bpe = cfg.dtype_bytes
    tp = par.tp_size
    sp = par.sp_size
    ffn = cfg.ffn_dim

    tensors: list[TensorInfo] = []

    # ── 1. MLP input (LayerNorm output) ──────────────────────────────────
    mlp_input_bytes = s * b * h * bpe / sp
    tensors.append(TensorInfo(
        name="mlp_input",
        block="mlp",
        size_bytes=mlp_input_bytes,
        recompute_flops=5 * s * b * h / sp,
        recompute_from=["ln2_input"],
        description="Input to first MLP linear (post-LayerNorm)",
    ))

    # ── 2. GeLU/SwiGLU input (output of first linear) ───────────────────
    # Shape: [s, b, ffn_dim] divided by tp
    if cfg.is_swiglu:
        # SwiGLU: gate and up projections, each [s, b, ffn_dim]
        # But the "GeLU input" equivalent is the pre-activation: [s, b, ffn_dim]
        # We need to store BOTH gate and up outputs for backward
        gate_bytes = s * b * ffn * bpe / tp
        up_bytes = s * b * ffn * bpe / tp
        gate_flops = 2 * s * b * h * ffn / tp
        tensors.append(TensorInfo(
            name="mlp_gate_output",
            block="mlp",
            size_bytes=gate_bytes,
            recompute_flops=gate_flops,
            recompute_from=["mlp_input"],
            description="Gate projection output (SwiGLU)",
        ))
        tensors.append(TensorInfo(
            name="mlp_up_output",
            block="mlp",
            size_bytes=up_bytes,
            recompute_flops=gate_flops,  # Same dims as gate
            recompute_from=["mlp_input"],
            description="Up projection output (SwiGLU)",
        ))
    else:
        gelu_input_bytes = s * b * ffn * bpe / tp
        gelu_input_flops = 2 * s * b * h * ffn / tp  # matmul
        tensors.append(TensorInfo(
            name="mlp_gelu_input",
            block="mlp",
            size_bytes=gelu_input_bytes,
            recompute_flops=gelu_input_flops,
            recompute_from=["mlp_input"],
            description="Input to GeLU activation (first linear output)",
        ))

    # ── 3. Second linear input (GeLU/SwiGLU output) ─────────────────────
    linear2_input_bytes = s * b * ffn * bpe / tp
    # Recomputing this requires rerunning first linear + activation
    if cfg.is_swiglu:
        # silu(gate) * up: need gate_output + up_output
        linear2_recompute = 2 * s * b * ffn / tp  # silu + elementwise mul
        recompute_from = ["mlp_gate_output", "mlp_up_output"]
    else:
        linear2_recompute = 8 * s * b * ffn / tp  # GeLU approximation
        recompute_from = ["mlp_gelu_input"]
    tensors.append(TensorInfo(
        name="mlp_linear2_input",
        block="mlp",
        size_bytes=linear2_input_bytes,
        recompute_flops=linear2_recompute,
        recompute_from=recompute_from,
        description="Input to second MLP linear (post-activation output)",
    ))

    # ── 4. MLP output dropout mask ───────────────────────────────────────
    if cfg.use_mlp_dropout:
        dropout_bytes = s * b * h * 1 / sp  # 1 byte per bool
        tensors.append(TensorInfo(
            name="mlp_dropout_mask",
            block="mlp",
            size_bytes=dropout_bytes,
            recompute_flops=0,
            recompute_from=[],
            recomputable=False,
            description="MLP output dropout mask (boolean)",
        ))

    return tensors


def get_layernorm_tensors(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> list[TensorInfo]:
    """Return per-tensor activation memory for LayerNorm layers.

    Two LayerNorms per transformer block: one before attention, one before MLP.
    Each stores its input for the backward pass of the affine parameters.
    LayerNorm inputs are typically stored in fp32 for numerical stability.
    """
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim
    sp = par.sp_size

    # LayerNorm stores input at fp32 precision for gradient accuracy
    ln_bytes = s * b * h * BYTES_FP32 / sp

    return [
        TensorInfo(
            name="ln1_input",
            block="layernorm",
            size_bytes=ln_bytes,
            recompute_flops=0,  # LN1 input = residual stream, always available
            recompute_from=[],
            description="Input to first LayerNorm (before attention)",
        ),
        TensorInfo(
            name="ln2_input",
            block="layernorm",
            size_bytes=ln_bytes,
            recompute_flops=0,  # LN2 input = residual + attn output, always available
            recompute_from=[],
            description="Input to second LayerNorm (before MLP)",
        ),
    ]


def get_all_tensors_per_layer(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> list[TensorInfo]:
    """Return all activation tensors for one transformer layer."""
    return (
        get_layernorm_tensors(cfg, par)
        + get_attention_tensors(cfg, par)
        + get_mlp_tensors(cfg, par)
    )


def get_total_activation_memory_per_layer(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> float:
    """Total activation memory (bytes) for one layer with no checkpointing."""
    return sum(t.size_bytes for t in get_all_tensors_per_layer(cfg, par))


def get_korthikanti_reference(
    cfg: ModelConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> dict[str, float]:
    """Compute Korthikanti's closed-form reference values for validation.

    Returns dict with 'no_ac', 'selective_ac' (recompute attention core),
    and 'full_ac' totals in bytes.
    """
    s, b, h = cfg.seq_len, cfg.micro_batch_size, cfg.hidden_dim
    a = cfg.n_heads
    tp = par.tp_size

    no_ac = s * b * h * (34 + 5 * a * s / h) / tp
    selective_ac = 34 * s * b * h / tp  # Recompute QK^T, softmax, dropout
    full_ac = s * b * h * cfg.dtype_bytes / tp  # Only store layer input

    return {
        "no_ac": no_ac,
        "selective_ac": selective_ac,
        "full_ac": full_ac,
    }
