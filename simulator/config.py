"""Configuration dataclasses for model, GPU hardware, and parallelism.

Extends llm-analysis configs with PCIe bandwidth (for offloading) and
compression-relevant fields.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────

BITS_PER_BYTE = 8
BYTES_FP32 = 4
BYTES_FP16 = 2
BYTES_BF16 = 2
BYTES_FP8 = 1


# ── Enums ────────────────────────────────────────────────────────────────────

class TensorAction(Enum):
    """What to do with an activation tensor during the forward pass."""
    KEEP = "keep"                   # Store in HBM for backward
    RECOMPUTE = "recompute"         # Discard; recompute in backward
    OFFLOAD_CPU = "offload_cpu"     # Async transfer to CPU RAM
    COMPRESS = "compress"           # Low-rank compression in-place


class ActivationFunction(Enum):
    """Non-linearity used in MLP."""
    GELU = "gelu"
    SWIGLU = "swiglu"
    RELU = "relu"


# ── Model Configuration ─────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Transformer model architecture specification."""
    name: str
    num_layers: int
    hidden_dim: int
    n_heads: int
    vocab_size: int
    seq_len: int
    micro_batch_size: int

    # Optional architecture details
    num_kv_heads: Optional[int] = None        # GQA; defaults to n_heads (MHA)
    ffn_dim: Optional[int] = None             # MLP intermediate dim; defaults to 4*hidden_dim
    activation_fn: ActivationFunction = ActivationFunction.GELU
    use_flash_attention: bool = True
    use_softmax_dropout: bool = False         # Dropout after softmax (rare in modern models)
    use_attn_dropout: bool = True             # Dropout after attention output proj
    use_mlp_dropout: bool = False             # Dropout after MLP (rare in modern models)
    use_rotary_embeddings: bool = False       # RoPE (Llama, etc.); creates extra Q/K intermediates
    use_qk_norm: bool = False                 # RMSNorm on Q/K after projection (Qwen3, etc.)
    dtype_bytes: int = BYTES_BF16             # Bytes per activation element

    # MoE
    moe_num_experts: int = 1
    moe_top_k: int = 1

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.n_heads
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.hidden_dim
        assert self.n_heads % self.num_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @property
    def num_kv_groups(self) -> int:
        return self.n_heads // self.num_kv_heads

    @property
    def expansion_ratio(self) -> float:
        return self.ffn_dim / self.hidden_dim

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads != self.n_heads

    @property
    def is_swiglu(self) -> bool:
        return self.activation_fn == ActivationFunction.SWIGLU


# ── GPU / Hardware Configuration ─────────────────────────────────────────────

@dataclass
class GPUConfig:
    """GPU hardware specification including PCIe bandwidth for offloading."""
    name: str
    hbm_capacity_gb: float                    # Total HBM in GB
    hbm_bandwidth_gb_s: float                 # HBM bandwidth in GB/s
    peak_fp16_tflops: float                   # Peak bf16/fp16 tensor TFLOPS
    pcie_bandwidth_gb_s: float                # PCIe bandwidth in GB/s (for offloading)

    # Optional
    nvlink_bandwidth_gb_s: float = 300.0      # Intra-node GPU-GPU bandwidth
    peak_fp8_tflops: Optional[float] = None

    @property
    def hbm_capacity_bytes(self) -> float:
        return self.hbm_capacity_gb * (1024 ** 3)

    @property
    def pcie_bandwidth_bytes_s(self) -> float:
        return self.pcie_bandwidth_gb_s * (1024 ** 3)

    @property
    def peak_flops_per_sec(self) -> float:
        return self.peak_fp16_tflops * 1e12


# ── Parallelism Configuration ────────────────────────────────────────────────

@dataclass
class ParallelismConfig:
    """Parallelism strategy."""
    tp_size: int = 1                          # Tensor parallelism
    pp_size: int = 1                          # Pipeline parallelism
    dp_size: int = 1                          # Data parallelism
    sp_size: Optional[int] = None             # Sequence parallelism (defaults to tp_size)

    def __post_init__(self):
        if self.sp_size is None:
            self.sp_size = self.tp_size


# ── Per-Tensor Decision (what the DP solver will output) ─────────────────────

@dataclass
class TensorDecision:
    """Decision for a single activation tensor."""
    action: TensorAction
    compress_rank: Optional[int] = None       # Only used when action == COMPRESS
    stored_size_bytes: Optional[float] = None # Override retained bytes when storage differs from tensor metadata
    allow_nonrecomputable: bool = False       # For enclosing checkpoint regions that replay opaque ops

    def __post_init__(self):
        if self.action == TensorAction.COMPRESS:
            assert self.compress_rank is not None and self.compress_rank > 0
        if self.stored_size_bytes is not None:
            assert self.stored_size_bytes >= 0


@dataclass
class LayerStrategy:
    """Per-tensor decisions for all activation tensors in a single layer."""
    layer_idx: int
    decisions: dict[str, TensorDecision] = field(default_factory=dict)
    # keys are tensor names like "qkv_input", "gelu_input", etc.


# ── Pre-built Hardware Profiles ──────────────────────────────────────────────

A100_80GB = GPUConfig(
    name="a100-sxm-80gb",
    hbm_capacity_gb=80.0,
    hbm_bandwidth_gb_s=2039.0,
    peak_fp16_tflops=312.0,
    pcie_bandwidth_gb_s=32.0,       # PCIe Gen4 x16 ≈ 32 GB/s
    nvlink_bandwidth_gb_s=300.0,
)

A100_40GB = GPUConfig(
    name="a100-sxm-40gb",
    hbm_capacity_gb=40.0,
    hbm_bandwidth_gb_s=1555.0,
    peak_fp16_tflops=312.0,
    pcie_bandwidth_gb_s=32.0,
    nvlink_bandwidth_gb_s=300.0,
)

H100_80GB = GPUConfig(
    name="h100-sxm-80gb",
    hbm_capacity_gb=80.0,
    hbm_bandwidth_gb_s=3350.0,
    peak_fp16_tflops=989.0,
    pcie_bandwidth_gb_s=64.0,       # PCIe Gen5 x16 ≈ 64 GB/s
    nvlink_bandwidth_gb_s=450.0,
    peak_fp8_tflops=1979.0,
)

H200_141GB = GPUConfig(
    name="h200-sxm-141gb",
    hbm_capacity_gb=141.0,
    hbm_bandwidth_gb_s=4800.0,      # HBM3e, 4.8 TB/s
    peak_fp16_tflops=989.0,         # Same compute as H100
    pcie_bandwidth_gb_s=64.0,       # PCIe Gen5 x16
    nvlink_bandwidth_gb_s=450.0,    # NVLink 4.0
    peak_fp8_tflops=1979.0,
)


# ── Pre-built Model Profiles ────────────────────────────────────────────────

def llama_7b(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="llama-7b",
        num_layers=32,
        hidden_dim=4096,
        n_heads=32,
        num_kv_heads=32,
        ffn_dim=11008,
        vocab_size=32000,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def llama_13b(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="llama-13b",
        num_layers=40,
        hidden_dim=5120,
        n_heads=40,
        num_kv_heads=40,
        ffn_dim=13824,
        vocab_size=32000,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def llama_70b(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="llama-70b",
        num_layers=80,
        hidden_dim=8192,
        n_heads=64,
        num_kv_heads=8,
        ffn_dim=28672,
        vocab_size=32000,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def qwen3_8b(seq_len: int = 4096, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="qwen3-8b",
        num_layers=36,
        hidden_dim=4096,
        n_heads=32,
        num_kv_heads=8,
        ffn_dim=12288,
        vocab_size=151936,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
        use_qk_norm=True,
    )


# ── Realistic large-model configs (verified from technical reports) ──────────

def llama3_70b(seq_len: int = 8192, micro_batch_size: int = 1) -> ModelConfig:
    """Llama-3.1-70B (Meta, 2024).

    Source: HuggingFace config.json + arxiv 2407.21783.
    Training: TP=8, PP=4, seq=8192 (extended to 128K), H100-80GB.
    Architecture: 80 layers, GQA 8:1, SwiGLU, RoPE.
    """
    return ModelConfig(
        name="llama3-70b",
        num_layers=80,
        hidden_dim=8192,
        n_heads=64,
        num_kv_heads=8,
        ffn_dim=28672,
        vocab_size=128256,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_flash_attention=True,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def llama3_405b(seq_len: int = 8192, micro_batch_size: int = 1) -> ModelConfig:
    """Llama-3.1-405B (Meta, 2024).

    Source: HuggingFace config.json + arxiv 2407.21783.
    Training: TP=8, PP=16, DP=128, seq=8192 (extended to 128K), 16384 H100s.
    Architecture: 126 layers, GQA 16:1, SwiGLU, RoPE.
    """
    return ModelConfig(
        name="llama3-405b",
        num_layers=126,
        hidden_dim=16384,
        n_heads=128,
        num_kv_heads=8,
        ffn_dim=53248,
        vocab_size=128256,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.SWIGLU,
        use_flash_attention=True,
        use_attn_dropout=False,
        use_rotary_embeddings=True,
    )


def gpt_neox_20b(seq_len: int = 2048, micro_batch_size: int = 4) -> ModelConfig:
    """GPT-NeoX-20B (EleutherAI, 2022).

    Source: arxiv 2204.06745 + HuggingFace config.json.
    Training: TP=2, PP=4, DP=12, mbs=4, 96 A100-40GB GPUs.
    Architecture: 44 layers, MHA, GeLU, partial RoPE (25%).
    Note: Predates FlashAttention.
    """
    return ModelConfig(
        name="gpt-neox-20b",
        num_layers=44,
        hidden_dim=6144,
        n_heads=64,
        ffn_dim=24576,
        vocab_size=50432,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,
        use_rotary_embeddings=True,
    )


def bloom_176b(seq_len: int = 2048, micro_batch_size: int = 2) -> ModelConfig:
    """BLOOM-176B (BigScience, 2022).

    Source: arxiv 2211.05100 + HuggingFace config.json.
    Training: TP=4, PP=12, DP=8, mbs=2, 384 A100-80GB GPUs.
    Architecture: 70 layers, MHA, GeLU, ALiBi (not RoPE).
    Note: Predates FlashAttention.
    """
    return ModelConfig(
        name="bloom-176b",
        num_layers=70,
        hidden_dim=14336,
        n_heads=112,
        ffn_dim=57344,
        vocab_size=250880,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,
    )


def falcon_180b(seq_len: int = 2048, micro_batch_size: int = 1) -> ModelConfig:
    """Falcon-180B (TII, 2023).

    Source: HuggingFace config.json.
    Training: TP=8, PP=8, DP=64, 4096 A100-80GB GPUs.
    Architecture: 80 layers, GQA 29:1 (8 KV heads), GeLU, RoPE.
    Note: Uses parallel attention+MLP (GPT-J style), which our sequential
    memory model approximates but doesn't perfectly capture.
    """
    return ModelConfig(
        name="falcon-180b",
        num_layers=80,
        hidden_dim=14848,
        n_heads=232,
        num_kv_heads=8,
        ffn_dim=59392,
        vocab_size=65024,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=True,  # Can use FA (GQA-compatible)
        use_rotary_embeddings=True,
    )


def gpt3_175b(seq_len: int = 2048, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="gpt3-175b",
        num_layers=96,
        hidden_dim=12288,
        n_heads=96,
        vocab_size=50257,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,  # Original GPT-3 didn't use FA
    )


def gpt3_22b(seq_len: int = 2048, micro_batch_size: int = 4) -> ModelConfig:
    return ModelConfig(
        name="gpt3-22b",
        num_layers=48,
        hidden_dim=6144,
        n_heads=64,
        vocab_size=50257,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,
    )


def mt_nlg_530b(seq_len: int = 2048, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="mt-nlg-530b",
        num_layers=105,
        hidden_dim=20480,
        n_heads=128,
        vocab_size=50257,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,
    )


def gpt3_1t(seq_len: int = 2048, micro_batch_size: int = 1) -> ModelConfig:
    return ModelConfig(
        name="gpt3-1t",
        num_layers=128,
        hidden_dim=25600,
        n_heads=160,
        vocab_size=50257,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        activation_fn=ActivationFunction.GELU,
        use_flash_attention=False,
    )
