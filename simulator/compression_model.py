"""Low-rank activation compression model.

Estimates memory savings, approximation error, **and compute cost** from
compressing activation tensors via low-rank decomposition.

Reference: LoRAct (Shi et al., 2025) achieves ~80% memory reduction at
rank r = d/8 with bounded error.

For a tensor A of shape [s·b, d]:
- Full storage: s·b·d elements
- Rank-r compressed: store U [s·b, r] and V [r, d] = s·b·r + r·d elements
- Compression ratio: (s·b·r + r·d) / (s·b·d)
- When s·b >> d (typical for activations): ratio ≈ r/d

Compute cost (previously ignored, now modeled):
- Forward (compress):  random projection A @ Ω where Ω is [d, r]
  → 2 × s·b × d × r FLOPs  (one matmul)
- Backward (reconstruct): U @ V^T to recover the approximation
  → 2 × s·b × r × d FLOPs  (one matmul)
- Total round-trip: 4 × s·b × d × r FLOPs
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .memory_model import TensorInfo


@dataclass(frozen=True)
class CompressionResult:
    """Result of compressing a single tensor."""
    tensor_name: str
    original_bytes: float
    compressed_bytes: float
    compression_ratio: float      # compressed / original (lower is better)
    memory_saved_bytes: float
    estimated_relative_error: float
    compress_flops: float         # FLOPs to compress in forward
    decompress_flops: float       # FLOPs to decompress in backward
    total_flops: float            # Round-trip compute cost


def compressed_size(
    rows: int,
    cols: int,
    rank: int,
    bytes_per_element: int = 2,
) -> float:
    """Memory for rank-r decomposition: U [rows, rank] + V [rank, cols]."""
    return (rows * rank + rank * cols) * bytes_per_element


def compression_ratio(rows: int, cols: int, rank: int) -> float:
    """Ratio of compressed size to original size."""
    original = rows * cols
    comp = rows * rank + rank * cols
    return comp / original


def compression_flops(rows: int, cols: int, rank: int) -> tuple[float, float]:
    """Compute cost for compression and decompression.

    Forward (compress): A @ Ω where A is [rows, cols], Ω is [cols, rank]
        → 2 × rows × cols × rank FLOPs

    Backward (decompress): U @ V^T where U is [rows, rank], V is [rank, cols]
        → 2 × rows × rank × cols FLOPs

    Returns (compress_flops, decompress_flops).
    """
    compress = 2 * rows * cols * rank
    decompress = 2 * rows * rank * cols  # same magnitude, different operation
    return compress, decompress


def estimate_error(rank: int, full_dim: int, spectral_decay: float = 1.0) -> float:
    """Estimate relative reconstruction error from low-rank approximation.

    Uses a power-law spectral decay model: sigma_k ∝ k^{-spectral_decay}.
    Higher spectral_decay = faster decay = better compression.

    For the default spectral_decay=1.0 (typical for MLP activations),
    error ≈ sqrt(sum_{k=rank+1}^{d} k^{-2}) / sqrt(sum_{k=1}^{d} k^{-2})

    This is a heuristic; real error depends on the actual spectrum.
    """
    if rank >= full_dim:
        return 0.0

    # Approximate with integral for large d
    # sum k^{-2α} from rank+1 to d ≈ integral
    alpha = spectral_decay
    if alpha == 0:
        # Flat spectrum: error = sqrt((d - rank) / d)
        return sqrt((full_dim - rank) / full_dim)

    # Energy in tail / total energy
    tail_energy = sum(k ** (-2 * alpha) for k in range(rank + 1, min(full_dim + 1, rank + 200)))
    total_energy = sum(k ** (-2 * alpha) for k in range(1, min(full_dim + 1, 500)))

    if total_energy == 0:
        return 0.0
    return sqrt(tail_energy / total_energy)


def compress_tensor(
    tensor: TensorInfo,
    seq_batch: int,
    feature_dim: int,
    rank: int,
    bytes_per_element: int = 2,
    spectral_decay: float = 1.0,
) -> CompressionResult:
    """Analyze compression of an activation tensor.

    Args:
        tensor: The tensor to compress.
        seq_batch: Product of sequence length and batch size (rows).
        feature_dim: Feature/hidden dimension (cols).
        rank: Target rank for decomposition.
        bytes_per_element: Bytes per element (2 for bf16).
        spectral_decay: Spectral decay rate (higher = more compressible).
    """
    comp_bytes = compressed_size(seq_batch, feature_dim, rank, bytes_per_element)
    ratio = compression_ratio(seq_batch, feature_dim, rank)
    error = estimate_error(rank, feature_dim, spectral_decay)
    c_flops, d_flops = compression_flops(seq_batch, feature_dim, rank)

    return CompressionResult(
        tensor_name=tensor.name,
        original_bytes=tensor.size_bytes,
        compressed_bytes=comp_bytes,
        compression_ratio=ratio,
        memory_saved_bytes=max(0.0, tensor.size_bytes - comp_bytes),
        estimated_relative_error=error,
        compress_flops=c_flops,
        decompress_flops=d_flops,
        total_flops=c_flops + d_flops,
    )
