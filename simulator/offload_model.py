"""PCIe offload transfer model.

Models async CPU offloading of activation tensors:
- Transfer time = tensor_size / pcie_bandwidth  (one-way)
- Round-trip = 2 × one-way  (send to CPU in forward, fetch back in backward)
- Overlap feasibility: transfer can overlap with subsequent forward compute

Key insight from knowledge base (Proposal 4.4): tensors created earlier
in forward have later deadlines in backward (agreeable deadlines),
enabling optimal O(n log n) scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GPUConfig
from .memory_model import TensorInfo


@dataclass(frozen=True)
class OffloadResult:
    """Result of offloading a single tensor."""
    tensor_name: str
    size_bytes: float
    send_time_s: float          # Time to transfer GPU → CPU
    recv_time_s: float          # Time to transfer CPU → GPU
    round_trip_s: float         # Total transfer time
    memory_freed_bytes: float   # HBM freed during the liveness gap
    stall_time_s: float         # GPU stall if transfer doesn't overlap


def transfer_time(size_bytes: float, gpu: GPUConfig) -> float:
    """One-way PCIe transfer time in seconds."""
    if gpu.pcie_bandwidth_bytes_s == 0:
        return float("inf")
    return size_bytes / gpu.pcie_bandwidth_bytes_s


def round_trip_time(size_bytes: float, gpu: GPUConfig) -> float:
    """Round-trip (send + receive) PCIe transfer time."""
    return 2 * transfer_time(size_bytes, gpu)


def can_overlap(
    tensor: TensorInfo,
    liveness_gap_s: float,
    gpu: GPUConfig,
) -> bool:
    """Check if offloading this tensor can fully overlap with compute.

    The liveness gap is the wall-clock time between when the tensor is
    created (forward) and when it's needed (backward).  If the round-trip
    PCIe transfer fits within this gap, offloading is free.
    """
    return round_trip_time(tensor.size_bytes, gpu) <= liveness_gap_s


def compute_offload_result(
    tensor: TensorInfo,
    liveness_gap_s: float,
    gpu: GPUConfig,
) -> OffloadResult:
    """Compute full offload analysis for a single tensor."""
    send = transfer_time(tensor.size_bytes, gpu)
    recv = transfer_time(tensor.size_bytes, gpu)
    rt = send + recv

    # Stall = how much the GPU must wait beyond the liveness gap
    stall = max(0.0, rt - liveness_gap_s)

    return OffloadResult(
        tensor_name=tensor.name,
        size_bytes=tensor.size_bytes,
        send_time_s=send,
        recv_time_s=recv,
        round_trip_s=rt,
        memory_freed_bytes=tensor.size_bytes,
        stall_time_s=stall,
    )


def schedule_offloads(
    tensors: list[tuple[TensorInfo, float]],
    gpu: GPUConfig,
) -> list[OffloadResult]:
    """Schedule offload transfers for a list of (tensor, liveness_gap) pairs.

    Uses the agreeable-deadline property: process tensors in decreasing
    liveness gap order (longest gap first).  Greedily assign PCIe time
    slots.  This is optimal for the weighted job scheduling problem with
    agreeable deadlines.

    Returns offload results sorted by scheduling order.
    """
    # Sort by liveness gap descending (longest gap → most likely to overlap)
    sorted_tensors = sorted(tensors, key=lambda x: x[1], reverse=True)

    results = []
    pcie_busy_until = 0.0  # Tracks PCIe bus availability (in wall-clock time)

    for tensor, gap in sorted_tensors:
        send = transfer_time(tensor.size_bytes, gpu)
        recv = transfer_time(tensor.size_bytes, gpu)

        # Send starts when PCIe is free (after tensor is created)
        send_start = max(0.0, pcie_busy_until)
        send_end = send_start + send
        pcie_busy_until = send_end

        # Check if the receive can complete before the tensor is needed
        # The tensor is needed at time = gap (from its creation)
        # Receive must start early enough: recv_start + recv <= gap
        recv_start = gap - recv
        stall = max(0.0, send_end - recv_start + recv - gap) if recv_start < send_end else 0.0

        results.append(OffloadResult(
            tensor_name=tensor.name,
            size_bytes=tensor.size_bytes,
            send_time_s=send,
            recv_time_s=recv,
            round_trip_s=send + recv,
            memory_freed_bytes=tensor.size_bytes,
            stall_time_s=max(0.0, (send + recv) - gap),
        ))

    return results
