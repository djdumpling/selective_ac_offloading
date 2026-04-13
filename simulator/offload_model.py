"""PCIe offload transfer model with NCCL contention awareness.

Models async CPU offloading of activation tensors:
- Transfer time = tensor_size / effective_pcie_bandwidth  (one-way)
- Round-trip = 2 × one-way  (send to CPU in forward, fetch back in backward)
- Overlap feasibility: transfer can overlap with subsequent forward compute

PCIe contention (previously ignored, now modeled):
- With FSDP (dp_size > 1), each layer's forward/backward involves
  all-gather + reduce-scatter of parameter shards.
- If these collectives transit PCIe (multi-node or non-NVLink intra-node),
  they compete with offload transfers for PCIe bandwidth.
- Effective offload bandwidth = raw_pcie_bw × (1 - nccl_pcie_utilization)

Key insight from knowledge base (Proposal 4.4): tensors created earlier
in forward have later deadlines in backward (agreeable deadlines),
enabling optimal O(n log n) scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GPUConfig, ParallelismConfig
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
    effective_bw_gb_s: float    # Effective PCIe bandwidth used (after contention)


def estimate_nccl_pcie_utilization(
    gpu: GPUConfig,
    par: ParallelismConfig,
) -> float:
    """Estimate fraction of PCIe bandwidth consumed by NCCL collectives.

    FSDP all-gather and reduce-scatter move parameter shards every
    forward+backward step.  When GPUs communicate over NVLink (intra-node
    TP/FSDP), PCIe is free for offloading.  When they communicate over
    PCIe (inter-node or non-NVLink setups), NCCL traffic contends.

    Returns a value in [0, 1) representing the fraction of PCIe bandwidth
    used by NCCL, leaving the rest for offloading.
    """
    if par.dp_size <= 1 and par.tp_size <= 1:
        return 0.0  # No collectives → PCIe is fully available

    # Heuristic: NCCL traffic is proportional to the fraction of time
    # spent in communication vs. compute.  For FSDP, each layer does
    # an all-gather (params) in forward and reduce-scatter (grads) in
    # backward.  On NVLink systems, this doesn't touch PCIe.

    nvlink_bw = gpu.nvlink_bandwidth_gb_s
    pcie_bw = gpu.pcie_bandwidth_gb_s

    if nvlink_bw > 0 and par.tp_size > 1:
        # TP typically runs over NVLink intra-node.
        # FSDP can run over NVLink (single-node) or PCIe (multi-node).
        # Conservative: assume single-node FSDP uses NVLink too.
        total_gpus = par.tp_size * par.dp_size * par.pp_size
        if total_gpus <= 8:
            # Likely single-node: NVLink handles everything
            return 0.0

    # Multi-node FSDP: NCCL uses PCIe for inter-node traffic.
    # Empirical estimate: FSDP all-gather + reduce-scatter consumes
    # roughly 30-60% of PCIe bandwidth during a training step,
    # depending on model size and compute-to-communication ratio.
    # Use 0.4 as a moderate estimate for typical large-model training.
    if par.dp_size > 1:
        return 0.4

    return 0.0


def effective_pcie_bandwidth(
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> float:
    """Effective PCIe bandwidth (bytes/s) available for offloading.

    Accounts for NCCL contention when using FSDP or other collective-heavy
    parallelism strategies.
    """
    nccl_frac = estimate_nccl_pcie_utilization(gpu, par)
    available_frac = max(0.0, 1.0 - nccl_frac)
    return gpu.pcie_bandwidth_bytes_s * available_frac


def transfer_time(
    size_bytes: float,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> float:
    """One-way PCIe transfer time in seconds, accounting for contention."""
    eff_bw = effective_pcie_bandwidth(gpu, par)
    if eff_bw == 0:
        return float("inf")
    return size_bytes / eff_bw


def round_trip_time(
    size_bytes: float,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> float:
    """Round-trip (send + receive) PCIe transfer time."""
    return 2 * transfer_time(size_bytes, gpu, par)


def can_overlap(
    tensor: TensorInfo,
    liveness_gap_s: float,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> bool:
    """Check if offloading this tensor can fully overlap with compute."""
    return round_trip_time(tensor.size_bytes, gpu, par) <= liveness_gap_s


def compute_offload_result(
    tensor: TensorInfo,
    liveness_gap_s: float,
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> OffloadResult:
    """Compute full offload analysis for a single tensor."""
    send = transfer_time(tensor.size_bytes, gpu, par)
    recv = transfer_time(tensor.size_bytes, gpu, par)
    rt = send + recv

    stall = max(0.0, rt - liveness_gap_s)
    eff_bw = effective_pcie_bandwidth(gpu, par)

    return OffloadResult(
        tensor_name=tensor.name,
        size_bytes=tensor.size_bytes,
        send_time_s=send,
        recv_time_s=recv,
        round_trip_s=rt,
        memory_freed_bytes=tensor.size_bytes,
        stall_time_s=stall,
        effective_bw_gb_s=eff_bw / (1024 ** 3),
    )


def schedule_offloads(
    tensors: list[tuple[TensorInfo, float]],
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
) -> list[OffloadResult]:
    """Schedule offload transfers for a list of (tensor, liveness_gap) pairs.

    Uses the agreeable-deadline property: process tensors in decreasing
    liveness gap order (longest gap first).  Greedily assign PCIe time
    slots.  This is optimal for the weighted job scheduling problem with
    agreeable deadlines.

    Returns offload results sorted by scheduling order.
    """
    sorted_tensors = sorted(tensors, key=lambda x: x[1], reverse=True)

    results = []
    pcie_busy_until = 0.0

    for tensor, gap in sorted_tensors:
        send = transfer_time(tensor.size_bytes, gpu, par)
        recv = transfer_time(tensor.size_bytes, gpu, par)

        send_start = max(0.0, pcie_busy_until)
        send_end = send_start + send
        pcie_busy_until = send_end

        recv_start = gap - recv
        stall = max(0.0, send_end - recv_start + recv - gap) if recv_start < send_end else 0.0

        eff_bw = effective_pcie_bandwidth(gpu, par)
        results.append(OffloadResult(
            tensor_name=tensor.name,
            size_bytes=tensor.size_bytes,
            send_time_s=send,
            recv_time_s=recv,
            round_trip_s=send + recv,
            memory_freed_bytes=tensor.size_bytes,
            stall_time_s=max(0.0, (send + recv) - gap),
            effective_bw_gb_s=eff_bw / (1024 ** 3),
        ))

    return results
