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
from typing import Literal

from .config import GPUConfig, ParallelismConfig
from .memory_model import TensorInfo


SyncMode = Literal["overlap", "serial"]
"""Describes how offload DMAs are scheduled relative to compute.

- "overlap" (default): transfers run on a dedicated CUDA stream with explicit
  stream-event synchronization. DMAs and compute proceed in parallel; stall
  only occurs when the bus is overcommitted within the liveness gap.
  This is what offload/hooks.py::CPUOffloadHook installs when called with a
  non-None offload_stream.
- "serial": transfers run on the default CUDA stream. Each `cpu.copy_` DMA
  blocks the next compute op in issue order, so stall per tensor equals the
  round-trip transfer time regardless of how much liveness gap there is.
  Matches measured +28% overhead at seq=2048 on H200, single-tensor offload.
"""


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
    sync_mode: SyncMode = "overlap",
) -> OffloadResult:
    """Compute full offload analysis for a single tensor.

    See `SyncMode` for the difference between "overlap" and "serial". In
    serial mode, stall = round_trip regardless of `liveness_gap_s`.
    """
    send = transfer_time(tensor.size_bytes, gpu, par)
    recv = transfer_time(tensor.size_bytes, gpu, par)
    rt = send + recv

    if sync_mode == "serial":
        stall = rt
    else:
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


def _earliest_free_slot(
    busy: list[tuple[float, float]],
    duration: float,
    lower_bound: float,
) -> float:
    """Earliest t >= lower_bound such that [t, t+duration] overlaps none of `busy`.

    `busy` must be sorted by start time and non-overlapping.
    """
    candidate = lower_bound
    for start, end in busy:
        if end <= candidate:
            continue  # interval is entirely before our candidate, skip
        if start >= candidate + duration:
            return candidate  # fits in the gap [candidate, start]
        candidate = end  # interval blocks us; jump past it
    return candidate


def _latest_free_slot_by_deadline(
    busy: list[tuple[float, float]],
    duration: float,
    lower_bound: float,
    deadline: float,
) -> float | None:
    """Latest t >= lower_bound such that [t, t+duration] is free and t+duration <= deadline.

    Returns None if no such slot exists. `busy` must be sorted by start time
    and non-overlapping. Used to schedule recvs ALAP (finishing at the deadline).
    """
    if deadline - lower_bound < duration:
        return None
    # Walk busy intervals right-to-left, tracking the latest end of a free slot.
    candidate_end = deadline
    for start, end in reversed(busy):
        if start >= candidate_end:
            continue  # interval is entirely after our candidate window, skip
        gap_start = max(end, lower_bound)
        if candidate_end - gap_start >= duration:
            return candidate_end - duration  # fits in [gap_start, candidate_end]
        candidate_end = min(candidate_end, start)
        if candidate_end - lower_bound < duration:
            return None
    # Final gap: [lower_bound, candidate_end]
    if candidate_end - lower_bound >= duration:
        return candidate_end - duration
    return None


def _insert_interval(busy: list[tuple[float, float]], new: tuple[float, float]) -> None:
    """Insert `new` into `busy` preserving sort-by-start order.

    Caller must ensure `new` doesn't overlap any existing interval.
    """
    lo, hi = 0, len(busy)
    while lo < hi:
        mid = (lo + hi) // 2
        if busy[mid][0] < new[0]:
            lo = mid + 1
        else:
            hi = mid
    busy.insert(lo, new)


def schedule_offloads(
    tensors: list[tuple[TensorInfo, float]],
    gpu: GPUConfig,
    par: ParallelismConfig = ParallelismConfig(),
    sync_mode: SyncMode = "overlap",
) -> list[OffloadResult]:
    """Schedule offload transfers on a shared half-duplex PCIe bus.

    Each tensor needs a send (GPU → CPU) at/after creation and a recv
    (CPU → GPU) completing by its backward deadline (= liveness gap).
    The bus can only service one transfer at a time, so we track busy
    intervals and place each transfer in the best available slot:

    - Sends go in the earliest free slot starting at t >= 0, so memory
      is freed as early as possible and the queue drains quickly.
    - Recvs go in the latest free slot whose end <= deadline (ALAP),
      so they don't unnecessarily block earlier-deadline recvs. When
      no such slot exists, the recv is pushed past the deadline,
      producing stall = recv_end - deadline.

    Tensors are processed in EDF (earliest-deadline-first) order. With
    zero release times for all sends, EDF with ALAP recv placement is
    optimal for the agreeable-deadlines job-scheduling problem. Processing
    longest-deadline-first (the previous order) over-committed the bus to
    the loose tensor and starved tighter ones — e.g., at gaps [t, 3t]
    longest-first predicts 3t stall while EDF predicts 2t.

    In `sync_mode="serial"`, the bus scheduler is bypassed entirely —
    each tensor incurs stall = round_trip because DMAs serialize with
    compute on the default CUDA stream. See `SyncMode`.

    Returns offload results sorted by scheduling order.
    """
    if sync_mode == "serial":
        # Default-stream behavior: each offload blocks compute for its full
        # round-trip regardless of liveness slack. Short-circuit the scheduler.
        eff_bw = effective_pcie_bandwidth(gpu, par)
        out: list[OffloadResult] = []
        for tensor, _gap in tensors:
            send = transfer_time(tensor.size_bytes, gpu, par)
            recv = transfer_time(tensor.size_bytes, gpu, par)
            out.append(OffloadResult(
                tensor_name=tensor.name,
                size_bytes=tensor.size_bytes,
                send_time_s=send,
                recv_time_s=recv,
                round_trip_s=send + recv,
                memory_freed_bytes=tensor.size_bytes,
                stall_time_s=send + recv,
                effective_bw_gb_s=eff_bw / (1024 ** 3),
            ))
        return out

    # EDF: earliest-deadline-first. Tightest deadlines take priority on the bus;
    # looser deadlines pack in around them.
    sorted_tensors = sorted(tensors, key=lambda x: x[1])
    busy: list[tuple[float, float]] = []  # sorted by start, non-overlapping
    results = []
    eff_bw = effective_pcie_bandwidth(gpu, par)

    for tensor, gap in sorted_tensors:
        send = transfer_time(tensor.size_bytes, gpu, par)
        recv = transfer_time(tensor.size_bytes, gpu, par)

        # Send: earliest free slot starting at t >= 0.
        send_start = _earliest_free_slot(busy, send, lower_bound=0.0)
        send_end = send_start + send
        _insert_interval(busy, (send_start, send_end))

        # Recv: latest free slot ending by the deadline, starting after send.
        # If none fits, schedule it ASAP after the bus clears — this is the
        # stall case.
        recv_start = _latest_free_slot_by_deadline(
            busy, recv, lower_bound=send_end, deadline=gap,
        )
        if recv_start is None:
            recv_start = _earliest_free_slot(busy, recv, lower_bound=send_end)
        recv_end = recv_start + recv
        _insert_interval(busy, (recv_start, recv_end))

        stall = max(0.0, recv_end - gap)

        results.append(OffloadResult(
            tensor_name=tensor.name,
            size_bytes=tensor.size_bytes,
            send_time_s=send,
            recv_time_s=recv,
            round_trip_s=send + recv,
            memory_freed_bytes=tensor.size_bytes,
            stall_time_s=stall,
            effective_bw_gb_s=eff_bw / (1024 ** 3),
        ))

    return results
