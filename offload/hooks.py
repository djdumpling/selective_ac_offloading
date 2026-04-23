"""Selective CPU-offload hooks backed by `torch.autograd.graph.saved_tensors_hooks`.

Wrap a forward region in `CPUOffloadHook(min_bytes=N)` to asynchronously move
every tensor autograd saves that is >= `min_bytes` to pinned CPU memory.
The unpack hook pulls it back to the original device when backward arrives.

Designed for selective offloading: narrow the hooked region (e.g. just the
`down_proj` call inside `LlamaMLP.forward`) so only one kind of activation
is captured. The simulator's `simulator/offload_model.py` predicts the
PCIe transfer cost and liveness-gap overlap we measure against.

Keep this module light on torch-runtime assumptions — the pure-Python parts
(e.g. the stats dataclass, `should_offload`) must remain importable on any
machine so they can be unit-tested without CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OffloadStats:
    """Counters for what a single `CPUOffloadHook` context saw and moved."""
    tensors_seen: int = 0
    tensors_offloaded: int = 0
    bytes_seen: int = 0
    bytes_offloaded: int = 0
    shapes: list[tuple[tuple[int, ...], str]] = field(default_factory=list)

    def record_seen(self, tensor: torch.Tensor) -> None:
        self.tensors_seen += 1
        self.bytes_seen += tensor.numel() * tensor.element_size()
        self.shapes.append((tuple(tensor.shape), str(tensor.dtype)))

    def record_offloaded(self, tensor: torch.Tensor) -> None:
        self.tensors_offloaded += 1
        self.bytes_offloaded += tensor.numel() * tensor.element_size()


def should_offload(tensor: torch.Tensor, min_bytes: int) -> bool:
    """Predicate: is this saved tensor large enough to be worth moving to CPU?

    Pure function so it's trivial to unit-test. Currently a size threshold;
    could be extended with shape matchers or per-tensor-name filters.
    """
    return tensor.numel() * tensor.element_size() >= min_bytes


def _is_parameter_like(tensor: torch.Tensor) -> bool:
    """True when `tensor` is a Parameter, or a view of one.

    Motivation: `F.linear` saves `W.T` for backward, and that transpose view
    has `requires_grad=True` but `is_leaf=False`, so a plain leaf check misses
    it. We check the view's `_base` too. Parameters and their views don't
    need offloading — they persist on-device anyway.
    """
    if tensor.is_leaf and tensor.requires_grad:
        return True
    base = getattr(tensor, "_base", None)
    if base is not None and base.is_leaf and base.requires_grad:
        return True
    return False


class CPUOffloadHook:
    """Context manager installing saved_tensors_hooks that offload large tensors.

    Usage:
        hook = CPUOffloadHook(min_bytes=1_000_000)
        with hook:
            out = my_module(x)
        out.sum().backward()  # unpack fires here, pulling tensors back

    The same instance can be reused across many forward passes; `stats`
    accumulates across the whole lifetime. Reset explicitly if you want
    per-step counts.
    """

    def __init__(
        self,
        min_bytes: int = 1_000_000,
        pin_memory: bool = True,
        offload_stream: "torch.cuda.Stream | None" = None,
    ):
        if min_bytes < 0:
            raise ValueError(f"min_bytes must be >= 0, got {min_bytes}")
        self.min_bytes = min_bytes
        self.pin_memory = pin_memory
        # A dedicated CUDA stream lets the GPU→CPU DMA run concurrently with
        # compute on the default stream. If None, we fall back to the default
        # stream and transfers serialize with compute.
        self.offload_stream = offload_stream
        self.stats = OffloadStats()
        self._ctx = None

    def _pack(self, tensor: torch.Tensor):
        self.stats.record_seen(tensor)
        if _is_parameter_like(tensor):
            # Parameters and their views (e.g. W.T from F.linear's x @ W.T)
            # live on GPU persistently; offloading them wastes PCIe bandwidth.
            return ("keep", tensor)
        if not should_offload(tensor, self.min_bytes):
            return ("keep", tensor)
        pin = self.pin_memory and tensor.is_cuda
        cpu = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=pin,
        )
        if self.offload_stream is not None and tensor.is_cuda:
            # Make the offload stream wait until the tensor is produced on the
            # default stream, then kick off the DMA on the offload stream.
            self.offload_stream.wait_stream(torch.cuda.current_stream(tensor.device))
            with torch.cuda.stream(self.offload_stream):
                cpu.copy_(tensor, non_blocking=True)
                # Record that the CPU buffer is being written by this stream
                # so unpack can wait on it.
                cpu_ready = self.offload_stream.record_event()
        else:
            cpu.copy_(tensor, non_blocking=True)
            cpu_ready = None
        self.stats.record_offloaded(tensor)
        return ("offload", cpu, tensor.device, cpu_ready)

    def _unpack(self, packed):
        tag = packed[0]
        if tag == "keep":
            return packed[1]
        _, cpu, device, cpu_ready = packed
        if cpu_ready is not None and self.offload_stream is not None:
            # Serve backward on the offload stream so the GPU→CPU DMA has
            # finished before we start CPU→GPU. Then have the default stream
            # wait on the CPU→GPU DMA event.
            self.offload_stream.wait_event(cpu_ready)
            with torch.cuda.stream(self.offload_stream):
                gpu = cpu.to(device, non_blocking=True)
                gpu_ready = self.offload_stream.record_event()
            torch.cuda.current_stream(device).wait_event(gpu_ready)
            return gpu
        return cpu.to(device, non_blocking=True)

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack, self._unpack
        )
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._ctx is not None
        return self._ctx.__exit__(exc_type, exc, tb)

    def reset_stats(self) -> None:
        self.stats = OffloadStats()
