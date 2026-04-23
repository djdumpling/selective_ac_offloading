"""CPU-runnable tests for offload/hooks.py.

These tests import torch but don't require CUDA — the hook mechanism itself
is exercised with plain CPU tensors (pin_memory is conditional on tensor.is_cuda
inside the hook).
"""

import pytest

torch = pytest.importorskip("torch")

from offload.hooks import CPUOffloadHook, OffloadStats, _is_parameter_like, should_offload


class TestShouldOffload:
    def test_below_threshold_returns_false(self):
        t = torch.zeros(10, dtype=torch.float32)  # 40 bytes
        assert should_offload(t, min_bytes=1024) is False

    def test_above_threshold_returns_true(self):
        t = torch.zeros(512, dtype=torch.float32)  # 2048 bytes
        assert should_offload(t, min_bytes=1024) is True

    def test_exact_threshold_returns_true(self):
        t = torch.zeros(256, dtype=torch.float32)  # 1024 bytes
        assert should_offload(t, min_bytes=1024) is True


class TestCPUOffloadHookValidation:
    def test_negative_min_bytes_raises(self):
        with pytest.raises(ValueError):
            CPUOffloadHook(min_bytes=-1)


class TestCPUOffloadHookStats:
    def test_initial_stats_zero(self):
        h = CPUOffloadHook(min_bytes=1024)
        assert h.stats.tensors_seen == 0
        assert h.stats.bytes_offloaded == 0

    def test_reset_stats(self):
        h = CPUOffloadHook(min_bytes=1024)
        h.stats.tensors_seen = 5
        h.stats.bytes_offloaded = 100
        h.reset_stats()
        assert h.stats.tensors_seen == 0
        assert h.stats.bytes_offloaded == 0


class TestPackUnpackRoundtrip:
    """Call the hook's pack/unpack functions directly. Doesn't engage autograd;
    verifies the packaging contract."""

    def test_small_tensor_kept_in_place(self):
        h = CPUOffloadHook(min_bytes=10_000)
        t = torch.randn(10)  # 40 bytes
        packed = h._pack(t)
        assert packed[0] == "keep"
        assert torch.equal(packed[1], t)
        assert h.stats.tensors_offloaded == 0
        assert h.stats.tensors_seen == 1

    def test_large_tensor_offloaded(self):
        h = CPUOffloadHook(min_bytes=100)
        t = torch.arange(256, dtype=torch.float32)  # 1024 bytes
        packed = h._pack(t)
        # packed = ("offload", cpu_tensor, device, cpu_ready_event|None)
        assert packed[0] == "offload"
        assert packed[1].device.type == "cpu"
        assert packed[1].shape == t.shape
        assert packed[2] == t.device
        assert packed[3] is None  # no offload stream configured → no event
        assert h.stats.tensors_offloaded == 1
        assert h.stats.bytes_offloaded == 1024

    def test_roundtrip_preserves_values(self):
        h = CPUOffloadHook(min_bytes=100)
        t = torch.randn(512, dtype=torch.float32)
        packed = h._pack(t)
        restored = h._unpack(packed)
        assert torch.allclose(restored, t)
        assert restored.shape == t.shape
        assert restored.dtype == t.dtype


class TestAutogradIntegration:
    """Autograd actually calls the hook around a realistic save. Confirms
    saved_tensors_hooks fires from inside a `with CPUOffloadHook():` block."""

    def test_hook_fires_during_matmul_save(self):
        h = CPUOffloadHook(min_bytes=0)  # offload everything
        x = torch.randn(8, 8, requires_grad=True)
        w = torch.randn(8, 8, requires_grad=True)
        with h:
            y = x @ w
            loss = y.sum()
        loss.backward()
        assert h.stats.tensors_seen > 0

    def test_parameters_are_not_offloaded(self):
        """Leaves with requires_grad (= Parameters) must stay put. Offloading
        them is pointless (they already live on-device) and saturates PCIe."""
        h = CPUOffloadHook(min_bytes=0)
        x = torch.randn(8, 8, requires_grad=True)  # a leaf with requires_grad
        packed = h._pack(x)
        assert packed[0] == "keep"
        assert h.stats.tensors_seen == 1
        assert h.stats.tensors_offloaded == 0

    def test_nonleaf_activations_are_offloaded(self):
        """Non-leaf intermediates (outputs of ops) flow to CPU as expected."""
        h = CPUOffloadHook(min_bytes=0)
        x = torch.randn(8, 8, requires_grad=True)
        y = x * 2  # non-leaf
        packed = h._pack(y)
        assert packed[0] == "offload"

    def test_weight_transpose_view_not_offloaded(self):
        """F.linear saves `W.T` for backward — a non-leaf view of a Parameter.
        Previous filter (is_leaf & requires_grad) missed it; _base check catches it."""
        import torch.nn as nn
        w = nn.Parameter(torch.randn(4, 8))
        w_t = w.T
        assert not w_t.is_leaf  # view of a leaf is non-leaf when it has a grad_fn
        assert _is_parameter_like(w_t)
        h = CPUOffloadHook(min_bytes=0)
        packed = h._pack(w_t)
        assert packed[0] == "keep"

    def test_parameter_itself_not_offloaded(self):
        import torch.nn as nn
        w = nn.Parameter(torch.randn(4, 8))
        assert _is_parameter_like(w)

    def test_regular_view_of_activation_still_offloaded(self):
        """A reshape/view of an intermediate activation should still offload —
        the _base check targets leaves-with-grad only."""
        h = CPUOffloadHook(min_bytes=0)
        x = torch.randn(4, 8, requires_grad=True)
        activation = (x * 2).reshape(8, 4)  # view of a non-leaf
        packed = h._pack(activation)
        assert packed[0] == "offload"

    def test_gradient_values_match_without_hook(self):
        """The round-trip must be bit-exact for fp32: packing is just a copy."""
        torch.manual_seed(0)
        x1 = torch.randn(16, 16, requires_grad=True)
        w1 = torch.randn(16, 16, requires_grad=True)
        y1 = x1 @ w1
        y1.sum().backward()
        grad_x_noop = x1.grad.clone()
        grad_w_noop = w1.grad.clone()

        torch.manual_seed(0)
        x2 = torch.randn(16, 16, requires_grad=True)
        w2 = torch.randn(16, 16, requires_grad=True)
        h = CPUOffloadHook(min_bytes=0)
        with h:
            y2 = x2 @ w2
        y2.sum().backward()

        assert torch.allclose(x2.grad, grad_x_noop)
        assert torch.allclose(w2.grad, grad_w_noop)
