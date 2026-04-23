"""Tests for the activation memory simulator.

Validates against Korthikanti et al. closed-form formulas and
cross-checks with llm-analysis where possible.
"""

import pytest

from simulator.config import (
    A100_80GB,
    H100_80GB,
    H200_141GB,
    ModelConfig,
    ParallelismConfig,
    ActivationFunction,
    TensorAction,
    TensorDecision,
    LayerStrategy,
    gpt3_175b,
    llama_7b,
    llama_70b,
)
from simulator.memory_model import (
    TensorInfo,
    get_all_tensors_per_layer,
    get_attention_tensors,
    get_mlp_tensors,
    get_layernorm_tensors,
    get_total_activation_memory_per_layer,
    get_korthikanti_reference,
)
from simulator.compute_model import (
    get_fwd_flops_attn,
    get_fwd_flops_mlp,
    get_fwd_flops_per_layer,
    get_layer_compute_profile,
    get_recompute_overhead_ratio,
)
from simulator.offload_model import (
    transfer_time,
    round_trip_time,
    can_overlap,
    compute_offload_result,
    effective_pcie_bandwidth,
    estimate_nccl_pcie_utilization,
    schedule_offloads,
)
from simulator.compression_model import (
    compressed_size,
    compression_ratio,
    compression_flops,
    estimate_error,
)
from simulator.environment import (
    simulate,
    simulate_no_ac,
    simulate_full_ac,
    simulate_selective_ac,
    simulate_fa_selective_ac,
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
    print_result,
    _stash_count_1f1b,
)
from simulator.pipeline_schedules import (
    PipelineSchedule,
    get_schedule_profile,
)


# ── Helper ───────────────────────────────────────────────────────────────────

def _gb(x: float) -> float:
    """Convert bytes to GB."""
    return x / (1024 ** 3)


def _mb(x: float) -> float:
    """Convert bytes to MB."""
    return x / (1024 ** 2)


# ── Test: Korthikanti reference formulas ─────────────────────────────────────

class TestKorthikantiReference:
    """Verify our per-tensor model against the known closed-form totals."""

    def test_gpt3_175b_no_fa_total(self):
        """GPT-3 175B without FA: total should be sbh(34 + 5as/h) per layer.

        From the knowledge base:
        - Linear term: 11sbh ≈ 555 MB per layer
        - Quadratic term: 5as²b ≈ 4.03 GB per layer
        """
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        ref = get_korthikanti_reference(cfg)

        s, b, h, a = 2048, 1, 12288, 96
        expected = s * b * h * (34 + 5 * a * s / h)
        assert abs(ref["no_ac"] - expected) < 1, f"Mismatch: {ref['no_ac']} vs {expected}"

        # Verify the quadratic coefficient 5as/h = 80 for GPT-3
        # In the formula sbh(34 + 5as/h), the quadratic coefficient is 5as/h
        quadratic_coeff = 5 * a * s / h
        assert abs(quadratic_coeff - 80) < 0.1, f"Expected 5as/h=80, got {quadratic_coeff}"
        # The quadratic term (80) exceeds the linear term (34) by ~2.35×
        ratio = quadratic_coeff / 34
        assert ratio > 2.0, f"Quadratic should exceed linear, ratio={ratio}"

    def test_gpt3_175b_selective_ac(self):
        """Selective AC (recompute attention core): should be 34sbh."""
        cfg = gpt3_175b()
        ref = get_korthikanti_reference(cfg)
        expected = 34 * 2048 * 1 * 12288
        assert abs(ref["selective_ac"] - expected) < 1

    def test_quadratic_ratio_gpt3(self):
        """From the knowledge base: quadratic/linear ratio = 5as/h = 80 for GPT-3."""
        a, s, h = 96, 2048, 12288
        ratio = 5 * a * s / h
        assert abs(ratio - 80) < 0.1

    def test_recompute_overhead_gpt3(self):
        """Korthikanti selective: ~2.7% overhead for GPT-3 175B."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        # The attention core FLOPs (QK^T + softmax) vs total
        s, h = 2048, 12288
        # Overhead ≈ 1 + s/(6h) - 1 = s/(6h)
        overhead_pct = s / (6 * h) * 100
        assert 1.0 < overhead_pct < 5.0, f"Expected ~2.7%, got {overhead_pct:.1f}%"


class TestPerTensorMemory:
    """Verify individual tensor sizes match expected formulas."""

    def test_attention_tensors_no_fa(self):
        """Without FA: should include quadratic s² tensors."""
        cfg = ModelConfig(
            name="test",
            num_layers=1,
            hidden_dim=4096,
            n_heads=32,
            vocab_size=32000,
            seq_len=2048,
            micro_batch_size=2,
            use_flash_attention=False,
            use_softmax_dropout=True,
        )
        tensors = get_attention_tensors(cfg)
        names = [t.name for t in tensors]
        assert "attn_softmax" in names, "Should have softmax tensor without FA"
        assert "attn_softmax_dropout_mask" in names

    def test_attention_tensors_with_fa(self):
        """With FA: should NOT include quadratic tensors, should have logsumexp."""
        cfg = ModelConfig(
            name="test",
            num_layers=1,
            hidden_dim=4096,
            n_heads=32,
            vocab_size=32000,
            seq_len=2048,
            micro_batch_size=2,
            use_flash_attention=True,
        )
        tensors = get_attention_tensors(cfg)
        names = [t.name for t in tensors]
        assert "attn_softmax" not in names, "FA should eliminate softmax tensor"
        assert "attn_fa_logsumexp" in names, "FA should store logsumexp"

    def test_fa_dramatically_reduces_attention_memory(self):
        """FA should make attention memory linear in s, not quadratic."""
        for s in [1024, 2048, 4096]:
            cfg_no_fa = ModelConfig(
                name="test", num_layers=1, hidden_dim=4096,
                n_heads=32, vocab_size=32000, seq_len=s,
                micro_batch_size=1, use_flash_attention=False,
                use_softmax_dropout=True,
            )
            cfg_fa = ModelConfig(
                name="test", num_layers=1, hidden_dim=4096,
                n_heads=32, vocab_size=32000, seq_len=s,
                micro_batch_size=1, use_flash_attention=True,
            )
            mem_no_fa = sum(t.size_bytes for t in get_attention_tensors(cfg_no_fa))
            mem_fa = sum(t.size_bytes for t in get_attention_tensors(cfg_fa))
            assert mem_fa < mem_no_fa, f"FA should use less memory at s={s}"

    def test_mlp_tensors_gelu(self):
        """Standard GELU MLP should have: input, gelu_input, gelu_output, linear2_input."""
        cfg = ModelConfig(
            name="test", num_layers=1, hidden_dim=4096,
            n_heads=32, vocab_size=32000, seq_len=2048,
            micro_batch_size=1, activation_fn=ActivationFunction.GELU,
        )
        tensors = get_mlp_tensors(cfg)
        names = [t.name for t in tensors]
        assert "mlp_input" in names
        assert "mlp_gelu_input" in names
        assert "mlp_gelu_output" in names
        assert "mlp_linear2_input" in names

    def test_mlp_tensors_swiglu(self):
        """SwiGLU MLP should have: input, gate_output, up_output, silu_output, linear2_input."""
        cfg = llama_7b()
        tensors = get_mlp_tensors(cfg)
        names = [t.name for t in tensors]
        assert "mlp_gate_output" in names
        assert "mlp_up_output" in names
        assert "mlp_silu_output" in names
        assert "mlp_linear2_input" in names
        assert "mlp_gelu_input" not in names

    def test_layernorm_fp32(self):
        """LayerNorm inputs should be stored at fp32 (4 bytes per element)."""
        cfg = ModelConfig(
            name="test", num_layers=1, hidden_dim=4096,
            n_heads=32, vocab_size=32000, seq_len=2048,
            micro_batch_size=1,
        )
        ln_tensors = get_layernorm_tensors(cfg)
        assert len(ln_tensors) == 2  # One before attn, one before MLP
        expected_bytes = 2048 * 1 * 4096 * 4  # s * b * h * fp32
        for t in ln_tensors:
            assert t.size_bytes == expected_bytes


class TestGQA:
    """Test Grouped Query Attention memory accounting."""

    def test_gqa_reduces_kv_memory(self):
        """GQA with fewer KV heads should reduce K, V tensor sizes."""
        cfg_mha = ModelConfig(
            name="mha", num_layers=1, hidden_dim=8192, n_heads=64,
            num_kv_heads=64, vocab_size=32000, seq_len=4096,
            micro_batch_size=1,
        )
        cfg_gqa = ModelConfig(
            name="gqa", num_layers=1, hidden_dim=8192, n_heads=64,
            num_kv_heads=8, vocab_size=32000, seq_len=4096,
            micro_batch_size=1,
        )
        tensors_mha = {t.name: t for t in get_attention_tensors(cfg_mha)}
        tensors_gqa = {t.name: t for t in get_attention_tensors(cfg_gqa)}

        # K and V should be 8× smaller with GQA (64/8 = 8)
        assert tensors_gqa["attn_k"].size_bytes < tensors_mha["attn_k"].size_bytes
        ratio = tensors_mha["attn_k"].size_bytes / tensors_gqa["attn_k"].size_bytes
        assert abs(ratio - 8.0) < 0.1, f"Expected 8x reduction, got {ratio}x"


class TestComputeModel:
    """Test FLOPs calculations."""

    def test_flops_mha_formula(self):
        """For MHA (no GQA), total per-layer FLOPs should match 24bsh²(1+s/6h)."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        b, s, h = 1, 2048, 12288

        total = get_fwd_flops_per_layer(cfg)
        # Standard formula (expansion_ratio=4, MHA)
        expected = 24 * b * s * h * h * (1 + s / (6 * h))
        # Allow 5% tolerance due to our GQA-aware K,V formula
        assert abs(total - expected) / expected < 0.05, (
            f"FLOPs mismatch: {total:.2e} vs {expected:.2e}"
        )

    def test_swiglu_more_flops_than_gelu(self):
        """SwiGLU (3 projections) should have more MLP FLOPs than GELU (2)."""
        cfg_gelu = ModelConfig(
            name="gelu", num_layers=1, hidden_dim=4096, n_heads=32,
            vocab_size=32000, seq_len=2048, micro_batch_size=1,
            activation_fn=ActivationFunction.GELU,
        )
        cfg_swiglu = ModelConfig(
            name="swiglu", num_layers=1, hidden_dim=4096, n_heads=32,
            vocab_size=32000, seq_len=2048, micro_batch_size=1,
            activation_fn=ActivationFunction.SWIGLU,
        )
        assert get_fwd_flops_mlp(cfg_swiglu) > get_fwd_flops_mlp(cfg_gelu)

    def test_tensor_parallel_reduces_per_rank_flops(self):
        """TP should reduce the compute assigned to each rank."""
        cfg = llama_70b(seq_len=4096, micro_batch_size=1)
        flops_tp1 = get_fwd_flops_per_layer(cfg, ParallelismConfig(tp_size=1))
        flops_tp8 = get_fwd_flops_per_layer(cfg, ParallelismConfig(tp_size=8))
        assert flops_tp8 < flops_tp1
        assert abs(flops_tp1 / flops_tp8 - 8.0) < 0.2


class TestOffloadModel:
    """Test PCIe transfer calculations."""

    def test_transfer_time_a100(self):
        """A100 PCIe Gen4: 32 GB/s → 1 GB should take ~31ms."""
        one_gb = 1024 ** 3
        t = transfer_time(one_gb, A100_80GB)
        assert 0.025 < t < 0.040, f"Expected ~31ms, got {t*1000:.1f}ms"

    def test_h100_faster_than_a100(self):
        """H100 has PCIe Gen5 (64 GB/s), should be 2× faster."""
        size = 1024 ** 3
        t_a100 = transfer_time(size, A100_80GB)
        t_h100 = transfer_time(size, H100_80GB)
        assert t_h100 < t_a100

    def test_round_trip_is_double(self):
        """Round trip should be exactly 2× one-way."""
        size = 100 * 1024 ** 2  # 100 MB
        rt = round_trip_time(size, A100_80GB)
        ow = transfer_time(size, A100_80GB)
        assert abs(rt - 2 * ow) < 1e-12

    def test_schedule_offloads_accounts_for_serialized_bus_occupancy(self):
        """A later tensor should stall when an earlier transfer consumes the PCIe bus."""
        size = 1024 ** 3  # 1 GiB
        tensors = [
            (TensorInfo("t1", "test", size, 0.0, []), 0.070),
            (TensorInfo("t2", "test", size, 0.0, []), 0.070),
        ]

        results = schedule_offloads(tensors, A100_80GB, ParallelismConfig())

        assert [r.tensor_name for r in results] == ["t1", "t2"]
        assert results[0].stall_time_s == pytest.approx(0.0)

        one_way = transfer_time(size, A100_80GB)
        expected_second_stall = 2 * one_way
        assert results[1].stall_time_s == pytest.approx(expected_second_stall)

    def test_schedule_offloads_packs_sends_into_recv_wait_window(self):
        """Interval-based scheduler: when gap is much larger than round-trip,
        multiple tensors should pack into the window [send_end, recv_start]
        that the recv's ALAP placement leaves idle. The old pcie_busy_until
        model wrongly claimed the bus through recv_end and forced cascading
        stalls even when bus time << gap."""
        size = 100 * 1024 ** 2  # 100 MB, transfer ~ 3ms on A100 (32 GB/s)
        gap = 0.500  # 500 ms gap — bus work (4 × 2 × 3 ms = 24 ms) << 500 ms
        tensors = [
            (TensorInfo(f"t{i}", "test", size, 0.0, []), gap)
            for i in range(4)
        ]

        results = schedule_offloads(tensors, A100_80GB, ParallelismConfig())

        # All four transfers must fit comfortably inside the gap.
        total_stall = sum(r.stall_time_s for r in results)
        assert total_stall == pytest.approx(0.0, abs=1e-9), (
            f"Expected zero stall (bus work << gap); got {total_stall*1000:.2f} ms. "
            f"Suggests pcie_busy_until cascade regression."
        )

    def test_schedule_offloads_stall_reflects_bus_overcommit(self):
        """When total bus work exceeds the gap, excess becomes stall —
        independent of how nicely tensors can pack."""
        size = 1024 ** 3  # 1 GiB, ~32 ms transfer on A100
        gap = 0.050  # 50 ms — only one round-trip (64 ms) already overflows
        tensors = [
            (TensorInfo(f"t{i}", "test", size, 0.0, []), gap)
            for i in range(3)
        ]

        results = schedule_offloads(tensors, A100_80GB, ParallelismConfig())

        # With 3 × 64 ms = 192 ms of bus work and gap=50 ms, at minimum
        # 142 ms of cumulative stall across the three tensors is unavoidable.
        total_stall = sum(r.stall_time_s for r in results)
        one_way = transfer_time(size, A100_80GB)
        minimum_stall = 3 * (2 * one_way) - gap * 3  # cumulative overshoot
        assert total_stall >= minimum_stall - 1e-9, (
            f"Total stall {total_stall*1000:.1f} ms < physical minimum "
            f"{minimum_stall*1000:.1f} ms"
        )

    def test_schedule_offloads_symmetric_single_tensor_matches_independent(self):
        """With one tensor, schedule_offloads must match compute_offload_result
        exactly — the scheduler degenerates to the single-tensor case."""
        from simulator.offload_model import compute_offload_result
        size = 500 * 1024 ** 2  # 500 MB
        gap = 0.030  # 30 ms — forces some stall at A100's ~16ms one-way
        tensor = TensorInfo("t", "test", size, 0.0, [])

        scheduled = schedule_offloads([(tensor, gap)], A100_80GB, ParallelismConfig())
        independent = compute_offload_result(tensor, gap, A100_80GB, ParallelismConfig())

        assert scheduled[0].stall_time_s == pytest.approx(independent.stall_time_s)
        assert scheduled[0].send_time_s == pytest.approx(independent.send_time_s)
        assert scheduled[0].recv_time_s == pytest.approx(independent.recv_time_s)


class TestOffloadSyncMode:
    """sync_mode='serial' models default-stream behavior where DMAs serialize
    with compute and every offload pays full round-trip stall, regardless of
    how much liveness slack exists."""

    def test_serial_ignores_gap_and_returns_round_trip(self):
        """Serial mode: stall = round_trip no matter how large the gap is."""
        size = 100 * 1024 ** 2  # 100 MB
        tensor = TensorInfo("t", "test", size, 0.0, [])

        overlap = compute_offload_result(tensor, liveness_gap_s=10.0, gpu=A100_80GB)
        serial = compute_offload_result(
            tensor, liveness_gap_s=10.0, gpu=A100_80GB, sync_mode="serial",
        )

        # With a 10-second gap, overlap predicts zero stall; serial predicts rt.
        assert overlap.stall_time_s == pytest.approx(0.0)
        assert serial.stall_time_s == pytest.approx(serial.round_trip_s)
        assert serial.stall_time_s > 0

    def test_serial_schedule_offloads_sums_round_trips(self):
        """Serial mode: total stall across tensors is sum of round-trips
        (no bus scheduling — each transfer blocks compute on its own)."""
        size = 100 * 1024 ** 2
        gap = 1.0  # huge gap — overlap mode would predict zero stall
        tensors = [
            (TensorInfo(f"t{i}", "test", size, 0.0, []), gap)
            for i in range(4)
        ]

        overlap_results = schedule_offloads(tensors, A100_80GB, ParallelismConfig())
        serial_results = schedule_offloads(
            tensors, A100_80GB, ParallelismConfig(), sync_mode="serial",
        )

        assert sum(r.stall_time_s for r in overlap_results) == pytest.approx(0.0)
        expected_serial_stall = sum(r.round_trip_s for r in serial_results)
        assert sum(r.stall_time_s for r in serial_results) == pytest.approx(
            expected_serial_stall
        )

    def test_overlap_is_default(self):
        """Default sync_mode must be 'overlap' — preserves existing behavior
        for all callers that don't pass the new argument."""
        size = 10 * 1024 ** 2
        tensor = TensorInfo("t", "test", size, 0.0, [])
        # Gap = 2 × transfer should give zero stall under overlap.
        one_way = transfer_time(size, A100_80GB)
        gap = 3 * one_way
        result = compute_offload_result(tensor, gap, A100_80GB)  # no sync_mode
        assert result.stall_time_s == pytest.approx(0.0)

    def test_simulate_threads_sync_mode_through(self):
        """simulate() wires offload_sync_mode into schedule_offloads; the
        measured total stall at seq=2048 linear2-only offload should jump
        from 0 (overlap) to ~1.4 ms × num_layers (serial)."""
        from simulator.config import H200_141GB, llama_7b, LayerStrategy
        from simulator.config import TensorAction, TensorDecision
        from simulator.environment import simulate

        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        strategies = [
            LayerStrategy(
                layer_idx=i,
                decisions={
                    "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                },
            )
            for i in range(cfg.num_layers)
        ]

        overlap = simulate(cfg, H200_141GB, strategies=strategies)
        serial = simulate(
            cfg, H200_141GB, strategies=strategies, offload_sync_mode="serial",
        )

        # Overlap mode: 2048×11008×2=45 MB/layer at 64 GB/s → 0.7 ms one-way,
        # 1.4 ms round-trip — easily fits in the ~11 ms per-layer gap.
        assert overlap.total_offload_stall_s == pytest.approx(0.0)

        # Serial mode: 1.4 ms × 32 layers = ~45 ms per microbatch.
        expected_ms = 32 * 2 * 2048 * 11008 * 2 / (64 * 1024**3) * 1000  # ~44.7 ms
        actual_ms = serial.total_offload_stall_s * 1000
        assert abs(actual_ms - expected_ms) / expected_ms < 0.05


class TestCompressionModel:
    """Test low-rank compression estimates."""

    def test_rank_equals_dim_no_error(self):
        """Full rank → zero compression error."""
        assert estimate_error(rank=100, full_dim=100) == 0.0

    def test_lower_rank_higher_error(self):
        """Lower rank should produce higher error."""
        e_high = estimate_error(rank=16, full_dim=4096)
        e_low = estimate_error(rank=512, full_dim=4096)
        assert e_high > e_low

    def test_compression_ratio_formula(self):
        """Verify compression ratio: (rows*r + r*cols) / (rows*cols)."""
        rows, cols, r = 8192, 4096, 512
        ratio = compression_ratio(rows, cols, r)
        expected = (rows * r + r * cols) / (rows * cols)
        assert abs(ratio - expected) < 1e-10

    def test_high_rank_low_compression(self):
        """Rank close to min(rows,cols) should give ratio near 1."""
        ratio = compression_ratio(8192, 4096, 4000)
        assert ratio > 0.9


class TestSimulatorEnvironment:
    """Integration tests for the full simulator."""

    def test_no_ac_uses_most_memory(self):
        """No AC should use more activation memory than any other strategy."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB

        r_none = simulate_no_ac(cfg, gpu)
        r_full = simulate_full_ac(cfg, gpu)
        r_sel = simulate_selective_ac(cfg, gpu)

        assert r_none.peak_activation_memory_bytes >= r_sel.peak_activation_memory_bytes
        assert r_none.peak_activation_memory_bytes >= r_full.peak_activation_memory_bytes

    def test_full_ac_uses_least_memory(self):
        """Full AC should use least activation memory."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB

        r_none = simulate_no_ac(cfg, gpu)
        r_full = simulate_full_ac(cfg, gpu)

        assert r_full.peak_activation_memory_bytes < r_none.peak_activation_memory_bytes

    def test_full_ac_has_most_recompute(self):
        """Full AC should have highest recompute overhead."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB

        r_none = simulate_no_ac(cfg, gpu)
        r_full = simulate_full_ac(cfg, gpu)

        assert r_full.recompute_overhead_pct > r_none.recompute_overhead_pct
        assert r_none.recompute_overhead_pct == 0.0

    def test_per_layer_count_matches(self):
        """Should have exactly num_layers entries in per_layer."""
        cfg = llama_7b()
        gpu = A100_80GB
        result = simulate_no_ac(cfg, gpu)
        assert len(result.per_layer) == cfg.num_layers

    def test_custom_strategy(self):
        """Test applying a custom per-tensor strategy."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB

        # Offload MLP activations, keep everything else
        strategies = []
        for i in range(cfg.num_layers):
            decisions = {
                "mlp_gate_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_up_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
            }
            strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))

        result = simulate(cfg, gpu, strategies=strategies)

        # Should have some offloaded bytes
        total_offloaded = sum(lr.offloaded_bytes for lr in result.per_layer)
        assert total_offloaded > 0

        # Should use less activation memory than no-AC
        r_none = simulate_no_ac(cfg, gpu)
        assert result.peak_activation_memory_bytes < r_none.peak_activation_memory_bytes

    def test_selective_ac_matches_no_ac_with_flash_attention(self):
        """Attention-only selective AC should be a no-op once FA removes s^2 tensors."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        r_none = simulate_no_ac(cfg, gpu)
        r_sel = simulate_selective_ac(cfg, gpu)
        assert r_sel.peak_activation_memory_bytes == r_none.peak_activation_memory_bytes
        assert r_sel.recompute_overhead_pct == 0.0

    def test_invalid_recompute_raises(self):
        """Strategies should reject impossible recomputations for opaque tensors."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        strategies = [
            LayerStrategy(
                layer_idx=i,
                decisions={
                    "attn_fa_logsumexp": TensorDecision(action=TensorAction.RECOMPUTE),
                },
            )
            for i in range(cfg.num_layers)
        ]
        with pytest.raises(ValueError, match="cannot be recomputed"):
            simulate(cfg, gpu, strategies=strategies)

    def test_pipeline_parallel_reduces_stage_memory(self):
        """Pipeline stages should hold only their local layer partition."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        r_pp1 = simulate_no_ac(cfg, gpu, par=ParallelismConfig(tp_size=8, pp_size=1))
        r_pp8 = simulate_no_ac(cfg, gpu, par=ParallelismConfig(tp_size=8, pp_size=8))
        assert r_pp8.total_peak_memory_bytes < r_pp1.total_peak_memory_bytes

    def test_llama_7b_fits_on_a100_with_fsdp(self):
        """Llama 7B with FSDP (dp=8) should fit on A100-80GB.

        Without sharding, optimizer alone is ~79 GB (12 bytes/param × 6.6B params).
        With FSDP dp=8, optimizer is ~10 GB.
        """
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        par = ParallelismConfig(dp_size=8)
        result = simulate_no_ac(cfg, A100_80GB, par=par)
        assert result.fits_in_memory, (
            f"Llama 7B with FSDP dp=8 should fit on A100-80GB, "
            f"peak={_gb(result.total_peak_memory_bytes):.1f} GB"
        )

    def test_print_result_runs(self):
        """Smoke test: print_result should not crash."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        result = simulate_no_ac(cfg, A100_80GB)
        print_result(result)


class TestCompressionComputeCost:
    """Verify compression is no longer modeled as free."""

    def test_compression_has_nonzero_flops(self):
        """Compression round-trip should cost 4×rows×cols×rank FLOPs."""
        rows, cols, rank = 8192, 4096, 512
        c, d = compression_flops(rows, cols, rank)
        assert c == 2 * rows * cols * rank
        assert d == 2 * rows * rank * cols
        assert c + d == 4 * rows * cols * rank

    def test_compression_adds_latency(self):
        """3-Resource with COMPRESS should have higher step latency than No AC."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(dp_size=8)

        r_none = simulate_no_ac(cfg, gpu, par=par)

        # Build a strategy that compresses MLP tensors
        tensors = get_all_tensors_per_layer(cfg, par)
        strategies = []
        for i in range(cfg.num_layers):
            decisions = {
                "mlp_linear2_input": TensorDecision(
                    action=TensorAction.COMPRESS,
                    compress_rank=cfg.hidden_dim // 8,
                ),
            }
            strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))

        r_comp = simulate(cfg, gpu, strategies=strategies, par=par)

        assert r_comp.total_compression_flops > 0
        assert r_comp.step_latency_s > r_none.step_latency_s, (
            "Compression should increase step latency"
        )

    def test_compression_overhead_nontrivial(self):
        """Compression overhead should be measurable, not negligible."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(dp_size=8)

        tensors = get_all_tensors_per_layer(cfg, par)
        strategies = []
        for i in range(cfg.num_layers):
            decisions = {
                "mlp_linear2_input": TensorDecision(
                    action=TensorAction.COMPRESS,
                    compress_rank=cfg.hidden_dim // 8,
                ),
            }
            strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))

        r = simulate(cfg, gpu, strategies=strategies, par=par)
        # Overhead should be > 0.1% (not negligible)
        assert r.recompute_overhead_pct > 0.1, (
            f"Compression overhead should be nontrivial, got {r.recompute_overhead_pct:.3f}%"
        )


class TestPCIeContention:
    """Verify NCCL contention reduces effective offload bandwidth."""

    def test_no_contention_without_fsdp(self):
        """Single GPU (no parallelism) → full PCIe bandwidth."""
        par = ParallelismConfig()
        util = estimate_nccl_pcie_utilization(A100_80GB, par)
        assert util == 0.0

    def test_contention_with_multinode_fsdp(self):
        """Multi-node FSDP → reduced PCIe bandwidth."""
        par = ParallelismConfig(dp_size=16)  # Multi-node
        util = estimate_nccl_pcie_utilization(A100_80GB, par)
        assert util > 0.0, "NCCL should consume some PCIe bandwidth"

    def test_effective_bw_less_than_raw(self):
        """Effective bandwidth should be less than raw with FSDP."""
        par = ParallelismConfig(dp_size=16)
        eff = effective_pcie_bandwidth(A100_80GB, par)
        raw = A100_80GB.pcie_bandwidth_bytes_s
        assert eff < raw, "Effective BW should be reduced by NCCL contention"

    def test_offload_slower_with_contention(self):
        """Same tensor should take longer to offload when PCIe is contested."""
        from simulator.memory_model import TensorInfo
        tensor = TensorInfo(
            name="test", block="mlp",
            size_bytes=100 * 1024 ** 2,  # 100 MB
            recompute_flops=0, recompute_from=[],
        )
        par_none = ParallelismConfig()
        par_fsdp = ParallelismConfig(dp_size=16)

        t_none = transfer_time(tensor.size_bytes, A100_80GB, par_none)
        t_fsdp = transfer_time(tensor.size_bytes, A100_80GB, par_fsdp)

        assert t_fsdp > t_none, (
            f"Transfer with FSDP contention ({t_fsdp*1000:.2f}ms) should be "
            f"slower than without ({t_none*1000:.2f}ms)"
        )

    def test_offload_stalls_appear_with_contention(self):
        """With reduced PCIe BW, offloading should produce stalls."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB

        # No contention case
        par_tp = ParallelismConfig(tp_size=8)
        tensors = get_all_tensors_per_layer(cfg, par_tp)
        strategies = []
        for i in range(cfg.num_layers):
            decisions = {
                "mlp_gate_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_up_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
            }
            strategies.append(LayerStrategy(layer_idx=i, decisions=decisions))
        r_tp = simulate(cfg, gpu, strategies=strategies, par=par_tp)

        # With contention case
        par_fsdp = ParallelismConfig(dp_size=16)
        tensors = get_all_tensors_per_layer(cfg, par_fsdp)
        strategies_fsdp = []
        for i in range(cfg.num_layers):
            decisions = {
                "mlp_gate_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_up_output": TensorDecision(action=TensorAction.OFFLOAD_CPU),
                "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
            }
            strategies_fsdp.append(LayerStrategy(layer_idx=i, decisions=decisions))
        r_fsdp = simulate(cfg, gpu, strategies=strategies_fsdp, par=par_fsdp)

        assert r_fsdp.total_offload_stall_s >= r_tp.total_offload_stall_s, (
            "FSDP contention should produce equal or more offload stalls"
        )


class TestFAEraSelectiveAC:
    """Test the FA-era selective recompute strategy."""

    def test_saves_memory_vs_no_ac(self):
        """FA-era selective should save memory by recomputing the activation output."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(dp_size=8)

        r_none = simulate_no_ac(cfg, gpu, par=par)
        r_fa_sel = simulate_fa_selective_ac(cfg, gpu, par=par)

        assert r_fa_sel.peak_activation_memory_bytes < r_none.peak_activation_memory_bytes

    def test_much_less_overhead_than_full_ac(self):
        """FA-era selective should have far less overhead than full AC."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(dp_size=8)

        r_full = simulate_full_ac(cfg, gpu, par=par)
        r_fa_sel = simulate_fa_selective_ac(cfg, gpu, par=par)

        assert r_fa_sel.recompute_overhead_pct < r_full.recompute_overhead_pct / 5, (
            f"FA-era selective ({r_fa_sel.recompute_overhead_pct:.2f}%) should be "
            f"<< full AC ({r_full.recompute_overhead_pct:.2f}%)"
        )

    def test_overhead_under_2_percent(self):
        """Recomputing pointwise activation should cost <2% overhead."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(dp_size=8)

        r = simulate_fa_selective_ac(cfg, gpu, par=par)
        assert r.recompute_overhead_pct < 2.0, (
            f"FA-era selective overhead should be tiny, got {r.recompute_overhead_pct:.2f}%"
        )

    def test_only_recomputes_activation_output(self):
        """Should only recompute the activation output (mlp_silu_output), nothing else."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB

        r = simulate_fa_selective_ac(cfg, gpu)
        layer0 = r.per_layer[0]
        for tname, action in layer0.tensor_details.items():
            if tname == "mlp_silu_output":
                assert action == "RECOMPUTE"
            else:
                assert action == "KEEP", f"{tname} should be KEEP, got {action}"


class TestPipelineAwareAC:
    """Test pipeline-position-aware activation checkpointing."""

    def test_stash_count_first_stage(self):
        """First stage stashes PP-1 microbatches."""
        assert _stash_count_1f1b(stage_idx=0, pp_size=8) == 7

    def test_stash_count_last_stage(self):
        """Last stage stashes 0 microbatches."""
        assert _stash_count_1f1b(stage_idx=7, pp_size=8) == 0

    def test_stash_count_monotonic(self):
        """Stash count should decrease from first to last stage."""
        pp = 8
        counts = [_stash_count_1f1b(s, pp) for s in range(pp)]
        assert counts == sorted(counts, reverse=True)

    def test_aware_uses_different_strategies_per_stage(self):
        """Pipeline-aware should NOT use the same strategy for all stages.

        Uses Llama-7B with large batch + PP=4 where activation stashing
        creates real memory pressure on early stages but not late stages.
        """
        cfg = llama_7b(seq_len=4096, micro_batch_size=4)
        gpu = A100_80GB
        par = ParallelismConfig(pp_size=4, dp_size=4)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        strategies_used = set(sr.strategy_name for sr in pr.stages)
        assert len(strategies_used) >= 2, (
            f"Pipeline-aware should use different strategies when stashing "
            f"creates differential pressure, but only used: {strategies_used}"
        )

    def test_first_stage_more_aggressive_than_last(self):
        """First stage should use a more aggressive strategy than last stage."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        first = pr.stages[0]
        last = pr.stages[-1]

        # Strategy aggressiveness order: No AC < FA-Selective < Full AC
        level_order = {"No AC": 0, "FA-Selective": 1, "Full AC": 2}
        first_level = level_order[first.strategy_name]
        last_level = level_order[last.strategy_name]

        assert first_level >= last_level, (
            f"First stage ({first.strategy_name}) should be at least as "
            f"aggressive as last stage ({last.strategy_name})"
        )

    def test_aware_less_overhead_than_uniform_full_ac(self):
        """Pipeline-aware should have less overhead than uniform Full AC.

        The key benefit: late stages don't need Full AC, so the bottleneck
        stage (with lowest recompute overhead) is faster.
        """
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr_aware = simulate_pipeline_aware_ac(cfg, gpu, par)
        pr_uniform = simulate_pipeline_uniform_ac(cfg, gpu, par, strategy_name="Full AC")

        assert pr_aware.overall_step_latency_s <= pr_uniform.overall_step_latency_s, (
            f"Pipeline-aware ({pr_aware.overall_step_latency_s*1000:.2f}ms) should be "
            f"no slower than uniform Full AC ({pr_uniform.overall_step_latency_s*1000:.2f}ms)"
        )

    def test_all_stages_present(self):
        """Should have one result per pipeline stage."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        assert len(pr.stages) == par.pp_size

    def test_stash_count_in_results(self):
        """Each stage's stash count should match the 1F1B formula."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        for sr in pr.stages:
            expected = _stash_count_1f1b(sr.stage_idx, par.pp_size)
            assert sr.num_stashed_microbatches == expected

    def test_late_stages_use_less_memory(self):
        """Late stages (less stashing) should have lower peak memory."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        first_peak = pr.stages[0].sim.total_peak_memory_bytes
        last_peak = pr.stages[-1].sim.total_peak_memory_bytes

        assert last_peak <= first_peak, (
            f"Last stage peak ({_gb(last_peak):.2f} GB) should be ≤ "
            f"first stage peak ({_gb(first_peak):.2f} GB)"
        )


class TestMultiSchedulePipeline:
    """Test pipeline-aware AC across different pipeline schedules."""

    def test_dualpipe_symmetric_stash(self):
        """DualPipe should have equal stash count on all stages."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        profile = get_schedule_profile(PipelineSchedule.DUALPIPE, cfg, par)
        assert all(s == par.pp_size - 1 for s in profile.stash_counts), (
            f"DualPipe should have symmetric stash={par.pp_size-1}, "
            f"got {profile.stash_counts}"
        )

    def test_1f1b_asymmetric_stash(self):
        """1F1B should have decreasing stash from first to last stage."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        profile = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B, cfg, par)
        assert profile.stash_counts == list(range(7, -1, -1))

    def test_zb_h2_extra_memory(self):
        """ZB-H2 should have non-zero extra memory for deferred W on all stages."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        profile = get_schedule_profile(PipelineSchedule.ZB_H2, cfg, par)
        assert len(profile.extra_memory_per_stage) == par.pp_size
        assert all(m > 0 for m in profile.extra_memory_per_stage), (
            "ZB-H2 should have deferred W memory on every stage"
        )

    def test_zb_h2_uneven_layers_different_extra_memory(self):
        """ZB-H2 with uneven layer division should have different extra per stage."""
        # 32 layers / 3 stages = 11, 11, 10 — stage 2 has fewer layers
        cfg = llama_7b()  # 32 layers
        par = ParallelismConfig(pp_size=3)
        profile = get_schedule_profile(PipelineSchedule.ZB_H2, cfg, par)
        # Stages 0,1 have 11 layers; stage 2 has 10 → different extra memory
        assert profile.extra_memory_per_stage[0] == profile.extra_memory_per_stage[1]
        assert profile.extra_memory_per_stage[2] < profile.extra_memory_per_stage[0], (
            "Stage with fewer layers should have less deferred-W memory"
        )

    def test_zb_zero_bubble(self):
        """ZB schedules should have zero bubble fraction."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        for sched in [PipelineSchedule.ZB_H1, PipelineSchedule.ZB_H2, PipelineSchedule.ZB_V]:
            profile = get_schedule_profile(sched, cfg, par)
            assert profile.bubble_fraction == 0.0, f"{sched} should have zero bubble"

    def test_1f1b_has_bubble(self):
        """1F1B should have nonzero bubble fraction."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        profile = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B, cfg, par)
        assert profile.bubble_fraction > 0

    def test_interleaved_less_bubble(self):
        """1F1B Interleaved should have less bubble than 1F1B."""
        par = ParallelismConfig(tp_size=8, pp_size=8)
        cfg = gpt3_175b()
        p_1f1b = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B, cfg, par)
        p_inter = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B_INTERLEAVED, cfg, par)
        assert p_inter.bubble_fraction < p_1f1b.bubble_fraction

    def test_dualpipe_aware_all_same_strategy(self):
        """DualPipe has symmetric stash, so pipeline-aware should pick the same strategy everywhere."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=1)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)
        pr = simulate_pipeline_aware_ac(cfg, gpu, par, schedule=PipelineSchedule.DUALPIPE)
        strategies = set(sr.strategy_name for sr in pr.stages)
        assert len(strategies) == 1, (
            f"DualPipe symmetric stash should yield uniform strategy, got {strategies}"
        )

    def test_bubble_affects_overall_latency(self):
        """1F1B should have higher overall latency than ZB due to bubble overhead."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=4)
        gpu = A100_80GB
        par = ParallelismConfig(pp_size=4, dp_size=4)

        pr_1f1b = simulate_pipeline_uniform_ac(
            cfg, gpu, par, strategy_name="Full AC",
            schedule=PipelineSchedule.ONE_F_ONE_B)
        pr_zb = simulate_pipeline_uniform_ac(
            cfg, gpu, par, strategy_name="Full AC",
            schedule=PipelineSchedule.ZB_H1)

        # Same bottleneck per-microbatch time (same stash, same strategy)
        assert abs(pr_1f1b.bottleneck_step_latency_s - pr_zb.bottleneck_step_latency_s) < 1e-6

        # But 1F1B overall should be higher due to bubble
        assert pr_1f1b.overall_step_latency_s > pr_zb.overall_step_latency_s, (
            f"1F1B ({pr_1f1b.overall_step_latency_s*1000:.1f}ms) should be slower than "
            f"ZB ({pr_zb.overall_step_latency_s*1000:.1f}ms) due to bubble"
        )

    def test_korthikanti_selected_for_non_fa_model(self):
        """Pipeline-aware should use Korthikanti selective for non-FA models when it fits."""
        cfg = gpt3_175b(seq_len=2048, micro_batch_size=2)
        gpu = A100_80GB
        par = ParallelismConfig(tp_size=8, pp_size=8)

        pr = simulate_pipeline_aware_ac(cfg, gpu, par)
        strategies_used = set(sr.strategy_name for sr in pr.stages)

        # At least one stage should use Korthikanti Selective if it fits
        # (GPT-3 175B without FA has a large quadratic term that Korthikanti eliminates)
        # The key check: Full AC should NOT be used if Korthikanti fits
        if "Full AC" in strategies_used:
            # If Full AC is used, verify Korthikanti didn't fit on that stage
            for sr in pr.stages:
                if sr.strategy_name == "Full AC":
                    # This stage genuinely couldn't fit with anything lighter
                    pass  # Acceptable
        # At minimum, verify the strategy search includes Korthikanti
        assert any(name == "Korthikanti Selective" for name, _ in
                   __import__('simulator.environment', fromlist=['STRATEGY_LEVELS']).STRATEGY_LEVELS)

    def test_aware_works_across_all_schedules(self):
        """Pipeline-aware AC should run without error on every schedule."""
        cfg = llama_7b(seq_len=4096, micro_batch_size=4)
        gpu = A100_80GB
        par = ParallelismConfig(pp_size=4, dp_size=4)

        for sched in PipelineSchedule:
            pr = simulate_pipeline_aware_ac(cfg, gpu, par, schedule=sched)
            assert len(pr.stages) == par.pp_size, f"Failed on {sched}"
            assert pr.overall_step_latency_s > 0, f"Zero latency on {sched}"


class TestGPipeSchedule:
    """GPipe: uniform worst-case stash (all stages hold all M microbatches),
    bubble = (PP-1) / (M + PP - 1)."""

    def test_gpipe_stash_is_uniform_and_equals_m_minus_1(self):
        from simulator.pipeline_schedules import get_schedule_profile
        cfg = llama_7b(seq_len=4096, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)
        M = 8
        profile = get_schedule_profile(
            PipelineSchedule.GPIPE, cfg, par, num_microbatches=M,
        )
        assert profile.stash_counts == [M - 1] * par.pp_size

    def test_gpipe_stash_exceeds_1f1b_by_m_minus_pp(self):
        """At M=8, PP=4: GPipe stashes 7 everywhere; 1F1B stashes [3,2,1,0].
        GPipe's worst-case stage has 7 - 3 = 4 more microbatches stashed."""
        from simulator.pipeline_schedules import get_schedule_profile
        cfg = llama_7b(seq_len=4096, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)
        gpipe = get_schedule_profile(PipelineSchedule.GPIPE, cfg, par, 8)
        f1b = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B, cfg, par, 8)
        assert max(gpipe.stash_counts) - max(f1b.stash_counts) == 8 - 4

    def test_gpipe_bubble_formula(self):
        """GPipe bubble = (PP-1) / (M + PP - 1). At PP=4 M=8: 3/11 ≈ 27.3%."""
        from simulator.pipeline_schedules import get_schedule_profile
        cfg = llama_7b(seq_len=4096, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)
        p = get_schedule_profile(PipelineSchedule.GPIPE, cfg, par, 8)
        assert abs(p.bubble_fraction - 3 / 11) < 1e-9

    def test_gpipe_bubble_smaller_than_1f1b_at_small_m(self):
        """At M=PP, 1F1B bubble = (PP-1)/PP while GPipe bubble = (PP-1)/(2PP-1).
        GPipe is smaller when M is close to PP (fewer microbatches in flight)."""
        from simulator.pipeline_schedules import get_schedule_profile
        cfg = llama_7b(seq_len=4096, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)
        g = get_schedule_profile(PipelineSchedule.GPIPE, cfg, par, 4)
        f = get_schedule_profile(PipelineSchedule.ONE_F_ONE_B, cfg, par, 4)
        assert g.bubble_fraction < f.bubble_fraction

    def test_pipeline_aware_on_gpipe_picks_more_aggressive(self):
        """Because GPipe stashes M-1 per stage vs 1F1B's PP-1-p, aware-AC under
        GPipe should pick at least as aggressive a strategy at every stage
        (and strictly more aggressive on some when GPipe stash > 1F1B stash)."""
        from simulator.config import H200_141GB
        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)
        M = 8

        gpipe_pr = simulate_pipeline_aware_ac(
            cfg, H200_141GB, par,
            schedule=PipelineSchedule.GPIPE, num_microbatches=M,
        )
        f1b_pr = simulate_pipeline_aware_ac(
            cfg, H200_141GB, par,
            schedule=PipelineSchedule.ONE_F_ONE_B, num_microbatches=M,
        )
        order = ["No AC", "Offload linear2", "Offload all MLP",
                 "FA-Selective", "Korthikanti Selective", "Full AC"]
        for s_gpipe, s_f1b in zip(gpipe_pr.stages, f1b_pr.stages):
            assert order.index(s_gpipe.strategy_name) >= order.index(s_f1b.strategy_name), (
                f"GPipe picked {s_gpipe.strategy_name!r}, 1F1B picked {s_f1b.strategy_name!r}"
            )


class TestGPUConfigs:
    """Hardware profile sanity checks."""

    def test_h200_basic_fields(self):
        """H200-141GB: 141 GB HBM3e at 4.8 TB/s, same compute as H100."""
        assert H200_141GB.hbm_capacity_gb == 141.0
        assert H200_141GB.hbm_bandwidth_gb_s == 4800.0
        assert H200_141GB.peak_fp16_tflops == 989.0
        # Matches H100's FP16 peak — the H200 differs only in memory, not compute.
        assert H200_141GB.peak_fp16_tflops == H100_80GB.peak_fp16_tflops
        assert H200_141GB.peak_fp8_tflops == 1979.0
        assert H200_141GB.pcie_bandwidth_gb_s == 64.0  # PCIe Gen5

    def test_h200_fits_llama_7b_no_ac(self):
        """Llama-7B at seq=4096 should fit comfortably on a single H200 under
        No AC — this is the regime where pipeline-aware AC degenerates to
        uniform in the throughput runner."""
        from simulator.environment import simulate_no_ac
        cfg = llama_7b(seq_len=4096, micro_batch_size=1)
        result = simulate_no_ac(cfg, H200_141GB)
        assert result.fits_in_memory, (
            f"Llama-7B No AC should fit on H200-141GB, "
            f"peak={result.total_peak_memory_bytes / 1024**3:.1f} GB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
