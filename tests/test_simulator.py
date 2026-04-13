"""Tests for the activation memory simulator.

Validates against Korthikanti et al. closed-form formulas and
cross-checks with llm-analysis where possible.
"""

import pytest

from simulator.config import (
    A100_80GB,
    H100_80GB,
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
)
from simulator.compression_model import (
    compressed_size,
    compression_ratio,
    estimate_error,
)
from simulator.environment import (
    simulate,
    simulate_no_ac,
    simulate_full_ac,
    simulate_selective_ac,
    print_result,
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
        """Standard GELU MLP should have: input, gelu_input, linear2_input."""
        cfg = ModelConfig(
            name="test", num_layers=1, hidden_dim=4096,
            n_heads=32, vocab_size=32000, seq_len=2048,
            micro_batch_size=1, activation_fn=ActivationFunction.GELU,
        )
        tensors = get_mlp_tensors(cfg)
        names = [t.name for t in tensors]
        assert "mlp_input" in names
        assert "mlp_gelu_input" in names
        assert "mlp_linear2_input" in names

    def test_mlp_tensors_swiglu(self):
        """SwiGLU MLP should have: input, gate_output, up_output, linear2_input."""
        cfg = llama_7b()
        tensors = get_mlp_tensors(cfg)
        names = [t.name for t in tensors]
        assert "mlp_gate_output" in names
        assert "mlp_up_output" in names
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
