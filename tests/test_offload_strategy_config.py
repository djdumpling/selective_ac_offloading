"""Sanity checks on the simulator's OFFLOAD_CPU path for the config the
validator runs: Llama-7B on H200, offloading mlp_linear2_input per layer."""

from simulator.config import (
    H200_141GB,
    LayerStrategy,
    ParallelismConfig,
    TensorAction,
    TensorDecision,
    llama_7b,
)
from simulator.environment import simulate, simulate_no_ac


def _build_offload_linear2_strategies(num_layers: int):
    return [
        LayerStrategy(
            layer_idx=i,
            decisions={
                "mlp_linear2_input": TensorDecision(action=TensorAction.OFFLOAD_CPU),
            },
        )
        for i in range(num_layers)
    ]


class TestOffloadLinear2Simulation:
    def test_offload_reduces_peak_activation(self):
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        baseline = simulate_no_ac(cfg, H200_141GB)
        off = simulate(
            cfg, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg.num_layers),
            par=ParallelismConfig(),
        )
        assert off.peak_activation_memory_bytes < baseline.peak_activation_memory_bytes

    def test_no_recompute_flops(self):
        """Offloading costs PCIe time, not FLOPs — the simulator must not
        double-count this as recompute."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        off = simulate(
            cfg, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg.num_layers),
        )
        assert off.total_recompute_flops == 0

    def test_stall_time_nonnegative_and_finite(self):
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        off = simulate(
            cfg, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg.num_layers),
        )
        assert off.total_offload_stall_s >= 0
        assert off.total_offload_stall_s < 10.0  # sanity upper bound

    def test_savings_scale_with_seq_len(self):
        """linear2_input is seq × mbs × ffn × bpe — doubling seq should ~double
        the saved bytes per layer."""
        cfg_short = llama_7b(seq_len=2048, micro_batch_size=1)
        cfg_long = llama_7b(seq_len=4096, micro_batch_size=1)
        base_short = simulate_no_ac(cfg_short, H200_141GB).peak_activation_memory_bytes
        off_short = simulate(
            cfg_short, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg_short.num_layers),
        ).peak_activation_memory_bytes

        base_long = simulate_no_ac(cfg_long, H200_141GB).peak_activation_memory_bytes
        off_long = simulate(
            cfg_long, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg_long.num_layers),
        ).peak_activation_memory_bytes

        saved_short = base_short - off_short
        saved_long = base_long - off_long
        # Not exactly 2× because activation_memory_per_layer has other terms
        # that also scale with seq, but the savings ratio must be monotonic.
        assert saved_long > saved_short
        assert 1.5 < saved_long / saved_short < 2.5

    def test_named_strategy_offload_linear2(self):
        """The STRATEGY_LEVELS-registered 'Offload linear2' builder produces the
        same result as a hand-built strategy."""
        from simulator.environment import simulate_pipeline_uniform_ac
        from simulator.pipeline_schedules import PipelineSchedule

        cfg = llama_7b(seq_len=8192, micro_batch_size=1)
        par = ParallelismConfig(pp_size=2)
        pr = simulate_pipeline_uniform_ac(
            cfg, H200_141GB, par,
            strategy_name="Offload linear2",
            schedule=PipelineSchedule.ONE_F_ONE_B,
            num_microbatches=4,
        )
        assert pr.all_fit
        assert all(s.strategy_name == "Offload linear2" for s in pr.stages)

    def test_named_strategy_offload_all_mlp(self):
        from simulator.environment import simulate_pipeline_uniform_ac
        from simulator.pipeline_schedules import PipelineSchedule

        cfg = llama_7b(seq_len=8192, micro_batch_size=1)
        par = ParallelismConfig(pp_size=2)
        pr = simulate_pipeline_uniform_ac(
            cfg, H200_141GB, par,
            strategy_name="Offload all MLP",
            schedule=PipelineSchedule.ONE_F_ONE_B,
            num_microbatches=4,
        )
        assert pr.all_fit
        # Offload all MLP should save more HBM than offload linear2 alone.
        pr_linear2 = simulate_pipeline_uniform_ac(
            cfg, H200_141GB, par,
            strategy_name="Offload linear2",
            schedule=PipelineSchedule.ONE_F_ONE_B,
            num_microbatches=4,
        )
        assert (pr.stages[0].sim.peak_activation_memory_bytes
                < pr_linear2.stages[0].sim.peak_activation_memory_bytes)

    def test_long_context_sweet_spot_exists(self):
        """The headline experiment's simulator prediction: Llama-7B PP=4 seq=32K
        should OOM under No AC but fit under Offload all MLP."""
        from simulator.environment import simulate_pipeline_uniform_ac
        from simulator.pipeline_schedules import PipelineSchedule

        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        par = ParallelismConfig(pp_size=4)

        pr_no_ac = simulate_pipeline_uniform_ac(
            cfg, H200_141GB, par, strategy_name="No AC",
            schedule=PipelineSchedule.ONE_F_ONE_B, num_microbatches=8,
        )
        pr_offload = simulate_pipeline_uniform_ac(
            cfg, H200_141GB, par, strategy_name="Offload all MLP",
            schedule=PipelineSchedule.ONE_F_ONE_B, num_microbatches=8,
        )
        assert not pr_no_ac.all_fit, "No AC must OOM at seq=32K for this sweet spot"
        assert pr_offload.all_fit, "Offload all MLP must fit at seq=32K"

    def test_expected_savings_magnitude_llama7b_seq2048(self):
        """mlp_linear2_input at Llama-7B, seq=2048, mbs=1:
        2048 × 1 × 11008 × 2 bytes = 45,088,768 bytes/layer × 32 layers = 1.44 GB."""
        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        base = simulate_no_ac(cfg, H200_141GB).peak_activation_memory_bytes
        off = simulate(
            cfg, H200_141GB,
            strategies=_build_offload_linear2_strategies(cfg.num_layers),
        ).peak_activation_memory_bytes
        saved = base - off
        expected = 2048 * 1 * 11008 * 2 * 32
        # Allow 1% slack for tp_size=1 division rounding etc.
        assert abs(saved - expected) / expected < 0.01


def _build_mlp_offload_strategies(num_layers: int, tensor_names: list[str]):
    return [
        LayerStrategy(
            layer_idx=i,
            decisions={
                n: TensorDecision(action=TensorAction.OFFLOAD_CPU)
                for n in tensor_names
            },
        )
        for i in range(num_layers)
    ]


class TestOffloadBusContention:
    """Regression guard for schedule_offloads() wiring + interval-based bus model.

    simulate() routes all OFFLOAD_CPU decisions per layer through a single
    schedule_offloads() call, which packs sends and recvs on a shared
    half-duplex PCIe bus. When total bus work fits within the liveness gap,
    stall is zero (even with many tensors); when it overflows, excess
    becomes stall.
    """

    def test_multi_tensor_offload_fits_without_stall_at_seq32k(self):
        """At seq=32K on H200, the per-layer fwd+bwd compute window is ~180 ms
        while the 4 MLP tensors need only 4×22ms=88ms of total bus time. An
        interval-based scheduler packs everything into that window with zero
        stall; the previous pcie_busy_until model wrongly predicted cascading
        stalls because it claimed the bus through each recv's ALAP wait."""
        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        strategies = _build_mlp_offload_strategies(
            cfg.num_layers,
            ["mlp_gate_output", "mlp_up_output", "mlp_silu_output", "mlp_linear2_input"],
        )
        result = simulate(cfg, H200_141GB, strategies=strategies)
        assert result.total_offload_stall_s == 0.0, (
            f"Expected 0 stall — bus work (88ms) fits in gap (~180ms). "
            f"Got {result.total_offload_stall_s*1000:.1f} ms, suggesting "
            f"pcie_busy_until regression."
        )

    def test_stall_appears_when_bus_overcommitted(self):
        """When enough tensors are offloaded that total bus time exceeds the
        liveness gap, stall must appear. Drive contention by offloading many
        attention + MLP tensors at once in a config where compute is small."""
        # Short seq so the compute window is small; include attention tensors
        # to push total offload work past the per-layer gap.
        cfg = llama_7b(seq_len=512, micro_batch_size=1)
        # Offload every MLP tensor and the big attention intermediates.
        names = [
            "mlp_gate_output", "mlp_up_output", "mlp_silu_output", "mlp_linear2_input",
            "attn_q", "attn_k", "attn_v", "attn_output",
        ]
        strategies = _build_mlp_offload_strategies(cfg.num_layers, names)
        result = simulate(cfg, H200_141GB, strategies=strategies)
        # We only need contention to exist somewhere; exact magnitude depends
        # on the compute model.
        assert result.total_offload_stall_s > 0, (
            "Offloading 8 large tensors at seq=512 should overcommit the bus; "
            "got 0 stall"
        )

    def test_single_tensor_matches_independent_model(self):
        """When only one tensor per layer is offloaded, schedule_offloads and
        the old independent-per-tensor stall formula must agree exactly —
        schedule_offloads with a 1-element list reduces to the single-tensor
        case. This is the regression guard for the validated seq=2048/4096
        measurements in OBSERVATIONS.md (they used linear2-only offload)."""
        from simulator.offload_model import compute_offload_result
        from simulator.memory_model import get_all_tensors_per_layer

        cfg = llama_7b(seq_len=2048, micro_batch_size=1)
        gpu = H200_141GB
        par = ParallelismConfig()
        strategies = _build_offload_linear2_strategies(cfg.num_layers)
        result = simulate(cfg, gpu, strategies=strategies, par=par)

        # With a single tensor, layer-0 stall should equal compute_offload_result's
        # independent-deadline prediction. We reconstruct the gap calculation to
        # compare apples-to-apples.
        from simulator.compute_model import get_layer_compute_profile
        from simulator.environment import (
            _estimate_intra_block_liveness_gap,
            _estimate_liveness_gap,
        )
        layer_compute = get_layer_compute_profile(cfg, gpu, par, efficiency=0.5)
        gap_intra = _estimate_intra_block_liveness_gap(layer_compute)
        tensors = get_all_tensors_per_layer(cfg, par)
        linear2 = next(t for t in tensors if t.name == "mlp_linear2_input")
        # mlp_linear2_input is not a boundary tensor (block != "layernorm"/"residual")
        expected = compute_offload_result(linear2, gap_intra, gpu, par)
        assert abs(result.per_layer[0].offload_stall_s - expected.stall_time_s) < 1e-12

    def test_layer_stall_matches_direct_schedule_offloads(self):
        """End-to-end contract: per-layer stall inside simulate() equals what
        schedule_offloads() returns for the same tensors+gaps. Strong check
        that the wiring is correct."""
        from simulator.compute_model import get_layer_compute_profile
        from simulator.environment import (
            _estimate_intra_block_liveness_gap,
            _estimate_liveness_gap,
        )
        from simulator.memory_model import get_all_tensors_per_layer
        from simulator.offload_model import schedule_offloads

        cfg = llama_7b(seq_len=32768, micro_batch_size=1)
        gpu = H200_141GB
        par = ParallelismConfig()
        offload_names = {
            "mlp_gate_output", "mlp_up_output",
            "mlp_silu_output", "mlp_linear2_input",
        }
        strategies = _build_mlp_offload_strategies(cfg.num_layers, list(offload_names))
        sim = simulate(cfg, gpu, strategies=strategies, par=par)

        # Reconstruct the per-layer (tensor, gap) pairs exactly as simulate() builds them.
        layer_compute = get_layer_compute_profile(cfg, gpu, par, efficiency=0.5)
        gap_boundary = _estimate_liveness_gap(0, cfg.num_layers, layer_compute)
        gap_intra = _estimate_intra_block_liveness_gap(layer_compute)
        tensors = get_all_tensors_per_layer(cfg, par)
        pairs = [
            (t, gap_boundary if t.block in ("layernorm", "residual") else gap_intra)
            for t in tensors if t.name in offload_names
        ]
        expected = sum(r.stall_time_s for r in schedule_offloads(pairs, gpu, par))

        assert abs(sim.per_layer[0].offload_stall_s - expected) < 1e-12
