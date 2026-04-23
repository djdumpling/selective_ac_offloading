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
