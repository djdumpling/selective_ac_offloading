"""Pipeline-parallel throughput validator.

Runs a Llama-style decoder stack under torch.distributed.pipelining's
Schedule1F1B, measures step latency and per-rank peak HBM, and compares
against the analytical simulator.

Launch (8× H200, one node):

  torchrun --standalone --nproc_per_node=4 throughput/run_pipeline.py \
      --pp 4 --ac pipeline-aware --model llama7b \
      --seq 4096 --mbs 1 --microbatches 8

Strategies (--ac):
  no-ac          : no activation checkpointing on any stage
  full-ac        : every LlamaDecoderLayer wrapped in checkpoint, on every stage
  pipeline-aware : early stages (higher stash) use full-ac, late stages use no-ac

Only 1F1B is supported in v1. Optimizer step is intentionally skipped so the
measured step time is apples-to-apples with the simulator's fwd+bwd+recompute
prediction.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)
from transformers import LlamaConfig, Qwen3Config
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from offload.hooks import CPUOffloadHook  # noqa: E402
from simulator.config import (  # noqa: E402
    H100_80GB,
    H200_141GB,
    ParallelismConfig,
    llama_13b,
    llama_7b,
    qwen3_8b,
)
from simulator.environment import (  # noqa: E402
    _stage_layer_span,
    simulate_pipeline_aware_ac,
    simulate_pipeline_uniform_ac,
)
from simulator.pipeline_schedules import PipelineSchedule  # noqa: E402
from throughput.strategies import (  # noqa: E402
    RUNNER_TO_SIM_STRATEGY,
    VALID_MODES,
    interleaved_chunk_layer_spans,
    pipeline_aware_stage_strategies,
    stage_strategies,
)


VALID_SCHEDULES = ("1f1b", "gpipe", "interleaved-1f1b")

RUNNER_TO_SIM_SCHEDULE = {
    "1f1b": PipelineSchedule.ONE_F_ONE_B,
    "gpipe": PipelineSchedule.GPIPE,
    "interleaved-1f1b": PipelineSchedule.ONE_F_ONE_B_INTERLEAVED,
}


# ── Model stage ──────────────────────────────────────────────────────────────

def build_hf_config(model: str, seq: int):
    """HuggingFace config matching one of the simulator presets."""
    if model == "llama7b":
        hidden, layers, heads, ffn = 4096, 32, 32, 11008
        kv_heads = heads
        cfg = LlamaConfig(
            hidden_size=hidden,
            intermediate_size=ffn,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            hidden_act="silu",
            max_position_embeddings=max(8192, seq),
            rms_norm_eps=1e-5,
            attention_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            vocab_size=32000,
            use_cache=False,
        )
    elif model == "llama13b":
        cfg = LlamaConfig(
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            hidden_act="silu",
            max_position_embeddings=max(8192, seq),
            rms_norm_eps=1e-5,
            attention_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            vocab_size=32000,
            use_cache=False,
        )
    elif model == "qwen3_8b":
        # Mirrors simulator.config.qwen3_8b: GQA (8 KV heads) + QK-norm.
        cfg = Qwen3Config(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            max_position_embeddings=max(8192, seq),
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
            vocab_size=151936,
            use_cache=False,
        )
    else:
        raise ValueError(f"unknown model: {model}")
    cfg._attn_implementation = "sdpa"  # SDPA dispatches to FA on H100/H200
    return cfg


class LlamaStage(nn.Module):
    """One pipeline stage: a contiguous slice of decoder layers plus optional
    embedding (first stage) and final norm (last stage)."""

    def __init__(
        self,
        hf_config: LlamaConfig,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        if is_first:
            self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hf_config, layer_idx=i)
            for i in range(layer_start, layer_end)
        ])
        if is_last:
            self.norm = LlamaRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        # Rotary is weight-free; cheapest to recompute per stage.
        self.rotary = LlamaRotaryEmbedding(hf_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            # x is input_ids [b, s] → hidden_states [b, s, h]
            x = self.embed_tokens(x)
        b, s, _ = x.shape
        position_ids = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        position_embeddings = self.rotary(x, position_ids)
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        if self.is_last:
            x = self.norm(x)
        return x


class Qwen3Stage(nn.Module):
    """Qwen3 analog of LlamaStage. Qwen3 adds per-head QK-norm and uses GQA
    (8 KV heads on a 32-head model), but the `x, position_ids, position_embeddings`
    call convention matches Llama's, so the stage wrapper is nearly identical."""

    def __init__(
        self,
        hf_config: Qwen3Config,
        layer_start: int,
        layer_end: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        if is_first:
            self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hf_config, layer_idx=i)
            for i in range(layer_start, layer_end)
        ])
        if is_last:
            self.norm = Qwen3RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(hf_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            x = self.embed_tokens(x)
        b, s, _ = x.shape
        position_ids = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        position_embeddings = self.rotary(x, position_ids)
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        if self.is_last:
            x = self.norm(x)
        return x


def build_stage_module(
    model: str,
    hf_config,
    layer_start: int,
    layer_end: int,
    is_first: bool,
    is_last: bool,
) -> nn.Module:
    """Factory: build the right stage class for the chosen model."""
    if model in ("llama7b", "llama13b"):
        return LlamaStage(hf_config, layer_start, layer_end, is_first, is_last)
    if model == "qwen3_8b":
        return Qwen3Stage(hf_config, layer_start, layer_end, is_first, is_last)
    raise ValueError(f"unknown model: {model}")


# ── AC / offload application ────────────────────────────────────────────────

def _build_offload_mlp_forward(offload_all: bool, hook_factory):
    """Return a patched `LlamaMLP.forward` that wraps the right sub-region in a
    CPUOffloadHook. When offload_all=True, the whole MLP body is hooked so every
    saved activation (gate_output, up_output, silu_output, linear2_input) goes
    to pinned CPU. When False, only down_proj is hooked so only linear2_input
    moves."""

    def forward_linear2_only(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        linear2_input = self.act_fn(gate) * up
        with hook_factory(self):
            return self.down_proj(linear2_input)

    def forward_all(self, x):
        with hook_factory(self):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            linear2_input = self.act_fn(gate) * up
            return self.down_proj(linear2_input)

    return forward_all if offload_all else forward_linear2_only


def _install_offload_on_mlps(stage_module: nn.Module, offload_all: bool,
                              stream: "torch.cuda.Stream | None",
                              hooks: list[CPUOffloadHook]) -> None:
    def hook_factory(_mlp):
        h = CPUOffloadHook(min_bytes=1_000_000, offload_stream=stream)
        hooks.append(h)
        return h

    patched = _build_offload_mlp_forward(offload_all, hook_factory)

    # Both Llama and Qwen3 decoder layers expose an `.mlp` child with a
    # monkey-patchable `forward`; wrap either when we encounter it.
    decoder_types = (LlamaDecoderLayer, Qwen3DecoderLayer)
    for mod in stage_module.modules():
        if isinstance(mod, decoder_types):
            mlp = mod.mlp
            if hasattr(mlp, "_original_forward_for_offload"):
                continue
            mlp._original_forward_for_offload = mlp.forward
            mlp.forward = types.MethodType(patched, mlp)


def apply_ac(
    stage_module: nn.Module,
    strategy: str,
    offload_stream: "torch.cuda.Stream | None" = None,
    offload_hooks: list[CPUOffloadHook] | None = None,
) -> None:
    """Install the per-stage strategy on `stage_module`.

    For offload strategies, the caller provides a dedicated CUDA stream so the
    GPU→CPU DMAs overlap with compute on the default stream; without it,
    transfers serialize and throughput collapses (see offload/hooks.py)."""
    if strategy == "no-ac":
        return
    if strategy == "full-ac":
        decoder_types = (LlamaDecoderLayer, Qwen3DecoderLayer)
        apply_activation_checkpointing(
            stage_module,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, preserve_rng_state=False),
            check_fn=lambda m: isinstance(m, decoder_types),
        )
        return
    if strategy in ("offload-linear2", "offload-all-mlp"):
        if offload_hooks is None:
            raise ValueError("offload_hooks list is required for offload strategies")
        _install_offload_on_mlps(
            stage_module,
            offload_all=(strategy == "offload-all-mlp"),
            stream=offload_stream,
            hooks=offload_hooks,
        )
        return
    raise ValueError(f"unknown per-stage strategy: {strategy}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama7b",
                   choices=["llama7b", "llama13b", "qwen3_8b"])
    p.add_argument("--pp", type=int, required=True)
    p.add_argument("--ac", default="pipeline-aware",
                   choices=list(VALID_MODES))
    p.add_argument("--schedule", default="1f1b", choices=list(VALID_SCHEDULES),
                   help="Pipeline schedule. 1f1b is Narayanan et al. 2021; "
                        "gpipe runs all forwards before any backward (worst memory); "
                        "interleaved-1f1b splits each rank into --num-chunks virtual stages "
                        "for a smaller bubble.")
    p.add_argument("--num-chunks", type=int, default=2,
                   help="Virtual stages per rank for --schedule interleaved-1f1b. "
                        "Ignored for other schedules.")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--mbs", type=int, default=1,
                   help="Per-microbatch batch size")
    p.add_argument("--microbatches", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--gpu", default="h200", choices=["h200", "h100"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--offload-sync-mode", default="overlap", choices=["overlap", "serial"],
        help=(
            "Which offload stream model the simulator should predict. "
            "'overlap' (default) matches the dedicated-stream fast path that "
            "CPUOffloadHook installs on GPU. 'serial' models the default-stream "
            "penalty (~+28%% at seq=2048 single-tensor offload) for users who "
            "run naive saved_tensors_hooks without a dedicated stream."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size != args.pp:
        raise SystemExit(
            f"world_size ({world_size}) must equal --pp ({args.pp}) in v1 "
            f"(no TP/DP yet)"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed + rank)

    # Build this rank's stage(s).
    hf_config = build_hf_config(args.model, args.seq)
    num_layers = hf_config.num_hidden_layers

    if args.model == "llama7b":
        sim_cfg = llama_7b(seq_len=args.seq, micro_batch_size=args.mbs)
    elif args.model == "llama13b":
        sim_cfg = llama_13b(seq_len=args.seq, micro_batch_size=args.mbs)
    elif args.model == "qwen3_8b":
        sim_cfg = qwen3_8b(seq_len=args.seq, micro_batch_size=args.mbs)
    else:
        raise SystemExit(f"unknown model: {args.model}")

    # Interleaved schedules split each rank into --num-chunks virtual stages;
    # 1f1b and gpipe use one stage per rank (num_chunks = 1).
    is_interleaved = args.schedule == "interleaved-1f1b"
    num_chunks = args.num_chunks if is_interleaved else 1
    num_virtual_stages = args.pp * num_chunks

    if is_interleaved:
        chunk_spans = interleaved_chunk_layer_spans(
            num_layers, args.pp, num_chunks, rank,
        )
    else:
        s, e = _stage_layer_span(num_layers, args.pp, rank)
        chunk_spans = [(rank, s, e)]

    stage_mods: list[nn.Module] = []
    for virtual_idx, layer_start, layer_end in chunk_spans:
        is_first_global = virtual_idx == 0
        is_last_global = virtual_idx == num_virtual_stages - 1
        mod = build_stage_module(
            args.model,
            hf_config,
            layer_start=layer_start,
            layer_end=layer_end,
            is_first=is_first_global,
            is_last=is_last_global,
        ).to(device=device, dtype=torch.bfloat16)
        stage_mods.append(mod)

    # Per-stage strategy assignment. For `pipeline-aware` we delegate to the
    # simulator, which walks STRATEGY_LEVELS least-aggressive-first and picks
    # the first that fits each stage's HBM budget given the schedule's stash.
    # The simulator is deterministic and torch-free, so every rank computes the
    # same list without coordination.
    sim_gpu = H200_141GB if args.gpu == "h200" else H100_80GB
    sim_par = ParallelismConfig(pp_size=args.pp)
    sim_schedule = RUNNER_TO_SIM_SCHEDULE[args.schedule]
    if args.ac == "pipeline-aware":
        strategies = pipeline_aware_stage_strategies(
            sim_cfg, sim_gpu, sim_par,
            schedule=sim_schedule,
            num_microbatches=args.microbatches,
        )
    else:
        strategies = stage_strategies(args.ac, args.pp)
    this_strat = strategies[rank]
    sync_mode = args.offload_sync_mode
    # Per-rank offload state: one shared CUDA stream across all MLP hooks on
    # this rank (including across interleaved chunks) so GPU→CPU DMAs serialize
    # on one PCIe lane but still overlap with compute on the default stream.
    offload_stream = None
    offload_hooks: list[CPUOffloadHook] = []
    if this_strat in ("offload-linear2", "offload-all-mlp"):
        offload_stream = torch.cuda.Stream(device=device)
    for mod in stage_mods:
        apply_ac(
            mod,
            this_strat,
            offload_stream=offload_stream,
            offload_hooks=offload_hooks,
        )

    if rank == 0:
        gpu_label = args.gpu.upper()
        print(
            f"=== Pipeline run: {args.model} pp={args.pp} ac={args.ac} "
            f"schedule={args.schedule} chunks={num_chunks} "
            f"seq={args.seq} mbs={args.mbs} μb={args.microbatches} "
            f"on {world_size}× {gpu_label} ==="
        )
        print(f"Per-rank strategies: {strategies}")
        print(f"Simulator offload_sync_mode: {sync_mode}")
    dist.barrier()
    nparams = sum(p.numel() for mod in stage_mods for p in mod.parameters())
    chunk_desc = ", ".join(
        f"[{ls},{le})" for _, ls, le in chunk_spans
    )
    print(
        f"[rank {rank}] virtual_stages={[v for v, _, _ in chunk_spans]} "
        f"layers={chunk_desc} strategy={strategies[rank]} "
        f"params={nparams / 1e9:.2f}B",
        flush=True,
    )
    dist.barrier()

    # Runtime shape inference from the first .step() call; no example needed.
    global_batch = args.mbs * args.microbatches
    stage_objs = [
        PipelineStage(mod, stage_index=v_idx, num_stages=num_virtual_stages, device=device)
        for mod, (v_idx, _, _) in zip(stage_mods, chunk_spans)
    ]

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simple scalar loss; scale down so the gradient norm stays reasonable.
        return pred.float().sum() * 1e-6

    if args.schedule == "1f1b":
        schedule = Schedule1F1B(
            stage_objs[0], n_microbatches=args.microbatches, loss_fn=loss_fn,
        )
    elif args.schedule == "gpipe":
        schedule = ScheduleGPipe(
            stage_objs[0], n_microbatches=args.microbatches, loss_fn=loss_fn,
        )
    elif args.schedule == "interleaved-1f1b":
        schedule = ScheduleInterleaved1F1B(
            stage_objs, n_microbatches=args.microbatches, loss_fn=loss_fn,
        )
    else:  # pragma: no cover - argparse choices guarantee this
        raise SystemExit(f"unknown schedule: {args.schedule}")

    input_ids = torch.randint(
        0, hf_config.vocab_size, (global_batch, args.seq), device=device
    )
    target = torch.zeros(
        global_batch, args.seq, hf_config.hidden_size,
        device=device, dtype=torch.bfloat16,
    )

    def one_step():
        if rank == 0 and rank == args.pp - 1:
            schedule.step(input_ids, target=target)
        elif rank == 0:
            schedule.step(input_ids)
        elif rank == args.pp - 1:
            schedule.step(target=target)
        else:
            schedule.step()

    # Warmup (also compiles/traces any lazy state)
    for i in range(args.warmup):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.reset_peak_memory_stats()

    # Timed window
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    end_evt.record()
    torch.cuda.synchronize()
    wall_s = time.time() - t0
    dist.barrier()

    step_ms = start_evt.elapsed_time(end_evt) / args.steps
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    offload_gb = sum(h.stats.bytes_offloaded for h in offload_hooks) / (1024 ** 3) / max(1, args.steps)

    # Gather per-rank numbers
    all_step_ms = [0.0] * world_size
    all_peak_gb = [0.0] * world_size
    all_offload_gb = [0.0] * world_size
    dist.all_gather_object(all_step_ms, step_ms)
    dist.all_gather_object(all_peak_gb, peak_gb)
    dist.all_gather_object(all_offload_gb, offload_gb)

    if rank == 0:
        print()
        print("=== Results ===")
        print(f"  {'rank':>4} {'strategy':<18} {'step (ms)':>11} {'peak HBM (GB)':>14} {'offload/step (GB)':>18}")
        print(f"  {'-'*4} {'-'*18} {'-'*11} {'-'*14} {'-'*18}")
        for r, (sm, pg, og) in enumerate(zip(all_step_ms, all_peak_gb, all_offload_gb)):
            print(f"  {r:>4} {strategies[r]:<18} {sm:>11.1f} {pg:>14.2f} {og:>18.2f}")

        bottleneck_ms = max(all_step_ms)
        tokens_per_s = global_batch * args.seq / (bottleneck_ms / 1000)
        print()
        print(f"  Bottleneck step:   {bottleneck_ms:.1f} ms")
        print(f"  Throughput:        {tokens_per_s:,.0f} tokens/s")
        print(f"  Wall clock / step: {wall_s / args.steps * 1000:.1f} ms")

        # ── Simulator comparison ─────────────────────────────────────────
        # Reuse sim_gpu / sim_par / sim_schedule from the strategy-assignment step
        # above so the predicted per-stage strategies exactly match what ran on GPU.
        if args.ac == "pipeline-aware":
            pr_sim = simulate_pipeline_aware_ac(
                sim_cfg, sim_gpu, sim_par,
                schedule=sim_schedule,
                num_microbatches=args.microbatches,
                num_chunks=num_chunks,
                offload_sync_mode=sync_mode,
            )
        else:
            name = RUNNER_TO_SIM_STRATEGY[args.ac]
            pr_sim = simulate_pipeline_uniform_ac(
                sim_cfg, sim_gpu, sim_par,
                strategy_name=name,
                schedule=sim_schedule,
                num_microbatches=args.microbatches,
                num_chunks=num_chunks,
                offload_sync_mode=sync_mode,
            )

        # Simulator's step_latency_s is per-microbatch at the bottleneck stage.
        # Convert to per-step (= per training iteration) by multiplying by M,
        # then add the bubble fraction the simulator already tracks.
        per_mb_ms = pr_sim.bottleneck_step_latency_s * 1000
        pred_step_ms = per_mb_ms * args.microbatches
        pred_with_bubble_ms = pr_sim.overall_step_latency_s * args.microbatches * 1000

        sim_strats = [s.strategy_name for s in pr_sim.stages]
        print()
        print("=== Simulator prediction ===")
        print(f"  Sim per-stage strategies: {sim_strats}")
        print(f"  Per-microbatch @ bottleneck: {per_mb_ms:.1f} ms")
        print(f"  Predicted step (no bubble):  {pred_step_ms:.1f} ms")
        print(f"  Predicted step (w/ bubble):  {pred_with_bubble_ms:.1f} ms "
              f"(bubble={pr_sim.bubble_fraction:.1%})")
        err_no_bubble = (bottleneck_ms / pred_step_ms - 1) * 100
        err_bubble = (bottleneck_ms / pred_with_bubble_ms - 1) * 100
        print(f"  Measured vs predicted:       "
              f"{err_no_bubble:+.1f}% (no bubble) / {err_bubble:+.1f}% (with bubble)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
