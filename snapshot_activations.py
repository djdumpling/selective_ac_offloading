"""Snapshot activation tensors by hooking into autograd's saved tensors.

This directly counts what autograd retains for backward, which is exactly
what the simulator models.

Usage: python snapshot_activations.py
"""

import gc
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaModel

# ── Config (same as validate_on_gpu.py) ─────────────────────────────────────

SEQ_LEN = 2048
MICRO_BATCH_SIZE = 1
DTYPE = torch.bfloat16

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    max_position_embeddings=4096,
    rms_norm_eps=1e-5,
    attention_dropout=0.0,
    attention_bias=False,
    mlp_bias=False,
    vocab_size=32000,
    use_cache=False,
    torch_dtype=DTYPE,
)
LLAMA_CONFIG._attn_implementation = "sdpa"


def main():
    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build model
    model = LlamaModel(LLAMA_CONFIG).to(dtype=DTYPE, device="cuda")
    model.train()

    input_ids = torch.randint(0, 32000, (MICRO_BATCH_SIZE, SEQ_LEN), device="cuda")

    # Warm up
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, use_cache=False)
        out.last_hidden_state.sum().backward()
        del out
    gc.collect()
    torch.cuda.empty_cache()

    # ── Method: Use saved_tensors_hooks to intercept autograd saves ──────
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append({
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "size_bytes": tensor.nelement() * tensor.element_size(),
            "data_ptr": tensor.data_ptr(),
        })
        return tensor

    def unpack_hook(tensor):
        return tensor

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outputs = model(input_ids=input_ids, use_cache=False)
        loss = outputs.last_hidden_state.sum()

    # Now saved_tensors has every tensor that autograd saved during forward
    print(f"\n=== Autograd saved tensors (total: {len(saved_tensors)}) ===\n")

    # Deduplicate by data_ptr (same tensor saved by multiple ops)
    unique_by_ptr = {}
    for t in saved_tensors:
        ptr = t["data_ptr"]
        if ptr not in unique_by_ptr:
            unique_by_ptr[ptr] = t
        # Keep the first occurrence (could track count too)

    unique = list(unique_by_ptr.values())
    print(f"Unique tensors (by data_ptr): {len(unique)}\n")

    # Group by size
    from collections import defaultdict
    size_groups = defaultdict(list)
    for t in unique:
        mib = round(t["size_bytes"] / (1024 * 1024), 2)
        size_groups[mib].append(t)

    total_bytes = 0
    print(f"{'Size (MiB)':>12}  {'Count':>6}  {'Per-layer':>10}  {'Shape':>30}  {'Dtype'}")
    print(f"{'-'*12}  {'-'*6}  {'-'*10}  {'-'*30}  {'-'*10}")

    for mib in sorted(size_groups.keys(), reverse=True):
        tensors = size_groups[mib]
        count = len(tensors)
        total_bytes += sum(t["size_bytes"] for t in tensors)
        per_layer = f"{count/32:.1f}/layer" if count >= 32 else ""
        # Show representative shape
        shapes = set(str(t["shape"]) for t in tensors)
        shape_str = ", ".join(sorted(shapes)[:2])
        if len(shapes) > 2:
            shape_str += f" +{len(shapes)-2}"
        dtype = tensors[0]["dtype"]
        print(f"{mib:>12.2}  {count:>6}  {per_layer:>10}  {shape_str:>30}  {dtype}")

    print(f"\n{'TOTAL':>12}  {len(unique):>6}  {'':>10}  {total_bytes/1024**3:.3f} GiB")
    print(f"\nMeasured retained activations (-out): 11.556 GiB")
    print(f"Model output: {SEQ_LEN * MICRO_BATCH_SIZE * 4096 * 2 / 1024**3:.3f} GiB")

    del outputs, loss


if __name__ == "__main__":
    main()
