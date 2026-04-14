"""Analyze a CUDA memory snapshot to list retained activation tensors.

Usage: python analyze_snapshot.py memory_snapshot_no_ac.pickle
"""

import pickle
import sys
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "memory_snapshot_no_ac.pickle"

    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    segments = snapshot.get("segments", []) if isinstance(snapshot, dict) else snapshot

    # Collect all active blocks with requested_size
    blocks = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                alloc_size = block["size"]
                req_size = block.get("requested_size", alloc_size)
                blocks.append((req_size, alloc_size))

    blocks.sort(key=lambda x: -x[0])

    # Size distribution by REQUESTED size
    print(f"=== Distribution by REQUESTED size (actual tensor size) ===")
    print(f"Total: {len(blocks)} active blocks\n")

    req_buckets = defaultdict(int)
    for req, alloc in blocks:
        mib = round(req / (1024 * 1024), 1)
        if mib >= 0.1:
            req_buckets[mib] += 1

    for mib in sorted(req_buckets.keys(), reverse=True):
        count = req_buckets[mib]
        total_mib = mib * count
        layers = ""
        if count % 32 == 0 and count >= 32:
            per = count // 32
            layers = f"  ({per}/layer × 32)"
        elif count == 1:
            layers = "  (one-time)"
        elif count == 2:
            layers = "  (one-time, ×2)"
        print(f"  {mib:>8.1f} MiB × {count:>4} = {total_mib:>8.1f} MiB{layers}")

    # Known tensor sizes for Llama-7B
    print(f"\n=== Reference: Llama-7B tensor sizes ===")
    print(f"  PARAMETERS:")
    print(f"    250.0 MiB = embed_tokens [32000×4096] bf16")
    print(f"     86.0 MiB = gate/up/down_proj [11008×4096] bf16")
    print(f"     32.0 MiB = q/k/v/o_proj [4096×4096] bf16")
    print(f"  ACTIVATIONS:")
    print(f"     43.0 MiB = ffn-sized: gate, up, silu(gate), silu(gate)*up  [2048×11008] bf16")
    print(f"     32.0 MiB = LN fp32 copy: ln1_input, ln2_input  [2048×4096] fp32")
    print(f"     16.0 MiB = hidden-sized: qkv_input, Q, K, V, rotary_q/k, FA_out, o_proj_in, mlp_in  [2048×4096] bf16")
    print(f"      0.25 MiB = FA logsumexp [1×32×2048] fp32")

    # Accounting
    print(f"\n=== Per-layer activation accounting ===")

    # Expected parameter block counts
    param_86 = 3 * 32   # gate, up, down per layer
    param_32 = 4 * 32   # q, k, v, o per layer

    n_43 = req_buckets.get(43.0, 0)
    n_32 = req_buckets.get(32.0, 0)
    n_16 = req_buckets.get(16.0, 0)
    n_86 = req_buckets.get(86.0, 0)
    n_250 = req_buckets.get(250.0, 0)
    n_025 = req_buckets.get(0.2, 0) + req_buckets.get(0.3, 0)  # logsumexp ~0.25 MiB

    act_43 = n_43  # all 43 MiB blocks are activations (params are 86 MiB)
    act_32 = n_32 - param_32  # subtract param weights
    act_16 = n_16  # all 16 MiB are activations

    print(f"  43 MiB blocks (ffn activations):  {n_43:>4} total = {n_43:>4} activation  →  {n_43/32:.1f}/layer")
    print(f"  32 MiB blocks (params + LN fp32): {n_32:>4} total - {param_32} params = {act_32:>4} activation  →  {act_32/32:.1f}/layer")
    print(f"  16 MiB blocks (hidden-dim acts):  {n_16:>4} total = {n_16:>4} activation")
    print(f"  86 MiB blocks (params only?):     {n_86:>4} total - {param_86} params = {n_86-param_86:>4} remaining")

    # Total activation memory
    act_total_mib = act_43 * 43 + act_32 * 32 + act_16 * 16
    print(f"\n  Activation total: {act_43}×43 + {act_32}×32 + {act_16}×16 = {act_total_mib} MiB = {act_total_mib/1024:.3f} GiB")
    print(f"  Measured retained activations (-out): 11.556 GiB")

    # What the simulator predicts per layer
    print(f"\n=== Simulator vs measured per-layer tensor counts ===")
    print(f"  Simulator expects per layer (No AC, Llama-7B with RoPE):")
    print(f"    43 MiB (ffn-sized):   4  (gate, up, silu_out, linear2_in)")
    print(f"    32 MiB (LN fp32):     2  (ln1_input, ln2_input)")
    print(f"    16 MiB (hidden-sized): 8  (qkv_input, q, k, v, rotary_q, rotary_k, fa_output, o_proj_input)")
    print(f"     0.25 MiB (FA lse):   1")
    print(f"    Total per layer: 4×43 + 2×32 + 8×16 + 0.25 = {4*43 + 2*32 + 8*16 + 0.25:.1f} MiB")
    print(f"    Total 32 layers: {(4*43 + 2*32 + 8*16 + 0.25)*32:.0f} MiB = {(4*43 + 2*32 + 8*16 + 0.25)*32/1024:.3f} GiB")
    print(f"")
    print(f"  GPU snapshot shows per layer:")
    print(f"    43 MiB (ffn-sized):   {n_43/32:.1f}")
    print(f"    32 MiB (LN/mixed):    {act_32/32:.1f}")
    print(f"    16 MiB (hidden-sized): {act_16/32:.1f} (+ possibly coalesced into 32 MiB blocks)")


if __name__ == "__main__":
    main()
