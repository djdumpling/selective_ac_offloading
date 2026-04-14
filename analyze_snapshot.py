"""Analyze a CUDA memory snapshot to list retained activation tensors.

Usage: python analyze_snapshot.py memory_snapshot_no_ac.pickle
"""

import pickle
import sys
from collections import defaultdict


def extract_python_frame(frames):
    """Find the most informative Python frame in a stack trace."""
    # Look through ALL frames for any Python source file
    best = None
    for frame in frames:
        filename = frame.get("filename", "")
        if not filename or filename == "??":
            continue
        # Skip C++ internal frames
        if any(x in filename for x in [".cpp:", "CUDACaching", "RegisterCUDA",
                                         "RegisterBackend", "RegisterComposite",
                                         "VariableType", "LinearAlgebra"]):
            continue
        name = frame.get("name", "")
        line = frame.get("line", 0)
        short = filename.split("/")[-1]
        candidate = f"{short}:{line}:{name}"
        # Prefer modeling_llama frames
        if "modeling_llama" in filename or "modeling_rope" in filename:
            return candidate
        if best is None:
            best = candidate
    return best or "unknown"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "memory_snapshot_no_ac.pickle"

    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    if isinstance(snapshot, dict):
        segments = snapshot.get("segments", [])
    elif isinstance(snapshot, list):
        segments = snapshot
    else:
        print(f"Unknown snapshot type: {type(snapshot)}")
        return

    # Debug: dump frame structure for one block to understand format
    print("=== Frame structure debug ===")
    for segment in segments[:3]:
        if not isinstance(segment, dict):
            continue
        for block in segment.get("blocks", [])[:1]:
            if block.get("state") == "active_allocated":
                frames = block.get("frames", [])
                print(f"Block size: {block['size']}, num frames: {len(frames)}")
                if frames:
                    print(f"Frame[0] keys: {list(frames[0].keys()) if isinstance(frames[0], dict) else type(frames[0])}")
                    for i, f in enumerate(frames[:5]):
                        print(f"  [{i}] {f}")
                # Also check if there's a 'history' field
                history = block.get("history", [])
                if history:
                    print(f"History entries: {len(history)}")
                    h0 = history[0] if history else None
                    if h0:
                        print(f"  history[0] keys: {list(h0.keys()) if isinstance(h0, dict) else type(h0)}")
                        h_frames = h0.get("frames", [])
                        if h_frames:
                            print(f"  history[0].frames[0]: {h_frames[0]}")
                break
        break

    # Also check segment-level keys
    if segments and isinstance(segments[0], dict):
        print(f"\nSegment keys: {list(segments[0].keys())}")
        blocks_sample = segments[0].get("blocks", [])
        if blocks_sample and isinstance(blocks_sample[0], dict):
            print(f"Block keys: {list(blocks_sample[0].keys())}")

    # Now collect all active blocks with frame info from history
    blocks = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                size = block["size"]
                # Try frames directly
                frames = block.get("frames", [])
                # Also try history (newer PyTorch stores frames in history)
                if not frames:
                    history = block.get("history", [])
                    if history and isinstance(history[0], dict):
                        frames = history[0].get("frames", [])
                source = extract_python_frame(frames)
                blocks.append((size, source, frames))

    # Sort by size descending
    blocks.sort(key=lambda x: -x[0])

    total = 0
    size_by_source = defaultdict(lambda: [0, 0])

    for size, source, frames in blocks:
        total += size
        size_by_source[source][0] += size
        size_by_source[source][1] += 1

    # Size distribution
    print(f"\n=== Allocation size distribution ===")
    print(f"Total: {total / 1024**3:.3f} GiB across {len(blocks)} active blocks\n")

    size_buckets = defaultdict(list)
    for size, source, _ in blocks:
        mib = round(size / (1024 * 1024), 1)
        size_buckets[mib].append(source)

    for mib in sorted(size_buckets.keys(), reverse=True):
        sources = size_buckets[mib]
        count = len(sources)
        total_mib = mib * count
        layers = ""
        if count % 32 == 0 and count > 0:
            per_layer = count // 32
            layers = f"  ({per_layer}/layer × 32)"
        unique_sources = set(sources)
        source_str = ", ".join(sorted(unique_sources)[:3])
        if len(unique_sources) > 3:
            source_str += f" +{len(unique_sources)-3} more"
        print(f"  {mib:>8.1f} MiB × {count:>4} = {total_mib:>8.1f} MiB{layers}  [{source_str}]")

    # Now do the accounting analysis
    print(f"\n=== Accounting analysis ===")
    print(f"Known Llama-7B tensor sizes (s=2048, b=1, h=4096, ffn=11008, bf16):")
    print(f"  Parameter weights:")
    print(f"    gate/up/down_proj: ffn×h×2 = 11008×4096×2 = 86.0 MiB")
    print(f"    q/k/v/o_proj:     h×h×2   = 4096×4096×2  = 32.0 MiB")
    print(f"    embed_tokens:     vocab×h×2 = 32000×4096×2 = 250.0 MiB")
    print(f"  Activation tensors:")
    print(f"    ffn-sized (gate/up/silu/linear2): s×b×ffn×2 = 43.0 MiB")
    print(f"    hidden-sized (Q/K/V/etc):         s×b×h×2   = 16.0 MiB")
    print(f"    LN fp32 copy:                     s×b×h×4   = 32.0 MiB")

    # Params: per layer = 3×86 MiB (MLP) + 4×32 MiB (attn) = 386 MiB
    # + embed = 250 MiB, + final LN ≈ 0
    param_per_layer_mib = 3 * 86 + 4 * 32  # 386 MiB
    param_total_mib = param_per_layer_mib * 32 + 250  # + embedding
    print(f"\n  Expected params: {param_total_mib} MiB = {param_total_mib/1024:.1f} GiB")
    print(f"  Measured params: 12.370 GiB")

    # Count 86 MiB blocks: should be 3/layer (gate, up, down) = 96 for params
    # Remaining 86 MiB blocks would be activation pairs?
    n_86 = len(size_buckets.get(86.0, []))
    n_32 = len(size_buckets.get(32.0, []))
    n_250 = len(size_buckets.get(250.0, []))
    n_16 = len(size_buckets.get(16.0, []))

    print(f"\n  86 MiB blocks: {n_86} total")
    print(f"    Expected param (gate/up/down × 32 layers): 96")
    print(f"    Remaining (activations?): {n_86 - 96}")
    print(f"    If activation: {n_86 - 96} × 86 MiB ÷ 32 layers = {(n_86-96)/32:.1f}/layer × 86 MiB")
    print(f"    Note: 86 MiB = 2 × 43 MiB (two ffn-sized activations coalesced)")

    print(f"\n  32 MiB blocks: {n_32} total")
    print(f"    Expected param (q/k/v/o × 32 layers): 128")
    print(f"    Remaining (activations?): {n_32 - 128}")
    print(f"    If activation: {n_32 - 128} × 32 MiB ÷ 32 layers = {(n_32-128)/32:.1f}/layer × 32 MiB")
    print(f"    Note: 32 MiB = LN fp32 copy OR 2 × 16 MiB hidden-sized tensors coalesced")

    print(f"\n  16 MiB blocks: {n_16} total")
    print(f"    These are individual hidden-sized activation tensors")

    # Total activation estimate
    act_86 = (n_86 - 96) * 86
    act_32 = (n_32 - 128) * 32
    act_16 = n_16 * 16
    act_total = act_86 + act_32 + act_16
    print(f"\n  Estimated activation memory from block counts:")
    print(f"    From 86 MiB blocks: {act_86} MiB")
    print(f"    From 32 MiB blocks: {act_32} MiB")
    print(f"    From 16 MiB blocks: {act_16} MiB")
    print(f"    TOTAL: {act_total} MiB = {act_total/1024:.3f} GiB")
    print(f"    Measured retained activations: 11.556 GiB")


if __name__ == "__main__":
    main()
