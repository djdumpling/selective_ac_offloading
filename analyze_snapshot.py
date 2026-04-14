"""Analyze a CUDA memory snapshot to list retained activation tensors.

Usage: python analyze_snapshot.py memory_snapshot_no_ac.pickle
"""

import pickle
import sys
from collections import defaultdict


def extract_python_frame(frames):
    """Find the most informative Python frame in a stack trace."""
    for frame in frames:
        filename = frame.get("filename", "")
        if "modeling_llama" in filename or "modeling_rope" in filename:
            name = frame.get("name", "")
            line = frame.get("line", 0)
            return f"{filename.split('/')[-1]}:{line}:{name}"
    for frame in frames:
        filename = frame.get("filename", "")
        if "validate_on_gpu" in filename or "selective_ac" in filename:
            name = frame.get("name", "")
            line = frame.get("line", 0)
            return f"{filename.split('/')[-1]}:{line}:{name}"
    return "unknown"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "memory_snapshot_no_ac.pickle"

    with open(path, "rb") as f:
        snapshot = pickle.load(f)

    # Detect snapshot format
    if isinstance(snapshot, dict):
        segments = snapshot.get("segments", [])
    elif isinstance(snapshot, list):
        segments = snapshot
    else:
        print(f"Unknown snapshot type: {type(snapshot)}")
        print(f"Keys: {snapshot.keys() if hasattr(snapshot, 'keys') else 'N/A'}")
        return

    # Debug: show structure if segments aren't dicts
    if segments and not isinstance(segments[0], dict):
        print(f"Snapshot top-level type: {type(snapshot)}")
        if isinstance(snapshot, dict):
            print(f"Keys: {list(snapshot.keys())}")
            for k, v in snapshot.items():
                print(f"  {k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")
                if isinstance(v, list) and v:
                    print(f"    [0] type={type(v[0])}")
                    if isinstance(v[0], dict):
                        print(f"    [0] keys={list(v[0].keys())[:10]}")
        return

    blocks = []
    for segment in segments:
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                size = block["size"]
                frames = block.get("frames", [])
                source = extract_python_frame(frames)
                blocks.append((size, source, frames))

    # Sort by size descending
    blocks.sort(key=lambda x: -x[0])

    # Print all allocations > 100KB
    print(f"\nActive allocations > 100 KB (total: {len(blocks)} blocks)")
    print(f"{'Size':>12}  {'MiB':>8}  Source")
    print(f"{'-'*12}  {'-'*8}  {'-'*60}")

    total = 0
    size_by_source = defaultdict(lambda: [0, 0])  # [total_bytes, count]

    for size, source, frames in blocks:
        total += size
        size_by_source[source][0] += size
        size_by_source[source][1] += 1
        if size >= 100 * 1024:
            mib = size / (1024 * 1024)
            print(f"{size:>12,}  {mib:>8.2f}  {source}")

    print(f"\n{'TOTAL':>12}  {total / 1024**3:>8.3f} GiB")

    # Summary by source
    print(f"\n\nAggregated by source (top 20):")
    print(f"{'Total':>12}  {'Count':>6}  {'Per-block':>10}  Source")
    print(f"{'-'*12}  {'-'*6}  {'-'*10}  {'-'*60}")
    for source, (tot, count) in sorted(
        size_by_source.items(), key=lambda x: -x[1][0]
    )[:20]:
        per = tot / count / (1024 * 1024)
        print(f"{tot/1024**2:>10.1f}MB  {count:>6}  {per:>8.1f}MB  {source}")

    # Count by size bucket for layer analysis
    print(f"\n\nAllocation size distribution (activation-sized blocks):")
    size_counts = defaultdict(int)
    for size, source, _ in blocks:
        mib = round(size / (1024 * 1024), 1)
        if mib >= 0.1:
            size_counts[mib] += 1
    for mib in sorted(size_counts.keys(), reverse=True):
        count = size_counts[mib]
        layers = ""
        if count % 32 == 0:
            layers = f"  ({count // 32} per layer × 32 layers)"
        elif count == 1:
            layers = "  (one-time)"
        print(f"  {mib:>8.1f} MiB: {count:>4} blocks{layers}")


if __name__ == "__main__":
    main()
