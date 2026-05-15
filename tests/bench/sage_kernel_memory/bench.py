"""Per-kernel-call CUDA memory profile for sage attention.

Isolates sage's memory behavior from any surrounding ComfyUI / model
state. For each config, wraps a single `sageattn()` call in
`torch.cuda.memory._record_memory_history()` and dumps a snapshot
plus peak/current allocation deltas. Load the snapshot at
https://docs.pytorch.org/memory_viz to drill into per-tensor lifetimes.

Useful for answering "does my CUDA sage kernel allocate more than I
expect at LTX 2.3 self-attn shape?" without ComfyUI's load/unload/
dynamic-vram behavior in the picture.

Complements `tests/bench/partitioned_mask_phase0/` which measures the
cumulative peak across a multi-call partition pattern; this bench
measures the per-config single-call working set at two LTX self-attn
shapes (T=23296 single-pass; T=42240 multi-guide-expanded) crossed
with two mask kinds (unmasked + broadcast-row `(1,1,1,T)`). The
per-Q-row "tracked" mask only makes sense in a Q-partition pattern
(Q sliced to tracked_count rows); see Phase 0 bench for that case.

Routes through `sageattention.sageattn()` (the dispatcher) -- so on
sm89 + CUDA >= 12.8 with v0.5.5 active, masked calls land on
`fp8_cuda++` with `MaskMode::kGeneral`. Other archs/versions land
on the Triton fallback.

Adapted from a sister-clone's `scripts/bench_sage_kernel_memory.py`
(2026-05-14); kept in tests/bench/ for cross-clone consistency.

Usage
-----
    ${VIRTUAL_ENV}/bin/python tests/bench/sage_kernel_memory/bench.py --output-dir tests/bench/sage_kernel_memory/run_<date>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import orjson
import torch


# LTX 2.3 video self-attn is B=1, H=32, D=128 across shipped workflows.
B, H, D = 1, 32, 128
DTYPE = torch.bfloat16
DEVICE = "cuda"

# T_VALUES: T=23296 is the single-pass LTX video self-attn loop-iter
# seq the existing tests/test_sageattn_ltx_shapes.py covers; T=42240
# is the multi-guide-token-expanded shape that real multi-guide
# workflows hit and is NOT in the regular bench harness today.
T_VALUES = [23296, 42240]
MASK_KINDS: list[str | None] = [None, "noisy"]

# Bound trace memory: a single kernel call records well under 20k events;
# default 200k produced ~MB-scale .bin per config.
MEMORY_HISTORY_MAX_ENTRIES = 20_000


def make_inputs(T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def make_mask(mask_q_dim: int, T: int) -> torch.Tensor:
    # mask_q_dim=1 -> broadcast across all Q rows.
    return torch.zeros(1, 1, mask_q_dim, T, dtype=DTYPE, device=DEVICE)


def profile_call(name: str, fn, snapshot_path: Path) -> dict[str, Any]:
    """Run `fn` once with memory history recording. Return peak/delta + snapshot path."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=MEMORY_HISTORY_MAX_ENTRIES)

    pre_mb = torch.cuda.memory_allocated() / 1024 / 1024
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    post_mb = torch.cuda.memory_allocated() / 1024 / 1024
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    del out
    torch.cuda.empty_cache()

    return {
        "name": name,
        "elapsed_ms": round(elapsed_ms, 2),
        "pre_alloc_mb": round(pre_mb, 1),
        "post_alloc_mb": round(post_mb, 1),
        "peak_alloc_mb": round(peak_mb, 1),
        "delta_mb": round(peak_mb - pre_mb, 1),
        "snapshot": str(snapshot_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where snapshots + summary.json land",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import sageattention
    except ImportError as e:
        print(f"ERROR: sageattention not importable: {e}", file=sys.stderr)
        print("Install the fork (or upstream) sage first.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"Sage version: {getattr(sageattention, '__version__', '<no __version__>')}")
    print(f"Sage module: {sageattention.__file__}")
    print()

    results: list[dict[str, Any]] = []

    # Outer loop on T so Q/K/V at the same shape are allocated once and
    # reused across the 3 mask configs sharing that T.
    for T in T_VALUES:
        q, k, v = make_inputs(T)
        for mask_kind in MASK_KINDS:
            if mask_kind == "noisy":
                mask = make_mask(mask_q_dim=1, T=T)
                name = f"masked_noisy_t{T}"
            else:
                mask = None
                name = f"unmasked_t{T}"

            def run():
                if mask is None:
                    return sageattention.sageattn(q, k, v, tensor_layout="HND")
                return sageattention.sageattn(q, k, v, attn_mask=mask, tensor_layout="HND")

            # Warmup once to absorb Triton autotune / lazy compile.
            run()
            torch.cuda.synchronize()

            result = profile_call(name, run, args.output_dir / f"{name}.snapshot.bin")
            result.update({"T": T, "mask_kind": mask_kind or "none"})
            results.append(result)
            print(f"  {name:30s} delta={result['delta_mb']:>8.1f} MB  peak={result['peak_alloc_mb']:>8.1f} MB  elapsed={result['elapsed_ms']:>7.2f} ms")

            if mask is not None:
                del mask
            torch.cuda.empty_cache()

        del q, k, v
        torch.cuda.empty_cache()

    # Record the sage module path as basename only to avoid leaking
    # the absolute venv path into committed bench output.
    # sageattention.__file__ is typed `str | None` per Pyright; guard.
    sage_module_path = Path(sageattention.__file__ or "<unknown>")
    sage_module_basename = f".../{sage_module_path.parent.name}/{sage_module_path.name}"

    summary = {
        "device": device_name,
        "sage_version": getattr(sageattention, "__version__", None),
        "sage_module": sage_module_basename,
        "dtype": str(DTYPE),
        "B": B,
        "H": H,
        "D": D,
        "configs": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    print()
    print(f"Wrote summary -> {summary_path}")
    print(f"Snapshots in   -> {args.output_dir}/")
    print("Load any *.snapshot.bin at https://docs.pytorch.org/memory_viz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
