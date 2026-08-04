#!/usr/bin/env python3
"""Test `sageattn_consume()`: same output as `sageattn()`, lower peak VRAM.

Attention is the memory peak in a large DiT block, and most of that peak
is the caller's float q/k/v sitting alive underneath sage's working set
rather than sage's own allocations. `sageattn()` cannot fix that -- the
caller's frame holds the references. `sageattn_consume()` takes a
`[q, k, v]` list, empties it, and releases each tensor as soon as its
quantized form exists.

The saving depends on a real ownership chain holding all the way down:
the list must be the last owner, the wrapper must not bind the tensors
into its own frame, and the kernel wrapper must drop its parameters
before allocating the output. Any one of those slipping silently
reverts the behaviour to `sageattn()`'s -- still correct, just with no
saving -- which is exactly why the peak is asserted numerically here
rather than eyeballed.

Claims, i.e. what breaks if a case is deleted:
  - matches_sageattn_output : that the memory optimization did not
                              change the math. Delete and a reordering
                              that corrupts results looks like a win.
  - empties_the_list        : the documented contract callers rely on to
                              know their references are gone.
  - peak_vram_is_lower      : the entire reason the function exists.
                              Delete and the ownership chain can rot
                              while every other test still passes.
  - rejects_malformed_input : the list is mutated, so a wrong-shaped
                              argument must fail loudly rather than
                              half-consume it.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
Needs CUDA. The peak-VRAM case needs ~6 GiB free and skips without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sageattention
from sageattention import sageattn, sageattn_consume

# MiniMax H3's attention config, at a shape small enough to run anywhere
# while still making the float tensors dominate the peak.
H, D = 56, 128
S_SMALL = 4096
S_PEAK = 41822  # fl2va at the node's default canvas
PEAK_VRAM_BYTES = 6 * 2**30


def _qkv(s, dtype=torch.bfloat16):
    return [
        torch.randn(1, H, s, D, device="cuda", dtype=dtype) for _ in range(3)
    ]


def test_matches_sageattn_output():
    torch.manual_seed(0)
    q, k, v = _qkv(S_SMALL)
    expect = sageattn(q, k, v, tensor_layout="HND", is_causal=False, smooth_k=False)
    box = [q, k, v]
    del q, k, v
    got = sageattn_consume(box, tensor_layout="HND", is_causal=False, smooth_k=False)
    assert got.shape == expect.shape, f"{got.shape} != {expect.shape}"
    # Same kernel, same inputs, same order of operations -> bit-identical.
    # Compared through a uint16 view because bf16 NaNs would make a plain
    # torch.equal return False at identical bit patterns.
    assert torch.equal(got.view(torch.uint16), expect.view(torch.uint16)), (
        "sageattn_consume must be bit-identical to sageattn; the early "
        "release changes only when tensors are freed, not the math"
    )
    print("  bit-identical to sageattn()")


def test_empties_the_list():
    box = _qkv(S_SMALL)
    sageattn_consume(box, tensor_layout="HND", smooth_k=False)
    assert all(t is None for t in box), (
        f"qkv entries must all be None after the call, got "
        f"{[type(t).__name__ for t in box]}"
    )
    print("  list emptied")


def test_rejects_malformed_input():
    for bad in ([], [torch.empty(0)], _qkv(64) + [None]):
        try:
            sageattn_consume(bad, tensor_layout="HND")
        except ValueError:
            continue
        raise AssertionError(
            f"expected ValueError for a {len(bad)}-element qkv list"
        )
    print("  malformed input rejected")


def _peak_over_call(use_consume):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    q, k, v = _qkv(S_PEAK)
    if use_consume:
        box = [q, k, v]
        del q, k, v
        o = sageattn_consume(box, tensor_layout="HND", smooth_k=False)
    else:
        o = sageattn(q, k, v, tensor_layout="HND", smooth_k=False)
        del q, k, v
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del o
    torch.cuda.empty_cache()
    return peak


def test_peak_vram_is_lower():
    # consume first: a prior arm trains the caching allocator and biases
    # whatever runs second (CLAUDE.md, peak-HBM measurement discipline).
    consume_peak = _peak_over_call(use_consume=True)
    plain_peak = _peak_over_call(use_consume=False)
    saved_mib = (plain_peak - consume_peak) / 2**20
    print(
        f"  sageattn {plain_peak / 2**20:.0f} MiB -> "
        f"sageattn_consume {consume_peak / 2**20:.0f} MiB "
        f"({saved_mib:+.0f} MiB)"
    )
    # Measured saving is ~1430 MiB at this shape; assert well under that so
    # allocator noise cannot flip the result, while still failing loudly if
    # the ownership chain breaks and the saving collapses to zero.
    assert saved_mib > 800, (
        f"sageattn_consume saved only {saved_mib:.0f} MiB at S={S_PEAK}; "
        f"expected >800. The ownership chain is broken somewhere -- some "
        f"frame is still holding q/k/v across the call."
    )


LIGHT_CASES = [
    test_matches_sageattn_output,
    test_empties_the_list,
    test_rejects_malformed_input,
]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.", file=sys.stderr)
        return 0
    assert hasattr(sageattention, "sageattn_consume"), (
        "sageattn_consume must be exported from the package root: consumers "
        "import it directly"
    )

    cases = list(LIGHT_CASES)
    free, _ = torch.cuda.mem_get_info()
    if free >= PEAK_VRAM_BYTES:
        cases.append(test_peak_vram_is_lower)
    else:
        print(
            f"Skipping the peak-VRAM case: needs "
            f"{PEAK_VRAM_BYTES / 2**30:.0f} GiB free, have {free / 2**30:.1f} GiB.\n"
        )

    failures = 0
    for fn in cases:
        print(f"{fn.__name__}:")
        try:
            fn()
            print("  ok")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures}/{len(cases)} cases failed.")
        return 1
    print(f"\nAll {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
