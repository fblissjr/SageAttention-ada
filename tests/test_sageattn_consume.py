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
  - prefers_cloned_v_*      : the predicate consumers gate their clone
                              on. Delete and it can drift away from the
                              arch set it is supposed to describe, which
                              costs a caller 572 MiB rather than erroring.
  - fused_caller_cloned_v_* : the caller-side route out of the fused
                              case. Delete and our release timing can
                              regress without any arm noticing, turning
                              a consumer's saving into a pure cost.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
Needs CUDA. The peak-VRAM case needs ~6 GiB free and skips without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sageattention
from sageattention import core, sageattn, sageattn_consume

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


def test_prefers_cloned_v_answers_for_a_device():
    """Claim: the predicate is usable from a forward pass, per device.

    Consumers call this to decide whether to clone v before handing the
    list over. They cannot answer it themselves: the arch set that decides
    it is private to our dispatch, and a caller-side copy of it drifts into
    a silent 572 MiB regression rather than an error.

    A downstream node calls it inside the forward keyed on `x.device`, not
    once at import -- ComfyUI patches a model before it is loaded or cast,
    so at patch time the device is whatever happens to be current rather
    than where the DiT runs. That makes three things contractual: it takes
    a device, it is cheap enough to call per forward, and repeated calls
    agree so the answer can be cached per device index.
    """
    from sageattention import sageattn_consume_prefers_cloned_v as prefers

    idx = torch.cuda.current_device()
    answers = [
        prefers(),
        prefers(idx),
        prefers(f"cuda:{idx}"),
        prefers(torch.device("cuda", idx)),
        prefers(torch.device("cuda")),
    ]
    assert len(set(answers)) == 1, (
        f"the same device spelled five ways gave {answers}; a caller "
        f"caching by device index would cache a coin flip"
    )
    # Called per forward and cached per device, so it must not do work and
    # must not disagree with itself between two calls on one device.
    assert prefers(idx) is prefers(idx)

    # A device we cannot answer for is an error, not a guess. False would
    # read as "don't clone" -- a plausible-looking answer to a question the
    # caller asked about the wrong device, which is the failure the whole
    # predicate exists to prevent.
    for bad in ("cpu", torch.device("cpu")):
        try:
            prefers(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for device {bad!r}")
    print(f"  prefers_cloned_v: {answers[0]} on {core._cuda_archs[idx]}")


def test_prefers_cloned_v_tracks_the_dispatch_arch_set():
    """Claim: the predicate reads the arch set, it does not restate it.

    The point of shipping this at all is that consumers stop copying
    `{"sm89", "sm100", "sm120", "sm121"}` into their own nodes. A second
    copy of that set *in here* has the same defect one layer up: whoever
    adds an arch to the dispatch gets a predicate that still answers for
    the old set, and the caller silently pays for a clone that frees
    nothing.

    Mutating the constant is the control. If the predicate were a
    hardcoded list this case would stay green with the set emptied.
    """
    original = core._EARLY_RELEASE_ARCHS
    idx = torch.cuda.current_device()
    assert core._cuda_archs[idx] in original, (
        f"this box is {core._cuda_archs[idx]}, which is not in the "
        f"early-release set -- the rest of this case assumes it is"
    )
    try:
        core._EARLY_RELEASE_ARCHS = frozenset()
        assert core.sageattn_consume_prefers_cloned_v(idx) is False, (
            "the predicate answered True with the early-release arch set "
            "emptied, so it is not reading that set"
        )
    finally:
        core._EARLY_RELEASE_ARCHS = original
    assert core.sageattn_consume_prefers_cloned_v(idx) is True
    print("  predicate follows _EARLY_RELEASE_ARCHS")


def _qkv_fused(s, dtype=torch.bfloat16):
    """Three NHD views into one packed QKV buffer -- how a DiT block makes them.

    `qkv_proj(x).split(heads*head_dim, dim=-1)` leaves q/k/v as views over a
    single allocation, so `stride_seq` is 3*H*D (21504 here) rather than H*D.
    That is what makes releasing q and k free nothing: v still references the
    same block. The buffer handle is dropped before returning so the views are
    its only owners, as in production.
    """
    buf = torch.randn(1, s, 3 * H * D, device="cuda", dtype=dtype)
    views = [
        buf[:, :, i * H * D : (i + 1) * H * D].view(1, s, H, D) for i in range(3)
    ]
    del buf
    return views


def _peak_over_call(use_consume, smooth_k=False, fused=False, clone_v=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    layout = "NHD" if fused else "HND"
    q, k, v = (_qkv_fused if fused else _qkv)(S_PEAK)
    if clone_v:
        # The move a caller can make without touching this library: give v its
        # own storage, so releasing q and k in here actually frees the fused
        # buffer instead of leaving v pinning all three thirds of it.
        v = v.clone()
    if use_consume:
        box = [q, k, v]
        del q, k, v
        o = sageattn_consume(box, tensor_layout=layout, smooth_k=smooth_k)
    else:
        o = sageattn(q, k, v, tensor_layout=layout, smooth_k=smooth_k)
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


def test_fused_caller_cloned_v_recovers_the_saving():
    """Claim: the fused case is not a dead end, and the fix is the caller's.

    `test_peak_across_shipped_config` shows fused q/k/v saving nothing,
    because v pins the buffer q and k were released from. The docstring on
    `sageattn_consume` frames the fix as work in here -- drop the transpose
    buffer, subtract the mean in place. There is a third route that needs no
    kernel change at all: the caller clones v first, which turns the fused
    case into the separate case for the price of one third of the buffer.

    Consumers rely on this now. ComfyUI does it in `comfy/ldm/minimax/model.py`
    ("Fix peak memory issue with H3"), and ComfyUI-h3-explorations does it in
    its H3 attention forward, where it is worth 286 MiB per call at this exact
    shape. If our ownership chain stops releasing early, that saving silently
    becomes a pure cost -- the clone is still paid for, and nothing is freed.

    Delete this case and that regression is invisible from in here: every
    other arm still passes, because none of them clone.
    """
    # consume first: a prior arm trains the caching allocator and biases
    # whatever runs second (CLAUDE.md, peak-HBM measurement discipline).
    cloned = _peak_over_call(True, fused=True, clone_v=True)
    plain = _peak_over_call(True, fused=True, clone_v=False)
    saved_mib = (plain - cloned) / 2**20
    print(
        f"  fused qkv, consume: {plain / 2**20:.0f} MiB -> "
        f"caller clones v: {cloned / 2**20:.0f} MiB ({saved_mib:+.0f} MiB)"
    )
    # Measured ~286 MiB at this shape. The floor is set well under that: the
    # clone costs a third of the buffer up front, so a broken release chain
    # does not merely zero this out, it drives it negative. Anything still
    # comfortably positive means q and k are being freed early.
    assert saved_mib > 150, (
        f"caller-cloned v saved only {saved_mib:.0f} MiB at S={S_PEAK}; "
        f"expected >150. Two different causes, and the fix differs:\n"
        f"  (a) q and k are no longer released before the kernel allocates, "
        f"or something in here is holding the fused buffer. A regression; "
        f"consumers are now paying for a clone that frees nothing.\n"
        f"  (b) `per_channel_fp8` dropped its transpose buffer and the "
        f"mean-subtraction moved in place (CHANGELOG Backlog). Then the "
        f"no-clone peak reaches 2573 while a cloning caller's floor is 2859 "
        f"-- fused 1715 + clone 572 + the int8 pair 572, all live before q "
        f"and k go -- so cloning became a cost and this case is correctly "
        f"red. Not a regression: flip "
        f"`sageattn_consume_prefers_cloned_v` to False, retire this case, "
        f"and say so in the CHANGELOG so consumers stop cloning."
    )


def test_peak_across_shipped_config():
    """Claim: the headline saving is configuration-dependent, and the case
    above is not the configuration consumers run.

    `test_peak_vram_is_lower` passes `smooth_k=False` with three separately
    allocated tensors. Production is the opposite on both axes: `smooth_k`
    defaults to True (core.py:948) and a DiT block hands over three views of
    one fused QKV buffer. Both axes move the peak, for different reasons:

      smooth_k=True  -- `per_thread_int8` allocates q_int8/k_int8 first, then
        does `k = k - km` (quant_per_thread.py:176-180), so a full bf16 copy
        of K is live on top of them.
      fused          -- releasing q and k frees nothing, because v still
        references the same allocation.

    This case does not assert a threshold. Its job is to publish the real
    numbers so a doc quoting one of them says which arm it came from; a
    threshold here would just re-freeze the same mistake one config over.
    """
    print(f"  {'arm':<40}{'sageattn':>10}{'consume':>10}{'saved':>10}")
    results = {}
    for fused in (False, True):
        for smooth_k in (False, True):
            # consume first: a prior arm trains the caching allocator and
            # biases whatever runs second (CLAUDE.md, peak-HBM discipline).
            c = _peak_over_call(True, smooth_k=smooth_k, fused=fused)
            p = _peak_over_call(False, smooth_k=smooth_k, fused=fused)
            label = f"{'fused qkv' if fused else 'separate'}, smooth_k={smooth_k}"
            results[(fused, smooth_k)] = (p, c)
            print(
                f"  {label:<40}{p / 2**20:>9.0f}M{c / 2**20:>9.0f}M"
                f"{(p - c) / 2**20:>+9.0f}M"
            )
    for smooth_k in (False, True):
        c = _peak_over_call(True, smooth_k=smooth_k, fused=True, clone_v=True)
        p = _peak_over_call(False, smooth_k=smooth_k, fused=True, clone_v=True)
        label = f"fused + caller clones v, smooth_k={smooth_k}"
        print(
            f"  {label:<40}{p / 2**20:>9.0f}M{c / 2**20:>9.0f}M"
            f"{(p - c) / 2**20:>+9.0f}M"
        )
    shipped = results[(True, True)]
    print(
        f"  shipped config (fused + smooth_k=True) saves "
        f"{(shipped[0] - shipped[1]) / 2**20:+.0f} MiB"
    )


LIGHT_CASES = [
    test_matches_sageattn_output,
    test_empties_the_list,
    test_rejects_malformed_input,
    test_prefers_cloned_v_answers_for_a_device,
    test_prefers_cloned_v_tracks_the_dispatch_arch_set,
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
        cases.append(test_fused_caller_cloned_v_recovers_the_saving)
        cases.append(test_peak_across_shipped_config)
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
