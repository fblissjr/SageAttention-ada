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
  - inert_under_*_chunking  : that a caller retaining q/k/v gets nothing,
                              which is advice we give in the predicate's
                              docstring. Delete and that paragraph is a
                              mechanism claim with no arm behind it.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
Needs CUDA. The peak-VRAM case needs ~6 GiB free and skips without it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import orjson
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sageattention
from sageattention import core, sageattn, sageattn_consume
from sageattention.core import KERNEL_FP16_TRITON

# MiniMax H3's attention config, at a shape small enough to run anywhere
# while still making the float tensors dominate the peak.
H, D = 56, 128
S_SMALL = 4096
S_PEAK = 41822  # fl2va, 16:9 canvas -- packed length, not a settable default
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


def test_mask_survives_the_early_release():
    """Claim: consuming q/k/v does not break a masked call.

    `sageattn_consume` declares `attn_mask` and forwards it, and on sm89 it
    routes to the fp8++ kernel, the one CUDA variant with native mask
    support (v0.5.5). That kernel prepares the mask from q and k -- dtype,
    device, and the broadcast target shape -- at a point the consuming path
    has already released them. A bool mask short-circuits past the dtype
    assert and dies on the device one instead.

    Delete this and a masked consume call raises `UnboundLocalError` from
    inside the kernel wrapper, which is exactly what it did from v0.7.0
    until this case was written: no other test in the suite passes a mask
    through `sageattn_consume`, so the whole path was uncovered while
    looking covered.
    """
    s = 512
    causal = torch.ones(1, 1, s, s, device="cuda", dtype=torch.bool).tril()
    additive = torch.where(
        causal,
        torch.zeros((), device="cuda", dtype=torch.bfloat16),
        torch.full((), float("-inf"), device="cuda", dtype=torch.bfloat16),
    )
    for label, mask in (("bool", causal), ("additive", additive)):
        torch.manual_seed(0)
        q, k, v = _qkv(s)
        expect = sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, smooth_k=False)
        unmasked = sageattn(q, k, v, tensor_layout="HND", smooth_k=False)
        box = [q, k, v]
        del q, k, v
        got = sageattn_consume(box, tensor_layout="HND", attn_mask=mask, smooth_k=False)
        # Agreement between the two arms is not enough on its own. Both route
        # to the same kernel, and the variants that do not support masks drop
        # them behind a `warnings.warn` rather than failing, so a silently
        # dropped mask would make both arms agree on the wrong answer and
        # leave this case green. The control is that a masked call must not
        # equal an unmasked one.
        assert not torch.equal(
            expect.view(torch.uint16), unmasked.view(torch.uint16)
        ), (
            f"masked and unmasked sageattn produced identical output "
            f"({label}); the mask is not reaching the kernel, so the "
            f"comparison below proves nothing"
        )
        assert torch.equal(got.view(torch.uint16), expect.view(torch.uint16)), (
            f"masked consume ({label}) diverged from masked sageattn; the "
            f"early release must change when tensors are freed, not the math"
        )
    print("  masked consume matches masked sageattn, and the mask reaches the kernel")


def test_both_entry_points_read_one_mask_gate():
    """Claim: masked routing is one decision, not two that happen to agree.

    `sageattn` sends masked calls to Triton unless the arch has a
    mask-correct CUDA kernel. Until v0.7.6 `sageattn_consume` applied no
    such gate at all, so a masked call took a mask-dropping kernel on
    sm89 with CUDA < 12.8 and the native-mask path on archs the dispatcher
    avoids. That divergence existed because the two entry points had
    separate routing code nobody had diffed.

    Copying the condition into the second caller would have closed the
    instance and left the class open. Both now call
    `_has_native_mask_kernel`, and this forces it False to prove they read
    it: with no native mask kernel anywhere, both must route a masked call
    to Triton.

    Shown red against the copied-condition version before being trusted.
    With consume holding its own inline copy, patching the shared function
    moves only `sageattn`, and this reports
    `sageattn='fp16_triton', consume='fp8_cuda++'`.
    """
    original = core._has_native_mask_kernel
    q, k, v = _qkv(256)
    mask = torch.ones(1, 1, 256, 256, device="cuda", dtype=torch.bool).tril()
    try:
        core._has_native_mask_kernel = lambda arch: False
        core.get_last_dispatched_kernel()  # clear any prior value
        sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, smooth_k=False)
        via_sageattn = core.get_last_dispatched_kernel()
        box = [q, k, v]
        del q, k, v
        sageattn_consume(box, tensor_layout="HND", attn_mask=mask, smooth_k=False)
        via_consume = core.get_last_dispatched_kernel()
    finally:
        core._has_native_mask_kernel = original
    assert via_sageattn == via_consume == KERNEL_FP16_TRITON, (
        f"with no native mask kernel, both entry points must route a masked "
        f"call to Triton; got sageattn={via_sageattn!r}, "
        f"consume={via_consume!r}. One of them is not reading the gate."
    )
    print(f"  both entry points routed to {via_sageattn} under a forced-False gate")


class _StandInContainer:
    """Shaped like ComfyUI's `AttentionTensorContainer`, without importing it.

    `sageattn_consume` duck-types on `take()` rather than importing comfy,
    because this library does not depend on ComfyUI. This stand-in proves the
    duck-typing accepts anything with that shape;
    `test_accepts_comfyui_containers` runs the real class when ComfyUI is
    importable, which is the half that catches the real contract drifting
    away from this copy.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("attention tensor container has already been consumed")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        return tensor


def test_accepts_containers_and_empties_them():
    """Claim: the ownership handoff works in ComfyUI's shape, not just ours.

    ComfyUI hands attention backends `AttentionTensorContainer` objects with
    `peek()`/`take()` (`comfy/ldm/modules/attention.py`), and H3's model wraps
    q/k/v in them on every call. Our own convention is a `[q, k, v]` list we
    empty. A caller on the container protocol should not have to unwrap into
    a list first, because unwrapping binds all three into their frame, which
    is exactly the retention this function exists to avoid.

    Delete this and a container-protocol caller either writes an adapter that
    reintroduces the retention, or passes containers and gets a confusing
    failure deep in the quant path.
    """
    torch.manual_seed(0)
    q, k, v = _qkv(S_SMALL)
    expect = sageattn(q, k, v, tensor_layout="HND", smooth_k=False)

    # Held separately from the list we hand over, because the call empties
    # both and each is a distinct guarantee: `take()` transfers the tensor,
    # clearing the slots drops the caller's handle on the container itself.
    containers = [_StandInContainer(t) for t in (q, k, v)]
    boxed = list(containers)
    del q, k, v
    got = sageattn_consume(boxed, tensor_layout="HND", smooth_k=False)

    assert torch.equal(got.view(torch.uint16), expect.view(torch.uint16)), (
        "the container path must produce the same output as passing tensors"
    )
    assert all(c.tensor is None for c in containers), (
        "every container must be emptied; one still holding its tensor means "
        "the caller's reference survived the call and nothing was released"
    )
    assert all(slot is None for slot in boxed), (
        "the list must be emptied too, same contract as the tensor path"
    )
    print("  containers consumed, list emptied, output matches the tensor path")


def test_rejects_spent_and_mixed_containers():
    """Claim: a container that cannot give up its tensor fails loudly.

    `take()` on a spent container raises RuntimeError from ComfyUI's own
    class. Letting that escape would surface as an unrelated-looking error
    from inside a quant kernel; a caller that half-unwrapped its arguments
    gets no signal at all. Both are shaped like the malformed-input case
    above and are raised the same way.
    """
    spent = _StandInContainer(torch.empty(1, H, 64, D, device="cuda", dtype=torch.bfloat16))
    spent.take()
    tensors = _qkv(64)
    bad_inputs = [
        [spent, _StandInContainer(tensors[1]), _StandInContainer(tensors[2])],
        [_StandInContainer(tensors[0]), tensors[1], tensors[2]],
    ]
    for bad in bad_inputs:
        try:
            sageattn_consume(bad, tensor_layout="HND", smooth_k=False)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {[type(b).__name__ for b in bad]}")
    print("  spent and mixed containers rejected")


def test_accepts_comfyui_containers():
    """Claim: the duck-typing matches the real class, not just our copy of it.

    Skips when ComfyUI cannot be located, which makes it worthless as the
    only coverage -- hence the stand-in cases above. Its job is to fail the
    day ComfyUI renames `take()` or changes what a spent container does.

    Resolution order is the repo's usual one: `$COMFYUI_ROOT`, then
    `comfyui_root` in `internal/local_config.json` (gitignored, so the path
    stays out of committed material), then skip.
    """
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        cfg = Path(__file__).resolve().parent.parent / "internal" / "local_config.json"
        if cfg.is_file():
            root = orjson.loads(cfg.read_bytes()).get("comfyui_root")
    if not root or not Path(root).is_dir():
        print("  skipped: set $COMFYUI_ROOT or comfyui_root in local_config.json")
        return
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer
    except Exception as exc:
        print(f"  skipped: ComfyUI not importable ({type(exc).__name__})")
        return
    torch.manual_seed(0)
    q, k, v = _qkv(S_SMALL)
    expect = sageattn(q, k, v, tensor_layout="HND", smooth_k=False)
    containers = [AttentionTensorContainer(t) for t in (q, k, v)]
    del q, k, v
    got = sageattn_consume(list(containers), tensor_layout="HND", smooth_k=False)
    assert torch.equal(got.view(torch.uint16), expect.view(torch.uint16))
    for c in containers:
        try:
            c.peek()
        except RuntimeError:
            continue
        raise AssertionError("a real AttentionTensorContainer was not emptied")
    print("  real AttentionTensorContainer consumed and emptied")


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


def _peak_over_chunked_call(use_consume, n_chunks):
    """Peak over a caller that slices q/k/v into head groups and loops.

    The low-VRAM pattern: attend `n_chunks` head groups in turn so the
    kernel's own int8/fp8 transients scale with the group rather than the
    full head count. The loop frame holds the un-sliced q/k/v throughout,
    which is what makes this the interesting case for `sageattn_consume`.

    Separate NHD allocations, deliberately. Fused views would suppress the
    saving on their own, so a flat result at `n_chunks > 1` could not be
    attributed to the chunking -- see the paired control in the case below.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    q, k, v = [
        torch.randn(1, S_PEAK, H, D, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    out = torch.empty((1, S_PEAK, H, D), dtype=q.dtype, device=q.device)
    if n_chunks is None:
        # The comparand: hand the whole thing over and retain nothing. This
        # is what an unchunked caller does, and it is the *only* difference
        # from the loop below -- same allocations, same output buffer.
        if use_consume:
            box = [q, k, v]
            del q, k, v
            out[:] = sageattn_consume(box, tensor_layout="NHD", smooth_k=False)
        else:
            out[:] = sageattn(q, k, v, tensor_layout="NHD", smooth_k=False)
            del q, k, v
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del out
        torch.cuda.empty_cache()
        return peak
    start = 0
    for i in range(n_chunks):
        end = start + H // n_chunks + (1 if i < H % n_chunks else 0)
        if use_consume:
            box = [q[:, :, start:end], k[:, :, start:end], v[:, :, start:end]]
            out[:, :, start:end] = sageattn_consume(
                box, tensor_layout="NHD", smooth_k=False
            )
        else:
            out[:, :, start:end] = sageattn(
                q[:, :, start:end], k[:, :, start:end], v[:, :, start:end],
                tensor_layout="NHD", smooth_k=False,
            )
        start = end
    del q, k, v
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    torch.cuda.empty_cache()
    return peak


def test_consume_is_inert_under_caller_side_chunking():
    """Claim: a True from the predicate does not survive a slicing caller.

    `sageattn_consume_prefers_cloned_v`'s docstring tells callers that if
    they slice q/k/v into groups and call in a loop, the per-call release
    frees nothing, because the frame running the loop holds the originals
    for its whole duration. That sentence is committed advice, and the
    clone half of it is measured -- a consumer found the loop shape costing
    a flat 572 MiB with nothing to recover it. This is the other half, which
    was reasoned rather than measured when it was written.

    Equality is the claim: under chunking, consume and plain `sageattn`
    reach the same peak. Red in either direction is informative. Lower means
    the release does reach something and the docstring understates
    consume; higher means consume costs something here and the advice
    should be stronger than "no benefit".

    Delete this and the docstring keeps asserting a mechanism nothing
    checks, which is how the fused-case number stayed wrong for three
    months.
    """
    n = 4
    # Consume arm first in each pair, per the peak-HBM ordering discipline.
    saved = {}
    for label, chunks in (("handover", None), (f"loop, n={n}", n), ("loop, n=1", 1)):
        c = _peak_over_chunked_call(True, chunks)
        p = _peak_over_chunked_call(False, chunks)
        saved[label] = (p - c) / 2**20
        print(
            f"  {label:<12} sageattn {p / 2**20:>5.0f} MiB, consume "
            f"{c / 2**20:>5.0f} MiB ({saved[label]:+.0f} MiB)"
        )
    # The control, asserted first because the claim below is meaningless
    # without it. Same allocations, same output buffer, differing only in
    # whether the caller retains q/k/v: consume must show its usual saving
    # here, or a flat result in the loop arms says nothing about looping and
    # only says this harness cannot see a saving at all.
    assert saved["handover"] > 400, (
        f"handover consume saved only {saved['handover']:.0f} MiB in this "
        f"harness; expected the usual ~858. Nothing below is trustworthy "
        f"until this pair reproduces the saving that looping suppresses."
    )
    # Retention is the mechanism, not the group count. A one-group loop
    # still passes a slice while the frame holds the parent, so it loses the
    # saving exactly like a four-group one. Without this row, a reader could
    # conclude that chunking is what costs them and that n=1 is safe.
    assert abs(saved["loop, n=1"]) < 64, (
        f"a one-group loop saved {saved['loop, n=1']:+.0f} MiB, so the "
        f"suppression tracks the group count rather than the retained "
        f"parents -- the stated mechanism is wrong"
    )
    chunked_saved = saved[f"loop, n={n}"]
    single_saved = saved["handover"]
    # Tolerance, not equality: both arms allocate identically, but the
    # allocator is free to round differently across arms. The suppressed
    # saving is ~858 MiB, so anything inside this band is unambiguous.
    assert abs(chunked_saved) < 64, (
        f"consume moved the peak by {chunked_saved:+.0f} MiB under {n}-way "
        f"head chunking, while saving {single_saved:.0f} MiB unchunked in "
        f"the same harness; expected it to be inert, because the loop frame "
        f"holds the un-sliced q/k/v so releasing a slice frees nothing. The "
        f"slice-and-loop paragraph in "
        f"`sageattn_consume_prefers_cloned_v`'s docstring is now wrong in "
        f"whichever direction this moved."
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
    test_mask_survives_the_early_release,
    test_both_entry_points_read_one_mask_gate,
    test_accepts_containers_and_empties_them,
    test_rejects_spent_and_mixed_containers,
    test_accepts_comfyui_containers,
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
        cases.append(test_consume_is_inert_under_caller_side_chunking)
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
