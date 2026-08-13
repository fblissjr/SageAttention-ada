#!/usr/bin/env python3
# Last updated: 2026-04-25
"""Test the get_last_dispatched_kernel() telemetry helper.

This is the consumer-facing observability surface that lets a
downstream tracer record which kernel sageattn() actually dispatched
to, instead of mirroring sage's routing table or treating the kernel
as opaque.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python so it
uses the editable install of sageattention.

Expected to be run on RTX 4090 / sm89 / CUDA >= 12.8. On other archs
some assertions about specific kernel names will need to change to
match what sage's dispatcher picks.
"""

from __future__ import annotations

import sys
import threading

import torch

import sageattention
from sageattention import (
    get_last_dispatched_kernel,
    sageattn,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp8_cuda,
)
from sageattention.core import (
    KERNEL_FP8_CUDA_FP32,
    KERNEL_FP8_CUDA_PP,
    KERNEL_FP16_CUDA,
    KERNEL_FP16_TRITON,
    _record_dispatch,
    _reset_dispatch_for_test,
)


def _make_qkv(B=1, H=4, S=128, D=64, dtype=torch.bfloat16, device="cuda"):
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return q, k, v


def test_initial_value_is_none():
    _reset_dispatch_for_test()
    assert get_last_dispatched_kernel() is None, (
        f"expected None before any dispatch, got {get_last_dispatched_kernel()!r}"
    )
    print("ok  initial value is None")


def test_helper_is_exported_from_package():
    assert hasattr(sageattention, "get_last_dispatched_kernel"), (
        "get_last_dispatched_kernel must be importable from the top-level package"
    )
    print("ok  helper exported from sageattention package")


def test_dispatch_counts_support_a_coverage_claim():
    # Claim: `get_last_dispatched_kernel` answers "is sage reachable on this
    # path" -- one call, one value. It cannot answer "did EVERY attention
    # call in this render reach sage", which is a coverage question, and a
    # consumer whose fallback logs once cannot answer it either: one warning
    # is emitted whether 1 call or 10,000 fell back.
    #
    # Counts are monotonic per thread and deliberately have no reset. A
    # consumer reads before and after a render and subtracts, so there is no
    # clear-the-state contract to honour and no window where a concurrent
    # call is lost to someone else's reset. Same reasoning that kept the
    # reset out of the last-dispatch helper.
    counts = sageattention.get_dispatch_counts()
    assert isinstance(counts, dict), f"expected a dict, got {type(counts)!r}"

    before = counts.get(KERNEL_FP8_CUDA_PP, 0)
    _record_dispatch(KERNEL_FP8_CUDA_PP)
    _record_dispatch(KERNEL_FP8_CUDA_PP)
    _record_dispatch(KERNEL_FP16_TRITON)
    after = sageattention.get_dispatch_counts()

    assert after.get(KERNEL_FP8_CUDA_PP, 0) - before == 2, (
        f"expected +2 on {KERNEL_FP8_CUDA_PP}, got "
        f"{after.get(KERNEL_FP8_CUDA_PP, 0) - before}"
    )
    assert after.get(KERNEL_FP16_TRITON, 0) >= 1, (
        "a second kernel name must be counted separately, so a coverage "
        "check can tell 'all fp8_cuda++' from 'mostly fp8_cuda++'"
    )

    # A snapshot, not a live view: a consumer holding `before` while the
    # render runs must not have it mutate underneath them, or the diff is
    # always zero.
    after[KERNEL_FP8_CUDA_PP] = 999_999
    assert sageattention.get_dispatch_counts()[KERNEL_FP8_CUDA_PP] != 999_999, (
        "get_dispatch_counts must return a copy; a live view makes the "
        "before/after diff silently zero"
    )
    print("ok  dispatch counts support a coverage claim")


def test_sageattn_dispatcher_records_fp8_pp_on_sm89():
    # On sm89 + CUDA >= 12.8 (the box this fork targets), sageattn()
    # routes unmasked calls to sageattn_qk_int8_pv_fp8_cuda with
    # pv_accum_dtype="fp32+fp16" -- aka fp8_cuda++ (SageAttention2++).
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn(q, k, v, is_causal=False)
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP8_CUDA_PP, (
        f"sageattn() unmasked on sm89/cuda12.8+ should record "
        f"{KERNEL_FP8_CUDA_PP!r}, got {got!r}"
    )
    print(f"ok  sageattn() dispatcher recorded {got!r}")


def test_sageattn_dispatcher_routes_masked_calls_correctly():
    # v0.5.5 changed the masked-call routing invariant. Pre-v0.5.5 all
    # masked calls went to Triton (the only mask-correct path). v0.5.5
    # added native CUDA mask support to the sm89 fp8++ kernel, so masked
    # calls on sm89 + CUDA >= 12.8 now route to fp8_cuda++. Other archs
    # still route to Triton (their CUDA kernels haven't gained mask
    # support yet).
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    mask = torch.ones(q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                       device=q.device, dtype=torch.bool)
    mask[..., -16:] = False  # the typical text-padding-tail shape
    _ = sageattn(q, k, v, attn_mask=mask, is_causal=False)
    got = get_last_dispatched_kernel()
    # We're on sm89 + CUDA >= 12.8 in this fork's target environment.
    expected = KERNEL_FP8_CUDA_PP
    assert got == expected, (
        f"sageattn() with attn_mask on sm89+CUDA>=12.8 must route to "
        f"{expected!r} (v0.5.5 native CUDA mask path), got {got!r}"
    )
    print(f"ok  sageattn() masked call routed to {got!r}")


def test_attn_mask_is_an_introspectable_parameter():
    # ComfyUI decides whether sage may see masks at all by introspecting
    # our signature, not by trying a call:
    #   comfy/ldm/modules/attention.py:27
    #   SAGE_ATTENTION_SUPPORTS_MASK = "attn_mask" in
    #       inspect.signature(sageattn).parameters
    # and attention_sage falls back to attention_pytorch for every masked
    # call when that reads False. With attn_mask arriving via **kwargs the
    # probe misses it, so the v0.5.5 native CUDA mask path above is
    # unreachable through the stock dispatcher no matter how correct the
    # kernel is. Delete this and that regression is invisible: every test
    # here calls sageattn() directly and keeps passing.
    import inspect

    params = inspect.signature(sageattn).parameters
    assert "attn_mask" in params, (
        "attn_mask must be a named parameter of sageattn(), not routed "
        "through **kwargs -- ComfyUI's SAGE_ATTENTION_SUPPORTS_MASK probe "
        f"inspects the signature. Got: {list(params)}"
    )
    assert params["attn_mask"].default is None, (
        "attn_mask must default to None so unmasked callers are unaffected"
    )
    print("ok  attn_mask is introspectable on sageattn()")


def test_attn_mask_still_accepted_positionally_free():
    # The probe fix must not reorder the existing positional parameters.
    # Consumers call sageattn(q, k, v, "HND", False) positionally; if
    # attn_mask were inserted among those, tensor_layout would silently
    # bind to a mask.
    import inspect

    ordered = [
        name for name, p in inspect.signature(sageattn).parameters.items()
        if p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    assert ordered[:7] == [
        "q", "k", "v", "tensor_layout", "is_causal", "sm_scale", "return_lse"
    ], f"positional prefix changed, breaking existing callers: {ordered}"
    print("ok  positional parameter order preserved")


def test_direct_triton_call_records_fp16_triton():
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=False)
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP16_TRITON, (
        f"sageattn_qk_int8_pv_fp16_triton should record "
        f"{KERNEL_FP16_TRITON!r}, got {got!r}"
    )
    print(f"ok  fp16_triton kernel recorded {got!r}")


def test_direct_fp16_cuda_call_records_fp16_cuda():
    _reset_dispatch_for_test()
    q, k, v = _make_qkv(D=128)  # fp16 cuda kernel needs head_dim 64 or 128
    _ = sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=False)  # default pv_accum="fp32"
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP16_CUDA, (
        f"sageattn_qk_int8_pv_fp16_cuda(pv_accum='fp32') should record "
        f"{KERNEL_FP16_CUDA!r}, got {got!r}"
    )
    print(f"ok  fp16_cuda kernel recorded {got!r}")


def test_fp8_cuda_variant_records_correct_subname():
    # The fp8 cuda kernel has multiple pv_accum variants. Each one
    # should record a distinct stable short name.
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=False, pv_accum_dtype="fp32+fp32")
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP8_CUDA_FP32, (
        f"sageattn_qk_int8_pv_fp8_cuda(pv_accum='fp32+fp32') should record "
        f"{KERNEL_FP8_CUDA_FP32!r}, got {got!r}"
    )
    print(f"ok  fp8_cuda(fp32+fp32) variant recorded {got!r}")


def test_sageattn_dispatcher_honors_pv_accum_dtype_override():
    # Regression test: v0.3.1 added **kwargs forwarding from the
    # dispatcher to per-kernel calls. Without care, a consumer passing
    # `pv_accum_dtype="fp32+fp32"` would TypeError ("got multiple
    # values for keyword argument") because the dispatcher's explicit
    # pv_accum_dtype= collides with the same key in **kwargs.
    # The dispatcher uses kwargs.setdefault so consumer overrides win
    # cleanly; this test pins that behavior.
    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    _ = sageattn(q, k, v, is_causal=False, pv_accum_dtype="fp32+fp32")
    got = get_last_dispatched_kernel()
    assert got == KERNEL_FP8_CUDA_FP32, (
        f"sageattn(pv_accum_dtype='fp32+fp32') on sm89 should override "
        f"the dispatcher's default and record "
        f"{KERNEL_FP8_CUDA_FP32!r}, got {got!r}"
    )
    print(f"ok  sageattn() honors pv_accum_dtype override -> {got!r}")


def test_hand_picked_cuda_kernel_warns_when_mask_passed():
    # The dispatcher routes masked calls to triton automatically. A
    # consumer that bypasses the dispatcher and hand-picks a _cuda
    # kernel directly (e.g. for benchmarking, or because they're
    # mirroring a known shape decision) should get a loud warning if
    # they ALSO pass attn_mask -- the kernel silently drops it and
    # the output is numerically wrong. Soft warn (warnings.warn) so
    # consumers that defensively pass attn_mask=None aren't penalized.
    import warnings as _w

    _reset_dispatch_for_test()
    q, k, v = _make_qkv()
    mask = torch.ones(q.shape[0], q.shape[1], q.shape[2], k.shape[2],
                       device=q.device, dtype=torch.bool)
    mask[..., -16:] = False

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _ = sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, is_causal=False, attn_mask=mask, pv_accum_dtype="fp32+fp32",
        )
    masked_warns = [w for w in caught if "attn_mask" in str(w.message)]
    assert len(masked_warns) >= 1, (
        f"hand-picked _cuda kernel + non-None attn_mask must warn; "
        f"caught warnings: {[str(w.message) for w in caught]}"
    )
    print(f"ok  hand-picked _cuda kernel warns on non-None attn_mask "
          f"({len(masked_warns)} warning emitted)")

    # And conversely: passing attn_mask=None must NOT warn (the
    # defensive-None-pass case the soft-warn is designed to spare).
    with _w.catch_warnings(record=True) as caught_none:
        _w.simplefilter("always")
        _ = sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, is_causal=False, attn_mask=None, pv_accum_dtype="fp32+fp32",
        )
    masked_warns_none = [w for w in caught_none if "attn_mask" in str(w.message)]
    assert len(masked_warns_none) == 0, (
        f"hand-picked _cuda kernel + attn_mask=None must NOT warn; "
        f"caught: {[str(w.message) for w in caught_none]}"
    )
    print(f"ok  hand-picked _cuda kernel + attn_mask=None does not warn")


def test_thread_isolation():
    # threading.local() means each thread sees only its own dispatch
    # value. A worker thread reading the helper before any dispatch
    # should see None even if the main thread has already dispatched.
    # Uses real kernel-name constants so the test stays inside the
    # KernelName Literal (no type-checker false flags).
    #
    # This is load-bearing beyond isolation. A consumer proving "the
    # composed attention path reached sage" needs a known-None baseline,
    # and gets it by running its probe on a fresh thread rather than by
    # importing a reset. Collapsing this state to a module global would
    # keep every other test in this file green while silently converting
    # that probe into a false negative on graphs that route consistently.
    # So this case is the guard on a downstream composition check, not
    # only a statement about threading -- which is why it is worth more
    # than the public reset function we considered adding and did not.
    # (contextvars, contemplated in core.py's comment, is safe here: a new
    # thread starts with an empty Context, so the baseline still holds.)
    _reset_dispatch_for_test()
    _record_dispatch(KERNEL_FP16_TRITON)

    worker_observations: list[str | None] = []

    def worker():
        # Worker starts fresh: its thread-local has no `last` attr yet
        worker_observations.append(get_last_dispatched_kernel())  # expect None
        _record_dispatch(KERNEL_FP8_CUDA_PP)
        worker_observations.append(get_last_dispatched_kernel())  # expect KERNEL_FP8_CUDA_PP

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert worker_observations[0] is None, (
        f"worker thread should not see main thread's dispatch; "
        f"got {worker_observations[0]!r}"
    )
    assert worker_observations[1] == KERNEL_FP8_CUDA_PP, (
        f"worker thread should see its own dispatch; "
        f"got {worker_observations[1]!r}"
    )
    assert get_last_dispatched_kernel() == KERNEL_FP16_TRITON, (
        f"main thread's value should be untouched by worker; "
        f"got {get_last_dispatched_kernel()!r}"
    )
    print("ok  threads see isolated dispatch values")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping kernel-dispatch tests.", file=sys.stderr)
        return 0

    test_helper_is_exported_from_package()
    test_initial_value_is_none()
    test_sageattn_dispatcher_records_fp8_pp_on_sm89()
    test_sageattn_dispatcher_routes_masked_calls_correctly()
    test_attn_mask_is_an_introspectable_parameter()
    test_attn_mask_still_accepted_positionally_free()
    test_direct_triton_call_records_fp16_triton()
    test_direct_fp16_cuda_call_records_fp16_cuda()
    test_fp8_cuda_variant_records_correct_subname()
    test_sageattn_dispatcher_honors_pv_accum_dtype_override()
    test_hand_picked_cuda_kernel_warns_when_mask_passed()
    test_thread_isolation()
    print()
    print("all dispatched-kernel telemetry tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
