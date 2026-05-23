#!/usr/bin/env python3
"""Test the session-start dispatch log.

`sageattn()` emits one `[INFO] sage routing: ...` line to stderr per
unique `(arch, cuda_version, mask_present, pv_accum_dtype, kernel)`
tuple in the current process. Subsequent calls with the same tuple
are silent. This gives consumers a grep-able ground-truth record of
which kernel actually got dispatched for their config, without
forcing them to call `get_last_dispatched_kernel()` programmatically.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.

Expected to be run on RTX 4090 / sm89 / CUDA >= 12.8. On other archs
the specific kernel-name assertions will need to be adjusted.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sageattention import sageattn
from sageattention.core import (
    _reset_routing_log_for_test,
)
from test_dispatched_kernel_telemetry import _make_qkv  # type: ignore[import-not-found]


def _call_and_capture_stderr(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def test_first_call_emits_routing_line():
    _reset_routing_log_for_test()
    q, k, v = _make_qkv()
    stderr = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" in stderr, f"expected routing log in stderr; got: {stderr!r}"
    assert "arch=" in stderr
    assert "cuda=" in stderr
    assert "mask=" in stderr
    assert "pv_accum=" in stderr
    print(f"  first-call stderr: {stderr.strip()}")


def test_second_call_with_same_tuple_is_silent():
    _reset_routing_log_for_test()
    q, k, v = _make_qkv()
    first = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" in first, "test setup invariant: first call should emit"
    second = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" not in second, (
        f"expected no routing log on second call with same tuple; got: {second!r}"
    )
    print(f"  second-call stderr (should be empty): {second.strip()!r}")


def test_different_mask_emits_separately():
    _reset_routing_log_for_test()
    q, k, v = _make_qkv()
    # First call without mask
    unmasked_stderr = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" in unmasked_stderr
    assert "mask=False" in unmasked_stderr
    # Second call WITH mask — different tuple, should emit again
    mask = torch.zeros(1, 1, 128, 128, device="cuda", dtype=torch.bool)
    masked_stderr = _call_and_capture_stderr(sageattn, q, k, v, attn_mask=mask)
    assert "sage routing" in masked_stderr, (
        f"expected new routing log with mask=True tuple; got: {masked_stderr!r}"
    )
    assert "mask=True" in masked_stderr
    print(f"  unmasked-call stderr: {unmasked_stderr.strip()}")
    print(f"  masked-call stderr:   {masked_stderr.strip()}")


def test_reset_helper_clears_state():
    _reset_routing_log_for_test()
    q, k, v = _make_qkv()
    first = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" in first
    second = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" not in second
    _reset_routing_log_for_test()
    third = _call_and_capture_stderr(sageattn, q, k, v)
    assert "sage routing" in third, (
        f"expected routing log to emit again after reset; got: {third!r}"
    )
    print(f"  third-call (post-reset) stderr: {third.strip()}")


def test_kernel_name_in_log_matches_dispatched():
    """The kernel name in the routing log should match
    get_last_dispatched_kernel() so consumers can grep stderr OR call
    the telemetry API and get the same answer."""
    from sageattention import get_last_dispatched_kernel

    _reset_routing_log_for_test()
    q, k, v = _make_qkv()
    stderr = _call_and_capture_stderr(sageattn, q, k, v)
    kernel = get_last_dispatched_kernel()
    assert kernel is not None
    assert kernel in stderr, (
        f"expected kernel name {kernel!r} to appear in routing log; got: {stderr!r}"
    )
    print(f"  log contains kernel name {kernel!r}: ok")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0
    cases = [
        test_first_call_emits_routing_line,
        test_second_call_with_same_tuple_is_silent,
        test_different_mask_emits_separately,
        test_reset_helper_clears_state,
        test_kernel_name_in_log_matches_dispatched,
    ]
    failures = 0
    for fn in cases:
        try:
            print(f"{fn.__name__}:")
            fn()
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
