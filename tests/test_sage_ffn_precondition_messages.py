"""Precondition messages for sage_ffn asserts.

Downstream wrappers that catch AssertionError and log `str(exc)`
(e.g. log-once-per-shape diagnostics on top of `sage_ffn`) need a
non-empty message to be actionable. The message is the contract:
each assert must name the failing precondition AND the actual
offending value so the consumer can diagnose without instrumenting
its own wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sageattention.triton.fused_mlp_fp8 import sage_ffn


HIDDEN = 64
INNER = 256
T = 8


def _ok_inputs():
    """Construct a valid input tuple. Mutate one field per case below."""
    x = torch.randn(1, T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w1_f32 = torch.randn(INNER, HIDDEN, dtype=torch.float32, device="cuda") * 0.05
    w1 = w1_f32.to(torch.float8_e4m3fn)
    w2_f32 = torch.randn(HIDDEN, INNER, dtype=torch.float32, device="cuda") * 0.05
    w2 = w2_f32.to(torch.float8_e4m3fn)
    s1 = 1.0
    s2 = 1.0
    return x, w1, s1, w2, s2


def _expect_assertion_with(substrings: list[str], *args, **kwargs) -> str:
    """Call sage_ffn(*args, **kwargs); assert AssertionError raised and
    every string in `substrings` appears in str(exc). Return the message."""
    try:
        sage_ffn(*args, **kwargs)
    except AssertionError as exc:
        msg = str(exc)
        missing = [s for s in substrings if s not in msg]
        assert not missing, f"assert message missing {missing!r}; full message: {msg!r}"
        return msg
    raise AssertionError("sage_ffn did not raise on bad input")


def test_x_dtype_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    x_bad = x.to(torch.float16)
    msg = _expect_assertion_with(["x.dtype", "bfloat16", "float16"], x_bad, w1, s1, w2, s2)
    print(f"  x.dtype: {msg}")


def test_w1_dtype_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    w1_bad = w1.to(torch.bfloat16)
    msg = _expect_assertion_with(["w1.dtype", "float8_e4m3fn", "bfloat16"], x, w1_bad, s1, w2, s2)
    print(f"  w1.dtype: {msg}")


def test_w2_dtype_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    w2_bad = w2.to(torch.bfloat16)
    msg = _expect_assertion_with(["w2.dtype", "float8_e4m3fn", "bfloat16"], x, w1, s1, w2_bad, s2)
    print(f"  w2.dtype: {msg}")


def test_w1_shape_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    w1_bad = torch.randn(INNER, HIDDEN + 1, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    msg = _expect_assertion_with(["w1.shape", str(INNER), str(HIDDEN + 1), str(HIDDEN)], x, w1_bad, s1, w2, s2)
    print(f"  w1.shape: {msg}")


def test_w2_shape_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    w2_bad = torch.randn(HIDDEN, INNER + 1, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    msg = _expect_assertion_with(["w2.shape", str(HIDDEN), str(INNER + 1), str(INNER)], x, w1, s1, w2_bad, s2)
    print(f"  w2.shape: {msg}")


def test_b1_shape_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    b1_bad = torch.randn(INNER + 1, dtype=torch.bfloat16, device="cuda")
    msg = _expect_assertion_with(["b1", str(INNER + 1), str(INNER)], x, w1, s1, w2, s2, b1=b1_bad)
    print(f"  b1.shape: {msg}")


def test_device_message():
    x, w1, s1, w2, s2 = _ok_inputs()
    x_cpu = x.cpu()
    msg = _expect_assertion_with(["cuda", "cpu"], x_cpu, w1, s1, w2, s2)
    print(f"  device: {msg}")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0
    cases = [
        test_x_dtype_message,
        test_w1_dtype_message,
        test_w2_dtype_message,
        test_w1_shape_message,
        test_w2_shape_message,
        test_b1_shape_message,
        test_device_message,
    ]
    failures = 0
    for fn in cases:
        try:
            print(f"{fn.__name__}:")
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
    if failures:
        print(f"\n{failures}/{len(cases)} cases failed.")
        return 1
    print(f"\nAll {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
