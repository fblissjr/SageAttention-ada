#!/usr/bin/env python3
"""Red TDD test for sageattn_partitioned.

Spec (from partitioned_sage_entry_plan.md):
    sageattn_partitioned(q, k, v, slices) must produce output
    equivalent to running `sageattn_qk_int8_pv_fp16_triton` on each
    slice independently and stitching the per-slice outputs into a
    full-Q-shaped tensor.

`slices` is `list[(q_start, q_end, attn_mask | None)]`. Mask shapes
mirror the Kijai PR 13735 pattern: `(1, 1, 1, kv_len)` (broadcast
across all Q rows in the slice) or `(1, 1, q_end - q_start, kv_len)`
(per-Q-row).

Tolerances reuse `accuracy_metrics` from test_sageattn_ltx_shapes so
the calibration matches every other rtol-vs-Triton row in the repo:
mean_rtol < 0.05, max_rtol < 0.5 (derived from v0.4.1 LTX self-attn
logs).

Edge cases:
    - 2-slice Kijai-PR pattern with unaligned boundary (load-bearing)
    - 1-slice noisy-only (tracked_count=0)
    - 1-slice tracked-only (guide_start=0)
    - 2-slice with aligned boundary (control)

Today this exits non-zero with "not importable" because
`sageattn_partitioned` doesn't exist yet. The implementation arc
flips it green.

Run with the venv that has sage installed active:
    ${VIRTUAL_ENV}/bin/python tests/test_partitioned.py
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from sageattention import sageattn_qk_int8_pv_fp16_triton

# Reuse the symmetric-denominator rtol helper used by every other
# accuracy bench in this repo. Keeps tolerance budgets comparable.
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]

try:
    from sageattention import sageattn_partitioned  # type: ignore[attr-defined]
    HAS_PARTITIONED = True
except ImportError:
    HAS_PARTITIONED = False


class SliceCase(NamedTuple):
    q_start: int
    q_end: int
    # None = no mask; 1 = broadcast row (PR noisy shape);
    # q_end - q_start = per-Q-row (PR tracked shape).
    mask_q_dim: int | None


def build_qkv(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device, generator=g)
    return q, k, v


def build_mask(case: SliceCase, kv_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if case.mask_q_dim is None:
        return None
    return torch.zeros((1, 1, case.mask_q_dim, kv_len), dtype=dtype, device=device)


def reference_perslice(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cases: list[SliceCase],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    full_out = torch.empty_like(q, dtype=output_dtype)
    kv_len = k.size(2)
    for case in cases:
        q_slice = q[:, :, case.q_start:case.q_end, :].contiguous()
        mask = build_mask(case, kv_len, q.dtype, q.device)
        out = sageattn_qk_int8_pv_fp16_triton(q_slice, k, v, attn_mask=mask)
        full_out[:, :, case.q_start:case.q_end, :] = out.to(output_dtype)
    return full_out


class TestResult(NamedTuple):
    name: str
    passed: bool
    detail: str


def run_case(
    name: str,
    cases: list[SliceCase],
    batch: int = 1,
    heads: int = 4,
    seq: int = 2048,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    mean_rtol_max: float = 0.05,
    max_rtol_max: float = 0.5,
) -> TestResult:
    if not HAS_PARTITIONED:
        return TestResult(name=name, passed=False, detail="sageattn_partitioned not importable")

    device = torch.device("cuda")
    q, k, v = build_qkv(batch, heads, seq, head_dim, dtype, device, seed)
    ref = reference_perslice(q, k, v, cases, output_dtype=dtype)

    kv_len = k.size(2)
    slices_arg = [
        (c.q_start, c.q_end, build_mask(c, kv_len, dtype, device)) for c in cases
    ]
    got = sageattn_partitioned(q, k, v, slices_arg)  # type: ignore[name-defined]

    if got.shape != ref.shape:
        return TestResult(
            name=name,
            passed=False,
            detail=f"shape mismatch: got {tuple(got.shape)}, ref {tuple(ref.shape)}",
        )

    mean_r, max_r, _, _ = accuracy_metrics(got, ref)
    passed = mean_r < mean_rtol_max and max_r < max_rtol_max
    detail = f"mean_rtol={mean_r:.4f} (<{mean_rtol_max}), max_rtol={max_r:.4f} (<{max_rtol_max})"
    return TestResult(name=name, passed=passed, detail=detail)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"sageattn_partitioned importable: {HAS_PARTITIONED}")
    print()

    # 1996 % 128 == 76: unaligned boundary, the load-bearing edge case.
    # The implementation must handle this without breaking the Q-scale's
    # BLOCK_M=128 block layout.
    assert 1996 % 128 == 76, "unaligned boundary precondition"

    cases_kijai_2call: list[SliceCase] = [
        SliceCase(q_start=0, q_end=1996, mask_q_dim=1),
        SliceCase(q_start=1996, q_end=2048, mask_q_dim=52),
    ]
    cases_only_noisy: list[SliceCase] = [
        SliceCase(q_start=0, q_end=2048, mask_q_dim=1),
    ]
    cases_only_tracked: list[SliceCase] = [
        SliceCase(q_start=0, q_end=2048, mask_q_dim=2048),
    ]
    cases_aligned: list[SliceCase] = [
        SliceCase(q_start=0, q_end=1920, mask_q_dim=1),
        SliceCase(q_start=1920, q_end=2048, mask_q_dim=128),
    ]

    results: list[TestResult] = [
        run_case("kijai_2call_unaligned_boundary", cases_kijai_2call),
        run_case("single_slice_noisy_only", cases_only_noisy),
        run_case("single_slice_tracked_only", cases_only_tracked),
        run_case("kijai_2call_aligned_boundary", cases_aligned),
    ]

    print("RESULTS")
    print("-" * 64)
    n_pass = sum(1 for r in results if r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    print()
    print(f"summary: {n_pass}/{len(results)} passed")
    if not HAS_PARTITIONED:
        print("red: sageattn_partitioned not yet implemented")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
