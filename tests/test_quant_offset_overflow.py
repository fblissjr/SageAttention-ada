#!/usr/bin/env python3
"""Guard the INT8 quant kernels against int32 element-offset overflow.

The Triton quant kernels form `off_b * stride_z + off_h * stride_h +
offs_n * stride_n + offs_k` in whatever integer type the operands carry.
Triton's `program_id` is int32 and the stride arguments are passed as
int32, so the whole expression is computed in int32 and wraps once a
tensor exceeds 2**31 elements (4 GiB at bf16). Measured behaviour before
the fix, at H=56 D=128 (MiniMax H3's attention config):

  NHD  S=303,689  ->  int8 output tail is all zero (silent corruption)
  HND  S=310,000  ->  CUDA error: an illegal memory access was encountered

Neither mode warns, which is what makes this worth a permanent test: the
NHD failure produces plausible-looking tensors full of zeros. The two
layouts overflow through different terms -- NHD through `offs_n *
stride_n` (stride_n = heads*head_dim) and HND through `off_h * stride_h`
(stride_h = seq_len*head_dim) -- so a fix that promotes only the row term
still crashes in HND. Both cases are covered below.

Claims, i.e. what breaks if a case is deleted:
  - selects_int64_*      : the wrapper's int32/int64 specialization
                           decision, at and around the boundary. Delete
                           and an off-by-one in the bound goes unnoticed
                           until it corrupts a real render.
  - stays_int32_*        : that ordinary shapes do NOT pay for int64
                           address arithmetic. Delete and a future
                           "just always use int64" simplification looks
                           free when it is not.
  - tail_correct_*       : the actual end-to-end quantization above the
                           boundary, per layout. Delete and the bound
                           could be right while the kernel is still
                           wrong.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
The tail_correct cases need ~13 GiB of free VRAM and skip without it;
everything else runs on meta tensors and needs no GPU at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sageattention.triton._int_offsets import (
    max_element_offset as _max_element_offset,
    needs_int64_offsets as _needs_int64,
)
from sageattention.triton.quant_per_thread import per_thread_int8


def _needs_int64_offsets(t, tensor_layout, blk):
    return _needs_int64(t, tensor_layout=tensor_layout, blk=blk)

INT32_MAX = 2**31 - 1
H, D = 56, 128  # MiniMax H3 attention config
HEAVY_VRAM_BYTES = 13 * 2**30


# --- specialization-decision cases (meta tensors; no GPU needed) ---


def _meta(shape):
    return torch.empty(shape, dtype=torch.bfloat16, device="meta")


def test_selects_int64_when_nhd_row_term_overflows():
    # NHD: stride_n = H*D, so the row term alone crosses int32 first.
    s_over = INT32_MAX // (H * D) + 1024
    q = _meta((1, s_over, H, D))
    assert _needs_int64_offsets(q, tensor_layout="NHD", blk=128), (
        f"NHD S={s_over} has max offset {_max_element_offset(q, 'NHD', 128)} "
        f"> int32 max {INT32_MAX}, must select int64"
    )


def test_selects_int64_when_hnd_head_term_overflows():
    # HND: stride_h = S*D dominates; the boundary sits slightly higher
    # than NHD's because the head index tops out at H-1, not H.
    s_over = INT32_MAX // ((H - 1) * D) + 1024
    q = _meta((1, H, s_over, D))
    assert _needs_int64_offsets(q, tensor_layout="HND", blk=128), (
        f"HND S={s_over} has max offset {_max_element_offset(q, 'HND', 128)} "
        f"> int32 max {INT32_MAX}, must select int64"
    )


def test_selects_int64_when_batch_pushes_it_over():
    # The batch term carries stride_z = S*H*D, so a tensor that is safe at
    # batch 1 is not safe at batch 4. Catches a bound derived from the
    # per-sample size instead of the whole tensor.
    s_safe = INT32_MAX // (H * D) - 4096
    assert not _needs_int64_offsets(_meta((1, s_safe, H, D)), "NHD", 128)
    assert _needs_int64_offsets(_meta((4, s_safe, H, D)), "NHD", 128)


def test_stays_int32_at_minimax_h3_production_shape():
    # fl2va at the node's default canvas, the shape this fork actually
    # runs. Must not pay for int64 addressing.
    q = _meta((1, 41822, H, D))
    assert not _needs_int64_offsets(q, "NHD", 128)
    assert not _needs_int64_offsets(_meta((1, H, 41822, D)), "HND", 128)


def test_stays_int32_at_ltx_production_shape():
    q = _meta((1, 32, 23296, D))
    assert not _needs_int64_offsets(q, "HND", 128)


def test_bound_accounts_for_block_padding():
    # The kernel forms pointers for the full padded grid before masking
    # the load, so offs_n runs to ceil(S/blk)*blk - 1, not S-1. A bound
    # that stops at S-1 under-reports by up to blk-1 rows.
    s = 4096 + 1
    q = _meta((1, s, H, D))
    padded = _max_element_offset(q, "NHD", blk=128)
    unpadded = (s - 1) * H * D + D
    assert padded > unpadded, (
        f"bound {padded} must exceed the unpadded estimate {unpadded}"
    )


# --- end-to-end cases (need a big GPU; skipped otherwise) ---


def _free_vram_bytes():
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return free


def _check_tail_roundtrip(layout, s):
    """Quantize, then dequantize the last rows and compare to the source.

    Reads the tail specifically: overflow wraps the largest offsets, so
    the head of the tensor stays correct either way and would not
    discriminate.
    """
    shape = (1, s, H, D) if layout == "NHD" else (1, H, s, D)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q_int8, q_scale, _, _ = per_thread_int8(
        q, q, tensor_layout=layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64
    )
    seq_dim = 1 if layout == "NHD" else 2
    src = q.narrow(seq_dim, s - 32, 32).float()
    got = q_int8.narrow(seq_dim, s - 32, 32).float()
    # Per-thread scales make an exact dequant fiddly; cosine similarity on
    # the raw int8 is enough to separate "quantized correctly" (~0.99)
    # from "wrapped and wrote zeros" (0.0).
    cos = torch.nn.functional.cosine_similarity(
        src.reshape(-1), got.reshape(-1), dim=0
    ).item()
    del q, q_int8, q_scale
    torch.cuda.empty_cache()
    return cos


def test_tail_correct_above_int32_boundary_nhd():
    s = INT32_MAX // (H * D) + 4096
    cos = _check_tail_roundtrip("NHD", s)
    assert cos > 0.95, (
        f"NHD S={s} (above the int32 boundary) quantized tail has "
        f"cosine {cos:.4f} vs source; 0.0 means the offsets wrapped and "
        f"the tail was never written"
    )


def test_tail_correct_above_int32_boundary_hnd():
    s = INT32_MAX // ((H - 1) * D) + 4096
    cos = _check_tail_roundtrip("HND", s)
    assert cos > 0.95, (
        f"HND S={s} (above the int32 boundary) quantized tail has "
        f"cosine {cos:.4f} vs source"
    )


LIGHT_CASES = [
    test_selects_int64_when_nhd_row_term_overflows,
    test_selects_int64_when_hnd_head_term_overflows,
    test_selects_int64_when_batch_pushes_it_over,
    test_stays_int32_at_minimax_h3_production_shape,
    test_stays_int32_at_ltx_production_shape,
    test_bound_accounts_for_block_padding,
]

HEAVY_CASES = [
    test_tail_correct_above_int32_boundary_nhd,
    test_tail_correct_above_int32_boundary_hnd,
]


def main() -> int:
    cases = list(LIGHT_CASES)
    free = _free_vram_bytes()
    if free >= HEAVY_VRAM_BYTES:
        cases += HEAVY_CASES
    else:
        print(
            f"Skipping {len(HEAVY_CASES)} above-boundary cases: need "
            f"{HEAVY_VRAM_BYTES / 2**30:.0f} GiB free VRAM, have "
            f"{free / 2**30:.1f} GiB.\n"
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
