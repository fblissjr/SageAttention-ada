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

The CUDA quant kernels (`csrc/fused/fused.cu`, the v side of the sm89 fp8
path and both sides of `qk_quant_gran="per_warp"`) had the same defect one
type wider: uint32 strides summed in uint32, wrapping at 2**32 elements,
which on a fused-QKV view at H=56 D=128 is 199,729 rows. This fork widened
those strides to int64 and stamps `ELEMENT_OFFSET_BITS` on the extension;
`sageattention/quant.py` refuses at the host when the installed build's
width is narrower than the tensor needs. Its cases:
  - cuda_guard_*         : the refusal fires past the boundary and not
                           below it, names the row count and the ceiling,
                           and runs before any kernel launch. Delete and a
                           stale 32-bit .so goes back to reading the wrong
                           address silently.
  - cuda_build_is_64bit  : the installed build carries the width. Delete
                           and a checkout from before the fix passes the
                           guard cases while its kernels still wrap.
  - cuda_tail_correct_*  : the widened kernels, driven past 2**32 elements
                           on a fused-QKV view -- the fp8 v path and the
                           fp16 kernels' sub_mean. Delete and the guard
                           could be right while the kernel is still wrong.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
The tail_correct cases need ~13 GiB of free VRAM (14 GiB for the CUDA one)
and skip without it; everything else runs on meta tensors and needs no GPU
at all.
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
from sageattention import quant as cuda_quant


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


# --- CUDA fused-kernel guard cases (meta tensors; no GPU needed) ---

UINT32_CEILING = 2**32
# Largest S at which every element a fused-QKV view (stride_seq = 3*H*D)
# dereferences still sits below 2**32 in the CUDA v-side kernels, from
# CHANGELOG.md's known-issue entry. Derived below rather than trusted; the
# constant is the claim. One row more and the last read wraps.
FUSED_VIEW_UINT32_ROWS = 199_729


def _meta_fused_v(s):
    """v as an H3 block hands it over: one view of a fused [S, 3, H, D] buffer."""
    return torch.empty((1, s, 3, H, D), dtype=torch.bfloat16, device="meta")[:, :, 2]


def _fused_v_rows_at_ceiling(bits):
    """Smallest S whose 64-row padded v view forms an offset >= 2**bits."""
    lo, hi = 1, 2**40
    while lo < hi:
        mid = (lo + hi) // 2
        if _max_element_offset(_meta_fused_v(mid), "NHD", 64) >= 2**bits:
            hi = mid
        else:
            lo = mid + 1
    return lo


def test_cuda_guard_ceiling_matches_the_recorded_row_count():
    # The CHANGELOG has carried 199,729 since 2026-08-05 as the fused-view
    # ceiling. Two readings have to agree with it, or one of the three is
    # wrong: the exact one (blk=1: the last row actually dereferenced) must
    # put S=199,729 inside and S=199,730 past; and the guard, which bounds
    # the padded grid the kernels form pointers for, must fire within one
    # block below that -- conservative, never late.
    inside = _max_element_offset(_meta_fused_v(FUSED_VIEW_UINT32_ROWS), "NHD", 1)
    past = _max_element_offset(_meta_fused_v(FUSED_VIEW_UINT32_ROWS + 1), "NHD", 1)
    assert inside < UINT32_CEILING <= past, (
        f"exact crossing is not at S={FUSED_VIEW_UINT32_ROWS:,}: offsets "
        f"{inside:,} (S) and {past:,} (S+1) around {UINT32_CEILING:,}"
    )
    got = _fused_v_rows_at_ceiling(32)
    assert got <= FUSED_VIEW_UINT32_ROWS + 1 < got + 64, (
        f"guard first refuses at S={got:,}; must be within one 64-row block "
        f"below the exact crossing at S={FUSED_VIEW_UINT32_ROWS + 1:,}"
    )


def test_cuda_guard_refuses_past_the_32bit_boundary_and_not_below():
    s = FUSED_VIEW_UINT32_ROWS
    try:
        cuda_quant._check_element_offsets(
            [("v", _meta_fused_v(s))], "NHD", 64, offset_bits=32
        )
    except ValueError as exc:
        msg = str(exc)
        assert f"{s:,} rows" in msg and f"{UINT32_CEILING:,}" in msg, (
            f"refusal must name the row count and the ceiling; got: {msg}"
        )
    else:
        raise AssertionError(f"fused view at S={s:,} must be refused on a 32-bit build")
    # 64 rows lower is the last padded block before the crossing.
    cuda_quant._check_element_offsets(
        [("v", _meta_fused_v(s - 64))], "NHD", 64, offset_bits=32
    )


def test_cuda_guard_is_silent_on_a_64bit_build():
    # Same tensor, the width this fork now builds: no refusal.
    cuda_quant._check_element_offsets(
        [("v", _meta_fused_v(FUSED_VIEW_UINT32_ROWS))], "NHD", 64, offset_bits=64
    )
    # And well past the old ceiling, where a 24 GB card cannot follow.
    cuda_quant._check_element_offsets(
        [("v", _meta_fused_v(4 * FUSED_VIEW_UINT32_ROWS))], "NHD", 64, offset_bits=64
    )


def test_cuda_build_is_64bit():
    assert cuda_quant.ELEMENT_OFFSET_BITS == 64, (
        f"installed _fused reports {cuda_quant.ELEMENT_OFFSET_BITS}-bit offsets; "
        f"a build from before the int64 strides, or a stale .so for this "
        f"interpreter. Rebuild with ./build.sh."
    )


def test_cuda_guard_runs_before_the_wrapper_launches():
    # On a meta tensor the extension itself would raise (CHECK_CUDA), so a
    # ValueError from our guard proves the check sits before the launch.
    # Patching the module width is the stale-build state the guard exists
    # for; a meta v has no storage, so this costs nothing.
    saved = cuda_quant.ELEMENT_OFFSET_BITS
    cuda_quant.ELEMENT_OFFSET_BITS = 32
    try:
        try:
            cuda_quant.per_channel_fp8(_meta_fused_v(FUSED_VIEW_UINT32_ROWS), tensor_layout="NHD")
        except ValueError as exc:
            assert "fused CUDA quant kernels" in str(exc), str(exc)
        else:
            raise AssertionError("per_channel_fp8 launched past the ceiling on a 32-bit build")
    finally:
        cuda_quant.ELEMENT_OFFSET_BITS = saved


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


# Inverse of TransposePadPermuteKernel's 16-row permutation: output row j
# of each 16-row group holds input row FP8_PERMUTE[j].
FP8_PERMUTE = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
CUDA_HEAVY_VRAM_BYTES = 14 * 2**30


def cuda_fused_view_tail_report(s, tail_rows=4096, fused_mod=None):
    """Drive per_channel_fp8 on a fused-QKV v view at S and score its tail.

    The buffer is zero except for the last `tail_rows`, so a wrapped read --
    which lands 2**32 elements earlier, inside the zero head -- produces a
    zero or NaN tail, and a correct one reproduces the source through the
    kernel's own permutation. Returns (cosine, zero_fraction, nan_fraction)
    over the dequantized tail. `fused_mod` substitutes another build of the
    extension, for a before/after record.
    """
    buf = torch.zeros((1, s, 3, H, D), dtype=torch.bfloat16, device="cuda")
    v = buf[:, :, 2]
    v[:, -tail_rows:].normal_()
    if fused_mod is None:
        v_fp8, v_scale, _ = cuda_quant.per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25, smooth_v=False)
    else:
        padded = (s + 63) // 64 * 64
        vt = torch.empty((1, D, H, padded), dtype=v.dtype, device=v.device)
        fused_mod.transpose_pad_permute_cuda(v, vt, 0)
        v_fp8 = torch.empty(vt.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((1, H, D), dtype=torch.float32, device=v.device)
        fused_mod.scale_fuse_quant_cuda(vt, v_fp8, v_scale, s, 2.25, 0)
        del vt
    # NHD output is [B, D, H, padded]; compare the last tail_rows of the
    # sequence axis, which sits at a 16-aligned offset when s is.
    assert s % 16 == 0 and tail_rows % 16 == 0
    got = v_fp8[0, :, :, s - tail_rows:s].float()                    # [D, H, T]
    got = got * v_scale[0].t().unsqueeze(-1)                          # dequant
    src = v[0, -tail_rows:].float().permute(2, 1, 0)                  # [D, H, T]
    src = src.reshape(D, H, tail_rows // 16, 16)[..., FP8_PERMUTE].reshape(D, H, tail_rows)
    nan_frac = torch.isnan(got).float().mean().item()
    got = torch.nan_to_num(got)
    zero_frac = (got == 0).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(got.reshape(-1), src.reshape(-1), dim=0).item()
    del buf, v, v_fp8, v_scale, got, src
    torch.cuda.empty_cache()
    return cos, zero_frac, nan_frac


def cuda_fused_view_sub_mean_tail_report(s, tail_rows=4096, fused_mod=None):
    """Same drive for `sub_mean` (the fp16 kernels' smooth_v path): v as a
    fused view past the ceiling, output is `v - mean` in fp16, scored on
    the tail against the same subtraction done in fp32."""
    buf = torch.zeros((1, s, 3, H, D), dtype=torch.bfloat16, device="cuda")
    v = buf[:, :, 2]
    v[:, -tail_rows:].normal_()
    if fused_mod is None:
        out, vm = cuda_quant.sub_mean(v, tensor_layout="NHD")
    else:
        vm = v.mean(dim=1)
        out = torch.empty(v.shape, dtype=torch.float16, device=v.device)
        fused_mod.sub_mean_cuda(v, vm, out, 0)
    got = out[:, -tail_rows:].float()
    src = (v[:, -tail_rows:].float() - vm.float()[:, None])
    nan_frac = torch.isnan(got).float().mean().item()
    got = torch.nan_to_num(got)
    zero_frac = (got == 0).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(got.reshape(-1), src.reshape(-1), dim=0).item()
    del buf, v, out, vm, got, src
    torch.cuda.empty_cache()
    return cos, zero_frac, nan_frac


def test_cuda_sub_mean_tail_correct_above_uint32_boundary_fused_view():
    s = (FUSED_VIEW_UINT32_ROWS + 1024 + 63) // 64 * 64
    cos, zero_frac, nan_frac = cuda_fused_view_sub_mean_tail_report(s)
    assert cos > 0.99, (
        f"sub_mean on a fused view at S={s:,} (past the uint32 ceiling) has tail "
        f"cosine {cos:.4f} vs the fp32 subtraction, {zero_frac:.1%} zeros, "
        f"{nan_frac:.1%} NaN; the kernel read or wrote a wrapped address"
    )


def test_cuda_tail_correct_above_uint32_boundary_fused_view():
    s = (FUSED_VIEW_UINT32_ROWS + 1024 + 63) // 64 * 64
    cos, zero_frac, nan_frac = cuda_fused_view_tail_report(s)
    assert cos > 0.95, (
        f"fused view S={s:,} (past the uint32 ceiling) dequantized tail has "
        f"cosine {cos:.4f} vs source, {zero_frac:.1%} zeros, {nan_frac:.1%} "
        f"NaN; the kernels read from a wrapped address"
    )


LIGHT_CASES = [
    test_selects_int64_when_nhd_row_term_overflows,
    test_selects_int64_when_hnd_head_term_overflows,
    test_selects_int64_when_batch_pushes_it_over,
    test_stays_int32_at_minimax_h3_production_shape,
    test_stays_int32_at_ltx_production_shape,
    test_bound_accounts_for_block_padding,
    test_cuda_guard_ceiling_matches_the_recorded_row_count,
    test_cuda_guard_refuses_past_the_32bit_boundary_and_not_below,
    test_cuda_guard_is_silent_on_a_64bit_build,
    test_cuda_build_is_64bit,
    test_cuda_guard_runs_before_the_wrapper_launches,
]

HEAVY_CASES = [
    test_tail_correct_above_int32_boundary_nhd,
    test_tail_correct_above_int32_boundary_hnd,
]

CUDA_HEAVY_CASES = [
    test_cuda_tail_correct_above_uint32_boundary_fused_view,
    test_cuda_sub_mean_tail_correct_above_uint32_boundary_fused_view,
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
    if free >= CUDA_HEAVY_VRAM_BYTES:
        cases += CUDA_HEAVY_CASES
    else:
        print(
            f"Skipping {len(CUDA_HEAVY_CASES)} CUDA above-boundary cases: need "
            f"{CUDA_HEAVY_VRAM_BYTES / 2**30:.0f} GiB free VRAM, have "
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
