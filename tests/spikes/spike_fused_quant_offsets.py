"""Before/after record for the int64 strides in csrc/fused/fused.cu.

Upstream's fused CUDA quant kernels take every stride as uint32_t and form
the global element offset as a uint32_t sum, so a tensor past 2**32 elements
reads from a wrapped address with no error. On MiniMax H3's config (heads 56,
head_dim 128) with q/k/v as views of one fused qkv buffer (stride_seq 21504)
that is 199,729 rows, unreachable on a 24 GB card through `sageattn` but two
of these kernels stand alone under `per_channel_fp8` and fit. This fork
widened the strides to int64; this spike is the record of what that cost and
what it fixed, measured on both builds in one process:

  cost     -- per-kernel wall time at H3's row range, old build vs new,
              on the fused-view layout production hands over. int64
              address arithmetic is the only change, so any difference
              here is its price.
  identity -- outputs bit-equal between builds below the ceiling, so the
              new build changes nothing a consumer's record could see.
  fix      -- the dequantized tail of `per_channel_fp8` past the ceiling,
              scored on both builds. The old one is expected to fail.

The old build is loaded from a saved .so, so the comparison needs a copy of
`sageattention/_fused.<tag>.so` from before the change:

    $VIRTUAL_ENV/bin/python tests/spikes/spike_fused_quant_offsets.py \
        --old-so /path/to/old/_fused.cpython-3xx-x86_64-linux-gnu.so

Without `--old-so` only the new build is measured. Needs ~14 GiB free VRAM
for the past-ceiling row; it is skipped otherwise.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import sageattention
from sageattention import _fused as fused_new  # type: ignore[attr-defined]
from test_quant_offset_overflow import (
    FUSED_VIEW_UINT32_ROWS,
    cuda_fused_view_sub_mean_tail_report,
    cuda_fused_view_tail_report,
)
from test_sageattn_ltx_shapes import time_and_vram

HEADS, HEAD_DIM = 56, 128
# H3's row range as rendered today: the node's 124-frame default at
# 1344x768, the 345-frame legal ceiling there, the 362-frame OOD row the
# long-sequence spike keeps, and the heaviest packed graph the consumer
# reports rendering on this card.
ROWS = [41_822, 104_030, 109_126, 149_000]
PAST_CEILING_ROWS = (FUSED_VIEW_UINT32_ROWS + 1024 + 63) // 64 * 64
PAST_CEILING_VRAM_BYTES = 14 * 2**30


def load_old(path):
    """Load a saved `_fused` .so beside the installed one.

    The extension's init symbol is `PyInit__fused`, which CPython derives
    from the last component of the module name, so any name ending in
    `_fused` binds it; a key distinct from `sageattention._fused` keeps both
    builds importable at once.
    """
    spec = importlib.util.spec_from_file_location("_fused", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load an extension from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fused_view(s):
    buf = torch.randn(1, s, 3, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    return buf, buf[:, :, 0], buf[:, :, 1], buf[:, :, 2]


def per_warp_q(mod, q):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    scale = torch.empty((1, HEADS, (q.shape[1] + 127) // 128 * 4), dtype=torch.float32, device=q.device)
    mod.quant_per_warp_int8_cuda(q, q_int8, scale, 128, 32, 0)
    return q_int8, scale


def per_block_k(mod, k):
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    scale = torch.empty((1, HEADS, (k.shape[1] + 63) // 64), dtype=torch.float32, device=k.device)
    mod.quant_per_block_int8_cuda(k, k_int8, scale, 64, 0)
    return k_int8, scale


def per_channel_v(mod, v):
    """`per_channel_fp8` as the sm89 fp32+fp16 path calls it (smooth_v off)."""
    s = v.shape[1]
    padded = (s + 63) // 64 * 64
    vt = torch.empty((1, HEAD_DIM, HEADS, padded), dtype=v.dtype, device=v.device)
    mod.transpose_pad_permute_cuda(v, vt, 0)
    v_fp8 = torch.empty(vt.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((1, HEADS, HEAD_DIM), dtype=torch.float32, device=v.device)
    mod.scale_fuse_quant_cuda(vt, v_fp8, v_scale, s, 2.25, 0)
    return v_fp8, v_scale


def fuse_sub_mean_k(mod, k):
    """`per_block_int8` with `km` given: the smooth_k=True k path on sm80/fp16."""
    km = k.float().mean(dim=1).to(k.dtype).squeeze(0)[None]        # [1, H, D]
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    scale = torch.empty((1, HEADS, (k.shape[1] + 63) // 64), dtype=torch.float32, device=k.device)
    mod.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, scale, 64, 0)
    return k_int8, scale


def sub_mean_v(mod, v):
    """`sub_mean`: the smooth_v path of the fp16 kernels, bf16 in, fp16 out."""
    vm = v.mean(dim=1)                                              # [1, H, D]
    out = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    mod.sub_mean_cuda(v, vm, out, 0)
    return (out,)


KERNELS = [
    ("per_warp q  (quant_per_warp_int8)", per_warp_q, 1),
    ("per_block k (quant_per_block_int8)", per_block_k, 2),
    ("per_channel v (transpose+scale_fuse)", per_channel_v, 3),
    # Not on the sm89 production path; widened by the same edit, so they
    # carry the same evidence.
    ("fuse_sub_mean k (per_block + km)", fuse_sub_mean_k, 2),
    ("sub_mean v (bf16 -> fp16)", sub_mean_v, 3),
]


def equal_all(a, b):
    return all(torch.equal(x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x,
                           y.view(torch.uint8) if y.dtype == torch.float8_e4m3fn else y)
               for x, y in zip(a, b))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-so", type=Path, default=None)
    ap.add_argument("--rows", type=int, nargs="*", default=ROWS)
    args = ap.parse_args()

    old = load_old(args.old_so) if args.old_so else None
    info = sageattention.build_info()
    print(f"torch {torch.__version__}  sage {info['describe']}  "
          f"{torch.cuda.get_device_name()}")
    print(f"new build: ELEMENT_OFFSET_BITS={getattr(fused_new, 'ELEMENT_OFFSET_BITS', 32)}"
          + (f"   old build: ELEMENT_OFFSET_BITS={getattr(old, 'ELEMENT_OFFSET_BITS', 32)}" if old else ""))
    print(f"heads {HEADS}, head_dim {HEAD_DIM}, fused-QKV view (stride_seq {3*HEADS*HEAD_DIM}), NHD\n")

    torch.manual_seed(0)
    hdr = f"{'S':>8}  {'kernel':38s} {'new ms':>8} {'old ms':>8} {'new/old':>8}  {'bit-equal':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in args.rows:
        buf, q, k, v = fused_view(s)
        for label, fn, idx in KERNELS:
            src = (q, k, v)[idx - 1]
            new_out = fn(fused_new, src)
            new_ms, _ = time_and_vram(lambda: fn(fused_new, src), warmup=2, runs=7)
            if old is not None:
                old_out = fn(old, src)
                old_ms, _ = time_and_vram(lambda: fn(old, src), warmup=2, runs=7)
                same = "yes" if equal_all(new_out, old_out) else "NO"
                print(f"{s:>8,}  {label:38s} {new_ms:>8.3f} {old_ms:>8.3f} {new_ms/old_ms:>8.3f}  {same:>9}")
                del old_out
            else:
                print(f"{s:>8,}  {label:38s} {new_ms:>8.3f} {'-':>8} {'-':>8}  {'-':>9}")
            del new_out
        del buf, q, k, v
        torch.cuda.empty_cache()
        print()

    free, _ = torch.cuda.mem_get_info()
    if free < PAST_CEILING_VRAM_BYTES:
        print(f"past-ceiling row skipped: need {PAST_CEILING_VRAM_BYTES/2**30:.0f} GiB free, have {free/2**30:.1f} GiB")
        return 0
    for name, report in (("per_channel_fp8", cuda_fused_view_tail_report),
                         ("sub_mean", cuda_fused_view_sub_mean_tail_report)):
        print(f"{name} tail at S={PAST_CEILING_ROWS:,} (fused view; uint32 ceiling is "
              f"S={FUSED_VIEW_UINT32_ROWS:,})")
        print(f"{'build':>6}  {'tail cos':>9} {'tail 0s':>8} {'tail NaN':>9}")
        for label, mod in (("new", None), ("old", old)):
            if label == "old" and old is None:
                continue
            cos, zeros, nans = report(PAST_CEILING_ROWS, fused_mod=mod)
            print(f"{label:>6}  {cos:>9.4f} {zeros:>8.1%} {nans:>9.1%}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
