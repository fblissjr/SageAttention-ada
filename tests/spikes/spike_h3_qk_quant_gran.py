"""per-thread Triton q/k versus per-warp CUDA q/k on sm89, at MiniMax H3's shape.

`sageattn_consume` on sm89 leaves `qk_quant_gran` at the fp8 wrapper's
`per_thread` default, so q and k are quantized by the Triton kernel in
`sageattention/triton/quant_per_thread.py` while v goes through the CUDA
kernels in `csrc/fused/fused.cu`. The dispatcher sets `per_warp` on
sm100/sm120/sm121 and its comment says the two branches were never graded
against each other; the sm89 choice is an inherited default, not a
measurement. This spike is that measurement, on the shape that ships: one
fused qkv buffer split three ways (stride_seq 21504), NHD, unmasked,
`smooth_k=False`, `pv_accum_dtype="fp32+fp16"`, through `sageattn_consume`
exactly as the consumer calls it.

Two things are reported per row, because a speed difference needs its
mechanism shown and an accuracy difference needs a referent:

  quant only  -- the q/k quantization step in isolation, both arms. This is
                 the only code that differs between them, so the whole-call
                 delta must be explained by this number or it is noise.
  whole call  -- `sageattn_consume` wall time and peak, both arms, plus mean
                 rtol against fp32-accumulated flash SDPA on the same inputs
                 and the tail scored separately.

The rtol here is on synthetic input and carries the caveat in CLAUDE.md: it
is a ranking between the two arms, not an accuracy figure for either. The
accuracy call is made on captured activations with
`spike_h3_real_activations.py`, which has per-warp arms for this purpose.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_qk_quant_gran.py [--rows S ...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sageattention
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton
from spike_h3_long_sequence import chunked_mean_rtol, make_qkv, reference, tail_report
from test_sageattn_ltx_shapes import time_and_vram

# 124-frame default and 345-frame ceiling at 1344x768, the 362-frame OOD row,
# and the heaviest packed graph the consumer reports rendering on 24 GB.
ROWS = [41_822, 104_030, 109_126, 149_000]

ARMS = [
    ("per_thread (Triton q/k)", "per_thread"),
    ("per_warp   (CUDA q/k)", "per_warp"),
]

QUANT_KW = dict(smooth_k=False, pv_accum_dtype="fp32+fp16")


def quant_step(gran, q, k):
    if gran == "per_thread":
        return per_thread_int8_triton(q, k, None, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
    return per_warp_int8_cuda(q, k, None, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=ROWS)
    args = ap.parse_args()

    info = sageattention.build_info()
    print(f"torch {torch.__version__}  sage {info['describe']}  {torch.cuda.get_device_name()}")
    print("fused-QKV view, NHD, heads 56, head_dim 128, smooth_k=False, fp32+fp16, via sageattn_consume")
    print("reference: flash SDPA on the same bf16 inputs\n")

    hdr = (f"{'S':>8}  {'arm':24s} {'quant ms':>9} {'call ms':>8} {'peak MiB':>9} "
           f"{'rtol':>7} {'tail rtol':>9} {'tail cos':>9}  {'vs other':>8}")
    print(hdr)
    print("-" * len(hdr))
    torch.manual_seed(0)
    for s in args.rows:
        q, k, v, buf = make_qkv(s, fused=True)
        ref = reference(q, k, v)
        outs = {}
        rows = []
        for label, gran in ARMS:
            quant_ms, _ = time_and_vram(lambda: quant_step(gran, q, k), warmup=2, runs=7)

            def call():
                # consume empties the list; q/k/v stay alive through our
                # own references, which is the fused-view case where the
                # release frees nothing anyway.
                return sageattention.sageattn_consume(
                    [q, k, v], tensor_layout="NHD", is_causal=False,
                    qk_quant_gran=gran, **QUANT_KW,
                )

            out = call()
            call_ms, peak = time_and_vram(call, warmup=1, runs=5)
            rtol = chunked_mean_rtol(out, ref)
            t_rtol, t_cos, _ = tail_report(out, ref)
            outs[gran] = out
            rows.append((label, gran, quant_ms, call_ms, peak, rtol, t_rtol, t_cos))
        cross = chunked_mean_rtol(outs["per_warp"], outs["per_thread"])
        for label, gran, quant_ms, call_ms, peak, rtol, t_rtol, t_cos in rows:
            print(f"{s:>8,}  {label:24s} {quant_ms:>9.3f} {call_ms:>8.2f} {peak:>9.0f} "
                  f"{rtol:>7.4f} {t_rtol:>9.4f} {t_cos:>9.4f}  {cross:>8.4f}")
        print()
        del q, k, v, buf, ref, outs
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
