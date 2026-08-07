"""Kernel divergence at MiniMax H3's attention config, across sequence length.

This is a tripwire, not a quality metric. It measures how far each kernel
lands from a higher-precision computation of the *same* attention, at
fixed inputs. That is meaningful because it has no sampler in the loop:
at 20 steps of a flow-matching ODE any perturbation diverges the
trajectory, so comparing finished renders numerically measures chaos
rather than degradation. Here there is no trajectory, so the number means
what it looks like it means.

What it is for: catching structural breakage (NaNs, a dropped mask, a
wrong layout) and answering one specific mechanistic question --

    does per-call error grow with sequence length?

There is a real candidate mechanism. `pv_accum_dtype="fp32+fp16"` co-varies
the V-quant `scale_max` down to 2.25 from 448 (core.py), and that scale is
per-channel across the whole sequence. A longer sequence is a longer
window for an outlier to compress the scale and coarsen every other value
in that channel. If that dominates, rtol should climb with S.

It also settles a claim written into ComfyUI-h3-explorations's kernel setup
without evidence -- that `smooth_k` "buys nothing measurable at these
shapes". SageAttention defaults it on, KJNodes' H3 patch enables it, and
this node turned it off. Either the claim holds or the node is needlessly
less accurate than its peer.

Reference is fp32 EFFICIENT_ATTENTION over the same bf16-rounded inputs,
so input rounding is shared and what is left is the kernel's own error.
bf16 SDPA would not do: it is another approximation, not ground truth.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_kernel_divergence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention
from test_sageattn_ltx_shapes import accuracy_metrics

H, D = 56, 128  # MiniMax H3 attention config

# Packed-sequence lengths for real video durations at the 1344x768 canvas
# (1008 rows per latent frame), plus a short one to anchor the low end.
SHAPES = [
    ("len~5    S=2k", 2048),
    ("len~39   S=11k", 11312),
    ("len 73   S=22k", 22496),
    ("len 124  S=38k", 37810),
]

ARMS = [
    ("fp8++  smooth_k=False", "sageattn_qk_int8_pv_fp8_cuda",
     dict(pv_accum_dtype="fp32+fp16", smooth_k=False)),
    ("fp8++  smooth_k=True", "sageattn_qk_int8_pv_fp8_cuda",
     dict(pv_accum_dtype="fp32+fp16", smooth_k=True)),
    ("fp8    smooth_k=False", "sageattn_qk_int8_pv_fp8_cuda",
     dict(pv_accum_dtype="fp32+fp32", smooth_k=False)),
    ("fp16   smooth_k=False", "sageattn_qk_int8_pv_fp16_cuda",
     dict(pv_accum_dtype="fp32", smooth_k=False)),
    ("fp16   smooth_k=True", "sageattn_qk_int8_pv_fp16_cuda",
     dict(pv_accum_dtype="fp32", smooth_k=True)),
]


@torch.inference_mode()
def main():
    print(f"MiniMax H3 config: heads={H} head_dim={D} bf16 unmasked HND")
    print("Reference: fp32 EFFICIENT_ATTENTION over the same bf16 inputs.")
    print("Tripwire only -- this is divergence, not perceived quality.\n")

    width = max(len(a[0]) for a in ARMS)
    header = f"{'':{width}s}" + "".join(f"{name:>18s}" for name, _ in SHAPES)
    print(header)

    # Attention is independent per head, so slicing heads is exact rather
    # than an approximation. Chunking keeps the fp32 reference bounded at
    # a few hundred MiB instead of scaling with S*H, which matters because
    # this is most useful when a model is already resident on the card.
    CHUNK = 8
    rows = {label: [] for label, _, _ in ARMS}
    for _, S in SHAPES:
        torch.manual_seed(0)
        q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)

        acc = {label: [] for label, _, _ in ARMS}
        for h0 in range(0, H, CHUNK):
            sl = slice(h0, min(h0 + CHUNK, H))
            qc, kc, vc = q[:, sl], k[:, sl], v[:, sl]
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                ref = F.scaled_dot_product_attention(
                    qc.float(), kc.float(), vc.float())
            for label, attr, kw in ARMS:
                fn = getattr(sageattention, attr)
                out = fn(qc.contiguous(), kc.contiguous(), vc.contiguous(),
                         tensor_layout="HND", is_causal=False, **kw)
                acc[label].append(accuracy_metrics(out, ref)[0])
                del out
            del ref
            torch.cuda.empty_cache()

        for label, _, _ in ARMS:
            rows[label].append(sum(acc[label]) / len(acc[label]))
        del q, k, v
        torch.cuda.empty_cache()

    for label, _, _ in ARMS:
        cells = "".join(f"{r:>18.4f}" for r in rows[label])
        print(f"{label:{width}s}{cells}")

    print()
    for label, _, _ in ARMS:
        first, last = rows[label][0], rows[label][-1]
        trend = "grows" if last > first * 1.15 else (
            "shrinks" if last < first * 0.87 else "flat")
        print(f"  {label:{width}s} 2k -> 38k: {first:.4f} -> {last:.4f}  ({trend})")

    print("\nsmooth_k effect at the longest shape:")
    for base, on in (("fp8++  smooth_k=False", "fp8++  smooth_k=True"),
                     ("fp16   smooth_k=False", "fp16   smooth_k=True")):
        off_v, on_v = rows[base][-1], rows[on][-1]
        delta = 100.0 * (on_v - off_v) / off_v
        print(f"  {base.split()[0]:6s} {off_v:.4f} -> {on_v:.4f}  ({delta:+.1f}%)")


if __name__ == "__main__":
    main()
