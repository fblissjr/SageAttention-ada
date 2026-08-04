"""Kernel divergence on REAL MiniMax H3 activations, not synthetic ones.

The synthetic version of this sweep (spike_h3_kernel_divergence.py) cannot
answer the `smooth_k` question and should not be read as if it does.
`smooth_k` subtracts the per-head channel mean of K before quantizing,
which buys headroom only when K actually carries a channel offset.
`torch.randn` has zero mean by construction, so that experiment reports
"no effect" whether or not the effect is real.

This runs the same comparison on q/k/v captured from an actual H3 forward
pass -- post-RMSNorm, post-RoPE, exactly the tensors sage receives -- and
first reports how large the K channel offset actually is, since that is
the quantity `smooth_k` exists to remove. If the offset is small relative
to K's spread there is nothing to win and `smooth_k=False` is right; if it
is large and rtol still does not move, the mechanism is not what limits
accuracy at these shapes.

Capture with a throwaway hook on `Attention.forward`; pass the resulting
.pt files (each holding q/k/v in HND) as arguments.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_real_activations.py <capture.pt> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention
from test_sageattn_ltx_shapes import accuracy_metrics

ARMS = [
    ("fp8++  smooth_k=False", "sageattn_qk_int8_pv_fp8_cuda",
     dict(pv_accum_dtype="fp32+fp16", smooth_k=False)),
    ("fp8++  smooth_k=True", "sageattn_qk_int8_pv_fp8_cuda",
     dict(pv_accum_dtype="fp32+fp16", smooth_k=True)),
    ("fp16   smooth_k=False", "sageattn_qk_int8_pv_fp16_cuda",
     dict(pv_accum_dtype="fp32", smooth_k=False)),
    ("fp16   smooth_k=True", "sageattn_qk_int8_pv_fp16_cuda",
     dict(pv_accum_dtype="fp32", smooth_k=True)),
]

CHUNK = 8


def describe_k(k):
    """How much of K is a constant offset, per head-channel.

    ratio = |mean| / std along the sequence axis. Near 0 means K is already
    centred and smooth_k has nothing to remove; >1 means the offset
    dominates the spread and is eating most of the INT8 range.
    """
    km = k.float().mean(dim=2)                  # [B, H, D]
    ks = k.float().std(dim=2).clamp(min=1e-6)   # [B, H, D]
    ratio = (km.abs() / ks)
    return ratio.mean().item(), ratio.max().item()


@torch.inference_mode()
def run(path):
    d = torch.load(path, map_location="cuda", weights_only=True)
    q, k, v = d["q"], d["k"], d["v"]
    _, H, S, D = q.shape
    mean_ratio, max_ratio = describe_k(k)
    print(f"\n=== {Path(path).name}  S={S} heads={H} head_dim={D} ===")
    print(f"K channel offset |mean|/std:  mean {mean_ratio:.3f}   max {max_ratio:.3f}")
    if mean_ratio < 0.1:
        print("  -> K is essentially centred already; smooth_k has little to remove.")
    elif mean_ratio > 0.5:
        print("  -> substantial offset; smooth_k should buy real INT8 headroom.")
    else:
        print("  -> moderate offset.")

    results = {}
    for label, attr, kw in ARMS:
        acc = []
        for h0 in range(0, H, CHUNK):
            sl = slice(h0, min(h0 + CHUNK, H))
            qc = q[:, sl].contiguous()
            kc = k[:, sl].contiguous()
            vc = v[:, sl].contiguous()
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                ref = F.scaled_dot_product_attention(
                    qc.float(), kc.float(), vc.float())
            out = getattr(sageattention, attr)(
                qc, kc, vc, tensor_layout="HND", is_causal=False, **kw)
            acc.append(accuracy_metrics(out, ref)[0])
            del ref, out, qc, kc, vc
            torch.cuda.empty_cache()
        results[label] = sum(acc) / len(acc)

    width = max(len(a[0]) for a in ARMS)
    for label, _, _ in ARMS:
        print(f"  {label:{width}s} mean_rtol {results[label]:.4f}")

    for base, on in (("fp8++  smooth_k=False", "fp8++  smooth_k=True"),
                     ("fp16   smooth_k=False", "fp16   smooth_k=True")):
        off_v, on_v = results[base], results[on]
        delta = 100.0 * (on_v - off_v) / off_v
        verdict = "helps" if delta < -3 else ("hurts" if delta > 3 else "no effect")
        print(f"  smooth_k on {base.split()[0]:6s}: "
              f"{off_v:.4f} -> {on_v:.4f}  ({delta:+.1f}%, {verdict})")

    del q, k, v
    torch.cuda.empty_cache()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    print("Reference: fp32 EFFICIENT_ATTENTION over the same captured bf16 inputs.")
    for path in sys.argv[1:]:
        run(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
