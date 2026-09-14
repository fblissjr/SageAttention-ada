"""Does balancing K's channels before INT8 quantization cut sage's error on H3?

Motivation, from two records that agree. The consumer's 2026-08-20 head
magnitude analysis attributes block 49's INT8 error to four K channels
carrying most of the energy (a property of the released K-norm weights), and
this fork's 2026-09-14 grade of `smooth_k` across ten capture cells found
quantization error rising ~5x from block 0 to block 49 with mean-subtraction
buying only a few percent at the deep blocks. Both point at the same
mechanism: sage quantizes K per block of tokens with one scale across all
128 channels, so a few loud channels set the scale and the quiet ones lose
resolution. Mean-subtraction removes only the constant part of a loud
channel.

The lever this tests is exact for the dot product: for any per-(head,
channel) factor s, q . k == (q * s) . (k / s). Choose s from the channel
magnitudes and the loud channels stop dominating the INT8 scale. Three
variants, in the SmoothQuant form s = rms_k^a / rms_q^(1-a), normalized per
head so the geometric mean over channels is one:

  free       a=0.5, any per-channel s. Upper bound on what balancing buys.
  pair       a=0.5, s equal within each RoPE pair. H3 rotates channels
             (i, i+48) together for i < 48 and leaves 96..127 unrotated
             (`comfy/ldm/minimax/model.py`, split-half, rot_dim 96), so a
             pair-equal s commutes with RoPE and folds into q_norm/k_norm
             weights on the consumer side at zero runtime cost. This is
             the deployable variant.
  pair-k     a=1.0, pair-equal. K fully equalized, Q takes the whole scale.

Graded like `spike_h3_real_activations.py`: captured q/k/v in HND, fp32
EFFICIENT_ATTENTION reference on the ORIGINAL q/k/v (the balanced call
computes the same function, so the referent does not move), mean rtol over
8-head chunks, sage fp8++ (`fp32+fp16`, `smooth_k=False`) as served. Also
prints the top-4 K channels by energy across heads with their RoPE mates,
to check against the consumer's 82/34/67/19.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_k_channel_balance.py <capture.pt> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention
from test_sageattn_ltx_shapes import accuracy_metrics

CHUNK = 8
ROT = 96           # rotated channels; pairs are (i, i + ROT // 2)

ARMS = [
    ("plain", None, False),
    ("free  a=0.5", 0.5, False),
    ("pair  a=0.5", 0.5, True),
    ("pair-k a=1.0", 1.0, True),
]

# --sweep replaces ARMS with a pair-equal alpha ladder, for picking a value
# once a block is known to respond.
SWEEP_ARMS = [("plain", None, False)] + [
    (f"pair  a={a:.2f}", a, True) for a in (0.15, 0.25, 0.35, 0.5, 0.65, 0.75)
]


def channel_rms(x):
    """[1, H, S, D] -> [H, D] rms over the sequence, in fp32, one head at a time."""
    out = torch.empty(x.shape[1], x.shape[3], device=x.device)
    for h in range(x.shape[1]):
        out[h] = x[0, h].float().pow(2).mean(dim=0).sqrt()
    return out


def balance_factor(rk, rq, alpha, pair):
    s = rk.clamp(min=1e-6).pow(alpha) / rq.clamp(min=1e-6).pow(1.0 - alpha)   # [H, D]
    if pair:
        half = ROT // 2
        g = (s[:, :half] * s[:, half:ROT]).sqrt()
        s = torch.cat([g, g, s[:, ROT:]], dim=1)
    s = s / torch.exp(torch.log(s).mean(dim=1, keepdim=True))                    # geo-mean 1 per head
    return s


@torch.inference_mode()
def run(path):
    d = torch.load(path, map_location="cuda", weights_only=True)
    q, k, v = d["q"], d["k"], d["v"]
    _, H, S, D = q.shape
    rk, rq = channel_rms(k), channel_rms(q)
    energy = (rk ** 2).sum(dim=0)
    top = torch.topk(energy, 4).indices.tolist()
    mates = [(c + ROT // 2) % ROT if c < ROT else None for c in top]
    print(f"\n=== {Path(path).name}  S={S} heads={H} head_dim={D} ===")
    print(f"top-4 K channels by energy across heads: {top}  (share {(energy[top].sum() / energy.sum()).item():.1%}); "
          f"RoPE mates: {mates}")
    print(f"per-head K rms spread (max/min of head means): "
          f"{(rk.mean(dim=1).max() / rk.mean(dim=1).min()).item():.1f}x")

    factors = {}
    for label, alpha, pair in ARMS:
        factors[label] = None if alpha is None else balance_factor(rk, rq, alpha, pair).to(q.dtype)

    acc = {label: [] for label, _, _ in ARMS}
    for h0 in range(0, H, CHUNK):
        sl = slice(h0, min(h0 + CHUNK, H))
        qc, kc, vc = q[:, sl].contiguous(), k[:, sl].contiguous(), v[:, sl].contiguous()
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            ref = F.scaled_dot_product_attention(qc.float(), kc.float(), vc.float())
        for label, _, _ in ARMS:
            s = factors[label]
            if s is None:
                qa, ka = qc, kc
            else:
                sc = s[sl][None, :, None, :]
                qa, ka = (qc * sc).contiguous(), (kc / sc).contiguous()
            out = sageattention.sageattn_qk_int8_pv_fp8_cuda(
                qa, ka, vc, tensor_layout="HND", is_causal=False,
                pv_accum_dtype="fp32+fp16", smooth_k=False)
            acc[label].append(accuracy_metrics(out, ref)[0])
            del out, qa, ka
        del ref, qc, kc, vc
        torch.cuda.empty_cache()

    base = sum(acc["plain"]) / len(acc["plain"])
    for label, _, _ in ARMS:
        m = sum(acc[label]) / len(acc[label])
        delta = 100.0 * (m - base) / base
        print(f"  {label:14s} mean_rtol {m:.4f}  ({delta:+.1f}% vs plain)")
    del q, k, v, d
    torch.cuda.empty_cache()


def main():
    global ARMS
    args = [a for a in sys.argv[1:] if a != "--sweep"]
    if "--sweep" in sys.argv:
        ARMS = SWEEP_ARMS
    if not args:
        print(__doc__)
        return 1
    sys.argv = [sys.argv[0]] + args
    info = sageattention.build_info()
    print(f"sage {info['describe']}  {torch.cuda.get_device_name()}")
    print("reference: fp32 EFFICIENT_ATTENTION on the original q/k/v; sage fp8++ fp32+fp16 smooth_k=False")
    for path in sys.argv[1:]:
        run(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
