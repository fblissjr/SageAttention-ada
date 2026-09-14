"""Where does MiniMax H3's block-49 quantization error come from? CPU, no kernel.

Follow-up to `spike_h3_k_channel_balance.py`, which showed that rebalancing
K's channels removes about a fifth of sage's INT8 error at block 49 and
nothing elsewhere. That names one cause. Balanced block 49 still sits at
several times block 0, so the rest needs a home before anything claims to
explain the block.

This simulates sage's quantization steps one at a time in fp32 on the
captured q/k/v, against exact fp32 attention on the same inputs, so each
arm's error is attributable to one rounding:

  K int8      one scale per 64-token block per head across all 128
              channels, sage's K scheme (`per_thread_int8`, BLKK=64).
  Q int8      one scale per token per head. Sage's per-thread scheme is a
              little coarser than that; this is a floor on Q's share.
  QK int8     both, the whole INT8 side.
  QK int8 bal both, after the pair-equal a=0.5 channel balancing.
  QK int8 bal-shared
              as above but with one factor per channel shared across heads,
              which is what folds into the per-channel norm weights.
  QK int8 bal-weights
              the factor taken from the checkpoint's own q_norm/k_norm
              weights (|kw|^a / |qw|^(1-a), pair-equal), no capture needed.
              Requires --norm-weights <unet safetensors>.

The fp8 P.V side is deliberately NOT simulated here: a first version cast
the un-normalized exp to e4m3 and overstated that error several-fold
against the real kernel, because the kernel scales P before the cast in a
way this file does not reproduce. The fp8-vs-fp16 split comes from the
kernel records instead (`spike_h3_real_activations.py`, fp8++ against the
fp16 kernel on the same cell).

Attention is exact fp32 over ALL keys for a SAMPLE of query rows, drawn per
packed segment (text, audio, video) so the error can be split by which
queries carry it. Sampling queries loses nothing about the keys, which is
where the K rounding lives.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_block49_error_anatomy.py \
        --text 395 --audio 1150 <capture.pt> ...

`--text/--audio` are the packed row counts from the capture manifest
(`token_accounting`); the video rows are the remainder. Reference rows per
segment default to 256/256/512.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from test_sageattn_ltx_shapes import accuracy_metrics

ROT, HALF = 96, 48
BLK_K = 64


def q_int8_per_token(x):
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


def k_int8_per_block(x):
    S, D = x.shape
    pad = (-S) % BLK_K
    xp = torch.nn.functional.pad(x, (0, 0, 0, pad)).reshape(-1, BLK_K, D)
    s = xp.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-7) / 127.0
    out = torch.round(xp / s).clamp(-127, 127) * s
    return out.reshape(-1, D)[:S]


def to_e4m3(x):
    return x.to(torch.float8_e4m3fn).float()


def v_fp8_per_channel(x, scale_max=2.25):
    s = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-7) / scale_max
    return to_e4m3(x / s) * s


def balance_factor(k, q, alpha=0.5):
    rk = k.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    rq = q.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    s = rk.pow(alpha) / rq.pow(1 - alpha)
    g = (s[:HALF] * s[HALF:ROT]).sqrt()
    s = torch.cat([g, g, s[ROT:]])
    return s / torch.exp(torch.log(s).mean())


def balance_factor_shared(k, q, alpha=0.5):
    """[H, S, D] -> [D]: one factor per channel across heads, pair-equal."""
    rk = k.float().pow(2).mean(dim=(0, 1)).sqrt().clamp(min=1e-6)
    rq = q.float().pow(2).mean(dim=(0, 1)).sqrt().clamp(min=1e-6)
    s = rk.pow(alpha) / rq.pow(1 - alpha)
    g = (s[:HALF] * s[HALF:ROT]).sqrt()
    s = torch.cat([g, g, s[ROT:]])
    return s / torch.exp(torch.log(s).mean())


def balance_factor_weights(kw, qw, alpha=0.5):
    """[D] norm weights -> [D] factor, pair-equal, geometric mean one."""
    s = kw.abs().clamp(min=1e-6).pow(alpha) / qw.abs().clamp(min=1e-6).pow(1 - alpha)
    g = (s[:HALF] * s[HALF:ROT]).sqrt()
    s = torch.cat([g, g, s[ROT:]])
    return s / torch.exp(torch.log(s).mean())


def attention(q, k, v, p_fp8=False, stats=None):
    scores = (q @ k.T) * (q.shape[-1] ** -0.5)
    m = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - m)                      # in (0, 1], max entry exactly 1
    l = e.sum(dim=-1, keepdim=True)                # fp32 row sum, as the kernel keeps it
    if stats is not None:
        p = e / l
        stats["row_max_p"].append(p.amax(dim=-1))
        stats["eff_keys"].append(torch.exp(-(p * torch.log(p.clamp(min=1e-30))).sum(dim=-1)))
        stats["logit_range"].append((m - scores.mean(dim=-1, keepdim=True)).squeeze(-1))
    if p_fp8:
        # The kernel casts the un-normalized exp to e4m3 and divides by the
        # fp32 row sum afterwards; entries below e4m3's smallest subnormal
        # (~2e-3 of the row max) vanish.
        e = to_e4m3(e)
    return (e @ v) / l


ARMS = ["K int8", "Q int8", "QK int8", "QK int8 bal", "QK int8 bal-shared", "QK int8 bal-weights"]


def run(path, seg_rows, n_ref, heads, norm_weights=None):
    d = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    q_all, k_all, v_all = d["q"], d["k"], d["v"]
    _, H, S, D = q_all.shape
    import re
    block = int(re.search(r"_b(\d+)_", Path(path).name).group(1))
    # Head-shared factor from a head subsample: the full [H, S, D] in fp32
    # is 3 GB per tensor; eight heads is plenty for a channel rms.
    sub = list(range(0, H, max(1, H // 8)))[:8]
    s_shared = balance_factor_shared(k_all[0, sub], q_all[0, sub])
    s_weights = None
    if norm_weights is not None:
        from safetensors import safe_open
        with safe_open(norm_weights, framework="pt", device="cpu") as sf:
            kw = sf.get_tensor(f"blocks.{block}.attn.k_norm.weight").float()
            qw = sf.get_tensor(f"blocks.{block}.attn.q_norm.weight").float()
        s_weights = balance_factor_weights(kw, qw)
    text, audio = seg_rows
    segs = {"text": (0, text), "audio": (text, text + audio), "video": (text + audio, S)}
    g = torch.Generator().manual_seed(0)
    rows = {}
    for name, (a, b) in segs.items():
        n = min(n_ref[name], b - a)
        rows[name] = (a + torch.randperm(b - a, generator=g)[:n]).sort().values
    idx = torch.cat(list(rows.values()))
    bounds, off = {}, 0
    for name in segs:
        bounds[name] = (off, off + len(rows[name]))
        off += len(rows[name])

    print(f"\n=== {Path(path).name}  S={S} heads={H} head_dim={D}  segments text {segs['text']} audio {segs['audio']} video {segs['video']}")
    print(f"query rows sampled: " + ", ".join(f"{n} {len(r)}" for n, r in rows.items()) + f"; heads {heads[0]}..{heads[-1]}")

    acc = {a: {s: [] for s in list(segs) + ["all"]} for a in ARMS}
    per_head_k = []
    stats = {"row_max_p": [], "eff_keys": [], "logit_range": []}
    t0 = time.time()
    for h in heads:
        q = q_all[0, h].float()
        k = k_all[0, h].float()
        v = v_all[0, h].float()
        qs = q[idx]
        ref = attention(qs, k, v, stats=stats)
        s = balance_factor(k, q)
        arms = {
            "K int8": (qs, k_int8_per_block(k), v, False),
            "Q int8": (q_int8_per_token(qs), k, v, False),
            "QK int8": (q_int8_per_token(qs), k_int8_per_block(k), v, False),
            "QK int8 bal": (q_int8_per_token(qs * s) / s, k_int8_per_block(k / s) * s, v, False),
            "QK int8 bal-shared": (q_int8_per_token(qs * s_shared) / s_shared, k_int8_per_block(k / s_shared) * s_shared, v, False),
        }
        if s_weights is not None:
            arms["QK int8 bal-weights"] = (q_int8_per_token(qs * s_weights) / s_weights, k_int8_per_block(k / s_weights) * s_weights, v, False)
        for name, (qa, ka, va, pf) in arms.items():
            out = attention(qa, ka, va, p_fp8=pf)
            acc[name]["all"].append(accuracy_metrics(out, ref)[0])
            for seg, (a, b) in bounds.items():
                acc[name][seg].append(accuracy_metrics(out[a:b], ref[a:b])[0])
        per_head_k.append(acc["K int8"]["all"][-1])
        del q, k, v, qs, ref, arms
    print(f"({time.time() - t0:.0f}s)")

    print(f"{'arm':14s} {'all':>8} {'text':>8} {'audio':>8} {'video':>8}")
    for name in ARMS:
        if not acc[name]["all"]:
            continue
        m = {s: sum(acc[name][s]) / len(acc[name][s]) for s in acc[name]}
        print(f"{name:14s} {m['all']:>8.4f} {m['text']:>8.4f} {m['audio']:>8.4f} {m['video']:>8.4f}")
    ph = torch.tensor(per_head_k)
    worst = torch.topk(ph, 4)
    print(f"K int8 per head: median {ph.median():.4f}, worst {[(heads[i], round(ph[i].item(), 4)) for i in worst.indices.tolist()]}")
    st = {k: torch.stack(v) for k, v in stats.items()}          # [heads, rows]
    print("attention shape of the exact reference (median over heads and rows, per query segment):")
    print(f"{'':14s} {'row max p':>10} {'eff. keys':>10} {'logit range':>12}")
    for seg, (a, b) in bounds.items():
        print(f"{seg:14s} {st['row_max_p'][:, a:b].median():>10.4f} {st['eff_keys'][:, a:b].median():>10.0f} {st['logit_range'][:, a:b].median():>12.2f}")
    print(f"{'worst-4 heads':14s} {st['row_max_p'][worst.indices].median():>10.4f} {st['eff_keys'][worst.indices].median():>10.0f} {st['logit_range'][worst.indices].median():>12.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--text", type=int, required=True)
    ap.add_argument("--audio", type=int, required=True)
    ap.add_argument("--rows", type=int, nargs=3, default=(256, 256, 512), metavar=("TEXT", "AUDIO", "VIDEO"))
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--norm-weights", default=None, help="unet safetensors, for the weights-only factor arm")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    n_ref = dict(zip(("text", "audio", "video"), args.rows))
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, fp32 on CPU; reference: exact attention on the sampled rows")
    for p in args.captures:
        run(p, (args.text, args.audio), n_ref, list(range(args.heads)), args.norm_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
