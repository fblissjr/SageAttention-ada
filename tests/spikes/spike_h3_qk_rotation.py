"""Would rotating q/k before sage's INT8 quantizer help on MiniMax H3, and does it replace qk_balance?

Sage's per-thread quantizer gives a group of rows one scale across all 128
head channels, so a loud channel (H3's last blocks) or a single-token spike
spends the INT8 range and starves the rest. `qk_balance` (v0.7.19) answers the
loud-channel half by rescaling channels per head. A fixed orthogonal rotation
answers both halves without detecting anything: multiply every q row and every
k row by the same R (random sign diagonal, then a normalised Hadamard), which
leaves every q.k score unchanged in exact arithmetic and spreads each row's
energy over all channels before rounding. It is what rotated-INT8 kernels do
internally, and what the consumer-side sparse kernel gained on 2026-09-15.

This spike needs NO kernel change: it rotates the captured inputs in PyTorch
and calls the kernel as it ships. That makes it the design input for a CUDA
version (is rotation worth building, alone or with the balance) and later its
oracle. What it cannot say is the cost: a matmul here is not the butterfly a
quantizer would run.

The reference is fp32 SDPA on the UNROTATED inputs for every arm, so the
rotated arms also pay for re-rounding the rotated tensors to bf16, as a kernel
that rotates before quantizing would not. That biases AGAINST rotation,
slightly; `rerounding` reports its size.

    $VIRTUAL_ENV/bin/python tests/spikes/spike_h3_qk_rotation.py <capture.pt> ... [--json out.json]

Real captured q/k/v only (HND). Synthetic input has no loud channels and no
attention structure and would report nothing.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention

CHUNK = 8
SEED = 0x51A6E


def _hadamard(n: int) -> torch.Tensor:
    assert n & (n - 1) == 0, "Sylvester construction needs a power of two"
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def rotation(dim: int, device, block: int | None = None, permute: bool = False) -> torch.Tensor:
    """x @ R with R = P @ diag(signs) @ blockdiag(H_block): orthogonal, fixed by SEED.

    `block=None` is one Hadamard over the whole head dim, log2(dim) butterfly
    stages in a kernel. A smaller block is cheaper (log2(block) stages, fits in
    registers) but mixes only within a block, so whether it helps depends on
    which channels share one; `permute` spreads neighbours apart first with a
    fixed stride, the cheap form of what DuQuant's zigzag permutation is for.
    """
    block = block or dim
    assert dim % block == 0
    r = torch.block_diag(*[_hadamard(block)] * (dim // block))
    g = torch.Generator().manual_seed(SEED)
    signs = torch.randint(0, 2, (dim,), generator=g).float() * 2 - 1
    r = signs[:, None] * r
    if permute:
        perm = (torch.arange(dim) * 37) % dim      # 37 is coprime with any power of two
        r = r[perm]
    return r.to(device)


def rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@torch.inference_mode()
def run(path):
    d = torch.load(path, map_location="cuda", weights_only=True)
    q, k, v = d["q"], d["k"], d["v"]
    _, H, S, D = q.shape
    rots = {"full": rotation(D, q.device),
            "block32": rotation(D, q.device, block=32),
            "block32+perm": rotation(D, q.device, block=32, permute=True)}
    ortho = max((R @ R.T - torch.eye(D, device=q.device)).abs().max().item() for R in rots.values())
    # arm -> (rotation name or None, qk_balance)
    arms = {"plain": (None, False), "qk_balance": (None, True),
            "rotated": ("full", False), "rotated+qk_balance": ("full", True),
            "rotated block32": ("block32", False), "rotated block32+perm": ("block32+perm", False)}
    acc = {a: [] for a in arms}
    reround = []
    for h0 in range(0, H, CHUNK):
        sl = slice(h0, min(h0 + CHUNK, H))
        qc, kc, vc = (x[:, sl].contiguous() for x in (q, k, v))
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            ref = F.scaled_dot_product_attention(qc.float(), kc.float(), vc.float())
        rotated = {n: ((qc.float() @ R).to(qc.dtype), (kc.float() @ R).to(kc.dtype)) for n, R in rots.items()}
        qr, kr = rotated["full"]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            reround.append(rel_l2(F.scaled_dot_product_attention(qr.float(), kr.float(), vc.float()), ref))
        for name, (rot, bal) in arms.items():
            qa, ka = rotated[rot] if rot else (qc, kc)
            out = sageattention.sageattn_qk_int8_pv_fp8_cuda(
                qa, ka, vc, tensor_layout="HND", is_causal=False,
                pv_accum_dtype="fp32+fp16", smooth_k=False, **({"qk_balance": True} if bal else {}))
            acc[name].append(rel_l2(out, ref))
            del out
        del ref, rotated, qr, kr, qc, kc, vc
        torch.cuda.empty_cache()
    row = {a: sum(x) / len(x) for a, x in acc.items()}
    row["rerounding"] = sum(reround) / len(reround)
    m = re.search(r"_b(\d+)_s(\d+)", Path(path).name)
    cell = f"b{m.group(1)}_s{m.group(2)}" if m else Path(path).stem
    base = row["plain"]
    print(f"{cell}  S={S} heads={H}  |RR^T-I| {ortho:.1e}  rerounding {row['rerounding']:.4f}")
    for a in arms:
        print(f"   {a:20s} rel_l2 {row[a]:.4f}  ({100 * (row[a] - base) / base:+.1f}% vs plain)")
    return cell, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()
    print(f"model: MiniMax H3 (captured q/k/v)   sageattention {getattr(sageattention, '__version__', '?')}   "
          f"kernel: fp8++ (qk int8 per-thread, pv fp8), smooth_k off")
    cells = dict(run(p) for p in args.captures)
    if args.json_out:
        import orjson
        Path(args.json_out).write_bytes(orjson.dumps({
            "what": "sage fp8++ on H3 captures: plain, qk_balance, q/k rotated in PyTorch before the call, both; rel L2 vs fp32 SDPA on the unrotated inputs",
            "model": "MiniMax H3", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "rotation": {"form": "x @ (P @ diag(random signs) @ blockdiag(Sylvester-Hadamard)); full = one block of head_dim, block32 = four of 32, +perm = stride-37 channel permutation first", "seed": SEED},
            "cells": cells}, option=orjson.OPT_INDENT_2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
