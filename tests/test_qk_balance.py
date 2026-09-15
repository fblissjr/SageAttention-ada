#!/usr/bin/env python3
"""Pin `qk_balance`, the per-thread quantizer's channel rebalancing.

Claims, i.e. what breaks if a case is deleted:
  - off_is_untouched      : `qk_balance=False` is the pre-change path, bit
                            for bit. Delete and the option can start
                            costing every caller who never asked for it.
  - gate_closed_is_plain  : with the gate closed on every head (a share
                            threshold nothing reaches) the balanced path
                            produces the same int8 codes and scales as the
                            plain one, in both layouts. Delete and a factor
                            of one can stop meaning "no change".
  - gate_is_per_head      : a flat head keeps a factor of one while a loud
                            head in the same tensor is balanced. Delete and
                            the gate degrades to all-or-nothing, which
                            measured as a few percent worse on flat blocks.
  - factor_is_exact       : the query and key factors multiply to one per
                            channel, which is the identity the attention
                            math rests on. Delete and a reciprocal slips.
  - gqa_maps_heads        : under grouped-query attention the factor is per
                            kv head and every query head in a group reads
                            its group's factor. Delete and a GQA model
                            (Krea2's 48/12) silently mixes factors.
  - loud_channels_improve : on a synthetic tensor with two loud K channels
                            the fp8++ kernel's error against fp32 attention
                            falls with the option on. Delete and the
                            mechanism could be inverted while every other
                            case stays green. Synthetic is correct here:
                            this is a fidelity property of the arithmetic,
                            not an accuracy claim about a model
                            (docs/testing_practices.md); the accuracy
                            record is on captured H3 activations in
                            CHANGELOG.md.

Standalone script (no pytest); needs a CUDA device (Triton kernels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sageattention.triton.quant_per_thread import per_thread_int8, qk_balance_factor  # noqa: E402

D = 128


def _qk(layout, h_q=8, h_kv=8, s=4096, loud=(82, 19), seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (1, h_q, s, D) if layout == "HND" else (1, s, h_q, D)
    shape_k = (1, h_kv, s, D) if layout == "HND" else (1, s, h_kv, D)
    q = torch.randn(shape_q, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(shape_k, dtype=torch.bfloat16, device="cuda", generator=g)
    for c in loud:
        k[..., c] *= 12
    return q, k


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_off_is_untouched():
    for layout in ("HND", "NHD"):
        q, k = _qk(layout)
        a = per_thread_int8(q, k, tensor_layout=layout)
        b = per_thread_int8(q, k, tensor_layout=layout, qk_balance=False)
        assert _equal(a, b), f"{layout}: qk_balance=False must be the plain path"


def test_gate_closed_is_plain():
    for layout in ("HND", "NHD"):
        q, k = _qk(layout)
        a = per_thread_int8(q, k, tensor_layout=layout)
        b = per_thread_int8(q, k, tensor_layout=layout, qk_balance=True, balance_min_share=1.01)
        assert _equal(a, b), f"{layout}: a closed gate must reproduce the plain codes and scales"
        c = per_thread_int8(q, k, tensor_layout=layout, qk_balance=True)
        assert not torch.equal(a[2], c[2]), f"{layout}: an open gate must change the key codes"


def test_gate_is_per_head():
    q, k = _qk("HND")
    k[:, 0] = torch.randn_like(k[:, 0])          # head 0 flat, the rest loud
    fq, _ = qk_balance_factor(q, k, "HND")
    assert bool((fq[0, 0] == 1).all()), "flat head must keep a factor of one"
    assert bool((fq[0, 1] != 1).any()), "loud head must be balanced"


def test_factor_is_exact():
    for layout in ("HND", "NHD"):
        q, k = _qk(layout)
        fq, fk = qk_balance_factor(q, k, layout)
        assert tuple(fq.shape) == (1, 8, D) and fq.dtype == torch.float32
        assert (fq * fk - 1).abs().max().item() < 1e-6, "q and k factors must be reciprocals"
        assert torch.log(fq).mean(dim=-1).abs().max().item() < 1e-5, "geometric mean must be one per head"


def test_gqa_maps_heads():
    q, k = _qk("HND", h_q=16, h_kv=8)
    # Make each group's two query heads identical, so the group-mean rq the
    # GQA path uses equals the single head's rq and the factors coincide.
    # Then the grouped call and a one-head call through the same kernel
    # must agree bit for bit; a mis-mapped GROUP index would not.
    q[:, 1::2] = q[:, 0::2]
    fq, _ = qk_balance_factor(q, k, "HND")
    assert tuple(fq.shape) == (1, 8, D), "factor is per kv head"
    q_int8 = per_thread_int8(q, k, tensor_layout="HND", qk_balance=True)[0]
    for g in range(8):
        single = per_thread_int8(q[:, 2 * g:2 * g + 1], k[:, g:g + 1], tensor_layout="HND", qk_balance=True)[0]
        for hq in (2 * g, 2 * g + 1):
            assert torch.equal(q_int8[:, hq:hq + 1], single), f"query head {hq} must use kv head {g}'s factor"


def test_loud_channels_improve():
    import sageattention
    torch.manual_seed(0)
    q = torch.randn(1, 8, 8192, D, dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    k[..., 82] *= 12
    k[..., 19] *= 10
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = {}
    for bal in (False, True):
        o = sageattention.sageattn_qk_int8_pv_fp8_cuda(
            q, k, v, tensor_layout="HND", pv_accum_dtype="fp32+fp16", smooth_k=False, qk_balance=bal).float()
        err[bal] = ((o - ref).abs() / (ref.abs() + 1e-3)).mean().item()
    assert err[True] < 0.85 * err[False], f"balanced {err[True]:.4f} must beat plain {err[False]:.4f} by a clear margin"


CASES = [test_off_is_untouched, test_gate_closed_is_plain, test_gate_is_per_head,
         test_factor_is_exact, test_gqa_maps_heads, test_loud_channels_improve]


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 1
    failures = 0
    for fn in CASES:
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
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
