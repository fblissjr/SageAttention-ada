#!/usr/bin/env python3
"""Pin `qk_rotate`, the per-thread quantizer's fused Hadamard rotation.

Claims, i.e. what breaks if a case is deleted:
  - off_is_untouched       : `qk_rotate=False` is the pre-change path, bit for
                             bit, in both layouts. Delete and the option can
                             start costing every caller who never asked.
  - matrix_is_orthogonal   : R @ R.T is the identity and its signs are the four
                             words comfy-kitchen's kernels use. Delete and a
                             changed word keeps every other case green while
                             the three kernels stop rotating alike.
  - kernel_is_the_matrix   : the fused kernel quantizes x @ R: its codes and
                             scales match the plain quantizer run on inputs
                             rotated in fp32 by the oracle matrix, to rounding.
                             Delete and a wrong butterfly stage or a dropped
                             sign still "rotates" and still lowers some error.
  - ragged_tail_is_zeroed  : a length that does not fill the last scale group
                             gives the same codes on the real rows as the same
                             rows inside a longer sequence would. Delete and
                             garbage past L can set the last group's scale.
  - scores_are_preserved   : dequantized (qR).(kR) tracks q.k. Delete and the
                             q and k sides could use different matrices.
  - spikes_improve         : on a tensor with loud K channels AND per-token
                             spikes the fp8++ kernel's error against fp32
                             attention falls with the option on, and falls
                             further than with `qk_balance`. Synthetic is
                             correct here: a fidelity property of the
                             arithmetic, not an accuracy claim about a model
                             (docs/testing_practices.md); the accuracy record
                             is on captured H3 activations in CHANGELOG.md.
  - refused_where_absent   : head_dim != 128 and qk_quant_gran="per_warp" raise
                             rather than run unrotated.

Standalone script (no pytest); needs a CUDA device (Triton kernels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sageattention  # noqa: E402
from sageattention.triton.quant_per_thread import (  # noqa: E402
    _ROT_SIGN_WORDS, per_thread_int8, qk_rotation_matrix)

DEV = "cuda"


def _qk(b=1, h=4, n=1000, d=128, layout="HND", seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    shape = (b, h, n, d) if layout == "HND" else (b, n, h, d)
    return (torch.randn(shape, generator=g, device=DEV, dtype=dtype),
            torch.randn(shape, generator=g, device=DEV, dtype=dtype))


def off_is_untouched():
    for layout in ("HND", "NHD"):
        q, k = _qk(layout=layout)
        a = per_thread_int8(q, k, tensor_layout=layout)
        b = per_thread_int8(q, k, tensor_layout=layout, qk_rotate=False)
        assert all(torch.equal(x, y) for x, y in zip(a, b)), f"{layout}: off path changed"


def matrix_is_orthogonal():
    r = qk_rotation_matrix("cpu", torch.float64)
    assert (r @ r.T - torch.eye(128, dtype=torch.float64)).abs().max() < 1e-12
    assert _ROT_SIGN_WORDS == (0x1035997B, 0x8087F5EE, 0xEE2E4E1A, 0x71132418)
    # row d of R is sign[d] * H[d]; H's first column is all ones
    signs = r[:, 0] * 128 ** 0.5
    want = torch.tensor([1.0 if (_ROT_SIGN_WORDS[d >> 5] >> (d & 31)) & 1 else -1.0 for d in range(128)],
                        dtype=torch.float64)
    assert torch.equal(signs.round(), want)


def _oracle(x, r64):
    # The oracle rotates in float64. An fp32 cuBLAS matmul is NOT a usable
    # oracle here: on this stack it lands ~2e-4 relative from the float64
    # product, a thousand times further than the kernel does, and an early
    # version of this test failed on the oracle rather than on the kernel.
    return (x.double() @ r64).float()


def kernel_is_the_matrix():
    r64 = qk_rotation_matrix(DEV, torch.float64)
    for layout in ("HND", "NHD"):
        q, k = _qk(layout=layout, dtype=torch.float32, seed=3)
        fused = per_thread_int8(q, k, tensor_layout=layout, qk_rotate=True)
        oracle = per_thread_int8(_oracle(q, r64), _oracle(k, r64), tensor_layout=layout)
        for name, a, b in (("q_scale", fused[1], oracle[1]), ("k_scale", fused[3], oracle[3])):
            assert torch.allclose(a, b, rtol=2e-6, atol=1e-7), f"{layout} {name} differs"
        for name, a, b in (("q", fused[0], oracle[0]), ("k", fused[2], oracle[2])):
            diff = (a.int() - b.int()).abs()
            # same values to fp32 rounding, so a code may differ by one at a tie
            assert diff.max() <= 1, f"{layout} {name}: codes differ by {int(diff.max())}"
            assert (diff != 0).float().mean() < 1e-4, f"{layout} {name}: {float((diff != 0).float().mean()):.4f} of codes differ"


def ragged_tail_is_zeroed():
    # 1000 is not a multiple of any group size; the last groups are partial.
    q, k = _qk(n=1000, dtype=torch.float32, seed=5)
    big_q = torch.cat([q, torch.zeros_like(q[:, :, :24])], dim=2)
    big_k = torch.cat([k, torch.zeros_like(k[:, :, :24])], dim=2)
    short = per_thread_int8(q, k, qk_rotate=True)
    long = per_thread_int8(big_q, big_k, qk_rotate=True)
    assert torch.equal(short[0], long[0][:, :, :1000]) and torch.equal(short[2], long[2][:, :, :1000]), \
        "rows past L leaked into the last group's scale"


def scores_are_preserved():
    q, k = _qk(h=2, n=512, dtype=torch.float32, seed=7)
    # The identity the option rests on, in float64 so the check is about the
    # matrix and not about a matmul: (qR)(kR)^T == q k^T.
    r64 = qk_rotation_matrix(DEV, torch.float64)
    exact = q.double() @ k.double().transpose(-1, -2)
    rotated = (q.double() @ r64) @ (k.double() @ r64).transpose(-1, -2)
    assert (exact - rotated).abs().max() < 1e-9, "R is not orthogonal, or not shared by q and k"
    # and the fused kernel applies that same R to both sides
    _q8, qs, _k8, ks = per_thread_int8(q, k, qk_rotate=True)
    _o8, os_, _ok8, oks = per_thread_int8(_oracle(q, r64), _oracle(k, r64))
    assert torch.allclose(qs, os_, rtol=2e-6) and torch.allclose(ks, oks, rtol=2e-6)


def _err(q, k, v, **kw):
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = sageattention.sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="HND", is_causal=False,
                                                       pv_accum_dtype="fp32+fp16", smooth_k=False, **kw)
    return ((out.float() - ref).norm() / ref.norm()).item()


def spikes_improve():
    g = torch.Generator(device=DEV).manual_seed(11)
    b, h, n, d = 1, 4, 4096, 128
    q = torch.randn(b, h, n, d, generator=g, device=DEV)
    k = torch.randn(b, h, n, d, generator=g, device=DEV)
    v = torch.randn(b, h, n, d, generator=g, device=DEV)
    k[..., 19] *= 25.0; k[..., 67] *= 25.0                 # loud channels, as H3's last block has
    rows = torch.arange(0, n, 16, device=DEV)
    k[:, :, rows, (rows // 16) % d] += 40.0                # one spiking token per scale group
    q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    plain, bal, rot = _err(q, k, v), _err(q, k, v, qk_balance=True), _err(q, k, v, qk_rotate=True)
    assert rot < plain * 0.8, f"rotation did not help: plain {plain:.4f} rotated {rot:.4f}"
    assert rot < bal, f"rotation did not beat the balance on spikes: balance {bal:.4f} rotated {rot:.4f}"
    print(f"    spikes: plain {plain:.4f}  qk_balance {bal:.4f}  qk_rotate {rot:.4f}")


def refused_where_absent():
    q, k = _qk(d=64)
    try:
        per_thread_int8(q, k, qk_rotate=True)
    except ValueError:
        pass
    else:
        raise AssertionError("head_dim 64 was accepted")
    q, k = _qk(n=256)
    v = torch.randn_like(q)
    try:
        sageattention.sageattn_qk_int8_pv_fp8_cuda(q, k, v, qk_quant_gran="per_warp", qk_rotate=True)
    except ValueError:
        pass
    else:
        raise AssertionError("per_warp accepted qk_rotate")


CASES = [off_is_untouched, matrix_is_orthogonal, kernel_is_the_matrix, ragged_tail_is_zeroed,
         scores_are_preserved, spikes_improve, refused_where_absent]


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0
    failed = 0
    for case in CASES:
        try:
            case()
            print(f"  ok    {case.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {case.__name__}: {exc}")
    print("all cases passed" if not failed else f"{failed} case(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
