"""Correctness + perf gate: sage_ffn vs torch reference at LTX FFN shapes.

Asserts mean_rtol < 0.10 against the torch fp8-dequant reference path
at both LTX stage-1 (T=10780) and stage-2 (T=44880 multi-guide
expanded) FFN shapes (hidden=4096, inner=16384). Also reports speed
via median-of-5 timed runs after autotune-absorbing warmup.

Run with:
    ${VIRTUAL_ENV}/bin/python tests/spikes/test_fused_mlp_fp8_correctness.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from sageattention.triton.fused_mlp_fp8 import FP8_E4M3_MAX, sage_ffn

# Reuse the symmetric-denominator rtol helper used by every other
# accuracy bench in this repo. Keeps tolerance budgets comparable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]


def quantize_weight_per_tensor_fp8(w_f32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor fp8 quant of a weight matrix. Returns (w_fp8, scalar_scale)."""
    w_max = w_f32.abs().amax().item()
    if w_max == 0:
        return w_f32.to(torch.float8_e4m3fn), 1.0
    scale = w_max / FP8_E4M3_MAX
    w_scaled = (w_f32 / w_max * FP8_E4M3_MAX).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return w_scaled.to(torch.float8_e4m3fn), scale


def run_correctness_at(T: int, hidden: int = 4096, inner: int = 16384, seed: int = 0, with_bias: bool = True) -> dict:
    bias_tag = "+bias" if with_bias else "no-bias"
    print(f"\n=== Correctness: T={T} hidden={hidden} inner={inner} ({bias_tag}) ===")
    device = torch.device("cuda")
    torch.manual_seed(seed)

    x = torch.randn(1, T, hidden, dtype=torch.bfloat16, device=device)

    # Realistic init: small std for weights, like a trained DiT block.
    w1_f32 = torch.randn(inner, hidden, dtype=torch.float32, device=device) * (1.0 / (hidden ** 0.5))
    w2_f32 = torch.randn(hidden, inner, dtype=torch.float32, device=device) * (1.0 / (inner ** 0.5))

    w1_fp8, w1_scale = quantize_weight_per_tensor_fp8(w1_f32)
    w2_fp8, w2_scale = quantize_weight_per_tensor_fp8(w2_f32)

    w1_bf16_ref = (w1_fp8.float() * w1_scale).to(torch.bfloat16)
    w2_bf16_ref = (w2_fp8.float() * w2_scale).to(torch.bfloat16)

    # LTX-style bf16 biases on both Linear layers (matches the distilled
    # checkpoint, which carries bf16 biases on both ff.net.0.proj and ff.net.2).
    if with_bias:
        b1 = torch.randn(inner, dtype=torch.bfloat16, device=device) * (1.0 / (inner ** 0.5))
        b2 = torch.randn(hidden, dtype=torch.bfloat16, device=device) * (1.0 / (hidden ** 0.5))
    else:
        b1 = None
        b2 = None

    def torch_ref():
        return F.linear(F.gelu(F.linear(x, w1_bf16_ref, bias=b1), approximate="tanh"), w2_bf16_ref, bias=b2)

    # Warmup (absorbs autotune-search cost + Triton lazy JIT)
    for _ in range(2):
        _ = torch_ref()
        _ = sage_ffn(x, w1_fp8, w1_scale, w2_fp8, w2_scale, b1=b1, b2=b2)
    torch.cuda.synchronize()

    ref_samples = []
    sage_samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref = torch_ref()
        torch.cuda.synchronize()
        ref_samples.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sage_ffn(x, w1_fp8, w1_scale, w2_fp8, w2_scale, b1=b1, b2=b2)
        torch.cuda.synchronize()
        sage_samples.append((time.perf_counter() - t0) * 1000)
    ref_samples.sort()
    sage_samples.sort()
    ref_ms = ref_samples[2]
    sage_ms = sage_samples[2]

    mean_rtol, max_rtol, mean_atol, max_atol = accuracy_metrics(out, ref)

    print(f"  shape out: {tuple(out.shape)}")
    print(f"  mean_rtol={mean_rtol:.4f}  max_rtol={max_rtol:.4f}")
    print(f"  mean_atol={mean_atol:.6f}  max_atol={max_atol:.6f}")
    print(f"  ref (torch ref)  : {ref_ms:.2f} ms")
    print(f"  sage_ffn (Triton): {sage_ms:.2f} ms  ({ref_ms/sage_ms:.2f}x vs ref)")

    return {
        "T": T, "with_bias": with_bias,
        "mean_rtol": mean_rtol, "max_rtol": max_rtol,
        "mean_atol": mean_atol, "max_atol": max_atol,
        "ref_ms": ref_ms, "sage_ms": sage_ms,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"torch: {torch.__version__}  triton: {triton.__version__}  device: {torch.cuda.get_device_name(0)}")

    results = []
    # Bias-inclusive matches LTX 2.3 usage; bias-free path is the v0.6.0
    # initial bench config, kept as a sanity check that the HAS_BIAS=False
    # constexpr branch still works.
    for T in (10780, 44880):
        results.append(run_correctness_at(T, with_bias=True))
    for T in (10780, 44880):
        results.append(run_correctness_at(T, with_bias=False))

    print("\n=== Verdict ===")
    rtol_budget = 0.10
    all_ok = all(r["mean_rtol"] < rtol_budget for r in results)
    for r in results:
        status = "PASS" if r["mean_rtol"] < rtol_budget else "FAIL"
        tag = "+bias" if r["with_bias"] else "no-bias"
        print(f"  [{status}] T={r['T']:>6} ({tag:7})  mean_rtol={r['mean_rtol']:.4f} (<{rtol_budget})")

    if all_ok:
        print("\nKernel correctness PASS at LTX FFN shapes (bias-inclusive and bias-free).")
        return 0
    print("\nFAIL: rtol budget exceeded at one or more shapes. Debug numerics before perf tuning.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
