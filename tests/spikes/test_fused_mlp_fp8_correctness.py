"""Day-2 correctness check: sage_ffn vs torch reference at LTX FFN shapes.

This is the day-4 decision-gate test, run early to catch design issues.
Per the v0.6 plan, mean_rtol must be < 0.10 vs torch reference (which
is itself running on bf16-dequantized weights, so fp8 quant noise is
acceptable as long as we're at the same floor as the v0.5.5 attention
path).

Shapes tested:
- LTX stage-1 FFN: hidden=4096, inner=16384, T=10780, bf16 act
- LTX stage-2 FFN: hidden=4096, inner=16384, T=44880, bf16 act (the
  multi-guide-expanded shape where the memory win materializes)

Run with:
    ${VIRTUAL_ENV}/bin/python tests/spikes/test_fused_mlp_fp8_correctness.py
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F
import triton

from sageattention.triton.fused_mlp_fp8 import sage_ffn

FP8_E4M3_MAX = 448.0


def quantize_weight_per_tensor_fp8(w_f32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor fp8 quant of a weight matrix. Returns (w_fp8, scalar_scale)."""
    w_max = w_f32.abs().amax().item()
    if w_max == 0:
        return w_f32.to(torch.float8_e4m3fn), 1.0
    scale = w_max / FP8_E4M3_MAX
    w_scaled = (w_f32 / w_max * FP8_E4M3_MAX).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return w_scaled.to(torch.float8_e4m3fn), scale


def run_correctness_at(T: int, hidden: int = 4096, inner: int = 16384, seed: int = 0) -> dict:
    print(f"\n=== Correctness: T={T} hidden={hidden} inner={inner} ===")
    device = torch.device("cuda")
    torch.manual_seed(seed)

    # bf16 input activation
    x = torch.randn(1, T, hidden, dtype=torch.bfloat16, device=device)

    # Build LTX-style fp8 weights with per-tensor scalar scale.
    # Realistic init: small std for weights, like a trained DiT block.
    w1_f32 = torch.randn(inner, hidden, dtype=torch.float32, device=device) * (1.0 / (hidden ** 0.5))
    w2_f32 = torch.randn(hidden, inner, dtype=torch.float32, device=device) * (1.0 / (inner ** 0.5))

    w1_fp8, w1_scale = quantize_weight_per_tensor_fp8(w1_f32)
    w2_fp8, w2_scale = quantize_weight_per_tensor_fp8(w2_f32)

    # Reference: stock torch implementation with dequantized weights.
    # This is what FA's fused_mlp_func would produce at bf16 weights;
    # we add fp8 quant noise on top.
    w1_bf16_ref = (w1_fp8.float() * w1_scale).to(torch.bfloat16)
    w2_bf16_ref = (w2_fp8.float() * w2_scale).to(torch.bfloat16)

    # Warmup (absorbs autotune-search cost + Triton lazy JIT)
    for _ in range(2):
        _ = F.linear(F.gelu(F.linear(x, w1_bf16_ref), approximate="tanh"), w2_bf16_ref)
        _ = sage_ffn(x, w1_fp8, w1_scale, w2_fp8, w2_scale)
    torch.cuda.synchronize()

    # Median of 5 timed runs for stability
    ref_samples = []
    sage_samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref = F.linear(F.gelu(F.linear(x, w1_bf16_ref), approximate="tanh"), w2_bf16_ref)
        torch.cuda.synchronize()
        ref_samples.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sage_ffn(x, w1_fp8, w1_scale, w2_fp8, w2_scale)
        torch.cuda.synchronize()
        sage_samples.append((time.perf_counter() - t0) * 1000)
    ref_samples.sort()
    sage_samples.sort()
    ref_ms = ref_samples[2]
    sage_ms = sage_samples[2]

    # Rtol
    a = out.float()
    e = ref.float()
    diff = (a - e).abs()
    eps = torch.finfo(a.dtype).eps
    rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), torch.tensor(eps, device=device))
    mean_rtol = rdiff.mean().item()
    max_rtol = rdiff.max().item()
    mean_atol = diff.mean().item()
    max_atol = diff.max().item()

    print(f"  shape out: {tuple(out.shape)}")
    print(f"  mean_rtol={mean_rtol:.4f}  max_rtol={max_rtol:.4f}")
    print(f"  mean_atol={mean_atol:.6f}  max_atol={max_atol:.6f}")
    print(f"  ref (torch ref)  : {ref_ms:.2f} ms")
    print(f"  sage_ffn (Triton): {sage_ms:.2f} ms  ({ref_ms/sage_ms:.2f}x vs ref)")

    return {
        "T": T, "mean_rtol": mean_rtol, "max_rtol": max_rtol,
        "mean_atol": mean_atol, "max_atol": max_atol,
        "ref_ms": ref_ms, "sage_ms": sage_ms,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"torch: {torch.__version__}  triton: {triton.__version__}  device: {torch.cuda.get_device_name(0)}")

    results = []
    for T in (10780, 44880):
        results.append(run_correctness_at(T))

    print("\n=== Day-2 verdict ===")
    rtol_budget = 0.10
    all_ok = all(r["mean_rtol"] < rtol_budget for r in results)
    for r in results:
        status = "PASS" if r["mean_rtol"] < rtol_budget else "FAIL"
        print(f"  [{status}] T={r['T']:>6}  mean_rtol={r['mean_rtol']:.4f} (<{rtol_budget})")

    if all_ok:
        print("\nKernel correctness PASS at LTX FFN shapes. Day-4 decision gate cleared early.")
        return 0
    else:
        print("\nFAIL: rtol budget exceeded at one or more shapes. Debug numerics before perf tuning.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
