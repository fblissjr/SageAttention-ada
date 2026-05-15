"""Day-1 spike for v0.6 fused-MLP work: verify Triton's fp8 mma path on sm89.

Tests the load-bearing dependency: can we compile + run
`tl.dot(fp8_a, fp8_b) -> f32 accum` on sm89 with the version-pinned
Triton (3.6.0 in this venv), at both small (API check) and
realistic (pipelining + register pressure) shapes?

The API audit
(`coderef/triton/python/triton/language/semantic.py:1426-1434`)
confirmed Triton requires both operands of `tl.dot` to be fp8 -- it
does NOT accept mixed fp8 + bf16. So the kernel design needs
on-the-fly activation quantization: bf16 input -> fp8 (per-token
scale) -> fp8 matmul -> f32 accum -> scalar dequant.

This spike tests that pattern end-to-end at two configs:
1. Small (T=128, K=4096, N=16384): API correctness, fast iteration.
2. Stage-1 LTX shape (T=10780, K=4096, N=16384): pipelining +
   register pressure realism at production tile size.

Decision gate: both configs must compile + produce mean_rtol < 0.10
vs torch reference. If either fails, the v0.6 plan needs revision.

Reference numerics: torch.matmul(X_bf16, (W_fp8.float() *
weight_scale).to(bf16).T). The fp8-quant noise floor is expected
~0.04-0.08; if mean_rtol > 0.10 something is wrong.

Run with the venv that has sage installed active:
    ${VIRTUAL_ENV}/bin/python tests/spikes/spike_fp8_mma.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sageattention.triton.fused_mlp_fp8 import FP8_E4M3_MAX

# Reuse the symmetric-denominator rtol helper used by every other
# accuracy bench in this repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]


# LTX 2.3 FFN shape constants -- hidden = 4096, inner = 16384, bf16 act.
HIDDEN = 4096
INNER = 16384


@triton.jit
def _fp8_matmul_kernel(
    X_ptr, W_ptr, X_scale_ptr, W_scale,  # X_scale per-row, W_scale scalar
    Out_ptr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fp8 x Fp8 -> bf16 matmul with per-row activation scale + scalar weight scale.

    X is pre-quantized fp8 (BLOCK_M-row tiles each have their own scale in
    X_scale[m]). W is offline fp8 with a single scalar scale.

    Output = (X_fp8 @ W_fp8.T) * X_row_scale[m] * W_scale, cast to bf16.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    W_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(X_ptrs, mask=(offs_m[:, None] < M) & ((offs_k + k)[None, :] < K), other=0.0)
        w = tl.load(W_ptrs, mask=(offs_n[:, None] < N) & ((offs_k + k)[None, :] < K), other=0.0)
        # Both operands fp8 -- the legal path per Triton 3.6 semantic.py:1426
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Per-row activation scale (one f32 per M row) + scalar weight scale.
    x_scale = tl.load(X_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc = acc * x_scale[:, None] * W_scale

    Out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(Out_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def quantize_per_token_fp8(x_bf16: torch.Tensor, fp8_dtype=torch.float8_e4m3fn) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) fp8 quantization of a bf16 activation tensor.

    Returns (x_fp8, scales_f32). scales_f32 has shape (M,) -- one scalar per row.
    Quantization: x_fp8 = round(x_bf16 / scale * fp8_max); scale = abs(x_row).max() / fp8_max.
    """
    x_f32 = x_bf16.float()
    abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scales = (abs_max / FP8_E4M3_MAX).squeeze(-1)  # (M,)
    x_scaled = (x_f32 / abs_max * FP8_E4M3_MAX).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(fp8_dtype)
    return x_fp8, scales.to(torch.float32)


class SpikeResult(NamedTuple):
    config_name: str
    compiled: bool
    mean_rtol: float
    max_rtol: float
    mean_atol: float
    err: str | None = None


def run_spike(M: int, N: int, K: int, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, name: str) -> SpikeResult:
    print(f"\n=== {name}: M={M} N={N} K={K} | BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} ===")
    device = torch.device("cuda")

    torch.manual_seed(0)
    # bf16 activation (the actual input to the FFN at runtime)
    X_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # fp8 weight with a per-tensor scalar scale (matches LTX 2.3 distilled checkpoint)
    W_f32 = torch.randn(N, K, dtype=torch.float32, device=device) * 0.02  # small init
    W_max = W_f32.abs().amax()
    W_scale = (W_max / FP8_E4M3_MAX).item()
    W_fp8 = (W_f32 / W_max * FP8_E4M3_MAX).to(torch.float8_e4m3fn)

    # On-the-fly activation quantization (per-token / per-row)
    X_fp8, X_scales = quantize_per_token_fp8(X_bf16)

    # Output buffer
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    try:
        _fp8_matmul_kernel[grid](
            X_fp8, W_fp8, X_scales, W_scale,
            out,
            X_fp8.stride(0), X_fp8.stride(1),
            W_fp8.stride(0), W_fp8.stride(1),
            out.stride(0), out.stride(1),
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        torch.cuda.synchronize()
        compiled = True
    except Exception as e:
        return SpikeResult(name, False, 0.0, 0.0, 0.0, err=f"{type(e).__name__}: {e}")

    # Reference numerics: torch matmul of bf16 X against dequantized W (back to bf16)
    # This is what stock torch would compute if W were bf16; we accept fp8 quant noise on top.
    W_bf16_ref = (W_fp8.float() * W_scale).to(torch.bfloat16)
    ref = torch.matmul(X_bf16, W_bf16_ref.T)

    mean_rtol, max_rtol, mean_atol, _ = accuracy_metrics(out, ref)

    print(f"  compiled: {compiled}")
    print(f"  mean_rtol={mean_rtol:.4f}  max_rtol={max_rtol:.4f}  mean_atol={mean_atol:.6f}")
    return SpikeResult(name, compiled, mean_rtol, max_rtol, mean_atol)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    print(f"torch: {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    results = []

    # Config 1: small (T=128) -- API correctness, fast iteration
    results.append(run_spike(
        M=128, N=INNER, K=HIDDEN,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        name="small (T=128)",
    ))

    # Config 2: stage-1 LTX shape (T=10780) -- pipelining + register pressure
    results.append(run_spike(
        M=10780, N=INNER, K=HIDDEN,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        name="stage-1 LTX (T=10780)",
    ))

    print("\n=== Spike verdict ===")
    rtol_budget = 0.10
    all_ok = all(r.compiled and r.mean_rtol < rtol_budget for r in results)
    for r in results:
        status = "PASS" if r.compiled and r.mean_rtol < rtol_budget else "FAIL"
        if r.err:
            print(f"  [{status}] {r.config_name}: err={r.err}")
        else:
            print(f"  [{status}] {r.config_name}: compiled={r.compiled} mean_rtol={r.mean_rtol:.4f} (<{rtol_budget})")
    print()
    if all_ok:
        print("SPIKE PASSED. tl.dot(fp8, fp8) on sm89 works at both configs; activation-quant pattern produces correct numerics.")
        return 0
    if not all(r.compiled for r in results):
        print("SPIKE FAILED: at least one config did not compile. v0.6 plan needs revision.")
        return 1
    print(f"SPIKE FAILED: rtol budget {rtol_budget} exceeded. fp8 quant noise too high; revisit activation-quant scheme.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
