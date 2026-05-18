#!/usr/bin/env python3
"""Synthetic micro-bench for sage_ffn vs torch's stock fp8 path.

Anchored to LTX 2.3 FFN shapes that the consumer-side chunked path
produces. Two parts:

(A) Shapes that match the production chunked call sites:
    - x.shape = [1, 4096, 4096] (full chunk)
    - x.shape = [1, 1808, 4096] (residual chunk from splitting
      seq=10000 input at chunk_seq=4096)
    Both at inner_dim=16384 (LTX FFN expansion = 4x).

(B) Chunk-size sweep at hidden=4096, inner=16384:
    seq ∈ {512, 1024, 2048, 4096, 8192, 16384}

The torch reference is the cuBLAS XMMA fp8 matmul path -- the same
kernels (`sm89_xmma_gemm_e4m3bf16_e4m3f32_*`) that a ComfyUI fp8
Linear dispatches to via `torch._scaled_mm`. Apples-to-apples with
the kernel an FFN consumer would see if sage_ffn falls back to
stock.

Output is markdown tables suitable for direct paste into a memo.

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
Expected on RTX 4090 / sm89 / CUDA >= 12.8.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sageattention.triton.fused_mlp_fp8 import FP8_E4M3_MAX, sage_ffn
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]


def quantize_weight_per_tensor_fp8(w_f32: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor fp8 quant of a weight matrix. Returns (w_fp8, scalar_scale)."""
    w_max = w_f32.abs().amax().item()
    if w_max == 0:
        return w_f32.to(torch.float8_e4m3fn), 1.0
    scale = w_max / FP8_E4M3_MAX
    w_scaled = (w_f32 / w_max * FP8_E4M3_MAX).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return w_scaled.to(torch.float8_e4m3fn), scale


def quantize_activation_per_tensor_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor fp8 quant of an activation. Returns
    (x_fp8, scale_f32_tensor). Mirrors what a ComfyUI fp8 Linear's
    forward does on x before dispatching to `torch._scaled_mm`."""
    x_max = x.abs().amax().to(torch.float32).clamp(min=1e-12)
    scale = x_max / FP8_E4M3_MAX
    x_fp8 = (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return x_fp8, scale.view(())


def torch_stock_fp8_mlp(
    x_bf16: torch.Tensor,
    w1_fp8: torch.Tensor, s1: float,
    w2_fp8: torch.Tensor, s2: float,
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """Reference implementation: torch's stock cuBLAS fp8 matmul path.
    Two `torch._scaled_mm` calls bracketing a GELU(tanh), with a
    re-quantization of the intermediate (the cost the consumer-side
    `quantize_fp8_tensor_kernel` represents in production traces)."""
    *batch_dims, hidden = x_bf16.shape
    M = 1
    for d in batch_dims:
        M *= d
    x_flat = x_bf16.reshape(M, hidden)

    # Stage 1
    x_fp8, x_scale = quantize_activation_per_tensor_fp8(x_flat)
    s1_tensor = torch.tensor(s1, dtype=torch.float32, device=x_bf16.device).view(())
    intermediate = torch._scaled_mm(
        x_fp8, w1_fp8.t(),
        scale_a=x_scale, scale_b=s1_tensor,
        out_dtype=torch.bfloat16,
    )
    if b1 is not None:
        intermediate = intermediate + b1
    intermediate = F.gelu(intermediate, approximate="tanh")

    # Stage 2
    interm_fp8, interm_scale = quantize_activation_per_tensor_fp8(intermediate)
    s2_tensor = torch.tensor(s2, dtype=torch.float32, device=x_bf16.device).view(())
    out_flat = torch._scaled_mm(
        interm_fp8, w2_fp8.t(),
        scale_a=interm_scale, scale_b=s2_tensor,
        out_dtype=torch.bfloat16,
    )
    if b2 is not None:
        out_flat = out_flat + b2

    return out_flat.reshape(*batch_dims, hidden)


def _build_block(T: int, hidden: int, inner: int, with_bias: bool, seed: int = 0):
    """Build a single FFN's worth of weights + activation for a given shape."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    x = torch.randn(1, T, hidden, dtype=torch.bfloat16, device=device)
    w1_f32 = torch.randn(inner, hidden, dtype=torch.float32, device=device) * (1.0 / (hidden ** 0.5))
    w2_f32 = torch.randn(hidden, inner, dtype=torch.float32, device=device) * (1.0 / (inner ** 0.5))
    w1_fp8, s1 = quantize_weight_per_tensor_fp8(w1_f32)
    w2_fp8, s2 = quantize_weight_per_tensor_fp8(w2_f32)
    if with_bias:
        b1 = torch.randn(inner, dtype=torch.bfloat16, device=device) * (1.0 / (inner ** 0.5))
        b2 = torch.randn(hidden, dtype=torch.bfloat16, device=device) * (1.0 / (hidden ** 0.5))
    else:
        b1 = None
        b2 = None
    return x, w1_fp8, s1, w2_fp8, s2, b1, b2


def _time_call(fn, n_warmup: int = 3, n_samples: int = 7) -> float:
    """Return median ms over n_samples after n_warmup runs."""
    for _ in range(n_warmup):
        _ = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n_samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def bench_one_shape(T: int, hidden: int, inner: int, with_bias: bool = True) -> dict:
    x, w1, s1, w2, s2, b1, b2 = _build_block(T, hidden, inner, with_bias)

    def call_sage():
        return sage_ffn(x, w1, s1, w2, s2, b1=b1, b2=b2)

    def call_stock():
        return torch_stock_fp8_mlp(x, w1, s1, w2, s2, b1, b2)

    out_sage = call_sage()
    out_stock = call_stock()
    mean_rtol, _max_rtol, _mean_atol, _max_atol = accuracy_metrics(out_sage, out_stock)

    sage_ms = _time_call(call_sage)
    stock_ms = _time_call(call_stock)

    return {
        "T": T, "hidden": hidden, "inner": inner,
        "sage_ms": sage_ms, "stock_ms": stock_ms,
        "ratio": stock_ms / sage_ms,
        "mean_rtol": mean_rtol,
    }


def _print_markdown_table(rows: list[dict], title: str) -> None:
    print(f"\n### {title}\n")
    print("| T (seq) | hidden | inner | sage_ffn ms | torch stock fp8 ms | stock/sage | mean rtol |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['T']} | {r['hidden']} | {r['inner']} | "
            f"{r['sage_ms']:.3f} | {r['stock_ms']:.3f} | "
            f"{r['ratio']:.2f}x | {r['mean_rtol']:.4f} |"
        )


def part_a_production_shapes() -> list[dict]:
    """Audio claude's Ask 1 shapes: the exact chunked call sites."""
    shapes = [
        (4096, 4096, 16384),
        (1808, 4096, 16384),
    ]
    rows = []
    for T, hidden, inner in shapes:
        rows.append(bench_one_shape(T, hidden, inner, with_bias=True))
    return rows


def part_b_chunk_sweep() -> list[dict]:
    """Audio claude's optional chunk-size sweep at LTX hidden=4096."""
    rows = []
    for T in [512, 1024, 2048, 4096, 8192, 16384]:
        rows.append(bench_one_shape(T, hidden=4096, inner=16384, with_bias=True))
    return rows


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return 0

    print("# sage_ffn vs torch stock fp8 -- synthetic micro-bench")
    print(f"\nEnv: torch={torch.__version__}, CUDA={torch.version.cuda}")
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    print(f"Arch: sm{cap_major}{cap_minor}, GPU: {torch.cuda.get_device_name(0)}")

    part_a = part_a_production_shapes()
    _print_markdown_table(part_a, "Part A: production chunked shapes (LTX 2.3 FFN, hidden=4096, inner=16384)")

    part_b = part_b_chunk_sweep()
    _print_markdown_table(part_b, "Part B: chunk-size sweep (hidden=4096, inner=16384)")

    print("\n## Notes\n")
    print("- `sage_ffn` is the Triton two-kernel fp8 path (`Linear -> GELU(tanh) -> Linear`).")
    print("- `torch stock fp8` is two `torch._scaled_mm` calls with a re-quantization of the")
    print("  intermediate, matching what a ComfyUI fp8 `Linear` dispatches to in production")
    print("  (the `sm89_xmma_gemm_e4m3bf16_e4m3f32_*` cuBLAS XMMA kernels).")
    print("- `stock/sage` ratio > 1 means sage_ffn is faster at that shape; < 1 means stock is faster.")
    print("- `mean rtol` is symmetric per-element relative error of sage_ffn vs the stock path,")
    print("  expected to sit around the fp8-quantization noise floor (~0.04-0.10 for FFN-shape weights).")
    print("- Synthetic only. Production e2e perf may differ -- see")
    print("  `docs/perf_research_framework.md` evidence ladder for the gating discipline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
