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

A third comparand (torchao `addmm_float8_unwrapped_inference`) is
included automatically if `torchao` is importable. NOTE: this
function is currently a thin Python wrapper around
`torch._scaled_mm` -- same underlying cuBLAS kernel as the
existing `_scaled_mm` arm, so it does NOT directly test Cell C
hypothesis 1 (stock comparand identity). The actual hypothesis-1
test would be `torchao.float8.Float8Linear` with per-row scaling.

What this arm DOES give us, even with identical kernel dispatch:
(a) regression detection if torchao ever diverges from
`_scaled_mm` (e.g. fast-accum default flip, preprocessing
addition), (b) ABI stability across torch version bumps
(torchao tends to abstract `_scaled_mm` ABI changes), (c)
future-proofing for the eventual ComfyUI migration to torchao
primitives (the scaffold is in place; we don't scramble), (d)
ecosystem documentation. Bench arm is opt-in via torchao
availability; skipped silently when torchao isn't installed.

See `build_torchao.sh` for building torchao from a local
checkout against the active venv.

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


def _torchao_inference_matmul_available() -> bool:
    """Lazy availability check for torchao's inference fp8 matmul."""
    try:
        from torchao.float8.inference import addmm_float8_unwrapped_inference  # noqa: F401
        return True
    except ImportError:
        return False


def _scale_to_0d_tensor(scale: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(scale, dtype=torch.float32, device=device).view(())


def _fp8_mlp_with(
    matmul_fn,
    x_bf16: torch.Tensor,
    w1_fp8: torch.Tensor, s1: float,
    w2_fp8: torch.Tensor, s2: float,
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """Shared two-Linear-around-GELU(tanh) implementation parameterized on
    the matmul primitive. Used by both `torch_stock_fp8_mlp` and
    `torchao_inference_fp8_mlp`; the only difference between the two arms
    is which fp8 matmul kernel runs. `matmul_fn(a_fp8, b_fp8_t, a_scale,
    b_scale)` returns a bf16 tensor."""
    *batch_dims, hidden = x_bf16.shape
    M = 1
    for d in batch_dims:
        M *= d
    x_flat = x_bf16.reshape(M, hidden)

    # Stage 1
    x_fp8, x_scale = quantize_activation_per_tensor_fp8(x_flat)
    s1_tensor = _scale_to_0d_tensor(s1, x_bf16.device)
    intermediate = matmul_fn(x_fp8, w1_fp8.t(), x_scale, s1_tensor)
    if b1 is not None:
        intermediate = intermediate + b1
    intermediate = F.gelu(intermediate, approximate="tanh")

    # Stage 2
    interm_fp8, interm_scale = quantize_activation_per_tensor_fp8(intermediate)
    s2_tensor = _scale_to_0d_tensor(s2, x_bf16.device)
    out_flat = matmul_fn(interm_fp8, w2_fp8.t(), interm_scale, s2_tensor)
    if b2 is not None:
        out_flat = out_flat + b2

    return out_flat.reshape(*batch_dims, hidden)


def _scaled_mm_adapter(a_fp8, b_fp8_t, a_scale, b_scale):
    return torch._scaled_mm(a_fp8, b_fp8_t, scale_a=a_scale, scale_b=b_scale, out_dtype=torch.bfloat16)


def _torchao_addmm_adapter(a_fp8, b_fp8_t, a_scale, b_scale):
    from torchao.float8.inference import addmm_float8_unwrapped_inference
    return addmm_float8_unwrapped_inference(
        a_data=a_fp8, a_scale=a_scale,
        b_data=b_fp8_t, b_scale=b_scale,
        output_dtype=torch.bfloat16,
    )


def torch_stock_fp8_mlp(
    x_bf16: torch.Tensor,
    w1_fp8: torch.Tensor, s1: float,
    w2_fp8: torch.Tensor, s2: float,
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """`torch._scaled_mm` comparand. cuBLAS XMMA fp8 matmul path --
    matches what a ComfyUI fp8 Linear dispatches to in production
    (the `sm89_xmma_gemm_e4m3bf16_e4m3f32_*` kernels)."""
    return _fp8_mlp_with(_scaled_mm_adapter, x_bf16, w1_fp8, s1, w2_fp8, s2, b1, b2)


def torchao_inference_fp8_mlp(
    x_bf16: torch.Tensor,
    w1_fp8: torch.Tensor, s1: float,
    w2_fp8: torch.Tensor, s2: float,
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """torchao inference-fp8 path comparand via
    `addmm_float8_unwrapped_inference`. Addresses Cell C hypothesis 1
    (stock comparand identity); see module docstring + CHANGELOG
    Decision log for motivation. Skipped automatically by callers
    when `_torchao_inference_matmul_available()` returns False."""
    return _fp8_mlp_with(_torchao_addmm_adapter, x_bf16, w1_fp8, s1, w2_fp8, s2, b1, b2)


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
    torchao_available = _torchao_inference_matmul_available()

    def call_sage():
        return sage_ffn(x, w1, s1, w2, s2, b1=b1, b2=b2)

    def call_stock():
        return torch_stock_fp8_mlp(x, w1, s1, w2, s2, b1, b2)

    def call_torchao():
        return torchao_inference_fp8_mlp(x, w1, s1, w2, s2, b1, b2)

    out_sage = call_sage()
    out_stock = call_stock()
    mean_rtol, _max_rtol, _mean_atol, _max_atol = accuracy_metrics(out_sage, out_stock)
    # Drop the stock output before timing; the timing loop will reallocate
    # its own intermediates and we only need out_sage live for the torchao
    # rtol comparison below. ~1.4 GiB at T=42240.
    del out_stock

    sage_ms = _time_call(call_sage)
    stock_ms = _time_call(call_stock)

    torchao_ms: float | None = None
    torchao_vs_sage_rtol: float | None = None
    if torchao_available:
        out_torchao = call_torchao()
        torchao_vs_sage_rtol, _, _, _ = accuracy_metrics(out_sage, out_torchao)
        # Release immediately after the rtol calc; the timing loop will
        # reallocate. Avoids stacking three ~1.4 GiB intermediates at T=42240.
        del out_torchao
        torchao_ms = _time_call(call_torchao)

    # Final cleanup between shapes. Triton autotune can hold multiple
    # config-compile-time intermediates concurrently; explicit reclaim
    # keeps the bench safe on a 24 GiB card.
    del x, w1, w2, b1, b2, out_sage
    torch.cuda.empty_cache()

    return {
        "T": T, "hidden": hidden, "inner": inner,
        "sage_ms": sage_ms, "stock_ms": stock_ms,
        "torchao_ms": torchao_ms,
        "ratio": stock_ms / sage_ms,
        "torchao_ratio": (torchao_ms / sage_ms) if torchao_ms is not None else None,
        "mean_rtol": mean_rtol,
        "torchao_rtol": torchao_vs_sage_rtol,
    }


def _print_markdown_table(rows: list[dict], title: str) -> None:
    print(f"\n### {title}\n")
    show_torchao = any(r.get("torchao_ms") is not None for r in rows)

    base_headers = ["T (seq)", "hidden", "inner", "sage_ffn ms", "torch stock fp8 ms", "stock/sage", "mean rtol (vs stock)"]
    torchao_headers = ["torchao fp8 ms", "torchao/sage", "rtol (vs torchao)"]
    headers = base_headers + torchao_headers if show_torchao else base_headers

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---:"] * len(headers)) + "|")

    for r in rows:
        cells = [
            str(r["T"]), str(r["hidden"]), str(r["inner"]),
            f"{r['sage_ms']:.3f}", f"{r['stock_ms']:.3f}",
            f"{r['ratio']:.2f}x", f"{r['mean_rtol']:.4f}",
        ]
        if show_torchao:
            tms, tratio, trtol = r.get("torchao_ms"), r.get("torchao_ratio"), r.get("torchao_rtol")
            cells += [
                f"{tms:.3f}" if tms is not None else "n/a",
                f"{tratio:.2f}x" if tratio is not None else "n/a",
                f"{trtol:.4f}" if trtol is not None else "n/a",
            ]
        print("| " + " | ".join(cells) + " |")


def part_a_production_shapes() -> list[dict]:
    """The exact chunked call sites + the two stage-level call sites
    consumer-side reports at production:

    - T=4096 (full chunk) / T=1808 (residual chunk): the per-chunk
      shapes a consumer-side seq-chunking wrapper emits.
    - T=10780 / T=42240: stage-1 / stage-2 unchunked FFN call sites
      seen in a canonical LTX 2.3 video render. Useful for direct
      comparison against in-pipeline measurements.
    """
    shapes = [
        (4096, 4096, 16384),
        (1808, 4096, 16384),
        (10780, 4096, 16384),
        (42240, 4096, 16384),
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
    print("- `torchao fp8` (if installed) is `torchao.float8.inference.addmm_float8_unwrapped_inference`")
    print("  bracketing the same GELU + re-quantization, addressing Cell C hypothesis 1")
    print("  (stock comparand identity). Skipped silently when torchao isn't importable.")
    print("- `stock/sage` ratio > 1 means sage_ffn is faster at that shape; < 1 means stock is faster.")
    print("  Same convention for `torchao/sage`.")
    print("- `mean rtol` (vs stock) is symmetric per-element relative error of sage_ffn vs the stock")
    print("  path; `rtol (vs torchao)` is sage_ffn vs the torchao path. Both expected to sit around")
    print("  the fp8-quantization noise floor (~0.04-0.10 for FFN-shape weights).")
    print("- Synthetic only. Production e2e perf may differ -- see")
    print("  `docs/perf_research_framework.md` evidence ladder for the gating discipline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
