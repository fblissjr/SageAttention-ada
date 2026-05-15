"""Two-kernel fp8 MLP for LTX 2.3 distilled FFN blocks.

Implements `sage_ffn(x, w1, s1, w2, s2)` -- a two-step
`Linear(fp8) -> GELU(tanh) -> Linear(fp8)` path. The two matmuls run
as separate Triton kernels; the intermediate (M, inner=16384) is
written to HBM between them.

Activation quantization is **per-block-K**: each (BLOCK_M, BLOCK_K)
tile of the activation gets its own f32 scale, computed inline
during the K-reduction. This eliminates the redundant K-pass that
per-row quantization would require, at the cost of slightly
coarser scaling (~0.005 rtol vs per-row in audio-loop's day-1
estimate; verified empirically here against the 0.10 budget).

The wedge against torch reference comes from fp8-native matmul on
sm89: torch's bf16 matmul against fp8 weights has to dequant first
(2x memory bandwidth, bf16 tensor cores at ~330 TFLOPS); this
kernel loads fp8 weights directly + uses sm89 fp8 tensor cores at
~660 TFLOPS. Realistic delivered: 1.5-2x vs torch reference.

Compose with `LTXVChunkFeedForward` on 24 GiB cards (the
intermediate hits HBM between kernels; on multi-guide T=44880
it's ~1.47 GiB unchunked).

v1 supports plain GELU MLP only (no gated SwiGLU/GEGLU).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_FP8_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_FP8_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_matmul_gelu_kernel(
    X_ptr, W_ptr, Out_ptr,
    W_scale,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute out = gelu_tanh(X @ W.T * scales) for one (BLOCK_M, BLOCK_N) tile.

    Per-block-K activation quantization: each (BLOCK_M, BLOCK_K) chunk
    of X gets its own f32 scale, applied inline. Eliminates the
    redundant K-pass that per-row quant would require.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    X_row_ptrs_base = X_ptr + offs_m[:, None] * stride_xm

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + offs_k

        # Load X chunk (BLOCK_M, BLOCK_K) bf16
        x_chunk_ptrs = X_row_ptrs_base + ks[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (ks[None, :] < K)
        x_chunk_bf16 = tl.load(x_chunk_ptrs, mask=x_mask, other=0.0)
        x_chunk_f32 = x_chunk_bf16.to(tl.float32)

        # Per-block-K quantization: one scale per row within this chunk.
        x_abs_max = tl.max(tl.abs(x_chunk_f32), axis=1)
        x_chunk_scale = x_abs_max / 448.0
        x_chunk_scale = tl.where(x_chunk_scale > 0.0, x_chunk_scale, 1e-6)

        x_normalized = x_chunk_f32 / x_chunk_scale[:, None]
        x_normalized = tl.minimum(tl.maximum(x_normalized, -448.0), 448.0)
        x_chunk_fp8 = x_normalized.to(tl.float8e4nv)

        # Load W chunk fp8
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + ks[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (ks[None, :] < K)
        w_chunk_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # fp8 x fp8 -> f32 accum, then scale by chunk-local x scale.
        partial = tl.dot(x_chunk_fp8, tl.trans(w_chunk_fp8), out_dtype=tl.float32)
        acc += partial * x_chunk_scale[:, None]

    # Apply weight scale (scalar).
    acc = acc * W_scale

    # GELU(approximate="tanh")
    out = 0.5 * acc * (1.0 + tl.extra.libdevice.tanh(
        0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    ))

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(tl.bfloat16), mask=out_mask)


_FP8_MATMUL_CONFIGS_K2 = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 256, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_FP8_MATMUL_CONFIGS_K2, key=["M", "N", "K"])
@triton.jit
def _fp8_matmul_kernel(
    X_ptr, W_ptr, Out_ptr,
    W_scale,
    stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_om, stride_ok,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Compute out = (X @ W.T) * scales for one (BLOCK_M, BLOCK_K) tile.

    Same per-block-N activation quant pattern as kernel 1 (here N is
    the reduction dim). No GELU epilogue.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    X_row_ptrs_base = X_ptr + offs_m[:, None] * stride_xm

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        ns = n_start + offs_n

        x_chunk_ptrs = X_row_ptrs_base + ns[None, :] * stride_xn
        x_mask = (offs_m[:, None] < M) & (ns[None, :] < N)
        x_chunk_bf16 = tl.load(x_chunk_ptrs, mask=x_mask, other=0.0)
        x_chunk_f32 = x_chunk_bf16.to(tl.float32)

        x_abs_max = tl.max(tl.abs(x_chunk_f32), axis=1)
        x_chunk_scale = x_abs_max / 448.0
        x_chunk_scale = tl.where(x_chunk_scale > 0.0, x_chunk_scale, 1e-6)

        x_normalized = x_chunk_f32 / x_chunk_scale[:, None]
        x_normalized = tl.minimum(tl.maximum(x_normalized, -448.0), 448.0)
        x_chunk_fp8 = x_normalized.to(tl.float8e4nv)

        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + ns[None, :] * stride_wn
        w_mask = (offs_k[:, None] < K) & (ns[None, :] < N)
        w_chunk_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)

        partial = tl.dot(x_chunk_fp8, tl.trans(w_chunk_fp8), out_dtype=tl.float32)
        acc += partial * x_chunk_scale[:, None]

    acc = acc * W_scale

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def sage_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: float,
    w2: torch.Tensor,
    w2_scale: float,
) -> torch.Tensor:
    """Two-kernel fp8 MLP: out = Linear_out(GELU_tanh(Linear_in(x))).

    Args:
        x: (B, T, hidden) bf16 activation.
        w1: (inner, hidden) fp8 e4m3fn up-projection weight.
        w1_scale: scalar f32 weight scale for w1.
        w2: (hidden, inner) fp8 e4m3fn down-projection weight.
        w2_scale: scalar f32 weight scale for w2.

    Returns:
        (B, T, hidden) bf16 output.
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
    assert x.dtype == torch.bfloat16
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    *batch_dims, hidden = x.shape
    inner = w1.shape[0]
    assert w1.shape == (inner, hidden)
    assert w2.shape == (hidden, inner)

    M = 1
    for d in batch_dims:
        M *= d

    x_flat = x.reshape(M, hidden).contiguous()
    intermediate = torch.empty(M, inner, dtype=torch.bfloat16, device=x.device)
    out_flat = torch.empty(M, hidden, dtype=torch.bfloat16, device=x.device)

    # Autotune picks BLOCK_M / BLOCK_N / BLOCK_K + num_warps + num_stages.
    grid1 = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(inner, meta["BLOCK_N"]))
    _fp8_matmul_gelu_kernel[grid1](
        x_flat, w1, intermediate,
        w1_scale,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(0), w1.stride(1),
        intermediate.stride(0), intermediate.stride(1),
        M, inner, hidden,
    )

    grid2 = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(hidden, meta["BLOCK_K"]))
    _fp8_matmul_kernel[grid2](
        intermediate, w2, out_flat,
        w2_scale,
        intermediate.stride(0), intermediate.stride(1),
        w2.stride(0), w2.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        M, inner, hidden,
    )

    return out_flat.reshape(*batch_dims, hidden)
