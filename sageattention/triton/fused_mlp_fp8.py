"""Fused fp8 MLP kernel for LTX 2.3 distilled FFN blocks.

Implements `sage_ffn(x, w1, s1, w2, s2)` -- a fused
`Linear(fp8) -> GELU(tanh) -> Linear(fp8)` kernel that keeps the
inner intermediate `(B, T, inner)` in SMEM / registers and never
materializes it to HBM. Targets LTX 2.3 multi-guide workflows on
sm89 / RTX 40xx where the FFN intermediate at multi-guide T=44880
is ~1.47 GiB bf16 and dominates peak memory.

Design (per `internal/design/ffn_fusion_scoping.md`):

- bf16 input X (B, T, hidden=4096)
- fp8 (E4M3FN) weight W1 (inner=16384, hidden=4096) + scalar f32 scale s1
- fp8 (E4M3FN) weight W2 (hidden=4096, inner=16384) + scalar f32 scale s2
- On-the-fly per-token bf16 -> fp8 activation quantization (required
  because Triton 3.6.0 does not support mixed-dtype tl.dot;
  verified by 2026-05-16 API audit + day-1 spike).
- tl.dot(fp8, fp8) -> f32 accumulator on sm89 native fp8 tensor cores.
- GELU(approximate="tanh") on the f32 accumulator.
- Two-matmul tile loop: produce intermediate from matmul1,
  GELU+quantize it, feed into matmul2, never write the intermediate
  tile to HBM.
- bf16 output X' (B, T, hidden=4096).

Numerical-correctness budget: mean_rtol < 0.10 vs torch
`F.linear -> F.gelu(tanh) -> F.linear` with dequantized weights.

This v1 supports plain GELU MLP only (no gated SwiGLU/GEGLU). LTX
2.3's `FeedForward` class is GELU-only despite call sites passing
`glu=True` (Comfy-Org upstream dead-kwarg bug). If a future model
class needs real GEGLU, add a template variant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_E4M3_MAX = 448.0  # E4M3FN max representable magnitude


@triton.jit
def _fused_mlp_fp8_kernel(
    # Pointers
    X_ptr,                     # (M, K) bf16 input activation, M = B*T, K = hidden
    W1_ptr,                    # (N, K) fp8 weight, N = inner
    W2_ptr,                    # (K, N) fp8 weight (note: second matmul reduces N back to K)
    Out_ptr,                   # (M, K) bf16 output
    # Scales
    W1_scale,                  # f32 scalar, weight scale for W1
    W2_scale,                  # f32 scalar, weight scale for W2
    # Strides
    stride_xm, stride_xk,
    stride_w1n, stride_w1k,
    stride_w2k, stride_w2n,
    stride_om, stride_ok,
    # Shape
    M, N, K,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    # N is iterated over BLOCK_N tiles inside the kernel; the second matmul
    # reduces over the full N. We use a single output tile per (BLOCK_M, BLOCK_K).
    BLOCK_N: tl.constexpr,
):
    """One CTA produces one (BLOCK_M, BLOCK_K) tile of the output.

    Algorithm per CTA:
      1. Load X_tile (BLOCK_M, K) into a row-strip of the input matrix.
      2. Quantize X_tile to fp8 per-token: compute X_scale[m] = max(|X_tile[m]|) / fp8_max,
         X_tile_fp8 = round(X_tile / X_scale[:, None]).
      3. For each n-tile of inner (BLOCK_N at a time):
           a. Load W1_tile (BLOCK_N, K) fp8.
           b. acc1 = tl.dot(X_tile_fp8, W1_tile.T, out_dtype=f32)  -- shape (BLOCK_M, BLOCK_N).
           c. Dequant: acc1 *= X_scale[:, None] * W1_scale  -- scalar mult.
           d. GELU(acc1) -> intermediate (BLOCK_M, BLOCK_N) in f32 registers.
           e. Quantize intermediate to fp8 per-token (inside this BLOCK_N slice):
              inter_scale[m] = max(|inter[m]|) / fp8_max,
              inter_fp8 = round(inter / inter_scale[:, None]).
           f. Load W2_tile (BLOCK_K, BLOCK_N) fp8.
           g. acc2 += tl.dot(inter_fp8, W2_tile.T, out_dtype=f32)
              -- shape (BLOCK_M, BLOCK_K). Note we want output column = K,
              so W2 is (K, N) and we read the (BLOCK_K, BLOCK_N) slice
              and transpose.
           h. Apply inter_scale + W2_scale dequant inline.
      4. Store acc2 cast to bf16 to Out_tile.

    The intermediate (step 3d-3e) lives in registers per BLOCK_N slice;
    we never materialize the full (BLOCK_M, N=16384) intermediate.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_inner_k = tl.arange(0, BLOCK_K)  # full hidden K is iterated for X load

    # --- Load X_tile (BLOCK_M, K) into registers ---
    # The kernel does the full K-reduction in matmul-1 over BLOCK_K chunks
    # interleaved with the inner-N loop. So we load X piecewise.

    # Per-token absolute max for quantization scale -- one scan over K per token.
    # Use the bf16 input for the abs-max scan, then build the fp8 chunk just
    # before each tl.dot call.

    # We need the per-row scale of the full X_tile (over all K), so do one pass
    # to compute it.
    X_row_ptrs_base = X_ptr + offs_m[:, None] * stride_xm
    x_abs_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + offs_inner_k
        x_ptrs = X_row_ptrs_base + ks[None, :] * stride_xk
        mask = (offs_m[:, None] < M) & (ks[None, :] < K)
        x_chunk = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        chunk_max = tl.max(tl.abs(x_chunk), axis=1)
        x_abs_max = tl.maximum(x_abs_max, chunk_max)
    x_scale = x_abs_max / 448.0
    x_scale = tl.where(x_scale > 0.0, x_scale, 1e-6)  # avoid div by zero

    # Accumulator for the final output tile (BLOCK_M, BLOCK_K).
    acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # --- Inner loop: iterate N (inner dim) BLOCK_N at a time ---
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # Recompute matmul-1 result for this BLOCK_N slice by iterating K.
        # acc1 = X @ W1[n:n+BLOCK_N, :].T  -- shape (BLOCK_M, BLOCK_N)
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            ks = k_start + offs_inner_k

            # Load X chunk (BLOCK_M, BLOCK_K) bf16, quantize to fp8 with per-row x_scale.
            x_chunk_ptrs = X_row_ptrs_base + ks[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (ks[None, :] < K)
            x_chunk_bf16 = tl.load(x_chunk_ptrs, mask=x_mask, other=0.0)
            x_chunk_f32 = x_chunk_bf16.to(tl.float32) / x_scale[:, None]
            # Clamp + cast to fp8
            x_chunk_f32 = tl.minimum(tl.maximum(x_chunk_f32, -448.0), 448.0)
            x_chunk_fp8 = x_chunk_f32.to(tl.float8e4nv)

            # Load W1 chunk (BLOCK_N, BLOCK_K) fp8.
            w1_ptrs = W1_ptr + offs_n[:, None] * stride_w1n + ks[None, :] * stride_w1k
            w1_mask = (offs_n[:, None] < N) & (ks[None, :] < K)
            w1_chunk_fp8 = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
            # tl.load default dtype matches the pointer dtype (fp8e4nv).

            # Matmul: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) via W1.T
            acc1 += tl.dot(x_chunk_fp8, tl.trans(w1_chunk_fp8), out_dtype=tl.float32)

        # Dequant matmul-1 result: acc1 *= x_scale[m] * W1_scale (broadcast).
        acc1 = acc1 * (x_scale[:, None] * W1_scale)

        # GELU(approximate="tanh") on f32 accumulator:
        #   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        inter = 0.5 * acc1 * (1.0 + tl.extra.libdevice.tanh(
            0.7978845608028654 * (acc1 + 0.044715 * acc1 * acc1 * acc1)
        ))

        # Quantize intermediate to fp8 per-token. Compute scale within this BLOCK_N slice.
        # (Per-token across the slice, not across the full N -- the next slice gets its own scale.)
        inter_abs_max = tl.max(tl.abs(inter), axis=1)
        inter_scale = inter_abs_max / 448.0
        inter_scale = tl.where(inter_scale > 0.0, inter_scale, 1e-6)
        inter_norm = inter / inter_scale[:, None]
        inter_norm = tl.minimum(tl.maximum(inter_norm, -448.0), 448.0)
        inter_fp8 = inter_norm.to(tl.float8e4nv)

        # Load W2 chunk (BLOCK_K, BLOCK_N) fp8.
        # W2 is (K, N). Slice rows [pid_k * BLOCK_K : (pid_k+1) * BLOCK_K],
        # cols [n_start : n_start + BLOCK_N].
        w2_ptrs = W2_ptr + offs_k[:, None] * stride_w2k + offs_n[None, :] * stride_w2n
        w2_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w2_chunk_fp8 = tl.load(w2_ptrs, mask=w2_mask, other=0.0)

        # Matmul-2: (BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_K) via W2.T
        # acc_out += inter_fp8 @ w2_chunk.T
        acc_partial = tl.dot(inter_fp8, tl.trans(w2_chunk_fp8), out_dtype=tl.float32)
        # Dequant: acc_partial *= inter_scale[m] * W2_scale.
        acc_out += acc_partial * (inter_scale[:, None] * W2_scale)

    # Store output tile cast to bf16.
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(out_ptrs, acc_out.to(tl.bfloat16), mask=out_mask)


def sage_ffn(
    x: torch.Tensor,           # (B, T, hidden) bf16
    w1: torch.Tensor,          # (inner, hidden) fp8 e4m3fn
    w1_scale: float,           # scalar f32 weight scale for w1
    w2: torch.Tensor,          # (hidden, inner) fp8 e4m3fn
    w2_scale: float,           # scalar f32 weight scale for w2
) -> torch.Tensor:
    """Fused fp8 MLP: out = Linear_out(GELU_tanh(Linear_in(x))).

    Args:
        x: (B, T, hidden) bf16 activation.
        w1: (inner, hidden) fp8 e4m3fn weight for the up-projection.
        w1_scale: scalar f32 weight scale for w1.
        w2: (hidden, inner) fp8 e4m3fn weight for the down-projection.
        w2_scale: scalar f32 weight scale for w2.

    Returns:
        (B, T, hidden) bf16 output.

    Constraints:
        - Plain GELU (approximate="tanh") only; no gated SwiGLU/GEGLU.
        - bf16 input dtype only (no fp16 in v1).
        - sm89 native fp8 tensor core support required.
    """
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
    assert x.dtype == torch.bfloat16
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn

    *batch_dims, hidden = x.shape
    inner = w1.shape[0]
    assert w1.shape == (inner, hidden), f"w1 shape {tuple(w1.shape)} != ({inner}, {hidden})"
    assert w2.shape == (hidden, inner), f"w2 shape {tuple(w2.shape)} != ({hidden}, {inner})"

    M = 1
    for d in batch_dims:
        M *= d
    K = hidden
    N = inner

    x_flat = x.reshape(M, K).contiguous()
    out_flat = torch.empty(M, K, dtype=torch.bfloat16, device=x.device)

    # Block sizes -- single config initially, no autotune.
    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_N = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _fused_mlp_fp8_kernel[grid](
        x_flat, w1, w2, out_flat,
        w1_scale, w2_scale,
        x_flat.stride(0), x_flat.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        out_flat.stride(0), out_flat.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return out_flat.reshape(*batch_dims, K)
