"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

from ._int_offsets import needs_int64_offsets


@triton.jit
def quant_query_per_thread_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        Factor, stride_fz, stride_fh,
                                        C: tl.constexpr, BLK: tl.constexpr,
                                        USE_I64: tl.constexpr = False,
                                        BALANCE: tl.constexpr = False,
                                        GROUP: tl.constexpr = 1):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    # Per-(batch, kv-head, channel) balance factor, [B, H_kv, C] fp32; a
    # query head maps to its kv head under GQA. Read before the int64
    # promotion below so the small tensor is indexed in int32 either way.
    if BALANCE:
        factor_ptrs = Factor + off_b * stride_fz + (off_h // GROUP) * stride_fh + tl.arange(0, C)
        f = tl.load(factor_ptrs)

    # Promoting the base offset covers every term at once: the batch and
    # head products overflow int32 before the row term does in HND, so
    # promoting only the row index still faults there.
    if USE_I64:
        off_h = off_h.to(tl.int64)
        off_b = off_b.to(tl.int64)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    if USE_I64:
        offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    if BALANCE:
        # q . k == (q * f) . (k / f): the factor moves INT8 resolution from
        # K's loud channels onto Q at no cost to the attention math.
        x = x * f[None, :]
    scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_key_per_thread_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        Factor, stride_fz, stride_fh,
                                        C: tl.constexpr, BLK: tl.constexpr,
                                        USE_I64: tl.constexpr = False,
                                        BALANCE: tl.constexpr = False):
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)
    if BALANCE:
        # The wrapper hands the key kernel 1/f, so this is a multiply too.
        factor_ptrs = Factor + off_b * stride_fz + off_h * stride_fh + tl.arange(0, C)
        f = tl.load(factor_ptrs)

    if USE_I64:
        off_h = off_h.to(tl.int64)
        off_b = off_b.to(tl.int64)

    # offs_n = off_blk * BLK + tl.cat(tl.arange(0, BLK // 8) * 8, tl.arange(0, BLK // 8) * 8 + 1, True) + off_tld * 2
    # offs_k = tl.arange(0, C)

    # input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    # output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    # scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    # x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    # x = x.to(tl.float32)
    # scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    # x_int8 = x / scale
    # x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    # x_int8 = x_int8.to(tl.int8)
    # tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    # tl.store(scale_ptrs, scale)

    offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
    if USE_I64:
        offs_n0 = offs_n0.to(tl.int64)
        offs_n1 = offs_n1.to(tl.int64)
    offs_k = tl.arange(0, C)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + offs_k[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + offs_k[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
    x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    if BALANCE:
        x0 = x0 * f[None, :]
        x1 = x1 * f[None, :]
    scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
    x0_int8 = x0 / scale
    x1_int8 = x1 / scale
    x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
    x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
    x0_int8 = x0_int8.to(tl.int8)
    x1_int8 = x1_int8.to(tl.int8)
    tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
    tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_query_per_thread_int4_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 7. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

@triton.jit
def quant_key_per_thread_int4_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        C: tl.constexpr, BLK: tl.constexpr):      
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.cat(tl.arange(0, BLK // 8) * 8, tl.arange(0, BLK // 8) * 8 + 1, True) + off_tld * 2
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 7. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def qk_balance_factor(q, k, tensor_layout="HND", alpha=0.5, min_share=0.2):
    """Per-(batch, kv-head, channel) factor that balances K's channels against Q's.

    For any per-channel f, `q . k == (q * f) . (k / f)`. INT8 quantization
    of K uses one scale across a token block's 128 channels, so a few loud
    channels set the scale and the rest lose resolution; dividing K by
    `f = rms_k^alpha / rms_q^(1-alpha)` (geometric mean one per head) and
    multiplying Q by the same evens that out, at the price of some of Q's
    resolution -- alpha 0.5 measured best, alpha 1.0 measured worse
    everywhere (CHANGELOG, workload intel "MiniMax H3, block 49").

    Gated per head on the energy share of the four loudest K channels:
    below `min_share` the factor is one and the head is untouched, because
    on flat heads moving resolution onto Q costs a few percent for nothing.
    The default 0.2 is measured at kernel level (CHANGELOG v0.7.19): on
    MiniMax H3 it takes a fifth off the last block's error and is neutral on
    the first block, where it opens most heads; 0.5 halves the gain for no
    measurable benefit anywhere. A CPU simulation of the QK side alone puts
    a ~3% cost on the first block at 0.2, which the real kernel's Q rounding
    and fp8 PV error dilute to nothing.
    Under GQA the factor is per kv head, with Q's rms taken over the head
    group. The channel rms are accumulated in fp32 by `vector_norm` without
    materializing an fp32 copy, which is what keeps this cheap on a 24 GB
    card at H3's lengths. Returns (f for q, 1/f for k), fp32 [B, H_kv, C].
    """
    if tensor_layout == "HND":
        seq_dim = 2
        b, h_q, _, c = q.shape
        h_kv = k.shape[1]
    else:
        seq_dim = 1
        b, _, h_q, c = q.shape
        h_kv = k.shape[2]
    # Reducing over the sequence axis leaves [B, H, C] in either layout.
    rk = torch.linalg.vector_norm(k, dim=seq_dim, dtype=torch.float32)
    rq = torch.linalg.vector_norm(q, dim=seq_dim, dtype=torch.float32)
    if h_q != h_kv:
        rq = rq.reshape(b, h_kv, h_q // h_kv, c).pow(2).mean(dim=2).sqrt()
    rk = rk.clamp(min=1e-6)
    rq = rq.clamp(min=1e-6)
    f = rk.pow(alpha) / rq.pow(1.0 - alpha)
    f = f / torch.exp(torch.log(f).mean(dim=-1, keepdim=True))
    e = rk.pow(2)
    share = torch.topk(e, 4, dim=-1).values.sum(dim=-1) / e.sum(dim=-1)   # [B, H_kv]
    f = torch.where((share >= min_share)[..., None], f, torch.ones_like(f))
    return f.contiguous(), (1.0 / f).contiguous()


def per_thread_int8(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, sm_scale=None, tensor_layout="HND",
                    qk_balance=False, balance_alpha=0.5, balance_min_share=0.2):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if qk_balance:
        fq, fk = qk_balance_factor(q, k, tensor_layout, balance_alpha, balance_min_share)
    else:
        fq = fk = q_int8   # unused placeholder pointer when BALANCE is off

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # int64 addressing only where the offsets actually need it: past 2**31
    # elements the int32 arithmetic wraps, which corrupts the tail silently
    # in NHD and faults in HND. Specializing keeps ordinary shapes on the
    # cheaper int32 path.
    # Check the int8 outputs too: a broadcast or otherwise oddly-strided
    # input can have a smaller bound than the contiguous tensor written back.
    q_i64 = needs_int64_offsets(q, q_int8, tensor_layout=tensor_layout, blk=BLKQ)
    k_i64 = needs_int64_offsets(k, k_int8, tensor_layout=tensor_layout, blk=BLKK)

    f_sz, f_sh = (fq.stride(0), fq.stride(1)) if qk_balance else (0, 0)
    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        fq, f_sz, f_sh,
        C=head_dim, BLK=WARPQ, USE_I64=q_i64, BALANCE=qk_balance, GROUP=h_qo // h_kv
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        fk, f_sz, f_sh,
        C=head_dim, BLK=WARPK, USE_I64=k_i64, BALANCE=qk_balance
    )

    return q_int8, q_scale, k_int8, k_scale