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


# ---------------------------------------------------------------------------
# qk_rotate: a fixed orthogonal rotation of every q and k row, inside the
# quantizer, before the INT8 rounding.
#
# One scale covers a group of rows across all 128 head channels, so a loud
# channel or a single spiking token spends the INT8 range and starves the
# rest. R = diag(signs) @ H128 / sqrt(128) spreads each row's energy over every
# channel first, and (qR).(kR) == q.k because R is orthogonal, so the attention
# math is unchanged and only the rounding sees the difference. Measured on
# MiniMax H3 captures by tests/spikes/spike_h3_qk_rotation.py (CHANGELOG,
# Workload intel, 2026-09-17): about half the block-49 error, where
# `qk_balance` removes about a third, a few percent better everywhere else, and
# with rotation on the balance factor changes nothing.
#
# The matrix is the one comfy-kitchen's rotated INT8 kernels use (the same four
# sign words, Sylvester order), so the three kernels rotate identically. It is
# fused here rather than applied by the caller because a separate pass costs a
# round trip through memory and a bf16 re-rounding of the rotated tensor, which
# the spike measured; in the kernel the rotated values exist only in fp32
# between the load and the rounding. The butterfly is written as
# reshape/split/join stages, the form vLLM's fused FWHT+quant kernel uses, so
# Triton emits no matmul. Head dim 128 only.
# ---------------------------------------------------------------------------
_ROT_SIGN_WORDS = (0x1035997B, 0x8087F5EE, 0xEE2E4E1A, 0x71132418)
_ROT_DIM = 128
_rot_sign_cache = {}


def qk_rotate_signs(device):
    """[128] float32 of +1/-1: channel d is +1 when bit (d & 31) of word d >> 5 is set."""
    key = str(device)
    if key not in _rot_sign_cache:
        bits = [(_ROT_SIGN_WORDS[d >> 5] >> (d & 31)) & 1 for d in range(_ROT_DIM)]
        _rot_sign_cache[key] = torch.tensor([1.0 if b else -1.0 for b in bits],
                                            dtype=torch.float32, device=device)
    return _rot_sign_cache[key]


def qk_rotation_matrix(device, dtype=torch.float32):
    """The rotation as a matrix, x @ R; the oracle the kernel is tested against."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < _ROT_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    r = qk_rotate_signs("cpu").double()[:, None] * h / _ROT_DIM ** 0.5
    return r.to(device=device, dtype=dtype)


@triton.jit
def _fwht128_stage(x, ROWS: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x4 = tl.reshape(x, (ROWS, GROUPS, 2, STRIDE))
    x4 = tl.trans(x4, 0, 1, 3, 2)
    a, b = tl.split(x4)
    x4 = tl.join(a + b, a - b)
    x4 = tl.trans(x4, 0, 1, 3, 2)
    return tl.reshape(x4, (ROWS, 128))


@triton.jit
def _rotate128(x, sign, ROWS: tl.constexpr):
    """x [ROWS, 128] fp32 -> x @ R, R = diag(sign) @ H128 / sqrt(128)."""
    x = x * sign[None, :]
    x = _fwht128_stage(x, ROWS, 64, 1)
    x = _fwht128_stage(x, ROWS, 32, 2)
    x = _fwht128_stage(x, ROWS, 16, 4)
    x = _fwht128_stage(x, ROWS, 8, 8)
    x = _fwht128_stage(x, ROWS, 4, 16)
    x = _fwht128_stage(x, ROWS, 2, 32)
    x = _fwht128_stage(x, ROWS, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def quant_query_per_thread_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        Factor, stride_fz, stride_fh, Sign,
                                        C: tl.constexpr, BLK: tl.constexpr,
                                        USE_I64: tl.constexpr = False,
                                        BALANCE: tl.constexpr = False,
                                        GROUP: tl.constexpr = 1,
                                        ROTATE: tl.constexpr = False):
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

    if ROTATE:
        # Rows past L must be zeros, not whatever a masked load leaves: the
        # rotation is per row, but the scale below is a max over all of them.
        x = tl.load(input_ptrs, mask=offs_n[:, None] < L, other=0.0)
    else:
        x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    if BALANCE:
        # q . k == (q * f) . (k / f): the factor moves INT8 resolution from
        # K's loud channels onto Q at no cost to the attention math.
        x = x * f[None, :]
    if ROTATE:
        x = _rotate128(x, tl.load(Sign + tl.arange(0, C)), BLK // 8)
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
                                        Factor, stride_fz, stride_fh, Sign,
                                        C: tl.constexpr, BLK: tl.constexpr,
                                        USE_I64: tl.constexpr = False,
                                        BALANCE: tl.constexpr = False,
                                        ROTATE: tl.constexpr = False):
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

    if ROTATE:
        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L, other=0.0)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L, other=0.0)
    else:
        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    if BALANCE:
        x0 = x0 * f[None, :]
        x1 = x1 * f[None, :]
    if ROTATE:
        sign = tl.load(Sign + tl.arange(0, C))
        x0 = _rotate128(x0, sign, BLK // 8)
        x1 = _rotate128(x1, sign, BLK // 8)
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
                    qk_balance=False, balance_alpha=0.5, balance_min_share=0.2, qk_rotate=False):
    if qk_rotate and q.shape[-1] != _ROT_DIM:
        raise ValueError(f"qk_rotate needs head_dim {_ROT_DIM}, got {q.shape[-1]}")
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
    sign = qk_rotate_signs(q.device) if qk_rotate else q_int8   # placeholder pointer when off
    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        fq, f_sz, f_sh, sign,
        C=head_dim, BLK=WARPQ, USE_I64=q_i64, BALANCE=qk_balance, GROUP=h_qo // h_kv,
        ROTATE=qk_rotate
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    quant_key_per_thread_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        fk, f_sz, f_sh, sign,
        C=head_dim, BLK=WARPK, USE_I64=k_i64, BALANCE=qk_balance, ROTATE=qk_rotate
    )

    return q_int8, q_scale, k_int8, k_scale