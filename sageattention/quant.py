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
from typing import Any, List, Literal, Optional, Tuple, Union

from . import _fused
from .triton._int_offsets import max_element_offset

# Width of the global element offsets the installed fused CUDA build can form.
# Upstream's kernels (csrc/fused/fused.cu) carried uint32_t strides and summed
# them in uint32_t, so a tensor past 2**32 elements read from the wrong
# address with no error; on a fused-QKV view at MiniMax H3's config that is
# ~199,729 rows. This fork widened the strides to int64 and stamps the width on
# the module, so a build that predates the change -- another interpreter's
# stale .so in this checkout, or a checkout from before the fix -- reports 32
# here and is refused at the host instead of trusted.
ELEMENT_OFFSET_BITS: int = getattr(_fused, "ELEMENT_OFFSET_BITS", 32)


def _check_element_offsets(named, tensor_layout, blk, offset_bits=None):
    """Refuse a launch whose largest element offset the kernels cannot form.

    `named` is an iterable of (name, tensor) in `tensor_layout`. The kernels
    build pointers for the whole padded grid before masking the load, so the
    row index runs to `ceil(seq/blk)*blk - 1`; `blk` is the launch's block
    size. `offset_bits` overrides the installed build's width, for tests.
    """
    bits = ELEMENT_OFFSET_BITS if offset_bits is None else offset_bits
    ceiling = 1 << bits
    seq_axis = 1 if tensor_layout == "NHD" else 2
    for name, t in named:
        off = max_element_offset(t, tensor_layout, blk)
        if off < ceiling:
            continue
        rows = t.shape[seq_axis]
        stride_seq = t.stride(seq_axis)
        padded_rows = (rows + blk - 1) // blk * blk
        fixed = off - (padded_rows - 1) * stride_seq
        rows_ceiling = (ceiling - 1 - fixed) // stride_seq + 1
        raise ValueError(
            f"{name}: {rows:,} rows at stride_seq={stride_seq:,} form element "
            f"offsets up to {off:,}, past the {bits}-bit ceiling "
            f"({ceiling:,}) of the installed fused CUDA quant kernels -- about "
            f"{rows_ceiling:,} rows at this stride. Rebuild sage with "
            f"./build.sh for the 64-bit build, or shorten the sequence; the "
            f"kernels would otherwise read from the wrong address silently."
        )


def _check_contiguous_numel(name, t, offset_bits=None):
    """Same refusal for a contiguous intermediate the kernels index to numel."""
    bits = ELEMENT_OFFSET_BITS if offset_bits is None else offset_bits
    ceiling = 1 << bits
    if t.numel() >= ceiling:
        raise ValueError(
            f"{name}: {t.numel():,} elements, past the {bits}-bit ceiling "
            f"({ceiling:,}) of the installed fused CUDA quant kernels. Rebuild "
            f"sage with ./build.sh for the 64-bit build, or shorten the sequence."
        )


def per_block_int8(
    q: torch.Tensor, 
    k: torch.Tensor, 
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128, 
    BLKK: int = 64, 
    sm_scale: Optional[float] = None, 
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` and the key tensor `k` with per block quantization.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    sm_scale : Optional[float]
        The scale factor for the softmax operation. Default is ``head_dim**-0.5``. 
        It will be multiplied by ``1.44269504`` to work together with the triton attention kernel.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    
    sm_scale *= 1.44269504

    _check_element_offsets([("q", q), ("q_int8", q_int8)], tensor_layout, BLKQ)
    _check_element_offsets([("k", k), ("k_int8", k_int8)], tensor_layout, BLKK)
    _fused.quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale

def per_warp_int8(
    q: torch.Tensor, 
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int =128,
    WARPQ: int =32,
    BLKK: int =64,
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` with per warp quantization and the key tensor `k` with per block quantization.
    Warp size of quantizing `q` is 16 or 32, with a block size of 64 or 128.
    Block size of quantizing `k` is 64 or 128.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ)]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    _check_element_offsets([("q", q), ("q_int8", q_int8)], tensor_layout, BLKQ)
    _check_element_offsets([("k", k), ("k_int8", k_int8)], tensor_layout, BLKK)
    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)
    
    return q_int8, q_scale, k_int8, k_scale

def sub_mean(
    v: torch.Tensor, 
    tensor_layout: str ="HND"
):
    """
    Calculate the mean of the tensor `v` along the sequence length dimension and subtract it from `v`. Result is stored as fp16.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - The tensor `v_smoothed` with the mean subtracted and stored as fp16. Shape: Same as `v` with `float16` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with dtype same as `v`.

    Note
    ----
    - The tensors `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - The returned tensor `v_smoothed` will have dtype ``torch.float16`` regardless of the input dtype.
    - The returned mean tensor will have the same dtype as the input tensor.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    vm = v.mean(dim=1 if _tensor_layout == 0 else 2)

    v_smoothed = torch.empty(v.shape, dtype=torch.float16, device=v.device)
    
    # subtract mean and store the result as fp16
    # SubMeanKernel pads to 64 rows at head_dim 128 and 128 otherwise; 128
    # is the conservative bound for both.
    _check_element_offsets([("v", v), ("v_smoothed", v_smoothed)], tensor_layout, 128)
    _fused.sub_mean_cuda(v, vm, v_smoothed, _tensor_layout)

    return v_smoothed, vm

def per_channel_fp8(
    v: torch.Tensor,
    tensor_layout: str ="HND",
    scale_max: float = 448.0,
    smooth_v: bool = True
):
    """
    Transpose, pad and permute the tensor `v` and quantize it to fp8 with per channel quantization.
    `v` is first transposed along the head dimension and the sequence length dimension, then padded to a multiple of 64.
    After that, the tensor is permuted along the sequence length dimension by ``[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]``.
    The quantization is done per channel, with the scale value and smooth factor calculated per channel.

    Parameters
    ----------
    v : torch.Tensor
        The input tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    scale_max : float
        The maximum scale value for the quantization. Default is 448.0 (upper bound of E4M3 data format).

    smooth_v : bool
        Whether to smooth the quantized tensor. Default is True.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        A tuple containing:
        - The quantized tensor `v_fp8`. Shape:
            - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, head_dim, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
            - If `tensor_layout` is "NHD": ``[batch_size, head_dim, num_kv_heads, (kv_len + 63) // 64 * 64]``, with `float8_e4m3fn` dtype.
        - The scale tensor of `v`. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.
        - The mean tensor of `v` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]`` with `float32` dtype.

    Note
    ----
    - The tensors `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - The returned mean tensor will be None if `smooth_v` is False. Otherwise it will have dtype ``torch.float32``.
    """

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype, device=v.device)
    
    # TransposePadPermuteKernel's CTA covers 64 rows; the transposed buffer
    # is contiguous and indexed to its numel by the scale kernels after it.
    _check_element_offsets([("v", v)], tensor_layout, 64)
    _check_contiguous_numel("v_transposed_permutted", v_transposed_permutted)
    _fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)

    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)

    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    vm = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)

    if smooth_v:
        _fused.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        _fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None



    
