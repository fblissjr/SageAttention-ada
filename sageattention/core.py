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
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.quant_per_block import per_block_int8_q as per_block_int8_q_triton
from .triton.quant_per_block import per_block_int8_k as per_block_int8_k_triton
from .triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
from .triton.attn_qk_int8_per_block import forward as attn_false
from .triton.attn_qk_int8_per_block_causal import forward as attn_true
from .triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
from .triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

try:
    from . import sm80_compile
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from . import sm89_compile
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

from .quant import per_block_int8 as per_block_int8_cuda
from .quant import per_warp_int8 as per_warp_int8_cuda
from .quant import sub_mean
from .quant import per_channel_fp8

from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union
import threading
import sys
import warnings


# Stable short names for the kernel dispatched by each public sageattn*
# entry point. Consumers can match these strings directly against
# get_last_dispatched_kernel() output without mirroring sage's routing
# table. Adding a new variant requires a new constant; renaming an
# existing constant is a breaking change for consumers.
KERNEL_FP16_TRITON = "fp16_triton"
KERNEL_FP16_CUDA = "fp16_cuda"               # pv_accum_dtype="fp32"
KERNEL_FP16_CUDA_FP16 = "fp16_cuda(fp16)"    # pv_accum_dtype="fp16"
KERNEL_FP16_CUDA_PP = "fp16_cuda++"          # pv_accum_dtype="fp16+fp32"
KERNEL_FP8_CUDA = "fp8_cuda"                 # pv_accum_dtype="fp32"
KERNEL_FP8_CUDA_FP32 = "fp8_cuda(fp32+fp32)" # pv_accum_dtype="fp32+fp32"
KERNEL_FP8_CUDA_PP = "fp8_cuda++"            # pv_accum_dtype="fp32+fp16" (SageAttention2++)
KERNEL_VARLEN_TRITON = "varlen_triton"

KernelName = Literal[
    "fp16_triton",
    "fp16_cuda",
    "fp16_cuda(fp16)",
    "fp16_cuda++",
    "fp8_cuda",
    "fp8_cuda(fp32+fp32)",
    "fp8_cuda++",
    "varlen_triton",
]

KNOWN_KERNEL_NAMES = frozenset({
    KERNEL_FP16_TRITON,
    KERNEL_FP16_CUDA,
    KERNEL_FP16_CUDA_FP16,
    KERNEL_FP16_CUDA_PP,
    KERNEL_FP8_CUDA,
    KERNEL_FP8_CUDA_FP32,
    KERNEL_FP8_CUDA_PP,
    KERNEL_VARLEN_TRITON,
})

# Thread-local: each thread sees only its own dispatches. CUDA work is
# synchronous and doesn't yield, so this is the right primitive for the
# (call sageattn -> read kernel name) pattern. Don't read across an
# `await`; use contextvars if asyncio support is ever needed.
_dispatch_state = threading.local()


def _record_dispatch(name: KernelName) -> None:
    _dispatch_state.last = name
    counts = getattr(_dispatch_state, "counts", None)
    if counts is None:
        counts = _dispatch_state.counts = {}
    counts[name] = counts.get(name, 0) + 1


def get_last_dispatched_kernel() -> Optional[KernelName]:
    """Return the kernel-name string of the most recent sageattn* call
    on this thread, or None if no call has happened yet on this thread.

    Stable values are listed in `KNOWN_KERNEL_NAMES`. Read this value
    immediately after the sage call -- if your code yields (asyncio,
    or another sage call from the same thread) between the call and
    the read, the value can be overwritten.
    """
    return getattr(_dispatch_state, "last", None)


def get_dispatch_counts() -> dict:
    """Per-kernel dispatch counts for this thread, as a snapshot copy.

    `get_last_dispatched_kernel` answers "is sage reachable on this path".
    This answers "how much of the work actually reached sage", which is a
    different question and the one a caller needs to claim end-to-end
    coverage: read before a render, read after, subtract, and compare the
    total against the number of attention calls the caller made. A
    consumer-side fallback that logs once cannot distinguish one degraded
    call from every call, so the caller's own count is the other half --
    this number alone proves sage ran, never that nothing else did.

    Counts are monotonic and there is deliberately no reset: differencing
    two snapshots needs no clear-the-state contract, and leaves no window
    where a concurrent call is lost to someone else's reset.

    Thread-local, like `get_last_dispatched_kernel` -- read it on the thread
    that made the calls. A reader on another thread sees that thread's
    counts, which for a fresh thread is `{}` rather than an error.
    """
    return dict(getattr(_dispatch_state, "counts", {}))


def _reset_dispatch_for_test() -> None:
    """Test-only: clear this thread's dispatch state so the next read
    returns None. Not part of the public API."""
    if hasattr(_dispatch_state, "last"):
        del _dispatch_state.last


# Session-start routing log: one stderr line per unique routing
# tuple, dedup'd process-wide. Module-level (not thread-local) so a
# multi-thread consumer sees one log per tuple regardless of which
# thread called first; matches "session start" semantics.
_CUDA_VERSION_AT_IMPORT = torch.version.cuda or "unknown"
_seen_routing_tuples: set[tuple[str, bool, str, str]] = set()


def _log_routing_choice_once(
    arch: str,
    mask_present: bool,
    pv_accum_dtype: str,
    kernel_name: str,
) -> None:
    key = (arch, mask_present, pv_accum_dtype, kernel_name)
    if key in _seen_routing_tuples:
        return
    _seen_routing_tuples.add(key)
    sys.stderr.write(
        f"[INFO] sage routing: arch={arch} cuda={_CUDA_VERSION_AT_IMPORT} "
        f"mask={mask_present} pv_accum={pv_accum_dtype} -> {kernel_name}\n"
    )


def _reset_routing_log_for_test() -> None:
    """Test-only: clear the seen-routing-tuples set so the next
    sageattn() call re-emits its routing line. Not part of the
    public API."""
    _seen_routing_tuples.clear()


def _warn_if_mask_passed_to_cuda_kernel(kwargs: dict, kernel_label: str) -> None:
    """Soft-warn when a consumer hand-picks a CUDA kernel and passes a
    non-None `attn_mask`. The CUDA kernels in this lineage silently
    drop masks (pybind layer never wires them through; the C++
    `MaskMode` enum only handles `{kNone, kCausal}`), so the call
    runs unmasked and produces numerically wrong output. The
    dispatcher `sageattn()` routes masked calls to the Triton kernel
    automatically as of v0.3.0; this guard catches consumers that
    bypass the dispatcher and pick a `_cuda` kernel directly with a
    mask. Soft warning (not raise) so consumers that defensively pass
    `attn_mask=None` aren't penalized -- the warn fires only when the
    mask is real. Python's default warning filter dedupes by source
    line, so a long iteration loop emits one warning total per
    process+location. Reference: internal/audit_2026-04-26.md."""
    mask = kwargs.get("attn_mask")
    if mask is not None:
        # Frames: warnings.warn (1) -> _warn_if_mask_passed_to_cuda_kernel (2)
        # -> the per-kernel wrapper body (3) -> the consumer's call site (4).
        warnings.warn(
            f"{kernel_label}: attn_mask was passed but this kernel does "
            f"not implement masked attention -- the mask is silently "
            f"dropped and the output is numerically wrong. Use "
            f"sageattn_qk_int8_pv_fp16_triton for masked calls, or call "
            f"sageattn() and let the dispatcher route by mask presence.",
            stacklevel=4,
        )


def get_cuda_version():
    version = torch.version.cuda
    major, minor = version.split('.')
    return int(major), int(minor)


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


# Currently get_cuda_arch_versions cannot be traced by torch.compile
_cuda_archs = get_cuda_arch_versions()

# Archs whose fp8 kernels take `sageattn_consume`'s early-release path. Kept
# as one constant because `sageattn_consume_prefers_cloned_v` answers for it
# too, and a second copy would let the predicate describe a dispatch that has
# moved on -- which reaches a caller as a memory regression, not an error.
_EARLY_RELEASE_ARCHS = frozenset({"sm89", "sm100", "sm120", "sm121"})


def _has_native_mask_kernel(arch: str) -> bool:
    """Whether this arch has a mask-correct CUDA kernel (v0.5.5, fp8++).

    One function rather than one condition written twice. Both `sageattn`
    and `sageattn_consume` decide masked routing on this, and before v0.7.6
    only `sageattn` decided it at all, so a masked consume call took a
    mask-dropping kernel on sm89 with CUDA < 12.8 and the native-mask path
    on archs the dispatcher deliberately sends to Triton. Copying the
    condition into the second caller would have fixed today's divergence
    and left the next one available: two call sites that agree by
    inspection drift the moment one is edited, which is how the first one
    happened.
    """
    return arch == "sm89" and get_cuda_version() >= (12, 8)


def _resolve_cuda_index(device) -> int:
    """Device index for `None` / int / str / `torch.device`, or ValueError."""
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        index = device
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(
                f"expected a CUDA device, got {device!r}. This asks about a "
                f"specific GPU's kernels; there is no answer for a CPU "
                f"tensor. If you are calling at model-patch time, the model "
                f"may not be on its device yet -- ask in the forward instead."
            )
        index = (
            torch.cuda.current_device() if resolved.index is None
            else resolved.index
        )
    if not 0 <= index < len(_cuda_archs):
        raise ValueError(
            f"CUDA device index {index} out of range; this machine has "
            f"{len(_cuda_archs)} device(s)"
        )
    return index


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

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

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - If `attn_mask` is passed (non-None), the routing depends on arch.
      On sm89 + CUDA >= 12.8, masked calls land on
      `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`
      (the v0.5.5 native CUDA general-mask path). On other archs, masked
      calls still route to `sageattn_qk_int8_pv_fp16_triton` because
      those CUDA kernels haven't gained mask support yet. `is_causal=True`
      is unaffected and continues to dispatch by arch (CUDA kernels
      handle causal mode natively via `MaskMode::kCausal`).
    """

    # attn_mask is a named parameter rather than a **kwargs entry because
    # ComfyUI gates masked calls on `"attn_mask" in
    # inspect.signature(sageattn).parameters` (comfy/ldm/modules/attention.py)
    # and routes everything masked to torch SDPA when that reads False --
    # which made the v0.5.5 native CUDA mask path unreachable in production.
    arch = _cuda_archs[q.device.index]

    if attn_mask is not None:
        # Native CUDA mask only on sm89 + CUDA >= 12.8 today. Other archs
        # still need the Triton fallback for mask correctness.
        if _has_native_mask_kernel(arch):
            kwargs.setdefault("pv_accum_dtype", "fp32+fp16")
            _log_routing_choice_once(arch, True, "fp32+fp16", KERNEL_FP8_CUDA_PP)
            return sageattn_qk_int8_pv_fp8_cuda(
                q, k, v,
                tensor_layout=tensor_layout, is_causal=is_causal,
                sm_scale=sm_scale, return_lse=return_lse,
                attn_mask=attn_mask, **kwargs,
            )
        _log_routing_choice_once(arch, True, "n/a", KERNEL_FP16_TRITON)
        return sageattn_qk_int8_pv_fp16_triton(
            q, k, v,
            tensor_layout=tensor_layout, is_causal=is_causal,
            attn_mask=attn_mask, sm_scale=sm_scale, return_lse=return_lse,
            **kwargs,
        )

    # Use setdefault so any kernel-specific kwarg the consumer passed
    # explicitly (e.g. pv_accum_dtype, qk_quant_gran) wins over the
    # dispatcher's default. Without this, dispatcher-set kwargs would
    # collide with the same key in **kwargs and raise TypeError on
    # forwarding -- a bug introduced by the v0.3.1 kwargs-forwarding
    # change. Override-wins matches Python's standard "caller-explicit
    # beats callee-default" convention.
    if arch == "sm75":
        _log_routing_choice_once(arch, False, "n/a", KERNEL_FP16_TRITON)
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, **kwargs)
    elif arch in {"sm80", "sm86", "sm87"}:
        kwargs.setdefault("pv_accum_dtype", "fp32")
        _log_routing_choice_once(arch, False, kwargs["pv_accum_dtype"], KERNEL_FP16_CUDA)
        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, **kwargs)
    elif arch == "sm89":
        if get_cuda_version() < (12, 8):
            kwargs.setdefault("pv_accum_dtype", "fp32+fp32")
            _log_routing_choice_once(arch, False, kwargs["pv_accum_dtype"], KERNEL_FP8_CUDA_FP32)
        else:
            # SageAttention2++
            kwargs.setdefault("pv_accum_dtype", "fp32+fp16")
            _log_routing_choice_once(arch, False, kwargs["pv_accum_dtype"], KERNEL_FP8_CUDA_PP)
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, **kwargs)
    elif arch in {"sm100", "sm120", "sm121"}:
        # Looks superficially mergeable with the sm89 branch but isn't:
        # this branch sets qk_quant_gran=per_warp; sm89 leaves it at the
        # kernel's per_thread default. Don't merge without re-grading
        # rtol on whichever branch you change.
        if get_cuda_version() < (12, 8):
            # sm120 has accurate fp32 accumulator for fp8 mma and triton kernel is currently not usable on sm120.
            kwargs.setdefault("pv_accum_dtype", "fp32")
            _log_routing_choice_once(arch, False, kwargs["pv_accum_dtype"], KERNEL_FP8_CUDA)
        else:
            kwargs.setdefault("pv_accum_dtype", "fp32+fp16")
            _log_routing_choice_once(arch, False, kwargs["pv_accum_dtype"], KERNEL_FP8_CUDA_PP)
        kwargs.setdefault("qk_quant_gran", "per_warp")
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, **kwargs)
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool =False, 
    attn_mask: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, FP16 PV with FP16 accumulation, implemented using Triton.
    The FP16 accumulator is added to a FP32 buffer immediately after each iteration.

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

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    attn_mask : Optional[torch.Tensor]
        The attention mask tensor, of dtype bool or float32.
        Should be able to broadcast to the shape of the matrix qk^T.
        Default: None.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == q.dtype, "attn_mask must be of dtype bool or the same dtype as q."
        assert attn_mask.device == q.device, "All tensors must be on the same device."

    _record_dispatch(KERNEL_FP16_TRITON)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2
    nh_dim = 2 if tensor_layout == "NHD" else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    if is_causal:
        assert attn_mask is None, "Mask should be None for causal attention."
        o, lse = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)
    else:
        if attn_mask is not None:
            if tensor_layout == "HND":
                target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
            elif tensor_layout == "NHD":
                target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
            else:
                raise ValueError(f"tensor_layout {tensor_layout} not supported")
            try:
                attn_mask = attn_mask.expand(target_shape)
            except Exception:
                raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=attn_mask, return_lse=return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_partitioned(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slices: Sequence[Tuple[int, int, Optional[torch.Tensor]]],
    tensor_layout: str = "HND",
    smooth_k: bool = True,
    sm_scale: Optional[float] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run sage Triton attention over multiple Q slices sharing K and V.

    Each slice is `(q_start, q_end, attn_mask | None)`. K is quantized
    once, V is cast to fp16 once, and the output is allocated once;
    only Q is re-quantized per slice (Q changes per slice, so there's
    no amortization to gain there). The inner Triton kernel writes
    each slice's output into a view of the pre-allocated full output.

    Targets multi-slice Q partition patterns (e.g. LTX 2.3 guide-mask
    workflows that split Q into noisy + tracked groups and dispatch
    per-group) where stock sageattn() called N times with the same K, V
    would re-quantize K and re-cast V every call. See
    tests/bench/partitioned_mask_phase0/ for the peak-HBM
    characterization.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Same shape/dtype/layout constraints as
        `sageattn_qk_int8_pv_fp16_triton`. q.dtype must be fp16 or bf16;
        bf16 V is cast to fp16 internally (matching the existing entry).
    slices : sequence of (q_start, q_end, attn_mask | None)
        Q-range and optional mask per call. Slices may overlap or have
        gaps; only the rows covered by some slice get written into the
        output (rows outside any slice contain uninitialized memory
        from torch.empty -- callers covering all of Q is the expected
        usage).
    tensor_layout : "HND" | "NHD"
    smooth_k : bool
        Subtract per-head K-mean before quantization (improves accuracy).
    sm_scale : float, optional
        Defaults to 1 / sqrt(head_dim).
    output_dtype : torch.dtype
        Output dtype. Must be fp16 today.

    Returns
    -------
    torch.Tensor
        Full-Q-shaped output. Same shape as q (modulo head_dim padding
        being stripped to the original head_dim).

    Notes
    -----
    - Masked slices stay on the only-mask-correct Triton path, mirroring
      the dispatcher's masked-call routing.
    - Records dispatch as `fp16_triton` per call -- the underlying kernel
      is the same. Consumers wanting partitioned-vs-non distinction can
      check entry-point identity instead of dispatch telemetry.
    """
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert k.dtype == dtype and v.dtype == dtype, "q, k, v must have the same dtype."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert output_dtype == torch.float16, "Only fp16 output is supported today."
    assert len(slices) >= 1, "slices must be non-empty"

    _record_dispatch(KERNEL_FP16_TRITON)

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        pad = 64 - head_dim_og
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
    elif head_dim_og > 64 and head_dim_og < 128:
        pad = 128 - head_dim_og
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
    else:
        km = None

    if dtype == torch.bfloat16:
        v = v.to(torch.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    # Amortized across slices: K-quant once, V-cast once (above), output
    # buffer once. Q-quant stays per-slice because Q changes per slice.
    k_int8, k_scale = per_block_int8_k_triton(k, km=km, tensor_layout=tensor_layout)
    full_out = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    for q_start, q_end, attn_mask in slices:
        assert 0 <= q_start < q_end <= q.size(seq_dim), f"invalid slice ({q_start}, {q_end}) for seq_len {q.size(seq_dim)}"

        if tensor_layout == "HND":
            q_slice = q[:, :, q_start:q_end, :].contiguous()
            out_slice = full_out[:, :, q_start:q_end, :]
        else:
            q_slice = q[:, q_start:q_end, :, :].contiguous()
            out_slice = full_out[:, q_start:q_end, :, :]

        q_int8, q_scale = per_block_int8_q_triton(q_slice, sm_scale=sm_scale, tensor_layout=tensor_layout)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool or attn_mask.dtype == dtype, "attn_mask must be bool or match q.dtype"
            assert attn_mask.device == q.device
            if tensor_layout == "HND":
                target_shape = (q.shape[0], q.shape[1], q_end - q_start, k.shape[2])
            else:
                target_shape = (q.shape[0], q.shape[2], q_end - q_start, k.shape[1])
            try:
                attn_mask = attn_mask.expand(target_shape)
            except Exception:
                raise AssertionError(f"attn_mask shape {attn_mask.shape} cannot be broadcast to {target_shape}")

        attn_false(
            q_int8, k_int8, v, q_scale, k_scale,
            tensor_layout=tensor_layout,
            output_dtype=output_dtype,
            attn_mask=attn_mask,
            return_lse=False,
            out=out_slice,
        )

    full_out = full_out[..., :head_dim_og]
    return full_out


def sageattn_varlen(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    is_causal: bool = False,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.
    
    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.
    
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _record_dispatch(KERNEL_VARLEN_TRITON)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.mean(dim=0, keepdim=True) # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen_triton(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    o = o[..., :head_dim_og]

    return o


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

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

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert SM80_ENABLED, "SM80 kernel is not available. make sure you GPUs with compute capability 8.0 or higher."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _warn_if_mask_passed_to_cuda_kernel(kwargs, "sageattn_qk_int8_pv_fp16_cuda")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64, WARPK=64)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32':
        _record_dispatch(KERNEL_FP16_CUDA)
        v = v.to(torch.float16)
        lse = sm80_compile.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16":
        _record_dispatch(KERNEL_FP16_CUDA_FP16)
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = sm80_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            v = v.to(torch.float16)
            lse = sm80_compile.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16+fp32":
        _record_dispatch(KERNEL_FP16_CUDA_PP)
        v = v.to(torch.float16)
        lse = sm80_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    _qkv_box: Optional[list] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

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

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """

    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # v0.5.5: native CUDA general-mask support on the fp32+fp16 (fp8++) variant.
    # The other pv_accum_dtype variants still silently drop masks and emit the
    # warn-on-misuse; preserves the v0.3.1 safety net there.
    attn_mask = kwargs.pop("attn_mask", None)
    if attn_mask is not None and pv_accum_dtype != "fp32+fp16":
        # Same soft-warn shape as _warn_if_mask_passed_to_cuda_kernel; can't
        # use it here because we already popped attn_mask out of kwargs.
        warnings.warn(
            f"sageattn_qk_int8_pv_fp8_cuda: attn_mask was passed but "
            f"pv_accum_dtype={pv_accum_dtype!r} doesn't yet support it. "
            f"Use pv_accum_dtype='fp32+fp16' (the dispatcher default on sm89), "
            f"or sageattn_qk_int8_pv_fp16_triton. Falling back to no-mask "
            f"semantics; results will be numerically wrong if the mask is non-trivial.",
            stacklevel=2,
        )
        attn_mask = None

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1    

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            # nheads_k => nheads_q
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    # Mask validation and its broadcast target shape read q and k, so they
    # have to happen while those are alive: the _qkv_box path below releases
    # them, and reading afterwards raised UnboundLocalError on every masked
    # consume call from v0.7.0 to v0.7.5. Only the two values needed later
    # are carried across, and deliberately as values rather than tensors --
    # the bool conversion allocates a full dtype-sized mask, so doing it
    # here would stack that on top of the still-live floats and raise the
    # very peak this entry point exists to lower.
    # attn_mask is already None unless pv_accum_dtype is fp32+fp16; the
    # other variants warn and drop it above.
    mask_target_shape = mask_device = None
    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == dtype, "attn_mask must be bool or match q dtype"
        assert attn_mask.device == q.device, "attn_mask must be on the same device"
        assert not is_causal, "attn_mask + is_causal is not supported; choose one"
        # Mirror the Triton path's expand semantics so the kernel sees a
        # full (B, H, qo_len, kv_len) view with stride-0 dims handling
        # broadcasts at zero memory cost.
        if _tensor_layout == 1:  # HND
            mask_target_shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        else:  # NHD
            mask_target_shape = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
        mask_device = q.device

    # Output is q-shaped, but allocating it here would stack on top of the
    # still-live float q/k/v. Capture the shape and allocate after those are
    # released instead -- see the _qkv_box path below.
    o_shape = q.size()
    if _qkv_box is not None:
        del q, k, km
        _qkv_box[0] = _qkv_box[1] = None

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32+fp16' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
        smooth_v = False

    quant_v_scale_max = 448.0
    if pv_accum_dtype == 'fp32+fp16':
        quant_v_scale_max = 2.25

    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v)

    if _qkv_box is not None:
        del v
        _qkv_box[2] = None

    o = torch.empty(o_shape, dtype=dtype, device=q_int8.device)

    if pv_accum_dtype == "fp32":
        _record_dispatch(KERNEL_FP8_CUDA)
        if smooth_v:
            lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        _record_dispatch(KERNEL_FP8_CUDA_FP32)
        lse = sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp16":
        _record_dispatch(KERNEL_FP8_CUDA_PP)
        if attn_mask is not None:
            # Validated above against q and k; converted here, after the
            # floats are gone, because this allocation is dtype-sized.
            if attn_mask.dtype == torch.bool:
                # Translate bool -> additive log-weights (0 / -inf) in q's
                # dtype. The kernel adds these to the scores before the
                # softmax max.
                attn_mask = torch.where(
                    attn_mask, torch.zeros((), dtype=dtype, device=mask_device),
                    torch.full((), float("-inf"), dtype=dtype, device=mask_device),
                )
            attn_mask = attn_mask.expand(mask_target_shape)
        lse = sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
            _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse,
            attn_mask=attn_mask,
        )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


def sageattn_consume(
    qkv: list,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    **kwargs: Any,
):
    """`sageattn()` that takes ownership of `qkv` to cut peak VRAM.

    Attention is the memory peak in a large DiT block, and most of that
    peak is not sage's working set -- it is the caller's float q/k/v
    sitting alive underneath it. A normal `sageattn(q, k, v)` call cannot
    do anything about that: the caller's frame still references those
    tensors, so nothing sage does internally can release them.

    This entry point takes a `[q, k, v]` list instead and empties it, so
    the float tensors are freed as soon as their quantized forms exist
    rather than at the end of the call.

    **What this saves depends entirely on how the caller allocated q/k/v,
    and in the most common arrangement it currently saves nothing.**
    Measured at MiniMax H3's fl2va shape (S=41822, heads 56, head_dim 128,
    bf16), peak per call, one arm per process:

        allocation             smooth_k   sageattn   consume    saved
        separate                  False      3148       2290     -858
        separate                  True       3148       2862     -287
        fused QKV views           False      3148       3148        0
        fused QKV views           True       3148       3148        0
        fused, caller clones v    False      3720       2862     -858
        fused, caller clones v    True       3720       3434     -287

    Both axes matter and neither is obvious:

    - `smooth_k` defaults to True, and `per_thread_int8` allocates
      q_int8/k_int8 *before* evaluating `k = k - km`, so a full bf16 copy of
      K sits on top of them and eats most of the saving.
    - When q/k/v are three views of one fused QKV buffer -- which is what
      `qkv_proj(x).split(...)` produces, i.e. how essentially every DiT block
      makes them -- releasing q and k frees nothing, because v still
      references the same allocation. By the time v is released,
      `per_channel_fp8`'s full-size bf16 transpose buffer has already set a
      higher peak.

    Making the fused case pay *from in here* requires dropping that transpose
    buffer **and** doing the mean-subtraction in place; either alone leaves
    the other setting the floor.

    **But the caller can fix it without us**, and callers do. Cloning v
    before handing the list over gives it its own storage, so releasing q and
    k actually frees the fused buffer -- it converts the fused case into the
    separate case for the price of one third of the buffer. Read the last two
    rows against the 3148 the caller would otherwise get, not against the
    3720 in their own sageattn column: -286 with `smooth_k=False`, and +286,
    i.e. *worse than not cloning*, with `smooth_k=True`. Both halves are
    load-bearing. Cloning without consuming is a pure 572 MiB cost, and
    consuming without `smooth_k=False` gives the clone back.

    ComfyUI does this in `comfy/ldm/minimax/model.py` ("Fix peak memory issue
    with H3") and ComfyUI-h3-explorations does it in its H3 attention
    forward, gated on the mode reaching this entry point at all.

    Gate that clone on `sageattn_consume_prefers_cloned_v(device)` rather
    than on an arch check of your own: only the archs below release early,
    and on the others the clone is a flat 572 MiB loss. Going through the
    predicate also means a caller picks up the day we make the fused case
    pay from in here, at which point cloning becomes a cost and the answer
    flips to False.

    So the honest summary is: worth calling when q/k/v are separate
    allocations or when the caller is willing to clone v and pass
    `smooth_k=False`; a no-op on fused views otherwise. Verified in
    `tests/test_sageattn_consume.py`, which measures every arm rather than
    asserting a single headline number, and asserts the caller-clone case
    specifically because consumers now depend on it.

    Parameters
    ----------
    qkv : list
        A 3-element `[q, k, v]` list. **Emptied by this call.** The caller
        must hold no other references to those tensors, or the saving does
        not materialize -- the whole point is that this list is the last
        owner. Pass `[q, k, v]` and `del q, k, v` before calling.

        The three entries may instead be single-owner containers exposing
        `peek()`/`take()`, which is what ComfyUI hands an attention backend
        and what its H3 model wraps q/k/v in on every call. They are taken
        here rather than by the caller on purpose: unwrapping in the
        caller's frame binds all three tensors there for the duration of
        the call, which is the retention this entry point exists to avoid.
        Containers and list slots are both emptied. Mixing tensors and
        containers, or passing one already consumed, raises `ValueError`.

    Everything else matches `sageattn()`.

    Returns
    -------
    Same as `sageattn()`.

    Notes
    -----
    Only the sm89 fp8 CUDA kernels release early today; other kernels fall
    back to the ordinary path, which is correct but keeps q/k/v alive for
    the duration and so saves nothing.
    """
    if len(qkv) != 3:
        raise ValueError(f"qkv must be a 3-element [q, k, v] list, got {len(qkv)}")

    # ComfyUI hands attention backends single-owner container objects rather
    # than tensors, and its H3 model wraps q/k/v in them on every call. Taking
    # them here rather than making the caller unwrap first is the whole point:
    # unwrapping in the caller's frame binds all three there for the duration
    # of the call, which is the retention this entry point exists to avoid.
    # Classified by what it is NOT, because `torch.Tensor.take(index)` exists
    # and duck-typing on `take` alone matches every tensor.
    boxed = [not isinstance(t, torch.Tensor) for t in qkv]
    if any(boxed):
        if not all(boxed):
            raise ValueError(
                "qkv must be all tensors or all containers, not a mix; got "
                f"{[type(t).__name__ for t in qkv]}"
            )
        if not all(hasattr(t, "take") and hasattr(t, "peek") for t in qkv):
            raise ValueError(
                "qkv entries must be tensors, or single-owner containers "
                "exposing peek()/take(); got "
                f"{[type(t).__name__ for t in qkv]}"
            )
        try:
            taken = [t.take() for t in qkv]
        except RuntimeError as exc:
            # A spent container reaching a quant kernel would surface far from
            # its cause, so it is rejected the same way a malformed list is.
            raise ValueError(f"a qkv container was already consumed: {exc}") from exc
        qkv[0] = qkv[1] = qkv[2] = None
        qkv = taken

    q = qkv[0]
    arch = _cuda_archs[q.device.index]
    fp8_arch = arch in _EARLY_RELEASE_ARCHS
    del q

    # Native CUDA mask support is sm89 + CUDA >= 12.8 only, the same gate
    # `sageattn()` applies. Without matching it here, a masked call on
    # sm89 + CUDA < 12.8 would take fp32+fp32, whose warn-block drops the
    # mask and returns numerically wrong output, and a masked call on
    # sm100/sm120/sm121 would take the native-mask path on archs the
    # dispatcher deliberately routes to Triton for mask correctness. Both
    # entry points have to agree about this or the safe default is only
    # safe depending on which one a consumer reached for.
    mask_needs_triton = attn_mask is not None and not _has_native_mask_kernel(arch)

    if not fp8_arch or mask_needs_triton:
        # No early-release path here; behave exactly like sageattn() and
        # let the list drop its references when the caller releases it.
        q, k, v = qkv
        qkv.clear()
        return sageattn(
            q, k, v, tensor_layout=tensor_layout, is_causal=is_causal,
            sm_scale=sm_scale, attn_mask=attn_mask, **kwargs,
        )

    if arch == "sm89" and get_cuda_version() < (12, 8):
        kwargs.setdefault("pv_accum_dtype", "fp32+fp32")
    else:
        kwargs.setdefault("pv_accum_dtype", "fp32+fp16")
    if arch != "sm89":
        kwargs.setdefault("qk_quant_gran", "per_warp")
    _log_routing_choice_once(
        arch, attn_mask is not None, kwargs["pv_accum_dtype"],
        KERNEL_FP8_CUDA_PP if kwargs["pv_accum_dtype"] == "fp32+fp16"
        else KERNEL_FP8_CUDA_FP32,
    )
    # The box is passed rather than unpacked into locals here: a local
    # binding in this frame would outlive the callee's `del` and pin the
    # tensor for the whole call, defeating the purpose.
    return sageattn_qk_int8_pv_fp8_cuda(
        qkv[0], qkv[1], qkv[2],
        tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
        attn_mask=attn_mask, _qkv_box=qkv, **kwargs,
    )


def sageattn_consume_prefers_cloned_v(device=None) -> bool:
    """Should a caller with a fused QKV buffer clone v before `sageattn_consume`?

    Answers the caller-side question, not a question about our internals: if
    your q/k/v are views of one allocation and you pass `smooth_k=False`,
    does giving v its own storage lower the peak on this device? See
    `sageattn_consume`'s docstring for the measurement -- 286 MiB per call at
    S=41822 when the answer is True, and a flat 572 MiB cost when it is False.

    The halves the caller still owns are deliberately not folded in here.
    Cloning only pays with `smooth_k=False`, only on a fused buffer, and
    only if the list you hand over is genuinely the last owner; the caller
    knows all three without asking, and rolling them in would make a False
    ambiguous between "wrong arch" and "your config".

    That last one is the easy one to get wrong, because a True here does not
    check it. If you slice q/k/v into groups and call per group, the frame
    doing the slicing holds the originals for the whole sequence, so each
    release frees nothing and the clone is a flat cost with nothing to
    recover it. Measured inert in `tests/test_sageattn_consume.py`, against
    a handover arm saving 858 MiB in the same harness.

    Note it is the retained parents, not the group count: a *single*-group
    pass through the same slicing code loses the saving just as completely.
    So a caller with both paths keeps the whole benefit only by routing the
    unchunked case through a real handover, and decides the clone *after*
    it knows which path it is on -- deciding earlier is how the loop case
    slips through.

    Cheap and stable: one list index against an arch table built at import,
    so it is safe to call per forward and to cache per device index. Prefer
    calling it in the forward over model-patch time -- a patched model may
    not be on its final device yet, and on a multi-GPU box that bakes an
    answer for the wrong GPU.

    Parameters
    ----------
    device : None, int, str or torch.device
        Which GPU to answer for. `None` means the current device. A non-CUDA
        device raises rather than returning False, because a plausible
        boolean for the wrong device is the failure this exists to prevent.

        If your clone decision sits *outside* a guard that degrades to a
        fallback attention on kernel failure, short-circuit non-CUDA to
        False on your side rather than letting this raise through: on a
        device that will not reach these kernels the clone is moot, and a
        raise there converts a graceful degrade into a dead render. The
        raise is aimed at the caller who asks at model-patch time, where
        an offloaded or uncast model gives a confidently wrong answer.

    Returns
    -------
    bool
        Today this is exactly "does this arch take the early-release path",
        because the caller's clone is what lets the fused buffer die once q
        and k are released. It stops being the same question if
        `per_channel_fp8` ever drops its transpose buffer *and* the
        mean-subtraction moves in place (CHANGELOG Backlog): the release
        would still happen, while the no-clone peak would fall below what a
        cloning caller can reach, so this would go False while the release
        path stayed on. Callers gate on this rather than on the arch set so
        that flip reaches them on upgrade instead of needing an edit.
    """
    return _cuda_archs[_resolve_cuda_index(device)] in _EARLY_RELEASE_ARCHS


def sageattn_warmup(
    shapes: List[Tuple[int, int, int, int, int]],
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    kernels: Sequence[Callable] = (sageattn_qk_int8_pv_fp16_triton,),
) -> None:
    """Pre-warm Triton JIT + autotune caches for a list of attention shapes.

    Consumers (e.g. ComfyUI nodes) can call this once at model-patch time
    to hide the first-call cache-miss latency (~100-500ms per new shape
    tuple on cold start) from the user's first gen. Best-effort: shape/
    kernel combinations that raise a dispatch-shape-related error are
    logged and skipped; OOMs and unexpected errors propagate.

    Parameters
    ----------
    shapes : list of (batch, heads, seq_q, seq_kv, head_dim) tuples
        The attention shapes to warm. For LTX-2.3 (head_dim=64, heads=32),
        pass the canonical self-attn + cross-attn shapes from the workflow.

    kernels : sequence of sage kernel callables to warm
        Defaults to the Triton kernel -- that's the only one with runtime
        autotune. CUDA kernels are fully compiled at build time and don't
        benefit from warmup. Pass callables directly
        (e.g. `sageattention.sageattn_qk_int8_pv_fp16_triton`) so typos
        fail at import time rather than silently at warmup.

    Notes
    -----
    - Q/K/V tensors are built in HND layout. The Triton kernel's
      autotune cache keys on shape dimensions (qo_len, kv_len, head_dim,
      block sizes), not layout, so a single HND warmup covers callers
      that use either HND or NHD at the same shape.
    - Triton caches results to disk under its standard cache dir, so the
      benefit survives process restarts. `./build.sh` invalidates the cache.
    """
    for shape in shapes:
        B, H, Sq, Skv, D = shape
        q = torch.randn(B, H, Sq, D, device=device, dtype=dtype)
        k = torch.randn(B, H, Skv, D, device=device, dtype=dtype)
        v = torch.randn(B, H, Skv, D, device=device, dtype=dtype)
        for kernel in kernels:
            try:
                kernel(q, k, v, is_causal=False, tensor_layout="HND")
            except (RuntimeError, ValueError, NotImplementedError) as exc:
                warnings.warn(
                    f"sageattn_warmup: {getattr(kernel, '__name__', kernel)} "
                    f"skipped shape {shape}: {type(exc).__name__}: {exc}"
                )
