#!/usr/bin/env python3
"""Measure sage accuracy + speed on LTX-2.3's actual attention shapes.

Runs each installed sage kernel against torch SDPA (EFFICIENT_ATTENTION
backend) at the shapes that appear in the LTX-2.3 audio-loop video-gen
workflow (head_dim=64, heads=32, typical video/text sequence lengths).

Why EFFICIENT_ATTENTION and not MATH: the MATH backend materializes the
full Sq x Skv attention matrix, which is ~120 GiB at the LTX self-attn
shape (31776 x 31776 x 32 heads x fp32). EFFICIENT_ATTENTION uses Flash /
memory-efficient kernels that are O(N) memory. The numerical difference
vs MATH is orders of magnitude smaller than sage's own quantization
error, so the comparison is still meaningful for this test's purpose.

Measurement, not CI gating -- prints metrics and soft-warns when
mean_rtol exceeds the fork README's "<0.1 on RTX 40xx/50xx" expectation.
The point is to answer:
  * How much accuracy does fp8++ cost vs SDPA on LTX's real shapes?
  * Is cross-attention (with mask) where the loss is concentrated?
  * How much speed does fp16_cuda cost relative to fp8++?
  * Do fp16_triton and fp8++ agree with each other? (The AudioLoopHelper
    consumer mixes them in one forward pass -- masked -> triton,
    unmasked -> fp8++. If they diverge more than their individual noise
    floors permit, that's a finding worth flagging.)

Run with the venv that has sage installed active:
    ${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py
"""

import time
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class Shape(NamedTuple):
    name: str
    batch: int
    heads: int
    seq_q: int
    seq_kv: int
    head_dim: int
    has_mask: bool
    v_std: float = 1.0  # scale V random init; >1 stresses fp8's dynamic range


class Metrics(NamedTuple):
    mean_rtol: float
    max_rtol: float
    mean_atol: float
    max_atol: float
    median_ms: float


# LTX-2.3: num_attention_heads=32, attention_head_dim=64. The cross-attn
# seq_kv values span typical Gemma 3 12B text-encoder padded lengths (both
# fp and fpmixed variants produce identical seq lengths; padding is
# precision-independent). Re-run the sweep if LTX switches encoders or if
# prompt lengths drift outside this range. Characterization valid as of
# 2026-04-23.
SHAPES = [
    Shape("self_attn_large_704x704x497",   1, 32, 31776, 31776, 64, False),
    Shape("self_attn_small_512x512x97",    1, 32,  8192,  8192, 64, False),
    Shape("cross_attn_text_kv32",          1, 32, 31776,    32, 64, True),
    Shape("cross_attn_text_kv64",          1, 32, 31776,    64, 64, True),
    Shape("cross_attn_text_kv128",         1, 32, 31776,   128, 64, True),
    Shape("cross_attn_text_kv226",         1, 32, 31776,   226, 64, True),
    Shape("cross_attn_text_kv512",         1, 32, 31776,   512, 64, True),
    Shape("cross_attn_text_kv1024",        1, 32, 31776,  1024, 64, True),
    # Non-LTX generality check: wide-V distribution stresses fp8 range.
    # Relevant for the v-scale default question (core.py:772): the
    # sm89 fp8_cuda path uses scale_max=448 by default; the ++ variant
    # uses 2.25. If we ever consider flipping the default to 2.25 for
    # non-++ callers, this shape characterizes how much precision the
    # non-LTX user loses.
    Shape("synthetic_wide_v_self_attn",    1, 32,  8192,  8192, 64, False, v_std=5.0),
    # Image-gen representative self-attn shape (Flux/Z-Image-class). 1024^2
    # output / 16^2 VAE compression -> ~4096 image tokens; head_dim=128 and
    # heads=24 are the Flux-1-dev family defaults. Used to confirm sage's
    # speedup story still holds on image-gen head_dim=128 workloads, not just
    # LTX's head_dim=64. If sage doesn't win here, the consumer-side router
    # would need a per-model-class branch.
    Shape("image_gen_self_attn_4096_h24_d128", 1, 24, 4096, 4096, 128, False),
]


# The two modes whose outputs AudioLoopHelper mixes in one forward pass
# (triton for masked, fp8++ for unmasked). Used downstream in the cross-kernel
# consistency check -- any typo in these labels would silently disable that
# check, so they're pulled out as constants.
TRITON_LABEL = "fp16_triton"
FP8PP_LABEL = "fp8_cuda++"

# Each mode: (label, kernel attribute name, extra kwargs). The "auto" entry
# has kernel=None and is dispatched specially to sageattn().
MODE_SPECS = [
    ("fp16_cuda",  "sageattn_qk_int8_pv_fp16_cuda",   {"pv_accum_dtype": "fp32"}),
    (TRITON_LABEL, "sageattn_qk_int8_pv_fp16_triton", {}),
    ("fp8_cuda",   "sageattn_qk_int8_pv_fp8_cuda",    {"pv_accum_dtype": "fp32+fp32"}),
    (FP8PP_LABEL,  "sageattn_qk_int8_pv_fp8_cuda",    {"pv_accum_dtype": "fp32+fp16"}),
    ("auto",       None,                              {}),
]

# Torch SDPA backends measured alongside sage. Serves as the "is sage still
# worth maintaining?" regression — if a future torch release closes the
# speedup gap this test will say so. SDPBackend.MATH is excluded (OOMs at
# LTX self-attn scale; uses the Sq x Skv full matrix).
TORCH_MODE_SPECS = [
    ("torch_flash",  SDPBackend.FLASH_ATTENTION),
    ("torch_eff",    SDPBackend.EFFICIENT_ATTENTION),
    ("torch_cudnn",  SDPBackend.CUDNN_ATTENTION),
]

# FlashInfer fp16 prefill, optional. Predicted to lag sage fp8++ on sm89
# because CUTLASS lacks native fp8 below sm90 -- FlashInfer's fp8 paths are
# Hopper-shaped. We measure fp16 anyway: if it beats torch_flash's fp16, it
# becomes a tracked fp16 fallback for paths sage can't run.
FLASHINFER_MODE_SPECS = [
    ("flashinfer_fp16", "single_prefill_with_kv_cache"),
]

# SpargeAttention: training-free sparse attention from the same lab as sage,
# explicitly built on sage 2 ("sparse computation is orthogonal to
# quantization"). API has no attn_mask, so masked shapes are skipped. topk
# controls how many tile groups are kept; 0.5 = compute half the attention.
# Pass = rtol no worse than 1.5x sage fp8++ rtol on the same shape AND
# wall-clock measurably faster.
SPARGE_MODE_SPECS = [
    ("sparge_topk0.5", 0.5),
]


def accuracy_metrics(actual: torch.Tensor, expect: torch.Tensor) -> tuple[float, float, float, float]:
    a = actual.float()
    e = expect.float()
    diff = (a - e).abs()
    eps = torch.tensor(torch.finfo(a.dtype).eps, device=a.device, dtype=a.dtype)
    rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), eps)
    return (rdiff.mean().item(), rdiff.max().item(), diff.mean().item(), diff.max().item())


def time_median_ms(fn: Callable, warmup: int = 1, runs: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def make_qkv(shape: Shape, dtype: torch.dtype, v_std: float = 1.0):
    q = torch.randn(shape.batch, shape.heads, shape.seq_q,  shape.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(shape.batch, shape.heads, shape.seq_kv, shape.head_dim, device="cuda", dtype=dtype)
    v = torch.randn(shape.batch, shape.heads, shape.seq_kv, shape.head_dim, device="cuda", dtype=dtype) * v_std
    return q, k, v


def build_padding_mask(shape: Shape, pad_tail: int = 30) -> torch.Tensor:
    """Boolean mask where the last pad_tail kv positions are masked out."""
    mask = torch.ones(shape.batch, shape.heads, shape.seq_q, shape.seq_kv, device="cuda", dtype=torch.bool)
    mask[..., -pad_tail:] = False
    return mask


def sdpa_reference(q, k, v, mask) -> torch.Tensor:
    """torch SDPA efficient-attention backend. Close-to-exact reference that
    stays in O(N) memory; see module docstring for the MATH-vs-EFFICIENT note."""
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def dispatch_sage(kernel_name: str | None, kwargs: dict):
    """Return a callable `fn(q, k, v, mask) -> out` for the given mode spec.
    Imports sage lazily so a missing kernel only breaks that mode's run."""
    import sageattention as _sa

    if kernel_name is None:
        # "auto" -- delegate to sage's own dispatch.
        return lambda q, k, v, mask: _sa.sageattn(
            q, k, v, attn_mask=mask, is_causal=False, tensor_layout="HND",
        )
    kernel = getattr(_sa, kernel_name)
    return lambda q, k, v, mask: kernel(
        q, k, v, attn_mask=mask, is_causal=False, tensor_layout="HND", **kwargs,
    )


def dispatch_torch(backend: SDPBackend) -> Callable:
    """Return a callable `fn(q, k, v, mask) -> out` for a torch SDPA backend.
    The sdpa_kernel context manager forces the dispatch; without it torch
    picks automatically."""
    def _fn(q, k, v, mask):
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return _fn


def dispatch_flashinfer(api_name: str) -> Callable:
    """Return a callable for a FlashInfer prefill API. Layout conversion:
    sage's HND is (B, H, S, D); FlashInfer NHD is (S, H, D). Batch size 1
    only -- DiTs run one sample per forward, so this matches our use."""
    import flashinfer

    fn = getattr(flashinfer, api_name)

    def _fn(q, k, v, mask):
        if mask is not None:
            raise NotImplementedError(
                "FlashInfer prefill paths don't accept arbitrary boolean masks "
                "in this configuration; skip masked shapes."
            )
        # (1, H, S, D) -> (S, H, D) for NHD layout
        q_n = q.squeeze(0).transpose(0, 1).contiguous()
        k_n = k.squeeze(0).transpose(0, 1).contiguous()
        v_n = v.squeeze(0).transpose(0, 1).contiguous()
        out = fn(q_n, k_n, v_n, kv_layout="NHD")
        # (Sq, H, D) -> (1, H, Sq, D)
        return out.transpose(0, 1).unsqueeze(0).contiguous()

    return _fn


def dispatch_sparge(topk: float) -> Callable:
    """Return a callable for SpargeAttention top-k unmasked self-attention.
    Sparge inherits sage's mask gap (no attn_mask kwarg), so masked shapes
    must be skipped at the call site, not here."""
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda

    def _fn(q, k, v, mask):
        if mask is not None:
            raise NotImplementedError(
                "SpargeAttention has no attn_mask support (inherits sage's gap)."
            )
        return spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=False)

    return _fn


def measure_mode(
    mode_fn: Callable, q, k, v, mask, out_ref: torch.Tensor,
) -> tuple[Metrics, torch.Tensor]:
    out = mode_fn(q, k, v, mask)
    mean_r, max_r, mean_a, max_a = accuracy_metrics(out, out_ref)
    median_ms = time_median_ms(lambda: mode_fn(q, k, v, mask))
    return Metrics(mean_r, max_r, mean_a, max_a, median_ms), out


def print_header(label_width: int):
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    try:
        import sageattention
        print(f"sage:   {getattr(sageattention, '__version__', '?')}")
    except ImportError:
        print("sage:   (not importable)")
    print(f"{'mode':<{label_width}}  {'mean_rtol':>10}  {'max_rtol':>10}  "
          f"{'mean_atol':>10}  {'max_atol':>10}  {'median_ms':>10}  speed_x")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this test measures kernel numerics on-GPU.")
        return

    label_width = max(
        max(len(name) for name, _, _ in MODE_SPECS),
        max(len(name) for name, _ in TORCH_MODE_SPECS),
        max(len(name) for name, _ in FLASHINFER_MODE_SPECS),
        max(len(name) for name, _ in SPARGE_MODE_SPECS),
    )
    print_header(label_width)

    dtype = torch.bfloat16
    warnings: list[str] = []

    for shape in SHAPES:
        print()
        print(f"=== {shape.name} ===")
        print(
            f"    B={shape.batch} H={shape.heads} Sq={shape.seq_q} Skv={shape.seq_kv} "
            f"D={shape.head_dim} mask={shape.has_mask} dtype={dtype}"
        )
        q, k, v = make_qkv(shape, dtype, v_std=shape.v_std)
        mask = build_padding_mask(shape) if shape.has_mask else None

        # Compute the reference exactly once per shape; share across all modes.
        out_ref = sdpa_reference(q, k, v, mask)
        sdpa_ms = time_median_ms(lambda: sdpa_reference(q, k, v, mask))
        print(f"    {'SDPA (math)':<{label_width}}"
              f"  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}  {sdpa_ms:>10.2f}  {1.0:>5.2f}x")

        def _print_row(
            label: str,
            mean_r: float, max_r: float, mean_a: float, max_a: float,
            median_ms: float | None,
            warn_threshold: float = 0.10,
            warn: bool = True,
        ):
            marker = "  !" if warn and mean_r > warn_threshold else ""
            ms_cell = f"{median_ms:>10.2f}" if median_ms is not None else f"{'-':>10}"
            speed_cell = f"{sdpa_ms / median_ms:>5.2f}x" if median_ms is not None else f"{'-':>5} "
            print(
                f"    {label:<{label_width}}  "
                f"{mean_r:>10.3g}  {max_r:>10.3g}  "
                f"{mean_a:>10.3g}  {max_a:>10.3g}  "
                f"{ms_cell}  {speed_cell}{marker}"
            )
            if warn and mean_r > warn_threshold:
                warnings.append(f"{shape.name} / {label}: mean_rtol={mean_r:.3g}")

        def _print_result(label: str, m: Metrics, warn_rtol: bool = True):
            _print_row(label, m.mean_rtol, m.max_rtol, m.mean_atol, m.max_atol,
                       median_ms=m.median_ms, warn=warn_rtol)

        cached_outs: dict[str, torch.Tensor] = {}
        for label, kernel_name, kwargs in MODE_SPECS:
            try:
                mode_fn = dispatch_sage(kernel_name, kwargs)
                m, out = measure_mode(mode_fn, q, k, v, mask, out_ref)
            except Exception as exc:
                print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            _print_result(label, m)
            if label in (TRITON_LABEL, FP8PP_LABEL):
                cached_outs[label] = out

        # Cross-kernel consistency: AudioLoopHelper routes masked calls to
        # triton and unmasked to fp8++ in one forward pass; do those two
        # kernels agree beyond the noise floor their individual errors
        # can explain? Unmasked shapes only -- on masked shapes the CUDA
        # path is known-broken, so divergence is the bug itself.
        #
        # Threshold 0.15: fp16_triton ~0.04 vs SDPA and fp8++ ~0.09 vs SDPA;
        # quadrature combination ~= sqrt(0.04^2 + 0.09^2) ~= 0.098 is the
        # floor. 0.15 leaves 50% headroom so only a secondary numerical
        # issue (beyond each kernel's independent error budget) breaches it.
        if not shape.has_mask and TRITON_LABEL in cached_outs and FP8PP_LABEL in cached_outs:
            mean_r, max_r, mean_a, max_a = accuracy_metrics(
                cached_outs[FP8PP_LABEL], cached_outs[TRITON_LABEL]
            )
            _print_row("fp8++vs.triton", mean_r, max_r, mean_a, max_a,
                       median_ms=None, warn_threshold=0.15)

        for label, backend in TORCH_MODE_SPECS:
            try:
                mode_fn = dispatch_torch(backend)
                m, _ = measure_mode(mode_fn, q, k, v, mask, out_ref)
            except Exception as exc:
                # Torch backends have shape/dtype/mask restrictions; skipping
                # is normal (e.g. FLASH rejects certain mask layouts).
                print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            # Torch backends reference themselves against SDPA-EFFICIENT, so
            # rtol can be ~0 (one of them IS the reference). Don't warn.
            _print_result(label, m, warn_rtol=False)

        # FlashInfer fp16 prefill -- optional, masked shapes skipped.
        for label, api_name in FLASHINFER_MODE_SPECS:
            try:
                mode_fn = dispatch_flashinfer(api_name)
                m, _ = measure_mode(mode_fn, q, k, v, mask, out_ref)
            except Exception as exc:
                print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            _print_result(label, m, warn_rtol=False)

        # SpargeAttention -- unmasked only, training-free top-k sparse.
        for label, topk in SPARGE_MODE_SPECS:
            try:
                mode_fn = dispatch_sparge(topk)
                m, _ = measure_mode(mode_fn, q, k, v, mask, out_ref)
            except Exception as exc:
                print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                continue
            # Soft-warn at the same threshold as sage modes; the gate criterion
            # in the optimization plan is "rtol <= 1.5x sage fp8++ rtol on the
            # same shape," which is a post-hoc analysis on these numbers, not
            # an in-loop check.
            _print_result(label, m)

    print()
    if warnings:
        print(f"Soft warnings ({len(warnings)}): mean_rtol > 0.10 on:")
        for w in warnings:
            print(f"  - {w}")
        print("Not a test failure -- the point is to measure where accuracy "
              "diverges, not to gate on a number.")
    else:
        print("All (shape, mode) pairs: mean_rtol <= 0.10.")


if __name__ == "__main__":
    main()
