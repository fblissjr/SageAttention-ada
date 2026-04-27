#!/usr/bin/env python3
"""Measure sage accuracy + speed on LTX-2.3's actual attention shapes.

Runs each installed sage kernel against torch SDPA (EFFICIENT_ATTENTION
backend) at the shapes that appear in LTX-2.3 video-gen workflows
(head_dim=64, heads=32, typical video/text sequence lengths).

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
  * Do fp16_triton and fp8++ agree with each other? (A mask-aware
    consumer mixes them in one forward pass -- masked -> triton,
    unmasked -> fp8++. If they diverge more than their individual noise
    floors permit, that's a finding worth flagging.)

Run with the venv that has sage installed active:
    ${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

import orjson
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


# LTX 2.3 default config (diffusers transformer_ltx2.py:907-947):
#   Video path: num_attention_heads=32, attention_head_dim=128 (inner_dim=4096)
#   Audio path: audio_num_attention_heads=32, audio_attention_head_dim=64
#                                              (audio_inner_dim=2048)
# Production seq lengths sourced from a real consumer trace
# (sage_2026-04-26_105851.jsonl, 6912 attention calls):
#   iter=null (init render):  seq=22932  (832x448 spatial, 497 frames,
#                                         patch=32x32x8)
#   iter>=1   (loop iters):   seq=23296  (one extra latent frame)
# Update via tests/bench_workload_profile.py against a fresh trace if
# resolution changes or a new iteration scheme lands.
SHAPES = [
    # LTX 2.3 video self-attn at production seq + corrected d=128.
    # 76% of attention wall-time on the trace lives on these two rows.
    Shape("ltx23_video_self_attn_init_22932",  1, 32, 22932, 22932, 128, False),
    Shape("ltx23_video_self_attn_loop_23296",  1, 32, 23296, 23296, 128, False),

    # LTX 2.3 audio-side self-attn at d=64 (audio config).
    # Same seq as video (joint AV at the latent level; both modalities
    # share the temporal axis after patchification).
    Shape("ltx23_audio_self_attn_init_22932",  1, 32, 22932, 22932, 64, False),
    Shape("ltx23_audio_self_attn_loop_23296",  1, 32, 23296, 23296, 64, False),

    # Short-Q path observed in trace at hidden=2048 (3456 calls total;
    # iter=null seq=497, iter>=1 seq=498). Likely Gemma 3 text-encoder
    # self-attn or audio-cross-attn through audio_caption_projection;
    # exact attribution is ambiguous from trace alone but the
    # (B, S, H*D) tuple is what sage actually runs.
    Shape("ltx23_short_q_init_497",            1, 32,   497,   497, 64, False),
    Shape("ltx23_short_q_loop_498",            1, 32,   498,   498, 64, False),

    # K-probe pair at the corrected video config. Two adjacent rows
    # so K = triton_masked_ms / fp8++_unmasked_ms is readable directly:
    # _unmasked is the denominator (fp8++), _kv226 masked is the
    # numerator (fp16_triton via dispatcher). kv=226 = typical Gemma 3
    # padded text length; encoder-driven, not model-config-driven.
    # Gates the deferred "native CUDA mask kernel" Backlog item per
    # CLAUDE.md / Performance research item 5.
    Shape("ltx23_video_cross_unmasked_kv226_kratio_probe",
                                               1, 32, 23296,   226, 128, False),
    # The one masked row that survives. Doubles as the v0.3.0
    # dispatcher mask-routing correctness witness (auto must dispatch
    # to fp16_triton, not fp8_cuda++) and the K-probe numerator.
    Shape("ltx23_video_cross_text_kv226",      1, 32, 23296,   226, 128, True),

    # Synthetic stress: not a workload, tests fp8 dynamic-range
    # robustness (V ~ N(0, 5)). Catches kernel-internal numerical
    # changes that production shapes might mask. Sub-second; cheap.
    Shape("synthetic_wide_v_self_attn",        1, 32,  8192,  8192, 64, False, v_std=5.0),
]


# The two modes a mask-aware consumer typically mixes in one forward pass
# (triton for masked, fp8++ for unmasked). Used in the cross-kernel
# consistency check -- any typo in these labels would silently disable
# that check, so they're pulled out as constants.
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
    ("flashinfer_fp16", None),
]

# SpargeAttention: training-free sparse attention from the same lab as sage,
# explicitly built on sage 2 ("sparse computation is orthogonal to
# quantization"). API has no attn_mask, so masked shapes are skipped. topk
# controls how many tile groups are kept; 0.5 = compute half the attention.
# Gate = rtol no worse than 1.5x sage fp8++ rtol on the same shape AND
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


def dispatch_flashinfer(_payload=None) -> Callable:
    """FlashInfer prefill, fp16 path. Sage uses HND (B, H, S, D); FlashInfer
    needs NHD (S, H, D). The layout conversion is hoisted into a per-call
    cache keyed on input tensor identity so the timed loop measures FlashInfer,
    not the transpose+contiguous overhead (which would otherwise dominate
    short kernels). Defensive mask-raise belt-and-suspenders the call-site
    skip."""
    from flashinfer import single_prefill_with_kv_cache

    nhd_cache: dict[tuple[int, int, int], tuple] = {}

    def _fn(q, k, v, mask):
        if mask is not None:
            raise NotImplementedError(
                "FlashInfer prefill doesn't accept arbitrary boolean masks here."
            )
        key = (id(q), id(k), id(v))
        cached = nhd_cache.get(key)
        if cached is None:
            cached = (
                q.squeeze(0).transpose(0, 1).contiguous(),
                k.squeeze(0).transpose(0, 1).contiguous(),
                v.squeeze(0).transpose(0, 1).contiguous(),
            )
            nhd_cache[key] = cached
        q_n, k_n, v_n = cached
        out = single_prefill_with_kv_cache(q_n, k_n, v_n, kv_layout="NHD")
        return out.transpose(0, 1).unsqueeze(0).contiguous()

    return _fn


def dispatch_sparge(topk: float) -> Callable:
    """SpargeAttention top-k unmasked self-attention. Inherits sage's mask
    gap (no attn_mask kwarg), so masks raise here as a defensive guard --
    the bench's main loop also skips masked shapes for sparge upstream."""
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


def run_shape_sweep(
    shapes: list[Shape], dtype: torch.dtype = torch.bfloat16
) -> tuple[list[str], dict[tuple[str, str], Metrics]]:
    """Run the per-shape table for any list of Shape entries. Returns
    (warnings, measurements). `warnings` is the soft-warn list (mean_rtol
    > 0.10 entries). `measurements` maps (shape_name, mode_label) ->
    Metrics so a regression-check pass can grade against pinned baselines
    without re-running the bench. The image-shape file reuses this entry
    point with its own SHAPES list."""
    label_width = max(
        len(spec[0]) for spec in
        (*MODE_SPECS, *TORCH_MODE_SPECS, *FLASHINFER_MODE_SPECS, *SPARGE_MODE_SPECS)
    )
    print_header(label_width)

    warnings: list[str] = []
    measurements: dict[tuple[str, str], Metrics] = {}

    for shape in shapes:
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
            measurements[(shape.name, label)] = m
            if label in (TRITON_LABEL, FP8PP_LABEL):
                cached_outs[label] = out

        # Cross-kernel consistency: a mask-aware consumer routes masked
        # calls to triton and unmasked to fp8++ in one forward pass; do
        # those two kernels agree beyond the noise floor their individual
        # errors can explain? Unmasked shapes only -- on masked shapes
        # the CUDA path is known-broken, so divergence is the bug itself.
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

        # Torch SDPA, FlashInfer, and SpargeAttention each iterate the same
        # try/measure/SKIP/print pattern; only the dispatcher differs. Sage's
        # MODE_SPECS loop above stays separate because it caches outputs for
        # the cross-kernel consistency row (fp8++ vs triton).
        def _run_aux(specs, dispatch_factory, warn_rtol):
            for label, payload in specs:
                try:
                    mode_fn = dispatch_factory(payload)
                    m, _ = measure_mode(mode_fn, q, k, v, mask, out_ref)
                except Exception as exc:
                    print(f"    {label:<{label_width}}  SKIP ({type(exc).__name__}: {str(exc)[:80]})")
                    continue
                _print_result(label, m, warn_rtol=warn_rtol)
                measurements[(shape.name, label)] = m

        # Torch backends reference themselves against SDPA-EFFICIENT, so rtol
        # can be ~0 (one of them IS the reference) -- don't warn.
        _run_aux(TORCH_MODE_SPECS, dispatch_torch, warn_rtol=False)
        _run_aux(FLASHINFER_MODE_SPECS, dispatch_flashinfer, warn_rtol=False)
        # Sparge gate ("rtol <= 1.5x sage fp8++ rtol") is a post-hoc analysis
        # on the printed numbers, not an in-loop check; same warn threshold
        # as sage modes is fine.
        _run_aux(SPARGE_MODE_SPECS, dispatch_sparge, warn_rtol=True)

    return warnings, measurements


def check_regressions(
    measurements: dict[tuple[str, str], Metrics],
    baselines_path: Path,
) -> tuple[int, list[str]]:
    """Grade fresh measurements against pinned baselines.

    Returns (exit_code, regression_lines). Exit 0 = clean; exit 1 = at
    least one load-bearing baseline regressed beyond the configured
    drift / rtol budget. Non-load-bearing baselines drift-warn but
    don't fail the gate.
    """
    if not baselines_path.exists():
        return 0, [f"(no baselines file at {baselines_path}; skipping check)"]

    cfg = orjson.loads(baselines_path.read_bytes())

    rtol_budget = float(cfg.get("rtol_budget", 0.10))
    perf_drift_pct = float(cfg.get("perf_drift_pct", 5.0))
    speedup_floor = float(cfg.get("speedup_ratio_floor", 1.5))

    regressions: list[str] = []
    notes: list[str] = []
    fail = False

    # Speedup-ratio anchor: the shape with both a sage-primary AND a
    # torch_flash baseline marked load_bearing. Auto-discovered to
    # survive shape renames without manual lookup updates. Errs on
    # multiple anchors; informational-only on zero.
    speedup_anchor: str | None = None
    sage_primary_ms: float | None = None
    torch_flash_ms: float | None = None

    sage_shapes = {
        e["shape"] for e in cfg.get("baselines", [])
        if e.get("mode") == "fp8_cuda++" and e.get("load_bearing")
    }
    torch_shapes = {
        e["shape"] for e in cfg.get("baselines", [])
        if e.get("mode") == "torch_flash" and e.get("load_bearing")
    }
    anchors = sage_shapes & torch_shapes
    if len(anchors) == 1:
        speedup_anchor = next(iter(anchors))
    elif len(anchors) > 1:
        notes.append(
            f"SPEEDUP  multiple shapes have both fp8_cuda++ and torch_flash "
            f"load_bearing baselines ({sorted(anchors)}); skipping ratio check. "
            f"Mark exactly one shape as the anchor."
        )

    for entry in cfg.get("baselines", []):
        shape = entry["shape"]
        mode = entry["mode"]
        baseline_ms = entry.get("median_ms")
        baseline_rtol = entry.get("mean_rtol")
        load_bearing = bool(entry.get("load_bearing", False))

        m = measurements.get((shape, mode))
        if m is None:
            line = f"MISSING  {shape} / {mode}: not measured this run"
            (regressions if load_bearing else notes).append(line)
            if load_bearing:
                fail = True
            continue

        if shape == speedup_anchor:
            if mode == "fp8_cuda++":
                sage_primary_ms = m.median_ms
            elif mode == "torch_flash":
                torch_flash_ms = m.median_ms

        if baseline_ms is not None and m.median_ms is not None:
            drift_pct = (m.median_ms - baseline_ms) / baseline_ms * 100.0
            if drift_pct > perf_drift_pct:
                line = (
                    f"PERF     {shape} / {mode}: {m.median_ms:.2f} ms vs "
                    f"baseline {baseline_ms:.2f} ms (+{drift_pct:.1f}%, "
                    f"threshold {perf_drift_pct:.1f}%)"
                )
                (regressions if load_bearing else notes).append(line)
                if load_bearing:
                    fail = True
            elif drift_pct < -perf_drift_pct:
                notes.append(
                    f"FASTER   {shape} / {mode}: {m.median_ms:.2f} ms vs "
                    f"baseline {baseline_ms:.2f} ms ({drift_pct:.1f}%) "
                    f"-- verify env stable before updating baselines"
                )

        if baseline_rtol is not None and m.mean_rtol is not None:
            if m.mean_rtol > rtol_budget:
                line = (
                    f"RTOL     {shape} / {mode}: mean_rtol={m.mean_rtol:.4f} "
                    f"exceeds budget {rtol_budget:.2f}"
                )
                (regressions if load_bearing else notes).append(line)
                if load_bearing:
                    fail = True
            elif m.mean_rtol > baseline_rtol * 1.5:
                # 1.5x baseline = kernel-internal numerical change signal
                # even when the absolute number is still under budget.
                notes.append(
                    f"RTOL_DRIFT {shape} / {mode}: mean_rtol={m.mean_rtol:.4f} "
                    f"vs baseline {baseline_rtol:.4f} (1.5x baseline)"
                )

    # Speedup-ratio floor: sage_fp8++ must remain at least speedup_floor x faster
    # than torch_flash on the anchor row, or the fork's load-bearing claim
    # collapses (per CLAUDE.md / "What we explicitly ignore" / torch row trigger).
    if (speedup_anchor is not None
            and sage_primary_ms is not None
            and torch_flash_ms is not None):
        ratio = torch_flash_ms / sage_primary_ms
        if ratio < speedup_floor:
            regressions.append(
                f"SPEEDUP  {speedup_anchor}: torch_flash/sage_fp8++ "
                f"= {ratio:.2f}x; below floor {speedup_floor:.2f}x. The fork's "
                f"reason to exist is empirically suspect -- see CLAUDE.md / "
                f"Performance research."
            )
            fail = True
        else:
            notes.append(
                f"SPEEDUP  {speedup_anchor}: torch_flash/sage_fp8++ "
                f"= {ratio:.2f}x (floor {speedup_floor:.2f}x)"
            )

    out_lines = regressions + notes
    return (1 if fail else 0), out_lines


def print_warnings_footer(warnings: list[str]) -> None:
    """Print the standard soft-warnings footer. Image-shape file uses the
    same footer so the two scripts read identically at the bottom."""
    print()
    if warnings:
        print(f"Soft warnings ({len(warnings)}): mean_rtol > 0.10 on:")
        for w in warnings:
            print(f"  - {w}")
        print("Not a test failure -- the point is to measure where accuracy "
              "diverges, not to gate on a number.")
    else:
        print("All (shape, mode) pairs: mean_rtol <= 0.10.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-regression", action="store_true",
        help="Compare results against tests/regression_baselines.json and "
             "exit non-zero on perf drift > threshold or rtol budget breach. "
             "Default: print-only (legacy behavior).",
    )
    parser.add_argument(
        "--baselines", type=Path,
        default=Path(__file__).parent / "regression_baselines.json",
        help="Path to regression baselines JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this test measures kernel numerics on-GPU.")
        return

    warnings, measurements = run_shape_sweep(SHAPES)
    print_warnings_footer(warnings)

    if args.check_regression:
        print()
        print("=== Regression check vs baselines ===")
        exit_code, lines = check_regressions(measurements, args.baselines)
        if not lines:
            print("(no baselines configured)")
        for line in lines:
            print(f"  {line}")
        if exit_code != 0:
            print()
            print("REGRESSION DETECTED -- exiting non-zero. Investigate before "
                  "updating baselines. Bench env discipline: re-snapshot "
                  "internal/bench_env_<date>.txt if any version bumped.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
