"""Ragged K/V tails must not read stale shared memory.

`load_global_to_share` (csrc/qattn/attn_utils.cuh) issues its `cp.async`
with a predicate so the out-of-range rows of the last K/V tile are not
fetched. The fill mode decides what those shared-memory lanes hold
afterwards:

  kNoFill   -- untouched. Whatever the previous launch left in that
               shared-memory bank stays there and feeds the MMA.
  kFillZero -- hardware zero-fills the predicated-off bytes. Free: the
               mode is a compile-time `if constexpr` in cp_async.cuh and
               `cp.async` zero-fill is a native PTX operand, not an extra
               store.

Whether stale residue is *visible* depends on which tensor rides that
load on a given arch:

  sm89 fp8 (the dispatcher default) -- K only. K shared memory is int8,
    so residue cannot be NaN, and the out-of-range columns are masked out
    of the softmax before they reach P@V. V does not use this helper at
    all; it takes `load_fp8_V_global_to_share`, unpredicated, and relies
    on `per_channel_fp8` having padded V (see the upstream warning at
    qk_int_sv_f8_cuda_sm89.cuh:263). This arch is the CONTROL here: it
    should pass before and after the fix.

  sm80 fp16 (`sageattn_qk_int8_pv_fp16_cuda`, an exported consumer entry
    point, forward-compatible onto Ada) -- V rides the predicated load as
    fp16. Residue is read as a float, and P@V has no mask to hide it:
    softmax gives those rows p=0, but 0 * inf = NaN, so a single stale
    non-finite lane poisons the whole output row.

The defect is state-dependent, and every case here has to force the state
or it proves nothing. Measured on the unfixed build: in a freshly started
process the short-length sweep PASSES (worst mean_rtol 0.0345), because
nothing has run yet and the residue happens to be benign. It only fails
once a previous launch has left non-finite values behind. So each sweep
below dirties shared memory with +inf first; a version of this file that
skipped that step would be green against a kernel that is actually broken.

Measurement also narrowed the exposure from the upstream description
("short/non-multiple") to a sharper boundary:

  kv_len <  64  -- exposed. The first K tile is itself partial, so the
                   predicated-off lanes hold whatever a FOREIGN kernel
                   left. Dirtied, all 12 lengths tested return 100%
                   non-finite output.
  kv_len >= 64  -- not exposed in practice, including non-multiples like
                   127 and 385. At least one full tile has been loaded by
                   then, so the stale lanes hold this launch's own earlier
                   K/V rows: wrong in principle, but finite, and softmax
                   gives them p=0, so they contribute nothing.

That boundary is why `test_ragged_lens_stay_clean_above_one_tile` exists
as a control rather than a defect detector.

Inherited from thu-ml/SageAttention; fixed upstream in
woct0rdho/SageAttention@e147939, which this fork had not picked up.

Standalone (no pytest), per the repo's test convention. Run:
    ${VIRTUAL_ENV}/bin/python tests/test_short_seq_tail.py
"""

import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention
from sageattention import sageattn, sageattn_qk_int8_pv_fp16_cuda

DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH, HEADS, HEAD_DIM = 1, 8, 128

# CTA_K on both the sm80 and sm89 kernels. A kv_len that is a whole
# multiple of this predicates nothing off, so the sweeps below deliberately
# avoid landing only on multiples.
CTA_K = 64

# Shorter than one K tile: the very first tile is partly predicated off, so
# its shared memory is whatever the previous launch left, not this call's K.
SHORT_LENS = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65]

# Past the first tile, so the ragged tail sits behind at least one full
# iteration. Catches the case where residue comes from this same launch's
# earlier iteration rather than a previous kernel.
RAGGED_LENS = [127, 129, 191, 193, 255, 257, 383, 385]

# Big enough to touch every SM, so the dirty residue is likely to be sitting
# in whichever shared-memory bank the short launch below lands on.
DIRTY_LEN = 4096
DIRTY_REPEATS = 8


def accuracy_metrics(actual: torch.Tensor, expect: torch.Tensor):
    """Symmetric-denominator rtol/atol; matches tests/test_sageattn_ltx_shapes.py:160."""
    a = actual.float()
    e = expect.float()
    diff = (a - e).abs()
    eps = torch.tensor(torch.finfo(a.dtype).eps, device=a.device, dtype=a.dtype)
    rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), eps)
    return (rdiff.mean().item(), rdiff.max().item(), diff.mean().item(), diff.max().item())


def _qkv(seq_q, seq_kv, generator):
    shape_q = (BATCH, HEADS, seq_q, HEAD_DIM)
    shape_kv = (BATCH, HEADS, seq_kv, HEAD_DIM)
    q = torch.randn(shape_q, device=DEVICE, dtype=DTYPE, generator=generator)
    k = torch.randn(shape_kv, device=DEVICE, dtype=DTYPE, generator=generator)
    v = torch.randn(shape_kv, device=DEVICE, dtype=DTYPE, generator=generator)
    return q, k, v


def _reference(q, k, v):
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def _dirty_shared_memory(kernel, fill):
    """Run `kernel` at a large kv_len with V saturated, to leave residue behind.

    The output is discarded; the point is the shared-memory state afterwards.
    Repeated so the launch covers enough CTAs to land on most SMs.
    """
    g = torch.Generator(device=DEVICE).manual_seed(0)
    q, k, v = _qkv(DIRTY_LEN, DIRTY_LEN, g)
    v.fill_(fill)
    for _ in range(DIRTY_REPEATS):
        kernel(q, k, v, tensor_layout="HND", is_causal=False)
    torch.cuda.synchronize()


def _sweep(kernel, lens, label, rtol_budget):
    """Assert finite + accurate output at each kv_len. Returns the worst rtol."""
    failures = []
    worst = 0.0
    for seq_kv in lens:
        g = torch.Generator(device=DEVICE).manual_seed(seq_kv)
        q, k, v = _qkv(seq_kv, seq_kv, g)
        out = kernel(q, k, v, tensor_layout="HND", is_causal=False)
        if not torch.isfinite(out).all():
            n_bad = (~torch.isfinite(out)).sum().item()
            failures.append(f"kv_len={seq_kv}: {n_bad} non-finite values in output")
            continue
        mean_rtol, _, _, _ = accuracy_metrics(out, _reference(q, k, v))
        worst = max(worst, mean_rtol)
        if mean_rtol > rtol_budget:
            failures.append(
                f"kv_len={seq_kv}: mean_rtol {mean_rtol:.4f} > {rtol_budget:.4f}"
            )
    assert not failures, f"{label}:\n  " + "\n  ".join(failures)
    return worst


def test_sm80_fp16_short_seq_lens():
    """Claim: every kv_len below one K tile is exposed, not just the one below.

    Deleting this leaves the fp16 consumer entry point unguarded across the
    whole sub-tile range. The dirty step is load-bearing: without it this
    sweep passes on the unfixed kernel (measured, worst mean_rtol 0.0345),
    so a version that skipped it would assert nothing.
    """
    _dirty_shared_memory(sageattn_qk_int8_pv_fp16_cuda, float("inf"))
    worst = _sweep(
        sageattn_qk_int8_pv_fp16_cuda, SHORT_LENS,
        "sm80 fp16 short kv_len", rtol_budget=0.10,
    )
    print(f"  worst mean_rtol over {len(SHORT_LENS)} short lens: {worst:.4f}")


def test_ragged_lens_stay_clean_above_one_tile():
    """Control, not a defect detector: pins the exposure boundary at CTA_K.

    These lengths are ragged too, but they pass on the UNFIXED kernel even
    when dirtied, because by the time the tail is loaded the stale lanes hold
    this launch's own earlier tile -- finite, and weighted p=0 by softmax.
    It fails only if the boundary moves, e.g. a tiling change that lets
    foreign residue reach a tail above 64. That is what it is here to catch.
    """
    _dirty_shared_memory(sageattn_qk_int8_pv_fp16_cuda, float("inf"))
    worst = _sweep(
        sageattn_qk_int8_pv_fp16_cuda, RAGGED_LENS,
        "sm80 fp16 ragged kv_len", rtol_budget=0.10,
    )
    print(f"  worst mean_rtol over {len(RAGGED_LENS)} ragged lens: {worst:.4f}")


def test_v_tail_does_not_inherit_stale_smem():
    """Claim: the specific NaN mechanism -- 0 * inf in P@V.

    Softmax gives the out-of-range V rows p=0, which would make any *finite*
    residue harmless. It is non-finite residue that survives the multiply.
    Dirtying with +inf first is what turns a latent read of stale memory into
    an observable NaN, so this is the case that distinguishes kNoFill from
    kFillZero rather than merely exercising short sequences.
    """
    _dirty_shared_memory(sageattn_qk_int8_pv_fp16_cuda, float("inf"))
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    q, k, v = _qkv(33, 33, g)
    out = sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="HND", is_causal=False)
    n_bad = (~torch.isfinite(out)).sum().item()
    assert n_bad == 0, (
        f"{n_bad}/{out.numel()} non-finite values at kv_len=33 after the V "
        f"shared memory was dirtied with +inf. The predicated cp.async left "
        f"the out-of-range lanes untouched, so this call multiplied p=0 "
        f"against the previous launch's inf."
    )


def test_sm89_fp8_default_path_is_unaffected():
    """Control: the dispatcher default must pass both before and after the fix.

    On sm89 the predicated load carries K (int8, so residue cannot be
    non-finite) and the out-of-range columns are masked out of the softmax;
    V takes the unpredicated fp8 helper instead. If this case ever goes red
    it means the arch routing changed and the reasoning above needs redoing --
    that is what it is here to detect, not the tail defect itself.
    """
    _dirty_shared_memory(sageattn, float("inf"))
    worst = _sweep(
        sageattn, SHORT_LENS + RAGGED_LENS,
        "sm89 fp8++ dispatcher default", rtol_budget=0.10,
    )
    print(f"  worst mean_rtol over {len(SHORT_LENS + RAGGED_LENS)} lens: {worst:.4f}")


def test_h3_production_tail_is_clean():
    """Claim: the unpredicated fp8 V load stays in-bounds at production scale.

    The sub-tile cases above run at toy sizes. This one runs H3's actual
    packed-sequence config, paired so the tail is the only variable: an exact
    multiple of BLKK beside the same scale with a ragged tail. 109126 leaves
    a 6-row tail in a 64-row tile, which is the most exposed ratio the layout
    can produce.

    109126 is a 362-frame packed length and 362 frames is **not renderable**
    -- the reference implementation rejects past 15.0 s measured after the
    17n+5 snap, so the ceiling is 345 frames (S = 104,030 at 1344x768, a
    30-row tail). Kept at 109126 deliberately rather than lowered to the
    shipped shape: a 6-row tail reads 58 padded rows against the shipped
    shape's 34, so this is the strictly harsher case and it still exercises
    the same unpredicated load. Do not read the number as a production
    geometry; it is a stress shape chosen for its tail ratio.

    This is about V, not the predicated loader the rest of this file covers.
    `load_fp8_V_global_to_share` (qk_int_sv_f8_cuda_sm89.cuh:266) takes no
    predicate at all, and upstream's comment at line 263 warns that it assumes
    V is padded. It is: `per_channel_fp8` allocates at (kv_len+63)//64*64 and
    `TransposePadPermuteKernel` instantiates with pad_zero=true, which selects
    kFillZero (csrc/fused/fused.cu:293). If either of those ever changes, the
    ragged member of each pair goes non-finite while its exact partner stays
    clean -- which is why the pairs exist rather than a bare sweep.
    """
    pairs = [(37760, 37810), (109120, 109126)]
    failures = []
    for exact, ragged in pairs:
        for seq, label in ((exact, "exact"), (ragged, "ragged")):
            g = torch.Generator(device=DEVICE).manual_seed(seq)
            q = torch.randn((1, H3_HEADS, seq, HEAD_DIM), device=DEVICE, dtype=DTYPE, generator=g)
            k = torch.randn((1, H3_HEADS, seq, HEAD_DIM), device=DEVICE, dtype=DTYPE, generator=g)
            v = torch.randn((1, H3_HEADS, seq, HEAD_DIM), device=DEVICE, dtype=DTYPE, generator=g)
            out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
            n_bad = (~torch.isfinite(out)).sum().item()
            tail = seq % CTA_K
            print(f"  S={seq:>6} ({label}, tail {tail:>2} rows): non-finite {n_bad}")
            if n_bad:
                failures.append(f"S={seq} (tail {tail} rows): {n_bad} non-finite")
            del q, k, v, out
            torch.cuda.empty_cache()
    assert not failures, "H3-scale tail:\n  " + "\n  ".join(failures)


CASES = [
    test_v_tail_does_not_inherit_stale_smem,
    test_sm80_fp16_short_seq_lens,
    test_ragged_lens_stay_clean_above_one_tile,
    test_sm89_fp8_default_path_is_unaffected,
]

# Needs ~9 GiB free for q/k/v plus quant buffers at S=109126.
H3_HEADS = 56
H3_VRAM_BYTES = 9 * 2**30


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping.", file=sys.stderr)
        return 0
    major, minor = torch.cuda.get_device_capability()
    print(
        f"sageattention {getattr(sageattention, '__version__', '?')} "
        f"on sm{major}{minor}, torch {torch.__version__}\n"
    )

    cases = list(CASES)
    free, _ = torch.cuda.mem_get_info()
    if free >= H3_VRAM_BYTES:
        cases.append(test_h3_production_tail_is_clean)
    else:
        print(
            f"Skipping the H3-scale case: needs {H3_VRAM_BYTES / 2**30:.0f} GiB "
            f"free, have {free / 2**30:.1f} GiB.\n"
        )

    failures = 0
    for fn in cases:
        print(f"{fn.__name__}:")
        try:
            fn()
            print("  ok")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, don't mask, unexpected errors
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        print()

    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
