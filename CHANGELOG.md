# Changelog

Local divergence from the `woct0rdho/SageAttention` fork this repo is
based on. Format follows [Keep a Changelog](https://keepachangelog.com/).
Semver only, no dates. `[Unreleased]` is the working state of `main`.

## Known kernel bugs

Real defects we've measured in this fork's kernels. We own the fork now;
these are ours to fix when we want to. If you're debugging
sage-attention-adjacent correctness problems, start here.

### CUDA kernels have no general attention-mask support

Not a bug to patch — a feature that was never implemented, and
inherited from the `thu-ml/SageAttention` origin by every downstream
fork. The Python wrappers `sageattn_qk_int8_pv_fp16_cuda` and
`sageattn_qk_int8_pv_fp8_cuda` accept `attn_mask` via `**kwargs` but
never pass it through to the C++ layer. The C++ `MaskMode` enum only
has `{kNone, kCausal}`. Masks are silently dropped on all CUDA code
paths.

The same pattern exists in `sageattention3_blackwell/sageattn3/api.py`:
`sageattn3_blackwell(q, k, v, attn_mask=None, ...)` declares the
parameter but never references it; the Blackwell kernel layer
(`csrc/blackwell/`) only exposes `is_causal` + sliding-window-causal
via `window_size_left/right`. So the mask gap is present across sage
2.x AND sage 3 — the Triton kernel
(`sageattn_qk_int8_pv_fp16_triton`) remains the only numerically
correct mask path in the entire lineage.

Observable effect on LTX-2.3 shapes (bf16, heads=32, head_dim=64,
seq_q=31776, varying seq_kv with ~30-position text-padding tail):
rtol 0.26–0.94 across the tested seq_kv range; NaN at very short
seq_kv (32) with proportionally small pad_tail (16).
`sageattn_qk_int8_pv_fp16_triton` has proper mask plumbing and is
correct (rtol ~0.04 across the range).

Repro: `tests/repros/repro_cuda_mask_kernel.py`.

Kernel sources to touch when we fix this:
- `sageattention/core.py:439-451, 616-628` — Python entry points where
  the mask gets dropped into `**kwargs` and never extracted.
- `csrc/qattn/pybind_sm80.cpp`, `csrc/qattn/pybind_sm89.cpp` — pybind
  signatures that would need a new `attn_mask` parameter.
- `csrc/qattn/attn_cuda_sm80.h`, `attn_cuda_sm89.h` — kernel declarations;
  `MaskMode` enum needs a `kGeneral` variant.
- `csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`, `csrc/qattn/sm89_qk_int8_sv_f8_*.cu` —
  kernel bodies; mask would be applied to scores before the per-block
  max reduction (see Triton reference in `sageattention/triton/`).

Consumer workaround (sufficient for now): a downstream ComfyUI node
patches the model's attention with a mask-aware router that sends
masked calls to `sageattn_qk_int8_pv_fp16_triton` regardless of the
configured mode.

Discovered: 2026-04-23 via `tests/test_sageattn_ltx_shapes.py` (the
seq_kv sweep exposes the rtol-vs-seq_kv scaling signature; an Explore
pass confirmed the missing-feature root cause).

## Open work

Fork-side TODOs. Each has an explicit trigger-to-act; we're not doing
these speculatively.

### Add mask support to the sm80/sm89 CUDA kernels

Scope, measurement, and consumer workaround are in "Known kernel bugs"
above. Size estimate: days of kernel work, not hours (pybind signature,
new `MaskMode::kGeneral`, kernel-loop mask application, plus perf and
register-pressure regression verification).

**Trigger to act:** triton's cross-attn perf becomes a measured
bottleneck on a real production render (not speculatively).

### sm89 fp8 quantization scale: closed as no-action

**Investigated 2026-04-23.** `sageattention/core.py:772-774` keeps
the fp8_cuda `scale_max` default at `448.0` for `pv_accum_dtype`
variants `"fp32"` and `"fp32+fp32"`, but flips to `2.25` for
`"fp32+fp16"` (the ++ variant, which is what sage's auto dispatch
picks on sm89 + CUDA >= 12.8). KJ's `LTX2MemoryEfficientSageAttentionPatch`
hard-codes `2.25`; the reasoning suggested flipping the non-++ default
to match, for consistency.

Measured via `tests/test_sageattn_ltx_shapes.py` on both LTX shapes
(V ~ N(0, 1)) and a synthetic wide-V shape (V ~ N(0, 5)). fp8_cuda
(`scale_max=448`) and fp8_cuda++ (`scale_max=2.25`) produced
essentially identical mean_rtol: 0.097 on LTX self-attn, 0.097 on
synthetic wide-V. No material difference.

**Decision:** don't flip the default. Two reasons:
1. Auto-dispatch already picks ++ on sm89 + CUDA >= 12.8, so the
   non-++ default only affects callers who explicitly choose
   `pv_accum_dtype="fp32"` or `"fp32+fp32"`. Those callers likely
   picked the older variants for a reason; silently changing v-quant
   behavior on them is worse than matching upstream.
2. The measurement showed equivalence, not improvement. Upside of
   flipping is zero; downside is silent divergence from upstream for
   explicit non-++ callers.

**Trigger to revisit:** a future model or workload shows measurable
quality improvement from `scale_max=2.25` on the non-++ path. Until
then, this is closed.

## [Unreleased]

### Added

- `build.sh` -- local build wrapper that targets whichever venv is
  active via `VIRTUAL_ENV`, pins `uv pip install --python
  ${VIRTUAL_ENV}/bin/python`, compiles for Ampere + Ada
  (`TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`) by default, and verifies the
  built `.so`s. Caps `MAX_JOBS` at 8 to keep high-core boxes from
  OOMing during `_qattn_sm89` compilation.
- `tests/test_sageattn_ltx_shapes.py` -- measures each installed sage
  kernel and three torch SDPA backends (`FLASH_ATTENTION`,
  `EFFICIENT_ATTENTION`, `CUDNN_ATTENTION`) against a common reference
  at LTX-2.3's actual shapes (head_dim=64, heads=32, self-attn +
  cross-attn-with-mask across seq_kv from 32 to 1024, plus a synthetic
  wide-V shape). Reports mean/max rtol+atol and median elapsed per
  (shape, mode). Soft-warns when mean_rtol exceeds the fork README's
  "<0.1 on RTX 40xx/50xx" expectation. The torch-backend rows serve as
  a regression test for future torch releases -- if a torch update
  closes the sage speedup gap this test will say so.
  First-measurement datapoints on RTX 4090 / CUDA 13.2 / torch 2.11 / bf16:
    - self-attn-large (31776x31776, no mask): sage fp8++ 19.67 ms,
      torch_flash 52.39 ms (2.7x slower), torch_cudnn ~360 ms (cuDNN's
      FA3 path is not competitive on sm89).
    - cross-attn + mask (kv=226): sage fp16_triton 0.78 ms, torch_flash
      SKIP (rejects 4D bool mask), torch_cudnn 2.20 ms (2.8x slower).
  Conclusion: sage remains load-bearing on sm89; the
  torch-SDPA-could-displace-sage scenario is retired by measurement.
  Companion to the one-shape `tests/test_sageattn.py`.
- `tests/repros/repro_cuda_mask_kernel.py` -- standalone repro for the
  CUDA mask-path missing-feature documented in Known kernel bugs.
- `CHANGELOG.md` -- this file.
- `CLAUDE.md` -- fork navigation guide.

### Changed (kernel internals)

- `sageattention/triton/attn_qk_int8_per_block.py` -- added
  `@triton.autotune` over `num_warps in {4, 8}` and
  `num_stages in {3, 4, 5}`, keyed on runtime shape. BLOCK_M/BLOCK_N
  stay hardcoded because they're locked by the per-block int8
  quantization step in `sageattention/quant.py` (changing them
  without matched quant changes would misalign scale tables).
  Measurement on RTX 4090 / LTX shapes: autotune confirms the existing
  hardcoded config (`num_warps=4`, `num_stages=3` for `head_dim=64`)
  was already at the optimum -- zero perf delta today. Value is
  structural (auto-adapts to future kernel changes, triton upgrades,
  or new shapes; catches regressions automatically).

### Changed

- `setup.py` -- `_qattn_sm80` is now also built when compute
  capability 8.9 (Ada) is detected. Framed as a regression fix from
  `woct0rdho/SageAttention`: thu-ml's setup.py gates the SM80
  extension on `HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or
  HAS_SM100 or HAS_SM120 or HAS_SM121` (Ampere + Ada + Hopper +
  Blackwell), but woct0rdho's refactor collapsed that to a tuple
  gate `("8.0", "8.6", "8.7")` — which silently drops Ada, Hopper,
  AND Blackwell. Ada-only source builds on woct0rdho's fork lose
  `sageattn_qk_int8_pv_fp16_cuda` (the fp16 fallback path); Hopper
  and Blackwell source builds lose it too. We only added `"8.9"`
  because that's the arch we test and care about here — if you run
  this fork on Hopper or Blackwell and want the fp16 fallback built
  from source, widen the tuple to match thu-ml's coverage.
- `README.md` -- reduced to attribution only (immediate fork:
  `woct0rdho/SageAttention`; original: `thu-ml/SageAttention`) plus a
  short build pointer. Windows-specific installation prose and wheel
  selection guidance removed -- this fork builds from source.
