# sage-fork

## TLDR

Local fork of `woct0rdho/SageAttention` (itself a fork of
`thu-ml/SageAttention`) used as an editable install for ComfyUI +
LTX-2.3 video generation on RTX 4090 (sm89 / Ada). History was
squashed at 2026-04-23: assume `main` is ours; upstream is not pulled
from anymore.

We care about exactly one GPU: **sm89 / RTX 40xx / Ada**. Other archs
compile and run (the code is upstream's), but we don't test or debug
them.

## Architecture

This is a CUDA extension + Triton + Python wrapper for int8/fp8
quantized attention. Upstream structure unchanged from the last sync.
Relevant pieces for us:

- `sageattention/core.py` -- `sageattn()` top-level dispatch. On
  sm89 + CUDA >= 12.8, picks `sageattn_qk_int8_pv_fp8_cuda` with
  `pv_accum_dtype="fp32+fp16"` (SageAttention2++).
- `csrc/qattn/pybind_sm80.cpp` + `csrc/qattn/qk_int_sv_f16_cuda_sm80.cu` --
  the SM80 kernel (INT8 QK + FP16 PV, fp32 accum). Forward-compatible to
  Ada. Powers `sageattn_qk_int8_pv_fp16_cuda`.
- `csrc/qattn/pybind_sm89.cpp` + `csrc/qattn/sm89_qk_int8_sv_f8_*.cu` --
  the SM89 kernel set (INT8 QK + FP8 PV, multiple accum variants). Powers
  `sageattn_qk_int8_pv_fp8_cuda{,++}`.
- `sageattention/triton/` -- JIT-compiled Triton kernels. Powers
  `sageattn_qk_int8_pv_fp16_triton`. Works anywhere, slower than CUDA
  on sm89. **The only masked path that's numerically correct** (see
  CHANGELOG "Known kernel bugs").
- `sageattention3_blackwell/` -- subpackage for Hopper/Blackwell.
  Irrelevant to us; leave alone.
- `setup.py` -- builds `_qattn_sm80`, `_qattn_sm89`, `_fused` on a
  typical Ada box after our patch (line 152 adds sm89 to the SM80
  build gate).
- `build.sh` -- our editable-install wrapper. Enforces `VIRTUAL_ENV`,
  pins `uv pip install --python ${VIRTUAL_ENV}/bin/python`, caps
  `MAX_JOBS` at 8. Compiles for arch 8.0;8.6;8.9 by default.

## Install / build

Always active-venv. Never bare `python`:

```bash
source /path/to/venv/bin/activate
cd /path/to/sage-fork
./build.sh                # build + install editable into $VIRTUAL_ENV
./build.sh clean          # wipe prior .so / build/ artifacts first
./build.sh verify         # import-check only, no rebuild
./build.sh full           # add Hopper (9.0) and Blackwell (12.0)
```

Build is 60–90s on an 8-core box with MAX_JOBS=8. Longer if you don't cap.

Confirm the editable install is live (path should point at our source tree):

```bash
${VIRTUAL_ENV}/bin/python -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))"
```

Post-build, run `${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py`
once before the first production LTX gen. Side effect: populates
Triton's on-disk autotune cache for every LTX shape the test
covers, so ComfyUI's first gen after a rebuild skips the ~100-500ms
per-new-shape autotune warmup. `./build.sh` invalidates this cache,
so re-run the test after every rebuild.

## Testing

Standalone scripts (no pytest). Run against the installed sage in
`$VIRTUAL_ENV`, not the source tree directly:

```bash
# Baseline numeric sanity (small shape, ~1s):
${VIRTUAL_ENV}/bin/python tests/test_sageattn.py

# LTX-2.3 shape + kernel sweep (9 shapes x {5 sage + 3 torch} modes, ~30-60s on 4090):
${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py

# Flash-attn 2/3 comparisons (if flash-attn installed):
${VIRTUAL_ENV}/bin/python tests/test_flashattn2.py
${VIRTUAL_ENV}/bin/python tests/test_flashattn3.py
```

`tests/test_sageattn_ltx_shapes.py` is the load-bearing test for LTX
workflows. It characterizes accuracy AND speed per (shape, mode)
using `SDPBackend.EFFICIENT_ATTENTION` as the reference (MATH backend
OOMs at LTX self-attn scale -- ~120 GiB for the full matrix).
Soft-warns when mean_rtol > 0.10. Measures five sage kernels
(fp16_cuda, fp16_triton, fp8_cuda, fp8_cuda++, auto) and three torch
SDPA backends (FLASH, EFFICIENT, CUDNN) in the same run, plus an
`fp8++vs.triton` cross-kernel rtol row on unmasked shapes (sanity
check that AudioLoopHelper's mix-routed fp8++ + triton in one forward
pass doesn't introduce a discontinuity beyond each kernel's own noise
floor; expected ~0.10, warns >0.15). Torch rows are the regression
guard for "did a torch upgrade close the sage perf gap?" questions. First call with a new shape tuple pays one-time
triton autotune warmup (~100ms per tuple); triton caches to disk so
subsequent runs are fast. Expect the first run after `./build.sh` to
be ~0.5-1s slower than subsequent ones.

`tests/repros/` holds minimal standalone repros for defects in this
fork's kernels that we haven't fixed yet. They double as regression
tests once we do fix them.

## Conventions

- Python: **always uv**. Never `pip`, never bare `python3`. Build
  script uses `${VIRTUAL_ENV}/bin/python` directly.
- JSON: **orjson**, never stdlib `json`.
- **No emojis** in any file or output.
- Comments: only non-obvious WHY. Don't explain what well-named code
  already does.
- **Never push without being asked.** Origin is the maintainer's
  personal GitHub fork; the maintainer decides when to force-push
  after any history rewrite.

## What's ours vs what's upstream

Upstream-from-woct0rdho code (unmodified unless noted):
- `csrc/`, `sageattention3_blackwell/`, `pyproject.toml`, `bench/`,
  `tests/test_sageattn.py`, `tests/test_flashattn{2,3}.py`.
- `sageattention/` mostly unmodified except
  `sageattention/triton/attn_qk_int8_per_block.py` (we added autotune).
- `setup.py` mostly unmodified except line 152 (sm89 → SM80 build gate).

Our additions and modifications (tracked in CHANGELOG.md):
- `setup.py:152` -- one-line tuple change so `_qattn_sm80` builds on
  sm89 boxes (was gated on 8.0/8.6/8.7 only; Ada is forward-compat to
  SM80).
- `sageattention/core.py::sageattn_warmup(shapes, kernels=...)` --
  public API that fires one-shot dispatches per (kernel, shape) to
  prime Triton's JIT + autotune cache. Cuts ~1s first-call latency on
  sm89 to ~2ms post-warm. Defaults to the Triton kernel only (CUDA
  kernels are build-time compiled, no warmup benefit). Consumer nodes
  call this at model-patch time.
- `sageattention/triton/attn_qk_int8_per_block.py` -- `@triton.autotune`
  over `num_warps` and `num_stages`. Zero immediate perf delta on
  sm89 + LTX shapes (hardcoded config was already optimal) but
  forward-compatible: catches future kernel/triton/shape shifts.
- `build.sh` -- editable-install wrapper with VIRTUAL_ENV check,
  `--python` pin, MAX_JOBS cap.
- `tests/test_sageattn_ltx_shapes.py` -- LTX-parametrized accuracy +
  perf measurement across sage kernels AND torch SDPA backends
  (FLASH / EFFICIENT / CUDNN). Doubles as a regression guard for
  "did a torch upgrade close the sage perf gap?"
- `tests/repros/repro_cuda_mask_kernel.py` -- minimal repro for the
  CUDA mask-path missing-feature documented in CHANGELOG.
- `CHANGELOG.md` -- local divergence + Known kernel bugs + Open work.
- `README.md` -- minimal; attribution only.
- `CLAUDE.md` -- this file.

Git history was squashed to a single "Fork baseline" commit with our
changes layered on top. All safety-backup branches have been deleted;
the squashed history is the canonical state. `origin/main` still
carries the pre-squash 196-commit upstream history; the next push
will need `git push --force-with-lease origin main`.

## The consumers on the other side

Two consumer paths matter:

1. **KJNodes (`kijai/ComfyUI-KJNodes`)** -- the general path most
   ComfyUI users hit. `PathchSageAttentionKJ` is a dropdown of sage
   modes. On `auto` it calls sage's top-level `sageattn()` dispatcher,
   which routes masked calls to Triton internally -- so it dodges the
   CUDA mask-path gap transparently. Explicitly overriding to
   `_cuda` / `sageattn3*` modes bypasses the dispatcher and exposes
   the bug. The second node, `LTX2MemoryEfficientSageAttentionPatch`,
   only patches LTX's `attn1` (self-attn), which doesn't carry a mask
   in LTX -- so scope alone makes it safe.

2. **Our own downstream ComfyUI custom-node** -- patches the model's
   attention with explicit routing. Defaults to `auto_mask_aware`:
   masked calls go to Triton (route around the CUDA mask bug),
   unmasked go to the user-chosen kernel (default fp8++). Different
   tradeoff from KJ: lets callers pick a fast non-Triton kernel for
   the unmasked half while staying correct on masked calls.

Routing policy lives in the consumer nodes. Sage-fork stays
primitive -- kernels only, no policy.

## If we ever need to fix a sage bug ourselves

We own this fork; there's no upstream to send PRs to anymore. If a
kernel defect blocks the LTX workflow:

1. Build a minimal repro in `tests/repros/<name>.py`.
2. Find the kernel in `csrc/qattn/`. sm80 = fp16 PV, sm89 = fp8 PV.
3. Mask-handling code is in the `pybind_sm*.cpp` files (PyTorch entry
   points) and the `.cu` files (kernel body).
4. Rebuild via `./build.sh` and re-run the repro.
5. Add a CHANGELOG entry under `[Unreleased] / Fixed` with the repro
   reference.

We deliberately have no CI. Verify by running the LTX-shape test and
the full downstream-consumer pytest suite on this box before trusting
a change.

## Linter / pyright noise to ignore

Every edit to `sageattention/` or `tests/` triggers these false
positives:

- `Import "torch" could not be resolved` -- pyright's default scan
  env doesn't have torch; our ComfyUI venv does. Runtime is fine.
- `"q_int8" / "k_int8" / "lse_correction" / "sm80_compile" is
  possibly unbound` in `core.py` around lines 325, 594, 598, 601 --
  pre-existing upstream control-flow pyright can't prove. Not
  introduced by any edit here. Ignore.
- `"sageattn_*" is not accessed` in `__init__.py` -- public
  re-exports; pyright doesn't model star-import consumers.

If pyright flags something inside code we actually added and it
looks substantive, investigate. Otherwise skip.

## fp16 matmul accumulation flag

`torch.backends.cuda.matmul.allow_fp16_accumulation` (available in
torch 2.7.1+; exposed by KJ's `CheckpointLoaderKJ` as
`enable_fp16_accumulation`): does NOT affect sage's internals.

Verified 2026-04-24:

- Sage's Q @ K^T and P @ V are done inside its own int8/fp8
  CUDA/Triton kernels via tensor cores. No cuBLAS on any path -- grep
  of `csrc/` finds zero `cublas` references.
- Sage does call `torch.matmul` exactly once, in
  `core.py::lse_correction` (when `smooth_k=True` AND
  `return_lse=True`). That path isn't taken by the ComfyUI
  `optimized_attention_override` hook, which never asks for LSE.
- Net effect on attention: zero.

What the flag DOES affect: torch's own matmuls for the Q/K/V/output
linear projections around attention (those go through cuBLAS). For
LTX-2.3 that's ~5-10% of total gen time in the linear layers. Safe
to enable for speedup; impact on sage attention output is nil.

## Compile / torch.compile

Not used. The downstream consumer node wraps sage in
`torch.compiler.disable()`. The fork's compile support is patchy (the
squashed history contained several compile-related revert cycles) and
attention kernels dominate cost anyway -- compile's folding of
adjacent ops would be marginal. Revisit only if compiling the whole
model becomes a goal.

## Related

- `README.md` -- attribution (woct0rdho, thu-ml).
- `CHANGELOG.md` -- our diff from upstream, plus Known issues.
- The downstream consumer (a ComfyUI custom-node) owns routing policy,
  tracing telemetry, and workflow integration. Its own CLAUDE.md has
  the LTX-2.3 audio-loop patch-chain analysis and test matrix.
