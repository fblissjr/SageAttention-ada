# sage-fork

L1 routing index. Detailed material lives in `docs/` (committed) and
`internal/` (gitignored); see "Deeper context" at the bottom.

## TLDR

Local fork of `woct0rdho/SageAttention` (itself a fork of
`thu-ml/SageAttention`). Two purposes:

1. **Editable install** for any PyTorch project that wants sage on
   sm89 / RTX 40xx / Ada. ComfyUI is one common consumer, but the
   fork is consumer-agnostic: anything that imports `sageattention`
   or replaces `torch.nn.functional.scaled_dot_product_attention`
   picks it up. The packaging-regression fix in setup.py is the
   load-bearing reason this fork exists at all.
2. **Experimentation and measurement surface** for sm89 attention
   kernel decisions. The LTX-shape bench harness
   (`tests/test_sageattn_ltx_shapes.py`) and torch.compile spike
   live here so kernel-side decisions can be made with numbers.
   Consumers handle their own routing policy; the fork stays
   primitive (kernels + bench).

We care about exactly one GPU: **sm89 / RTX 40xx / Ada**. Other archs
compile and run via the dispatcher's sm100/sm120/sm121 fallback to
the sm89 kernel, but we don't test or debug them. We do not carry
Hopper/Blackwell-specific kernels (all removed in v0.5.0). Windows
install paths are also gone; build is Linux+source only.

## Architecture

CUDA extension + Triton + Python wrapper for int8/fp8 quantized
attention. Relevant pieces:

- `sageattention/core.py` -- `sageattn()` top-level dispatch. On
  sm89 + CUDA >= 12.8, picks `sageattn_qk_int8_pv_fp8_cuda` with
  `pv_accum_dtype="fp32+fp16"` (SageAttention2++).
- `csrc/qattn/pybind_sm80.cpp` + `qk_int_sv_f16_cuda_sm80.cu` --
  SM80 kernel (INT8 QK + FP16 PV). Forward-compatible to Ada.
- `csrc/qattn/pybind_sm89.cpp` + `sm89_qk_int8_sv_f8_*.cu` -- SM89
  kernel set (INT8 QK + FP8 PV, multiple accum variants).
- `sageattention/triton/` -- JIT Triton kernels. The only masked
  path that's numerically correct (CHANGELOG / Known kernel bugs).
- `setup.py` -- builds `_qattn_sm80`, `_qattn_sm89`, `_fused`. Our
  patch at line 152 adds sm89 to the SM80 build gate.
- `build.sh` -- editable-install wrapper. Enforces `VIRTUAL_ENV`,
  `--python` pin, MAX_JOBS cap.

Full upstream-vs-ours inventory in `docs/whats_ours_vs_upstream.md`.

## Install / build

Always active-venv. Never bare `python`. Use `${VIRTUAL_ENV}/bin/python`
or `${VENV}/bin/python` directly. `python -m pip freeze` fails on
uv-managed venvs -- use `VIRTUAL_ENV=<venv> <venv>/bin/uv pip freeze`
for env snapshots.

```bash
source /path/to/venv/bin/activate
cd /path/to/sage-fork
./build.sh                # build + install editable into $VIRTUAL_ENV
./build.sh clean          # wipe prior .so / build/ artifacts first
./build.sh verify         # import-check only, no rebuild
```

Build is 60-90s on an 8-core box with MAX_JOBS=8.

Confirm install (path should point at our source tree):

```bash
${VIRTUAL_ENV}/bin/python -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))"
```

Post-build, run `${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py`
once before the first production LTX gen. Side effect: populates
Triton's on-disk autotune cache for every LTX shape the test covers,
so the first gen after a rebuild skips the ~100-500ms per-new-shape
autotune warmup. `./build.sh` invalidates this cache.

**First `--check-regression` after `./build.sh` is expected to fail**
on triton-autotune-pending rows (200-300% drift on sub-millisecond
rows is typical -- autotune sweep dominates the median). Run the
bench once without the flag to populate the cache, then re-run with
`--check-regression` for the gate.

### `tests/bench_e2e_ltx.py` warmup auto-detection

The e2e bench has `--warmup {auto,always,never}` (default `auto`).
Auto-mode skips the warmup-and-discard prompt only when BOTH:
1. A non-empty `coderef/.../data/runs/<RUN_ID>/sage.jsonl` exists
   on disk with mtime < 30 min, AND
2. ComfyUI's `/history/1` HTTP endpoint returns a non-empty dict.

Either signal alone is unreliable. If you suspect the auto-detection
is wrong, pass `--warmup always` explicitly. Asymmetric-cost
reasoning: false-positive (skip warmup when cold) -> cold-start
measurement bias -> bench reads as "sage 0.5x SLOWER"; false-negative
-> wasted 250s. Always errs toward warmup.

## Testing

Standalone scripts (no pytest). Run against the installed sage in
`$VIRTUAL_ENV`, not the source tree directly.

```bash
./tests/run_all.sh                     # env snapshot + ltx + image + spike
VENV=/path/to/venv ./tests/run_all.sh  # explicit venv

# Individual:
${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py
${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py --check-regression
${VIRTUAL_ENV}/bin/python tests/test_sageattn_image_shapes.py
${VIRTUAL_ENV}/bin/python tests/spike_torch_compile.py
${VIRTUAL_ENV}/bin/python tests/test_flashattn2.py     # if flash-attn installed
${VIRTUAL_ENV}/bin/python tests/test_flashattn3.py
```

Shape coverage is derivable -- run `tests/bench_workload_profile.py`
against a recent consumer trace and read its "Coverage gaps" section.
Bench-shape changes have their own discipline; see
`docs/bench_discipline.md`.

`tests/test_sageattn_ltx_shapes.py` is the load-bearing test. It
characterizes accuracy AND speed per (shape, mode) using
`SDPBackend.EFFICIENT_ATTENTION` as the reference (MATH backend
OOMs at LTX self-attn scale). Soft-warns when mean_rtol > 0.10.
Measures five sage kernels and three torch SDPA backends in one run,
plus an `fp8++vs.triton` cross-kernel rtol row.

`tests/repros/` holds minimal standalone repros for kernel defects.

GPU OOM mid-test usually means contention, not a bug. Check
`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`
before debugging -- a sibling process likely holds the VRAM Triton
autotune needs.

## Conventions

- Python: **always uv**. Never `pip`, never bare `python3`.
- JSON: **orjson**, never stdlib `json`.
- **No emojis** in any file or output.
- Comments: only non-obvious WHY.
- **Never push without being asked.** Origin is the maintainer's
  personal fork.
- **Consumer-agnostic framing in committed material.** Refer to
  downstream callers as "downstream consumer" / "consumer" --
  generic. Model class is fine (LTX 2.3, Flux, Z-Image-Turbo); a
  specific custom node by name is not. Two narrow carve-outs where
  the name is itself load-bearing: the downstream-known-symbols
  audit (specific importer) and measurement provenance (workflow
  filename in perf claims).
- **Project-internal phase numbers don't ship.** Belong in the plan
  file, not in code / CLI / CHANGELOG.
- **Path discipline.** Every committed path is repo-relative. The
  `path-privacy` plugin's pre-commit and commit-msg hooks hard-block
  leaks; don't bypass. Belt-and-suspenders manual scan:
  `git add <files> && git diff --cached | grep -nE '/home/|~/dev|~/ComfyUI|fbliss|/Users/'`  <!-- path-privacy: ignore -->
- **Session logs append, never overwrite.** `internal/log/log_<date>.md`
  may already exist on the same day -- append a new `## Update N --
  <topic> (<time-of-day>)` section.
- **Local-machine config in `internal/local_config.json`** (gitignored).
  Resolution order: CLI arg > env var > `local_config.json` > hard
  error pointing at the runbook. Don't hardcode local-machine values
  in committed code.
- **`coderef/` is gitignored** alongside `internal/`. Holds
  symlinks/clones of consumer source trees for verification. Use
  proactively as a verification surface (perf-mechanism claims,
  aspirational API doc claims) -- discipline in
  `docs/perf_research_framework.md`.
- **Perf-mechanism claims need both arms measured.** A *number* can
  come from one measurement; a *mechanism* claim needs both A/B
  arms directly instrumented. Full rule + the v0.5.1 retirement
  story in `docs/perf_research_framework.md`.

## The consumer surface

Sage exposes two surfaces to downstream consumers:

1. **`sageattn()` top-level dispatcher.** Picks a kernel based on
   `(detected arch, CUDA version, mask presence)`. On sm89 + CUDA >=
   12.8 unmasked: lands on `sageattn_qk_int8_pv_fp8_cuda` with
   `pv_accum_dtype="fp32+fp16"`. With `attn_mask` passed: routes to
   `sageattn_qk_int8_pv_fp16_triton` regardless of arch, because
   that's the only mask-correct path. Implementation:
   `sageattention/core.py::sageattn` pulls `attn_mask` out of
   `**kwargs` before the arch branch. The mask-routing claim is
   enforced by a test in `tests/test_dispatched_kernel_telemetry.py`
   that fails the moment the dispatcher reverts to arch-only routing.
   **Most consumers should just call this and let dispatch decide.**
2. **Specific kernel exports** -- `sageattn_qk_int8_pv_fp16_cuda`,
   `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`,
   etc. Bypass the dispatcher; caller picks. **Masked attention only
   works with `_fp16_triton`**; passing `attn_mask` to a `_cuda`
   kernel silently drops the mask and produces numerically wrong
   output. CUDA wrappers accept `attn_mask` via `**kwargs` but never
   read it. If a consumer hand-picks a CUDA kernel + mask, that's a
   consumer bug -- the dispatcher is the safe default.

Mask-routing fix landed v0.3.0 (2026-04-26); audit trail in
`internal/audit_2026-04-26.md`.

There is also an undocumented L3 contract -- underscore-prefixed
symbols and pybind methods that downstream consumers import by name.
Before removing or renaming any of those, read
`docs/downstream_symbols.md` and run the pre-removal checklist.

## Performance research

ALL perf decisions are graded against one row of one test:
`tests/test_sageattn_ltx_shapes.py`, shape
`ltx23_video_self_attn_init_22932`, mode `fp8_cuda++`. E2e ratios are
workload-dependent (attention share varies). Full framework --
metric, reasoning chain, side-effect checks, experiment-selection
patterns, ignore-triggers, "what we might be wrong about",
pre-trigger briefing -- in `docs/perf_research_framework.md`. Load
that before running a perf experiment.

## Compile / torch.compile

Not used. Consumer wraps sage in `torch.compiler.disable()`. The
spike rejects on rtol drift, not perf. Two pybind kernels Dynamo
graph-breaks at, the trigger to revisit, and the estimated work in
`docs/torch_compile_spike.md`.

## Deeper context (L3 references)

- `docs/perf_research_framework.md` -- load-bearing metric, reasoning
  chain, side-effect checks, five experiment-selection patterns,
  ignore-triggers, uncertainty record, mechanism-claim + aspirational-
  claim discipline, prior-recording, pre-trigger briefing.
- `docs/whats_ours_vs_upstream.md` -- file-by-file inventory: upstream
  unmodified, removed-in-v0.5.0, our additions + status of each.
- `docs/downstream_symbols.md` -- de-facto public surface (underscore
  symbols + pybind methods), known importers, pre-removal checklist.
- `docs/sage_bug_fix_workflow.md` -- 5-step procedure when a kernel
  defect blocks a workflow: minimal repro -> locate kernel in
  csrc/qattn/ -> patch -> rebuild -> CHANGELOG entry.
- `docs/bench_discipline.md` -- env snapshot rules, cross-session
  ratio comparison, before-changing-bench-shapes workflow,
  regression_baselines.json source-of-truth rule.
- `docs/fp16_matmul_accum.md` -- whether KJ's
  `enable_fp16_accumulation` affects sage output (no).
- `internal/pyright_noise.md` (gitignored) -- pyright false-positives
  to ignore in `sageattention/` and `tests/`.

## Related

- `VISION.md` -- canonical scope doc. What this fork is, what it
  isn't, the single load-bearing metric, what we might be wrong
  about. Rare edits.
- `README.md` -- attribution + minimal user-facing summary.
- `CHANGELOG.md` -- versioned divergence + Known kernel bugs +
  Backlog + Decision log + Recurring process items. Source of truth
  for closed decisions.
- `internal/PLAN.md` (gitignored) -- live operational doc: current
  state, active backlog with triggers, cross-repo coupling,
  experiment log. Edit every session. Mirrors CHANGELOG's Backlog
  and Recurring sections in active form. Pairs with
  `internal/log/log_<date>.md` and `internal/audit_<date>.md`.
- `.claude.local.md` (gitignored) -- personal companion to this
  file. Holds the specific local-machine details that would leak
  in committed material: active venv path, consumer-install
  location, `coderef/` symlink targets.
- A downstream ComfyUI consumer (any custom node patching attention)
  owns routing policy, tracing telemetry, and workflow integration.
  Sage-fork stays primitive: kernels and the bench harness only.
