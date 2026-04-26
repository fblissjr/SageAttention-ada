# sage-fork

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
   live here so kernel-side decisions (autotune coverage,
   mask-kernel work, FlashInfer / SpargeAttn comparisons) can be
   made with numbers. Consumers handle their own routing policy; the
   fork stays primitive (kernels + bench).

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

Always active-venv. Never bare `python`. Use `${VIRTUAL_ENV}/bin/python`
or `${VENV}/bin/python` directly when `VIRTUAL_ENV` isn't exported in
the shell. `python -m pip freeze` fails on uv-managed venvs (no pip
module installed) -- use `VIRTUAL_ENV=<venv> <venv>/bin/uv pip freeze`
for env snapshots.

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
`$VIRTUAL_ENV`, not the source tree directly. Easiest path: the
one-shot runner.

```bash
./tests/run_all.sh                     # env snapshot + ltx + image + spike
VENV=/path/to/venv ./tests/run_all.sh  # explicit venv
```

Individual tests if you want one specifically:

```bash
# LTX-2.3 shape + kernel sweep (head_dim=64; ~30s on 4090):
${VIRTUAL_ENV}/bin/python tests/test_sageattn_ltx_shapes.py

# Image-gen shape sweep (head_dim ∈ {120, 128}; Flux-class + Z-Image-Turbo):
${VIRTUAL_ENV}/bin/python tests/test_sageattn_image_shapes.py

# torch.compile compatibility spike (re-run after torch upgrades):
${VIRTUAL_ENV}/bin/python tests/spike_torch_compile.py

# Upstream's flash-attn comparisons (if flash-attn installed):
${VIRTUAL_ENV}/bin/python tests/test_flashattn2.py
${VIRTUAL_ENV}/bin/python tests/test_flashattn3.py
```

`tests/test_sageattn.py` exists upstream as a one-shape sanity test;
mostly subsumed by `test_sageattn_ltx_shapes.py` (which covers a
broader sweep including the same kind of small self-attn shape). Run
it only if you want a 1-second smoke check before the longer bench.

Shape coverage today: head_dim in {64 (LTX, in `test_sageattn_ltx_shapes.py`),
128 (Flux-class) and 120 (Z-Image-Turbo S3-DiT), both in
`test_sageattn_image_shapes.py`}. sage's CUDA kernels handle all three
cleanly on sm89 -- including the non-power-of-2 d=120 (verified
2026-04-25). If a new model class brings a different head_dim, add a
row to the appropriate file before assuming compatibility.

`tests/test_sageattn_ltx_shapes.py` is the load-bearing test for LTX
workflows. It characterizes accuracy AND speed per (shape, mode)
using `SDPBackend.EFFICIENT_ATTENTION` as the reference (MATH backend
OOMs at LTX self-attn scale -- ~120 GiB for the full matrix).
Soft-warns when mean_rtol > 0.10. Measures five sage kernels
(fp16_cuda, fp16_triton, fp8_cuda, fp8_cuda++, auto) and three torch
SDPA backends (FLASH, EFFICIENT, CUDNN) in the same run, plus an
`fp8++vs.triton` cross-kernel rtol row on unmasked shapes (sanity
check that a consumer's mix-routed fp8++ + triton in one forward
pass doesn't introduce a discontinuity beyond each kernel's own noise
floor; expected ~0.10, warns >0.15). Torch rows are the regression
guard for "did a torch upgrade close the sage perf gap?" questions. First call with a new shape tuple pays one-time
triton autotune warmup (~100ms per tuple); triton caches to disk so
subsequent runs are fast. Expect the first run after `./build.sh` to
be ~0.5-1s slower than subsequent ones.

`tests/repros/` holds minimal standalone repros for defects in this
fork's kernels that we haven't fixed yet. They double as regression
tests once we do fix them.

GPU OOM mid-test usually means contention, not a real bug. Check
`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`
before debugging — a sibling process (e.g. ComfyUI loading a model)
likely holds the VRAM Triton autotune needs (~256 MiB for the small
telemetry test; multiple GiB for the LTX bench).

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
- **Consumer-agnostic framing in committed material.** Anything checked
  in (README, CLAUDE.md, CHANGELOG, code comments, CLI labels, commit
  messages) refers to downstream callers as "downstream consumer" or
  "consumer" -- generic. Do **not** name specific consumer projects
  or custom nodes by name. Model targets are different: naming the
  *model class* the bench supports (LTX 2.3, Z-Image-Turbo, Flux,
  etc.) is fine and useful. The distinction:
    - model class -> name it (bench shapes, head_dim coverage)
    - consumer project -> generic phrasing
  Real-world validation runs against private consumer projects /
  workflows; that work stays out of this repo so this repo stays
  focused on kernels + bench. If you find yourself typing the name
  of a specific custom node here, stop and rephrase.
- **Project-internal phase numbers don't ship.** "Phase 0", "Phase 5",
  "TDD red-first", task IDs -- those belong in the plan file or
  commit messages, not in shipped code, CLI output, or CHANGELOG.
  Operators reading this repo months later don't have the plan
  context.
- **Path discipline.** Every committed path is repo-relative. Absolute
  home paths and tilde-prefixed external paths leak. Use generic
  placeholders (`<repo_root>`, `/path/to/venv`, `<some-path>`) or just
  drop the prefix (`./scripts/foo.sh`). `internal/` is gitignored and
  exempt from anything ever shipping. The `path-privacy` plugin's
  pre-commit and commit-msg hooks hard-block the leak class; they're
  installed in this repo and run automatically. If a hook fires, edit
  the file to remove the absolute portion -- do not bypass.
  Belt-and-suspenders manual scan before every commit (catches strings
  the hook's regex doesn't, e.g. consumer-project names):
  `git add <files> && git diff --cached | grep -nE '/home/|~/dev|~/ComfyUI|fbliss|/Users/'`  <!-- path-privacy: ignore -->
  -- empty output = clean. (The scan-pattern line above is itself a
  documented regex source; the trailing token tells the path-privacy
  hook to skip it.)
- **Session logs append, never overwrite.** `internal/log/log_<date>.md`
  is gitignored and may already exist when a new session starts on
  the same day (see today's file with multiple `## Update N — ...`
  sections). Append a new `## Update N — <topic> (<time-of-day>)`
  section at the bottom rather than rewriting the file. Earlier-in-day
  work is the audit trail across sibling sessions.
- **Local-machine config in `internal/local_config.json`** (gitignored).
  Resolution order for any test/script that needs host:port or local
  paths: CLI arg > env var > `local_config.json` > hard error pointing
  at the runbook. Don't hardcode local-machine values in committed
  code; this is the documented escape hatch. Established 2026-04-26
  by `tests/bench_e2e_ltx.py`; schema in
  `internal/runbook_bench_e2e_ltx.md`.
- **`coderef/` is gitignored** alongside `internal/`. Used as a local
  working area for symlinks/clones of consumer source trees we want to
  grep against (verifying our public-API claims, reading their
  scratch.md, etc.). Repo cloners won't have it; that's by design.
- **Verify aspirational doc claims against actual code.** Twice in one
  session (dispatcher mask routing, `sageattn_warmup` "consumers call
  it") a public-API doc claimed "X is used by Y" or "dispatcher does
  Z" — both turned out to be aspirational, not implemented. One
  `grep -r "<api_name>" coderef/` for consumer call sites + a quick
  read of the dispatch code catches this in seconds. Audit trail in
  `internal/audit_2026-04-26.md`.

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
  prime Triton's JIT + autotune cache. Defaults to the Triton kernel
  only (CUDA kernels are build-time compiled, no warmup benefit).
  **Status (verified 2026-04-26):** available API; no consumers in
  our coordinated set currently call it. The mechanism (Triton
  autotune cache hit on subsequent calls) is real; the "~1s → ~2ms"
  perf claim circulating in earlier docs was the documented
  mechanism, not a measured number from our box. Treat as an opt-in
  optimization we offer; remove from this list if no consumer adopts
  it within ~6 months.
- `sageattention.get_last_dispatched_kernel() -> Optional[KernelName]`
  -- public helper exposing which kernel the most recent `sageattn*`
  call on this thread dispatched to, as a stable short string
  (`fp16_triton`, `fp8_cuda++`, etc.; full set in
  `KNOWN_KERNEL_NAMES`, type alias in `KernelName`). Lets consumer
  tracers record sage's routing decision instead of mirroring the
  dispatch table or treating it as opaque. Backed by a
  `threading.local()` set inside each entry point's dispatch branch.
  Read immediately after the sage call -- thread-local, not
  contextvar-aware. Test:
  `tests/test_dispatched_kernel_telemetry.py`. **Adding a new kernel
  variant requires three coupled edits in `core.py`:** a new
  `KERNEL_*` string constant, the matching entry in the
  `KNOWN_KERNEL_NAMES` frozenset, and the matching string in the
  `KernelName = Literal[...]` alias. Forgetting the Literal silently
  breaks consumer type-checking but not runtime; forgetting the
  frozenset silently breaks consumer `assert kernel in
  KNOWN_KERNEL_NAMES` validators. The constant + set + Literal trio
  is the public contract.
- `sageattention/triton/attn_qk_int8_per_block.py` -- `@triton.autotune`
  over `num_warps` and `num_stages`. Zero immediate perf delta on
  sm89 + LTX shapes (hardcoded config was already optimal) but
  forward-compatible: catches future kernel/triton/shape shifts.
- `sageattention/core.py::sageattn` -- v0.3.0 dispatcher mask-routing
  fix. Pre-fix the dispatcher routed purely by arch and silently
  dropped `attn_mask` (CHANGELOG / Known kernel bugs). Post-fix:
  pulls `attn_mask` out of `**kwargs` before the arch branch and
  short-circuits to `sageattn_qk_int8_pv_fp16_triton` when non-None.
  Forwards remaining `**kwargs` (with `kwargs.setdefault` on
  dispatcher-set keys like `pv_accum_dtype`) so non-mask kwargs are
  no longer silently swallowed.
- `sageattention/core.py::_warn_if_mask_passed_to_cuda_kernel` --
  v0.3.1 soft-warn helper. Hooked into the three CUDA entry-point
  wrappers (`sageattn_qk_int8_pv_fp16_cuda`,
  `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp8_cuda_sm90`)
  right after the assert block. Catches consumers who bypass the
  dispatcher and hand-pick a `_cuda` kernel with a non-None mask.
  Soft (warns, not raises) so `attn_mask=None` defensive callers
  aren't penalized.
- `build.sh` -- editable-install wrapper with VIRTUAL_ENV check,
  `--python` pin, MAX_JOBS cap.
- `tests/test_sageattn_ltx_shapes.py` -- LTX-parametrized accuracy +
  perf measurement across sage kernels AND torch SDPA backends
  (FLASH / EFFICIENT / CUDNN). Doubles as a regression guard for
  "did a torch upgrade close the sage perf gap?"
- `tests/test_dispatched_kernel_telemetry.py` -- standalone-script
  test for the `get_last_dispatched_kernel()` helper. 11 tests
  covering: exports, initial-None state, dispatcher-routes-to-fp8++
  on sm89 (unmasked), dispatcher-routes-masked-calls-to-triton
  (v0.3.0 invariant), `pv_accum_dtype` override honored
  (v0.3.1 regression test), per-variant kernel name strings,
  hand-picked-_cuda-warns-on-mask (v0.3.1 soft-warn),
  thread-local isolation. Pure Python; no rebuild needed after
  edits to `core.py` or `__init__.py`.
- `tests/test_sageattn_ltx_shapes.py::cross_attn_unmasked_kv226_kratio_probe`
  -- v0.3.1 K-ratio probe row. Same shape as `cross_attn_text_kv226`
  but unmasked, so `K = triton_masked_ms / fp8++_unmasked_ms` is
  readable from two bench rows. Gates the deferred native CUDA
  mask kernel work (Backlog).
- `tests/bench_e2e_ltx.py` -- v0.4.0 end-to-end gen-time bench.
  Submits a workflow via the ComfyUI HTTP API, runs N times sage-on
  vs sage-disabled, reports wall-time speedup +
  attention-fraction-of-step. Closes the framework's "kernel ms is
  not gen ms" gap. Host resolved from CLI / `$COMFYUI_HOST` /
  `internal/local_config.json` (no hardcoded default). See
  `internal/runbook_bench_e2e_ltx.md` for the operational runbook.
- `tests/repros/repro_cuda_mask_kernel.py` -- minimal repro for the
  CUDA mask-path missing-feature documented in CHANGELOG.
- `CHANGELOG.md` -- versioned divergence + Known kernel bugs + Backlog + Decision log + Recurring process items.
- `README.md` -- minimal; attribution only.
- `CLAUDE.md` -- this file.

Git history was squashed to a single "Fork baseline" commit with our
changes layered on top. All safety-backup branches have been deleted;
the squashed history is the canonical state. `origin/main` still
carries the pre-squash 196-commit upstream history; the next push
will need `git push --force-with-lease origin main`.

## The consumer surface

Sage exposes two surfaces to downstream consumers:

1. **`sageattn()` top-level dispatcher.** Picks a kernel based on
   `(detected arch, CUDA version, mask presence)`. On sm89 + CUDA >=
   12.8 with no mask: lands on `sageattn_qk_int8_pv_fp8_cuda` with
   `pv_accum_dtype="fp32+fp16"` (sage 2++). With `attn_mask` passed:
   routes to `sageattn_qk_int8_pv_fp16_triton` regardless of arch,
   because that's the only mask-correct path (see CHANGELOG / Known
   kernel bugs). Implementation: `sageattention/core.py::sageattn`
   pulls `attn_mask` out of `**kwargs` before the arch branch and
   short-circuits to triton when non-None. The mask-routing claim is
   enforced by a test
   (`tests/test_dispatched_kernel_telemetry.py::test_sageattn_dispatcher_routes_masked_calls_to_triton`)
   that fails the moment the dispatcher reverts to arch-only routing.
   **Most consumers should just call this and let dispatch decide.**
2. **Specific kernel exports** -- `sageattn_qk_int8_pv_fp16_cuda`,
   `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`,
   etc. Bypass the dispatcher; the caller picks. **Masked attention
   only works with `_fp16_triton`**; passing `attn_mask` to a `_cuda`
   kernel silently drops the mask and produces numerically wrong
   output. The CUDA wrappers accept `attn_mask` via `**kwargs` (a
   pre-existing upstream signature shape) but never read it. If a
   consumer is hand-picking a CUDA kernel and also passing a mask,
   that's a consumer bug -- the dispatcher is the safe default.

The dispatcher's mask-routing fix landed v0.3.0 (2026-04-26). Prior
to that the dispatcher routed purely by arch and silently dropped
`attn_mask` -- the docs claimed otherwise but the code didn't match.
See `internal/audit_2026-04-26.md` for the audit trail. If we ever
add a kernel-side mask implementation (Backlog item), the dispatcher
gets a second look: it's possible we'd want `attn_mask` to land on
the native CUDA path for some shapes once the mask kernel exists.

Beyond mask routing, sage-fork stays primitive. The bench harness
lives here; consumer-facing routing policy beyond the mask gap (e.g.
shape-specific kernel preference) belongs in the consumer.

## If we ever need to fix a sage bug ourselves

We own this fork; there's no upstream to send PRs to anymore. If a
kernel defect blocks the LTX workflow:

1. Build a minimal repro in `tests/repros/<name>.py`.
2. Find the kernel in `csrc/qattn/`. sm80 = fp16 PV, sm89 = fp8 PV.
3. Mask-handling code is in the `pybind_sm*.cpp` files (PyTorch entry
   points) and the `.cu` files (kernel body).
4. Rebuild via `./build.sh` and re-run the repro.
5. Add a CHANGELOG entry under the latest version block (Fixed
   subsection) with the repro reference.

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

## Performance research: the load-bearing metric

When you're trying to make this fork perform better, ALL perf
decisions on the sm89 box are graded against one row of one test:

```
tests/test_sageattn_ltx_shapes.py
  shape: self_attn_large_704x704x497  (B=1, H=32, Sq=Skv=31776, D=64, no mask, bf16)
  mode:  fp8_cuda++
  -> primary perf metric: median_ms (today: 19.95 ms)
  -> accuracy guard:      mean_rtol ≤ 0.10 (today: ~0.097)
```

Anything else you might want to measure is secondary, useful as a
guard against side effects, or explicitly ignored — see below.

### Why this is the metric (the load-bearing reasoning)

The chain matters; if any link breaks, the metric moves.

1. **LTX self-attn at 31776×31776 dominates real gen wall-time.**
   On LTX-2.3 video gen, this single attention shape accounts for
   the overwhelming majority of attention cost per step. Cross-attn
   kv ≤ 1024 is sub-millisecond per call; image-gen shapes
   (head_dim ∈ {120, 128}) are 1-2 ms. The 31776×31776 row is where
   milliseconds compound into seconds of gen time.
2. **`fp8_cuda++` is what `sageattn()` picks on sm89 + CUDA ≥ 12.8
   unmasked.** That's the consumer's actual hot path — the
   dispatcher routes there for self-attn after the v0.3.0 mask-aware
   fix. Optimizing a kernel that the dispatcher doesn't pick is
   research that doesn't ship.
3. **The fp8++ kernel is where every plausible perf change lands.**
   Edits to `csrc/qattn/sm89_qk_int8_sv_f8_*.cu`,
   `sageattention/quant.py` (per-block / per-warp INT8 quant), the
   fp8 V-quant `scale_max` in `core.py:914-918`, or the SM89 PV
   accumulator variants all show up on this row. Triton-side
   changes show up only on the cross-attn rows.
4. **The README's "<0.1 mean rtol" promise is the accuracy ceiling.**
   If a perf change pushes mean_rtol > 0.10, the fork's documented
   accuracy floor is gone. That's not a tradeoff to make silently;
   it's a re-pitching of the fork.

### How we measure it

`tests/test_sageattn_ltx_shapes.py` is the only thing you need to
run. The bench's `time_median_ms` does 1 warmup + median over 3 runs
to kill within-session noise; absolute median_ms is the
within-session signal you optimize against during a research sitting.

For comparing across sessions (after a torch / triton / CUDA / driver
bump, or after a cold boot), use the **`torch_flash / sage_fp8++`
ratio** instead of absolute time (today: 2.62x). The ratio
normalizes against driver-thermal drift, which is on the order of
1-2% across cold boots even with no code changes — see CHANGELOG's
cu128→cu130 transition note. If absolute fp8++ time drifts but the
ratio holds, it's the box, not the code.

Bench env (torch / triton / CUDA / sage rev) pinned to
`internal/bench_env_<date>.txt`; resnapshot after any version bump
per "Bench env discipline" below.

### How we detect unintended side effects

The harness already prints every check side-by-side in one run. Read
all of these every time you change a kernel — don't tunnel-vision on
the primary row.

- **All 5 sage kernels + 3 torch backends on every shape.** A change
  that helps fp8++ but hurts fp16_cuda or fp16_triton means you
  shifted a knob that's shared between code paths; either intentional
  or a foot-gun.
- **The cross-attn-with-mask kv sweep (32, 64, 128, 226, 512, 1024).**
  Catches regressions in the masked path. Pre-v0.3.0 the dispatcher
  silently dropped masks here; now it routes to triton. The triton
  row's rtol should stay ≈0.04 across the sweep; CUDA rows stay
  pinned at the documented mask-bug fingerprint (0.94→0.13).
- **The cross-kernel `fp8++ vs triton` rtol row** (unmasked shapes
  only). Should sit ≈0.10 — quadrature of each kernel's independent
  ~0.04 / ~0.09 vs SDPA. If it spikes above 0.15, a kernel-internal
  numerical change broke the cross-kernel agreement, even if neither
  kernel's solo rtol-vs-SDPA changed.
- **Image-gen shapes** in `tests/test_sageattn_image_shapes.py`
  (head_dim=120 Z-Image, head_dim=128 Flux). A kernel change keyed on
  head_dim=64 might silently break the non-power-of-2 d=120 path.
- **Dispatcher telemetry** (`tests/test_dispatched_kernel_telemetry.py`).
  Verifies routing invariants — the `auto` row matching the wrong
  kernel name post-change is the v0.3.0 mask-routing regression
  signal in primitive form.
- **`tests/run_all.sh`** runs all of the above in one shot. Use it
  before declaring a perf change done.

### How we use the metric to pick what to try next

The bench output is also a diagnostic for where to spend the next
research hour. Five patterns to look for:

1. **Where kernels disagree on rtol, the gap is the optimization
   target.** If fp8_cuda++ shows 0.097 rtol at 19.95ms and fp16_cuda
   shows 0.04 rtol at ~22ms, the 0.057 rtol delta is "FP8
   quantization cost." The research question becomes: is there a
   variant of fp8 quant (scale_max, granularity, per-block Q mean,
   etc.) that closes some of that gap at similar speed? If you
   measure two fp8 variants and they're indistinguishable in rtol,
   you're at the FP8 information floor and should look elsewhere.
2. **Where kernels agree, you're at the numeric floor — stop
   optimizing the kernel and look elsewhere.** Two kernels with
   different code paths producing the same number means the
   underlying numerics, not the implementation, is the bottleneck.
   Move up the stack: torch.compile around sage, fusion with
   adjacent ops, model-side activation reformulation.
3. **Speedup-ratio degradation tells you which torch path got
   better.** If `torch_flash / sage_fp8++` drops from 2.62x to 1.8x
   on a future torch release, torch closed gap somewhere — check
   the `torch_flash`, `torch_eff`, `torch_cudnn` row that improved
   most and figure out what changed. That's where fp8++ is leaving
   perf on the table.
4. **Outliers in the kv sweep are kernel-boundary effects.** A
   cross-attn rtol that breaks the smooth `1/seq_kv` trend at one
   specific kv is a block-size or autotune-config artifact. Target
   the outlier with a focused experiment; don't change global
   defaults.
5. **The unmasked-vs-masked timing gap quantifies the deferred CUDA
   mask kernel.** Today triton is the only mask-correct path; if
   `triton @ kv=N` is K× slower than `fp8++ @ kv=N` (unmasked) at
   the same shape, K is the speedup ceiling for the deferred Backlog
   item "Add mask support to the sm80/sm89 CUDA kernels." If K < 2x,
   the kernel work probably isn't worth days of effort. If K > 5x at
   shapes the consumer actually hits, the trigger fires. The probe
   row `cross_attn_unmasked_kv226_kratio_probe` in
   `tests/test_sageattn_ltx_shapes.py` exists specifically so K is
   measurable -- it pairs with `cross_attn_text_kv226` (same shape,
   masked) so K = triton_masked_ms / fp8++_unmasked_ms is just two
   numbers from the bench output. **Measured 2026-04-26:** K ≈ 1.68
   at kv=226, K ≈ 2.0 at kv=1024 -- both below the 5x trigger, so
   the deferred kernel work is not perf-justified today. Re-measure
   after every kernel-side optimization that lands on the unmasked
   cross-attn path; if fp8++ at small kv gets meaningfully faster,
   K grows and the trigger could fire even with no triton change.

### What we explicitly ignore — and the trigger that would change that

These rows print every run; we don't optimize against them today.
Each one comes with a re-evaluation trigger so we don't keep
ignoring them after the world changes.

- **`fp8++ vs triton` cross-kernel rtol row as a perf signal.**
  It's a consistency check, not a speed measurement. **Trigger to
  care:** the row spikes above 0.15, indicating mixed-route
  consumer forward passes are now seeing a discontinuity beyond
  combined-noise.
- **Cross-attn-with-mask perf rows.** Triton at sub-millisecond is
  already fast enough that perf wins on this path don't move real
  gen time. **Trigger to care:** a downstream consumer's per-call
  JSONL trace, aggregated over a real production gen, reports
  masked-triton as >5% of total gen wall-time. (See CHANGELOG
  Recurring process items / "Session-level attention telemetry
  summary.")
- **Image-gen perf rows.** Already 1.7-2.1x faster than `torch_flash`
  on Flux + Z-Image-Turbo; not the hot path for the sm89 box's
  primary workload. **Trigger to care:** a new model class lands
  with shapes that show < 1.3x speedup, or a consumer reports image
  gen wall-time is now attention-bound.
- **`torch_eff` and `torch_cudnn` rows.** Regression telemetry for
  "is sage still load-bearing as a fork?" — not a perf signal for
  sage changes. **Trigger to care:** sage's speedup ratio drops below
  1.5x on the primary row, which triggers a "is the fork still worth
  maintaining?" review rather than a perf experiment.
- **Spike `tests/spike_torch_compile.py` perf delta.** Verdict on
  torch 2.11: keep the consumer-side `torch.compiler.disable()`.
  **Trigger to care:** re-run after any torch upgrade; the spike
  itself records the reopen condition (bounded rtol AND measurable
  speedup).

### What we might be wrong about (the framework is V1)

This metric reflects the workload mix on this box as of 2026-04-26.
We may be wrong in ways that take time to surface; record the
disconfirming evidence rather than wait for it to be obvious.

- **The "LTX self-attn dominates" assumption is workload-specific.**
  If a new model class with fundamentally different attention
  patterns (very short autoregressive seq, sliding-window, MQA/GQA
  with very different head ratios) becomes the primary use case, the
  load-bearing shape moves and the metric should be re-derived.
  Disconfirming signal: a downstream-consumer telemetry summary
  showing a non-LTX-class shape consuming > 30% of gen attention
  time.
- **Mean rtol is a proxy for "does the output look right," not the
  truth.** A perf change that improves mean_rtol but visibly
  degrades a real render fails the spirit of the guard. We don't
  currently have a perceptual eval in this repo (it'd be a new
  bench, probably keyed on per-frame structural similarity vs an
  fp32 reference render). If we ever ship a kernel change that
  passes the rtol guard but causes a consumer-reported visual
  regression, the rtol guard isn't the right floor and we need to
  add the perceptual layer.
- **Kernel ms is not gen ms.** A 2x kernel speedup is invisible
  end-to-end if attention is already < 50% of step time. We don't
  measure end-to-end here (it's downstream-consumer telemetry); a
  refinement would be a `gen_wall_time / attention_kernel_time`
  ratio captured per-render, so kernel improvements get an
  end-to-end translation factor. Until then, a "this saved 5ms per
  call" claim should be paired with "and we observed a real LTX gen
  go from X seconds to Y seconds" before ranking high.
- **The "find next experiment" framework above is forward-looking;
  we haven't run a perf experiment through it yet.** The first time
  we use it to pick a direction and either succeed or fail, the
  framework gets refined. Treat the five patterns as starting
  hypotheses, not validated playbook.

If any of the above bullets fires (disconfirming signal observed),
update this section and record the change in the session log so the
re-derivation is auditable later.

## Bench env discipline

Every wall-clock comparison in `test_sageattn_ltx_shapes.py` is pinned
to the version surface in `internal/bench_env_<date>.txt`. After any
torch/triton/CUDA/sage-rev bump, re-run the test and resnapshot. Trigger
doc + drift threshold: `CHANGELOG.md` / Recurring process items /
"Bench env re-snapshot."

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

- `VISION.md` -- canonical scope doc. What this fork is, what it
  isn't, the single load-bearing metric, what we might be wrong
  about. Rare edits; update when the fork's philosophy shifts.
- `README.md` -- what changed vs upstream, what was measured, what
  tradeoffs come with using it. The longer per-feature explanation
  surface; cross-linked from VISION.md.
- `CHANGELOG.md` -- versioned divergence + Known kernel bugs +
  Backlog + Decision log + Recurring process items. Source of truth
  for closed decisions.
- `internal/PLAN.md` (gitignored) -- live operational doc: current
  state, active backlog with triggers, cross-repo coupling,
  experiment log (TSV), the research loop. Edit every session.
  Mirrors CHANGELOG's Backlog and Recurring sections in active
  form. Pairs with `internal/log/log_<date>.md` (narrative session
  notes) and `internal/audit_<date>.md` (durable findings).
- A downstream ComfyUI consumer (any custom node patching attention)
  owns routing policy, tracing telemetry, and workflow integration.
  Sage-fork stays primitive: kernels and the bench harness only.
