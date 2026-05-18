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
- `sageattention/triton/` -- JIT Triton kernels. Mask-correct on
  all archs (the only mask-correct path before v0.5.5; still gates
  archs that haven't gained native CUDA mask support). Also home
  of v0.6 `fused_mlp_fp8.py` -- the `sage_ffn` two-kernel fp8 MLP
  primitive (FFN-side, not attention).
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

Reuse `accuracy_metrics` from `tests/test_sageattn_ltx_shapes.py:160`
for rtol/atol comparisons (symmetric denominator; matches every
other accuracy bench in the repo, including `tests/test_partitioned.py`).
Scripts under `tests/spikes/` need a one-line
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
before the import.

For bit-identicality checks on bf16/fp16 outputs (e.g. stream-safety
spikes), use `torch.equal(a.view(torch.uint16), b.view(torch.uint16))`
-- bare `torch.equal(a, b)` returns False whenever either tensor has
NaN even at identical bit patterns, and random mockup weights at LTX
shapes frequently produce NaN positions. The uint16-view sidesteps
NaN-equality semantics. Worked example:
`tests/spikes/spike_concurrent_dispatch_submodule.py::correctness_sanity`.

Spike scripts under `tests/spikes/` wrap measurement loops in
`torch.inference_mode()` (stricter than `no_grad` -- drops
version-counter tracking; matches the sampler/consumer path under
which these kernels actually run). Add NVTX ranges
(`torch.cuda.nvtx.range("label")`) on every measurement region so
`nsys profile` timeline view shows labeled kernels in addition to
raw aggregation. Both conventions applied in
`tests/spikes/spike_concurrent_dispatch{,_submodule}.py`.

`tests/repros/` holds minimal standalone repros for kernel defects.

GPU OOM mid-test usually means contention, not a bug. Check
`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`
before debugging -- a sibling process likely holds the VRAM Triton
autotune needs.

**Peak-HBM cumulative measurement benches: do NOT precede the
cumulative arm with a per-call-reset arm that does
`gc.collect`/`empty_cache` between calls.** The reset arm trains the
pytorch caching allocator into a state that biases the cumulative
number downward. Caught 2026-05-13 by /simplify in
`tests/bench/partitioned_mask_phase0/`; the pre-fix bench
underreported the K-quant+V-cast redundancy delta by ~535 MiB and
nearly shipped wrong numbers to a downstream consumer. Symptom:
cumulative-with-mask and cumulative-no-mask measurements that look
identical at a shape where they shouldn't.

## Conventions

- Python: **always uv**. Never `pip`, never bare `python3`.
- JSON: **orjson**, never stdlib `json`.
- **No emojis** in any file or output.
- Comments: only non-obvious WHY.
- **Never push without being asked.** Origin is the maintainer's
  personal fork.
- **Retract wrong-framing in committed docs via `git revert`, not
  in-place edit.** The revert preserves the wrong commit + its
  message in `git log` and supersedes it with a revert commit on
  top; the audit trail of "we believed X, then disproved X" stays
  reconstructible. An in-place edit leaves the wrong framing in the
  diff history as an unflagged precursor that future `git log -p
  <file>` would surface without context. Worked example: the
  2026-05-16 "47% comfy-aimdo offload" workload-profile claim
  (commit `a05fdf4`) retracted via `git revert` at `95af2cf` after
  a `nodynvram` A/B disproved the framing.
- **Consumer-agnostic framing in committed material.** Refer to
  downstream callers as "downstream consumer" / "consumer" --
  generic. Model class is fine (LTX 2.3, Flux, Z-Image-Turbo); a
  specific custom node by name is not. Two narrow carve-outs where
  the name is itself load-bearing: the downstream-known-symbols
  audit (specific importer) and measurement provenance (workflow
  filename in perf claims).
- **Task/caller refs in committed code don't age.** Memo timestamps
  ("07:45Z"), cross-clone references ("per X's note"), and
  session-specific framings ("today's spike showed") decay -- the
  memo trail isn't in the repo, and a reader six months later can't
  reconstruct context. Replace with the substantive reason: cite the
  production precedent, the cross-version stability concern, or the
  file:line of the canonical source. Easy to introduce; easy to
  scrub in /simplify; the second cycle is wasted effort.
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
- **Kernel signature changes are a four-place coupling on the sm89
  path.** Adding a runtime param to a sm89 kernel (e.g. v0.5.5
  `attn_mask`) needs (a) the `.cuh` template + kernel-launch sites
  in all 7 sm89 `.cu` files, (b) the C++ entry + `attn_cuda_sm89.h`
  decl, (c) pybind def with `py::arg(...)=c10::nullopt` defaults,
  (d) `sageattention/sm89_compile.py::@torch.library.custom_op`
  schema + matching `register_fake` stub. Forget (d) and the call
  fails at runtime with "expected at most N argument(s) but
  received N+1" -- pybind alone isn't enough. Worked example:
  CHANGELOG v0.5.5 + the kernel-correctness-reviewer agent.
- **Gate ship-decisions on in-pipeline A/B when synthetic-bench
  can't measure the dominant cost.** Two layers to this rule:
  (a) Don't *claim* a delivered speedup from a synthetic number;
  default framing is "synthetic-bench above, e2e pending in-pipeline
  measurement" until a downstream A/B confirms the wedge transfers.
  (b) For kernel-day work with structural risk that synthetic-bench
  *specifically* can't measure (L2 contention with neighboring
  modules in the production hot loop, cumulative dispatch overhead
  at high call counts, memory-allocator behavior under fragmentation,
  thermal/clock state during sustained renders), gate the v0.X ship
  commit on an in-pipeline measurement BEFORE the commit lands, not
  after. The v0.6 walk-back was the cost of running this rule
  ship-first-validate-later.
  Two precedents: v0.5.5 chunk-bypass A/B (synthetic mask-kernel win
  softened once `LTXVChunkFeedForward` was shown to be doing the
  load-bearing memory work) and v0.6 sage_ffn (synthetic 1.26-1.36x
  came back +1.79% e2e slower on a two-sampler LTX FML2V workflow
  due to L2 contention + cumulative launch overhead). Especially
  load-bearing for per-call-heavy primitives (FFN/MLP fire ~1000
  times per LTX render -- any per-call overhead compounds and any
  cache-locality assumption made under isolation can break).
- **Triton kernel-day discipline.** Four recurring traps:
  (a) `@triton.jit` can't read module-level Python globals -- inline
      literals in the kernel body (e.g. `448.0` for FP8_E4M3_MAX).
  (b) For DiT FFN/MLP kernels, audit BOTH Linear layers for `bias=True`
      on the target checkpoint. LTX 2.3 distilled has bf16 biases on
      both `ff.net.0.proj` and `ff.net.2`; shipping a bias-free kernel
      silently corrupts output (not "fp8 quant noise" wrong, "missing
      a constant offset everywhere" wrong). Caught pre-day-9 in v0.6.
  (c) Broad `@triton.autotune` sweeps (>~30 configs) burn minutes of
      first-render-per-shape on user hardware (126 configs = ~7 min
      cold). Pattern: tune full sweep once, extract winners via
      `kernel.cache.items()`, hardcode ~8 configs + neighbors. Worked
      example: v0.6 sage_ffn (CHANGELOG v0.6.0).
  (d) CUDA event timing across streams captures queue-wait + execution,
      not just kernel duration. When `e_start.record()` is on the
      default stream and `e_end.record(s_other)`, `elapsed_time` between
      them measures `s_other`'s wait-for-SMs plus the kernel. A `t_ms`
      variable name telegraphs "this is kernel time"; future readers
      will misread. Use `*_end_offset_ms` or similar to signal the
      asymmetry vs single-stream timing. Worked example:
      `tests/spikes/spike_concurrent_dispatch.py` (renamed in /simplify
      pass after the bare metric misread).
  (e) Raw CUDA kernel launches default to stream 0 if the 4th arg is
      omitted. `<<<grid, block, smem>>>` silently breaks any caller that
      wraps sage in `with torch.cuda.stream(...)` -- Triton kernels in
      the same Python call respect current stream, the CUDA launch
      doesn't, and the race surfaces as small-but-stable rtol drift
      (~0.02) or NaN under a partial fix that only patches the attn
      kernel but leaves `csrc/fused/fused.cu`'s quant pre-kernels on
      stream 0. Use `<<<grid, block, smem,
      at::cuda::getCurrentCUDAStream()>>>` and
      `#include <ATen/cuda/CUDAContext.h>` on every site (`csrc/qattn/`
      sm89/sm80 + `csrc/fused/fused.cu`). Worked example: v0.6.1
      (CHANGELOG).

## The consumer surface

Sage exposes three surfaces to downstream consumers:

1. **`sageattn()` top-level dispatcher.** Picks a kernel based on
   `(detected arch, CUDA version, mask presence)`. On sm89 + CUDA >=
   12.8 unmasked: lands on `sageattn_qk_int8_pv_fp8_cuda` with
   `pv_accum_dtype="fp32+fp16"`. With `attn_mask` passed: routes to
   the same `sageattn_qk_int8_pv_fp8_cuda` (the v0.5.5 native CUDA
   mask path); other archs still route to `sageattn_qk_int8_pv_fp16_triton`
   since their CUDA kernels haven't gained mask support yet.
   Implementation: `sageattention/core.py::sageattn` pulls `attn_mask`
   out of `**kwargs` before the arch branch and bifurcates on
   `(arch, cuda_version, mask_present)`. The routing invariant is
   enforced by a test in `tests/test_dispatched_kernel_telemetry.py`.
   **Most consumers should just call this and let dispatch decide.**
2. **Specific kernel exports** -- `sageattn_qk_int8_pv_fp16_cuda`,
   `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`,
   etc. Bypass the dispatcher; caller picks. **Masked attention works
   on sm89 fp8++ (`pv_accum_dtype="fp32+fp16"`) as of v0.5.5** and on
   the Triton kernel; other CUDA variants (sm80 fp16, sm89 non-fp8++)
   still silently drop the mask and warn. If a consumer hand-picks a
   non-mask-correct CUDA kernel + mask, the v0.3.1 soft-warn fires;
   the dispatcher is the safe default.
3. **`sage_ffn(x, w1, s1, w2, s2, b1=None, b2=None)`** (v0.6) -- a
   separate FFN primitive, not an attention kernel. Two-kernel
   Triton fp8 MLP (`Linear(fp8) -> GELU(tanh) -> Linear(fp8)`)
   targeting LTX 2.3-class FFN blocks (hidden=4096, inner=16384,
   per-tensor fp8 E4M3FN weights, optional bf16 biases on both
   Linear layers). Synthetic bench shows 1.26-1.36x vs torch's
   fp8-dequant reference; **in-pipeline A/B on a two-sampler LTX
   workflow came back +1.79% e2e slower (+20% per-call at stage-2)**,
   so this ships as a completeness primitive, not a perf win. Root
   cause is L2 cache contention with neighboring attention modules
   + cumulative kernel-launch overhead at LTX's ~1000-FFN-calls/render
   count. Not wired into `sageattn()`; consumer imports it directly
   from the top-level package. The qualitative wedge holds (no other
   library ships fp8-native fused MLP for ComfyUI consumer-app on
   sm89); the quantitative wedge does not on the tested workload.
   v0.6.1 candidates to close the gap: persistent-CTA hybrid and
   CUTLASS-based CUDA backend (see CHANGELOG Backlog).

Mask-routing fix landed v0.3.0 (2026-04-26); audit trail in
`internal/audit_2026-04-26.md`. Native CUDA mask landed v0.5.5
(2026-05-13) on sm89 fp8++; scoping doc + measurement trail in
`docs/cuda_mask_kernel_scoping.md`. FFN
fusion landed v0.6.0 (2026-05-15); scoping + day-by-day execution
journal + cross-claude memo trail in
`internal/design/ffn_fusion_scoping.md` (gitignored).

There is also an undocumented L3 contract -- underscore-prefixed
symbols and pybind methods that downstream consumers import by name.
Before removing or renaming any of those, read
`docs/downstream_symbols.md` and run the pre-removal checklist.

## Performance research

Two complementary inputs for any perf decision:

1. **Kernel-isolation gate**: `tests/test_sageattn_ltx_shapes.py`,
   shape `ltx23_video_self_attn_init_22932`, mode `fp8_cuda++`.
   This is the single load-bearing row for "does this kernel work
   at speed X at this shape" -- the isolation question. Synthetic
   kernel-bench number; do not promote as delivered consumer-app
   speedup.
2. **E2e leverage input**: `docs/ltx_workload_profile.md` -- where
   wall-time actually lives in the FML2V multi-guide workflow.
   Canonical source for sub-module shares + the FFN-share triplet
   (total / video-only / stage-2-only readings). Cite that doc
   rather than restating percentages locally; the next render-data
   refresh updates one place. The framework in
   `docs/perf_research_framework.md` says: measure attention-share-
   of-CUDA-time on each workload of interest, apply Amdahl with the
   per-kernel ratio observed on that workload's actual call mix.

E2e ratios are workload-dependent (attention share varies). Full
framework -- reasoning chain, side-effect checks, experiment-
selection patterns, ignore-triggers, "what we might be wrong about",
pre-trigger briefing -- in `docs/perf_research_framework.md`. Load
that before running a perf experiment.

Snapshot `torch` + `triton` versions right before any commit that
cites perf numbers -- env can drift mid-session via the venv's uv
pip activity, and a CHANGELOG number is only honest under the
stack that produced it. One-liner: `${VIRTUAL_ENV}/bin/python -c
"import torch, triton; print(torch.__version__,
torch.version.cuda, triton.__version__)"`.

## Compile / torch.compile

Not used. Consumer wraps sage in `torch.compiler.disable()`. The
spike rejects on rtol drift, not perf. Two pybind kernels Dynamo
graph-breaks at, the trigger to revisit, and the estimated work in
`docs/torch_compile_spike.md`.

## Deeper context (L3 references)

- `docs/perf_research_framework.md` -- load-bearing metric, reasoning
  chain, side-effect checks, five experiment-selection patterns,
  ignore-triggers, uncertainty record, mechanism-claim + aspirational-
  claim discipline, prior-recording, pre-trigger briefing, evidence
  ladder for kernel-replacement audits (kernel-name presence >
  per-call logs > attribution coverage > sub-module time delta;
  every fallback path needs a log line).
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
- `docs/ltx_workload_profile.md` -- canonical FML2V render
  breakdown + FFN-share triplet. Use this for ranking perf bets by
  leverage.
- `docs/fp16_accum_fp8_matmul.md` -- analysis of why fp16-accum
  fp8 matmul throughput work (LinkedIn-article-style "473 TFLOPS
  at LLM shape") doesn't help LTX FFN-class workloads. Throughput
  claim is real and independently replicated; the per-MMA
  accumulator overflow constraint kills the rtol budget on DiT
  activation distributions specifically.
- `internal/pyright_noise.md` (gitignored) -- pyright false-positives
  to ignore in `sageattention/` and `tests/`. Two recurring categories
  worth knowing up front: "unreachable code" on `@triton.jit` kernel
  bodies, and "is not accessed" on package `__init__.py` re-exports.
  Both are persistent; don't try to fix.

## Related

- `VISION.md` -- canonical scope doc. What this fork is, what it
  isn't, the single load-bearing metric, what we might be wrong
  about. Rare edits.
- `docs/roadmap.md` -- forward-looking record of directions worth
  pursuing, tiered by relevance to the current workload + trigger-
  conditional. Not a committed schedule; the user remains the
  scheduler. Edited when the option space shifts (a tier item gets
  promoted to CHANGELOG Backlog, demoted to Decision-log, or a new
  candidate is enumerated).
- `README.md` -- attribution + minimal user-facing summary.
- `CHANGELOG.md` -- versioned divergence + Known kernel bugs +
  Backlog + Decision log + Recurring process items. Source of truth
  for closed decisions.
- `internal/PLAN.md` (gitignored) -- live operational doc: current
  state, active backlog with triggers, cross-repo coupling,
  experiment log. Edit every session. Mirrors CHANGELOG's Backlog
  and Recurring sections in active form. Pairs with
  `internal/log/log_<date>.md` and `internal/audit_<date>.md`.
- Scoping-doc precedent for kernel-day work that needs a discipline
  check (PTX bit-identity diff of the kNone specialization, register-
  pressure read, four-place-coupling audit) BEFORE committing to the
  full implementation. Cheap investigation that de-risks the kernel
  work; produces an effort-estimate refinement that ages better than
  the "days, not hours" rule of thumb in CHANGELOG / Backlog. Public
  worked example shipped at `docs/cuda_mask_kernel_scoping.md`
  (v0.5.5); current gitignored work-in-progress at
  `internal/design/ffn_fusion_scoping.md` (v0.6).
- `.claude.local.md` (gitignored) -- personal companion to this
  file. Holds the specific local-machine details that would leak
  in committed material: active venv path, consumer-install
  location, `coderef/` symlink targets.
- A downstream ComfyUI consumer (any custom node patching attention)
  owns routing policy, tracing telemetry, and workflow integration.
  Sage-fork stays primitive: kernels and the bench harness only.
