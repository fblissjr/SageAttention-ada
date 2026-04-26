# Changelog

Local divergence record for this fork. **Not a public release timeline**:
this is a personal editable install used as an attention-kernel
measurement surface for sm89 / RTX 40xx / Ada. The version blocks below
are commit-history snapshots, not semver releases -- they group
changes into coherent chunks for retrospective navigation. There are
no compatibility promises across versions.

Sections:

- **Versions** -- additions and changes layered on top of the
  upstream-from-woct0rdho baseline. Newest first.
- **Known kernel bugs** -- factual record of real defects we've
  measured in this fork's kernels. Start here if you're debugging
  sage-attention-adjacent correctness problems.
- **Backlog** -- real open TODOs with explicit triggers to act.
- **Decision log** -- investigations that closed without action,
  recorded so we don't re-derive them. Each entry has a reopen-trigger.
- **Recurring process items** -- cron-like checks, not engineering work.

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

## Backlog

Real open TODOs. Each has an explicit trigger-to-act; we don't do these
speculatively.

### Extract `tests/_helpers.py` for shared `make_qkv` + `require_cuda`

Four standalone test files now duplicate near-identical scaffolding:
`test_sageattn_ltx_shapes.py`, `test_sageattn_image_shapes.py`,
`spike_torch_compile.py`, `test_dispatched_kernel_telemetry.py`. Each
defines its own `_make_qkv()` (or `make_qkv()`) for building random
QKV tensors, and each opens with the same `if not torch.cuda.is_available(): skip`
guard. Lifting both into `tests/_helpers.py` would consolidate ~30
lines across files; the standalone-script convention (no pytest, no
conftest) means a flat helper module is the right shape, not a
fixture.

**Trigger to act:** next time one of these four test files needs
editing for an unrelated reason. Don't do it speculatively -- the
duplication is currently inert and the import path
(`from _helpers import make_qkv`) needs `tests/` on `sys.path` which
the standalone scripts don't set up today. Mild churn for tiny gain
unless we're already in the file.

### Add mask support to the sm80/sm89 CUDA kernels

Scope, measurement, and consumer workaround are in "Known kernel bugs"
above. Size estimate: days of kernel work, not hours (pybind signature,
new `MaskMode::kGeneral`, kernel-loop mask application, plus perf and
register-pressure regression verification).

**Trigger to act:** triton's cross-attn perf becomes a measured
bottleneck on a real production render (not speculatively).

## Decision log

Investigations that closed without action. Recorded so we don't
re-derive them. Each entry has an explicit reopen-trigger.

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

### Sage 3 per-block Q mean backport: closed as low-impact on sm89 fp8++

**Investigated 2026-04-24** (research spike, not a full backport).

Sage 3's `sageattention3_blackwell/sageattn3/api.py::preprocess_qkv`
adds a preprocessing step sage 2.x lacks: Q is split into groups of
128 tokens along the sequence dim, each group's mean is subtracted
before quantization, and a `delta_s = qm @ K^T` correction tensor is
passed to the kernel for use during softmax reconstruction. Sage 2
only centers K (via `smooth_k=True`) and never centers Q at all.

For FP4 (sage 3's quant format, 16 levels) this is a first-order
precision win. Question: is it worth backporting to sage 2's sm89
fp8++ path, which uses INT8 Q (256 levels)?

**Empirical check** via a standalone INT8 Q quant-roundtrip experiment
(per-block quantize -> dequantize, measure rtol to original fp32):

| Q distribution    | |DC|/std | rtol_baseline | rtol_centered | improvement |
|-------------------|----------|---------------|---------------|-------------|
| typical           | ~0.16    | 0.0363        | 0.0331        | ~9%         |
| skewed (large DC) | ~0.80    | 0.0351        | 0.0252        | ~28%        |

That is Q quant precision. Translating to end-to-end attention rtol
(fp8++ currently measured at ~0.097 vs SDPA on LTX shapes): the Q
quant floor is roughly a third of total rtol budget (rest is FP8 V +
accumulation). A 9% improvement on Q yields ~2-4% end-to-end rtol
improvement -- well below the run-to-run noise floor. A 28%
improvement on Q (for a skewed model) yields ~8-10% end-to-end.

**Decision:** don't backport. Three reasons:

1. LTX's Q activations almost certainly fall into the "typical" DC
   range (normalized transformer activations with modest channel
   biases), where the expected fp8++ rtol improvement is ~0.002-0.004
   absolute -- imperceptible at render level.
2. The kernel-side work is non-trivial: `csrc/qattn/sm89_qk_int8_sv_f8_*.cu`
   would need a new `delta_s` input, modifications to the softmax
   reduction to apply the correction, and matched changes in the
   Python quantization path (`sageattention/quant.py`) to compute and
   pack per-block Q means. Days of work for a sub-noise-floor win on
   our primary workload.
3. Sage 2's existing `smooth_k` already captures the K-centering
   half of the idea. The specific new capability (per-block Q mean)
   is the marginal addition, not the main event.

**Trigger to reopen:**
- LTX's actual Q DC offset measured at |DC|/std > 0.5 in production.
  (Would require instrumenting the LTX model's Q projections;
  a downstream consumer could capture this if its telemetry grows that far.)
- A visible artifact in fp8++ output that isn't explained by bf16
  activations, fp8 weight quant, or VAE.
- A future workload with shorter-bit Q quantization (fp4 or below)
  where per-block mean becomes a first-order win rather than a
  third-order refinement.

## Recurring process items

Cron-like checks; not engineering work. Each one has a frequency or
trigger; act when the trigger fires, otherwise note in passing.

### Bench env re-snapshot

Process item, not engineering work. The `tests/test_sageattn_ltx_shapes.py`
baselines are pinned to a specific (torch, triton, CUDA, sage) version
tuple recorded in `internal/bench_env_<date>.txt`. Re-run the test with
soft-warn enabled and re-snapshot the env file when ANY of these change:

- `torch` major or minor (e.g. 2.11 -> 2.12, or any cuXYZ swap)
- `triton` minor (e.g. 3.6 -> 3.7)
- CUDA toolkit (e.g. 13.0 -> 13.1)
- `sage` git rev — automatic on every `./build.sh`, but worth a fresh
  bench if the rev changed since last measurement

What "real change" means: any (shape, mode) wall-clock that drifted >5%
from the previous snapshot is worth investigating. <5% is run-to-run
noise (we logged 1.4% on the cu128 -> cu130 transition). The rtol
fingerprints (e.g. cross-attn-with-mask 0.94 at kv=32) should be
*invariant* across these upgrades; if they drift, that's a kernel change
upstream and warrants tracing back.

The snapshot file lives under `internal/` (gitignored). Naming convention:
`bench_env_YYYY-MM-DD.txt`. Keep prior snapshots; they're the audit trail
when someone says "this used to be faster."

**Trigger to act:** any version bump in the list above, OR a measured >5%
shape-level drift on a routine workflow gen.

### Session-level attention telemetry summary (consumer side)

Cross-repo backlog item, tracked here because it feeds sage-fork's
mask-kernel work (the "is triton cross-attn a bottleneck?" trigger
above). A consumer-side sage-routing node typically writes a per-call
JSONL row (shape, mode, effective_mode, elapsed_us, fell_back) when an
opt-in trace env var is set. Raw per-call data is straightforward; the
aggregation question is "what percent of gen wall time is masked-triton?"
which is exactly what the mask-kernel trigger above needs.

**Shape of the work (consumer side):** emit a one-line summary at
gen-end: median/p90 elapsed_us for masked-triton calls, total call
count, and that median as a percent of total gen time if measurable.
No new telemetry plumbing required -- just aggregation over the
existing JSONL rows.

**Trigger to act:** when a downstream consumer wants to justify backing
a sage-fork kernel push with data. Until then the raw JSONL is
sufficient.

## Versions

### v0.4.0 -- 2026-04-26  (end-to-end gen-time bench harness)

Closes the load-bearing "kernel ms is not gen ms" gap that the
v0.3.x perf-research framework explicitly flagged. Until this lands,
every claim about sage-fork's perf impact was theoretical -- we
measured 19.95 ms on the primary kernel row but never showed that
translated into a real DiT render moving from X seconds to Y.

#### Added

- **`tests/bench_e2e_ltx.py`** -- end-to-end gen-wall-time bench via
  ComfyUI's HTTP API. Submits an LTX (or Flux / Z-Image) render
  workflow N times sage-on, N times sage-disabled, captures wall
  time per run, reads the consumer's sage trace JSONL (when
  `AUDIOLOOPHELPER_SAGE_TRACE=auto`), and reports:
  - median wall time per arm
  - speedup ratio: `wall_off / wall_on`
  - attention-fraction-of-step on the sage arm
  - interpretation: ≥ 1.5× = sage load-bearing on this workload,
    1.10–1.50× = helps but not dominant, < 1.10× = wash, < 0.95× =
    regression
  
  Prereqs: ComfyUI running, launched with the trace env var, and an
  API-format workflow JSON (saved via UI → Workflow → Save (API
  Format)). The script does not convert UI-format workflows -- the
  conversion is JS-side in ComfyUI's frontend; reimplementing adds
  enough complexity that one click in the UI is the better tradeoff.
  Mode toggle is via the `inputs.mode` field on the
  `AudioLoopHelperSageAttention` node, found by class_type so it's
  resilient to id renumbering across workflow versions.

- **Backlog entry** in `internal/PLAN.md`: "Simplify e2e bench
  correlation once consumer ships RUN_ID + prompt_id." Tracks an
  upcoming consumer-side change that bundles per-session artifacts
  under `data/runs/${RUN_ID}/` and stamps `prompt_id` on each sage
  trace row. When that lands, the bench drops ts-windowing entirely
  (~30 lines deleted; fence-post bugs eliminated; parallel-queue
  resilient). Until then ts-windowing is the correlation primitive.

#### Why this matters more than the kernel-level work that preceded it

The v0.3.x work fixed a real correctness bug (dispatcher mask
routing) and built measurement infrastructure (K-probe row,
soft-warn, telemetry helper). All of that is real but downstream
of an unverified premise: that kernel-level speedup translates to
gen-level speedup at all. This bench is the first instrument that
can verify (or disprove) that premise.

If the first execution shows speedup < 1.10×, the framework's
"kernel ms ≠ gen ms" caveat fires for real and the kernel-side
research priorities reset. If ≥ 1.5×, sage-fork's reason to exist
is empirically grounded for the first time.

### v0.3.1 -- 2026-04-26  (mask-gap follow-ups: soft-warn + K-ratio probe)

Two follow-ups graded against the load-bearing-metric framework added
to CLAUDE.md this session. Both pass the "ship now" bar; one
deliberate deferral got grounded in actual measurement instead of
hand-wave.

#### Added

- **Soft-warn from CUDA wrappers when `attn_mask` is passed.**
  `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`,
  and `sageattn_qk_int8_pv_fp8_cuda_sm90` now call a shared helper
  (`_warn_if_mask_passed_to_cuda_kernel`) that emits a one-time
  `warnings.warn` per source location when a non-None `attn_mask`
  reaches the wrapper. The dispatcher routes masked calls to triton
  automatically since v0.3.0; this guard catches consumers that
  bypass the dispatcher and hand-pick a `_cuda` kernel directly.
  Soft (warn, not raise) so consumers who defensively pass
  `attn_mask=None` aren't penalized -- the warn fires only on real
  masks. Python's default warning filter dedupes by source line, so
  long iteration loops emit one warning total per location, not one
  per call. Test:
  `tests/test_dispatched_kernel_telemetry.py::test_hand_picked_cuda_kernel_warns_when_mask_passed`.
  This was the deferred Solution C from v0.3.0's audit; ship-now
  reasoning recorded in the audit doc.
- **K-ratio probe row in the LTX bench.** New shape
  `cross_attn_unmasked_kv226_kratio_probe` in
  `tests/test_sageattn_ltx_shapes.py`. Same shape as
  `cross_attn_text_kv226` (the typical LTX text-encoder padded length)
  but with no mask. Lets us read off
  `K = triton_masked_ms / fp8++_unmasked_ms` directly from the bench
  output. K is the speedup ceiling for the deferred Backlog item
  "Add mask support to the sm80/sm89 CUDA kernels"; without a probe
  row, K was unmeasurable and the trigger could never fire. **First
  measurement (RTX 4090 / sm89 / CUDA 13.0 / torch 2.11):** K ≈ 1.68
  at kv=226 (triton 0.79 ms vs fp8++ unmasked 0.47 ms), K ≈ 2.0 at
  kv=1024. Both below the framework's 5x trigger; the deferred kernel
  work stays deferred, with an actual number behind it now.

#### Changed

- **CLAUDE.md "Performance research" section** -- the
  unmasked-vs-masked timing-gap framework item now names the K-probe
  row and records the measurement (see Added). Re-measure after any
  kernel-side change that lands on the unmasked cross-attn path.

#### Verified, no action

- **Dispatcher fix end-to-end.** Re-running the LTX bench post-fix
  shows `auto` rows on every masked cross-attn shape now matching
  the `fp16_triton` row to the precision the bench prints
  (mean_rtol 0.0392 / median_ms 0.79 at kv=226; same pattern at
  kv ∈ {32, 64, 128, 512, 1024}). Pre-fix `auto` would have mirrored
  `fp8_cuda++`'s broken fingerprint exactly.

#### Still deferred (with concrete reopen-trigger)

- **D: tighten `**kwargs` to explicit named parameters.** Re-evaluated
  this session; bigger than initially scoped. The dispatcher
  legitimately needs `**kwargs` for forward-compat (kernel-specific
  knobs like `pv_accum_dtype` that callers may want to override).
  Tightening per-kernel signatures creates a real conflict with
  dispatcher-forwarded kwargs. Real API design question with no
  current pain. Trigger unchanged: next time we touch these
  signatures for an unrelated reason.
- **Native CUDA-kernel mask support.** K-probe measurement (above)
  shows K ≈ 1.68-2.0 across the LTX cross-attn kv range. Days of
  kernel work for at most ~2x speedup on a path that's already
  sub-millisecond per call. Trigger is now grounded in a concrete
  per-bench measurement: re-evaluate when K > 5x at a shape a
  consumer actually hits.

### v0.3.0 -- 2026-04-26  (dispatcher mask routing -- correctness fix)

Closes the load-bearing inconsistency between what the fork documented
and what the dispatcher did. README and CLAUDE.md had claimed for
months that `sageattn()` "routes masked calls to the Triton kernel
transparently." The code routed purely by GPU arch and silently
dropped `attn_mask` on every CUDA path. A consumer-side workaround
covered the gap in practice; this version moves the fix to the right
layer (the dispatcher) so every consumer gets it without re-implementing.

Audit trail in `internal/audit_2026-04-26.md` (gitignored) -- captures
how the gap was missed, the alternatives considered, and the
revisit-triggers for the deferred items (loud-raise on hand-picked
CUDA kernels with masks; tightening the `**kwargs` surface; native
CUDA-kernel mask support).

#### Fixed

- **`sageattention/core.py::sageattn`** -- the top-level dispatcher
  now extracts `attn_mask` from `**kwargs` before the arch branch and
  short-circuits to `sageattn_qk_int8_pv_fp16_triton` when it's
  non-None, regardless of GPU arch. Unmasked calls dispatch by arch
  exactly as before. `is_causal=True` still dispatches by arch (CUDA
  kernels handle causal natively via `MaskMode::kCausal`); only
  `attn_mask` triggers the triton route. Side effect: `**kwargs` is
  now forwarded to every per-kernel call, so non-mask kwargs
  (`smooth_k`, `qk_quant_gran`, etc.) stop being silently swallowed.
- **End-to-end accuracy delta on cross-attn-with-mask shapes** (LTX
  cross-attn kv=226 example): `sageattn(q, k, v, attn_mask=m)` mean
  rtol drops from 0.4405 (broken: mask dropped, ran fp8++ unmasked)
  to 0.0391 (correct: routed to fp16_triton). Bare
  `sageattn_qk_int8_pv_fp8_cuda` still shows 0.44 -- the underlying
  CUDA-kernel mask gap is unchanged; this version routes around it,
  not through it. Known kernel bugs entry stays.

#### Added

- `tests/test_dispatched_kernel_telemetry.py::test_sageattn_dispatcher_routes_masked_calls_to_triton`
  -- enforces the new routing rule. Calls `sageattn()` with a
  text-padding-tail mask and asserts
  `get_last_dispatched_kernel() == 'fp16_triton'`. Failed red on the
  pre-fix dispatcher (recorded `'fp8_cuda++'`); passes green after
  the fix. Lives next to the existing dispatcher test so the next
  reader sees both the masked and unmasked invariants enforced
  side by side.

#### Changed

- **CLAUDE.md "The consumer surface"** -- the dispatcher's mask
  behavior is now described as a real implementation with a test
  reference, not as an aspirational claim. Cross-link to the audit
  doc + the new test added.
- **README.md** -- rewrite covering what changed, why, what was
  measured, and what tradeoffs the fork carries. The mask-gap
  language now describes the post-fix behavior (dispatcher routes;
  hand-picked CUDA kernels still drop). No-hype framing, numbers
  cited from `internal/log/log_2026-04-25.md` and the bench harness
  output.

#### Why this wasn't done sooner

The mask gap was discovered 2026-04-23 via the LTX-shape harness's
cross-attn rtol scaling signature. A consumer-side workaround
(downstream ComfyUI node patching the model's attention with a
mask-aware router) landed the same week because that was the fastest
path to a correct render. README + CLAUDE.md picked up an aspirational
"the dispatcher does this transparently" framing that drifted
unchallenged because no test enforced it. The fix here is small (~10
lines) and would have landed earlier if the dispatcher's mask
behavior had been pinned by a test from day one. The new test in this
version exists specifically to prevent the same kind of doc/code
drift from happening again.

### v0.2.0 -- 2026-04-25  (bench instrumentation, image-shape split, telemetry tooling)

A coherent chunk of measurement-surface work: the LTX-shape harness gained
FlashInfer + SpargeAttention rows, the image-gen shapes split into their
own file, the torch.compile spike got a re-runnable script with a clean
verdict, and the one-shot runner `tests/run_all.sh` ties it all together.
Conventions tightened: consumer-agnostic framing rule, project-internal
phase numbers don't ship, path-privacy hooks installed.

#### Added

- `sageattention.get_last_dispatched_kernel() -> str | None` -- public
  helper that returns the kernel-name string of the most recent
  `sageattn*` call on the current thread, or `None` if no call has
  happened yet on this thread. Stable short names exposed as module
  constants (`KERNEL_FP16_TRITON`, `KERNEL_FP8_CUDA_PP`, etc.) and
  enumerated in `KNOWN_KERNEL_NAMES`. Backed by a `threading.local()`
  set at the top of each entry point with the resolved kernel name --
  zero API change for callers who don't read the helper. Lets a
  downstream tracer record what sage actually dispatched to (instead
  of mirroring the routing table from `core.py::sageattn` or treating
  the kernel as opaque), which is the missing input the
  "mask-kernel work justified?" gate in a consumer-side summary needs
  to fire correctly. Read the value immediately after the sage call
  -- if your code yields (asyncio, or another sage call from the same
  thread) between call and read, the value can be overwritten.
  Verified end-to-end on RTX 4090 / sm89 / CUDA 13.0 / torch 2.11 via
  `tests/test_dispatched_kernel_telemetry.py`.
- `tests/run_all.sh` -- one-shot validation runner. Resolves the venv from
  `$VENV` or `$VIRTUAL_ENV`, snapshots the env to
  `internal/bench_env_<today>.txt`, runs the LTX bench, the image bench,
  and the torch.compile spike in sequence; archives logs under
  `internal/log/`. `set -euo pipefail`.
- `tests/test_sageattn_image_shapes.py` -- companion to
  `test_sageattn_ltx_shapes.py`, holds head_dim ∈ {120, 128} shapes
  (Z-Image-Turbo S3-DiT, Flux-class). Reuses the LTX file's
  `run_shape_sweep()` helper; ~50 lines.
- `tests/test_sageattn_ltx_shapes.py` -- new bench rows on top of v0.1.0:
  * FlashInfer fp16 prefill row (optional; SKIPs cleanly when not
    installed). Predicted to lag sage fp8++ on sm89 because CUTLASS
    lacks native fp8 below sm90.
  * SpargeAttention top-k=0.5 row on unmasked self-attn shapes only
    (Sparge inherits sage's mask gap; SKIPs when `spas_sage_attn` not
    installed).
  * `run_shape_sweep(shapes)` extracted as the per-shape engine so
    `test_sageattn_image_shapes.py` reuses it without duplicating
    ~85 lines of scaffolding.
- `tests/spike_torch_compile.py` -- re-runnable spike measuring whether
  `torch.compile` around sage produces bounded mean-rel-error AND
  speedup. Verdict on torch 2.11: keep the consumer-side
  `torch.compiler.disable()`. Both compile modes drift ~2.8% vs eager,
  consistent across modes (autocast or op fusion around sage's int8/fp8
  dispatch). Reopen-trigger: "compile produces bounded rtol AND a
  measurable speedup" on a future torch release.
- `internal/bench_env_2026-04-25.txt` -- env snapshot (torch
  2.11.0+cu130, triton 3.6.0, sage editable, RTX 4090 / sm89, CUDA 13.0)
  locking the version surface so later phase deltas are real perf
  changes.

#### Measured

First-measurement datapoints on RTX 4090 / CUDA 13.0 / torch 2.11 /
bf16, captured during this version's work:

- **self-attn-large** (31776×31776, head_dim=64, no mask): sage fp8++
  19.95 ms, torch_flash 52.23 ms (2.62x), torch_cudnn 53.98 ms (2.72x).
  ~1.4% drift from the v0.1.0 baseline of 19.67 ms (cu128 -> cu130 +
  triton 3.6 upgrade); within run-to-run noise. Yardstick is now
  19.95 ms.
- **image_gen 4096×4096 h24 d128** (Flux-class): sage fp8++ 0.64 ms vs
  torch_flash 1.31 ms (2.05x). Closes the "do we need a per-model-class
  router branch?" question with a no.
- **z_image_turbo 4608×4608 h32 d120** (S3-DiT single-stream): sage
  fp8++ 1.32 ms vs torch_flash 2.23 ms (1.69x). Confirms sage's CUDA
  kernels handle the non-power-of-2 head_dim=120 cleanly.
- **cross-attn + mask** rtol fingerprints (CUDA-mask-bug signature)
  unchanged from v0.1.0.

#### Changed

- `README.md` and `CLAUDE.md` -- reframed to be consumer-agnostic. Sage
  is a general PyTorch attention library; the fork compiles cleanly
  into any consumer. README now lists what's in the fork beyond the
  upstream (bench harness, compile spike, warmup API, autotune
  addition); CLAUDE.md TLDR states the two purposes (editable install
  + experimentation/measurement surface). Conventions added: consumer-
  agnostic framing in committed material, project-internal phase
  numbers don't ship, path discipline enforced by the path-privacy
  plugin's pre-commit hook.

### v0.1.0 -- 2026-04-23  (post-squash baseline)

Initial fork divergence from `woct0rdho/SageAttention` after the
history squash. Everything below is what makes this fork different
from upstream as of the squash commit.

#### Added

- `setup.py` -- `_qattn_sm80` is now built when compute capability 8.9
  (Ada) is detected. Framed as a regression fix from
  `woct0rdho/SageAttention`: thu-ml's setup.py gates the SM80 extension
  on `HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or
  HAS_SM120 or HAS_SM121` (Ampere + Ada + Hopper + Blackwell), but
  woct0rdho's refactor collapsed that to a tuple gate `("8.0", "8.6",
  "8.7")` -- which silently drops Ada, Hopper, AND Blackwell.
  Ada-only source builds on woct0rdho's fork lose
  `sageattn_qk_int8_pv_fp16_cuda` (the fp16 fallback). We added `"8.9"`
  because that's the arch we test and care about; widen the tuple to
  match thu-ml's coverage if you run this fork on Hopper or Blackwell
  and want the fp16 fallback built from source.
- `sageattention/core.py::sageattn_warmup(shapes, kernels=...)` --
  public API that fires one-shot dispatches per (kernel, shape) to
  prime Triton's JIT + autotune cache. Cuts ~1s first-call latency on
  sm89 to ~2ms post-warm. Defaults to the Triton kernel only (CUDA
  kernels are build-time compiled, no warmup benefit).
- `sageattention/triton/attn_qk_int8_per_block.py` -- added
  `@triton.autotune` over `num_warps in {4, 8}` and
  `num_stages in {3, 4, 5}`, keyed on runtime shape. BLOCK_M/BLOCK_N
  stay hardcoded because they're locked by the per-block int8
  quantization step in `sageattention/quant.py`. Measurement on RTX
  4090 / LTX shapes: autotune confirmed the existing hardcoded config
  (`num_warps=4`, `num_stages=3` for `head_dim=64`) was already at the
  optimum -- zero perf delta then. Value is structural: auto-adapts to
  future kernel / triton / shape shifts.
- `build.sh` -- editable-install wrapper. Enforces `VIRTUAL_ENV`, pins
  `uv pip install --python ${VIRTUAL_ENV}/bin/python`, compiles for
  Ampere + Ada (`TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`) by default. Caps
  `MAX_JOBS` at 8 to keep high-core boxes from OOMing during
  `_qattn_sm89` compilation.
- `tests/test_sageattn_ltx_shapes.py` -- LTX-2.3-shape accuracy and
  speed harness. Measures every installed sage kernel and three torch
  SDPA backends against `SDPBackend.EFFICIENT_ATTENTION` at LTX's
  actual shapes (head_dim=64, heads=32, self-attn + cross-attn-with-
  mask across seq_kv from 32 to 1024, plus a synthetic wide-V shape).
  Reports mean/max rtol+atol and median elapsed. Soft-warns when
  mean_rtol > 0.10. Cross-kernel `fp8++vs.triton` consistency row on
  unmasked shapes: mean_rtol ~0.10 across self-attn shapes, equal to
  the combined-noise floor (triton ~0.04 + fp8++ ~0.09 vs SDPA, added
  in quadrature). No hidden discontinuity; mixing is safe.
  First-measurement datapoints (RTX 4090 / CUDA 13.2 / torch 2.11 /
  bf16): self-attn-large (31776×31776, no mask) sage fp8++ 19.67 ms vs
  torch_flash 52.39 ms (2.7x), torch_cudnn ~360 ms (cuDNN FA3 path not
  competitive on sm89); cross-attn + mask (kv=226) sage fp16_triton
  0.78 ms vs torch_cudnn 2.20 ms (2.8x). Sage remains load-bearing on
  sm89.
- `tests/repros/repro_cuda_mask_kernel.py` -- standalone repro for the
  CUDA mask-path missing-feature documented in Known kernel bugs.
- `CHANGELOG.md`, `CLAUDE.md` -- this file plus the fork navigation
  guide.

#### Changed

- `README.md` -- reduced to attribution only (immediate fork:
  `woct0rdho/SageAttention`; original: `thu-ml/SageAttention`) plus a
  short build pointer. Windows-specific installation prose and wheel
  selection guidance removed -- this fork builds from source.
