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

### CUDA kernels have partial attention-mask support (sm89 fp8++ landed v0.5.5; sm80 + other variants still missing)

The original gap inherited from `thu-ml/SageAttention`: Python wrappers
`sageattn_qk_int8_pv_fp16_cuda` and `sageattn_qk_int8_pv_fp8_cuda`
accept `attn_mask` via `**kwargs` but never pass it through to the
C++ layer. The C++ `MaskMode` enum originally had only `{kNone,
kCausal}`. Masks were silently dropped on all CUDA code paths.

**v0.5.5 (2026-05-13) closed this on the load-bearing sm89 path** --
the `MaskMode::kGeneral` variant + `apply_general_mask` helper landed
in the fp8++ kernel (`qk_int_sv_f8_cuda_sm89.cuh` +
`sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`). The
dispatcher routes masked calls on sm89 + CUDA >= 12.8 to the new
path. Measurement at LTX-class shape (bf16, T=2048, D=128) shows
mean_rtol=0.098 vs Triton reference (same accuracy profile as the
unmasked fp8++ path); the historic ~0.94 silent-drop signature is
gone.

**Still missing** (deferred per scope discipline):
- sm80 fp16_cuda path (`qk_int_sv_f16_cuda_sm80.cu`). Same kernel
  pattern; deferred until a workload hits sm80 + masks frequently.
- The other 6 sm89 variants (`accum_f32_*`, `accum_f16_attn_inst_buf`,
  the no-`inst_buf` versions). Not dispatcher-hit on sm89 + CUDA >= 12.8.
- The sage 3 Blackwell path (also lacks mask support upstream;
  removed in v0.5.0).
- Whole-block-skip optimization on sparse bool masks (Triton has it
  via the `tl.max(mask_block) == 0 -> skip` short-circuit; the CUDA
  kernel's pipelined K-iteration loop makes the analog non-trivial
  to mirror without serializing the pipeline).

Until those land, hand-picking a `_cuda` kernel with a non-None mask
on those paths still warns + drops (the v0.3.1 soft-warn safety net).
The dispatcher's safe-default routing handles this for consumers
that call `sageattn()` without overrides.

Repro: `tests/repros/repro_cuda_mask_kernel.py` (predates the v0.5.5
fix; documents the original symptom on sm89 fp8++; now passes
through the kGeneral path).

Discovered: 2026-04-23 via `tests/test_sageattn_ltx_shapes.py` (the
seq_kv sweep exposed the rtol-vs-seq_kv scaling signature). Closed
on sm89 fp8++: 2026-05-13 (v0.5.5).

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

### Extend CUDA mask support beyond sm89 fp8++ (sm80, other sm89 variants)

The sm89 fp8++ kernel landed mask support in v0.5.5 (2026-05-13). The
remaining surfaces:

1. **sm80 fp16_cuda** (`qk_int_sv_f16_cuda_sm80.cu`). Same kernel
   pattern as sm89; would mirror the `MaskMode::kGeneral` +
   `apply_general_mask` work. Deferred -- sm89 is the load-bearing
   arch for this fork and the dispatcher already routes correctly
   on sm89.
2. **Other 6 sm89 variants** (`accum_f32_*`, `accum_f16_attn_inst_buf`,
   the no-`inst_buf` versions). Same kernel template; the call sites
   already pass nullptr/0 for the mask params after v0.5.5 (the
   `if constexpr` branches dissolve in the kCausal/kNone
   specializations). Adding the kGeneral path here would mirror what
   the fp8++ variant does; deferred until a dispatcher branch hits
   one of them with a mask.
3. **Whole-block-skip optimization on sparse bool masks**. Triton has
   the `tl.max(mask_block) == 0 -> skip=True` early-exit on
   all-False BLOCK_N tiles. The sm89 fp8++ CUDA kernel's pipelined
   K-iteration loop (with `cp_async::commit_group` / `wait_group`)
   makes the analog non-trivial -- a runtime skip would either
   serialize the pipeline or require restructuring the loop. Real
   perf win only on sparse masks (LTX-2.3 guide masks are dense);
   defer until a consumer hits a sparse-mask workload where it
   matters.

**Trigger to act on (1) or (2):** the dispatcher routes a masked call
to a surface that hits one of these kernels in production. Today
sm89 + CUDA >= 12.8 routes to the v0.5.5 path; other archs route to
Triton. If a downstream consumer reports masked calls on sm80 (or
forces a non-fp8++ pv_accum_dtype on sm89 with a mask), revisit.

**Trigger to act on (3):** a consumer workflow with sparse (>50%
all-False BLOCK_N tiles) masks reports Triton masked-path wall-time
> 5% of total gen.

### `torch.library.custom_op` registration for fused-pybind kernels

Three pybind kernels in `csrc/fused/pybind.cpp:30-32` cause Dynamo
graph breaks under `torch.compile`, with deterministic precision loss
(0.0276 rtol drift, > 0.01 budget) from partial-graph reordering across
the pybind boundary. Empirically verified on torch 2.11.0+cu130 via
`tests/spike_torch_compile.py` (2026-05-01). The three kernels:

- `_fused.transpose_pad_permute_cuda`
- `_fused.scale_fuse_quant_cuda`
- `_fused.mean_scale_fuse_quant_cuda` (smooth_v=True branch, same
  risk class)

All called from `sageattention/quant.py:281, 289, 292` in the
`per_channel_fp8` V-quant path -- load-bearing on every fp8 sage call
on sm89. Registering each as `torch.library.custom_op` with proper
meta/abstract registrations would let Dynamo trace through them
without graph breaks.

Size estimate: ~1-2 days per kernel (~3-6 days total) including
correctness verification under compile.

**Trigger to act:** consumer's path 1 (CUDA graphs on the LTX
denoiser) fails AND consumer wants path 2 (`torch.compile` the
denoiser) -- at which point the spike's "keep the disable" verdict
needs to flip and this work becomes the gating dependency. Until
then, keep the disable. Re-run the spike after every torch upgrade.

### `arm_kj` synthetic head-to-head VRAM bench

Audio-loop's empirical evidence (2026-05-01 N1-N4 memo) and
sage-side bench data converged on: the 3.5x VRAM gap of sage
`fp8_cuda++` (~628 MiB) vs `torch_flash` (~182 MiB) at the
load-bearing LTX video shape is structural to the int8/fp8 quant
approach, not specific to sage's dispatcher wrapper -- KJ's
per-block path also pre-materializes the same int8/fp8
intermediates before the kernel call.

To falsify conclusively, add a row to
`tests/test_sageattn_ltx_shapes.py` that calls KJNodes'
`ltx2_sageattn_forward` (the per-block sage path in their LTX-2
node module) directly with a synthetic input at the same shape,
measuring working-set VRAM via the same `time_and_vram` helper.
Workflow-arm swap is NOT viable -- DAG trace (2026-05-01)
confirmed `LTX2MemoryEfficientSageAttentionPatch` is NOT in the
consumer's iclora workflow, so the test would have to install +
wire KJNodes synthetically, with skip-if-unavailable handling for
the import.

Size estimate: ~half a day (test row + KJNodes import guard +
skip helper).

**Trigger to act:** the question becomes load-bearing for some other
decision (e.g. "should we restructure sage's Python wrappers to
avoid intermediate materialization?" -- which today the consumer's
evidence already resolves as "no, gap is structural to int8/fp8
quant"). Today: not load-bearing.

### Block-along-T optimization on `fused_rope_split` Triton kernel

`/simplify` efficiency review (2026-05-01) flagged that
`_rope_qk_split_kernel` launches one program per `(t, h, b)` --
733k programs at the LTX video shape (B=1, H=32, T=22932). Each
program does ~1024 bytes total I/O on D//2=64 elements, below the
bf16 cache-line sweet spot. Block along T with `BLOCK_T=8` or `16`
(one program per `(b, h, t_block)`, inner loop over `t`) cuts the
grid 8-16x and amortizes program-launch overhead.

Size estimate: ~half a day (kernel restructure + perf
measurement against a new `tests/bench_fused_rope.py` micro-bench).

**Trigger to act:** a future workflow brings `fused_rope_split`
above 5% of GPU time. Today: 0.55% on the consumer's iclora
workflow (their 21:02Z memo) -- not worth the perf-measurement
work.

### `fused_rope_split` removal candidate

v0.5.3 shipped the primitive on the strength of a comparison-doc
finding ("only structural kernel-side gap vs KJ's per-block
patch") that turned out to overstate the value -- consumer
measured RoPE at 0.55% of GPU time, retracted the ask. Kernel
earns its space as a sage-fork primitive (low maintenance, ~280
LOC self-contained, available for future DiT consumers), but the
immediate ROI is zero. Same disposition as `sageattn_warmup`:
candidate for removal if no consumer adopts within ~6 months.

**Trigger to act:** by 2026-11-01, audit `coderef/` for any
consumer importing `sageattention.fused_rope_split`. If none, drop
the kernel + tests + CLAUDE.md inventory entry in a focused
deletion arc. Lesson: see `feedback_walltime_before_kernel_day`
memory entry -- ask for wall-time contribution before kernel-day
spend on a "kernel-side gap" finding.

## Decision log

Investigations that closed without action. Recorded so we don't
re-derive them. Each entry has an explicit reopen-trigger.

### "FFN-adjacent reach" / launch-overhead / cache-footprint hypotheses on iclora: all three falsified

**Investigated 2026-05-07** (cross-claude bounded investigation
with AudioLoopHelper claude; memo trail in
`internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md` +
`internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md`).

The v0.5.1 entry's claim that sage's e2e speedup "extends beyond
per-call attention rows into FFN-adjacent amortization within the
sampler step" was a single-data-point inference from the
audio_loop_latent workload (arm-2 attention time was never directly
traced; +17pt above strict Amdahl was attributed to FFN-adjacent
mechanism without measurement).

A clean sage-on/sage-off A/B on the iclora workflow (consumer's
production-scale workload, profiler on, matched 3456 attention calls
per arm) decomposes the savings cleanly:

| Mechanism | Savings | Status |
|---|---|---|
| Attention kernel time (sage 7.20ms vs torch flash 22.14ms) | -51.6s | confirmed (dominant) |
| Non-attention named CUDA kernels | +2.1s | sage-on slightly slower |
| cudaLaunchKernel call-count delta (0.82%) | -0.4s | negligible |
| Wall-clock vs CUDA-time gap (CPU-side) | -3.9s | not GPU-side |

Strict Amdahl with iclora's actual attention share (~42% of CUDA
time) and the actual per-kernel ratio (3.08x) predicts 1.39x e2e
speedup; measured wall ratio is 1.41x -- match within 1.4%. **No
non-Amdahl mechanism is required on iclora.**

Three hypotheses closed:
1. **Launch-overhead reduction** ("sage replaces torch SDPA's
   decomposed-op path with one fused kernel, saving ~6-10 launches
   per attention call"): falsified. Sage-off is already routed
   through `aten::_scaled_dot_product_flash_attention` -- one fused
   launch per call. Sage replaces flash with sage; one-fused-for-
   one-fused. Total launch delta is 2205 of 270k (0.82%).
2. **FFN-adjacent reach via int8 amortization**: falsified on
   iclora. Non-attention CUDA kernel time is essentially identical
   sage-on vs sage-off (delta -2.1s, in the wrong direction).
3. **Cache-footprint helping adjacent matmul/elementwise**:
   falsified on iclora. If int8 K + fp8 V freed L2/HBM bandwidth
   for adjacent ops, the named matmul kernels would be faster
   sage-on; instead they're 0.13-0.73s slower per kernel (sage's
   own quant work pollutes their cache lines).

**Decision:** retire all three from sage-fork's mental model on
iclora. The original v0.5.1 +17pt residual on audio_loop_latent
remains unexplained but is single-data-point and measurement-
methodology-dependent (no arm-2 attention tracer); not load-
bearing for any current decision.

**Trigger to reopen:**
- A future workload measures non-trivial sage-on-vs-off non-
  attention kernel-time delta (>5s on a render of similar scale to
  iclora's 180s sage-off baseline). Direct evidence of any of the
  three mechanisms operating.
- The audio_loop_latent +17pt gets re-measured with arm-2
  attention tracing in place, confirming it's real and not an
  inference artifact. If real and reproducible, the mechanism
  question reopens for that specific workload.

**Process note:** the pre-committed-prior-and-decision-rule
discipline (recorded in CLAUDE.md / "Pre-trigger briefing
pattern") fired correctly twice in this exchange. Both times an
investigation produced a "your hypothesis is wrong" outcome
without consuming downstream code-change budget. Confirmed as
default for future cross-claude bounded investigations.

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

**Open observation (2026-05-07):** synthetic LTX bench
(`tests/test_sageattn_ltx_shapes.py`) reports `torch_flash / sage_fp8++`
= **2.66x** at the primary shape; the iclora production A/B (cross-
claude memo trail, 2026-05-07) measured the same kernel pair at
**3.08x** averaged over 3456 cache-warm calls. The 16% gap is unaddressed
but probably not a real kernel-side change. Most likely candidates,
roughly ranked: (1) median-of-3 (synthetic) vs sum/calls average over
3456 calls (production) is a different statistical animal — long-tail
distributions on warm production state can shift the central estimate
10-15% even on identical shapes; (2) cache state (synthetic short-burst
vs production sustained); (3) driver-thermal asymmetry across kernel
types under sustained load; (4) torch/triton/CUDA drift between when the
synthetic bench was last snapshotted and when iclora ran. Action: log
this gap on the next bench env re-snapshot. If the synthetic ratio shifts
toward 3.08x without a kernel-side change, hypothesis (1) is the answer
and the synthetic bench's median-of-3 protocol may be worth widening.

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

### v0.6.0 -- 2026-05-15  (sage_ffn: fp8-native two-kernel fused MLP for LTX 2.3-class FFN blocks on sm89)

Ships `sage_ffn(x, w1, s1, w2, s2)` -- a Triton two-kernel
`Linear(fp8) -> GELU(tanh) -> Linear(fp8)` MLP path for DiT
FFN blocks whose weights are stored as per-tensor fp8 (E4M3FN).
The primary motivating workload is LTX 2.3 distilled, whose
transformer blocks have hidden=4096, inner=16384 and 44 of 48
blocks shipped as fp8 (bookend blocks `{0, 1, 46, 47}` stay
bf16 per the distilled checkpoint's design).

**The wedge is qualitative, not just quantitative.** No other
library ships an fp8-native fused MLP for ComfyUI consumer-app
shapes on sm89. FA's `fused_mlp_func` is bf16/fp16 only (cuBLASLt
epilogue path); torch's `F.linear` against fp8 weights dequants
to bf16 first, paying 2x the weight-bandwidth and using bf16
tensor cores at ~330 TFLOPS instead of fp8 tensor cores at
~660 TFLOPS. `sage_ffn` loads fp8 weights directly and computes
in fp8.

**Synthetic-bench numbers (RTX 4090, CUDA 13.0, torch 2.12.0+cu130,
triton 3.7.0):**

| shape | mean_rtol vs torch ref | sage_ffn | torch ref | speedup |
|---|---|---|---|---|
| stage-1 (T=10780, h=4096, inner=16384) | 0.0915 | 13.7 ms | 18.2 ms | **1.33x** |
| stage-2 (T=44880, multi-guide expanded) | 0.0914 | 59.7 ms | 75.7 ms | **1.27x** |

These are standalone matmul-GELU-matmul measurements against
randomly-initialized weights, not end-to-end ComfyUI rendering.
mean_rtol is well under the 0.10 budget that gates all sage
kernels. The reference is `F.linear(F.gelu(F.linear(x, w1_bf16_ref),
approximate="tanh"), w2_bf16_ref)` with weights dequantized once
outside the timing loop -- i.e. torch's best-case fp8-weight path,
not its naive one.

Validated against the full 126-config sweep at the same env:
hardcoded 8-config winners deliver 1.33x / 1.27x; full sweep
delivers 1.33x / 1.27x. Bit-identical numerics, hardcoded matches
full-sweep perf within run-to-run noise.

**Real-world e2e wall-time impact: not yet measured.** Several
production factors can shift the ratio either way: L2 cache
contention from neighboring sub-modules in the DiT block, GPU
thermal/clock state during sustained renders, ComfyUI's
model-patching machinery overhead. The v0.5.5 precedent showed
that synthetic kernel-bench numbers project but don't substitute
for in-pipeline measurement. The number that fills this gap will
come from a downstream consumer's A/B against a chunking-only
baseline.

**Design: two-kernel split, not single-kernel fusion.** Kernel 1
matmul + GELU(tanh) epilogue + write intermediate. Kernel 2
matmul down-projection. Intermediate at LTX stage-2 multi-guide
is ~1.47 GiB; users on 24 GiB cards should compose with an
FFN-chunking node (e.g. `LTXVChunkFeedForward` from KJNodes).
The single-kernel "intermediate never hits HBM" design was
explored on day 1-2 and rejected at day 2's perf wall: the
X_tile (1 MB at BLOCK_M=128, K=4096, bf16) won't fit in sm89's
100 KB SMEM, and the nested-loop structure was forcing Triton
into L2 evictions. FA's `fused_mlp_func` is also a two-kernel
split for the same reason -- this design converges on the
industry standard.

**Per-block-K activation quantization.** Each (BLOCK_M, BLOCK_K)
chunk of the bf16 activation gets its own f32 scale, computed
inline during the K-reduction. This avoids a separate amax pass
over the full K dimension; the slight coarsening (~0.005 rtol
cost vs per-row) is well within the 0.10 budget.

**Autotune.** Each kernel carries 8 hardcoded `@triton.autotune`
configs -- the winners from a 126-config sweep against the two
LTX FFN shapes, plus a few neighbors for shapes the winners may
not cover. First call at a new shape pays ~10-15 seconds
autotune-search per kernel (~30-60 sec total across the full
sage_ffn at two new shapes); subsequent calls hit Triton's
on-disk cache. The full 126-config sweep cost 7+ minutes
first-render-per-shape on consumer hardware -- unshippable UX.
The pruned 8-config set preserves the 1.27-1.33x delivered
numbers at acceptable cold-start cost (validated against the
full sweep at the same env). To re-derive winners for a new
LTX-class shape: run the kernel against the shape, inspect
`_fp8_matmul_gelu_kernel.cache` for the picked config.

**E2e wall-time projection (NOT measured).** At an FFN-time-share
of 24-27% for LTX 2.3 multi-guide workloads, the synthetic
1.27-1.33x FFN speedup would project to roughly 4-7% e2e
reduction *if* synthetic numbers hold in production. They may
not -- see "Real-world e2e wall-time impact" above. The
qualitative wedge is what's load-bearing for consumer framing;
absolute e2e numbers will land when a downstream A/B has run.
The synthetic fp8-native ceiling is closer to 1.5-2x; the gap
to delivered 1.27-1.33x is explained by Triton's matmul codegen
vs cuBLASLt's hand-tuned kernels.

**Composes with chunking, doesn't replace it.** Users with a
chunking node already in place (the common case on 24 GiB
cards) keep the chunking and gain the FFN wall-time win on each
chunk's matmul. Users with 32+ GiB headroom can drop chunking
and run `sage_ffn` against the full multi-guide tensor at the
cost of the 1.47 GiB intermediate.

**Limitations / scope.**

- Plain GELU MLP only. No gated SwiGLU/GEGLU variant in v0.6;
  the FFN structure has to be `Linear -> GELU(tanh) -> Linear`.
- Bookend bf16 blocks must be handled by the caller. Consumer-side
  dispatch typically inspects `block.ff.net[0].proj.weight.dtype`
  and falls through to `F.linear` for bf16 blocks.
- Per-tensor scalar weight scale only. Per-row / per-channel
  weight scales are a v0.6.1 extension if a workload demands.
- Not wired into `sageattn()` -- this is a separate FFN primitive,
  not an attention kernel. Consumer imports `sage_ffn` directly.

**Files added / changed:**

- `sageattention/triton/fused_mlp_fp8.py` (new) -- the two-kernel
  implementation + `sage_ffn` Python wrapper.
- `sageattention/__init__.py` -- `sage_ffn` export.
- `tests/spikes/spike_fp8_mma.py` (new) -- day-1 spike verifying
  `tl.dot(fp8, fp8) -> f32` on sm89 at small + LTX stage-1
  shapes.
- `tests/spikes/test_fused_mlp_fp8_correctness.py` (new) --
  correctness + perf gate at LTX FFN shapes, median-of-5 timing
  after autotune-absorbing warmup.

Design narrative (cross-claude memo trail, v0.6 scoping doc,
day-by-day execution journal, decision-gate framework) lives in
`internal/design/ffn_fusion_scoping.md` (gitignored).

### v0.5.5 -- 2026-05-13  (native general-mask support in the sm89 fp8++ CUDA kernel)

First downstream-driven kernel-day work on the fork. Lands the
load-bearing piece of the long-standing CUDA mask gap (in "Known
kernel bugs" since 2026-04-23): the sm89 fp8++ kernel now applies an
additive attn_mask to QK scores before the softmax max reduction,
matching the Triton reference's behavior. Dispatcher routes masked
calls on sm89 + CUDA >= 12.8 to the new path.

Triggered by a high-leverage downstream consumer surface raising
the structural-correctness concern (the "1 high-leverage surface"
clause added in v0.5.4 backlog reformulation). Scoping note +
implementation discipline in
`internal/design/cuda_mask_kernel_scoping.md` (gitignored).

#### Added

- **`MaskMode::kGeneral`** value in `csrc/qattn/attn_utils.cuh` +
  **`apply_general_mask<num_tiles_q, num_tiles_k, DTypeMask,
  DTypeQKAccum>`** helper. Mirrors `apply_causal_mask`'s index math;
  adds in-bounds-guarded mask load + additive apply per (q_idx,
  kv_idx) for each thread's 8-entry register fragment. Supports
  bool masks (translated to additive {-inf, 0} log-weights upstream
  in Python) and dtype-matching float masks (half or nv_bfloat16),
  mirroring the two Triton mask paths.

- **`DTypeMask` template parameter + mask runtime params** on
  `qk_int_sv_f8_attn_kernel` (`qk_int_sv_f8_cuda_sm89.cuh`). Defaults
  to `nv_bfloat16` / nullptr / 0 strides so existing instantiations
  compile unchanged. `if constexpr (mask_mode == MaskMode::kGeneral)`
  branches at the two existing mask-application points (the
  steady-state K-iteration block + the last-iter block) call the
  helper; both branches dissolve in the kCausal / kNone
  specializations.

- **Optional `attn_mask` parameter on `qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`**
  (C++ entry + pybind + `sm89_compile.py` custom_op schema +
  register_fake stub). Backward-compatible: existing positional
  callers don't need updates. The dispatcher in the .cu branches on
  `attn_mask.has_value() && attn_mask->numel() > 0`: kGeneral
  specialization gets launched when the branch is taken, kCausal /
  kNone otherwise.

- **Python wiring**: `sageattn_qk_int8_pv_fp8_cuda` extracts
  `attn_mask` from kwargs and passes it to the fp32+fp16 (fp8++)
  variant. bool->additive {-inf, 0} translation happens here, plus
  `attn_mask.expand((B, H, qo_len, kv_len))` for broadcast support
  (mirrors `core.py:441` in the Triton path). Other `pv_accum_dtype`
  variants still warn + drop the mask (their kernels haven't gained
  the kGeneral path).

#### Changed

- **Dispatcher routing**: `sageattn()` masked-call routing now
  arch-aware. On sm89 + CUDA >= 12.8, masked calls land on
  `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`
  (the new CUDA mask path). On other archs (sm80, sm86, sm87, sm75,
  sm100/sm120/sm121 fallback), masked calls still route to
  `sageattn_qk_int8_pv_fp16_triton` since their CUDA kernels haven't
  gained mask support yet.

- **Test invariant update**: `tests/test_dispatched_kernel_telemetry.py`
  renamed `test_sageattn_dispatcher_routes_masked_calls_to_triton`
  -> `*_correctly`. The v0.3.0 "all masked calls -> triton"
  invariant is superseded; v0.5.5's invariant is arch-aware.

#### Measured

At LTX-class shape (B=1, H=4, T=2048, D=128, bf16, RTX 4090):

| measurement | value |
|---|---|
| CUDA mask vs Triton reference (bool mask) | mean_rtol=0.098, max_rtol=2.0, mean_atol=0.0011 |
| bool->additive translation equivalence | max_atol=0.000000 (bit-identical) |
| mask actually applied (vs unmasked output) | max_atol=0.055 (changes output meaningfully) |
| zero-mask sanity (vs unmasked) | max_atol=0.000000 (additive zero is true no-op) |
| LTX bench fp8_cuda++ unmasked (`ltx23_video_self_attn_init_22932`) | 19.98 ms (baseline 20.20; no regression) |
| LTX bench fp8_cuda++ masked cross-attn rows | mean_rtol ~0.09 (was ~0.94 pre-v0.5.5 silent-drop) |

The zero-mask bit-identity test is the strongest correctness signal
for the `if constexpr` discipline: the kGeneral branch infrastructure
adds no perturbation to the kNone specialization when mask values
are no-ops. The 0.055 masked-vs-unmasked atol confirms the mask is
actually applied (not silently dropped, which would produce 0 atol
like pre-v0.5.5).

#### Measured (in-pipeline, preliminary; framing softened 2026-05-15)

Beyond the synthetic correctness + perf measurements above, we ran
A/B comparisons on a real LTX 2.3 workflow (multi-guide at
768×512×97) on a 4090 with dynamic VRAM disabled
(`--disable-dynamic-vram --disable-async-offload --reserve-vram 0
--cuda-malloc --supports-fp8-compute --mmap-torch-files
--cache-none`). Updated count after additional repetitions:

| arm | outcome |
|---|---|
| `auto` (v0.5.5 fp8_cuda++ masked) + FFN chunking ON | N=3+ success, 0 OOM |
| `auto_mask_aware` (Triton masked) + FFN chunking ON | N=1 success, N=2 OOM (non-deterministic) |
| `auto` (fp8_cuda++) + FFN chunking OFF | deterministic OOM at stage-2 FFN GELU |

Both Triton OOMs hit the same site
(`comfy/ldm/lightricks/model.py` `AdaLNSingle.linear`,
downstream of attention; 727 MiB requested, ~16 MiB free,
~22.4 GiB allocated of 23.52 GiB), at exactly 48 masked dispatches.
The chunking-off fp8++ OOM hits the FFN GELU projection at
multi-guide T=44880 (proj output `(1, 44880, 16384)` bf16 ≈ 1.47 GiB).

**The corrected reading (after the chunk-bypass A/B)**: at LTX 2.3
multi-guide scale on 24 GiB, FFN chunking (`LTXVChunkFeedForward`)
is doing the heavy lifting on peak memory. With chunking on, the
attention working-set delta between Triton and fp8++ is the knob
that distinguishes "fits comfortably" (fp8++) from "fits 1/3 of the
time" (Triton non-determ OOM at AdaLN). Without chunking, both
kernels hit a different (FFN-intermediate) memory wall.

So the honest claim becomes: **with FFN chunking enabled, sage's
v0.5.5 CUDA mask path has more headroom for the attention-side
delta than the Triton fallback** (N=3+ vs N=1 success in the
observed sample). The earlier framing -- "fits where Triton
doesn't" -- was incomplete; sage choice is the second-order win
once chunking handles the first-order FFN memory cost.

Per-shape masked p50 latency (successful arm, single-run sample):
`(1, 10780, 4096)` masked Triton 195 μs vs fp8++ unmasked 158 μs;
`(1, 44880, 4096)` masked Triton 482 μs vs fp8++ unmasked 249 μs.
Run-to-run variance at the larger shape is real and not yet
attributed (autotune-cache warmth, per-call mask sparsity, thermal
state are all candidates); don't read p50 as precision.

**Status**: preliminary. Sample sizes are small; the Triton OOM is
non-deterministic; further repetitions and a smaller-mask variant
test are in progress. The synthetic measurements above remain the
primary v0.5.5 validation. The in-pipeline observation is
corroboration that's reproducible by anyone with the same workflow,
the routing flag flip, and the nodynvram config -- independent
reproduction welcome.

What's deferred and on what triggers: see Backlog / "Extend CUDA mask
support beyond sm89 fp8++".

### v0.5.4 -- 2026-05-13  (sageattn_partitioned + multi-slice peak-HBM bench + honest negative result on the masked-call scenario)

Driven by a consumer-side report that the masked Triton path
OOM'd on a 4090 in a workflow that partitions Q into noisy + tracked
slices and fires two back-to-back `sageattn_qk_int8_pv_fp16_triton`
calls per layer with the same K, V. Hypothesis: each call
re-quantizing K and re-casting V is the removable peak-HBM lever.
Built the entry, measured against the hypothesis, found the
masked-call scenario already efficient enough that the entry doesn't
help in synthetic isolation. Documented for future consumers and
real-pipeline validation.

#### Added

- **`sageattention.sageattn_partitioned(q, k, v, slices, ...)`** --
  public entry that runs Triton attention over multiple Q slices
  sharing K and V. Quantizes K once, casts V to fp16 once,
  allocates the output once; Q is re-quantized per slice (Q
  changes per slice). Each slice is `(q_start, q_end, attn_mask
  | None)`. Inner kernel writes each slice's output into a view
  of the pre-allocated full output via a new optional `out=`
  parameter on
  `sageattention.triton.attn_qk_int8_per_block.forward`. Records
  dispatch as `fp16_triton` (same underlying kernel). Test:
  `tests/test_partitioned.py` (4 cases: 2-call aligned/unaligned
  boundary, noisy-only, tracked-only; reuses `accuracy_metrics`
  from `test_sageattn_ltx_shapes.py` so tolerance budgets match
  every other rtol-vs-Triton row in the repo).

- **`sageattention.triton.quant_per_block.per_block_int8_q`** and
  **`per_block_int8_k`** -- factored from the existing
  `per_block_int8` Q+K quant. `per_block_int8` now wraps both
  helpers, signature unchanged for existing callers.
  `sageattn_partitioned` uses them to quantize K once across all
  slices while Q re-quantizes per slice.

- **`tests/bench/partitioned_mask_phase0/`** -- peak-HBM
  characterization at the LTX 2.3 self-attn shape (T=23296, h=32,
  d=128, bf16) for the two-call partition pattern (noisy + tracked
  slices sharing K, V). Six measurement rows: single-call no-mask
  reference, single-call with broadcast `(1,1,1,T)` mask,
  2-independent-call cumulative no-mask, 2-independent-call
  cumulative with-mask, `sageattn_partitioned` no-mask,
  `sageattn_partitioned` with-mask. Reports savings vs the
  2-independent-call baseline. Checked-in `results.json` + memory
  snapshot for the audit trail; uncorrected wrong-mask-shape
  variants preserved as `*.uncorrected.{json,bin}` (caught by a
  sister-clone audit of the actual mask shapes the consumer
  workflow uses).

#### Measured: synthetic Phase 0 + Phase 3 result

At the LTX 2.3 self-attn shape on RTX 4090, the 2-call
partition's cumulative peak HBM:

| call pattern | no mask | with two-call partition masks |
|---|---|---|
| single full-T call (reference) | 1096 MiB | 1096 MiB |
| 2 independent sage calls (cumulative) | 1807 MiB | 1272 MiB |
| `sageattn_partitioned` | 1528 MiB | 1298 MiB |
| **savings** | **+279 MiB** | **-26 MiB** |

The partitioned entry saves +279 MiB in the no-mask isolation
(as predicted: K-quant + V-cast amortization is real). But in
the masked-call scenario -- the originating use case -- the
2-independent-call cumulative peak is already only +176 MiB
above the single-call reference, and the partitioned entry
doesn't reduce it further. Cross-clone hypothesis (filed by the
sister clone as "hypothesis 2" before Phase 3): allocating the
22 MiB mask first in each call may bias the pytorch caching
allocator's bucket layout to consolidate K_int8 / V_fp16
allocations better than the partitioned entry's
K-first / V-first / output-first pattern. Not investigated;
matches the observed asymmetry.

**Net for the originating use case**: the partitioned entry
doesn't address the consumer-reported OOM in synthetic measurement.
Real-pipeline validation (sister-clone side with the LTX denoiser
loaded, against fragmented post-model-load allocator state) may
show a different picture; sage-fork side considers that an open
question. The entry stays shipped because (a) it's correct, (b) the
no-mask savings are real for any future consumer that hits a similar
pattern without masks, and (c) it provides the primitive to test
against in real-pipeline scenarios.



Three coupled additions, all driven by a consumer-side comparison
doc that surfaced (a) one structural kernel-side gap vs KJNodes'
`LTX2MemoryEfficientSageAttentionPatch` (his fused-RoPE Triton
kernel), (b) an unverifiable "memory efficient" framing, and (c) a
fork API stability concern after v0.5.0 dropped `_qattn_sm90`.

#### Added

- **`sageattention.fused_rope_split(q, k, freqs_cis, *, use_triton=True)`**
  -- public fused split-RoPE primitive. Matches LTX's
  `apply_split_rotary_emb` (comfy/ldm/lightricks/model.py:343)
  exactly via a clean-room Triton kernel; falls back to a torch
  reference when preconditions fail (non-cuda, non-split-pe,
  shape mismatch, `use_triton=False`). Lives in
  `sageattention/triton/fused_rope.py`. Lets consumers drop their
  own per-block fused-rope kernel (e.g. KJNodes'
  `fused_rope_qk`) without sage going DiT-aware -- the API is a
  standalone helper, not bolted into `sageattn()`. v1 covers the
  split-RoPE convention only (LTX 2.3 video + audio); interleaved
  variants and other model classes silently fall back.
  Test: `tests/test_fused_rope.py` (3 CPU tests + 7 GPU tests
  covering rtol vs reference, dtype coverage, in-place
  semantics, fallback-path equivalence, dtype guards, public
  export).

- **`tests/test_sageattn_ltx_shapes.py` peak-VRAM column.** Folded
  into `time_and_vram(fn) -> (median_ms, peak_vram_mib)` (renamed
  from `time_median_ms`). Reports per-(shape, kernel) working-set
  VRAM at zero extra kernel cost -- the warmup pass that already
  absorbs autotune now also seeds the VRAM baseline, then
  `reset_peak_memory_stats()` rebases before the 3 timed runs.
  Initial finding from the new column: at the load-bearing LTX
  video self-attn shape (D=128, seq=22932), sage `fp8_cuda++`
  uses ~628 MiB working-set vs `torch_flash` ~182 MiB --
  empirically refutes "sage = memory efficient" framing for the
  high-level dispatch path on these shapes (sage materializes
  q_int8/k_int8/v_fp8/scale intermediates that flash keeps fused
  in registers/SMEM). Print-only, no regression gate.

- **CLAUDE.md "Downstream-known internal symbols" section.**
  Documents the de-facto-public underscore surface in
  `sageattention.core` that downstream consumers (KJNodes' LTX-2
  patch as canonical example) import by name. Lists the
  protected symbols, the protected pybind methods on
  `_qattn_sm89`, and a pre-removal checklist (grep coderef, memo
  before removal, major-bump on break). Triggered by the v0.5.0
  `_qattn_sm90` removal, which broke a downstream import case
  without prior consideration.

### v0.5.2 -- 2026-04-27 PM  (bench reliability: auto-warmup correctness + honest cold-start interpretation)

Two real bugs surfaced in the same hour AFTER v0.5.1 shipped, both
caught by running the bench end-to-end against ComfyUI restarts:

1. **`--warmup auto` false-positive on ComfyUI restart.** The
   filesystem-mtime heuristic from v0.5.1 (`aae9b9e`) detected
   "recent sage trace exists" and skipped warmup. But the trace
   file persists across ComfyUI restarts; mtime stays recent even
   though the in-process state (model load, JIT, per-node cache)
   was reset. Auto-mode correctly identified "sage was active 30
   min ago" and incorrectly concluded "caches warm now."
2. **Bench `Interpretation` mislabeled cold-start asymmetry as
   "sage SLOWER end-to-end."** Two real benches today landed at
   0.508x and 0.900x raw -- both cold-start-confounded. The bench
   printed "Sage SLOWER ... Check for instrumentation overhead,
   fallback paths, or a real regression." A future operator
   without per-node `exec.jsonl` analysis would walk away thinking
   sage broke. Sage's actual contribution was 1.22x e2e
   (audio-loop-helper claude's per-node analysis salvaged the
   reading; the bench output didn't).

#### Fixed

- **`tests/bench_e2e_ltx.py::_caches_appear_warm(host)`** (commit
  `a461ddb`). Auto-mode now requires BOTH (a) a recent non-empty
  sage.jsonl on disk AND (b) ComfyUI's `/history` HTTP endpoint
  to return a non-empty result. `/history` is in-memory; restart
  empties it. Combined signal correctly catches the
  restart-after-trace case the v0.5.1 heuristic missed.
- **`tests/bench_e2e_ltx.py` Interpretation block** (commit
  `1a06586`). Added `cold_start_suspected` branch BEFORE the
  generic SLOWER message. Triggers when `speedup < 0.95x` AND
  `attn_pct_on < 20%` -- structural signal that the slowdown is
  in non-attention work where sage doesn't run, almost certainly
  cold-start asymmetry between arms. Prints a diagnostic message
  with concrete next-steps (`--warmup always` from fresh ComfyUI,
  or aggregate exec.jsonl per-node) instead of the misleading
  "real regression" hint.

#### Refactored (simplify-pass on the fixes above)

- **Reuse existing `_http_get` helper** for the `/history` probe
  instead of duplicating `urllib.request.urlopen`. The helper
  was already in the file at line 329.
- **`/history/1`** instead of `/history`. ComfyUI's `/history` is
  unbounded across a session; the `/{max_items}` form caps the
  response server-side. Same empty-vs-not semantics, bounded
  payload.
- **`ATTN_PCT_LOW_FRACTION = 20.0` constant** alongside the
  speedup tier constants. Used twice in `report()`; bump-one-
  forget-the-other risk neutralized.
- **`_attn_pct_of_wall(results_on, on_med)` helper** extracted.
  Removes the declare-default-then-conditionally-assign pattern
  in the report block.
- **Drop `urllib.error.URLError`** from the `_http_get` exception
  tuple (subclass of `OSError`; redundant).
- **Trim `_caches_appear_warm` docstring** to a one-line summary
  pointing at `main()`'s warmup-policy comment for the
  operational rationale.

#### Verified

- `_comfyui_session_has_history` correctly returns True when
  ComfyUI has processed >=1 prompt this session, False on fresh
  restart (empty `/history`).
- `cold_start_suspected` branch fires correctly when replaying
  yesterday's actual numbers (0.508x raw + 4.5% attn pct);
  produces the diagnostic message instead of the old "SLOWER ...
  real regression" message.
- Other interpretation tiers unchanged; `cold_start_suspected`
  only intercepts the < 0.95x AND low-attn-pct combination.

#### CLAUDE.md

Added a "Testing / `tests/bench_e2e_ltx.py` warmup auto-detection"
subsection documenting the two-signal requirement and the
asymmetric-cost reasoning behind the auto-mode design (false-
positive worse than false-negative; always errs toward warmup).

### v0.5.1 -- 2026-04-27  (e2e validation: bench infra fixes + first end-to-end speedup measurement)

The first-ever empirical measurement of sage's end-to-end speedup
on a production workload, after a series of bench-infrastructure
fixes that landed across the day. Headline result: **sage delivers
1.22x e2e speedup on the canonical LTX 2.3 audio-loop workload** at
832x480x497 / 25fps / 8-step distilled, with consumer-side fixes
(skip_under_seq_len short-Q skip + VAE decode normalized to a
single tile) in place.

**VISION.md item-3 status: confirmed (with refinement).** Kernel-ms
partially translates to gen-ms. Sage's 2.66x-at-attention kernel
speedup translates to ~1.22x end-to-end. The translation factor is
bounded by both Amdahl (attention is 8% of wall on this workload)
AND sage's per-call reach beyond attention rows -- FFN-adjacent
amortization adds ~30% of headline savings on top of the strict-
attention prediction. The "kernel ms = gen ms" simplification is
not literally true; sage is load-bearing, kernel work is justified,
but the next round of e2e wins routes to non-attention bottlenecks
(VAE decode amortization, caching, scheduler overhead) per
downstream consumer's Phase 2.1.

#### Added

- **`tests/bench_e2e_ltx.py --warmup {auto,always,never}`**
  (commit `aae9b9e`). Replaces the boolean `--no-warmup` (preserved
  as alias). `auto` mode probes
  `coderef/.../data/runs/<RUN_ID>/sage.jsonl` mtime; skips warmup
  when a recent trace (< 30 min) is found. Saves ~250s on
  iterative bench sessions. Discovered necessary 2026-04-27 when a
  cold `--runs 1` bench reported sage 2x SLOWER end-to-end while
  attention was only 4.5% of wall -- structurally impossible from
  sage alone; cold-start order effect was the real cause. The
  warmup-and-discard fix (`05a63e8`) addressed the bias; the
  auto-detect (`aae9b9e`) made it free on warm sessions.
- **`tests/bench_workload_profile.py` skip_reason aggregation**
  (commit `6802b2d`). `parse_traces` now buckets rows where
  consumer policy short-circuits sage (e.g.
  AudioLoopHelper's `skip_under_seq_len`, their commit `04919fd`,
  2026-04-27) under `skipped:<reason>` synthetic kernel names.
  New `print_skip_reasons()` section surfaces "X calls
  policy-skipped" alongside sage dispatch counts. Module constant
  `SKIPPED_KERNEL_PREFIX = "skipped:"` so downstream readers can
  discover the bucket explicitly.

#### Fixed

- **`tests/bench_e2e_ltx.py::resolve_run_id` RUN_ID auto-resolution**
  (commit `ea93006`, fix flagged by AudioLoopHelper claude as their
  Bug #2). When neither `--run-id` flag nor `$RUN_ID` env var was
  set, the bench fell back to globbing the legacy
  `internal/analysis/runs/sage/sage_*.jsonl` directory and found
  yesterday's most-recent file. Today's active trace at
  `data/runs/<RUN_ID>/sage.jsonl` was never considered. Fix scans
  for the most-recently-modified directory matching the consumer's
  `start_experiment.sh` format (`^\d{8}T\d{6}Z_[0-9a-f]{4}$`) and
  uses it. Common-case foot-gun for any operator running the bench
  from a fresh terminal that didn't inherit RUN_ID.
- **`tests/bench_e2e_ltx.py` non-attn-time print formula**
  (commit `05a63e8`). Old line 525-526 computed
  `non_attn_off - (off_med - on_med)` which simplifies to `on_med`
  -- the print was always showing the on-arm wall time mistakenly
  labeled as off-arm non-attn estimate. Replaced with factual
  off-arm wall surface; off-arm has no per-call tracer, so attn
  vs non-attn breakdown is unavailable and we say so.

#### Refactored

- **`tests/bench_e2e_ltx.py::_iter_trace_rows()` helper**
  (commit `aae9b9e`). Extracts the JSONL row-iteration primitive
  from four near-identical loops (`sum_attn_us_for_prompt`,
  `sum_attn_us_in_window`, `_trace_has_prompt_id`, plus
  `bench_workload_profile::parse_traces`). The four call sites
  collapse to ~3 lines each. Skips empty lines, JSON parse
  failures (mid-write tail), and (by default) framing rows
  (`event in {"header", "summary"}`).

#### Changed

- **Dropped the misleading Amdahl-ceiling note in bench output**
  (commit `aae9b9e`). The earlier note (`05a63e8`) printed a
  "ceiling X.XXx" derived from attention-fraction Amdahl when
  `attn_pct < 20%`. Per the 2026-04-27 cross-arm `exec_log`
  analysis, sage's reach extends beyond per-call attention rows
  (~26-28s sampler savings on top of ~11s pure-attention delta);
  pure-attention Amdahl is a LOWER bound, not a ceiling. An
  operator reading "ceiling 1.03x" while the actual ratio is
  1.22x walks away with the wrong story. Replaced with a factual
  one-liner pointing the reader at the empirical speedup ratio
  printed above.

#### Measured (new headline numbers)

`tests/bench_e2e_ltx.py` against
`audio_loop_latent.api.json` at 832x480x497 / 25fps / 8-step
distilled / `[1,1,1]` VAE decode tiles, with
AudioLoopHelper's `skip_under_seq_len=1024` widget enabled
(consumer commit `04919fd`):

| arm                | wall    | sampler  | VAE decode | sage attn  |
|--------------------|--------:|---------:|-----------:|-----------:|
| sage_on (cold VAE) | 138.8s  | 82.4s    | 47.4s      | 11.45s (8.2% wall) |
| sage_off (warm VAE) | 123.8s | 110.1s   | 10.4s      | n/a (no tracer) |

- **sage_on / sage_off raw**: 0.900x (sage 14s slower; VAE
  cold-start dominates).
- **VAE-cold-start-normalized** (subtract 37s arm-1 premium):
  138.8s - 37s = 101.8s vs 123.8s = **1.22x e2e speedup**.
- **Sage sampler savings**: 110.08s - 82.41s = **27.67s** (25%
  of sampler wall).
- **Pure-attention Amdahl prediction**: 8.2% attn x (1 - 1/2.66)
  = ~5.1% e2e speedup, i.e. ~1.05x. Observed 1.22x is **17 points
  higher** than this prediction; the surplus is sage's
  FFN-adjacent reach via int8 amortization + kernel pipelining
  effects within the sampler step beyond the attention rows
  themselves.

#### Findings worth flagging

- **VAE decode is the new headline bottleneck on this workload.**
  Even with single-tile decode, arm-1 cold-start carries a 37s
  premium over arm-2 (5GB+ activation buffer alloc, per-shape
  autotune, possibly cuBLAS workspace). Consumer-side Phase 2.1
  routes here next; sage-fork has no scope claim on VAE work
  (conv-style operators, not attention).
- **Both priors were wrong in the same direction.** sage-fork
  predicted 0.95-1.05x (wash), then revised to 1.05-1.10x
  (small win). AudioLoopHelper claude predicted 1.05-1.10x
  (revised from earlier 1.30-1.80x). Actual 1.22x. Both anchored
  on attention-fraction Amdahl and missed the FFN-adjacent reach
  empirically. Lesson: when sage's per-call wins translate to
  end-to-end, measure the boundary sage actually patches (the
  sampler), not the boundary the per-call timing exposes (the
  attention row).
- **`skip_under_seq_len=1024` working as designed.** Workload
  profile confirms 2304 of 4608 attention calls (50%) are
  policy-skipped before reaching sage; all are seq < 1024
  short-Q rows where sage was 0.45x torch_flash per the v0.4.1
  bench. The skip widget delivers ~11% wall-time savings on its
  own (cold-vs-cold; AudioLoopHelper's pre-vs-post-skip
  measurement, isolated from cold-start).

#### Cross-repo coordination

- AudioLoopHelper claude shipped `skip_under_seq_len=1024` widget
  + `prompt_id` contextvar fix in `04919fd`, plus per-prompt
  RUN_ID routing in `abe443b` (`AUDIOLOOPHELPER_PER_PROMPT=1` env
  var). Memo trail at `coderef/.../internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md`
  + `coderef/.../internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md`.
- Field-name compat discipline: their addition of
  `skipped: bool` + `skip_reason: str` trace fields was
  pre-vetted (pure-additive; default-skip-unknown semantics on our
  side). No breaking change.

### v0.5.0 -- 2026-04-27  (dead-code removal: Hopper/Blackwell + Windows + upstream bench)

Aggressive cut of upstream code that doesn't serve sm89/Ada or our
active research surface. We own this fork; we never push to upstream;
the cost of carrying unused code is more than the cost of removing
it. Each removal landed as its own commit so `git revert <sha>`
works granularly per arc.

#### Removed (4 arcs, ~7000 lines, no functional change for sm89)

**Arc 1 -- `sageattention3_blackwell/` (commit ceddb19, -5027 lines)**
The sage 3 Blackwell subpackage targets sm120+ with FP4 quantization.
Completely isolated -- zero imports from `sageattention/`, no
`setup.py` references, no test coverage. 27 files removed.

**Arc 2 -- sm90 Hopper kernel + Python wrapper + entry point (-1285 lines)**
- `csrc/qattn/{attn_cuda_sm90.h, pybind_sm90.cpp, qk_int_sv_f8_cuda_sm90.cu}`
  -- the Hopper CUDA kernel sources.
- `sageattention/sm90_compile.py` -- the Python custom-op wrapper.
- `sageattention/core.py::sageattn_qk_int8_pv_fp8_cuda_sm90` (~170-line
  function), the `SM90_ENABLED` guard + try/import block, and the
  `arch == 'sm90'` dispatcher branch in `sageattn()`.
- `KERNEL_FP8_CUDA_SM90` constant + `'fp8_cuda_sm90'` entry from
  `KernelName` Literal and `KNOWN_KERNEL_NAMES` frozenset.
- `sageattn_qk_int8_pv_fp8_cuda_sm90` from
  `sageattention/__init__.py` exports.
- `setup.py`: the SM90 `CUDAExtension` build block + the
  CUDA-12.3-for-9.0 minimum-version check.
- `build.sh`: `_qattn_sm90` from the verify-extensions inventory + the
  "sageattn3 (Hopper/Blackwell)" line in the available-kernels print.

Build verifies clean post-removal: `_qattn_sm80` + `_qattn_sm89` +
`_fused` import. Telemetry test passes (dispatcher routes correctly,
fp8_cuda++ on unmasked / fp16_triton on masked, soft-warns on
hand-picked CUDA + mask). Regression-check unit tests pass.

**Arc 3 -- `bench/` upstream one-shape benchmarks (-706 lines)**
9 files (`bench_baseline.py`, `bench_fa3.py`, `bench_fa3_fp8.py`,
`bench_qk_int8_pv_fp16_cuda.py`, `bench_qk_int8_pv_fp16_triton.py`,
`bench_qk_int8_pv_fp8_cuda.py`, `bench_qk_int8_pv_fp8_cuda_sm90.py`,
`utils.py`, `README.md`). Zero references from any code we keep.
Superseded by:
- `tests/test_sageattn_ltx_shapes.py` -- production-shape sweep across
  every sage kernel + 3 torch SDPA backends + FlashInfer/Sparge gates.
- `tests/test_sageattn_image_shapes.py` -- image-gen head_dim coverage.
- `tests/bench_e2e_ltx.py` -- end-to-end gen wall-time bench via
  ComfyUI HTTP API.
- `tests/bench_workload_profile.py` -- consumer-trace aggregator with
  coverage-gap analysis.

**Arc 4 -- build.sh `full` mode + Windows compile flags (-15 lines)**
- `build.sh`: dropped `./build.sh full` action that targeted Hopper +
  Blackwell arches we never validate. Default `./build.sh` targets
  `8.0;8.6;8.9` (Ampere + Ada); env override `CUDA_ARCHES`
  preserved for explicit Hopper/Blackwell builds.
- `setup.py`: dropped `os.name == "nt"` branches in `CXX_FLAGS`
  (MSVC `/O2 /openmp /std:c++17 /permissive-`) and
  `NVCC_FLAGS_COMMON` (`-D_WIN32=1 -DUSE_CUDA=1`). README narrows
  install to Linux+source; the Windows wheel paths upstream
  maintained are not validated here.

#### Why this is right

We own the fork; there's no upstream to send PRs to. Hopper/Blackwell
kernels won't run on sm89 even if compiled. Windows-build paths exist
upstream but are untested on this fork. Upstream's one-shape `bench/`
scripts measured one number with no rtol guardrails -- our LTX-shape
bench measures every kernel + every torch backend on production-
relevant shapes with `--check-regression` gating. Carrying unused
code inflates audit surface, build time, pyright noise, and the
"what does this fork actually do" question every reader has to
re-answer. The cost of removal is bounded; the cost of carrying it
recurs every time anyone reads the tree.

#### What's preserved

- sm80 kernel (forward-compatible to Ada via CUDA backward-compat;
  still powers `sageattn_qk_int8_pv_fp16_cuda`).
- sm89 kernels (the production hot path; `sageattn_qk_int8_pv_fp8_cuda`
  and the `++` variant).
- Triton kernels (mask-correct path; `sageattn_qk_int8_pv_fp16_triton`).
- The dispatcher's `arch in {"sm100", "sm120", "sm121"}` branch -- it
  routes to the existing sm89 kernel; one-line forward-compat for
  Blackwell users who happen to install this fork. Removing it would
  be churn with no win.

#### Verified, no action

- Dispatcher telemetry test -- 11 cases pass post-removal.
- Regression-check unit tests -- 7 cases pass post-removal.
- Build verify mode -- all expected extensions importable.

### v0.4.1 -- 2026-04-27  (bench coverage realigned to production; head_dim claim corrected)

Closes a load-bearing measurement bug. Every perf decision the fork
made before today was graded against `self_attn_large_704x704x497`
at `seq=31776, head_dim=64` -- a synthetic shape with the wrong
head_dim. LTX 2.3's video path is `attention_head_dim=128`
(`diffusers/models/transformers/transformer_ltx2.py:907-947`); the
`d=64` value was the audio side, mis-attributed to the whole model.
Production traces show zero calls at the synthetic shape.

The new bench coverage is sourced from a real consumer trace
(`sage_2026-04-26_105851.jsonl`, 6912 attention calls). The
`tests/bench_workload_profile.py` script (also shipped this session)
is the durable discovery tool -- every future trace gets routed
through it before bench-shape decisions.

#### Added

- **`tests/regression_baselines.json`** -- pinned `(shape, mode) ->
  (median_ms, mean_rtol)` baselines for the load-bearing rows. Schema
  documents `rtol_budget=0.10`, `perf_drift_pct=5.0`,
  `speedup_ratio_floor=1.5`. v0.4.1 baselines captured on RTX 4090 /
  sm89 / CUDA 13.0 / torch 2.11.0+cu130 / triton 3.6.0 / sage rev
  `8f737c3`.
- **`tests/test_sageattn_ltx_shapes.py --check-regression`** -- new
  CLI flag that grades fresh measurements against the baselines and
  exits non-zero on perf drift > 5%, rtol budget breach (> 0.10),
  speedup-ratio floor breach (sage_fp8++/torch_flash < 1.5x), or
  missing measurement on a load-bearing row. Soft `RTOL_DRIFT` alarm
  at 1.5x baseline (kernel-internal numerical change signal even when
  under the budget).
- **`tests/bench_workload_profile.py`** -- aggregates a consumer's
  sage trace JSONL into per-`(shape, has_mask, dispatched_kernel)`
  call counts, total wall time, and a coverage-check pass against
  `regression_baselines.json`. Surfaces "Coverage gaps" -- trace
  shapes the bench doesn't measure -- which is what justifies adding
  bench rows. Trace-freshness diagnostic mirrors the consumer's
  `sage_telemetry / legacy_inferred` bucketing.

#### Changed

- **`tests/test_sageattn_ltx_shapes.py` SHAPES list** replaced with
  production-aligned rows. New set: LTX 2.3 video self-attn at d=128
  (init seq=22932, loop seq=23296), audio self-attn at d=64 (same
  seq), short-Q paths at seq=497/498 (Gemma 3 text-encoder or audio
  cross-attn, attribution ambiguous from trace), K-probe pair at
  d=128 / kv=226 (the one masked row that survives -- doubles as
  v0.3.0 dispatcher mask-routing correctness witness). Synthetic
  wide-V stress shape kept (kernel robustness check, not a workload).
- **Dropped from SHAPES**:
  - `self_attn_large_704x704x497`, `self_attn_small_512x512x97` --
    speculative seq + wrong head_dim.
  - `cross_attn_text_kv32`, `kv64`, `kv128`, `kv512`, `kv1024` --
    re-measured a documented CUDA-mask-bug fingerprint without any
    gating purpose. Kept only `kv226` as the v0.3.0 dispatcher
    correctness witness + K-probe pair anchor.
  - `cross_attn_unmasked_kv226_kratio_probe` (old d=64 version) --
    replaced by the d=128 version at production seq.
- **CLAUDE.md** -- "Performance research / load-bearing metric"
  section updated. Primary row is now `ltx23_video_self_attn_init_22932 /
  fp8_cuda++` at 20.20 ms / 0.098 rtol. Speedup ratio at the
  production shape: 2.66x (vs old 2.62x at the synthetic shape; the
  perf story holds). "Shape coverage today" paragraph rewritten with
  the corrected video d=128 / audio d=64 split + cite to
  `transformer_ltx2.py:907-947`.

#### Measured (load-bearing)

RTX 4090 / sm89 / CUDA 13.0 / torch 2.11.0+cu130 / bf16:

| shape                                     | mode         | median_ms | mean_rtol | vs torch_flash |
|-------------------------------------------|--------------|----------:|----------:|---------------:|
| ltx23_video_self_attn_init_22932          | fp8_cuda++   |     20.20 |    0.0978 |          2.66x |
| ltx23_video_self_attn_loop_23296          | fp8_cuda++   |     20.52 |    0.0977 |          2.70x |
| ltx23_audio_self_attn_init_22932          | fp8_cuda++   |     10.65 |    0.0980 |          2.59x |
| ltx23_audio_self_attn_loop_23296          | fp8_cuda++   |     10.92 |    0.0977 |          2.53x |
| ltx23_short_q_init_497                    | fp8_cuda++   |      0.08 |    0.0934 |          0.45x |
| ltx23_short_q_loop_498                    | fp8_cuda++   |      0.09 |    0.0923 |          0.45x |
| ltx23_video_cross_unmasked_kv226_probe    | fp8_cuda++   |      0.74 |    0.0904 |          1.12x |
| ltx23_video_cross_text_kv226              | fp16_triton  |      1.16 |    0.0406 |       n/a (mask) |
| ltx23_video_cross_text_kv226              | auto         |      1.16 |    0.0406 |       n/a (mask) |

K-probe at d=128 / kv=226: K = 1.16 / 0.74 = **1.57** (vs 1.68 at
the old d=64 row; well below the 5x trigger for native CUDA mask
kernel work).

#### Findings worth flagging

- **Short-Q rows where sage loses to torch_flash.** seq=497 / 498
  fp8_cuda++ runs at ~0.45x of torch_flash's wall-time. int8 quant +
  kernel launch overhead exceeds the matmul work at that shape. This
  is the empirical evidence behind the consumer's `nodes_sage.py`
  deferred "min-sequence skip" backlog item -- the gate is now
  measurable. Trigger to act: a downstream consumer wires the
  short-Q skip and we re-measure end-to-end gen time.
- **Speedup ratio held up.** The "wrong head_dim" framing was a
  correctness-of-narrative bug, not a perf-magnitude bug. fp8++ at
  d=128 / production seq is 2.66x torch_flash, vs the old 2.62x at
  d=64 / synthetic seq. Sage's load-bearing claim is intact.
- **fp16_cuda is still mask-broken at d=128.** rtol 0.44 on
  `ltx23_video_cross_text_kv226 / fp16_cuda` -- the v0.3.1 soft-warn
  fires correctly. Same underlying bug fingerprint as the old d=64
  measurement.

#### Why this wasn't done sooner

The bench's primary shape was inherited from an earlier session
without checking it against a real trace. CLAUDE.md's "LTX-2.3:
head_dim=64" claim was treated as ground truth without grepping the
diffusers config. The discovery happened only because
`tests/bench_workload_profile.py` shipped this session and the first
trace it consumed produced zero `[HIT ]` lines on the load-bearing
set. The fix is to make the workload-profile coverage check the
default discovery tool before any future bench-shape decision.

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
