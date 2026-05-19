# Roadmap

Last updated: 2026-05-19

Forward-looking record of directions worth pursuing on this fork --
ranked by relevance to the current workload, technically scoped, and
trigger-conditional. **Not a committed schedule.** This doc enumerates
the option space so future-session decisions don't re-derive it; the
user remains the scheduler.

## How this doc relates to other docs

- **`VISION.md`** -- canonical scope (what this fork IS and IS NOT).
  Edit rarely; this roadmap edits don't normally require a VISION
  edit unless scope itself moves.
- **`CHANGELOG.md` Backlog + Decision log** -- concrete items with
  specific triggers. The roadmap can promote items to Backlog
  (active) or demote them to Decision-log (skipped). When an item
  here gets concrete enough to act on, it migrates to Backlog.
- **`docs/perf_research_framework.md`** -- methodology that anything
  here will be measured by. Especially the evidence ladder and the
  synthetic-vs-in-pipeline discipline.
- **`docs/ltx_workload_profile.md`** -- canonical workload share
  data. Anything claiming "X% of wall" cites this.
- **`internal/PLAN.md`** (gitignored) -- live operational state.

## What we're specialized in (the portfolio)

The repo is called "sage-fork" but the substantive expertise is
broader than sage attention. The actual portfolio:

1. **sm89 quantized kernel work** in Triton + CUDA. Both attention
   (sage today) and non-attention (sage_ffn proved the path).
   fp8/int8 quantization patterns, mask kernels, fused activations.
2. **Profiler-driven perf research methodology.** Synthetic-vs-
   in-pipeline gap framework, kernel-name-presence evidence ladder,
   both-arms-measured discipline, disprove-test rule, cpu_op.dur
   trap. Reusable across any kernel-day work.
3. **ComfyUI custom-node integration patterns.** fp8 storage
   conventions across ComfyUI versions, `add_object_patch`
   composition discipline, `prior_forward` chaining, downstream-
   known-symbols audit, cross-clone coordination protocols.
4. **DiT-class architecture analysis on sm89/fp8 specifically.**
   Workload profiling, sub-module attribution, kernel-fire vs
   kernel-share separation, mask-routing reasoning.

This portfolio is the lens for ranking what's worth pursuing. **Not
"all GPU work."** It's *"sm89 fp8/int8 kernel work for ComfyUI-class
consumer workloads, with rigorous measurement."*

## Tier 1: High-relevance, concrete, anchored to current workload

The reference workload: LTX 2.3 video gen + Gemma3 12B text encoder
+ VAE encode/decode, on RTX 4090 / sm89 / 24 GB VRAM. Sub-module
shares per `docs/ltx_workload_profile.md`.

### 1.1 Workflow profiler tool -- RETIRED (consume the consumer-side `inspect_run.py` instead)

**Status (2026-05-19): retired from active build queue.** The
consumer-side audio-loop claude is already building the equivalent
tool as `scripts/inspect_run.py` on their side. Their phase plan:

- **Phase 1** (concrete specs landed): console-log error scan, sage
  routing health, per-iteration consistency, workflow-vs-execution
  mismatch, tracer manifest reconciliation.
- **Phase 2** (deferred, requires chrome-trace parsing): per-stage
  kernel-time breakdown (the methodology used to land the Cell C
  verdict), initial-render-vs-loop-body symmetry, attention shape
  outliers.
- **Phase 3** (deferred, A/B mode): treatment-vs-baseline kernel-
  time diff that auto-generates the per-stage decomposition table
  the Cell C audit required as a 50-line one-off script.

**Why we retire rather than parallel-build:** they have the data
context (consumer-side runtime, workload exposure across the audio-
loop + two-pass + IC-LoRA classes), and building a parallel
sage-side workflow profiler would duplicate effort against the same
methodology. Better disposition: **consume their tool's output for
our bench-side decisions** and pair on Phase 2 / Phase 3 if useful.

**Implication for downstream items:** Tier 1.2 (VAE decoder fp8
fusion experiments) gating on "workflow profiler data shows VAE is
non-trivial share" now waits on Phase 2 of their tool, not on us
building a tool. Same for Tier 2.4 (cross-modal attention coverage)
which gates on profiler data showing cross-modal attn is non-
trivial share. We are consumers of their output, not blocked on our
own build.

**Trigger to revive (rebuild on our side):** if their tool stalls
indefinitely AND a kernel-side decision blocks on data we can't get
from manual chrome-trace inspection, we'd build a sage-side version
focused on kernel-isolation methodology (Phase 2 check 7 in their
phase plan -- the part that intersects our sm89 optimization work).
Until then, retired.

### 1.2 VAE decoder fp8 fusion experiments

**What:** fused fp8 Triton kernels for the LTX VAE decoder's
conv + norm + activation blocks. Same toolkit as sage_ffn -- per-
tensor fp8 weights, fused activation, mask the intermediate from
HBM. Different kernel shape (3D conv with small kernels, not
matmul) but same fp8/sm89 methodology.

**Why:** VAE decode is often the longest serial chunk in video
output. If it's >5% of e2e wall on the canonical workload, this is
the highest single-kernel wedge after the denoiser loop. If <2%,
not worth the kernel-day cost.

**Technical shape:** depends on (a) whether LTX VAE is fp8-quantized
in the distilled checkpoint or bf16 (significantly different
problem -- bf16 fusion has a smaller wedge), (b) the actual decoder
module structure (norms, activations, residuals), (c) which
sub-graphs are fusable under the convolution operators sage's
Triton stack supports.

**Effort:** hard to scope without profiler data. If VAE is fp8:
likely 2-3 weeks of work matching sage_ffn's pattern. If VAE is
bf16: different problem, smaller wedge, lower priority.

**Trigger to act:** sub-module attribution data showing VAE decode
>= 5% of e2e wall on a canonical workload AND VAE weight format
confirmed as fp8 or bf16 from the checkpoint audit.

### 1.3 Persistent-CTA hybrid for sage_ffn (and sage attention)

**What:** rewrite `_fp8_matmul_gelu_kernel` + `_fp8_matmul_kernel`
as a single fused-three-stage kernel with persistent CTAs. CTAs
hold M-tile state in registers / L2 across the gate + up + down
pipeline (or across attention's QK + softmax + PV), reducing the
L2 thrash that the v0.6 walk-back identified as the root cause of
the +1.79% e2e regression at the canonical workload.

**Why:** directly addresses the v0.6 e2e gap that synthetic bench
projected (1.26-1.36x) but production refused to follow (-1.79%).
Without persistent-CTA, sage_ffn ships as "completeness primitive"
indefinitely.

**Technical shape:** persistent-CTA in Triton is non-trivial.
SMEM budget on sm89 (164 KB per SM) constrains tile size. Likely
2-stage SMEM pipelining for the intermediate. Need to validate
correctness on rtol vs the existing 2-kernel reference at each
shape in the LTX FFN coverage.

**Sub-pattern shape (per the "External design references" section
below):** the cleanest articulation is producer/consumer warp split
with explicit roles -- loader warps doing `cp.async` for x_chunk +
w_chunk; launcher/controller logic managing the per-stage semaphore
state; consumer warps doing `tl.dot` + scale + GELU. Next-tile-id
sourced from `atomicAdd` on a global counter (sm89-equivalent of
cluster launch control). Pipeline depth (4-deep load + 8-deep
epilogue from the Megakernels reference) retuned for sm89's 164KB
SMEM. Compile-time `static_assert`-equivalent shmem budget check at
autotune-config-generation time so budget violations surface at
compile rather than runtime.

**Effort:** 2-3 weeks per CHANGELOG estimate, Triton-shaped. Could
drop to 1-2 weeks if Triton 3.7's persistent-kernel primitives land
cleanly on sm89 -- worth a ~half-day spike against
`tl.dot_scaled` + improved `tl.program_id` patterns before
committing. Higher confidence on FFN first (concrete e2e regression
data); attention follows if the pattern holds. A faithful `.cu`-side
persistent-CTA modeled on Megakernels' shape would be 4-8 weeks and
isn't worth the additional cost over the Triton-shaped version.

**Trigger refined 2026-05-19 (Cell C verdict confirmed):** the
v0.6 e2e gap is NOT closable via consumer-side `prior_forward`
chaining alone. The re-baseline render landed with TREATMENT at
+0.75% wall (188.3s vs 185.7s, within ±3s noise) AND with sage_ffn
*per-kernel* at 22% slower (stage-1) / 5% slower (stage-2) vs
production stock fp8. Synthetic 1.39x/1.60x advantage at the same
shapes did not transfer; production has the sign flipped. Two open
hypotheses for the inversion documented in CHANGELOG Decision log:

  (1) Stock comparand identity (synthetic vs `torch._scaled_mm`,
      production vs `comfy.ops.fp8_linear` + ChunkFFN).
  (2) Sage autotune state under interleaved attention + FFN dispatch.

Persistent-CTA targets (1)/(2) symmetrically by removing the
L2-thrash pathway between matmuls; this item stays load-bearing for
closing the v0.6 e2e gap. Promotes to **active Backlog status** in
CHANGELOG (was conditional; now confirmed).

Alternative attack vector worth considering before committing to
the 2-3 week kernel-day spend: §6.1 (concurrent-dispatch consumer
wrapper, ~5-13% e2e prize untapped per the v0.6.1 stream-safety
fix). If concurrent-dispatch ships first and closes the e2e gap by
launching attention + FFN streams concurrently, persistent-CTA's
priority drops back to "validates the technique" rather than
"closes the gap."

## Tier 2: Medium-relevance, conditional

### 2.1 Generalize sage_ffn to handle GeGLU

**What:** extend sage_ffn from
`(x, w1, s1, w2, s2)` to
`(x, w_gate, s_gate, w_up, s_up, w_down, s_down)` with GeGLU gating
(`gate(x) * GELU(up(x))` rather than `GELU(linear(x))`). Adds one
matmul (the gate projection) and a pointwise gate*GELU(up) at the
intermediate.

**Why:** Gemma3 12B uses gated FFN. Many other modern transformer
models (LLaMA-class, Mistral-class) use SwiGLU which is the same
shape with different activation. Today sage_ffn is plain-GELU only
because LTX 2.3's FFN was confirmed plain-GELU.

**Technical shape:** new top-level `sage_ffn_geglu(...)` plus a new
`_fp8_matmul_geglu_kernel` modeled on the existing
`_fp8_matmul_gelu_kernel`. Significant code reuse. Tests follow
the v0.6.2 pattern -- informative asserts + happy-path correctness.

**Effort:** ~1 week.

**Trigger to act:** confirmed time-share or memory-pressure data
showing Gemma3 text encoder is a real wedge in the workload. Today
the prior is: text encoder runs once per prompt, amortized across
~25-50 denoising steps, so the per-render cost is low; but memory
pressure on a 24 GB card with a 12B-param encoder is real. fp8
quant could help even without speed gain. Need data before
committing.

### 2.2 Triton autotune pre-bake as a release artifact

**What:** ship `sage_autotune_cache_sm89_<env>.json` alongside the
sage package. Pre-computed Triton autotune winners for every known
shape on the LTX 2.3 coverage. Loads on first import; skips
cold-render autotune sweeps entirely. Generalize across all sage
kernels + sage_ffn + any future kernels (the discipline rule from
CLAUDE.md's "Triton kernel-day discipline" section, productized).

**Why:** cold-render UX on user hardware is bad (~100-500 ms per
new shape × ~30 unique LTX shapes = ~10 s of first-render lag).
Pre-baking eliminates this. Independent of any kernel speedup.

**Technical shape:** capture autotune cache via `kernel.cache.items()`
after a full bench run; serialize to JSON; load on
`sageattention.__init__` via `triton.autotuner.load_cache()` (or
equivalent if the API has shifted). May need per-(torch version,
triton version, CUDA version) split since autotune output can vary.

**Effort:** ~3-5 days including the per-version-split logic.

**Trigger to act:** user-reported friction with first-render cost,
OR demonstrated benefit from a one-off pre-bake on the canonical
workload. Cheap enough that "wait for trigger" is conservative.

If 2.4 (cross-modal attention coverage) activates first, the
pre-bake scope must cover both HEAD-DIM 128 AND HEAD-DIM 64
template instantiations -- otherwise pre-bake benefits only the
main self-attention path while cross-modal attns still pay the
cold-render autotune cost.

### 2.3 fp8/int8 research for video diffusion specifically

**What:** methodology + measurement work documenting how fp8/int8
techniques transfer (and don't) from LLM regimes to video diffusion
regimes. The differences: bidirectional vs causal attention, very
long sequences (10k-50k tokens vs 2-8k typical), non-causal masks
that span the full attention matrix, GeGLU/GELU mix vs SwiGLU-
dominated, fp8 weights at rest (model is stored fp8) vs inference-
time quant.

**Why:** real research gap. Most published fp8 work is LLM-shaped.
Video diffusion at production sequence lengths is an underexplored
regime. Plausible publishable artifact -- blog, paper, or talk --
if the user wants public-facing output.

**Trigger to act:** user-driven (not data-driven). If findings
worth sharing surface organically, share them; otherwise this stays
"keep the option open" and structures no work.

### 2.4 Cross-modal attention bench coverage + HEAD-DIM 64 autotune

**What:** extend `tests/test_sageattn_ltx_shapes.py` to cover
HEAD-DIM 64 attention shapes in addition to today's HEAD-DIM 128
coverage. Multi-modal workflows (audio-conditioned video, cross-
attention pipelines) dispatch the HEAD-64 sage template alongside
the HEAD-128 template in the same render -- different sub-modules,
different tile-config space. Today's bench measures only one of
the two template instantiations.

**Why:** two specific data points motivate this. (a) A cross-clone
trace observation (CHANGELOG "Workload intel") of a two-pass tensor-
loop workflow shows 1536 HEAD-128 + 384 HEAD-64 dispatches per
render. (b) An attention-kernel slowdown of 2.14x at a 3% seq-length
increase in the same trace surfaced direct evidence of autotune
flipping under interleaved dispatch (Cell C hypothesis 2,
corroborated). Without HEAD-64 bench rows we cannot pre-bake the
right autotune configs and we cannot measure the cross-modal-attn
share of e2e wall time.

**Technical shape:** add shape entries to the LTX shape table for
HEAD-64 cross-attention call sites (audio cross-attns, upsampler
stages, or whatever the consumer workflow surfaces). Reuse the
existing `accuracy_metrics` + median-timing harness. Companion
update to `tests/regression_baselines.json` once shapes are
committed.

**Effort:** ~1-2 days for bench rows + baseline calibration.
Companion autotune-pre-bake scope expansion is captured in 2.2's
note above.

**Trigger to act:** consumer-side `inspect_run.py` Phase 2 data lands showing
cross-modal attention is non-trivial (>3% of e2e wall) on a real
consumer workload, OR a second cross-workload observation of HEAD-64
dispatches surfaces (turning the "Workload intel" entry from
one-off to recurrent). Until either fires, the existing HEAD-128
coverage is sufficient.

### 2.5 ComfyUI quant compatibility shim package

**What:** generalize `sageattention.extract_fp8_weight_and_scale`
(v0.6.4, currently a single utility in `sageattention/comfyui_compat.py`)
into a small standalone package or subpackage covering the broader
"resolve quantized parameters across ComfyUI versions" surface.
Candidates: fp8 weights + scales (today), fp8 biases (if a future
convention surfaces), int8 weights + scales (if int8 quant lands in
ComfyUI), Linear-vs-Conv2d quant convention variance.

**Why:** the v0.6.4 utility was driven by one specific consumer-
side bug (QuantizedTensor wrapper unwrap). ComfyUI's fp8 storage
convention has shifted at least three times (legacy `scale_weight`
attr → older `weight_scale` attr → modern `QuantizedTensor._params`
/ `layout_params`). Every future consumer who wants to extract
quantized parameters re-derives the same probe. Centralizing the
probe protects every consumer from re-deriving the conventions and
getting bitten by the same trap. Tier 2 rather than Tier 1 because
the v0.6.4 single-utility version covers the immediate case; this
is "round out the shim into a more general surface as new
conventions surface."

**Technical shape:** either (a) keep as a subpackage of
`sageattention` and add new probe functions (`extract_int8_weight_and_scale`,
`extract_quantized_bias`, etc.) as ComfyUI conventions surface, OR
(b) split into a sibling repo (`comfyui-quant-compat` or similar)
that has no sage dependency and can be vendored by consumers that
don't want a sage import. Repo-structure question is open per the
existing "Repo structure" section below; current prior is
subpackage-of-sageattention.

**Effort:** ~3 hours per added probe function (matching the v0.6.4
estimate). Test coverage via mock-objects pattern established in
`tests/test_comfyui_compat.py`.

**Trigger to act:** a third (post-v0.6.4) ComfyUI fp8 storage
convention surfaces (would extend the existing probe), OR an int8
or other quantization scheme lands in ComfyUI that a downstream
consumer hits, OR a second consumer wrapper hits the missing-unwrap
trap on a different attribute (e.g. bias storage convention).
Three-incident threshold rather than two because the methodology
fold from v0.6.4 + the framework rung 2 expansion together cover
the single-utility case adequately.

## Tier 3: Lower-relevance, real but conditional

### 3.1 Mask-correct CUDA paths for sm80 + remaining sm89 variants

CHANGELOG-listed as deferred per scope discipline. Only matters if
sm80-masked workload surfaces or if our sm89 dispatcher routes to
one of the unfixed variants. Pattern is established from v0.5.5
sm89 fp8++ work; ~1 week each per variant.

**Trigger to act:** workload data showing one of the deferred
variants is being dispatcher-selected on a masked path.

### 3.2 CUTLASS-based CUDA backend for sage_ffn

CHANGELOG Decision-log: skipped per workload-profile analysis. The
revisit trigger is narrow ("persistent-CTA hybrid lands AND a
workload class surfaces where matmul throughput IS the bottleneck").
Reference intel for the lookup work that would be required is at
`internal/design/comfyui_fp8_storage_conventions.md`.

### 3.3 `torch.compile` compatibility revisit

CHANGELOG / `docs/torch_compile_spike.md`: skipped because pybind
kernels graph-break Dynamo. If torch 2.13+ changes the breakage
rules, the spike is worth re-running. Low priority.

### 3.4 Dispatcher session-start info log

Identified in the 2026-05-18 dispatcher audit (`sageattention/core.py`).
~5-line edit adds a one-shot `[INFO] sage routing: arch=... cuda=...
mask=... pv_accum=... -> <kernel>` per unique routing tuple at first
call to `sageattn(...)`. Helps consumer debugging; the routing-
correctness gate (test_dispatched_kernel_telemetry.py) catches drift
at test time but not in production logs.

**Trigger to act:** downstream consumer ask (already received).
Cheap enough that "wait for trigger" was the wrong call originally;
will ship next session it gets touched.

## Stack leverage opportunities

The modern stack has primitives we're underutilizing. Each entry:
what it offers, which open problem it intersects, effort. Version
anchors are inline where a feature is new-in-that-version; section
headers stay version-neutral so this section ages with the stack.

### Triton features

- **`tl.dot_scaled`** -- fp8 matmul with scale tensors as direct
  kernel args. If sm89-supported and stable, *kills the v0.6.5
  `w*_scale` must-be-Python-scalar foot-gun at the kernel level*
  rather than at the assert boundary. Our wrapper assert becomes
  defense-in-depth, not load-bearing. Worth a spike before any
  persistent-CTA rewrite to decide whether the new ABI is in the
  scope of the rewrite.
- **Improved persistent-kernel primitives** -- better `tl.program_id`
  patterns for persistent grids + cleaner async-copy primitives.
  Could *cut Tier 1.3 (persistent-CTA `sage_ffn`) effort estimate
  from 2-3 weeks to 1-2 weeks* if the primitives land cleanly on
  sm89.
- **Autotune cache control APIs** -- improved `kernel.cache`
  introspection + load/save. Intersects Cell C hypothesis 2
  (autotune flips between contexts) and Tier 2.2 (autotune pre-
  bake). May simplify pre-bake by eliminating the per-(torch,
  triton, CUDA) version-split logic.

**Trigger to act:** before any persistent-CTA rewrite (Tier 1.3) --
worth ~half a day spike confirming which primitives are sm89-stable
at the current Triton (3.7 at time of writing). If `tl.dot_scaled`
works, the v0.6.1+ kernel ABI could ship with native tensor-scale
support and we walk back the v0.6.5 wrapper assert to defense-in-depth.

### torchao primitives

A local torchao checkout is available; we can build from source.
torchao (0.18+ at time of writing) relevant surfaces:

- **`Float8Linear` with per-tensor + per-row scaling** -- production-
  quality fp8 Linear. Plausibly *what ComfyUI moves to* if/when they
  upgrade their fp8 stack. Our `extract_fp8_weight_and_scale` v0.6.4
  shim doesn't currently probe for a torchao storage convention; if
  it lands in ComfyUI, that's a 5th storage convention to handle.
- **`Float8RowwiseScaledTensor` / `Float8TensorWise`** -- proper
  tensor subclasses for fp8 with scales. Could be *the right
  comparand for our synthetic bench* (Cell C hypothesis 1: "stock
  comparand identity"). Benching sage_ffn vs
  `torchao.float8.Float8Linear` might give a more honest "is
  sage_ffn faster than the modern production-ready alternative"
  answer than benching vs raw `torch._scaled_mm`.
- **`torchao.prototype` kernels** -- research code including fused
  fp8 MLP variants. Possible direct prior art for sage_ffn that
  re-shapes the persistent-CTA effort or surfaces tile-config
  patterns we haven't tried.
- **MX scaling formats (mxfp8, mxfp4)** -- sm100+ only for the
  hardware path, but the software per-block scaling pattern could
  inform a future sage_ffn variant where per-tensor scaling hits an
  rtol ceiling.

**Trigger to act:** consumer-side `inspect_run.py` Phase 2 data lands AND the
comparand-identity hypothesis (Cell C hypothesis 1) needs
resolution, OR ComfyUI surfaces a torchao storage convention in
production. Either fires the read on torchao primitives.

### PyTorch features

- **`torch._scaled_mm_v2`** (PyTorch 2.12+) -- newer than
  `_scaled_mm`. Different tile selection. Our synthetic bench uses
  `_scaled_mm`; switching to `_scaled_mm_v2` could shift the
  synthetic-bench comparand and possibly the Cell C verdict at the
  bench layer.
- **`torch.cuda.tunable`** -- PyTorch's auto-tuning API for
  cuBLAS/cuBLASLt. Cross-session caching. May help with the
  autotune-state-under-interleaving problem (Cell C hypothesis 2)
  from a different angle than Triton's own autotune cache.
- **`torch.compile` improvements** -- `docs/torch_compile_spike.md`
  documented our skip at torch 2.11. If a later torch release
  changes the Dynamo graph-break rules at our pybind sites, the
  spike is worth re-running. CHANGELOG Backlog item 3.3.

**Trigger to act:** independently of any of the above firing.
`_scaled_mm_v2` swap in the bench is a ~30-min experiment that
could be done opportunistically; `torch.compile` revisit is
conditional on a spike at a torch version newer than 2.11.

## External design references

Cross-arch projects whose code we do NOT port (sm90+ only) but whose
patterns are worth internalizing before persistent-CTA work. Both
were audit-read 2026-05-19 via subagent investigation. Full notes
will land in `internal/design/persistent_cta_sage_ffn_scoping.md`
(gitignored; not yet authored, created when Tier 1.3 work
activates). Line numbers below are as of the 2026-05-19 audit-read;
refresh on re-read since upstream projects may renumber.

**Trigger to act:** before Tier 1.3 (persistent-CTA hybrid for
sage_ffn) work begins. ~5 hours of read time to internalize
patterns; no code port. The patterns transfer to sm89 by re-
implementing against `cp.async` + `mma.sync` instead of the sm90+
TMA/WGMMA primitives the upstream code uses.

### Megakernels (local checkout; sm100/sm103 only)

Specific patterns transferable to sm89 by re-implementing against
`cp.async` + `mma.sync` instead of TMA/WGMMA/cluster-launch:

- **Warp-role decomposition** (`csrc/megakittens.cuh`,
  `csrc/controller.cuh`, `csrc/itypes/gemm.cuh` in the Megakernels
  repo) -- explicit roles: controller / loader / launcher / consumer
  / storer warps. Software pattern, sm70+. On sm89: loader warps use
  `cp.async`; the next-tile-id source is `atomicAdd` instead of
  cluster launch control.
- **Page-release ordering** (`csrc/itypes/gemm.cuh` lines 39-46) --
  semaphore-driven SMEM page reuse. Portable as-is.
- **Pipeline depth tuning** (`LOAD_PIPE_DEPTH=4, EPI_PIPE_DEPTH=8`)
  -- discipline transfers; specific numbers retune for sm89's 164KB
  SMEM budget.
- **Fused two-Linear-around-activation pattern**
  (`csrc/itypes/llama1b/upgate.cuh`) -- decode-shaped (M=1) so not a
  code port, but validates the abstraction shape sage_ffn is moving
  toward.

### ThunderKittens (local checkout; sm90+ only)

Specific patterns transferable to sm89:

- **Producer/consumer warp split with per-stage semaphores**
  (`prototype/interpreter/interpreter.cuh:355`,
  `templates.cuh:9`) -- 4 producer warps doing async loads, 8
  consumer warps doing MMA + math. Software pattern. On sm89: same
  structure, `cp.async` instead of `tma::load_async`, sync `mma.sync`
  instead of WGMMA.
- **`task_iter`-driven dispatch** -- persistent kernel loops on a
  task index pulled from a global queue. On sm89: same structure,
  `atomicAdd` on a global counter instead of cluster launch control.
- **Compile-time tile shape `static_assert` shmem budget check**
  (`interpreter.cuh:50-56`) -- Triton-portable. Verify
  `BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N` fits SMEM at autotune-config-
  generation time, catching budget violations before runtime.
- **Bias-init lane-shuffle pattern** (`kernels/flux/flux_gelu.cu`
  `init_bias` at lines 26-41) -- portable; corroborates the bias
  handling sage_ffn already does.
- **fp8 per-tensor scale layout**
  (`kernels/gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu` lines 116-119)
  -- "accumulate fp32, scale at end, dequant to bf16" matches
  sage_ffn's numerics ordering. Cross-check, no port.

**What's NOT transferable from either project:** TMA, WGMMA,
`setmaxnreg`, `tcgen05`, MXFP8/NVFP4 hardware paths, 2-CTA clusters.
sm90+ features that don't exist on sm89. Megakernels' DAG-fusion
`torch.compile` backend is also out of scope (wrong abstraction
level for a kernel library).

**Adoption cost:** read ~5 hours before any persistent-CTA work to
internalize patterns. No code adoption. Effort estimate for the
Tier 1.3 rewrite (2-3 weeks Triton-shaped) is unchanged by reading;
a faithful `.cu`-side persistent-CTA modeled directly on
Megakernels' shape would be 4-8 weeks and not worth it.

## Tier 4: Explicit non-goals

- **Hopper / Blackwell support.** Out of scope per VISION.md.
  Reopen only if audience shifts.
- **Generic / cross-arch kernel rewrites.** Fights the scope.
- **Becoming an LLM-inference engine.** See VISION.md "What we are
  NOT" for the framing. Short form: shared kernel surfaces in scope
  on sm89 as DiT and LLM converge; autoregressive serving stack is
  not.
- **Polished public release infrastructure** (CI builds for
  multiple Python / CUDA / torch versions, polished docs site,
  user-onboarding flows). Solo-hobbyist scope; the README +
  CHANGELOG + this roadmap are sufficient. Revisit if scope shifts
  toward broader audience.

## Repo structure (open question)

Three plausible structures for new kernel work beyond sage attention
+ sage_ffn:

1. **Adjacent repos.** Each new kernel project (VAE fusion, GeGLU
   extension, profiler tool) ships in its own repo, depending on
   sage's methodology and bench discipline. sage-fork stays
   primitive. Higher discoverability cost; cleaner scope per repo.
2. **Subpackages in sage-fork.** Add `sage_vae/`, `sage_tools/`,
   `sage_ffn_geglu/` as subpackages. Single import surface for
   users; more churn for existing sage-fork consumers.
3. **Umbrella project** (e.g. `sm89-comfy-kernels/`) with sage as
   the first member. Most polish work; cleanest long-term if scope
   grows significantly.

**Current prior:** adjacent repos for new kernel projects (#1).
Sage-fork stays primitive per VISION. With Tier 1.1 retired in
favor of consuming the consumer-side `inspect_run.py`, the
first-candidate-for-sibling-repo slot is open; the next concrete
candidate is whatever Tier 2.5 (ComfyUI quant compat shim) becomes
if it splits into its own package.

## What we might be wrong about

This roadmap reflects current understanding as of 2026-05-19 and
may be revised when:

1. **Profiler data we don't have yet shifts the leverage ranking.**
   VAE decode share, Gemma3 encoder share, memory pressure on
   24 GB card are all hypotheses without measurements. Tier 1
   priorities may re-rank when data lands.
2. **The v0.6 e2e gap closes via consumer-side wrapper changes
   alone.** If `prior_forward` chaining + chunk-size tuning closes
   the gap without persistent-CTA, Tier 1.3 drops to Tier 3.
3. **A new model class with fundamentally different attention or
   FFN patterns becomes the primary workload.** The whole roadmap
   re-anchors. The VISION "What we might be wrong about" #1 covers
   this; the roadmap follows.
4. **Public-facing ambition shifts.** If the user wants to publish
   findings (Tier 2.3), polish + reproducibility + documentation
   costs add up. The structure of the work changes; we don't get
   the same wins on the same budget.

When any of the above fires, edit this doc, record the change in
`internal/log/log_<date>.md`, and revisit `VISION.md` only if the
philosophy itself shifted.
