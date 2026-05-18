# Roadmap

Last updated: 2026-05-19 (Cell C verdict folded into Tier 1.3 trigger)

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

### 1.1 Workflow profiler tool

**What:** a Python utility that ingests a ComfyUI chrome trace
(plus `record_function` annotations) and emits:

- Per-sub-module kernel-time breakdown (matmul / norm / activation /
  attention / FFN / VAE / etc.)
- Kernel-name presence audit (which Triton / CUDA kernels actually
  fired -- the rung-1 evidence from the evidence ladder)
- Dispatch ladder readout (which sage kernel was chosen per call,
  via correlation with `get_last_dispatched_kernel()`)
- cpu_op.dur vs kernel-time aggregation cross-check (catches the
  47%-offload-trap class of misinterpretation)
- Two-arm comparison helper (BASELINE + TREATMENT side-by-side,
  kernel-name diff, sub-module delta, attribution coverage delta)

**Why it's load-bearing:** every kernel decision on this fork
benefits from sub-module attribution. Downstream consumer side has
been hand-rolling this; this side has been hand-rolling it in
`tests/bench/`. A unified tool means we both stop reinventing AND
the methodology in `perf_research_framework.md` gets a concrete data
layer that any new consumer can adopt.

**Technical shape:** standalone Python package, likely shipped as a
sibling repo `sage-bench-tools/` (or similar) to keep sage-fork
primitive. Reads chrome JSON; uses `pandas` + `orjson` for
aggregation; emits markdown tables + optional plots. Interface
roughly `bench-tools profile <trace.json> [--baseline <trace>]
[--treatment <trace>] [--out report.md]`.

**Effort:** ~1 week for v0.1 covering the five bullet points above.
Ongoing refinement as we use it.

**Trigger to act:** downstream-consumer authorization to adopt v0.1
spec + workload-profiler agreement on interface from cross-clone
discussion. Most relevant prerequisite: their requirements on what
data layer matters most to them, since they're the heaviest
consumer.

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

**Effort:** 2-3 weeks per CHANGELOG estimate. Higher confidence on
FFN first (concrete e2e regression data); attention follows if the
pattern holds.

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

## Tier 4: Explicit non-goals

- **Hopper / Blackwell support.** Out of scope per VISION.md.
  Reopen only if audience shifts.
- **Generic / cross-arch kernel rewrites.** Fights the scope.
- **Becoming an LLM-inference engine.** vLLM / SGLang / others own
  that space. We stay diffusion-focused.
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
Sage-fork stays primitive per VISION. The workflow profiler is the
first candidate to live in a sibling repo. Open to downstream-
consumer input on ergonomics.

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
