last updated: 2026-05-23 (library reframe: a sm89 kernel library for ComfyUI consumer workloads, consumed by a personal repo ecosystem; sage attention is one module in it)

# sage-fork

*A trusted sm89 build, a measurement rig, and a place to test ideas.*

Three things, in order of how much they have actually delivered:

**1. A trusted fork.** The build that compiles for sm89 and stays correct
on the shapes we run. Concretely: `setup.py:152` (without which sage does
not build for Ada from source at all), the int32 offset overflow fix that
keeps 362-frame renders from silently zeroing their tail, the stream-safety
fix, the dispatcher mask-routing fix, and `attn_mask` as an introspectable
parameter. Upstream defects, mostly -- found here, fixed here, and several
of them worth sending back.

**2. A measurement rig.** The LTX and H3 bench surfaces, the e2e harness,
the workload profiler, and `docs/perf_research_framework.md`. This is what
produces the results people can act on -- that upstream's Sol-Attn defaults
are wrong past 300 frames, that a 32B text encoder costs 1.5s, that H3's
area cap makes 1:1 three times cheaper than 16:9, that `smooth_k` is a
wash. None of it required writing a fast kernel.

**3. A place to test ideas.** `sage_ffn`, `sageattn_partitioned`,
`fused_rope_split`, the torch.compile spike. Kept because negative results
are worth keeping, not because they shipped wins.

**What this repo has never done is make anything faster.** Every speed
number in these docs is upstream's kernel -- the sm89 INT8-QK / FP8-PV
design from `thu-ml` via `woct0rdho`, shipped here unmodified. The version
titles record the pattern honestly: v0.6.0 `sage_ffn` "not currently a perf
win in production", v0.5.4 `sageattn_partitioned` "honest negative result",
v0.3.0 "correctness fix", v0.6.6 "build robustness". Even the Triton
autotune addition only confirmed the existing hardcoded config was already
optimal. Not one release has shipped a measured speedup from code written
here.

That is not a failure to fix by trying harder at kernels. It is what the
evidence says this repo is good at, and the framing should match: **prove
things, fix things, and be the build you trust** -- for whatever workload
is in front of us. Nothing here is limited to sage attention or to
attention at all; the scope is whatever a ComfyUI render on sm89 actually
spends time on, and the deliverable is usually a measurement or a fix
rather than a kernel.

If a real kernel win ever does land, it should arrive with an in-pipeline
A/B attached, and this section should be rewritten to say so.

**The hard constraint: sm89 / RTX 4090 only.** Hopper / Blackwell /
Ampere stay out of scope. Everything else -- kernel surface
(attention, FFN, VAE, cross-modal, anything else that shows up in a
ComfyUI render hot loop), abstraction layer (raw CUDA, Triton,
CUTLASS-pattern, or modern Triton features), third-party leverage
(torchao primitives, modern PyTorch fp8 APIs, sub-pattern references
from Megakernels / ThunderKittens etc.) -- is on the table.

## How this works

The work surface, by layer:

- **Kernels** -- `sageattention/` (attention) + `sageattention/triton/`
  (sage_ffn, fused_rope_split). New kernel work lands here on sm89-
  bounded scope: CUDA + Triton, fp8 / int8 quantized where the
  workload supports it, mature `mma.sync` + `cp.async` primitives
  (no TMA / WGMMA / TMEM).
- **Bench + measurement** -- `tests/test_sageattn_ltx_shapes.py`
  (attention shapes), `tests/bench_sage_ffn_shapes.py` (fp8 MLP
  shapes vs torch comparand), `tests/test_dispatcher_routing_log.py`
  (routing observability). New primitives add their own measurement
  surface; the methodology framework codifies how we trade across them.
- **ComfyUI integration patterns** --
  `sageattention/comfyui_compat.py` (fp8 storage probe across known
  conventions), the cross-clone memo protocol with the audio-loop
  consumer-side claude, the wrapper-discipline rules in the
  perf-research framework (rung-2 silent-fallback-pattern enumeration).
- **Methodology + decision discipline** --
  `docs/perf_research_framework.md` codifies the rules every kernel-
  day decision is graded against: load-bearing metric, synthetic-vs-
  in-pipeline 2x2 matrix (Cell A/B/C/D vocabulary), evidence ladder
  for kernel-replacement audits, rung-2 silent-fallback-pattern
  enumeration, disprove-test discipline.
- **Forward record** -- `docs/roadmap.md` (tiered directions, trigger-
  conditional, including stack-leverage opportunities and external
  design references), `CHANGELOG.md` (versioned divergence + Decision
  log + Backlog + Workload intel).

## The metric

The load-bearing measurement for sage attention work is unchanged --
attention dominates wall time on the canonical workload class
(DiT-class video / image gen) and the kernel that ships is the one
the dispatcher actually picks. New primitives (sage_ffn, future VAE
fusion, future cross-modal attention coverage) each carry their own
measurement surface; the methodology framework codifies how we trade
across them and which decisions are gated on which evidence.

```
tests/test_sageattn_ltx_shapes.py
  shape: ltx23_video_self_attn_init_22932  (B=1, H=32, Sq=Skv=22932, D=128, no mask, bf16)
  mode:  fp8_cuda++
  -> primary perf metric: median_ms — lower is better (today: 20.20 ms)
  -> accuracy guard:      mean_rtol ≤ 0.10 (today: ~0.098)
  -> cross-session normalizer: torch_flash / sage_fp8++ ratio (today: 2.66×)
```

**What does good look like?** `median_ms` goes down. `mean_rtol`
stays ≤ 0.10. The bench's other rows (image-gen shapes, cross-attn
kv sweep, cross-kernel rtol consistency, the dispatcher telemetry
test) don't regress.

**How do we know if we're regressing vs progressing?**

- *Within a session:* the same row, the same number. A median_ms
  drop ≥ 5 % is a real win; smaller is run-to-run noise.
- *Across sessions* (after a torch / triton / CUDA / driver bump):
  the speedup ratio `torch_flash / sage_fp8++`. Driver-thermal
  variance drifts absolute time 1–2 % even with no code change; the
  ratio doesn't move.
- *Across kernels:* side-effect checks. A change that helps fp8++
  but hurts fp16_triton means you shifted a knob that's shared
  between code paths — either intentional or a foot-gun.

The keep/discard rule mirrors autoresearch's: median_ms improved AND
rtol stayed under 0.10 → keep, ship. Anything else → discard, revert.

## Why this metric, why this row, why this hardware

**The shape (LTX 2.3 video self-attn at production seq).** On LTX 2.3
video gen, video self-attn accounts for the overwhelming majority of
attention cost per sampling step (~76% of total attention wall-time
per a real consumer trace; see CHANGELOG v0.4.1). Per gen, ~25–50
sampling steps × this row = the real wall-clock the user feels.
Cross-attn (kv ≤ 1024) is sub-millisecond per call; image-gen shapes
(Flux head_dim = 128, Z-Image head_dim = 120) are 1–2 ms. Production
seq is 22932 (init render) or 23296 (loop iter); the LTX 2.3 video
path uses `attention_head_dim=128` (`transformer_ltx2.py:907-947`),
not the d=64 the audio path uses.

**The kernel (`fp8_cuda++`).** That's what `sageattn()` picks on sm89
+ CUDA ≥ 12.8 unmasked, after the v0.3.0 dispatcher mask-routing
fix. It's the kernel that actually runs in production. Optimizing a
kernel the dispatcher doesn't pick is research that doesn't ship.

**The hardware (RTX 4090 / sm89 / Ada).** It's the GPU we own. Sage
also runs on Hopper / Blackwell — those stay with upstream. We
compile and run on Ampere too (the SM80 kernel is forward-compatible
to sm86/87/89), but we don't validate. The bench shapes, the rtol
baselines, and the kernel decisions are all sm89-tied.

## What `rtol` means here, and why 0.10 is the line

`rtol` — relative tolerance — is element-wise
`|actual − expected| / max(|actual|, |expected|)`, then averaged
over every element of the attention output. The reference
(`expected`) is torch SDPA's `EFFICIENT_ATTENTION` backend at the
same shape and dtype (bf16). That backend is close enough to ground
truth here — its numerical difference vs the `MATH` backend is
orders of magnitude smaller than sage's quantization error.

**Why this matters for DiT generation specifically:** the attention
output drives diffusion sampling. Each sampling step takes a
velocity / noise prediction and integrates it into the next latent;
per-step errors compound across 25–50 steps. Mean rtol ≤ 0.10 is
empirically the level at which individual frames stay visually
indistinguishable from an SDPA-reference render at full sampling
length on LTX / Flux / Z-Image-class models. Above 0.10,
frame-level artifacts (smearing, discoloration, geometry drift on
small features) start showing.

Concretely, what the kernels can hit:

- `fp8_cuda++` today: ~0.097 — under the ceiling, close to it.
  Further fp8++ optimization can't push rtol below ~0.04 without
  changing the quantization format. FP8 has an information floor.
- `fp16_cuda` / `fp16_triton`: ~0.04 — comfortably below the
  ceiling, slower, and not what the dispatcher picks unmasked on
  sm89. They mark the noise floor of "what attention numerics can
  do here at this dtype."

**Caveat (also in "What we might be wrong about" below):** mean rtol
is a proxy for perceptual quality, not the truth. We don't run
PSNR / SSIM / LPIPS in this repo — that's downstream-consumer work.
If a kernel change ever passes the rtol guard but causes a visible
regression in a real render, the rtol guard isn't the right floor
and we add a perceptual layer.

## What we ARE

- **A kernel library for ComfyUI sm89 workloads.** A coherent set of
  primitives -- attention (sage's founding module) + `sage_ffn` (v0.6
  fp8 MLP) + `fused_rope_split` (the fp8/int8 quant lives inside these
  kernels, not as separately-importable ops) -- with a documented
  import surface (the de-facto public symbols in
  `docs/downstream_symbols.md`) that its sibling consumer node pins
  (today: the audio-loop node). Forward: VAE fp8 fusion, cross-modal
  attention coverage, GeGLU sage_ffn extension, persistent-CTA
  rewrites, and whatever the measurement says next. All sm89-bounded.
  The repo name is "sage-fork" for historical reasons; sage attention
  is one module, not the whole library.
- **A bench harness** that measures attention kernels at DiT shapes.
  Sage variants (5 modes), SpargeAttention, FlashInfer, three torch
  SDPA backends — every row prints every run. Expanding to new
  workload classes (cross-modal, multi-modal pipelines) as those
  become recurrent.
- **An editable install** with the SM80 build gate widened to
  compile from source on Ada. Load-bearing because every kernel-
  side change ships through it.
- **A ComfyUI integration surface.**
  `sageattention.extract_fp8_weight_and_scale` (v0.6.4) shims the
  four known fp8 storage conventions; the cross-clone memo protocol
  coordinates with the audio-loop consumer-side claude; the
  wrapper-discipline rules in the perf-research framework (5-pattern
  silent-fallback enumeration) codify what every kernel-replacement
  consumer node needs to handle.
- **A perf-research methodology framework**
  (`docs/perf_research_framework.md`) that codifies the rules every
  kernel-day decision is graded against: load-bearing metric, the
  synthetic-vs-in-pipeline 2x2 matrix (Cell A/B/C/D), evidence
  ladder for kernel-replacement audits, rung-2 silent-fallback-pattern
  enumeration, disprove-test discipline. Reusable across any future
  kernel work on this fork or its consumers.
- **A decision log + tiered roadmap** that grades every change
  against the metric, with explicit triggers for promoting items
  from "candidate" to "active backlog."

## What we are NOT

- **A general sage replacement.** Hopper / Blackwell stay upstream.
  We don't validate or optimize for non-Ada.
- **An LLM-inference engine.** vLLM / SGLang / others own that
  space. We stay diffusion-and-multi-modal-diffusion focused. As DiT
  and LLM worlds converge in practice (audio-conditioned models,
  text-conditioned video, multimodal pipelines), shared kernel
  surfaces (attention, FFN, normalization) are in scope on sm89; we
  just don't ship an autoregressive serving stack.
- **A polished public release.** A library for a *personal* repo
  ecosystem, not a published PyPI package. The "consumable boundary"
  is for the maintainer's own sibling consumer node (the audio-loop
  node today),
  not external users -- solo-hobbyist scope. README + CHANGELOG +
  roadmap + perf-research framework are sufficient. Revisit only if
  audience shifts.
- **A perf consultancy for individual workloads.** If a model class
  brings a head_dim or sequence pattern outside our coverage, the
  fix is a new bench row + a methodology cycle, not a workload-
  specific kernel.
- **A `torch.compile` target.** Verified 2026-04-25 on torch 2.11:
  compile-around-sage produces ~2.8 % rtol drift with no measurable
  speedup. Revisit when a future torch release makes the spike show
  bounded rtol AND measurable speedup.

## Design choices

- **Primitive over policy.** The consumer routes; the fork measures.
  v0.3.0's dispatcher mask-routing fix is the limit case: it was
  correctness, not policy, so it landed here.
- **Correctness before perf.** v0.3.0's silent-mask-drop took a
  10-line dispatcher fix and a regression test. The native CUDA mask
  kernel on sm89 fp8++ landed v0.5.5 — not because the K-ratio
  trigger (last measured 1.57) crossed 5×, but because the
  structural-correctness trigger fired: the masked Triton fallback
  isn't a free correctness substitute, it's a real memory footprint
  that pushes 24 GiB LTX renders over the edge. Preliminary
  in-pipeline A/B (CHANGELOG v0.5.5) shows the Triton fallback
  OOM'ing where fp8_cuda++ fits. sm80 + other sm89 variants still
  deferred.
- **Measurement before decision.** Triggers fire on measurement, not
  speculation. The K-ratio probe row gates perf-based action; the
  structural-correctness clause gates routing-based action. Both
  are readable from artifacts (bench output / Backlog signal log).
- **Simplicity criterion** (cribbed from autoresearch's
  `program.md`): all else being equal, simpler is better. A small
  median_ms gain that adds ugly complexity isn't worth it. Removing
  code and getting equal-or-better results is a great outcome.
- **Stack leverage > reinvention.** Use modern torchao / Triton /
  PyTorch features when they unlock something rather than rebuilding
  from raw CUDA. Cross-arch design references (Megakernels'
  persistent-CTA shape, ThunderKittens' producer/consumer split) are
  read for sub-patterns; their code is not ported. See
  `docs/roadmap.md` "Stack leverage opportunities" + "External
  design references."
- **Cross-clone coordination discipline.** Tight memo protocol with
  the audio-loop consumer-side claude: durable-surface-first for
  substantive numbers (commit to CHANGELOG / docs before the memo
  references them), `supersedes: <timestamp>` prefix on retractions,
  check the outbox-mirror directly when the inbox looks stale.
- **Honest about what's V1.** See "What we might be wrong about."

## Where to go next

- [`README.md`](./README.md) — what changed vs the upstream codebase,
  what was measured, signatures and caveats for each public entry
  point.
- [`CLAUDE.md`](./CLAUDE.md) "Performance research: the load-bearing
  metric" — the full perf-research framework: side-effect checks,
  next-experiment patterns, what we ignore and the trigger that
  would change that.
- [`docs/perf_research_framework.md`](./docs/perf_research_framework.md)
  — the methodology framework in full: synthetic-vs-in-pipeline 2x2
  matrix (Cell A/B/C/D), evidence ladder for kernel-replacement
  audits, rung-2 silent-fallback-pattern enumeration.
- [`docs/roadmap.md`](./docs/roadmap.md) — forward-looking record
  of directions worth pursuing, tiered by relevance and trigger-
  conditional. Includes Stack leverage opportunities (torchao,
  Triton 3.7, PyTorch 2.12) + External design references
  (Megakernels, ThunderKittens) sections. Not a committed schedule;
  the user remains the scheduler.
- [`CHANGELOG.md`](./CHANGELOG.md) — versioned divergence record,
  Known kernel bugs, Decision log, Backlog with triggers, Workload
  intel. Single source of truth for both open triggers and closed
  decisions; the `internal/PLAN.md` that used to mirror it was
  retired 2026-08-05 after drifting out of date against it.

## What we might be wrong about

The metric and framework reflect the workload mix on this box as of
2026-05-19. Four candid limitations:

1. **The "LTX self-attn dominates" assumption is workload-specific.**
   If a new model class with fundamentally different attention
   patterns (very-short autoregressive seq, sliding-window, MQA / GQA
   with very different head ratios) becomes the primary use case,
   the load-bearing shape moves and the metric should be re-derived.
   Disconfirming signal: a downstream-consumer telemetry summary
   showing a non-LTX-class shape consuming > 30 % of gen attention
   time.
2. **Mean rtol is a proxy, not the truth.** See "What rtol means
   here" above for the definition and the 0.10 ceiling.
   Disconfirming signal: a kernel change passes the rtol guard but
   triggers a consumer-reported visual regression — that means the
   rtol guard isn't the right floor and we add a perceptual layer
   (PSNR / SSIM / LPIPS).
3. **Kernel ms is not gen ms.** A 2× kernel speedup is invisible
   end-to-end if attention is already < 50 % of step time.
   **Status: confirmed (with two refinements).**

   *v0.5.1 first e2e measurement* on the canonical LTX 2.3 audio-
   loop workload (832×480×497 / 25fps / 8-step distilled): sage's
   2.66× kernel-row speedup translates to **1.22× end-to-end**,
   with attention at 8.2% of wall. Pure-attention Amdahl predicts
   ~1.05×; observed 1.22× is +17 points higher because sage's reach
   extends beyond the per-call attention rows into FFN-adjacent
   amortization within the sampler step.

   *v0.6 sage_ffn e2e walk-back* on a two-sampler LTX FML2V
   workflow (CHANGELOG v0.6.0): synthetic kernel-bench projected
   1.26-1.36× vs torch fp8-dequant reference, but the in-pipeline
   A/B came back **+1.79% e2e SLOWER** (+20% per-call at stage-2).
   Root cause was L2 cache contention with neighboring attention
   modules + cumulative kernel-launch overhead at LTX's ~1000-FFN-
   calls/render count. **This is the cost of running synthetic-
   first / in-pipeline-validate-later** for kernel work with
   structural risk that synthetic bench specifically can't measure
   (L2 contention, dispatch overhead, fragmentation, sustained
   thermal). Codified as the discipline rule in CLAUDE.md
   "Gate ship-decisions on in-pipeline A/B when synthetic-bench
   can't measure the dominant cost." Going forward, kernel-day
   work with this risk shape gates the v0.X ship commit on in-
   pipeline A/B BEFORE the commit lands, not after.

   *v0.6 Cell C verdict at the per-kernel level (2026-05-19).* With
   the consumer-side integration chain fully closed (six bugs across
   two A/B cycles) and sage_ffn dispatching end-to-end, the v0.6
   synthetic-vs-production gap was measured at the per-stage kernel
   boundary: sage_ffn is 22% slower at stage-1 (T=10780) and 5%
   slower at stage-2 (T=42240) vs production stock fp8, despite
   synthetic isolation showing 1.39x / 1.60x sage advantage at the
   same shapes. **Production has the sign flipped.** The gap is not
   framework overhead -- it sits at the kernel boundary itself. Two
   open hypotheses (CHANGELOG Decision log): stock comparand
   identity (synthetic vs `torch._scaled_mm`, production vs
   `comfy.ops.fp8_linear`), and sage autotune state under
   interleaved dispatch. Neither is a sage correctness bug; the
   kernel works as designed and the bench just isn't measuring the
   production-relevant thing.

   Concrete answer at the VISION level: kernel work IS justified
   per v0.5.1; the simplification "kernel ms = gen ms" isn't
   literally true; non-attention bottlenecks (VAE decode, caching,
   scheduler overhead) are where the next round of e2e wins routes;
   synthetic-bench projections need in-pipeline validation before
   being claimed as e2e wins, especially for per-call-heavy
   primitives; and the in-pipeline validation needs to verify the
   *comparand* is what production actually runs, not just an
   isolated reference. v0.6 Cell C exposed comparand-identity as
   the hidden assumption synthetic bench glosses over.
4. **The next-experiment framework is V1.** It codifies a strategy;
   the strategy hasn't been validated by running through it on a
   real perf change yet. The first time we use it to pick a
   direction and either succeed or fail, the framework gets
   refined. Treat the five patterns in `CLAUDE.md` as starting
   hypotheses, not playbook.

When any of the above fires (disconfirming signal observed), update
[`CLAUDE.md`](./CLAUDE.md) "Performance research" / "What we might
be wrong about," record the change in the session log, and revisit
this VISION.md if the philosophy shifted.
