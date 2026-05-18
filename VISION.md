last updated: 2026-05-19 (Cell C verdict folded into "what we might be wrong about" #3)

# sage-fork

A 4090 / Ada sm89 kernel optimization repo for **DiT-class local
generation**: LTX 2.3 video, Flux-class image (Flux 2 Klein and
predecessors), Z-Image-Turbo S3-DiT, and other diffusion transformers
we run locally. The kernel base is sage attention (the attention
primitive); v0.6 added `sage_ffn`, an fp8-native fused MLP for
DiT FFN blocks. The bench harness measures these alongside
SpargeAttention, FlashInfer, and torch SDPA at the actual shapes our
models run, on the actual GPU we have (RTX 4090). One number drives
every attention decision; sage_ffn has its own measurement surface.

## How this works

Three things matter:

- **`tests/test_sageattn_ltx_shapes.py`** — the bench harness. Every
  kernel × every shape × (median_ms, mean_rtol). One run prints
  everything side-by-side. ~30s on a warm cache.
- **`sageattention/core.py`** — the dispatcher and per-kernel entry
  points. Where most kernel-side changes land. The `csrc/qattn/*.cu`
  files are the real kernel bodies; `core.py` orchestrates and
  dispatches.
- **`internal/PLAN.md`** — the live operational doc. Backlog,
  experiment log (TSV-style), the research loop. Pairs with this
  file the way [karpathy/autoresearch](https://github.com/karpathy/autoresearch)'s
  `program.md` pairs with its README.

## The metric

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

- **A bench harness** that measures attention kernels at DiT shapes.
  Sage variants (5 modes), SpargeAttention top-k = 0.5,
  FlashInfer fp16 prefill, three torch SDPA backends — every row
  prints every run. The bench output is a one-shot answer to "which
  attention kernel for which shape on this GPU?", not a
  sage-internal characterization.
- **An editable install of sage** with the SM80 build gate widened
  so it actually compiles from source on Ada (woct0rdho's `setup.py`
  refactor accidentally dropped it). Load-bearing only because every
  kernel-side change ships through the editable install.
- **A fused fp8 MLP primitive (`sage_ffn`, v0.6)** for DiT FFN
  blocks. Two-kernel Triton fp8 path
  (`Linear(fp8) -> GELU(tanh) -> Linear(fp8)`) targeting LTX 2.3-
  class FFN. Ships as completeness primitive while the v0.6 e2e gap
  is investigated; the qualitative wedge holds, the quantitative
  one is workload-dependent. Not wired into the `sageattn()`
  dispatcher; consumer imports it directly.
- **A perf-research methodology framework** (`docs/perf_research_framework.md`)
  that codifies the rules every kernel-day decision is graded
  against: load-bearing metric, synthetic-vs-in-pipeline gap,
  evidence ladder for kernel-replacement audits (kernel-name
  presence > per-call logs > attribution coverage > sub-module
  time delta), disprove-test discipline. Reusable across any
  future kernel work on this fork.
- **A decision log** that grades every kernel-side change against the
  metric above. Deferrals carry concrete reopen-numbers, not vague
  "trigger fires."

## What we are NOT

- **A general sage replacement.** Hopper / Blackwell stays upstream.
  We don't validate or optimize for non-Ada.
- **A perf consultancy for individual workloads.** If a model class
  brings a head_dim or sequence pattern outside our coverage, the
  fix is a new bench row, not a workload-specific kernel.
- **A `torch.compile` target.** Verified 2026-04-25 on torch 2.11:
  compile-around-sage produces ~2.8 % rtol drift with no measurable
  speedup. Consumers should keep `torch.compiler.disable()` around
  sage calls. Revisit when a future torch release makes
  [`tests/spike_torch_compile.py`](./tests/spike_torch_compile.py)
  show bounded rtol AND measurable speedup.

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
  — the methodology framework in full, including the evidence
  ladder for kernel-replacement audits.
- [`docs/roadmap.md`](./docs/roadmap.md) — forward-looking record
  of directions worth pursuing, tiered by relevance and trigger-
  conditional. Not a committed schedule; the user remains the
  scheduler.
- [`internal/PLAN.md`](./internal/) (gitignored) — live operational
  doc. Backlog with triggers, experiment log (TSV), the research
  loop. Edit every session.
- [`CHANGELOG.md`](./CHANGELOG.md) — versioned divergence record,
  Known kernel bugs, Decision log.

## What we might be wrong about

The metric and framework reflect the workload mix on this box as of
2026-04-26. Four candid limitations:

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
