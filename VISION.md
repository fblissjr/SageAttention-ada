last updated: 2026-04-26

# sage-fork

A 4090 / Ada attention-kernel optimization repo for **DiT-class local
generation**: LTX 2.3 video, Flux-class image (Flux 2 Klein and
predecessors), Z-Image-Turbo S3-DiT, and other diffusion transformers
we run locally. The kernel base is sage attention; the bench harness
measures it alongside SpargeAttention, FlashInfer, and torch SDPA at
the actual shapes our models run, on the actual GPU we have (RTX
4090). One number drives every decision.

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
  kernel — days of work for at most ~2× speedup at sub-ms cross-attn
  — stays deferred, with the K = 1.68 measurement supporting the
  deferral.
- **Measurement before decision.** The
  `cross_attn_unmasked_kv226_kratio_probe` row exists specifically
  so the K-ratio is readable every bench run; without it, the
  trigger could never fire.
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
   end-to-end if attention is already < 50 % of step time. We don't
   measure end-to-end here. A "this saved 5 ms per call" claim
   should be paired with "and we observed a real LTX gen go from X
   seconds to Y" before ranking high.
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
