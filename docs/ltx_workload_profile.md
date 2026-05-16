Last updated: 2026-05-16

# LTX 2.3 FML2V workload profile -- where the wall-time actually lives

Canonical sage-side copy of the production wall-time breakdown for LTX
2.3's FML2V multi-guide workflow. Sourced from a downstream consumer's
in-pipeline A/B measurements + their render-time profiler trace
(2026-05-15). This data is the input for any "what's the next biggest
perf lever" decision on sage's side; cite it instead of guessing at
workload composition.

## Top-level segments

Production FML2V multi-guide render breakdown:

| segment | wall-time | % of total |
|---|---|---|
| Stage-1 (8 sampler steps x ~5s/step) | ~40s | ~27% |
| **Stage-2 (3 sampler steps x ~25s/step)** | **~75s** | **~50%** |
| VAE encode/decode + setup | ~35s | ~23% |
| **Total** | ~150s | 100% |

Stage-2 dominates. Within it, per-step:

| sub-module | per-step share of stage-2 | share of TOTAL render |
|---|---|---|
| **stage-2 attn1 (video self-attention at T=42240)** | ~58% | **~29%** |
| stage-2 ff (FFN at T=42240) | ~24% | ~12% |
| stage-2 attn2 (cross-attention at T=42240) | ~17% | ~9% |

## What this means for sage-side perf work

The single heaviest sub-module across the entire render is **stage-2
attn1 at ~29% of total wall-time**. FFN (v0.6's target) is 12%.
Attention is a 2.4x bigger lever than FFN.

This retroactively explains why v0.6 was always going to be a small
e2e wedge even if it worked perfectly:

- FFN share is 12% of render
- Even a hypothetical 2x FFN matmul kernel can only move ~12% of wall-time
- Best-case e2e wedge from FFN optimization alone: ~5-6%
- v0.6.0's actual delivered: -1.79% (worse than baseline, due to L2
  contention + dispatch overhead on the cache-hostile multi-sampler
  hot loop)

By contrast, a 2x sage attention kernel applied to stage-2 attn1
would move ~29% of wall-time. Best-case e2e wedge: ~15%. That's the
single largest perf lever in the LTX 2.3 stack on sm89.

## Sequencing implications

The leverage calculus implies this order:

1. **Persistent-CTA hybrid for FFN first.** Smaller payoff, but
   validates the persistent-CTA technique at lower risk before
   committing to the larger attention port. 1-2 weeks of work.
2. **Persistent-CTA hybrid for attention second**, once the pattern
   is proven on FFN. 2-3 weeks of work. ~15% e2e ceiling.

CUTLASS / fp16-accum-style matmul throughput work does NOT make the
priority list. The bottleneck on stage-2 isn't matmul throughput;
it's L2 cache locality + dispatch overhead. See
`docs/fp16_accum_fp8_matmul.md` for the analysis of why that path
doesn't help here.

## What's not in this profile (but matters for context)

- **Workflow-level normalization knobs** (e.g. NAG-class
  guidance/normalization that's default-on at production-grade
  strength): disabling or making opt-in can save ~40% wall-time per
  render. Consumer-side workflow decision, not a kernel-side one.
  Not actionable from sage-fork. Worth knowing because it's
  typically the biggest e2e wedge available to a user willing to
  accept the quality change.
- **Stage-2 step count** (currently 3): reducing saves proportional
  wall-time. Lightricks-level decision, not us.
- **AdaLN-Single fusion**: fires as the OOM site in chunk-bypass
  testing. Wall-time share is ~2-3% so kernel fusion here is low
  leverage despite being an obvious fusion target.
- **Single-pass / non-multi-guide LTX workloads** would have a
  different profile -- fewer FFN calls per render, less L2 thrash
  from neighboring attention. v0.6.0's sage_ffn might be net-positive
  on those; we haven't measured. If a single-pass A/B comes in
  positive, the persistent-CTA work prioritization could shift.

## Provenance

- Measurement date: 2026-05-15, in-pipeline A/B (4 renders interleaved
  baseline/treatment/baseline/treatment under fixed-VRAM run conditions).
- Workflow: LTX 2.3 distilled fp8, FML2V multi-guide,
  768x512x97 frames, 4-sec audio, 8-step stage-1 + 3-step stage-2
  refine.
- Hardware/env: 4090, sm89, torch 2.12.0+cu130, triton 3.7.0,
  sageattention v0.6.0 at sage-fork commit `4f8a090` (bias-inclusive).
- Per-call FFN timing details + the full A/B table are in CHANGELOG
  v0.6.0 entry (production result section).

This profile is workload-specific. Other LTX workloads (different
resolution, different step counts, single-pass, audio-only) will have
different breakdowns. The framework in `docs/perf_research_framework.md`
says: measure attention-share-of-CUDA-time on each workload of
interest, apply Amdahl with the per-kernel ratio observed on that
workload's actual call mix. Don't generalize this profile to other
workloads without re-measuring.

## Related

- `docs/perf_research_framework.md` -- the framework for using this
  kind of data (Amdahl with measured attention-share, treat residual
  as hypothesis-needing-its-own-measurement).
- `CHANGELOG.md` v0.6.0 entry -- the production A/B that prompted
  the broader render breakdown.
- `CHANGELOG.md` Backlog -- v0.6.1 / v0.7 candidates derived from
  this profile (persistent-CTA FFN then attention; CUTLASS skipped).
- `docs/fp16_accum_fp8_matmul.md` -- why fp16-accum matmul throughput
  work doesn't help here even though it's the obvious next
  throughput knob.
