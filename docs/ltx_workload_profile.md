Last updated: 2026-05-16

# LTX 2.3 FML2V workload profile -- where the wall-time actually lives

Canonical sage-side copy of the production wall-time breakdown for LTX
2.3's FML2V multi-guide workflow. Sourced from a downstream consumer's
in-pipeline A/B measurements + an extended tracer audit on the same
workflow. This data is the input for any "what's the next biggest perf
lever" decision on sage's side; cite it instead of guessing at workload
composition.

**TL;DR**: Sampler is 81.9% of render. Single heaviest sub-module is
stage-2 attn1 (~25.7% of render). Total FFN is ~16% of render (see
the FFN-share triplet below for the right reading for your question).
VAE decode is ~9.8%. Everything else is small. Persistent-CTA
attention is the biggest available lever; persistent-CTA FFN is
secondary; CUTLASS / fp16-accum throughput chasing is off the
priority list because sage v0.5.5 is already at the sm89 fp8
hardware ceiling.

This is rev 2; rev 1 carried three rough estimates (23% VAE share,
29% stage-2 attn1, ambiguous FFN scope) that were inherited from an
earlier informal breakdown rather than measured. The
"Reconciliation" section near the bottom records the walk-backs.

## Top-level render breakdown

Representative render: 148.27 s total exec-tracked wall-time.

| node-class | s | % of render |
|---|---:|---:|
| `SamplerCustomAdvanced` (stage-1 + stage-2 combined) | 121.49 | 81.9% |
| `LTXVTiledVAEDecode` (video VAE decode) | 14.46 | 9.8% |
| `CLIPTextEncode` (Gemma3 batch pre-encode, once outside loop) | 2.89 | 2.0% |
| `LatentUpsample` chain (between stage-1 and stage-2) | 2.85 | 1.9% |
| `VHS_VideoCombine` (ffmpeg mux) | 2.56 | 1.7% |
| `LTX2_NAG` patch installation (in-sampler NAG cost folded into Sampler) | 1.38 | 0.9% |
| `LTXVAudioVAEDecode` (audio decode tail) | 0.26 | 0.2% |
| other (loaders, preprocess, sigma schedule, ~30 small calls each <0.3s) | 2.38 | 1.6% |

**Sampler dominates at 81.9%; VAE decode is 9.8%; "everything else" is ~8%.** VAE+setup is ~13-15% of render combined, not the 23% earlier informal estimate cited.

## Within-sampler breakdown (121.49 s = 100% of sampler)

The existing tracer covers 75.4% of sampler wall-time. The remaining 24.6% is a residual bucket characterized below.

| label | T | s | % of sampler | sub-module |
|---|---:|---:|---:|---|
| video `attn1` | varies | 41.91 | 34.5% | spatial+temporal self-attention on video tokens |
| video `attn2` | varies | 14.07 | 11.6% | cross-attn from video to Gemma3 prompt |
| video `ff` | varies | 23.24 | 19.1% | FFN (Linear -> GELU -> Linear), `inner_dim = 16384` |
| `audio_attn1` | 100 | 1.80 | 1.5% | audio-stream self-attn |
| `audio_attn2` | 100 | 3.21 | 2.6% | audio-stream cross-attn to prompt |
| `audio_ff` | 100 | 0.78 | 0.6% | audio FFN |
| `video_to_audio_attn` | 100 | 6.60 | 5.4% | one direction of AV cross-attn (audio Q, video KV) |
| **traced subtotal** | | **91.61** | **75.4%** | |
| **residual (untraced)** | | **29.88** | **24.6%** | |

The 24.6% residual contains:

1. **`audio_to_video_attn`** -- the other direction of AV cross-attn (video Q, audio KV). Present in `BasicAVTransformerBlock` but missing from the existing tracer's `SUB_MODULE_NAMES`. Estimated 1-6% of sampler. Tracer extension landed consumer-side, awaiting re-render to populate.
2. **`cross_attention_adaln`** -- AdaLN-Single applied per transformer block (48 blocks * 22 sampler steps * ~7 calls/block per the dataflow audit). Implemented as `apply_cross_attention_adaln(...)` free function over `nn.Parameter` tables, not a hookable Module. Estimated 3-4% of sampler. torch.profile aten-op trace landed consumer-side, awaiting re-render.
3. **RoPE rotation kernels** -- sage ships `fused_rope_split` at ~0.55% share.
4. **Norm layers** -- `norm{1,2,3}` LayerNorm/RMSNorm per block, un-fused.
5. **NAG cross-attn calls within sampler** -- 5-15% of cross-attn cost when active. Unmeasured directly.
6. **Mask construction + token-shape ops + init noise + sigma arithmetic + hook overhead** -- small but non-zero.

## Per-stage / per-stream decomposition

The tracer's T-buckets are stream-bucketed, not stage-bucketed: T=100 is the audio stream (fires every step regardless of stage); T=10780 is video at stage-1 sizes; T=42240 is video at stage-2 sizes.

| stream / stage | s | % of sampler | calls |
|---|---:|---:|---:|
| video stage-1 (T=10780) | 24.60 | 20.2% | 768 ff + 768 attn1 + 768 attn2 |
| video stage-2 (T=42240) | 54.62 | 45.0% | 288 ff + 288 attn1 + 288 attn2 |
| audio + v2a-attn (T=100, all stages) | 12.39 | 10.2% | 1056 calls each of audio_*, v2a |
| within-sampler residual | 29.88 | 24.6% | unattributed |

**Stage-2 attn1 alone = 25.7% of total render** (31.18 s / 121.49 s sampler share x 81.9% sampler-of-render). The single heaviest sub-module across the whole render -- confirms the "biggest lever" framing.

Stage-2 attn1 + stage-2 ff together = 37.6% of total render. That's the combined target for stage-2 persistent-CTA work.

## Dynamic VRAM offload dominates the sampler under production conditions

A subsequent torch.profiler chrome-trace audit (2026-05-16, fingerprint-on so absolute timing is inflated ~1.5-2x vs the table above but the proportional split is meaningful) revealed that ComfyUI's dynamic VRAM offload (comfy-aimdo / `--reserve-vram 0.5`) accounts for ~47% of sampler wall-time on this workload:

| aten op | count | total ms | what it is |
|---|---:|---:|---|
| `aten::copy_` | 39,989 | ~35,300 | weight shuttle CPU<->GPU |
| `aten::to` | 65,171 | ~31,700 | dtype/device conversion |
| `aten::linear` | 25,976 | ~1,700 | qkv + out_proj + ff Linears (the work) |

**~47% of sampler wall-time is PCIe weight-shuttle, NOT GPU compute.**

This is a critical reframing for kernel-side perf decisions:

- Concurrent-dispatch parallelism overlaps GPU compute with GPU compute; it cannot overlap with `aten::copy_` (PCIe-bandwidth-bound, not on the GPU compute engine). The 47% offload pool is outside sage's reach.
- Persistent-CTA-class kernel work targets L2 contention within GPU compute. The 47% offload pool is unaffected by L2 work either.
- The "biggest single lever" reshuffles. Stage-2 attn1 at ~25.7% of total render is still the largest single GPU-compute lever, but it competes for engineering attention against the larger offload-share that no kernel work touches.

A `--reserve-vram 0`-style follow-up audit (less VRAM pressure -> less offload) is pending; would tell us whether this 47% is structural to LTX 2.3's working set on 24 GiB or specific to the production VRAM-reserve setting. Until that runs, treat the offload share as audit-derived under one VRAM-reserve config, not as a fixed property of the workload.

## What this means for sage-side perf work

Lever ranking by structural prize. **Wedges below are bounded by Amdahl against the ~53% of sampler that is GPU compute under production VRAM-pressure conditions (the other ~47% is offload that no kernel-side work touches).** Numbers cited as `% of render` already include the offload-Amdahl ceiling implicitly.

- **Stage-2 attn1**: 25.7% of total render. Persistent-CTA-class kernel work targeting L2 contention can plausibly move 20-40% of this -> ~5-10% e2e wall-time. Single largest GPU-compute lever.
- **Stage-2 ff**: 9.8% of total render (the third FFN-share-triplet reading). Persistent-CTA on FFN tested in v0.6.0 ran +20% slower per-call due to L2 contention; persistent-CTA done right could plausibly recover that AND improve, ~3-5% e2e.
- **Audio + v2a stream**: 8.4% of total render currently observable, expected slightly larger with full Window-B (V2A || A2V) accounted for. Concurrent-dispatch parallelism could in principle overlap most of this with the video path's wall-time -- mechanism confirmed by 3-run path-B sub-module spike (audio fully absorbed into video bulk work, video slowdown -0.6% to -1.7% under concurrent issue). Prize ceiling revised down to 3-7% e2e under VRAM-pressure conditions because the offload pool reduces overlappable surface; could be higher under nodynvram conditions pending audit.
- **VAE decode**: 9.8% of render. Kernel-side surface unknown; chrome-trace audit pending. If GPU-bound + sm89-tunable, real lever. If memory-bound, rules it out from kernel-side work.
- **AdaLN-Single**: ~3-4% of sampler / ~3% of render. Low individual leverage. Composes with RoPE + norms (other small fusion targets) potentially.
- **Workflow-level VRAM-reserve tuning**: NOT a sage-side lever, but worth noting -- if a user can run with less aggressive VRAM-reserve (`--reserve-vram 0` or model-offload-disabled), the 47% offload pool shrinks and EVERY kernel-side optimization gets correspondingly more leverage on the new (mostly-GPU-compute) sampler total. Bigger impact than any single kernel-side intervention.

CUTLASS / fp16-accum-style matmul throughput work does NOT make the priority list. The bottleneck on stage-2 isn't matmul throughput; it's L2 cache locality + dispatch overhead under VRAM-pressure conditions. See `docs/fp16_accum_fp8_matmul.md`.

## Sequencing implications

The leverage calculus implies this order if appetite exists for kernel-engineering work:

1. **Persistent-CTA hybrid for FFN first.** Smaller payoff (~3-5% e2e), but validates the persistent-CTA technique at lower risk before committing to the larger attention port. 1-2 weeks of work.
2. **Persistent-CTA hybrid for attention second**, once the pattern is proven on FFN. 2-3 weeks. ~5-10% e2e ceiling.
3. **Concurrent-dispatch parallelism spike** (independent axis from persistent-CTA -- attacks dispatch-layer scheduling rather than per-kernel cache locality). Half-day gating spike before any infrastructure commitment.

Lower-leverage candidates (AdaLN/RoPE/norm fusion, VAE kernel work pending chrome-trace audit) stay below the line until the higher-leverage levers either ship or rule out.

## What's not in this profile (but matters for context)

- **Workflow-level normalization knobs** (e.g. NAG-class guidance/normalization that's default-on at production-grade strength): disabling or making opt-in can save ~40% wall-time per render on workloads where the knob is the dominant compute. Consumer-side workflow decision, not a kernel-side one. Not actionable from sage-fork. Worth knowing because it's typically the biggest e2e wedge available to a user willing to accept the quality change.
- **Stage-2 step count** (currently 3 on the measured workload): reducing saves proportional wall-time. Model-side decision, not ours.
- **AdaLN-Single fusion as an OOM target**: AdaLN-Single fires as the OOM site in some chunk-bypass testing configurations. Wall-time share is low (~3%); fusion here would be a correctness / memory-pressure intervention more than a perf intervention.
- **Single-pass / non-multi-guide LTX workloads** would have a different profile -- fewer FFN calls per render, less L2 thrash from neighboring attention. v0.6.0's sage_ffn might be net-positive on those; we haven't measured. If a single-pass A/B comes in positive, the persistent-CTA work prioritization could shift.

## Reconciliation against rev 1

Three numbers in the rev 1 doc were rough estimates from an earlier informal breakdown; rev 2 walks them back to tracer-grounded values:

| rev 1 framing | rev 2 (tracer-grounded) | reconciliation |
|---|---|---|
| "VAE encode/decode + setup ~23% of render" | **VAE decode 9.8% + non-sampler residual ~6% = 13-15% of render** | rev 1 was a rough estimate, not a measurement. Rev 2 is from node-time tracking on a specific render. |
| "Stage-2 attn1 ~29% of total render" | **25.7% of render** | small variance across workloads; the rev 1 number may have been from a different audio length or step-count config. |
| "Stage-2 FFN ~12% of total render" | See "FFN-share triplet" below -- rev 1 framing was ambiguous about scope. |
| "Stage-2 sub-modules 58%/24%/17% of stage-2" | 57.1%/26.7%/16.2% | matches within rounding. |
| "AdaLN-Single ~2-3% of render" | **~3-4% of sampler / ~3% of render** | needs torch.profile re-render to confirm; expected to be close. |

## FFN-share triplet -- three distinct readings, name them explicitly

"FFN share" can mean any of three different numbers depending on what decision it's an input to. Going forward, cite the one that matches your question:

| FFN reading | % of render | use case |
|---|---|---|
| **Total FFN (video_ff + audio_ff, all stages)** | **~16%** | "Lever sizing" -- the right number when ranking persistent-CTA-on-FFN against other interventions like persistent-CTA-on-attention or concurrent-dispatch parallelism. |
| Video FFN only, all stages | 15.7% | Closest analog to "what would sage_ffn touch if dispatched across stages 1+2." Audio_ff is small enough (~0.6% of sampler) that the difference vs total is ~0.3pp. |
| Stage-2 video FFN only | 9.8% | "Amdahl ceiling input" for stage-2-specific analysis. The right number for "what's the max e2e wedge from a stage-2-only kernel optimization." |

The v0.6.0 walk-back analysis used the third reading (stage-2-only) implicitly; future docs should pick one explicitly.

## Provenance

- v0.6.0 day-9 A/B measurement: 2026-05-15, 4 renders interleaved baseline/treatment/baseline/treatment under fixed-VRAM run conditions. Wall-time deltas + per-call timing in CHANGELOG v0.6.0.
- Extended tracer audit: 2026-05-16, same workflow, one representative render from a set whose splits are shape-consistent within +/-2%.
- Workflow: LTX 2.3 distilled fp8, FML2V multi-guide, 768x512x97 frames, 4-sec audio, 8-step stage-1 + 3-step stage-2 refine.
- Hardware/env: 4090, sm89, torch 2.12.0+cu130, triton 3.7.0, sageattention v0.6.0 at sage-fork commit `4f8a090` (bias-inclusive).

This profile is workload-specific. Other LTX workloads (different resolution, different step counts, single-pass, audio-only) will have different breakdowns. The framework in `docs/perf_research_framework.md` says: measure attention-share-of-CUDA-time on each workload of interest, apply Amdahl with the per-kernel ratio observed on that workload's actual call mix. Don't generalize this profile to other workloads without re-measuring.

## Pending audits

Two extensions to the existing tracer infrastructure are landed but not yet re-rendered:

- `audio_to_video_attn` coverage -- closes the 5-7% sampler undercount in audio-side wall-time. Required for tight sizing of concurrent-dispatch parallelism prize.
- AdaLN/RoPE/norm aten-op trace via `torch.profile` -- characterizes the 24.6% sampler residual. Required for assessing whether AdaLN dominates the residual (gates parallelism payoff) or norms / NAG / hook overhead do.

Both will populate on the next render of the FML2V benchmark workflow with the appropriate env vars set. Expected to refine the numbers above but not change the lever ranking.

## Related

- `docs/perf_research_framework.md` -- the framework for using this kind of data (Amdahl with measured attention-share, treat residual as hypothesis-needing-its-own-measurement).
- `CHANGELOG.md` v0.6.0 entry -- the production A/B that prompted this profile.
- `CHANGELOG.md` Backlog -- v0.6.1 / v0.7 candidates derived from this profile (persistent-CTA FFN then attention; concurrent-dispatch spike open; CUTLASS skipped).
- `docs/fp16_accum_fp8_matmul.md` -- why fp16-accum matmul throughput work doesn't help here.
- `docs/cuda_mask_kernel_scoping.md` -- precedent for the kernel-day-scoping discipline applied to v0.5.5; the same discipline applies to any v0.6.1+ persistent-CTA work.
