Last updated: 2026-05-16 (rev 2: replaced rough breakdown estimates with tracer-grounded numbers from the v2 audit; walked back 23% VAE share, 29% stage-2 attn1 share, FFN-share ambiguity)

# LTX 2.3 FML2V workload profile -- where the wall-time actually lives

Canonical sage-side copy of the production wall-time breakdown for LTX
2.3's FML2V multi-guide workflow. Sourced from a downstream consumer's
in-pipeline A/B measurements + an extended tracer audit (2026-05-15
A/B + 2026-05-16 v2 tracer audit on the same workflow). This data is
the input for any "what's the next biggest perf lever" decision on
sage's side; cite it instead of guessing at workload composition.

The v2 audit (2026-05-16) replaces the earlier rough breakdown that
shipped in rev 1. Three numbers tightened materially; see the
"Reconciliation against rev 1" section below.

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

1. **`audio_to_video_attn`** -- the other direction of AV cross-attn (video Q, audio KV). Present in `BasicAVTransformerBlock` but missing from the v2 tracer's `SUB_MODULE_NAMES`. Estimated 1-6% of sampler; **closed in tracer commit `9575a0f`** on the consumer side, awaiting re-render to populate.
2. **`cross_attention_adaln`** -- AdaLN-Single applied per transformer block (48 blocks * 22 sampler steps * ~7 calls/block per the dataflow audit). Implemented as `apply_cross_attention_adaln(...)` free function over `nn.Parameter` tables, not a hookable Module. Estimated 3-4% of sampler. **Closed via torch.profile extension in tracer commit `9de86a3`**, awaiting re-render.
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

## What this means for sage-side perf work

Lever ranking by structural prize:

- **Stage-2 attn1**: 25.7% of total render. Persistent-CTA-class kernel work targeting L2 contention can plausibly move 20-40% of this -> ~5-10% e2e wall-time. Single largest lever.
- **Stage-2 ff**: 12.0% of total render (sampler share * sampler-of-render). Persistent-CTA on FFN tested in v0.6.0 ran +20% slower per-call due to L2 contention; persistent-CTA done right could plausibly recover that AND improve, ~3-5% e2e.
- **Audio + v2a stream (with audio_to_video_attn included once that tracer extension re-renders)**: 10.2% of sampler observable today; expected 13-17% with the missing direction. Concurrent-dispatch parallelism could in principle overlap most of this with the video path's wall-time -- 2-5% e2e if the mechanism works on the 4090. Spike pending.
- **VAE decode**: 9.8% of render. Kernel-side surface unknown; chrome-trace audit pending. If GPU-bound + sm89-tunable, real lever. If memory-bound, rules it out from kernel-side work.
- **AdaLN-Single**: ~3-4% of sampler / ~3% of render. Low individual leverage. Composes with RoPE + norms (other small fusion targets) potentially.

CUTLASS / fp16-accum-style matmul throughput work does NOT make the priority list. The bottleneck on stage-2 isn't matmul throughput; it's L2 cache locality + dispatch overhead. See `docs/fp16_accum_fp8_matmul.md`.

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

The v0.6.0 walk-back analysis used the third reading (stage-2-only) implicitly; future docs should pick one explicitly to stop the next cross-clone exchange from re-deriving.

## Provenance

- v0.6.0 day-9 A/B measurement: 2026-05-15, 4 renders interleaved baseline/treatment/baseline/treatment under fixed-VRAM run conditions. Wall-time deltas + per-call timing in CHANGELOG v0.6.0.
- v2 tracer audit: 2026-05-16, same workflow, representative render `b71c69d0-9180-438a-b839-366b19a27353`. Other renders in the same set show shape-consistent splits within +/-2%.
- Workflow: LTX 2.3 distilled fp8, FML2V multi-guide, 768x512x97 frames, 4-sec audio, 8-step stage-1 + 3-step stage-2 refine.
- Hardware/env: 4090, sm89, torch 2.12.0+cu130, triton 3.7.0, sageattention v0.6.0 at sage-fork commit `4f8a090` (bias-inclusive).

This profile is workload-specific. Other LTX workloads (different resolution, different step counts, single-pass, audio-only) will have different breakdowns. The framework in `docs/perf_research_framework.md` says: measure attention-share-of-CUDA-time on each workload of interest, apply Amdahl with the per-kernel ratio observed on that workload's actual call mix. Don't generalize this profile to other workloads without re-measuring.

## Pending audits

Two extensions to the existing tracer infrastructure are landed but not yet re-rendered:

- `audio_to_video_attn` coverage (tracer commit `9575a0f`) -- closes the 5-7% sampler undercount in audio-side wall-time. Required for tight sizing of concurrent-dispatch parallelism prize.
- AdaLN/RoPE/norm aten-op trace via `torch.profile` (tracer commit `9de86a3`) -- characterizes the 24.6% sampler residual. Required for assessing whether AdaLN dominates the residual (gates parallelism payoff) or norms / NAG / hook overhead do.

Both will populate on the next render of the FML2V benchmark workflow with the appropriate env vars set. Expected to refine the numbers above but not change the lever ranking.

## Related

- `docs/perf_research_framework.md` -- the framework for using this kind of data (Amdahl with measured attention-share, treat residual as hypothesis-needing-its-own-measurement).
- `CHANGELOG.md` v0.6.0 entry -- the production A/B that prompted this profile.
- `CHANGELOG.md` Backlog -- v0.6.1 / v0.7 candidates derived from this profile (persistent-CTA FFN then attention; concurrent-dispatch spike open; CUTLASS skipped).
- `docs/fp16_accum_fp8_matmul.md` -- why fp16-accum matmul throughput work doesn't help here.
- `docs/cuda_mask_kernel_scoping.md` -- precedent for the kernel-day-scoping discipline applied to v0.5.5; the same discipline applies to any v0.6.1+ persistent-CTA work.
