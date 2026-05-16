Last updated: 2026-05-16 (moved from internal/analysis/ to docs/ for public sharing; expert-confirmation paragraphs paraphrased to drop attributions)

# fp16 accumulation with fp8 matmul on sm89 -- what's real, what's not, what it means for sage

## TL;DR

The throughput claim is real: a hand-rolled inline-PTX fp8 matmul with fp16 accumulation can hit ~450-475 TFLOPS on a 4090 at LLM-shape matmuls (72% of the 660 TFLOPS theoretical peak for fp8->fp16-accum on sm89). cuBLASLt at the same shapes with fp32 accumulation runs ~36-43% slower; cuBLASLt with fp16 accumulation is presumably similar to the hand-rolled number, not slower.

The load-bearing constraint that the throughput claim ignores: at the per-MMA-instruction granularity (`mma.sync.aligned.m16n8k32`), the accumulator can already overflow fp16 if inputs aren't pre-scaled aggressively. fp8 max is ~448; worst-case fp8 x fp8 product is ~200K per multiply; summed over k=32 in a single instruction the peak partial is ~6.4M, which is ~100x past fp16 max (65504). LLM weights are tight enough that this stays within range in practice; DiT activations at LTX scale are not necessarily safe.

For sage_ffn specifically: irrelevant to the v0.6.0 production gap, which is L2-contention-bound and dispatch-bound, not matmul-throughput-bound. Doubling matmul TFLOPS does not change the bytes/sec the kernel can pull from HBM when L2 is hostile, and does not reduce the per-call launch overhead at LTX's ~1000-FFN-calls-per-render count.

For sage attention: the fp8++ kernel already uses fp16 accumulation safely (the `pv_accum_dtype="fp32+fp16"` split-accum path). Attention's bounded dynamic range (softmax + post-scaling) is what makes this safe. The same approach does not generalize to FFN/MLP without per-shape range bounds being known in advance.

Decision: do not pursue fp16-accum for sage_ffn as a v0.6.1 candidate. Bookmark the technique for the v0.6.1 CUTLASS backlog entry (CUTLASS exposes both accum modes, and if the CUTLASS port ever fires, evaluating fp16-accum within the rtol budget is one of the levers to try).

## Context

This analysis came out of two events in the same window:

1. v0.6.0 walk-back. In-pipeline A/B on a two-sampler LTX FML2V workflow showed sage_ffn at +1.79% e2e slower than the chunking-only baseline (+20% per-call at stage-2 T=42240, +3% at stage-1 T=10780). Root cause: L2 cache contention with neighboring attention modules + cumulative kernel-launch overhead. Walk-back landed at commit `1044d00`; v0.6.0 ships as a completeness primitive, not a perf win.

2. A suggestion to chase the next rabbit hole: "figure out how to use fp16 accum with fp8 matmuls without huge quality loss." The basis for the suggestion was a public LinkedIn article reverse-engineering SM89's `QMMA.16832` register layout and reporting ~473 TFLOPS at LLM-shape matmuls on a 4090. An independent implementer who tried the technique on a similar kernel reported the same throughput regime but bounced off the dynamic-range constraint: at the per-MMA-instruction granularity, the accumulator can already overflow fp16 if inputs aren't pre-scaled aggressively, and pre-scaling them tight enough to be safe costs accuracy past the rtol budget for DiT generation.

The article's technical content is credible; what it omits is the range/overflow constraint that decides whether the kernel is usable for a given workload.

## What the article shows

Achievements:

- 473 TFLOPS at M x N x K = 2048 x 4096 x 11008 (an LLM-shape matmul, common for transformer FFN up-projection on a 11k-context model)
- 448.4 TFLOPS at 8192 x 8192 x 8192
- 72% of the 660 TFLOPS theoretical peak for fp8 -> fp16-accum on sm89
- +43% vs cuBLASLt fp32-accumulation at the LLM shape, +36% vs the same baseline at 8192^3

Mechanism reverse-engineered:

- The author noticed SM89's `QMMA.16832` instruction uses a non-standard A-fragment register layout vs the PTX ISA documentation. Specifically: registers a1 and a2 are swapped relative to the spec. Hardware interleaves rows before k-halves; PTX docs grouped by row then k-position.
- Corrected layout: `a1` reads `A[group_id + 8, k_low]` (not k_high), `a2` reads `A[group_id, k_high]` (not k_low).
- Verified via SASS inspection showing the expected `QMMA.16832.F16.E4M3.E4M3` opcode.

Implementation:

- Custom inline PTX in raw CUDA. Not CUTLASS, not cuBLASLt.
- Instruction: `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`
- Tile: 128 x 128 x 64 (BM x BN x BK)
- Warp layout: 2 x 4 warps per block
- Shared memory: 20 KB (10 KB A + 10 KB B) with 80-byte stride (64 data + 16 padding)
- Register budget: ~60 registers/thread
- Global loads: 128-bit uint4 for coalescing
- No `cp.async` pipelining yet; flagged as future headroom

Performance gap to peak (28%) attributed to global-to-shared latency and ~8-way bank conflicts on stores, per Nsight Compute.

## What the article omits

Two things, ordered by load-bearingness:

1. **Accuracy data.** No rtol or relative error numbers vs an fp32 reference are shown. fp16 accumulation has a precision cost; whether it's within the budget for any specific consumer is the actual deciding question. For LLM inference at the article's tested shapes, the cost is probably fine (LLM weights are well-conditioned and the activations stay in modest ranges). For DiT-class workloads at the activation distributions LTX hits, this is not safe by default.

2. **Comparison baseline choice.** The article compares to cuBLASLt fp32-accumulation. cuBLASLt also supports fp16-accumulation; an apples-to-apples comparison would presumably shrink the headline 43% delta significantly. The article's true claim should be "we found a register-layout quirk in the PTX docs and our hand-written kernel approaches what cuBLASLt fp16-accum mode already does at this shape" -- which is a much smaller and more useful claim than the headline.

## Independent confirmation

An independent implementer who tried the technique reported the
same throughput regime -- well past 450 TFLOPS on a 4090 doing
inline-PTX `mma.sync.aligned.m16n8k32` matmul. So the article's
headline number is not a baseline-selection artifact; it's a real
measurement of a real PTX pattern, replicable by anyone with the
discipline to write the kernel.

What stopped them from shipping it: the per-MMA accumulator overflow.
Pre-scaling the inputs tight enough to keep accumulation safe within
fp16's 65504 dynamic range cost too much usable input range; the
resulting kernel was faster but exceeded the accuracy budget for the
intended workload.

Confirmed empirically on a real workload, not just synthetic. The
same implementer took the fp16-accum kernel as far as running it
under ComfyUI on LTX 2.3: it was faster on the large-input matmuls
(matching the article's TFLOPS claim qualitatively), but the
generated output was "visibly worse" -- not catastrophically broken,
not bit-identical to the fp32-accum reference, just degraded enough
that a human looking at the result would notice. That degradation
exceeds the rtol budget that gates production DiT generation
(~0.10), even though it's not a hard correctness failure.

This is the strongest form of evidence: a domain expert built the
kernel, deployed it in production conditions, and observed the
quality degradation directly. The dynamic-range problem isn't
theoretical; it bites LTX distributions specifically.

## The actual constraint: per-MMA accumulator overflow

The fp16 max is 65504 (positive normal). The fp8 e4m3 max representable is ~448.

A single `mma.sync.aligned.m16n8k32` instruction accumulates 32 fp8 multiplies in one shot. Worst-case bound on a single fp8 x fp8 product:

```
448 * 448 = 200,704
```

Already past fp16 max at one multiply. Summed over k=32:

```
peak_partial = 32 * 200,704 = 6,422,528
```

That's ~98x past fp16 max. To stay in range you have to pre-scale the inputs so that the accumulator across k=32 stays bounded. The per-block-K activation quant pattern sage_ffn already uses (each `(BLOCK_M, BLOCK_K)` chunk of activation gets its own f32 scale, applied inline) is *exactly* this kind of pre-scaling, but it's currently structured to allow safe fp32 accumulation, not safe fp16 accumulation. Tightening the scaling to make fp16 accumulation safe would either:

- Sacrifice useful dynamic range (the "cap the range too much" failure mode -- accuracy degradation outside the rtol budget), or
- Require finer-grained scaling (e.g. per-k=32 chunk instead of per-`BLOCK_K`), which adds amax/scale-compute cost that competes with the throughput gain.

LLM workloads stay in range because trained LLM weights are well-conditioned. The author's tested shapes (K=11008, K=8192) on LLM-style weights don't hit the overflow regime often enough to break the rtol budget. DiT activations at LTX scale, especially at the stage-2 multi-guide-expanded shapes where we already see the production regression, are not the same distribution.

## Amdahl ceiling: FFN is only 12% of LTX FML2V render

Even granting an optimistic counterfactual where fp16-accum gives sage_ffn
a 2x matmul-throughput boost AND that boost transfers to production
(neither is true today, both would have to be earned), the e2e impact
is bounded by Amdahl. Per `docs/ltx_workload_profile.md` (rev 2,
tracer-grounded numbers):

- Total FFN (video + audio, all stages) = ~16% of total FML2V render
  wall-time
- Stage-2 video FFN alone = ~10% of total render
- Hypothetical 2x speedup on total FFN caps e2e reduction at ~8%
- Hypothetical 2x speedup on stage-2 ff specifically caps it at ~5%
- 1.5x speedup caps e2e reduction at ~3-5% depending on FFN scope

So even in the best-case fp16-accum scenario where the throughput
transfers AND the rtol stays in budget AND L2 contention isn't a
problem, the e2e ceiling is single-digit percent. By contrast,
persistent-CTA on stage-2 attention (25.7% of render) has a ~10-13%
e2e ceiling at the same kernel-side speedup. The leverage calculus
strongly prefers attention over FFN-matmul-throughput as the next
sage-side perf investment.

## Why this doesn't help sage_ffn's production gap

The v0.6.0 production A/B verdict attributed the regression to:

1. L2 cache contention with neighboring attention modules. `attn1` (~107 ms at T=42240) runs immediately before `ff` at stage-2; the attention pass evicts FFN's L2 residency. The X-tile-lives-in-L2 assumption from day-3 perf analysis breaks; cold-L2 FFN is bandwidth-bound.
2. Cumulative kernel-launch overhead. LTX 2.3 fires ~1056 ff calls per render across transformer blocks. sage_ffn is two kernel launches per call; torch reference is one cuBLASLt call per matmul.

Neither is matmul-throughput-bound. A 473 TFLOPS matmul still has to:

- Pull the X tile from HBM if it's not in L2 (the bandwidth bottleneck); fp16-accum doesn't change byte-movement, only ALU throughput.
- Pay 1056 x 2 = 2112 kernel launches per render; fp16-accum doesn't change launch count.

The article's mechanism would matter if sage_ffn's bottleneck were "the matmul is compute-bound and we want more FLOPS." It isn't. It's bandwidth-bound at stage-2 (cold L2) and dispatch-bound across the call count.

So even in the optimistic case where we hit 473 TFLOPS in production conditions (which is itself unlikely -- the article's number is also synthetic-isolation), the production e2e number on the FML2V workflow would move by approximately zero.

## Where fp16-accum is already deployed safely

The sm89 fp8++ attention kernel (`sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`) already uses split fp32+fp16 accumulation in production. It works because:

- Attention's PV step accumulates over the seq_kv dimension (typically a few thousand) but the partial sums are bounded by the softmax probabilities (which sum to 1.0 over the same axis). The PV accumulator stays well within fp16 range at all seq_kv sizes we've tested.
- The QK step (which dominates the dynamic-range concern) accumulates to fp32 in the fp32+fp16 mode. Only PV uses fp16.

This is the structural difference between attention and FFN: attention has an explicit normalization (softmax) that bounds the accumulator partial sums; FFN has no such normalization, so the dynamic range of the matmul output is whatever the input distribution produces.

The "fp16 accum works where the operator already has a bounded dynamic range; it breaks where the operator's range is workload-dependent" heuristic is what falls out of this. Attention satisfies the precondition (bounded). FFN/MLP doesn't (workload-dependent). This is the structural reason fp8++ attention with fp16-accum ships in production while the analogous FFN experiment doesn't.

## Where fp16-accum would be a wedge for sage

Two hypothetical paths, neither currently scheduled:

1. **A different workload class where FFN matmul throughput is the bottleneck.** Single-pass video (no two-stage refiner, no multi-guide expansion), or smaller-inner-dim FFN blocks where the cache locality picture is different. The article's LLM shape is structurally similar (single forward pass, ~10k context); LLM inference on sm89 is a workload where fp16-accum could plausibly win without the L2-contention issue. Sage-fork is not in the LLM inference business, but the structural point stands: if a video DiT lands with cleaner cache behavior or smaller FFN call counts, fp16-accum could matter.

2. **A CUTLASS port of sage_ffn.** The v0.6.1 Backlog already mentions this. CUTLASS exposes both fp16 and fp32 accumulation modes; the right discipline is to port the kernel, evaluate fp16-accum behind the rtol gate at LTX activation distributions, and ship whichever passes. The register-layout quirk the article discovered would save 1-3 days of debugging time if CUTLASS doesn't already account for it (probably does, but worth knowing).

Trigger for either path: user demand for "actually faster than torch reference" on a real workload, or a different workload class lands net-positive under sage_ffn and we want to generalize. Neither is the current case.

## On low-cost signals as a credibility filter

Worth recording. The first read on the article was "publication-venue heuristics + author bio + buzzword density suggest skepticism warranted." That's reasonable triage on low-cost signals, but it would have been wrong here: independent expert replication confirmed the throughput claim. The technical content of the article holds up; what the article omits (the per-MMA range constraint) is what decides whether the kernel is useful for any specific workload.

General heuristic: noisy-signal triage (author affiliations, follower counts, buzzword density, publication venue) is fine for prioritizing what to read, but it does not survive contact with a domain expert who can replicate. The expert-replication step is the actual gate.

## Open questions worth noting

1. **What does fp16-accum cost on rtol at LTX FFN distributions?** Not measured directly here. Would need: take the v0.6 kernel, swap fp32 accumulator for fp16, re-run the correctness test at both LTX shapes, measure mean_rtol. If it stays under 0.10 at LTX activations, the throughput is gettable; if it blows past 0.10, that's the same wall the independent replication hit. Cheap experiment (~half-day), worth doing if the v0.6.1 CUTLASS work ever fires.

2. **Is the per-MMA overflow actually load-bearing, or does in-practice averaging keep partials in range even with bad worst-case bounds?** The 6.4M peak partial is worst-case; in practice partials are roughly Gaussian-distributed around zero (positive and negative fp8 products cancel) and the actual accumulator stays much smaller. But the "cap the range too much" failure mode reported by the independent implementer suggests the worst-case does matter for the rtol gate at LTX activation distributions. Worth measuring directly rather than reasoning from bounds.

3. **Does CUTLASS already use the corrected register layout?** The article's "PTX docs are wrong about a1/a2 ordering" finding implies a bug in the docs, not in CUTLASS (which is empirically tuned). Worth verifying before assuming the register-swap discovery is useful new information for a CUTLASS port; if CUTLASS already encodes the correct layout, the article's only contribution is "here's what the PTX docs got wrong, in case you're writing inline PTX from scratch."

4. **Is there a shape regime where the per-MMA overflow doesn't bite even on FFN?** Probably yes for FFN blocks with small K. LTX has hidden=4096 which is not small enough; SDXL/Flux/Wan2 may have different shape characteristics worth checking. Not a current priority.

## Related docs

- `CHANGELOG.md` v0.6.0 entry -- the production verdict that prompted this analysis.
- `CHANGELOG.md` Backlog -- v0.6.1 candidates including persistent-CTA hybrid and CUTLASS port.
- `CLAUDE.md` Conventions, "Gate ship-decisions on in-pipeline A/B when synthetic-bench can't measure the dominant cost" bullet -- the discipline rule that applies here.
- `docs/ltx_workload_profile.md` -- the FML2V wall-time breakdown that grounds the Amdahl ceiling above.
- `docs/fp16_matmul_accum.md` -- prior analysis of whether a downstream `enable_fp16_accumulation` knob affects sage output (no). Different question (fp16 accum on bf16/fp16 matmul, not on fp8) but adjacent.
