# MiniMax H3 kernel measurements

Last updated: 2026-09-13

Every H3 number this repo has, with the conditions that produced it.
Extracted verbatim from `CLAUDE.md` on 2026-09-08, where it had grown to
roughly a hundred lines of always-loaded context whose own caption said
not to quote most of it. Measurements belong in a dated record with
their conditions attached, not in the routing index.

## Open on 2026-09-13: two measurements drafted, harnesses committed, GPU held by a render

Recorded before the numbers exist so the question and its instrument are
on file even if the result is a revert. Both run on the RTX 4090 (sm89),
torch 2.14, the build identified by `build_info()` at the time of the run;
the tables land in `CHANGELOG.md` under v0.7.17 with that stamp.

1. **Cost and identity of the int64 strides in `csrc/fused/fused.cu`.**
   `tests/spikes/spike_fused_quant_offsets.py`, loading the pre-change
   `_fused` `.so` beside the new one so both builds run in one process on
   the same tensors. Per-kernel wall time for the per-warp q, per-block k
   and per-channel v launches at 41,822 / 104,030 / 109,126 / 149,000 rows
   on the fused-QKV view (stride_seq 21504, NHD), plus `torch.equal` on
   every output pair; then `per_channel_fp8`'s dequantized tail on both
   builds at a row count past the uint32 ceiling. What turns it red: any
   `bit-equal: NO` row, or a new/old ratio the noise floor cannot explain.
   What the entry commits to: if the ratio shows a real price, the record
   says so and the widening is reverted, not kept.
2. **per-thread Triton q/k versus per-warp CUDA q/k on sm89**, the
   dispatcher default sm89 inherits against the one it sets on sm100.
   Speed: `tests/spikes/spike_h3_qk_quant_gran.py`, whole call through
   `sageattn_consume` as the consumer calls it (`smooth_k=False`,
   `fp32+fp16`, fused view, unmasked) at the same four row counts, with
   the quantization step timed in isolation because it is the only code
   that differs, so the whole-call delta must be explained by it or it is
   noise. Accuracy: `tests/spikes/spike_h3_real_activations.py`, which
   now carries a per-warp arm, on the consumer's 2026-09-03 base16 t2v
   capture set (S=104,361, 25 cells, retained to 2026-09-20). The
   synthetic rtol column of the speed spike ranks the two arms and is not
   an accuracy figure for either, per the rule above. What flips the
   default: per-warp faster by more than the quantization step's own
   share can be noise, and real-activation rtol inside what per-thread
   scores on the same cells. Either way the dispatcher comment at
   `sageattention/core.py` is made true afterwards.

**Read the header on each block before quoting anything from it.** The
short version: the synthetic tables are valid for speed and VRAM and are
NOT a measurement of accuracy at H3 config; the real-activation figures
are the ones that bear on a mode decision. `CLAUDE.md` carries the rule,
this file carries the evidence.

**Do not run accum-config A/Bs expecting an accuracy result. The PV
accumulator is not an accuracy lever; the width of the PV operands is.**
Measured 2026-08-13 at a **MiniMax H3** self-attn shape
(1, 56, 41822, 128) bf16 against `EFFICIENT_ATTENTION`, on **synthetic
`torch.randn` inputs** -- see the real-activation correction below
before quoting any figure in this table:

| mode | mean_rtol | ms | vs torch |
|---|---|---|---|
| `fp8_cuda++` (`fp32+fp16`, default) | 0.0984 | 111.8 | 4.21x |
| `fp8_cuda` (`fp32+fp32`) | 0.0980 | 129.3 | 3.64x |
| `fp8_cuda` (`fp32`) | 0.0983 | 127.6 | 3.69x |
| `fp16_cuda` (`fp16+fp32`) | **0.0367** | 177.7 | 2.65x |
| `fp16_cuda` (`fp32`) | 0.0375 | 195.8 | 2.40x |
| `fp16_triton` | 0.0426 | 164.9 | 2.85x |

All three fp8 variants land within 0.0004 of each other **despite
`fp32+fp16` also co-varying the V-quant `scale_max`** (`core.py:1219-1221`
-> 2.25 vs 448.0). So that pair is not merely a confounded isolation --
both variables are close to inert, and it could not have produced an
accuracy finding in either direction. The same inertness was measured
on **LTX 2.3** shapes in 2026-04-23's "sm89 fp8 quantization scale" entry
(448 vs 2.25, 0.097 either way) -- a different workload, so it
corroborates the pattern rather than confirming it for H3. The H3
number above is the H3 evidence.

**What the two arms differ in is both PV operands, not V alone.** An
earlier version of this section said the gap was "entirely fp8-vs-fp16
V storage". It is not, and nothing here can isolate V, because P moves
with it:

| | P (post-exp) | V | QK |
|---|---|---|---|
| `fp8_cuda*` | fp32 -> e4m3, unscaled (`csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh:323`) | e4m3, per-channel scale (`core.py:1223`) | int8 |
| `fp16_cuda*` | fp32 -> fp16 (`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu:317`) | plain `v.to(float16)` cast, no scale (`core.py:988`) | int8 |

QK is int8 on both arms, so that much is held constant; everything else
in the PV matmul moves together, and there is no fp8-P/fp16-V kernel to
separate them with. Read the gap as **8-bit PV versus 16-bit PV**. This
is load-bearing when ranking against a third-party kernel that quantizes
P and V on different schemes -- "we measured 8-bit V costing Nx" is a
claim this repo cannot support.

On the synthetic numbers above, our default buys speed at ~98% of the
0.10 rtol budget; the fp16-PV path spends 1.59x the time and +287 MiB to
sit at ~37% of it.

**The ratio is measured, not a floor, and it is flat across S.** An
earlier version of this section guessed that the fp16 arm might be
reference-limited by the bf16 comparand. **Disproved by re-running
against an fp32 reference:** 0.0363 vs 0.0367 for fp16, 0.0984 either
way for fp8++. The reference dtype does not matter here.

Swept across a 17x range of sequence length (fp32 reference, synthetic):

| S | `fp8_cuda++` | `fp16_cuda` (`fp16+fp32`) | ratio |
|---|---|---|---|
| 4,608 | 2.0 ms / 0.0969 | 2.8 ms / 0.0363 | 2.67x |
| 24,576 | 39.4 ms / 0.0979 | 62.2 ms / 0.0363 | 2.70x |
| 41,822 | 112.2 ms / 0.0984 | 178.0 ms / 0.0363 | 2.71x |
| 78,336 | 394.7 ms / 0.0981 | 623.9 ms / 0.0362 | 2.71x |

What those four S values are is only partly recoverable, since the script
was not committed. 41,822 is a real packed fl2va length (1344x768, 124
frames). 24,576 and 78,336 coincide exactly with 1024x768 video-row
counts at on-grid latent_t of 32 and 102 -- i.e. 107-frame and 345-frame
clips -- but they are equally exactly 24x1024 and 76.5x1024, so intent is
not recoverable from the number. 4,608 maps to no on-grid H3 geometry and
reads as a low-end anchor. Either way the top of the sweep is at or near
the **legal** maximum clip length, so the range is not short of
production; describe it as a 17x span of S at H3's head config.

**No crossover exists**, so there is no "use fp16 above S=X" rule: the
synthetic accuracy ratio is flat at ~2.7x and the *synthetic per-call*
speed cost converges to ~1.58x (1.40x at the small end). **That 1.58x is
a kernel number, not a delivered one** -- the only e2e observation runs
the other way (a consumer render measured fp16 at 2.5 min against
fp8++'s 2.7, confounded by first-arm model loading), so the e2e effect
is unmeasured and must not be quoted as 1.58x slower renders. H3's S
varies widely -- it is the packed
`[text | refs | audio | video]` length, so canvas, aspect, clip length
and reference count all move it (a real 345-frame 1024x768 reference
render measures 143,386) -- and **one mode decision covers the whole
range**. `fp16_triton` tracks at ~0.042 and OOMs at 78,336.

**Every figure above is synthetic, and on real H3 activations the gap is
1.3x, not 2.7x.** This was measured before any of the above and recorded
in CHANGELOG v0.7.0 (2026-08-04): on q/k/v captured from an actual H3
forward -- post-RMSNorm, post-RoPE, exactly what sage receives -- fp8++
lands at mean_rtol **0.026** against an fp32 reference, roughly 4x below
what the synthetic bench reports, and the fp8++-to-fp16 gap narrows to
**1.3x** (so fp16 is ~0.020 there; only the ratio was recorded).
**Quote the real-activation figures for H3 mode decisions, not the
synthetic ones.** Real attention has structure that quantization handles
far better than iid gaussian noise does; on synthetic input softmax is
near-uniform, the output is a near-cancelling average of S random
vectors, and a symmetric-denominator element-wise rtol is then dominated
by cancellation.

An earlier version of this section called that 0.026 an **unresolved**
3.7x discrepancy and attributed it to
`tests/spikes/spike_h3_kernel_divergence.py`. Both halves were wrong.
That spike is the *synthetic* one and CHANGELOG v0.7.0 records its
output as 0.0960 (S=2k) to 0.0979 (S=38k), consistent with everything
above; 0.026 comes from its real-capture sibling
`tests/spikes/spike_h3_real_activations.py`. The two landed in the same
commit (`13b19e0`, "kernel divergence at H3 config, synthetic and
real"). Nothing was ever unexplained.

Further caveats on the synthetic figures: `torch.randn` inputs have zero
channel mean, so `smooth_k` is inert; and rtol-against-SDPA is not
perceptual quality. On `smooth_k` specifically, it has been measured at
H3 config **on real activations** and is a wash (0.0264 -> 0.0266, fp32
reference) even though K there carries a substantial channel offset
(|mean|/std 0.68 mean, 6.09 max), which is the precondition for it to
help. Do not re-litigate it, and note that a synthetic sweep *cannot*
reproduce that result in either direction.

**Provenance gap.** The 2026-08-13 mode table and S sweep have **no
committed script and no raw log** -- they exist in this file plus commit
messages `12a5872` and `1f619b4`, and `internal/log/` has no entry for
that date. Reproducible in principle, not reproducible from artifacts.
The two committed and runnable H3 accuracy harnesses are the divergence
spike pair named above; prefer them, and prefer the real-activation one.
`fp16_cuda` also silently drops `attn_mask` -- irrelevant to H3, whose
sole call site passes `mask=None`, but not to masked workloads.
