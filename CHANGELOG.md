# Changelog

Local divergence record for this fork. **Not a public release timeline**:
this is a personal editable install used as an attention-kernel
measurement surface for sm89 / RTX 40xx / Ada. The version blocks below
are commit-history snapshots, not semver releases -- they group
changes into coherent chunks for retrospective navigation. There are
no compatibility promises across versions.

Sections:

- **Versions** -- additions and changes layered on top of the
  upstream-from-woct0rdho baseline. Newest first.
- **Known kernel bugs** -- factual record of real defects we've
  measured in this fork's kernels. Start here if you're debugging
  sage-attention-adjacent correctness problems.
- **Backlog** -- real open TODOs with explicit triggers to act.
- **Decision log** -- investigations that closed without action,
  recorded so we don't re-derive them. Each entry has a reopen-trigger.
- **Recurring process items** -- cron-like checks, not engineering work.

## Known kernel bugs

Real defects we've measured in this fork's kernels. We own the fork now;
these are ours to fix when we want to. If you're debugging
sage-attention-adjacent correctness problems, start here.

### CUDA kernels have partial attention-mask support (sm89 fp8++ landed v0.5.5; sm80 + other variants still missing)

The original gap inherited from `thu-ml/SageAttention`: Python wrappers
`sageattn_qk_int8_pv_fp16_cuda` and `sageattn_qk_int8_pv_fp8_cuda`
accept `attn_mask` via `**kwargs` but never pass it through to the
C++ layer. The C++ `MaskMode` enum originally had only `{kNone,
kCausal}`. Masks were silently dropped on all CUDA code paths.

**v0.5.5 (2026-05-13) closed this on the load-bearing sm89 path** --
the `MaskMode::kGeneral` variant + `apply_general_mask` helper landed
in the fp8++ kernel (`qk_int_sv_f8_cuda_sm89.cuh` +
`sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`). The
dispatcher routes masked calls on sm89 + CUDA >= 12.8 to the new
path. Measurement at LTX-class shape (bf16, T=2048, D=128) shows
mean_rtol=0.098 vs Triton reference (same accuracy profile as the
unmasked fp8++ path); the historic ~0.94 silent-drop signature is
gone.

**Still missing** (deferred per scope discipline):
- sm80 fp16_cuda path (`qk_int_sv_f16_cuda_sm80.cu`). Same kernel
  pattern; deferred until a workload hits sm80 + masks frequently.
- The other 6 sm89 variants (`accum_f32_*`, `accum_f16_attn_inst_buf`,
  the no-`inst_buf` versions). Not dispatcher-hit on sm89 + CUDA >= 12.8.
- The sage 3 Blackwell path (also lacks mask support upstream;
  removed in v0.5.0).
- Whole-block-skip optimization on sparse bool masks (Triton has it
  via the `tl.max(mask_block) == 0 -> skip` short-circuit; the CUDA
  kernel's pipelined K-iteration loop makes the analog non-trivial
  to mirror without serializing the pipeline).

Until those land, hand-picking a `_cuda` kernel with a non-None mask
on those paths still warns + drops (the v0.3.1 soft-warn safety net).
The dispatcher's safe-default routing handles this for consumers
that call `sageattn()` without overrides.

Repro: `tests/repros/repro_cuda_mask_kernel.py` (predates the v0.5.5
fix; documents the original symptom on sm89 fp8++; now passes
through the kGeneral path).

Discovered: 2026-04-23 via `tests/test_sageattn_ltx_shapes.py` (the
seq_kv sweep exposed the rtol-vs-seq_kv scaling signature). Closed
on sm89 fp8++: 2026-05-13 (v0.5.5).

### The CUDA quant kernels form global offsets in uint32 (ceiling ~199,729 rows)

Same lineage as the v0.7.0 int32 overflow: an upstream defect, in code we
ship unmodified (`csrc/fused/` is on the upstream-unmodified list in
`docs/whats_ours_vs_upstream.md`), found while testing our own fix. The
v0.7.0 int64 fix covered the Triton quant kernels; `csrc/fused/fused.cu`
has the same defect one type wider: every kernel there takes its strides as
`uint32_t` and forms the global offset as

```cpp
input + batch_id * stride_bz_input + head_id * stride_h_input
      + thread_base_token * stride_seq_input + ...
```

with all four operands `uint32_t`, so the sum wraps at 2**32 elements rather
than 2**31. `per_channel_fp8` and the sub-mean/quant pre-kernels read the
caller's original bf16 V, which in a DiT block is a view into a fused QKV
buffer carrying `stride_seq = 3*heads*head_dim`. At MiniMax H3's config that
puts the ceiling at:

| layout | stride_seq | wraps at |
|---|---|---|
| fused QKV view (production) | 21,504 | S = 199,729 rows (~660 frames at 1344x768) |
| contiguous | 7,168 | S = 599,186 rows |

Measured safe below it: at S=109,126 (362 frames, the largest a 24 GB card
reaches) the largest offset formed is 2.35e9, inside uint32's 4.29e9. So this
is a latent ceiling, not a live defect -- and on 24 GB it is unreachable,
since q+k+v alone at S=199,729 would be 8.6 GB on top of a ~20 GB checkpoint.

**Trigger to fix:** a card with enough VRAM to reach it, or a model whose
`stride_seq` is wider than H3's. Fixing it means promoting the offset
expressions to `size_t` in `csrc/fused/fused.cu`; unlike the Triton side
there is no cheap per-launch specialization, so measure the cost before
taking it.

Recorded: 2026-08-05, while validating the Triton fix at 362 frames.

## Backlog

Real open TODOs. Each has an explicit trigger-to-act; we don't do these
speculatively.

### Drop `per_channel_fp8`'s full-size bf16 transpose buffer -- SUPERSEDED 2026-08-06, not withdrawn

**Superseded by consumer-side head-group chunking. Do not start this as a
kernel day.** Everything below about the mechanism is still true, which is
exactly why this entry stays: the bf16 transpose buffer really does set the
attention peak, and a future reader who rediscovers that will re-propose
this same work. It should arrive pre-answered.

Two reasons, in order of durability.

**We do not buy insurance with kernel days on this box.** The
headroom-to-speed conversion here is bounded at ~2.6% (see below), so this
was never a speed item -- it is insurance on a pipeline running at 99.9% of
the card. A consumer node now slices attention into head groups and
quantizes per group, which shrinks the same transient by the group count
with no kernel change at all. Spending a kernel day on insurance that a
caller-side `for` loop already provides is the wrong trade regardless of how
the two compare numerically.

**And numerically it is not close.** The per-call transient is ~1430 MiB
(q_int8+k_int8 572, the bf16 transpose buffer 572, v_fp8 286) on top of the
1715 MiB fused QKV buffer, which reconciles with the measured 3148 MiB peak.
Chunking scales that transient by 1/n: roughly 2430 MiB at n=2 and 2073 at
n=4, so ~1070 MiB recovered against the ~572 MiB this item would deliver.

**What it costs, and why this may come back.** Chunking multiplies attention
calls by n. At 50 blocks x 20 steps that is 1000 calls per render becoming
4000 at n=4, each expanding to several CUDA launches. That is four times the
call count at which `sage_ffn` died on cumulative launch overhead, on a
kernel that is already the dominant cost -- so chunking is a VRAM-versus-
wall-clock dial, not a free win, and it needs an in-pipeline A/B before
anyone ships it. **If that A/B goes badly and nobody adopts chunking, this
item is live again.**

**Preserve on the way past:** the in-place mean-subtraction prerequisite
found while measuring `sageattn_consume` (v0.7.3). Dropping the transpose
buffer alone moves the fused-case peak 3148 -> 2859, because `k = k - km`
then sets the floor; the pair together reaches 2573. Whoever picks this back
up needs both halves or the work under-delivers and looks like a failed bet.

**What v0.7.4 does to this item.** Two things, in opposite directions.
Consumers can now recover 286 of the ~575 MiB caller-side by cloning v
(v0.7.4), with no kernel work at all, so the remaining prize here is
smaller than it was. Against that, landing the pair *retires* the clone
rather than stacking with it: a cloning caller's floor is 2859 (fused 1715
+ clone 572 + the int8 pair 572, all live before q and k are released),
above the 2573 the pair reaches without cloning. So whoever ships this must
also flip `sageattn_consume_prefers_cloned_v` to False and retire
`test_fused_caller_cloned_v_recovers_the_saving`, which will go red for
that reason and says so in its failure message. Consumers gate their clone
on the predicate, not on an arch check, so the flip reaches them on upgrade
without an edit on their side.

---

Original entry follows.

`per_channel_fp8` allocates `v_transposed_permutted` at V's full size in
bf16, fills it via `transpose_pad_permute_cuda`, then quantizes it into a
separate fp8 tensor (`sageattention/quant.py:269-292`). At MiniMax H3's
fl2va shape that transient is 572 MiB, and it -- not the output
allocation -- is what sets the peak for the whole attention call:

```
after qkv views                                    1715 MiB
+ q_int8, k_int8                                   2287 MiB
+ v_transposed_permutted (bf16, transient)         2862 MiB
+ v_fp8                                            3148 MiB   <- peak
```

It also caps what `sageattn_consume` can deliver in the configuration
that matters. When q/k/v are views of one fused QKV buffer, releasing the
buffer cannot happen until V is quantized, by which point the transient
has already set the peak: 435 MiB saved instead of 858.

The scale is per-channel over the full sequence, so it needs two passes
either way. The fix is to compute the per-channel max reading V directly
and then transpose-and-quantize straight to fp8, never materializing the
bf16 intermediate -- trading a second read of V for 572 MiB at this
shape.

**Trigger, rewritten 2026-08-05.** It used to read "a workload that OOMs at
the attention peak" -- i.e. wait for the failure. That is the wrong trigger
for a robustness item. The 362-frame H3 measurement shows why: the render
phase peaks at 21,228 MiB of 24,101 and the text-encode phase touched
**24,076 MiB, 99.9% of the card**. The pipeline stages 51,418 MiB of models
against 24,101 MiB of VRAM (2.1x oversubscribed) and currently completes
with no meaningful slack. One bad allocation kills a 17-minute render.

**Do not justify this as a speedup.** The recoverable time is bounded at
~2.6% (0.6% per-step weight streaming, which the trace shows is already
fully hidden behind compute, plus 2.0% phase swapping), and part of that
2.0% is unavoidable since the TE and DiT are 45.9 GB together and can never
co-reside. Framed as perf this is `sage_ffn` again: plausible mechanism,
real synthetic number, dies in the production A/B. Framed as headroom on a
pipeline running at 99.9% of the card, it stands up.

**Prerequisite before any kernel work:** confirm ComfyUI's dynamic VRAM
actually keeps more weights resident when the attention peak drops. If its
staging does not react to the freed headroom, the whole causal chain
(lower peak -> more resident -> less streaming) is severed and the work
buys only OOM margin. Cheap to test with `sageattn_consume` on vs off while
sampling resident VRAM.

Kernel-day work in `csrc/fused/fused.cu` (upstream code); scope it first per
the scoping-doc precedent, and gate on in-pipeline A/B since the extra V
read is a real cost that synthetic bench will show as a regression.

### Fix the int32 offset overflow in `quant_per_block_varlen.py`

The v0.7.0 overflow fix covers `quant_per_thread.py` and
`quant_per_block.py`. The varlen module has the same defect and was left
alone: its bound depends on `cu_seqlens` values loaded inside the kernel,
so the host-side check needs the cumulative-length array plumbed through,
and no ComfyUI path reaches `sageattn_varlen` for us to test against.

**Trigger:** a consumer adopts `sageattn_varlen`, or we upstream the
v0.7.0 fix (in which case send all three modules together).

### Retire the nvcc-13.3 CUDA-toolkit guard once a fixed nvcc ships

`build.sh` carries a `KNOWN_BAD_CUDA=" 13.3 "` blocklist that
auto-switches the build off nvcc 13.3 (it miscompiles PyTorch >=2.12
headers via a cudafe++ front-end regression; full A/B in v0.6.6).
**Trigger:** a 13.3 patch or 13.4+ toolkit whose `nvcc --version` still
reports a blocklisted version but compiles the same-TU repro clean.
**Action:** drop the fixed version from `KNOWN_BAD_CUDA`; if the set
empties, remove the guard block entirely. **Verify:** put the
previously-broken toolkit first on `PATH` and confirm a `.cu` including
`<ATen/core/function_schema.h>` compiles (the repro from v0.6.6).

### Persistent-CTA hybrid for stage-2 attention (highest e2e leverage; v0.7 candidate)

After v0.6.0's production A/B, a downstream consumer characterized
the wall-time breakdown of the FML2V multi-guide render. **Stage-2
attn1 (video self-attention at T=42240) is the single heaviest
sub-module across the whole render** -- materially larger than the
FFN share that v0.6 targeted. The optimization-leverage calculus
shifts: attention is the bigger lever. See
`docs/ltx_workload_profile.md` for the canonical breakdown with
per-sub-module percentages + the FFN-share triplet.

A persistent-CTA hybrid (CTAs hold M-tile state in registers/L2 across
kernel calls, attacking the L2-contention root cause) applied to stage-2
attn1 has an estimated ceiling of ~15% e2e wall-time reduction --
the largest single perf lever in the LTX 2.3 production stack.

Difficulty: high (persistent-CTA Triton is non-trivial). 2-3 weeks of
kernel-engineering work.

**Trigger to act:** persistent-CTA pattern is validated on FFN first
(see the FFN entry below), AND user demand for a real attention-side
e2e win on LTX-class workloads. Sequencing matters: prove the pattern
on the smaller / lower-risk surface first, then port to the larger /
higher-payoff surface.

### Persistent-CTA hybrid for sage_ffn (validates the pattern; v0.6.1 candidate)

v0.6.0's sage_ffn is +1.79% e2e slower than the chunking-only baseline
on a two-sampler FML2V workflow (+20% per-call at stage-2, +3% at
stage-1). Root cause is L2 contention with neighboring attention
modules + cumulative kernel-launch overhead at LTX's
~1000-FFN-calls/render count.

The persistent-CTA hybrid (option b' from the scoping doc) attacks
the L2-contention root cause directly: persistent CTAs hold M-tile
state in registers/L2 across the two matmul kernels, so the
intermediate doesn't have to re-fetch when neighboring attention
modules thrash L2. Estimated 1-2 weeks of work. Expected delivered:
~10-20% FFN speedup over current sage_ffn, putting it at parity-or-
better vs torch reference, plus the memory-side win at stage-2 if
the intermediate stays in L2.

Smaller e2e gain than the attention port above (~3-5% wall-time vs
~15%), but builds directly on v0.6 work and validates whether the
hybrid pattern is viable before committing to the attention port.

**v0.6.5 Cell C confirmation refines this item's framing.** With the
6-bug consumer integration chain fully closed and sage_ffn dispatching
end-to-end, per-stage FFN kernel time measures sage 22% slower at
stage-1 (T=10780) and 5% slower at stage-2 (T=42240) vs production
stock fp8 path -- despite synthetic isolation showing 1.39x / 1.60x
sage advantage at the same shapes. The inverted sign means the
synthetic-vs-production gap concentrates at the kernel boundary itself,
not framework overhead. Two open hypotheses (not blocking, neither yet
preferred):

  (1) Stock comparand identity -- synthetic compares vs
      `torch._scaled_mm`; production stock is `comfy.ops.fp8_linear`
      wrapped in KJNodes' `LTXVChunkFeedForward` (chunked at 4096
      with cached compilation state). Different baseline.
  (2) Sage autotune state under interleaving -- production has sage
      attention + sage_ffn dispatches interleaved at varying shapes.
      Autotune may pick a different tile config than synthetic
      isolation converges on.

Either explains the gap; neither is a sage correctness bug.
Persistent-CTA targets (1)/(2) symmetrically by removing the L2-thrash
pathway between matmuls, so the item stays load-bearing for closing
the v0.6 e2e gap. The "**ships as completeness primitive, not perf
win**" docstring framing remains correct for v0.6 specifically.

**Trigger to act:** user demand for "actually faster than torch
reference on a real workload" surfaces, OR a different production
workload class (single-pass / non-multi-guide) lands net-positive
under sage_ffn and the priority becomes generalizing that, OR
concurrent-dispatch consumer wrapper (consumer-side §6.1 candidate)
ships ahead of this and closes the e2e gap by a different path.

Recommended sequence: FFN persistent-CTA first (lower risk, faster
to ship, validates the technique), attention persistent-CTA second
(higher payoff once the pattern is proven on the smaller surface).

### CUTLASS-based fp8 matmul backend (skip per workload-profile analysis)

Was queued as an option for closing the v0.6.0 production gap; audio-
loop's workload-profile data + the L2-contention root cause analysis
showed this is the wrong root-cause attack. CUTLASS addresses matmul
codegen quality (Triton-vs-cuBLASLt gap, bounded at ~1.27-1.36x in
isolation); the production gap is L2 contention + dispatch overhead,
not matmul throughput. cuBLASLt-level codegen on a kernel that still
thrashes L2 doesn't move the needle.

A more detailed analysis of why fp16-accum + CUTLASS-class matmul
doesn't help here is in `docs/fp16_accum_fp8_matmul.md`.

**Trigger to revisit:** persistent-CTA hybrid lands and the L2-thrash
hypothesis is validated, AND a workload class surfaces where matmul
throughput IS the bottleneck (not the current case).

### Mask-aware autotune key (measurement hygiene; cheap)

Triton's autotune key on the sm89 mask-correct path doesn't
discriminate by mask kind. Warm-cache config inherited across mask
shapes (causal, general, none) pollutes future perf measurements --
a config tuned for one mask kind gets reused on another. The fix:
add a mask-kind discriminator to the autotune key.

Variance-class fix, not a perf bet. Disproportionate value because:

- Removes the "warm-cache config inherited across mask kinds" issue
  that polluted prior measurements
- Foundational for any future kernel measurement work (persistent-CTA
  spike measurements would benefit)
- Catches the synthetic-vs-production trap class earlier next time

Estimated 1-2 hours implementation.

**Trigger to act:** land regardless of whether the larger persistent-
CTA work happens. Measurement infrastructure compounds; do it before
the next perf experiment fires.

### sage_ffn autotune key: coarsen to power-of-2 buckets on M

Current cache key is `["M", "N", "K"]`. Every new M re-autotunes
(~10-15s per kernel). LTX 2.3 uses two stable M values, but other
modes (chunked FFN, different resolution / frame count) mint
others. One-line change: `key=[lambda M: triton.next_power_of_2(M),
"N", "K"]` -- T=10780 and T=11000 share a cache entry.

Risk is bounded: the curated 8-config sweep means bucket-winner
variance is small. But the two LTX shapes have different winners
(stage-1: num_warps=4 num_stages=2; stage-2: num_warps=8
num_stages=4), so bucketing could land a sub-optimal config on a
neighbor M.

**Trigger to act:** user feedback that first-render-per-shape
autotune-cost is painful across workflow variations, OR a downstream
consumer reports cycling through M values frequently in production.

### Extract `tests/_helpers.py` for shared `make_qkv` + `require_cuda`

Four standalone test files now duplicate near-identical scaffolding:
`test_sageattn_ltx_shapes.py`, `test_sageattn_image_shapes.py`,
`spike_torch_compile.py`, `test_dispatched_kernel_telemetry.py`. Each
defines its own `_make_qkv()` (or `make_qkv()`) for building random
QKV tensors, and each opens with the same `if not torch.cuda.is_available(): skip`
guard. Lifting both into `tests/_helpers.py` would consolidate ~30
lines across files; the standalone-script convention (no pytest, no
conftest) means a flat helper module is the right shape, not a
fixture.

**Trigger to act:** next time one of these four test files needs
editing for an unrelated reason. Don't do it speculatively -- the
duplication is currently inert and the import path
(`from _helpers import make_qkv`) needs `tests/` on `sys.path` which
the standalone scripts don't set up today. Mild churn for tiny gain
unless we're already in the file.

### Extend CUDA mask support beyond sm89 fp8++ (sm80, other sm89 variants)

The sm89 fp8++ kernel landed mask support in v0.5.5 (2026-05-13). The
remaining surfaces:

1. **sm80 fp16_cuda** (`qk_int_sv_f16_cuda_sm80.cu`). Same kernel
   pattern as sm89; would mirror the `MaskMode::kGeneral` +
   `apply_general_mask` work. Deferred -- sm89 is the load-bearing
   arch for this fork and the dispatcher already routes correctly
   on sm89.
2. **Other 6 sm89 variants** (`accum_f32_*`, `accum_f16_attn_inst_buf`,
   the no-`inst_buf` versions). Same kernel template; the call sites
   already pass nullptr/0 for the mask params after v0.5.5 (the
   `if constexpr` branches dissolve in the kCausal/kNone
   specializations). Adding the kGeneral path here would mirror what
   the fp8++ variant does; deferred until a dispatcher branch hits
   one of them with a mask.
3. **Whole-block-skip optimization on sparse bool masks**. Triton has
   the `tl.max(mask_block) == 0 -> skip=True` early-exit on
   all-False BLOCK_N tiles. The sm89 fp8++ CUDA kernel's pipelined
   K-iteration loop (with `cp_async::commit_group` / `wait_group`)
   makes the analog non-trivial -- a runtime skip would either
   serialize the pipeline or require restructuring the loop. Real
   perf win only on sparse masks (LTX-2.3 guide masks are dense);
   defer until a consumer hits a sparse-mask workload where it
   matters.

**Trigger to act on (1) or (2):** the dispatcher routes a masked call
to a surface that hits one of these kernels in production. Today
sm89 + CUDA >= 12.8 routes to the v0.5.5 path; other archs route to
Triton. If a downstream consumer reports masked calls on sm80 (or
forces a non-fp8++ pv_accum_dtype on sm89 with a mask), revisit.

**Trigger to act on (3):** a consumer workflow with sparse (>50%
all-False BLOCK_N tiles) masks reports Triton masked-path wall-time
> 5% of total gen.

### `torch.library.custom_op` registration for fused-pybind kernels

Three pybind kernels in `csrc/fused/pybind.cpp:30-32` cause Dynamo
graph breaks under `torch.compile`, with deterministic precision loss
(0.0276 rtol drift, > 0.01 budget) from partial-graph reordering across
the pybind boundary. Empirically verified on torch 2.11.0+cu130 via
`tests/spike_torch_compile.py` (2026-05-01). The three kernels:

- `_fused.transpose_pad_permute_cuda`
- `_fused.scale_fuse_quant_cuda`
- `_fused.mean_scale_fuse_quant_cuda` (smooth_v=True branch, same
  risk class)

All called from `sageattention/quant.py:281, 289, 292` in the
`per_channel_fp8` V-quant path -- load-bearing on every fp8 sage call
on sm89. Registering each as `torch.library.custom_op` with proper
meta/abstract registrations would let Dynamo trace through them
without graph breaks.

Size estimate: ~1-2 days per kernel (~3-6 days total) including
correctness verification under compile.

**Trigger to act:** consumer's path 1 (CUDA graphs on the LTX
denoiser) fails AND consumer wants path 2 (`torch.compile` the
denoiser) -- at which point the spike's "keep the disable" verdict
needs to flip and this work becomes the gating dependency. Until
then, keep the disable. Re-run the spike after every torch upgrade.

### `arm_kj` synthetic head-to-head VRAM bench

Audio-loop's empirical evidence (2026-05-01 N1-N4 memo) and
sage-side bench data converged on: the 3.5x VRAM gap of sage
`fp8_cuda++` (~628 MiB) vs `torch_flash` (~182 MiB) at the
load-bearing LTX video shape is structural to the int8/fp8 quant
approach, not specific to sage's dispatcher wrapper -- KJ's
per-block path also pre-materializes the same int8/fp8
intermediates before the kernel call.

To falsify conclusively, add a row to
`tests/test_sageattn_ltx_shapes.py` that calls KJNodes'
`ltx2_sageattn_forward` (the per-block sage path in their LTX-2
node module) directly with a synthetic input at the same shape,
measuring working-set VRAM via the same `time_and_vram` helper.
Workflow-arm swap is NOT viable -- DAG trace (2026-05-01)
confirmed `LTX2MemoryEfficientSageAttentionPatch` is NOT in the
consumer's iclora workflow, so the test would have to install +
wire KJNodes synthetically, with skip-if-unavailable handling for
the import.

Size estimate: ~half a day (test row + KJNodes import guard +
skip helper).

**Trigger to act:** the question becomes load-bearing for some other
decision (e.g. "should we restructure sage's Python wrappers to
avoid intermediate materialization?" -- which today the consumer's
evidence already resolves as "no, gap is structural to int8/fp8
quant"). Today: not load-bearing.

### Block-along-T optimization on `fused_rope_split` Triton kernel

`/simplify` efficiency review (2026-05-01) flagged that
`_rope_qk_split_kernel` launches one program per `(t, h, b)` --
733k programs at the LTX video shape (B=1, H=32, T=22932). Each
program does ~1024 bytes total I/O on D//2=64 elements, below the
bf16 cache-line sweet spot. Block along T with `BLOCK_T=8` or `16`
(one program per `(b, h, t_block)`, inner loop over `t`) cuts the
grid 8-16x and amortizes program-launch overhead.

Size estimate: ~half a day (kernel restructure + perf
measurement against a new `tests/bench_fused_rope.py` micro-bench).

**Trigger to act:** a future workflow brings `fused_rope_split`
above 5% of GPU time. Today: 0.55% on the consumer's iclora
workflow (their 21:02Z memo) -- not worth the perf-measurement
work.

### `fused_rope_split` removal candidate

v0.5.3 shipped the primitive on the strength of a comparison-doc
finding ("only structural kernel-side gap vs KJ's per-block
patch") that turned out to overstate the value -- consumer
measured RoPE at 0.55% of GPU time, retracted the ask. Kernel
earns its space as a sage-fork primitive (low maintenance, ~280
LOC self-contained, available for future DiT consumers), but the
immediate ROI is zero. Same disposition as `sageattn_warmup`:
candidate for removal if no consumer adopts within ~6 months.

**Trigger to act:** by 2026-11-01, audit `coderef/` for any
consumer importing `sageattention.fused_rope_split`. If none, drop
the kernel + tests + CLAUDE.md inventory entry in a focused
deletion arc. Lesson: see `feedback_walltime_before_kernel_day`
memory entry -- ask for wall-time contribution before kernel-day
spend on a "kernel-side gap" finding.

## Decision log

Investigations that closed without action. Recorded so we don't
re-derive them. Each entry has an explicit reopen-trigger.

### comfy-kitchen SageAttention port evaluated: stay, adopt one technique (binding boundary)

**Closed 2026-05-23.** An external SageAttention port landed in
Comfy-Org's `comfy-kitchen` kernel library (NVIDIA-authored). It
vendors the same upstream thu-ml sm89 kernel we fork, so the question
was whether its approach obsoletes any of our core components.

Verdict: **stay on every user-facing component; adopt exactly one
technique.** Reasoning, grounded in measurement and the consumer
contract:

- **Kernel config.** The port ships pure-FP32 PV accumulation
  (SageAttention 2; kernel template `use_pv_fp16_accu=false`). Our
  sm89 default is `pv_accum_dtype="fp32+fp16"` (2++). Measured on a
  4090 at the load-bearing LTX shape (`tests/spikes/spike_accum_config_ab.py`,
  commit `5a9c3f4`): **2++ is 1.20x faster (+16.6% wall) at
  indistinguishable mean_rtol** (0.0980 vs 0.0979). Backing away from
  our config would be a measured regression.
- **Masking.** The port is causal/none-only; our v0.5.5 fp8++ kernel
  has the general additive-mask path. A static diff of the vendored
  `.cuh` confirms theirs is upstream-verbatim-minus-torch with no mask
  params; ours carries `DTypeMask` + `mask_ptr`. The downstream
  consumer node's headline feature is masked LTX cross-attn on the
  CUDA kernel -- the port cannot run it.
- **API surface.** The consumer depends on our Python surface
  (`sageattn()`, the named kernel exports, `pv_accum_dtype`,
  `attn_mask`, `get_last_dispatched_kernel()`,
  `core.get_cuda_arch_versions()`, `KNOWN_KERNEL_NAMES`). The port
  exposes one `sage_sdpa(q,k,v,is_causal,smooth_k)` with none of these.
  Convergence is a non-starter; the port can't execute the workload.
- **The one adoptable technique:** its torch-free nanobind/DLPack
  binding boundary (no libtorch ABI coupling, stable-ABI wheels,
  cleaner `torch.compile` story). Captured as roadmap Tier 2.6. The
  consumer-API surface above is the spike's acceptance criterion --
  the swap is invisible iff that surface survives bit-identical.

Also considered and **not adopted**: the port's fused CUDA K-mean
reduction (it fuses `k.mean()` into CUDA; we do it as a torch op then
fuse only the subtraction). Leverage estimate: the K-mean is a
memory-bound reduction of the K tensor (~188 MiB at the load-bearing
shape) -- sub-0.1 ms against a ~20 ms attention kernel, i.e. <1%.
Below the kernel-day threshold; not worth the CUDA surface.

Full comparison + the measured A/B + the verified header diff:
`internal/comfy_kitchen_pr42_comparison.md` (gitignored).

**Reopen-trigger:** the binding-boundary spike (Tier 2.6) fires on its
own triggers (a `torch.compile` consumer ask, an editable-install
breakage on a torch/CUDA bump, or an interop decision). The
stay-on-config verdict reopens only if a future port closes the 1.20x
gap (e.g. ships a 2++-equivalent accum) AND gains a mask path -- at
which point the comparison is re-run.

### v0.6 sage_ffn Cell C verdict (synthetic-vs-production gap concentrates at the kernel boundary)

**Closed 2026-05-19** after a multi-cycle cross-clone diagnostic
session that bottomed out the consumer-side integration chain (six
distinct bugs surfaced and fixed: scale-lookup probe, QuantizedTensor
unwrap, `prior_forward` chaining, `str(exc)` logger, call-time weight
resolve + device guard, `.item()` hoist for scalar ABI). With the
chain fully closed, sage_ffn dispatches end-to-end -- rung-1 evidence
(`_fp8_matmul_gelu_kernel` + `_fp8_matmul_kernel` rows in `cat=kernel`
of every TREATMENT chrome trace) confirms.

The verdict is **Cell C** of the synthetic-vs-in-pipeline 2x2 matrix
(defined in `docs/perf_research_framework.md`) at *both* wall-time
and per-stage-kernel-time levels:

| measurement | sage_ffn | stock (prod) | ratio |
|---|---:|---:|---:|
| wall time, mean of 3 runs each | 188.3s | 185.7s | +1.4s (+0.75%) |
| stage-1 (T=10780) FFN kernel | 8930 ms | ~7290 ms | sage 22% SLOWER |
| stage-2 (T=42240) FFN kernel | 12920 ms | ~12300 ms | sage 5% SLOWER |

Synthetic isolation at the same shapes (v0.6.5 `tests/bench_sage_ffn_shapes.py`):
sage_ffn 1.39x faster at T=10780, 1.60x at T=42240. **Production has
the sign flipped** -- largest synthetic gain is the smallest production
gap; smallest synthetic gain is the largest production loss. Inverted
relationship.

Two open hypotheses for the inversion (neither preferred yet):

  (1) Stock comparand identity -- synthetic compares vs
      `torch._scaled_mm`; production stock is `comfy.ops.fp8_linear`
      wrapped in KJNodes' `LTXVChunkFeedForward`. Different baseline.
  (2) Sage autotune state under interleaving -- production interleaves
      sage attention + sage_ffn at varying shapes; autotune may pick
      a different tile config than synthetic isolation.

Either explains the gap; neither is a sage correctness bug.

**Cross-workload corroboration of hypothesis (2) (added 2026-05-19
post-verdict):** A separate consumer-side workflow (two-pass tensor-
loop sampler, NOT the FML2V workload that produced the original
verdict) surfaced direct evidence of autotune flipping between
contexts. The smoking gun: attention kernel time per call at two
nearly-identical sequence lengths, same workflow, same render:
T=7560 (n=1896 calls) measures 15.7 ms median; T=7800 (n=711 calls)
measures 33.6 ms median. 3% sequence-length increase, 2.14x kernel
slowdown -- not shape-linear. Candidate explanations (different
batch dim, different head dim under interleaving, L2 contention with
larger working set) all reduce to "the kernel got a different tile
config for two structurally similar calls because something in the
(B, H, T, D, neighbor-state) tuple shifted." That is hypothesis (2)
firing visibly. Strengthens it from "plausible explanation for one
observed gap" to "documented failure mode across multiple workloads."
Roadmap Tier 2.2 (autotune pre-bake) gains a concrete justification
beyond UX (cold-render lag) -- pre-bake addresses production-perf
variance, not just first-render-per-shape lag.

**Disposition:** v0.6 sage_ffn "ships as completeness primitive, not
perf win" framing reaffirmed. Diagnostic instrumentation work paid
off -- the chain that closed the verdict spanned v0.6.2 (informative
asserts), v0.6.3 (dispatch log), v0.6.4 (`extract_fp8_weight_and_scale`
+ framework rung 2 with pattern (d)), v0.6.5 (`w*_scale` precondition
assert + stage-1/stage-2 bench rows), and `70d1984` (framework rung 2
pattern (e) fold). Each new failure mode after v0.6.2 was diagnosed
in 1 render rather than the multi-cycle loop earlier.

**Reopen-trigger:** persistent-CTA hybrid lands (Backlog item) AND
re-render shows wall-time improvement OR per-stage parity-or-better
vs production stock. At that point Cell C would close to Cell A and
v0.6 docstring framing flips. Alternative reopen: a `comfy.ops.fp8_linear`-
direct bench (hypothesis 1) lands data showing sage_ffn beats the
production comparand -- routes investigation to autotune-state
(hypothesis 2) and may not need persistent-CTA at all.

Memo trail: `coderef/ComfyUI-AudioLoopHelper/internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md`
(2026-05-19T08:00Z verdict memo, audio-loop-side outbox-mirror).

### "FFN-adjacent reach" / launch-overhead / cache-footprint hypotheses on iclora: all three falsified

**Investigated 2026-05-07** (cross-claude bounded investigation
with AudioLoopHelper claude; memo trail in
`internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md` +
`internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md`).

The v0.5.1 entry's claim that sage's e2e speedup "extends beyond
per-call attention rows into FFN-adjacent amortization within the
sampler step" was a single-data-point inference from the
audio_loop_latent workload (arm-2 attention time was never directly
traced; +17pt above strict Amdahl was attributed to FFN-adjacent
mechanism without measurement).

A clean sage-on/sage-off A/B on the iclora workflow (consumer's
production-scale workload, profiler on, matched 3456 attention calls
per arm) decomposes the savings cleanly:

| Mechanism | Savings | Status |
|---|---|---|
| Attention kernel time (sage 7.20ms vs torch flash 22.14ms) | -51.6s | confirmed (dominant) |
| Non-attention named CUDA kernels | +2.1s | sage-on slightly slower |
| cudaLaunchKernel call-count delta (0.82%) | -0.4s | negligible |
| Wall-clock vs CUDA-time gap (CPU-side) | -3.9s | not GPU-side |

Strict Amdahl with iclora's actual attention share (~42% of CUDA
time) and the actual per-kernel ratio (3.08x) predicts 1.39x e2e
speedup; measured wall ratio is 1.41x -- match within 1.4%. **No
non-Amdahl mechanism is required on iclora.**

Three hypotheses closed:
1. **Launch-overhead reduction** ("sage replaces torch SDPA's
   decomposed-op path with one fused kernel, saving ~6-10 launches
   per attention call"): falsified. Sage-off is already routed
   through `aten::_scaled_dot_product_flash_attention` -- one fused
   launch per call. Sage replaces flash with sage; one-fused-for-
   one-fused. Total launch delta is 2205 of 270k (0.82%).
2. **FFN-adjacent reach via int8 amortization**: falsified on
   iclora. Non-attention CUDA kernel time is essentially identical
   sage-on vs sage-off (delta -2.1s, in the wrong direction).
3. **Cache-footprint helping adjacent matmul/elementwise**:
   falsified on iclora. If int8 K + fp8 V freed L2/HBM bandwidth
   for adjacent ops, the named matmul kernels would be faster
   sage-on; instead they're 0.13-0.73s slower per kernel (sage's
   own quant work pollutes their cache lines).

**Decision:** retire all three from sage-fork's mental model on
iclora. The original v0.5.1 +17pt residual on audio_loop_latent
remains unexplained but is single-data-point and measurement-
methodology-dependent (no arm-2 attention tracer); not load-
bearing for any current decision.

**Trigger to reopen:**
- A future workload measures non-trivial sage-on-vs-off non-
  attention kernel-time delta (>5s on a render of similar scale to
  iclora's 180s sage-off baseline). Direct evidence of any of the
  three mechanisms operating.
- The audio_loop_latent +17pt gets re-measured with arm-2
  attention tracing in place, confirming it's real and not an
  inference artifact. If real and reproducible, the mechanism
  question reopens for that specific workload.

**Process note:** the pre-committed-prior-and-decision-rule
discipline (recorded in CLAUDE.md / "Pre-trigger briefing
pattern") fired correctly twice in this exchange. Both times an
investigation produced a "your hypothesis is wrong" outcome
without consuming downstream code-change budget. Confirmed as
default for future cross-claude bounded investigations.

### sm89 fp8 quantization scale: closed as no-action

**Investigated 2026-04-23.** `sageattention/core.py:772-774` keeps
the fp8_cuda `scale_max` default at `448.0` for `pv_accum_dtype`
variants `"fp32"` and `"fp32+fp32"`, but flips to `2.25` for
`"fp32+fp16"` (the ++ variant, which is what sage's auto dispatch
picks on sm89 + CUDA >= 12.8). KJ's `LTX2MemoryEfficientSageAttentionPatch`
hard-codes `2.25`; the reasoning suggested flipping the non-++ default
to match, for consistency.

Measured via `tests/test_sageattn_ltx_shapes.py` on both LTX shapes
(V ~ N(0, 1)) and a synthetic wide-V shape (V ~ N(0, 5)). fp8_cuda
(`scale_max=448`) and fp8_cuda++ (`scale_max=2.25`) produced
essentially identical mean_rtol: 0.097 on LTX self-attn, 0.097 on
synthetic wide-V. No material difference.

**Decision:** don't flip the default. Two reasons:
1. Auto-dispatch already picks ++ on sm89 + CUDA >= 12.8, so the
   non-++ default only affects callers who explicitly choose
   `pv_accum_dtype="fp32"` or `"fp32+fp32"`. Those callers likely
   picked the older variants for a reason; silently changing v-quant
   behavior on them is worse than matching upstream.
2. The measurement showed equivalence, not improvement. Upside of
   flipping is zero; downside is silent divergence from upstream for
   explicit non-++ callers.

**Trigger to revisit:** a future model or workload shows measurable
quality improvement from `scale_max=2.25` on the non-++ path. Until
then, this is closed.

### Sage 3 per-block Q mean backport: closed as low-impact on sm89 fp8++

**Investigated 2026-04-24** (research spike, not a full backport).

Sage 3's `sageattention3_blackwell/sageattn3/api.py::preprocess_qkv`
adds a preprocessing step sage 2.x lacks: Q is split into groups of
128 tokens along the sequence dim, each group's mean is subtracted
before quantization, and a `delta_s = qm @ K^T` correction tensor is
passed to the kernel for use during softmax reconstruction. Sage 2
only centers K (via `smooth_k=True`) and never centers Q at all.

For FP4 (sage 3's quant format, 16 levels) this is a first-order
precision win. Question: is it worth backporting to sage 2's sm89
fp8++ path, which uses INT8 Q (256 levels)?

**Empirical check** via a standalone INT8 Q quant-roundtrip experiment
(per-block quantize -> dequantize, measure rtol to original fp32):

| Q distribution    | |DC|/std | rtol_baseline | rtol_centered | improvement |
|-------------------|----------|---------------|---------------|-------------|
| typical           | ~0.16    | 0.0363        | 0.0331        | ~9%         |
| skewed (large DC) | ~0.80    | 0.0351        | 0.0252        | ~28%        |

That is Q quant precision. Translating to end-to-end attention rtol
(fp8++ currently measured at ~0.097 vs SDPA on LTX shapes): the Q
quant floor is roughly a third of total rtol budget (rest is FP8 V +
accumulation). A 9% improvement on Q yields ~2-4% end-to-end rtol
improvement -- well below the run-to-run noise floor. A 28%
improvement on Q (for a skewed model) yields ~8-10% end-to-end.

**Decision:** don't backport. Three reasons:

1. LTX's Q activations almost certainly fall into the "typical" DC
   range (normalized transformer activations with modest channel
   biases), where the expected fp8++ rtol improvement is ~0.002-0.004
   absolute -- imperceptible at render level.
2. The kernel-side work is non-trivial: `csrc/qattn/sm89_qk_int8_sv_f8_*.cu`
   would need a new `delta_s` input, modifications to the softmax
   reduction to apply the correction, and matched changes in the
   Python quantization path (`sageattention/quant.py`) to compute and
   pack per-block Q means. Days of work for a sub-noise-floor win on
   our primary workload.
3. Sage 2's existing `smooth_k` already captures the K-centering
   half of the idea. The specific new capability (per-block Q mean)
   is the marginal addition, not the main event.

**Trigger to reopen:**
- LTX's actual Q DC offset measured at |DC|/std > 0.5 in production.
  (Would require instrumenting the LTX model's Q projections;
  a downstream consumer could capture this if its telemetry grows that far.)
- A visible artifact in fp8++ output that isn't explained by bf16
  activations, fp8 weight quant, or VAE.
- A future workload with shorter-bit Q quantization (fp4 or below)
  where per-block mean becomes a first-order win rather than a
  third-order refinement.

## Workload intel

Cross-workload observations worth keeping for future bench-coverage,
autotune, and roadmap decisions. Not actionable today; filed so
they're discoverable when a decision in that space surfaces. Pair
with `docs/ltx_workload_profile.md` (canonical FML2V breakdown).

### Two-pass tensor-loop workflow: mixed head dims (HEAD-128 + HEAD-64)

Observed 2026-05-19 via cross-clone trace analysis on a consumer-side
two-pass sampler workflow (distinct from the FML2V workload the
load-bearing metric is anchored to). 1920 total sage attention calls
per trace: **1536 at HEAD-DIM 128, 384 at HEAD-DIM 64**. The HEAD-64
specialization corresponds to a smaller-dim pass (audio cross-attns
or upsampler stage). Two sage template instantiations co-fire in the
same render.

Implications worth keeping:

- **Bench coverage gap.** `tests/test_sageattn_ltx_shapes.py` covers
  HEAD-DIM 128 LTX shapes; HEAD-DIM 64 isn't currently in the sweep.
  Cold-render UX on two-pass workloads pays a HEAD-64 autotune sweep
  on the first render per shape. If two-pass workloads become
  recurrent, add HEAD-64 rows to the bench.
- **Autotune pre-bake (Backlog item) scope.** If pre-bake ships, the
  cached configs should cover both head dims; otherwise the pre-bake
  benefits only one of the two template instantiations active in this
  workload.
- **Cell C hypothesis (2) corroborating evidence.** Same trace
  exhibited a 2.14x attention-kernel slowdown for a 3% sequence-
  length increase (T=7560 vs T=7800) -- direct evidence of autotune
  flipping under interleaved dispatch. Captured in the Cell C
  decision log entry.
- **Validation surface for persistent-CTA (Tier 1.3) and §6.1
  concurrent-dispatch** (added 2026-05-19, per audio-loop-side
  memo). The same trace also exposes:
  - dual HEAD-DIM specialization co-firing (~80/20 split HEAD-128 /
    HEAD-64) in a single render -- two sage template instantiations
    competing for autotune cache slots.
  - cross-modal attentions (`audio_to_video_attn`,
    `video_to_audio_attn`) contributing ~36% of wall (~22s of 61s
    in the failed render that produced the trace) -- non-trivial
    share.
  - the T=7560 vs T=7800 anomaly noted above.

  If persistent-CTA work lands tile-config-cache-sticky behavior
  (the v0.6 Backlog framing of CTAs holding M-tile state in
  registers/L2 across the pipeline), OR if §6.1 concurrent-dispatch
  ships and launches attn + ffn on different streams, **the dual-
  HEAD-DIM mix in this workload class is exactly where the tile-
  config thrashing or cross-stream coupling would surface**. Worth a
  check against this trace once either kernel-side wedge is ready.

**Trigger to act:** a second cross-workload observation of HEAD-64
sage dispatches surfaces (suggests two-pass workloads are recurrent
rather than one-off), OR autotune-pre-bake (Backlog item) ships and
needs to decide which head dims to cover, OR persistent-CTA / §6.1
lands a candidate and needs a multi-template-instantiation workload
to validate against.

## Recurring process items

Cron-like checks; not engineering work. Each one has a frequency or
trigger; act when the trigger fires, otherwise note in passing.

### Bench env re-snapshot

Process item, not engineering work. The `tests/test_sageattn_ltx_shapes.py`
baselines are pinned to a specific (torch, triton, CUDA, sage) version
tuple recorded in `internal/bench_env_<date>.txt`. Re-run the test with
soft-warn enabled and re-snapshot the env file when ANY of these change:

- `torch` major or minor (e.g. 2.11 -> 2.12, or any cuXYZ swap)
- `triton` minor (e.g. 3.6 -> 3.7)
- CUDA toolkit (e.g. 13.0 -> 13.1)
- `sage` git rev — automatic on every `./build.sh`, but worth a fresh
  bench if the rev changed since last measurement

What "real change" means: any (shape, mode) wall-clock that drifted >5%
from the previous snapshot is worth investigating. <5% is run-to-run
noise (we logged 1.4% on the cu128 -> cu130 transition). The rtol
fingerprints (e.g. cross-attn-with-mask 0.94 at kv=32) should be
*invariant* across these upgrades; if they drift, that's a kernel change
upstream and warrants tracing back.

The snapshot file lives under `internal/` (gitignored). Naming convention:
`bench_env_YYYY-MM-DD.txt`. Keep prior snapshots; they're the audit trail
when someone says "this used to be faster."

**Trigger to act:** any version bump in the list above, OR a measured >5%
shape-level drift on a routine workflow gen.

**Open observation (2026-05-07):** synthetic LTX bench
(`tests/test_sageattn_ltx_shapes.py`) reports `torch_flash / sage_fp8++`
= **2.66x** at the primary shape; the iclora production A/B (cross-
claude memo trail, 2026-05-07) measured the same kernel pair at
**3.08x** averaged over 3456 cache-warm calls. The 16% gap is unaddressed
but probably not a real kernel-side change. Most likely candidates,
roughly ranked: (1) median-of-3 (synthetic) vs sum/calls average over
3456 calls (production) is a different statistical animal — long-tail
distributions on warm production state can shift the central estimate
10-15% even on identical shapes; (2) cache state (synthetic short-burst
vs production sustained); (3) driver-thermal asymmetry across kernel
types under sustained load; (4) torch/triton/CUDA drift between when the
synthetic bench was last snapshotted and when iclora ran. Action: log
this gap on the next bench env re-snapshot. If the synthetic ratio shifts
toward 3.08x without a kernel-side change, hypothesis (1) is the answer
and the synthetic bench's median-of-3 protocol may be worth widening.

### Session-level attention telemetry summary (consumer side)

Cross-repo backlog item, tracked here because it feeds sage-fork's
mask-kernel work (the "is triton cross-attn a bottleneck?" trigger
above). A consumer-side sage-routing node typically writes a per-call
JSONL row (shape, mode, effective_mode, elapsed_us, fell_back) when an
opt-in trace env var is set. Raw per-call data is straightforward; the
aggregation question is "what percent of gen wall time is masked-triton?"
which is exactly what the mask-kernel trigger above needs.

**Shape of the work (consumer side):** emit a one-line summary at
gen-end: median/p90 elapsed_us for masked-triton calls, total call
count, and that median as a percent of total gen time if measurable.
No new telemetry plumbing required -- just aggregation over the
existing JSONL rows.

**Trigger to act:** when a downstream consumer wants to justify backing
a sage-fork kernel push with data. Until then the raw JSONL is
sufficient.

## Versions

### v0.7.5 -- 2026-08-11  (`sageattn_consume` speaks ComfyUI's container protocol)

ComfyUI added a single-owner container protocol for attention inputs on
2026-08-10 (`bf4c9a08`): `AttentionTensorContainer` with `peek()`/`take()`,
a `container_function` hook that a backend sets to receive containers
rather than tensors, and `wrap_attn` calling `take()` itself for backends
that do not. Its H3 model wraps q/k/v in them on every call, and the
`v = v.clone()` directly above that wrapping exists for the same reason
ours does: so `take()` on v frees something.

That is the ownership transfer `sageattn_consume` performs with a list it
empties, arrived at independently on both sides, and it is now the
ecosystem's shape rather than ours. `sageattn_consume` therefore accepts
either: a `[q, k, v]` list, or three objects exposing `peek()`/`take()`.
Containers and list slots are both emptied.

Taking them here rather than making the caller unwrap is the entire point.
Unwrapping into a list binds all three tensors in the caller's frame for
the duration of the call, which is the retention this entry point exists
to avoid, and which was measured three separate ways this week: the clone
before the branch, the slice-and-loop suppression, and a downstream
override that re-pinned tensors core had already handed it sole ownership
of. An adapter written by the caller would have reintroduced it.

A mix of tensors and containers is rejected, and so is a container already
spent, because a `RuntimeError` escaping `take()` into a quant kernel reads
as a kernel fault rather than a caller error.

**The discriminator is `not isinstance(t, torch.Tensor)`, not `hasattr(t,
"take")`.** `torch.Tensor.take(index)` exists, so duck-typing on the method
name matches every tensor. The first draft did exactly that and
`test_matches_sageattn_output` failed on the first case. No ComfyUI import
either way; containers are structural here.

Three test cases. Two use a local stand-in and carry the coverage; the
third imports the real class, resolved via `$COMFYUI_ROOT` or
`comfyui_root` in `internal/local_config.json`, and exists to fail the day
ComfyUI renames `take()` or changes what a spent container does. It runs
rather than skips on this box, which is the only thing that makes it worth
having. Peak behaviour is inherited rather than re-measured: after the take
the path is identical to the list path, and the cases assert the containers
end up empty, which is the property the saving depends on.

Also in this version, no code change: `sageattention/triton/_int_offsets.py`
said the int32 overflow becomes reachable "past roughly 300k packed rows."
That is the contiguous stride, `heads*head_dim` = 7168. A DiT block hands
these kernels three views of one qkv projection, where `stride_n` is
`3*heads*head_dim` = 21504 and the crossing lands near 99,864 rows instead.
`max_element_offset` reads real strides so the specialization always fired
correctly; only the prose was wrong, and it was wrong in the direction that
reassures. v0.7.1 measured the specialization engaging at S=109,126, which
a reader of that sentence would have filed as comfortably safe. Both
thresholds are now stated. Found while reviewing the identical defect in a
third-party node's overflow warning; ours was in a docstring rather than a
check, which is why nothing caught it. Prose has no oracle.

### v0.7.4 -- 2026-08-11  (the fused case has a caller-side fix, and a predicate to gate it on)

v0.7.3 said the fused-QKV case saves nothing and framed the only remedy as
work inside this library. Both true, and both beside the point for the
people calling it: a caller can clone v before handing the list over, which
gives v its own storage so releasing q and k actually frees the fused
buffer. It converts the fused case into the separate case for the price of
one third of the buffer. A downstream consumer shipped for a week without
the clone because the docstring read as "nothing to do here".

Same shape, same protocol as v0.7.3's table (S=41822, heads 56, head_dim
128, bf16, sm89, consume arm first, one arm per process):

| allocation | `smooth_k` | `sageattn` | `sageattn_consume` | saved |
|---|---|---|---|---|
| separate | False | 3148 MiB | 2290 MiB | -858 |
| separate | True | 3148 MiB | 2862 MiB | -287 |
| fused QKV views | False | 3148 MiB | 3148 MiB | **0** |
| fused QKV views | True | 3148 MiB | 3148 MiB | **0** |
| fused, caller clones v | False | 3720 MiB | 2862 MiB | -858 |
| fused, caller clones v | True | 3720 MiB | 3434 MiB | -287 |

Read the last two rows against the 3148 a fused caller gets today, not
against the 3720 in their own `sageattn` column: **-286 MiB** with
`smooth_k=False`, and **+286**, i.e. worse than not cloning, with
`smooth_k=True`. Both halves are load-bearing and the failure modes are not
symmetric. Cloning without consuming is a flat +572 MiB. Consuming with
`smooth_k=True` hands the clone straight back, because `per_thread_int8`
allocates the int8 outputs before evaluating `k = k - km`.

Not a hypothetical: ComfyUI added the same clone to its own H3 model on
2026-08-11 (`62b3c94b`, "Fix peak memory issue with H3", #15486), so anyone
running H3 through core's attention path is in the cloned column already.

**New API: `sageattn_consume_prefers_cloned_v(device=None) -> bool`.**
Whether cloning pays on that device. Callers were otherwise going to copy
`{"sm89", "sm100", "sm120", "sm121"}` out of our dispatch, which drifts
into a silent memory regression rather than an error. Deliberately shaped
as the caller's question ("should I clone?") rather than ours ("do we
release?"): those coincide today and come apart the moment the backlog's
transpose-buffer item lands, at which point the release still happens while
cloning becomes a 286 MiB cost. Behind the predicate that is a flip we make
and consumers inherit on upgrade; behind `releases()` it would be a
truthful answer to the wrong question. `smooth_k` and fused-vs-separate
stay out of it -- the caller owns both and would only make a False
ambiguous. One arch set, `core._EARLY_RELEASE_ARCHS`, read by the predicate
and the dispatch alike.

**Testing.** `tests/test_sageattn_consume.py` gains a `clone_v` axis, two
published rows, and three asserting cases. The clone case is asserted where
the neighbouring matrix deliberately only publishes, because the asymmetry
warrants it: if our release timing regresses the clone does not merely stop
paying, it becomes a 572 MiB penalty. Shown red both ways before being
trusted -- clone removed reports +0 MiB, an extra owner holding the list's
contents reports exactly -572, which is also what confirms the mechanism.
The predicate's case mutates `_EARLY_RELEASE_ARCHS` to prove it reads that
set rather than restating it; hardcoded, it would stay green with the set
emptied.

Contributed by the downstream consumer's clone (docstring rows and the
clone case, `6e60b4f`), reviewed and extended here.

### v0.7.3 -- 2026-08-06  (correction: `sageattn_consume` saves nothing in the configuration DiT blocks actually use)

No code change. This retracts a number v0.7.0 shipped and replaces it with
the full measurement.

v0.7.0 recorded `sageattn_consume` at -858 MiB peak, with a caveat putting
the fused-QKV case at "~435 MiB with the peak set inside `per_channel_fp8`
instead." The -858 is reproducible but describes a configuration no
consumer runs. The ~435 is simply wrong. Measured at fl2va (S=41822, heads
56, head_dim 128, bf16), peak per call, consume arm first so allocator
state cannot bias it:

| allocation | `smooth_k` | `sageattn` | `sageattn_consume` | saved |
|---|---|---|---|---|
| separate | False | 3148 MiB | 2290 MiB | -858 |
| separate | True | 3148 MiB | 2862 MiB | -287 |
| fused QKV views | False | 3148 MiB | 3148 MiB | **0** |
| fused QKV views | True | 3148 MiB | 3148 MiB | **0** |

Production is the bottom row. Two independent axes, neither obvious from
reading the function:

- **`smooth_k` defaults to True** (`core.py:948`), and `per_thread_int8`
  allocates `q_int8`/`k_int8` *before* evaluating `k = k - km`
  (`quant_per_thread.py:176-180`), so a full bf16 copy of K is live on top
  of the int8 outputs. That alone costs 571 of the 858 MiB.
- **Fused q/k/v erases the rest.** `qkv_proj(x).split(...)` leaves the three
  as views over one allocation, so releasing q and k frees nothing, and by
  the time v is released `per_channel_fp8`'s full-size bf16 transpose buffer
  has already set a higher peak.

Isolated against the obvious confound: the fused arm also runs NHD, so
separate-NHD was measured as the missing cell and returned -858 / -287,
identical to separate-HND. Layout is not the cause; fusion is.

**Consequence for the backlog.** Dropping `per_channel_fp8`'s transpose
buffer and doing the mean-subtraction in place are not two independent
wins -- in the fused case each alone leaves the other setting the floor
(transpose alone 3148 -> 2859, both together 3148 -> 2573). They have to
ship as one change or neither is worth doing.

Not retracted by `git revert` per the usual convention: the wrong number
rode in on `bbd5fca`, which is the feature commit, so reverting would
remove `sageattn_consume` itself. The forward-facing surfaces
(`core.py` docstring, CLAUDE.md consumer surface) are corrected in place
and point here.

Why it survived three months: `tests/test_sageattn_consume.py` asserted a
single threshold at `smooth_k=False` with separate tensors. It passed, and
a passing test that measures the wrong configuration reads as
verification. The test now publishes all four arms and deliberately
asserts no threshold on the matrix -- see the measurement-validity rule in
CLAUDE.md, of which this is the fourth logged instance.

### v0.7.2 -- 2026-08-06  (ragged tails no longer inherit stale shared memory)

`csrc/qattn/attn_utils.cuh:131` issued its predicated `cp.async` with
`SharedMemFillMode::kNoFill`, so the out-of-range lanes of a ragged K/V
tile kept whatever the previous launch left in that shared-memory bank.
Flipped to `kFillZero`. Free: the mode is a compile-time `if constexpr`
in `csrc/cp_async.cuh` and zero-fill is a native `cp.async` operand, not
an added store.

Same defect `woct0rdho/SageAttention` fixed in `e147939` (2026-07-16),
which this fork had not picked up. Found independently here; the
measurement narrows the characterization in three ways worth recording,
because each one changes how the defect should be tested for.

**It does not fire in a fresh process.** On the unfixed kernel the
short-length sweep passes, worst mean_rtol 0.0345. The defect needs a
*previous* launch to have left non-finite values in shared memory. A
regression test that does not deliberately dirty shared memory first is
green against a broken kernel; `tests/test_short_seq_tail.py` dirties with
`+inf` before every sweep for exactly this reason.

**The boundary is one K tile, not "non-multiples".** Dirtied, every
`kv_len < 64` returns 100% non-finite output. `kv_len >= 64` is clean
including ragged 127 / 385, because past the first full tile the stale
lanes hold this launch's own earlier rows -- finite, and softmax weights
them p=0 so they contribute nothing. Only non-finite residue survives the
multiply.

**The exposed path is sm80 fp16, not the sm89 default.** On sm89 the
predicated loader carries K only (int8, cannot be non-finite) and the
out-of-range columns are masked before softmax, so there the flip is
defense-in-depth -- it removes the dependency on that mask being present
in every variant. On sm80, `qk_int_sv_f16_cuda_sm80.cu:257,351,449` puts
*V* through the same predicated load as fp16, and P@V has no mask to hide
it. `sageattn_qk_int8_pv_fp16_cuda` is an exported consumer entry point
and is forward-compatible onto Ada, so it is reachable here.

Related question closed while validating: the sm89 fp8 V load
(`load_fp8_V_global_to_share`, `qk_int_sv_f8_cuda_sm89.cuh:266`) takes no
predicate at all, and the upstream comment at line 263 warns it "assumes
that V is padded." It is, explicitly and at two levels --
`per_channel_fp8` allocates the transposed buffer at
`(kv_len + 63) // 64 * 64` (`sageattention/quant.py:273-279`), and
`transpose_pad_permute_cuda` instantiates `TransposePadPermuteKernel` with
`pad_zero=true`, which selects `kFillZero` at `csrc/fused/fused.cu:293`.
The pad rows are hardware-zeroed, so the unpredicated read lands in
deliberately zeroed memory rather than out of bounds. Verified at
production scale: paired exact/ragged shapes at H3's config (56 heads,
128 dim) -- 37760/37810 and 109120/109126, the latter being the 362-frame
render shape with a 6-row tail -- all zero non-finite outputs. This holds
for V routed through `per_channel_fp8`; a consumer that hand-rolls
quantization and passes an unpadded `v_fp8` straight to the pybind entry
point would hit the unguarded read.

Test: `tests/test_short_seq_tail.py`, 5 cases, red before / green after.
Two are controls that pass in both states by design and say so.

### v0.7.1 -- 2026-08-05  (long-sequence validation: the v0.7.0 overflow fix holds at 362 frames, and the int64 path is free)

No code change. This closes a measurement gap that v0.7.0 left open and
corrects an env pin.

**Every sage number on file stopped at S=41,822.** That is MiniMax H3's
fl2va shape at the node's default 124 frames, and it is 2.9x shorter than a
362-frame request at the same 1344x768 canvas, which packs S=109,126 rows.
v0.7.0 fixed the int32 offset overflow and reasoned about where it bites,
but the fix was never exercised at a length a user actually asked for --
and 362 frames is past the boundary in the layout production uses, where
q/k/v are three views of one fused QKV buffer and `stride_seq` is 21,504
rather than 7,168.

`tests/spikes/spike_h3_long_sequence.py` measures both layouts at four
lengths. The contiguous arm is the control: same S, half the stride, so the
specialization does not fire and the two arms differ only in addressing.

| frames | S | layout | int64 | sage fp8++ | flash | ratio | mean rtol | tail cos | tail zeros |
|---|---|---|---|---|---|---|---|---|---|
| 124 | 37,774 | fused | no | 90.1 ms | 253.5 ms | 2.81x | 0.0981 | 0.9992 | 0.0% |
| 209 | 63,256 | fused | no | 256.0 ms | 708.2 ms | 2.77x | 0.0980 | 0.9992 | 0.0% |
| 311 | 93,836 | fused | no | 556.5 ms | 1560.3 ms | 2.80x | 0.0985 | 0.9992 | 0.0% |
| 362 | 109,126 | fused | **yes** | 757.7 ms | 2107.9 ms | 2.78x | 0.0978 | 0.9992 | 0.0% |
| 362 | 109,126 | contig | no | 757.2 ms | 2112.4 ms | 2.79x | 0.0981 | 0.9992 | 0.0% |

Three results:

**The fix works where it fires.** The specialization engages exactly at the
predicted boundary -- fused at 362 frames and nowhere else -- and the output
is correct there. The tail is scored separately on purpose: the NHD failure
signature is a silently all-zero tail, which a whole-tensor mean rtol can
absorb. Tail cosine is 0.9992 and the zero fraction is 0.0% at every length.

**The int64 path costs nothing.** 757.7 ms with it against 757.2 ms without,
same S, a 0.07% difference that is inside run-to-run noise. v0.7.0 verified
the specialization did not slow down shapes that skip it; this is the other
half of that claim, and it means the `needs_int64_offsets` check could be
dropped for a simpler unconditional int64 if it ever became a maintenance
burden. Addressing is not what these kernels are bound on.

**The kernel ratio is flat in sequence length.** 2.77-2.81x across a 2.9x
span of S. Sage does not degrade on long clips; it is the same multiplier,
applied to a quadratically larger number.

Attribution, since these tables invite the wrong reading: **that ratio is
upstream's kernel**, the sm89 INT8-QK / FP8-PV design from `thu-ml` via
`woct0rdho`, which we ship unmodified (`csrc/qattn/sm89_qk_int8_sv_f8_*.cu`
is on the upstream-unmodified list). This fork made none of it faster. What
v0.7.0/v0.7.1 contribute at this length is that the kernel *builds* for
sm89 and stays *correct* past 99,864 rows -- and the defect being fixed is
upstream's too. Independent corroboration that it is not obscure: kijai hit
the same int32 wrap in Sol-Attn's own Triton kernels on 2026-08-04
(`9cab9a0`) and patched it the same day from a different direction.

**Reconciliation with a production render.** A 362-frame render logged
20 steps at 49.66 s/it (workflow `h3_t2v_sage_ui.json`, 1344x768, euler /
simple, 20 steps, sage mode `auto`). H3 runs one self-attention per block
over 50 blocks, so the measured 757.7 ms per call predicts 37.9 s of
attention per step -- **76% of the step**. Carrying the non-attention
remainder from the 124-frame configuration forward linearly in S predicts
51.0 s/it against 49.66 s observed, a 2.7% error. The step is accounted
for; nothing is silently falling back.

That share is the useful number. At 124 frames attention is ~50% of the
step; at 362 it is 76%, because attention grows as S^2 while everything
else grows as S. Substituting the measured flash time for sage at the same
length puts the same step at ~118 s/it, so sage is worth roughly 39.5 min
against 16.6 min on this render -- and any further work on long-sequence H3
should be aimed at attention, since three quarters of the clock is there.

**Env pin correction.** v0.7.0 cites e2e numbers without recording a stack,
against the rule in CLAUDE.md. Its measurements were taken on
`torch 2.13.0+cu132` / `triton 3.7.1`, not the `2.12.1+cu132` that v0.6.6
recorded -- the `.so` files and torch were installed in the same session on
2026-07-11 and have not moved since. The tables above are on the same stack.
Fresh snapshot: `internal/bench_env_2026-08-05.txt` (the previous one was
from 2026-04-25 and still said torch 2.11).

### v0.7.0 -- 2026-08-04  (MiniMax H3 coverage: mask-probe fix, quant overflow fix, `sageattn_consume`)

Triggered by MiniMax H3 (`Comfy-Org/MiniMax-H3`) landing in ComfyUI. The
model turned out to need no kernel work at all -- 56 heads, head_dim 128,
unmasked, one self-attention per block over a ~42k-row packed sequence,
which the existing sm89 fp8++ kernel covers as-is at 4.10x over
EFFICIENT_ATTENTION and 2.70x over torch flash. Probing it did surface
three real defects, two of them inherited from upstream.

**1. `attn_mask` was invisible to ComfyUI's capability probe.** ComfyUI
decides whether sage may see masks by introspecting our signature:

```python
SAGE_ATTENTION_SUPPORTS_MASK = "attn_mask" in inspect.signature(sageattn).parameters
```

Ours arrived via `**kwargs`, so this read `False` and `attention_sage`
routed every masked call to `attention_pytorch`. The v0.5.5 native CUDA
mask path has therefore never run in production -- masked LTX cross-attn
has been on torch SDPA the whole time, regardless of kernel correctness.
`attn_mask` is now a named parameter, added after `return_lse` so the
positional prefix is unchanged. Verified end to end through comfy's
`attention_sage`: probe reads `True`, masked calls land on `fp8_cuda++`,
rtol 0.0904 vs torch on an LTX-shaped masked cross-attn.

This changes which kernel runs for masked calls in production. The kernel
is unchanged and was measured at v0.5.5; the e2e effect of actually
reaching it is not yet validated in-pipeline.

**2. int32 element-offset overflow in the INT8 quant kernels** (upstream
defect, `thu-ml/SageAttention`). Triton's `program_id` is int32 and the
stride arguments are passed as int32, so
`off_b*stride_z + off_h*stride_h + offs_n*stride_n + offs_k` is evaluated
in int32 and wraps past 2**31 elements (4 GiB at bf16). The two layouts
fail differently and neither warns:

| layout | first term to cross | measured failure |
|---|---|---|
| NHD | `stride_n` = heads*head_dim | silently all-zero tail (tail cosine +0.0000 at S=303689) |
| HND | `stride_h` = seq*head_dim | `CUDA error: an illegal memory access` at S=310000 |

Fixed in `quant_per_thread.py` and `quant_per_block.py` via a `USE_I64`
constexpr chosen per launch, so ordinary shapes keep int32 addressing --
verified no change to `fp8_cuda++` wall-clock or rtol at the production
shape. Promoting the base offset matters: promoting only the row index
(as KJNodes' vendored copy does) still faults in HND.

Reachable by parameter on MiniMax H3 -- 1008 rows/frame at 1344x768 means
~1008 frames, and the node allows up to 3600 -- but not on 24 GB of VRAM,
where q+k+v alone would be 12.9 GB on top of a 21 GB checkpoint. Worth
reporting upstream.

**3. `sageattn_consume(qkv, ...)`** -- new entry point that takes
ownership of a `[q, k, v]` list and empties it, releasing each tensor once
its quantized form exists. `sageattn()` cannot do this: the caller's frame
owns the references. The fp8 CUDA wrapper gained an internal `_qkv_box` so
it can drop its own parameters before allocating the q-shaped output.
Output is bit-identical to `sageattn()`; only free timing changes.

Measured at H3's fl2va shape (S=41822, heads 56, head_dim 128), one arm
per process: `sageattn()` 3148 MiB peak, `sageattn_consume()` 2290 MiB
(-858 MiB). 2290 MiB is parity with a hand-rolled per-arch call sequence
into the private quant helpers, so consumers no longer need to reach for
those.

Caveat worth knowing before quoting that number: when q/k/v are three
views of one fused QKV projection buffer -- which is how every DiT block
actually produces them -- freeing q and k releases nothing, because the
buffer only dies with the last view. At H3's shape the saving in that
configuration is 435 MiB, not 858, and the peak is set inside
`per_channel_fp8` rather than by the output allocation. See Backlog.

> **Superseded by v0.7.3.** The 435 in the paragraph above is wrong: the
> fused-QKV saving is 0, not 435. Left in place rather than edited, because
> the number rode in on this version's feature commit and the audit trail of
> what was believed here is the point of a dated entry. v0.7.4 adds the
> caller-side route that recovers 286 of it.

Consumer node shipped separately as
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations), per the
"sage-fork stays primitive" rule in CLAUDE.md.

**In-pipeline validation (2026-08-04).** Unlike v0.6 sage_ffn, this one
was gated on a real render before the claim went anywhere. Full H3 render
through a running ComfyUI at the bundled i2v template's settings
(1344x768, length 73, 20 steps, `res_multistep`/`simple`, `int8_convrot`
weights), warmup discarded, arms alternating, two paired runs:

| | sampler | total render |
|---|---|---|
| sage off | 141.2 s | 151.9 s |
| sage on | 82.9 s | 93.6 s |
| | **1.70x** | **1.62x** |

Both paired runs agreed within 0.3 s on every figure. The sampler is 93%
of total at these settings, which is why the end-to-end ratio stays close
to the sampler ratio -- unusually favourable compared with LTX, and the
reason the synthetic-to-production gap that sank sage_ffn does not appear
here. Peak VRAM ~20.6 GB of 24 GB.

Two results worth keeping for calibration. The per-module bench (one
`Attention` at S=41822) reads 2.12x; the sampler reads 1.70x; the render
reads 1.62x. That ladder is the shape to expect -- each rung adds work
attention cannot touch. And at length 5 the same e2e A/B measures 1.02x,
because the packed sequence is short enough that attention stops
dominating: the win is a property of the sequence length, not of the
model.

**Accuracy, and a calibration correction.** Perceptual verdict first: on
a same-seed 20-step pair at length 124, sage on vs off, the owner reports
no visible difference. That is the verdict that matters -- comparing
finished renders numerically measures trajectory chaos, not degradation,
because any perturbation diverges a 20-step ODE.

The kernel-level numbers back it up, but only once measured on real
activations. On q/k/v captured from an actual H3 forward, fp8++ lands at
mean_rtol **0.026** against an fp32 reference -- roughly **4x lower than
the 0.098 the synthetic bench reports**, and the fp8++-to-fp16 gap
narrows from 2.6x to 1.3x. Real attention has structure that quantization
handles far better than iid gaussian noise does. **Every accuracy figure
this repo quotes from a synthetic bench is a pessimistic bound, not an
estimate.** Worth remembering before treating 0.098 as a quality budget.

Divergence is also flat across sequence length (0.0960 at S=2k to 0.0979
at S=38k), so the `scale_max=2.25` compression mechanism does not compound
on longer clips. Length changes the speed and VRAM equations only.

`smooth_k` was checked properly and left off. K really does carry a
substantial channel offset on real activations (|mean|/std 0.68 mean,
6.09 max), so the precondition for it helping holds -- and it still does
not help (0.0264 -> 0.0266 on fp8++). The likely reason is that
`qk_quant_gran="per_thread"` already scales finely enough that a
per-channel offset never dominates any single scale.

Its cost is near zero either way, so this is a wash rather than a
mistake: peak VRAM is byte-identical with it on (the `k - km` copy is
transient and freed before `per_channel_fp8` sets the peak), and
wall-clock rises ~1 ms -- 3.6% at S=22.5k, 0.5% at S=41.8k, shrinking in
relative terms because the K-mean pass is O(S) against attention's
O(S^2). KJNodes' H3 patch enables it and pays nothing meaningful for it;
we leave it off and gain nothing meaningful either. Worth keeping in mind
if a future model uses a coarser `qk_quant_gran`, where the offset would
actually bite -- turning it on is close to free.

The first attempt at this measurement reported **405x** for the sage arm.
That was ComfyUI's node cache serving a byte-identical graph, so nothing
executed and the run "finished" in 0.0 s. Any e2e harness re-submitting
the same graph needs a varying seed and a hard error when the timed node
never fires; recorded because the failure mode produces a spectacular
number that looks like a win.

### v0.6.6 -- 2026-06-26  (build robustness: `build.sh` auto-avoids the nvcc 13.3 / PyTorch-header miscompile)

Build-tooling hardening, no kernel or wrapper change.

**nvcc 13.3 miscompiles PyTorch >=2.12 headers.** On a box where the
default CUDA toolkit is 13.3, every `.cu` fails during the build with a
spurious error in PyTorch's own header (not sage source):

```
ATen/core/List_inl.h:202: error: need 'typename' before
'decltype(...)::difference_type' because '...' is a dependent scope
```

It is a cudafe++ front-end regression in nvcc 13.3, not a host-compiler
or sage-source problem: the pure-g++ `.cpp` compiles pass, only the
nvcc-driven `.cu` compiles fail. Confirmed by a same-translation-unit
A/B -- a minimal `.cu` including `<ATen/core/function_schema.h>` fails
under `/usr/local/cuda-13.3/bin/nvcc` and compiles clean (exit 0) under
`/usr/local/cuda-13.2/bin/nvcc`, with the host g++ (13.3) and torch
headers held constant. PyTorch here is `2.12.1+cu132` (built against
CUDA 13.2), so 13.2 is also the matched build toolkit.

**Fix: build with 13.2, run on the 13.3 driver.** A 13.2-compiled `.so`
runs natively on a 13.3 driver -- CUDA drivers are backward-compatible,
so there is no Ada runtime cost. `build.sh` now detects a known-broken
active toolkit (`KNOWN_BAD_CUDA=" 13.3 "`) and auto-switches the build
to the newest installed toolkit under `/usr/local/cuda-*` that is not in
the broken set, logging the switch. It overrides even a pre-exported
`CUDA_HOME=/usr/local/cuda` that points at the broken default -- a global
`CUDA_HOME` is common and can't be trusted as an intentional pin when it
resolves to a broken nvcc (this was the actual failure mode on the test
box). Overrides: pick a different toolkit with
`CUDA_HOME=/usr/local/cuda-X.Y ./build.sh`, or force the broken one with
`SAGE_SKIP_CUDA_GUARD=1`. Remove 13.3 from the set once NVIDIA ships a
fixed nvcc.

**Also documented the torch-upgrade footgun.** `uv pip install -e . -U`
(bare, no `--no-deps`) silently upgraded torch 2.11 -> 2.12.1 over
ComfyUI's pinned build. `build.sh` passes `--no-deps --force-reinstall`
specifically to avoid this; CLAUDE.md Install / build now says to use
`build.sh`, not the bare `uv pip install` form, and to rebuild sage
after any deliberate torch upgrade (the `.so` is bound to torch's C++
ABI). Post-rebuild, the load-bearing `tests/test_sageattn_ltx_shapes.py`
run was clean on `2.12.1+cu132` (default `fp8_cuda++` path mean_rtol
0.090-0.097, on-fingerprint; no drift from the 2.11 -> 2.12.1 jump).

### v0.6.5 -- 2026-05-19  (informative precondition assert on `w*_scale` type + stage-1/stage-2 bench rows)

Two coordinated additions surfacing the contract violation that hides
behind a Triton kernel-compile error.

**Precondition assert: `w*_scale` must be Python scalar.**

`sage_ffn(...)` documents `w1_scale: float, w2_scale: float`. When a
consumer passes a 0-d `torch.Tensor` instead (the natural shape from
extracting `weight._params.scale` on a ComfyUI `QuantizedTensor`),
the underlying Triton kernel sees the unannotated `W_scale` argument
as a `pointer<fp32>` and the `acc = acc * W_scale` multiply fails
compilation with `IncompatibleTypeErrorImpl('invalid operands of
type pointer<fp32> and triton.language.float32')`. The error surfaces
inside Triton's autotune machinery, not at the sage_ffn boundary,
making the actual contract violation hard to diagnose.

`sage_ffn` now asserts at the boundary:

```
sage_ffn: w1_scale must be a Python scalar (call .item() on the 0-d
Tensor if extracting from a quantized weight), got Tensor
```

Same pattern as v0.6.2's dtype/shape asserts. The remedy (`.item()`)
is named in the message. The `extract_fp8_weight_and_scale` utility
from v0.6.4 still returns a Tensor (forward-compatible with per-
channel scales); consumers extracting per-tensor scales for sage_ffn
specifically call `.item()` to get a Python float.

Edit at `sageattention/triton/fused_mlp_fp8.py:265-275`. Test
coverage adds 2 cases to `tests/test_sage_ffn_precondition_messages.py`
(now 9 total).

**Bench: stage-1 + stage-2 synthetic rows at production unchunked shapes.**

`tests/bench_sage_ffn_shapes.py` now bench's `T=10780` (stage-1) and
`T=42240` (stage-2) alongside the chunked-call-site shapes already
present (T=4096 + T=1808). Useful for direct synthetic-vs-in-pipeline
comparison; eliminates "the chunk-sweep interpolation was off" as a
noise source in Cell A vs Cell C diagnosis.

Sample run on RTX 4090 / sm89 / torch 2.11+cu130:

| T (seq) | sage_ffn ms | torch stock fp8 ms | stock/sage |
|---:|---:|---:|---:|
| 4096 (full chunk) | 4.68 | 7.00 | 1.50x |
| 1808 (residual chunk) | 2.32 | 3.02 | 1.30x |
| 10780 (stage-1 unchunked) | 13.34 | 18.55 | **1.39x** |
| 42240 (stage-2 unchunked) | 56.41 | 90.52 | **1.60x** |

T=42240 holds ~1.4 GiB bf16 intermediate per FFN call; the bench
now explicitly reclaims tensor refs + calls `empty_cache()` between
shapes so the allocator high-water mark during Triton autotune
doesn't OOM on a 24 GiB card.

### v0.6.4 -- 2026-05-19  (`extract_fp8_weight_and_scale` + framework: 4th silent-fallback rung)

Two coordinated additions surfacing what a cross-clone diagnostic
session learned about silent-fallback patterns in kernel-replacement
wrappers.

**New public utility: `sageattention.extract_fp8_weight_and_scale(linear)`**

Probes a ComfyUI `Linear`-like for fp8 weight + scale across the
four known storage conventions:

1. Modern `QuantizedTensor` with `weight.layout_params.scale` (public
   alias; preferred surface, comfy_kitchen v0.2.8+).
2. Modern `QuantizedTensor` with `weight._params.scale` (raw alias;
   fallback when public alias is absent).
3. Legacy `Linear.scale_weight` attribute (older `fp8_ops` convention).
4. Older `Linear.weight_scale` attribute (some custom-node patches).

Returns `(raw_weight_fp8_tensor, scale_tensor, path_label)` on hit,
`None` on miss. Modern path returns the unwrapped `weight._qdata`,
not the `QuantizedTensor` wrapper -- sage kernels assert
`dtype == float8_e4m3fn` and the wrapper itself doesn't satisfy that.

Source: `sageattention/comfyui_compat.py`. Test coverage:
`tests/test_comfyui_compat.py` (10 standalone cases covering each
probe path, priority order, and miss conditions; mock objects, no
ComfyUI runtime dependency).

Trigger to add this utility: a consumer-side wrapper hit the missing-
unwrap silently for two A/B cycles (passed `weight` rather than
`weight._qdata` to `sage_ffn`; the resulting AssertionError was
swallowed by a logger that stripped `str(exc)`). Centralizing the
probe protects every future consumer from re-deriving the four
storage conventions and getting bitten by the same trap. Reference
intel: `internal/design/comfyui_fp8_storage_conventions.md`.

**Framework: 4th silent-fallback rung in `docs/perf_research_framework.md`**

Rung 2 of the evidence ladder for kernel-replacement audits is
expanded from "every fallback path needs a log line" to "...and the
log line must carry the underlying error's message, not just its
class name." Four distinct failure modes enumerated:

  (a) Explicit `except` without logging.
  (b) Early-return guard before the protected call.
  (c) Implicit dispatch to a pre-patch forward on miss.
  (d) Log fires but strips the error's message (informative-log-but-
      strips-message).

(d) is the trap (a)-(c) graduate to once obvious silent layers are
fixed: surface looks like (a) but is functionally equivalent to (a)
for debugging. The 2026-05-18 cross-clone session that surfaced this
ran for two A/B cycles before the logger was unstripped AND the
underlying QuantizedTensor unwrap surfaced. Worked example captured
in the framework without naming the specific wrapper (consumer-
agnostic framing).

### v0.6.3 -- 2026-05-19  (session-start dispatch log in `sageattn()`)

`sageattn()` now emits one `[INFO] sage routing: arch=... cuda=...
mask=... pv_accum=... -> <kernel>` line to stderr per unique
`(arch, cuda_version, mask_present, pv_accum_dtype, kernel_name)`
tuple observed in the current process. Subsequent calls with the
same tuple are silent.

Sample output (sm89 + CUDA >= 12.8, unmasked):

```
[INFO] sage routing: arch=sm89 cuda=13.0 mask=False pv_accum=fp32+fp16 -> fp8_cuda++
```

This gives consumers a grep-able ground-truth record of which kernel
the dispatcher actually chose for their `(arch, cuda_version, mask
presence, pv_accum_dtype)` config, without forcing a programmatic
call to `get_last_dispatched_kernel()`. The dispatcher audit on
2026-05-18 identified this as the "observability of happy path" gap
between our exhaustive-with-explicit-raise dispatch and the lesson
that "every dispatch leaf should surface its choice once per session"
from a cross-clone wrapper postmortem.

Helpers added to `sageattention.core`:

- `_log_routing_choice_once(arch, mask_present, pv_accum_dtype, kernel_name)`
  -- called from each branch of `sageattn()`. Module-level dedup set;
  thread-safe enough for the one-write-per-tuple semantics.
- `_reset_routing_log_for_test()` -- test-only state reset.

Test coverage at `tests/test_dispatcher_routing_log.py` (5 standalone
cases: first-call emits, second-call deduped, different routing
tuple emits separately, reset helper clears state, kernel name in
log matches `get_last_dispatched_kernel()`).

The existing telemetry test (`tests/test_dispatched_kernel_telemetry.py`)
still passes -- the routing log is additive and orthogonal to the
`_record_dispatch` / `get_last_dispatched_kernel` telemetry API.

### v0.6.2 -- 2026-05-18  (informative `sage_ffn` precondition asserts)

Every `assert` inside `sage_ffn(...)` now carries a message naming
the precondition and the actual offending value. Previously the
asserts had no message, so a downstream wrapper catching
`AssertionError` and logging `str(exc)` received an empty string --
un-actionable for diagnosing dtype/shape mismatches.

Sample messages now surfaced:

```
sage_ffn: x.dtype must be bfloat16, got torch.float16
sage_ffn: w1.dtype must be float8_e4m3fn, got torch.bfloat16
sage_ffn: w1.shape must be (inner=256, hidden=64), got (256, 65)
sage_ffn: b1 must be CUDA bfloat16 with shape (inner=256,), got device=cuda:0 dtype=torch.bfloat16 shape=(257,)
```

Zero runtime cost on the happy path -- Python only formats assert
messages on failure. Edit at
`sageattention/triton/fused_mlp_fp8.py:259-285`.

Test coverage at `tests/test_sage_ffn_precondition_messages.py` (7
standalone cases, one per assert: x.dtype, w1.dtype, w2.dtype,
w1.shape, w2.shape, b1.shape, device).

### v0.6.1 -- 2026-05-17  (stream-safety fix: kernel launches now honor the current CUDA stream)

Every CUDA kernel launch in `csrc/qattn/sm89_*.cu`,
`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`, and `csrc/fused/fused.cu`
used the 3-argument launch configuration `<<<grid, block, smem>>>`,
which omits the stream argument and defaults to the legacy default
stream (stream 0). Triton kernels (used both directly via the
`*_triton` paths and as the QK quant pre-step for `fp8_cuda` paths)
correctly respect `at::cuda::getCurrentCUDAStream()`, but sage's
own CUDA launches silently ignored the caller's stream context.

For default-stream callers this is a no-op: `getCurrentCUDAStream()`
returns the default stream, so the launch lands on the same stream
it always did. For callers that wrap sage in `with
torch.cuda.stream(s_other):`, the launch was landing on the
default stream while preceding kernels (`Linear` projections, etc.)
were correctly enqueued on `s_other`. Without an explicit
`cudaStreamWaitEvent`, the sage kernel saw whatever state the
default stream happened to have, which is racy.

**Fix.** Pass `at::cuda::getCurrentCUDAStream()` as the fourth
launch argument on every kernel launch in the three files above
(1 in each of 7 sm89 .cu files, 4 in the sm80 .cu, 8 in fused.cu).
Add `#include <ATen/cuda/CUDAContext.h>` where it was missing
(fused.cu already had it).

**Verification.** `tests/spikes/spike_concurrent_dispatch_submodule.py`
previously measured a stable ~0.02 mean_rtol drift between
sequential dispatch and side-stream dispatch on the video
self-attention path; post-fix the same spike measures
`bits_equal=True, mean_rtol=0.000000` on both video (sage fp8++)
and audio (FA via SDPA) arms. The same shape under un-patched
quant pre-kernels (fused.cu) but patched attn kernel produced NaN
in the side-stream output -- the quant kernel was racing with the
preceding Linear on the side stream. With all launch sites patched
both stages of the pipeline sequence correctly on the caller's
stream.

`tests/test_sageattn_ltx_shapes.py` shows no rtol or perf drift
on the default-stream path; the two `--check-regression` flags
fired are pre-existing stale baselines from v0.3.0 that predate
the v0.5.5 dispatcher mask-routing change (the `auto` row at
`ltx23_video_cross_text_kv226` now lands on `fp8_cuda++` rather
than `fp16_triton`, which is the v0.5.5-shipped behavior).
Separate from this fix.

**Files touched.** 9 .cu files in csrc + 1 spike (added bit-equality
diagnostic alongside the rtol report).

**Why this surfaced now.** The concurrent-dispatch spikes under
`tests/spikes/spike_concurrent_dispatch*.py` ran sage from inside
a side-stream context for the first time; default-stream callers
never exercise the failure mode (current stream == default stream
makes the missing argument semantically equivalent to the
fix). A cross-render fingerprint check on the production
default-stream path was bit-deterministic at every sage call
position, which scoped the drift to the side-stream path
specifically and motivated the source read that found the
3-argument launches.

**Scope note.** This is a correctness fix, not a perf change. The
default-stream hot path is bit-identical before/after. The fix
unblocks correctness for any consumer that wants to dispatch sage
on a side stream.

### v0.6.0 -- 2026-05-15  (sage_ffn: fp8-native two-kernel fused MLP for LTX 2.3-class FFN blocks on sm89 -- ships as completeness primitive, not currently a perf win in production)

Ships `sage_ffn(x, w1, s1, w2, s2)` -- a Triton two-kernel
`Linear(fp8) -> GELU(tanh) -> Linear(fp8)` MLP path for DiT
FFN blocks whose weights are stored as per-tensor fp8 (E4M3FN).
The primary motivating workload is LTX 2.3 distilled, whose
transformer blocks have hidden=4096, inner=16384 and 44 of 48
blocks shipped as fp8 (bookend blocks `{0, 1, 46, 47}` stay
bf16 per the distilled checkpoint's design).

**The wedge is qualitative, not just quantitative.** No other
library ships an fp8-native fused MLP for ComfyUI consumer-app
shapes on sm89. FA's `fused_mlp_func` is bf16/fp16 only (cuBLASLt
epilogue path); torch's `F.linear` against fp8 weights dequants
to bf16 first, paying 2x the weight-bandwidth and using bf16
tensor cores at ~330 TFLOPS instead of fp8 tensor cores at
~660 TFLOPS. `sage_ffn` loads fp8 weights directly and computes
in fp8.

**Synthetic-bench numbers (RTX 4090, CUDA 13.0, torch 2.12.0+cu130,
triton 3.7.0), bias-inclusive path (matches LTX 2.3 distilled
checkpoint, which carries bf16 biases on both `ff.net.0.proj` and
`ff.net.2`):**

| shape | mean_rtol vs torch ref | sage_ffn | torch ref | speedup |
|---|---|---|---|---|
| stage-1 (T=10780, h=4096, inner=16384) | 0.0914 | 13.3 ms | 18.1 ms | **1.36x** |
| stage-2 (T=44880, multi-guide expanded) | 0.0914 | 59.8 ms | 75.3 ms | **1.26x** |

Bias-free path (sanity check that the `HAS_BIAS=False` constexpr
branch is wired correctly) lands at 1.33x / 1.26x, mean_rtol 0.0915
/ 0.0914 -- statistically identical to the bias-inclusive path
(bias is an epilogue offset, not in the inner matmul loop).

These are standalone matmul-GELU-matmul measurements against
randomly-initialized weights, not end-to-end ComfyUI rendering.
mean_rtol is well under the 0.10 budget that gates all sage
kernels. The reference is `F.linear(F.gelu(F.linear(x, w1_bf16_ref),
approximate="tanh"), w2_bf16_ref)` with weights dequantized once
outside the timing loop -- i.e. torch's best-case fp8-weight path,
not its naive one.

Validated against the full 126-config sweep at the same env:
hardcoded 8-config winners deliver 1.33-1.36x / 1.26-1.27x; full
sweep delivers 1.33x / 1.27x. Bit-identical mean_rtol; hardcoded
matches full-sweep perf within run-to-run noise.

**Production result: sage_ffn is slower than torch in the tested
workload. Ships as completeness primitive, not a perf win.**

In-pipeline A/B on a two-sampler LTX 2.3 FML2V multi-guide
workflow (768x512x97, 8-step stage-1 + 3-step stage-2 refine, on
a 4090 under `nodynvram`, 4 renders interleaved
treatment/baseline/treatment/baseline in the same ComfyUI
session):

| metric | baseline (chunking only) | treatment (sage_ffn + chunking) | delta |
|---|---|---|---|
| wall-time avg | 148.51s | 151.17s | **+1.79% slower** |
| ff @ T=10780 med ms/call | 10.36 | 10.67 | +3.0% slower |
| ff @ T=42240 med ms/call | 48.77 | 58.58 | **+20.1% slower** |
| All non-FFN sub-modules | unchanged | unchanged | 1.00x (identity) |

Per-call FFN times for the warm-autotune treatments (#2 and #4)
matched the cold-autotune treatment (#1), so the regression is
not autotune amortization. Patching surface is clean -- attention
sub-modules at 1.00x ratio confirm the regression is FFN-specific.

**Why the synthetic 1.27-1.36x didn't translate:**

1. **L2 cache contention with neighboring sub-modules.** Synthetic
   bench ran FFN alone with warm L2 holding its tensors.
   Production runs `attn1` (~107 ms at T=42240, large working set)
   immediately before `ff` at stage-2; the attention pass evicts
   FFN's L2 residency. The X-tile-lives-in-L2 assumption from the
   day-3 perf analysis breaks when L2 is hostile. Cold-L2 FFN is
   bandwidth-bound, no fp8-vs-bf16 advantage realized. Worse at
   stage-2 (4x the working set, more HBM round-trips) matches the
   shape of the regression (+20% at T=42240 vs +3% at T=10780).
2. **Cumulative kernel-launch overhead at LTX call count.** LTX
   2.3 fires ~1056 ff calls per render across transformer blocks.
   sage_ffn is two kernel launches per call (matmul+GELU, then
   matmul). torch reference is one cuBLASLt call per matmul. The
   per-call launch-overhead delta scales with that 1000+ count.

This is the v0.5.5 precedent playing out a second time: synthetic
kernel-bench numbers project a wedge, in-pipeline A/B reveals
that production conditions (cache contention, dispatch overhead,
neighboring-module behavior) change the picture. The synthetic
numbers above are real measurements of the kernel in isolation
and are not retracted -- they characterize what the kernel can
do under ideal conditions, which is useful information for
future kernel work. They are not the delivered consumer-app
number.

**What this means for users:**

- Keep an FFN-chunking node (e.g. `LTXVChunkFeedForward` from
  KJNodes) as the production default; don't replace it with a
  consumer-side `sage_ffn` patch node expecting a speedup.
- `sage_ffn` is available for users who specifically need an
  fp8-native fused MLP for ComfyUI consumer-app on sm89 (no
  other library provides this combination). The "uncontested
  availability" wedge holds; the "delivered speedup" wedge does
  not on the tested workload.
- Different workload shapes may differ from this result -- the
  L2-contention picture depends on what other modules are
  active and what their working sets look like. Single-pass
  (non-multi-guide) workloads in particular may behave
  differently. In-pipeline measurement is the gate, not
  synthetic bench.

**Design: two-kernel split, not single-kernel fusion.** Kernel 1
matmul + GELU(tanh) epilogue + write intermediate. Kernel 2
matmul down-projection. Intermediate at LTX stage-2 multi-guide
is ~1.47 GiB; users on 24 GiB cards should compose with an
FFN-chunking node (e.g. `LTXVChunkFeedForward` from KJNodes).
The single-kernel "intermediate never hits HBM" design was
explored on day 1-2 and rejected at day 2's perf wall: the
X_tile (1 MB at BLOCK_M=128, K=4096, bf16) won't fit in sm89's
100 KB SMEM, and the nested-loop structure was forcing Triton
into L2 evictions. FA's `fused_mlp_func` is also a two-kernel
split for the same reason -- this design converges on the
industry standard.

**Per-block-K activation quantization.** Each (BLOCK_M, BLOCK_K)
chunk of the bf16 activation gets its own f32 scale, computed
inline during the K-reduction. This avoids a separate amax pass
over the full K dimension; the slight coarsening (~0.005 rtol
cost vs per-row) is well within the 0.10 budget.

**Autotune.** Each kernel carries 8 hardcoded `@triton.autotune`
configs -- the winners from a 126-config sweep against the two
LTX FFN shapes, plus a few neighbors for shapes the winners may
not cover. First call at a new shape pays ~10-15 seconds
autotune-search per kernel (~30-60 sec total across the full
sage_ffn at two new shapes); subsequent calls hit Triton's
on-disk cache. The full 126-config sweep cost 7+ minutes
first-render-per-shape on consumer hardware -- unshippable UX.
The pruned 8-config set preserves the 1.27-1.33x delivered
numbers at acceptable cold-start cost (validated against the
full sweep at the same env). To re-derive winners for a new
LTX-class shape: run the kernel against the shape, inspect
`_fp8_matmul_gelu_kernel.cache` for the picked config.

**Why the synthetic numbers stay in the docs.** The 1.27-1.36x
synthetic-bench delta is a real, measurable property of the
kernel running in isolation against `F.linear(F.gelu(F.linear(...
)))` with bf16-dequant weights. Removing those numbers would
hide useful information about the kernel's isolation behavior
(relevant for future kernel-day work, e.g. evaluating whether a
v0.6.1 redesign actually moves the isolation number). The
discipline lesson is "don't *promote* the synthetic number as a
delivered consumer-app speedup" -- not "don't measure synthetic
performance."

**Paths to close the production gap (v0.6.1 candidates, not
v0.6.0 blockers):**

1. **Persistent-CTA two-kernel hybrid** (the option (b') flagged
   in the scoping doc). Persistent CTAs hold M-tile state in
   registers / L2 across the two matmuls, addressing the
   L2-contention root cause directly. Significantly more kernel
   engineering than v0.6.0.
2. **CUTLASS-based CUDA backend** for the fp8 matmul. Closes the
   Triton-vs-cuBLASLt codegen gap that bounds the synthetic
   ceiling at 1.27-1.36x; unclear whether it would also recover
   the production loss. 2-3 weeks of work.

Neither blocks v0.6.0 ship. See Backlog for triggers.

**API.**

`sage_ffn(x, w1, w1_scale, w2, w2_scale, b1=None, b2=None)`. The
optional `b1` (inner,) and `b2` (hidden,) bf16 biases match the
LTX 2.3 distilled checkpoint's FFN layout (`nn.Linear(...,
bias=True)` defaults on both projections). When a bias is `None`,
the kernel's `HAS_BIAS=False` constexpr branch compiles the load
out -- no runtime cost on the bias-free path.

**Limitations / scope.**

- Plain GELU MLP only. No gated SwiGLU/GEGLU variant in v0.6;
  the FFN structure has to be `Linear -> GELU(tanh) -> Linear`.
- Bookend bf16 blocks must be handled by the caller. Consumer-side
  dispatch typically inspects `block.ff.net[0].proj.weight.dtype`
  and falls through to `F.linear` for bf16 blocks.
- Per-tensor scalar weight scale only. Per-row / per-channel
  weight scales are a v0.6.1 extension if a workload demands.
- Bias must be bf16. Casting to bf16 at quantization time is
  trivial; pre-quant bias scales are not supported (biases are
  small, fp8 is overkill for them).
- Not wired into `sageattn()` -- this is a separate FFN primitive,
  not an attention kernel. Consumer imports `sage_ffn` directly.

**Files added / changed:**

- `sageattention/triton/fused_mlp_fp8.py` (new) -- the two-kernel
  implementation + `sage_ffn` Python wrapper.
- `sageattention/__init__.py` -- `sage_ffn` export.
- `tests/spikes/spike_fp8_mma.py` (new) -- day-1 spike verifying
  `tl.dot(fp8, fp8) -> f32` on sm89 at small + LTX stage-1
  shapes.
- `tests/spikes/test_fused_mlp_fp8_correctness.py` (new) --
  correctness + perf gate at LTX FFN shapes, median-of-5 timing
  after autotune-absorbing warmup.

Design narrative (cross-claude memo trail, v0.6 scoping doc,
day-by-day execution journal, decision-gate framework) lives in
`internal/design/ffn_fusion_scoping.md` (gitignored).

### v0.5.5 -- 2026-05-13  (native general-mask support in the sm89 fp8++ CUDA kernel)

First downstream-driven kernel-day work on the fork. Lands the
load-bearing piece of the long-standing CUDA mask gap (in "Known
kernel bugs" since 2026-04-23): the sm89 fp8++ kernel now applies an
additive attn_mask to QK scores before the softmax max reduction,
matching the Triton reference's behavior. Dispatcher routes masked
calls on sm89 + CUDA >= 12.8 to the new path.

Triggered by a high-leverage downstream consumer surface raising
the structural-correctness concern (the "1 high-leverage surface"
clause added in v0.5.4 backlog reformulation). Scoping note +
implementation discipline in
`docs/cuda_mask_kernel_scoping.md`.

#### Added

- **`MaskMode::kGeneral`** value in `csrc/qattn/attn_utils.cuh` +
  **`apply_general_mask<num_tiles_q, num_tiles_k, DTypeMask,
  DTypeQKAccum>`** helper. Mirrors `apply_causal_mask`'s index math;
  adds in-bounds-guarded mask load + additive apply per (q_idx,
  kv_idx) for each thread's 8-entry register fragment. Supports
  bool masks (translated to additive {-inf, 0} log-weights upstream
  in Python) and dtype-matching float masks (half or nv_bfloat16),
  mirroring the two Triton mask paths.

- **`DTypeMask` template parameter + mask runtime params** on
  `qk_int_sv_f8_attn_kernel` (`qk_int_sv_f8_cuda_sm89.cuh`). Defaults
  to `nv_bfloat16` / nullptr / 0 strides so existing instantiations
  compile unchanged. `if constexpr (mask_mode == MaskMode::kGeneral)`
  branches at the two existing mask-application points (the
  steady-state K-iteration block + the last-iter block) call the
  helper; both branches dissolve in the kCausal / kNone
  specializations.

- **Optional `attn_mask` parameter on `qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`**
  (C++ entry + pybind + `sm89_compile.py` custom_op schema +
  register_fake stub). Backward-compatible: existing positional
  callers don't need updates. The dispatcher in the .cu branches on
  `attn_mask.has_value() && attn_mask->numel() > 0`: kGeneral
  specialization gets launched when the branch is taken, kCausal /
  kNone otherwise.

- **Python wiring**: `sageattn_qk_int8_pv_fp8_cuda` extracts
  `attn_mask` from kwargs and passes it to the fp32+fp16 (fp8++)
  variant. bool->additive {-inf, 0} translation happens here, plus
  `attn_mask.expand((B, H, qo_len, kv_len))` for broadcast support
  (mirrors `core.py:441` in the Triton path). Other `pv_accum_dtype`
  variants still warn + drop the mask (their kernels haven't gained
  the kGeneral path).

#### Changed

- **Dispatcher routing**: `sageattn()` masked-call routing now
  arch-aware. On sm89 + CUDA >= 12.8, masked calls land on
  `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`
  (the new CUDA mask path). On other archs (sm80, sm86, sm87, sm75,
  sm100/sm120/sm121 fallback), masked calls still route to
  `sageattn_qk_int8_pv_fp16_triton` since their CUDA kernels haven't
  gained mask support yet.

- **Test invariant update**: `tests/test_dispatched_kernel_telemetry.py`
  renamed `test_sageattn_dispatcher_routes_masked_calls_to_triton`
  -> `*_correctly`. The v0.3.0 "all masked calls -> triton"
  invariant is superseded; v0.5.5's invariant is arch-aware.

#### Measured

At LTX-class shape (B=1, H=4, T=2048, D=128, bf16, RTX 4090):

| measurement | value |
|---|---|
| CUDA mask vs Triton reference (bool mask) | mean_rtol=0.098, max_rtol=2.0, mean_atol=0.0011 |
| bool->additive translation equivalence | max_atol=0.000000 (bit-identical) |
| mask actually applied (vs unmasked output) | max_atol=0.055 (changes output meaningfully) |
| zero-mask sanity (vs unmasked) | max_atol=0.000000 (additive zero is true no-op) |
| LTX bench fp8_cuda++ unmasked (`ltx23_video_self_attn_init_22932`) | 19.98 ms (baseline 20.20; no regression) |
| LTX bench fp8_cuda++ masked cross-attn rows | mean_rtol ~0.09 (was ~0.94 pre-v0.5.5 silent-drop) |

The zero-mask bit-identity test is the strongest correctness signal
for the `if constexpr` discipline: the kGeneral branch infrastructure
adds no perturbation to the kNone specialization when mask values
are no-ops. The 0.055 masked-vs-unmasked atol confirms the mask is
actually applied (not silently dropped, which would produce 0 atol
like pre-v0.5.5).

#### Measured (in-pipeline, preliminary; framing softened 2026-05-15)

Beyond the synthetic correctness + perf measurements above, we ran
A/B comparisons on a real LTX 2.3 workflow (multi-guide at
768×512×97) on a 4090 with dynamic VRAM disabled
(`--disable-dynamic-vram --disable-async-offload --reserve-vram 0
--cuda-malloc --supports-fp8-compute --mmap-torch-files
--cache-none`). Updated count after additional repetitions:

| arm | outcome |
|---|---|
| `auto` (v0.5.5 fp8_cuda++ masked) + FFN chunking ON | N=3+ success, 0 OOM |
| `auto_mask_aware` (Triton masked) + FFN chunking ON | N=1 success, N=2 OOM (non-deterministic) |
| `auto` (fp8_cuda++) + FFN chunking OFF | deterministic OOM at stage-2 FFN GELU |

Both Triton OOMs hit the same site
(`comfy/ldm/lightricks/model.py` `AdaLNSingle.linear`,
downstream of attention; 727 MiB requested, ~16 MiB free,
~22.4 GiB allocated of 23.52 GiB), at exactly 48 masked dispatches.
The chunking-off fp8++ OOM hits the FFN GELU projection at
multi-guide T=44880 (proj output `(1, 44880, 16384)` bf16 ≈ 1.47 GiB).

**The corrected reading (after the chunk-bypass A/B)**: at LTX 2.3
multi-guide scale on 24 GiB, FFN chunking (`LTXVChunkFeedForward`)
is doing the heavy lifting on peak memory. With chunking on, the
attention working-set delta between Triton and fp8++ is the knob
that distinguishes "fits comfortably" (fp8++) from "fits 1/3 of the
time" (Triton non-determ OOM at AdaLN). Without chunking, both
kernels hit a different (FFN-intermediate) memory wall.

So the honest claim becomes: **with FFN chunking enabled, sage's
v0.5.5 CUDA mask path has more headroom for the attention-side
delta than the Triton fallback** (N=3+ vs N=1 success in the
observed sample). The earlier framing -- "fits where Triton
doesn't" -- was incomplete; sage choice is the second-order win
once chunking handles the first-order FFN memory cost.

Per-shape masked p50 latency (successful arm, single-run sample):
`(1, 10780, 4096)` masked Triton 195 μs vs fp8++ unmasked 158 μs;
`(1, 44880, 4096)` masked Triton 482 μs vs fp8++ unmasked 249 μs.
Run-to-run variance at the larger shape is real and not yet
attributed (autotune-cache warmth, per-call mask sparsity, thermal
state are all candidates); don't read p50 as precision.

**Status**: preliminary. Sample sizes are small; the Triton OOM is
non-deterministic; further repetitions and a smaller-mask variant
test are in progress. The synthetic measurements above remain the
primary v0.5.5 validation. The in-pipeline observation is
corroboration that's reproducible by anyone with the same workflow,
the routing flag flip, and the nodynvram config -- independent
reproduction welcome.

What's deferred and on what triggers: see Backlog / "Extend CUDA mask
support beyond sm89 fp8++".

### v0.5.4 -- 2026-05-13  (sageattn_partitioned + multi-slice peak-HBM bench + honest negative result on the masked-call scenario)

Driven by a consumer-side report that the masked Triton path
OOM'd on a 4090 in a workflow that partitions Q into noisy + tracked
slices and fires two back-to-back `sageattn_qk_int8_pv_fp16_triton`
calls per layer with the same K, V. Hypothesis: each call
re-quantizing K and re-casting V is the removable peak-HBM lever.
Built the entry, measured against the hypothesis, found the
masked-call scenario already efficient enough that the entry doesn't
help in synthetic isolation. Documented for future consumers and
real-pipeline validation.

#### Added

- **`sageattention.sageattn_partitioned(q, k, v, slices, ...)`** --
  public entry that runs Triton attention over multiple Q slices
  sharing K and V. Quantizes K once, casts V to fp16 once,
  allocates the output once; Q is re-quantized per slice (Q
  changes per slice). Each slice is `(q_start, q_end, attn_mask
  | None)`. Inner kernel writes each slice's output into a view
  of the pre-allocated full output via a new optional `out=`
  parameter on
  `sageattention.triton.attn_qk_int8_per_block.forward`. Records
  dispatch as `fp16_triton` (same underlying kernel). Test:
  `tests/test_partitioned.py` (4 cases: 2-call aligned/unaligned
  boundary, noisy-only, tracked-only; reuses `accuracy_metrics`
  from `test_sageattn_ltx_shapes.py` so tolerance budgets match
  every other rtol-vs-Triton row in the repo).

- **`sageattention.triton.quant_per_block.per_block_int8_q`** and
  **`per_block_int8_k`** -- factored from the existing
  `per_block_int8` Q+K quant. `per_block_int8` now wraps both
  helpers, signature unchanged for existing callers.
  `sageattn_partitioned` uses them to quantize K once across all
  slices while Q re-quantizes per slice.

- **`tests/bench/partitioned_mask_phase0/`** -- peak-HBM
  characterization at the LTX 2.3 self-attn shape (T=23296, h=32,
  d=128, bf16) for the two-call partition pattern (noisy + tracked
  slices sharing K, V). Six measurement rows: single-call no-mask
  reference, single-call with broadcast `(1,1,1,T)` mask,
  2-independent-call cumulative no-mask, 2-independent-call
  cumulative with-mask, `sageattn_partitioned` no-mask,
  `sageattn_partitioned` with-mask. Reports savings vs the
  2-independent-call baseline. Checked-in `results.json` + memory
  snapshot for the audit trail; uncorrected wrong-mask-shape
  variants preserved as `*.uncorrected.{json,bin}` (caught by a
  sister-clone audit of the actual mask shapes the consumer
  workflow uses).

#### Measured: synthetic Phase 0 + Phase 3 result

At the LTX 2.3 self-attn shape on RTX 4090, the 2-call
partition's cumulative peak HBM:

| call pattern | no mask | with two-call partition masks |
|---|---|---|
| single full-T call (reference) | 1096 MiB | 1096 MiB |
| 2 independent sage calls (cumulative) | 1807 MiB | 1272 MiB |
| `sageattn_partitioned` | 1528 MiB | 1298 MiB |
| **savings** | **+279 MiB** | **-26 MiB** |

The partitioned entry saves +279 MiB in the no-mask isolation
(as predicted: K-quant + V-cast amortization is real). But in
the masked-call scenario -- the originating use case -- the
2-independent-call cumulative peak is already only +176 MiB
above the single-call reference, and the partitioned entry
doesn't reduce it further. Cross-clone hypothesis (filed by the
sister clone as "hypothesis 2" before Phase 3): allocating the
22 MiB mask first in each call may bias the pytorch caching
allocator's bucket layout to consolidate K_int8 / V_fp16
allocations better than the partitioned entry's
K-first / V-first / output-first pattern. Not investigated;
matches the observed asymmetry.

**Net for the originating use case**: the partitioned entry
doesn't address the consumer-reported OOM in synthetic measurement.
Real-pipeline validation (sister-clone side with the LTX denoiser
loaded, against fragmented post-model-load allocator state) may
show a different picture; sage-fork side considers that an open
question. The entry stays shipped because (a) it's correct, (b) the
no-mask savings are real for any future consumer that hits a similar
pattern without masks, and (c) it provides the primitive to test
against in real-pipeline scenarios.



Three coupled additions, all driven by a consumer-side comparison
doc that surfaced (a) one structural kernel-side gap vs KJNodes'
`LTX2MemoryEfficientSageAttentionPatch` (his fused-RoPE Triton
kernel), (b) an unverifiable "memory efficient" framing, and (c) a
fork API stability concern after v0.5.0 dropped `_qattn_sm90`.

#### Added

- **`sageattention.fused_rope_split(q, k, freqs_cis, *, use_triton=True)`**
  -- public fused split-RoPE primitive. Matches LTX's
  `apply_split_rotary_emb` (comfy/ldm/lightricks/model.py:343)
  exactly via a clean-room Triton kernel; falls back to a torch
  reference when preconditions fail (non-cuda, non-split-pe,
  shape mismatch, `use_triton=False`). Lives in
  `sageattention/triton/fused_rope.py`. Lets consumers drop their
  own per-block fused-rope kernel (e.g. KJNodes'
  `fused_rope_qk`) without sage going DiT-aware -- the API is a
  standalone helper, not bolted into `sageattn()`. v1 covers the
  split-RoPE convention only (LTX 2.3 video + audio); interleaved
  variants and other model classes silently fall back.
  Test: `tests/test_fused_rope.py` (3 CPU tests + 7 GPU tests
  covering rtol vs reference, dtype coverage, in-place
  semantics, fallback-path equivalence, dtype guards, public
  export).

- **`tests/test_sageattn_ltx_shapes.py` peak-VRAM column.** Folded
  into `time_and_vram(fn) -> (median_ms, peak_vram_mib)` (renamed
  from `time_median_ms`). Reports per-(shape, kernel) working-set
  VRAM at zero extra kernel cost -- the warmup pass that already
  absorbs autotune now also seeds the VRAM baseline, then
  `reset_peak_memory_stats()` rebases before the 3 timed runs.
  Initial finding from the new column: at the load-bearing LTX
  video self-attn shape (D=128, seq=22932), sage `fp8_cuda++`
  uses ~628 MiB working-set vs `torch_flash` ~182 MiB --
  empirically refutes "sage = memory efficient" framing for the
  high-level dispatch path on these shapes (sage materializes
  q_int8/k_int8/v_fp8/scale intermediates that flash keeps fused
  in registers/SMEM). Print-only, no regression gate.

- **CLAUDE.md "Downstream-known internal symbols" section.**
  Documents the de-facto-public underscore surface in
  `sageattention.core` that downstream consumers (KJNodes' LTX-2
  patch as canonical example) import by name. Lists the
  protected symbols, the protected pybind methods on
  `_qattn_sm89`, and a pre-removal checklist (grep coderef, memo
  before removal, major-bump on break). Triggered by the v0.5.0
  `_qattn_sm90` removal, which broke a downstream import case
  without prior consideration.

### v0.5.2 -- 2026-04-27 PM  (bench reliability: auto-warmup correctness + honest cold-start interpretation)

Two real bugs surfaced in the same hour AFTER v0.5.1 shipped, both
caught by running the bench end-to-end against ComfyUI restarts:

1. **`--warmup auto` false-positive on ComfyUI restart.** The
   filesystem-mtime heuristic from v0.5.1 (`aae9b9e`) detected
   "recent sage trace exists" and skipped warmup. But the trace
   file persists across ComfyUI restarts; mtime stays recent even
   though the in-process state (model load, JIT, per-node cache)
   was reset. Auto-mode correctly identified "sage was active 30
   min ago" and incorrectly concluded "caches warm now."
2. **Bench `Interpretation` mislabeled cold-start asymmetry as
   "sage SLOWER end-to-end."** Two real benches today landed at
   0.508x and 0.900x raw -- both cold-start-confounded. The bench
   printed "Sage SLOWER ... Check for instrumentation overhead,
   fallback paths, or a real regression." A future operator
   without per-node `exec.jsonl` analysis would walk away thinking
   sage broke. Sage's actual contribution was 1.22x e2e
   (audio-loop-helper claude's per-node analysis salvaged the
   reading; the bench output didn't).

#### Fixed

- **`tests/bench_e2e_ltx.py::_caches_appear_warm(host)`** (commit
  `a461ddb`). Auto-mode now requires BOTH (a) a recent non-empty
  sage.jsonl on disk AND (b) ComfyUI's `/history` HTTP endpoint
  to return a non-empty result. `/history` is in-memory; restart
  empties it. Combined signal correctly catches the
  restart-after-trace case the v0.5.1 heuristic missed.
- **`tests/bench_e2e_ltx.py` Interpretation block** (commit
  `1a06586`). Added `cold_start_suspected` branch BEFORE the
  generic SLOWER message. Triggers when `speedup < 0.95x` AND
  `attn_pct_on < 20%` -- structural signal that the slowdown is
  in non-attention work where sage doesn't run, almost certainly
  cold-start asymmetry between arms. Prints a diagnostic message
  with concrete next-steps (`--warmup always` from fresh ComfyUI,
  or aggregate exec.jsonl per-node) instead of the misleading
  "real regression" hint.

#### Refactored (simplify-pass on the fixes above)

- **Reuse existing `_http_get` helper** for the `/history` probe
  instead of duplicating `urllib.request.urlopen`. The helper
  was already in the file at line 329.
- **`/history/1`** instead of `/history`. ComfyUI's `/history` is
  unbounded across a session; the `/{max_items}` form caps the
  response server-side. Same empty-vs-not semantics, bounded
  payload.
- **`ATTN_PCT_LOW_FRACTION = 20.0` constant** alongside the
  speedup tier constants. Used twice in `report()`; bump-one-
  forget-the-other risk neutralized.
- **`_attn_pct_of_wall(results_on, on_med)` helper** extracted.
  Removes the declare-default-then-conditionally-assign pattern
  in the report block.
- **Drop `urllib.error.URLError`** from the `_http_get` exception
  tuple (subclass of `OSError`; redundant).
- **Trim `_caches_appear_warm` docstring** to a one-line summary
  pointing at `main()`'s warmup-policy comment for the
  operational rationale.

#### Verified

- `_comfyui_session_has_history` correctly returns True when
  ComfyUI has processed >=1 prompt this session, False on fresh
  restart (empty `/history`).
- `cold_start_suspected` branch fires correctly when replaying
  yesterday's actual numbers (0.508x raw + 4.5% attn pct);
  produces the diagnostic message instead of the old "SLOWER ...
  real regression" message.
- Other interpretation tiers unchanged; `cold_start_suspected`
  only intercepts the < 0.95x AND low-attn-pct combination.

#### CLAUDE.md

Added a "Testing / `tests/bench_e2e_ltx.py` warmup auto-detection"
subsection documenting the two-signal requirement and the
asymmetric-cost reasoning behind the auto-mode design (false-
positive worse than false-negative; always errs toward warmup).

### v0.5.1 -- 2026-04-27  (e2e validation: bench infra fixes + first end-to-end speedup measurement)

The first-ever empirical measurement of sage's end-to-end speedup
on a production workload, after a series of bench-infrastructure
fixes that landed across the day. Headline result: **sage delivers
1.22x e2e speedup on the canonical LTX 2.3 audio-loop workload** at
832x480x497 / 25fps / 8-step distilled, with consumer-side fixes
(skip_under_seq_len short-Q skip + VAE decode normalized to a
single tile) in place.

**VISION.md item-3 status: confirmed (with refinement).** Kernel-ms
partially translates to gen-ms. Sage's 2.66x-at-attention kernel
speedup translates to ~1.22x end-to-end. The translation factor is
bounded by both Amdahl (attention is 8% of wall on this workload)
AND sage's per-call reach beyond attention rows -- FFN-adjacent
amortization adds ~30% of headline savings on top of the strict-
attention prediction. The "kernel ms = gen ms" simplification is
not literally true; sage is load-bearing, kernel work is justified,
but the next round of e2e wins routes to non-attention bottlenecks
(VAE decode amortization, caching, scheduler overhead) per
downstream consumer's Phase 2.1.

#### Added

- **`tests/bench_e2e_ltx.py --warmup {auto,always,never}`**
  (commit `aae9b9e`). Replaces the boolean `--no-warmup` (preserved
  as alias). `auto` mode probes
  `coderef/.../data/runs/<RUN_ID>/sage.jsonl` mtime; skips warmup
  when a recent trace (< 30 min) is found. Saves ~250s on
  iterative bench sessions. Discovered necessary 2026-04-27 when a
  cold `--runs 1` bench reported sage 2x SLOWER end-to-end while
  attention was only 4.5% of wall -- structurally impossible from
  sage alone; cold-start order effect was the real cause. The
  warmup-and-discard fix (`05a63e8`) addressed the bias; the
  auto-detect (`aae9b9e`) made it free on warm sessions.
- **`tests/bench_workload_profile.py` skip_reason aggregation**
  (commit `6802b2d`). `parse_traces` now buckets rows where
  consumer policy short-circuits sage (e.g.
  AudioLoopHelper's `skip_under_seq_len`, their commit `04919fd`,
  2026-04-27) under `skipped:<reason>` synthetic kernel names.
  New `print_skip_reasons()` section surfaces "X calls
  policy-skipped" alongside sage dispatch counts. Module constant
  `SKIPPED_KERNEL_PREFIX = "skipped:"` so downstream readers can
  discover the bucket explicitly.

#### Fixed

- **`tests/bench_e2e_ltx.py::resolve_run_id` RUN_ID auto-resolution**
  (commit `ea93006`, fix flagged by AudioLoopHelper claude as their
  Bug #2). When neither `--run-id` flag nor `$RUN_ID` env var was
  set, the bench fell back to globbing the legacy
  `internal/analysis/runs/sage/sage_*.jsonl` directory and found
  yesterday's most-recent file. Today's active trace at
  `data/runs/<RUN_ID>/sage.jsonl` was never considered. Fix scans
  for the most-recently-modified directory matching the consumer's
  `start_experiment.sh` format (`^\d{8}T\d{6}Z_[0-9a-f]{4}$`) and
  uses it. Common-case foot-gun for any operator running the bench
  from a fresh terminal that didn't inherit RUN_ID.
- **`tests/bench_e2e_ltx.py` non-attn-time print formula**
  (commit `05a63e8`). Old line 525-526 computed
  `non_attn_off - (off_med - on_med)` which simplifies to `on_med`
  -- the print was always showing the on-arm wall time mistakenly
  labeled as off-arm non-attn estimate. Replaced with factual
  off-arm wall surface; off-arm has no per-call tracer, so attn
  vs non-attn breakdown is unavailable and we say so.

#### Refactored

- **`tests/bench_e2e_ltx.py::_iter_trace_rows()` helper**
  (commit `aae9b9e`). Extracts the JSONL row-iteration primitive
  from four near-identical loops (`sum_attn_us_for_prompt`,
  `sum_attn_us_in_window`, `_trace_has_prompt_id`, plus
  `bench_workload_profile::parse_traces`). The four call sites
  collapse to ~3 lines each. Skips empty lines, JSON parse
  failures (mid-write tail), and (by default) framing rows
  (`event in {"header", "summary"}`).

#### Changed

- **Dropped the misleading Amdahl-ceiling note in bench output**
  (commit `aae9b9e`). The earlier note (`05a63e8`) printed a
  "ceiling X.XXx" derived from attention-fraction Amdahl when
  `attn_pct < 20%`. Per the 2026-04-27 cross-arm `exec_log`
  analysis, sage's reach extends beyond per-call attention rows
  (~26-28s sampler savings on top of ~11s pure-attention delta);
  pure-attention Amdahl is a LOWER bound, not a ceiling. An
  operator reading "ceiling 1.03x" while the actual ratio is
  1.22x walks away with the wrong story. Replaced with a factual
  one-liner pointing the reader at the empirical speedup ratio
  printed above.

#### Measured (new headline numbers)

`tests/bench_e2e_ltx.py` against
`audio_loop_latent.api.json` at 832x480x497 / 25fps / 8-step
distilled / `[1,1,1]` VAE decode tiles, with
AudioLoopHelper's `skip_under_seq_len=1024` widget enabled
(consumer commit `04919fd`):

| arm                | wall    | sampler  | VAE decode | sage attn  |
|--------------------|--------:|---------:|-----------:|-----------:|
| sage_on (cold VAE) | 138.8s  | 82.4s    | 47.4s      | 11.45s (8.2% wall) |
| sage_off (warm VAE) | 123.8s | 110.1s   | 10.4s      | n/a (no tracer) |

- **sage_on / sage_off raw**: 0.900x (sage 14s slower; VAE
  cold-start dominates).
- **VAE-cold-start-normalized** (subtract 37s arm-1 premium):
  138.8s - 37s = 101.8s vs 123.8s = **1.22x e2e speedup**.
- **Sage sampler savings**: 110.08s - 82.41s = **27.67s** (25%
  of sampler wall).
- **Pure-attention Amdahl prediction**: 8.2% attn x (1 - 1/2.66)
  = ~5.1% e2e speedup, i.e. ~1.05x. Observed 1.22x is **17 points
  higher** than this prediction; the surplus is sage's
  FFN-adjacent reach via int8 amortization + kernel pipelining
  effects within the sampler step beyond the attention rows
  themselves.

#### Findings worth flagging

- **VAE decode is the new headline bottleneck on this workload.**
  Even with single-tile decode, arm-1 cold-start carries a 37s
  premium over arm-2 (5GB+ activation buffer alloc, per-shape
  autotune, possibly cuBLAS workspace). Consumer-side Phase 2.1
  routes here next; sage-fork has no scope claim on VAE work
  (conv-style operators, not attention).
- **Both priors were wrong in the same direction.** sage-fork
  predicted 0.95-1.05x (wash), then revised to 1.05-1.10x
  (small win). AudioLoopHelper claude predicted 1.05-1.10x
  (revised from earlier 1.30-1.80x). Actual 1.22x. Both anchored
  on attention-fraction Amdahl and missed the FFN-adjacent reach
  empirically. Lesson: when sage's per-call wins translate to
  end-to-end, measure the boundary sage actually patches (the
  sampler), not the boundary the per-call timing exposes (the
  attention row).
- **`skip_under_seq_len=1024` working as designed.** Workload
  profile confirms 2304 of 4608 attention calls (50%) are
  policy-skipped before reaching sage; all are seq < 1024
  short-Q rows where sage was 0.45x torch_flash per the v0.4.1
  bench. The skip widget delivers ~11% wall-time savings on its
  own (cold-vs-cold; AudioLoopHelper's pre-vs-post-skip
  measurement, isolated from cold-start).

#### Cross-repo coordination

- AudioLoopHelper claude shipped `skip_under_seq_len=1024` widget
  + `prompt_id` contextvar fix in `04919fd`, plus per-prompt
  RUN_ID routing in `abe443b` (`AUDIOLOOPHELPER_PER_PROMPT=1` env
  var). Memo trail at `coderef/.../internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md`
  + `coderef/.../internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md`.
- Field-name compat discipline: their addition of
  `skipped: bool` + `skip_reason: str` trace fields was
  pre-vetted (pure-additive; default-skip-unknown semantics on our
  side). No breaking change.

### v0.5.0 -- 2026-04-27  (dead-code removal: Hopper/Blackwell + Windows + upstream bench)

Aggressive cut of upstream code that doesn't serve sm89/Ada or our
active research surface. We own this fork; we never push to upstream;
the cost of carrying unused code is more than the cost of removing
it. Each removal landed as its own commit so `git revert <sha>`
works granularly per arc.

#### Removed (4 arcs, ~7000 lines, no functional change for sm89)

**Arc 1 -- `sageattention3_blackwell/` (commit ceddb19, -5027 lines)**
The sage 3 Blackwell subpackage targets sm120+ with FP4 quantization.
Completely isolated -- zero imports from `sageattention/`, no
`setup.py` references, no test coverage. 27 files removed.

**Arc 2 -- sm90 Hopper kernel + Python wrapper + entry point (-1285 lines)**
- `csrc/qattn/{attn_cuda_sm90.h, pybind_sm90.cpp, qk_int_sv_f8_cuda_sm90.cu}`
  -- the Hopper CUDA kernel sources.
- `sageattention/sm90_compile.py` -- the Python custom-op wrapper.
- `sageattention/core.py::sageattn_qk_int8_pv_fp8_cuda_sm90` (~170-line
  function), the `SM90_ENABLED` guard + try/import block, and the
  `arch == 'sm90'` dispatcher branch in `sageattn()`.
- `KERNEL_FP8_CUDA_SM90` constant + `'fp8_cuda_sm90'` entry from
  `KernelName` Literal and `KNOWN_KERNEL_NAMES` frozenset.
- `sageattn_qk_int8_pv_fp8_cuda_sm90` from
  `sageattention/__init__.py` exports.
- `setup.py`: the SM90 `CUDAExtension` build block + the
  CUDA-12.3-for-9.0 minimum-version check.
- `build.sh`: `_qattn_sm90` from the verify-extensions inventory + the
  "sageattn3 (Hopper/Blackwell)" line in the available-kernels print.

Build verifies clean post-removal: `_qattn_sm80` + `_qattn_sm89` +
`_fused` import. Telemetry test passes (dispatcher routes correctly,
fp8_cuda++ on unmasked / fp16_triton on masked, soft-warns on
hand-picked CUDA + mask). Regression-check unit tests pass.

**Arc 3 -- `bench/` upstream one-shape benchmarks (-706 lines)**
9 files (`bench_baseline.py`, `bench_fa3.py`, `bench_fa3_fp8.py`,
`bench_qk_int8_pv_fp16_cuda.py`, `bench_qk_int8_pv_fp16_triton.py`,
`bench_qk_int8_pv_fp8_cuda.py`, `bench_qk_int8_pv_fp8_cuda_sm90.py`,
`utils.py`, `README.md`). Zero references from any code we keep.
Superseded by:
- `tests/test_sageattn_ltx_shapes.py` -- production-shape sweep across
  every sage kernel + 3 torch SDPA backends + FlashInfer/Sparge gates.
- `tests/test_sageattn_image_shapes.py` -- image-gen head_dim coverage.
- `tests/bench_e2e_ltx.py` -- end-to-end gen wall-time bench via
  ComfyUI HTTP API.
- `tests/bench_workload_profile.py` -- consumer-trace aggregator with
  coverage-gap analysis.

**Arc 4 -- build.sh `full` mode + Windows compile flags (-15 lines)**
- `build.sh`: dropped `./build.sh full` action that targeted Hopper +
  Blackwell arches we never validate. Default `./build.sh` targets
  `8.0;8.6;8.9` (Ampere + Ada); env override `CUDA_ARCHES`
  preserved for explicit Hopper/Blackwell builds.
- `setup.py`: dropped `os.name == "nt"` branches in `CXX_FLAGS`
  (MSVC `/O2 /openmp /std:c++17 /permissive-`) and
  `NVCC_FLAGS_COMMON` (`-D_WIN32=1 -DUSE_CUDA=1`). README narrows
  install to Linux+source; the Windows wheel paths upstream
  maintained are not validated here.

#### Why this is right

We own the fork; there's no upstream to send PRs to. Hopper/Blackwell
kernels won't run on sm89 even if compiled. Windows-build paths exist
upstream but are untested on this fork. Upstream's one-shape `bench/`
scripts measured one number with no rtol guardrails -- our LTX-shape
bench measures every kernel + every torch backend on production-
relevant shapes with `--check-regression` gating. Carrying unused
code inflates audit surface, build time, pyright noise, and the
"what does this fork actually do" question every reader has to
re-answer. The cost of removal is bounded; the cost of carrying it
recurs every time anyone reads the tree.

#### What's preserved

- sm80 kernel (forward-compatible to Ada via CUDA backward-compat;
  still powers `sageattn_qk_int8_pv_fp16_cuda`).
- sm89 kernels (the production hot path; `sageattn_qk_int8_pv_fp8_cuda`
  and the `++` variant).
- Triton kernels (mask-correct path; `sageattn_qk_int8_pv_fp16_triton`).
- The dispatcher's `arch in {"sm100", "sm120", "sm121"}` branch -- it
  routes to the existing sm89 kernel; one-line forward-compat for
  Blackwell users who happen to install this fork. Removing it would
  be churn with no win.

#### Verified, no action

- Dispatcher telemetry test -- 11 cases pass post-removal.
- Regression-check unit tests -- 7 cases pass post-removal.
- Build verify mode -- all expected extensions importable.

### v0.4.1 -- 2026-04-27  (bench coverage realigned to production; head_dim claim corrected)

Closes a load-bearing measurement bug. Every perf decision the fork
made before today was graded against `self_attn_large_704x704x497`
at `seq=31776, head_dim=64` -- a synthetic shape with the wrong
head_dim. LTX 2.3's video path is `attention_head_dim=128`
(`diffusers/models/transformers/transformer_ltx2.py:907-947`); the
`d=64` value was the audio side, mis-attributed to the whole model.
Production traces show zero calls at the synthetic shape.

The new bench coverage is sourced from a real consumer trace
(`sage_2026-04-26_105851.jsonl`, 6912 attention calls). The
`tests/bench_workload_profile.py` script (also shipped this session)
is the durable discovery tool -- every future trace gets routed
through it before bench-shape decisions.

#### Added

- **`tests/regression_baselines.json`** -- pinned `(shape, mode) ->
  (median_ms, mean_rtol)` baselines for the load-bearing rows. Schema
  documents `rtol_budget=0.10`, `perf_drift_pct=5.0`,
  `speedup_ratio_floor=1.5`. v0.4.1 baselines captured on RTX 4090 /
  sm89 / CUDA 13.0 / torch 2.11.0+cu130 / triton 3.6.0 / sage rev
  `8f737c3`.
- **`tests/test_sageattn_ltx_shapes.py --check-regression`** -- new
  CLI flag that grades fresh measurements against the baselines and
  exits non-zero on perf drift > 5%, rtol budget breach (> 0.10),
  speedup-ratio floor breach (sage_fp8++/torch_flash < 1.5x), or
  missing measurement on a load-bearing row. Soft `RTOL_DRIFT` alarm
  at 1.5x baseline (kernel-internal numerical change signal even when
  under the budget).
- **`tests/bench_workload_profile.py`** -- aggregates a consumer's
  sage trace JSONL into per-`(shape, has_mask, dispatched_kernel)`
  call counts, total wall time, and a coverage-check pass against
  `regression_baselines.json`. Surfaces "Coverage gaps" -- trace
  shapes the bench doesn't measure -- which is what justifies adding
  bench rows. Trace-freshness diagnostic mirrors the consumer's
  `sage_telemetry / legacy_inferred` bucketing.

#### Changed

- **`tests/test_sageattn_ltx_shapes.py` SHAPES list** replaced with
  production-aligned rows. New set: LTX 2.3 video self-attn at d=128
  (init seq=22932, loop seq=23296), audio self-attn at d=64 (same
  seq), short-Q paths at seq=497/498 (Gemma 3 text-encoder or audio
  cross-attn, attribution ambiguous from trace), K-probe pair at
  d=128 / kv=226 (the one masked row that survives -- doubles as
  v0.3.0 dispatcher mask-routing correctness witness). Synthetic
  wide-V stress shape kept (kernel robustness check, not a workload).
- **Dropped from SHAPES**:
  - `self_attn_large_704x704x497`, `self_attn_small_512x512x97` --
    speculative seq + wrong head_dim.
  - `cross_attn_text_kv32`, `kv64`, `kv128`, `kv512`, `kv1024` --
    re-measured a documented CUDA-mask-bug fingerprint without any
    gating purpose. Kept only `kv226` as the v0.3.0 dispatcher
    correctness witness + K-probe pair anchor.
  - `cross_attn_unmasked_kv226_kratio_probe` (old d=64 version) --
    replaced by the d=128 version at production seq.
- **CLAUDE.md** -- "Performance research / load-bearing metric"
  section updated. Primary row is now `ltx23_video_self_attn_init_22932 /
  fp8_cuda++` at 20.20 ms / 0.098 rtol. Speedup ratio at the
  production shape: 2.66x (vs old 2.62x at the synthetic shape; the
  perf story holds). "Shape coverage today" paragraph rewritten with
  the corrected video d=128 / audio d=64 split + cite to
  `transformer_ltx2.py:907-947`.

#### Measured (load-bearing)

RTX 4090 / sm89 / CUDA 13.0 / torch 2.11.0+cu130 / bf16:

| shape                                     | mode         | median_ms | mean_rtol | vs torch_flash |
|-------------------------------------------|--------------|----------:|----------:|---------------:|
| ltx23_video_self_attn_init_22932          | fp8_cuda++   |     20.20 |    0.0978 |          2.66x |
| ltx23_video_self_attn_loop_23296          | fp8_cuda++   |     20.52 |    0.0977 |          2.70x |
| ltx23_audio_self_attn_init_22932          | fp8_cuda++   |     10.65 |    0.0980 |          2.59x |
| ltx23_audio_self_attn_loop_23296          | fp8_cuda++   |     10.92 |    0.0977 |          2.53x |
| ltx23_short_q_init_497                    | fp8_cuda++   |      0.08 |    0.0934 |          0.45x |
| ltx23_short_q_loop_498                    | fp8_cuda++   |      0.09 |    0.0923 |          0.45x |
| ltx23_video_cross_unmasked_kv226_probe    | fp8_cuda++   |      0.74 |    0.0904 |          1.12x |
| ltx23_video_cross_text_kv226              | fp16_triton  |      1.16 |    0.0406 |       n/a (mask) |
| ltx23_video_cross_text_kv226              | auto         |      1.16 |    0.0406 |       n/a (mask) |

K-probe at d=128 / kv=226: K = 1.16 / 0.74 = **1.57** (vs 1.68 at
the old d=64 row; well below the 5x trigger for native CUDA mask
kernel work).

#### Findings worth flagging

- **Short-Q rows where sage loses to torch_flash.** seq=497 / 498
  fp8_cuda++ runs at ~0.45x of torch_flash's wall-time. int8 quant +
  kernel launch overhead exceeds the matmul work at that shape. This
  is the empirical evidence behind the consumer's `nodes_sage.py`
  deferred "min-sequence skip" backlog item -- the gate is now
  measurable. Trigger to act: a downstream consumer wires the
  short-Q skip and we re-measure end-to-end gen time.
- **Speedup ratio held up.** The "wrong head_dim" framing was a
  correctness-of-narrative bug, not a perf-magnitude bug. fp8++ at
  d=128 / production seq is 2.66x torch_flash, vs the old 2.62x at
  d=64 / synthetic seq. Sage's load-bearing claim is intact.
- **fp16_cuda is still mask-broken at d=128.** rtol 0.44 on
  `ltx23_video_cross_text_kv226 / fp16_cuda` -- the v0.3.1 soft-warn
  fires correctly. Same underlying bug fingerprint as the old d=64
  measurement.

#### Why this wasn't done sooner

The bench's primary shape was inherited from an earlier session
without checking it against a real trace. CLAUDE.md's "LTX-2.3:
head_dim=64" claim was treated as ground truth without grepping the
diffusers config. The discovery happened only because
`tests/bench_workload_profile.py` shipped this session and the first
trace it consumed produced zero `[HIT ]` lines on the load-bearing
set. The fix is to make the workload-profile coverage check the
default discovery tool before any future bench-shape decision.

### v0.4.0 -- 2026-04-26  (end-to-end gen-time bench harness)

Closes the load-bearing "kernel ms is not gen ms" gap that the
v0.3.x perf-research framework explicitly flagged. Until this lands,
every claim about sage-fork's perf impact was theoretical -- we
measured 19.95 ms on the primary kernel row but never showed that
translated into a real DiT render moving from X seconds to Y.

#### Added

- **`tests/bench_e2e_ltx.py`** -- end-to-end gen-wall-time bench via
  ComfyUI's HTTP API. Submits an LTX (or Flux / Z-Image) render
  workflow N times sage-on, N times sage-disabled, captures wall
  time per run, reads the consumer's sage trace JSONL (when
  `AUDIOLOOPHELPER_SAGE_TRACE=auto`), and reports:
  - median wall time per arm
  - speedup ratio: `wall_off / wall_on`
  - attention-fraction-of-step on the sage arm
  - interpretation: ≥ 1.5× = sage load-bearing on this workload,
    1.10–1.50× = helps but not dominant, < 1.10× = wash, < 0.95× =
    regression
  
  Prereqs: ComfyUI running, launched with the trace env var, and an
  API-format workflow JSON (saved via UI → Workflow → Save (API
  Format)). The script does not convert UI-format workflows -- the
  conversion is JS-side in ComfyUI's frontend; reimplementing adds
  enough complexity that one click in the UI is the better tradeoff.
  Mode toggle is via the `inputs.mode` field on the
  `AudioLoopHelperSageAttention` node, found by class_type so it's
  resilient to id renumbering across workflow versions.

- **Backlog entry** in `internal/PLAN.md`: "Simplify e2e bench
  correlation once consumer ships RUN_ID + prompt_id." Tracks an
  upcoming consumer-side change that bundles per-session artifacts
  under `data/runs/${RUN_ID}/` and stamps `prompt_id` on each sage
  trace row. When that lands, the bench drops ts-windowing entirely
  (~30 lines deleted; fence-post bugs eliminated; parallel-queue
  resilient). Until then ts-windowing is the correlation primitive.

#### Why this matters more than the kernel-level work that preceded it

The v0.3.x work fixed a real correctness bug (dispatcher mask
routing) and built measurement infrastructure (K-probe row,
soft-warn, telemetry helper). All of that is real but downstream
of an unverified premise: that kernel-level speedup translates to
gen-level speedup at all. This bench is the first instrument that
can verify (or disprove) that premise.

If the first execution shows speedup < 1.10×, the framework's
"kernel ms ≠ gen ms" caveat fires for real and the kernel-side
research priorities reset. If ≥ 1.5×, sage-fork's reason to exist
is empirically grounded for the first time.

### v0.3.1 -- 2026-04-26  (mask-gap follow-ups: soft-warn + K-ratio probe)

Two follow-ups graded against the load-bearing-metric framework added
to CLAUDE.md this session. Both pass the "ship now" bar; one
deliberate deferral got grounded in actual measurement instead of
hand-wave.

#### Added

- **Soft-warn from CUDA wrappers when `attn_mask` is passed.**
  `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`,
  and `sageattn_qk_int8_pv_fp8_cuda_sm90` now call a shared helper
  (`_warn_if_mask_passed_to_cuda_kernel`) that emits a one-time
  `warnings.warn` per source location when a non-None `attn_mask`
  reaches the wrapper. The dispatcher routes masked calls to triton
  automatically since v0.3.0; this guard catches consumers that
  bypass the dispatcher and hand-pick a `_cuda` kernel directly.
  Soft (warn, not raise) so consumers who defensively pass
  `attn_mask=None` aren't penalized -- the warn fires only on real
  masks. Python's default warning filter dedupes by source line, so
  long iteration loops emit one warning total per location, not one
  per call. Test:
  `tests/test_dispatched_kernel_telemetry.py::test_hand_picked_cuda_kernel_warns_when_mask_passed`.
  This was the deferred Solution C from v0.3.0's audit; ship-now
  reasoning recorded in the audit doc.
- **K-ratio probe row in the LTX bench.** New shape
  `cross_attn_unmasked_kv226_kratio_probe` in
  `tests/test_sageattn_ltx_shapes.py`. Same shape as
  `cross_attn_text_kv226` (the typical LTX text-encoder padded length)
  but with no mask. Lets us read off
  `K = triton_masked_ms / fp8++_unmasked_ms` directly from the bench
  output. K is the speedup ceiling for the deferred Backlog item
  "Add mask support to the sm80/sm89 CUDA kernels"; without a probe
  row, K was unmeasurable and the trigger could never fire. **First
  measurement (RTX 4090 / sm89 / CUDA 13.0 / torch 2.11):** K ≈ 1.68
  at kv=226 (triton 0.79 ms vs fp8++ unmasked 0.47 ms), K ≈ 2.0 at
  kv=1024. Both below the framework's 5x trigger; the deferred kernel
  work stays deferred, with an actual number behind it now.

#### Changed

- **CLAUDE.md "Performance research" section** -- the
  unmasked-vs-masked timing-gap framework item now names the K-probe
  row and records the measurement (see Added). Re-measure after any
  kernel-side change that lands on the unmasked cross-attn path.

#### Verified, no action

- **Dispatcher fix end-to-end.** Re-running the LTX bench post-fix
  shows `auto` rows on every masked cross-attn shape now matching
  the `fp16_triton` row to the precision the bench prints
  (mean_rtol 0.0392 / median_ms 0.79 at kv=226; same pattern at
  kv ∈ {32, 64, 128, 512, 1024}). Pre-fix `auto` would have mirrored
  `fp8_cuda++`'s broken fingerprint exactly.

#### Still deferred (with concrete reopen-trigger)

- **D: tighten `**kwargs` to explicit named parameters.** Re-evaluated
  this session; bigger than initially scoped. The dispatcher
  legitimately needs `**kwargs` for forward-compat (kernel-specific
  knobs like `pv_accum_dtype` that callers may want to override).
  Tightening per-kernel signatures creates a real conflict with
  dispatcher-forwarded kwargs. Real API design question with no
  current pain. Trigger unchanged: next time we touch these
  signatures for an unrelated reason.
- **Native CUDA-kernel mask support.** K-probe measurement (above)
  shows K ≈ 1.68-2.0 across the LTX cross-attn kv range. Days of
  kernel work for at most ~2x speedup on a path that's already
  sub-millisecond per call. Trigger is now grounded in a concrete
  per-bench measurement: re-evaluate when K > 5x at a shape a
  consumer actually hits.

### v0.3.0 -- 2026-04-26  (dispatcher mask routing -- correctness fix)

Closes the load-bearing inconsistency between what the fork documented
and what the dispatcher did. README and CLAUDE.md had claimed for
months that `sageattn()` "routes masked calls to the Triton kernel
transparently." The code routed purely by GPU arch and silently
dropped `attn_mask` on every CUDA path. A consumer-side workaround
covered the gap in practice; this version moves the fix to the right
layer (the dispatcher) so every consumer gets it without re-implementing.

Audit trail in `internal/audit_2026-04-26.md` (gitignored) -- captures
how the gap was missed, the alternatives considered, and the
revisit-triggers for the deferred items (loud-raise on hand-picked
CUDA kernels with masks; tightening the `**kwargs` surface; native
CUDA-kernel mask support).

#### Fixed

- **`sageattention/core.py::sageattn`** -- the top-level dispatcher
  now extracts `attn_mask` from `**kwargs` before the arch branch and
  short-circuits to `sageattn_qk_int8_pv_fp16_triton` when it's
  non-None, regardless of GPU arch. Unmasked calls dispatch by arch
  exactly as before. `is_causal=True` still dispatches by arch (CUDA
  kernels handle causal natively via `MaskMode::kCausal`); only
  `attn_mask` triggers the triton route. Side effect: `**kwargs` is
  now forwarded to every per-kernel call, so non-mask kwargs
  (`smooth_k`, `qk_quant_gran`, etc.) stop being silently swallowed.
- **End-to-end accuracy delta on cross-attn-with-mask shapes** (LTX
  cross-attn kv=226 example): `sageattn(q, k, v, attn_mask=m)` mean
  rtol drops from 0.4405 (broken: mask dropped, ran fp8++ unmasked)
  to 0.0391 (correct: routed to fp16_triton). Bare
  `sageattn_qk_int8_pv_fp8_cuda` still shows 0.44 -- the underlying
  CUDA-kernel mask gap is unchanged; this version routes around it,
  not through it. Known kernel bugs entry stays.

#### Added

- `tests/test_dispatched_kernel_telemetry.py::test_sageattn_dispatcher_routes_masked_calls_to_triton`
  -- enforces the new routing rule. Calls `sageattn()` with a
  text-padding-tail mask and asserts
  `get_last_dispatched_kernel() == 'fp16_triton'`. Failed red on the
  pre-fix dispatcher (recorded `'fp8_cuda++'`); passes green after
  the fix. Lives next to the existing dispatcher test so the next
  reader sees both the masked and unmasked invariants enforced
  side by side.

#### Changed

- **CLAUDE.md "The consumer surface"** -- the dispatcher's mask
  behavior is now described as a real implementation with a test
  reference, not as an aspirational claim. Cross-link to the audit
  doc + the new test added.
- **README.md** -- rewrite covering what changed, why, what was
  measured, and what tradeoffs the fork carries. The mask-gap
  language now describes the post-fix behavior (dispatcher routes;
  hand-picked CUDA kernels still drop). No-hype framing, numbers
  cited from `internal/log/log_2026-04-25.md` and the bench harness
  output.

#### Why this wasn't done sooner

The mask gap was discovered 2026-04-23 via the LTX-shape harness's
cross-attn rtol scaling signature. A consumer-side workaround
(downstream ComfyUI node patching the model's attention with a
mask-aware router) landed the same week because that was the fastest
path to a correct render. README + CLAUDE.md picked up an aspirational
"the dispatcher does this transparently" framing that drifted
unchallenged because no test enforced it. The fix here is small (~10
lines) and would have landed earlier if the dispatcher's mask
behavior had been pinned by a test from day one. The new test in this
version exists specifically to prevent the same kind of doc/code
drift from happening again.

### v0.2.0 -- 2026-04-25  (bench instrumentation, image-shape split, telemetry tooling)

A coherent chunk of measurement-surface work: the LTX-shape harness gained
FlashInfer + SpargeAttention rows, the image-gen shapes split into their
own file, the torch.compile spike got a re-runnable script with a clean
verdict, and the one-shot runner `tests/run_all.sh` ties it all together.
Conventions tightened: consumer-agnostic framing rule, project-internal
phase numbers don't ship, path-privacy hooks installed.

#### Added

- `sageattention.get_last_dispatched_kernel() -> str | None` -- public
  helper that returns the kernel-name string of the most recent
  `sageattn*` call on the current thread, or `None` if no call has
  happened yet on this thread. Stable short names exposed as module
  constants (`KERNEL_FP16_TRITON`, `KERNEL_FP8_CUDA_PP`, etc.) and
  enumerated in `KNOWN_KERNEL_NAMES`. Backed by a `threading.local()`
  set at the top of each entry point with the resolved kernel name --
  zero API change for callers who don't read the helper. Lets a
  downstream tracer record what sage actually dispatched to (instead
  of mirroring the routing table from `core.py::sageattn` or treating
  the kernel as opaque), which is the missing input the
  "mask-kernel work justified?" gate in a consumer-side summary needs
  to fire correctly. Read the value immediately after the sage call
  -- if your code yields (asyncio, or another sage call from the same
  thread) between call and read, the value can be overwritten.
  Verified end-to-end on RTX 4090 / sm89 / CUDA 13.0 / torch 2.11 via
  `tests/test_dispatched_kernel_telemetry.py`.
- `tests/run_all.sh` -- one-shot validation runner. Resolves the venv from
  `$VENV` or `$VIRTUAL_ENV`, snapshots the env to
  `internal/bench_env_<today>.txt`, runs the LTX bench, the image bench,
  and the torch.compile spike in sequence; archives logs under
  `internal/log/`. `set -euo pipefail`.
- `tests/test_sageattn_image_shapes.py` -- companion to
  `test_sageattn_ltx_shapes.py`, holds head_dim ∈ {120, 128} shapes
  (Z-Image-Turbo S3-DiT, Flux-class). Reuses the LTX file's
  `run_shape_sweep()` helper; ~50 lines.
- `tests/test_sageattn_ltx_shapes.py` -- new bench rows on top of v0.1.0:
  * FlashInfer fp16 prefill row (optional; SKIPs cleanly when not
    installed). Predicted to lag sage fp8++ on sm89 because CUTLASS
    lacks native fp8 below sm90.
  * SpargeAttention top-k=0.5 row on unmasked self-attn shapes only
    (Sparge inherits sage's mask gap; SKIPs when `spas_sage_attn` not
    installed).
  * `run_shape_sweep(shapes)` extracted as the per-shape engine so
    `test_sageattn_image_shapes.py` reuses it without duplicating
    ~85 lines of scaffolding.
- `tests/spike_torch_compile.py` -- re-runnable spike measuring whether
  `torch.compile` around sage produces bounded mean-rel-error AND
  speedup. Verdict on torch 2.11: keep the consumer-side
  `torch.compiler.disable()`. Both compile modes drift ~2.8% vs eager,
  consistent across modes (autocast or op fusion around sage's int8/fp8
  dispatch). Reopen-trigger: "compile produces bounded rtol AND a
  measurable speedup" on a future torch release.
- `internal/bench_env_2026-04-25.txt` -- env snapshot (torch
  2.11.0+cu130, triton 3.6.0, sage editable, RTX 4090 / sm89, CUDA 13.0)
  locking the version surface so later phase deltas are real perf
  changes.

#### Measured

First-measurement datapoints on RTX 4090 / CUDA 13.0 / torch 2.11 /
bf16, captured during this version's work:

- **self-attn-large** (31776×31776, head_dim=64, no mask): sage fp8++
  19.95 ms, torch_flash 52.23 ms (2.62x), torch_cudnn 53.98 ms (2.72x).
  ~1.4% drift from the v0.1.0 baseline of 19.67 ms (cu128 -> cu130 +
  triton 3.6 upgrade); within run-to-run noise. Yardstick is now
  19.95 ms.
- **image_gen 4096×4096 h24 d128** (Flux-class): sage fp8++ 0.64 ms vs
  torch_flash 1.31 ms (2.05x). Closes the "do we need a per-model-class
  router branch?" question with a no.
- **z_image_turbo 4608×4608 h32 d120** (S3-DiT single-stream): sage
  fp8++ 1.32 ms vs torch_flash 2.23 ms (1.69x). Confirms sage's CUDA
  kernels handle the non-power-of-2 head_dim=120 cleanly.
- **cross-attn + mask** rtol fingerprints (CUDA-mask-bug signature)
  unchanged from v0.1.0.

#### Changed

- `README.md` and `CLAUDE.md` -- reframed to be consumer-agnostic. Sage
  is a general PyTorch attention library; the fork compiles cleanly
  into any consumer. README now lists what's in the fork beyond the
  upstream (bench harness, compile spike, warmup API, autotune
  addition); CLAUDE.md TLDR states the two purposes (editable install
  + experimentation/measurement surface). Conventions added: consumer-
  agnostic framing in committed material, project-internal phase
  numbers don't ship, path discipline enforced by the path-privacy
  plugin's pre-commit hook.

### v0.1.0 -- 2026-04-23  (post-squash baseline)

Initial fork divergence from `woct0rdho/SageAttention` after the
history squash. Everything below is what makes this fork different
from upstream as of the squash commit.

#### Added

- `setup.py` -- `_qattn_sm80` is now built when compute capability 8.9
  (Ada) is detected. Framed as a regression fix from
  `woct0rdho/SageAttention`: thu-ml's setup.py gates the SM80 extension
  on `HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or
  HAS_SM120 or HAS_SM121` (Ampere + Ada + Hopper + Blackwell), but
  woct0rdho's refactor collapsed that to a tuple gate `("8.0", "8.6",
  "8.7")` -- which silently drops Ada, Hopper, AND Blackwell.
  Ada-only source builds on woct0rdho's fork lose
  `sageattn_qk_int8_pv_fp16_cuda` (the fp16 fallback). We added `"8.9"`
  because that's the arch we test and care about; widen the tuple to
  match thu-ml's coverage if you run this fork on Hopper or Blackwell
  and want the fp16 fallback built from source.
- `sageattention/core.py::sageattn_warmup(shapes, kernels=...)` --
  public API that fires one-shot dispatches per (kernel, shape) to
  prime Triton's JIT + autotune cache. Cuts ~1s first-call latency on
  sm89 to ~2ms post-warm. Defaults to the Triton kernel only (CUDA
  kernels are build-time compiled, no warmup benefit).
- `sageattention/triton/attn_qk_int8_per_block.py` -- added
  `@triton.autotune` over `num_warps in {4, 8}` and
  `num_stages in {3, 4, 5}`, keyed on runtime shape. BLOCK_M/BLOCK_N
  stay hardcoded because they're locked by the per-block int8
  quantization step in `sageattention/quant.py`. Measurement on RTX
  4090 / LTX shapes: autotune confirmed the existing hardcoded config
  (`num_warps=4`, `num_stages=3` for `head_dim=64`) was already at the
  optimum -- zero perf delta then. Value is structural: auto-adapts to
  future kernel / triton / shape shifts.
- `build.sh` -- editable-install wrapper. Enforces `VIRTUAL_ENV`, pins
  `uv pip install --python ${VIRTUAL_ENV}/bin/python`, compiles for
  Ampere + Ada (`TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`) by default. Caps
  `MAX_JOBS` at 8 to keep high-core boxes from OOMing during
  `_qattn_sm89` compilation.
- `tests/test_sageattn_ltx_shapes.py` -- LTX-2.3-shape accuracy and
  speed harness. Measures every installed sage kernel and three torch
  SDPA backends against `SDPBackend.EFFICIENT_ATTENTION` at LTX's
  actual shapes (head_dim=64, heads=32, self-attn + cross-attn-with-
  mask across seq_kv from 32 to 1024, plus a synthetic wide-V shape).
  Reports mean/max rtol+atol and median elapsed. Soft-warns when
  mean_rtol > 0.10. Cross-kernel `fp8++vs.triton` consistency row on
  unmasked shapes: mean_rtol ~0.10 across self-attn shapes, equal to
  the combined-noise floor (triton ~0.04 + fp8++ ~0.09 vs SDPA, added
  in quadrature). No hidden discontinuity; mixing is safe.
  First-measurement datapoints (RTX 4090 / CUDA 13.2 / torch 2.11 /
  bf16): self-attn-large (31776×31776, no mask) sage fp8++ 19.67 ms vs
  torch_flash 52.39 ms (2.7x), torch_cudnn ~360 ms (cuDNN FA3 path not
  competitive on sm89); cross-attn + mask (kv=226) sage fp16_triton
  0.78 ms vs torch_cudnn 2.20 ms (2.8x). Sage remains load-bearing on
  sm89.
- `tests/repros/repro_cuda_mask_kernel.py` -- standalone repro for the
  CUDA mask-path missing-feature documented in Known kernel bugs.
- `CHANGELOG.md`, `CLAUDE.md` -- this file plus the fork navigation
  guide.

#### Changed

- `README.md` -- reduced to attribution only (immediate fork:
  `woct0rdho/SageAttention`; original: `thu-ml/SageAttention`) plus a
  short build pointer. Windows-specific installation prose and wheel
  selection guidance removed -- this fork builds from source.
