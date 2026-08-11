last updated: 2026-08-04

# sage-fork

*sm89 kernel optimization for ComfyUI consumer workloads.*

*Based on [SageAttention](https://github.com/thu-ml/SageAttention) by thu-ml,
via [woct0rdho's fork](https://github.com/woct0rdho/SageAttention), with
low-level CUDA primitives adapted from
[FlashInfer](https://github.com/flashinfer-ai/flashinfer). This project has
diverged substantially since and is no longer a thin patch on either; treat
it as its own thing. Full attribution in [NOTICE](NOTICE); Apache-2.0 per the
upstream lineage.*

A sm89 / RTX 4090 kernel optimization and measurement surface for
ComfyUI consumer workloads. The mission is to make the workflows we
actually run faster, more memory-efficient, and more measurable --
anchored in DiT-class diffusion (LTX 2.3 video, Flux / Z-Image image
gen) and expanding to multi-modal pipelines as those become consumer
workload classes worth attacking. See `VISION.md` for the full
scope framing.

Sage attention is the historical foundation and remains a primary
deliverable: INT8-quantized Q/K with FP8 PV accumulation, runtime-
dispatched kernel selector that picks the right variant for the GPU
+ CUDA combination it finds, Triton fallbacks for paths the native
kernels don't cover. v0.6 added `sage_ffn` (fp8-native fused MLP for
DiT FFN blocks). Forward directions span attention, FFN, VAE,
ComfyUI integration shims, workflow profiling, persistent-CTA
rewrites, and whatever the load-bearing measurement says next. The
repo name is "sage-fork" for historical reasons; the substantive
scope is broader. See `VISION.md` for the full mission framing.

**The hard constraint: sm89 / Ada / 4090 only.** Kernels compile and
run on other archs via dispatcher fallbacks (sm80 forward-compat,
sm100 / sm120 / sm121 through the sm89 path), but the bench
baselines, the rtol expectations, and the perf-decision criteria are
all calibrated for sm89. Treat results elsewhere as "should work"
rather than "validated."

---

## What's in the box

- **`sageattn(q, k, v, ...)`** -- a top-level dispatcher that picks
  a kernel based on `(arch, CUDA version, mask presence)`. Most
  consumers should just call this and let it decide.
- **Specific kernel exports** -- `sageattn_qk_int8_pv_fp8_cuda`,
  `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp16_triton`.
  Bypass the dispatcher if you want to pick the kernel yourself.
- **Native CUDA mask support on sm89 fp8++** (v0.5.5), *reachable
  through ComfyUI as of v0.7.0*. ComfyUI gates masked calls on
  `"attn_mask" in inspect.signature(sageattn).parameters`; ours lived
  in `**kwargs`, so that probe read False and every masked call went to
  torch SDPA. The kernel was correct the whole time and simply never
  ran. If you are on < v0.7.0, you do not have this in practice.
- **`sageattn_consume(qkv, ...)`** (v0.7) -- `sageattn()` that takes
  ownership of q/k/v and releases each float tensor as soon as its
  quantized form exists. A normal call cannot do this: the caller's frame
  owns the references. Takes either a `[q, k, v]` list, which it empties,
  or three single-owner containers exposing `peek()`/`take()`, which is
  what ComfyUI hands an attention backend (v0.7.5).

  What it saves is configuration-dependent, and the configuration DiT
  blocks actually use is the one where it saves nothing. Peak per call at
  MiniMax H3's fl2va shape: -858 MiB with separate allocations and
  `smooth_k=False`, -287 MiB at the shipped `smooth_k=True`, and **0
  against a fused QKV buffer**, because releasing q and k frees nothing
  while v still references the same allocation. An earlier "~435 MiB in
  the fused case" figure was wrong and is retracted; see CHANGELOG v0.7.3.

  A caller gets it back by cloning v before handing over, which gives v
  its own storage and converts the fused case into the separate one for
  the price of a third of the buffer: -286 MiB at that shape, against a
  flat +572 MiB cost if the release does not happen. Gate that clone on
  **`sageattn_consume_prefers_cloned_v(device)`** (v0.7.4) rather than on
  an arch check of your own, so a future change to when the release pays
  reaches you on upgrade instead of drifting silently.
- **`sageattn_partitioned(q, k, v, slices)`** -- amortizes K-quant +
  V-cast across multiple Q slices sharing the same K, V. Targets
  multi-slice partition patterns; correctness verified, peak HBM
  benefit is workload-dependent and currently looks small (see
  `tests/bench/partitioned_mask_phase0/` for the measurement).
- **`fused_rope_split(q, k, freqs_cis)`** -- clean-room Triton
  kernel matching the LTX split-rotary-embed convention; standalone
  helper, not bolted into `sageattn()`.
- **`sage_ffn(x, w1, s1, w2, s2, b1=None, b2=None)`** (v0.6) -- a
  two-kernel fp8-native fused MLP for DiT FFN blocks with per-tensor
  fp8 (E4M3FN) weights. Targets LTX 2.3 distilled. **Ships as a
  completeness primitive, not a perf win**: synthetic-bench shows
  1.26-1.36x vs torch's fp8-dequant path, but a two-sampler LTX
  production A/B came back +1.79% e2e slower (+20% at stage-2
  per-call) -- the synthetic-vs-in-pipeline gap the perf-research
  framework calls Cell C (defined in
  `docs/perf_research_framework.md`). Available for users who
  specifically need fp8-native fused MLP on sm89; no other library
  provides this combination. See "What we've measured" for the
  production breakdown.
- **A bench harness** -- `tests/test_sageattn_ltx_shapes.py` measures
  every sage kernel + torch SDPA backend (FLASH / EFFICIENT / CUDNN)
  at the LTX-class shapes our models actually hit, reporting both
  accuracy (rtol vs SDPA) and speed (median ms over 3 runs).

---

## Install

Linux + an active venv only. The build pins the install to whatever
venv is currently active so multiple installs don't collide.

```bash
source /path/to/your/venv/bin/activate
./build.sh                 # builds for Ampere + Ada (TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9)
./build.sh clean           # wipe prior .so / build/ artifacts first
./build.sh verify          # import-check, no rebuild
```

Build is 60-90s on an 8-core box with `MAX_JOBS=8` (the script caps
at 8 because uncapped nvcc parallelism OOMs on the sm89 kernel
compile). Confirm the editable install is live:

```bash
${VIRTUAL_ENV}/bin/python -c "import sageattention, os; print(os.path.dirname(sageattention.__file__))"
```

Should print a path inside this repo.

---

## Quick start

The dispatcher does the right thing by default on sm89:

```python
import torch
from sageattention import sageattn

q = torch.randn(1, 32, 23296, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Unmasked self-attn -- lands on the fp8++ CUDA kernel on sm89+CUDA>=12.8.
out = sageattn(q, k, v)

# With a mask -- also lands on fp8++ CUDA on sm89+CUDA>=12.8 (as of v0.5.5);
# falls back to the mask-correct Triton kernel on other archs.
mask = torch.zeros(1, 1, 1, 23296, device="cuda", dtype=torch.bfloat16)
out_masked = sageattn(q, k, v, attn_mask=mask)
```

The kernel actually used can be read back per-thread for telemetry:

```python
from sageattention import get_last_dispatched_kernel
print(get_last_dispatched_kernel())  # 'fp8_cuda++', 'fp16_triton', etc.
```

---

## What we've measured

Setup: RTX 4090, CUDA 13.0, torch 2.11, bf16 inputs. Speed = median ms
over 3 timed runs after 1 warmup. `MATH` SDPA backend OOMs at LTX
self-attn scale, so the accuracy reference is `SDPBackend.EFFICIENT_ATTENTION`.

### Unmasked self-attn

Per-kernel speedup at synthetic LTX-class shapes (median over 3 timed
runs). **These are isolation measurements, not e2e wall-time deltas
on a render** -- the e2e contribution depends on the workload's
attention share. Per-workload e2e numbers below.

| shape                                    | sage fp8++ | torch_flash | speedup |
|------------------------------------------|-----------:|------------:|--------:|
| LTX self-attn (31776x31776, h=32, d=64)  |  19.95 ms  |   52.23 ms  |  2.62x  |
| Flux-class self-attn (4096x4096, h=24, d=128) |  0.64 ms |    1.31 ms  |  2.05x  |
| Z-Image-Turbo S3-DiT (4608x4608, h=32, d=120) |  1.32 ms |    2.23 ms  |  1.69x  |

Quantization-induced rtol is ~0.097 on these shapes (well below the
0.10 line we treat as the acceptable ceiling for DiT generation
work). In practice this is below VAE noise on the image/video gen
workloads we've tested; we haven't run task-level quality benchmarks.

E2e ratio for the iclora workflow (downstream consumer A/B
2026-05-07, attention share ~42% of CUDA kernel time): measured
1.41x wall ratio, matches pure-Amdahl prediction within 1.4%. For
the FML2V multi-guide workflow: stage-2 attn1 is the single
heaviest sub-module and gives a materially larger e2e lever than
the FFN-side primitive. The canonical breakdown + FFN-share triplet
(three distinct readings depending on the question being asked) is
in `docs/ltx_workload_profile.md`.

### MiniMax H3 (v0.7) -- the cleanest e2e result we have

H3 is a single-stream AV DiT: one unmasked self-attention per block over
the whole packed `[text | cond | audio | video]` sequence, 56 heads,
head_dim 128, 50 blocks. No kernel work was needed -- the existing sm89
fp8++ kernel covers it as-is.

Full render through ComfyUI at the bundled i2v template's settings
(1344x768, 20 steps, `res_multistep`/`simple`, `int8_convrot` weights),
warmup discarded, arms alternating on a shared seed:

| length | sampler | total render | note |
|---|---|---|---|
| 73 frames | 1.70x | 1.62x | 152s -> 94s |
| 124 frames | **1.91x** | **1.83x** | 360s -> 197s |

Paired runs agreed within 0.3s. **The speedup grows with clip length**
(attention is quadratic in sequence, the rest is not) while per-call
accuracy stays flat, so longer clips are strictly the better case.
Profiling one forward: attention 47.5%, int8 linears 40.7%, weight
streaming only 4.5% -- so this is compute-bound, not PCIe-bound, and
attention is still the largest single cost even after the win.

Peak VRAM ~20.6 GB of 24 GB at length 73. Consumer node lives separately
in [ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations),
per the "sage-fork stays primitive" rule. Its
[SOLATTN.md](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/SOLATTN.md)
carries the sage + Sol-Attn stacking experiments.

### An accuracy calibration worth knowing

Every rtol figure in this README comes from a synthetic bench over
`torch.randn` inputs. On **real captured activations** the same fp8++
kernel measures **0.026, roughly 4x lower**, and the fp8++-to-fp16 gap
narrows from 2.6x to 1.3x. Real attention has structure -- concentrated
softmax, correlated keys -- that quantization handles far better than
iid gaussian noise.

So the synthetic numbers below are a **pessimistic bound, not an
estimate**. Do not read 0.098 as a quality budget. It also means a
synthetic sweep cannot answer questions about input *distribution*: our
first `smooth_k` experiment reported "no effect" on `torch.randn`, which
has zero mean by construction and therefore no channel offset for
`smooth_k` to remove. On real K the offset is substantial
(|mean|/std 0.68) and it still does not help, most likely because
`per_thread` quantization granularity already handles it.

### Masked self-attn (post-v0.5.5)

Before v0.5.5, the sm89 CUDA kernels silently dropped `attn_mask` --
the C++ `MaskMode` enum only had `{kNone, kCausal}` and the pybind
layer never wired the parameter through. Masked calls produced
rtol that scaled with `1 / seq_kv` (the silent-drop fingerprint:
0.94 at kv=64, 0.13 at kv=1024). The Triton kernel was the only
mask-correct path.

v0.5.5 added native general-mask support on the sm89 fp8++ kernel
(`MaskMode::kGeneral` + an `apply_general_mask` helper in
`csrc/qattn/attn_utils.cuh`). Masked rtol on the same kv sweep is
now ~0.09 across the range -- matching the fp8++ unmasked-vs-Triton
floor. The dispatcher routes masked sm89+CUDA>=12.8 calls to the
new path automatically; other archs still use the Triton fallback.

### Preliminary in-pipeline observation

We ran A/B comparisons on a real LTX 2.3 multi-guide workflow at
768x512x97 on a 4090 with dynamic VRAM disabled. Updated count
after additional repetitions:

| arm | outcome |
|---|---|
| fp8_cuda++ masked path + FFN chunking ON | N=3+ success, 0 OOM |
| Triton masked fallback + FFN chunking ON | N=1 success, N=2 OOM (non-deterministic) |
| fp8_cuda++ masked path + FFN chunking OFF | deterministic OOM at stage-2 FFN GELU |

Both Triton OOMs hit `AdaLNSingle.linear` (downstream of attention) --
727 MiB requested, ~16 MiB free, after 48 masked dispatches. The
chunking-off fp8++ OOM hits the FFN GELU projection at the
multi-guide expanded shape (proj output `(1, 44880, 16384)` bf16 ≈
1.47 GiB).

**Honest reading**: at this workload scale on 24 GiB, the
`LTXVChunkFeedForward` FFN-chunking node is doing the heavy lifting
on peak memory. *With chunking on*, sage choice matters at the
margin -- the v0.5.5 CUDA mask path has more headroom for the
attention-side delta than the Triton fallback. *Without chunking*,
both kernels hit a different (FFN-intermediate) memory wall.

So the in-pipeline observation is "with FFN chunking enabled, the
v0.5.5 CUDA mask path tolerates the workload more reliably than
the Triton fallback (N=3+ vs N=1 success in the observed sample)."
This is preliminary, small-N, and contingent on chunking being
present. **Don't take "the fork fixes OOM" as established** -- take
it as "looks promising at the margin, more testing needed, and FFN
chunking is doing most of the load-bearing memory work upstream."
The A/B recipe is reproducible: same workflow + flip the sage
routing flag + ComfyUI flags `--disable-dynamic-vram
--disable-async-offload --reserve-vram 0 --cuda-malloc --cache-none`.
Independent reproduction welcome.

### sage_ffn (fp8-native fused MLP, v0.6)

`sage_ffn` is a separate primitive from the attention kernels --
two Triton kernels (`Linear -> GELU(tanh)` then `Linear`) computing
in fp8 against per-tensor-fp8 weights. The wedge is qualitative:
torch's `F.linear` against fp8 weights dequants to bf16 before the
matmul (paying 2x weight bandwidth and using bf16 tensor cores at
~330 TFLOPS); `sage_ffn` loads fp8 directly and uses sm89 fp8
tensor cores at ~660 TFLOPS. No other library ships an fp8-native
fused MLP for these consumer-app DiT shapes on sm89 (FA's
`fused_mlp_func` is bf16/fp16 only).

LTX 2.3 distilled FFN shapes (hidden=4096, inner=16384), bias-inclusive
(matches the LTX 2.3 distilled checkpoint), measured on RTX 4090,
CUDA 13.0, torch 2.12.0+cu130, triton 3.7.0 -- **synthetic standalone
bench, not end-to-end ComfyUI rendering**:

| shape | sage_ffn | torch ref (fp8-dequant) | speedup | mean_rtol |
|---|---:|---:|---:|---:|
| stage-1 (T=10780) | 13.3 ms | 18.1 ms | **1.36x** | 0.091 |
| stage-2 (T=44880 multi-guide) | 59.8 ms | 75.3 ms | **1.26x** | 0.091 |

mean_rtol is well under the 0.10 budget. The reference is
`F.linear(F.gelu(F.linear(x, w1_bf16), approximate="tanh"), w2_bf16)`
with weights dequantized once outside the timing loop, so this is
torch's *best-case* fp8-weight path, not its naive one.

**Production result on a two-sampler LTX workflow: sage_ffn is
slower than the chunking-only baseline. Ships as completeness
primitive, not a perf win.**

In-pipeline A/B on a two-sampler FML2V multi-guide workflow
(768x512x97, 8-step stage-1 + 3-step stage-2, 4 renders
interleaved baseline/treatment/baseline/treatment on a 4090 under
`nodynvram`):

| metric | baseline | with sage_ffn | delta |
|---|---|---|---|
| wall-time avg | 148.51s | 151.17s | **+1.79% slower** |
| ff @ T=10780 med ms/call | 10.36 | 10.67 | +3.0% slower |
| ff @ T=42240 med ms/call | 48.77 | 58.58 | **+20.1% slower** |

Same workflow / prompt / seed across both sides; interleaving
controls for time-varying noise; non-FFN sub-modules at 1.00x
ratio confirm the patching surface is clean. Per-call FFN times
match between cold-autotune and warm-autotune treatments, so
autotune amortization is not the explanation.

Why the synthetic 1.26-1.36x didn't translate:

1. **L2 cache contention with neighboring sub-modules.** Synthetic
   bench ran FFN alone with warm L2. Production runs `attn1` (~107
   ms at T=42240) immediately before `ff` at stage-2; the attention
   pass evicts FFN's L2 residency. The X-tile-lives-in-L2
   assumption breaks when L2 is hostile; cold-L2 FFN is
   bandwidth-bound and loses the fp8-vs-bf16 advantage. Worse at
   stage-2 (4x working set) matches the regression shape.
2. **Cumulative kernel-launch overhead at LTX call count.** LTX
   2.3 fires ~1056 ff calls per render across transformer blocks.
   sage_ffn is two kernel launches per call; torch reference is
   one cuBLASLt call per matmul.

The v0.5.5 precedent played out a second time -- synthetic kernel-
bench projects a wedge, in-pipeline A/B reveals production
conditions change the picture. Different workload shapes (e.g.
single-pass, non-multi-guide) may behave differently; in-pipeline
measurement is the gate.

Design notes:

- Two-kernel split, intermediate hits HBM between them. This is
  the same design FA's `fused_mlp_func` uses on bf16/fp16 -- the
  single-kernel "intermediate never hits HBM" design hits an
  sm89 SMEM wall at LTX-class K dims.
- Plain GELU MLP only in v0.6. No gated SwiGLU/GEGLU variant.
- Bookend bf16 blocks (LTX 2.3 keeps blocks `{0, 1, 46, 47}` as
  bf16) need consumer-side dispatch -- `sage_ffn` only handles
  fp8-weight blocks; the bf16 bookend blocks fall through to
  `F.linear` in the caller.
- First call at a new shape pays ~10-15s Triton autotune-search per
  kernel (~30s total across both kernels at the two LTX shapes);
  subsequent calls hit the on-disk cache. Configs are hardcoded
  winners from a broader sweep so that first-render cost stays
  bounded.
- v0.6.1 candidates for closing the production gap: persistent-CTA
  hybrid (addresses L2 contention directly) and a CUTLASS-based
  CUDA backend (closes the Triton-vs-cuBLASLt codegen gap). See
  CHANGELOG Backlog.

### Things we have NOT measured

- Task-level quality (FVD / FID / preference) on any of the
  rtol-degraded shapes. We measure rtol against SDPA; rtol below
  ~0.10 has been "fine" on image/video gen in our hands but that's
  observational.
- Real-pipeline behavior on archs other than sm89. The kernels
  compile and run on sm80 / sm100 / sm120 / sm121 via the
  dispatcher, but none of those are in our bench loop.
- `torch.compile` of attention. The spike at
  `tests/spike_torch_compile.py` rejects the wrap on bounded
  rtol grounds (Dynamo graph-breaks at our fused pybind kernels
  cause precision drift). Consumer-side `torch.compiler.disable()`
  around sage stays the recommendation; re-run after torch upgrades.

---

## What's open

Tracked in `CHANGELOG.md` under "Known kernel bugs" + "Backlog".
Summary of the things worth knowing:

- **int32 offset overflow in `quant_per_block_varlen.py`.** Inherited
  from upstream: the Triton quant kernels compute element offsets in
  int32, which wraps past 2^31 elements (4 GiB at bf16) -- silently
  zeroing the tail in NHD, faulting in HND. Fixed in
  `quant_per_thread.py` and `quant_per_block.py` in v0.7.0 via a
  per-launch `USE_I64` specialization, so ordinary shapes keep int32
  addressing. The varlen module is deliberately unfixed: its bound
  needs `cu_seqlens` plumbed to the host and no ComfyUI path reaches
  it. Reachable only above ~300k packed rows at H3's head config.
- **sm80 + non-fp8++ sm89 CUDA paths still drop masks.** Same kernel
  pattern as the v0.5.5 fix; deferred until a workload hits one of
  those paths frequently. The dispatcher routes around the gap on
  sm89; sm80 still routes to Triton for masked calls.
- **No whole-block-skip on sparse masks.** Triton has it
  (`tl.max(mask_block) == 0 -> skip`); the CUDA pipelined K-iteration
  loop makes the analog non-trivial. Currently relevant only for
  workloads we haven't measured.
- **Persistent-CTA hybrid for stage-2 attention** (highest e2e lever,
  ~15% wall-time ceiling on LTX multi-guide workloads) and **for
  sage_ffn** (validates the technique at lower risk). Both deferred;
  see CHANGELOG Backlog for triggers. CUTLASS-based fp8 matmul backend
  was queued and is now demoted to "skip per workload-profile analysis"
  -- the v0.6 production gap was L2 contention + dispatch overhead,
  not matmul codegen.
- **Mask-aware autotune key** as measurement-hygiene infrastructure
  (1-2 hour change, recommended regardless of larger work).

---

## Tradeoffs

You get:

- 2-2.7x per-call speedup over torch's flash backend on sm89 self-attn
  at the DiT-class shapes we validated (head_dim ∈ {64, 120, 128}).
  Synthetic kernel-bench measurement; e2e wall-time wedge depends on
  the workload's attention share. Measured: 1.41x e2e on the iclora
  workflow at ~42% attention share, matches pure-Amdahl within 1.4%.
- A faster cross-attn path via `sageattn_qk_int8_pv_fp16_triton`
  (~2.8x over `torch_cudnn` at LTX cross-attn shapes). Same caveat:
  per-call, not e2e.
- Native mask support on the sm89 fp8++ CUDA path -- masked calls
  run at fp8++ speed instead of paying the Triton fallback
  overhead. (Other archs still use Triton.)
- An fp8-native fused MLP primitive (`sage_ffn`, v0.6) for LTX
  2.3-class FFN blocks. The only fp8-native fused MLP available
  for these workloads on sm89. **Note**: synthetic-bench shows
  1.26-1.36x but a two-sampler LTX production A/B came back
  net slower; ships as a completeness primitive only. See
  "What we've measured" for detail.

You give up:

- **Mask correctness when you hand-pick a non-fp8++ `_cuda` kernel.**
  The dispatcher routes around this for you, but if you call e.g.
  `sageattn_qk_int8_pv_fp8_cuda(q, k, v, attn_mask=m, pv_accum_dtype="fp32+fp32")`
  directly, the mask is silently dropped and you get a soft warning.
  Use the dispatcher or `sageattn_qk_int8_pv_fp16_triton` for masked
  calls if you're picking by hand.
- **bf16/fp16 input only.** No fp32 input path.
- **`torch.compile` around sage.** Wrap with `torch.compiler.disable()`
  until the spike's verdict flips.
- **One platform's worth of validation.** Ada / sm89 only.
- **No assertion about quality.** Our quality claim is "rtol < 0.10
  vs SDPA on the shapes we care about." That's a numerical
  invariant, not a perceptual one. The one perceptual check we have
  run: same-seed 20-step H3 renders with sage on and off were
  indistinguishable to the maintainer's eye. That is one prompt, one
  model, one pair of eyes -- not a quality benchmark. Note also that
  comparing finished renders *numerically* measures trajectory chaos
  rather than degradation, since any perturbation diverges a 20-step
  ODE; the honest instruments are fixed-input kernel divergence and
  human judgement, not PSNR between samples.

---

## Hardware target

sm89 / RTX 40xx / Ada only as a first-class target. Other archs
(sm80 Ampere, sm100 / sm120 / sm121 Blackwell-via-fallback) compile
and run via the dispatcher's fallback paths. We don't actively test
or maintain those paths -- if something breaks on sm80, we'll likely
fix it, but the bench harness won't catch the regression first.

Linux + source build only. We don't carry Windows or Mac install
paths.

---

## Layout

```
sageattention/          # Python package
  core.py               # dispatcher + Python entry points
  triton/               # JIT Triton kernels
    fused_mlp_fp8.py    # sage_ffn -- v0.6 two-kernel fp8 fused MLP
    fused_rope.py       # fused_rope_split helper
  sm89_compile.py       # torch.library.custom_op schemas for sm89 kernels
  quant.py              # quantization helpers
csrc/qattn/             # CUDA kernel sources (sm80 + sm89)
csrc/fused/             # fused pre-kernels (transpose/pad/permute, scale-fuse-quant, ...)
tests/                  # bench + correctness scripts (no pytest; standalone runners)
tests/bench/            # focused micro-benches (partition-pattern peak HBM, etc.)
docs/                   # deeper design docs
CLAUDE.md               # the day-to-day routing index
VISION.md               # what this is, what it isn't, the load-bearing metric
CHANGELOG.md            # versioned changes + Known kernel bugs + Backlog
```

---

## Caveats and posture

This repo is maintained as a personal kernel-research surface for
one user's workflows. It ships when it ships; there's no release
calendar, no stability commitment beyond "the things in the bench
harness keep working." If you depend on a specific kernel signature,
pin a commit -- internal symbols may move.

Numbers in this README are measured on a single 4090 and represent
our load-bearing shapes. Your shapes may behave differently,
especially at the small-cross-attn end where launch overhead
dominates. Use the bench harness against your actual shapes before
making routing decisions.

---

## License

Apache 2.0, per the SageAttention origin lineage.
