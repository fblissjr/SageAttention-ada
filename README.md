last updated: 2026-05-15

# SageAttention-ada

*Based on [SageAttention](https://github.com/thu-ml/SageAttention) by thu-ml,
via [woct0rdho's fork](https://github.com/woct0rdho/SageAttention). This
project has diverged substantially since and is no longer a thin patch
on either; treat it as its own thing.*

An attention-kernel library for **DiT-class local generation on sm89 /
RTX 40xx / Ada** -- LTX 2.3 video, Flux-class image, Z-Image-Turbo, and
similar models that run on a single 4090.

The core is INT8-quantized Q/K with FP8 PV accumulation, running
through a runtime-dispatched kernel selector that picks the right
variant for the GPU + CUDA combination it finds. There are also
Triton fallbacks for archs without the native kernels and a couple of
extra primitives (a fused split-RoPE, a multi-Q-slice attention entry)
that downstream consumers occasionally find useful.

We care about **one GPU**: sm89 / Ada / 4090. The kernels compile and
run on other archs via dispatcher fallbacks (sm80 forward-compat, sm100
/ sm120 / sm121 through the sm89 path), but the bench baselines, the
rtol expectations, and the perf-decision criteria are all calibrated
for sm89. Treat results elsewhere as "should work" rather than
"validated."

---

## What's in the box

- **`sageattn(q, k, v, ...)`** -- a top-level dispatcher that picks
  a kernel based on `(arch, CUDA version, mask presence)`. Most
  consumers should just call this and let it decide.
- **Specific kernel exports** -- `sageattn_qk_int8_pv_fp8_cuda`,
  `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp16_triton`.
  Bypass the dispatcher if you want to pick the kernel yourself.
- **Native CUDA mask support on sm89 fp8++** (v0.5.5). In-pipeline
  observation under "What we've measured" is preliminary; the
  kernel-correctness piece is not.
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
  per-call). Available for users who specifically need fp8-native
  fused MLP on sm89; no other library provides this combination.
  See "What we've measured" for the production breakdown.
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
the FML2V multi-guide workflow (in-pipeline A/B 2026-05-15, stage-2
attn1 is ~29% of total render): the per-kernel ratio above gives a
much larger e2e lever than the FFN-side primitive (~15% e2e ceiling
for a hypothetical 2x stage-2 attention vs ~6% for FFN). See the
canonical workload profile in `docs/ltx_workload_profile.md` for
the full breakdown.

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
  invariant, not a perceptual one.

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
