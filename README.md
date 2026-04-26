last updated: 2026-04-26

# SageAttention (sm89 fork)

A personal fork of [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention)
(itself a fork of [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention))
that exists to do two things:

1. Make sage build cleanly from source on RTX 40xx / Ada (sm89), which
   woct0rdho's `setup.py` refactor accidentally broke.
2. Serve as a measurement surface for sm89 attention — a place to
   characterize accuracy and speed on real model shapes (LTX-2.3,
   Flux-class, Z-Image-Turbo) before any kernel-side decision lands.

The kernels themselves are upstream's. We measure on sm89 / RTX 4090
only; other archs compile and run but are not validated here.

For the high-level scope statement (what this fork is, what it
isn't, the single metric, what we might be wrong about), see
[`VISION.md`](./VISION.md). For the operational philosophy and
day-to-day perf-research framework, see
[`CLAUDE.md`](./CLAUDE.md) "Performance research: the load-bearing
metric."

## Build

```bash
source /path/to/your/venv/bin/activate
./build.sh                 # Ampere + Ada (default)
./build.sh full            # adds Hopper + Blackwell
./build.sh clean           # wipe prior artifacts first
./build.sh verify          # import-check, no rebuild
```

`./build.sh` requires `VIRTUAL_ENV` to be set so the install lands in
the right venv. It pins `uv pip install --python ${VIRTUAL_ENV}/bin/python`
and caps `MAX_JOBS` at 8. See [`CLAUDE.md`](./CLAUDE.md) for the rest of
the build details.

## What this fork changes vs upstream

Five concrete additions/changes. Everything else in the tree is
unmodified upstream code.

### 1. `setup.py:152` — sm89 added to the SM80 build gate

One-line tuple change: `("8.0", "8.6", "8.7")` → `("8.0", "8.6", "8.7", "8.9")`.

**Why it matters.** `thu-ml`'s `setup.py` builds the SM80 extension on
Ampere + Ada + Hopper + Blackwell. `woct0rdho`'s refactor narrowed the
gate to a three-tuple that silently dropped Ada (and Hopper, and
Blackwell). On an Ada-only source build, the SM80 extension was never
compiled, which removed `sageattn_qk_int8_pv_fp16_cuda` (the FP16 CUDA
fallback). Adding `"8.9"` restores it for our target box. Prebuilt
wheels include every `.so` and aren't affected.

**Scope.** Only matters if you build from source on Ada. If you run
this fork on Hopper or Blackwell and want the FP16 fallback, widen the
tuple further.

### 2. `sageattention/core.py::sageattn_warmup(shapes, kernels=...)`

Public API. Fires a one-shot dispatch per `(kernel, shape)` to prime
Triton's JIT + autotune cache.

**Why it matters.** Triton's first call on a new shape pays a JIT-compile
cost (~100–500ms per shape tuple) plus autotune time. Subsequent calls
hit Triton's on-disk cache. Calling `sageattn_warmup()` once at
model-patch time hides that latency from the first user-visible
generation. Defaults to the Triton kernel only — CUDA kernels are
fully compiled at build time and don't benefit.

**Measured cost reduction.** ~1s → ~2ms first-call latency on sm89 for
the warmed shapes (one shape, one warmup call).

**Scope.** Inert if the consumer doesn't call it. Triton's disk cache
is invalidated by `./build.sh`, so re-warm after every rebuild.

### 3. `sageattention.get_last_dispatched_kernel() -> str | None`

Public helper. After any `sageattn*()` call, returns a stable short
string naming the kernel that ran on the current thread (e.g.
`fp16_triton`, `fp8_cuda++`). The full set of names is exported as
`KNOWN_KERNEL_NAMES` and the type alias `KernelName`.

**Why it matters.** Lets a downstream tracer record what sage actually
dispatched to, instead of mirroring sage's routing table or treating
the kernel as opaque. The dispatch decision depends on `(arch, CUDA
version, requested kwargs, mask presence, shape constraints)` — too
many inputs for the consumer to re-derive reliably.

**Scope.** Backed by `threading.local()`. Read it immediately after
the sage call. If the thread yields (asyncio await, or another sage
call from the same thread) between call and read, the value can be
overwritten. Not contextvar-aware.

**Test.** [`tests/test_dispatched_kernel_telemetry.py`](./tests/test_dispatched_kernel_telemetry.py)
verifies exports, initial-`None` state, dispatcher routing on sm89,
per-variant kernel names, and thread-local isolation.

### 4. `sageattention/triton/attn_qk_int8_per_block.py` — `@triton.autotune`

Added an autotune sweep over `num_warps in {4, 8}` and
`num_stages in {3, 4, 5}`, keyed on runtime shape. `BLOCK_M`/`BLOCK_N`
stay hardcoded because they're locked by the per-block INT8
quantization step in `sageattention/quant.py`.

**Why it matters — and what it doesn't change today.** Measured on
RTX 4090 / LTX shapes: autotune confirmed the existing hardcoded
config (`num_warps=4`, `num_stages=3` for `head_dim=64`) was already
at the optimum. **Zero immediate perf delta.** The structural value
is forward compatibility — if a future kernel rewrite, Triton
release, or new shape category shifts the optimum, the autotune
harness adapts without a manual config edit.

**Scope.** No-op on current sm89 + LTX shapes. Don't claim a speedup
from this; it's insurance, not optimization.

### 5. `build.sh` — editable-install wrapper

Wraps `uv pip install --python ${VIRTUAL_ENV}/bin/python -e . --no-deps --no-build-isolation`
with `VIRTUAL_ENV` enforcement, `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9`
default, and `MAX_JOBS=8` cap. The cap exists because nvcc peaks at
several GB per parallel job on the `_qattn_sm89` kernel and uncapped
parallelism OOMs an 8-core box.

**Scope.** Pure ergonomics; no kernel-side effect.

## What we measured

All numbers below are first-measurement points on **RTX 4090 / sm89 /
CUDA 13.0 / torch 2.11.0+cu130 / triton 3.6 / bf16**, captured by
[`tests/test_sageattn_ltx_shapes.py`](./tests/test_sageattn_ltx_shapes.py)
and [`tests/test_sageattn_image_shapes.py`](./tests/test_sageattn_image_shapes.py).
Reference for accuracy is `SDPBackend.EFFICIENT_ATTENTION` (the MATH
backend OOMs at LTX self-attn scale — ~120 GiB for the full Sq×Skv
matrix). Speed comparisons are wall-clock medians over 3 runs after
1 warmup.

### Speed: sage fp8++ vs torch SDPA on sm89

| shape                                    | sage fp8++ | torch_flash | speedup |
|------------------------------------------|-----------:|------------:|--------:|
| LTX self-attn (31776×31776, h=32, d=64)  |  19.95 ms  |   52.23 ms  |  2.62×  |
| Flux-class self-attn (4096×4096, h=24, d=128) |  0.64 ms |    1.31 ms  |  2.05×  |
| Z-Image-Turbo S3-DiT (4608×4608, h=32, d=120) |  1.32 ms |    2.23 ms  |  1.69×  |

LTX cross-attn-with-mask (kv=226), masked path: sage `fp16_triton`
0.78 ms vs `torch_cudnn` 2.20 ms (~2.8×). The CUDA paths are faster
on this shape but produce wrong output — see the mask gap below.

### Accuracy: sage vs SDPA-EFFICIENT (mean rtol)

On unmasked shapes the four sage kernels cluster in a narrow band:

| kernel              | mean rtol (LTX self-attn) |
|---------------------|--------------------------:|
| `fp16_cuda`         | ~0.04                     |
| `fp16_triton`       | ~0.04                     |
| `fp8_cuda`          | ~0.097                    |
| `fp8_cuda++` (auto) | ~0.097                    |

`fp8_cuda` and `fp8_cuda++` are numerically equivalent on the LTX
shape and on a synthetic wide-V (`v_std=5.0`) shape — tested as part
of the "should we flip the non-++ default scale_max from 448 to 2.25?"
investigation; see [`CHANGELOG.md`](./CHANGELOG.md) "Decision log."

Cross-kernel `fp8++ vs fp16_triton` mean rtol on unmasked shapes is
~0.10, equal to the quadrature combination of each kernel's
independent error vs SDPA (`sqrt(0.04² + 0.09²) ≈ 0.098`). No hidden
discontinuity from mixing the two in one forward pass.

### Mask gap: cross-attn-with-mask, kv-sweep on LTX

`tests/test_sageattn_ltx_shapes.py` sweeps cross-attn seq_kv from 32
to 1024 with a ~30-position text-padding tail. CUDA kernels show
mean rtol that scales with `1 / seq_kv` — exactly the signature of
"the mask was dropped, so the masked positions corrupt softmax
proportional to how many of them there are":

| seq_kv | sage fp8++ rtol | sage fp16_triton rtol |
|-------:|----------------:|----------------------:|
|     32 | NaN             | ~0.04                 |
|     64 | 0.94            | ~0.04                 |
|    128 | 0.66            | ~0.04                 |
|    226 | 0.44            | ~0.04                 |
|    512 | 0.20            | ~0.04                 |
|   1024 | 0.13            | ~0.04                 |

The Triton kernel handles masks correctly across the full sweep. See
[`CHANGELOG.md`](./CHANGELOG.md) "Known kernel bugs" for the root
cause and the kernel-source pointers if we ever fix it.

### torch.compile

[`tests/spike_torch_compile.py`](./tests/spike_torch_compile.py) wraps
sage's `auto` dispatch in `torch.compile` and compares output and
wall-clock against eager.

**Verdict on torch 2.11:** compile produces ~2.8% mean relative error
drift vs eager on both `mode='reduce-overhead'` and `mode='default'`,
with no measurable speedup. We keep the consumer-side
`torch.compiler.disable()` around sage. Re-run the spike after any
torch upgrade; trigger to remove the disable is "bounded rtol AND a
measurable speedup."

## Tradeoffs

**You get:**

- A 2–2.7× speedup over torch's flash backend on sm89 self-attn
  (across head_dim ∈ {64, 120, 128} on the model classes we
  validated).
- A faster cross-attn path via `sageattn_qk_int8_pv_fp16_triton` (the
  Triton kernel is the only mask-correct path; ~2.8× over
  `torch_cudnn` at LTX cross-attn shapes).
- Quantization-induced rtol of ~0.097 on unmasked shapes vs SDPA. In
  practice this is below VAE noise on image/video gen workloads we've
  tested; not measured against any task-level quality benchmark.

**You give up:**

- **Mask correctness on every CUDA path.** The `_cuda` kernels' Python
  wrappers accept `attn_mask` via `**kwargs` but never forward it —
  the C++ `MaskMode` enum only has `{kNone, kCausal}`. The top-level
  `sageattn()` dispatcher does **not** redirect masked calls to
  Triton: it dispatches purely on arch and drops `attn_mask` along
  with the rest of `**kwargs`. A consumer that needs masked attention
  must explicitly call `sageattn_qk_int8_pv_fp16_triton`. Repro:
  [`tests/repros/repro_cuda_mask_kernel.py`](./tests/repros/repro_cuda_mask_kernel.py).
- bf16/fp16 input only. No fp32 input path.
- `torch.compile` around sage. The consumer must wrap calls in
  `torch.compiler.disable()` until the spike's verdict flips.
- One platform's worth of validation. Code compiles on
  Ampere/Hopper/Blackwell; we don't bench or debug those.

## Hardware target

sm89 / RTX 40xx / Ada only. Other archs compile and run (the kernels
are upstream's), but the bench harness, the rtol baselines, and the
`scale_max` decision log are all tied to sm89.

## Where it gets used

Sage is a general PyTorch attention library. Any consumer that imports
`sageattention` or replaces `torch.nn.functional.scaled_dot_product_attention`
picks this fork up via the editable install. Public API:

- `sageattention.sageattn(q, k, v, ...)` — top-level dispatcher.
  Routes by `(arch, CUDA version)`. **Does not route by mask.** On
  sm89 + CUDA ≥ 12.8 this lands on `sageattn_qk_int8_pv_fp8_cuda`
  with `pv_accum_dtype="fp32+fp16"` (SageAttention2++).
- Per-kernel exports: `sageattn_qk_int8_pv_fp16_cuda`,
  `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`,
  `sageattn_qk_int8_pv_fp8_cuda_sm90`.
- `sageattention.sageattn_warmup(shapes, kernels=...)` — optional;
  primes Triton's JIT + autotune cache.
- `sageattention.get_last_dispatched_kernel() -> str | None` —
  optional; tells a tracer which kernel just ran.

Routing policy (which kernel for which call) is the consumer's
responsibility. Sage-fork stays primitive.

## Files in this fork worth knowing about

- [`CHANGELOG.md`](./CHANGELOG.md) — version-by-version divergence,
  Known kernel bugs, Backlog with explicit triggers to act, Decision
  log of investigations that closed without action, Recurring process
  items (quarterly upstream survey, bench env re-snapshot).
- [`CLAUDE.md`](./CLAUDE.md) — fork navigation guide and conventions.
- [`tests/test_sageattn_ltx_shapes.py`](./tests/test_sageattn_ltx_shapes.py)
  — LTX-shape accuracy + speed harness. Measures every installed sage
  kernel + three torch SDPA backends + (when installed) FlashInfer
  fp16 prefill + SpargeAttention top-k=0.5.
- [`tests/test_sageattn_image_shapes.py`](./tests/test_sageattn_image_shapes.py)
  — same harness, head_dim ∈ {120, 128} shapes (Z-Image-Turbo,
  Flux-class).
- [`tests/spike_torch_compile.py`](./tests/spike_torch_compile.py) —
  re-runnable torch.compile spike.
- [`tests/repros/`](./tests/repros/) — minimal standalone repros for
  defects we haven't fixed yet.
- [`tests/run_all.sh`](./tests/run_all.sh) — env snapshot + LTX bench
  + image bench + compile spike, in one command.
