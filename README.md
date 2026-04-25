last updated: 2026-04-25

# SageAttention

Building from source.

- Forked from: https://github.com/woct0rdho/SageAttention
- Original source: https://github.com/thu-ml/SageAttention

## Build

```bash
source /path/to/your/venv/bin/activate
./build.sh
```

`./build.sh` compiles for Ampere + Ada (sm80/86/89) by default. Other options: `./build.sh full` (adds Hopper + Blackwell), `./build.sh clean`, `./build.sh verify`. Requires `VIRTUAL_ENV` to be set so the install lands in the right venv.

## Why this fork exists

Two things:

1. **One-line packaging-regression fix** for Ada (RTX 40xx) source builds. `woct0rdho`'s setup.py refactor narrowed the SM80 build gate to `(8.0, 8.6, 8.7)` and silently dropped Ada / Hopper / Blackwell. We add `8.9` back so `_qattn_sm80` (carrying `sageattn_qk_int8_pv_fp16_cuda` -- the fp16 fallback) builds again on RTX 4090. Prebuilt wheels include every `.so`; this only matters for source builds.

2. **Experimentation and measurement surface** for sm89 attention. The fork is the place where new sage-adjacent ideas (autotune coverage, mask-kernel work, FlashInfer / SpargeAttention comparisons, torch.compile compatibility, head_dim coverage for new model classes) get measured before they ship anywhere. See [`CHANGELOG.md`](./CHANGELOG.md) for the version-by-version divergence record, Known kernel bugs (CUDA mask path missing across sage 2.x and sage 3 Blackwell), the open Backlog with explicit triggers to act, and the Decision log of investigations that closed.

## What's in here beyond the upstream

- `tests/test_sageattn_ltx_shapes.py` — LTX 2.3-shape accuracy and speed harness. Measures every installed sage kernel + three torch SDPA backends + (when installed) FlashInfer fp16 prefill + SpargeAttention top-k=0.5. Shapes cover head_dim ∈ {64 (LTX), 120 (Z-Image S3-DiT), 128 (Flux-class)}. Soft-warns on rtol drift; serves as the regression yardstick for any kernel or torch change.
- `tests/spike_torch_compile.py` — re-runnable 30-min spike measuring whether `torch.compile` around sage produces bounded mean-relative-error AND a speedup. Current verdict on torch 2.11: keep the consumer-side `torch.compiler.disable()`. Re-run after torch upgrades.
- `tests/repros/` — minimal standalone repros for defects in this fork's kernels we haven't fixed yet.
- `sageattention/core.py::sageattn_warmup(shapes, kernels=...)` — public API to prime Triton's JIT autotune cache. Cuts ~1s first-call latency on sm89 to ~2ms post-warm.
- `sageattention/triton/attn_qk_int8_per_block.py` — added `@triton.autotune` over `(num_warps, num_stages)`. Measured no-op on sm89 today (the hardcoded config was already optimal); structural value is forward-compat to future kernel / triton / shape shifts.
- `build.sh` — editable-install wrapper that targets the active `VIRTUAL_ENV`, pins `uv pip install --python ${VIRTUAL_ENV}/bin/python`, caps `MAX_JOBS` at 8.

## Hardware target

We measure and validate on **sm89 / RTX 40xx / Ada** only. The code compiles and runs on Ampere / Hopper / Blackwell (it's upstream's), but those archs don't get tested here. Other-arch users who want the fp16 source build should widen the SM80 build gate in `setup.py:152` to match thu-ml's coverage.

## Where it gets used

Sage is a general PyTorch attention library. Any consumer that imports `sageattention` or replaces `torch.nn.functional.scaled_dot_product_attention` picks this fork up via the editable install. The kernel API is `sageattention.sageattn(q, k, v, ...)`; specific kernels (`sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`, ...) are also exported.

Model classes the bench harness validates against on sm89 today: **LTX 2.3** (head_dim=64), **Z-Image-Turbo** (S3-DiT, head_dim=120), and **Flux-class** image gen (head_dim=128). Add a row to `tests/test_sageattn_image_shapes.py` (or the LTX file, depending on shape category) before relying on a new model class.

**Mask-aware caveat.** Sage's `_cuda` kernels silently drop `attn_mask`; only `sageattn_qk_int8_pv_fp16_triton` handles masks correctly. Consumers that route masked vs unmasked calls separately should send masked calls to the Triton kernel; consumers that just call `sageattn()` get this routing for free via the top-level dispatcher. See [`CHANGELOG.md`](./CHANGELOG.md) "Known kernel bugs" for the full story.
