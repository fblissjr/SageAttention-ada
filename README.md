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

One-line packaging-regression fix for Ada (RTX 40xx) source builds and a small local test/repro surface. See [`CHANGELOG.md`](./CHANGELOG.md) for the divergence list plus Known kernel bugs (CUDA mask path missing across all sage variants, including sage 3 Blackwell).

## Consumers

- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — ships `PathchSageAttentionKJ` and `LTX2MemoryEfficientSageAttentionPatch`, the most common path most ComfyUI users have to sage. The `auto` setting on `PathchSageAttentionKJ` calls sage's top-level dispatcher, which routes masked calls to the Triton kernel internally and so dodges the mask-path gap documented in `CHANGELOG.md`.
