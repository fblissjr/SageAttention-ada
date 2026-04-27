#!/usr/bin/env python3
"""Image-gen self-attn shapes for sage on sm89.

Companion to `test_sageattn_ltx_shapes.py`. Same harness, different shape
catalog. Imports the shape-sweep runner from the LTX file rather than
re-implementing the per-shape table -- keeps the two scripts identical at
the per-shape level and lets a future column / threshold change happen in
one place.

Two shapes today:

- **Flux-class** self-attn at 1024^2 output. 1024^2 / 16^2 VAE = 4096
  image tokens; head_dim=128, heads=24 are the Flux-1-dev family
  defaults. Confirms sage's speedup holds on head_dim=128 workloads,
  not just LTX's head_dim=64.
- **Z-Image-Turbo** self-attn (S3-DiT, single-stream). Architecture:
  30 layers, hidden=3840, 32 heads, head_dim=120 (3840/32). Single-
  stream means text+image tokens concatenate into one sequence;
  ~4096 image tokens + ~512 text tokens ~= 4608. head_dim=120 is
  non-power-of-2; this row would SKIP if sage's CUDA kernels can't
  handle it (verified 2026-04-25: they do).

Adding a new image-gen shape: edit `IMAGE_SHAPES` below.
"""

from __future__ import annotations

import torch

from test_sageattn_ltx_shapes import Shape, print_warnings_footer, run_shape_sweep


IMAGE_SHAPES = [
    # Flux-class self-attn at 1024^2.
    Shape("image_gen_self_attn_4096_h24_d128", 1, 24, 4096, 4096, 128, False),
    # Z-Image-Turbo (S3-DiT) self-attn. head_dim=120 is the unusual one.
    Shape("z_image_turbo_self_attn_4608_h32_d120", 1, 32, 4608, 4608, 120, False),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this test measures kernel numerics on-GPU.")
        return
    warnings, _ = run_shape_sweep(IMAGE_SHAPES)
    print_warnings_footer(warnings)


if __name__ == "__main__":
    main()
