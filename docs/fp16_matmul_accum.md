# fp16 matmul accumulation flag

Last updated: 2026-05-13

L3 reference for CLAUDE.md. Load this when someone asks whether
`torch.backends.cuda.matmul.allow_fp16_accumulation` (or KJ's
`enable_fp16_accumulation` exposure of it) affects sage output.

`torch.backends.cuda.matmul.allow_fp16_accumulation` (available in
torch 2.7.1+; exposed by KJ's `CheckpointLoaderKJ` as
`enable_fp16_accumulation`): does NOT affect sage's internals.

Verified 2026-04-24:

- Sage's Q @ K^T and P @ V are done inside its own int8/fp8
  CUDA/Triton kernels via tensor cores. No cuBLAS on any path -- grep
  of `csrc/` finds zero `cublas` references.
- Sage does call `torch.matmul` exactly once, in
  `core.py::lse_correction` (when `smooth_k=True` AND
  `return_lse=True`). That path isn't taken by the ComfyUI
  `optimized_attention_override` hook, which never asks for LSE.
- Net effect on attention: zero.

What the flag DOES affect: torch's own matmuls for the Q/K/V/output
linear projections around attention (those go through cuBLAS). For
LTX-2.3 that's ~5-10% of total gen time in the linear layers. Safe
to enable for speedup; impact on sage attention output is nil.
