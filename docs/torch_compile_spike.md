# torch.compile compatibility status

Last updated: 2026-05-13

L3 reference for CLAUDE.md. Load this when someone proposes wrapping
sage in `torch.compile`, or when a torch upgrade just landed and the
spike needs re-running.

## Current verdict

Not used. The downstream consumer node wraps sage in
`torch.compiler.disable()`. Keep the disable until the trigger below
fires.

## Empirical status (verified 2026-05-01 on RTX 4090 / torch 2.11.0+cu130 / sage `main`)

The spike (`tests/spike_torch_compile.py`) rejects on **rtol drift**,
not perf. Both `reduce-overhead` and `default` modes trace + run, but
produce mean relative error 0.02759 vs eager (rejection threshold
0.01). Same drift on both modes (5 sig figs) indicates deterministic
precision loss from partial-graph reordering across pybind
boundaries, not a stochastic compile artifact.

## The two pybind kernels Dynamo graph-breaks at

- `_fused.transpose_pad_permute_cuda` (csrc/fused/pybind.cpp:30)
- `_fused.scale_fuse_quant_cuda` (csrc/fused/pybind.cpp:31)

Both are called from `sageattention/quant.py:281,289,292` in
`per_channel_fp8` (the V-quant path on every fp8 sage call --
load-bearing dispatch on sm89). Same risk class extends to
`mean_scale_fuse_quant_cuda` (csrc/fused/pybind.cpp:32, smooth_v=True
branch).

## Trigger to do the work

If/when consumer-side path 1 (CUDA graphs on the LTX denoiser) fails
AND consumer wants path 2 (torch.compile the denoiser), register the
three named kernels as `torch.library.custom_op` with proper
meta/abstract registrations so Dynamo can trace through them without
graph breaks. Estimate ~1-2 days per kernel (~3-6 days total). Until
that trigger fires, keep the disable.

Re-run the spike after every torch upgrade; reopen if a mode produces
bounded-rtol speedup > 1.05x.
