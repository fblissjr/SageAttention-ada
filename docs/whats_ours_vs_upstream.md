# What's ours vs what's upstream

Last updated: 2026-05-13

L3 reference for CLAUDE.md. Load this when editing a file and you
need to know whether the file is unmodified upstream code (lighter
touch), a v0.5.0 deletion target (don't reintroduce), or one of our
additions (we own the contract).

## Upstream-from-woct0rdho code (unmodified unless noted)

- `csrc/qattn/{pybind_sm80.cpp, pybind_sm89.cpp, qk_int_sv_f16_cuda_sm80.cu,
  sm89_qk_int8_sv_f8_*.cu}`, `csrc/fused/`, `pyproject.toml`,
  `tests/test_sageattn.py`, `tests/test_flashattn{2,3}.py`.
- `sageattention/` mostly unmodified except
  `sageattention/triton/attn_qk_int8_per_block.py` (we added autotune).
- `setup.py` mostly unmodified except line 152 (sm89 -> SM80 build gate)
  + v0.5.0 trims (Hopper SM90 block, CUDA-12.3-for-9.0 check, Windows
  MSVC compile flags).

## Removed in v0.5.0 (we own the fork; not building or running these)

- `sageattention3_blackwell/` -- sage 3 Blackwell subpackage (FP4).
- `csrc/qattn/{attn_cuda_sm90.h, pybind_sm90.cpp,
  qk_int_sv_f8_cuda_sm90.cu}` -- Hopper kernel.
- `sageattention/sm90_compile.py` + `core.py::sageattn_qk_int8_pv_fp8_cuda_sm90`
  function + dispatcher branch + KERNEL_FP8_CUDA_SM90 constant.
- `bench/` -- 9 one-shape upstream benchmarks superseded by our LTX +
  image + e2e + workload-profile bench files.

## Our additions and modifications (tracked in CHANGELOG.md)

- `setup.py:152` -- one-line tuple change so `_qattn_sm80` builds on
  sm89 boxes (was gated on 8.0/8.6/8.7 only; Ada is forward-compat to
  SM80).
- `sageattention/core.py::sageattn_warmup(shapes, kernels=...)` --
  public API that fires one-shot dispatches per (kernel, shape) to
  prime Triton's JIT + autotune cache. Defaults to the Triton kernel
  only (CUDA kernels are build-time compiled, no warmup benefit).
  **Status (verified 2026-04-26):** available API; no consumers in
  our coordinated set currently call it. The mechanism (Triton
  autotune cache hit on subsequent calls) is real; the "~1s -> ~2ms"
  perf claim circulating in earlier docs was the documented
  mechanism, not a measured number from our box. Treat as an opt-in
  optimization we offer; remove from this list if no consumer adopts
  it within ~6 months.
- `sageattention.get_last_dispatched_kernel() -> Optional[KernelName]`
  -- public helper exposing which kernel the most recent `sageattn*`
  call on this thread dispatched to, as a stable short string
  (`fp16_triton`, `fp8_cuda++`, etc.; full set in
  `KNOWN_KERNEL_NAMES`, type alias in `KernelName`). Lets consumer
  tracers record sage's routing decision instead of mirroring the
  dispatch table or treating it as opaque. Backed by a
  `threading.local()` set inside each entry point's dispatch branch.
  Read immediately after the sage call -- thread-local, not
  contextvar-aware. Test:
  `tests/test_dispatched_kernel_telemetry.py`. **Adding a new kernel
  variant requires three coupled edits in `core.py`:** a new
  `KERNEL_*` string constant, the matching entry in the
  `KNOWN_KERNEL_NAMES` frozenset, and the matching string in the
  `KernelName = Literal[...]` alias. Forgetting the Literal silently
  breaks consumer type-checking but not runtime; forgetting the
  frozenset silently breaks consumer `assert kernel in
  KNOWN_KERNEL_NAMES` validators. The constant + set + Literal trio
  is the public contract.
- `sageattention.fused_rope_split(q, k, freqs_cis, *, use_triton=True)
  -> tuple[Tensor, Tensor]` -- v0.5.3 fused split-RoPE primitive.
  Clean-room Triton kernel matching LTX's `apply_split_rotary_emb`;
  falls back to torch reference on non-CUDA / non-split-pe / shape
  mismatch / `use_triton=False`. Lives in
  `sageattention/triton/fused_rope.py`. v1 supports the LTX split-pe
  convention only; interleaved variants silently fall back.
  **Status (verified 2026-05-01):** consumer measured RoPE at 0.55%
  of GPU time on the iclora workflow, so immediate ROI is ~zero --
  candidate for removal at the next deletion arc if no consumer
  adopts within ~6 months (same disposition as `sageattn_warmup`).
  Test: `tests/test_fused_rope.py` (3 CPU + 7 GPU + export-check).
- `sageattention/triton/attn_qk_int8_per_block.py` -- `@triton.autotune`
  over `num_warps` and `num_stages`. Zero immediate perf delta on
  sm89 + LTX shapes (hardcoded config was already optimal) but
  forward-compatible: catches future kernel/triton/shape shifts.
- `sageattention/core.py::sageattn` -- v0.3.0 dispatcher mask-routing
  fix. See CLAUDE.md "The consumer surface" section and CHANGELOG
  v0.3.0 for the mechanism.
- `sageattention/core.py::_warn_if_mask_passed_to_cuda_kernel` --
  v0.3.1 soft-warn helper. Hooked into the two CUDA entry-point
  wrappers (`sageattn_qk_int8_pv_fp16_cuda`,
  `sageattn_qk_int8_pv_fp8_cuda`) right after the assert block.
  Catches consumers who bypass the dispatcher and hand-pick a `_cuda`
  kernel with a non-None mask. (v0.5.0 dropped the third wrapper,
  `sageattn_qk_int8_pv_fp8_cuda_sm90`, along with the rest of the
  Hopper plumbing.)
  Soft (warns, not raises) so `attn_mask=None` defensive callers
  aren't penalized.
- `build.sh` -- editable-install wrapper with VIRTUAL_ENV check,
  `--python` pin, MAX_JOBS cap.
- `tests/test_sageattn_ltx_shapes.py` -- LTX-parametrized accuracy +
  perf measurement across sage kernels AND torch SDPA backends
  (FLASH / EFFICIENT / CUDNN). Doubles as a regression guard for
  "did a torch upgrade close the sage perf gap?"
- `tests/test_dispatched_kernel_telemetry.py` -- standalone-script
  test for the `get_last_dispatched_kernel()` helper. 11 tests
  covering: exports, initial-None state, dispatcher-routes-to-fp8++
  on sm89 (unmasked), dispatcher-routes-masked-calls-to-triton
  (v0.3.0 invariant), `pv_accum_dtype` override honored
  (v0.3.1 regression test), per-variant kernel name strings,
  hand-picked-_cuda-warns-on-mask (v0.3.1 soft-warn),
  thread-local isolation. Pure Python; no rebuild needed after
  edits to `core.py` or `__init__.py`.
- `tests/test_sageattn_ltx_shapes.py::cross_attn_unmasked_kv226_kratio_probe`
  -- v0.3.1 K-ratio probe row. Same shape as `cross_attn_text_kv226`
  but unmasked, so `K = triton_masked_ms / fp8++_unmasked_ms` is
  readable from two bench rows. Gates the deferred native CUDA
  mask kernel work (Backlog).
- `tests/bench_e2e_ltx.py` -- v0.4.0 end-to-end gen-time bench.
  Submits a workflow via the ComfyUI HTTP API, runs N times sage-on
  vs sage-disabled, reports wall-time speedup +
  attention-fraction-of-step. Closes the framework's "kernel ms is
  not gen ms" gap. Host resolved from CLI / `$COMFYUI_HOST` /
  `internal/local_config.json` (no hardcoded default). See
  `internal/runbook_bench_e2e_ltx.md` for the operational runbook.
- `tests/repros/repro_cuda_mask_kernel.py` -- minimal repro for the
  CUDA mask-path missing-feature documented in CHANGELOG.
- `CHANGELOG.md` -- versioned divergence + Known kernel bugs + Backlog + Decision log + Recurring process items.
- `README.md` -- minimal; attribution only.
- `CLAUDE.md` -- L1 routing index for this repo's claude.

## Git history

Git history was squashed to a single "Fork baseline" commit with our
changes layered on top. All safety-backup branches have been deleted;
the squashed history is the canonical state. `origin/main` still
carries the pre-squash 196-commit upstream history; the next push
will need `git push --force-with-lease origin main`.
