# Downstream-known internal symbols (de-facto public surface)

Last updated: 2026-05-13

L3 reference for CLAUDE.md. Read this before removing or renaming any
underscore-prefixed symbol in `sageattention.core` or any pybind
method on `_qattn_sm*`.

## The undocumented contract

The two documented surfaces (`sageattn()` dispatcher + the named
`sageattn_qk_int8_pv_*` kernel exports) are not the whole story.
Underscore-prefixed and module-level re-exports in
`sageattention.core` are imported by name at module-load time by
downstream consumers. Removing or renaming them breaks those
consumers without warning.

This bit us in v0.5.0: dropping `_qattn_sm90` + `sm90_compile`
without considering downstream callers means KJNodes'
`LTX2MemoryEfficientSageAttentionPatch` can fail to import against
our fork even on sm89 boxes (depends on whether KJ's import is
try/except-guarded). The Hopper kernel removal was correct; the
*consideration* of the downstream blast radius was missing.

## Known importers (audit 2026-05-01)

KJNodes `ComfyUI-KJNodes/nodes/ltxv_nodes.py` (the LTX-2 per-block
patch) imports these from `sageattention.core` directly:

```
_qattn_sm80, _qattn_sm89, _qattn_sm90        # compiled .so extensions
sm80_compile, sm89_compile, sm90_compile     # Python fallback modules
per_thread_int8_triton                       # quant fn
per_warp_int8_cuda                           # quant fn
per_block_int8_triton                        # quant fn
per_channel_fp8                              # quant fn
attn_false                                   # triton attn entry point
get_cuda_arch_versions                       # arch-detection util
```

Plus a specific pybind method on `_qattn_sm89`:

```
_qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
```

The other pybind methods in `csrc/qattn/pybind_sm89.cpp:23-30` are
load-bearing for our own dispatcher and effectively in the same
risk class -- treat the full list as protected.

## Pre-removal checklist

Before removing or renaming any symbol in the list above, OR adding
to the list of compiled `_qattn_sm*` modules / their pybind
methods:

1. `grep -r "<symbol>" coderef/` -- check if any consumer in our
   coordinated set imports it. Empty result lowers risk but doesn't
   eliminate it (we don't grep all of GitHub).
2. If found: either (a) keep a compatibility shim (e.g. a stub
   module that raises `NotImplementedError` on use), (b) coordinate
   a memo to the consumer-side claude before removal, or (c) bump
   the major version and document the break in CHANGELOG /
   "Breaking changes."
3. If not found in `coderef/` but the symbol is on the list above:
   document the removal in CHANGELOG with a "downstream-known
   internal symbol removed" note. Treats the de-facto contract as
   real even when not currently exercised.

## Why we don't just promote these to the typed public API

Two reasons. First, the underscore prefix is doing real work:
these are implementation details that change shape across
upstream sageattention versions (the dual-name fallback pattern
`_qattn_sm89` / `sm89_compile as _qattn_sm89` is evidence upstream
has already restructured them once). Promoting them locks us in.
Second, the typed API (`sageattn`, `sageattn_qk_int8_pv_*`) is the
right entry point for almost every consumer; the only legitimate
reason to import the underscore symbols is to do something the
typed API doesn't expose (per-block patching with custom RoPE
fusion, KJ's case). Documenting the de-facto contract is the
balance: we don't promise stability, but we promise not to break
it without consideration.
