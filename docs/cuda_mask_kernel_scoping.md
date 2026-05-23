Last updated: 2026-05-16 (moved from internal/design/ to docs/ for public sharing; trigger-event paragraph genericized)

# Scoping note: native general-mask support in the sm89 fp8++ CUDA kernel

Scope: a side-by-side read of the Triton mask reference and the
load-bearing sm89 CUDA kernel, identifying exactly where mask support
would slot in. **Not a commitment to do the work.** The backlog
triggers (CHANGELOG / Backlog / "Add mask support to the sm80/sm89 CUDA
kernels") have not fired; this note is preparation so we can move fast
if/when they do.

## The reference: Triton's mask handling

`sageattention/triton/attn_qk_int8_per_block.py:31-51` is the spec.
Two code paths keyed on mask dtype:

**Bool mask path** (mask_ptrs.dtype.element_ty == tl.int1):
```python
mask_block = tl.load(mask_ptrs + start_n * stride_maskn,
                     mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n),
                     other=False)
if tl.max(mask_block) == 0:
    skip = True   # whole BLOCK_N is masked-out -> skip this K iteration entirely
...
qk = qk + tl.where(mask_block, 0, -1.0e6)
```

**Float-dtype mask path** (matching q.dtype):
```python
mask_block = tl.load(mask_ptrs + start_n * stride_maskn,
                     mask=...,
                     other=-1.0e6)
...
qk = qk + mask_block   # additive log-weight semantics
```

Both apply *before* the per-block max reduction (`m_ij = tl.maximum(m_i, tl.max(qk, 1))`).
The mask layout is `(B, H, M, N)` with explicit strides per dim;
broadcast over H or M is via stride-0 (the dispatcher's `attn_mask.expand(target_shape)`
in `core.py:441` makes broadcasted dims look like full-rank tensors with
zero stride in the broadcast dims).

Two correctness behaviors worth preserving:
1. **Whole-block skip**: when a BLOCK_N's bool mask is all-False, the
   kernel skips that K-block's compute entirely. This is a real perf
   feature, not just correctness.
2. **Out-of-bounds via the same `other=False/-1.0e6`**: the mask load's
   own predicate handles the `offs_m < qo_len` / `offs_n < kv_len -
   start_n` boundary; doesn't need a separate `apply_out_of_bound_mask`
   over the mask path. The CUDA kernel currently calls a separate
   `apply_out_of_bound_mask` for the kv-len tail (`qk_int_sv_f8_cuda_sm89.cuh:517`).

## The target: sm89 fp8++ CUDA kernel

The load-bearing variant is `qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`
(`csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu`
calling into `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`). That's what
`sageattention.core.py:1063` dispatches to for sm89 + CUDA >= 12.8 + bf16
inputs, the SageAttention2++ path.

**Where the mask would slot in (kernel body):**

`qk_int_sv_f8_cuda_sm89.cuh` has two mask-application points in the
inner loop:

1. Line 406: `if constexpr (mask_mode == MaskMode::kCausal) apply_causal_mask(...)` — in the steady-state K-iteration block.
2. Line 513: same `if constexpr` — in the "last iter" K-iteration block (which also calls `apply_out_of_bound_mask` for the kv-len tail at line 517).

Both points operate on `RS_f32[num_tiles_q][num_tiles_k][8]` — the
per-warp register-file holding the dequantized QK scores. Mask
application happens *after* dequantization, *before* the softmax
`update_mdo`. Same logical position as in the Triton kernel.

**The pattern to add:** a new `MaskMode::kGeneral` branch alongside
`kCausal`, calling an `apply_general_mask<num_tiles_q, num_tiles_k>(...)`
helper that loads mask values from a `DTypeMask*` pointer and applies
them. Mirror `apply_causal_mask` at `attn_utils.cuh:296-323` and
`apply_out_of_bound_mask` at 326-351 for the loop shape and index math
(those helpers already know how to map `(fq, fk, k)` register positions
to `(q_idx, kv_idx)` global positions).

**Mask-load address arithmetic** is the hard part. `apply_causal_mask`
only needs the abstract `q_idx, kv_idx` (`q_idx >= kv_idx`).
`apply_general_mask` needs to read a value from the mask tensor at
`(batch, head, q_idx, kv_idx)`. The kernel doesn't currently have a
mask base-pointer or mask strides in scope; the C++ entry would need
to pass them in (analogous to how Triton receives
`stride_maskz, stride_maskh, stride_maskm, stride_maskn`).

## Pybind + dispatch surface

The current C++ entry signature is at
`sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu:3-14` and
takes `is_causal` as an `int` flag. To add mask support:

1. **C++ signature change**: add `torch::Tensor attn_mask` (optional --
   pass an empty tensor for "no mask") and a `mask_mode` int flag
   (or pass the dtype as a separate enum). Keep `is_causal` for backward
   compat; the dispatcher would choose between `kCausal` and `kGeneral`
   based on input.
2. **MaskMode enum** (`csrc/qattn/attn_utils.cuh:36-39`): add `kGeneral = 2`.
3. **Kernel template parameter**: `mask_mode` is already a template
   parameter (`qk_int_sv_f8_cuda_sm89.cuh:45`), so dispatching to the
   right specialization is the same DISPATCH_BOOL pattern that already
   exists.
4. **Python entry** (`sageattention/core.py:747+`): `sageattn_qk_int8_pv_fp8_cuda`
   currently accepts `attn_mask` via `**kwargs` and ignores it (with a
   v0.3.1 soft-warn). Pull it out of kwargs, pass to the C++ entry.
5. **Dispatcher** (`sageattn` in `core.py:163`): the v0.3.0 mask-routing
   fix that sends masked calls to Triton would need a second look. With
   a mask-correct CUDA kernel, masked calls on sm89 + CUDA >= 12.8 could
   route to the CUDA path. The Triton fallback stays for archs without
   the CUDA-mask kernel (sm80, future arch fallbacks).

## Register-pressure risk read

The kernel is at `qk_int_sv_f8_cuda_sm89.cuh:45` with this template:

```cpp
template <..., MaskMode mask_mode = MaskMode::kNone, ...>
```

`mask_mode` is `constexpr`, and the existing causal-mask branches are
`if constexpr (mask_mode == MaskMode::kCausal)`. Adding a `kGeneral`
branch under the same `if constexpr` pattern means the unmasked
specialization compiles to bit-identical PTX (zero added registers,
zero added instructions). Verified by reading lines 406-410 and
513-517 -- both branches disappear in the `kNone` specialization.

**Risk verdict**: the unmasked path can remain bit-identical *if* the
implementation discipline holds (no new variables declared outside
the `if constexpr`). Easy to verify post-implementation: compile both
specializations to PTX and diff. If diff is non-empty for `kNone`,
the discipline failed.

The pressure risk is on the `kGeneral` path itself: the mask-load
needs additional registers for the mask base-pointer, strides, and
intermediate values. Order-of-magnitude estimate from the
`apply_causal_mask` shape: ~4-6 additional registers per thread on
the kGeneral path. Likely won't drop occupancy on sm89 (it's not
near the edge today per the autotune sweep landing at num_warps=4,
num_stages=3-5 -- if it were occupancy-bound, num_warps=8 would have
won), but worth measuring.

## What we'd NOT build (scope discipline)

Per CLAUDE.md the fork only cares about sm89 / RTX 40xx / Ada. So:

- Only the sm89 fp8++ variant (`qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf`).
  The other 6 sm89 variants (`accum_f32_*`, `accum_f16_attn_inst_buf`,
  the no-`inst_buf` versions) can wait until a consumer hits them.
- Not the sm80 fp16 kernel (`qk_int_sv_f16_cuda_sm80.cu`). Same pattern
  applies but the dispatcher hits it less often; defer.
- Not the upstream Hopper/Blackwell kernels (removed in v0.5.0).
- Not the fp8 sm89 with `accum_f32` (the non-`++` variant). Dispatcher
  doesn't pick it on sm89 + CUDA >= 12.8.

This keeps the work bounded to one kernel + one helper + one pybind
entry + one Python dispatch branch.

## Effort estimate (refined)

Original CHANGELOG: "days of kernel work, not hours."

Refined breakdown:
- `apply_general_mask` helper (mirror existing causal/oob helpers):
  ~2-4 hr including the index-math debugging.
- `MaskMode::kGeneral` enum value + template specialization plumbing
  in `qk_int_sv_f8_cuda_sm89.cuh`: ~1-2 hr.
- C++ entry signature change + pybind: ~1 hr.
- Python dispatch glue (`sageattn_qk_int8_pv_fp8_cuda` extraction
  from kwargs, dispatcher routing on masked sm89 calls): ~1-2 hr.
- Build/iterate cycle (rebuild takes 60-90s each, expect 5-10 cycles
  during debug): ~2-3 hr.
- Correctness verification: red TDD test against Triton reference at
  multiple shapes including the `repro_cuda_mask_kernel.py` shape,
  the LTX cross-attn shape with `(1,1,T,T)` boolean mask, and the
  broadcast `(1,1,1,T)` shape from the upstream ComfyUI core PR
  that fired the trigger: ~2-3 hr to write + run.
- Perf regression verification on unmasked path (LTX bench
  `--check-regression` against the v0.5.4 baseline; PTX diff of the
  kNone specialization vs HEAD): ~1-2 hr.

**Total: ~10-17 hr of focused work. Call it 2 days with
context-switching overhead.** Lower bound assumes everything goes
smoothly; upper bound includes one round of "the mask-load address
arithmetic is subtly wrong and produces an off-by-one for non-
power-of-2 q_idx alignments."

Confidence in the estimate: medium-low. The unknowns are (a) whether
the `RS_f32` register file has room for the mask-load intermediates
without spilling, and (b) whether the index math for `apply_general_mask`
matches the broadcast semantics the dispatcher already implements
in `attn_mask.expand(target_shape)`. Both are answerable with a
small spike (~half day, see below) if/when we commit.

## Suggested first move when the trigger fires

A focused half-day spike:

1. Write `apply_general_mask` helper in a scratch file mirroring
   `apply_causal_mask`. Index math only -- no actual mask loading yet.
2. Add `MaskMode::kGeneral` to the enum.
3. Add the `if constexpr (mask_mode == MaskMode::kGeneral)` branches
   at lines 406 and 513, calling the new helper.
4. Compile the `kNone` specialization to PTX (via `nvcc -ptx`) and
   diff against HEAD. If the diff is empty, the discipline held.
   If not, the implementation needs restructuring before going further.
5. Compile the `kGeneral` specialization and inspect register count.
   If above ~96 (the occupancy threshold for the current num_warps=4
   config), need to revisit.

That spike answers the register-pressure question definitively in
~half a day, *before* committing to the full implementation. If the
spike fails, we have falsifiable evidence to update the backlog
estimate; if it succeeds, the remaining work is mostly
mechanical.

## Trigger status

Per the updated v0.5.4 backlog formulation (CHANGELOG / Backlog /
"Add mask support to the sm80/sm89 CUDA kernels"), three dimensions
can fire the trigger:

1. *Perf*: K = `triton_masked_ms / fp8++_unmasked_ms` crosses 5x at
   a production-hit shape. Last K = 1.57 (2026-04-27). Re-check
   after each kernel optimization that lands on the unmasked path.
   Today: not fired.
2. *Wall-time*: masked-triton > 5% of total gen wall-time on a real
   consumer trace. Today: not measured at that level.
3. *Structural-correctness consumer signal*: 2+ independent downstream
   consumers raise it, OR 1 consumer with high-leverage rationale.

**Trigger #3 fired 2026-05-13.** An upstream ComfyUI core PR landed
that adds LTX 2.3 self-attn guide-mask support; the relevant code
path forces an SDPA fallback specifically because the sage CUDA
path silently drops `attn_mask`. An upstream-core PR is high-leverage
rationale on its own: it represents the ComfyUI consumer surface for
that workflow class (every LTX 2.3 gen with `guide_strength<1.0`),
not a single individual user, and the upstream surface has surfaced
the structural-correctness concern explicitly. Committing to the
work; the half-day spike above is the next move.

## Open questions

- Does the existing `qk_int_sv_f8_cuda_sm89.cuh` template specialization
  pattern (`if constexpr`) actually compile to bit-identical kNone PTX?
  Asserted above; needs PTX diff to verify.
- The Triton kernel's whole-block-skip optimization
  (`tl.max(mask_block) == 0` -> `skip=True`) is a real perf win on
  sparse masks. Does the CUDA kernel's `if constexpr` structure allow
  the same early-exit, or would it serialize the K-iteration loop?
  Likely answerable by reading the loop structure more carefully;
  marked for the spike.
- The mask-broadcast semantics (Triton uses `attn_mask.expand(target_shape)`
  to fake stride-0 broadcasts; CUDA would need to handle stride-0 dims
  in the address arithmetic). Need to confirm CUDA mask-load can use
  zero strides without UB.
