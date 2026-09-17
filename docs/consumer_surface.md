# The consumer surface

Last updated: 2026-09-17

What this fork exposes to downstream callers, in full. Extracted
verbatim from `CLAUDE.md` on 2026-09-08.

**Two contracts here break silently rather than loudly, and they are the
reason this file exists:** `attn_mask` must stay a *named* parameter of
`sageattn()` because a consumer gates on `inspect.signature`, and
`build_info()`'s key set is embedded verbatim in a consumer's dated
records. Both are pinned by tests -- see `docs/downstream_symbols.md`
for the pre-removal checklist that applies to everything here.

## The consumer surface

Sage exposes three surfaces to downstream consumers:

1. **`sageattn()` top-level dispatcher.** Picks a kernel based on
   `(detected arch, CUDA version, mask presence)`. On sm89 + CUDA >=
   12.8 unmasked: lands on `sageattn_qk_int8_pv_fp8_cuda` with
   `pv_accum_dtype="fp32+fp16"`. With `attn_mask` passed: routes to
   the same `sageattn_qk_int8_pv_fp8_cuda` (the v0.5.5 native CUDA
   mask path); other archs still route to `sageattn_qk_int8_pv_fp16_triton`
   since their CUDA kernels haven't gained mask support yet.
   Implementation: `sageattention/core.py::sageattn` pulls `attn_mask`
   out of `**kwargs` before the arch branch and bifurcates on
   `(arch, cuda_version, mask_present)`. The routing invariant is
   enforced by a test in `tests/test_dispatched_kernel_telemetry.py`.
   **Most consumers should just call this and let dispatch decide.**
2. **Specific kernel exports** -- `sageattn_qk_int8_pv_fp16_cuda`,
   `sageattn_qk_int8_pv_fp8_cuda`, `sageattn_qk_int8_pv_fp16_triton`,
   etc. Bypass the dispatcher; caller picks. **Masked attention works
   on sm89 fp8++ (`pv_accum_dtype="fp32+fp16"`) as of v0.5.5** and on
   the Triton kernel; other CUDA variants (sm80 fp16, sm89 non-fp8++)
   still silently drop the mask and warn. If a consumer hand-picks a
   non-mask-correct CUDA kernel + mask, the v0.3.1 soft-warn fires;
   the dispatcher is the safe default.
3. **`sage_ffn(x, w1, s1, w2, s2, b1=None, b2=None)`** (v0.6) -- a
   separate FFN primitive, not an attention kernel. Two-kernel
   Triton fp8 MLP (`Linear(fp8) -> GELU(tanh) -> Linear(fp8)`)
   targeting LTX 2.3-class FFN blocks (hidden=4096, inner=16384,
   per-tensor fp8 E4M3FN weights, optional bf16 biases on both
   Linear layers). A synthetic bench shows it faster than torch's
   fp8-dequant reference; **an in-pipeline A/B on a two-sampler LTX
   workflow came back slower end to end, and worse per call at
   stage-2** (CHANGELOG v0.6.0), so this ships as a completeness
   primitive, not a perf win. Root
   cause is L2 cache contention with neighboring attention modules
   + cumulative kernel-launch overhead at LTX's ~1000-FFN-calls/render
   count. Not wired into `sageattn()`; consumer imports it directly
   from the top-level package. The qualitative wedge holds (no other
   library ships fp8-native fused MLP for ComfyUI consumer-app on
   sm89); the quantitative wedge does not on the tested workload.
   v0.6.1 candidates to close the gap: persistent-CTA hybrid and
   CUTLASS-based CUDA backend (see CHANGELOG Backlog).
4. **`sageattn_consume(qkv, ...)`** (v0.7) -- `sageattn()` that takes
   ownership of q/k/v so the float tensors are released once quantized
   instead of at end-of-call. `sageattn()` cannot do this: the caller's
   frame owns the refs. Takes a `[q, k, v]` list, which it empties, or
   three single-owner containers exposing `peek()`/`take()` -- the
   protocol ComfyUI added in `bf4c9a08` and wraps H3's q/k/v in on every
   call (v0.7.5). Taken here rather than by the caller deliberately:
   unwrapping in the caller's frame re-binds all three for the duration
   of the call, which is the retention this entry point exists to avoid.
   Same signature otherwise; output bit-identical. **What it saves is
   configuration-dependent, and in the arrangement DiT blocks actually
   use it currently saves nothing** -- measured at fl2va, peak per call.
   Separate allocations save substantially at `smooth_k=False` and much
   less at the shipped `smooth_k=True`, because `per_thread_int8`
   allocates the int8 outputs before evaluating `k = k - km`, so a full
   bf16 K copy lands on top. Fused QKV views save **nothing either
   way**: releasing q and k frees nothing while v holds the same
   allocation, and by the time v goes, `per_channel_fp8`'s bf16
   transpose buffer has set a higher peak. Figures and the retraction of
   an earlier wrong number for the fused case are in CHANGELOG v0.7.3. Making the fused case pay
   *from in here* needs the transpose buffer dropped **and** the
   mean-subtraction done in place -- either alone leaves the other
   setting the floor. Only the sm89 fp8 path releases early; other
   kernels fall back to the ordinary path, correct but with no saving.
6. **`qk_balance` keyword** (v0.7.19, 2026-09-15) on
   `sageattn_qk_int8_pv_fp8_cuda`, and through `sageattn()` /
   `sageattn_consume()` kwargs: rebalances K's channels against Q's inside
   the per-thread INT8 quantizer, exact for the attention math, gated per
   head so flat heads are untouched. Off by default. `balance_alpha`
   (0.5) and `balance_min_share` (0.2) tune it. Per-thread quantization
   only; ignored under `qk_quant_gran="per_warp"`. Why and what it buys:
   CHANGELOG v0.7.19 and Workload intel "MiniMax H3, block 49".
7. **`qk_rotate` keyword** (v0.7.20, 2026-09-17) on
   `sageattn_qk_int8_pv_fp8_cuda`, and through `sageattn()` /
   `sageattn_consume()` kwargs: a fixed Hadamard rotation of every q and k
   row inside the per-thread INT8 quantizer, exact for the attention math,
   no statistics and no gate. Off by default. Head dim 128 and per-thread
   quantization only, and it RAISES where it cannot apply rather than being
   ignored. An alternative to `qk_balance`, and the stronger one. **The
   entry points swallow unknown keywords**, so against a build older than
   v0.7.20 the option is silently a plain call: check
   `inspect.signature(sageattn_qk_int8_pv_fp8_cuda).parameters` first. Why
   and what it buys: CHANGELOG v0.7.20.
5. **`sageattention.quant.ELEMENT_OFFSET_BITS`** (v0.7.17, 2026-09-13)
   -- the width of the global element offsets the installed fused CUDA
   quant build can form: 64 on any build from v0.7.17 on, 32 on anything
   earlier (the attribute is read off `_fused.ELEMENT_OFFSET_BITS` and
   defaults to 32 when the extension lacks it). This is the fact a
   consumer's sequence-length preflight should key on instead of
   hardcoding the old ceiling: the tracked consumer's preflight computes
   `2**32 // stride_seq` and reports the CUDA v quantizer's crossing as
   "NOT fixed", which is true of a 32-bit build and false of a 64-bit
   one, and only this attribute tells the two apart. Below the width the
   wrappers refuse with a `ValueError` naming the row count, the stride
   and the ceiling, before any kernel launches; a consumer that wants to
   pre-empt that can call `sageattention.triton._int_offsets.max_element_offset`
   on its own tensors against `2**ELEMENT_OFFSET_BITS`. Adding the
   attribute is compatible; its name and integer type are now part of
   this surface.
5. **`sageattn_consume_prefers_cloned_v(device)`** (v0.7.4) -- the
   caller-side way out of that fused case, and the answer to "should I
   clone?". A caller that clones v before handing the list over gives
   it its own storage, so releasing q and k frees the fused buffer, for
   the price of one third of it. Both halves are load-bearing and
   asymmetric: cloning without consuming is a flat cost of twice that,
   and consuming at `smooth_k=True` hands the clone straight back --
   the wrong way. Figures in CHANGELOG v0.7.4.

   **ComfyUI did this as of 2026-08-11 and does NOT any more -- re-read
   2026-09-08.** `comfy/ldm/minimax/model.py` now builds q/k/v as
   `self.qkv_proj(x).split(...)`, i.e. three views of one fused buffer,
   and wraps each in `AttentionTensorContainer`, which stores the tensor
   and does not copy it. There is no `.clone()` on the v path. So the
   built-in path is exactly the fused-views case measured at **0 MiB
   saved either way**: releasing q and k frees nothing while v holds the
   same allocation.

   **So whether the saving happens depends on which path handles
   attention, and the two differ.** A consumer attention-patch node that
   owns the call site can clone `v` itself, and the one tracked here
   does: it gates on `sageattn_consume_prefers_cloned_v` for the device
   it is actually running on, keeps `smooth_k=False` because that is
   what makes the clone pay, and pins the wiring with its own test. On
   that path the saving is realized. On ComfyUI's own built-in sage path
   it is not, and `sageattn_consume` there is correct, bit-identical and
   buying nothing.

   Do not "fix" this by cloning unconditionally. Both halves are
   asymmetric -- the +572/-286 figures above are why a half-applied
   version costs memory instead of saving it -- and the predicate exists
   so the decision tracks the arch and the fork version rather than a
   copied constant.

   Consumers gate on the predicate rather than on an arch check so that
   a) they don't copy `core._EARLY_RELEASE_ARCHS` into their node where
   it drifts silently, and b) they inherit the flip to False if the
   transpose-buffer backlog item ever lands, which retires the clone
   rather than stacking with it. Numbers + reasoning in CHANGELOG
   v0.7.4.

**`attn_mask` must stay a named parameter of `sageattn()`**, not a
`**kwargs` entry. ComfyUI gates masked calls on `"attn_mask" in
inspect.signature(sageattn).parameters`; when that reads False it
routes every masked call to torch SDPA, which silently stranded the
v0.5.5 CUDA mask kernel until v0.7.0. Enforced by a test in
`tests/test_dispatched_kernel_telemetry.py`.

Mask-routing fix landed v0.3.0 (2026-04-26); audit trail in
`internal/audit_2026-04-26.md`. Native CUDA mask landed v0.5.5
(2026-05-13) on sm89 fp8++; scoping doc + measurement trail in
`docs/cuda_mask_kernel_scoping.md`. FFN
fusion landed v0.6.0 (2026-05-15); scoping + day-by-day execution
journal + cross-claude memo trail in
`internal/design/ffn_fusion_scoping.md` (gitignored).

There is also an undocumented L3 contract -- underscore-prefixed
symbols and pybind methods that downstream consumers import by name.
Before removing or renaming any of those, read
`docs/downstream_symbols.md` and run the pre-removal checklist.
