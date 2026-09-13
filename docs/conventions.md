# Conventions: the long-form versions

Last updated: 2026-09-13

`CLAUDE.md` carries every rule in this repo as a one-liner. This file
carries the reasoning and the worked example behind the ones where the
reasoning is the persuasive part. Extracted verbatim on 2026-09-08.

If a rule here and a rule in `CLAUDE.md` ever disagree, `CLAUDE.md` is
the one that was loaded and the one to follow -- then fix this file.

## `main` is what the server runs (2026-09-13)

The consumer's ComfyUI venv holds an editable install of this checkout.
That means the server does not run a version of sage; it runs the working
tree, whatever it holds, at the moment the process starts. A half-finished
experiment left checked out overnight is what tomorrow's renders use, and
nothing warns.

Three rules follow, agreed with the owner on 2026-09-13:

1. **Only measured, changelog-recorded changes land on `main`.** The
   measurement is the thing that makes a change servable; the changelog
   entry is where a later reader finds the conditions. A kernel change
   with a pending table is not on `main` -- the one exception so far,
   v0.7.17, was committed with its table pending and the entry saying so,
   and only because the static check and bit-identity below the ceiling
   were already in hand. Do not generalize from it.
2. **Experiments go on a branch, and the tree is back on `main` before a
   session ends.** Spikes that measure without changing shipped code can
   land on `main` as harnesses; anything that changes what `sageattn()`
   does or what the kernels compute stays on a branch until its record
   exists. The last act of a session that branched is `git switch main`,
   because the next server restart does not ask.
3. **Tag each build the server has run.** `served/<date>` on the commit
   whose tree the server started on, with the provenance in the tag
   message: which commit the kernels were compiled from and when, into
   which venv, the extension's `ELEMENT_OFFSET_BITS`, and the measurement
   that graded it. A consumer's render record then names a commit rather
   than "HEAD at the time", which is the gap `build_info()` was added to
   close from the other side. Tags are local until someone is asked to
   push.

The first tag is `served/2026-09-13`. Its message is the template.

## Retracting a wrong framing

- **Retract wrong-framing in committed docs via `git revert`, not
  in-place edit.** The revert preserves the wrong commit + its
  message in `git log` and supersedes it with a revert commit on
  top; the audit trail of "we believed X, then disproved X" stays
  reconstructible. An in-place edit leaves the wrong framing in the
  diff history as an unflagged precursor that future `git log -p
  <file>` would surface without context. Worked example: the
  2026-05-16 "47% comfy-aimdo offload" workload-profile claim
  (commit `a05fdf4`) retracted via `git revert` at `95af2cf` after
  a `nodynvram` A/B disproved the framing.

## Task and caller references in committed code

- **Task/caller refs in committed code don't age.** Memo timestamps
  ("07:45Z"), cross-clone references ("per X's note"), and
  session-specific framings ("today's spike showed") decay -- the
  memo trail isn't in the repo, and a reader six months later can't
  reconstruct context. Replace with the substantive reason: cite the
  production precedent, the cross-version stability concern, or the
  file:line of the canonical source. Easy to introduce; easy to
  scrub in /simplify; the second cycle is wasted effort.

## Kernel signature changes: the four-place coupling

- **Kernel signature changes are a four-place coupling on the sm89
  path.** Adding a runtime param to a sm89 kernel (e.g. v0.5.5
  `attn_mask`) needs (a) the `.cuh` template + kernel-launch sites
  in all 7 sm89 `.cu` files, (b) the C++ entry + `attn_cuda_sm89.h`
  decl, (c) pybind def with `py::arg(...)=c10::nullopt` defaults,
  (d) `sageattention/sm89_compile.py::@torch.library.custom_op`
  schema + matching `register_fake` stub. Forget (d) and the call
  fails at runtime with "expected at most N argument(s) but
  received N+1" -- pybind alone isn't enough. Worked example:
  CHANGELOG v0.5.5 + the kernel-correctness-reviewer agent.

## Gating ship decisions on in-pipeline measurement

- **Gate ship-decisions on in-pipeline A/B when synthetic-bench
  can't measure the dominant cost.** Two layers to this rule:
  (a) Don't *claim* a delivered speedup from a synthetic number;
  default framing is "synthetic-bench above, e2e pending in-pipeline
  measurement" until a downstream A/B confirms the wedge transfers.
  (b) For kernel-day work with structural risk that synthetic-bench
  *specifically* can't measure (L2 contention with neighboring
  modules in the production hot loop, cumulative dispatch overhead
  at high call counts, memory-allocator behavior under fragmentation,
  thermal/clock state during sustained renders), gate the v0.X ship
  commit on an in-pipeline measurement BEFORE the commit lands, not
  after. The v0.6 walk-back was the cost of running this rule
  ship-first-validate-later.
  Two precedents: v0.5.5 chunk-bypass A/B (synthetic mask-kernel win
  softened once `LTXVChunkFeedForward` was shown to be doing the
  load-bearing memory work) and v0.6 sage_ffn (a synthetic win came
  back slower end to end on a two-sampler LTX FML2V workflow, due to L2
  contention plus cumulative launch overhead; figures in CHANGELOG
  v0.6.0). Especially load-bearing for per-call-heavy primitives, since
  an LTX render fires on the order of a thousand FFN/MLP calls
  times per LTX render -- any per-call overhead compounds and any
  cache-locality assumption made under isolation can break).

## Triton kernel-day traps

- **Triton kernel-day discipline.** Four recurring traps:
  (a) `@triton.jit` can't read module-level Python globals -- inline
      literals in the kernel body (e.g. `448.0` for FP8_E4M3_MAX).
  (b) For DiT FFN/MLP kernels, audit BOTH Linear layers for `bias=True`
      on the target checkpoint. LTX 2.3 distilled has bf16 biases on
      both `ff.net.0.proj` and `ff.net.2`; shipping a bias-free kernel
      silently corrupts output (not "fp8 quant noise" wrong, "missing
      a constant offset everywhere" wrong). Caught pre-day-9 in v0.6.
  (c) Broad `@triton.autotune` sweeps (>~30 configs) burn minutes of
      first-render-per-shape on user hardware (126 configs = ~7 min
      cold). Pattern: tune full sweep once, extract winners via
      `kernel.cache.items()`, hardcode ~8 configs + neighbors. Worked
      example: v0.6 sage_ffn (CHANGELOG v0.6.0).
  (d) CUDA event timing across streams captures queue-wait + execution,
      not just kernel duration. When `e_start.record()` is on the
      default stream and `e_end.record(s_other)`, `elapsed_time` between
      them measures `s_other`'s wait-for-SMs plus the kernel. A `t_ms`
      variable name telegraphs "this is kernel time"; future readers
      will misread. Use `*_end_offset_ms` or similar to signal the
      asymmetry vs single-stream timing. Worked example:
      `tests/spikes/spike_concurrent_dispatch.py` (renamed in /simplify
      pass after the bare metric misread).
  (e) Raw CUDA kernel launches default to stream 0 if the 4th arg is
      omitted. `<<<grid, block, smem>>>` silently breaks any caller that
      wraps sage in `with torch.cuda.stream(...)` -- Triton kernels in
      the same Python call respect current stream, the CUDA launch
      doesn't, and the race surfaces as small-but-stable rtol drift
      (~0.02) or NaN under a partial fix that only patches the attn
      kernel but leaves `csrc/fused/fused.cu`'s quant pre-kernels on
      stream 0. Use `<<<grid, block, smem,
      at::cuda::getCurrentCUDAStream()>>>` and
      `#include <ATen/cuda/CUDAContext.h>` on every site (`csrc/qattn/`
      sm89/sm80 + `csrc/fused/fused.cu`). Worked example: v0.6.1
      (CHANGELOG).
