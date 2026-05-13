# Performance research framework

Last updated: 2026-05-13

L3 reference for CLAUDE.md's "Performance research" pointer. Load this
when you are about to make a kernel change, run a perf experiment, or
revisit the load-bearing metric. The metric itself lives in CLAUDE.md;
the reasoning chain, the side-effect checks, the experiment-selection
patterns, and the recorded uncertainties live here.

## Load-bearing metric

```
tests/test_sageattn_ltx_shapes.py
  shape: ltx23_video_self_attn_init_22932  (B=1, H=32, Sq=Skv=22932, D=128, no mask, bf16)
  mode:  fp8_cuda++
  -> primary perf metric: median_ms (today: 20.20 ms)
  -> accuracy guard:      mean_rtol <= 0.10 (today: ~0.098)
  -> kernel speedup ratio: torch_flash / sage_fp8++ = 2.66x (today)
  -> e2e speedup ratio:    1.22x (v0.5.1 first empirical measurement;
                           831x480x497 / 25fps / 8-step distilled;
                           VAE-decode-cold-start-normalized)
```

The kernel ratio (2.66x) is what the bench measures directly. The
e2e ratio depends on the workload's attention share -- and the share
varies. Two measurements on file:

- **audio_loop_latent.api.json (832x480x497 / 25fps / 8-step distilled,
  v0.5.1 e2e bench)**: attention 8.2% of wall, e2e ratio 1.22x
  VAE-decode-cold-start-normalized. Pure-attention Amdahl with the
  2.66x kernel ratio predicts ~1.05x; observed exceeded that by 17
  points. The v0.5.1 entry attributed the +17pt to "FFN-adjacent
  reach within the sampler step" -- but that was a single-data-point
  inference (arm-2 attention time was never directly traced; the
  attribution assumed the kernel ratio held but didn't measure it on
  the actual workload). Treat the 1.22x as load-bearing; treat the
  FFN-adjacent mechanism story as an unverified hypothesis.
- **iclora at production scale (audio-loop-music-video_latent_iclora
  workflow, sage-on/sage-off A/B 2026-05-07)**: attention is ~42% of
  CUDA kernel time (76.5/183.4 sage-off). Per-kernel ratio is 3.08x
  on the actual call mix (matched 3456 calls/render in both arms;
  sage 7.20 ms/call vs torch flash 22.14 ms/call). Strict Amdahl with
  these inputs predicts 1.39x; measured wall ratio is 1.41x -- match
  within 1.4%. **No non-Amdahl term is needed on iclora.** Kernel-
  level decomposition shows non-attention CUDA time is essentially
  identical sage-on vs sage-off (delta -2.1s, in the wrong direction
  for any "FFN-adjacent" or "cache-footprint" reach). Launch-overhead
  delta is 0.82% of total launches -- ~0.4s, also negligible. The
  full saving comes from sage's faster attention kernel.

Both numbers are real; they describe different workloads with
different attention shares. The framework: measure attention-share-
of-CUDA-time on each workload of interest, apply Amdahl with the
per-kernel ratio observed on that workload's actual call mix, and
treat any residual as a hypothesis that needs its own measurement
before going into a perf claim. Don't generalize one workload's
ratio to another.

Sourced from real consumer traces; see CHANGELOG v0.4.1 for the
kernel-bench shape re-derivation, v0.5.1 for the audio_loop_latent
e2e measurement, and the 2026-05-07 cross-claude memo trail
(`internal/AUDIO_LOOP_CLAUDE_TO_SAGE_CLAUDE_MEMO.md` +
`internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md`) for the iclora
A/B decomposition that retired the launch-overhead, FFN-adjacent,
and cache-footprint hypotheses on that workload.

The earlier metric (`self_attn_large_704x704x497` at seq=31776, D=64)
was a synthetic shape with the wrong head_dim -- LTX 2.3 video is
D=128, not D=64.

## Why this is the metric (the load-bearing reasoning)

The chain matters; if any link breaks, the metric moves.

1. **LTX 2.3 video self-attn at seq=22932/23296 dominates real gen
   wall-time.** Per the consumer trace
   `sage_2026-04-26_105851.jsonl`, video self-attn (init seq=22932 +
   loop seq=23296, both at d=128) accounts for **76% of total
   attention wall-time** across a typical render. Audio self-attn
   (d=64) is another ~5%; short-Q paths (text-encoder / audio
   cross-attn at seq~497) are ~19% by call count but only ~3% by
   wall-time because each call is sub-millisecond. The video d=128
   row is where milliseconds compound into seconds of gen time.
2. **`fp8_cuda++` is what `sageattn()` picks on sm89 + CUDA >= 12.8
   unmasked.** That's the consumer's actual hot path -- the
   dispatcher routes there for self-attn after the v0.3.0 mask-aware
   fix. Optimizing a kernel that the dispatcher doesn't pick is
   research that doesn't ship.
3. **The fp8++ kernel is where every plausible perf change lands.**
   Edits to `csrc/qattn/sm89_qk_int8_sv_f8_*.cu`,
   `sageattention/quant.py` (per-block / per-warp INT8 quant), the
   fp8 V-quant `scale_max` in `core.py:905-909`, or the SM89 PV
   accumulator variants all show up on this row. Triton-side
   changes show up only on the cross-attn rows.
4. **The README's "<0.1 mean rtol" promise is the accuracy ceiling.**
   If a perf change pushes mean_rtol > 0.10, the fork's documented
   accuracy floor is gone. That's not a tradeoff to make silently;
   it's a re-pitching of the fork.

## How we measure it

`tests/test_sageattn_ltx_shapes.py` is the only thing you need to
run. The bench's `time_and_vram` does 1 warmup + median over 3 runs
to kill within-session noise; absolute median_ms is the
within-session signal you optimize against during a research sitting.
Peak working-set VRAM (MiB) is captured in the same pass at zero
extra kernel cost.

For comparing across sessions (after a torch / triton / CUDA / driver
bump, or after a cold boot), use the **`torch_flash / sage_fp8++`
ratio** instead of absolute time (today: 2.66x at the v0.4.1 primary
shape). The ratio normalizes against driver-thermal drift, which is
on the order of 1-2% across cold boots even with no code changes --
see CHANGELOG's cu128->cu130 transition note. If absolute fp8++ time
drifts but the ratio holds, it's the box, not the code.

The regression-gate floor is encoded in `tests/regression_baselines.json`
under `speedup_ratio_floor` (currently 1.5x) -- sage drops below that
on the primary row -> the fork's reason to exist is empirically
suspect. The `--check-regression` flag exits non-zero on any
load-bearing perf drift > 5%, rtol budget breach (> 0.10), or
speedup-ratio floor breach.

Bench env (torch / triton / CUDA / sage rev) pinned to
`internal/bench_env_<date>.txt`; resnapshot after any version bump
per `docs/bench_env_discipline.md`.

## How we detect unintended side effects

The harness already prints every check side-by-side in one run. Read
all of these every time you change a kernel -- don't tunnel-vision on
the primary row.

- **All 5 sage kernels + 3 torch backends on every shape.** A change
  that helps fp8++ but hurts fp16_cuda or fp16_triton means you
  shifted a knob that's shared between code paths; either intentional
  or a foot-gun.
- **The cross-attn-with-mask kv sweep (32, 64, 128, 226, 512, 1024).**
  Catches regressions in the masked path. Pre-v0.3.0 the dispatcher
  silently dropped masks here; now it routes to triton. The triton
  row's rtol should stay ~=0.04 across the sweep; CUDA rows stay
  pinned at the documented mask-bug fingerprint (0.94->0.13).
- **The cross-kernel `fp8++ vs triton` rtol row** (unmasked shapes
  only). Should sit ~=0.10 -- quadrature of each kernel's independent
  ~0.04 / ~0.09 vs SDPA. If it spikes above 0.15, a kernel-internal
  numerical change broke the cross-kernel agreement, even if neither
  kernel's solo rtol-vs-SDPA changed.
- **Image-gen shapes** in `tests/test_sageattn_image_shapes.py`
  (head_dim=120 Z-Image, head_dim=128 Flux). A kernel change keyed on
  head_dim=64 might silently break the non-power-of-2 d=120 path.
- **Dispatcher telemetry** (`tests/test_dispatched_kernel_telemetry.py`).
  Verifies routing invariants -- the `auto` row matching the wrong
  kernel name post-change is the v0.3.0 mask-routing regression
  signal in primitive form.
- **`tests/run_all.sh`** runs all of the above in one shot. Use it
  before declaring a perf change done.

## How we use the metric to pick what to try next

The bench output is also a diagnostic for where to spend the next
research hour. Five patterns to look for:

1. **Where kernels disagree on rtol, the gap is the optimization
   target.** If fp8_cuda++ shows 0.098 rtol at 20ms and fp16_cuda
   shows 0.037 rtol at ~34ms, the 0.061 rtol delta is "FP8
   quantization cost." The research question becomes: is there a
   variant of fp8 quant (scale_max, granularity, per-block Q mean,
   etc.) that closes some of that gap at similar speed? If you
   measure two fp8 variants and they're indistinguishable in rtol,
   you're at the FP8 information floor and should look elsewhere.
2. **Where kernels agree, you're at the numeric floor -- stop
   optimizing the kernel and look elsewhere.** Two kernels with
   different code paths producing the same number means the
   underlying numerics, not the implementation, is the bottleneck.
   Move up the stack: torch.compile around sage, fusion with
   adjacent ops, model-side activation reformulation.
3. **Speedup-ratio degradation tells you which torch path got
   better.** If `torch_flash / sage_fp8++` drops from 2.66x to 1.8x
   on a future torch release, torch closed gap somewhere -- check
   the `torch_flash`, `torch_eff`, `torch_cudnn` row that improved
   most and figure out what changed. That's where fp8++ is leaving
   perf on the table.
4. **Short-Q rows where sage loses to torch_flash.** v0.4.1 bench
   shows the seq=497/498 short-Q paths (Gemma 3 text-encoder /
   audio cross-attn) at ~0.45x vs torch_flash -- sage is materially
   slower on short shapes because int8 quant + kernel launch
   overhead exceeds the matmul work. The consumer's `nodes_sage.py`
   has a deferred "min-sequence skip" backlog item; this is the
   empirical evidence that gates it.
5. **The unmasked-vs-masked timing gap quantifies the deferred CUDA
   mask kernel.** Today triton is the only mask-correct path; if
   `triton @ kv=N` is K x slower than `fp8++ @ kv=N` (unmasked) at
   the same shape, K is the speedup ceiling for the deferred Backlog
   item "Add mask support to the sm80/sm89 CUDA kernels." If K < 2x,
   the kernel work probably isn't worth days of effort. If K > 5x at
   shapes the consumer actually hits, the trigger fires. The probe
   row `ltx23_video_cross_unmasked_kv226_kratio_probe` in
   `tests/test_sageattn_ltx_shapes.py` exists specifically so K is
   measurable -- it pairs with `ltx23_video_cross_text_kv226` (same
   shape, masked) so K = triton_masked_ms / fp8++_unmasked_ms is just
   two numbers from the bench output. **Measured 2026-04-27 at the
   corrected d=128 video config:** K ~= 1.57 at kv=226 (triton 1.16ms
   / fp8++ 0.74ms), still well below the 5x trigger, so the deferred
   kernel work is not perf-justified today. Re-measure after every
   kernel-side optimization that lands on the unmasked cross-attn
   path; if fp8++ at small kv gets meaningfully faster, K grows and
   the trigger could fire even with no triton change.

## What we explicitly ignore -- and the trigger that would change that

These rows print every run; we don't optimize against them today.
Each one comes with a re-evaluation trigger so we don't keep
ignoring them after the world changes.

- **`fp8++ vs triton` cross-kernel rtol row as a perf signal.**
  It's a consistency check, not a speed measurement. **Trigger to
  care:** the row spikes above 0.15, indicating mixed-route
  consumer forward passes are now seeing a discontinuity beyond
  combined-noise.
- **Cross-attn-with-mask perf rows.** Triton at sub-millisecond is
  already fast enough that perf wins on this path don't move real
  gen time. **Trigger to care:** a downstream consumer's per-call
  JSONL trace, aggregated over a real production gen, reports
  masked-triton as >5% of total gen wall-time. (See CHANGELOG
  Recurring process items / "Session-level attention telemetry
  summary.")
- **Image-gen perf rows.** Already 1.7-2.1x faster than `torch_flash`
  on Flux + Z-Image-Turbo; not the hot path for the sm89 box's
  primary workload. **Trigger to care:** a new model class lands
  with shapes that show < 1.3x speedup, or a consumer reports image
  gen wall-time is now attention-bound.
- **`torch_eff` and `torch_cudnn` rows.** Regression telemetry for
  "is sage still load-bearing as a fork?" -- not a perf signal for
  sage changes. **Trigger to care:** sage's speedup ratio drops below
  1.5x on the primary row, which triggers a "is the fork still worth
  maintaining?" review rather than a perf experiment.
- **Spike `tests/spike_torch_compile.py` perf delta.** Verdict on
  torch 2.11: keep the consumer-side `torch.compiler.disable()`.
  **Trigger to care:** re-run after any torch upgrade; the spike
  itself records the reopen condition (bounded rtol AND measurable
  speedup).

## What we might be wrong about (the framework is V1)

This metric reflects the workload mix on this box as of 2026-04-26.
We may be wrong in ways that take time to surface; record the
disconfirming evidence rather than wait for it to be obvious.

- **The "LTX self-attn dominates" assumption is workload-specific.**
  If a new model class with fundamentally different attention
  patterns (very short autoregressive seq, sliding-window, MQA/GQA
  with very different head ratios) becomes the primary use case, the
  load-bearing shape moves and the metric should be re-derived.
  Disconfirming signal: a downstream-consumer telemetry summary
  showing a non-LTX-class shape consuming > 30% of gen attention
  time.
- **Mean rtol is a proxy for "does the output look right," not the
  truth.** A perf change that improves mean_rtol but visibly
  degrades a real render fails the spirit of the guard. We don't
  currently have a perceptual eval in this repo (it'd be a new
  bench, probably keyed on per-frame structural similarity vs an
  fp32 reference render). If we ever ship a kernel change that
  passes the rtol guard but causes a consumer-reported visual
  regression, the rtol guard isn't the right floor and we need to
  add the perceptual layer.
- **Kernel ms is not gen ms.** A 2x kernel speedup is invisible
  end-to-end if attention is already < 50% of step time. We don't
  measure end-to-end here (it's downstream-consumer telemetry); a
  refinement would be a `gen_wall_time / attention_kernel_time`
  ratio captured per-render, so kernel improvements get an
  end-to-end translation factor. Until then, a "this saved 5ms per
  call" claim should be paired with "and we observed a real LTX gen
  go from X seconds to Y seconds" before ranking high.
- **The "find next experiment" framework above is forward-looking;
  we haven't run a perf experiment through it yet.** The first time
  we use it to pick a direction and either succeed or fail, the
  framework gets refined. Treat the five patterns as starting
  hypotheses, not validated playbook.

If any of the above bullets fires (disconfirming signal observed),
update this section and record the change in the session log so the
re-derivation is auditable later.

## Verification discipline: coderef/ + both-arms-measured

Two rules that gate perf-claim quality:

- **Verify aspirational doc claims against actual code.** Twice in
  one session (dispatcher mask routing, `sageattn_warmup` "consumers
  call it") a public-API doc claimed "X is used by Y" or "dispatcher
  does Z" -- both turned out to be aspirational, not implemented. One
  `grep -r "<api_name>" coderef/` for consumer call sites + a quick
  read of the dispatch code catches this in seconds. Use `coderef/`
  proactively as a verification surface, not reactively at audit
  time. Audit trail in `internal/audit_2026-04-26.md`. Beyond code:
  `coderef/<consumer>/data/runs/*/profiler/summary.txt` chrome-trace
  categorizations are the authoritative answer to "where does GPU
  time actually go." Read before promoting any perf *mechanism*
  claim. The 2026-05-07 FFN-adjacent retirement was sitting in the
  archive for 6 days before use.
- **Mechanism claims need both arms measured, not inferred from
  one.** A perf *number* can come from one measurement; a perf
  *mechanism* claim ("sage's reach extends beyond attention into
  the sampler") needs both A/B arms directly instrumented. CHANGELOG
  v0.5.1's "FFN-adjacent reach" claim was promoted to load-bearing
  status on a single-data-point inference (arm-2 attention time was
  never traced) and lived for 10 days before the 2026-05-07
  cross-claude A/B retired it (CHANGELOG Decision log). When in
  doubt, write the claim with the workload + measurement context
  attached so the inference-vs-measurement distinction stays
  visible.

## Record priors before measurement

For any non-trivial measurement (bench-fire, perceptual eval,
ablation), commit the expected result in writing before the
measurement runs. "Did the number confirm or surprise?" extracts
more learning than "what was the number?" The bilateral pre-bench
briefing exercise produced two priors (literature-estimate vs
Amdahl-from-exec-log); the more grounded one won, both sides
aligned. Without the commit-before-measuring step, post-hoc
rationalization wins.

## Pre-trigger briefing pattern

For any user-gated trigger (currently: e2e bench run, perceptual
data from the consumer's eval track), stage a pre-trigger brief BEFORE
the trigger fires. Lives at `internal/brief_pre_<trigger>.md`
(gitignored). Forces a state-audit that catches doc drift the
regular audit pass misses -- this caught a stale `start.sh` claim
in `internal/runbook_bench_e2e_ltx.md` that would have misled an
operator at bench-fire time. Format: recorded prior (commit before
measuring, per the framework's "what we might be wrong about" item
3), output expectations, durable context the other side would
otherwise re-derive at trigger time.
