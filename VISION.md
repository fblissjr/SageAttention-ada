last updated: 2026-04-26

# sage-fork vision

## Why this fork exists

Two reasons. First, [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention)'s
`setup.py` refactor narrowed the SM80 build gate to `(8.0, 8.6, 8.7)`,
silently dropping Ada source builds on RTX 40xx. We add `8.9` back so
sage compiles on the hardware we actually run. Second — the
load-bearing reason — this fork is the **measurement surface for sm89
attention-kernel decisions.** New kernel variants, autotune coverage,
mask-handling work, perf comparisons against torch SDPA / FlashInfer /
SpargeAttention all get characterized here, against real model shapes
(LTX 2.3 video, Z-Image-Turbo, Flux-class image gen), before they
land anywhere.

History was squashed at 2026-04-23 to drop ~196 commits of upstream
divergence. Assume `main` is ours; upstream
([thu-ml](https://github.com/thu-ml/SageAttention) and
[woct0rdho](https://github.com/woct0rdho/SageAttention)) are
reference points, not destinations. Periodic survey, no remote
tracking. See [`CHANGELOG.md`](./CHANGELOG.md) "Recurring process
items" / "Periodic upstream survey" for the schedule.

## What we are

- **An editable install** for any PyTorch project that wants sage on
  sm89 / RTX 40xx / Ada. Consumer-agnostic: anything that imports
  `sageattention` or replaces
  `torch.nn.functional.scaled_dot_product_attention` picks up this
  fork via the editable install.
- **A bench harness + decision log** that grades every kernel-side
  change against ONE row of one test (the load-bearing metric below).
- **A consumer-agnostic primitive.** Kernels and bench, not policy.
  Routing decisions, telemetry plumbing, and workflow integration
  belong in the consumer.

## What we are not

- **A general sage replacement.** Hopper / Blackwell support stays
  with upstream. We compile and run on those archs (the kernels are
  upstream's), but we don't validate, debug, or optimize for them.
- **A perf-tuning consultancy for individual workloads.** Consumers
  bring shapes; if a model class brings a head_dim or sequence
  pattern outside our coverage, the fix is a new bench row, not a
  workload-specific kernel. See [`CLAUDE.md`](./CLAUDE.md) "Hardware
  target."
- **A `torch.compile` target.** Verified 2026-04-25 on torch 2.11:
  compile-around-sage produces ~2.8% rtol drift with no measurable
  speedup. Consumers should keep `torch.compiler.disable()` around
  sage calls. Revisit when a future torch release makes
  [`tests/spike_torch_compile.py`](./tests/spike_torch_compile.py)
  show bounded rtol AND a measurable speedup.
- **An upstream tracker.** Squashed 2026-04-23. Quarterly survey
  (`CHANGELOG.md` Recurring process items) catches kernel-side
  bugfixes from `thu-ml` or regressions from `woct0rdho`; everything
  else stays in their tree.

## The single metric

```
tests/test_sageattn_ltx_shapes.py
  shape: self_attn_large_704x704x497  (B=1, H=32, Sq=Skv=31776, D=64, no mask, bf16)
  mode:  fp8_cuda++
  -> primary perf metric: median_ms (today: 19.95 ms)
  -> accuracy guard:      mean_rtol ≤ 0.10 (today: ~0.097)
  -> cross-session normalizer: torch_flash / sage_fp8++ ratio (today: 2.62×)
```

This row dominates real LTX gen wall-time. `fp8_cuda++` is what
`sageattn()` picks on sm89 + CUDA ≥ 12.8 unmasked, so it's the only
kernel a real perf change actually moves on the hot path. The rtol
guard (≤ 0.10) is the documented accuracy floor for this fork —
crossing it means breaking the README's stated promise, which is
not a tradeoff we make silently.

The full framework — why this row, what side-effects to check on
every change, how to use the bench output to pick the next
experiment, what we explicitly ignore and the trigger that would
change that, what the framework might be wrong about — lives in
[`CLAUDE.md`](./CLAUDE.md) under "Performance research: the
load-bearing metric."

## Hardware target

sm89 / RTX 40xx / Ada only. Other archs compile and run; we don't
validate them. Other-arch source builds: widen the SM80 build gate
in [`setup.py`](./setup.py) line 152 to match thu-ml's coverage.

## Design choices

- **Primitive over policy.** The consumer routes; the fork measures.
  The v0.3.0 dispatcher mask-routing fix is the limit case: the
  routing was correctness, not policy, so it landed here.
- **Correctness before perf.** v0.3.0's silent-mask-drop took a
  10-line dispatcher fix and a regression test. The native CUDA mask
  kernel — days of kernel work for at most ~2× speedup at sub-ms
  cross-attn — stays deferred, with the K=1.68 measurement
  supporting the deferral.
- **Measurement before decision.** Deferrals carry concrete
  reopen-numbers, not vague "trigger fires." The
  `cross_attn_unmasked_kv226_kratio_probe` row exists specifically
  so K is readable every bench run; without it, the trigger could
  never fire.
- **Forks-not-PRs.** We own this. Upstream is a reference; our
  CHANGELOG is the canonical divergence record.
- **Honest about what's V1.** The load-bearing metric assumes LTX
  self-attn dominates the workload mix. Mean rtol is a proxy for
  perceptual quality, not the truth. Kernel ms is not gen ms. The
  five-pattern next-experiment framework hasn't been validated on
  an actual experiment yet. See "What we might be wrong about"
  below.

## Where this gets used

Four public entry points: `sageattn()` (the top-level dispatcher,
which routes by `(arch, CUDA version, mask presence)` on sm89 + CUDA
≥ 12.8), per-kernel exports for callers that want to bypass the
dispatcher, `sageattn_warmup()` for priming Triton's JIT cache, and
`get_last_dispatched_kernel()` for consumer tracers. See
[`README.md`](./README.md) "Where it gets used" for signatures,
scopes, and the mask-handling caveats on hand-picked kernels.

## What we might be wrong about

The metric and framework above reflect the workload mix on this box
as of 2026-04-26. Four candid limitations:

1. **The "LTX self-attn dominates" assumption is workload-specific.**
   If a new model class with fundamentally different attention
   patterns (very-short autoregressive seq, sliding-window, MQA/GQA
   with very different head ratios) becomes the primary use case,
   the load-bearing shape moves and the metric should be re-derived.
   Disconfirming signal: a downstream-consumer telemetry summary
   showing a non-LTX-class shape consuming > 30% of gen attention
   time.
2. **Mean rtol is a proxy, not the truth.** A perf change that
   improves rtol but visibly degrades a real render fails the spirit
   of the guard. We don't have a perceptual eval here (it'd be a
   per-frame structural-similarity bench against an fp32 reference
   render). If we ever ship a kernel change that passes rtol but
   triggers a consumer-reported visual regression, the rtol guard
   isn't the right floor and we add the perceptual layer.
3. **Kernel ms is not gen ms.** A 2× kernel speedup is invisible
   end-to-end if attention is already < 50 % of step time. We don't
   measure end-to-end here; that's downstream-consumer telemetry. A
   "this saved 5 ms per call" claim should be paired with "and we
   observed a real LTX gen go from X seconds to Y" before ranking
   high.
4. **The next-experiment framework is V1.** It codifies a strategy;
   the strategy hasn't been validated by running through it on a
   real perf change yet. The first time we use it to pick a
   direction and either succeed or fail, the framework gets
   refined. Treat the five patterns as starting hypotheses, not
   playbook.

When any of the above fires (disconfirming signal observed), update
[`CLAUDE.md`](./CLAUDE.md) "Performance research" / "What we might
be wrong about," record the change in the session log, and revisit
this VISION.md if the philosophy shifted.
