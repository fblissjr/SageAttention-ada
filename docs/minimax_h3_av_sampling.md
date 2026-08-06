Last updated: 2026-08-06 (tracking `Comfy-Org/ComfyUI#15243`, open and unmerged at time of writing)

# MiniMax H3 audio-video sampling: the schedule fix and what it does to our numbers

## TL;DR

H3 packs video and audio into one latent but the two streams were trained on
different flow shifts (video 12.0, audio 3.0), while the sampler has only one
sigma schedule. The shipped code reconciles that by scaling the returned audio
velocity by the chain-rule slope `d(sigma_a)/d(sigma_v)`. That is correct to
first order and no further: `denoised` is not a valid x0 estimate for the audio
stream, and stochastic samplers inject noise sized by the video sigma into a
latent sitting at the audio sigma. Audio therefore only worked on deterministic
Euler at high step counts.

Upstream PR 15243 replaces the derivative patch with a change of variables. The
two shift schedules are the same curve up to a constant factor in the SNR
variable, so carrying the audio latent at scale `sigma_v/sigma_a` turns the pack
into an honest single-schedule flow latent. Every sampler becomes correct on it
with no audio-specific knowledge.

For this fork: **nothing kernel-side moves.** Sequence lengths, head counts,
call counts per step and the attention share are all unchanged, so the fl2va
shape work, `sageattn_consume`, and the quant-overflow fix are untouched. Two
things do move. Output changes at every step count, which retires any H3 quality
baseline that straddles the merge; and low step counts becoming usable is a
larger wall-time lever on H3 than anything available in the kernel.

## Why H3 has two schedules

H3 generates video and audio jointly. Both streams live in one packed latent (a
`NestedTensor` of `[video, audio]`) and both are flow-matching, but they were
trained under different flow shifts: `sigma_shift` 12.0 for video, 3.0 for
audio. The sampler drives a single sigma schedule, the video's. So the audio
latent is integrated against a schedule that is not its own, and something has
to reconcile the two.

The flow shift map, for base-grid point `u` and shift `s`:

```
sigma_s = s*u / (1 + (s-1)*u)
```

Its useful property, which the fix turns on: in the SNR variable
`r = sigma/(1-sigma)`, the map is a pure constant multiplier.

```
sigma_s / (1 - sigma_s) = s*u / (1-u)     =>     r_v = (shift_v / shift_a) * r_a
```

So the video and audio schedules are the same curve stretched by
`k = shift_v/shift_a = 4`. This is the quantity PR 15243 exposes as
`ModelSamplingAV.audio_scale`.

## The defect in the shipped path

The shipped code reconciles at the velocity. `_forward` multiplies the returned
audio velocity by `d(sigma_a)/d(sigma_v)` (the `time_shift_slope` helper the PR
deletes), so that the flat ODE every sampler integrates,
`dX/dsigma_v = (X - denoised)/sigma_v`, carries the right instantaneous
derivative for the audio stream.

That is exactly correct for the derivative and nothing else. Two consequences:

- **`denoised` is not a real x0 prediction for audio.** It is a value
  reverse-engineered to reproduce the correct slope. Plain Euler only ever
  consumes the slope, so it survives. Samplers that treat `denoised` as an
  actual clean-sample estimate -- the DPM++ family, multistep, higher-order --
  are computing on a quantity that does not mean what they assume.
- **Stochastic samplers inject noise at the wrong level.** Ancestral and SDE
  samplers add noise sized by `sigma_v` into a latent whose true noise level is
  `sigma_a`. A velocity correction cannot reach the diffusion term at all.

Add low step counts, where linearizing a nonlinear schedule map across a coarse
step stops being a rounding error, and the observable behaviour is what the PR
title reports: audio worked on deterministic Euler at high step counts and
degraded elsewhere.

The per-sampler-family attribution above is a reading of the mechanism, not a
claim made in the PR. The PR states the symptom (stochastic samplers and low
step counts) and the fix.

## The fix: change of variables, not a derivative patch

Carry the audio latent at scale `c = sigma_v/sigma_a` instead of correcting its
velocity after the fact. Substituting the flow interpolation
`x_a = (1-sigma_a)*x0 + sigma_a*eps` and using `r_v = k*r_a`:

```
y = (sigma_v/sigma_a) * x_a = (1-sigma_v)*(k*x0) + sigma_v*eps
```

The noise term is untouched and the clean target is scaled by a constant `k`.
The pack is now a genuine single-schedule flow latent. Nothing downstream needs
to know an audio schedule exists.

That result drives the shape of the plumbing:

- `process_latent_in` / `process_latent_out` scale the audio slice by the
  **constant** `k`, not by `c`. The carried ratio
  `c = k / (1 + (k-1)*sigma_a)` runs to `k` as `sigma -> 0` and to `1` as
  `sigma -> 1`, so the clean latent needs the constant and the initial pure
  noise needs nothing. Choosing `sigma_v/sigma_a` over the other admissible
  carrying scale is what makes the x0 target a constant multiple, which in turn
  makes masked and img2img blending correct for free: noising `k*x0` at
  `sigma_v` is exactly the carried variable.
- `forward()` multiplies the audio back by `sigma_a/sigma_v` before the wrapper
  chain, so the DiT sees the stream at its own sigma exactly as trained, then
  converts the returned velocity:

  ```
  out[1] = (1-k)*audio_x + (1 + (k-1)*sigma_a) * out[1]
  ```

  Those two coefficients are exactly `dc/dsigma_v` and
  `c * d(sigma_a)/d(sigma_v)`. Verified by hand: with
  `D = 1 + (k-1)*sigma_a`, the map gives `sigma_v = k*sigma_a/D`, hence
  `d(sigma_a)/d(sigma_v) = D^2/k` and `c = k/D`, so `c * slope = D` and
  `dc/dsigma_v = -(k-1)`. The conversion is exact, not a linearization.

- `samplers.py` hands `latent_shapes` to the model so `audio_scale()` returns
  1.0 when it is not actually sampling a packed latent.

Secondary changes, unrelated to the schedule math: the AV-latent packing node
gains swap-into-existing-AV-latent, and trims or zero-pads an audio clip to the
video length with the padded tail left unmasked so the model generates it.

## What changes observably

- Stochastic samplers and low step counts become usable for H3 audio.
- **Output changes at every step count, video included.** The PR author states
  this. The mechanism: old and new are different discretizations of the same
  continuous ODE, so they agree only in the infinite-step limit; and because
  both streams denoise inside one attention sequence, a changed audio trajectory
  feeds back into video. Deterministic Euler is not exempt.
- Initial noise is unchanged (`eps' = eps`), so seed semantics survive even
  though trajectories do not.
- No workflow breakage from the node rename. The PR body says the shift node is
  renamed to `ModelSamplingMiniMaxH3`, but the diff only changes `display_name`
  and adds search aliases; `node_id` stays `MiniMaxH3SigmaShift`, and `node_id`
  is what workflows serialize.
- Latent shapes and packing layout do not change.

## What this does to our H3 measurements

### Unaffected: everything kernel-side

The packed video+audio sequence still goes through one self-attention call per
block. Sequence length, head count, head dim, call count per step and the
attention share of the step are all untouched. So:

- the fl2va shape (S=41822, heads 56, head_dim 128) and every number taken at it
- `sageattn_consume` and its -858 MiB / ~435 MiB peak readings
- the `per_channel_fp8` 572 MiB transient
- the v0.7.0 mask-probe fix and the int32 quant-overflow fix
- the 757.7 ms per-call figure and the 76%-of-step attention share at 362 frames

all stand as kernel measurements. Nothing in this PR touches the attention path.

### Affected: e2e wall time and any quality baseline

The 362-frame reconciliation in CHANGELOG v0.7.1 -- 20 steps at 49.66 s/it,
workflow `h3_t2v_sage_ui.json`, 1344x768, euler / simple, sage mode `auto` --
was taken under the old sampling math. The per-step accounting stays valid
because the step is the same work. What does not survive is any comparison of
*rendered output* across the merge boundary.

Concretely: the Sol-Attn tau sweep on H3 was gated on visual quality of renders
produced under the old sampler. If ComfyUI is updated past this PR, the
reference renders move underneath that verdict. An A/B where one arm predates
the merge and the other follows it is not a valid comparison, on quality or on
anything read off the output. That sits alongside the already-open caveat that
the verdict was established at one aspect ratio and needs re-checking at others.
Detail on the sweep is in the internal record, not here.

Rule for the transition: re-baseline before comparing. Any H3 quality arm has to
be re-shot on whichever side of the merge the comparison lives.

### The bigger lever: low step counts

Our long-sequence H3 conclusion has been that attention is three quarters of the
clock at 362 frames, so further work should aim there. That is still true per
step. But step count is a multiplier on the whole render and sits outside the
kernel entirely.

If H3 becomes correct at low step counts, cutting 20 steps to single digits is a
larger wall-time win than any kernel change available to us, and it is
orthogonal to kernel work: shapes are identical, only the number of calls per
render drops. Worth establishing the minimum viable step count on the new
sampler before ranking further long-sequence H3 attention work by leverage --
the Amdahl input changes, even though the per-step profile does not.

## Watch items

Unverified, flagged for checking rather than asserted:

- `MiniMaxH3.audio_scale()` reads `self.model_sampling.audio_scale`, which only
  exists on `ModelSamplingAV`. A workflow that patches `model_sampling` with a
  generic flow-shift node rather than the H3-specific one would install an
  object without that attribute. Worth confirming what happens on that path
  before running H3 benches with a patched sampling object.
- Whether `audio_shift` reaches `set_parameters` in every path that constructs
  H3 sampling, or only via `sampling_settings` and the H3 shift node.

## Status and triggers

`Comfy-Org/ComfyUI#15243`, opened 2026-08-03, open and unmerged as of
2026-08-06. Three inline review comments from one contributor, all minor: one
asking for `get_model_object('model_sampling')` for consistency and safety, two
file-encoding fixes. None touch the schedule math, so nothing in review so far
bears on the analysis above. (The `get_model_object` note lands on the same seam
as the first watch item below, which is weak corroboration that the seam is
worth checking.) The mechanism analysis above is durable whether
or not this specific PR is the version that lands -- the two-schedule problem
and the change-of-variables solution are properties of the model, not of the
patch.

Triggers:

- **On merge:** re-baseline H3 renders before any further output comparison;
  re-check the minimum viable step count; confirm the watch items above.
- **Cross-clone:** the consumer-side clone tracking audio workloads should know
  its own H3 baselines are affected the same way.
- **No kernel action.** Nothing here justifies work in `csrc/` or
  `sageattention/`.
