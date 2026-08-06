last updated: 2026-08-06

# Gating approximate-attention quality on long video

Block-sparse attention approximations trade output fidelity for speed, and
the trade is worth taking on DiT video at long clip lengths -- attention
follows S^2, so the win grows with the clip. What is not obvious is how to
decide *which* setting to run, because the failure they introduce is easy to
confuse with a failure the model has anyway.

This is a measurement-methodology note, not a recommendation. The specific
threshold that passed here is workload- and geometry-specific and is
recorded with the workload, not in this file.

## The knobs

Approximations of this class typically expose two:

- **A global sparsity threshold.** Higher means more aggressive pruning,
  faster, more approximation error. This is the main speed lever.
- **Per-layer dense exemption.** A list of transformer blocks to compute
  exactly, no pruning at all. Marketed as the targeted fix: pay full price
  in a few sensitive layers, keep the aggressive threshold everywhere else.

The natural assumption is that the exemption is the precise instrument and
the threshold is the blunt one. At the geometry measured here, the reverse
was true.

## Two late-clip phenomena, and merging them costs a day

Long clips near the top of a model's trained frame range show **broad
late-clip softening** -- detail loss, worst at the end, most visible on
distant or small content. This appears with the approximation disabled. It
is not an attention-approximation artifact. **Its lever is frame count**:
stepping down one or two frame-count quanta costs proportionally less
attention *and* reduces the decay. No attention setting addresses it.

Separately, an aggressive threshold produces **content instability late in
the clip** -- objects that formed correctly earlier stop being themselves,
and are replaced by different content rather than merely blurring. This one
is threshold-graded: present above an onset, absent below it.

Both appear late. Both read, on a first pass, as "the end of the clip gets
worse." The first is louder and will absorb the attention the second
deserves, which is how a session concludes "ordinary decay, the
approximation is exonerated" while sitting on a reproduction of exactly the
artifact it was looking for.

**The diagnostic that separates them is the threshold itself.** Re-render
the same seed at a materially lower threshold. Broad softening is unchanged.
Content instability disappears. Anything that survives a large threshold
reduction is not caused by the approximation, and no attention setting will
fix it.

A secondary tell, useful before spending a render: the instability is
*specific* -- one object, replaced. The decay is *general* -- everything
distant, softened. A failure described as "the whole background gets mushy"
points away from the approximation; "that thing turned into a different
thing" points at it.

## Why the per-layer exemption lost

A per-layer sensitivity profile -- approximate-versus-exact output error,
measured per block at a clean input -- shows several-fold variation across
depth. It is real data, and it invites exempting the worst blocks.

Measured here, exempting the highest-error blocks did **not** remove the
content instability, while reducing the global threshold did. The exemption
cost 4-10% of render time for a difference at the edge of perceptibility.

The profile is not wrong; it is being read for something it does not
measure. Per-block error at a clean input ranks blocks by **local,
single-step** sensitivity. The instability is not local -- it is the loss of
long-range temporal association, accumulated across the clip. A block can
rank unremarkably on single-step error and still carry that association.
Ranking by the first and intervening on the second is a category error, and
it is invisible unless you test the intervention against a reproduction.

The structural point generalizes: **a binary knob is the wrong shape for a
graded profile.** If sensitivity varies continuously across depth and the
threshold is what actually governs the failure, then the instrument that
matches the data is a *per-layer threshold*, not a per-layer on/off. Where an
implementation offers one, that is the experiment worth running: aggressive
globally, conservative only where the profile says it matters.

## How the gate fails

Two instrument failures, both the same shape -- sampling where the failure
is not, then reading the silence as evidence.

1. **Tracking the subject where it is easiest to see.** If the shot changes
   framing, an object can be large and readable mid-clip and small and
   unreadable at the end. The failure lives at the end. Dense sampling of
   the readable region returns "clean" about a clip that is not.
2. **Letting the louder phenomenon absorb the diagnosis.** Once *some*
   degradation is visible and has a ready explanation, the second
   phenomenon stops being looked for.

Both are the general form: an absence of observation is not an observation
of absence. Before accepting a clean result, confirm the instrument could
have shown a positive -- which for a visual gate means confirming you looked
where the failure would be, not where the subject is convenient.

## Procedure that holds up

- **Judge intra-clip, never inter-clip.** Arms with different attention
  output diverge into genuinely different videos within seconds; by the end
  of a long clip they share a prompt and nothing else. "Is the object
  present in arm A versus arm B" compares two different renders and answers
  nothing. Judge each arm against *itself*: did something form and then stop
  being itself?
- **Never numerically.** A multi-step ODE diverges under any perturbation,
  so per-pixel metrics between arms measure trajectory chaos, not
  degradation. Summary statistics over a whole track (e.g. loudness) are
  legitimate; per-pixel correspondence is not.
- **Video, not stills.** A grid of stills at sampled times cannot show an
  event that consists of something changing identity over ~4 frames.
- **Same seed across arms**, and establish the positive control before
  interpreting any null: if the most aggressive arm shows nothing, the
  prompt or seed may simply not reproduce the failure, and a clean
  conservative arm then means nothing.
- **Prove the knob fired.** A null from an arm whose setting silently failed
  to apply is indistinguishable from a null that means the setting works.
  Where the implementation logs a per-call path line, the cheapest sound
  check is differential: run the knob at its no-op extreme and at its normal
  value, and compare presence of that line. No timing, so no noise floor,
  and it stays valid regardless of cache warmth or run order.

## Scope limits

**Threshold verdicts are geometry-scoped.** Where the approximation orders
tokens by spatial locality within a frame, rows-per-frame varies with canvas
aspect, so the router's block neighbourhoods are a different shape at every
aspect ratio. A threshold validated at one aspect has not been validated at
another. Timing needs no re-check -- attention follows S^2 and S is
computable from the geometry.

**Clip length matters in both directions.** The instability accumulates with
clip time, so longer clips are the higher-risk case, not the safer one. A
verdict from a short clip does not license a long one.

**One knob not covered here.** Implementations that gate sparsity to a
sampling-percentage window leave the first and last steps exact. Widening
that window is a speed lever, but it removes dense warm-up steps, and early
steps establish composition rather than detail. That is a different failure
mode from threshold aggressiveness and wants its own gate -- it would show
up early in a clip, not late.
