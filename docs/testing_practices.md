# Testing practices

Last updated: 2026-09-13

Conventions and traps for writing measurements in this repo. Extracted
verbatim from `CLAUDE.md` on 2026-09-08. `CLAUDE.md` keeps the one-line
rule and points here; this file keeps the worked examples, which are the
part that makes the rule persuasive but which nobody needs loaded on
every session.

The load-bearing item is the instrument-can-fail catalogue below.

## The synthetic-input rule, and the three-way split it rests on

This is the rule `CLAUDE.md` states in three lines. The split below is
the part people get wrong -- it is not "synthetic benches are bad".

**A synthetic-input accuracy number is not a measurement of anything we
ship. Capture real q/k/v.** `torch.randn` has no channel offset (so
`smooth_k` is inert), no attention structure (so softmax is near-uniform
and the output is a near-cancelling average, which inflates every
element-wise relative error), and no relationship to the activation
magnitudes a trained model produces. Measured consequence at H3 config:
the synthetic bench reports fp8++ **4x worse** than reality and the
fp8-vs-fp16 gap **2x wider**. `tests/spikes/spike_h3_real_activations.py`
is the harness; it takes `.pt` files of captured q/k/v and a
throwaway hook on the consumer's attention call site produces them. Do
that instead of adding synthetic rows.

**The same trap applies harder to anything approximate.** A quantizer's
error does not depend on attention having structure; a block-sparse
router's entire function does, so synthetic input measures a routed
method precisely where its premise fails. Any approximation-vs-dense
figure taken on `randn` is a bound so loose it is misleading in the
pessimistic direction -- see the referent rule under Conventions.

**The rule is about accuracy specifically, and the three-way split
matters -- do not read it as "synthetic benches are bad".**

| question | synthetic input | why |
|---|---|---|
| **speed, VRAM** | fine | input distribution does not change the work done |
| **fidelity** -- does this kernel compute what its own reference computes; is the output bit-identical; does the load stay in bounds | fine, and often **required** | a property of the arithmetic, not of the data. Constructed inputs beat real ones here: `tests/test_short_seq_tail.py` dirties smem with `+inf` before every sweep, which real activations never produce, and without it the file is green against a kernel already proven broken |
| **accuracy** -- distance from exact attention, or anything standing in for perceptual quality | **not a measurement** | the whole error depends on structure `randn` does not have |

Cross-implementation agreement rows (`fp8++vs.triton`) are fidelity, not
accuracy. The tables below are kept as the record of what synthetic
inputs said and for their speed and VRAM columns. **Do not quote their
rtol columns.**
 It is
this repo's most-repeated defect class, and `docs/drift_audit_and_directions.md`
adds two more instances found on 2026-09-08.

Reuse `accuracy_metrics` from `tests/test_sageattn_ltx_shapes.py:160`
for rtol/atol comparisons (symmetric denominator; matches every
other accuracy bench in the repo, including `tests/test_partitioned.py`).
Scripts under `tests/spikes/` need a one-line
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`
before the import.

For bit-identicality checks on bf16/fp16 outputs (e.g. stream-safety
spikes), use `torch.equal(a.view(torch.uint16), b.view(torch.uint16))`
-- bare `torch.equal(a, b)` returns False whenever either tensor has
NaN even at identical bit patterns, and random mockup weights at LTX
shapes frequently produce NaN positions. The uint16-view sidesteps
NaN-equality semantics. Worked example:
`tests/spikes/spike_concurrent_dispatch_submodule.py::correctness_sanity`.

Spike scripts under `tests/spikes/` wrap measurement loops in
`torch.inference_mode()` (stricter than `no_grad` -- drops
version-counter tracking; matches the sampler/consumer path under
which these kernels actually run). Add NVTX ranges
(`torch.cuda.nvtx.range("label")`) on every measurement region so
`nsys profile` timeline view shows labeled kernels in addition to
raw aggregation. Both conventions applied in
`tests/spikes/spike_concurrent_dispatch{,_submodule}.py`.

`tests/repros/` holds minimal standalone repros for kernel defects.

GPU OOM mid-test usually means contention, not a bug. Check
`nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`
before debugging -- a sibling process likely holds the VRAM Triton
autotune needs.

**Before trusting a green result, ask what state would turn it red --
then check that you actually created that state.** A check that passes
because the condition it guards was never reproduced reads as coverage
while asserting nothing. The recurring defect class in this repo is not
wrong kernels; it is instruments that structurally could not have
detected what they were trusted to detect. The mechanism is the same
every time: the thing under test is not a pure function of its declared
inputs, but gets measured as though it were. Four instances, each caught
only after the fact:

- **Prior launch's shared memory.** The v0.7.2 `kNoFill` defect does not
  fire in a fresh process -- the unfixed kernel passes its own regression
  sweep at mean_rtol 0.0345. It needs an earlier launch to have left
  non-finite values in the smem bank, so
  `tests/test_short_seq_tail.py` dirties with `+inf` before every sweep.
  Written without that step the file is green against a kernel already
  proven broken.
- **Prior arm's allocator state.** The peak-HBM rule below.
- **Neighboring modules' L2 footprint.** v0.6 sage_ffn benched faster in
  isolation and came back slower e2e (CHANGELOG v0.6.0). The synthetic
  harness
  could not see cache contention by construction, not by oversight.
- **Position in time within a render.** The Sol-Attn quality gate compared
  four still frames per arm and signed off; the failure it missed is a
  small object losing its identity mid-clip, which no still can show. The
  instrument was fine, the time sampling was wrong.

Two corollaries worth applying by default. **Measure the config that
ships:** `tests/test_sageattn_consume.py` records its peak with
`smooth_k=False` while production defaults to True (`core.py:948`), so
those numbers describe a configuration no consumer runs. And **prefer a
paired comparand to a bare sweep:** an exact-multiple shape measured
beside a ragged one attributes a delta to the tail, whereas a sweep in
which every shape is ragged smears the same effect across all rows with
nothing to attribute it against.

**Peak-HBM cumulative measurement benches: do NOT precede the
cumulative arm with a per-call-reset arm that does
`gc.collect`/`empty_cache` between calls.** The reset arm trains the
pytorch caching allocator into a state that biases the cumulative
number downward. Caught 2026-05-13 by /simplify in
`tests/bench/partitioned_mask_phase0/`; the pre-fix bench
underreported the K-quant+V-cast redundancy delta by a wide margin and
nearly shipped wrong numbers to a downstream consumer. Symptom:
cumulative-with-mask and cumulative-no-mask measurements that look
identical at a shape where they shouldn't.

## A guard with a failure state: key it on the build, not on a constant (2026-09-13)

Worked example from v0.7.17. The fused CUDA quant kernels wrapped their
global offsets at 2**32 elements; the fix widened the strides to int64. A
host-side guard was asked for alongside, "refuse rather than wrap
silently". Written the obvious way -- compare the tensor's largest offset
against `2**32` -- the guard is wrong on the new build (it refuses shapes
the kernels now handle) and redundant on the old one only if someone
remembers to install it there. Written against `2**64` it can never fire,
which is the catalogue's defect: an instrument with no state that turns it
red.

The version that ships keys on a fact the build itself asserts.
`csrc/fused/pybind.cpp` stamps `ELEMENT_OFFSET_BITS = 64` on the
extension; `sageattention/quant.py` reads it with a default of 32 for any
build that lacks the attribute, and every wrapper bounds its launch against
`2**bits`. Now the guard has a real red state: a `.so` from before the
change, which this checkout actually contains (the interpreter-scoped
`clean` in `build.sh` leaves the other interpreter's tag in place). The
test creates that state by patching the width to 32 on a meta tensor and
checks that the refusal arrives before the extension's own device check
would have, so it also pins *where* the guard sits. The heavy case then
drives the real kernel past the boundary, because a correct bound over a
still-wrong kernel is the other half of the catalogue.

Two readings of one number had to be kept apart to write the boundary
case: the exact crossing (the last row the kernel dereferences) and the
padded one (the grid it forms pointers for, up to one block wider). The
guard uses the padded bound and so fires up to 63 rows early; the record
carries the exact row. A test asserting the two are equal fails
correctly, and did.
