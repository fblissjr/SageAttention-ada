# sage-fork

Routing index. This file holds **scope, commands, and rules that fit on
one line**. Everything else -- evidence, worked examples, measurements,
history -- lives in `docs/` (committed) or `internal/` (gitignored) and
is linked below. If a rule needs a paragraph to justify it, the rule
belongs here and the paragraph belongs in `docs/`.

Rewritten 2026-09-08 from a 1000-line version. If you are looking for
something that used to be here, it is in `docs/` and named in the map at
the bottom; `git log -p CLAUDE.md` shows exactly what moved where.

## Scope

**MiniMax H3 is the only current target. sm89 / Ada / 4090 only.**

- **LTX 2.3 is parked** (2026-09-08) **and its bench is off by default**
  (2026-09-14, owner's call). Nothing is coded, tested or benched *for*
  it; `tests/run_all.sh` skips it unless `RUN_LTX=1`. The file and its
  baselines stay because the kernels are shared and it covers a second
  head config plus a masked path H3 cannot reach; run it by hand when that
  is the question.
- **The v0.5.5 native-mask kernel is zero priority** -- LTX-motivated,
  and H3 never passes a mask.
- **`sage_ffn` and the FFN line stay zero priority.** On the path that
  ships, attention is the majority of a DiT block at every length
  measured and the MLP never exceeds it; the shares and their conditions
  are in `docs/h3_workload_profile.md`. Attention scales O(S^2) against
  O(S) for everything else, so its share rises with clip length -- never
  quote it without a sequence length. Separately, and not a contradiction:
  the MLP holds the largest single memory transient at every length
  measured, which is a question about allocation rather than time.
- Flux / Z-Image are bench shapes, not targets.
- Other archs fall back to the sm89 kernel and are not tested. No
  Hopper/Blackwell kernels (removed v0.5.0). Linux + source build only.
- **Anything dated before 2026-08-04 describes LTX or Z-Image**, not H3.
  It can corroborate a pattern; it cannot confirm an H3 result.

H3 shape: one attention call site over a packed
`[text | refs | audio | video]` sequence, `mask=None` hardcoded
(`comfy/ldm/minimax/model.py:199`, re-checked 2026-09-08), no
cross-attention.

Mission and forward directions: `VISION.md`, `docs/roadmap.md`.

## Build

Always an active venv. Never bare `python`. `python -m pip freeze` fails
on uv venvs -- use `VIRTUAL_ENV=<venv> <venv>/bin/uv pip freeze`.

```bash
source /path/to/venv/bin/activate
./build.sh              # build + editable install into $VIRTUAL_ENV
./build.sh clean        # wipe THIS interpreter's artifacts, then build
./build.sh clean-all    # wipe every interpreter's artifacts, then build
./build.sh verify       # import-check only
```

- **Use `./build.sh`, never `uv pip install -e . -U`** -- the bare form
  silently upgrades torch over ComfyUI's pinned build.
- **The `.so` is ABI-bound to torch.** Upgrade torch, rebuild sage.
- **C++20 is a hard floor**, set by torch's headers. A checkout from
  before v0.7.8 cannot build against torch >= 2.14 at all; the symptom
  is every file failing identically inside `torch/extension.h`.
- **Any CUDA toolkit your torch accepts.** The `KNOWN_BAD_CUDA` guard is
  present but empty; 13.2 and 13.3 were measured bit-identical. Re-arm
  it if a future nvcc breaks the torch headers.
- **Several venvs can share this checkout.** Extensions are tagged per
  interpreter, not abi3, so plain `clean` removes only the tag it is
  rebuilding. `clean-all` then needs a plain `./build.sh` per other venv,
  or they fail at `from . import _fused` with nothing else changed.
- Restart ComfyUI after a build.
- Not ours but shares the venv: **torchaudio raises where torch only
  warns** on a CUDA minor mismatch, and takes ComfyUI's startup with it.

## Test

```bash
./tests/run_all.sh                     # env snapshot + h3 gate + image + correctness + spike
VENV=/path/to/venv ./tests/run_all.sh  # explicit venv
RUN_LTX=1 ./tests/run_all.sh           # also the parked LTX bench (off by default)
${VIRTUAL_ENV}/bin/python tests/test_sageattn_h3_shapes.py --check-regression
```

- **`tests/test_sageattn_h3_shapes.py` is the gate.** It decides whether
  the suite passed. The LTX bench is skipped unless `RUN_LTX=1`.
- **The H3 gate covers speed, peak VRAM and cross-kernel fidelity. Not
  accuracy** -- deliberately. Its baselines carry no rtol-vs-SDPA rows,
  so the shared gate skips that check by construction.
- **A synthetic-input accuracy number is not a measurement of anything
  we ship.** `torch.randn` has no channel offset and no attention
  structure, so softmax is near-uniform, the output is a near-cancelling
  average, and element-wise error is dominated by cancellation. At H3 it
  reads ~4x worse than reality. Speed, VRAM and fidelity are unaffected
  and synthetic input is correct for those.
- **H3 accuracy lives in `tests/spikes/spike_h3_real_activations.py`**,
  on captured q/k/v. Run it rather than adding synthetic rows.
- **First `--check-regression` after a build is expected to fail** on
  triton-autotune-pending rows. Run once without the flag first.
- GPU OOM mid-test usually means contention. Check `nvidia-smi` before
  debugging.
- The suite imports `orjson`, which ComfyUI's venv does not ship. A fresh
  venv fails at import in the LTX helper; `uv pip install orjson` there.

## Rules

- Python: **always uv**. Never `pip`, never bare `python3`.
- JSON: **orjson**, never stdlib `json`.
- **No emojis** in any file or output.
- Comments: only non-obvious WHY.
- **`main` is what the ComfyUI server runs.** The venv's editable install
  serves whatever this tree holds at the next server start, so only
  measured, changelog-recorded changes land on `main`; unmeasured work
  goes on a branch, and the tree is back on `main` before a session ends.
  Tag each build the server has run (`served/<date>`, provenance in the
  tag message) so a render record names a commit. Why: `docs/conventions.md`.
- **Never push without being asked.** `gh pr create` defaults `--base` to
  the upstream parent -- pin `--repo`/`--base`/`--head` or it opens a
  public PR against thu-ml.
- **Label the model on every measurement, claim and doc.**
- **Prose never carries a measurement.** Point at the script, the
  constant, or the dated record that holds it.
- **Before trusting a green result, ask what state would turn it red,
  then check you created that state.** This repo's recurring defect is
  instruments that could not have detected what they were trusted to
  detect. Catalogue: `docs/testing_practices.md`.
- **A mutation that perturbs its own oracle proves nothing.** If breaking
  the thing under test also moves what it is compared against, a green
  result is uninformative.
- **Perf-mechanism claims need both arms measured.** A number can come
  from one measurement; a mechanism claim cannot.
- **Gate ship-decisions on in-pipeline A/B** when synthetic-bench cannot
  measure the dominant cost (L2 contention, dispatch overhead,
  fragmentation, sustained-clock state). Before the ship commit, not
  after.
- **Never compare an approximate kernel's accuracy to ours without
  checking what its reference computes.** Check the referent before the
  metric.
- **Retract wrong framing via `git revert`, not in-place edit**, so the
  audit trail stays reconstructible.
- **Consumer-agnostic framing in committed material.** Model class is
  fine; a specific custom node by name is not.
- **No task/caller refs or project-internal phase numbers in committed
  code.** They decay; cite the substantive reason instead.
- **Path discipline.** Every committed path is repo-relative; the
  `path-privacy` hooks hard-block leaks and are not to be bypassed.
- **Session logs append, never overwrite** (`internal/log/log_<date>.md`).
- **Local-machine config in `internal/local_config.json`** (gitignored).
  CLI arg > env var > that file > hard error.
- **`coderef/` is gitignored** and holds consumer source trees. Use it to
  verify claims about upstream rather than trusting a note.
- **Re-check a claim about a fast-moving dependency before citing it**,
  and record what would make it stale. Six claims in this repo went false
  without anything failing; see `docs/drift_audit_and_directions.md`.

## Two contracts that break silently

Loudly-breaking contracts do not need documenting. These two do.

1. **`attn_mask` must stay a named parameter of `sageattn()`.** ComfyUI
   gates masked calls on `"attn_mask" in inspect.signature(...)`; when
   that reads False it routes every masked call to torch SDPA, which
   stranded the v0.5.5 kernel until v0.7.0. Pinned by
   `tests/test_dispatched_kernel_telemetry.py`.
2. **`build_info()`'s key set is embedded in a consumer's dated
   records.** Adding a key is compatible; removing, renaming or retyping
   one is not, and `revision` is pinned at 12 characters. Pinned by
   `tests/test_build_info_contract.py`.

Everything else: `docs/consumer_surface.md`, and
`docs/downstream_symbols.md` for the underscore surface and the
pre-removal checklist.

## Where things are

**Code.** `sageattention/core.py` is `sageattn()` dispatch (sm89 +
CUDA >= 12.8 lands on `sageattn_qk_int8_pv_fp8_cuda`,
`pv_accum_dtype="fp32+fp16"`). `csrc/qattn/` holds the sm80 and sm89
kernel sets, `sageattention/triton/` the JIT kernels,
`sageattention/comfyui_compat.py` the fp8 storage-convention shim,
`setup.py` the build (our patch adds sm89 to the SM80 gate).

**Docs, by the question you are asking:**

| question | file |
|---|---|
| What did we measure on H3, under what conditions? | `docs/h3_kernel_measurements.md` |
| What actually runs on H3, and what reaches sage? | `docs/h3_attention_stack.md` |
| How do I write a measurement here without fooling myself? | `docs/testing_practices.md` |
| Why is that rule a rule? | `docs/conventions.md` |
| What do consumers import, and what breaks if I change it? | `docs/consumer_surface.md`, `docs/downstream_symbols.md` |
| How do I read a perf result? | `docs/perf_research_framework.md` |
| Can I change a bench shape or a baseline? | `docs/bench_discipline.md` |
| Which claims went stale, and what should we do next? | `docs/drift_audit_and_directions.md` |
| Is this approximate-attention setting acceptable? | `docs/sparse_attention_quality_gating.md` |
| How do I work against a dependency that moves hourly? | `docs/moving_targets.md` |
| What is ours vs upstream? | `docs/whats_ours_vs_upstream.md` |
| A kernel defect is blocking a workflow | `docs/sage_bug_fix_workflow.md` |
| H3's two flow schedules under one sampler | `docs/minimax_h3_av_sampling.md` |
| Where does H3 block time and memory go? | `docs/h3_workload_profile.md` |
| Where does LTX wall-time go? (parked) | `docs/ltx_workload_profile.md` |
| Why not torch.compile? | `docs/torch_compile_spike.md` |
| Does fp16 accumulation change our output? (no) | `docs/fp16_matmul_accum.md`, `docs/fp16_accum_fp8_matmul.md` |
| Scope, mission, what we might be wrong about | `VISION.md` |
| Open triggers, closed decisions, known kernel bugs | `CHANGELOG.md` |

**Gitignored.** `internal/log/log_<date>.md` session narrative,
`internal/audit_<date>.md` durable findings, `internal/h3_sol_diary.md`
the H3 + sparse-attention narrative index (names third-party nodes, so it
cannot be committed), `internal/pyright_noise.md` known false positives,
`.claude.local.md` local paths.

`CHANGELOG.md` is the single source of truth for open triggers (Backlog)
and closed decisions (Decision log). There is no separate plan file --
`internal/PLAN.md` was retired for drifting against it.
