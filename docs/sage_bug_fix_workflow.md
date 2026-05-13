# If we ever need to fix a sage bug ourselves

Last updated: 2026-05-13

L3 reference for CLAUDE.md. Load this only when a kernel defect
blocks a real workflow.

We own this fork; there's no upstream to send PRs to anymore. If a
kernel defect blocks the LTX workflow:

1. Build a minimal repro in `tests/repros/<name>.py`.
2. Find the kernel in `csrc/qattn/`. sm80 = fp16 PV, sm89 = fp8 PV.
3. Mask-handling code is in the `pybind_sm*.cpp` files (PyTorch entry
   points) and the `.cu` files (kernel body).
4. Rebuild via `./build.sh` and re-run the repro.
5. Add a CHANGELOG entry under the latest version block (Fixed
   subsection) with the repro reference.

We deliberately have no CI. Verify by running the LTX-shape test and
the full downstream-consumer pytest suite on this box before trusting
a change.
