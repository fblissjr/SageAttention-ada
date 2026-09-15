#!/usr/bin/env bash
# One-command sage-fork validation:
#   - snapshot the env to internal/bench_env_<today>.txt
#   - run the H3 gate (the pass criterion); the LTX bench only with RUN_LTX=1
#   - run the torch.compile spike
#   - archive both logs under internal/log/
#
# Usage:
#   source /path/to/venv/bin/activate && ./tests/run_all.sh
#   VENV=/path/to/.venv ./tests/run_all.sh    # explicit override

set -euo pipefail

# Resolve the venv. Prefer explicit $VENV, fall back to $VIRTUAL_ENV.
# No further fallback: pinning a default path here would either leak a
# private path into a public repo or pick the wrong venv silently.
if [ -n "${VENV:-}" ]; then
    VENV_DIR="${VENV}"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV_DIR="${VIRTUAL_ENV}"
else
    echo "error: no venv found. Either activate a venv or set \$VENV explicitly." >&2
    exit 1
fi

PY="${VENV_DIR}/bin/python"
UV="${VENV_DIR}/bin/uv"
if [ ! -x "${PY}" ]; then
    echo "error: ${PY} not executable" >&2
    exit 1
fi

# All commands run from the repo root regardless of where this is invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p internal/log

DATE="$(date +%F)"
ENV_FILE="internal/bench_env_${DATE}.txt"
BENCH_LOG="internal/log/test_sageattn_ltx_shapes_${DATE}.log"
IMAGE_LOG="internal/log/test_sageattn_image_shapes_${DATE}.log"
H3_LOG="internal/log/test_sageattn_h3_shapes_${DATE}.log"
SPIKE_LOG="internal/log/spike_torch_compile_${DATE}.log"

echo "== venv:        ${VENV_DIR}"
echo "== repo root:   ${REPO_ROOT}"
echo "== logs to:     ${ENV_FILE}, ${BENCH_LOG}, ${H3_LOG}, ${IMAGE_LOG}, ${SPIKE_LOG}"
echo

# 1. env snapshot. uv pip freeze (since uv venvs lack a pip module).
echo "[1/6] snapshotting env -> ${ENV_FILE}"
{
    echo "# bench env snapshot, captured $(date -Iseconds)"
    echo "# venv: replaced for privacy"
    echo
    if [ -x "${UV}" ]; then
        # The editable sage install lists as "-e file:///..." rather than
        # "sageattention==", so an anchored name match silently dropped the
        # one package these numbers are actually about. Match the editable
        # line too, and strip the path so the snapshot stays repo-relative.
        VIRTUAL_ENV="${VENV_DIR}" "${UV}" pip freeze 2>/dev/null \
            | grep -iE "^(torch|triton|sageattention|flashinfer|spas)|^-e .*sage" \
            | sed 's|^-e file://.*|sageattention @ editable (this repo)|' \
            | sort
    else
        "${PY}" -c "
import importlib
for m in ['torch', 'triton', 'sageattention']:
    try:
        mod = importlib.import_module(m)
        print(f'{m}=={getattr(mod, \"__version__\", \"?\")}')
    except ImportError:
        print(f'{m}: NOT INSTALLED')
"
    fi
} > "${ENV_FILE}"

# 2. H3-shape bench (packed AV self-attn, 56 heads, d=128). THE gate.
# H3 is the only current optimization target, so this run decides whether the
# suite passed. It gates speed, peak VRAM and cross-kernel fidelity -- but NOT
# rtol against SDPA, which is not a measurement at H3 config on synthetic
# input. See the bench's module docstring.
echo "[2/6] running tests/test_sageattn_h3_shapes.py --check-regression"
set +e
"${PY}" tests/test_sageattn_h3_shapes.py --check-regression 2>&1 | tee "${H3_LOG}"
H3_EXIT="${PIPESTATUS[0]}"
set -e
if [ "${H3_EXIT}" -ne 0 ]; then
    echo "error: H3 bench exited ${H3_EXIT} (regression detected). See ${H3_LOG}." >&2
    exit "${H3_EXIT}"
fi

# 3. LTX-shape bench. OFF by default as of 2026-09-14 (owner's call: LTX is
# not a workload here and its GPU minutes are not worth spending on every
# run). It went non-blocking on 2026-09-08 for the same reason (CHANGELOG
# v0.7.12). Opt in with RUN_LTX=1 when a second head config or a masked path
# is the question; the file and its baselines are untouched.
echo
if [ "${RUN_LTX:-0}" = "1" ]; then
    echo "[3/6] running tests/test_sageattn_ltx_shapes.py --check-regression (non-blocking, RUN_LTX=1)"
    set +e
    "${PY}" tests/test_sageattn_ltx_shapes.py --check-regression 2>&1 | tee "${BENCH_LOG}"
    LTX_EXIT="${PIPESTATUS[0]}"
    set -e
    if [ "${LTX_EXIT}" -ne 0 ]; then
        echo "WARNING: LTX bench exited ${LTX_EXIT}. Not a current target, so it does" >&2
        echo "         not fail the suite -- but read ${BENCH_LOG} before assuming it" >&2
        echo "         is only LTX: these kernels are shared with the H3 path." >&2
    fi
else
    echo "[3/6] LTX bench skipped (not a workload; RUN_LTX=1 to run it)"
fi

# 4. Image-shape bench (head_dim ∈ {120, 128}). Separate file so the
# LTX file stays focused; both reuse the same dispatch helpers.
echo
echo "[4/6] running tests/test_sageattn_image_shapes.py"
"${PY}" tests/test_sageattn_image_shapes.py 2>&1 | tee "${IMAGE_LOG}"

# 5. Correctness suites that gate on an assertion rather than a number.
# These are fast and have no baseline to drift, so they run unconditionally
# and fail the whole script -- unlike the benches above, a failure here is
# a defect, not a measurement.
echo
echo "[5/6] running correctness suites"
# test_regression_check guards check_regressions itself -- the code that
# grades every other bench in this script. It was absent here until
# 2026-09-08, so the gate's own logic was the one thing the runner never
# checked. Pure-python and fast; no reason for it to have been outside.
for t in test_quant_offset_overflow test_sageattn_consume test_dispatched_kernel_telemetry test_regression_check \
         test_build_info_contract test_upstream_contracts test_qk_balance; do
    echo "  - tests/${t}.py"
    "${PY}" "tests/${t}.py" > "internal/log/${t}_${DATE}.log" 2>&1 || {
        echo "error: tests/${t}.py failed. See internal/log/${t}_${DATE}.log" >&2
        tail -20 "internal/log/${t}_${DATE}.log" >&2
        exit 1
    }
done

# 6. torch.compile spike.
echo
echo "[6/6] running tests/spike_torch_compile.py"
"${PY}" tests/spike_torch_compile.py 2>&1 | tee "${SPIKE_LOG}"

echo
echo "done. summary:"
echo "  env:     ${ENV_FILE}"
echo "  ltx:     ${BENCH_LOG}  ($(grep -c '^===' "${BENCH_LOG}" || echo 0) shapes)"
echo "  h3:      ${H3_LOG}  ($(grep -c '^===' "${H3_LOG}" || echo 0) shapes)"
echo "  image:   ${IMAGE_LOG}  ($(grep -c '^===' "${IMAGE_LOG}" || echo 0) shapes)"
echo "  spike:   ${SPIKE_LOG}  ($(grep -E '^(verdict|final)' "${SPIKE_LOG}" | tail -1))"
