#!/usr/bin/env bash
# One-command sage-fork validation:
#   - snapshot the env to internal/bench_env_<today>.txt
#   - run the LTX-shape accuracy + speed bench
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
SPIKE_LOG="internal/log/spike_torch_compile_${DATE}.log"

echo "== venv:        ${VENV_DIR}"
echo "== repo root:   ${REPO_ROOT}"
echo "== logs to:     ${ENV_FILE}, ${BENCH_LOG}, ${SPIKE_LOG}"
echo

# 1. env snapshot. uv pip freeze (since uv venvs lack a pip module).
echo "[1/3] snapshotting env -> ${ENV_FILE}"
{
    echo "# bench env snapshot, captured $(date -Iseconds)"
    echo "# venv: replaced for privacy"
    echo
    if [ -x "${UV}" ]; then
        VIRTUAL_ENV="${VENV_DIR}" "${UV}" pip freeze 2>/dev/null \
            | grep -iE "^(torch|triton|sageattention|flashinfer|spas)" \
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

# 2. LTX-shape bench.
echo "[2/3] running tests/test_sageattn_ltx_shapes.py"
"${PY}" tests/test_sageattn_ltx_shapes.py 2>&1 | tee "${BENCH_LOG}"

# 3. torch.compile spike.
echo
echo "[3/3] running tests/spike_torch_compile.py"
"${PY}" tests/spike_torch_compile.py 2>&1 | tee "${SPIKE_LOG}"

echo
echo "done. summary:"
echo "  env:     ${ENV_FILE}"
echo "  bench:   ${BENCH_LOG}  ($(grep -c '^===' "${BENCH_LOG}" || echo 0) shapes)"
echo "  spike:   ${SPIKE_LOG}  ($(grep -E '^(verdict|final)' "${SPIKE_LOG}" | tail -1))"
