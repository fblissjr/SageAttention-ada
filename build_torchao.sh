#!/usr/bin/env bash
# Update the editable install of torchao from a local checkout.
#
# Mirrors sage-fork's build.sh discipline: active-venv enforcement,
# subcommand dispatch (build/clean/verify). Source location: coderef/ao
# (expected to be a symlink or clone of your local torchao checkout;
# `coderef/` is gitignored, so the symlink is per-user setup).
#
# Why a script for what's basically `uv pip install -e .`:
# - Active-venv enforcement so we don't accidentally install into the
#   wrong Python environment.
# - --force-reinstall --no-deps to defeat uv's "already installed"
#   short-circuit when only the torchao git revision changed.
# - --no-build-isolation so the build uses the venv's torch directly
#   rather than re-resolving a build-environment torch.
# - Diagnostic verify covering both the Python entry points we use
#   and the optional compiled extensions, so the actual usable surface
#   is visible after install.
#
# What torchao 0.18 ships on sm89:
# - Pure-Python `addmm_float8_unwrapped_inference` (thin wrapper around
#   `torch._scaled_mm`). This is the bench's torchao comparand entry
#   point and works without any compiled C extension.
# - The compiled `torchao._C_cutlass_90a` and `torchao._C_mxfp8` modules
#   target sm90a (Hopper) and sm100 (Blackwell) respectively. Neither
#   loads on sm89; both are non-blocking for our use. We do NOT skip
#   their compile via TORCH_CUDA_ARCH_LIST because torchao's setup.py
#   auto-enables them on CUDA >= 12.6 regardless of arch list. They
#   take ~5 min of CPU to compile and produce binaries we never call.
# - The base `torchao._C` extension is not produced on torchao 0.18:
#   all candidate sources got refactored into the sm90a/sm100 sub-
#   extensions. Absent _C is the expected state.
#
# Usage:
#   source /path/to/venv/bin/activate
#   ./build_torchao.sh              # editable install + verify
#   ./build_torchao.sh clean        # wipe build artifacts, then reinstall
#   ./build_torchao.sh verify       # diagnostic check only, no install
#
# Env overrides:
#   MAX_JOBS               build parallelism for the (sm90a + sm100)
#                          compile that torchao insists on doing.
#                          Default: 8.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TORCHAO_DIR="${SCRIPT_DIR}/coderef/ao"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: \$VIRTUAL_ENV is not set." >&2
    echo "Activate the target venv first (source /path/to/venv/bin/activate)." >&2
    exit 1
fi

PYTHON="${VIRTUAL_ENV}/bin/python"
UV="${VIRTUAL_ENV}/bin/uv"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: ${PYTHON} not executable. Is the venv built?" >&2
    exit 1
fi

ACTION="${1:-build}"

verify_state() {
    "${PYTHON}" - <<'PY'
import sys
try:
    import torchao
    print(f"torchao {torchao.__version__} -> {torchao.__file__}")
except Exception as e:
    print(f"FAIL: torchao import: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from torchao.float8.inference import addmm_float8_unwrapped_inference  # noqa: F401
    print("addmm_float8_unwrapped_inference: ok (pure-Python; the bench's torchao entry point)")
except Exception as e:
    print(f"FAIL: addmm_float8_unwrapped_inference: {e}", file=sys.stderr)
    sys.exit(1)

# torchao 0.18 ships _C_cutlass_90a (sm90a) and _C_mxfp8 (sm100). Neither
# loads on sm89, which is expected and non-blocking -- our bench doesn't
# use either. Report status only, don't fail the verify.
for mod_name in ("torchao._C_cutlass_90a", "torchao._C_mxfp8"):
    try:
        __import__(mod_name)
        print(f"{mod_name}: loaded (sm90a/sm100 binary; not used by sage-fork bench)")
    except Exception as e:
        msg = str(e)[:120]
        print(f"{mod_name}: not loaded ({type(e).__name__}: {msg})")
PY
}

case "${ACTION}" in
    clean)
        if [[ -d "${TORCHAO_DIR}" ]]; then
            echo "==> Cleaning torchao build artifacts at ${TORCHAO_DIR}"
            find "${TORCHAO_DIR}/torchao" -name '*.so' -delete 2>/dev/null || true
            rm -rf "${TORCHAO_DIR}/build" "${TORCHAO_DIR}/torchao.egg-info"
        fi
        ACTION="build"
        ;;
    verify)
        verify_state
        exit 0
        ;;
    build)
        ;;
    *)
        echo "ERROR: unknown subcommand '${ACTION}'" >&2
        echo "Usage: $0 [build|clean|verify]" >&2
        exit 1
        ;;
esac

# --- Install flow (reached for `build` or post-`clean`) ---

if [[ ! -d "${TORCHAO_DIR}" ]]; then
    echo "ERROR: torchao checkout not found at ${TORCHAO_DIR}" >&2
    echo "" >&2
    echo "Expected: a symlink or clone at coderef/ao -> your local torchao." >&2
    echo "  ln -s /path/to/your/ao ${TORCHAO_DIR}" >&2
    exit 1
fi

export MAX_JOBS="${MAX_JOBS:-8}"

echo "==> Installing torchao (editable)"
echo "  source: ${TORCHAO_DIR}"
echo "  venv:   ${VIRTUAL_ENV}"
echo ""
echo "Note: torchao's setup.py compiles _C_cutlass_90a (sm90a) and"
echo "_C_mxfp8 (sm100) on CUDA >= 12.6 regardless of TORCH_CUDA_ARCH_LIST."
echo "Neither is used on sm89; the compile takes ~5 min and we tolerate it"
echo "as the cost of getting the Python entry points refreshed."
echo ""

(cd "${TORCHAO_DIR}" && "${UV}" pip install -e . \
    --no-build-isolation --force-reinstall --no-deps)

echo ""
echo "==> Verifying"
verify_state
