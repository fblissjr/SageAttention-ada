#!/usr/bin/env bash
# Build torchao from a local checkout against the active venv.
#
# Mirrors sage-fork's build.sh discipline: active-venv enforcement, sm89-
# only arch list to skip sm90+/sm100+ kernels we don't use on Ada, MAX_JOBS
# cap, editable install into $VIRTUAL_ENV.
#
# Why: torchao ships several CUDA extensions. The base `_C.abi3.so` carries
# generic kernels (some sm89-relevant); `_C_cutlass_90a.abi3.so` is sm90
# (Hopper) only; `_C_mxfp8.so` is sm100+ (Blackwell) only. The setup.py
# gates the latter two on `compute_90a` / `compute_100` being in the CUDA
# arch list -- restricting to `TORCH_CUDA_ARCH_LIST=8.9` naturally skips
# both, cutting build time from ~20-30 min down to ~5 min and avoiding
# the host-compiler issues that surface on sm90a sources.
#
# Source location: coderef/ao (expected to be a symlink to your local
# torchao checkout; e.g. ln -s /path/to/your/ao coderef/ao). The
# `coderef/` directory is gitignored; the symlink is per-user setup.
#
# Usage:
#   source /path/to/venv/bin/activate
#   ./build_torchao.sh              # full build + editable install
#   ./build_torchao.sh clean        # wipe prior .so / build/ artifacts
#   ./build_torchao.sh verify       # import-check only, no rebuild
#
# Env overrides:
#   TORCH_CUDA_ARCH_LIST   override the default "8.9"
#   MAX_JOBS               override build parallelism (default: 8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TORCHAO_DIR="${SCRIPT_DIR}/coderef/ao"

# Active venv enforcement -- same shape as build.sh
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

# Subcommand dispatch
CMD="${1:-build}"

verify_import() {
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
    print("addmm_float8_unwrapped_inference: ok (pure-Python, our bench's current entry point)")
except Exception as e:
    print(f"FAIL: addmm_float8_unwrapped_inference: {e}", file=sys.stderr)
    sys.exit(1)

# Probe the C extension separately. It's NOT load-bearing for our current
# bench (we only use the pure-Python wrapper), but if it fails to load,
# any future compiled-kernel use will silently fall through to Python-
# level fallbacks. Report status, don't fail the verify on _C alone.
try:
    import torchao._C  # noqa: F401
    print("_C extension: loaded")
except Exception as e:
    print(f"_C extension: not loaded ({type(e).__name__}: {str(e)[:120]})")
    print("  -> rebuild via ./build_torchao.sh to make compiled torchao ops available")
PY
}

case "${CMD}" in
    clean)
        if [[ ! -d "${TORCHAO_DIR}" ]]; then
            echo "Nothing to clean: ${TORCHAO_DIR} not found." >&2
            exit 0
        fi
        echo "Cleaning torchao build artifacts at ${TORCHAO_DIR}..."
        find "${TORCHAO_DIR}/torchao" -name '*.so' -delete 2>/dev/null || true
        rm -rf "${TORCHAO_DIR}/build" "${TORCHAO_DIR}/torchao.egg-info"
        echo "Clean done."
        ;;

    verify)
        verify_import
        ;;

    build)
        if [[ ! -d "${TORCHAO_DIR}" ]]; then
            echo "ERROR: torchao checkout not found at ${TORCHAO_DIR}" >&2
            echo "" >&2
            echo "Expected: a symlink or clone at coderef/ao -> your local torchao." >&2
            echo "  ln -s /path/to/your/ao ${TORCHAO_DIR}" >&2
            exit 1
        fi

        # sm89-only arch list. setup.py gates _C_cutlass_90a on compute_90a
        # and _C_mxfp8 on compute_100, so restricting to 8.9 skips both
        # automatically. Cuts build to ~5 min on an 8-core box.
        export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
        export MAX_JOBS="${MAX_JOBS:-8}"

        echo "Building torchao..."
        echo "  source:               ${TORCHAO_DIR}"
        echo "  venv:                 ${VIRTUAL_ENV}"
        echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
        echo "  MAX_JOBS:             ${MAX_JOBS}"
        echo ""

        # --no-build-isolation: use the venv's torch (the build needs it),
        # rather than re-resolving a build-env torch that may not match.
        (cd "${TORCHAO_DIR}" && "${UV}" pip install -e . --no-build-isolation)

        echo ""
        echo "Build done. Verifying..."
        verify_import
        ;;

    *)
        echo "Usage: $0 [build|clean|verify]" >&2
        exit 1
        ;;
esac
