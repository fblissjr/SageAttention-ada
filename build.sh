#!/usr/bin/env bash
# Build sage-fork for the architectures needed on Ada (RTX 40xx).
#
# This fork's setup.py detects sm89 as a valid target for the SM80 extension
# (upstream gated on 8.0/8.6/8.7 only, which skipped _qattn_sm80 on Ada-only
# boxes and broke `sageattn_qk_int8_pv_fp16_cuda`). We still pass 8.0 in
# TORCH_CUDA_ARCH_LIST so nvcc actually produces the SM80 binary; setup.py
# decides *whether* to add the extension, the arch list decides *what*
# binaries it contains.
#
# Usage:
#   ./build.sh              # build for Ada + Ampere backward compat (default)
#   ./build.sh clean        # remove prior build artifacts first
#   ./build.sh verify       # verify a previous build without rebuilding
#
# Env overrides:
#   CUDA_ARCHES  override the default arch list (e.g. CUDA_ARCHES="8.0;8.9")
#   MAX_JOBS     override build parallelism (default: auto)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default: Ampere + Ada. Compiles _qattn_sm80 (via "8.0") and _qattn_sm89
# (via "8.9"). SM80 kernel runs on Ada via backward compat, SM89 kernel
# uses Ada's native fp8 tensor cores.
: "${CUDA_ARCHES:=8.0;8.6;8.9}"

ACTION="${1:-build}"

case "${ACTION}" in
    clean)
        echo "==> Cleaning prior build artifacts"
        rm -rf build/ dist/ sageattention.egg-info/ sageattention/*.so
        ACTION="build"
        ;;
    verify)
        ;;
    build)
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        echo "Usage: $0 [build|clean|verify]" >&2
        exit 1
        ;;
esac

# --- Pre-flight checks ---
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. CUDA toolkit is required to build." >&2
    exit 1
fi

# Sage must install into whichever venv your ComfyUI uses. If VIRTUAL_ENV
# isn't set, `uv pip install` would install into uv's default project env
# (or system python) and ComfyUI would never see the new .so files.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: VIRTUAL_ENV is not set." >&2
    echo "Activate the venv sage should install into first, e.g.:" >&2
    echo "    source /path/to/your/venv/bin/activate" >&2
    exit 1
fi

CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "==> Detected CUDA:   ${CUDA_VER}"
echo "==> Target archs:    ${CUDA_ARCHES}"
echo "==> Target venv:     ${VIRTUAL_ENV}"

# setup.py asserts CUDA >= 12.4 for 8.9. We don't replicate that here --
# let setup.py fail with its own message if the toolkit is too old.

if [[ "${ACTION}" == "build" ]]; then
    # Use the project's venv if one is active; otherwise uv will pick the
    # system interpreter. We force --no-deps so we don't pull random torch
    # versions over ComfyUI's installed one.
    echo "==> Building sage-fork with TORCH_CUDA_ARCH_LIST=${CUDA_ARCHES}"
    echo "    This takes 10-30 minutes on a multi-core box. First build is"
    echo "    the slowest; incremental rebuilds are much faster."
    echo ""

    # nvcc peaks at several GB per parallel job on the _qattn_sm89 kernel.
    # Cap the default at 8 so high-core boxes don't OOM; override with
    # MAX_JOBS=N if you know your memory headroom.
    _AUTO_JOBS=$(nproc)
    (( _AUTO_JOBS > 8 )) && _AUTO_JOBS=8
    MAX_JOBS="${MAX_JOBS:-${_AUTO_JOBS}}"
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHES}"
    export MAX_JOBS

    # Editable install so the dev checkout stays live.
    # --no-deps         : don't shadow the active torch / triton installs
    # --no-build-isolation : reuse the existing venv's torch for the build
    #                        instead of installing a fresh torch into an
    #                        isolated build env (would pull a different
    #                        torch version and double CPU time).
    # --python          : pin to the active venv explicitly so uv doesn't
    #                        try to manage a sage-fork-local .venv (uv's
    #                        default on a project dir with pyproject.toml).
    uv pip install --python "${VIRTUAL_ENV}/bin/python" -e . \
        --no-deps --no-build-isolation --force-reinstall
fi

# --- Post-build verification ---
echo ""
echo "==> Verifying extensions are importable in ${VIRTUAL_ENV}"

"${VIRTUAL_ENV}/bin/python" - <<'PY'
import importlib
import sys

expected = {
    "_qattn_sm80": "SM80 (Ampere + Ada backward-compat; powers fp16_cuda)",
    "_qattn_sm89": "SM89 (Ada native fp8; powers fp8_cuda variants)",
    "_fused":      "fused ops (always built)",
}

missing = []
for name, desc in expected.items():
    try:
        importlib.import_module(f"sageattention.{name}")
        print(f"  [OK]  {name:<18}  {desc}")
    except ImportError:
        missing.append(name)
        print(f"  [--]  {name:<18}  {desc}  (not compiled)")

# Exit non-zero only if NO CUDA extensions built at all.
if "_fused" in missing or ("_qattn_sm80" in missing and "_qattn_sm89" in missing):
    print("\nERROR: critical extensions missing.", file=sys.stderr)
    sys.exit(1)

print()
print("Available kernels for your build:")
if "_qattn_sm80" not in missing:
    print("  sageattn_qk_int8_pv_fp16_cuda       (INT8 QK + FP16 PV, fp32 accum)")
if "_qattn_sm89" not in missing:
    print("  sageattn_qk_int8_pv_fp8_cuda        (INT8 QK + FP8 PV,  fp32+fp32 accum)")
    print("  sageattn_qk_int8_pv_fp8_cuda++      (INT8 QK + FP8 PV,  fp32+fp16 accum)")
print("  sageattn_qk_int8_pv_fp16_triton     (JIT Triton; always available)")
PY

echo ""
echo "==> Build verification complete."
echo ""
echo "Recommended next step:"
echo "  If you have a consumer ComfyUI node that uses sage, restart"
echo "  ComfyUI to pick up the freshly-built extensions."
