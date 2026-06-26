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

# --- CUDA toolkit selection ---
# nvcc 13.3 ships a cudafe++ front-end regression that miscompiles PyTorch's
# bundled headers: every .cu fails in ATen/core/List_inl.h with a spurious
# "need 'typename' before ...::difference_type" error, even though the source
# is valid (proven by a same-TU A/B -- 13.3 fails, 13.2 compiles clean; the
# host g++ and sage's own sources are fine, it is purely the nvcc version).
# So if the *active* toolkit is a known-broken version, switch to the newest
# installed toolkit that is NOT in the broken set. This overrides even a
# pre-exported CUDA_HOME -- a global CUDA_HOME=/usr/local/cuda is common and
# usually points at the default/latest (broken) toolkit, so it can't be
# trusted as an intentional pin. The resulting .so runs fine on a newer
# driver (drivers are backward-compatible). Drop a version from
# KNOWN_BAD_CUDA once a fixed nvcc for it ships.
# Overrides: pick a good toolkit with `CUDA_HOME=/usr/local/cuda-X.Y
# ./build.sh`; force the broken one anyway with `SAGE_SKIP_CUDA_GUARD=1`.
KNOWN_BAD_CUDA=" 13.3 "

_nvcc_ver() { "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1; }

# Honor an explicit CUDA_HOME by putting its nvcc first, so the version we
# detect reflects the toolkit that would actually compile.
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

CUDA_VER=$(_nvcc_ver "$(command -v nvcc)")

if [[ "${KNOWN_BAD_CUDA}" == *" ${CUDA_VER} "* && "${SAGE_SKIP_CUDA_GUARD:-0}" != "1" ]]; then
    _good="" _good_ver=""
    for _d in $(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -rV); do
        _v=$(_nvcc_ver "${_d}bin/nvcc")
        [[ -z "${_v}" || "${KNOWN_BAD_CUDA}" == *" ${_v} "* ]] && continue
        _good="${_d%/}" _good_ver="${_v}"; break
    done
    if [[ -n "${_good}" ]]; then
        echo "==> WARNING: nvcc ${CUDA_VER} miscompiles PyTorch headers (cudafe++ regression)."
        echo "    Auto-switching the build to ${_good}."
        echo "    Override: CUDA_HOME=/usr/local/cuda-X.Y (pick another) or"
        echo "    SAGE_SKIP_CUDA_GUARD=1 (force ${CUDA_VER} anyway)."
        export CUDA_HOME="${_good}"
        export PATH="${CUDA_HOME}/bin:${PATH}"
        CUDA_VER="${_good_ver}"
    else
        echo "ERROR: active nvcc is ${CUDA_VER}, which miscompiles PyTorch headers," >&2
        echo "       and no working alternative toolkit was found under /usr/local/cuda-*." >&2
        echo "       Install a known-good CUDA toolkit (e.g. 13.2), set CUDA_HOME to one," >&2
        echo "       or set SAGE_SKIP_CUDA_GUARD=1 to force the broken toolkit." >&2
        exit 1
    fi
fi

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
