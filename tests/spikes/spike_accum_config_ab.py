"""A/B: comfy-kitchen's fp32 PV-accum config vs our fp32+fp16 (2++) default.

Substantiates the "2++ is the Ada speed win" claim from the comfy-kitchen
PR #42 comparison (internal/comfy_kitchen_pr42_comparison.md) on the actual
4090, isolating the single variable that differs: the PV accumulation config.

comfy-kitchen #42 instantiates qk_int_sv_f8_attn_kernel with
DTypeSVAccum=float, use_inst_buffer=false, use_pv_fp16_accu=false,
fuse_v_scale=true. That maps EXACTLY to our
pv_accum_dtype="fp32" -> qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.
Our dispatcher default on sm89 is pv_accum_dtype="fp32+fp16" (2++).

Same kernel template, same quant inputs, same shape -- only the accum
config changes. So any speed/rtol delta here is attributable to the
accum choice, not to binding/quant/packaging differences. This is a
cleaner isolation than building comfy-kitchen (which would confound the
accum delta with their different quant kernels + nanobind boundary).

Reuses the load-bearing shape + accuracy_metrics + time_and_vram from
tests/test_sageattn_ltx_shapes.py so the numbers are comparable to the
canonical bench.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from tests.test_sageattn_ltx_shapes import (
    SHAPES,
    accuracy_metrics,
    make_qkv,
    sdpa_reference,
    time_and_vram,
)

# The single load-bearing row from CLAUDE.md / perf framework.
LOAD_BEARING = "ltx23_video_self_attn_init_22932"

# (label, pv_accum_dtype). "fp32" is the comfy-kitchen #42 comparand;
# "fp32+fp16" is our sm89 dispatcher default (SageAttention2++).
CONFIGS = [
    ("fp32 (comfy-kitchen #42 config, SA2)", "fp32"),
    ("fp32+fp32 (inst-buffer variant)",      "fp32+fp32"),
    ("fp32+fp16 (our default, SA2++)",       "fp32+fp16"),
]

DTYPE = torch.bfloat16  # LTX runs bf16


def main() -> None:
    import sageattention as sa

    cap = torch.cuda.get_device_capability()
    assert cap >= (8, 9), f"spike targets sm89; got {cap}"
    print(f"device={torch.cuda.get_device_name()} cap={cap[0]}.{cap[1]}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}\n")

    shape = next(s for s in SHAPES if s.name == LOAD_BEARING)
    print(
        f"shape={shape.name}  B={shape.batch} H={shape.heads} "
        f"N={shape.seq_q} D={shape.head_dim} causal={getattr(shape, 'is_causal', False)} "
        f"dtype={DTYPE}\n"
    )

    q, k, v = make_qkv(shape, DTYPE)

    # Reference: torch efficient-attention (the canonical bench reference;
    # MATH OOMs at this scale). No mask on the self-attn init row.
    with torch.inference_mode():
        ref = sdpa_reference(q, k, v, None)

    results = []
    with torch.inference_mode():
        for label, pv in CONFIGS:
            def call(pv=pv):
                return sa.sageattn_qk_int8_pv_fp8_cuda(
                    q, k, v, is_causal=False, tensor_layout="HND",
                    pv_accum_dtype=pv,
                )

            # Accuracy on a single fresh call.
            out = call()
            mean_r, max_r = accuracy_metrics(out, ref)[:2]

            # Timing (median of 3, 1 warmup -- matches the canonical bench).
            with torch.cuda.nvtx.range(f"accum_ab/{pv}"):
                median_ms, peak_mib = time_and_vram(call, warmup=1, runs=3)

            results.append((label, pv, median_ms, peak_mib, mean_r, max_r))

    base_ms = next(ms for (_, pv, ms, *_ ) in results if pv == "fp32")

    print(f"{'config':<40} {'median_ms':>10} {'vs fp32':>9} {'peak_MiB':>9} "
          f"{'mean_rtol':>10} {'max_rtol':>9}")
    print("-" * 92)
    for label, pv, ms, mib, mean_r, max_r in results:
        speed = base_ms / ms
        print(f"{label:<40} {ms:>10.3f} {speed:>8.2f}x {mib:>9.0f} "
              f"{mean_r:>10.4f} {max_r:>9.3f}")

    print()
    pp_ms = next(ms for (_, pv, ms, *_ ) in results if pv == "fp32+fp16")
    speedup = base_ms / pp_ms
    delta_pct = (base_ms - pp_ms) / base_ms * 100.0
    print(f"2++ vs comfy-kitchen-config (fp32): {speedup:.3f}x "
          f"({delta_pct:+.1f}% wall on this kernel at this shape)")


if __name__ == "__main__":
    main()
