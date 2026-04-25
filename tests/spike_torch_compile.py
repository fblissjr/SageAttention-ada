"""Spike: does torch.compile around sage still need to be disabled?

A downstream consumer wraps every sage call in `torch.compiler.disable()`
per a comment claiming "sage fork's torch.compile support is recent and
thin." That decision was made on a prior torch version. This spike
re-tests on the current torch to either confirm the disable is still
warranted or open a follow-up to remove it.

What the spike does, for one canonical LTX self-attn shape:

1. Run sage's `auto` dispatch eagerly. Capture output and median wall-clock.
2. Wrap the same call in `torch.compile(mode='reduce-overhead', dynamic=False)`.
3. Run it. Capture output (allow first call to be a slow trace) and median.
4. Compare outputs (rtol < 1e-2) and speeds.
5. Print one of three verdicts:
     - "compile failed/errored: keep the disable"
     - "compile worked but no speedup: keep the disable (overhead, no value)"
     - "compile worked AND speedup: open a follow-up to remove the disable"

Run from the venv that has sage editable + torch 2.11+:
    ${VIRTUAL_ENV}/bin/python tests/spike_torch_compile.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import torch


SHAPE = (1, 32, 31776, 64)  # LTX self-attn-large, the load-bearing path
DTYPE = torch.bfloat16
# Mean relative error tolerance: |compiled - eager| / |eager|, mean over
# the output tensor. Not torch.allclose-style worst-case rtol; we accept
# a fraction-of-elements approximation since compile may fuse precision-
# affecting ops uniformly.
MEAN_REL_ERR_TOLERANCE = 1e-2
WARMUP = 2
RUNS = 5


def _make_qkv():
    q = torch.randn(*SHAPE, device="cuda", dtype=DTYPE)
    k = torch.randn(*SHAPE, device="cuda", dtype=DTYPE)
    v = torch.randn(*SHAPE, device="cuda", dtype=DTYPE)
    return q, k, v


def _time_calls(fn: Callable, q, k, v, runs: int) -> list[float]:
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(q, k, v)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _eager_call(q, k, v):
    import sageattention as _sa
    return _sa.sageattn(q, k, v, is_causal=False, attn_mask=None, tensor_layout="HND")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available -- this spike measures kernel timing on-GPU.")
        return 1

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"shape:  {SHAPE}, dtype={DTYPE}")
    print()

    q, k, v = _make_qkv()

    # Eager baseline
    out_eager = _eager_call(q, k, v)
    for _ in range(WARMUP):
        _eager_call(q, k, v)
    eager_samples = _time_calls(_eager_call, q, k, v, RUNS)
    eager_median = statistics.median(eager_samples)
    print(f"eager:    median {eager_median:.2f} ms over {RUNS} runs (warmup {WARMUP})")

    # Two compile modes worth trying:
    #   - 'reduce-overhead' uses CUDA Graphs. Spike-tested first because
    #     it's the typical pick for CUDA-bound kernels. Known caveat:
    #     CUDA Graphs reuse output buffers, which breaks if the caller
    #     reads the output between calls.
    #   - 'default' is the conservative path -- AOT trace + inductor, no
    #     CUDA Graphs. Less acceleration but more compatibility.
    for compile_mode in ("reduce-overhead", "default"):
        print(f"--- compile_mode={compile_mode!r} ---")
        try:
            compiled = torch.compile(_eager_call, mode=compile_mode, dynamic=False)
        except Exception as exc:
            print(f"compile setup failed: {type(exc).__name__}: {exc}")
            continue

        # First call: trace + compile. Slow; not measured.
        try:
            t0 = time.perf_counter()
            out_compiled = compiled(q, k, v)
            torch.cuda.synchronize()
            first_call_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  first call (trace + compile): {first_call_ms:.0f} ms")
        except Exception as exc:
            print(f"  first compiled call errored: {type(exc).__name__}: {str(exc)[:140]}")
            continue

        # Numerical sanity BEFORE timing -- if outputs are bad we don't
        # care about the speed number. Clone to dodge CUDA Graphs buffer
        # reuse if mode='reduce-overhead'.
        try:
            out_compiled = out_compiled.clone()
            diff = (out_compiled.float() - out_eager.float()).abs()
            mean_rel_err = (diff / out_eager.float().abs().clamp(min=1e-8)).mean().item()
            print(f"  mean relative error (compiled vs eager): {mean_rel_err:.4g}")
            if mean_rel_err > MEAN_REL_ERR_TOLERANCE:
                print(f"  REJECT: mean_rel_err {mean_rel_err:.3g} > {MEAN_REL_ERR_TOLERANCE}; output drift")
                continue
        except Exception as exc:
            print(f"  output comparison errored: {type(exc).__name__}: {str(exc)[:140]}")
            continue

        # Warmup + measure
        try:
            for _ in range(WARMUP):
                compiled(q, k, v)
            compiled_samples = _time_calls(compiled, q, k, v, RUNS)
            compiled_median = statistics.median(compiled_samples)
            print(f"  compiled: median {compiled_median:.2f} ms over {RUNS} runs")
        except Exception as exc:
            print(f"  compiled timing errored: {type(exc).__name__}: {str(exc)[:140]}")
            continue

        speedup = eager_median / compiled_median if compiled_median > 0 else 0.0
        print(f"  speedup vs eager: {speedup:.3f}x  (eager {eager_median:.2f} ms)")
        print()
        if speedup > 1.05:
            print(f"verdict ({compile_mode}): compile yields {speedup:.2f}x speedup with bounded rtol -- "
                  "open a follow-up to investigate removing the disable")
            return 0
        elif speedup > 0.95:
            print(f"verdict ({compile_mode}): compile no meaningful speedup ({speedup:.2f}x) -- keep the disable")
        else:
            print(f"verdict ({compile_mode}): compile is SLOWER ({speedup:.2f}x) -- keep the disable")

    print()
    print("final: no compile mode produced a clean speedup. keep the disable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
