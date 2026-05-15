#!/usr/bin/env python3
"""Phase 0 measurement: peak HBM for a two-call Q-partition pattern.

Context: a downstream consumer workflow (LTX 2.3 self-attn guide
masks) partitions Q into two groups -- noisy `[0, guide_start)` and
tracked `[guide_start, guide_start + tracked_count)` -- and fires
two back-to-back `sageattn_qk_int8_pv_fp16_triton` calls per layer
with the same K, V. Each call independently re-quantizes K and
re-casts V to fp16 even though K, V are identical across calls.

The consumer's mask shapes (broadcast-minimal to avoid an
`(1, 1, T, T)` dense-mask blowup):
  noisy_mask   : (1, 1, 1, T)             bf16  -- broadcast across all noisy Q rows
  tracked_mask : (1, 1, tracked_count, T) bf16

Open question this bench answers: does pytorch's caching allocator
reuse the K_int8 / V_fp16 buffers across the two calls, or does the
peak HBM scale with N? If the allocator reuses, a partitioned entry
(quantize-once, slice-Q-many-times) buys latency, not peak HBM. If
the allocator doesn't reuse, the partitioned entry targets a real
peak opportunity.

Decision gate from the plan: peak >700 MB (~+130 MiB above the
~570 MiB single-call reference) -> proceed to build the partitioned
entry. Otherwise reframe.

Run with the venv that has sage installed active. From the repo root:
    ${VIRTUAL_ENV}/bin/python tests/bench/partitioned_mask_phase0/bench_partitioned_mask.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import NamedTuple

import torch

from sageattention import sageattn_qk_int8_pv_fp16_triton, sageattn_partitioned


class SliceSpec(NamedTuple):
    """One call in the partition pattern."""

    name: str
    q_start: int
    q_end: int
    # 1 = broadcast across all Q rows in this slice (noisy-mask shape);
    # q_end - q_start = one row per Q row (tracked-mask shape).
    mask_q_dim: int


def make_two_call_partition(total_t: int, guide_start: int, tracked_count: int) -> list[SliceSpec]:
    """Two-call partition: noisy + tracked slices sharing K, V."""
    assert 0 < guide_start < total_t
    assert tracked_count > 0
    tracked_end = guide_start + tracked_count
    assert tracked_end <= total_t
    return [
        SliceSpec(name="noisy", q_start=0, q_end=guide_start, mask_q_dim=1),
        SliceSpec(
            name="tracked",
            q_start=guide_start,
            q_end=tracked_end,
            mask_q_dim=tracked_count,
        ),
    ]


def make_mask(mask_q_dim: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build attn_mask shaped (1, 1, mask_q_dim, kv_len), zeros (no-op log-weights).

    Phase 0 measures allocation pattern, not mask content. Use zeros in
    matching dtype (additive log-weight semantics: 0 = no attenuation).
    Sage's Triton path accepts bool OR matching-dtype masks; the PR
    uses dtype (bf16) masks holding log-weights, so we mirror that.
    """
    return torch.zeros((1, 1, mask_q_dim, kv_len), dtype=dtype, device=device)


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def measure_single_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> float:
    """Peak HBM for one independent sage call. Returns MiB."""
    reset_peak()
    out = sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=attn_mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del out
    return peak


def measure_partitioned_peak(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slices: list[SliceSpec],
    with_mask: bool,
) -> float:
    """Peak HBM for a single sageattn_partitioned call covering all slices."""
    reset_peak()
    slices_arg = [
        (
            s.q_start,
            s.q_end,
            make_mask(s.mask_q_dim, k.size(2), q.device, q.dtype) if with_mask else None,
        )
        for s in slices
    ]
    out = sageattn_partitioned(q, k, v, slices_arg)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del out, slices_arg
    return peak


def measure_cumulative_peak(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slices: list[SliceSpec],
    with_mask: bool,
) -> tuple[float, list[int]]:
    """Peak HBM across sequential sage calls sharing K, V.

    Outputs are retained for the duration of the sequence to prevent
    the allocator from freeing intermediate buffers between iterations
    -- otherwise the measurement would underreport peak.

    Returns (cumulative_peak_mib, mask_bytes_per_call).
    """
    reset_peak()
    outs: list[torch.Tensor] = []
    mask_bytes: list[int] = []
    for spec in slices:
        q_slice = q[:, :, spec.q_start:spec.q_end, :].contiguous()
        attn_mask = (
            make_mask(spec.mask_q_dim, k.size(2), q.device, q.dtype) if with_mask else None
        )
        mask_bytes.append(attn_mask.element_size() * attn_mask.numel() if attn_mask is not None else 0)
        out = sageattn_qk_int8_pv_fp16_triton(q_slice, k, v, attn_mask=attn_mask)
        outs.append(out)
        del attn_mask
    torch.cuda.synchronize()
    cumulative_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    del outs
    return cumulative_peak, mask_bytes


def fmt(mb: float) -> str:
    return f"{mb:7.1f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=23296, help="LTX self-attn loop-iter seq")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--guide-start", type=int, default=22796)
    parser.add_argument("--tracked-count", type=int, default=500)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--snapshot", type=Path, default=None,
                        help="If set, dump _record_memory_history snapshot to this path.")
    parser.add_argument("--results-json", type=Path, default=None,
                        help="If set, write structured results to this JSON path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")

    slices = make_two_call_partition(args.seq, args.guide_start, args.tracked_count)
    partition_name = " + ".join(
        f"{s.name}[{s.q_start}:{s.q_end}] mask_q={s.mask_q_dim}" for s in slices
    )

    print(f"shape:     B={args.batch} H={args.heads} T={args.seq} D={args.head_dim} dtype={args.dtype}")
    print(f"partition: {partition_name}")
    print(f"device:    {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device=device)
    k = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device=device)
    v = torch.randn(args.batch, args.heads, args.seq, args.head_dim, dtype=dtype, device=device)

    print("warmup: priming triton autotune for full-T and slice shapes ...")
    sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=None)
    sageattn_qk_int8_pv_fp16_triton(q, k, v, attn_mask=make_mask(1, args.seq, device, dtype))
    for spec in slices:
        q_slice = q[:, :, spec.q_start:spec.q_end, :].contiguous()
        sageattn_qk_int8_pv_fp16_triton(
            q_slice, k, v, attn_mask=make_mask(spec.mask_q_dim, args.seq, device, dtype)
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print()

    if args.snapshot is not None:
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    peak_full_no_mask = measure_single_call(q, k, v, attn_mask=None)
    full_mask_broadcast = make_mask(1, args.seq, device, dtype)
    peak_full_with_mask = measure_single_call(q, k, v, attn_mask=full_mask_broadcast)
    del full_mask_broadcast

    cum_no_mask, _ = measure_cumulative_peak(q, k, v, slices, with_mask=False)
    cum_with_mask, mask_bytes = measure_cumulative_peak(q, k, v, slices, with_mask=True)

    # Warm partitioned (own Q-slice shapes are already cached via the
    # earlier per-slice warmup, but the partitioned entry's K-once /
    # V-once allocation pattern is a new path -- one prime call here).
    sageattn_partitioned(
        q, k, v,
        [(s.q_start, s.q_end, make_mask(s.mask_q_dim, args.seq, device, dtype)) for s in slices],
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    partitioned_no_mask = measure_partitioned_peak(q, k, v, slices, with_mask=False)
    partitioned_with_mask = measure_partitioned_peak(q, k, v, slices, with_mask=True)

    if args.snapshot is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"snapshot:  {args.snapshot}")
        print()

    total_mask_mib = sum(mask_bytes) / (1024 * 1024)
    print("RESULTS")
    print("-" * 64)
    print(f"  A. single full-T call, no mask:           {fmt(peak_full_no_mask)}")
    print(f"  B. single full-T call, (1,1,1,T) mask:    {fmt(peak_full_with_mask)}")
    print(f"  C. N-call partition cumulative no mask:   {fmt(cum_no_mask)}")
    print(f"  D. N-call partition cumulative w/ mask:   {fmt(cum_with_mask)}")
    print(f"     per-slice mask sizes:                  "
          + ", ".join(f"{b/1024/1024:.2f} MiB" for b in mask_bytes)
          + f"  (total {total_mask_mib:.2f} MiB)")
    print(f"  E. sageattn_partitioned, no mask:         {fmt(partitioned_no_mask)}")
    print(f"  F. sageattn_partitioned, with mask:       {fmt(partitioned_with_mask)}")
    print()

    saved_no_mask = cum_no_mask - partitioned_no_mask
    saved_with_mask = cum_with_mask - partitioned_with_mask
    print("PARTITIONED ENTRY SAVINGS (vs N-call cumulative)")
    print("-" * 64)
    print(f"  no mask:  {fmt(cum_no_mask)} -> {fmt(partitioned_no_mask)}  ({saved_no_mask:+7.1f} MiB)")
    print(f"  w/ mask:  {fmt(cum_with_mask)} -> {fmt(partitioned_with_mask)}  ({saved_with_mask:+7.1f} MiB)")
    print()

    print("DECISION GATE")
    print("-" * 64)
    ref = peak_full_no_mask
    delta_no_mask = cum_no_mask - ref
    delta_with_mask = cum_with_mask - ref
    print(f"  reference (1 call, no mask):              {fmt(ref)}")
    print(f"  N-call cumulative (no mask)  delta:       {delta_no_mask:+7.1f} MiB")
    print(f"  N-call cumulative (w/ mask)  delta:       {delta_with_mask:+7.1f} MiB")
    print()
    gate_threshold_mib = 130.0
    proceed = max(delta_no_mask, delta_with_mask) > gate_threshold_mib
    if proceed:
        print(f"  GATE: PROCEED. Cumulative peak exceeds reference by >{gate_threshold_mib:.0f} MiB.")
        print("        Allocator is NOT reusing K_int8/V_fp16 across calls.")
        print("        Partitioned entry targets a real peak opportunity.")
    else:
        print(f"  GATE: REFRAME. Cumulative peak within {gate_threshold_mib:.0f} MiB of reference.")
        print("        Allocator IS reusing K_int8/V_fp16 across calls.")
        print("        Partitioned entry would buy LATENCY (skipped K-quant compute),")
        print("        not peak HBM. Reframe consumer-facing message accordingly.")

    if args.results_json is not None:
        import orjson

        payload = {
            "config": {
                "batch": args.batch,
                "heads": args.heads,
                "seq": args.seq,
                "head_dim": args.head_dim,
                "guide_start": args.guide_start,
                "tracked_count": args.tracked_count,
                "dtype": args.dtype,
                "device_name": torch.cuda.get_device_name(0),
                "partition": partition_name,
                "slices": [
                    {"name": s.name, "q_start": s.q_start, "q_end": s.q_end,
                     "mask_q_dim": s.mask_q_dim}
                    for s in slices
                ],
            },
            "results_mib": {
                "single_full_T_no_mask": peak_full_no_mask,
                "single_full_T_with_broadcast_mask": peak_full_with_mask,
                "N_call_cumulative_no_mask": cum_no_mask,
                "N_call_cumulative_with_mask": cum_with_mask,
                "sageattn_partitioned_no_mask": partitioned_no_mask,
                "sageattn_partitioned_with_mask": partitioned_with_mask,
                "savings_no_mask": saved_no_mask,
                "savings_with_mask": saved_with_mask,
                "per_call_mask_bytes": mask_bytes,
                "total_mask_mib": total_mask_mib,
            },
            "decision_gate": {
                "reference_mib": ref,
                "delta_no_mask_mib": delta_no_mask,
                "delta_with_mask_mib": delta_with_mask,
                "threshold_mib": gate_threshold_mib,
                "verdict": "proceed" if proceed else "reframe",
            },
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"results:   {args.results_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
