#!/usr/bin/env python3
"""Aggregate a consumer's sage trace JSONL into a workload distribution.

The bench harness (`tests/test_sageattn_ltx_shapes.py`) measures sage at
a fixed list of synthetic shapes. This script answers the dual question:
which of those shapes the consumer's *actual* renders hit, and how
heavily. Without it, we'd be optimizing whatever shape we picked, not
whatever shape matters.

Inputs:
- One or more sage trace JSONL files. Format produced by the consumer's
  `nodes_sage.py` tracer when `AUDIOLOOPHELPER_SAGE_TRACE=auto` is set;
  one row per attention call with `shape`, `has_mask`, `dispatched_kernel`,
  `elapsed_us`, `mode`, `effective_mode`, `iter`, `prompt_id`.
- Optional --baselines path (default: tests/regression_baselines.json)
  to surface which load-bearing bench shapes the trace exercised.

Outputs:
- Per-(shape, has_mask, dispatched_kernel) aggregate: call count, total
  ms, median us. Sorted by total ms desc.
- "Top N by total time" summary so the load-bearing tuples surface.
- "Coverage check" cross-referencing with `regression_baselines.json`'s
  load-bearing shapes — confirms the bench grades against shapes the
  consumer actually exercises.
- "Coverage gaps" section listing trace shapes the bench doesn't cover.
  This is the input that justifies adding bench rows.

Usage:
    ${VIRTUAL_ENV}/bin/python tests/bench_workload_profile.py \\
        /path/to/sage_2026-04-26_105851.jsonl

    # Multiple files (e.g. all traces from a session under data/runs/):
    ${VIRTUAL_ENV}/bin/python tests/bench_workload_profile.py \\
        path/to/runs/*/sage.jsonl

    # Filter by prompt_id (post v0.4.0 consumer trace format):
    ${VIRTUAL_ENV}/bin/python tests/bench_workload_profile.py \\
        --prompt-id abc123 trace.jsonl

Note on shape interpretation: the consumer logs `[B, S, H*D]` flat
shape because that's what comes into the override hook before sage's
internal reshape. The bench's `Shape` rows are `(B, H, S, D)` per-head.
We don't try to recover head_dim from the flat shape (would need
heads, which the trace doesn't carry); we report the flat shape and
let the operator cross-reference. For LTX 2.3: 32 heads * 64 d = 2048
hidden; for Flux/Z-Image: 32 * 128 = 4096 or 32 * 120 = 3840.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import orjson


# Synthetic kernel-name prefix used for rows where consumer policy
# short-circuited sage entirely (e.g. AudioLoopHelper's
# skip_under_seq_len). A downstream filter `dispatched_kernel ==
# "fp8_cuda++"` would silently miss these rows otherwise; the prefix
# makes the asymmetry visible at the field level. Match with
# `kernel.startswith(SKIPPED_KERNEL_PREFIX)` to find policy-skipped
# rows; everything else is a real sage dispatch.
SKIPPED_KERNEL_PREFIX = "skipped:"


@dataclass
class ShapeAggregate:
    shape: tuple[int, ...]
    has_mask: bool
    dispatched_kernel: str
    elapsed_us_samples: list[float]

    @property
    def calls(self) -> int:
        return len(self.elapsed_us_samples)

    @property
    def total_ms(self) -> float:
        return sum(self.elapsed_us_samples) / 1000.0

    @property
    def median_us(self) -> float:
        return statistics.median(self.elapsed_us_samples) if self.elapsed_us_samples else 0.0

    @property
    def p90_us(self) -> float:
        if not self.elapsed_us_samples:
            return 0.0
        sorted_s = sorted(self.elapsed_us_samples)
        idx = max(0, int(len(sorted_s) * 0.90) - 1)
        return sorted_s[idx]

    def shape_str(self) -> str:
        return "[" + ", ".join(str(x) for x in self.shape) + "]"


def parse_traces(
    paths: list[Path],
    prompt_id_filter: str | None = None,
) -> tuple[list[ShapeAggregate], dict[str, int], int, dict[str, int]]:
    """Parse one or more trace JSONL files. Returns
    (aggregates, kernel_source_counts, total_rows, skip_reason_counts).
    `kernel_source_counts` buckets rows by dispatched_kernel attribution
    (`sage_telemetry` if dispatched_kernel field is present, `legacy` if
    only effective_mode is). `skip_reason_counts` buckets rows where the
    consumer's policy short-circuited sage entirely (e.g.
    skip_under_seq_len -> skip_reason="under_seq_len" since their commit
    04919fd, 2026-04-27); rows without a skip_reason field bucket as
    "not_skipped" so the dict's totals match total_rows.

    The consumer's summary script
    (`coderef/ComfyUI-AudioLoopHelper/scripts/sage_telemetry_summary.py`)
    parses the same JSONL schema for a different question
    ("what % of gen wall-time is masked-triton?"); any field-name
    change must land in both places."""
    bucket: dict[tuple[tuple[int, ...], bool, str], list[float]] = defaultdict(list)
    kernel_source_counts: dict[str, int] = {"sage_telemetry": 0, "legacy_inferred": 0}
    skip_reason_counts: dict[str, int] = {"not_skipped": 0}
    total_rows = 0

    for path in paths:
        if not path.exists():
            print(f"WARN: trace file does not exist: {path}", file=sys.stderr)
            continue
        with path.open("rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                try:
                    row = orjson.loads(raw)
                except orjson.JSONDecodeError:
                    continue
                if row.get("event") == "header":
                    continue
                if "shape" not in row:
                    continue
                if prompt_id_filter is not None and row.get("prompt_id") != prompt_id_filter:
                    continue

                total_rows += 1
                shape = tuple(row["shape"])
                has_mask = bool(row.get("has_mask", False))
                elapsed_us = float(row.get("elapsed_us", 0.0))

                # Skip-reason bucketing. Consumer policy (e.g.
                # AudioLoopHelper's skip_under_seq_len, shipped 2026-04-27
                # in their commit 04919fd) short-circuits sage and routes
                # to torch SDPA on shapes where sage loses; trace rows on
                # the skip path carry skipped=true + skip_reason=<code>.
                # Aggregating these separately so the workload-profile
                # output makes "policy-skipped" calls visible at a glance,
                # distinct from sage's own dispatch decisions and from
                # error fallbacks.
                if row.get("skipped"):
                    reason = str(row.get("skip_reason") or "unknown")
                    skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                    kernel = f"{SKIPPED_KERNEL_PREFIX}{reason}"
                    kernel_source_counts["sage_telemetry"] += 1
                else:
                    skip_reason_counts["not_skipped"] += 1
                    # Prefer the v0.2.0+ dispatched_kernel field. Fall back
                    # to effective_mode for legacy traces (records the
                    # routing decision, not the kernel that actually ran).
                    if "dispatched_kernel" in row and row["dispatched_kernel"]:
                        kernel = str(row["dispatched_kernel"])
                        kernel_source_counts["sage_telemetry"] += 1
                    else:
                        kernel = str(row.get("effective_mode", "unknown"))
                        kernel_source_counts["legacy_inferred"] += 1

                bucket[(shape, has_mask, kernel)].append(elapsed_us)

    aggregates = [
        ShapeAggregate(shape=k[0], has_mask=k[1], dispatched_kernel=k[2],
                       elapsed_us_samples=v)
        for k, v in bucket.items()
    ]
    aggregates.sort(key=lambda a: a.total_ms, reverse=True)
    return aggregates, kernel_source_counts, total_rows, skip_reason_counts


def load_baseline_shapes(baselines_path: Path) -> set[str]:
    """Read load-bearing bench shape names from regression_baselines.json.
    Empty set if file absent — graceful degradation; the workload profile
    still reports its own data."""
    if not baselines_path.exists():
        return set()
    cfg = orjson.loads(baselines_path.read_bytes())
    return {
        e["shape"] for e in cfg.get("baselines", [])
        if e.get("load_bearing", False)
    }


def print_aggregates(aggregates: list[ShapeAggregate]) -> None:
    if not aggregates:
        print("(no rows)")
        return

    grand_total_ms = sum(a.total_ms for a in aggregates)

    shape_w = max(len(a.shape_str()) for a in aggregates)
    shape_w = max(shape_w, len("shape"))
    kernel_w = max(len(a.dispatched_kernel) for a in aggregates)
    kernel_w = max(kernel_w, len("kernel"))

    header = (
        f"{'shape':<{shape_w}}  {'mask':>5}  {'kernel':<{kernel_w}}  "
        f"{'calls':>7}  {'total_ms':>10}  {'%':>6}  {'median_us':>10}  {'p90_us':>10}"
    )
    print(header)
    print("-" * len(header))
    for a in aggregates:
        pct = (a.total_ms / grand_total_ms * 100.0) if grand_total_ms > 0 else 0.0
        print(
            f"{a.shape_str():<{shape_w}}  {str(a.has_mask):>5}  "
            f"{a.dispatched_kernel:<{kernel_w}}  "
            f"{a.calls:>7}  {a.total_ms:>10.2f}  {pct:>5.1f}%  "
            f"{a.median_us:>10.1f}  {a.p90_us:>10.1f}"
        )
    print("-" * len(header))
    print(f"total: {grand_total_ms:.2f} ms across {sum(a.calls for a in aggregates)} calls")


def print_top_n(aggregates: list[ShapeAggregate], n: int = 5) -> None:
    """Top-N tuples by total wall time. The point of this section: if
    one (shape, kernel) tuple is >50% of total attention time, that's
    the row to optimize. Anything <5% is decorative."""
    if not aggregates:
        return
    grand_total_ms = sum(a.total_ms for a in aggregates)
    print()
    print(f"Top {min(n, len(aggregates))} by total wall time:")
    for i, a in enumerate(aggregates[:n], 1):
        pct = (a.total_ms / grand_total_ms * 100.0) if grand_total_ms > 0 else 0.0
        print(
            f"  {i}. {a.shape_str()} mask={a.has_mask} {a.dispatched_kernel}: "
            f"{a.total_ms:.2f} ms ({pct:.1f}%) over {a.calls} calls"
        )


def print_coverage_check(
    aggregates: list[ShapeAggregate],
    baselines_path: Path,
) -> None:
    """Cross-reference trace shapes with bench load-bearing rows.

    Two outputs:
    - Coverage hit: which load-bearing baseline shape names map to a
      shape we saw in the trace. (Heuristic since trace shape is flat
      [B, S, H*D] and bench shape names embed semantic labels like
      "ltx23_video_self_attn_init_22932".)
    - Coverage gap: trace shapes that don't match any bench row.
      Justifies adding a new bench row.
    """
    print()
    print("Coverage check vs bench load-bearing shapes:")
    load_bearing = load_baseline_shapes(baselines_path)
    if not load_bearing:
        print("  (no baselines configured at "
              f"{baselines_path}; skipping)")
        return

    # Heuristic mapping: bench rows we know how to recognize from the
    # flat trace shape. The trace logs [B, S, H*D] (q.shape going into
    # the optimized_attention_override hook, post sage's reshape input).
    # We map by well-known (seq, hidden) patterns. Low-confidence
    # fallbacks are flagged.
    #
    # LTX 2.3 default config (transformer_ltx2.py:907-947):
    #   Video path: 32 heads * 128 d  -> hidden=4096
    #   Audio path: 32 heads *  64 d  -> hidden=2048
    #
    # Production seq lengths from a real consumer trace:
    #   iter=null (init render):  seq=22932
    #   iter>=1   (loop iters):   seq=23296
    # Short-Q paths: seq=497 (init) and seq=498 (loop iters).
    known_patterns: dict[tuple[int, int], str] = {
        # LTX 2.3 video self-attn (d=128). 76% of attention wall-time.
        (22932, 4096): "ltx23_video_self_attn_init_22932",
        (23296, 4096): "ltx23_video_self_attn_loop_23296",
        # LTX 2.3 audio-side self-attn (d=64).
        (22932, 2048): "ltx23_audio_self_attn_init_22932",
        (23296, 2048): "ltx23_audio_self_attn_loop_23296",
        # Short-Q at hidden=2048: Gemma 3 text-encoder self-attn or
        # audio-cross-attn (attribution ambiguous from trace alone).
        (497, 2048):   "ltx23_short_q_init_497",
        (498, 2048):   "ltx23_short_q_loop_498",
        # Image-gen reference rows (live in test_sageattn_image_shapes.py).
        (4096, 4096):  "image_gen_self_attn_4096_h24_d128",
        (4608, 3840):  "z_image_turbo_self_attn_4608_h32_d120",
    }

    seen_bench_rows: set[str] = set()
    unmatched: list[ShapeAggregate] = []
    for a in aggregates:
        if len(a.shape) != 3:
            unmatched.append(a)
            continue
        _, seq, hidden = a.shape
        bench_name = known_patterns.get((seq, hidden))
        if bench_name and not a.has_mask:
            seen_bench_rows.add(bench_name)
        elif bench_name and a.has_mask:
            # Production traces with masks on these q-shapes route to
            # the K-probe pair (cross_unmasked_kv226 + cross_text_kv226).
            # Surface as the K-probe witness if the q-side matches video.
            if hidden == 4096:
                seen_bench_rows.add("ltx23_video_cross_text_kv226")
            else:
                seen_bench_rows.add(
                    f"{bench_name} (with-mask, audio cross-attn ambiguous)"
                )
        elif a.has_mask:
            # Masked path on a q-shape we don't recognize. Surface for
            # the operator to investigate.
            seen_bench_rows.add(
                f"unknown-masked-q (trace seq={seq}, hidden={hidden})"
            )
        else:
            unmatched.append(a)

    for shape_name in sorted(load_bearing):
        hit = any(shape_name in seen for seen in seen_bench_rows)
        marker = "HIT " if hit else "MISS"
        print(f"  [{marker}] {shape_name}")
    extras = seen_bench_rows - load_bearing
    for s in sorted(extras):
        print(f"  [hit ] {s}  (matched but not in load-bearing set)")

    if unmatched:
        print()
        print("Coverage gaps (trace shapes the bench doesn't cover):")
        for a in unmatched[:10]:
            print(
                f"  {a.shape_str()} mask={a.has_mask}: "
                f"{a.calls} calls, {a.total_ms:.2f} ms total"
            )
        if len(unmatched) > 10:
            print(f"  ... +{len(unmatched) - 10} more")


def print_freshness(kernel_source_counts: dict[str, int]) -> None:
    """Surface trace freshness, mirroring the consumer's summary script.
    Pre-v0.2.0 traces only carry effective_mode (the routing decision);
    v0.2.0+ adds dispatched_kernel (the actual kernel)."""
    print()
    total = sum(kernel_source_counts.values())
    if total == 0:
        return
    print("Trace freshness:")
    for source, count in kernel_source_counts.items():
        pct = count / total * 100.0
        print(f"  {source:<20} {count:>7} ({pct:5.1f}%)")
    if kernel_source_counts.get("legacy_inferred", 0) > 0:
        print("  (legacy_inferred > 0 means some rows predate the v0.2.0 "
              "dispatched_kernel field; bench coverage call-graph is "
              "approximate for those rows.)")


def print_skip_reasons(skip_reason_counts: dict[str, int]) -> None:
    """Surface consumer-side skip-policy reach. Suppressed when no
    rows were skipped (the common case before AudioLoopHelper's
    skip_under_seq_len landed)."""
    skipped_total = sum(c for r, c in skip_reason_counts.items() if r != "not_skipped")
    if skipped_total == 0:
        return
    print()
    grand = sum(skip_reason_counts.values())
    print("Consumer skip policy:")
    for reason, count in sorted(skip_reason_counts.items(),
                                key=lambda kv: kv[1], reverse=True):
        pct = count / grand * 100.0
        print(f"  {reason:<20} {count:>7} ({pct:5.1f}%)")
    print(f"  -> {skipped_total} of {grand} attention calls "
          f"({skipped_total / grand * 100.0:.1f}%) were policy-skipped "
          f"before reaching sage.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_paths", nargs="+", type=Path,
        help="One or more sage trace JSONL files.",
    )
    parser.add_argument(
        "--prompt-id", type=str, default=None,
        help="Filter rows by prompt_id (post-v0.4.0 consumer traces).",
    )
    parser.add_argument(
        "--baselines", type=Path,
        default=Path(__file__).parent / "regression_baselines.json",
        help="Path to regression_baselines.json for the coverage check.",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="How many rows to surface in the top-N section.",
    )
    args = parser.parse_args()

    aggregates, kernel_source_counts, total_rows, skip_reason_counts = parse_traces(
        args.trace_paths, prompt_id_filter=args.prompt_id,
    )

    print(f"Workload profile across {len(args.trace_paths)} trace file(s)")
    if args.prompt_id:
        print(f"  prompt_id filter: {args.prompt_id}")
    print(f"  total rows: {total_rows}")
    print(f"  distinct (shape, has_mask, kernel) tuples: {len(aggregates)}")
    print()

    print_aggregates(aggregates)
    print_top_n(aggregates, n=args.top_n)
    print_coverage_check(aggregates, args.baselines)
    print_skip_reasons(skip_reason_counts)
    print_freshness(kernel_source_counts)


if __name__ == "__main__":
    main()
