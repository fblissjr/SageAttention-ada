#!/usr/bin/env python3
# Last updated: 2026-04-26
"""End-to-end gen-wall-time bench: sage-on vs sage-disabled via the ComfyUI HTTP API.

Closes the load-bearing "kernel ms is not gen ms" gap in this fork's
perf-research framework (see VISION.md "What we might be wrong
about" / item 3 and CLAUDE.md "Performance research"). Until this
script runs, every claim about sage-fork's perf impact is theoretical
-- we measured 19.95 ms on the primary kernel row but never showed
that translated into a real DiT render going from X seconds to Y.

What this does
--------------
For a given LTX (or Flux / Z-Image) render workflow:

1. Submits the workflow N times with the AudioLoopHelperSageAttention
   node set to a sage-active mode (default: `auto_mask_aware`).
2. Submits the same workflow N times with the node set to `disabled`
   (which bypasses sage and falls back to ComfyUI's pytorch SDPA).
3. Captures total wall-time per run (POST /prompt to /history shows
   the prompt completed).
4. Reads the most recent sage trace JSONL (written by the consumer's
   nodes_sage.py when AUDIOLOOPHELPER_SAGE_TRACE=auto), filters per
   run by timestamp window, sums `elapsed_us` to get attention-time
   per run.
5. Reports median wall-time per arm, the speedup ratio
   wall_off / wall_on, and the attention-fraction-of-step on the
   sage arm.

Output is a one-shot answer to "does this fork move real gen time
on the workload we run?" If wall_off / wall_on < 1.10, the answer is
"no, kernel ms doesn't translate to gen ms here, attention isn't the
bottleneck." If > 1.5, sage is load-bearing on this workload.

Prerequisites
-------------
- ComfyUI is running and reachable. Host is resolved from (in order):
  --host CLI arg, $COMFYUI_HOST env var, or `internal/local_config.json`
  (key: comfyui_host). No hardcoded default -- if none of those
  resolve, the bench errors out with a pointer to the runbook.
- ComfyUI was launched with AUDIOLOOPHELPER_SAGE_TRACE=auto in the
  environment, so the consumer's tracer wrote a JSONL file. Without
  this, the bench falls back to wall-time-only output (no
  attention-fraction).
- The workflow is in API format (UI: Workflow -> Save (API Format)).
  This script does NOT convert UI-format workflows. UI format lacks
  the per-node input bindings the API needs; the conversion is
  JS-side in ComfyUI's frontend and reimplementing it adds enough
  complexity that we'd rather you click the export button once.
- The workflow contains a node with class_type
  AudioLoopHelperSageAttention (the consumer's first-party node).
  The bench finds it by class_type, not node id, so it works across
  workflows.

Usage
-----
    # See internal/runbook_bench_e2e_ltx.md for the operational runbook,
    # including the one-time UI export step and the local_config.json
    # schema.

    # Once ComfyUI is running and the workflow is saved as API format:
    python tests/bench_e2e_ltx.py --workflow /path/to/workflow.api.json --runs 3

    # Override host (otherwise resolved from env or internal/local_config.json):
    python tests/bench_e2e_ltx.py --host <host:port> --workflow ... --runs 3

What's measured per arm
-----------------------
- median wall-time across N runs (medianed to kill within-arm noise)
- on the sage arm: median attn_total_ms across N runs (sum of
  per-call elapsed_us within each run's wall window) and
  attn_pct = attn_total / wall_time

What's deliberately NOT measured here
-------------------------------------
- Per-step time. The consumer's tracer stamps `iter` from
  transformer_options; aggregating per-iter is in
  scripts/sage_telemetry_summary.py on the consumer side. This
  bench is the gen-level summary; iter-level is downstream.
- Output quality (PSNR / SSIM / LPIPS). Listed in VISION.md "What
  we might be wrong about" as a separate gap; closing it is a
  bigger workstream.
"""

from __future__ import annotations

import argparse
import copy
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import orjson


# Defaults documented inline so an operator running with `--help` sees
# the canonical values without reading source. Host has no default --
# resolved from CLI / env / config file; see resolve_host().
DEFAULT_RUNS = 3
DEFAULT_SAGE_MODE = "auto_mask_aware"
DEFAULT_BASELINE_MODE = "disabled"
SAGE_NODE_CLASS = "AudioLoopHelperSageAttention"

# Speedup-ratio interpretation thresholds. Single source of truth -- the
# CHANGELOG v0.4.0 entry and runbook reference these by value, so when a
# threshold moves, grep these constants to find every callout.
SPEEDUP_LOAD_BEARING = 1.50  # >= : sage carries the workload, kernel research justified
SPEEDUP_HELPS = 1.10         # >= : sage helps but isn't dominant
SPEEDUP_WASH_FLOOR = 0.95    # >= : noise band; below = real regression

# Local-config file (gitignored). Lets an operator pin host / workflow
# paths once on this box without committing them. See
# internal/runbook_bench_e2e_ltx.md for the schema.
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_PATH = REPO_ROOT / "internal" / "local_config.json"

# Where the consumer's tracer writes JSONL when AUDIOLOOPHELPER_SAGE_TRACE=auto.
# Resolved from sage-fork's repo root via the gitignored coderef/ symlink/copy.
# Override with --trace-dir if your consumer install is elsewhere.
DEFAULT_TRACE_DIR = (
    REPO_ROOT / "coderef" / "ComfyUI-AudioLoopHelper" / "internal" / "analysis" / "runs" / "sage"
)


def _load_local_config() -> dict:
    """Read internal/local_config.json if it exists. Returns {} if not.

    Gitignored on purpose -- holds local-machine paths and host info
    we don't want in committed material. Missing file is fine (CLI args
    or env vars are the alternatives); malformed file is a hard error
    so silent typos don't fall through to "default behavior" surprises.
    """
    if not LOCAL_CONFIG_PATH.is_file():
        return {}
    try:
        return orjson.loads(LOCAL_CONFIG_PATH.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise SystemExit(
            f"ERROR: {LOCAL_CONFIG_PATH} exists but is not valid JSON: {exc}\n"
            f"       See internal/runbook_bench_e2e_ltx.md for the schema."
        )


def resolve_host(cli_host: str | None) -> str:
    """Resolve the ComfyUI host from CLI > env > local_config.json.

    Hard-error with a runbook pointer if none of the three resolves --
    we don't want to fall back to a hardcoded value that might silently
    point at the wrong box.
    """
    if cli_host:
        return cli_host
    env_host = os.environ.get("COMFYUI_HOST", "").strip()
    if env_host:
        return env_host
    cfg_host = _load_local_config().get("comfyui_host")
    if cfg_host:
        return cfg_host
    raise SystemExit(
        "ERROR: ComfyUI host is not configured. Set ONE of:\n"
        "  --host <host:port>                       (per-invocation)\n"
        "  export COMFYUI_HOST=<host:port>          (per-shell)\n"
        f"  echo '{{\"comfyui_host\": \"<host:port>\"}}' > {LOCAL_CONFIG_PATH}"
        "  (per-checkout, gitignored)\n"
        "See internal/runbook_bench_e2e_ltx.md for the full setup."
    )


# ---------------------------------------------------------------------------
# ComfyUI HTTP API
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 10.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _http_post(url: str, body: bytes, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def submit_prompt(host: str, prompt: dict, client_id: str) -> str:
    """POST a prompt; return its prompt_id."""
    body = orjson.dumps({"prompt": prompt, "client_id": client_id})
    raw = _http_post(f"http://{host}/prompt", body)
    payload = orjson.loads(raw)
    if "prompt_id" not in payload:
        raise RuntimeError(f"ComfyUI rejected prompt: {payload}")
    return payload["prompt_id"]


def wait_for_completion(host: str, prompt_id: str, *, poll_interval_s: float = 1.0,
                         timeout_s: float = 1800.0) -> dict:
    """Poll /history/<prompt_id> until the prompt appears (= done). Return its history entry."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            raw = _http_get(f"http://{host}/history/{prompt_id}", timeout=10.0)
            history = orjson.loads(raw)
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError as exc:
            # Transient network errors during a long render are common; log
            # and keep polling. A persistent failure trips the timeout.
            print(f"  poll error (continuing): {exc}", file=sys.stderr)
        time.sleep(poll_interval_s)
    raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout_s:.0f}s")


# ---------------------------------------------------------------------------
# Prompt mutation
# ---------------------------------------------------------------------------

def find_sage_node_id(prompt: dict) -> str:
    """Return the node id (string key) for the AudioLoopHelperSageAttention node.

    Class-type lookup, not id-based, so this stays robust if the
    consumer renumbers nodes between workflow versions.
    """
    candidates = [
        node_id for node_id, node in prompt.items()
        if isinstance(node, dict) and node.get("class_type") == SAGE_NODE_CLASS
    ]
    if not candidates:
        raise ValueError(
            f"workflow has no {SAGE_NODE_CLASS!r} node. Either the workflow doesn't "
            f"use sage, or it uses KJNodes' PathchSageAttentionKJ -- run the "
            f"consumer's apply_audioloophelper_sage.py to swap it first."
        )
    if len(candidates) > 1:
        raise ValueError(f"workflow has multiple {SAGE_NODE_CLASS} nodes: {candidates}")
    return candidates[0]


def set_sage_mode(prompt: dict, mode: str) -> dict:
    """Return a copy of `prompt` with the sage node's `mode` input set.

    Deep-copy is correct at this scale (~20-50 nodes, small dicts) --
    micro-optimizations don't pay off against a 30-90s render.
    """
    out = copy.deepcopy(prompt)
    out[find_sage_node_id(out)]["inputs"]["mode"] = mode
    return out


# ---------------------------------------------------------------------------
# Sage trace JSONL
# ---------------------------------------------------------------------------

def latest_trace_file(trace_dir: Path) -> Path | None:
    if not trace_dir.is_dir():
        return None
    candidates = sorted(trace_dir.glob("sage_*.jsonl"))
    return candidates[-1] if candidates else None


def sum_attn_us_in_window(trace_path: Path, ts_min: float, ts_max: float) -> tuple[int, float]:
    """Sum elapsed_us over sage rows whose ts falls within [ts_min, ts_max].

    Returns (call_count, total_elapsed_us). Inclusive boundaries -- a row
    on either edge belongs to this run; half-open windows misattribute the
    first/last attention call of every prompt.
    """
    total_us = 0.0
    n = 0
    with trace_path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if row.get("event") in {"header", "summary"}:
                continue
            ts = row.get("ts")
            if ts is None or ts < ts_min or ts > ts_max:
                continue
            elapsed = row.get("elapsed_us")
            if elapsed is None:
                continue
            total_us += float(elapsed)
            n += 1
    return n, total_us


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    mode: str
    wall_s: float
    ts_start: float
    ts_end: float
    prompt_id: str
    # Filled in post-hoc by attach_attn_times(); zero on the off-arm
    # (no sage trace by design).
    attn_calls: int = 0
    attn_total_ms: float = 0.0


def run_one(host: str, prompt: dict, mode: str, run_idx: int, client_id: str) -> RunResult:
    """Submit one render, wait for completion, capture timing.

    `time.time()` (not perf_counter) is used for ts_start/ts_end because
    the sage tracer also uses time.time() -- correlation requires the
    same wall clock. perf_counter would be sub-microsecond more accurate
    but unaligned to the tracer's stamps.
    """
    submit_prompt_dict = set_sage_mode(prompt, mode)
    label = f"[{mode:>17s} run {run_idx + 1}]"
    print(f"{label} submitting...", flush=True)
    ts_start = time.time()
    pid = submit_prompt(host, submit_prompt_dict, client_id)
    wait_for_completion(host, pid)
    ts_end = time.time()
    wall_s = ts_end - ts_start
    print(f"{label} done in {wall_s:.2f}s  (prompt_id={pid[:8]}...)", flush=True)
    return RunResult(mode, wall_s, ts_start, ts_end, pid)


def attach_attn_times(results: list[RunResult], trace_path: Path | None) -> None:
    """For each run, sum attention elapsed_us within its wall window."""
    if trace_path is None or not trace_path.is_file():
        return
    for r in results:
        n, total_us = sum_attn_us_in_window(trace_path, r.ts_start, r.ts_end)
        r.attn_calls = n
        r.attn_total_ms = total_us / 1000.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results_on: list[RunResult], results_off: list[RunResult],
            trace_path: Path | None) -> None:
    print()
    print("=== Summary ===")
    print(f"trace file: {trace_path if trace_path else '(not found; wall-time only)'}")
    print()

    def _arm_line(label: str, runs: list[RunResult], with_attn: bool) -> str:
        wall_med = statistics.median(r.wall_s for r in runs)
        if with_attn and any(r.attn_calls for r in runs):
            attn_med = statistics.median(r.attn_total_ms for r in runs)
            calls_med = statistics.median(r.attn_calls for r in runs)
            attn_pct = attn_med / 1000.0 / wall_med * 100.0
            return (f"  {label:<10s}  wall={wall_med:7.2f}s  "
                    f"attn={attn_med/1000.0:6.2f}s  ({attn_pct:5.1f}% of wall)  "
                    f"calls={calls_med:.0f}")
        return f"  {label:<10s}  wall={wall_med:7.2f}s"

    print(_arm_line("sage_on", results_on, with_attn=True))
    print(_arm_line("sage_off", results_off, with_attn=False))
    print()

    on_med = statistics.median(r.wall_s for r in results_on)
    off_med = statistics.median(r.wall_s for r in results_off)
    speedup = off_med / on_med if on_med > 0 else float("nan")
    delta_s = off_med - on_med
    pct_saved = (delta_s / off_med * 100.0) if off_med > 0 else 0.0

    print(f"speedup:  sage_off / sage_on = {speedup:.3f}x")
    print(f"saved:    {delta_s:+.2f}s ({pct_saved:+.1f}% of off-baseline)")

    if any(r.attn_calls for r in results_on):
        attn_med_on = statistics.median(r.attn_total_ms for r in results_on) / 1000.0
        non_attn_on = on_med - attn_med_on
        non_attn_off = off_med  # off-arm has no sage trace; treat as all non-sage time
        print()
        print(f"attn-time on sage path: {attn_med_on:.2f}s")
        print(f"non-attn time (sage):   {non_attn_on:.2f}s")
        print(f"non-attn time (off):    {non_attn_off - (off_med - on_med):.2f}s  "
              f"(estimated; off-arm has no per-call breakdown)")

    print()
    print("Interpretation:")
    if speedup >= SPEEDUP_LOAD_BEARING:
        print(f"  Sage is load-bearing on this workload "
              f"({speedup:.2f}x speedup). Kernel-side perf research justified.")
    elif speedup >= SPEEDUP_HELPS:
        print(f"  Sage helps but isn't dominant ({speedup:.2f}x). "
              f"Kernel improvements translate at ~{(speedup - 1) / 1:.0%} per kernel-x.")
    elif speedup >= SPEEDUP_WASH_FLOOR:
        print(f"  Sage is a wash on this workload ({speedup:.2f}x). "
              f"Attention isn't the bottleneck end-to-end -- look elsewhere.")
    else:
        print(f"  Sage is SLOWER end-to-end ({speedup:.2f}x). Check for "
              f"instrumentation overhead, fallback paths, or a real regression.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workflow", required=True, type=Path,
                         help="Path to API-format workflow JSON. Save via UI: Workflow -> Save (API Format).")
    parser.add_argument("--host", default=None,
                         help="ComfyUI host:port. If omitted, resolves from $COMFYUI_HOST or "
                              "internal/local_config.json (key: comfyui_host).")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                         help=f"Number of runs per arm (default: {DEFAULT_RUNS}).")
    parser.add_argument("--sage-mode", default=DEFAULT_SAGE_MODE,
                         help=f"Sage mode for the on-arm (default: {DEFAULT_SAGE_MODE}).")
    parser.add_argument("--baseline-mode", default=DEFAULT_BASELINE_MODE,
                         help=f"Sage mode for the off-arm (default: {DEFAULT_BASELINE_MODE}).")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR,
                         help=f"Where the sage tracer JSONL lives (default: {DEFAULT_TRACE_DIR}).")
    parser.add_argument("--inter-run-sleep", type=float, default=1.0,
                         help="Seconds to sleep between runs (helps disambiguate trace ts windows).")
    args = parser.parse_args()

    host = resolve_host(args.host)

    if not args.workflow.is_file():
        print(f"ERROR: workflow not found: {args.workflow}", file=sys.stderr)
        return 1

    raw = args.workflow.read_bytes()
    try:
        prompt = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        print(f"ERROR: workflow is not valid JSON: {exc}", file=sys.stderr)
        return 1

    # Sanity-check format. UI workflows have a top-level `nodes` list;
    # API workflows have node-id keys at the top level.
    if "nodes" in prompt and isinstance(prompt.get("nodes"), list):
        print(
            "ERROR: This looks like a UI-format workflow (has top-level 'nodes' list).\n"
            "       Open it in the ComfyUI UI and use Workflow -> Save (API Format).\n"
            "       Then re-run this bench with the .api.json file.",
            file=sys.stderr,
        )
        return 1

    try:
        sage_id = find_sage_node_id(prompt)
        print(f"sage node: id={sage_id} class={SAGE_NODE_CLASS}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    client_id = str(uuid.uuid4())

    # Pre-flight: ComfyUI reachable?
    try:
        _http_get(f"http://{host}/system_stats", timeout=5.0)
    except Exception as exc:
        print(f"ERROR: ComfyUI not reachable at http://{host}: {exc}", file=sys.stderr)
        return 1

    trace_path_before = latest_trace_file(args.trace_dir)
    if trace_path_before is None:
        print(f"WARN: no sage trace file found in {args.trace_dir}.")
        print( "      Either ComfyUI was launched without AUDIOLOOPHELPER_SAGE_TRACE=auto,")
        print( "      or the consumer's tracer hasn't written yet. Bench will report wall-time only.")
    else:
        print(f"trace file: {trace_path_before}")
    print()

    # Run interleaved (on / off / on / off / ...) so that wall-time bias from
    # any global drift (thermal, system load, model state) hits both arms
    # equally. Within-arm variance still meaningful; cross-arm bias minimised.
    results_on: list[RunResult] = []
    results_off: list[RunResult] = []
    for run_idx in range(args.runs):
        results_on.append(run_one(host, prompt, args.sage_mode, run_idx, client_id))
        time.sleep(args.inter_run_sleep)
        results_off.append(run_one(host, prompt, args.baseline_mode, run_idx, client_id))
        time.sleep(args.inter_run_sleep)

    # Re-resolve in case ComfyUI restarted mid-bench and rotated the file.
    trace_path = latest_trace_file(args.trace_dir) if args.trace_dir.is_dir() else None
    attach_attn_times(results_on, trace_path)
    report(results_on, results_off, trace_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
