"""(C) Concurrent-dispatch parallelism spike: sage fp8++ video || pytorch default audio.

Tests Window A of the LTX 2.3 transformer block's concurrent-dispatch
parallelism opportunity. The block forward has video and audio paths
that operate on disjoint tensors with disjoint weights between
cross-attention sync points; Python eager-mode dispatch serializes
them today, but the data dependency doesn't require it. This spike
issues the two kernels on parallel CUDA streams and measures whether
they overlap on the 4090.

Specifically: video-side sage fp8++ attention at production stage-2
shape (T=42240, h=32, d=128) on one CUDA stream, concurrently with
pytorch's default attention on the audio side (T=100, h=32, d=64 --
falls below the production `skip_under_seq_len=1024` threshold so
pytorch is what actually runs in production for audio sub-modules).

Decision gates:
- audio-side wall-time recovered: > 50%  (concurrent dispatch must
  actually overlap, not just serialize behind the saturating video
  kernel)
- video-side slowdown from concurrent issue: < 5%  (no L2 thrash
  from running a second kernel concurrently -- recall v0.6 sage_ffn
  died on exactly this contention mechanism)

Both clear -> pursue dispatch-infrastructure work (~2 weeks: stream-
aware kernel arg on sage's side + monkey-patch the block forward on
the consumer side). Either fails -> drop the parallelism axis.

Run with sage installed in $VIRTUAL_ENV:
    ${VIRTUAL_ENV}/bin/python tests/spikes/spike_concurrent_dispatch.py

For kernel-level overlap evidence (recommended), wrap in nsys:
    nsys profile -o /tmp/spike_concurrent ${VIRTUAL_ENV}/bin/python \\
        tests/spikes/spike_concurrent_dispatch.py
The NVTX ranges below ("seq_video", "seq_audio", "conc_video",
"conc_audio") show up directly in nsys timeline view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Reuse the canonical rtol helper for the correctness sanity check.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]

from sageattention import sageattn_qk_int8_pv_fp8_cuda


# LTX 2.3 production shapes (per docs/ltx_workload_profile.md +
# tests/test_sageattn_ltx_shapes.py:63-64 architecture notes).
# Video path: 32 heads, head_dim=128 (inner_dim=4096), T=42240 at stage-2
# Audio path: 32 heads, head_dim=64, T~100 (production uses pytorch default
#             attn because T < skip_under_seq_len=1024 sage threshold).
VIDEO_BATCH = 1
VIDEO_HEADS = 32
VIDEO_HEAD_DIM = 128
VIDEO_T = 42240

AUDIO_BATCH = 1
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_T = 100

# Test methodology.
WARMUP_RUNS = 3
TIMED_RUNS = 5

# PASS thresholds. Audio recovered >50% means concurrent dispatch
# actually overlaps; video slowdown <5% means no L2 thrash from the
# concurrent issue (the v0.6 sage_ffn precedent died on >5% slowdown).
THRESHOLD_AUDIO_RECOVERED = 0.50
THRESHOLD_VIDEO_SLOWDOWN = 0.05


def make_qkv(batch: int, heads: int, T: int, head_dim: int, device, seed: int = 0):
    """Random Q/K/V at canonical sage (B, H, T, D) layout."""
    g = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, heads, T, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=g)
    return q, k, v


def sage_video_call(q, k, v):
    """sage fp8++ on the video Q/K/V. Returns out tensor."""
    return sageattn_qk_int8_pv_fp8_cuda(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        pv_accum_dtype="fp32+fp16",
    )


def torch_audio_call(q, k, v):
    """PyTorch default SDPA on the audio Q/K/V. Returns out tensor."""
    return F.scaled_dot_product_attention(q, k, v)


def probe_pytorch_backend(q, k, v) -> None:
    """Log which SDPA backend pytorch picks at the audio test shape.

    Tests each forced backend + the default (pytorch's own choice). Records
    which are supported and which raise. Useful for the spike report so we
    know what audio is being compared against; sm89 + bf16 + head_dim=64
    + T=100 typically lands on FlashAttention but worth confirming.
    """
    backends = [
        ("flash",         SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math",          SDPBackend.MATH),
        ("cudnn",         SDPBackend.CUDNN_ATTENTION),
    ]
    print("PyTorch SDPA backend availability at audio shape:")
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            print(f"  {name:14s}: supported")
        except RuntimeError as e:
            print(f"  {name:14s}: unsupported ({type(e).__name__})")
    # default (no context override) -- pytorch picks at runtime
    _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    print(f"  {'default':14s}: pytorch picks at runtime; one of the above")


def elapsed_ms(e0: torch.cuda.Event, e1: torch.cuda.Event) -> float:
    """ms between two cuda Events (after synchronize())."""
    return e0.elapsed_time(e1)


def measure_sequential(video_qkv, audio_qkv) -> tuple[float, float, float]:
    """Baseline: video then audio on the default stream. No overlap.

    Returns (video_ms, audio_ms, total_ms) as medians over TIMED_RUNS.
    """
    video_q, video_k, video_v = video_qkv
    audio_q, audio_k, audio_v = audio_qkv

    video_samples, audio_samples, total_samples = [], [], []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        e_video_end = torch.cuda.Event(enable_timing=True)
        e_audio_end = torch.cuda.Event(enable_timing=True)

        e_start.record()
        with torch.cuda.nvtx.range("seq_video"):
            _ = sage_video_call(video_q, video_k, video_v)
        e_video_end.record()
        with torch.cuda.nvtx.range("seq_audio"):
            _ = torch_audio_call(audio_q, audio_k, audio_v)
        e_audio_end.record()
        torch.cuda.synchronize()

        video_samples.append(elapsed_ms(e_start, e_video_end))
        audio_samples.append(elapsed_ms(e_video_end, e_audio_end))
        total_samples.append(elapsed_ms(e_start, e_audio_end))

    video_samples.sort(); audio_samples.sort(); total_samples.sort()
    mid = TIMED_RUNS // 2
    return video_samples[mid], audio_samples[mid], total_samples[mid]


def measure_concurrent(video_qkv, audio_qkv) -> tuple[float, float, float]:
    """Concurrent: video on stream s_video, audio on stream s_audio.

    Returns (video_ms, audio_end_offset_ms, total_ms). The audio reading
    is `e_start (default stream) -> e_audio_end (s_audio)` and INCLUDES
    the s_audio queue-wait while video saturates SMs -- it is NOT a
    pure audio kernel duration. Total is max(video_end, audio_end_offset)
    since both streams can complete independently; for the shape pair
    (video ~108ms, audio kernel ~13us) the video end dominates.
    """
    video_q, video_k, video_v = video_qkv
    audio_q, audio_k, audio_v = audio_qkv

    s_video = torch.cuda.Stream()
    s_audio = torch.cuda.Stream()

    video_samples, audio_offset_samples, total_samples = [], [], []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()

        e_start = torch.cuda.Event(enable_timing=True)
        e_video_end = torch.cuda.Event(enable_timing=True)
        e_audio_end = torch.cuda.Event(enable_timing=True)

        e_start.record()

        with torch.cuda.stream(s_video):
            with torch.cuda.nvtx.range("conc_video"):
                _ = sage_video_call(video_q, video_k, video_v)
            e_video_end.record(s_video)

        with torch.cuda.stream(s_audio):
            with torch.cuda.nvtx.range("conc_audio"):
                _ = torch_audio_call(audio_q, audio_k, audio_v)
            e_audio_end.record(s_audio)

        torch.cuda.synchronize()

        video_ms = elapsed_ms(e_start, e_video_end)
        audio_end_offset_ms = elapsed_ms(e_start, e_audio_end)
        total_ms = max(video_ms, audio_end_offset_ms)

        video_samples.append(video_ms)
        audio_offset_samples.append(audio_end_offset_ms)
        total_samples.append(total_ms)

    video_samples.sort(); audio_offset_samples.sort(); total_samples.sort()
    mid = TIMED_RUNS // 2
    return video_samples[mid], audio_offset_samples[mid], total_samples[mid]


def correctness_sanity(video_qkv, audio_qkv) -> None:
    """Rtol check across stream-context: does sage produce the same output
    when called from default stream vs from a side stream? Load-bearing on
    the submodule spike (caught a stable 0.02 mean_rtol drift on the video
    side that's deterministic but not bit-identical).
    """
    video_q, video_k, video_v = video_qkv
    audio_q, audio_k, audio_v = audio_qkv

    v_seq = sage_video_call(video_q, video_k, video_v)
    a_seq = torch_audio_call(audio_q, audio_k, audio_v)
    torch.cuda.synchronize()

    s_video = torch.cuda.Stream()
    s_audio = torch.cuda.Stream()
    with torch.cuda.stream(s_video):
        v_conc = sage_video_call(video_q, video_k, video_v)
    with torch.cuda.stream(s_audio):
        a_conc = torch_audio_call(audio_q, audio_k, audio_v)
    torch.cuda.synchronize()

    v_mean_rtol, _, _, _ = accuracy_metrics(v_conc, v_seq)
    a_mean_rtol, _, _, _ = accuracy_metrics(a_conc, a_seq)
    print(f"  video seq vs concurrent: mean_rtol = {v_mean_rtol:.6f}")
    print(f"  audio seq vs concurrent: mean_rtol = {a_mean_rtol:.6f}")
    # bf16 deterministic kernels should give bit-identical; some L2 reordering
    # under concurrent issue may produce tiny float drift. 1e-5 is paranoid.
    if v_mean_rtol > 1e-5 or a_mean_rtol > 1e-5:
        print("  WARN: drift between sequential and concurrent outputs. Investigate.")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    import triton
    cap = torch.cuda.get_device_capability(0)
    print(f"torch: {torch.__version__}  triton: {triton.__version__}  CUDA: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]})")
    print()
    print(f"Video pair: B={VIDEO_BATCH} H={VIDEO_HEADS} T={VIDEO_T} D={VIDEO_HEAD_DIM}  (sage fp8++)")
    print(f"Audio pair: B={AUDIO_BATCH} H={AUDIO_HEADS} T={AUDIO_T} D={AUDIO_HEAD_DIM}  (pytorch SDPA)")
    print()

    device = torch.device("cuda")

    print("Allocating Q/K/V tensors...")
    video_qkv = make_qkv(VIDEO_BATCH, VIDEO_HEADS, VIDEO_T, VIDEO_HEAD_DIM, device, seed=0)
    audio_qkv = make_qkv(AUDIO_BATCH, AUDIO_HEADS, AUDIO_T, AUDIO_HEAD_DIM, device, seed=1)
    print(f"  video Q/K/V: {tuple(video_qkv[0].shape)} bf16, ~{video_qkv[0].numel() * 2 / 1e6:.0f} MB each")
    print(f"  audio Q/K/V: {tuple(audio_qkv[0].shape)} bf16")
    print()

    probe_pytorch_backend(*audio_qkv)
    print()

    print(f"Warmup ({WARMUP_RUNS} runs each side, absorbs sage autotune + pytorch dispatcher pick)...")
    for _ in range(WARMUP_RUNS):
        _ = sage_video_call(*video_qkv)
        _ = torch_audio_call(*audio_qkv)
    torch.cuda.synchronize()
    print()

    print("Correctness sanity (sequential vs concurrent outputs should match):")
    correctness_sanity(video_qkv, audio_qkv)
    print()

    print(f"Sequential baseline ({TIMED_RUNS} timed runs, median reported):")
    seq_video_ms, seq_audio_ms, seq_total_ms = measure_sequential(video_qkv, audio_qkv)
    print(f"  video: {seq_video_ms:7.2f} ms")
    print(f"  audio: {seq_audio_ms:7.2f} ms")
    print(f"  total: {seq_total_ms:7.2f} ms")
    print()

    print(f"Concurrent dispatch ({TIMED_RUNS} timed runs, median reported):")
    conc_video_ms, conc_audio_end_offset_ms, conc_total_ms = measure_concurrent(video_qkv, audio_qkv)
    print(f"  video:             {conc_video_ms:7.2f} ms")
    print(f"  audio_end_offset:  {conc_audio_end_offset_ms:7.2f} ms  (includes s_audio queue-wait, NOT kernel time)")
    print(f"  total:             {conc_total_ms:7.2f} ms")
    print()

    audio_recovered = (seq_total_ms - conc_total_ms) / seq_audio_ms if seq_audio_ms > 0 else 0.0
    video_slowdown = (conc_video_ms - seq_video_ms) / seq_video_ms if seq_video_ms > 0 else 0.0

    print("=== Verdict ===")
    print(f"audio wall-time recovered: {audio_recovered:7.1%}  (threshold >{THRESHOLD_AUDIO_RECOVERED:.0%})")
    print(f"video slowdown:            {video_slowdown:7.1%}  (threshold <{THRESHOLD_VIDEO_SLOWDOWN:.0%})")

    audio_pass = audio_recovered > THRESHOLD_AUDIO_RECOVERED
    video_pass = video_slowdown < THRESHOLD_VIDEO_SLOWDOWN

    if audio_pass and video_pass:
        print()
        print("SPIKE PASSED. Concurrent-dispatch parallelism is viable on this hardware.")
        print("Next move: scope the dispatch-layer infrastructure work (~2 weeks).")
        print("  - sage side: stream-aware kernel arg + don't auto-sync")
        print("  - consumer side: monkey-patch block forward to issue concurrent streams")
        return 0

    print()
    print("SPIKE FAILED. Drop the parallelism axis or revisit assumptions.")
    if not audio_pass:
        print(f"  audio overlap insufficient: {audio_recovered:.1%} < {THRESHOLD_AUDIO_RECOVERED:.0%}")
    if not video_pass:
        print(f"  video slowdown too large:   {video_slowdown:.1%} >= {THRESHOLD_VIDEO_SLOWDOWN:.0%}")
    print()
    print("If this fails, try Window B (V2A || A2V, both sage-touched) before")
    print("dropping the parallelism axis entirely -- the dependency-graph case")
    print("for B is distinct from A's sage/pytorch mix.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
