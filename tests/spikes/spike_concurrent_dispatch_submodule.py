"""(C) Concurrent-dispatch parallelism spike, path B: full sub-module pairs.

Refinement of `spike_concurrent_dispatch.py` (path A / bare-SDPA).
The bare-SDPA spike measured 0.01 ms per audio-side call, vs the
tracer-grounded production cost of ~1.70 ms per `audio_attn1` call.
The 170x gap means the bare-SDPA test wasn't exercising the same
dispatcher-scheduling-overhead landscape that production hits.

This spike fixes that by testing **two full sub-module mockups**
against each other:

- Video sub-module: `RMSNorm + AdaLN modulation + qkv_proj + sage fp8++
  attention + out_proj + gated residual` at video stage-2 shape
  (B=1, T=42240, hidden=4096, h=32, d=128).
- Audio sub-module: same shape composition at audio shape (B=1, T=100,
  hidden=2048, h=32, d=64) with `F.scaled_dot_product_attention` forced
  to FlashAttention (matches production -- pytorch's default pick at
  this shape).

Structure mirrors `BasicAVTransformerBlock.forward` in LTX-2's
`ltx_core/model/transformer/transformer.py:187-376`. Not weight-loaded
from the production checkpoint -- randomly-initialized weights are
fine for timing + concurrent-dispatch overlap evidence.

Decision gates:
- audio-side wall-time recovered: > 50%
- video-side slowdown from concurrent issue: < 5%  (no L2 thrash;
  recall v0.6 sage_ffn died on exactly this contention mechanism)

Run with sage installed in $VIRTUAL_ENV:
    ${VIRTUAL_ENV}/bin/python tests/spikes/spike_concurrent_dispatch_submodule.py

For kernel-level overlap evidence:
    nsys profile -o /tmp/spike_path_b ${VIRTUAL_ENV}/bin/python \\
        tests/spikes/spike_concurrent_dispatch_submodule.py
NVTX ranges ("seq_video_block", "seq_audio_block", "conc_video_block",
"conc_audio_block") show in nsys timeline view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_sageattn_ltx_shapes import accuracy_metrics  # type: ignore[import-not-found]

from sageattention import sageattn_qk_int8_pv_fp8_cuda


# Shape constants from tests/test_sageattn_ltx_shapes.py:63-64 architecture
# notes for LTX 2.3:
#   Video: num_attention_heads=32, attention_head_dim=128 -> hidden=4096
#   Audio: audio_num_attention_heads=32, audio_attention_head_dim=64 -> hidden=2048
VIDEO_BATCH = 1
VIDEO_HEADS = 32
VIDEO_HEAD_DIM = 128
VIDEO_HIDDEN = VIDEO_HEADS * VIDEO_HEAD_DIM
VIDEO_T = 42240

AUDIO_BATCH = 1
AUDIO_HEADS = 32
AUDIO_HEAD_DIM = 64
AUDIO_HIDDEN = AUDIO_HEADS * AUDIO_HEAD_DIM
AUDIO_T = 100

WARMUP_RUNS = 3
TIMED_RUNS = 5

THRESHOLD_AUDIO_RECOVERED = 0.50
THRESHOLD_VIDEO_SLOWDOWN = 0.05


class VideoSubModuleMockup(nn.Module):
    """Mock of LTX 2.3 video self-attn sub-module within `BasicAVTransformerBlock`.

    Composition mirrors transformer.py:210-246 (video MSA section): AdaLN
    modulation slice -> RMSNorm -> qkv projection -> sage fp8++ attention
    -> output projection -> gated residual. Skips text cross-attn -- this
    spike only tests the self-attn-side sub-module.
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.scale_shift = nn.Parameter(torch.randn(3, hidden, device=device, dtype=dtype) * 0.02)
        self.norm = nn.RMSNorm(hidden, device=device, dtype=dtype)
        self.q_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        residual = x
        ada = self.scale_shift + timesteps
        scale, shift, gate = ada[0], ada[1], ada[2]
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale[None, None, :]) + shift[None, None, :]
        q = self.q_proj(x_mod)
        k = self.k_proj(x_mod)
        v = self.v_proj(x_mod)
        B, T, _ = q.shape
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        attn_out = sageattn_qk_int8_pv_fp8_cuda(
            q, k, v,
            tensor_layout="HND",
            is_causal=False,
            pv_accum_dtype="fp32+fp16",
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden)
        out = self.out_proj(attn_out)
        out = out * gate[None, None, :]
        return residual + out


class AudioSubModuleMockup(nn.Module):
    """Mock of LTX 2.3 audio self-attn sub-module.

    Same composition as video but at audio shape (hidden=2048, head_dim=64)
    and uses pytorch's default SDPA (FlashAttention forced for repro,
    matches production behavior under skip_under_seq_len=1024).
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, device, dtype=torch.bfloat16):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.scale_shift = nn.Parameter(torch.randn(3, hidden, device=device, dtype=dtype) * 0.02)
        self.norm = nn.RMSNorm(hidden, device=device, dtype=dtype)
        self.q_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden, hidden, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        residual = x
        ada = self.scale_shift + timesteps
        scale, shift, gate = ada[0], ada[1], ada[2]
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale[None, None, :]) + shift[None, None, :]
        q = self.q_proj(x_mod)
        k = self.k_proj(x_mod)
        v = self.v_proj(x_mod)
        B, T, _ = q.shape
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        # Force FlashAttention for reproducibility (pytorch's heuristic pick
        # at this shape; lock explicitly because the dispatcher's choice can
        # shift across torch versions).
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden)
        out = self.out_proj(attn_out)
        out = out * gate[None, None, :]
        return residual + out


def make_block_inputs(B, T, hidden, device, dtype=torch.bfloat16, seed=0):
    """Random activation tensor + AdaLN modulation table."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, T, hidden, device=device, dtype=dtype, generator=g)
    timesteps = torch.randn(3, hidden, device=device, dtype=dtype, generator=g) * 0.01
    return x, timesteps


def elapsed_ms(e0: torch.cuda.Event, e1: torch.cuda.Event) -> float:
    return e0.elapsed_time(e1)


def measure_sequential(video_block, audio_block, video_x, video_ts, audio_x, audio_ts):
    """Both sub-modules on default stream, video then audio. Median of TIMED_RUNS."""
    video_samples, audio_samples, total_samples = [], [], []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        e_start = torch.cuda.Event(enable_timing=True)
        e_video_end = torch.cuda.Event(enable_timing=True)
        e_audio_end = torch.cuda.Event(enable_timing=True)

        e_start.record()
        with torch.cuda.nvtx.range("seq_video_block"):
            _ = video_block(video_x, video_ts)
        e_video_end.record()
        with torch.cuda.nvtx.range("seq_audio_block"):
            _ = audio_block(audio_x, audio_ts)
        e_audio_end.record()
        torch.cuda.synchronize()

        video_samples.append(elapsed_ms(e_start, e_video_end))
        audio_samples.append(elapsed_ms(e_video_end, e_audio_end))
        total_samples.append(elapsed_ms(e_start, e_audio_end))

    video_samples.sort(); audio_samples.sort(); total_samples.sort()
    mid = TIMED_RUNS // 2
    return video_samples[mid], audio_samples[mid], total_samples[mid]


def measure_concurrent(video_block, audio_block, video_x, video_ts, audio_x, audio_ts):
    """Video on s_video, audio on s_audio. Median of TIMED_RUNS.

    Returns (video_ms, audio_end_offset_ms, total_ms). The audio reading
    is `e_start (default stream) -> e_audio_end (s_audio)` and INCLUDES
    the s_audio queue-wait while video saturates SMs -- it is NOT a pure
    audio sub-module duration. Total is max(video_end, audio_end_offset).
    """
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
            with torch.cuda.nvtx.range("conc_video_block"):
                _ = video_block(video_x, video_ts)
            e_video_end.record(s_video)
        with torch.cuda.stream(s_audio):
            with torch.cuda.nvtx.range("conc_audio_block"):
                _ = audio_block(audio_x, audio_ts)
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


def correctness_sanity(video_block, audio_block, video_x, video_ts, audio_x, audio_ts):
    """Does sage produce the same output when called from default stream
    vs from a side stream? Caught a stable ~0.02 mean_rtol drift on the
    video side that's deterministic across runs but not bit-identical
    -- material for whether sage's CUDA kernel is fully stream-safe or
    whether the consumer wrapper needs to declare an extended rtol
    budget for concurrent-dispatch mode.
    """
    v_seq = video_block(video_x, video_ts)
    a_seq = audio_block(audio_x, audio_ts)
    torch.cuda.synchronize()

    s_video = torch.cuda.Stream()
    s_audio = torch.cuda.Stream()
    with torch.cuda.stream(s_video):
        v_conc = video_block(video_x, video_ts)
    with torch.cuda.stream(s_audio):
        a_conc = audio_block(audio_x, audio_ts)
    torch.cuda.synchronize()

    v_mean_rtol, _, _, _ = accuracy_metrics(v_conc, v_seq)
    a_mean_rtol, _, _, _ = accuracy_metrics(a_conc, a_seq)
    # view-as-uint16 because torch.equal treats NaN != NaN, while the
    # stream-safety question is whether bit patterns match regardless of
    # NaN semantics. Random mockup weights can produce NaN positions
    # naturally; what matters is whether the side-stream path produces
    # the same bits as the default-stream path.
    v_bits_eq = torch.equal(v_seq.view(torch.uint16), v_conc.view(torch.uint16))
    a_bits_eq = torch.equal(a_seq.view(torch.uint16), a_conc.view(torch.uint16))
    print(f"  video seq vs concurrent: bits_equal={v_bits_eq}  mean_rtol={v_mean_rtol:.6f}")
    print(f"  audio seq vs concurrent: bits_equal={a_bits_eq}  mean_rtol={a_mean_rtol:.6f}")
    if v_mean_rtol > 1e-5 or a_mean_rtol > 1e-5:
        print("  WARN: drift between sequential and concurrent. Investigate.")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2

    import triton
    cap = torch.cuda.get_device_capability(0)
    print(f"torch: {torch.__version__}  triton: {triton.__version__}  CUDA: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]})")
    print()
    print(f"Video sub-module: B={VIDEO_BATCH} T={VIDEO_T} hidden={VIDEO_HIDDEN} ({VIDEO_HEADS}h x {VIDEO_HEAD_DIM}d)")
    print(f"Audio sub-module: B={AUDIO_BATCH} T={AUDIO_T} hidden={AUDIO_HIDDEN} ({AUDIO_HEADS}h x {AUDIO_HEAD_DIM}d)")
    print(f"Audio backend forced: FlashAttention (production default at this shape)")
    print()

    device = torch.device("cuda")

    print("Building sub-module mockups...")
    video_block = VideoSubModuleMockup(VIDEO_HIDDEN, VIDEO_HEADS, VIDEO_HEAD_DIM, device)
    audio_block = AudioSubModuleMockup(AUDIO_HIDDEN, AUDIO_HEADS, AUDIO_HEAD_DIM, device)
    for p in video_block.parameters():
        p.requires_grad_(False)
    for p in audio_block.parameters():
        p.requires_grad_(False)

    print("Allocating inputs...")
    video_x, video_ts = make_block_inputs(VIDEO_BATCH, VIDEO_T, VIDEO_HIDDEN, device, seed=0)
    audio_x, audio_ts = make_block_inputs(AUDIO_BATCH, AUDIO_T, AUDIO_HIDDEN, device, seed=1)
    print(f"  video x: {tuple(video_x.shape)} bf16, ~{video_x.numel() * 2 / 1e6:.0f} MB")
    print(f"  audio x: {tuple(audio_x.shape)} bf16")
    print()

    # All sub-module calls below are inference-only; inference_mode is
    # stricter than no_grad (also drops version-counter tracking) and
    # matches the consumer path where these blocks run under sampler.
    with torch.inference_mode():
        print(f"Warmup ({WARMUP_RUNS} runs each side, absorbs sage autotune + FA dispatch pick)...")
        for _ in range(WARMUP_RUNS):
            _ = video_block(video_x, video_ts)
            _ = audio_block(audio_x, audio_ts)
        torch.cuda.synchronize()
        print()

        print("Correctness sanity (does sage produce the same output under different stream context?):")
        correctness_sanity(video_block, audio_block, video_x, video_ts, audio_x, audio_ts)
        print()

        print(f"Sequential baseline ({TIMED_RUNS} timed runs, median):")
        seq_video_ms, seq_audio_ms, seq_total_ms = measure_sequential(
            video_block, audio_block, video_x, video_ts, audio_x, audio_ts
        )
        print(f"  video: {seq_video_ms:7.2f} ms")
        print(f"  audio: {seq_audio_ms:7.2f} ms")
        print(f"  total: {seq_total_ms:7.2f} ms")
        print()

        print(f"Concurrent dispatch ({TIMED_RUNS} timed runs, median):")
        conc_video_ms, conc_audio_end_offset_ms, conc_total_ms = measure_concurrent(
            video_block, audio_block, video_x, video_ts, audio_x, audio_ts
        )
        print(f"  video:             {conc_video_ms:7.2f} ms")
        print(f"  audio_end_offset:  {conc_audio_end_offset_ms:7.2f} ms  (includes s_audio queue-wait, NOT sub-module duration)")
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
        print("SPIKE PASSED. Concurrent-dispatch parallelism is viable at full sub-module granularity.")
        print("Next move: scope the dispatch-layer infrastructure work (~2 weeks).")
        print("  - sage side: stream-aware kernel arg + don't auto-sync")
        print("  - consumer side: monkey-patch block forward to issue concurrent streams")
        return 0

    print()
    print("SPIKE FAILED at full sub-module granularity. Reconsider parallelism axis.")
    if not audio_pass:
        print(f"  audio overlap insufficient: {audio_recovered:.1%} < {THRESHOLD_AUDIO_RECOVERED:.0%}")
    if not video_pass:
        print(f"  video slowdown too large:   {video_slowdown:.1%} >= {THRESHOLD_VIDEO_SLOWDOWN:.0%}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
