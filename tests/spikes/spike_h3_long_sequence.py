"""Does the sm89 kernel still behave past the int32 offset boundary?

Every sage measurement on file stops at S=41,822 -- MiniMax H3's fl2va shape
at the node's default 124 frames. The longest legal request, 345 frames at
the same 1344x768 canvas, packs S=104,030 rows, which is 2.5x longer and, in
the layout production actually uses, past the point where int32 element
offsets wrap.

345 is the ceiling, not 362. The reference implementation checks duration
*after* the frame count snaps to the 17n+5 grid and rejects anything over
15.0 s (`modular_pipelines/minimax_h3/before_denoise.py`), so 362 frames is
15.083 s and illegal; 345 (14.375 s) is the largest count on the grid. An
earlier version of this file swept 362 and called it "a shape a user
actually requested" -- that came from ComfyUI's node tooltip, which claims a
trained range of ~124-362 and has no ceiling of its own. 362 is kept in the
sweep below as an out-of-distribution row, because the kernel question is
about S and not about training windows, but it is not a production shape and
no rendered output should be taken at it.

Two things need separating at that size, and the synthetic bench answers
neither today:

  correctness -- q/k/v in an H3 block are three views of one fused qkv
    projection buffer, so `stride_seq` is 3*heads*head_dim = 21504 rather
    than 7168. The wrap therefore arrives at S=99,864 in the layout that
    matters and at S=299,593 in a contiguous copy. v0.7.0 fixed this with a
    per-launch USE_I64 specialization; this spike is the disprove-test for
    that fix at a shape a user actually requested. The NHD failure signature
    is a silently all-zero tail, so a whole-tensor rtol can pass while the
    end of the clip is garbage -- the tail is scored separately.

  cost -- USE_I64 has never been timed. It was verified not to change
    wall-clock at S=41,822, but that shape does not trigger it, so what the
    int64 path costs when it does fire is unmeasured.

Both layouts are built at every size so the fused-view arm can be read
against a contiguous control at the same S.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import sageattention
from sageattention.triton._int_offsets import needs_int64_offsets
from test_sageattn_ltx_shapes import accuracy_metrics, time_and_vram

HEADS = 56
HEAD_DIM = 128
WIDTH, HEIGHT = 1344, 768
FRAME_COUNTS = [124, 200, 300, 345, 362]  # 345 is the legal ceiling; 362 is OOD

# Rows the DiT actually attends over, from comfy_extras/nodes_minimax_h3.py
# geometry. Kept in step with tests/spikes/spike_minimax_h3_shapes.py.
FPS, AUDIO_LATENT_FPS = 24, 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def packed_seq_len(length, width=WIDTH, height=HEIGHT, prompt_tokens=64):
    frame_count = align_frame_count(max(5, length))
    latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    frame_rows = (height // 16 // 2) * (width // 16 // 2)
    return prompt_tokens + audio_t * 2 + latent_t * frame_rows, frame_count


def make_qkv(seq, fused):
    """(q, k, v) at H3's config.

    fused=True reproduces what an H3 block hands the kernel: one
    `qkv_proj` output split three ways, so each view carries the 3x
    sequence stride that moves the overflow boundary down to S=99,864.
    fused=False is the contiguous control at the same S.
    """
    if fused:
        buf = torch.randn(1, seq, 3, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        q, k, v = (buf[:, :, i] for i in range(3))
        return q, k, v, buf
    t = [torch.randn(1, seq, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    return t[0], t[1], t[2], None


def reference(q, k, v):
    # NHD -> HND for SDPA, and back, so the comparison is layout-for-layout.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
    return o.transpose(1, 2)


def chunked_mean_rtol(actual, expect, chunk=8192):
    """`accuracy_metrics`' mean rtol without its full-size fp32 temporaries.

    The shared helper upcasts both tensors and builds three more of the same
    size, which is ~15 GiB at S=109,126 and OOMs next to the inputs it is
    scoring. Slicing the sequence keeps the identical symmetric-denominator
    formula -- verified equal to `accuracy_metrics(...)[0]` at the sizes
    where both fit.
    """
    total, count = 0.0, 0
    for i in range(0, actual.shape[1], chunk):
        a = actual[:, i:i + chunk].float()
        e = expect[:, i:i + chunk].float()
        diff = (a - e).abs()
        eps = torch.tensor(torch.finfo(a.dtype).eps, device=a.device, dtype=a.dtype)
        rdiff = diff / torch.maximum(torch.maximum(a.abs(), e.abs()), eps)
        total += rdiff.sum().item()
        count += rdiff.numel()
        del a, e, diff, rdiff
    return total / count


def tail_report(actual, expect, rows=4096):
    """Score the last `rows` separately -- an int32 wrap in NHD zeroes the
    tail while leaving enough of the tensor intact to pass a mean rtol."""
    a, e = actual[:, -rows:], expect[:, -rows:]
    zero_frac = (a.float().abs().sum(dim=-1) == 0).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        a.float().flatten(), e.float().flatten(), dim=0
    ).item()
    return accuracy_metrics(a, e)[0], cos, zero_frac


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  sage {getattr(sageattention, '__version__', '?')}")
    print(f"canvas {WIDTH}x{HEIGHT}, heads {HEADS}, head_dim {HEAD_DIM}\n")

    hdr = (f"{'frames':>6} {'S':>8} {'layout':>7} {'i64':>4} "
           f"{'sage ms':>9} {'flash ms':>9} {'ratio':>6} "
           f"{'rtol':>7} {'tail rtol':>9} {'tail cos':>9} {'tail 0s':>8} {'MiB':>7}")
    print(hdr)
    print("-" * len(hdr))

    for frames in FRAME_COUNTS:
        seq, aligned = packed_seq_len(frames)
        for fused in (True, False):
            q, k, v, buf = make_qkv(seq, fused)
            # Same predicate the wrapper uses to pick the specialization.
            q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
            i64 = needs_int64_offsets(q, q_int8, tensor_layout="NHD", blk=128)
            del q_int8

            with torch.inference_mode():
                torch.cuda.nvtx.range_push(f"ref_s{seq}_fused{int(fused)}")
                ref = reference(q, k, v)
                torch.cuda.nvtx.range_pop()

                def run_sage():
                    return sageattention.sageattn(
                        q, k, v, tensor_layout="NHD", is_causal=False,
                        smooth_k=False, pv_accum_dtype="fp32+fp16",
                    )

                torch.cuda.nvtx.range_push(f"sage_s{seq}_fused{int(fused)}")
                out = run_sage()
                sage_ms, mib = time_and_vram(run_sage, warmup=1, runs=3)
                torch.cuda.nvtx.range_pop()

                flash_ms, _ = time_and_vram(lambda: reference(q, k, v), warmup=1, runs=3)

                rtol = chunked_mean_rtol(out, ref)
                t_rtol, t_cos, t_zero = tail_report(out, ref)

            print(f"{aligned:>6} {seq:>8,} {'fused' if fused else 'contig':>7} "
                  f"{'yes' if i64 else 'no':>4} "
                  f"{sage_ms:>9.1f} {flash_ms:>9.1f} {flash_ms/sage_ms:>5.2f}x "
                  f"{rtol:>7.4f} {t_rtol:>9.4f} {t_cos:>9.4f} {t_zero:>7.1%} {mib:>7.0f}")

            del q, k, v, buf, ref, out
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
