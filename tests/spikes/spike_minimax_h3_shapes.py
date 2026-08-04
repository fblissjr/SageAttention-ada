"""MiniMax H3 attention-shape probe: does the sm89 kernel set fit this model?

MiniMax H3 (Comfy-Org/MiniMax-H3) is a single-stream packed-token AV DiT.
Every DiT block runs ONE unmasked self-attention over the whole packed
sequence [text | cond | audio | video] -- there is no cross-attention and
no attn_mask anywhere (comfy/ldm/minimax/model.py:181 passes mask=None).

Config read from the shipped checkpoint headers + comfy/model_detection.py:
  hidden 5376, 50 DiT blocks + 2 token-refiner blocks,
  heads 56, head_dim 128 (MHA, not GQA), ffn 14336 SwiGLU, qkv bias=False.

Sequence length is derived here from comfy_extras/nodes_minimax_h3.py so the
numbers track the node's own geometry rules rather than a hand-copied constant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from test_sageattn_ltx_shapes import (
    Shape,
    print_warnings_footer,
    run_shape_sweep,
)

FPS = 24
AUDIO_LATENT_FPS = 40
HEADS = 56
HEAD_DIM = 128


def video_latent_t(frame_count):
    # nodes_minimax_h3.py:35
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def packed_seq_len(width, height, length, n_keyframes=0, prompt_tokens=64):
    """Rows in the DiT's packed sequence for one (canvas, duration) request.

    Mirrors PackedLayout in comfy/ldm/minimax/model.py:297. Video rows dominate;
    text length is the only estimated term (Qwen3-VL vision blocks contribute
    one token per 32x32 pixel patch, same 1008-row grid as a video frame at
    1344x768).
    """
    frame_count = align_frame_count(max(5, length))
    latent_t = video_latent_t(frame_count)
    audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    lat_h, lat_w = height // 16, width // 16
    frame_rows = (lat_h // 2) * (lat_w // 2)

    vision_tokens = (height // 16) * (width // 16) // 4  # Qwen3-VL patch16 + merge2
    text_len = prompt_tokens + n_keyframes * (vision_tokens + 8)

    return dict(
        text=text_len,
        cond=n_keyframes * frame_rows,
        audio=audio_t * 2,
        video=latent_t * frame_rows,
        total=text_len + n_keyframes * frame_rows + audio_t * 2 + latent_t * frame_rows,
        frame_count=frame_count,
        latent_t=latent_t,
        frame_rows=frame_rows,
    )


T2VA = packed_seq_len(1344, 768, 124, n_keyframes=0)
FL2VA = packed_seq_len(1344, 768, 124, n_keyframes=2)
T2VA_SHORT = packed_seq_len(768, 768, 124, n_keyframes=0)

SHAPES = [
    # t2va at the node's default canvas (1344x768) and default 124 frames.
    Shape("mmh3_t2va_selfattn_1344x768_124f", 1, HEADS, T2VA["total"], T2VA["total"], HEAD_DIM, False),
    # fl2va: two keyframes add both cond rows and Qwen vision tokens.
    Shape("mmh3_fl2va_selfattn_1344x768_124f", 1, HEADS, FL2VA["total"], FL2VA["total"], HEAD_DIM, False),
    # Square canvas -- the cheapest shape a user is likely to run.
    Shape("mmh3_t2va_selfattn_768x768_124f", 1, HEADS, T2VA_SHORT["total"], T2VA_SHORT["total"], HEAD_DIM, False),
    # Token refiner: 2 blocks over the text span only, same heads/head_dim.
    Shape("mmh3_token_refiner_fl2va", 1, HEADS, FL2VA["text"], FL2VA["text"], HEAD_DIM, False),
]


def main():
    for name, seg in (("t2va", T2VA), ("fl2va", FL2VA), ("768sq", T2VA_SHORT)):
        print(
            f"# {name}: frames={seg['frame_count']} latent_t={seg['latent_t']} "
            f"frame_rows={seg['frame_rows']} | text={seg['text']} cond={seg['cond']} "
            f"audio={seg['audio']} video={seg['video']} -> S={seg['total']}"
        )
    print()
    warnings, _ = run_shape_sweep(SHAPES, dtype=torch.bfloat16)
    print_warnings_footer(warnings)


if __name__ == "__main__":
    main()
