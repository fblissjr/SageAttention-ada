"""Shared int32-overflow bound for the Triton INT8 quant kernels.

Triton's `program_id` is int32 and stride arguments are passed as int32,
so `off_b * stride_z + off_h * stride_h + offs_n * stride_n + offs_k`
is evaluated in int32 and wraps once a tensor exceeds 2**31 elements
(4 GiB at bf16). The failure is not uniform across layouts:

  NHD -- `stride_n` is heads*head_dim, so the row term crosses first and
         the wrapped stores land outside the output. Measured result is a
         silently all-zero tail.
  HND -- `stride_h` is seq_len*head_dim, so the head term crosses first.
         Measured result is an illegal memory access.

Both are reachable at MiniMax H3's attention config (heads 56, head_dim
128), and where they become reachable depends on how the caller allocated
q/k/v, not on sequence length alone. `max_element_offset` reads the real
strides and so gets this right; quoting a single row count does not:

  contiguous NHD -- `stride_n` is heads*head_dim = 7168, crossing near
                    299,593 rows.
  fused QKV view -- `stride_n` is 3*heads*head_dim = 21504, crossing near
                    99,864 rows. Three times sooner, and it is the layout
                    a DiT block actually produces from one qkv projection.

The fused figure is the one that matters in practice. H3 at 362 frames is
S=109,126, which is comfortably inside a 300k budget and past the real
crossing; v0.7.1 measured the specialization firing exactly there. Any
check written against the contiguous number stays silent on the layout
that overflows first, so an absent warning is not evidence of clearance.

Rather than pay int64 address arithmetic everywhere, the kernels take a
`USE_I64` constexpr and the wrappers call `needs_int64_offsets` to pick a
specialization per launch.

Upstream (`thu-ml/SageAttention`) has this bug in all three quant
modules; see CHANGELOG.md for the divergence record.
"""

INT32_MAX = 2**31 - 1


def max_element_offset(t, tensor_layout, blk):
    """Largest element offset the quant kernels can form for `t`.

    The kernels build pointers for the whole padded grid and only mask the
    load, so the row index runs to `ceil(seq/blk)*blk - 1` rather than
    `seq-1`. Batch and head indices contribute their own stride terms, so
    a tensor that is safe at batch 1 need not be safe at batch 4.
    """
    if tensor_layout == "HND":
        b, h, seq, head_dim = t.shape
        stride_z, stride_h, stride_n = t.stride(0), t.stride(1), t.stride(2)
    elif tensor_layout == "NHD":
        b, seq, h, head_dim = t.shape
        stride_z, stride_h, stride_n = t.stride(0), t.stride(2), t.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    padded_rows = (seq + blk - 1) // blk * blk
    return ((b - 1) * stride_z + (h - 1) * stride_h
            + (padded_rows - 1) * stride_n + head_dim)


def needs_int64_offsets(*tensors, tensor_layout, blk):
    """Whether any of `tensors` forces int64 address arithmetic.

    Split out of the wrappers so the boundary is testable on meta tensors,
    without allocating the multi-GiB inputs that actually trigger it.
    """
    return any(
        max_element_offset(t, tensor_layout, blk) > INT32_MAX for t in tensors
    )
