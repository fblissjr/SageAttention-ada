"""Cross-version compatibility shims for ComfyUI fp8 storage conventions.

ComfyUI's fp8 weight storage has shifted multiple times. A consumer
that wants to extract the raw fp8 weight + scale from a quantized
`Linear` (e.g. to pass into `sage_ffn` directly) has to probe several
conventions. This module centralizes that probe so every consumer
doesn't reinvent it.

See `internal/design/comfyui_fp8_storage_conventions.md` for the
detailed lookup table and source citations.

Probe order, modern first:
  1. Modern QuantizedTensor with public `layout_params.scale`
     (comfy_kitchen v0.2.8+; preferred surface).
  2. Modern QuantizedTensor with private `_params.scale` (raw alias
     of layout_params; fallback when the public alias is absent).
  3. Legacy `Linear.scale_weight` attribute (older fp8_ops convention).
  4. Older `Linear.weight_scale` attribute (some custom-node patches).

Returns `(raw_weight_fp8_tensor, scale_tensor, path_label)` on hit,
`None` on miss. The probe is intentionally non-raising so consumers
can decide whether a miss should fall back to a stock path, log a
warning, or raise an informative error.
"""

from __future__ import annotations

import torch

# Module returns the path label using the same vocabulary the
# scoping doc uses, so a grep across logs / scoping docs / code
# converges to the same name.
_PATH_LAYOUT_PARAMS = "weight.layout_params.scale"
_PATH_PRIVATE_PARAMS = "weight._params.scale"
_PATH_SCALE_WEIGHT = "scale_weight"
_PATH_WEIGHT_SCALE = "weight_scale"


def extract_fp8_weight_and_scale(
    linear: object,
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    """Probe `linear` for fp8 weight + scale across known ComfyUI conventions.

    Args:
        linear: A `torch.nn.Linear`-like object (duck-typed; only
            `weight` and the relevant scale attribute matter). The
            probe tolerates arbitrary input and returns `None` on
            miss; callers can decide whether to raise.

    Returns:
        `(raw_weight_fp8_tensor, scale_tensor, path_label)` on hit.

        `raw_weight_fp8_tensor` is the actual fp8 storage. If the
        weight was a `QuantizedTensor` wrapper, this is the unwrapped
        `weight._qdata`, NOT the wrapper -- sage's kernels assert
        `dtype == float8_e4m3fn` and the wrapper won't satisfy that.

        `scale_tensor` is the per-tensor (0-d) or per-channel scale
        as stored. Caller is responsible for `.item()`-ing it down to
        a Python float if the kernel signature requires that (e.g.
        `sage_ffn`'s `w1_scale: float`).

        `path_label` is a short string identifying which convention
        matched. Useful for one-shot install-time logging so the
        operator can see which storage layout was detected.

        Returns `None` if no known convention matches (weight is bf16
        without a scale; QuantizedTensor wrapper with neither
        `_params` nor `layout_params` carrying a tensor scale; etc.).
    """
    weight = getattr(linear, "weight", None)
    if weight is None:
        return None

    # Modern QuantizedTensor path: `weight` is a wrapper holding raw
    # fp8 storage on `_qdata` plus a Params object carrying the scale.
    # Public `layout_params` alias is preferred; fall back to the raw
    # `_params` only if `layout_params` is absent or has no scale.
    qdata = getattr(weight, "_qdata", None)
    if isinstance(qdata, torch.Tensor):
        for attr, label in (
            ("layout_params", _PATH_LAYOUT_PARAMS),
            ("_params", _PATH_PRIVATE_PARAMS),
        ):
            params = getattr(weight, attr, None)
            if params is None:
                continue
            scale = getattr(params, "scale", None)
            if isinstance(scale, torch.Tensor):
                return qdata, scale, label
        return None

    # Legacy paths: `weight` is itself the raw fp8 storage (no wrapper).
    # The scale lives as a sibling attribute on the Linear.
    if not isinstance(weight, torch.Tensor):
        return None
    if weight.dtype != torch.float8_e4m3fn:
        return None

    for attr, label in (
        ("scale_weight", _PATH_SCALE_WEIGHT),
        ("weight_scale", _PATH_WEIGHT_SCALE),
    ):
        scale = getattr(linear, attr, None)
        if isinstance(scale, torch.Tensor):
            return weight, scale, label

    return None
