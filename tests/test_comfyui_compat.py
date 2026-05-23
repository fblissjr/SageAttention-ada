#!/usr/bin/env python3
"""Test the ComfyUI fp8 storage probe utility.

`extract_fp8_weight_and_scale(linear)` resolves fp8 weight + scale
across known ComfyUI conventions. This test exercises the probe
order (modern QuantizedTensor first, then legacy attrs) using mock
objects that simulate each storage layout without requiring ComfyUI
to be installed.

Reference: internal/design/comfyui_fp8_storage_conventions.md

Standalone script (no pytest); run via $VIRTUAL_ENV/bin/python.
Does not require CUDA -- mocks use CPU tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sageattention import extract_fp8_weight_and_scale


# --- Mocks for ComfyUI storage layouts ---


class _MockParams:
    """Simulates comfy_kitchen's `Params` object that holds the
    per-tensor scale on a QuantizedTensor."""

    def __init__(self, scale):
        self.scale = scale


class _MockQuantizedTensor:
    """Minimal stand-in for comfy_kitchen `QuantizedTensor`.

    Real implementation subclasses torch.Tensor; for the probe test
    we only need the `_qdata` attribute (raw fp8 storage) and one of
    (`_params`, `layout_params`) carrying a `.scale` attribute. The
    probe does duck-typing via `getattr`, so a non-Tensor object
    with the right attrs is sufficient."""

    def __init__(self, qdata, scale, public_alias=True, raw_alias=True):
        self._qdata = qdata
        if public_alias:
            self.layout_params = _MockParams(scale)
        if raw_alias:
            self._params = _MockParams(scale)


class _MockLinear:
    """Stand-in for a torch.nn.Linear with arbitrary additional attrs.
    Used to model both legacy `scale_weight` and older `weight_scale`
    direct-attribute conventions."""

    def __init__(self, weight, **extra_attrs):
        self.weight = weight
        for name, value in extra_attrs.items():
            setattr(self, name, value)


def _make_fp8_tensor(shape=(8, 16)):
    return torch.randn(*shape).clamp(-3, 3).to(torch.float8_e4m3fn)


def _make_scale_tensor():
    return torch.tensor(0.0123, dtype=torch.float32)


# --- Tests ---


def test_modern_quantized_tensor_layout_params_path():
    """QuantizedTensor with `layout_params.scale` (public alias)
    should win; path label names the public alias."""
    qdata = _make_fp8_tensor()
    scale = _make_scale_tensor()
    weight = _MockQuantizedTensor(qdata, scale, public_alias=True, raw_alias=False)
    linear = _MockLinear(weight)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None, "expected probe hit on layout_params path"
    w, s, label = result
    assert w is qdata, "expected raw _qdata, not the wrapper"
    assert s is scale
    assert "layout_params" in label, f"expected layout_params in label, got {label!r}"


def test_modern_quantized_tensor_private_params_only():
    """QuantizedTensor with `_params.scale` but no `layout_params`
    (raw-only storage) should still match; path label names _params."""
    qdata = _make_fp8_tensor()
    scale = _make_scale_tensor()
    weight = _MockQuantizedTensor(qdata, scale, public_alias=False, raw_alias=True)
    linear = _MockLinear(weight)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None
    w, s, label = result
    assert w is qdata
    assert s is scale
    assert "_params" in label


def test_modern_prefers_layout_params_over_private_params():
    """When BOTH `layout_params` and `_params` exist (the common case
    in comfy_kitchen v0.2.8+), probe must prefer the public alias."""
    qdata = _make_fp8_tensor()
    scale = _make_scale_tensor()
    weight = _MockQuantizedTensor(qdata, scale, public_alias=True, raw_alias=True)
    linear = _MockLinear(weight)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None
    _, _, label = result
    assert "layout_params" in label, f"expected layout_params priority, got {label!r}"


def test_legacy_scale_weight_attr():
    """Older fp8_ops convention: `linear.weight` is raw fp8,
    `linear.scale_weight` is the per-tensor scale."""
    weight = _make_fp8_tensor()
    scale = _make_scale_tensor()
    linear = _MockLinear(weight, scale_weight=scale)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None
    w, s, label = result
    assert w is weight
    assert s is scale
    assert label == "scale_weight"


def test_older_weight_scale_attr():
    """Some custom-node patches expose the scale as `weight_scale`
    rather than `scale_weight`. Probe should catch this too."""
    weight = _make_fp8_tensor()
    scale = _make_scale_tensor()
    linear = _MockLinear(weight, weight_scale=scale)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None
    w, s, label = result
    assert w is weight
    assert s is scale
    assert label == "weight_scale"


def test_legacy_prefers_scale_weight_over_weight_scale():
    """If both attrs exist on a legacy Linear, probe must pick the
    canonical `scale_weight` (the fp8_ops convention)."""
    weight = _make_fp8_tensor()
    scale_a = _make_scale_tensor()
    scale_b = _make_scale_tensor()
    linear = _MockLinear(weight, scale_weight=scale_a, weight_scale=scale_b)
    result = extract_fp8_weight_and_scale(linear)
    assert result is not None
    _, s, label = result
    assert label == "scale_weight"
    assert s is scale_a


def test_returns_none_when_weight_is_not_fp8_and_no_qdata():
    """Bf16 weight with no QuantizedTensor wrapper and no scale
    attributes -> probe returns None. Consumer is responsible for
    raising / falling back."""
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    linear = _MockLinear(weight)
    result = extract_fp8_weight_and_scale(linear)
    assert result is None, f"expected None for bf16 weight; got {result!r}"


def test_returns_none_when_linear_has_no_weight_attr():
    """Object that doesn't expose `weight` at all -> None."""

    class _NoWeight:
        pass

    result = extract_fp8_weight_and_scale(_NoWeight())
    assert result is None


def test_returns_none_when_quantized_tensor_has_qdata_but_no_scale():
    """Malformed QuantizedTensor: `_qdata` present but neither
    `layout_params` nor `_params` carries a tensor scale. Probe
    should not match this -- consumer's fp8 storage is broken."""
    qdata = _make_fp8_tensor()
    weight = _MockQuantizedTensor(qdata, scale=None, public_alias=True, raw_alias=True)
    linear = _MockLinear(weight)
    result = extract_fp8_weight_and_scale(linear)
    assert result is None, f"expected None for malformed QuantizedTensor; got {result!r}"


def test_returns_none_when_fp8_weight_has_no_scale_attrs():
    """Raw fp8 weight with no scale_weight or weight_scale attribute
    -> None. Consumer's fp8 layer is missing its scale."""
    weight = _make_fp8_tensor()
    linear = _MockLinear(weight)  # no scale_weight, no weight_scale
    result = extract_fp8_weight_and_scale(linear)
    assert result is None


def main() -> int:
    cases = [
        test_modern_quantized_tensor_layout_params_path,
        test_modern_quantized_tensor_private_params_only,
        test_modern_prefers_layout_params_over_private_params,
        test_legacy_scale_weight_attr,
        test_older_weight_scale_attr,
        test_legacy_prefers_scale_weight_over_weight_scale,
        test_returns_none_when_weight_is_not_fp8_and_no_qdata,
        test_returns_none_when_linear_has_no_weight_attr,
        test_returns_none_when_quantized_tensor_has_qdata_but_no_scale,
        test_returns_none_when_fp8_weight_has_no_scale_attrs,
    ]
    failures = 0
    for fn in cases:
        try:
            print(f"{fn.__name__}:")
            fn()
            print("  ok")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures}/{len(cases)} cases failed.")
        return 1
    print(f"\nAll {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
