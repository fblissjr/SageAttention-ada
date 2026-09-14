#!/usr/bin/env python3
"""Pin `sageattention.build_info()`'s contract, because a consumer embeds it.

A downstream consumer stamps this into dated evaluation records it does not
rewrite, so the shape of the return value is a promise and not an
implementation detail. This file is what makes that promise enforceable
rather than prose -- the same reason `tests/test_dispatched_kernel_telemetry.py`
exists for the dispatcher's `attn_mask` parameter.

Why it exists at all: `__version__` is `2.2.0` and does not move when the
kernels are rebuilt, so it cannot distinguish one build of this fork from
another. A consumer's record identified this library by that constant across a
rebuild under a changed language standard, compiler and framework headers.

Standalone-script style per the rest of `tests/`. No CUDA needed.
Run directly:  ${VIRTUAL_ENV}/bin/python tests/test_build_info_contract.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sageattention


KEYS = {"version", "revision", "dirty", "describe", "element_offset_bits"}


def test_exported_from_package():
    assert hasattr(sageattention, "build_info"), \
        "build_info must stay exported from the package root; a consumer imports it"


def test_key_set_is_exactly_the_contract():
    info = sageattention.build_info()
    assert set(info) == KEYS, (
        f"build_info() keys are a downstream contract. Got {sorted(info)}, "
        f"expected {sorted(KEYS)}. Adding a key is a compatible change and this "
        f"assertion should be widened deliberately; removing or renaming one is "
        f"not -- see docs/downstream_symbols.md."
    )


def test_types():
    info = sageattention.build_info()
    assert isinstance(info["version"], str) and info["version"], "version must be a non-empty str"
    assert info["revision"] is None or isinstance(info["revision"], str)
    assert info["dirty"] is None or isinstance(info["dirty"], bool)
    assert isinstance(info["describe"], str) and info["describe"]
    assert info["element_offset_bits"] in (32, 64), info["element_offset_bits"]


def test_revision_is_a_pinned_width_prefix_of_head():
    """Same commit must stamp the same string every time, in any repo size."""
    info = sageattention.build_info()
    if info["revision"] is None:
        print("    (skip: not a git checkout)")
        return
    assert len(info["revision"]) == 12, \
        f"revision width is pinned at 12; got {len(info['revision'])}. A consumer " \
        f"embeds this in records it does not rewrite, so the width must not drift."
    full = subprocess.run(
        ["git", "-C", str(Path(sageattention.__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert full.startswith(info["revision"]), \
        f"revision {info['revision']} is not a prefix of HEAD {full}"


def test_dirty_ignores_untracked_files():
    """The semantics a consumer inherits: untracked files are NOT dirty.

    This is the specific false positive that motivated the whole entry point --
    a consumer's own scratch file in the checkout reported as a modified build,
    sending a reader hunting source edits that do not exist. Verified against
    git rather than asserted: build_info's dirty must agree with a
    tracked-files-only status, whatever the untracked set looks like.
    """
    info = sageattention.build_info()
    if info["dirty"] is None:
        print("    (skip: not a git checkout)")
        return
    pkg = str(Path(sageattention.__file__).resolve().parent)
    tracked_only = subprocess.run(
        ["git", "-C", pkg, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert info["dirty"] == bool(tracked_only), (
        f"dirty={info['dirty']} disagrees with tracked-only status "
        f"({'non-empty' if tracked_only else 'empty'}). If this fires with "
        f"untracked files present, the --untracked-files=no flag was lost."
    )


def test_describe_composes_the_others():
    info = sageattention.build_info()
    assert info["describe"].startswith(info["version"])
    if info["revision"]:
        assert info["revision"] in info["describe"]
        assert info["describe"].endswith("-dirty") == bool(info["dirty"])


TESTS = [
    test_exported_from_package,
    test_key_set_is_exactly_the_contract,
    test_types,
    test_revision_is_a_pinned_width_prefix_of_head,
    test_dirty_ignores_untracked_files,
    test_describe_composes_the_others,
]


def main():
    failures = []
    for t in TESTS:
        try:
            t()
        except AssertionError as exc:
            failures.append(t.__name__)
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            failures.append(t.__name__)
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {t.__name__}")
    print()
    if failures:
        print(f"FAIL {len(failures)}/{len(TESTS)}")
        sys.exit(1)
    print(f"PASS {len(TESTS)}/{len(TESTS)}")


if __name__ == "__main__":
    main()
