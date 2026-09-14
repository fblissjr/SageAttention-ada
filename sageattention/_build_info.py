"""Which build of this fork is running -- for anyone stamping a measurement.

`__version__` is `2.2.0` and has been for months. It comes from `setup.py`
and it does not move when the kernels are rebuilt, so it cannot distinguish
one build of this fork from another. That is fine for a dependency pin and
useless for provenance, and the difference bit a downstream consumer: their
blind-evaluation record stamped this library as `2.2.0`, then a full rebuild
under a different language standard, a different compiler and different
framework headers produced a record that read `2.2.0` as well. The identifier
could not distinguish the builds it named, so the question "were those renders
made by the kernels running now" had to be answered from this repo's git log
instead of from their record.

`build_info()` closes that. On an editable install -- which is how this fork
ships to consumers -- the source revision is the thing that identifies the
kernels, so report it alongside the version.

Deliberately NOT computed at import. This package is imported on every
consumer startup and a subprocess there is rude; the cost belongs to whoever
asks. Cached after the first call, since the answer cannot change within a
process without someone editing the checkout underneath it.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, or return None for any reason at all.

    Provenance must never be the thing that breaks a consumer's startup: no
    git on the box, a wheel install rather than a checkout, an unreadable
    directory, a hung filesystem. All of it degrades to "unknown".
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@lru_cache(maxsize=1)
def build_info() -> dict:
    """Identify the running build.

    Returns a dict with:
      version   the dist version, e.g. "2.2.0". Never None.
      revision  short commit sha of the source tree, or None if this is not
                a git checkout (a wheel install, typically).
      dirty     True if TRACKED files are modified, False if not, None if
                unknown. Untracked files are deliberately NOT dirty: a
                consumer's own scratch file sitting in the checkout does not
                change the kernels, and reporting it as a modified build
                sends a reader looking for source edits that do not exist.
      describe  "<version> @ <sha>[-dirty]", or just "<version>" with no
                checkout. The one-line form to put in a log.
      element_offset_bits
                Width of the global element offsets the installed fused
                CUDA quant build forms: 64 from v0.7.17 on, 32 for any
                build without the attribute. Distinguishes a widened build
                from a stale one in a record, since `revision` names the
                source and not the compiled artifact.
    """
    from . import __version__ as version
    from .quant import ELEMENT_OFFSET_BITS as element_offset_bits

    pkg_dir = Path(__file__).resolve().parent
    # --short=12, not bare --short: git auto-scales the abbreviation as a repo
    # grows, so bare --short can stamp the SAME commit at different widths in
    # two records. A consumer embeds this verbatim in dated records it does not
    # rewrite, so the width is pinned. Still a prefix of the full sha, so it
    # stays resolvable by git either way.
    sha = _git(["rev-parse", "--short=12", "HEAD"], pkg_dir)

    dirty: bool | None = None
    if sha is not None:
        porcelain = _git(["status", "--porcelain", "--untracked-files=no"], pkg_dir)
        if porcelain is not None:
            dirty = bool(porcelain)

    describe = version
    if sha:
        describe = f"{version} @ {sha}" + ("-dirty" if dirty else "")

    return {
        "version": version,
        "revision": sha,
        "dirty": dirty,
        "describe": describe,
        "element_offset_bits": element_offset_bits,
    }
