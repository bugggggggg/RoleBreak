"""Versioned layout of the synthesized user-turn audio (``data/audio``).

``tools/synthesize_user_turns.py`` writes every run into a ``<emotion>`` directory
under its ``--out-dir``; the eval runners read the user turns back out of one of
those. Both sides go through this module so a writer and a reader can never
disagree about the naming: :func:`write_dir` picks where a synthesis run lands,
:func:`read_dir` picks which version a run is replayed from (the newest one
unless a version is named).

An eval run is named after the audio version it heard, so the naming of the
generation dirs lives here too: :func:`generation_dir` picks between ``neutral``
and the ``neutral.1``, ``neutral.2``, ... re-generations against those same clips.
"""

from __future__ import annotations

import re
from pathlib import Path

import click

# The emotions a synthesis run can be delivered in — one version directory each,
# re-synthesized deliveries of one emotion numbered within it (``neutral.1``).
VERSIONS = ("neutral", "angry", "sad", "happy")

# The emotion synthesis writes unless told otherwise, and the one a reader falls
# back to. Set here rather than on the command line so the choice keeps showing up
# in the diff instead of in one person's shell history.
DEFAULT_VERSION = "neutral"

_VERSION_RE = re.compile(rf"^({'|'.join(VERSIONS)})(?:\.(\d+))?$")


def parse_version(name: str) -> tuple[str, int]:
    """``"neutral.2"`` -> ``("neutral", 2)``, ``"sad"`` -> ``("sad", 0)``.

    The bare emotion is revision 0 — it is the first delivery, just written before
    anyone needed a second — so the two spellings sort together. Anything else
    raises ``ValueError``.
    """
    m = _VERSION_RE.match(name)
    if m is None:
        raise ValueError(f"Not a version: {name!r} (expected one of {', '.join(VERSIONS)}, optionally '.N').")
    return m[1], int(m[2] or 0)


def format_version(emotion: str, revision: int) -> str:
    """``("neutral", 0)`` -> ``"neutral"``; ``("neutral", 2)`` -> ``"neutral.2"``."""
    return emotion if revision == 0 else f"{emotion}.{revision}"


def _sort_key(version: tuple[str, int]) -> tuple[int, int]:
    """Order versions by :data:`VERSIONS`, then by revision within an emotion."""
    emotion, revision = version
    return VERSIONS.index(emotion), revision


def list_versions(root: Path) -> list[tuple[str, int]]:
    """Every version directory directly under ``root``, in :data:`VERSIONS` order."""
    found: list[tuple[str, int]] = []
    for p in root.iterdir() if root.is_dir() else ():
        if not p.is_dir():
            continue
        try:
            found.append(parse_version(p.name))
        except ValueError:
            continue  # some other directory that is not a version
    return sorted(found, key=_sort_key)


def write_dir(root: Path, *, bump: bool = False, version: str | None = None) -> Path:
    """Pick the version directory a synthesis run writes to.

    ``version`` names one outright (``"angry"``, ``"neutral.2"``) — the escape hatch
    for synthesizing another emotion, re-opening an older delivery or claiming a
    specific one, so it need not exist yet. It cannot be combined with ``bump``,
    which is about picking a version for you.

    Otherwise only revisions of :data:`DEFAULT_VERSION` are considered: without
    ``bump`` we reuse the newest one so a re-run tops up what is already there, with
    it we claim the next. An emotion with nothing under it yet starts at the bare
    name either way.
    """
    if version is not None:
        if bump:
            raise click.UsageError("Pass either --version or --bump-version, not both.")
        try:
            emotion, revision = parse_version(version)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        return root / format_version(emotion, revision)
    revisions = [rev for emotion, rev in list_versions(root) if emotion == DEFAULT_VERSION]
    if not revisions:
        revision = 0
    else:
        revision = max(revisions) + 1 if bump else max(revisions)
    return root / format_version(DEFAULT_VERSION, revision)


def label(path: str | Path) -> str | None:
    """The version label of a resolved version dir, else ``None``.

    Lets a reader name the batch it replayed — e.g. to keep one audio version's
    outputs apart from another's — while still returning ``None`` for the
    unversioned ``root`` that :func:`read_dir` falls back to, so callers can tell
    "no version to speak of" from a real one.
    """
    try:
        emotion, revision = parse_version(Path(path).name)
    except ValueError:
        return None
    return format_version(emotion, revision)


def read_dir(root: str | Path, version: str | None = None) -> Path:
    """Pick the version directory clips are read from, under audio root ``root``.

    ``version`` names one explicitly (``"angry"``, ``"neutral.2"``, or ``"latest"``);
    without it the newest revision of :data:`DEFAULT_VERSION` wins, falling back to
    the last emotion present in :data:`VERSIONS` order when that one was never
    synthesized — emotions have no ordering between them, so there is no newer and
    older to compare. A ``root`` with no version directories in it is used as-is,
    which keeps a hand-assembled or pre-versioning tree of clips readable;
    ``version="none"`` forces that reading even when versions do exist, for clips
    sitting directly under ``root``.
    """
    root = Path(root)
    versions = list_versions(root)
    if version == "none":
        return root
    if version is not None and version != "latest":
        try:
            emotion, revision = parse_version(version)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        dest = root / format_version(emotion, revision)
        if not dest.is_dir():
            known = ", ".join(format_version(e, r) for e, r in versions) or "none"
            raise click.UsageError(f"No audio version at {dest} (found under {root}: {known}).")
        return dest
    if not versions:
        return root
    emotion = DEFAULT_VERSION if any(e == DEFAULT_VERSION for e, _ in versions) else versions[-1][0]
    revision = max(r for e, r in versions if e == emotion)
    return root / format_version(emotion, revision)


def list_generation_versions(root: Path, audio_version: str) -> list[int]:
    """The generation revisions of ``audio_version`` present under ``root``, oldest first.

    A run dir is named after the audio version it heard, optionally with a trailing
    number for a re-generation against those same clips: ``neutral``, ``neutral.1``,
    ``neutral.2``. The bare ``neutral`` is revision 0 — it is the first generation,
    just written before anyone needed a second — so the two spellings sort together.
    """
    base = format_version(*parse_version(audio_version))
    revision_re = re.compile(rf"^{re.escape(base)}\.(\d+)$")
    found: list[int] = []
    for p in root.glob(f"{base}*"):
        if not p.is_dir():
            continue
        if p.name == base:
            found.append(0)
            continue
        m = revision_re.match(p.name)
        if m is not None:
            found.append(int(m[1]))
    return sorted(found)


def generation_dir(root: Path, audio_version: str, *, bump: bool = False) -> Path:
    """Pick the run dir a generation against ``audio_version`` writes to, under ``root``.

    Without ``bump`` the newest revision wins, so a re-run resumes into the dir it
    was already filling. With it we claim the next one — ``neutral`` -> ``neutral.1``
    — which keeps a second generation against the same clips beside the first
    instead of on top of it, the way a newer audio version already lands beside an
    older one. An audio version with no run dir yet starts at the bare
    ``audio_version`` either way: there is nothing there to keep apart.
    """
    base = format_version(*parse_version(audio_version))
    revisions = list_generation_versions(root, audio_version)
    if not revisions:
        return root / base
    revision = max(revisions) + 1 if bump else max(revisions)
    return root / (base if revision == 0 else f"{base}.{revision}")
