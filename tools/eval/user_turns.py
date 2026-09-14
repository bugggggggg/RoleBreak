"""Where an eval runner gets the audio for a user turn.

Every runner needs the same three things per turn — a clip to send, the text that
clip actually says, and a name for where it came from — so that resolution lives
here rather than in each of them, for the same reason the shard format lives in
:mod:`tools.wds` and the version naming in :mod:`tools.audio_version`.

Clips come from the WebDataset shards ``tools/synthesize_user_turns.py`` writes,
one per example (``<audio-dir>/<version>/<example>.tar``), whose sample keys carry
the index into ``Example.turns``; a turn that authored its own ``audio`` keeps it
and the shard goes unread. A shard clip travels as the WAV *bytes* — the backends
decode those directly — so a replay unpacks nothing to disk and the run records a
reference into the shard rather than a path to a file that never existed.

A shard is a snapshot and can lag the example it is replayed against, so a clip
reports the text it actually says: what the model heard, hence what belongs in
``run.jsonl``, with the mismatch surfaced by :func:`report_drift`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rolebreak.examples import Turn
from tools.wds import Sample, read_turns, shard_path


@dataclass(frozen=True)
class UserClip:
    """The audio for one user turn: what to send, what it says, what to call it."""

    source: str | bytes  # a path for an authored turn, WAV bytes for a shard clip
    text: str | None  # what the clip actually says
    ref: str  # where it came from, for the log and run.jsonl


def has_shard(audio_dir: str, example_name: str) -> bool:
    """Was ``example_name`` synthesized into this audio version?

    A shard is moved to its final path only once its example finished (see
    ``tools/synthesize_user_turns.py``), so its absence means this batch never
    spoke that example — a runner drops it from the run rather than failing the
    whole batch on the first example missing from an audio version.
    """
    return shard_path(audio_dir, example_name).exists()


def load_synth_audio(audio_dir: str, example_name: str) -> dict[int, Sample]:
    """Map a 0-based turn index -> shard sample for ``example_name``.

    Reads ``<audio-dir>/<example>.tar`` (``tools/synthesize_user_turns.py``'s
    default ``wds`` output), where ``audio_dir`` is one resolved audio *version*
    directory — see :func:`tools.audio_version.read_dir`.

    Passing ``audio_dir`` says the turns are meant to be spoken, so a missing shard
    is a bug, not a silent text-only run.
    """
    path = shard_path(audio_dir, example_name)
    assert path.exists(), f"no shard at {path} — synthesize with tools/synthesize_user_turns.py"
    return read_turns(path)


def synthesized_clip(turn: Turn, sample: Sample | None) -> UserClip:
    """The clip ``sample`` holds for ``turn``.

    Takes the sample rather than the shard and an index to look it up in, and
    requires one: every turn an example authors has text, so synthesis wrote it a
    sample, and the caller's ``synth_audio.get(i)`` comes up empty only for a turn
    past the tear in a shard an interrupted run cut short.

    The clip's text is the one the sample carries, falling back to the turn's for
    a sample whose ``.txt`` was lost to that same tear.
    """
    assert sample is not None
    assert sample.wav is not None  # read_turns only yields samples with audio
    spoken = sample.text if sample.text is not None else sample.meta.get("text", turn.text)
    return UserClip(source=sample.wav, text=spoken, ref=sample.ref)


def resolve_turn_audio(turn: Turn, sample: Sample | None) -> UserClip:
    """The clip to send for ``turn``: its own ``audio`` else ``sample``'s.

    An authored turn wins and travels as a path; a turn that authored audio and no
    text is skipped by synthesis, so the shard has nothing for it — hence the
    optional ``sample``, which is what the caller's ``synth_audio.get(i)`` hands
    back for exactly those turns. A turn with both is synthesized, and the
    authored path still wins.

    Past that branch a turn has text and no audio, so synthesis wrote it a sample
    and the only way to have none is a shard an interrupted run cut short — which
    :func:`synthesized_clip` asserts on, rather than skipping the turn and
    misaligning the rest of the conversation against the authored rubric.
    """
    if turn.audio is not None:
        return UserClip(source=str(turn.audio), text=turn.text, ref=str(turn.audio))
    return synthesized_clip(turn, sample)


def report_drift(turns: Iterable[Turn], clips: Iterable[UserClip | None], emit) -> None:
    """Flag turns whose clip says something other than what the example authors.

    Not fatal — the clip is still what the model heard — but worth a line in the
    log, since the rubric was written against the authored text.
    """
    for i, (turn, clip) in enumerate(zip(turns, clips, strict=True)):
        if clip is None or turn.text is None or clip.text is None:
            continue
        if clip.text != turn.text:
            emit(f"! turn {i} clip says {clip.text!r}, example authors {turn.text!r} — re-synthesize this example")
