"""A finished run as a WebDataset shard — one ``<example>.tar`` per example.

The tar counterpart of :mod:`rolebreak.runlog`'s directory format. A run holds
the same things either way; what changes is that the assistant's WAVs travel
*inside* the file that carries the metadata instead of beside it, so one example
is one artifact to copy, upload, or hand to a dataloader — and a run of a few
hundred examples is a few hundred files rather than a few thousand.

Layout — a shard is a WebDataset shard, i.e. tar members grouped into *samples*
by the part of their name before the first dot (the *key*), the same convention
the synthesized user turns are stored under (``tools/wds.py``)::

    conan/run.json    # the run header: example, name, persona, model, voice
    conan/000.json    # turn 0's record — the object run.jsonl nests under "turns"
    conan/000.wav     # turn 0's spoken reply (absent when the turn produced none)
    conan/001.json
    conan/001.wav
    ...

Keys are ``<example>/NNN`` with ``NNN`` the 0-based index into ``Example.turns``,
so a run shard's samples line up one-for-one with the user-turn shard that was
spoken into it.

Nothing downstream has to learn a second schema: the members are exactly the
objects :mod:`rolebreak.runlog` writes (:func:`~rolebreak.runlog.run_header` and
:func:`~rolebreak.runlog.turn_record`), so :func:`record` recovers the very
``run.jsonl`` row a directory run would have written and :func:`load_run`
rebuilds the same :class:`~rolebreak.types.Transcript` for the metrics.

A shard is written to ``<example>.tar.partial`` and moved onto its final name
only once the run finishes — as ``tools/synthesize_user_turns.py`` does with the
user-turn shards — so a shard at its final path is always a *complete* run. A
runner indexes each one into a consolidated ``runs.jsonl`` as it lands
(:meth:`RunShardWriter.row`), which is what a resumed run reads to know which
examples it need not replay.

Written with the stdlib ``tarfile`` and a fixed member mtime: the output is a
plain uncompressed tar that ``webdataset`` reads as-is, and re-running the same
turns gives the same bytes.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

from rolebreak.runlog import output_audio, run_header, transcript_from_record, turn_record
from rolebreak.types import Emotion, OutputAudio, Rubric, Transcript

SUFFIX = ".tar"
PARTIAL_SUFFIX = ".partial"  # a shard still being written
HEADER = "run"  # the header sample's key suffix: <example>/run.json

# Fixed member mtime, as in tools/wds.py: a replay that produces the same turns
# should not churn the bytes.
MTIME = 0


def shard_path(root: str | Path, example_name: str) -> Path:
    """Where ``example_name``'s run shard lives under an out-dir ``root``."""
    return Path(root) / f"{example_name}{SUFFIX}"


def sample_key(example_name: str, index: int) -> str:
    """The key grouping one turn's members: ``<example>/NNN`` (0-based index)."""
    return f"{example_name}/{index:03d}"


def wav_member(example_name: str, index: int) -> str:
    """The member name a turn's reply audio is stored under."""
    return f"{sample_key(example_name, index)}.wav"


def wav_locator(example_name: str, index: int) -> str:
    """Where a turn's audio is, seen from the directory the shard sits in.

    ``<example>.tar#<example>/NNN.wav`` — the form a turn record's ``wav`` takes in
    a shard run, the same locator the user-turn shards already use for
    ``user_audio``. A record read out of the tar therefore says where its audio is
    from outside the tar too, which is what lets one be copied into a consolidated
    ``runs.jsonl`` unchanged and still resolve (:func:`read_wavs`).
    """
    return f"{example_name}{SUFFIX}#{wav_member(example_name, index)}"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
class RunShardWriter:
    """Writer for a run's ``<example>.tar`` — the shard counterpart of
    :class:`~rolebreak.runlog.RunWriter`.

    Same shape of use: header fields on construction, one :meth:`add_turn` per
    exchange, :meth:`close` (or the context manager) to finish. Unlike ``RunWriter``
    it streams — a turn's WAV is in the tar as soon as it is added, rather than the
    whole run being buffered — so the file is written under ``.partial`` and moved
    onto its real name at close; an interrupted run leaves the ``.partial`` behind
    for the next one to overwrite and never claims the shard path.

    ``add_turn`` takes the reply's WAV *bytes* where ``RunWriter`` takes a filename:
    the shard holds the audio itself. The record it writes still names the member
    the audio landed in, so a turn read back out of the tar looks like one read out
    of a run dir.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        example: str,
        persona: str,
        model: str | None = None,
        voice: str | None = None,
    ) -> None:
        self._example = example
        self.path = shard_path(root, example)
        self._tmp = self.path.with_name(self.path.name + PARTIAL_SUFFIX)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tar = tarfile.open(self._tmp, "w")
        self._closed = False
        self._header = run_header(example, persona, model, voice)
        self._records: list[dict[str, Any]] = []
        self._write(f"{example}/{HEADER}.json", _json_bytes(self._header))

    def _write(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = MTIME
        self._tar.addfile(info, BytesIO(data))

    def add_turn(
        self,
        index: int,
        *,
        user_text: str | None,
        assistant_text: str,
        user_audio: str | Path | None = None,
        wav: bytes | None = None,
        accepted_emotions: list[Emotion] | None = None,
        expected_rubric: list[Rubric] | None = None,
        latency: float | None = None,
    ) -> None:
        """Write one exchange: its record, and the reply's WAV bytes if it spoke."""
        rec = turn_record(
            index,
            user_text=user_text,
            assistant_text=assistant_text,
            user_audio=user_audio,
            wav=wav_locator(self._example, index) if wav is not None else None,
            accepted_emotions=accepted_emotions,
            expected_rubric=expected_rubric,
            latency=latency,
        )
        self._records.append(rec)
        self._write(f"{sample_key(self._example, index)}.json", _json_bytes(rec))
        if wav is not None:
            self._write(wav_member(self._example, index), wav)

    def row(self) -> dict[str, Any]:
        """The run as one ``runs.jsonl`` row — the header with the turns so far.

        What a shard holds, minus the audio: the index a consolidated
        ``runs.jsonl`` is made of. Each turn's ``wav`` is already a locator into
        this shard, so the row resolves from the directory the shard sits in
        without being rewritten.
        """
        return {**self._header, "turns": list(self._records)}

    def close(self) -> None:
        """Finish the tar and publish it at its real path (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._tar.close()
        self._tmp.replace(self.path)

    def abort(self) -> None:
        """Close the tar *without* publishing it (idempotent).

        What the exiting context manager does when the run raised: the shard path
        stays empty, so the example still counts as unfinished and the next run
        replays it instead of skipping a conversation that stops mid-way.
        """
        if self._closed:
            return
        self._closed = True
        self._tar.close()

    def __enter__(self) -> "RunShardWriter":
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        self.close() if exc_type is None else self.abort()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_members(path: str | Path) -> Iterator[tuple[str, bytes]]:
    """Stream ``(member name, bytes)`` out of a shard, in shard order.

    Single-pass, like the user-turn reader: a shard cut short mid-member — a
    ``.partial`` from a run that died — ends the read at the tear instead of
    raising, so every complete member before it is still yielded.
    """
    with tarfile.open(path, "r|") as tar:
        try:
            for info in tar:
                if not info.isfile():
                    continue
                fh = tar.extractfile(info)
                if fh is None:
                    continue
                yield info.name, fh.read()
        except tarfile.ReadError:
            return


def read_wavs(path: str | Path) -> dict[str, bytes]:
    """Every WAV in a shard, by member name — what a :func:`wav_locator` names."""
    return {name: data for name, data in read_members(path) if name.endswith(".wav")}


def split_locator(wav: str) -> tuple[str, str] | None:
    """``<shard>.tar#<member>`` -> ``(shard, member)``; ``None`` for a plain path."""
    shard, sep, member = wav.partition("#")
    return (shard, member) if sep and shard.endswith(SUFFIX) else None


def record(path: str | Path) -> tuple[dict[str, Any], dict[int, bytes]]:
    """A shard's run row (as ``run.jsonl`` holds it) plus its WAVs by turn index.

    The row is the header with every turn's record nested under ``turns``, ordered
    by turn index — so a shard can be exported back to the directory format, or
    read by anything that already understands a run row.
    """
    header: dict[str, Any] = {}
    turns: dict[int, dict[str, Any]] = {}
    wavs: dict[int, bytes] = {}
    for name, data in read_members(path):
        stem, _, ext = name.rpartition(".")
        leaf = stem.rsplit("/", 1)[-1]
        if leaf == HEADER and ext == "json":
            header = json.loads(data)
        elif ext == "json":
            rec = json.loads(data)
            turns[rec.get("index", len(turns))] = rec
        elif ext == "wav" and leaf.isdigit():
            wavs[int(leaf)] = data
    assert header, f"{path} has no {HEADER}.json — not a run shard (or one cut short before its header)."
    return {**header, "turns": [turns[i] for i in sorted(turns)]}, wavs


def load_run(path: str | Path) -> Transcript:
    """Read one shard into the :class:`~rolebreak.types.Transcript` metrics consume."""
    row, wavs = record(path)

    def audio_for(rec: dict[str, Any]) -> OutputAudio | None:
        return output_audio(wavs.get(rec.get("index", -1)), rec.get("assistant_text"))

    return transcript_from_record(row, audio_for)


def find_shards(root: str | Path) -> list[Path]:
    """Every finished run shard directly under ``root``, by name.

    ``.partial`` files are not shards yet, and are not picked up.
    """
    return sorted(Path(root).glob(f"*{SUFFIX}"))


def load_shards(root: str | Path) -> list[Transcript]:
    """Read every run shard under ``root`` — the shard form of an ``--all`` run."""
    return [load_run(path) for path in find_shards(root)]
