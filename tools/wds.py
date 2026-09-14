"""The WebDataset shard format the synthesized user turns are stored in.

``tools/synthesize_user_turns.py`` writes one shard per example
(``<out-dir>/vX.Y/<example>.tar``) and the eval runners read the user turns back
out of it. Both sides go through this module — as with
:mod:`tools.audio_version` for the directory naming — so a writer and a reader
can never disagree about what a sample looks like. The tar-level work is the
``webdataset`` library's (``pip install webdataset``); what lives here is the
convention layered on top of it.

A WebDataset *sample* is a set of tar members sharing the part of their name
before the first dot (the *key*); the extension after it names the field. A
synthesized user turn is keyed ``<example>/NNN`` (``NNN`` is the 0-based index of
the turn in ``Example.turns``, so keys line up with the authored conversation)
and carries three fields:

* ``.wav`` — the spoken clip;
* ``.txt`` — the turn text that was spoken;
* ``.json`` — metadata: ``example``, ``index``, ``text``, ``system``.

Shards are written with a fixed mtime, so a re-synthesis of the same turns gives
the same bytes. Reading is streaming and single-pass, which is what WebDataset
shards are for — and it means a shard truncated by an interrupted synthesis run
still yields every complete sample before the tear (see :func:`read_shard`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import webdataset

SUFFIX = ".tar"

# Fixed member mtime: shards are content-addressed by their version directory, so
# a re-run that synthesizes the same turns should not churn the bytes.
MTIME = 0


def shard_path(audio_dir: str | Path, example_name: str) -> Path:
    """Where ``example_name``'s shard lives under one resolved version dir."""
    return Path(audio_dir) / f"{example_name}{SUFFIX}"


def sample_key(example_name: str, index: int) -> str:
    """The key grouping one turn's members: ``<example>/NNN`` (0-based index)."""
    return f"{example_name}/{index:03d}"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def open_writer(path: str | Path) -> webdataset.TarWriter:
    """Open ``path`` as a shard to append samples to with :func:`write_sample`."""
    return webdataset.TarWriter(str(path), mtime=MTIME)


def write_sample(writer: webdataset.TarWriter, key: str, *, wav: bytes, text: str, meta: dict[str, Any]) -> None:
    """Write one turn: its clip, the text spoken, and the turn's metadata.

    Named per field rather than taking a dict so the three members a synthesized
    turn is made of are declared in exactly one place — here — and the readers
    below stay in step with them.
    """
    writer.write({"__key__": key, "wav": wav, "txt": text, "json": meta})


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One synthesized user turn read back out of a shard."""

    key: str
    wav: bytes | None = None
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    url: str = ""  # the shard it came out of

    @property
    def ref(self) -> str:
        """Where the clip lives, as ``<shard>.tar#<key>.wav``.

        A locator rather than a path: the readers hand the bytes straight to a
        model, so there is no file of their own for a run to point at.
        """
        return f"{self.url}#{self.key}.wav" if self.url else f"{self.key}.wav"

    @property
    def index(self) -> int:
        """The turn's 0-based index in ``Example.turns``.

        Taken from ``.json`` when it is there, else parsed off the key's ``NNN``
        so a sample whose metadata member was lost to a truncated shard is still
        placed correctly.
        """
        idx = self.meta.get("index")
        if isinstance(idx, int):
            return idx
        return int(self.key.rsplit("/", 1)[-1])


def read_shard(path: str | Path) -> Iterator[Sample]:
    """Stream ``path``'s samples in shard order.

    Only samples that carry a ``.wav`` are yielded — an audio-less group is
    nothing a runner can speak. A shard cut short mid-member (an interrupted
    synthesis run) is not an error: ``warn_and_stop`` ends the read at the tear
    and every sample completed before it is still yielded, so a partially
    synthesized example can be replayed as far as it goes.

    The clip stays raw bytes: no ``.decode()`` in the pipeline, because the
    callers hand whole WAV files to a model rather than tensors to a training
    loop.
    """
    dataset = webdataset.WebDataset(
        str(path),
        shardshuffle=False,
        empty_check=False,  # a truncated or empty shard is the caller's to report
        handler=webdataset.warn_and_stop,
    )
    for sample in dataset:
        wav = sample.get("wav")
        if wav is None:
            continue
        text = sample.get("txt")
        meta = sample.get("json")
        yield Sample(
            key=sample["__key__"],
            wav=wav,
            text=text.decode("utf-8") if text is not None else None,
            meta=json.loads(meta) if meta is not None else {},
            url=sample.get("__url__", str(path)),
        )


def read_turns(path: str | Path) -> dict[int, Sample]:
    """Read a whole shard into memory, keyed by turn index.

    A conversation is a handful of clips and the runners need them all before
    they open a session anyway, so the shard is read once up front rather than
    streamed per turn. The clips stay in memory as bytes — the backends decode
    those directly (``load_audio``), so nothing has to be unpacked to disk.
    """
    return {sample.index: sample for sample in read_shard(path)}
