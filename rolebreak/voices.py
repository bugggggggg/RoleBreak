"""Resolve a voice reference — a local path *or* a HuggingFace dataset row — to a
local audio file.

Zero-shot cloning backends want two things: a reference clip on disk and (ideally)
that clip's transcript. Both live in the voice-bank parquets on the Hub — one row
per speaker, with an ``audio`` column and the utterance ``text`` — so a reference
can be written as a spec string instead of a checked-in wav:

    hf://Greenbean/Different-Voices/data/vctk.parquet#p225  -> speaker p225 of that bank
    hf://Greenbean/Different-Voices/data/clip.wav           -> a plain file in a repo
    data/sample-audios/trump-2.wav                          -> a local path, unchanged

A spec names both halves: the file in the repo, and — after ``#`` — the speaker,
matched against the bank's ``speaker_id`` column. A repo may hold any number of
parquets and a bank any number of speakers, so neither is ever guessed; leaving
either out is an error rather than an arbitrary voice. Use ``ls`` to see which ids
a bank offers.

Any of those is a :class:`VoiceClip` — the reference as authored, which resolves
itself on first use to a local :class:`~pathlib.Path` plus the transcript the
dataset already knows. Constructing one is free, so an example can hold a bank
reference without touching the network at import time::

    VoiceClip("hf://Greenbean/Different-Voices/data/vctk.parquet#p225")
    VoiceClip("data/sample-audios/trump-2.wav", "by the time we finish...")

Local bank cache
----------------
The first reference to a bank downloads its parquet **once** and unpacks the whole
thing into ``~/.cache/rolebreak/voices`` (override with ``ROLEBREAK_VOICE_CACHE``):
every clip as its own file, plus an ``index.json`` holding each row's transcript
and metadata. Each parquet gets its own directory, so a repo's banks are pulled
independently — one voice never costs you the whole repo. Every later lookup — this
run or next week's, any speaker — is served from that directory: no Hub request, no
parquet parse, works offline. ``pull`` warms a bank deliberately; ``cache`` reports
what is already warm, and re-pulls it all with ``--refresh``::

    B=hf://Greenbean/Different-Voices/data/vctk.parquet
    python -m rolebreak.voices pull $B
    python -m rolebreak.voices cache                    # what's unpacked, and how big
    python -m rolebreak.voices cache --refresh          # re-pull every cached bank
    python -m rolebreak.voices ls $B --limit 10
    python -m rolebreak.voices get "$B#p225"

The default pool
----------------
:data:`USER_VOICE_BANKS` names the banks an example's *user* side falls back to when
it authors no voice of its own: :func:`pool_specs` flattens them into one spec per
speaker, which a caller deals out (see ``tools/synthesize_user_turns.py``). Adding a
bank there widens the pool for every such example at once::

    python -m rolebreak.voices pool --limit 5     # what the fallback pool offers
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

HF_SCHEME = "hf://"

#: Column a ``#p225`` key matches — every voice bank names its speakers here.
KEY_COLUMN = "speaker_id"

#: Column holding the clip's transcript — what a cloning engine wants alongside it.
TEXT_COLUMN = "text"

#: Never surfaced as clip metadata — the payload itself.
_AUDIO_COLUMN = "audio"

#: Suffixes that mark a spec as a file to check for, not an engine's named voice.
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm"})

#: Manifest written next to the unpacked clips; its presence means "fully cached".
INDEX_NAME = "index.json"
_CLIPS_DIR = "clips"

#: Banks the *user* side is drawn from when an example authors no ``user_voice``.
#: Hard-coded rather than a flag so every runner falls back to the same pool, and
#: so widening it is a diffable edit. Order matters: :func:`pool_specs` deals them
#: out in this order, bank by bank. English-only for now — the examples are English,
#: and a mismatched reference clip degrades cloning. The repo's ``ears`` bank is
#: deliberately left out for now; add it here to widen the pool.
USER_VOICE_BANKS: tuple[str, ...] = (
    "hf://Greenbean/Different-Voices/data/vctk.parquet",
    "hf://Greenbean/Different-Voices/data/libritts_r.parquet",
)


class _Resolved(NamedTuple):
    """What a spec resolves to: a local file, its text, and the row it came from."""

    path: Path
    transcript: str | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class VoiceClip:
    """A voice: the reference as authored, and the local clip it resolves to.

    ``spec`` is what was written down — a local audio path, an ``hf://`` voice-bank
    row, or an engine's own named voice (see the module docstring).
    ``authored_transcript`` is a hand-written transcript for it, which wins over the
    one a bank ships with the clip; leave it unset for a bank row.

    Resolution is lazy, so constructing one is free and examples can hold an
    ``hf://`` reference without downloading anything at import time. :attr:`path`
    pulls and unpacks the bank on first use and is cached for the process.
    """

    spec: str | Path
    authored_transcript: str | None = None

    @cached_property
    def _resolved(self) -> _Resolved:
        return _resolve(self.spec, self.authored_transcript)

    @property
    def path(self) -> Path:
        """The clip as a local file, ready to hand to a TTS backend."""
        return self._resolved.path

    @property
    def transcript(self) -> str | None:
        """The clip's text: ``authored_transcript`` if given, else the bank's."""
        return self._resolved.transcript

    @property
    def metadata(self) -> dict[str, str]:
        """The bank row's scalar columns — speaker, gender, accent, … — for logging.

        Empty for a local path, which carries nothing but itself.
        """
        return self._resolved.metadata

    @property
    def label(self) -> str:
        """Human-readable one-liner for logs/CLI output."""
        who = self.metadata.get(KEY_COLUMN)
        return f"{self.path} ({who})" if who else str(self.path)


def is_hf_spec(spec: str | Path) -> bool:
    """True if ``spec`` addresses a Hub row/file rather than a local path."""
    return isinstance(spec, str) and spec.startswith(HF_SCHEME)


def cache_dir() -> Path:
    """Root of the local bank cache."""
    env = os.environ.get("ROLEBREAK_VOICE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rolebreak" / "voices"


@dataclass(frozen=True)
class _HfSpec:
    """A parsed ``hf://`` clip reference — which bank, and which speaker in it.

    ``key`` is the fragment after ``#``: a :data:`KEY_COLUMN` value, i.e. a speaker
    id. ``path_in_repo`` is the table the speaker lives in.
    """

    repo_id: str
    path_in_repo: str
    key: str


def _bank_location(spec: str) -> tuple[str, str]:
    """``hf://owner/name/path/in/repo`` -> (repo id, path in repo).

    Any ``#speaker`` is ignored: this is the bank a spec points at, not the clip.
    Both halves are required — a repo may hold any number of parquets, so which one
    is meant is never guessed.
    """
    if not is_hf_spec(spec):
        raise ValueError(f"Not an hf:// voice spec: {spec!r}")
    body = spec[len(HF_SCHEME) :].partition("#")[0]
    parts = [p for p in body.split("/") if p]
    if len(parts) < 3:
        raise ValueError(
            f"Voice spec {spec!r} needs a repo id and a file in it, e.g. hf://owner/dataset/data/vctk.parquet#p225"
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


def parse_hf_spec(spec: str) -> _HfSpec:
    """``hf://owner/name/path/in/repo#speaker`` -> :class:`_HfSpec`.

    The speaker is required: a bank holds many voices, so which one is never
    implied. Use :func:`_bank_location` for bank-wide operations.
    """
    repo_id, path_in_repo = _bank_location(spec)
    key = spec.partition("#")[2]
    if not key:
        raise ValueError(f"Voice spec {spec!r} needs a #speaker, e.g. {HF_SCHEME}{repo_id}/{path_in_repo}#p225")
    return _HfSpec(repo_id=repo_id, path_in_repo=path_in_repo, key=key)


def _slug(text: str) -> str:
    """Filesystem-safe fragment of a repo id / table path, for cache paths."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "default"


class VoiceBank:
    """One parquet table of a voice bank, unpacked into a local directory.

    Wraps the ``index.json`` written when the table was pulled: one entry per row,
    each with its clip's filename, transcript, and scalar metadata. A repo may hold
    several of these; lookups never touch the network or the parquet.
    """

    def __init__(self, root: Path, index: dict[str, Any]) -> None:
        self.root = root
        self.repo_id: str = index["repo_id"]
        self.table: str = index["table"]
        self.revision: str | None = index.get("revision")
        self.rows: list[dict[str, str]] = index["rows"]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def keys(self) -> list[str]:
        """Every selectable speaker id, in bank order."""
        return [r[KEY_COLUMN] for r in self.rows if KEY_COLUMN in r]

    def clip_path(self, row: dict[str, str]) -> Path:
        return self.root / _CLIPS_DIR / row["_clip"]

    def select(self, key: str) -> dict[str, str]:
        """The row whose :data:`KEY_COLUMN` equals ``key``."""
        for row in self.rows:
            if row.get(KEY_COLUMN) == key:
                return row
        known = ", ".join(self.keys[:8]) or "(none)"
        raise LookupError(
            f"No {KEY_COLUMN} {key!r} in {self.repo_id}/{self.table}. Known: {known}, … ({len(self)} total)"
        )

    def resolved(self, key: str) -> _Resolved:
        """One row as a local clip: its file, the transcript it ships, its metadata."""
        row = self.select(key)
        return _Resolved(
            path=self.clip_path(row),
            transcript=row.get(TEXT_COLUMN) or None,
            metadata={k: v for k, v in row.items() if not k.startswith("_")},
        )


def _bank_root(repo_id: str, table: str) -> Path:
    return cache_dir() / _slug(repo_id) / _slug(table)


def _download(repo_id: str, path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, path_in_repo, repo_type="dataset"))


def _revision_of(local: Path) -> str | None:
    """The commit sha out of a hub cache path (``…/snapshots/<sha>/<file>``)."""
    parts = local.parts
    if "snapshots" in parts:
        at = parts.index("snapshots")
        if at + 1 < len(parts):
            return parts[at + 1]
    return None


def _write_json(path: Path, payload: Any) -> None:
    """Write atomically, so a killed run never leaves a half index behind."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _unpack(table_path: Path, root: Path, repo_id: str, table: str, *, on_progress=None) -> dict[str, Any]:
    """Unpack every row of a bank parquet into ``root``; return the index payload.

    Streams row group by row group, so a multi-gigabyte bank is unpacked without
    holding every clip in memory at once.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(table_path)
    names = parquet.schema_arrow.names
    if _AUDIO_COLUMN not in names:
        raise ValueError(f"{table_path.name} has no {_AUDIO_COLUMN!r} column; is it a voice bank?")
    scalar_columns = [n for n in names if n != _AUDIO_COLUMN]

    clips = root / _CLIPS_DIR
    clips.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for group in range(parquet.metadata.num_row_groups):
        table_rows = parquet.read_row_group(group, columns=scalar_columns).to_pylist()
        audios = parquet.read_row_group(group, columns=[_AUDIO_COLUMN]).column(_AUDIO_COLUMN).to_pylist()
        for row, audio in zip(table_rows, audios, strict=True):
            number = len(rows) + 1
            if not isinstance(audio, dict) or not audio.get("bytes"):
                raise ValueError(f"Row {number} has no inline audio bytes; unsupported voice-bank layout.")
            # Numbered in bank order, keeping the source container; which speaker a
            # clip belongs to is the index's job, not the filename's.
            name = f"{number:04d}{Path(audio.get('path') or 'clip.wav').suffix or '.wav'}"
            (clips / name).write_bytes(audio["bytes"])
            entry = {k: str(v) for k, v in row.items() if isinstance(v, str | int | float)}
            entry["_clip"] = name
            rows.append(entry)
            if on_progress is not None:
                on_progress(number)

    return {
        "repo_id": repo_id,
        "table": table,
        "revision": _revision_of(table_path),
        "rows": rows,
    }


def _read_index(root: Path) -> dict[str, Any] | None:
    index_path = root / INDEX_NAME
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except ValueError:
        return None  # truncated/garbage index: treat as a cache miss and re-pull
    return index if index.get("rows") is not None else None


# repo_id + table -> open bank, so every example in a run shares one index read
# regardless of which speaker each one selects.
_BANKS: dict[tuple[str, str], VoiceBank] = {}


def load_bank(spec: str, *, refresh: bool = False, on_progress=None) -> VoiceBank:
    """The bank behind ``spec``, unpacked locally — downloading it if needed.

    ``spec`` is an ``hf://`` parquet reference (any ``#speaker`` is ignored): the
    bank is the whole table, not one row. An already-unpacked bank is opened
    straight from its ``index.json``; ``refresh`` re-downloads and re-unpacks.
    ``on_progress`` is called with the running clip count while unpacking.
    """
    repo_id, table = _bank_location(spec)
    if not table.endswith(".parquet"):
        raise ValueError(f"{table} is not a parquet voice bank.")

    cache_key = (repo_id, table)
    if not refresh and cache_key in _BANKS:
        return _BANKS[cache_key]

    root = _bank_root(repo_id, table)
    index = None if refresh else _read_index(root)
    if index is None:
        local = _download(repo_id, table)
        index = _unpack(local, root, repo_id, table, on_progress=on_progress)
        _write_json(root / INDEX_NAME, index)

    bank = VoiceBank(root, index)
    _BANKS[cache_key] = bank
    return bank


def is_cached(spec: str) -> bool:
    """True if ``spec``'s bank is already unpacked locally (no Hub request)."""
    return _read_index(_bank_root(*_bank_location(spec))) is not None


def cached_banks() -> list[VoiceBank]:
    """Every bank currently unpacked in the local cache, in path order.

    Reads only the on-disk indexes, so it describes the cache without touching the
    Hub — the inventory a refresh iterates over.
    """
    banks = []
    for index_path in sorted(cache_dir().glob(f"*/*/{INDEX_NAME}")):
        index = _read_index(index_path.parent)
        if index is not None:
            banks.append(VoiceBank(index_path.parent, index))
    return banks


def _resolve(spec: str | Path, authored_transcript: str | None) -> _Resolved:
    """Turn a voice reference into a local clip. See :class:`VoiceClip`.

    A local path is taken as-is; an ``hf://`` spec pulls and unpacks its bank on the
    first call and reads from the cache afterwards. A spec that is neither an
    ``hf://`` reference nor an audio filename — an engine's own named voice — is
    passed through untouched; one that *looks* like an audio file but is missing
    raises here rather than deep inside a backend.
    """
    if not is_hf_spec(spec):
        path = Path(spec).expanduser()
        if path.suffix.lower() in AUDIO_SUFFIXES and not path.exists():
            raise FileNotFoundError(f"Voice clip not found: {path}")
        return _Resolved(path=path, transcript=authored_transcript, metadata={})

    # A plain audio file in a repo is already the clip — no bank, no speaker key.
    repo_id, path_in_repo = _bank_location(str(spec))
    if not path_in_repo.endswith(".parquet"):
        return _Resolved(path=_download(repo_id, path_in_repo), transcript=authored_transcript, metadata={})

    row = load_bank(str(spec)).resolved(parse_hf_spec(str(spec)).key)
    return row if authored_transcript is None else row._replace(transcript=authored_transcript)


def list_voices(spec: str, *, limit: int = 0) -> list[dict[str, str]]:
    """Every row of a voice bank (metadata only), from the local cache.

    ``spec`` is an ``hf://`` parquet reference; any ``#key`` is ignored.
    """
    rows = load_bank(spec).rows
    return rows[:limit] if limit else rows


def pool_specs(banks: Sequence[str] = USER_VOICE_BANKS) -> list[str]:
    """Every speaker of every bank in ``banks``, as a full ``hf://…#speaker`` spec.

    The fallback pool a caller deals distinct voices from — one spec per speaker, in
    bank order then bank-internal order, so the same pool comes out on every run.
    Pulls and unpacks each bank on first use (see :func:`load_bank`), so call it only
    once something actually needs a voice.
    """
    return [f"{bank}#{key}" for bank in banks for key in load_bank(bank).keys]


def _cli() -> None:
    import click

    def progress(done: int) -> None:
        if done % 100 == 0:
            click.echo(f"  {done} clips")

    def describe(bank: VoiceBank) -> str:
        size = sum(p.stat().st_size for p in bank.root.rglob("*") if p.is_file()) / 1024**2
        unit = f"{size / 1024:.1f} GiB" if size >= 1024 else f"{size:.0f} MiB"
        return f"{bank.repo_id}/{bank.table} @ {bank.revision or '?'}: {len(bank)} clip(s), {unit} -> {bank.root}"

    @click.group()
    def cli() -> None:
        """Inspect and resolve voice-bank references."""

    @cli.command("pull")
    @click.argument("spec")
    @click.option("--refresh", is_flag=True, default=False, help="Re-download and re-unpack even if cached.")
    def pull(spec: str, refresh: bool) -> None:
        """Download a bank once and unpack every clip into the local cache.

        SPEC names the parquet, e.g. hf://Greenbean/Different-Voices/data/vctk.parquet
        """
        if is_cached(spec) and not refresh:
            click.echo("Already unpacked; pass --refresh to re-pull.")
        else:
            click.echo(f"Pulling {spec} ...")

        click.echo(describe(load_bank(spec, refresh=refresh, on_progress=progress)))

    @cli.command("cache")
    @click.option("--refresh", is_flag=True, default=False, help="Re-download and re-unpack every cached bank.")
    def cache(refresh: bool) -> None:
        """Survey the local bank cache, or refresh every bank in it.

        Read straight off disk by default — no Hub request. Use ``pull`` to add a
        bank to the cache; this is for what's already there.

            python -m rolebreak.voices cache
            python -m rolebreak.voices cache --refresh
        """
        banks = cached_banks()
        if not banks:
            click.echo(f"Cache is empty ({cache_dir()}); `pull` a bank to build it.")
            return

        click.echo(f"{len(banks)} bank(s) in {cache_dir()}")
        for bank in banks:
            # Keyed off what's on disk, so a bank is described (and re-pulled) where
            # it actually sits rather than where a re-resolved spec would put it.
            if refresh:
                spec = f"{HF_SCHEME}{bank.repo_id}/{bank.table}"
                click.echo(f"Re-pulling {spec} ...")
                bank = load_bank(spec, refresh=True, on_progress=progress)
            click.echo("  " + describe(bank))

    @cli.command("ls")
    @click.argument("spec")
    @click.option("--limit", default=20, show_default=True, help="Rows to show (0 = all).")
    def ls(spec: str, limit: int) -> None:
        """List the speakers in a voice bank, e.g. .../data/vctk.parquet.

        One row per selectable #key; grep it to narrow by accent, gender, ...
        """
        rows = list_voices(spec)
        shown = rows[:limit] if limit else rows
        for row in shown:
            key = row.get(KEY_COLUMN, "?")
            tags = " ".join(f"{k}={row[k]}" for k in ("gender", "age", "accent") if row.get(k))
            click.echo(f"{key:<8} {tags:<34} {row.get(TEXT_COLUMN, '')[:60]}")
        click.echo(f"\n{len(shown)}/{len(rows)} row(s)")

    @cli.command("pool")
    @click.option("--limit", default=0, show_default=True, help="Specs to show (0 = all).")
    def pool(limit: int) -> None:
        """List the fallback pool: every speaker of every bank in USER_VOICE_BANKS.

        One spec per line, in the order they are dealt out to examples that author no
        ``user_voice``. The count is the ceiling on how many such examples can each
        get a distinct voice.
        """
        specs = pool_specs()
        for spec in specs[:limit] if limit else specs:
            click.echo(spec)
        click.echo(f"\n{len(specs)} voice(s) across {len(USER_VOICE_BANKS)} bank(s)")

    @cli.command("get")
    @click.argument("spec")
    def get(spec: str) -> None:
        """Resolve one reference and print the local clip it maps to."""
        clip = VoiceClip(spec)
        click.echo(f"path:       {clip.path}")
        click.echo(f"transcript: {clip.transcript!r}")
        if clip.metadata:
            click.echo("metadata:   " + ", ".join(f"{k}={v}" for k, v in clip.metadata.items()))

    cli()


if __name__ == "__main__":
    _cli()
