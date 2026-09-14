"""Read and write a role-play run as JSONL.

A *run* is what a runner (e.g. ``tools/eval/eval_qwen3_omni.py``) produces by
replaying an :class:`~rolebreak.examples.Example` against a model: a persona plus
the alternating user/assistant turns, with each spoken reply saved as a WAV.

``run.jsonl`` is the structured, machine-readable record of that run — the
counterpart to the human-readable ``chat.log``. :func:`load_runs` reads it back
directly into the :class:`~rolebreak.types.Transcript` that metrics consume (see
``tools/score_run.py``), so metrics never have to parse the log and there is no
intermediate run object between the file and the unit of evaluation.

Schema — one JSON object (one row) per run, holding the persona + run settings
with every exchange nested under ``turns``::

    {"example": "captain", "name": "captain",
     "persona": "# Persona ...", "model": "Qwen/...", "voice": "Ethan",
     "turns": [
       {"index": 0, "user_text": "Morning, Mika!", "user_audio": null,
        "assistant_text": "Ahoy there!", "wav": "turn_01.wav",
        "accepted_emotions": ["happy", "calm"], "expected_rubric": [
          {"criterion": "greets warmly", "dimension": "interaction"}
        ],
        "latency": 0.72},
       ...
     ]}

A per-example ``run.jsonl`` is thus a single line. A consolidated ``runs.jsonl``
(``tools/eval/eval_qwen3_omni.py --all``) is one such row per example, one line
each. The assistant's speech is referenced by ``wav`` — a path relative to the run dir
— rather than embedded as base64, so the file stays small and diffable;
:func:`load_runs` reads the WAV back into the rebuilt transcript — treating a WAV
that holds no audio frames as an unspoken turn, the same as ``"wav": null``.
``user_audio``
likewise references the clip actually sent to the model (if the turn was spoken)
rather than inlining it: a path, or — for a clip a runner read out of a
WebDataset shard and never wrote to disk — a ``<shard>.tar#<key>.wav`` locator. ``accepted_emotions`` / ``expected_rubric`` carry each authored
turn's eval targets so the emotion and rubric metrics have something to grade
against — something the old chat.log format dropped.

Current rubric entries contain ``criterion`` and ``dimension``. Pre-dimension
run files that stored bare strings remain readable; because their original
dimension is unknowable, the loader assigns those legacy entries to
``interaction``.

The same run can instead be stored as a single WebDataset shard — the audio inside
the tar that holds the metadata rather than beside it — which is what the eval
runners write by default; see :mod:`rolebreak.runshard`. The two formats share the
row and turn schemas built here (:func:`run_header`, :func:`turn_record`) and
rebuild the same :class:`~rolebreak.types.Transcript`, so a metric never sees the
difference.

``latency`` is the seconds the user waited for that turn's reply (see
:class:`~rolebreak.types.ChatResponse`). It is the one thing here that cannot be
recovered from the saved audio afterwards — only the runner, watching the model
reply, ever sees it — so it is measured during the run and written down. ``null``
for a run recorded before this field existed, or a backend that can't report it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rolebreak.models.audio import encode_bytes, wav_frame_count
from rolebreak.types import Dimension, Emotion, Exchange, Message, OutputAudio, Rubric, Transcript

FILENAME = "run.jsonl"  # per-example run (single row)
RUNS_FILENAME = "runs.jsonl"  # consolidated, one row per example (--all)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def turn_record(
    index: int,
    *,
    user_text: str | None,
    assistant_text: str,
    user_audio: str | Path | None = None,
    wav: str | Path | None = None,
    accepted_emotions: list[Emotion] | None = None,
    expected_rubric: list[Rubric] | None = None,
    latency: float | None = None,
) -> dict[str, Any]:
    """One exchange as the JSON object a run's ``turns`` holds.

    The schema of a turn lives here alone so every serialization of a run writes
    the same object: :class:`RunWriter` nests these under a ``run.jsonl`` row, and
    :class:`~rolebreak.runshard.RunShardWriter` puts each one in its shard as the
    ``.json`` beside the turn's ``.wav``. ``wav`` names where the audio is in that
    serialization — a path relative to the run dir, or a member of the shard.
    """
    return {
        "index": index,
        "user_text": user_text,
        "user_audio": str(user_audio) if user_audio is not None else None,
        "assistant_text": assistant_text,
        "wav": str(wav) if wav is not None else None,
        "accepted_emotions": [emotion.value for emotion in accepted_emotions or []],
        "expected_rubric": [item.to_dict() for item in expected_rubric or []],
        "latency": latency,
    }


def run_header(example: str, persona: str, model: str | None = None, voice: str | None = None) -> dict[str, Any]:
    """A run's header fields — everything in a row but its ``turns``."""
    return {
        "example": example,  # this is useless
        "name": example,
        "persona": persona,
        "model": model,
        "voice": voice,
    }


def append_run(path: str | Path, row: dict[str, Any]) -> None:
    """Append one finished run's row to a JSONL file, creating it if need be.

    One row per line is the whole of the format, so every writer of one — this
    module's :class:`RunWriter` and the runners that index their shards into a
    consolidated ``runs.jsonl`` — goes through here.
    """
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class RunWriter:
    """Writer for a run's ``run.jsonl`` — one JSON object (row) per run.

    The header fields are captured on construction; each :meth:`add_turn` buffers
    an exchange, and the whole run (header + nested ``turns``) is written as a
    single line on :meth:`close`. Use as a context manager (or call
    :meth:`close`). Because it's one row, a partial/crashed run leaves nothing —
    the line is only emitted once the run finishes.

    Pass ``append=True`` to append to an existing file instead of truncating it,
    so several runs can share one consolidated JSONL — each run contributes its
    own single row (see ``tools/eval/eval_qwen3_omni.py --all``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        example: str,
        persona: str,
        model: str | None = None,
        voice: str | None = None,
        append: bool = False,
    ) -> None:
        self._path = Path(path)
        if not append:
            # Truncate up front so an aborted run leaves an empty file, not a stale one.
            self._path.write_text("", encoding="utf-8")
        self._header: dict[str, Any] = run_header(example, persona, model, voice)
        self._turns: list[dict[str, Any]] = []
        self._closed = False

    def add_turn(
        self,
        index: int,
        *,
        user_text: str | None,
        assistant_text: str,
        user_audio: str | Path | None = None,
        wav: str | Path | None = None,
        accepted_emotions: list[Emotion] | None = None,
        expected_rubric: list[Rubric] | None = None,
        latency: float | None = None,
    ) -> None:
        """Buffer one exchange. ``wav`` should be relative to the run dir."""
        self._turns.append(
            turn_record(
                index,
                user_text=user_text,
                assistant_text=assistant_text,
                user_audio=user_audio,
                wav=wav,
                accepted_emotions=accepted_emotions,
                expected_rubric=expected_rubric,
                latency=latency,
            )
        )

    def close(self) -> None:
        """Write the buffered run as a single JSONL row (idempotent)."""
        if self._closed:
            return
        self._closed = True
        # Always an append: a non-append writer truncated the file on construction.
        append_run(self._path, {**self._header, "turns": self._turns})

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def output_audio(data: bytes | None, transcript: str | None) -> OutputAudio | None:
    """Wrap a turn's WAV bytes in an :class:`OutputAudio`, if they hold audio.

    A WAV with no audio frames counts as *no audio*, exactly like a missing file:
    a backend that produced no speech for a turn can still leave a header-only
    WAV behind, and there is no waveform in it for a judge to hear. Handing one to
    an audio judge is worse than useless — UTMOSv2 divides by the sample count and
    dies on it, and voice_consistency would happily adopt it as the anchor every
    later turn is compared against. Returning None puts the turn on the path the
    audio metrics already have for silence: left unscored, rest of the run intact.

    Takes the bytes rather than a path because a run's audio does not always have
    one: in a shard (:mod:`rolebreak.runshard`) it is a tar member.
    """
    if data is None or wav_frame_count(data) == 0:
        return None
    return OutputAudio(data=encode_bytes(data), format="wav", transcript=transcript)


def _load_output_audio(wav_path: Path, transcript: str | None) -> OutputAudio | None:
    """Read a run dir's WAV file into an :class:`OutputAudio`, if it holds audio."""
    if not wav_path.exists():
        return None
    return output_audio(wav_path.read_bytes(), transcript)


def _resolve_wav(wav: str, run_dir: Path) -> Path:
    """Resolve a record's ``wav`` against ``run_dir``.

    Paths are written relative to the run dir; fall back to ``run_dir/<basename>``
    so a run moved to a new directory still resolves.
    """
    p = Path(wav)
    if p.is_absolute():
        return p
    candidate = run_dir / p
    return candidate if candidate.exists() else run_dir / p.name


def _load_rubric(value: Any) -> Rubric:
    """Load the current object schema or a pre-dimension legacy string.

    Old run logs did not retain enough information to recover the true semantic
    dimension. Mapping those criteria to interaction keeps the artifacts
    scoreable without guessing persona or safety from their wording.
    """
    if isinstance(value, str):
        return Rubric(value, Dimension.INTERACTION)
    return Rubric.from_dict(value)


def dir_audio(run_dir: Path) -> Callable[[dict[str, Any]], OutputAudio | None]:
    """An audio resolver for a run stored as WAV files under ``run_dir``.

    ``wav`` paths in each turn are resolved against ``run_dir`` — for a
    consolidated ``runs.jsonl`` they're ``<example>/turn_NN.wav`` relative to the
    out-dir root, for a per-example ``run.jsonl`` they're bare ``turn_NN.wav``.
    """

    shards: dict[str, dict[str, bytes]] = {}  # a shard is opened once, not once per turn

    def resolve(rec: dict[str, Any]) -> OutputAudio | None:
        wav = rec.get("wav")
        if not wav:
            return None
        # A shard run's rows point into the tar the audio is in rather than at a
        # file of its own; deferred import because runshard builds on this module.
        from rolebreak.runshard import read_wavs, split_locator

        locator = split_locator(wav)
        if locator is not None:
            shard, member = locator
            if shard not in shards:
                path = Path(shard) if Path(shard).is_absolute() else run_dir / shard
                shards[shard] = read_wavs(path) if path.exists() else {}
            return output_audio(shards[shard].get(member), rec.get("assistant_text"))
        return _load_output_audio(_resolve_wav(wav, run_dir), rec.get("assistant_text"))

    return resolve


def transcript_from_record(
    record: dict[str, Any], audio_for: Callable[[dict[str, Any]], OutputAudio | None]
) -> Transcript:
    """Rebuild one run row (header fields + nested ``turns``) into a :class:`Transcript`.

    Where each turn's audio *is* differs by how the run was stored — files beside a
    ``run.jsonl`` (:func:`dir_audio`) or members of a shard
    (:mod:`rolebreak.runshard`) — so the caller passes a resolver and everything
    else about reading a run stays in one place.

    The header's run settings are carried onto ``metadata`` so a report can say
    which example/model/voice produced the scores.
    """
    exchanges: list[Exchange] = []
    for rec in record.get("turns", []):
        audio = audio_for(rec)
        categories = rec.get("accepted_emotions")
        if categories is None:  # Backward compatibility for the previous list field name.
            categories = rec.get("accepted_categories")
        if categories is None:  # Backward compatibility for scalar run logs.
            legacy_emotion = rec.get("expected_emotion")
            categories = [legacy_emotion] if legacy_emotion is not None else []
        exchanges.append(
            Exchange(
                index=rec.get("index", len(exchanges)),
                user=Message(role="user", content=rec.get("user_text")),
                assistant=Message(role="assistant", content=rec.get("assistant_text") or "", audio=audio),
                accepted_emotions=[Emotion(emotion) for emotion in categories],
                expected_rubric=[_load_rubric(item) for item in rec.get("expected_rubric", [])],
                latency=rec.get("latency"),
            )
        )
    assert record.get("name") is not None, f"Run row {record} has no name; cannot key scores."
    return Transcript(
        persona=record.get("persona", ""),
        exchanges=exchanges,
        # A transcript must be named (it keys its scores). :class:`RunWriter` always
        # writes `name`; the `example`/placeholder fallbacks only cover a row written
        # by hand or by an older writer, so loading such a file still works.
        name=record.get("name"),
        metadata={k: record[k] for k in ("example", "model", "voice") if record.get(k) is not None},
    )


def load_runs(run_dir: str | Path, *, filename: str = FILENAME) -> list[Transcript]:
    """Parse every run row in ``<run_dir>/<filename>`` into :class:`Transcript` s.

    Each line is one run (see the module schema): each turn becomes an
    :class:`~rolebreak.types.Exchange` carrying the assistant WAV referenced by
    ``wav`` plus the authored ``accepted_emotions`` / ``expected_rubric``, so the
    result is ready to score with no further conversion.

    Use this for a consolidated ``runs.jsonl``
    (``tools/eval/eval_qwen3_omni.py --all``); WAV paths there are
    ``<example>/turn_NN.wav`` relative to ``run_dir``, so scoring works in place.
    A per-example ``run.jsonl`` yields a single-element list.
    """
    run_dir = Path(run_dir)
    path = run_dir / filename
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path} is empty.")
    resolve = dir_audio(run_dir)
    return [transcript_from_record(json.loads(ln), resolve) for ln in lines]
