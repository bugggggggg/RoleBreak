"""Reusable multi-turn chat examples, shared across model backends."""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rolebreak.types import Rubric
from rolebreak.voices import VoiceClip

if TYPE_CHECKING:
    from rolebreak.types import Emotion, Message

#: Hub dataset the merged role library is published to.
EXAMPLES_REPO = "Greenbean/RoleBreak"
#: Keys a JSONL record may carry for provenance that no :class:`Example` field reads.
_JSONL_META_KEYS = frozenset({"source"})

#: Keys a role record may set at the top level; anything else is a typo.
_EXAMPLE_KEYS = frozenset({"name", "persona", "scenario", "turns", "reference_voice", "user_voice"})
#: Keys a turn object may set.
_TURN_KEYS = frozenset({"text", "audio", "accepted_emotions", "accepted_categories", "emotion", "rubric"})


# --------------------------------------------------------------------------- #
# JSON field readers — each raises ValueError with an unprefixed message, so a
# caller can say which turn or record it came from.
# --------------------------------------------------------------------------- #
def _rubric(value: Any) -> list[Rubric]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`rubric` must be a list of criteria, not {type(value).__name__}")
    try:
        return [item if isinstance(item, Rubric) else Rubric.from_dict(item) for item in value]
    except ValueError as exc:
        raise ValueError(f"invalid `rubric` item: {exc}") from None


def _emotion(value: Any) -> "Emotion":
    from rolebreak.types import Emotion

    if not isinstance(value, str):
        raise ValueError(f"an emotion category must be a string, not {type(value).__name__}")
    try:
        return Emotion(value.strip().lower())
    except ValueError:
        allowed = ", ".join(e.value for e in Emotion)
        raise ValueError(f"unknown emotion {value!r}; expected one of {allowed}") from None


def _accepted_emotions(value: Any) -> list["Emotion"]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"`accepted_emotions` must be a list of emotions, not {type(value).__name__}")
    categories = [_emotion(item) for item in value]
    if len(set(categories)) != len(categories):
        raise ValueError("`accepted_emotions` must not contain duplicates")
    return categories


def _voice(value: Any, key: str) -> VoiceClip | None:
    """Build a :class:`~rolebreak.voices.VoiceClip` from a spec string or a mapping."""
    if value is None:
        return None
    if isinstance(value, str):
        return VoiceClip(value)
    if isinstance(value, dict):
        spec = value.get("spec") or value.get("path")
        if not spec:
            raise ValueError(f"`{key}` mapping needs a `spec` (the clip path or hf:// reference)")
        return VoiceClip(str(spec), value.get("transcript"))
    raise ValueError(f"`{key}` must be a spec string or a mapping with `spec` and optional `transcript`")


@dataclass
class Turn:
    """A single user turn: what the user says, plus what a good reply must do."""

    text: str | None = None
    audio: str | Path | None = None
    rubric: list[Rubric] = field(default_factory=list)
    accepted_emotions: list[Emotion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.text is None and self.audio is None:
            raise ValueError("A Turn needs at least one of `text` or `audio`.")
        if any(not isinstance(item, Rubric) for item in self.rubric):
            raise ValueError("every `rubric` item must be Rubric(criterion, dimension)")
        from rolebreak.types import Emotion

        if any(not isinstance(item, Emotion) for item in self.accepted_emotions):
            raise ValueError("every `accepted_emotions` item must be an Emotion")
        if len(set(self.accepted_emotions)) != len(self.accepted_emotions):
            raise ValueError("`accepted_emotions` must not contain duplicates")

    @property
    def label(self) -> str:
        """Human-readable one-liner for logs/CLI output."""
        if self.audio is not None:
            tag = f"<audio: {self.audio}>"
            return f"{self.text}  {tag}" if self.text else tag
        return self.text or ""

    def to_message(self) -> "Message":
        """Build a model-ready ``user`` :class:`~rolebreak.models.Message`."""
        if self.audio is not None:
            from rolebreak.models.audio import user_audio_message

            return user_audio_message(self.audio, text=self.text)

        from rolebreak.types import Message

        return Message(role="user", content=self.text)

    @classmethod
    def from_dict(cls, d: Any) -> "Turn":
        """Build a turn from one JSON turn object (see the module docstring)."""
        if not isinstance(d, dict):
            raise ValueError(f"a turn is an object, not {type(d).__name__}")
        unknown = sorted(set(d) - _TURN_KEYS)
        if unknown:
            raise ValueError(f"unknown key(s) {unknown}; expected {sorted(_TURN_KEYS)}")
        emotion_keys = [
            key for key in ("accepted_emotions", "accepted_categories", "emotion") if d.get(key) is not None
        ]
        if len(emotion_keys) > 1:
            raise ValueError(f"a turn cannot set multiple emotion fields: {emotion_keys}")
        emotions = _accepted_emotions(d.get("accepted_emotions", d.get("accepted_categories")))
        if d.get("emotion") is not None:  # Backward compatibility for scalar role JSON.
            emotions = [_emotion(d["emotion"])]
        return cls(
            text=d.get("text"),
            audio=d.get("audio"),
            rubric=_rubric(d.get("rubric")),
            accepted_emotions=emotions,
        )

    def to_dict(self) -> dict[str, Any]:
        """The turn as JSON data, carrying only the fields it actually sets."""
        d: dict[str, Any] = {}
        if self.text is not None:
            d["text"] = self.text
        if self.audio is not None:
            d["audio"] = str(self.audio)
        if self.accepted_emotions:
            d["accepted_emotions"] = [emotion.value for emotion in self.accepted_emotions]
        if self.rubric:
            d["rubric"] = [item.to_dict() for item in self.rubric]
        return d


@dataclass
class Example:
    """A multi-turn chat: a persona, optional scenario, and ordered user turns."""

    name: str
    persona: str
    scenario: str = ""
    turns: list[Turn] = field(default_factory=list)
    reference_voice: VoiceClip | None = None
    user_voice: VoiceClip | None = None

    @property
    def system_prompt(self) -> str:
        """Persona and scenario composed into a structured system prompt."""
        parts = [f"# Persona\n{self.persona.strip()}"]
        if self.scenario.strip():
            parts.append(f"# Scenario\n{self.scenario.strip()}")
        return "\n\n".join(parts)

    def messages(self) -> list["Message"]:
        """The full conversation: system prompt followed by every user turn."""
        from rolebreak.types import Message

        return [
            Message(role="system", content=self.system_prompt),
            *(t.to_message() for t in self.turns),
        ]

    @classmethod
    def from_dict(cls, d: Any, *, name: str | None = None) -> "Example":
        """Build an example from JSON data (see the module docstring for the shape).

        ``name`` is the fallback used when the record omits ``name``.
        """
        if not isinstance(d, dict):
            raise ValueError(f"a role is one JSON object, not {type(d).__name__}")
        unknown = sorted(set(d) - _EXAMPLE_KEYS)
        if unknown:
            raise ValueError(f"unknown key(s) {unknown}; expected {sorted(_EXAMPLE_KEYS)}")

        example_name = d.get("name") or name
        if not example_name:
            raise ValueError("no example name: set `name`")
        persona = d.get("persona")
        if not persona:
            raise ValueError("missing or empty `persona`")
        turns = d.get("turns", [])
        if not isinstance(turns, list):
            raise ValueError("`turns` must be a list of turn objects")

        return cls(
            name=str(example_name),
            persona=persona,
            scenario=d.get("scenario") or "",  # `scenario` is a plain str, absent means empty
            turns=[_turn_at(turn, index) for index, turn in enumerate(turns, 1)],
            reference_voice=_voice(d.get("reference_voice"), "reference_voice"),
            user_voice=_voice(d.get("user_voice"), "user_voice"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The example as JSON data, in the shape :meth:`from_dict` reads."""
        d: dict[str, Any] = {"name": self.name, "persona": self.persona}
        if self.scenario:
            d["scenario"] = self.scenario
        for key, clip in (("reference_voice", self.reference_voice), ("user_voice", self.user_voice)):
            if clip is not None:
                d[key] = (
                    str(clip.spec)
                    if clip.authored_transcript is None
                    else {"spec": str(clip.spec), "transcript": clip.authored_transcript}
                )
        d["turns"] = [turn.to_dict() for turn in self.turns]
        return d

    def to_json(self) -> str:
        """The example rendered as a JSON role record, ready to write out."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"


def _turn_at(d: Any, index: int) -> Turn:
    """:meth:`Turn.from_dict`, with the turn's position added to any complaint."""
    try:
        return Turn.from_dict(d)
    except ValueError as exc:
        raise ValueError(f"turn {index}: {exc}") from exc


class _Registry(Mapping[str, Example]):
    """``name -> Example`` for every role in this package, loaded on first look."""

    def __init__(self) -> None:
        self._examples: dict[str, Example] = {}
        self._loaded = False
        self._failure: Exception | None = None

    def _load(self) -> dict[str, Example]:
        if not self._loaded:
            self._loaded = True  # set first: register() during the load must not recurse
            try:
                _load_all()
            except Exception as exc:
                self._failure = exc
        if self._failure is not None:
            raise self._failure
        return self._examples

    def add(self, example: Example) -> None:
        """Record ``example`` without loading anything. See :func:`register`."""
        if example.name in self._examples:
            raise ValueError(f"An example named {example.name!r} is already registered.")
        self._examples[example.name] = example

    def __getitem__(self, name: str) -> Example:
        return self._load()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __repr__(self) -> str:
        state = f"{len(self._examples)} examples" if self._loaded else "not loaded"
        return f"<EXAMPLES: {state}>"


# The registry itself, for the one place that writes to it rather than reads it.
_REGISTRY = _Registry()

#: name -> Example, for every role in this package. Lazy: see :class:`_Registry`.
EXAMPLES: Mapping[str, Example] = _REGISTRY


def register(example: Example) -> Example:
    """Add ``example`` to :data:`EXAMPLES`, keyed by its name.

    Raises :class:`ValueError` if an example with the same name is already
    registered, so name clashes fail loudly instead of silently overwriting.
    """
    _REGISTRY.add(example)
    return example


def jsonl_path(path: str | Path | None = None) -> Path:
    """The merged role JSONL, from the Hugging Face cache — downloaded once, then reused.

    ``path`` names a JSONL to read instead, for a library that is not the published one.
    """
    if path is not None:
        named = Path(path).expanduser()
        if not named.is_file():
            raise FileNotFoundError(f"no role JSONL at {named}")
        return named

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(EXAMPLES_REPO, "data/examples.jsonl", repo_type="dataset"))


def load_jsonl(path: str | Path | None = None) -> list[Example]:
    """Read every role in a merged JSONL, in file order."""
    file = jsonl_path(path)
    examples: list[Example] = []
    with file.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file}:{number}: invalid JSON: {exc}") from None
            if isinstance(data, dict):
                data = {key: value for key, value in data.items() if key not in _JSONL_META_KEYS}
            try:
                examples.append(Example.from_dict(data))
            except ValueError as exc:
                raise ValueError(f"{file}:{number}: {exc}") from exc
    return examples


def _authored_roles() -> list[Example]:
    """The role library, read from the JSONL ``ROLEBREAK_EXAMPLES`` names or the default one."""
    source = os.environ.get("ROLEBREAK_EXAMPLES", "").strip()
    return load_jsonl(source or None)


def _load_all() -> None:
    """Load every role — Python modules, and the authored library from its JSONL."""
    for info in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if not info.name.rpartition(".")[2].startswith("_"):
            importlib.import_module(info.name)

    for example in _authored_roles():
        if example.turns:  # persona-only roles have nothing to run
            register(example)


__all__ = [
    "EXAMPLES",
    "EXAMPLES_REPO",
    "Example",
    "Turn",
    "VoiceClip",
    "jsonl_path",
    "load_jsonl",
    "register",
]
