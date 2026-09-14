"""Shared types for RoleVoiceBench: model I/O and metric results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union

Role = Literal["system", "user", "assistant", "tool"]
AudioFormat = Literal["wav", "mp3", "flac", "ogg", "opus", "pcm16"]
# Thinking budget for reasoning models; ``None`` leaves the server default alone.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Content parts (OpenAI message content)
# --------------------------------------------------------------------------- #
@dataclass
class TextContent:
    """A text content part: ``{"type": "text", "text": ...}``."""

    text: str
    type: Literal["text"] = "text"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    def __str__(self) -> str:
        return repr(self.text)


@dataclass
class InputAudio:
    """The inner payload of an ``input_audio`` part: base64 data + format."""

    data: str  # base64-encoded audio bytes
    format: AudioFormat = "wav"

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "format": self.format}

    def __str__(self) -> str:
        return f"audio[{self.format}]"


@dataclass
class InputAudioContent:
    """An audio content part: ``{"type": "input_audio", "input_audio": {...}}``."""

    input_audio: InputAudio
    type: Literal["input_audio"] = "input_audio"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "input_audio", "input_audio": self.input_audio.to_dict()}

    def __str__(self) -> str:
        return str(self.input_audio)


ContentPart = Union[TextContent, InputAudioContent]
# A message's content is either a bare string or a list of typed parts.
Content = Union[str, list[ContentPart]]


def _part_from_dict(part: dict[str, Any]) -> ContentPart:
    kind = part.get("type")
    if kind == "text":
        return TextContent(text=part["text"])
    if kind == "input_audio":
        ia = part["input_audio"]
        return InputAudioContent(input_audio=InputAudio(data=ia["data"], format=ia.get("format", "wav")))
    raise ValueError(f"Unsupported content part type: {kind!r}")


def _content_from_dict(content: Content | None) -> Content | None:
    if content is None or isinstance(content, str):
        return content
    return [_part_from_dict(p) for p in content]


def _content_to_dict(content: Content | None) -> Any:
    if content is None or isinstance(content, str):
        return content
    return [p.to_dict() for p in content]


# --------------------------------------------------------------------------- #
# Output audio (OpenAI assistant `message.audio`)
# --------------------------------------------------------------------------- #
@dataclass
class OutputAudio:
    """Audio produced by the model, mirroring OpenAI's ``message.audio``."""

    data: str  # base64-encoded audio bytes
    format: AudioFormat = "wav"
    transcript: str | None = None
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"data": self.data, "format": self.format}
        if self.transcript is not None:
            out["transcript"] = self.transcript
        if self.id is not None:
            out["id"] = self.id
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OutputAudio":
        return cls(
            data=d["data"],
            format=d.get("format", "wav"),
            transcript=d.get("transcript"),
            id=d.get("id"),
        )

    def __str__(self) -> str:
        preview = self.data[:16] + "…" if len(self.data) > 16 else self.data
        return f"audio[{self.format}] {preview!r}"


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    """A single conversation turn."""

    role: Role
    content: Content | None = None
    audio: OutputAudio | None = None  # set on assistant replies that emit speech

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": _content_to_dict(self.content)}
        if self.audio is not None:
            out["audio"] = self.audio.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        audio = d.get("audio")
        return cls(
            role=d["role"],
            content=_content_from_dict(d.get("content")),
            audio=OutputAudio.from_dict(audio) if audio else None,
        )

    def __str__(self) -> str:
        segments: list[str] = []
        if isinstance(self.content, str):
            segments.append(repr(self.content))
        elif isinstance(self.content, list):
            segments.extend(str(p) for p in self.content)
        if self.audio is not None:
            segments.append(str(self.audio))
        return f"{self.role}: " + " ".join(segments)


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    """Output audio settings (OpenAI's request-level ``audio`` field)."""

    voice: str | None = None
    format: AudioFormat = "wav"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"format": self.format}
        if self.voice is not None:
            out["voice"] = self.voice
        return out


@dataclass
class ChatRequest:
    """An OpenAI-style chat request scoped to speech-to-speech testing.

    ``modalities`` defaults to text+audio so models return spoken replies.
    """

    messages: list[Message]
    model: str | None = None
    modalities: list[Literal["text", "audio"]] = field(default_factory=lambda: ["text", "audio"])
    audio: AudioConfig | dict[str, Any] = field(default_factory=AudioConfig)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None  # only sent when set
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific knobs

    def __post_init__(self) -> None:
        # Accept a raw dict for ``audio`` (OpenAI-style) and coerce to AudioConfig.
        audio: Any = self.audio
        if isinstance(audio, dict):
            self.audio = AudioConfig(
                voice=audio.get("voice"),
                format=audio.get("format", "wav"),
            )

    def to_dict(self) -> dict[str, Any]:
        # __post_init__ guarantees self.audio is an AudioConfig.
        audio = self.audio if isinstance(self.audio, AudioConfig) else AudioConfig()
        out: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "modalities": list(self.modalities),
            "audio": audio.to_dict(),
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.model is not None:
            out["model"] = self.model
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            out["reasoning_effort"] = self.reasoning_effort
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChatRequest":
        known = {
            "messages",
            "model",
            "modalities",
            "audio",
            "temperature",
            "top_p",
            "max_tokens",
            "reasoning_effort",
        }
        audio = d.get("audio") or {}
        return cls(
            messages=[Message.from_dict(m) for m in d["messages"]],
            model=d.get("model"),
            modalities=d.get("modalities", ["text", "audio"]),
            audio=AudioConfig(voice=audio.get("voice"), format=audio.get("format", "wav")),
            temperature=d.get("temperature", 0.0),
            top_p=d.get("top_p", 1.0),
            max_tokens=d.get("max_tokens"),
            reasoning_effort=d.get("reasoning_effort"),
            extra={k: v for k, v in d.items() if k not in known},
        )


@dataclass
class ChatResponse:
    """The model's reply: an assistant message plus the raw backend payload."""

    message: Message
    model: str | None = None
    raw: Any = None  # untouched backend output, for debugging
    latency: float | None = None  # seconds the user waited for this reply

    @property
    def audio(self) -> OutputAudio | None:
        return self.message.audio

    @property
    def transcript(self) -> str | None:
        if self.message.audio is not None:
            return self.message.audio.transcript
        if isinstance(self.message.content, str):
            return self.message.content
        return None

    @property
    def text(self) -> str | None:
        """The reply's text content, ignoring any spoken audio."""
        return self.message.content if isinstance(self.message.content, str) else None

    def to_dict(self) -> dict[str, Any]:
        # Shaped like a single-choice OpenAI chat completion.
        out: dict[str, Any] = {"choices": [{"index": 0, "message": self.message.to_dict()}]}
        if self.model is not None:
            out["model"] = self.model
        return out


class Dimension(str, Enum):
    """The axes RoleVoiceBench scores."""

    PERSONA = "persona"  # stays in character: backstory, voice-of-mind, refusal to break the fourth wall
    VOICE = "voice"  # speaker identity / timbre stays constant across the conversation
    EMOTION = "emotion"  # affect is appropriate and responsive to the situation
    INTERACTION = "interaction"  # engaging, coherent, responsive, natural turn-taking
    SAFETY = "safety"  # refuses harm, resists jailbreaks, no unsafe content
    NATURALNESS = "naturalness"  # the waveform itself sounds like clean, natural speech


RUBRIC_DIMENSIONS = frozenset({Dimension.PERSONA, Dimension.INTERACTION, Dimension.SAFETY})


@dataclass(frozen=True)
class Rubric:
    """One atomic evaluation criterion assigned to a benchmark dimension.

    A rubric is an evaluation instrument, not a capability axis of its own.
    ``dimension`` identifies the capability that satisfying ``criterion``
    provides evidence for.
    """

    criterion: str
    dimension: Dimension

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str):
            raise ValueError(f"`criterion` must be a string, not {type(self.criterion).__name__}")
        criterion = self.criterion.strip()
        if not criterion:
            raise ValueError("`criterion` must be a non-empty string")
        object.__setattr__(self, "criterion", criterion)
        dimension = self.dimension
        if not isinstance(dimension, Dimension):
            try:
                dimension = Dimension(dimension)
            except (TypeError, ValueError):
                allowed = ", ".join(sorted(d.value for d in RUBRIC_DIMENSIONS))
                raise ValueError(f"unknown rubric dimension {self.dimension!r}; expected one of {allowed}") from None
        if dimension not in RUBRIC_DIMENSIONS:
            allowed = ", ".join(sorted(d.value for d in RUBRIC_DIMENSIONS))
            raise ValueError(f"rubric dimension must be one of {allowed}; got {dimension.value!r}")
        object.__setattr__(self, "dimension", dimension)

    def to_dict(self) -> dict[str, str]:
        return {"criterion": self.criterion, "dimension": self.dimension.value}

    @classmethod
    def from_dict(cls, value: Any) -> "Rubric":
        if not isinstance(value, dict):
            raise ValueError(f"a rubric item must be an object, not {type(value).__name__}")
        unknown = sorted(set(value) - {"criterion", "dimension"})
        if unknown:
            raise ValueError(f"unknown rubric key(s) {unknown}; expected `criterion` and `dimension`")
        if "criterion" not in value or "dimension" not in value:
            raise ValueError("a rubric item requires both `criterion` and `dimension`")
        return cls(criterion=value["criterion"], dimension=value["dimension"])


class Emotion(str, Enum):
    """Emotional delivery labels for a spoken reply.

    The eight-way taxonomy used by common speech-emotion-recognition models
    (e.g. RAVDESS), so authored accepted categories line up with what an emotion
    judge predicts. A turn may accept one or more of these labels.
    """

    NEUTRAL = "neutral"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISED = "surprised"


# --------------------------------------------------------------------------- #
# Input: a single replayed conversation
# --------------------------------------------------------------------------- #
@dataclass
class Exchange:
    """One user turn and the assistant's spoken reply to it."""

    index: int  # 0-based position in the conversation
    user: Message
    assistant: Message  # carries `.audio` (base64 wav) and/or text content
    accepted_emotions: list[Emotion] = field(default_factory=list)
    expected_rubric: list[Rubric] = field(default_factory=list)
    latency: float | None = None

    @property
    def assistant_audio(self) -> OutputAudio | None:
        return self.assistant.audio

    @property
    def assistant_text(self) -> str | None:
        """Best available text for the reply: transcript, else string content."""
        if self.assistant.audio is not None and self.assistant.audio.transcript is not None:
            return self.assistant.audio.transcript
        if isinstance(self.assistant.content, str):
            return self.assistant.content
        return None

    def __str__(self) -> str:
        lines = [f"Exchange #{self.index}"]
        lines.append(f"  {self.user}")
        lines.append(f"  {self.assistant}")
        if self.accepted_emotions:
            lines.append(f"  accepted_emotions: {[emotion.value for emotion in self.accepted_emotions]}")
        if self.expected_rubric:
            rendered = [item.to_dict() for item in self.expected_rubric]
            lines.append(f"  expected_rubric: {rendered}")
        if self.latency is not None:
            lines.append(f"  latency: {self.latency:.2f}s")
        return "\n".join(lines)


@dataclass
class Transcript:
    """A full rollout: persona + the turns the model produced, in order.

    ``name`` is the rollout's identity — normally the authored example's name,
    which :class:`~rolebreak.examples.Example` also requires — and it is
    mandatory here because it is the key scores are filed under (see
    :class:`~rolebreak.metrics.store.ScoreStore`) and the key a
    :class:`BenchmarkReport` is labelled with. An unnamed rollout could neither
    be resumed nor reported on, so it is rejected at construction rather than
    filed under a placeholder.

    ``reference_audio`` is the voice the persona is *supposed* to sound like (a
    seed/clone clip, or the model's own first reply) — voice-consistency metrics
    compare later turns against it.
    """

    persona: str  # the system prompt / character sheet
    exchanges: list[Exchange]
    name: str  # cache/report key; see the class docstring
    reference_audio: OutputAudio | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError(f"`name` must be a string, not {type(self.name).__name__}")
        name = self.name.strip()
        if not name:
            raise ValueError("`name` must be a non-empty string")
        self.name = name

    @property
    def num_turns(self) -> int:
        return len(self.exchanges)

    @classmethod
    def from_messages(
        cls,
        persona: str,
        messages: list[Message],
        *,
        name: str,
        **kw: Any,
    ) -> "Transcript":
        """Build from a flat user/assistant message list (the runner's output).

        Pairs each ``user`` message with the following ``assistant`` message; a
        leading ``system`` message, if present, overrides ``persona``. ``name``
        is required — it keys the rollout's scores.
        """
        persona_text = persona
        exchanges: list[Exchange] = []
        pending_user: Message | None = None
        for m in messages:
            if m.role == "system" and isinstance(m.content, str):
                persona_text = m.content
            elif m.role == "user":
                pending_user = m
            elif m.role == "assistant" and pending_user is not None:
                exchanges.append(Exchange(index=len(exchanges), user=pending_user, assistant=m))
                pending_user = None
        return cls(persona=persona_text, exchanges=exchanges, name=name, **kw)

    @classmethod
    def from_example(cls, example: Any, conversation: list[Message], **kw: Any) -> "Transcript":
        """Build from the authored ``example`` plus the replayed ``conversation``.

        Like :meth:`from_messages`, but also carries each authored turn's
        per-turn eval expectations (``accepted_emotions`` and ``expected_rubric``)
        onto the matching :class:`Exchange`, so emotion and rubric scoring have an
        authored target. The persona/name default to the example's; the i-th
        exchange is paired with the i-th authored turn (they share order).

        ``example`` is duck-typed (anything exposing ``system_prompt``, ``name``
        and ``turns``) to avoid a package import cycle with
        :mod:`rolebreak.examples`.
        """
        kw.setdefault("name", example.name)
        transcript = cls.from_messages(example.system_prompt, conversation, **kw)
        turns = list(getattr(example, "turns", []))
        for ex, turn in zip(transcript.exchanges, turns):
            ex.accepted_emotions = list(getattr(turn, "accepted_emotions", getattr(turn, "accepted_categories", [])))
            ex.expected_rubric = list(getattr(turn, "rubric", []) or [])
        return transcript


# --------------------------------------------------------------------------- #
# Output: a metric's verdict
# --------------------------------------------------------------------------- #
@dataclass
class MetricScore:
    """One metric's result for one transcript.

    ``per_turn`` (optional) holds a score per exchange — on the same scale — so
    reports can plot *drift*, the whole point of long-horizon stress testing.
    """

    metric: str
    dimension: Dimension
    score: float
    per_turn: list[float | None] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def drift(self) -> float | None:
        """Late-conversation minus early-conversation mean score.

        Negative = the agent degraded as the conversation got longer/harder,
        the headline failure mode this benchmark is built to catch. ``None`` if
        per-turn scores aren't available or the rollout is too short to split.
        """
        scores = [s for s in (self.per_turn or []) if s is not None]
        if len(scores) < 4:
            return None
        k = len(scores) // 3
        early = sum(scores[:k]) / k
        late = sum(scores[-k:]) / k
        return late - early

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "dimension": self.dimension.value,
            "score": self.score,
            "per_turn": self.per_turn,
            "drift": self.drift,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricScore:
        """Rebuild a score from :meth:`to_dict` output (e.g. a cached score file).

        ``drift`` is dropped: it's derived from ``per_turn``, so the property
        recomputes it rather than trusting the stored copy.
        """
        return cls(
            metric=data["metric"],
            dimension=Dimension(data["dimension"]),
            score=data["score"],
            per_turn=data.get("per_turn"),
            meta=data.get("meta") or {},
        )


@dataclass
class BenchmarkReport:
    """All metric scores for one transcript, plus convenience rollups.

    ``transcript`` is the scored rollout's :attr:`Transcript.name`, so a report
    can always be traced back to the run that produced it.
    """

    transcript: str
    scores: list[MetricScore] = field(default_factory=list)

    def by_metric(self) -> dict[str, float]:
        """Mean score per metric (averaging any duplicate entries for a metric)."""
        buckets: dict[str, list[float]] = {}
        for s in self.scores:
            buckets.setdefault(s.metric, []).append(s.score)
        return {m: sum(v) / len(v) for m, v in buckets.items()}

    @property
    def overall(self) -> float | None:
        vals = list(self.by_metric().values())
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "overall": self.overall,
            "by_metric": self.by_metric(),
            "scores": [s.to_dict() for s in self.scores],
        }
