"""Text-to-speech: a compact, uniform text-in / speech-out interface."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from rolebreak.models.audio import output_audio_from_waveform, save_output_audio
from rolebreak.types import OutputAudio


class TextToSpeechModel(abc.ABC):
    """A text-in, speech-out model.

    Subclasses load weights in :meth:`_load` (or lazily) and implement
    :meth:`_synthesize`, which returns a ``(samples, sample_rate)`` pair; the
    base class wraps that waveform into an :class:`OutputAudio`. Construction
    should stay cheap — defer heavy work to ``load()`` so the registry can build
    a model without paying for it.
    """

    #: Stable backend identifier, e.g. ``"cosyvoice3"``.
    backend: str = "base"

    #: Voice / language used when the caller passes neither (purely informational;
    #: each backend also resolves its own sensible default inside ``_synthesize``).
    default_voice: str | None = None
    default_language: str | None = None

    def __init__(self, model: str | None = None, **config: Any) -> None:
        # ``model`` is optional: some engines resolve a default checkpoint on
        # their own, so the registry can build one with no id at all.
        self.model = model
        self.config = config
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------- #
    def load(self) -> "TextToSpeechModel":
        """Load weights / connect to the engine. Idempotent."""
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:  # pragma: no cover - overridden by backends
        """Backend-specific load. Default no-op for lazy backends."""

    def warmup(self, text: str = "Hello, world.", **kwargs: Any) -> "TextToSpeechModel":
        """Run a throwaway synthesis to prime the engine.

        The first real :meth:`synthesize` on a fresh model often pays a one-off
        cold-start cost — CUDA kernel compilation, graph capture, buffer
        allocation, autoregressive cache setup. Calling ``warmup`` once after
        :meth:`load` moves that cost off the critical path so latency-sensitive
        callers see a warm engine. Loads first if needed; the rendered audio is
        discarded. Extra ``kwargs`` pass through to the engine.
        """
        if not self._loaded:
            self.load()
        self._warmup(text, **kwargs)
        return self

    def _warmup(self, text: str, **kwargs: Any) -> None:
        """Backend warmup hook. Default: synthesize a short utterance with the
        backend defaults and discard the result. Backends override to prime a
        specific code path (e.g. the streaming decoder) or skip cheaply."""
        self._synthesize(text, voice=None, language=None, speed=1.0, **kwargs)

    def close(self) -> None:  # pragma: no cover - overridden by backends
        """Release GPU memory / handles."""

    def __enter__(self) -> "TextToSpeechModel":
        return self.load()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- synthesis ---------------------------------------------------------- #
    @abc.abstractmethod
    def _synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        speed: float,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """Render ``text`` to a mono waveform.

        Return ``(samples, sample_rate)`` where ``samples`` is a 1-D array-like
        (float in ``[-1, 1]`` or int16). ``voice``/``language`` may be ``None`` —
        resolve a backend default in that case.
        """

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> OutputAudio:
        """Synthesize ``text`` into an :class:`OutputAudio` (base64 WAV).

        ``voice`` and ``language`` accept whatever the backend documents (a named
        voice, an ISO code like ``"ja"``, a reference-clip path for cloning, ...);
        both default to the backend's own pick when omitted. ``speed`` is a
        playback-rate multiplier (``1.0`` = natural). Extra ``kwargs`` pass
        through to the engine.
        """
        if not self._loaded:
            self.load()
        samples, sample_rate = self._synthesize(text, voice=voice, language=language, speed=speed, **kwargs)
        return output_audio_from_waveform(samples, sample_rate, transcript=text)

    def save(self, text: str, path: str | Path, **kwargs: Any) -> Path:
        """Synthesize ``text`` and write it to ``path``. Returns the path."""
        return save_output_audio(self.synthesize(text, **kwargs), path)

    # -- streaming (optional) ---------------------------------------------- #
    def stream(
        self,
        text: str | Iterable[str],
        *,
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[tuple[Any, int]]:
        """Incrementally synthesize, yielding ``(samples, sample_rate)`` audio
        chunks as they are produced.

        ``text`` may be a plain string or an *iterable of text chunks* — e.g. the
        tokens streaming out of an upstream LLM — for backends that accept
        streaming input. Each yielded ``samples`` is a 1-D array-like covering one
        chunk of the utterance, at the shared ``sample_rate``; concatenating them
        reconstructs the full :meth:`synthesize` waveform.

        Backends without incremental support fall back to synthesizing the whole
        utterance and yielding it as a single chunk (joining a chunk iterable into
        one string first).
        """
        if not self._loaded:
            self.load()
        yield from self._stream(text, voice=voice, language=language, speed=speed, **kwargs)

    def _stream(
        self,
        text: str | Iterable[str],
        *,
        voice: str | None,
        language: str | None,
        speed: float,
        **kwargs: Any,
    ) -> Iterator[tuple[Any, int]]:
        """Backend streaming hook. Default: no incremental support — join a chunk
        iterable into one string, synthesize the whole utterance, and yield it as a
        single chunk. Backends that can stream override this."""
        if not isinstance(text, str):
            text = "".join(text)
        yield self._synthesize(text, voice=voice, language=language, speed=speed, **kwargs)

    # -- discovery (optional) ---------------------------------------------- #
    @property
    def voices(self) -> list[str]:
        """Named voices this model ships, if it has a fixed set (else empty)."""
        return []

    @property
    def languages(self) -> list[str]:
        """Languages this model can speak, if enumerable (else empty)."""
        return []


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# backend name -> zero-arg importer returning the TextToSpeechModel subclass.
_TTS_REGISTRY: dict[str, Callable[[], type[TextToSpeechModel]]] = {}


def register_tts_backend(
    name: str,
) -> Callable[[type[TextToSpeechModel]], type[TextToSpeechModel]]:
    """Class decorator registering a TTS backend under ``name``."""

    def deco(cls: type[TextToSpeechModel]) -> type[TextToSpeechModel]:
        _TTS_REGISTRY[name] = lambda: cls
        return cls

    return deco


# Built-in name -> (module, class). Kept as a table so the error message and
# the resolver stay in sync.
_BUILTIN_TTS = {
    "cosyvoice3": ("cosyvoice", "CosyVoiceTTSModel"),
}


def _builtin_tts(name: str) -> type[TextToSpeechModel]:
    # Lazy imports keep optional deps (CosyVoice) out of the import path
    # until a backend is actually requested.
    try:
        module, cls_name = _BUILTIN_TTS[name]
    except KeyError:
        raise KeyError(name) from None
    import importlib

    mod = importlib.import_module(f".backends.{module}", package=__package__)
    return getattr(mod, cls_name)


def get_tts_backend(name: str) -> type[TextToSpeechModel]:
    """Resolve a TTS backend class by name. ``cosyvoice3:foo`` resolves to
    ``cosyvoice3`` unless ``cosyvoice3:foo`` was explicitly registered."""
    if name in _TTS_REGISTRY:
        return _TTS_REGISTRY[name]()
    base = name.split(":", 1)[0]
    if base in _TTS_REGISTRY:
        return _TTS_REGISTRY[base]()
    try:
        return _builtin_tts(base)
    except KeyError:
        builtins = sorted(set(_BUILTIN_TTS))
        raise ValueError(
            f"Unknown TTS backend {name!r}. Built-ins: {builtins}. Registered: {sorted(_TTS_REGISTRY)}."
        ) from None


def load_tts(backend: str, model: str | None = None, **config: Any) -> TextToSpeechModel:
    """Instantiate a TTS model for ``backend``.

    ``model`` is the engine's checkpoint id (or, for engines selected by language
    rather than weights, may be omitted to take the backend default). Does not
    call ``.load()`` — do that yourself (or use the model as a context manager)
    when ready to allocate.
    """
    cls = get_tts_backend(backend)
    return cls(model, **config) if model is not None else cls(**config)
