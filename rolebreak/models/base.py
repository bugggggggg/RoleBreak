"""Backend-agnostic speech-to-speech model interface.

Every backend (transformers, vLLM, a model-specific repo, ...) implements
:class:`SpeechToSpeechModel`. Today the contract is turn-based: feed a
conversation, get one spoken reply back. The streaming hooks below sketch the
full-duplex path we expect to add later; they raise until a backend supports
them, so turn-based callers are unaffected.
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from rolebreak.types import ChatRequest, ChatResponse, Message


class SpeechToSpeechModel(abc.ABC):
    """A speech-in, speech-out model.

    Subclasses load weights in ``load()`` (or lazily) and implement
    :meth:`generate`. Construction should stay cheap; defer heavy work to
    ``load()`` so the registry can build a model without paying for it.
    """

    #: Stable backend identifier, e.g. ``"qwen3-omni-api"`` / ``"moshi-api"``.
    backend: str = "base"

    def __init__(self, model: str, **config: Any) -> None:
        self.model = model
        self.config = config
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------- #
    def load(self) -> "SpeechToSpeechModel":
        """Load weights / connect to the serving backend. Idempotent."""
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:  # pragma: no cover - overridden by backends
        """Backend-specific load. Default no-op for lazy/remote backends."""

    def close(self) -> None:  # pragma: no cover - overridden by backends
        """Release GPU memory / connections."""

    def __enter__(self) -> "SpeechToSpeechModel":
        return self.load()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- turn-based inference ---------------------------------------------- #
    @abc.abstractmethod
    def generate(self, request: ChatRequest) -> ChatResponse:
        """Run one turn: OpenAI-style request in, spoken reply out."""

    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse:
        """Convenience wrapper around :meth:`generate`, timing the call.

        A backend that streams knows when its reply actually started and sets
        :attr:`~rolebreak.types.ChatResponse.latency` itself; for a one-shot
        backend the whole call *is* the wait — it is handed a finished clip and
        answers when it answers — so we time it from out here.
        """
        if not self._loaded:
            self.load()
        started = time.monotonic()
        response = self.generate(ChatRequest(messages=messages, model=self.model, **kwargs))
        if response.latency is None:
            response.latency = time.monotonic() - started
        return response

    # -- full-duplex (future) ---------------------------------------------- #
    @property
    def supports_full_duplex(self) -> bool:
        """Whether :meth:`stream_duplex` is implemented by this backend."""
        return False

    def stream_duplex(self, audio_in: AsyncIterator[bytes], **kwargs: Any) -> AsyncIterator[bytes]:
        """Full-duplex streaming: consume input audio chunks, yield output
        audio chunks concurrently. Not yet supported by any backend."""
        raise NotImplementedError(f"{type(self).__name__} does not support full-duplex streaming yet.")

    def stream_generate(self, request: ChatRequest, **kwargs: Any) -> Iterator[bytes]:
        """Turn-based streaming: yield output audio chunks for one reply.
        Optional; defaults to emitting the whole reply at once."""
        resp = self.generate(request)
        if resp.message.audio is not None:
            from rolebreak.models.audio import decode_b64

            yield decode_b64(resp.message.audio.data)
