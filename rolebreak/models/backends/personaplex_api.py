"""PersonaPlex (NVIDIA) speech-to-speech backend, talking to a running server.

PersonaPlex is a **full-duplex** conversational model built on the Moshi
architecture. Its server (``moshi.server``) speaks the exact same WebSocket wire
protocol as Kyutai's Moshi — a single ``/api/chat`` endpoint streaming 24 kHz
audio both ways in lock-step:

* server -> client handshake: a single ``\\x00`` byte on connect;
* audio frame: ``\\x01`` + Opus bytes (24 kHz mono, 80 ms frames);
* text token: ``\\x02`` + UTF-8 bytes.

Because the protocol is identical, this backend reuses **all** of the
duplex/VAD/turn-boundary machinery in :mod:`.moshi_api` and only adds what
PersonaPlex layers on top: **persona and voice control**, passed as query
parameters when the WebSocket is opened:

* ``text_prompt`` — the role/persona system prompt. Taken per-request from the
  request's ``system`` message (falling back to the ``text_prompt`` constructor
  arg). The server wraps it in ``<system> ... <system>`` tags itself.
* ``voice_prompt`` — the filename of a voice-conditioning embedding on the
  server (e.g. ``NATF2.pt``). PersonaPlex ships a fixed set: ``NATF0..3``,
  ``NATM0..3`` (natural female/male) and ``VARF0..4``, ``VARM0..4`` (variety).
  Selectable per-request via ``request.audio.voice``, else the ``voice_prompt``
  constructor arg.

Everything else — the turn-based :meth:`generate`, the multi-turn
:meth:`converse`, the silence/VAD turn cutting — is inherited unchanged from
:class:`~rolebreak.models.backends.moshi_api.MoshiAPIModel`.

Deploy the server first, e.g.::

    docker run --rm -it --name personaplex --gpus '"device=0"' --network host \\
      --env "HF_TOKEN=$HF_TOKEN" -v "$HOME/.cache:/root/.cache" \\
      rolebreak:personaplex \\
      uv run -m moshi.server --host 0.0.0.0

(PersonaPlex is a gated model — accept the license at
https://huggingface.co/nvidia/personaplex-7b-v1 and export ``HF_TOKEN``.)

Requires (install into your env): ``aiohttp``, ``numpy``, ``sphn``, and ``torch``
(for the Silero VAD used by :meth:`converse`)::

    uv pip install aiohttp numpy sphn torch
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
from urllib.parse import urlencode

import numpy as np

from rolebreak.models.backends.moshi_api import MoshiAPIModel
from rolebreak.types import AudioConfig, ChatRequest, ChatResponse


class PersonaplexAPIModel(MoshiAPIModel):
    """Turn-based adapter over a running full-duplex PersonaPlex server.

    Shares the entire duplex/VAD implementation with
    :class:`~rolebreak.models.backends.moshi_api.MoshiAPIModel`; the only
    additions are the ``text_prompt`` / ``voice_prompt`` query parameters
    PersonaPlex uses for persona and voice control.
    """

    backend = "personaplex-api"

    def __init__(
        self,
        model: str = "nvidia/personaplex-7b-v1",
        *,
        text_prompt: str = "",
        voice_prompt: str = "NATF2.pt",
        seed: int | None = None,
        **config: Any,
    ) -> None:
        # MoshiAPIModel builds ``self.url`` as ws://host:port/api/chat (with the
        # PersonaPlex default port 8998); we append the persona/voice query.
        super().__init__(model, **config)
        self.text_prompt = text_prompt
        self.voice_prompt = voice_prompt
        self.seed = seed
        # Base endpoint without query — the concrete URL is built per call so a
        # request's own system message / voice can override the constructor
        # defaults.
        self._base_url = self.url
        self.url = self._chat_url(self.text_prompt, self.voice_prompt)

    # -- inference --------------------------------------------------------- #
    def generate(self, request: ChatRequest) -> ChatResponse:
        text_prompt = self._extract_persona(request) or self.text_prompt
        voice = self._request_voice(request) or self.voice_prompt
        # generate() is one stateless session, so pin the URL to this request's
        # persona/voice before delegating to the inherited duplex driver.
        self.url = self._chat_url(text_prompt, voice)
        return super().generate(request)

    def converse(
        self,
        user_audios: Sequence[np.ndarray],
        *,
        text_prompt: str | None = None,
        voice_prompt: str | None = None,
    ) -> Iterator[ChatResponse]:
        """Multi-turn duplex conversation with a fixed persona and voice.

        A whole conversation is one WebSocket session, so the persona
        (``text_prompt``) and ``voice_prompt`` are fixed for its lifetime. Pass
        them here to override the constructor defaults for this conversation;
        otherwise the constructor values are used. Otherwise identical to
        :meth:`MoshiAPIModel.converse`.
        """
        text_prompt = self.text_prompt if text_prompt is None else text_prompt
        voice_prompt = self.voice_prompt if voice_prompt is None else voice_prompt
        self.url = self._chat_url(text_prompt, voice_prompt)
        yield from super().converse(user_audios)

    # -- helpers ----------------------------------------------------------- #
    def _chat_url(self, text_prompt: str, voice_prompt: str) -> str:
        """Build the ``/api/chat`` URL with PersonaPlex's query parameters.

        The server reads ``text_prompt`` and ``voice_prompt`` off the query
        string unconditionally (a missing key is a hard error), so both are
        always sent — an empty ``text_prompt`` simply means no persona. ``seed``
        is only appended when explicitly configured.
        """
        params: list[tuple[str, str]] = [
            ("text_prompt", text_prompt or ""),
            ("voice_prompt", voice_prompt or ""),
        ]
        if self.seed is not None:
            params.append(("seed", str(self.seed)))
        return f"{self._base_url}?{urlencode(params)}"

    @staticmethod
    def _extract_persona(request: ChatRequest) -> str | None:
        """Return the persona from the request's last ``system`` message, if any."""
        for msg in reversed(request.messages):
            if msg.role == "system" and isinstance(msg.content, str) and msg.content.strip():
                return msg.content.strip()
        return None

    @staticmethod
    def _request_voice(request: ChatRequest) -> str | None:
        """Return the voice-prompt filename requested via ``request.audio.voice``."""
        audio = request.audio
        if isinstance(audio, AudioConfig) and audio.voice:
            return audio.voice
        return None
