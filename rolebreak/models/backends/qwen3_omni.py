"""Qwen3-Omni (MoE) speech-to-speech backend, served over an OpenAI-compatible API.

``Qwen/Qwen3-Omni-30B-A3B-Instruct`` perceives audio (also image/video) and
emits both text and speech. We drive it out-of-process: run the checkpoint
behind ``vllm serve`` and point this backend at that endpoint — the model's
shipped voices are Chelsie / Ethan / Aiden, default ``Ethan``.

Serving quirks (why this is more than a pass-through) are documented on
:class:`Qwen3OmniAPIModel`.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

from rolebreak.models.audio import decode_b64, encode_bytes
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import (
    ChatRequest,
    ChatResponse,
    Message,
    OutputAudio,
)

DEFAULT_VOICE = "Ethan"  # other shipped voices: "Chelsie", "Aiden"

# A locally deployed `vllm serve` OpenAI-compatible endpoint for Qwen3-Omni.
DEFAULT_API_BASE_URL = "http://localhost:8901/v1"

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
# A cold three-stage Omni server needs several minutes to come back: it reloads
# 30B of weights, recompiles, and recaptures CUDA graphs per stage. Waiting is
# the whole point of the retry, so the budget is generous.
DEFAULT_HEALTH_TIMEOUT = 900.0
HEALTH_POLL_SECONDS = 10.0
# Let the freshly-restarted engine settle before the next request: /health answers
# as soon as the API server is up, which is a little before the stages are warm.
RETRY_SETTLE_SECONDS = 5.0

# Transport failures that mean "the server went away mid-request", not "this
# request is bad" — the stream is cut when the engine takes the process down.
_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    socket.timeout,
    TimeoutError,
)

# An in-stream error is worth another attempt only when the *server* broke.
# vllm-omni's talker (stage 1) dies on a CUDA illegal-memory-access under load,
# taking the whole orchestrator with it; the container restarts and the same
# request then succeeds. See docs/models.md.
_ENGINE_DEATH_MARKERS = (
    "enginecore",
    "engine core",
    "engine dead",
    "enginedead",
    "illegal memory access",
    "cuda error",
)
# A request the server refuses on its merits fails identically every time, so
# retrying only burns the budget and hides the real cause.
_FATAL_MARKERS = (
    "context",
    "maximum",
    "too long",
    "too large",
    "invalid",
)


class Qwen3OmniAPIModel(SpeechToSpeechModel):
    """Qwen3-Omni via a locally deployed vLLM OpenAI-compatible endpoint.

    This backend keeps no weights: it forwards our OpenAI-shaped
    :class:`ChatRequest` to a running ``vllm serve`` at ``base_url`` and parses
    the chat-completion back.

    Two quirks of the ``vllm-omni`` build make this more than a pass-through when
    speech is requested:

    * the talker reads its voice from a **top-level ``speaker``** field, not from
      OpenAI's ``audio.voice`` — so we mirror the voice there;
    * the non-streaming ``message.audio`` is unreliable (usually ``null``);
      audio only comes back when we **stream**, arriving as base64 WAV segments
      interleaved into ``delta.content`` (each a full WAV, flagged by a
      re-announced ``role: assistant``). We collect those segments and
      concatenate their PCM into one WAV.

    Text-only requests (no ``audio`` modality) take the plain non-streaming path.
    """

    backend = "qwen3-omni-api"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        **config: Any,
    ) -> None:
        super().__init__(model, base_url=base_url, **config)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.health_timeout = health_timeout

    def generate(self, request: ChatRequest) -> ChatResponse:
        payload = request.to_dict()
        payload["model"] = self.model
        # Default the voice so spoken replies use a shipped Qwen3-Omni speaker.
        audio = payload.setdefault("audio", {})
        audio.setdefault("voice", DEFAULT_VOICE)

        if "audio" in payload.get("modalities", []):
            # The talker reads its voice from a top-level ``speaker`` field, and
            # only emits audio over a stream — see the class docstring.
            payload["speaker"] = audio.get("voice") or DEFAULT_VOICE
            payload["stream"] = True
        return self._generate_with_retry(payload)

    # -- resilience --------------------------------------------------------- #
    def _dispatch(self, payload: dict[str, Any]) -> ChatResponse:
        """One attempt at the turn, over whichever path the payload asks for."""
        if payload.get("stream"):
            return self._generate_stream(payload)
        return self._parse_completion(self._post(payload))

    def _generate_with_retry(self, payload: dict[str, Any]) -> ChatResponse:
        """Re-issue the turn when the *server* dies under it.

        The talker stage of vllm-omni crashes the whole engine on a CUDA fault
        every few hundred requests, which ends a long ``--all`` eval partway
        through. Re-issuing is sound here because this backend keeps no
        server-side session: every turn already resends the full windowed
        history as audio, so an attempt against a restarted server is
        indistinguishable from a first attempt. A partial reply from the killed
        stream is discarded rather than stitched.

        The retry count rides back on :attr:`ChatResponse.raw` so a run can log
        that a turn needed one — a recovered turn should be visible in the
        transcript, not silently identical to a clean one.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self._dispatch(payload)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                logger.warning(
                    "%s: attempt %d/%d failed (%s); waiting for the server to come back",
                    self.backend,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if self._wait_for_health():
                    self._warmup()
                continue
            if attempt:
                response.raw = {**(response.raw or {}), "retries": attempt}
            return response
        raise AssertionError("unreachable: the loop either returns or raises")

    def _is_retryable(self, exc: BaseException) -> bool:
        """Whether ``exc`` looks like a dead server rather than a bad request."""
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code >= 500  # a 4xx is our fault and will not improve
        if isinstance(exc, _TRANSPORT_ERRORS):
            return True
        if isinstance(exc, RuntimeError):
            # An in-stream failure: vLLM answers 200 before generation runs, so a
            # mid-generation crash arrives as a payload, not an HTTP status.
            message = str(exc).lower()
            if any(marker in message for marker in _FATAL_MARKERS):
                return False
            return any(marker in message for marker in _ENGINE_DEATH_MARKERS)
        return False

    def _warmup(self) -> bool:
        """Run one throwaway turn so the retried request does not pay cold-start costs.

        ``/health`` answers as soon as the API server binds its port, which is
        before the stages have JIT-compiled their Triton kernels — a restarted
        server logs ``Triton kernel JIT compilation during inference`` on its
        first real request and stalls for it. Spending that on a turn whose
        answer we discard keeps it out of the eval's latency numbers, and keeps a
        slow first reply from tripping the request timeout.

        Best-effort by design: a failed warmup says nothing the next real attempt
        would not say better, so it logs and returns instead of raising.
        """
        # Short prompt, both modalities, so all three stages (thinker, talker,
        # code2wav) are exercised.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Introduce yourself in one short sentence."}],
            "modalities": ["text", "audio"],
            "speaker": "chelsie",
        }
        started = time.monotonic()
        try:
            self._post(payload)
        except Exception as exc:  # noqa: BLE001 - warmup is advisory, never fatal
            logger.warning("%s: warmup turn failed (%s); trying the real request anyway", self.backend, exc)
            return False
        logger.info("%s: warmup turn done in %.1fs", self.backend, time.monotonic() - started)
        return True

    def _health_url(self) -> str:
        """``/health`` on the endpoint's origin — it sits beside ``/v1``, not under it."""
        parts = urllib.parse.urlsplit(self.base_url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    def _wait_for_health(self) -> bool:
        """Poll ``/health`` until the restarted server answers, or give up.

        Giving up does not raise: the next attempt's own failure is a better
        error to surface than one from this helper.
        """
        url = self._health_url()
        deadline = time.monotonic() + self.health_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10.0) as resp:
                    if resp.status < 400:
                        logger.info("%s: server healthy again at %s", self.backend, url)
                        time.sleep(RETRY_SETTLE_SECONDS)
                        return True
            except Exception:  # noqa: BLE001 - still down; keep polling until the deadline
                pass
            time.sleep(HEALTH_POLL_SECONDS)
        logger.error(
            "%s: server did not answer %s within %.0fs",
            self.backend,
            url,
            self.health_timeout,
        )
        return False

    # -- transport --------------------------------------------------------- #
    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _generate_stream(self, payload: dict[str, Any]) -> ChatResponse:
        """Consume the SSE stream, splitting text deltas from base64 WAV segments.

        Streaming is also the only place this backend can time the reply: the
        turn-based contract hands the model a finished clip, so the wait starts
        when the request goes out and ends at the talker's first WAV segment —
        or, for a turn that never produced one, the thinker's first text.
        """
        text_parts: list[str] = []
        wav_segments: list[bytes] = []
        started = time.monotonic()
        first_text_s: float | None = None
        first_audio_s: float | None = None
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                if "choices" not in data:
                    raise RuntimeError(self._stream_error(data))
                if not data["choices"]:
                    continue  # usage-only final chunk carries no delta
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue
                # Base64 WAV segments start with "UklGR" ("RIFF"); else it's text.
                if content.startswith("UklGR"):
                    wav_segments.append(decode_b64(content))
                    if first_audio_s is None:
                        first_audio_s = time.monotonic() - started
                else:
                    text_parts.append(content)
                    if first_text_s is None:
                        first_text_s = time.monotonic() - started

        transcript = "".join(text_parts) or None
        audio = None
        if wav_segments:
            audio = OutputAudio(
                data=encode_bytes(_merge_wav_segments(wav_segments)),
                format="wav",
                transcript=transcript,
            )
        return ChatResponse(
            message=Message(role="assistant", content=transcript, audio=audio),
            model=self.model,
            raw={"text": transcript, "audio_segments": len(wav_segments)},
            latency=first_audio_s if first_audio_s is not None else first_text_s,
        )

    def _stream_error(self, data: dict[str, Any]) -> str:
        """Describe a ``data:`` payload that carries a failure instead of a chunk.

        vLLM answers 200 and opens the stream before generation runs, so a
        failure raised partway through (context overflow, a talker crash) can
        only be reported inside the stream: it yields one final payload shaped
        ``{"object": "error", "message": ...}`` with no ``choices``.
        """
        err = data.get("error", data)
        detail = err.get("message") if isinstance(err, dict) else err
        return f"{self.backend} stream failed at {self.base_url}: {detail or json.dumps(data)[:500]}"

    def _parse_completion(self, data: dict[str, Any]) -> ChatResponse:
        if "choices" not in data:
            raise RuntimeError(self._stream_error(data))
        choice = data["choices"][0]["message"]
        audio = OutputAudio.from_dict(choice["audio"]) if choice.get("audio") else None
        return ChatResponse(
            message=Message(role="assistant", content=choice.get("content"), audio=audio),
            model=data.get("model", self.model),
            raw=data,
        )


def _merge_wav_segments(segments: list[bytes]) -> bytes:
    """Concatenate the PCM frames of several WAV byte-strings into one WAV.

    Each streamed segment is a complete standalone WAV; joining them at the raw
    bytes level would leave stray headers mid-stream, so we read each with
    :mod:`wave` and rewrite the merged frames under a single header.
    """
    params = None
    frames = bytearray()
    for seg in segments:
        with wave.open(io.BytesIO(seg), "rb") as w:
            if params is None:
                params = w.getparams()
            frames += w.readframes(w.getnframes())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()
