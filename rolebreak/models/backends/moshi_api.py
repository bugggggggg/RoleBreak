"""Moshi (Kyutai) speech-to-speech backend, talking to a running server.

Moshi is a **full-duplex** conversational model: the server (``moshi.server``)
exposes a single WebSocket endpoint ``/api/chat`` and streams audio both ways in
lock-step. There is no text prompt, no system persona, no temperature, and no
"one request -> one reply" — moshika's persona is baked into the checkpoint.

This adapter bridges that duplex stream onto RoleBreak's turn-based
:class:`SpeechToSpeechModel` contract: it streams the request's last user audio
into the server, then streams trailing silence so Moshi can finish talking, and
collects the returned speech (Opus) + text tokens as one assistant reply. The
reply is cut off with a simple silence-based VAD (stop once Moshi has produced
text and then gone quiet for ``silence_stop_s``), bounded by ``max_reply_s``.

Wire protocol (see ``external/moshi/moshi/moshi/server.py``):

* server -> client handshake: a single ``\\x00`` byte on connect;
* audio frame: ``\\x01`` + Opus bytes (24 kHz mono, 80 ms frames);
* text token: ``\\x02`` + UTF-8 bytes.

The adaptation is lossy — Moshi may speak over the user and timing is
approximate — but it is the standard way to benchmark Moshi offline.

Two ways to drive it:

* :meth:`MoshiAPIModel.generate` — the turn-based contract. Opens a fresh
  connection, sends the request's *last* user audio, and returns one reply. Each
  call is a new, stateless session.
* :meth:`MoshiAPIModel.converse` — a multi-turn interface for a whole
  conversation. It holds **one** connection open and streams every user turn's
  audio through it in order, yielding one reply per turn. Because the server only
  advances its state when fed a frame, the session's memory carries across turns
  and idling between them is safe. Turn boundaries are found with a streaming
  Silero VAD (:mod:`.vad_iterator`) on Moshi's own output: once it has spoken and
  gone quiet for ``silence_stop_s`` the turn is done. This is what
  ``tools/eval/eval_moshi.py`` uses.

Deploy the server first, e.g.::

    docker run --rm -it --name moshi --gpus '"device=0"' --network host \\
      -v "$HOME/.cache:/root/.cache" rolebreak:moshi \\
      uv run -p 3.10 -m moshi.server --host 0.0.0.0 \\
      --hf-repo kyutai/moshika-pytorch-bf16

Requires (install into your env): ``aiohttp``, ``numpy``, and ``sphn`` (the same
Opus codec the server/client use)::

    uv pip install aiohttp numpy sphn
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rolebreak.models.audio import decode_b64, output_audio_from_waveform
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import ChatRequest, ChatResponse, InputAudioContent, Message

# Fixed by Mimi / the Moshi server.
SAMPLE_RATE = 24_000
FRAME_SIZE = 1920  # 80 ms at 24 kHz

# Silero VAD (used for multi-turn turn-boundary detection) supports 8 k / 16 k
# only and, at 16 kHz, wants exactly 512-sample windows per call.
VAD_SAMPLE_RATE = 16_000
VAD_WINDOW = 512


class MoshiAPIModel(SpeechToSpeechModel):
    """Turn-based adapter over a running full-duplex Moshi WebSocket server.

    Construction is cheap; ``load()`` only validates config. The server must be
    reachable at ``url`` (or ``host``/``port``) before :meth:`generate` runs.
    """

    backend = "moshi-api"

    def __init__(
        self,
        model: str = "kyutai/moshika-pytorch-bf16",
        *,
        url: str | None = None,
        host: str = "localhost",
        port: int = 8998,
        silence_stop_s: float = 2.0,
        warmup_s: float = 2.0,
        min_reply_s: float = 1.0,
        max_reply_s: float = 30.0,
        connect_timeout_s: float = 30.0,
        vad_threshold: float = 0.5,
        prime: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(model, **config)
        self.url = url or f"ws://{host}:{port}/api/chat"
        self.silence_stop_s = silence_stop_s
        self.warmup_s = warmup_s
        self.min_reply_s = min_reply_s
        self.max_reply_s = max_reply_s
        self.connect_timeout_s = connect_timeout_s
        self.vad_threshold = vad_threshold
        self.prime = prime
        self._vad_model: Any = None  # lazily loaded Silero VAD (multi-turn only)

    # -- inference --------------------------------------------------------- #
    def generate(self, request: ChatRequest) -> ChatResponse:
        user_pcm = self._extract_user_pcm(request)
        pcm, transcript = asyncio.run(self._converse(user_pcm))
        return self._make_response(pcm, transcript)

    def converse(self, user_audios: Sequence[np.ndarray]) -> Iterator[ChatResponse]:
        """Stream a multi-turn conversation over a single duplex session.

        ``user_audios`` is one 24 kHz mono float PCM clip per user turn (decode
        files with :meth:`load_audio`). Yields one :class:`ChatResponse` per turn
        as Moshi finishes replying to it — the connection stays open across the
        whole sequence so the model's state carries between turns.

        This is a generator: iterate it to completion (or close it) so the
        underlying WebSocket is torn down. Turn boundaries use the Silero VAD;
        tune them with the ``silence_stop_s`` / ``min_reply_s`` / ``max_reply_s``
        / ``vad_threshold`` constructor args.

        Unless ``prime`` is ``False``, Moshi's spontaneous opening turn (it starts
        talking the moment you connect) is consumed and discarded first, so the
        first user turn gets a clean reply instead of one collected on top of that
        opening.
        """
        if not self._loaded:
            self.load()
        vad_model = self._load_vad()
        loop = asyncio.new_event_loop()
        session = DuplexSession(
            self.url,
            vad_model,
            vad_threshold=self.vad_threshold,
            silence_stop_s=self.silence_stop_s,
            warmup_s=self.warmup_s,
            min_reply_s=self.min_reply_s,
            max_reply_s=self.max_reply_s,
            connect_timeout_s=self.connect_timeout_s,
        )
        try:
            loop.run_until_complete(session.open())
            if self.prime:
                loop.run_until_complete(session.prime())  # drop Moshi's opening turn
            try:
                for pcm in user_audios:
                    clip = np.asarray(pcm, dtype=np.float32).reshape(-1)
                    reply_pcm, transcript, latency = loop.run_until_complete(session.send_turn(clip))
                    yield self._make_response(reply_pcm, transcript, latency)
            finally:
                loop.run_until_complete(session.close())
        finally:
            loop.close()

    def _make_response(self, pcm: np.ndarray, transcript: str, latency: float | None = None) -> ChatResponse:
        audio = output_audio_from_waveform(pcm, SAMPLE_RATE, transcript=transcript or None)
        return ChatResponse(
            message=Message(role="assistant", content=transcript or None, audio=audio),
            model=self.model,
            raw={"num_samples": int(pcm.shape[-1]), "text": transcript},
            latency=latency,
        )

    def _load_vad(self) -> Any:
        """Load the Silero VAD once (torch.hub, cached after the first download)."""
        if self._vad_model is None:
            import torch

            self._vad_model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True
            )
        return self._vad_model

    async def _converse(self, user_pcm: np.ndarray) -> tuple[np.ndarray, str]:
        """Drive one full-duplex exchange and return (reply_pcm, transcript)."""
        import aiohttp
        import sphn

        opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

        out_chunks: list[np.ndarray] = []
        text_parts: list[str] = []

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.connect_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(self.url) as ws:
                await self._await_handshake(ws)

                sender = asyncio.create_task(self._send_audio(ws, opus_writer, user_pcm))
                try:
                    await self._recv_loop(ws, opus_reader, out_chunks, text_parts)
                finally:
                    sender.cancel()
                    try:
                        await sender
                    except (asyncio.CancelledError, Exception):
                        pass

        pcm = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
        return pcm.astype(np.float32), "".join(text_parts).strip()

    async def _await_handshake(self, ws: Any) -> None:
        import aiohttp

        msg = await ws.receive(timeout=self.connect_timeout_s)
        if msg.type != aiohttp.WSMsgType.BINARY or not msg.data or msg.data[0] != 0x00:
            raise RuntimeError(f"unexpected handshake from Moshi server: {msg.type} {msg.data!r}")

    async def _send_audio(self, ws: Any, opus_writer: Any, user_pcm: np.ndarray) -> None:
        """Stream the user's audio, then a long silence tail so Moshi can reply.

        The receive loop is what actually decides when to stop; this task just
        keeps feeding silence (bounded by ``max_reply_s``) and is cancelled once
        the receiver has its answer.
        """
        # User speech first.
        for frame in self._frames(user_pcm):
            await self._send_pcm_frame(ws, opus_writer, frame)
            await asyncio.sleep(0)  # yield so the receiver interleaves

        # Trailing silence — enough to cover max_reply_s of Moshi talking.
        silence = np.zeros(FRAME_SIZE, dtype=np.float32)
        n_silence = int(self.max_reply_s * SAMPLE_RATE / FRAME_SIZE) + 4
        for _ in range(n_silence):
            await self._send_pcm_frame(ws, opus_writer, silence)
            await asyncio.sleep(0)

    async def _send_pcm_frame(self, ws: Any, opus_writer: Any, frame: np.ndarray) -> None:
        # sphn API differs across versions: newer builds return the Opus bytes
        # straight from append_pcm(); older ones require a separate read_bytes().
        data = opus_writer.append_pcm(frame)
        if data is None and hasattr(opus_writer, "read_bytes"):
            data = opus_writer.read_bytes()
        if data:
            await ws.send_bytes(b"\x01" + data)

    async def _recv_loop(
        self,
        ws: Any,
        opus_reader: Any,
        out_chunks: list[np.ndarray],
        text_parts: list[str],
    ) -> None:
        """Collect Moshi's speech/text until it goes quiet (VAD) or the cap hits."""
        import aiohttp

        produced_text = False
        samples_since_voice = 0
        total_samples = 0
        silence_stop = int(self.silence_stop_s * SAMPLE_RATE)
        min_samples = int(self.min_reply_s * SAMPLE_RATE)
        max_samples = int(self.max_reply_s * SAMPLE_RATE)
        # Absolute idle guard: if the server sends nothing at all, don't hang.
        recv_timeout = max(self.silence_stop_s * 2, 5.0)

        while True:
            try:
                msg = await ws.receive(timeout=recv_timeout)
            except (asyncio.TimeoutError, TimeoutError):
                break
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                break
            if msg.type != aiohttp.WSMsgType.BINARY or not msg.data:
                continue

            kind = msg.data[0]
            payload = msg.data[1:]
            if kind == 0x01:  # audio
                # Newer sphn returns PCM from append_bytes(); older needs read_pcm().
                pcm = opus_reader.append_bytes(payload)
                if pcm is None and hasattr(opus_reader, "read_pcm"):
                    pcm = opus_reader.read_pcm()
                if pcm is None or np.asarray(pcm).shape[-1] == 0:
                    continue
                pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
                out_chunks.append(pcm)
                total_samples += pcm.shape[-1]
                # Cheap energy-based VAD on Moshi's own output.
                if float(np.sqrt(np.mean(pcm**2))) > 1e-3:
                    samples_since_voice = 0
                else:
                    samples_since_voice += pcm.shape[-1]
            elif kind == 0x02:  # text token
                produced_text = True
                samples_since_voice = 0
                text_parts.append(payload.decode("utf-8", errors="replace"))

            if total_samples >= max_samples:
                break
            if produced_text and total_samples >= min_samples and samples_since_voice >= silence_stop:
                break

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _frames(pcm: np.ndarray):
        """Yield fixed-size mono frames, zero-padding the final partial frame."""
        n = pcm.shape[-1]
        for start in range(0, n, FRAME_SIZE):
            frame = pcm[start : start + FRAME_SIZE]
            if frame.shape[-1] < FRAME_SIZE:
                frame = np.pad(frame, (0, FRAME_SIZE - frame.shape[-1]))
            yield np.ascontiguousarray(frame, dtype=np.float32)

    @classmethod
    def load_audio(cls, source: str | Path | bytes, fmt: str = "wav") -> np.ndarray:
        """Decode an audio file — or a container's raw bytes — to mono float32 PCM at 24 kHz.

        Convenience for building the ``user_audios`` list passed to
        :meth:`converse`. A path takes its container format from its suffix;
        ``bytes`` declare theirs with ``fmt``, so a clip that is already in memory
        (a ``.wav`` member read out of a WebDataset shard, say) can be sent
        without being written out to a file first.
        """
        if isinstance(source, bytes | bytearray | memoryview):
            return cls._to_mono_24k(bytes(source), fmt)
        p = Path(source)
        return cls._to_mono_24k(p.read_bytes(), p.suffix.lstrip(".").lower() or fmt)

    def _extract_user_pcm(self, request: ChatRequest) -> np.ndarray:
        """Decode the last user audio in the request to 24 kHz mono float PCM."""
        audio_part = self._last_user_audio(request.messages)
        if audio_part is None:
            raise ValueError(
                "MoshiAPIModel needs user audio: no input_audio part found in the request. "
                "Moshi is audio-in/audio-out only — text-only turns are unsupported."
            )
        ia = audio_part.input_audio
        raw = decode_b64(ia.data)
        return self._to_mono_24k(raw, ia.format)

    @staticmethod
    def _last_user_audio(messages: list[Message]) -> InputAudioContent | None:
        for msg in reversed(messages):
            if msg.role != "user" or not isinstance(msg.content, list):
                continue
            for part in reversed(msg.content):
                if isinstance(part, InputAudioContent):
                    return part
        return None

    @staticmethod
    def _to_mono_24k(raw: bytes, fmt: str) -> np.ndarray:
        """Decode container bytes to mono float32 PCM at 24 kHz using sphn."""
        import sphn

        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(raw)
            tmp_path = Path(f.name)
        try:
            try:
                data, _ = sphn.read(str(tmp_path), sample_rate=SAMPLE_RATE)
            except TypeError:  # older sphn without resample kwarg
                data, src_sr = sphn.read(str(tmp_path))
                if src_sr != SAMPLE_RATE:
                    data = sphn.resample(data, src_sample_rate=src_sr, dst_sample_rate=SAMPLE_RATE)
        finally:
            tmp_path.unlink(missing_ok=True)

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:  # (channels, samples) -> mono
            arr = arr.mean(axis=0)
        return arr.reshape(-1)


# --------------------------------------------------------------------------- #
# Multi-turn duplex machinery (used by MoshiAPIModel.converse)
# --------------------------------------------------------------------------- #
class TurnEndDetector:
    """Streaming Silero VAD over Moshi's 24 kHz output, one reply turn at a time.

    Feed each decoded output chunk with :meth:`feed`; :attr:`ended` flips to
    ``True`` once Moshi has produced speech and then gone quiet for the
    configured silence (the VAD reports a completed utterance). Call
    :meth:`reset` between turns to clear state.
    """

    def __init__(self, vad_model: Any, *, threshold: float, silence_stop_s: float) -> None:
        from rolebreak.models.backends.vad_iterator import VADIterator

        self._iter = VADIterator(
            vad_model,
            threshold=threshold,
            sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=int(silence_stop_s * 1000),
            speech_pad_ms=30,
        )
        self._buf = np.zeros(0, dtype=np.float32)
        self.ended = False
        self.spoke = False

    def reset(self) -> None:
        self._iter.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
        self.ended = False
        self.spoke = False

    def feed(self, pcm24: np.ndarray) -> None:
        import sphn
        import torch

        if pcm24.shape[-1] == 0:
            return
        pcm16 = np.asarray(
            sphn.resample(pcm24, src_sample_rate=SAMPLE_RATE, dst_sample_rate=VAD_SAMPLE_RATE),
            dtype=np.float32,
        ).reshape(-1)
        self._buf = np.concatenate([self._buf, pcm16])
        while self._buf.shape[-1] >= VAD_WINDOW:
            window = np.ascontiguousarray(self._buf[:VAD_WINDOW])
            self._buf = self._buf[VAD_WINDOW:]
            utterance = self._iter(torch.from_numpy(window))
            if self._iter.triggered:
                self.spoke = True
            if utterance is not None:  # speech -> silence: a turn just closed
                self.ended = True


class DuplexSession:
    """One Moshi WebSocket held open across every turn of a conversation.

    :meth:`send_turn` streams a user clip, then paces trailing silence while
    reading Moshi's reply, stopping when the VAD reports the turn is over. Send
    and receive stay balanced (one frame sent, one message read) so no output is
    left buffered between turns — that is what keeps the multi-turn session in
    sync. Built and driven by :meth:`MoshiAPIModel.converse`.
    """

    def __init__(
        self,
        url: str,
        vad_model: Any,
        *,
        vad_threshold: float,
        silence_stop_s: float,
        warmup_s: float,
        min_reply_s: float,
        max_reply_s: float,
        connect_timeout_s: float,
    ) -> None:
        self.url = url
        self.silence_stop_s = silence_stop_s
        self.warmup_s = warmup_s
        self.min_reply_s = min_reply_s
        self.max_reply_s = max_reply_s
        self.connect_timeout_s = connect_timeout_s
        self._detector = TurnEndDetector(vad_model, threshold=vad_threshold, silence_stop_s=silence_stop_s)
        self._session: Any = None
        self._ws: Any = None
        self._opus_writer: Any = None
        self._opus_reader: Any = None

    async def open(self) -> "DuplexSession":
        import aiohttp
        import sphn

        self._opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        self._opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.connect_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._ws = await self._session.ws_connect(self.url)
        await self._await_handshake()
        return self

    async def close(self) -> None:
        # Moshi keeps emitting output for the frames we already sent. Drain that
        # in-flight audio first so the server goes idle and isn't mid-send when we
        # close — otherwise its recv_loop dies with "Cannot write to closing
        # transport". Swallow errors: on the way down the connection may be gone.
        if self._ws is not None and not self._ws.closed:
            try:
                await self._drain(0.5)
                await self._ws.close()
            except Exception:
                pass
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "DuplexSession":
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _await_handshake(self) -> None:
        import aiohttp

        msg = await self._ws.receive(timeout=self.connect_timeout_s)
        if msg.type != aiohttp.WSMsgType.BINARY or not msg.data or msg.data[0] != 0x00:
            raise RuntimeError(f"unexpected handshake from Moshi server: {msg.type} {msg.data!r}")

    async def _drain(self, timeout: float) -> None:
        """Read and discard buffered output until the server falls quiet.

        We send no input while draining, so Moshi stops after flushing whatever
        it had in flight; the receive then times out and we return.
        """
        import aiohttp

        while True:
            try:
                msg = await self._ws.receive(timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError):
                return
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                return

    async def _send_pcm_frame(self, frame: np.ndarray) -> None:
        # See MoshiAPIModel._send_pcm_frame for the sphn version differences.
        data = self._opus_writer.append_pcm(frame)
        if data is None and hasattr(self._opus_writer, "read_bytes"):
            data = self._opus_writer.read_bytes()
        if data:
            await self._ws.send_bytes(b"\x01" + data)

    def _decode_audio(self, payload: bytes) -> np.ndarray:
        pcm = self._opus_reader.append_bytes(payload)
        if pcm is None and hasattr(self._opus_reader, "read_pcm"):
            pcm = self._opus_reader.read_pcm()
        if pcm is None:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(pcm, dtype=np.float32).reshape(-1)

    async def prime(self) -> tuple[np.ndarray, str, float | None]:
        """Consume Moshi's spontaneous opening turn before the first user turn.

        Moshi starts talking as soon as the connection opens, unprompted. This
        runs a turn with an empty user clip — feeding only silence — so that
        opening is captured (and then discarded by the caller) via the usual VAD
        boundary, leaving the session listening for the first real user turn.
        Returns the discarded opening as ``(pcm, transcript, latency)`` for
        inspection.
        """
        return await self.send_turn(np.zeros(0, dtype=np.float32))

    async def send_turn(self, user_pcm: np.ndarray) -> tuple[np.ndarray, str, float | None]:
        """Stream one user clip and collect Moshi's spoken reply.

        Returns ``(reply_pcm_24k, transcript, latency_s)``. The connection is left
        open and in sync for the next turn. An empty ``user_pcm`` feeds only
        silence, used by :meth:`prime` to soak up Moshi's unprompted opening turn.

        The latency is wall clock: how long we actually waited, from sending the
        last user frame to the message carrying Moshi's first speech coming back.
        The loop below is lock-step — one frame in, one message out — so that
        span is the server's compute for the frames it needed before it started
        answering, which is what a caller feels.
        """
        import aiohttp

        self._detector.reset()
        out_chunks: list[np.ndarray] = []
        text_parts: list[str] = []
        produced_text = False
        # Monotonic timestamps: when the user stopped talking (so the wait
        # starts), and when the first speech the VAD hears / the first text token
        # arrived. Text is the fallback: it is no use to a listener, but a turn
        # that produced only text still says something about how long the model
        # took.
        user_done_at: float | None = None
        first_audio_at: float | None = None
        first_text_at: float | None = None
        # Reply length is measured from *after* the user clip — the output Moshi
        # streams while it's still hearing the user is listening-silence, not a
        # reply, so it must not count toward the floor or the saved audio.
        reply_samples = 0
        warmup_samples = int(self.warmup_s * SAMPLE_RATE)
        min_samples = int(self.min_reply_s * SAMPLE_RATE)
        max_samples = int(self.max_reply_s * SAMPLE_RATE)
        recv_timeout = max(self.silence_stop_s * 2, 5.0)

        user_frames = list(MoshiAPIModel._frames(user_pcm))
        silence = np.zeros(FRAME_SIZE, dtype=np.float32)

        i = 0
        while reply_samples < max_samples:
            # Send the next user frame, or silence once the clip is exhausted.
            in_user_turn = i < len(user_frames)
            if not in_user_turn and user_done_at is None:
                user_done_at = time.monotonic()  # the listener starts waiting here
            try:
                await self._send_pcm_frame(user_frames[i] if in_user_turn else silence)
            except (ConnectionResetError, aiohttp.ClientError):
                break  # server dropped the socket — end the turn with what we have
            i += 1

            # Read the output the server produced for that frame (lock-step).
            try:
                msg = await self._ws.receive(timeout=recv_timeout)
            except (asyncio.TimeoutError, TimeoutError):
                break
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                break
            if msg.type != aiohttp.WSMsgType.BINARY or not msg.data:
                continue

            kind = msg.data[0]
            payload = msg.data[1:]
            if kind == 0x01:  # audio
                pcm = self._decode_audio(payload)
                # Only start collecting / VAD-ing once the user is done speaking.
                if pcm.shape[-1] and not in_user_turn:
                    out_chunks.append(pcm)
                    reply_samples += pcm.shape[-1]
                    self._detector.feed(pcm)
                    if first_audio_at is None and self._detector.spoke:
                        first_audio_at = time.monotonic()
            elif kind == 0x02 and not in_user_turn:  # text token
                produced_text = True
                if first_text_at is None:
                    first_text_at = time.monotonic()
                text_parts.append(payload.decode("utf-8", errors="replace"))

            # Hold the turn open through the warmup window so Moshi has time to
            # start talking, then close once it has spoken, cleared the floor,
            # and the VAD has seen it go quiet.
            spoke = produced_text or self._detector.spoke
            past_warmup = reply_samples >= warmup_samples
            if not in_user_turn and past_warmup and spoke and reply_samples >= min_samples and self._detector.ended:
                break

        # Flush the audio Moshi already had in flight for the frames we sent, so
        # it isn't left buffered to bleed into (or desync) the next turn while the
        # driver writes files. We send no input, so this drains and stops quickly.
        with contextlib.suppress(Exception):
            await self._drain(min(self.silence_stop_s, 0.5))

        reply_at = first_audio_at if first_audio_at is not None else first_text_at
        latency = reply_at - user_done_at if reply_at is not None and user_done_at is not None else None
        pcm = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
        return pcm.astype(np.float32), "".join(text_parts).strip(), latency
