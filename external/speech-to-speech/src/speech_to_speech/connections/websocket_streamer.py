import asyncio
import json
import logging
from queue import Empty, Queue
from threading import Event
from typing import Any

import numpy as np
from websockets.asyncio.server import ServerConnection

from speech_to_speech.pipeline.control import END_OF_TURN, SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import PipelineEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem, TextEventItem

logger = logging.getLogger(__name__)


class WebSocketStreamer:
    """
    Handles bidirectional audio streaming over WebSocket.

    Receives audio from clients and puts it in the input_queue.
    Sends audio from the output_queue to clients.
    Sends text messages (transcripts/tools) from text_output_queue to clients.
    """

    def __init__(
        self,
        stop_event: Event,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: Event,
        text_output_queue: Queue[TextEventItem] | None = None,
        runtime_config: Any = None,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.stop_event = stop_event
        self.input_queue = input_queue  # clients -> VAD
        self.output_queue = output_queue  # TTS -> clients
        self.text_output_queue = text_output_queue  # Text messages -> clients
        # Shared RuntimeConfig; a client may set the persona over the wire by
        # sending a ``{"type": "text_prompt", "text": ...}`` JSON message, which
        # we write into ``session.instructions``. None disables the feature.
        self.runtime_config = runtime_config
        # The server's configured default persona, captured before any client can
        # override it. Restored on each new session so a prior conversation's
        # text_prompt does not leak into the next.
        self._default_instructions = runtime_config.session.instructions if runtime_config is not None else None
        self.should_listen = should_listen
        self.host = host
        self.port = port
        self.clients: set[ServerConnection] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server: Any = None

    def run(self) -> None:
        """Run the WebSocket server (called from a thread)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self.loop.close()

    async def _run_server(self) -> None:
        """Main async server loop."""
        import websockets

        logger.info(f"WebSocket server starting on ws://{self.host}:{self.port}")

        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
        )

        logger.info("WebSocket server ready, waiting for connections...")

        # Start the sender task
        sender_task = asyncio.create_task(self._send_loop())

        # Wait until stop_event is set
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

        # Cleanup
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # Close all clients
        for client in list(self.clients):
            try:
                await client.close()
            except Exception:
                pass

        self.server.close()
        await self.server.wait_closed()
        logger.info("WebSocket server closed")

    async def _handle_client(self, websocket: ServerConnection) -> None:
        """Handle a single WebSocket client connection."""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected")
        self.clients.add(websocket)
        recv_buffer = bytearray()

        # Enable listening when first client connects. This makes a new WebSocket
        # connection a clean-slate handshake, like Moshi/PersonaPlex — the client
        # opens one connection per sample. It is best-effort only: it is skipped
        # entirely if the previous client's socket has not been reaped yet (then
        # this connection is the *second* one, not the first), so a client that
        # needs a guarantee should send ``{"type": "reset"}`` instead.
        if len(self.clients) == 1:
            self._reset_session("new connection")
            self.should_listen.set()
            logger.debug("Listening enabled, edge queues drained (should_listen.set())")

        try:
            logger.debug(f"Client {client_id}: Starting message receive loop")
            async for message in websocket:
                if isinstance(message, str):
                    ack = self._handle_text_message(client_id, message, recv_buffer)
                    if ack is not None:
                        await websocket.send(ack)
                    continue
                if isinstance(message, bytes):
                    logger.debug(f"Client {client_id}: Received {len(message)} bytes of audio")
                    if self.should_listen.is_set():
                        # Split into 512-sample (1024 bytes) chunks for VAD.
                        # Keep a per-client remainder buffer so no samples are dropped
                        # when WebSocket frame boundaries are not aligned.
                        chunk_size_bytes = 512 * 2  # 512 samples * 2 bytes per int16
                        recv_buffer.extend(message)
                        num_chunks = 0
                        while len(recv_buffer) >= chunk_size_bytes:
                            chunk = bytes(recv_buffer[:chunk_size_bytes])
                            del recv_buffer[:chunk_size_bytes]
                            self.input_queue.put(chunk)
                            num_chunks += 1
                        logger.debug(f"Client {client_id}: Queued {num_chunks} chunks for processing")
                    else:
                        logger.debug(f"Client {client_id}: Skipping audio (should_listen not set)")

        except Exception as e:
            logger.error(f"Client {client_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected (finally block)")

            if len(self.clients) == 0:
                logger.debug("Last WebSocket client disconnected, ending session")
                self.input_queue.put(SESSION_END)

    def _reset_session(self, reason: str, *, end_session: bool = False) -> None:
        """Drop every trace of the previous conversation.

        Three things carry across conversations and all three are cleared here:

        * the **edge queues** — a previous conversation's trailing TTS audio and
          its unread events would otherwise be read as this conversation's first
          reply (``SESSION_END`` may not have flushed everything yet);
        * the **chat buffer** — it is process-global, so without a reset the next
          conversation is a continuation of the last one;
        * the **persona** — restored to the server's configured default, so a
          prior conversation's ``text_prompt`` does not leak. A fresh
          ``text_prompt`` may follow to override it.

        With ``end_session`` a ``SESSION_END`` is also pushed into the input
        queue, so the handlers themselves (VAD segmentation state, STT revision
        keys, TTS voice overrides) drop their per-session state. The input queue
        is FIFO, so it is ordered ahead of any audio this connection sends next.
        """
        for q in (self.output_queue, self.text_output_queue):
            if q is not None:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Empty:
                        break
        if end_session:
            self.input_queue.put(SESSION_END)
        if self.runtime_config is not None:
            self.runtime_config.chat.reset()
            self.runtime_config.session.instructions = self._default_instructions
        logger.info(f"Session reset ({reason}): chat history cleared, persona restored to default")

    def _handle_text_message(self, client_id: int, message: str, recv_buffer: bytearray | None = None) -> str | None:
        """Handle a JSON text message from a client; return a reply to send, if any.

        Supports:

        * ``{"type": "text_prompt", "text": <persona>}`` — sets the LLM's system
          prompt (persona) for the rest of the session by writing into the shared
          ``RuntimeConfig``. The LLM handler re-reads ``session.instructions``
          every turn, so this takes effect immediately.
        * ``{"type": "reset"}`` — clears the conversation unconditionally and
          replies ``{"type": "session_reset"}``. Unlike the connect-time reset
          this does not depend on the previous client having been reaped, so a
          benchmark client that opens one connection per sample can guarantee
          that a sample never sees the previous one's history. Messages on a
          connection are processed in order, so a ``text_prompt`` sent after a
          ``reset`` survives it.
        * ``{"type": "end_of_turn"}`` — the user has stopped speaking. Closes
          the turn the VAD has buffered and sends it downstream, and replies
          ``{"type": "turn_committed"}``. This is the boundary a client that
          *knows* where its turn ends should use instead of letting the silence
          heuristic guess; pair it with ``--manual_turn_end`` so the heuristic
          cannot fire first. Harmless without that flag, where it only flushes a
          trailing utterance the VAD has not closed yet.
        """
        try:
            event = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Client {client_id}: Ignoring non-JSON text message")
            return None
        if not isinstance(event, dict):
            return None
        if event.get("type") == "reset":
            self._reset_session(f"client {client_id} request", end_session=True)
            self.should_listen.set()
            return json.dumps({"type": "session_reset"})
        if event.get("type") == "end_of_turn":
            # Whatever is left in the remainder buffer is the tail of this very
            # utterance — under 512 samples, but that is its last phoneme. The
            # VAD only consumes whole 512-sample blocks, so pad it out rather
            # than let the commit strand it.
            if recv_buffer:
                chunk_size_bytes = 512 * 2
                self.input_queue.put(bytes(recv_buffer).ljust(chunk_size_bytes, b"\x00"))
                recv_buffer.clear()
            # FIFO, so this lands behind every audio block already queued: the
            # VAD sees the whole clip before it is told the clip is over.
            self.input_queue.put(END_OF_TURN)
            logger.debug(f"Client {client_id}: end_of_turn committed")
            return json.dumps({"type": "turn_committed"})
        if event.get("type") == "text_prompt":
            text = event.get("text")
            if not isinstance(text, str):
                logger.warning(f"Client {client_id}: text_prompt missing string 'text' field")
                return None
            if self.runtime_config is None:
                logger.warning(f"Client {client_id}: text_prompt received but no runtime_config to apply it to")
                return None
            self.runtime_config.session.instructions = text
            logger.info(f"Client {client_id}: persona set via text_prompt ({len(text)} chars)")
            return None
        logger.debug(f"Client {client_id}: Ignoring text message of type {event.get('type')!r}")
        return None

    async def _send_loop(self) -> None:
        """Send audio and text from queues to all connected clients."""
        # Buffer audio until we have at least 100ms worth (3200 bytes = 1600 samples at 16kHz int16)
        MIN_AUDIO_BYTES = 3200
        audio_buffer = bytearray()

        while not self.stop_event.is_set():
            try:
                # Check for audio
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, bytes) and audio_chunk == PIPELINE_END:
                        if audio_buffer and self.clients:
                            data = bytes(audio_buffer)
                            audio_buffer.clear()
                            await asyncio.gather(
                                *[client.send(data) for client in self.clients],
                                return_exceptions=True,
                            )
                        break
                    if isinstance(audio_chunk, bytes) and audio_chunk == AUDIO_RESPONSE_DONE:
                        if audio_buffer and self.clients:
                            data = bytes(audio_buffer)
                            audio_buffer.clear()
                            await asyncio.gather(
                                *[client.send(data) for client in self.clients],
                                return_exceptions=True,
                            )
                        self.should_listen.set()
                        logger.debug("Response complete, listening re-enabled")
                        continue
                    if is_control_message(audio_chunk, SESSION_END.kind):
                        audio_buffer.clear()
                        continue

                    if isinstance(audio_chunk, PipelineControlMessage):
                        continue

                    if self.clients:
                        chunk_bytes: bytes
                        if isinstance(audio_chunk, bytes):
                            chunk_bytes = audio_chunk
                        elif isinstance(audio_chunk, np.ndarray):
                            chunk_bytes = audio_chunk.tobytes()
                        elif hasattr(audio_chunk, "tobytes"):
                            chunk_bytes = audio_chunk.tobytes()
                        else:
                            continue
                        audio_buffer.extend(chunk_bytes)

                        if len(audio_buffer) >= MIN_AUDIO_BYTES:
                            data = bytes(audio_buffer)
                            audio_buffer.clear()
                            logger.debug(f"Sending {len(data)} bytes of audio to {len(self.clients)} client(s)")
                            await asyncio.gather(
                                *[client.send(data) for client in self.clients], return_exceptions=True
                            )
                except Empty:
                    # Flush any buffered audio when queue is empty
                    if audio_buffer and self.clients:
                        data = bytes(audio_buffer)
                        audio_buffer.clear()
                        logger.debug(f"Flushing {len(data)} bytes of audio to {len(self.clients)} client(s)")
                        await asyncio.gather(*[client.send(data) for client in self.clients], return_exceptions=True)

                # Check for text/tool messages
                if self.text_output_queue:
                    try:
                        text_message = self.text_output_queue.get_nowait()
                        if self.clients:
                            if isinstance(text_message, PipelineEvent):
                                payload = text_message.model_dump()
                                await asyncio.gather(
                                    *[client.send(json.dumps(payload)) for client in self.clients],
                                    return_exceptions=True,
                                )
                            elif isinstance(text_message, (PipelineControlMessage, bytes)):
                                continue
                    except Empty:
                        pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")
