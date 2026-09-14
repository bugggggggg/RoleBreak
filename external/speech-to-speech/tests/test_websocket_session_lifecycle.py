import asyncio
from queue import Queue
from threading import Event, Thread

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.connections.websocket_streamer import WebSocketStreamer
from speech_to_speech.pipeline.control import END_OF_TURN, SESSION_END, is_control_message
from speech_to_speech.pipeline.messages import PIPELINE_END


class EchoHandler(BaseHandler):
    def setup(self):
        self.processed = []
        self.session_end_calls = 0

    def process(self, item):
        self.processed.append(item)
        yield item.upper()

    def on_session_end(self):
        self.session_end_calls += 1


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_base_handler_session_end_resets_without_stopping():
    stop_event = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = EchoHandler(stop_event, queue_in=queue_in, queue_out=queue_out)

    thread = Thread(target=handler.run)
    thread.start()

    queue_in.put(SESSION_END)
    queue_in.put("hello")
    queue_in.put(PIPELINE_END)

    thread.join(timeout=2)
    assert not thread.is_alive()

    outputs = [queue_out.get(timeout=1) for _ in range(3)]
    assert is_control_message(outputs[0], SESSION_END.kind)
    assert outputs[1] == "HELLO"
    assert outputs[2] == PIPELINE_END
    assert handler.processed == ["hello"]
    assert handler.session_end_calls == 1


def test_websocket_streamer_last_disconnect_queues_session_end():
    streamer = WebSocketStreamer(
        stop_event=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    asyncio.run(streamer._handle_client(FakeWebSocket([])))

    queued = streamer.input_queue.get_nowait()
    assert is_control_message(queued, SESSION_END.kind)
    assert streamer.should_listen.is_set()


def test_websocket_send_loop_ignores_session_end_until_stop():
    stop_event = Event()
    streamer = WebSocketStreamer(
        stop_event=stop_event,
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    async def exercise_send_loop():
        task = asyncio.create_task(streamer._send_loop())
        streamer.output_queue.put(SESSION_END)
        await asyncio.sleep(0.05)
        assert not task.done()
        stop_event.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise_send_loop())


class TurnHandler(BaseHandler):
    def setup(self):
        self.processed = []
        self.commits = 0

    def process(self, item):
        self.processed.append(item)
        return iter(())

    def on_end_of_turn(self):
        self.commits += 1
        yield f"turn-{self.commits}"


def test_base_handler_end_of_turn_emits_the_forced_turn_and_does_not_forward():
    """The commit produces a turn on the output queue; the message itself stops here."""
    stop_event = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = TurnHandler(stop_event, queue_in=queue_in, queue_out=queue_out)

    thread = Thread(target=handler.run)
    thread.start()

    queue_in.put("audio")
    queue_in.put(END_OF_TURN)
    queue_in.put(PIPELINE_END)

    thread.join(timeout=2)
    assert not thread.is_alive()

    outputs = [queue_out.get(timeout=1) for _ in range(2)]
    assert outputs[0] == "turn-1"
    assert outputs[1] == PIPELINE_END, "END_OF_TURN must not be forwarded downstream"
    assert handler.processed == ["audio"]
    assert handler.commits == 1


def test_base_handler_end_of_turn_is_a_no_op_for_a_handler_that_buffers_nothing():
    stop_event = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = EchoHandler(stop_event, queue_in=queue_in, queue_out=queue_out)

    thread = Thread(target=handler.run)
    thread.start()

    queue_in.put(END_OF_TURN)
    queue_in.put("hello")
    queue_in.put(PIPELINE_END)

    thread.join(timeout=2)
    assert not thread.is_alive()

    outputs = [queue_out.get(timeout=1) for _ in range(2)]
    assert outputs == ["HELLO", PIPELINE_END]


def test_websocket_end_of_turn_lands_behind_the_whole_clip():
    """The commit must be ordered after every sample of the clip it closes.

    The trailing 500 bytes are the tail of the utterance and are shorter than the
    VAD's 512-sample block, so they only reach it because the commit pads and
    flushes them -- otherwise the last phoneme is stranded in the buffer.
    """
    streamer = WebSocketStreamer(
        stop_event=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    clip = bytes(range(256)) * 4 + b"\x01" * 500  # one full 1024-byte block + a remainder
    socket = FakeWebSocket([clip, '{"type": "end_of_turn"}'])
    asyncio.run(streamer._handle_client(socket))

    queued = []
    while not streamer.input_queue.empty():
        queued.append(streamer.input_queue.get_nowait())

    assert queued[0] == clip[:1024], "the full block goes first"
    assert queued[1] == b"\x01" * 500 + bytes(524), "then the zero-padded remainder"
    assert is_control_message(queued[2], END_OF_TURN.kind), "then the commit"
    assert is_control_message(queued[3], SESSION_END.kind), "then the disconnect"
    assert socket.sent == ['{"type": "turn_committed"}']


def test_websocket_end_of_turn_without_a_remainder_queues_only_the_commit():
    streamer = WebSocketStreamer(
        stop_event=Event(),
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    clip = b"\x02" * 1024  # exactly one block, nothing left over
    asyncio.run(streamer._handle_client(FakeWebSocket([clip, '{"type": "end_of_turn"}'])))

    queued = []
    while not streamer.input_queue.empty():
        queued.append(streamer.input_queue.get_nowait())

    assert queued[0] == clip
    assert is_control_message(queued[1], END_OF_TURN.kind)
    assert is_control_message(queued[2], SESSION_END.kind)
