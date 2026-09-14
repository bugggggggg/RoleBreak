import torch

from speech_to_speech.VAD.vad_iterator import VADIterator


class _FakeVADModel:
    def __init__(self, probs: list[float]) -> None:
        self._probs = iter(probs)

    def reset_states(self) -> None:
        pass

    def __call__(self, x: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        return torch.tensor(next(self._probs), dtype=torch.float32)


def _finish_utterance(iterator: VADIterator, silence_chunk: torch.Tensor):
    spoken_utterance = None
    for _ in range(5):
        spoken_utterance = iterator(silence_chunk)
        if spoken_utterance is not None:
            break
    return spoken_utterance


def test_triggering_chunk_is_kept_in_buffer() -> None:
    model = _FakeVADModel([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None
    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 7
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert all(torch.equal(chunk, silence_chunk) for chunk in spoken_utterance[2:])


def test_pre_speech_padding_is_prepended_to_final_utterance() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=64,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    third_chunk = torch.ones(512) * 3
    fourth_chunk = torch.ones(512) * 4
    silence_chunk = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None
    assert iterator(third_chunk) is None
    assert iterator(fourth_chunk) is None

    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 9
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert torch.equal(spoken_utterance[2], third_chunk)
    assert torch.equal(spoken_utterance[3], fourth_chunk)
    assert all(torch.equal(chunk, silence_chunk) for chunk in spoken_utterance[4:])


def test_speech_buffer_keeps_prefix_out_of_active_speech_buffer() -> None:
    model = _FakeVADModel([0.1, 0.1, 0.9])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=32,
    )

    older_chunk = torch.ones(512)
    latest_pre_speech_chunk = torch.ones(512) * 2
    triggering_chunk = torch.ones(512) * 3

    assert iterator(older_chunk) is None
    assert iterator(latest_pre_speech_chunk) is None
    assert iterator(triggering_chunk) is None

    assert len(iterator.buffer) == 1
    assert torch.equal(iterator.buffer[0], triggering_chunk)

    speech_buffer = iterator.speech_buffer()
    assert len(speech_buffer) == 2
    assert torch.equal(speech_buffer[0], latest_pre_speech_chunk)
    assert torch.equal(speech_buffer[1], triggering_chunk)


def test_final_samples_are_kept_until_vad_declares_done() -> None:
    model = _FakeVADModel([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=64,
    )

    first_chunk = torch.ones(512)
    second_chunk = torch.ones(512) * 2
    trailing_chunks = [torch.ones(512) * value for value in (10, 11, 12, 13, 14)]

    assert iterator(first_chunk) is None
    assert iterator(second_chunk) is None

    spoken_utterance = None
    for chunk in trailing_chunks:
        spoken_utterance = iterator(chunk)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 7
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], second_chunk)
    assert torch.equal(spoken_utterance[2], trailing_chunks[0])
    assert torch.equal(spoken_utterance[3], trailing_chunks[1])
    assert torch.equal(spoken_utterance[4], trailing_chunks[2])
    assert torch.equal(spoken_utterance[5], trailing_chunks[3])
    assert torch.equal(spoken_utterance[6], trailing_chunks[4])


def test_brief_silence_is_preserved_when_speech_resumes() -> None:
    model = _FakeVADModel([0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    first_chunk = torch.ones(512)
    pause_chunks = [torch.ones(512) * value for value in (8, 9)]
    resumed_chunk = torch.ones(512) * 2
    ending_silence = torch.zeros(512)

    assert iterator(first_chunk) is None
    assert iterator(pause_chunks[0]) is None
    assert iterator(pause_chunks[1]) is None
    assert iterator(resumed_chunk) is None

    spoken_utterance = _finish_utterance(iterator, ending_silence)

    assert spoken_utterance is not None
    assert len(spoken_utterance) == 9
    assert torch.equal(spoken_utterance[0], first_chunk)
    assert torch.equal(spoken_utterance[1], pause_chunks[0])
    assert torch.equal(spoken_utterance[2], pause_chunks[1])
    assert torch.equal(spoken_utterance[3], resumed_chunk)
    assert all(torch.equal(chunk, ending_silence) for chunk in spoken_utterance[4:])


def test_active_speech_samples_include_hysteresis_band_and_exclude_trailing_silence() -> None:
    model = _FakeVADModel([0.1, 0.9, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
    iterator = VADIterator(
        model=model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=512,
    )

    pre_speech_chunk = torch.ones(512)
    speech_chunk = torch.ones(512) * 2
    maintained_speech_chunk = torch.ones(512) * 3
    silence_chunk = torch.zeros(512)

    assert iterator(pre_speech_chunk) is None
    assert iterator(speech_chunk) is None
    assert iterator(maintained_speech_chunk) is None
    assert iterator.active_speech_samples == 1024

    spoken_utterance = _finish_utterance(iterator, silence_chunk)

    assert spoken_utterance is not None
    assert iterator.last_utterance_active_speech_samples == 1024
    assert iterator.active_speech_samples == 0
    assert len(spoken_utterance) > 2


def test_flush_matches_detection_when_it_commits_at_the_same_moment() -> None:
    """Committing where detection would have fired must yield the same turn.

    The committed iterator has the silence bar pushed out of reach (what
    ``--manual_turn_end`` does), so it is still buffering after the seventh chunk
    that ends the detected one. Both have therefore seen exactly the same audio,
    and the commit must not lose or reshape any of it.
    """
    probs = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    detected = VADIterator(
        model=_FakeVADModel(probs),
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    committed = VADIterator(
        model=_FakeVADModel(probs),
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )
    committed.min_silence_samples = 16000 * 3600

    chunks = [torch.full((512,), float(i + 1)) for i in range(len(probs))]

    by_detection = None
    for chunk in chunks:
        by_detection = detected(chunk)
    assert by_detection is not None, "the detected iterator should have closed on the last chunk"

    for chunk in chunks:
        assert committed(chunk) is None
    by_commit = committed.flush()

    assert by_commit is not None
    assert torch.equal(torch.cat(by_commit), torch.cat(by_detection))
    assert committed.last_utterance_active_speech_samples == detected.last_utterance_active_speech_samples


def test_flush_is_a_no_op_when_no_utterance_is_open() -> None:
    """A commit during silence, or a second commit, must not invent a turn."""
    iterator = VADIterator(
        model=_FakeVADModel([0.1, 0.9, 0.9]),
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
    )

    assert iterator(torch.zeros(512)) is None
    assert iterator.flush() is None  # nothing was said

    assert iterator(torch.ones(512)) is None
    assert iterator(torch.ones(512)) is None
    assert iterator.flush() is not None
    assert iterator.flush() is None  # already committed


def test_an_out_of_reach_silence_bar_leaves_the_utterance_for_flush() -> None:
    """How ``--manual_turn_end`` works: silence never endpoints, the commit does.

    The pause in the middle is exactly what ``min_silence_ms 64`` cut turns on,
    so it must survive inside the committed utterance rather than split it.
    """
    iterator = VADIterator(
        model=_FakeVADModel([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1]),
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=64,
        speech_pad_ms=0,
    )
    iterator.min_silence_samples = 16000 * 3600

    chunks = [torch.full((512,), float(i + 1)) for i in range(9)]
    for chunk in chunks:
        assert iterator(chunk) is None, "silence must never close the turn in manual mode"

    spoken_utterance = iterator.flush()

    assert spoken_utterance is not None
    assert len(spoken_utterance) == len(chunks)
    assert torch.equal(torch.cat(spoken_utterance), torch.cat(chunks))
