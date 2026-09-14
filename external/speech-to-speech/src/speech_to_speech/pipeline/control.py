from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlKind(str, Enum):
    """Strongly-typed kinds for :class:`PipelineControlMessage`."""

    SESSION_END = "session_end"
    END_OF_TURN = "end_of_turn"


@dataclass(frozen=True)
class PipelineControlMessage:
    kind: ControlKind


SESSION_END = PipelineControlMessage(ControlKind.SESSION_END)
# The client declaring "the user has stopped speaking" — a fact the silence
# heuristic can only guess at. Handled by whichever handler reads the client's
# queue (the VAD), which closes the turn it has buffered and passes it on.
END_OF_TURN = PipelineControlMessage(ControlKind.END_OF_TURN)


def is_control_message(message: object, kind: ControlKind | None = None) -> bool:
    return isinstance(message, PipelineControlMessage) and (kind is None or message.kind == kind)
