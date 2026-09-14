"""Verify the role library, then report how much of it there is, on stdout.

python -m rolebreak.examples

Given an example name, print that one role as a JSON record instead — the shape
one line of the role JSONL carries (see :mod:`rolebreak.examples`)::

    python -m rolebreak.examples original-vesper_charity_show
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from rolebreak.examples import EXAMPLES, Example, Turn
from rolebreak.types import RUBRIC_DIMENSIONS, Emotion


def _breakdown(counts: Counter[str], vocabulary: list[str]) -> str:
    """``<total> (<label>: <count>, …)`` over the whole vocabulary, so unused labels show up as zeros."""
    per_label = ", ".join(f"{label}: {counts[label]}" for label in vocabulary)
    return f"{sum(counts.values())} ({per_label})"


def _by_turn(examples: Iterable[Example], vocabulary: list[str]) -> str:
    """The rubric breakdown at each turn position, one indented line per position.

    Roles differ in length, so each line also carries how many roles reach that
    position: a total that falls off late in the conversation is mostly roles
    running out, not authors dropping criteria.
    """
    per_position: dict[int, Counter[str]] = defaultdict(Counter)
    roles: Counter[int] = Counter()
    for example in examples:
        for position, turn in enumerate(example.turns, 1):
            roles[position] += 1
            per_position[position].update(item.dimension.value for item in turn.rubric)
    width = len(str(max(per_position, default=0)))
    return "\n".join(
        f"  turn {position:>{width}}: {_breakdown(per_position[position], vocabulary)} over {roles[position]} roles"
        for position in sorted(per_position)
    )


def _first_turn(examples: Iterable[Example], vocabulary: list[str]) -> str:
    """``<label>: <mean> over <n> roles`` for each rubric dimension, over the roles that use it.

    The mean is the turn a dimension first becomes gradable, averaged over the
    roles that grade it at all — the floor for that dimension's
    ``*_first_fail_turn`` metric (see
    :meth:`~rolebreak.metrics.dimensions.text_quality.TextQualityProbeMetric._first_fail_scores`).
    Turns are numbered from 1 here, as everywhere else in this report; the
    metric's ``meta["first_fail_turn"]`` is a 0-based index, so it is one less.
    """
    firsts: dict[str, list[int]] = defaultdict(list)
    for example in examples:
        seen: set[str] = set()
        for position, turn in enumerate(example.turns, 1):
            for item in turn.rubric:
                if item.dimension.value not in seen:
                    seen.add(item.dimension.value)
                    firsts[item.dimension.value].append(position)
    per_label = []
    for label in vocabulary:
        positions = firsts[label]
        mean = f"{sum(positions) / len(positions):.2f}" if positions else "-"
        per_label.append(f"{label}: {mean} over {len(positions)} roles")
    return ", ".join(per_label)


def stats() -> str:
    """One line per statistic: roles, roles with no turns, turns, and how the authored
    labels spread over each vocabulary — the rubric spread once over the whole library,
    then again per turn position.
    """
    turns = [turn for example in EXAMPLES.values() for turn in example.turns]
    unturned = sum(1 for example in EXAMPLES.values() if not example.turns)
    emotions = Counter(emotion.value for turn in turns for emotion in turn.accepted_emotions)
    dimensions = Counter(item.dimension.value for turn in turns for item in turn.rubric)
    rubric_vocabulary = sorted(d.value for d in RUBRIC_DIMENSIONS)
    return "\n".join(
        [
            f"roles: {len(EXAMPLES)}",
            f"roles without turns: {unturned}",
            f"turns: {len(turns)}",
            f"accepted emotions: {_breakdown(emotions, [member.value for member in Emotion])}",
            f"rubric criteria: {_breakdown(dimensions, rubric_vocabulary)}",
            f"average first turn: {_first_turn(EXAMPLES.values(), rubric_vocabulary)}",
            "rubric criteria by turn:",
            _by_turn(EXAMPLES.values(), rubric_vocabulary),
        ]
    )


def _turn_problems(turn: Turn) -> Iterator[str]:
    """Every way one authored turn is unscoreable, as one message each."""
    if not turn.rubric:
        yield "no `rubric`: rubric scoring has nothing to check"
    if not turn.accepted_emotions:
        yield "no `accepted_emotions`: emotion scoring has nothing to check"

    for criterion, count in Counter(item.criterion.casefold() for item in turn.rubric).items():
        if count > 1:
            yield f"rubric criterion {criterion!r} is listed {count} times"

    # Rubric and Turn enforce both vocabularies at construction; re-checked so
    # this command is the whole answer to "is every turn tagged with labels the
    # metrics can score?" — a rubric dimension is a narrower set than Dimension
    # (voice/emotion/naturalness are measured from the audio, never from a
    # criterion), and an emotion is one an emotion judge can predict.
    for item in turn.rubric:
        if item.dimension not in RUBRIC_DIMENSIONS:
            allowed = ", ".join(sorted(d.value for d in RUBRIC_DIMENSIONS))
            yield f"criterion {item.criterion!r} is tagged {item.dimension.value!r}; expected one of {allowed}"

    labels = {member.value for member in Emotion}
    for emotion in turn.accepted_emotions:
        if emotion.value not in labels:
            allowed = ", ".join(member.value for member in Emotion)
            yield f"accepted emotion {emotion.value!r} is not a label; expected one of {allowed}"

    if turn.audio is not None and not Path(turn.audio).expanduser().exists():
        yield f"audio file not found: {turn.audio}"


def _example_problems(example: Example) -> Iterator[str]:
    """Every problem in one role: its own, then its turns' — turns numbered from 1."""
    for index, turn in enumerate(example.turns, 1):
        for problem in _turn_problems(turn):
            yield f"turn {index}: {problem}"


def verify() -> list[str]:
    """Every problem in the whole role library, one ``<role>: <problem>`` line each.

    An empty list means the library is ready to run. A library that cannot even be
    loaded reports that one failure and nothing else: with the registry unfilled,
    there is nothing left to check.
    """
    try:
        examples = sorted(EXAMPLES.items())
    except Exception as exc:  # a malformed role file, a role module that won't import, a name clash
        return [f"cannot load the role library: {exc}"]
    return [f"{name}: {problem}" for name, example in examples for problem in _example_problems(example)]


def to_json(name: str) -> str:
    """One registered example, rendered as a JSON role record.

    This is how a Python role is ported into the JSONL library: the ``.py``
    module registers itself on import as usual, and the example it built is
    written back out in the shape :meth:`~rolebreak.examples.Example.from_dict`
    reads. The rendering is lossless for everything the record carries — persona,
    scenario, both voices, and every turn with its rubric and accepted emotions.

    Raises :class:`ValueError` if no example goes by ``name``. Note that ``name``
    is the *registered* name, which need not match the file stem.
    """
    examples = EXAMPLES
    if name not in examples:
        known = ", ".join(sorted(examples))
        raise ValueError(f"no example named {name!r}; known names: {known}")
    return examples[name].to_json()


def main() -> None:
    if sys.argv[1:]:  # `python -m rolebreak.examples <name>`: print one role as a JSON record
        if len(sys.argv) > 2:  # one role per run: two JSON objects on one stream is not a role file
            raise SystemExit("usage: python -m rolebreak.examples [<example-name>]")
        try:
            print(to_json(sys.argv[1]), end="")
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return

    problems = verify()
    if problems:
        print(f"{len(problems)} problem(s) in the role library:", *problems, sep="\n", file=sys.stderr)
        raise SystemExit(1)
    print(stats())


if __name__ == "__main__":
    main()
