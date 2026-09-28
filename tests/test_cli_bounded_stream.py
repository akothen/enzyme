"""``_BoundedStream`` reports truncation only when the source outran the limit."""

from __future__ import annotations

from typing import Any

from axon.cli import _BoundedStream


class _Source:
    """A counted iterator carrying the ``outcome`` attribute the CLI forwards."""

    def __init__(self, count: int) -> None:
        self._items = iter(range(count))
        self.outcome = "live"
        self.pulls = 0

    def __iter__(self) -> _Source:
        return self

    def __next__(self) -> Any:
        self.pulls += 1
        return next(self._items)


def test_a_source_under_the_limit_is_not_truncated() -> None:
    source = _Source(2)
    stream = _BoundedStream(source, 4)
    assert list(stream) == [0, 1]
    assert not stream.truncated_by_limit
    assert stream.outcome == "live"


def test_a_source_of_exactly_the_limit_is_not_truncated() -> None:
    source = _Source(3)
    stream = _BoundedStream(source, 3)
    assert list(stream) == [0, 1, 2]
    assert not stream.truncated_by_limit


def test_a_source_past_the_limit_is_truncated() -> None:
    source = _Source(5)
    stream = _BoundedStream(source, 3)
    assert list(stream) == [0, 1, 2]
    assert stream.truncated_by_limit
    # The surplus graph is pulled to prove truncation but never yielded.
    assert source.pulls == 4


def test_a_truncated_stream_stays_stopped() -> None:
    stream = _BoundedStream(_Source(5), 1)
    assert list(stream) == [0]
    assert list(stream) == []
    assert stream.truncated_by_limit
