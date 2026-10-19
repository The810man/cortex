"""Signals and the bus that carries them between regions."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Signal:
    kind: str
    source: str
    content: str
    id: int = 0  # memory row id; 0 for signals that are not remembered
    parent: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    vec: Any = None  # the content as its embedding
    bits: Any = None  # the content as a pattern of neurons: what the regions read without the model


class Bus:
    def __init__(self) -> None:
        self._subs: list[tuple[frozenset[str] | None, asyncio.Queue[Signal]]] = []

    def subscribe(self, *kinds: str, maxsize: int = 512) -> asyncio.Queue[Signal]:
        """Queue of signals of the given kinds (all kinds when none are given)."""
        queue: asyncio.Queue[Signal] = asyncio.Queue(maxsize)
        self._subs.append((frozenset(kinds) or None, queue))
        return queue

    def publish(self, signal: Signal) -> None:
        for kinds, queue in self._subs:
            if kinds is not None and signal.kind not in kinds:
                continue
            if queue.full():  # a slow listener loses its oldest signal, never blocks the brain
                queue.get_nowait()
            queue.put_nowait(signal)
