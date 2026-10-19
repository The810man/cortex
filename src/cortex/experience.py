"""Its experience: every reply its model gave, with what it was shown, kept to learn from in sleep.

A memory holds what came of a thought; to teach the model itself, what it was shown and what it
answered are needed too. So each region's model call is held until the memory it led to is
written, and kept beside it, one line a call, a file a day. In sleep each reply is labelled by
what that memory turned out to be worth (the critic's view of how it went at it and what was
measured, half and half, credit flowing back from what it led to), and the model learns from
the good and the bad ones (`train.py`).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

EXPERIENCE_DIR = "experience"
HELD = 180.0  # seconds a reply waits for the memory it leads to; after that it led to none
GOOD, BAD = 0.15, -0.15  # what a memory must have come to for its reply to be taught as good, or as bad


class Experience:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.held: dict[str, tuple[float, str, str, str]] = {}  # region: when, system, prompt, reply

    def hold(self, region: str, system: str, prompt: str, reply: BaseModel) -> None:
        self.held[region] = (time.time(), system, prompt, reply.model_dump_json())

    def keep(self, region: str, memory_id: int) -> None:
        """The memory a region's last reply led to is written: the two are kept together."""
        held = self.held.pop(region, None)
        if held is None or time.time() - held[0] > HELD:
            return
        self.folder.mkdir(exist_ok=True)
        record = {"id": memory_id, "region": region, "system": held[1], "prompt": held[2], "reply": held[3]}
        with (self.folder / f"{datetime.now():%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as day:
            day.write(json.dumps(record, ensure_ascii=False) + "\n")

    def lessons(self, value_of: Any, since: int = 0) -> list[dict]:
        """The replies since a memory id, each labelled by what its memory came to: good above
        GOOD, bad below BAD, and the ones in between left out, since they teach nothing either way.
        `value_of(memory_id)` gives a memory's value, or None for one that was forgotten."""
        found = []
        for day in sorted(self.folder.glob("*.jsonl")) if self.folder.is_dir() else []:
            for line in day.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                value = value_of(record["id"]) if record["id"] > since else None
                if value is None or BAD <= value <= GOOD:
                    continue
                found.append({
                    "id": record["id"], "region": record["region"], "value": value, "label": value > GOOD,
                    "prompt": [{"role": "system", "content": record["system"]}, {"role": "user", "content": record["prompt"]}],
                    "completion": [{"role": "assistant", "content": record["reply"]}],
                })
        return found
