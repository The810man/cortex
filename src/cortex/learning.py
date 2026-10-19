"""Reward learning.

A reward does three things: it sets the value of the thought it was given for, it strengthens or
weakens the heuristics that were active when that thought was produced, and it travels back along
the chain of parent thoughts with a discount, so the idea behind a good outcome gets credit too.
Heuristics learn from the reward prediction error (reward minus what the region usually earns),
which is what makes a merely average result neutral and a better-than-expected one reinforcing.

Which heuristics a region is handed is decided by their patterns, not by weight alone. A heuristic
that is about a theme the mind once dwelt on speaks up only when the mind is on that theme again,
and of several that say the same thing only the strongest is heard.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from .config import LearningSettings
from .memory import Memory
from .neural import Bits

# Measured on a brain with 28 heuristics and 603 goals and thoughts: a rule about one theme shared
# 60% of its pattern with hundreds of them, a general rule ("Run code before claiming it works")
# with none; two wordings of one rule shared 72% to 95% of theirs, two different rules about 40%.
THEMED = 0.6  # a goal or thought sharing this much of a heuristic's pattern is the theme it is about
AT_HOME = 3  # and this many of them make the heuristic one about a theme, not a general rule
ON_THEME = 0.5  # how much of it the matter in hand must share for the mind to be on that theme now
SAME = 0.7  # two heuristics sharing this much say the same thing


def overlap(a: Bits, b: Bits) -> float:
    """How much of the sparser of two patterns the other one shares, 0 to 1."""
    return float((a & b).sum()) / max(min(int(a.sum()), int(b.sum())), 1)


@dataclass(slots=True)
class Heuristic:
    id: int
    region: str
    text: str
    weight: float
    uses: int
    reward_sum: float
    active: bool
    origin: str

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.uses if self.uses else 0.0


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class Learner:
    def __init__(self, memory: Memory, cfg: LearningSettings, rng: random.Random | None = None) -> None:
        self.memory = memory
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.baseline: dict[str, float] = memory.get_state("baseline", {})
        self.dopamine: float = memory.get_state("dopamine", 0.0)
        self.patterns: dict[int, Bits] = {}  # each heuristic as a pattern of neurons; see `Brain.attune`
        self.themed: set[int] = set()  # those that are about a theme the mind once dwelt on

    def persist(self) -> None:
        self.memory.set_state("baseline", self.baseline)
        self.memory.set_state("dopamine", self.dopamine)

    # -- heuristics -----------------------------------------------------------------

    def heuristics(self, region: str | None = None, active_only: bool = True) -> list[Heuristic]:
        sql, args = "SELECT * FROM heuristic WHERE 1=1", []
        if region:
            sql, args = sql + " AND region=?", [region]
        if active_only:
            sql += " AND active=1"
        rows = self.memory.db.execute(sql + " ORDER BY weight DESC, id", args).fetchall()
        return [
            Heuristic(r["id"], r["region"], r["text"], r["weight"], r["uses"], r["reward_sum"],
                      bool(r["active"]), r["origin"])
            for r in rows
        ]

    def select(self, region: str, near: Bits | None = None) -> list[Heuristic]:
        """The strongest heuristics that fit what the mind is on (`near`), with an occasional
        weaker one swapped in so it gets a trial. No model: patterns against patterns."""
        pool: list[Heuristic] = []
        for heuristic in self.heuristics(region):
            bits = self.patterns.get(heuristic.id)
            if bits is not None:
                if heuristic.id in self.themed and (near is None or overlap(bits, near) < ON_THEME):
                    continue  # it is about a theme the mind is not on now
                if any(overlap(bits, self.patterns[h.id]) >= SAME for h in pool if h.id in self.patterns):
                    continue  # a stronger one already says this
            pool.append(heuristic)
        k = self.cfg.heuristics_per_prompt
        chosen, rest = pool[:k], pool[k:]
        if chosen and rest and self.rng.random() < self.cfg.explore:
            chosen[-1] = self.rng.choice(rest)
        return chosen

    def add(self, region: str, text: str, origin: str = "cerebellum") -> Heuristic | None:
        """Add a heuristic unless the region already knows (or has already rejected) one like it."""
        text = " ".join(text.split())
        if not text:
            return None
        for known in self.heuristics(region, active_only=False):
            if SequenceMatcher(None, known.text.lower(), text.lower()).ratio() > 0.85:
                return None
        cur = self.memory.db.execute(
            "INSERT INTO heuristic (region, text, origin, created) VALUES (?,?,?,?)",
            (region, text, origin, time.time()),
        )
        crowd = self.heuristics(region)
        for weakest in crowd[self.cfg.max_heuristics :]:
            if weakest.id != cur.lastrowid:
                self._suppress(weakest.id)
        return Heuristic(cur.lastrowid, region, text, 0.5, 0, 0.0, True, origin)

    def prune(self) -> int:
        """Suppress heuristics that had a fair trial and kept losing. They stay on record."""
        cur = self.memory.db.execute(
            "UPDATE heuristic SET active=0 WHERE active=1 AND uses>=? AND weight<?",
            (self.cfg.prune_after_uses, self.cfg.prune_below),
        )
        return cur.rowcount

    def _suppress(self, heuristic_id: int) -> None:
        self.memory.db.execute("UPDATE heuristic SET active=0 WHERE id=?", (heuristic_id,))

    def _credit(self, heuristic_ids: list[int], delta: float, reward: float) -> None:
        if not heuristic_ids:
            return
        marks = ",".join("?" * len(heuristic_ids))
        self.memory.db.execute(
            f"UPDATE heuristic SET weight=MAX(0, MIN(1, weight+?)), uses=uses+1, "
            f"reward_sum=reward_sum+? WHERE id IN ({marks})",
            (delta, reward, *heuristic_ids),
        )

    # -- reward ---------------------------------------------------------------------

    def reward(
        self, memory_id: int, reward: float, source: str, reason: str = "", lesson: str | None = None
    ) -> float | None:
        """Apply a reward in [-1, 1] to a remembered thought. Returns the prediction error."""
        row = self.memory.get(memory_id)
        if row is None:
            return None
        reward = _clip(reward)
        region = row["region"]
        expected = self.baseline.get(region, 0.0)
        rpe = reward - expected
        self.baseline[region] = expected + 0.1 * rpe
        self.dopamine = _clip(0.9 * self.dopamine + 0.1 * rpe)

        step = self.cfg.learning_rate * (self.cfg.human_weight if source == "human" else 1.0)
        # A second opinion on the same thought moves its value half as far as the first. The human's
        # is never a second opinion: it counts in full, and more, over whatever the critic said.
        delta = self.cfg.human_weight * reward if source == "human" else reward if row["rewards"] == 0 else 0.5 * reward
        self.memory.adjust_value(memory_id, delta, lesson)
        self._credit(json.loads(row["heuristics"]), step * rpe, reward)

        trace, parent = 1.0, row["parent"]
        for _ in range(self.cfg.trace_depth):
            ancestor = self.memory.get(parent) if parent else None
            if ancestor is None:
                break
            trace *= self.cfg.discount
            self.memory.adjust_value(ancestor["id"], 0.5 * trace * reward)
            self._credit(json.loads(ancestor["heuristics"]), step * trace * rpe, trace * reward)
            parent = ancestor["parent"]

        self.memory.log_reward(memory_id, region, reward, rpe, source, reason)
        return rpe

    def temperature(self, base: float) -> float:
        """Doing worse than expected widens the search; doing better narrows it."""
        return max(0.05, min(1.5, base * (1.0 - 0.4 * self.dopamine)))
