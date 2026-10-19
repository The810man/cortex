"""The regions' own language, and the tissue it travels in: neurons, where they lie, and what joins them.

Words are for the model, and the model is slow. Between regions a thought also travels as a
pattern: which of 512 neurons fire for it, 64 bytes. Every region has its own 512 neurons, lying
on a sheet at that region's place in the head, and the regions are joined by tracts: bundles of
weighted synapses from the neurons of one to the neurons of another. A tract has a length, the
distance between the two regions, and a signal takes that long to arrive; it has a strength, how
much of the signal arrives at all, which anatomy sets and reward moves; and each of its synapses
has a weight, which says how it has gone when this neuron here fired right after that one there.

Everything in this file works on those patterns and never asks the model, so it runs on every
heartbeat, between thoughts and in sleep:

- how alike two thoughts are: the neurons they share;
- what a thought like this has been worth: a readout that every reward trains (`feel`);
- what tends to follow it in the region it is going to: the synapses of the tract it travels,
  which reward strengthens and punishment weakens (`bind`, `reinforce`, `follow`, `lean`);
- a thought on its way: fired in one region it arrives in the others a little later, the later
  the farther, fainter where the tract is weak, and with it what that region has learned to
  expect after it (`send`, `beat`);
- what the mind has dwelt on too long: neurons tire, and a worn theme stops pulling (`worn`);
- the undercurrent: activity that keeps moving when no region is writing, so that something is
  always on the mind's mind (`beat`, `drift`, `visit`).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

NEURONS = 512
ACTIVE = NEURONS // 5  # how many fire for one thought
ROWS, COLUMNS = 8, NEURONS // 8  # the sheet a region's neurons lie on
PITCH = 0.5  # millimetres between two neighbours on it
UNJUDGED = 0.1  # what one thought following another is worth to their synapses before anyone has judged it
PLASTICITY = 0.1  # how far one reward moves the synapses it reaches
INTUITION = 0.2  # how far one reward moves the readout
TRACES = 2048  # thoughts whose synapses a late reward can still find
FADING = 0.7  # how much of a neuron's firing is left a heartbeat later
LATERAL = 0.5  # how much of its neighbours' firing on the sheet a neuron picks up
SPEED = 45.0  # millimetres a signal travels along a tract in one heartbeat
REACH = 60.0  # the length over which a tract nobody laid down loses most of what it carries
THIN, THICK = 0.05, 1.0  # how weak and how strong a tract can become
FORESIGHT = 0.3  # how far one reward moves what a tract foresees for the neurons a step set out from
STRENGTHEN = 0.02  # how far one reward moves the strength of the tract that carried the step
EXPECTED = 0.6  # how brightly a region lights what it has learned to expect, beside what arrives
FAINT = 0.2  # a signal fainter than this is not passed on
Bits = np.ndarray  # NEURONS booleans
Path2 = tuple[str, str]  # a tract by its ends: from which region, to which

# Where each region lies, in millimetres: right of the midline, forward of the ears, above them.
PLACES: dict[str, tuple[float, float, float]] = {
    "pfc": (-40.0, 42.0, 30.0),
    "motor": (-38.0, -22.0, 58.0),
    "cerebellum": (-20.0, -62.0, -30.0),
    "dmn": (0.0, -52.0, 28.0),
    "thalamus": (-10.0, -18.0, 8.0),
    "hippocampus": (-28.0, -22.0, -14.0),
    "basal_ganglia": (-14.0, 8.0, 4.0),
    "amygdala": (-24.0, -4.0, -18.0),
    "brainstem": (0.0, -28.0, -30.0),
}
# The tracts anatomy lays down, and how much of a signal each carries before anything is learned.
# Any other two regions grow one the first time a thought of one leads to a thought of the other:
# thin at first, and the farther apart the thinner.
INNATE: dict[Path2, float] = {
    ("thalamus", "pfc"): 0.7, ("pfc", "thalamus"): 0.5, ("thalamus", "motor"): 0.5,
    ("thalamus", "dmn"): 0.5, ("thalamus", "hippocampus"): 0.5, ("thalamus", "amygdala"): 0.5,
    ("pfc", "motor"): 0.7, ("motor", "pfc"): 0.6, ("pfc", "dmn"): 0.5, ("dmn", "pfc"): 0.6,
    ("pfc", "hippocampus"): 0.5, ("hippocampus", "pfc"): 0.6, ("pfc", "basal_ganglia"): 0.7,
    ("motor", "basal_ganglia"): 0.6, ("dmn", "basal_ganglia"): 0.5, ("basal_ganglia", "thalamus"): 0.7,
    ("basal_ganglia", "pfc"): 0.5, ("hippocampus", "dmn"): 0.7, ("dmn", "hippocampus"): 0.6,
    ("hippocampus", "amygdala"): 0.5, ("amygdala", "hippocampus"): 0.6, ("amygdala", "pfc"): 0.5,
    ("motor", "cerebellum"): 0.6, ("cerebellum", "motor"): 0.5, ("cerebellum", "thalamus"): 0.4,
    ("pfc", "cerebellum"): 0.4, ("amygdala", "brainstem"): 0.5, ("brainstem", "thalamus"): 0.6,
}


def encode(vec: np.ndarray | None) -> Bits | None:
    """A thought's pattern: the fifth of the neurons its embedding points along most. Measured
    on 4,005 real memories, the neurons two thoughts share track their similarity at r = 0.94."""
    if vec is None:
        return None
    part = np.asarray(vec[:NEURONS], dtype=np.float32)
    bits = np.zeros(NEURONS, dtype=bool)
    bits[: len(part)] = part > max(float(np.quantile(part, 1 - ACTIVE / NEURONS)), 0.0)
    return bits


def pack(bits: Bits) -> np.ndarray:
    """The pattern as it travels and is stored: 64 bytes."""
    return np.packbits(bits)


def shared(packed: np.ndarray, bits: Bits) -> np.ndarray:
    """How much each packed pattern in a matrix has in common with `bits`, 0 to 1: the neurons
    both fire, out of those the sparser of the two fires."""
    both = np.bitwise_count(packed & pack(bits)).sum(axis=-1, dtype=np.int32)
    fewer = np.minimum(np.bitwise_count(packed).sum(axis=-1, dtype=np.int32), int(bits.sum()))
    return both / np.maximum(fewer, 1)


def strongest(drive: np.ndarray) -> Bits:
    bits = np.zeros(NEURONS, dtype=bool)
    bits[np.argpartition(drive, -ACTIVE)[-ACTIVE:]] = True
    return bits


def distance(one: str, other: str) -> float:
    """How far apart two regions lie, in millimetres."""
    return math.dist(PLACES[one], PLACES[other])


class Tract:
    """The fibres from one region to another: how long, how strong, and what each synapse has learned."""

    __slots__ = ("length", "delay", "innate", "strength", "usual", "weights", "foresight", "flight", "carried")

    def __init__(self, path: Path2) -> None:
        self.length = distance(*path)
        self.delay = max(1, round(self.length / SPEED))  # heartbeats a signal is under way
        laid = INNATE.get(path)
        # Within a region nothing has to travel; between two that anatomy did not join, a tract
        # grown by use starts thin, and thinner the farther it has to reach.
        self.innate = laid if laid is not None else 0.5 if path[0] == path[1] else 0.3 * math.exp(-self.length / REACH)
        self.strength = self.innate
        self.usual = 0.0  # what a step along it usually earns: only better or worse than that moves its strength
        self.foresight: np.ndarray | None = None  # [neuron here]: what a step that set out from it has been worth
        self.weights: np.ndarray | None = None  # [neuron there, neuron here]; none until a step has travelled it
        self.flight: list[tuple[int, Bits, float, int]] = []  # signals under way: beats left, pattern, level, hops left
        self.carried = 0  # how many thoughts have led along it to the next

    def learned(self) -> np.ndarray:
        if self.weights is None:
            self.weights = np.zeros((NEURONS, NEURONS), dtype=np.float32)
        return self.weights


class Net:
    """The neurons of every region and the tracts between them. One per brain, kept in its folder
    between lives."""

    def __init__(self, path: Path | None = None, seed: int | None = None) -> None:
        self.path = path
        self.fire: dict[str, np.ndarray] = {key: np.zeros(NEURONS, dtype=np.float32) for key in PLACES}
        self.tracts: dict[Path2, Tract] = {ends: Tract(ends) for ends in INNATE}
        self.order = np.arange(NEURONS)  # which neuron lies at each place on the sheet, column by column
        self.arrived: dict[str, float] = {}  # which regions a signal reached on the last heartbeat, and how strongly
        self.worth = np.zeros(NEURONS, dtype=np.float32)  # the readout: what firing here has been worth
        self.worn_out = np.zeros(NEURONS, dtype=np.float32)  # slow: themes the mind has dwelt on
        self.tired = np.zeros(NEURONS, dtype=np.float32)  # fast: what the undercurrent has just visited
        self.charge = np.zeros(NEURONS, dtype=np.float32)  # the undercurrent: what is still lit
        self.here: Bits | None = None  # where the undercurrent is: the last pattern it arrived at
        self.traces: OrderedDict[int, tuple[Path2, tuple[np.ndarray, np.ndarray]]] = OrderedDict()
        self.rng = np.random.default_rng(seed)
        self.born = True  # no tracts of its own on disk yet: the brain may replay its past into them
        self.before = False  # whether the file on disk is from before there were tracts
        if path is not None and path.is_file():
            with np.load(path) as kept:
                if kept["worth"].shape == self.worth.shape:
                    self.worth, self.worn_out, self.charge = kept["worth"], kept["worn_out"], kept["charge"]
                    self._grow(kept)

    def _grow(self, kept) -> None:
        """Take up the tracts a former life left. Synapses from before there were tracts were one
        sheet for the whole mind: those are let go, and the brain relives its past into tracts."""
        if "tracts" not in kept:
            self.before = True  # what that life left is set aside, not written over, when this one first saves
            return
        self.order = kept["order"]
        for name, strength, usual, carried in zip(
            kept["tracts"], kept["strengths"], kept["usual"], kept["carried"], strict=True
        ):
            src, _, dst = str(name).partition(">")
            if src in PLACES and dst in PLACES:
                tract = self.tract((src, dst))
                tract.strength, tract.usual, tract.carried = float(strength), float(usual), int(carried)
                if f"w:{name}" in kept:
                    tract.weights = kept[f"w:{name}"].astype(np.float32)
                if f"f:{name}" in kept:
                    tract.foresight = kept[f"f:{name}"].astype(np.float32)
        self.born = False

    def save(self) -> None:
        if self.path is None or self.born:  # opened but never woken: there is nothing of its own to keep
            return
        names = [f"{src}>{dst}" for src, dst in self.tracts]
        learned = {
            f"w:{name}": tract.weights.astype(np.float16)  # a weight is known to two decimals at best
            for name, tract in zip(names, self.tracts.values(), strict=True) if tract.weights is not None
        }
        learned.update({
            f"f:{name}": tract.foresight
            for name, tract in zip(names, self.tracts.values(), strict=True) if tract.foresight is not None
        })
        fresh = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez_compressed(
            fresh, worth=self.worth, worn_out=self.worn_out, charge=self.charge, order=self.order,
            tracts=np.array(names), strengths=np.array([t.strength for t in self.tracts.values()], dtype=np.float32),
            usual=np.array([t.usual for t in self.tracts.values()], dtype=np.float32),
            carried=np.array([t.carried for t in self.tracts.values()], dtype=np.int64), **learned,
        )
        if self.before and self.path.is_file():
            os.replace(self.path, self.path.with_name(self.path.name + ".before-tracts"))
            self.before = False
        os.replace(fresh, self.path)  # a life that ends mid-write keeps the tracts it had

    # -- where things lie -----------------------------------------------------------------

    def tract(self, path: Path2) -> Tract:
        """The tract from one region to another. Where there was none, one grows."""
        if path not in self.tracts:
            self.tracts[path] = Tract(path)
        return self.tracts[path]

    def arrange(self, codes: np.ndarray) -> tuple[float, float] | None:
        """Lay the neurons out on their sheet so that those which fire together lie together, from
        the patterns of the brain's own memories. Returns how far, in millimetres, a neuron then
        lies from the one it most often fires with, and how far from any other on average."""
        fired = np.unpackbits(codes, axis=1)[:, :NEURONS].astype(np.float32)
        if len(fired) < 64:
            return None
        together = np.nan_to_num(np.corrcoef(fired.T))
        _, axes = np.linalg.eigh(together)
        first, second = axes[:, -1], axes[:, -2]  # the two directions along which firing varies most
        columns = np.argsort(first).reshape(COLUMNS, ROWS)
        self.order = np.concatenate([column[np.argsort(second[column])] for column in columns])
        place = np.empty(NEURONS, dtype=np.int64)
        place[self.order] = np.arange(NEURONS)
        where = np.stack([place // ROWS, place % ROWS], axis=1).astype(np.float32) * PITCH
        np.fill_diagonal(together, -1.0)
        partner = together.argmax(axis=1)
        near = float(np.linalg.norm(where - where[partner], axis=1).mean())
        far = float(np.linalg.norm(where[:, None] - where[None], axis=2).mean())
        return near, far

    def sheet(self, region: str) -> np.ndarray:
        """A region's firing as it lies on its sheet, column by column."""
        return self.fire[region][self.order]

    # -- intuition: what a pattern has been worth -----------------------------------

    def feel(self, bits: Bits | None) -> float:
        """The gut feeling about a thought, -1 to 1, before anyone has read it."""
        if bits is None or not bits.any():
            return 0.0
        return float(np.clip(self.worth[bits].mean(), -1.0, 1.0))

    def value(self, bits: Bits | None, reward: float, rate: float = INTUITION) -> None:
        if bits is not None:
            self.worth[bits] += rate * (reward - self.feel(bits))

    # -- association: what follows what along a tract, and whether it paid ----------------

    def bind(self, memory_id: int, before: Bits | None, after: Bits | None, path: Path2 = ("dmn", "dmn")) -> None:
        """A thought in one region led to a thought in another: the neurons of the first are wired
        to those of the second along the tract between them, weakly. The reward the second one
        earns, when it comes, finds these same synapses."""
        if before is None or after is None:
            return
        pair = (np.flatnonzero(after), np.flatnonzero(before))
        tract = self.tract(path)
        tract.carried += 1
        self._wire(tract, pair, UNJUDGED, PLASTICITY / 2)
        self.traces[memory_id] = (path, pair)
        if len(self.traces) > TRACES:
            self.traces.popitem(last=False)

    def reinforce(self, memory_id: int, reward: float) -> None:
        """What a step turned out to be worth reaches the synapses that carried it, and the tract
        as a whole: one whose steps go better than they usually do carries more of the next
        signal, one whose steps go worse carries less. Measured on a brain whose steps mostly
        failed: moved by the reward itself, its busiest tracts thinned to nothing within a day."""
        trace = self.traces.get(memory_id)
        if trace is not None:
            tract = self.tract(trace[0])
            self._wire(tract, trace[1], reward, PLASTICITY)
            if tract.foresight is None:
                tract.foresight = np.zeros(NEURONS, dtype=np.float32)
            here = trace[1][1]  # the neurons it set out from learn what setting out from them came to
            tract.foresight[here] += FORESIGHT * (reward - tract.foresight[here].mean())
            surprise = reward - tract.usual
            tract.usual += 0.05 * surprise
            tract.strength = float(np.clip(tract.strength + STRENGTHEN * surprise, THIN, THICK))

    def foresee(self, bits: Bits | None, path: Path2) -> float | None:
        """What a step about to set out along a tract will be worth, by what the steps that set out
        from neurons like these came to; None where the tract has learned nothing yet. Replayed over
        this brain's history, asked before each step was taken: of the steps it foresaw at -0.25 or
        worse, and not right after a step that had worked, 89 in 100 went on to be punished (174
        steps; of all steps, 69 in 100 were)."""
        tract = self.tracts.get(path)
        if bits is None or tract is None or tract.foresight is None or not bits.any():
            return None
        return float(tract.foresight[bits].mean())

    def _wire(self, tract: Tract, pair: tuple[np.ndarray, np.ndarray], worth: float, rate: float) -> None:
        """Move a block of synapses toward what the step they carried turned out to be worth. A
        synapse therefore says how it has gone when this neuron fired right after that one."""
        cell = np.ix_(*pair)
        weights = tract.learned()
        weights[cell] += rate * (worth - weights[cell])

    def pull(self, bits: Bits, path: Path2 | None = None) -> np.ndarray:
        """For every neuron, how well it has gone when it fired right after this pattern: in the
        region one tract leads to, or anywhere, over all tracts, each as strong as it is."""
        fired = max(int(bits.sum()), 1)
        if path is not None:
            tract = self.tracts.get(path)
            if tract is None or tract.weights is None:
                return np.zeros(NEURONS, dtype=np.float32)
            return tract.weights[:, bits].sum(axis=1) / fired
        total = np.zeros(NEURONS, dtype=np.float32)
        for tract in self.tracts.values():
            if tract.weights is not None:
                total += tract.strength * tract.weights[:, bits].sum(axis=1)
        return np.clip(total / fired, -1.0, 1.0)

    def follow(self, bits: Bits, path: Path2 | None = None) -> Bits:
        """The pattern that has tended to come after this one, and to be rewarded."""
        return strongest(self.pull(bits, path))

    def lean(self, bits: Bits, path: Path2 | None = None) -> Bits:
        """This pattern, leaned toward what has paid after it: a moment like now that went well."""
        return strongest(bits + 2.0 * self.pull(bits, path))

    def expect(self, bits: Bits, path: Path2) -> Bits | None:
        """What the region at the end of a tract has learned to expect after this pattern: the
        neurons that fired there after the like of it, and paid. None where the tract has learned
        too little about such a pattern to expect anything."""
        tract = self.tracts.get(path)
        if tract is None or tract.weights is None:
            return None
        expected = self._expected(tract, bits)
        return expected if expected.sum() >= ACTIVE // 2 else None

    @staticmethod
    def _expected(tract: Tract, bits: Bits) -> Bits:
        """The neurons a tract's synapses drive for this pattern beyond what they would for any
        pattern. Two patterns that have nothing to do with each other still share a fifth of
        their neurons, and without this every plan would call up the one step that paid most."""
        pull = tract.weights[:, bits].sum(axis=1) / max(int(bits.sum()), 1) - tract.weights.mean(axis=1)
        return strongest(pull) & (pull > 0.02)

    # -- a thought on its way -----------------------------------------------------------

    def light(self, region: str, bits: Bits | None, level: float = 1.0) -> None:
        """A pattern fires in a region's own neurons."""
        if bits is not None:
            field = self.fire[region]
            field[bits] = np.maximum(field[bits], level)

    def send(self, region: str, bits: Bits | None, level: float = 1.0, hops: int = 1) -> None:
        """A region has produced a thought: its pattern sets out along every tract that leaves it."""
        if bits is None:
            return
        for (src, dst), tract in self.tracts.items():
            if src == region and dst != region:
                tract.flight.append((tract.delay, bits, level, hops))

    def _arrive(self, tract: Tract, dst: str, bits: Bits, level: float) -> None:
        """A signal reaches the end of its tract: the neurons it names fire there, as strongly as
        the tract carried it, and more faintly those the synapses say have followed and paid."""
        self.light(dst, bits, level)
        if tract.weights is not None:
            self.light(dst, self._expected(tract, bits), EXPECTED * level)
        self.arrived[dst] = max(self.arrived.get(dst, 0.0), level)

    def beat(self) -> Bits:
        """One heartbeat with no model in it. Firing spreads a little to the neighbours on each
        sheet and fades; signals under way move on, and those that arrive fire where they land;
        charge fades, tired neurons recover, and what is still lit is what the mind has on it."""
        self.arrived = {}
        for field in self.fire.values():
            if field.any():
                lying = field[self.order].reshape(COLUMNS, ROWS)
                near = np.zeros_like(lying)
                near[1:] += lying[:-1]
                near[:-1] += lying[1:]
                near[:, 1:] += lying[:, :-1]
                near[:, :-1] += lying[:, 1:]
                field[self.order] = np.maximum(lying, LATERAL * near / 4).ravel()
                field *= FADING
                field[field < 0.01] = 0.0
        onward: list[tuple[str, str, Bits, float, int]] = []
        for (src, dst), tract in self.tracts.items():
            if not tract.flight:
                continue
            under_way = []
            for left, bits, level, hops in tract.flight:
                if left > 1:
                    under_way.append((left - 1, bits, level, hops))
                    continue
                level *= tract.strength
                self._arrive(tract, dst, bits, level)
                if hops > 0 and level >= FAINT:
                    onward.append((src, dst, bits, level, hops - 1))
            tract.flight = under_way
        for came_from, region, bits, level, hops in onward:  # passed on, but never straight back
            for (src, dst), tract in self.tracts.items():
                if src == region and dst not in (region, came_from):
                    tract.flight.append((tract.delay, bits, level, hops))
        self.charge *= 0.93
        self.tired *= 0.97
        self.worn_out *= 0.9995  # a theme left alone is fresh again in a quarter of an hour
        return self.charge > 0.25

    # -- the undercurrent -----------------------------------------------------------

    def imprint(self, bits: Bits | None) -> None:
        """A thought passes: it charges its neurons, and the undercurrent is now there."""
        if bits is not None:
            self.charge[bits] = 1.0
            self.here = bits

    def drift(self, toward: Bits | None = None) -> Bits:
        """Where the undercurrent goes next from where it is: toward what has followed this and
        paid, held near what was thought lately and what the regions are lit with right now, away
        from what it has just visited, and always a little `toward` what the mind is for."""
        lit = sum(self.fire.values()) / len(self.fire)
        drive = self.charge + 0.3 * lit - self.tired + 0.05 * self.rng.random(NEURONS, dtype=np.float32)
        if toward is not None:
            drive += 0.6 * toward
        if self.here is None:
            return strongest(drive)
        return strongest(self.here + 2.0 * self.pull(self.here) + drive - 0.5 * self.charge)

    def visit(self, bits: Bits) -> None:
        """The undercurrent arrived at a memory: it is there now, and those neurons tire so that
        the next step leads somewhere else."""
        self.charge[bits] = np.maximum(self.charge[bits], 0.7)
        self.tired[bits] = np.minimum(self.tired[bits] + 0.7, 1.0)
        self.here = bits

    # -- wear: boredom, in neurons ------------------------------------------------------

    def worn(self, bits: Bits | None) -> float:
        """How much of this theme the mind has had lately, 0 to 1."""
        if bits is None or not bits.any():
            return 0.0
        return float(self.worn_out[bits].mean())

    def wear(self, bits: Bits | None, by: float = 0.1) -> None:
        if bits is not None:
            self.worn_out = (1 - by) * self.worn_out + by * bits

    # -- sleep --------------------------------------------------------------------------

    def replay(
        self, episodes: list[tuple[int, Bits | None, Bits, float, Path2]], passes: int = 8
    ) -> tuple[float, float]:
        """Go through the day again, many times, with what each thought turned out to be worth.

        Awake, the readout hears only the first verdict on a thought. By night its value has
        settled: credit has come back to it from what it led to. Replay teaches the readout
        that settled value and wires in the steps that paid, each along the tract it travelled.
        Returns how often the gut feeling had the sign of the settled value before the replay,
        and how often after, 0 to 1.
        """
        judged = [(after, worth) for _, _, after, worth, _ in episodes if abs(worth) > 0.05]

        def agreement() -> float:
            return sum((self.feel(after) > 0) == (worth > 0) for after, worth in judged) / max(len(judged), 1)

        before = agreement()
        for _ in range(passes):
            for index in self.rng.permutation(len(episodes)):
                _, earlier, after, worth, path = episodes[index]
                self.value(after, worth, INTUITION / 2)
                if earlier is not None and abs(worth) > 0.05:
                    pair = (np.flatnonzero(after), np.flatnonzero(earlier))
                    self._wire(self.tract(path), pair, worth, PLASTICITY / passes)
        return before, agreement()

    def foretell(self, episodes: list[tuple[int, Bits | None, Bits, float, Path2]]) -> None:
        """Learn what steps came to, once and in the order they happened, as `reinforce` does living."""
        for _, earlier, _, worth, path in episodes:
            if earlier is not None and abs(worth) > 0.05 and path[0] != path[1]:
                tract = self.tract(path)
                if tract.foresight is None:
                    tract.foresight = np.zeros(NEURONS, dtype=np.float32)
                here = np.flatnonzero(earlier)
                tract.foresight[here] += FORESIGHT * (worth - tract.foresight[here].mean())

    def rest(self) -> None:
        """What sleep does to the tissue itself: every synapse shrinks a little, so only what is
        used again stays strong; a tract drifts back toward the strength it was laid down with;
        and the worn themes recover: tomorrow they are fresh."""
        for tract in self.tracts.values():
            if tract.weights is not None:
                tract.weights *= 0.98
            tract.strength += 0.05 * (tract.innate - tract.strength)
        self.worn_out *= 0.5
        self.tired[:] = 0.0
