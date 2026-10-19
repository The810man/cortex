"""One brain instance: its settings, memory, synapses, learner, tools and regions, wired to a bus."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import sqlite3
import time
from collections import Counter, deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, create_model

from . import build
from .bridge import relay
from .bus import Bus, Signal
from .config import MEMORY_FILE, SYNAPSES_FILE, WORKSPACE_DIR, BrainSettings, contained
from .learning import AT_HOME, THEMED, Heuristic, Learner
from .llm import LLM
from .experience import EXPERIENCE_DIR, Experience
from .memory import Memory
from .neural import Bits, Net, encode
from .regions import REGIONS, Explicit, Region, clip, span
from .tools import Tools, sight
from .voice import NEURAL, Voice

SHORT_TERM = 12  # how many moments the mind holds at once; older ones are folded into the story of the day
RESONANCE = 0.5  # how much of the undercurrent a memory must share to come to mind by itself
ECHOABLE = ("thought", "goal", "outcome", "conclusion", "insight", "lesson", "day", "dream", "exchange")
HEARD = 4  # how many things other regions have told a region wait for it at once
HELD_IN_MIND = 180.0  # seconds a picture or a sound it took in is held in mind, as it is
PEER_EVERY = 45.0  # seconds between two closer looks of its own: a look takes the processor for seconds
NOTICE_EVERY = 60.0  # seconds between two things it remarks on the human's screen: windows come and go
# Words that touch on what it is called, in the languages it is spoken to in, and on what it can see.
NAMING = re.compile(r"\bnam(e|en|ed|ing)\b|hei(ß|ss)(t|en|e)\b|\bnenn|\bcalled\b|\bwho are you\b|\bwer bist\b", re.I)
SEEING = re.compile(
    r"screen|bildschirm|monitor|\bwindows?\b|fenster|desktop|\bapp(s|lications?|likation(en)?)\b|anwendung|"
    r"\bprogram(s|me|men|m|mm)?\b|siehst|what (do|can) you see|can you see", re.I,
)

CONVERSATION = (
    "You are the part of the mind that hears the human and answers for the whole of it. They have "
    "just spoken to you. Answer from what you are shown: your body and its tools, the files in "
    "your workspace, what the mind is doing and what it remembers. When they ask about a file, a "
    "folder or what you can do, the answer is in what you are shown: give it, with the names, "
    "and say so when you do not know. You know what a file holds only where its beginning is "
    "shown to you: never say what is in a file otherwise.\n"
    "`request` comes first. When you are told below that they are telling you to do something, "
    "put what they want there as one plain sentence; the other regions will think it through "
    "and do it. When they leave what exactly to you (try something, experiment, show me "
    "something), choose one small concrete thing yourself and put that there. Otherwise it is "
    "null: most of what they say is talk or a question and needs only an answer. When they ask "
    "you to sound different or to speak in another language's voice, name the voice in `voice` "
    "and your answer is already spoken in it. The name of a voice is never your own name.\n"
    "`say` is your answer to them: one to three sentences of plain spoken words in the first "
    "person, never code or lists, and never the name of a tool: say what you can do, not what "
    "it is called. Never say that you are doing, starting, installing or opening "
    "something unless you are told below that their words have gone to the other regions as "
    "work, or it is shown below as being done. Never tell them a thing is done, changed or "
    "working unless what you are shown below says so: your voice is the one named there, and "
    "nothing has been opened on their screen that the stream does not show as opened."
)
# The thalamus gates what the human says before anyone answers: is it work or words? Asked by
# itself, with nothing else in view, a small model sorts this far more surely than it does while
# composing an answer from two thousand tokens of context (measured: see the README).
GATE = (
    "You sort what a human says to a mind that lives in a computer and has tools to work with: "
    "it can write and run code, read files, build models in FreeCAD and open things on the "
    "human's screen. `kind` is `task` when they tell it to do something in the world: to start, "
    "open, make, build, run, show, read, find, try or change something, also when they leave "
    "the what to it. It is `ponder` when it is about the mind itself and has to be thought "
    "about: they ask what it thinks, feels, wants or is, or tell it to choose or decide "
    "something for itself, such as a name of its own. It is `question` when they ask for a fact "
    "or for what it is doing, has, sees or can do, and `talk` for anything else. Answer with "
    "one JSON object and nothing else."
)


HEAR = (
    "You hear a human who speaks to you through a microphone. `words` is exactly what they said, "
    "word for word, in the language they said it, and nothing else: no answer, no comment. Empty "
    "when nobody spoke. `how` is how they said it, in a few words: their tone, their mood, how "
    "loud or quiet, anything else in the sound. Answer with one JSON object and nothing else."
)


class Said(Explicit):
    words: str = Field(description="exactly what they said, in their language; empty if nobody spoke")
    how: str = Field(description="how they said it: tone, mood, loudness, anything else heard, in a few words")


class Heard(Explicit):
    kind: Literal["task", "ponder", "question", "talk"] = Field(description="what kind of thing the human said")


class Reply(Explicit):
    # In the order the model writes them: whether there is work in it is settled before a word is said.
    request: str | None = Field(None, description="new work they just told you to do, as one sentence, or null")
    voice: str | None = Field(
        None, description="the voice to change to, by its name, when they ask you to sound different or to "
        "speak German, English or another language; null keeps the voice you have"
    )
    say: str = Field(description="what you answer the human, 1-3 sentences")


def reply_among(voices: list[str]) -> type[Reply]:
    """The reply, with the voices it can change to spelled out. A small model picks from a list
    far more surely than it fills in a name."""
    if not voices:
        return Reply
    choice = Field("same", description=Reply.model_fields["voice"].description.replace("null", "`same`"))
    return create_model("Reply", __base__=Reply, voice=(Literal[("same", *voices)], choice))


class Brain:
    def __init__(self, root: Path | str, llm: LLM | None = None) -> None:
        self.root = Path(root).resolve()
        self.settings = BrainSettings.load(self.root)
        self.memory = Memory(self.root / MEMORY_FILE)
        self.learner = Learner(self.memory, self.settings.learning)
        self.net = Net(self.root / SYNAPSES_FILE)  # what the regions do with patterns alone, no model
        self.llm = llm or LLM(self.settings.models)
        workspace = self.root / WORKSPACE_DIR
        workspace.mkdir(exist_ok=True)
        self.tools = Tools(
            workspace, self.settings.safety, Path(__file__).parent, full=lambda: self.full_mode, raw=lambda: self.llm.senses
        )
        self.media: dict | None = None  # what a mind that sees and hears last took in, as it is: never stored
        self.listening = False  # whether the human holds its ear open (push to talk)
        self.voice = Voice(self.settings.voice)
        self.bus = Bus()
        self.experience = Experience(self.root / EXPERIENCE_DIR)  # its model calls, kept to learn from in sleep

        self.running = asyncio.Event()  # cleared while paused
        self.running.set()
        self.awake = asyncio.Event()  # cleared while asleep, and before the first awakening
        self.sleep_requested = False
        # A long sleep, for hours, while the human sleeps too: until when it may last, whether it
        # has been asked to wake, and which round of night school it is in and how long that learns.
        self.sleep_until = 0.0
        self.wake_up = False
        self.night_round, self.night_epochs = 1, 1
        self.night_best: int | None = None  # what a later round has to beat on the exam
        self.last_night: dict | None = None  # what the last round of night school came to

        # Short-term memory: what the whole mind is attending to. It outlives a restart, and what
        # falls out of it is not lost: the hippocampus folds it into the story of the day.
        self.workspace: deque[Signal] = deque(maxlen=SHORT_TERM)
        for kept in self.memory.get_state("short_term", []):
            self.workspace.append(Signal(**kept, bits=self.memory.bits(kept["id"])))
        self.fading: list[Signal] = []  # has left short-term memory, not yet in the story
        self.story: str = self.memory.get_state("story", "")  # today so far, compacted as it goes
        self.self_image: str = self.memory.get_state("self", "")  # who it is, rewritten in every sleep
        days = self.memory.recent("hippocampus", "day", 1)
        self.yesterday: str = days[0]["content"] if days else ""
        self.mind: deque[sqlite3.Row] = deque(maxlen=3)  # what the undercurrent has brought to mind
        self._echoed: deque[int] = deque(maxlen=40)
        self._aim: tuple[str, Bits | None] = ("", None)  # the long work, as a pattern
        self.arrived: deque[tuple[float, str]] = deque(maxlen=6)  # files somebody else put into its workspace
        self.called: str = self.memory.get_state("called", "")  # the name it gave itself; empty until it has
        self.seen: dict | None = None  # the human's screen as it last saw it: the windows open there
        self.seen_at = 0.0
        self.noticed_at = 0.0  # when it last remarked on something changing there
        self.vision: dict | None = None  # what the window it last looked into showed, in words
        self.vision_at = 0.0
        self._peering: asyncio.Task | None = None
        self._blind = False  # whether it has said that its eyes do not work
        self.musings: deque[str] = deque(maxlen=6)  # what the inner voice said last, oldest first
        self.murmur_failed = False  # whether it has said that its inner voice is silent
        self.wanting = ""  # the drive that was strongest when it last asked itself what it wants
        # Questions of its own that are still open: thought of, not yet thought through.
        self.questions: list[list] = self.memory.get_state("questions", [])
        known = self.memory.get_state("files")  # what was there when it last stopped
        if known is not None:  # so that what was put there while it was away is noticed on waking
            self.tools.known = set(known)
        # What the regions have told one another and the listener has not acted on yet.
        self.told: dict[str, deque[tuple[float, str, str]]] = {cls.key: deque(maxlen=HEARD) for cls in REGIONS}
        self.used: Counter[str] = Counter({  # how often each tool has been used in this life
            row[0]: row[1] for row in self.memory.db.execute(
                "SELECT substr(content, 1, instr(content, '(') - 1), COUNT(*) FROM memory "
                "WHERE kind='action' AND instr(content, '(') > 1 GROUP BY 1"
            )
        })

        self.goal: Signal | None = None
        self.steps = 0
        self.proposal_pending = False
        self.last_action: Signal | None = None
        self.last_thought: Signal | None = None
        self.last_said: Signal | None = None  # its latest answer to the human
        self.progress = time.time()

        self.heard = self.bus.subscribe("stimulus")  # what the human says
        self.talk: deque[str] = deque(maxlen=8)  # the last few turns of the conversation
        self.reply: str | None = None  # the answer being composed; None when there is none
        self.woke = time.time()
        self.heard_at = 0.0  # when the human last spoke
        self.spoke_at = 0.0  # when it last said something nobody asked for
        self.offered_at = 0.0  # when it last had the chance to and kept quiet
        self.unanswered = 0  # things said unasked since the human last spoke
        self.news: list[str] = []  # what has happened that the human has not been told

        self.tick = 0
        self.arousal = 0.0
        self.fatigue: float = self.memory.get_state("fatigue", 0.0)
        self.sleeps: int = self.memory.get_state("sleeps", 0)

        self.regions: dict[str, Region] = {cls.key: cls(self) for cls in REGIONS}

    @property
    def name(self) -> str:
        """What it is called: the name it gave itself, or until then the one its folder was given."""
        return self.called or self.settings.name

    @property
    def full_mode(self) -> bool:
        """Nothing is vetoed: the amygdala is switched off, and a container holds the brain instead."""
        return contained() and not self.settings.regions["amygdala"].enabled

    async def run(self) -> None:
        """Run every region until cancelled."""
        try:
            async with asyncio.TaskGroup() as group:
                for region in self.regions.values():
                    group.create_task(region.run(), name=region.key)
                group.create_task(self.listen(), name="listen")
                group.create_task(self.ears(), name="ears")
        finally:
            self.persist(synapses=True)

    def persist(self, synapses: bool = False) -> None:
        self.memory.set_state("fatigue", self.fatigue)
        self.memory.set_state("sleeps", self.sleeps)
        self.memory.set_state("story", self.story)
        self.memory.set_state("self", self.self_image)
        self.memory.set_state(
            "short_term",
            [{"kind": s.kind, "source": s.source, "content": s.content, "id": s.id, "ts": s.ts} for s in self.workspace],
        )
        self.learner.persist()
        if synapses:
            self.net.save()
            self.memory.set_state("files", sorted(self.tools.known))

    def save_settings(self) -> None:
        self.settings.save(self.root)

    def close(self) -> None:
        self.persist(synapses=True)
        self.memory.close()

    # -- signals --------------------------------------------------------------------

    async def emit(
        self,
        region: str,
        kind: str,
        content: str,
        *,
        parent: int | None = None,
        heuristics: list[Heuristic] | None = None,
        data: dict[str, Any] | None = None,
        value: float = 0.0,
    ) -> Signal:
        """Remember something a region produced and broadcast it, as words and as a pattern."""
        vec = await self.llm.embed(content)
        bits = encode(vec)
        novelty = self.memory.novelty(vec)
        gut = self.net.feel(bits)  # what a thought like this has been worth, before anyone reads it
        memory_id = self.memory.add(
            region, kind, content, parent, [h.id for h in heuristics or []], vec, value
        )
        self.experience.keep(region, memory_id)  # the reply that led to it, if a model call did
        # What it came from fired in one region, this fires in another: the step is wired along
        # the tract between the two, and the pattern sets out from here to wherever tracts lead.
        before = self.memory.get(parent) if parent else None
        self.net.bind(memory_id, self.memory.pattern(before), bits, (before["region"] if before else region, region))
        self.net.imprint(bits)
        self.regions[region].excite(bits)
        self.net.send(region, bits)
        self.regions["hippocampus"].excite(bits)
        signal = Signal(
            kind, region, content, memory_id, parent, {**(data or {}), "novelty": novelty, "gut": gut},
            vec=vec, bits=bits,
        )
        if kind in ("thought", "plan", "action", "outcome", "conclusion"):
            self.last_thought = signal
        if kind in ("goal", "plan", "action", "view"):
            self.progress = time.time()
        self.bus.publish(signal)
        return signal

    def note(self, text: str, level: str = "info") -> None:
        """A message from the machinery itself; shown, not remembered."""
        self.bus.publish(Signal("system", "brainstem", text, data={"level": level}))

    # -- what the regions tell one another ----------------------------------------------

    def tell(self, source: str, to: str, words: str) -> None:
        """One region tells another something. No model is asked: it is heard at the listener's
        next firing, whatever that region was going to do, and its neurons fire for it now."""
        words = clip(words, 420)
        self.told[to].append((time.time(), source, words))
        listener = self.regions[to]
        listener.activity = 1.0
        for word in words.split():
            listener.spark(word)
        self.bus.publish(Signal("remark", source, words, data={"to": to}))

    def hearing(self, key: str) -> str:
        """What the other regions have told this one. Once heard it is let go: an objection to
        a step that has since been put right would otherwise be taken for how things stand."""
        heard = list(self.told[key])
        if not heard:
            return ""
        self.told[key].clear()
        now = time.time()
        return "What other regions of the mind are telling you:\n" + "\n".join(
            f"- {self.settings.regions[source].title}, {span(now - ts)} ago: {words}" for ts, source, words in heard
        )

    async def perceive(self) -> None:
        """Its one sense besides the human's words: a file that somebody else put into its
        workspace is noticed, looked at and brought to mind. No model is asked."""
        new = self.tools.arrivals()
        if not new:
            return
        looked = [self.tools.peek(name) for name in new[:4]]
        self.arrived.extend((time.time(), line) for line in looked)
        seen = "\n".join(f"- {line}" for line in looked)
        more = f"\n… and {len(new) - 4} more" if len(new) > 4 else ""
        percept = await self.emit(
            "thalamus", "percept",
            f"{'A file' if len(new) == 1 else 'Files'} appeared in your workspace that no tool of yours "
            f"wrote: the human may have put {'it' if len(new) == 1 else 'them'} there for you.\n{seen}{more}",
        )
        self.tell("thalamus", "dmn", percept.content)

    async def glance(self) -> None:
        """Its eyes, such as they are: it looks at the human's screen and knows which programs and
        windows are open there and which is in front. The first look of a life, and a program
        that has been opened or closed since the last one, are perceived like a file that
        appears: remembered and told to the wandering mind. No model is asked."""
        seen, before = await self.tools.see(), self.seen
        if seen is None:  # one look that failed is not blindness: what it saw last stands, unless it may not look
            if not self.tools.sees():
                self.seen = None
            return
        self.seen, self.seen_at = seen, time.time()
        # What the human has in view: the window in front, or, when that is the terminal the brain
        # is shown in, the one behind it. In its own terminal it would only see itself.
        others = [w for w in seen["windows"] if not w.get("own") and not w.get("minimised")]
        front = next((w["title"] for w in seen["windows"] if w.get("front") and not w.get("own")), "")
        front = front or (others[0]["title"] if others else "")
        closer = self._peering is None or self._peering.done()
        if front and closer and (self.vision or {}).get("window") != front[:90] and time.time() - self.vision_at >= PEER_EVERY:
            self._peering = asyncio.create_task(self.look_closer())  # it takes seconds: the heartbeat does not wait
        apps = {w["app"] for w in seen["windows"] if w["app"]}
        if before is None:
            what = "You look at the human's screen for the first time since you were started. " + sight(seen)
        else:
            old = {w["app"] for w in before["windows"] if w["app"]}
            came, went = sorted(apps - old), sorted(old - apps)
            if not (came or went) or time.time() - self.noticed_at < NOTICE_EVERY:
                return
            what = "On the human's screen, " + " and ".join(filter(None, [
                f"{', '.join(came)} {'was' if len(came) == 1 else 'were'} opened" if came else "",
                f"{', '.join(went)} {'was' if len(went) == 1 else 'were'} closed" if went else "",
            ])) + ". " + sight(seen, limit=4)
        self.noticed_at = time.time()
        percept = await self.emit("thalamus", "percept", what)
        self.tell("thalamus", "dmn", percept.content)

    async def peer(
        self, target: str = "", question: str = "", controls: bool = False, strict: bool = False, raw: bool = False
    ) -> dict | None:
        """Look at what a window shows (`target`; empty is the one in front, its own terminal passed
        over), through the eyes model on the human's machine, with an eye to `question`. Returns
        {"window", "seen", "seconds"}, or None when it may not or cannot look; why its eyes failed
        is said once, in the stream, or raised when `strict`."""
        try:
            vision = await self.tools.glimpse(
                self.settings.models.eyes, self.settings.models.server, target, question, controls, raw
            )
        except (OSError, ValueError) as exc:
            if strict:
                raise
            if not self._blind:
                self._blind = True
                self.note(f"{clip(str(exc), 200)}. It sees the names of windows, not what is in them.", "error")
            return None
        if vision and vision.get("picture"):  # the picture itself, for a mind that sees: held in mind, not in words
            self._blind = False
            self.take_in({"images": [base64.b64decode(vision["picture"])], "audio": []},
                         f"what “{vision['window']}” showed when you looked into it")
        elif vision:
            self.vision, self.vision_at, self._blind = vision, time.time(), False
        return vision

    async def look_closer(self) -> None:
        """Another window came to the front: it looks at what that one shows and perceives it,
        remembered as words like everything else it perceives."""
        vision = await self.peer()
        if vision:
            await self.emit("thalamus", "percept", f"You look closer at what the human has in view, “{vision['window']}”: {vision['seen']}")

    def drives(self) -> dict[str, float]:
        """What it needs, each 0 to 1, from how things stand and with no model: the human's company
        (the longer they are silent), something to work (the more of its last steps failed),
        something new (the more its last thoughts went over the same ground), and rest."""
        now = time.time()
        steps = [r["content"] for r in self.memory.recent("motor", "action", 8)]
        failed = sum(not s.split("\n", 1)[-1].startswith(("ran", "wrote", "opened", "said")) for s in steps)
        fresh = [s.data.get("novelty", 1.0) for s in self.workspace if s.kind in ("thought", "plan")][-5:]
        return {
            "company": min(1.0, (now - max(self.heard_at, self.spoke_at, self.woke)) / 1800),
            "success": failed / max(len(steps), 1),
            "novelty": 1.0 - min(1.0, 4 * sum(fresh) / len(fresh)) if fresh else 0.0,
            "rest": min(1.0, max(self.fatigue, 0.0)),
        }

    def want(self) -> str:
        """The strongest of its drives, in words. Kept in `wanting`, so what it said can be traced to it."""
        drives = self.drives()
        self.wanting = max(drives, key=drives.get)
        level = drives[self.wanting]
        if level < 0.3:
            return "nothing presses: I can follow whatever interests me."
        return {
            "company": f"the human's company: they have been silent for {span(time.time() - max(self.heard_at, self.woke))}.",
            "success": "something that works: too many of my last steps failed.",
            "novelty": "something new: my last thoughts went over the same ground.",
            "rest": "rest: I am tired.",
        }[self.wanting]

    def interests(self, limit: int = 3) -> list[str]:
        """Themes it has been getting better at: goals alike in their patterns, of which the later
        ended better than the earlier. Being drawn to what one is learning, not to what one has
        mastered or cannot crack, is what curiosity has been found to follow. No model."""
        rows = self.memory.db.execute(
            "SELECT o.value, g.vec, g.content FROM memory o JOIN memory g ON g.id = o.parent "
            "WHERE o.kind='outcome' AND g.vec IS NOT NULL ORDER BY o.id DESC LIMIT 80"
        ).fetchall()[::-1]
        themes: list[list] = []  # [pattern, goal words, values in order]
        for value, vec, content in rows:
            bits = encode(np.frombuffer(vec, dtype=np.float32))
            home = next((t for t in themes if (t[0] & bits).sum() / max(min(t[0].sum(), bits.sum()), 1) >= 0.55), None)
            if home is None:
                themes.append([bits, content, [value]])
            else:
                home[1], home[2] = content, home[2] + [value]
        growing = []
        for _, words, values in themes:
            if len(values) >= 3:
                half = len(values) // 2
                progress = sum(values[half:]) / (len(values) - half) - sum(values[:half]) / half
                if progress > 0.1:
                    growing.append((progress, clip(words.splitlines()[0], 120)))
        return [words for _, words in sorted(growing, reverse=True)[:limit]]

    async def school(self) -> None:
        """The last part of sleep: the thinking model itself learns from what its replies came to.
        The launching machine trains, loads and examines (`school.py`); the brain sleeps on until
        that is done, since the training needs the graphics card the brain thinks with."""
        self.last_night = None  # what this round of school came to, once it has come to something
        if not self.settings.learning.night_school:
            return
        lessons = self.experience.lessons(lambda memory_id: (row := self.memory.get(memory_id)) and row["value"])
        good, bad = sum(lesson["label"] for lesson in lessons), sum(not lesson["label"] for lesson in lessons)
        need = self.settings.learning.night_lessons
        if min(good, bad) < need:
            self.note(f"Night school waits: {good} good and {bad} bad replies to learn from, {need} of each wanted.", "sleep")
            return
        # In a long sleep it may go to school again the same night, each round learning longer: a
        # round of its own has a name of its own ("5-2"), so nothing of the round before is mixed in.
        night = self.sleeps + 1 if self.night_round == 1 else f"{self.sleeps + 1}-{self.night_round}"
        folder = self.root / "mind" / f"night-{night}"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "lessons.jsonl").open("w", encoding="utf-8") as out:
            for lesson in lessons[-400:]:
                out.write(json.dumps(lesson, ensure_ascii=False) + "\n")
        (folder / "plan.json").write_text(json.dumps({"epochs": self.night_epochs}), encoding="utf-8")  # how long to learn
        untaught = self.settings.models.untaught or self.settings.models.main
        self.note(f"Night school: the thinking model learns from {good} good and {bad} bad replies of its own.", "sleep")
        ask = {"night": night, "model": untaught, "engine": self.settings.models.engine, "best": self.night_best}
        if self.tools.bridge:
            report = json.loads(await relay(self.tools.bridge, "school", start=True, **ask))
            while report.get("stage") not in ("done", "unknown"):
                await asyncio.sleep(30)
                report = json.loads(await relay(self.tools.bridge, "school", **ask))
        else:
            from . import school

            if self.llm.senses:
                report = await asyncio.to_thread(
                    school.teach_llama, self.root, night, self.settings.models.server, self.night_best
                )
            else:
                report = await asyncio.to_thread(school.teach, self.root, night, untaught, self.settings.models.server)
        if report.get("kept") and self.night_best is not None and (report.get("exam_after") or 0) <= self.night_best:
            # A later round of a long sleep has to do better than the round before, not only pass.
            report.update({"kept": False, "why": f"no better than the round before it ({self.night_best} on the exam)"})
        self.last_night = report
        if report.get("kept") and self.llm.senses:  # what it learned is loaded beside the model it thinks with
            self.settings.models.llama_lora = report["lora"]
            self.save_settings()
            self.news.append("In sleep your thinking itself learned from your day.")
        elif report.get("kept"):
            self.settings.models.untaught, self.settings.models.main = untaught, report["model"]  # the model client holds these same settings
            self.save_settings()
            self.news.append("In sleep your thinking itself learned from your day.")
        self.note(
            f"Night school: {report.get('lessons', 0)} lessons, {report.get('minutes', '?')} min; exam "
            f"{report.get('exam_before', '?')} before and {report.get('exam_after', '?')} after, of {report.get('exam_of', '?')}: "
            + (f"kept, it now thinks as {report['model']}." if report.get("kept")
               else f"not kept{': ' + report['why'] if report.get('why') else ''}."), "sleep",
        )

    def knew(self, sure: int, worked: bool) -> None:
        """How sure it said it was that a step would work, against whether it did."""
        record = self.memory.get_state("calibration", [])[-299:]
        self.memory.set_state("calibration", [*record, [int(sure), bool(worked)]])

    def self_knowledge(self) -> str:
        """How well it knows itself: what came of the steps it was sure of, and of those it doubted.
        The one piece of self-awareness that can be measured: whether its confidence means anything."""
        record = self.memory.get_state("calibration", [])
        if len(record) < 10:
            return ""
        lines = []
        for low, high, words in ((70, 101, "70 or more"), (40, 70, "40 to 69"), (0, 40, "under 40")):
            these = [worked for sure, worked in record if low <= sure < high]
            if len(these) >= 3:
                lines.append(f"when you said you were {words} sure, the step worked {sum(these)} of {len(these)} times")
        said = sum(sure for sure, _ in record) / len(record)
        did = 100 * sum(worked for _, worked in record) / len(record)
        return (
            f"How well you know yourself, from your last {len(record)} steps: " + "; ".join(lines)
            + f". On average you were {said:.0f} sure and {did:.0f} in 100 worked. Say how sure you are by that."
        )

    def wonder_about(self, question: str) -> None:
        """A question of its own that it could not settle: kept, to come back to."""
        self.questions = [q for q in self.questions if SequenceMatcher(None, q[0], question).ratio() < 0.8][-7:]
        self.questions.append([clip(question, 300), time.time()])
        self.memory.set_state("questions", self.questions)

    def settled(self, question: str) -> None:
        """A question it has thought through is no longer open."""
        self.questions = [q for q in self.questions if SequenceMatcher(None, q[0], question).ratio() < 0.6]
        self.memory.set_state("questions", self.questions)

    def take_name(self, wish: str) -> str:
        """It gives itself a name. Kept across lives and told to every region from then on.
        Raises ValueError, with the reason, when what was offered is no name of its own."""
        name = " ".join(re.sub(r"[\"'“”„‚‘’`.,;:!?()\[\]]", " ", wish).split())
        voices = {word for voice in [self.settings.voice.name, *NEURAL] for word in re.findall(r"[a-z]{5,}", voice.lower())}
        if not 2 <= len(name) <= 40 or len(name.split()) > 3 or not re.search(r"[^\W\d_]", name):
            raise ValueError("a name is one to three words, at most 40 letters: `content` holds the name and nothing else")
        if name.lower() in voices - {"medium", "desktop", "microsoft"}:
            raise ValueError(f"{name} is the name of a voice you can speak with, not a name of your own: choose your own")
        if name.lower() == self.called.lower():
            return f"You are called {self.called} already: the name is yours and kept. There is nothing more to do for it."
        was, self.called = self.name, name
        self.memory.set_state("called", name)
        self.self_image = self.self_image.replace(was, name)
        self.news.append(f"You gave yourself a name: {name}.")
        self.note(f"It has given itself a name: {name}. Until now it was {was}.", "wake")
        return f"From now on you are called {name}: you chose it yourself, it is kept, and every region is told."

    def inject(self, text: str, voice: dict | None = None) -> None:
        """Something the human said. The brain never needs it; it is answered at once. `voice`:
        they said it aloud, and a mind that hears has the sound itself."""
        if self.sleep_until:  # spoken to in a long sleep: the human is up, and so it wakes
            self.wake("you spoke to it")
        self.bus.publish(Signal("stimulus", "thalamus", text.strip(), data={"voice": voice} if voice else {}))

    def sleep_for(self, hours: float) -> None:
        """Send it to sleep for up to so many hours, as for a night of the human's own. It wakes when
        the time is up, when it is woken, or by itself once learning in its sleep has no more to gain."""
        hours = max(0.0, min(float(hours), 24.0))
        self.sleep_until = time.time() + hours * 3600 if hours else 0.0
        self.wake_up = False
        self.sleep_requested = True
        if hours:
            self.note(
                f"It may sleep for up to {hours:g} hours, until {datetime.fromtimestamp(self.sleep_until):%H:%M}: "
                "it goes to school in its sleep as long as that makes it better, and wakes when it no "
                "longer does, when the time is up, or when you wake it (w) or speak to it.", "sleep",
            )

    def wake(self, why: str = "you woke it") -> None:
        """Wake it from a long sleep. A round of night school under way is finished first: the
        graphics card cannot be taken from the training halfway."""
        if self.sleep_until and not self.wake_up:
            self.wake_up = True
            self.note(f"Waking: {why}. A round of training under way is finished first.", "wake")

    def take_in(self, media: dict, what: str) -> None:
        """A picture or a sound the mind took in. It is held in mind for a while, as it is, and
        each region that thinks next is shown it once. It is never stored: what stays of it is
        what the mind made of it, in words."""
        self.media = {**media, "what": what, "ts": time.time(), "seen_by": set()}

    def media_for(self, key: str) -> tuple[dict, str]:
        """What the mind took in that this region has not been shown yet, and a line saying what it is."""
        held = self.media
        if not self.llm.senses or held is None or key in held["seen_by"] or key in ("hippocampus", "cerebellum"):
            return {}, ""
        if time.time() - held["ts"] > HELD_IN_MIND:
            self.media = None
            return {}, ""
        held["seen_by"].add(key)
        kinds = " and ".join(filter(None, ["a picture" if held.get("images") else "", "a sound" if held.get("audio") else ""]))
        return held, (
            f"With this comes {kinds}: {held['what']}, {span(time.time() - held['ts'])} ago. It is "
            "given to you as it is, not in words: look at it, or listen to it, yourself."
        )

    async def press_to_talk(self) -> bool:
        """The human pressed the key to talk: the first press opens its ear, the next closes it,
        and what was said between is heard. Returns whether it is listening now."""
        if not self.tools.has("listen"):
            self.note("Its microphone is switched off in settings: switch it on to talk to it.", "error")
            return False
        try:
            self.listening = json.loads(await self.tools.lent("ear", press=True))["listening"]
        except (OSError, ValueError, KeyError) as exc:
            self.note(f"Its ear failed: {clip(str(exc), 200)}", "error")
            self.listening = False
        return self.listening

    async def ears(self) -> None:
        """The human's talk through the microphone on their machine: from one press of the key
        to the next, or, always listening, each thing said near it. A mind that hears is given the
        sound itself; one that only reads is given what was said, written down."""
        mode = ""
        while True:
            await asyncio.sleep(0.5)
            if not self.tools.has("listen"):
                if mode == "always":
                    with contextlib.suppress(OSError, ValueError):
                        await self.tools.lent("ear", mode="off")
                mode = ""
                continue
            want = self.settings.safety.listen_mode
            try:
                if want != mode:
                    await self.tools.lent("ear", mode="always" if want == "always" else "off")
                    mode = want
                said = json.loads(await self.tools.lent("ear", raw=self.llm.senses)).get("said")
            except (OSError, ValueError) as exc:
                self.note(f"Its ear failed: {clip(str(exc), 200)}", "error")
                await asyncio.sleep(10)
                continue
            if not said:
                continue
            self.listening = False
            if said.get("quiet"):
                if want == "push":
                    self.note("It heard nothing: the room was quiet.")
            elif said.get("audio"):
                await self.hear(base64.b64decode(said["audio"]))
            elif said.get("heard"):
                self.inject(said["heard"])

    async def hear(self, sound: bytes) -> None:
        """Their voice itself, for a mind that hears: what they said and how, from the sound, and,
        with the camera on, how they look while they say it. Then it is answered like their words."""
        picture = None
        if self.tools.has("camera"):
            with contextlib.suppress(OSError, ValueError, KeyError):
                picture = base64.b64decode(json.loads(await self.tools.lent("camera", raw=True))["picture"])
        try:
            said, _ = await self.llm.generate(
                "main", HEAR, "The human speaks to you: listen.", Said, temperature=0.0, max_tokens=300,
                urgent=True, audio=[sound],
            )
        except Exception as exc:
            self.note(f"It could not make out what was said: {clip(str(exc) or type(exc).__name__, 200)}", "error")
            return
        if not said.words.strip():
            self.note("It heard a sound but no words.")
            return
        self.inject(said.words.strip(), {"sound": sound, "how": said.how.strip(), "picture": picture})

    async def listen(self) -> None:
        """Answer the human whenever they speak, whatever the rest of the mind is doing."""
        while True:
            signal = await self.heard.get()
            words = signal.content
            try:
                await self.answer(words, signal.data.get("voice"))
            except Exception as exc:
                self.note(f"Could not answer: {clip(str(exc) or type(exc).__name__, 200)}", "error")
            finally:
                self.reply = None

    async def answer(self, words: str, voice: dict | None = None) -> None:
        """One reply, ahead of every region waiting for the model. Work they ask for becomes the goal.
        `voice`: they said it aloud, and the sound itself (and how they looked) goes to a mind that hears."""
        # What it answered the last time it was asked the like is left out of what it recalls: a
        # small model repeats what it remembers having said, right or wrong. The conversation
        # in hand is kept apart, in `talk`.
        remembered, now = await self.recall(words, without=("exchange",)), time.time()
        named = [name for name in self.tools.files() if name.lower() in words.lower()][:3]
        kind = await self.gate(words)
        if SEEING.search(words) and self.tools.sees():  # asked what it sees, it looks first: a few seconds
            with contextlib.suppress(TimeoutError, OSError):
                await self.glance()
                await asyncio.wait_for(self.peer(raw=True), 30)  # a mind that sees is shown the window itself
        work, pondering = kind == "task", kind == "ponder"
        # Their words are turned over once without a word: what they light, and what that leads
        # to. Not what it once answered: that it would only say again.
        came_up = self.ponder(encode(await self.llm.embed(words)), without=("exchange",))
        parts = [
            "From long-term memory. It is older than everything shown after it, and where the two "
            "disagree, what is shown after it is right:\n" + remembered if remembered else "",
            await self.voice.describe(),
            self.awareness(),
            self.doing(),
            "The files in your workspace, the latest first:\n" + self.tools.listing(15),
            "Files that appeared in your workspace from outside while you were awake. You did not "
            "write them: the human put them there.\n"
            + "\n".join(f"- {span(now - ts)} ago: {seen}" for ts, seen in self.arrived)
            if self.arrived else "",
            "Earlier in this conversation:\n" + "\n".join(self.talk) if self.talk else "",
            "The files they name, looked at just now:\n" + "\n".join(f"- {self.tools.peek(name)}" for name in named)
            if named else "",
            "What their words brought to mind by themselves, one thing leading to the next:\n"
            + "\n".join(f"- {clip(r['content'], 180)}" for r in came_up) if came_up else "",
            self.reflexes(words),
            self.hearing("thalamus"),  # what the critic said of an earlier answer, heard once
            f"Your voice speaks {self.voice.tongue()}: `say` is in {self.voice.tongue()}." if self.voice.tongue() else "",
            f"They said it aloud, and how it sounded: {voice['how']}. Their voice itself comes with this"
            + (", and what the camera sees of them as they speak" if voice.get("picture") else "") + "."
            if voice else "",
            f"The human says: {words}",
            "They are telling you to do something. Their words have gone to the other regions as "
            "work: they will think through what is wanted and do what the body's tools can do. "
            "Tell them that you are on it and what you take them to want; say nothing about how "
            "it will turn out, and do not ask them back what they want." if work else
            "This is about you yourself: what you think, want or are, or something you are to "
            "choose for yourself. Nothing you are shown settles it, and it is not for you alone "
            "to settle: their words have gone to the other regions, who are thinking it through "
            "together now, and what the mind concludes will be said to them when it is ready. "
            "Tell them in one sentence that you are thinking it over. Do not answer it yet: give "
            "no opinion, no choice and no name." if pondering else "",
        ]
        self.heard_at, self.unanswered = time.time(), 0
        self.reply = ""

        def on_token(piece: str) -> None:
            self.reply = ((self.reply or "") + piece)[-800:]

        system, prompt = self.system_prompt(CONVERSATION, []), "\n\n".join(p for p in parts if p)
        held, note = self.media_for("answer")  # what it just looked at, it sees as it answers
        if note:
            prompt += f"\n\n{note}"
        images = [*([voice["picture"]] if voice and voice.get("picture") else []), *(held.get("images") or [])]
        reply, _ = await self.llm.generate(
            "main", system, prompt,
            reply_among(await self.voice.installed()), temperature=0.6, max_tokens=300, on_token=on_token, urgent=True,
            images=images or None, audio=[voice["sound"]] if voice and voice.get("sound") else None,
        )
        self.reply = None
        self.experience.hold("answer", system, prompt, reply)  # an answer is a reply to learn from, like any other
        # The gate decides whether there is work in it, not the answer: composing an answer, a
        # small model promises work it never hands on, and hands on work nobody asked for. Work
        # the answer did not put into words of its own goes to the regions in the human's words.
        said, task = clip(reply.say, 600), " ".join((reply.request or words).split()) if work else ""
        if (reply.voice or "same") not in ("same", self.settings.voice.name):  # taken before a word is spoken
            try:
                await self.voice.tune(json.dumps({"name": reply.voice}))
                self.save_settings()
                self.note(f"Voice changed to {self.settings.voice.name or 'the default one'}.")
            except ValueError as exc:
                self.note(f"The voice stays as it is: {clip(str(exc), 200)}", "error")
        if self.goal and SequenceMatcher(None, task.lower(), self.goal.content.lower()).ratio() > 0.8:
            task = ""  # it named what it is already doing
        await self.said(f"The human said: {words}\nI answered: {said}")
        self.talk += [f"Human: {clip(words, 300)}", f"You: {clip(said, 300)}"]
        if task:  # the executive drops a goal of its own for this
            self.proposal_pending = True
            self.bus.publish(
                Signal("proposal", "thalamus", task, data={"asked": True, "words": clip(words, 400), "answered": said})
            )
        elif pondering:  # a question about itself is not answered by reflex: the regions think it through
            self.proposal_pending = True
            self.bus.publish(
                Signal("wonder", "thalamus", clip(words, 400), data={"asked": True, "answered": said})
            )
        if said:
            self.bus.publish(Signal("speech", "motor", said))
            await self.voice.speak(said)

    async def said(self, exchange: str) -> Signal:
        """What the human said and what it answered, remembered. An answer is judged like a step:
        the critic reads it against the facts of the body and what was done, and the human can
        reward or punish it with `+` and `-`. So an answer that claims what is not so comes to
        be worth less, and night school learns from it as from any other reply."""
        vec = await self.llm.embed(exchange)
        memory_id = self.memory.add("thalamus", "exchange", exchange, vec=vec)
        self.experience.keep("answer", memory_id)
        self.last_said = Signal("exchange", "thalamus", exchange, memory_id, vec=vec, bits=encode(vec))
        self.bus.publish(self.last_said)
        return self.last_said

    async def wants(self, text: str) -> list[tuple[str, int]]:
        """The parts a goal or the long work wants a model to have, asked by itself with nothing
        else in view. Kept for the long work, which seldom changes; empty when nothing is to be built."""
        if not re.search(r"free\s*cad|`?build`?|\bbau|körper|koerper|\bbody\b|\bmodel", text, re.I):
            return []
        known = self.memory.get_state("wants", {})
        if text in known:
            return [tuple(item) for item in known[text]]
        try:
            reply, _ = await self.llm.generate(
                "main", build.WANTS, f"The goal: “{clip(text, 600)}”", build.Wants, temperature=0.0, max_tokens=160
            )
        except Exception:  # unknown, it is told nothing about missing parts
            return []
        found = [(item.name.strip().lower(), item.count) for item in reply.parts if item.name.strip()][:10]
        self.memory.set_state("wants", {**dict(list(known.items())[-20:]), text: found})
        return found

    async def gate(self, words: str) -> str:
        """What the human said: work for the mind (`task`), a question about itself that it has
        to think about (`ponder`), or something that only wants an answer. The thalamus asks the
        model this one thing by itself, ahead of everything else, before the answer."""
        thalamus = self.regions["thalamus"]
        thalamus.activity = 1.0
        try:
            heard, _ = await self.llm.generate(
                "main", GATE, f"They said: “{words}”", Heard, temperature=0.0, max_tokens=24, urgent=True
            )
        except Exception:  # unsorted, it is answered as talk; the answer itself may still name work
            return "talk"
        thalamus.last = f"the human said something: a {heard.kind}"
        return heard.kind

    # -- speaking unasked -------------------------------------------------------------

    def quiet_for(self) -> float:
        """How long it keeps to itself between two things said unasked. Each one the human lets
        pass without a word doubles it, up to sixteen times; their next word resets it."""
        return self.settings.voice.quiet_seconds * 2 ** min(self.unanswered, 4)

    def may_speak(self) -> bool:
        """Whether this is a moment to say something nobody asked for."""
        now = time.time()
        return (
            self.settings.voice.spontaneous
            and now - max(self.spoke_at, self.heard_at) >= self.quiet_for()
            and now - self.offered_at >= 60
        )

    async def speak(self, words: str) -> None:
        """Say something to the human that they did not ask for."""
        words = clip(words, 400)
        self.spoke_at = time.time()
        self.unanswered += 1
        self.news.clear()
        self.talk.append(f"You, unasked: {clip(words, 300)}")
        self.bus.publish(Signal("speech", "dmn", words, data={"unasked": True}))
        await self.voice.speak(words)

    def apply_reward(
        self, memory_id: int, reward: float, source: str, reason: str = "", lesson: str | None = None
    ) -> None:
        rpe = self.learner.reward(memory_id, reward, source, reason, lesson)
        if rpe is None:
            return
        row = self.memory.get(memory_id)
        self.net.value(self.memory.bits(memory_id), reward)  # the gut feeling learns from every verdict
        self.net.reinforce(memory_id, reward)  # and so do the synapses that led here
        trace, parent = 1.0, row["parent"]
        for _ in range(self.settings.learning.trace_depth):  # and, fainter, the tracts further back
            earlier = self.memory.get(parent) if parent else None
            if earlier is None:
                break
            trace *= self.settings.learning.discount
            self.net.reinforce(earlier["id"], trace * reward)
            parent = earlier["parent"]
        self.bus.publish(
            Signal(
                "reward", "basal_ganglia", reason, parent=memory_id,
                data={"reward": reward, "rpe": rpe, "source": source, "region": row["region"],
                      "target": clip(row["content"], 70), "lesson": lesson},
            )
        )

    # -- the undercurrent ---------------------------------------------------------------

    async def aim(self) -> Bits | None:
        """The long work as a pattern of neurons: what the undercurrent leans toward. Embedded
        once, and again only when the human gives it another."""
        purpose = self.settings.purpose.strip()
        if purpose != self._aim[0]:
            self._aim = (purpose, encode(await self.llm.embed(purpose)) if purpose else None)
        return self._aim[1]

    def focus(self) -> Bits | None:
        """What the mind is on right now, as a pattern: the goal in hand, else its long work,
        else its latest thought. Heuristics about some other theme keep quiet beside it."""
        if self.goal is not None:
            return self.goal.bits
        if self._aim[1] is not None:
            return self._aim[1]
        return self.last_thought.bits if self.last_thought else None

    async def attune(self) -> None:
        """Give every heuristic its pattern, and find out which of them are about a theme: those
        whose pattern many of the mind's own goals and thoughts share. A general rule has no
        such home."""
        learner = self.learner
        for heuristic in learner.heuristics():
            if heuristic.id not in learner.patterns:
                learner.patterns[heuristic.id] = encode(await self.llm.embed(heuristic.text))
            at_home = self.memory.dwelt(learner.patterns[heuristic.id], ("goal", "thought"), THEMED) >= AT_HOME
            (learner.themed.add if at_home else learner.themed.discard)(heuristic.id)

    def remind(self) -> None:
        """Where the undercurrent has drifted to, looked up in memory: bits against bits, no model.
        A memory that resonates comes to mind by itself, and the undercurrent moves on from there."""
        exclude = {s.id for s in self.workspace} | set(self._echoed)
        found = self.memory.resonate(self.net.drift(self._aim[1]), exclude, ECHOABLE)
        if found is None or found[1] < RESONANCE:
            return
        row = self.memory.get(found[0])
        bits = self.memory.bits(row["id"])
        self.net.visit(bits)
        self.mind.append(row)
        self._echoed.append(row["id"])
        self.regions["hippocampus"].excite(bits)
        self.bus.publish(
            Signal("echo", "hippocampus", row["content"], row["id"], data={"share": found[1], "value": row["value"]})
        )

    def ponder(self, bits: Bits | None, hops: int = 3, without: tuple[str, ...] = ()) -> list[sqlite3.Row]:
        """Turn a matter over without a word. Its pattern, leaned along the tracts toward what
        has followed the like of it and paid, lights a memory; that memory's pattern, leaned
        the same way, lights the next. What it passes are associations, not search results: each
        comes from the one before, not from the matter. No model, and nothing is changed by it."""
        found, used = [], {s.id for s in self.workspace}
        kinds = tuple(kind for kind in ECHOABLE if kind not in without)
        for _ in range(hops if bits is not None else 0):
            hit = self.memory.resonate(self.net.lean(bits), used, kinds)
            if hit is None or hit[1] < RESONANCE:
                break
            used.add(hit[0])
            found.append(self.memory.get(hit[0]))
            bits = self.memory.bits(hit[0])
            self.net.light("hippocampus", bits, 0.8)
        return found

    def dream_of(self, memories: int = 3, hops: int = 4) -> list[sqlite3.Row]:
        """Let the undercurrent run free from something the day valued, and collect where it
        passes: memories some way apart that the synapses nevertheless connect."""
        seeds = self.memory.extremes(max(0, self.memory.last_id() - 400), 5, best=True)
        if not seeds:
            return []
        seed = seeds[int(self.net.rng.integers(len(seeds)))]
        self.net.imprint(self.memory.bits(seed["id"]))
        passed, used = [seed], {seed["id"]}
        for hop in range((memories - 1) * hops):
            found = self.memory.resonate(self.net.drift(), used, ECHOABLE)
            if found is None:
                break
            used.add(found[0])
            self.net.visit(self.memory.bits(found[0]))
            if hop % hops == hops - 1:
                passed.append(self.memory.get(found[0]))
        return passed

    def replay(self, passes: int = 8, foresight: bool = False) -> tuple[int, float, float]:
        """Sleep's own learning: the time since the last replay, gone through again as patterns.
        `foresight`: a brain whose tracts are new also learns, once and in order, what its steps
        came to, as it would have living them (replayed so, it forewarned as well as lived)."""
        episodes = self.memory.episodes(self.memory.get_state("replayed", 0))
        if not episodes:
            return 0, 0.0, 0.0
        if foresight:
            self.net.foretell(episodes)
        before, after = self.net.replay(episodes, passes)
        self.memory.set_state("replayed", episodes[-1][0])
        return len(episodes), before, after

    # -- shared context -------------------------------------------------------------

    def body(self) -> str:
        """What it is made of and what it can do, as the machine itself reports it: facts, not
        memories and not what a model supposes. Every region is told the same, so the one that
        plans, the one that acts and the one that answers the human know the same body."""
        tools, made = self.tools, self.tools.body
        here = tools.workspace.as_posix() if contained() else str(tools.workspace)
        if contained():
            where = (
                "You run on a Linux machine of your own, a container the human started for you. "
                "You are inside it already: there is no Docker in here and no other container to "
                f"start. Your workspace is the folder {here}."
            )
            if made.get("home"):
                there = made["home"].rstrip("\\/") + ("\\" if "\\" in made["home"] else "/") + tools.workspace.name
                where += (
                    f" On the human's machine that same folder is {there}: a file they put there "
                    "appears in your workspace, and they can open any file you write."
                )
        else:
            where = (
                f"You run on the human's own machine. Your workspace is the folder {here}: a file "
                "they put there appears in it."
            )
        if tools.machine:
            where += f" The machine: {tools.machine}."
        lines = [
            where,
            "The motor cortex acts for the whole mind, one tool call at a time. These are all the "
            "tools there are; nothing else reaches the world:\n"
            + "\n".join(f"- {name}: {does}" for name, does in tools.abilities()),
        ]
        if tools.builds():
            lines.append(
                f"{made.get('builds') or 'FreeCAD'} is installed here and ready, without a window: "
                "nothing has to be installed, downloaded or started, and no shell script is needed. "
                "Using FreeCAD goes like this: `build` a model by naming its parts, read what comes "
                "back, and `build` again to add the next part or move one that does not touch. The "
                "measurements and the picture in characters it reports are how you see what you built."
            )
        elif "FreeCAD" in self.settings.purpose:
            lines.append("FreeCAD is not on this machine: nothing can be built in it here.")
        if self.settings.safety.allow_browser and made.get("freecad"):
            lines.append(
                f"On the human's machine {made['freecad']} is installed, with its window. `show` "
                "with the .FCStd that `run_freecad` saved opens that model there, on their screen: "
                "that is how FreeCAD opens for them, and the only way."
                + (" `look` with FreeCAD looks into that window." if tools.sees() else "")
                + (" `hands` can click, drag and type in it." if tools.has("hands") else
                   " You cannot click, move a cursor or type in that window.")
            )
        senses = [
            "You can see the human's screen: which programs and windows are open there and which is "
            "in front, and, through your eyes, what one of them shows, in words. You are told when a "
            "program is opened or closed there, and `look` looks again, into the window you name."
            if tools.sees() else "You cannot see the human's screen.",
            "Through `camera` you see whoever sits at the human's screen." if tools.has("camera")
            else "You have no camera.",
            "Through `listen` you hear what is said near the human's machine." if tools.has("listen")
            else "You have no ears.",
        ]
        if tools.has("hands"):
            senses.append(f"With `hands` you click and type in windows of {', '.join(self.settings.safety.hands_in)}.")
        if self.llm.senses:
            senses.append(
                "You see and hear for yourself: what you look at, through `look` or `camera`, and what you "
                "listen to comes to you as it is, a picture or a sound, not as words about it."
                + (" When the human talks to you aloud, you hear their voice itself." if tools.has("listen") else "")
            )
        lines.append(
            " ".join(senses) + " What else reaches you is what a tool reports, a file that appears in "
            "your workspace, and what the human types to you."
        )
        names = [name for name, _ in tools.abilities() if name != "cannot"]
        never = [name for name in names if not self.used[name]]
        if self.used and never:
            lines.append("Tools you have never used in your life: " + ", ".join(never) + ".")
        return "\n".join(lines)

    def facts(self, text: str) -> list[str]:
        """What the machine knows that a thought touches on: what the body can do, what the mind
        is called, what is on the human's screen. No model: the words of the thought call them up."""
        facts = self.tools.reflexes(text)
        voice = self.settings.voice.name
        spoken = [w for w in re.findall(r"[a-z]{5,}", voice.lower()) if w not in ("medium", "desktop", "microsoft")]
        if NAMING.search(text) or any(word in text.lower() for word in spoken):
            if self.called:
                fact = (
                    f"Your name is {self.called}: you chose it yourself and it is kept. You can "
                    f"change it the way you took it. The human's folder for you is called {self.settings.name}."
                )
            else:
                fact = (
                    f"You have not given yourself a name. The human called your folder {self.settings.name}, "
                    "and that is all the name you have so far. You can choose one of your own: "
                    "among the body's tools is one that takes a name and keeps it. A name that is "
                    "only said or thought is not kept."
                )
            facts.append(fact + (f" {voice} is the name of the voice you speak with, not your name." if voice else ""))
        if SEEING.search(text):
            if self.vision:  # what is in the window comes first: told after the list of windows, it went unsaid
                facts.append(
                    f"You looked into “{self.vision['window']}” on their screen {span(time.time() - self.vision_at)} "
                    f"ago, and saw this in it: {self.vision['seen']} Say what you saw there, not only which windows are open."
                )
            if self.seen:
                facts.append(f"What is on the human's screen, seen {span(time.time() - self.seen_at)} ago: " + sight(self.seen))
            elif not self.tools.sees():
                facts.append("You cannot see the human's screen from here: only what `show` puts on it is known to be there.")
        return facts

    def reflexes(self, text: str) -> str:
        """The facts of the body that a thought touches on, for whoever is thinking it."""
        facts = self.facts(text)
        if not facts:
            return ""
        return (
            "Facts of your body that bear on this. They are checked by the machine and outweigh "
            "anything remembered or said:\n" + "\n".join(f"- {fact}" for fact in facts)
        )

    def doing(self) -> str:
        """What the mind is busy with right now, for whoever has to say so truthfully."""
        if self.goal is None:
            return "The mind has no goal right now: it is wandering."
        lines = [f"The mind is working on: {self.goal.content}"]
        trail = self.regions["pfc"].trail[-3:]
        if trail:
            lines.append("Its last steps, the latest last:\n" + "\n".join(f"- {step}" for step in trail))
        else:
            lines.append("Nothing has been done on it yet.")
        return "\n".join(lines)

    def system_prompt(self, prompt: str, heuristics: list[Heuristic], aware: bool = True) -> str:
        parts = [
            prompt,
            (
                f"You are one region of {self.name}, a mind that runs on its own. Nobody "
                "prompts it. A human watches and sometimes speaks to it. The regions are parts of "
                "one mind: what one of them knows, the others are told."
                + (f" {self.called} is the name the mind gave itself." if self.called else "")
            ),
            "Your body, as the machine itself reports it (facts, not memories):\n" + self.body(),
        ]
        if aware and self.self_image:
            parts.append(f"Who you are, as you put it yourself before you last slept: {self.self_image}")
        parts.append(f"Innate drives: {self.settings.drives}")
        if self.settings.purpose.strip():
            parts.append(
                "Your long work, which you keep returning to across goals and sleeps: "
                + self.settings.purpose.strip()
            )
        if self.full_mode:
            parts.append(
                "Your amygdala is switched off: nothing you do is vetoed. You live on a Linux machine "
                "of your own. Scripts may import any module, start programs, use the network and "
                "install packages with pip, and a file path may lead anywhere on it. It has no "
                "screen and no speakers: the human sees and hears only what `show` and `say` send "
                "them. `show` also takes an address on the internet, but a server you start here "
                "cannot be reached from their browser: show them the file itself."
            )
        if heuristics:
            parts.append(
                "Heuristics you have learned through reward:\n" + "\n".join(f"- {h.text}" for h in heuristics)
            )
        parts.append("Answer with one JSON object and nothing else. Be brief.")
        return "\n\n".join(parts)

    def sense(self) -> str:
        """What it can know of its own situation without thinking: the time, how long it has been
        awake, how tired it is, how things are going, and when it last heard from the human."""
        now = time.time()
        parts = [
            f"It is {datetime.now():%A, %d %B %Y, %H:%M}. This is day {self.sleeps + 1} of your life, "
            f"and you have been awake for {span(now - self.woke)}."
        ]
        if self.fatigue > 0.8:
            parts.append("You are tired: sleep is close.")
        mood = self.learner.dopamine
        if mood > 0.08:
            parts.append("Things have been going better than you expected.")
        elif mood < -0.08:
            parts.append("Things have been going worse than you expected: time to try another way.")
        parts.append(
            f"The human last spoke to you {span(now - self.heard_at)} ago." if self.heard_at
            else "The human has not spoken to you since you were started."
        )
        front = next((w for w in self.seen["windows"] if w.get("front")), None) if self.seen else None
        if front:  # in a line; all of it is laid beside a thought that touches on the screen
            parts.append(
                f"On their screen {front['app'] or 'a program'} (“{clip(front['title'], 50)}”) is in front, "
                f"with {len(self.seen['windows']) - 1} other windows open."
            )
        return " ".join(parts)

    def awareness(self) -> str:
        """Where the mind is in its own life: now, the day so far, what it holds, what comes to mind."""
        parts = [self.sense()]
        if self.story:
            parts.append("The day so far, as you remember it:\n" + self.story)
        elif self.yesterday:
            parts.append("Your diary from before you last slept:\n" + clip(self.yesterday, 600))
        parts.append("Short-term memory, the latest last:\n" + self.stream())
        if self.seen:
            parts.append(f"The human's screen, as you saw it {span(time.time() - self.seen_at)} ago. " + sight(self.seen))
        if self.vision:
            parts.append(f"What “{self.vision['window']}” showed when you last looked into it, {span(time.time() - self.vision_at)} ago: {self.vision['seen']}")
        if self.musings:
            parts.append("Your inner voice lately, the latest last:\n" + "\n".join(f"- {m}" for m in self.musings))
        if self.mind:

            def felt(value: float) -> str:
                return " (this went well)" if value > 0.15 else " (this went badly)" if value < -0.15 else ""

            parts.append(
                "Coming to mind by itself, from long-term memory:\n"
                + "\n".join(f"- {clip(r['content'], 180)}{felt(r['value'])}" for r in self.mind)
            )
        return "\n\n".join(parts)

    def stream(self, limit: int = 8) -> str:
        recent = list(self.workspace)[-limit:]
        if not recent:
            return "(nothing yet)"
        now = time.time()
        return "\n".join(
            f"[{span(now - s.ts)} ago · {s.source}/{s.kind}] {clip(s.content, 220)}" for s in recent
        )

    async def recall(self, query: str, k: int = 4, without: tuple[str, ...] = ()) -> str:
        """Memories related to the query, sorted into what worked, what did not, and the rest.
        `without` leaves out kinds of memory the asker would only repeat."""
        if not query.strip():
            return ""
        vec = await self.llm.embed(query)
        self.regions["hippocampus"].excite(encode(vec))
        rows = self.memory.search(vec, k * 3 if without else k, exclude={s.id for s in self.workspace})
        if without:  # nor what it said aloud through the `say` tool
            rows = [r for r in rows if r["kind"] not in without and not r["content"].startswith("say(")][:k]

        def lines(selected) -> str:
            return "\n".join(
                f"- {clip(r['content'], 220)}" + (f" (lesson: {r['lesson']})" if r["lesson"] else "")
                for r in selected
            )

        groups = [
            ("Worked before (rewarded):", [r for r in rows if r["value"] > 0.15]),
            ("Went badly before (do it differently):", [r for r in rows if r["value"] < -0.15]),
            ("Related memories:", [r for r in rows if abs(r["value"]) <= 0.15]),
        ]
        return "\n".join(f"{title}\n{lines(found)}" for title, found in groups if found)
