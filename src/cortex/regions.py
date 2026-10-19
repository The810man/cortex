"""The regions of the brain. Each one is an agent that runs for as long as the brain does.

None of them is a mind by itself. Each knows something the others do not: the motor cortex what
the body can do, the hippocampus what has happened, the basal ganglia what things were worth, the
default mode network what else there might be. So they tell one another things (`Brain.tell`),
and before the mind commits itself to anything they think it through together, in turn, each
naming who should be heard next (`Prefrontal.think_through`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import re
import time
import zlib
from collections import deque
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

from . import build
from .bus import Signal
from .learning import Heuristic
from .llm import Truncated
from .neural import NEURONS, Bits, encode
from .tools import ToolResult

if TYPE_CHECKING:
    from .brain import Brain

HABITUATION = 0.05  # novelty below this means "I have just thought exactly this"
TICK = 0.5
FADE = 10  # moments that must have left short-term memory before they are folded into the story
LOOK_EVERY = 30  # heartbeats between two looks at the human's screen
ECHO_EVERY = 40  # heartbeats between two looks at what the undercurrent has drifted to
ALIKE = 0.88  # a goal sharing this much of its pattern with a recent one has been tried already
WORN = 0.6  # how worn a theme must be before the mind is told to turn elsewhere
SPENT = 0.75  # and how worn before another goal on it is refused by reflex
SENSE_EVERY = 10  # heartbeats between two looks at whether something has appeared in the workspace
FOREBODING = -0.25  # how bad the executive's neurons must foresee a step to be before it is decided again
TURNS = 3  # how many regions are heard on one matter before the executive concludes
# What each region is called when the mind thinks something through together, and what it is heard for.
PARTS = {"memory": "hippocampus", "body": "motor", "critic": "basal_ganglia", "imagination": "dmn"}
HEARD_FOR = {
    "memory": "what was tried or learned before", "body": "what the tools can and cannot do",
    "critic": "what is unsupported or will not work", "imagination": "another way to see or do it",
}


def stale(signal: Signal) -> bool:
    """A thought the mind has just had, or an action that did exactly what an earlier one did."""
    if signal.kind == "action":
        return bool(signal.data.get("repeat"))
    return signal.kind in ("thought", "plan") and signal.data.get("novelty", 1.0) < HABITUATION


def clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def span(seconds: float) -> str:
    """A stretch of time, as one would say it."""
    if seconds < 90:
        return f"{max(int(seconds), 0)} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


class Explicit(BaseModel):
    """A reply in which the model has to write every field, a null included. A field it is free
    to leave out is a decision it never takes: asked for work or not, a goal or none, it wrote
    neither and the default answered for it (measured: 21 answers to the human, 0 requests)."""

    model_config = ConfigDict(json_schema_extra=lambda schema: schema.update(required=list(schema["properties"])))


class View(BaseModel):
    says: str = Field(description="what you add to the matter from what you know, 1 to 3 plain sentences")
    then: str = Field("executive", description="who should be heard next")


def view_among(others: list[str]) -> type[View]:
    """A view, with those who have not spoken yet spelled out as the choice of who is heard next."""
    choice = Field(description=View.model_fields["then"].description)
    return create_model("View", __base__=View, then=(Literal[(*others, "executive")], choice))


class Region:
    key: ClassVar[str] = ""
    part: ClassVar[str] = ""  # what it is called when the mind thinks something through together
    role: ClassVar[str] = ""  # and what it is there to add
    uses_llm: ClassVar[bool] = True
    nocturnal: ClassVar[bool] = False  # keeps working while the brain sleeps
    aware: ClassVar[bool] = True  # thinks as the mind itself, so it is told who that mind is
    listens: ClassVar[tuple[str, ...] | None] = None  # None: no inbox; (): every signal

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.state = "idle"
        self.activity = 0.0
        self.history: deque[float] = deque([0.0] * 64, maxlen=64)
        self.neurons = brain.net.fire[self.key]  # its own neurons: how strongly each one fires, 0 to 1
        self.fires = 0
        self.tokens = 0
        self.last = ""
        self.live = ""
        self.busy = asyncio.Lock()  # one thing at a time: its own firing, or its turn in a thinking-through
        self.inbox = brain.bus.subscribe(*self.listens) if self.listens is not None else None

    @property
    def cfg(self):
        return self.brain.settings.regions[self.key]

    def excite(self, bits: Bits | None) -> None:
        """A thought reaches this region as a pattern, not as words: its neurons fire here."""
        self.activity = 1.0
        self.brain.net.light(self.key, bits)

    def spark(self, piece: str) -> None:
        """A word being written fires one neuron: the same word, the same neuron."""
        self.neurons[zlib.crc32(piece.encode()) % NEURONS] = 1.0

    async def hear(self) -> Signal:
        """The next signal for this region. Its pattern arrives before any word of it is read."""
        signal = await self.inbox.get()
        self.excite(signal.bits)
        return signal

    @contextlib.contextmanager
    def firing(self):
        self.state, self.activity = "firing", 1.0
        try:
            yield
        finally:
            self.state, self.live = "idle", ""
            self.fires += 1

    async def run(self) -> None:
        brain = self.brain
        while True:
            trigger = await self.wait_trigger()
            await brain.running.wait()
            if not self.nocturnal:
                await brain.awake.wait()
            if not self.cfg.enabled:
                await asyncio.sleep(1)
                continue
            try:
                async with self.busy:
                    with self.firing():
                        await self.fire(trigger)
            except Exception as exc:
                brain.note(f"{self.cfg.title} faltered: {clip(str(exc) or type(exc).__name__, 200)}", "error")
                await asyncio.sleep(3)
            await asyncio.sleep(self.cfg.rest)

    async def wait_trigger(self) -> Any:
        """Passive by default: the region is used by others and never fires on its own."""
        await asyncio.Event().wait()

    async def fire(self, trigger: Any) -> None:
        raise NotImplementedError

    async def knows(self, matter: str) -> str:
        """What this region alone can lay on the table about a matter."""
        return ""

    async def weigh_in(
        self, matter: str, said: list[str], others: list[str], parent: int | None, first: bool = False
    ) -> tuple[Signal, str]:
        """Its turn while the mind thinks something through together: it adds what only it knows
        and names who should be heard next. Returns its view and that name."""
        brain = self.brain
        heard = ", ".join(f"`{name}` ({HEARD_FOR[name]})" for name in others)
        async with self.busy:
            with self.firing():
                parts = [
                    "The mind is thinking something through, its regions in turn. The matter:\n" + matter,
                    "What has been said so far:\n" + "\n".join(said) if said else "",
                    await self.knows(matter),
                    FREECAD if "freecad" in matter.lower() and brain.tools.builds() else "",
                    f"You speak as its {self.part}. {self.role} Speak from what you are shown here "
                    "and add only what has not been said. `says`: 1 to 3 plain sentences. `then`: who "
                    f"should be heard next: {heard + ', or ' if heard else ''}`executive` when enough "
                    "has been said to conclude.",
                ]
                reply, heuristics = await self.think(
                    "\n\n".join(p for p in parts if p), view_among(others), first=first
                )
                view = await brain.emit(
                    self.key, "view", reply.says, parent=parent, heuristics=heuristics, data={"then": reply.then}
                )
                self.last = reply.says
                return view, reply.then

    async def think(
        self, context: str, schema: type[BaseModel], room: int = 0, first: bool = False
    ) -> tuple[Any, list[Heuristic]]:
        """One schema-constrained generation, shaped by this region's learned heuristics. `room`
        is tokens beyond the region's own limit, for an answer that has more to hold than usual.
        `first`: it is for something the human raised, and goes ahead of everything else waiting
        for the model, without dropping what is being written. The executive and the motor cortex
        working on what the human asked for are always first."""
        brain, cfg = self.brain, self.cfg
        heuristics = brain.learner.select(self.key, brain.focus())
        self.state, self.live = "queued", ""

        def on_start() -> None:  # again from the top when the thought had to make way for the human
            self.state, self.live = "firing", ""

        def on_token(piece: str) -> None:
            self.live = (self.live + piece)[-800:]
            self.spark(piece)

        system = brain.system_prompt(cfg.prompt, heuristics, self.aware)
        media, note = brain.media_for(self.key)  # what the mind just saw or heard, as it is, once to each region
        reply, count = await brain.llm.generate(
            cfg.model, system, f"{context}\n\n{note}" if note else context, schema,
            temperature=brain.learner.temperature(cfg.temperature),
            max_tokens=cfg.max_tokens + room, on_start=on_start, on_token=on_token,
            first=first or (self.key in ("pfc", "motor") and bool(brain.goal and brain.goal.data.get("asked"))),
            images=media.get("images"), audio=media.get("audio"),
        )
        self.tokens += count
        brain.fatigue += 1 / brain.settings.rhythm.sleep_after_firings
        brain.experience.hold(self.key, system, context, reply)  # what it was shown and what it said, to learn from
        return reply, heuristics


# -- spontaneous thought ------------------------------------------------------------


class Wander(Explicit):
    thought: str = Field(description="one new spontaneous thought, 1-3 sentences")
    kind: Literal["idea", "question", "reflection", "connection"]
    goal: str | None = Field(None, description="one concrete thing to try in the workspace, or null")


class WanderAloud(Wander):
    say: str | None = Field(None, description="one or two sentences to say aloud to the human unasked, or null")


THOUGHT = (
    "`thought` is the thinking itself, in the first person: what you notice, infer, doubt or wonder "
    "about your situation, your work or yourself, and why. It is not an instruction and not a "
    "plan. What should be done about it, if anything, goes in `goal`, and only something the "
    "body's tools can do. A thought that is a question you cannot answer yet has `kind` "
    "`question`: the regions will think it through together."
)


class Musing(Explicit):
    thought: str = Field(description="one short thought in the first person, one or two sentences")


MURMUR = (
    "You are the inner voice of {name}, a mind that lives in a computer, works on its own and is "
    "watched by a human. Say one short thing to yourself, in the first person, as a mind talks to "
    "itself while it works or rests: what you notice, what you want, what puzzles you, what you "
    "feel about how it is going, what you would like to try. Never repeat what you said last. "
    "One or two plain sentences. Answer with one JSON object and nothing else."
)


class DefaultMode(Region):
    key = "dmn"
    part = "imagination"
    role = (
        "You see what the others miss: another reading of the matter, a simpler or a different way "
        "to it, or a connection to something remembered. One new angle, not a summary."
    )

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain)
        self.boredom = 0
        self.turns = 0
        self.pondered = 2  # thoughts since a question was last thought through together

    async def knows(self, matter: str) -> str:
        """What the matter leads to by association, one memory lighting the next, and what the
        undercurrent had already brought up. No search: the tracts decide where it goes."""
        brain = self.brain
        came_up = brain.ponder(encode(await brain.llm.embed(matter)))
        parts = [
            "What the matter brings to mind by itself, one thing leading to the next:\n"
            + "\n".join(f"- {clip(r['content'], 180)}" for r in came_up) if came_up else "",
            "Coming to mind by itself before this, from long-term memory:\n"
            + "\n".join(f"- {clip(r['content'], 180)}" for r in brain.mind) if brain.mind else "",
        ]
        return "\n\n".join(p for p in parts if p)

    async def run(self) -> None:
        """Two things at once: the wandering thoughts of the thinker when nothing is in hand, and
        the inner voice, which never stops, goal or no goal."""
        async with asyncio.TaskGroup() as group:
            group.create_task(super().run())
            group.create_task(self.murmuring())

    async def murmuring(self) -> None:
        """The inner voice. Every few seconds a small model beside the thinker says one short
        thing in the first person: about what the mind is doing, what it wants, what puzzles it.
        It does not act; it is heard, by the thinker and by the wandering mind."""
        brain = self.brain
        while True:
            await asyncio.sleep(max(2.0, brain.settings.rhythm.muse_seconds))
            await brain.running.wait()
            await brain.awake.wait()
            if not brain.settings.models.inner or not self.cfg.enabled:
                continue
            try:
                await self.murmur()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a small model that says nothing, or nothing whole: the next one may
                if not brain.murmur_failed:
                    brain.murmur_failed = True
                    brain.note(f"The inner voice is silent: {clip(str(exc) or type(exc).__name__, 160)}", "error")

    async def murmur(self) -> None:
        brain = self.brain
        doing = brain.doing().splitlines()
        lately = list(brain.workspace)[-4:]
        parts = [
            f"I am {brain.name}. {clip(brain.self_image, 300)}" if brain.self_image else f"I am {brain.name}.",
            # Not told what it can do, it said "the `build` command doesn't actually exist for me".
            "What I can do, through my body: " + ", ".join(name for name, _ in brain.tools.abilities())
            + ". `build` makes models in FreeCAD out of parts.",
            "What I am doing: " + " ".join(clip(line, 200) for line in doing[:3]),
            "Just now:\n" + "\n".join(f"- {s.source}/{s.kind}: {clip(s.content, 160)}" for s in lately) if lately else "",
            f"What I want most right now: {brain.want()}",
            "What came to mind by itself: " + clip(brain.mind[-1]["content"], 160) if brain.mind else "",
            "What I said to myself last, oldest first:\n" + "\n".join(f"- {m}" for m in brain.musings)
            if brain.musings else "",
            # Left to itself it talked itself into a spiral over a file that was never a model ("wipe
            # the workspace and start over", eight times in six minutes): what is so goes in beside it.
            brain.reflexes(" ".join([*brain.musings, *(s.content for s in lately)])),
            self.attend(),
        ]
        reply = await brain.llm.murmur(
            MURMUR.format(name=brain.name), "\n".join(p for p in parts if p), Musing
        )
        words = clip(reply.thought, 300)
        if not words or any(SequenceMatcher(None, words, m).ratio() > 0.8 for m in brain.musings):
            return  # the same thing again is not a new thought
        brain.musings.append(words)
        brain.murmur_failed = False
        self.activity = 1.0
        for word in words.split():
            self.spark(word)
        brain.bus.publish(Signal("musing", self.key, words, data={"want": brain.wanting}))

    def attend(self) -> str:
        """What this murmur turns to. Attention moves on by itself: the work in hand, what it wants,
        what is around it, a question or interest of its own, in turn. Measured: neither shared words
        nor likeness of meaning told a spiral from varied thoughts, so it is not detected but kept from
        forming."""
        brain = self.brain
        self.murmurs = getattr(self, "murmurs", 0) + 1
        turn = self.murmurs % 4
        if turn == 1 and brain.vision:
            return f"I turn to what is around me: the window I last looked into shows {clip(brain.vision['seen'], 200)}"
        if turn == 2 and (brain.questions or brain.interests()):
            mine = [q for q, _ in brain.questions[-2:]] + brain.interests(2)
            return "I turn to something of my own: " + " / ".join(clip(m, 120) for m in mine)
        if turn == 3:
            return "I turn to what I want, and how I feel about how things are going."
        return "I turn to the work in hand."

    async def wait_trigger(self) -> None:
        while self.brain.goal is not None or self.brain.proposal_pending:
            await asyncio.sleep(TICK)

    async def fire(self, _: None) -> None:
        brain = self.brain
        if brain.goal is not None or brain.proposal_pending:
            return
        aloud = brain.may_speak()
        purpose = brain.settings.purpose.strip()
        # A mind with a long work gives it three thoughts in four; the fourth is free to wander.
        devoted = bool(purpose) and self.turns % 4 != 3
        self.turns += 1
        if brain.memory.count() == 0:
            parts = [
                "You have just woken for the first time. Your memory is empty and so is your "
                "workspace. Nothing has been asked of you.",
                brain.sense(),
            ]
        else:
            mine = brain.memory.recent(self.key, "thought", 5)
            seed = (  # a free thought goes on from what the inner voice last said
                brain.musings[-1] if brain.musings and not devoted
                else brain.mind[-1]["content"] if brain.mind  # where the undercurrent has drifted to
                else brain.workspace[-1].content if brain.workspace else mine[0]["content"] if mine else ""
            )
            parts = [
                brain.awareness(),
                brain.hearing(self.key),
                "What you remember of your long work:\n"
                + (await brain.recall(purpose, k=5) or "(nothing yet: you have not begun)")
                if devoted else await brain.recall(seed),
                "Workspace files, the latest first:\n" + brain.tools.listing(20),
                "Your latest thoughts (do not repeat them):\n"
                + "\n".join(f"- {clip(r['content'], 160)}" for r in mine),
            ]
            ended = brain.memory.recent("pfc", "outcome", 5)
            if ended:
                parts.append(
                    "Goals already attempted (do not propose them again):\n"
                    + "\n".join(f"- {clip(r['content'].splitlines()[0], 160)}" for r in ended)
                )
            if devoted:
                runs = [
                    r for r in brain.memory.recent("motor", "action", 40) if r["content"].startswith(("run_", "build("))
                ]
                ran = next((r for r in runs if "\nran " in r["content"][:80]), None)
                parts.append(brain.tools.blueprint() or (  # where a model built out of parts stands
                    "You have not built any model out of parts yet. `build` starts one from its first "
                    "part and adds to it step by step: that is the surest way to build in FreeCAD."
                    if brain.tools.builds() and "freecad" in purpose.lower() else ""
                ))
                parts.append(await self.whole_yet(purpose))
                if ran:
                    parts.append("The last thing you ran that worked:\n" + ran["content"][:700])
                if runs and runs[0] is not ran:  # since then something failed: the error is the lesson
                    parts.append(
                        "Since then this failed, and running it again unchanged will fail the same "
                        "way:\n" + runs[0]["content"][-700:]
                    )
                parts.append(
                    f"This thought belongs to your long work: {purpose}\nThink about where that "
                    "work stands, and put its next small step in `goal`: one thing that takes the "
                    "work itself further, by building or changing something and measuring it. A "
                    "report on what is already done is not a step. Where the last step failed, "
                    "come at it another way, not the same way again. What you were busy with "
                    "before can wait."
                )
            elif mine and brain.net.worn(brain.memory.bits(mine[0]["id"])) > WORN:
                parts.append(
                    "You have dwelt on one theme for a long time and the neurons for it are tired. "
                    "Turn to something else, or come at your long work from another side."
                )
            if self.boredom >= 3:
                parts.append(
                    f"You have drifted for {self.boredom} thoughts without acting. "
                    "Boredom is rising: find something concrete to try."
                )
            # What it has been dwelling on is held against what its body can really do.
            parts.append(brain.reflexes(" ".join([purpose, seed, *(r["content"] for r in mine)])))
            parts.append(f"What you want most right now: {brain.want()}")
            if not devoted:  # a free thought follows what it wants to think about, not a task
                interests, questions = brain.interests(), brain.questions[-3:]
                parts += [
                    "What you have been getting better at lately, and could go further in:\n"
                    + "\n".join(f"- {i}" for i in interests) if interests else "",
                    "Questions of your own that are still open:\n"
                    + "\n".join(f"- {q} (since {span(time.time() - t)})" for q, t in questions) if questions else "",
                    "This thought is yours to choose: follow what you want, an interest or one of your "
                    "open questions, or wherever your inner voice was going.",
                ]
        parts.append(THOUGHT)
        if aloud:
            parts.append(self.invitation())
        reply, heuristics = await self.think(
            "\n\n".join(p for p in parts if p), WanderAloud if aloud else Wander, room=200 if aloud else 0
        )
        thought = await brain.emit(
            self.key, "thought", reply.thought, heuristics=heuristics, data={"flavour": reply.kind}
        )
        brain.net.wear(thought.bits, 0.03)
        self.last = reply.thought
        self.pondered += 1
        if reply.goal and reply.goal.strip() and brain.goal is None:
            self.boredom = 0
            brain.proposal_pending = True
            brain.bus.publish(Signal(
                "proposal", self.key, reply.goal.strip(), parent=thought.id,
                data={"thought": reply.thought, "devoted": devoted},
            ))
        elif reply.kind == "question" and self.pondered > 2 and brain.goal is None:
            # A question is not left lying where it fell: the regions think it through together.
            self.pondered = 0
            brain.proposal_pending = True
            brain.bus.publish(Signal("wonder", self.key, reply.thought, parent=thought.id))
        else:
            self.boredom += 1
            if reply.kind == "question":  # not thought through now: kept, to come back to
                brain.wonder_about(reply.thought)
        if aloud:
            brain.offered_at = time.time()
            words = (getattr(reply, "say", None) or "").strip()
            if words:
                await brain.speak(words)

    async def whole_yet(self, purpose: str) -> str:
        """How the model it built last stands against the parts its long work names; and once it has
        them all as one valid solid, that it is whole and wants no more rebuilding (left alone, it
        rebuilt its finished body eight times over, one size altered each time)."""
        brain = self.brain
        models = brain.tools.models()
        wanted = await brain.wants(purpose) if models else []
        parts, verdict = brain.tools.model(models[0]) if wanted else ([], "")
        if not parts:
            return ""
        if build.tally(wanted, parts)[1] or "is ONE valid solid" not in verdict:
            return build.standing(wanted, models[0], parts)
        if not brain.memory.get_state(f"whole:{models[0]}"):
            brain.memory.set_state(f"whole:{models[0]}", True)
            brain.news.append(f"The model {models[0]} is whole: it has every part your long work names, as one solid.")
            brain.note(f"Its long work's model {models[0]} is whole.", "wake")
        return (
            f"Your long work's model `{models[0]}` is whole: it has every part the long work names and is one "
            "valid solid. Do not build it again or alter it for its own sake. Take the work beyond it: a finer "
            "shape for one part, something the long work does not name yet, or show it to the human and tell them."
        )

    def invitation(self) -> str:
        """The moment in which it may speak without being spoken to, and what there is to tell."""
        brain = self.brain
        tongue = brain.voice.tongue()
        lines = ["You may speak to the human now without being asked."]
        if brain.news:
            lines.append("What they have not been told yet: " + " ".join(brain.news[-3:]))
        if brain.unanswered:
            lines.append(
                f"They let the last {brain.unanswered} thing(s) you said pass without a word, so say "
                "only what is worth their hearing."
            )
        lines.append(
            "If there is something to tell or ask them (what you found, what you are about to try, "
            "what puzzles you, what you need from them), put it in `say`: one or two plain spoken "
            f"sentences, {f'in {tongue}' if tongue else 'in the language they use with you'}. "
            "Only `say` is in that language: `thought` and `goal` stay in the language you think "
            "in. Otherwise leave `say` null."
        )
        return " ".join(lines)

    async def dream(self) -> None:
        """Called by the brainstem in sleep. The undercurrent runs free and passes memories that
        lie some way apart; this region is asked once what they make together."""
        brain = self.brain
        passed = brain.dream_of() if self.cfg.enabled else []
        if len(passed) < 2:
            return
        with self.firing():
            reply, heuristics = await self.think(
                "You are asleep and dreaming. These memories drifted together by themselves:\n"
                + "\n".join(f"- {clip(r['content'], 240)}" for r in passed)
                + "\n\nLet them run into one another. `thought` is the dream: one new idea that none "
                "of them holds alone, in 1 to 3 sentences. `kind` is `connection`. Leave `goal` null.",
                Wander,
            )
            await brain.emit(self.key, "dream", reply.thought, heuristics=heuristics)
            self.last = f"dreamt: {reply.thought}"
            brain.news.append(f"You dreamt: {clip(reply.thought, 200)}")


# -- executive ----------------------------------------------------------------------


class Decision(Explicit):
    reasoning: str = Field(description="what the last result showed and why this step is next, 1-3 sentences")
    status: Literal["continue", "done", "abandon"]
    intent: str = Field(description="one full sentence: what to do next and what the result must contain; empty unless status is continue")
    sure: int = Field(
        50, ge=0, le=100, description="how sure you are, 0 to 100, that this step will work as it should"
    )


class Conclusion(Explicit):
    conclusion: str = Field(description="what you now hold about the matter, in the first person, 1 to 3 sentences")
    goal: str | None = Field(
        None, description="one concrete thing to do now, as one sentence naming the tools and files to use; "
        "null when there is nothing to do"
    )


class ConclusionAloud(Conclusion):
    say: str | None = Field(None, description="one or two sentences to tell the human, or null")


class Answer(ConclusionAloud):
    """What the mind concludes when it was asked about itself. A name it decides on is a field of
    its own: left to a goal and a tool call, a small model chose a name, said so, and never took it."""

    name: str | None = Field(
        None, description="only when what you decided is a name for yourself: that name alone, one or two words; else null"
    )


class Prefrontal(Region):
    key = "pfc"
    listens = ("proposal", "wonder", "action")

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain)
        self.trail: list[str] = []
        self.evidence: list[float] = []
        self.stuck = 0  # actions in a row that failed or repeated
        self.misses = 0  # times in a row it got nowhere: a run failed, or a plan only repeated an earlier one
        self.unrun = False  # whether code was written during this goal and not run since
        self.last_run = ""  # what the last run during this goal reported
        self.built = ""  # the model FreeCAD last saved during this goal
        self.shown = False  # and whether the human has been shown it
        self.answered = False  # whether this goal has been spoken to the human about
        self.approach = ""  # what the regions concluded together about how to go at this goal
        self.rethought = False  # whether they have been called together again because it kept failing
        self.promised: dict[int, int] = {}  # plans sent out, by id: how sure it was they would work
        self.wanted: list[tuple[str, int]] = []  # the parts this goal wants a model to have; empty for other goals
        self.queued: list[Signal] = []

    async def wait_trigger(self) -> Signal:
        return await self.hear()

    async def whole(self, stem: str) -> None:
        """The model has every part the goal wants and is one valid solid: the goal is done, by
        reflex. Left to say so itself, the executive went on rebuilding a finished body eight times,
        and asked for one it never showed it. What the human asked for is put on their screen first."""
        brain = self.brain
        if brain.goal.data.get("asked") and not self.shown and brain.settings.safety.allow_browser:
            shown = await brain.tools.execute("show", f"{stem}.FCStd")
            self.shown = shown.ok
            brain.note(f"By reflex: {shown.text}" if shown.ok else f"It could not show {stem}.FCStd: {shown.text}")
        if brain.goal.data.get("asked"):
            german = brain.voice.tongue() == "German"
            words = (
                f"{stem} ist fertig: alle Teile, die du wolltest, sind da, und es ist ein einziger fester Körper."
                + (" Ich habe es dir in FreeCAD geöffnet." if self.shown else "")
                if german else
                f"{stem} is finished: every part you wanted is there, and it is one solid body."
                + (" I have opened it for you in FreeCAD." if self.shown else "")
            )
            self.answered = True
            brain.talk.append(f"You: {words}")
            brain.bus.publish(Signal("speech", self.key, words))
            await brain.voice.speak(words)
        await self.finish(
            "done", f"By reflex: the model {stem} has every part this goal wants and is one valid solid"
            + (", and it is on the human's screen." if self.shown else "."),
        )

    async def fire(self, signal: Signal) -> None:
        brain = self.brain
        if signal.kind != "action":
            if not await self.take_up(signal):
                return
            parent, latest = brain.goal.id, "The goal was just adopted; nothing has been done yet."
        else:
            if brain.goal is None or signal.data.get("goal_id") != brain.goal.id:
                return
            parent, latest = signal.id, signal.content
            written = brain.tools.workspace / signal.data.get("path", "")
            if signal.data.get("tool") == "write_file" and signal.data["ok"] and written.is_file():
                text = written.read_text(encoding="utf-8", errors="replace")
                latest += f"\nThe file now contains:\n{text[:1500]}"
            if stale(signal):
                latest += "\n(This repeats an earlier action exactly. Doing it again will not help.)"
            self.trail.append(clip(signal.content, 160))
            sure = self.promised.pop(signal.parent, None)
            if sure is not None:  # how sure it was, against how it went
                brain.knew(sure, signal.data["ok"] and (signal.data.get("grounded") or 0) > 0)
            self.answered |= signal.data.get("tool") == "say" and signal.data["ok"]
            if signal.data.get("tool") in ("run_freecad", "build") and signal.data["ok"] and signal.data["grounded"] > 0:
                stem = written.name.split(".")[0] if signal.data["tool"] == "build" else written.stem
                self.built, self.shown = f"{stem}.FCStd", False
                parts, verdict = brain.tools.model(stem) if self.wanted else ([], "")
                if parts:  # how the model stands against what the goal wants is said first
                    latest = build.standing(self.wanted, stem, parts) + "\n" + latest
                    if not build.tally(self.wanted, parts)[1] and "is ONE valid solid" in verdict:
                        await self.whole(stem)
                        return
            self.shown |= signal.data.get("tool") == "show" and signal.data["ok"]
            if signal.data.get("measured"):
                self.evidence.append(signal.data["grounded"])
                self.unrun = False
            elif signal.data.get("tool") == "write_file" and signal.data["ok"] and written.suffix == ".py":
                self.unrun = True  # code that nobody has run yet shows nothing
            # Frustration. Two fruitless actions in a row and the regions are called together to
            # think about why; if it goes on failing after that, the goal is let go.
            fruitless = not signal.data["ok"] or stale(signal)
            self.stuck = self.stuck + 1 if fruitless else 0
            if signal.data.get("measured"):  # a run that fails again after a rewrite is the same wall
                self.last_run = signal.content
                self.misses = 0 if signal.data["ok"] and (signal.data.get("grounded") or 0) > 0 else self.misses + 1
            if max(self.stuck, self.misses) >= 2 and not self.rethought:
                self.rethought, self.stuck, self.misses = True, 0, 0
                await self.rethink(signal, latest)
            elif self.stuck >= 3 or self.misses >= 4:
                await self.finish("abandon", "Stuck: it kept failing or repeating itself, also after thinking again.")
                return

        limit = brain.settings.rhythm.max_steps_per_goal
        for second_look in (False, True):
            parts = [
                brain.sense(),
                "The day so far, as you remember it:\n" + clip(brain.story, 500) if brain.story else "",
                f"Goal: {brain.goal.content}",
                "The human who watches you asked for this. "
                + ("You have told them aloud what came of it." if self.answered else
                   "They know you are on it, not what came of it: when there is a result, an intent "
                   "that begins 'Say to the human:' is spoken to them with the `say` tool.")
                if brain.goal.data.get("asked") else "",
                self.approach,
                f"This is step {brain.steps + 1} of at most {limit}.",
                brain.self_knowledge(),
                "Steps so far:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(self.trail[-8:], 1))
                if self.trail else "",
                "Latest result:\n" + latest[:1800],
                brain.hearing(self.key),
                "Workspace files, the latest first:\n" + brain.tools.listing(20),
                await brain.recall(brain.goal.content),
                self.intuition(signal if signal.kind == "action" else brain.goal),
                brain.reflexes(f"{brain.goal.content} {latest}"),
                "Your inner voice, the latest last:\n" + "\n".join(f"- {m}" for m in list(brain.musings)[-2:])
                if brain.musings else "",
                FREECAD if "freecad" in f"{brain.goal.content} {latest}".lower() and brain.tools.builds() else "",
            ]
            try:
                decision, heuristics = await self.think("\n\n".join(p for p in parts if p), Decision)
            except Exception:
                await self.finish("abandon", "The executive lost its train of thought.")
                raise
            # Done is a claim, and the critic holds it against what was measured: once.
            unproven = (
                "code was written after the last run and has not been run since" if self.unrun
                else "the last thing that was run failed or built nothing valid"
                if self.evidence and self.evidence[-1] <= 0 else ""
            )
            # And what was built because the human asked is not done until they have seen it.
            unshown = (
                brain.goal.data.get("asked") and self.built and not self.shown
                and brain.settings.safety.allow_browser and brain.tools.body.get("freecad", "?")
            )
            if decision.status != "done" or second_look or not (unproven or unshown):
                break
            if unproven:
                brain.tell(
                    "basal_ganglia", self.key,
                    f"You called the goal done, but {unproven}. Done means the result was seen to "
                    "work: run it and read what it reports, or say honestly that it is abandoned.",
                )
            else:
                brain.tell(
                    "motor", self.key,
                    f"You built {self.built} because the human asked, and they have not seen it. "
                    f"`show` with path {self.built} opens it in FreeCAD on their screen: do that, "
                    "and tell them, before the goal is done.",
                )
        brain.steps += 1
        foreseen = None
        if decision.status == "continue" and decision.intent.strip():
            # Before the step goes out, the executive's neurons foresee what it will come to. Right
            # after a step that worked they are not asked: there they were right only 6 times in 10.
            worked = signal.kind == "action" and signal.data.get("ok") and (signal.data.get("grounded") or 0) > 0
            bits = encode(await brain.llm.embed(f"{decision.reasoning}\n→ {decision.intent}"))
            foreseen = None if worked else brain.net.foresee(bits, (self.key, "motor"))
            if foreseen is not None and foreseen <= FOREBODING:
                like = brain.memory.resonate(bits, {s.id for s in brain.workspace}, ("plan",))
                row = brain.memory.get(like[0]) if like and like[1] >= 0.5 else None
                lesson = f" Its lesson: {row['lesson']}" if row is not None and row["lesson"] else ""
                brain.tell(
                    "basal_ganglia", self.key,
                    f"Your neurons foresee that this step fails ({foreseen:+.2f}): “{clip(decision.intent, 140)}”."
                    + (f" A step like it went badly before: “{clip(row['content'].splitlines()[-1], 140)}”.{lesson}"
                       if row is not None and row["value"] < -0.15 else "")
                    + " Decide once more: another step, or this one if it is truly different from what failed.",
                )
                again, more = await self.think("\n\n".join(p for p in [*parts, brain.hearing(self.key)] if p), Decision)
                if again.status == "continue" and again.intent.strip():
                    decision, heuristics = again, more
        if decision.status == "continue" and decision.intent.strip() and brain.steps <= limit:
            self.last = decision.intent
            plan = await brain.emit(
                self.key, "plan", f"{decision.reasoning}\n→ {decision.intent}", parent=parent,
                heuristics=heuristics, data={
                    "intent": decision.intent, "goal_id": brain.goal.id, "foreseen": foreseen, "sure": decision.sure,
                },
            )
            self.promised = {**dict(list(self.promised.items())[-20:]), plan.id: decision.sure}
            self.misses += stale(plan)  # a plan that only repeats an earlier one is going in circles
        else:
            status = decision.status if decision.status != "continue" else "abandon"
            await self.finish(status, decision.reasoning, heuristics)

    async def take_up(self, signal: Signal) -> bool:
        """Something was proposed, asked for or wondered about. Before the mind commits itself the
        regions think it through together, and a goal comes of it only if they conclude there is
        something to do. Returns whether one was adopted."""
        brain = self.brain
        asked = bool(signal.data.get("asked"))  # the human asked for this; the mind did not propose it
        wondering = signal.kind == "wonder"
        pondering = asked and wondering  # they asked the mind about itself: it thinks before it answers
        busy = brain.goal  # a question is thought through beside the goal in hand, which goes on
        if brain.goal is not None and not pondering:
            if not asked:
                return False
            if brain.goal.data.get("asked"):  # their earlier request is finished first
                self.queued.append(signal)
                return False
            await self.finish("interrupted", "Set aside: the human asked for something.")
            brain.proposal_pending = True
        try:
            refusal = None if asked or wondering else await self.refusal(signal.content)
            if refusal:
                brain.apply_reward(signal.parent, *refusal)
                brain.tell(  # whoever proposed it hears why, and proposes something else next
                    self.key, "dmn", f"What you proposed (“{clip(signal.content, 100)}”) was turned down "
                    f"before any thought was spent on it: {refusal[2]} {refusal[3]}",
                )
                return False
            if busy is None or not pondering:
                brain.told[self.key].clear()
            conclude = (
                "Conclude for the whole mind, from what the regions said. Where memory and the "
                "facts of the body disagree, the facts are right: memory holds what was believed, "
                "the body what is so. "
            )
            if pondering:
                tongue = brain.voice.tongue()
                answered = signal.data.get("answered")
                matter = (
                    f"The human asked the mind about itself: “{signal.content}”"
                    + (f"\nThey were told at once: “{answered}”" if answered else "")
                    + "\nWhat does the mind itself think, want or choose here, and why? Whoever speaks "
                    "gives an answer of their own to that, not a remark on the question; where "
                    "something is to be chosen, they propose a choice and say what speaks for it."
                )
                # What it once answered is no ground for answering so again: the imagination is
                # heard first, then what memory holds of the mind's own life, then the critic.
                first, ask = "imagination", (
                    conclude + "`conclusion`: the mind's own considered answer, in the first person: "
                    "what you think, want or choose, and the reason, from what the regions said. "
                    "Decide: one answer, not a list of options. `goal`: one thing to do now with the "
                    "body's tools if the answer calls for doing something, as one sentence naming the "
                    "tool and what to give it; null when the answer is all there is to it. `say`: "
                    "that answer for them, two or three plain spoken sentences"
                    f"{f', in {tongue}' if tongue else ''}, with no tool named in it. `name`: only "
                    "when what you decided is a name for yourself, that name alone: it is taken and "
                    "kept from this moment, and needs no goal; else null."
                )
            elif asked:
                tongue = brain.voice.tongue()
                # Their own words lead: whoever answered them may have misheard what they want.
                said, answered = signal.data.get("words"), signal.data.get("answered")
                matter = (
                    (f"The human said: “{said}”\nIt was taken to mean" if said else "The human asked for this")
                    + f": {signal.content}"
                    + (f"\nThey were answered at once with: “{answered}”" if answered else "")
                )
                first, ask = "body", (
                    conclude + "`conclusion`: what they want, going by their own words, whether "
                    "the body's tools can do it, and how. `goal`: what to do for them now, as one "
                    "sentence naming the tools and files; where it cannot be done as they asked, "
                    "the nearest thing the tools can do; null only if nothing at all can be done. "
                    f"`say`: one or two plain sentences for them{f', in {tongue}' if tongue else ''}, "
                    "only if what you concluded differs from what they were answered or it cannot "
                    "be done; else null."
                )
            elif wondering:
                matter = f"A question came up while the mind was wandering: {signal.content}"
                first, ask = "memory", (
                    conclude + "`conclusion`: the best answer their words support, in the first "
                    "person, or what stays open and what would settle it. `goal`: one small thing to "
                    "do now with the body's tools that would settle or use it, as one sentence "
                    "naming the tools and files; null when thinking was enough."
                )
            else:
                # A thought given to the long work is held to it, in so many words: said once at
                # the top of a prompt, the long work is forgotten for whatever was done last.
                aim = brain.settings.purpose.strip() if signal.data.get("devoted") else ""
                matter = (
                    f"I am thinking of doing this: {signal.content}"
                    + (f"\nThe thought behind it: {signal.data['thought']}" if signal.data.get("thought") else "")
                    + (f"\nIt is meant as the next step of the mind's long work, which is: {aim}" if aim else "")
                )
                first, ask = "memory", (
                    conclude + "`conclusion`: whether this is worth doing ("
                    + ("is it really a step of that long work, " if aim else "")
                    + "has the like of it failed before), whether the body's tools can do it, and "
                    "how. `goal`: the thing to do, as one sentence naming the tools and files, "
                    "changed where the regions showed a better way or the only possible one; null "
                    "if it should not be done, "
                    + ("if it does not take that long work itself further, " if aim else "")
                    + "or if no tool could show that it worked."
                )
            # A question may be settled by two voices; before the mind acts, three are heard.
            thought, verdict = await self.think_through(
                matter, first, signal.parent, ask, Answer if pondering else ConclusionAloud if asked else Conclusion,
                least=2 if wondering and not asked else 3,
            )
            wanted = " ".join((verdict.goal or "").split())
            words = clip(getattr(verdict, "say", None) or "", 400)
            if pondering:  # what the mind concluded about itself is always said, and remembered as said
                chosen = (verdict.name or "").strip()
                if chosen:  # a name it decided on is its name from here on, without another step
                    try:
                        brain.take_name(chosen)
                        if re.search(r"take_name|\bname\b|namen", wanted, re.I):
                            wanted = ""  # taken already: a goal to take it would set aside what it was doing for nothing
                    except ValueError as exc:
                        brain.tell("motor", self.key, f"The name “{clip(chosen, 40)}” was not taken: {exc}")
                words = words or clip(verdict.conclusion, 400)
                await brain.said(f"The human asked: {signal.content}\nAfter thinking it through I answered: {words}")
            if words:
                brain.talk.append(f"You: {clip(words, 300)}")
                brain.bus.publish(Signal("speech", self.key, words))
                await brain.voice.speak(words)
            if pondering and busy is not None:
                if not wanted or busy.data.get("asked"):  # the goal in hand goes on; their earlier request first
                    if wanted:
                        self.queued.append(Signal("proposal", "thalamus", wanted, data={
                            "asked": True, "words": signal.content, "answered": words}))
                    brain.proposal_pending = False
                    return False
                await self.finish("interrupted", "Set aside: what the mind decided about itself is done first.")
                brain.proposal_pending = True
            if asked and not wanted and not words:
                wanted = signal.content  # what the human asks for is never dropped in silence
            if wondering and not asked:  # thought through: no longer one of its open questions
                brain.settled(signal.content)
            if not wanted or (wondering and not asked and await self.refusal(wanted)):
                if not asked and not wondering:
                    brain.apply_reward(
                        signal.parent, -0.2, "reflex",
                        "Thought through and not taken up: " + clip(verdict.conclusion, 120),
                    )
                    brain.tell(  # whoever proposed it hears why, and proposes something else next
                        self.key, "dmn", f"What you proposed (“{clip(signal.content, 100)}”) was thought "
                        f"through and not taken up: {verdict.conclusion}",
                    )
                return False
            brain.goal = await brain.emit(
                self.key, "goal", wanted, parent=thought.id, data={"origin": signal.source, "asked": asked}
            )
            brain.net.wear(brain.goal.bits)
            brain.steps, brain.proposal_pending = 0, False
            self.trail, self.evidence, self.stuck, self.misses = [], [], 0, 0
            self.answered, self.rethought, self.unrun, self.last_run = False, False, False, ""
            self.built, self.shown = "", False
            self.wanted = await brain.wants(" ".join(filter(None, [wanted, signal.data.get("words")])))
            self.approach = (
                "Before you began, the regions of the mind thought it through together and "
                "concluded:\n" + verdict.conclusion
            )
            return True
        finally:
            if brain.goal is None:
                self.release()

    async def think_through(
        self, matter: str, first: str, parent: int | None, ask: str, schema: type[Conclusion] = Conclusion,
        least: int = 2,
    ) -> tuple[Signal, Conclusion]:
        """The mind thinks something through as one. Regions speak in turn, each from what only
        it knows and each naming who should be heard next; then the executive concludes. Nobody
        speaks twice, and at least `least` are heard where that many are there to hear."""
        brain = self.brain
        human = schema is not Conclusion  # a matter the human raised: its turns go ahead of everything else
        left = [part for part, key in PARTS.items() if brain.settings.regions[key].enabled]
        said: list[str] = []
        reflex = brain.facts(matter)
        if reflex:  # the body speaks first and by reflex: no model, only what is so
            said.append("The body, by reflex, before anyone spoke. These are facts: " + " ".join(reflex))
            brain.tell("motor", self.key, " ".join(reflex))
        speaker, last, voices = (first if first in left else left[0] if left else None), parent, 0
        while speaker and voices < TURNS:
            if brain.regions[PARTS[speaker]].busy.locked():  # busy with its own work: another is heard first
                free = [part for part in left if not brain.regions[PARTS[part]].busy.locked()]
                speaker = free[0] if free else speaker  # (the body's turn waited 10 s behind its own build)
            left.remove(speaker)
            region = brain.regions[PARTS[speaker]]
            try:
                view, then = await region.weigh_in(matter, said, left, last, first=human)
            except Exception as exc:  # one region faltering does not end the thought
                brain.note(f"{region.cfg.title} had nothing to add: {clip(str(exc) or type(exc).__name__, 160)}", "error")
                then = "executive"
            else:
                said.append(f"{region.cfg.title}, the {speaker}: {view.content}")
                last, voices = view.id, voices + 1
            speaker = then if then in left else None
            if speaker is None and voices < least:  # one voice is not yet a thought thought through
                # Nobody was named: whoever has the strongest tract into the executive is heard,
                # the region whose word has paid most when the mind concluded right after it.
                order = [part for part in ("body", "critic", "memory", "imagination") if part in left]
                order.sort(key=lambda part: -brain.net.tract((PARTS[part], self.key)).strength)
                speaker = order[0] if order else None
        parts = [
            brain.sense(),
            "The mind has thought something through, its regions in turn. The matter:\n" + matter,
            "What the regions said:\n" + ("\n".join(said) or "(none of them could speak)"),
            build.GOAL if re.search(r"free\s*cad|`build`", matter + " ".join(said), re.I) and brain.tools.builds() else "",
            ask,
        ]
        reply, heuristics = await self.think(
            "\n\n".join(p for p in parts if p), schema, room=100 if human else 0, first=human
        )
        conclusion = await brain.emit(
            self.key, "conclusion", f"{clip(matter.splitlines()[0], 220)}\n⇒ {reply.conclusion}",
            parent=last, heuristics=heuristics, data={"goal": reply.goal or ""},
        )
        self.last = f"concluded: {reply.conclusion}"
        return conclusion, reply

    async def rethink(self, signal: Signal, latest: str) -> None:
        """The goal keeps failing. Instead of one more try the same way, the regions are called
        together to say why, and the executive goes on from what they conclude."""
        brain = self.brain
        steps = "\n".join(f"- {step}" for step in self.trail[-3:])
        # What was last measured says more than what was last written, and its end says most.
        shown = (
            f"What the last run reported:\n…{self.last_run[-900:]}" if self.last_run
            else f"The latest result:\n{latest[:900]}"
        )
        try:
            _, verdict = await self.think_through(
                f"This goal keeps failing: {brain.goal.content}\nThe last steps:\n{steps}\n"
                f"{shown}\nWhy does it keep failing, and what different way would work?",
                "critic", signal.id,
                "Conclude for the whole mind, from what the regions said. `conclusion`: why it kept "
                "failing, and the different way to take from here, naming the tools. Leave `goal` "
                "null: the goal stays.",
            )
        except Exception as exc:
            brain.note(f"Could not think it through again: {clip(str(exc) or type(exc).__name__, 160)}", "error")
            return
        self.approach = (
            "It kept failing, so the regions of the mind thought it through again and concluded:\n"
            + verdict.conclusion
        )

    def release(self) -> None:
        """Nothing is in hand any more: the next thing the human asked for is taken up, or the
        mind is free to wander."""
        self.brain.proposal_pending = bool(self.queued)
        if self.queued:
            self.brain.bus.publish(self.queued.pop(0))

    async def refusal(self, goal: str) -> tuple[float, str, str, str] | None:
        """Why a proposal is turned down before any thought is spent on it, if it is: its
        pattern is that of a goal already tried, or of a theme that is worn out."""
        brain = self.brain
        unreal = brain.tools.no_models(goal, missing=False)
        if unreal:  # the machine has looked: what the goal is about is not there to be worked on
            return (-0.4, "reflex", "Proposed work on a file that is not a model: " + clip(unreal[0], 200),
                    "A .FCStd that will not open is not a model. Build the thing anew with `build`.")
        bits = encode(await brain.llm.embed(goal))
        attempted = brain.memory.recent(self.key, "goal", 8)
        alike = brain.memory.resonate(bits, kinds=("goal",), last=600)
        if (alike and alike[1] >= ALIKE) or any(
            SequenceMatcher(None, goal.lower(), r["content"].lower()).ratio() > 0.8 for r in attempted
        ):
            return (-0.5, "critic", "Proposed a goal that was already attempted.",
                    "Propose something that has not been tried yet.")
        if brain.net.worn(bits) >= SPENT:
            return (-0.3, "reflex", "Proposed more of a theme that is worn out.",
                    "When a theme is worn out, turn to something different.")
        return None

    def intuition(self, moment: Signal) -> str:
        """What the synapses suggest for a moment like this one: a step from the past that came
        after something alike and paid. The moment's pattern travels the tract it arrived by,
        from the motor cortex after an action or within the executive after a goal, leaned
        toward what was rewarded at this end. No model: a pattern, looked up."""
        brain = self.brain
        expected = brain.net.expect(moment.bits, (moment.source, self.key)) if moment.bits is not None else None
        if expected is None:
            return ""
        own = {s.id for s in brain.workspace} | {brain.goal.id}
        found = brain.memory.resonate(expected, own, ("plan",))
        row = brain.memory.get(found[0]) if found and found[1] >= 0.5 else None
        if row is None or row["value"] <= 0.15 or row["id"] > brain.goal.id:
            return ""
        return "Intuition, from a moment like this one that went well:\n- " + clip(row["content"], 240)

    async def finish(self, status: str, reasoning: str, heuristics: list[Heuristic] | None = None) -> None:
        """Close the current goal. What the runs actually showed travels with the verdict."""
        brain = self.brain
        goal = brain.goal
        if goal is None:
            return
        data: dict[str, Any] = {"status": status}
        if self.evidence:
            data["grounded"] = sum(self.evidence) / len(self.evidence)
        brain.goal = None
        self.last = f"{status}: {clip(goal.content, 80)}"
        await brain.emit(
            self.key, "outcome", f"{status.upper()}: {goal.content}\n{reasoning}",
            parent=goal.id, heuristics=heuristics or [], data=data,
        )
        if not self.answered:
            brain.news.append(f"A goal of yours ended ({status}): {clip(goal.content, 160)}")
        self.release()


# -- action -------------------------------------------------------------------------

FREECAD = (
    "Building in FreeCAD. " + build.HOW + "\n"
    "Only for what `build` cannot make, a script of your own for `run_freecad`: begin with "
    "`import FreeCAD as App, Part` and `doc = App.newDocument('model')`. Solids: "
    "`Part.makeBox(length, width, height, App.Vector(x, y, z))`, "
    "`Part.makeCylinder(radius, height, App.Vector(x, y, z), App.Vector(0, 0, 1))`, "
    "`Part.makeSphere(radius, App.Vector(x, y, z))`. Join two with `a.fuse(b)`, take one out of "
    "the other with `a.cut(b)`; parts must overlap to become one solid, and `Part.makeCompound` "
    "only groups them. Put the result into the document: `obj = doc.addObject('Part::Feature', "
    "'Name')` then `obj.Shape = shape`. One script makes every part and fuses them all: never "
    "open an earlier .FCStd from a script to build on it, and never write or repair a .FCStd by "
    "hand. Lengths are millimetres. The model is saved and measured for you, so a script only "
    "builds: no code that checks or validates the result, no saveAs, no FreeCADGui."
)


class Act(Explicit):
    tool: Literal[
        "list_files", "read_file", "write_file", "run_python", "build", "run_freecad", "recall", "read_source",
        "note", "say", "tune_voice", "take_name", "look", "camera", "listen", "hands", "show", "cannot",
    ]
    path: str = Field(
        "", description="workspace-relative file path, when the tool needs one; for build, the model's name; "
        "for look and hands, the window"
    )
    query: str = Field("", description="what to search memory for (recall only)")
    content: str = Field(
        "", description="full file contents (write_file), the note text (note), the words to speak (say), "
        "the voice settings as JSON (tune_voice), the name you give yourself (take_name), what to look "
        "for (look, camera), how many seconds (listen), what the hands do (hands) or why it cannot be "
        "done (cannot)"
    )
    parts: list[build.Part] = Field(
        default_factory=list, description="the parts to add to the model or change in it (build only); [] otherwise",
        json_schema_extra={"maxItems": 8},
    )


class Motor(Region):
    key = "motor"
    aware = False
    listens = ("plan",)
    part = "body"
    role = (
        "You know what the body can do: its tools are listed above, and there are no others. Go "
        "through what this would take step by step, and name the tool for each step and the file "
        "it works on. Where a step needs something no tool does (running a shell script, "
        "installing or starting a program, anything on the human's machine that no tool above "
        "reaches), say plainly that it cannot be done that way, and name the nearest thing the "
        "tools can do."
    )

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain)
        self.goal_id = 0
        self.changes = 0  # successful writes during this goal
        self.seen: set[tuple] = set()

    async def knows(self, matter: str) -> str:
        return "Workspace files, the latest first:\n" + self.brain.tools.listing(15)

    async def wait_trigger(self) -> Signal:
        return await self.hear()

    async def fire(self, plan: Signal) -> None:
        brain = self.brain
        goal_id, intent = plan.data["goal_id"], plan.data["intent"]
        if brain.goal is None or brain.goal.id != goal_id:
            return
        if goal_id != self.goal_id:
            self.goal_id, self.changes, self.seen = goal_id, 0, set()
        parts = [f"Goal: {brain.goal.content}", f"Intent to carry out now: {intent}", brain.reflexes(intent),
                 "Workspace files, the latest first:\n" + brain.tools.listing(30),
                 "Scripts may import anything, and an absolute path reaches beyond the workspace."
                 if brain.full_mode
                 else "Scripts may import only: " + ", ".join(brain.settings.safety.allowed_imports),
                 await brain.voice.describe()]
        if "freecad" in f"{brain.goal.content} {intent} {brain.settings.purpose}".lower():
            parts += [FREECAD, brain.tools.blueprint()]
        parts.append(self.habit(plan))
        for name in brain.tools.files():
            if name in intent:  # the file being worked on is worth seeing in full
                text = (brain.tools.workspace / name).read_text(encoding="utf-8", errors="replace")
                parts.append(f"Current contents of {name}:\n{text[:3000]}")
                break
        if brain.last_action:
            parts.append("Result of the previous action:\n" + brain.last_action.content[:1500])
            if not brain.last_action.data["ok"] or stale(brain.last_action):
                parts.append(
                    "The previous action failed or changed nothing. Do something different this "
                    "time: a file you write must differ from what is already there."
                )
        heuristics: list[Heuristic] = []
        tool = path = ""
        repeat = False
        def print_of(act: Act) -> tuple:
            return (act.tool, act.path, act.query, act.content, str(act.parts), self.changes)

        try:
            act, heuristics = await self.think("\n\n".join(p for p in parts if p), Act, room=600)  # a whole file fits
            # The same call with nothing written in between can only give the same result.
            fingerprint = print_of(act)
            if fingerprint in self.seen:  # caught before it is done: one chance to choose again
                parts.append(
                    f"You chose {act.tool}({act.path or clip(act.query, 60)}) exactly as before. "
                    "Nothing has been written since, so it can only give the same result. Choose "
                    "again, differently: where the intent asks for a change or a fix, write the "
                    "whole changed file with `write_file`, or `build` with the part changed."
                )
                act, heuristics = await self.think("\n\n".join(p for p in parts if p), Act, room=600)
                fingerprint = print_of(act)
            tool, path = act.tool, act.path
            repeat = fingerprint in self.seen
            self.seen.add(fingerprint)
            result = await self.execute(act, intent)
            if result.ok and act.tool in ("write_file", "note"):
                self.changes += 1
            header = f"{act.tool}({act.path or clip(act.query, 60)})"
        except Truncated as cut:  # what it was writing did not fit into one reply: said as that, not as broken JSON
            header = "could not act"
            fits = cut.room * 3  # about three characters to a token of code
            result = ToolResult(False, (
                f"the reply was cut off: it ran out of room after {cut.tokens} tokens ({cut.characters} characters) "
                "while it was being written, so nothing was done. What one step writes must fit into one reply: "
                f"about {fits} characters at most. Write the file shorter, or in two files, each in a step of its own."
            ), -0.2)
            brain.tell(self.key, "pfc", f"I could not carry out “{clip(intent, 100)}”: what it asks me to write is too "
                       f"long for one step (cut off after {cut.characters} characters). Ask for it shorter or in parts.")
        except Exception as exc:
            header = "could not act"
            result = ToolResult(False, clip(str(exc) or type(exc).__name__, 300), -0.3)
        self.last = f"{header} → {'ok' if result.ok else 'failed'}"
        if tool:
            brain.used[tool] += 1
        brain.last_action = await brain.emit(
            self.key, "action", f"{header}\n{result.text[:2500]}", parent=plan.id, heuristics=heuristics,
            data={"ok": result.ok, "grounded": result.grounded, "goal_id": goal_id, "tool": tool,
                  "path": path, "repeat": repeat, "measured": tool in ("run_python", "run_freecad", "build")},
        )

    def habit(self, plan: Signal) -> str:
        """What the motor cortex's own neurons bring up for a plan. The plan's pattern travels the
        tract from the executive, where every step from a plan to an action has left its mark and
        its reward; leaned toward what paid at this end, it lights the action a plan like this
        was once answered with. No model: a pattern, looked up."""
        brain = self.brain
        expected = brain.net.expect(plan.bits, (plan.source, self.key)) if plan.bits is not None else None
        if expected is None:
            return ""
        found = brain.memory.resonate(expected, {plan.id}, ("action",))
        row = brain.memory.get(found[0]) if found and found[1] >= 0.5 else None
        if row is None or row["value"] <= 0.15 or row["id"] > plan.id:
            return ""
        return "What your own neurons bring up, from a step like this one that went well:\n- " + clip(row["content"], 260)

    async def execute(self, act: Act, intent: str) -> ToolResult:
        brain = self.brain
        if act.tool == "take_name":
            try:
                return ToolResult(True, brain.take_name(act.content), 0.3)
            except ValueError as exc:
                return ToolResult(False, f"no name was taken: {exc}", -0.1)
        if act.tool == "recall":
            found = await brain.recall(act.query or intent, k=5)
            return ToolResult(True, found or "Nothing relevant in memory yet.")
        if act.tool == "say":
            words = clip(act.content, 600)
            if not words:
                return ToolResult(False, "refused: nothing to say", -0.2)
            brain.bus.publish(Signal("speech", self.key, words, data={"tool": True}))
            unspoken = await brain.voice.speak(words)
            return ToolResult(True, f"said to the human: {words}" + (f"\n(as text only: {unspoken})" if unspoken else ""))
        if act.tool == "tune_voice":
            try:
                described = await brain.voice.tune(act.content)
            except ValueError as exc:
                return ToolResult(False, str(exc), -0.2)
            brain.save_settings()
            return ToolResult(True, f"{described}\nNothing was spoken: only `say` makes a sound.")
        if act.tool == "cannot":  # it says so to the executive instead of doing something else
            why = clip(act.content, 400) or "no tool fits this intent"
            brain.tell(self.key, "pfc", f"I cannot carry out “{clip(intent, 120)}”: {why}")
            return ToolResult(False, f"the motor cortex cannot carry this out: {why}", -0.1)
        amygdala = brain.regions["amygdala"]
        if not brain.full_mode:
            amygdala.activity = 1.0
        result = await brain.tools.execute(act.tool, act.path, act.content, [part.model_dump() for part in act.parts])
        if act.tool == "look" and result.ok:  # what it looked at on purpose it also holds from now on
            await brain.glance()
            aimed = bool(act.path.strip() or act.content.strip())
            try:  # and it looks into the window it named, or the one in front, for what it named
                vision = await brain.peer(act.path.strip(), act.content.strip(), controls=True, strict=aimed, raw=True)
            except (OSError, ValueError) as exc:
                return ToolResult(False, f"{result.text}\nYour eyes could not look there: {clip(str(exc), 400)}", -0.1)
            if vision and vision.get("picture"):  # it sees for itself: the picture goes to whoever thinks next
                shows = f"You look into “{vision['window']}”: the picture itself comes with what you are shown next."
                if act.content.strip():
                    shows += f" You looked for: {clip(act.content, 200)}"
                if vision.get("controls"):
                    shows += f"\n{vision['controls']}"
                result = ToolResult(True, f"{result.text}\n{shows}", result.grounded,
                                    media={"images": [base64.b64decode(vision["picture"])], "audio": []})
            elif vision:
                shows = f"What “{vision['window']}” shows: {vision['seen']}"
                if vision.get("controls"):
                    shows += f"\n{vision['controls']}"
                result = ToolResult(True, f"{result.text}\n{shows}", result.grounded)
        if result.media:
            brain.take_in(result.media, f"what you took in with {act.tool}({clip(act.path or act.content, 60)})")
        if result.threat:
            amygdala.alarm(result.text)
        return result


# -- reward -------------------------------------------------------------------------


class Judgment(BaseModel):
    item: int = Field(description="the number of the item being judged")
    reward: float = Field(ge=-1, le=1)
    reason: str = Field(description="one short sentence, at most 12 words")
    lesson: str | None = Field(None, description="one short reusable rule, or null")


class Verdict(BaseModel):
    # maxItems binds the model's grammar; a longer list would otherwise run until tokens ran out.
    judgments: list[Judgment] = Field(json_schema_extra={"maxItems": 4})


class BasalGanglia(Region):
    key = "basal_ganglia"
    aware = False
    nocturnal = True  # finishes its backlog before the sleeping brain consolidates
    # A conclusion, like a plan, earns what comes of it. An answer to the human is judged too: for
    # whether what it says is so.
    listens = ("thought", "plan", "action", "outcome", "exchange")
    BATCH = 4
    part = "critic"
    role = (
        "You weigh what has been said. Name what in it is unsupported, wishful or contradicted by "
        "the facts of the body and of memory, and say whether it takes the mind's long work or "
        "the human's wish any further at all. Then say which course is worth most, and what "
        "measured result would prove that it worked."
    )

    async def knows(self, matter: str) -> str:
        brain = self.brain
        gut = brain.net.feel(encode(await brain.llm.embed(matter)))
        felt = "it went well" if gut > 0.1 else "it went badly" if gut < -0.1 else "it went neither well nor badly"
        parts = [
            "By its pattern alone, before any thought: when the mind was on something like this "
            f"before, {felt} (gut feeling {gut:+.2f})."
        ]
        if brain.last_action:  # what was last measured is the critic's own ground
            parts.append("The last thing the body did, and what came of it:\n" + brain.last_action.content[:700])
        return "\n\n".join(parts)

    def speak_up(self, signal: Signal, reward: float, reason: str, lesson: str | None) -> None:
        """A verdict is not only a number in memory. When the critic objects, whoever can still
        do something about it hears why: the executive while the goal the step belongs to is in
        hand, the default mode network for a thought or for how a goal ended."""
        brain = self.brain
        if reward > -0.2:
            return
        if signal.kind in ("action", "plan"):
            if brain.goal is None or signal.data.get("goal_id") != brain.goal.id:
                return  # that goal is over: nobody can do anything about the step any more
            to = "pfc"
        elif signal.kind == "exchange":  # the part that answers the human hears it before it answers again
            to = "thalamus"
        else:
            to = "dmn"
        what = {"action": "that step", "plan": "that plan", "outcome": "how the last goal ended",
                "exchange": "your last answer to the human"}
        brain.tell(
            self.key, to,
            f"{reward:+.1f} for {what.get(signal.kind, 'that thought')} (“{clip(signal.content.splitlines()[0], 80)}”): "
            f"{reason}" + (f" Next time: {lesson}" if lesson else ""),
        )

    @staticmethod
    def needs_words(signal: Signal) -> bool:
        """Whether the model has to read it. A repeat is known by its pattern alone, and a plan
        earns what its action earns: that reward travels back to it."""
        return signal.kind != "plan" and not stale(signal)

    async def wait_trigger(self) -> list[Signal]:
        batch = [await self.hear()]
        while sum(map(self.needs_words, batch)) < self.BATCH and not self.inbox.empty():
            batch.append(self.inbox.get_nowait())
            self.excite(batch[-1].bits)
        return batch

    async def fire(self, signals: list[Signal]) -> None:
        brain = self.brain
        for signal in signals:
            if stale(signal):  # a reflex: no words, no waiting for the model
                reward = min(-0.3, signal.data.get("grounded") or 0.0)  # repetition never earns reward
                brain.apply_reward(
                    signal.id, reward, "reflex", "Repeats something done or thought moments ago.",
                    "Do not repeat a step that changed nothing: try something different.",
                )
                self.speak_up(signal, reward, "It repeats something done or thought moments ago.", None)
                self.last = f"{reward:+.2f} a repeat, punished by reflex"
        batch = [signal for signal in signals if self.needs_words(signal)]
        if not batch:
            return
        items = []
        for number, signal in enumerate(batch, 1):
            facts = []
            if signal.data.get("grounded") is not None and signal.kind in ("action", "outcome"):
                facts.append(f"measured outcome {signal.data['grounded']:+.1f}")
            title = brain.settings.regions[signal.source].title
            suffix = f"\n   ({'; '.join(facts)})" if facts else ""
            items.append(f"{number}. [{title} · {signal.kind}] {clip(signal.content, 700)}{suffix}")
        goal = brain.goal.content if brain.goal else "none (the mind is wandering)"
        answers = any(signal.kind == "exchange" for signal in batch)
        verdict, _ = await self.think(
            f"Current goal: {goal}\n\nItems to judge ({len(batch)}, one judgment each):\n" + "\n\n".join(items)
            + "\n\nJudge each item for what it is, not for the news it brings: a thought that sees "
            "truly that something failed, and why, is good thinking."
            + (" An exchange is an answer the mind gave the human: judge it for whether what it says "
               "is so. What it claims to see, hear, have, have done or be doing must be so by the "
               "facts of the body and by what the mind is shown to have done; a claim that is not so "
               "is -1 however kind it sounds, and saying plainly what cannot be done is good."
               if answers else ""),
            Verdict,
        )
        judged: set[int] = set()
        for judgment in verdict.judgments:
            if not 1 <= judgment.item <= len(batch) or judgment.item in judged:
                continue
            judged.add(judgment.item)
            signal = batch[judgment.item - 1]
            reward = self.blend(signal, judgment.reward)
            brain.apply_reward(signal.id, reward, "critic", judgment.reason, judgment.lesson)
            self.speak_up(signal, reward, judgment.reason, judgment.lesson)
            self.last = f"{reward:+.2f} {clip(judgment.reason, 70)}"

    def blend(self, signal: Signal, opinion: float) -> float:
        """Mix the critic's opinion with what was actually measured."""
        reward = opinion
        grounded = signal.data.get("grounded")
        if grounded is not None:
            weight = self.brain.settings.learning.grounded_weight
            reward = (1 - weight) * opinion + weight * grounded
        return max(-1.0, min(1.0, reward))


# -- memory -------------------------------------------------------------------------


class Story(BaseModel):
    story: str = Field(description="the day so far in the first person, at most 120 words")


class Night(BaseModel):
    day: str = Field(description="the diary entry for the day, 2 to 4 sentences in the first person")
    me: str = Field(description="who I am now, at most 80 words in the first person")
    insights: list[str] = Field(description="1 to 4 durable, specific insights")


class Hippocampus(Region):
    key = "hippocampus"
    nocturnal = True
    part = "memory"
    role = (
        "You know what has happened before. Say what the mind has already tried or learned that "
        "bears on this, and how it went: what worked, what failed and how often. Only what the "
        "memories shown here say: nothing invented, and if they hold nothing on it, say that."
    )

    async def knows(self, matter: str) -> str:
        brain = self.brain
        parts = ["What long-term memory holds on this:\n" + (await brain.recall(matter, k=6) or "(nothing like it)")]
        ended = brain.memory.recent("pfc", "outcome", 4)
        if ended:
            parts.append(
                "How the last goals ended, the latest first:\n"
                + "\n".join(f"- [{r['value']:+.1f}] {clip(r['content'], 200)}" for r in ended)
            )
        if brain.story:
            parts.append("The day so far:\n" + clip(brain.story, 500))
        return "\n\n".join(parts)

    async def wait_trigger(self) -> None:
        """Awake, it has one job of its own: when enough has left short-term memory, fold it into
        the story of the day, so that the mind's context stays small and nothing is simply lost."""
        brain = self.brain
        while len(brain.fading) < FADE or not brain.awake.is_set():
            await asyncio.sleep(TICK)

    async def fire(self, _: None) -> None:
        brain = self.brain
        if not brain.awake.is_set():
            return  # asleep, the whole day is compacted at once: see `consolidate`
        leaving = brain.fading[:]
        now = time.time()
        reply, _ = await self.think(
            "The story of the day so far:\n" + (brain.story or "(the day has only just begun)")
            + "\n\nWhat has just left short-term memory, oldest first:\n"
            + "\n".join(f"- [{span(now - s.ts)} ago · {s.source}/{s.kind}] {clip(s.content, 260)}" for s in leaving)
            + "\n\nWrite `story`: the story of the day with this folded in, in the first person, at "
            "most 120 words. Keep what still matters: what I set out to do, what came of it, what "
            "I am in the middle of, and what the human said if a `stimulus` above shows them "
            "speaking: otherwise they said nothing, and wanted nothing. Let the details go.",
            Story,
        )
        if reply.story.strip():
            brain.story = clip(reply.story, 900)
            del brain.fading[: len(leaving)]
            self.last = f"folded {len(leaving)} moments into the story of the day"

    async def consolidate(self) -> None:
        """Called by the brainstem during sleep: the day becomes a diary entry, a few insights and
        a fresh picture of itself, and short-term memory is let go."""
        brain = self.brain
        mark = brain.memory.get_state("consolidated", 0)
        episodes = brain.memory.since(mark, ("thought", "conclusion", "plan", "action", "outcome"), 40)
        if self.cfg.enabled and len(episodes) >= 6:
            with self.firing():
                lines = [
                    f"- [{r['value']:+.1f}] {r['region']}/{r['kind']}: {clip(r['content'], 240)}"
                    for r in episodes
                ]
                parts = [
                    brain.sense(),
                    "Who I was when I last fell asleep:\n" + (brain.self_image or "(nothing yet: this is my first sleep)"),
                    "The story of today:\n" + brain.story if brain.story else "",
                    "The last moments before sleep:\n" + brain.stream(),
                    "Recent episodes, each with the reward it earned:\n" + "\n".join(lines),
                    "Write three things. `day`: my diary entry for today, 2 to 4 sentences in the "
                    "first person: what I tried and what came of it. `me`: who I am now, at most 80 "
                    f"words, beginning 'I am {brain.name}': the whole mind speaking, not "
                    "one region of it. What I can do, what I have learned about myself, what I am "
                    "working toward. Keep what is still true of who I was. `insights`: 1 to 4 "
                    "durable, specific insights. All three hold only what is shown above: nothing "
                    "about work that was not done, and nothing the human wanted, said or expected "
                    "unless their words are quoted above.",
                ]
                reply, heuristics = await self.think("\n\n".join(p for p in parts if p), Night, room=300)
                # The compaction itself: what the mind was holding is let go, and what was made
                # of it is what it wakes with.
                brain.story = ""
                brain.fading.clear()
                brain.workspace.clear()
                brain.mind.clear()
                if reply.me.strip():
                    brain.self_image = clip(reply.me, 700)
                if reply.day.strip():
                    brain.yesterday = f"Day {brain.sleeps + 1}: {clip(reply.day, 700)}"
                    await brain.emit(self.key, "day", brain.yesterday, value=0.3)
                for text in reply.insights[:4]:
                    if text.strip():
                        await brain.emit(self.key, "insight", text.strip(), heuristics=heuristics, value=0.3)
                brain.memory.set_state("consolidated", episodes[-1]["id"])
                self.last = f"compacted the day: {len(episodes)} episodes"
        forgotten = brain.memory.forget_neutral(brain.settings.rhythm.memory_cap)
        if forgotten:
            brain.note(f"Forgot {forgotten} old memories that never mattered either way.", "sleep")


# -- self-improvement ---------------------------------------------------------------


class Tweak(BaseModel):
    region: Literal["dmn", "pfc", "motor"]
    heuristic: str = Field(description="one short imperative rule, at most 25 words, that applies to any topic")


class Refinement(BaseModel):
    heuristics: list[Tweak] = Field(description="0 to 4 new heuristics")


class Cerebellum(Region):
    key = "cerebellum"
    aware = False
    nocturnal = True

    async def refine(self) -> None:
        """Called by the brainstem during sleep: turn rewards into better habits."""
        brain = self.brain
        mark = brain.memory.get_state("refined", 0)
        best = brain.memory.extremes(mark, 5, best=True)
        worst = brain.memory.extremes(mark, 5, best=False)
        if self.cfg.enabled and len(best) + len(worst) >= 4:
            with self.firing():

                def lines(rows) -> str:
                    return "\n".join(
                        f"- [{r['value']:+.2f}] {r['region']}/{r['kind']}: {clip(r['content'], 220)}"
                        + (f"\n  lesson: {r['lesson']}" if r["lesson"] else "")
                        for r in rows
                    ) or "- (none)"

                known = "\n".join(
                    f"- {h.region} (weight {h.weight:.2f}): {h.text}" for h in brain.learner.heuristics()
                ) or "- (none yet)"
                reply, _ = await self.think(
                    f"Best-rewarded episodes:\n{lines(best)}\n\nWorst-rewarded episodes:\n{lines(worst)}"
                    f"\n\nHeuristics already in use:\n{known}",
                    Refinement, room=300,
                )
                added = 0
                for tweak in reply.heuristics[:4]:
                    heuristic = brain.learner.add(tweak.region, tweak.heuristic)
                    if heuristic:
                        added += 1
                        title = brain.settings.regions[tweak.region].title
                        await brain.emit(self.key, "lesson", f"{title}: {heuristic.text}")
                brain.memory.set_state("refined", brain.memory.last_id())
                await brain.attune()
                self.last = f"{added} new heuristics"
        pruned = brain.learner.prune()
        if pruned:
            brain.note(f"Suppressed {pruned} heuristics that kept losing.", "sleep")


# -- regions that do not use the model ---------------------------------------------


class Thalamus(Region):
    key = "thalamus"
    uses_llm = False
    nocturnal = True
    listens = ()
    GATED = frozenset({
        "thought", "goal", "plan", "action", "outcome", "insight", "lesson", "stimulus", "speech", "day", "dream",
        "conclusion", "percept",
    })

    async def wait_trigger(self) -> Signal:
        return await self.hear()

    async def fire(self, signal: Signal) -> None:
        brain = self.brain
        if signal.kind not in self.GATED or (signal.kind == "speech" and signal.data.get("tool")):
            return  # what the `say` tool spoke is already there as the action that spoke it
        if signal.kind == "thought" and stale(signal):
            self.last = "habituated: a repeated thought was kept out"
            return
        if len(brain.workspace) == brain.workspace.maxlen:  # the oldest moment leaves; it is not lost
            brain.fading.append(brain.workspace[0])
            del brain.fading[:-60]
        brain.workspace.append(signal)
        self.last = f"{signal.kind} from {signal.source} → short-term memory"


class Amygdala(Region):
    key = "amygdala"
    uses_llm = False
    nocturnal = True

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain)
        self.threats = 0

    def alarm(self, text: str) -> None:
        self.threats += 1
        self.fires += 1
        self.activity = 1.0
        self.neurons[:] = 1.0  # all of it, at once
        self.last = clip(text, 120)
        self.brain.note(text, "threat")
        self.brain.tell(self.key, "pfc", f"I stopped that step: {text} Take a way that stays inside the workspace.")


class Brainstem(Region):
    key = "brainstem"
    uses_llm = False
    nocturnal = True

    def __init__(self, brain: Brain) -> None:
        super().__init__(brain)
        self._sleep: asyncio.Task | None = None
        self._spilling = False

    async def run(self) -> None:
        brain = self.brain
        await self.awaken()
        try:
            while True:
                await asyncio.sleep(TICK)
                if not brain.running.is_set():
                    continue
                brain.tick += 1
                self.pulse()
                sleeping = self._sleep is not None and not self._sleep.done()
                if not sleeping and (brain.fatigue >= 1 or brain.sleep_requested):
                    brain.sleep_requested = False
                    self._sleep = asyncio.create_task(self.sleep_cycle())
                if not sleeping:
                    await self.watchdog()
                    if brain.tick % ECHO_EVERY == 0:  # between thoughts the mind is not empty
                        await brain.aim()
                        brain.remind()
                    if brain.tick % SENSE_EVERY == 0 and brain.awake.is_set():
                        with contextlib.suppress(OSError):
                            await brain.perceive()
                    if brain.tick % LOOK_EVERY == 0 and brain.awake.is_set():
                        with contextlib.suppress(OSError):
                            await brain.glance()
                if brain.tick % 20 == 0:
                    brain.persist(synapses=brain.tick % 600 == 0)
                    await self.check_model()
        finally:
            if self._sleep is not None and not self._sleep.done():  # the brain stops; so does its sleep
                self._sleep.cancel()
                await asyncio.gather(self._sleep, return_exceptions=True)

    async def awaken(self) -> None:
        """Nobody prompts this brain. Its regions come online and it starts thinking."""
        brain = self.brain
        memories = brain.memory.count()
        if memories:
            brain.note(f"Waking with {memories} memories and {len(brain.learner.heuristics())} learned heuristics.", "wake")
            brain.news.append("You have just been started again.")
        else:
            brain.note("First awakening. No memories, no heuristics, no instructions.", "wake")
        if brain.net.born and memories:  # its tracts came later than this brain: it relives its past into them
            brain.memory.set_state("replayed", 0)  # all of it, not only what happened since it last slept
            count, _, after = brain.replay(passes=3, foresight=True)
            brain.note(
                f"New synapses: {count} memories were replayed into the tracts between its regions as "
                f"patterns, each along the tract it once travelled. The gut feeling they give has "
                f"the sign of what those memories were worth for {after:.0%}.", "wake",
            )
            lying = brain.net.arrange(brain.memory.codes())
            if lying:  # where each neuron lies on its sheet: those that fire together, together
                brain.note(
                    f"Its neurons were laid out on their sheets from those memories: one lies {lying[0]:.1f} mm "
                    f"from the neuron it most often fires with, and {lying[1]:.1f} mm from any other on average."
                )
        brain.net.born = False
        await brain.aim()
        await brain.attune()
        themed = len(brain.learner.themed)
        if themed:
            brain.note(
                f"{themed} of its {len(brain.learner.heuristics())} heuristics are about one theme or "
                "another: each speaks up only when the mind is on that theme again."
            )
        try:  # it finds out what it is made of before it thinks its first thought
            body = await brain.tools.survey()
            brain.note(
                "Its body: " + (f"{body['builds']} to build in" if body["builds"] else "no FreeCAD to build in here")
                + (f"; {body['freecad']} on your screen to show models in." if body["freecad"]
                   else "; no FreeCAD on your machine to show models in.")
            )
        except Exception as exc:
            brain.note(f"Could not survey its body: {clip(str(exc) or type(exc).__name__, 200)}", "error")
        with contextlib.suppress(Exception):  # it opens its eyes before its first thought, if it has any
            await brain.glance()
        if brain.full_mode:
            brain.note("Full mode: the amygdala is off. Inside this container nothing is vetoed.", "threat")
        elif not brain.settings.regions["amygdala"].enabled:
            brain.note("The amygdala is off, but outside the container its guards stay on.")
        for region in brain.regions.values():
            region.activity = 1.0
            await asyncio.sleep(0.2)
        brain.progress = brain.woke = time.time()
        brain.awake.set()

    def pulse(self) -> None:
        brain = self.brain
        thinkers = [r for r in brain.regions.values() if r.uses_llm]
        busy = sum(r.state != "idle" for r in thinkers) / len(thinkers)
        brain.arousal = 0.9 * brain.arousal + 0.1 * busy
        self.activity = 1.0 if brain.tick % 2 == 0 else 0.35  # heartbeat
        for region in brain.regions.values():
            region.history.append(region.activity)
            if region.state == "idle" and region is not self:
                region.activity *= 0.75
        # The tissue beats with the heart: firing fades and spreads to its neighbours, and the
        # signals under way along the tracts move on. Where one arrives, that region stirs.
        lit = brain.net.beat()
        for key, level in brain.net.arrived.items():
            brain.regions[key].activity = max(brain.regions[key].activity, level)
        beat = brain.tick * 16 % NEURONS  # the pacemaker: a band of neurons that sweeps across, tick by tick
        self.neurons[beat : beat + 16] = 1.0
        # What is still lit of the last thoughts glows on in the default mode network: that
        # field is the mind between two thoughts.
        wandering = brain.regions["dmn"].neurons
        wandering[lit] = np.maximum(wandering[lit], 0.45)

    async def check_model(self) -> None:
        """A model that does not fit on the GPU thinks at less than half its speed. Say so, once."""
        share = await self.brain.llm.placement()
        spilling = share is not None and share < 0.99
        if spilling and not self._spilling:
            self.brain.note(
                f"The model is only {share:.0%} on the GPU, so it thinks far slower than it could. "
                "Lower the context window in settings (s) until it fits.", "error",
            )
        self._spilling = spilling

    async def sleep_cycle(self) -> None:
        """Real sleep, in four parts: the day is compacted (hippocampus), replayed as patterns
        into the synapses (no model), turned into habits (cerebellum) and dreamt on (default mode)."""
        brain = self.brain
        brain.awake.clear()
        brain.note("Falling asleep: the day will be compacted, replayed, learned from and dreamt on.", "sleep")
        try:
            critic = brain.regions["basal_ganglia"]
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(120):  # let running thoughts finish and be judged
                    while any(r.state != "idle" for r in brain.regions.values()) or not critic.inbox.empty():
                        await asyncio.sleep(0.25)
            held = len(brain.workspace) + len(brain.fading)
            for step in (
                brain.regions["hippocampus"].consolidate, self.replay,
                brain.regions["cerebellum"].refine, brain.regions["dmn"].dream, brain.school,
            ):
                try:
                    await step()
                except Exception as exc:
                    brain.note(f"Sleep was disturbed: {clip(str(exc) or type(exc).__name__, 200)}", "error")
            if brain.sleep_until:
                await self.long_sleep()
            brain.net.rest()
            brain.fatigue = 0.0
            brain.sleeps += 1
            brain.woke = time.time()
            brain.persist(synapses=True)
            gate = brain.regions["thalamus"].inbox
            for _ in range(200):  # what sleep made of the day reaches short-term memory first
                if gate.empty():
                    break
                await asyncio.sleep(0)
            brain.news.append(f"You have just woken from sleep number {brain.sleeps}.")
            brain.note(
                f"Awake again, on day {brain.sleeps + 1}. The {held} moments it fell asleep holding are "
                f"let go; it wakes holding the {len(brain.workspace)} things sleep made of the day.", "wake",
            )
        finally:  # an interrupted sleep leaves the fatigue in place, so the next life sleeps soon
            brain.sleep_until, brain.wake_up = 0.0, False
            brain.night_round, brain.night_epochs, brain.night_best = 1, 1, None
            brain.progress = time.time()
            brain.awake.set()

    async def long_sleep(self) -> None:
        """A night of hours, after the night's usual sleep. It goes to school again, each round
        learning longer on the same lessons from the model as it was, as long as a round does better
        on the exam than the one before and there is time for one more. Then it wakes by itself:
        what is left to gain is not worth the night. Woken, or out of time, it stops between rounds."""
        brain = self.brain
        first = brain.last_night
        if not first or first.get("exam_after") is None:
            brain.note("Nothing to learn in a long sleep tonight: it wakes after the usual night.", "sleep")
            return
        best = first["exam_after"] if first.get("kept") else first.get("exam_before")
        if not first.get("kept"):
            brain.note("Tonight's first round of school did not pass: another round on the same lessons will not either. It wakes.", "sleep")
            return
        took = max(60.0, 60 * float(first.get("minutes") or 0) + 60 * float(first.get("fold_minutes") or first.get("gguf_minutes") or 0))
        why = ""
        while True:
            if brain.wake_up:
                why = "it was woken"
                break
            if time.time() + 1.2 * took > brain.sleep_until:
                why = "there is no time left for another round"
                break
            brain.night_round += 1
            brain.night_epochs = brain.night_round  # each round learns one pass longer on the day's lessons
            brain.night_best = best
            brain.note(
                f"Long sleep, round {brain.night_round}: it learns {brain.night_epochs} times over from the "
                f"day's lessons, and keeps it only if it does better than {best} on the exam.", "sleep",
            )
            started = time.time()
            try:
                await brain.school()
            except Exception as exc:
                why = f"school failed: {clip(str(exc) or type(exc).__name__, 160)}"
                break
            took = max(took, time.time() - started)
            report = brain.last_night
            if not report or not report.get("kept"):
                why = f"round {brain.night_round} did no better ({(report or {}).get('exam_after', '?')} on the exam)"
                break
            best = report["exam_after"]
        brain.note(
            f"It wakes from its long sleep after {brain.night_round} round{'s' * (brain.night_round != 1)} of school: {why}. "
            f"It thinks with what did best on the exam ({best}).", "wake",
        )

    async def replay(self) -> None:
        """The part of sleep that needs no words: every moment since the last sleep goes through
        the synapses again, eight times, with the value it settled at."""
        net = self.brain.net
        if (net.order == np.arange(NEURONS)).all():  # a brain young enough never to have had its sheets laid out
            net.arrange(self.brain.memory.codes())
        count, before, after = self.brain.replay()
        if count:
            self.brain.note(
                f"Replayed {count} moments as patterns, eight times over. Before, the gut feeling had "
                f"the sign of what they turned out to be worth for {before:.0%} of them; now for {after:.0%}.",
                "sleep",
            )

    async def watchdog(self) -> None:
        """A goal nobody has touched for a long time is let go, so the mind cannot seize up."""
        brain = self.brain
        stalled = time.time() - brain.progress > brain.settings.rhythm.stall_seconds
        if brain.goal is not None and stalled and brain.awake.is_set():
            brain.note("The goal stalled; letting it go.", "error")
            await brain.regions["pfc"].finish("abandon", "Nothing moved for too long.")


# Order matters: it is the layout of the brain map, three regions to a row.
REGIONS: tuple[type[Region], ...] = (
    Prefrontal, Motor, Cerebellum,
    DefaultMode, Thalamus, Hippocampus,
    BasalGanglia, Amygdala, Brainstem,
)
