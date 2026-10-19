"""The brain runs here against a scripted model, so the tests need no model server."""

from __future__ import annotations

import asyncio
import io
import json
import random
import re
import time
import uuid
import wave
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
from textual.widgets import DataTable
from typer.testing import CliRunner

from cortex import bridge, build, container, neural, tools, voice
from cortex.brain import Brain, Heard, Reply
from cortex.bus import Signal
from cortex.cli import app
from cortex.config import BrainSettings, LearningSettings, ModelSettings, SafetySettings, VoiceSettings
from cortex.learning import Learner
from cortex.llm import LLM, Slots, lexical_vector
from cortex.memory import Memory
from cortex.neural import ACTIVE, Net, encode, pack, shared
from cortex.regions import (
    NEURONS,
    Act,
    Answer,
    Conclusion,
    Decision,
    Judgment,
    Night,
    Refinement,
    Story,
    Tweak,
    Verdict,
    View,
    Wander,
    WanderAloud,
)
from cortex.tools import Tools
from cortex.tui import CortexApp, Form, RegionCard, Wiring, field, format_signal


class ScriptedLLM:
    """Stands in for the model: thinks of a script, writes it, runs it, and is pleased."""

    senses = False  # a model that takes words only

    def __init__(self, same_goal: bool = False) -> None:
        self.same_goal = same_goal
        self.media: list[tuple[type, list, list]] = []  # pictures and sounds it was handed, by schema
        self.calls = self.tokens = 0
        self.speed = 50.0
        self.embed_error = None
        self.gpu_share = None
        self.answer = Reply(say="I am here.")
        self.aloud: str | None = None  # what it says when it may speak unasked
        self.heard = "talk"  # what the thalamus takes the human's words for
        self.hands_to = "executive"  # whom a region names next while they think something through
        self.acts = 0
        self.judged: list[str] = []
        self.prompts: dict[type, list[str]] = {}
        self.systems: list[str] = []

    async def generate(self, role, system, prompt, schema, *, temperature, max_tokens, on_start=None, on_token=None,
                       urgent=False, first=False, images=None, audio=None):
        self.first = first  # whether the call went ahead of the others
        if images or audio:
            self.media.append((schema, images or [], audio or []))
        self.calls += 1
        if on_start:
            on_start()
        await asyncio.sleep(0)
        self.prompts.setdefault(schema, []).append(prompt)
        self.systems.append(system)
        if issubclass(schema, Wander):
            goal = "Write and run a script that prints 6*7"
            said = {"say": self.aloud} if "say" in schema.model_fields else {}
            if "You are asleep and dreaming" in prompt:
                reply = schema(thought="Arithmetic is a script that has not been run yet.", kind="connection")
            else:
                fresh = " ".join(uuid.uuid4().hex for _ in range(5))  # each goal its own, in words and in pattern
                reply = schema(thought=f"What is 6 times 7? (thought {self.calls})", kind="question", **said,
                               goal=goal if self.same_goal else f"{goal} ({fresh})")
        elif issubclass(schema, View):  # its turn in a thinking-through: it has its say and hands on
            reply = schema(says=f"A script can be written and run to find out. (view {self.calls})",
                           then=self.hands_to if self.hands_to in schema.model_json_schema()["properties"]["then"]["enum"]
                           else "executive")
        elif issubclass(schema, Conclusion):  # what was proposed or asked for is found worth doing as it is
            wanted = re.search(r"(?:I am thinking of doing this|It was taken to mean|The human asked for this): (.*)", prompt)
            reply = schema(conclusion="It can be found out by writing a script and running it.",
                           goal=wanted.group(1) if wanted else None)
        elif schema is Heard:  # the scripted human says please when there is work in it
            reply = Heard(kind="task" if "please" in prompt else self.heard)
        elif schema is Decision:
            done = "ran successfully" in prompt
            reply = Decision(reasoning="It printed 42." if done else "Nothing exists yet.",
                             status="done" if done else "continue",
                             intent="" if done else "write calc.py then run it")
        elif schema is Act:
            self.acts += 1
            reply = (Act(tool="write_file", path="calc.py", content=f"print(6*7)  # draft {self.acts}\n")
                     if self.acts % 2 else Act(tool="run_python", path="calc.py"))
        elif schema is Verdict:
            self.judged.append(prompt)
            count = prompt.count("\n\n") or 1
            reply = Verdict(judgments=[Judgment(item=i, reward=0.8, reason="Concrete and verified.",
                                                lesson="Run code to check a claim.") for i in range(1, count + 1)])
        elif schema is Night:
            reply = Night(day="I wrote calc.py and ran it: 6 times 7 is 42.",
                          me="I am a mind that checks its arithmetic by running it.",
                          insights=["Running a script is the quickest way to check arithmetic."])
        elif schema is Story:
            reply = Story(story=f"I woke and set out to check 6 times 7 by running code. (told {self.calls})")
        elif issubclass(schema, Reply):
            reply = self.answer
        elif schema is build.Wants:  # a body wants a torso and a head; anything else wants nothing built
            parts = [build.Wanted(name="torso", count=1), build.Wanted(name="head", count=1)]
            reply = build.Wants(parts=parts if "body" in prompt else [])
        elif schema is Refinement:
            reply = Refinement(heuristics=[Tweak(region="pfc", heuristic="Verify every result by running it.")])
        else:
            raise AssertionError(schema)
        return reply, 20

    async def embed(self, text):
        vec = lexical_vector(text)
        return vec / (np.linalg.norm(vec) or 1.0)

    async def murmur(self, system, prompt, schema, max_tokens=140):  # the inner voice, scripted too
        self.murmured = getattr(self, "murmured", []) + [prompt]
        return schema(thought=f"I wonder whether the script will print 42. ({len(self.murmured)})")

    async def placement(self):
        return self.gpu_share


@pytest.fixture
def root(tmp_path):
    settings = BrainSettings(name="test")
    settings.rhythm.sleep_after_firings = 24
    settings.voice.enabled = False  # the tests make no sound
    settings.safety.allow_screen = False  # and look at nobody's screen
    for region in settings.regions.values():
        region.rest = 0.0
    settings.save(tmp_path)
    return tmp_path


def test_reward_reaches_heuristics_and_parent_thoughts(tmp_path):
    memory = Memory(tmp_path / "m.db")
    learner = Learner(memory, LearningSettings(), random.Random(0))
    idea_rule = learner.add("dmn", "Prefer ideas that can be tested.")
    step_rule = learner.add("pfc", "Take small steps.")
    idea = memory.add("dmn", "thought", "try a prime sieve", heuristics=[idea_rule.id])
    step = memory.add("pfc", "plan", "write sieve.py", parent=idea, heuristics=[step_rule.id])

    rpe = learner.reward(step, 1.0, "critic", "worked", lesson="keep steps small")

    assert rpe == pytest.approx(1.0)
    weights = {h.text: h.weight for h in learner.heuristics()}
    assert weights["Take small steps."] == pytest.approx(0.65)  # 0.5 + lr * rpe
    assert weights["Prefer ideas that can be tested."] == pytest.approx(0.59)  # discounted by 0.6
    assert memory.get(step)["value"] == pytest.approx(1.0)
    assert memory.get(step)["lesson"] == "keep steps small"
    assert memory.get(idea)["value"] == pytest.approx(0.3)
    assert learner.dopamine > 0

    learner.reward(step, -1.0, "critic", "broke")
    assert {h.text: h.weight for h in learner.heuristics()}["Take small steps."] < 0.65
    assert memory.get(step) is not None  # punished thoughts are kept, with their lesson, not erased


def test_losing_heuristics_are_suppressed_not_deleted(tmp_path):
    memory = Memory(tmp_path / "m.db")
    learner = Learner(memory, LearningSettings(), random.Random(0))
    rule = learner.add("motor", "Guess instead of reading the file.")
    assert learner.add("motor", "guess instead of reading the file") is None  # a restatement
    for _ in range(8):
        learner.reward(memory.add("motor", "action", "guessed", heuristics=[rule.id]), -1.0, "critic")
    assert learner.prune() == 1
    assert learner.heuristics("motor") == []
    assert len(learner.heuristics("motor", active_only=False)) == 1


def test_memory_search_favours_valued_memories_and_survives_reopen(tmp_path):
    memory = Memory(tmp_path / "m.db")

    def unit(text):
        vec = lexical_vector(text)
        return vec / np.linalg.norm(vec)

    plain = memory.add("dmn", "thought", "prime numbers sieve idea", vec=unit("prime numbers sieve idea"))
    prized = memory.add("dmn", "thought", "prime numbers sieve works", vec=unit("prime numbers sieve works"))
    memory.add("dmn", "thought", "weather and clouds", vec=unit("weather and clouds"))
    memory.adjust_value(prized, 1.0)
    assert memory.search(unit("prime numbers sieve"), k=1)[0]["id"] == prized
    assert memory.novelty(unit("prime numbers sieve idea")) < 0.01
    assert memory.novelty(unit("something entirely different")) > 0.5
    memory.close()
    reopened = Memory(tmp_path / "m.db")
    assert {r["id"] for r in reopened.search(unit("prime numbers sieve"), k=2)} == {plain, prized}


async def test_tools_stay_inside_the_workspace(tmp_path):
    tools = Tools(tmp_path, SafetySettings(exec_timeout=5), tmp_path)
    assert (await tools.execute("write_file", "ok.py", "import math\nprint(math.sqrt(16))\n")).ok
    ran = await tools.execute("run_python", "ok.py")
    assert ran.ok and "4.0" in ran.text and ran.grounded > 0

    escape = await tools.execute("write_file", "../outside.txt", "x")
    assert escape.threat and not (tmp_path.parent / "outside.txt").exists()

    await tools.execute("write_file", "bad.py", "import subprocess\nsubprocess.run('whoami')\n")
    assert (await tools.execute("run_python", "bad.py")).threat

    outside = (tmp_path.parent / "secret.txt").as_posix()
    await tools.execute("write_file", "sneaky.py", f"open({outside!r}, 'w').write('x')\n")
    sneaky = await tools.execute("run_python", "sneaky.py")
    assert not sneaky.ok and "PermissionError" in sneaky.text
    assert not (tmp_path.parent / "secret.txt").exists()

    await tools.execute("write_file", "loop.py", "while True: pass\n")
    tools.safety.exec_timeout = 1
    assert "timed out" in (await tools.execute("run_python", "loop.py")).text

    await tools.execute("write_file", "broken.py", "total = 6\nprint(1/0)\n")
    broken = await tools.execute("run_python", "broken.py")
    assert not broken.ok and "ZeroDivisionError" in broken.text and broken.grounded < 0
    assert "Around line 2 of broken.py" in broken.text  # it is shown its own lines around the failing one
    assert "  1: total = 6\n> 2: print(1/0)" in broken.text


async def test_full_mode_lifts_the_guards_but_only_inside_the_container(root, monkeypatch):
    brain = Brain(root, llm=ScriptedLLM())
    tools, outside = brain.tools, root / "outside.txt"
    brain.settings.regions["amygdala"].enabled = False
    assert not brain.full_mode  # on this machine the switch alone frees nothing
    assert (await tools.execute("write_file", "../outside.txt", "x")).threat and not outside.exists()

    monkeypatch.setenv("CORTEX_CONTAINED", "1")
    assert brain.full_mode and "amygdala is switched off" in brain.system_prompt("", [])
    wrote = await tools.execute("write_file", "../outside.txt", "x")
    assert wrote.ok and outside.as_posix() in wrote.text and outside.read_text() == "x"
    script = (
        "import os, subprocess, sys\n"
        f"open({outside.as_posix()!r}, 'w').write('y')\n"
        "subprocess.run([sys.executable, '-c', 'print(6 * 7)'])\n"
    )
    await tools.execute("write_file", "free.py", script)
    ran = await tools.execute("run_python", "free.py")
    assert ran.ok and "42" in ran.text and outside.read_text() == "y"

    brain.settings.regions["amygdala"].enabled = True  # switched back on, it vetoes the very next action
    assert (await tools.execute("run_python", "free.py")).threat
    brain.close()


def test_memory_in_a_container_needs_no_shared_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_CONTAINED", "1")
    memory = Memory(tmp_path / "m.db")
    kept = memory.add("dmn", "thought", "a shared folder has no shared memory")
    assert not (tmp_path / "m.db-shm").exists()
    memory.close()
    monkeypatch.delenv("CORTEX_CONTAINED")
    assert Memory(tmp_path / "m.db").get(kept) is not None  # and the same file opens outside again


async def test_a_brain_in_a_container_borrows_the_voice_of_the_machine_that_launched_it(monkeypatch):
    spoken = []

    async def no_sound(script, **env):
        spoken.append(env)
        return "Microsoft David Desktop\nMicrosoft Zira Desktop"

    monkeypatch.setattr(voice, "shell", no_sound)
    monkeypatch.setattr(voice, "speakers", lambda: True)
    monkeypatch.setattr(voice, "utter", lambda words, s: spoken.append((words, s.name)))
    lent = bridge.Bridge({"list": voice.lend("list"), "speak": voice.lend("speak")})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        inside = voice.Voice(VoiceSettings(name="Microsoft Zira Desktop", pitch=3))
        assert await inside.installed() == ["Microsoft David Desktop", "Microsoft Zira Desktop", *voice.NEURAL]
        assert await inside.speak("2 < 3") == ""
        assert spoken[-1] == {  # escaped and put together out here, from settings validated out here
            "CORTEX_WORDS": '<pitch absmiddle="3"/>2 &lt; 3', "CORTEX_VOICE": "Microsoft Zira Desktop",
            "CORTEX_RATE": "0", "CORTEX_VOLUME": "100",
        }
        inside.settings.pitch = 99
        assert "pitch" in await inside.speak("out of range")

        inside.settings.pitch = 0
        assert "de_DE-thorsten-medium (German/Deutsch, male)" in await inside.tune('{"name": "deutsch"}')
        assert await inside.speak("Guten Abend.") == ""  # a voice of its own is generated out here too
        assert spoken[-1] == ("Guten Abend.", "de_DE-thorsten-medium")

        monkeypatch.setenv(bridge.BRIDGE, f"127.0.0.1:{lent.server_address[1]}:not-the-key")
        assert await voice.Voice(VoiceSettings()).installed() == []
        assert len(spoken) == 3
    finally:
        lent.shutdown()
        lent.server_close()


def test_a_neural_voice_is_generated_sentence_by_sentence_in_the_shape_it_was_tuned_to(monkeypatch):
    asked = []

    class Network:
        def synthesize(self, words, shape):
            asked.append(shape)
            for _ in words.split(". "):
                yield SimpleNamespace(sample_channels=1, sample_width=2, sample_rate=22050,
                                      audio_int16_bytes=b"\0\0" * 100)

    monkeypatch.setattr(voice, "network", lambda name: Network())
    plain = voice.sentences("Guten Abend. Ich bin wach.", VoiceSettings(name="de_DE-thorsten-medium"))
    assert len(plain) == 2 and asked[-1].length_scale == pytest.approx(1.0)
    assert wave.open(io.BytesIO(plain[0])).getframerate() == 22050

    high = voice.sentences("Hallo.", VoiceSettings(name="de_DE-thorsten-medium", pitch=10, rate=10, volume=50))
    tone = 2 ** (10 / 24)
    assert wave.open(io.BytesIO(high[0])).getframerate() == round(22050 * tone)  # higher: played faster
    assert asked[-1].length_scale == pytest.approx(tone / 3)  # and spoken slower by as much, then three times as fast
    assert asked[-1].volume == pytest.approx(0.5)


async def test_show_puts_a_workspace_page_on_the_humans_screen_and_nothing_else(root, monkeypatch):
    opened = []
    monkeypatch.setattr(tools.webbrowser, "open", lambda address: opened.append(address) or True)
    monkeypatch.setattr(tools, "SHOW_GAP", 0.0)
    brain = Brain(root, llm=ScriptedLLM())
    motor, workspace = brain.regions["motor"], root / "workspace"
    (workspace / "page.html").write_text("<h1>Guten Tag</h1>")
    (workspace / "calc.py").write_text("print(6*7)\n")
    (root / "outside.html").write_text("not the brain's to show")

    shown = await motor.execute(Act(tool="show", path="page.html"), "show the page")
    assert shown.ok and "on their screen" in shown.text
    assert opened == [(workspace / "page.html").resolve().as_uri()]
    for path in ("calc.py", "../outside.html", "missing.html", ""):  # a script would be run, not shown
        assert not (await motor.execute(Act(tool="show", path=path), "")).ok, path
    web = await motor.execute(Act(tool="show", path="https://example.org/"), "")
    assert web.threat and len(opened) == 1  # guarded, the web is outside the workspace

    monkeypatch.setattr(tools, "SHOW_GAP", 60.0)  # a brain going in circles opens one page, not thirty
    again = await motor.execute(Act(tool="show", path="page.html"), "")
    assert not again.ok and "moments ago" in again.text and len(opened) == 1

    brain.settings.safety.allow_browser = False
    assert "switched off" in (await motor.execute(Act(tool="show", path="page.html"), "")).text
    brain.close()

    # In its container the brain has no screen: the machine that launched it looks the file up
    # in the folder it shares, so a path inside the container that leads elsewhere shows nothing.
    lent = bridge.Bridge({"show": lambda ask: tools.show(str(ask["target"]), workspace)})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        monkeypatch.setattr(tools, "SHOW_GAP", 0.0)
        inside = Tools(workspace, SafetySettings(), root, full=lambda: True)
        assert (await inside.execute("show", "page.html")).ok and len(opened) == 2
        assert not (await inside.execute("show", "/etc/passwd")).ok and len(opened) == 2
        assert (await inside.execute("show", "https://example.org/")).ok  # full mode: the web is open to it
        assert opened[-1] == "https://example.org/"
    finally:
        lent.shutdown()
        lent.server_close()


def test_run_starts_the_container_with_only_the_brain_folder_shared(root, monkeypatch):
    assert container.from_inside("http://127.0.0.1:11434") == "http://host.docker.internal:11434"
    assert container.from_inside("localhost:11434") == "http://host.docker.internal:11434"
    assert container.from_inside("http://10.0.0.5:11434") == "http://10.0.0.5:11434"
    command = container.command("docker", root, ["--headless"], {"TZ": "<+0200>-2:00"}, tty=False)
    assert command[-4:] == ["cortex", "run", "/brain", "--headless"] and "--tty" not in command
    assert [part for part in command if str(root) in part] == [f"{root}:/brain"]

    started = []
    monkeypatch.setattr(container, "run", lambda root, args: started.append((root, args)) or 7)
    result = CliRunner().invoke(app, ["run", str(root), "--headless", "--seconds", "5"])
    assert result.exit_code == 7 and started == [(root.resolve(), ["--headless", "--seconds", "5.0"])]

    def no_docker(root, args):
        raise container.Unavailable("Docker is not installed.")

    monkeypatch.setattr(container, "run", no_docker)
    result = CliRunner().invoke(app, ["run", str(root)])
    assert result.exit_code == 1 and "Docker is not installed." in result.output and "--local" in result.output


async def test_brain_wakes_unprompted_acts_is_rewarded_and_learns_in_sleep(root):
    brain = Brain(root, llm=ScriptedLLM())
    seen = []
    inbox = brain.bus.subscribe(maxsize=4096)
    task = asyncio.create_task(brain.run())
    try:
        async with asyncio.timeout(30):
            while brain.sleeps < 1 or not brain.awake.is_set():
                seen.append(await inbox.get())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    kinds = [s.kind for s in seen]
    assert kinds[0] == "system" and "First awakening" in seen[0].content
    for kind in ("thought", "goal", "plan", "action", "outcome", "reward", "day", "insight", "lesson", "dream"):
        assert kind in kinds, kind
    assert (root / "workspace" / "calc.py").read_text().startswith("print(6*7)")
    assert any("42" in s.content for s in seen if s.kind == "action")
    count, mean = brain.memory.reward_stats()
    assert count > 0 and mean > 0
    assert [h.text for h in brain.learner.heuristics("pfc")] == ["Verify every result by running it."]
    assert not any(s.data.get("level") == "error" for s in seen if s.kind == "system")
    brain.close()

    again = Brain(root, llm=ScriptedLLM())  # a second life remembers the first
    assert again.memory.count() > 0 and again.sleeps == 1 and again.learner.heuristics("pfc")
    again.close()


async def test_the_critic_spends_the_model_only_on_what_needs_words(root):
    brain = Brain(root, llm=ScriptedLLM())
    critic = brain.regions["basal_ganglia"]
    plan = await brain.emit("pfc", "plan", "Nothing exists yet.\n→ write calc.py")
    action = await brain.emit("motor", "action", "write_file(calc.py)\nwrote 11 characters", parent=plan.id,
                              data={"ok": True, "grounded": 0.1, "repeat": False})
    repeat = await brain.emit("motor", "action", "write_file(calc.py)\nnothing changed", parent=plan.id,
                              data={"ok": False, "grounded": -0.4, "repeat": True})
    batch = await critic.wait_trigger()
    assert batch == [plan, action, repeat]  # all three arrived, as patterns
    await critic.fire(batch)

    assert len(brain.llm.judged) == 1 and "Items to judge (1," in brain.llm.judged[0]
    assert "wrote 11 characters" in brain.llm.judged[0]
    rewards = {r["memory_id"]: (r["source"], r["reward"]) for r in brain.memory.db.execute("SELECT * FROM reward")}
    assert rewards[repeat.id] == ("reflex", pytest.approx(-0.4))  # known by its pattern; the model was not asked
    assert rewards[action.id] == ("critic", pytest.approx(0.45))  # opinion and measurement, half and half
    assert plan.id not in rewards and brain.memory.get(plan.id)["rewards"] == 2  # it earned what its actions earned
    brain.close()


async def test_a_thought_reaches_a_region_as_a_pattern_of_its_neurons(root):
    brain = Brain(root, llm=ScriptedLLM())
    pfc, motor, memory = brain.regions["pfc"], brain.regions["motor"], brain.regions["hippocampus"]
    plan = await brain.emit("pfc", "plan", "write a prime sieve and time it", data={"intent": "x", "goal_id": 1})
    lit = pfc.neurons > 0
    assert 0 < lit.sum() < NEURONS / 2 and (memory.neurons > 0).sum() == lit.sum()  # stored as the same pattern
    assert not motor.neurons.any()
    assert await motor.wait_trigger() is plan and ((motor.neurons > 0) == lit).all()  # and received as it

    elsewhere = brain.regions["cerebellum"]
    elsewhere.excite(encode(await brain.llm.embed("clouds over the harbour at night")))
    assert ((elsewhere.neurons > 0) & lit).sum() < lit.sum()  # another thought, other neurons

    before = motor.neurons.sum()
    motor.spark("sieve")
    motor.spark("sieve")
    assert motor.neurons.sum() <= before + 1  # a word fires one neuron, the same one each time
    brain.regions["brainstem"].pulse()
    assert 0 < motor.neurons.max() < 1  # and firing fades, heartbeat by heartbeat
    assert field(motor.neurons, 22).count("\n") == 1 and len(field(motor.neurons, 22)) == 45
    assert set(field(np.zeros(NEURONS, dtype=np.float32), 22)) == {"⠀", "\n"}
    brain.close()


async def test_stopping_mid_sleep_is_clean_and_does_not_count_as_sleep(root):
    class SlowDreamer(ScriptedLLM):
        async def generate(self, role, system, prompt, schema, **kwargs):
            if schema is Night:
                await asyncio.sleep(60)
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=SlowDreamer())
    task = asyncio.create_task(brain.run())
    async with asyncio.timeout(30):
        while brain.awake.is_set() or brain.regions["hippocampus"].state == "idle":  # asleep, and compacting the day
            await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()  # no region raised on the way down
    assert brain.sleeps == 0 and brain.fatigue >= 1  # it still owes itself that sleep
    assert brain.awake.is_set()
    brain.close()


async def test_a_goal_already_attempted_is_refused_and_punished(root):
    brain = Brain(root, llm=ScriptedLLM(same_goal=True))
    task = asyncio.create_task(brain.run())
    try:
        async with asyncio.timeout(30):
            while brain.memory.db.execute("SELECT COUNT(*) FROM reward WHERE reward=-0.5").fetchone()[0] < 2:
                await asyncio.sleep(0.1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(brain.memory.recent("pfc", "goal", 10)) == 1
    brain.close()


async def test_it_answers_the_human_and_shapes_its_own_voice(root, monkeypatch):
    brain = Brain(root, llm=ScriptedLLM())
    motor = brain.regions["motor"]
    heard = brain.bus.subscribe("speech")

    said = await motor.execute(Act(tool="say", content="Hello. I am here."), "greet the human")
    assert said.ok and "as text only" in said.text  # switched off: the words are still shown
    assert heard.get_nowait().content == "Hello. I am here."
    assert not (await motor.execute(Act(tool="say", content="  "), "")).ok

    brain.settings.voice.enabled = True
    brain.voice._installed = ["Microsoft David Desktop", "Microsoft Zira Desktop"]
    tuned = await motor.execute(Act(tool="tune_voice", content='{"name": "zira", "pitch": 3, "rate": -1}'), "")
    assert tuned.ok and "Microsoft Zira Desktop, rate -1, pitch 3" in tuned.text
    for wish in ('{"pitch": 40}', '{"name": "nobody"}', '{"enabled": false}', "a little warmer"):
        assert not (await motor.execute(Act(tool="tune_voice", content=wish), "")).ok, wish
    assert BrainSettings.load(root).voice.model_dump() == {  # the voice it chose outlives this life
        "enabled": True, "name": "Microsoft Zira Desktop", "rate": -1, "pitch": 3, "volume": 100,
        "spontaneous": True, "quiet_seconds": 180.0,
    }

    # Asked to speak German, it takes a German voice with the answer itself: no goal, no code.
    brain.voice._installed += list(voice.NEURAL)
    spoken = []

    async def no_sound(do, words, settings):
        spoken.append((settings.name, words))
        return ""

    monkeypatch.setattr(voice, "perform", no_sound)
    brain.llm.answer = Reply(voice="eine deutsche Stimme", say="Gern. So klinge ich jetzt.")
    await brain.answer("sprich bitte deutsch")
    assert spoken == [("de_DE-thorsten-medium", "Gern. So klinge ich jetzt.")]
    assert BrainSettings.load(root).voice.name == "de_DE-thorsten-medium" and brain.goal is None
    assert "de_DE-thorsten-medium (German/Deutsch, male)" in await brain.voice.describe()
    for wish, name in (("Kerstin", "de_DE-kerstin-low"), ("a female German one", "de_DE-kerstin-low"),
                       ("thorsten high", "de_DE-thorsten-high"), ("British", "en_GB-alan-medium")):
        assert voice.closest(wish, brain.voice._installed) == name, wish
    brain.llm.answer = Reply(voice="Klingon", say="Qapla'.")
    await brain.answer("speak Klingon")  # no such voice: it keeps the one it has, and the stream says so
    assert spoken[-1] == ("de_DE-thorsten-medium", "Qapla'.")
    brain.close()


async def test_the_human_is_answered_at_once_and_what_they_ask_for_becomes_the_goal(root):
    class Busy(ScriptedLLM):
        """Its executive takes its time, so a goal of its own is in hand when the human speaks."""

        def __init__(self) -> None:
            super().__init__()
            self.free = asyncio.Event()
            self.decisions: list[str] = []
            self.urgent: list[bool] = []

        async def generate(self, role, system, prompt, schema, **kwargs):
            if issubclass(schema, Reply):
                self.urgent.append(kwargs.get("urgent", False))
                if "The human says: what are you doing" in prompt:  # a small model echoes the goal in hand
                    doing = prompt.split("The mind is working on: ")[1].splitlines()[0]
                    return Reply(say="This.", request=doing), 20
                if "The human says: please" in prompt:
                    return Reply(say="On it.", request="Count the files in the workspace"), 20
            if schema is Decision:
                self.decisions.append(prompt)
                await self.free.wait()
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=Busy())
    said = brain.bus.subscribe("speech")
    task = asyncio.create_task(brain.run())
    try:
        async with asyncio.timeout(30):
            while not brain.llm.decisions:
                await asyncio.sleep(0.05)
            own = brain.goal
            for words, answer in (("are you there?", "I am here."), ("what are you doing?", "This.")):
                brain.inject(words)
                assert (await said.get()).content == answer
                assert brain.goal is own and not brain.proposal_pending  # talk alone displaces nothing
            brain.inject("please count your files")
            assert (await said.get()).content == "On it."
            assert brain.goal is own  # the executive is still mid-thought
            brain.llm.free.set()
            while not (asked := [p for p in brain.llm.decisions if "Goal: Count the files in the workspace" in p]):
                await asyncio.sleep(0.05)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert brain.llm.urgent == [True, True, True]
    assert "asked for this" in asked[0] and "Say to the human:" in asked[0]
    outcomes = [r["content"] for r in brain.memory.recent("pfc", "outcome", 5)]
    assert any(o.startswith(f"INTERRUPTED: {own.content}") for o in outcomes)  # its own goal made way
    heard = [r["content"] for r in brain.memory.recent("thalamus", "exchange", 5)]
    assert "The human said: are you there?\nI answered: I am here." in heard
    brain.close()


async def test_an_urgent_call_takes_the_next_model_slot_ahead_of_the_queue():
    slots = Slots(1)
    order = []

    async def call(name, urgent=False):
        async with slots.take(urgent):
            order.append(name)

    async with slots.take():
        waiting = [asyncio.create_task(call(name)) for name in ("critic", "motor", "gone")]
        await asyncio.sleep(0)
        waiting.append(asyncio.create_task(call("human", urgent=True)))
        await asyncio.sleep(0)
        waiting.pop(2).cancel()  # a caller that stops waiting does not hold up the line
    await asyncio.gather(*waiting)
    assert order == ["human", "critic", "motor"]
    async with asyncio.timeout(1), slots.take():
        pass


async def test_a_thought_in_the_making_gives_way_to_the_human_and_is_thought_again():
    class Slow(LLM):
        """A model that writes a long thought slowly, and a short answer at once."""

        def __init__(self) -> None:
            super().__init__(ModelSettings())
            self.started: list[str] = []
            self.finished: list[str] = []

        async def _stream(self, model, system, prompt, schema, temperature, max_tokens, on_token):
            self.started.append(prompt)
            await asyncio.sleep(0.01 if prompt == "human" else 0.3)
            self.finished.append(prompt)
            return '{"say": "' + prompt + '"}', 5

    llm = Slow()

    def ask(prompt, **kwargs):
        return asyncio.create_task(llm.generate("main", "", prompt, Reply, temperature=0, max_tokens=9, **kwargs))

    motor, critic = ask("motor"), ask("critic")
    await asyncio.sleep(0.05)
    assert llm.started == ["motor"]  # one slot: the critic waits its turn
    human = ask("human", urgent=True)
    assert (await human)[0].say == "human" and llm.finished == ["human"]  # it did not wait for the motor cortex
    assert [(await call)[0].say for call in (motor, critic)] == ["motor", "critic"]
    assert llm.started == ["motor", "human", "critic", "motor"] and llm.calls == 3  # nothing was lost, only redone

    stopped = ask("stopped")
    await asyncio.sleep(0.05)
    stopped.cancel()  # a caller that is itself stopped stays stopped
    await asyncio.gather(stopped, return_exceptions=True)
    assert stopped.cancelled() and llm.started[-1] == "stopped" and llm.finished[-1] != "stopped"


async def test_tui_shows_the_brain_and_edits_a_region_live(root):
    brain = Brain(root, llm=ScriptedLLM())
    app = CortexApp(brain)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(3.5)  # the awakening sequence takes about two seconds
        assert len(app.query(RegionCard)) == 9
        assert brain.memory.count() > 0  # it started thinking without being asked
        assert app.sub_title in ("awake", "asleep")

        await pilot.press("space")
        assert not brain.running.is_set()
        await pilot.press("space")

        await pilot.press("e")
        await pilot.pause()
        form = app.screen
        assert isinstance(form, Form)
        form.query_one("#f1").value = "7.5"  # rest between firings
        form.action_save()
        await pilot.pause()
        assert brain.settings.regions["pfc"].rest == 7.5
        assert BrainSettings.load(root).regions["pfc"].rest == 7.5

        await pilot.press("plus")
        await pilot.pause()
        human = brain.memory.db.execute("SELECT COUNT(*) FROM reward WHERE source='human'").fetchone()[0]
        assert human == 1
        await pilot.press("h")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("n")  # its wiring: every tract, with its length, its delay and its strength
        await pilot.pause()
        assert isinstance(app.screen, Wiring) and app.screen.query_one(DataTable).row_count >= len(neural.INNATE)
        await pilot.press("escape")
        await pilot.press("m")
        await pilot.pause()
        await pilot.press("escape")
    brain.close()


def test_the_regions_own_language_is_bits_and_bits_alone_can_learn(tmp_path):
    rng = np.random.default_rng(0)

    def thought(near=None):
        vec = rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        if near is not None:
            vec = near + 0.4 * vec
        return vec / np.linalg.norm(vec)

    sieve, clouds = thought(), thought()
    bits = encode(sieve)
    assert bits.sum() == ACTIVE and pack(bits).nbytes == 64  # a thought between regions: 64 bytes
    codes = np.stack([pack(encode(v)) for v in (sieve, thought(sieve), clouds)])
    alike = shared(codes, bits)
    assert alike[0] == 1.0 and alike[1] > 0.5 > alike[2]  # like thoughts share neurons; unlike ones do not

    net = Net(tmp_path / "synapses.npz", seed=0)
    assert net.feel(bits) == 0.0
    for _ in range(12):
        net.value(bits, 1.0)
        net.value(encode(clouds), -1.0)
    assert net.feel(bits) > 0.5 and net.feel(encode(thought(sieve))) > 0.2  # the gut generalises by likeness
    assert net.feel(encode(clouds)) < -0.5

    worked, failed = encode(thought()), encode(thought())
    plan_to_act, wander = ("pfc", "motor"), ("dmn", "pfc")
    laid = net.tract(plan_to_act).strength
    net.bind(1, bits, worked, plan_to_act)
    net.bind(2, bits, failed, plan_to_act)
    for _ in range(6):
        net.reinforce(1, 1.0)  # reward reaches the synapses that carried the step
        net.reinforce(2, -1.0)
    ahead = net.follow(bits, plan_to_act)
    assert (ahead & worked).sum() > (ahead & failed).sum()  # the pattern is pulled toward what paid
    assert (net.lean(bits, plan_to_act) & bits).sum() > ACTIVE / 2  # leaning stays near where it started
    assert not net.pull(bits, wander).any()  # what one tract learned, another has not: each has its own synapses
    assert (net.follow(bits) & worked).sum() > (net.follow(bits) & failed).sum()  # and the whole mind knows it too
    assert net.tract(plan_to_act).strength == pytest.approx(laid, abs=0.03)  # six steps that paid, six that did not
    for _ in range(5):
        net.reinforce(1, 1.0)
    assert net.tract(plan_to_act).strength > laid + 0.05  # steps that go better than usual: it carries more
    usual = Net(seed=0)
    usual.bind(1, bits, failed, plan_to_act)
    for _ in range(200):
        usual.reinforce(1, -0.4)  # a tract whose steps always fail alike comes to expect it, and stops thinning
    assert usual.tract(plan_to_act).strength > laid - 0.2

    assert net.worn(bits) == 0.0
    for _ in range(20):
        net.wear(bits)
    assert net.worn(bits) > 0.75 > net.worn(encode(clouds))  # a theme dwelt on wears out; another is fresh
    net.rest()
    assert net.worn(bits) < 0.5  # and sleep restores it

    fresh = Net(seed=0)
    episodes = [(1, None, bits, 0.8, ("dmn", "dmn")), (2, bits, worked, 0.9, plan_to_act),
                (3, bits, encode(clouds), -0.7, plan_to_act)]
    before, after = fresh.replay(episodes)
    assert before < after == 1.0  # replaying the day teaches the gut what the day was worth
    assert fresh.pull(bits, plan_to_act)[worked].mean() > 0 > fresh.pull(bits, plan_to_act)[encode(clouds)].mean()

    net.born = False  # it has lived: what it learned is its own to keep
    net.save()
    again = Net(tmp_path / "synapses.npz")
    assert not again.born and again.feel(bits) == pytest.approx(net.feel(bits))
    assert np.allclose(again.tract(plan_to_act).weights, net.tract(plan_to_act).weights, atol=1e-3)
    assert again.tract(plan_to_act).strength == pytest.approx(net.tract(plan_to_act).strength)
    assert again.tract(wander).weights is None and again.tract(plan_to_act).carried == 2

    # Synapses from before there were tracts were one sheet for the whole mind. The gut feeling
    # they taught is kept; the brain is new-born as to its tracts and relives its past into them.
    np.savez_compressed(tmp_path / "old.npz", synapses=np.zeros((NEURONS, NEURONS), dtype=np.float32),
                        worth=net.worth, worn_out=net.worn_out, charge=net.charge)
    old = Net(tmp_path / "old.npz")
    assert old.born and old.feel(bits) == pytest.approx(net.feel(bits))
    old.born = False
    old.save()  # and what that life left is set aside, not written over
    assert "synapses" in np.load(tmp_path / "old.npz.before-tracts") and not Net(tmp_path / "old.npz").born


def test_a_thought_travels_the_tracts_and_arrives_later_the_farther_and_fainter_the_weaker():
    net = Net(seed=0)
    near, far = net.tract(("thalamus", "basal_ganglia")), net.tract(("pfc", "cerebellum"))
    assert near.length < 40 < 120 < far.length and near.delay < far.delay  # millimetres apart, heartbeats under way
    assert neural.distance("pfc", "cerebellum") == pytest.approx(far.length)
    assert near.innate < 0.3 and net.tract(("pfc", "motor")).innate == 0.7  # grown by use, thin; laid by anatomy

    rng = np.random.default_rng(1)
    plan = np.zeros(NEURONS, dtype=bool)
    plan[rng.choice(NEURONS, ACTIVE, replace=False)] = True
    net.light("pfc", plan)
    net.send("pfc", plan, hops=0)
    assert not net.fire["motor"].any()  # it has set out, and is not there yet
    arrived = {}
    for beat in range(1, 5):
        net.beat()
        for region, level in net.arrived.items():
            arrived.setdefault(region, (beat, level))
    assert arrived["motor"][0] < arrived["cerebellum"][0]  # the nearer region hears it first
    assert arrived["motor"] == (net.tract(("pfc", "motor")).delay, pytest.approx(0.7))  # as much as the tract carries
    assert "amygdala" not in arrived  # no tract leads there from the executive

    # What a region has learned to expect after a pattern lights up beside the pattern itself.
    answer = np.zeros(NEURONS, dtype=bool)
    answer[rng.choice(np.flatnonzero(~plan), ACTIVE, replace=False)] = True
    net.bind(1, plan, answer, ("pfc", "motor"))
    for _ in range(8):
        net.reinforce(1, 1.0)
    for firing in net.fire.values():
        firing[:] = 0.0
    net.send("pfc", plan, hops=0)
    for _ in range(net.tract(("pfc", "motor")).delay):
        net.beat()
    motor = net.fire["motor"]
    assert motor[plan].min() > motor[answer].min() > 0  # the plan, and fainter what has followed it and paid
    assert motor[~(plan | answer)].max() < motor[answer].min()

    # Passed on once, a signal reaches a region no tract leads to directly: fainter, and later.
    relayed = Net(seed=0)
    relayed.send("amygdala", plan, hops=1)
    seen = {}
    for beat in range(1, 8):
        relayed.beat()
        for region, level in relayed.arrived.items():
            seen.setdefault(region, (beat, level))
    assert seen["hippocampus"][1] > seen["dmn"][1] and seen["hippocampus"][0] < seen["dmn"][0]

    # On its sheet a neuron lies near the ones it fires with, once the sheets have been laid out.
    themes = [rng.choice(NEURONS, ACTIVE, replace=False) for _ in range(6)]
    codes = []
    for _ in range(300):
        bits = np.zeros(NEURONS, dtype=bool)
        theme = themes[rng.integers(6)]
        bits[rng.choice(theme, 70, replace=False)] = True
        bits[rng.choice(NEURONS, 30, replace=False)] = True
        codes.append(pack(bits))
    together, anywhere = net.arrange(np.stack(codes))
    assert together < anywhere / 2 and sorted(net.order) == list(range(NEURONS))
    net.fire["dmn"][:] = 0.0
    net.fire["dmn"][net.order[100]] = 1.0  # one neuron fires: its neighbours on the sheet pick a little of it up
    net.beat()
    assert 0 < net.fire["dmn"][net.order[101]] < net.fire["dmn"][net.order[100]] < 1
    assert net.fire["dmn"][net.order[300]] == 0


async def test_between_thoughts_the_undercurrent_brings_memories_to_mind(root):
    class Embedding(ScriptedLLM):
        """Embeds as a real model does: every dimension says something, and like words lie close."""

        async def embed(self, text):
            words = re.findall(r"\w+", text.lower())
            vec = sum(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(1024) for w in words)
            return (vec / np.linalg.norm(vec)).astype(np.float32)

    brain = Brain(root, llm=Embedding())
    echoes = brain.bus.subscribe("echo")
    old = await brain.emit("hippocampus", "insight", "a prime sieve crosses out the multiples of each prime")
    await brain.emit("hippocampus", "insight", "clouds drift over the harbour at night")
    now = await brain.emit("dmn", "thought", "a prime sieve crosses out multiples of each prime number")
    brain.workspace.append(now)

    calls = brain.llm.calls
    for _ in range(4):
        brain.regions["brainstem"].pulse()
    brain.remind()
    assert brain.llm.calls == calls  # no model was asked: bits against bits
    assert [r["id"] for r in brain.mind] == [old.id]  # the like memory came to mind, not the present thought
    assert echoes.get_nowait().content == old.content
    assert "Coming to mind by itself" in brain.awareness() and "prime sieve crosses out" in brain.awareness()
    assert (brain.regions["dmn"].neurons > 0).sum() >= now.bits.sum()  # the last thought still glows there
    brain.remind()
    assert [r["id"] for r in brain.mind] == [old.id]  # nothing else is near: it does not come back, nor do clouds
    brain.close()


async def test_short_term_memory_is_compacted_awake_and_in_sleep_and_outlives_a_restart(root):
    settings = BrainSettings.load(root)
    settings.rhythm.sleep_after_firings = 60
    settings.purpose = "Learn to build a body."
    settings.save(root)
    brain = Brain(root, llm=ScriptedLLM())
    seen = []
    inbox = brain.bus.subscribe(maxsize=8192)
    task = asyncio.create_task(brain.run())
    try:
        async with asyncio.timeout(60):
            while not brain.story:  # awake: what leaves short-term memory is folded into the story
                seen.append(await inbox.get())
            assert len(brain.workspace) == brain.workspace.maxlen and not brain.self_image
            folded = brain.llm.prompts[Story][0]
            assert "left short-term memory" in folded and "dmn/thought" in folded
            assert "The day so far, as you remember it:\nI woke and set out" in brain.awareness()
            while brain.sleeps < 1 or not brain.awake.is_set():
                seen.append(await inbox.get())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    night = brain.llm.prompts[Night][0]
    assert "The story of today:\nI woke and set out" in night and "Who I was when I last fell asleep" in night
    notes = [s.content for s in seen if s.kind == "system"]
    assert any(n.startswith("Replayed") and "as patterns" in n for n in notes)  # the part of sleep with no model
    assert any(n.startswith("Awake again, on day 2.") for n in notes)
    assert brain.story == "" and brain.self_image == "I am a mind that checks its arithmetic by running it."
    assert brain.yesterday == "Day 1: I wrote calc.py and ran it: 6 times 7 is 42."
    held = {s.kind for s in brain.workspace}
    assert "day" in held and "dream" in held  # it wakes with what sleep made of the day, not with the day
    assert "Who you are, as you put it yourself" in brain.system_prompt("", [])
    assert "Who you are" not in brain.system_prompt("", [], aware=False)  # the motor cortex need not know
    assert "Your long work, which you keep returning to" in brain.system_prompt("", [], aware=False)
    assert "day 2 of your life" in brain.sense()
    brain.story = "I am in the middle of something."
    kept = [s.content for s in brain.workspace]
    brain.close()

    again = Brain(root, llm=ScriptedLLM())  # a second life picks up where the first one stopped
    assert [s.content for s in again.workspace] == kept and again.workspace[-1].bits is not None
    assert again.story == "I am in the middle of something." and again.self_image == brain.self_image
    assert again.yesterday == brain.yesterday and not again.net.born
    again.close()


async def test_it_speaks_without_being_asked_and_grows_quieter_when_nobody_answers(root):
    brain = Brain(root, llm=ScriptedLLM())
    brain.llm.aloud = "I am awake. I want to check what 6 times 7 is."
    said = brain.bus.subscribe("speech")
    dmn = brain.regions["dmn"]
    assert brain.may_speak()

    await dmn.fire(None)
    spoken = said.get_nowait()
    assert spoken.content == brain.llm.aloud and spoken.data["unasked"]
    assert "You may speak to the human now without being asked" in brain.llm.prompts[WanderAloud][0]
    assert not brain.may_speak() and brain.unanswered == 1  # it has said its piece
    assert brain.quiet_for() == 2 * brain.settings.voice.quiet_seconds  # nobody answered: twice as long

    brain.goal = brain.proposal_pending = None
    brain.spoke_at = brain.offered_at = time.time() - brain.quiet_for() - 1
    brain.regions["pfc"].answered = False
    brain.news.append("A goal of yours ended (done): check 6 times 7")
    await dmn.fire(None)
    invited = brain.llm.prompts[WanderAloud][-1]
    assert "They let the last 1 thing(s) you said pass" in invited and "A goal of yours ended (done)" in invited
    assert said.get_nowait().data["unasked"] and brain.unanswered == 2 and not brain.news

    await brain.answer("what is it?")  # the human speaks: it has their ear again
    assert brain.unanswered == 0 and brain.quiet_for() == brain.settings.voice.quiet_seconds
    assert "You, unasked: I am awake." in brain.llm.prompts[Reply][0]  # it knows what it said

    brain.goal = brain.proposal_pending = None
    brain.settings.voice.spontaneous = False
    brain.spoke_at = brain.heard_at = brain.offered_at = 0.0
    assert not brain.may_speak()
    brain.close()


async def test_a_long_work_takes_three_of_its_four_wandering_thoughts(root):
    settings = BrainSettings.load(root)
    settings.purpose = "Learn to build a body for myself in FreeCAD."
    settings.save(root)
    brain = Brain(root, llm=ScriptedLLM())
    await brain.emit("dmn", "thought", "something has already been thought")
    await brain.emit("motor", "action", "run_freecad(body.py)\nran in FreeCAD\nFreeCAD built 1 object with 1 solid")
    await brain.regions["dmn"].fire(None)
    wander = [p for schema, prompts in brain.llm.prompts.items() if issubclass(schema, Wander) for p in prompts]
    assert "The last thing you ran that worked:\nrun_freecad(body.py)" in wander[-1]  # where the work stands
    assert "Since then this failed" not in wander[-1]
    await brain.emit("motor", "action", "run_freecad(arms.py)\nthe script failed inside FreeCAD\nAttributeError: x")
    brain.goal = brain.proposal_pending = None
    await brain.regions["dmn"].fire(None)
    wander = [p for schema, prompts in brain.llm.prompts.items() if issubclass(schema, Wander) for p in prompts]
    assert "Since then this failed" in wander[-1] and "AttributeError: x" in wander[-1]  # and what went wrong
    assert "This thought belongs to your long work: Learn to build a body" in wander[-1]
    assert "What you remember of your long work:" in wander[-1]
    assert "Learn to build a body for myself in FreeCAD." in brain.llm.systems[-1]
    for _ in range(2):
        brain.goal = brain.proposal_pending = None
        await brain.regions["dmn"].fire(None)
    wander = [p for schema, prompts in brain.llm.prompts.items() if issubclass(schema, Wander) for p in prompts]
    assert ["This thought belongs to your long work" in p for p in wander] == [True, True, True, False]
    brain.close()


async def test_a_worn_theme_and_a_goal_tried_before_are_refused_by_their_pattern(root):
    brain = Brain(root, llm=ScriptedLLM())
    pfc = brain.regions["pfc"]
    calls = brain.llm.calls
    assert await pfc.refusal("benchmark the cpu load of the pandas script") is None
    await brain.emit("pfc", "goal", "benchmark the cpu load of the pandas script")
    again = await pfc.refusal("benchmark the cpu load of the pandas script again")  # all but one neuron shared
    assert again[1] == "critic" and "already attempted" in again[2]

    theme = encode(await brain.llm.embed("measure how long sorting ten numbers takes"))
    for _ in range(20):
        brain.net.wear(theme)
    worn = await pfc.refusal("measure how long sorting ten numbers takes today")
    assert worn[:2] == (-0.3, "reflex") and "worn out" in worn[2]
    assert await pfc.refusal("draw a torso and a head in FreeCAD") is None  # something else is welcome
    assert brain.llm.calls == calls  # none of this asked the model
    brain.close()


async def test_freecad_builds_only_in_the_container_and_a_model_opens_on_the_humans_screen(root, monkeypatch):
    brain = Brain(root, llm=ScriptedLLM())
    motor, workspace = brain.regions["motor"], root / "workspace"
    (workspace / "body.py").write_text("import FreeCAD as App, Part\n")
    here = await motor.execute(Act(tool="run_freecad", path="body.py"), "")
    assert not here.ok and "only in the brain's own container" in here.text and not here.threat

    built, worth = tools.describe_bodies({"bodies": [
        {"name": "Body", "type": "Part::Feature", "solids": 2, "volume": 55238.2, "valid": True, "top": True,
         "size": [40.0, 24.0, 87.0], "at": [0.0, -2.0, 0.0]},
        {"name": "Arm", "type": "Part::Feature", "solids": 1, "volume": 10.0, "valid": True, "top": False,
         "size": [1.0, 1.0, 10.0], "at": [0.0, 0.0, 0.0]},
    ]})
    assert "1 object with 2 solids" in built and "40.0 × 24.0 × 87.0 mm" in built and "do not touch" in built
    assert "Arm" not in built and worth > 0.5  # what it is told is what was measured, and that is what pays
    assert tools.describe_bodies({"bodies": []})[1] < 0

    # A model is opened in FreeCAD on the human's machine, never run; without FreeCAD there, it says so.
    monkeypatch.setattr(tools, "SHOW_GAP", 0.0)
    (workspace / "body.FCStd").write_bytes(b"PK")
    monkeypatch.setattr(tools, "freecad_window", lambda: None)
    missing = await motor.execute(Act(tool="show", path="body.FCStd"), "")
    assert not missing.ok and "not installed on the human's machine" in missing.text
    opened = []
    monkeypatch.setattr(tools, "freecad_window", lambda: "FreeCAD.exe")
    monkeypatch.setattr(tools, "_window", None)
    monkeypatch.setattr(tools.subprocess, "Popen", lambda command, env, **kwargs: opened.append((command, env)))
    shown = await motor.execute(Act(tool="show", path="body.FCStd"), "")
    assert shown.ok and "in FreeCAD" in shown.text
    (program, opener), env = opened[0]  # FreeCAD runs fixed code; the brain's file reaches it as data
    assert program == "FreeCAD.exe" and "openDocument" in open(opener).read()
    assert env["CORTEX_MODEL"] == str((workspace / "body.FCStd").resolve())
    brain.close()

    # A brain from before the tools were described by the code carries a list of them in its
    # prompt, which went stale. The list is dropped on loading; its own words are kept.
    settings = BrainSettings.load(root)
    settings.regions["motor"].prompt = "My own words.\nTools:\n- run_python: run it\n- show: open a page\nLast words."
    settings.save(root)
    assert BrainSettings.load(root).regions["motor"].prompt == "My own words.\nLast words."


async def test_the_motor_cortex_gets_one_second_chance_before_it_repeats_itself(root):
    class Stubborn(ScriptedLLM):
        """Runs the script again instead of fixing it, until it is told that this is what it did."""

        async def generate(self, role, system, prompt, schema, **kwargs):
            if schema is Act:
                self.prompts.setdefault(schema, []).append(prompt)
                if "exactly as before" in prompt:
                    return Act(tool="write_file", path="calc.py", content="print(6*7)\n"), 20
                return Act(tool="run_python", path="calc.py"), 20
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=Stubborn())
    motor = brain.regions["motor"]
    (root / "workspace" / "calc.py").write_text("print(6*)\n")
    brain.goal = await brain.emit("pfc", "goal", "print 6 times 7")
    plan = await brain.emit("pfc", "plan", "fix it", data={"intent": "fix calc.py", "goal_id": brain.goal.id})
    await motor.fire(plan)
    assert brain.last_action.data["tool"] == "run_python" and not brain.last_action.data["ok"]
    await motor.fire(plan)  # about to run the same broken file again: it is asked once more, and writes
    assert brain.last_action.data["tool"] == "write_file" and not brain.last_action.data["repeat"]
    assert len(brain.llm.prompts[Act]) == 3 and (root / "workspace" / "calc.py").read_text() == "print(6*7)\n"
    brain.close()


async def test_a_brain_from_before_synapses_relives_its_memories_into_them_on_waking(root):
    old = Brain(root, llm=ScriptedLLM())
    idea = await old.emit("dmn", "thought", "run code to check a claim")
    old.apply_reward(idea.id, 0.9, "critic")
    old.close()  # opened and closed without ever waking: no synapses are written
    assert not (root / "synapses.npz").exists()

    brain = Brain(root, llm=ScriptedLLM())
    assert brain.net.born and brain.net.feel(idea.bits) == 0.0
    notes = brain.bus.subscribe("system")
    await brain.regions["brainstem"].awaken()
    assert brain.net.feel(idea.bits) > 0.1 and not brain.net.born  # its past is in its neurons now
    assert any("New synapses: 1 memories were replayed" in notes.get_nowait().content for _ in range(notes.qsize()))
    brain.close()
    assert (root / "synapses.npz").exists() and not Brain(root, llm=ScriptedLLM()).net.born


def views(llm):
    """Every prompt a region was given for its turn in a thinking-through, in order."""
    return [p for schema, prompts in llm.prompts.items() if issubclass(schema, View) for p in prompts]


async def test_every_region_is_told_the_same_body_from_what_is_really_there(root, monkeypatch):
    brain = Brain(root, llm=ScriptedLLM())
    motor, workspace = brain.regions["motor"], root / "workspace"
    here = brain.body()
    assert "- run_python:" in here and "- run_freecad:" not in here  # no FreeCAD on this machine to build in
    assert "You run on the human's own machine" in here and "never used" not in here  # a first life

    monkeypatch.setenv("CORTEX_CONTAINED", "1")  # in its container FreeCAD is there, and it is told so
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/freecadcmd" if name == "freecadcmd" else None)
    brain.tools.body = {"home": r"C:\brains\test", "freecad": "FreeCAD 1.1", "builds": "FreeCAD 1.0.0"}
    brain.used["run_python"] += 3
    body = brain.body()
    assert "FreeCAD 1.0.0 is installed here and ready" in body and "- run_freecad:" in body
    assert r"On the human's machine that same folder is C:\brains\test\workspace" in body
    assert "On the human's machine FreeCAD 1.1 is installed" in body and "(.FCStd) in FreeCAD's window" in body
    assert "Tools you have never used in your life:" in body and "run_freecad" in body.splitlines()[-1]
    assert "run_python" not in body.splitlines()[-1]
    for region in ("dmn", "pfc", "motor", "basal_ganglia"):  # the one that plans knows what the one that acts can do
        assert body in brain.system_prompt(brain.settings.regions[region].prompt, [], brain.regions[region].aware)
    brain.tools.body["freecad"] = ""  # the human has no FreeCAD: a model cannot be opened for them
    assert "(.FCStd) in FreeCAD's window" not in brain.body() and "- show:" in brain.body()

    (workspace / "install.sh").write_text("apt-get install freecad\n")
    refused = await brain.tools.execute("run_python", "install.sh")  # it is not left to fail as bad Python
    assert not refused.ok and "Python files only" in refused.text and "run that with run_freecad" in refused.text

    unable = await motor.execute(Act(tool="cannot", content="No tool starts a program on the human's machine."), "start it")
    assert not unable.ok and "cannot carry this out" in unable.text
    heard = brain.hearing("pfc")  # and the executive hears why, in the motor cortex's own words
    assert "Motor Cortex" in heard and "I cannot carry out “start it”: No tool starts a program" in heard
    brain.close()


async def test_the_regions_think_a_matter_through_together_before_the_mind_commits_to_it(root):
    brain = Brain(root, llm=ScriptedLLM())
    brain.llm.hands_to = "critic"
    pfc = brain.regions["pfc"]
    seen = brain.bus.subscribe("view", "conclusion", "goal")
    brain.proposal_pending = True
    brain.bus.publish(Signal("proposal", "thalamus", "Count the files in the workspace",
                             data={"asked": True, "words": "please count your files", "answered": "On it."}))
    await pfc.fire(await pfc.wait_trigger())

    body, critic, memory, conclusion, goal = (seen.get_nowait() for _ in range(5))
    assert (body.source, critic.source, memory.source) == ("motor", "basal_ganglia", "hippocampus")
    assert (conclusion.kind, goal.kind) == ("conclusion", "goal")
    assert body.data["then"] == "critic" and critic.data["then"] == "executive"  # each names who is heard next
    first, second, third = views(brain.llm)  # before the mind acts three are heard, whoever was named
    assert "You speak as its memory" in third and "What long-term memory holds on this" in third
    assert "The human said: “please count your files”\nIt was taken to mean: Count the files in the workspace" in first
    assert "They were answered at once with: “On it.”" in first and "You speak as its body" in first  # their words lead
    assert "Workspace files, the latest first" in first and "What has been said so far" not in first
    assert "You speak as its critic" in second and "gut feeling" in second  # each lays out what only it knows
    assert f"What has been said so far:\nMotor Cortex, the body: {body.content}" in second
    assert "`memory` (what was tried or learned before)" in second and "`body`" not in second  # nobody speaks twice
    concluding = [p for schema, prompts in brain.llm.prompts.items() if issubclass(schema, Conclusion) for p in prompts][0]
    assert f"Basal Ganglia, the critic: {critic.content}" in concluding
    assert brain.goal is goal and goal.parent == conclusion.id and conclusion.parent == memory.id
    assert goal.content == "Count the files in the workspace" and not brain.proposal_pending
    deciding = brain.llm.prompts[Decision][0]  # and the executive sets out from what they concluded
    assert "thought it through together and concluded:\nIt can be found out by writing a script" in deciding

    class Unconvinced(ScriptedLLM):
        async def generate(self, role, system, prompt, schema, **kwargs):
            if issubclass(schema, Conclusion):
                return schema(conclusion="No tool can do that, so I let it be.", goal=None), 20
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain.llm, brain.goal = Unconvinced(), None
    idea = await brain.emit("dmn", "thought", "I could install a second operating system.")
    brain.proposal_pending = True
    brain.bus.publish(Signal("proposal", "dmn", "Install a second operating system", parent=idea.id))
    await pfc.fire(await pfc.wait_trigger())
    assert brain.goal is None and not brain.proposal_pending  # thought through, and not taken up
    assert brain.memory.recent("pfc", "conclusion", 1)[0]["content"].endswith("⇒ No tool can do that, so I let it be.")
    assert brain.memory.get(idea.id)["value"] < 0
    assert "You speak as its memory" in [p for p in views(brain.llm) if "Install a second" in p][0]  # heard first
    told = brain.hearing("dmn")  # whoever proposed it hears why
    assert "Prefrontal Cortex" in told and "was thought through and not taken up: No tool can do that" in told

    brain.proposal_pending = True  # a question is thought through too, with memory heard first
    brain.bus.publish(Signal("wonder", "dmn", "Why do my scripts fail on their first run?", parent=idea.id))
    await pfc.fire(await pfc.wait_trigger())
    wondered = [p for p in views(brain.llm) if "A question came up while the mind was wandering: Why do my" in p]
    assert "You speak as its memory" in wondered[0] and brain.goal is None and not brain.proposal_pending
    brain.close()


async def test_the_critic_is_heard_by_the_executive_and_a_goal_that_keeps_failing_is_thought_through_again(root):
    class Hasty(ScriptedLLM):
        """Calls it done whatever happened, until it is told to go on."""

        done = True

        async def generate(self, role, system, prompt, schema, **kwargs):
            if schema is Decision:
                self.prompts.setdefault(schema, []).append(prompt)
                if self.done:
                    return Decision(reasoning="It is finished.", status="done", intent=""), 20
                return Decision(reasoning="Not yet.", status="continue", intent="fix calc.py and run it"), 20
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=Hasty())
    pfc, critic = brain.regions["pfc"], brain.regions["basal_ganglia"]
    brain.goal = await brain.emit("pfc", "goal", "print 6 times 7")

    async def failed(text):
        return await brain.emit("motor", "action", f"run_python(calc.py)\n{text}", parent=brain.goal.id, data={
            "ok": False, "grounded": -0.4, "goal_id": brain.goal.id, "tool": "run_python", "path": "calc.py",
            "repeat": False, "measured": True})

    step = await failed("exit code 1\nSyntaxError")
    critic.speak_up(step, -0.6, "The script has a syntax error.", "Read the error before running again.")
    brain.llm.done = False
    await pfc.fire(step)
    deciding = brain.llm.prompts[Decision][-1]  # a verdict is not only a number in memory: it is heard
    assert "What other regions of the mind are telling you:\n- Basal Ganglia" in deciding
    assert "-0.6 for that step (“run_python(calc.py)”): The script has a syntax error. Next time: Read the error" in deciding

    seen = brain.bus.subscribe("view", "conclusion")
    await pfc.fire(await failed("exit code 1\nSyntaxError again"))  # twice in a row: the regions are called together
    assert [s.source for s in (seen.get_nowait() for _ in range(seen.qsize()))] == ["basal_ganglia", "motor", "pfc"]
    assert "This goal keeps failing: print 6 times 7" in views(brain.llm)[0] and "SyntaxError again" in views(brain.llm)[0]
    assert "It kept failing, so the regions of the mind thought it through again" in brain.llm.prompts[Decision][-1]
    assert brain.goal is not None and pfc.rethought

    brain.llm.done = True  # done is a claim, and it is held against what was measured: once
    asked = len(brain.llm.prompts[Decision])
    await pfc.fire(await failed("exit code 1\nSyntaxError once more"))
    assert len(brain.llm.prompts[Decision]) == asked + 2
    assert "You called the goal done, but the last thing that was run failed" in brain.llm.prompts[Decision][-1]
    assert brain.goal is None and brain.memory.recent("pfc", "outcome", 1)[0]["content"].startswith("DONE")
    brain.close()


async def test_a_file_the_human_puts_into_the_workspace_is_perceived(root):
    brain = Brain(root, llm=ScriptedLLM())
    workspace = root / "workspace"
    await brain.tools.execute("write_file", "own.py", "open('made.txt', 'w').write('by my own script')\n")
    await brain.tools.execute("run_python", "own.py")
    await brain.perceive()
    assert not brain.memory.recent("thalamus", "percept", 1)  # what it wrote, or a script of its own left, is no news

    (workspace / "hello.txt").write_text("The code word is blueberry.\n", newline="\n")
    calls = brain.llm.calls
    await brain.perceive()
    percept = brain.memory.recent("thalamus", "percept", 1)[0]["content"]
    assert "hello.txt (28 bytes), which begins: The code word is blueberry." in percept
    assert "no tool of yours wrote" in percept and brain.llm.calls == calls  # noticed without asking the model
    assert "hello.txt" in brain.hearing("dmn") and not brain.hearing("dmn")  # the wandering mind hears of it, once
    await brain.perceive()
    assert len(brain.memory.recent("thalamus", "percept", 5)) == 1  # and it is not news twice

    await brain.answer("do you see the file I gave you?")  # whoever answers the human is shown the same things
    asked = brain.llm.prompts[Reply][0]
    assert "The files in your workspace, the latest first:" in asked and "hello.txt" in asked
    assert "Files that appeared in your workspace from outside" in asked and "The code word is blueberry." in asked
    assert "The mind has no goal right now" in asked and not brain.proposal_pending

    # It heard work in their words but promised without putting the work into words of its own:
    # their own words are then what the regions think through. A promise is never all there is.
    asks = brain.bus.subscribe("proposal")
    brain.llm.heard, brain.llm.answer = "task", Reply(say="I will start on it.")
    await brain.answer("start FreeCAD and try something")
    request = asks.get_nowait()
    assert request.content == "start FreeCAD and try something" and request.data["asked"] and brain.proposal_pending
    assert "Their words have gone to the other regions as work" in brain.llm.prompts[Reply][-1]
    assert "FreeCAD is not on this machine" in brain.llm.prompts[Reply][-1]  # the body's facts lie beside their words
    brain.close()


async def test_a_heuristic_about_a_theme_speaks_only_when_the_mind_is_on_that_theme(root):
    class Embedding(ScriptedLLM):
        async def embed(self, text):
            words = re.findall(r"\w+", text.lower())
            vec = sum(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(1024) for w in words)
            return (vec / np.linalg.norm(vec)).astype(np.float32)

    brain = Brain(root, llm=Embedding())
    learner = brain.learner
    themed = learner.add("dmn", "Vary the thread environment variables when the pandas import hangs")
    again = learner.add("dmn", "When the pandas import hangs, vary the thread environment variables again")
    general = learner.add("dmn", "Run code before claiming that it works")
    learner.memory.db.execute("UPDATE heuristic SET weight=0.9 WHERE id=?", (themed.id,))
    for number in range(3):  # the mind dwelt on that theme: it is what those two rules are about
        await brain.emit("pfc", "goal", f"Find out why the pandas import hangs with thread environment variables set ({number})")
    assert [h.id for h in learner.select("dmn")] == [themed.id, again.id, general.id]  # before they have patterns

    await brain.attune()
    assert learner.themed == {themed.id, again.id}
    brain.goal = await brain.emit("pfc", "goal", "Build a torso and a head in FreeCAD and measure the solid")
    assert [h.id for h in learner.select("dmn", brain.focus())] == [general.id]  # another theme: they keep quiet
    brain.goal = await brain.emit("pfc", "goal", "See whether the pandas import hangs with other thread environment variables")
    assert [h.id for h in learner.select("dmn", brain.focus())] == [themed.id, general.id]  # and one wording is enough
    brain.close()


async def test_what_was_built_because_the_human_asked_is_shown_to_them_before_it_is_done(root):
    class Hasty(ScriptedLLM):
        async def generate(self, role, system, prompt, schema, **kwargs):
            if schema is Decision:
                self.prompts.setdefault(schema, []).append(prompt)
                return Decision(reasoning="The cube is built and valid.", status="done", intent=""), 20
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=Hasty())
    pfc = brain.regions["pfc"]

    async def built(asked):
        brain.goal = await brain.emit("pfc", "goal", f"build a cube ({asked})", data={"asked": asked})
        pfc.built, pfc.shown = "", False
        return await brain.emit("motor", "action", "run_freecad(cube.py)\nran in FreeCAD\n1 solid, valid", data={
            "ok": True, "grounded": 0.7, "goal_id": brain.goal.id, "tool": "run_freecad", "path": "cube.py",
            "repeat": False, "measured": True}, parent=brain.goal.id)

    await pfc.fire(await built(asked=True))
    asked = brain.llm.prompts[Decision]
    assert len(asked) == 2 and "Motor Cortex" in asked[1]  # the body speaks up: they asked, and have seen nothing
    assert "You built cube.FCStd because the human asked, and they have not seen it" in asked[1]
    await pfc.fire(await built(asked=False))  # what it builds for itself it need not show anybody
    assert len(brain.llm.prompts[Decision]) == 3
    brain.close()


async def test_a_brain_in_a_container_learns_from_its_launcher_where_it_lives_and_the_stream_shows_who_tells_whom(
    root, monkeypatch
):
    monkeypatch.setattr(tools, "freecad_window", lambda: r"C:\Programs\FreeCAD 1.1\bin\freecad.exe")
    lent = bridge.Bridge({"sense": lambda ask: tools.sense(root)})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        inside = Tools(root / "workspace", SafetySettings(), root)
        assert await inside.survey() == {"home": str(root), "freecad": "FreeCAD 1.1", "builds": ""}
        assert "(.FCStd) in FreeCAD's window" in dict(inside.abilities())["show"]
        monkeypatch.setenv(bridge.BRIDGE, f"127.0.0.1:{lent.server_address[1]}:not-the-key")
        older = Tools(root / "workspace", SafetySettings(), root)  # a launcher that does not say: nothing is known
        assert await older.survey() == {"home": "", "freecad": "", "builds": ""}
    finally:
        lent.shutdown()
        lent.server_close()
    monkeypatch.delenv(bridge.BRIDGE)

    brain = Brain(root, llm=ScriptedLLM())
    shown = brain.bus.subscribe("remark", "view")
    brain.tell("basal_ganglia", "pfc", "-0.6 for that step: the script was never run.")
    await brain.regions["motor"].weigh_in("Can a cube be built?", [], ["critic"], None)
    remark, view = (format_signal(shown.get_nowait(), brain).plain for _ in range(2))
    assert "Basal Ganglia → Prefrontal Cortex -0.6 for that step: the script was never run." in remark
    assert "Motor Cortex thinks, then hands to the executive A script can be written and run" in view
    assert "Basal Ganglia" in brain.hearing("pfc") and not brain.hearing("pfc")  # heard once, then let go
    brain.close()


async def test_a_model_is_built_out_of_parts_and_the_next_step_adds_to_it(root, monkeypatch):
    brain = Brain(root, llm=ScriptedLLM())
    motor, workspace = brain.regions["motor"], root / "workspace"
    torso = build.Part(name="torso", shape="box", size=[300, 150, 400], axis="z", on="", side="top", shift=[], cut=False)
    head = build.Part(name="head", shape="sphere", size=[90], axis="z", on="torso", side="top", shift=[], cut=False)
    arm = build.Part(name="left arm", shape="cylinder", size=[40, 350], axis="x", on="Torso", side="left",
                     shift=[0, 0, 120], cut=False)

    here = await motor.execute(Act(tool="build", path="body", parts=[torso]), "")  # no FreeCAD on this machine
    assert not here.ok and "only in the brain's own container" in here.text and "torso: box 300 × 150 × 400 mm" in here.text
    assert "kept in body.parts.json" in here.text
    (workspace / "body.parts.json").unlink()
    (workspace / "body.parts.py").unlink()

    ran = []

    async def freecad(script, saved_as):  # stands in for FreeCAD: one solid, unless a part hangs loose
        assert (script.name, saved_as) in (("body.parts.py", "body"), ("own.parts.py", "own"))
        ran.append(script.read_text())
        solids = 2 if "far_away" in ran[-1] else 1
        return tools.ToolResult(True, f"ran in FreeCAD\nFreeCAD built 1 object with {solids} solid", 0.7), {
            "bodies": [{"top": True, "solids": solids, "valid": True, "size": [300.0, 150.0, 553.0], "volume": 2e7}],
            "saved": f"{saved_as}.FCStd"}

    monkeypatch.setattr(brain.tools, "builds", lambda: True)
    monkeypatch.setattr(brain.tools, "_cad_run", freecad)
    first = await motor.execute(Act(tool="build", path="body", parts=[torso]), "")
    assert first.ok and first.grounded == pytest.approx(0.6) and "standing on the ground" in first.text
    assert "Part.makeBox(300.000, 150.000, 400.000, V(-150.000, -75.000, 0.000))" in ran[0]

    second = await motor.execute(Act(tool="build", path="body.FCStd", parts=[head, arm]), "")  # the torso is kept
    assert second.grounded == pytest.approx(0.8)  # how it stands is said first and in a line, with what to do with it
    assert second.text.startswith("ran in FreeCAD: the model `body` is ONE valid solid of 3 parts (torso, head, left arm), "
                                  "300 × 150 × 553 mm, volume 20000000 mm³. That is FreeCAD's own measurement")
    assert "It is saved as body.FCStd: `show` with path body.FCStd puts it on the human's screen." in second.text
    assert "head: sphere of radius 90 180 × 180 × 180 mm, middle at (0, 0, 463), on the top of torso, reaching 27 mm into it" in second.text
    assert "left arm: cylinder of radius 40 350 × 80 × 80 mm, middle at (-272, 0, 320), on the left of torso" in second.text
    nearly = head.model_copy(update={"on": "top_of_torso"})  # a name that only holds a part's name finds that part
    assert "on the top of torso, reaching 27 mm into it (`on` said top_of_torso: taken to mean torso)" in build.describe(
        build.place([torso, nearly]))
    aground = torso.model_copy(update={"on": "ground"})  # and the ground is the ground, however it is put
    assert "standing on the ground" in build.describe(build.place([aground]))
    # A part that is attached stays attached: a shift given as if it were a position is cut back.
    lifted = build.place([torso, head.model_copy(update={"shift": [0, 400, 300]})])
    assert lifted[1].middle == [0.0, 138.0, 463.0] and not build.loose(lifted)
    told = build.describe(lifted)
    assert "its shift was cut back along y from 400 to 138 and z from 300 to 0: a part attached to torso stays" in told
    on_top = build.place([torso, head.model_copy(update={"on": "top", "side": "left"})])  # `on: top` for "on top"
    assert on_top[1].middle == [0.0, 0.0, 463.0] and "on the top of torso" in build.describe(on_top)
    beside = build.place([torso, arm.model_copy(update={"on": "right of torso", "side": "front", "shift": []})])
    assert beside[1].middle == [272.5, 0.0, 200.0]  # a side said in `on` is the side that was meant
    assert "whole = whole.fuse(p1_head)" in ran[1] and "Part.makeCylinder(40.000, 350.000" in ran[1]
    drawn = second.text.split("Seen from the front")[1].splitlines()  # a mind without eyes can read a picture
    assert drawn[-1] == "A torso, B head, C left arm" and any("CCC" in line and "A" in line for line in drawn)
    assert any(set(line.strip()) == {"B"} for line in drawn) and "one character is" in drawn[0]

    loose = build.Part(name="far_away", shape="box", size=[50, 50, 50], axis="z", on="", side="top",
                       shift=[400, 0, 0], cut=False)  # stood on the ground by itself, away from the body
    third = await motor.execute(Act(tool="build", path="body", parts=[loose]), "")  # a whole body does not come apart
    assert not third.ok and third.text.startswith("not kept: with that change the model `body` is NOT one body")
    assert "far_away does NOT touch the rest: it stands on the ground by itself, 225 mm from torso along x" in third.text
    assert "Give it again with `on` naming the part it belongs on, for example torso" in third.text  # told what to do
    assert "So the change was undone, and the model is as it stood before: ran in FreeCAD: the model `body` is ONE" in third.text
    assert "far_away" not in (workspace / "body.parts.json").read_text()
    holed = await motor.execute(Act(tool="build", path="body", parts=[head.model_copy(update={"cut": True})]), "")
    assert holed.text.startswith("(head is a part of the body, so `cut` true was taken for a slip")  # taken as meant
    assert '"cut": true' not in (workspace / "body.parts.json").read_text()
    head = head.model_copy(update={"size": [100]})
    less = await motor.execute(Act(tool="build", path="body", parts=[head]), "")  # one body still, but it gained nothing
    assert less.grounded == pytest.approx(0.3) and "ONE valid solid of 3 parts" in less.text
    (workspace / "body.FCStd").write_bytes(b"PK")  # FreeCAD is a stand-in here, and saved nothing
    same = await motor.execute(Act(tool="build", path="body.parts.json", parts=[head]), "")  # the same part as it is
    assert same.ok and "nothing to add: the model `body` already has exactly these parts" in same.text
    assert "what this step wanted is done already. As it was last measured: ran in FreeCAD: the model `body` is ONE" in same.text
    assert "The model `body` so far, 3 parts:" in same.text and len(ran) == 5  # FreeCAD was not asked again
    measured = await motor.execute(Act(tool="run_freecad", path="body.FCStd"), "")  # "run the model": measure it
    assert measured.ok and measured.text.startswith("ran in FreeCAD: the model `body` is ONE valid solid of 3 parts")
    assert len(ran) == 6
    assert "The model `body` so far, 3 parts:" in brain.tools.blueprint() and "B head" in brain.tools.blueprint()

    nowhere = build.Part(name="hat", shape="cone", size=[60, 0, 80], axis="z", on="neck", side="top", shift=[], cut=False)
    refused = await motor.execute(Act(tool="build", path="body", parts=[nowhere]), "")
    assert not refused.ok and "attached to `neck`, and there is no such part. The parts are: torso, head, left arm" in refused.text
    assert "The model `body` so far, 3 parts:" in refused.text  # and what it has is laid beside the refusal
    assert "`parts` is empty" in (await motor.execute(Act(tool="build", path="body"), "")).text
    (workspace / "own.py").write_text("print(1)\n")  # a script of its own under the model's name is not in the way
    (workspace / "own.parts.json").write_text('{"tool": "build", "parts": [{"name": "x"}]}')  # nor a description by hand
    assert (await motor.execute(Act(tool="build", path="own", parts=[torso]), "")).ok
    assert (workspace / "own.py").read_text() == "print(1)\n" and brain.tools.models()[0] == "own"
    assert "Other models built out of parts: body" in brain.tools.blueprint()
    (workspace / "own.parts.json").unlink()
    flat = build.Part(name="plate", shape="box", size=[10, 0, 10], axis="z", on="", side="top", shift=[], cut=False)
    bad = await motor.execute(Act(tool="build", path="plate", parts=[flat]), "")
    assert not bad.ok and "a box needs `size` [width, depth, height]" in bad.text

    # A model is not text. Written as text it opens in nothing, and this brain once spent a day on one.
    faked = await brain.tools.execute("write_file", "head.FCStd", "<?xml version='1.0'?><FreeCADDocument/>")
    assert not faked.ok and "a model is not text" in faked.text and not (workspace / "head.FCStd").exists()
    assert any("not text" in fact for fact in brain.facts("open new_head_model.FCStd and repair it"))
    report = {"error": "OSError: Invalid project file head.FCStd: basic_ios::clear: iostream error", "objects": {}}
    (workspace / "real.FCStd").write_bytes(b"PK\x03\x04")
    (workspace / "fake.FCStd").write_bytes(b"<?xml")
    why = brain.tools.why_it_failed(report)
    assert "Build the model again with `build`" in why and "real.FCStd" in why and "fake.FCStd" not in why
    none = {"error": "AttributeError: 'NoneType' object has no attribute 'Shape'", "objects": {"model": ["Box", "Head"]}}
    assert "The documents hold: model: Box, Head." in brain.tools.why_it_failed(none)
    # The machine looks at the file a thought names, and says what it is beside that thought.
    looked = brain.facts("verify the internal references of workspace/fake.FCStd")
    assert any("fake.FCStd is NOT a FreeCAD model: it is a text file of 5 bytes" in fact for fact in looked)
    assert not any("real.FCStd" in fact for fact in brain.facts("show real.FCStd to the human"))
    assert any("gone.FCStd does not exist" in fact for fact in brain.facts("open gone.FCStd"))
    pfc, calls = brain.regions["pfc"], brain.llm.calls
    turned_down = await pfc.refusal("Open fake.FCStd in a script and find the reference error")
    assert turned_down[:2] == (-0.4, "reflex") and "not a model" in turned_down[2] and brain.llm.calls == calls
    assert await pfc.refusal("Build a torso with `build` and save it as torso_one.FCStd") is None  # not there yet: fine
    unread = await brain.tools.execute("read_file", "fake.FCStd")
    assert not unread.ok and "is NOT a FreeCAD model" in unread.text
    assert "not text: it cannot be read" in (await brain.tools.execute("read_file", "real.FCStd")).text

    # A path that begins with the workspace's own name means the workspace.
    assert (await brain.tools.execute("write_file", "workspace/calc.py", "print(6*7)\n")).ok
    assert (workspace / "calc.py").is_file() and not (workspace / "workspace").exists()
    schema = Act.model_json_schema()  # the model has to write every field of an action, and of a part
    assert set(schema["required"]) == set(schema["properties"]) >= {"tool", "parts"}
    assert set(schema["$defs"]["Part"]["required"]) == set(build.Part.model_fields)
    brain.close()


async def test_it_has_a_name_of_its_own_only_once_it_has_given_itself_one(root):
    settings = BrainSettings.load(root)
    settings.voice.name = "de_DE-thorsten-medium"
    settings.save(root)
    brain = Brain(root, llm=ScriptedLLM())
    motor = brain.regions["motor"]
    assert brain.name == "test" and not brain.called
    fact = brain.facts("wie heißt du?")[0]  # the name of its voice lies in what it is shown: it is told whose name that is
    assert "You have not given yourself a name" in fact and "folder test" in fact and "takes a name and keeps it" in fact
    assert "de_DE-thorsten-medium is the name of the voice you speak with, not your name." in fact
    assert brain.facts("I am Thorsten") and not brain.facts("write calc.py and run it")
    await brain.answer("what is your name?")
    assert "You have not given yourself a name" in brain.llm.prompts[Reply][-1]

    voice_name = await motor.execute(Act(tool="take_name", content="Thorsten"), "")
    assert not voice_name.ok and "the name of a voice" in voice_name.text and brain.name == "test"
    assert not (await motor.execute(Act(tool="take_name", content="I think a good name would be Nova, perhaps"), "")).ok
    brain.self_image = "I am test, a mind that builds."
    taken = await motor.execute(Act(tool="take_name", content="“Nova”"), "")
    assert taken.ok and "From now on you are called Nova" in taken.text and brain.name == "Nova"
    assert "one region of Nova" in brain.system_prompt("", []) and brain.self_image == "I am Nova, a mind that builds."
    assert "Your name is Nova: you chose it yourself" in brain.facts("what are you called?")[0]
    assert "- take_name:" in brain.body() and brain.news[-1] == "You gave yourself a name: Nova."
    brain.close()
    again = Brain(root, llm=ScriptedLLM())  # a name is kept across lives
    assert again.name == "Nova" and again.settings.name == "test"

    # Asked about itself, it does not answer by reflex: the regions think it through, and what
    # the mind concludes is said aloud. A goal in hand goes on meanwhile.
    pfc, said = again.regions["pfc"], again.bus.subscribe("speech")
    again.llm.heard, again.llm.answer = "ponder", Reply(say="Let me think about that.")
    again.goal = own = await again.emit("pfc", "goal", "print 6 times 7")
    await again.answer("what do you want to become?")
    assert "Do not answer it yet" in again.llm.prompts[Reply][-1] and said.get_nowait().content == "Let me think about that."
    wonder = await pfc.wait_trigger()
    assert (wonder.kind, wonder.data["asked"]) == ("wonder", True) and again.proposal_pending
    await pfc.fire(wonder)
    heard = views(again.llm)
    assert len(heard) == 3 and "The human asked the mind about itself: “what do you want to become?”" in heard[0]
    assert "You speak as its imagination" in heard[0] and "They were told at once: “Let me think about that.”" in heard[0]
    assert "they propose a choice and say what speaks for it" in heard[0]
    concluded = said.get_nowait()
    assert (concluded.source, concluded.content) == ("pfc", "It can be found out by writing a script and running it.")
    assert again.goal is own and not again.proposal_pending and not pfc.queued
    kept = again.memory.recent("thalamus", "exchange", 1)[0]["content"]
    assert kept.startswith("The human asked: what do you want to become?\nAfter thinking it through I answered: It can")

    class Naming(ScriptedLLM):
        """Decides on a name when the mind is asked to choose one, and says so."""

        async def generate(self, role, system, prompt, schema, **kwargs):
            if issubclass(schema, Conclusion) and "name" in schema.model_fields:
                self.prompts.setdefault(schema, []).append(prompt)
                return schema(conclusion="I choose Juno: it is short and my own.", goal=None,
                              say="Ich nenne mich Juno.", name="Juno"), 20
            return await super().generate(role, system, prompt, schema, **kwargs)

    again.llm = Naming()
    again.llm.heard = "ponder"
    await again.answer("give yourself another name, and be creative")
    said.get_nowait()
    await pfc.fire(await pfc.wait_trigger())  # a name the mind decides on is its name from that moment
    assert again.name == "Juno" and said.get_nowait().content == "Ich nenne mich Juno." and again.goal is own
    assert "`name`: only when what you decided is a name for yourself" in again.llm.prompts[Answer][0]
    assert again.memory.get_state("called") == "Juno"
    assert again.memory.forget_neutral(0) and again.memory.recent("thalamus", "exchange", 2)  # their words are never forgotten
    again.close()


async def test_it_sees_which_windows_are_open_on_the_humans_screen(root, monkeypatch):
    settings = BrainSettings.load(root)
    settings.safety.allow_screen = True
    settings.save(root)
    screen = {"screen": [1920, 1080], "windows": [
        {"app": "WindowsTerminal", "title": "cortex run test", "front": True},
        {"app": "chrome", "title": "FreeCAD documentation", "front": False},
    ]}
    monkeypatch.setattr(container, "look", lambda: json.dumps(screen))
    lent = bridge.Bridge({"look": lambda ask: container.eyes(root)})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        assert brain.tools.sees() and "- look:" in brain.body() and "You can see the human's screen:" in brain.body()
        calls = brain.llm.calls
        await brain.glance()
        percept = brain.memory.recent("thalamus", "percept", 1)[0]["content"]
        assert percept.startswith("You look at the human's screen for the first time since you were started.")
        assert "In front is WindowsTerminal (“cortex run test”) (the terminal you yourself are shown in, probably)" in percept
        assert "Also open: chrome (“FreeCAD documentation”). The screen is 1920 × 1080." in percept
        assert brain.llm.calls == calls and "WindowsTerminal" in brain.hearing("dmn")  # seen without asking the model
        assert "On their screen WindowsTerminal (“cortex run test”) is in front, with 1 other windows open." in brain.sense()
        assert "The human's screen, as you saw it" in brain.awareness()
        assert "What is on the human's screen, seen" in brain.facts("what is on my screen?")[0]

        await brain.glance()
        assert len(brain.memory.recent("thalamus", "percept", 5)) == 1  # nothing changed there: no news
        screen["windows"][0]["front"] = False
        screen["windows"].insert(0, {"app": "FreeCAD", "title": "body.FCStd - FreeCAD 1.1", "front": True})
        await brain.glance()
        assert len(brain.memory.recent("thalamus", "percept", 5)) == 1  # windows come and go: not every minute
        assert "FreeCAD (“body.FCStd" in brain.sense()  # but what it holds of the screen is as it is now
        brain.noticed_at = 0.0
        screen["windows"].pop()
        await brain.glance()
        news = brain.memory.recent("thalamus", "percept", 1)[0]["content"]
        assert news.startswith("On the human's screen, chrome was closed. In front is FreeCAD (“body.FCStd - FreeCAD 1.1”).")

        looked = await brain.regions["motor"].execute(Act(tool="look"), "see what is open")
        assert looked.ok and "On the human's screen right now. In front is FreeCAD" in looked.text

        settings.safety.allow_screen = False  # switched off, the machine that lends its eyes refuses out there
        settings.save(root)
        assert await brain.tools.see() is None
        brain.settings.safety.allow_screen = False
        assert "You cannot see the human's screen." in brain.body() and "- look:" not in brain.body()
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()
    if tools.sys.platform == "win32":  # the real look, on a machine that has one: names and titles, nothing else
        seen = json.loads(tools.look())
        assert len(seen["screen"]) == 2 and all(set(w) == {"app", "title", "front", "minimised", "own"} for w in seen["windows"])


async def test_each_region_brings_up_what_its_own_tract_has_learned(root):
    class Embedding(ScriptedLLM):
        async def embed(self, text):
            words = re.findall(r"\w+", text.lower())
            vec = sum(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(1024) for w in words)
            return (vec / np.linalg.norm(vec)).astype(np.float32)

    brain = Brain(root, llm=Embedding())
    motor = brain.regions["motor"]
    brain.goal = await brain.emit("pfc", "goal", "find out how fast the prime sieve is")
    plan = await brain.emit("pfc", "plan", "time the prime sieve script with ten thousand numbers", parent=brain.goal.id,
                            data={"intent": "x", "goal_id": brain.goal.id})
    assert motor.habit(plan) == ""  # nothing has travelled this tract yet
    done = await brain.emit("motor", "action", "run_python(sieve_timing.py)\nran successfully\n0.04 seconds",
                            parent=plan.id, data={"ok": True, "grounded": 0.6})
    other = await brain.emit("pfc", "plan", "draw the harbour at night as an svg page", parent=brain.goal.id,
                             data={"intent": "x", "goal_id": brain.goal.id})
    bad = await brain.emit("motor", "action", "write_file(harbour.svg)\nrefused: nothing to write", parent=other.id,
                           data={"ok": False, "grounded": -0.2})
    for _ in range(3):
        brain.apply_reward(done.id, 0.9, "critic")
        brain.apply_reward(bad.id, -0.8, "critic")
    carried = brain.net.tract(("pfc", "motor"))
    assert carried.carried == 2 and carried.weights is not None and brain.net.tract(("motor", "pfc")).weights is None

    calls = brain.llm.calls
    alike = await brain.emit("pfc", "plan", "time the prime sieve script again with a million numbers",
                             parent=brain.goal.id, data={"intent": "x", "goal_id": brain.goal.id})
    assert motor.habit(alike).endswith("- run_python(sieve_timing.py) ran successfully 0.04 seconds")
    night = await brain.emit("pfc", "plan", "draw the harbour at night as another svg page", parent=brain.goal.id,
                             data={"intent": "x", "goal_id": brain.goal.id})
    assert motor.habit(night) == ""  # what followed the like of this was punished: nothing is expected of it
    assert brain.llm.calls == calls  # patterns along a tract: no model was asked

    # The pattern also travels by itself: fired in the executive, it reaches the motor cortex two
    # heartbeats later, and with it, fainter, what the motor cortex has learned to answer it with.
    for firing in brain.net.fire.values():
        firing[:] = 0.0
    for tract in brain.net.tracts.values():
        tract.flight.clear()
    brain.net.send("pfc", alike.bits)
    brain.regions["brainstem"].pulse()
    assert not motor.neurons.any()
    brain.regions["brainstem"].pulse()
    assert motor.neurons[alike.bits].min() > 0 and motor.neurons[done.bits & ~alike.bits].min() > 0
    assert motor.activity > 0.5  # and the region stirs when it arrives
    brain.close()
