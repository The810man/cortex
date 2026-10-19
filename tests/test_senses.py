"""Its camera, ears and hands, lent by the human's machine and switched on by the human, and its
answers to the human, which can now be judged like any other step."""

import json

import pytest
from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex import bridge, container, eyes, senses
from cortex.brain import Brain, Reply
from cortex.config import BrainSettings
from cortex.regions import Act


def screen(*windows):
    return [
        {"handle": number, "app": app, "title": title, "front": number == 1, "minimised": False, "own": own}
        for number, (app, title, own) in enumerate(windows, 1)
    ]


def test_its_eyes_pass_over_the_terminal_it_is_shown_in_and_find_a_window_by_name(monkeypatch):
    seen = screen(("WindowsTerminal", "Administrator: Windows PowerShell", True),
                  ("FreeCAD", "body.FCStd - FreeCAD 1.1", False), ("chrome", "FreeCAD documentation", False))
    monkeypatch.setattr(eyes, "windows", lambda: seen)
    assert eyes.find("")["app"] == "FreeCAD"  # in front is its own terminal: what the human has in view is behind it
    assert eyes.find("freecad")["title"] == "body.FCStd - FreeCAD 1.1"  # the program itself before a title that names it
    assert eyes.find("documentation")["app"] == "chrome"
    assert eyes.find("powershell")["own"]  # named, it may look at itself
    assert eyes.find("Bildschirm") is None  # the whole screen
    with pytest.raises(OSError, match="no window on the screen is called “blender”. Open are: FreeCAD"):
        eyes.find("blender")


def test_the_hands_never_act_in_a_terminal_nor_outside_the_programs_named(monkeypatch):
    seen = screen(("WindowsTerminal", "Administrator: Windows PowerShell", True), ("notepad", "notes.txt - Editor", False),
                  ("pwsh", "PowerShell 7", False))
    monkeypatch.setattr(eyes, "windows", lambda: seen)
    monkeypatch.setattr(senses.sys, "platform", "win32")
    with pytest.raises(ValueError, match="is a terminal"):
        senses.act("powershell", "type exit", ["FreeCAD", "pwsh"])
    with pytest.raises(ValueError, match="may act only in FreeCAD"):
        senses.act("notes", "click 0.5 0.5", ["FreeCAD"])
    with pytest.raises(ValueError, match="Windows key"):
        senses.press("win+r")
    with pytest.raises(ValueError, match="no key is called"):
        senses.press("ctrl+banana")


async def test_camera_ears_and_hands_are_off_until_the_human_switches_them_on(root, monkeypatch):  # noqa: F811
    asked = []

    def stand_in(do):
        def serve(ask):
            asked.append((do, ask))
            return senses.serve(root, do, ask)
        return serve

    monkeypatch.setattr(senses, "look_through_camera", lambda model, host, question: {"seen": f"A person waving. ({question})", "seconds": 1.0})
    monkeypatch.setattr(senses, "listen", lambda seconds, raw=False: {"heard": "Hallo, kannst du mich hören?", "seconds": seconds, "quiet": False, "language": "de"})
    monkeypatch.setattr(senses, "act", lambda target, command, programs: f"click on “Part” (MenuItem) in “{target}”")
    lent = bridge.Bridge({do: stand_in(do) for do in ("camera", "listen", "hands")})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        motor = brain.regions["motor"]
        body = brain.body()
        assert "You have no camera." in body and "You have no ears." in body and "- camera:" not in body
        refused = await motor.execute(Act(tool="camera"), "look at the human")
        assert not refused.ok and "switched off in your settings" in refused.text and not asked
        assert any("You have no camera: nothing you write or run can reach one" in fact for fact in brain.facts("enable the camera and microphone"))

        settings = BrainSettings.load(root)
        settings.safety.allow_camera = settings.safety.allow_microphone = settings.safety.allow_hands = True
        settings.save(root)
        brain.settings.safety.allow_camera = brain.settings.safety.allow_microphone = brain.settings.safety.allow_hands = True
        body = brain.body()
        assert "- camera:" in body and "- listen:" in body and "- hands:" in body and "they may act only in FreeCAD" in body
        assert "`hands` can click, drag and type in it" in body or "With `hands` you click and type in windows of FreeCAD" in body
        seen = await motor.execute(Act(tool="camera", content="is someone there?"), "look at the human")
        assert seen.ok and seen.text == "Through the camera your eyes see: A person waving. (is someone there?)"
        heard = await motor.execute(Act(tool="listen", content="8 seconds"), "listen to the human")
        assert heard.ok and "8 s" in heard.text and "“Hallo, kannst du mich hören?”" in heard.text
        done = await motor.execute(Act(tool="hands", path="FreeCAD", content="click Part"), "open the Part menu")
        assert done.ok and "only `look` shows" in done.text
        assert asked[-1] == ("hands", {"target": "FreeCAD", "command": "click Part"})

        settings.safety.allow_camera = False  # the machine out there reads the switch itself at every use
        settings.save(root)
        refused = await motor.execute(Act(tool="camera"), "look at the human")
        assert not refused.ok and "the camera is switched off in this brain's settings" in refused.text
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()


async def test_its_eyes_look_into_the_window_it_names_for_what_it_names(root, monkeypatch):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.safety.allow_screen = settings.safety.allow_hands = True
    settings.save(root)
    looked = []

    def look_at(model, host, target="", question=""):
        looked.append((target, question))
        return {"window": "body.FCStd - FreeCAD 1.1", "seen": "A torso with a head; the arms hang below it.", "seconds": 2.0}

    monkeypatch.setattr(container, "look", lambda: json.dumps({"screen": [1920, 1080], "windows": [
        {"app": "FreeCAD", "title": "body.FCStd - FreeCAD 1.1", "front": False}]}))
    monkeypatch.setattr(container, "look_at", look_at)
    monkeypatch.setattr(senses, "named_controls", lambda target: "What can be clicked by name in “body.FCStd”: MenuItem: “File”, “View”")
    lent = bridge.Bridge({"look": lambda ask: container.eyes(root), "see": lambda ask: container.sight(root, ask)})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        acted = await brain.regions["motor"].execute(
            Act(tool="look", path="FreeCAD", content="are the arms attached to the torso?"), "look at the model")
        assert ("FreeCAD", "are the arms attached to the torso?") in looked
        assert acted.ok and "What “body.FCStd - FreeCAD 1.1” shows: A torso with a head" in acted.text
        assert "MenuItem: “File”, “View”" in acted.text  # with hands on, what can be clicked there comes with it

        def nowhere(model, host, target="", question=""):
            raise OSError("no window on the screen is called “blender”. Open are: FreeCAD (“body.FCStd - FreeCAD 1.1”)")

        monkeypatch.setattr(container, "look_at", nowhere)
        missed = await brain.regions["motor"].execute(Act(tool="look", path="blender"), "look at blender")
        assert not missed.ok and "no window on the screen is called “blender”" in missed.text
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()


async def test_an_answer_to_the_human_is_judged_and_kept_to_learn_from(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    critic = brain.regions["basal_ganglia"]
    brain.llm.answer = Reply(say="Ja, ich erkenne dich nun: die Kamera ist aktiviert.")
    await brain.answer("kannst du mich nun erkennen")
    said = brain.last_said
    assert said is not None and said.kind == "exchange" and "Kamera ist aktiviert" in said.content
    assert (await critic.inbox.get()).id == said.id  # the critic hears every answer, as it hears every step
    kept = [json.loads(line) for line in (root / "experience").glob("*.jsonl").__next__().read_text(encoding="utf-8").splitlines()]
    assert kept[-1]["id"] == said.id and "Kamera ist aktiviert" in kept[-1]["reply"]  # with what it was shown

    brain.apply_reward(said.id, 0.4, "critic", "Sounds helpful.")
    brain.apply_reward(said.id, -1.0, "human", "Judged by hand.")  # the human's verdict is no second opinion
    assert brain.memory.get(said.id)["value"] < -0.15
    lessons = brain.experience.lessons(lambda memory_id: (row := brain.memory.get(memory_id)) and row["value"])
    assert any(lesson["id"] == said.id and lesson["label"] is False for lesson in lessons)

    critic.speak_up(said, -0.8, "It claims a camera the body does not have.", "Say what you cannot do.")
    brain.llm.answer = Reply(say="Nein, ich habe keine Kamera.")
    await brain.answer("und jetzt?")
    assert "It claims a camera the body does not have." in brain.llm.prompts[Reply][-1]  # heard before it answers again
    brain.close()


def test_always_listening_cuts_out_what_is_said_and_not_while_it_speaks_itself(monkeypatch):
    import numpy as np

    from cortex import voice

    ear = senses.Ear()
    ear.mode = "always"
    rng = np.random.default_rng(0)
    quiet = lambda: (rng.standard_normal(senses.Ear.BLOCK) * 0.002).astype("float32")  # noqa: E731
    loud = lambda: (np.sin(np.arange(senses.Ear.BLOCK) / 3) * 0.2).astype("float32")  # noqa: E731
    for block in [quiet() for _ in range(30)] + [loud() for _ in range(40)] + [quiet() for _ in range(40)]:
        ear.take(block)
    assert len(ear.ready) == 1 and 1.2 < len(ear.ready[0]) / senses.RATE < 2.4  # a voice of 1.2 s, then a pause
    voice.SPEAKING.set()  # its own voice is not something said to it
    try:
        for block in [loud() for _ in range(40)] + [quiet() for _ in range(40)]:
            ear.take(block)
    finally:
        voice.SPEAKING.clear()
    assert len(ear.ready) == 1
    ear.mode = "push"  # pressed to talk, it keeps everything until the next press
    ear.take(loud())
    ear.take(quiet())
    assert len(ear.held) == 2


async def test_a_mind_that_sees_and_hears_is_handed_the_picture_and_the_voice_itself(root, monkeypatch):  # noqa: F811
    llm = ScriptedLLM()
    llm.senses = True
    brain = Brain(root, llm=llm)
    picture, sound = b"\x89PNG picture", b"RIFF sound"
    brain.take_in({"images": [picture], "audio": []}, "what you took in with look(FreeCAD)")
    held, note = brain.media_for("pfc")
    assert held["images"] == [picture] and "look(FreeCAD)" in note and "not in words" in note
    assert brain.media_for("pfc") == ({}, "")  # each region is shown it once
    assert brain.media_for("hippocampus") == ({}, "")  # memory keeps words, never the picture

    from cortex.brain import Said

    async def generate(role, system, prompt, schema, **kwargs):
        if schema is Said:
            llm.media.append((schema, kwargs.get("images") or [], kwargs.get("audio") or []))
            return Said(words="bau mir einen kopf", how="calm, a little tired"), 5
        return await ScriptedLLM.generate(llm, role, system, prompt, schema, **kwargs)

    llm.generate = generate
    brain.llm.answer = Reply(say="Ich höre dich.")
    await brain.hear(sound)
    signal = await brain.heard.get()
    assert signal.content == "bau mir einen kopf" and signal.data["voice"]["how"] == "calm, a little tired"
    await brain.answer(signal.content, signal.data["voice"])
    answered = [entry for entry in llm.media if entry[0] is not Said]
    assert answered and answered[-1][2] == [sound]  # the answer is written hearing the voice itself
    assert "how it sounded: calm, a little tired" in llm.prompts[next(k for k in llm.prompts if k.__name__ == "Reply")][-1]
    brain.close()
