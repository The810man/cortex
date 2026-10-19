"""It knows when what it builds is whole: it stops there, shows it, and does not spoil it."""

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex import build, tools
from cortex.brain import Brain
from cortex.config import BrainSettings
from cortex.regions import Decision

SOUND = "ran in FreeCAD: the model `body` is ONE valid solid of 2 parts (torso, head), 300 × 150 × 553 mm."


def part(name, shape="box", size=(300, 150, 400), on="", side="top"):
    return build.Part(name=name, shape=shape, size=list(size), axis="z", on=on, side=side, shift=[], cut=False)


async def built(brain, goal):
    return await brain.emit("motor", "action", f"build(body)\n{SOUND}", parent=goal.id, data={
        "ok": True, "grounded": 0.7, "goal_id": goal.id, "tool": "build", "path": "body", "repeat": False,
        "measured": True})


async def test_a_goal_whose_model_has_every_wanted_part_is_done_by_reflex_and_shown(root, monkeypatch):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    pfc, workspace = brain.regions["pfc"], root / "workspace"
    assert await brain.wants("Build a body in FreeCAD with a torso and a head") == [("torso", 1), ("head", 1)]
    assert await brain.wants("Write a script that prints 6*7") == []  # nothing to be built: no call at all
    calls = brain.llm.calls
    assert await brain.wants("Build a body in FreeCAD with a torso and a head") and brain.llm.calls == calls  # kept

    torso, head = part("torso"), part("head", "sphere", (90,), on="torso")
    (workspace / f"body{build.KEPT}").write_text(build.dump([torso], SOUND.replace("2 parts", "1 part")))
    brain.goal = goal = await brain.emit("pfc", "goal", "Build a body with a torso and a head", data={"asked": True})
    pfc.wanted = [("torso", 1), ("head", 1)]
    await pfc.fire(await built(brain, goal))  # only the torso: the executive is told what is still missing
    assert brain.goal is goal
    assert "Wanted of the model `body`: torso, head. It has: torso. Still missing: head." in brain.llm.prompts[Decision][-1]

    (workspace / f"body{build.KEPT}").write_text(build.dump([torso, head], SOUND))
    (workspace / "body.FCStd").write_bytes(b"PK")
    opened, said = [], brain.bus.subscribe("speech")
    monkeypatch.setattr(tools, "SHOW_GAP", 0.0)
    monkeypatch.setattr(tools, "freecad_window", lambda: "FreeCAD.exe")
    monkeypatch.setattr(tools, "_window", None)
    monkeypatch.setattr(tools.subprocess, "Popen", lambda command, env, **kwargs: opened.append(env["CORTEX_MODEL"]))
    asked = len(brain.llm.prompts[Decision])
    await pfc.fire(await built(brain, goal))  # every part there, one valid solid: done, with no model asked
    assert brain.goal is None and len(brain.llm.prompts[Decision]) == asked
    assert opened and opened[0].endswith("body.FCStd") and pfc.shown  # what the human asked for is on their screen
    assert said.get_nowait().content == "body is finished: every part you wanted is there, and it is one solid body. " \
                                        "I have opened it for you in FreeCAD."
    outcome = brain.memory.recent("pfc", "outcome", 1)[0]["content"]
    assert outcome.startswith("DONE: Build a body") and "By reflex: the model body has every part" in outcome
    brain.close()


async def test_the_long_work_is_told_when_its_body_is_whole(root):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.purpose = "Learn to build a body for yourself in FreeCAD: a torso and a head as one valid solid."
    settings.save(root)
    brain = Brain(root, llm=ScriptedLLM())
    dmn, workspace = brain.regions["dmn"], root / "workspace"
    assert await dmn.whole_yet(settings.purpose) == ""  # nothing built yet
    (workspace / f"body{build.KEPT}").write_text(build.dump([part("torso")], SOUND))
    assert await dmn.whole_yet(settings.purpose) == (
        "Wanted of the model `body`: torso, head. It has: torso. Still missing: head.")
    (workspace / f"body{build.KEPT}").write_text(build.dump([part("torso"), part("head", on="torso")], SOUND))
    told = await dmn.whole_yet(settings.purpose)
    assert told.startswith("Your long work's model `body` is whole") and "Do not build it again" in told
    assert brain.news[-1].startswith("The model body is whole")
    await dmn.whole_yet(settings.purpose)
    assert sum("is whole" in news for news in brain.news) == 1  # the human hears of it once
    brain.close()


def test_what_is_wanted_is_found_by_name_and_forgivingly():
    parts = [part("torso"), part("left_arm"), part("right arm"), part("Leg"), part("hole", on="torso")]
    parts[-1] = parts[-1].model_copy(update={"cut": True})
    has, lacks = build.tally([("torso", 1), ("arms", 2), ("leg", 2), ("hole", 1)], parts)
    assert has == ["torso", "2 × arms"] and lacks == ["2 × leg (has 1)", "hole"]  # a hole is no part of the body
