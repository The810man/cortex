"""In sleep the thinking model learns from the day; what it learned is kept only if it passes the exam."""

import json

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex import bridge
from cortex.brain import Brain
from cortex.config import BrainSettings
from cortex.regions import Decision


async def test_night_school_teaches_from_the_days_replies_and_keeps_only_what_passes(root, monkeypatch):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.learning.night_lessons = 2
    settings.save(root)
    asked = []

    def school(ask):  # stands in for the launcher: trains at once, and passes or fails the exam
        asked.append(ask)
        if ask.get("start"):
            lessons = (root / "mind" / f"night-{ask['night']}" / "lessons.jsonl").read_text(encoding="utf-8")
            assert len(lessons.splitlines()) == 4 and '"label": true' in lessons and '"label": false' in lessons
        kept = ask["night"] == 1
        return json.dumps({"stage": "done", "kept": kept, "model": f"granite4.1-night-{ask['night']}", "lessons": 4,
                           "minutes": 12.5, "exam_before": 27, "exam_after": 28 if kept else 20, "exam_of": 28})

    lent = bridge.Bridge({"school": school})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        notes = brain.bus.subscribe("system")
        await brain.school()
        assert "Night school waits: 0 good and 0 bad" in notes.get_nowait().content and not asked

        pfc = brain.regions["pfc"]
        for worth in (0.8, 0.6, -0.7, -0.9):  # four replies of its own, two that went well and two that did not
            await pfc.think("Goal: print 6 times 7", Decision)
            plan = await brain.emit("pfc", "plan", f"step worth {worth}")
            brain.apply_reward(plan.id, worth, "critic", "judged")
        await brain.school()
        assert asked[0] == {"night": 1, "model": "granite4.1:8b", "engine": "ollama", "best": None, "start": True}
        assert brain.settings.models.main == "granite4.1-night-1"
        assert brain.settings.models.untaught == "granite4.1:8b"  # what it learns on stays the model it was
        assert BrainSettings.load(root).models.main == "granite4.1-night-1"
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert any("exam 27 before and 28 after, of 28: kept, it now thinks as granite4.1-night-1" in n for n in said)

        brain.sleeps = 1
        await brain.school()  # the next night fails its exam: it goes on thinking as it did
        assert asked[-1]["model"] == "granite4.1:8b" and brain.settings.models.main == "granite4.1-night-1"
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert any("exam 27 before and 20 after, of 28: not kept" in n for n in said)
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()


async def test_a_mind_on_llama_cpp_keeps_what_its_night_taught_as_an_adapter(root, monkeypatch):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.learning.night_lessons = 1
    settings.models.engine = "llama"
    settings.save(root)
    asked = []

    def school(ask):
        asked.append(ask)
        return json.dumps({"stage": "done", "kept": True, "lora": "mind/night-1/adapter.gguf", "lessons": 2,
                           "model": "gemma-4-E4B with mind/night-1/adapter.gguf", "minutes": 9.0,
                           "exam_before": 26, "exam_after": 26, "exam_of": 28})

    lent = bridge.Bridge({"school": school})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        llm = ScriptedLLM()
        llm.senses = True
        brain = Brain(root, llm=llm)
        pfc = brain.regions["pfc"]
        for worth in (0.8, -0.7):
            await pfc.think("Goal: print 6 times 7", Decision)
            plan = await brain.emit("pfc", "plan", f"step worth {worth}")
            brain.apply_reward(plan.id, worth, "critic", "judged")
        await brain.school()
        assert asked[0]["engine"] == "llama" and asked[0]["start"]
        kept = BrainSettings.load(root).models
        assert kept.llama_lora == "mind/night-1/adapter.gguf" and kept.main == "granite4.1:8b"  # Ollama's model is not touched
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()
