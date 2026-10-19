"""A long sleep: hours of it, with as many rounds of night school as make it better."""

import json
import time

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex import bridge
from cortex.brain import Brain
from cortex.config import BrainSettings
from cortex.regions import Decision


async def test_a_long_sleep_goes_to_school_while_it_gets_better_then_wakes(root, monkeypatch):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.learning.night_lessons = 2
    settings.save(root)
    asked, scores = [], {1: 27, "1-2": 28, "1-3": 28}  # the third round is no better than the second

    def school(ask):  # stands in for the launcher: each round trains at once and sits the exam
        if ask.get("start"):
            plan = json.loads((root / "mind" / f"night-{ask['night']}" / "plan.json").read_text(encoding="utf-8"))
            asked.append((ask["night"], plan["epochs"]))
        return json.dumps({"stage": "done", "kept": True, "model": f"granite4.1-night-{ask['night']}", "lessons": 4,
                           "minutes": 0.01, "exam_before": 26, "exam_after": scores[ask["night"]], "exam_of": 28})

    lent = bridge.Bridge({"school": school})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        pfc, stem = brain.regions["pfc"], brain.regions["brainstem"]
        for worth in (0.8, 0.6, -0.7, -0.9):
            await pfc.think("Goal: print 6 times 7", Decision)
            plan = await brain.emit("pfc", "plan", f"step worth {worth}")
            brain.apply_reward(plan.id, worth, "critic", "judged")
        brain.sleep_for(8)
        assert brain.sleep_requested and brain.sleep_until > time.time() + 7.9 * 3600
        notes = brain.bus.subscribe("system")
        await brain.school()  # the night's first round, as every night
        await stem.long_sleep()
        assert asked == [(1, 1), ("1-2", 2), ("1-3", 3)]  # each round learns one pass longer
        assert brain.settings.models.main == "granite4.1-night-1-2"  # the best round is kept, not the last
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert any("no better than the round before it (28 on the exam)" in n for n in said)
        assert any("after 3 rounds of school: round 3 did no better" in n for n in said)

        brain.night_round, brain.night_best = 1, None
        brain.sleep_for(8)
        await brain.school()
        brain.inject("guten morgen")  # spoken to, it wakes: the next round is not begun
        assert brain.wake_up
        before = len(asked)
        await stem.long_sleep()
        assert len(asked) == before
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert any("it was woken" in n for n in said)

        brain.night_round, brain.night_best, brain.wake_up = 1, None, False
        brain.sleep_until = time.time() + 30  # no time for another round of a minute or more
        await brain.school()
        await stem.long_sleep()
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert any("there is no time left for another round" in n for n in said)
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()
