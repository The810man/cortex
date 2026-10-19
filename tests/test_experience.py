"""What it was shown and what it answered is kept, and labelled at night by what it came to."""

import json

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex.brain import Brain
from cortex.regions import Decision


async def test_every_reply_is_kept_with_the_memory_it_led_to_and_labelled_by_its_worth(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    pfc = brain.regions["pfc"]
    brain.goal = goal = await brain.emit("pfc", "goal", "print 6 times 7")
    reply, _ = await pfc.think("Goal: print 6 times 7", Decision)
    plan = await brain.emit("pfc", "plan", reply.intent, parent=goal.id)
    await brain.emit("thalamus", "percept", "a file appeared")  # no model call led to this: nothing kept for it
    days = list((root / "experience").glob("*.jsonl"))
    kept = [json.loads(line) for line in days[0].read_text(encoding="utf-8").splitlines()]
    assert len(days) == 1 and len(kept) == 1
    assert kept[0]["id"] == plan.id and kept[0]["region"] == "pfc" and kept[0]["prompt"] == "Goal: print 6 times 7"
    assert "You are the prefrontal cortex" in kept[0]["system"] and json.loads(kept[0]["reply"])["status"] == "continue"

    def value(memory_id):
        row = brain.memory.get(memory_id)
        return row["value"] if row else None

    assert brain.experience.lessons(value) == []  # nobody has judged it yet: it teaches nothing
    brain.apply_reward(plan.id, -0.8, "critic", "it repeated a failed step")
    lesson = brain.experience.lessons(value)[0]
    assert lesson["label"] is False and lesson["prompt"][1] == {"role": "user", "content": "Goal: print 6 times 7"}
    assert lesson["completion"][0]["role"] == "assistant" and lesson["value"] < -0.15
    assert brain.experience.lessons(value, since=plan.id) == []  # what was learned from already is not learned again
    brain.close()
