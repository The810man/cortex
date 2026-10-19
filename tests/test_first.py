"""What the human raises comes first on the model's one slot, and a reply cut off for length is said as that."""

import asyncio

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex.brain import Brain
from cortex.llm import Slots, Truncated
from cortex.regions import Act, Decision


async def test_the_human_s_words_then_what_they_raised_then_everything_else():
    slots, order = Slots(1), []

    async def call(name, rank):
        async with slots.take(rank):
            order.append(name)

    async with slots.take():
        waiting = [asyncio.create_task(call(name, rank)) for name, rank in
                   (("critic", 0), ("their matter", 1), ("motor", 0), ("their words", True), ("their next turn", 1))]
        await asyncio.sleep(0)
    await asyncio.gather(*waiting)
    assert order == ["their words", "their matter", "their next turn", "critic", "motor"]


async def test_a_step_for_what_the_human_asked_goes_first_and_a_cut_off_reply_is_said_for_what_it_is(root):  # noqa: F811
    class Long(ScriptedLLM):
        """Writes a file too long for one reply."""

        async def generate(self, role, system, prompt, schema, **kwargs):
            if schema is Act:
                self.firsts = getattr(self, "firsts", []) + [kwargs.get("first")]
                raise Truncated(2000, 5913, 2000)
            return await super().generate(role, system, prompt, schema, **kwargs)

    brain = Brain(root, llm=Long())
    brain.goal = await brain.emit("pfc", "goal", "write a long report", data={"asked": True})
    plan = await brain.emit("pfc", "plan", "write it", data={"intent": "write report.md", "goal_id": brain.goal.id})
    await brain.regions["motor"].fire(plan)
    assert brain.llm.firsts == [True]  # work the human asked for goes ahead of the other regions
    acted = brain.last_action.content
    assert acted.startswith("could not act\nthe reply was cut off: it ran out of room after 2000 tokens (5913 characters)")
    assert "about 6000 characters at most" in acted and "Validation" not in acted
    assert "too long for one step (cut off after 5913 characters)" in brain.hearing("pfc")

    brain.goal = await brain.emit("pfc", "goal", "a goal of its own")
    await brain.regions["dmn"].think("wander", Decision)
    assert brain.llm.first is False  # what it does for itself waits its turn
    brain.close()
