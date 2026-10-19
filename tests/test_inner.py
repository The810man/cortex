"""It thinks on: an inner voice beside the thinker, wants of its own, interests and open questions."""

import time

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex.brain import Brain
from cortex.regions import Decision, Wander


async def test_the_inner_voice_murmurs_beside_the_thinker_and_is_heard(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    dmn, heard = brain.regions["dmn"], brain.bus.subscribe("musing")
    brain.goal = await brain.emit("pfc", "goal", "print 6 times 7")  # it murmurs while a goal is in hand too
    await dmn.murmur()
    musing = heard.get_nowait()
    assert musing.content == "I wonder whether the script will print 42. (1)" and musing.source == "dmn"
    told = brain.llm.murmured[0]
    assert "What I am doing: The mind is working on: print 6 times 7" in told and "What I want most right now" in told
    await dmn.murmur()
    assert "What I said to myself last, oldest first:\n- I wonder whether" in brain.llm.murmured[1]
    assert "Your inner voice lately, the latest last:\n- I wonder" in brain.awareness()
    assert not brain.memory.recent("dmn", "musing", 1)  # heard, not remembered: one every few seconds would bury the rest
    brain.close()


async def test_it_wants_what_its_situation_lacks(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    brain.woke = brain.heard_at = time.time()
    assert brain.want() == "nothing presses: I can follow whatever interests me."
    brain.heard_at = brain.woke = time.time() - 3600
    assert brain.want().startswith("the human's company: they have been silent for 60 min") and brain.wanting == "company"
    brain.heard_at = time.time()
    for _ in range(6):
        await brain.emit("motor", "action", "run_python(calc.py)\nexit code 1\nSyntaxError")
    assert brain.drives()["success"] == 1.0
    brain.close()


async def test_interests_are_what_it_is_getting_better_at_and_its_questions_stay_open_until_thought_through(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    for value in (-0.6, -0.4, 0.2, 0.6):  # the same kind of goal, ending better each time
        goal = await brain.emit("pfc", "goal", "build a body in FreeCAD with a torso and a head")
        await brain.emit("pfc", "outcome", "DONE: build a body", parent=goal.id, value=value)
    for value in (0.5, 0.2, -0.3, -0.6):  # and one it is getting worse at
        goal = await brain.emit("pfc", "goal", "benchmark the cpu load of pandas threads")
        await brain.emit("pfc", "outcome", "ABANDON: benchmark", parent=goal.id, value=value)
    assert brain.interests() == ["build a body in FreeCAD with a torso and a head"]

    brain.wonder_about("Why do my arms never line up with my shoulders?")
    brain.wonder_about("Why do my arms never line up with my shoulders?")  # the same question once
    assert [q for q, _ in brain.questions] == ["Why do my arms never line up with my shoulders?"]
    await brain.regions["dmn"].fire(None)
    free = [p for schema, prompts in brain.llm.prompts.items() if issubclass(schema, Wander) for p in prompts][-1]
    assert "Questions of your own that are still open:\n- Why do my arms" in free and "This thought is yours" in free
    assert "What you have been getting better at lately" in free and "What you want most right now" in free
    brain.settled("Why do my arms never line up with my shoulders?")
    assert brain.questions == []  # thought through, it is no longer open
    brain.close()
    assert Brain(root, llm=ScriptedLLM()).questions == []


async def test_it_learns_how_well_it_knows_itself(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    pfc = brain.regions["pfc"]
    assert brain.self_knowledge() == ""  # too few steps to say
    brain.goal = goal = await brain.emit("pfc", "goal", "print 6 times 7")
    plans = brain.bus.subscribe("plan")
    await pfc.fire(await brain.emit("motor", "action", "write_file(calc.py)\nwrote 11 characters", parent=goal.id,
                                    data={"ok": True, "grounded": 0.1, "goal_id": goal.id, "tool": "write_file",
                                          "path": "calc.py", "repeat": False, "measured": False}))
    plan = plans.get_nowait()
    assert plan.data["sure"] == 50 and pfc.promised[plan.id] == 50  # what it said, kept with the plan
    await pfc.fire(await brain.emit("motor", "action", "run_python(calc.py)\nexit code 1", parent=plan.id, data={
        "ok": False, "grounded": -0.4, "goal_id": goal.id, "tool": "run_python", "path": "calc.py",
        "repeat": False, "measured": True}))
    assert brain.memory.get_state("calibration") == [[50, False]]
    for _ in range(9):
        brain.knew(90, False)
    for _ in range(3):
        brain.knew(20, True)
    told = brain.self_knowledge()
    assert "when you said you were 70 or more sure, the step worked 0 of 9 times" in told
    assert "when you said you were under 40 sure, the step worked 3 of 3 times" in told
    await pfc.fire(await brain.emit("motor", "action", "read_file(calc.py)\nprint(6*)", parent=goal.id, data={
        "ok": True, "grounded": 0.05, "goal_id": goal.id, "tool": "read_file", "path": "calc.py",
        "repeat": False, "measured": False}))
    assert "How well you know yourself, from your last 13 steps" in brain.llm.prompts[Decision][-1]
    brain.close()
