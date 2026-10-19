"""The executive's neurons foresee what a step will come to, and a step they foresee failing is decided once more."""

import numpy as np
from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex.brain import Brain
from cortex.neural import ACTIVE, NEURONS, Net
from cortex.regions import Decision


def pattern(rng, near=None):
    bits = np.zeros(NEURONS, dtype=bool)
    if near is None:
        bits[rng.choice(NEURONS, ACTIVE, replace=False)] = True
    else:
        bits[rng.choice(np.flatnonzero(near), ACTIVE * 3 // 4, replace=False)] = True
        bits[rng.choice(np.flatnonzero(~near), ACTIVE // 4, replace=False)] = True
    return bits


def test_a_tract_learns_what_steps_setting_out_from_its_neurons_come_to(tmp_path):
    rng, net, path = np.random.default_rng(3), Net(tmp_path / "synapses.npz", seed=0), ("pfc", "motor")
    failing, working = pattern(rng), pattern(rng)
    assert net.foresee(failing, path) is None  # nothing has set out along it yet
    for step in range(4):
        net.bind(2 * step, failing, pattern(rng), path)
        net.bind(2 * step + 1, working, pattern(rng), path)
        net.reinforce(2 * step, -0.9)
        net.reinforce(2 * step + 1, 0.8)
    assert net.foresee(failing, path) < -0.25 < 0.25 < net.foresee(working, path)
    assert net.foresee(pattern(rng, near=failing), path) < -0.25  # a step like one that failed is foreseen to fail
    net.born = False
    net.save()
    assert Net(tmp_path / "synapses.npz").foresee(failing, path) == net.foresee(failing, path)


async def test_a_step_the_neurons_foresee_failing_is_decided_once_more(root):  # noqa: F811
    brain = Brain(root, llm=ScriptedLLM())
    pfc = brain.regions["pfc"]
    brain.goal = goal = await brain.emit("pfc", "goal", "print 6 times 7")
    failed = await brain.emit("motor", "action", "run_python(calc.py)\nexit code 1", parent=goal.id, data={
        "ok": False, "grounded": -0.4, "goal_id": goal.id, "tool": "run_python", "path": "calc.py",
        "repeat": False, "measured": True})
    brain.net.tract(("pfc", "motor")).foresight = np.full(NEURONS, -0.8, dtype=np.float32)  # every step foreseen bad
    plans = brain.bus.subscribe("plan")
    await pfc.fire(failed)
    decided = brain.llm.prompts[Decision]
    assert len(decided) == 2 and "Your neurons foresee that this step fails (-0.80)" in decided[1]
    assert "Decide once more" in decided[1] and "foresee" not in decided[0]
    assert abs(plans.get_nowait().data["foreseen"] + 0.8) < 1e-6

    worked = await brain.emit("motor", "action", "run_python(calc.py)\nran successfully\n42", parent=goal.id, data={
        "ok": True, "grounded": 0.6, "goal_id": goal.id, "tool": "run_python", "path": "calc.py",
        "repeat": False, "measured": True})
    asked = len(decided)
    await pfc.fire(worked)  # right after a step that worked, the neurons are not asked
    assert len(brain.llm.prompts[Decision]) == asked + 1
    brain.close()
