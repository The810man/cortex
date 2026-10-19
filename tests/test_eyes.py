"""Its eyes: what the window in front shows, described out on the human's machine, and only words go in."""

import json
import struct
import zlib

from test_cortex import ScriptedLLM, root  # noqa: F401

from cortex import bridge, container, eyes
from cortex.brain import Brain, Reply
from cortex.config import BrainSettings
from cortex.regions import Act


def test_a_picture_is_made_small_and_written_as_a_png_without_any_library():
    width, height = 3000, 10
    bgra = bytes([10, 20, 30, 0]) * (width * height)  # blue 10, green 20, red 30
    w, h, rgb = eyes.shrink(width, height, bgra, side=1000)
    assert (w, h) == (1000, 4) and rgb[:3] == bytes([30, 20, 10]) and len(rgb) == w * h * 3
    picture = eyes.png(w, h, rgb)
    assert picture.startswith(b"\x89PNG\r\n\x1a\n") and struct.unpack(">II", picture[16:24]) == (1000, 4)
    rows = zlib.decompress(picture[41:-12])  # each row: a filter byte, then the pixels
    assert len(rows) == h * (1 + w * 3) and rows[1:4] == bytes([30, 20, 10])


async def test_it_looks_closer_at_the_window_in_front_and_only_words_come_in(root, monkeypatch):  # noqa: F811
    settings = BrainSettings.load(root)
    settings.safety.allow_screen = True
    settings.save(root)
    screen = {"screen": [1920, 1080], "windows": [{"app": "FreeCAD", "title": "body.FCStd - FreeCAD 1.1", "front": True}]}
    looked = []

    def look_at(model, host):  # stands in for the camera and the vision model out there
        looked.append(model)
        return {"window": screen["windows"][0]["title"], "seen": "A grey torso with a round head on it.", "seconds": 2.1}

    monkeypatch.setattr(container, "look", lambda: json.dumps(screen))
    monkeypatch.setattr(container, "look_at", look_at)
    lent = bridge.Bridge({"look": lambda ask: container.eyes(root), "see": lambda ask: container.sight(root)})
    try:
        monkeypatch.setenv(bridge.BRIDGE, lent.address())
        brain = Brain(root, llm=ScriptedLLM())
        await brain.glance()  # the first look: names and titles, and a closer look at the window in front
        await brain._peering
        assert looked == ["qwen3.5:2b-q4_K_M"] and brain.vision["seen"] == "A grey torso with a round head on it."
        closer = brain.memory.recent("thalamus", "percept", 1)[0]["content"]
        assert closer == "You look closer at what the human has in view, “body.FCStd - FreeCAD 1.1”: A grey torso with a round head on it."
        assert "What “body.FCStd - FreeCAD 1.1” showed when you last looked into it" in brain.awareness()
        assert any("A grey torso" in fact for fact in brain.facts("what do you see on my screen?"))
        await brain.glance()
        assert len(looked) == 1  # the same window in front: no second look

        brain.llm.answer = Reply(say="I see your model.")
        await brain.answer("was siehst du auf meinem bildschirm?")  # asked what it sees, it looks first
        assert len(looked) == 2 and "A grey torso" in brain.llm.prompts[Reply][-1]
        acted = await brain.regions["motor"].execute(Act(tool="look"), "look at the screen")
        assert acted.ok and "What “body.FCStd - FreeCAD 1.1” shows: A grey torso" in acted.text

        settings.models.eyes = ""  # no eyes model: it says so once, and sees names only
        settings.save(root)
        notes = brain.bus.subscribe("system")
        assert await brain.peer() is None and await brain.peer() is None
        said = [notes.get_nowait().content for _ in range(notes.qsize())]
        assert len(said) == 1 and "no eyes model" in said[0] and "sees the names of windows" in said[0]
        settings.models.eyes, settings.safety.allow_screen = "qwen3.5:2b-q4_K_M", False
        settings.save(root)  # switched off, the machine that lends its eyes refuses out there
        brain._blind = False
        assert await brain.peer() is None and len(looked) == 3
        brain.close()
    finally:
        lent.shutdown()
        lent.server_close()
