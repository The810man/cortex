"""What the human's machine lends a brain besides its screen: a camera, ears and hands.

Each is a plain means, not a skill: a picture from the camera put into words, a few seconds of
what the microphone hears written down, a click, a key or some typing in a window. When to use
them, and what for, the brain has to find out, as it found out how to build. Each is switched on
in the brain's settings and off by default, and the switch is read out here at every use.

As with the eyes, nothing but words goes in: no picture and no sound enters its container, its
memory or its prompts. The hands act only in the programs the human named, never in a terminal,
and not while the human is using the mouse or the keyboard themselves.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

from . import eyes
from .config import BrainSettings

CAMERA_LOOK = (
    "This is a picture from the camera of a computer, which looks at whoever sits in front of "
    "its screen. Say in three to five plain sentences what is to be seen: whether a person is "
    "there, what they look like and do, their face and how they look, and what is around them. "
    "Say nothing you cannot see."
)
EARS = Path.home() / ".cortex" / "ears"  # where the model that writes down what it hears is kept
HEARING = "small"  # 480 MB, many languages, a few seconds of speech in a second or two on a processor
LONGEST = 20.0  # seconds it may listen at once
RATE = 16000
QUIET = 0.01  # a peak below this is a quiet room (measured here: 0.016 with nobody speaking); the rest is sorted by voice detection
BUSY = 3.0  # seconds since the human last touched mouse or keyboard in which the hands keep still
TERMINALS = {"windowsterminal", "openconsole", "conhost", "powershell", "pwsh", "cmd", "wt", "mintty", "bash"}
CONTROLS = {  # the kinds of thing in a window that can be named and clicked
    "ButtonControl", "MenuItemControl", "TabItemControl", "ListItemControl", "TreeItemControl",
    "CheckBoxControl", "RadioButtonControl", "EditControl", "ComboBoxControl", "HyperlinkControl",
    "SplitButtonControl",
}
_camera, _hands = threading.Lock(), threading.Lock()
_heard_by = None  # the model that writes down speech, loaded the first time it listens
_acted = 0.0  # when the hands last did something: what the system counts as the human's last input


# -- the camera ---------------------------------------------------------------------


def snapshot(index: int = 0) -> bytes:
    """One picture from the camera, as a PNG at most eyes.SIDE pixels long. The camera is opened
    for it and let go after it, so its light is on only while it looks."""
    import cv2

    with _camera:
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not camera.isOpened():
                raise OSError("no camera could be opened on this machine")
            picture = None
            for _ in range(12):  # a camera that has just woken sees dark until its exposure settles
                ok, frame = camera.read()
                if ok:
                    picture = frame
            if picture is None:
                raise OSError("the camera gave no picture")
        finally:
            camera.release()
    high, wide = picture.shape[:2]
    scale = eyes.SIDE / max(high, wide)
    if scale < 1:
        picture = cv2.resize(picture, (int(wide * scale), int(high * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", picture)
    if not ok:
        raise OSError("the camera's picture could not be read")
    return encoded.tobytes()


def look_through_camera(model: str, host: str, question: str = "") -> dict:
    """What the camera sees, in words, with an eye to `question`."""
    started = time.monotonic()
    words = eyes.describe(snapshot(), model, host, CAMERA_LOOK, question)
    if not words:
        raise OSError("the eyes saw nothing in the camera's picture they could put into words")
    return {"seen": words[:1000], "seconds": round(time.monotonic() - started, 1)}


# -- the ears -----------------------------------------------------------------------


def hearing():
    global _heard_by
    if _heard_by is None:
        from faster_whisper import WhisperModel

        EARS.mkdir(parents=True, exist_ok=True)
        _heard_by = WhisperModel(HEARING, device="cpu", compute_type="int8", download_root=str(EARS))
    return _heard_by


def wav(audio) -> bytes:
    """Sound as a WAV file: one channel, 16 bits, RATE samples a second."""
    import io
    import wave

    import numpy as np

    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def heard(audio, raw: bool) -> dict:
    """What was heard, for a brain: the sound itself as WAV for a mind that hears (`raw`), or what
    was said, written down, for one that only reads."""
    import base64

    import numpy as np

    seconds = round(len(audio) / RATE, 1)
    if not len(audio) or float(np.abs(audio).max()) < QUIET:
        return {"heard": "", "seconds": seconds, "quiet": True, "language": ""}
    if raw:
        return {"audio": base64.b64encode(wav(audio)).decode(), "seconds": seconds, "quiet": False}
    segments, info = hearing().transcribe(audio, beam_size=1, vad_filter=True)
    words = " ".join(segment.text.strip() for segment in segments).strip()
    return {"heard": words[:1500], "seconds": seconds, "quiet": False, "language": info.language if words else ""}


def listen(seconds: float, raw: bool = False) -> dict:
    """Listen through the microphone for so many seconds. It waits while its own voice is
    speaking, so as not to hear only itself."""
    import sounddevice

    from . import voice

    seconds = max(2.0, min(LONGEST, seconds))
    waited = time.monotonic()
    while voice.SPEAKING.is_set() and time.monotonic() - waited < 30:
        time.sleep(0.2)
    recording = sounddevice.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
    sounddevice.wait()
    return heard(recording[:, 0], raw)


class Ear:
    """The microphone held open for the human, on their machine. Pressed to talk: it hears from
    one press to the next. Always: it hears all the time and cuts out each thing said, from the
    first sound of a voice to a pause after it. While its own voice speaks, and a moment after,
    it hears nothing, so as not to answer itself."""

    BLOCK = 480  # 30 ms
    PAUSE = 0.8  # seconds of quiet that end something said
    SHORTEST = 0.6  # seconds: shorter is a cough or a click

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stream = None
        self.mode = "off"
        self.held: list = []  # what push-to-talk has heard since the press
        self.said: list = []  # always: what is being said now
        self.quiet_for = 0.0
        self.floor = 0.004  # the room's own loudness, learned as it listens
        self.before: list = []  # the moment before a voice began
        self.ready: list = []  # things said, cut out, waiting to be fetched
        self.mute_until = 0.0

    def _open(self) -> None:
        import sounddevice

        if self.stream is None:
            self.stream = sounddevice.InputStream(
                samplerate=RATE, channels=1, dtype="float32", blocksize=self.BLOCK, callback=self._block
            )
            self.stream.start()

    def _close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _block(self, data, frames, when, status) -> None:
        self.take(data[:, 0].copy())

    def take(self, block) -> None:
        """One block of sound from the microphone."""
        import numpy as np

        from . import voice

        with self.lock:
            if self.mode == "push":
                self.held.append(block)
                return
            if self.mode != "always":
                return
            if voice.SPEAKING.is_set():
                self.mute_until = time.monotonic() + 0.6
            if time.monotonic() < self.mute_until:
                self.said, self.before = [], []
                return
            level = float(np.sqrt(np.mean(block**2)))
            loud = level > max(0.012, 3 * self.floor)
            if not self.said:
                if not loud:
                    self.floor = 0.98 * self.floor + 0.02 * level
                self.before = (self.before + [block])[-10:]
                if loud:
                    self.said, self.quiet_for = self.before + [block], 0.0
                return
            self.said.append(block)
            self.quiet_for = 0.0 if loud else self.quiet_for + len(block) / RATE
            length = sum(map(len, self.said)) / RATE
            if self.quiet_for >= self.PAUSE or length >= LONGEST:
                audio = np.concatenate(self.said)
                self.said = []
                if length - self.quiet_for >= self.SHORTEST:
                    self.ready.append(audio)

    def set(self, mode: str) -> None:
        with self.lock:
            self.mode = mode
            self.held, self.said, self.ready = [], [], []
        if mode == "off":
            self._close()
        else:
            self._open()

    def press(self) -> bool:
        """Push-to-talk: the first press opens the ear, the next closes it, and what was said
        between waits to be fetched. Returns whether it is listening now."""
        import numpy as np

        if self.mode != "push":
            self.set("push")
            return True
        with self.lock:
            audio = np.concatenate(self.held) if self.held else np.zeros(0, dtype="float32")
            self.held, self.mode = [], "off"
            self.ready.append(audio)
        self._close()
        return False

    def fetch(self):
        """The next thing said, or None. In always mode a stretch that held no voice is let go."""
        while True:
            with self.lock:
                audio = self.ready.pop(0) if self.ready else None
                always = self.mode == "always"
            if audio is None or not always or speaks(audio):
                return audio


def speaks(audio) -> bool:
    """Whether a stretch of sound holds a voice, by the voice detector Whisper comes with."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        return sum(s["end"] - s["start"] for s in get_speech_timestamps(audio, VadOptions())) >= 0.4 * RATE
    except Exception:
        return True


EAR = Ear()


# -- the hands ----------------------------------------------------------------------


def _user():
    import ctypes
    from ctypes import wintypes

    user = ctypes.windll.user32
    user.SetForegroundWindow.argtypes = user.BringWindowToTop.argtypes = [wintypes.HWND]
    user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user.GetForegroundWindow.restype = wintypes.HWND
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    user.VkKeyScanW.argtypes = [wintypes.WCHAR]
    user.VkKeyScanW.restype = ctypes.c_short
    return ctypes, wintypes, user


def human_busy() -> bool:
    """Whether the human touched the mouse or the keyboard in the last few seconds."""
    ctypes, wintypes, user = _user()

    class LastInput(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    last = LastInput(ctypes.sizeof(LastInput), 0)
    if not user.GetLastInputInfo(ctypes.byref(last)):
        return False
    idle = (ctypes.windll.kernel32.GetTickCount() - last.dwTime) / 1000
    return idle < BUSY and time.monotonic() - _acted > idle + 0.5  # the last input was not its own


def bring_forward(window: dict) -> None:
    """Put the window in front, so that what the hands do goes to it."""
    ctypes, _, user = _user()
    handle = window["handle"]
    if window["minimised"]:
        user.ShowWindow(handle, 9)  # SW_RESTORE
    front = user.GetForegroundWindow()
    if front == handle:
        return
    mine = ctypes.windll.kernel32.GetCurrentThreadId()
    theirs = user.GetWindowThreadProcessId(front, None) if front else 0
    joined = bool(theirs) and user.AttachThreadInput(mine, theirs, True)  # a program in the back may not take the front alone
    try:
        user.BringWindowToTop(handle)
        user.SetForegroundWindow(handle)
    finally:
        if joined:
            user.AttachThreadInput(mine, theirs, False)
    time.sleep(0.25)
    if user.GetForegroundWindow() != handle:
        raise OSError(f"“{window['title']}” could not be brought to the front, so nothing was done in it")


def controls(window: dict, limit: int = 50, seconds: float = 3.0) -> list[tuple[str, str, object]]:
    """The things in a window that have a name and can be clicked, as the window itself says:
    (kind, name, control), the menus and buttons first, at most a few seconds of looking."""
    import uiautomation

    found, until = [], time.monotonic() + seconds
    root = uiautomation.ControlFromHandle(window["handle"])
    for control, _ in uiautomation.WalkControl(root, includeTop=False, maxDepth=12):
        if time.monotonic() > until or len(found) >= limit * 3:
            break
        name = (control.Name or "").strip()
        if control.ControlTypeName in CONTROLS and name and len(name) < 60:
            try:
                if control.IsOffscreen:
                    continue
            except Exception:
                continue
            found.append((control.ControlTypeName.removesuffix("Control"), name, control))
    order = ["MenuItem", "Button", "SplitButton", "TabItem", "CheckBox", "RadioButton", "ComboBox", "Edit"]
    found.sort(key=lambda item: order.index(item[0]) if item[0] in order else len(order))
    unique, seen = [], set()
    for item in found:
        if (item[0], item[1]) not in seen:
            seen.add((item[0], item[1]))
            unique.append(item)
    return unique[:limit]


def named_controls(target: str) -> str:
    """The things in a window that the hands can click by name, in a line."""
    window = eyes.find(target)
    if window is None:
        return ""
    found = controls(window)
    if not found:
        return ""
    kinds: dict[str, list[str]] = {}
    for kind, name, _ in found:
        kinds.setdefault(kind, []).append(f"“{name}”")
    return f"What can be clicked by name in “{window['title']}”: " + "; ".join(
        f"{kind}: {', '.join(names)}" for kind, names in kinds.items()
    )


KEYS = {
    "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "tab": 0x09, "space": 0x20, "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    **{f"f{n}": 0x6F + n for n in range(1, 13)},
}
MODIFIERS = {"ctrl": 0x11, "control": 0x11, "strg": 0x11, "alt": 0x12, "shift": 0x10}


def press(combo: str) -> str:
    """Press keys together, such as ctrl+s, enter, 1 or shift+tab."""
    _, _, user = _user()
    parts = [part for part in re.split(r"\s*\+\s*", combo.strip().lower()) if part]
    if not parts:
        raise ValueError("no key named: say which, such as `key ctrl+s` or `key enter`")
    if any(part in ("win", "windows", "super", "meta", "cmd") for part in parts):
        raise ValueError("the Windows key is not for the hands: it reaches the whole machine, not the window")
    held, key = [MODIFIERS[p] for p in parts[:-1] if p in MODIFIERS], parts[-1]
    if len(held) != len(parts) - 1:
        raise ValueError(f"not keys that can be held: {combo}. Use ctrl, alt or shift, then one key")
    if key in KEYS:
        code = KEYS[key]
    elif len(key) == 1:
        scan = user.VkKeyScanW(key)
        if scan == -1:
            raise ValueError(f"no key on this keyboard makes “{key}”")
        code = scan & 0xFF
    else:
        raise ValueError(f"no key is called “{key}”. Named keys: {', '.join(sorted(KEYS))}, or one letter")
    if held == [0x11, 0x12] and code == 0x2E or (0x11 in held and 0x10 in held and code == 0x1B):
        raise ValueError("that key is the machine's own, not the window's")
    for vk in held:
        user.keybd_event(vk, 0, 0, 0)
    user.keybd_event(code, 0, 0, 0)
    user.keybd_event(code, 0, 2, 0)
    for vk in reversed(held):
        user.keybd_event(vk, 0, 2, 0)
    return f"pressed {combo.strip()}"


def type_text(text: str) -> str:
    """Type text into whatever has the focus, letter by letter, as if on a keyboard."""
    ctypes, wintypes, user = _user()

    class KeyInput(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class Input(ctypes.Structure):
        class Union(ctypes.Union):
            _fields_ = [("ki", KeyInput), ("padding", ctypes.c_byte * 32)]

        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", Union)]

    text = text[:400]
    for letter in text:
        if letter == "\n":
            press("enter")
            continue
        for flags in (0x0004, 0x0004 | 0x0002):  # KEYEVENTF_UNICODE, then with KEYUP
            event = Input(type=1)
            event.ki = KeyInput(0, ord(letter), flags, 0, 0)
            user.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))
        time.sleep(0.01)
    return f"typed {len(text)} characters"


def point(window: dict, x: float, y: float) -> tuple[int, int]:
    """A place in a window, given as fractions of its width and height from its top left."""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError(f"{x} {y} is not in the window: a place is two numbers from 0 to 1, across then down")
    left, top, wide, high = eyes.frame(window["handle"])
    return int(left + x * (wide - 1)), int(top + y * (high - 1))


BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}


def click_at(at: tuple[int, int], button: str = "left", times: int = 1) -> None:
    _, _, user = _user()
    down, up = BUTTONS[button]
    user.SetCursorPos(*at)
    time.sleep(0.05)
    for _ in range(times):
        user.mouse_event(down, 0, 0, 0, 0)
        user.mouse_event(up, 0, 0, 0, 0)
        time.sleep(0.08)


def drag(start: tuple[int, int], end: tuple[int, int], button: str = "left") -> None:
    _, _, user = _user()
    down, up = BUTTONS[button]
    user.SetCursorPos(*start)
    user.mouse_event(down, 0, 0, 0, 0)
    for step in range(1, 21):  # moved, not jumped: a program sees a drag only as a movement
        user.SetCursorPos(start[0] + (end[0] - start[0]) * step // 20, start[1] + (end[1] - start[1]) * step // 20)
        time.sleep(0.015)
    user.mouse_event(up, 0, 0, 0, 0)


def scroll(at: tuple[int, int], notches: int) -> None:
    _, _, user = _user()
    user.SetCursorPos(*at)
    user.mouse_event(0x0800, 0, 0, 120 * notches, 0)


NUMBER = r"(-?\d+(?:\.\d+)?)"
PLACE = re.compile(rf"^(?:at\s+)?{NUMBER}[\s,;]+{NUMBER}$")


def find_control(window: dict, name: str):
    """The control in a window a name means: the one called exactly that, else the first whose
    name holds it. Raises ValueError naming what there is."""
    found = controls(window, limit=200)
    wanted = name.strip().strip("“”\"'").lower()
    exact = [item for item in found if item[1].lower() == wanted]
    near = [item for item in found if wanted in item[1].lower()]
    if exact or near:
        return (exact or near)[0]
    there = ", ".join(f"“{n}”" for _, n, _ in found[:30])
    raise ValueError(f"nothing in “{window['title']}” is called “{name}”. What has a name there: {there or 'nothing'}")


def act(target: str, command: str, programs: list[str]) -> str:
    """Do one thing with the hands in a window: `target` names the window, `command` what to do.

        click <name> | click <x> <y> | double-click … | right-click …
        menu <name> > <name> > …
        type <text> | key <keys, such as ctrl+s> | scroll up|down <notches> [at <x> <y>]
        drag <x> <y> to <x> <y> [with left|middle|right] | move <x> <y>

    A place is two numbers from 0 to 1: across the window, then down it."""
    global _acted
    if sys.platform != "win32":
        raise OSError("there are no hands on this machine")
    with _hands:
        window = eyes.find(target)
        if window is None:
            raise ValueError("the hands act in a window: name it in `path`, such as FreeCAD")
        app = window["app"].lower()
        if window["own"] or app in TERMINALS:
            raise ValueError(f"“{window['title']}” is a terminal: the hands never act in one")
        allowed = [p.lower() for p in programs]
        if not any(app == p or app.startswith(p) for p in allowed):
            raise ValueError(
                f"the hands may act only in {', '.join(programs) or 'no program'}, as the human set it, "
                f"and “{window['title']}” belongs to {window['app'] or 'another program'}"
            )
        if human_busy():
            raise OSError("the human is using the mouse or the keyboard right now: the hands keep still. Try again in a moment")
        verb, _, rest = command.strip().partition(" ")
        verb, rest = verb.lower().rstrip(":"), rest.strip()
        bring_forward(window)
        try:
            if verb in ("click", "double-click", "doubleclick", "right-click", "rightclick", "klick", "klicke"):
                button = "right" if verb.startswith("right") else "left"
                times = 2 if verb.startswith("double") else 1
                place = PLACE.match(rest)
                if place:
                    click_at(point(window, float(place[1]), float(place[2])), button, times)
                    return f"{verb} at {place[1]} {place[2]} in “{window['title']}”"
                kind, name, control = find_control(window, rest)
                box = control.BoundingRectangle
                click_at(((box.left + box.right) // 2, (box.top + box.bottom) // 2), button, times)
                return f"{verb} on “{name}” ({kind}) in “{window['title']}”"
            if verb == "menu":
                steps = [step.strip() for step in re.split(r">|→|/", rest) if step.strip()]
                if not steps:
                    raise ValueError("name the menu and its item, such as `menu File > Save`")
                for step in steps:
                    kind, name, control = find_control(window, step)
                    box = control.BoundingRectangle
                    click_at(((box.left + box.right) // 2, (box.top + box.bottom) // 2))
                    time.sleep(0.4)  # a menu takes a moment to open
                return f"chose {' > '.join(steps)} in “{window['title']}”"
            if verb in ("type", "write", "tippe"):
                return type_text(rest.strip("“”\"")) + f" into “{window['title']}”"
            if verb in ("key", "keys", "press", "taste"):
                return press(rest) + f" in “{window['title']}”"
            if verb == "scroll":
                way, _, more = rest.partition(" ")
                if way.lower() not in ("up", "down"):
                    raise ValueError("a scroll is `scroll up <n>` or `scroll down <n>`, optionally `at <x> <y>`")
                count = re.match(r"\s*(\d+)", more)
                notches = min(int(count[1]), 20) if count else 3
                place = re.search(rf"at\s+{NUMBER}[\s,;]+{NUMBER}", more)
                at = point(window, float(place[1]), float(place[2])) if place else point(window, 0.5, 0.5)
                scroll(at, notches if way.lower() == "up" else -notches)
                return f"scrolled {way} {notches} in “{window['title']}”"
            if verb == "drag":
                found = re.match(rf"^(?:from\s+)?{NUMBER}[\s,;]+{NUMBER}\s+to\s+{NUMBER}[\s,;]+{NUMBER}(?:\s+with\s+(\w+))?", rest)
                if not found:
                    raise ValueError("a drag is `drag <x> <y> to <x> <y>`, optionally `with middle` or `with right`")
                button = (found[5] or "left").lower()
                if button not in BUTTONS:
                    raise ValueError("the buttons are left, middle and right")
                drag(point(window, float(found[1]), float(found[2])), point(window, float(found[3]), float(found[4])), button)
                return f"dragged from {found[1]} {found[2]} to {found[3]} {found[4]} with the {button} button in “{window['title']}”"
            if verb == "move":
                place = PLACE.match(rest)
                if not place:
                    raise ValueError("a move is `move <x> <y>`")
                _user()[2].SetCursorPos(*point(window, float(place[1]), float(place[2])))
                return f"moved the pointer to {place[1]} {place[2]} in “{window['title']}”"
            raise ValueError(
                f"the hands do not know “{verb}”. They can: click, double-click, right-click (a name, or x y), "
                "menu A > B, type <text>, key <keys>, scroll up|down <n>, drag <x> <y> to <x> <y>, move <x> <y>"
            )
        finally:
            _acted = time.monotonic()


# -- lent to a brain ------------------------------------------------------------------

SWITCHES = {
    "camera": ("allow_camera", "the camera is switched off in this brain's settings"),
    "listen": ("allow_microphone", "the microphone is switched off in this brain's settings"),
    "ear": ("allow_microphone", "the microphone is switched off in this brain's settings"),
    "hands": ("allow_hands", "the hands are switched off in this brain's settings"),
}


def serve(root: Path, do: str, ask: dict) -> str:
    """Do for a brain what it asked of the camera, the ears or the hands, if its settings, read
    out here as they are now, allow it. The answer is a line of JSON or text, never a picture."""
    switch, refusal = SWITCHES[do]
    try:
        settings = BrainSettings.load(root)
    except Exception:  # caught while the brain was writing them
        raise OSError("the brain's settings could not be read just now") from None
    if not getattr(settings.safety, switch):
        raise OSError(refusal)
    try:
        if do == "camera":
            if ask.get("raw"):  # a mind that sees itself is given the picture
                import base64

                return json.dumps({"picture": base64.b64encode(snapshot()).decode()})
            if not settings.models.eyes:
                raise OSError("this brain has no eyes model to say what the camera sees")
            return json.dumps(look_through_camera(settings.models.eyes, settings.models.host, str(ask.get("question", ""))))
        if do == "listen":
            return json.dumps(listen(float(ask.get("seconds") or 5), bool(ask.get("raw"))))
        if do == "ear":  # the human's own talk: pressed to talk, or always listening
            if ask.get("mode") in ("off", "push", "always"):
                if ask["mode"] != EAR.mode and not (ask["mode"] == "off" and EAR.mode == "push"):
                    EAR.set(ask["mode"])
                return json.dumps({"mode": EAR.mode})
            if ask.get("press"):
                return json.dumps({"listening": EAR.press(), "mode": EAR.mode})
            audio = EAR.fetch()
            said = {"said": heard(audio, bool(ask.get("raw")))} if audio is not None else {}
            return json.dumps({"mode": EAR.mode, **said})
        return act(str(ask.get("target", "")), str(ask.get("command", "")), settings.safety.hands_in)
    except (OSError, ValueError):
        raise
    except Exception as exc:  # a library out here that failed: said, not raised past the bridge
        raise OSError(f"its {do} failed: {str(exc)[:160] or type(exc).__name__}") from None
