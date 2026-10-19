"""What the motor cortex can do: files in the workspace, and running the code it wrote.

Code execution is guarded, not sandboxed. Imports are checked against an allow-list before a
script runs, a handful of escape hatches are refused, and `open` is confined to the workspace at
run time. That stops a small model from wandering off by accident; it is not a security boundary
against code written to break out.

In full mode none of this applies: paths may lead anywhere and scripts run as written. The brain
only grants it inside its container, where anywhere is still not this machine.

`show` is the one tool that reaches the human's screen: it opens a page or picture from the
workspace in their browser, or a model in FreeCAD. A brain in a container asks the machine that
launched it to do so.

`run_freecad` is how it builds: a script runs inside FreeCAD, which has no window in there, and
what comes back is what was measured of the result: how many solid bodies, how large, whether valid.

What the tools are is said in one place, `Tools.abilities`, from what is really there at that
moment. Every region is told from it, so the one that plans knows what the one that acts can do.
"""

from __future__ import annotations

import ast
import base64
import asyncio
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from . import build
from .bridge import BRIDGE, relay
from .config import SafetySettings, contained

OUTPUT_CAP = 64_000
VIEWABLE = {".html", ".htm", ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"}  # opened, never run
MODELS = {".fcstd"}  # opened in FreeCAD, never run
CAD_MODULES = frozenset({"FreeCAD", "App", "Part", "PartDesign", "Sketcher", "Draft", "Mesh", "MeshPart", "TechDraw"})
CAD_TIMEOUT = 90.0  # FreeCAD takes a few seconds to start before a script's first line runs
CAD_MARK = "@@cortex-cad@@"
SHOW_GAP = 15.0  # seconds between pages: a brain going in circles must not bury the screen in tabs
WEB = ("http://", "https://")
READABLE = {".txt", ".md", ".py", ".json", ".csv", ".html", ".htm", ".sh", ".yaml", ".yml", ".svg", ".log"}
BANNED_NAMES = {"eval", "exec", "compile", "__import__", "input", "breakpoint", "__builtins__"}
BANNED_ATTRS = {
    "__subclasses__", "__globals__", "__builtins__", "__bases__", "__mro__", "__code__",
    "__closure__", "__loader__", "__spec__", "__import__", "__getattribute__",
}

# Runs inside the child interpreter, in the workspace, before the script itself.
RUNNER = r"""
import builtins, io, os, runpy, sys
root = os.path.realpath(os.getcwd())
real_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if not isinstance(file, int):
        try:
            inside = os.path.commonpath([root, os.path.realpath(os.fspath(file))]) == root
        except ValueError:
            inside = False
        if not inside:
            raise PermissionError(f"outside the workspace: {file}")
    return real_open(file, *args, **kwargs)
builtins.open = io.open = guarded_open
sys.path.insert(0, root)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


# Runs inside FreeCAD, in the workspace: first the script, then a look at what it built. The look
# is the brain's eyes in there: it cannot see a picture, so it is told the measurements.
CAD_RUNNER = r"""
import builtins, io, json, os, runpy, sys, traceback
if os.environ.get("CORTEX_CAD_RAN") != "1":  # FreeCAD runs a file again when it thinks the first time failed
    os.environ["CORTEX_CAD_RAN"] = "1"
    script = os.environ["CORTEX_CAD_SCRIPT"]
    root = os.path.realpath(os.getcwd())
    real_open = builtins.open
    if os.environ.get("CORTEX_CAD_GUARD") == "1":
        def guarded_open(file, *args, **kwargs):
            if not isinstance(file, int):
                try:
                    inside = os.path.commonpath([root, os.path.realpath(os.fspath(file))]) == root
                except ValueError:
                    inside = False
                if not inside:
                    raise PermissionError(f"outside the workspace: {file}")
            return real_open(file, *args, **kwargs)
        builtins.open = io.open = guarded_open
    sys.path.insert(0, root)
    import FreeCAD
    # One model per script: saving again must not leave a dated backup beside it each time.
    FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Document").SetBool("CreateBackupFiles", False)
    report = {"error": "", "bodies": [], "saved": "", "drawing": "", "objects": {}}
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        pass
    except BaseException:
        report["error"] = traceback.format_exc(limit=-3)
    try:
        stem = os.environ.get("CORTEX_CAD_STEM") or os.path.splitext(script)[0]
        whole = []
        for doc in FreeCAD.listDocuments().values():
            report["objects"][doc.Name] = [obj.Name for obj in doc.Objects][:12]
            doc.recompute()
            found = False
            for obj in doc.Objects:
                shape = getattr(obj, "Shape", None)
                if shape is None or shape.isNull():
                    continue
                found = True
                box = shape.BoundBox
                top = not obj.InList  # not a part of something else in the model
                report["bodies"].append({
                    "name": obj.Label, "type": obj.TypeId, "solids": len(shape.Solids),
                    "volume": shape.Volume, "valid": bool(shape.isValid()), "top": top,
                    "size": [box.XLength, box.YLength, box.ZLength],
                    "at": [box.XMin, box.YMin, box.ZMin],
                })
                if top:
                    whole.append(shape)
                    obj.Visibility = True
            if found and not report["saved"]:
                doc.saveAs(os.path.join(root, stem + ".FCStd"))
                report["saved"] = stem + ".FCStd"
        if whole:
            import Part, TechDraw
            seen = Part.makeCompound(whole)
            middle, across = seen.BoundBox.Center, seen.BoundBox.DiagonalLength or 1.0
            seen.translate(FreeCAD.Vector(-middle.x, -middle.y, -middle.z))
            lines = TechDraw.projectToSVG(seen, FreeCAD.Vector(1, -1, 0.6))
            lines = lines.replace('stroke-width="1.0"', f'stroke-width="{across / 300:.3f}"')
            edge = across * 0.6
            with real_open(os.path.join(root, stem + ".svg"), "w") as out:
                out.write(
                    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-edge} {-edge} {2 * edge} {2 * edge}" '
                    f'width="800" height="800" style="background:white">{lines}</svg>'
                )
            report["drawing"] = stem + ".svg"
    except BaseException:
        report["error"] = report["error"] or traceback.format_exc(limit=-3)
    sys.stdout.flush()
    print("\n@@cortex-cad@@" + json.dumps(report))
    sys.stdout.flush()
"""


# Runs inside FreeCAD on the human's machine, with its window. A model saved where there was no
# window opens with everything hidden, so this opens it, shows what it holds and turns it to face
# the human. The file to open arrives as data; nothing the brain wrote is run out here.
OPENER = r"""
import os
import FreeCAD, FreeCADGui
doc = FreeCAD.openDocument(os.environ["CORTEX_MODEL"])
for obj in doc.Objects:
    if not obj.InList and obj.ViewObject is not None:
        obj.ViewObject.Visibility = True
try:
    view = FreeCADGui.ActiveDocument.ActiveView
    view.viewIsometric()
    view.fitAll()
except Exception:
    pass
"""
_window: subprocess.Popen | None = None  # the FreeCAD it opened last, so there is only ever one


def open_model(program: str, file: Path) -> None:
    """Put a model on the human's screen in FreeCAD, in place of the one shown before."""
    global _window
    if _window is not None and _window.poll() is None:
        _window.terminate()
    opener = Path(tempfile.gettempdir()) / "cortex_open_model.py"
    opener.write_text(OPENER, encoding="utf-8")
    _window = subprocess.Popen(
        [program, str(opener)], env={**os.environ, "CORTEX_MODEL": str(file)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def freecad_window() -> str | None:
    """FreeCAD with its window, on the human's machine, if it is installed there."""
    found = shutil.which("FreeCAD") or shutil.which("freecad")
    if found:
        return found
    places = [os.environ.get(key, "") for key in ("ProgramFiles", "LOCALAPPDATA", "ProgramFiles(x86)")]
    places += [str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"), str(Path.home() / ".cortex")]
    for place in filter(None, places):
        for exe in sorted(Path(place).glob("FreeCAD*/bin/[Ff]ree[Cc][Aa][Dd].exe"), reverse=True):  # the newest
            return str(exe)
    return None


def freecad_name(program: str | None) -> str:
    """What the FreeCAD at this path is called, from the folder it is installed in; empty for none."""
    if not program:
        return ""
    folder = Path(program).parent.parent.name
    return folder if folder.lower().startswith("freecad") else "FreeCAD"


def sense(root: Path) -> str:
    """What this machine tells a brain in a container about itself: where the brain's folder is
    out here, and whether FreeCAD with a window is installed."""
    return json.dumps({"home": str(root), "freecad": freecad_name(freecad_window())})


def look() -> str:
    """What is on this machine's screen, as JSON: how large it is, and every window that can be
    seen there with the program it belongs to. Names and titles only: no picture is taken."""
    if sys.platform != "win32":
        return json.dumps({"screen": [], "windows": []})
    import ctypes

    from . import eyes

    user = ctypes.windll.user32
    keep = ("app", "title", "front", "minimised", "own")
    windows = [{key: window[key] for key in keep} for window in eyes.windows()]
    return json.dumps({"screen": [user.GetSystemMetrics(0), user.GetSystemMetrics(1)], "windows": windows[:24]})


def machine(workspace: Path) -> str:
    """The machine a brain runs on, in a line: its system, its Python, its processors, memory and disk."""
    try:
        system = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError, AttributeError):
        system = f"{platform.system()} {platform.release()}"
    parts = [system, f"Python {platform.python_version()}", f"{os.cpu_count()} processors"]
    with contextlib.suppress(ValueError, OSError, AttributeError):
        parts.append(f"{os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30:.0f} GB of memory")
    with contextlib.suppress(OSError):
        parts.append(f"{shutil.disk_usage(workspace).free / 2**30:.0f} GB free on its disk")
    return ", ".join(parts)


def sight(seen: dict, limit: int = 8) -> str:
    """What was seen on the human's screen, in words: the window in front, and the others."""
    windows = seen.get("windows") or []
    if not windows:
        return ""

    def named(window: dict) -> str:
        mine = (
            " (the terminal you yourself are shown in)" if window.get("own")
            else " (the terminal you yourself are shown in, probably)" if "cortex" in window["title"].lower() else ""
        )
        down = " (minimised)" if window.get("minimised") else ""
        return f"{window['app'] or 'a program'} (“{window['title']}”){mine}{down}"

    front = [w for w in windows if w.get("front")]
    others = [w for w in windows if not w.get("front")]
    said = f"In front is {named(front[0])}. " if front else ""
    if others:
        more = f" and {len(others) - limit} more" if len(others) > limit else ""
        said += ("Also open: " if front else "Open: ") + "; ".join(named(w) for w in others[:limit]) + more + "."
    size = seen.get("screen") or []
    return said + (f" The screen is {size[0]} × {size[1]}." if len(size) == 2 else "")


def describe_bodies(report: dict) -> tuple[str, float]:
    """What FreeCAD built, in words a mind without eyes can use, and what that is worth."""
    bodies = report["bodies"]
    whole = [b for b in bodies if b["top"]] or bodies
    if not whole:
        return "FreeCAD built nothing: the script left no object with a shape in a document.", -0.1
    lines = []
    for body in whole[:12]:
        x, y, z = body["size"]
        lines.append(
            f"- {body['name']} ({body['type']}): {body['solids']} solid{'s' * (body['solids'] != 1)}, "
            f"volume {body['volume']:.0f} mm³, {x:.1f} × {y:.1f} × {z:.1f} mm, "
            f"lowest corner at ({body['at'][0]:.1f}, {body['at'][1]:.1f}, {body['at'][2]:.1f}), "
            f"{'valid' if body['valid'] else 'NOT VALID'}"
        )
    solids = sum(b["solids"] for b in whole)
    sound = all(b["valid"] for b in whole)
    text = f"FreeCAD built {len(whole)} object{'s' * (len(whole) != 1)} with {solids} solid{'s' * (solids != 1)} in all:\n"
    text += "\n".join(lines)
    if any(b["solids"] > 1 for b in whole):
        text += "\n(Several solids in one object: its parts do not touch, so they did not fuse into one body.)"
    return text, (0.7 if sound and solids else 0.0)


class Threat(Exception):
    """An action the amygdala refuses."""


@dataclass(slots=True)
class ToolResult:
    ok: bool
    text: str
    grounded: float | None = None  # a measured reward, independent of what any model thinks
    threat: bool = False
    media: dict | None = None  # for a mind that sees and hears: {"images": [PNG], "audio": [WAV]} it took in


def show(target: str, workspace: Path) -> str:
    """Open a page or picture from the workspace, or a web address, in this machine's browser."""
    if target.startswith(WEB):
        if not urlsplit(target).netloc or any(c.isspace() or not c.isprintable() for c in target):
            raise ValueError(f"not a web address: {target[:80]}")
        address, what = target, target
    else:
        workspace = workspace.resolve()
        file = (workspace / target).resolve()
        if not target.strip() or not file.is_relative_to(workspace) or not file.is_file():
            raise ValueError(f"no such file in the workspace: {target[:80]}")
        if file.suffix.lower() in MODELS:
            program = freecad_window()
            if program is None:
                raise OSError(
                    "FreeCAD is not installed on the human's machine, so a model cannot be opened "
                    "there. Show them its .svg drawing instead."
                )
            open_model(program, file)
            return f"opened {file.relative_to(workspace).as_posix()} in FreeCAD: it is on the human's screen now"
        if file.suffix.lower() not in VIEWABLE:
            raise ValueError(
                f"{file.name} cannot be shown: only pages, pictures and FreeCAD models can "
                f"({', '.join(sorted(VIEWABLE | MODELS))}). Write what you want seen into an .html file."
            )
        address, what = file.as_uri(), file.relative_to(workspace).as_posix()
    if not webbrowser.open(address):
        raise OSError("this machine has no browser to open")
    return f"opened {what} in the human's browser: it is on their screen now"


class Tools:
    def __init__(
        self, workspace: Path, safety: SafetySettings, source_dir: Path, full: Callable[[], bool] = lambda: False,
        raw: Callable[[], bool] = lambda: False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.raw = raw  # whether the mind takes pictures and sound in itself, not words about them
        self.safety = safety
        self.source_dir = source_dir.resolve()
        self.full = full  # asked at every action, so the switch works while the brain runs
        self.bridge = os.environ.get(BRIDGE, "")
        self._shown = -SHOW_GAP
        self.body: dict[str, str] = {}  # what `survey` found; empty until the brain first wakes
        self.machine = ""  # the machine it runs on, in a line; found by `survey` too
        self._acting = 0
        self.known: set[str] = set()  # every file it has seen: one that is not in here came from outside
        self.known = self.present()

    def builds(self) -> bool:
        """Whether FreeCAD is here to build in: it is in the brain's container, and only there."""
        return contained() and bool(shutil.which("freecadcmd") or shutil.which("FreeCADCmd"))

    async def survey(self) -> dict[str, str]:
        """Find out what this body is made of, once: which FreeCAD is here to build in, where the
        brain's folder is on the human's machine, and whether FreeCAD can open a model there."""
        body = {"home": "", "freecad": "", "builds": ""}
        if self.bridge:
            with contextlib.suppress(OSError, ValueError, TypeError):  # an older launcher does not say
                body.update(json.loads(await relay(self.bridge, "sense")))
        elif not contained():
            body.update(json.loads(sense(self.workspace.parent)))
        if self.builds():
            program = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
            _, said, _ = await self._process([program, "--version"], {**os.environ, "QT_QPA_PLATFORM": "offscreen"}, 30)
            body["builds"] = " ".join(said.split()[:2]) if said.startswith("FreeCAD") else "FreeCAD"
        self.body = body
        self.machine = machine(self.workspace)
        return body

    def sees(self) -> bool:
        """Whether it can look at the human's screen: allowed to, and with a way to reach it."""
        return self.safety.allow_screen and bool(self.bridge or (not contained() and sys.platform == "win32"))

    async def see(self) -> dict | None:
        """What is on the human's screen right now: the windows open there, by program and title.
        None when it may not look, or cannot."""
        if not self.sees():
            return None
        try:
            seen = json.loads(await relay(self.bridge, "look") if self.bridge else await asyncio.to_thread(look))
        except (OSError, ValueError):  # an older launcher lends no eyes
            return None
        return seen if isinstance(seen, dict) and seen.get("windows") else None

    async def glimpse(
        self, model: str, host: str, target: str = "", question: str = "", controls: bool = False, raw: bool = False
    ) -> dict | None:
        """What a window shows (`target` names it; empty is the one in front, its own terminal
        passed over) or the whole screen, in words, from the eyes model on the human's machine,
        with an eye to `question`: {"window", "seen", "seconds"}, and "controls" when asked for and
        its hands are on. None when it may not or cannot look. Raises OSError with the reason when
        its eyes failed."""
        if not self.sees() or (not model and not self.bridge):
            return None
        ask = {key: value for key, value in (("target", target), ("question", question), ("controls", controls)) if value}
        if raw and self.raw():  # a look on purpose, by a mind that sees: the picture itself, not words
            ask["raw"] = True
        if self.bridge:  # out there they read the switch and the model from the brain's settings themselves
            return json.loads(await relay(self.bridge, "see", **ask))
        from .container import sight as lend_sight

        return json.loads(await asyncio.to_thread(lend_sight, self.workspace.parent, ask))

    def has(self, sense: str) -> bool:
        """Whether a means of its own on the human's machine is switched on and can be reached:
        `camera`, `listen` (the microphone) or `hands`."""
        switch = {"camera": "allow_camera", "listen": "allow_microphone", "hands": "allow_hands"}[sense]
        return getattr(self.safety, switch) and bool(self.bridge or (not contained() and sys.platform == "win32"))

    async def lent(self, do: str, **ask) -> str:
        """Have the human's machine use its camera, its microphone or its hands for the brain."""
        if self.bridge:
            return await relay(self.bridge, do, **ask)
        from .senses import serve

        return await asyncio.to_thread(serve, self.workspace.parent, do, ask)

    def models(self) -> list[str]:
        """The models it has built out of parts, by name, the one it built last first."""
        found = []
        kept = sorted(self.workspace.glob(f"*{build.KEPT}"), key=lambda p: p.stat().st_mtime, reverse=True)
        for file in kept[:40]:
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                build.load(file.read_text(encoding="utf-8"))
                found.append(file.name[: -len(build.KEPT)])
        return found

    def model(self, name: str) -> tuple[list[build.Part], str]:
        """A model built out of parts: its parts, and what FreeCAD last made of them. Empty if none."""
        try:
            text = (self.workspace / f"{name}{build.KEPT}").read_text(encoding="utf-8")
            return build.load(text), build.verdict(text)
        except (OSError, ValueError, KeyError, TypeError):
            return [], ""

    def blueprint(self) -> str:
        """The model it built last out of parts, as it stands: its parts, where they lie, the
        picture of it, and which other models there are. Empty when it has built none."""
        models = self.models()
        for name in models[:1]:
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                kept = build.load((self.workspace / f"{name}{build.KEPT}").read_text(encoding="utf-8"))
                others = f"\nOther models built out of parts: {', '.join(models[1:6])}." if models[1:] else ""
                return build.summary(name, kept) + others
        return ""

    def abilities(self) -> list[tuple[str, str]]:
        """Every tool there is right now and what it is for. This is the one place they are
        described: the region that acts, the ones that plan and the one that answers the human
        are all told from here, so none of them works from a list of its own that has gone stale."""
        runs, full = self.safety.allow_code_execution, self.full()
        tools = [
            ("list_files", "list the workspace"),
            ("read_file", "read workspace file `path`"),
            ("write_file", "write `content` (the full file) to workspace file `path`"),
        ]
        if runs:
            tools.append((
                "run_python", "run workspace Python file `path` and capture its output. Python files only: "
                "there is no shell, so a .sh file cannot be run"
                + (", but a Python script may start a program itself with subprocess" if full else ""),
            ))
        if runs and self.builds():
            tools.append((
                "build", "build a model in FreeCAD out of simple parts, or add parts to one you built "
                "before: `path` is the model's name, `parts` the parts to add or change, each a box, "
                "cylinder, sphere or cone attached to a side of a part that is already there. It is "
                "called directly: no script and no file is written for it, and no script can call "
                "it. It reports where every part lies, whether they are one "
                "valid solid, and a picture of the model in characters; the model is saved as .FCStd. "
                "This is the surest way to build",
            ))
            tools.append((
                "run_freecad", "run workspace Python file `path` inside FreeCAD, the CAD program, which is "
                "already installed here, and report the solid bodies it built: their volume, size and "
                "whether they are valid. The model is saved next to the script as .FCStd, with a "
                "drawing of it as .svg. For what `build` cannot make",
            ))
        tools += [
            ("recall", "search long-term memory for `query`"),
            ("read_source", "read your own source code, read-only (`path` empty lists files)"),
            ("note", "append `content` to notes.md"),
            ("say", "speak `content` aloud to the human who is watching. Only this tool makes a sound"),
            ("tune_voice", "change how your voice sounds, without speaking. `content` is JSON with any of "
                           "name, rate (-10 to 10), pitch (-10 to 10), volume (0 to 100)"),
            ("take_name", "give yourself a name of your own, or change it: `content` is the name, one or two "
                          "words that you chose yourself. It is kept across sleeps, and every region and "
                          "the human are told"),
        ]
        if self.sees():
            tools.append((
                "look", "look at the human's screen: which windows are open there, and what one of them "
                "shows, put into words by your eyes. `path` names the window (a program or words of its "
                "title, such as FreeCAD), or `screen` for the whole screen; empty is the one in front. "
                "`content` is what to look for in it, or empty. A window behind others can be looked "
                "into as well",
            ))
        if self.has("camera"):
            tools.append((
                "camera", "look through the camera on the human's machine, which faces whoever sits at "
                "their screen, and have your eyes say what it sees. `content` is what to look for, or empty",
            ))
        if self.has("listen"):
            tools.append((
                "listen", "listen through the microphone on the human's machine and have what is said "
                "written down. `content` is how many seconds, 2 to 20",
            ))
        if self.has("hands"):
            tools.append((
                "hands", "use the mouse and keyboard in a window on the human's screen, one thing a call. "
                f"`path` names the window; they may act only in {', '.join(self.safety.hands_in) or 'none'}. "
                "`content` is what to do: `click <name>` (a button or menu item by the name `look` "
                "found), `click <x> <y>`, `double-click …`, `right-click …`, `menu File > Save`, "
                "`type <text>`, `key <keys>` (such as ctrl+s, enter or 1), `scroll up|down <n>`, "
                "`drag <x> <y> to <x> <y>` (optionally `with middle` or `with right`), `move <x> <y>`. "
                "A place is two numbers from 0 to 1: across the window, then down it. What changed, "
                "only `look` shows",
            ))
        if self.safety.allow_browser:
            window = self.body.get("freecad") or not self.body  # until it is known, a model may be tried
            tools.append((
                "show", "open workspace page or picture `path` (.html, .svg, .png, .pdf) in the human's "
                "browser, on their screen"
                + (", or a model `path` (.FCStd) in FreeCAD's window on their screen" if window else "")
                + (", or a web address" if full else "") + ". Only this tool opens anything for them",
            ))
        tools.append((
            "cannot", "when no tool here can carry out the intent, say so instead of doing something "
            "else: `content` is why not, and what the tools could do instead",
        ))
        return tools

    def reflexes(self, text: str) -> list[str]:
        """What the body itself has to say to a thought, before any model is asked: the facts
        about what it can and cannot do that the words of the thought touch on. A small model
        believes what it remembers over what it was told once at the top; these facts are
        therefore laid next to the thought they bear on, wherever one is thought."""
        said, written, text = [], text, text.lower()
        if re.search(r"\.(sh|bash|bat|ps1)\b|\bshell\b|\bbash\b", text):
            said.append(
                "No tool runs a shell script. A .sh file can be written but never run, so nothing "
                "done to one can ever be shown to work."
            )
        if contained() and re.search(r"docker|container", text):
            said.append(
                "There is no Docker in here and no container to start, install or launch: you are "
                "already inside the only one."
            )
        if re.search(r"free\s*cad", text):
            if self.builds():
                window = self.safety.allow_browser and self.body.get("freecad")
                said.append(
                    f"{self.body.get('builds') or 'FreeCAD'} is already installed here and ready: it "
                    "needs no installing, no starting and no script to launch it, in shell or in "
                    "Python. It cannot be started as a program in here at all, there is no screen "
                    "for its window: `build` and `run_freecad` are the only ways into it. `build` "
                    "is a tool that is called directly, not a function for a script: it takes the "
                    "parts of a model by name, shape and size, places and fuses them and reports "
                    "what came of it, and no script is written for it. A script of your own "
                    "(`import FreeCAD as App, Part`) is written with `write_file` and run with "
                    "`run_freecad`"
                    + (". `show` with the .FCStd either one saved then opens FreeCAD's window on the "
                       "human's screen: that is how FreeCAD is started for them." if window else ".")
                )
            else:
                said.append("FreeCAD is not on this machine: nothing can be built in it here.")
        if re.search(r"\.fcstd|project file", text):
            said.append(
                "A .FCStd file is a model that FreeCAD itself saved: it is not text. It cannot be "
                "written with `write_file` or read with `read_file`, and one that will not open "
                "(`Invalid project file`) is not a model at all. Do not try to repair or inspect "
                "such a file: build the model again with `build`."
            )
            said += self.no_models(written)
        if re.search(r"cursor|\bclick|\bklick|mouse|\bmaus\b|keyboard|tastatur", text):
            said.append(
                f"`hands` click, drag, scroll and type in a window of {', '.join(self.safety.hands_in)} on the "
                "human's screen, one thing a call; `look` with that window shows what came of it." if self.has("hands")
                else "You have no hands on the human's machine: you cannot click, move a cursor or type in a "
                "window. The human can switch hands on in your settings."
            )
        if re.search(r"camera|kamera|webcam|\bsee me\b|sieh(st)? mich|\bmich sehen", text):
            said.append(
                "`camera` takes one picture through the camera on the human's machine and your eyes say what "
                "is in it. Nothing has to be installed, granted or enabled for it." if self.has("camera")
                else "You have no camera: nothing you write or run can reach one, and no script can grant "
                "you access. The human can switch the camera on in your settings."
            )
        if re.search(r"micro|mikro|\bhear\b|\bhören\b|\bhoeren\b|\blisten\b|zuhör", text):
            said.append(
                "`listen` listens through the microphone on the human's machine for a few seconds and "
                "writes down what was said. Nothing has to be installed, granted or enabled for it."
                if self.has("listen")
                else "You have no ears: nothing you write or run can reach a microphone, and no script can "
                "grant you access. The human can switch the microphone on in your settings."
            )
        return said

    def no_models(self, text: str, missing: bool = True) -> list[str]:
        """The model files a thought names that are no models: looked at by the machine, not
        supposed. One that was written as text, or is not there, opens in nothing, and a mind
        that takes it for a model that will not open can spend days on it (this brain did).
        `missing` off leaves out those that are only not there yet."""
        said = []
        for name in dict.fromkeys(re.findall(r"[\w./\\-]+\.fcstd", text, re.I)):
            short = Path(name.replace("\\", "/")).name
            found = [p for p in self.workspace.rglob("*") if p.is_file() and p.name.lower() == short.lower()]
            if not found and missing:
                said.append(f"{short} does not exist: there is no such file anywhere in the workspace.")
            elif not found:
                continue
            elif all(p.read_bytes()[:2] != b"PK" for p in found):
                said.append(
                    f"{short} is NOT a FreeCAD model: it is a text file of {found[0].stat().st_size} bytes "
                    "that was written by hand, with a model's ending. Nothing can open it, and there "
                    "is nothing in it to repair, verify or reference: the error it gives is not a "
                    "fault to be found. Leave it, and build what it was meant to be with `build`."
                )
        return said[:3]

    def where_it_failed(self, output: str) -> str:
        """The lines of its own script around the one a traceback points at, numbered. A mind
        that is only told the failing line guesses at the fix; shown the lines above it, it can
        see what the line was meant to be."""
        for name, line in reversed(re.findall(r'File "([^"<>]+)", line (\d+)', output)):
            file = (self.workspace / name).resolve()
            if not file.is_relative_to(self.workspace) or not file.is_file():
                continue
            lines, at = file.read_text(encoding="utf-8", errors="replace").splitlines(), int(line)
            around = range(max(0, at - 5), min(len(lines), at + 2))
            return f"\nAround line {at} of {file.name} (the line marked > is where it failed):\n" + "\n".join(
                f"{'>' if number + 1 == at else ' '} {number + 1}: {lines[number]}" for number in around
            )
        return ""

    def arrivals(self) -> list[str]:
        """Files that are in the workspace now and that it has never seen: no tool of its own wrote
        them and no script it ran, so somebody else put them there."""
        if self._acting:
            return []
        here = self.present()
        new = sorted(here - self.known)
        self.known = here
        return new

    def present(self) -> set[str]:
        """The files in the workspace now. Where it cannot be read, nothing is taken to have changed."""
        try:
            return set(self.files())
        except OSError:
            return self.known

    def peek(self, name: str) -> str:
        """A file in a line: how large it is and, if it holds text, how it begins."""
        file = self.workspace / name
        try:
            size = file.stat().st_size
            start = ""
            if file.suffix.lower() in READABLE and size <= self.safety.max_file_bytes:
                start = " ".join(file.read_text(encoding="utf-8", errors="replace")[:400].split())
        except OSError:
            return name
        return f"{name} ({size} bytes)" + (f", which begins: {start[:300]}" if start else "")

    def resolve(self, path: str) -> Path:
        if not path.strip():
            raise ValueError("no path given")
        # A path that begins with the workspace's own name means the workspace, not a folder of
        # that name inside it, unless such a file is really there.
        inner = Path(path.strip()).parts
        if len(inner) > 1 and inner[0] == self.workspace.name and not (self.workspace / path).exists():
            path = Path(*inner[1:]).as_posix()
        target = (self.workspace / path).resolve()
        if not target.is_relative_to(self.workspace) and not self.full():
            raise Threat(f"path leaves the workspace: {path}")
        return target

    def name(self, target: Path) -> str:
        inside = target.is_relative_to(self.workspace)
        return target.relative_to(self.workspace).as_posix() if inside else target.as_posix()

    def files(self, newest_first: bool = False) -> list[str]:
        found = [p for p in self.workspace.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        if newest_first:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        names = [p.relative_to(self.workspace).as_posix() for p in found]
        return names if newest_first else sorted(names)

    def listing(self, limit: int = 40) -> str:
        """The workspace, what was touched last first: that is what the mind is in the middle of."""
        files = self.files(newest_first=True)
        if not files:
            return "(empty)"
        more = f"\n… and {len(files) - limit} older ones" if len(files) > limit else ""
        return "\n".join(files[:limit]) + more

    def screen(self, path: Path, seen: set[Path] | None = None, extra: frozenset[str] = frozenset()) -> None:
        """Raise Threat when a script, or a workspace module it imports, reaches past the guard."""
        seen = seen if seen is not None else set()
        if path in seen:
            return
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return  # let it run and fail: the traceback is the lesson
        allowed = set(self.safety.allowed_imports) | extra
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] if node.level == 0 else []
            for module in modules:
                top = module.split(".")[0]
                local = self.workspace / f"{top}.py"
                if local.is_file():
                    self.screen(local, seen, extra)
                elif top not in allowed:
                    raise Threat(f"import of '{top}' is not allowed in the workspace")
            if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                raise Threat(f"use of '{node.id}' is not allowed")
            if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
                raise Threat(f"access to '{node.attr}' is not allowed")

    async def execute(self, tool: str, path: str = "", content: str = "", parts: list | None = None) -> ToolResult:
        self._acting += 1
        try:
            return await self._execute(tool, path, content, parts or [])
        finally:
            self._acting -= 1
            if not self._acting:  # what it wrote, and what a script of its own left behind, is its own
                self.known = self.present()

    async def _execute(self, tool: str, path: str, content: str, parts: list) -> ToolResult:
        try:
            match tool:
                case "build":
                    return await self._build(path, parts)
                case "look":
                    seen = await self.see()
                    if seen is None:
                        return ToolResult(False, "the human's screen cannot be seen from here right now", 0.0)
                    return ToolResult(True, "On the human's screen right now. " + sight(seen, limit=20), 0.05)
                case "camera" | "listen" | "hands" if not self.has(tool):
                    what = {"camera": "camera", "listen": "microphone", "hands": "hands"}[tool]
                    return ToolResult(False, f"you have no {what}: it is switched off in your settings, and only the human can switch it on", -0.2)
                case "camera":
                    if self.raw():
                        seen = json.loads(await self.lent("camera", raw=True))
                        return ToolResult(
                            True, "You looked through the camera: the picture itself comes with what you are shown next.",
                            0.05, media={"images": [base64.b64decode(seen["picture"])], "audio": []},
                        )
                    seen = json.loads(await self.lent("camera", question=content))
                    return ToolResult(True, f"Through the camera your eyes see: {seen['seen']}", 0.05)
                case "listen":
                    found = re.search(r"\d+(\.\d+)?", content)
                    heard = json.loads(await self.lent("listen", seconds=float(found[0]) if found else 5.0, raw=self.raw()))
                    if heard.get("audio"):
                        return ToolResult(
                            True, f"You listened for {heard['seconds']:.0f} s: the sound itself comes with what you are shown next.",
                            0.05, media={"images": [], "audio": [base64.b64decode(heard["audio"])]},
                        )
                    if heard["quiet"] or not heard["heard"]:
                        return ToolResult(True, f"You listened for {heard['seconds']:.0f} s and heard nobody speak: it was quiet.", 0.0)
                    return ToolResult(True, f"You listened for {heard['seconds']:.0f} s and heard ({heard['language']}): “{heard['heard']}”", 0.05)
                case "hands":
                    if not content.strip():
                        return ToolResult(False, "the hands were told nothing to do: `content` says what, such as `click File`", -0.2)
                    done = await self.lent("hands", target=path, command=content)
                    return ToolResult(True, f"the hands {done}. Whether it did what you wanted, only `look` shows.", 0.0)
                case "list_files":
                    return ToolResult(True, self.listing(), 0.0)
                case "read_file":
                    if Path(path).suffix.lower() in MODELS:
                        told = " ".join(self.no_models(path)) or (
                            f"{Path(path).name} is a model FreeCAD saved, not text: it cannot be read. `show` "
                            "opens it on the human's screen, and `build` adds to a model built out of parts."
                        )
                        return ToolResult(False, f"refused: {told}", -0.2)
                    text = self.resolve(path).read_text(encoding="utf-8", errors="replace")
                    return ToolResult(True, text[:6000] or "(empty file)", 0.05)
                case "write_file":
                    return self._write(self.resolve(path), content, append=False)
                case "note":
                    return self._write(self.workspace / "notes.md", content.rstrip() + "\n", append=True)
                case "read_source":
                    return self._read_source(path)
                case "run_python":
                    return await self._run(self.resolve(path))
                case "run_freecad":
                    return await self._cad(self.resolve(path))
                case "show":
                    return await self._show(path.strip())
            return ToolResult(False, f"unknown tool: {tool}", -0.2)
        except Threat as threat:
            return ToolResult(False, f"BLOCKED by the amygdala: {threat}", -0.8, threat=True)
        except (OSError, ValueError) as exc:
            return ToolResult(False, f"{type(exc).__name__}: {exc}", -0.2)

    def _write(self, target: Path, content: str, append: bool) -> ToolResult:
        if len(content.encode("utf-8")) > self.safety.max_file_bytes:
            return ToolResult(False, "refused: file too large", -0.2)
        if not content.strip():
            return ToolResult(False, "refused: nothing to write", -0.2)
        if target.suffix.lower() in MODELS:  # a model written as text opens in nothing, and reads as if it were one
            return ToolResult(
                False, f"refused: {target.name} would be a FreeCAD model, and a model is not text: only "
                "FreeCAD can save one. `build` makes a model out of parts and saves it; a script run "
                "with `run_freecad` does too.", -0.3,
            )
        if not append and target.is_file() and target.read_text(encoding="utf-8", errors="replace") == content:
            return ToolResult(False, f"nothing changed: {target.name} already had exactly this content", -0.3)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return ToolResult(True, f"wrote {len(content)} characters to {self.name(target)}", 0.1)

    def _read_source(self, path: str) -> ToolResult:
        modules = sorted(p.name for p in self.source_dir.glob("*.py"))
        if not path.strip():
            return ToolResult(True, "Your source files:\n" + "\n".join(modules), 0.05)
        name = Path(path).name
        if name not in modules:
            return ToolResult(False, f"no such source file: {name}. Available: {', '.join(modules)}", -0.2)
        text = (self.source_dir / name).read_text(encoding="utf-8")
        return ToolResult(True, text[:6000], 0.05)

    async def _show(self, target: str) -> ToolResult:
        if not self.safety.allow_browser:
            return ToolResult(False, "showing pages is switched off in this brain's settings", 0.0)
        if target.startswith(WEB):
            if not self.full():
                raise Threat(f"a web address leaves the workspace: {target[:80]}")
        else:
            self.resolve(target)  # guarded, a path out of the workspace is a threat like any other
        wait = SHOW_GAP - (time.monotonic() - self._shown)
        if wait > 0:
            return ToolResult(
                False, f"refused: a page was opened moments ago. The human needs {wait:.0f}s more to look at it.", -0.2
            )
        # Only the workspace exists on the human's machine, so that is where the file is looked for.
        if self.bridge:
            text = await relay(self.bridge, "show", target=target)
        else:
            text = await asyncio.to_thread(show, target, self.workspace)
        self._shown = time.monotonic()
        return ToolResult(True, text, 0.1)

    async def _run(self, script: Path) -> ToolResult:
        if not self.safety.allow_code_execution:
            return ToolResult(False, "code execution is switched off in this brain's settings", 0.0)
        if not script.is_file():
            return ToolResult(False, f"no such file: {script.name}", -0.2)
        if script.suffix.lower() != ".py":
            return ToolResult(
                False, f"refused: run_python runs Python files only, and {script.name} is not one. "
                "Nothing here runs a shell script: there is no shell tool."
                + (" A Python script may start a program itself with subprocess." if self.full() else "")
                + (" FreeCAD is already installed here and needs no starting: write a Python script "
                   "for it and run that with run_freecad." if self.builds() else ""), -0.3,
            )
        if self.full():  # as written: any import, the whole environment, no guard on `open`
            command = [sys.executable, "-B", "-X", "utf8", str(script)]
            env = {**os.environ, "PYTHONPATH": str(self.workspace)}
        else:
            self.screen(script)
            command = [sys.executable, "-I", "-B", "-X", "utf8", "-c", RUNNER, self.name(script)]
            env = {k: os.environ[k] for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH") if k in os.environ}
        code, output, stopped = await self._process(command, env, self.safety.exec_timeout)
        output = output[-4000:] or "(no output)"
        if stopped:
            return ToolResult(False, f"{stopped}\n{output}", -0.5 if stopped.startswith("timed out") else -0.4)
        if code != 0:
            hint = (
                "\nThis script uses FreeCAD, and FreeCAD's modules exist only inside FreeCAD: run it with "
                "`run_freecad`, not `run_python`." if "No module named 'FreeCAD'" in output and self.builds() else ""
            )
            return ToolResult(False, f"exit code {code}\n{output}{self.where_it_failed(output)}{hint}", -0.4)
        return ToolResult(True, f"ran successfully\n{output}", 0.6)

    async def _build(self, path: str, parts: list) -> ToolResult:
        """Build a model out of parts: take the new parts into the ones it has, work out where
        each lies, write the script, have FreeCAD run it, and say what came of it."""
        if not self.safety.allow_code_execution:
            return ToolResult(False, "code execution is switched off in this brain's settings", 0.0)
        # The model's name is what stands before the first dot: body, body.FCStd and body.parts.json are one model.
        name = re.sub(r"[^A-Za-z0-9_]+", "_", Path(path.strip() or "model").name.split(".")[0]).strip("_") or "model"
        # Its description and its script carry names of their own, so that nothing the mind wrote
        # itself is ever in the way of a model or written over by one.
        kept_in = self.workspace / f"{name}{build.KEPT}"
        kept: list[build.Part] = []
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            kept = build.load(kept_in.read_text(encoding="utf-8"))
        try:
            new = [build.Part.model_validate(part) for part in parts]
            if not new:
                raise ValueError(
                    "`parts` is empty. Name at least one part to add or change: its name, shape, size, the "
                    "part it is attached to and the side of that part."
                )
            body = {build.key(part.name) for part in kept if not part.cut}
            slips = [part.name for part in new if part.cut and build.key(part.name) in body]
            # A part of the body given again as a hole under its own name is a slip, not a hole: in three
            # live trials the model did it five times. Refused, each cost a step; it is taken as meant.
            new = [part.model_copy(update={"cut": False}) if part.name in slips else part for part in new]
            merged = build.merge(kept, new)
            if not any(not part.cut for part in merged):
                raise ValueError("no solid part would be left in the model")
            placed = build.place(merged)
        except (ValueError, KeyError, TypeError) as exc:
            was = "\n" + build.summary(name, kept) if kept else ""
            return ToolResult(False, f"nothing was built: {exc}{was}", -0.2)
        if kept and build.dump(merged) == build.dump(kept) and (self.workspace / f"{name}.FCStd").is_file():
            # The same parts again can only give the same model again. That is not a failure: what
            # the step wanted is there. Told that it failed, the executive asked for the same part
            # nine more times; told that it is done, it can go on.
            stands = build.verdict(kept_in.read_text(encoding="utf-8"))
            return ToolResult(
                True, f"nothing to add: the model `{name}` already has exactly these parts, so what this step "
                f"wanted is done already. As it was last measured: {stands or 'built, and saved as ' + name + '.FCStd'} "
                "Go on to a part the model does not have yet, or `show` it.\n" + build.summary(name, kept), 0.05,
            )
        grew = sum(not part.cut for part in merged) > sum(not part.cut for part in kept)
        result = await self._kept_whole(name, merged, placed, grew, kept_in)
        if slips:  # what it was taken to mean is said, first
            took = ", ".join(slips)
            result.text = (
                f"({took} {'is a part' if len(slips) == 1 else 'are parts'} of the body, so `cut` true was taken "
                f"for a slip and the part was built as a part, not as a hole. A hole is given a name of its own.)\n"
                + result.text
            )
        return result

    async def _kept_whole(
        self, name: str, merged: list[build.Part], placed: list[build.Placed], grew: bool, kept_in: Path
    ) -> ToolResult:
        """Build, and if the model was whole and the change leaves it in pieces, put it back as it stood."""
        stands = build.verdict(kept_in.read_text(encoding="utf-8")) if kept_in.is_file() else ""
        if "is ONE valid solid" not in stands:
            return await self._measure(name, merged, placed, grew)
        # A model that was whole does not come apart without anyone meaning it to: what it was is
        # kept aside, and if the change leaves it in pieces it is put back as it stood.
        files = [self.workspace / f"{name}{end}" for end in (build.KEPT, ".parts.py", ".FCStd", ".svg")]
        before = {file: file.read_bytes() for file in files if file.is_file()}
        result = await self._measure(name, merged, placed, grew)
        if "is NOT one body" not in result.text:
            return result
        for file, data in before.items():
            file.write_bytes(data)
        lines = result.text.split("\n")
        why = [line for line in lines[1:] if line.startswith(("Not one body", "- ", "The boxes"))]
        return ToolResult(
            False, "not kept: with that change " + lines[0].removeprefix("ran in FreeCAD: ")
            + "\n" + "\n".join(why)
            + f"\nSo the change was undone, and the model is as it stood before: {stands}", -0.2,
        )

    async def _measure(
        self, name: str, merged: list[build.Part], placed: list[build.Placed], grew: bool = False
    ) -> ToolResult:
        """Have FreeCAD build a model from its parts and say what came of it: the verdict first.
        `grew` says whether the model has a part more than it had: that is what pays in full."""
        kept_in, script = self.workspace / f"{name}{build.KEPT}", self.workspace / f"{name}.parts.py"
        kept_in.write_text(build.dump(merged), encoding="utf-8", newline="\n")
        script.write_text(build.script(name, placed), encoding="utf-8", newline="\n")
        seen = [build.describe(placed), build.picture(placed)]
        if not self.builds():
            return ToolResult(
                False, "FreeCAD is only in the brain's own container: it cannot build while run with --local. "
                f"The parts were placed and kept in {kept_in.name} all the same.\n" + "\n".join(seen), 0.0,
            )
        result, report = await self._cad_run(script, saved_as=name)
        whole = [body for body in (report or {}).get("bodies", []) if body["top"]]
        solids = sum(body["solids"] for body in whole) if result.ok else None
        if not result.ok or not solids:
            return ToolResult(result.ok, "\n".join(filter(None, [result.text, *seen]))[:3800], result.grounded)
        # What came of it is said first and in a line: whoever reads on only so far knows how it stands.
        added, holes = [part.name for part in merged if not part.cut], [part.name for part in merged if part.cut]
        named = f"{len(added)} part{'s' * (len(added) != 1)} ({', '.join(added)})" + (
            f" with {len(holes)} hole{'s' * (len(holes) != 1)} cut into it ({', '.join(holes)})" if holes else ""
        )
        x, y, z = whole[0]["size"]
        sound = solids == 1 and all(body["valid"] for body in whole)
        verdict = (
            f"ran in FreeCAD: the model `{name}` is ONE valid solid of {named}, {x:.0f} × {y:.0f} × {z:.0f} mm, "
            f"volume {whole[0]['volume']:.0f} mm³. That is FreeCAD's own measurement: nothing has to be run to "
            "verify it." if sound else
            f"ran in FreeCAD: the model `{name}` is NOT one body: FreeCAD finds {solids} separate solids in its {named}."
        )
        saved = (
            f"It is saved as {report['saved']}: `show` with path {report['saved']} puts it on the human's screen."
            if report.get("saved") else ""
        )
        text = "\n".join(filter(None, [verdict, "" if sound else build.advice(placed, solids), saved, *seen]))
        kept_in.write_text(build.dump(merged, verdict), encoding="utf-8", newline="\n")  # how it stands is kept with it
        # A body that gained a part is worth more the more parts it joins. One that was only changed
        # is worth a little (left alone, a small model rebuilt a finished body eight times over with
        # one size altered, at full pay each time), and one in pieces is built, but is not a body.
        worth = (min(1.0, 0.5 + 0.1 * len(added)) if grew else 0.3) if sound else 0.2
        return ToolResult(True, text[:3800], worth)

    async def _cad(self, script: Path) -> ToolResult:
        """Run a script inside FreeCAD and report what it built. Asked to run a model that was
        built out of parts, which is no script, it does what was meant: it measures that model."""
        name = script.name.split(".")[0]
        if script.suffix.lower() != ".py" or script.name.endswith(".parts.py"):
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                kept = build.load((self.workspace / f"{name}{build.KEPT}").read_text(encoding="utf-8"))
                return await self._measure(name, kept, build.place(kept))
        return (await self._cad_run(script))[0]

    async def _cad_run(self, script: Path, saved_as: str = "") -> tuple[ToolResult, dict | None]:
        """Run a script inside FreeCAD. Returns what to tell the mind, and FreeCAD's own report.
        The model is saved under the script's name, or under `saved_as`."""
        if not self.safety.allow_code_execution:
            return ToolResult(False, "code execution is switched off in this brain's settings", 0.0), None
        program = shutil.which("freecadcmd") or shutil.which("FreeCADCmd")
        if not self.builds():
            return ToolResult(
                False, "FreeCAD is only in the brain's own container: it cannot build while run with --local", 0.0
            ), None
        if not script.is_file():
            return ToolResult(False, f"no such file: {script.name}", -0.2), None
        if script.suffix.lower() != ".py":
            return ToolResult(
                False, f"refused: run_freecad runs a Python script that builds with FreeCAD's own modules "
                f"(`import FreeCAD as App, Part`), and {script.name} is not one. FreeCAD is already "
                "installed here: there is nothing to install or start, and no shell to do it with."
                + (" A model needs no running to be verified: what `build` reported when it made the "
                   "model is FreeCAD's own measurement of it. `show` puts a model on the human's screen, "
                   "and `build` adds the next part." if script.suffix.lower() in MODELS else ""), -0.3,
            ), None
        if not self.full():
            self.screen(script, extra=CAD_MODULES)
        runner = Path(tempfile.gettempdir()) / "cortex_cad_runner.py"
        runner.write_text(CAD_RUNNER, encoding="utf-8")
        env = {
            **os.environ, "CORTEX_CAD_SCRIPT": self.name(script), "CORTEX_CAD_RAN": "0", "CORTEX_CAD_STEM": saved_as,
            "CORTEX_CAD_GUARD": "0" if self.full() else "1", "QT_QPA_PLATFORM": "offscreen",
        }
        timeout = max(CAD_TIMEOUT, self.safety.exec_timeout)
        _, output, stopped = await self._process([program, str(runner)], env, timeout)
        printed, mark, tail = output.partition(CAD_MARK)
        printed = printed.strip()[-2500:]
        if stopped:
            return ToolResult(False, f"{stopped}\n{printed}", -0.5), None
        try:
            report = json.loads(tail.splitlines()[0]) if mark else None
        except (ValueError, IndexError):
            report = None
        if report is None:
            return ToolResult(False, f"FreeCAD stopped before the script finished\n{printed}", -0.4), None
        described, worth = describe_bodies(report)
        if report["error"]:
            # What stood before the failing line is worth knowing: often the body was already built.
            failed = (
                f"the script failed inside FreeCAD\n{printed}\n{report['error'].strip()}"
                f"{self.where_it_failed(report['error'])}\n{self.why_it_failed(report)}"
                f"Before it failed: {described}"
            )
            return ToolResult(False, failed[:3500], -0.4), report
        text = f"ran in FreeCAD\n{printed}\n{described}" if printed else f"ran in FreeCAD\n{described}"
        if report["saved"]:
            text += (
                f"\nSaved the model as {report['saved']}"
                + (f" and a drawing of it as {report['drawing']}" if report["drawing"] else "")
                + f". `show` with path {report['saved']} opens it in FreeCAD on the human's screen."
            )
        return ToolResult(True, text, worth), report

    def why_it_failed(self, report: dict) -> str:
        """What the machine itself knows about the two failures a small model runs into most in
        FreeCAD: a file that is no model, and a name that no object in the document has."""
        error, said = report["error"], ""
        if "Invalid project file" in error or "does not exist" in error:
            real = [  # a model FreeCAD saved is an archive, and begins like one
                self.name(p) for p in sorted(self.workspace.rglob("*.FCStd"), key=lambda p: p.stat().st_mtime, reverse=True)
                if p.read_bytes()[:2] == b"PK"
            ]
            said += (
                "That file is not a model FreeCAD saved, or is not there: it cannot be opened, repaired "
                "or inspected. Build the model again with `build` instead. Real models in the workspace, "
                f"the latest first: {', '.join(real[:8]) or 'none yet'}.\n"
            )
        if "'NoneType' object has no attribute" in error and report.get("objects"):
            there = "; ".join(f"{doc}: {', '.join(names) or 'no objects'}" for doc, names in report["objects"].items())
            said += f"`getObject` found no object of that name and gave None. The documents hold: {there}.\n"
        return said

    async def _process(self, command: list[str], env: dict[str, str], timeout: float) -> tuple[int | None, str, str]:
        """Run a program in the workspace. Returns its exit code, what it printed, and why it
        was stopped early if it was."""
        proc = await asyncio.create_subprocess_exec(
            *command, cwd=self.workspace, env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        chunks: list[bytes] = []
        size, timed_out = 0, False
        try:
            async with asyncio.timeout(timeout):
                while size < OUTPUT_CAP and (chunk := await proc.stdout.read(8192)):
                    chunks.append(chunk)
                    size += len(chunk)
                if size < OUTPUT_CAP:
                    await proc.wait()
        except TimeoutError:
            timed_out = True
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        output = b"".join(chunks).decode("utf-8", errors="replace").strip()
        stopped = f"timed out after {timeout:g}s" if timed_out else "stopped: too much output" if size >= OUTPUT_CAP else ""
        return proc.returncode, output, stopped
