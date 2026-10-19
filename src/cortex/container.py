"""A brain in a container: the same `cortex run`, on a machine of its own.

Only the brain's folder is shared with the container, so that folder is all a brain can change,
whatever it runs and even with its amygdala switched off. The model server stays on this machine,
and so do the voice and the screen: the container reaches all three over the network.
"""

from __future__ import annotations

import base64
import contextlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console

from . import eyes as eyesight
from . import llama, school, senses, voice
from .bridge import BRIDGE, Bridge
from .config import LLAMA_HOST, MODEL_HOST, WORKSPACE_DIR, BrainSettings
from .eyes import see as look_at
from .tools import look, sense, show

IMAGE = "cortex"
BRAIN = "/brain"  # where the brain's folder appears inside
HOST = "host.docker.internal"  # this machine, as a container names it
SOURCE = Path(__file__).resolve().parents[2]  # the checkout this is running from
console = Console()


class Unavailable(Exception):
    """No container can be started here, and why."""


def engine() -> str:
    """The docker command, once it is known to be installed and running."""
    docker = shutil.which("docker")
    if docker is None:
        raise Unavailable("Docker is not installed. Install Docker Desktop to give the brain its container.")
    if subprocess.run([docker, "info"], capture_output=True).returncode:
        raise Unavailable("Docker is installed but not running. Start Docker Desktop and try again.")
    return docker


def build(docker: str) -> None:
    """Bring the image up to date with this source. After the first time it takes a second or two."""
    if not (SOURCE / "Dockerfile").is_file():
        return  # installed without its source: the image that exists is the one to use
    first = subprocess.run([docker, "image", "inspect", IMAGE], capture_output=True).returncode != 0
    if first:
        console.print("Building the brain's container. The first time takes a few minutes.")
    built = subprocess.run(
        [docker, "build", "--tag", IMAGE, str(SOURCE)],
        capture_output=not first, encoding="utf-8", errors="replace",
    )
    if built.returncode:
        raise Unavailable("The container image could not be built." + ("" if first else f"\n{built.stderr[-2000:]}"))


def from_inside(url: str) -> str:
    """A server address as a container must spell it: its own localhost is not this machine."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return parts._replace(netloc=f"{HOST}:{parts.port or 11434}").geturl()
    return url


def local_offset() -> str:
    """This machine's offset from UTC as a TZ value, so times in the stream read as they do here."""
    minutes = int(datetime.now().astimezone().utcoffset().total_seconds() // 60)
    hours, rest = divmod(abs(minutes), 60)
    east = minutes >= 0  # POSIX counts the other way: hours to add to reach UTC
    return f"<{'+' if east else '-'}{hours:02d}{rest:02d}>{'-' if east else '+'}{hours}:{rest:02d}"


def eyes(root: Path) -> str:
    """Lend the brain a look at this screen: the windows open on it, by program and title. The
    switch is read out here each time, from the brain's settings as they are now."""
    try:
        allowed = BrainSettings.load(root).safety.allow_screen
    except Exception:  # caught while the brain was writing them: no look this time, and no noise out here
        raise OSError("the brain's settings could not be read just now") from None
    if not allowed:
        raise OSError("looking at the screen is switched off in this brain's settings")
    return look()


def sight(root: Path, ask: dict | None = None) -> str:
    """Lend the brain a look at what a window shows (`target`; empty is the one in front) or the
    whole screen, with an eye to `question`: the picture is taken and described out here, and only
    the words go in. Switch and model are read from its settings at each look. With its hands
    switched on, what can be clicked by name in that window comes with it."""
    ask = ask or {}
    try:
        settings = BrainSettings.load(root)
    except Exception:
        raise OSError("the brain's settings could not be read just now") from None
    models, safety = settings.models, settings.safety
    if not safety.allow_screen:
        raise OSError("looking at the screen is switched off in this brain's settings")
    target, question = str(ask.get("target", "")), str(ask.get("question", ""))
    if ask.get("raw"):  # a mind that sees itself is given the picture, not words about it
        try:
            window = eyesight.find(target)
            picture = base64.b64encode(eyesight.capture(window)).decode()
        except Exception as exc:
            raise OSError(f"its eyes failed: {str(exc)[:240] or type(exc).__name__}") from None
        seen = {"window": window["title"][:90] if window else "the whole screen", "picture": picture}
        if safety.allow_hands and ask.get("controls"):
            with contextlib.suppress(Exception):
                seen["controls"] = senses.named_controls(target)[:1500]
        return json.dumps(seen)
    if not models.eyes:
        raise OSError("this brain has no eyes model: it sees the names of windows only")
    try:
        seen = look_at(models.eyes, models.host, target, question) if (target or question) else look_at(models.eyes, models.host)
    except Exception as exc:  # no such model, no server, a window that would not be read: said, not raised out here
        raise OSError(f"its eyes failed: {str(exc)[:240] or type(exc).__name__}") from None
    if safety.allow_hands and ask.get("controls"):
        with contextlib.suppress(Exception):
            seen["controls"] = senses.named_controls(target)[:1500]
    return json.dumps(seen)


def command(docker: str, root: Path, args: list[str], env: dict[str, str], tty: bool) -> list[str]:
    return [
        docker, "run", "--rm", "--init", "--interactive", *(["--tty"] if tty else []),
        "--add-host", f"{HOST}:host-gateway",
        "--pids-limit", "512",
        "--volume", f"{root}:{BRAIN}",
        *(part for key, value in env.items() for part in ("--env", f"{key}={value}")),
        IMAGE, "run", BRAIN, *args,
    ]


def run(root: Path, args: list[str]) -> int:
    """Run `cortex run` on this brain inside its container. Returns the exit code."""
    eyesight.remember_own()  # the terminal it is started from is the one it is shown in
    docker = engine()
    build(docker)
    models = BrainSettings.load(root).models
    try:  # a mind that sees and hears thinks with llama.cpp's server, which runs out here
        llama.start(models, root)
    except OSError as exc:
        raise Unavailable(str(exc)) from None
    env = {
        MODEL_HOST: from_inside(models.host),
        LLAMA_HOST: f"http://{HOST}:{models.llama_port}",
        "TZ": local_offset(),
        "COLORTERM": "truecolor",
    }
    lends = {  # what this machine does for the brain; the voice finds out for itself whether it has one
        "list": voice.lend("list"),
        "speak": voice.lend("speak"),
        "show": lambda ask: show(str(ask["target"]), root / WORKSPACE_DIR),
        "sense": lambda ask: sense(root),  # where its folder is out here, and whether FreeCAD is
        "look": lambda ask: eyes(root),  # which windows are open on this screen
        "see": lambda ask: sight(root, ask),  # what a window shows, in words
        "camera": lambda ask: senses.serve(root, "camera", ask),  # what the camera sees, in words
        "listen": lambda ask: senses.serve(root, "listen", ask),  # what the microphone hears, written down
        "hands": lambda ask: senses.serve(root, "hands", ask),  # a click, a key or typing in a window
        "ear": lambda ask: senses.serve(root, "ear", ask),  # the human's talk: pressed to talk, or always listening
        "school": school.lend(root, BrainSettings.load(root).models.host),  # night school, on this machine's card
    }
    env[BRIDGE] = Bridge(lends).address(HOST)
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    try:
        return subprocess.call(command(docker, root, args, env, tty))
    except KeyboardInterrupt:
        return 130
