"""The brain's own voice: it speaks aloud through this machine's speakers.

It has two kinds of voice. A neural one it generates itself: a small network turns the words into
sound, sentence by sentence, in the language that voice was trained on. And the voices installed
in Windows, which the system speaks for it.

Speech happens in the brain's own process, not in the workspace. The code that speaks is fixed;
what the brain wants to say reaches it as data and is never run.

A brain in a container has no speakers. The machine that launched it lends its voice through the
bridge: the brain sends the words and how to say them, and the same fixed code speaks them here.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pydantic import ValidationError

from .bridge import BRIDGE, relay
from .config import VoiceSettings

TIMEOUT = 60.0
MAX_WORDS = 600
SPEAKING = threading.Event()  # set while this machine speaks for a brain
# Voices the brain generates itself. Each is a network of some 60 MB, fetched the first time it is chosen.
NEURAL = {
    "de_DE-thorsten-medium": "German/Deutsch, male",
    "de_DE-thorsten-high": "German/Deutsch, male, finer",
    "de_DE-kerstin-low": "German/Deutsch, female",
    "de_DE-ramona-low": "German/Deutsch, female",
    "de_DE-karlsson-low": "German/Deutsch, male",
    "en_US-lessac-medium": "English, American, female",
    "en_US-ryan-medium": "English, American, male",
    "en_GB-alan-medium": "English, British, male",
    "en_GB-alba-medium": "English, Scottish, female",
}
NETWORKS = Path.home() / ".cortex" / "voices"
SHELL = ("powershell", "-NoProfile", "-NonInteractive", "-Command")
LIST = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
(New-Object -ComObject SAPI.SpVoice).GetVoices() | ForEach-Object { $_.GetDescription() }
"""
# The 8 is SVSFIsXML: it makes the pitch tag count. The words themselves arrive escaped.
SPEAK = r"""
$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
foreach ($known in $voice.GetVoices()) {
    if ($known.GetDescription() -eq $env:CORTEX_VOICE) { $voice.Voice = $known }
}
$voice.Rate = [int]$env:CORTEX_RATE
$voice.Volume = [int]$env:CORTEX_VOLUME
[void]$voice.Speak($env:CORTEX_WORDS, 8)
"""

_networks: dict[str, Any] = {}
_fetching = threading.Lock()


def speakers() -> bool:
    """Whether this machine can play what a neural voice generates."""
    return sys.platform == "win32"


def network(name: str) -> Any:
    """The network behind a neural voice: fetched once, then kept in memory for as long as we run."""
    with _fetching:
        if name not in _networks:
            from piper import PiperVoice
            from piper.download_voices import download_voice

            # It warns about a sound its German voices never learned; a line printed out here
            # would land in the middle of the interface.
            logging.getLogger("piper").setLevel(logging.ERROR)
            model = NETWORKS / f"{name}.onnx"
            settings = NETWORKS / f"{name}.onnx.json"
            if not settings.is_file():  # it arrives last, so a fetch that was cut short starts over
                NETWORKS.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=NETWORKS) as fresh:
                    download_voice(name, Path(fresh))
                    for part in sorted(Path(fresh).iterdir()):
                        part.replace(NETWORKS / part.name)
            _networks[name] = PiperVoice.load(model)
        return _networks[name]


def sentences(words: str, s: VoiceSettings) -> list[bytes]:
    """Generate the speech: one finished sound per sentence, in this voice."""
    from piper import SynthesisConfig

    # Pitch is the same samples played faster or slower, up to five semitones either way; the
    # network then speaks that much slower or faster, so the pace stays where `rate` puts it.
    tone = 2 ** (s.pitch / 24)
    shape = SynthesisConfig(length_scale=tone * 3 ** (-s.rate / 10), volume=s.volume / 100)
    sounds = []
    for chunk in network(s.name).synthesize(words, shape):
        sound = io.BytesIO()
        with wave.open(sound, "wb") as out:
            out.setnchannels(chunk.sample_channels)
            out.setsampwidth(chunk.sample_width)
            out.setframerate(round(chunk.sample_rate * tone))
            out.writeframes(chunk.audio_int16_bytes)
        sounds.append(sound.getvalue())
    return sounds


def utter(words: str, s: VoiceSettings) -> None:
    """Say the words in a neural voice through this machine's speakers."""
    if not speakers():
        raise OSError("this machine has no speakers the brain can reach")
    import winsound

    for sound in sentences(words, s):
        winsound.PlaySound(sound, winsound.SND_MEMORY)


def warm(s: VoiceSettings) -> None:
    """Have a neural voice fetched and loaded before it is first needed, without holding anything up."""

    def ready() -> None:
        with contextlib.suppress(Exception):  # no network, no disk: it will say why when it speaks
            sentences("Hallo.", s)

    if s.enabled and s.name in NEURAL and speakers():
        threading.Thread(target=ready, daemon=True).start()


async def shell(script: str, **env: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *SHELL, script, env={**os.environ, **env},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with asyncio.timeout(TIMEOUT):
            output, _ = await proc.communicate()
    finally:
        if proc.returncode is None:  # timed out, or the brain stopped mid-sentence
            proc.kill()
            await proc.wait()
    text = output.decode("utf-8", errors="replace").strip()
    if proc.returncode:
        raise OSError(text or f"exit code {proc.returncode}")
    return text


async def perform(do: str, words: str, s: VoiceSettings) -> str:
    """List the voices this machine can speak with, or say the words aloud with one of them."""
    if do == "list":
        system: list[str] = []
        with contextlib.suppress(OSError):  # no Windows voices here: the neural ones remain
            system = (await shell(LIST)).splitlines()
        return "\n".join(system + list(NEURAL if speakers() else ()))
    SPEAKING.set()  # its ears wait while it speaks, so as not to hear only itself
    try:
        if s.name in NEURAL:
            await asyncio.to_thread(utter, words[:MAX_WORDS], s)
            return ""
        return await shell(
            SPEAK, CORTEX_WORDS=f'<pitch absmiddle="{s.pitch}"/>{escape(words[:MAX_WORDS])}',
            CORTEX_VOICE=s.name, CORTEX_RATE=str(s.rate), CORTEX_VOLUME=str(s.volume),
        )
    finally:
        SPEAKING.clear()


def lend(do: str) -> Callable[[dict[str, Any]], str]:
    """This machine's voice, for a bridge to offer. Settings are validated and the words escaped
    out here, so a container decides what is said and nothing else."""

    def serve(ask: dict[str, Any]) -> str:
        words = str(ask.pop("words", ""))
        return asyncio.run(perform(do, words, VoiceSettings.model_validate(ask)))

    return serve


def named(voice: str) -> str:
    return f"{voice} ({NEURAL[voice]})" if voice in NEURAL else voice


def closest(wish: str, voices: list[str]) -> str | None:
    """The voice a wish most likely means: by name ("zira", "kerstin") or by what it is ("a
    German one", "deutsche Stimme", "female")."""

    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"[^\W_]+", text.lower()) if len(w) > 3]  # not "the", "der", "des"

    def alike(voice: str) -> int:
        known = words(named(voice))
        return sum(any(w.startswith(k) or k.startswith(w) for k in known) for w in words(wish))

    if wish in voices:
        return wish
    best = max(voices, key=alike, default=None)  # the first of equals, in the order they are offered
    return best if best and alike(best) else None


class Voice:
    def __init__(self, settings: VoiceSettings) -> None:
        self.settings = settings
        self.bridge = os.environ.get(BRIDGE, "")
        self._installed: list[str] | None = None
        self._speaking = asyncio.Lock()  # one sentence at a time: a reply never talks over a `say`

    async def _do(self, do: str, words: str = "") -> str:
        if self.bridge:
            return await relay(self.bridge, do, words=words, **self.settings.model_dump())
        return await perform(do, words, self.settings)

    async def installed(self) -> list[str]:
        """The voices this machine has. Empty when the voice is off or the system offers none."""
        if not self.settings.enabled:
            return []
        if self._installed is None:
            try:
                here = sys.platform == "win32" or self.bridge
                self._installed = (await self._do("list")).splitlines() if here else []
            except OSError:
                self._installed = []
        return self._installed

    async def speak(self, words: str) -> str:
        """Say the words aloud. Returns the reason when they could not be spoken, else nothing."""
        s = self.settings
        if not await self.installed():
            return "the voice is switched off" if not s.enabled else "this machine offers no voice"
        try:
            async with self._speaking:
                await self._do("speak", words)
        except OSError as exc:
            return str(exc)[:200] or type(exc).__name__
        return ""

    def tongue(self) -> str:
        """The language this voice speaks, when that is known: a neural voice has exactly one."""
        return NEURAL.get(self.settings.name, "").split("/")[0].split(",")[0]

    async def describe(self) -> str:
        s = self.settings
        installed = await self.installed()
        if not installed:
            return "You have no voice right now: what you `say` reaches the human as text only."
        return (
            f"The voice you speak with now: {named(s.name) if s.name else 'the default one'}, rate {s.rate}, "
            f"pitch {s.pitch}, volume {s.volume}. Voices you can choose from: "
            f"{'; '.join(map(named, installed))}. These are the names of voices, never your own "
            "name. A voice speaks its own language: German words sound German only in a German voice."
        )

    async def tune(self, wishes: str) -> str:
        """Reshape the voice from a JSON object of wishes. With none, only describe it."""
        if wishes.strip():
            try:
                asked = json.loads(wishes)
                if not isinstance(asked, dict) or not asked.keys() <= {"name", "rate", "pitch", "volume"}:
                    raise ValueError("only name, rate, pitch and volume can be set")
                if "name" in asked:
                    installed = await self.installed()
                    match = closest(str(asked["name"]), installed)
                    if match is None:
                        raise ValueError(
                            f"no voice called {asked['name']!r}; choose from: {'; '.join(map(named, installed))}"
                        )
                    asked["name"] = match
                clean = VoiceSettings.model_validate({**self.settings.model_dump(), **asked})
            except ValidationError as exc:
                error = exc.errors()[0]
                raise ValueError(f"{'.'.join(map(str, error['loc']))}: {error['msg']}") from None
            except json.JSONDecodeError:
                raise ValueError(
                    "`content` must be one JSON object with any of name, rate, pitch, volume"
                ) from None
            for key in asked:
                setattr(self.settings, key, getattr(clean, key))
        return await self.describe()
