"""llama.cpp's server, for a mind that thinks with a model which sees and hears.

Ollama cannot hand a model sound, so when a brain's engine is `llama` the machine that starts it
starts llama.cpp's server beside Ollama, with the model and the part of it that takes pictures and
sound (`mmproj`), and stops it when the brain stops. Every region's call then goes there; Ollama
keeps the embeddings and the inner voice.
"""

from __future__ import annotations

import atexit
import contextlib
import subprocess
import time
from pathlib import Path

import httpx

from .config import ModelSettings

HOME = Path.home() / ".cortex" / "llama"
BINARY = HOME / "bin" / "llama-server.exe"
LOG = HOME / "server.log"
RELEASE = "https://github.com/ggml-org/llama.cpp/releases (llama-bXXXX-bin-win-cuda-13.4-x64.zip and cudart-llama-bin-win-cuda-13.4-x64.zip, unpacked into ~/.cortex/llama/bin)"


def up(models: ModelSettings) -> bool:
    try:
        return httpx.get(f"{models.llama}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


SERVER: subprocess.Popen | None = None  # the server started here, so night school can stop and restart it


def adapter(models: ModelSettings, root: Path | None) -> Path | None:
    """What night school taught this brain, if it kept anything: an adapter in the brain's folder."""
    if not models.llama_lora or root is None:
        return None
    file = root / models.llama_lora
    return file if file.is_file() else None


def start(
    models: ModelSettings, root: Path | None = None, lora: Path | None = None, wait: float = 180.0
) -> subprocess.Popen | None:
    """Start the server for this brain's model unless one answers already, with what night school
    taught it (`lora`, else the adapter its settings name). Raises OSError, saying what is missing,
    when it cannot be started."""
    global SERVER
    if models.engine != "llama" or up(models):
        return None
    lora = lora or adapter(models, root)
    missing = [str(path) for path in (BINARY, Path(models.llama_model), Path(models.llama_mmproj)) if not path.is_file()]
    if missing:
        raise OSError("this brain thinks with llama.cpp, and these are missing: " + ", ".join(missing)
                      + (f". The server comes from {RELEASE}" if str(BINARY) in missing else ""))
    # llama.cpp fits the model into the card memory that is free when it starts: a model Ollama
    # still holds there (Granite stays loaded for half an hour) would push much of it onto the processor.
    with contextlib.suppress(Exception):
        from .school import free_the_card

        free_the_card(models.host)
    HOME.mkdir(parents=True, exist_ok=True)
    log = LOG.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [str(BINARY), "--model", models.llama_model, "--mmproj", models.llama_mmproj,
         "--host", "127.0.0.1", "--port", str(models.llama_port), "--ctx-size", str(models.llama_ctx * max(1, models.parallel)),
         "--parallel", str(max(1, models.parallel)), "--n-gpu-layers", "99", "--jinja", "--no-webui",
         *(["--lora", str(lora)] if lora else [])],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    atexit.register(stop, server)
    until = time.monotonic() + wait
    while time.monotonic() < until:
        if server.poll() is not None:
            raise OSError(f"llama.cpp's server stopped while starting: see {LOG}")
        if up(models):
            SERVER = server
            return server
        time.sleep(1.0)
    stop(server)
    raise OSError(f"llama.cpp's server did not come up within {wait:.0f} s: see {LOG}")


def stop(server: subprocess.Popen | None) -> None:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
