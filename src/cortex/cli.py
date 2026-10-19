"""Command line: create a brain somewhere, then let it run."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Annotated

import ollama
import typer
from rich.console import Console
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import CONFIG_FILE, WORKSPACE_DIR, BrainSettings, ModelSettings, contained

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A self-awakening, self-improving brain of agents that runs on local models.",
)
console = Console()
REGISTRY = Path.home() / ".cortex" / "instances.json"


def registry() -> dict[str, str]:
    if not REGISTRY.is_file():
        return {}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def register(name: str, root: Path) -> None:
    known = registry()
    known[name] = str(root)
    REGISTRY.parent.mkdir(exist_ok=True)
    REGISTRY.write_text(json.dumps(known, indent=2), encoding="utf-8")


def locate(target: str | None) -> Path:
    """Find a brain by path, by registered name, or by being the only one there is."""
    known = registry()
    if target is None:
        if (Path.cwd() / CONFIG_FILE).is_file():
            return Path.cwd()
        if len(known) == 1:
            return Path(next(iter(known.values())))
        console.print("Which brain? Give a path or a name. `cortex list` shows the ones you have created.")
        raise typer.Exit(1)
    root = Path(known.get(target, target)).expanduser()
    if not (root / CONFIG_FILE).is_file():
        console.print(f"[red]No brain at {root}.[/] Create one with `cortex new`.")
        raise typer.Exit(1)
    return root.resolve()


def installed_models(host: str) -> list[str] | None:
    """Model tags on the Ollama server, or None when it cannot be reached."""
    try:
        return [m.model for m in ollama.Client(host=host).list().models]
    except Exception:
        return None


def has_model(model: str, installed: list[str]) -> bool:
    return model in installed or f"{model}:latest" in installed


def pull(host: str, model: str) -> bool:
    try:
        with Progress(console=console) as progress:
            task = progress.add_task(f"pulling {model}", total=None)
            for update in ollama.Client(host=host).pull(model, stream=True):
                if update.total:
                    progress.update(task, total=update.total, completed=update.completed or 0)
        return True
    except Exception as exc:
        console.print(f"[red]Could not pull {model}:[/] {exc}")
        return False


def preflight(models: ModelSettings, assume_yes: bool = False) -> bool:
    """Check the model server and offer to fetch what is missing."""
    installed = installed_models(models.server)
    if installed is None:
        console.print(
            f"[red]No Ollama server at {models.server}.[/] Start Ollama (or run `ollama serve`) and try again."
        )
        return False
    ready = True
    roles = (("main", models.main), ("fast", models.fast)) if models.engine == "ollama" else ()  # else llama.cpp thinks
    for role, model in (*roles, ("embed", models.embed)):
        if has_model(model, installed):
            continue
        console.print(f"[yellow]{role} model {model} is not installed.[/]")
        if (assume_yes or Confirm.ask(f"Pull {model} now?", default=True)) and pull(models.server, model):
            installed.append(model)
        elif role != "embed":  # without embeddings recall degrades; without a chat model nothing thinks
            ready = False
    return ready


@app.command()
def new(
    path: Annotated[Path | None, typer.Argument(help="Folder to create the brain in.")] = None,
    name: Annotated[str | None, typer.Option(help="Name of the brain.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept the defaults and ask nothing.")] = False,
) -> None:
    """Create a new brain instance: a folder with its settings, memory and workspace."""
    defaults = ModelSettings()
    ask = not yes
    name = name or (Prompt.ask("Name of this brain", default="cortex-1") if ask else "cortex-1")
    if path is None:
        suggestion = Path.home() / "brains" / name
        path = Path(Prompt.ask("Where should it live", default=str(suggestion))) if ask else suggestion
    root = path.expanduser().resolve()
    if (root / CONFIG_FILE).exists():
        console.print(f"[red]A brain already lives at {root}.[/]")
        raise typer.Exit(1)

    settings = BrainSettings(name=name)
    if ask:
        installed = installed_models(defaults.host)
        if installed:
            console.print("Installed models: " + ", ".join(sorted(installed)))
        models = settings.models
        models.main = Prompt.ask("Main model (thinking, planning, coding)", default=models.main)
        models.fast = Prompt.ask("Fast model (the reward critic)", default=models.fast)
        models.embed = Prompt.ask("Embedding model (memory)", default=models.embed)
        settings.safety.allow_code_execution = Confirm.ask(
            "Let it run the Python it writes? (guarded: allow-listed imports, files confined to its workspace)",
            default=True,
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / WORKSPACE_DIR).mkdir(exist_ok=True)
    settings.save(root)
    register(name, root)
    console.print(f"[green]Created {name}[/] at {root}")
    console.print(f"Every region's prompt and every setting is in {root / CONFIG_FILE}.")
    if ask and Confirm.ask("Open it now, dormant, to edit parts of the brain before it first wakes?", default=False):
        start(root, dormant=True)
    else:
        console.print(f"Wake it with: [bold]cortex run {name}[/]")


@app.command()
def run(
    target: Annotated[str | None, typer.Argument(help="Path or name of the brain.")] = None,
    dormant: Annotated[bool, typer.Option(help="Open without waking it, to edit first.")] = False,
    headless: Annotated[bool, typer.Option(help="No interface: print the stream of thought.")] = False,
    seconds: Annotated[float | None, typer.Option(help="Headless only: stop after this long.")] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Run it on this machine itself, not in its container. No full mode.")
    ] = False,
) -> None:
    """Start a brain. It gets no prompt: it wakes and thinks on its own until you quit."""
    start(locate(target), dormant, headless, seconds, local)


def start(
    root: Path, dormant: bool = False, headless: bool = False, seconds: float | None = None, local: bool = False
) -> None:
    """Run the brain in its container. Inside that container, or when asked to, run it right here."""
    if not contained():  # the speakers are on this machine, wherever the brain runs
        from .eyes import remember_own
        from .voice import warm

        remember_own()  # and so is the terminal it is shown in, which its eyes pass over
        warm(BrainSettings.load(root).voice)
    if not (local or contained()):
        from . import container

        args = ["--dormant"] * dormant + ["--headless"] * headless
        args += ["--seconds", str(seconds)] if seconds is not None else []
        try:
            code = container.run(root, args)
        except container.Unavailable as exc:
            console.print(f"[red]{exc}[/]")
            console.print("To run it on this machine itself, guards always on, add [bold]--local[/].")
            raise typer.Exit(1) from None
        raise typer.Exit(code)
    if not contained():  # a mind that sees and hears thinks with llama.cpp's server, on this machine
        from . import llama

        try:
            llama.start(BrainSettings.load(root).models, root)
        except OSError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from None
    if headless:
        asyncio.run(run_headless(root, seconds))
    else:
        launch(root, dormant)


def launch(root: Path, dormant: bool) -> None:
    from .brain import Brain
    from .tui import CortexApp

    brain = Brain(root)
    if not preflight(brain.settings.models):
        brain.close()
        raise typer.Exit(1)
    try:
        CortexApp(brain, dormant=dormant).run()
    finally:
        brain.close()


async def run_headless(root: Path, seconds: float | None) -> None:
    from .brain import Brain
    from .tui import format_signal

    brain = Brain(root)
    if not preflight(brain.settings.models, assume_yes=True):
        brain.close()
        raise typer.Exit(1)
    inbox = brain.bus.subscribe(maxsize=2048)
    task = asyncio.create_task(brain.run())
    try:
        async with asyncio.timeout(seconds):
            while not task.done():
                with contextlib.suppress(TimeoutError):
                    signal = await asyncio.wait_for(inbox.get(), 1)
                    if signal.kind != "proposal":
                        console.print(format_signal(signal, brain))
    except TimeoutError:
        pass
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        count, mean = brain.memory.reward_stats()
        console.print(
            f"[dim]{brain.memory.count()} memories · {count} rewards (mean {mean:+.2f}) · "
            f"{len(brain.learner.heuristics())} heuristics · {brain.llm.speed:.0f} tok/s[/]"
        )
        brain.close()


@app.command("list")
def list_brains() -> None:
    """Show the brains you have created."""
    table = Table("name", "path", "exists")
    for name, root in registry().items():
        table.add_row(name, root, "yes" if (Path(root) / CONFIG_FILE).is_file() else "gone")
    console.print(table)


@app.command()
def doctor(
    target: Annotated[str | None, typer.Argument(help="Path or name of the brain.")] = None,
) -> None:
    """Check the model server, the models a brain needs, and how fast they run here."""
    models = BrainSettings.load(locate(target)).models if target or registry() else ModelSettings()
    if not preflight(models):
        raise typer.Exit(1)
    client = ollama.Client(host=models.server)
    for model in dict.fromkeys((models.main, models.fast)):
        started = time.time()
        reply = client.chat(
            model=model, messages=[{"role": "user", "content": "Count from 1 to 30."}],
            think=models.think, options={"num_ctx": models.num_ctx, "num_predict": 120},
            keep_alive=models.keep_alive,
        )
        rate = reply.eval_count / (reply.eval_duration / 1e9) if reply.eval_duration else 0
        console.print(f"{model}: {rate:.0f} tokens/s (first reply took {time.time() - started:.1f}s including load)")
    client.embed(model=models.embed, input="hello")
    table = Table("loaded model", "size", "on GPU", "context")
    for entry in client.ps().models:
        share = entry.size_vram / entry.size if entry.size else 0
        table.add_row(entry.model, f"{entry.size / 1e9:.1f} GB", f"{share:.0%}", str(getattr(entry, "context_length", "")))
    console.print(table)
    console.print("Anything under 100% on GPU is spilling to the CPU: lower the context window or use a smaller model.")
