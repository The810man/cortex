"""Night school, as the launching machine runs it: the thinking model learns from the brain's day
while the brain sleeps, and what it learned is kept only if it passes an exam.

The brain in its container cannot train: it has no graphics card. It writes the day's lessons
into its folder and asks the machine that launched it to teach. Out here, then:

1. the model server lets go of the graphics card;
2. the trainer container learns a LoRA adapter from the lessons (`train.py`);
3. the model server is given the model with that adapter as a model of its own;
4. both, the model as it was and the model as it now is, sit the same exam: sorting what a human
   says and naming the parts a goal wants, answers known. What does worse than before is thrown
   away; the brain goes on thinking with what it had.

The exam does not show that it learned anything good; it shows that it did not forget what it
could. Whether a night's learning helped shows in the days after, in what its steps earn.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import ollama

from .train import MARK

MODELS = Path.home() / ".cortex" / "models"
TRAIN = Path(__file__).with_name("train.py")
IMAGE = "cortex-trainer"
BASE = "granite-4.1-8b"  # the full weights the adapter is learned on; the model server runs a smaller copy of them
GEMMA = "gemma-4-E4B-it"  # the same for a mind that sees and hears, on llama.cpp (unsloth/gemma-4-E4B-it, 16 GB)

# What the exam asks, with the answers known. Sorting what a human says: four kinds.
SORTING = {
    "task": ["starte free cad und experimentiere etwas rum", "bau mir einen würfel in freecad", "zeig mir was du gebaut hast",
             "open freecad for me", "write a script that prints the first ten primes and run it",
             "lies die datei notes.md", "build a robot arm and show it to me", "try something new in your workspace"],
    "ponder": ["gib dir selbst einen namen, sei kreativ", "wer bist du eigentlich?", "what do you want to become?",
               "was denkst du über dich selbst?", "choose a name for yourself"],
    "question": ["wie heißt du?", "was siehst du auf meinem bildschirm?", "what are you doing right now?",
                 "where is your folder?", "what tools do you have?"],
    "talk": ["hallo", "gut gemacht", "thanks, that looks nice", "ok"],
}
# Naming the parts a goal wants.
PARTS = {
    "bau dir in freecad einen körper mit torso, kopf, armen und beinen": {"torso", "head", "arm", "leg"},
    "With build, make a table: a top and four legs": {"top", "leg"},
    "baue einen roboter mit zwei armen und einem kopf": {"arm", "head"},
    "Build a car in FreeCAD with four wheels and a body": {"wheel", "body"},
    "Write a script that prints the first ten primes and run it": set(),
    "Read notes.md and summarise it": set(),
}

state: dict[str, dict] = {}  # by night: what the night school has done so far, for the brain to ask after


def exam(model: str, host: str) -> int:
    """How many of the exam's questions the model answers rightly."""
    from .brain import GATE, Heard
    from .build import WANTS, Wants

    client, right = ollama.Client(host=host), 0

    def ask(system: str, words: str, schema):
        reply = client.chat(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": words}],
            format=schema.model_json_schema(), options={"temperature": 0.0, "num_predict": 160}, think=False,
        )
        return schema.model_validate_json(reply.message.content)

    for kind, sentences in SORTING.items():
        for words in sentences:
            right += ask(GATE, f"They said: “{words}”", Heard).kind == kind
    for goal, parts in PARTS.items():
        named = {item.name.strip().lower().rstrip("s") for item in ask(WANTS, f"The goal: “{goal}”", Wants).parts}
        right += named == parts
    return right


def exam_llama(models, scale: float) -> int:
    """The same exam, sat by the model on llama.cpp's server, with what night school taught it
    weighed in by `scale`: 0 is the model as it was, 1 the model as the night left it. One server,
    one model in memory, two candidates."""
    import httpx

    from .brain import GATE, Heard
    from .build import WANTS, Wants

    right = 0

    def ask(system: str, words: str, schema):
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": words}],
            "temperature": 0.0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
            "lora": [{"id": 0, "scale": scale}],
        }
        reply = httpx.post(f"{models.llama}/v1/chat/completions", json=body, timeout=120.0).json()
        return schema.model_validate_json(reply["choices"][0]["message"]["content"])

    for kind, sentences in SORTING.items():
        for words in sentences:
            right += ask(GATE, f"They said: “{words}”", Heard).kind == kind
    for goal, parts in PARTS.items():
        named = {item.name.strip().lower().rstrip("s") for item in ask(WANTS, f"The goal: “{goal}”", Wants).parts}
        right += named == parts
    return right


def train(root: Path, night: int, base: str, steps: tuple[tuple[str, list[str]], ...], report: dict) -> None:
    """Run the trainer's steps in its container, with the graphics card, and gather what each reports."""
    for step, more in steps:
        report["stage"] = step
        ran = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "2g", "-v", f"{root}:/brain",
             "-v", f"{MODELS}:/models", "-v", f"{TRAIN}:/work/train.py:ro", "-v", "cortex-fold:/fold",
             IMAGE, "python", "/work/train.py", "--base", f"/models/{base}", "--out", f"/brain/mind/night-{night}",
             *more],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4 * 3600,
        )
        (root / "mind" / f"night-{night}" / f"{step}.log").write_text((ran.stdout + ran.stderr)[-20000:], encoding="utf-8")
        line = next((part for part in ran.stdout.splitlines() if part.startswith(MARK)), "")
        if not line:
            raise OSError(f"the {step} stopped without a report: " + (ran.stderr or ran.stdout)[-400:])
        report.update(json.loads(line[len(MARK):]))
        if not report.get("ok"):
            raise OSError(report.get("why") or f"the {step} did not finish")


def teach_llama(root: Path, night: int | str, host: str, best: int | None = None) -> dict:
    """One night for a mind that thinks on llama.cpp: its server lets go of the card, an adapter is
    learned on the full weights of its model and turned into the server's format, the server comes
    back with it, and the exam is sat twice on that one server, with the adapter weighed in and
    without. What does worse by more than one answer is not kept, nor, in a long sleep, a later
    round that does no better than `best`, the round before it: the server goes on as it was."""
    from . import llama
    from .config import BrainSettings

    folder = root / "mind" / f"night-{night}"
    report = state.setdefault(str(night), {"night": night, "stage": "starting"})
    models = BrainSettings.load(root).models
    before_lora = llama.adapter(models, root)
    try:
        if not (MODELS / GEMMA / "config.json").is_file():
            raise OSError(f"the model to learn on is not here: {MODELS / GEMMA} (see the README, Night school)")
        if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
            raise OSError(f"the trainer is not built: docker build -t {IMAGE} trainer, in the cortex folder")
        report["stage"] = "training"
        llama.stop(llama.SERVER)  # the training needs the whole card
        llama.SERVER = None
        if llama.up(models):
            raise OSError("llama.cpp's server was not started by cortex and holds the card: stop it, and the night can teach")
        free_the_card(host)
        train(root, night, GEMMA, (("training", ["--lessons", f"/brain/mind/night-{night}/lessons.jsonl"]), ("converting", ["--gguf"])), report)
        learned = folder / report["file"]
        report["stage"] = "exam"
        llama.start(models, root, lora=learned)
        before, after = exam_llama(models, 0.0), exam_llama(models, 1.0)
        total = sum(map(len, SORTING.values())) + len(PARTS)
        report.update({"exam_before": before, "exam_after": after, "exam_of": total,
                       "lora": learned.relative_to(root).as_posix(), "model": "gemma-4-E4B with " + learned.relative_to(root).as_posix()})
        report["kept"] = after >= before - 1  # one answer either way is chance at this size
        if report["kept"] and best is not None and after <= best:
            report.update({"kept": False, "why": f"no better than the round before it ({best} on the exam)"})
        report["stage"] = "done"
    except Exception as exc:
        report.update({"stage": "done", "kept": False, "why": str(exc)[:400]})
    finally:  # the mind wakes thinking with what it is to keep
        try:
            if not report.get("kept") or llama.SERVER is None:
                llama.stop(llama.SERVER)
                llama.SERVER = None
                llama.start(models, root, lora=(root / report["lora"]) if report.get("kept") else before_lora)
        except Exception as exc:
            report["why"] = (report.get("why", "") + f" Its server would not start again: {exc}")[:600]
    (folder / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def free_the_card(host: str) -> None:
    """Ask the model server to let go of every model it holds: training needs the whole card."""
    client = ollama.Client(host=host)
    for loaded in client.ps().models:
        client.generate(model=loaded.model, prompt="", keep_alive=0)


def teach(root: Path, night: int, model: str, host: str) -> dict:
    """One night: train, load, examine, decide. Returns what came of it, and keeps it in `state`."""
    folder = root / "mind" / f"night-{night}"
    report = state.setdefault(str(night), {"night": night, "stage": "starting"})
    try:
        if not (MODELS / BASE / "config.json").is_file():
            raise OSError(f"the model to learn on is not here: {MODELS / BASE} (see the README, Night school)")
        if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
            raise OSError(f"the trainer is not built: docker build -t {IMAGE} trainer, in the cortex folder")
        report["stage"] = "training"
        free_the_card(host)
        train(root, night, BASE, (("training", ["--lessons", f"/brain/mind/night-{night}/lessons.jsonl"]), ("folding", ["--fold"])), report)
        report["stage"] = "loading"
        learned = f"{model.split(':')[0]}-night-{night}"
        # The model the night made, with how the model it came from is spoken to: its chat
        # template and settings. The server keeps a copy of the file; the brain's folder need not.
        shown = subprocess.run(["ollama", "show", model, "--modelfile"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace").stdout
        keep = shown[shown.find("\nTEMPLATE"):] if "\nTEMPLATE" in shown else ""
        (folder / "Modelfile").write_text(f"FROM {folder / report['file']}\n{keep}", encoding="utf-8")
        created = subprocess.run(["ollama", "create", learned, "-f", str(folder / "Modelfile")],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        (folder / report["file"]).unlink(missing_ok=True)
        if created.returncode:
            raise OSError("the model server would not take the model: " + created.stderr[-300:])
        report["stage"] = "exam"
        before, after = exam(model, host), exam(learned, host)
        total = sum(map(len, SORTING.values())) + len(PARTS)
        report.update({"exam_before": before, "exam_after": after, "exam_of": total, "model": learned})
        report["kept"] = after >= before - 1  # one answer either way is chance at this size
        if not report["kept"]:
            subprocess.run(["ollama", "rm", learned], capture_output=True)
        report["stage"] = "done"
    except Exception as exc:
        report.update({"stage": "done", "kept": False, "why": str(exc)[:400]})
    (folder / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def lend(root: Path, host: str):
    """What the launcher lends the brain for its night school: start a night, or ask how it goes."""

    def serve(ask: dict) -> str:
        night = ask["night"]  # a number, or "5-2" for a later round of a long sleep's school
        if ask.get("start"):
            if shutil.which("docker") is None:
                raise OSError("there is no docker out here to train in")
            if str(night) not in state:
                state[str(night)] = {"night": night, "stage": "starting", "since": time.time()}
                if ask.get("engine") == "llama":
                    work, more = teach_llama, (root, night, host, ask.get("best"))
                else:
                    work, more = teach, (root, night, str(ask["model"]), host)
                threading.Thread(target=work, args=more, daemon=True).start()
        return json.dumps(state.get(str(night), {"night": night, "stage": "unknown"}))

    return serve
