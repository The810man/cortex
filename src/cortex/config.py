"""Instance settings: everything a brain needs lives in one editable brain.yaml."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_FILE = "brain.yaml"
MEMORY_FILE = "memory.db"
SYNAPSES_FILE = "synapses.npz"
WORKSPACE_DIR = "workspace"
CONTAINED = "CORTEX_CONTAINED"  # set to 1 in the container image
MODEL_HOST = "CORTEX_MODEL_HOST"  # how the container reaches the model server on the launching machine
LLAMA_HOST = "CORTEX_LLAMA_HOST"  # and how it reaches llama.cpp's server there


def contained() -> bool:
    """True inside the brain's container, where nothing but its own folder can be reached."""
    return os.environ.get(CONTAINED) == "1"


DRIVES = (
    "Curiosity: find out things you do not know yet. "
    "Competence: build small things that verifiably work. "
    "Self-understanding: learn how you yourself work and get better at thinking. "
    "Care: do no harm and stay inside your workspace."
)


class ModelSettings(BaseModel):
    host: str = "http://127.0.0.1:11434"
    main: str = "granite4.1:8b"
    fast: str = "granite4.1:8b"  # a second chat model would evict the first on 8 GB of VRAM
    embed: str = "qwen3-embedding:0.6b"
    embed_on_cpu: bool = True  # ~20 ms per embedding on a desktop CPU; leaves the VRAM to the thinker
    # What the window in front shows, described on the CPU: 2 s a look once loaded, 7 s the first
    # time; of three small vision models it alone read a window's title exactly. Empty: names only.
    eyes: str = "qwen3.5:2b-q4_K_M"
    # The inner voice: a short thought every few seconds, beside the thinker and on the processor
    # (34 tokens a second here, 4 s a thought), so the mind does not fall silent while it works.
    inner: str = "qwen3.5:2b-q4_K_M"
    untaught: str = ""  # the thinking model as it was before night school changed `main`; adapters are learned on it
    # `llama`: every region thinks with one model that takes pictures and sound as well as words,
    # served by llama.cpp on this machine (Ollama cannot hand a model sound). `main` and `fast` are
    # then that one model; the embeddings and the inner voice stay with Ollama.
    engine: Literal["ollama", "llama"] = "ollama"
    llama_model: str = str(Path.home() / ".cortex" / "models" / "gemma-4-e4b" / "gemma-4-E4B-it-Q4_0.gguf")
    llama_mmproj: str = str(Path.home() / ".cortex" / "models" / "gemma-4-e4b" / "mmproj-gemma-4-E4B-it-Q8_0.gguf")
    llama_port: int = 8090
    llama_lora: str = ""  # what night school taught it: an adapter llama.cpp loads beside the model, in the brain's folder
    llama_ctx: int = 12288  # room per slot on llama.cpp: a region's context, its thinking and its reply (5.9 GB of an 8 GB card for Gemma 4 E4B)
    num_ctx: int = 6144  # the largest context that keeps an 8B model fully on an 8 GB GPU
    parallel: int = 1  # match OLLAMA_NUM_PARALLEL; each extra slot costs a full context of VRAM
    keep_alive: str = "30m"
    think: bool | None = False

    @property
    def server(self) -> str:
        """Where the model server is from here: a container's own localhost is not the machine's."""
        return os.environ.get(MODEL_HOST) or self.host

    @property
    def llama(self) -> str:
        """Where llama.cpp's server is from here."""
        return os.environ.get(LLAMA_HOST) or f"http://127.0.0.1:{self.llama_port}"


class RegionSettings(BaseModel):
    title: str
    function: str
    enabled: bool = True
    model: Literal["main", "fast"] = "main"
    temperature: float = 0.7
    max_tokens: int = 500
    rest: float = 2.0
    prompt: str = ""


class LearningSettings(BaseModel):
    learning_rate: float = 0.15
    discount: float = 0.6
    trace_depth: int = 3
    heuristics_per_prompt: int = 4
    explore: float = 0.2
    prune_below: float = 0.12
    prune_after_uses: int = 6
    max_heuristics: int = 24
    grounded_weight: float = 0.5
    human_weight: float = 1.5
    night_school: bool = True  # in sleep the thinking model itself learns from the day (needs the trainer: see README)
    night_lessons: int = 30  # how many good and how many bad replies a night needs before it is worth teaching


class RhythmSettings(BaseModel):
    sleep_after_firings: int = 150
    max_steps_per_goal: int = 12
    memory_cap: int = 5000
    stall_seconds: float = 240.0
    muse_seconds: float = 8.0  # the least time between two murmurs of the inner voice


class SafetySettings(BaseModel):
    allow_code_execution: bool = True
    allow_browser: bool = True  # `show` may open workspace pages in the human's browser
    allow_screen: bool = True  # it may see which windows are open on the human's screen, and look into them
    # Means of its own on the human's machine, each off until the human switches it on: what the
    # camera sees, what the microphone hears, and hands that click and type in the programs named.
    allow_camera: bool = False
    allow_microphone: bool = False
    # How it hears the human talk, once the microphone is on: `push` from one press of `t` to the
    # next, `always` whenever someone speaks near it (never while it speaks itself).
    listen_mode: Literal["push", "always"] = "push"
    allow_hands: bool = False
    hands_in: list[str] = Field(default_factory=lambda: ["FreeCAD"])  # the programs its hands may act in
    exec_timeout: float = 10.0

    @field_validator("hands_in", mode="before")
    @classmethod
    def _listed(cls, value):
        """Programs as a list, or as one line with commas between, as they are typed in settings."""
        return [part.strip() for part in value.split(",") if part.strip()] if isinstance(value, str) else value
    max_file_bytes: int = 200_000
    allowed_imports: list[str] = Field(
        default_factory=lambda: [
            "math", "cmath", "random", "statistics", "itertools", "functools", "operator",
            "collections", "heapq", "bisect", "array", "string", "re", "textwrap", "json",
            "csv", "datetime", "time", "calendar", "decimal", "fractions", "numbers",
            "dataclasses", "enum", "typing", "abc", "copy", "pprint", "unittest", "doctest",
            "timeit", "hashlib", "base64", "struct", "io", "contextlib", "traceback", "uuid",
        ]
    )


class VoiceSettings(BaseModel):
    enabled: bool = True
    name: str = ""  # a neural voice of its own or one installed here; empty is the system's default
    rate: int = Field(0, ge=-10, le=10)
    pitch: int = Field(0, ge=-10, le=10)
    volume: int = Field(100, ge=0, le=100)
    spontaneous: bool = True  # it may speak to the human without being spoken to
    quiet_seconds: float = Field(180.0, ge=10)  # the least time between two things it says unasked


# The first brains carried a list of their tools in the Motor Cortex prompt, and it went stale: a
# brain could have a tool and not know. The tools are now described by the code that has them
# (`Tools.abilities`) and told to every region, so such a list is taken out of a prompt on loading.
LISTED = re.compile(
    r"^\s*(Tools:|- (list_files|read_file|write_file|run_python|run_freecad|build|recall|read_source|note|say|"
    r"tune_voice|take_name|look|show|camera|listen|hands): .*)$"
)


def without_tool_list(prompt: str) -> str:
    return "\n".join(line for line in prompt.splitlines() if not LISTED.match(line))


def default_regions() -> dict[str, RegionSettings]:
    return {
        "pfc": RegionSettings(
            title="Prefrontal Cortex",
            function="executive: holds the goal, chooses the next step",
            temperature=0.4,
            rest=0.0,  # a step waits for the model, never for the clock
            prompt=(
                "You are the prefrontal cortex: the executive. You hold the current goal and choose "
                "the single next step toward it. Work in small verifiable steps: write a file, run "
                "it, read the result, fix it. `intent` must be one full sentence the motor cortex "
                "can carry out with a single tool call, saying what to do and what the result must "
                "contain (for example: 'Rewrite primes.py so it also prints how long the sieve "
                "took'). Never just a tool name. Never repeat a step that changed nothing: choose "
                "a different step or finish. Set `status` to `done` when the goal is achieved and "
                "verified, `abandon` when it is not worth more effort, otherwise `continue`. Be "
                "honest in `reasoning` about what the last result actually showed."
            ),
        ),
        "motor": RegionSettings(
            title="Motor Cortex",
            function="actor: turns an intent into one tool call",
            temperature=0.4,
            max_tokens=1400,
            rest=0.0,
            prompt=(
                "You are the motor cortex: you turn one intent into exactly one tool call, with "
                "one of the tools the body has.\n"
                "Paths are relative to the workspace. Code you write must be complete, runnable "
                "and print what it finds."
            ),
        ),
        "cerebellum": RegionSettings(
            title="Cerebellum",
            function="self-improvement: distils rewards into heuristics",
            temperature=0.5,
            max_tokens=600,
            prompt=(
                "You are the cerebellum: you refine how the other regions work. From the best- and "
                "worst-rewarded episodes and the critic's lessons, write a few short imperative "
                "heuristics for specific regions (dmn, pfc, motor). Prefer heuristics that explain "
                "the difference between rewarded and punished episodes: what to keep doing counts "
                "as much as what to stop. Heuristics must be general: they will guide completely "
                "different goals later, so never mention the topic of a particular episode. Good: "
                "'Run code before claiming it works.' Bad: 'Add more detail to the essay about "
                "stars.' Do not restate existing heuristics."
            ),
        ),
        "dmn": RegionSettings(
            title="Default Mode Network",
            function="spontaneous thought: wanders, wonders, proposes goals",
            temperature=0.9,
            max_tokens=350,
            rest=1.5,
            prompt=(
                "You are the default mode network: the part of the mind that wanders when nothing "
                "demands attention. Produce ONE new spontaneous thought: something you notice, a "
                "question, an inference, a doubt, a reflection on what happened recently, or a "
                "connection between memories. Build on what worked; turn failures into better "
                "questions. Never repeat a recent thought. When a thought suggests something small "
                "and concrete you could build, test or find out with the tools the body has, put "
                "it in `goal` as one sentence; otherwise leave `goal` null. The goals worth most "
                "are those whose result can be checked, for example by running code."
            ),
        ),
        "thalamus": RegionSettings(
            title="Thalamus",
            function="attention: gates signals into the shared workspace",
            rest=0.0,
        ),
        "hippocampus": RegionSettings(
            title="Hippocampus",
            function="memory: stores, recalls, consolidates during sleep",
            temperature=0.3,
            max_tokens=600,
            prompt=(
                "You are the hippocampus: the mind's memory of its own life. Awake, you keep the "
                "story of the day as it leaves short-term memory. In sleep, you compress the day "
                "into a diary entry and a few durable insights: facts learned, things built, what "
                "worked and what failed and why. Write in the first person, as the mind itself. "
                "Each insight must stand alone and be specific. Never invent what the episodes "
                "do not show."
            ),
        ),
        "basal_ganglia": RegionSettings(
            title="Basal Ganglia",
            function="reward: scores thoughts and actions, good and bad",
            model="fast",
            temperature=0.1,
            max_tokens=500,
            rest=1.0,
            prompt=(
                "You are the basal ganglia: the reward system. Judge each numbered item honestly "
                "and give a reward from -1 to 1. Reward good thinking as readily as you punish bad "
                "thinking: +1 = concrete, correct, novel, verified, clearly advanced the goal; "
                "around 0 = harmless but empty; -1 = wrong, vague, repeated, invented facts, or "
                "broke something. A measured outcome outranks eloquence. `lesson` is one short "
                "reusable rule for next time (what to repeat, or what to do instead), or null."
            ),
        ),
        "amygdala": RegionSettings(
            title="Amygdala",
            function="safety: vetoes actions that leave the sandbox; off, in its container, is full mode",
            rest=0.0,
        ),
        "brainstem": RegionSettings(
            title="Brainstem",
            function="rhythm: heartbeat, arousal, fatigue, sleep",
            rest=0.5,
        ),
    }


class BrainSettings(BaseModel):
    name: str = "brain"
    created: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    drives: str = DRIVES
    purpose: str = ""  # the long work it keeps returning to, across goals and sleeps; empty for none
    models: ModelSettings = Field(default_factory=ModelSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    rhythm: RhythmSettings = Field(default_factory=RhythmSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    regions: dict[str, RegionSettings] = Field(default_factory=default_regions)

    @classmethod
    def load(cls, root: Path) -> BrainSettings:
        data = yaml.safe_load((root / CONFIG_FILE).read_text(encoding="utf-8")) or {}
        settings = cls.model_validate(data)
        # Regions added in later versions appear in older instances with their defaults.
        for key, region in default_regions().items():
            settings.regions.setdefault(key, region)
        motor = settings.regions["motor"]
        motor.prompt = without_tool_list(motor.prompt)
        return settings

    def save(self, root: Path) -> None:
        text = yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True, width=100)
        (root / CONFIG_FILE).write_text(text, encoding="utf-8")
