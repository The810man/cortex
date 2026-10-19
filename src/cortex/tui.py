"""The terminal interface: a live map of the brain and its stream of thought."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TextArea,
)
from textual.worker import Worker, WorkerState

from .brain import Brain
from .bus import Signal
from .regions import NEURONS, Region, clip, span
from .tools import sight

COLOURS = {
    "pfc": "#5fd7ff", "motor": "#87d75f", "cerebellum": "#d787ff",
    "dmn": "#8787ff", "thalamus": "#ffd75f", "hippocampus": "#ffaf5f",
    "basal_ganglia": "#ff5faf", "amygdala": "#ff5f5f", "brainstem": "#a8a8a8",
}
LEVELS = {"error": "red", "threat": "bold red", "sleep": "#8787ff", "wake": "bold", "info": "dim"}
GLYPHS = {"firing": "●", "queued": "◍", "idle": "○", "asleep": "☾", "off": "×"}
BARS = "▁▂▃▄▅▆▇█"
DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))  # a braille cell's dots, by row and column
# How far each neuron's firing must have faded before its dot goes out: no two at the same moment.
DARK = np.random.default_rng(7).uniform(0.1, 0.8, NEURONS).astype(np.float32)
LONG_FIELDS = {"prompt", "drives", "purpose"}
Field = tuple[str, BaseModel, str]  # label, settings object, attribute


def spark(values: Iterable[float], lo: float = 0.0, hi: float = 1.0) -> str:
    span = (hi - lo) or 1.0
    return "".join(BARS[max(0, min(7, int((v - lo) / span * 7.999)))] for v in values)


def field(neurons: np.ndarray, width: int) -> str:
    """A region's neurons as they lie on their sheet: two lines of braille, eight to a character,
    lit where they fire."""
    width = min(width, NEURONS // 16)  # two lines of them
    cells = [[0] * width, [0] * width]
    for index in np.flatnonzero(neurons[: width * 16] > DARK[: width * 16]):
        column, row = divmod(int(index), 8)  # a neuron's place: columns of eight, top to bottom
        cells[row // 4][column // 2] |= DOTS[row % 4][column % 2]
    return "\n".join("".join(chr(0x2800 + cell) for cell in line) for line in cells)


def meter(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 18) -> str:
    filled = round(max(0.0, min(1.0, (value - lo) / (hi - lo))) * width)
    return "█" * filled + "░" * (width - filled)


def format_signal(signal: Signal, brain: Brain) -> Text:
    """One line of the stream of consciousness."""
    line = Text(time.strftime("%H:%M:%S ", time.localtime(signal.ts)), style="dim")
    titles = {key: region.title for key, region in brain.settings.regions.items()}
    if signal.kind == "reward":
        reward = signal.data["reward"]
        line.append(f"{'▲' if reward >= 0 else '▼'} {reward:+.2f} ", style="bold green" if reward >= 0 else "bold red")
        who = {"human": "you", "reflex": "reflex"}.get(signal.data["source"], "critic")
        line.append(f"{who} → {titles[signal.data['region']]} ", style=COLOURS[signal.data["region"]])
        line.append(f"“{signal.data['target']}” ", style="dim")
        line.append(signal.content)
        if signal.data.get("lesson"):
            line.append(f"  ⟶ {signal.data['lesson']}", style="italic")
    elif signal.kind == "system":
        line.append(f"· {signal.content}", style=LEVELS.get(signal.data.get("level", "info"), "dim"))
    elif signal.kind == "stimulus":
        line.append("outside stimulus ", style="bold white")
        line.append(signal.content)
    elif signal.kind == "speech":
        unasked = " unasked" if signal.data.get("unasked") else ""
        line.append(f"{brain.name} says{unasked} ", style=f"bold {COLOURS[signal.source]}")
        line.append(f"“{signal.content}”", style="bold")
    elif signal.kind == "musing":  # the inner voice, beside the thinker
        line.append("… inner voice ", style=COLOURS[signal.source])
        line.append(signal.content, style="italic")
    elif signal.kind == "echo":  # the undercurrent: no model wrote this, a pattern found it
        line.append("⋯ comes to mind ", style=COLOURS[signal.source])
        line.append(clip(signal.content, 160), style="dim italic")
    elif signal.kind == "remark":  # one region telling another: no model wrote this either
        line.append(f"{titles[signal.source]} → {titles[signal.data['to']]} ", style=COLOURS[signal.source])
        line.append(signal.content, style="italic")
    elif signal.kind == "view":  # its turn while the regions think something through together
        line.append(f"{titles[signal.source]} ", style=f"bold {COLOURS[signal.source]}")
        line.append(f"thinks, then hands to the {signal.data.get('then', 'executive')} ", style="dim")
        line.append(signal.content)
    else:
        line.append(f"{titles[signal.source]} ", style=f"bold {COLOURS[signal.source]}")
        line.append(f"{signal.kind} ", style="dim")
        body = "\n".join(signal.content.splitlines()[:10])
        line.append(body if len(body) <= 700 else body[:700] + "…")
    return line


class RegionCard(Static):
    def __init__(self, region: Region) -> None:
        super().__init__(classes="card")
        self.part = region
        self.look: tuple[bool, bool] | None = None

    def on_mount(self) -> None:
        self.border_title = self.part.cfg.title

    def refresh_card(self, asleep: bool) -> None:
        region = self.part
        colour = COLOURS[region.key]
        state = region.state
        if not region.cfg.enabled:
            state = "off"
        elif asleep and not region.nocturnal and state == "idle":
            state = "asleep"
        hot = state in ("firing", "queued")
        width = max(10, self.size.width - 4)  # inside the border and padding
        status = f"{GLYPHS[state]} {state}"
        counts = f" {region.fires}×" + (f" {region.tokens / 1000:.1f}k" if region.tokens else "")
        text = Text()
        text.append(status, style=f"bold {colour}" if hot else "dim")
        text.append(counts[: width - len(status)], style="dim")
        text.append("\n" + field(region.brain.net.sheet(region.key), width) + "\n", style=colour)
        text.append(clip(region.last or region.cfg.function, width * 2), style="" if region.last else "dim italic")
        self.update(text)
        self.border_subtitle = spark(list(region.history)[-min(16, width):])
        look = (hot, region.activity > 0.25)
        if look != self.look:
            self.look = look
            self.styles.border = ("heavy" if hot else "round", colour if look[1] else "#444444")


class Form(ModalScreen[str | None]):
    """Edits fields of settings objects in place. Validation is the settings models' own."""

    BINDINGS = [Binding("escape", "cancel", "Close"), Binding("ctrl+s", "save", "Save")]

    def __init__(
        self,
        title: str,
        fields: list[Field],
        on_save: Callable[[], None],
        chooser: tuple[list[tuple[str, str]], str] | None = None,
        hint: str = "",
    ) -> None:
        super().__init__()
        self.title_text, self.fields, self.on_save, self.chooser, self.hint = title, fields, on_save, chooser, hint

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dialog"):
            yield Label(self.title_text, id="dialog-title")
            if self.chooser:
                options, current = self.chooser
                yield Select(options, value=current, allow_blank=False, id="chooser")
            if self.hint:
                yield Label(self.hint, classes="hint")
            for index, (label, obj, attr) in enumerate(self.fields):
                value = getattr(obj, attr)
                yield Label(label, classes="field-label")
                if isinstance(value, bool) or value is None:
                    yield Switch(value=bool(value), id=f"f{index}")
                elif isinstance(value, list):
                    yield Input(", ".join(map(str, value)), id=f"f{index}")
                elif attr in LONG_FIELDS:
                    yield TextArea(value, id=f"f{index}", classes="long")
                else:
                    yield Input(str(value), id=f"f{index}")
            with Horizontal(id="buttons"):
                yield Button("Save  ctrl+s", variant="primary", id="save")
                yield Button("Close  esc", id="cancel")

    @on(Select.Changed, "#chooser")
    def choose(self, event: Select.Changed) -> None:
        if self.chooser and event.value != self.chooser[1]:
            self.dismiss(str(event.value))

    @on(Button.Pressed, "#save")
    def action_save(self) -> None:
        edits: dict[int, tuple[BaseModel, dict[str, Any]]] = {}
        for index, (_, obj, attr) in enumerate(self.fields):
            widget = self.query_one(f"#f{index}")
            value = widget.text if isinstance(widget, TextArea) else widget.value
            edits.setdefault(id(obj), (obj, {}))[1][attr] = value
        try:
            checked = [
                (obj, new, type(obj).model_validate({**obj.model_dump(), **new})) for obj, new in edits.values()
            ]
        except ValidationError as exc:
            error = exc.errors()[0]
            self.notify(f"{'.'.join(map(str, error['loc']))}: {error['msg']}", severity="error")
            return
        for obj, new, clean in checked:
            for attr in new:
                setattr(obj, attr, getattr(clean, attr))
        self.on_save()
        self.notify("Saved. It takes effect on the next firing.")
        self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class Asleep(ModalScreen[str | None]):
    """How long it may sleep: empty for one night as usual, or hours, as for a night of your own."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Send it to sleep", id="dialog-title")
            yield Label(
                "Empty: one night as usual. Or the hours it may sleep, such as 8: it goes to school in "
                "its sleep for as long as that makes it better, then wakes by itself, or when the time "
                "is up, or when you press w or speak to it.",
                classes="hint",
            )
            yield Input(placeholder="hours, or empty")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class Whisper(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Talk to it", id="dialog-title")
            yield Label(
                "The brain needs none. It answers what you type here at once, aloud, and drops a "
                "goal of its own for anything you ask it to do.",
                classes="hint",
            )
            yield Input(placeholder="something to say, ask or have it try…")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class Mind(ModalScreen[None]):
    """What the brain knows of itself right now: the same words its regions are given."""

    BINDINGS = [Binding("escape,m", "dismiss", "Close")]

    def __init__(self, brain: Brain) -> None:
        super().__init__()
        self.brain = brain

    def compose(self) -> ComposeResult:
        brain = self.brain
        text = Text()
        for title, body in (
            ("What it is called", f"{brain.called}: a name it gave itself. Its folder is {brain.settings.name}."
             if brain.called else f"{brain.settings.name}, after its folder. It has not given itself a name."),
            ("Who it is, in its own words from its last sleep", brain.self_image or "It has not slept yet."),
            ("What it wants most right now", brain.want()),
            ("Its inner voice, the latest last", "\n".join(f"- {m}" for m in brain.musings) or "Silent so far."),
            ("What it has been getting better at", "\n".join(f"- {i}" for i in brain.interests()) or "Nothing yet."),
            ("Its open questions", "\n".join(f"- {q}" for q, _ in brain.questions) or "None."),
            ("How well it knows itself", brain.self_knowledge() or "Too few steps yet to say."),
            ("Its long work", brain.settings.purpose or "None. Give it one in settings (s)."),
            ("Its body, as the machine reports it to every region", brain.body()),
            ("What it senses", brain.sense()),
            ("What it sees on your screen", sight(brain.seen) if brain.seen else "Nothing: it cannot look, or may not."),
            ("What the window in front showed when it last looked closer",
             f"{brain.vision['window']}: {brain.vision['seen']}" if brain.vision else "It has not looked closer yet."),
            ("The model it is building", brain.tools.blueprint() or "None built out of parts yet."),
            ("The day so far, as it has compacted it", brain.story or "Nothing has left short-term memory yet."),
            ("Its diary from before it last slept", brain.yesterday or "No entry yet."),
            ("Short-term memory, the latest last", brain.stream(len(brain.workspace) or 1)),
            ("Coming to mind by itself", "\n".join(f"- {clip(r['content'], 200)}" for r in brain.mind) or "Nothing yet."),
        ):
            text.append(f"{title}\n", style="bold")
            text.append(f"{body}\n\n")
        with VerticalScroll(id="dialog"):
            yield Label("Its mind — what it is aware of right now", id="dialog-title")
            yield Static(text)
            yield Label("esc to close", classes="hint")


class Wiring(ModalScreen[None]):
    """The tracts between the regions: how long each is, how strong, and what it has learned."""

    BINDINGS = [Binding("escape,n", "dismiss", "Close")]

    def __init__(self, brain: Brain) -> None:
        super().__init__()
        self.brain = brain

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Its wiring — the tracts between its regions", id="dialog-title")
            yield DataTable(zebra_stripes=True, cursor_type="row")
            yield Label(
                "strength: how much of a signal arrives, laid down by anatomy and moved by reward · steps: "
                "thoughts that led along it to the next · synapses: pairs of neurons it has wired · esc to close",
                classes="hint",
            )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("from", "to", "length", "delay", "strength", "laid down", "steps", "synapses", "mean weight")
        titles = {key: region.title for key, region in self.brain.settings.regions.items()}
        tracts = sorted(self.brain.net.tracts.items(), key=lambda item: (-item[1].carried, -item[1].strength))
        for (src, dst), tract in tracts:
            wired = tract.weights[tract.weights != 0] if tract.weights is not None else np.zeros(0)
            table.add_row(
                titles[src], titles[dst], f"{tract.length:.0f} mm", f"{tract.delay * 0.5:.1f} s",
                f"{meter(tract.strength, width=8)} {tract.strength:.2f}", f"{tract.innate:.2f}", str(tract.carried),
                str(len(wired)), f"{wired.mean():+.3f}" if len(wired) else "",
            )


class Heuristics(ModalScreen[None]):
    BINDINGS = [Binding("escape,h", "dismiss", "Close")]

    def __init__(self, brain: Brain) -> None:
        super().__init__()
        self.brain = brain

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Learned heuristics — what reward has taught each region", id="dialog-title")
            yield DataTable(zebra_stripes=True, cursor_type="row")
            yield Label("weight rises with better-than-expected reward and falls with worse · esc to close", classes="hint")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("region", "weight", "uses", "avg reward", "status", "heuristic")
        for h in self.brain.learner.heuristics(active_only=False):
            table.add_row(
                self.brain.settings.regions[h.region].title, f"{meter(h.weight, width=8)} {h.weight:.2f}",
                str(h.uses), f"{h.mean_reward:+.2f}", "active" if h.active else "suppressed", h.text,
            )


class CortexApp(App[None]):
    CSS = """
    #left { width: 1fr; min-width: 60; }
    #right { width: 1fr; }
    #map { grid-size: 3 3; grid-rows: 7; grid-gutter: 0 1; height: auto; }
    .card { height: 7; padding: 0 1; border: round #444444; border-subtitle-color: #666666; }
    #live, #vitals, #stream { border: round #444444; padding: 0 1; }
    #live { height: 1fr; }
    #vitals { height: auto; }
    #stream { height: 1fr; }
    Form, Whisper, Heuristics, Mind, Wiring { align: center middle; }
    #dialog { width: 90%; max-width: 110; height: auto; max-height: 90%; padding: 1 2;
              border: thick $accent; background: $surface; }
    #dialog-title { text-style: bold; margin-bottom: 1; }
    .hint { color: $text-muted; margin-bottom: 1; }
    .field-label { color: $text-muted; margin-top: 1; }
    .long { height: 10; }
    #buttons { height: auto; margin-top: 1; }
    #buttons Button { margin-right: 2; }
    DataTable { height: auto; max-height: 30; }
    """

    BINDINGS = [
        Binding("space", "pause", "Pause"),
        Binding("e", "edit", "Edit region"),
        Binding("s", "settings", "Settings"),
        Binding("h", "heuristics", "Heuristics"),
        Binding("m", "mind", "Mind"),
        Binding("n", "wiring", "Neurons"),
        Binding("i", "whisper", "Talk"),
        Binding("t", "speak_aloud", "Speak"),
        Binding("plus,equals_sign", "praise", "Reward"),
        Binding("minus", "scold", "Punish"),
        Binding("z", "sleep", "Sleep"),
        Binding("w", "wake", "Wake"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, brain: Brain, dormant: bool = False) -> None:
        super().__init__()
        self.brain = brain
        self.dormant = dormant
        self.inbox = brain.bus.subscribe(maxsize=2048)
        self.started = time.time()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left"):
                with Grid(id="map"):
                    for region in self.brain.regions.values():
                        yield RegionCard(region)
                yield Static(id="live")
            with Vertical(id="right"):
                yield Static(id="vitals")
                yield RichLog(id="stream", wrap=True, max_lines=3000)
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"cortex · {self.brain.settings.name}"
        self.query_one("#live").border_title = "firing now"
        self.query_one("#vitals").border_title = "state"
        self.query_one("#stream").border_title = "stream of thought"
        self.set_interval(0.25, self.refresh_view)
        self.refresh_view()
        if self.dormant:
            self.query_one(RichLog).write(
                Text("Dormant. Edit regions (e) and settings (s), then press w to let it awaken.", style="bold")
            )
        else:
            self.action_wake()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "brain" and event.state == WorkerState.ERROR:
            self.notify(f"The brain stopped: {event.worker.error}", severity="error", timeout=30)

    # -- drawing ----------------------------------------------------------------------

    def refresh_view(self) -> None:
        brain = self.brain
        while not self.inbox.empty():
            signal = self.inbox.get_nowait()
            if signal.kind != "proposal":
                self.query_one(RichLog).write(format_signal(signal, brain))
        asleep = not brain.awake.is_set()
        self.title = f"cortex · {brain.name}"  # it may have given itself a name since the last look
        if self.dormant:
            self.sub_title = "dormant"
        elif not brain.running.is_set():
            self.sub_title = "paused"
        else:
            self.sub_title = "asleep" if asleep and brain.tick else "awake"
        for card in self.query(RegionCard):
            card.refresh_card(asleep)

        live = Text()
        if brain.reply is not None:
            live.append(f"{brain.name} is answering you ", style="bold")
            live.append((brain.reply[-420:].replace("\\n", "\n") or "…") + "\n")
        for region in brain.regions.values():
            if region.state == "queued":
                live.append(f"{region.cfg.title} is waiting for the model…\n", style="dim")
            elif region.live:
                live.append(f"{region.cfg.title} ", style=f"bold {COLOURS[region.key]}")
                live.append(region.live[-420:].replace("\\n", "\n") + "\n")
        self.query_one("#live", Static).update(live or Text("quiet", style="dim"))
        self.query_one("#vitals", Static).update(self.vitals())

    def vitals(self) -> Text:
        brain = self.brain
        text = Text()
        if brain.goal:
            limit = brain.settings.rhythm.max_steps_per_goal
            text.append("goal      ", style="dim")
            text.append(f"{clip(brain.goal.content, 150)} ", style="bold")
            text.append(f"(step {brain.steps}/{limit})\n", style="dim")
        else:
            text.append("goal      none: the mind is wandering\n", style="dim")
        if brain.settings.purpose.strip():
            text.append("long work ", style="dim")
            text.append(f"{clip(brain.settings.purpose, 150)}\n")
        text.append("on mind   ", style="dim")
        text.append(
            (clip(brain.mind[-1]["content"], 150) if brain.mind else "nothing has come to mind by itself yet") + "\n",
            style="dim italic",
        )
        if brain.sleep_until:
            text.append(
                f"sleep     long, until {datetime.fromtimestamp(brain.sleep_until):%H:%M}; school round {brain.night_round}"
                + (", waking after this round" if brain.wake_up else ". Press w to wake it") + "\n",
                style="bold blue",
            )
        if brain.listening:
            text.append("ear       open: it hears you. Press t again when you are done\n", style="bold green")
        elif brain.tools.has("listen") and brain.settings.safety.listen_mode == "always":
            text.append("ear       always listening\n", style="green")
        if brain.full_mode:
            text.append("mode      full: the amygdala is off, nothing is vetoed\n", style="bold red")
        elif not brain.settings.regions["amygdala"].enabled:
            text.append("mode      guarded: the amygdala only comes off inside the container\n", style="dim")
        dopamine = brain.learner.dopamine
        text.append("dopamine  ", style="dim")
        text.append(meter(dopamine, -0.5, 0.5), style="green" if dopamine >= 0 else "red")
        text.append(f" {dopamine:+.2f}  reward vs. expectation\n", style="dim")
        gut = brain.last_thought.data.get("gut", 0.0) if brain.last_thought else 0.0
        text.append("gut       ", style="dim")
        text.append(meter(gut, -0.5, 0.5), style="green" if gut >= 0 else "red")
        text.append(f" {gut:+.2f}  what its latest thought felt like, by pattern\n", style="dim")
        text.append("arousal   ", style="dim")
        text.append(meter(brain.arousal), style="#ffd75f")
        text.append(f" {brain.arousal:.2f}\n", style="dim")
        text.append("fatigue   ", style="dim")
        text.append(meter(brain.fatigue), style="#8787ff")
        text.append(f" {brain.fatigue:.2f}  sleeps at 1.00 · awake {span(time.time() - brain.woke)}\n", style="dim")
        history = brain.memory.reward_history(48)
        count, mean = brain.memory.reward_stats()
        text.append("rewards   ", style="dim")
        text.append(spark(history, -1.0, 1.0) or "none yet", style="#ff5faf")
        text.append(f"  {count} given, mean {mean:+.2f}\n", style="dim")
        uptime = int(time.time() - self.started)
        text.append(
            f"memories {brain.memory.count()} · heuristics {len(brain.learner.heuristics())} · "
            f"day {brain.sleeps + 1} · threats {brain.regions['amygdala'].threats} · "
            f"{brain.llm.speed:.0f} tok/s · {brain.llm.tokens / 1000:.1f}k tokens · "
            f"up {uptime // 3600}:{uptime % 3600 // 60:02d}:{uptime % 60:02d}",
            style="dim",
        )
        if brain.llm.embed_error:
            text.append("\nembedding model unavailable: recall is using a crude word match", style="red")
        share = brain.llm.gpu_share
        if share is not None and share < 0.99:
            text.append(f"\nthe model is only {share:.0%} on the GPU: lower the context window (s)", style="red")
        return text

    # -- actions ----------------------------------------------------------------------

    def action_wake(self) -> None:
        if self.brain.sleep_until:  # in a long sleep: it wakes once a round of training under way is done
            self.brain.wake()
            return
        if any(worker.name == "brain" for worker in self.workers):
            return
        self.dormant = False
        self.run_worker(self.brain.run(), name="brain", exit_on_error=False)

    def action_pause(self) -> None:
        running = self.brain.running
        running.clear() if running.is_set() else running.set()

    def action_sleep(self) -> None:
        """Send it to sleep: one night as usual, or a long sleep of hours, as for a night of your own."""

        def chosen(hours: str | None) -> None:
            if hours is None:
                return
            try:
                self.brain.sleep_for(float(hours.replace(",", ".")) if hours else 0)
            except ValueError:
                self.notify(f"Not a number of hours: {hours}", severity="error")

        self.push_screen(Asleep(), chosen)

    def action_edit(self, key: str | None = None) -> None:
        regions = self.brain.settings.regions
        key = key or next(iter(regions))
        cfg, region = regions[key], self.brain.regions[key]
        fields: list[Field] = [("enabled", cfg, "enabled"), ("rest between firings (seconds)", cfg, "rest")]
        if region.uses_llm:
            fields += [
                ("model: main or fast", cfg, "model"),
                ("temperature", cfg, "temperature"),
                ("max tokens per reply", cfg, "max_tokens"),
                ("prompt: who this region is and what it does", cfg, "prompt"),
            ]
        self.push_screen(
            Form(
                "Edit a part of the brain", fields, self.brain.save_settings,
                chooser=([(r.title, k) for k, r in regions.items()], key), hint=cfg.function,
            ),
            lambda chosen: self.action_edit(chosen) if chosen else None,
        )

    def action_settings(self) -> None:
        s = self.brain.settings
        fields: list[Field] = [
            ("name", s, "name"),
            ("innate drives: the temperament every region shares", s, "drives"),
            ("long work: what it keeps returning to across goals and sleeps (empty for none)", s, "purpose"),
            ("engine: ollama (the main model below) or llama (Gemma 4 on llama.cpp: it sees and hears itself; applies after restart)", s.models, "engine"),
            ("main model", s.models, "main"),
            ("eyes model: describes what the window in front shows, on the CPU (empty: window names only)", s.models, "eyes"),
            ("fast model (critic)", s.models, "fast"),
            ("embedding model (applies after restart)", s.models, "embed"),
            ("context window (tokens)", s.models, "num_ctx"),
            ("parallel model calls (applies after restart)", s.models, "parallel"),
            ("run the embedding model on the CPU", s.models, "embed_on_cpu"),
            ("let the model think before answering (slower)", s.models, "think"),
            ("learning rate", s.learning, "learning_rate"),
            ("discount: how much credit flows back to earlier thoughts", s.learning, "discount"),
            ("exploration: chance of trying a weaker heuristic", s.learning, "explore"),
            ("weight of measured outcomes against the critic's opinion", s.learning, "grounded_weight"),
            ("model calls before sleep", s.rhythm, "sleep_after_firings"),
            ("max steps per goal", s.rhythm, "max_steps_per_goal"),
            ("allow running the code it writes", s.safety, "allow_code_execution"),
            ("code timeout (seconds)", s.safety, "exec_timeout"),
            ("let it open workspace pages in your browser", s.safety, "allow_browser"),
            ("let it see your screen: which windows are open, and what one shows, in words", s.safety, "allow_screen"),
            ("let it look through your camera (one picture a look, put into words; no picture is kept)", s.safety, "allow_camera"),
            ("let it hear you through your microphone", s.safety, "allow_microphone"),
            ("how it hears you talk: push (press t to speak, t again when done) or always", s.safety, "listen_mode"),
            ("give it hands: mouse and keyboard in the programs named below, never in a terminal", s.safety, "allow_hands"),
            ("programs its hands may act in", s.safety, "hands_in"),
            ("speak aloud in its own voice", s.voice, "enabled"),
            ("speak to you without being asked", s.voice, "spontaneous"),
            ("seconds it keeps quiet between two things said unasked", s.voice, "quiet_seconds"),
            ("voice: a neural one such as de_DE-thorsten-medium, or one installed in Windows", s.voice, "name"),
        ]
        self.push_screen(Form("Settings", fields, self.brain.save_settings))

    def action_heuristics(self) -> None:
        self.push_screen(Heuristics(self.brain))

    def action_mind(self) -> None:
        self.push_screen(Mind(self.brain))

    def action_wiring(self) -> None:
        self.push_screen(Wiring(self.brain))

    def action_whisper(self) -> None:
        self.push_screen(Whisper(), lambda text: self.brain.inject(text) if text else None)

    async def action_speak_aloud(self) -> None:
        """Push to talk: the first press opens its ear on your microphone, the next closes it, and
        it hears what you said in between."""
        if self.brain.settings.safety.listen_mode == "always" and self.brain.tools.has("listen"):
            self.notify("It is always listening: just speak.")
            return
        listening = await self.brain.press_to_talk()
        if listening:
            self.notify("Listening. Speak, then press t again.")
        elif self.brain.tools.has("listen"):
            self.notify("Heard. It is answering.")

    def action_praise(self) -> None:
        self.judge(1.0)

    def action_scold(self) -> None:
        self.judge(-1.0)

    def judge(self, reward: float) -> None:
        """Reward or punish what it did last: its latest thought or step, or its latest answer to
        you, whichever came later."""
        brain = self.brain
        target = max(filter(None, (brain.last_thought, brain.last_said)), key=lambda s: s.ts, default=None)
        if target is None:
            self.notify("Nothing to judge yet.")
            return
        brain.apply_reward(target.id, reward, "human", "Judged by hand.")
        what = "its answer to you" if target.kind == "exchange" else f"its {target.kind}"
        said = target.content.split("I answered: ", 1)[-1] if target.kind == "exchange" else target.content
        self.notify(f"{'+' if reward > 0 else '-'} for {what}: {said[:90]}")
