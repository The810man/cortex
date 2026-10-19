"""Building a model out of parts: what `build` does before and after FreeCAD is asked.

A small model writes FreeCAD's Python from memory, and its memory of it is poor: of this brain's
first 161 runs in FreeCAD, 142 failed, most of them on a part it looked for in an earlier model
and did not find, or on a file from an earlier step that would not open. So a model is not written as a script here. It is described as
parts, each a simple shape attached to a side of a part that is already there, and the machine
does the rest: works out where every part lies, writes the script, and after FreeCAD has run it
says what was built in words and in a picture drawn in characters. The description is kept, so
the next step adds to the model instead of starting again.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SIDES = {"top": (2, 1), "bottom": (2, -1), "left": (0, -1), "right": (0, 1), "front": (1, -1), "back": (1, 1)}
AXES = {"x": 0, "y": 1, "z": 2}
SINK = 0.15  # how much of its own length a part reaches into the part it is attached to, so the two fuse
KEPT = ".parts.json"  # what a model's description is kept as, beside the model
WIDE, TALL = 44, 18  # the picture in characters, at most
HOW = (
    "`build` makes a model out of parts and is the surest way to build anything. It is a tool of "
    "its own that the motor cortex calls directly, as it calls `write_file`: it is not a function "
    "that a script can use, and no script and no file is written for it. A step for it reads "
    "like this: 'With `build`, add to the model body a head: a sphere of radius 90 on top of "
    "the torso.' `path` is the "
    "model's name (for example body) and `parts` the parts to add to it or change in it. Each part "
    "has a `name`, a `shape` (box, cylinder, sphere or cone), a `size` in millimetres (box: [width "
    "along x, depth along y, height along z]; cylinder: [radius, length]; sphere: [radius]; cone: "
    "[radius at its start, radius at its end, length]), the `axis` a cylinder or cone runs along, "
    "the part it is attached to (`on`, empty for the first part, which stands on the ground) and "
    "the `side` of that part it sits on: top, bottom, left (toward -x), right (+x), front (-y) or "
    "back (+y). It is placed in the middle of that side, reaching a little into the part so the "
    "two become one solid. `shift` [x, y, z] slides it along that side from the middle, for "
    "example an arm on the left side moved up by [0, 0, 120]; it is not a position, it is "
    "usually [], and it cannot take a part off the part it sits on. A part given again under its "
    "name replaces the old one, and shape `remove` takes it out; the parts you do not mention "
    "stay as they are. One `build` may add several parts. Nothing has to be computed, saved or "
    "checked by you: what comes back says where every part lies, whether they are one valid "
    "solid, which part does not touch and by how much, and shows the model from the front."
)


# For whoever puts a goal into words. Told only that `build` exists, a small model that has written
# scripts all its life sets the goal "write a script that uses build", and then writes that script.
GOAL = (
    "A goal that builds something in FreeCAD is set like this: 'With `build`, make the model "
    "body part by part: first a torso, then a head on top of it, then an arm on each side, and "
    "`show` body.FCStd when it is one solid.' `build` is a tool that is called directly, one "
    "call for a part or for several: never 'write a script that uses build', and no "
    "`run_freecad` afterwards to verify, because `build` reports FreeCAD's own measurements."
)


class Part(BaseModel):
    """One part of a model. Every field is written out, so none is decided by being left out."""

    model_config = ConfigDict(json_schema_extra=lambda schema: schema.update(required=list(schema["properties"])))

    name: str = Field(description="what the part is, in a word or two: torso, head, left_arm")
    shape: Literal["box", "cylinder", "sphere", "cone", "remove"]
    size: list[float] = Field(
        description="millimetres. box: [width x, depth y, height z]; cylinder: [radius, length]; sphere: "
        "[radius]; cone: [radius at its start, radius at its end, length]",
        json_schema_extra={"maxItems": 3},
    )
    axis: Literal["x", "y", "z"] = Field("z", description="the direction a cylinder or cone runs in")
    on: str = Field("", description="the name of the part this one is attached to; empty for the first part")
    side: Literal["top", "bottom", "left", "right", "front", "back"] = Field(
        "top", description="the side of that part it sits on"
    )
    shift: list[float] = Field(
        default_factory=list, description="[x, y, z] millimetres to slide it along that side from the middle: not a "
        "position; [] for none, which is usual",
        json_schema_extra={"maxItems": 3},
    )
    cut: bool = Field(
        False, description="false for a part of the body, which is nearly always; true only to drill this shape "
        "out of the model as a hole"
    )


class Placed:
    """A part where it lies: its middle and half its size along x, y and z."""

    def __init__(self, part: Part, half: tuple[float, float, float]) -> None:
        self.part, self.half = part, half
        self.middle = [0.0, 0.0, 0.0]
        self.sunk = 0.0
        self.on = ""  # the part it was placed against, by that part's own name; empty on the ground
        self.side = part.side  # and the side of that part it was put on
        self.cut_back: list[str] = []  # where its shift was cut back so that it stays on that part

    def low(self, axis: int) -> float:
        return self.middle[axis] - self.half[axis]

    def high(self, axis: int) -> float:
        return self.middle[axis] + self.half[axis]


GROUND = {"", "ground", "floor", "none", "null", "nothing", "origin", "world", "base_plane"}


def key(name: str) -> str:
    """A part's name as it is compared: `Left Arm`, `left-arm` and `left_arm` are one part."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def host_of(part: Part, names: tuple[str, ...] = ()) -> str:
    """The part another one is attached to, as it is compared; empty for one that stands on the
    ground, however the ground was put. Among `names`, the parts there are, a part is found by a
    name that only holds its name: told seven times that there is no part `top_of_torso`, only
    `torso`, a small model wrote `top_of_torso` an eighth time."""
    host = key(part.on)
    if host in GROUND or host in names:
        return "" if host in GROUND else host
    meant = [name for name in names if name != key(part.name) and (name in host or host in name)]
    return max(meant, key=len) if meant else host


def half_size(part: Part) -> tuple[float, float, float]:
    """Half of a part's size along x, y and z. Raises ValueError when the sizes make no shape."""
    size = [abs(float(v)) for v in part.size]
    need = {"box": 3, "cylinder": 2, "sphere": 1, "cone": 3}[part.shape]
    given = size[:need]
    sound = len(given) == need and given[-1] > 0 and (
        max(given[:2]) > 0 and given[0] != given[1] if part.shape == "cone" else min(given) > 0
    )
    if not sound:
        raise ValueError(
            f"{part.name}: a {part.shape} needs `size` "
            + {"box": "[width, depth, height]", "cylinder": "[radius, length]", "sphere": "[radius]",
               "cone": "[radius at its start, radius at its end, length]"}[part.shape]
            + f" in millimetres, above 0{' and the two radii different' * (part.shape == 'cone')}; "
            f"it was given {part.size}"
        )
    if part.shape == "box":
        return size[0] / 2, size[1] / 2, size[2] / 2
    if part.shape == "sphere":
        return size[0], size[0], size[0]
    radius, length = (size[0], size[1]) if part.shape == "cylinder" else (max(size[0], size[1]), size[2])
    half = [radius, radius, radius]
    half[AXES[part.axis]] = length / 2
    return half[0], half[1], half[2]


def merge(kept: list[Part], new: list[Part]) -> list[Part]:
    """The model's parts with the new ones taken in: one under a name already there replaces it,
    `remove` takes it out, and the rest stay as they were."""
    parts = {key(part.name): part for part in kept}
    for part in new:
        if not key(part.name):
            raise ValueError("every part needs a name")
        if part.shape == "remove":
            if parts.pop(key(part.name), None) is None:
                raise ValueError(f"there is no part called {part.name} to remove; the parts are: {', '.join(parts) or 'none'}")
        else:
            parts[key(part.name)] = part
    return list(parts.values())


def place(parts: list[Part]) -> list[Placed]:
    """Where every part lies. The first stands on the ground; each other one sits in the middle of
    a side of the part it names, reaching a little into it, moved by its `shift`.

    A part that is attached stays attached. A small model gives `shift` as if it were a position
    (a head "at height 300" on a torso 400 high), and the part then hangs in the air beside the
    part it was put on: so a shift is cut back to where the two still reach into each other."""
    placed: dict[str, Placed] = {}
    waiting = [(part, half_size(part)) for part in parts]
    names = tuple(key(part.name) for part in parts)
    while waiting:
        later = []
        for part, half in waiting:
            host, side = host_of(part, names), part.side
            if host in SIDES and host not in names and names[0] != key(part.name):
                host, side = names[0], host  # `on: top` for "on top": on top of the first part, then
            elif host in names and key(part.on) != host:  # `on: left of torso`: the side was said there
                side = next((word for word in key(part.on).split("_") if word in SIDES), side)
            shift = [float(v) for v in part.shift[:3]] + [0.0] * (3 - len(part.shift[:3]))
            lying = Placed(part, half)
            if not host:
                lying.middle = [shift[0], shift[1], half[2] + shift[2]]
            elif host == key(part.name) or (host not in placed and host not in {key(p.name) for p, _ in waiting}):
                others = ", ".join(p.name for p in parts if p is not part)
                raise ValueError(
                    f"{part.name} is attached to `{part.on}`, and there is no such part. "
                    + (f"The parts are: {others}. Give `on` one of those names, or" if others else "There are no "
                       "other parts yet:")
                    + " leave `on` empty to stand the part on the ground."
                )
            elif host not in placed:
                later.append((part, half))
                continue
            else:
                under = placed[host]
                lying.on, lying.side = under.part.name, side
                axis, sign = SIDES[side]
                lying.sunk = min(max(1.0, SINK * 2 * half[axis]), 2 * half[axis], 2 * under.half[axis])
                lying.middle = list(under.middle)
                lying.middle[axis] += sign * (under.half[axis] + half[axis] - lying.sunk)
                for across in range(3):
                    if across == axis:  # away from the part it sits on it cannot go; further in, it can
                        held = shift[across] if sign * shift[across] <= 0 else 0.0
                    else:  # along the side, as far as the two still reach into each other
                        reach = max(under.half[across] + half[across] - max(1.0, SINK * 2 * half[across]), 0.0)
                        held = min(max(shift[across], -reach), reach)
                    if held != shift[across]:
                        lying.cut_back.append(f"{'xyz'[across]} from {shift[across]:g} to {held:g}")
                    lying.middle[across] += held
            placed[key(part.name)] = lying
        if len(later) == len(waiting):
            raise ValueError("these parts are attached to one another in a circle: " + ", ".join(p.name for p, _ in later))
        waiting = later
    return [placed[key(part.name)] for part in parts]


def gap(one: Placed, other: Placed) -> tuple[float, int, float]:
    """How far two parts are apart, going by the boxes around them: the largest gap along any
    axis (negative when they reach into each other), that axis, and which way `one` has to move."""
    worst, along, way = -float("inf"), 0, 1.0
    for axis in range(3):
        apart = max(one.low(axis) - other.high(axis), other.low(axis) - one.high(axis))
        if apart > worst:
            worst, along = apart, axis
            way = -1.0 if one.middle[axis] > other.middle[axis] else 1.0
    return worst, along, way


def loose(placed: list[Placed]) -> list[tuple[Placed, Placed, float, int, float]]:
    """The parts that are joined to nothing the first part is joined to, each with its nearest
    joined part, the gap between them, the axis of that gap and the way to move to close it."""
    solid = [p for p in placed if not p.part.cut]
    if not solid:
        return []
    joined, apart = [solid[0]], solid[1:]
    grew = True
    while grew:
        grew = False
        for part in apart[:]:
            if any(gap(part, other)[0] < -0.5 for other in joined):
                joined.append(part)
                apart.remove(part)
                grew = True
    found = []
    for part in apart:  # measured against the part it was meant to sit on, where that one is joined
        meant = [other for other in joined if other.part.name == part.on] or joined
        found.append((part, *min(((other, *gap(part, other)) for other in meant), key=lambda near: near[1])))
    return found


def covers(lying: Placed, x: float, z: float) -> bool:
    """Whether a part, seen from the front, covers the point (x, z)."""
    dx, dz = x - lying.middle[0], z - lying.middle[2]
    hx, hz = lying.half[0], lying.half[2]
    if abs(dx) > hx or abs(dz) > hz:
        return False
    shape, axis = lying.part.shape, lying.part.axis
    if shape == "sphere" or (shape in ("cylinder", "cone") and axis == "y"):
        return (dx / hx) ** 2 + (dz / hz) ** 2 <= 1.0
    if shape == "cone":
        start, end = abs(lying.part.size[0]), abs(lying.part.size[1])
        along, across, half = (dz, dx, hz) if axis == "z" else (dx, dz, hx)
        return abs(across) <= start + (end - start) * (along + half) / (2 * half)
    return True


def picture(placed: list[Placed]) -> str:
    """The model seen from the front, drawn in characters: each part by its letter. A mind
    without eyes cannot be shown a picture, but it can read one."""
    solid = [p for p in placed if not p.part.cut]
    if not solid:
        return ""
    left, right = min(p.low(0) for p in solid), max(p.high(0) for p in solid)
    low, high = min(p.low(2) for p in solid), max(p.high(2) for p in solid)
    wide = max((right - left) / WIDE, (high - low) / (2 * TALL), 0.01)  # a character is twice as tall as wide
    for round_to in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 500, 1000):
        if round_to >= wide:
            wide = float(round_to)
            break
    tall = 2 * wide
    letters = {id(p): chr(ord("A") + i % 26) for i, p in enumerate(placed)}
    inside = [(dx / 4, dz / 8) for dx in (1, 3) for dz in (1, 3, 5, 7)]  # a thin part must not fall between two points
    rows = []
    for row in range(int((high - low) / tall - 1e-9), -1, -1):
        line = ""
        for column in range(int((right - left) / wide - 1e-9) + 1):
            x, z = left + column * wide, low + row * tall
            on = [p for p in solid if any(covers(p, x + dx * wide, z + dz * tall) for dx, dz in inside)]
            line += letters[id(on[-1])] if on else " "
        rows.append(line.rstrip())
    legend = ", ".join(f"{letters[id(p)]} {p.part.name}" for p in solid)
    return (
        f"Seen from the front (x to the right, z up; one character is {wide:g} mm wide and {tall:g} mm tall):\n"
        + "\n".join(rows) + f"\n{legend}"
    )


def describe(placed: list[Placed]) -> str:
    """Every part as it was placed: what it is, where its middle is, and what it sits on."""
    lines = []
    for lying in placed:
        part, (hx, hy, hz) = lying.part, lying.half
        x, y, z = lying.middle
        if lying.on:
            where = f"on the {lying.side} of {lying.on}, reaching {lying.sunk:.0f} mm into it"
            if key(part.on) != key(lying.on):
                where += f" (`on` said {part.on}: taken to mean {lying.on})"
            if any(part.shift):
                where += f", shifted by {[round(float(v), 1) for v in part.shift]}"
            if lying.cut_back:
                where += (
                    f" (its shift was cut back along {' and '.join(lying.cut_back)}: a part attached to "
                    f"{lying.on} stays touching it, and `shift` only moves it along that side)"
                )
        else:
            where = "standing on the ground"
        round_one = {"sphere": f" of radius {hx:.0f}", "cylinder": f" of radius {abs(part.size[0]):.0f}"}.get(part.shape, "")
        lines.append(
            f"- {part.name}: {'a hole, ' if part.cut else ''}{part.shape}{round_one} {2 * hx:.0f} × {2 * hy:.0f} × "
            f"{2 * hz:.0f} mm, middle at ({x:.0f}, {y:.0f}, {z:.0f}), {where}"
        )
    return "The parts, as placed (each with the size of the box around it, along x × y × z):\n" + "\n".join(lines)


def advice(placed: list[Placed], solids: int | None) -> str:
    """Which parts are not joined to the rest, and how to join them. Empty when all of them are."""
    apart = loose(placed)
    if apart:
        lines = []
        for part, nearest, by, axis, way in apart:
            if not part.on:  # it stands on the ground by itself: what it needs is a part to sit on
                lines.append(
                    f"- {part.part.name} does NOT touch the rest: it stands on the ground by itself, "
                    f"{max(by, 0.0):.0f} mm from {nearest.part.name} along {'xyz'[axis]}. Give it again with `on` "
                    f"naming the part it belongs on, for example {nearest.part.name}, and the `side` of that part."
                )
                continue
            move = [0.0, 0.0, 0.0]
            move[axis] = round(way * (max(by, 0.0) + max(1.0, SINK * 2 * part.half[axis])), 1)
            old = [float(v) for v in part.part.shift[:3]] + [0.0] * (3 - len(part.part.shift[:3]))
            lines.append(
                f"- {part.part.name} does NOT touch the rest: the nearest part is {nearest.part.name}, "
                f"{max(by, 0.0):.0f} mm away along {'xyz'[axis]}. Give it again with `shift` "
                f"{[round(o + m, 1) for o, m in zip(old, move, strict=True)]}, or attach it to another side."
            )
        return "Not one body yet:\n" + "\n".join(lines)
    if solids is not None and solids > 1:
        return (
            f"The boxes around the parts all reach into one another, but FreeCAD finds {solids} separate "
            "solids: somewhere two round parts only come near at a corner. Move those further into each other."
        )
    return ""


def script(name: str, placed: list[Placed]) -> str:
    """The FreeCAD script that builds the model: every part made where it lies, the solid ones
    fused in turn, the holes taken out, and the result put into a document as one object."""
    lines = [
        "# Written by `build` from " + name + KEPT + ": change the model with `build`, not here.",
        "import FreeCAD as App, Part",
        "V = App.Vector",
        f"doc = App.newDocument({name!r})",
    ]
    made: list[tuple[str, bool]] = []
    for index, lying in enumerate(placed):
        part, (hx, hy, hz), (x, y, z) = lying.part, lying.half, lying.middle
        var = f"p{index}_{re.sub(r'[^a-z0-9]+', '_', part.name.lower()).strip('_') or 'part'}"
        if part.shape == "box":
            make = f"Part.makeBox({2 * hx:.3f}, {2 * hy:.3f}, {2 * hz:.3f}, V({x - hx:.3f}, {y - hy:.3f}, {z - hz:.3f}))"
        elif part.shape == "sphere":
            make = f"Part.makeSphere({hx:.3f}, V({x:.3f}, {y:.3f}, {z:.3f}))"
        else:
            axis = AXES[part.axis]
            base, way = [x, y, z], [0, 0, 0]
            base[axis] -= lying.half[axis]
            way[axis] = 1
            start, along = f"V({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})", f"V({way[0]}, {way[1]}, {way[2]})"
            if part.shape == "cylinder":
                make = f"Part.makeCylinder({abs(part.size[0]):.3f}, {2 * lying.half[axis]:.3f}, {start}, {along})"
            else:
                make = (
                    f"Part.makeCone({abs(part.size[0]):.3f}, {abs(part.size[1]):.3f}, "
                    f"{2 * lying.half[axis]:.3f}, {start}, {along})"
                )
        lines.append(f"{var} = {make}")
        made.append((var, part.cut))
    solid = [var for var, cut in made if not cut]
    if solid:
        lines.append(f"whole = {solid[0]}")
        lines += [f"whole = whole.fuse({var})" for var in solid[1:]]
        lines += [f"whole = whole.cut({var})" for var, cut in made if cut]
        lines += [
            "try:",
            "    whole = whole.removeSplitter()",
            "except Exception:",
            "    pass",
            f"obj = doc.addObject('Part::Feature', {name!r})",
            "obj.Shape = whole",
        ]
    return "\n".join(lines) + "\n"


def load(text: str) -> list[Part]:
    """The parts a model was last built from, out of its kept description. Raises ValueError for
    a file `build` did not write: a small model, told that descriptions are kept as JSON, will
    write one by hand."""
    kept = json.loads(text)
    if not isinstance(kept, dict) or kept.get("written_by") != "build":
        raise ValueError("not a model description that `build` wrote")
    return [Part.model_validate(part) for part in kept["parts"]]


def dump(parts: list[Part], verdict: str = "") -> str:
    """A model's description as it is kept: its parts, and what FreeCAD last made of them."""
    kept = {"written_by": "build", "verdict": verdict, "parts": [part.model_dump() for part in parts]}
    return json.dumps(kept, indent=1) + "\n"


def verdict(text: str) -> str:
    """What FreeCAD made of a model when it was last built, from its kept description."""
    try:
        return str(json.loads(text).get("verdict", ""))
    except (ValueError, AttributeError):
        return ""


def summary(name: str, parts: list[Part]) -> str:
    """A model as it stands, for a mind about to take its next step on it."""
    try:
        placed = place(parts)
    except ValueError as exc:
        return f"The model `{name}` cannot be placed: {exc}"
    return "\n".join(filter(None, [
        f"The model `{name}` so far, {len(parts)} part{'s' * (len(parts) != 1)}:", describe(placed), picture(placed),
        advice(placed, None),
        f"To add the next part to this model, `build` with `path` {name} and the new part alone, attached to one of "
        "the parts above: under another name a new model is begun.",
    ]))


# What a goal wants built, found out once by a tiny call of its own, like the thalamus's gate: asked
# while doing anything else, a small model does not keep track of which parts a body still lacks.
WANTS = (
    "You read a goal or a long work of a mind that builds models in FreeCAD out of parts, and list "
    "the parts it wants the model to have: each by one English word in the singular (torso, head, "
    "arm, leg, wheel, ...) with how many. `parts` is empty when the goal wants nothing built or "
    "names no parts. Answer with one JSON object and nothing else."
)


class Wanted(BaseModel):
    model_config = ConfigDict(json_schema_extra=lambda schema: schema.update(required=list(schema["properties"])))

    name: str = Field(description="the part, one English word in the singular")
    count: int = Field(ge=1, le=8, description="how many of it")


class Wants(BaseModel):
    model_config = ConfigDict(json_schema_extra=lambda schema: schema.update(required=list(schema["properties"])))

    parts: list[Wanted] = Field(json_schema_extra={"maxItems": 10})


def tally(wanted: list[tuple[str, int]], parts: list[Part]) -> tuple[list[str], list[str]]:
    """Which of the wanted parts the model has and which it still lacks, by name and forgivingly:
    "arm" twice is met by left_arm and right_arm."""
    names = [key(part.name) for part in parts if not part.cut]
    has, lacks = [], []
    for name, count in wanted:
        word = key(name)
        stem = word[:-1] if word.endswith("s") and len(word) > 3 else word
        found = [n for n in names if stem and stem in n]
        label = name if count == 1 else f"{count} × {name}"
        (has if len(found) >= count else lacks).append(
            label if len(found) >= count or not found else f"{label} (has {len(found)})"
        )
    return has, lacks


def standing(wanted: list[tuple[str, int]], name: str, parts: list[Part]) -> str:
    """How a model stands against what is wanted of it, in a line."""
    has, lacks = tally(wanted, parts)
    return (
        f"Wanted of the model `{name}`: {', '.join(f'{c} × {n}' if c > 1 else n for n, c in wanted)}. "
        f"It has: {', '.join(has) or 'none of them'}. "
        + (f"Still missing: {', '.join(lacks)}." if lacks else "Nothing is missing.")
    )
