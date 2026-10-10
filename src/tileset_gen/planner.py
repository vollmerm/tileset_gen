"""LLM planner: turns a free-text theme into a TilesetSpec using Claude Haiku.

The model only chooses palette colors, generators and parameters; all pixels
come from the deterministic generators. Structured outputs can't express
free-form dicts, so the model fills a "wire" schema where params are
"name=value" strings, which is converted to a TilesetSpec and then checked
against the generator registry. Validation errors are fed back once for a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from . import registry
from .spec import (
    AnimationSpec,
    CharacterSpec,
    NamedColor,
    ObjectSpec,
    TerrainSpec,
    TilesetSpec,
    TransitionSpec,
)
from .validate import validate_spec

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000


class PlannerError(RuntimeError):
    pass


@dataclass
class PlanStats:
    """Filled in by `plan()` when passed: one entry of validation errors per attempt, plus token usage."""

    attempts: list[list[str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def param_str(name: str, value) -> str:
    """One wire param, "name=value"; bools are written true/false."""
    return f"{name}={str(value).lower() if isinstance(value, bool) else value}"


def parse_params(items: list[str]) -> dict:
    """Wire params back to a dict typed the way params models expect: "true" -> True, "3" -> 3, "0.4" -> 0.4."""
    out: dict = {}
    for item in items:
        name, _, raw = item.partition("=")
        v = raw.strip()
        if v.lower() in ("true", "false"):
            out[name.strip()] = v.lower() == "true"
            continue
        for cast in (int, float):
            try:
                n = cast(v)
            except ValueError:
                continue
            out[name.strip()] = int(n) if isinstance(n, float) and n.is_integer() else n
            break
        else:
            out[name.strip()] = v
    return out


class WireTerrain(BaseModel):
    name: str
    generator: str
    colors: list[str]
    variants: int
    decor: int
    patches: float
    params: list[str]  # "name=value"


class WireTransition(BaseModel):
    lower: str
    upper: str
    edge_style: Literal["soft", "shore", "cliff", "blend"]
    roughness: float


class WireObject(BaseModel):
    name: str
    generator: str
    colors: list[str]
    width: int
    height: int
    params: list[str]  # "name=value"
    place_on: list[str]
    density: float


class WireAnimation(BaseModel):
    name: str
    terrain: str  # "" when unused
    generator: str  # "" when unused
    colors: list[str]
    frames: int
    frame_ms: int
    params: list[str]  # "name=value"


class WireCharacter(BaseModel):
    name: str
    generator: str
    colors: list[str]
    params: list[str]  # "name=value"
    place_on: list[str]
    count: int


class WireSpec(BaseModel):
    name: str
    theme: str
    palette: list[NamedColor]
    terrains: list[WireTerrain]
    transitions: list[WireTransition]
    objects: list[WireObject]
    animations: list[WireAnimation]
    characters: list[WireCharacter]

    @classmethod
    def from_spec(cls, spec: TilesetSpec) -> WireSpec:
        def kvs(params: dict) -> list[str]:
            return [param_str(k, v) for k, v in params.items()]

        def anim(a: AnimationSpec) -> dict:
            return {
                **a.model_dump(),
                "terrain": a.terrain or "",
                "generator": a.generator or "",
                "params": kvs(a.params),
            }

        return cls.model_validate(
            {
                "name": spec.name,
                "theme": spec.theme,
                "palette": [c.model_dump() for c in spec.palette],
                "terrains": [{**t.model_dump(), "params": kvs(t.params)} for t in spec.terrains],
                "transitions": [t.model_dump() for t in spec.transitions],
                "objects": [{**o.model_dump(), "params": kvs(o.params)} for o in spec.objects],
                "animations": [anim(a) for a in spec.animations],
                "characters": [{**c.model_dump(), "params": kvs(c.params)} for c in spec.characters],
            }
        )

    def to_spec(self, seed: int, tile_size: int, max_colors: int) -> TilesetSpec:
        return TilesetSpec(
            name=self.name,
            theme=self.theme,
            seed=seed,
            tile_size=tile_size,
            max_colors=max_colors,
            palette=self.palette,
            terrains=[
                TerrainSpec(**t.model_dump(exclude={"params"}), params=parse_params(t.params)) for t in self.terrains
            ],
            transitions=[TransitionSpec(**t.model_dump()) for t in self.transitions],
            objects=[
                ObjectSpec(**o.model_dump(exclude={"params"}), params=parse_params(o.params)) for o in self.objects
            ],
            animations=[
                AnimationSpec(
                    **a.model_dump(exclude={"params", "terrain", "generator"}),
                    terrain=a.terrain or None,
                    generator=a.generator or None,
                    params=parse_params(a.params),
                )
                for a in self.animations
            ],
            characters=[
                CharacterSpec(**c.model_dump(exclude={"params"}), params=parse_params(c.params))
                for c in self.characters
            ],
        )


SYSTEM_TEMPLATE = """\
You are the art director for a procedural pixel-art tileset generator in the style of 8-bit and 16-bit JRPGs \
(Final Fantasy IV-VI, Chrono Trigger, Dragon Quest, Earthbound). Given a theme, you plan a tileset spec. \
You never draw pixels; you choose colors, generators and parameters, and deterministic code renders them.

# Art direction
- Pick a cohesive palette of 6-14 named base colors. The renderer derives 5-step hue-shifted shading ramps \
from each base color, so choose mid-tone, moderately saturated bases (avoid near-black and near-white bases \
except for snow).
- Terrains should contrast in value as well as hue so autotile edges read clearly at 16x16 pixels.
- Favor classic, readable JRPG staples for the theme: 3-5 terrains, 5-10 objects, 1-3 animations.

# Spec rules
- palette: every `hex` is exactly 7 characters, "#" plus 6 hex digits (e.g. "#5aa83c").
- terrains: each needs a unique name, a terrain generator, colors (palette names or #rrggbb, in the \
generator's color-slot order; omitted slots use defaults), variants, decor, and params. Variants are \
seamless plain versions of the terrain: use 3 for natural ground (grass, dirt, sand, snow, cobble), 2 for \
water and lava, and 1-2 for regular floors. decor (0-6) adds rare detail tiles drawn from the generator's \
feature set (tall grass, flowers, stones, puddles, cracks, lily pads...) that break up large areas: \
use 3-4 for open ground, 2 for water, lava and floors, 0 only for a deliberately uniform surface. \
patches (0-0.6) covers that share of the terrain's open areas with darker multi-tile patches of the same \
ground (lush grass, damp earth, deep water) that blend in without an outline: 0.25-0.4 for natural ground \
and water, 0-0.15 for built floors.
- transitions: blend `upper` over `lower` with corner autotiles; at most one transition per terrain pair. \
Connect every terrain: transitions should form a chain ordered like elevation \
(e.g. water -> sand -> grass -> dirt), optionally with side branches (grass -> flower_meadow). \
A terrain with no transition can only appear as hard-edged rectangles, so only leave one unconnected \
when that is intended (a carpet, a plaza). \
edge_style: "shore" for land over water, "cliff" for raised ground (mesas, plateaus, floors over lava pits), \
"blend" only between two tones of the same kind of ground (light and dark grass; never grass to dirt), \
"soft" otherwise. roughness 0-1 (0.3-0.6 typical).
- objects: object generator, size in tiles (width/height 1-4; trees 1x1 or 2x2, houses 2x2 or 3x3, \
pine 1x2, wells 1x2 or 2x2), place_on lists terrain names it may stand on (never water/lava). \
density (0-1) controls how often it appears in the preview map: 0.5-0.8 for common scenery \
(trees, bushes, rocks, flowers), 0.3-0.5 for props (barrels, signs), 0.2-0.4 for landmarks \
(houses, chests). [connected] generators such as fence link up tile by tile: always 1x1, and the preview \
draws fenced enclosures with a gate (density 0.3-0.7 gives 1-2 pens), so give them place_on terrains with \
room, like grass or dirt. Values below 0.1 make the object nearly invisible.
- animations: either set `terrain` to an animatable terrain's name (generator "") to animate that terrain, \
or set `generator` to an animation generator (terrain ""). frames 2-8, frame_ms 80-400.
- characters: 3-5 walking sprites that live in this world, such as a hero, townsfolk or guards suited to the \
theme, and one or two monsters. Dress them in the tileset's own palette names (a robe in the roof color, armor \
in the stone color) so they match it; add skin and hair base colors to the palette or give them as hex. \
Give every character a different silhouette: vary outfit, hair, headgear and cape. Keep outfits in value \
contrast with the ground they stand on. place_on lists walkable terrains (never water/lava); count (0-4) \
is how many copies stand in the preview map.
- params: list of "name=value" strings (e.g. "density=0.4", "clumps=3", "cape=true", "outfit=robe") \
using only the params listed for that generator, within their ranges \
(for a|b|c params use one of the listed values). Use an empty list to accept the defaults.
- Names are lowercase snake_case and unique across terrains, objects, animations and characters.

# Generators
{catalog}"""


def system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(catalog=registry.describe_for_prompt())


def _assistant_content(response) -> list[dict]:
    """The response's content blocks as request params (append-only history for the retry turn)."""
    blocks = []
    for block in response.content:
        d = block.to_dict() if hasattr(block, "to_dict") else dict(block)
        d.pop("parsed_output", None)
        blocks.append(d)
    return blocks


def _run_planner(
    messages: list[dict],
    *,
    seed: int,
    tile_size: int,
    max_colors: int,
    client: anthropic.Anthropic | None,
    retries: int,
    stats: PlanStats | None,
) -> TilesetSpec:
    """Call the model, validate, and feed errors back up to `retries` times."""
    client = client or anthropic.Anthropic()
    stats = stats if stats is not None else PlanStats()
    errors: list[str] = []
    for attempt in range(retries + 1):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt(),
            messages=messages,
            output_format=WireSpec,
            output_config={"effort": "low"},
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            stats.input_tokens += usage.input_tokens
            stats.output_tokens += usage.output_tokens
        if response.stop_reason == "refusal":
            raise PlannerError("the model declined to plan this tileset")
        if response.stop_reason == "max_tokens":
            raise PlannerError("the planner response was cut off (max_tokens)")
        wire = response.parsed_output
        try:
            spec = wire.to_spec(seed, tile_size, max_colors)
            errors = validate_spec(spec)
        except ValidationError as e:
            errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        stats.attempts.append(errors)
        if not errors:
            return spec
        if attempt < retries:
            messages.append({"role": "assistant", "content": _assistant_content(response)})
            messages.append(
                {
                    "role": "user",
                    "content": "That spec has problems:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\nReturn the full corrected spec.",
                }
            )
    raise PlannerError("planner produced an invalid spec:\n" + "\n".join(f"- {e}" for e in errors))


def plan(
    prompt: str,
    *,
    seed: int = 0,
    tile_size: int = 16,
    max_colors: int = 48,
    client: anthropic.Anthropic | None = None,
    retries: int = 1,
    stats: PlanStats | None = None,
) -> TilesetSpec:
    """Plan a new spec from a theme."""
    messages: list[dict] = [{"role": "user", "content": f"Plan a tileset for this theme: {prompt}"}]
    return _run_planner(
        messages, seed=seed, tile_size=tile_size, max_colors=max_colors, client=client, retries=retries, stats=stats
    )


REFINE_TEMPLATE = """\
Here is the current tileset spec:

{spec}

Apply this change: {instruction}

Keep everything the change doesn't touch exactly as it is: same names, colors, generators and params. \
Return the full updated spec."""


def refine(
    spec: TilesetSpec,
    instruction: str,
    *,
    client: anthropic.Anthropic | None = None,
    retries: int = 1,
    stats: PlanStats | None = None,
) -> TilesetSpec:
    """Edit an existing spec according to a natural-language instruction.

    Seed, tile size and color limit are carried over; the model never sees them.
    """
    wire = WireSpec.from_spec(spec).model_dump_json(indent=1)
    messages: list[dict] = [{"role": "user", "content": REFINE_TEMPLATE.format(spec=wire, instruction=instruction)}]
    new = _run_planner(
        messages,
        seed=spec.seed,
        tile_size=spec.tile_size,
        max_colors=spec.max_colors,
        client=client,
        retries=retries,
        stats=stats,
    )
    # The wire schema leaves out per-character palette size and timing; keep what the old spec had.
    old = {c.name: c for c in spec.characters}
    for c in new.characters:
        if c.name in old:
            c.max_colors, c.frame_ms = old[c.name].max_colors, old[c.name].frame_ms
    return new


def spec_diff(old: TilesetSpec, new: TilesetSpec) -> list[str]:
    """Human-readable summary of what changed between two specs, by element name."""
    lines: list[str] = []
    if old.theme != new.theme:
        lines.append(f"~ theme: {new.theme!r}")
    sections = [
        ("palette", lambda s: {c.name: c for c in s.palette}),
        ("terrain", lambda s: {t.name: t for t in s.terrains}),
        ("transition", lambda s: {f"{t.lower}->{t.upper}": t for t in s.transitions}),
        ("object", lambda s: {o.name: o for o in s.objects}),
        ("animation", lambda s: {a.name: a for a in s.animations}),
        ("character", lambda s: {c.name: c for c in s.characters}),
    ]
    for label, index in sections:
        a, b = index(old), index(new)
        for name in b.keys() - a.keys():
            lines.append(f"+ {label} {name}")
        for name in a.keys() - b.keys():
            lines.append(f"- {label} {name}")
        for name in a.keys() & b.keys():
            if a[name] != b[name]:
                before, after = a[name].model_dump(), b[name].model_dump()
                changed = ", ".join(f"{k}: {before[k]!r} -> {after[k]!r}" for k in after if before.get(k) != after[k])
                lines.append(f"~ {label} {name}: {changed}")
    return sorted(lines, key=lambda line: (line[2:].split()[0], line))
