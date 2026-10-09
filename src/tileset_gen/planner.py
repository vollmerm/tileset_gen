"""LLM planner: turns a free-text theme into a TilesetSpec using Claude Haiku.

The model only chooses palette colors, generators and parameters; all pixels
come from the deterministic generators. Structured outputs can't express
free-form dicts, so the model fills a "wire" schema where params are
[{name, value}] pairs, which is converted to a TilesetSpec and then checked
against the generator registry. Validation errors are fed back once for a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from . import registry
from .spec import AnimationSpec, NamedColor, ObjectSpec, TerrainSpec, TilesetSpec, TransitionSpec
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


class ParamKV(BaseModel):
    name: str
    value: float | bool | str


class WireTerrain(BaseModel):
    name: str
    generator: str
    colors: list[str]
    variants: int
    params: list[ParamKV]


class WireTransition(BaseModel):
    lower: str
    upper: str
    edge_style: Literal["soft", "shore", "cliff"]
    roughness: float


class WireObject(BaseModel):
    name: str
    generator: str
    colors: list[str]
    width: int
    height: int
    params: list[ParamKV]
    place_on: list[str]
    density: float


class WireAnimation(BaseModel):
    name: str
    terrain: str | None
    generator: str | None
    colors: list[str]
    frames: int
    frame_ms: int
    params: list[ParamKV]


class WireSpec(BaseModel):
    name: str
    theme: str
    palette: list[NamedColor]
    terrains: list[WireTerrain]
    transitions: list[WireTransition]
    objects: list[WireObject]
    animations: list[WireAnimation]

    def to_spec(self, seed: int, tile_size: int, max_colors: int) -> TilesetSpec:
        def params(kvs: list[ParamKV]) -> dict:
            # The wire schema only has `number`; turn 3.0 back into 3 so int/Literal params validate.
            return {
                kv.name: int(kv.value) if isinstance(kv.value, float) and kv.value.is_integer() else kv.value
                for kv in kvs
            }

        return TilesetSpec(
            name=self.name,
            theme=self.theme,
            seed=seed,
            tile_size=tile_size,
            max_colors=max_colors,
            palette=self.palette,
            terrains=[TerrainSpec(**t.model_dump(exclude={"params"}), params=params(t.params)) for t in self.terrains],
            transitions=[TransitionSpec(**t.model_dump()) for t in self.transitions],
            objects=[ObjectSpec(**o.model_dump(exclude={"params"}), params=params(o.params)) for o in self.objects],
            animations=[
                AnimationSpec(**a.model_dump(exclude={"params"}), params=params(a.params)) for a in self.animations
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
generator's color-slot order; omitted slots use defaults), variants, and params. Use 2-3 variants for \
natural ground (grass, dirt, sand, snow, cobble) and 1 for water, lava and regular floors.
- transitions: blend `upper` over `lower` with corner autotiles; at most one transition per terrain pair. \
Connect every terrain: transitions should form a chain ordered like elevation \
(e.g. water -> sand -> grass -> dirt), optionally with side branches (grass -> flower_meadow). \
A terrain with no transition can only appear as hard-edged rectangles, so only leave one unconnected \
when that is intended (a carpet, a plaza). \
edge_style: "shore" for land over water, "cliff" for raised ground (mesas, plateaus, floors over lava pits), \
"soft" otherwise. roughness 0-1 (0.3-0.6 typical).
- objects: object generator, size in tiles (width/height 1-4; trees 1x1 or 2x2, houses 2x2 or 3x3, \
pine 1x2), place_on lists terrain names it may stand on (never water/lava). \
density (0-1) controls how often it appears in the preview map: 0.5-0.8 for common scenery \
(trees, bushes, rocks, flowers), 0.3-0.5 for props (barrels, fences, signs), 0.2-0.4 for landmarks \
(houses, chests). Values below 0.1 make the object nearly invisible.
- animations: either set `terrain` to an animatable terrain's name (generator null) to animate that terrain, \
or set `generator` to an animation generator (terrain null). frames 2-8, frame_ms 80-400.
- params: list of {{name, value}} using only the params listed for that generator, within their ranges \
(for a|b|c params use one of the listed values). Use an empty list to accept the defaults.
- Names are lowercase snake_case and unique across terrains, objects and animations.

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
    client = client or anthropic.Anthropic()
    stats = stats if stats is not None else PlanStats()
    messages: list[dict] = [{"role": "user", "content": f"Plan a tileset for this theme: {prompt}"}]
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
