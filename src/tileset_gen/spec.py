"""The tileset spec: the JSON document the planner writes and the builder reads."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .style import STYLES, StyleName

ParamValue = float | int | str | bool
EdgeStyle = Literal["soft", "shore", "cliff", "blend"]


class NamedColor(BaseModel):
    name: str
    hex: str = Field(description="#rrggbb base color; shading ramps are derived from it")


class TerrainSpec(BaseModel):
    name: str
    generator: str
    colors: list[str] = Field(default_factory=list, description="palette names or hex, in the generator's slot order")
    variants: int = Field(2, ge=1, le=4)
    decor: int = Field(
        2, ge=0, le=6, description="rare detail tiles (tall grass, stones, cracks...) mixed in to break up large areas"
    )
    patches: float = Field(
        0.3,
        ge=0.0,
        le=0.6,
        description="share of the terrain's open areas covered by darker multi-tile patches (lush grass, damp earth)",
    )
    params: dict[str, ParamValue] = Field(default_factory=dict)


class TransitionSpec(BaseModel):
    lower: str = Field(description="terrain underneath")
    upper: str = Field(description="terrain drawn on top, inside the corner masks")
    edge_style: EdgeStyle = "soft"
    roughness: float = Field(0.5, ge=0.0, le=1.0)


class ObjectSpec(BaseModel):
    name: str
    generator: str
    colors: list[str] = Field(default_factory=list)
    width: int = Field(1, ge=1, le=4, description="in tiles")
    height: int = Field(1, ge=1, le=4, description="in tiles")
    params: dict[str, ParamValue] = Field(default_factory=dict)
    place_on: list[str] = Field(default_factory=list, description="terrains this object may stand on in the preview")
    density: float = Field(0.3, ge=0.0, le=1.0, description="how often it appears in the preview map")


class AnimationSpec(BaseModel):
    name: str
    terrain: str | None = Field(None, description="animate this terrain (its generator must be animatable)")
    generator: str | None = Field(None, description="standalone animated tile generator")
    colors: list[str] = Field(default_factory=list)
    frames: int = Field(4, ge=2, le=8)
    frame_ms: int = Field(200, ge=50, le=2000)
    params: dict[str, ParamValue] = Field(default_factory=dict)


class CharacterSpec(BaseModel):
    name: str
    generator: str
    colors: list[str] = Field(default_factory=list, description="palette names or hex, in the generator's slot order")
    params: dict[str, ParamValue] = Field(default_factory=dict)
    max_colors: int = Field(
        15,
        ge=3,
        le=31,
        description="colors in this sprite's own palette: 15 like a SNES sprite, 3 like the NES "
        "(defaults to the tileset style's sprite size)",
    )
    frame_ms: int = Field(160, ge=60, le=600, description="duration of each walk frame")
    place_on: list[str] = Field(default_factory=list, description="terrains it may stand on in the preview")
    count: int = Field(1, ge=0, le=4, description="copies standing around the preview map")


class TilesetSpec(BaseModel):
    name: str
    theme: str = ""
    seed: int = 0
    tile_size: Literal[16, 32] = 16
    style: StyleName = Field(
        "snes",
        description="snes: 16-bit shading; nes: NES hardware palette, 3 shades, black outlines; "
        "gbc: Game Boy Color 15-bit color, 4 shades",
    )
    max_colors: int = Field(48, ge=8, le=255, description="tileset color budget (defaults to the style's)")
    palette: list[NamedColor] = Field(default_factory=list)
    terrains: list[TerrainSpec]
    transitions: list[TransitionSpec] = Field(default_factory=list)
    objects: list[ObjectSpec] = Field(default_factory=list)
    animations: list[AnimationSpec] = Field(default_factory=list)
    characters: list[CharacterSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _style_defaults(self) -> TilesetSpec:
        """Color budgets left unset follow the style: 16 colors and 3-color sprites for the NES, say."""
        # Written through __dict__ so they still count as unset, and `restyled` can recompute them.
        st = STYLES[self.style]
        if "max_colors" not in self.model_fields_set:
            self.__dict__["max_colors"] = st.max_colors
        for c in self.characters:
            if "max_colors" not in c.model_fields_set:
                c.__dict__["max_colors"] = st.sprite_colors
        return self

    def restyled(self, style: StyleName) -> TilesetSpec:
        """A copy in another style, with the tileset and sprite color budgets reset to that style's."""
        data = self.model_dump(exclude_unset=True, exclude={"max_colors"})
        for c in data.get("characters", []):
            c.pop("max_colors", None)
        return TilesetSpec.model_validate({**data, "style": style})

    def palette_dict(self) -> dict[str, str]:
        return {c.name: c.hex for c in self.palette}

    @classmethod
    def load(cls, path: str | Path) -> TilesetSpec:
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2, exclude_none=True) + "\n")
