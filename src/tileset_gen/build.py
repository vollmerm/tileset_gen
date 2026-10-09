"""Spec -> generators -> atlas -> exported files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import registry
from .atlas import Atlas, ObjectGroup, WangSet
from .context import GenContext
from .generators.autotile import MIXED_CORNERS, compose, corner_mask
from .noise import Noise
from .palette import Palette
from .spec import TilesetSpec
from .validate import merged_colors, validate_spec


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid spec:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass
class BuildResult:
    spec: TilesetSpec
    atlas: Atlas
    palette: Palette
    terrain_tiles: dict[str, list[int]] = field(default_factory=dict)
    wangsets: list[WangSet] = field(default_factory=list)
    objects: list[ObjectGroup] = field(default_factory=list)
    animations: dict[str, int] = field(default_factory=dict)  # standalone animation -> first tile id
    remap: np.ndarray | None = None
    lut: np.ndarray | None = None

    def finalize_palette(self) -> None:
        """Merge colors down to spec.max_colors and build the RGBA lookup table."""
        sheet = self.atlas.sheet()
        counts = np.bincount(sheet.ravel(), minlength=len(self.palette.colors))
        self.remap = self.palette.reduce(counts, self.spec.max_colors)
        self.lut = self.palette.rgba_lut(self.remap)

    def to_rgba(self, indices: np.ndarray) -> np.ndarray:
        return self.lut[self.remap[indices]]


def _add_animated(
    atlas: Atlas, frames: list[np.ndarray], name: str, kind: str, group: str, frame_ms: int, **props
) -> int:
    tid = atlas.add(frames[0], name, kind, group, **props)
    if len(frames) > 1:
        ids = [tid] + [atlas.add(f, f"{name}_f{k}", "frame", group) for k, f in enumerate(frames[1:], 1)]
        atlas.meta[tid].animation = [(i, frame_ms) for i in ids]
    return tid


def generate(spec: TilesetSpec) -> BuildResult:
    errors = validate_spec(spec)
    if errors:
        raise SpecError(errors)
    T = spec.tile_size
    palette = Palette(spec.palette_dict())
    atlas = Atlas(T, columns=max(8, 256 // T))
    res = BuildResult(spec, atlas, palette)

    def ctx_for(key: str, colors: list[str], info: registry.GeneratorInfo) -> GenContext:
        return GenContext(T, spec.seed, key, palette, merged_colors(colors, info))

    terrain_anim = {a.terrain: a for a in spec.animations if a.terrain}
    frames: dict[str, list[np.ndarray]] = {}
    frame_ms: dict[str, int] = {}
    ramps: dict[str, list[int]] = {}

    for t in spec.terrains:
        info = registry.get(t.generator)
        p = info.params.model_validate(t.params)
        ctx = ctx_for(f"terrain:{t.name}", t.colors, info)
        ramps[t.name] = ctx.ramp(0)
        anim = terrain_anim.get(t.name)
        n = anim.frames if anim else 1
        ids = []
        for v in range(t.variants):
            fr = [info.fn(ctx, p, v, k / n) for k in range(n)]
            ids.append(
                _add_animated(
                    atlas,
                    fr,
                    f"{t.name}_{v}",
                    "terrain",
                    t.name,
                    anim.frame_ms if anim else 0,
                    terrain=t.name,
                    variant=v,
                )
            )
            if v == 0:
                frames[t.name] = fr
        if anim:
            frame_ms[t.name] = anim.frame_ms
        res.terrain_tiles[t.name] = ids

    for tr in spec.transitions:
        ws = WangSet(f"{tr.lower}/{tr.upper}", tr.lower, tr.upper)
        ws.tiles[(0, 0, 0, 0)] = list(res.terrain_tiles[tr.lower])
        ws.tiles[(1, 1, 1, 1)] = list(res.terrain_tiles[tr.upper])
        ctx = GenContext(T, spec.seed, f"transition:{tr.lower}:{tr.upper}", palette, [])
        edge_noise = Noise(ctx.rng("edge"), T, cells=4, octaves=2)
        lf, uf = frames[tr.lower], frames[tr.upper]
        n = min(8, math.lcm(len(lf), len(uf)))
        ms = frame_ms.get(tr.lower) or frame_ms.get(tr.upper) or 0
        for corners in MIXED_CORNERS:
            mask = corner_mask(T, corners, edge_noise, tr.roughness)
            fr = [
                compose(lf[k % len(lf)], uf[k % len(uf)], mask, tr.edge_style, ramps[tr.lower], ramps[tr.upper])
                for k in range(n)
            ]
            code = "".join(map(str, corners))
            ws.tiles[corners] = [
                _add_animated(
                    atlas, fr, f"{tr.lower}_{tr.upper}_{code}", "transition", ws.name, ms, corners=list(corners)
                )
            ]
        res.wangsets.append(ws)

    for o in spec.objects:
        info = registry.get(o.generator)
        p = info.params.model_validate(o.params)
        ctx = ctx_for(f"object:{o.name}", o.colors, info)
        canvas = info.fn(ctx, p, o.width, o.height)
        if o.width * o.height == 1:
            rows = [[atlas.add(canvas, o.name, "object", o.name, object=o.name)]]
        else:
            rows = atlas.add_block(canvas, o.name, "object", o.name, object=o.name)
        res.objects.append(ObjectGroup(o.name, o.width, o.height, rows))

    for a in spec.animations:
        if a.generator is None:
            continue
        info = registry.get(a.generator)
        p = info.params.model_validate(a.params)
        ctx = ctx_for(f"animation:{a.name}", a.colors, info)
        fr = [info.fn(ctx, p, k, a.frames) for k in range(a.frames)]
        res.animations[a.name] = _add_animated(atlas, fr, a.name, "animation", a.name, a.frame_ms)

    res.finalize_palette()
    return res


def build(
    spec: TilesetSpec, out_dir: str | Path, map_size: tuple[int, int] = (24, 16), preview_scale: int = 3
) -> BuildResult:
    from .export.png_json import write_png_json
    from .export.tiled import write_tmx, write_tsx
    from .preview import make_preview, render_preview

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = generate(spec)
    spec.save(out / "spec.json")
    write_png_json(res, out)
    write_tsx(res, out)
    pmap = make_preview(res, *map_size)
    write_tmx(res, pmap, out)
    render_preview(res, pmap, preview_scale).save(out / "preview.png")
    return res
