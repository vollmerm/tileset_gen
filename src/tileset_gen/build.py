"""Spec -> generators -> atlas -> exported files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import registry
from .atlas import Atlas, ConnectedSet, ObjectGroup, WangSet
from .context import GenContext
from .generators.autotile import MIXED_CORNERS, compose, corner_mask
from .generators.characters import EMOTES, FRAMES, ROWS, emote_balloon
from .generators.objects import E, W
from .generators.terrain import decorate
from .noise import Noise
from .palette import SHADOW, Palette
from .spec import AnimationSpec, TerrainSpec, TilesetSpec, TransitionSpec
from .validate import merged_colors, validate_spec

DECOR_PROBABILITY = 0.12  # relative to 1.0 for each plain variant


PATCH_SUFFIX = "_patch"


def with_patches(spec: TilesetSpec) -> tuple[TilesetSpec, dict[str, str]]:
    """Add a darker "patch" terrain plus a blend transition for every terrain with `patches` > 0.

    A patch is drawn with its host's colors one ramp step darker (see `GenContext.tone`),
    so it costs no palette colors. Otherwise patches are ordinary terrains to the rest
    of the pipeline, so they get autotiles, animation, a Tiled brush color and preview
    placement for free.
    Returns the expanded spec and a patch -> host map.
    """
    names = {t.name for t in spec.terrains}
    out = spec.model_copy(deep=True)
    patch_of: dict[str, str] = {}
    for t in spec.terrains:
        name = t.name + PATCH_SUFFIX
        if t.patches <= 0 or name in names:
            continue
        out.terrains.append(
            TerrainSpec(
                name=name,
                generator=t.generator,
                colors=t.colors,
                variants=min(2, t.variants),
                decor=t.decor // 2,
                patches=0.0,
                params=t.params,
            )
        )
        out.transitions.append(TransitionSpec(lower=t.name, upper=name, edge_style="blend", roughness=0.7))
        for a in spec.animations:
            if a.terrain == t.name:
                out.animations.append(
                    AnimationSpec(name=a.name + PATCH_SUFFIX, terrain=name, frames=a.frames, frame_ms=a.frame_ms)
                )
        patch_of[name] = t.name
    return out, patch_of


EMOTE_COLORS = 32
EMOTE_MS = 120


@dataclass
class CharacterSheet:
    """A sprite sheet: frames[row][frame], each (h, w) palette indices.

    For characters the rows are `ROWS` (walk and attack per direction, then the
    emotes); for the shared emote balloons they are `EMOTES`. Each sheet gets its
    own reduced palette (`max_colors`, like a SNES sprite palette) instead of
    sharing the tileset's budget.
    """

    name: str
    frames: np.ndarray  # (rows, FRAMES, h, w)
    frame_ms: int
    remap: np.ndarray | None = None
    lut: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Frame (width, height) in pixels."""
        return self.frames.shape[3], self.frames.shape[2]

    def sheet(self) -> np.ndarray:
        """One row per animation row, one column per frame."""
        return np.concatenate([np.concatenate(list(row), axis=1) for row in self.frames], axis=0)

    def to_rgba(self, indices: np.ndarray) -> np.ndarray:
        return self.lut[self.remap[indices]]

    def reduce_palette(self, palette: Palette, max_colors: int) -> None:
        counts = np.bincount(self.frames.ravel(), minlength=len(palette.colors))
        self.remap = palette.reduce(counts, max_colors)
        self.lut = palette.rgba_lut(self.remap)

    def head_top(self) -> int:
        """First row of the standing front frame with a solid pixel: where an emote balloon sits."""
        solid = (self.frames[0, 1] > 0) & (self.frames[0, 1] != SHADOW)
        return int(np.argmax(solid.any(axis=1)))


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid spec:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass
class BuildResult:
    spec: TilesetSpec  # with patch terrains added
    atlas: Atlas
    palette: Palette
    patch_of: dict[str, str] = field(default_factory=dict)  # patch terrain -> host terrain
    terrain_tiles: dict[str, list[int]] = field(default_factory=dict)
    wangsets: list[WangSet] = field(default_factory=list)
    objects: list[ObjectGroup] = field(default_factory=list)
    connected: list[ConnectedSet] = field(default_factory=list)
    animations: dict[str, int] = field(default_factory=dict)  # standalone animation -> first tile id
    characters: list[CharacterSheet] = field(default_factory=list)
    emotes: CharacterSheet | None = None  # shared emote balloons, when there are characters
    remap: np.ndarray | None = None
    lut: np.ndarray | None = None

    def finalize_palette(self) -> None:
        """Merge colors down to spec.max_colors and build the RGBA lookup table."""
        sheet = self.atlas.sheet()
        counts = np.bincount(sheet.ravel(), minlength=len(self.palette.colors))
        if counts[SHADOW]:
            # The preview darkens ground under shadows to colors the atlas itself may never use;
            # keep them in the palette (or merged into a neighbor) instead of dropping them.
            darker = self.palette.shadow_lut()
            used = np.nonzero(counts)[0]
            np.add.at(counts, darker[used[used > SHADOW]], 1)
        self.remap = self.palette.reduce(counts, self.spec.max_colors)
        self.lut = self.palette.rgba_lut(self.remap)

    def to_rgba(self, indices: np.ndarray) -> np.ndarray:
        return self.lut[self.remap[indices]]

    def reduce_palette(self, palette: Palette, max_colors: int) -> None:
        counts = np.bincount(self.frames.ravel(), minlength=len(palette.colors))
        self.remap = palette.reduce(counts, max_colors)
        self.lut = palette.rgba_lut(self.remap)

    def head_top(self) -> int:
        """First row of the standing front frame with a solid pixel: where an emote balloon sits."""
        solid = (self.frames[0, 1] > 0) & (self.frames[0, 1] != SHADOW)
        return int(np.argmax(solid.any(axis=1)))

    def probability(self, tid: int) -> float:
        """Relative chance of picking this tile among its terrain's tiles (decor tiles are rare)."""
        return self.atlas.meta[tid].props.get("probability", 1.0)

    def pick_terrain(self, name: str, rng: np.random.Generator) -> int:
        ids = self.terrain_tiles[name]
        w = np.array([self.probability(t) for t in ids])
        return int(rng.choice(ids, p=w / w.sum()))


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
    spec, patch_of = with_patches(spec)
    T = spec.tile_size
    palette = Palette(spec.palette_dict())
    atlas = Atlas(T, columns=max(8, 256 // T))
    res = BuildResult(spec, atlas, palette, patch_of=patch_of)

    def ctx_for(key: str, colors: list[str], info: registry.GeneratorInfo, tone: int = 0) -> GenContext:
        return GenContext(T, spec.seed, key, palette, merged_colors(colors, info), tone)

    terrain_anim = {a.terrain: a for a in spec.animations if a.terrain}
    frames: dict[str, list[np.ndarray]] = {}
    frame_ms: dict[str, int] = {}
    ramps: dict[str, list[int]] = {}

    for t in spec.terrains:
        info = registry.get(t.generator)
        p = info.params.model_validate(t.params)
        ctx = ctx_for(f"terrain:{t.name}", t.colors, info, tone=-1 if t.name in patch_of else 0)
        ramps[t.name] = ctx.ramp(0)
        anim = terrain_anim.get(t.name)
        n = anim.frames if anim else 1
        ms = anim.frame_ms if anim else 0
        ids = []
        variants = []
        for v in range(t.variants):
            fr = [info.fn(ctx, p, v, k / n) for k in range(n)]
            ids.append(_add_animated(atlas, fr, f"{t.name}_{v}", "terrain", t.name, ms, terrain=t.name, variant=v))
            variants.append(fr)
        frames[t.name] = variants[0]
        for k in range(t.decor if info.decor else 0):
            fr = [decorate(info, ctx, p, f, k) for f in variants[k % t.variants]]
            ids.append(
                _add_animated(
                    atlas,
                    fr,
                    f"{t.name}_decor{k}",
                    "terrain",
                    t.name,
                    ms,
                    terrain=t.name,
                    decor=k,
                    probability=DECOR_PROBABILITY,
                )
            )
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
        if info.connected:
            # All 16 neighbor masks as a 4x4 block (mask = row * 4 + col), then the gate.
            T = ctx.T
            block = np.zeros((4 * T, 4 * T), dtype=np.uint8)
            for m in range(16):
                block[(m // 4) * T : (m // 4 + 1) * T, (m % 4) * T : (m % 4 + 1) * T] = info.fn(ctx, p, m)
            rows = atlas.add_block(block, o.name, "object", o.name, object=o.name)
            cs = ConnectedSet(o.name, {m: rows[m // 4][m % 4] for m in range(16)})
            for m, tid in cs.tiles.items():
                atlas.meta[tid].props["mask"] = m
            if getattr(p, "gate", False):
                cs.gate = atlas.add(info.fn(ctx, p, E | W, gate=True), f"{o.name}_gate", "object", o.name, gate=True)
            res.connected.append(cs)
            continue
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

    for ch in spec.characters:
        info = registry.get(ch.generator)
        p = info.params.model_validate(ch.params)
        ctx = ctx_for(f"character:{ch.name}", ch.colors, info)
        drawn = {
            (action, d): [info.fn(ctx, p, action, d, k) for k in range(FRAMES)] for action, d in ROWS if d != "right"
        }
        for action, d in ROWS:
            if d == "right":  # mirrored, as on the SNES
                drawn[action, d] = [np.fliplr(f) for f in drawn[action, "left"]]
        sheet = CharacterSheet(ch.name, np.stack([np.stack(drawn[row]) for row in ROWS]), ch.frame_ms)
        sheet.reduce_palette(palette, ch.max_colors)
        res.characters.append(sheet)
    if spec.characters:
        ctx = GenContext(T, spec.seed, "emotes", palette, [])
        balloons = np.stack([np.stack([emote_balloon(ctx, e, k) for k in range(FRAMES)]) for e in EMOTES])
        res.emotes = CharacterSheet("emotes", balloons, EMOTE_MS)
        res.emotes.reduce_palette(palette, EMOTE_COLORS)

    res.finalize_palette()
    return res


def build(
    spec: TilesetSpec, out_dir: str | Path, map_size: tuple[int, int] = (24, 16), preview_scale: int = 3
) -> BuildResult:
    from .export.characters import write_characters
    from .export.png_json import write_png_json
    from .export.tiled import write_tmx, write_tsx
    from .preview import make_preview, render_preview

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = generate(spec)
    spec.save(out / "spec.json")
    write_png_json(res, out)
    write_tsx(res, out)
    write_characters(res, out)
    pmap = make_preview(res, *map_size)
    write_tmx(res, pmap, out)
    render_preview(res, pmap, preview_scale).save(out / "preview.png")
    return res
