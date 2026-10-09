"""Seamless ground tiles.

Signature: fn(ctx, params, variant, phase) -> uint8[T, T].

Variants share one periodic base field, plus a per-variant field that fades to
zero at the tile border (see `_field`). Any two variants therefore meet without
a seam while their interiors differ, which breaks up the 1-tile repeat.
`phase` in [0, 1) is the animation time for animatable terrains; phase 0 is the
static tile. Noise lattice sizes are given for 16px tiles and scale with T.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..noise import Noise, contrast, quantize, voronoi
from ..registry import Params, register


def _window(T: int) -> np.ndarray:
    w = np.sin(np.pi * (np.arange(T) + 0.5) / T)
    return np.outer(w, w)


def _field(
    ctx: GenContext,
    variant: int,
    cells: int | tuple[int, int],
    octaves: int = 2,
    amount: float = 0.6,
    persistence: float = 0.5,
    t: float = 0.0,
    tcells: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    edge_power: float = 1.0,
) -> np.ndarray:
    """Tile-periodic base noise plus edge-windowed per-variant noise.

    Raise `edge_power` when the caller amplifies small differences (e.g. ridges),
    to pin the border closer to the shared base field.
    """
    T = ctx.T
    k = max(1, T // 16)
    scaled = cells * k if isinstance(cells, int) else (cells[0] * k, cells[1] * k)
    base = Noise(ctx.rng("base"), T, cells=scaled, octaves=octaves, persistence=persistence, tcells=tcells)
    f = base.tile(t=t, dx=dx, dy=dy)
    local = Noise(ctx.rng("local", variant), T, cells=3 * k, octaves=2).tile()
    return f + (local - 0.5) * amount * _window(T) ** edge_power


def _patch(tile: np.ndarray, x: int, y: int, shadow: int, light: int, rng: np.random.Generator) -> None:
    """A small irregular shadow blob with a light rim on top (dips, drifts, worn spots)."""
    w = int(rng.integers(2, 4))
    for dx in range(-w, w + 1):
        tile[y, x + dx] = shadow
        if abs(dx) < w:
            tile[y + 1, x + dx] = shadow
            tile[y - 1, x + dx] = light
    if rng.random() < 0.5:
        tile[y + 1, x + w] = shadow


def _interior(rng: np.random.Generator, T: int, n: int, margin: int = 1):
    """`n` random interior points, kept off the tile border so variants stay seamless."""
    xs = rng.integers(margin, T - margin, size=n)
    ys = rng.integers(margin, T - margin, size=n)
    return zip(xs.tolist(), ys.tolist())


class GrassParams(Params):
    density: float = Field(0.5, ge=0.0, le=1.0, description="grass tuft density")
    flowers: float = Field(0.0, ge=0.0, le=1.0, description="flower density (uses accent color)")
    roughness: float = Field(0.5, ge=0.0, le=1.0, description="patchiness of the base color")


@register(
    "grass",
    "terrain",
    GrassParams,
    colors=[("grass", "#5aa83c"), ("flower", "#f0d848")],
    doc="Grassy ground with tufts and optional flowers.",
)
def grass(ctx: GenContext, p: GrassParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 2, octaves=3), 1.2 + p.roughness * 1.2)
    tile = quantize(f, [r[2], r[2], r[3]], dither=0.8)
    tile[f < 0.3] = r[1]
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, int(p.density * T * T / 14), margin=2):
        tile[y, x] = r[3]
        tile[y + 1, x - 1] = r[1]
        tile[y + 1, x + 1] = r[1]
        tile[y + 1, x] = r[2]
        if rng.random() < 0.5:
            tile[y - 1, x] = r[4]
    if p.flowers > 0:
        a = ctx.ramp(1)
        for x, y in _interior(rng, T, int(round(p.flowers * T * T / 40)), margin=2):
            tile[y - 1, x] = tile[y + 1, x] = tile[y, x - 1] = tile[y, x + 1] = a[2]
            tile[y, x] = a[4]
            tile[y + 2, x] = r[0] if y + 2 < T - 1 else tile[y + 2, x]
    return tile


class DirtParams(Params):
    pebbles: float = Field(0.4, ge=0.0, le=1.0, description="pebble density")
    roughness: float = Field(0.5, ge=0.0, le=1.0)


@register("dirt", "terrain", DirtParams, colors=[("soil", "#9a6b42")], doc="Bare earth or a dirt path, with pebbles.")
def dirt(ctx: GenContext, p: DirtParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 5, amount=0.5), 0.7 + p.roughness * 0.6)
    tile = quantize(f, [r[2], r[2], r[2], r[3]], dither=1.0)
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, max(1, round(p.roughness * T * T / 128)), margin=4):
        _patch(tile, x, y, r[1], r[3], rng)
    for x, y in _interior(rng, T, int(p.pebbles * T * T / 28)):
        tile[y, x] = r[3]
        tile[y + 1, x] = r[0]
        if rng.random() < 0.5 and x + 1 < T - 1:
            tile[y, x + 1] = r[2]
            tile[y + 1, x + 1] = r[0]
    return tile


class SandParams(Params):
    ripples: float = Field(0.6, ge=0.0, le=1.0, description="wind ripple coverage")
    speckles: float = Field(0.3, ge=0.0, le=1.0)


@register("sand", "terrain", SandParams, colors=[("sand", "#e2c47e")], doc="Desert or beach sand with wind ripples.")
def sand(ctx: GenContext, p: SandParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 2), 2.0)
    tile = quantize(f, [r[2], r[3]], dither=1.0)
    ys, xs = np.mgrid[0:T, 0:T]
    wave = (ys + 1.5 * np.sin(2 * np.pi * xs / T) + 0.8 * np.sin(4 * np.pi * xs / T + 1.0)) % (T / 2)
    gate = Noise(ctx.rng("ripple"), T, cells=2).tile() + (_field(ctx, variant, 2) - 0.5) * 0.5 < 0.2 + p.ripples * 0.4
    tile[(wave < 1.0) & gate] = r[2]
    tile[(wave >= 1.0) & (wave < 2.0) & gate] = r[4]
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, int(p.speckles * T * T / 30)):
        tile[y, x] = r[1] if rng.random() < 0.6 else r[4]
    return tile


class CobbleParams(Params):
    stones: int = Field(3, ge=2, le=6, description="stones per tile edge")
    cracks: float = Field(0.3, ge=0.0, le=1.0)


@register(
    "cobble",
    "terrain",
    CobbleParams,
    colors=[("stone", "#8d8a8a"), ("mortar", "#4c4447")],
    doc="Irregular cobblestones / flagstone floor (towns, dungeons, caves).",
)
def cobble(ctx: GenContext, p: CobbleParams, variant: int, phase: float) -> np.ndarray:
    T, r, m = ctx.T, ctx.ramp(0), ctx.ramp(1)
    ids, f1, f2 = voronoi(ctx.rng("base"), T, p.stones)
    mortar = (f2 - f1) < 1.3
    shades = ctx.rng("shade").choice([r[2], r[2], r[3]], size=ids.max() + 1)
    border = np.unique(np.concatenate([ids[0], ids[-1], ids[:, 0], ids[:, -1]]))
    redo = np.setdiff1d(np.arange(ids.max() + 1), border)
    shades[redo] = ctx.rng("shade", variant).choice([r[1], r[2], r[2], r[3]], size=len(redo))
    tile = shades[ids].astype(np.uint8)
    inner = ~mortar
    tile[inner & np.roll(mortar, 1, axis=0)] = r[4]
    tile[inner & np.roll(mortar, 1, axis=1) & ~np.roll(mortar, 1, axis=0)] = r[3]
    tile[inner & (np.roll(mortar, -1, axis=0) | np.roll(mortar, -1, axis=1))] = r[1]
    tile[mortar] = m[1]
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, int(p.cracks * T / 3), margin=2):
        if not mortar[y, x]:
            tile[y, x] = r[0]
            tile[y + 1, x + rng.choice([-1, 1])] = r[1]
    return tile


class BrickParams(Params):
    rows: Literal[2, 4, 8] = Field(4, description="brick courses per tile (divides the tile size)")
    wear: float = Field(0.3, ge=0.0, le=1.0, description="chipped / darker bricks")


@register(
    "brick_floor",
    "terrain",
    BrickParams,
    colors=[("brick", "#a3604a"), ("mortar", "#5a4a44")],
    doc="Regular laid brick or tile floor (castles, towns, dungeons).",
)
def brick_floor(ctx: GenContext, p: BrickParams, variant: int, phase: float) -> np.ndarray:
    T, r, m = ctx.T, ctx.ramp(0), ctx.ramp(1)
    bh = T // p.rows
    bw = bh * 2
    ys, xs = np.mgrid[0:T, 0:T]
    row = ys // bh
    off = (row % 2) * (bw // 2)
    col = ((xs + off) % T) // bw
    brick_id = row * 64 + col
    shade_rng = ctx.rng("shade")
    lut = shade_rng.choice([r[2], r[2], r[2], r[3], r[1]], size=64 * 64)
    tile = lut[brick_id].astype(np.uint8)
    tile[ys % bh == 0] = np.where(tile[ys % bh == 0] == r[1], r[2], r[3])
    mortar = (ys % bh == bh - 1) | ((xs + off) % bw == bw - 1)
    tile[mortar] = m[1]
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, int(p.wear * T / 2)):
        if not mortar[y, x]:
            tile[y, x] = r[1]
    return tile


class WoodParams(Params):
    planks: Literal[2, 4, 8] = Field(4, description="planks per tile (divides the tile size)")
    grain: float = Field(0.5, ge=0.0, le=1.0)


@register(
    "wood_floor",
    "terrain",
    WoodParams,
    colors=[("wood", "#a8743e")],
    doc="Wooden plank floor (interiors, docks, bridges).",
)
def wood_floor(ctx: GenContext, p: WoodParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    pw = T // p.planks
    ys, xs = np.mgrid[0:T, 0:T]
    plank = xs // pw
    rng = ctx.rng("base")
    shades = rng.choice([r[2], r[3], r[2]], size=p.planks)
    joints = rng.integers(0, T, size=p.planks)
    tile = shades[plank].astype(np.uint8)
    phase_off = rng.random(p.planks) * T
    grain = np.sin((ys + phase_off[plank]) * 2 * np.pi * 3 / T + xs * 0.9) > 1.0 - p.grain * 0.5
    tile[grain & (xs % pw > 0)] = r[1]
    tile[xs % pw == 0] = r[3 if p.planks > 2 else 4]
    tile[xs % pw == pw - 1] = r[0]
    joint = ys == joints[plank]
    tile[joint & (xs % pw != pw - 1)] = r[0]
    tile[np.roll(joint, 1, axis=0) & (xs % pw != pw - 1)] = r[4]
    drng = ctx.rng("detail", variant)
    for x, y in _interior(drng, T, int(T / 8)):
        if x % pw not in (0, pw - 1):
            tile[y, x] = r[1]
    return tile


class SnowParams(Params):
    drifts: float = Field(0.5, ge=0.0, le=1.0, description="blue shadow drifts")


@register("snow", "terrain", SnowParams, colors=[("snow", "#dfe8f2")], doc="Snowfield with soft shadowed drifts.")
def snow(ctx: GenContext, p: SnowParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 4, amount=0.5), 1.4)
    tile = quantize(f, [r[3], r[3], r[4]], dither=0.8)
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, max(1, round(p.drifts * T * T / 100)), margin=4):
        _patch(tile, x, y, r[2], r[4], rng)
    for x, y in _interior(rng, T, T // 6):
        tile[y, x] = r[2]
    return tile


class WaterParams(Params):
    depth: float = Field(0.5, ge=0.0, le=1.0, description="0 = shallow and bright, 1 = deep and dark")
    waves: float = Field(0.5, ge=0.0, le=1.0, description="amount of wave highlights")


@register(
    "water",
    "terrain",
    WaterParams,
    colors=[("water", "#3a78c8")],
    animatable=True,
    doc="Lake, river or sea water with wave highlights; animates by drifting waves.",
)
def water(ctx: GenContext, p: WaterParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    base = _field(ctx, variant, 2, amount=0.9, t=phase, tcells=2)
    levels = [r[1], r[1], r[2]] if p.depth > 0.6 else [r[2], r[2], r[3]] if p.depth < 0.25 else [r[1], r[2], r[2]]
    tile = quantize(contrast(base, 1.3), levels, dither=1.0)
    crest = Noise(ctx.rng("waves"), T, cells=(2, 5), octaves=2, tcells=2).tile(t=phase, dx=phase * T)
    band = 0.02 + p.waves * 0.04
    tile[np.abs(crest - 0.66) < band] = r[3]
    tile[np.abs(crest - 0.70) < band * 0.3] = r[4]
    return tile


class LavaParams(Params):
    crust: float = Field(0.5, ge=0.0, le=1.0, description="share of dark cooled crust")


@register(
    "lava",
    "terrain",
    LavaParams,
    colors=[("lava", "#e8601c")],
    animatable=True,
    doc="Molten lava with glowing veins and dark crust; animates by flowing.",
)
def lava(ctx: GenContext, p: LavaParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    flow = _field(ctx, variant, 2, octaves=3, amount=0.4, t=phase, tcells=2, dy=-phase * T, edge_power=3.0)
    ridge = np.abs(contrast(flow, 3.0) - 0.5) * 2.0  # 0 along the veins
    glow = np.clip(1.0 - ridge / (0.2 + (1.0 - p.crust) * 0.5), 0.0, 1.0) ** 0.7
    tile = quantize(glow, [r[0], r[1], r[2], r[3], r[4]], dither=0.7)
    return tile


class FloorParams(Params):
    style: Literal["tiles", "checker"] = Field("tiles", description="square tiles or a two-tone checker")
    size: Literal[1, 2, 4] = Field(2, description="floor tiles per tile edge (divides the tile size)")


@register(
    "stone_floor",
    "terrain",
    FloorParams,
    colors=[("floor", "#9c9aa6"), ("alt", "#7e7c8c")],
    doc="Square-cut stone floor tiles (temples, castles, dungeons).",
)
def stone_floor(ctx: GenContext, p: FloorParams, variant: int, phase: float) -> np.ndarray:
    T, r, a = ctx.T, ctx.ramp(0), ctx.ramp(1)
    s = T // p.size
    ys, xs = np.mgrid[0:T, 0:T]
    cell = (ys // s) + (xs // s)
    use_alt = (cell % 2 == 1) if p.style == "checker" else np.zeros_like(cell, dtype=bool)
    tile = np.where(use_alt, a[2], r[2]).astype(np.uint8)
    f = _field(ctx, variant, 4, octaves=1)
    tile[(f > 0.62) & ~use_alt] = r[3]
    tile[(f > 0.62) & use_alt] = a[3]
    lx, ly = xs % s, ys % s
    tile[(ly == 0) | (lx == 0)] = np.where(use_alt, a[3], r[4])[(ly == 0) | (lx == 0)]
    tile[(ly == s - 1) | (lx == s - 1)] = np.where(use_alt, a[0], r[0])[(ly == s - 1) | (lx == s - 1)]
    rng = ctx.rng("detail", variant)
    for x, y in _interior(rng, T, T // 6, margin=2):
        tile[y, x] = r[1]
    return tile
