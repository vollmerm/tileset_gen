"""Object tiles drawn on a transparent canvas of (height*T, width*T).

Signature: fn(ctx, params, width, height) -> uint8[H, W]. Multi-tile objects
are drawn whole and sliced into tiles by the builder. Shapes keep a 1px margin
so the selective outline fits inside the canvas.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..draw import cylinder_levels, ellipse_height, grid, outline, put, rect, shade, shift
from ..noise import Noise
from ..registry import Params, register


def _canvas(ctx: GenContext, w: int, h: int) -> np.ndarray:
    return np.zeros((h * ctx.T, w * ctx.T), dtype=np.uint8)


def _trunk(c: np.ndarray, cx: float, top: int, bottom: int, width: int, ramp: list[int]) -> None:
    x0 = int(round(cx - width / 2))
    x1 = x0 + width - 1
    xs = np.arange(x0, x1 + 1)
    c[top : bottom + 1, x0 : x1 + 1] = cylinder_levels(xs + 0.5, x0, x1 + 1, ramp[1:4])[None, :]
    put(c, x0 - 1, bottom, ramp[1])
    put(c, x1 + 1, bottom, ramp[2])


class TreeParams(Params):
    canopy: float = Field(0.8, ge=0.4, le=1.0, description="canopy size relative to the canvas")
    clumps: int = Field(5, ge=2, le=9, description="number of leaf clumps")
    fruit: float = Field(0.0, ge=0.0, le=1.0, description="fruit/blossom amount (accent color)")


@register(
    "tree",
    "object",
    TreeParams,
    colors=[("foliage", "#3f8f3a"), ("trunk", "#7a5230"), ("fruit", "#d8443a")],
    doc="Round deciduous tree. 1x1 is a small tree, 2x2 a big one.",
)
def tree(ctx: GenContext, p: TreeParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    fr, tr = ctx.ramp(0), ctx.ramp(1)
    rng = ctx.rng("shape")
    _trunk(c, W / 2, int(H * 0.5), H - 2, max(2, W // 6), tr)
    R = min(W * 0.42, H * 0.36) * p.canopy
    cx, cy = W / 2, R * 1.05 + 1
    blobs = [(cx, cy, R * 0.95, R * 0.85)]
    for i in range(p.clumps):
        ang = 2 * np.pi * i / p.clumps + rng.uniform(-0.3, 0.3)
        d = R * rng.uniform(0.45, 0.62)
        rr = R * rng.uniform(0.42, 0.55)
        blobs.append((cx + np.cos(ang) * d, cy + np.sin(ang) * d * 0.8, rr, rr * 0.9, 0.9))
    height = ellipse_height(H, W, blobs)
    leaves = Noise(ctx.rng("leaves"), max(W, H), cells=max(2, W // 4)).tile()[:H, :W]
    height = np.where(height > 0, height + (leaves - 0.5) * 0.25, 0)
    mask = height > 0.05
    mask[:1] = mask[-1:] = False
    mask[:, :1] = mask[:, -1:] = False
    canopy = shade(height, mask, fr[1:5], dither=0.7, ambient=0.2)
    c = np.where(mask, canopy, c).astype(np.uint8)
    # Shadow where the canopy hangs over the trunk.
    under = (c == tr[1]) | (c == tr[2]) | (c == tr[3])
    c[under & shift(mask, 1, 0, False)] = tr[0]
    if p.fruit > 0:
        a = ctx.ramp(2)
        ys, xs = np.nonzero(mask & (canopy >= fr[2]))
        n = int(len(ys) * p.fruit * 0.06)
        for i in rng.choice(len(ys), size=min(n, len(ys)), replace=False):
            put(c, xs[i], ys[i], a[2])
            put(c, xs[i], ys[i] - 1, a[4]) if mask[max(ys[i] - 1, 0), xs[i]] else None
    return outline(c, ctx.palette)


class PineParams(Params):
    tiers: int = Field(3, ge=2, le=5, description="stacked branch tiers")
    snow: bool = Field(False, description="snow on the branches")


@register(
    "pine",
    "object",
    PineParams,
    colors=[("needles", "#2f6e4a"), ("trunk", "#6b4528")],
    doc="Conifer tree with stacked triangular tiers. 1x1 or 1x2 / 2x2.",
)
def pine(ctx: GenContext, p: PineParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    nr, tr = ctx.ramp(0), ctx.ramp(1)
    trunk_h = max(2, H // 7)
    _trunk(c, W / 2, H - 2 - trunk_h, H - 2, max(2, W // 7), tr)
    ys, xs = grid(H, W)
    cx = W / 2
    top, bottom = 1.0, H - 2 - trunk_h + 1
    span = (bottom - top) / (p.tiers * 0.75 + 0.25)
    half_max = W / 2 - 1.2
    for i in range(p.tiers):
        t0 = top + i * span * 0.75
        t1 = t0 + span
        half = half_max * (0.55 + 0.45 * (i + 1) / p.tiers)
        rel = (ys - t0) / (t1 - t0)
        inside = (rel >= 0) & (rel <= 1) & (np.abs(xs - cx) <= rel * half + 0.5)
        lit = xs < cx
        tier = np.where(lit, nr[3], nr[2])
        tier = np.where(lit & (np.abs(xs - cx) > rel * half - 1.5), nr[4], tier)
        tier = np.where(~lit & (np.abs(xs - cx) > rel * half - 1.5), nr[1], tier)
        tier = np.where(rel > 0.82, nr[1], tier)
        c = np.where(inside, tier, c).astype(np.uint8)
        if p.snow:
            s = ctx.palette.ramp("#eef4fa")
            cap = inside & (rel < 0.35) & (xs < cx + 1)
            c[cap] = s[3]
    return outline(c, ctx.palette)


class BushParams(Params):
    berries: float = Field(0.0, ge=0.0, le=1.0, description="berry/flower amount (accent color)")


@register("bush", "object", BushParams, colors=[("leaves", "#4c9a3e"), ("berry", "#c83a5a")], doc="Low round shrub.")
def bush(ctx: GenContext, p: BushParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    rng = ctx.rng("shape")
    base_y = H - 2
    R = W * 0.36
    blobs = [(W / 2, base_y - R * 0.75, R, R * 0.8)]
    for dx in (-0.55, 0.55):
        blobs.append((W / 2 + dx * R, base_y - R * 0.55, R * 0.6, R * 0.55, 0.85))
    blobs.append((W / 2 + rng.uniform(-0.2, 0.2) * R, base_y - R * 1.2, R * 0.55, R * 0.5, 0.9))
    height = ellipse_height(H, W, blobs)
    mask = height > 0.05
    mask[base_y + 1 :] = False
    c = shade(height, mask, r[1:5], dither=0.7)
    if p.berries > 0:
        a = ctx.ramp(1)
        ys, xs = np.nonzero(mask)
        for i in rng.choice(len(ys), size=int(len(ys) * p.berries * 0.08), replace=False):
            c[ys[i], xs[i]] = a[3]
    return outline(c, ctx.palette)


class RockParams(Params):
    lumps: int = Field(3, ge=1, le=5, description="number of rock lumps")
    moss: float = Field(0.0, ge=0.0, le=1.0, description="moss on top (accent color)")


@register(
    "rock",
    "object",
    RockParams,
    colors=[("stone", "#8b8794"), ("moss", "#5c8a3a")],
    doc="Rock or boulder; 2x2 makes a large boulder.",
)
def rock(ctx: GenContext, p: RockParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    rng = ctx.rng("shape")
    base = H - 2
    R = min(W, H) * 0.34
    blobs = [(W / 2, base - R * 0.55, R * 1.15, R * 0.85)]
    for _ in range(p.lumps - 1):
        rx = R * rng.uniform(0.45, 0.7)
        blobs.append((W / 2 + rng.uniform(-0.6, 0.6) * R, base - rx * 0.6, rx, rx * 0.8, 0.9))
    height = ellipse_height(H, W, blobs)
    mask = height > 0.04
    mask[base + 1 :] = False
    facets = Noise(ctx.rng("facets"), max(W, H), cells=3).tile()[:H, :W]
    height = height + (facets - 0.5) * 0.18 * (height > 0)
    c = shade(height, mask, [r[1], r[2], r[2], r[3], r[4]], dither=0.3, strength=5.0, ambient=0.1)
    ys, xs = np.nonzero(mask)
    if len(ys):
        i = rng.integers(len(ys))
        x, y = xs[i], ys[i]
        for _ in range(int(R)):
            if mask[y, x]:
                c[y, x] = r[0]
            x += rng.choice([-1, 0, 1])
            y += 1
            if y >= H:
                break
    if p.moss > 0:
        m = ctx.ramp(1)
        top = mask & ~shift(mask, 1, 0, False)
        for _ in range(2):
            top = top | (mask & shift(top, 1, 0, False))
        f = Noise(ctx.rng("moss"), max(W, H), cells=4).tile()[:H, :W]
        c[top & (f < p.moss * 0.8 + 0.1)] = m[2]
        c[top & (f < p.moss * 0.4)] = m[3]
    return outline(c, ctx.palette)


class FlowerParams(Params):
    count: int = Field(5, ge=1, le=12, description="flowers per tile")


@register(
    "flowers",
    "object",
    FlowerParams,
    colors=[("petals", "#e85a8a"), ("stems", "#3f8f3a")],
    doc="A small patch of flowers on a transparent background (place over grass).",
)
def flowers(ctx: GenContext, p: FlowerParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    pr, sr = ctx.ramp(0), ctx.ramp(1)
    rng = ctx.rng("shape")
    for _ in range(p.count * w * h):
        x, y = int(rng.integers(2, W - 2)), int(rng.integers(2, H - 3))
        c[y + 1, x] = sr[2]
        put(c, x, y + 2, sr[1])
        for dx, dy in ((0, -1), (-1, 0), (1, 0)):
            put(c, x + dx, y + dy, pr[2])
        c[y, x] = pr[4]
    return c


class ChestParams(Params):
    open: bool = Field(False, description="draw the chest opened")


@register(
    "chest",
    "object",
    ChestParams,
    colors=[("wood", "#9a5a2e"), ("metal", "#e0b640")],
    doc="Treasure chest with metal bands.",
)
def chest(ctx: GenContext, p: ChestParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, mr = ctx.ramp(0), ctx.ramp(1)
    x0, x1 = 2, W - 3
    y_lid, y_seam, y1 = int(H * 0.3), int(H * 0.55), H - 2
    if p.open:
        rect(c, x0, y_lid - 3, x1, y_seam - 2, wr[1])
        rect(c, x0 + 1, y_seam - 2, x1 - 1, y_seam, wr[0])
        put(c, (x0 + x1) // 2, y_seam - 1, mr[4])
    else:
        rect(c, x0, y_lid, x1, y_seam, wr[2])
        rect(c, x0 + 1, y_lid, x1 - 1, y_lid, wr[4])
        rect(c, x0, y_lid + 1, x1, y_lid + 1, wr[3])
    rect(c, x0, y_seam + 1, x1, y1, wr[2])
    rect(c, x0, y1, x1, y1, wr[1])
    rect(c, x0, y_seam + 1, x1, y_seam + 1, wr[0])
    for bx in (x0 + 2, x1 - 2):
        rect(c, bx, y_lid if not p.open else y_seam + 1, bx, y1, mr[2])
    lx = (x0 + x1) // 2
    rect(c, lx, y_seam, lx + 1, y_seam + 2, mr[3])
    put(c, lx, y_seam + 1, mr[0])
    return outline(c, ctx.palette)


@register("sign", "object", Params, colors=[("wood", "#a8743e")], doc="Wooden signpost.")
def sign(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    px = W // 2 - 1
    rect(c, px, H // 2, px, H - 2, r[2])
    rect(c, px + 1, H // 2, px + 1, H - 2, r[1])
    x0, x1, y0, y1 = 2, W - 3, int(H * 0.2), int(H * 0.6)
    rect(c, x0, y0, x1, y1, r[3])
    rect(c, x0, y0, x1, y0, r[4])
    rect(c, x0, y1, x1, y1, r[1])
    for y in range(y0 + 2, y1 - 1, 2):
        rect(c, x0 + 2, y, x1 - 3, y, r[1])
    return outline(c, ctx.palette)


@register(
    "barrel",
    "object",
    Params,
    colors=[("wood", "#9a6236"), ("bands", "#5c5866")],
    doc="Wooden barrel with metal hoops.",
)
def barrel(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, br = ctx.ramp(0), ctx.ramp(1)
    x0, x1, y0, y1 = 3, W - 4, int(H * 0.2), H - 2
    xs = np.arange(x0, x1 + 1)
    body = cylinder_levels(xs + 0.5, x0, x1 + 1, wr[1:5])
    c[y0 : y1 + 1, x0 : x1 + 1] = body[None, :]
    bands = cylinder_levels(xs + 0.5, x0, x1 + 1, br[1:4])
    for y in (y0 + 2, y1 - 2):
        c[y, x0 : x1 + 1] = bands
    rect(c, x0 + 1, y0 - 1, x1 - 1, y0, wr[3])
    rect(c, x0 + 2, y0, x1 - 2, y0, wr[1])
    return outline(c, ctx.palette)


@register("fence", "object", Params, colors=[("wood", "#b08048")], doc="Wooden fence segment that tiles horizontally.")
def fence(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    for y in (int(H * 0.35), int(H * 0.65)):
        rect(c, 0, y, W - 1, y + 1, r[2])
        rect(c, 0, y, W - 1, y, r[3])
    for px in range(ctx.T // 2 - 1, W, ctx.T):
        rect(c, px, int(H * 0.15), px + 1, H - 2, r[2])
        rect(c, px, int(H * 0.15), px, H - 2, r[3])
        put(c, px, int(H * 0.15) - 1, r[4])
    return outline(c, ctx.palette)


class WallParams(Params):
    style: Literal["stone", "brick"] = Field("stone", description="large stone blocks or small bricks")


@register(
    "wall",
    "object",
    WallParams,
    colors=[("stone", "#8a8494")],
    doc="Solid wall face that tiles in both directions (castles, dungeons, town walls).",
)
def wall(ctx: GenContext, p: WallParams, w: int, h: int) -> np.ndarray:
    T = ctx.T
    H, W = h * T, w * T
    r = ctx.ramp(0)
    bh = T // 2 if p.style == "stone" else T // 4
    bw = bh * 2
    ys, xs = np.mgrid[0:H, 0:W]
    off = ((ys // bh) % 2) * (bw // 2)
    lx, ly = (xs + off) % bw, ys % bh
    block = (ys // bh) * 97 + (xs + off) // bw
    shades = ctx.rng("shade").choice([r[2], r[2], r[3]], size=int(block.max()) + 1)
    c = shades[block].astype(np.uint8)
    c[ly == 0] = r[4]
    c[(lx == 0) & (ly > 0)] = r[3]
    c[(ly == bh - 2) & (lx > 0)] = r[1]
    c[(ly == bh - 1) | (lx == bw - 1)] = r[0]
    f = Noise(ctx.rng("wear"), T, cells=4).tile()
    worn = np.tile(f, (h, w)) > 0.7
    c[worn & (c == r[2])] = r[1]
    return c


@register(
    "roof", "object", Params, colors=[("roof", "#b8483a")], doc="Shingled roof surface that tiles in both directions."
)
def roof(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    T = ctx.T
    H, W = h * T, w * T
    r = ctx.ramp(0)
    return _shingles(H, W, T, r)


def _shingles(H: int, W: int, T: int, r: list[int]) -> np.ndarray:
    sh, sw = max(3, T // 4), max(4, T // 4)
    ys, xs = np.mgrid[0:H, 0:W]
    off = ((ys // sh) % 2) * (sw // 2)
    lx, ly = (xs + off) % sw, ys % sh
    c = np.full((H, W), r[2], dtype=np.uint8)
    c[ly == 0] = r[3]
    c[(ly == sh - 1)] = r[1]
    c[(lx == sw - 1) & (ly < sh - 1)] = r[1]
    c[(lx == 0) & (ly == 0)] = r[4]
    return c


@register(
    "door",
    "object",
    Params,
    colors=[("wood", "#8a5430"), ("frame", "#6e6878")],
    doc="Arched wooden door in a stone frame (transparent outside the frame).",
)
def door(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, fr = ctx.ramp(0), ctx.ramp(1)
    ys, xs = grid(H, W)
    cx = W / 2
    half = W / 2 - 1
    arch_y = half + 1
    frame = (np.abs(xs - cx) <= half) & ((ys >= arch_y) | (((xs - cx) / half) ** 2 + ((ys - arch_y) / half) ** 2 <= 1))
    frame &= ys < H - 1
    inner_half = half - 2
    inner = (np.abs(xs - cx) <= inner_half) & (
        (ys >= arch_y) | (((xs - cx) / inner_half) ** 2 + ((ys - arch_y) / inner_half) ** 2 <= 1)
    )
    inner &= ys < H - 1
    c[frame] = fr[3]
    c[frame & (xs > cx)] = fr[2]
    c[inner] = wr[2]
    c[inner & (xs.astype(int) % 3 == 0)] = wr[1]
    c[inner & (np.abs(ys - (H * 0.45)) < 0.6)] = wr[0]
    c[inner & (np.abs(ys - (H * 0.8)) < 0.6)] = wr[0]
    put(c, int(cx + inner_half - 2), int(H * 0.62), fr[4])
    return outline(c, ctx.palette)


class HouseParams(Params):
    windows: bool = Field(True, description="windows either side of the door")
    chimney: bool = Field(True)
    walls: Literal["plaster", "wood", "stone"] = Field("plaster")


@register(
    "house",
    "object",
    HouseParams,
    colors=[("roof", "#b8483a"), ("walls", "#e6d6b0"), ("trim", "#7a5230")],
    doc="Front-facing village house with roof, door and windows. Use 2x2 or 3x3.",
)
def house(ctx: GenContext, p: HouseParams, w: int, h: int) -> np.ndarray:
    T = ctx.T
    c = _canvas(ctx, w, h)
    H, W = c.shape
    rr, wr, tr = ctx.ramp(0), ctx.ramp(1), ctx.ramp(2)
    glass = ctx.palette.ramp("#6ec3e8")
    roof_bottom = int(H * 0.52)
    wx0, wx1 = 2, W - 3
    # Walls
    if p.walls == "plaster":
        c[roof_bottom : H - 1, wx0 : wx1 + 1] = wr[3]
        f = Noise(ctx.rng("plaster"), T, cells=4).tile()
        speck = np.tile(f, (h, w)) > 0.68
        region = np.zeros_like(speck)
        region[roof_bottom : H - 1, wx0 : wx1 + 1] = True
        c[speck & region] = wr[2]
        rect(c, wx0, roof_bottom, wx0, H - 2, tr[2])
        rect(c, wx1, roof_bottom, wx1, H - 2, tr[1])
    elif p.walls == "wood":
        ys, xs = np.mgrid[0:H, 0:W]
        planks = np.where(ys % 3 == 2, wr[1], np.where(ys % 3 == 0, wr[3], wr[2]))
        c[roof_bottom : H - 1, wx0 : wx1 + 1] = planks[roof_bottom : H - 1, wx0 : wx1 + 1]
    else:
        wall_ctx = GenContext(ctx.T, ctx.seed, ctx.key + ":walls", ctx.palette, [ctx.colors[1]])
        stone = wall(wall_ctx, WallParams(style="brick"), w, h)
        c[roof_bottom : H - 1, wx0 : wx1 + 1] = stone[roof_bottom : H - 1, wx0 : wx1 + 1]
    rect(c, wx0, H - 2, wx1, H - 2, wr[1])
    rect(c, wx0, roof_bottom, wx1, roof_bottom + 1, wr[0])
    # Door
    dw = max(4, T // 3)
    dx0 = W // 2 - dw // 2
    dy0 = H - 2 - int(T * 0.6)
    rect(c, dx0 - 1, dy0 - 1, dx0 + dw, H - 2, tr[0])
    rect(c, dx0, dy0, dx0 + dw - 1, H - 2, tr[2])
    for x in range(dx0 + 1, dx0 + dw, 2):
        rect(c, x, dy0 + 1, x, H - 2, tr[1])
    put(c, dx0 + dw - 2, dy0 + int(T * 0.3), tr[4])
    # Windows
    if p.windows and W >= 2 * T:
        ws = max(3, T // 4)
        wy = roof_bottom + max(3, (H - roof_bottom) // 4)
        for wx in (wx0 + (dx0 - wx0) // 2 - ws // 2, dx0 + dw + (wx1 - dx0 - dw) // 2 - ws // 2 + 1):
            rect(c, wx - 1, wy - 1, wx + ws, wy + ws, tr[1])
            rect(c, wx, wy, wx + ws - 1, wy + ws - 1, glass[2])
            put(c, wx, wy, glass[4])
            rect(c, wx - 1, wy + ws, wx + ws, wy + ws, tr[3])
    # Roof: trapezoid of shingles with a dark eave and light ridge
    shingles = _shingles(H, W, T, rr)
    ys, xs = grid(H, W)
    inset = (roof_bottom - ys) / roof_bottom * (W * 0.18)
    roof_mask = (ys >= 1) & (ys < roof_bottom + 1) & (xs >= 0.5 + inset) & (xs <= W - 0.5 - inset)
    c[roof_mask] = shingles[roof_mask]
    side = roof_mask & ((xs < 1.5 + inset) | (xs > W - 1.5 - inset))
    c[side & (xs < W / 2)] = rr[3]
    c[side & (xs > W / 2)] = rr[1]
    c[roof_mask & (ys >= roof_bottom)] = rr[0]
    c[roof_mask & (ys < 2)] = rr[4]
    if p.chimney:
        stone = ctx.palette.ramp("#8a8494")
        cw = max(3, T // 4)
        cx0 = int(W * 0.68)
        ctop, cbot = 1, int(roof_bottom * 0.45)
        rect(c, cx0, ctop, cx0 + cw - 1, cbot, stone[2])
        rect(c, cx0, ctop, cx0, cbot, stone[3])
        rect(c, cx0 - 1, ctop, cx0 + cw, ctop + 1, stone[3])
        rect(c, cx0, ctop + 1, cx0 + cw - 1, ctop + 1, stone[1])
    c[0, :] = np.where(c[0, :] > 0, c[0, :], 0)
    return outline(c, ctx.palette)
