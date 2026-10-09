"""Object tiles drawn on a transparent canvas of (height*T, width*T).

Signature: fn(ctx, params, width, height) -> uint8[H, W]. Multi-tile objects
are drawn whole and sliced into tiles by the builder. Shapes keep a 1px margin
so the selective outline fits inside the canvas.

Objects use the 3/4 top-down view of 16-bit JRPGs: you see the front of a thing
and its top (chest lids, barrel heads, wall caps, roofs), light comes from the
upper left, and everything standing on the ground casts a translucent `SHADOW`
down and to the right. Shadows are added after the outline so they never get one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..draw import (
    cylinder_levels,
    drop_shadow,
    ellipse_height,
    grid,
    ground_shadow,
    outline,
    put,
    rect,
    shade,
    shift,
)
from ..noise import Noise
from ..registry import Params, register

STONE = "#8a8494"
GLASS = "#6ec3e8"


def _canvas(ctx: GenContext, w: int, h: int) -> np.ndarray:
    return np.zeros((h * ctx.T, w * ctx.T), dtype=np.uint8)


def _trunk(c: np.ndarray, cx: float, top: int, bottom: int, width: int, ramp: list[int]) -> None:
    """A shaded trunk with roots flaring out at the base."""
    x0 = int(round(cx - width / 2))
    x1 = x0 + width - 1
    xs = np.arange(x0, x1 + 1)
    c[top : bottom + 1, x0 : x1 + 1] = cylinder_levels(xs + 0.5, x0, x1 + 1, ramp[1:4])[None, :]
    put(c, x0 - 1, bottom, ramp[2])
    put(c, x1 + 1, bottom, ramp[1])
    put(c, x1 + 1, bottom - 1, ramp[1])


def _top_faces(mask: np.ndarray, depth: int) -> np.ndarray:
    """The upper `depth` pixels of each column of a mask: the part of a solid seen from above."""
    top = np.zeros_like(mask)
    run = mask & ~shift(mask, 1, 0, False)
    for _ in range(depth):
        top |= run
        run = shift(run, 1, 0, False) & mask
    return top


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
    base = H - 3
    _trunk(c, W / 2, int(H * 0.5), base, max(2, W // 6), tr)
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
    # Seen from slightly above, the underside of the crown sinks into shadow.
    under = mask & ~shift(mask, -max(2, H // 12), 0, False)
    canopy[under & (canopy > fr[1])] = fr[1]
    canopy[mask & ~shift(mask, -1, 0, False)] = fr[0]
    c = np.where(mask, canopy, c).astype(np.uint8)
    # Shadow where the canopy hangs over the trunk.
    bark = (c == tr[1]) | (c == tr[2]) | (c == tr[3])
    c[bark & shift(mask, 1, 0, False)] = tr[0]
    c[bark & shift(mask, 2, 0, False) & (c != tr[0])] = tr[1]
    if p.fruit > 0:
        a = ctx.ramp(2)
        ys, xs = np.nonzero(mask & (canopy >= fr[2]))
        n = int(len(ys) * p.fruit * 0.06)
        for i in rng.choice(len(ys), size=min(n, len(ys)), replace=False):
            put(c, xs[i], ys[i], a[2])
            put(c, xs[i], ys[i] - 1, a[4]) if mask[max(ys[i] - 1, 0), xs[i]] else None
    c = outline(c, ctx.palette)
    return ground_shadow(c, W / 2 + W * 0.06, base + 0.5, W * 0.34, max(1.6, H * 0.07))


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
    base = H - 3
    trunk_h = max(2, H // 7)
    _trunk(c, W / 2, base - trunk_h, base, max(2, W // 7), tr)
    ys, xs = grid(H, W)
    cx = W / 2
    top, bottom = 1.0, base - trunk_h + 1
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
        # The skirt of each tier is its underside, seen from above: dark.
        tier = np.where(rel > 0.8, nr[1], tier)
        tier = np.where(rel > 0.92, nr[0], tier)
        c = np.where(inside, tier, c).astype(np.uint8)
        if p.snow:
            s = ctx.palette.ramp("#eef4fa")
            cap = inside & (rel < 0.4) & (xs < cx + 1)
            c[cap] = s[3]
            c[cap & (rel < 0.15)] = s[4]
            c[inside & (rel >= 0.4) & (rel < 0.5) & (xs < cx - 1)] = s[2]
    c = outline(c, ctx.palette)
    return ground_shadow(c, W / 2 + W * 0.08, base + 0.5, W * 0.36, max(1.6, H * 0.06))


class BushParams(Params):
    berries: float = Field(0.0, ge=0.0, le=1.0, description="berry/flower amount (accent color)")


@register("bush", "object", BushParams, colors=[("leaves", "#4c9a3e"), ("berry", "#c83a5a")], doc="Low round shrub.")
def bush(ctx: GenContext, p: BushParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    rng = ctx.rng("shape")
    base_y = H - 3
    R = W * 0.36
    blobs = [(W / 2, base_y - R * 0.75, R, R * 0.8)]
    for dx in (-0.55, 0.55):
        blobs.append((W / 2 + dx * R, base_y - R * 0.55, R * 0.6, R * 0.55, 0.85))
    blobs.append((W / 2 + rng.uniform(-0.2, 0.2) * R, base_y - R * 1.2, R * 0.55, R * 0.5, 0.9))
    height = ellipse_height(H, W, blobs)
    mask = height > 0.05
    mask[base_y + 1 :] = False
    c = shade(height, mask, r[1:5], dither=0.7)
    c[mask & ~shift(mask, -1, 0, False)] = r[0]
    if p.berries > 0:
        a = ctx.ramp(1)
        ys, xs = np.nonzero(mask & (c > r[1]))
        for i in rng.choice(len(ys), size=int(len(ys) * p.berries * 0.08), replace=False):
            c[ys[i], xs[i]] = a[3]
            put(c, xs[i] + 1, ys[i], a[1]) if mask[ys[i], min(xs[i] + 1, W - 1)] else None
    c = outline(c, ctx.palette)
    return ground_shadow(c, W / 2 + W * 0.08, base_y + 0.5, R * 1.15, max(1.5, H * 0.07))


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
    base = H - 3
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
    # A flat-ish lit top plane over a darker front face, the way rocks read in 3/4 view.
    top = _top_faces(mask, max(2, int(R * 0.45)))
    c[top & (c < r[3])] = r[3]
    c[mask & ~shift(mask, 1, 0, False)] = r[4]
    c[mask & ~top & shift(top, 1, 0, False)] = r[2]
    c[mask & ~shift(mask, -1, 0, False)] = r[0]
    ys, xs = np.nonzero(mask & ~top)
    if len(ys):
        i = rng.integers(len(ys))
        x, y = xs[i], ys[i]
        for _ in range(int(R)):
            if mask[y, x]:
                c[y, x] = r[0]
            x = int(np.clip(x + rng.choice([-1, 0, 1]), 0, W - 1))
            y += 1
            if y >= H:
                break
    if p.moss > 0:
        m = ctx.ramp(1)
        f = Noise(ctx.rng("moss"), max(W, H), cells=4).tile()[:H, :W]
        c[top & (f < p.moss * 0.8 + 0.1)] = m[2]
        c[top & (f < p.moss * 0.4)] = m[3]
    c = outline(c, ctx.palette)
    return ground_shadow(c, W / 2 + R * 0.25, base + 0.5, R * 1.3, max(1.5, H * 0.07))


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
    shadow = np.zeros_like(c, dtype=bool)
    for _ in range(p.count * w * h):
        x, y = int(rng.integers(2, W - 2)), int(rng.integers(2, H - 3))
        c[y + 1, x] = sr[2]
        put(c, x, y + 2, sr[1])
        for dx, dy in ((0, -1), (-1, 0), (1, 0)):
            put(c, x + dx, y + dy, pr[2])
        c[y, x] = pr[4]
        if x + 1 < W and y + 2 < H:
            shadow[y + 2, x + 1] = True
    return drop_shadow(c, shadow)


class ChestParams(Params):
    open: bool = Field(False, description="draw the chest opened")


@register(
    "chest",
    "object",
    ChestParams,
    limit=3,
    colors=[("wood", "#9a5a2e"), ("metal", "#e0b640")],
    doc="Treasure chest with metal bands, seen from above and in front.",
)
def chest(ctx: GenContext, p: ChestParams, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, mr = ctx.ramp(0), ctx.ramp(1)
    x0, x1 = 2, W - 3
    y1 = H - 3
    y_seam = int(H * 0.58)  # top of the front face of the box
    y_top = int(H * 0.3)  # back edge of the lid's top face
    y_lid = int(H * 0.46)  # front edge of the lid's top face
    bands = (x0 + 2, x1 - 2)
    if p.open:
        # Lid swung back: its inside faces us above the box.
        rect(c, x0, y_top - 3, x1, y_lid - 3, wr[1])
        rect(c, x0, y_top - 3, x1, y_top - 3, wr[2])
        rect(c, x0 + 1, y_lid - 3, x1 - 1, y_lid - 3, wr[0])
        # The open box from above: rim, dark interior, glinting treasure.
        rect(c, x0, y_lid - 2, x1, y_seam, wr[3])
        rect(c, x0 + 1, y_lid - 1, x1 - 1, y_seam - 1, wr[0])
        for i, x in enumerate(range(x0 + 2, x1 - 1)):
            put(c, x, y_seam - 1 - (i % 2), mr[3] if i % 3 else mr[4])
        put(c, (x0 + x1) // 2, y_seam - 2, mr[4])
    else:
        # Lid: top face (lit, seen from above) then a narrow front face.
        rect(c, x0, y_top, x1, y_lid, wr[3])
        rect(c, x0 + 1, y_top, x1 - 1, y_top, wr[4])
        rect(c, x0, y_lid + 1, x1, y_seam - 1, wr[2])
        rect(c, x0, y_lid + 1, x1, y_lid + 1, wr[4])
        for bx in bands:
            rect(c, bx, y_top, bx, y_lid, mr[3])
            put(c, bx, y_top, mr[4])
            rect(c, bx, y_lid + 1, bx, y_seam - 1, mr[2])
    # Box front face.
    rect(c, x0, y_seam, x1, y1, wr[2])
    rect(c, x0, y_seam, x1, y_seam, wr[0])
    rect(c, x0, y1, x1, y1, wr[1])
    rect(c, x1, y_seam + 1, x1, y1, wr[1])
    for bx in bands:
        rect(c, bx, y_seam + 1, bx, y1, mr[2])
    lx = (x0 + x1) // 2
    rect(c, lx, y_seam - (0 if p.open else 1), lx + 1, y_seam + 2, mr[3])
    put(c, lx, y_seam + 1, mr[0])
    put(c, lx, y_seam - (0 if p.open else 1), mr[4])
    c = outline(c, ctx.palette)
    return ground_shadow(c, W / 2 + 2, y1 + 1, (x1 - x0) / 2 + 1.5, 1.8)


@register("sign", "object", Params, colors=[("wood", "#a8743e")], doc="Wooden signpost.", limit=3)
def sign(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    r = ctx.ramp(0)
    px = W // 2 - 1
    base = H - 3
    rect(c, px, H // 2, px, base, r[2])
    rect(c, px + 1, H // 2, px + 1, base, r[1])
    x0, x1, y0, y1 = 2, W - 3, int(H * 0.2), int(H * 0.58)
    rect(c, x0, y0 - 1, x1, y0 - 1, r[4])  # the board's top edge, seen from above
    rect(c, x0, y0, x1, y1, r[3])
    rect(c, x0, y0, x1, y0, r[2])
    rect(c, x1, y0, x1, y1, r[2])
    rect(c, x0, y1, x1, y1, r[1])
    for y in range(y0 + 2, y1 - 1, 2):
        rect(c, x0 + 2, y, x1 - 3, y, r[1])
    rect(c, px, y1 + 1, px + 1, y1 + 1, r[0])  # the board shades the post
    c = outline(c, ctx.palette)
    return ground_shadow(c, px + 2.5, base + 0.5, 3.0, 1.4)


@register(
    "barrel",
    "object",
    Params,
    colors=[("wood", "#9a6236"), ("bands", "#5c5866")],
    doc="Wooden barrel with metal hoops; its lid is visible from above.",
)
def barrel(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, br = ctx.ramp(0), ctx.ramp(1)
    ys, xs = grid(H, W)
    cx = W / 2
    half = W / 2 - 3
    lid_ry = max(1.5, H * 0.1)
    y_lid = 2 + lid_ry  # center of the lid ellipse
    y1 = H - 3
    t = np.clip((ys - y_lid) / max(y1 - y_lid, 1), 0, 1)
    bulge = half + 0.8 * np.sin(np.pi * t)
    body = (ys >= y_lid) & (ys <= y1 + 0.5) & (np.abs(xs - cx) <= bulge)
    lid = ((xs - cx) / half) ** 2 + ((ys - y_lid) / lid_ry) ** 2 <= 1.0
    staves = cylinder_levels(xs, cx - bulge, cx + bulge, wr[1:5])
    c[body] = staves[body]
    c[body & (np.abs((xs - cx) / np.maximum(bulge, 1)) * 3 % 1 < 0.12)] = wr[1]
    for frac in (0.18, 0.78):
        yb = int(y_lid + frac * (y1 - y_lid))
        hoop = body & (ys.astype(int) == yb)
        c[hoop] = cylinder_levels(xs, cx - bulge, cx + bulge, br[1:4])[hoop]
        c[body & (ys.astype(int) == yb + 1)] = wr[0]
    # The lid: a light ellipse with a dark rim at the front and planks across.
    c[lid] = wr[3]
    c[lid & (xs.astype(int) % 3 == 0)] = wr[2]
    rim = lid & ~(((xs - cx) / (half - 1)) ** 2 + ((ys - y_lid) / (lid_ry - 0.6)) ** 2 <= 1.0)
    c[rim] = br[3]
    c[rim & (ys > y_lid)] = br[1]
    c = outline(c, ctx.palette)
    return ground_shadow(c, cx + 1.5, y1 + 0.5, half + 1.5, 1.6)


N, E, S, W = 1, 2, 4, 8  # connection bits for linked objects (fences)


class FenceParams(Params):
    gate: bool = Field(True, description="also draw a gate tile for east-west runs")


@register(
    "fence",
    "object",
    FenceParams,
    colors=[("wood", "#b08048")],
    connected=True,
    doc="Wooden rail fence that links up: straight runs, corners, T-joins and a gate. "
    "The preview builds fenced enclosures with it; size is always 1x1.",
)
def fence(ctx: GenContext, p: FenceParams, mask: int, gate: bool = False) -> np.ndarray:
    """One fence tile. `mask` says which neighbors (N/E/S/W bits) the fence continues into.

    East-west runs are seen from the front: two rails between posts. North-south
    runs recede into the screen, so their rails are seen from above, as a narrow
    strip joining one post to the next. `gate` draws a gate panel across an
    east-west run instead.
    """
    T = ctx.T
    k = max(1, T // 16)  # stroke width: rails and posts thicken at 32px
    c = np.zeros((T, T), dtype=np.uint8)
    r = ctx.ramp(0)
    px = T // 2 - k  # the post covers px .. px + 2k - 1
    pr = px + 2 * k - 1
    top, ground = int(T * 0.2), T - 3
    rails = (int(T * 0.33), int(T * 0.58))
    ys, xs = np.mgrid[0:T, 0:T]
    if gate:
        rect(c, 0, top, k - 1, ground, r[2])  # the latch and hinge stiles
        rect(c, T - k, top, T - 1, ground, r[1])
        for y in (top + 1, ground - 1 - 2 * k):
            rect(c, k, y, T - 1 - k, y + k - 1, r[4])
            rect(c, k, y + k, T - 1 - k, y + 2 * k - 1, r[2])
        step = 3 * k
        for x in range(k + 1, T - k - 1, step):
            rect(c, x, top + 1 + 2 * k, x + k - 1, ground - 2 - 2 * k, r[3])
            rect(c, x + k, top + 1 + 2 * k, x + k, ground - 2 - 2 * k, r[1])
        if T >= 32:  # a diagonal brace only fits at 32px
            for i in range(T - 2 * k - 2):
                y = int(round(ground - 2 - 2 * k - i * (ground - top - 4 - 4 * k) / (T - 2 * k - 3)))
                rect(c, k + 1 + i, y, k + 1 + i, y + k - 1, r[2])
        c = outline(c, ctx.palette)
        return drop_shadow(c, (ys >= T - 2) & (xs % 2 == 0))
    # North-south rails: a strip seen from above, running out of the tile past the post.
    for bit, y0, y1 in ((N, 0, rails[1]), (S, rails[0], T - 1)):
        if mask & bit:
            rect(c, px, y0, px + k - 1, y1, r[3])
            rect(c, px + k, y0, pr, y1, r[2])
            rect(c, px, y0, px, y1, r[4])
    # East-west rails, seen from the front: lit top edge, face, shaded underside.
    for bit, x0, x1 in ((W, 0, px - 1), (E, pr + 1, T - 1)):
        if mask & bit:
            for y in rails:
                rect(c, x0, y, x1, y + k - 1, r[4])
                rect(c, x0, y + k, x1, y + 2 * k - 1, r[2])
                rect(c, x0, y + 2 * k, x1, y + 2 * k, r[1])
    rect(c, px, top, pr, ground, r[2])
    rect(c, px, top, px + k - 1, ground, r[3])
    rect(c, pr, top, pr, ground, r[1])
    rect(c, px, top - 1, pr, top - 1, r[4])  # post cap
    rect(c, px, ground, pr, ground, r[1])  # foot, darker where it meets the ground
    if mask & E:
        for y in rails:
            rect(c, pr + 1, y + k, pr + 1, y + 2 * k - 1, r[0])  # the rail tucks into the post's shadow
    c = outline(c, ctx.palette)
    # Shadows fall down and to the right: a broken line under rails, a strip beside receding runs.
    cast = (ys >= ground) & (ys <= ground + 1) & (xs >= px + 1) & (xs <= pr + 2)
    if mask & W:
        cast |= (ys >= T - 2) & (xs < px) & (xs % 2 == 0)
    if mask & E:
        cast |= (ys >= T - 2) & (xs > pr) & (xs % 2 == 0)
    if mask & N:
        cast |= (xs >= pr + 1) & (xs <= pr + 2) & (ys < top) & (ys % 2 == 0)
    if mask & S:
        cast |= (xs >= pr + 1) & (xs <= pr + 2) & (ys > ground) & (ys % 2 == 0)
    return drop_shadow(c, cast)


class WallParams(Params):
    style: Literal["stone", "brick"] = Field("stone", description="large stone blocks or small bricks")
    top: bool = Field(
        True,
        description="draw the wall's top surface along its upper edge (3/4 view); "
        "off for a bare face that also tiles vertically",
    )


@register(
    "wall",
    "object",
    WallParams,
    colors=[("stone", "#8a8494")],
    doc="Wall segment: a top cap seen from above over a block face; tiles horizontally "
    "(and vertically with top=false). Castles, dungeons, town walls.",
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
    if p.top:
        cap = T // 2 if h >= 2 else T // 4
        # Flat capstones seen from above: staggered slabs, each a slightly different shade.
        sw, sh = max(4, T // 4), max(2, cap // 2)
        row = ys // sh
        sx = (xs + (row % 2) * (sw // 2)) % sw
        slab = row * 97 + ((xs + (row % 2) * (sw // 2)) % W) // sw
        tones = ctx.rng("cap").choice([r[3], r[3], r[4]], size=int(slab.max()) + 1)
        top = ys < cap
        c[top] = tones[slab][top]
        c[top & (sx == sw - 1)] = r[2]
        c[top & (ys % sh == sh - 1)] = r[2]
        c[ys == 0] = r[2]  # far edge of the cap
        c[ys == cap - 1] = r[4]  # lit front lip
        c[ys == cap] = r[0]  # the lip's shadow on the face below
        c[ys == cap + 1] = np.where(c[ys == cap + 1] > r[1], r[1], c[ys == cap + 1])
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
    limit=3,
    colors=[("wood", "#8a5430"), ("frame", "#6e6878")],
    doc="Arched wooden door set back in a stone frame (transparent outside the frame).",
)
def door(ctx: GenContext, p: Params, w: int, h: int) -> np.ndarray:
    c = _canvas(ctx, w, h)
    H, W = c.shape
    wr, fr = ctx.ramp(0), ctx.ramp(1)
    ys, xs = grid(H, W)
    cx = W / 2
    half = W / 2 - 1
    arch_y = half + 1
    bottom = H - 2

    def arch(hw: float) -> np.ndarray:
        m = (np.abs(xs - cx) <= hw) & ((ys >= arch_y) | (((xs - cx) / hw) ** 2 + ((ys - arch_y) / hw) ** 2 <= 1))
        return m & (ys < bottom)

    frame, inner = arch(half), arch(half - 2)
    c[frame] = fr[3]
    c[frame & (xs > cx)] = fr[2]
    c[frame & ~shift(frame, 1, 0, False)] = fr[4]  # top of the arch catches the light
    c[inner] = wr[2]
    c[inner & (xs.astype(int) % 3 == 0)] = wr[1]
    c[inner & (np.abs(ys - (H * 0.45)) < 0.6)] = wr[0]
    c[inner & (np.abs(ys - (H * 0.8)) < 0.6)] = wr[0]
    # The door is recessed: the frame shades its top and left edges.
    c[inner & ~shift(inner, 1, 0, False)] = wr[0]
    c[inner & ~shift(inner, 2, 0, False) & (xs < cx)] = wr[0]
    c[inner & ~shift(inner, 0, 1, False)] = wr[0]
    put(c, int(cx + half - 4), int(H * 0.62), fr[4])
    # Stone threshold in front of the door.
    step = (np.abs(xs - cx) <= half - 1) & (ys >= bottom) & (ys < bottom + 1)
    c[step] = fr[4]
    c[step & (xs > cx + half - 3)] = fr[3]
    return outline(c, ctx.palette)


class HouseParams(Params):
    windows: bool = Field(True, description="windows either side of the door")
    chimney: bool = Field(True)
    walls: Literal["plaster", "wood", "stone"] = Field("plaster")
    roof: Literal["hip", "gable"] = Field(
        "hip", description="hip: sloped on all sides with lit/shaded side slopes; gable: one front slope"
    )


def _house_roof(c: np.ndarray, T: int, top: int, eave: int, x0: int, x1: int, rr: list[int], style: str) -> np.ndarray:
    """Draw the roof seen from above and in front; returns its mask."""
    H, W = c.shape
    ys, xs = grid(H, W)
    span = eave - top
    region = (ys >= top) & (ys < eave + 1) & (xs >= x0) & (xs < x1 + 1)
    front = _shingles(H, W, T, rr)
    if style == "gable":
        mask = region
        c[mask] = front[mask]
        c[mask & (xs < x0 + 1)] = rr[3]  # verge boards
        c[mask & (xs >= x1)] = rr[1]
    else:
        rx0, rx1 = x0 + (x1 - x0) * 0.24, x1 - (x1 - x0) * 0.24
        d = span * 0.45
        t = np.clip((ys - top) / span, 0, 1)
        chamfer = np.clip((ys - top) / d, 0, 1)
        mask = region & (xs >= rx0 - (rx0 - x0) * chamfer) & (xs <= rx1 + (x1 - rx1) * chamfer)
        fl = rx0 + (x0 - rx0) * t  # front slope edges (the hips)
        fr = rx1 + (x1 - rx1) * t
        lit = _shingles(H, W, T, [rr[1], rr[2], rr[3], rr[4], rr[4]])
        dark = _shingles(H, W, T, [rr[0], rr[0], rr[1], rr[2], rr[2]])
        c[mask] = front[mask]
        left, right = mask & (xs < fl), mask & (xs > fr)
        c[left] = lit[left]
        c[right] = dark[right]
        c[mask & (np.abs(xs - fl) < 0.75)] = rr[4]
        c[mask & (np.abs(xs - fr) < 0.75)] = rr[1]
    # Ridge cap along the top, a dark fascia along the eave.
    c[mask & (ys < top + 1)] = rr[4]
    c[mask & (ys >= top + 1) & (ys < top + 2)] = rr[3]
    c[mask & (ys >= eave)] = rr[0]
    return mask


@register(
    "house",
    "object",
    HouseParams,
    limit=3,
    colors=[("roof", "#b8483a"), ("walls", "#e6d6b0"), ("trim", "#7a5230")],
    doc="Village house in 3/4 view: a big roof seen from above over a front wall with door and windows. "
    "Use 2x2 or 3x3.",
)
def house(ctx: GenContext, p: HouseParams, w: int, h: int) -> np.ndarray:
    T = ctx.T
    c = _canvas(ctx, w, h)
    H, W = c.shape
    rr, wr, tr = ctx.ramp(0), ctx.ramp(1), ctx.ramp(2)
    glass, stone = ctx.palette.ramp(GLASS), ctx.palette.ramp(STONE)
    eave = int(H * 0.54)
    base = H - 3  # last wall row; the rows below hold the doorstep and shadow
    wx0, wx1 = 3, W - 4
    wall_rows = slice(eave, base + 1)
    wall_cols = slice(wx0, wx1 + 1)
    # Walls
    if p.walls == "plaster":
        c[wall_rows, wall_cols] = wr[3]
        f = Noise(ctx.rng("plaster"), T, cells=4).tile()
        speck = np.tile(f, (h, w)) > 0.68
        region = np.zeros_like(speck)
        region[wall_rows, wall_cols] = True
        c[speck & region] = wr[2]
        rect(c, wx0, eave, wx0, base, tr[3])  # corner posts of the timber frame
        rect(c, wx1, eave, wx1, base, tr[1])
    elif p.walls == "wood":
        ys, xs = np.mgrid[0:H, 0:W]
        planks = np.where(ys % 3 == 2, wr[1], np.where(ys % 3 == 0, wr[3], wr[2]))
        c[wall_rows, wall_cols] = planks[wall_rows, wall_cols]
        rect(c, wx1, eave, wx1, base, wr[1])
    else:
        wall_ctx = GenContext(ctx.T, ctx.seed, ctx.key + ":walls", ctx.palette, [ctx.colors[1]])
        blocks = wall(wall_ctx, WallParams(style="brick", top=False), w, h)
        c[wall_rows, wall_cols] = blocks[wall_rows, wall_cols]
    # Stone footing along the bottom of the wall.
    foot = max(2, T // 8)
    rect(c, wx0, base - foot + 1, wx1, base, stone[2])
    rect(c, wx0, base - foot + 1, wx1, base - foot + 1, stone[3])
    for x in range(wx0 + 2, wx1, max(4, T // 4)):
        rect(c, x, base - foot + 2, x, base, stone[1])
    # The eave throws a band of shadow across the top of the wall.
    for k, lv in enumerate((0, 1)):
        row = c[eave + 1 + k, wx0 : wx1 + 1]
        c[eave + 1 + k, wx0 : wx1 + 1] = np.where(row >= wr[2], wr[lv], row)
    # Door, set into the wall with a step in front.
    dw = max(4, T // 3)
    dx0 = W // 2 - dw // 2
    dy0 = base - int(T * 0.6)
    rect(c, dx0 - 1, dy0 - 1, dx0 + dw, base, tr[0])
    rect(c, dx0, dy0, dx0 + dw - 1, base, tr[2])
    for x in range(dx0 + 1, dx0 + dw, 2):
        rect(c, x, dy0 + 1, x, base, tr[1])
    rect(c, dx0, dy0, dx0 + dw - 1, dy0, tr[0])  # recess shadow under the lintel
    rect(c, dx0, dy0, dx0, base, tr[1])
    put(c, dx0 + dw - 2, dy0 + int(T * 0.3), tr[4])
    rect(c, dx0 - 1, base + 1, dx0 + dw, base + 1, stone[4])
    rect(c, dx0 + dw - 1, base + 1, dx0 + dw, base + 1, stone[2])
    # Windows with a lit sill and a shadow under it.
    if p.windows and W >= 2 * T:
        ws = max(3, T // 4)
        wy = eave + max(4, (base - eave) // 4)
        for wx in (wx0 + (dx0 - wx0) // 2 - ws // 2, dx0 + dw + (wx1 - dx0 - dw) // 2 - ws // 2 + 1):
            rect(c, wx - 1, wy - 1, wx + ws, wy + ws, tr[1])
            rect(c, wx, wy, wx + ws - 1, wy + ws - 1, glass[2])
            rect(c, wx, wy, wx + ws - 1, wy, glass[1])  # recessed: the top of the pane is in shade
            put(c, wx + 1, wy + 1, glass[4])
            put(c, wx + ws - 1, wy + ws - 1, glass[3])
            rect(c, wx - 1, wy + ws, wx + ws, wy + ws, tr[4])
            rect(c, wx, wy + ws + 1, wx + ws, wy + ws + 1, wr[1])
    # Roof, overhanging the walls on both sides.
    roof_mask = _house_roof(c, T, 1, eave, wx0 - 2, wx1 + 2, rr, p.roof)
    if p.chimney:
        cw = max(3, T // 4)
        cx0 = int(W * 0.66)
        ctop, cbot = 2, int(eave * 0.5)
        lid = max(2, T // 8)
        rect(c, cx0, ctop, cx0 + cw - 1, cbot, stone[2])
        rect(c, cx0, ctop, cx0, cbot, stone[3])
        rect(c, cx0 + cw - 1, ctop, cx0 + cw - 1, cbot, stone[1])
        # Its top, seen from above: a rim around a sooty opening.
        rect(c, cx0 - 1, ctop - 1, cx0 + cw, ctop + lid - 1, stone[4])
        rect(c, cx0, ctop, cx0 + cw - 1, ctop + lid - 2, stone[0])
        rect(c, cx0 - 1, ctop + lid, cx0 + cw, ctop + lid, stone[1])
        # And its shadow falls on the roof to the lower right.
        sh = np.zeros_like(roof_mask)
        sh[ctop + lid + 1 : cbot + 3, cx0 + cw : cx0 + cw + 2] = True
        sh[cbot + 1 : cbot + 3, cx0 + 1 : cx0 + cw] = True
        c[sh & roof_mask] = rr[0]
    c[0, :] = 0
    c = outline(c, ctx.palette)
    # The house's own shadow: down the right side and along the ground in front.
    ys, xs = np.mgrid[0:H, 0:W]
    cast = ((xs >= wx1 + 2) & (xs <= wx1 + 3) & (ys >= eave + 2) & (ys <= base + 1)) | (
        (ys >= base + 1) & (ys <= base + 2) & (xs >= wx0 + 2) & (xs <= wx1 + 3)
    )
    return drop_shadow(c, cast)


class WellParams(Params):
    roof: bool = Field(True, description="a small shingled roof on posts, with a rope and bucket (needs height 2)")
    water: bool = Field(True, description="water inside; off for a dry well")


@register(
    "well",
    "object",
    WellParams,
    limit=2,
    colors=[("stone", "#8a8494"), ("wood", "#8a5a34"), ("water", "#3a78c8")],
    doc="Round stone well seen from above and in front. 1x1 is a bare well; 1x2 or 2x2 adds a roof.",
)
def well(ctx: GenContext, p: WellParams, w: int, h: int) -> np.ndarray:
    T = ctx.T
    c = _canvas(ctx, w, h)
    H, W = c.shape
    sr, wr, ar = ctx.ramp(0), ctx.ramp(1), ctx.ramp(2)
    ys, xs = grid(H, W)
    cx = W / 2
    rx = min(W / 2 - 2, T * 0.42 * max(1, w * 0.8))
    ry = max(2.0, rx * 0.45)
    base = H - 3
    body_h = max(4, int(T * 0.3))
    yc = base - ry - body_h  # center of the rim ellipse
    dx = (xs - cx) / rx
    curve = ry * np.sqrt(np.clip(1 - dx**2, 0, 1))
    # Front of the cylinder: staggered stone courses, lit on the left.
    body = (np.abs(dx) <= 1) & (ys >= yc) & (ys <= base - ry + curve)
    course = max(3, T // 5)
    row = ((ys - yc - curve) // course).astype(int)
    blk = max(3, T // 4)
    joint_x = ((xs + (row % 2) * (blk // 2)) % blk) < 1
    lit = cylinder_levels(xs, cx - rx, cx + rx, sr[1:5])
    c[body] = lit[body]
    c[body & ((ys - yc - curve) % course < 1)] = sr[0]
    c[body & joint_x] = sr[1]
    # The rim from above, and the opening: the far inner wall in shadow, then water.
    rim = dx**2 + ((ys - yc) / ry) ** 2 <= 1
    hole = ((xs - cx) / (rx - 2)) ** 2 + ((ys - yc) / (ry - 1)) ** 2 <= 1
    c[rim] = sr[3]
    c[rim & (ys < yc - ry * 0.5)] = sr[4]
    c[rim & ~hole & (ys > yc + ry * 0.4)] = sr[2]
    c[hole] = sr[0]
    if p.water:
        pool = hole & (ys >= yc - 0.2)
        c[pool] = ar[1]
        c[pool & (np.abs(xs - (cx - rx * 0.3)) < 1) & (ys < yc + 1)] = ar[3]  # glint
    else:
        c[hole & (ys >= yc)] = sr[1]
    if p.roof and h >= 2:
        roof_bottom = int(yc - ry - max(4, T * 0.45))
        post_top = roof_bottom
        for px in (int(cx - rx), int(cx + rx) - 1):
            rect(c, px, post_top, px + 1, int(yc + ry * 0.3), wr[2])
            rect(c, px + 1, post_top, px + 1, int(yc + ry * 0.3), wr[1])
        bar = post_top + 2
        rect(c, int(cx - rx) + 2, bar, int(cx + rx) - 2, bar, wr[3])
        rect(c, int(cx - rx) + 2, bar + 1, int(cx + rx) - 2, bar + 1, wr[1])
        # Rope and bucket hanging over the opening.
        bx = int(cx)
        rect(c, bx, bar + 2, bx, int(yc - 2), wr[4])
        rect(c, bx - 1, int(yc - 2), bx + 1, int(yc - 1), wr[2])
        put(c, bx - 1, int(yc - 2), wr[3])
        # Roof seen from above and in front: a shingled plane with a ridge and a dark eave.
        roof_top = 1
        x0, x1 = 1, W - 2
        shingles = _shingles(H, W, T, wr)
        plane = (ys >= roof_top) & (ys < roof_bottom + 1) & (xs >= x0) & (xs < x1 + 1)
        c[plane] = shingles[plane]
        c[plane & (ys < roof_top + 1)] = wr[4]
        c[plane & (ys >= roof_bottom)] = wr[0]
        c[plane & (xs >= x1)] = wr[1]
        c[plane & (xs < x0 + 1)] = wr[3]
        # The roof shades the posts and bar just below it.
        under = (ys >= roof_bottom + 1) & (ys < roof_bottom + 2) & (c > 0)
        c[under] = np.where(c[under] >= wr[2], wr[1], c[under])
    c = outline(c, ctx.palette)
    return ground_shadow(c, cx + 1.5, base + 0.5, rx + 1, max(1.5, ry * 0.7))
