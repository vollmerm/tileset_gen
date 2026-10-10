"""Seamless ground tiles.

Signature: fn(ctx, params, variant, phase) -> uint8[T, T].

Variants share one periodic base field along the tile border and replace it
with their own field inside (see `_field`), so any two variants meet without a
seam while their interiors differ, which breaks up the 1-tile repeat.
`phase` in [0, 1) is the animation time for animatable terrains; phase 0 is the
static tile. Noise lattice sizes are given for 16px tiles and scale with T.

Each generator also lists decor features: small details (tall grass, stones,
cracks, lily pads) that `decorate` draws onto copies of the base variants to
make rare detail tiles. Features stay off the border, so detail tiles are as
seamless as the variants they are drawn on.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..noise import Noise, contrast, quantize, voronoi
from ..registry import GeneratorInfo, Params, register


def _inner(T: int, ramp: float = 0.3) -> np.ndarray:
    """0 on the tile's 1px border ring, rising smoothly to 1 about `ramp * T` pixels in."""
    i = np.arange(T)
    d = np.minimum(i, T - 1 - i)
    s = np.clip(np.minimum.outer(d, d) / (T * ramp), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _vary(
    shared: np.ndarray,
    own: np.ndarray,
    amount: float = 0.9,
    edge_power: float = 1.0,
    flatten: float = 0.5,
) -> np.ndarray:
    """Keep `shared` on the border and fade to `own` inside.

    The border is the one part every variant repeats, so its contrast is reduced
    by `flatten` to keep the tile grid from showing.
    """
    w = amount * _inner(shared.shape[0]) ** edge_power
    shared = 0.5 + (shared - 0.5) * (1.0 - flatten)
    return shared + (own - shared) * w


def _field(
    ctx: GenContext,
    variant: int,
    cells: int | tuple[int, int],
    octaves: int = 2,
    amount: float = 0.9,
    persistence: float = 0.5,
    t: float = 0.0,
    tcells: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    edge_power: float = 1.0,
    flatten: float = 0.6,
    fine: bool = True,
    key: str = "field",
) -> np.ndarray:
    """Tile-periodic noise that every variant shares at the border, with its own noise inside.

    `amount` is how much of the interior is the variant's own field. Raise
    `edge_power` when the caller amplifies small differences (e.g. ridges), to
    widen the band that stays close to the shared field. `fine` gives the shared
    border twice the lattice resolution (turn it off when the caller turns the
    field into lines, which fine noise would make into a dense mesh). Use distinct
    `key`s for independent fields within one generator.
    """
    T = ctx.T
    k = max(1, T // 16)
    scaled = cells * k if isinstance(cells, int) else (cells[0] * k, cells[1] * k)

    def layer(cells, octaves, *key) -> np.ndarray:
        n = Noise(ctx.rng(*key), T, cells=cells, octaves=octaves, persistence=persistence, tcells=tcells)
        return n.tile(t=t, dx=dx, dy=dy)

    # The shared border uses finer noise: large shapes would repeat visibly, fine texture doesn't.
    if fine:
        border = scaled * 2 if isinstance(scaled, int) else (scaled[0] * 2, scaled[1] * 2)
        shared = layer(border, max(1, octaves - 1), key)
    else:
        shared = layer(scaled, octaves, key)
    return _vary(shared, layer(scaled, octaves, key, variant), amount, edge_power, flatten)


def _px(tile: np.ndarray, x: int, y: int, color: int) -> None:
    """Set a pixel, wrapping around the tile (details that cross the border continue on the far side)."""
    T = tile.shape[0]
    tile[y % T, x % T] = color


def _patch(tile: np.ndarray, x: int, y: int, shadow: int, light: int, u: np.ndarray) -> None:
    """A small irregular shadow blob with a light rim on top (dips, drifts, worn spots)."""
    w = 2 + int(u[0] * 2)
    for dx in range(-w, w + 1):
        _px(tile, x + dx, y, shadow)
        if abs(dx) < w:
            _px(tile, x + dx, y + 1, shadow)
            _px(tile, x + dx, y - 1, light)
    if u[1] < 0.5:
        _px(tile, x + w, y + 1, shadow)


def _interior(rng: np.random.Generator, T: int, n: int, margin: int = 1):
    """`n` random interior points, kept off the tile border so variants stay seamless."""
    xs = rng.integers(margin, T - margin, size=n)
    ys = rng.integers(margin, T - margin, size=n)
    return zip(xs.tolist(), ys.tolist())


def _scatter(ctx: GenContext, variant: int, n: int, margin: int = 1, key: str = "detail", edge: bool = True):
    """About `n` detail points per tile as (x, y, u), with `u` four uniforms for per-point choices.

    `margin` is how far a detail drawn at a point reaches. Points closer than that
    to the border would be cut off by the tile edge, so they come from a stream
    every variant shares and must be drawn with `_px` so they wrap. The variant's
    own points sit one pixel further in, so they never touch the 1px border ring
    and every variant keeps the same border. This keeps the border as busy as the
    interior without breaking seams. With `edge=False` only own points are kept.
    """
    T = ctx.T

    def draw(rng: np.random.Generator):
        xy = rng.integers(0, T, size=(n, 2))
        return [(int(x), int(y), u) for (x, y), u in zip(xy, rng.random((n, 4)))]

    def inside(x: int, y: int, m: int) -> bool:
        return m <= x < T - m and m <= y < T - m

    shared = [pt for pt in draw(ctx.rng(key, "shared")) if edge and not inside(pt[0], pt[1], margin)]
    own = [pt for pt in draw(ctx.rng(key, variant)) if inside(pt[0], pt[1], margin + 1)]
    return shared + own


# --- Decor features -------------------------------------------------------
#
# Signature: feature(tile, ctx, params, rng) -> None, drawing in place. Features
# stay inside a 2px margin so detail tiles keep their base variant's border.
# Fixed hexes are for things a terrain's own colors can't express (stone, foliage,
# ice); the palette merge folds them into nearby colors when the budget is tight.

STONE = "#8b8794"
LEAF = "#4a9a48"
TWIG = "#7a5a3a"
ICE = "#a8d4ec"
CAP = "#c8503a"


def decorate(info: GeneratorInfo, ctx: GenContext, p: Params, tile: np.ndarray, k: int) -> np.ndarray:
    """Detail tile number `k`: one of the generator's decor features drawn onto `tile`."""
    out = tile.copy()
    order = ctx.rng("decor-order").permutation(len(info.decor))
    info.decor[order[k % len(info.decor)]](out, ctx, p, ctx.rng("decor", k))
    return out


def _spot(rng: np.random.Generator, T: int, w: int, h: int, margin: int = 2) -> tuple[int, int]:
    """Top-left corner for a w x h feature kept `margin` pixels inside the tile."""
    return int(rng.integers(margin, T - margin - w + 1)), int(rng.integers(margin, T - margin - h + 1))


def _instances(T: int, base: int = 1) -> int:
    """Small features repeat at 32px so they keep the same density as at 16px."""
    return base * max(1, T // 16)


def _blob(T: int, cx: float, cy: float, rx: float, ry: float, rng: np.random.Generator, rough: float = 0.3):
    """Noisy filled ellipse, clipped to the 2px decor margin."""
    ys, xs = np.mgrid[0:T, 0:T] + 0.5
    n = Noise(rng, T, cells=4).tile()
    m = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 + (n - 0.5) * rough * 2 < 1.0
    keep = np.zeros_like(m)
    keep[2 : T - 2, 2 : T - 2] = True
    return m & keep


def _rim(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top edge, bottom edge) pixels of a mask, for a lit top and shaded underside."""
    above = np.zeros_like(mask)
    above[1:] = mask[:-1]
    below = np.zeros_like(mask)
    below[:-1] = mask[1:]
    return mask & ~above, mask & ~below


def _blades(tile, rng, ramp, cx: int, base: int, w: int, height: int) -> None:
    """A dome-shaped clump of blades standing on row `base`: dark body, striped blades, spiky light tips."""
    T = tile.shape[0]
    for i in range(w):
        x = cx - w // 2 + i
        if not 2 <= x <= T - 3:
            continue
        t = abs(i - (w - 1) / 2) / (w / 2)
        h = max(1, int(round(height * (1.0 - t * t) ** 0.5)) + int(rng.integers(-1, 1)) + (i % 2))
        for j in range(h + 1):
            y = base - j
            if not 2 <= y <= T - 3:
                continue
            if j == 0:
                c = ramp[0]
            elif j == h:
                c = ramp[4] if i % 2 else ramp[3]
            elif j >= h - 1:
                c = ramp[3] if i % 2 else ramp[2]
            else:
                c = ramp[1] if (i + j // 2) % 2 else ramp[2] if j > h // 2 else ramp[0]
            tile[y, x] = c
    for x in (cx - w // 2 - 1, cx + w - w // 2):  # shadow at the clump's feet, longer on the right
        if 2 <= x <= T - 3 and 2 <= base <= T - 3:
            tile[base, x] = ramp[1]


def _tall_grass(tile, ctx, p, rng):
    T = ctx.T
    for _ in range(_instances(T)):
        cx, base = _spot(rng, T, 1, 1, margin=6)
        _blades(tile, rng, ctx.ramp(0), cx, base + 3, int(rng.integers(7, 10)), int(rng.integers(5, 7)))


def _flower_clump(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    a = ctx.ramp(1) if len(ctx.colors) > 1 else ctx.palette.ramp(CAP)
    for _ in range(_instances(T)):
        cx, cy = _spot(rng, T, 1, 1, margin=6)
        offsets = [(dx, dy) for dx in (-3, 0, 3) for dy in (-3, 0, 3)]
        for i in rng.choice(len(offsets), size=int(rng.integers(4, 7)), replace=False):
            dx, dy = offsets[i]
            x = int(np.clip(cx + dx + rng.integers(-1, 2), 3, T - 4))
            y = int(np.clip(cy + dy + (dx // 3) % 2, 3, T - 5))
            tile[y + 2, x] = r[0]
            tile[y + 1, x + 1] = r[1]
            for ox, oy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                tile[y + oy, x + ox] = a[3]
            tile[y, x] = a[4]


def _pebbles(hexv: str | None = None, big: bool = False):
    def feature(tile, ctx, p, rng):
        T, r = ctx.T, ctx.ramp(0)
        s = ctx.palette.ramp(hexv) if hexv else r
        for _ in range(_instances(T, int(rng.integers(1, 3)) + int(big))):
            w = int(rng.integers(3, 5 + int(big)))
            x, y = _spot(rng, T, w + 1, 4)
            tile[y, x + 1 : x + w - 1] = s[4]
            tile[y + 1, x : x + w] = s[3]
            tile[y + 1, x + w - 1] = s[2]
            tile[y + 2, x : x + w] = s[2]
            tile[y + 2, x] = s[3]
            tile[y + 3, x : x + w] = r[0]  # contact shadow, longer on the right
            tile[y + 2, x + w] = r[0]
            tile[y + 3, x + w] = r[1]

    feature.__name__ = "stones"
    return feature


def _mushrooms(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    c, stem = ctx.palette.ramp(CAP), ctx.palette.ramp("#e8dcc0")
    for _ in range(_instances(T, int(rng.integers(2, 4)))):
        x, y = _spot(rng, T, 3, 3)
        tile[y, x : x + 3] = c[2]
        tile[y, x] = c[3]
        tile[y, x + 1] = stem[4]  # white spot
        tile[y + 1, x + 1] = stem[3]
        tile[y + 2, x : x + 3] = r[1]
        tile[y + 2, x + 1] = stem[2]


def _bare(hexv: str):
    """A worn bare patch showing the ground underneath, ringed by short tufts."""

    def feature(tile, ctx, p, rng):
        T, r = ctx.T, ctx.ramp(0)
        d = ctx.palette.ramp(hexv)
        cx, cy = _spot(rng, T, 1, 1, margin=6)
        m = _blob(T, cx, cy, T * 0.26, T * 0.17, rng, rough=0.35)
        ys, xs = np.mgrid[0:T, 0:T]
        tile[m] = d[2]
        tile[m & ((xs * 7 + ys * 3) % 9 == 0)] = d[3]
        top, bottom = _rim(m)
        tile[top] = d[1]  # the grass edge overhangs and shades the top of the patch
        tile[bottom] = d[3]
        rim = np.zeros_like(m)
        rim[:-1] = top[1:]
        for y, x in zip(*np.nonzero(rim & ~m)):
            if rng.random() < 0.5:
                tile[y, x] = r[3]

    feature.__name__ = "bare_patch"
    return feature


def _crack(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    x, y = int(rng.integers(3, T - 3)), int(rng.integers(3, T // 3 + 1))  # start high so it has room to run
    dx = int(rng.choice([-1, 1]))
    for _ in range(int(T * 0.5)):
        tile[y, x] = r[0]
        if 2 <= x + 1 <= T - 3:
            tile[y, x + 1] = r[3] if tile[y, x + 1] != r[0] else r[0]  # lit lip on the far side
        if rng.random() < 0.3:
            dx = -dx
        x = int(np.clip(x + (dx if rng.random() < 0.6 else 0), 2, T - 3))
        y += 1 if rng.random() < 0.7 else 0
        if y > T - 3:
            break
        if rng.random() < 0.15:  # branch
            bx = int(np.clip(x - dx, 2, T - 3))
            tile[y, bx] = r[0]


def _stain(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    cx, cy = _spot(rng, T, 1, 1, margin=5)
    m = _blob(T, cx, cy, T * 0.25, T * 0.18, rng, rough=0.5)
    ys, xs = np.mgrid[0:T, 0:T]
    tile[m & ((xs + ys) % 2 == 0)] = r[1]
    top, _ = _rim(m)
    tile[m & ~top & ((xs + ys) % 4 == 1)] = r[0]


def _wet(tile, ctx, p, rng):
    """A puddle: dark water-filled dip with a sky glint, rimmed by damp ground."""
    T, r = ctx.T, ctx.ramp(0)
    cx, cy = _spot(rng, T, 1, 1, margin=5)
    m = _blob(T, cx, cy, T * 0.24, T * 0.14, rng, rough=0.2)
    damp = _blob(T, cx, cy + 0.5, T * 0.24 + 1.5, T * 0.14 + 1.2, rng, rough=0.2) & ~m
    tile[damp] = r[1]
    tile[m] = r[0]
    top, _ = _rim(m)
    tile[top] = r[1]
    ys, xs = np.nonzero(m & ~top)
    if len(ys):
        i = int(rng.integers(len(ys)))
        tile[ys[i], xs[i]] = r[4]


def _twig(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    w = ctx.palette.ramp(TWIG)
    n = int(T * 0.4)
    x, y = _spot(rng, T, n, 3)
    dy = int(rng.choice([-1, 1]))
    yy = y + 1
    for i in range(n):
        tile[yy, x + i] = w[2] if i % 3 else w[3]
        tile[min(yy + 1, T - 3), x + i] = r[0] if tile[min(yy + 1, T - 3), x + i] != w[2] else w[2]
        if i == n // 2:
            tile[max(yy - 1, 2), x + i + 1] = w[2]  # side shoot
        if rng.random() < 0.25:
            yy = int(np.clip(yy + dy, 2, T - 4))


def _footprints(tile, ctx, p, rng):
    """A trail of prints crossing the tile diagonally, alternating left and right feet."""
    T, r = ctx.T, ctx.ramp(0)
    down = rng.random() < 0.5
    x, y, step = 3, 3 if down else T - 6, 0
    while 2 <= x <= T - 4 and 2 <= y <= T - 4:
        tile[y, x : x + 2] = r[1]
        tile[y + 1, x : x + 2] = r[2]
        x += 3
        y += (2 if down else -2) + (1 if step % 2 else -1)
        step += 1


def _mound(tile, ctx, p, rng):
    """A drift or dune hump: lit crest, shaded lee side."""
    T, r = ctx.T, ctx.ramp(0)
    cx, cy = _spot(rng, T, 1, 1, margin=5)
    m = _blob(T, cx, cy, T * 0.28, T * 0.14, rng, rough=0.15)
    top, bottom = _rim(m)
    ys, xs = np.mgrid[0:T, 0:T] + 0.5
    tile[m & (ys < cy)] = r[4]
    tile[m & (ys >= cy) & (xs > cx)] = r[2]
    tile[top] = r[4]
    tile[bottom] = r[1]


def _ice(tile, ctx, p, rng):
    T = ctx.T
    i = ctx.palette.ramp(ICE)
    cx, cy = _spot(rng, T, 1, 1, margin=5)
    m = _blob(T, cx, cy, T * 0.28, T * 0.16, rng, rough=0.25)
    ys, xs = np.mgrid[0:T, 0:T]
    tile[m] = i[2]
    tile[m & ((xs - ys) % 5 == 0)] = i[4]  # glare streaks
    tile[m & ((xs - ys) % 5 == 1)] = i[3]
    _, bottom = _rim(m)
    tile[bottom] = i[1]


def _sprout(hexv: str):
    def feature(tile, ctx, p, rng):
        T = ctx.T
        for _ in range(_instances(T)):
            cx, base = _spot(rng, T, 1, 1, margin=5)
            _blades(tile, rng, ctx.palette.ramp(hexv), cx, base + 2, int(rng.integers(4, 7)), int(rng.integers(3, 5)))

    feature.__name__ = "sprouts"
    return feature


def _shell(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    s = ctx.palette.ramp("#e8a898")
    for _ in range(_instances(T, int(rng.integers(1, 3)))):
        x, y = _spot(rng, T, 3, 3)
        tile[y, x + 1] = s[4]
        tile[y + 1, x : x + 3] = [s[3], s[4], s[2]]
        tile[y + 2, x : x + 3] = [s[2], s[1], r[1]]


def _moss(tile, ctx, p, rng):
    """Moss creeping along the darkest pixels (mortar, cracks), plus a few tufts."""
    T, r = ctx.T, ctx.ramp(0)
    g = ctx.palette.ramp(LEAF)
    cx, cy = _spot(rng, T, 1, 1, margin=5)
    near = _blob(T, cx, cy, T * 0.32, T * 0.26, rng, rough=0.5)
    mortar = ctx.ramp(1)[:2] if len(ctx.colors) > 1 else []
    dark = np.isin(tile, r[:2] + mortar)
    tile[near & dark] = g[1]
    ys, xs = np.nonzero(near)
    for i in rng.choice(len(ys), size=min(len(ys), max(2, len(ys) // 6)), replace=False):
        tile[ys[i], xs[i]] = g[2] if rng.random() < 0.6 else g[3]


def _knot(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    for _ in range(_instances(T, int(rng.integers(1, 3)))):
        x, y = _spot(rng, T, 4, 3)
        tile[y, x + 1 : x + 3] = r[1]
        tile[y + 1, x] = r[1]
        tile[y + 1, x + 1 : x + 3] = r[0]
        tile[y + 1, x + 3] = r[1]
        tile[y + 2, x + 1 : x + 3] = r[1]
        tile[y + 2, x + 3] = r[3]


def _lily(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    g = ctx.palette.ramp(LEAF)
    for k in range(_instances(T, int(rng.integers(1, 3)))):
        w = int(rng.integers(5, 8))
        x, y = _spot(rng, T, w + 1, 5)
        pad = _blob(T, x + w / 2, y + 2, w / 2, 2.0, rng, rough=0.05)
        shade = np.zeros_like(pad)
        shade[1:, 1:] = pad[:-1, :-1]
        tile[shade & ~pad] = r[0]  # shadow on the water, down and right
        tile[pad] = g[2]
        top, bottom = _rim(pad)
        tile[top] = g[3]
        tile[bottom] = g[1]
        nx = x + w // 2
        tile[y + 2, nx] = r[1]  # the notch
        tile[y + 3, nx] = r[1] if pad[y + 3, nx] else tile[y + 3, nx]
        if k == 0 and rng.random() < 0.6:
            f = ctx.palette.ramp("#f0a0c0")
            fx = x + 1
            tile[y, fx] = f[3]
            tile[y + 1, fx - 1 : fx + 2] = [f[3], f[4], f[2]]


def _water_rock(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    s = ctx.palette.ramp(STONE)
    w = int(rng.integers(4, 7))
    x, y = _spot(rng, T, w + 2, 5)
    rock = _blob(T, x + 1 + w / 2, y + 2, w / 2, 2.0, rng, rough=0.05)
    foam = _blob(T, x + 1 + w / 2, y + 2.8, w / 2 + 1.2, 1.9, rng, rough=0.1) & ~rock
    tile[foam] = r[4]
    tile[rock] = s[2]
    top, bottom = _rim(rock)
    tile[top] = s[4]
    tile[bottom] = s[1]
    ys, xs = np.mgrid[0:T, 0:T] + 0.5
    tile[rock & ~top & ~bottom & (xs > x + 1 + w * 0.6)] = s[1]


def _sparkle(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    for _ in range(_instances(T, int(rng.integers(2, 4)))):
        x, y = _spot(rng, T, 3, 3)
        tile[y + 1, x : x + 3] = r[3]
        tile[y : y + 3, x + 1] = r[3]
        tile[y + 1, x + 1] = r[4]


def _ripple_ring(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    cx, cy = _spot(rng, T, 1, 1, margin=6)
    ys, xs = np.mgrid[0:T, 0:T] + 0.5
    d = np.sqrt(((xs - cx) / 1.6) ** 2 + (ys - cy) ** 2)
    keep = np.zeros((T, T), bool)
    keep[2 : T - 2, 2 : T - 2] = True
    tile[keep & (np.abs(d - T * 0.18) < 0.5) & (ys < cy)] = r[4]
    tile[keep & (np.abs(d - T * 0.18) < 0.5) & (ys >= cy)] = r[3]
    tile[keep & (np.abs(d - T * 0.08) < 0.5)] = r[3]


def _crust(tile, ctx, p, rng):
    """A floating slab of cooled rock with a white-hot rim."""
    T, r = ctx.T, ctx.ramp(0)
    s = ctx.palette.ramp("#5a4850")
    cx, cy = _spot(rng, T, 1, 1, margin=6)
    m = _blob(T, cx, cy, T * 0.24, T * 0.16, rng, rough=0.25)
    glow = _blob(T, cx, cy + 0.5, T * 0.24 + 1.3, T * 0.16 + 1.3, rng, rough=0.25) & ~m
    tile[glow] = r[4]
    tile[m] = s[2]
    top, bottom = _rim(m)
    tile[top] = s[3]
    tile[bottom] = s[1]
    ys, xs = np.mgrid[0:T, 0:T]
    tile[m & ~top & ~bottom & ((xs * 3 + ys * 5) % 11 == 0)] = r[3]  # glowing cracks


def _bubble(tile, ctx, p, rng):
    T, r = ctx.T, ctx.ramp(0)
    for _ in range(_instances(T, int(rng.integers(1, 3)))):
        x, y = _spot(rng, T, 5, 5)
        tile[y + 4, x : x + 5] = r[0]  # shadow ring where the surface dips
        tile[y, x + 1 : x + 4] = r[4]
        tile[y + 1 : y + 4, x] = r[4]
        tile[y + 1 : y + 4, x + 4] = r[2]
        tile[y + 3, x + 1 : x + 4] = r[2]
        tile[y + 1 : y + 3, x + 1 : x + 4] = r[3]
        tile[y + 1, x + 1] = r[4]


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
    decor=(_tall_grass, _flower_clump, _pebbles(STONE), _mushrooms, _bare("#9a6b42")),
)
def grass(ctx: GenContext, p: GrassParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 3, octaves=3), 1.2 + p.roughness * 1.2)
    tile = quantize(f, [r[2], r[2], r[3]], dither=ctx.dither(0.8))
    tile[f < 0.3] = r[1]
    for x, y, u in _scatter(ctx, variant, int(p.density * T * T / 14)):
        _px(tile, x, y, r[3])
        _px(tile, x - 1, y + 1, r[1])
        _px(tile, x + 1, y + 1, r[1])
        _px(tile, x, y + 1, r[2])
        if u[0] < 0.5:
            _px(tile, x, y - 1, r[4])
    if p.flowers > 0:
        a = ctx.ramp(1)
        for x, y, _ in _scatter(ctx, variant, int(round(p.flowers * T * T / 40)), margin=2, key="flowers"):
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                _px(tile, x + dx, y + dy, a[2])
            _px(tile, x, y, a[4])
            _px(tile, x, y + 2, r[0])
    return tile


class DirtParams(Params):
    pebbles: float = Field(0.4, ge=0.0, le=1.0, description="pebble density")
    roughness: float = Field(0.5, ge=0.0, le=1.0)


@register(
    "dirt",
    "terrain",
    DirtParams,
    colors=[("soil", "#9a6b42")],
    doc="Bare earth or a dirt path, with pebbles.",
    decor=(_pebbles(STONE, big=True), _crack, _wet, _twig, _sprout(LEAF)),
)
def dirt(ctx: GenContext, p: DirtParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    # Mottled earth: darker and lighter clumps rather than one flat brown.
    f = contrast(_field(ctx, variant, 3), 1.3 + p.roughness * 0.9)
    tile = quantize(f, [r[1], r[2], r[2], r[3]], dither=ctx.dither(0.9))
    for x, y, u in _scatter(
        ctx, variant, max(1, round(p.roughness * T * T / 128)), margin=4, key="patches", edge=False
    ):
        _patch(tile, x, y, r[1], r[3], u)
    for x, y, u in _scatter(ctx, variant, int(p.pebbles * T * T / 28)):
        # A clod or pebble: lit top, shaded bottom.
        _px(tile, x, y, r[4] if u[1] < 0.3 else r[3])
        _px(tile, x, y + 1, r[0])
        if u[0] < 0.5:
            _px(tile, x + 1, y, r[3])
            _px(tile, x + 1, y + 1, r[0])
    for x, y, _ in _scatter(ctx, variant, T * T // 40, key="crumbs"):
        _px(tile, x, y, r[1])
    return tile


class SandParams(Params):
    ripples: float = Field(0.6, ge=0.0, le=1.0, description="wind ripple coverage")
    speckles: float = Field(0.3, ge=0.0, le=1.0)


@register(
    "sand",
    "terrain",
    SandParams,
    colors=[("sand", "#e2c47e")],
    doc="Desert or beach sand with wind ripples.",
    decor=(_shell, _pebbles(STONE), _mound, _sprout(LEAF), _twig),
)
def sand(ctx: GenContext, p: SandParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 3), 2.0)
    tile = quantize(f, [r[2], r[3], r[3]], dither=ctx.dither(1.0))
    ys, xs = np.mgrid[0:T, 0:T]
    bend = (Noise(ctx.rng("bend", variant), T, cells=2).tile() - 0.5) * T * 0.25 * _inner(T)
    wave = (ys + 1.5 * np.sin(2 * np.pi * xs / T) + 0.8 * np.sin(4 * np.pi * xs / T + 1.0) + bend) % (T / 2)
    gate = _field(ctx, variant, 2, octaves=1, amount=1.0, flatten=0.0, key="ripple") < 0.2 + p.ripples * 0.4
    tile[(wave < 1.0) & gate] = r[2]
    tile[(wave >= 1.0) & (wave < 2.0) & gate] = r[4]
    for x, y, u in _scatter(ctx, variant, int(p.speckles * T * T / 30), margin=1):
        _px(tile, x, y, r[1] if u[0] < 0.6 else r[4])
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
    decor=(_moss, _crack, _stain, _sprout(LEAF), _wet),
)
def cobble(ctx: GenContext, p: CobbleParams, variant: int, phase: float) -> np.ndarray:
    T, r, m = ctx.T, ctx.ramp(0), ctx.ramp(1)
    ids, f1, f2 = voronoi(ctx.rng("base"), T, p.stones)
    # Split stones inside the tile with extra seed points, unless that would move the border.
    vrng = ctx.rng("split", variant)
    for _ in range(8):
        extra = [tuple(vrng.uniform(T * 0.3, T * 0.7, size=2)) for _ in range(int(vrng.integers(1, 3)))]
        ids2, g1, g2 = voronoi(ctx.rng("base"), T, p.stones, extra=extra)
        ring = ~np.pad(np.ones((T - 4, T - 4), bool), 2)
        if np.array_equal(ids2[ring], ids[ring]) and np.array_equal((g2 - g1)[ring] < 1.3, ((f2 - f1) < 1.3)[ring]):
            ids, f1, f2 = ids2, g1, g2
            break
    mortar = (f2 - f1) < 1.3
    shades = ctx.rng("shade").choice([r[2], r[2], r[3]], size=64)  # fixed size: same draws for every variant
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
    decor=(_crack, _stain, _moss, _pebbles()),
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
    decor=(_knot, _stain, _crack),
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


@register(
    "snow",
    "terrain",
    SnowParams,
    colors=[("snow", "#dfe8f2")],
    doc="Snowfield with soft shadowed drifts.",
    decor=(_footprints, _mound, _ice, _pebbles(STONE), _twig),
)
def snow(ctx: GenContext, p: SnowParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    f = contrast(_field(ctx, variant, 4), 1.6)
    # Near-white bases clamp their top ramp steps to the same white, so shade from step 2 up.
    tile = quantize(f, [r[2], r[3], r[3]], dither=ctx.dither(0.8))
    for x, y, u in _scatter(ctx, variant, max(1, round(p.drifts * T * T / 100)), margin=4, key="drifts", edge=False):
        _patch(tile, x, y, r[1], r[3], u)
    for x, y, _ in _scatter(ctx, variant, T // 6, margin=1):
        _px(tile, x, y, r[2])
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
    decor=(_lily, _water_rock, _sparkle, _ripple_ring),
)
def water(ctx: GenContext, p: WaterParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    base = _field(ctx, variant, 2, t=phase, tcells=2)
    levels = [r[1], r[1], r[2]] if p.depth > 0.6 else [r[2], r[2], r[3]] if p.depth < 0.25 else [r[1], r[2], r[2]]
    tile = quantize(contrast(base, 1.3), levels, dither=ctx.dither(1.0))
    crest = _field(ctx, variant, (2, 5), t=phase, tcells=2, dx=phase * T, amount=1.0, key="waves")
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
    decor=(_crust, _bubble, _pebbles("#4a4050")),
)
def lava(ctx: GenContext, p: LavaParams, variant: int, phase: float) -> np.ndarray:
    T, r = ctx.T, ctx.ramp(0)
    # Veins are lines along flow == 0.5, so variants only nudge the shared field: blending in a whole
    # field of their own would draw a ring of vein wherever the blend crosses 0.5.
    flow = _field(
        ctx,
        variant,
        2,
        octaves=3,
        amount=0.3,
        t=phase,
        tcells=2,
        dy=-phase * T,
        edge_power=3.0,
        flatten=0.0,
        fine=False,
        key="base",
    )
    ridge = np.abs(contrast(flow, 3.0) - 0.5) * 2.0  # 0 along the veins
    glow = np.clip(1.0 - ridge / (0.2 + (1.0 - p.crust) * 0.5), 0.0, 1.0) ** 0.7
    tile = quantize(glow, [r[0], r[1], r[2], r[3], r[4]], dither=ctx.dither(0.7))
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
    decor=(_crack, _stain, _moss, _pebbles()),
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
