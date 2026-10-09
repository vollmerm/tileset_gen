"""A sample map that exercises the tileset: terrain bands, autotiles and objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .atlas import WangSet
from .build import BuildResult
from .context import GenContext
from .noise import Noise
from .palette import SHADOW


@dataclass
class PreviewMap:
    width: int
    height: int
    ground: np.ndarray  # (H, W) tile ids
    objects: np.ndarray  # (H, W) tile ids, -1 = empty
    chain: list[str]


def terrain_chain(res: BuildResult) -> list[str]:
    """Longest simple path through the transition graph, used as elevation bands."""
    names = [t.name for t in res.spec.terrains]
    adj: dict[str, list[str]] = {n: [] for n in names}
    for ws in res.wangsets:
        adj[ws.lower].append(ws.upper)
        adj[ws.upper].append(ws.lower)
    best: list[str] = names[:1]

    def dfs(path: list[str]) -> None:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        for nxt in adj[path[-1]]:
            if nxt not in path:
                dfs(path + [nxt])

    for n in names:
        dfs([n])
    return best


def _corner_levels(rng_ctx: GenContext, w: int, h: int, n_levels: int) -> np.ndarray:
    period = max(w, h) + 1
    noise = Noise(rng_ctx.rng("height"), period, cells=3, octaves=3, persistence=0.45)
    ys, xs = np.mgrid[0 : h + 1, 0 : w + 1] + 0.5
    field = noise.sample(xs, ys)
    if n_levels == 1:
        return np.zeros(field.shape, dtype=int)
    qs = np.quantile(field, np.linspace(0, 1, n_levels + 1)[1:-1])
    levels = np.searchsorted(qs, field)
    # Neighboring corners may differ by at most one band, so every cell is
    # either solid or a two-terrain transition.
    while True:
        p = np.pad(levels, 1, mode="edge")
        nmax = np.max(
            [p[1 + dy : p.shape[0] - 1 + dy, 1 + dx : p.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
            axis=0,
        )
        fixed = np.maximum(levels, nmax - 1)
        if np.array_equal(fixed, levels):
            return levels
        levels = fixed


def _neighbors8_all(mask: np.ndarray) -> np.ndarray:
    """True where a corner and all 8 of its neighbors are set."""
    p = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return out


def _corner_terrains(res: BuildResult, ctx: GenContext, width: int, height: int) -> tuple[np.ndarray, list[str]]:
    """Terrain index per map corner: chain terrains as elevation bands, branch terrains as patches."""
    names = [t.name for t in res.spec.terrains]
    chain = terrain_chain(res)
    levels = _corner_levels(ctx, width, height, len(chain))
    corners = np.array([names.index(chain[lv]) for lv in levels.ravel()]).reshape(levels.shape)

    adj: dict[str, list[str]] = {n: [] for n in names}
    for ws in res.wangsets:
        adj[ws.lower].append(ws.upper)
        adj[ws.upper].append(ws.lower)
    placed = list(chain)
    queue = list(chain)
    ys, xs = np.mgrid[0 : height + 1, 0 : width + 1] + 0.5
    while queue:
        host = queue.pop(0)
        for branch in adj[host]:
            if branch in placed:
                continue
            placed.append(branch)
            queue.append(branch)
            # Patches go only where the host fully surrounds a corner, so every
            # cell touching the patch mixes just {host, branch}: a valid transition.
            inside = _neighbors8_all(corners == names.index(host))
            if not inside.any():
                continue
            noise = Noise(ctx.rng("patch", branch), max(width, height) + 1, cells=4, octaves=2).sample(xs, ys)
            cut = np.quantile(noise[inside], 0.6)
            corners[inside & (noise > cut)] = names.index(branch)
    return corners, names


def make_preview(res: BuildResult, width: int = 24, height: int = 16) -> PreviewMap:
    spec = res.spec
    ctx = GenContext(spec.tile_size, spec.seed, "preview", res.palette, [])
    rng = ctx.rng("pick")
    corners, names = _corner_terrains(res, ctx, width, height)
    by_pair: dict[frozenset, WangSet] = {frozenset((w.lower, w.upper)): w for w in res.wangsets}

    ground = np.zeros((height, width), dtype=int)
    cell_terrain: list[list[str | None]] = [[None] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            c = tuple(names[i] for i in (corners[y, x], corners[y, x + 1], corners[y + 1, x + 1], corners[y + 1, x]))
            kinds = set(c)
            if len(kinds) == 1:
                ground[y, x] = res.pick_terrain(c[0], rng)
                cell_terrain[y][x] = c[0]
                continue
            ws = by_pair[frozenset(kinds)]
            ground[y, x] = rng.choice(ws.tiles[tuple(int(t == ws.upper) for t in c)])

    # Terrains with no transitions at all can only appear as hard-edged rectangles (rugs, plazas).
    connected = {n for w in res.wangsets for n in (w.lower, w.upper)} | {names[int(corners[0, 0])]}
    for t in spec.terrains:
        if t.name in connected:
            continue
        for _ in range(2):
            rw, rh = int(rng.integers(3, 6)), int(rng.integers(2, 5))
            for _try in range(40):
                x0, y0 = int(rng.integers(0, width - rw + 1)), int(rng.integers(0, height - rh + 1))
                host = cell_terrain[y0][x0]
                if host and all(cell_terrain[y0 + j][x0 + i] == host for j in range(rh) for i in range(rw)):
                    for j in range(rh):
                        for i in range(rw):
                            ground[y0 + j, x0 + i] = res.pick_terrain(t.name, rng)
                            cell_terrain[y0 + j][x0 + i] = t.name
                    break

    objects = np.full((height, width), -1, dtype=int)
    animated = {t.name for t in spec.terrains if any(a.terrain == t.name for a in spec.animations)}
    for spec_obj, group in sorted(zip(spec.objects, res.objects), key=lambda p: -p[1].width * p[1].height):
        allowed = set(spec_obj.place_on) or {t.name for t in spec.terrains if t.name not in animated}
        gw, gh = group.width, group.height

        def fits(x0: int, y0: int) -> bool:
            return all(
                cell_terrain[y0 + j][x0 + i] in allowed and objects[y0 + j, x0 + i] < 0
                for j in range(gh)
                for i in range(gw)
            )

        spots = [(x, y) for y in range(height - gh + 1) for x in range(width - gw + 1) if fits(x, y)]
        target = int(np.ceil(spec_obj.density * len(spots) * 0.3 / (gw * gh))) if spots else 0
        placed = 0
        for k in rng.permutation(len(spots)):
            if placed >= target:
                break
            x0, y0 = spots[k]
            if fits(x0, y0):
                for j, row in enumerate(group.tiles):
                    for i, tid in enumerate(row):
                        objects[y0 + j, x0 + i] = tid
                placed += 1
    return PreviewMap(width, height, ground, objects, terrain_chain(res))


def render_preview(res: BuildResult, pmap: PreviewMap, scale: int = 3) -> Image.Image:
    T = res.spec.tile_size
    img = np.zeros((pmap.height * T, pmap.width * T), dtype=np.uint8)
    tiles = res.atlas.tiles
    darker = res.palette.shadow_lut()
    for y in range(pmap.height):
        for x in range(pmap.width):
            block = tiles[pmap.ground[y, x]].copy()
            oid = pmap.objects[y, x]
            if oid >= 0:
                o = tiles[oid]
                shadow = o == SHADOW
                block[shadow] = darker[block[shadow]]
                solid = (o > 0) & ~shadow
                block[solid] = o[solid]
            img[y * T : (y + 1) * T, x * T : (x + 1) * T] = block
    rgba = res.to_rgba(img)
    out = Image.fromarray(rgba, "RGBA")
    return out.resize((out.width * scale, out.height * scale), Image.NEAREST)
