"""A sample map that exercises the tileset: terrain bands, autotiles and objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import registry
from .atlas import ConnectedSet, ObjectGroup, WangSet
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
    characters: list[tuple[str, int, int, int]] = field(default_factory=list)  # (name, x, y, direction row)


def terrain_chain(res: BuildResult) -> list[str]:
    """Longest simple path through the transition graph, used as elevation bands. Tone patches never count."""
    names = [t.name for t in res.spec.terrains if t.name not in res.patch_of]
    adj: dict[str, list[str]] = {n: [] for n in names}
    for ws in res.wangsets:
        if ws.lower in adj and ws.upper in adj:
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

    def carve(host: str, branch: str, share: float) -> None:
        # Patches go only where the host fully surrounds a corner, so every
        # cell touching the patch mixes just {host, branch}: a valid transition.
        inside = _neighbors8_all(corners == names.index(host))
        if not inside.any():
            return
        noise = Noise(ctx.rng("patch", branch), max(width, height) + 1, cells=4, octaves=2).sample(xs, ys)
        cut = np.quantile(noise[inside], 1.0 - share)
        corners[inside & (noise > cut)] = names.index(branch)

    while queue:
        host = queue.pop(0)
        for branch in adj[host]:
            if branch in placed or branch in res.patch_of:
                continue
            placed.append(branch)
            queue.append(branch)
            carve(host, branch, 0.4)
    # Tone patches last, inside whatever area their host ended up with.
    shares = {t.name: t.patches for t in res.spec.terrains}
    for patch, host in res.patch_of.items():
        carve(host, patch, shares[host])
    return corners, names


def make_preview(res: BuildResult, width: int = 24, height: int = 16) -> PreviewMap:
    spec = res.spec
    ctx = GenContext(spec.tile_size, spec.seed, "preview", res.palette, [])
    rng = ctx.rng("pick")
    corners, names = _corner_terrains(res, ctx, width, height)
    by_pair: dict[frozenset, WangSet] = {frozenset((w.lower, w.upper)): w for w in res.wangsets}

    ground = np.zeros((height, width), dtype=int)
    cell_terrain: list[list[str | None]] = [[None] * width for _ in range(height)]
    cell_corners: list[list[set[str]]] = [[set()] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            c = tuple(names[i] for i in (corners[y, x], corners[y, x + 1], corners[y + 1, x + 1], corners[y + 1, x]))
            kinds = set(c)
            cell_corners[y][x] = kinds
            if len(kinds) == 1:
                ground[y, x] = res.pick_terrain(c[0], rng)
                cell_terrain[y][x] = c[0]
                continue
            ws = by_pair[frozenset(kinds)]
            ground[y, x] = rng.choice(ws.tiles[tuple(int(t == ws.upper) for t in c)])

    # Terrains with no transitions at all can only appear as hard-edged rectangles (rugs, plazas).
    linked = [w for w in res.wangsets if w.upper not in res.patch_of]
    connected = {n for w in linked for n in (w.lower, w.upper)} | {names[int(corners[0, 0])]}
    for t in spec.terrains:
        if t.name in connected or t.name in res.patch_of:
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

    def allowed_on(spec_obj) -> set[str]:
        allowed = set(spec_obj.place_on) or {t.name for t in spec.terrains if t.name not in animated}
        return allowed | {patch for patch, host in res.patch_of.items() if host in allowed}

    def free(x: int, y: int, allowed: set[str]) -> bool:
        return cell_terrain[y][x] in allowed and objects[y, x] < 0

    def place(group: ObjectGroup, x0: int, y0: int) -> None:
        for j, row in enumerate(group.tiles):
            for i, tid in enumerate(row):
                objects[y0 + j, x0 + i] = tid

    # Linked objects (fences) first: rectangular enclosures with a gate in the south side.
    enclosures = []
    spec_by_name = {o.name: o for o in spec.objects}

    def fenceable(x: int, y: int, allowed: set[str]) -> bool:
        # A fence can run along a terrain edge too: any cell touching allowed ground, clear of water and lava.
        kinds = cell_corners[y][x]
        return objects[y, x] < 0 and bool(kinds & allowed) and not kinds & animated

    for cs in res.connected:
        spec_obj = spec_by_name[cs.name]
        allowed = allowed_on(spec_obj)
        for _ in range(max(1, round(spec_obj.density * 3)) if spec_obj.density > 0 else 0):
            # Biggest pen that fits anywhere, down to 3x3 (one cell inside).
            sizes = sorted(((w, h) for w in range(3, 9) for h in range(3, 7)), key=lambda wh: -wh[0] * wh[1])
            for ew, eh in sizes:
                spots = [
                    (x0, y0)
                    for y0 in range(height - eh + 1)
                    for x0 in range(width - ew + 1)
                    if all(fenceable(x0 + i, y0 + j, allowed) for j in range(eh) for i in range(ew))
                ]
                if spots:
                    x0, y0 = spots[int(rng.integers(len(spots)))]
                    _fence_rectangle(objects, cs, x0, y0, ew, eh, rng)
                    enclosures.append((x0 + 1, y0 + 1, ew - 2, eh - 2))
                    break

    groups = {g.name: g for g in res.objects}
    scattered = [(o, groups[o.name]) for o in spec.objects if o.name in groups]
    for spec_obj, group in sorted(scattered, key=lambda p: -p[1].width * p[1].height):
        allowed = allowed_on(spec_obj)
        gw, gh = group.width, group.height

        def fits(x0: int, y0: int) -> bool:
            return all(free(x0 + i, y0 + j, allowed) for j in range(gh) for i in range(gw))

        spots = [(x, y) for y in range(height - gh + 1) for x in range(width - gw + 1) if fits(x, y)]
        target = int(np.ceil(spec_obj.density * len(spots) * 0.3 / (gw * gh))) if spots else 0
        limit = registry.get(spec_obj.generator).limit
        if limit:
            # Landmarks (houses, wells, chests) stay rare however much room there is.
            target = min(target, limit * max(1, width * height // (24 * 16)))
        placed = 0
        for k in rng.permutation(len(spots)):
            if placed >= target:
                break
            x0, y0 = spots[k]
            if fits(x0, y0):
                place(group, x0, y0)
                placed += 1

    # A fence should be keeping something in: fill enclosures the scatter left empty.
    for ix, iy, iw, ih in enclosures:
        if (objects[iy : iy + ih, ix : ix + iw] >= 0).any():
            continue
        for spec_obj, group in sorted(scattered, key=lambda p: (-p[1].width * p[1].height, p[0].name)):
            allowed = allowed_on(spec_obj)
            spots = [
                (x, y)
                for y in range(iy, iy + ih - group.height + 1)
                for x in range(ix, ix + iw - group.width + 1)
                if all(free(x + i, y + j, allowed) for j in range(group.height) for i in range(group.width))
            ]
            if spots:
                place(group, *spots[int(rng.integers(len(spots)))])
                break

    # Characters stand on open ground, a little apart, mostly facing the camera.
    characters: list[tuple[str, int, int, int]] = []
    taken: set[tuple[int, int]] = set()
    for ch in spec.characters:
        allowed = allowed_on(ch)
        spots = [
            (x, y)
            for y in range(1, height)
            for x in range(width)
            if free(x, y, allowed) and not any((x + i, y + j) in taken for i in (-1, 0, 1) for j in (-1, 0, 1))
        ]
        for k in rng.permutation(len(spots))[: ch.count]:
            x, y = spots[k]
            if any((x + i, y + j) in taken for i in (-1, 0, 1) for j in (-1, 0, 1)):
                continue
            taken.add((x, y))
            characters.append((ch.name, x, y, int(rng.choice(4, p=[0.55, 0.15, 0.15, 0.15]))))
    characters.sort(key=lambda c: (c[2], c[1]))
    return PreviewMap(width, height, ground, objects, terrain_chain(res), characters)


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
    # Characters have their own palettes: cast their shadows in index space, then paint them over the RGBA map.
    sheets = {c.name: c for c in res.characters}
    sprites = []
    for name, x, y, row in pmap.characters:
        sheet = sheets[name]
        sprite = sheet.frames[row, 1]
        h, w = sprite.shape
        x0, y0 = x * T + (T - w) // 2, (y + 1) * T - h
        cut = max(0, -y0)
        sprite, y0 = sprite[cut:], y0 + cut
        region = img[y0 : y0 + sprite.shape[0], x0 : x0 + w]
        shadow = sprite == SHADOW
        region[shadow] = darker[region[shadow]]
        sprites.append((sheet, sprite, x0, y0))
    rgba = res.to_rgba(img)
    for sheet, sprite, x0, y0 in sprites:
        region = rgba[y0 : y0 + sprite.shape[0], x0 : x0 + sprite.shape[1]]
        solid = (sprite > 0) & (sprite != SHADOW)
        region[solid] = sheet.to_rgba(sprite)[solid]
    out = Image.fromarray(rgba, "RGBA")
    return out.resize((out.width * scale, out.height * scale), Image.NEAREST)


def _fence_rectangle(
    objects: np.ndarray, cs: ConnectedSet, x0: int, y0: int, w: int, h: int, rng: np.random.Generator
) -> None:
    """Draw a closed fence loop around a w x h block of cells, with a gate in the south side."""
    N, E, S, W = 1, 2, 4, 8
    x1, y1 = x0 + w - 1, y0 + h - 1
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            mask = 0
            if y in (y0, y1) and x > x0:
                mask |= W
            if y in (y0, y1) and x < x1:
                mask |= E
            if x in (x0, x1) and y > y0:
                mask |= N
            if x in (x0, x1) and y < y1:
                mask |= S
            if mask:
                objects[y, x] = cs.tiles[mask]
    if cs.gate is not None:
        objects[y1, int(rng.integers(x0 + 1, x1))] = cs.gate
