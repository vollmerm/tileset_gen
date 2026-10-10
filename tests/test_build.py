import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tileset_gen.build import SpecError, build, generate
from tileset_gen.context import GenContext
from tileset_gen.generators.autotile import ALL_CORNERS, MIXED_CORNERS, PAD, corner_mask
from tileset_gen.noise import Noise
from tileset_gen.palette import Palette
from tileset_gen.registry import REGISTRY, describe_for_prompt
from tileset_gen.spec import TilesetSpec
from tileset_gen.validate import validate_spec

EXAMPLES = sorted((Path(__file__).parent.parent / "examples" / "specs").glob("*.json"))


def load(name: str) -> TilesetSpec:
    return TilesetSpec.load(next(p for p in EXAMPLES if p.stem == name))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_builds_and_exports(path: Path, tmp_path: Path):
    spec = TilesetSpec.load(path)
    res = build(spec, tmp_path, map_size=(12, 8), preview_scale=1)
    for name in ("tileset.png", "tileset.json", "tileset.tsx", "preview.tmx", "preview.png", "spec.json"):
        assert (tmp_path / name).exists(), name

    img = Image.open(tmp_path / "tileset.png").convert("RGBA")
    opaque = np.asarray(img).reshape(-1, 4)
    colors = {tuple(c[:3]) for c in opaque if c[3] > 0}
    assert len(colors) <= spec.max_colors

    count = res.atlas.tile_count
    tsx = ET.parse(tmp_path / "tileset.tsx").getroot()
    assert int(tsx.get("tilecount")) == count
    for frame in tsx.iter("frame"):
        assert 0 <= int(frame.get("tileid")) < count
    corner_sets = [w for w in tsx.iter("wangset") if w.get("type") == "corner"]
    assert len(corner_sets) == 1
    terrain_set = corner_sets[0]
    n_colors = len(list(terrain_set.iter("wangcolor")))
    assert n_colors == len(res.spec.terrains)  # includes the generated tone-patch terrains
    tile_ids = [int(wt.get("tileid")) for wt in terrain_set.iter("wangtile")]
    assert len(tile_ids) == len(set(tile_ids))
    expected = {t for ws in res.wangsets for ids in ws.tiles.values() for t in ids}
    expected |= {t for ids in res.terrain_tiles.values() for t in ids}
    assert set(tile_ids) == expected
    for wt in terrain_set.iter("wangtile"):
        ids = [int(v) for v in wt.get("wangid").split(",")]
        assert len(ids) == 8 and ids[0::2] == [0, 0, 0, 0]
        assert all(1 <= v <= n_colors for v in ids[1::2])
    edge_sets = {w.get("name"): w for w in tsx.iter("wangset") if w.get("type") == "edge"}
    assert set(edge_sets) == {c.name for c in res.connected}
    for cs in res.connected:
        labeled = {int(wt.get("tileid")): wt.get("wangid") for wt in edge_sets[cs.name].iter("wangtile")}
        assert set(labeled) == {t for m, t in cs.tiles.items() if m}
        for m, t in cs.tiles.items():
            if m:
                ids = [int(v) for v in labeled[t].split(",")]
                assert ids[1::2] == [0, 0, 0, 0] and ids[0::2] == [int(bool(m & b)) for b in (1, 2, 4, 8)]

    tmx = ET.parse(tmp_path / "preview.tmx").getroot()
    for data in tmx.iter("data"):
        gids = [int(v) for v in data.text.replace("\n", "").split(",")]
        assert len(gids) == 12 * 8
        assert all(g == 0 or (g - 1) in res.atlas.tiles for g in gids)

    meta = json.loads((tmp_path / "tileset.json").read_text())
    assert meta["tile_count"] == count
    assert TilesetSpec.load(tmp_path / "spec.json") == spec


def test_build_is_deterministic(tmp_path: Path):
    spec = load("grassland")
    build(spec, tmp_path / "a", map_size=(8, 6), preview_scale=1)
    build(spec, tmp_path / "b", map_size=(8, 6), preview_scale=1)
    for name in ("tileset.png", "tileset.json", "tileset.tsx", "preview.tmx", "preview.png"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_seed_changes_output():
    spec = load("grassland")
    a = generate(spec).atlas.sheet()
    b = generate(spec.model_copy(update={"seed": spec.seed + 1})).atlas.sheet()
    assert not np.array_equal(a, b)


def test_tile_size_32():
    spec = load("dungeon").model_copy(update={"tile_size": 32})
    res = generate(spec)
    assert all(t.shape == (32, 32) for t in res.atlas.tiles.values())


def test_palette_reduction_respects_limit():
    spec = load("grassland").model_copy(update={"max_colors": 16})
    res = generate(spec)
    used = np.unique(res.remap[res.atlas.sheet()])
    assert len(used[used > 0]) <= 16


def test_noise_is_periodic():
    n = Noise(np.random.default_rng(1), 16, cells=(3, 5), octaves=3, tcells=4)
    ys, xs = np.mgrid[0:16, 0:16] + 0.5
    assert np.allclose(n.sample(xs, ys, 0.25), n.sample(xs + 16, ys - 32, 1.25))


@pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
def test_corner_masks_respect_corners(roughness: float):
    T = 16
    noise = Noise(np.random.default_rng(3), T, cells=4, octaves=2)
    for corners in ALL_CORNERS:
        m = corner_mask(T, corners, noise, roughness)[PAD : PAD + T, PAD : PAD + T]
        nw, ne, se, sw = corners
        # The pixel at each corner shows that corner's terrain.
        assert (m[0, 0], m[0, -1], m[-1, -1], m[-1, 0]) == tuple(bool(c) for c in corners)
        # Edges whose two corners agree are pure, so tiles meet seamlessly there.
        for a, b, edge in ((nw, ne, m[0]), (ne, se, m[:, -1]), (se, sw, m[-1]), (sw, nw, m[:, 0])):
            if a == b:
                assert (edge == bool(a)).all(), corners


def test_wangsets_cover_all_corner_combinations():
    res = generate(load("grassland"))
    for ws in res.wangsets:
        assert set(ws.tiles) == set(ALL_CORNERS)
        assert all(ws.tiles[c] for c in MIXED_CORNERS)


def test_multi_tile_objects_are_packed_as_rectangles():
    res = generate(load("grassland"))
    cols = res.atlas.columns
    for group in res.objects:
        top_left = group.tiles[0][0]
        for j, row in enumerate(group.tiles):
            for i, tid in enumerate(row):
                assert tid == top_left + j * cols + i


def test_terrain_animation_frame_zero_is_static_tile():
    spec = load("dungeon")
    res = generate(spec)
    lava = res.terrain_tiles["lava"][0]
    anim = res.atlas.meta[lava].animation
    assert anim and anim[0][0] == lava and len(anim) == 4


def test_validation_catches_mistakes():
    spec = load("grassland")
    bad = spec.model_copy(deep=True)
    bad.terrains[0].generator = "marshmallow"
    bad.terrains[1].params = {"ripples": 7.0}
    bad.transitions[0].upper = "nowhere"
    bad.objects[0].colors = ["not_a_color"]
    errors = validate_spec(bad)
    joined = "\n".join(errors)
    assert "marshmallow" in joined and "ripples" in joined and "nowhere" in joined and "not_a_color" in joined
    with pytest.raises(SpecError):
        generate(bad)


def test_registry_describes_every_generator():
    text = describe_for_prompt()
    for name in REGISTRY:
        assert f"- {name}" in text


def test_ramps_are_dark_to_light():
    pal = Palette({"g": "#5aa83c"})
    ramp = [pal.colors[i] for i in pal.ramp("g")]
    lum = [0.3 * r + 0.59 * g + 0.11 * b for r, g, b in ramp]
    assert lum == sorted(lum)


def test_context_rng_is_stable():
    ctx = GenContext(16, 5, "terrain:grass", Palette(), [])
    assert ctx.rng("a", 1).random() == ctx.rng("a", 1).random()
    assert ctx.rng("a", 1).random() != ctx.rng("a", 2).random()


@pytest.mark.parametrize("tile_size", [16, 32])
def test_terrain_variants_share_borders_but_differ_inside(tile_size: int):
    from tileset_gen.registry import by_kind

    structural = {"brick_floor", "wood_floor"}  # regular patterns; variants only add wear marks
    for g in by_kind("terrain"):
        ctx = GenContext(tile_size, 1, "t", Palette(), g.default_colors())
        a, b = g.fn(ctx, g.params(), 0, 0.0), g.fn(ctx, g.params(), 1, 0.0)
        edge = np.ones(a.shape, bool)
        edge[1:-1, 1:-1] = False
        assert np.mean(a[edge] != b[edge]) <= 0.05, g.name
        if g.name not in structural:
            assert np.mean(a[~edge] != b[~edge]) >= 0.04, g.name


@pytest.mark.parametrize("tile_size", [16, 32])
def test_decor_tiles_keep_their_variants_border(tile_size: int):
    from tileset_gen.generators.terrain import decorate
    from tileset_gen.registry import by_kind

    for g in by_kind("terrain"):
        assert g.decor, f"{g.name} has no decor features"
        for seed in range(3):
            ctx = GenContext(tile_size, seed, "t", Palette(), g.default_colors())
            base = g.fn(ctx, g.params(), 0, 0.0)
            for k in range(len(g.decor)):
                d = decorate(g, ctx, g.params(), base, k)
                assert np.array_equal(d[:2], base[:2]) and np.array_equal(d[-2:], base[-2:]), (g.name, k)
                assert np.array_equal(d[:, :2], base[:, :2]) and np.array_equal(d[:, -2:], base[:, -2:]), (g.name, k)
                assert np.mean(d != base) >= 0.015 * 16 / tile_size, f"{g.name} decor {k} is barely visible"


def test_decor_tiles_are_built_and_rare(tmp_path: Path):
    spec = load("grassland")
    res = build(spec, tmp_path, map_size=(8, 6), preview_scale=1)
    for t in spec.terrains:
        ids = res.terrain_tiles[t.name]
        assert len(ids) == t.variants + t.decor
        decor = [i for i in ids if "decor" in res.atlas.meta[i].props]
        assert len(decor) == t.decor
        assert all(0 < res.probability(i) < 1 for i in decor)
    tsx = ET.parse(tmp_path / "tileset.tsx").getroot()
    probs = {int(t.get("id")): float(t.get("probability")) for t in tsx.iter("tile") if t.get("probability")}
    assert set(probs) == {i for ids in res.terrain_tiles.values() for i in ids if res.probability(i) < 1}

    rng = np.random.default_rng(0)
    picks = [res.pick_terrain("grass", rng) for _ in range(2000)]
    share = np.mean([res.probability(p) < 1 for p in picks])
    assert 0.03 < share < 0.15


def test_decor_zero_disables_detail_tiles():
    spec = load("grassland").model_copy(deep=True)
    for t in spec.terrains:
        t.decor = 0
    res = generate(spec)
    assert all(len(res.terrain_tiles[t.name]) == t.variants for t in spec.terrains)


def test_object_shadows_are_translucent_and_preview_is_opaque(tmp_path: Path):
    from tileset_gen.palette import SHADOW

    spec = load("grassland")
    res = build(spec, tmp_path, map_size=(12, 8), preview_scale=1)
    tree = res.atlas.tiles[next(o for o in res.objects if o.name == "oak").tiles[0][0]]
    assert (tree == SHADOW).any()
    alpha = np.asarray(Image.open(tmp_path / "tileset.png").convert("RGBA"))[..., 3]
    assert ((alpha > 0) & (alpha < 255)).any(), "shadows should be semi-transparent in the atlas"
    assert set(np.unique(alpha)) <= {0, 255, int(res.lut[res.remap[SHADOW], 3])}
    preview = np.asarray(Image.open(tmp_path / "preview.png").convert("RGBA"))
    assert (preview[..., 3] == 255).all(), "shadows over ground must composite to opaque pixels"


def test_shadow_counts_against_color_budget():
    spec = load("grassland").model_copy(update={"max_colors": 16})
    res = generate(spec)
    used = np.unique(res.remap[res.atlas.sheet()])
    assert len(used[used > 0]) <= 16


def test_wall_without_top_tiles_vertically():
    from tileset_gen.generators.objects import WallParams, wall

    ctx = GenContext(16, 2, "w", Palette(), ["#8a8494"])
    mortar = ctx.ramp(0)[0]
    bare = wall(ctx, WallParams(top=False), 2, 2) == mortar
    assert np.array_equal(bare[:16], bare[16:]), "block joints must line up across tile rows"
    capped = wall(ctx, WallParams(top=True), 2, 2) == mortar
    assert np.array_equal(capped[:, :16], capped[:, 16:]), "the cap must still tile horizontally"
    assert not np.array_equal(capped[:16], capped[16:])


def test_preview_shadows_survive_palette_reduction(tmp_path: Path):
    """Shadowed ground uses darker shades that may appear nowhere in the atlas itself."""
    spec = TilesetSpec.model_validate(
        {
            "name": "wells",
            "seed": 3,
            "palette": [{"name": "grass", "hex": "#5aa83c"}],
            "terrains": [{"name": "grass", "generator": "grass", "colors": ["grass"], "variants": 2, "decor": 0}],
            "objects": [
                {"name": "well", "generator": "well", "width": 2, "height": 2, "place_on": ["grass"], "density": 1.0}
            ],
        }
    )
    build(spec, tmp_path, map_size=(8, 6), preview_scale=1)
    preview = np.asarray(Image.open(tmp_path / "preview.png").convert("RGBA"))
    assert (preview[..., 3] == 255).all()


def test_tone_patches_become_blended_terrains():
    from tileset_gen.build import PATCH_SUFFIX

    spec = load("grassland")
    res = generate(spec)
    for t in spec.terrains:
        patch = t.name + PATCH_SUFFIX
        assert res.patch_of[patch] == t.name
        ws = next(w for w in res.wangsets if (w.lower, w.upper) == (t.name, patch))
        assert set(ws.tiles) == set(ALL_CORNERS)
        # Darker: the patch's plain tile averages a lower luminance than its host's.
        lum = lambda tid: res.to_rgba(res.atlas.tiles[tid])[..., :3].astype(float).mean()  # noqa: E731
        assert lum(res.terrain_tiles[patch][0]) < lum(res.terrain_tiles[t.name][0])
    no_patches = spec.model_copy(deep=True)
    for t in no_patches.terrains:
        t.patches = 0.0
    assert generate(no_patches).patch_of == {}


def test_preview_shows_patches_and_keeps_them_off_the_elevation_chain():
    from tileset_gen.preview import make_preview, terrain_chain

    res = generate(load("grassland"))
    assert not set(terrain_chain(res)) & set(res.patch_of)
    pmap = make_preview(res, 32, 20)
    patch_tiles = {tid for p in res.patch_of for tid in res.terrain_tiles[p]}
    assert np.isin(pmap.ground, list(patch_tiles)).any()


def test_fence_links_up_in_the_preview():
    from tileset_gen.preview import make_preview

    res = generate(load("grassland"))
    cs = res.connected[0]
    assert set(cs.tiles) == set(range(16)) and cs.gate is not None
    mask_of = {tid: m for m, tid in cs.tiles.items()} | {cs.gate: 2 | 8}
    pmap = make_preview(res, 32, 20)
    fence = np.isin(pmap.objects, list(mask_of))
    assert fence.sum() >= 8 and (pmap.objects == cs.gate).any()
    h, w = fence.shape
    for y, x in zip(*np.nonzero(fence)):
        m = mask_of[int(pmap.objects[y, x])]
        for bit, back, dy, dx in ((1, 4, -1, 0), (2, 8, 0, 1), (4, 1, 1, 0), (8, 2, 0, -1)):
            ny, nx = y + dy, x + dx
            linked = 0 <= ny < h and 0 <= nx < w and fence[ny, nx] and mask_of[int(pmap.objects[ny, nx])] & back
            assert bool(m & bit) == bool(linked), (y, x, m)


def test_connected_objects_must_be_one_tile():
    spec = load("grassland").model_copy(deep=True)
    next(o for o in spec.objects if o.generator == "fence").width = 2
    assert any("width and height must be 1" in e for e in validate_spec(spec))


def test_landmarks_are_capped_in_the_preview():
    from tileset_gen.preview import make_preview

    spec = load("grassland").model_copy(deep=True)
    for o in spec.objects:
        o.density = 1.0
    res = generate(spec)
    pmap = make_preview(res, 24, 16)
    for o, group in ((o, next(g for g in res.objects if g.name == o.name)) for o in spec.objects if o.name != "fence"):
        limit = REGISTRY[o.generator].limit
        if limit:
            copies = int(np.sum(pmap.objects == group.tiles[0][0]))
            assert copies <= limit, (o.name, copies)


def test_characters_export_sheets_tilesets_and_map_objects(tmp_path: Path):
    spec = load("grassland")
    res = build(spec, tmp_path, map_size=(16, 12), preview_scale=1)
    assert [c.name for c in res.characters] == [c.name for c in spec.characters]
    meta = json.loads((tmp_path / "characters.json").read_text())
    for ch, doc in zip(res.characters, meta["characters"]):
        w, h = ch.size
        assert w == spec.tile_size and h in (spec.tile_size, spec.tile_size * 3 // 2)
        img = np.asarray(Image.open(tmp_path / doc["image"]).convert("RGBA"))
        assert img.shape == (4 * h, 3 * w, 4)
        opaque = {tuple(px[:3]) for px in img.reshape(-1, 4) if px[3] == 255}
        assert len(opaque) <= next(c for c in spec.characters if c.name == ch.name).max_colors
        assert ((img[..., 3] > 0) & (img[..., 3] < 255)).any(), "a translucent ground shadow"
        # "right" is "left" mirrored.
        assert np.array_equal(ch.frames[2], ch.frames[1][:, :, ::-1])
        tsx = ET.parse(tmp_path / "characters" / f"{ch.name}.tsx").getroot()
        assert tsx.get("tilecount") == "12" and tsx.get("objectalignment") == "bottom"
        assert len(list(tsx.iter("animation"))) == 4
    tmx = ET.parse(tmp_path / "preview.tmx").getroot()
    tilesets = [(int(t.get("firstgid")), t.get("source")) for t in tmx.iter("tileset")]
    assert tilesets[0] == (1, "tileset.tsx") and len(tilesets) == 1 + len(res.characters)
    objs = list(tmx.iter("object"))
    assert objs and len(objs) <= sum(c.count for c in spec.characters)
    last = tilesets[-1][0] + 12
    for o in objs:
        assert res.atlas.tile_count < int(o.get("gid")) < last


def test_characters_stay_off_water_and_objects(tmp_path: Path):
    from tileset_gen.preview import make_preview

    res = generate(load("grassland"))
    pmap = make_preview(res, 24, 16)
    names = {t.name for t in res.spec.terrains}
    allowed = {c.name: set(c.place_on) for c in res.spec.characters}
    for name, x, y, row in pmap.characters:
        assert pmap.objects[y, x] < 0 and 0 <= row < 4
        tid = int(pmap.ground[y, x])
        assert res.atlas.meta[tid].props.get("terrain") in names
        host = res.patch_of.get(res.atlas.meta[tid].props["terrain"], res.atlas.meta[tid].props["terrain"])
        assert host in allowed[name]


def test_character_validation():
    spec = load("grassland")
    spec.characters[0].generator = "tree"
    spec.characters[1].place_on = ["lava"]
    spec.characters[2].params = {"outfit": "bikini"}
    spec.characters[3].name = "oak"
    errors = "\n".join(validate_spec(spec))
    assert "unknown character generator 'tree'" in errors
    assert "unknown terrain 'lava'" in errors
    assert "outfit" in errors
    assert "'oak': duplicate name" in errors
