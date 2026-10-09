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
    for wt in tsx.iter("wangtile"):
        assert int(wt.get("tileid")) in res.atlas.tiles
        assert len(wt.get("wangid").split(",")) == 8

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
