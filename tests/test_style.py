"""8-bit console styles: hardware colors, short ramps, black ink, tiny sprite palettes."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tileset_gen import registry
from tileset_gen.build import build, generate
from tileset_gen.context import GenContext
from tileset_gen.generators.characters import ROWS
from tileset_gen.palette import SHADOW, Palette, hex_to_rgb, to_lab
from tileset_gen.spec import TilesetSpec
from tileset_gen.style import NES_COLORS, STYLES
from tileset_gen.validate import merged_colors

EXAMPLES = Path(__file__).parent.parent / "examples" / "specs"
NES = {hex_to_rgb(h) for h in NES_COLORS}


def lightness(rgb) -> float:
    return float(to_lab(np.array(rgb, dtype=float))[0])


@pytest.mark.parametrize("base", ["#5aa83c", "#3a78c8", "#e2c47e", "#8b8794", "#dfe8f2", "#7a5230", "#202020"])
def test_nes_ramps_use_three_hardware_shades(base: str):
    pal = Palette({"c": base}, "nes")
    ramp = [pal.colors[i] for i in pal.ramp("c")]
    assert set(ramp) <= NES
    assert len(set(ramp)) <= 3
    assert [lightness(c) for c in ramp] == sorted(lightness(c) for c in ramp)


@pytest.mark.parametrize("base", ["#5aa83c", "#3a78c8", "#e2c47e", "#8b8794"])
def test_gbc_ramps_are_15_bit_with_four_shades(base: str):
    pal = Palette({"c": base}, "gbc")
    ramp = [pal.colors[i] for i in pal.ramp("c")]
    assert len(set(ramp)) == 4
    assert all(STYLES["gbc"].snap(c) == c for c in ramp), "already on the 5-bit grid"
    assert [lightness(c) for c in ramp] == sorted(lightness(c) for c in ramp)


def test_retro_outlines_are_ink():
    pal = Palette({"c": "#5aa83c"}, "nes")
    ink = pal.index((0, 0, 0))
    assert all(pal.outline_for(i) == ink for i in pal.ramp("c"))


def test_style_sets_default_budgets_and_restyled_resets_them():
    spec = TilesetSpec.load(EXAMPLES / "grassland.json")
    assert spec.max_colors == 48 and {c.max_colors for c in spec.characters} == {15}
    nes = spec.restyled("nes")
    assert nes.style == "nes" and nes.max_colors == 16 and {c.max_colors for c in nes.characters} == {3}
    assert nes.restyled("snes").max_colors == 48
    explicit = TilesetSpec.model_validate({**nes.model_dump(), "max_colors": 24})
    assert explicit.max_colors == 24
    assert TilesetSpec.model_validate_json(nes.model_dump_json()) == nes


@pytest.mark.parametrize("style", ["nes", "gbc"])
@pytest.mark.parametrize("name", sorted(registry.REGISTRY))
def test_generators_render_in_retro_styles(style: str, name: str):
    g = registry.get(name)
    ctx = GenContext(16, 3, f"retro:{name}", Palette(style=style), g.default_colors())
    p = g.params()
    if g.kind == "terrain":
        out = [g.fn(ctx, p, v, 0.0) for v in range(2)]
    elif g.connected:
        out = [g.fn(ctx, p, m) for m in range(16)]
    elif g.kind == "object":
        out = [g.fn(ctx, p, 2, 2)]
    elif g.kind == "character":
        out = [g.fn(ctx, p, a, d, k) for a, d in ROWS if d != "right" for k in range(3)]
    else:
        out = [g.fn(ctx, p, f, 4) for f in range(4)]
    for arr in out:
        colors = {ctx.palette.colors[i] for i in np.unique(arr) if i > SHADOW}
        if style == "nes":
            assert colors <= NES, name


def test_nes_build_stays_on_the_hardware(tmp_path: Path):
    spec = TilesetSpec.load(EXAMPLES / "nes_kingdom.json")
    res = build(spec, tmp_path, map_size=(12, 8), preview_scale=1)
    rgba = np.asarray(Image.open(tmp_path / "tileset.png").convert("RGBA"))
    assert set(np.unique(rgba[..., 3])) <= {0, 255}, "no translucent shadows on the NES"
    opaque = {tuple(int(v) for v in c[:3]) for c in rgba.reshape(-1, 4) if c[3]}
    assert opaque <= NES and len(opaque) <= 16
    for sheet in res.characters:
        assert not (sheet.frames == SHADOW).any()
        colors = {tuple(int(v) for v in row[:3]) for row in sheet.lut[1:]}
        assert len(colors) <= 3 and (0, 0, 0) in colors, sheet.name
        assert colors <= NES


def test_tiny_sprite_palettes_keep_skin_and_outfit():
    spec = TilesetSpec.load(EXAMPLES / "grassland.json").restyled("nes")
    res = generate(spec)
    for ch, sheet in zip(spec.characters, res.characters):
        if ch.generator != "humanoid":
            continue
        kept = {tuple(int(v) for v in row[:3]) for row in sheet.lut[1:]}
        ctx = GenContext(16, 0, "", res.palette, merged_colors(ch.colors, registry.get(ch.generator)))
        skin, outfit = (res.palette.colors[ctx.ramp(s)[2]] for s in (0, 2))
        assert {skin, outfit, (0, 0, 0)} == kept, ch.name


def test_merged_colors_show_the_color_they_kept():
    pal = Palette({"a": "#5aa83c"})
    ramp = pal.ramp("a")
    counts = np.zeros(len(pal.colors), dtype=int)
    counts[ramp] = [1, 1, 1000, 1, 1]
    remap, lut = pal.reduce(counts, 1)
    assert len(set(remap[ramp])) == 1
    assert tuple(lut[remap[ramp[2]], :3]) == pal.colors[ramp[2]], "the heavy middle shade survives"
