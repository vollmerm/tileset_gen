"""Console styles: how many shades a color gets, which colors exist at all, and how sprites are inked.

`snes` is the default 16-bit look: 5-step hue-shifted ramps in free RGB, selective
outlines and translucent cast shadows. The 8-bit styles restrict that:

- `nes`: every color snaps to the NES (2C02) hardware palette. A ramp has three
  shades (dark, base, light), like one NES sub-palette, and the shared black
  backdrop outlines sprites and objects. There are no translucent shadows.
- `gbc`: 15-bit color (5 bits per channel), four shades per ramp like a Game Boy
  Color palette, near-black outlines and no translucent shadows.

Generators always ask for 5-step ramps and index them 0-4. An 8-bit ramp repeats
shades to fill those steps, so the 2-vs-3 noise most textures are built from
collapses into flat fills, while shadows (steps 0-1) and highlights (step 4) stay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .palette import RGB, hex_to_rgb, make_ramp, to_lab

StyleName = Literal["snes", "nes", "gbc"]

# The NES 2C02 master palette as most emulators show it, without its duplicate blacks.
NES_COLORS: tuple[str, ...] = (
    # $00-$0C
    "#7c7c7c", "#0000fc", "#0000bc", "#4428bc", "#940084", "#a80020", "#a81000", "#881400",
    "#503000", "#007800", "#006800", "#005800", "#004058",
    # $10-$1C
    "#bcbcbc", "#0078f8", "#0058f8", "#6844fc", "#d800cc", "#e40058", "#f83800", "#e45c10",
    "#ac7c00", "#00b800", "#00a800", "#00a844", "#008888",
    # $20-$2D
    "#f8f8f8", "#3cbcfc", "#6888fc", "#9878f8", "#f878f8", "#f85898", "#f87858", "#fca044",
    "#f8b800", "#b8f818", "#58d854", "#58f898", "#00e8d8", "#787878",
    # $30-$3D, then black ($0F)
    "#fcfcfc", "#a4e4fc", "#b8b8f8", "#d8b8f8", "#f8b8f8", "#f8a4c0", "#f0d0b0", "#fce0a8",
    "#f8d878", "#d8f878", "#b8f8b8", "#b8f8d8", "#00fcfc", "#f8d8f8", "#000000",
)  # fmt: skip

MIN_STEP = 12.0  # CIELAB lightness between neighboring shades of a snapped ramp, so they stay distinct


@dataclass(frozen=True)
class Style:
    name: StyleName
    shades: int  # distinct colors per ramp
    layout: tuple[int, ...]  # which shade fills each of the 5 ramp steps, darkest first
    ink: RGB | None  # outline color for sprites and objects; None = selective (darkest of each ramp)
    shadows: bool  # translucent cast shadows under objects and characters
    max_colors: int  # default tileset color budget
    sprite_colors: int  # default per-character palette size (not counting transparency)
    emote_colors: int  # the shared emote balloon sheet: ink, white and the icons
    dither: float = 1.0  # scales generators' ordered dithering between shades
    simple: bool = False  # generators with an 8-bit branch draw fewer, bolder details (water, wood floors)
    hardware: tuple[RGB, ...] = ()  # allowed colors; empty = any RGB at `bits` per channel
    bits: int = 8

    @property
    def retro(self) -> bool:
        return self.name != "snes"

    def snap(self, rgb: RGB) -> RGB:
        """The nearest color this console can show."""
        if self.hardware:
            return self.hardware[int(np.argmin(self._dist(rgb)))]
        if self.bits >= 8:
            return rgb
        top = (1 << self.bits) - 1
        # Expand n-bit channels the way hardware does: replicate the high bits into the low ones.
        return tuple((v << (8 - self.bits)) | (v >> (2 * self.bits - 8)) for v in (round(c * top / 255) for c in rgb))

    def _dist(self, rgb: RGB) -> np.ndarray:
        d = _hw_lab(self.hardware) - to_lab(np.array(rgb, dtype=float))
        return np.sum(d * d, axis=-1)

    def ramp(self, base: RGB) -> list[RGB]:
        """Five steps, darkest first, built from `shades` distinct console colors."""
        if not self.retro:
            return make_ramp(base)
        # Extra-dark and extra-light targets for punchy 8-bit contrast. Hardware ramps mostly stay in one hue
        # column (sky blue, blue, dark blue), so shift hue less than the 16-bit ramps do.
        wide = make_ramp(base, 7, hue_shift=0.015)
        targets = {3: [wide[1], base, wide[5]], 4: [wide[0], wide[2], base, wide[5]]}[self.shades]
        mid = self.shades - 2
        shades: list[RGB] = [self.snap(base)] * self.shades
        for i in range(mid - 1, -1, -1):
            shades[i] = self._step(targets[i], shades[i + 1], darker=True)
        for i in range(mid + 1, self.shades):
            shades[i] = self._step(targets[i], shades[i - 1], darker=False)
        return [shades[k] for k in self.layout]

    def _step(self, target: RGB, neighbor: RGB, darker: bool) -> RGB:
        """Snap `target`, but keep it clearly darker (or lighter) than its `neighbor` shade."""
        sign = -1 if darker else 1

        def ok(c: RGB) -> bool:
            return sign * (_lightness(c) - _lightness(neighbor)) >= MIN_STEP

        snapped = self.snap(target)
        if ok(snapped) or not self.hardware:
            return snapped
        return next((self.hardware[i] for i in np.argsort(self._dist(target)) if ok(self.hardware[i])), neighbor)


def _lightness(c: RGB) -> float:
    return float(to_lab(np.array(c, dtype=float))[0])


_LAB_CACHE: dict[tuple[RGB, ...], np.ndarray] = {}


def _hw_lab(colors: tuple[RGB, ...]) -> np.ndarray:
    if colors not in _LAB_CACHE:
        _LAB_CACHE[colors] = to_lab(np.array(colors, dtype=float))
    return _LAB_CACHE[colors]


STYLES: dict[str, Style] = {
    "snes": Style(
        "snes", shades=5, layout=(0, 1, 2, 3, 4), ink=None, shadows=True, max_colors=48, sprite_colors=15,
        emote_colors=32,
    ),
    "nes": Style(
        "nes", shades=3, layout=(0, 0, 1, 1, 2), ink=(0, 0, 0), shadows=False, max_colors=16, sprite_colors=3,
        emote_colors=4, dither=0.0, simple=True, hardware=tuple(hex_to_rgb(h) for h in NES_COLORS),
    ),
    "gbc": Style(
        "gbc", shades=4, layout=(0, 1, 2, 2, 3), ink=(16, 16, 24), shadows=False, max_colors=32, sprite_colors=3,
        emote_colors=4, dither=0.4, bits=5,
    ),
}  # fmt: skip


def get(name: str) -> Style:
    return STYLES[name]
