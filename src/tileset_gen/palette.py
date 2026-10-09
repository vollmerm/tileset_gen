"""Color ramps and the indexed palette shared by every generator.

Index 0 is always fully transparent. Every other index is an opaque RGB color.
Ramps run darkest -> lightest; shadows drift toward blue/purple and highlights
toward warm yellow, the classic hue-shifted shading of 16-bit sprite work.
"""

from __future__ import annotations

import colorsys
import re

import numpy as np

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")

RGB = tuple[int, int, int]

SHADOW_HUE = 0.68  # blue-violet
HIGHLIGHT_HUE = 0.14  # warm yellow


def hex_to_rgb(value: str) -> RGB:
    if not HEX_RE.match(value):
        raise ValueError(f"not a hex color: {value!r}")
    v = value.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def rgb_to_hex(color: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _toward(h: float, target: float, amount: float) -> float:
    """Move hue `h` toward `target` along the shorter arc by at most `amount`."""
    d = ((target - h + 0.5) % 1.0) - 0.5
    return (h + max(-amount, min(amount, d))) % 1.0


def make_ramp(base: RGB, steps: int = 5, hue_shift: float = 0.035) -> list[RGB]:
    """Build a hue-shifted shading ramp with `base` at the middle step."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in base))
    gray = s < 0.06
    mid = steps // 2
    out: list[RGB] = []
    for i in range(steps):
        k = i - mid
        if k < 0:
            hh = SHADOW_HUE if gray else _toward(h, SHADOW_HUE, hue_shift * -k)
            ss = min(1.0, s + (0.05 if gray else 0.09) * -k)
            vv = v * (0.74**-k)
        elif k > 0:
            hh = HIGHLIGHT_HUE if gray else _toward(h, HIGHLIGHT_HUE, hue_shift * k)
            ss = max(0.0, s - (0.0 if gray else 0.11) * k + (0.03 if gray else 0.0))
            vv = min(1.0, v + (1.0 - v) * 0.42 * k + 0.05 * k)
        else:
            hh, ss, vv = h, s, v
        r, g, b = colorsys.hsv_to_rgb(hh, ss, vv)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return out


class Palette:
    """Indexed palette built up lazily as generators request ramps."""

    def __init__(self, named: dict[str, str] | None = None):
        self.named: dict[str, RGB] = {k.lower(): hex_to_rgb(v) for k, v in (named or {}).items()}
        self.colors: list[RGB] = [(0, 0, 0)]
        self._index: dict[RGB, int] = {}
        self._ramps: dict[tuple[RGB, int], list[int]] = {}
        self._outline: dict[int, int] = {}

    def resolve(self, color: str) -> RGB:
        key = color.lower()
        if key in self.named:
            return self.named[key]
        if HEX_RE.match(color):
            return hex_to_rgb(color)
        raise KeyError(f"unknown color {color!r} (not a palette name or hex value)")

    def is_known(self, color: str) -> bool:
        return color.lower() in self.named or bool(HEX_RE.match(color))

    def index(self, rgb: RGB) -> int:
        idx = self._index.get(rgb)
        if idx is None:
            idx = len(self.colors)
            self.colors.append(rgb)
            self._index[rgb] = idx
        return idx

    def ramp(self, color: str, steps: int = 5) -> list[int]:
        """Palette indices for a ramp around `color`, darkest first."""
        base = self.resolve(color)
        key = (base, steps)
        if key not in self._ramps:
            idxs = [self.index(c) for c in make_ramp(base, steps)]
            self._ramps[key] = idxs
            for i in idxs:
                self._outline.setdefault(i, idxs[0])
        return self._ramps[key]

    def outline_for(self, idx: int) -> int:
        """Darkest color of the ramp `idx` belongs to (used for selective outlines)."""
        return self._outline.get(idx, idx)

    def reduce(self, counts: np.ndarray, max_colors: int) -> np.ndarray:
        """Merge the closest rarely-used colors until at most `max_colors` opaque remain.

        Returns a remap table: new_index = remap[old_index]. Index 0 stays 0.
        """
        n = len(self.colors)
        remap = np.arange(n)
        alive = [i for i in range(1, n) if counts[i] > 0]
        weight = {i: int(counts[i]) for i in alive}
        cols = np.array(self.colors, dtype=float)
        lum = np.array([0.30, 0.59, 0.11])

        def dist(a: int, b: int) -> float:
            d = cols[a] - cols[b]
            return float(np.sum(d * d * lum))

        while len(alive) > max_colors:
            best = None
            for ai, a in enumerate(alive):
                for b in alive[ai + 1 :]:
                    d = dist(a, b)
                    if best is None or d < best[0]:
                        best = (d, a, b)
            _, a, b = best
            keep, drop = (a, b) if weight[a] >= weight[b] else (b, a)
            weight[keep] += weight[drop]
            remap[remap == drop] = keep
            alive.remove(drop)

        # Compact surviving indices into 1..k, preserving first-use order.
        compact = {0: 0}
        for i in range(1, n):
            target = int(remap[i])
            if counts[i] == 0 and target == i:
                continue
            if target not in compact:
                compact[target] = len(compact)
        out = np.zeros(n, dtype=np.int64)
        for i in range(n):
            out[i] = compact.get(int(remap[i]), 0)
        return out

    def rgba_lut(self, remap: np.ndarray | None = None) -> np.ndarray:
        """(N, 4) uint8 lookup table; with a remap, rows are the compacted palette."""
        if remap is None:
            lut = np.array([(*c, 255) for c in self.colors], dtype=np.uint8)
        else:
            size = int(remap.max()) + 1
            lut = np.zeros((size, 4), dtype=np.uint8)
            for old, new in enumerate(remap):
                if new:
                    lut[new] = (*self.colors[old], 255)
        lut[0] = (0, 0, 0, 0)
        return lut
