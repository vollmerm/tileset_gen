"""Pixel primitives operating on palette-index arrays (0 = transparent)."""

from __future__ import annotations

import numpy as np

from .noise import quantize
from .palette import Palette

LIGHT = np.array([-0.55, -0.7, 0.55])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def shift(arr: np.ndarray, dy: int, dx: int, fill=0) -> np.ndarray:
    """Non-wrapping shift: out[y, x] = arr[y - dy, x - dx]."""
    out = np.full_like(arr, fill)
    h, w = arr.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = arr[yd, xd]
    return out


def neighbors4(mask: np.ndarray) -> np.ndarray:
    """True where any 4-neighbor of a pixel is set."""
    return shift(mask, 1, 0, False) | shift(mask, -1, 0, False) | shift(mask, 0, 1, False) | shift(mask, 0, -1, False)


def grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w]
    return ys + 0.5, xs + 0.5


def ellipse_height(h: int, w: int, ellipses) -> np.ndarray:
    """Height field (0..1) from a union of ellipsoid bumps (cx, cy, rx, ry[, z])."""
    ys, xs = grid(h, w)
    out = np.zeros((h, w))
    for e in ellipses:
        cx, cy, rx, ry = e[:4]
        z = e[4] if len(e) > 4 else 1.0
        d = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2
        out = np.maximum(out, np.sqrt(np.clip(1.0 - d, 0.0, None)) * z)
    return out


def shade(
    height: np.ndarray,
    mask: np.ndarray,
    levels: list[int],
    dither: float = 0.6,
    ambient: float = 0.25,
    strength: float = 6.0,
) -> np.ndarray:
    """Lambert-shade a height field into `levels` (darkest first) inside `mask`."""
    gy, gx = np.gradient(height * strength)
    n = np.stack([-gx, -gy, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lam = np.clip(n @ LIGHT, 0.0, 1.0)
    light = np.clip(ambient + (1.0 - ambient) * lam, 0.0, 1.0)
    out = quantize(light, levels, dither)
    return np.where(mask, out, 0).astype(np.uint8)


def outline(canvas: np.ndarray, palette: Palette, color: int | None = None) -> np.ndarray:
    """Draw a 1px outline around opaque pixels.

    Without an explicit `color` this is a selective outline: each outline pixel
    takes the darkest shade of the ramp it borders.
    """
    out = canvas.copy()
    solid = canvas > 0
    border = ~solid & neighbors4(solid)
    if color is not None:
        out[border] = color
        return out
    for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        src = shift(canvas, dy, dx)
        take = border & (out == 0) & (src > 0)
        out[take] = np.vectorize(palette.outline_for, otypes=[np.uint8])(src[take]) if take.any() else 0
    return out


def rect(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
    """Fill the inclusive rectangle [x0, x1] x [y0, y1]."""
    canvas[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1] = color


def put(canvas: np.ndarray, x: int, y: int, color: int) -> None:
    h, w = canvas.shape
    if 0 <= x < w and 0 <= y < h:
        canvas[y, x] = color


def cylinder_levels(x: np.ndarray, x0: float, x1: float, levels: list[int]) -> np.ndarray:
    """Horizontal cylinder shading: lit slightly left of center, dark at the right edge."""
    t = (x - x0) / max(x1 - x0, 1e-6)
    light = np.clip(1.0 - np.abs(t - 0.35) * 1.6, 0.0, 1.0)
    k = np.clip((light * len(levels)).astype(int), 0, len(levels) - 1)
    return np.asarray(levels, dtype=np.uint8)[k]
