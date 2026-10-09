"""Periodic noise, ordered dithering and Voronoi cells.

Everything here wraps on a pixel period (normally the tile size) so generated
tiles repeat without seams. Noise can also loop in time for animations.
"""

from __future__ import annotations

import numpy as np

BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return a + (b - a) * t


class Noise:
    """Fractal value noise that repeats every `period` pixels on both axes.

    `cells` is the lattice resolution of the first octave (an int, or (cx, cy)
    for stretched noise). `tcells` > 1 adds a time axis that loops over t in [0, 1).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        period: int,
        cells: int | tuple[int, int] = 4,
        octaves: int = 1,
        persistence: float = 0.5,
        tcells: int = 1,
    ):
        cx, cy = (cells, cells) if isinstance(cells, int) else cells
        self.period = period
        self.tcells = tcells
        self.layers = []
        amp, total = 1.0, 0.0
        for o in range(octaves):
            sx, sy = cx * 2**o, cy * 2**o
            self.layers.append((rng.random((tcells, sy, sx)), sx, sy, amp))
            total += amp
            amp *= persistence
        self.total = total

    def sample(self, xs: np.ndarray, ys: np.ndarray, t: float = 0.0) -> np.ndarray:
        out = np.zeros(np.broadcast(xs, ys).shape)
        tt = t * self.tcells
        t0 = int(np.floor(tt))
        ft = _fade(np.asarray(tt - t0))
        for grid, sx, sy, amp in self.layers:
            u = xs * sx / self.period
            v = ys * sy / self.period
            x0 = np.floor(u).astype(int)
            y0 = np.floor(v).astype(int)
            fx, fy = _fade(u - x0), _fade(v - y0)
            x0 %= sx
            y0 %= sy
            x1, y1 = (x0 + 1) % sx, (y0 + 1) % sy

            def plane(g):
                return _lerp(_lerp(g[y0, x0], g[y0, x1], fx), _lerp(g[y1, x0], g[y1, x1], fx), fy)

            ta, tb = t0 % self.tcells, (t0 + 1) % self.tcells
            out += amp * _lerp(plane(grid[ta]), plane(grid[tb]), ft)
        return out / self.total

    def tile(self, size: int | None = None, t: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        n = size or self.period
        ys, xs = np.mgrid[0:n, 0:n] + 0.5
        return self.sample(xs + dx, ys + dy, t)


def contrast(field: np.ndarray, amount: float, center: float = 0.5) -> np.ndarray:
    return np.clip((field - center) * amount + 0.5, 0.0, 1.0)


def quantize(
    field: np.ndarray,
    levels: list[int],
    dither: float = 0.5,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
) -> np.ndarray:
    """Map a [0, 1] field onto palette `levels` with 4x4 ordered dithering."""
    h, w = field.shape
    if xs is None or ys is None:
        ys, xs = np.mgrid[0:h, 0:w]
    thr = (BAYER4[ys.astype(int) % 4, xs.astype(int) % 4] - 0.5) * dither
    n = len(levels)
    k = np.clip(np.floor(field * n + thr), 0, n - 1).astype(int)
    return np.asarray(levels, dtype=np.uint8)[k]


def voronoi(rng: np.random.Generator, size: int, grid: int, jitter: float = 0.8):
    """Periodic jittered-grid Voronoi. Returns (cell_id, f1, f2) arrays of shape (size, size)."""
    cell = size / grid
    gy, gx = np.mgrid[0:grid, 0:grid]
    px = ((gx + 0.5 + (rng.random((grid, grid)) - 0.5) * jitter) * cell).ravel()
    py = ((gy + 0.5 + (rng.random((grid, grid)) - 0.5) * jitter) * cell).ravel()
    ys, xs = np.mgrid[0:size, 0:size] + 0.5
    dx = np.abs(xs[None] - px[:, None, None])
    dy = np.abs(ys[None] - py[:, None, None])
    dx = np.minimum(dx, size - dx)
    dy = np.minimum(dy, size - dy)
    d = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(d, axis=0)
    f = np.take_along_axis(d, order[:2], axis=0)
    return order[0], f[0], f[1]
