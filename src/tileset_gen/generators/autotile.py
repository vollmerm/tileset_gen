"""Corner-based Wang transition tiles (16 per terrain pair).

Corners are ordered (NW, NE, SE, SW); a 1 means the *upper* terrain owns that
corner. The boundary comes from bilinear interpolation of the corner values
plus tile-periodic noise, so neighboring tiles meet seamlessly: along a shared
edge both tiles see the same two corners and the same noise.

Masks are computed on a padded canvas so edge styling (outlines, cliff faces)
can look a few pixels past the tile border.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..draw import neighbors4, shift
from ..noise import Noise

Corners = tuple[int, int, int, int]

ALL_CORNERS: list[Corners] = [c for c in itertools.product((0, 1), repeat=4)]
MIXED_CORNERS: list[Corners] = [c for c in ALL_CORNERS if 0 < sum(c) < 4]

PAD = 4


def corner_mask(T: int, corners: Corners, noise: Noise, roughness: float, pad: int = PAD) -> np.ndarray:
    """Boolean mask (T+2*pad square), True where the upper terrain shows."""
    nw, ne, se, sw = corners
    ys, xs = np.mgrid[-pad : T + pad, -pad : T + pad] + 0.5
    u = np.clip(xs / T, 0.0, 1.0)
    v = np.clip(ys / T, 0.0, 1.0)
    top = nw + (ne - nw) * u
    bottom = sw + (se - sw) * u
    f = top + (bottom - top) * v
    g = f + (noise.sample(xs, ys) - 0.5) * (0.15 + roughness * 0.55)
    return g > 0.5


def compose(
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray,
    style: str,
    lower_ramp: list[int],
    upper_ramp: list[int],
    pad: int = PAD,
) -> np.ndarray:
    """Composite two seamless terrain tiles through a padded mask, then style the edge."""
    T = lower.shape[0]
    lo = np.pad(lower, pad, mode="wrap")
    up = np.pad(upper, pad, mode="wrap")
    out = np.where(mask, up, lo).astype(np.uint8)
    a, b = ~mask, mask
    up_edge = b & neighbors4(a)
    low_edge = a & neighbors4(b)
    below_upper = a & shift(b, 1, 0, False)

    if style == "soft":
        out[up_edge] = upper_ramp[1]
        out[below_upper] = lower_ramp[1]
    elif style == "shore":
        out[low_edge] = lower_ramp[4]
        out[low_edge & shift(low_edge, 0, 1, False) & ~shift(b, 1, 0, False)] = lower_ramp[3]
        out[up_edge] = upper_ramp[1]
    elif style == "blend":
        # Tone change within one kind of ground: no outline, just a dithered seam.
        ys, xs = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
        checker = (xs + ys) % 2 == 0
        out[up_edge & checker] = lo[up_edge & checker]
        out[low_edge & checker] = up[low_edge & checker]
    elif style == "cliff":
        depth = max(3, T // 4)
        ys, xs = np.mgrid[0 : out.shape[0], 0 : out.shape[1]]
        dist = np.zeros(out.shape, dtype=int)
        for d in range(depth, 0, -1):
            dist[a & shift(b, d, 0, False)] = d
        face = dist > 0
        stripes = (xs + (dist // 2)) % 3 == 0
        out[face] = upper_ramp[1]
        out[face & stripes] = upper_ramp[0]
        out[face & (dist == 1)] = upper_ramp[0]
        shadow = a & ~face & shift(face, 1, 0, False)
        out[shadow] = lower_ramp[0]
        lip = b & shift(face, -1, 0, False)
        out[lip] = upper_ramp[4]
        out[up_edge & ~lip] = upper_ramp[1]
    return out[pad : pad + T, pad : pad + T]
