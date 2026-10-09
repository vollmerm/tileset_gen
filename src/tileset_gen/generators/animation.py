"""Standalone animated tiles.

Signature: fn(ctx, params, frame, n_frames) -> uint8[T, T]. Every frame is
drawn from the same seeded RNGs with time-periodic noise so the loop is seamless.
Terrain animations (water, lava) instead re-run the terrain generator with a phase.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..draw import grid, outline, put, rect
from ..noise import Noise, quantize
from ..registry import Params, register


class TorchParams(Params):
    flame: float = Field(0.6, ge=0.2, le=1.0, description="flame size")


@register(
    "torch",
    "animation",
    TorchParams,
    colors=[("flame", "#f08a24"), ("handle", "#6b4528")],
    doc="Wall or standing torch with a flickering flame.",
)
def torch(ctx: GenContext, p: TorchParams, frame: int, n_frames: int) -> np.ndarray:
    T = ctx.T
    c = np.zeros((T, T), dtype=np.uint8)
    fl, hr = ctx.ramp(0), ctx.ramp(1)
    cx = T / 2
    rect(c, int(cx) - 1, int(T * 0.55), int(cx), T - 2, hr[2])
    rect(c, int(cx), int(T * 0.55), int(cx), T - 2, hr[1])
    rect(c, int(cx) - 2, int(T * 0.5), int(cx) + 1, int(T * 0.55), hr[3])
    c = outline(c, ctx.palette)
    t = frame / n_frames
    wob = Noise(ctx.rng("flicker"), T, cells=3, tcells=n_frames).tile(t=t)
    sway = np.sin(2 * np.pi * t) * 0.8
    ys, xs = grid(T, T)
    base_y = T * 0.5
    height = T * 0.42 * p.flame * (0.9 + 0.2 * np.cos(2 * np.pi * t * 2))
    rel = (base_y - ys) / height
    width = T * 0.18 * p.flame * np.clip(1.0 - rel, 0, 1) ** 0.7 * (1.0 + 0.4 * (rel < 0.3))
    dx = xs - cx - sway * np.clip(rel, 0, 1)
    flame = (rel >= -0.15) & (rel <= 1.0) & (np.abs(dx) <= width + (wob - 0.5) * 1.5)
    heat = np.clip(1.0 - np.abs(dx) / (width + 0.5) * 0.7 - rel * 0.6 + (wob - 0.5) * 0.3, 0, 1)
    flame_px = quantize(heat, [fl[1], fl[2], fl[3], fl[4]], dither=0.4)
    c = np.where(flame, flame_px, c).astype(np.uint8)
    rng = ctx.rng("spark", frame)
    if rng.random() < 0.6:
        put(c, int(cx + rng.integers(-2, 3)), int(base_y - height - rng.integers(1, 3)), fl[4])
    return c


class WaterfallParams(Params):
    foam: float = Field(0.4, ge=0.0, le=1.0, description="white water amount")


@register(
    "waterfall",
    "animation",
    WaterfallParams,
    colors=[("water", "#4a8ad8")],
    doc="Falling water that tiles vertically and horizontally; scrolls downward.",
)
def waterfall(ctx: GenContext, p: WaterfallParams, frame: int, n_frames: int) -> np.ndarray:
    T = ctx.T
    r = ctx.ramp(0)
    t = frame / n_frames
    f = Noise(ctx.rng("streaks"), T, cells=(6, 1), octaves=2).tile(dy=-t * T)
    g = Noise(ctx.rng("foam"), T, cells=(4, 2)).tile(dy=-t * T)
    tile = quantize(np.clip((f - 0.3) * 2.0, 0, 1), [r[1], r[2], r[3]], dither=0.5)
    tile[g > 0.75 - p.foam * 0.2] = r[4]
    return tile


class CrystalParams(Params):
    size: float = Field(0.7, ge=0.3, le=1.0)


@register(
    "crystal",
    "animation",
    CrystalParams,
    colors=[("crystal", "#7ad8e8")],
    doc="Glowing save-point style crystal that pulses and twinkles.",
)
def crystal(ctx: GenContext, p: CrystalParams, frame: int, n_frames: int) -> np.ndarray:
    T = ctx.T
    r = ctx.ramp(0)
    ys, xs = grid(T, T)
    cx, cy = T / 2, T / 2
    bob = np.sin(2 * np.pi * frame / n_frames) * 0.9
    hw, hh = T * 0.25 * p.size, T * 0.42 * p.size
    d = np.abs(xs - cx) / hw + np.abs(ys - (cy + bob)) / hh
    gem = d <= 1.0
    c = np.zeros((T, T), dtype=np.uint8)
    c[gem] = np.where(xs < cx, r[3], r[2])[gem]
    c[gem & (ys < cy + bob) & (xs < cx)] = r[4]
    c[gem & (ys > cy + bob) & (xs > cx)] = r[1]
    c = outline(c, ctx.palette)
    pulse = frame % n_frames
    sx, sy = int(cx - hw * 0.4), int(cy + bob - hh * 0.4)
    if pulse in (0, n_frames // 2):
        put(c, sx, sy, r[4])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(c, sx + dx, sy + dy, r[3] if pulse == 0 else c[sy + dy, sx + dx])
    return c
