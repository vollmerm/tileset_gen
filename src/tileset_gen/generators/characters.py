"""Walking character sprites in the style of 8/16-bit JRPG field sprites.

Signature: fn(ctx, params, direction, frame) -> uint8[h, w], with direction one
of "down", "left" or "up" and frame 0-2 (step, stand, other step: the
RPG Maker order, played 0-1-2-1). The builder mirrors "left" to get "right",
as the SNES games did. Every frame of a character has the same size: one tile
wide and 1x (chibi, 8-bit) or 1.5x (classic, 16-bit) a tile tall.

Sprites follow the tileset's rules so they sit in it naturally: colors come from
the same hue-shifted ramps, light comes from the upper left, shapes get the
selective outline and a translucent `SHADOW` ellipse on the ground.

Shapes are laid out in units of a 16px-wide sprite and rasterized at the tile
size, so a 32px sprite has the same proportions with finer shading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..draw import cylinder_levels, ellipse_height, ground_shadow, neighbors4, outline, shade, shift
from ..registry import Params, register

DIRECTIONS = ("down", "left", "right", "up")
FRAMES = 3
WALK_CYCLE = (0, 1, 2, 1)
EYE = "#282038"
WHITE = "#f4f0e8"


class _Figure:
    """A canvas addressed in 16px-sprite units: masks from shapes, plus shaded fills."""

    def __init__(self, ctx: GenContext, height_units: float):
        self.ctx = ctx
        self.s = s = ctx.T / 16
        H = int(round(height_units * s))
        self.c = np.zeros((H, ctx.T), dtype=np.uint8)
        ys, xs = np.mgrid[0:H, 0 : ctx.T]
        self.ys, self.xs = (ys + 0.5) / s, (xs + 0.5) / s

    def box(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        return (self.xs >= x0) & (self.xs < x1) & (self.ys >= y0) & (self.ys < y1)

    def oval(self, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        return ((self.xs - cx) / rx) ** 2 + ((self.ys - cy) / ry) ** 2 <= 1.0

    def trapezoid(self, cx: float, y0: float, y1: float, hw0: float, hw1: float) -> np.ndarray:
        """Rows y0..y1, half-width growing linearly from hw0 to hw1 (a cone when hw0 = 0)."""
        t = np.clip((self.ys - y0) / max(y1 - y0, 1e-6), 0, 1)
        return (self.ys >= y0) & (self.ys < y1) & (np.abs(self.xs - cx) <= hw0 + (hw1 - hw0) * t)

    def cyl(self, mask: np.ndarray, ramp: list[int], lo: int = 1, hi: int = 4) -> None:
        """Fill with horizontal cylinder shading (per row), darkening the bottom edge."""
        if not mask.any():
            return
        levels = ramp[lo:hi]
        cols = np.where(mask, np.arange(mask.shape[1])[None, :], -1)
        x1 = cols.max(axis=1, keepdims=True) + 1
        x0 = np.where(mask, np.arange(mask.shape[1])[None, :], 10**6).min(axis=1, keepdims=True)
        xs = np.arange(mask.shape[1])[None, :] + 0.5
        fill = cylinder_levels(np.broadcast_to(xs, mask.shape), x0, x1, levels)
        self.c[mask] = fill[mask]
        self.c[mask & ~shift(mask, -1, 0, False)] = ramp[max(lo - 1, 0)]

    def dome(self, mask: np.ndarray, ramp: list[int], cx: float, cy: float, rx: float, ry: float, lo: int = 1) -> None:
        """Fill with ellipsoid shading lit from the upper left, using ramp[lo:lo + 4]."""
        s = self.s
        H, W = self.c.shape
        height = ellipse_height(H, W, [(cx * s, cy * s, rx * s, ry * s)])
        lit = shade(height, mask, ramp[lo : lo + 4], dither=0.0, ambient=0.3, strength=4.0)
        self.c[mask] = lit[mask]

    def flat(self, mask: np.ndarray, color: int) -> None:
        self.c[mask] = color

    def finish(self, cx: float, ground: float, rx: float, shadow: bool = True) -> np.ndarray:
        c = outline(self.c, self.ctx.palette)
        if not shadow:
            return c
        s = self.s
        return ground_shadow(c, (cx + 0.6) * s, ground * s, rx * s, max(1.2, 1.1 * s))


@dataclass(frozen=True)
class _Body:
    height: float  # canvas height in units
    ground: float  # bottom of the feet
    torso_top: float
    waist: float
    hw: float  # torso half-width (front view)
    head_cy: float
    head_rx: float
    head_ry: float


BODIES = {
    # FF6/Chrono Trigger field sprites: 16x24, a big head on a short body.
    "classic": _Body(24, 22.6, 11.4, 17.4, 3.6, 6.8, 5.3, 4.9),
    # Dragon Quest/early FF overworld sprites: 16x16, mostly head.
    "chibi": _Body(16, 15.2, 9.6, 12.8, 3.2, 5.4, 5.2, 4.4),
}


class HumanoidParams(Params):
    body: Literal["classic", "chibi"] = Field(
        "classic", description="classic: 1x1.5 tiles (16-bit, FF6); chibi: 1x1 tile (8-bit, Dragon Quest)"
    )
    outfit: Literal["tunic", "robe", "armor", "dress"] = Field(
        "tunic", description="tunic + belt over trousers; long robe; plate armor with pauldrons; dress"
    )
    hair: Literal["short", "long", "spiky", "ponytail", "bald"] = Field("short")
    headgear: Literal["none", "hood", "wizard_hat", "helmet", "bandana", "crown"] = Field(
        "none", description="hood and wizard hat use the outfit color; the rest use the accent color"
    )
    cape: bool = Field(False, description="cape in the accent color")
    beard: bool = Field(False, description="beard in the hair color (elders, dwarves, kings)")


def _step(frame: int) -> int:
    """-1 / 0 / +1: which leg is forward in this walk frame."""
    return (-1, 0, 1)[frame]


@register(
    "humanoid",
    "character",
    HumanoidParams,
    colors=[
        ("skin", "#f0c49a"),
        ("hair", "#6a4028"),
        ("outfit", "#3a68b8"),
        ("accent", "#e0b640"),
        ("legs", "#5a4a40"),
    ],
    doc="Person: hero, party member, townsfolk, guard, mage, king... Mix outfit, hair and headgear.",
)
def humanoid(ctx: GenContext, p: HumanoidParams, direction: str, frame: int) -> np.ndarray:
    b = BODIES[p.body]
    f = _Figure(ctx, b.height)
    skin, hair, cloth, accent, legs = (ctx.ramp(i) for i in range(5))
    eye = ctx.palette.ramp(EYE)[1]
    step = _step(frame)
    cx = 8.0
    side = direction == "left"
    back = direction == "up"
    hw = b.hw - (0.9 if side else 0.0)
    robe = p.outfit == "robe"
    hem = {"robe": b.ground - 0.7, "dress": (b.waist + b.ground) / 2 + 0.6}.get(p.outfit)
    leg_len = b.ground - b.waist
    chibi = p.body == "chibi"

    # Cape behind the body.
    if p.cape:
        if back:
            cape = f.trapezoid(cx, b.torso_top - 0.2, b.ground - 1.6, hw + 1.6, hw + 2.4)
        elif side:
            sway = 0.6 * abs(step)
            cape = f.trapezoid(cx + 1.2, b.torso_top, b.ground - 1.4, 1.6, 3.0 + sway) & (f.xs > cx - 0.5)
        else:
            cape = f.trapezoid(cx, b.torso_top + 0.4, b.waist + leg_len * 0.6, hw + 1.6, hw + 2.2)
        f.cyl(cape, accent, 0, 3 if not back else 4)

    # Legs and feet.
    foot_h = 1.2 if not chibi else 1.0
    if side:
        legs_xy = [(cx - 1.4 + 2.0 * step, 0), (cx - 1.4 - 2.0 * step, 1)] if step else [(cx - 1.5, 0)]
        for x0, far in sorted(legs_xy, key=lambda v: -v[1]):
            leg = f.box(x0, b.waist - 0.5, x0 + 2.8, b.ground)
            f.cyl(leg, legs, 0 if far else 1, 3 if far else 4)
            boot = f.box(x0 - 0.9, b.ground - foot_h, x0 + 2.6, b.ground)
            f.flat(boot, legs[0] if far else legs[1])
    else:
        for i, (x0, x1) in enumerate(((cx - 3.0, cx - 0.1), (cx + 0.1, cx + 3.0))):
            lift = 1.0 if step == (-1 if i == 0 else 1) else 0.0
            bottom = b.ground - lift
            leg = f.box(x0, b.waist - 0.5, x1, bottom)
            f.cyl(leg, legs, 1, 4)
            f.flat(f.box(x0, bottom - foot_h, x1, bottom) & leg, legs[1] if not back else legs[0])
            if not back:
                f.flat(f.box(x0, bottom - foot_h, x1, bottom - foot_h + 0.6) & leg & (f.xs < x0 + 1.4), legs[2])

    # Skirt of a robe or dress, over the legs.
    if hem is not None:
        if side:
            sway = 0.8 * step
            skirt = f.trapezoid(cx + 0.2, b.waist - 0.8, hem, hw + 0.2, hw + 1.2) | (
                f.trapezoid(cx + 0.2 + sway, b.waist - 0.8, hem, hw + 0.2, hw + 1.2) & (f.ys > hem - 2)
            )
        else:
            skirt = f.trapezoid(cx, b.waist - 0.8, hem, hw, hw + 1.3)
        f.cyl(skirt, cloth, 1, 4)
        f.flat(skirt & ~shift(skirt, -max(1, int(f.s)), 0, False), accent[2] if robe else cloth[1])
        if robe and not side and not back:
            f.flat(skirt & (np.abs(f.xs - cx) < 0.5), accent[3])

    # Torso.
    shoulder = 0.8
    torso = f.trapezoid(cx, b.torso_top, b.torso_top + shoulder, hw - 0.9, hw) | f.box(
        cx - hw, b.torso_top + shoulder, cx + hw, b.waist + (0.6 if p.outfit == "tunic" else 0)
    )
    if side:
        torso = torso | f.box(cx - hw - 0.4, b.torso_top + 1.2, cx - hw + 0.5, b.waist - 0.8)  # chest
    f.cyl(torso, cloth, 1, 4)
    if p.outfit == "armor":
        plate = torso & f.box(cx - hw + 1, b.torso_top + 1, cx + hw - 1.2, b.waist - 1.2)
        if not back:
            f.flat(plate & (f.xs < cx - 0.3), cloth[4])
    if p.outfit in ("tunic", "armor", "dress"):
        f.flat(torso & f.box(0, b.waist - 1.0, 16, b.waist), accent[2])
        if not back and not side:
            f.flat(torso & f.box(cx - 0.6, b.waist - 1.0, cx + 0.6, b.waist), accent[4])
    if not back and not side and p.outfit != "armor":
        neckline = f.trapezoid(cx, b.torso_top, b.torso_top + 1.2, 1.2, 0.2)
        f.flat(neckline & torso, accent[3] if robe else cloth[1])

    # Arms: both in front and back view, only the near one from the side.
    arm_w = 1.9
    arm_top, arm_bottom = b.torso_top + 0.5, b.waist + 0.3
    hand = 1.3
    if side:
        swing = -1.6 * step
        x0 = cx - 0.6 + swing
        arm = f.box(x0, arm_top, x0 + arm_w, arm_bottom - abs(step) * 0.4)
        f.cyl(arm & (f.ys < arm_bottom - hand - abs(step) * 0.4), cloth, 1, 4)
        f.dome(arm & (f.ys >= arm_bottom - hand - abs(step) * 0.4), skin, x0 + 0.8, arm_bottom - 0.6, 1.2, 1.0)
        f.flat(arm & ~shift(arm, 0, 1, False) & (f.ys < arm_bottom - hand), cloth[0] if not robe else cloth[1])
    else:
        for i, x0 in enumerate((cx - hw - arm_w + 0.2, cx + hw - 0.2)):
            fwd = (step == (1 if i == 0 else -1)) if not back else (step == (-1 if i == 0 else 1))
            bottom = arm_bottom + (0.8 if fwd else 0.0)
            arm = f.box(x0, arm_top, x0 + arm_w, bottom)
            f.cyl(arm & (f.ys < bottom - hand), cloth, 1 if i == 0 else 0, 4 if i == 0 else 3)
            f.dome(arm & (f.ys >= bottom - hand), skin, x0 + arm_w / 2, bottom - 0.6, 1.2, 1.0)
            if robe:
                f.flat(arm & f.box(0, bottom - hand - 0.9, 16, bottom - hand), accent[2])
    if p.outfit == "armor":
        for x in (cx - hw - 0.4, cx + hw + 0.4) if not side else (cx + 0.3 - 1.6 * step,):
            pad = f.oval(x, b.torso_top + 0.9, 1.6, 1.3)
            f.dome(pad, accent, x, b.torso_top + 0.9, 1.6, 1.3)

    # Head.
    hy, rx, ry = b.head_cy, b.head_rx, b.head_ry
    head = f.oval(cx, hy, rx, ry)
    f.dome(head, skin, cx - 0.5, hy - 0.5, rx, ry)
    _hair(f, p, hair, cx, hy, rx, ry, direction, b)
    if p.beard and not back:
        if side:
            beard = head & (f.ys > hy + 1.2) & (f.xs < cx + 1.0) & (f.xs > cx - rx + 0.6)
            beard |= f.trapezoid(cx - 2.2, hy + 2.0, hy + ry + 2.0, 1.8, 0.4)
        else:
            beard = head & (f.ys > hy + 1.6) | f.trapezoid(cx, hy + 2.0, hy + ry + 2.4, 3.6, 1.0)
            beard &= ~f.box(cx - 1.2, hy + 2.1, cx + 1.2, hy + 2.9)  # mouth
        f.dome(beard, hair, cx - 1.0, hy + 2.0, rx, ry)
    _headgear(f, p, cloth, accent, cx, hy, rx, ry, direction)
    if not back:
        ey0, ey1 = hy + 0.2, hy + (2.0 if not chibi else 1.6)
        eyes = (cx - 3.6,) if side else (cx - 2.4, cx + 1.4)
        for ex in eyes:
            f.flat(f.box(ex, ey0, ex + 1.0, ey1) & head, eye)
    return f.finish(cx, b.ground - 0.2, 4.6 if not chibi else 4.0)


def _hair(f: _Figure, p: HumanoidParams, ramp: list[int], cx, hy, rx, ry, direction: str, b: _Body) -> None:
    if p.hair == "bald":
        if direction != "down":  # a fringe of hair around the back of the head
            fringe = f.oval(cx + (1.5 if direction == "left" else 0), hy + 0.8, rx * 0.9, ry * 0.75)
            fringe &= f.oval(cx, hy, rx, ry) & (f.ys > hy + 0.2)
            f.dome(fringe, ramp, cx, hy, rx, ry)
        return
    if p.headgear in ("hood", "helmet") and p.hair in ("short", "spiky"):
        return
    side, back = direction == "left", direction == "up"
    vol_rx, vol_ry, vol_cy = rx + 0.5, ry + 0.5, hy - 0.5
    cap = f.oval(cx, vol_cy, vol_rx, vol_ry)
    rng = f.ctx.rng("bangs")
    jag = rng.integers(0, 2, size=f.c.shape[1]) * 0.8
    jag = np.broadcast_to(jag[None, :], f.c.shape)
    if back:
        mask = cap & (f.ys < hy + ry - 0.6)
    elif side:
        hairline = hy - 1.0 + jag
        mask = cap & ((f.ys < hairline) | (f.xs > cx + 0.6))
        mask &= ~(f.box(cx - rx - 1, hy + 0.4, cx + 0.6, 30) & (f.xs < cx + 0.6))
        mask &= f.ys < hy + ry - 0.8
    else:
        hairline = hy - 1.2 + jag
        mask = cap & ((f.ys < hairline) | ((np.abs(f.xs - cx) > rx - 1.3) & (f.ys < hy + 1.8)))
    if p.hair == "spiky":
        n = 5
        for i in range(n):
            x = cx - rx + 0.8 + i * (2 * rx - 1.6) / (n - 1)
            tip = vol_cy - vol_ry - 1.6 - (0.8 if i in (1, 3) else 0)
            lean = (x - cx) * 0.25
            spike = f.trapezoid(x + lean, tip, vol_cy - vol_ry + 2.5, 0.0, 1.6)
            mask |= spike
    if p.hair == "long":
        if back:
            mask |= f.trapezoid(cx, hy, b.torso_top + 3.0, rx - 0.2, rx + 0.4)
        elif side:
            mask |= f.trapezoid(cx + 2.2, hy, b.torso_top + 2.6, rx - 2.6, rx - 1.6) & (f.xs > cx - 0.2)
        else:
            mask |= (np.abs(f.xs - cx) > rx - 1.6) & f.trapezoid(cx, hy, b.torso_top + 2.6, rx + 0.4, rx + 0.6)
    if p.hair == "ponytail":
        if back:
            mask |= f.trapezoid(cx, hy + 1.0, b.torso_top + 2.8, 1.4, 0.6)
        elif side:
            mask |= f.oval(cx + rx + 0.4, hy + 1.6, 1.4, 2.6)
        else:
            mask |= f.oval(cx + rx + 0.2, hy + 1.8, 0.9, 2.0) & ~f.oval(cx, hy, rx, ry)
    # Hair is a darker, glossier mass than skin: shade it a step down its ramp.
    f.dome(mask, ramp, cx - 0.5, hy - 1.2, vol_rx + 0.5, vol_ry + 1.0, lo=0)
    # Separate the hair from the face with its darkest shade.
    edge = mask & ~shift(mask, -1, 0, False) & shift(f.oval(cx, hy, rx, ry), 1, 0, False)
    if not back:
        f.flat(edge & ~(f.c == 0), ramp[1])


def _headgear(f: _Figure, p: HumanoidParams, cloth, accent, cx, hy, rx, ry, direction: str) -> None:
    g = p.headgear
    side, back = direction == "left", direction == "up"
    top = hy - ry
    if g == "hood":
        hood = f.oval(cx + (0.6 if side else 0), hy - 0.2, rx + 0.9, ry + 0.9)
        if side:
            hood &= ~(f.oval(cx - 1.6, hy + 1.2, rx - 1.6, ry - 1.2) & (f.xs < cx + 0.4))
        elif not back:
            hood &= ~f.oval(cx, hy + 1.3, rx - 1.4, ry - 1.3)
        f.dome(hood, cloth, cx - 0.8, hy - 1.0, rx + 1.2, ry + 1.4)
        # The face sits in the hood's shadow: darken the rim around the opening.
        face = ~hood & f.oval(cx, hy, rx, ry)
        f.flat(hood & neighbors4(face), cloth[1])
    elif g == "wizard_hat":
        brim_y = top + 1.8
        brim = f.oval(cx, brim_y, rx + 1.8, 1.3)
        lean = 1.2 if not side else 2.0
        cone = f.trapezoid(cx + lean, max(top - 4.2, -0.2), brim_y, 0.4, rx - 1.0)
        f.cyl(cone, cloth, 1, 4)
        f.flat(cone & f.box(0, brim_y - 1.6, 16, brim_y - 0.6), accent[3])
        f.cyl(brim, cloth, 0, 3)
    elif g == "helmet":
        dome = f.oval(cx, hy - 0.4, rx + 0.5, ry + 0.4) & (f.ys < hy - (0.4 if not back else -1.6))
        if side:
            dome |= f.oval(cx, hy - 0.4, rx + 0.5, ry + 0.4) & (f.xs > cx + 0.8) & (f.ys < hy + 2.0)
        f.dome(dome, accent, cx - 0.8, hy - 1.6, rx + 0.5, ry + 0.4)
        f.flat(dome & ~shift(dome, -1, 0, False), accent[1])
        if direction == "down":
            f.flat(f.box(cx - 0.5, hy - 1.0, cx + 0.5, hy + 1.2), accent[2])
    elif g == "bandana":
        band = f.oval(cx, hy - 0.4, rx + 0.6, ry + 0.6) & f.box(0, hy - 2.4, 16, hy - 1.0)
        if side or back:
            tx = cx + rx if side else cx + 1.0
            band |= f.trapezoid(tx, hy - 2.0, hy + 1.0, 0.4, 1.1) & f.box(tx - 1.0, 0, tx + 2.0, 30)
        f.cyl(band, accent, 1, 4)
    elif g == "crown":
        y0 = top - 0.4
        crown = f.box(cx - 3.0, y0, cx + 3.0, y0 + 1.6)
        for x in (cx - 2.5, cx, cx + 2.5):
            crown |= f.trapezoid(x, y0 - 1.8, y0, 0.0, 0.9)
        f.cyl(crown, accent, 2, 5)
        if not back:
            f.flat(f.box(cx - 0.5, y0 + 0.4, cx + 0.5, y0 + 1.2), cloth[3])


class SlimeParams(Params):
    shape: Literal["drop", "round"] = Field("drop", description="drop: teardrop with a curled tip; round: a blob")
    smile: bool = Field(True)


@register(
    "slime",
    "character",
    SlimeParams,
    colors=[("body", "#4a90e0"), ("mouth", "#c83a4a")],
    doc="Bouncing slime monster (1x1 tile). Squashes and stretches as it moves.",
)
def slime(ctx: GenContext, p: SlimeParams, direction: str, frame: int) -> np.ndarray:
    f = _Figure(ctx, 16)
    body, mouth = ctx.ramp(0), ctx.ramp(1)
    white, eye = ctx.palette.ramp(WHITE)[3], ctx.palette.ramp(EYE)[1]
    squash = (0.9, 0.0, -0.7)[frame]  # wide and low, resting, tall and narrow
    ground = 14.6
    cx = 8.0
    rx, ry = 5.6 + squash * 0.8, 4.6 - squash
    cy = ground - ry
    blob = f.oval(cx, cy, rx, ry) & (f.ys < ground) | f.box(cx - rx + 1.0, cy, cx + rx - 1.0, ground)
    if p.shape == "drop":
        tip_y = cy - ry - 2.4 + squash * 0.4
        blob |= f.trapezoid(cx + 0.4, tip_y, cy - ry + 2.0, 0.0, 2.4)
        blob |= f.oval(cx + 1.4, tip_y + 0.4, 1.0, 0.8)  # the curl
    f.dome(blob, body, cx - 0.8, cy - 0.6, rx, ry + 1.0)
    f.flat(blob & ~shift(blob, -1, 0, False), body[0])
    shine = f.oval(cx - rx * 0.45, cy - ry * 0.35, 0.9, 1.2)
    f.flat(shine & blob, ctx.palette.ramp(WHITE)[4])
    if direction != "up":
        eyes = (cx - 3.6,) if direction == "left" else (cx - 2.8, cx + 1.2)
        ey = cy - 0.4
        for ex in eyes:
            f.flat(f.box(ex, ey, ex + 1.6, ey + 2.2), white)
            px = ex if direction == "left" else ex + 0.6
            f.flat(f.box(px, ey + 0.8, px + 1.0, ey + 2.2), eye)
        if p.smile:
            mx = cx - 4.2 if direction == "left" else cx - 2.0
            mw = 2.0 if direction == "left" else 4.0
            m = f.box(mx, ey + 2.8, mx + mw, ey + 3.6) | f.box(mx + 0.8, ey + 3.6, mx + mw - 0.8, ey + 4.4)
            f.flat(m & blob, mouth[2])
    return f.finish(cx, ground - 0.2, rx + 0.2)


class GhostParams(Params):
    arms: bool = Field(True, description="little stubby arms")


@register(
    "ghost",
    "character",
    GhostParams,
    colors=[("body", "#dce4f4"), ("eyes", "#3a3058")],
    doc="Floating sheet ghost (1x1 tile). Bobs up and down; its hem ripples.",
)
def ghost(ctx: GenContext, p: GhostParams, direction: str, frame: int) -> np.ndarray:
    f = _Figure(ctx, 16)
    body, eyes = ctx.ramp(0), ctx.ramp(1)
    bob = (0.0, -0.8, 0.0)[frame]
    cx = 8.0
    cy = 6.0 + bob
    rx, ry = 4.6, 4.4
    hem = 12.6 + bob
    sheet = f.oval(cx, cy, rx, ry) | f.box(cx - rx, cy, cx + rx, hem)
    wave = np.sin((f.xs - cx) * 2.1 + frame * 2.1) * 0.9
    sheet |= f.box(cx - rx, hem, cx + rx, hem + 1.2) & (f.ys < hem + 0.4 + wave)
    if direction == "left":
        sheet |= f.trapezoid(cx + 1.0, cy, hem + 1.0, rx - 0.4, rx + 1.6) & (f.xs > cx) & (f.ys < hem + 0.2 + wave)
    f.dome(sheet, body, cx - 1.0, cy, rx + 0.4, ry + 4.0)
    if p.arms and direction != "up":
        xs = (cx - 1.0,) if direction == "left" else (cx - rx - 0.6, cx + rx + 0.6)
        for x in xs:
            arm = f.oval(x, cy + 3.4, 1.2, 0.9)
            f.dome(arm, body, x - 0.3, cy + 3.0, 1.2, 0.9)
    if direction != "up":
        ex = (cx - 3.2,) if direction == "left" else (cx - 2.4, cx + 1.2)
        for x in ex:
            f.flat(f.oval(x + 0.6, cy + 0.4, 1.0, 1.4), eyes[1])
        if direction == "down":
            f.flat(f.oval(cx, cy + 3.0, 1.0, 0.9), eyes[1])
    # Floating: a smaller, detached shadow.
    return f.finish(cx, 15.0, 3.2)
