"""Character sprites in the style of 8/16-bit JRPG field sprites: walking, attacking and emoting.

Signature: fn(ctx, params, action, direction, frame) -> uint8[h, w]. `ROWS` lists
every (action, direction) row of a sheet; each row has `FRAMES` frames. Walk and
attack rows exist for all four directions. Emotes (cheer, surprise, sad, hurt)
face the camera, the way RPG Maker events act out a scene. Generators are only
asked for "down", "left" and "up": the builder mirrors "left" to get "right", as
the SNES games did.

Every frame of a character has the same size: 2 tiles wide, so a sword has room
to swing (SNES attack frames were wider than walk frames too), with headroom
above the head for hops, hats and raised weapons. Feet stand at the bottom
center, so the frame anchors bottom-center on a map tile. Weapons that would
still leave the frame are turned toward horizontal until they fit.

Sprites follow the tileset's rules so they sit in it naturally: colors come from
the same hue-shifted ramps, light comes from the upper left, shapes get the
selective outline and a translucent `SHADOW` ellipse on the ground.

Shapes are laid out in units of a 16px tile and rasterized at the tile size, so a
32px sprite has the same proportions with finer shading. Each character's look
comes from a `Pose` per frame (arms, hop, lean, face, weapon angle), so walk,
attack and emote frames share one rig.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from pydantic import Field

from ..context import GenContext
from ..draw import cylinder_levels, ellipse_height, ground_shadow, neighbors4, outline, shade, shift
from ..palette import SHADOW
from ..registry import Params, register

DIRECTIONS = ("down", "left", "right", "up")
FRAMES = 3


@dataclass(frozen=True)
class Action:
    directions: tuple[str, ...]
    sequence: tuple[int, ...]  # frame order to play
    loop: bool
    speed: float  # frame duration as a multiple of the character's frame_ms


ACTIONS = {
    "walk": Action(DIRECTIONS, (0, 1, 2, 1), True, 1.0),
    "attack": Action(DIRECTIONS, (0, 1, 2), False, 0.6),
    "cheer": Action(("down",), (0, 2, 1, 2), True, 1.2),
    "surprise": Action(("down",), (0, 1, 2), False, 1.2),
    "sad": Action(("down",), (0, 1, 2, 1), True, 2.0),
    "hurt": Action(("down",), (0, 1, 0, 1, 2), False, 0.6),
}
ROWS = [(a, d) for a, spec in ACTIONS.items() for d in spec.directions]

EYE = "#282038"
WHITE = "#f4f0e8"
TEAR = "#58a8e8"
MOUTH = "#a8344a"
WIDTH = 32  # frame width in units (2 tiles)


@dataclass(frozen=True)
class Pose:
    step: int = 0  # walk phase: which leg is forward (-1, 0, 1); also a held lunge stance
    dy: float = 0.0  # hop (negative is up); the shadow stays on the ground
    dx: float = 0.0  # lean or lunge
    arms: tuple[str, str] = ("swing", "swing")  # screen-left, screen-right (side view: near arm first)
    weapon: float | None = None  # weapon angle in degrees (0 = right, 90 = down), held in the weapon hand
    behind: bool = False  # weapon passes behind the body
    face: str = "normal"  # normal | happy | wide | sad | tear | hurt | angry
    mouth: bool = False  # open mouth
    head_dy: float = 0.0  # head drop (sad)
    flash: bool = False  # white damage flash
    squash: float = 0.0  # monsters: + wide and low, - tall and narrow
    stretch: float = 0.0  # monsters: extra width when lunging


class _Figure:
    """A canvas addressed in 16px-tile units: masks from shapes, plus shaded fills."""

    def __init__(self, ctx: GenContext, height_units: float, width_units: float = WIDTH):
        self.ctx = ctx
        self.s = s = ctx.T / 16
        H, W = int(round(height_units * s)), int(round(width_units * s))
        self.c = np.zeros((H, W), dtype=np.uint8)
        ys, xs = np.mgrid[0:H, 0:W]
        self.ys, self.xs = (ys + 0.5) / s, (xs + 0.5) / s
        self.cx = width_units / 2

    def box(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        return (self.xs >= x0) & (self.xs < x1) & (self.ys >= y0) & (self.ys < y1)

    def oval(self, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        return ((self.xs - cx) / rx) ** 2 + ((self.ys - cy) / ry) ** 2 <= 1.0

    def trapezoid(self, cx: float, y0: float, y1: float, hw0: float, hw1: float) -> np.ndarray:
        """Rows y0..y1, half-width growing linearly from hw0 to hw1 (a cone when hw0 = 0)."""
        t = np.clip((self.ys - y0) / max(y1 - y0, 1e-6), 0, 1)
        return (self.ys >= y0) & (self.ys < y1) & (np.abs(self.xs - cx) <= hw0 + (hw1 - hw0) * t)

    def along(self, x: float, y: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates along (t) and across (n) a ray from (x, y) at `angle` degrees."""
        a = np.radians(angle)
        ux, uy = np.cos(a), np.sin(a)
        dx, dy = self.xs - x, self.ys - y
        return dx * ux + dy * uy, -dx * uy + dy * ux

    def segment(self, x, y, angle, t0, t1, hw0, hw1=None) -> np.ndarray:
        """A bar along the ray from t0 to t1, its half-width tapering from hw0 to hw1."""
        t, n = self.along(x, y, angle)
        hw1 = hw0 if hw1 is None else hw1
        half = hw0 + (hw1 - hw0) * np.clip((t - t0) / max(t1 - t0, 1e-6), 0, 1)
        return (t >= t0) & (t <= t1) & (np.abs(n) <= half)

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

    def finish(self, pose: Pose, cx: float, ground: float, rx: float) -> np.ndarray:
        """Move the figure by the pose's hop and lean, outline it, flash it, and cast its shadow."""
        s = self.s
        c = shift(self.c, int(round(pose.dy * s)), int(round(pose.dx * s)))
        c = outline(c, self.ctx.palette)
        if pose.flash:
            white = self.ctx.palette.ramp(WHITE)
            solid = c > 0
            c = np.where(solid, white[4], 0).astype(np.uint8)
            c[solid & neighbors4(~solid)] = white[2]
        # Hopping shrinks the shadow a little; lunging carries it along.
        rx = rx * (1.0 + min(pose.dy, 0.0) * 0.08)
        return ground_shadow(c, (cx + pose.dx + 0.6) * s, ground * s, rx * s, max(1.2, 1.1 * s))


def _eye_marks(f: _Figure, kind: str, centers, y0: float, h: float, color: int, mask=None) -> None:
    """Simple dark eyes: open slits, happy/closed bars, small sad eyes and squeezed > < eyes."""
    for i, ecx in enumerate(centers):
        if kind in ("happy",):
            m = f.box(ecx - 0.9, y0 + h * 0.35, ecx + 0.9, y0 + h * 0.35 + 0.8)
        elif kind == "hurt":
            # ">" for the left eye, "<" for the right (and for a lone side-view eye).
            point = 1 if i == 0 and len(centers) > 1 else -1
            m = f.box(ecx - 0.5 - 0.5 * point, y0, ecx + 0.5 - 0.5 * point, y0 + 0.9)
            m |= f.box(ecx - 0.5 + 0.5 * point, y0 + 0.9, ecx + 0.5 + 0.5 * point, y0 + 1.8)
            m |= f.box(ecx - 0.5 - 0.5 * point, y0 + 1.8, ecx + 0.5 - 0.5 * point, y0 + 2.7)
        elif kind in ("sad", "tear"):
            m = f.box(ecx - 0.5, y0 + h * 0.45, ecx + 0.5, y0 + h)
        elif kind == "wide":
            m = f.box(ecx - 0.5, y0 - 0.7, ecx + 0.5, y0 + h + 0.2)
        else:
            m = f.box(ecx - 0.5, y0, ecx + 0.5, y0 + h)
        f.flat(m if mask is None else m & mask, color)
        if kind == "wide":
            f.flat(f.box(ecx - 0.5, y0 - 0.7, ecx + 0.5, y0 + 0.2), f.ctx.palette.ramp(WHITE)[4])
        if kind == "tear":
            f.flat(f.box(ecx - 0.5, y0 + h + 0.2, ecx + 0.5, y0 + h + 1.6), f.ctx.palette.ramp(TEAR)[3])


# --------------------------------------------------------------------------------------------
# Humanoids


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
    blade: float  # sword length


BODIES = {
    # FF6/Chrono Trigger field sprites: 16x24 of figure, a big head on a short body, in a 2-tile-tall frame.
    "classic": _Body(32, 30.6, 19.4, 25.4, 3.6, 14.8, 5.3, 4.9, 8.0),
    # Dragon Quest/early FF overworld sprites: 16x16 of figure, mostly head, in a 1.5-tile-tall frame.
    "chibi": _Body(24, 23.2, 17.6, 20.8, 3.2, 13.4, 5.2, 4.4, 6.0),
}


class HumanoidParams(Params):
    body: Literal["classic", "chibi"] = Field(
        "classic", description="classic: 16-bit proportions (FF6); chibi: 8-bit, mostly head (Dragon Quest)"
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
    weapon: Literal["sword", "axe", "spear", "staff", "none"] = Field(
        "sword", description="swung in the attack animation (none: a punch); staff has an orb in the accent color"
    )


def _humanoid_pose(action: str, direction: str, frame: int) -> Pose:
    side, back = direction == "left", direction == "up"
    if action == "walk":
        return Pose(step=(-1, 0, 1)[frame])
    if action == "attack":  # wind-up, strike, follow-through
        if side:
            poses = (
                Pose(arms=("up", "swing"), weapon=-60, behind=True, dx=0.6),
                Pose(arms=("out", "swing"), weapon=-172, dx=-1.0, step=-1),
                Pose(arms=("fwd", "swing"), weapon=130, dx=-1.0, step=-1),
            )
        elif back:
            poses = (
                Pose(arms=("swing", "up"), weapon=-80),
                Pose(arms=("swing", "up"), weapon=-140, behind=True, dy=0.4),
                Pose(arms=("swing", "fwd"), weapon=110, behind=True, dy=0.4),
            )
        else:
            poses = (
                Pose(arms=("swing", "up"), weapon=-60, behind=True),
                Pose(arms=("swing", "fwd"), weapon=50, dy=0.4),
                Pose(arms=("swing", "fwd"), weapon=135, dy=0.4),
            )
        return poses[frame]
    up, out = ("up", "up"), ("out", "out")
    if action == "cheer":
        return (
            Pose(arms=up, face="happy", mouth=True),
            Pose(arms=up, face="happy", mouth=True, dy=-2.4),
            Pose(arms=out, face="happy", mouth=True, dy=-1.0),
        )[frame]
    if action == "surprise":
        return (Pose(arms=out, face="wide", mouth=True, dy=-2.0), Pose(face="wide", mouth=True), Pose(face="wide"))[
            frame
        ]
    if action == "sad":
        return (
            Pose(face="sad", head_dy=0.8),
            Pose(face="sad", head_dy=1.0, dy=0.3),
            Pose(face="tear", head_dy=0.8),
        )[frame]
    if action == "hurt":
        hit = Pose(arms=out, face="hurt", mouth=True, dy=-0.8)
        return (hit, replace(hit, flash=True), Pose(face="hurt", head_dy=0.4))[frame]
    raise ValueError(f"unknown action {action!r}")


@dataclass
class _Arm:
    sleeve: np.ndarray
    hand: np.ndarray
    hx: float
    hy: float
    lit: bool


def _arms(f: _Figure, b: _Body, pose: Pose, side: bool, back: bool, hw: float, cx: float) -> list[_Arm]:
    """Arm shapes for the pose; the last arm holds the weapon."""
    arm_w, hand = 1.9, 1.3
    top, bottom = b.torso_top + 0.5, b.waist + 0.3
    step = pose.step
    if side:
        kind = pose.arms[0]
        if kind == "up":
            x0, y0 = cx + 0.2, top - 5.2
            return [
                _Arm(
                    f.box(x0, y0 + hand, x0 + arm_w, top + 1.5),
                    f.box(x0, y0, x0 + arm_w, y0 + hand),
                    x0 + 1,
                    y0 + 0.6,
                    True,
                )
            ]
        if kind == "out":
            y0, xa = top + 0.2, cx - 5.4
            sleeve = f.box(xa + hand, y0, cx + 1.0, y0 + arm_w)
            return [_Arm(sleeve, f.box(xa, y0 - 0.1, xa + hand, y0 + arm_w + 0.1), xa + 0.6, y0 + arm_w / 2, True)]
        if kind == "fwd":
            sleeve = f.box(cx - 1.0, top, cx + 1.0, top + 2.0) | f.box(
                cx - 3.4 + hand, top + 1.6, cx + 0.2, top + 1.6 + arm_w
            )
            fist = f.box(cx - 3.4, top + 1.4, cx - 3.4 + hand, top + 1.8 + arm_w)
            return [_Arm(sleeve, fist, cx - 2.8, top + 2.5, True)]
        swing = -1.6 * step
        x0 = cx - 0.6 + swing
        end = bottom - abs(step) * 0.4
        sleeve = f.box(x0, top, x0 + arm_w, end - hand)
        return [_Arm(sleeve, f.box(x0, end - hand, x0 + arm_w, end), x0 + arm_w / 2, end - 0.6, True)]
    arms = []
    for i, kind in enumerate(pose.arms):
        outward = -1 if i == 0 else 1
        x0 = cx - hw - arm_w + 0.2 if i == 0 else cx + hw - 0.2
        lit = i == 0
        if kind == "up":
            x = x0 + outward * 0.6
            y0 = top - 5.2
            arms.append(
                _Arm(
                    f.box(x, y0 + hand, x + arm_w, top + 1.2),
                    f.box(x, y0, x + arm_w, y0 + hand),
                    x + arm_w / 2,
                    y0 + 0.6,
                    lit,
                )
            )
        elif kind == "out":
            y0 = top - 0.4
            xa, xb = (cx - hw - 4.0, cx - hw + 0.4) if i == 0 else (cx + hw - 0.4, cx + hw + 4.0)
            hx0, hx1 = (xa, xa + hand) if i == 0 else (xb - hand, xb)
            sleeve = f.box(xa, y0, xb, y0 + arm_w) & ~f.box(hx0, y0, hx1, y0 + arm_w)
            arms.append(_Arm(sleeve, f.box(hx0, y0 - 0.1, hx1, y0 + arm_w + 0.1), (hx0 + hx1) / 2, y0 + arm_w / 2, lit))
        elif kind == "fwd":
            x = x0 - outward * 1.3
            end = top + (b.waist - b.torso_top) * 0.6
            sleeve = f.box(x, top, x + arm_w, end - hand)
            fist = f.box(x - 0.1, end - hand, x + arm_w + 0.1, end + 0.3)
            arms.append(_Arm(sleeve, fist, x + arm_w / 2, end - 0.4, lit))
        else:
            fwd = (step == (1 if i == 0 else -1)) if not back else (step == (-1 if i == 0 else 1))
            end = bottom + (0.8 if fwd else 0.0)
            sleeve = f.box(x0, top, x0 + arm_w, end - hand)
            arms.append(_Arm(sleeve, f.box(x0, end - hand, x0 + arm_w, end), x0 + arm_w / 2, end - 0.6, lit))
    return arms


# How far each weapon reaches from the grip, (forward, backward), in units of a classic body.
REACH = {"sword": (9.2, 1.4), "axe": (7.4, 1.2), "spear": (10.4, 3.0), "staff": (8.4, 4.0)}


def _fit_angle(f: _Figure, kind: str, hx: float, hy: float, angle: float, k: float) -> float:
    """Turn the weapon a few degrees at a time, toward horizontal first, until both ends stay in the frame.

    `hx, hy` already include the pose's hop and lean, which move the drawing after the fact.
    """
    H, W = f.c.shape[0] / f.s, f.c.shape[1] / f.s
    front, back = (r * k for r in REACH[kind])

    def fits(a: float) -> bool:
        ux, uy = np.cos(np.radians(a)), np.sin(np.radians(a))
        ends = ((hx + ux * front, hy + uy * front), (hx - ux * back, hy - uy * back))
        return all(1.6 <= x <= W - 1.6 and 1.6 <= y <= H - 1.6 for x, y in ends)  # room for width + outline

    toward = 1 if np.sin(np.radians(angle)) * np.cos(np.radians(angle)) < 0 else -1  # rotation that levels it
    for delta in range(0, 181, 5):
        for a in (angle + toward * delta, angle - toward * delta):
            if fits(a):
                return a
    return angle


def _weapon(f: _Figure, kind: str, hx: float, hy: float, angle: float, k: float, steel, accent, wood) -> None:
    """Draw a weapon gripped at (hx, hy), pointing at `angle`, scaled by `k` (smaller for chibi bodies)."""
    t, n = f.along(hx, hy, angle)
    t = t / k
    seg = lambda t0, t1, hw0, hw1=None, a=angle: f.segment(hx, hy, a, t0 * k, t1 * k, hw0, hw1)  # noqa: E731
    if kind == "sword":
        f.flat(seg(-1.4, 0.6, 0.5), wood[1])
        f.flat(seg(-1.6, 1.6, 0.5, a=angle + 90) & (np.abs(t - 0.9) <= 0.5 / k), accent[2])
        blade = seg(1.2, 9.2, 1.0, 0.45)
        f.flat(blade, steel[2])
        f.flat(blade & (n < 0), steel[4])  # the edge facing the light
        f.flat(blade & (t > 8.4), steel[4])
    elif kind == "axe":
        f.flat(seg(-1.2, 6.6, 0.5), wood[2])
        head = (t >= 4.4) & (t <= 7.2) & (n >= -0.4) & (n <= (2.8 - np.abs(t - 5.8) * 0.4) * k)
        f.flat(head, steel[3])
        f.flat(head & (n > 1.8 * k), steel[4])
    elif kind == "spear":
        f.flat(seg(-3.0, 7.6, 0.45), wood[2])
        f.flat(seg(7.4, 10.4, 1.1, 0.1), steel[3])
        f.flat(seg(7.4, 8.0, 1.2), accent[2])
    elif kind == "staff":
        f.flat(seg(-4.0, 6.0, 0.5), wood[2])
        a = np.radians(angle)
        ox, oy = hx + np.cos(a) * 7.0 * k, hy + np.sin(a) * 7.0 * k
        f.dome(f.oval(ox, oy, 1.4, 1.4), accent, ox - 0.4, oy - 0.4, 1.4, 1.4)


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
        ("weapon", "#b8c0d0"),
    ],
    doc="Person: hero, party member, townsfolk, guard, mage, king... Mix outfit, hair, headgear and weapon.",
    key_colors=(0, 2),
)
def humanoid(ctx: GenContext, p: HumanoidParams, action: str, direction: str, frame: int) -> np.ndarray:
    b = BODIES[p.body]
    pose = _humanoid_pose(action, direction, frame)
    f = _Figure(ctx, b.height)
    skin, hair, cloth, accent, legs, steel = (ctx.ramp(i) for i in range(6))
    step = pose.step
    cx = f.cx
    side = direction == "left"
    back = direction == "up"
    hw = b.hw - (0.9 if side else 0.0)
    robe = p.outfit == "robe"
    hem = {"robe": b.ground - 0.7, "dress": (b.waist + b.ground) / 2 + 0.6}.get(p.outfit)
    leg_len = b.ground - b.waist
    chibi = p.body == "chibi"
    arms = _arms(f, b, pose, side, back, hw, cx)
    armed = pose.weapon is not None and p.weapon != "none"

    def weapon() -> None:
        hand = arms[-1]
        k = b.blade / BODIES["classic"].blade
        angle = _fit_angle(f, p.weapon, hand.hx + pose.dx, hand.hy + pose.dy, pose.weapon, k)
        _weapon(f, p.weapon, hand.hx, hand.hy, angle, k, steel, accent, legs)

    if armed and pose.behind:
        weapon()

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
        f.flat(torso & f.box(0, b.waist - 1.0, 99, b.waist), accent[2])
        if not back and not side:
            f.flat(torso & f.box(cx - 0.6, b.waist - 1.0, cx + 0.6, b.waist), accent[4])
    if not back and not side and p.outfit != "armor":
        neckline = f.trapezoid(cx, b.torso_top, b.torso_top + 1.2, 1.2, 0.2)
        f.flat(neckline & torso, accent[3] if robe else cloth[1])

    # Arms: both in front and back view, only the near one from the side.
    for arm in arms:
        f.cyl(arm.sleeve, cloth, 1 if arm.lit else 0, 4 if arm.lit else 3)
        f.dome(arm.hand, skin, arm.hx - 0.3, arm.hy - 0.3, 1.2, 1.0)
        if robe:
            f.flat(arm.sleeve & neighbors4(arm.hand), accent[2])  # wide cuffs
        if side and pose.arms[0] == "swing":
            f.flat(arm.sleeve & ~shift(arm.sleeve, 0, 1, False), cloth[0] if not robe else cloth[1])
    if p.outfit == "armor":
        for x in (cx - hw - 0.4, cx + hw + 0.4) if not side else (cx + 0.3 - 1.6 * step,):
            pad = f.oval(x, b.torso_top + 0.9, 1.6, 1.3)
            f.dome(pad, accent, x, b.torso_top + 0.9, 1.6, 1.3)

    # Head.
    hy, rx, ry = b.head_cy + pose.head_dy, b.head_rx, b.head_ry
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
        eye_h = 1.8 if not chibi else 1.4
        centers = (cx - 3.1,) if side else (cx - 1.9, cx + 1.9)
        _eye_marks(f, pose.face, centers, hy + 0.2, eye_h, ctx.palette.ramp(EYE)[1], head)
        if pose.mouth:
            mx = cx - 3.9 if side else cx
            f.flat(f.box(mx - 0.7, hy + 2.5, mx + 0.7, hy + 3.7) & head, ctx.palette.ramp(MOUTH)[1])
    if armed and not pose.behind:
        weapon()
    return f.finish(pose, cx, b.ground - 0.2, 4.6 if not chibi else 4.0)


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
        mask &= ~(f.box(cx - rx - 1, hy + 0.4, cx + 0.6, 99) & (f.xs < cx + 0.6))
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
        cone = f.trapezoid(cx + lean, top - 4.2, brim_y, 0.4, rx - 1.0)
        f.cyl(cone, cloth, 1, 4)
        f.flat(cone & f.box(0, brim_y - 1.6, 99, brim_y - 0.6), accent[3])
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
        band = f.oval(cx, hy - 0.4, rx + 0.6, ry + 0.6) & f.box(0, hy - 2.4, 99, hy - 1.0)
        if side or back:
            tx = cx + rx if side else cx + 1.0
            band |= f.trapezoid(tx, hy - 2.0, hy + 1.0, 0.4, 1.1) & f.box(tx - 1.0, 0, tx + 2.0, 99)
        f.cyl(band, accent, 1, 4)
    elif g == "crown":
        y0 = top - 0.4
        crown = f.box(cx - 3.0, y0, cx + 3.0, y0 + 1.6)
        for x in (cx - 2.5, cx, cx + 2.5):
            crown |= f.trapezoid(x, y0 - 1.8, y0, 0.0, 0.9)
        f.cyl(crown, accent, 2, 5)
        if not back:
            f.flat(f.box(cx - 0.5, y0 + 0.4, cx + 0.5, y0 + 1.2), cloth[3])


# --------------------------------------------------------------------------------------------
# Monsters


def _monster_pose(action: str, direction: str, frame: int, walk: tuple[Pose, Pose, Pose]) -> Pose:
    """Shared acting for blob-like monsters: they lunge to attack and hop, stretch or droop to emote."""
    side = direction == "left"
    if action == "walk":
        return walk[frame]
    if action == "attack":
        if side:
            return (
                Pose(squash=0.9, dx=1.0, dy=-0.4),
                Pose(squash=-0.2, stretch=1.6, dx=-3.0, mouth=True, face="angry"),
                Pose(squash=0.5, dx=-1.2),
            )[frame]
        return (Pose(squash=1.0), Pose(squash=-0.8, dy=-2.4, mouth=True, face="angry"), Pose(squash=0.6))[frame]
    if action == "cheer":
        return (
            Pose(squash=0.6, face="happy", mouth=True),
            Pose(squash=-0.6, dy=-2.4, face="happy", mouth=True),
            Pose(squash=0.2, dy=-0.8, face="happy", mouth=True),
        )[frame]
    if action == "surprise":
        return (
            Pose(squash=-1.0, dy=-1.6, face="wide", mouth=True),
            Pose(squash=-0.4, face="wide", mouth=True),
            Pose(face="wide"),
        )[frame]
    if action == "sad":
        return (Pose(squash=1.2, face="sad"), Pose(squash=1.4, face="sad"), Pose(squash=1.2, face="tear"))[frame]
    if action == "hurt":
        hit = Pose(squash=0.8, dx=1.0, face="hurt", mouth=True)
        return (hit, replace(hit, flash=True), Pose(squash=0.3, face="hurt"))[frame]
    raise ValueError(f"unknown action {action!r}")


class SlimeParams(Params):
    shape: Literal["drop", "round"] = Field("drop", description="drop: teardrop with a curled tip; round: a blob")
    smile: bool = Field(True)


@register(
    "slime",
    "character",
    SlimeParams,
    colors=[("body", "#4a90e0"), ("mouth", "#c83a4a")],
    doc="Bouncing slime monster (1 tile of figure). Squashes and stretches as it moves; lunges to attack.",
    key_colors=(0,),
)
def slime(ctx: GenContext, p: SlimeParams, action: str, direction: str, frame: int) -> np.ndarray:
    walk = (Pose(squash=0.9), Pose(), Pose(squash=-0.7))  # wide and low, resting, tall and narrow
    pose = _monster_pose(action, direction, frame, walk)
    f = _Figure(ctx, 24)
    body, mouth = ctx.ramp(0), ctx.ramp(1)
    whites, eye = ctx.palette.ramp(WHITE), ctx.palette.ramp(EYE)[1]
    side = direction == "left"
    squash = pose.squash
    ground = 22.6
    cx = f.cx
    rx, ry = 5.6 + squash * 0.8 + pose.stretch, 4.6 - squash
    cy = ground - ry
    blob = f.oval(cx, cy, rx, ry) & (f.ys < ground) | f.box(cx - rx + 1.0, cy, cx + rx - 1.0, ground)
    if p.shape == "drop":
        tip_y = cy - ry - 2.4 + squash * 0.4
        lean = 1.2 if side and pose.stretch else 0.0  # the tip trails behind a lunge
        blob |= f.trapezoid(cx + 0.4 + lean, tip_y, cy - ry + 2.0, 0.0, 2.4)
        blob |= f.oval(cx + 1.4 + lean, tip_y + 0.4, 1.0, 0.8)  # the curl
    f.dome(blob, body, cx - 0.8, cy - 0.6, rx, ry + 1.0)
    f.flat(blob & ~shift(blob, -1, 0, False), body[0])
    shine = f.oval(cx - rx * 0.45, cy - ry * 0.35, 0.9, 1.2)
    f.flat(shine & blob, whites[4])
    if direction != "up":
        lefts = (cx - 3.6 - pose.stretch,) if side else (cx - 2.8, cx + 1.2)
        ey = cy - 0.4
        if pose.face in ("normal", "wide", "angry", "sad", "tear"):
            for ex in lefts:
                top = ey - (0.6 if pose.face == "wide" else 0.0) + (1.0 if pose.face in ("sad", "tear") else 0.0)
                f.flat(f.box(ex, top, ex + 1.6, ey + 2.2), whites[3])
                px = ex if side else ex + 0.6
                pupil = (ey + 0.6, ey + 1.6) if pose.face == "wide" else (ey + 0.8, ey + 2.2)
                f.flat(f.box(px, pupil[0], px + 1.0, pupil[1]), eye)
                if pose.face == "angry":
                    f.flat(f.box(ex, ey, ex + 1.6, ey + 0.8), body[0])
                if pose.face == "tear":
                    f.flat(f.box(ex + 0.4, ey + 2.4, ex + 1.4, ey + 3.8), ctx.palette.ramp(TEAR)[3])
        else:
            _eye_marks(f, pose.face, [x + 0.8 for x in lefts], ey, 2.2, eye, blob)
        mx = cx - 4.2 - pose.stretch if side else cx - 2.0
        mw = 2.0 if side else 4.0
        if pose.mouth:
            f.flat(f.box(mx + 0.4, ey + 2.8, mx + mw - 0.4, ey + 4.6) & blob, mouth[1])
        elif p.smile and pose.face not in ("sad", "tear", "hurt"):
            m = f.box(mx, ey + 2.8, mx + mw, ey + 3.6) | f.box(mx + 0.8, ey + 3.6, mx + mw - 0.8, ey + 4.4)
            f.flat(m & blob, mouth[2])
    return f.finish(pose, cx, ground - 0.2, rx + 0.2)


class GhostParams(Params):
    arms: bool = Field(True, description="little stubby arms")


@register(
    "ghost",
    "character",
    GhostParams,
    colors=[("body", "#dce4f4"), ("eyes", "#3a3058")],
    doc="Floating sheet ghost (1 tile of figure). Bobs up and down; its hem ripples; swoops to attack.",
    key_colors=(0,),
)
def ghost(ctx: GenContext, p: GhostParams, action: str, direction: str, frame: int) -> np.ndarray:
    walk = (Pose(), Pose(dy=-0.8), Pose())
    pose = _monster_pose(action, direction, frame, walk)
    f = _Figure(ctx, 24)
    body, eyes = ctx.ramp(0), ctx.ramp(1)
    side = direction == "left"
    cx = f.cx
    # A ghost has no squash; it rears up (arms raised) instead, and sinks when sad.
    raised = pose.squash > 0.5 and action in ("attack", "cheer") or pose.face in ("wide", "hurt")
    sink = 0.6 if pose.face in ("sad", "tear") else 0.0
    cy = 14.0 + sink
    rx, ry = 4.6 + pose.stretch * 0.5, 4.4
    hem = 20.6 + sink
    sheet = f.oval(cx, cy, rx, ry) | f.box(cx - rx, cy, cx + rx, hem)
    wave = np.sin((f.xs - cx) * 2.1 + frame * 2.1) * 0.9
    sheet |= f.box(cx - rx, hem, cx + rx, hem + 1.2) & (f.ys < hem + 0.4 + wave)
    if side:
        trail = 1.6 + pose.stretch * 1.5
        sheet |= f.trapezoid(cx + 1.0, cy, hem + 1.0, rx - 0.4, rx + trail) & (f.xs > cx) & (f.ys < hem + 0.2 + wave)
    f.dome(sheet, body, cx - 1.0, cy, rx + 0.4, ry + 4.0)
    if p.arms and direction != "up":
        ay = cy - 1.2 if raised else cy + 3.4
        xs = (cx - 1.0 - pose.stretch * 2,) if side else (cx - rx - 0.6, cx + rx + 0.6)
        for x in xs:
            arm = f.oval(x, ay, 1.2, 0.9)
            f.dome(arm, body, x - 0.3, ay - 0.4, 1.2, 0.9)
    if direction != "up":
        ex = (cx - 2.6,) if side else (cx - 1.8, cx + 1.8)
        if pose.face in ("normal", "wide", "angry"):
            r = 1.4 if pose.face == "wide" else 1.0
            for x in ex:
                f.flat(f.oval(x, cy + 0.4, r, 1.4 * r), eyes[1])
                if pose.face == "angry":
                    f.flat(f.box(x - 1.2, cy - 1.2, x + 1.2, cy - 0.2), body[2])
        else:
            _eye_marks(f, pose.face, ex, cy - 0.6, 2.2, eyes[1], sheet)
        if direction == "down" or pose.mouth:
            mx = cx - 2.4 if side else cx
            r = 1.4 if pose.mouth else 0.9
            f.flat(f.oval(mx, cy + 3.0, r, r), eyes[1])
    # Floating: a smaller, detached shadow.
    return f.finish(pose, cx, 23.0, 3.2 + pose.stretch)


# --------------------------------------------------------------------------------------------
# Emote balloons

EMOTES = ("exclaim", "question", "heart", "music", "anger", "sweat", "sleep", "silence")
EMOTE_SEQUENCE = (0, 1, 2)  # pop in, overshoot, settle (then hold the last frame)

_ICONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "exclaim": ("#e04040", ("  ##  ", "  ##  ", "  ##  ", "  ##  ", "      ", "  ##  ")),
    "question": ("#3a8ad8", (" #### ", "##  ##", "   ## ", "  ##  ", "      ", "  ##  ")),
    "heart": ("#e84a7a", (" ## ## ", "#######", "#######", " ##### ", "  ###  ", "   #   ")),
    "music": ("#4a6ae0", ("  ##  ", "  # # ", "  #  #", "  #   ", "###   ", "###   ")),
    "anger": ("#e04030", ("##  ##", "#    #", "      ", "      ", "#    #", "##  ##")),
    "sweat": ("#58a8e8", ("   #   ", "  ###  ", " ##### ", " ##### ", "  ###  ")),
    "sleep": ("#7a6ad8", ("####   ", "  #    ", " #  ###", "####  #", "     # ", "    ###")),
    "silence": ("#585060", ("       ", "       ", "# # # #", "       ")),
}


def emote_balloon(ctx: GenContext, emote: str, frame: int) -> np.ndarray:
    """A speech balloon with an icon, one tile square, to pop up above a character's head."""
    f = _Figure(ctx, 16, 16)
    white = ctx.palette.ramp(WHITE)
    scale = (0.6, 1.0, 1.0)[frame]
    cx, cy = 8.0, 6.8 + (1.0 - scale) * 4.0
    rx, ry = 6.6 * scale, 5.4 * scale
    bubble = f.oval(cx, cy, rx, ry) | f.trapezoid(cx - 2.0, cy + ry - 1.4, cy + ry + 2.2 * scale, 1.4 * scale, 0.0)
    if ctx.palette.style.retro:
        f.flat(bubble, white[3])  # 8-bit balloons are flat white, leaving the sprite palette to the icon
    else:
        f.dome(bubble, white, cx - 1.5, cy - 1.5, rx + 1.0, ry + 1.0)
    if frame > 0:
        hexv, rows = _ICONS[emote]
        ramp = ctx.palette.ramp(hexv)
        bounce = -0.8 if frame == 1 else 0.0
        h, w = len(rows), max(len(r) for r in rows)
        x0, y0 = cx - w / 2, cy - h / 2 + bounce
        icon = np.zeros_like(bubble)
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch == "#":
                    icon |= f.box(x0 + i, y0 + j, x0 + i + 1, y0 + j + 1)
        f.flat(icon, ramp[2])
        f.flat(icon & ~shift(icon, 1, 0, False), ramp[3])  # lit tops
    c = outline(f.c, ctx.palette)
    c[c == SHADOW] = 0
    return c
