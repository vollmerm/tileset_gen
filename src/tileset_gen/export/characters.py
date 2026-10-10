"""Character sprite sheets: characters/<name>.png and .tsx, emote balloons, plus characters.json."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from ..build import BuildResult, CharacterSheet
from ..generators.characters import ACTIONS, EMOTE_SEQUENCE, EMOTES, FRAMES, ROWS
from ..palette import rgb_to_hex
from .tiled import TILED_VERSION, _write

EMOTE_HOLD_MS = 1200  # a balloon holds its last frame before Tiled loops it


def animation_name(action: str, direction: str) -> str:
    """walk_down, attack_left... and plain names for actions that only face the camera (cheer)."""
    return action if len(ACTIONS[action].directions) == 1 else f"{action}_{direction}"


def animation_tile(row: int, action: str) -> int:
    """The tile carrying a row's Tiled animation: walk rows animate their standing frame."""
    return row * FRAMES + (1 if action == "walk" else 0)


def write_characters(res: BuildResult, out: Path) -> None:
    if not res.characters:
        return
    folder = out / "characters"
    folder.mkdir(exist_ok=True)
    docs = []
    for ch in res.characters:
        _save_png(ch, folder)
        animations = {}
        for r, (action, d) in enumerate(ROWS):
            spec = ACTIONS[action]
            animations[animation_name(action, d)] = {
                "frames": [[k, r] for k in spec.sequence],
                "ms": round(ch.frame_ms * spec.speed),
                "loop": spec.loop,
            }
        _write_tsx(
            ch,
            folder,
            [(animation_name(a, d), animation_tile(r, a), r, ACTIONS[a]) for r, (a, d) in enumerate(ROWS)],
        )
        w, h = ch.size
        docs.append(
            {
                "name": ch.name,
                "image": f"characters/{ch.name}.png",
                "tileset": f"characters/{ch.name}.tsx",
                "frame_width": w,
                "frame_height": h,
                "head_top": ch.head_top(),
                "palette": _palette(ch),
                "animations": animations,
                "stand": {d: [1, r] for r, (a, d) in enumerate(ROWS) if a == "walk"},
            }
        )
    emotes = res.emotes
    _save_png(emotes, folder)
    rows = [(e, r * FRAMES, r, None) for r, e in enumerate(EMOTES)]
    _write_tsx(emotes, folder, rows, hold=(EMOTE_SEQUENCE, emotes.frame_ms))
    w, h = emotes.size
    doc = {
        "layout": "frame [col, row] in each sheet; rows: " + ", ".join(animation_name(a, d) for a, d in ROWS),
        "anchor": "bottom-center of the frame stands at the bottom-center of a map tile",
        "characters": docs,
        "emotes": {
            "image": "characters/emotes.png",
            "tileset": "characters/emotes.tsx",
            "frame_width": w,
            "frame_height": h,
            "placement": "balloon bottom-center 2px below the character's head_top, centered on the character",
            "palette": _palette(emotes),
            "animations": {
                e: {"frames": [[k, r] for k in EMOTE_SEQUENCE], "ms": emotes.frame_ms, "loop": False}
                for r, e in enumerate(EMOTES)
            },
        },
    }
    (out / "characters.json").write_text(json.dumps(doc, indent=2) + "\n")


def _palette(sheet: CharacterSheet) -> list[str]:
    return [rgb_to_hex(tuple(int(v) for v in row[:3])) for row in sheet.lut[1:]]


def _save_png(sheet: CharacterSheet, folder: Path) -> None:
    Image.fromarray(sheet.to_rgba(sheet.sheet()), "RGBA").save(folder / f"{sheet.name}.png", optimize=True)


def _write_tsx(sheet: CharacterSheet, folder: Path, rows, hold=None) -> None:
    """A Tiled tileset for a sheet; each row's animation sits on one of its tiles.

    `rows` holds (name, animated tile, row, Action). Emote balloons (Action None) play
    once and hold, which Tiled can't express, so their last frame is shown for a while.
    """
    w, h = sheet.size
    n_rows = sheet.frames.shape[0]
    root = ET.Element(
        "tileset",
        {
            "version": TILED_VERSION,
            "name": sheet.name,
            "tilewidth": str(w),
            "tileheight": str(h),
            "tilecount": str(n_rows * FRAMES),
            "columns": str(FRAMES),
            "objectalignment": "bottom",
        },
    )
    ET.SubElement(root, "image", {"source": f"{sheet.name}.png", "width": str(w * FRAMES), "height": str(h * n_rows)})
    for name, tid, r, action in rows:
        tile = ET.SubElement(root, "tile", {"id": str(tid), "type": "character" if action else "emote"})
        props = ET.SubElement(tile, "properties")
        ET.SubElement(props, "property", {"name": "animation", "value": name})
        anim = ET.SubElement(tile, "animation")
        if action is None:
            sequence, ms = hold
            durations = [ms] * (len(sequence) - 1) + [EMOTE_HOLD_MS]
        else:
            sequence = action.sequence
            durations = [round(sheet.frame_ms * action.speed)] * len(sequence)
        for k, ms in zip(sequence, durations):
            ET.SubElement(anim, "frame", {"tileid": str(r * FRAMES + k), "duration": str(ms)})
    _write(root, folder / f"{sheet.name}.tsx")
