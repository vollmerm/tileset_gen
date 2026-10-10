"""Character sprite sheets: characters/<name>.png and .tsx, plus characters.json."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from ..build import BuildResult
from ..generators.characters import DIRECTIONS, FRAMES, WALK_CYCLE
from ..palette import rgb_to_hex
from .tiled import TILED_VERSION, _write


def write_characters(res: BuildResult, out: Path) -> None:
    if not res.spec.characters:
        return
    folder = out / "characters"
    folder.mkdir(exist_ok=True)
    docs = []
    for ch in res.characters:
        w, h = ch.size
        Image.fromarray(ch.to_rgba(ch.sheet()), "RGBA").save(folder / f"{ch.name}.png", optimize=True)
        _write_character_tsx(ch, folder)
        docs.append(
            {
                "name": ch.name,
                "image": f"characters/{ch.name}.png",
                "frame_width": w,
                "frame_height": h,
                "palette": [rgb_to_hex(tuple(int(v) for v in row[:3])) for row in ch.lut[1:]],
                "frame_ms": ch.frame_ms,
                # Frame (col, row) in the sheet for each direction's walk cycle; the middle frame is standing.
                "walk": {d: [[k, r] for k in WALK_CYCLE] for r, d in enumerate(DIRECTIONS)},
                "stand": {d: [1, r] for r, d in enumerate(DIRECTIONS)},
            }
        )
    doc = {
        "layout": f"rows are directions ({', '.join(DIRECTIONS)}), columns are {FRAMES} walk frames",
        "anchor": "bottom-center of the frame stands at the bottom-center of a map tile",
        "characters": docs,
    }
    (out / "characters.json").write_text(json.dumps(doc, indent=2) + "\n")


def _write_character_tsx(ch, folder: Path) -> None:
    """A Tiled tileset per character; each direction's standing frame plays its walk cycle."""
    w, h = ch.size
    root = ET.Element(
        "tileset",
        {
            "version": TILED_VERSION,
            "name": ch.name,
            "tilewidth": str(w),
            "tileheight": str(h),
            "tilecount": str(len(DIRECTIONS) * FRAMES),
            "columns": str(FRAMES),
            "objectalignment": "bottom",
        },
    )
    ET.SubElement(
        root, "image", {"source": f"{ch.name}.png", "width": str(w * FRAMES), "height": str(h * len(DIRECTIONS))}
    )
    for r, d in enumerate(DIRECTIONS):
        tile = ET.SubElement(root, "tile", {"id": str(r * FRAMES + 1), "type": "character"})
        props = ET.SubElement(tile, "properties")
        ET.SubElement(props, "property", {"name": "direction", "value": d})
        anim = ET.SubElement(tile, "animation")
        for k in WALK_CYCLE:
            ET.SubElement(anim, "frame", {"tileid": str(r * FRAMES + k), "duration": str(ch.frame_ms)})
    _write(root, folder / f"{ch.name}.tsx")
