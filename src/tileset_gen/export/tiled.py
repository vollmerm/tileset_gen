"""Tiled map editor exports: tileset.tsx (with corner Wang sets) and preview.tmx."""

from __future__ import annotations

import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from ..build import BuildResult

if TYPE_CHECKING:
    from ..preview import PreviewMap

TILED_VERSION = "1.10"


def _wang_color(i: int) -> str:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.8, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _wangid(nw: int, ne: int, se: int, sw: int) -> str:
    # Tiled order: top, top-right, right, bottom-right, bottom, bottom-left, left, top-left.
    # Edge slots are 0 in a corner set; colors are 1-based.
    return f"0,{ne},0,{se},0,{sw},0,{nw}"


def _write_wangset(res: BuildResult, root: ET.Element) -> None:
    """One corner Wang set over every terrain, so Tiled offers a single terrain brush palette.

    Each terrain is a Wang color; each transition contributes its tiles with
    corners mapped to the two terrains' colors. Pairs without a transition simply
    have no tiles, and Tiled's brush routes through the pairs that do exist.
    """
    terrains = [t.name for t in res.spec.terrains]
    color = {name: i + 1 for i, name in enumerate(terrains)}
    sets = ET.SubElement(root, "wangsets")
    first = res.terrain_tiles[terrains[0]][0]
    el = ET.SubElement(sets, "wangset", {"name": "terrain", "type": "corner", "tile": str(first)})
    for i, name in enumerate(terrains):
        ET.SubElement(
            el,
            "wangcolor",
            {"name": name, "color": _wang_color(i), "tile": str(res.terrain_tiles[name][0]), "probability": "1"},
        )
    wangids: dict[int, str] = {}
    for name in terrains:
        for tid in res.terrain_tiles[name]:
            c = color[name]
            wangids[tid] = _wangid(c, c, c, c)
    for ws in res.wangsets:
        lo, up = color[ws.lower], color[ws.upper]
        for corners, ids in ws.tiles.items():
            if 0 < sum(corners) < 4:
                for tid in ids:
                    wangids[tid] = _wangid(*(up if c else lo for c in corners))
    for tid in sorted(wangids):
        ET.SubElement(el, "wangtile", {"tileid": str(tid), "wangid": wangids[tid]})


def _write(root: ET.Element, path: Path) -> None:
    ET.indent(root, space=" ")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n")


def write_tsx(res: BuildResult, out: Path) -> None:
    atlas = res.atlas
    T = atlas.T
    root = ET.Element(
        "tileset",
        {
            "version": TILED_VERSION,
            "name": res.spec.name,
            "tilewidth": str(T),
            "tileheight": str(T),
            "tilecount": str(atlas.tile_count),
            "columns": str(atlas.columns),
        },
    )
    ET.SubElement(
        root,
        "image",
        {
            "source": "tileset.png",
            "width": str(atlas.columns * T),
            "height": str(atlas.rows * T),
        },
    )
    for m in atlas.sorted_meta():
        if m.kind == "frame" and not m.animation:
            continue
        tile = ET.SubElement(root, "tile", {"id": str(m.id), "type": m.kind})
        props = ET.SubElement(tile, "properties")
        ET.SubElement(props, "property", {"name": "name", "value": m.name})
        ET.SubElement(props, "property", {"name": "group", "value": m.group})
        if m.animation:
            anim = ET.SubElement(tile, "animation")
            for tid, ms in m.animation:
                ET.SubElement(anim, "frame", {"tileid": str(tid), "duration": str(ms)})
    _write_wangset(res, root)
    _write(root, out / "tileset.tsx")


def write_tmx(res: BuildResult, pmap: PreviewMap, out: Path) -> None:
    T = res.atlas.T
    root = ET.Element(
        "map",
        {
            "version": TILED_VERSION,
            "orientation": "orthogonal",
            "renderorder": "right-down",
            "width": str(pmap.width),
            "height": str(pmap.height),
            "tilewidth": str(T),
            "tileheight": str(T),
            "infinite": "0",
            "nextlayerid": "3",
            "nextobjectid": "1",
        },
    )
    ET.SubElement(root, "tileset", {"firstgid": "1", "source": "tileset.tsx"})
    for lid, (name, grid) in enumerate((("ground", pmap.ground), ("objects", pmap.objects)), 1):
        layer = ET.SubElement(
            root, "layer", {"id": str(lid), "name": name, "width": str(pmap.width), "height": str(pmap.height)}
        )
        rows = [",".join(str(int(v) + 1) for v in row) for row in grid]  # gid 0 = empty, so -1 -> 0
        data = ET.SubElement(layer, "data", {"encoding": "csv"})
        data.text = "\n" + ",\n".join(rows) + "\n"
    _write(root, out / "preview.tmx")
