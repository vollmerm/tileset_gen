"""tileset.png + tileset.json (engine-neutral metadata)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ..build import BuildResult
from ..palette import rgb_to_hex


def write_png_json(res: BuildResult, out: Path) -> None:
    atlas = res.atlas
    Image.fromarray(res.to_rgba(atlas.sheet()), "RGBA").save(out / "tileset.png", optimize=True)
    T = atlas.T
    doc = {
        "name": res.spec.name,
        "theme": res.spec.theme,
        "image": "tileset.png",
        "tile_size": T,
        "columns": atlas.columns,
        "tile_count": atlas.tile_count,
        "palette": [rgb_to_hex(tuple(int(v) for v in row[:3])) for row in res.lut[1:]],
        "tiles": [
            {
                "id": m.id,
                "name": m.name,
                "kind": m.kind,
                "group": m.group,
                "x": (m.id % atlas.columns) * T,
                "y": (m.id // atlas.columns) * T,
                **({"props": m.props} if m.props else {}),
                **({"animation": [{"tile": t, "ms": ms} for t, ms in m.animation]} if m.animation else {}),
            }
            for m in atlas.sorted_meta()
        ],
        "terrains": res.terrain_tiles,
        "wangsets": [
            {
                "name": ws.name,
                "lower": ws.lower,
                "upper": ws.upper,
                "corner_order": "NW,NE,SE,SW (1 = upper)",
                "tiles": {"".join(map(str, k)): v for k, v in sorted(ws.tiles.items())},
            }
            for ws in res.wangsets
        ],
        "objects": [{"name": o.name, "width": o.width, "height": o.height, "tiles": o.tiles} for o in res.objects],
        "connected": [
            {
                "name": c.name,
                "mask_bits": "N=1,E=2,S=4,W=8",
                "tiles": {str(m): t for m, t in c.tiles.items()},
                "gate": c.gate,
            }
            for c in res.connected
        ],
        "animations": res.animations,
    }
    (out / "tileset.json").write_text(json.dumps(doc, indent=2) + "\n")
