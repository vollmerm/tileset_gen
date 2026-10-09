"""Tile collection: index-space tiles plus metadata, packed into a sheet.

Tile ids are sheet positions (row * columns + col). Multi-tile objects are
packed as rectangles so they look right in the sheet and can be picked as a
stamp in Tiled; this can leave empty (transparent) slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TileMeta:
    id: int
    name: str
    kind: str  # terrain | transition | object | animation | frame
    group: str
    props: dict = field(default_factory=dict)
    animation: list[tuple[int, int]] | None = None  # (tile id, duration ms)


@dataclass
class WangSet:
    name: str
    lower: str
    upper: str
    # corners (NW, NE, SE, SW) -> tile ids; several ids = random variants
    tiles: dict[tuple[int, int, int, int], list[int]] = field(default_factory=dict)


@dataclass
class ObjectGroup:
    name: str
    width: int
    height: int
    tiles: list[list[int]]  # [row][col] tile ids


class Atlas:
    def __init__(self, tile_size: int, columns: int = 16):
        self.T = tile_size
        self.columns = columns
        self.tiles: dict[int, np.ndarray] = {}
        self.meta: dict[int, TileMeta] = {}
        self._cursor = 0

    def _free(self, tid: int) -> bool:
        return tid not in self.tiles

    def _place(self, tid: int, arr: np.ndarray, name: str, kind: str, group: str, props: dict) -> int:
        assert arr.shape == (self.T, self.T), arr.shape
        self.tiles[tid] = arr.astype(np.uint8)
        self.meta[tid] = TileMeta(tid, name, kind, group, props)
        return tid

    def add(self, arr: np.ndarray, name: str, kind: str, group: str, **props) -> int:
        while not self._free(self._cursor):
            self._cursor += 1
        return self._place(self._cursor, arr, name, kind, group, props)

    def add_block(self, canvas: np.ndarray, name: str, kind: str, group: str, **props) -> list[list[int]]:
        """Place a (h*T, w*T) canvas as a w x h rectangle of tiles; returns ids [row][col]."""
        T, cols = self.T, self.columns
        h, w = canvas.shape[0] // T, canvas.shape[1] // T
        if w == 1 and h == 1:
            return [[self.add(canvas, name, kind, group, col=0, row=0, **props)]]
        if w > cols:
            raise ValueError(f"{name}: {w} tiles wide does not fit {cols} columns")
        start = self._cursor
        while True:
            r, c = divmod(start, cols)
            if c + w <= cols and all(self._free((r + j) * cols + c + i) for j in range(h) for i in range(w)):
                break
            start += 1
        ids = []
        for j in range(h):
            row = []
            for i in range(w):
                tid = (r + j) * cols + c + i
                tile = canvas[j * T : (j + 1) * T, i * T : (i + 1) * T]
                row.append(self._place(tid, tile, f"{name}_{i}_{j}", kind, group, {"col": i, "row": j, **props}))
            ids.append(row)
        return ids

    @property
    def tile_count(self) -> int:
        return max(self.tiles, default=-1) + 1

    @property
    def rows(self) -> int:
        return max(1, -(-self.tile_count // self.columns))

    def sorted_meta(self) -> list[TileMeta]:
        return [self.meta[k] for k in sorted(self.meta)]

    def sheet(self) -> np.ndarray:
        T = self.T
        out = np.zeros((self.rows * T, self.columns * T), dtype=np.uint8)
        for i, t in self.tiles.items():
            r, c = divmod(i, self.columns)
            out[r * T : (r + 1) * T, c * T : (c + 1) * T] = t
        return out
