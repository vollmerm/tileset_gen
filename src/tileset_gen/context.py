"""Per-element generation context: tile size, palette access and seeded RNGs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .palette import Palette


@dataclass
class GenContext:
    T: int
    seed: int
    key: str
    palette: Palette
    colors: list[str]

    def rng(self, *parts) -> np.random.Generator:
        """A generator seeded only by (spec seed, element key, parts): stable across runs."""
        digest = hashlib.blake2b(repr((self.seed, self.key, parts)).encode(), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def ramp(self, slot: int = 0, steps: int = 5) -> list[int]:
        return self.palette.ramp(self.colors[slot], steps)
