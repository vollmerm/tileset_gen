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
    tone: int = 0  # shifts the first color's ramp this many steps (-1 = one shade darker throughout)

    def rng(self, *parts) -> np.random.Generator:
        """A generator seeded only by (spec seed, element key, parts): stable across runs."""
        digest = hashlib.blake2b(repr((self.seed, self.key, parts)).encode(), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def dither(self, amount: float) -> float:
        """Scale a generator's ordered-dither strength to the style: 8-bit art mostly used flat fills."""
        return amount * self.palette.style.dither

    def ramp(self, slot: int = 0, steps: int = 5) -> list[int]:
        r = self.palette.ramp(self.colors[slot], steps)
        if slot == 0 and self.tone:
            r = [r[min(max(i + self.tone, 0), steps - 1)] for i in range(steps)]
        return r
