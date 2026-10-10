"""Every generator must render at its parameter extremes, at both tile sizes, without crashing."""

import itertools
from typing import Literal, get_args, get_origin

import numpy as np
import pytest

from tileset_gen import registry
from tileset_gen.context import GenContext
from tileset_gen.generators.characters import EMOTES, ROWS, emote_balloon
from tileset_gen.palette import Palette

registry._load()


def extremes(params_model) -> list[dict]:
    """Param dicts covering each field's min, max and every enum/bool option (one field varied at a time)."""
    out = [{}]
    for name, field in params_model.model_fields.items():
        ann = field.annotation
        if get_origin(ann) is Literal:
            values = list(get_args(ann))
        elif ann is bool:
            values = [False, True]
        else:
            lo = next((m.ge for m in field.metadata if hasattr(m, "ge")), None)
            hi = next((m.le for m in field.metadata if hasattr(m, "le")), None)
            values = [v for v in (lo, hi) if v is not None]
        out += [{name: v} for v in values]
    return out


CASES = [
    (g.name, ps, T) for g in registry.REGISTRY.values() for ps, T in itertools.product(extremes(g.params), (16, 32))
]


@pytest.mark.parametrize("name,params,T", CASES, ids=lambda v: str(v))
def test_generator_extremes(name: str, params: dict, T: int):
    g = registry.get(name)
    p = g.params.model_validate(params)
    sizes = [(1, 1), (2, 2), (3, 3), (1, 2)] if g.kind == "object" else [None]
    seeds = range(40) if g.kind == "object" else range(6) if g.kind == "character" else range(3)
    for seed, size in itertools.product(seeds, sizes):
        ctx = GenContext(T, seed, f"fuzz:{name}", Palette(), g.default_colors())
        if g.kind == "terrain":
            out = [g.fn(ctx, p, v, ph) for v in (0, 3) for ph in (0.0, 0.5)]
            shape = (T, T)
        elif g.connected:
            out = [g.fn(ctx, p, mask) for mask in range(16)] + [g.fn(ctx, p, 2 | 8, gate=True)]
            shape = (T, T)
        elif g.kind == "character":
            out = [g.fn(ctx, p, a, d, k) for a, d in ROWS if d != "right" for k in range(3)]
            shape = out[0].shape
            assert shape[1] == 2 * T and shape[0] in (T * 3 // 2, 2 * T)
            assert all((a > 0).any() for a in out), "every frame draws something"
            # Nothing important may be cut off at the frame edge: only the outline may touch it.
            hurt_flash = [r for r, d in ROWS if d != "right"].index("hurt") * 3 + 1
            for a in out[:hurt_flash] + out[hurt_flash + 1 :]:
                edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
                assert not (edge > 1).any() or set(edge[edge > 1]) <= {ctx.palette.outline_for(int(v)) for v in edge}
        elif g.kind == "object":
            out = [g.fn(ctx, p, *size)]
            shape = (size[1] * T, size[0] * T)
        else:
            out = [g.fn(ctx, p, f, 6) for f in range(6)]
            shape = (T, T)
        for arr in out:
            assert arr.shape == shape and arr.dtype == np.uint8
            assert arr.max() < len(ctx.palette.colors)
            if g.kind == "terrain":
                assert arr.min() > 0, "terrain tiles must be fully opaque"


@pytest.mark.parametrize("T", (16, 32))
def test_emote_balloons(T: int):
    ctx = GenContext(T, 0, "emotes", Palette(), [])
    for e in EMOTES:
        frames = [emote_balloon(ctx, e, k) for k in range(3)]
        assert all(f.shape == (T, T) and (f > 0).any() for f in frames)
        assert (frames[0] > 0).sum() < (frames[2] > 0).sum(), "the balloon pops in small"
