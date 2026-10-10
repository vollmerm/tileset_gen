"""Generator registry.

Each generator declares its kind, a Pydantic params model, default colors per
slot and a one-line description. The planner prompt and spec validation are
both derived from this table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

Kind = Literal["terrain", "object", "animation", "character"]


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class GeneratorInfo:
    name: str
    kind: Kind
    fn: Callable
    params: type[Params]
    colors: tuple[tuple[str, str], ...]  # (slot name, default hex)
    doc: str
    animatable: bool = False
    decor: tuple[Callable, ...] = ()  # terrain only: features drawn onto rare detail tiles
    connected: bool = False  # object only: fn(ctx, params, mask, gate) draws one tile per N/E/S/W neighbor mask
    limit: int | None = None  # object only: landmark cap per 24x16 of preview map

    def default_colors(self) -> list[str]:
        return [hexv for _, hexv in self.colors]


REGISTRY: dict[str, GeneratorInfo] = {}


def register(
    name: str,
    kind: Kind,
    params: type[Params] = Params,
    colors=(),
    doc: str = "",
    animatable: bool = False,
    decor=(),
    connected: bool = False,
    limit: int | None = None,
):
    def deco(fn):
        REGISTRY[name] = GeneratorInfo(
            name, kind, fn, params, tuple(colors), doc, animatable, tuple(decor), connected, limit
        )
        return fn

    return deco


def get(name: str) -> GeneratorInfo:
    _load()
    return REGISTRY[name]


def by_kind(kind: Kind) -> list[GeneratorInfo]:
    _load()
    return [g for g in REGISTRY.values() if g.kind == kind]


def _load() -> None:
    from . import generators  # noqa: F401  (registers on import)


def _describe_field(name: str, field) -> str:
    ann = field.annotation
    lo = hi = None
    for m in field.metadata:
        lo = getattr(m, "ge", lo)
        hi = getattr(m, "le", hi)
    if getattr(ann, "__origin__", None) is Literal:
        kind = "|".join(str(a) for a in get_args(ann))
    elif ann is bool:
        kind = "bool"
    elif lo is not None and hi is not None:
        kind = f"{ann.__name__} {lo}..{hi}"
    else:
        kind = getattr(ann, "__name__", str(ann))
    desc = f" - {field.description}" if field.description else ""
    return f"{name} ({kind}, default {field.default}){desc}"


def describe_for_prompt() -> str:
    """Human/LLM-readable catalog of every generator, its color slots and params."""
    _load()
    lines = []
    for kind in ("terrain", "object", "animation", "character"):
        lines.append(f"## {kind} generators")
        for g in REGISTRY.values():
            if g.kind != kind:
                continue
            flag = " [animatable]" if g.animatable else " [connected]" if g.connected else ""
            limit = f" (landmark: at most {g.limit} per preview map)" if g.limit else ""
            lines.append(f"- {g.name}{flag}: {g.doc}{limit}")
            if g.colors:
                lines.append("  colors: " + ", ".join(f"{slot} (default {hexv})" for slot, hexv in g.colors))
            if g.decor:
                lines.append("  decor: " + ", ".join(f.__name__.lstrip("_") for f in g.decor))
            for fname, field in g.params.model_fields.items():
                lines.append("  param " + _describe_field(fname, field))
        lines.append("")
    return "\n".join(lines)
