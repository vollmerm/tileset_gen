"""Semantic checks on a spec beyond what Pydantic's field types enforce."""

from __future__ import annotations

from pydantic import ValidationError

from . import registry
from .palette import HEX_RE
from .spec import TilesetSpec


def merged_colors(colors: list[str], info: registry.GeneratorInfo) -> list[str]:
    """Spec colors filled out with the generator's defaults for any missing slots."""
    defaults = info.default_colors()
    return list(colors[: len(defaults)]) + defaults[len(colors) :] if defaults else list(colors)


def validate_spec(spec: TilesetSpec) -> list[str]:
    errors: list[str] = []
    registry._load()
    names = {c.name.lower() for c in spec.palette}
    for c in spec.palette:
        if not HEX_RE.match(c.hex):
            errors.append(f"palette color {c.name!r}: {c.hex!r} is not #rrggbb")

    seen: set[str] = set()

    def check_name(kind: str, name: str) -> None:
        if name in seen:
            errors.append(f"{kind} {name!r}: duplicate name")
        seen.add(name)

    def check_colors(where: str, colors: list[str]) -> None:
        for col in colors:
            if col.lower() not in names and not HEX_RE.match(col):
                errors.append(f"{where}: color {col!r} is neither a palette name nor #rrggbb")

    def check_gen(where: str, gen: str, kind: str, params: dict) -> registry.GeneratorInfo | None:
        info = registry.REGISTRY.get(gen)
        if info is None or info.kind != kind:
            valid = ", ".join(g.name for g in registry.by_kind(kind))
            errors.append(f"{where}: unknown {kind} generator {gen!r} (valid: {valid})")
            return None
        try:
            info.params.model_validate(params)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"{where}: param {loc}: {err['msg']}")
        return info

    if not spec.terrains:
        errors.append("spec needs at least one terrain")
    terrains = {}
    for t in spec.terrains:
        check_name("terrain", t.name)
        check_colors(f"terrain {t.name!r}", t.colors)
        info = check_gen(f"terrain {t.name!r}", t.generator, "terrain", t.params)
        terrains[t.name] = info

    pairs = set()
    for tr in spec.transitions:
        where = f"transition {tr.lower}->{tr.upper}"
        for n in (tr.lower, tr.upper):
            if n not in terrains:
                errors.append(f"{where}: unknown terrain {n!r}")
        if tr.lower == tr.upper:
            errors.append(f"{where}: lower and upper must differ")
        pair = frozenset((tr.lower, tr.upper))
        if pair in pairs:
            errors.append(f"{where}: duplicate transition for this terrain pair")
        pairs.add(pair)

    for o in spec.objects:
        check_name("object", o.name)
        check_colors(f"object {o.name!r}", o.colors)
        info = check_gen(f"object {o.name!r}", o.generator, "object", o.params)
        if info is not None and info.connected and (o.width, o.height) != (1, 1):
            errors.append(f"object {o.name!r}: {o.generator} links up tile by tile, so width and height must be 1")
        for n in o.place_on:
            if n not in terrains:
                errors.append(f"object {o.name!r}: place_on references unknown terrain {n!r}")

    for a in spec.animations:
        where = f"animation {a.name!r}"
        check_name("animation", a.name)
        check_colors(where, a.colors)
        if (a.terrain is None) == (a.generator is None):
            errors.append(f"{where}: set exactly one of 'terrain' or 'generator'")
        elif a.terrain is not None:
            info = terrains.get(a.terrain)
            if a.terrain not in terrains:
                errors.append(f"{where}: unknown terrain {a.terrain!r}")
            elif info is not None and not info.animatable:
                errors.append(f"{where}: terrain generator {info.name!r} is not animatable")
        else:
            check_gen(where, a.generator, "animation", a.params)

    for ch in spec.characters:
        where = f"character {ch.name!r}"
        check_name("character", ch.name)
        check_colors(where, ch.colors)
        check_gen(where, ch.generator, "character", ch.params)
        for n in ch.place_on:
            if n not in terrains:
                errors.append(f"{where}: place_on references unknown terrain {n!r}")
    return errors
