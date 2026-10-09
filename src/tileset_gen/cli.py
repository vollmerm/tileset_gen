"""Command-line interface: plan, build, generate, validate, generators."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .spec import TilesetSpec

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Procedurally generate JRPG-style pixel-art tilesets."
)

SeedOpt = Annotated[Optional[int], typer.Option("--seed", help="Override the spec's random seed.")]
TileOpt = Annotated[int, typer.Option("--tile-size", help="Tile size in pixels (16 or 32).")]
MapOpt = Annotated[str, typer.Option("--map-size", help="Preview map size in tiles, WxH.")]
ScaleOpt = Annotated[int, typer.Option("--preview-scale", help="Nearest-neighbor upscale for preview.png.")]


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in value.lower().split("x"))
        return w, h
    except ValueError as e:
        raise typer.BadParameter(f"expected WxH, got {value!r}") from e


def _check_tile_size(tile_size: int) -> int:
    if tile_size not in (16, 32):
        raise typer.BadParameter("tile size must be 16 or 32")
    return tile_size


def _do_build(spec: TilesetSpec, out: Path, map_size: str, scale: int) -> None:
    from .build import SpecError, build

    try:
        res = build(spec, out, _parse_size(map_size), scale)
    except SpecError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{len(res.atlas.tiles)} tiles, {len(res.lut) - 1} colors -> {out}/")
    for name in ("tileset.png", "tileset.json", "tileset.tsx", "preview.tmx", "preview.png", "spec.json"):
        typer.echo(f"  {out / name}")


def _do_plan(prompt: str, seed: int | None, tile_size: int) -> TilesetSpec:
    import anthropic

    from .planner import PlannerError, plan

    typer.echo("Planning with Claude Haiku...", err=True)
    try:
        return plan(prompt, seed=seed or 0, tile_size=_check_tile_size(tile_size))
    except (PlannerError, anthropic.APIError) as e:
        typer.echo(f"planner failed: {e}", err=True)
        raise typer.Exit(1) from e


@app.command("plan")
def plan_cmd(
    prompt: Annotated[str, typer.Argument(help='Theme, e.g. "haunted swamp village".')],
    output: Annotated[Path, typer.Option("-o", "--output", help="Where to write the spec.")] = Path("spec.json"),
    seed: SeedOpt = None,
    tile_size: TileOpt = 16,
) -> None:
    """Ask the LLM planner for a spec and save it (no rendering)."""
    spec = _do_plan(prompt, seed, tile_size)
    spec.save(output)
    typer.echo(f"wrote {output}")


@app.command("build")
def build_cmd(
    spec_path: Annotated[Path, typer.Argument(help="Spec JSON file.", exists=True, dir_okay=False)],
    output: Annotated[Optional[Path], typer.Option("-o", "--output", help="Output directory.")] = None,
    seed: SeedOpt = None,
    map_size: MapOpt = "24x16",
    preview_scale: ScaleOpt = 3,
) -> None:
    """Render a spec to a tileset (offline, no API calls)."""
    spec = TilesetSpec.load(spec_path)
    if seed is not None:
        spec.seed = seed
    _do_build(spec, output or Path("out") / spec.name, map_size, preview_scale)


@app.command("generate")
def generate_cmd(
    prompt: Annotated[str, typer.Argument(help='Theme, e.g. "snowy mountain village".')],
    output: Annotated[Optional[Path], typer.Option("-o", "--output", help="Output directory.")] = None,
    seed: SeedOpt = None,
    tile_size: TileOpt = 16,
    map_size: MapOpt = "24x16",
    preview_scale: ScaleOpt = 3,
) -> None:
    """Plan a spec with the LLM, then build it."""
    spec = _do_plan(prompt, seed, tile_size)
    _do_build(spec, output or Path("out") / spec.name, map_size, preview_scale)


@app.command("validate")
def validate_cmd(spec_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Check a spec against the generator registry."""
    from .validate import validate_spec

    errors = validate_spec(TilesetSpec.load(spec_path))
    for e in errors:
        typer.echo(f"- {e}", err=True)
    if errors:
        raise typer.Exit(1)
    typer.echo("ok")


@app.command("generators")
def generators_cmd() -> None:
    """List available generators, their color slots and params."""
    from .registry import describe_for_prompt

    typer.echo(describe_for_prompt())
