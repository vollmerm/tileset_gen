"""Procedural JRPG pixel-art tileset generator."""

from .spec import TilesetSpec


def main() -> None:
    from .cli import app

    app()


__all__ = ["TilesetSpec", "main"]
