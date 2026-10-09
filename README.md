# tileset-gen

Procedurally generates 8/16-bit JRPG-style pixel-art tilesets: terrain autotiles, objects and animated tiles, exported as a PNG atlas, JSON metadata and a Tiled tileset, with a rendered preview map.

An optional LLM planner (Claude Haiku) turns a theme like `"haunted swamp village"` into a **spec**: palette, terrains, transitions, objects and animations, plus their generator parameters. The model never draws pixels. Seeded Python generators render everything, so the same spec and seed always produce the same output.

## Quick start

```sh
uv sync

# Offline: render a hand-written spec
uv run tileset-gen build examples/specs/grassland.json -o out/grassland

# With the planner (needs ANTHROPIC_API_KEY or `ant auth login`)
uv run tileset-gen generate "snowy mountain village" -o out/snow
uv run tileset-gen plan "haunted swamp" -o swamp.json   # spec only; edit it, then `build`
```

Other commands: `tileset-gen validate SPEC`, `tileset-gen generators` (lists every generator, its color slots and params). Build options: `--seed`, `--map-size 24x16`, `--preview-scale 3`. Plan options: `--seed`, `--tile-size 16|32`.

## Output

| File | Contents |
|---|---|
| `tileset.png` | Tile atlas (16 columns at 16px). Multi-tile objects are packed as rectangles. |
| `tileset.json` | Engine-neutral metadata: per-tile name/kind/position, palette, wang sets, object groups, animations. |
| `tileset.tsx` | Tiled tileset with a corner Wang set per transition (terrain brush) and tile animations. |
| `preview.tmx` | Tiled map of the preview (ground + objects layers). |
| `preview.png` | Rendered preview map. |
| `spec.json` | The exact spec that was built, so you can rebuild offline. |

## Spec format

See `examples/specs/` for complete specs. The outline:

```jsonc
{
  "name": "grassland", "seed": 7, "tile_size": 16, "max_colors": 48,
  "palette": [{"name": "grass", "hex": "#5aa83c"}, ...],        // base colors; ramps are derived
  "terrains": [{"name": "grass", "generator": "grass", "colors": ["grass"], "variants": 3,
                "params": {"density": 0.5}}, ...],
  "transitions": [{"lower": "water", "upper": "sand", "edge_style": "shore"}, ...],  // soft | shore | cliff
  "objects": [{"name": "big_oak", "generator": "tree", "width": 2, "height": 2,
               "place_on": ["grass"], "density": 0.4}, ...],
  "animations": [{"name": "waves", "terrain": "water", "frames": 4, "frame_ms": 250},   // animate a terrain
                 {"name": "torch", "generator": "torch", "frames": 4}]                 // standalone
}
```

`colors` entries are palette names or `#rrggbb`, in the generator's slot order. Missing slots use the generator defaults.

## How it works

- **Index-space rendering.** Generators draw palette indices, not RGB. Each base color expands into a 5-step ramp whose shadows shift toward blue and highlights toward yellow. Before export, the palette is merged down to `max_colors`.
- **Seamless terrain.** Noise is periodic on the tile size. Variants share one base field and differ only in interior details, so any variant sits next to any other without a seam.
- **Autotiles.** Each transition produces the 16 corner-Wang combinations. The boundary is a bilinear blend of the four corner values plus tile-periodic noise, so neighboring tiles agree along shared edges. Edge styles add outlines, foam or cliff faces.
- **Animations.** Animatable terrains (water, lava) are re-rendered with a looping time phase, and frame 0 is the static tile. Transitions that touch an animated terrain get animated too.
- **Planner.** `planner.py` calls `claude-haiku-5-5` with structured outputs. The system prompt's generator catalog comes from the registry, so the prompt and code stay in sync. If the planned spec fails validation, the errors go back to the model once for a fix.

## Adding a generator

Register a function in `src/tileset_gen/generators/` with `@register(name, kind, ParamsModel, colors=[...], doc=...)`. Pydantic `Field(ge=, le=, description=)` constraints become both spec validation and planner documentation.

- Terrain: `fn(ctx, params, variant, phase) -> uint8[T, T]`
- Object: `fn(ctx, params, width, height) -> uint8[h*T, w*T]`, with 0 as transparent
- Animation: `fn(ctx, params, frame, n_frames) -> uint8[T, T]`

Use `ctx.ramp(slot)` for colors and `ctx.rng(...)` for randomness.

## Development

```sh
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
```

## Known limitations (v1)

- Autotiles are 16-tile corner sets. There's no 47-tile blob set, and no multi-terrain Wang set (each pair is its own set in Tiled).
- A single terrain variant repeats every tile. Low-contrast base fields hide most of it, but lava and water still show some repetition.
- Detail sprites (tufts, pebbles) are fixed-size, so they look sparse at 32px.
- The preview map only uses the longest chain of connected transitions; terrains off that chain don't appear in it.
