# tileset-gen

Procedurally generates 8/16-bit JRPG-style pixel-art tilesets: terrain autotiles, objects and animated tiles, exported as a PNG atlas, JSON metadata and a Tiled tileset, with a rendered preview map. It also draws walking character sprite sheets (heroes, townsfolk, monsters) in the same palette and lighting, so they fit the tileset.

An optional LLM planner (Claude Haiku) turns a theme like `"haunted swamp village"` into a **spec**: palette, terrains, transitions, objects, animations and characters, plus their generator parameters. The model never draws pixels. Seeded Python generators render everything, so the same spec and seed always produce the same output.

## Quick start

```sh
uv sync

# Offline: render a hand-written spec
uv run tileset-gen build examples/specs/grassland.json -o out/grassland

# With the planner (needs ANTHROPIC_API_KEY or `ant auth login`)
uv run tileset-gen generate "snowy mountain village" -o out/snow
uv run tileset-gen plan "haunted swamp" -o swamp.json   # spec only; edit it, then `build`

# Edit an existing spec in plain language; prints a diff, keeps swamp.json.bak
uv run tileset-gen refine swamp.json "add a graveyard and make the water murkier" --build out/swamp
```

Other commands: `tileset-gen validate SPEC`, `tileset-gen generators` (lists every generator, its color slots and params). Build options: `--seed`, `--map-size 24x16`, `--preview-scale 3`. Plan options: `--seed`, `--tile-size 16|32`.

## Output

| File | Contents |
|---|---|
| `tileset.png` | Tile atlas (16 columns at 16px). Multi-tile objects are packed as rectangles. Object shadows are semi-transparent pixels. |
| `tileset.json` | Engine-neutral metadata: per-tile name/kind/position, palette, wang sets, object groups, animations. Decor tiles carry `props.decor` and `props.probability`. `connected` lists each fence's tiles by neighbor mask (N=1, E=2, S=4, W=8), plus its gate. |
| `tileset.tsx` | Tiled tileset with one corner Wang set covering every terrain, including tone patches (a single terrain-brush palette), and tile animations. Each fence gets an edge Wang set, so the Wang brush draws fences that join up; place the gate by hand. Decor tiles have a tile `probability`, so the terrain brush scatters them rarely. |
| `preview.tmx` | Tiled map of the preview (ground + objects layers). |
| `preview.png` | Rendered preview map. |
| `spec.json` | The exact spec that was built, so you can rebuild offline. |
| `characters/<name>.png` | One sprite sheet per character: rows are directions (down, left, right, up), columns are 3 walk frames (step, stand, step), the RPG Maker layout. Frames are one tile wide and 1 or 1.5 tiles tall. |
| `characters/<name>.tsx` | Tiled tileset for that sheet. Each direction's standing frame plays the 0-1-2-1 walk cycle, and objects anchor bottom-center. |
| `characters.json` | Frame size, per-character palette, frame timing and the frame coordinates of each direction's walk cycle. |

`preview.tmx` also gets a `characters` object layer and `preview.png` shows them standing around the map.

## Spec format

See `examples/specs/` for complete specs. The outline:

```jsonc
{
  "name": "grassland", "seed": 7, "tile_size": 16, "max_colors": 48,
  "palette": [{"name": "grass", "hex": "#5aa83c"}, ...],        // base colors; ramps are derived
  "terrains": [{"name": "grass", "generator": "grass", "colors": ["grass"], "variants": 3, "decor": 3, "patches": 0.3,
                "params": {"density": 0.5}}, ...],
  "transitions": [{"lower": "water", "upper": "sand", "edge_style": "shore"}, ...],  // soft | shore | cliff | blend
  "objects": [{"name": "big_oak", "generator": "tree", "width": 2, "height": 2,
               "place_on": ["grass"], "density": 0.4}, ...],
  "animations": [{"name": "waves", "terrain": "water", "frames": 4, "frame_ms": 250},   // animate a terrain
                 {"name": "torch", "generator": "torch", "frames": 4}],                // standalone
  "characters": [{"name": "hero", "generator": "humanoid", "colors": ["#f0c49a", "#7a4a2a", "water", "gold"],
                  "params": {"hair": "spiky", "cape": true}, "place_on": ["grass"], "count": 1,
                  "max_colors": 15, "frame_ms": 160}, ...]
}
```

`colors` entries are palette names or `#rrggbb`, in the generator's slot order. Missing slots use the generator defaults.

`variants` (1-4) are plain versions of a terrain that tile with each other. `decor` (0-6, default 2) adds rare detail tiles drawn from the generator's feature set: tall grass, flowers and stones on grass, puddles and cracks on dirt, lily pads on water, footprints and ice on snow. `tileset-gen generators` lists each terrain's features. Each decor tile has probability 0.12, against 1.0 for a plain variant.

`patches` (0-0.6, default 0.3) adds a darker tone of the terrain, such as lush grass, damp earth or deep water. The preview scatters it in organic patches several tiles across, over that share of the terrain's open ground. Behind the scenes it is an extra terrain `<name>_patch`, drawn with the host's ramp one step darker so it costs no palette colors. It joins the host through a `blend` transition: a dithered seam with no outline. Set it to 0 for a uniform surface such as a tiled floor.

Landmark generators (house, well, chest, sign, door) have a `limit`: the preview places at most that many copies per 24x16 of map, whatever their density. `tileset-gen generators` shows the limits. Fences are *connected* objects. The generator draws one tile for each combination of neighbors the fence continues into, plus a gate. The preview builds rectangular pens with a gate in the south side, then puts something inside. Connected objects must be 1x1.

### Characters

Character generators draw a 4-direction, 3-frame walk cycle:

- `humanoid` (5 color slots: skin, hair, outfit, accent, legs). `body` is `classic` (16x24 at 16px tiles, like FF6 or Chrono Trigger) or `chibi` (16x16, like Dragon Quest). Pick an `outfit` (tunic, robe, armor, dress), `hair` (short, long, spiky, ponytail, bald) and `headgear` (none, hood, wizard_hat, helmet, bandana, crown), and optionally a `cape` and a `beard`. Together these cover heroes, mages, knights, villagers, kings and so on.
- `slime`: a Dragon Quest-style teardrop or round blob that squashes and stretches as it moves.
- `ghost`: a floating sheet that bobs, with a rippling hem and a detached shadow.

To make characters match a tileset, name its palette colors in `colors` (a robe in the roof color, armor in the stone color). Their ramps, light direction, selective outlines and translucent ground shadow then match the tileset's objects. Each character gets its own palette, reduced to `max_colors` (default 15, the size of a SNES sprite palette; 3 gives an NES look). Characters don't use up the tileset's `max_colors`. "Right" frames are "left" mirrored, as on the SNES. In the preview, `count` copies stand on open `place_on` ground, kept off objects and away from each other.

## How it works

- **Index-space rendering.** Generators draw palette indices, not RGB. Each base color expands into a 5-step ramp whose shadows shift toward blue and highlights toward yellow. Before export, the palette is merged down to `max_colors`: the closest pair of rarely used colors is merged first, with distance measured in CIELAB so that merges follow what the eye sees.
- **Seamless terrain.** Every variant has the same 1px border ring, taken from a shared tile-periodic field. Inside, the field fades to the variant's own noise within a few pixels. The shared border uses finer, lower-contrast noise than the interior so the tile grid doesn't show. Small details near the edge (tufts, pebbles) come from a shared random stream and wrap across the border. Details further in belong to the variant. Any two variants meet without a seam but look different inside.
- **Tone patches.** Variants can only differ inside one tile, so on their own they can't make variation bigger than a tile. Patches add that larger scale: they are a darker terrain joined to the host by blend autotiles, placed with low-frequency noise.
- **Decor tiles.** A decor tile is a copy of a variant with one feature drawn inside a 2px margin, so its border matches. The builder adds `decor` of them per terrain at low probability, both in Tiled and in the preview.
- **3/4 perspective.** Objects use the SNES-JRPG view: you see their fronts and their tops. Examples are the chest lid, barrel head, wall cap, house roof with lit and shaded hip slopes, and the chimney top. Light comes from the upper left. The eaves shade the wall below them, doors and windows sit recessed, and anything standing on the ground casts a shadow down and to the right. Shadows use a reserved palette index that is exported as translucent dark violet, so in Tiled they darken whatever ground is underneath. The preview composites them by stepping the ground two shades down its ramp.
- **Autotiles.** Each transition produces the 16 corner-Wang combinations. The boundary is a bilinear blend of the four corner values plus tile-periodic noise, so neighboring tiles agree along shared edges. Edge styles add outlines, foam or cliff faces.
- **Animations.** Animatable terrains (water, lava) are re-rendered with a looping time phase, and frame 0 is the static tile. Transitions that touch an animated terrain get animated too.
- **Planner.** `planner.py` calls `claude-haiku-5-5` with structured outputs. The system prompt's generator catalog comes from the registry, so the prompt and code stay in sync. If the planned spec fails validation, the errors go back to the model once for a fix. The wire schema avoids unions and nullable fields, so params are `"name=value"` strings: the API compiles the schema into a grammar with a size limit, and the spec is close to it. Keep new spec fields off the wire unless the model needs to choose them (`refine` carries per-character `max_colors` and `frame_ms` over from the old spec).

## Adding a generator

Register a function in `src/tileset_gen/generators/` with `@register(name, kind, ParamsModel, colors=[...], doc=...)`. Pydantic `Field(ge=, le=, description=)` constraints become both spec validation and planner documentation.

- Terrain: `fn(ctx, params, variant, phase) -> uint8[T, T]`
- Object: `fn(ctx, params, width, height) -> uint8[h*T, w*T]`, with 0 as transparent
- Animation: `fn(ctx, params, frame, n_frames) -> uint8[T, T]`
- Character: `fn(ctx, params, direction, frame) -> uint8[h, T]`, where direction is `down`, `left` or `up` and frame is 0-2. Every frame must have the same height. Lay shapes out with `_Figure` in `generators/characters.py`, which works in 16px-sprite units so one design renders at both tile sizes.

Use `ctx.ramp(slot)` for colors and `ctx.rng(...)` for randomness.

Terrains can pass `decor=(feature, ...)` to `register`. A feature is `fn(tile, ctx, params, rng)` that draws into the tile in place, staying 2px inside the edge. Pass `connected=True` for objects that link up tile by tile. Their signature is `fn(ctx, params, mask, gate=False) -> uint8[T, T]`. Objects should call `outline` and then `ground_shadow`/`drop_shadow` (from `draw.py`), so the shadow doesn't get an outline of its own.

## Development

```sh
uv run pytest
uv run ruff check src tests scripts && uv run ruff format src tests scripts
```

`tests/test_generators.py` renders every generator at each param's extremes, across seeds and both tile sizes. If a new generator crashes on some valid spec, that test should catch it.

### Planner eval

`scripts/eval_planner.py` runs the planner over a set of themes (or `--themes FILE`, one per line). It reports first-try and final validity, render crashes, latency, token usage and chosen densities, and saves each spec under `out/eval/`. `--build` also renders each one. One run of the 8 default themes costs a few cents with Haiku. Re-run it after changing the planner prompt or the generator catalog.

## Known limitations (v1)

- Autotiles are 16-tile corner sets per transition; there's no 47-tile blob set. Tiled's brush can only put two terrains next to each other if the spec has a transition for that pair.
- Terrain variants share their border pixels and differ only inside, so a terrain with one variant still repeats every tile. Use 2–3 variants for natural ground.
- Transition tiles are built from each terrain's first variant only, and there is one tile per corner combination. Long, thin strips made of transitions repeat more than solid areas do.
- Lava's veins are lines, and blending in a whole per-variant field would draw a vein ring inside each tile. So lava variants only nudge the shared field, and lava repeats more than other terrains. Decor tiles (crust slabs, bubbles) help.
- In the preview map, the longest transition chain sets the elevation bands. Side branches appear as patches inside their host terrain, and terrains with no transitions appear as hard-edged rectangles.
