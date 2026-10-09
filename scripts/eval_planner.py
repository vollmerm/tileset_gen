"""Run the LLM planner over a set of themes and report how reliably it produces valid, usable specs.

Usage: uv run python scripts/eval_planner.py [--themes FILE] [--out DIR] [--build]

Makes one or two Claude Haiku calls per theme (needs ANTHROPIC_API_KEY).
Writes each spec (and with --build, each tileset) under --out for inspection.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from tileset_gen.build import build, generate
from tileset_gen.planner import PlannerError, PlanStats, plan

DEFAULT_THEMES = [
    "snowy mountain village",
    "haunted swamp with a ruined chapel",
    "tropical island port town",
    "volcanic dungeon with lava rivers",
    "autumn forest with a woodcutter's camp",
    "desert oasis bazaar",
    "crystal cave",
    "royal castle courtyard",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes", type=Path, help="text file, one theme per line")
    ap.add_argument("--out", type=Path, default=Path("out/eval"))
    ap.add_argument("--build", action="store_true", help="also render each spec")
    args = ap.parse_args()
    themes = args.themes.read_text().splitlines() if args.themes else DEFAULT_THEMES
    themes = [t.strip() for t in themes if t.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for theme in themes:
        stats = PlanStats()
        t0 = time.monotonic()
        try:
            spec = plan(theme, stats=stats)
            ok = True
        except PlannerError as e:
            spec, ok = None, False
            print(f"  FAILED {theme!r}: {e}")
        dt = time.monotonic() - t0
        slug = theme.replace(" ", "_").replace("'", "")[:40]
        if spec:
            try:
                generate(spec)
            except Exception as e:  # noqa: BLE001 - report any render crash as an eval failure
                ok = False
                print(f"  RENDER CRASH {theme!r}: {type(e).__name__}: {e}")
            spec.save(args.out / f"{slug}.json")
            if args.build:
                build(spec, args.out / slug, preview_scale=2)
        rows.append((theme, ok, stats, dt, spec))
        if not stats.attempts:
            first = "?"
        else:
            first = "ok" if not stats.attempts[0] else f"{len(stats.attempts[0])} err"
        detail = ""
        if spec:
            dens = [o.density for o in spec.objects]
            detail = (
                f"T{len(spec.terrains)} X{len(spec.transitions)} O{len(spec.objects)} A{len(spec.animations)} "
                f"density med {statistics.median(dens) if dens else 0:.2f}"
            )
        print(
            f"{theme[:40]:40s} first={first:6s} final={'ok' if ok else 'FAIL':4s} {dt:5.1f}s "
            f"in={stats.input_tokens} out={stats.output_tokens} {detail}"
        )
        for i, errs in enumerate(stats.attempts):
            for e in errs:
                print(f"    attempt {i + 1}: {e}")

    n = len(rows)
    first_ok = sum(1 for r in rows if r[2].attempts and not r[2].attempts[0])
    final_ok = sum(1 for r in rows if r[1])
    print(
        f"\nfirst-try valid {first_ok}/{n}, final valid {final_ok}/{n}, "
        f"mean latency {statistics.mean(r[3] for r in rows):.1f}s, "
        f"tokens in {sum(r[2].input_tokens for r in rows)} out {sum(r[2].output_tokens for r in rows)}"
    )


if __name__ == "__main__":
    main()
