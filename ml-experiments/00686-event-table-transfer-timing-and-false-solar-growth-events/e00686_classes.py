"""Check 2, step 2: assign each false event to a class, and print counts, shares, and rates.

    DATA_PATH_INTERNAL=... PYTHONPATH=. uv run --no-sync python e00686_classes.py

Reads `false_events.parquet` written by `e00686_false_events.py` and writes `classes.parquet` and
`classes.json` beside it. The class is the first that matches, in this order (fixed in the plan):
sunny-run step, slow ramp, slow-level misfit, other step.
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00686-event-table-transfer-timing-and-false-solar-growth-events"
)
SUNNY_CLEARNESS: Final[float] = 0.7
STEP_DAYS: Final[float] = 1.0
RAMP_COMPLETE_DAYS: Final[float] = 14.0
MISFIT_UNITS: Final[float] = 0.5
PRIMARY_YEARS_PER_WORLD: Final[float] = 3 * 2.5
"""Three injected Primaries over 2024-01-01 to 2026-06-30."""
CLASSES: Final[tuple[str, ...]] = ("sunny-run step", "slow ramp", "slow-level misfit", "other step")


def assign(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the `cls686` column (first matching class)."""
    step = pl.col("edge_complete") & (pl.col("edge_days") <= STEP_DAYS)
    sunny = pl.col("mean_clearness_3d") >= SUNNY_CLEARNESS
    ramp = ~pl.col("edge_complete") | (pl.col("edge_days") > STEP_DAYS)
    misfit = pl.col("slow_misfit_units").abs() >= MISFIT_UNITS
    return frame.with_columns(
        pl.when(step & sunny)
        .then(pl.lit("sunny-run step"))
        .when(ramp)
        .then(pl.lit("slow ramp"))
        .when(misfit)
        .then(pl.lit("slow-level misfit"))
        .otherwise(pl.lit("other step"))
        .alias("cls686")
    )


def wilson(*, k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a share."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(c - h), float(c + h)


def main() -> None:
    """Assign classes and print the numbers the write-up quotes."""
    table = pl.read_parquet(WORK / "false_events.parquet")
    worlds = table.filter(pl.col("kind") == "world")["world"].to_list()
    fe = assign(table.filter(pl.col("kind") == "false_event"))
    fe.write_parquet(WORK / "classes.parquet")
    n = fe.height
    rng = np.random.default_rng(686)
    per_world = {
        c: np.array(
            [fe.filter((pl.col("world") == w) & (pl.col("cls686") == c)).height for w in worlds]
        )
        for c in CLASSES
    }
    total_w = sum(per_world.values())
    out: dict = {"n_events": n, "n_worlds": len(worlds)}
    rate = total_w / PRIMARY_YEARS_PER_WORLD
    boots = [rate[rng.integers(0, len(worlds), len(worlds))].mean() for _ in range(2000)]
    out["rate_per_primary_year"] = {
        "mean": float(rate.mean()),
        "lo": float(np.percentile(boots, 2.5)),
        "hi": float(np.percentile(boots, 97.5)),
    }
    first2 = [w for w in worlds if w < 2]
    out["worlds_0_1_false_events"] = [fe.filter(pl.col("world") == w).height for w in first2]
    out["rate_worlds_0_1"] = float(
        sum(out["worlds_0_1_false_events"]) / (2 * PRIMARY_YEARS_PER_WORLD)
    )
    out["classes"] = {}
    for c in CLASSES:
        k = int(per_world[c].sum())
        lo, hi = wilson(k=k, n=n)
        r = per_world[c] / PRIMARY_YEARS_PER_WORLD
        rb = [r[rng.integers(0, len(worlds), len(worlds))].mean() for _ in range(2000)]
        out["classes"][c] = {
            "count": k,
            "share": k / n,
            "share_lo": lo,
            "share_hi": hi,
            "rate": float(r.mean()),
            "rate_lo": float(np.percentile(rb, 2.5)),
            "rate_hi": float(np.percentile(rb, 97.5)),
        }
    # Worlds as the resampling unit for the two P3 shares.
    ramp_misfit = per_world["slow ramp"] + per_world["slow-level misfit"]
    for label, num in (
        ("ramp_or_misfit", ramp_misfit),
        ("sunny_step", per_world["sunny-run step"]),
    ):
        sh = []
        for _ in range(2000):
            pick = rng.integers(0, len(worlds), len(worlds))
            sh.append(num[pick].sum() / max(total_w[pick].sum(), 1))
        out[f"share_{label}_world_bootstrap"] = [
            float(num.sum() / total_w.sum()),
            float(np.percentile(sh, 2.5)),
            float(np.percentile(sh, 97.5)),
        ]
    out["unknown_clearness"] = int(fe["mean_clearness_3d"].is_nan().sum())
    out["edge_incomplete"] = int((~fe["edge_complete"]).sum())
    out["near_season_boundary_14d"] = int((fe["dist_season_boundary_days"] <= 14).sum())
    out["share_midday_ge_0p6"] = float((fe["share_midday"] >= 0.6).mean())
    out["median_share_midday_by_class"] = {
        c: float(fe.filter(pl.col("cls686") == c)["share_midday"].median()) for c in CLASSES
    }
    out["median_days_by_class"] = {
        c: float(fe.filter(pl.col("cls686") == c)["days"].median()) for c in CLASSES
    }
    # Post hoc (after the first look): duration and clear-sky cross-tabulation, because the
    # pre-registered ramp class mixes slow ramps with short bumps whose edge never completes.
    clear = fe["mean_clearness_3d"] >= SUNNY_CLEARNESS
    short = fe["days"] <= 3
    cross = {}
    for dur_name, dur in (("3 days or less", short), ("more than 3 days", ~short)):
        for sky_name, sky in (("clear >= 0.7", clear), ("clear < 0.7 or unknown", ~clear)):
            k = int((dur & sky).sum())
            lo, hi = wilson(k=k, n=n)
            cross[f"{dur_name} | {sky_name}"] = {"count": k, "share": k / n, "lo": lo, "hi": hi}
    out["post_hoc_cross"] = cross
    k = int(clear.sum())
    out["post_hoc_clear_share"] = [k / n, *wilson(k=k, n=n)]
    k = int(short.sum())
    out["post_hoc_short_share"] = [k / n, *wilson(k=k, n=n)]
    k = int(
        (fe["cls686"] == "slow ramp").sum()
        - ((fe["cls686"] == "slow ramp") & ~fe["edge_complete"]).sum()
    )
    out["post_hoc_completed_ramp"] = [k, k / n, *wilson(k=k, n=n)]
    (WORK / "classes.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
