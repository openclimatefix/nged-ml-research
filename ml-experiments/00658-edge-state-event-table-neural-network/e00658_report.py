"""Print every number the page of experiment 658 quotes.

    DATA_PATH_INTERNAL=.. PYTHONPATH=<dir of 653>:. uv run --no-sync python e00658_report.py STAGE

Stages: `injected` and `cross`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00658_train import WORK

KINDS: Final[list[str]] = ["edge", "solo", "permanent_transfer", "permanent_shift"]
BANDS: Final[list[str]] = ["3h-1d", "1-7d", "7-21d", "21-60d", "60d+", "permanent"]
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
PREFIX: Final[str] = os.environ.get("E658_PREFIX", "main")


def wrec(df: pl.DataFrame, col: str) -> float:
    """MW-weighted share of rows with `col` true."""
    w = df["amp_mw"].abs()
    return float((df[col].cast(pl.Float64) * w).sum()) / float(w.sum()) if df.height else math.nan


def big(df: pl.DataFrame) -> pl.DataFrame:
    """Injected positive events of 4 noise units or more lasting a day or more (or no end)."""
    return df.filter(
        pl.col("kind").is_in(KINDS)
        & (pl.col("units") >= 4)
        & ((pl.col("days") >= 1) | (pl.col("days") < 0))
    )


def band(df: pl.DataFrame) -> pl.DataFrame:
    """Add the duration band."""
    d = pl.col("days")
    return df.with_columns(
        band=pl.when(d < 0)
        .then(pl.lit("permanent"))
        .when(d < 1)
        .then(pl.lit("3h-1d"))
        .when(d < 7)
        .then(pl.lit("1-7d"))
        .when(d < 21)
        .then(pl.lit("7-21d"))
        .when(d < 60)
        .then(pl.lit("21-60d"))
        .otherwise(pl.lit("60d+"))
    )


def timing(df: pl.DataFrame) -> tuple[float, float, float]:
    """Start within 2 h, end within 6 h (misses fail), and end within 6 h among found events."""
    s = np.mean([(not math.isnan(a)) and abs(a) <= 4 for a in df["start_err_hh"]])
    ends = df.filter(pl.col("end") > 0)
    e_all = np.mean([(not math.isnan(a)) and abs(a) <= 12 for a in ends["end_err_hh"]])
    fnd = ends.filter(pl.col("found"))
    e_found = (
        np.mean([abs(a) <= 12 for a in fnd["end_err_hh"].drop_nans()]) if fnd.height else math.nan
    )
    return float(s), float(e_all), float(e_found)


def summary(df: pl.DataFrame) -> dict[str, float]:
    """The headline recall, pairing, and timing numbers of one set of rows."""
    b = big(df)
    edge = b.filter(pl.col("edge_truth"))
    s, e, ef = timing(b)
    return {
        "n": b.height,
        "found": wrec(b, "found"),
        "found_unweighted": float(b["found"].cast(pl.Float64).sum()) / b.height,
        "paired": wrec(edge, "right_class") if edge.height else math.nan,
        "start2h": s,
        "end6h_all": e,
        "end6h_found": ef,
    }


def boot(
    df: pl.DataFrame, fn: Callable[[pl.DataFrame], float], n: int = 2000
) -> tuple[float, float]:
    """95% interval resampling whole worlds."""
    rng = np.random.default_rng(3)
    worlds = df["world"].unique().to_list()
    vals = []
    for _ in range(n):
        pick = rng.choice(worlds, size=len(worlds))
        vals.append(fn(pl.concat([df.filter(pl.col("world") == w) for w in pick])))
    return tuple(np.round(np.nanpercentile(vals, [2.5, 97.5]), 3))  # type: ignore[return-value]


def picked(tag: str) -> dict[str, float]:
    """Costs chosen on validation."""
    return json.loads((WORK / f"costs_{tag}.json").read_text())


def rate_of(tag: str, on: float, boost: float) -> float:
    """Real-record events per Primary-year at a grid point (scored years)."""
    g = json.loads((WORK / f"rewired_{tag}.json").read_text())["grid"]
    n = next(r["n_scored"] for r in g if r["on"] == on and r["boost"] == boost)
    return n / (15 * YEARS)


def injected() -> None:
    """Recall, pairing, timing, bands, and the recall-against-rate curve."""
    base = band(pl.read_parquet(WORK / "baseline_e653.parquet"))
    print("== Experiment 653 default arm re-run (spec default)")
    print(summary(base.filter(pl.col("spec") == "default")))
    stored = band(
        pl.read_parquet(
            WORK.parent
            / "00653-edge-state-event-table-penalised-segmentation/injected_truth_scored.parquet"
        )
    )
    print(
        "   stored table:",
        summary(stored.filter((pl.col("spec") == "default") & (pl.col("arm") == "masked"))),
    )
    for arm in ("net", "e653pair"):
        print(f"\n== Network, arm {arm}, picked costs, spec default, per seed")
        per = []
        for s in SEEDS:
            tag = f"{PREFIX}_seed{s}"
            p = picked(tag)
            d = band(pl.read_parquet(WORK / f"injected_{tag}.parquet")).filter(
                (pl.col("arm") == arm)
                & (pl.col("on") == p["on"])
                & (pl.col("boost") == p["boost"])
                & ((pl.col("cutoff") == p["cutoff"]) | (pl.col("arm") == "e653pair"))
            )
            dd = d.filter(pl.col("spec") == "default")
            per.append(dd)
            print(tag, p, {k: round(v, 3) for k, v in summary(dd).items()})
        pooled = pl.concat(per)
        print(
            "seed mean found",
            np.mean([summary(x)["found"] for x in per]).round(3),
            "paired",
            np.nanmean([summary(x)["paired"] for x in per]).round(3),
        )
        if arm == "net":
            print("interval (seed 0 worlds) found", boot(per[0], lambda g: summary(g)["found"]))
            for b in BANDS:
                g = big(pooled).filter(pl.col("band") == b)
                print(f"  {b}: found {wrec(g, 'found'):.2f} (n={g.height})")
            for spec in ("constant", "proportional", "short", "long", "stress"):
                g = pl.concat(
                    [
                        band(pl.read_parquet(WORK / f"injected_{PREFIX}_seed{s}.parquet")).filter(
                            (pl.col("arm") == "net")
                            & (pl.col("spec") == spec)
                            & (pl.col("on") == picked(f"{PREFIX}_seed{s}")["on"])
                            & (pl.col("boost") == picked(f"{PREFIX}_seed{s}")["boost"])
                            & (pl.col("cutoff") == picked(f"{PREFIX}_seed{s}")["cutoff"])
                        )
                        for s in SEEDS
                    ]
                )
                print(
                    f"  spec {spec}: found {summary(g)['found']:.3f}, "
                    f"paired {summary(g)['paired']:.3f}"
                )
    print("\n== Experiment 653 by spec (same worlds)")
    for spec in ("default", "constant", "proportional", "short", "long", "stress"):
        s = summary(base.filter(pl.col("spec") == spec))
        print(f"  {spec}: found {s['found']:.3f} paired {s['paired']:.3f}")
    for b in BANDS:
        g = big(base.filter(pl.col("spec") == "default")).filter(pl.col("band") == b)
        print(f"  e653 {b}: found {wrec(g, 'found'):.2f} (n={g.height})")
    print("\n== Recall against real-record events per Primary-year (all grid points, seed mean)")
    rows = []
    for s in SEEDS:
        tag = f"{PREFIX}_seed{s}"
        d = pl.read_parquet(WORK / f"injected_{tag}.parquet").filter(
            (pl.col("arm") == "net")
            & (pl.col("spec") == "default")
            & (pl.col("cutoff") == picked(tag)["cutoff"])
        )
        for (on, boost), g in d.group_by(["on", "boost"]):
            rows.append(
                {
                    "seed": s,
                    "on": on,
                    "boost": boost,
                    "rate": rate_of(tag, on, boost),
                    "found": wrec(big(g), "found"),
                }
            )
    print(
        pl.DataFrame(rows)
        .group_by(["on", "boost"])
        .agg(pl.col("rate").mean(), pl.col("found").mean())
        .sort("rate")
    )
    print(
        "   experiment 653 point: rate 6.57, found",
        round(summary(base.filter(pl.col("spec") == "default"))["found"], 3),
    )


def cross() -> None:
    """Cross-injector drop: same-injector score minus cross score, both directions."""
    base = pl.read_parquet(WORK / "baseline_e653.parquet")
    for test in ("g1_test", "g2_test"):
        b = summary(band(base.filter(pl.col("spec") == test)))
        print(f"experiment 653 on {test}: found {b['found']:.3f} paired {b['paired']:.3f}")
    for s in SEEDS:
        for test, own, other in (("g1_test", "g1", "g2"), ("g2_test", "g2", "g1")):
            out = {}
            for name in (own, other):
                tag = f"{name}_seed{s}"
                p = picked(tag)
                d = band(pl.read_parquet(WORK / f"cross_{tag}.parquet")).filter(
                    (pl.col("spec") == test)
                    & (pl.col("arm") == "net")
                    & (pl.col("on") == p["on"])
                    & (pl.col("boost") == p["boost"])
                    & (pl.col("cutoff") == p["cutoff"])
                )
                out[name] = summary(d)
            print(
                f"seed {s} test {test}: same-injector found {out[own]['found']:.3f} "
                f"cross {out[other]['found']:.3f} "
                f"drop {out[own]['found'] - out[other]['found']:+.3f}; "
                f"paired {out[own]['paired']:.3f} / {out[other]['paired']:.3f}"
            )


if __name__ == "__main__":
    {"injected": injected, "cross": cross}[sys.argv[1]]()
