# ruff: noqa: E501
"""Score experiment 493: annual against quarterly refits of the pooled solar network.

Rows are experiment 310's scored-year runs that every compared arm holds. An arm's six seeds are
averaged quantile by quantile, the loss is computed per row on sorted quantiles, and a contrast is
the percentage change of the farm-equal mean score with the 12 scored months resampled in blocks.
The interval level is set by the Bonferroni count of the three planned contrasts (family level 5%).

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations \
        uv run --no-sync python ml-experiments/00493-*/e00493_score.py
"""

from pathlib import Path
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00430_score as s430  # ty: ignore[unresolved-import]
import e00493_common as common
import numpy as np
import polars as pl

BONFERRONI_COUNT: Final[int] = 3
FAMILY_ALPHA: Final[float] = 0.05
LEVEL_PCT: Final[float] = 100.0 * (1.0 - FAMILY_ALPHA / BONFERRONI_COUNT)
LO_PCT = (100.0 - LEVEL_PCT) / 2.0
HI_PCT = 100.0 - LO_PCT
"""Percentiles of the two-sided 98.33% interval (0.833 and 99.167)."""
LIMIT_PERCENT: Final[float] = 1.0
CONTROL_TOLERANCE_PERCENT: Final[float] = 0.01
FARM_LIMIT_PERCENT: Final[float] = 1.0
FIRST_QUARTER_INITS: Final[tuple[str, ...]] = ("2025-09", "2025-10", "2025-11")
PAIRS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("annual_g0_onepass", "q_g0_onepass"),
    "C2": ("annual_a2_member", "q_a2_member"),
    "C3": ("annual_g0_onepass", "q_a2_member"),
}
OUT_DIR: Final[Path] = common.SCORE_DIR
s430.LO_PCT, s430.HI_PCT = LO_PCT, HI_PCT


def load(*, name: str) -> pl.DataFrame:
    """Return an arm's six-seed average: `annual_<arm>` or `q_<arm>` (experiment 430's quarterly)."""
    if name.startswith("annual_"):
        folder = common.PRED_DIR / name
    else:
        folder = common.E430_PRED_DIR / name.removeprefix("q_")
    files = sorted(f for s in common.SEEDS for f in folder.glob(f"s{s}_*.parquet"))
    per_seed = 1 if name.startswith("annual_") else 4
    assert len(files) == per_seed * len(common.SEEDS), (name, len(files))
    return s430.average_seeds(frame=pl.concat([pl.read_parquet(f) for f in files]), arm=name)


def per_row_losses(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return per-row losses of every arm on the rows all of them hold, with the init month added."""
    aligned = s430.common_rows(frames=frames)
    return {
        n: s430.losses(frame=f).with_columns(init_month=pl.col("init_time").dt.strftime("%Y-%m"))
        for n, f in aligned.items()
    }


def contrast(*, a: pl.DataFrame, b: pl.DataFrame) -> dict[str, float]:
    """Return the percentage change of a against b over the rows given, with its interval."""
    va, _ = s430.farm_equal_by_month(rows=a, metric="score")
    vb, _ = s430.farm_equal_by_month(rows=b, metric="score")
    point, lo, hi = s430.block_bootstrap(per_month_a=va, per_month_b=vb)
    return {
        "score_a": float(va.mean()),
        "score_b": float(vb.mean()),
        "pct": point,
        "lo": lo,
        "hi": hi,
    }


def seed_mde(*, name: str) -> dict[str, float]:
    """Return the minimum detectable effect (percent) from the seed spread of an arm's full-year score."""
    if name.startswith("annual_"):
        folder = common.PRED_DIR / name
    else:
        folder = common.E430_PRED_DIR / name.removeprefix("q_")
    frame = pl.concat(
        [pl.read_parquet(f) for s in common.SEEDS for f in folder.glob(f"s{s}_*.parquet")]
    )
    frame = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in s430.QCOLS]))
    keys = None
    per_seed = {s: frame.filter(pl.col("seed") == s) for s in common.SEEDS}
    for s in common.SEEDS:
        k = per_seed[s].select(s430.KEYS)
        keys = k if keys is None else keys.join(k, on=s430.KEYS, how="inner")
    assert keys is not None
    scores = []
    for s in common.SEEDS:
        rows = s430.losses(
            frame=keys.join(per_seed[s], on=s430.KEYS, how="left").with_columns(
                arm=pl.lit(name), setting=pl.lit("own_history")
            )
        )
        values, _ = s430.farm_equal_by_month(rows=rows, metric="score")
        scores.append(float(values.mean()))
    sd, mean = float(np.std(scores, ddof=1)), float(np.mean(scores))
    return {
        "sd": sd,
        "mean": mean,
        "mde_pct": 100.0 * 1.96 * np.sqrt(2.0) * sd / np.sqrt(len(scores)) / mean,
    }


def main() -> None:
    """Print the minimum detectable effects, control, contrasts, per-farm, per-quarter, and tails."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(14)
    names = ["annual_g0_onepass", "q_g0_onepass", "annual_a2_member", "q_a2_member"]
    rows = per_row_losses(frames={n: load(name=n) for n in names})
    print(
        f"interval level {LEVEL_PCT:.2f}% (Bonferroni count {BONFERRONI_COUNT}); rows {rows[names[0]].height}"
    )

    mde = {n: seed_mde(name=n) for n in names}
    for n, m in mde.items():
        print("MDE", n, m)
    limit = max(LIMIT_PERCENT, *(m["mde_pct"] for m in mde.values()))
    print(f"limit {limit:.3f}% (1% or the largest minimum detectable effect)")

    # Control: the first quarter's annual and quarterly forecasts come from the same fits.
    for annual, quarterly in (
        ("annual_g0_onepass", "q_g0_onepass"),
        ("annual_a2_member", "q_a2_member"),
    ):
        first = {
            n: rows[n].filter(pl.col("init_month").is_in(FIRST_QUARTER_INITS))
            for n in (annual, quarterly)
        }
        c = contrast(a=first[annual], b=first[quarterly])
        status = "PASSED" if abs(c["pct"]) <= CONTROL_TOLERANCE_PERCENT else "MISSED"
        merged = (
            first[annual]
            .select(*s430.KEYS, a=pl.col("score"))
            .join(first[quarterly].select(*s430.KEYS, b=pl.col("score")), on=s430.KEYS)
        )
        max_diff = float((merged["a"] - merged["b"]).abs().max())  # ty: ignore[invalid-argument-type]
        print(f"control max |row score difference| {max_diff:.6f}")
        print(f"CONTROL {annual} vs {quarterly}, first quarter: {c['pct']:+.5f}% {status}")

    out = []
    for cid, (a, b) in PAIRS.items():
        sets = {"all": c310.BAND_ORDER, "d1": ("d1",)}
        for set_name, bands in sets.items():
            c = contrast(
                a=rows[a].filter(pl.col("band").is_in(bands)),
                b=rows[b].filter(pl.col("band").is_in(bands)),
            )
            out.append({"id": cid, "a": a, "b": b, "cut": f"bands_{set_name}", **c})
        for farm in c310.FARMS:
            c = contrast(
                a=rows[a].filter(pl.col("time_series_id") == farm),
                b=rows[b].filter(pl.col("time_series_id") == farm),
            )
            out.append({"id": cid, "a": a, "b": b, "cut": f"farm_{farm}", **c})
        months = sorted(rows[a]["init_month"].unique().to_list())
        assert len(months) == 12, months
        for k in range(4):
            quarter = months[3 * k : 3 * k + 3]
            c = contrast(
                a=rows[a].filter(pl.col("init_month").is_in(quarter)),
                b=rows[b].filter(pl.col("init_month").is_in(quarter)),
            )
            out.append({"id": cid, "a": a, "b": b, "cut": f"quarter_{k + 1}", **c})
    table = pl.DataFrame(out)
    print(table)
    table.write_csv(OUT_DIR / "contrasts.csv")

    head = table.filter(pl.col("cut") == "bands_all")
    farms = table.filter(pl.col("cut").str.starts_with("farm_"))
    for r in head.iter_rows(named=True):
        worst = float(farms.filter(pl.col("id") == r["id"])["pct"].max())  # ty: ignore[invalid-argument-type]
        equivalent = r["hi"] < limit and worst <= FARM_LIMIT_PERCENT
        verdict = (
            "equivalent"
            if equivalent
            else ("inconclusive" if r["lo"] < limit <= r["hi"] else "not equivalent")
        )
        print(
            f"{r['id']}: {r['pct']:+.3f}% ({r['lo']:+.3f} to {r['hi']:+.3f}); worst farm {worst:+.3f}% -> {verdict}"
        )

    absolute = []
    for n in names:
        rec: dict[str, float | str] = {"arm": n}
        for metric in (
            "score",
            "pin01",
            "pin05",
            "pin95",
            "pin99",
            "cover01",
            "cover05",
            "cover95",
            "cover99",
            "median_minus_mean",
            "ae_mean",
        ):
            values, _ = s430.farm_equal_by_month(rows=rows[n], metric=metric)
            rec[metric] = float(values.mean())
        absolute.append(rec)
    absolute_table = pl.DataFrame(absolute)
    print(absolute_table)
    absolute_table.write_csv(OUT_DIR / "absolute.csv")


if __name__ == "__main__":
    main()
