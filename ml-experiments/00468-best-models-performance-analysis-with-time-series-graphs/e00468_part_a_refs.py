"""Experiment 468, Part A: solar contrasts against the stored references, on the same rows.

Joins the table-P rows of experiment 461's solar arms (Solar A and B, old capacity rule) from
`e00468_part_a.py` to experiment 310's stored per-row median errors for its blend (the metered
solar research default 1), its XGBoost model, climatology, and the manual heuristic (setting
`own_history`), keeping only half-hours present in every arm. Writes error by lead day and a
month-block bootstrap of each contrast of 461's `a2_member` minus each reference, in percent of
capacity (median error) with the minimum detectable effect beside it.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
uv run --no-sync python ml-experiments/00468-*/e00468_part_a_refs.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00468-best-models-performance-analysis-with-time-series-graphs" / "part_a"
)
R310: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring" / "row_losses.parquet"
NAMES: Final[dict[str, str]] = {
    "blend": "Metered solar research default 1 (experiment 310 blend)",
    "xgb": "XGBoost alone (experiment 310)",
    "floor_climatology": "Climatology",
    "floor_heuristic": "Manual heuristic",
}
DRAWS: Final[int] = 5000
WIDTH: Final[int] = 640
COLOURS: Final[dict[str, str]] = {
    "Experiment 461 a2_member": ocf.BRAND_ORANGE,
    NAMES["blend"]: ocf.DATA_BLUE,
    NAMES["xgb"]: ocf.DATA_PURPLE,
    "Climatology": ocf.DATA_GREEN,
    "Manual heuristic": ocf.DATA_SKY,
}


def joined() -> pl.DataFrame:
    """Return one row per (series, valid time, init) with the median error of every arm."""
    mine = pl.read_parquet(OUT / "rows_after_p_solar.parquet").filter(pl.col("arm") == "a2_member")
    mine = mine.select(
        site=pl.col("site"),
        farm=pl.col("site").str.replace("Solar ", ""),
        time=pl.col("valid"),
        init_time=pl.col("init"),
        lead_hours="lead_hours",
        cap_mw="cap_mw",
        mine=(pl.col("q50") - pl.col("actual")).abs() / pl.col("cap_mw") * 100,
        month=pl.col("valid").dt.strftime("%Y-%m"),
    )
    ref = (
        pl.scan_parquet(R310)
        .filter((pl.col("setting") == "own_history") & pl.col("arm").is_in(list(NAMES)))
        .select("farm", "time", "init_time", "arm", pl.col("ae_median"))
        .collect()
        .pivot(on="arm", index=["farm", "time", "init_time"], values="ae_median")
    )
    out = mine.join(ref, on=["farm", "time", "init_time"], how="inner").drop_nulls()
    print("rows mine", mine.height, "after join with all four references", out.height)
    return out


def block_bootstrap(*, d: pl.DataFrame, ref: str, rng: np.random.Generator, level: float) -> dict:
    """Return the contrast (mine minus `ref`), its month-block interval, and the detectable effect."""
    by = d.group_by("month").agg(
        s=(pl.col("mine") - pl.col(ref)).sum(), n=pl.len()
    ).sort("month")
    s, n = by["s"].to_numpy(), by["n"].to_numpy().astype(float)
    draws = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.integers(0, len(n), size=len(n))
        draws[i] = s[pick].sum() / n[pick].sum()
    lo, hi = np.percentile(draws, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    # minimum detectable effect: 2.8 standard errors (80% power at the 5% level)
    mde = 2.8 * float(draws.std())
    return dict(
        reference=ref, contrast_points=float(s.sum() / n.sum()), low=float(lo), high=float(hi),
        mde_points=mde, n_rows=int(n.sum()), n_months=len(n),
        mine_mae=float(d["mine"].mean()), ref_mae=float(d[ref].mean()),
    )  # fmt: skip


def main() -> None:
    """Write the reference error-by-lead chart and the contrast table."""
    d = joined()
    d = d.with_columns(day=(pl.col("lead_hours") // 24).cast(pl.Int32) + 0.5)
    arms = {"mine": "Experiment 461 a2_member", **NAMES}
    long = (
        d.unpivot(index=["site", "day"], on=list(arms), variable_name="arm", value_name="ae")
        .group_by("arm", "site", "day")
        .agg(mae=pl.col("ae").mean(), n=pl.len())
        .with_columns(label=pl.col("arm").replace(arms))
        .group_by("label", "day")
        .agg(mae=pl.col("mae").mean(), n=pl.col("n").sum())
        .sort("label", "day")
    )
    long.write_csv(OUT / "solar_error_by_lead_with_references.csv")
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("day:Q", title="Lead time (days ahead)", axis=alt.Axis(values=list(range(0, 11, 2)))),
            y=alt.Y("mae:Q", title="Median error (% of capacity; smaller is better)"),
            color=alt.Color(
                "label:N", title=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=220,
            title=alt.TitleParams(
                [
                    "Solar A and B: median absolute error by lead day on the same half-hours for every",
                    f"approach ({d.height} half-hours, equal weight to the two farms by row count).",
                ],
                anchor="start",
            ),
        )
    )
    (OUT / "png").mkdir(exist_ok=True)
    chart.save(OUT / "solar_error_by_lead_with_references.svg")
    chart.save(OUT / "png" / "solar_error_by_lead_with_references.png", scale_factor=1.5)
    rng = np.random.default_rng(0)
    level = 1 - 0.05 / len(NAMES)  # Bonferroni across the four contrasts shown
    rows = [block_bootstrap(d=d, ref=k, rng=rng, level=level) for k in NAMES]
    pl.DataFrame(rows).write_csv(OUT / "solar_contrasts.csv")
    print(f"interval level {level:.4f} (Bonferroni count {len(NAMES)})")
    print(pl.DataFrame(rows))
    print("12 to 36 hour band only:")
    band = d.filter(pl.col("lead_hours") <= 36)
    print(pl.DataFrame([block_bootstrap(d=band, ref=k, rng=rng, level=level) for k in NAMES]))


if __name__ == "__main__":
    main()
