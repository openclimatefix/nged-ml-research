# ruff: noqa: E501
"""Experiment 482: wind reference contrasts on the rows experiment 461 scored.

Reads the four stored wind references (research default 1 of experiments 339 and 340, research
default 2 of experiment 347, climatology of experiment 230, and a manual heuristic recomputed from
observed power as in experiment 303's stakeholder charts), all in fractions of the old effective
capacity, row-aligned with `e00347_s4_data.scored_frame()`. Converts each to MW with the old
capacity, joins them to the table-P rows of `member_plain` (p99.9 capacity) from experiment 468's
`rows_after_p_wind.parquet`, keeps the half-hours present in every arm, and writes the median
absolute error in points of the p99.9 capacity by lead day and by farm, and a month-block
bootstrap of `member_plain` minus each reference with the minimum detectable effect beside it.

Needs PyTorch only through the imported experiment modules.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
source <paths of ml-experiments/*/ on PYTHONPATH>
uv run --no-sync python ml-experiments/00482-*/e00482_wind_refs.py
```
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import altair as alt  # noqa: E402
import e00230_common as c230  # noqa: E402  # ty: ignore[unresolved-import]
import e00230_references as refs  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_common as c303  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_ens_final as ef  # noqa: E402  # ty: ignore[unresolved-import]
import e00340_common as c340  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import plotting.ocf_theme as ocf  # noqa: E402
import polars as pl  # noqa: E402
from e00468_part_a_refs import block_bootstrap  # noqa: E402  # ty: ignore[unresolved-import]
from studies.power import scan_power  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00482-best-models-performance-analysis-injected-shifts-and-wind-references"
    / "wind"
)
ROWS_468: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00468-best-models-performance-analysis-with-time-series-graphs"
    / "part_a"
    / "rows_after_p_wind.parquet"
)
Y2_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00347-wind-physics-hybrid-plant-then-mdn" / "s4" / "parts"
)
LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
Q50: Final[int] = list(c303.LEVELS).index(0.5)
LAGS: Final[list[int]] = [168 * k for k in range(1, 7)] + [8232, 8400, 8568, 8736, 8904, 9072, 9240]
NAMES: Final[dict[str, str]] = {
    "default1": "Wind research default 1 (superseded)",
    "default2": "Wind research default 2",
    "clim": "Climatology",
    "heur": "Manual heuristic",
}
COLOURS: Final[dict[str, str]] = {
    "Experiment 461 member_plain (p99.9)": ocf.BRAND_ORANGE,
    NAMES["default1"]: ocf.DATA_BLUE,
    NAMES["default2"]: ocf.DATA_PURPLE,
    NAMES["clim"]: ocf.DATA_GREEN,
    NAMES["heur"]: ocf.DATA_SKY,
}
WIDTH: Final[int] = 640


def manual_heuristic(*, sites: pl.DataFrame, cases: pl.DataFrame) -> np.ndarray:
    """Return the median of same-weekday observed fractions with lag longer than the lead, per case."""
    cap = sites.select("time_series_id", "effective_capacity_mw")
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", "time", "power")
        .collect()
        .join(cap, on="time_series_id")
        .select("time_series_id", "time", obs=pl.col("power") / pl.col("effective_capacity_mw"))
        .drop_nulls()
    )
    base = cases.select(*c303.KEY_COLUMNS, "lead_hours").with_row_index("i")
    cols = []
    for lag in LAGS:
        j = base.select(
            "i",
            "time_series_id",
            t=pl.col("valid_time") - pl.duration(hours=lag),
            lead="lead_hours",
        ).join(power.rename({"time": "t"}), on=["time_series_id", "t"], how="left")
        cols.append(
            j.sort("i").select(
                pl.when(pl.col("lead") < lag).then(pl.col("obs")).otherwise(None).alias(f"l{lag}")
            )
        )
    return (
        pl.concat(cols, how="horizontal")
        .select(m=pl.concat_list(pl.all()).list.drop_nulls().list.median())["m"]
        .to_numpy()
    )


def reference_table() -> pl.DataFrame:
    """Return one row per scored case with every reference's median in MW (old capacity)."""
    sites = c303.site_table()
    cases = ef.scored_cases(sites=sites)
    scored = data347.scored_frame()
    assert (scored["valid_time"].to_numpy() == cases["valid_time"].to_numpy()).all()
    assert (scored["site"].to_numpy() == cases["site"].to_numpy()).all()
    assert (
        scored["power_fcst_init_time"].to_numpy() == cases["power_fcst_init_time"].to_numpy()
    ).all()
    assert (scored["lead_hours"].to_numpy() == cases["lead_hours"].to_numpy()).all()
    n_rows = scored.height
    old_cap = {
        r["site"]: float(r["effective_capacity_mw"])
        for r in c230.site_table().iter_rows(named=True)
    }
    default1 = np.load(c340.FINAL_DIR / "default.npy")[:, :, Q50].mean(axis=0)
    default2 = np.stack([np.load(Y2_DIR / f"y2_h{s}.npy")[:, Q50] for s in range(5)]).mean(axis=0)
    assert default1.shape[0] == default2.shape[0] == n_rows
    clim_frame = refs.climatology(sites=sites, cases=cases.select(*c303.KEY_COLUMNS))
    clim = (
        cases.select(*c303.KEY_COLUMNS)
        .join(
            clim_frame.select(*c303.KEY_COLUMNS, "q50"),
            on=list(c303.KEY_COLUMNS),
            how="left",
            validate="1:1",
        )["q50"]
        .to_numpy()
    )
    heur = manual_heuristic(sites=sites, cases=cases)
    cap = scored["site"].replace_strict(old_cap, return_dtype=pl.Float64).to_numpy()
    return scored.select(
        site=pl.col("site").replace_strict(LABEL),
        init=pl.col("power_fcst_init_time"),
        valid=pl.col("valid_time"),
    ).with_columns(
        default1=pl.Series(default1 * cap),
        default2=pl.Series(default2 * cap),
        clim=pl.Series(clim * cap),
        heur=pl.Series(heur * cap),
    )


def joined(*, refs_table: pl.DataFrame) -> pl.DataFrame:
    """Return `member_plain`'s scored rows with absolute errors (points of p99.9 capacity) for all arms."""
    mine = pl.read_parquet(ROWS_468).filter(pl.col("arm") == "member_plain")
    out = mine.join(refs_table, on=["site", "init", "valid"], how="inner", validate="1:1")
    keep = pl.all_horizontal(*[pl.col(c).is_finite() for c in NAMES])
    kept = out.filter(keep)
    print(
        "rows member_plain", mine.height, "joined", out.height, "all references finite", kept.height
    )
    return kept.select(
        "site",
        "month",
        "lead_hours",
        "cap_mw",
        mine=(pl.col("q50") - pl.col("actual")).abs() / pl.col("cap_mw") * 100,
        **{k: (pl.col(k) - pl.col("actual")).abs() / pl.col("cap_mw") * 100 for k in NAMES},
    )


def main() -> None:
    """Write the wind reference error-by-lead chart and the contrast table."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = reference_table()
    table.write_parquet(OUT / "reference_medians_mw.parquet")
    d = joined(refs_table=table).with_columns(day=(pl.col("lead_hours") // 24).cast(pl.Int32) + 0.5)
    arms = {"mine": "Experiment 461 member_plain (p99.9)", **NAMES}
    long = (
        d.unpivot(index=["site", "day"], on=list(arms), variable_name="arm", value_name="ae")
        .group_by("arm", "site", "day")
        .agg(mae=pl.col("ae").mean(), n=pl.len())
        .with_columns(label=pl.col("arm").replace(arms))
        .group_by("label", "day")
        .agg(mae=pl.col("mae").mean(), n=pl.col("n").sum())
        .sort("label", "day")
    )
    long.write_csv(OUT / "wind_error_by_lead_with_references.csv")
    per_farm = (
        d.group_by("site")
        .agg(*[pl.col(k).mean().alias(k) for k in ("mine", *NAMES)], n=pl.len())
        .sort("site")
    )
    per_farm.write_csv(OUT / "wind_error_by_farm.csv")
    print(per_farm)
    chart = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:Q", title="Lead time (days ahead)", axis=alt.Axis(values=list(range(0, 11, 2)))
            ),
            y=alt.Y(
                "mae:Q",
                title="Mean absolute error of the median forecast (% of capacity; smaller is better)",
            ),
            color=alt.Color(
                "label:N",
                title=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=220,
            title=alt.TitleParams(
                [
                    "Farm A, B, and C: mean absolute error of the median forecast by lead day, same half-hours",
                    f"for every approach ({d.height:,} half-hours; the three farms weigh equally in each day's mean).",
                ],
                anchor="start",
            ),
        )
    )
    (OUT / "png").mkdir(exist_ok=True)
    chart.save(OUT / "wind_error_by_lead_with_references.svg")
    chart.save(OUT / "png" / "wind_error_by_lead_with_references.png", scale_factor=1.5)
    rng = np.random.default_rng(0)
    level = 1 - 0.05 / len(NAMES)
    rows = [block_bootstrap(d=d, ref=k, rng=rng, level=level) for k in NAMES]
    pl.DataFrame(rows).write_csv(OUT / "wind_contrasts.csv")
    print(f"interval level {level:.4f} (Bonferroni count {len(NAMES)})")
    print(pl.DataFrame(rows))
    band = d.filter(pl.col("lead_hours") <= 36)
    band_rows = [block_bootstrap(d=band, ref=k, rng=rng, level=level) for k in NAMES]
    pl.DataFrame(band_rows).write_csv(OUT / "wind_contrasts_24_36h.csv")
    print("24 to 36 hour band:")
    print(pl.DataFrame(band_rows))


if __name__ == "__main__":
    main()
