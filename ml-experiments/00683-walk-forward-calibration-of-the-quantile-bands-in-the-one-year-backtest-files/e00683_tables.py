"""Tables for experiment 683 from the per-day sums that `e00683_calibrate.py` wrote (CPU only).

PYTHONPATH=ml-experiments/00668-* uv run --no-sync python ml-experiments/00683-*/e00683_tables.py

Prints every number the write-up quotes and writes CSV tables to `<data>/00683-*/tables/`.
Coverage and the one-sided misses are in percent of rows; CRPS and pinball loss are in points of
capacity. Intervals are day-block bootstraps over forecast-origin days, with the picks shared
between arms. Wind and solar are never split by series and month together.
"""

from typing import Final

import numpy as np
import polars as pl
from e00683_calibrate import BAND_LABELS, EXPERIMENT_DIR, FIRST_SCORED_MONTH

N_BOOT: Final[int] = 2000
FILES: Final[tuple[str, ...]] = (
    "primaries_NRA",
    "primaries_prevailing",
    "bsps",
    "gsps",
    "wind",
    "solar",
)
BEFORE: Final[str] = "2026-07"
"""Origin months before this one are the months that decide the verdict."""
SUMS: Final[tuple[str, ...]] = (
    "crps",
    "mae",
    "coverage",
    "below",
    "above",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "hit01",
    "hit05",
    "hit95",
    "hit99",
    "width",
)
ARM_NAMES: Final[dict[int, str]] = {0: "stored", 1: "fixed 1.0934", 2: "scale", 3: "scale + shift"}
TABLES: Final = EXPERIMENT_DIR / "tables"


def load(*, file: str) -> pl.DataFrame:
    """Return the per-day sums of one file, from the first scored month on."""
    return pl.read_parquet(EXPERIMENT_DIR / "daily" / f"{file}.parquet").filter(
        pl.col("origin_month") >= FIRST_SCORED_MONTH
    )


def period(*, frame: pl.DataFrame, which: str) -> pl.DataFrame:
    """Return the rows of the months before July (`dev`) or from July on (`final`)."""
    month = pl.col("origin_month")
    return frame.filter(month < BEFORE if which == "dev" else month >= BEFORE)


def means(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the row-weighted mean of every metric in each group."""
    return (
        frame.group_by(by)
        .agg(pl.col("n").sum(), *[pl.col(f"sum_{c}").sum() for c in SUMS])
        .with_columns(*[(pl.col(f"sum_{c}") / pl.col("n")).alias(c) for c in SUMS])
        .drop([f"sum_{c}" for c in SUMS])
        .sort(by)
    )


def boot_contrast(*, frame: pl.DataFrame, arm: int, band: int | None) -> dict[str, float]:
    """Return the change in CRPS and the gain in |coverage - 90| of `arm`, with 95% intervals."""
    sub = frame if band is None else frame.filter(pl.col("band") == band)
    by_day = (
        sub.group_by("arm", "origin_day")
        .agg(pl.col("n").sum(), pl.col("sum_crps").sum(), pl.col("sum_coverage").sum())
        .sort("origin_day")
    )
    days = by_day["origin_day"].unique().sort().to_numpy()
    out: dict[str, np.ndarray] = {}
    for a in (0, arm):
        part = by_day.filter(pl.col("arm") == a).sort("origin_day")
        aligned = pl.DataFrame({"origin_day": days}).join(part, on="origin_day", how="left")
        out[f"n{a}"] = aligned["n"].fill_null(0).to_numpy().astype(float)
        out[f"crps{a}"] = aligned["sum_crps"].fill_null(0).to_numpy()
        out[f"cov{a}"] = aligned["sum_coverage"].fill_null(0).to_numpy()
    rng = np.random.default_rng(683)
    pick = rng.integers(0, len(days), size=(N_BOOT, len(days)))

    def stat(idx: np.ndarray) -> tuple[float, float]:
        crps = {a: out[f"crps{a}"][idx].sum() / out[f"n{a}"][idx].sum() for a in (0, arm)}
        gap = {a: abs(out[f"cov{a}"][idx].sum() / out[f"n{a}"][idx].sum() - 90.0) for a in (0, arm)}
        return crps[arm] - crps[0], gap[0] - gap[arm]

    point = stat(np.arange(len(days)))
    draws = np.array([stat(p) for p in pick])
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "crps_change": point[0],
        "crps_lo": lo[0],
        "crps_hi": hi[0],
        "gap_reduction_pts": point[1],
        "gap_lo": lo[1],
        "gap_hi": hi[1],
    }


def main() -> None:  # noqa: PLR0915
    """Write and print the tables."""
    TABLES.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(40)
    all_cells: list[pl.DataFrame] = []
    contrast_rows: list[dict] = []
    for file in FILES:
        frame = load(file=file)
        for which in ("dev", "final"):
            sub = period(frame=frame, which=which)
            cell = means(frame=sub, by=["arm", "band"]).with_columns(
                file=pl.lit(file), period=pl.lit(which)
            )
            all_cells.append(cell)
            contrast_rows.extend(
                {"file": file, "period": which, "arm": arm, "band": band}
                | boot_contrast(frame=sub, arm=arm, band=band)
                for arm in (1, 2, 3)
                for band in (None, 0, 1, 2, 3)
            )
    cells = pl.concat(all_cells)
    cells.write_csv(TABLES / "cells.csv")
    contrasts = pl.DataFrame(contrast_rows)
    contrasts.write_csv(TABLES / "contrasts.csv")
    for which in ("dev", "final"):
        print(f"\n=== coverage (%) and CRPS by file, band, arm; {which} ===")
        view = cells.filter(pl.col("period") == which).select(
            "file", "band", "arm", pl.col("coverage").round(1), pl.col("crps").round(3), "n"
        )
        print(view.pivot(on="arm", index=["file", "band"], values=["coverage", "crps"]))
    print("\n=== contrasts against arm 0 (band null = all bands) ===")
    print(
        contrasts.with_columns(pl.col(pl.Float64).round(3)).filter(pl.col("band").is_null()),
    )
    print("\n=== P1/P2 cells, dev period, arm 2 ===")
    arm2 = cells.filter((pl.col("period") == "dev") & (pl.col("arm") == 2)).join(
        cells.filter((pl.col("period") == "dev") & (pl.col("arm") == 0)).select(
            "file", "band", crps0="crps"
        ),
        on=["file", "band"],
    )
    arm2 = arm2.with_columns(
        cov_pass=(pl.col("coverage") / 100 - 0.9).abs() <= 0.03,
        crps_pass=(pl.col("crps") - pl.col("crps0")) <= 0.05,
    )
    print(arm2.select("file", "band", pl.col("coverage").round(1), "cov_pass", "crps_pass"))
    # primaries are two files; the verdict cells are five files, so Primaries are NRA here
    five = arm2.filter(pl.col("file") != "primaries_prevailing")
    print(
        "failing cells (NRA as the Primaries):",
        five.filter(~(pl.col("cov_pass") & pl.col("crps_pass"))).height,
        "of",
        five.height,
    )

    # monthly coverage (the stop rule) and one-sided misses
    monthly = []
    for file in FILES:
        frame = load(file=file)
        m = means(frame=frame, by=["arm", "origin_month"]).with_columns(file=pl.lit(file))
        monthly.append(m)
        if file in ("wind", "solar"):
            continue
    monthly = pl.concat(monthly)
    monthly.write_csv(TABLES / "monthly.csv")
    print("\n=== monthly pooled coverage, all bands ===")
    print(
        monthly.select("file", "arm", "origin_month", pl.col("coverage").round(1)).pivot(
            on="origin_month", index=["file", "arm"], values="coverage"
        )
    )
    print("\n=== monthly below/above q05/q95 shares, arm 0 and 2 ===")
    print(
        monthly.filter(pl.col("arm").is_in([0, 2]))
        .select("file", "arm", "origin_month", pl.col("below").round(1), pl.col("above").round(1))
        .pivot(on="origin_month", index=["file", "arm"], values=["below", "above"])
    )

    # primaries 13 and 16 in the two months with and without
    nra = load(file="primaries_NRA")
    for label, drop in (
        ("all", False),
        ("without D13 and D16 in 2026-07 and 2026-08", True),
    ):
        sub = nra
        if drop:
            sub = nra.filter(
                ~(
                    pl.col("time_series_id").is_in([13, 16])
                    & pl.col("origin_month").is_in(["2026-07", "2026-08"])
                )
            )
        print(f"\n=== NRA Primaries monthly coverage, {label} ===")
        print(
            means(frame=sub, by=["arm", "origin_month"])
            .select("arm", "origin_month", pl.col("coverage").round(1))
            .pivot(on="origin_month", index="arm", values="coverage")
        )
    for tid in (13, 16):
        print(f"\n=== D{tid} NRA monthly coverage ===")
        print(
            means(frame=nra.filter(pl.col("time_series_id") == tid), by=["arm", "origin_month"])
            .select("arm", "origin_month", pl.col("coverage").round(1))
            .pivot(on="origin_month", index="arm", values="coverage")
        )

    # tails
    tails = []
    for file in FILES:
        frame = period(frame=load(file=file), which="dev")
        tails.append(means(frame=frame, by=["arm"]).with_columns(file=pl.lit(file)))
    tails = pl.concat(tails)
    tails.write_csv(TABLES / "tails_dev.csv")
    print("\n=== tails, dev period, all bands: hit rate (% of y <= q) and pinball ===")
    print(
        tails.select(
            "file",
            "arm",
            *[pl.col(f"hit{t}").round(2) for t in ("01", "05", "95", "99")],
            *[pl.col(f"pin{t}").round(3) for t in ("01", "05", "95", "99")],
        )
    )

    # per-series coverage for substations (not for generators)
    for file in ("primaries_NRA", "bsps", "gsps"):
        by = means(frame=period(frame=load(file=file), which="dev"), by=["arm", "time_series_id"])
        by.write_csv(TABLES / f"series_{file}.csv")
        print(f"\n=== {file}: dev coverage by series ===")
        print(
            by.select("arm", "time_series_id", pl.col("coverage").round(1)).pivot(
                on="arm", index="time_series_id", values="coverage"
            )
        )
    for file in ("wind", "solar"):
        by = means(frame=period(frame=load(file=file), which="dev"), by=["arm", "time_series_id"])
        letters = "ABCDEF"
        ids = sorted(by["time_series_id"].unique().to_list())
        by = by.with_columns(
            pl.col("time_series_id")
            .replace_strict({t: f"{file.capitalize()} {letters[i]}" for i, t in enumerate(ids)})
            .alias("series")
        ).drop("time_series_id")
        print(f"\n=== {file}: dev coverage by series (pooled over months) ===")
        print(
            by.select("arm", "series", pl.col("coverage").round(1)).pivot(
                on="arm", index="series", values="coverage"
            )
        )
    print("\nbands:", BAND_LABELS)


if __name__ == "__main__":
    main()
