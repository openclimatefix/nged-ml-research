"""Extra year-level numbers for experiment 473: idle texture, last active period, coverage."""

import json
import os
from pathlib import Path
from typing import Final

import polars as pl

DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"])
OUT: Final[Path] = DATA / "ml-experiments" / "00473-three-generators-data-look"
LABEL_TO_TYPE: Final[dict[str, str]] = json.loads((OUT / "label_map_PRIVATE.json").read_text())
ACTIVE: Final[float] = 0.05


def main() -> None:  # noqa: D103
    meta = pl.read_parquet(DATA / "NGED" / "metadata.parquet")
    power = pl.scan_delta(str(DATA / "NGED" / "cleaned_power_time_series.delta"))
    result: dict[str, dict] = {}
    cover = []
    for label, kind in LABEL_TO_TYPE.items():
        tsid = meta.filter(pl.col("time_series_type") == kind)["time_series_id"][0]
        d = (
            power.filter(pl.col("time_series_id") == tsid)
            .select("time", "power")
            .sort("time")
            .collect()
        )
        scale = float(d["power"].abs().max())  # ty: ignore[invalid-argument-type]
        d = d.with_columns(
            (pl.col("power") / scale).alias("f"),
            (pl.col("power").abs() / scale >= ACTIVE).alias("active"),
        )
        r: dict = {}
        by_year = (
            d.group_by(pl.col("time").dt.year().alias("year"))
            .agg(
                pl.len().alias("n"),
                pl.col("active").sum().alias("active_half_hours"),
                (pl.col("f") == 0).sum().alias("exact_zero"),
                pl.col("f").abs().max().alias("max_abs_fraction"),
                pl.col("f").filter(~pl.col("active")).median().alias("idle_median"),
                pl.col("f")
                .filter(~pl.col("active") & (pl.col("f") != 0))
                .median()
                .alias("idle_median_nonzero"),
                (pl.col("f").filter(~pl.col("active")) == 0).mean().alias("idle_exact_zero_share"),
            )
            .sort("year")
        )
        by_year.write_csv(OUT / f"by_year_{label}.csv")
        act = d.filter(pl.col("active"))
        r["last_active_month"] = str(act["time"].max())[:7]
        r["first_active_month"] = str(act["time"].min())[:7]
        r["active_half_hours_total"] = act.height
        # After the last active half-hour: length of the non-operating tail and its idle texture.
        tail = d.filter(pl.col("time") > act["time"].max())
        r["tail_half_hours_after_last_active"] = tail.height
        r["tail_max_abs_fraction"] = float(tail["f"].abs().max()) if tail.height else None  # ty: ignore[invalid-argument-type]
        r["tail_exact_zero_share"] = float((tail["f"] == 0).mean()) if tail.height else None  # ty: ignore[invalid-argument-type]
        # Valid half-hours at least 12 months before the end of the last active period.
        r["valid_half_hours_up_to_last_active"] = int(
            d.filter(pl.col("time") <= act["time"].max()).height
        )
        # Idle texture: exact zeros inside the longest zero run versus outside.
        z = d.with_columns(((pl.col("f") != 0).cum_sum()).alias("g"))
        runs = (
            z.filter(pl.col("f") == 0)
            .group_by("g")
            .agg(pl.len().alias("len"), pl.col("time").min().alias("start"))
            .sort("len", descending=True)
        )
        longest = runs.head(1)
        r["longest_zero_run_half_hours"] = int(longest["len"][0])
        r["longest_zero_run_start_year"] = int(longest["start"][0].year)
        r["longest_zero_run_start_to_next_row_gap_note"] = "see by_year table"
        outside = d.filter(pl.col("f") != 0)
        r["nonzero_share_of_inactive_outside_longest_run"] = float((~outside["active"]).mean())  # ty: ignore[invalid-argument-type]
        # Coverage per calendar month (valid half-hours / 1440 on average).
        cov = (
            d.group_by(pl.col("time").dt.truncate("1mo").alias("month"))
            .agg(pl.len().alias("n"))
            .with_columns(pl.lit(label).alias("series"))
        )
        cover.append(cov)
        # Ratio of monthly active: winter (Oct-Mar) versus summer (Apr-Sep).
        m = d.with_columns(pl.col("time").dt.month().alias("m"))
        r["active_share_oct_to_mar"] = float(
            m.filter(pl.col("m").is_in([10, 11, 12, 1, 2, 3]))["active"].mean()  # ty: ignore[invalid-argument-type]
        )
        r["active_share_apr_to_sep"] = float(
            m.filter(~pl.col("m").is_in([10, 11, 12, 1, 2, 3]))["active"].mean()  # ty: ignore[invalid-argument-type]
        )
        result[label] = r
    pl.concat(cover).write_csv(OUT / "coverage.csv")
    (OUT / "extra.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1))
    for label in LABEL_TO_TYPE:
        print(label)
        print(pl.read_csv(OUT / f"by_year_{label}.csv"))


if __name__ == "__main__":
    main()
