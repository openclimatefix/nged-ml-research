"""Stage 1 data look at three metered generators, labelled X, Y, and Z.

Reads the raw-with-flags cleaned Delta table, runs the data-validation checklist, describes each
series' regime, and relates it to the system price and national demand. Writes private tables to the
experiment's data folder and anonymised aggregate tables for the write-up. The mapping from a label
to a metadata row stays in this private run: nothing printed or written outside the data folder
carries a fuel type, a capacity in MW, or an identifier.
"""

import json
import os
from pathlib import Path
from typing import Final

import polars as pl

DATA: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"])
OUT: Final[Path] = DATA / "ml-experiments" / "00473-three-generators-data-look"
LABEL_TO_TYPE: Final[dict[str, str]] = json.loads((OUT / "label_map_PRIVATE.json").read_text())
ACTIVE_FRACTION: Final[float] = 0.05
"""A half-hour is 'active' when the absolute output is at least this fraction of the maximum."""
HALF_HOUR_US: Final[int] = 30 * 60 * 1_000_000


def _runs(*, values: pl.Series) -> pl.DataFrame:
    """Run-length encode a series, keeping the value and the start index."""
    frame = pl.DataFrame({"v": values}).with_row_index("i")
    return (
        frame.with_columns(
            (pl.col("v") != pl.col("v").shift(1)).fill_null(True).cum_sum().alias("r")
        )
        .group_by("r", maintain_order=True)
        .agg(pl.col("v").first(), pl.col("i").first().alias("start"), pl.len().alias("length"))
    )


def main() -> None:  # noqa: PLR0915
    """Run the stage 1 checks and write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pl.read_parquet(DATA / "NGED" / "metadata.parquet")
    power = pl.scan_delta(str(DATA / "NGED" / "cleaned_power_time_series.delta"))
    price = (
        pl.read_parquet(
            DATA
            / "ml-experiments/00010-elexon-price-lags-stage1a-demand/prices/market_index.parquet"
        )
        .filter((pl.col("dataProvider") == "APXMIDP") & (pl.col("volume") > 0))
        .unique(subset="time")
        .select("time", pl.col("price").alias("price_gbp_mwh"))
    )
    demand = (
        pl.concat(
            [
                pl.read_csv(p, schema_overrides={"SETTLEMENT_DATE": pl.String})
                for p in sorted(
                    (
                        DATA
                        / "ml-experiments/00185-dp-price-dfs-national-signals-past-demand/public"
                    ).glob("demanddata_*.csv")
                )
            ],
            how="diagonal_relaxed",
        )
        .with_columns(
            (
                pl.coalesce(
                    pl.col("SETTLEMENT_DATE").str.to_date("%Y-%m-%d", strict=False),
                    pl.col("SETTLEMENT_DATE").str.to_date("%d-%b-%Y", strict=False),
                )
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("Europe/London")
                .dt.convert_time_zone("UTC")
                + pl.duration(minutes=30 * (pl.col("SETTLEMENT_PERIOD") - 1))
            ).alias("time")
        )
        .select(
            "time",
            pl.col("ND").alias("national_demand_mw"),
            pl.col("EMBEDDED_WIND_GENERATION").alias("embedded_wind_mw"),
        )
        .unique(subset="time")
    )
    report: dict[str, dict] = {}
    for label, kind in LABEL_TO_TYPE.items():
        row = meta.filter(pl.col("time_series_type") == kind)
        assert row.height == 1, kind
        tsid = row["time_series_id"][0]
        raw = (
            power.filter(pl.col("time_series_id") == tsid)
            .select("time", "power", "drop_reason")
            .sort("time")
            .collect()
        )
        r: dict = {"units": str(row["units"][0])}
        r["rows"] = raw.height
        r["first"] = str(raw["time"].min())
        r["last"] = str(raw["time"].max())
        # Duplicates, nulls, NaNs, timestamp grid.
        r["duplicate_keys"] = raw.height - raw["time"].n_unique()
        r["null_power"] = int(raw["power"].null_count())
        r["nan_power"] = int(raw["power"].is_nan().sum())
        r["off_grid_timestamps"] = int(
            raw.filter(
                (pl.col("time").dt.minute() % 30 != 0) | (pl.col("time").dt.second() != 0)
            ).height
        )
        r["time_zone"] = str(raw.schema["time"])
        r["flagged_by_reason"] = {
            str(k): int(v)
            for k, v in raw.filter(pl.col("drop_reason").is_not_null())
            .group_by("drop_reason")
            .len()
            .iter_rows()
        }
        r["flagged_exact_zero"] = int(
            raw.filter(pl.col("drop_reason").is_not_null() & (pl.col("power") == 0)).height
        )
        valid = (
            raw.filter(pl.col("power").is_not_null() & pl.col("power").is_not_nan())
            .unique(subset="time")
            .sort("time")
        )
        r["valid_half_hours_including_flagged"] = valid.height
        r["valid_half_hours_unflagged"] = int(valid.filter(pl.col("drop_reason").is_null()).height)
        # Gaps.
        dt = valid["time"].diff().dt.total_microseconds()
        gaps = dt.filter(dt > HALF_HOUR_US)
        r["gap_count"] = int(gaps.len())
        r["gaps_over_1h"] = int((gaps > 2 * HALF_HOUR_US).sum())
        r["gaps_over_1d"] = int((gaps > 48 * HALF_HOUR_US).sum())
        r["gaps_over_7d"] = int((gaps > 7 * 48 * HALF_HOUR_US).sum())
        r["longest_gap_days"] = float(gaps.max() / HALF_HOUR_US / 48) if gaps.len() else 0.0  # ty: ignore[unsupported-operator]
        r["missing_half_hours_in_gaps"] = int((gaps // HALF_HOUR_US - 1).sum())
        span = (valid["time"].max() - valid["time"].min()).total_seconds() / 1800 + 1  # ty: ignore[unsupported-operator]
        r["half_hours_in_span"] = int(span)
        # Values, in capacity fractions of the maximum absolute output.
        v = valid["power"]
        scale = float(v.abs().max())  # ty: ignore[invalid-argument-type]
        r["max_abs_mw_PRIVATE"] = scale
        f = (v / scale).alias("f")
        valid = valid.with_columns(
            f, pl.col("time").dt.convert_time_zone("Europe/London").alias("local")
        )
        fv = valid["f"]
        r["share_negative"] = float((fv < 0).mean())  # ty: ignore[invalid-argument-type]
        r["share_exact_zero"] = float((fv == 0).mean())  # ty: ignore[invalid-argument-type]
        r["share_positive"] = float((fv > 0).mean())  # ty: ignore[invalid-argument-type]
        r["fraction_quantiles"] = {
            str(q): float(fv.quantile(q))  # ty: ignore[invalid-argument-type]
            for q in (0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1)
        }
        top = valid.group_by("f").len().sort("len", descending=True).head(8)
        r["most_common_values_fraction"] = [[round(a, 4), int(b)] for a, b in top.iter_rows()]
        r["distinct_values"] = int(fv.n_unique())
        # Implausible values: large half-hour changes and isolated single-half-hour spikes.
        d1 = valid["f"].diff().abs()
        mad = float(d1.median())  # ty: ignore[invalid-argument-type]
        r["half_hour_change_median_abs"] = mad
        r["half_hour_changes_over_0.5_cap"] = int((d1 > 0.5).sum())
        r["isolated_spikes_over_0.5_cap"] = int(
            (
                ((valid["f"] - valid["f"].shift(1)).abs() > 0.5)
                & ((valid["f"] - valid["f"].shift(-1)).abs() > 0.5)
                & ((valid["f"].shift(1) - valid["f"].shift(-1)).abs() < 0.1)
            ).sum()
        )
        # Stuck runs.
        runs = _runs(values=fv)
        stuck = runs.sort("length", descending=True)
        r["longest_runs"] = [
            [round(a, 4), int(b)] for a, b in stuck.select("v", "length").head(6).iter_rows()
        ]
        nz = stuck.filter(pl.col("v") != 0)
        r["longest_nonzero_runs"] = [
            [round(a, 4), int(b)] for a, b in nz.select("v", "length").head(5).iter_rows()
        ]
        r["nonzero_runs_over_6_half_hours"] = int(nz.filter(pl.col("length") > 6).height)
        r["zero_runs_over_48_half_hours"] = int(
            stuck.filter((pl.col("v") == 0) & (pl.col("length") > 48)).height
        )
        r["zero_runs_over_1000_half_hours"] = int(
            stuck.filter((pl.col("v") == 0) & (pl.col("length") > 1000)).height
        )
        r["half_hours_in_zero_runs_over_48"] = int(
            stuck.filter((pl.col("v") == 0) & (pl.col("length") > 48))["length"].sum()
        )
        # Monthly levels.
        monthly = (
            valid.with_columns(pl.col("time").dt.truncate("1mo").alias("month"))
            .group_by("month")
            .agg(
                pl.len().alias("n"),
                pl.col("f").mean().alias("mean"),
                pl.col("f").std().alias("std"),
                pl.col("f").min().alias("min"),
                pl.col("f").max().alias("max"),
                (pl.col("f") < 0).mean().alias("share_negative"),
                (pl.col("f") == 0).mean().alias("share_zero"),
                (pl.col("f").abs() >= ACTIVE_FRACTION).mean().alias("share_active"),
            )
            .sort("month")
        )
        monthly.write_csv(OUT / f"monthly_{label}.csv")
        # Hour-of-day and weekday profiles, UTC and local, summer and winter.
        prof = {}
        for tz in ("UTC", "local"):
            col = pl.col("time") if tz == "UTC" else pl.col("local")
            vv = valid.with_columns(
                col.dt.hour().alias("h"), col.dt.weekday().alias("wd"), col.dt.month().alias("m")
            )
            vv.group_by("h").agg(
                pl.col("f").mean().alias("mean"),
                (pl.col("f") < 0).mean().alias("share_negative"),
                (pl.col("f") == 0).mean().alias("share_zero"),
                (pl.col("f").abs() >= ACTIVE_FRACTION).mean().alias("share_active"),
            ).sort("h").write_csv(OUT / f"hour_{tz}_{label}.csv")
            vv.group_by("wd").agg(
                pl.col("f").mean().alias("mean"),
                (pl.col("f").abs() >= ACTIVE_FRACTION).mean().alias("share_active"),
            ).sort("wd").write_csv(OUT / f"weekday_{tz}_{label}.csv")
            for season, months in (("summer", [6, 7, 8]), ("winter", [12, 1, 2])):
                p = (
                    vv.filter(pl.col("m").is_in(months))
                    .group_by("h")
                    .agg(pl.col("f").mean())
                    .sort("h")
                )
                prof[f"{tz}_{season}_peak_hour"] = int(p["h"][int(p["f"].arg_max())])  # ty: ignore[invalid-argument-type]
                prof[f"{tz}_{season}_trough_hour"] = int(p["h"][int(p["f"].arg_min())])  # ty: ignore[invalid-argument-type]
        r["profile_peaks"] = prof
        # Regime: active spells.
        active = valid["f"].abs() >= ACTIVE_FRACTION
        r["share_active"] = float(active.mean())  # ty: ignore[invalid-argument-type]
        ar = _runs(values=active)
        act = ar.filter(pl.col("v"))["length"]
        idl = ar.filter(~pl.col("v"))["length"]
        r["active_spells"] = int(act.len())
        r["active_spell_length_quantiles_half_hours"] = (
            {str(q): float(act.quantile(q)) for q in (0.1, 0.5, 0.9)} if act.len() else {}  # ty: ignore[invalid-argument-type]
        )
        r["idle_spell_length_quantiles_half_hours"] = (
            {str(q): float(idl.quantile(q)) for q in (0.1, 0.5, 0.9)} if idl.len() else {}  # ty: ignore[invalid-argument-type]
        )
        r["longest_idle_spell_days"] = float(idl.max() / 48) if idl.len() else 0.0  # ty: ignore[unsupported-operator]
        r["longest_active_spell_days"] = float(act.max() / 48) if act.len() else 0.0  # ty: ignore[unsupported-operator]
        r["starts_per_year"] = float(act.len() / (span / 17520))
        pos = valid.filter(pl.col("f") >= ACTIVE_FRACTION)["f"]
        neg = valid.filter(pl.col("f") <= -ACTIVE_FRACTION)["f"]
        r["positive_active_quantiles"] = (
            {str(q): float(pos.quantile(q)) for q in (0.05, 0.5, 0.95)} if pos.len() else {}  # ty: ignore[invalid-argument-type]
        )
        r["negative_active_quantiles"] = (
            {str(q): float(neg.quantile(q)) for q in (0.05, 0.5, 0.95)} if neg.len() else {}  # ty: ignore[invalid-argument-type]
        )
        r["idle_level_median_fraction"] = float(valid.filter(~active)["f"].median())  # ty: ignore[invalid-argument-type]
        # Dependence on price and national demand.
        j = (
            valid.select(
                "time",
                "f",
                pl.col("time").dt.convert_time_zone("Europe/London").dt.hour().alias("h"),
            )
            .join(price, on="time", how="inner")
            .join(demand, on="time", how="left")
        )
        r["joined_price_rows"] = j.height
        if j.height > 1000:
            hmean = j.group_by("h").agg(
                pl.col("f").mean().alias("f_h"), pl.col("price_gbp_mwh").mean().alias("p_h")
            )
            j = j.join(hmean, on="h").with_columns(
                (pl.col("f") - pl.col("f_h")).alias("f_res"),
                (pl.col("price_gbp_mwh") - pl.col("p_h")).alias("p_res"),
            )
            r["spearman_power_vs_price"] = float(
                j.select(pl.corr("f", "price_gbp_mwh", method="spearman")).item()
            )
            r["spearman_power_vs_price_hour_demeaned"] = float(
                j.select(pl.corr("f_res", "p_res", method="spearman")).item()
            )
            r["spearman_active_vs_price"] = float(
                j.with_columns((pl.col("f").abs() >= ACTIVE_FRACTION).cast(pl.Float64).alias("a"))
                .select(pl.corr("a", "price_gbp_mwh", method="spearman"))
                .item()
            )
            jd = j.drop_nulls("national_demand_mw")
            r["joined_demand_rows"] = jd.height
            if jd.height > 1000:
                r["spearman_power_vs_national_demand"] = float(
                    jd.select(pl.corr("f", "national_demand_mw", method="spearman")).item()
                )
                r["spearman_power_vs_embedded_wind"] = float(
                    jd.select(pl.corr("f", "embedded_wind_mw", method="spearman")).item()
                )
                dh = jd.group_by("h").agg(pl.col("national_demand_mw").mean().alias("d_h"))
                jd = jd.join(dh, on="h").with_columns(
                    (pl.col("national_demand_mw") - pl.col("d_h")).alias("d_res")
                )
                r["spearman_power_vs_national_demand_hour_demeaned"] = float(
                    jd.select(pl.corr("f_res", "d_res", method="spearman")).item()
                )
            dec = (
                j.with_columns(
                    pl.col("price_gbp_mwh")
                    .qcut(10, labels=[str(i) for i in range(10)])  # ty: ignore[deprecated]
                    .alias("decile")
                )
                .group_by("decile")
                .agg(
                    pl.col("f").mean().alias("mean"),
                    (pl.col("f") < 0).mean().alias("share_negative"),
                    (pl.col("f").abs() >= ACTIVE_FRACTION).mean().alias("share_active"),
                    pl.col("price_gbp_mwh").median().alias("price_median"),
                )
                .sort("price_median")
            )
            dec.write_csv(OUT / f"price_decile_{label}.csv")
        report[label] = r
    (OUT / "report_PRIVATE.json").write_text(json.dumps(report, indent=1, default=str))
    # Anonymised copy: drop scale in MW.
    anon = {k: {a: b for a, b in v.items() if "PRIVATE" not in a} for k, v in report.items()}
    (OUT / "report.json").write_text(json.dumps(anon, indent=1, default=str))
    print(json.dumps(anon, indent=1, default=str))


if __name__ == "__main__":
    main()
