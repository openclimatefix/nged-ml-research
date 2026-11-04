# ruff: noqa: E501, PLW2901, RET504
"""Tables of experiment 556: leak, unlearn time, adaptation, and the five-block month bootstrap.

Reads the per-cell shift tables written by `e00556_fit.py cells`. A cell's rows are sums of the
sign-adjusted median shift (points of capacity) by refit, Primary, lead band, and origin day. The
leak of the first refit is read over the 7 days after the origin; the leak of a later refit is read
over the origins that refit alone serves (from its own cutoff to the next cutoff of the arm).
"""

from typing import Final

import e00556_fit as fit
import numpy as np
import polars as pl

SITE_MAP: Final = (
    fit.ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
OUT: Final = fit.WORK / "scores"
GROUPS: Final[dict[str, str]] = {
    "MVA-only eight": "eight",
    "signed-MW six": "six",
    "D15 and D18": "report",
}
BAND_NAMES: Final[list[str]] = list(fit.BANDS)
CADENCE: Final[dict[str, int]] = {a.tag: a.cadence for a in fit.ARMS.values()}
RESAMPLES: Final[int] = 20_000
DEPTH_POINTS: Final[float] = 100.0 * fit.DEPTH_FRACTION


def load() -> pl.DataFrame:
    """Return every cell's rows with the Primary's group."""
    sites = pl.read_csv(SITE_MAP).select(
        "primary", group=pl.col("group").replace_strict(GROUPS, default=None)
    )
    files = sorted((fit.WORK / "cells").glob("*/*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files])
    return frame.join(sites, on="primary", how="left").with_columns(
        band_name=pl.col("band").replace_strict(dict(enumerate(BAND_NAMES)), default=None)
    )


def own_window(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep each refit's own scored origins: from its cutoff to the next cutoff of its arm."""
    cad = pl.col("arm").replace_strict(CADENCE, return_dtype=pl.Int64)
    return frame.filter(
        (pl.col("origin_off") >= pl.col("refit_off"))
        & (pl.col("origin_off") < pl.col("refit_off") + cad)
    )


def mean_by(*, frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Return the n-weighted mean shift for each key combination."""
    return frame.group_by(keys).agg(
        (pl.col("sum").sum() / pl.col("n").sum()).alias("leak"), pl.col("n").sum().alias("n")
    )


def first_refit_leak(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the leak of the first refit (7 days of origins) per arm, scenario, month, origin, sign, and group."""
    first = frame.filter(
        (pl.col("j") == 0) & (pl.col("origin_off") >= 0) & (pl.col("origin_off") < fit.WINDOW_DAYS)
    )
    return mean_by(frame=first, keys=["arm", "scenario", "month", "origin_day", "sign", "group"])


def fold_values(*, leaks: pl.DataFrame, scenario: str, group: str) -> pl.DataFrame:
    """Return each arm's leak per month (mean over origins and signs) for one scenario and group."""
    return (
        leaks.filter((pl.col("scenario") == scenario) & (pl.col("group") == group))
        .group_by("arm", "month")
        .agg(pl.col("leak").mean())
    )


def block_interval(*, diffs: np.ndarray, level: float) -> tuple[float, float, float]:
    """Return the mean and the percentile interval of a five-block month bootstrap (near min-to-max)."""
    rng = np.random.default_rng(0)
    draws = rng.choice(diffs, size=(RESAMPLES, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(diffs.mean()), float(lo), float(hi)


def leak_contrasts(*, leaks: pl.DataFrame, tests: int = 24) -> pl.DataFrame:
    """Return the leak of each arm minus F1d at step3_p10 with a Bonferroni block interval and fold signs."""
    level = 1 - 0.05 / tests
    rows = []
    for group in ("eight", "six"):
        values = fold_values(leaks=leaks, scenario="step3_p10", group=group)
        base = values.filter(pl.col("arm") == "F1d").sort("month")
        for arm in ("F3d", "F7d", "F1d-182", "F3d-E3", "F3d-R"):
            other = values.filter(pl.col("arm") == arm).sort("month")
            if other.height != base.height or base.height == 0:
                continue
            diff = other["leak"].to_numpy() - base["leak"].to_numpy()
            mean, lo, hi = block_interval(diffs=diff, level=level)
            rows.append(
                {"group": group, "contrast": f"{arm} - F1d", "mean": mean, "lo": lo, "hi": hi,
                 "fold_values": ", ".join(f"{d:+.2f}" for d in diff), "folds_negative": int((diff < 0).sum()),
                 "base_leak": float(base["leak"].mean())}  # ty: ignore[invalid-argument-type]
            )  # fmt: skip
    return pl.DataFrame(rows)


def unlearn(*, frame: pl.DataFrame, scenario: str) -> pl.DataFrame:
    """Return the leak per refit (days since the origin).

    Also return the days until the leak falls below 10% of its first value.
    """
    sub = own_window(frame=frame.filter(pl.col("scenario") == scenario))
    by = mean_by(frame=sub, keys=["arm", "group", "refit_off", "month", "origin_day", "sign"])
    curve = (
        by.group_by("arm", "group", "refit_off")
        .agg(pl.col("leak").mean())
        .sort("arm", "group", "refit_off")
    )
    return curve


def days_below(*, curve: pl.DataFrame, frac: float = 0.10) -> pl.DataFrame:
    """Return, per arm and group, the first refit offset (days) with the leak below `frac`.

    `frac` is a fraction of the first leak. The offset is null if the leak never falls that far.
    """
    rows = []
    for (arm, group), part in curve.group_by("arm", "group", maintain_order=True):
        part = part.sort("refit_off")
        first = float(part["leak"][0])
        hit = part.filter(pl.col("leak") < frac * first)
        rows.append(
            {
                "arm": arm,
                "group": group,
                "first_leak": first,
                "days_below_10pct": int(hit["refit_off"][0]) if hit.height else None,
                "last_leak": float(part["leak"][-1]),
                "last_day": int(part["refit_off"][-1]),
            }
        )
    return pl.DataFrame(rows)


def adaptation(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the share of a lasting 10% step in the median, by arm, group, and days since the origin."""
    sub = own_window(frame=frame.filter(pl.col("scenario") == "lasting0_p10"))
    by = (
        sub.group_by("arm", "group", "origin_off", "month", "sign")
        .agg((pl.col("sum").sum() / pl.col("n").sum()).alias("leak"))
        .group_by("arm", "group", "origin_off")
        .agg(pl.col("leak").mean())
        .with_columns(share=pl.col("leak") / DEPTH_POINTS)
        .sort("arm", "group", "origin_off")
    )
    return by


def days_to(*, curve: pl.DataFrame, level: float) -> pl.DataFrame:
    """Return, per arm and group, the first day (origin offset) at which the share reaches `level`, or null."""
    rows = []
    for (arm, group), part in curve.group_by("arm", "group", maintain_order=True):
        hit = part.sort("origin_off").filter(pl.col("share") >= level)
        rows.append(
            {
                "arm": arm,
                "group": group,
                f"days_to_{round(100 * level)}": int(hit["origin_off"][0]) if hit.height else None,
            }
        )
    return pl.DataFrame(rows)


def real_event_table() -> pl.DataFrame:
    """Return the absorbed share of real labelled events per arm.

    The table covers all events, all events without the largest, and the fold means with a
    five-block interval.
    """
    events = pl.read_parquet(OUT / "real_events.parquet").drop_nulls("absorbed_share")
    rows = []
    for arm in ("F1d", "F3d", "F7d"):
        part = events.filter(pl.col("arm") == arm)
        folds = (
            part.group_by("month")
            .agg(pl.col("absorbed_share").mean())
            .sort("month")["absorbed_share"]
            .to_numpy()
        )
        trimmed = part.sort("absorbed_share").head(part.height - 1)
        trimmed_folds = (
            trimmed.group_by("month")
            .agg(pl.col("absorbed_share").mean())
            .sort("month")["absorbed_share"]
            .to_numpy()
        )
        mean, lo, hi = block_interval(diffs=folds, level=0.95)
        tmean, tlo, thi = block_interval(diffs=trimmed_folds, level=0.95)
        rows.append(
            {
                "arm": arm,
                "events": part.height,
                "mean": part["absorbed_share"].mean(),
                "median": part["absorbed_share"].median(),
                "fold_means": ", ".join(f"{x:+.3f}" for x in folds),
                "fold_mean": mean,
                "lo": lo,
                "hi": hi,
                "mean_without_largest": trimmed["absorbed_share"].mean(),
                "fold_mean_without_largest": tmean,
                "lo_wl": tlo,
                "hi_wl": thi,
                "events_with_depth_10_points_or_more": part.filter(
                    pl.col("depth_points") >= 10
                ).height,
                "mean_depth_10_plus": part.filter(pl.col("depth_points") >= 10)[
                    "absorbed_share"
                ].mean(),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Write and print every table."""
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    leaks = first_refit_leak(frame=frame)
    leaks.write_csv(OUT / "leak_cells.csv")
    cfg = pl.Config(tbl_rows=80, tbl_cols=20, fmt_str_lengths=60)
    with cfg:
        table = (
            leaks.filter(pl.col("group").is_in(["eight", "six", "report"]))
            .group_by("arm", "scenario", "group")
            .agg(
                pl.col("leak").mean(),
                pl.col("leak").min().alias("min"),
                pl.col("leak").max().alias("max"),
            )
            .sort("scenario", "group", "arm")
        )
        table.write_csv(OUT / "leak_by_scenario.csv")
        print("LEAK (points of capacity, first refit, mean over folds, origins, signs)")
        print(table.with_columns(pl.col(pl.Float64).round(2)))
        bands = mean_by(
            frame=frame.filter(
                (pl.col("j") == 0)
                & (pl.col("origin_off") >= 0)
                & (pl.col("origin_off") < fit.WINDOW_DAYS)
            ),
            keys=["arm", "scenario", "group", "band"],
        ).sort("scenario", "group", "arm", "band")
        bands.write_csv(OUT / "leak_by_band.csv")
        contrasts = leak_contrasts(leaks=leaks)
        contrasts.write_csv(OUT / "leak_contrasts.csv")
        print(contrasts.with_columns(pl.col(pl.Float64).round(2)))
        for scenario in ("step3_p10", "step7_p10"):
            curve = unlearn(frame=frame, scenario=scenario)
            curve.write_csv(OUT / f"unlearn_{scenario}.csv")
            print(scenario, "unlearn")
            print(days_below(curve=curve).with_columns(pl.col(pl.Float64).round(2)))
        if (OUT / "real_events.parquet").exists():
            table_re = real_event_table()
            table_re.write_csv(OUT / "real_event_summary.csv")
            print(table_re.with_columns(pl.col(pl.Float64).round(3)))
        ad = adaptation(frame=frame)
        if ad.height:
            ad.write_csv(OUT / "adaptation.csv")
            print(
                days_to(curve=ad, level=0.5).join(days_to(curve=ad, level=0.9), on=["arm", "group"])
            )


def score_slices() -> None:
    """Print CRPS by lead band and on cold winter nights for each arm.

    The rows are those with no labelled event, at the eight and at the six.
    """
    import e00521_stats as s521  # ty: ignore[unresolved-import]

    arms = ["L-Ww", "F1d", "F3d", "F1d-182", "F3d-E3", "F3d-R"]
    frames = [
        pl.read_parquet(
            OUT / f"rows_{m}.parquet",
            columns=[
                "primary",
                "origin",
                "target",
                "group",
                "cls",
                "lead_hours",
                "cold",
                "main16",
                "tgt_real",
                *[f"{a}__crps" for a in arms],
            ],
        )
        for m in fit.c.MONTHS
    ]
    t = pl.concat(frames).filter(pl.col("main16") & ~pl.col("tgt_real"))
    t = t.with_columns(
        set=pl.when(pl.col("group") == "mva_eight")
        .then(pl.lit("eight"))
        .when((pl.col("group") == "signed_six") & (pl.col("cls") == "clean"))
        .then(pl.lit("six"))
        .otherwise(None),
        band=s521.band_expr(),
    ).drop_nulls("set")
    by_band = (
        t.group_by("set", "band")
        .agg(*[pl.col(f"{a}__crps").mean().alias(a) for a in arms])
        .sort("set", "band")
    )
    cold = (
        t.filter(pl.col("cold"))
        .group_by("set")
        .agg(pl.len().alias("rows"), *[pl.col(f"{a}__crps").mean().alias(a) for a in arms])
    )
    by_band.write_csv(OUT / "crps_by_band.csv")
    cold.write_csv(OUT / "crps_cold_nights.csv")
    with pl.Config(tbl_rows=40, tbl_cols=12):
        print(by_band.with_columns(pl.col(pl.Float64).round(3)))
        print(cold.with_columns(pl.col(pl.Float64).round(3)))


def noise_proxy() -> None:
    """Print the sign-pair noise proxy of the first-refit leak.

    The proxy is the standard deviation, over cells, of (leak at +1 minus leak at -1), divided by
    root 2.
    """
    cells = pl.read_csv(OUT / "leak_cells.csv").filter(
        pl.col("scenario").is_in(["step3_p10", "step7_p10"])
        & pl.col("group").is_in(["eight", "six"])
        & pl.col("arm").is_in(["F1d", "F3d-E3", "F3d-R", "F1d-182"])
    )
    wide = cells.pivot(
        on="sign", index=["arm", "scenario", "month", "origin_day", "group"], values="leak"
    ).with_columns(diff=pl.col("1") - pl.col("-1"))
    print(
        wide.group_by("arm", "scenario", "group")
        .agg(
            (pl.col("diff").std() / 2**0.5).alias("noise_points"),
            pl.col("diff").mean().alias("mean_sign_asymmetry"),
        )
        .sort("scenario", "group", "arm")
        .with_columns(pl.col(pl.Float64).round(2))
    )


if __name__ == "__main__":
    main()
    score_slices()
    noise_proxy()
