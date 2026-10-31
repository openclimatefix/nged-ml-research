"""Experiment 431: texture, timing, and agreement of idle half-hours at three metered wind farms.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue nice -n 19 \
        uv run --no-sync python -I ml-experiments/00431-wind-idle-data-look/e00431_analysis.py

Reads experiment 210's `joined.parquet` and `report.json` and writes summary tables (published-safe:
shares and areas under the curve only) and per-half-hour texture features (private) under
`data/ml-experiments/00431-wind-idle-data-look/`. Farms are W1, W2, W3 in the stored tables and
Farm A, Farm B, Farm C in every written page. No fit is run.
"""

import json
import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from scipy.stats import rankdata

DATA_ROOT: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
SRC_DIR: Final[Path] = DATA_ROOT / "00210-wind-idle-noise-mva-artefact"
OUT_DIR: Final[Path] = DATA_ROOT / "00431-wind-idle-data-look"
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
IDLE_WIND_MAX: Final[float] = 2.0
HIGH_WIND_FACTOR: Final[float] = 1.5
GEN_WIND_MIN: Final[float] = 6.0
GEN_POWER_SHARE: Final[tuple[float, float]] = (0.03, 0.08)
WINDOWS: Final[dict[str, int]] = {"3h": 6, "24h": 48}
N_BOOT: Final[int] = 2000
MAX_PER_CLASS_YEAR: Final[int] = 3000
SEED: Final[int] = 431
LEVEL_BIN: Final[float] = 0.005
LEVEL_PREFIXES: Final[tuple[str, ...]] = ("zero_share_", "amp_")
LEVEL_NAMES: Final[tuple[str, ...]] = ("is_zero",)
DAY_HOURS: Final[tuple[int, int]] = (8, 17)
FEATURE_PREFIXES: Final[tuple[str, ...]] = (
    "sd_diff_",
    "ac1_",
    "ac2_",
    "zero_share_",
    "repeat_share_",
)


def add_features(*, frame: pl.DataFrame, idle_p95: dict[str, float]) -> pl.DataFrame:
    """Add the texture statistics, computed from each farm's own power series only.

    Rolling windows never cross a gap of more than 30 minutes: each unbroken stretch is a segment.
    """
    frame = frame.sort("site", "valid_time").with_columns(
        gap=(pl.col("valid_time").diff().over("site") != pl.duration(minutes=30)).fill_null(
            value=True
        )
    )
    frame = frame.with_columns(seg=pl.col("gap").cum_sum().over("site"))
    by = ["site", "seg"]
    p95 = pl.col("site").replace_strict(idle_p95, return_dtype=pl.Float64)
    frame = frame.with_columns(
        power=pl.col("power").cast(pl.Float64),
        d=pl.col("power").cast(pl.Float64).diff().over(by),
    ).with_columns(
        is_zero=(pl.col("power") == 0).cast(pl.Float64),
        is_repeat=(pl.col("d") == 0).cast(pl.Float64),
        amp_idle_p95=pl.col("power").abs() / p95,
        amp_capacity=pl.col("power").abs() / pl.col("capacity_mw"),
    )
    exprs: list[pl.Expr] = []
    for label, size in WINDOWS.items():
        for mode, centre in (("trail", False), ("centre", True)):
            tag = f"{label}_{mode}"
            kw: dict[str, Any] = {"window_size": size, "min_samples": size // 2, "center": centre}

            exprs += [
                pl.col("d").rolling_std(**kw).over(by).alias(f"sd_diff_{tag}"),
                _windowed_corr(lag=1, size=size, centre=centre).alias(f"ac1_{tag}"),
                _windowed_corr(lag=2, size=size, centre=centre).alias(f"ac2_{tag}"),
                pl.col("is_zero").rolling_mean(**kw).over(by).alias(f"zero_share_{tag}"),
                pl.col("is_repeat").rolling_mean(**kw).over(by).alias(f"repeat_share_{tag}"),
            ]
    return frame.with_columns(exprs)


def _windowed_corr(*, lag: int, size: int, centre: bool) -> pl.Expr:
    """Correlation of the first difference with itself `lag` steps earlier, within a segment.

    `pl.rolling_corr` has no `center` argument, so a centred window is the trailing window shifted
    back by half its width. A constant window (all zeros, all repeats) has an undefined correlation
    and counts as 0, so the most idle-like stretches are scored and not dropped.
    """
    by = ["site", "seg"]
    lagged = pl.col("d").shift(lag).over(by)
    corr = pl.rolling_corr(pl.col("d"), lagged, window_size=size, min_samples=size // 2).over(by)
    if centre:
        corr = corr.shift(-(size // 2)).over(by)
    return corr.fill_nan(0.0)


def auc(*, score: np.ndarray, label: np.ndarray) -> float:
    """Return the area under the ROC curve of `score` for `label` (1 = idle), by ranks."""
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def folded(*, value: float) -> float:
    """Return the larger of the AUC and 1 minus the AUC (the statistic's sign is not pre-set)."""
    return max(value, 1.0 - value) if np.isfinite(value) else value


def sure_sets(*, frame: pl.DataFrame, bands: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """Label each half-hour `idle`, `gen`, or null (neither sure set), and flag the idle band."""
    lo = pl.col("site").replace_strict({s: b[0] for s, b in bands.items()}, return_dtype=pl.Float64)
    hi = pl.col("site").replace_strict({s: b[1] for s, b in bands.items()}, return_dtype=pl.Float64)
    in_band = (pl.col("power") >= lo) & (pl.col("power") <= hi)
    share = pl.col("power") / pl.col("capacity_mw")
    return frame.with_columns(
        in_band=in_band,
        sure=pl.when((pl.col("wind_speed") < IDLE_WIND_MAX) & in_band)
        .then(pl.lit("idle"))
        .when(
            (pl.col("wind_speed") >= GEN_WIND_MIN)
            & (share >= GEN_POWER_SHARE[0])
            & (share <= GEN_POWER_SHARE[1])
        )
        .then(pl.lit("gen"))
        .otherwise(None),
    )


def stratified_auc(*, score: np.ndarray, label: np.ndarray, level: np.ndarray) -> float:
    """Return the AUC within bins of `level`, averaged with weight (idle count x generating count).

    Bins hold power as a share of capacity, so a statistic that only reads the power level scores
    0.5. A bin with only one class adds nothing; the result is NaN when no bin holds both classes.
    """
    bins = np.floor(level / LEVEL_BIN).astype(int)
    total = 0.0
    weight = 0.0
    for b in np.unique(bins):
        sel = bins == b
        n_pos = int(label[sel].sum())
        n_neg = int(sel.sum()) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        total += auc(score=score[sel], label=label[sel]) * n_pos * n_neg
        weight += n_pos * n_neg
    return total / weight if weight else float("nan")


def auc_table(*, sure: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Return the AUC of each statistic, raw and level-matched, with year-block bootstrap intervals.

    The sign of each statistic is fixed once on the full sample (the larger of AUC and 1 - AUC), and
    the bootstrap resamples the signed AUC, so the intervals can fall below 0.5.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for site in SITES:
        sub = sure.filter((pl.col("site") == site) & pl.col("sure").is_not_null()).with_columns(
            year=pl.col("valid_time").dt.year(), level=pl.col("power") / pl.col("capacity_mw")
        )
        # Cap each class-year so the bootstrap stays fast; the cap is seeded.
        parts = [
            grp.sample(MAX_PER_CLASS_YEAR, seed=SEED) if grp.height > MAX_PER_CLASS_YEAR else grp
            for _, grp in sub.group_by("sure", "year")
        ]
        sub = pl.concat(parts)
        years = sorted(sub["year"].unique().to_list())
        by_year = {y: sub.filter(pl.col("year") == y) for y in years}
        for feat in features:
            data = {
                y: (
                    g[feat].fill_null(np.nan).to_numpy().astype(float),
                    (g["sure"] == "idle").to_numpy().astype(int),
                    g["level"].to_numpy().astype(float),
                )
                for y, g in by_year.items()
            }

            def pooled(ys: list[int], feat_data: dict = data) -> tuple[np.ndarray, ...]:
                score = np.concatenate([feat_data[y][0] for y in ys])
                label = np.concatenate([feat_data[y][1] for y in ys])
                level = np.concatenate([feat_data[y][2] for y in ys])
                ok = np.isfinite(score)
                return score[ok], label[ok], level[ok]

            score, label, _ = pooled(years)
            raw_point = auc(score=score, label=label)
            sign = 1.0 if not np.isfinite(raw_point) or raw_point >= 0.5 else -1.0

            def signed(ys: list[int], sgn: float = sign, pool: Any = pooled) -> tuple[float, float]:
                sc, lb, lv = pool(ys)
                raw = auc(score=sgn * sc, label=lb)
                matched = stratified_auc(score=sgn * sc, label=lb, level=lv)
                return raw, matched

            boots = np.array(
                [signed(list(rng.choice(years, size=len(years)))) for _ in range(N_BOOT)]
            )
            point_raw, point_matched = signed(years)

            def interval(column: np.ndarray) -> tuple[float, float]:
                finite = column[np.isfinite(column)]
                if not len(finite):
                    return float("nan"), float("nan")
                return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))

            raw_lo, raw_hi = interval(boots[:, 0])
            m_lo, m_hi = interval(boots[:, 1])
            rows.append(
                {
                    "site": site,
                    "statistic": feat,
                    "gate_statistic": not feat.startswith(LEVEL_PREFIXES)
                    and feat not in LEVEL_NAMES,
                    "sign": sign,
                    "auc_raw": point_raw,
                    "raw_lo": raw_lo,
                    "raw_hi": raw_hi,
                    "auc_level_matched": point_matched,
                    "matched_lo": m_lo,
                    "matched_hi": m_hi,
                    # Rows actually scored (finite statistic), not rows before the filter.
                    "n_idle": int(label.sum()),
                    "n_gen": int(len(label) - label.sum()),
                    "dropped_share": 1.0 - len(label) / sum(len(v[1]) for v in data.values()),
                }
            )
    return pl.DataFrame(rows)


def high_wind_idle(*, frame: pl.DataFrame, cut_in: dict[str, float]) -> pl.DataFrame:
    """Return counts of high-wind half-hours and idle ones, by farm, year, weekday, and UTC hour."""
    thr = pl.col("site").replace_strict(
        {s: HIGH_WIND_FACTOR * c for s, c in cut_in.items()}, return_dtype=pl.Float64
    )
    return (
        frame.filter(pl.col("wind_speed") > thr)
        .with_columns(
            weekday=pl.col("valid_time").dt.weekday(),
            hour=pl.col("valid_time").dt.hour(),
            year=pl.col("valid_time").dt.year(),
        )
        .group_by("site", "year", "weekday", "hour")
        .agg(n=pl.len(), n_idle=pl.col("in_band").sum())
    )


def maintenance_ratio(*, hi: pl.DataFrame) -> pl.DataFrame:
    """Return the idle share at weekday daytimes over the share at weekends and nights, per farm.

    The interval resamples calendar years (year blocks).
    """
    rng = np.random.default_rng(SEED)
    flagged = hi.with_columns(
        daytime=(pl.col("weekday") <= 5) & pl.col("hour").is_between(DAY_HOURS[0], DAY_HOURS[1])
    )
    rows = []
    for site in SITES:
        sub = flagged.filter(pl.col("site") == site)
        years = sorted(sub["year"].unique().to_list())
        # Counts per year: (daytime n, daytime idle, other n, other idle).
        counts = {}
        for y in years:
            part = sub.filter(pl.col("year") == y)
            day = part.filter(pl.col("daytime"))
            off = part.filter(~pl.col("daytime"))
            counts[y] = np.array(
                [day["n"].sum(), day["n_idle"].sum(), off["n"].sum(), off["n_idle"].sum()],
                dtype=float,
            )

        def shares(ys: list[int], table: dict = counts) -> tuple[float, float]:
            tot = np.sum([table[y] for y in ys], axis=0)
            return float(tot[1] / max(tot[0], 1)), float(tot[3] / max(tot[2], 1))

        def ratio(ys: list[int]) -> float:
            s_day, s_off = shares(ys)
            return s_day / s_off if s_off > 0 else float("nan")

        boots = np.array([ratio(list(rng.choice(years, size=len(years)))) for _ in range(N_BOOT)])
        boots = boots[np.isfinite(boots)]
        s_day, s_off = shares(years)
        rows.append(
            {
                "site": site,
                "ratio": ratio(years),
                "lo": float(np.quantile(boots, 0.025)) if len(boots) else np.nan,
                "hi": float(np.quantile(boots, 0.975)) if len(boots) else np.nan,
                "share_daytime": s_day,
                "share_other": s_off,
            }
        )
    return pl.DataFrame(rows)


def floor_by_year(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the idle floor (ERA5 wind below 2 m/s) per farm and year, as a share of capacity."""
    return (
        frame.filter(pl.col("wind_speed") < IDLE_WIND_MAX)
        .with_columns(
            year=pl.col("valid_time").dt.year(), rel=pl.col("power") / pl.col("capacity_mw")
        )
        .group_by("site", "year")
        .agg(
            n=pl.len(),
            median=pl.col("rel").median(),
            p05=pl.col("rel").quantile(0.05),
            p95=pl.col("rel").quantile(0.95),
            share_zero=(pl.col("power") == 0).mean(),
            share_negative=(pl.col("power") < 0).mean(),
        )
        .sort("site", "year")
    )


def farm_b_negatives_by_hour(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return Farm B's share of negative readings by UTC hour, at ERA5 wind below 2 m/s."""
    return (
        frame.filter((pl.col("site") == "W2") & (pl.col("wind_speed") < IDLE_WIND_MAX))
        .with_columns(hour=pl.col("valid_time").dt.hour())
        .group_by("hour")
        .agg(n=pl.len(), share_negative=(pl.col("power") < 0).mean())
        .sort("hour")
    )


def b_vs_c(*, frame: pl.DataFrame, cut_in: dict[str, float]) -> pl.DataFrame:
    """Cross-tabulate Farm B and Farm C idle-band status, by whether the shared wind is high."""
    b = frame.filter(pl.col("site") == "W2").select(
        "valid_time", "wind_speed", b_idle=pl.col("in_band")
    )
    c = frame.filter(pl.col("site") == "W3").select("valid_time", c_idle=pl.col("in_band"))
    both = b.join(c, on="valid_time").with_columns(
        wind_class=pl.when(pl.col("wind_speed") < IDLE_WIND_MAX)
        .then(pl.lit("below 2 m/s"))
        .when(pl.col("wind_speed") > HIGH_WIND_FACTOR * max(cut_in["W2"], cut_in["W3"]))
        .then(pl.lit("high wind"))
        .otherwise(pl.lit("between")),
        year=pl.col("valid_time").dt.year(),
    )
    return (
        both.group_by("wind_class", "b_idle", "c_idle", "year")
        .agg(n=pl.len())
        .sort("wind_class", "b_idle", "c_idle", "year")
    )


def main() -> None:
    """Run every table and write the outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((SRC_DIR / "report.json").read_text())
    joined = pl.read_parquet(SRC_DIR / "joined.parquet")
    idle_p95 = {s: float(report[s]["idle_floor"]["p95"]) for s in SITES}
    bands = {
        s: (float(report[s]["idle_floor"]["p05"]), float(report[s]["idle_floor"]["p95"]))
        for s in SITES
    }
    cut_in = {s: float(report[s]["cut_in_used"]) for s in SITES}
    frame = sure_sets(frame=add_features(frame=joined, idle_p95=idle_p95), bands=bands)
    features = [c for c in frame.columns if c.startswith(FEATURE_PREFIXES)] + [
        "is_zero",
        "is_repeat",
        "amp_idle_p95",
        "amp_capacity",
    ]
    frame.write_parquet(OUT_DIR / "features_private.parquet")

    sets = (
        frame.filter(pl.col("sure").is_not_null())
        .with_columns(year=pl.col("valid_time").dt.year())
        .group_by("site", "sure", "year")
        .agg(n=pl.len())
        .sort("site", "sure", "year")
    )
    sets.write_parquet(OUT_DIR / "sure_set_sizes.parquet")
    auc_table(sure=frame, features=features).write_parquet(OUT_DIR / "auc.parquet")
    empirical = {s: float(report[s]["cut_in_empirical"]) for s in SITES}
    ratios = []
    for kind, speeds in (("used (pre-registered)", cut_in), ("empirical (sensitivity)", empirical)):
        hi = high_wind_idle(frame=frame, cut_in=speeds)
        hi.write_parquet(OUT_DIR / f"high_wind_idle_{kind.split()[0]}.parquet")
        ratios.append(maintenance_ratio(hi=hi).with_columns(cut_in=pl.lit(kind)))
    pl.concat(ratios).write_parquet(OUT_DIR / "maintenance_ratio.parquet")
    floor_by_year(frame=frame).write_parquet(OUT_DIR / "floor_by_year.parquet")
    farm_b_negatives_by_hour(frame=frame).write_parquet(
        OUT_DIR / "farm_b_negatives_by_hour.parquet"
    )
    b_vs_c(frame=frame, cut_in=cut_in).write_parquet(OUT_DIR / "b_vs_c.parquet")
    texture = (
        frame.filter(pl.col("sure").is_not_null())
        .group_by("site", "sure")
        .agg(
            zero=pl.col("is_zero").mean(),
            repeat=pl.col("is_repeat").mean(),
            # Quantisation step: smallest non-zero absolute first difference.
            step_share_of_capacity=(pl.col("d").abs() / pl.col("capacity_mw"))
            .filter(pl.col("d").abs() > 0)
            .min(),
            sd24=pl.col("sd_diff_24h_centre").median(),
        )
        .sort("site", "sure")
    )
    texture.write_parquet(OUT_DIR / "texture_by_set.parquet")


if __name__ == "__main__":
    main()
