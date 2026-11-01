# ruff: noqa: E501
"""Score the forecasts of experiment 476 against climatology and persistence (no fit).

Scores are the quantile score of `studies.eval_protocol.metrics` (twice the mean pinball loss over
the 11 levels) in points, one point being 1% of the series' maximum absolute output. Per arm the
three seeds' per-row scores are averaged row by row. Climatology (quantiles of the fold's
training window by local month, half-hour, and weekend flag) and persistence (the newest usable
reading as a point forecast, climatology's median where it is missing) are scored on the same rows.
Contrasts are paired row by row; the interval is a bootstrap over blocks of one fold and one origin
day, at the 99% level that the Bonferroni count of 5 sets, and is interim (four folds). Everything
is written to the private data folder.

```bash
source env; PYTHONPATH=ml-experiments/00087-*:ml-experiments/00397-*:ml-experiments/00476-* \
  uv run --no-sync python ml-experiments/00476-*/e00476_score.py
```
"""

import json
import math
from datetime import datetime, timedelta
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00476_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

POINTS: Final[float] = 100.0
RESAMPLES: Final[int] = 4_000
LEVEL: Final[float] = 0.99
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
NAMES: Final[tuple[str, ...]] = tuple(f"q{round(level * 100):02d}" for level in fit.LEVELS)
MEDIAN: Final[int] = 5
IDLE: Final[float] = 0.05
"""A half-hour is idle when its absolute output is under this share of the maximum."""
WIDENINGS: Final[tuple[float, ...]] = (1.0, 1.15, 1.3)
BANDS: Final[dict[str, tuple[float, float]]] = {
    "days 1-10": (0.0, 240.0),
    "12-36 h": (12.0, 36.0),
    "3-10 d": (72.0, 240.0),
}
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 A0 - climatology": ("A0", "clim"),
    "C2 A0 - persistence": ("A0", "persist"),
    "C3 A1 - A0": ("A1", "A0"),
    "C4 A2 - A0": ("A2", "A0"),
    "C5 A3 - climatology": ("A3", "clim"),
}
REFERENCES: Final[tuple[str, ...]] = ("clim", "persist")


def widened(*, q: np.ndarray, factor: float) -> np.ndarray:
    """Return the sorted quantiles with every spread about the median multiplied by `factor`."""
    median = q[:, [MEDIAN]]
    return np.sort(median + factor * (q - median), axis=1)


def pinball(*, actual: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Return the pinball loss of one quantile."""
    d = actual - q
    return np.maximum(level * d, (level - 1.0) * d)


def local_cell(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the climatology cell (local month, half-hour, weekend flag) of each valid time."""
    t = pl.Series(valid_us).cast(pl.Datetime("us", "UTC")).dt.convert_time_zone("Europe/London")
    half = t.dt.hour() * 2 + t.dt.minute() // 30
    return (
        (t.dt.month().cast(pl.Int64) * 100 + half.cast(pl.Int64)) * 2 + (t.dt.weekday() >= 6)
    ).to_numpy()


def climatology(
    *, power: np.ndarray, refit: datetime, valid_us: np.ndarray, smooth: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return climatology quantiles (rows, 11) and P(negative) for `valid_us`, from the training window.

    The cell is the local month, half-hour, and weekend flag. With `smooth`, the cell pools the
    same half-hour and weekend flag over the month and its two neighbours (a month occurs only
    twice in the 24-month window, so a single month's cell holds about 16 readings). Without it,
    a cell with under 20 readings falls back to the whole window (the audit's sensitivity row).
    """
    grid = fit.common.grid_times()
    lo = fit.common.grid_index(times=refit - timedelta(days=prep.WINDOW_DAYS))
    hi = fit.common.grid_index(times=refit - timedelta(hours=3.5))
    window = np.arange(lo, hi + 1)
    window = window[np.isfinite(power[window])]
    cells = local_cell(valid_us=grid.dt.epoch("us").to_numpy()[window])
    values = power[window].astype(np.float64)
    wanted = local_cell(valid_us=valid_us)
    q = np.empty((len(valid_us), len(fit.LEVELS)))
    p_neg = np.empty(len(valid_us))
    for cell in np.unique(wanted):
        if smooth:
            month, rest = divmod(int(cell), 200)
            months = [(month - 2 + d) % 12 + 1 for d in (-1, 0, 1)]
            keep = np.isin(cells // 200, months) & (cells % 200 == rest)
            sel = values[keep]
        else:
            sel = values[cells == cell]
        if len(sel) < 20:  # too few readings: use the whole training window
            sel = values
        rows = wanted == cell
        q[rows] = np.quantile(sel, fit.LEVELS)
        p_neg[rows] = float((sel < 0).mean())
    return q, p_neg


def collect(*, seeds: tuple[int, ...], factor: float) -> pl.DataFrame:
    """Return one row per scored (arm, fold, origin day) with the summed scores and counts."""
    power, price, _ = fit.grid_arrays()
    frame = fit.build_frame(power=power, price=price, weather=fit.weather_table())
    records = []
    for month in fit.SCREEN_MONTHS:
        refit = prep.refit_of(month=month)
        keys = None
        arms: dict[str, dict] = {}
        for arm in fit.ARMS:
            files = [pl.read_parquet(fit.PRED_DIR / arm / f"{month}_s{s}.parquet") for s in seeds]
            if keys is None:
                keys = files[0].select("origin_us", "valid_us", "lead", "actual")
            for f in files:
                assert f["valid_us"].equals(keys["valid_us"]), "seed files differ in row order"
            qs = [
                widened(q=f.select(list(NAMES)).to_numpy().astype(np.float64), factor=factor)
                for f in files
            ]
            arms[arm] = {"q": qs, "p_neg": np.mean([f["p_neg"].to_numpy() for f in files], axis=0)}
        assert keys is not None
        actual = keys["actual"].to_numpy().astype(np.float64)
        ok = np.isfinite(actual) & (keys["lead"].to_numpy() <= 240.0)
        valid_us = keys["valid_us"].to_numpy()
        cq, cneg = climatology(power=power, refit=refit, valid_us=valid_us, smooth=True)
        cell_q, cell_neg = climatology(power=power, refit=refit, valid_us=valid_us, smooth=False)
        order = pl.DataFrame(
            {"origin": frame.origin, "valid": frame.valid, "i": np.arange(len(frame.y))}
        )
        joined = keys.select("origin_us", "valid_us").join(
            order,
            left_on=["origin_us", "valid_us"],
            right_on=["origin", "valid"],
            how="left",
            validate="1:1",
            maintain_order="left",
        )
        newest = frame.lags[joined["i"].to_numpy(), 0].astype(np.float64)
        persist_missing = ~np.isfinite(newest)
        point = np.where(persist_missing, cq[:, MEDIAN], newest)
        persist_q = np.repeat(point[:, None], len(fit.LEVELS), axis=1)
        persist_neg = (point < 0).astype(np.float64)
        refs = {
            "clim": ([cq], cneg),
            "clim_cell": ([cell_q], cell_neg),
            "persist": ([persist_q], persist_neg),
        }
        sources = {**{a: (v["q"], v["p_neg"]) for a, v in arms.items()}, **refs}
        origin_day = (keys["origin_us"].to_numpy() // (86_400 * 1_000_000)).astype(np.int64)
        lead = keys["lead"].to_numpy()
        for name, (qs, p_neg) in sources.items():
            per_seed = np.stack(
                [metrics.quantile_score(actual=actual[ok], quantiles=q[ok]) for q in qs]
            )
            score = per_seed.mean(0)
            seed_cols = (
                {f"score_s{s}": per_seed[i] for i, s in enumerate(seeds)}
                if len(qs) == len(seeds)
                else {}
            )
            mean_q = np.mean([np.asarray(q) for q in qs], axis=0)[ok]
            frame_ = pl.DataFrame(
                {
                    "arm": name,
                    "month": month,
                    "day": origin_day[ok],
                    "lead": lead[ok],
                    "idle": np.abs(actual[ok]) < IDLE,
                    "score": score,
                    **seed_cols,
                    **{
                        f"pin_{t}": pinball(
                            actual=actual[ok], q=mean_q[:, NAMES.index(t)], level=lv
                        )
                        for t, lv in TAILS.items()
                    },
                    **{
                        f"cov_{t}": (actual[ok] <= mean_q[:, NAMES.index(t)]).astype(np.float64)
                        for t in TAILS
                    },
                    "sign_brier": (p_neg[ok] - (actual[ok] < 0)) ** 2,
                    "sign_right": ((p_neg[ok] > 0.5) == (actual[ok] < 0)).astype(np.float64),
                    "negative": (actual[ok] < 0).astype(np.float64),
                    "median_error": np.abs(actual[ok] - mean_q[:, MEDIAN]),
                    "persist_missing": persist_missing[ok]
                    if name == "persist"
                    else np.zeros(ok.sum(), dtype=bool),
                }
            )
            records.append(frame_)
    return pl.concat(records, how="diagonal_relaxed")


def band_filter(*, band: str) -> pl.Expr:
    """Return the lead filter of a band."""
    lo, hi = BANDS[band]
    return (pl.col("lead") > lo) & (pl.col("lead") <= hi)


def absolute_table(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return absolute scores by arm and band, and idle and active scores for days 1 to 10."""
    rows = []
    for band in BANDS:
        sub = table.filter(band_filter(band=band))
        for kind, expr in (
            ("all", pl.lit(True)),
            ("idle", pl.col("idle")),
            ("active", ~pl.col("idle")),
        ):
            part = sub.filter(expr)
            g = (
                part.group_by("arm")
                .agg(
                    pl.len().alias("n"),
                    (pl.col("score").mean() * POINTS).alias("crps"),
                    *[(pl.col(f"pin_{t}").mean() * POINTS).alias(f"pinball_{t}") for t in TAILS],
                    *[pl.col(f"cov_{t}").mean().alias(f"coverage_{t}") for t in TAILS],
                    pl.col("sign_brier").mean().alias("sign_brier"),
                    pl.col("sign_right").mean().alias("sign_accuracy"),
                    (pl.col("median_error").mean() * POINTS).alias("median_abs_error"),
                )
                .with_columns(band=pl.lit(band), rows=pl.lit(kind))
            )
            rows.append(g)
    return pl.concat(rows).sort("band", "rows", "arm")


def contrast_table(*, table: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return each contrast with a block-bootstrap 99% interval, by band."""
    out = []
    for band in BANDS:
        sub = table.filter(band_filter(band=band))
        for name, (a, b) in CONTRASTS.items():
            left = sub.filter(pl.col("arm") == a)
            right = sub.filter(pl.col("arm") == b)
            assert left.height == right.height
            diff = left["score"].to_numpy() - right["score"].to_numpy()
            d = pl.DataFrame({"month": left["month"], "day": left["day"], "diff": diff})
            blocks = d.group_by("month", "day").agg(pl.col("diff").sum(), pl.len().alias("n"))
            n, s = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
            draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
            boot = s[draws].sum(1) / n[draws].sum(1) * POINTS
            tail = (1.0 - LEVEL) / 2.0 * 100.0
            by_month = d.group_by("month").agg(pl.col("diff").mean() * POINTS)
            out.append({
                "band": band, "contrast": name, "n": int(n.sum()),
                "difference": float(s.sum() / n.sum() * POINTS),
                "low": float(np.percentile(boot, tail)), "high": float(np.percentile(boot, 100 - tail)),
                "fold_differences": json.dumps({r["month"]: round(r["diff"], 4) for r in by_month.iter_rows(named=True)}),
            })  # fmt: skip
    return pl.DataFrame(out)


def minimum_detectable_effect(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> float:
    """Return 1.96 x sqrt(2) x (A0 seed sd) / sqrt(seeds), in points, from A0's per-fold seed means."""
    sub = table.filter((pl.col("arm") == "A0") & band_filter(band="days 1-10"))
    per_fold = sub.group_by("month").agg(*[(pl.col(f"score_s{s}").mean() * POINTS) for s in seeds])
    values = per_fold.select([f"score_s{s}" for s in seeds]).to_numpy()
    folds, n_seeds = values.shape
    residual = values - values.mean(axis=1, keepdims=True)
    sd = math.sqrt((residual**2).sum() / (folds * (n_seeds - 1)))
    return 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)


def main() -> None:
    """Score every widening and write the tables."""
    seeds = fit.SEEDS
    rng = np.random.default_rng(0)
    report: dict = {}
    for factor in WIDENINGS:
        table = collect(seeds=seeds, factor=factor)
        tag = f"widen{factor:g}"
        absolute = absolute_table(table=table)
        absolute.write_csv(fit.OUT / f"absolute_{tag}.csv")
        if factor == 1.0:
            contrasts = contrast_table(table=table, rng=rng)
            contrasts.write_csv(fit.OUT / "contrasts.csv")
            mde = minimum_detectable_effect(table=table, seeds=seeds)
            report["minimum_detectable_effect_points"] = mde
            report["persistence_rows_missing_newest"] = int(
                table.filter(pl.col("arm") == "persist")["persist_missing"].sum()
            )
            report["rows_per_arm"] = int(table.filter(pl.col("arm") == "A0").height)
            print(f"minimum detectable effect: {mde:.4f} points")
            with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
                print(contrasts.drop("fold_differences"))
        with pl.Config(tbl_rows=100, tbl_cols=20):
            print(tag)
            print(
                absolute.filter(pl.col("rows") == "all").select(
                    "band",
                    "arm",
                    "n",
                    "crps",
                    "median_abs_error",
                    "pinball_q01",
                    "pinball_q99",
                    "coverage_q01",
                    "coverage_q99",
                    "sign_brier",
                    "sign_accuracy",
                )
            )
    (fit.OUT / "score_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
