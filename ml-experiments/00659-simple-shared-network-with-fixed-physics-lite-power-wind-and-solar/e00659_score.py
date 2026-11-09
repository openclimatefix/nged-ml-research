# ruff: noqa: E501
"""Score experiment 659's arms and the two defaults: lead bands, intervals, tails, widening.

Reads the tables of `e00659_assemble.py` (fractions of experiment 615's scoring capacity) and writes
CSVs under `scores/`. Intervals resample the 5 forecast-origin months with replacement (month blocks).
Every contrast is "arm minus reference" on the rows both cover, so a negative number means the arm
scored better. The minimum detectable effect is (2.576 + 0.842) times the bootstrap standard error.
Tail widening follows experiment 499's rule `e`: two constants per series (one for q01 and q05, one
for q95 and q99) fitted by grid search on the earlier development months only, then clipped as in
experiment 451 (tails at least 0, every level at most the larger of 1.0 and the earlier maximum).
The first development month is only fitted on, so widened scores cover the other four months.

    uv run --no-sync python ml-experiments/00659-*/e00659_score.py
"""

import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00659_common as common
import numpy as np
import polars as pl

Q: Final = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
LEVELS: Final = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
BANDS: Final = {
    "0-24 h": (0, 24),
    "24-120 h": (24, 120),
    "120-240 h": (120, 240),
    "all (0-240 h)": (0, 240),
}
SCORES: Final = common.DATA_DIR / "scores"
GRID: Final = np.round(np.arange(0.5, 1.801, 0.01), 2)
RESAMPLES: Final[int] = 10_000
WIND_REF: Final[str] = "wind_default2_old_capacity"
SOLAR_REF: Final[str] = "solar_a2_member_quarterly"
E615: Final = (
    common.TABLE_DIR.parent.parent
    / "00615-bakeoff-wind-and-solar-development-month-backtests"
    / "tables"
)


def load(*, name: str) -> pl.DataFrame:
    """Return a candidate's table (this experiment's folder first, then experiment 615's)."""
    path = common.TABLE_DIR / f"{name}.parquet"
    frame = pl.read_parquet(path if path.exists() else E615 / f"{name}.parquet")
    if "masked" not in frame.columns:
        frame = frame.with_columns(masked=pl.lit(value=False))
    if "actual_raw" not in frame.columns:
        frame = frame.with_columns(actual_raw=pl.col("actual"))
    return frame.with_columns(month=pl.col("init_time").dt.strftime("%Y-%m")).filter(
        pl.col("lead_hours") <= 240
    )


def losses(*, frame: pl.DataFrame, actual: str) -> dict[str, np.ndarray]:
    """Return per-row mean pinball (points), absolute error of the median (points), and indicators."""
    y, q = frame[actual].to_numpy(), frame.select(Q).to_numpy()
    d = y[:, None] - q
    return {
        "score": 100 * np.maximum(LEVELS * d, (LEVELS - 1) * d).mean(axis=1),
        "mae_median": 100 * np.abs(d[:, 5]),
        "inside_q05_q95": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(float),
        "above_q99": (y > q[:, 10]).astype(float),
        "below_q01": (y < q[:, 0]).astype(float),
        "pin_q01": 100 * np.maximum(0.01 * d[:, 0], -0.99 * d[:, 0]),
        "pin_q99": 100 * np.maximum(0.99 * d[:, 10], -0.01 * d[:, 10]),
    }


def block_bootstrap(
    *, values: np.ndarray, months: np.ndarray, seed: int = 659
) -> tuple[float, float, float, float]:
    """Return the mean, the 99% interval, and the standard error, resampling whole months."""
    labels = sorted(set(months))
    sums = np.array([values[months == m].sum() for m in labels])
    counts = np.array([(months == m).sum() for m in labels], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(labels), size=(RESAMPLES, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    low, high = np.quantile(draws, [0.005, 0.995])
    return float(values.mean()), float(low), float(high), float(draws.std())


def pick(*, frame: pl.DataFrame, variant: str) -> tuple[pl.DataFrame, str]:
    """Return the rows and the actual column of a wind variant (solar uses `all_rows`)."""
    if variant == "cleaned_zeros":
        return frame, "actual"
    if variant == "kept_rows":
        return frame.filter(~pl.col("masked")), "actual_raw"
    return frame, "actual_raw"


def overall(*, names: list[str], variant: str) -> pl.DataFrame:
    """Return every candidate's scores per lead band with month-block intervals."""
    out = []
    for name in names:
        frame, col = pick(frame=load(name=name), variant=variant)
        for band, (lo, hi) in BANDS.items():
            part = frame.filter(
                (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi + (1 if hi == 240 else 0))
            )
            if part.height == 0:
                continue
            values, months = losses(frame=part, actual=col), part["month"].to_numpy()
            row = {"candidate": name, "variant": variant, "band": band, "rows": part.height}
            for key, v in values.items():
                m, lo_i, hi_i, _ = block_bootstrap(values=v, months=months)
                row[key] = m
                if key in ("score", "mae_median"):
                    row[f"{key}_lo"], row[f"{key}_hi"] = lo_i, hi_i
            out.append(row)
    return pl.DataFrame(out)


def contrasts(*, names: list[str], reference: str, variant: str) -> pl.DataFrame:
    """Return "arm minus reference" on shared rows with month-block intervals and the minimum detectable effect."""
    keys = ["series", "init_time", "valid_time"]
    ref_frame, col = pick(frame=load(name=reference), variant=variant)
    out = []
    for name in names:
        if name == reference:
            continue
        frame, _ = pick(frame=load(name=name), variant=variant)
        joined = frame.join(
            ref_frame.select(*keys, *[pl.col(q).alias(f"{q}_r") for q in Q]), on=keys, how="inner"
        )
        for band, (lo, hi) in BANDS.items():
            part = joined.filter(
                (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi + (1 if hi == 240 else 0))
            )
            if part.height == 0:
                continue
            arm = losses(frame=part, actual=col)
            ref = losses(frame=part.drop(Q).rename({f"{q}_r": q for q in Q}), actual=col)
            months = part["month"].to_numpy()
            for metric in ("score", "mae_median"):
                diff = arm[metric] - ref[metric]
                m, low, high, se = block_bootstrap(values=diff, months=months)
                out.append(
                    {
                        "arm": name,
                        "reference": reference,
                        "variant": variant,
                        "band": band,
                        "metric": metric,
                        "rows": part.height,
                        "difference": m,
                        "lo99": low,
                        "hi99": high,
                        "mde": 3.418 * se,
                    }
                )
    return pl.DataFrame(out)


def pinball(*, y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Return the pinball loss at level `tau`."""
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def fit_constant(
    *, y: np.ndarray, q50: np.ndarray, tails: np.ndarray, taus: tuple[float, ...]
) -> float:
    """Return the grid factor that minimises the summed pinball loss of the tails."""
    loss = np.zeros(len(GRID))
    for j, tau in enumerate(taus):
        for i, f in enumerate(GRID):
            loss[i] += pinball(y=y, q=q50 + f * (tails[:, j] - q50), tau=tau).mean()
    return float(GRID[int(np.argmin(loss))])


def widen(*, frame: pl.DataFrame, actual: str) -> tuple[pl.DataFrame, list[dict]]:
    """Return the frame (months after the first) with each series' tails widened by earlier months."""
    months = sorted(frame["month"].unique())
    pieces, constants = [], []
    for series in sorted(frame["series"].unique()):
        part = frame.filter(pl.col("series") == series)
        for k, month in enumerate(months[1:], start=1):
            fit = part.filter(pl.col("month").is_in(months[:k]) & ~pl.col("masked"))
            held = part.filter(pl.col("month") == month)
            if fit.height < 200 or held.height == 0:
                continue
            y, q50 = fit[actual].to_numpy(), fit["q50"].to_numpy()
            low = fit_constant(
                y=y, q50=q50, tails=fit.select("q01", "q05").to_numpy(), taus=(0.01, 0.05)
            )
            up = fit_constant(
                y=y, q50=q50, tails=fit.select("q95", "q99").to_numpy(), taus=(0.95, 0.99)
            )
            ceiling = max(1.0, float(np.nanmax(y)))
            mid = held["q50"]
            widened = held.with_columns(
                *[
                    (mid + low * (pl.col(c) - mid)).clip(lower_bound=0.0).alias(c)
                    for c in ("q01", "q05")
                ],
                *[(mid + up * (pl.col(c) - mid)).alias(c) for c in ("q95", "q99")],
            ).with_columns(*[pl.col(c).clip(upper_bound=ceiling) for c in Q])
            pieces.append(widened)
            constants.append({"series": series, "month": month, "lower": low, "upper": up})
    return pl.concat(pieces), constants


def tail_table(*, names: list[str], variant: str) -> pl.DataFrame:
    """Return raw and widened tail scores on the months after the first."""
    rows = []
    for name in names:
        frame, col = pick(frame=load(name=name), variant=variant)
        wide, _ = widen(frame=frame, actual=col)
        raw = frame.filter(pl.col("month") != min(frame["month"].unique()))
        for label, data in (("raw", raw), ("widened", wide)):
            v = losses(frame=data, actual=col)
            rows.append(
                {
                    "candidate": name,
                    "variant": variant,
                    "rule": label,
                    "rows": data.height,
                    **{k: float(x.mean()) for k, x in v.items()},
                }
            )
    return pl.DataFrame(rows)


def special_days(*, name: str) -> dict[str, float]:
    """Return coverage on high-wind days (wind) or very sunny days (solar), and January and March wind."""
    frame, col = pick(
        frame=load(name=name), variant="kept_rows" if name.startswith("wind") else "all_rows"
    )
    out: dict[str, float] = {}
    day = frame.with_columns(day=pl.col("valid_time").dt.date())
    if name.startswith("wind"):
        peaks = day.group_by("series", "day").agg(pl.col(col).max().alias("peak"))
        high = (
            peaks.filter(pl.col("peak") > 0.8)
            .group_by("day")
            .len()
            .filter(pl.col("len") >= 2)["day"]
        )
        sub = day.filter(pl.col("day").is_in(high))
        v = losses(frame=sub, actual=col)
        out |= {
            "high_wind_days": float(sub["day"].n_unique()),
            "high_wind_cov": float(v["inside_q05_q95"].mean()),
            "high_wind_above_q99": float(v["above_q99"].mean()),
        }
        for m in ("2026-01", "2026-03"):
            s = frame.filter(pl.col("month") == m)
            out[f"cov_{m}"] = float(losses(frame=s, actual=col)["inside_q05_q95"].mean())
    else:
        p95 = (
            day.with_columns(ym=pl.col("valid_time").dt.strftime("%Y-%m"))
            .group_by("series", "ym")
            .agg(pl.col(col).quantile(0.95).alias("p95"))
        )
        d2 = day.with_columns(ym=pl.col("valid_time").dt.strftime("%Y-%m")).join(
            p95, on=["series", "ym"]
        )
        share = d2.group_by("series", "day").agg(
            (pl.col(col) >= 0.9 * pl.col("p95")).mean().alias("share")
        )
        sunny = share.filter(pl.col("share") >= 0.5).select("series", "day")
        sub = d2.join(sunny, on=["series", "day"])
        v = losses(frame=sub, actual=col)
        out |= {
            "sunny_series_days": float(sunny.height),
            "sunny_rows": float(sub.height),
            "sunny_cov": float(v["inside_q05_q95"].mean()),
            "sunny_above_q99": float(v["above_q99"].mean()),
        }
    return out


def main() -> None:
    """Write every score table."""
    SCORES.mkdir(parents=True, exist_ok=True)
    wind = [f"wind_{a}" for a in ("S", "PLP", "PL", "S_cal", "PLP_cal")] + [
        "wind_default2_old_capacity",
        "wind_default2_730",
        "wind_S_single",
        "wind_S_quarterly",
        "wind_climatology",
    ]
    solar = [f"solar_{a}" for a in ("S", "PLP", "PL")] + [
        "solar_a2_member_quarterly",
        "solar_a2_member_730",
        "solar_gam",
        "solar_climatology",
    ]
    wind = [
        n
        for n in wind
        if (common.TABLE_DIR / f"{n}.parquet").exists() or (E615 / f"{n}.parquet").exists()
    ]
    solar = [
        n
        for n in solar
        if (common.TABLE_DIR / f"{n}.parquet").exists() or (E615 / f"{n}.parquet").exists()
    ]
    for variant in ("cleaned_zeros", "kept_rows", "all_rows"):
        overall(names=wind, variant=variant).write_csv(SCORES / f"wind_{variant}_overall.csv")
        contrasts(names=wind, reference=WIND_REF, variant=variant).write_csv(
            SCORES / f"wind_{variant}_vs_default2.csv"
        )
        contrasts(
            names=[n for n in wind if n != "wind_climatology"], reference="wind_S", variant=variant
        ).write_csv(SCORES / f"wind_{variant}_vs_S.csv")
        tail_table(names=wind, variant=variant).write_csv(SCORES / f"wind_{variant}_tails.csv")
    overall(names=solar, variant="all_rows").write_csv(SCORES / "solar_overall.csv")
    contrasts(names=solar, reference=SOLAR_REF, variant="all_rows").write_csv(
        SCORES / "solar_vs_a2_member.csv"
    )
    contrasts(
        names=[n for n in solar if n != "solar_climatology"],
        reference="solar_S",
        variant="all_rows",
    ).write_csv(SCORES / "solar_vs_S.csv")
    tail_table(names=solar, variant="all_rows").write_csv(SCORES / "solar_tails.csv")
    specials = [{"candidate": n, **special_days(name=n)} for n in wind + solar]
    pl.DataFrame(specials, infer_schema_length=None).write_csv(SCORES / "special_days.csv")
    with pl.Config(tbl_rows=80, tbl_cols=14, float_precision=3, fmt_str_lengths=40):
        for f in (
            "wind_cleaned_zeros_overall",
            "solar_overall",
            "wind_cleaned_zeros_vs_default2",
            "solar_vs_a2_member",
        ):
            print(f)
            print(
                pl.read_csv(SCORES / f"{f}.csv").filter(pl.col("band") == "all (0-240 h)")
                if "band" in pl.read_csv(SCORES / f"{f}.csv").columns
                else ""
            )


if __name__ == "__main__":
    main()
