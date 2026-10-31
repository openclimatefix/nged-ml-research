"""Score the arms by tercile of the ensemble spread, from the stored forecasts (no new fits).

Rows are all stored half-hours of days 1 to 10 with a finite actual, not the scorer's shared slice
table. The spread of a row is the standard deviation across the 51 ensemble members of
`temperature_2m` (all hours) or of `shortwave` (June to August sunny half-hours) at the row's
site, forecast run, and lead. Terciles are cut within each Primary and each lead band of
experiment 389, so a bin compares like with like. Sunny means the ensemble-mean shortwave is in
the top quarter of all of that Primary's June-to-August rows, night included, so it selects
roughly the brighter half of daytime rows. Intervals resample the calendar months of the valid
time (or, for June to August, whole days).

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00389-* DATA_PATH_INTERNAL=... \
        uv run --no-sync python ml-experiments/00394-*/e00394_bins.py
"""

from pathlib import Path
from typing import Final

import e00389_calibrate as cal  # ty: ignore[unresolved-import]
import e00389_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00394-xgboost-ensemble-mean-and-std-columns-separately" / "work"
)
ARMS: Final[tuple[str, ...]] = ("S0M_base", "S1M_base", "MEAN_base", "STD_base")
LEVELS: Final[dict[str, float]] = {"q05": 0.05, "q95": 0.95}
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed-MW six": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "MVA-only eight": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
}
N_BOOT: Final[int] = 5_000
MIN_UNITS: Final[int] = 6
HEADLINE_HOURS: Final[float] = 240.0


def _pinball(*, y: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    d = y - q
    return np.maximum(level * d, (level - 1.0) * d)


def load() -> pl.DataFrame:
    """Return one row per scored half-hour with every arm's q05 and q95 and the spreads."""
    stored = (
        pl.read_parquet(cal.STORED_BG2)
        .select(*cal.KEYS, "actual", "site")
        .filter(pl.col("actual").is_finite())
    )
    frame = None
    for arm in ARMS:
        path = WORK / "arms" / f"{arm}.parquet"
        if not path.exists():
            continue
        part = pl.read_parquet(path).select(
            *cal.KEYS,
            "lead_hours",
            pl.col("q05").alias(f"{arm}_q05"),
            pl.col("q95").alias(f"{arm}_q95"),
        )
        frame = (
            part if frame is None else frame.join(part.drop("lead_hours"), on=cal.KEYS, how="inner")
        )
    if frame is None:
        msg = "no arm files found"
        raise FileNotFoundError(msg)
    frame = frame.filter(pl.col("lead_hours") <= HEADLINE_HOURS).join(
        stored, on=cal.KEYS, how="inner", validate="1:1"
    )
    pieces = []
    for site in sorted(frame["site"].unique().to_list()):
        part = fit.attach_spread(frame=frame.filter(pl.col("site") == site), site=site)
        pieces.append(part)
    out = pl.concat(pieces).filter(
        pl.all_horizontal(pl.col(c).is_finite() for c in fit.SPREAD_COLUMNS)
    )
    return out.with_columns(
        band=pl.Series(cal.band_of(lead=out["lead_hours"].to_numpy())),
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        day=pl.col("valid_time").dt.strftime("%Y-%m-%d"),
    )


def tercile(*, frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Add `bin` (0 low, 1 middle, 2 high) cut within each Primary and lead band."""
    return frame.with_columns(
        bin=(pl.col(column).rank("ordinal").over("site", "band") - 1)
        * 3
        // pl.len().over("site", "band")
    )


def _boot(
    *,
    values: np.ndarray,
    base: np.ndarray | None,
    idx: list[np.ndarray],
    draws: list[np.ndarray],
) -> tuple[float, float, float]:
    """Return the mean over units (minus `base` if given) and its 99% resampling interval."""
    per = np.array([values[i].mean() if len(i) else np.nan for i in idx])
    if base is not None:
        per = per - np.array([base[i].mean() if len(i) else np.nan for i in idx])
    means = np.array([np.nanmean(per[d]) for d in draws])
    low, high = np.quantile(means, [0.005, 0.995])
    return float(np.nanmean(per)), float(low), float(high)


def summarise(*, frame: pl.DataFrame, unit: str, title: str) -> None:
    """Print coverage and pinball by group, bin, and arm, with bootstrap intervals."""
    rng = np.random.default_rng(0)
    arms = [a for a in ARMS if f"{a}_q05" in frame.columns]
    for gname, labels in GROUPS.items():
        sub = frame.filter(pl.col("site").is_in(list(labels)))
        print(f"\n== {title}, {gname} ==")
        for b, name in enumerate(("low", "middle", "high")):
            part = sub.filter(pl.col("bin") == b)
            units = part[unit].unique().sort().to_list()
            if not units:
                continue
            thin = f" (too thin: fewer than {MIN_UNITS} {unit}s)" if len(units) < MIN_UNITS else ""
            print(
                f"bin {b} ({name} spread): {part.height:,} half-hours, {len(units)} {unit}s{thin}"
            )
            y = part["actual"].to_numpy()
            column = part[unit].to_numpy()
            idx = [np.flatnonzero(column == u) for u in units]
            draws = [rng.integers(0, len(units), len(units)) for _ in range(N_BOOT)]
            pins, covs = {}, {}
            for a in arms:
                for q, level in LEVELS.items():
                    qv = part[f"{a}_{q}"].to_numpy().astype(np.float64)
                    pins[(a, q)] = _pinball(y=y, q=qv, level=level) * 100.0
                    covs[(a, q)] = (y <= qv).astype(np.float64)
            for q in LEVELS:
                for a in arms:
                    base = None if a == "S0M_base" else pins[("S0M_base", q)]
                    p = _boot(values=pins[(a, q)], base=base, idx=idx, draws=draws)
                    c = _boot(values=covs[(a, q)], base=None, idx=idx, draws=draws)
                    tag = "level" if a == "S0M_base" else "minus S0M"
                    print(
                        f"  {q} {a:9s} coverage {c[0]:.3f} ({c[1]:.3f} to {c[2]:.3f}); "
                        f"pinball {tag} {p[0]:+.4f} ({p[1]:+.4f} to {p[2]:+.4f})"
                    )


def main() -> None:
    """Print the tercile tables."""
    frame = load()
    print(f"scored half-hours, days 1 to 10: {frame.height:,}")
    temp = tercile(frame=frame, column="ens_temp_std")
    summarise(frame=temp, unit="month", title="all hours, terciles of temperature spread")
    summer = frame.filter(pl.col("valid_time").dt.month().is_in([6, 7, 8]))
    cutoff = summer.group_by("site").agg(pl.col("ens_sw_mean").quantile(0.75).alias("top"))
    sunny = summer.join(cutoff, on="site").filter(pl.col("ens_sw_mean") >= pl.col("top"))
    sunny = tercile(frame=sunny, column="ens_sw_std")
    summarise(frame=sunny, unit="day", title="June to August sunny, terciles of shortwave spread")


if __name__ == "__main__":
    main()
