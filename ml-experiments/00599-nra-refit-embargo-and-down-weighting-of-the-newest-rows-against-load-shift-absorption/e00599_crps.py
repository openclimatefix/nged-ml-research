# ruff: noqa: E501
"""Experiment 599: event-free CRPS and tail coverage of E0, E1, E2w, and E3.

The event-free forecasts are each arm's fits in the no-event world `obs` (the cleaned observed
power), which the injected cases share as their `N` world. Actuals are the cleaned power
(`clean_stack`, NaN where flagged; those half-hours are dropped and counted). CRPS is twice the
trapezoid integral over the 11 stored quantile levels (0.01 to 0.99) of the pinball loss, in MW
(share of capacity times the Primary's capacity), with the quantiles sorted per row. The band is
set by the lead in hours (valid time minus forecast origin): 0 to 24 h, and 24 h to the end.

Writes `scores/crps_by_cell.parquet` (arm, group, Primary, month, band: mean CRPS, rows, coverage of
the q05-q95 band, and the shares of rows below q01 and q99) and prints the contrast tables:
the arm-minus-E0 difference per group, with a bootstrap interval that resamples Primary-by-month
cells within the group (2000 draws). The half-width of that interval is the minimum detectable effect.
"""

from typing import Final

import e00510_common as c510  # ty: ignore[unresolved-import]
import e00587_common as c587  # ty: ignore[unresolved-import]
import e00599_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
COLS: Final[tuple[str, ...]] = tuple(f"q{round(x * 100):02d}" for x in LEVELS)
N_DRAWS: Final[int] = 2000
GROUPS: Final[tuple[str, ...]] = ("MVA-only eight", "signed-MW six", "D15 and D18")
HOUR_US: Final[int] = 3_600_000_000


def crps(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return 2 x the trapezoid integral of the pinball loss over the stored levels."""
    q = np.sort(q, axis=1)
    lv = np.asarray(LEVELS)[None, :]
    d = actual[:, None] - q
    pin = np.where(d >= 0, d * lv, -d * (1 - lv))
    return 2.0 * ((pin[:, 1:] + pin[:, :-1]) / 2 * np.diff(np.asarray(LEVELS))[None, :]).sum(axis=1)


def arm_cells(*, arm: str, obs: np.ndarray, sm: pl.DataFrame) -> pl.DataFrame:
    """Return per (Primary, month, band) mean CRPS and coverage of one arm's fits on `obs`."""
    d = common.world_dir(arm=arm, world="obs") / "work" / "fits" / common.TAG
    frames = []
    for f in sorted(d.glob("*.parquet")):
        month, day, seed = f.stem.split("_")
        frames.append(
            pl.read_parquet(f).with_columns(month=pl.lit(month), seed=pl.lit(seed), day=pl.lit(day))
        )
    x = pl.concat(frames).join(
        sm.select("time_series_id", "primary", "capacity_mw", "group"), on="time_series_id"
    )
    idx = (
        (x["valid_time"].dt.epoch("us") - int(c587.GRID_START.timestamp() * 1e6)) // 1_800_000_000
    ).to_numpy()
    actual = obs[x["primary"].to_numpy(), idx].astype(np.float64)
    ok = np.isfinite(actual)
    n_dropped = int((~ok).sum())
    q = x.select(COLS).to_numpy().astype(np.float64)
    score = np.where(
        ok, crps(actual=np.nan_to_num(actual), q=q) * x["capacity_mw"].to_numpy(), np.nan
    )
    lead_h = (
        x["valid_time"].dt.epoch("us") - x["power_fcst_init_time"].dt.epoch("us")
    ).to_numpy() / HOUR_US
    x = x.with_columns(
        crps=pl.Series(score),
        band=pl.Series(np.where(lead_h < 24, "0-24 h", "24 h+")),
        inside=pl.Series(((actual >= q[:, 1]) & (actual <= q[:, 9])).astype(float)),
        below_q01=pl.Series((actual < q[:, 0]).astype(float)),
        above_q99=pl.Series((actual > q[:, 10]).astype(float)),
    ).filter(pl.col("crps").is_not_null() & ~pl.col("crps").is_nan())
    return (
        x.group_by("group", "primary", "month", "band")
        .agg(
            pl.col("crps").mean(),
            pl.len().alias("rows"),
            pl.col("inside").mean().alias("coverage_q05_q95"),
            pl.col("below_q01").mean(),
            pl.col("above_q99").mean(),
        )
        .with_columns(arm=pl.lit(arm), dropped=pl.lit(n_dropped))
    )


def boot(*, diff: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Return the mean and the 95% percentile interval over resampled cells."""
    rng = np.random.default_rng(seed)
    d = rng.choice(diff, size=(N_DRAWS, len(diff))).mean(axis=1)
    return float(diff.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    """Write the per-cell table and print the contrasts against E0."""
    sm = c587.site_map()
    obs = c510.clean_stack()
    cells = pl.concat([arm_cells(arm=a, obs=obs, sm=sm) for a in common.ALL_ARMS])
    # keep only the months that every arm has, so the contrasts compare the same rows
    keep = (
        cells.group_by("month")
        .agg(pl.col("arm").n_unique())
        .filter(pl.col("arm") == len(common.ALL_ARMS))["month"]
    )
    cells = cells.filter(pl.col("month").is_in(keep))
    out = common.OUT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(out / "crps_by_cell.parquet")
    lines = [
        f"months scored: {sorted(keep.to_list())}",
        "dropped half-hours (flagged actuals) per arm: "
        + str(cells.group_by("arm").agg(pl.col("dropped").first()).sort("arm").rows()),
    ]
    for band in ("all", "0-24 h", "24 h+"):
        lines.append(f"\n## CRPS (MW) and difference from E0, band {band}")
        lines.append(
            "| group | arm | mean CRPS | minus E0 (95% interval) | MDE | coverage q05-q95 | below q01 | above q99 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for g in GROUPS:
            sel = cells.filter(pl.col("group") == g)
            if band != "all":
                sel = sel.filter(pl.col("band") == band)
            # cell weights: rows; per (Primary, month) the mean CRPS over bands is row-weighted
            cell = sel.group_by("arm", "primary", "month").agg(
                (pl.col("crps") * pl.col("rows")).sum() / pl.col("rows").sum(),
                (pl.col("coverage_q05_q95") * pl.col("rows")).sum() / pl.col("rows").sum(),
                (pl.col("below_q01") * pl.col("rows")).sum() / pl.col("rows").sum(),
                (pl.col("above_q99") * pl.col("rows")).sum() / pl.col("rows").sum(),
            )
            base = cell.filter(pl.col("arm") == "E0")
            for a in common.ALL_ARMS:
                x = cell.filter(pl.col("arm") == a)
                if a == "E0":
                    lines.append(
                        f"| {g} | E0 | {x['crps'].mean():.4f} | - | - | {x['coverage_q05_q95'].mean():.3f} | {x['below_q01'].mean():.4f} | {x['above_q99'].mean():.4f} |"
                    )
                    continue
                j = x.join(base, on=["primary", "month"], suffix="_b")
                m, lo, hi = boot(diff=(j["crps"] - j["crps_b"]).to_numpy())
                lines.append(
                    f"| {g} | {a} | {x['crps'].mean():.4f} | {m:+.4f} ({lo:+.4f} to {hi:+.4f}) | {(hi - lo) / 2:.4f} | {x['coverage_q05_q95'].mean():.3f} | {x['below_q01'].mean():.4f} | {x['above_q99'].mean():.4f} |"
                )
    text = "\n".join(lines)
    (out / "crps_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
