"""Part E of experiment 395: a lag-encroachment sweep on experiment 391's stored `b0` boosters.

Prediction only. For each Primary and each monthly booster, one clean forecast run (the 15th of the
refit month) is predicted again with a synthetic load-shift event of depth D (2%, 5%, 10% of
capacity, both signs) added to the first `k` of the three lagged-power inputs, ordered by recency:
the newest usable reading (k = 1), the day-earlier lag (k = 2), the week-earlier lag (k = 3). A lag
reads inside the event when it falls after the event's onset, so a longer event reaches further
back. Weather, calendar, and every other column stay unchanged. `b0` has no surprise or detector
columns, so nothing else needs recomputing.

    ... e00395_part_e.py run [--sites D5 D9 ...] | report
"""
# ruff: noqa: E501

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00391_fit as f391  # ty: ignore[unresolved-import]
import e00395_build as b395  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb

WORK: Final[Path] = b395.WORK
OUT: Final[Path] = WORK / "sweep"
BOOSTERS: Final[Path] = f391.BOOSTER_DIR / "b0"
DEPTHS: Final[tuple[float, ...]] = (0.02, 0.05, 0.10)
SIGNS: Final[tuple[int, ...]] = (1, -1)
LEVELS_K: Final[tuple[int, ...]] = (0, 1, 2, 3)
RUN_DAY: Final[int] = 15
NTHREAD: Final[int] = 2
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24h", 0, 24), ("24-72h", 24, 72), ("72-168h", 72, 168), ("168-240h", 168, 240),
)  # fmt: skip
Q: Final[dict[str, int]] = {c: i for i, c in enumerate(common.LEVEL_COLUMNS)}


def predict_quantiles(*, model: xgb.Booster, matrix: np.ndarray, members: int) -> np.ndarray:
    """Return the mixture's 11 quantiles per valid time for a matrix of member rows."""
    raw = model.predict(xgb.DMatrix(matrix))
    cube = raw.reshape(-1, members, len(common.LEVELS))
    return common.mixture_quantiles(member_quantiles=cube)


def run_site(*, site: str) -> pl.DataFrame:
    """Return the sweep's aggregated metrics for one Primary."""
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    arm = f391.ARMS["b0"]
    columns = arm.columns()
    lag_idx = [columns.index(c) for c in common.LAG_COLUMNS]
    masks = f391.event_masks()
    power = arrays[site].power
    out = []
    for month in range(7, 19):
        year, mon = (2025, month) if month <= 12 else (2026, month - 12)
        refit = datetime(year, mon, 1, tzinfo=UTC)
        origin = datetime(year, mon, RUN_DAY, 9, tzinfo=UTC)
        path = BOOSTERS / f"{site}_{refit:%Y%m%d}.ubj"
        model = xgb.Booster()
        model.load_model(path)
        model.set_param({"nthread": NTHREAD, "device": "cpu"})
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row, arrays=arrays[site], a7=False,
        )  # fmt: skip
        keys = prepared.select(*common.KEYS).unique(maintain_order=True)
        members = prepared["ensemble_member"].n_unique()
        target = np.asarray(common.grid_index(times=keys["valid_time"]))
        origin_idx = np.asarray(common.grid_index(times=keys["power_fcst_init_time"]))
        actual = common.take(values=power, index=target)
        clean = (
            np.isfinite(actual)
            & ~masks[site][target]
            & ~f391.lag_flags(mask=masks[site], target=target, origin=origin_idx).any(axis=1)
        )
        lead = (
            keys["valid_time"] - keys["power_fcst_init_time"]
        ).dt.total_minutes().to_numpy() / 60
        matrix = prepared.select(columns).to_numpy().astype(np.float32)
        base = predict_quantiles(model=model, matrix=matrix, members=members)
        for sign in SIGNS:
            for depth in DEPTHS:
                for k in LEVELS_K:
                    shifted = matrix.copy()
                    for i in range(k):
                        shifted[:, lag_idx[i]] += sign * depth
                    quant = (
                        base
                        if k == 0
                        else predict_quantiles(model=model, matrix=shifted, members=members)
                    )
                    event = actual + sign * depth
                    for band, lo, hi in BANDS:
                        sel = clean & (lead >= lo) & (lead < hi)
                        if not sel.any():
                            continue
                        q, q0 = quant[sel], base[sel]
                        out.append({
                            "site": site, "month": mon + 12 * (year - 2025), "band": band,
                            "sign": sign, "depth": depth, "k": k, "n": int(sel.sum()),
                            "median_shift": float(((q[:, Q["q50"]] - q0[:, Q["q50"]]) / (sign * depth)).mean()),
                            "w90": float((q[:, Q["q95"]] - q[:, Q["q05"]]).mean()),
                            "w90_0": float((q0[:, Q["q95"]] - q0[:, Q["q05"]]).mean()),
                            "w98": float((q[:, Q["q99"]] - q[:, Q["q01"]]).mean()),
                            "w98_0": float((q0[:, Q["q99"]] - q0[:, Q["q01"]]).mean()),
                            "clean_in90": float(((actual[sel] >= q[:, Q["q05"]]) & (actual[sel] <= q[:, Q["q95"]])).mean()),
                            "event_in90": float(((event[sel] >= q[:, Q["q05"]]) & (event[sel] <= q[:, Q["q95"]])).mean()),
                            "clean_in98": float(((actual[sel] >= q[:, Q["q01"]]) & (actual[sel] <= q[:, Q["q99"]])).mean()),
                            "event_in98": float(((event[sel] >= q[:, Q["q01"]]) & (event[sel] <= q[:, Q["q99"]])).mean()),
                            "reach95": float((q[:, Q["q95"]] >= event[sel]).mean() if sign > 0 else (q[:, Q["q05"]] <= event[sel]).mean()),
                            "reach99": float((q[:, Q["q99"]] >= event[sel]).mean() if sign > 0 else (q[:, Q["q01"]] <= event[sel]).mean()),
                        })  # fmt: skip
        print(site, refit.date(), "done", flush=True)
    return pl.DataFrame(out)


def run(*, sites: list[str]) -> None:
    """Run the sweep for the named Primaries, skipping those already written."""
    OUT.mkdir(parents=True, exist_ok=True)
    for site in sites:
        target = OUT / f"{site}.parquet"
        if target.exists():
            continue
        frame = run_site(site=site)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)


def report() -> None:
    """Print the sweep tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(220)
    sweep = pl.concat([pl.read_parquet(p) for p in sorted(OUT.glob("*.parquet"))])
    six = ["D5", "D9", "D10", "D11", "D14", "D16"]
    sweep = sweep.with_columns(
        group=pl.when(pl.col("site").is_in(six)).then(pl.lit("six")).otherwise(pl.lit("other"))
    )
    agg = lambda by: (  # noqa: E731
        sweep.group_by(by)
        .agg(
            n=pl.col("n").sum(),
            median_share=(pl.col("median_shift") * pl.col("n")).sum() / pl.col("n").sum(),
            w90_ratio=(pl.col("w90") * pl.col("n")).sum() / (pl.col("w90_0") * pl.col("n")).sum(),
            w98_ratio=(pl.col("w98") * pl.col("n")).sum() / (pl.col("w98_0") * pl.col("n")).sum(),
            clean_in90=(pl.col("clean_in90") * pl.col("n")).sum() / pl.col("n").sum(),
            event_in90=(pl.col("event_in90") * pl.col("n")).sum() / pl.col("n").sum(),
            clean_in98=(pl.col("clean_in98") * pl.col("n")).sum() / pl.col("n").sum(),
            event_in98=(pl.col("event_in98") * pl.col("n")).sum() / pl.col("n").sum(),
            reach95=(pl.col("reach95") * pl.col("n")).sum() / pl.col("n").sum(),
            reach99=(pl.col("reach99") * pl.col("n")).sum() / pl.col("n").sum(),
        )
        .sort(by)
        .with_columns(pl.selectors.float().round(3))
    )
    print("== E1: by k and lead band (both signs, depths 2/5/10%, all Primaries) ==")
    print(agg(["k", "band"]))
    print("\n== E2: by k and depth (all bands) ==")
    print(agg(["k", "depth"]))
    print("\n== E3: by k and sign (all bands) ==")
    print(agg(["k", "sign"]))
    print("\n== E4: by k and group ==")
    print(agg(["k", "group"]))
    print("\n== E5: per Primary, k=3 vs k=1, median share ==")
    print(
        agg(["site", "k"])
        .filter(pl.col("k").is_in([1, 3]))
        .select("site", "k", "median_share", "w90_ratio")
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "report"])
    parser.add_argument("--sites", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "run":
        run(sites=args.sites)
    else:
        report()
