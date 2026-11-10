"""Band-width steps at the model refits under a fixed-lookback scale, by seed handling (CPU only).

PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-* DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00689-*/e00689_boundary.py

Reads the stored normal-running Primaries file (seed mean) and the per-seed quantiles of experiment
667 (`P7_D50_<month>/work/fits/P5/<month>_d0_s{0,1}.parquet`). Fits the scale of experiment 683 (its
rule is copied here with the lookback as an argument) and writes `tables/*.csv` under this
experiment's data folder. Arms: 0 stored; A 90-day scale on the seed mean; B scale fitted per seed
and applied to the same seed, then the two seeds' quantiles averaged; B-cross the scale of the other
seed; C 45 and 180 days on the seed mean. Boundaries are the starts of 2026-01 to 2026-06.
"""

from itertools import pairwise
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00683_calibrate import (
    DELAY_HALF_HOURS,
    MIN_FIT_DAYS,
    SCALE_BOUNDS,
    TARGET_COVERAGE,
    band_index,
    load_truth,
    z_scores,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00689-fixed-lookback-width-scale-versus-seed-averaging-at-refit-boundaries"
    / "tables"
)
EXP667: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
)
KEYS: Final[list[str]] = ["time_series_id", "valid_time", "power_fcst_init_time"]
BOUNDARIES: Final[tuple[str, ...]] = tuple(f"2026-{m:02d}" for m in range(1, 7))
SCORED: Final[tuple[str, str]] = ("2025-12", "2026-06")
STEP: Final[float] = 1.10
PASS_LOW: Final[float] = 0.90
"""A calibrated ratio passes if it lies in PASS_LOW to STEP (the plan's 0.90 to 1.10)."""
WINDOW_DAYS: Final[int] = 7
N_BOOT: Final[int] = 1000
RNG: Final = np.random.default_rng(689)


def load_rows(*, frame: pl.DataFrame, ids: list[int], cap: np.ndarray, grid: np.ndarray) -> dict:
    """Return arrays (percent of capacity) of q05, q50, q95, actual, and the row indices."""
    index = {t: i for i, t in enumerate(ids)}
    row = frame["time_series_id"].cast(pl.Int64).replace_strict(index).to_numpy()
    vidx = lib.grid_index(times=frame["valid_time"])
    oidx = lib.grid_index(times=frame["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        pad = int(vidx.max()) + 1 - grid.shape[1]
        grid = np.pad(grid, ((0, 0), (0, pad)), constant_values=np.nan)
    scale = 100.0 / cap[row]
    q = frame.select("q05", "q50", "q95").to_numpy().astype(np.float64) * scale[:, None]
    y = grid[row, vidx] * scale
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    return {
        "row": row,
        "vidx": vidx,
        "day": oidx // 48,
        "band": band_index(lead_h=(vidx - oidx) / 2.0),
        "q": q,
        "qfit": q,
        "y": np.where(ok, y, 0.0),
        "ok": ok,
        "month": frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
    }


def fit_scales(*, d: dict, month: str, lookback: int, n: int) -> np.ndarray:
    """Return the (series, band) scale for an origin month, fitted on the previous `lookback` days."""
    start = pl.select(pl.datetime(int(month[:4]), int(month[5:]), 1, time_zone="UTC")).to_series()
    cut = int(lib.grid_index(times=start)[0]) - DELAY_HALF_HOURS
    win = (d["vidx"] <= cut) & (d["vidx"] > cut - 48 * lookback) & d["ok"]
    out = np.ones((n, 4))
    for i in range(n):
        for b in range(4):
            sel = win & (d["row"] == i) & (d["band"] == b)
            if len(np.unique(d["vidx"][sel] // 48)) < MIN_FIT_DAYS:
                continue
            q = d["qfit"][sel]
            z = z_scores(y=d["y"][sel], q05=q[:, 0], q50=q[:, 1], q95=q[:, 2])
            z = z[np.isfinite(z)]
            if len(z):
                out[i, b] = np.clip(np.quantile(z, TARGET_COVERAGE), *SCALE_BOUNDS)
    return out


def apply(*, d: dict, fit: dict, lookback: int, nonneg: np.ndarray, n: int) -> np.ndarray:
    """Return lo and hi (rows x 2) of the scaled q05 and q95, fitted on `fit`, applied to `d`."""
    lo, hi = d["q"][:, 0].copy(), d["q"][:, 2].copy()
    for m in sorted(set(d["month"])):
        s = fit_scales(d=fit, month=m, lookback=lookback, n=n)
        sel = d["month"] == m
        f = s[d["row"][sel], d["band"][sel]]
        med = d["q"][sel, 1]
        lo[sel] = med - f * (med - d["q"][sel, 0])
        hi[sel] = med + f * (d["q"][sel, 2] - med)
    return np.column_stack([lo, hi])


def clamp(*, lohi: np.ndarray, d: dict, nonneg: np.ndarray) -> np.ndarray:
    """Clamp the lower quantile of the MVA Primaries at zero (after any seed averaging)."""
    out = lohi.copy()
    out[:, 0] = np.where(nonneg[d["row"]], np.maximum(out[:, 0], 0.0), out[:, 0])
    return out


def daily(*, d: dict, band_lohi: np.ndarray, n: int) -> pl.DataFrame:
    """Return per series, origin day, band: mean width, rows scored, rows covered, month."""
    keep = d["ok"]
    cover = (d["y"] >= band_lohi[:, 0]) & (d["y"] <= band_lohi[:, 1])
    frame = pl.DataFrame(
        {
            "row": d["row"][keep],
            "day": d["day"][keep],
            "band": d["band"][keep],
            "month": d["month"][keep],
            "width": (band_lohi[:, 1] - band_lohi[:, 0])[keep],
            "cover": cover[keep].astype(np.float64),
        }
    )
    return frame.group_by("row", "day", "band", "month").agg(
        width=pl.col("width").mean(), n=pl.len(), covered=pl.col("cover").sum()
    )


def previous_month(month: str) -> str:
    """Return the calendar month before `month` (YYYY-MM)."""
    y, m = int(month[:4]), int(month[5:])
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"


def ratios(*, dl: pl.DataFrame, ids: list[int]) -> tuple[pl.DataFrame, dict]:
    """Return the boundary ratio per cell and the day vectors needed for the bootstrap."""
    rows, vecs = [], {}
    for i, tid in enumerate(ids):
        for b in range(4):
            part = dl.filter((pl.col("row") == i) & (pl.col("band") == b)).sort("day")
            months = part["month"].unique(maintain_order=True).to_list()
            for prev, cur in pairwise(months):
                if cur not in BOUNDARIES or prev != previous_month(cur):
                    continue
                bef = part.filter(pl.col("month") == prev).tail(WINDOW_DAYS)["width"].to_numpy()
                aft = part.filter(pl.col("month") == cur).head(WINDOW_DAYS)["width"].to_numpy()
                vecs[(tid, cur, b)] = (bef, aft)
                rows.append(
                    {
                        "time_series_id": tid,
                        "boundary": cur,
                        "band": b,
                        "ratio": float(np.median(aft) / np.median(bef)),
                    }
                )
    return pl.DataFrame(rows), vecs


def boot_share(*, vecs: dict, cells: list, lo: float, hi: float) -> tuple[float, float, float]:
    """Return the share of `cells` with ratio in [lo, hi] and a basic 95% day-block interval.

    One set of uniform draws per boundary and replicate picks the days in the before and after
    windows of every series and band at that boundary, because those cells share their origin days
    and their refit. The interval is the percentile interval. Cells are chosen because the stored
    ratio is extreme, so resampled days regress towards 1 and the interval sits above the point
    share for the stored arm; read the interval as a spread, not as a bias-free bound.
    """
    boundaries = sorted({c[1] for c in cells})
    u = {
        (bd, w): RNG.random((N_BOOT, WINDOW_DAYS)) for bd in boundaries for w in ("before", "after")
    }
    ratio = np.empty((N_BOOT, len(cells)))
    for k, c in enumerate(cells):
        bef, aft = vecs[c]
        bi = (u[(c[1], "before")] * len(bef)).astype(int)
        ai = (u[(c[1], "after")] * len(aft)).astype(int)
        ratio[:, k] = np.median(aft[ai], axis=1) / np.median(bef[bi], axis=1)
    shares = ((ratio >= lo) & (ratio <= hi)).mean(axis=1)
    point = float(
        np.mean([lo <= np.median(vecs[c][1]) / np.median(vecs[c][0]) <= hi for c in cells])
    )
    q_lo, q_hi = np.quantile(shares, [0.025, 0.975])
    return point, float(q_lo), float(q_hi)


def shift_to_contract(*, frame: pl.DataFrame, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return a per-seed fit in the stored file's units and timing.

    The fits are native 09:00 UTC origins in fractions of capacity. The stored file's 12:00 UTC
    origin is the same fit with the origin moved 3 hours later (experiment 667's timing check), its
    quantiles multiplied by capacity, and the half-hours before the new origin dropped. The clamp
    at zero of the eight MVA Primaries is applied after averaging in the stored file, so it is
    applied after scaling in every arm here and not to the seed fits.
    """
    cols = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
    cap = capacity.select(pl.col("time_series_id").cast(pl.Int32), "capacity")
    return (
        frame.with_columns(pl.col("power_fcst_init_time") + pl.duration(hours=3))
        .filter(pl.col("valid_time") >= pl.col("power_fcst_init_time"))
        .join(cap, on="time_series_id")
        .with_columns([(pl.col(c) * pl.col("capacity")).alias(c) for c in cols])
        .drop("capacity")
    )


def main() -> None:
    """Run all arms and write the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    stored = pl.read_parquet(lib.FOLDER / "primaries.parquet").filter(
        pl.col("NRA_or_prevailing") == "NRA"
    )
    ids = sorted(stored["time_series_id"].unique().to_list())
    capacity = capacity.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")
    cap = capacity["capacity"].to_numpy().astype(np.float64)
    nonneg = np.array([u == "MVA" for u in capacity["power_unit"].to_list()])
    n = len(ids)
    grid = load_truth(name="primaries", ids=ids)
    seeds = []
    for s in (0, 1):
        parts = [
            pl.read_parquet(p)
            for p in sorted(EXP667.glob(f"P7_D50_*/work/fits/P5/*_d0_s{s}.parquet"))
        ]
        seeds.append(
            shift_to_contract(
                frame=pl.concat(parts).unique(subset=KEYS, keep="first", maintain_order=True),
                capacity=capacity,
            )
        )
    mean_check = (
        stored.select(KEYS + ["q05", "q95"])
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
        .join(seeds[0].select(KEYS + ["q05", "q95"]), on=KEYS, suffix="_0")
        .join(seeds[1].select(KEYS + ["q05", "q95"]), on=KEYS, suffix="_1")
        .select(
            rows=pl.len(),
            max_abs_q05_where_stored_positive=(
                ((pl.col("q05_0") + pl.col("q05_1")) / 2 - pl.col("q05"))
                .abs()
                .filter(pl.col("q05") > 0)
                .max()
            ),
            max_abs_q95=((pl.col("q95_0") + pl.col("q95_1")) / 2 - pl.col("q95")).abs().max(),
        )
    )
    mean_check.write_csv(OUT / "seed_mean_check.csv")
    print(mean_check)
    # Common rows: stored keys present in both seed files, so every arm uses identical rows.
    common = (
        stored.select(KEYS)
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
        .join(seeds[0].select(KEYS), on=KEYS)
        .join(seeds[1].select(KEYS), on=KEYS)
    )
    st = stored.with_columns(pl.col("time_series_id").cast(pl.Int32)).join(common, on=KEYS)
    sd = [x.join(common, on=KEYS) for x in seeds]
    data_mean = load_rows(frame=st, ids=ids, cap=cap, grid=grid)
    data_seed = [load_rows(frame=x, ids=ids, cap=cap, grid=grid) for x in sd]
    # The seed files are joined to `common` in a different row order, so align by sorting.
    for frame, label in ((st, "stored"), (sd[0], "s0"), (sd[1], "s1")):
        assert frame.height == common.height, label
    order = [f.select(KEYS).with_row_index().sort(KEYS)["index"].to_numpy() for f in (st, *sd)]
    for key in data_mean:
        data_mean[key] = data_mean[key][order[0]]
        for s in (0, 1):
            data_seed[s][key] = data_seed[s][key][order[s + 1]]
    for s in (0, 1):
        # The stored file clamps after averaging; the seed fits are fitted on a clamped copy.
        data_seed[s]["qfit"] = data_seed[s]["q"].copy()
        data_seed[s]["qfit"][:, 0] = np.where(
            nonneg[data_seed[s]["row"]],
            np.maximum(data_seed[s]["q"][:, 0], 0.0),
            data_seed[s]["q"][:, 0],
        )
    keep = data_mean["month"] <= SCORED[1]
    for dd in (data_mean, *data_seed):
        for key in dd:
            dd[key] = dd[key][keep]
    arms: dict[str, np.ndarray] = {}
    arms["0 stored"] = np.column_stack([data_mean["q"][:, 0], data_mean["q"][:, 2]])
    for lb in (45, 90, 180):
        name = "A 90 days" if lb == 90 else f"C {lb} days"
        arms[name] = clamp(
            lohi=apply(d=data_mean, fit=data_mean, lookback=lb, nonneg=nonneg, n=n),
            d=data_mean,
            nonneg=nonneg,
        )
    own = [apply(d=data_seed[s], fit=data_seed[s], lookback=90, nonneg=nonneg, n=n) for s in (0, 1)]
    cross = [
        apply(d=data_seed[s], fit=data_seed[1 - s], lookback=90, nonneg=nonneg, n=n) for s in (0, 1)
    ]
    arms["B per seed, averaged"] = clamp(lohi=(own[0] + own[1]) / 2, d=data_mean, nonneg=nonneg)
    arms["B-cross"] = clamp(lohi=(cross[0] + cross[1]) / 2, d=data_mean, nonneg=nonneg)
    arms["0 seed mean of s0 and s1"] = clamp(
        lohi=np.column_stack(
            [(data_seed[0]["q"][:, c] + data_seed[1]["q"][:, c]) / 2 for c in (0, 2)]
        ),
        d=data_mean,
        nonneg=nonneg,
    )
    d = data_mean
    res, vec_all, cov_rows = {}, {}, []
    for name, lohi in arms.items():
        dl = daily(d=d, band_lohi=lohi, n=n)
        r, vecs = ratios(dl=dl, ids=ids)
        res[name], vec_all[name] = r, vecs
        sc = dl.filter(pl.col("month").is_between(pl.lit(SCORED[0]), pl.lit(SCORED[1])))
        for b in range(4):
            g = (
                sc.filter(pl.col("band") == b)
                .group_by(week=pl.col("day") // 7)
                .agg(pl.col("covered").sum(), pl.col("n").sum())
            )
            cs, ns = g["covered"].to_numpy(), g["n"].to_numpy()
            idx = RNG.integers(0, len(cs), (N_BOOT, len(cs)))
            cv = cs[idx].sum(axis=1) / ns[idx].sum(axis=1)
            cov_rows.append(
                {
                    "arm": name,
                    "band": b,
                    "coverage": float(cs.sum() / ns.sum()),
                    "lo": float(np.quantile(cv, 0.025)),
                    "hi": float(np.quantile(cv, 0.975)),
                }
            )
    pl.DataFrame(cov_rows).sort("arm", "band").write_csv(OUT / "coverage.csv")
    base = res["0 stored"]
    affected = base.filter((pl.col("ratio") > STEP) | (pl.col("ratio") < 1 / STEP))
    cells = [
        (r["time_series_id"], r["boundary"], r["band"]) for r in affected.iter_rows(named=True)
    ]
    summary = []
    for name in arms:
        sh = boot_share(vecs=vec_all[name], cells=cells, lo=PASS_LOW, hi=STEP)
        summary.append(
            {"arm": name, "affected_cells": len(cells), "share": sh[0], "lo": sh[1], "hi": sh[2]}
        )
    a = res["A 90 days"].rename({"ratio": "ratio_A"})
    for name in ("B per seed, averaged", "B-cross"):
        j = (
            affected.select("time_series_id", "boundary", "band")
            .join(a, on=["time_series_id", "boundary", "band"])
            .join(res[name].rename({"ratio": "ratio_o"}), on=["time_series_id", "boundary", "band"])
        )
        summary.append(
            {
                "arm": f"{name} within 0.05 of A",
                "affected_cells": j.height,
                "share": float(((j["ratio_o"] - j["ratio_A"]).abs() <= 0.05).mean()),
                "lo": None,
                "hi": None,
            }
        )
    pl.DataFrame(summary).write_csv(OUT / "summary.csv")
    allr = pl.concat([r.with_columns(arm=pl.lit(k)) for k, r in res.items()])
    allr.write_csv(OUT / "ratios.csv")
    affected.write_csv(OUT / "affected_cells.csv")
    pl.Config.set_tbl_rows(60)
    print(pl.DataFrame(summary))
    print(pl.DataFrame(cov_rows).sort("arm", "band"))


if __name__ == "__main__":
    main()
