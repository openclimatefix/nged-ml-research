# ruff: noqa: E501, D103, D101
"""Experiment 632: fold statistics, test rows, and the stored comparators P2-pre and K3-pre.

**Fold statistics.** `fold_stats` calls experiment 428's `build_fold` once per fold month, the same
function that gave P2-pre and K3-pre their input standardisation, target mean and standard
deviation, and lag statistics, and caches the small result. Seq uses the same numbers.

**Comparators.** P2-pre (experiment 591's `AuxLagMdn` (shared lag encoder) in stage `pre`, weights
from experiment 613's chain) and K3-pre (experiment 565's `OutputOffsetDrift` in stage `pre`)
are scored by inference only.
Their test rows are built here from the dense ERA5 and CAMS arrays: the 37 lag-free columns (weather
and datetime filled; spread and lead columns are not read by the `pre` head), the 8 lag columns with
the 24 h cap, the knot position, and for P2 the 14 B1 columns.

Usage: python e00632_comparators.py MONTH...
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data428  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00559_extra as extra  # ty: ignore[unresolved-import]
import e00565_model_variants as var565  # ty: ignore[unresolved-import]
import e00569_columns as cols569  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00591_model as aux591  # ty: ignore[unresolved-import]
import e00632_data as d
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SCORE_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
N_FUT: Final[int] = 336
D = ML_EXPERIMENTS_DATA_DIR
P2_DIR: Final[Path] = D / "00613-nra-roster-contrasts-with-intervals-five-seeds/chains/d5/work"
K3_DIR: Final[Path] = D / "00565-drift-embedding-variants-six-designs/work/pretrained_D5/real"
N_B1: Final[int] = 14


def idx_of(*, t: datetime) -> int:
    return int((int(t.timestamp() * 1_000_000) - prep422.GRID_START_US) // prep422.STEP_US)


def month_start(*, month: str) -> datetime:
    return prep.refit_of(month=month)


def test_origins(*, month: str) -> np.ndarray:
    """Return the grid indices of the month's 09:00 UTC origins whose 168 h window ends before 2026-07-01."""
    t = month_start(month=month)
    stop = c510.month_stop(month=month)
    out = []
    while t < stop:
        o = idx_of(t=t.replace(hour=d.ORIGIN_HOUR_UTC))
        if o + N_FUT < idx_of(t=SCORE_END):
            out.append(o)
        t += timedelta(days=1)
    return np.array(out, dtype=np.int64)


@dataclass
class FoldStats:
    mean37: np.ndarray
    std37: np.ndarray
    mask37: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    lag_stats: np.ndarray  # (2, 2)


def fold_stats(*, month: str, clean: np.ndarray) -> FoldStats:
    f = d.CACHE / f"fold_{month}.npz"
    if f.exists():
        z = np.load(f)
        return FoldStats(**{k: z[k] for k in z.files})
    rows = data428.build_fold(refit=month_start(month=month), clean=clean)
    stats = lg.lag_stats(rows=rows, clean=clean, cap_steps=lg.CAP_24H)
    s = FoldStats(
        rows.standardiser.mean, rows.standardiser.std, rows.standardiser.mask,
        rows.target_mean, rows.target_std, stats,
    )  # fmt: skip
    d.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **s.__dict__)
    return s


def fold_for_seq(*, s: FoldStats) -> d.Fold:
    """Return Seq's `Fold` built from the stored statistics (the 28 weather and datetime columns)."""
    cols = d.WEATHER_IN_46 + d.DATETIME_IN_46
    # indices into the 37 kept columns: weather 0-6, datetime 14-34
    keep37 = list(range(7)) + list(range(14, 35))
    del cols
    return d.Fold(
        mean=s.mean37[keep37], std=s.std37[keep37], standardise=s.mask37[keep37].astype(bool),
        target_mean=s.target_mean, target_std=s.target_std, train_end=0,
        first_valid=np.zeros(len(s.target_mean), dtype=np.int64),
    )  # fmt: skip


def check_weather_coverage(*, raw: np.ndarray, origins: np.ndarray) -> str:
    """Assert ERA5 and CAMS weather exists for every test valid time of every Primary."""
    missing = 0
    for p in range(raw.shape[0]):
        for o in origins:
            w = np.asarray(raw[p, o + 1 : o + N_FUT + 1, : d.N_WEATHER])
            missing += int((~np.isfinite(w)).any(axis=1).sum())
    assert missing == 0, f"{missing} test half-hours lack ERA5 or CAMS weather"
    return f"weather coverage: all {len(origins) * N_FUT * raw.shape[0]} test half-hours complete"


def test_rows(
    *, raw: np.ndarray, clean: np.ndarray, s: FoldStats, month: str, device: str
) -> dict:  # fmt: skip
    """Return the P2-pre and K3-pre input matrices and keys of the month's test rows."""
    origins = test_origins(month=month)
    check_weather_coverage(raw=raw, origins=origins)
    n_p = raw.shape[0]
    primary = np.repeat(np.arange(n_p), len(origins) * N_FUT)
    origin_idx = np.tile(np.repeat(origins, N_FUT), n_p)
    step = np.tile(np.arange(1, N_FUT + 1), n_p * len(origins))
    target_idx = origin_idx + step
    lead = (step * 0.5).astype(np.float32)
    keep37 = list(range(7)) + list(range(14, 35))
    x37 = np.zeros((len(primary), 37), dtype=np.float32)
    for p in range(n_p):
        sel = primary == p
        r = np.asarray(raw[p])[target_idx[sel]]
        z = np.clip((r - s.mean37[keep37]) / s.std37[keep37], -d.CLIP_SD, d.CLIP_SD)
        # columns that are not standardised pass through (mask False: mean 0, std 1)
        x37[np.ix_(sel, keep37)] = np.nan_to_num(z, nan=0.0)
    blk = lg.lag_block(
        clean=clean, primary=primary, target_idx=target_idx, origin_idx=origin_idx,
        stats=s.lag_stats, cap_steps=lg.CAP_24H,
    )  # fmt: skip
    # the newest lag read is at least 24 h before the origin
    assert (
        blk[:, 6:8][blk[:, 6:8] >= 0]
        <= np.repeat(origin_idx, 2).reshape(-1, 2)[blk[:, 6:8] >= 0] - lg.CAP_24H
    ).all()
    tau = extra.DriftColumns.tau(valid_us=prep422.GRID_START_US + target_idx * prep422.STEP_US)
    return {
        "primary": primary, "origin_idx": origin_idx, "target_idx": target_idx, "lead": lead,
        "x37": x37, "blk": blk, "tau": tau.astype(np.float32),
    }  # fmt: skip


def b1_columns(*, month: str, rows: dict) -> np.ndarray:
    ex = cols569.Extras(arm="B1", proximity={})
    st = np.load(P2_DIR / "stats" / "B1" / f"{month}.npz")
    ex.mean, ex.std = st["mean"], st["std"]
    parts = []
    n = len(rows["primary"])
    for a in range(0, n, 500_000):
        sl = slice(a, a + 500_000)
        raw_b1 = ex.raw_pretrain(
            primary=rows["primary"][sl], origin_idx=rows["origin_idx"][sl],
            target_idx=rows["target_idx"][sl], shift=None,
        )  # fmt: skip
        parts.append(ex.scale(raw=raw_b1))
    out = np.concatenate(parts)
    assert out.shape[1] == N_B1
    return out


def quantiles_mw(
    *, out: torch.Tensor, primary: np.ndarray, s: FoldStats, capacity: np.ndarray
) -> np.ndarray:
    log_w, mean, std = base.mixture_parts(out=out, components=3)
    q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std).cpu().numpy()
    q = np.sort(q, axis=1)
    power = q * s.target_std[primary, None] + s.target_mean[primary, None]
    return power * capacity[primary, None]


def run_comparator(
    *, arm: str, month: str, seed: int, rows: dict, b1: np.ndarray, table: torch.Tensor,
    s: FoldStats, capacity: np.ndarray, device: str,
) -> np.ndarray:  # fmt: skip
    """Return the 11 quantiles (rows, 11) in MW or MVA for `arm` in {"P2-pre", "K3-pre"}."""
    if arm == "P2-pre":
        state = torch.load(P2_DIR / "pretrained_P2" / f"{month}_s{seed}.pt", map_location="cpu")
        model = aux591.AuxLagMdn(
            n_primaries=len(s.target_mean), stage="pre", n_knots=state["knots"].shape[1],
            n_b1=N_B1, kind="shared",
        )  # fmt: skip
        model.table = table
    else:
        state = torch.load(K3_DIR / f"{month}_s{seed}.pt", map_location="cpu")
        model = var565.OutputOffsetDrift(
            n_primaries=len(s.target_mean), stage="pre", n_knots=state["knots"].shape[-2]
        )
    model.load_state_dict(state)
    model.to(device).eval()
    out_q = np.empty((len(rows["primary"]), 11), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(rows["primary"]), 200_000):
            sl = slice(a, a + 200_000)
            x37 = rows["x37"][sl]
            if arm == "P2-pre":
                x = np.concatenate([x37, rows["blk"][sl], rows["tau"][sl, None], b1[sl]], axis=1)
            else:
                x = np.concatenate([x37, rows["tau"][sl, None]], axis=1)
            xt = torch.from_numpy(x).to(device)
            pt = torch.from_numpy(rows["primary"][sl]).to(device)
            out = model(xt, pt)
            out_q[sl] = quantiles_mw(out=out, primary=rows["primary"][sl], s=s, capacity=capacity)
    return out_q


def capacity_of_primary() -> np.ndarray:
    """Return capacity (MW or MVA) per Primary index; `clean` is power over capacity."""
    import polars as pl

    sub = pl.read_csv(ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07/substations.csv")
    table = prep.primaries().select("primary", "time_series_id")
    j = table.join(sub, on="time_series_id").sort("primary")
    assert j.height == 16
    return j["capacity"].to_numpy()


def main() -> None:
    import sys

    import polars as pl

    months = sys.argv[1:] or list(MONTHS)
    seeds = (0, 1, 2)
    dev = "cuda"
    raw = d.dense_raw()
    clean = d.clean_power()
    cap = capacity_of_primary()
    table_prim = prep.primaries().sort("primary")
    for month in months:
        s = fold_stats(month=month, clean=clean)
        rows = test_rows(raw=raw, clean=clean, s=s, month=month, device=dev)
        b1 = b1_columns(month=month, rows=rows)
        fold = fold_for_seq(s=s)
        table = d.standardised_inputs(raw=raw, fold=fold, device=dev)
        origin_t = pl.from_epoch(
            pl.Series(prep422.GRID_START_US + rows["origin_idx"] * prep422.STEP_US), time_unit="us"
        )
        valid_t = pl.from_epoch(
            pl.Series(prep422.GRID_START_US + rows["target_idx"] * prep422.STEP_US), time_unit="us"
        )
        ids = table_prim["time_series_id"].to_numpy()[rows["primary"]]
        for arm in ("P2-pre", "K3-pre"):
            for seed in seeds:
                target = d.ROOT / "predictions" / arm / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                q = run_comparator(
                    arm=arm, month=month, seed=seed, rows=rows, b1=b1, table=table, s=s,
                    capacity=cap, device=dev,
                )  # fmt: skip
                frame = pl.DataFrame(
                    {
                        "time_series_id": ids,
                        "power_fcst_init_time": origin_t,
                        "valid_time": valid_t,
                        "lead_hours": rows["lead"],
                        **{f"q{round(t * 100):02d}": q[:, i] for i, t in enumerate(base.LEVELS)},
                    }
                ).with_columns(
                    pl.col("power_fcst_init_time", "valid_time").dt.replace_time_zone("UTC")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                frame.write_parquet(tmp)
                tmp.rename(target)
                print(f"{arm} {month} s{seed}: {frame.height} rows", flush=True)


if __name__ == "__main__":
    main()
