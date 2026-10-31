# ruff: noqa: ANN201, E501, RUF028, SIM108
"""Experiment 402, step 4 data: injected worlds, the forecasts re-predicted on them, gate features.

For one injection seed (a "world") and one Primary, the script:

1. regenerates the world's injected power for all 16 Primaries with `e00402_inject.inject` over
   2025-09-15 to 2026-06-30 (clean base periods only: padded labelled events are excluded);
2. for every refit month of experiment 392 and every `ORIGIN_STRIDE`th daily run of the month,
   re-predicts the arm's lagged booster with the injected lags (the clean series is the truth), and
   reads the stored forecast without lags (the lagless booster never sees the lags);
3. writes one row per (origin, valid time) with both forecasts' 11 quantiles, the clean actual, the
   exact CRPS terms `A`, `B`, `C`, the gate features, and the injected truth flags.

Surprises (`u`) are experiment 391's stage-1 quantiles (out of sample at every lag time) passed
through the injected value; every window is truncated at the telemetry cut-off `NEWEST_AGE`.
Training rows and scored rows are separated downstream by `e00402_gate.py`. `--real` writes the
same table for the un-injected series and the stored forecasts of the arm (the real rows).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00377-demand-v1-first-run-feasibility:ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:ml-experiments/00391-xgboost-lag-surprise-second-stage:\
ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored \
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=3G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00402-*/e00402_worlds.py \
    --seed 0 --sites D5 --arm r1
```
"""

import argparse
import dataclasses
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00391_stage1 as stage1  # ty: ignore[unresolved-import]
import e00400_blend as blend400  # ty: ignore[unresolved-import]
import e00402_counterfactual as cf
import e00402_inject as inj
import e00402_mixture as mix
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = cf.WORK
OUT: Final[Path] = WORK / "worlds"
ARM_PREDS_392: Final[Path] = (
    DATA / "00392-xgboost-nra-load-shift-masked-targets" / "work" / "arm_preds"
)
LABELS: Final[Path] = blend400.WORK / "label_table_corrected.csv"
FIRST_DAY: Final[datetime] = datetime(2025, 9, 15, tzinfo=UTC)
END_DAY: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
ORIGIN_STRIDE: Final[int] = 3
COVERAGE: Final[float] = 0.10
HALF_WIDTHS: Final[tuple[int, ...]] = (6, 24)
"""Window half-widths in half-hours (3 h and 12 h)."""
EVENT_PADDING: Final[int] = 48
LAGLESS_ARM: Final[dict[str, str]] = {"r1": "r1_lagless", "r0": "r0_lagless"}
LAGGED_ARM: Final[dict[str, str]] = {"r1": "r1_lagged", "r0": "r0_lagged"}


@dataclasses.dataclass(frozen=True)
class Context:
    """Everything that is shared between worlds: the clean series, stage 1, and the graph."""

    labels: list[str]
    rows: dict[str, dict]
    arrays: dict[str, common.SeriesArrays]
    capacity: dict[str, float]
    stage1_q: dict[str, np.ndarray]
    stage1_mad_mw: dict[str, float]
    neighbours: dict[str, list[str]]
    magnitude_only: set[str]
    event_mask: dict[str, np.ndarray]


def load_context() -> Context:
    """Load the clean power, the stage-1 quantiles on the grid, the graph, and the event masks."""
    series, arrays = fd._series_and_arrays()
    events = pl.read_parquet(cf.EVENTS)
    labels = sorted(set(events["label"].to_list()), key=lambda x: int(x[1:]))
    rows = {r["site"]: r for r in series.iter_rows(named=True) if r["site"] in labels}
    size = common.grid_times().len()
    capacity = {s: float(rows[s]["effective_capacity_mw"]) for s in labels}
    q: dict[str, np.ndarray] = {}
    mad: dict[str, float] = {}
    for s in labels:
        table = pl.read_parquet(cf.STAGE1 / f"{s}.parquet")
        grid = np.full((size, len(common.LEVELS)), np.nan, dtype=np.float32)
        index = np.asarray(common.grid_index(times=table["valid_time"]))
        grid[index] = table.select(list(common.LEVEL_COLUMNS)).to_numpy() / capacity[s]
        q[s] = grid
        r = table["r_mw"].to_numpy()
        mad[s] = float(np.nanmedian(np.abs(r - np.nanmedian(r))))
    mask: dict[str, np.ndarray] = {}
    for s in labels:
        m = np.zeros(size, dtype=bool)
        for e in events.filter(pl.col("label") == s).iter_rows(named=True):
            m[max(0, e["onset_idx"] - EVENT_PADDING) : e["end_idx"] + 1 + EVENT_PADDING] = True
        mask[s] = m
    nb = blend400.nged_neighbours()
    return Context(
        labels=labels,
        rows=rows,
        arrays={s: arrays[s] for s in labels},
        capacity=capacity,
        stage1_q=q,
        stage1_mad_mw=mad,
        neighbours={s: [n for n in nb.get(s, ()) if n in labels] for s in labels},
        magnitude_only={s for s in labels if rows[s]["units"] == "MVA"},
        event_mask=mask,
    )


def make_pool(*, ctx: Context) -> inj.EventPool:
    """Return the event pool resampled from the corrected label table."""
    table = pl.read_csv(LABELS)
    means = {
        s: float(np.nanmean(np.abs(ctx.arrays[s].power)) * ctx.capacity[s]) for s in ctx.labels
    }
    return inj.event_pool(table=table, trailing_mean_mw=means)


def build_world(*, ctx: Context, seed: int, no_recipient_share: float = inj.NO_RECIPIENT_SHARE):
    """Return the injected world for a seed."""
    first = int(common.grid_index(times=FIRST_DAY))
    last = int(common.grid_index(times=END_DAY))
    return inj.inject(
        power={s: ctx.arrays[s].power.astype(np.float64) for s in ctx.labels},
        capacity_mw=ctx.capacity,
        neighbours=ctx.neighbours,
        magnitude_only=ctx.magnitude_only,
        spike_mad_mw=ctx.stage1_mad_mw,
        pool=make_pool(ctx=ctx),
        first=first,
        last=last,
        coverage=COVERAGE,
        seed=seed,
        no_recipient_share=no_recipient_share,
        allowed={s: ~ctx.event_mask[s] for s in ctx.labels},
    )


def surprise_series(*, ctx: Context, power: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Return per Primary the stage-1 normal score `u` and the residual (fraction) on the grid."""
    u, r = {}, {}
    for s in ctx.labels:
        u[s] = stage1.pit_normal_score(
            quantiles=ctx.stage1_q[s].astype(np.float64), actual=power[s]
        )
        r[s] = power[s] - ctx.stage1_q[s][:, common.LEVEL_COLUMNS.index("q50")]
    return u, r


def cumulative(*, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative sums of the finite values and of their count, with a leading zero."""
    finite = np.isfinite(x)
    return (
        np.concatenate([[0.0], np.cumsum(np.where(finite, x, 0.0))]),
        np.concatenate([[0], np.cumsum(finite)]).astype(np.float64),
    )


def window_mean(*, cum: tuple, centre: np.ndarray, half: int, newest: np.ndarray) -> np.ndarray:
    """Return the mean over `[centre - half, min(centre + half, newest)]` (NaN if no value)."""
    cs, cn = cum
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(centre + half, newest)
    n = cn[hi + 1] - cn[lo]
    return np.where(n > 0, (cs[hi + 1] - cs[lo]) / np.maximum(n, 1), np.nan)


FEATURES_SINGLE: Final[tuple[str, ...]] = (
    "u_0", "u_1", "u_2", "r_0", "r_1", "r_2", "lead_days",
)  # fmt: skip


def gate_features(
    *, site: str, ctx: Context, u: dict, r: dict, target: np.ndarray, origin: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the gate's inputs for rows of one Primary: all computable at the forecast origin."""
    size = common.grid_times().len()
    lag = common.origin_lags(
        power=np.arange(size, dtype=np.float32), target=target, origin=origin
    ).astype(np.int64)
    newest = origin - common.NEWEST_AGE
    out: dict[str, np.ndarray] = {}
    cum_u, cum_r = cumulative(x=u[site]), cumulative(x=r[site])
    nbs = ctx.neighbours.get(site, [])
    nb_u = np.stack([u[n][lag] for n in nbs], axis=-1) if nbs else np.full((*lag.shape, 1), np.nan)
    for j in range(3):
        out[f"u_{j}"] = u[site][lag[:, j]]
        out[f"r_{j}"] = r[site][lag[:, j]]
        for h in HALF_WIDTHS:
            out[f"u_mean{h}_{j}"] = window_mean(cum=cum_u, centre=lag[:, j], half=h, newest=newest)
            out[f"r_mean{h}_{j}"] = window_mean(cum=cum_r, centre=lag[:, j], half=h, newest=newest)
        v = nb_u[:, j, :]
        finite = np.isfinite(v).any(axis=1)
        out[f"nb_max_{j}"] = np.where(
            finite, np.nanmax(np.where(np.isfinite(v), v, -np.inf), axis=1), np.nan
        )
        out[f"nb_min_{j}"] = np.where(
            finite, np.nanmin(np.where(np.isfinite(v), v, np.inf), axis=1), np.nan
        )
        out[f"nb_up_{j}"] = (v > 1).sum(axis=1).astype(float)
        out[f"nb_down_{j}"] = (v < -1).sum(axis=1).astype(float)
    out["lead_days"] = (target - origin) * 0.5 / 24.0
    out["lag_0_idx"], out["lag_1_idx"], out["lag_2_idx"] = lag[:, 0], lag[:, 1], lag[:, 2]
    return out


def world_rows(
    *, ctx: Context, site: str, power: dict, mask: dict | None, arm: str
) -> pl.DataFrame:
    """Return the rows of one Primary: both forecasts, the clean truth, `A`, `B`, `C`, and features."""
    arrays = ctx.arrays[site]
    injected = dataclasses.replace(arrays, power=power[site].astype(arrays.power.dtype))
    u, r = surprise_series(ctx=ctx, power=power)
    columns = fd.input_columns(a7=False)
    pieces = []
    for refit, origins in cf.origins_by_refit().items():
        model = cf.load_model(site=site, refit=refit, threads=1)
        lagless = pl.read_parquet(
            ARM_PREDS_392 / LAGLESS_ARM[arm] / f"{site}_s0_{refit:%Y%m%d}T0000.parquet"
        )
        for origin in origins[::ORIGIN_STRIDE]:
            prepared = cf.run_prepared(site_row=ctx.rows[site], arrays=injected, origin=origin)
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            q_lags = cf.quantiles(model=model, prepared=prepared, columns=columns)
            frame = keys.with_columns(
                **{f"qa_{c}": pl.Series(q_lags[:, i]) for i, c in enumerate(common.LEVEL_COLUMNS)}
            ).join(
                lagless.select(
                    "power_fcst_init_time",
                    "valid_time",
                    **{f"qb_{c}": pl.col(c) for c in common.LEVEL_COLUMNS},
                ),  # fmt: skip
                on=["power_fcst_init_time", "valid_time"],
                how="inner",
            )
            frame = frame.with_columns(refit=pl.lit(refit))
            pieces.append(frame)
    rows = pl.concat(pieces)
    target = np.asarray(common.grid_index(times=rows["valid_time"]))
    origin = np.asarray(common.grid_index(times=rows["power_fcst_init_time"]))
    feats = gate_features(site=site, ctx=ctx, u=u, r=r, target=target, origin=origin)
    qa = rows.select([f"qa_{c}" for c in common.LEVEL_COLUMNS]).to_numpy().astype(np.float64)
    qb = rows.select([f"qb_{c}" for c in common.LEVEL_COLUMNS]).to_numpy().astype(np.float64)
    actual = arrays.power[target].astype(np.float64)
    finite = np.isfinite(actual) & np.isfinite(qa).all(axis=1) & np.isfinite(qb).all(axis=1)
    pair = mix.Pair(q1=qa[finite], q2=qb[finite], actual=actual[finite])
    n = int(finite.sum())
    crps = {w: pair.crps(w=np.full(n, w)) * 100.0 for w in (0.0, 0.5, 1.0)}
    a = 2.0 * (crps[1.0] + crps[0.0] - 2.0 * crps[0.5])
    b = crps[1.0] - crps[0.0] - a
    lag_idx = np.stack([feats["lag_0_idx"], feats["lag_1_idx"], feats["lag_2_idx"]], axis=1)[finite]
    if mask is None:
        code = np.zeros((n, 3), dtype=np.int8)
    else:
        code = mask[site][lag_idx]
    own_event = ((code != inj.NONE).sum(axis=1)).astype(np.int8)
    kind = np.where(own_event > 0, code.max(axis=1), 0).astype(np.int8)
    target_event = (
        (mask[site][target[finite]] != inj.NONE) if mask is not None else np.zeros(n, bool)
    )
    out = {
        "site": np.full(n, site),
        "power_fcst_init_time": rows["power_fcst_init_time"].to_numpy()[finite],
        "valid_time": rows["valid_time"].to_numpy()[finite],
        "A": a,
        "B": b,
        "C": crps[0.0],
        "crps_lags": crps[1.0],
        "actual": actual[finite],
        "lags_in_event": own_event,
        "event_kind": kind,
        "target_in_event": target_event,
        "median_gap": np.abs(qa[finite, cf.Q50] - qb[finite, cf.Q50]),
        "lag_0_idx": lag_idx[:, 0],
        "lag_1_idx": lag_idx[:, 1],
        "lag_2_idx": lag_idx[:, 2],
        **{k: v[finite] for k, v in feats.items() if not k.startswith("lag_")},
    }
    df = pl.DataFrame(out)
    return df.with_columns(
        qa5=pl.Series(qa[finite, cf.Q50]), qb5=pl.Series(qb[finite, cf.Q50]),
        lead_hours=pl.col("lead_days") * 24.0,
    )  # fmt: skip


def main() -> None:
    """Write one world's (or the real) rows for the named Primaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real", action="store_true", help="un-injected series (the real rows)")
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--arm", choices=("r1", "r0"), default="r1")
    parser.add_argument("--no-recipient-share", type=float, default=inj.NO_RECIPIENT_SHARE)
    args = parser.parse_args()
    if args.arm == "r0":
        cf.BOOSTERS = cf.WORK / "boosters" / "r0_lagged"
    ctx = load_context()
    if args.real:
        power = {s: ctx.arrays[s].power.astype(np.float64) for s in ctx.labels}
        mask, name = None, f"real_{args.arm}"
    else:
        world = build_world(ctx=ctx, seed=args.seed, no_recipient_share=args.no_recipient_share)
        power, mask = world.power, world.mask
        tag = (
            ""
            if args.no_recipient_share == inj.NO_RECIPIENT_SHARE
            else f"_nr{args.no_recipient_share:g}"
        )
        name = f"world{args.seed}{tag}_{args.arm}"
    OUT.mkdir(parents=True, exist_ok=True)
    for site in args.sites:
        target = OUT / f"{name}_{site}.parquet"
        if target.exists():
            continue
        frame = world_rows(ctx=ctx, site=site, power=power, mask=mask, arm=args.arm)
        frame.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(site, frame.height, "rows", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
