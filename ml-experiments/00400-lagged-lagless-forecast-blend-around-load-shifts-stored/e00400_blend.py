"""Experiment 400, blend step: fit the weights, build each arm's forecast, and write the frames.

Forecast with lags: experiment 381's weekly bundle. Forecast without lags: experiment 383's raw
lagless arm. Suspicion score: experiment 391's stage 1 surprise tables (`u`) at the three lag times
of a row, for the row's own Primary and for its NGED-listed neighbours (list copied from
experiment 383's `e00383_neighbours.py`). Rows: experiment 395's Part F slice table.

The weight on the forecast with lags is `1 / (1 + exp(theta . x))`, with inputs `x` fixed by the
arm. The fit minimises the exact CRPS of the mixture (plus a ridge penalty of 1e-3 |theta|^2) over a
fixed random subsample of 200,000 earlier rows (valid time before the first of the scored month,
flagged half-hours dropped), pooled over Primaries, with BFGS.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored \
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=6G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00400-*/e00400_blend.py
```
"""

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00400_mixture as mix
import numpy as np
import polars as pl
from scipy.optimize import minimize
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored" / "work"
PART_F: Final[Path] = (
    DATA / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts" / "work" / "part_f"
)
LAGLESS: Final[Path] = (
    DATA / "00383-load-shift-xgboost-oracle-first-part-1" / "work" / "arms" / "lagless.parquet"
)
SURPRISE: Final[Path] = (
    DATA / "00391-xgboost-lag-surprise-second-stage" / "surprise_tables" / "half_hourly"
)
NGED_DIR: Final[Path] = DATA.parent / "NGED" / "substation_hierarchy_2026-10-15"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
WARMUP_END: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
N_FIT: Final[int] = 200_000
RIDGE: Final[float] = 1e-3
CHUNK: Final[int] = 20_000
SERIES_TO_SITE: Final[dict[int, str]] = {
    3: "D3", 7: "D9", 8: "D11", 9: "D19", 10: "D10", 11: "D4", 12: "D15", 13: "D18",
    14: "D12", 15: "D22", 16: "D14", 17: "D5", 18: "D16", 24: "D21", 25: "D2", 26: "D8",
}  # fmt: skip
"""Experiment 298's mapping, copied from experiment 383's `e00383_neighbours.py`."""
ARMS: Final[dict[str, tuple[str, ...]]] = {
    "b1": ("s_own",),
    "b2": ("s_own", "s_nb"),
    "b3": ("s_own", "own_signed", "nb_pos_sum", "nb_neg_sum", "nb_pos_max", "nb_neg_max"),
}
"""Input columns of each arm; every arm also takes the intercept and `lead_term`."""
INPUT_SCALE: Final[float] = 2.0


def nged_neighbours() -> dict[str, tuple[str, ...]]:
    """Return each metered Primary's metered neighbours (copied from experiment 383)."""
    with (NGED_DIR / "Monitor Direction.csv").open() as handle:
        monitors = list(csv.DictReader(handle))
    sites_of_number: dict[str, set[str]] = {}
    for r in monitors:
        if int(r["TimeSeriesID"]) in SERIES_TO_SITE:
            sites_of_number.setdefault(r["SubstationNumber"], set()).add(
                SERIES_TO_SITE[int(r["TimeSeriesID"])]
            )
    with (NGED_DIR / "Primary Substation Interconnections.csv").open() as handle:
        pairs = [(r["PrimarySubstation1"], r["PrimarySubstation2"]) for r in csv.DictReader(handle)]
    metered: dict[str, set[str]] = {s: set() for sites in sites_of_number.values() for s in sites}
    for first, second in pairs:
        for here, there in ((first, second), (second, first)):
            for site in sites_of_number.get(here, ()):
                if there in sites_of_number:
                    metered[site] |= sites_of_number[there]
    return {s: tuple(sorted(v - {s}, key=lambda x: int(x[1:]))) for s, v in metered.items()}


def surprise_arrays(*, labels: list[str]) -> dict[str, np.ndarray]:
    """Return experiment 391's `u` of each Primary on the dense half-hourly grid (NaN if absent)."""
    size = common.grid_times().len()
    out = {}
    for label in labels:
        table = pl.read_parquet(SURPRISE / f"{label}.parquet", columns=["valid_time", "u"])
        values = np.full(size, np.nan)
        values[np.asarray(common.grid_index(times=table["valid_time"]))] = table["u"].to_numpy()
        out[label] = values
    return out


def surprise_features(*, keys: pl.DataFrame, label: str, u: dict, nb: tuple[str, ...]) -> dict:
    """Return one Primary's surprise inputs for its rows, from the u values at the three lags."""
    size = common.grid_times().len()
    target = np.asarray(common.grid_index(times=keys["valid_time"]))
    origin = np.asarray(common.grid_index(times=keys["power_fcst_init_time"]))
    lag_index = common.origin_lags(
        power=np.arange(size, dtype=np.float32), target=target, origin=origin
    ).astype(np.int64)
    own = u[label][lag_index]
    missing = np.isnan(own)
    own = np.nan_to_num(own)
    star = np.abs(own).argmax(axis=1)
    rows = np.arange(len(own))
    pos_sum = np.zeros(own.shape)
    neg_sum = np.zeros(own.shape)
    pos_max = np.zeros(own.shape)
    neg_max = np.zeros(own.shape)
    nb_abs = np.zeros(len(own))
    for name in nb:
        values = np.nan_to_num(u[name][lag_index])
        pos_sum += np.maximum(values, 0)
        neg_sum += np.minimum(values, 0)
        pos_max = np.maximum(pos_max, values)
        neg_max = np.minimum(neg_max, values)
        nb_abs = np.maximum(nb_abs, np.abs(values).max(axis=1))
    return {
        "s_own": np.abs(own).max(axis=1),
        "s_nb": nb_abs,
        "own_signed": own[rows, star],
        "nb_pos_sum": pos_sum[rows, star],
        "nb_neg_sum": neg_sum[rows, star],
        "nb_pos_max": pos_max[rows, star],
        "nb_neg_max": neg_max[rows, star],
        "u_missing": missing.any(axis=1),
    }


def design(*, features: dict, columns: tuple[str, ...], rows: np.ndarray) -> np.ndarray:
    """Return the design matrix: intercept, scaled inputs, and `lead_term`."""
    parts = [np.ones(len(rows))] + [features[c][rows] / INPUT_SCALE for c in columns]
    parts.append(features["lead_term"][rows])
    return np.stack(parts, axis=1)


def weight(*, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the weight on the forecast with lags."""
    return 1.0 / (1.0 + np.exp(np.clip(x @ theta, -30.0, 30.0)))


def fit(*, pair: mix.Pair, x: np.ndarray) -> np.ndarray:
    """Return theta minimising the exact mixture CRPS plus the ridge penalty."""

    def loss(theta: np.ndarray) -> float:
        return float(pair.crps(w=weight(theta=theta, x=x)).mean() + RIDGE * (theta**2).sum())

    return minimize(loss, np.zeros(x.shape[1]), method="BFGS", options={"maxiter": 60}).x


def load_frames(*, series: pl.DataFrame, rows: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return both forecasts (MW) restricted to the rows, in the rows' order."""
    lags = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    stored = pl.scan_parquet(LAGLESS).filter(pl.col("lead_hours") <= 336).collect()
    lagless = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    out = []
    for name, frame in (("with lags", lags), ("without lags", lagless)):
        joined = rows.select(*KEYS).join(frame, on=KEYS, how="left", maintain_order="left")
        missing = joined["q50"].is_null().sum()
        print(f"{name}: {frame.height} stored rows; {missing} of {rows.height} scored rows missing")
        out.append(joined)
    return out[0], out[1]


def main() -> None:  # noqa: PLR0915
    """Fit month by month, write each arm's frame, the weights, and the fitted parameters."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    labels = series["label"].to_list()
    scored = pl.read_parquet(
        PART_F / "rows_bundle.parquet",
        columns=[*KEYS, "label", "lead_hours", "capacity_mw", "actual_mw", "flagged"],
    ).sort(KEYS)
    print("rows", scored.height, flush=True)
    lags_mw, lagless_mw = load_frames(series=series, rows=scored)
    capacity = scored["capacity_mw"].to_numpy()
    scale = 1.0 / capacity[:, None]
    q_lags = (lags_mw.select(list(QUANTILE_COLUMNS)).to_numpy() * scale).astype(np.float32)
    q_lagless = (lagless_mw.select(list(QUANTILE_COLUMNS)).to_numpy() * scale).astype(np.float32)
    del lags_mw, lagless_mw
    actual = (scored["actual_mw"].to_numpy() / capacity).astype(np.float64)
    usable = ~scored["flagged"].to_numpy() & np.isfinite(actual)
    nb_of = nged_neighbours()
    u = surprise_arrays(labels=labels)
    features: dict[str, np.ndarray] = {}
    for label in labels:
        index = np.flatnonzero(scored["label"].to_numpy() == label)
        part = surprise_features(
            keys=scored[index],
            label=label,
            u=u,
            nb=tuple(n for n in nb_of.get(label, ()) if n in u),
        )
        for name, values in part.items():
            features.setdefault(name, np.zeros(scored.height, dtype=values.dtype))[index] = values
    features["lead_term"] = np.log(scored["lead_hours"].to_numpy() / 24.0)
    valid = scored["valid_time"]
    month_epoch = valid.dt.truncate("1mo").dt.epoch("us").to_numpy()
    month_list = sorted(np.unique(month_epoch).tolist())
    thetas: dict[str, dict[str, list[float]]] = {a: {} for a in ARMS}
    w_all = {a: np.full(scored.height, 0.5) for a in ARMS}
    rng = np.random.default_rng(0)
    for month in month_list:
        month_dt = datetime.fromtimestamp(month / 1e6, tz=UTC)
        in_month = np.flatnonzero(month_epoch == month)
        if month_dt < WARMUP_END:
            print("warm-up month", month_dt.date(), "w = 0.5", flush=True)
            continue
        eligible = np.flatnonzero((month_epoch < month) & usable)
        sample = np.sort(rng.choice(eligible, size=min(N_FIT, len(eligible)), replace=False))
        pair = mix.Pair(q1=q_lags[sample], q2=q_lagless[sample], actual=actual[sample])
        for arm, columns in ARMS.items():
            theta = fit(pair=pair, x=design(features=features, columns=columns, rows=sample))
            thetas[arm][str(month_dt.date())] = theta.tolist()
            w_all[arm][in_month] = weight(
                theta=theta, x=design(features=features, columns=columns, rows=in_month)
            )
            print(
                month_dt.date(),
                arm,
                np.round(theta, 3),
                "mean w",
                w_all[arm][in_month].mean().round(3),
                flush=True,
            )
    (WORK / "thetas.json").write_text(json.dumps(thetas, indent=1))
    weights = {"fixed": np.full(scored.height, 0.5), **w_all}
    pl.DataFrame(
        {
            **{k: scored[k] for k in KEYS},
            **{f"w_{a}": v.astype(np.float32) for a, v in weights.items()},
            **{c: v for c, v in features.items() if c != "lead_term"},
        }
    ).write_parquet(WORK / "weights.parquet")
    write_arm(name="lags", q=q_lags, scored=scored)
    write_arm(name="lagless", q=q_lagless, scored=scored)
    for arm, w in weights.items():
        out = np.empty_like(q_lags)
        for start in range(0, scored.height, CHUNK):
            sl = slice(start, start + CHUNK)
            pair = mix.Pair(q1=q_lags[sl], q2=q_lagless[sl])
            out[sl] = pair.quantiles(w=w[sl])
        write_arm(name=f"mix_{arm}", q=out, scored=scored)
        write_arm(
            name=f"avg_{arm}",
            q=(w[:, None] * q_lags + (1 - w[:, None]) * q_lagless).astype(np.float32),
            scored=scored,
        )
        print("wrote", arm, flush=True)


def write_arm(*, name: str, q: np.ndarray, scored: pl.DataFrame) -> None:
    """Write an arm's forecast in MW (keys and 11 quantile columns), one file per arm."""
    mw = q.astype(np.float64) * scored["capacity_mw"].to_numpy()[:, None]
    frame = scored.select(*KEYS).with_columns(
        *[pl.Series(c, mw[:, i].astype(np.float32)) for i, c in enumerate(list(QUANTILE_COLUMNS))]
    )
    frame.write_parquet(WORK / f"arm_{name}.parquet")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
