"""Experiment 401, Job 2, arms step: fixed-weight and per-lead mixtures of two stored forecasts.

Forecast with lags and forecast without lags are experiment 400's `arm_lags.parquet` and
`arm_lagless.parquet` (experiment 381's weekly bundle and experiment 383's raw lagless arm, in MW,
on experiment 395's Part F scored rows). Nothing is refitted. Each weight function is applied as a
mixture of distributions (experiment 400's CDF algebra) and as an average of quantiles, and every
arm is also written widened by 1.15: each quantile moves from the arm's median by 1.15 times its
distance. The per-lead weight is, for each scored month and each lead day 1 to 10, the `w` in
{0.1, ..., 0.9} with the lowest exact CRPS of the mixture on earlier rows (flagged half-hours
dropped, pooled over Primaries, a fixed random subsample per lead day); warm-up months use 0.5.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored \
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=6G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00401-*/e00401_arms.py
```
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00400_blend as blend
import e00400_mixture as mix
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = (
    DATA / "00401-real-and-synthetic-load-shift-median-share-and-fixed-forecast-mixture" / "work"
)
KEYS: Final[list[str]] = blend.KEYS
FIXED: Final[dict[str, float]] = {"w30": 0.3, "w50": 0.5, "w70": 0.7}
GRID: Final[np.ndarray] = np.round(np.arange(1, 10) / 10, 1)
LEAD_DAYS: Final[int] = 10
N_PER_DAY: Final[int] = 30_000
WIDEN: Final[float] = 1.15
CHUNK: Final[int] = 20_000
MEDIAN: Final[int] = 5


def widen(*, q: np.ndarray, factor: float) -> np.ndarray:
    """Scale each quantile's distance from the median by `factor`."""
    return (q[:, [MEDIAN]] + (q - q[:, [MEDIAN]]) * factor).astype(np.float32)


def write_arm(*, name: str, q: np.ndarray, scored: pl.DataFrame) -> None:
    """Write an arm's forecast in MW (keys and 11 quantile columns), one file per arm."""
    mw = q.astype(np.float64) * scored["capacity_mw"].to_numpy()[:, None]
    frame = scored.select(*KEYS).with_columns(
        *[pl.Series(c, mw[:, i].astype(np.float32)) for i, c in enumerate(list(QUANTILE_COLUMNS))]
    )
    frame.write_parquet((WORK / f"arm_{name}.parquet").with_suffix(".tmp"))
    (WORK / f"arm_{name}.parquet").with_suffix(".tmp").rename(WORK / f"arm_{name}.parquet")


def write_with_widening(*, name: str, q: np.ndarray, scored: pl.DataFrame) -> None:
    """Write an arm raw and widened by 1.15."""
    write_arm(name=name, q=q, scored=scored)
    write_arm(name=f"{name}_x115", q=widen(q=q, factor=WIDEN), scored=scored)


def per_lead_weights(
    *, scored: pl.DataFrame, q_lags: np.ndarray, q_lagless: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Return the per-row weight on the forecast with lags, and the chosen weights by month."""
    lead_day = np.minimum((scored["lead_hours"].to_numpy() // 24).astype(int), LEAD_DAYS - 1)
    usable = ~scored["flagged"].to_numpy() & np.isfinite(actual)
    month_epoch = scored["valid_time"].dt.truncate("1mo").dt.epoch("us").to_numpy()
    weights = np.full(scored.height, 0.5)
    chosen: dict[str, list[float]] = {}
    rng = np.random.default_rng(0)
    for month in sorted(np.unique(month_epoch).tolist()):
        month_dt = datetime.fromtimestamp(month / 1e6, tz=UTC)
        if month_dt < blend.WARMUP_END:
            continue
        in_month = month_epoch == month
        best = []
        for day in range(LEAD_DAYS):
            eligible = np.flatnonzero((month_epoch < month) & usable & (lead_day == day))
            if not len(eligible):
                msg = f"no earlier rows at lead day {day} before {month_dt.date()}"
                raise ValueError(msg)
            sample = np.sort(
                rng.choice(eligible, size=min(N_PER_DAY, len(eligible)), replace=False)
            )
            pair = mix.Pair(q1=q_lags[sample], q2=q_lagless[sample], actual=actual[sample])
            scores = [float(pair.crps(w=np.full(len(sample), w)).mean()) for w in GRID]
            best.append(float(GRID[int(np.argmin(scores))]))
            weights[in_month & (lead_day == day)] = best[-1]
        chosen[str(month_dt.date())] = best
        print(month_dt.date(), "per-lead weights", best, flush=True)
    return weights, chosen


def main() -> None:
    """Write every arm's forecast."""
    WORK.mkdir(parents=True, exist_ok=True)
    scored = pl.read_parquet(
        blend.PART_F / "rows_bundle.parquet",
        columns=[*KEYS, "label", "lead_hours", "capacity_mw", "actual_mw", "flagged"],
    ).sort(KEYS)
    cap = scored["capacity_mw"].to_numpy()[:, None]
    qcols = list(QUANTILE_COLUMNS)
    arm_lags = pl.read_parquet(blend.WORK / "arm_lags.parquet").sort(KEYS)
    arm_lagless = pl.read_parquet(blend.WORK / "arm_lagless.parquet").sort(KEYS)
    for frame in (arm_lags, arm_lagless):
        if not frame.select(*KEYS).equals(scored.select(*KEYS)):
            msg = "stored arm rows differ from the scored rows"
            raise ValueError(msg)
    q_lags = (arm_lags.select(qcols).to_numpy() / cap).astype(np.float32)
    q_lagless = (arm_lagless.select(qcols).to_numpy() / cap).astype(np.float32)
    del arm_lags, arm_lagless
    actual = (scored["actual_mw"].to_numpy() / cap[:, 0]).astype(np.float64)
    write_with_widening(name="lags", q=q_lags, scored=scored)
    write_with_widening(name="lagless", q=q_lagless, scored=scored)
    weight_sets = {k: np.full(scored.height, v) for k, v in FIXED.items()}
    weight_sets["lead"], chosen = per_lead_weights(
        scored=scored, q_lags=q_lags, q_lagless=q_lagless, actual=actual
    )
    (WORK / "per_lead_weights.json").write_text(json.dumps(chosen, indent=1))
    for key, w in weight_sets.items():
        out = np.empty_like(q_lags)
        for start in range(0, scored.height, CHUNK):
            sl = slice(start, start + CHUNK)
            out[sl] = mix.Pair(q1=q_lags[sl], q2=q_lagless[sl]).quantiles(w=w[sl])
        write_with_widening(name=f"mix_{key}", q=out, scored=scored)
        avg = (w[:, None] * q_lags + (1 - w[:, None]) * q_lagless).astype(np.float32)
        write_with_widening(name=f"avg_{key}", q=avg, scored=scored)
        print("wrote", key, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
