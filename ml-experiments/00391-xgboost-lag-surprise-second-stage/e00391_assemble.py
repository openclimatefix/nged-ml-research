"""Assemble experiment 391's arms from the stored refit predictions (no fits).

Reads `work/arm_preds/monthly/<arm>/<site>_s0_<refit>.parquet` for every arm of `e00391_fit.ARMS`
that has files, the stored weekly member of experiment 87 (`n1h`, read only) as arm `d`, and the
event masks of the feature cache. Writes each arm to `work/arms/<arm>.parquet` (keys, 11 quantiles
as fractions of effective capacity, `lead_hours`) and the table of contaminated-lag rows (a lag
inside an event) to `work/arms/contaminated.parquet`. An arm with no files is skipped with a
warning, so the script runs on partial data.

Arms written:

- `d`: the stored weekly XGBoost member, labelled "weekly, no widening" (to be verified against
  the stored bundle in the write-up).
- `b0`, `n0`, `n1`, `n2`, `n2_noveto`, `n3`, `o`, `n1_era5`: our monthly refits, no widening.
- `<arm>_w`: the arm widened by the fixed factor 1.15 about the median on every row.

Only rows (keys) that every written arm holds are kept, so all arms are scored on the same rows.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00391-xgboost-lag-surprise-second-stage \
    uv run --no-sync python ml-experiments/00391-*/e00391_assemble.py [--smoke]
```
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00391_fit as fit
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = fit.WORK_DIR
ARM_DIR: Final[Path] = fit.ARM_DIR / "monthly"
SMOKE_ARM_DIR: Final[Path] = WORK_DIR / "arm_preds_smoke"
OUT_DIR: Final[Path] = WORK_DIR / "arms"
SMOKE_OUT_DIR: Final[Path] = WORK_DIR / "arms_smoke"
STORED_87: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "work"
)
FIXED_K: Final[float] = 1.15
"""The fixed widening factor: experiment 87's spread factor, applied about the median."""
LEVELS: Final[tuple[str, ...]] = tuple(common.LEVEL_COLUMNS)
MONTHLY_REFITS: Final[tuple[datetime, ...]] = regimes.REGIMES["monthly"]
WEEKLY_REFITS: Final[tuple[datetime, ...]] = regimes.REGIMES["weekly"]
OWN_ARMS: Final[tuple[str, ...]] = tuple(fit.ARMS)
STORED_ARM: Final[str] = "d"
ARM_LABELS: Final[dict[str, str]] = {
    "d": "weekly, no widening",
    "b0": "monthly baseline",
    "n0": "own surprise",
    "n1": "own + neighbour surprise",
    "n2": "n1 + detector state",
    "n2_noveto": "n2 without the veto",
    "n3": "detector state only",
    "o": "oracle flags (hindsight)",
    "n1_era5": "n1 from ERA5-fed stage 1 (hindsight)",
}


def read_arm(*, directory: Path, name: str, sites: list[str], refits: tuple[datetime, ...],
             stored: bool) -> pl.DataFrame | None:  # fmt: skip
    """Return an arm's per-run quantiles for the given sites, one row per key, or `None` if absent.

    A stored file of experiment 87 holds the origins of both regimes, so for `stored` only the
    origins that the weekly regime assigns to the file's refit are kept.
    """
    pieces = []
    for site in sites:
        for refit in refits:
            path = fd.job_path(root=directory, config=name, site=site, seed=0, refit=refit)
            if not path.exists():
                continue
            frame = pl.read_parquet(path)
            if stored:
                origins = [
                    t
                    for t in frame["power_fcst_init_time"].unique().to_list()
                    if regimes.chosen_refit(refits=refits, origin=t) == refit
                ]
                frame = frame.filter(pl.col("power_fcst_init_time").is_in(origins))
            pieces.append(frame)
    if not pieces:
        return None
    return (
        pl.concat(pieces)
        .drop("seed", strict=False)
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
    )


def contaminated_flags(*, frame: pl.DataFrame, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the frame's keys with `lag_in_event` (any of the three lags inside an event)."""
    series = common.series_table().select("time_series_id", "site")
    label_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    pieces = []
    for series_id, part in frame.select(*common.KEYS).group_by(
        "time_series_id", maintain_order=True
    ):
        label = label_of[series_id[0]]
        target = np.asarray(common.grid_index(times=part["valid_time"]))
        origin = np.asarray(common.grid_index(times=part["power_fcst_init_time"]))
        flags = fit.lag_flags(mask=masks[label], target=target, origin=origin)
        pieces.append(part.with_columns(lag_in_event=pl.Series(flags.any(axis=1))))
    return pl.concat(pieces)


def widen(
    *, frame: pl.DataFrame, rows: pl.Series | pl.Expr | bool, factor: float | pl.Expr | pl.Series
) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor` where `rows` is true."""
    median = pl.col("q50")
    return frame.with_columns(
        pl.when(rows).then(median + (pl.col(c) - median) * factor).otherwise(pl.col(c)).alias(c)
        for c in LEVELS
    )


def write_arm(*, name: str, frame: pl.DataFrame, out_dir: Path) -> None:
    """Write one arm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_dir / f"{name}.parquet")
    print(f"wrote {name}: {frame.height:,} rows", flush=True)


def main() -> None:
    """Assemble and write every arm that has files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="read arm_preds_smoke, write arms_smoke"
    )
    args = parser.parse_args()
    arm_dir, out_dir = (SMOKE_ARM_DIR, SMOKE_OUT_DIR) if args.smoke else (ARM_DIR, OUT_DIR)
    series = common.series_table()
    sites = series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    masks = fit.event_masks()
    arms: dict[str, pl.DataFrame] = {}
    for name in OWN_ARMS:
        frame = read_arm(
            directory=arm_dir, name=name, sites=sites, refits=MONTHLY_REFITS, stored=False
        )
        if frame is None:
            print(f"WARNING: no files for arm {name} in {arm_dir}; skipped", flush=True)
            continue
        arms[name] = frame
    stored = read_arm(
        directory=STORED_87 / "anchor_preds", name="n1h", sites=sites, refits=WEEKLY_REFITS,
        stored=True,
    )  # fmt: skip
    if stored is None:
        print("WARNING: no stored weekly member (arm d); skipped", flush=True)
    else:
        arms[STORED_ARM] = stored
    if not arms:
        msg = f"no arm has files in {arm_dir}"
        raise ValueError(msg)
    keys = list(common.KEYS)
    shared = None
    for frame in arms.values():
        shared = frame.select(keys) if shared is None else shared.join(frame.select(keys), on=keys)
    arms = {n: f.join(shared, on=keys, how="semi").sort(keys) for n, f in arms.items()}
    first = next(iter(arms.values()))
    for name, frame in arms.items():
        if not first.select(keys).equals(frame.select(keys)):
            msg = f"arm {name} keys differ after the join"
            raise ValueError(msg)
    flags = contaminated_flags(frame=first, masks=masks).sort(keys)
    if not first.select(keys).equals(flags.select(keys)):
        msg = "contaminated-lag keys differ from the arm keys"
        raise ValueError(msg)
    print(f"arms: {list(arms)}; shared rows: {first.height:,}", flush=True)
    write_arm(name="contaminated", frame=flags, out_dir=out_dir)
    for name, frame in arms.items():
        write_arm(name=name, frame=frame, out_dir=out_dir)
        write_arm(
            name=f"{name}_w", frame=widen(frame=frame, rows=True, factor=FIXED_K), out_dir=out_dir
        )


if __name__ == "__main__":
    main()
