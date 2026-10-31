r"""Injected-transfer check (NRA check 3) for experiment 392: refit with synthetic transfers added.

Fixed design. Fold: the 2026-01-01 refit only. Primaries: D5 and D10 (signed-MW) and D3 and D4
(MVA-only). Before the refit, synthetic load-shift events are added to each Primary's training
history (valid times up to 3.5 hours before the refit): a step of depth +/- (0.71, 1.1, 2.3) MW over
capacity (signs alternate, depths cycle) and a duration of 12, 24, or 72 hours drawn with a seed
set by the site. Events start at random half-hours in clean training periods (outside the padded
hand-label mask and the 48 hours after it), never overlap, and together cover
`COVERAGE_SHARE` of the training span. The step is added to the observed power (the target) and to
every lag column whose time lies inside an event, as a real transfer would be.

Arms, each in a `lagged` and a `lagless` form: `r0` (no rows dropped) and `r1` (drops training rows
inside a hand-labelled or injected event, or within 48 hours after one). The fit uses experiment
392's parameters through `e00392_fit`, `e00383_fit`, and `e00087_fit_demand`. Test-month lags are
clean. Forecasts go to `work/injected/arm_preds/<arm>_<form>/`, boosters to
`work/injected/boosters/<arm>_<form>/`.

`score` compares each injected-trained forecast with the clean-trained forecast of the same arm
and form (stored in `e00392_score.SOURCES`), on test-month rows (valid time in 2026-01) with lead
up to `HEADLINE_LEAD_HOURS` that are outside hand-labelled events and the 48 hours after them. No
injected event reaches the test month. It prints the mean absolute q50 difference (points of
capacity), the CRPS change (points), and the depth recovered by `r1`:
`1 - error(r1 injected vs r1 clean) / error(r0 injected vs r0 clean)` for the same form.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets DATA_PATH_INTERNAL=... \\
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00392-*/e00392_injected.py \\
    fit --site D5 --arm r1 --form lagged [--smoke]
# then: ... e00392_injected.py score
```
"""

import argparse
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00383_fit as base  # ty: ignore[unresolved-import]
import e00392_fit as fit  # ty: ignore[unresolved-import]
import e00392_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import KEY_COLUMNS

REFIT: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
FOLD_END: Final[datetime] = datetime(2026, 2, 1, tzinfo=UTC)
SITES: Final[tuple[str, ...]] = ("D5", "D10", "D3", "D4")
ARMS: Final[tuple[str, ...]] = ("r0", "r1")
FORMS: Final[tuple[str, ...]] = ("lagged", "lagless")
DEPTHS_MW: Final[tuple[float, ...]] = (0.71, 1.1, 2.3)
DURATION_STEPS: Final[tuple[int, ...]] = (24, 48, 144)
"""12, 24, and 72 hours in half-hours."""
COVERAGE_SHARE: Final[float] = 0.04
MAX_PLACEMENT_TRIES: Final[int] = 200
LAST_TRAINING_LAG_STEPS: Final[int] = 7
"""A refit sees valid times up to 3.5 hours (7 half-hours) before it."""
HEADLINE_LEAD_HOURS: Final[float] = 240.0
WORK_DIR: Final[Path] = fit.WORK_DIR / "injected"
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
SMOKE_ROUNDS: Final[int] = 2


def clean_period_mask(*, padded: np.ndarray) -> np.ndarray:
    """Return the steps that are NOT clean: padded hand-label mask and the 48 hours after it."""
    csum = np.concatenate([[0], np.cumsum(padded.astype(np.int64))])
    after = fit.POST_EVENT_STEPS
    low = np.maximum(np.arange(padded.size) - after, 0)
    return (csum[np.arange(padded.size) + 1] - csum[low]) > 0


def inject_events(
    *, site: str, padded: np.ndarray, first: int, last: int, capacity_mw: float
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return the injected step on the grid (fraction of capacity) and the events' `[start, end)`.

    Events are placed at random onsets in `[first, last]`, outside `clean_period_mask`, without
    overlap, until their durations cover `COVERAGE_SHARE` of the span. The generator is seeded by
    the site.
    """
    generator = np.random.default_rng(zlib.crc32(f"injected:{site}".encode()))
    blocked = clean_period_mask(padded=padded)
    step = np.zeros(padded.size, dtype=np.float32)
    events: list[tuple[int, int]] = []
    covered, goal = 0, COVERAGE_SHARE * (last - first + 1)
    while covered < goal:
        length = int(generator.choice(DURATION_STEPS))
        for _ in range(MAX_PLACEMENT_TRIES):
            start = int(generator.integers(first, last + 2 - length))
            if not blocked[start : start + length].any():
                break
        else:
            msg = f"{site}: no clean placement for an event of {length} steps"
            raise ValueError(msg)
        sign = 1.0 if len(events) % 2 == 0 else -1.0
        depth = sign * DEPTHS_MW[len(events) % len(DEPTHS_MW)] / capacity_mw
        step[start : start + length] = np.float32(depth)
        blocked[start : start + length] = True
        events.append((start, start + length))
        covered += length
    return step, sorted(events)


def injected_training_rows(
    *, known: pl.DataFrame, step: np.ndarray, capacity_mw: float
) -> pl.DataFrame:
    """Return `known` with `step` added to the power (MW) and to each lag at a time it covers."""
    target = np.asarray(common.grid_index(times=known["valid_time"]))
    origin = np.asarray(common.grid_index(times=known["power_fcst_init_time"]))
    target_step = np.nan_to_num(common.take(values=step, index=target))
    lag_step = np.nan_to_num(common.origin_lags(power=step, target=target, origin=origin))
    return known.with_columns(
        (pl.col("power") + pl.Series(target_step * np.float32(capacity_mw))).alias("power"),
        *[
            (pl.col(name) + pl.Series(lag_step[:, i])).alias(name)
            for i, name in enumerate(common.LAG_COLUMNS)
        ],
    )


def run_fit(
    *, site: str, arm_name: str, form: str, rounds: int, out_dir: Path, device: str
) -> None:
    """Fit one injected arm and form for one Primary, and forecast the fold's runs."""
    name = f"{arm_name}_{form}"
    arm = base.Arm(
        name, drop_event_targets=False, mask_lags=False, pseudo=False, use_lags=form == "lagged"
    )
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    seed = common.SEED
    started = time.monotonic()
    known = fd.known_rows(
        rows=base.flagged_training_rows(row=row, arrays=arrays[site], with_pseudo=False),
        refit=REFIT,
        config=config,
    )
    target = np.asarray(common.grid_index(times=known["valid_time"]))
    refit_step = int(common.grid_index(times=REFIT))
    step, events = inject_events(
        site=site,
        padded=base.worker_state()[0][site],
        first=int(target.min()),
        last=refit_step - LAST_TRAINING_LAG_STEPS,
        capacity_mw=row["effective_capacity_mw"],
    )
    known = injected_training_rows(known=known, step=step, capacity_mw=row["effective_capacity_mw"])
    if arm_name == "r1":
        drop = fit.drop_mask(
            runs=[*fit.hand_runs(site=site), *events],
            refit_step=refit_step,
            size=common.grid_times().len(),
        )
        known = known.filter(
            pl.Series(common.take(values=drop.astype(np.float32), index=target) < 0.5)
        )
    model = xgb.train(
        {**fd.parameters(config=config, seed=seed, nthread=1), "device": device},
        base.train_matrix(known=known, row=row, refit=REFIT, arm=arm),
        num_boost_round=rounds,
    )
    booster = BOOSTER_DIR if out_dir == ARM_DIR else out_dir.parent / "boosters_smoke"
    booster_path = booster / name / f"{site}_s{seed}_{REFIT:%Y%m%d}.ubj"
    booster_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(booster_path)
    print(
        f"{name} {site}: {len(events)} injected events, {known.height:,} rows, "
        f"fit {time.monotonic() - started:.0f} s",
        flush=True,
    )
    origins = next(
        runs
        for refit, runs in regimes.jobs_for(
            regime_names=("monthly",), origins=common.all_origins()
        ).items()
        if refit == REFIT
    )
    if rounds == SMOKE_ROUNDS:
        origins = origins[:3]
    tables = []
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[site],
            a7=False,
        )
        tables.append(base.forecast_run(model=model, prepared=prepared, arm=arm, seed=seed))
    common.atomic_write(
        frame=pl.concat(tables),
        target=fd.job_path(root=out_dir, config=name, site=site, seed=seed, refit=REFIT),
    )


def clean_test_rows(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the keys of test-month rows outside hand-labelled events and the 48 hours after."""
    masks, _ = base.worker_state()
    grid = common.grid_times()
    pieces = []
    for series_id, label in zip(series["time_series_id"], series["label"], strict=True):
        runs = fit.unpadded_runs(padded=masks[label])
        flagged = fit.drop_mask(runs=runs, refit_step=grid.len(), size=grid.len())
        in_event = np.zeros(grid.len(), dtype=bool)
        for start, end in runs:
            in_event[start:end] = True
        bad = flagged | in_event
        pieces.append(
            pl.DataFrame({"valid_time": grid, "clean": ~bad}).with_columns(
                time_series_id=pl.lit(series_id, dtype=series.schema["time_series_id"])
            )
        )
    return pl.concat(pieces).filter(pl.col("clean")).select("time_series_id", "valid_time")


def read_forecast(*, path: Path, series: pl.DataFrame, clean: pl.DataFrame) -> pl.DataFrame:
    """Return one stored forecast in MW, on clean test-month rows up to the headline lead."""
    stored = pl.read_parquet(path, columns=[*KEY_COLUMNS, *score.QUANTILES])
    frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return frame.filter(
        pl.col("power_fcst_init_time") < FOLD_END,
        pl.col("valid_time") < FOLD_END,
        score.lead_expression() <= HEADLINE_LEAD_HOURS,
    ).join(clean, on=["time_series_id", "valid_time"], how="semi")


def run_score() -> None:
    """Print the injected-transfer comparison per arm, form, and Primary, and pooled."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    series = adapter.series_table(labels=SITES)
    actuals, cache_actuals, _ = score.load_actuals(source="cache", series=series)
    thresholds = metrics.twcrps_thresholds(
        history=cache_actuals.rename({"actual": "power_mw"}),
        series=series,
        before=score.SCORED_FIRST,
    )
    clean = clean_test_rows(series=series)
    refit_name = f"{common.SEED}_{REFIT:%Y%m%dT%H%M}"
    rows = []
    for form in FORMS:
        for arm in ARMS:
            clean_dir = score.SOURCES[f"{arm}_{form}"]
            for label in SITES:
                sid = series.filter(pl.col("label") == label)
                name = f"{label}_s{refit_name}.parquet"
                injected = read_forecast(
                    path=ARM_DIR / f"{arm}_{form}" / name, series=sid, clean=clean
                )
                original = read_forecast(path=clean_dir / name, series=sid, clean=clean)
                joined = injected.join(
                    original, on=list(KEY_COLUMNS), suffix="_clean", validate="1:1"
                )
                losses = {}
                for tag, frame in (("inj", injected), ("clean", original)):
                    shared = frame.join(
                        joined.select(*KEY_COLUMNS), on=list(KEY_COLUMNS), how="semi"
                    )
                    losses[tag] = metrics.row_losses(
                        forecast=shared, actuals=actuals, series=sid, thresholds=thresholds
                    )["crps"]
                capacity = float(sid["capacity_mw"][0])
                rows.append(
                    {
                        "form": form,
                        "arm": arm,
                        "label": label,
                        "n": joined.height,
                        "abs_q50_diff_points": float(
                            joined.select(
                                (
                                    (pl.col("q50") - pl.col("q50_clean")).abs() / capacity * 100
                                ).mean()
                            ).item()
                        ),
                        "sum_diff": float(
                            joined.select(
                                ((pl.col("q50") - pl.col("q50_clean")).abs() / capacity * 100).sum()
                            ).item()
                        ),
                        "crps_injected": float(losses["inj"].to_numpy().mean()),
                        "crps_clean": float(losses["clean"].to_numpy().mean()),
                    }
                )
    table = pl.DataFrame(rows).with_columns(
        crps_change_points=pl.col("crps_injected") - pl.col("crps_clean"),
        crps_change_percent=100 * (pl.col("crps_injected") / pl.col("crps_clean") - 1),
    )
    print("\n== Injected vs clean-trained, per arm, form, and Primary ==")
    print(table.drop("sum_diff"))
    pooled = (
        table.group_by("form", "arm")
        .agg(
            pl.col("n").sum(),
            (pl.col("sum_diff").sum() / pl.col("n").sum()).alias("abs_q50_diff_points"),
            (pl.col("crps_injected") * pl.col("n")).sum().alias("ci"),
            (pl.col("crps_clean") * pl.col("n")).sum().alias("cc"),
        )
        .with_columns(
            crps_change_points=(pl.col("ci") - pl.col("cc")) / pl.col("n"),
            crps_change_percent=100 * (pl.col("ci") / pl.col("cc") - 1),
        )
        .drop("ci", "cc")
        .sort("form", "arm")
    )
    print("\n== Pooled over the four Primaries (row-weighted) ==")
    print(pooled)
    r0 = table.filter(pl.col("arm") == "r0").select(
        "form", "label", denominator=pl.col("abs_q50_diff_points")
    )
    recovered = (
        table.filter(pl.col("arm") == "r1")
        .join(r0, on=["form", "label"])
        .select(
            "form",
            "label",
            depth_recovered=1 - pl.col("abs_q50_diff_points") / pl.col("denominator"),
        )
        .sort("form", "label")
    )
    print("\n== Depth recovered by r1 = 1 - error(r1) / error(r0), per Primary ==")
    print(recovered)
    wide = pooled.select("form", "arm", "abs_q50_diff_points").pivot(
        on="arm", index="form", values="abs_q50_diff_points"
    )
    print("\n== Depth recovered by r1, pooled ==")
    print(wide.with_columns(depth_recovered=1 - pl.col("r1") / pl.col("r0")))


def main() -> None:
    """Fit one injected arm, or score the 16 fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--site", required=True, choices=SITES)
    fit_parser.add_argument("--arm", required=True, choices=ARMS)
    fit_parser.add_argument("--form", required=True, choices=FORMS)
    fit_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit_parser.add_argument("--smoke", action="store_true", help="2 rounds, 3 runs")
    commands.add_parser("score")
    args = parser.parse_args()
    if args.command == "score":
        run_score()
        return
    out_dir = WORK_DIR / "arm_preds_smoke" if args.smoke else ARM_DIR
    run_fit(
        site=args.site,
        arm_name=args.arm,
        form=args.form,
        rounds=SMOKE_ROUNDS if args.smoke else common.N_ROUNDS,
        out_dir=out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
