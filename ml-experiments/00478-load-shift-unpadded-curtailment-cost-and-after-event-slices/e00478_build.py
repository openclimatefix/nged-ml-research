"""Experiment 478 (Part F2 of experiment 395), build step: row tables and curtailment cost.

Builds row tables for experiment 391's forecasts and the avoidable-curtailment cost of every
forecast, per Primary and per event.

Nothing is refitted. The six forecasts of experiment 395 Part F keep their stored row files; the
three forecasts of experiment 391 (`b0`, `n1`, `o`) get row files built by experiment 395's code on
the same tags. The cost is `metrics.avoidable_curtailment_cost` at the signed-MW six, with the
flagged exact-zero half-hours dropped, for three row sets: targets during an event, targets in the
7 days after an event ends, and all rows.

```bash
D=ml-experiments
PYTHONPATH=$D/00377-demand-v1-first-run-feasibility:\
$D/00381-simple-weekly-xgboost-demand-against-gam-stack:\
$D/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
$D/00379-bg2-components-all-16-primaries-to-14-days:\
$D/00390-xgboost-d11-d14-gap-and-load-shift-event-lags:\
$D/00395-xgboost-level-state-winter-spread-anchor-stored-forecasts \
    systemd-run --user --scope -q -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python $D/00478-*/e00478_build.py
```
"""

import gc
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import e00395_part_f_build as pf  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = (
    DATA / "00478-load-shift-unpadded-curtailment-cost-and-after-event-slices" / "work"
)
E391_ARMS: Final[Path] = DATA / "00391-xgboost-lag-surprise-second-stage" / "work" / "arm_preds"
ARMS_391: Final[tuple[str, ...]] = ("b0", "n1", "o")
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
AFTER_HOURS: Final[float] = 168.0
KEYS: Final[list[str]] = pf.KEYS


def frame_391(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an experiment 391 arm in MW: every refit file of every Primary, concatenated."""
    files = sorted((E391_ARMS / "monthly" / arm).glob("*.parquet"))
    stored = pl.concat([pl.read_parquet(f) for f in files]).filter(
        pl.col("power_fcst_init_time") >= SCORED_FIRST,
        (pl.col("valid_time") - pl.col("power_fcst_init_time")) <= pl.duration(hours=336),
    )
    return adapter.forecast_frame(stored=stored, series=series, unit="fraction")


def all_arms(*, series: pl.DataFrame):  # noqa: ANN201
    """Yield (name, frame): the six Part F forecasts, then experiment 391's three."""
    yield from pf.arm_frames(series=series)
    for arm in ARMS_391:
        yield f"b391_{arm}", frame_391(arm=arm, series=series)


def cost_rows(
    *, name: str, frame: pl.DataFrame, tags: pl.DataFrame, flags: pl.DataFrame,
    actuals: pl.DataFrame, series: pl.DataFrame,
) -> list[pl.DataFrame]:  # fmt: skip
    """Return cost tables (pooled by Primary, and per event) of one forecast, all fractions."""
    six = series.filter(pl.col("label").is_in(SIX))
    flows = actuals.rename({"actual": "flow_mw"})
    kept = (
        frame.join(tags, on=KEYS, how="inner", validate="1:1")
        .join(flags, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .filter(pl.col("flagged").is_null())
        .join(actuals, on=["time_series_id", "valid_time"])
    )
    during = pl.col("target_event_id") >= 0
    after = (pl.col("target_event_id") < 0) & (pl.col("hours_after_end") <= AFTER_HOURS)
    sets = {"during events": during, "7 days after events": after, "all rows": pl.lit(value=True)}
    pieces: list[pl.DataFrame] = []

    def run(*, sub: pl.DataFrame, label: str, fraction: float) -> pl.DataFrame | None:
        if sub.height == 0:
            return None
        one = six.filter(pl.col("label") == label)
        out = metrics.avoidable_curtailment_cost(
            forecasts={name: sub.select(*KEYS, "actual", *metrics.QUANTILE_COLUMNS)},
            series=one,
            flows=flows,
            scored_start=SCORED_FIRST,
            fraction=fraction,
        )
        return out.select("arm", "label", "avoidable_cost_gbp", "n_node_half_hours", "status")

    for label in SIX:
        sid = six.filter(pl.col("label") == label)["time_series_id"][0]
        mine = kept.filter(pl.col("time_series_id") == sid)
        for fraction in FRACTIONS:
            for set_name, cond in sets.items():
                sub = mine.filter(cond)
                res = run(sub=sub, label=label, fraction=fraction)
                if res is not None:
                    pieces.append(
                        res.with_columns(
                            fraction=pl.lit(fraction),
                            row_set=pl.lit(set_name),
                            event_id=pl.lit(None, dtype=pl.Int64),
                        )
                    )
            for set_name, cond, column in (
                ("during events", during, "target_event_id"),
                ("7 days after events", after, "after_event_id"),
            ):
                sub_all = mine.filter(cond)
                for event in sub_all[column].unique().to_list():
                    res = run(
                        sub=sub_all.filter(pl.col(column) == event), label=label, fraction=fraction
                    )
                    if res is not None:
                        pieces.append(
                            res.with_columns(
                                fraction=pl.lit(fraction),
                                row_set=pl.lit(set_name),
                                event_id=pl.lit(int(event), dtype=pl.Int64),
                            )
                        )
    return pieces


def main() -> None:
    """Write the 391 row files and the cost table."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    table = pl.read_parquet(pf.WORK / "slice_table.parquet")
    tags = pl.read_parquet(pf.WORK / "tags.parquet")
    flags = pf.flagged_keys(series=series)
    costs: list[pl.DataFrame] = []
    for name, frame in all_arms(series=series):
        frame = frame.join(table.select(*KEYS), on=KEYS, how="semi")  # noqa: PLW2901
        print(
            name,
            "rows",
            frame.height,
            "of",
            table.height,
            "slice-table keys; series",
            frame["time_series_id"].n_unique(),
            flush=True,
        )
        if name.startswith("b391_"):
            use = table.join(frame.select(*KEYS), on=KEYS, how="semi")
            rows = b390.trimmed_rows(name=name, frame=frame, table=use, ctx=ctx, series=series)
            rows = rows.join(flags, on=["time_series_id", "valid_time"], how="left").with_columns(
                pl.col("flagged").fill_null(value=False)
            )
            extra = frame.join(ctx["actuals"], on=["time_series_id", "valid_time"]).select(
                *KEYS,
                q50_mw=pl.col("q50"),
                actual_mw=pl.col("actual"),
                q95_mw=pl.col("q95"),
                q99_mw=pl.col("q99"),
            )
            rows.join(extra, on=KEYS, how="left", validate="1:1").write_parquet(
                WORK / f"rows_{name}.parquet"
            )
            del rows, extra, use
        costs += cost_rows(
            name=name, frame=frame, tags=tags, flags=flags, actuals=ctx["actuals"], series=series
        )
        print("cost done", name, flush=True)
        del frame
        gc.collect()
    pl.concat(costs).write_parquet(WORK / "cost.parquet")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
