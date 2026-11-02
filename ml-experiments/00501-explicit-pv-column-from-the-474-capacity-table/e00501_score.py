# ruff: noqa: E501
r"""Score experiment 501's solar-output-column arm (`APV`) against experiment 425's arm A.

- `mde`: before any contrast, print the minimum detectable effect from arm A's five stored seeds
  alone (no `APV` forecast is read).
- `score`: print the checks, absolute scores, the three tested contrasts, and the audit items.

Scored rows are experiment 425's: cleaned actuals (flagged exact-zero half-hours dropped), days 1
to 10 for the headline, five folds, forecasts widened by 1.15 about the median. The sunny-midday
flag is experiment 416's (CAMS clear-sky index of 0.7 or more, 10:00 to 15:00 UK local time).

```bash
source <(cat /tmp/e501_env.sh)   # PYTHONPATH of e00501_fit.py
uv run --no-sync python ml-experiments/00501-*/e00501_score.py mde
uv run --no-sync python ml-experiments/00501-*/e00501_score.py score
```
"""

import argparse
import itertools
import json
from datetime import timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00425_score as s425  # ty: ignore[unresolved-import]
import e00501_capacity as capacity
import e00501_fit as fit501
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MONTHS: Final[tuple[str, ...]] = capacity.FOLDS
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_TESTED_CONTRASTS: Final[int] = 3
LEVEL: Final[float] = 100.0 - 5.0 / N_TESTED_CONTRASTS
N_RESAMPLES: Final[int] = 20_000
SUNNY_CSI: Final[float] = 0.7
MIDDAY_START_HOUR: Final[int] = 10
MIDDAY_END_HOUR: Final[int] = 15
SIX: Final[tuple[str, ...]] = capacity.SIGNED_SIX
CAMS_FRAME_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
OUT_DIR: Final = fit501.WORK_DIR
FIGURE_TABLE_DIR: Final = OUT_DIR / "figure_tables"
KEYS: Final[list[str]] = list(metrics.KEY_COLUMNS)
SLICE_WHERE: Final[dict[str, pl.Expr]] = {
    "C1 sunny-midday half-hours": pl.col("sunny_midday"),
    "C2 all daylight half-hours": pl.col("daylight"),
    "C3 all half-hours": pl.lit(value=True),
}


def slice_flags(*, table: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Return `table` with `sunny_midday` and `daylight` flags from the realised CAMS values.

    A valid time is the end of its half-hour. Daylight means the clear-sky irradiance is positive.
    """
    cams = (
        pl.read_parquet(CAMS_FRAME_PATH, columns=["site", "kind", "time", "ghi_u", "ghi_u_clear"])
        .filter(pl.col("kind") == "substation", pl.col("ghi_u_clear") > 0)
        .select(
            label=pl.col("site"),
            valid_time=pl.col("time"),
            csi=pl.col("ghi_u") / pl.col("ghi_u_clear"),
        )
    )
    start = (pl.col("valid_time") - timedelta(minutes=30)).dt.convert_time_zone("Europe/London")
    keyed = (
        table.select(KEYS)
        .join(series.select("time_series_id", "label"), on="time_series_id", how="left")
        .join(cams, on=["label", "valid_time"], how="left")
        .with_columns(
            daylight=pl.col("csi").is_not_null(),
            sunny_midday=(
                (pl.col("csi").is_finite() & (pl.col("csi") >= SUNNY_CSI))
                & (start.dt.hour() >= MIDDAY_START_HOUR)
                & (start.dt.hour() < MIDDAY_END_HOUR)
            ).fill_null(value=False),
        )
        .select(*KEYS, "daylight", "sunny_midday")
    )
    return table.join(keyed, on=KEYS, how="left", validate="1:1")


def add_flags(*, rows: dict[str, pl.DataFrame], table: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Join the slice flags of `table` onto every arm's row losses."""
    flags = table.select(*KEYS, "daylight", "sunny_midday")
    return {n: r.join(flags, on=KEYS, how="left", validate="m:1") for n, r in rows.items()}


def arm_files(*, name: str, folder: str, seeds: tuple[int, ...]) -> s425.ArmFiles:
    """Return an arm's file set; an absolute `folder` points outside experiment 425's folder."""
    return s425.ArmFiles(name, folder, seeds, MONTHS)


def widened(*, files: s425.ArmFiles, series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's seed-averaged forecast in MW, widened by 1.15."""
    stored = s425.seed_average(arm=files)
    return adapter.forecast_frame(
        stored=s407.widen_global(frame=stored), series=series, unit="fraction"
    )


def seed_frames(*, series: pl.DataFrame, seeds: tuple[int, ...]) -> dict[str, pl.DataFrame]:
    """Return arm A (experiment 425's seeds) and `APV` (this experiment's), one frame each."""
    apv_folder = str(fit501.SEED_DIR)
    return {
        "A_w115": widened(files=arm_files(name="A", folder="A", seeds=seeds), series=series),
        "APV_w115": widened(
            files=arm_files(name="APV", folder=apv_folder, seeds=seeds), series=series
        ),
    }


def monthly_means(*, rows: pl.DataFrame, flag: pl.Expr) -> dict[str, float]:
    """Return the signed-six mean CRPS in a slice (days 1 to 10) per fold month."""
    kept = rows.filter(
        pl.col("label").is_in(list(SIX)), pl.col("lead_days_1_10"), flag
    ).with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    return dict(kept.group_by("month").agg(pl.col("crps").mean()).iter_rows())


def run_mde() -> None:
    """Print the minimum detectable effect from arm A's five seeds (sunny-midday CRPS, six)."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    per_seed = {}
    table = None
    for seed in range(5):
        frame = widened(files=arm_files(name="A", folder="A", seeds=(seed,)), series=series)
        table_s, _, rows = s381.score_arms(arms={"A": frame}, series=series, ctx=ctx)
        if table is None:
            table = slice_flags(table=table_s, series=series)
        rows = add_flags(rows=rows, table=table)
        per_seed[seed] = {
            sl: monthly_means(rows=rows["A"], flag=pl.col(flag))
            for sl, flag in (
                ("sunny_midday", "sunny_midday"),
                ("daylight", "daylight"),
                ("all", "all"),
            )
        }
    assert table is not None
    print(
        f"shared half-hours {table.height}; sunny-midday half-hours {int(table['sunny_midday'].sum())}"
    )
    out = {}
    for sl in ("sunny_midday", "daylight", "all"):
        sds = []
        for s1, s2 in itertools.combinations(range(5), 2):
            diff = [per_seed[s1][sl][m] - per_seed[s2][sl][m] for m in MONTHS]
            sds.append(float(np.std(diff, ddof=1)))
        sd_pair = float(np.mean(sds))
        sd_seed = sd_pair / np.sqrt(2)
        mde_3 = 2.0 * sd_seed * np.sqrt(2.0 / 3.0)
        out[sl] = {"sd_pair": sd_pair, "mde_3_seeds": mde_3}
        print(
            f"{sl}: mean SD over folds of a two-seed difference {sd_pair:.4f}; one seed's SD "
            f"{sd_seed:.4f}; MDE for a contrast of two 3-seed averages (2 x SD) {mde_3:.4f} points"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mde.json").write_text(json.dumps(out, indent=1))


def contrast(*, rows: dict, table: pl.DataFrame, where: pl.Expr, group: tuple[str, ...]) -> dict:
    """Return the month-block contrast `APV - A` of CRPS on a slice for a group."""
    keep = pl.col("label").is_in(list(group)) & pl.col("lead_days_1_10") & where
    r = metrics.month_bootstrap_contrast(
        treatment=rows["APV_w115"],
        reference=rows["A_w115"],
        metric="crps",
        where=keep,
        level_percent=LEVEL,
        n_resamples=N_RESAMPLES,
        slice_table=table,
    )
    return {k: float(r[k]) for k in ("difference", "lower", "upper", "n_half_hours")}


def print_checks(*, frames: dict[str, pl.DataFrame], ctx: dict) -> None:
    """Print the fine-tuning rows dropped per fold, the capacities, and the audit "Checks"."""
    print(
        "\n== rows dropped per fold (fit logs of this experiment, seed 0) and capacities (MW AC) =="
    )
    for month in MONTHS:
        meta = json.loads((fit501.MODEL_DIR / month / "s0" / "meta.json").read_text())
        print(month, meta["dropped"], {k: round(v, 3) for k, v in meta["capacity_mw"].items()})
    print(
        "\n== median minus mean forecast per Primary (MW, days 1 to 10, mean of the 11 quantiles) =="
    )
    levels = list(common.LEVEL_COLUMNS)
    actuals = ctx["actuals"]
    for name in ("A_w115", "APV_w115"):
        frame = frames[name].join(
            actuals.filter(pl.col("actual").is_not_nan()), on=["time_series_id", "valid_time"]
        )
        frame = frame.filter(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")) < timedelta(hours=240)
        ).with_columns(mean_forecast=pl.concat_list(levels).list.mean())
        rows = frame.group_by("time_series_id").agg(
            median_minus_mean=(pl.col("q50") - pl.col("mean_forecast")).mean(),
            mae_median=(pl.col("q50") - pl.col("actual")).abs().mean(),
            mae_mean=(pl.col("mean_forecast") - pl.col("actual")).abs().mean(),
        )
        print(name, rows.sort("time_series_id").to_dicts())


def main() -> None:
    """Run the `mde` or the `score` command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mde", "score"))
    args = parser.parse_args()
    if args.command == "mde":
        run_mde()
        return
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    pair = seed_frames(series=series, seeds=SEEDS)
    # Contrasts and tables on the two arms alone (the largest equal row set).
    table, _scored, rows = s381.score_arms(arms=pair, series=series, ctx=ctx)
    table = slice_flags(table=table, series=series)
    rows = add_flags(rows=rows, table=table)
    print(
        f"shared half-hours (A and APV): {table.height}; sunny-midday {int(table['sunny_midday'].sum())}"
    )
    mde = json.loads((OUT_DIR / "mde.json").read_text())
    print(
        "\n== minimum detectable effect (from arm A seeds 0 to 4, printed before the contrasts) ==",
        mde,
    )
    print(
        f"\n== tested contrasts APV minus A, CRPS days 1 to 10, level {LEVEL:.2f}% (Bonferroni count {N_TESTED_CONTRASTS}) =="
    )
    summary: dict = {"mde": mde}
    for name, where in SLICE_WHERE.items():
        for group_name, group in (
            ("signed-MW six", SIX),
            ("MVA-only eight", tuple(s425.SUBSETS["MVA-only eight"])),
            ("D15 and D18", ("D15", "D18")),
        ):
            r = contrast(rows=rows, table=table, where=where, group=group)
            summary[f"{name} | {group_name}"] = r
            print(
                f"{name:28s} {group_name:16s} {r['difference']:+.4f} ({r['lower']:+.4f} to {r['upper']:+.4f}) n={int(r['n_half_hours'])}"
            )
    # Per fold and leave-one-Primary-out for C1.
    print("\n== C1 per fold (signed-MW six): mean CRPS A, APV, difference ==")
    for sl, flag in (("C1", pl.col("sunny_midday")), ("C3", pl.lit(value=True))):
        a = monthly_means(rows=rows["A_w115"], flag=flag)
        b = monthly_means(rows=rows["APV_w115"], flag=flag)
        print(sl, {m: (round(a[m], 3), round(b[m], 3), round(b[m] - a[m], 3)) for m in MONTHS})
    print("\n== C1 leave one Primary out (point estimate, signed-MW six) ==")
    for left_out in SIX:
        rest = tuple(s for s in SIX if s != left_out)
        r = contrast(rows=rows, table=table, where=pl.col("sunny_midday"), group=rest)
        print(left_out, f"{r['difference']:+.4f} ({r['lower']:+.4f} to {r['upper']:+.4f})")
    for band, lo, hi in (("12 to 36 h", 12.0, 36.0), ("36 h to 10 days", 36.0, 240.0)):
        print(f"\n== CRPS in the {band} band, signed-MW six ==")
        for n, r in rows.items():
            sub = r.filter(
                pl.col("label").is_in(list(SIX)),
                pl.col("lead_hours") >= lo,
                pl.col("lead_hours") < hi,
            )
            print(
                n,
                round(float(sub.group_by("label").agg(pl.col("crps").mean())["crps"].mean()), 4),
                sub.height,
            )
    # Tables with the references on the half-hours every arm shares.
    refs = s425.load_frames(series=series, months=MONTHS, seeds=SEEDS, with_references=True)
    all_frames = {"A_w115": pair["A_w115"], "APV_w115": pair["APV_w115"]} | {
        k: v for k, v in refs.items() if k in ("S0M_w115", *s425.REFERENCE_NAMES)
    }
    s425.FIGURE_TABLE_DIR = FIGURE_TABLE_DIR
    table_all, scored_all, rows_all = s381.score_arms(arms=all_frames, series=series, ctx=ctx)
    print(f"\nshared half-hours with the references: {table_all.height}")
    s425.crps_tables(rows=rows_all)
    s425.slice_tables(scored=scored_all)
    s425.tails(rows=rows_all, table=table_all, reference=all_frames["A_w115"])
    s425.curtailment(frames=all_frames, series=series, ctx=ctx, table=table_all)
    print_checks(frames=pair, ctx=ctx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "score_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
