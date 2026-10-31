r"""Score the screen's arms fold by fold against the control and print the arm table.

For each fold, every arm is restricted to the half-hours all the compared arms share, scored with
`studies.eval_protocol` (days 1 to 10 headline), and compared with C0 in two groups (the signed-MW
six and the MVA-only eight). The refit noise floor is the pooled absolute C1 minus C0 difference of
the same fold and group. Nothing here makes a 99% claim: a screen prints point differences, the
floor, the sign, and the number of half-hours. Tail coverage and the slice scores of the plan
(winter, top demand quartile, cold nights, sunny, holiday days) are printed beside the headline.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \\
    uv run --no-sync python ml-experiments/00396-*/e00396_score.py --folds 2026-01 \\
    --sites D2 D3 D4 D8 D5 D9 D10 D11
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as base_score  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00396-xgboost-recipe-screen-demand-calendar-winter-hyperparameters-ensemble-summary"
    / "work"
)
PRED_DIR: Final[Path] = WORK / "arm_preds"
HEADLINE: Final[str] = base_score.HEADLINE
GROUPS: Final[dict[str, pl.Expr]] = {
    "signed-MW": pl.col("group") == "signed-MW",
    "MVA-only": pl.col("group") == "MVA-only",
}
SLICES: Final[tuple[str, ...]] = (
    "all",
    "winter",
    "winter_nights",
    "cold_nights",
    "sunny_unit_shape",
    "christmas",
    "bank_holidays",
)
TAIL_COLUMNS: Final[tuple[str, ...]] = ("coverage_q95", "coverage_q99")
CONTROL: Final[str] = "C0"
SECOND_SEED: Final[str] = "C1"
WIDENING: Final[float] = 1.15
CLEANED_POWER_DELTA: Final[str] = str(
    ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "cleaned_power_time_series.delta"
)
"""The cleaned power Delta table, which keeps flagged half-hours with a `drop_reason`."""


def flagged_keys() -> pl.DataFrame:
    """Return the flagged half-hours (`drop_reason` set, such as `substation_zero`) of every series.

    The cleaned power table keeps flagged rows with a reason; readers of cleaned power drop them.
    The keys are `time_series_id` and `valid_time`, both on the table's period-ending stamps.
    """
    return (
        pl.scan_delta(CLEANED_POWER_DELTA)
        .filter(pl.col("drop_reason").is_not_null())
        .select("time_series_id", valid_time=pl.col("time"))
        .collect()
    )


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return the frame with each quantile moved away from the median by `factor`."""
    columns = [c for c in frame.columns if c.startswith("q") and c[1:].isdigit()]
    return frame.with_columns(
        *[
            (pl.col("q50") + factor * (pl.col(c) - pl.col("q50"))).alias(c)
            for c in columns
            if c != "q50"
        ]
    )


def arm_frame(*, arm: str, fold: str, sites: list[str], series: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecasts for one fold and the given Primaries, in MW."""
    files = [PRED_DIR / arm / fold / f"{s}_s{_seed(arm=arm)}.parquet" for s in sites]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        msg = f"{arm} {fold}: missing {missing}"
        raise FileNotFoundError(msg)
    stored = pl.concat([pl.read_parquet(f) for f in files])
    return adapter.forecast_frame(
        stored=stored, series=series, unit="fraction", seed=_seed(arm=arm)
    )


def _seed(*, arm: str) -> int:
    return 1 if arm == SECOND_SEED else 0


def available_sites(*, arms: list[str], fold: str, sites: list[str]) -> list[str]:
    """Return the sites for which every arm has a forecast file in the fold."""
    return [
        s
        for s in sites
        if all((PRED_DIR / a / fold / f"{s}_s{_seed(arm=a)}.parquet").exists() for a in arms)
    ]


def fold_table(
    *, fold: str, arms: list[str], sites: list[str], ctx: dict, factor: float
) -> list[dict]:
    """Print the arm table of one fold and return it as one record per arm and group.

    The records hold the headline CRPS difference from C0, the slice differences, both tail
    coverages, the C0 CRPS, the noise floor, and the number of scored half-hours.
    """
    records: list[dict] = []
    series = adapter.series_table()
    names = [CONTROL, SECOND_SEED, *[a for a in arms if a not in (CONTROL, SECOND_SEED)]]
    sites = available_sites(arms=names, fold=fold, sites=sites)
    if not sites:
        print(f"{fold}: no site has every arm yet")
        return records
    chosen = series.filter(pl.col("label").is_in(sites))
    frames = {a: arm_frame(arm=a, fold=fold, sites=sites, series=chosen) for a in names}
    if factor != 1.0:
        frames = {a: widen(frame=f, factor=factor) for a, f in frames.items()}
    flagged = ctx["flagged"].cast({"time_series_id": chosen.schema["time_series_id"]})
    in_fold = (
        frames[CONTROL]
        .select("time_series_id", "valid_time")
        .unique()
        .join(flagged, on=["time_series_id", "valid_time"])
        .join(chosen.select("time_series_id", "label"), on="time_series_id")
    )
    print(f"flagged half-hours dropped from the scored rows, {fold}: {in_fold.height}")
    print(in_fold.group_by("label").len().sort("label").transpose(include_header=True))
    ctx = {
        **ctx,
        "actuals": ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti"),
    }
    table, _, rows = base_score.score_arms(arms=frames, series=chosen, ctx=ctx)
    print(f"\n=== fold {fold}, widening {factor}, sites {sites}, half-hours {table.height} ===")
    for gname, where in GROUPS.items():
        control = rows[CONTROL].filter(where & pl.col(HEADLINE))
        if control.is_empty():
            continue
        floor = (
            rows[SECOND_SEED].filter(where & pl.col(HEADLINE))["crps"].mean()
            - control["crps"].mean()
        )
        print(
            f"\n{gname}: C0 CRPS {control['crps'].mean():.3f}, noise floor |C1-C0| {abs(floor):.3f}"
        )
        out = []
        for arm in names[1:]:
            line = {"arm": arm}
            for sl in SLICES:
                cond = where & pl.col(HEADLINE) & (pl.lit(True) if sl == "all" else pl.col(sl))
                a = rows[arm].filter(cond)
                c = rows[CONTROL].filter(cond)
                line[sl] = round(a["crps"].mean() - c["crps"].mean(), 3) if c.height else None
            cond = where & pl.col(HEADLINE)
            for tail in TAIL_COLUMNS:
                line[tail] = round(rows[arm].filter(cond)[tail].mean(), 3)
            out.append(line)
            records.append(
                {
                    "fold": fold,
                    "group": gname,
                    "widening": factor,
                    "sites": len(sites),
                    "half_hours": int(table.height),
                    "control_crps": round(control["crps"].mean(), 4),
                    "noise_floor": round(abs(floor), 4),
                    **line,
                }
            )
        print(pl.DataFrame(out).sort("all"))
    return records


def main() -> None:
    """Print the arm table of each requested fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", required=True)
    parser.add_argument("--arms", nargs="*", default=None, help="default: every arm on disk")
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--widening", type=float, nargs="+", default=[1.0, WIDENING])
    parser.add_argument("--out", type=Path, default=None, help="write the records to this CSV")
    args = parser.parse_args()
    ctx = base_score.build_context(series=adapter.series_table())
    ctx["flagged"] = flagged_keys()
    records: list[dict] = []
    for fold in args.folds:
        arms = args.arms or sorted(
            p.name for p in PRED_DIR.iterdir() if any((p / fold).glob("*.parquet"))
        )
        for factor in args.widening:
            records += fold_table(fold=fold, arms=arms, sites=args.sites, ctx=ctx, factor=factor)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_csv(args.out)


if __name__ == "__main__":
    main()
