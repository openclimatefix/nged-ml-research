"""Score experiment 472's arms against the lagless anchors, B-mix, and K2-S1, with 454's machinery.

Copied in structure from experiment 467's `e00467_score.py` (merged, so not edited there). Reuses
experiment 454's row base, widening and table-P variants, tail summaries, month-block intervals, and
minimum detectable effect, and adds what experiment 472's pre-registered rule needs:

- `cells`: group, class, month, variant, and basis scores of every arm cell, and the contrasts
  against LL, LL-ens3, B-mix, and K2-S1 (Bonferroni count fixed at 27).
- `rows`: per-Primary contrasts with D14 and D11 removed (and the minimum detectable effect on the
  remaining rows), the cold-winter-night slice, and the median error, mean error, and bias.
- `inj`: the held-out injected load-shift worlds 200 to 202 (gate 2), against each arm's anchor.
- `events`: real-event rows by event cluster, with and without the largest event, on
  NRA-corrected and metered actuals.
- `shift`: the share of rows whose shift is below 0.01 of capacity.

```bash
<PYTHONPATH as e00472_arm.py shows, all ml-experiments folders> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00472-*/e00472_score.py all
```
"""

import argparse
import itertools
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_report as report404  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_report as report  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import e00472_arm as arm_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SCORE_DIR: Final[Path] = arm_mod.WORK_DIR / "scores"
ARM_CELLS: Final[dict[str, tuple[int, ...]]] = {
    "P100": (0, 1, 2),
    "AvgAnchor": (0, 1, 2),
    "LL": (0, 1, 2),
    "LL-ens3": (0,),
    "B-mix": (0, 1, 2),
    "K2-S1": (0, 1, 2),
    "C1e-2": (0, 1),
    "C1": (0, 1),
    "Z1e-2": (0, 1),
}
"""Each arm's stored cells (fit seeds, or the single cell of the three-seed ensemble)."""
OWN: Final[frozenset[str]] = frozenset({"P100", "AvgAnchor"})
EXPERIMENT_467: Final[frozenset[str]] = frozenset({"C1e-2", "C1", "Z1e-2"})
ANCHOR: Final[dict[str, str]] = {"P100": "LL", "AvgAnchor": "LL-ens3"}
REFERENCES: Final[tuple[str, ...]] = ("LL", "LL-ens3", "B-mix", "K2-S1")
BONFERRONI: Final[int] = 27
"""3 arms among the references times 3 row classes times 3 references (the pre-registered count)."""
LEVEL: Final[float] = 1.0 - 0.05 / BONFERRONI
LEVEL_WEIGHTS: Final[np.ndarray] = np.array(
    [0.03, 0.045, 0.10, 0.15, 0.125, 0.10, 0.125, 0.15, 0.10, 0.045, 0.03]
)
"""Weights of the 11 quantile levels that approximate the mean of each forecast distribution
(each level carries the probability mass between the midpoints to its neighbours, and the
outer levels carry the tails; the weights sum to 1)."""
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight")
INJECTED_WORLDS: Final[tuple[int, ...]] = score.SCORING_SEEDS
EXCLUSIONS: Final[dict[str, tuple[str, ...]]] = {
    "all": (),
    "no D14": ("D14",),
    "no D11": ("D11",),
    "no D14, D11": ("D14", "D11"),
}


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return a stored forecast file: experiment 472's, 467's (in 404's folder), or 454's."""
    if arm in OWN:
        return arm_mod.prediction_path(arm=arm, month=month, seed=cell, world=-1)
    if arm in EXPERIMENT_467:
        return fit.PRED_DIR / arm / f"{month}_s{cell}_w-1.parquet"
    return _ORIGINAL(arm=arm, month=month, cell=cell)


_ORIGINAL = build.prediction_path


def injected_path(*, arm: str, month: str, cell: int, world: int) -> Path:
    """Return a stored forecast file on an injected world (this experiment's or 467's)."""
    if arm in OWN:
        return arm_mod.prediction_path(arm=arm, month=month, seed=cell, world=world)
    return fit.PRED_DIR / arm / f"{month}_s{cell}_w{world}.parquet"


def contrasts(*, scores: pl.DataFrame, reference: str, column: str = "crps") -> pl.DataFrame:
    """Return each arm's pooled contrast against `reference` (month-block, Bonferroni 27)."""
    ref = scores.filter(pl.col("arm") == reference).select(
        "month", "variant", "basis", "group", "cls", ref=pl.col(column), ref_n="n"
    )
    joined = scores.filter(pl.col("arm") != reference).join(
        ref, on=["month", "variant", "basis", "group", "cls"]
    )
    rows = []
    keys = ["arm", "variant", "basis", "group", "cls"]
    for key, part in joined.group_by(*keys, maintain_order=True):
        sorted_part = part.sort("month")
        assert (sorted_part["n"] == sorted_part["ref_n"]).all(), f"{key}: rows differ"
        diff = (sorted_part[column] - sorted_part["ref"]).to_numpy()
        low, high = report.month_block_interval(diffs=diff, level=LEVEL)
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "reference": reference, "months": len(diff),
                "absolute": float(sorted_part[column].mean()),  # ty: ignore[invalid-argument-type]
                "ref_absolute": float(sorted_part["ref"].mean()),  # ty: ignore[invalid-argument-type]
                "diff": float(diff.mean()), "low": low, "high": high,
                "better": int((diff < 0).sum()),
            }
        )  # fmt: skip
    return pl.DataFrame(rows)


def score_cells(*, band: tuple[float, float] | None) -> pl.DataFrame:
    """Score every arm cell over all leads, or over a lead band in hours."""
    bases = {m: scoring.month_base(month=m) for m in build.MONTHS}
    if band is not None:
        bases = {
            m: b.filter(pl.col("lead_hours").is_between(band[0], band[1])) for m, b in bases.items()
        }
    parts: list[pl.DataFrame] = []
    for arm, cells in ARM_CELLS.items():
        for cell in cells:
            parts.extend(scoring.score_arm_cell(arm=arm, cell=cell, bases=bases))
        print("scored", arm, flush=True)
    return pl.concat(parts)


def cells_command() -> None:
    """Write the cell scores, absolute scores, contrasts, and tail contrasts."""
    value_columns = ["crps", *[f"{k}_{t}" for k in ("cov", "pin") for t in report.TAILS]]
    for name, band in (("all", None), ("lead_12_36", (12.0, 36.0))):
        cells = score_cells(band=band)
        cells.write_parquet(SCORE_DIR / f"cells_{name}.parquet")
        scores = cells.group_by("arm", "month", "variant", "basis", "group", "cls").agg(
            pl.col("n").first(), *[pl.col(c).mean() for c in value_columns]
        )
        absolute = (
            scores.filter((pl.col("variant") == "raw") & (pl.col("basis") == "metered"))
            .group_by("arm", "group", "cls")
            .agg(pl.col("crps").mean(), *[pl.col(f"{k}_{t}").mean() for k in ("cov", "pin")
                                          for t in report.TAILS])
            .sort("group", "cls", "arm")
        )  # fmt: skip
        absolute.write_csv(SCORE_DIR / f"absolute_{name}.csv", float_precision=5)
        for reference in REFERENCES:
            crps = contrasts(scores=scores, reference=reference)
            crps.write_csv(
                SCORE_DIR / f"contrasts_crps_{name}_vs_{reference}.csv", float_precision=4
            )
            tails = pl.concat(
                [
                    contrasts(scores=scores, reference=reference, column=f"{k}_{t}").with_columns(
                        metric=pl.lit(f"{k}_{t}")
                    )
                    for k in ("pin", "cov")
                    for t in report.TAILS
                ]
            )
            tails.write_csv(
                SCORE_DIR / f"contrasts_tails_{name}_vs_{reference}.csv", float_precision=6
            )
        print(name, "done", flush=True)


def cold_flags() -> np.ndarray:
    """Return the cold-night flag (primaries, grid): sun below -5 degrees, ERA5 at most 3 C."""
    path = SCORE_DIR / "cold_flags.npy"
    if path.exists():
        return np.load(path)
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)["context"]
    labels = fit.prep.primaries()["site"].to_list()
    grid_us = ctx["valid_time"].dt.epoch("us").to_numpy().astype(np.int64)
    grid = (grid_us - fit.GRID_START_US) // fit.STEP_US
    flags = np.zeros((16, int(grid.max()) + 1), dtype=bool)
    label_of = dict(series.select("time_series_id", "label").iter_rows())
    cold = (ctx["elevation_deg"].to_numpy() < -5.0) & (ctx["temperature_c"].to_numpy() <= 3.0)
    ids = ctx["time_series_id"].to_numpy()
    for p, label in enumerate(labels):
        sel = np.array([label_of.get(int(i)) == label for i in np.unique(ids)])
        member = np.isin(ids, np.unique(ids)[sel])
        flags[p, grid[member & cold]] = True
    np.save(path, flags)
    return flags


def row_stats(*, arm: str, month: str, base: pl.DataFrame) -> pl.DataFrame:
    """Return the base rows with the cell-averaged row score, median error, and mean error."""
    weights = LEVEL_WEIGHTS / LEVEL_WEIGHTS.sum()
    cells = ARM_CELLS[arm]
    crps = np.zeros(base.height)
    median_err = np.zeros(base.height)
    mean_err = np.zeros(base.height)
    for cell in cells:
        frame = scoring.load_cell(base=base, arm=arm, month=month, cell=cell)
        crps += frame.select(score.pinball_points(frame=frame)).to_series().to_numpy()
        q = frame.select(list(score.LEVEL_NAMES)).to_numpy()
        actual = frame["actual"].to_numpy()
        median_err += actual - q[:, 5]
        mean_err += actual - q @ weights
    return base.with_columns(
        crps=pl.Series(crps / len(cells)),
        median_error=pl.Series(median_err / len(cells)),
        mean_error=pl.Series(mean_err / len(cells)),
        arm=pl.lit(arm),
        month=pl.lit(month),
    )


def rows_command() -> None:
    """Write per-Primary scores, the exclusion contrasts, the cold-night slice, and errors."""
    cold = cold_flags()
    labels = np.array(fit.prep.primaries()["site"].to_list())
    parts = []
    for month in build.MONTHS:
        base = scoring.month_base(month=month)
        primary = base["primary"].to_numpy().astype(np.int64)
        target = base["target"].to_numpy().astype(np.int64)
        base = base.with_columns(
            label=pl.Series(labels[primary]),
            cold=pl.Series(
                np.where(
                    target < cold.shape[1],
                    cold[primary, np.clip(target, 0, cold.shape[1] - 1)],
                    False,
                )
            ),
        )
        parts.extend(row_stats(arm=arm, month=month, base=base) for arm in ARM_CELLS)
        print("rows", month, flush=True)
    rows = pl.concat(parts)
    rows.group_by("arm", "month", "group", "cls", "label").agg(
        pl.len().alias("n"), pl.col("crps").sum().alias("crps_sum")
    ).write_parquet(SCORE_DIR / "primary_cells.parquet")
    clean = rows.filter(pl.col("cls") == "clean")
    exclusion_rows = []
    for group in GROUPS:
        part = clean.filter(pl.col("group") == group)
        for label, drop in EXCLUSIONS.items():
            kept = part.filter(~pl.col("label").is_in(list(drop)))
            month_scores = kept.group_by("arm", "month").agg(
                pl.col("crps").mean(), pl.len().alias("n")
            )
            for arm in ("P100", "AvgAnchor", "C1e-2"):
                for reference in dict.fromkeys((*REFERENCES, ANCHOR.get(arm, "LL"))):
                    a = month_scores.filter(pl.col("arm") == arm).sort("month")
                    b = month_scores.filter(pl.col("arm") == reference).sort("month")
                    diff = (a["crps"] - b["crps"]).to_numpy()
                    low, high = report.month_block_interval(diffs=diff, level=LEVEL)
                    exclusion_rows.append(
                        {"group": group, "excluded": label, "arm": arm, "reference": reference,
                         "absolute": float(a["crps"].mean()), "diff": float(diff.mean()),  # ty: ignore[invalid-argument-type]
                         "low": low, "high": high}
                    )  # fmt: skip
        # Minimum detectable effect on the kept rows: B-mix's seed spread (454's formula).
        for label, drop in EXCLUSIONS.items():
            sds = []
            for i, j in itertools.combinations(build.CELL_SEEDS, 2):
                per = {
                    cell: np.array(
                        [
                            _cell_month_score(
                                arm="B-mix", month=m, cell=cell, group=group, drop=drop,
                                labels=labels,
                            )
                            for m in build.MONTHS
                        ]
                    )
                    for cell in (i, j)
                }  # fmt: skip
                sds.append(float(np.std(per[i] - per[j], ddof=1)))
            exclusion_rows.append(
                {"group": group, "excluded": label, "arm": "MDE", "reference": "B-mix",
                 "absolute": 0.0, "diff": 2.8 * float(np.mean(sds)) / np.sqrt(len(build.MONTHS)),
                 "low": 0.0, "high": 0.0}
            )  # fmt: skip
    pl.DataFrame(exclusion_rows).write_csv(SCORE_DIR / "exclusion_contrasts.csv", float_precision=4)
    by_primary = (
        clean.group_by("arm", "group", "label")
        .agg(pl.col("crps").mean(), pl.len().alias("n"))
        .sort("group", "label", "arm")
    )
    by_primary.write_csv(SCORE_DIR / "by_primary_clean.csv", float_precision=4)
    cold_rows = (
        rows.filter(pl.col("cold") & (pl.col("cls") == "clean"))
        .group_by("arm", "group", "month")
        .agg(pl.col("crps").mean(), pl.len().alias("n"))
        .sort("group", "arm", "month")
    )
    cold_rows.write_csv(SCORE_DIR / "cold_nights_by_month.csv", float_precision=4)
    errors = (
        clean.group_by("arm", "group", "label")
        .agg(
            pl.col("median_error").mean().alias("bias_median"),
            pl.col("mean_error").mean().alias("bias_mean"),
            pl.col("median_error").abs().mean().alias("mae_median"),
            pl.col("mean_error").abs().mean().alias("mae_mean"),
        )
        .group_by("arm", "group")
        .agg(
            pl.col("bias_median").mean(),
            pl.col("bias_mean").mean(),
            pl.col("bias_median").abs().mean().alias("mean_abs_bias_median"),
            pl.col("mae_median").mean(),
            pl.col("mae_mean").mean(),
        )
        .sort("group", "arm")
    )
    errors.write_csv(SCORE_DIR / "errors_clean.csv", float_precision=4)
    with pl.Config(tbl_rows=100, tbl_width_chars=200, tbl_cols=20):
        print(pl.DataFrame(exclusion_rows).filter(pl.col("group") == "signed_six"))
        print(errors)
        print(cold_rows.group_by("arm", "group").agg(pl.col("crps").mean(), pl.col("n").sum()))


def _cell_month_score(
    *, arm: str, month: str, cell: int, group: str, drop: tuple[str, ...], labels: np.ndarray
) -> float:
    """Return one cell's mean score in a month on clean rows of a group, leaving out Primaries."""
    base = scoring.month_base(month=month).filter(
        (pl.col("group") == group) & (pl.col("cls") == "clean")
    )
    keep = ~np.isin(labels[base["primary"].to_numpy().astype(np.int64)], drop)
    base = base.filter(pl.Series(keep))
    frame = scoring.load_cell(base=base, arm=arm, month=month, cell=cell)
    return float(frame.select(score.pinball_points(frame=frame)).to_series().mean())  # ty: ignore[invalid-return-type]


def inj_command() -> None:
    """Score gate 2: each arm against its anchor on the held-out injected worlds 200 to 202."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = report404.group_of(sites=sites)
    arms = ("P100", "AvgAnchor", "C1e-2")
    out = []
    for arm in arms:
        anchor = ANCHOR.get(arm, "LL")
        for month in build.MONTHS:
            for cell in ARM_CELLS[arm]:
                anchor_cell = cell if anchor == "LL" else 0
                anchor_q = pl.read_parquet(
                    prediction_path(arm=anchor, month=month, cell=anchor_cell)
                ).select("primary", "origin", "target", *score.LEVEL_NAMES)
                for world in INJECTED_WORLDS:
                    frame = pl.read_parquet(
                        injected_path(arm=arm, month=month, cell=cell, world=world)
                    )
                    anchored = frame.drop(list(score.LEVEL_NAMES)).join(
                        anchor_q, on=["primary", "origin", "target"], how="left"
                    )
                    assert anchored["q50"].null_count() == 0, "anchor rows differ"
                    for name, data in ((arm, frame), (anchor, anchored)):
                        classed = data.with_columns(
                            group=pl.Series(groups[data["primary"].to_numpy().astype(np.int64)]),
                            cls=pl.when(pl.col("tgt_inj"))
                            .then(pl.lit("tgt_inj"))
                            .when(pl.col("win_inj"))
                            .then(pl.lit("harm_inj"))
                            .otherwise(pl.lit("clean_inj")),
                            crps=score.pinball_points(frame=data),
                        )
                        agg = classed.group_by("group", "cls").agg(
                            pl.col("crps").mean(), pl.len().alias("n"),
                            (pl.col("actual") <= pl.col("q99")).mean().alias("cov_q99"),
                        )  # fmt: skip
                        out.append(
                            agg.with_columns(
                                arm=pl.lit(name),
                                role=pl.lit("arm" if name == arm else "anchor"),
                                of=pl.lit(arm),
                                month=pl.lit(month),
                                cell=pl.lit(cell),
                                world=pl.lit(world),
                            )
                        )
        print("inj", arm, flush=True)
    cells = pl.concat(out)
    cells.write_parquet(SCORE_DIR / "injected_cells.parquet")
    rows = []
    for (of, group, cls), part in cells.group_by("of", "group", "cls"):
        a = part.filter(pl.col("role") == "arm").group_by("month").agg(pl.col("crps").mean())
        b = part.filter(pl.col("role") == "anchor").group_by("month").agg(pl.col("crps").mean())
        j = a.join(b, on="month", suffix="_anchor").sort("month")
        diff = (j["crps"] - j["crps_anchor"]).to_numpy()
        low, high = report.month_block_interval(diffs=diff, level=0.95)
        rows.append(
            {"arm": of, "group": group, "cls": cls, "absolute": float(j["crps"].mean()),  # ty: ignore[invalid-argument-type]
             "anchor": float(j["crps_anchor"].mean()), "diff": float(diff.mean()),  # ty: ignore[invalid-argument-type]
             "low": low, "high": high, "months": len(diff)}
        )  # fmt: skip
    table = pl.DataFrame(rows).sort("group", "cls", "arm")
    table.write_csv(SCORE_DIR / "injected_contrasts.csv", float_precision=4)
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(table)


def event_ids(*, base: pl.DataFrame, real: np.ndarray) -> np.ndarray:
    """Return the real event of each row's target (a negative id; 0 where the target is outside)."""
    primary = base["primary"].to_numpy().astype(np.int64)
    target = base["target"].to_numpy().astype(np.int64)
    ids = np.zeros(len(primary), dtype=np.int64)
    for p in np.unique(primary):
        run = np.cumsum(np.diff(np.concatenate([[0], real[p].astype(np.int8)])) == 1)
        sel = primary == p
        ids[sel] = np.where(real[p][target[sel]], -1 - (p * 10_000 + run[target[sel]]), 0)
    return ids


def events_command() -> None:
    """Score real-event rows by event cluster, with and without the largest event."""
    real = score.real_event_mask()
    arms = ("P100", "AvgAnchor", "C1e-2", "B-mix", "K2-S1")
    out = []
    for month in build.MONTHS:
        base = scoring.month_base(month=month).filter(pl.col("group") == "signed_six")
        cluster = event_ids(base=base, real=real)
        event = base["cls"].to_numpy() == "event"
        keep = event & base["tgt_real"].to_numpy()
        data = {
            "month": np.full(int(keep.sum()), month),
            "event": cluster[keep].astype(str),
        }
        for arm in (*arms, "LL", "LL-ens3"):
            for basis in ("metered", "nra"):
                total = np.zeros(int(keep.sum()))
                for cell in ARM_CELLS[arm]:
                    frame = scoring.load_cell(base=base, arm=arm, month=month, cell=cell)
                    if basis == "nra":
                        frame = frame.with_columns(actual=pl.col("actual") - pl.col("nra"))
                    row = frame.select(score.pinball_points(frame=frame)).to_series().to_numpy()
                    total += row[keep]
                data[f"{arm}|{basis}"] = total / len(ARM_CELLS[arm])
        out.append(pl.DataFrame(data))
        print("events", month, flush=True)
    rows = pl.concat(out)
    rng = np.random.default_rng(0)
    result = []
    for arm in arms:
        anchor = ANCHOR.get(arm, "LL")
        for reference in sorted({anchor, "B-mix", "LL"} - {arm}):
            for basis in ("nra", "metered"):
                diff = rows[f"{arm}|{basis}"] - rows[f"{reference}|{basis}"]
                grouped = (
                    rows.with_columns(diff=diff)
                    .group_by("event")
                    .agg(pl.col("diff").sum().alias("sum"), pl.len().alias("count"))
                )
                sums = grouped["sum"].to_numpy()
                counts = grouped["count"].to_numpy().astype(float)
                largest = int(np.argmax(np.abs(sums)))
                keep_idx = np.arange(len(sums)) != largest
                for label, s, c in (
                    ("all events", sums, counts),
                    ("largest removed", sums[keep_idx], counts[keep_idx]),
                ):
                    draws = rng.integers(0, len(s), size=(5000, len(s)))
                    est = (s[draws].sum(1)) / c[draws].sum(1)
                    result.append(
                        {"arm": arm, "reference": reference, "basis": basis, "pooling": label,
                         "events": len(s), "rows": int(c.sum()), "diff": float(s.sum() / c.sum()),
                         "low": float(np.quantile(est, 0.025)),
                         "high": float(np.quantile(est, 0.975))}
                    )  # fmt: skip
    table = pl.DataFrame(result)
    table.write_csv(SCORE_DIR / "event_contrasts.csv", float_precision=4)
    with pl.Config(tbl_rows=80, tbl_width_chars=200):
        print(table)


def shift_command() -> None:
    """Write the share of rows whose shift is below 0.01 of capacity, per arm and group."""
    sites = fit.prep.primaries()["site"].to_list()
    groups = report404.group_of(sites=sites)
    rows = []
    for arm in ("P100", "AvgAnchor", "C1e-2", "C1", "Z1e-2"):
        for month in build.MONTHS:
            for cell in ARM_CELLS[arm]:
                frame = pl.read_parquet(prediction_path(arm=arm, month=month, cell=cell))
                group = groups[frame["primary"].to_numpy().astype(np.int64)]
                shift = frame["shift_capacity"].abs().to_numpy()
                logscale = frame["log_scale_change"].to_numpy()
                for g in np.unique(group):
                    sel = group == g
                    rows.append(
                        {"arm": arm, "month": month, "cell": cell, "group": str(g),
                         "share_below_001": float((shift[sel] < 0.01).mean()),
                         "mean_abs_shift": float(shift[sel].mean()),
                         "mean_log_scale_change": float(logscale[sel].mean())}
                    )  # fmt: skip
    table = pl.DataFrame(rows)
    table.write_csv(SCORE_DIR / "shift.csv", float_precision=4)
    summary = table.group_by("arm", "group").agg(
        pl.col("share_below_001").mean(), pl.col("mean_abs_shift").mean(),
        pl.col("mean_log_scale_change").mean(),
    ).sort("group", "arm")  # fmt: skip
    print(summary)


def mde_command() -> None:
    """Print experiment 454's minimum detectable effect (read before any contrast)."""
    table = report.minimum_detectable_effect()
    table.write_csv(SCORE_DIR / "mde_454.csv", float_precision=4)
    print(table)


def main() -> None:
    """Run one scoring step, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("mde", "cells", "rows", "inj", "events", "shift", "all")
    )
    args = parser.parse_args()
    build.prediction_path = prediction_path
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    steps = {
        "mde": mde_command, "cells": cells_command, "rows": rows_command, "inj": inj_command,
        "events": events_command, "shift": shift_command,
    }  # fmt: skip
    for name in steps if args.command == "all" else (args.command,):
        steps[name]()


if __name__ == "__main__":
    main()
