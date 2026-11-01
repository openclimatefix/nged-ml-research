# ruff: noqa: E501
"""Score the S1 family beside the stored references on the same half-hours (no fit).

The S1-family arms are scored as experiment 428 scores them: per seed, 1.15 widening about the
median, the mean of the per-seed scores (`e00428_score.read_arm`), world 0, and the `clean` row set
(no real event inside the three windows, target outside the padded event mask and the 96
half-hours after it). The references are read from experiment 87 (`bg2`, `manual_heuristic`,
`climatology`) and experiment 389 (`S0M_raw`, the monthly XGBoost recipe, raw and widened by 1.15),
joined on `(time_series_id, power_fcst_init_time, valid_time)` to the S1 family's clean rows, and
scored on those rows' actuals. The shared row set is the clean rows that every reference also
holds; the rows lost in the join are printed per month and group.

Tables (parquet and csv) go to `work/figure_tables/`.

```bash
source /tmp/e448_env.sh   # PYTHONPATH of every ml-experiments folder
systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00448-*/e00448_refs.py [--months ...] [--tag four]
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as s377  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00397_score as s397  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00448_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics, slices

LEVEL_NAMES: Final[tuple[str, ...]] = tuple(metrics.QUANTILE_COLUMNS)
TAILS: Final[tuple[str, ...]] = tuple(metrics.TAIL_COLUMNS)
KEYS: Final[list[str]] = list(common.KEYS)
LADDER_ARMS: Final[tuple[str, ...]] = (
    "S1", "S1PT", "S1PT-fullLR", "S1-w256", "S1-Bmix", "S1-pool3", "S1-pool5", "S1PT-pool5",
    "S1-pool5-Bmix", "S1PT-pool5-Bmix", "S1PT-noemb-pool5", "RinjPT-pool5",
)  # fmt: skip
"""The ladder arms of experiment 448 (scored where their forecast files exist)."""
FIVE_MONTH_ARMS: Final[tuple[str, ...]] = ("S1", "S1-pool3", "S1-pool5")
"""The arms fitted for 2025-12 as well."""
REFERENCES: Final[tuple[str, ...]] = (
    "BG2", "manual_heuristic", "climatology", "XGB-monthly", "XGB-monthly-w115",
)  # fmt: skip
FOUR_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
GROUPS: Final[dict[str, tuple[str, ...]]] = score.GROUPS
SUBSETS: Final[dict[str, str | None]] = {
    "days 1 to 10": None,
    "cold nights": "cold_nights",
    "sunny half-hours": "sunny_unit_shape",
    "winter nights": "winter_nights",
    "Economy 7 step": "economy_7_step",
}
CURTAILMENT_FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
TABLE_DIR: Final[Path] = run.WORK_DIR / "figure_tables"
POINTS: Final[float] = 100.0
DAY_EDGES: Final[tuple[float, ...]] = tuple(24.0 * d for d in range(11))


def sites_table() -> pl.DataFrame:
    """Return the Primary index, label, time_series_id, capacity, and group of the 16 Primaries."""
    series = adapter.series_table()
    table = (
        prep.primaries()
        .select("primary", label=pl.col("site"))
        .join(series.select("label", "time_series_id", "capacity_mw"), on="label")
    )
    group_of = {label: group for group, labels in GROUPS.items() for label in labels}
    return table.with_columns(
        group=pl.col("label").replace_strict(group_of, return_dtype=pl.String)
    )


def pinball_points(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the pinball loss per row and level, shape (rows, 11), in the units of `actual`."""
    error = actual[:, None] - q
    return np.maximum(metrics.LEVELS[None, :] * error, (metrics.LEVELS[None, :] - 1.0) * error)


def row_scores(*, actual: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return CRPS (points), the four tail pinball losses (points), and the four coverage flags."""
    pin = pinball_points(actual=actual, q=q)
    out: dict[str, np.ndarray] = {"crps": POINTS * 2.0 * pin.mean(axis=1)}
    for tail in TAILS:
        i = LEVEL_NAMES.index(tail)
        out[f"pinball_{tail}"] = POINTS * pin[:, i]
        out[f"coverage_{tail}"] = (actual <= q[:, i]).astype(np.float64)
    return out


def read_arm_rows(
    *, arm: str, month: str, seeds: tuple[int, ...], factor: float, with_quantiles: bool
) -> pl.DataFrame | None:
    """Return the clean world-0 rows of an arm: keys, actual, flags, per-seed-mean scores.

    CRPS is the mean over seeds of each seed's widened score; the tail pinball losses and coverage
    flags are scored on the seed-mean of the widened quantiles (both as experiment 428 does). The
    quantile columns, when requested, are that seed mean (fraction of capacity).
    """
    paths = [run.WORK_DIR / "predictions" / arm / f"{month}_s{s}_w0.parquet" for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    sites = sites_table().sort("primary")
    frames = [pl.read_parquet(p) for p in paths]
    first = frames[0]
    for f in frames[1:]:
        assert f["valid_us"].equals(first["valid_us"]), "seed files differ in row order"
    actual = first["actual"].to_numpy().astype(np.float64)
    crps = np.zeros(first.height)
    q_mean = np.zeros((first.height, len(LEVEL_NAMES)))
    for f in frames:
        q = score.widened(
            q=f.select(list(LEVEL_NAMES)).to_numpy().astype(np.float64), factor=factor
        )
        q_mean += q / len(frames)
        crps += row_scores(actual=actual, q=q)["crps"] / len(frames)
    # Experiment 428 scores the tails on the seed-mean quantiles and CRPS as the mean of seed scores.
    sums = row_scores(actual=actual, q=q_mean) | {"crps": crps}
    flags = first.select("tgt_real", "win_real", "tgt_post")
    keep = ~first["win_real"].to_numpy() & ~first["tgt_post"].to_numpy() & np.isfinite(actual)
    out = first.select("primary", "origin_us", "valid_us", "lead").with_columns(
        actual=pl.Series(actual),
        **{name: pl.Series(v) for name, v in sums.items()},
        **{c: flags[c] for c in flags.columns},
    )
    if with_quantiles:
        out = out.with_columns(*[pl.Series(n, q_mean[:, i]) for i, n in enumerate(LEVEL_NAMES)])
    out = out.filter(pl.Series(keep)).join(
        sites.select("primary", "time_series_id", "label", "group", "capacity_mw"), on="primary"
    )
    return out.with_columns(
        power_fcst_init_time=pl.from_epoch("origin_us", time_unit="us").dt.replace_time_zone("UTC"),
        valid_time=pl.from_epoch("valid_us", time_unit="us").dt.replace_time_zone("UTC"),
        lead_hours=pl.col("lead").cast(pl.Float64),
        time_series_id=pl.col("time_series_id").cast(pl.Int32),
    ).drop("origin_us", "valid_us", "lead")


def reference_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the references in MW: BG2, manual heuristic, climatology, monthly XGBoost (raw, widened)."""
    out = dict(s377.reference_frames(series=series))
    stored = pl.read_parquet(s397.E389_ARMS / "S0M_raw.parquet")
    xgb = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return {
        "BG2": out["bg2"],
        "manual_heuristic": out["manual_heuristic"],
        "climatology": out["climatology"],
        "XGB-monthly": xgb,
        "XGB-monthly-w115": s397.widen(frame=xgb),
    }


def score_reference(*, rows: pl.DataFrame, reference: pl.DataFrame) -> pl.DataFrame:
    """Return the shared rows scored for one reference (its MW quantiles over capacity)."""
    joined = rows.select(*KEYS, "actual", "capacity_mw").join(
        reference.select(*KEYS, *LEVEL_NAMES), on=KEYS, how="left", maintain_order="left"
    )
    q = joined.select(list(LEVEL_NAMES)).to_numpy().astype(np.float64)
    q = np.sort(q / joined["capacity_mw"].to_numpy()[:, None], axis=1)
    scores = row_scores(actual=joined["actual"].to_numpy().astype(np.float64), q=q)
    return rows.select(*KEYS).with_columns(*[pl.Series(n, v) for n, v in scores.items()])


def build_month(
    *, month: str, arms: tuple[str, ...], refs: dict[str, pl.DataFrame], seeds: tuple[int, ...]
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    """Return per-arm score tables, the base rows (clean S1 rows), and the join-loss records."""
    tables: dict[str, pl.DataFrame] = {}
    base_rows: pl.DataFrame | None = None
    for arm in arms:
        rows = read_arm_rows(
            arm=arm, month=month, seeds=seeds, factor=score.WIDENING, with_quantiles=False
        )
        if rows is None:
            print(f"  {month}: arm {arm} skipped, forecast files missing")
            continue
        tables[arm] = rows
        if base_rows is None:
            base_rows = rows
        elif rows.height != base_rows.height:
            msg = f"{month}: {arm} has {rows.height} clean rows, the first arm {base_rows.height}"
            raise ValueError(msg)
    assert base_rows is not None
    lost = []
    present = pl.Series(np.ones(base_rows.height, dtype=bool))
    for name, ref in refs.items():
        joined = base_rows.select(*KEYS, "label", "group").join(
            ref.select(*KEYS, "q50").with_columns(pl.col("q50").is_finite().alias("_ok")),
            on=KEYS,
            how="left",
            maintain_order="left",
        )
        ok = joined["_ok"].fill_null(value=False)
        present &= ok
        for (group,), part in joined.with_columns(ok=ok).group_by("group"):
            lost.append(
                {"month": month, "reference": name, "group": group, "clean_rows": part.height,
                 "lost": int((~part["ok"]).sum())}
            )  # fmt: skip
    keep_keys = base_rows.filter(present).select(*KEYS)
    refs_scored = {
        name: score_reference(rows=base_rows, reference=ref).join(keep_keys, on=KEYS, how="semi")
        for name, ref in refs.items()
    }
    shared_base = base_rows.join(keep_keys, on=KEYS, how="semi")
    for arm in list(tables):
        tables[arm] = tables[arm].join(keep_keys, on=KEYS, how="semi")
    tables.update(
        {
            n: t.join(
                shared_base.select(*KEYS, "label", "group", "capacity_mw", "lead_hours", "actual",
                                   "tgt_real"),
                on=KEYS,
            )
            for n, t in refs_scored.items()
        }
    )  # fmt: skip
    return tables, shared_base, pl.DataFrame(lost)


def slice_flags(*, rows: pl.DataFrame, ctx: dict) -> pl.DataFrame:
    """Return the slice table of the shared rows, from experiment 425's scoring context."""
    keys = rows.select(*KEYS).unique(subset=KEYS)
    joined = keys.join(ctx["context"], on=["time_series_id", "valid_time"], how="left")
    missing = joined["temperature_c"].null_count()
    print(f"  slice context: {keys.height} shared half-hours, {missing} without temperature")
    return slices.build_slice_table(rows=joined)


def aggregate(*, table: pl.DataFrame, arm: str, flags: pl.DataFrame) -> list[dict]:
    """Return metric records per subset, per group and per Primary, pooled and equal-weighted."""
    rows = table.join(flags.select(*KEYS, *[c for c in SUBSETS.values() if c]), on=KEYS)
    rows = rows.with_columns(day=(pl.col("lead_hours") / 24.0).ceil().cast(pl.Int32).clip(1, 10))
    value_cols = ["crps", *[f"pinball_{t}" for t in TAILS], *[f"coverage_{t}" for t in TAILS]]
    records: list[dict] = []
    for subset, flag in SUBSETS.items():
        part = rows if flag is None else rows.filter(pl.col(flag))
        per_primary = part.group_by("label", "group").agg(
            pl.len().alias("n"), *[pl.col(c).mean() for c in value_cols]
        )
        records.extend(
            {
                "arm": arm,
                "subset": subset,
                "scope": "primary",
                "name": r["label"],
                "weighting": "pooled",
            }
            | {k: r[k] for k in ["n", *value_cols]}
            for r in per_primary.iter_rows(named=True)
        )
        for group in [*GROUPS, "all 16"]:
            members = (
                per_primary if group == "all 16" else per_primary.filter(pl.col("group") == group)
            )
            pooled = members.select(
                pl.col("n").sum(),
                *[
                    ((pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()).alias(c)
                    for c in value_cols
                ],
            ).row(0, named=True)
            equal = members.select(*[pl.col(c).mean() for c in value_cols]).row(0, named=True)
            for weighting, values in (("pooled", pooled), ("equal_primary", equal)):
                records.append(
                    {"arm": arm, "subset": subset, "scope": "group", "name": group,
                     "weighting": weighting, "n": pooled["n"],
                     **{c: values[c] for c in value_cols}}
                )  # fmt: skip
    by_day = rows.group_by("group", "day").agg(pl.len().alias("n"), pl.col("crps").mean())
    records.extend(
        {
            "arm": arm,
            "subset": f"day {r['day']}",
            "scope": "group",
            "name": r["group"],
            "weighting": "pooled",
            "n": r["n"],
            "crps": r["crps"],
        }
        for r in by_day.iter_rows(named=True)
    )
    all_day = rows.group_by("day").agg(pl.len().alias("n"), pl.col("crps").mean())
    records.extend(
        {
            "arm": arm,
            "subset": f"day {r['day']}",
            "scope": "group",
            "name": "all 16",
            "weighting": "pooled",
            "n": r["n"],
            "crps": r["crps"],
        }
        for r in all_day.iter_rows(named=True)
    )
    return records


def curtailment(
    *, months: tuple[str, ...], refs: dict[str, pl.DataFrame], keys: pl.DataFrame, ctx: dict,
    series: pl.DataFrame, arms: tuple[str, ...],
) -> pl.DataFrame:  # fmt: skip
    """Return the avoidable curtailment cost (pounds) per arm and series, signed-MW six only.

    The arms' forecast is the seed mean of the widened quantiles (quantile space), in MW; the
    references are as stored. Every forecast is restricted to the shared half-hours.
    """
    sites = sites_table()
    forecasts: dict[str, pl.DataFrame] = {}
    for arm in arms:
        parts = []
        for month in months:
            rows = read_arm_rows(
                arm=arm, month=month, seeds=SEEDS, factor=score.WIDENING, with_quantiles=True
            )
            assert rows is not None
            q = np.sort(rows.select(list(LEVEL_NAMES)).to_numpy(), axis=1)
            cap = rows["capacity_mw"].to_numpy()[:, None]
            parts.append(
                rows.select(*KEYS).with_columns(
                    *[pl.Series(n, q[:, i] * cap[:, 0]) for i, n in enumerate(LEVEL_NAMES)]
                )
            )
        forecasts[arm] = pl.concat(parts)
    for name, ref in refs.items():
        forecasts[name] = ref.select(*KEYS, *LEVEL_NAMES)
    actuals = ctx["actuals"]
    flows = actuals.rename({"actual": "flow_mw"})
    prepared = {
        n: f.join(keys, on=KEYS, how="semi").join(actuals, on=["time_series_id", "valid_time"])
        for n, f in forecasts.items()
    }
    print("  curtailment rows per forecast:", {n: f.height for n, f in prepared.items()})
    series_signed = series.join(sites.select("label", "group"), on="label").filter(
        pl.col("group") == "signed-MW six"
    )
    out = []
    for fraction in CURTAILMENT_FRACTIONS:
        cost = metrics.avoidable_curtailment_cost(
            forecasts=prepared, series=series_signed.select("time_series_id", "label", "capacity_mw"),
            flows=flows, scored_start=s377.SCORED_FIRST, fraction=fraction,
        )  # fmt: skip
        out.append(cost.with_columns(fraction=pl.lit(fraction)))
    return pl.concat(out)


def main() -> None:
    """Score the arms and references and write every table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(FOUR_MONTHS))
    parser.add_argument("--tag", default="four_months")
    parser.add_argument("--arms", nargs="+", default=list(LADDER_ARMS))
    parser.add_argument("--no-curtailment", action="store_true")
    args = parser.parse_args()
    months, arms = tuple(args.months), tuple(args.arms)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    refs = reference_frames(series=series)
    per_arm: dict[str, list[pl.DataFrame]] = {}
    base_parts, lost_parts = [], []
    for month in months:
        print(f"month {month}")
        tables, shared_base, lost = build_month(month=month, arms=arms, refs=refs, seeds=SEEDS)
        for name, table in tables.items():
            per_arm.setdefault(name, []).append(table)
        base_parts.append(shared_base.select(*KEYS))
        lost_parts.append(lost)
    lost = pl.concat(lost_parts)
    lost.write_csv(TABLE_DIR / f"rows_lost_{args.tag}.csv")
    shared = pl.concat(base_parts)
    print(f"\nshared half-hours (clean, all references present): {shared.height}")
    print(
        "rows lost in the join per month, reference, and group (clean S1 rows with no reference row):"
    )
    with pl.Config(tbl_rows=200):
        print(lost.filter(pl.col("lost") > 0).sort("month", "reference", "group"))
        total = lost.group_by("reference").agg(pl.col("clean_rows").sum(), pl.col("lost").sum())
        print(total)
    flags = slice_flags(rows=shared, ctx=ctx)
    for s in ("cold_nights", "sunny_unit_shape", "winter_nights", "economy_7_step"):
        print(f"  slice {s}: {int(flags[s].sum())} half-hours")
    records: list[dict] = []
    for name, parts in per_arm.items():
        table = pl.concat(parts)
        # An arm is only comparable with the references over the months it holds.
        records += aggregate(table=table, arm=name, flags=flags)
    metrics_table = pl.DataFrame(records, infer_schema_length=None)
    metrics_table.write_parquet(TABLE_DIR / f"metrics_{args.tag}.parquet")
    metrics_table.write_csv(TABLE_DIR / f"metrics_{args.tag}.csv")
    summary(metrics=metrics_table)
    if not args.no_curtailment:
        cost = curtailment(
            months=months, refs=refs, keys=shared, ctx=ctx, series=series,
            arms=tuple(a for a in ("S1", "S1-pool5") if a in per_arm),
        )  # fmt: skip
        cost.write_parquet(TABLE_DIR / f"curtailment_{args.tag}.parquet")
        cost.write_csv(TABLE_DIR / f"curtailment_{args.tag}.csv")
        print("\n== avoidable curtailment cost (pounds), signed-MW six, lead 12 to 36 h ==")
        with pl.Config(tbl_rows=80, tbl_cols=20):
            totals = cost.group_by("fraction", "arm").agg(
                pl.col("avoidable_cost_gbp").sum(), pl.col("status").first()
            )
            print(totals.sort("fraction", "avoidable_cost_gbp"))
            print(cost.filter(pl.col("fraction") == 0.5).pivot(on="arm", index="label",
                                                              values="avoidable_cost_gbp"))  # fmt: skip


def summary(*, metrics: pl.DataFrame) -> None:
    """Print every number a write-up would quote."""
    order = [*LADDER_ARMS, *REFERENCES]
    shown = [a for a in order if a in set(metrics["arm"].to_list())]
    group_names = [*GROUPS, "all 16"]
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_float="mixed"):
        for subset in SUBSETS:
            for weighting in ("pooled", "equal_primary"):
                part = metrics.filter(
                    pl.col("scope") == "group", pl.col("subset") == subset,
                    pl.col("weighting") == weighting,
                )  # fmt: skip
                wide = part.pivot(on="name", index="arm", values="crps").with_columns(
                    n=part.filter(pl.col("name") == "all 16").sort("arm")["n"][0]
                    if part.height
                    else None
                )
                wide = (
                    wide.with_columns(
                        i=pl.col("arm").replace_strict(
                            {a: k for k, a in enumerate(shown)}, default=99
                        )
                    )
                    .sort("i")
                    .drop("i")
                )
                print(f"\n== CRPS (points), {subset}, {weighting} ==")
                print(
                    wide.select("arm", *[g for g in group_names if g in wide.columns]).with_columns(
                        pl.exclude("arm").round(3)
                    )
                )
        for subset in ("days 1 to 10", "cold nights", "sunny half-hours"):
            tail = metrics.filter(
                pl.col("scope") == "group", pl.col("subset") == subset,
                pl.col("weighting") == "equal_primary", pl.col("name").is_in(group_names),
            ).select(
                "arm", "name",
                *[f"coverage_{t}" for t in TAILS], *[f"pinball_{t}" for t in TAILS],
            )  # fmt: skip
            print(
                f"\n== tails, {subset}, equal weight per Primary: coverage then pinball (points) =="
            )
            print(tail.with_columns(pl.exclude("arm", "name").round(3)).sort("name", "arm"))
        per_primary = metrics.filter(
            pl.col("scope") == "primary", pl.col("subset") == "days 1 to 10"
        ).pivot(on="arm", index="name", values="crps")
        print("\n== CRPS (points) per Primary, days 1 to 10 ==")
        print(
            per_primary.select("name", *[a for a in shown if a in per_primary.columns])
            .sort(pl.col("name").str.slice(1).cast(pl.Int32))
            .with_columns(pl.exclude("name").round(3))
        )


if __name__ == "__main__":
    main()
