# ruff: noqa: E501, PTH123, SIM115
"""Score experiment 471's heuristic variants against the faithful heuristic (no fit).

Rows come from experiment 452's `PT1` clean world-0 half-hours through experiment 448's row builder
(the row set of experiment 457, `fold_a_daily_gap_v1`), which ends at 168 hours, so 168 to 240 hours is
not scored. Two passes: `main` keeps only half-hours that every reference holds, BG2 and the stored
arm A included; `long` drops BG2 and arm A and scores the same half-hours. Contrasts are paired
month-by-origin-day block bootstraps (experiment 457's code). The Bonferroni count `TESTS` covers only
the gating cells (CRPS, all rows, 0-240 h label, per variant and group). Every other cell is secondary,
and its interval is uncorrected for the wider family. The minimum detectable effect of a contrast is
the half-width of its Bonferroni interval.

```bash
source /tmp/e471_env.sh
uv run --no-sync python ml-experiments/00471-*/e00471_score.py
```
"""

import json
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00452_refs as refs452  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import e00457_refs as refs457  # ty: ignore[unresolved-import]
import e00457_score as s457  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

OUT: Final = common.WORK_DIR / "scores"
PRED: Final = common.WORK_DIR / "predictions"
PRIMARY: Final[str] = "E56"
VARIANTS: Final[tuple[str, ...]] = (
    "E28", "E56", "E56m13", "E56m99", "MeanLS56", "Q56", "Tau95_56",
    "Lvl7", "Lvl14", "Lvl28",
)  # fmt: skip
GROUPS_GATING: Final[tuple[str, ...]] = ("MVA-only eight", "signed-MW six")
ROW_SETS: Final[dict[str, tuple[float, float]]] = {
    "0-240h": (0.0, 240.0), "12-36h": (12.0, 36.0), "120-240h": (120.0, 240.0),
    "0-24h": (0.0, 24.0), "24-72h": (24.0, 72.0), "72-120h": (72.0, 120.0),
}  # fmt: skip
SLICES: Final[tuple[str, ...]] = ("all", "cold nights", "Christmas week")
METRICS: Final[tuple[str, ...]] = ("crps", *[f"pinball_{t}" for t in s457.TAILS])
TESTS: Final[int] = len(VARIANTS) * len(GROUPS_GATING)
s457.TESTS = TESTS


def variant_frame(*, name: str) -> pl.DataFrame:
    """Return a stored variant's quantiles in MW, keyed for the scorer."""
    return pl.read_parquet(PRED / f"{name}.parquet").select(*common.KEYS, *common.LEVELS)


def faithful_frame(*, members: pl.DataFrame) -> pl.DataFrame:
    """Return the faithful heuristic's quantiles from the member table (linear interpolation, MW)."""
    q = np.nanquantile(
        members.select(common.MEMBER_COLUMNS).to_numpy(), common.LEVEL_VALUES, axis=1
    ).T
    return members.select(*common.KEYS).with_columns(
        *[pl.Series(k, q[:, i]) for i, k in enumerate(common.LEVELS)]
    )


def select(
    *, table: pl.DataFrame, group: str, slice_name: str, lead: tuple[float, float]
) -> pl.DataFrame:
    """Return a table's rows for a group, slice, and lead range (hours, upper end inclusive)."""
    part = s457.select_rows(
        table=table,
        group=group,
        row_set="cold nights"
        if slice_name == "cold nights"
        else ("Christmas week" if slice_name == "Christmas week" else "days 1 to 10"),
    )
    return part.filter(
        pl.col("lead_hours").is_between(lead[0], lead[1], closed="right" if lead[0] > 0 else "both")
    )


def contrast(
    *, tables: dict[str, pl.DataFrame], a: str, b: str, group: str, slice_name: str, lead: tuple[float, float],
    metric: str, rng: np.random.Generator,
) -> dict:  # fmt: skip
    """Return `a - b` for one metric, group, slice, and lead range (negative favours `a`)."""
    left = select(table=tables[a], group=group, slice_name=slice_name, lead=lead)
    right = tables[b].select(*base.KEYS, pl.col(metric).alias("m_b"))
    joined = left.select(*base.KEYS, "month", pl.col(metric)).join(right, on=base.KEYS)
    if joined.height == 0:
        return {}
    joined = joined.with_columns(
        block=pl.col("month") + pl.col("power_fcst_init_time").dt.date().cast(pl.String)
    )
    blocks = joined.group_by("block", "month").agg(
        pl.len().alias("n"), (pl.col(metric) - pl.col("m_b")).sum().alias("diff")
    )
    n, diff = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
    draws = rng.integers(0, len(n), (s457.RESAMPLES, len(n)))
    boot = diff[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * 0.05 / TESTS / 2.0
    low, high = float(np.percentile(boot, alpha)), float(np.percentile(boot, 100.0 - alpha))
    return {
        "contrast": f"{a} - {b}", "group": group, "slice": slice_name, "lead": f"{lead[0]:g}-{lead[1]:g}h",
        "metric": metric, "gating": bool(slice_name == "all" and lead == (0.0, 240.0) and metric == "crps"),
        "n": int(n.sum()), "difference": float(diff.sum() / n.sum()),
        "low_bonferroni": low, "high_bonferroni": high, "mde": (high - low) / 2.0,
        "upper_one_sided_95": float(np.percentile(boot, 95.0)),
    }  # fmt: skip


def absolute(*, tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return mean CRPS, tail pinball loss, and tail coverage per arm, group, slice, and lead range."""
    cols = ["crps", *[f"pinball_{t}" for t in s457.TAILS], *[f"coverage_{t}" for t in s457.TAILS]]
    records = []
    for name, table in tables.items():
        for group in (*GROUPS_GATING, "D15 and D18", "all 16"):
            for slice_name in SLICES:
                for lead_name, lead in ROW_SETS.items():
                    part = select(table=table, group=group, slice_name=slice_name, lead=lead)
                    if part.height:
                        records.append({"arm": name, "group": group, "slice": slice_name, "lead": lead_name, "n": part.height,
                                        **part.select([pl.col(c).mean() for c in cols]).row(0, named=True)})  # fmt: skip
    return pl.DataFrame(records)


def add_lead(*, tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return the tables restricted to 0 to 240 hours (every table carries `lead_hours`)."""
    return {n: t.filter(pl.col("lead_hours") <= 240.0) for n, t in tables.items()}


def run_pass(
    *,
    name: str,
    refs: dict[str, pl.DataFrame],
    ctx: dict,
    variants: tuple[str, ...],
    months: tuple[str, ...],
) -> None:
    """Score one pass and write its absolute scores and contrasts."""
    base.run = run  # ty: ignore[invalid-assignment]
    per: dict[str, list[pl.DataFrame]] = {}
    lost_all = []
    for month in months:
        tables, shared_base, lost = base.build_month(
            month=month, arms=("PT1",), refs=refs, seeds=run.SEEDS5
        )
        lost_all.append(lost)
        flags = base.slice_flags(rows=shared_base, ctx=ctx).select(*base.KEYS, "cold_nights")
        for arm, table in tables.items():
            per.setdefault(arm, []).append(
                table.join(flags, on=base.KEYS).with_columns(month=pl.lit(month))
            )
    tables = add_lead(tables={n: pl.concat(p) for n, p in per.items()})
    tables.pop("PT1", None)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(lost_all).write_csv(OUT / f"{name}_lost_rows.csv")
    abs_table = absolute(tables=tables)
    abs_table.write_csv(OUT / f"{name}_absolute.csv")
    rng = np.random.default_rng(0)
    records = []
    for v in variants:
        for group in GROUPS_GATING:
            for slice_name in SLICES:
                for lead in ROW_SETS.values():
                    for metric in METRICS:
                        rec = contrast(
                            tables=tables,
                            a=v,
                            b="manual_heuristic",
                            group=group,
                            slice_name=slice_name,
                            lead=lead,
                            metric=metric,
                            rng=rng,
                        )
                        if rec:
                            records.append(rec)
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(OUT / f"{name}_contrasts.csv")
    with pl.Config(tbl_rows=120, tbl_cols=16, tbl_width_chars=220, fmt_str_lengths=40):
        print(
            f"== {name}: Bonferroni tests {TESTS}; shared rows per arm {next(iter(tables.values())).height}"
        )
        print(
            abs_table.filter(
                (pl.col("slice") == "all")
                & (pl.col("lead") == "0-240h")
                & pl.col("group").is_in(GROUPS_GATING)
            ).select(
                "arm",
                "group",
                "n",
                pl.col("crps", "pinball_q95", "pinball_q99", "coverage_q95", "coverage_q99").round(
                    3
                ),
            )
        )
        print(
            contrasts.filter(
                (pl.col("metric") == "crps")
                & (pl.col("slice") == "all")
                & (pl.col("lead") == "0-240h")
            ).select(
                "contrast",
                "group",
                "n",
                pl.col("difference", "low_bonferroni", "high_bonferroni", "mde").round(4),
            )
        )
    json.dump(
        {"tests": TESTS, "shared_rows": next(iter(tables.values())).height},
        open(OUT / f"{name}_meta.json", "w"),
    )
    for arm, t in tables.items():
        t.write_parquet(OUT / f"{name}_rows_{arm}.parquet")


def main() -> None:
    """Run both passes."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    members = common.load_members()
    variants = {v: variant_frame(name=v) for v in VARIANTS}
    faithful = faithful_frame(members=members)
    stored = {
        k: v
        for k, v in refs452.reference_frames(series=series).items()
        if k in ("BG2", "manual_heuristic", "climatology")
    }
    stored["A-avg5"] = refs457.arm_frame(
        series=series, folder="A", seeds=refs457.SEEDS, root="425", factor=1.15
    )
    stored["XGB-monthly-w115"] = base.reference_frames(series=series)["XGB-monthly-w115"]
    # Check: the faithful quantiles rebuilt from members equal experiment 87's reference on the stored rows.
    check = faithful.join(stored["manual_heuristic"], on=common.KEYS, suffix="_87")
    gap = max((check[k] - check[f"{k}_87"]).abs().max() for k in common.LEVELS)  # ty: ignore[invalid-argument-type]
    print(
        f"faithful from members vs experiment 87 reference: {check.height:,} rows, largest gap {gap:.2e} MW"
    )
    assert gap < 1e-4, gap
    main_refs = {**stored, **variants}
    run_pass(name="main", refs=main_refs, ctx=ctx, variants=VARIANTS, months=common.MONTHS)
    long_refs = {
        "manual_heuristic": stored["manual_heuristic"],
        "climatology": stored["climatology"],
        **variants,
    }
    run_pass(name="long", refs=long_refs, ctx=ctx, variants=VARIANTS, months=common.MONTHS)


if __name__ == "__main__":
    main()
