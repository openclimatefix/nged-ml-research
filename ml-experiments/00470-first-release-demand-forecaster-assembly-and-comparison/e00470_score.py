"""Score experiment 470's candidates on common rows.

The outputs are the minimum detectable effect, absolute scores, paired contrasts with month-block
bootstrap intervals, and the tail and widening tables.

Row sets (all within experiment 454's rows, 0 to 240 hours, after the cleaned-data drops):

- `gam14`: rows that every candidate holds, including A-gam-in (the eight MVA-only Primaries and the
  signed-MW six; the gating set).
- `main16`: rows that every candidate except A-gam-in holds (adds D15 and D18, report-only).
- `ref14`: `gam14` rows that BG2, `manual_heuristic`, climatology, and the XGBoost forecast also
  hold (BG2 stops at 168 hours, so days 8 to 10 drop out).

Slices: `eight`, `six_clean` (rows with no labelled event), `six_nra` (real-event rows scored
against normal-running-arrangement actuals), `cold_eight`, `cold_six`, `band_eight` and `band_six`
(12 to 36 hours), `xmas_eight` and `xmas_six` (22 to 31 December, report-only), and `report`
(D15 and D18).
All scores are selection-biased: the five folds that choose the candidate also score it.

```bash
source .e470env.sh
uv run --no-sync python ml-experiments/00470-*/e00470_score.py rows
uv run --no-sync python ml-experiments/00470-*/e00470_score.py stats
```
"""

import argparse
import json
import math
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import TAIL_LEVELS, widen  # ty: ignore[unresolved-import]

MONTHS: Final[tuple[str, ...]] = build470.MONTHS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
KEYS: Final[list[str]] = build470.KEYS
SCORE_DIR = heads470.WORK_DIR / "scores"
CANDIDATES: Final[tuple[str, ...]] = ("A-pool", "K2-S1", "K2-425A", "K2-425A+S1", "A-gam-in")
CONTROLS: Final[tuple[str, ...]] = ("A-ens2", "B-mix", "S1-ens3")
DESCRIPTIVE: Final[tuple[str, ...]] = ("AvgAnchor",)
"""Experiment 472's AvgAnchor: unreplicated, scored beside the candidates and outside the rule."""
SINGLES: Final[tuple[str, ...]] = tuple(f"A-s{s}" for s in build470.POOL_SEEDS)
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology", "XGB-monthly")
NOT_WIDENED: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology")
MAIN_ARMS: Final[tuple[str, ...]] = tuple(a for a in (*CANDIDATES, *CONTROLS) if a != "A-gam-in")
CELLS: Final[dict[str, int]] = {
    "B-mix": 3,
    "K2-S1": 3,
    "K2-425A": 3,
    "K2-425A+S1": 3,
    "A-ens2": 3,
    "AvgAnchor": 3,
}
WIDEN_SWEEP: Final[tuple[float, ...]] = (1.05, 1.10, 1.15)
HEADLINE: Final[float] = 1.15
TESTS: Final[int] = 54
RESAMPLES: Final[int] = 20_000
CHRISTMAS: Final[tuple[int, int]] = (22, 31)
POINTS: Final[float] = 100.0
SLICES: Final[dict[str, tuple[str, str]]] = {
    "eight": ("mva_eight", "all"),
    "six_clean": ("signed_six", "clean"),
    "six_nra": ("signed_six", "nra"),
    "cold_eight": ("mva_eight", "cold"),
    "cold_six": ("signed_six", "cold"),
    "band_eight": ("mva_eight", "band"),
    "band_six": ("signed_six", "band"),
    "xmas_eight": ("mva_eight", "xmas"),
    "xmas_six": ("signed_six", "xmas"),
    "report": ("report_only", "all"),
}
GATING_CRPS: Final[tuple[str, ...]] = (
    "eight", "six_clean", "six_nra", "cold_eight", "cold_six", "band_eight", "band_six",
)  # fmt: skip
GATING_Q99: Final[tuple[str, ...]] = ("eight", "six_clean")
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("K2-S1", "A-pool"), ("K2-425A", "A-pool"), ("K2-425A+S1", "A-pool"), ("A-gam-in", "A-pool"),
    ("K2-425A", "K2-S1"), ("K2-425A+S1", "K2-S1"),
)  # fmt: skip
assert len(CONTRASTS) * len(GATING_CRPS) + len(CONTRASTS) * len(GATING_Q99) == TESTS


def arm_cells(*, arm: str) -> int:
    """Return the number of stored cells of an arm."""
    return CELLS.get(arm, 1)


def read_cell(*, arm: str, month: str, cell: int) -> pl.DataFrame:
    """Return an arm cell's quantiles (fractions)."""
    return pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=cell))


def month_context(*, month: str, sites: pl.DataFrame, ctx: dict) -> pl.DataFrame:
    """Return a month's base rows with group, class, lead band, NRA correction, and slice flags."""
    base = score454.month_base(month=month)
    grid = common.grid_times()
    base = base.with_columns(
        power_fcst_init_time=grid.gather(base["origin"].to_numpy().astype(np.int64)),
        valid_time=grid.gather(base["target"].to_numpy().astype(np.int64)),
    ).join(sites.select("primary", "time_series_id"), on="primary")
    flags = refs448.slice_flags(
        rows=base.select(*refs448.KEYS, "actual", "primary").rename({"actual": "actual_unused"}),
        ctx=ctx,
    ).select(*refs448.KEYS, "cold_nights")
    base = base.join(flags, on=refs448.KEYS, how="left", maintain_order="left")
    base = base.with_columns(
        cold=pl.col("cold_nights").fill_null(value=False),
        xmas=(pl.col("valid_time").dt.month() == 12)
        & (pl.col("valid_time").dt.day().is_between(*CHRISTMAS))
        & (pl.lit(month == "2025-12")),
        band=pl.col("lead_band") == "12-36h",
        day=(pl.col("origin") // 48).cast(pl.Int32),
    )
    return base.drop("cold_nights")


def widen_arm(*, arm: str, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return the frame widened about the median (not for the references that carry their own)."""
    if factor == 1.0 or arm in NOT_WIDENED:
        return frame
    return widen(frame=frame, factor=factor)


def row_metrics(*, frame: pl.DataFrame, actual: str = "actual") -> dict[str, np.ndarray]:
    """Return per-row CRPS (points of capacity) and tail pinball (points) and coverage (0 or 1)."""
    q = frame.select(list(LEVELS)).to_numpy().astype(np.float64)
    q = np.sort(q, axis=1)
    y = frame[actual].to_numpy().astype(np.float64)[:, None]
    lev = np.array(models.LEVELS)[None, :]
    pin = np.maximum(lev * (y - q), (lev - 1.0) * (y - q))
    out = {"crps": 2.0 * POINTS * (pin * common.ATOM_MASSES[None, :]).sum(axis=1)}
    for name in TAIL_LEVELS:
        i = LEVELS.index(name)
        out[f"pin_{name}"] = POINTS * pin[:, i]
        out[f"cov_{name}"] = (y[:, 0] <= q[:, i]).astype(np.float64)
    return out


def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
    """Return the three row-set masks from the arms' q50 columns (NaN where an arm has no row).

    A-gam-in holds a forecast for D15 and D18 with its GAM input set to zero, so those two Primaries
    are left out of `gam14` and `ref14`.
    """
    have = lambda arms: pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in arms])  # noqa: E731
    frame = wide.select(
        main16=have(MAIN_ARMS),
        gam14=have((*MAIN_ARMS, "A-gam-in")) & (pl.col("group") != "report_only"),
        ref14=have((*MAIN_ARMS, "A-gam-in", *REFERENCES)) & (pl.col("group") != "report_only"),
    )
    return {c: frame[c] for c in frame.columns}


def month_rows(*, month: str, base: pl.DataFrame) -> pl.DataFrame:
    """Return a month's row-level metrics of every arm (headline widening, cells averaged)."""
    arms = (*CANDIDATES, *CONTROLS, *DESCRIPTIVE, *SINGLES, *REFERENCES)
    wide = base.select(*KEYS)
    cols: dict[str, np.ndarray] = {}
    for arm in arms:
        cells = []
        for cell in range(arm_cells(arm=arm)):
            frame = base.select(*KEYS, "actual", "nra", "tgt_real").join(
                read_cell(arm=arm, month=month, cell=cell),
                on=KEYS,
                how="left",
                maintain_order="left",
            )
            frame = widen_arm(arm=arm, frame=frame, factor=HEADLINE)
            ok = frame["q50"].is_not_null().to_numpy() & np.isfinite(frame["q50"].to_numpy())
            frame = frame.with_columns(*[pl.col(c).fill_null(0.0) for c in LEVELS])
            metered = row_metrics(frame=frame)
            nra = row_metrics(
                frame=frame.with_columns(actual_nra=pl.col("actual") - pl.col("nra")),
                actual="actual_nra",
            )
            masked = [{k: np.where(ok, v, np.nan) for k, v in m.items()} for m in (metered, nra)]
            cells.append((masked[0], masked[1], ok, frame["q50"].to_numpy()))
        for metric in cells[0][0]:
            cols[f"{arm}__{metric}"] = np.mean([c[0][metric] for c in cells], axis=0)
            cols[f"{arm}__nra_{metric}"] = np.mean([c[1][metric] for c in cells], axis=0)
        every = np.logical_and.reduce([c[2] for c in cells])
        cols[f"{arm}__q50"] = np.where(every, np.mean([c[3] for c in cells], axis=0), np.nan)
        # Per-cell CRPS gives the seed spread of the constructs.
        if arm_cells(arm=arm) > 1:
            for i, c in enumerate(cells):
                cols[f"{arm}__crps_cell{i}"] = c[0]["crps"]
    out = wide.hstack(pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()}))
    masks_ = masks(wide=out.with_columns(base["group"]))
    return (
        out.hstack(
            base.select(
                "group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "xmas", "band", "day"
            )
        )
        .hstack(pl.DataFrame(dict(masks_.items())))
        .with_columns(month=pl.lit(month))
    )


def rows_command() -> None:
    """Write the row-level table (one parquet per month) and print the row-set sizes."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        base = month_context(month=month, sites=sites, ctx=ctx)
        table = month_rows(month=month, base=base)
        table.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        sizes = {c: int(table[c].sum()) for c in ("main16", "gam14", "ref14")}
        print(f"{month}: base rows {base.height}, sets {sizes}", flush=True)


def load_rows() -> pl.DataFrame:
    """Return the row-level table of all months."""
    return pl.concat([pl.read_parquet(SCORE_DIR / f"rows_{m}.parquet") for m in MONTHS])


def slice_filter(
    *, table: pl.DataFrame, name: str, basis: str = "metered"
) -> tuple[pl.DataFrame, str]:
    """Return the rows of a slice and the metric prefix (`nra_` for the real-event basis)."""
    group, kind = SLICES[name]
    part = table.filter(pl.col("group") == group)
    prefix = ""
    if kind == "clean":
        part = part.filter(pl.col("cls") == "clean")
    elif kind == "nra":
        part = part.filter(pl.col("tgt_real"))
        prefix = "nra_"
    elif kind == "cold":
        part = part.filter(pl.col("cold"))
    elif kind == "band":
        part = part.filter(pl.col("band"))
    elif kind == "xmas":
        part = part.filter(pl.col("xmas"))
    return part, prefix


def rowset_for(*, a: str, b: str) -> str:
    """Return the smallest row set that holds both arms."""
    needs_ref = any(x in REFERENCES for x in (a, b))
    needs_gam = "A-gam-in" in (a, b)
    return "ref14" if needs_ref else "gam14" if needs_gam else "main16"


def block_bootstrap(*, diff: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Return the difference and its intervals from month-by-origin-day blocks (diff in `d`)."""
    blocks = diff.group_by("month", "day").agg(pl.len().alias("n"), pl.col("d").sum().alias("s"))
    n, s = blocks["n"].to_numpy().astype(float), blocks["s"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = s[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * 0.05 / TESTS / 2.0
    by_month = diff.group_by("month").agg(pl.col("d").mean())
    return {
        "n": int(n.sum()), "difference": float(s.sum() / n.sum()),
        "low": float(np.percentile(boot, 2.5)), "high": float(np.percentile(boot, 97.5)),
        "low_bonferroni": float(np.percentile(boot, alpha)),
        "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
        "by_month": json.dumps(
            {r["month"]: round(r["d"], 4) for r in by_month.iter_rows(named=True)}
        ),
    }  # fmt: skip


def mde(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect: arm A's five seeds, and the cells."""
    records = []
    sets = {
        "eight": "mva_eight", "six (all rows)": "signed_six", "D15 and D18": "report_only",
    }  # fmt: skip
    for label, group in sets.items():
        part = table.filter((pl.col("group") == group) & pl.col("main16"))
        for name, columns in (
            ("A single seeds (5)", [f"A-s{s}__crps" for s in build470.POOL_SEEDS]),
            *[(f"{arm} cells (3)", [f"{arm}__crps_cell{i}" for i in range(3)]) for arm in CELLS],
        ):
            values = np.array(
                [
                    [part.filter(pl.col("month") == m)[c].mean() for c in columns]
                    for m in MONTHS
                    if part.filter(pl.col("month") == m).height
                ]
            )
            folds, k = values.shape
            resid = values - values.mean(axis=1, keepdims=True)
            sd = math.sqrt((resid**2).sum() / (folds * (k - 1)))
            records.append({"group": label, "source": name, "seed_sd": sd,
                            "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(k)})  # fmt: skip
    return pl.DataFrame(records)


def stats_command() -> None:
    """Write the minimum detectable effect, absolute scores, contrasts, and tails."""
    table = load_rows()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    minimum = mde(table=table)
    print("minimum detectable effect (printed before any contrast):")
    print(minimum)
    minimum.write_csv(SCORE_DIR / "mde.csv")
    metrics = ["crps", *[f"pin_{t}" for t in TAIL_LEVELS], *[f"cov_{t}" for t in TAIL_LEVELS]]
    absolute = []
    for sname in SLICES:
        for rowset in ("gam14", "main16", "ref14"):
            for arm in (*CANDIDATES, *CONTROLS, *DESCRIPTIVE, *REFERENCES):
                if arm in REFERENCES and rowset != "ref14":
                    continue
                part, prefix = slice_filter(table=table, name=sname)
                part = part.filter(pl.col(rowset))
                if part.height == 0 or part[f"{arm}__{prefix}crps"].null_count() == part.height:
                    continue
                absolute.append(
                    {"slice": sname, "rowset": rowset, "arm": arm, "n": part.height}
                    | {m: part[f"{arm}__{prefix}{m}"].mean() for m in metrics}
                )
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")
    rng = np.random.default_rng(0)
    records = []
    for a, b in CONTRASTS:
        rowset = rowset_for(a=a, b=b)
        for sname in (*GATING_CRPS, "xmas_eight", "xmas_six", "report"):
            for metric in (
                "crps",
                *(
                    [f"pin_{t}" for t in TAIL_LEVELS]
                    if sname in ("eight", "six_clean", "report")
                    else []
                ),
            ):
                if sname == "report" and "A-gam-in" in (a, b):
                    continue
                part, prefix = slice_filter(table=table, name=sname)
                part = part.filter(pl.col(rowset))
                diff = (
                    part.select(
                        "month",
                        "day",
                        d=pl.col(f"{a}__{prefix}{metric}") - pl.col(f"{b}__{prefix}{metric}"),
                    )
                    .drop_nulls()
                    .drop_nans()
                )
                if diff.height == 0:
                    continue
                records.append(
                    {"contrast": f"{a} - {b}", "slice": sname, "metric": metric, "rowset": rowset,
                     "counted": (sname in GATING_CRPS and metric == "crps")
                     or (sname in GATING_Q99 and metric == "pin_q99"),
                     **block_bootstrap(diff=diff, rng=rng)}
                )  # fmt: skip
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(SCORE_DIR / "contrasts.csv")
    print(f"counted tests: {int(contrasts['counted'].sum())} (pre-registered {TESTS})")
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(
            contrasts.filter(pl.col("metric") == "crps")
            .drop("by_month")
            .with_columns(pl.col(pl.Float64).round(3))
        )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rows", "stats"))
    args = parser.parse_args()
    {"rows": rows_command, "stats": stats_command}[args.command]()


if __name__ == "__main__":
    main()
