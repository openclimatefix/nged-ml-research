# ruff: noqa: E501, E731, PERF401, RUF034
r"""Score experiment 485's arms on experiment 470's rows, slices, and 1.15 widening.

H0 is experiment 425's arm A seeds 0 to 2 (experiment 470's stored `A-s0` to `A-s2`, the same
files as 425's `seed_preds/A`). Each arm is scored seed by seed and the per-row metrics averaged,
as experiment 470 does for its three-cell constructs. Gate slices drop the window days (22
December to 2 January by UTC valid date); the Christmas slices are the window days of fold 2025-12
(22 to 31 December), the days either side (17 to 21 December, 3 to 6 January), and the window days
of fold 2026-01 (1 and 2 January; report-only because that fold trains on Christmas 2025).

```bash
source /home/jack/e485env.sh
uv run --no-sync python ml-experiments/00485-*/e00485_score.py rows
uv run --no-sync python ml-experiments/00485-*/e00485_score.py stats
```
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00485_fit as fit485  # ty: ignore[unresolved-import]
import e00485_regime as regime  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTHS: Final[tuple[str, ...]] = tuple(sorted(fit485.MONTHS5))
ARMS: Final[tuple[str, ...]] = ("H1", "H2", "H3", "H4", "P")
SEEDS: Final[tuple[int, ...]] = fit485.SEEDS3
REFS: Final[tuple[str, ...]] = ("BG2", "manual_heuristic", "climatology")
TESTS: Final[int] = 45
SCORE_DIR: Final[Path] = fit485.WORK_DIR / "scores"
METRICS: Final[tuple[str, ...]] = (
    "crps", "pin_q01", "pin_q05", "pin_q95", "pin_q99", "cov_q01", "cov_q05", "cov_q95", "cov_q99",
)  # fmt: skip
GATE: Final[dict[str, tuple[str, str]]] = {
    "eight": ("mva_eight", "all"),
    "six_clean": ("signed_six", "clean"),
    "six_nra": ("signed_six", "nra"),
    "cold_eight": ("mva_eight", "cold"),
    "cold_six": ("signed_six", "cold"),
    "band_eight": ("mva_eight", "band"),
    "band_six": ("signed_six", "band"),
}
CHRISTMAS: Final[dict[str, tuple[str, str]]] = {
    "xmas_eight": ("mva_eight", "xmas"),
    "xmas_six": ("signed_six", "xmas"),
    "sides_eight": ("mva_eight", "sides"),
    "sides_six": ("signed_six", "sides"),
    "jan12_eight": ("mva_eight", "jan12"),
    "jan12_six": ("signed_six", "jan12"),
}
GATE_Q99: Final[tuple[str, ...]] = ("eight", "six_clean")


def seed_frame(*, arm: str, month: str, seed: int, sites: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's raw seed quantiles keyed by experiment 470's (primary, origin, target)."""
    if arm == "H0":
        return pl.read_parquet(build470.prediction_path(arm=f"A-s{seed}", month=month, cell=0))
    frame = pl.read_parquet(fit485.OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet")
    frame = frame.join(sites.select("time_series_id", "primary"), on="time_series_id")
    return build470.grid_keys(frame=frame)


def month_table(*, month: str, sites: pl.DataFrame, ctx: dict) -> pl.DataFrame:
    """Return a month's row-level metrics for every arm (headline widening, seeds averaged)."""
    base = s470.month_context(month=month, sites=sites, ctx=ctx)
    valid_day = base["valid_time"].dt.date().dt.epoch("d").to_numpy().astype(np.int64)
    window = regime.in_window_day(day=valid_day)
    month_d = base["valid_time"].dt.month().to_numpy()
    dom = base["valid_time"].dt.day().to_numpy()
    sides = ((month_d == 12) & (dom >= 17) & (dom <= 21)) | (
        (month_d == 1) & (dom >= 3) & (dom <= 6)
    )
    cols: dict[str, np.ndarray] = {}
    for arm in ("H0", *ARMS, *REFS):
        cells = []
        for seed in SEEDS if arm not in REFS else (0,):
            stored = (
                s470.read_cell(arm=arm, month=month, cell=0)
                if arm in REFS
                else seed_frame(arm=arm, month=month, seed=seed, sites=sites)
            )
            frame = base.select(*s470.KEYS, "actual", "nra", "tgt_real").join(
                stored.select(*s470.KEYS, *s470.LEVELS),
                on=s470.KEYS,
                how="left",
                maintain_order="left",
            )
            frame = s470.widen_arm(arm=arm, frame=frame, factor=s470.HEADLINE)
            ok = frame["q50"].is_not_null().to_numpy() & np.isfinite(frame["q50"].to_numpy())
            frame = frame.with_columns(*[pl.col(c).fill_null(0.0) for c in s470.LEVELS])
            metered = s470.row_metrics(frame=frame)
            nra = s470.row_metrics(
                frame=frame.with_columns(actual_nra=pl.col("actual") - pl.col("nra")),
                actual="actual_nra",
            )
            cells.append(({k: np.where(ok, v, np.nan) for k, v in metered.items()},
                          {k: np.where(ok, v, np.nan) for k, v in nra.items()}, ok))  # fmt: skip
        for metric in cells[0][0]:
            cols[f"{arm}__{metric}"] = np.mean([c[0][metric] for c in cells], axis=0)
            cols[f"{arm}__nra_{metric}"] = np.mean([c[1][metric] for c in cells], axis=0)
        if arm == "H0":
            for i, c in enumerate(cells):
                cols[f"H0__crps_cell{i}"] = c[0]["crps"]
    out = pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()})
    have = lambda arms: np.logical_and.reduce(
        [np.isfinite(out[f"{a}__crps"].to_numpy()) for a in arms]
    )
    return out.hstack(
        base.select("group", "cls", "lead_hours", "lead_band", "tgt_real", "cold", "band")
    ).with_columns(
        month=pl.lit(month),
        valid_day=pl.Series(valid_day),
        window=pl.Series(window),
        xmas=pl.Series(window & (month == "2025-12") & (month_d == 12)),
        sides=pl.Series(sides),
        jan12=pl.Series(window & (month == "2026-01")),
        main=pl.Series(have(("H0", *ARMS))),
        ref=pl.Series(have(("H0", *ARMS, *REFS))),
    )


def rows_command() -> None:
    """Write the row-level table of each month."""
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    sites = build470.site_map()
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        table = month_table(month=month, sites=sites, ctx=ctx)
        table.write_parquet(SCORE_DIR / f"rows_{month}.parquet")
        print(
            f"{month}: {table.height} rows; main {int(table['main'].sum())}; ref {int(table['ref'].sum())}; window {int(table['window'].sum())}",
            flush=True,
        )


def pick(*, table: pl.DataFrame, group: str, kind: str, gate: bool) -> tuple[pl.DataFrame, str]:
    """Return the rows of a slice and the metric prefix; gate slices drop the window days."""
    part = table.filter(pl.col("group") == group)
    prefix = ""
    if gate:
        part = part.filter(~pl.col("window"))
    if kind == "clean":
        part = part.filter(pl.col("cls") == "clean")
    elif kind == "nra":
        part = part.filter(pl.col("tgt_real"))
        prefix = "nra_"
    elif kind == "cold":
        part = part.filter(pl.col("cold"))
    elif kind == "band":
        part = part.filter(pl.col("band"))
    elif kind in ("xmas", "sides", "jan12"):
        part = part.filter(pl.col(kind))
    return part, prefix


def mde(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect from H0's three-seed spread."""
    records = []
    for name, (group, kind) in {**GATE, **CHRISTMAS}.items():
        part, prefix = pick(table=table, group=group, kind=kind, gate=name in GATE)
        part = part.filter(pl.col("main"))
        if prefix:
            continue
        rows = []
        for m in MONTHS:
            sub = part.filter(pl.col("month") == m)
            if sub.height:
                rows.append([sub[f"H0__crps_cell{i}"].mean() for i in range(len(SEEDS))])
        if len(rows) < 1:
            continue
        values = np.array(rows)
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (len(rows) * (len(SEEDS) - 1)))
        records.append({"slice": name, "folds": len(rows), "rows": part.height, "seed_sd": sd,
                        "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(SEEDS))})  # fmt: skip
    return pl.DataFrame(records)


def stats_command() -> None:
    """Write the minimum detectable effect, absolute scores, tails, and contrasts."""
    table = pl.concat([pl.read_parquet(SCORE_DIR / f"rows_{m}.parquet") for m in MONTHS])
    minimum = mde(table=table)
    print("minimum detectable effect (printed before any contrast):")
    print(minimum)
    minimum.write_csv(SCORE_DIR / "mde.csv")
    s470.TESTS = TESTS
    absolute = []
    for name, (group, kind) in {**GATE, **CHRISTMAS}.items():
        for rowset in ("main", "ref"):
            part, prefix = pick(table=table, group=group, kind=kind, gate=name in GATE)
            part = part.filter(pl.col(rowset))
            if not part.height:
                continue
            for arm in ("H0", *ARMS, *(REFS if rowset == "ref" else ())):
                absolute.append(
                    {"slice": name, "rowset": rowset, "arm": arm, "n": part.height}
                    | {m: part[f"{arm}__{prefix}{m}"].mean() for m in METRICS}
                )  # fmt: skip
    pl.DataFrame(absolute).write_csv(SCORE_DIR / "absolute.csv")
    rng = np.random.default_rng(0)
    records = []
    for arm in ARMS:
        for name, (group, kind) in {**GATE, **CHRISTMAS}.items():
            gate = name in GATE
            part, prefix = pick(table=table, group=group, kind=kind, gate=gate)
            part = part.filter(pl.col("main"))
            for metric in ("crps", "pin_q99") if (name in GATE_Q99 or not gate) else ("crps",):
                diff = (
                    part.select(
                        "month",
                        day=pl.col("valid_day") if not gate else pl.col("valid_day"),
                        d=pl.col(f"{arm}__{prefix}{metric}") - pl.col(f"H0__{prefix}{metric}"),
                    )
                    .drop_nulls()
                    .drop_nans()
                )
                if not diff.height:
                    continue
                records.append(
                    {"contrast": f"{arm} - H0", "slice": name, "metric": metric,
                     "counted": gate and (metric == "crps" or name in GATE_Q99),
                     **s470.block_bootstrap(diff=diff, rng=rng)}
                )  # fmt: skip
    contrasts = pl.DataFrame(records)
    contrasts.write_csv(SCORE_DIR / "contrasts.csv")
    print(f"counted tests: {int(contrasts['counted'].sum())} (pre-registered {TESTS})")
    with pl.Config(tbl_rows=300, tbl_width_chars=220):
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
