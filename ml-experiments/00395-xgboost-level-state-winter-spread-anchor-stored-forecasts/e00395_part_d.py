"""Part D of experiment 395: a trailing level anchor on the stored weekly bundle (L1).

The anchor of a forecast origin is the median of the bundle's own residuals (actual minus median,
in points) over the previous 7, 14, or 28 days of readings at least 3.5 hours before the origin,
times `min(1, days since the Primary's first residual / window days)`, added to every quantile.
Nothing is refitted. Reuses Part A's slice table and the K0 rows.

    ... e00395_part_d.py build | report
"""
# ruff: noqa: E501

import sys
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00390_build as b390  # ty: ignore[unresolved-import]
import e00395_build as b395  # ty: ignore[unresolved-import]
import e00395_levels as lv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

WORK: Final[Path] = b395.WORK
KEYS: Final[list[str]] = lv.KEYS
WINDOWS: Final[tuple[int, ...]] = (7, 14, 28)
HEADLINE: Final[str] = "lead_days_1_10"
SHORT: Final[tuple[str, ...]] = ("D15", "D18")
HALF_HOURS_PER_DAY: Final[int] = 48


def anchor(*, pts: pl.DataFrame, streams_: dict[int, np.ndarray], days: int) -> pl.DataFrame:
    """Return the bundle with the trailing-median anchor added to every quantile (points)."""
    pieces = []
    for sid, residual in streams_.items():
        part = pts.filter(pl.col("time_series_id") == sid)
        cutoff, _ = lv.case_positions(cases=part)
        finite = np.flatnonzero(np.isfinite(residual))
        first = finite[0] if finite.size else 0
        shift = np.zeros(part.height)
        for origin_cutoff in np.unique(cutoff):
            lo = origin_cutoff - days * HALF_HOURS_PER_DAY + 1
            window = residual[max(lo, 0) : origin_cutoff + 1]
            window = window[np.isfinite(window)]
            if window.size:
                weight = min(1.0, (origin_cutoff - first) / HALF_HOURS_PER_DAY / days)
                shift[cutoff == origin_cutoff] = float(np.median(window)) * max(weight, 0.0)
        pieces.append(part.with_columns(*[pl.col(q) + pl.Series(shift) for q in lv.QUANTILES]))
    return pl.concat(pieces)


def build() -> None:
    """Write the per-row losses of L1 at 7, 14, and 28 days, raw and at 1.15."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    actuals = b395.actual_points(ctx=ctx, series=series)
    bundle_mw = s381.arm_frame(regime="weekly", arm="bundle_solar+sunny+hl1", series=series)
    pts = b395.to_points(frame=bundle_mw, series=series)
    streams_ = b395.streams(pts=pts, actuals=actuals)
    table = pl.read_parquet(WORK / "slice_table.parquet")
    for days in WINDOWS:
        shifted = anchor(pts=pts, streams_=streams_, days=days)
        for suffix, frame in (("", shifted), ("_w", lv.widen(frame=shifted, factor=b395.WIDEN))):
            name = f"L1_{days}{suffix}"
            rows = b390.trimmed_rows(
                name=name,
                frame=b395.to_mw(frame=frame, series=series),
                table=table,
                ctx=ctx,
                series=series,
            )
            rows.write_parquet(WORK / f"rows_{name}.parquet")
            print("wrote", name, rows.height, flush=True)


def report() -> None:
    """Print the Part D tables."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(250)
    table = pl.read_parquet(WORK / "slice_table.parquet")
    names = (
        ["K0", "K0_w"]
        + [f"L1_{d}{s}" for d in WINDOWS for s in ("", "_w")]
        + ["stack", "bg2", "manual_heuristic", "climatology"]
    )
    arms = {
        n: pl.read_parquet(WORK / f"rows_{n}.parquet")
        .join(table, on=KEYS, how="left", validate="1:1")
        .filter(pl.col(HEADLINE))
        for n in names
    }
    others = [lab for lab in arms["K0"]["label"].unique().to_list() if lab not in SHORT]
    print("== D1: CRPS days 1-10, D15 and D18 and the other 14 ==")
    for gname, labs in (("D15", ["D15"]), ("D18", ["D18"]), ("other 14", others)):
        print(
            gname,
            {
                n: round(a.filter(pl.col("label").is_in(labs))["crps"].mean(), 3)  # ty: ignore[no-matching-overload]
                for n, a in arms.items()
            },
        )
    print("\n== D2: signed median bias (forecast minus actual, points) and months scored ==")
    for gname, labs in (("D15", ["D15"]), ("D18", ["D18"]), ("other 14", others)):
        print(
            gname,
            {
                n: round(a.filter(pl.col("label").is_in(labs))["q50_error"].mean(), 3)  # ty: ignore[no-matching-overload]
                for n, a in arms.items()
                if n in ("K0", "L1_7", "L1_14", "L1_28", "K0_w")
            },
        )
    for lab in SHORT:
        k = (
            arms["K0"]
            .filter(pl.col("label") == lab)
            .with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
        )
        print(
            lab,
            "months scored:",
            k["month"].n_unique(),
            "rows",
            k.height,
            "first",
            k["month"].min(),
        )
    print(
        "\n== D3: by calendar month (D15, D18): CRPS and bias for K0, L1_7, L1_14, L1_28 (raw) =="
    )
    for lab in SHORT:
        pieces = [
            arms[n]
            .filter(pl.col("label") == lab)
            .with_columns(month=pl.col("valid_time").dt.truncate("1mo").dt.strftime("%Y-%m"))
            .group_by("month")
            .agg(crps=pl.col("crps").mean(), bias=pl.col("q50_error").mean())
            .sort("month")
            .rename({"crps": f"{n}_crps", "bias": f"{n}_bias"})
            for n in ("K0", "L1_7", "L1_14", "L1_28")
        ]
        out = pieces[0]
        for p in pieces[1:]:
            out = out.join(p, on="month")
        print(lab)
        print(out.with_columns(pl.selectors.numeric().round(3)))
    print("\n== D4: lead bands (0-24, 24-72, 72-168, 168-240) CRPS, D15+D18 ==")
    band = (
        pl.when(pl.col("lead_hours") < 24)
        .then(pl.lit("0-24h"))
        .when(pl.col("lead_hours") < 72)
        .then(pl.lit("24-72h"))
        .when(pl.col("lead_hours") < 168)
        .then(pl.lit("72-168h"))
        .otherwise(pl.lit("168-240h"))
    )
    for n in ("K0", "L1_7", "L1_14", "L1_28"):
        a = arms[n].filter(pl.col("label").is_in(list(SHORT))).with_columns(band=band)
        print(
            n,
            {
                r["band"]: round(r["crps"], 3)
                for r in a.group_by("band")
                .agg(pl.col("crps").mean())
                .sort("band")
                .iter_rows(named=True)
            },
        )
    print("\n== D5: coverage q01,q05,q95,q99 D15+D18 (1.15 arms) ==")
    for n in ("K0_w", "L1_7_w", "L1_14_w", "L1_28_w", "stack", "bg2"):
        a = arms[n].filter(pl.col("label").is_in(list(SHORT)))
        print(
            n,
            {
                c: round(a[c].mean(), 3)  # ty: ignore[no-matching-overload]
                for c in ("coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99")
            },
        )
    print("\n== D6: L1 minus K0 (same widening), month bootstrap ==")
    full = {
        n: pl.read_parquet(WORK / f"rows_{n}.parquet").join(
            table, on=KEYS, how="left", validate="1:1"
        )
        for n in ("K0", "K0_w", *[f"L1_{d}{s}" for d in WINDOWS for s in ("", "_w")])
    }
    for d in WINDOWS:
        for s in ("", "_w"):
            for gname, labs in (("D15", ["D15"]), ("D18", ["D18"]), ("other 14", others)):
                where = pl.col("label").is_in(labs) & pl.col(HEADLINE)
                r = metrics.month_bootstrap_contrast(
                    treatment=full[f"L1_{d}{s}"],
                    reference=full[f"K0{s}"],
                    metric="crps",
                    where=where,
                    slice_table=table,
                )
                print(
                    f"L1_{d}{s:2s} {gname:9s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f}) n={r['n_half_hours']}"
                )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    {"build": build, "report": report}[sys.argv[1]]()
