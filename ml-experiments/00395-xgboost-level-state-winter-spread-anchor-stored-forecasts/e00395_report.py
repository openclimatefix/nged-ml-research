"""Print every number experiment 395's write-up quotes for Part A, from `e00395_build.py` tables.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00395-*/e00395_report.py a
```
"""
# ruff: noqa: E501, PLR0915

import sys
from pathlib import Path
from typing import Final

import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "work"
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
HEADLINE: Final[str] = "lead_days_1_10"
SIX: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EIGHT: Final[list[str]] = ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
MAIN: Final[list[str]] = [
    "K0",
    "K0_w",
    "K1",
    "K1_w",
    "K2",
    "K2_w",
    "K3",
    "K3_w",
    "stack",
    "bg2",
    "bg2_only_W",
    "manual_heuristic",
    "climatology",
]
BANDS: Final[list[tuple[str, float, float]]] = [
    ("0-24h", 0, 24),
    ("24-72h", 24, 72),
    ("72-168h", 72, 168),
    ("168-240h", 168, 240),
]


def load(name: str, table: pl.DataFrame, prefix: str = "rows") -> pl.DataFrame:
    """Return an arm's rows joined to the slice table."""
    return pl.read_parquet(WORK / f"{prefix}_{name}.parquet").join(
        table, on=KEYS, how="left", validate="1:1"
    )


def band_expr() -> pl.Expr:
    """Return the lead band of a row, including the 0 to 24 hour band."""
    out = pl.when(pl.col("lead_hours") < BANDS[0][2]).then(pl.lit(BANDS[0][0]))
    for name, _, hi in BANDS[1:]:
        out = out.when(pl.col("lead_hours") < hi).then(pl.lit(name))
    return out.otherwise(pl.lit("beyond"))


def part_a() -> None:  # noqa: PLR0912
    """Print the Part A tables."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    table = pl.read_parquet(WORK / "slice_table.parquet")
    arms = {n: load(n, table) for n in MAIN}
    head = {n: a.filter(pl.col(HEADLINE)) for n, a in arms.items()}
    print("\n== A1: CRPS (points), days 1 to 10, per Primary ==")
    t = None
    for n, a in head.items():
        piece = a.group_by("label").agg(pl.col("crps").mean().alias(n))
        t = piece if t is None else t.join(piece, on="label")
    t = t.sort(pl.col("label").str.slice(1).cast(pl.Int32))  # ty: ignore[unresolved-attribute]
    print(
        t.select("label", *[c for c in t.columns if c != "label"]).with_columns(
            pl.selectors.numeric().round(3)
        )
    )
    print("\n== A1b: mean over groups ==")
    for gname, labs in (("signed-MW six", SIX), ("MVA-only eight", EIGHT)):
        print(
            gname,
            {
                n: round(a.filter(pl.col("label").is_in(labs))["crps"].mean(), 3)  # ty: ignore[no-matching-overload]
                for n, a in head.items()
            },
        )
    print(
        "\n== A2: share of gap to stack closed (CRPS days 1 to 10); raw and 1.15, K vs K0 same widening =="
    )
    for lab in ("D11", "D14"):
        s = head["stack"].filter(pl.col("label") == lab)["crps"].mean()
        for suffix in ("", "_w"):
            b = head[f"K0{suffix}"].filter(pl.col("label") == lab)["crps"].mean()
            line = [f"{lab}{suffix or ' raw'}: K0 {b:.3f} stack {s:.3f} gap {b - s:+.3f}"]  # ty: ignore[unsupported-operator]
            for k in ("K1", "K2", "K3"):
                v = head[f"{k}{suffix}"].filter(pl.col("label") == lab)["crps"].mean()
                line.append(f"{k} {v:.3f} closes {(b - v) / (b - s):.0%}")  # ty: ignore[unsupported-operator]
            print("; ".join(line))
    print("\n== A3: K minus K0 (same widening), 95% month bootstrap ==")
    for k in ("K1", "K2", "K3"):
        for suffix in ("", "_w"):
            for lab in ["D11", "D14", "D2", "D4", "six", "eight"]:
                labs = SIX if lab == "six" else EIGHT if lab == "eight" else [lab]
                where = pl.col("label").is_in(labs) & pl.col(HEADLINE)
                r = metrics.month_bootstrap_contrast(
                    treatment=arms[f"{k}{suffix}"],
                    reference=arms[f"K0{suffix}"],
                    metric="crps",
                    where=where,
                    slice_table=table,
                )
                print(
                    f"{k}{suffix:2s} {lab:6s} {r['difference']:+.3f} ({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
    print("\n== A4: CRPS by lead band, signed-MW six and eight (1.15 arms) ==")
    for gname, labs in (("six", SIX), ("eight", EIGHT)):
        for n in ("K0_w", "K1_w", "K2_w", "K3_w", "stack"):
            a = (
                arms[n]
                .filter(pl.col("label").is_in(labs), pl.col("lead_hours") < 240)
                .with_columns(band=band_expr())
            )
            print(
                gname,
                n,
                {
                    r["band"]: round(r["crps"], 3)
                    for r in a.group_by("band")
                    .agg(pl.col("crps").mean())
                    .sort("band")
                    .iter_rows(named=True)
                },
            )
    print("\n== A5: winter and top-quartile gap closure at D11 and D14 (1.15 arms) ==")
    for lab in ("D11", "D14"):
        for sname, cond in (
            ("winter", pl.col("winter")),
            (
                "top quartile",
                pl.col("actual_points")
                >= pl.col("actual_points").quantile(0.75).over("time_series_id"),
            ),
        ):
            vals = {}
            for n in ("K0_w", "K1_w", "K2_w", "K3_w", "stack"):
                a = arms[n].filter(pl.col("label") == lab, pl.col(HEADLINE)).filter(cond)
                vals[n] = a["crps"].mean()
            gap = vals["K0_w"] - vals["stack"]  # ty: ignore[unsupported-operator]
            print(
                lab,
                sname,
                {k: round(v, 3) for k, v in vals.items()},  # ty: ignore[no-matching-overload]
                {k: f"{(vals['K0_w'] - vals[k]) / gap:.0%}" for k in ("K1_w", "K2_w", "K3_w")},  # ty: ignore[unsupported-operator]
            )
    print("\n== A6: coverage at q01,q05,q95,q99 and cold-night q95, six and eight (1.15 arms) ==")
    for gname, labs in (("six", SIX), ("eight", EIGHT)):
        for n in ("K0_w", "K1_w", "K2_w", "K3_w", "stack"):
            a = arms[n].filter(pl.col("label").is_in(labs), pl.col(HEADLINE))
            cov = {
                c: round(a[c].mean(), 3)  # ty: ignore[no-matching-overload]
                for c in ("coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99")
            }
            cold = round(a.filter(pl.col("cold_nights"))["coverage_q95"].mean(), 3)  # ty: ignore[no-matching-overload]
            print(gname, n, cov, "cold-night q95", cold)
    print(
        "\n== A7: event rows (CRPS 1.15 arms, six): lag in event / target in event / 7 days after =="
    )
    flags = pl.read_parquet(WORK / "event_flags.parquet")
    for n in ("K0_w", "K1_w", "K2_w", "K3_w", "stack"):
        a = (
            arms[n]
            .join(flags, on=KEYS, how="left")
            .filter(pl.col("label").is_in(SIX), pl.col(HEADLINE))
        )
        out = {}
        for nm, cond in (
            ("lag in event", pl.col("lag_in_event")),
            ("in event", pl.col("event_rows")),
            ("7d after", pl.col("event_following_7_days")),
        ):
            sub = a.filter(cond)
            out[nm] = (sub.height, round(sub["crps"].mean(), 3) if sub.height else None)  # ty: ignore[no-matching-overload]
        print(n, out)
    print("\n== A8: K4 (six Primaries, own slice table) ==")
    table6 = pl.read_parquet(WORK / "slice_table_six.parquet")
    a6 = {
        n: load(n, table6, "rows6").filter(pl.col(HEADLINE))
        for n in ("stack", "stack_off", "K4", "K4_w", "K0", "K2")
    }
    for n, a in a6.items():
        print(
            n,
            round(a["crps"].mean(), 3),  # ty: ignore[no-matching-overload]
            {
                r["label"]: round(r["crps"], 3)
                for r in a.group_by("label")
                .agg(pl.col("crps").mean())
                .sort("label")
                .iter_rows(named=True)
            },
        )


if __name__ == "__main__":
    {"a": part_a}[sys.argv[1]]()
