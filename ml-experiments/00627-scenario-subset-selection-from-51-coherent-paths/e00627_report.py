# ruff: noqa: E501, D103
"""Experiment 627: contrasts with month-block bootstrap intervals, printed as a report."""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00627-scenario-subset-selection-from-51-coherent-paths" / "work"
)
ALPHA: Final[float] = 0.01
N_BOOT: Final[int] = 20000
RNG = np.random.default_rng(627)
LABEL_FREE: Final[tuple[str, ...]] = ("S-wind", "S-solar", "S-net", "S-demand", "C")
ARMS: Final[tuple[str, ...]] = ("R", *LABEL_FREE, "Oracle")
COVER_BAR: Final[float] = 0.10
HEAD: Final[dict] = {"coupling": "J3", "selection": "d1_10", "band": "d1_10", "k": 5}


def month() -> pl.Expr:
    return pl.col("power_fcst_init_time").dt.strftime("%Y-%m").alias("month")


def boot_mean(x: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Return the mean of x with a bootstrap interval over its entries (month blocks)."""
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def per_run(sel: pl.DataFrame, **flt: object) -> pl.DataFrame:
    """Return per-run means over the four summaries of the coverage and tail measures."""
    f = sel
    for key, val in flt.items():
        f = f.filter(pl.col(key) == val)
    return (
        f.with_columns((pl.col("tail_low") + pl.col("tail_high")).truediv(2).alias("tail"))
        .group_by("power_fcst_init_time", "arm")
        .agg(pl.col(["w1", "gap05", "gap50", "gap95", "tail", "w1_sd", "gap95_sd"]).mean())
    )


def monthly(df: pl.DataFrame, col: str, arm: str) -> np.ndarray:
    f = df.filter(pl.col("arm") == arm).with_columns(month())
    return f.group_by("month").agg(pl.col(col).mean()).sort("month")[col].to_numpy()


def bsp_part() -> None:
    """Print the BSP-group scores: coverage of the peak export, tail, and the gate."""
    print("\n== Part 2: BSP group peak export (J3, days 1-10) ==")
    bsp = pl.read_parquet(W / "bsp.parquet").with_columns(pl.col(pl.Float64).fill_nan(None))
    d = bsp.filter((pl.col("coupling") == "J3") & (pl.col("band") == "d1_10"))
    alls = (
        d.filter(pl.col("arm") == "All")
        .group_by("power_fcst_init_time")
        .agg(pl.col("exceed").mean().alias("all"))
    )
    for k in (3, 4, 5, 10):
        for arm in ("R", "S-bsp", "C", "Q"):
            f = d.filter((pl.col("arm") == arm) & (pl.col("k") == k))
            if f.is_empty():
                continue
            pr = (
                f.group_by("power_fcst_init_time")
                .agg(pl.col(["gap05", "gap50", "gap95", "tail_high", "exceed"]).mean())
                .join(alls, on="power_fcst_init_time")
                .drop_nulls()
            )
            m = (
                pr.with_columns(month())
                .group_by("month")
                .agg(pl.col(["gap05", "gap50", "gap95", "tail_high", "exceed", "all"]).mean())
                .sort("month")
            )
            dm = (m["exceed"] - m["all"]).to_numpy()
            est, lo, hi = boot_mean(dm)
            print(
                f"K={k} {arm:6s} gap05 {m['gap05'].mean():.3f} gap50 {m['gap50'].mean():.3f} gap95 {m['gap95'].mean():.3f} export-tail {m['tail_high'].mean():.3f} | exceed {m['exceed'].mean():.3f} All {m['all'].mean():.3f} diff {est:+.3f} ({lo:+.3f},{hi:+.3f}) gate(<=0.05): {est <= 0.05} runs {pr.height}"
            )
    print("per BSP, K=5 and K=4:")
    with pl.Config(tbl_rows=60):
        t = (
            d.filter(pl.col("k").is_in([4, 5, 51]))
            .group_by("bsp", "arm", "k")
            .agg(pl.col(["gap05", "gap95", "tail_high", "exceed"]).mean())
            .sort("bsp", "k", "arm")
        )
        print(t.with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:  # noqa: PLR0915
    sel = pl.read_parquet(W / "selection.parquet").with_columns(pl.col(pl.Float64).fill_nan(None))
    joint = pl.read_parquet(W / "joint.parquet")
    runs = sel["power_fcst_init_time"].unique().sort()
    months = runs.dt.strftime("%Y-%m").unique().sort()
    print(f"runs: {len(runs)}, months: {len(months)} ({months[0]} to {months[-1]})")
    print(f"last run: {runs[-1]}")

    print("\n== Headline: K=5, days 1-10, J3, selection on days 1-10 (mean over 4 summaries) ==")
    pr = per_run(sel, **HEAD)
    r_tail = monthly(pr, "tail", "R")
    verdict_rows = []
    for arm in ARMS:
        row = {c: monthly(pr, c, arm).mean() for c in ("gap05", "gap50", "gap95", "w1", "tail")}
        cover_ok = all(row[c] <= COVER_BAR for c in ("gap05", "gap50", "gap95"))
        line = (
            f"{arm:9s} gap05 {row['gap05']:.3f} gap50 {row['gap50']:.3f} gap95 {row['gap95']:.3f} "
            f"w1 {row['w1']:.3f} tail {row['tail']:.3f}"
        )
        if arm in LABEL_FREE:
            d = monthly(pr, "tail", arm) - r_tail
            m, lo, hi = boot_mean(d)
            line += (
                f" | tail-R {m:+.3f} ({lo:+.3f}, {hi:+.3f}) months>0 {int((d > 0).sum())}/{len(d)}"
            )
            verdict_rows.append((arm, cover_ok, lo > 0))
        print(line, f"| cover<=0.10: {cover_ok}")
    if verdict_rows:
        both = [a for a, c, t in verdict_rows if c and t]
        tail_only = [a for a, c, t in verdict_rows if t]
        print("pass both:", both, "| pass tail:", tail_only)
        word = "supported" if both else "partly supported" if tail_only else "not supported"
        print("VERDICT:", word)
    # spread of R across draws (mean of per-run sd)
    r_rows = pr.filter(pl.col("arm") == "R")
    print(
        f"R spread across draws (mean per-run sd): w1 {r_rows['w1_sd'].mean():.3f} gap95 {r_rows['gap95_sd'].mean():.3f}"
    )

    print("\n== Gap (mean of 5/50/95), tail coverage by K, band, selection, coupling ==")
    table = (
        sel.with_columns(
            ((pl.col("gap05") + pl.col("gap50") + pl.col("gap95")) / 3).alias("gap"),
            ((pl.col("tail_low") + pl.col("tail_high")) / 2).alias("tail"),
        )
        .group_by("coupling", "selection", "band", "k", "arm")
        .agg(pl.col(["gap", "gap05", "gap95", "w1", "tail"]).mean())
        .sort("coupling", "selection", "band", "k", "arm")
    )
    with pl.Config(tbl_rows=400, tbl_cols=12, fmt_float="mixed"):
        print(
            table.filter((pl.col("coupling") == "J3") & (pl.col("band") == "d1_10")).with_columns(
                pl.col(pl.Float64).round(3)
            )
        )
        print(
            table.filter(
                (pl.col("coupling") == "J3") & (pl.col("selection") == "d1_10") & (pl.col("k") == 5)
            ).with_columns(pl.col(pl.Float64).round(3))
        )

    print("\n== Per summary, K=5, days 1-10, J3 ==")
    ps = (
        sel.filter(
            (pl.col("coupling") == "J3")
            & (pl.col("selection") == "d1_10")
            & (pl.col("band") == "d1_10")
            & (pl.col("k") == 5)
        )
        .with_columns(((pl.col("tail_low") + pl.col("tail_high")) / 2).alias("tail"))
        .group_by("summary", "arm")
        .agg(pl.col(["gap05", "gap50", "gap95", "tail"]).mean())
        .sort("summary", "arm")
    )
    with pl.Config(tbl_rows=100):
        print(ps.with_columns(pl.col(pl.Float64).round(3)))

    print(
        "\n== Joint energy and variogram score relative to All (days 1-10, J3, selection days 1-10) =="
    )
    for k in (3, 5, 10):
        for score in ("energy", "variogram"):
            base = joint.filter(
                (pl.col("arm") == "All")
                & (pl.col("coupling") == "J3")
                & (pl.col("band") == "d1_10")
            )
            b = (
                base.with_columns(month())
                .group_by("month")
                .agg(pl.col(score).mean())
                .sort("month")[score]
                .to_numpy()
            )
            out = []
            for arm in ("R", *LABEL_FREE, "Oracle"):
                f = joint.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("coupling") == "J3")
                    & (pl.col("band") == "d1_10")
                    & (pl.col("selection") == "d1_10")
                    & (pl.col("k") == k)
                )
                a = (
                    f.with_columns(month())
                    .group_by("month")
                    .agg(pl.col(score).mean())
                    .sort("month")[score]
                    .to_numpy()
                )
                n = min(len(a), len(b))
                idx = RNG.integers(0, n, size=(N_BOOT, n))
                r = a[:n][idx].mean(1) / b[:n][idx].mean(1) - 1
                lo, hi = np.quantile(r, [ALPHA / 2, 1 - ALPHA / 2])
                out.append(f"{arm} {a.mean() / b.mean() - 1:+.1%} ({lo:+.1%},{hi:+.1%})")
            print(f"K={k} {score}: abs All {b.mean():.3f} | " + "; ".join(out))

    print("\n== Stability ==")
    summ = pl.read_parquet(W / "summ.parquet")
    for coupling in ("J3", "J2"):
        for name in ("wind", "solar", "net", "demand"):

            def vec(band: str, name: str = name, coupling: str = coupling) -> pl.DataFrame:
                return summ.filter(
                    (pl.col("coupling") == coupling)
                    & (pl.col("summary") == name)
                    & (pl.col("band") == band)
                ).sort("power_fcst_init_time")

            a, b = vec("d1_3"), vec("d4_10")
            rho, same = [], []
            for va, vb in zip(a["values"].to_list(), b["values"].to_list(), strict=True):
                ra = np.argsort(np.argsort(va))
                rb = np.argsort(np.argsort(vb))
                rho.append(np.corrcoef(ra, rb)[0, 1])
                sa, sb = ra * 5 // 51, rb * 5 // 51
                same.append(float((sa == sb).mean()))
            print(
                f"{coupling} within-run d1_3 vs d4_10 {name}: spearman {np.mean(rho):+.3f} same-stratum(K=5) {np.mean(same):.3f} (chance 0.2)"
            )
    bsp_part()


if __name__ == "__main__":
    main()
