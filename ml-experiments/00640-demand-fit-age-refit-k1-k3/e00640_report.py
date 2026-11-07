# ruff: noqa: E501
"""Experiment 640: slopes of CRPS against fit age, with month-block bootstrap intervals (CPU).

Reads `score/cells_{k1,k3}.parquet` (written by `e00640_score.py`) and prints every number the
page quotes, and writes `score/slopes.csv`, `score/by_age.csv`, and `score/by_week.csv`.

The slope is the change in CRPS per 4 weeks of fit age, as a percentage of the age-0 CRPS: the
ordinary-least-squares slope of the month-mean CRPS on the five fit ages (0, 4, 8, 16, 26 weeks),
times 4, divided by the age-0 month-mean CRPS. The bootstrap resamples the five development months
(blocks) with replacement, so every Primary stays paired within a month.

Usage: python e00640_report.py
"""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00640-demand-fit-age-refit-k1-k3" / "score"
AGES = np.array([0, 4, 8, 16, 26])
MONTHS = ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
N_BOOT = 20000
LEVEL = 97.5  # two planned slopes (K1 and K3): Bonferroni
BANDS = {None: "0-240 h", 0: "0-24 h", 1: "24-120 h", 2: "120-240 h"}
SHORT = ["D15 and D18"]


def cells(*, net: str) -> pl.DataFrame:
    """Return the cells of a network with the Primary-month event flag added."""
    d = pl.read_parquet(ROOT / f"cells_{net}.parquet")
    has_event = (
        d.group_by(["month", "prim"])
        .agg((~pl.col("free")).any().alias("event_month"))
        .select("month", "prim", "event_month")
    )
    return d.join(has_event, on=["month", "prim"])


def grid(*, d: pl.DataFrame, metric: str = "crps") -> np.ndarray:
    """Return the month by age matrix of mean `metric` over the rows of `d`."""
    g = d.group_by(["month", "age"]).agg(pl.col(metric).sum() / pl.col("n").sum())
    out = np.full((len(MONTHS), len(AGES)), np.nan)
    for m, a, v in g.iter_rows():
        out[MONTHS.index(m), list(AGES).index(a)] = v
    return out


def slope_pct(*, g: np.ndarray, ages: np.ndarray = AGES, keep: np.ndarray | None = None) -> float:
    """Return the CRPS change per 4 weeks as a percentage of the age-0 mean over the months."""
    mean = np.nanmean(g, axis=0)
    if keep is not None:
        mean = np.nanmean(np.where(keep, g, np.nan), axis=0)
    ok = np.isfinite(mean)
    x, y = ages[ok], mean[ok]
    b = np.polyfit(x, y, 1)[0]
    return float(b * 4 / y[x == 0][0] * 100)


def boot(
    *, g: np.ndarray, keep: np.ndarray | None = None, seed: int = 0
) -> tuple[float, float, float]:
    """Return the slope and its month-block bootstrap interval at `LEVEL`."""
    rng = np.random.default_rng(seed)
    pt = slope_pct(g=g, keep=keep)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(MONTHS), len(MONTHS))
        draws.append(slope_pct(g=g[idx], keep=None if keep is None else keep[idx]))
    lo, hi = np.percentile(draws, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return pt, float(lo), float(hi)


def contrasts(*, nets: tuple[str, ...] = ("k1", "k3")) -> pl.DataFrame:
    """Return the CRPS ratio of each older fit age to age 0, in percent, with month-block intervals."""
    rng = np.random.default_rng(1)
    out = []
    for net in nets:
        d = cells(net=net).filter(pl.col("free"))
        for sname, sub in {
            "all Primaries": d,
            "without D15 and D18": d.filter(~pl.col("group").is_in(SHORT)),
        }.items():
            g = grid(d=sub)
            draws = [
                g[rng.integers(0, len(MONTHS), len(MONTHS))].mean(axis=0) for _ in range(N_BOOT)
            ]
            ratio = np.array([(x / x[0] - 1) * 100 for x in draws])
            pt = (g.mean(axis=0) / g.mean(axis=0)[0] - 1) * 100
            lo, hi = np.percentile(ratio, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2], axis=0)
            for i, a in enumerate(AGES[1:], start=1):
                out.append(
                    {
                        "net": net,
                        "subset": sname,
                        "age": int(a),
                        "pct_vs_age0": pt[i],
                        "lo": lo[i],
                        "hi": hi[i],
                    }
                )
    return pl.DataFrame(out)


def main() -> None:
    """Print the tables and write the CSV files."""
    rows, by_age, by_week = [], [], []
    for net in ("k1", "k3"):
        d = cells(net=net)
        subsets = {
            "all Primaries, event-free rows": d.filter(pl.col("free")),
            "without D15 and D18": d.filter(pl.col("free") & ~pl.col("group").is_in(SHORT)),
            "Primary-months with an event (event-free rows)": d.filter(
                pl.col("free") & pl.col("event_month")
            ),
            "Primary-months without an event": d.filter(pl.col("free") & ~pl.col("event_month")),
            "all rows, events included": d,
        }
        for sname, sub in subsets.items():
            for band, bname in BANDS.items():
                s = sub if band is None else sub.filter(pl.col("band") == band)
                g = grid(d=s)
                pt, lo, hi = boot(g=g)
                rows.append(
                    {
                        "net": net,
                        "subset": sname,
                        "lead": bname,
                        "slope_pct_per_4wk": pt,
                        "lo": lo,
                        "hi": hi,
                    }
                )
                # the same slope with the 2025-10 fits older than 16 weeks left out (history-limited)
                keep = np.ones_like(g, dtype=bool)
                keep[0, AGES > 16] = False
                pt2, lo2, hi2 = boot(g=g, keep=keep)
                rows.append(
                    {
                        "net": net,
                        "subset": sname + " | 2025-10 ages up to 16 weeks",
                        "lead": bname,
                        "slope_pct_per_4wk": pt2,
                        "lo": lo2,
                        "hi": hi2,
                    }
                )
        free = d.filter(pl.col("free"))
        for age in AGES:
            a = free.filter(pl.col("age") == age)
            n = a["n"].sum()
            by_age.append(
                {"net": net, "age": int(age), "crps": a["crps"].sum() / n, "mae": a["abs_err"].sum() / n,
                 "cover90": a["cover90"].sum() / n, "cover98": a["cover98"].sum() / n,
                 **{k: a[k].sum() / n for k in ("pin01", "pin05", "pin95", "pin99")}}
            )  # fmt: skip
        w = (
            free.with_columns(since=pl.col("age") + pl.col("week"))
            .group_by(["month", "age", "since"])
            .agg(pl.col("crps").sum() / pl.col("n").sum())
        )
        w = w.group_by(["age", "since"]).agg(pl.col("crps").mean()).with_columns(net=pl.lit(net))
        by_week.append(w)
    pl.DataFrame(rows).write_csv(ROOT / "slopes.csv")
    pl.DataFrame(by_age).write_csv(ROOT / "by_age.csv")
    pl.concat(by_week).write_csv(ROOT / "by_week.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=80, float_precision=4):
        print(pl.DataFrame(rows))
        print(pl.DataFrame(by_age))
        ct = contrasts()
        ct.write_csv(ROOT / "contrasts.csv")
        print(ct)
    # P4: the age-0 fit on the first 14 days against experiment 613's roster (K2 system, 5 seeds)
    for net in ("k1",):
        p = ROOT / f"cells_{net}.parquet"
        if p.exists():
            d = pl.read_parquet(p).filter((pl.col("age") == 0) & (pl.col("week") < 2))
            mine, ref = d["crps"].sum() / d["n"].sum(), d["ref_crps_k2"].sum() / d["n"].sum()
            print(
                f"P4 {net}: age-0 first 14 days CRPS {mine:.5f}, roster K2 {ref:.5f}, ratio {mine / ref:.4f}"
            )


main()
