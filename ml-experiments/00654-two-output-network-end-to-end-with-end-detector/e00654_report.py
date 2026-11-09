# ruff: noqa: E501, D103, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654, step 3: tables and month-block intervals from `score/agg.parquet` (CPU).

Cell value = mean over lead days 1 to 9 of the weighted mean, seeds averaged, then months. Intervals
are percentile bootstraps over the 5 development months (month-block), as in experiment 652.
Prints the tables the write-up quotes and writes `score/tables.json`.

Usage: python e00654_report.py
"""

import json

import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SC = c654.WORK / "score"
METRICS = ["crps", "crps_s", "mae", "cov90", "pin01", "pin05", "pin95", "pin99", "cov98"]
MONTHS = ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]


def cells(*, agg: pl.DataFrame, setname: str, stratum: str) -> pl.DataFrame:
    """Return one row per (month, arm, seed) with the lead-day 1 to 9 mean of each metric."""
    d = agg.filter(
        (pl.col("set") == setname)
        & (pl.col("stratum") == stratum)
        & pl.col("lead_day").is_between(1, 9)
    )
    d = d.with_columns(*[(pl.col(m) / pl.col("sw")).alias(m) for m in METRICS])
    return d.group_by("month", "arm", "seed").agg(
        *[pl.col(m).mean() for m in METRICS], pl.col("n").sum().alias("n")
    )


def month_values(*, c: pl.DataFrame, arm: str, metric: str) -> np.ndarray:
    v = c.filter(pl.col("arm") == arm).group_by("month").agg(pl.col(metric).mean()).sort("month")
    return v[metric].to_numpy()


def boot(*, a: np.ndarray, b: np.ndarray, n: int = 20000) -> tuple[float, float, float]:
    d = a - b
    rng = np.random.default_rng(0)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    agg = pl.read_parquet(SC / "agg.parquet")
    out: dict = {"abs": {}, "contrast": {}}
    for setname, strata in (
        (
            "fam",
            [
                "inforce",
                "far",
                "middle",
                "near",
                "ended24",
                "just_ended",
                "free",
                "free14",
                "onset_ahead",
                "all",
            ],
        ),
        ("held", ["inforce", "all"]),
        ("real", ["real_free", "verified", "all"]),
    ):
        for st in strata:
            c = cells(agg=agg, setname=setname, stratum=st)
            if c.height == 0:
                continue
            t = (
                c.group_by("arm")
                .agg(*[pl.col(m).mean() for m in METRICS], pl.col("n").sum().alias("n"))
                .sort("arm")
            )
            out["abs"][f"{setname}/{st}"] = t.to_dicts()
            print(f"== {setname} / {st}")
            print(
                t.select(
                    "arm",
                    pl.col("crps").round(4),
                    pl.col("crps_s").round(4),
                    pl.col("mae").round(4),
                    pl.col("cov90").round(3),
                    pl.col("pin95").round(4),
                    pl.col("pin99").round(4),
                    "n",
                )
            )
    contrasts = [
        ("J2", "E0"),
        ("J1", "E0"),
        ("J2", "J1"),
        ("J2", "C0"),
        ("J1", "C0"),
        ("J2", "Bhaz"),
        ("J2", "R0-168"),
        ("J2", "K1"),
        ("E0", "Bhaz"),
    ]
    for setname, st, metric in (
        ("fam", "inforce", "crps_s"),
        ("fam", "inforce", "crps"),
        ("fam", "near", "crps"),
        ("fam", "ended24", "crps"),
        ("fam", "free", "crps"),
        ("fam", "free14", "crps"),
        ("fam", "just_ended", "crps"),
        ("held", "inforce", "crps_s"),
    ):
        c = cells(agg=agg, setname=setname, stratum=st)
        for x, y in contrasts:
            mean, lo, hi = boot(
                a=month_values(c=c, arm=x, metric=metric), b=month_values(c=c, arm=y, metric=metric)
            )
            out["contrast"][f"{setname}/{st}/{metric}/{x}-{y}"] = [mean, lo, hi]
    for st in (
        "free",
        "free14",
    ):  # NRA non-inferiority vs K1 on the injected bank (against the unshifted base) and real event-free rows
        c = cells(agg=agg, setname="fam", stratum=st)
        for x in ("C0_nra", "J1_nra", "J2_nra"):
            out["contrast"][f"fam/{st}/crps/{x}-K1_nra"] = list(
                boot(
                    a=month_values(c=c, arm=x, metric="crps"),
                    b=month_values(c=c, arm="K1_nra", metric="crps"),
                )
            )
    c = cells(agg=agg, setname="real", stratum="real_free")
    for x in ("C0", "J1", "J2", "E0"):
        out["contrast"][f"real/real_free/crps/{x}-K1"] = list(
            boot(
                a=month_values(c=c, arm=x, metric="crps"),
                b=month_values(c=c, arm="K1", metric="crps"),
            )
        )
        out["contrast"][f"real/real_free/mae/{x}-K1"] = list(
            boot(
                a=month_values(c=c, arm=x, metric="mae"),
                b=month_values(c=c, arm="K1", metric="mae"),
            )
        )
    for k, v in out["contrast"].items():
        print(f"{k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]")
    (SC / "tables.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
