# ruff: noqa: ANN001, D103, E501, PLR0912, PLR0915, ANN201, C901
"""Experiment 663, step 6: contrasts with month-block intervals, absolute tables, and the veto (CPU).

Reads `score/agg.parquet`. A contrast is the difference between two arms in one metric over lead
days 1 to 9 of a stratum (rows weighted by their count), per scored month, with seeds paired and
averaged within a month. The interval is a percentile bootstrap over the 5 development months.

Writes `score/tables.json`.

Usage: python e00663_report.py
"""

import json

import e00663_common as c663
import numpy as np
import polars as pl

AGG = c663.WORK / "score" / "agg.parquet"
CONTRASTS = (
    ("E0d", "E0"), ("J2d", "J2"), ("E0d", "E0d_rw0"), ("E0d", "E0d_rw1"), ("E0d", "E0d_rw2"), ("E0d", "E0d_noneigh"),
    ("J2d", "J2d_rw0"), ("J2d", "J2d_rw1"), ("J2d", "J2d_rw2"), ("J2d", "J2d_noneigh"), ("J2", "E0"),
)  # fmt: skip
STRATA = ("far", "middle", "near", "ended24", "just_ended", "free", "free_far14", "inforce", "all", "onset_ahead")
METRICS = ("crps_s", "crps", "nmae", "cov90", "cov98", "pin01", "pin05", "pin95", "pin99", "dep", "dep_crps", "dep_crps_nra", "crps_nra")


def month_means(*, a: pl.DataFrame, arm: str, stratum: str, band: str, metric: str, setname: str = "fam") -> dict[tuple[str, int], float]:
    f = a.filter((pl.col("arm") == arm) & (pl.col("stratum") == stratum) & (pl.col("set") == setname) & pl.col("lead_day").is_between(1, 9))
    if band != "all":
        f = f.filter(pl.col("band") == band)
    g = f.group_by("month", "seed").agg(pl.col(metric).sum(), pl.col("n").sum())
    return {(r["month"], r["seed"]): r[metric] / r["n"] for r in g.iter_rows(named=True) if r["n"] > 0}


def seed_mean(d: dict[tuple[str, int], float]) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for (m, _s), v in d.items():
        out.setdefault(m, []).append(v)
    return {m: float(np.mean(v)) for m, v in out.items()}


def interval(per_month: dict[str, float], *, n: int = 20000, seed: int = 663) -> dict:
    v = np.array([per_month[m] for m in sorted(per_month)])
    if len(v) == 0:
        return {}
    pick = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    boot = v[pick].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)), "min": float(v.min()), "max": float(v.max()), "months": len(v)}


def main() -> None:
    a = pl.read_parquet(AGG)
    arms = sorted(a["arm"].unique().to_list())
    out: dict = {"absolute": {}, "contrasts": {}, "counts": {}}
    for band in ("all", "short", "long"):
        for st in STRATA:
            for arm in arms:
                for met in METRICS:
                    pm = seed_mean(month_means(a=a, arm=arm, stratum=st, band=band, metric=met))
                    if pm:
                        out["absolute"][f"{band}|{st}|{arm}|{met}"] = interval(pm)
    for band in ("all", "short", "long"):
        for st in STRATA:
            cnt = a.filter((pl.col("arm") == "K1") & (pl.col("stratum") == st) & (pl.col("set") == "fam") & pl.col("lead_day").is_between(1, 9))
            if band != "all":
                cnt = cnt.filter(pl.col("band") == band)
            out["counts"][f"{band}|{st}"] = int(cnt["n"].sum())
    for band in ("all", "short", "long"):
        for st in STRATA:
            for x, y in CONTRASTS:
                if x not in arms or y not in arms:
                    continue
                for met in ("crps_s", "crps", "nmae", "dep"):
                    dx = month_means(a=a, arm=x, stratum=st, band=band, metric=met)
                    dy = month_means(a=a, arm=y, stratum=st, band=band, metric=met)
                    diff = {k: dx[k] - dy[k] for k in dx if k in dy}
                    pm = seed_mean(diff)
                    if pm:
                        out["contrasts"][f"{band}|{st}|{x} minus {y}|{met}"] = interval(pm)
    # real rows (no injected shift): the false-alarm share and CRPS on event-free real rows
    for arm in arms:
        for met in ("crps", "nmae", "dep", "dep_crps", "dep_crps_nra", "cov90"):
            pm = seed_mean(month_means(a=a, arm=arm, stratum="real_free", band="real", metric=met, setname="real"))
            if pm:
                out["absolute"][f"real|real_free|{arm}|{met}"] = interval(pm)
    (c663.WORK / "score" / "tables.json").write_text(json.dumps(out, indent=1))
    print(len(out["absolute"]), "absolute cells,", len(out["contrasts"]), "contrasts")


if __name__ == "__main__":
    main()
