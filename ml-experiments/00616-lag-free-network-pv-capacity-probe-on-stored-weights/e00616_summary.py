# ruff: noqa: E501
"""Experiment 616: summarise the capacity draws into the tables, checks, and comparisons of the write-up.

Reads `draws_<fold>.parquet` (from `e00616_analyse.py`) and the probe tables and responses, and writes
`summary.json` and `tables.parquet` to the private data folder. Prints the numbers quoted in the page.
"""

import json
from datetime import date

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00616_analyse as an
import e00616_regress as reg
import numpy as np
import polars as pl
from scipy.stats import spearmanr

OUT = reg.OUT
FOLDS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
D474 = ML = None
R474 = (
    OUT.parent
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
    / "nged_table.parquet"
)
R479 = OUT.parent / "00479-pv-capacity-anomaly-regression-rung" / "work" / "capacity_table.parquet"


def site_of() -> dict[int, str]:
    """Return the Primary index to label map."""
    t = prep.primaries()
    return dict(zip(t["primary"].to_list(), t["site"].to_list(), strict=True))


def pooled(*, draws: pl.DataFrame, dates: tuple[date, date] | None) -> pl.DataFrame:
    """Return the capacity pooled over the probe dates of a set (mean over dates), by draw."""
    d = draws
    if dates is not None:
        d = d.filter(pl.col("date").is_between(dates[0], dates[1]))
    return d.group_by(["fold", "arm", "variant", "primary", "boot", "seed"]).agg(
        capacity_mw=pl.col("capacity_mw").mean(), n_dates=pl.col("date").n_unique()
    )


def summarise(*, p: pl.DataFrame) -> pl.DataFrame:
    """Return the point value, seed and bootstrap spread, and 10/50/90 percentiles by fold and Primary."""
    point = p.filter(pl.col("boot") == 0)
    boots = p.filter(pl.col("boot") > 0)
    keys = ["fold", "arm", "variant", "primary"]
    by_seed = point.group_by(keys).agg(
        mean=pl.col("capacity_mw").mean(),
        seed_sd=pl.col("capacity_mw").std(),
        seed_p10=pl.col("capacity_mw").quantile(0.1),
        seed_p90=pl.col("capacity_mw").quantile(0.9),
        n_dates=pl.col("n_dates").first(),
    )
    boot_sd = (
        boots.group_by([*keys, "seed"])
        .agg(sd=pl.col("capacity_mw").std())
        .group_by(keys)
        .agg(boot_sd=pl.col("sd").mean())
    )
    total = boots.group_by(keys).agg(
        p10=pl.col("capacity_mw").quantile(0.1),
        p50=pl.col("capacity_mw").quantile(0.5),
        p90=pl.col("capacity_mw").quantile(0.9),
    )
    return by_seed.join(boot_sd, on=keys).join(total, on=keys).sort(keys)


def profile_and_monotone(*, month: str) -> pl.DataFrame:
    """Return hour-profile correlation and monotonicity of the seed-mean response, per arm and Primary."""
    rows = pl.read_parquet(OUT / f"probe_{month}.parquet").with_row_index("global_row")
    y_all = np.load(OUT / f"probe_{month}_y.npy")
    table = reg.unit_table()
    rows = an.group_table(rows=rows, table=table)
    out = []
    low, high = date(2025, 7, 1), date(2026, 6, 30)
    for a, arm in enumerate(an.ARMS):
        ym = y_all[a].mean(0)
        for (primary,), g in rows.filter(pl.col("date").is_between(low, high)).group_by(
            ["primary"], maintain_order=True
        ):
            idx = g["global_row"].to_numpy()
            y = ym[idx].astype(np.float64)
            u = g["u"].to_numpy()
            slot = g["slot"].to_numpy()
            date_ids = g["date"].to_numpy()
            corrs, ys, us = [], [], []
            for d in np.unique(date_ids):
                m = date_ids == d
                dy, du = [], []
                for s in np.unique(slot[m]):
                    ms = m & (slot == s)
                    if ms.sum() < 30:
                        continue
                    q1, q2 = np.quantile(u[ms], [1 / 3, 2 / 3])
                    top, bot = ms & (u >= q2), ms & (u <= q1)
                    dy.append(y[bot].mean() - y[top].mean())
                    du.append(u[top].mean() - u[bot].mean())
                corrs.append(float(np.corrcoef(dy, du)[0, 1]))
                # slot-demeaned pooled response for the monotonicity check
                key = date_ids.astype(str) + "_" + slot.astype(str)
                _, k = np.unique(key, return_inverse=True)
                n = np.bincount(k)
                ys = y - (np.bincount(k, weights=y) / n)[k]
                us = u - (np.bincount(k, weights=u) / n)[k]
            edges = np.quantile(us, np.linspace(0, 1, 9))
            which = np.clip(np.searchsorted(edges, us, side="right") - 1, 0, 7)
            means = np.array([ys[which == b].mean() for b in range(8)])
            drop = means[0] - means[-1]
            rises = np.diff(means).max()
            rho = float(spearmanr(np.arange(8), means).statistic)
            out.append(
                {
                    "fold": month,
                    "arm": arm,
                    "primary": primary,
                    "profile_corr_median": float(np.median(corrs)),
                    "profile_corr_min": float(np.min(corrs)),
                    "spearman": rho,
                    "total_drop_mw": float(drop),
                    "max_rise_share": float(max(rises, 0.0) / max(abs(drop), 1e-9)),
                    "monotone": bool(rho <= -0.95 and rises <= 0.1 * abs(drop)),
                }
            )
    return pl.DataFrame(out)


def references() -> pl.DataFrame:
    """Return experiment 474's rung R4 and experiment 479's capacity (both with cloud terms)."""
    r4 = pl.read_parquet(R474).filter(pl.col("rung").str.contains("cloudterm"))
    r4 = r4.select(
        site="site", season_year="season_year", r4_p10="ac_p10", r4_p50="ac_p50", r4_p90="ac_p90"
    )
    r479 = pl.read_parquet(R479).filter(pl.col("cloud") & (pl.col("prior") == "none"))
    r479 = r479.select("site", r479_p05="p05", r479_p50="p50", r479_p95="p95")
    return r4.join(r479, on="site", how="left")


def main() -> None:
    """Write the summary tables and print the headline numbers."""
    names = site_of()
    draws = pl.concat(
        [
            pl.read_parquet(OUT / f"draws_{m}.parquet")
            for m in FOLDS
            if (OUT / f"draws_{m}.parquet").exists()
        ]
    )
    all_sets = {"all_dates": None, **an.DATE_SETS}
    tables = []
    for label, dates in all_sets.items():
        s = summarise(p=pooled(draws=draws, dates=dates)).with_columns(date_set=pl.lit(label))
        tables.append(s)
    t = pl.concat(tables).with_columns(
        site=pl.col("primary").replace_strict(names, return_dtype=pl.String)
    )
    t.write_parquet(OUT / "tables.parquet")
    per_date = (
        draws.filter((pl.col("boot") == 0) & (pl.col("variant") == "T"))
        .group_by(["fold", "arm", "primary", "date"])
        .agg(capacity_mw=pl.col("capacity_mw").mean())
        .with_columns(site=pl.col("primary").replace_strict(names, return_dtype=pl.String))
        .sort("fold", "arm", "site", "date")
    )
    per_date.write_parquet(OUT / "per_date.parquet")
    pm = pl.concat(
        [profile_and_monotone(month=m) for m in FOLDS if (OUT / f"probe_{m}.parquet").exists()]
    ).with_columns(site=pl.col("primary").replace_strict(names, return_dtype=pl.String))
    pm.write_parquet(OUT / "profile_monotone.parquet")
    refs = references()
    refs.write_parquet(OUT / "references.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    head = t.filter(
        (pl.col("fold") == "2026-06")
        & (pl.col("arm") == "D5")
        & (pl.col("variant") == "T")
        & (pl.col("date_set") == "SY2025_dates")
    )
    print(head.select("site", "mean", "seed_sd", "boot_sd", "p10", "p50", "p90", "n_dates"))
    print(pm.filter(pl.col("fold") == "2026-06"))
    print(refs.filter(pl.col("season_year") == 2025))
    (OUT / "summary.json").write_text(json.dumps({"folds": list(FOLDS)}))


if __name__ == "__main__":
    main()
