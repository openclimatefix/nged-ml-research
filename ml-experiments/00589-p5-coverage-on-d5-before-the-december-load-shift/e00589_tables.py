# ruff: noqa: E501, D103, ANN001, ANN201, PLR2004
"""Experiment 589, stages 1 and 2 (no fit): coverage of the observed power by C0, P5, and C1.

Reads the stored `row_arrays.npz` of experiment 584's `score_c0`, `score_p5`, and `score_c1` (the
seed-mean quantiles of experiments 572, 574, and 584, at 0-120 h leads, observed power from cleaned
data, `free` = no labelled or verified event within 7 days). The three files list the same rows in a
different order, so they are joined on (Primary, origin, valid time).

Writes CSV tables to `data/ml-experiments/00589-*/tables/` and prints every number the page quotes.

Usage: DATA_PATH_INTERNAL=... python e00589_tables.py
"""

from pathlib import Path

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_common as k582  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SRC = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6"
OUT = ML_EXPERIMENTS_DATA_DIR / "00589-p5-coverage-on-d5-before-the-december-load-shift/tables"
ARMS = {"C0": "c0", "P5": "p5", "C1": "c1"}
IQ = {0.01: 0, 0.05: 1, 0.5: 5, 0.95: 9, 0.99: 10}
KEYS = ["prim", "origin_idx", "idx"]
RNG = np.random.default_rng(589)


def load() -> pl.DataFrame:
    """Return one row per (Primary, origin, valid time) with every arm's q01, q05, q50, q95, q99."""
    sites = k578.sites()
    lab = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    cap = dict(zip(sites["primary"].to_list(), sites["capacity_mw"].to_list(), strict=True))
    grp = dict(zip(sites["primary"].to_list(), sites["group"].to_list(), strict=True))
    base = None
    for arm, folder in ARMS.items():
        z = np.load(SRC / f"score_{folder}/row_arrays.npz", allow_pickle=True)
        cols = {
            "prim": z["prim"],
            "origin_idx": z["origin_idx"],
            "idx": z["idx"],
            "lead": z["lead"],
            "obs": z["obs"],
            "nra": z["nra"],
            "free": z["free"],
            "month": z["month"].astype(str),
        }
        cols["M"] = z["M"][:, IQ[0.5]]  # lags-masked twin's median
        for q, i in IQ.items():
            cols[f"{arm}_q{round(q * 100):02d}"] = z["V"][:, i]
        df = pl.DataFrame(cols)
        if base is None:
            base = df.rename({"M": f"{arm}_M"})
        else:
            keep = [c for c in df.columns if c.startswith(arm)] + ["M", *KEYS]
            base = base.join(
                df.select(keep).rename({"M": f"{arm}_M"}), on=KEYS, how="inner", validate="1:1"
            )
    assert base is not None
    return base.with_columns(
        label=pl.col("prim").replace_strict(lab, return_dtype=pl.String),
        group=pl.col("prim").replace_strict(grp, return_dtype=pl.String),
        cap=pl.col("prim").replace_strict(cap, return_dtype=pl.Float64),
        band=pl.when(pl.col("lead") < 24).then(pl.lit("0-24 h")).otherwise(pl.lit("24-120 h")),
    )


def stats(*, df: pl.DataFrame, by: list[str], y: str = "obs") -> pl.DataFrame:
    """Per group and arm: n, coverage, below q05, above q95, median minus y, width (% of capacity)."""
    parts = []
    for arm in ARMS:
        d = df.group_by(by).agg(
            pl.len().alias("n"),
            ((pl.col(y) >= pl.col(f"{arm}_q05")) & (pl.col(y) <= pl.col(f"{arm}_q95")))
            .mean()
            .alias("cover90"),
            (pl.col(y) < pl.col(f"{arm}_q05")).mean().alias("below_q05"),
            (pl.col(y) > pl.col(f"{arm}_q95")).mean().alias("above_q95"),
            ((pl.col(f"{arm}_q50") - pl.col(y)) * 100).mean().alias("median_minus_y_pct"),
            ((pl.col(f"{arm}_q95") - pl.col(f"{arm}_q05")) * 100).mean().alias("width90_pct"),
            ((pl.col(f"{arm}_q99") - pl.col(f"{arm}_q01")) * 100).mean().alias("width98_pct"),
        )
        parts.append(d.with_columns(arm=pl.lit(arm)))
    return pl.concat(parts).sort(by + ["arm"])


def month_block_ci(*, df: pl.DataFrame, arm: str, n: int = 4000) -> tuple[float, float, float]:
    """Pooled coverage with a 95% interval from resampling the five fold months."""
    ok = ((df["obs"] >= df[f"{arm}_q05"]) & (df["obs"] <= df[f"{arm}_q95"])).cast(pl.Float64)
    t = df.with_columns(ok=ok).group_by("month").agg(pl.col("ok").sum().alias("s"), pl.len().alias("n"))
    s, m = t["s"].to_numpy(), t["n"].to_numpy()
    idx = RNG.integers(0, len(s), size=(n, len(s)))
    boot = s[idx].sum(1) / m[idx].sum(1)
    return float(s.sum() / m.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    df = load()
    print("rows", df.height, "event-free rows", int(df["free"].sum()))
    free = df.filter(pl.col("free") & (pl.col("band") == "24-120 h"))

    # Stage 1: event-free coverage by Primary, month, lead band, group, with pooled intervals.
    by_p = stats(df=free, by=["label", "group"])
    by_p.write_csv(OUT / "event_free_by_primary.csv")
    wide = by_p.pivot(on="arm", index=["label", "group", "n"], values="cover90").with_columns(
        p5_minus_c0=pl.col("P5") - pl.col("C0"), c1_minus_c0=pl.col("C1") - pl.col("C0")
    )
    print("\n== event-free 24-120 h coverage of observed power, by Primary ==\n", wide.sort("label"))
    stats(df=free, by=["group"]).write_csv(OUT / "event_free_by_group.csv")
    print("\n== by group ==\n", stats(df=free, by=["group"]))
    for g, d in free.group_by("group"):
        for arm in ARMS:
            print("pooled", g[0], arm, [round(v, 3) for v in month_block_ci(df=d, arm=arm)])
    for arm in ARMS:
        print("pooled all 16", arm, [round(v, 3) for v in month_block_ci(df=free, arm=arm)])
    stats(df=free, by=["month"]).write_csv(OUT / "event_free_by_month.csv")
    print("\n== by fold month ==\n", stats(df=free, by=["month"]).pivot(on="arm", index="month", values="cover90"))
    stats(df=df.filter(pl.col("free")), by=["band"]).write_csv(OUT / "event_free_by_band.csv")
    print("\n== by lead band ==\n", stats(df=df.filter(pl.col("free")), by=["band"]))
    pm = stats(df=free, by=["label", "month"]).pivot(on="arm", index=["label", "month"], values="cover90")
    pm = pm.with_columns(p5_minus_c0=pl.col("P5") - pl.col("C0")).sort("label", "month")
    pm.write_csv(OUT / "event_free_by_primary_month.csv")
    print("\n== Primary-month cells where P5 trails C0 by more than 0.15 ==\n", pm.filter(pl.col("p5_minus_c0") < -0.15))
    print("n Primary-month cells", pm.height, "P5 trails by >0.15:", pm.filter(pl.col("p5_minus_c0") < -0.15).height,
          "C1 trails by >0.15:", pm.with_columns(d=pl.col("C1") - pl.col("C0")).filter(pl.col("d") < -0.15).height)

    # Stage 2: D5 in the 2025-12 fold, by phase of the verified event.
    ev = sc.merged_events().filter(pl.col("label") == "D5")
    ev = ev.with_columns(date=pl.Series([k582.date_of(i) for i in ev["onset_idx"].to_list()]))
    print("\nD5 verified spans\n", ev)
    dec = ev.filter(pl.col("date") == "2025-12-29")
    onset, end = int(dec["onset_idx"][0]), int(dec["end_idx"][0])
    d5 = df.filter(
        (pl.col("label") == "D5") & (pl.col("band") == "24-120 h") & (pl.col("month") == "2025-12")
    )
    phase = (
        pl.when(pl.col("idx") < onset - 336).then(pl.lit("earlier"))
        .when(pl.col("idx") < onset).then(pl.lit("before (7 d)"))
        .when(pl.col("idx") < end).then(pl.lit("during"))
        .when(pl.col("idx") < end + 336).then(pl.lit("after0-7"))
        .otherwise(pl.lit("later"))
    )
    d5 = d5.with_columns(phase=phase)
    ph = stats(df=d5, by=["phase"])
    ph.write_csv(OUT / "d5_by_phase.csv")
    print("\n== D5, 24-120 h, observed power vs bands by phase ==\n", ph)
    nr = stats(df=d5, by=["phase"], y="nra")
    print("\n== D5 same, against the NRA ==\n", nr.filter(pl.col("phase").is_in(["before (7 d)", "during", "after0-7"])))

    # Stage 2b: is the December miss specific to D5? All Primaries, 2025-12 fold, valid time before D5's onset.
    dec_all = df.filter((pl.col("month") == "2025-12") & (pl.col("band") == "24-120 h") & (pl.col("idx") < onset))
    pdec = stats(df=dec_all, by=["label"])
    pdec.write_csv(OUT / "december_before_d5_onset_by_primary.csv")
    wide = pdec.pivot(on="arm", index=["label", "n"], values="cover90").join(
        pdec.pivot(on="arm", index="label", values="median_minus_y_pct").rename({a: f"{a}_med_minus_obs_pct" for a in ARMS}),
        on="label",
    ).sort("C0")
    print("\n== 2025-12 fold, valid time before D5's onset (event-free or not), by Primary ==\n", wide)
    print("\nPrimaries with P5 coverage below 0.70 in that window:", wide.filter(pl.col("P5") < 0.7)["label"].to_list())
    # D5 observed daily level from the cleaned data, November and December, to see when the level rose.
    clean = k578.clean()
    d5i = int(dec["primary"][0])
    day0 = (onset // 48) - 60
    lv = [(d, float(np.nanmean(clean[d5i, d * 48 : (d + 1) * 48])) * 100) for d in range(day0, onset // 48 + 1)]
    print("\nD5 daily mean observed power, % of capacity, 60 days up to onset:", [(k582.date_of(d * 48)[5:], round(v, 1)) for d, v in lv])
    # Daily series of D5 in the fold: observed, median of each arm, lags-masked median.
    daily = (
        df.filter((pl.col("label") == "D5") & (pl.col("lead") >= 24) & (pl.col("lead") < 48))
        .with_columns(day=(pl.col("idx") // 48))
        .group_by("day").agg(
            pl.col("obs").mean(), pl.col("nra").mean(),
            *[pl.col(f"{a}_q50").mean() for a in ARMS], *[pl.col(f"{a}_M").mean() for a in ARMS],
            *[pl.col(f"{a}_q05").mean() for a in ARMS], *[pl.col(f"{a}_q95").mean() for a in ARMS],
        ).sort("day")
    )
    daily.write_csv(OUT / "d5_daily.csv")
    print("onset day", onset // 48, "end day", end // 48)
    print(daily.filter(pl.col("day").is_between(onset // 48 - 30, end // 48 + 8)).with_columns((pl.col(c) * 100).round(1) for c in daily.columns if c != "day"))


if __name__ == "__main__":
    main()
