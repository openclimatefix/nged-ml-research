# ruff: noqa: E501, D103, PLR0912, PLR0915, F841, RUF059, PLW2901
"""Experiment 687, step 4: print every number the write-up quotes (CPU)."""

import e00687_score as s
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)


def per_month(
    *, agg: pl.DataFrame, set_: str, stratum: str, col: str, band: str | None = None
) -> pl.DataFrame:
    d = agg.filter((pl.col("set") == set_) & (pl.col("stratum") == stratum))
    if band:
        d = d.filter(pl.col("band") == band)
    d = d.filter(pl.col("band") != "other")
    return d.group_by("arm", "seed", "month").agg((pl.col(col).sum() / pl.col("n").sum()).alias("v")).group_by("arm", "month").agg(pl.col("v").mean()).sort("arm", "month")  # fmt: skip


def boot(*, v: np.ndarray, n: int = 20000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    m = rng.integers(0, len(v), (n, len(v)))
    d = v[m].mean(axis=1)
    return float(v.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def main() -> None:
    agg = pl.read_parquet(s.OUT / "agg.parquet")
    arms = [
        "K1",
        "R0-168",
        "R652",
        "B0",
        "B0x",
        "B0g",
        "L1",
        "F1",
        "F1d46",
        "F2",
        "P",
        "F1h4",
        "F1nd",
    ]
    for set_, strata in (
        ("fam", ["inforce", "near", "just_ended", "free"]),
        ("held", ["inforce"]),
        ("real", ["real_free"]),
    ):
        for st in strata:
            for col in ("crps_s", "crps"):
                pm = per_month(agg=agg, set_=set_, stratum=st, col=col)
                wide = pm.pivot(on="month", index="arm", values="v")
                means = pm.group_by("arm").agg(pl.col("v").mean().alias("mean")).sort("arm")
                ref = pm.filter(pl.col("arm") == "B0").sort("month")["v"].to_numpy()
                out = []
                for a in arms:
                    x = pm.filter(pl.col("arm") == a).sort("month")["v"].to_numpy()
                    if len(x) == 5:
                        mu, lo, hi = boot(v=x - ref)
                        out.append((a, round(float(x.mean()), 2), round(mu, 2), round(lo, 2), round(hi, 2)))  # fmt: skip
                print(
                    f"\n== {set_} {st} {col} (points of capacity; contrast = arm minus B0, month-block 95%)"
                )
                print(
                    pl.DataFrame(out, schema=["arm", "score", "minus_B0", "lo", "hi"], orient="row")
                )
    print(
        "\n== false-alarm rate on event-free rows (share with |head - K1 median| >= 3 points), bank and real"
    )
    for set_, st in (("fam", "free"), ("real", "real_free")):
        d = agg.filter(
            (pl.col("set") == set_) & (pl.col("stratum") == st) & (pl.col("band") != "other")
        )
        print(
            set_,
            d.group_by("arm")
            .agg((pl.col("false_alarm").sum() / pl.col("n").sum()).alias("rate"))
            .sort("arm"),
        )
    print("\n== median absolute error and 90% coverage by band, in force (fam)")
    d = agg.filter(
        (pl.col("set") == "fam") & (pl.col("stratum") == "inforce") & (pl.col("band") != "other")
    )
    print(d.group_by("arm", "band").agg((pl.col("ae_med").sum() / pl.col("n").sum()).alias("mae_pts"), (pl.col("crps").sum() / pl.col("n").sum()).alias("crps"), (pl.col("cov90").sum() / pl.col("n").sum()).alias("cov90")).sort("band", "arm"))  # fmt: skip
    print("\n== tails in force (fam, unscaled)")
    print(d.group_by("arm").agg(*[(pl.col(c).sum() / pl.col("n").sum()).alias(c) for c in ("pin01", "pin05", "pin95", "pin99")], (pl.col("cov90").sum() / pl.col("n").sum()).alias("cov90")).sort("arm"))  # fmt: skip
    r = pl.read_parquet(s.OUT / "reaction.parquet")
    print(
        "\n== reaction time (held-out passes): events", r.filter(pl.col("arm") == "B0").height // 2
    )
    print(r.group_by("arm").agg(pl.col("start_days").median().alias("start_med"), pl.col("start_days").mean().alias("start_mean"), pl.col("start_reacted").mean().alias("start_reacted"), pl.col("end_days").median().alias("end_med"), pl.col("end_days").mean().alias("end_mean"), pl.col("end_released").mean().alias("end_released")).sort("arm"))  # fmt: skip
    base = r.filter(pl.col("arm") == "B0").sort("month", "seed", "pass", "prim")
    rng = np.random.default_rng(1)
    for a in ("F1", "F1d46", "F2", "F1h4", "L1", "P", "B0g", "B0x", "F1nd"):
        x = r.filter(pl.col("arm") == a).sort("month", "seed", "pass", "prim")
        assert x.height == base.height
        for col in ("start_days", "end_days"):
            dlt = x[col].to_numpy() - base[col].to_numpy()
            key = (
                base["month"] + "_" + base["pass"].cast(str) + "_" + base["prim"].cast(str)
            ).to_numpy()
            u, inv = np.unique(key, return_inverse=True)
            ev = np.bincount(inv, weights=dlt) / np.bincount(inv)
            mm = rng.integers(0, len(ev), (4000, len(ev)))
            bs = ev[mm].mean(axis=1)
            print(f"{a} minus B0, {col} (days, event-block 95%): mean {ev.mean():+.2f} [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}], median {np.median(ev):+.2f}, n_events {len(ev)}")  # fmt: skip
    lref = r.filter(pl.col("arm") == "L1").sort("month", "seed", "pass", "prim")
    for a in ("F1", "F1d46", "F2"):
        x = r.filter(pl.col("arm") == a).sort("month", "seed", "pass", "prim")
        for col in ("start_days", "end_days"):
            dlt = x[col].to_numpy() - lref[col].to_numpy()
            key = (
                lref["month"] + "_" + lref["pass"].cast(str) + "_" + lref["prim"].cast(str)
            ).to_numpy()
            u, inv = np.unique(key, return_inverse=True)
            ev = np.bincount(inv, weights=dlt) / np.bincount(inv)
            bs = ev[rng.integers(0, len(ev), (4000, len(ev)))].mean(axis=1)
            print(f"{a} minus L1, {col}: mean {ev.mean():+.2f} [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}]")  # fmt: skip
    # checks: median minus mean forecast, and mean absolute error of the mean forecast, in force (fam), by arm
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00650_fit as f650  # ty: ignore[unresolved-import]
    import e00652_fit as f652  # ty: ignore[unresolved-import]
    import e00687_fit as f687

    tau = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
    acc: dict[str, list] = {}
    for m in c650.MONTHS:
        z = f650.sources(month=m, bank="new")["bank"]
        f652.with_state(z=z, month=m)
        for a in ("B0", "L1", "F1"):
            f = np.load(f687.OUT / f"{a}_{m}_s0.npz")
            idx = f["fam_idx"]
            q = np.sort(f["fam_q"], axis=1)
            mean = (
                q[:, 0] * 0.01
                + q[:, -1] * 0.01
                + (0.5 * (q[:, 1:] + q[:, :-1]) * np.diff(tau)[None, :]).sum(1)
            )
            sel = np.isin(z["stratum"][idx], ["far", "middle", "near"]) & np.isfinite(z["y"][idx])
            y = (z["y"][idx] + z["shift"][idx])[sel]
            acc.setdefault(a, []).append(
                (
                    (q[sel, 5] - mean[sel]).mean() * 100,
                    np.abs(y - mean[sel]).mean() * 100,
                    np.abs(y - q[sel, 5]).mean() * 100,
                )
            )
    for a, v in acc.items():
        v = np.array(v)
        print(
            f"checks {a}: median minus mean {v[:, 0].mean():+.2f}, MAE of mean {v[:, 1].mean():.2f}, MAE of median {v[:, 2].mean():.2f} (in force, points of capacity)"
        )
    c = pl.read_csv(s.OUT / "cost.csv")
    print("\n== avoidable-curtailment cost (units of experiment 601, per Primary-day)")
    print(
        c.group_by("set", "arm")
        .agg(pl.col("cost").mean().round(0))
        .sort("set", "arm")
        .pivot(on="set", index="arm", values="cost")
    )


if __name__ == "__main__":
    main()
