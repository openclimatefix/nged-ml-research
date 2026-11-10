# ruff: noqa: E501, D103, PLR0912, PLR0915, F841, PLW2901
"""Experiment 691, step 3: print every number the write-up quotes (CPU).

This is experiment 687's `e00687_report.py` with the arms cut to `s.ARMS` and the contrasts of FL1 against
B0, L1, and F1 (and against the better of L1 and F1 at the start of a shift).
"""

import e00687_report as r687  # ty: ignore[unresolved-import]
import e00691_score as s
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)


per_month = r687.per_month
boot = r687.boot


def main() -> None:
    agg = pl.read_parquet(s.OUT / "agg.parquet")
    arms = ["K1", "R0-168", "R652", *s.ARMS]
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
    rng = np.random.default_rng(1)

    def paired(*, a: str, b: str) -> None:
        xa = r.filter(pl.col("arm") == a).sort("month", "seed", "pass", "prim")
        xb = r.filter(pl.col("arm") == b).sort("month", "seed", "pass", "prim")
        assert xa.height == xb.height
        key = (xb["month"] + "_" + xb["pass"].cast(str) + "_" + xb["prim"].cast(str)).to_numpy()
        _u, inv = np.unique(key, return_inverse=True)
        for col in ("start_days", "end_days"):
            dlt = xa[col].to_numpy() - xb[col].to_numpy()
            ev = np.bincount(inv, weights=dlt) / np.bincount(inv)
            bs = ev[rng.integers(0, len(ev), (4000, len(ev)))].mean(axis=1)
            print(f"{a} minus {b}, {col} (days, event-block 95%): mean {ev.mean():+.2f} [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}], median {np.median(ev):+.2f}, n_events {len(ev)}")  # fmt: skip

    for a, b in (
        ("FL1", "B0"),
        ("F1", "B0"),
        ("L1", "B0"),
        ("FL1", "F1"),
        ("FL1", "L1"),
        ("F1", "L1"),
    ):
        paired(a=a, b=b)
    means = r.group_by("arm").agg(pl.col("start_days").mean()).to_dict(as_series=False)
    best = min(("F1", "L1"), key=lambda a: means["start_days"][means["arm"].index(a)])
    print("better single arm at the start (lower mean start_days):", best)
    paired(a="FL1", b=best)
    for length in (5, 10, 21, 45):
        rr = r.filter(pl.col("length_days").round(0) == length)
        if rr.height:
            for a, b in (("FL1", "F1"), ("FL1", "L1")):
                da = (
                    rr.filter(pl.col("arm") == a)
                    .sort("month", "seed", "pass", "prim")["start_days"]
                    .to_numpy()
                )
                db = (
                    rr.filter(pl.col("arm") == b)
                    .sort("month", "seed", "pass", "prim")["start_days"]
                    .to_numpy()
                )
                print(f"length {length} d: {a} minus {b} start mean {np.mean(da - db):+.2f} (n {len(da)})")  # fmt: skip
    # checks: median minus mean forecast, and mean absolute error of the mean forecast, in force (fam), by arm
    import e00650_common as c650  # ty: ignore[unresolved-import]
    import e00650_fit as f650  # ty: ignore[unresolved-import]
    import e00652_fit as f652  # ty: ignore[unresolved-import]
    import e00687_fit as f687
    import e00691_fit as f691

    tau = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
    acc: dict[str, list] = {}
    for m in c650.MONTHS:
        z = f650.sources(month=m, bank="new")["bank"]
        f652.with_state(z=z, month=m)
        for a in ("B0", "L1", "F1", "FL1"):
            f = np.load((f691.OUT if a == "FL1" else f687.OUT) / f"{a}_{m}_s0.npz")
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
