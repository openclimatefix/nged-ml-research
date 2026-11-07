# ruff: noqa: E501
"""Print the report of experiment 631 and apply the registered decision rule to `work/scores.parquet`."""

import e00631_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BOOT = 20000
BAND = "0-240h"
G = {1: "Boston", 2: "Skegness"}


def contrast(
    *,
    df: pl.DataFrame,
    a: str,
    b: str,
    g: int,
    rowset: str,
    stat: str = "crps_month",
    side: str = "both",
) -> tuple[float, float, float]:
    """Return the relative difference (percent) of monthly means a/b - 1 with a month-block 95% interval."""

    def get(src: str) -> np.ndarray:
        d = df.filter(
            (pl.col("group") == g)
            & (pl.col("rowset") == rowset)
            & (pl.col("source") == src)
            & (pl.col("band") == BAND)
            & (pl.col("stat") == stat)
            & (pl.col("side") == side)
        ).sort("month")
        return d["value"].to_numpy()

    x, y = get(a), get(b)
    rng = np.random.default_rng(631)
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    r = x[idx].mean(1) / y[idx].mean(1) - 1
    return 100 * (x.mean() / y.mean() - 1), 100 * np.quantile(r, 0.025), 100 * np.quantile(r, 0.975)


def main() -> None:
    """Print the tables and the verdict."""
    df = pl.read_parquet(c.WORK / "scores.parquet")
    ex = df.filter((pl.col("stat") == "exceed_0.05") & (pl.col("band") == BAND))
    crps = (
        df.filter((pl.col("stat") == "crps_month") & (pl.col("band") == BAND))
        .group_by("group", "rowset", "source")
        .agg(pl.col("value").mean().alias("crps"), pl.col("n_rows").sum().alias("n"))
    )
    print(
        df.filter(pl.col("source") == "census").select(
            "group", "stat", "value", "se_nominal", "lo", "hi", "n_rows"
        )
    )
    for rowset in ("common", "full"):
        for g in c.BSP_GROUPS:
            print(
                f"\n=== {G[g]} rowset {rowset}: share of actuals below q05 (export) / above q95 (import), 0-240 h; CRPS MW ==="
            )
            e = ex.filter((pl.col("group") == g) & (pl.col("rowset") == rowset))
            exp = e.filter(pl.col("side") == "export").select(
                "source",
                pl.col("value").alias("exp"),
                pl.col("lo").alias("exp_lo"),
                pl.col("hi").alias("exp_hi"),
                "se_nominal",
                "n_days",
            )
            imp = e.filter(pl.col("side") == "import").select(
                "source",
                pl.col("value").alias("imp"),
                pl.col("lo").alias("imp_lo"),
                pl.col("hi").alias("imp_hi"),
            )
            cr = crps.filter((pl.col("group") == g) & (pl.col("rowset") == rowset)).select(
                "source", "crps", "n"
            )
            t = exp.join(imp, on="source").join(cr, on="source").sort("source")
            with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3):
                print(t)
    print("\n=== decision rule (common row set) ===")
    verdicts = {}
    for rem in ("const", "tod"):
        ok_all = {}
        for coupled in ("J3", "J5"):
            ok = True
            for g in c.BSP_GROUPS:
                s = f"{coupled}|w|{rem}"
                e = ex.filter(
                    (pl.col("group") == g)
                    & (pl.col("rowset") == "common")
                    & (pl.col("source") == s)
                )
                se = float(e["se_nominal"][0])
                exp_v = float(e.filter(pl.col("side") == "export")["value"][0])
                imp_v = float(e.filter(pl.col("side") == "import")["value"][0])
                d, lo, hi = contrast(df=df, a=s, b=f"J0|w|{rem}", g=g, rowset="common")
                calibrated = abs(exp_v - 0.05) <= 2 * se and abs(imp_v - 0.05) <= 2 * se
                better = hi < 0
                print(
                    f"{rem} {coupled} {G[g]}: export {exp_v:.3f} import {imp_v:.3f} (2 SE {2 * se:.3f}); CRPS vs J0 {d:+.2f}% [{lo:+.2f}, {hi:+.2f}]; calibrated {calibrated}; better {better}"
                )
                ok &= calibrated and better
            ok_all[coupled] = ok
        verdicts[rem] = any(ok_all.values())
    print(
        "sum usable by remainder:",
        verdicts,
        "->",
        "sum usable" if any(verdicts.values()) else "direct needed",
    )
    print("\n=== CRPS contrasts, common rows: coupled vs J0 and direct vs best sum ===")
    for g in c.BSP_GROUPS:
        for variant in ("wo", "w"):
            for rem in ("none", "tod"):
                for arm in ("J2", "J3", "J5"):
                    if arm == "J2" and variant == "w":
                        continue
                    d, lo, hi = contrast(
                        df=df,
                        a=f"{arm}|{variant}|{rem}",
                        b=f"J0|{variant}|{rem}",
                        g=g,
                        rowset="common",
                    )
                    print(
                        f"{G[g]} {arm} vs J0 [{variant},{rem}] CRPS {d:+.2f}% [{lo:+.2f}, {hi:+.2f}]"
                    )
        d, lo, hi = contrast(df=df, a="D:d622_lagged", b="J3|w|tod", g=g, rowset="common")
        print(f"{G[g]} direct (622 lagged) vs J3|w|tod CRPS {d:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
        for side in ("export", "import"):
            d, lo, hi = contrast(
                df=df,
                a="D:d622_lagged",
                b="J3|w|tod",
                g=g,
                rowset="common",
                stat="twcrps_month",
                side=side,
            )
            print(f"{G[g]} direct vs J3|w|tod twCRPS {side} {d:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
    print("\n=== bias (actual minus median, MW) and observed remainder ===")
    b = (
        df.filter(
            pl.col("stat").str.starts_with("bias_median")
            & (pl.col("rowset") == "common")
            & pl.col("source").is_in(
                ["J3|wo|none", "J3|wo|tod", "J3|w|none", "J3|w|const", "J3|w|tod", "D:d622_lagged"]
            )
        )
        .pivot(on="stat", index=["group", "source"], values="value")
        .sort("group", "source")
    )
    with pl.Config(float_precision=2):
        print(b)
        print(
            df.filter(pl.col("rowset") == "observed")
            .pivot(on="stat", index=["group"], values="value")
            .sort("group")
        )


if __name__ == "__main__":
    main()
