# ruff: noqa: E501, FURB171
"""Print the report of experiment 629 from `work/scores_<network>.parquet` and apply the decision rule."""

import argparse

import e00629_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BOOT = 20000


def contrast(
    *,
    stat: str,
    t: pl.DataFrame,
    g: int,
    a: str,
    b: str,
    side: str,
    band: str,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Relative difference (percent) of monthly mean twCRPS of source a against b, month-block interval."""
    f = t.filter(
        (pl.col("group") == g)
        & (pl.col("stat") == stat)
        & (pl.col("side") == side)
        & (pl.col("band") == band)
    )
    va = f.filter(pl.col("source") == a).sort("month")["value"].to_numpy()
    vb = f.filter(pl.col("source") == b).sort("month")["value"].to_numpy()
    if va.mean() == 0 or vb.mean() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(va), size=(BOOT, len(va)))
    r = va[idx].mean(1) / vb[idx].mean(1) - 1
    return (
        100 * (va.mean() / vb.mean() - 1),
        100 * np.nanquantile(r, 0.025),
        100 * np.nanquantile(r, 0.975),
    )


def main() -> None:
    """Print tables and decisions."""
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="K1")
    args = p.parse_args()
    t = pl.read_parquet(c.WORK / f"scores_{args.network}.parquet")
    rng = np.random.default_rng(1)
    dk = "D:" + c.HEADLINE_DIRECT[args.network]
    print(f"=== {args.network}; headline direct {dk} ===")
    gap = t.filter(pl.col("stat").str.starts_with("gap_"))
    print("GAP meter minus Primaries' observed sum (MW):")
    print(
        gap.select("group", "stat", "value", "n_rows").pivot(
            on="stat", index="group", values="value"
        )
    )
    for g in c.GROUP_IDS[:2]:
        print(f"\n--- {c.GROUP_NAMES[g]} ---")
        for band in ("0-240h", "0-12h", "12-36h", "36-120h", "120-240h"):
            ex = t.filter(
                (pl.col("group") == g)
                & (pl.col("band") == band)
                & (pl.col("stat").str.starts_with("exceed"))
            )
            print(f"[{band}] exceedance (rate; nominal se; n_days)")
            for side, tau, nom in (
                ("export", 0.95, 0.05),
                ("export", 0.9, 0.1),
                ("import", 0.95, 0.05),
                ("import", 0.9, 0.1),
                ("export", 0.5, 0.5),
                ("import", 0.5, 0.5),
            ):
                if band not in ("0-240h",) and tau == 0.5:
                    continue
                r = ex.filter((pl.col("side") == side) & (pl.col("stat") == f"exceed_{tau}"))
                line = []
                for s in ["J0", "J2", "J3", "J5", "J0+g", "J2+g", "J3+g", "J5+g", dk]:
                    x = r.filter(pl.col("source") == s)
                    if x.height:
                        line.append(f"{s}:{100 * x['value'][0]:.1f}%")
                se = r["se_nominal"][0] * 100 if r.height else float("nan")
                print(
                    f"  {side} q{int(tau * 100)} (nominal {nom * 100:.0f}% se {se:.1f}): "
                    + " ".join(line)
                )
        census = t.filter((pl.col("group") == g) & (pl.col("source") == "census"))
        print("census:", census.select("stat", "side", "value", "se_nominal", "n_rows").rows())
        for stat in ("twcrps_month", "twcrps_ph_month"):
            print(stat, "twCRPS (all leads) mean of monthly means, MW, export / import:")
            for side in ("export", "import"):
                f = (
                    t.filter(
                        (pl.col("group") == g)
                        & (pl.col("stat") == stat)
                        & (pl.col("side") == side)
                        & (pl.col("band") == "0-240h")
                    )
                    .group_by("source")
                    .agg(pl.col("value").mean())
                    .sort("source")
                )
                print(
                    "  ", side, {r["source"]: round(r["value"], 4) for r in f.iter_rows(named=True)}
                )
            print(
                "CRPS (all leads) MW:",
                {
                    r["source"]: round(r["value"], 3)
                    for r in t.filter(
                        (pl.col("group") == g)
                        & (pl.col("stat") == "crps")
                        & (pl.col("band") == "0-240h")
                    ).iter_rows(named=True)
                },
            )
            for version, sfx in (("as-is", ""), ("gap-corrected", "+g")):
                print(f"contrasts, {version}, export twCRPS 0-240h (percent, 95% month-block):")
                for a in ("J2", "J3", "J5"):
                    m, lo, hi = contrast(
                        stat=stat,
                        t=t,
                        g=g,
                        a=a + sfx,
                        b="J0" + sfx,
                        side="export",
                        band="0-240h",
                        rng=rng,
                    )
                    print(f"  {a}{sfx} vs J0{sfx}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]")
                best = {
                    s: contrast(
                        stat=stat, t=t, g=g, a=dk, b=s + sfx, side="export", band="0-240h", rng=rng
                    )
                    for s in ("J0", "J2", "J3", "J5")
                }
                print(
                    "  direct vs sums:",
                    {k: f"{v[0]:+.1f} [{v[1]:+.1f},{v[2]:+.1f}]" for k, v in best.items()},
                )
                # decision rule
                ex = t.filter(
                    (pl.col("group") == g)
                    & (pl.col("band") == "0-240h")
                    & (pl.col("stat") == "exceed_0.95")
                    & (pl.col("side") == "export")
                )
                usable = []
                for a in ("J3", "J5"):
                    row = ex.filter(pl.col("source") == a + sfx)
                    near = abs(row["value"][0] - 0.05) <= 2 * row["se_nominal"][0]
                    m, lo, hi = contrast(
                        stat=stat,
                        t=t,
                        g=g,
                        a=a + sfx,
                        b="J0" + sfx,
                        side="export",
                        band="0-240h",
                        rng=rng,
                    )
                    usable.append((a, bool(near), hi < 0))
                direct_needed = all(v[2] < -5 for v in best.values())
                print(
                    f"  sum usable candidates (set, within 2 se of 5%, beats J0 wholly): {usable}"
                )
                print(
                    f"  sum usable: {any(n and b for _, n, b in usable)}; direct needed (upper end below -5% vs every sum): {direct_needed}"
                )
        bias = t.filter((pl.col("group") == g) & (pl.col("stat").str.starts_with("bias_median")))
        print("bias y - median sum (MW):")
        print(
            bias.select("source", "stat", "value")
            .pivot(on="stat", index="source", values="value")
            .filter(pl.col("source").is_in(["J3", "J3+g", dk]))
        )


if __name__ == "__main__":
    main()
