# ruff: noqa: E501
"""Experiment 634, part 2: recoveries, calibration, step bars, noise floor, and coverage (CPU).

Reads `work/probes/<cutoff>/<condition>.parquet` (from `e00634_probe.py`) and writes
`work/step_recoveries.csv` and `work/summary.json`, and prints every number the write-up quotes.
Definitions are in the "Definitions" section of the result page.

Usage: python e00634_analyse.py
"""

import argparse
import itertools
import json

import e00623_probe as p623  # ty: ignore[unresolved-import]
import e00634_common as k
import numpy as np
import polars as pl

Z80 = 1.2816
STEP_BAR = 0.70
FLOOR_BAR_MW = 0.3
FARM_CONDITIONS = ("farm_mw4_r0", "farm_mw4_r1")
STEPS = tuple(c for c in k.CONDITIONS if c.startswith("step_"))
PLACEBOS = tuple(c for c in k.CONDITIONS if c.startswith("shufstep_"))


def load(*, probes: str) -> pl.DataFrame:
    """Return every plain-regression probe result."""
    frames = [pl.read_parquet(f) for f in sorted((k.WORK / probes).glob("*/*.parquet"))]
    return pl.concat(frames).filter(pl.col("model") == "plain")


def paired(df: pl.DataFrame) -> pl.DataFrame:
    """Return the capacity difference of each injected network to its control.

    The control has the same cutoff, seed, and Primary.
    """
    ctl = df.filter(pl.col("condition") == "control").select(
        "cutoff", "seed", "site", ctl=pl.col("capacity_mw"), ctl_boot=pl.col("boot")
    )
    inj = df.filter(pl.col("condition") != "control").join(ctl, on=["cutoff", "seed", "site"])
    boot = np.array(inj["boot"].to_list()) - np.array(inj["ctl_boot"].to_list())
    return inj.select("cutoff", "condition", "seed", "site").with_columns(
        delta_mw=pl.Series(inj["capacity_mw"].to_numpy() - inj["ctl"].to_numpy()),
        boot_sd=pl.Series(boot.std(axis=1)),
    )


def injected_farm(*, site: str, condition: str) -> str:
    """Return the farm injected at a Primary by a condition (rotation 0 unless the name says `_r1`)."""
    rotation = 1 if condition.endswith("_r1") else 0
    return k.FARMS[(list(k.SIX).index(site) + rotation) % len(k.FARMS)]


def post_step_share() -> pl.DataFrame:
    """Return, per cutoff, Primary, and step month, the share of donor rows dated on or after the step."""
    out = []
    for cutoff in k.CUTOFFS:
        with np.load(k.frame_path(cutoff=cutoff)) as z:
            frame = {name: z[name] for name in z.files}
        for site, i in k.six_indices().items():
            probe = p623.cells_and_donors(frame=frame, primary=i)
            donor_day = probe["days"][probe["donor"]]
            for month, start in STEP_STARTS.items():
                out.append(
                    {
                        "cutoff": cutoff, "site": site, "month": month,
                        "share": float((donor_day >= start).mean()),
                        "cells": len(np.unique(probe["cell"])),
                    }
                )  # fmt: skip
    return pl.DataFrame(out)


STEP_STARTS = {m: int(np.datetime64(d, "D").astype(np.int64)) for m, d in k.STEP_STARTS.items()}
"""First day of each step, in days since the epoch (the unit of the probe's `days`)."""


def main() -> None:  # noqa: PLR0915
    """Print and store the recoveries, bars, noise floor, and coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes",
        default="probes",
        help="`probes` (base dates in 2025) or `probes_shift364` (base dates in 2024)",
    )
    args = parser.parse_args()
    tag = "" if args.probes == "probes" else "_" + args.probes
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    df = load(probes=args.probes)
    rec = paired(df)
    rec = rec.with_columns(
        farm=pl.struct("site", "condition").map_elements(
            lambda r: injected_farm(site=r["site"], condition=r["condition"]),
            return_dtype=pl.String,
        ),
        mw=pl.col("condition").map_elements(lambda c: k.parse(name=c).mw, return_dtype=pl.Float64),
    )
    # leave-one-farm-out factor per cutoff and held-out farm, from the constant-injection networks
    const = rec.filter(pl.col("condition").is_in(FARM_CONDITIONS)).with_columns(
        recovery=pl.col("delta_mw") / pl.col("mw")
    )
    factors = []
    for cutoff in k.CUTOFFS:
        sub = const.filter(pl.col("cutoff") == cutoff)
        for farm in k.FARMS:
            held = sub.filter(pl.col("farm") != farm)
            factors.append(
                {"cutoff": cutoff, "farm": farm, "factor": float(held["recovery"].median())}  # ty: ignore[invalid-argument-type]
            )
    fac = pl.DataFrame(factors)
    spread = fac.group_by("cutoff").agg(
        factor_sd=pl.col("factor").std(), factor_mean=pl.col("factor").mean()
    )
    print("leave-one-farm-out factor by cutoff and held-out farm:")
    print(fac.pivot(on="farm", index="cutoff", values="factor").sort("cutoff"))
    print(spread.sort("cutoff"))
    print(
        "constant-injection raw recovery by cutoff and Primary (median over seeds and rotations):"
    )
    print(
        const.group_by("cutoff", "site")
        .agg(raw=pl.col("recovery").median())
        .pivot(on="site", index="cutoff", values="raw")
        .sort("cutoff")
    )

    rec = rec.join(fac, on=["cutoff", "farm"]).join(spread, on="cutoff")
    rec = rec.with_columns(
        recovery_raw=pl.col("delta_mw") / pl.col("mw"),
        recovery_cal=pl.col("delta_mw") / pl.col("mw") / pl.col("factor"),
    )
    sd = (
        rec.group_by("cutoff", "condition", "site")
        .agg(seed_sd=pl.col("delta_mw").std())
        .fill_null(0.0)
    )
    rec = rec.join(sd, on=["cutoff", "condition", "site"])
    rel_f = pl.col("factor_sd") / pl.col("factor")
    rec = rec.with_columns(
        half_cal=Z80
        * (
            (pl.col("boot_sd") ** 2 + pl.col("seed_sd") ** 2).sqrt()
            / pl.col("mw")
            / pl.col("factor") ** 1
        )
        .pow(2)
        .add((pl.col("recovery_cal") * rel_f) ** 2)
        .sqrt()
    )
    shares = post_step_share()
    print("cells per Primary by cutoff and post-step donor share (median over Primaries):")
    print(
        shares.group_by("cutoff", "month")
        .agg(cells=pl.col("cells").median(), share=pl.col("share").median())
        .sort("cutoff", "month")
    )
    rec = rec.with_columns(month=pl.col("condition").str.extract(r"_(spring|summer)_", 1)).join(
        shares, on=["cutoff", "site", "month"], how="left"
    )
    rec.write_csv(k.WORK / f"step_recoveries{tag}.csv")

    # step bar
    print(
        "\nSTEP WORLDS: median calibrated recovery of the full step (per Primary, median over seeds), by cutoff"
    )
    steps = rec.filter(pl.col("condition").is_in(STEPS))
    per = steps.group_by("cutoff", "condition", "site").agg(
        cal=pl.col("recovery_cal").median(),
        raw=pl.col("recovery_raw").median(),
        share=pl.col("share").first(),
    )
    for cond in STEPS:
        print(cond)
        print(
            per.filter(pl.col("condition") == cond)
            .pivot(on="site", index="cutoff", values="cal")
            .sort("cutoff")
        )
        print(
            per.filter(pl.col("condition") == cond)
            .group_by("cutoff")
            .agg(
                median_cal=pl.col("cal").median(),
                median_raw=pl.col("raw").median(),
                median_share=pl.col("share").median(),
                n_ge_bar=(pl.col("cal") >= STEP_BAR).sum(),
            )
            .sort("cutoff")
        )
    print("\nCONSTANT INJECTION: median calibrated recovery by cutoff (should be near 1)")
    cper = (
        rec.filter(pl.col("condition").is_in(FARM_CONDITIONS))
        .group_by("cutoff", "site")
        .agg(cal=pl.col("recovery_cal").median())
    )
    print(cper.pivot(on="site", index="cutoff", values="cal").sort("cutoff"))

    # placebos
    print(
        "\nSHUFFLED-SHAPE STEP PLACEBOS: raw capacity difference in MW (injected 4 MW), per Primary, 95th percentile of the absolute value over seeds"
    )
    pl_rows = (
        rec.filter(pl.col("condition").is_in(PLACEBOS))
        .group_by("cutoff", "condition", "site")
        .agg(p95=pl.col("delta_mw").abs().quantile(0.95))
    )
    print(
        pl_rows.group_by("cutoff", "condition")
        .agg(max_p95=pl.col("p95").max(), median_p95=pl.col("p95").median())
        .sort("cutoff", "condition")
    )

    # determinism check: summer step at 2025-07
    det = rec.filter((pl.col("cutoff") == "2025-07") & pl.col("condition").str.contains("summer"))
    print(
        "\nDETERMINISM (summer-step networks at 2025-07 train on the control's rows): max |raw delta| MW =",
        float(det["delta_mw"].abs().max()),  # ty: ignore[invalid-argument-type]
    )

    # seed noise floor
    ctl = df.filter(pl.col("condition") == "control")
    diffs = []
    for cutoff, site in itertools.product(k.CUTOFFS, k.SIX):
        s = (
            ctl.filter((pl.col("cutoff") == cutoff) & (pl.col("site") == site))
            .sort("seed")["capacity_mw"]
            .to_list()
        )
        diffs += [abs(a - b) for a, b in itertools.combinations(s, 2)]
    floor = float(np.percentile(diffs, 95))
    print(
        f"\nSEED NOISE FLOOR: {len(diffs)} differences, 95th percentile {floor:.3f} MW, median {np.median(diffs):.3f} MW (bar {FLOOR_BAR_MW})"
    )

    # coverage
    print("\nCOVERAGE of the widened (calibrated, leave-one-farm-out) interval, per network")
    cov = {}
    for name, sel, truth in (
        ("constant injection, truth 1", FARM_CONDITIONS, pl.lit(1.0)),
        ("step, truth = full step 1", STEPS, pl.lit(1.0)),
        ("step, truth = calendar-blind share", STEPS, pl.col("share")),
    ):
        sub = rec.filter(pl.col("condition").is_in(sel)).with_columns(truth=truth)
        sub = sub.with_columns(
            covered=((pl.col("recovery_cal") - pl.col("truth")).abs() <= pl.col("half_cal"))
        )
        cov[name] = float(sub["covered"].mean())  # ty: ignore[invalid-argument-type]
        by = sub.group_by("cutoff").agg(cov=pl.col("covered").mean()).sort("cutoff")
        print(name, f"{cov[name]:.2f}", dict(zip(by["cutoff"], by["cov"].round(2), strict=True)))
    (k.WORK / f"summary{tag}.json").write_text(
        json.dumps({"noise_floor_p95_mw": floor, "n_diffs": len(diffs), "coverage": cov}, indent=1)
    )


if __name__ == "__main__":
    main()
