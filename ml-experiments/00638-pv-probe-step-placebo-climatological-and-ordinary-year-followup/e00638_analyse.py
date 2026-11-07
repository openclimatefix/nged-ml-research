# ruff: noqa: E501
"""Experiment 638: calibrated step reads, placebo ratios, stop rule, and coverage (CPU).

Reads the probes of experiment 634 (`control`, `farm_mw4_r0`, `farm_mw4_r1`, the real 2025 steps, and
the shuffled-shape placebos) and of this experiment, and writes `work/step_reads.csv`, `work/summary.json`
and prints every number the write-up quotes. Calibration is experiment 634's: the paired difference
to the control, divided by the injected megawatts and by the leave-one-farm-out factor.

Usage: python e00638_analyse.py
"""

import json
from datetime import UTC, datetime

import e00634_analyse as a634  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00634_probe as p634  # ty: ignore[unresolved-import]
import e00638_common as k
import numpy as np
import polars as pl

Z80 = 1.2816
STOP_RATIO = 0.5
SHIFTS = (0, 364)


def load(*, shift_days: int) -> pl.DataFrame:
    """Return the plain-regression probe results of both experiments for one probe-date shift."""
    sub = "probes" if shift_days == 0 else f"probes_shift{shift_days}"
    files = sorted((k634.WORK / sub).glob("*/*.parquet")) + sorted(
        k.probes_dir(shift_days=shift_days).glob("*/*.parquet")
    )
    return pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("model") == "plain")


def factors_and_reads(*, df: pl.DataFrame) -> pl.DataFrame:
    """Return every injected network's calibrated recovery and widened half-width."""
    rec = a634.paired(df)
    rec = rec.with_columns(
        farm=pl.struct("site", "condition").map_elements(
            lambda r: a634.injected_farm(site=r["site"], condition=r["condition"]),
            return_dtype=pl.String,
        ),
        mw=pl.lit(4.0),
    )
    const = rec.filter(pl.col("condition").is_in(a634.FARM_CONDITIONS)).with_columns(
        recovery=pl.col("delta_mw") / 4.0
    )
    fac = pl.DataFrame(
        [
            {
                "cutoff": cutoff, "farm": farm,
                "factor": float(const.filter((pl.col("cutoff") == cutoff) & (pl.col("farm") != farm))["recovery"].median()),  # ty: ignore[invalid-argument-type]
            }
            for cutoff in k.CUTOFFS
            for farm in k634.FARMS
        ]
    )  # fmt: skip
    spread = fac.group_by("cutoff").agg(factor_sd=pl.col("factor").std())
    rec = rec.join(fac, on=["cutoff", "farm"]).join(spread, on="cutoff")
    rec = rec.with_columns(
        recovery_raw=pl.col("delta_mw") / 4.0,
        recovery_cal=pl.col("delta_mw") / 4.0 / pl.col("factor"),
    )
    sd = (
        rec.group_by("cutoff", "condition", "site")
        .agg(seed_sd=pl.col("delta_mw").std())
        .fill_null(0.0)
    )
    rec = rec.join(sd, on=["cutoff", "condition", "site"])
    rel_f = pl.col("factor_sd") / pl.col("factor")
    return rec.with_columns(
        half_cal=Z80 * (((pl.col("boot_sd") ** 2 + pl.col("seed_sd") ** 2).sqrt() / 4.0 / pl.col("factor")).pow(2) + (pl.col("recovery_cal") * rel_f) ** 2).sqrt()
    )  # fmt: skip


def step_table(*, rec: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    """Return, per start key, cutoff, and kind, the median calibrated read over Primaries and seeds."""
    names = {}
    for key in k.STEP_STARTS:
        names[k.real_condition(key=key)] = (key, "real")
        names[f"climstep_{key}_mw4"] = (key, "climatological")
    names["shufstep_spring_mw4"] = ("spring25", "shuffled")
    names["shufstep_summer_mw4"] = ("summer25", "shuffled")
    sub = rec.filter(pl.col("condition").is_in(list(names))).with_columns(
        key=pl.col("condition").replace_strict({c: v[0] for c, v in names.items()}),
        kind=pl.col("condition").replace_strict({c: v[1] for c, v in names.items()}),
    )
    sub = sub.join(context.select("key", "cutoff", "share"), on=["key", "cutoff"], how="left")
    return (
        sub.group_by("key", "kind", "cutoff")
        .agg(
            cal=pl.col("recovery_cal").median(),
            raw_mw=pl.col("delta_mw").median(),
            p95_abs_mw=pl.col("delta_mw").abs().quantile(0.95),
            n_networks=pl.len(),
            covered_0=((pl.col("recovery_cal")).abs() <= pl.col("half_cal")).mean(),
            covered_share=(
                (pl.col("recovery_cal") - pl.col("share")).abs() <= pl.col("half_cal")
            ).mean(),
        )
        .sort("key", "kind", "cutoff")
    )


def donor_context() -> pl.DataFrame:
    """Return, per cutoff and start key, the post-step share of donor rows and the clearness before and after the step."""
    out = []
    for cutoff in k.CUTOFFS:
        frame = p634.load_frame(cutoff=cutoff)
        ghi_all = np.clip(
            frame["x"][:, 0].astype(np.float64) * frame["std_std"][0] + frame["std_mean"][0],
            0.0,
            None,
        )
        for site, i in k634.six_indices().items():
            probe = p634.cells_and_donors(frame=frame, primary=i)
            sel = frame["primary"] == i
            order = np.argsort(frame["target_idx"][sel])
            ghi = ghi_all[sel][order]
            clear = k634.haurwitz_w_m2(cos_zenith=frame["x"][sel][order][:, 14].astype(np.float64))
            kt = ghi / np.maximum(clear, 1.0)
            donor = probe["donor"]
            day = probe["days"][donor]
            for key, start in k.STEP_STARTS.items():
                start_day = int(np.datetime64(start, "D").astype(np.int64))
                post = day >= start_day
                out.append(
                    {
                        "cutoff": cutoff, "site": site, "key": key, "share": float(post.mean()),
                        "kt_pre": float(kt[donor][~post].mean()) if (~post).any() else np.nan,
                        "kt_post": float(kt[donor][post].mean()) if post.any() else np.nan,
                    }
                )  # fmt: skip
    return (
        pl.DataFrame(out)
        .group_by("cutoff", "key")
        .agg(pl.col("share", "kt_pre", "kt_post").median())
        .with_columns(kt_contrast=pl.col("kt_post") - pl.col("kt_pre"))
        .sort("key", "cutoff")
    )


def energy_ratio() -> pl.DataFrame:
    """Return the climatological shape's energy over the real farm's, 10:00 to 15:00 UTC after each start."""
    import e00405_prep as p5  # ty: ignore[unresolved-import]

    rows = []
    for farm in k634.FARMS:
        t, v = p5.farm_norm_series(farm=farm)
        _, clim = k.climatological_profile(series=(t, v))
        hour = (t // 3_600_000_000) % 24
        for key, start in k.STEP_STARTS.items():
            su = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1e6)
            m = (t >= su) & (t < k.DEV_END_US) & ~np.isnan(v) & (hour >= 10) & (hour <= 15)
            rows.append(
                {"farm": farm, "key": key, "clim_over_real": float(clim[m].sum() / v[m].sum())}
            )
    return pl.DataFrame(rows).pivot(on="farm", index="key", values="clim_over_real")


def main() -> None:
    """Print and store the reads, ratios, stop-rule verdict, and coverage."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(3)
    context = donor_context()
    print(
        "POST-STEP SHARE OF DONOR ROWS AND CLEARNESS BEFORE/AFTER THE STEP (median over Primaries)"
    )
    print(context)
    print("CLIMATOLOGICAL SHAPE ENERGY / REAL FARM ENERGY after each start, by farm")
    print(energy_ratio())
    summary: dict = {}
    for shift in SHIFTS:
        rec = factors_and_reads(df=load(shift_days=shift))
        rec.write_csv(k.WORK / f"step_reads_shift{shift}.csv")
        table = step_table(rec=rec, context=context)
        print(
            f"\nBASE DATES SHIFTED BY {shift} DAYS: calibrated read (median over 6 Primaries x 3 seeds)"
        )
        wide = table.pivot(on="kind", index=["key", "cutoff"], values="cal").sort("key", "cutoff")
        wide = wide.join(
            context.select("key", "cutoff", "share", "kt_contrast"),
            on=["key", "cutoff"],
            how="left",
        )
        wide = wide.with_columns(
            ratio_clim=pl.col("climatological") / pl.col("real"),
            ratio_shuf=pl.col("shuffled") / pl.col("real"),
        )
        print(wide)
        print(
            "raw MW, p95 abs, and coverage of the widened interval (covered_share: reference is the post-step donor share; covered_0: zero)"
        )
        print(table)
        if shift == 0:
            test = wide.filter((pl.col("key") == "spring25") & (pl.col("cutoff") != "2025-07"))
            stop = int(((test["ratio_clim"] >= STOP_RATIO) & (test["real"] > 0.2)).sum())
            summary["stop_rule"] = {
                "pairs_tested": test.height,
                "pairs_with_ratio_ge_half": stop,
                "stop": stop >= 2,
                "ratios": dict(zip(test["cutoff"], test["ratio_clim"].round(3), strict=True)),
            }
            print(
                "\nSTOP RULE (spring-2025 step, cutoffs 2025-10, 2026-03, 2026-06):",
                summary["stop_rule"],
            )
        summary[f"shift{shift}"] = json.loads(wide.write_json())
    (k.WORK / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
