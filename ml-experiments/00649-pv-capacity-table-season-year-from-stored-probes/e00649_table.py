# ruff: noqa: E501
"""Experiment 649: the season-year solar capacity table for the six MW Primaries, from stored probes.

Reads the stored `plain` donor-day probe outputs of experiment 634 (`work/probes` for base dates in
April to September 2025, `work/probes_shift364` for April to September 2024) and refits nothing.
For each cutoff, probe half-year, and Primary it writes the raw capacity (median over 3 seeds of the
uninjected control), the calibrated capacity (raw divided by the leave-one-farm-out factor, widened
by the spread of recoveries across the four injected farms), and the comparison with experiment
474's R4 table, experiment 479's anomaly rung, and experiment 267's register floor. Prints every
number the write-up quotes.

Usage: python e00649_table.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00649-pv-capacity-table-season-year-from-stored-probes"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
E634: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00634-pv-probe-capacity-step-world-and-energy-recovery" / "work"
)
E474: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
)
E479: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00479-pv-capacity-anomaly-regression-rung" / "work"
E129: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00129-differentiable-physics-pv-capacity-known-answer-test" / "work"
)
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six signed-megawatt Primaries, in the order experiment 634 injects farms."""
FARMS: Final[str] = "ABCF"
CUTOFFS: Final[tuple[str, ...]] = ("2025-07", "2025-10", "2026-03", "2026-06")
HEADLINE_CUTOFF: Final = "2026-06"
PROBE_SETS: Final[dict[int, str]] = {2025: "probes", 2024: "probes_shift364"}
"""Probe half-year (April to September of the year) and the folder of experiment 634's probes."""
INJECTED_MW: Final = 4.0
Z80: Final = 1.2816
"""Standard-normal 90th percentile: the 10th and 90th percentiles are the centre minus and plus this many standard deviations."""
TARGET_COVERAGE: Final = 0.80
HEADLINE_STOP_SEED_SD_MW: Final = 0.5
ENERGY_BASES: Final[tuple[str, ...]] = ("rows", "donors")


def load_probes(*, folder: str, year: int) -> pl.DataFrame:
    """Return the `plain` control and 4 MW real-farm probes of one probe half-year."""
    frames = [
        pl.read_parquet(E634 / folder / cutoff / f"{condition}.parquet")
        for cutoff in CUTOFFS
        for condition in ("control", "farm_mw4_r0", "farm_mw4_r1")
    ]
    out = pl.concat(frames).filter(pl.col("model") == "plain")
    assert set(out["site"].unique()) == set(SIX), out["site"].unique()
    return out.with_columns(
        probe_year=pl.lit(year),
        boot_sd=pl.col("boot").list.eval(pl.element().std()).list.first(),
    )


def raw_capacity(*, probes: pl.DataFrame) -> pl.DataFrame:
    """Return the raw capacity per probe half-year, cutoff, and Primary.

    The centre is the median of the 3 seeds' controls. The spread adds, in quadrature, the mean
    bootstrap standard deviation and the standard deviation over seeds.
    """
    return (
        probes.filter(pl.col("condition") == "control")
        .group_by("probe_year", "cutoff", "site")
        .agg(
            raw_p50=pl.col("capacity_mw").median(),
            seed_sd=pl.col("capacity_mw").std(),
            boot_sd=pl.col("boot_sd").mean(),
            n_seeds=pl.len(),
        )
        .with_columns(sigma_raw=(pl.col("seed_sd") ** 2 + pl.col("boot_sd") ** 2).sqrt())
    )


def farm_recoveries(*, probes: pl.DataFrame) -> pl.DataFrame:
    """Return the recovery of each injected 4 MW farm: the probed capacity minus the twin's, over 4 MW.

    The twin is the uninjected control with the same probe half-year, cutoff, seed, and Primary.
    Farm `FARMS[(Primary index + rotation) % 4]` is injected at each Primary.
    """
    ctl = probes.filter(pl.col("condition") == "control").select(
        "probe_year", "cutoff", "seed", "site", ctl=pl.col("capacity_mw"), ctl_boot=pl.col("boot")
    )
    inj = probes.filter(pl.col("condition") != "control").join(
        ctl, on=["probe_year", "cutoff", "seed", "site"]
    )
    boot = np.array(inj["boot"].to_list()) - np.array(inj["ctl_boot"].to_list())
    return inj.select("probe_year", "cutoff", "condition", "seed", "site").with_columns(
        delta_mw=pl.Series(inj["capacity_mw"].to_numpy() - inj["ctl"].to_numpy()),
        boot_sd=pl.Series(boot.std(axis=1, ddof=1)),
        farm=pl.struct("site", "condition").map_elements(
            lambda r: FARMS[
                (SIX.index(r["site"]) + (1 if r["condition"].endswith("_r1") else 0)) % 4
            ],
            return_dtype=pl.String,
        ),
        recovery=pl.Series(inj["capacity_mw"].to_numpy() - inj["ctl"].to_numpy()) / INJECTED_MW,
    )


def factors(*, rec: pl.DataFrame, value: str = "recovery") -> pl.DataFrame:
    """Return, per probe half-year and cutoff, the leave-one-farm-out factor and the spread across farms.

    `factor` is the mean over the four held-out farms of the median recovery of the other three
    farms. `farm_sd` is the standard deviation across the four farms' median recoveries.
    """
    per_farm = rec.group_by("probe_year", "cutoff", "farm").agg(m=pl.col(value).median())
    out = []
    for (year, cutoff), g in per_farm.group_by("probe_year", "cutoff"):
        med = dict(zip(g["farm"], g["m"], strict=True))
        lofo = [float(np.median([med[f] for f in FARMS if f != held])) for held in FARMS]
        out.append(
            {
                "probe_year": year, "cutoff": cutoff, "factor": float(np.mean(lofo)),
                "farm_sd": float(np.std([med[f] for f in FARMS], ddof=1)),
                "farm_min": min(med.values()), "farm_max": max(med.values()),
            }
        )  # fmt: skip
    return pl.DataFrame(out)


def calibrate(*, raw: pl.DataFrame, fac: pl.DataFrame, widen: float = 1.0) -> pl.DataFrame:
    """Return the calibrated 10th, 50th, and 90th percentiles of AC capacity.

    The centre is raw / factor. The standard deviation combines the raw spread and the farm spread
    in quadrature: sqrt((sigma_raw / factor)^2 + (raw * farm_sd / factor^2)^2). `widen` multiplies it.
    """
    out = raw.join(fac, on=["probe_year", "cutoff"]).with_columns(
        cal_p50=pl.col("raw_p50") / pl.col("factor")
    )
    return out.with_columns(
        sigma_cal=widen
        * (
            (pl.col("sigma_raw") / pl.col("factor")) ** 2
            + (pl.col("raw_p50") * pl.col("farm_sd") / pl.col("factor") ** 2) ** 2
        ).sqrt()
    ).with_columns(
        raw_p10=pl.col("raw_p50") - Z80 * pl.col("sigma_raw"),
        raw_p90=pl.col("raw_p50") + Z80 * pl.col("sigma_raw"),
        cal_p10=pl.col("cal_p50") - Z80 * pl.col("sigma_cal"),
        cal_p90=pl.col("cal_p50") + Z80 * pl.col("sigma_cal"),
    )


def coverage(*, rec: pl.DataFrame, widen: float = 1.0) -> pl.DataFrame:
    """Return, per injected network, whether the calibrated 10th-to-90th interval covers the truth.

    The truth is a recovery of 1. The factor is leave-one-farm-out (the other three farms'
    median recovery) and the farm spread is the standard deviation of those three farms' medians.
    A network is one cutoff, Primary, and injected condition; the seed standard deviation of
    its probed capacity difference is taken over its 3 seeds.
    """
    per_farm = rec.group_by("probe_year", "cutoff", "farm").agg(m=pl.col("recovery").median())
    lofo = []
    for (year, cutoff), g in per_farm.group_by("probe_year", "cutoff"):
        med = dict(zip(g["farm"], g["m"], strict=True))
        for held in FARMS:
            others = [med[f] for f in FARMS if f != held]
            lofo.append(
                {
                    "probe_year": year, "cutoff": cutoff, "farm": held,
                    "factor": float(np.median(others)), "farm_sd": float(np.std(others, ddof=1)),
                }
            )  # fmt: skip
    net = (
        rec.group_by("probe_year", "cutoff", "condition", "site", "farm")
        .agg(
            delta=pl.col("delta_mw").median(),
            seed_sd=pl.col("delta_mw").std(),
            boot_sd=pl.col("boot_sd").mean(),
        )
        .join(pl.DataFrame(lofo), on=["probe_year", "cutoff", "farm"])
    )
    return net.with_columns(
        rec_cal=pl.col("delta") / INJECTED_MW / pl.col("factor"),
        sigma=widen
        * (
            (
                (pl.col("seed_sd") ** 2 + pl.col("boot_sd") ** 2).sqrt()
                / INJECTED_MW
                / pl.col("factor")
            )
            ** 2
            + (
                pl.col("delta")
                / INJECTED_MW
                / pl.col("factor")
                * pl.col("farm_sd")
                / pl.col("factor")
            )
            ** 2
        ).sqrt(),
    ).with_columns(covered=(pl.col("rec_cal") - 1.0).abs() <= Z80 * pl.col("sigma"))


def widening_for_target(*, rec: pl.DataFrame) -> float:
    """Return the smallest multiplier on the standard deviation that gives 80% coverage (post hoc)."""
    for widen in np.arange(1.0, 4.01, 0.05):
        if float(coverage(rec=rec, widen=float(widen))["covered"].mean()) >= TARGET_COVERAGE:  # ty: ignore[invalid-argument-type]
            return float(widen)
    print("WARNING: no multiplier up to 4 reaches the target coverage")
    return float("nan")


def energy_basis(*, cal: pl.DataFrame) -> pl.DataFrame:
    """Return the energy-basis sensitivity rows for the headline cutoff and the 2025 half-year.

    Experiment 634's `energy_recoveries.csv` holds the 4 MW real-farm injections of the 2026-03 and
    2026-06 cutoffs, with the recovery per injected megawatt-hour and the farm-over-unit-curve energy
    ratio on two bases (all daylight rows, and the donor rows). `physics_equiv` is the raw capacity
    divided by the energy recovery: the capacity of a physics-shaped plant that carries the same
    energy. `ac_energy` divides that by the median energy ratio: the AC capacity of a fleet shaped
    like the injected farms. The spread adds the farm spread of the energy recovery.
    """
    e = pl.read_csv(E634 / "energy_recoveries.csv").filter(
        pl.col("condition").str.starts_with("farm_mw4"),
        pl.col("cutoff") == HEADLINE_CUTOFF,
        pl.col("experiment") == 621,
        pl.col("arm") == "plain",
    )
    rows = []
    raw = cal.filter(pl.col("probe_year") == 2025, pl.col("cutoff") == HEADLINE_CUTOFF)
    for basis in ENERGY_BASES:
        per_farm = e.group_by("farm").agg(
            rec=pl.col(f"recovery_energy_{basis}").median(), ratio=pl.col(f"ratio_{basis}").median()
        )
        rec_med, rec_sd = float(per_farm["rec"].median()), float(per_farm["rec"].std())  # ty: ignore[invalid-argument-type]
        ratio = float(per_farm["ratio"].median())  # ty: ignore[invalid-argument-type]
        for r in raw.iter_rows(named=True):
            p50 = r["raw_p50"] / rec_med
            sd = float(np.hypot(r["sigma_raw"] / rec_med, r["raw_p50"] * rec_sd / rec_med**2))
            rows.append(
                {
                    "site": r["site"], "basis": basis, "energy_recovery": rec_med,
                    "energy_ratio": ratio, "physics_equiv_p10": p50 - Z80 * sd,
                    "physics_equiv_p50": p50, "physics_equiv_p90": p50 + Z80 * sd,
                    "ac_energy_p10": (p50 - Z80 * sd) / ratio, "ac_energy_p50": p50 / ratio,
                    "ac_energy_p90": (p50 + Z80 * sd) / ratio,
                }
            )  # fmt: skip
    return pl.DataFrame(rows)


def comparators() -> dict[str, pl.DataFrame]:
    """Return experiment 474's R4 table, experiment 479's anomaly rung, and the register floor."""
    r4 = (
        pl.read_parquet(E474 / "nged_table.parquet")
        .filter(pl.col("site").is_in(SIX))
        .select("site", "season_year", "ac_p10", "ac_p50", "ac_p90", "status")
    )
    a479 = (
        pl.read_parquet(E479 / "capacity_table.parquet")
        .filter(~pl.col("cloud"), pl.col("prior") == "none", pl.col("site").is_in(SIX))
        .select("site", a479_p10="p10", a479_p50="p50", a479_p90="p90")
    )
    floor = (
        pl.read_parquet(E129 / "registers.parquet")
        .filter(pl.col("arm") == "arm2", pl.col("cut_mw") == 1.0)
        .select("site", floor_ac=pl.col("fit_ac_mw"), floor_dc=pl.col("fit_dc_mw"))
    )
    return {"r4": r4, "a479": a479, "floor": floor}


def overlap(*, lo1: float, hi1: float, lo2: float, hi2: float) -> tuple[float, float]:
    """Return the overlap of two intervals in MW (negative: the gap) and as a share of the narrower."""
    ov = min(hi1, hi2) - max(lo1, lo2)
    return ov, ov / min(hi1 - lo1, hi2 - lo2)


def comparison(*, cal: pl.DataFrame, comp: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return, per probe half-year and Primary at the headline cutoff, the overlap with each comparator."""
    r4 = comp["r4"]
    rows = []
    for r in cal.filter(pl.col("cutoff") == HEADLINE_CUTOFF).iter_rows(named=True):
        y, site = r["probe_year"], r["site"]
        lo, hi = r["cal_p10"], r["cal_p90"]
        sy = {
            s: row
            for s in (y - 1, y)
            for row in r4.filter(pl.col("site") == site, pl.col("season_year") == s).iter_rows(
                named=True
            )
        }
        hull = (min(v["ac_p10"] for v in sy.values()), max(v["ac_p90"] for v in sy.values()))
        row = {
            "probe_year": y, "site": site, "cal_p10": lo, "cal_p50": r["cal_p50"], "cal_p90": hi,
            "r4_hull_p10": hull[0], "r4_hull_p90": hull[1],
            "r4_p50_ratio": r["cal_p50"] / float(np.mean([v["ac_p50"] for v in sy.values()])),
            "r4_status": "; ".join(sorted({v["status"] for v in sy.values() if v["status"]})),
        }  # fmt: skip
        row["overlap_r4_mw"], row["overlap_r4_share"] = overlap(
            lo1=lo, hi1=hi, lo2=hull[0], hi2=hull[1]
        )
        for s, v in sy.items():
            row[f"overlap_r4_{s}_mw"], _ = overlap(lo1=lo, hi1=hi, lo2=v["ac_p10"], hi2=v["ac_p90"])
        a = comp["a479"].filter(pl.col("site") == site).row(0, named=True)
        row["a479_p10"], row["a479_p50"], row["a479_p90"] = (
            a["a479_p10"],
            a["a479_p50"],
            a["a479_p90"],
        )
        row["overlap_479_mw"], row["overlap_479_share"] = overlap(
            lo1=lo, hi1=hi, lo2=a["a479_p10"], hi2=a["a479_p90"]
        )
        row["a479_p50_ratio"] = r["cal_p50"] / a["a479_p50"]
        fl = comp["floor"].filter(pl.col("site") == site)
        row["floor_ac"] = float(fl["floor_ac"][0]) if len(fl) else None
        row["p90_at_or_above_floor"] = (hi >= row["floor_ac"]) if len(fl) else None
        rows.append(row)
    return pl.DataFrame(rows).sort("probe_year", "site")


def figure(*, comp_tbl: pl.DataFrame) -> None:
    """Write the headline figure: capacity with 80% intervals per Primary against experiment 474's R4 table."""
    y = 2025
    d = comp_tbl.filter(pl.col("probe_year") == y)
    long = []
    for r in d.iter_rows(named=True):
        long += [
            {"site": r["site"], "source": "This experiment (calibrated probe)", "lo": r["cal_p10"], "mid": r["cal_p50"], "hi": r["cal_p90"]},
            {"site": r["site"], "source": "Experiment 474, R4 (both season-years)", "lo": r["r4_hull_p10"], "mid": None, "hi": r["r4_hull_p90"]},
            {"site": r["site"], "source": "Experiment 479, anomaly rung", "lo": r["a479_p10"], "mid": r["a479_p50"], "hi": r["a479_p90"]},
        ]  # fmt: skip
    df = pl.DataFrame(long)
    sources = [
        "This experiment (calibrated probe)",
        "Experiment 474, R4 (both season-years)",
        "Experiment 479, anomaly rung",
    ]
    colours = [theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_GREEN]
    base = alt.Chart(df).encode(
        y=alt.Y("source:N", sort=sources, axis=None),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=sources, range=colours),
            legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=500),
        ),
    )
    rule = base.mark_rule(strokeWidth=4).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="AC capacity of unmetered solar (MW)", scale=alt.Scale(zero=True)),
        x2="hi:Q",
    )
    dot = base.mark_point(filled=True, size=70).encode(x="mid:Q")  # ty: ignore[unresolved-attribute]
    charts = []
    for site in SIX:
        fl = d.filter(pl.col("site") == site, pl.col("floor_ac").is_not_null())
        layers = [
            rule.transform_filter(alt.datum.site == site),
            dot.transform_filter(alt.datum.site == site),
        ]
        if len(fl):
            layers.append(
                alt.Chart(fl).mark_rule(color="black", strokeDash=[4, 3]).encode(x="floor_ac:Q")  # ty: ignore[unresolved-attribute]
            )
        charts.append(alt.layer(*layers, data=df).properties(width=150, height=60, title=site))  # ty: ignore[invalid-argument-type]
    rows = [alt.hconcat(*charts[i : i + 3]).resolve_scale(x="shared") for i in (0, 3)]
    chart = alt.vconcat(*rows).properties(
        title=alt.TitleParams(
            [
                "Capacity of unmetered solar at the six MW Primaries, April to September 2025 (AC megawatts; one panel per Primary)",
                "Bar: 10th to 90th percentile. Dot: median. Dashed line: Feed-in Tariff register floor (no floor for D9).",
                "Orange: the calibrated probe. Blue: experiment 474's R4 table, hull of its two season-years. Green: experiment 479.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "capacity_by_primary.svg"))


def main() -> None:  # noqa: PLR0915
    """Build the table, print every quoted number, and write the stored files and the figure."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    WORK.mkdir(parents=True, exist_ok=True)
    raws, recs, facs = [], [], []
    for year, folder in PROBE_SETS.items():
        probes = load_probes(folder=folder, year=year)
        raws.append(raw_capacity(probes=probes))
        rec = farm_recoveries(probes=probes)
        recs.append(rec)
        facs.append(factors(rec=rec))
    raw, rec, fac = pl.concat(raws), pl.concat(recs), pl.concat(facs)
    print("SEED SD of the raw capacity (MW), max over cutoffs, by probe half-year and Primary")
    print(
        raw.group_by("probe_year", "site")
        .agg(max_seed_sd=pl.col("seed_sd").max())
        .sort("probe_year", "site")
    )
    print("FACTORS (leave-one-farm-out mean, spread across the four farms)")
    print(fac.sort("probe_year", "cutoff"))
    cal = calibrate(raw=raw, fac=fac)
    cov = coverage(rec=rec)
    cov_by = (
        cov.group_by("probe_year", "cutoff")
        .agg(coverage=pl.col("covered").mean(), n=pl.len())
        .sort("probe_year", "cutoff")
    )
    cov_all = float(cov["covered"].mean())  # ty: ignore[invalid-argument-type]
    print(
        f"COVERAGE of the pre-registered interval on leave-one-farm-out constant injections: {cov_all:.3f}"
    )
    print(cov_by)
    widen = widening_for_target(rec=rec)
    print(f"POST HOC widening factor for {TARGET_COVERAGE:.0%} coverage: {widen:.2f}")
    cal_ph = calibrate(raw=raw, fac=fac, widen=widen).select(
        "probe_year", "cutoff", "site", ph_p10="cal_p10", ph_p90="cal_p90"
    )
    cal = cal.join(cal_ph, on=["probe_year", "cutoff", "site"])
    print("RAW AND CALIBRATED CAPACITY, headline cutoff")
    print(
        cal.filter(pl.col("cutoff") == HEADLINE_CUTOFF)
        .select(
            "probe_year",
            "site",
            "raw_p10",
            "raw_p50",
            "raw_p90",
            "factor",
            "cal_p10",
            "cal_p50",
            "cal_p90",
            "ph_p10",
            "ph_p90",
            "seed_sd",
        )
        .sort("probe_year", "site")
    )
    print("CALIBRATED MEDIAN BY CUTOFF (sensitivity)")
    print(
        cal.select("probe_year", "cutoff", "site", "cal_p50")
        .pivot(on="cutoff", index=["probe_year", "site"], values="cal_p50")
        .sort("probe_year", "site")
    )
    spread = cal.group_by("probe_year", "site").agg(
        cut_ratio=pl.col("cal_p50").max() / pl.col("cal_p50").min()
    )
    print(
        "MAX / MIN of the calibrated median across the four cutoffs; Primaries above 1.3 (stop rule 2):",
        int((spread["cut_ratio"] > 1.3).sum()),
    )
    print(spread.sort("probe_year", "site"))
    print(
        "RATIO calibrated / raw median (headline cutoff):",
        cal.filter(pl.col("cutoff") == HEADLINE_CUTOFF)
        .select((pl.col("cal_p50") / pl.col("raw_p50")).alias("r"))["r"]
        .describe()
        .to_dict(as_series=False),
    )
    en = energy_basis(cal=cal)
    print("ENERGY-BASIS SENSITIVITY (2025 half-year, headline cutoff)")
    print(en)
    comp = comparators()
    tbl = comparison(cal=cal, comp=comp)
    print("COMPARISON at the headline cutoff")
    print(tbl)
    for y in PROBE_SETS:
        t = tbl.filter(pl.col("probe_year") == y)
        print(
            f"half-year {y}: overlap with R4 hull at {int((t['overlap_r4_mw'] > 0).sum())} of 6; "
            f"with 479 at {int((t['overlap_479_mw'] > 0).sum())} of 6; "
            f"p50 within 0.75-1.33x of R4 at {int(t['r4_p50_ratio'].is_between(0.75, 1.33).sum())} of 6; "
            f"p90 at or above the floor at {int(t['p90_at_or_above_floor'].fill_null(False).sum())} of {int(t['floor_ac'].is_not_null().sum())}"
        )
    en_p50 = (
        en.filter(pl.col("basis") == "rows")
        .select("site", "ac_energy_p50")
        .join(
            cal.filter(pl.col("probe_year") == 2025, pl.col("cutoff") == HEADLINE_CUTOFF).select(
                "site", "cal_p50"
            ),
            on="site",
        )
        .with_columns(rel=pl.col("ac_energy_p50") / pl.col("cal_p50") - 1)
    )
    print(
        "ENERGY-BASIS AC median relative to the megawatt-basis calibrated median:",
        en_p50["rel"].to_list(),
    )
    cal.write_parquet(WORK / "capacity_table.parquet")
    tbl.write_parquet(WORK / "comparison.parquet")
    en.write_parquet(WORK / "energy_basis.parquet")
    cov.write_parquet(WORK / "coverage_networks.parquet")
    (WORK / "summary.json").write_text(
        json.dumps({"coverage": cov_all, "post_hoc_widening": widen}, indent=1)
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    figure(comp_tbl=tbl)
    noisy = (
        raw.filter(pl.col("cutoff") == HEADLINE_CUTOFF)
        .group_by("probe_year", "site")
        .agg(s=pl.col("seed_sd").max())
        .filter(pl.col("s") > HEADLINE_STOP_SEED_SD_MW)
        .group_by("probe_year")
        .len()
    )
    print(
        f"stop rule 1: Primaries per half-year with a seed sd above {HEADLINE_STOP_SEED_SD_MW} MW: {noisy.to_dicts()}"
    )


if __name__ == "__main__":
    main()
