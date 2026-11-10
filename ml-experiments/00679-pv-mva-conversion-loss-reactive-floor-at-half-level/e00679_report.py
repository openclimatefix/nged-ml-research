# ruff: noqa: E501
"""Experiment 679: the printed report and the figure of the write-up.

Conversion loss of a probe = median-over-seeds signed-world recovery minus apparent-power-world
recovery of the same farm at the same level, in percentage points, per MW Primary; the headline is
the median over the six. The interval of the median draws each recovery from a normal distribution
whose spread is the day-block bootstrap half-width and the between-seed spread added in quadrature
(experiment 662's interval method), subtracts the draws, and takes the median over the six.

Usage: python e00679_report.py
"""

from pathlib import Path

import altair as alt
import e00662_analyse as an  # ty: ignore[unresolved-import]
import e00662_common as k  # ty: ignore[unresolved-import]
import e00679_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00679-pv-mva-conversion-loss-reactive-floor-at-half-level"
)
ARMS = {
    "pf": "Power factor 0.97 (reactive power 0.25 times real power)",
    "floor": "Reactive floor (constant, 0.10 times the midday reading)",
    "q0": "Zero reactive power",
}
SHORT = {"pf": "Power factor 0.97", "floor": "Reactive floor", "q0": "Zero reactive"}
METHODS = ("L0", "L1", "FA")
LEVELS = {"l05": 0.5, "l1": 1.0}
N_DRAWS = 4000
Z80 = 1.2816


def site_stats(rec: pl.DataFrame) -> pl.DataFrame:
    """Return centre and standard deviation of the recovery per condition, method, and site."""
    rows = []
    sub = rec.filter(pl.col("site").is_in(k.SIX) & pl.col("method").is_in(METHODS))
    for (cond, meth, site), g in sub.group_by(["condition", "method", "site"]):
        assert g.height == len(run.SEEDS), (cond, meth, site, g.height)
        assert g["diff_boot"].null_count() == 0, (cond, meth, site)
        half_boot = float(
            np.mean([np.subtract(*np.percentile(d, [90, 10])) / 2 for d in g["diff_boot"]])
        )
        sd_seed = float(np.std(g["diff_mw"].to_numpy(), ddof=1))
        inj = float(g["inj_mw"][0])
        rows.append({
            "condition": cond, "method": meth, "site": site, "centre": float(g["rec"].median()),  # ty: ignore[invalid-argument-type]
            "sd": float(np.hypot(half_boot, Z80 * sd_seed)) / inj / Z80,  # ty: ignore[invalid-argument-type] "n_seeds": g.height,
        })  # fmt: skip
    return pl.DataFrame(rows)


def loss_table(stats: pl.DataFrame) -> pl.DataFrame:
    """Return the conversion loss per arm, level, method, and site, and its median over the six with an interval."""
    rng = np.random.default_rng(679)
    lookup = {(r["condition"], r["method"], r["site"]): r for r in stats.iter_rows(named=True)}
    out = []
    for arm in ARMS:
        for lv, level in LEVELS.items():
            cond, sig = f"{arm}_fleet_{lv}_r0", f"sig_fleet_{lv}_r0"
            if (cond, "L0", k.SIX[0]) not in lookup:
                continue
            for meth in METHODS:
                draws = []
                for site in k.SIX:
                    a, s = lookup[(cond, meth, site)], lookup[(sig, meth, site)]
                    loss = (s["centre"] - a["centre"]) * 100
                    d = (
                        rng.normal(s["centre"], s["sd"], N_DRAWS)
                        - rng.normal(a["centre"], a["sd"], N_DRAWS)
                    ) * 100
                    draws.append(d)
                    out.append({"arm": arm, "level": level, "method": meth, "site": site,
                                "signed": s["centre"], "arm_rec": a["centre"], "loss_pp": loss,
                                "lo": float(np.percentile(d, 10)), "hi": float(np.percentile(d, 90))})  # fmt: skip
                med_draws = np.median(np.stack(draws), axis=0)
                per_site = [
                    r["loss_pp"]
                    for r in out
                    if r["arm"] == arm and r["level"] == level and r["method"] == meth
                ]
                out.append({"arm": arm, "level": level, "method": meth, "site": "median of six",
                            "signed": float(np.median([r["signed"] for r in out[-6:]])),
                            "arm_rec": float(np.median([r["arm_rec"] for r in out[-6:]])),
                            "loss_pp": float(np.median(np.array(per_site))),
                            "lo": float(np.percentile(med_draws, 10)), "hi": float(np.percentile(med_draws, 90))})  # fmt: skip
    return pl.DataFrame(out)


def fig_loss(loss: pl.DataFrame) -> None:
    """Draw the conversion loss at 0.5 times by reactive model and probe."""
    d = loss.filter(pl.col("level") == 0.5).with_columns(
        arm_label=pl.col("arm").replace(SHORT),
        probe=pl.col("method").replace(
            {"L0": "Linear (L0)", "L1": "One-sided linear (L1)", "FA": "Fold model (FA)"}
        ),
    )
    order = list(SHORT.values())
    colour = alt.Color(
        "probe:N",
        title="Probe",
        scale=alt.Scale(
            domain=["Linear (L0)", "One-sided linear (L1)", "Fold model (FA)"],
            range=[theme.BRAND_ORANGE, theme.DATA_GREEN, theme.DATA_BLUE],
        ),
    )
    x = alt.X(
        "arm_label:N",
        title="Reactive-power model of the apparent-power world",
        sort=order,
        axis=alt.Axis(labelAngle=0),
    )
    off = alt.XOffset("probe:N")
    med = d.filter(pl.col("site") == "median of six")
    pts = (
        alt.Chart(d.filter(pl.col("site") != "median of six"))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=22, opacity=0.35, aria=False)
        .encode(
            x=x,
            y=alt.Y("loss_pp:Q", title="Conversion loss (percentage points of recovery)"),
            color=colour,
            xOffset=off,
        )
    )
    bars = (
        alt.Chart(med)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2.5)
        .encode(x=x, y="lo:Q", y2="hi:Q", color=colour, xOffset=off)
    )
    diamonds = (
        alt.Chart(med)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=110, shape="diamond", aria=False)
        .encode(x=x, y="loss_pp:Q", color=colour, xOffset=off)
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [0.0, 5.0, 12.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.2, color=theme.BLACK_2)
        .encode(y="y:Q")
    )
    chart = (rules + pts + bars + diamonds).properties(
        width=640,
        height=320,
        title=alt.TitleParams(
            "Conversion loss at 0.5 times the midday reading, by reactive-power model and probe",
            subtitle=[
                "2026-06 networks, 3 seeds. Faint dot: one MW Primary. Diamond: median of the six. Bar: 80% interval.",
                "Dashed lines: 0, 5, and 12 points (the decision thresholds).",
            ],
            anchor="start", fontSize=13, subtitleFontSize=11,
        ),
    )  # fmt: skip
    chart.save(DOCS / "conversion_loss_at_half_level.svg")


def main() -> None:
    """Print every number the write-up quotes and draw the figure."""
    k.WORK = run.WORK
    DOCS.mkdir(parents=True, exist_ok=True)
    res = an.load(cutoffs=("2026-06",))
    rec = an.recoveries(res=res.filter(pl.col("seed").is_in(run.SEEDS)))
    rec.drop("diff_boot", strict=False).write_csv(run.WORK / "recoveries.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    six = rec.filter(pl.col("site").is_in(k.SIX))
    print("== Median over the six of the median-over-seeds recovery in MW ==")
    per = six.group_by("condition", "method", "site").agg(r=pl.col("rec").median())
    print(
        per.group_by("condition", "method")
        .agg(r=pl.col("r").median())
        .pivot(on="method", index="condition", values="r")
        .sort("condition")
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print("== Uninjected-twin read check: seeds per condition ==")
    print(six.group_by("condition").agg(n_seeds=pl.col("seed").n_unique()).sort("condition"))
    stats = site_stats(rec)
    loss = loss_table(stats)
    loss.write_csv(run.WORK / "conversion_loss.csv")
    print("== Conversion loss (percentage points): median of six, 80% interval ==")
    print(
        loss.filter(pl.col("site") == "median of six")
        .select(
            "arm",
            "level",
            "method",
            pl.col("signed").round(3),
            pl.col("arm_rec").round(3),
            pl.col("loss_pp", "lo", "hi").round(1),
        )
        .sort("level", "method", "arm")
    )
    print("== Conversion loss per Primary at 0.5 times, L0 ==")
    print(
        loss.filter((pl.col("level") == 0.5) & (pl.col("method") == "L0"))
        .pivot(on="arm", index="site", values="loss_pp")
        .with_columns(pl.col(pl.Float64).round(1))
    )
    energy_path = run.WORK / "energy_factor.csv"
    if energy_path.exists():
        ef = pl.read_csv(energy_path)
        print("== Energy factor per Primary (energy recovery = MW recovery times factor) ==")
        print(ef.with_columns(pl.col("energy_factor").round(3)))
        fac = dict(zip(ef["site"], ef["energy_factor"], strict=True))
        e = (
            loss.filter((pl.col("site") != "median of six") & (pl.col("level") == 0.5))
            .with_columns(
                signed_e=pl.col("signed") * pl.col("site").replace_strict(fac, default=None),
                arm_e=pl.col("arm_rec") * pl.col("site").replace_strict(fac, default=None),
            )
            .with_columns(loss_e=(pl.col("signed_e") - pl.col("arm_e")) * 100)
        )
        print("== Recovery in energy at 0.5 times, median of six; loss in energy points ==")
        print(
            e.group_by("arm", "method")
            .agg(
                signed_e=pl.col("signed_e").median(),
                arm_e=pl.col("arm_e").median(),
                loss_e=pl.col("loss_e").median(),
            )
            .sort("method", "arm")
            .with_columns(pl.col(pl.Float64).round(3))
        )
    fig_loss(loss)


if __name__ == "__main__":
    main()
