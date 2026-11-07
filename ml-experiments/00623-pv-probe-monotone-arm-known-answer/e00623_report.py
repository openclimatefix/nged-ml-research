# ruff: noqa: E501
"""Experiment 623: pass or fail of every relaxed bar for the plain and monotone arms, and the figures.

Reads the probe results of both arms (`probes/<cutoff>/<condition>.parquet`; the plain arm's are
experiment 621's, in its private folder), pairs every injected network with the uninjected twin of
the same arm, cutoff, seed, Primary and regression, and writes `work/summary.json`,
`work/recoveries.csv`, and the figures next to the result page.
"""

import json
import math
from pathlib import Path

import altair as alt
import e00623_common as k
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00623-pv-probe-monotone-arm-known-answer"
)
Z80 = 1.2816
BAND = (0.80, 1.20)
NEED_PRIMARIES = 4
PLACEBO_LIMIT = 0.25
CLOUD_LIMIT = 0.20
COVERAGE_MIN = 0.60
STOP_RECOVERY = 0.70
STOP_PLACEBO = 0.30
ARMS = {"plain": k.PLAIN_WORK, "monotone": k.WORK}
ARM_LABEL = {"plain": "Plain network", "monotone": "Monotone network"}
ARM_COLOUR = {"plain": theme.DATA_BLUE, "monotone": theme.BRAND_ORANGE}
WORLD_LABEL = {
    "farm": "Real farm",
    "phys": "Physics curve (easy world)",
    "profile": "Profile-only placebo",
    "shuffled": "Shuffled-sky placebo",
    "cloud": "Cloud-following load",
}


def load() -> pl.DataFrame:
    """Return every probe result of both arms with its injected megawatts and world."""
    frames = []
    for arm, root in ARMS.items():
        frames.extend(
            pl.read_parquet(f)
            .select("cutoff", "condition", "seed", "site", "model", "capacity_mw", "boot")
            .with_columns(arm=pl.lit(arm))
            for f in sorted((root / "probes").glob("*/*.parquet"))
        )
    df = pl.concat(frames)
    cond = {n: k.parse(name=n) for n in df["condition"].unique().to_list()}
    return df.with_columns(
        world=pl.col("condition").replace_strict({n: c.kind for n, c in cond.items()}),
        mw=pl.col("condition").replace_strict({n: c.mw for n, c in cond.items()}),
        rotation=pl.col("condition").replace_strict({n: c.rotation for n, c in cond.items()}),
    )


def recoveries(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per injected network: the capacity difference to its twin, with intervals."""
    keys = ["arm", "cutoff", "seed", "site", "model"]
    ctl = df.filter(pl.col("world") == "control").select(
        *keys, ctl=pl.col("capacity_mw"), ctl_boot=pl.col("boot")
    )
    inj = df.filter(pl.col("world") != "control").join(ctl, on=keys)
    delta = inj["capacity_mw"].to_numpy() - inj["ctl"].to_numpy()
    boot = np.array(inj["boot"].to_list()) - np.array(inj["ctl_boot"].to_list())
    out = inj.select(
        "arm", "cutoff", "condition", "world", "mw", "rotation", "seed", "site", "model"
    ).with_columns(
        delta_mw=pl.Series(delta),
        boot_sd=pl.Series(boot.std(axis=1)),
        boot_lo=pl.Series(np.percentile(boot, 10, axis=1)),
        boot_hi=pl.Series(np.percentile(boot, 90, axis=1)),
    )
    spread = (
        out.group_by("arm", "cutoff", "condition", "site", "model")
        .agg(seed_sd=pl.col("delta_mw").std())
        .fill_null(0.0)
    )
    out = out.join(spread, on=["arm", "cutoff", "condition", "site", "model"])
    out = out.with_columns(
        recovery=pl.col("delta_mw") / pl.col("mw"),
        half_width=Z80 * (pl.col("boot_sd") ** 2 + pl.col("seed_sd") ** 2).sqrt(),
    ).with_columns(
        lo=pl.col("delta_mw") - pl.col("half_width"), hi=pl.col("delta_mw") + pl.col("half_width")
    )
    truth = (
        pl.when(pl.col("world").is_in(["cloud", "profile", "shuffled"]))
        .then(0.0)
        .otherwise(pl.col("mw"))
    )
    return out.with_columns(truth_mw=truth).with_columns(
        covered_wide=(pl.col("lo") <= pl.col("truth_mw")) & (pl.col("truth_mw") <= pl.col("hi")),
        covered_boot=(pl.col("boot_lo") <= pl.col("truth_mw"))
        & (pl.col("truth_mw") <= pl.col("boot_hi")),
    )


def bars(rec: pl.DataFrame, arm: str) -> dict:
    """Return the pass or fail of every relaxed bar for one arm."""
    rec = rec.filter(pl.col("arm") == arm)
    res: dict = {}

    def per_site(world: str, mw: float, agg: str) -> pl.DataFrame:
        sel = rec.filter(
            (pl.col("world") == world) & (pl.col("mw") == mw) & (pl.col("model") == "plain")
        )
        q = pl.col("recovery").median() if agg == "median" else pl.col("recovery").quantile(0.95)
        return sel.group_by("site").agg(v=q).sort("site")

    for mw in (4.0, 2.0):
        t = per_site("farm", mw, "median")
        inside = int(((t["v"] >= BAND[0]) & (t["v"] <= BAND[1])).sum())
        res[f"farm_{int(mw)}mw"] = {
            "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "inside_band": inside,
            "need": NEED_PRIMARIES,
            "pass": inside >= NEED_PRIMARIES,
            "median_of_medians": round(float(t["v"].median()), 3),
        }
    t = per_site("phys", 4.0, "median")
    res["phys_4mw"] = {
        "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
        "median_of_medians": round(float(t["v"].median()), 3),
    }
    for world in ("profile", "shuffled"):
        t = per_site(world, 4.0, "p95")
        below = int((t["v"] < PLACEBO_LIMIT).sum())
        res[f"placebo_{world}"] = {
            "p95_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "below_limit": below,
            "need": NEED_PRIMARIES,
            "pass": below >= NEED_PRIMARIES,
            "max_p95": round(float(t["v"].max()), 3),
        }
    for model in ("plain", "cloud_tpl"):
        for cond in ("cloudlin_weak", "cloudsq_weak", "cloudlin_strong", "cloudsq_strong"):
            sel = rec.filter((pl.col("condition") == cond) & (pl.col("model") == model))
            t = sel.group_by("site").agg(v=(pl.col("delta_mw") / 2.0).median()).sort("site")
            below = int((t["v"] < CLOUD_LIMIT).sum())
            res[f"cloud_{cond}_{model}"] = {
                "fake_share_of_2mw_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
                "below_limit": below,
                "pass": (below >= NEED_PRIMARIES) if model == "cloud_tpl" else None,
            }
    for label, sel in (
        (
            "coverage_phys_easy_world",
            rec.filter((pl.col("world") == "phys") & (pl.col("model") == "plain")),
        ),
        (
            "coverage_cloud_linear_strong",
            rec.filter(
                (pl.col("condition") == "cloudlin_strong") & (pl.col("model") == "cloud_tpl")
            ),
        ),
        (
            "coverage_real_farm",
            rec.filter((pl.col("world") == "farm") & (pl.col("model") == "plain")),
        ),
    ):
        res[label] = {
            "n": sel.height,
            "covered_with_seed_spread": round(float(sel["covered_wide"].mean()), 3),
            "covered_bootstrap_only": round(float(sel["covered_boot"].mean()), 3),
            "pass": bool(float(sel["covered_wide"].mean()) >= COVERAGE_MIN),
        }
    # the bar says "the injected capacity": use the real-farm and physics worlds together
    pooled = rec.filter(pl.col("world").is_in(["farm", "phys"]) & (pl.col("model") == "plain"))
    res["coverage_injections"] = {
        "n": pooled.height,
        "covered_with_seed_spread": round(float(pooled["covered_wide"].mean()), 3),
        "pass": bool(float(pooled["covered_wide"].mean()) >= COVERAGE_MIN),
    }
    # seed spread of the probed difference (MW), median over cells
    sd = (
        rec.filter(pl.col("world").is_in(["farm", "phys"]) & (pl.col("model") == "plain"))
        .group_by("site")
        .agg(seed_sd=pl.col("seed_sd").median())
        .sort("site")
    )
    res["seed_sd_mw_by_site"] = dict(zip(sd["site"], sd["seed_sd"].round(3), strict=True))
    res["stop"] = {
        "median_recovery_4mw": res["farm_4mw"]["median_of_medians"],
        "max_placebo_p95": max(
            res["placebo_profile"]["max_p95"], res["placebo_shuffled"]["max_p95"]
        ),
    }
    return res


def seed_sd_of_capacity(df: pl.DataFrame) -> dict:
    """Return the across-seed standard deviation (MW) of the probed capacity of the uninjected networks."""
    out: dict = {}
    for arm in ARMS:
        t = (
            df.filter(
                (pl.col("arm") == arm)
                & (pl.col("world") == "control")
                & (pl.col("model") == "plain")
            )
            .group_by("cutoff", "site")
            .agg(sd=pl.col("capacity_mw").std())
            .group_by("site")
            .agg(sd=pl.col("sd").mean())
            .sort("site")
        )
        out[arm] = dict(zip(t["site"], t["sd"].round(3), strict=True))
    return out


def skill() -> dict:
    """Return the validation-row CRPS of the control networks, by arm and cutoff (mean over seeds)."""
    out: dict = {}
    for cutoff in k.CUTOFFS:
        f = k.WORK / f"skill_{cutoff}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for arm in ARMS:
            for scope in ("all", "six"):
                v = [d[f"{arm}_s{s}"][f"crps_mw_{scope}"] for s in k.SEEDS]
                out[f"{cutoff}_{arm}_{scope}"] = {"mean": float(np.mean(v)), "per_seed": v}
    nll = {}
    for arm, root in ARMS.items():
        for cutoff in k.CUTOFFS:
            v = [
                json.loads(f.read_text())["val_nll"]
                for f in (root / "logs" / cutoff).glob("control_s*.json")
            ]
            if v:
                nll[f"{cutoff}_{arm}"] = float(np.mean(v))
    out["val_nll_control"] = nll
    return out


def figures(rec: pl.DataFrame, df: pl.DataFrame) -> None:
    """Write the SVG figures."""
    order = list(k.SIX)
    arm_domain, arm_range = list(ARM_LABEL.values()), [ARM_COLOUR[a] for a in ARM_LABEL]
    worlds = [
        ("farm", 2.0, "Real farm, 2 MW"), ("farm", 4.0, "Real farm, 4 MW"),
        ("phys", 2.0, "Physics curve, 2 MW"), ("phys", 4.0, "Physics curve, 4 MW"),
        ("profile", 4.0, "Profile-only placebo, 4 MW"), ("shuffled", 4.0, "Shuffled-sky placebo, 4 MW"),
    ]  # fmt: skip
    rows = []
    for world, mw, label in worlds:
        sel = rec.filter(
            (pl.col("world") == world) & (pl.col("mw") == mw) & (pl.col("model") == "plain")
        )
        for (arm, site), g in sel.group_by("arm", "site"):
            v = g["recovery"].to_numpy()
            rows.append(
                {
                    "panel": label,
                    "arm": ARM_LABEL[arm],
                    "site": site,
                    "med": float(np.median(v)),
                    "lo": float(v.min()),
                    "hi": float(v.max()),
                }
            )
    d_all = pl.DataFrame(rows)
    panels = []
    for _, _, label in worlds:
        d = d_all.filter(pl.col("panel") == label)
        base = alt.Chart(d).encode(
            y=alt.Y("site:N", sort=order, title=None),
            yOffset=alt.YOffset("arm:N", sort=arm_domain),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=arm_domain, range=arm_range),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        band = (
            alt.Chart(pl.DataFrame({"a": [BAND[0]], "b": [BAND[1]]}))
            .mark_rect(opacity=0.12, color="grey")
            .encode(x="a:Q", x2="b:Q")
        )
        rule = base.mark_rule().encode(
            x=alt.X("lo:Q", scale=alt.Scale(domain=[-0.4, 1.6]), title=None), x2="hi:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=50).encode(x="med:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(band, rule, dot).properties(
                width=560, height=130, title=alt.TitleParams(label, anchor="start", fontSize=12)
            )
        )  # ty: ignore[invalid-argument-type]
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                f"Recovery of injected PV capacity at the six MW Primaries (1 is exact; grey band is the {BAND[0]} to {BAND[1]} pass bar)",
                "Dot: median over 2 cutoffs, 3 seeds and 2 farm rotations. Line: range of those. Recovery is the probed capacity minus the uninjected twin's, over the injected MW.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "recovery_by_world.svg"))
    sel = rec.filter(
        pl.col("world").is_in(["farm", "phys", "profile", "shuffled"])
        & (pl.col("model") == "plain")
    ).with_columns(
        world_label=pl.col("world").replace_strict(WORLD_LABEL),
        arm_label=pl.col("arm").replace_strict(ARM_LABEL),
    )
    wdom = [WORLD_LABEL[w] for w in ("farm", "phys", "profile", "shuffled")]
    wrng = [theme.DATA_GREEN, theme.DATA_BLUE, theme.DATA_PURPLE, theme.DATA_SKY]
    diag = alt.Chart().mark_line(color="grey", strokeDash=[4, 3]).encode(x="mw:Q", y="mw:Q")
    sc = (
        alt.Chart()
        .mark_point(filled=True, size=22, opacity=0.7)
        .encode(
            x=alt.X(
                "mw:Q",
                title="Injected AC capacity (MW; placebos carry a 4 MW farm's shape)",
                scale=alt.Scale(domain=[0, 4.5]),
            ),
            y=alt.Y("delta_mw:Q", title="Probed capacity minus twin (MW)"),
            color=alt.Color(
                "world_label:N",
                scale=alt.Scale(domain=wdom, range=wrng),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=300, height=140)
    )  # ty: ignore[unresolved-attribute]
    alt.layer(diag, sc).facet(
        row=alt.Row("site:N", sort=order, title=None),
        column=alt.Column("arm_label:N", sort=arm_domain, title=None),
        data=sel,
    ).properties(
        title=alt.TitleParams(
            [
                "Probed against injected capacity, every network, at each Primary",
                "Dashed line: exact recovery. Each dot is one network (cutoff, seed, farm rotation).",
            ],
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "recovered_vs_injected.svg"))
    cl = rec.filter(pl.col("world") == "cloud").with_columns(
        form=pl.col("condition")
        .str.replace("cloudlin", "linear")
        .str.replace("cloudsq", "squared")
        .str.replace("_", ", ")
        + " load",
        regression=pl.col("model").replace_strict(
            {"plain": "No cloud template", "cloud_tpl": "With cloud template"}
        ),
        arm_label=pl.col("arm").replace_strict(ARM_LABEL),
        fake=pl.col("delta_mw") / 2.0,
    )
    rule = alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(CLOUD_LIMIT))
    box = (
        alt.Chart()
        .mark_boxplot(extent="min-max", size=9)
        .encode(
            x=alt.X(
                "fake:Q",
                title=f"Spurious capacity as a share of a 2 MW farm (smaller is better; dashed line is the {CLOUD_LIMIT} bar)",
            ),
            y=alt.Y("arm_label:N", sort=arm_domain, title=None, axis=alt.Axis(labels=False)),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=arm_domain, range=arm_range),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=520, height=60)
    )  # ty: ignore[unresolved-attribute]
    alt.layer(box, rule).facet(
        row=alt.Row("form:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        column=alt.Column("regression:N", title=None),
        data=cl,
    ).properties(  # ty: ignore[invalid-argument-type]
        title=alt.TitleParams(
            [
                "How much of a cloud-following load the probe reads as PV",
                "Each box covers the six Primaries, 2 cutoffs and 3 seeds; whiskers are the minimum and maximum.",
            ],
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "cloud_load_faking.svg"))
    # seed spread of the uninjected twins' probed capacity
    sd = (
        df.filter((pl.col("world") == "control") & (pl.col("model") == "plain"))
        .group_by("arm", "cutoff", "site")
        .agg(sd=pl.col("capacity_mw").std())
        .with_columns(arm_label=pl.col("arm").replace_strict(ARM_LABEL))
    )
    alt.Chart(sd).mark_bar().encode(
        y=alt.Y("site:N", sort=order, title=None),
        x=alt.X("sd:Q", title="Across-seed standard deviation of the probed capacity (MW)"),
        yOffset=alt.YOffset("arm_label:N", sort=arm_domain),
        color=alt.Color(
            "arm_label:N",
            scale=alt.Scale(domain=arm_domain, range=arm_range),
            legend=alt.Legend(orient="top", title=None),
        ),
        row=alt.Row("cutoff:N", title="Cutoff"),
    ).properties(
        width=520,
        height=130,
        title=alt.TitleParams(
            "Seed spread of the probe on networks trained with no injection",
            anchor="start",
            fontSize=11,
        ),
    ).save(str(DOCS / "seed_spread.svg"))  # ty: ignore[invalid-argument-type]


def response_figure() -> None:
    """Plot the probed response of the 4 MW real-farm and physics worlds against the unit curve, both arms."""
    frames = []
    for arm, sub in (("plain", "probes_plain_rerun"), ("monotone", "probes")):
        for f in sorted((k.WORK / sub).glob("*/*.parquet")):
            d = pl.read_parquet(f)
            if "resp_y" not in d.columns:
                continue
            frames.append(
                d.filter(pl.col("model") == "plain")
                .select("cutoff", "condition", "seed", "site", "resp_y", "resp_u")
                .with_columns(arm=pl.lit(arm))
            )
    if not frames:
        return
    df = pl.concat(frames)
    ctl = df.filter(pl.col("condition") == "control").select(
        "arm", "cutoff", "seed", "site", ctl=pl.col("resp_y")
    )
    rows = []
    for r in (
        df.filter(pl.col("condition").is_in(["phys_mw4", "farm_mw4_r0", "farm_mw4_r1"]))
        .join(ctl, on=["arm", "cutoff", "seed", "site"])
        .iter_rows(named=True)
    ):
        for u, y, c0 in zip(r["resp_u"], r["resp_y"], r["ctl"], strict=True):
            if all(map(math.isfinite, (u, y, c0))):
                rows.append(
                    {
                        "arm": ARM_LABEL[r["arm"]],
                        "site": r["site"],
                        "u": u,
                        "diff": y - c0,
                        "world": "Physics curve" if r["condition"] == "phys_mw4" else "Real farm",
                    }
                )
    d = (
        pl.DataFrame(rows)
        .group_by("arm", "site", "world", pl.col("u").round(2))
        .agg(diff=pl.col("diff").mean())
    )
    offset = float((d["diff"] + 4.0 * d["u"]).mean())
    d = d.with_columns(ideal=-4.0 * pl.col("u") + offset)
    ideal = alt.Chart().mark_line(color="grey", strokeDash=[4, 3]).encode(x="u:Q", y="ideal:Q")
    pts = (
        alt.Chart()
        .mark_line(point=True)
        .encode(
            x=alt.X("u:Q", title="Unit PV curve of the donor weather (share of capacity)"),
            y=alt.Y("diff:Q", title="Injected minus twin (MW)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(ARM_LABEL.values()), range=[ARM_COLOUR[a] for a in ARM_LABEL]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=240, height=120)
    )  # ty: ignore[unresolved-attribute]
    alt.layer(ideal, pts).facet(
        row=alt.Row("site:N", sort=list(k.SIX), title=None),
        column=alt.Column("world:N", title=None),
        data=d,
    ).properties(
        title=alt.TitleParams(
            [
                "Probed response against the unit PV curve, 4 MW injections",
                "Dashed line: the slope of exactly 4 MW of injected PV, set to the same average level as the probed lines. Mean over cutoffs, seeds, and farm rotations; centred within each probe cell.",
            ],
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "probed_response_vs_unit_curve.svg"))


def main() -> None:
    """Write the summary, the recovery table, and the figures."""
    df = load()
    rec = recoveries(df)
    rec.write_csv(k.WORK / "recoveries.csv")
    summary = {arm: bars(rec, arm) for arm in ARMS}
    summary["seed_sd_of_uninjected_capacity_mw"] = seed_sd_of_capacity(df)
    summary["skill"] = skill()
    (k.WORK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    figures(rec, df)
    response_figure()


if __name__ == "__main__":
    main()
