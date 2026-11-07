# ruff: noqa: E501
"""Experiment 621: pass or fail of every bar, and the figures.

Reads `work/probes/<cutoff>/<condition>.parquet` (from `e00621_probe.py`), pairs every injected
network with the uninjected twin of the same cutoff, seed, Primary and regression, and writes
`work/summary.json`, `work/recoveries.csv`, and the figures next to the result page.
"""

import json
from pathlib import Path

import altair as alt
import e00621_common as k
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00621-pv-probe-injected-known-answer-and-placebo-test"
)
Z80 = 1.2816
BAND = (0.85, 1.15)
WORLD_LABEL = {
    "farm": "Real farm",
    "phys": "Physics curve (easy world)",
    "profile": "Profile-only placebo",
    "shuffled": "Shuffled-sky placebo",
    "cloud": "Cloud-following load",
}
WORLD_COLOUR = {
    "farm": theme.DATA_GREEN,
    "phys": theme.DATA_BLUE,
    "profile": theme.DATA_PURPLE,
    "shuffled": theme.DATA_SKY,
    "cloud": theme.BRAND_ORANGE,
}


def load() -> pl.DataFrame:
    """Return every probe result with its injected megawatts and world."""
    frames = [pl.read_parquet(f) for f in sorted((k.WORK / "probes").glob("*/*.parquet"))]
    df = pl.concat(frames)
    cond = {n: k.parse(name=n) for n in df["condition"].unique().to_list()}
    return df.with_columns(
        world=pl.col("condition").replace_strict({n: c.kind for n, c in cond.items()}),
        mw=pl.col("condition").replace_strict({n: c.mw for n, c in cond.items()}),
        rotation=pl.col("condition").replace_strict({n: c.rotation for n, c in cond.items()}),
        share=pl.col("condition").replace_strict({n: c.share for n, c in cond.items()}),
    )


def recoveries(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per injected network: the capacity difference to its twin, with intervals."""
    ctl = df.filter(pl.col("world") == "control").select(
        "cutoff", "seed", "site", "model", ctl=pl.col("capacity_mw"), ctl_boot=pl.col("boot")
    )
    inj = df.filter(pl.col("world") != "control").join(ctl, on=["cutoff", "seed", "site", "model"])
    delta = inj["capacity_mw"].to_numpy() - inj["ctl"].to_numpy()
    boot = np.array(inj["boot"].to_list()) - np.array(inj["ctl_boot"].to_list())
    out = inj.select(
        "cutoff", "condition", "world", "mw", "rotation", "share", "seed", "site", "model"
    ).with_columns(
        delta_mw=pl.Series(delta),
        boot_sd=pl.Series(boot.std(axis=1)),
        boot_lo=pl.Series(np.percentile(boot, 10, axis=1)),
        boot_hi=pl.Series(np.percentile(boot, 90, axis=1)),
    )
    # across-seed spread of the difference within a cutoff, Primary, condition, and regression
    spread = (
        out.group_by("cutoff", "condition", "site", "model")
        .agg(seed_sd=pl.col("delta_mw").std())
        .fill_null(0.0)
    )
    out = out.join(spread, on=["cutoff", "condition", "site", "model"])
    out = out.with_columns(
        recovery=pl.col("delta_mw") / pl.col("mw"),
        half_width=Z80 * (pl.col("boot_sd") ** 2 + pl.col("seed_sd") ** 2).sqrt(),
    ).with_columns(
        lo=pl.col("delta_mw") - pl.col("half_width"),
        hi=pl.col("delta_mw") + pl.col("half_width"),
    )
    truth = (
        pl.when(pl.col("world") == "cloud")
        .then(0.0)
        .when(pl.col("world").is_in(["profile", "shuffled"]))
        .then(0.0)
        .otherwise(pl.col("mw"))
    )
    return out.with_columns(truth_mw=truth).with_columns(
        covered_wide=(pl.col("lo") <= pl.col("truth_mw")) & (pl.col("truth_mw") <= pl.col("hi")),
        covered_boot=(pl.col("boot_lo") <= pl.col("truth_mw"))
        & (pl.col("truth_mw") <= pl.col("boot_hi")),
    )


def bars(rec: pl.DataFrame) -> dict:
    """Return the pass or fail of every pre-registered bar."""
    res: dict = {}
    sites = list(k.SIX)

    def per_site(world: str, mw: float, model: str, agg: str) -> pl.DataFrame:
        sel = rec.filter(
            (pl.col("world") == world) & (pl.col("mw") == mw) & (pl.col("model") == model)
        )
        q = pl.col("recovery").median() if agg == "median" else pl.col("recovery").quantile(0.95)
        return sel.group_by("site").agg(v=q).sort("site")

    for mw, need in ((4.0, 5), (2.0, 4)):
        t = per_site("farm", mw, "plain", "median")
        inside = int(((t["v"] >= BAND[0]) & (t["v"] <= BAND[1])).sum())
        res[f"farm_{int(mw)}mw"] = {
            "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "inside_band": inside,
            "need": need,
            "pass": inside >= need,
        }
    for mw in (4.0, 2.0):
        t = per_site("phys", mw, "plain", "median")
        res[f"phys_{int(mw)}mw"] = {
            "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True))
        }
    for world in ("profile", "shuffled"):
        t = per_site(world, 4.0, "plain", "p95")
        below = int((t["v"] < 0.20).sum())
        res[f"placebo_{world}"] = {
            "p95_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "below_0.20": below,
            "need": 5,
            "pass": below >= 5,
        }
    for model in ("plain", "cloud_tpl"):
        for cond in ("cloudlin_weak", "cloudsq_weak", "cloudlin_strong", "cloudsq_strong"):
            sel = rec.filter((pl.col("condition") == cond) & (pl.col("model") == model))
            t = sel.group_by("site").agg(v=(pl.col("delta_mw") / 2.0).median()).sort("site")
            below = int((t["v"] < 0.15).sum())
            res[f"cloud_{cond}_{model}"] = {
                "fake_share_of_2mw_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
                "below_0.15": below,
                "need": 5,
                "pass": below >= 5 if model == "cloud_tpl" else None,
            }
    for label, sel in (
        (
            "coverage_phys_easy_world",
            rec.filter((pl.col("world") == "phys") & (pl.col("model") == "plain")),
        ),
        (
            "coverage_cloud_linear_weak",
            rec.filter((pl.col("condition") == "cloudlin_weak") & (pl.col("model") == "cloud_tpl")),
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
            "covered_with_seed_spread": round(float(sel["covered_wide"].mean()), 3),  # ty: ignore[invalid-argument-type]
            "covered_bootstrap_only": round(float(sel["covered_boot"].mean()), 3),  # ty: ignore[invalid-argument-type]
            "pass": bool(float(sel["covered_wide"].mean()) >= 0.70),  # ty: ignore[invalid-argument-type]
        }
    res["_sites"] = sites
    return res


def figures(rec: pl.DataFrame) -> None:
    """Write the SVG figures."""
    order = list(k.SIX)
    # 1. headline: recovery by world and Primary, dot = median, line = range over cutoffs, seeds, rotations
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
        for site, g in sel.group_by("site"):
            v = g["recovery"].to_numpy()
            rows.append(
                {
                    "panel": label,
                    "world": world,
                    "site": site[0],
                    "med": float(np.median(v)),
                    "lo": float(v.min()),
                    "hi": float(v.max()),
                }
            )
    df = pl.DataFrame(rows)
    panels = []
    for _, _, label in worlds:
        d = df.filter(pl.col("panel") == label)
        colour = WORLD_COLOUR[d["world"][0]]
        base = alt.Chart(d).encode(y=alt.Y("site:N", sort=order, title=None))
        band = (
            alt.Chart(pl.DataFrame({"a": [BAND[0]], "b": [BAND[1]]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color="grey")
            .encode(x="a:Q", x2="b:Q")
        )
        rule = base.mark_rule(color=colour).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", scale=alt.Scale(domain=[-0.4, 1.6]), title=None), x2="hi:Q"
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60, color=colour).encode(x="med:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(band, rule, dot).properties(
                width=560, height=110, title=alt.TitleParams(label, anchor="start", fontSize=12)
            )
        )  # ty: ignore[invalid-argument-type]
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Recovery of the injected PV capacity at the six MW Primaries (1 is exact; grey band is the 0.85 to 1.15 pass bar)",
                "Dot: median over 2 cutoffs, 3 seeds and 2 farm rotations. Line: range of those.",
                "Recovery is the probed capacity minus the uninjected twin's, over the injected MW.",
            ],
            anchor="start",
            fontSize=11,
        )
    )
    chart.save(str(DOCS / "recovery_by_world.svg"))
    # 2. recovered vs injected megawatts for every network
    sel = rec.filter(
        pl.col("world").is_in(["farm", "phys", "profile", "shuffled"])
        & (pl.col("model") == "plain")
    )
    pts = sel.with_columns(world_label=pl.col("world").replace_strict(WORLD_LABEL))
    dom = [WORLD_LABEL[w] for w in ("farm", "phys", "profile", "shuffled")]
    rng = [WORLD_COLOUR[w] for w in ("farm", "phys", "profile", "shuffled")]
    diag = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_line(color="grey", strokeDash=[4, 3])
        .transform_calculate(ideal="datum.mw")
        .encode(x="mw:Q", y="ideal:Q")
    )
    sc = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=26, opacity=0.7)
        .encode(
            x=alt.X(
                "mw:Q",
                title="Injected AC capacity (MW; placebos: a 4 MW farm's shape)",
                scale=alt.Scale(domain=[0, 4.5]),
            ),
            y=alt.Y("delta_mw:Q", title="Probed minus twin (MW)"),
            color=alt.Color(
                "world_label:N",
                scale=alt.Scale(domain=dom, range=rng),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=560, height=150)
    )  # ty: ignore[unresolved-attribute]
    fig2 = (
        alt.layer(diag, sc)
        .facet(row=alt.Row("site:N", sort=order, title=None), data=pts)
        .properties(
            title=alt.TitleParams(
                [
                    "Probed against injected capacity, every network, at each Primary",
                    "Dashed line: exact recovery. Each dot is one network (cutoff, seed, farm rotation).",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    fig2.save(str(DOCS / "recovered_vs_injected.svg"))
    # 3. cloud-following load: spurious capacity with and without the template
    cl = rec.filter(pl.col("world") == "cloud").with_columns(
        form=pl.col("condition")
        .str.replace("cloudlin", "linear")
        .str.replace("cloudsq", "squared")
        .str.replace("_", ", ")
        + " load",
        regression=pl.col("model").replace_strict(
            {"plain": "No cloud template", "cloud_tpl": "With cloud template"}
        ),
        fake=pl.col("delta_mw") / 2.0,
    )
    box = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_boxplot(extent="min-max", size=10)
        .encode(
            x=alt.X(
                "fake:Q",
                title="Spurious capacity, share of a 2 MW farm (smaller is better; dashed: 0.15 bar)",
            ),
            y=alt.Y("regression:N", title=None, axis=alt.Axis(labels=False)),
            color=alt.Color(
                "regression:N",
                scale=alt.Scale(
                    domain=["No cloud template", "With cloud template"],
                    range=[theme.BRAND_ORANGE, theme.DATA_BLUE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=560, height=70)
    )  # ty: ignore[unresolved-attribute]
    bar = alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(0.15))  # ty: ignore[unresolved-attribute]
    fig3 = (
        alt.layer(box, bar)
        .facet(
            row=alt.Row("form:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            data=cl,
        )
        .properties(  # ty: ignore[invalid-argument-type]
            title=alt.TitleParams(
                [
                    "How much of a cloud-following load the probe reads as PV",
                    "Each box covers the six Primaries, 2 cutoffs and 3 seeds; whiskers are the minimum and maximum.",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    fig3.save(str(DOCS / "cloud_load_faking.svg"))


def main() -> None:
    """Write the summary, the recovery table, and the figures."""
    df = load()
    rec = recoveries(df)
    rec.drop("boot", strict=False).write_csv(k.WORK / "recoveries.csv")
    summary = bars(rec)
    (k.WORK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    figures(rec)


if __name__ == "__main__":
    main()
