# ruff: noqa: E501
"""Experiment 624: bars for arm (a) against the plain network, the twin responses of arms (a), (b), (c), and figures.

Reads probe results (`probes_<family>/<cutoff>/<condition>.parquet`), experiment 621's plain probes,
and `skill_<cutoff>.json`; writes `work/summary.json`, `work/recoveries.csv`, and the figures.
"""

import json
import math
from pathlib import Path

import altair as alt
import e00623_report as r623  # ty: ignore[unresolved-import]
import e00624_common as k
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00624-pv-probe-contrastive-injection-arms"
)
ARM_LABEL = {"plain": "Plain network", "a": "Contrastive loss (a)"}
ARM_COLOUR = {"plain": theme.DATA_BLUE, "a": theme.BRAND_ORANGE}
BAND, NEED, PLACEBO, CLOUD, COVER = (
    r623.BAND,
    r623.NEED_PRIMARIES,
    r623.PLACEBO_LIMIT,
    r623.CLOUD_LIMIT,
    r623.COVERAGE_MIN,
)
KEEP = set(k.CONDITIONS)


def load() -> pl.DataFrame:
    """Return the real-vector probe results of the plain network and arm (a), with worlds."""
    frames = [
        pl.read_parquet(f)
        .filter(pl.col("condition").is_in(KEEP))
        .select("cutoff", "condition", "seed", "site", "model", "capacity_mw", "boot")
        .with_columns(arm=pl.lit("plain"))
        for f in sorted((k.k.PLAIN_WORK / "probes").glob("*/*.parquet"))
    ]
    frames += [
        pl.read_parquet(f)
        .filter(pl.col("emb_kind") == "real")
        .select("cutoff", "condition", "seed", "site", "model", "capacity_mw", "boot")
        .with_columns(arm=pl.lit("a"))
        for f in sorted((k.WORK / "probes_a").glob("*/*.parquet"))
    ]
    df = pl.concat(frames)
    cond = {n: k.k.parse(name=n) for n in df["condition"].unique().to_list()}
    return df.with_columns(
        world=pl.col("condition").replace_strict({n: c.kind for n, c in cond.items()}),
        mw=pl.col("condition").replace_strict({n: c.mw for n, c in cond.items()}),
        rotation=pl.col("condition").replace_strict({n: c.rotation for n, c in cond.items()}),
    )


def bars(rec: pl.DataFrame, arm: str) -> dict:
    """Return the pass or fail of every bar for one arm."""
    rec = rec.filter(pl.col("arm") == arm)
    res: dict = {}
    t = (
        rec.filter((pl.col("world") == "farm") & (pl.col("model") == "plain"))
        .group_by("site")
        .agg(v=pl.col("recovery").median())
        .sort("site")
    )
    inside = int(((t["v"] >= BAND[0]) & (t["v"] <= BAND[1])).sum())
    res["farm_4mw"] = {
        "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
        "inside_band": inside,
        "pass": inside >= NEED,
        "median_of_medians": round(float(t["v"].median()), 3),
    }
    t = (
        rec.filter((pl.col("world") == "phys") & (pl.col("model") == "plain"))
        .group_by("site")
        .agg(v=pl.col("recovery").median())
        .sort("site")
    )
    res["phys_4mw"] = {
        "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
        "median_of_medians": round(float(t["v"].median()), 3),
    }
    for world in ("profile", "shuffled"):
        t = (
            rec.filter((pl.col("world") == world) & (pl.col("model") == "plain"))
            .group_by("site")
            .agg(v=pl.col("recovery").quantile(0.95))
            .sort("site")
        )
        below = int((t["v"] < PLACEBO).sum())
        res[f"placebo_{world}"] = {
            "p95_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "below_limit": below,
            "pass": below >= NEED,
            "max_p95": round(float(t["v"].max()), 3),
        }
    for model in ("plain", "cloud_tpl"):
        t = (
            rec.filter((pl.col("world") == "cloud") & (pl.col("model") == model))
            .group_by("site")
            .agg(v=(pl.col("delta_mw") / 2.0).median())
            .sort("site")
        )
        below = int((t["v"] < CLOUD).sum())
        res[f"cloud_strong_{model}"] = {
            "fake_share_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
            "below_limit": below,
            "pass": below >= NEED if model == "cloud_tpl" else None,
        }
    pooled = rec.filter(pl.col("world").is_in(["farm", "phys"]) & (pl.col("model") == "plain"))
    res["coverage_injections"] = {
        "n": pooled.height,
        "covered_with_seed_spread": round(float(pooled["covered_wide"].mean()), 3),
        "pass": bool(float(pooled["covered_wide"].mean()) >= COVER),
    }
    for world in ("farm", "phys"):
        sel = pooled.filter(pl.col("world") == world)
        res[f"coverage_{world}"] = round(float(sel["covered_wide"].mean()), 3)
    return res


def twin_response() -> pl.DataFrame:
    """Return the synthetic +1 MW response (twin minus real vector) of every network of arms a, b, and c."""
    frames = []
    for family in ("a", "b", "c"):
        frames.extend(
            pl.read_parquet(f)
            .filter(pl.col("model") == "plain")
            .select("cutoff", "condition", "seed", "site", "emb_kind", "capacity_mw")
            .with_columns(family=pl.lit(family))
            for f in sorted((k.WORK / f"probes_{family}").glob("*/*.parquet"))
        )
    if not frames:
        return pl.DataFrame()
    d = pl.concat(frames)
    real = d.filter(pl.col("emb_kind") == "real").select(
        "family", "cutoff", "condition", "seed", "site", real=pl.col("capacity_mw")
    )
    return (
        d.filter(pl.col("emb_kind") != "real")
        .join(real, on=["family", "cutoff", "condition", "seed", "site"])
        .with_columns(response=pl.col("capacity_mw") - pl.col("real"))
    )


def skill() -> dict:
    """Return the validation-row CRPS of the control networks, by arm and cutoff."""
    out: dict = {}
    for cutoff in k.CUTOFFS:
        f = k.WORK / f"skill_{cutoff}.json"
        if f.exists():
            d = json.loads(f.read_text())
            for arm in ("plain", "a"):
                for scope in ("all", "six"):
                    out[f"{cutoff}_{arm}_{scope}"] = float(
                        np.mean([d[f"{arm}_s{s}"][f"crps_mw_{scope}"] for s in k.SEEDS])
                    )
    return out


def figures(rec: pl.DataFrame, twin: pl.DataFrame) -> None:
    """Write the SVG figures."""
    order = list(k.k.SIX)
    dom, rng = list(ARM_LABEL.values()), [ARM_COLOUR[a] for a in ARM_LABEL]
    worlds = [
        ("farm", "Real farm, 4 MW"),
        ("phys", "Physics curve, 4 MW"),
        ("profile", "Profile-only placebo, 4 MW"),
        ("shuffled", "Shuffled-sky placebo, 4 MW"),
    ]
    rows = []
    for world, label in worlds:
        sel = rec.filter((pl.col("world") == world) & (pl.col("model") == "plain"))
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
    for _, label in worlds:
        d = d_all.filter(pl.col("panel") == label)
        base = alt.Chart(d).encode(
            y=alt.Y("site:N", sort=order, title=None),
            yOffset=alt.YOffset("arm:N", sort=dom),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=dom, range=rng),
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
        )
        dot = base.mark_point(filled=True, size=50).encode(x="med:Q")
        panels.append(
            alt.layer(band, rule, dot).properties(
                width=560, height=130, title=alt.TitleParams(label, anchor="start", fontSize=12)
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                f"Recovery of injected PV capacity in held-out worlds (1 is exact; grey band is the {BAND[0]} to {BAND[1]} bar)",
                "Dot: median over 2 cutoffs, 3 seeds and (real farm) 2 rotations. Line: range of those.",
            ],
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "recovery_by_world.svg"))
    if twin.height:
        t = (
            twin.filter(pl.col("condition") == "control")
            .group_by("family", "emb_kind", "site")
            .agg(
                med=pl.col("response").median(),
                lo=pl.col("response").min(),
                hi=pl.col("response").max(),
            )
            .with_columns(
                family_label=pl.col("family").replace_strict(
                    {
                        "a": "(a) trained with the loss",
                        "b": "(b) post-hoc check on the plain network",
                        "c": "(c) both",
                    }
                ),
                kind=pl.col("emb_kind").replace_strict(
                    {
                        "twin_phys": "Physics twin (+1 MW of the unit curve)",
                        "twin_prof": "Profile twin (+1 MW, no sky link)",
                    }
                ),
            )
        )
        base = alt.Chart().encode(
            y=alt.Y("site:N", sort=order, title=None),
            yOffset=alt.YOffset("family_label:N"),
            color=alt.Color("family_label:N", legend=alt.Legend(orient="top", title=None)),
        )
        ref = alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(1.0))
        layer = alt.layer(
            base.mark_rule().encode(
                x=alt.X("lo:Q", title="Probed response to the synthetic +1 MW (MW)"), x2="hi:Q"
            ),
            base.mark_point(filled=True, size=50).encode(x="med:Q"),
            ref,
            data=t,
        )
        layer.facet(
            row=alt.Row("kind:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        ).properties(
            title=alt.TitleParams(
                [
                    "Probed response to the synthetic +1 MW PV, uninjected-world networks",
                    "Dashed line: the injected 1 MW. Dot: median over cutoffs and seeds; line: range.",
                ],
                anchor="start",
                fontSize=11,
            )
        ).save(str(DOCS / "twin_response.svg"))
    cl = rec.filter(pl.col("world") == "cloud").with_columns(
        regression=pl.col("model").replace_strict(
            {"plain": "No cloud template", "cloud_tpl": "With cloud template"}
        ),
        arm_label=pl.col("arm").replace_strict(ARM_LABEL),
        fake=pl.col("delta_mw") / 2.0,
    )
    box = (
        alt.Chart()
        .mark_boxplot(extent="min-max", size=12)
        .encode(
            x=alt.X(
                "fake:Q",
                title=f"Spurious capacity as a share of a 2 MW farm (dashed line is the {CLOUD} bar)",
            ),
            y=alt.Y("arm_label:N", sort=dom, title=None, axis=alt.Axis(labels=False)),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=dom, range=rng),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    alt.layer(
        box,
        alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(CLOUD)),
        data=cl,
    ).facet(column=alt.Column("regression:N", title=None)).properties(
        title=alt.TitleParams(
            "Strong linear cloud-following load read as PV; box covers six Primaries, 2 cutoffs, 3 seeds",
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "cloud_load_faking.svg"))


def main() -> None:
    """Write the summary, the recovery table, and the figures."""
    df = load()
    rec = r623.recoveries(df)
    rec.write_csv(k.WORK / "recoveries.csv")
    summary = {arm: bars(rec, arm) for arm in ARM_LABEL}
    twin = twin_response()
    if twin.height:
        t = (
            twin.group_by("family", "emb_kind", "condition")
            .agg(median=pl.col("response").median())
            .group_by("family", "emb_kind")
            .agg(
                median_over_worlds=pl.col("median").median(),
                min=pl.col("median").min(),
                max=pl.col("median").max(),
            )
        )
        summary["twin_response_mw"] = {
            f"{r['family']}_{r['emb_kind']}": [
                round(r["median_over_worlds"], 3),
                round(r["min"], 3),
                round(r["max"], 3),
            ]
            for r in t.iter_rows(named=True)
        }
        ctl = (
            twin.filter(pl.col("condition") == "control")
            .group_by("family", "emb_kind", "site")
            .agg(m=pl.col("response").median())
        )
        summary["twin_response_control_by_site"] = {
            f"{r['family']}_{r['emb_kind']}_{r['site']}": round(r["m"], 3)
            for r in ctl.sort("family", "emb_kind", "site").iter_rows(named=True)
        }
    summary["skill"] = skill()
    nll = {}
    for fam, root in (("plain", k.k.PLAIN_WORK / "logs"), ("a", k.WORK / "logs" / "a")):
        for cutoff in k.CUTOFFS:
            v = [
                json.loads(p.read_text())["val_nll"]
                for p in (root / cutoff).glob("control_s*.json")
            ]
            if v:
                nll[f"{cutoff}_{fam}"] = float(np.mean(v))
    summary["val_nll_control"] = nll
    (k.WORK / "summary.json").write_text(
        json.dumps(
            summary,
            indent=1,
            default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o),
        )
    )
    print(json.dumps(summary, indent=1, default=str))
    DOCS.mkdir(parents=True, exist_ok=True)
    figures(rec, twin)


if __name__ == "__main__":
    main()
