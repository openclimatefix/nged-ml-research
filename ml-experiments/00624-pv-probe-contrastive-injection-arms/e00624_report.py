# ruff: noqa: E501
"""Experiment 624: bars for arm (a) against the plain network, twin responses, and figures.

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
Z80 = r623.Z80
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
        "median_of_medians": round(float(t["v"].median()), 3),  # ty: ignore[invalid-argument-type]
    }
    t = (
        rec.filter((pl.col("world") == "phys") & (pl.col("model") == "plain"))
        .group_by("site")
        .agg(v=pl.col("recovery").median())
        .sort("site")
    )
    res["phys_4mw"] = {
        "median_by_site": dict(zip(t["site"], t["v"].round(3), strict=True)),
        "median_of_medians": round(float(t["v"].median()), 3),  # ty: ignore[invalid-argument-type]
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
            "max_p95": round(float(t["v"].max()), 3),  # ty: ignore[invalid-argument-type]
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
        "covered_with_seed_spread": round(float(pooled["covered_wide"].mean()), 3),  # ty: ignore[invalid-argument-type]
        "pass": bool(float(pooled["covered_wide"].mean()) >= COVER),  # ty: ignore[invalid-argument-type]
    }
    for world in ("farm", "phys"):
        sel = pooled.filter(pl.col("world") == world)
        res[f"coverage_{world}"] = round(float(sel["covered_wide"].mean()), 3)  # ty: ignore[invalid-argument-type]
    return res


def calibration(rec: pl.DataFrame, arm: str, *, model: str = "plain") -> dict:
    """Return the leave-one-farm-out calibration of the 4 MW real-farm recovery, and the coverage it gives.

    A network's farm is `FARMS[(j + rotation) % 4]` for the j-th MW Primary. The factor applied to a
    network is the median recovery of every network whose farm differs. The raw interval is centred
    on the raw probed capacity and widened by the spread (standard deviation) of the per-farm median
    recoveries; the calibrated interval is centred on the probed capacity divided by the factor.
    """
    farms = k.k.FARMS
    sites = list(k.k.SIX)
    d = rec.filter(
        (pl.col("arm") == arm) & (pl.col("world") == "farm") & (pl.col("model") == model)
    ).with_columns(
        farm=pl.struct("site", "rotation").map_elements(
            lambda r: farms[(sites.index(r["site"]) + r["rotation"]) % len(farms)],
            return_dtype=pl.String,
        )
    )
    per_farm = d.group_by("farm").agg(med=pl.col("recovery").median()).sort("farm")
    spread = float(per_farm["med"].std())  # ty: ignore[invalid-argument-type]
    factor = {
        f: float(d.filter(pl.col("farm") != f)["recovery"].median())  # ty: ignore[invalid-argument-type]
        for f in per_farm["farm"]  # ty: ignore[invalid-argument-type]
    }
    d = d.with_columns(factor=pl.col("farm").replace_strict(factor), mw_f=pl.col("mw"))
    bs = (pl.col("boot_sd") ** 2 + pl.col("seed_sd") ** 2).sqrt()
    f = list(factor.values())
    rel = float(np.std(f) / np.median(f))
    d = d.with_columns(
        raw_hw=Z80 * (bs**2 + (pl.col("mw") * spread) ** 2).sqrt(),
        cal_est=pl.col("delta_mw") / pl.col("factor"),
    ).with_columns(
        cal_hw=Z80 * ((bs / pl.col("factor")) ** 2 + (pl.col("cal_est") * rel) ** 2).sqrt()
    )
    raw_cov = (
        (d["delta_mw"] - d["raw_hw"] <= d["mw"]) & (d["mw"] <= d["delta_mw"] + d["raw_hw"])
    ).mean()
    cal_cov = (
        (d["cal_est"] - d["cal_hw"] <= d["mw"]) & (d["mw"] <= d["cal_est"] + d["cal_hw"])
    ).mean()
    return {
        "per_farm_median_recovery": dict(
            zip(per_farm["farm"], per_farm["med"].round(3), strict=True)
        ),
        "leave_one_farm_out_factor": {a: round(b, 3) for a, b in factor.items()},
        "factor_range": [round(min(f), 3), round(max(f), 3)],
        "factors_in_0.55_to_0.85": int(sum(0.55 <= v <= 0.85 for v in f)),
        "n_farms": len(f),
        "calibrated_recovery_median": round(float((d["cal_est"] / d["mw"]).median()), 3),  # ty: ignore[invalid-argument-type]
        "coverage_raw_widened_by_farm_spread": round(float(raw_cov), 3),  # ty: ignore[invalid-argument-type]
        "coverage_calibrated_widened": round(float(cal_cov), 3),  # ty: ignore[invalid-argument-type]
        "n": d.height,
    }


def worlds_summary() -> pl.DataFrame:
    """Return the recovery of every clipped and displaced world network against experiment 621's control."""
    frames = [
        pl.read_parquet(f).select(
            "cutoff", "condition", "seed", "site", "model", "capacity_mw", "boot"
        )
        for f in sorted((k.WORK / "probes_worlds").glob("*/*.parquet"))
    ]
    ctl = [
        pl.read_parquet(f)
        .filter(pl.col("condition") == "control")
        .select("cutoff", "condition", "seed", "site", "model", "capacity_mw", "boot")
        for f in sorted((k.k.PLAIN_WORK / "probes").glob("*/*.parquet"))
    ]
    df = pl.concat(frames + ctl).with_columns(
        arm=pl.lit("plain"),
        world=pl.when(pl.col("condition") == "control")
        .then(pl.lit("control"))
        .otherwise(pl.lit("w")),
        mw=pl.lit(4.0),
        rotation=pl.lit(0),
    )
    return r623.recoveries(df)


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
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
        rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", scale=alt.Scale(domain=[-0.4, 1.6]), title=None), x2="hi:Q"
        )
        dot = base.mark_point(filled=True, size=50).encode(x="med:Q")  # ty: ignore[unresolved-attribute]
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
                        "a": "(a) loss",
                        "b": "(b) post-hoc",
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
        ref = alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(1.0))  # ty: ignore[unresolved-attribute]
        layer = alt.layer(
            base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title="Probed response to the synthetic +1 MW (MW)"), x2="hi:Q"
            ),
            base.mark_point(filled=True, size=50).encode(x="med:Q"),  # ty: ignore[unresolved-attribute]
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
        .encode(  # ty: ignore[unresolved-attribute]
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
        alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(CLOUD)),  # ty: ignore[unresolved-attribute]
        data=cl,
    ).facet(column=alt.Column("regression:N", title=None)).properties(
        title=alt.TitleParams(
            "Strong linear cloud-following load read as PV; box covers six Primaries, 2 cutoffs, 3 seeds",
            anchor="start",
            fontSize=11,
        )
    ).save(str(DOCS / "cloud_load_faking.svg"))


def worlds_figure(sel: pl.DataFrame, neighbours: dict) -> None:
    """Plot the recovery of the clipped worlds and of the displaced-sky worlds against distance."""
    order = list(k.k.SIX)
    clip = sel.filter(pl.col("condition").str.starts_with("physclip")).with_columns(
        world=pl.col("condition")
        .str.extract(r"dc([\d.]+)")
        .map_elements(lambda r: f"DC-to-AC ratio {r}", return_dtype=pl.String)
    )
    rule = alt.Chart().mark_rule(color="grey", strokeDash=[4, 3]).encode(x=alt.datum(1.0))  # ty: ignore[unresolved-attribute]
    base = alt.Chart().encode(
        y=alt.Y("site:N", sort=order, title=None),
        yOffset=alt.YOffset("world:N"),
        color=alt.Color("world:N", legend=alt.Legend(orient="top", title=None)),
    )
    alt.layer(
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q", title="Recovery of the injected 4 MW", scale=alt.Scale(domain=[0.2, 1.3])
            ),
            x2="hi:Q",
        ),
        base.mark_point(filled=True, size=50).encode(x="med:Q"),  # ty: ignore[unresolved-attribute]
        rule,
        data=clip,
    ).properties(
        width=520,
        height=220,
        title=alt.TitleParams(
            [
                "Clipped physics-curve world: recovery by DC-to-AC ratio",
                "Dot: median over 2 cutoffs and 3 seeds. Line: range. Dashed line: exact.",
            ],
            anchor="start",
            fontSize=11,
        ),
    ).save(str(DOCS / "clipped_world.svg"))
    rows = []
    for r in sel.iter_rows(named=True):
        if r["condition"].startswith("physdisp"):
            n = r["condition"].split("_")[1]
            rows.append(
                {
                    "site": r["site"],
                    "n": f"{n[1:]}st nearest Primary's sky"
                    if n == "n1"
                    else f"{n[1:]}nd nearest Primary's sky",
                    "distance_km": neighbours[r["site"]][n]["distance_km"],
                    "recovery": r["med"],
                }
            )
    pl_rows = pl.DataFrame(rows)
    pts = (
        alt.Chart(pl_rows)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "distance_km:Q",
                title="Distance from the Primary to the Primary whose sky drives the injection (km)",
            ),
            y=alt.Y(
                "recovery:Q",
                title="Recovery of the injected 4 MW",
                scale=alt.Scale(domain=[0.2, 1.3]),
            ),
            color=alt.Color("n:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["site"],
        )
    )
    lab = (
        alt.Chart(pl_rows)
        .mark_text(dx=10, fontSize=10)
        .encode(x="distance_km:Q", y="recovery:Q", text="site:N")  # ty: ignore[unresolved-attribute]
    )
    alt.layer(pts, lab).properties(
        width=520,
        height=260,
        title=alt.TitleParams(
            [
                "Displaced-sky world: recovery against the displacement",
                "Dot: median over 2 cutoffs and 3 seeds at one Primary; label names the Primary.",
            ],
            anchor="start",
            fontSize=11,
        ),
    ).save(str(DOCS / "displaced_sky_world.svg"))


def main() -> None:
    """Write the summary, the recovery table, and the figures."""
    df = load()
    rec = r623.recoveries(df)
    rec.write_csv(k.WORK / "recoveries.csv")
    summary = {arm: bars(rec, arm) for arm in ARM_LABEL}
    for arm in ARM_LABEL:
        summary[arm]["calibration"] = calibration(rec, arm)
        summary[arm]["sensitivity_with_cloud_template"] = calibration(rec, arm, model="cloud_tpl")
    ctx = pl.read_csv(k.k.WORK / "recoveries.csv")
    summary["context_623_calibration"] = {
        arm: calibration(ctx, arm) for arm in ("plain", "monotone")
    }
    wr = worlds_summary()
    sel = (
        wr.filter(pl.col("model") == "plain")
        .group_by("condition", "site")
        .agg(
            med=pl.col("recovery").median(),
            lo=pl.col("recovery").min(),
            hi=pl.col("recovery").max(),
        )
    )
    summary["worlds_recovery_median_by_site"] = {
        c: dict(zip(g["site"], g["med"].round(3), strict=True))
        for (c,), g in sel.sort("condition", "site").group_by("condition", maintain_order=True)
    }
    summary["worlds_recovery_median_over_sites"] = {
        c: round(float(g["med"].median()), 3)  # ty: ignore[invalid-argument-type]
        for (c,), g in sel.group_by("condition")  # ty: ignore[invalid-argument-type]
    }
    summary["worlds_neighbour_distance_km"] = json.loads(
        (k.WORK / "worlds_neighbours_2026-06.json").read_text()
    )
    summary["audit"] = {c: json.loads((k.WORK / f"audit_{c}.json").read_text()) for c in k.CUTOFFS}
    worlds_figure(sel, summary["worlds_neighbour_distance_km"])
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
