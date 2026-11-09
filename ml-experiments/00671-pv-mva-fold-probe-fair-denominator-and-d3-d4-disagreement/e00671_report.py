# ruff: noqa: E501
"""Experiment 671: the printed numbers and the figures of the write-up.

Usage: python e00671_report.py   (after e00671_a1.py and e00671_part2.py)
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from e00671_a1 import OUT, SRC
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement"
)
W = 680
SITES = ("D2", "D3", "D4", "D19", "D21", "D5", "D10")
INJ_MW = 2.0
PROBE_COL = {"L0": theme.BRAND_ORANGE, "L1": theme.DATA_GREEN, "FA": theme.DATA_BLUE}


def load_part2() -> dict[str, dict[int, dict]]:
    """Return {condition: {seed: {site: result}}}."""
    out: dict[str, dict[int, dict]] = {}
    for p in sorted((OUT / "part2").glob("*.json")):
        cond, seed = p.stem.rsplit("_s", 1)
        out.setdefault(cond, {})[int(seed)] = json.loads(p.read_text())
    return out


def fig_a1() -> None:
    """A1' by level and probe: two panels, the four Primaries where the floor does not apply and D5 and D10."""
    df = pl.read_csv(OUT / "a1_prime.csv").filter(pl.col("ratio").is_not_null())
    df = df.with_columns(lab=pl.col("site") + " " + pl.col("level").cast(pl.String) + "x")
    panels = []
    for title, sites in (
        ("D9, D11, D14, D16: the floor does not apply", ["D9", "D11", "D14", "D16"]),
        ("D5 and D10: every injected world sits past the fold", ["D5", "D10"]),
    ):
        sub = df.filter(pl.col("site").is_in(sites))
        order = [f"{s} {lv}x" for s in sites for lv in (0.5, 1.0, 2.0)]
        band = (
            alt.Chart(pl.DataFrame({"lo": [0.9], "hi": [1.1]}))
            .mark_rect(opacity=0.15, color=theme.DATA_GREEN)
            .encode(y="lo:Q", y2="hi:Q")  # ty: ignore[unresolved-attribute]
        )
        ref = (
            alt.Chart(pl.DataFrame({"y": [1.0]}))
            .mark_rule(strokeDash=[5, 3], color=theme.BLACK_2)
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        pts = (
            alt.Chart(sub)
            .mark_point(filled=True, size=70, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lab:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y(
                    "ratio:Q",
                    title="PF recovery / signed-world L0 recovery",
                    scale=alt.Scale(domain=[-0.5, 1.4]),
                ),
                color=alt.Color(
                    "method:N",
                    scale=alt.Scale(domain=list(PROBE_COL), range=list(PROBE_COL.values())),
                    title=None,
                    legend=alt.Legend(orient="top") if sites[0] == "D9" else None,
                ),
                xOffset="method:N",
            )
        )
        panels.append(
            (band + ref + pts).properties(
                width=W - 80, height=230, title=alt.TitleParams(title, anchor="start", fontSize=12)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "A1': does converting to apparent power cost recovery, with a fair denominator?",
            subtitle=[
                "2026-06 networks, median over 5 seeds. Green band: the 0.90 to 1.10 bar; dashed line: no loss. Orange: linear probe L0; green: one-sided",
                "linear L1; blue: fold model FA. Level is the injected capacity as a multiple of the midday reading (2.0 MW floor).",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
    )
    chart.save(DOCS / "a1_prime.svg")


def fold_curve(*, c: dict, name: str) -> np.ndarray:
    """Return the fitted fold-model reading at the stored cell's rows."""
    th = np.array(c[f"fold_{name}"]["theta"])
    a = c[f"fold_{name}"]["a"]
    kk = len(th) - 3
    ub, dt = np.array(c["u_basis"]), np.array(c["dt"])
    mm = a - ub @ th[:kk] + th[kk] * dt
    qq = th[kk + 1] + th[kk + 2] * a
    return np.sqrt(mm**2 + qq**2)


def fig_curves(p2: dict) -> None:
    """S(u) of one probe cell per Primary, with the linear and fold fits overlaid."""
    rows = []
    summary = {}
    for site in SITES:
        c = (p2["pf_twin"][0][site])["curve"]
        u, y = np.array(c["u"]), np.array(c["y"])
        for ui, yi in zip(u, y, strict=True):
            rows.append({"site": site, "u": ui, "S": yi, "series": "Probed reading"})
        for ui, yi in zip(u, c["a_lin"] - c["c_lin"] * u, strict=True):
            rows.append({"site": site, "u": ui, "S": yi, "series": "Linear fit"})
        for name, label in (
            ("free", "Fold fit, reactive terms free"),
            ("Z", "Fold fit, reactive terms zero"),
        ):
            for ui, yi in zip(u, fold_curve(c=c, name=name), strict=True):
                rows.append({"site": site, "u": ui, "S": yi, "series": label})
        order = np.argsort(u)
        lo, hi = order[: max(len(u) // 3, 1)], order[-max(len(u) // 3, 1) :]
        s_lo = np.polyfit(u[lo], y[lo], 1)[0] if len(lo) > 2 else float("nan")
        s_hi = np.polyfit(u[hi], y[hi], 1)[0] if len(hi) > 2 else float("nan")
        cls = (
            "falling"
            if s_lo < 0 and s_hi < 0
            else ("V" if s_lo < 0 < s_hi else ("rising" if s_lo > 0 and s_hi > 0 else "mixed"))
        )
        flat = abs(np.polyfit(u, y, 1)[0]) < 0.05 * (y.max() - y.min() + 1e-9) / max(
            np.ptp(u), 1e-9
        )
        summary[site] = {
            "class": "flat" if flat else cls,
            "slope_low_third": round(float(s_lo), 2),
            "slope_high_third": round(float(s_hi), 2),
            "hh": c["cell_hh"],
            "month": c["cell_month"],
            "u_range": [round(float(u.min()), 2), round(float(u.max()), 2)],
        }
    print("curve classes", json.dumps(summary, indent=1))
    df = pl.DataFrame(rows)
    dom = [
        "Probed reading",
        "Linear fit",
        "Fold fit, reactive terms free",
        "Fold fit, reactive terms zero",
    ]
    rng = [theme.DATA_SKY, theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE]
    base = alt.Chart(df).encode(
        x=alt.X("u:Q", title="Solar-output curve u (donor weather)"),
        y=alt.Y("S:Q", title="Probed reading S (MVA)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=dom, range=rng),
            title=None,
            legend=alt.Legend(orient="top", columns=2, labelLimit=320),
        ),
    )
    pts = base.transform_filter(alt.datum.series == "Probed reading").mark_point(
        filled=True, size=14, opacity=0.5, aria=False
    )
    lines = base.transform_filter(alt.datum.series != "Probed reading").mark_line(
        strokeWidth=2, aria=False
    )
    chart = (
        (pts + lines)  # ty: ignore[unsupported-operator]
        .properties(width=190, height=150)
        .facet(facet=alt.Facet("site:N", sort=list(SITES), title=None), columns=3)
        .properties(
            title=alt.TitleParams(
                "S(u) of one probe cell per Primary, with the linear and fold fits",
                subtitle=[
                    "Stored 2026-06 networks, seed 0, cell nearest 12:00 UTC in July. D2 to D21: uninjected networks. D5 and D10: uninjected too",
                    "(the injected truth is read from the 2.0 MW fleet networks in the table). The linear line ignores the temperature term.",
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            )
        )
    )
    chart.save(DOCS / "s_of_u.svg")


def main() -> None:
    """Print every number the write-up quotes and draw the figures."""
    p2 = load_part2()
    seeds = sorted(p2["pf_twin"])
    print("seeds", seeds)
    tw = {s: {sd: p2["pf_twin"][sd][s] for sd in seeds} for s in SITES}
    print(
        "Capacity (MW), median over seeds [min-max]: L0, FA free, FA Z, FA N; extra share from reactive"
    )
    for s in SITES:
        v = {k: np.array([tw[s][sd][k] for sd in seeds]) for k in ("L0", "FA_free", "FA_Z", "FA_N")}
        med = {k: float(np.median(x)) for k, x in v.items()}
        extra = med["FA_free"] - med["L0"]
        share_z = (med["FA_free"] - med["FA_Z"]) / extra if abs(extra) > 0.2 else float("nan")
        share_n = (med["FA_free"] - med["FA_N"]) / extra if abs(extra) > 0.2 else float("nan")
        q = np.median([tw[s][sd]["FA_free_q"] for sd in seeds], axis=0)
        print(
            s,
            {k: f"{med[k]:.2f} [{v[k].min():.2f}-{v[k].max():.2f}]" for k in v},
            f"extra={extra:.2f} share_gone_with_Z={share_z:.2f} with_N={share_n:.2f} free q0,q1={np.round(q, 2)}",
        )
    if "pf_fleet_l05_r0" in p2:
        print("Known answer at D5, D10 (2.0 MW fleet minus twin over 2.0 MW), per probe:")
        for s in ("D5", "D10"):
            fl = p2["pf_fleet_l05_r0"]
            sd_ok = [sd for sd in seeds if sd in fl and s in fl[sd]]
            if not sd_ok:
                continue
            out = {}
            for k in ("L0", "FA_free", "FA_Z", "FA_N"):
                out[k] = round(
                    float(np.median([(fl[sd][s][k] - tw[s][sd][k]) / INJ_MW for sd in sd_ok])), 2
                )
            print(
                s,
                "recovery per 2.0 MW",
                out,
                "(2.0 MW at D5/D10 only; at native Primaries the injected MW differs)",
            )
    rec = pl.read_csv(SRC)
    nat = rec.filter(
        pl.col("site").is_in(["D2", "D3", "D4", "D19", "D21"])
        & pl.col("condition").str.starts_with("pf_fleet")
        & pl.col("method").is_in(["L0", "FA"])
    )
    print("662 A2 rows (native PF worlds), median recovery by site, condition, method")
    print(
        nat.group_by("site", "condition", "method")
        .agg(r=pl.col("rec").median(), inj=pl.col("inj_mw").first())
        .sort("site", "condition", "method")
        .pivot(on="method", index=["site", "condition", "inj"], values="r")
    )
    fig_a1()
    fig_curves(p2)


if __name__ == "__main__":
    main()
