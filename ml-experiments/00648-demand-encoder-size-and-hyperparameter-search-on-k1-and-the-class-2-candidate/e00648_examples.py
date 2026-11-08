# ruff: noqa: D103, E501
"""Experiment 648: example forecasts of the unchanged candidate and the arm with the largest gain (CPU).

Draws a day-ahead forecast (one Primary, three days of January 2026) and the full 10-day horizon of one
forecast (one Primary, one origin in June 2026). Each figure has two panels: the unchanged class 2
candidate and the candidate with lag-encoder width 128. Primaries only (demand), so no anonymisation.
The forecasts and the observed power come from experiment 613's scoring tables (`k597.Inputs`).

Usage: python e00648_examples.py
"""

import sys
from pathlib import Path

import altair as alt
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME648 = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
NAME613 = "00613-nra-roster-contrasts-with-intervals-five-seeds"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME648
ROOT613 = ML_EXPERIMENTS_DATA_DIR / NAME613
DOCS = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / NAME648
IQ = c.IQ
MONTHS = ("2025-10", "2026-01", "2026-06")

c.MONTHS = MONTHS
a.SEEDS = (0,)
work = ROOT / "arms" / "k1_ref" / "work"
k584.ARMS["k1_ref"] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
twin_root = ROOT613 / "arms" / "k3_d5" / "work"
_orig = g.a._load
g.a._load = lambda *, root, tag, months: _orig(
    root=twin_root if tag == "D5" else root, tag=tag, months=months
)
k597.WORK = ROOT / "score" / "figures_tmp"
t = k597.Inputs(arm="k1_ref").t
g.a._load = _orig


def aligned(*, arm: str) -> np.ndarray:
    e = g.a._load(root=ROOT / "arms" / arm / "work", tag="P5", months=MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US)
            // c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = g.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    pos = np.clip(np.searchsorted(kk[order], g.s.key(t.prim, t.origin_idx, t.idx)), 0, len(kk) - 1)
    ok = kk[order][pos] == g.s.key(t.prim, t.origin_idx, t.idx)
    q = np.sort(e.select(list(a.QCOLS)).to_numpy().astype(np.float32)[order[pos]], axis=1)
    q[~ok] = np.nan
    return q


ARMS = {
    "c2_ref": "Class 2 candidate, unchanged",
    "c2_lag128": "Class 2 candidate, lag-encoder width 128",
}
Q = {k: aligned(arm=k) for k in ARMS}


def panel(
    *,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: list[str],
    xtitle: str,
    xdom: list[float],
    height: int = 200,
) -> alt.Chart:
    """Draw one panel per arm; each series is (x, quantiles in fractions, observed in fractions)."""
    parts = []
    for label, (x, q, obs) in series.items():
        q100 = q * 100
        parts.append(
            pl.DataFrame(
                {
                    "x": x,
                    "arm": label,
                    "lo90": q100[:, IQ[0.05]],
                    "hi90": q100[:, IQ[0.95]],
                    "lo50": q100[:, IQ[0.25]],
                    "hi50": q100[:, IQ[0.75]],
                    "med": q100[:, IQ[0.5]],
                    "obs": obs * 100,
                }
            )
        )
    d = pl.concat(parts)
    base = alt.Chart(d).encode(
        x=alt.X(
            "x:Q",
            title=xtitle,
            scale=alt.Scale(domain=xdom, nice=False),
            axis=alt.Axis(values=list(range(int(xdom[0]), int(xdom[1]) + 1))),
        )
    )
    y = alt.Y("lo90:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False))
    b90 = base.mark_area(opacity=0.25, color=theme.DATA_SKY).encode(y=y, y2="hi90:Q")  # ty: ignore[unresolved-attribute]
    b50 = base.mark_area(opacity=0.45, color=theme.DATA_BLUE).encode(y=alt.Y("lo50:Q"), y2="hi50:Q")  # ty: ignore[unresolved-attribute]
    med = base.mark_line(color=theme.DATA_BLUE, strokeWidth=1.5).encode(y="med:Q")  # ty: ignore[unresolved-attribute]
    obs = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=1.5).encode(y="obs:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(b90, b50, med, obs)  # ty: ignore[invalid-return-type]
        .properties(width=480, height=height)
        .facet(
            row=alt.Row(
                "arm:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=11)
            )
        )
        .properties(title=title)
    )


def day_ahead(*, p: int, month: str, first_day: int, days: int) -> dict:
    lo = first_day * 48
    sel = (
        (t.prim == p)
        & (t.month == month)
        & (t.idx >= lo)
        & (t.idx < lo + days * 48)
        & (t.idx // 48 == t.origin_idx // 48 + 1)
    )
    rows = np.flatnonzero(sel)
    rows = rows[np.argsort(t.idx[rows])]
    x = (t.idx[rows] - lo) / 48.0 + 1
    return {label: (x, Q[arm][rows], t.obs[rows]) for arm, label in ARMS.items()}


def horizon(*, p: int, month: str) -> dict:
    """One forecast's full 10 days, read from the stored fits; observed power from the scoring table."""
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    obs = {}
    for i in np.flatnonzero((t.prim == p) & ~np.isnan(t.obs)):
        obs[int(t.idx[i])] = float(t.obs[i])
    out, origin = {}, None
    for arm, label in ARMS.items():
        f = pl.read_parquet(ROOT / "arms" / arm / "work" / "fits" / "P5" / f"{month}_d0_s0.parquet")
        f = f.filter(pl.col("time_series_id").replace_strict(prim_of, default=-1) == p)
        if origin is None:
            origins = sorted(f["power_fcst_init_time"].unique().to_list())
            origin = origins[len(origins) // 3]
        f = f.filter(pl.col("power_fcst_init_time") == origin).sort("valid_time")
        idx = (
            (f["valid_time"].dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).to_numpy()
        o0 = (int(origin.timestamp() * 1e6) - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        q = np.sort(f.select(list(a.QCOLS)).to_numpy().astype(np.float32), axis=1)
        out[label] = ((idx - o0) / 48.0, q, np.array([obs.get(int(i), np.nan) for i in idx]))
    return out


eight = [p for p in range(16) if t.group[p] == "MVA-only eight"]
six = [p for p in range(16) if t.group[p] == "signed-MW six"]
free_days = np.unique(t.idx[(t.prim == eight[0]) & (t.month == "2026-01") & (t.band == 1)] // 48)
panel(
    series=day_ahead(
        p=eight[0], month="2026-01", first_day=int(free_days[len(free_days) // 2]), days=3
    ),
    title=[
        "Day-ahead forecasts for three days of January 2026, one Primary (MVA-only group).",
        "Line: median; shades: 50% and 90% bands; orange: observed.",
    ],
    xtitle="Day of the window",
    xdom=[1, 4],
).save(str(DOCS / "figure_3_day_ahead.svg"))
panel(
    series=horizon(p=six[0], month="2026-06"),
    title=[
        "One forecast in June 2026, one Primary (signed-MW group), from its origin to 10 days ahead.",
        "Line: median; shades: 50% and 90% bands; orange: observed.",
    ],
    xtitle="Days after the forecast origin",
    xdom=[0, 10],
).save(str(DOCS / "figure_4_horizon.svg"))
sys.exit(0)
