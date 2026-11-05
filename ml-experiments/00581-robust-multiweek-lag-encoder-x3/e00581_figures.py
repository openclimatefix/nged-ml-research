# ruff: noqa: E501, D103, PERF401
"""Experiment 581 figures: day-ahead forecasts with their q05-q95 bands around injected and verified events.

Each figure stacks one panel per forecast (v3+D5, X3-med, and the lags-masked twin of v3+D5) for one
Primary. The injected events add the shift to the lag inputs only, so the dark line (the power with
the event removed) is the normal-running-arrangement (NRA) reference. For a verified event the dark
line is the observed power minus the fitted depth. Day-ahead means the forecast issued 24 to 48 h
before the target half-hour. Time is in days since the event's onset.

Usage: python e00581_figures.py [arm ...]   (extra arms, e.g. x3syn, add panels)
"""

import subprocess
import sys
from pathlib import Path

import altair as alt
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank
import e00581_tails as tl
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
SITE_MAP = (
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
COLOURS = {
    "v3+D5": theme.DATA_BLUE, "X3-med": theme.BRAND_ORANGE, "Lags masked (v3+D5)": theme.DATA_GREEN,
    "X3-syn": theme.DATA_PURPLE, "v3+D5-syn": theme.DATA_SKY,
}  # fmt: skip
LABEL = {
    "v3d5": "v3+D5",
    "x3med": "X3-med",
    "x3syn": "X3-syn",
    "v3syn": "v3+D5-syn",
    "placebo": "Placebo",
    "x335": "X3-3.5h",
}
COLOURS |= {"Placebo": "#777777", "X3-3.5h": theme.DATA_GREEN}
KEYS = tl.KEYS


def day_ahead(*, paths: list[Path], prim_ts: int, lo: int, hi: int) -> pl.DataFrame:
    """Return the seed-mean quantiles of the 24-48 h forecasts for one series between grid indices `lo` and `hi`."""
    df = tl.seed_mean(paths=paths).filter(pl.col("time_series_id") == prim_ts)
    df = df.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        idx=((pl.col("valid_time").dt.epoch("us") - k578.prep.GRID_START_US) // k578.prep.STEP_US),
    )
    return df.filter(
        (pl.col("lead") >= 24)
        & (pl.col("lead") < 48)
        & (pl.col("idx") >= lo)
        & (pl.col("idx") < hi)
    )


def panel(
    *,
    frames: dict[str, pl.DataFrame],
    truth: pl.DataFrame,
    onset: int,
    end: int,
    title: str,
    show_x: bool,
) -> alt.Chart:
    rows = []
    for name, d in frames.items():
        for r in d.iter_rows(named=True):
            rows.append(
                {
                    "series": name,
                    "day": (r["idx"] - onset) / 48.0,
                    "q05": 100 * r["q05"],
                    "q50": 100 * r["q50"],
                    "q95": 100 * r["q95"],
                }
            )
    src = pl.DataFrame(rows)
    names = list(frames)
    yaxis = alt.Y("q05:Q", title="Power (% of capacity)", scale=alt.Scale(domain=[30, 135]))
    base = alt.Chart(src).encode(
        x=alt.X(
            "day:Q",
            title="Days since the event's onset" if show_x else None,
            axis=alt.Axis(labels=show_x),
        )
    )  # ty: ignore[unresolved-attribute]
    band = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        y=yaxis, y2="q95:Q", color=alt.value(COLOURS[names[0]])
    )
    med = base.mark_line(strokeWidth=1.3).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q50:Q", scale=alt.Scale(domain=[30, 135])), color=alt.value(COLOURS[names[0]])
    )
    t = pl.DataFrame(
        {"day": (truth["idx"].to_numpy() - onset) / 48.0, "y": 100 * truth["y"].to_numpy()}
    )
    tr = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#222222", strokeWidth=1.0)
        .encode(x="day:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[30, 135])))
    )  # ty: ignore[unresolved-attribute]
    ev = (
        alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - onset) / 48.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color="#999999", opacity=0.15)
        .encode(x="a:Q", x2="b:Q")
    )  # ty: ignore[unresolved-attribute]
    return alt.layer(ev, band, med, tr).properties(width=620, height=120, title=title)  # ty: ignore[invalid-return-type]


def figure(
    *,
    name: str,
    sub: list[str],
    panels: list[tuple[str, dict[str, pl.DataFrame], pl.DataFrame, int, int]],
) -> None:
    charts = [
        panel(frames=f, truth=t, onset=o, end=e, title=ti, show_x=(i == len(panels) - 1))
        for i, (ti, f, t, o, e) in enumerate(panels)
    ]
    chart = (
        alt.vconcat(*charts, spacing=8)
        .properties(
            title=alt.TitleParams(
                text=[name], subtitle=sub, anchor="start", fontSize=13, subtitleFontSize=10
            )
        )
        .resolve_scale(color="shared")
    )  # ty: ignore[unresolved-attribute]
    DOCS.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower().replace("%", "pct"))
    out = DOCS / f"{slug}.svg"
    chart.save(out)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(out)],
        check=False,
        capture_output=True,
    )
    print("wrote", out)


def main() -> None:
    arms = ["v3d5", "x3med", *sys.argv[1:]]
    sm = pl.read_csv(SITE_MAP)
    ts_of = dict(zip(sm["label"].to_list(), sm["time_series_id"].to_list(), strict=True))
    p_of = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
    clean = k578.clean()
    # injected examples: (Primary label, month, pass)
    ex = [("D5", "2025-12", 3), ("D16", "2025-12", 22), ("D3", "2025-12", 2), ("D8", "2026-01", 13)]
    for lab, month, r in ex:
        p = p_of[lab]
        depth, length, sign = bank.event(primary=p, r=r)
        start = (
            k578.c.us(t=k578.c.month_start(month=month)) - k578.prep.GRID_START_US
        ) // k578.prep.STEP_US
        end = start + bank.END_DAY * bank.PER_DAY
        onset = end - length * bank.PER_DAY
        lo, hi = max(onset - 3 * 48, start), end + 8 * 48
        panels = []
        for a in arms:
            paths = [
                tl.inject_dir(a) / f"{month}_r{r:02d}_{d}_s{s}.parquet"
                for d in ("d0", "d14")
                for s in tl.SEEDS
            ]
            panels.append(
                (LABEL[a], {LABEL[a]: day_ahead(paths=paths, prim_ts=ts_of[lab], lo=lo, hi=hi)})
            )
        masked = day_ahead(
            paths=[
                tl.W572 / "fits" / "V3D5-removed" / f"{month}_{d}_s{s}.parquet"
                for d in ("d0", "d14")
                for s in tl.SEEDS
            ],
            prim_ts=ts_of[lab],
            lo=lo,
            hi=hi,
        )
        truth = pl.DataFrame({"idx": np.arange(lo, hi), "y": clean[p, lo:hi]}).filter(
            pl.col("y").is_finite()
        )
        figs = [(t, f, truth, onset, end) for t, f in panels] + [
            ("Lags masked (v3+D5)", {"Lags masked (v3+D5)": masked}, truth, onset, end)
        ]
        figure(
            name=f"Injected event {lab} {int(abs(depth) * 100)}% for {length} days",
            sub=[
                f"A shift of {'+' if sign > 0 else '-'}{int(abs(depth) * 100)}% of capacity is added to the lag inputs for {length} days (grey span); the targets are untouched.",
                "Shaded: q05-q95 of the day-ahead forecast. Line: median. Dark line: the unshifted power. Substation "
                + lab
                + ".",
            ],
            panels=figs,
        )
    # verified event 23 (D5 and D16): observed lags, NRA = observed minus the fitted depth
    sm_ev = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
    )
    events = bank.verified_events(
        capacity_mw=dict(zip(sm["label"].to_list(), sm["capacity_mw"].to_list(), strict=True)),
        label_to_primary=p_of, table=sm_ev,
    )  # fmt: skip
    fixed = bank.corrected_clean(clean=clean, events=events)
    for q in events:
        if q["transfer_id"] != 23:
            continue
        lab = q["label"]
        lo, hi = q["onset_idx"] - 3 * 48, q["end_idx"] + 8 * 48
        panels = []
        for a in arms:
            paths = [
                tl.arm_root(a) / "fits" / "V3D5" / f"{m}_{d}_s{s}.parquet"
                for m in ("2025-12", "2026-01")
                for d in ("d0", "d14")
                for s in tl.SEEDS
            ]
            panels.append(
                (LABEL[a], {LABEL[a]: day_ahead(paths=paths, prim_ts=ts_of[lab], lo=lo, hi=hi)})
            )
        mpaths = [
            tl.W572 / "fits" / "V3D5-removed" / f"{m}_{d}_s{s}.parquet"
            for m in ("2025-12", "2026-01")
            for d in ("d0", "d14")
            for s in tl.SEEDS
        ]
        masked = day_ahead(paths=mpaths, prim_ts=ts_of[lab], lo=lo, hi=hi)
        truth = pl.DataFrame({"idx": np.arange(lo, hi), "y": fixed[q["primary"], lo:hi]}).filter(
            pl.col("y").is_finite()
        )
        figs = [(t, f, truth, q["onset_idx"], q["end_idx"]) for t, f in panels] + [
            (
                "Lags masked (v3+D5)",
                {"Lags masked (v3+D5)": masked},
                truth,
                q["onset_idx"],
                q["end_idx"],
            )
        ]
        figure(
            name=f"Verified event 23 at {lab}",
            sub=[
                f"A load transfer found in the telemetry (about {abs(q['depth']) * 100:.0f}% of capacity, {q['length_days']:.0f} days, grey span; December 2025 to January 2026).",
                "Shaded: q05-q95 of the day-ahead forecast made with the observed lags. Line: median. Dark line: observed power minus the fitted transfer (the NRA). Substation "
                + lab
                + ".",
            ],
            panels=figs,
        )


def summary() -> None:
    """Draw follow ratio and NRA-side q95 coverage by phase for every arm (injected bank, 24-120 h)."""
    import e00581_summary as sm_

    t = sm_.injected(arms=list(LABEL)).with_columns(
        arm_name=pl.col("arm").replace_strict(LABEL),
        phase_n=pl.col("phase").replace_strict(sm_.PHASE_ORDER),
    )
    names = [LABEL[a] for a in LABEL]
    colour = alt.Color(
        "arm_name:N",
        scale=alt.Scale(domain=names, range=[COLOURS[n] for n in names]),
        legend=alt.Legend(orient="top", title=None, columns=3),
    )
    phases = list(sm_.PHASE_ORDER)
    rows = []
    for grp in ("MVA-only eight", "signed-MW six"):
        sub = t.filter(pl.col("group") == grp)
        row = []
        for value, title, rule in (
            ("follow_q50", "Median follow ratio", None),
            ("nra_side95", "NRA-side coverage at q95/q05", 0.95),
        ):
            base = alt.Chart(sub).encode(
                x=alt.X("phase:N", sort=phases, title=None),
                y=alt.Y(f"{value}:Q", title=None),
                color=colour,
            )  # ty: ignore[unresolved-attribute]
            layers = [base.mark_line(point=True)]
            if rule is not None:
                layers.append(
                    alt.Chart(pl.DataFrame({"y": [rule]}))  # ty: ignore[unresolved-attribute]
                    .mark_rule(color="#999999", strokeDash=[4, 3])
                    .encode(y="y:Q")
                )  # ty: ignore[unresolved-attribute]
            row.append(
                alt.layer(*layers).properties(
                    width=290,
                    height=150,
                    title=f"{title}: {'eight MVA Primaries' if grp.startswith('MVA') else 'six MW Primaries'}",
                )
            )
        rows.append(alt.hconcat(*row))
    chart = alt.vconcat(*rows).properties(
        title=alt.TitleParams(
            text=["Lag-only injected events: following and tail coverage by phase"],
            subtitle=[
                "Events of 3, 7, and 15 days, depths 15%, 30%, and 50% of capacity, both signs; forecasts 24-120 h ahead. Dashed: nominal 0.95."
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=10,
        )
    )  # ty: ignore[unresolved-attribute]
    out = DOCS / "tails_by_phase.svg"
    chart.save(out)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(out)],
        check=False,
        capture_output=True,
    )
    print("wrote", out)


if __name__ == "__main__":
    summary()
    main()
