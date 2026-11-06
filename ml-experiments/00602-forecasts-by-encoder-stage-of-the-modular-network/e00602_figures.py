# ruff: noqa: E501, D103, ANN001, ANN201, DTZ001
"""Experiment 602: figures from the staged probe's parquet files (no GPU).

Usage: DATA_PATH_INTERNAL=... PYTHONPATH=(all ml-experiments folders) python e00602_figures.py
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00510_common as c  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00602-forecasts-by-encoder-stage-of-the-modular-network"
PROBE = ML_EXPERIMENTS_DATA_DIR / NAME / "probe"
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
V580 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00580-verified-load-shift-event-set-and-label-audit/work/events_verified.parquet"
)
cp.PANEL_HEIGHT = 320
_x_parts = cp._x_parts
cp._x_parts = lambda **k: _x_parts(**{**k, "labels": True})  # date labels on every panel
NETS = ("P5", "C1", "A1")
IQ = {0.05: 0, 0.5: 1, 0.95: 2}
PRIM = {"D5": 3, "D16": 11}
SWITCH_TIMES = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
CONTRIB = "#306BFF"
DEPTH = "#7A1FA2"
STAGE_TITLE = {
    "s1": "Stage 1: datetime encoder only",
    "s2": "Stage 2: datetime + weather encoders",
    "s3": "Stage 3: datetime + weather + lag encoders (the full network)",
}


def _utc(idx) -> datetime:
    from datetime import UTC

    t0 = datetime.fromtimestamp(c.prep422.GRID_START_US / 1e6, tz=UTC)
    return (t0 + timedelta(microseconds=int(idx) * c.prep422.STEP_US)).replace(tzinfo=None)


def load(*, net: str, case: str, month: str, r: int = -1) -> pl.DataFrame:
    files = sorted((PROBE / net).glob(f"{case}_{month}_r{r:02d}_d*_s0.parquet"))
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("target_idx")
        .unique(subset=["primary", "target_idx"], keep="last", maintain_order=True)
    )


def stage_q(df: pl.DataFrame, stage: str) -> np.ndarray:
    return df.select([f"{stage}_q05", f"{stage}_q50", f"{stage}_q95"]).to_numpy()


def contribution_panel(*, x, contrib, depth, xdom, title, ydom, rules=(), span=None):
    """Line panel: the lag encoder's contribution (stage-3 minus stage-2 median) in points of capacity."""
    df = pl.DataFrame({"x": [str(v) for v in x], "contrib": contrib * 100, "depth": depth * 100})
    sx = alt.Scale(domain=[str(v) for v in xdom])
    ax = alt.Axis(format="%d %b", tickCount=8)
    sy = alt.Scale(domain=list(ydom), nice=False)
    base = alt.Chart(df)
    layers = []
    if span is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"a": [str(span[0])], "b": [str(span[1])]}))
            .mark_rect(color=cp.SHADE_COLOUR, opacity=0.7, clip=True, aria=False)
            .encode(x=alt.X("a:T", scale=sx, axis=ax), x2="b")
        )
    layers += [
        base.mark_line(color=CONTRIB, strokeWidth=1.6, aria=False, clip=True).encode(
            x=alt.X("x:T", scale=sx, axis=ax, title=None),
            y=alt.Y("contrib:Q", scale=sy, title="points of capacity"),
        ),
    ]
    if np.isfinite(depth).any():
        layers.append(
            base.mark_line(
                color=DEPTH, strokeDash=[5, 3], strokeWidth=1.5, aria=False, clip=True
            ).encode(x=alt.X("x:T", scale=sx, axis=ax), y=alt.Y("depth:Q", scale=sy))
        )
    if len(rules):
        layers.append(
            alt.Chart(pl.DataFrame({"x": [str(v) for v in rules]}))
            .mark_rule(color=cp.RULE_COLOUR, strokeDash=[6, 3], strokeWidth=1.3, aria=False)
            .encode(x=alt.X("x:T", scale=sx, axis=ax))
        )
    return alt.layer(*layers).properties(
        width=cp.W,
        height=cp.PANEL_HEIGHT,
        title=alt.TitleParams(title, fontSize=12, anchor="start"),
    )


def forecast_panel(*, df, stage, obs, xdom, ydom, rules=(), span=None, title=None, xtitle=None):
    x = [_utc(i) for i in df["target_idx"]]
    unit = cp.arm_panel(
        x=x,
        q=stage_q(df, stage),
        iq=IQ,
        obs=obs,
        xdom=xdom,
        title=title or STAGE_TITLE[stage],
        ydom=ydom,
        xtype="T",
        xtitle=xtitle,
        span=span,
        rules=rules,
    )
    return unit.vconcat[0]  # drop the thin strip: it only repeated the shade


def ydom_for(dfs, obs):
    lo, hi = np.inf, -np.inf
    for d in dfs:
        for s in ("s1", "s2", "s3"):
            q = stage_q(d, s)
            lo, hi = min(lo, np.nanmin(q[:, 0])), max(hi, np.nanmax(q[:, 2]))
    lo, hi = min(lo, np.nanmin(obs)), max(hi, np.nanmax(obs))
    return lo * 100 - 2, hi * 100 + 2


def key_for(*, span_name=None, rules=False, contrib=False, depth=None):
    key = [*cp.FORECAST_ENTRIES, cp.OBS]
    if span_name:
        key.append(cp.entry(span_name, "band", cp.SHADE_COLOUR, 0.7))
    if rules:
        key.append(cp.entry("Switch times", "rule", cp.RULE_COLOUR))
    if contrib:
        key.append(cp.entry("Lag encoder contribution", "line", CONTRIB))
    if depth:
        key.append(cp.entry(depth, "dashed", DEPTH))
    return key


def two_figures(
    *, tag, df, obs, xdom, ydom, rules, span, span_name, depth=None, depth_name=None, subtitle
):
    """Write the stage 1|2 figure and the stage 3|contribution figure."""
    x = [_utc(i) for i in df["target_idx"]]
    p1 = forecast_panel(df=df, stage="s1", obs=obs, xdom=xdom, ydom=ydom, rules=rules, span=span)
    p2 = forecast_panel(
        df=df,
        stage="s2",
        obs=obs,
        xdom=xdom,
        ydom=ydom,
        rules=rules,
        span=span,
        xtitle="Valid time (UTC)",
    )
    cp.figure(
        units=[p1, p2], title=subtitle, key=key_for(span_name=span_name, rules=bool(len(rules)))
    ).save(str(DOCS / f"{tag}_stages_1_2.svg"))
    contrib = df["s3_q50"].to_numpy() - df["s2_q50"].to_numpy()
    finite = contrib[np.isfinite(contrib)] * 100
    lo_c, hi_c = float(finite.min()), float(finite.max())
    if depth is not None and np.isfinite(depth).any():
        lo_c, hi_c = (
            min(lo_c, float(np.nanmin(depth)) * 100),
            max(hi_c, float(np.nanmax(depth)) * 100),
        )
    cd = (min(lo_c, 0.0) - 2, max(hi_c, 0.0) + 2)
    p3 = forecast_panel(df=df, stage="s3", obs=obs, xdom=xdom, ydom=ydom, rules=rules, span=span)
    p4 = contribution_panel(
        x=x,
        contrib=contrib,
        depth=np.full(len(x), np.nan) if depth is None else depth,
        xdom=xdom,
        title="The lag encoder's contribution",
        ydom=cd,
        rules=rules,
        span=span,
    )
    cp.figure(
        units=[p3, p4],
        title=subtitle,
        key=key_for(span_name=span_name, rules=bool(len(rules)), contrib=True, depth=depth_name),
    ).save(str(DOCS / f"{tag}_stage_3_and_lag_contribution.svg"))


def obs_of(clean, prim, df):
    return clean[prim, df["target_idx"].to_numpy()].astype(float)


def real_figures(clean):
    ev = pl.read_parquet(V580).filter(pl.col("onset_a").str.starts_with("2025-12")).to_dicts()
    assert len(ev) == 1, ev
    ev = ev[0]
    xdom = (datetime(2025, 12, 16), datetime(2026, 1, 28))
    for net in NETS:
        d = (
            pl.concat(
                [
                    load(net=net, case="real", month="2025-12"),
                    load(net=net, case="real", month="2026-01"),
                ]
            )
            .unique(subset=["primary", "target_idx"], keep="last")
            .sort("target_idx")
        )
        for lab, p in PRIM.items():
            df = d.filter(pl.col("primary") == p)
            obs = obs_of(clean, p, df)
            side = "a" if ev["primary_a"] == lab else "b"
            span = (
                SWITCH_TIMES[0],
                SWITCH_TIMES[1],
            )  # the readings step at these times, not at experiment 580's window
            # keep the depth sign: positive means the Primary carries more power after the onset
            sign = np.sign(ev[f"depth_{side}_mw"])
            x = [_utc(i) for i in df["target_idx"]]
            depth_share = np.array(
                [
                    sign * np.nan if not (span[0] <= t < span[1]) else ev[f"depth_{side}_mw"] / 1.0
                    for t in x
                ]
            )
            _ = depth_share  # the real depth is in MW; shown in the text only
            lo, hi = ydom_for([df], obs[np.isfinite(obs)])
            two_figures(
                tag=f"real_{net}_{lab}",
                df=df,
                obs=obs,
                xdom=xdom,
                ydom=(lo, hi),
                rules=SWITCH_TIMES,
                span=span,
                span_name="Load-shift window",
                subtitle=f"{net}, {lab}, December 2025 example event, day-ahead forecasts",
            )


def free_figures(clean):
    for net in NETS:
        d = load(net=net, case="free", month="2025-10")
        for lab, p in {"D5": PRIM["D5"]}.items():
            df = d.filter(pl.col("primary") == p)
            obs = obs_of(clean, p, df)
            xdom = (_utc(df["target_idx"].min()), _utc(df["target_idx"].max()))
            lo, hi = ydom_for([df], obs[np.isfinite(obs)])
            two_figures(
                tag=f"free_{net}_{lab}",
                df=df,
                obs=obs,
                xdom=xdom,
                ydom=(lo, hi),
                rules=(),
                span=None,
                span_name=None,
                subtitle=f"{net}, {lab}, a month with no verified load-shift (2025-10), day-ahead forecasts",
            )


DESIGNS = [(15, 0.30, 1), (15, 0.30, -1), (30, 0.50, 1), (7, 0.15, -1)]


def pass_for(*, prim: int, length: int, depth: float, sign: int) -> int:
    for r in range(bank.N_PASSES):
        d, n, s = bank.event(primary=prim, r=r)
        if n == length and abs(abs(d) - depth) < 1e-9 and s == sign:
            return r
    raise AssertionError


def injected_figures(clean):
    month_start = _utc_idx_of_month("2026-03")
    for net in NETS:
        for lab, p in {"D5": PRIM["D5"]}.items():
            for length, depth, sign in DESIGNS:
                r = pass_for(prim=p, length=length, depth=depth, sign=sign)
                df = load(net=net, case="inject", month="2026-03", r=r).filter(
                    pl.col("primary") == p
                )
                obs = obs_of(clean, p, df)
                end = month_start + bank.END_DAY * bank.PER_DAY
                on = end - length * bank.PER_DAY
                span = (_utc(on), _utc(end))
                xdom = (_utc(int(df["target_idx"].min())), _utc(month_start + 28 * bank.PER_DAY))
                lo, hi = ydom_for([df], obs[np.isfinite(obs)])
                sgn = "up" if sign > 0 else "down"
                two_figures(
                    tag=f"injected_{net}_{lab}_{length}d_{sgn}_{int(depth * 100)}pct",
                    df=df,
                    obs=obs,
                    xdom=xdom,
                    ydom=(lo, hi),
                    rules=(),
                    span=span,
                    span_name="Injected load-shift (lag inputs only)",
                    subtitle=f"{net}, {lab}, injected {length}-day load-shift: {sign * depth * 100:+.0f} points of capacity added to the lag inputs",
                )


def _utc_idx_of_month(month: str) -> int:
    start = c.month_start(month=month)
    return (c.us(t=start) - c.prep422.GRID_START_US) // c.prep422.STEP_US


def response_figure():
    """Per design: the lag encoder's response to an injected load-shift, with the daily-cycle baseline cancelled.

    Pass r (sign +1) and pass r + 12 (sign -1) share the stage 1 and stage 2 forecasts (neither reads the lag inputs),
    so the difference of their stage-3 medians is the response to a shift of twice the depth.
    """
    month_start = _utc_idx_of_month("2026-03")
    end = month_start + bank.END_DAY * bank.PER_DAY
    rows = []
    for net in NETS:
        for r in range(len(bank.COMBOS)):
            up, down = (
                load(net=net, case="inject", month="2026-03", r=r),
                load(net=net, case="inject", month="2026-03", r=r + len(bank.COMBOS)),
            )
            for p in sorted(up["primary"].unique().to_list()):
                d, n, _ = bank.event(primary=p, r=r)
                on = end - n * bank.PER_DAY
                a = up.filter(
                    (pl.col("primary") == p)
                    & (pl.col("target_idx") >= on + 2 * bank.PER_DAY)
                    & (pl.col("target_idx") < end)
                )
                b = down.filter(
                    (pl.col("primary") == p)
                    & (pl.col("target_idx") >= on + 2 * bank.PER_DAY)
                    & (pl.col("target_idx") < end)
                )
                j = a.join(b, on="target_idx", suffix="_dn")
                if j.height < 24:
                    continue
                share = float((j["s3_q50"] - j["s3_q50_dn"]).mean()) / (2 * d)
                rows.append(
                    {"network": net, "primary": p, "depth": d * 100, "length": n, "share": share}
                )
    out = pl.DataFrame(rows)
    out.write_csv(PROBE / "response_share_by_design.csv")
    depth_colours = {15.0: "#10C5F7", 30.0: "#306BFF", 50.0: cp.FORECAST_COLOUR}
    out = out.with_columns((pl.col("depth").round(0)).alias("depth"))
    panels = []
    for net in NETS:
        sub = out.filter(pl.col("network") == net)
        pts = (
            alt.Chart(sub)
            .mark_point(filled=True, size=45, opacity=0.7)
            .encode(
                x=alt.X("length:O", title="Length of the injected load-shift (days)"),
                y=alt.Y(
                    "share:Q",
                    title="Response, share of injected depth",
                    scale=alt.Scale(domain=[-0.05, 0.5]),
                ),
                xOffset="depth:O",
                color=alt.Color(
                    "depth:O",
                    scale=alt.Scale(domain=[15.0, 30.0, 50.0], range=list(depth_colours.values())),
                    legend=None,
                ),
            )
        )
        panels.append(pts.properties(width=300, height=320, title=net))
    key = [
        cp.entry(f"Depth {int(d)} points of capacity", "band", c_, 1.0)
        for d, c_ in depth_colours.items()
    ]
    cp_key = cp.legend(key)
    (
        alt.vconcat(alt.hconcat(*panels), cp_key.properties(width=900)).configure_view(stroke=None)
    ).save(str(DOCS / "summary_response_share_by_design.svg"))
    return out


def step_table(clean) -> pl.DataFrame:
    """Return the 24-hour mean power (MW) either side of each switch time at D5 and D16 (capacity from experiment 521's site map)."""
    cap = {"D5": 5.723, "D16": 9.115}
    rows = []
    for lab, p in PRIM.items():
        for name, dt in zip(("start", "end"), SWITCH_TIMES, strict=True):
            i = (
                int(dt.replace(tzinfo=UTC).timestamp() * 1e6) - c.prep422.GRID_START_US
            ) // c.prep422.STEP_US
            before, after = (
                np.nanmean(clean[p, i - 48 : i]) * cap[lab],
                np.nanmean(clean[p, i : i + 48]) * cap[lab],
            )
            rows.append(
                {
                    "primary": lab,
                    "switch": name,
                    "mean_before_mw": before,
                    "mean_after_mw": after,
                    "step_mw": after - before,
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(PROBE / "switch_time_steps.csv")
    return out


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    clean = c.clean_stack()
    real_figures(clean)
    free_figures(clean)
    injected_figures(clean)
    print(step_table(clean))
    resp = response_figure()
    print(
        resp.group_by("network").agg(
            pl.col("share").mean().alias("mean"),
            pl.col("share").median().alias("median"),
            pl.col("share").std().alias("sd"),
            pl.len(),
        )
    )
    print(resp.group_by("network", "length").agg(pl.col("share").mean()).sort("network", "length"))
