# ruff: noqa: E501
"""Experiment 696: draw the write-up's three figures from the audit's per-row losses.

    uv run --no-sync python ml-experiments/00696-*/e00696_figures.py OUTPUT_DIR

Reads `audit/losses.parquet` and the truth through `e00696_audit_common` (made by
`e00696_audit_load.py`). Nothing is refitted. No figure carries a date or a series name: the host
appears only as "the host Primary", in aggregate scores, and "Gas A" is the engine's anonymised
label. Each interval is recomputed here with the audit's method (t over 8 monthly differences of seed
means, a month bootstrap of seed-averaged losses, and a t over per-seed means), so every plotted
number has a script as its source.
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from e00696_audit_common import HOST_CAPACITY_MVA, OUT, SH
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, GREY_2
from scipy import stats

WIDTH = 680
SEEDS = 10
THRESHOLD = 0.16


def _rows(*, arm: str, seeds: range, mask: pl.Expr | None = None) -> pl.DataFrame:
    """Return one arm's loss rows, wide by seed, sorted by origin and valid time."""
    d = pl.read_parquet(OUT / "losses.parquet").filter(
        pl.col("arm") == arm, pl.col("seed").is_in(list(seeds))
    )
    if mask is not None:
        d = d.filter(mask)
    return d.pivot(on="seed", index=["origin", "valid_time", "month"], values="crps").sort(
        "origin", "valid_time"
    )


def _t(*, v: np.ndarray, level: float) -> tuple[float, float]:
    """Return the paired t interval of the mean of `v`."""
    half = stats.t.ppf(0.5 + level / 200, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - half), float(v.mean() + half)


def contrast(
    *, test: str, ref: str, level: float, seeds: range = range(SEEDS), mask: pl.Expr | None = None
) -> dict[str, float]:
    """Return the estimate and the month-bootstrap and per-seed t intervals of test minus ref."""
    a, b = _rows(arm=test, seeds=seeds, mask=mask), _rows(arm=ref, seeds=seeds, mask=mask)
    j = a.join(b, on=["origin", "valid_time", "month"], suffix="_r")
    assert len(j) == len(a) == len(b)
    diff = np.stack([j[str(s)].to_numpy() - j[f"{s}_r"].to_numpy() for s in seeds])
    months = j["month"].to_numpy()
    seed_mean = diff.mean(0)
    names = sorted(set(months))
    sums = np.array([seed_mean[months == m].sum() for m in names])
    n = np.array([(months == m).sum() for m in names], float)
    draws = np.random.default_rng(7).multinomial(
        len(names), np.ones(len(names)) / len(names), size=40000
    )
    tail = (100 - level) / 2
    lo, hi = np.percentile((draws @ sums) / (draws @ n), [tail, 100 - tail])
    s_lo, s_hi = _t(v=diff.mean(1), level=level)
    return {
        "est": float(diff.mean()),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "seed_lo": s_lo,
        "seed_hi": s_hi,
    }


def host_running() -> pl.DataFrame:
    """Return the host's valid times with a flag for half-hours on which the engine runs."""
    return (
        pl.read_parquet(SH / "ground_truth_primaries.parquet")
        .filter(pl.col("time_series_id").cast(pl.Int64) == 11)
        .select("valid_time", real=pl.col("power") * 100 / HOST_CAPACITY_MVA)
    )


def _dot_chart(*, frame: pl.DataFrame, title: str, subtitle: list[str], height: int) -> alt.Chart:
    """Return a dot-and-interval chart of contrasts, one row each, with a zero rule."""
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["D8", "Host Primary"], range=[DATA_BLUE, BRAND_ORANGE]),
            title=None,
            legend=alt.Legend(orient="top"),
        ),
    )
    wide = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="CRPS change (points of capacity; negative is better)"), x2="hi:Q"
    )
    dot = base.mark_circle(size=90).encode(x="est:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=GREY_2)
        .encode(x="x:Q")
    )
    return (wide + dot + zero).properties(
        width=WIDTH - 330, height=height, title=alt.TitleParams(title, subtitle=subtitle)
    )


def figure_1(*, out: Path) -> None:
    """The four planned contrasts with the month-bootstrap interval at 98.75%."""
    spec = [
        ("C1  D8: S1 minus A0", "S1", "A0", "D8"),
        ("C2  D8: S1 minus SP (placebo)", "S1", "SP", "D8"),
        ("C3  Host: WST minus W0 (true state)", "WST", "W0", "Host Primary"),
        ("C4  Host: WS1 minus W0 (tracked state)", "WS1", "W0", "Host Primary"),
    ]
    rows = []
    for label, t, r, kind in spec:
        c = contrast(test=t, ref=r, level=98.75)
        rows.append(
            {"label": label, "kind": kind, "est": c["est"], "lo": c["boot_lo"], "hi": c["boot_hi"]}
        )
    chart = _dot_chart(
        frame=pl.DataFrame(rows),
        title="No contrast reaches the 0.16-point gain that would justify porting",
        subtitle=[
            "Change in CRPS from giving the pre-training mixer the engine state, mean of 10 seeds,",
            "8 refit months. Dot: estimate. Line: 98.75% interval from resampling whole months.",
            "The gain needed at D8, -0.16, lies below the lower end of every interval.",
        ],
        height=150,
    )
    chart.save(out / "figure_1_contrasts.svg")


def figure_2(*, out: Path) -> None:
    """The host's change split by whether the engine runs, and D8's split by season."""
    run = host_running()
    host = pl.read_parquet(OUT / "losses.parquet").filter(pl.col("arm").str.starts_with("W"))
    flag = host.join(run, on="valid_time", how="left").with_columns(
        running=(pl.col("real") - pl.col("act")) > 0.5
    )
    losses = pl.read_parquet(OUT / "losses.parquet")

    def seed_t(*, df: pl.DataFrame, test: str, ref: str) -> tuple[float, float, float]:
        g = df.group_by("arm", "seed").agg(pl.col("crps").mean())
        a = g.filter(pl.col("arm") == test).sort("seed")["crps"].to_numpy()
        b = g.filter(pl.col("arm") == ref).sort("seed")["crps"].to_numpy()
        lo, hi = _t(v=a - b, level=95)
        return float((a - b).mean()), lo, hi

    rows = []
    for label, test, mask in (
        ("Host, engine off: WST minus W0", "WST", ~pl.col("running")),
        ("Host, engine running: WST minus W0", "WST", pl.col("running")),
        ("Host, engine off: WS1 minus W0", "WS1", ~pl.col("running")),
        ("Host, engine running: WS1 minus W0", "WS1", pl.col("running")),
    ):
        est, lo, hi = seed_t(df=flag.filter(mask), test=test, ref="W0")
        rows.append({"label": label, "kind": "Host Primary", "est": est, "lo": lo, "hi": hi})
    months = pl.col("valid_time").dt.month()
    for label, mask in (
        ("D8, first 4 refit months: S1 minus A0", months.is_in([11, 12, 1, 2])),
        ("D8, last 4 refit months: S1 minus A0", ~months.is_in([11, 12, 1, 2])),
    ):
        est, lo, hi = seed_t(
            df=losses.filter(pl.col("arm").is_in(["S1", "A0"]), mask), test="S1", ref="A0"
        )
        rows.append({"label": label, "kind": "D8", "est": est, "lo": lo, "hi": hi})
    chart = _dot_chart(
        frame=pl.DataFrame(rows),
        title="At the host, the true state helps when the engine is off and hurts when it runs",
        subtitle=[
            "Post hoc split, not a planned contrast: 95% intervals over 10 seeds, not Bonferroni-adjusted.",
            "Dot: estimate. Line: 95% interval. Host rows are split by whether the injected engine runs",
            "at the valid time. D8 rows are split by the first and last 4 of the 8 refit months.",
        ],
        height=200,
    )
    chart.save(out / "figure_2_engine_off_and_running.svg")


def figure_3(*, out: Path) -> None:
    """CRPS at D8 of each seed, by arm, with the seed groups of the control marked."""
    d = (
        pl.read_parquet(OUT / "losses.parquet")
        .filter(pl.col("arm").is_in(["A0", "S1", "SP"]))
        .group_by("arm", "seed")
        .agg(pl.col("crps").mean())
        .with_columns(
            group=pl.when(pl.col("seed") < 3)
            .then(pl.lit("Seeds 0 to 2 (A0 fitted in experiment 695)"))
            .otherwise(pl.lit("Seeds 3 to 9 (all arms fitted here)"))
        )
    )
    order = ["A0", "S1", "SP"]
    dots = (
        alt.Chart(d)
        .mark_circle(size=70, opacity=0.9)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X(
                "crps:Q",
                title="CRPS at D8 (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[7.1, 7.5]),
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=[
                        "Seeds 0 to 2 (A0 fitted in experiment 695)",
                        "Seeds 3 to 9 (all arms fitted here)",
                    ],
                    range=[BRAND_ORANGE, DATA_BLUE],
                ),
                title=None,
                legend=alt.Legend(orient="top", columns=1, labelLimit=400),
            ),
        )
        .properties(
            width=WIDTH - 120,
            height=130,
            title=alt.TitleParams(
                "The control's first three seeds scored better than its other seven",
                subtitle=[
                    "Each dot is one seed's mean CRPS over all 8 refit months. The three seeds that",
                    "experiment 695 fitted score about 0.08 points better than the seven fitted here,",
                    "for A0 alone; S1 and SP score the same in both groups.",
                ],
            ),
        )
    )
    dots.save(out / "figure_3_seed_groups.svg")


def main() -> None:
    """Draw the three figures into the folder named on the command line."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    figure_1(out=out)
    figure_2(out=out)
    figure_3(out=out)


if __name__ == "__main__":
    main()
