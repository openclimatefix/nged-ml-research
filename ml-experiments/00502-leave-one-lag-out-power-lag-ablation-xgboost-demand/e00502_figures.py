r"""Draw experiment 502's figures: the sweep share per lag subset and the CRPS contrasts.

Reads `work/sweep/sweep.parquet` (arms `nowk`, `noday`), experiment 465's sweep table (arms `a` and
`b`, the same two Primaries and refit dates), and `work/scores/rows.parquet`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00502-leave-one-lag-out-power-lag-ablation-xgboost-demand \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00502-*/e00502_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00502_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00502-leave-one-lag-out-power-lag-ablation-xgboost-demand"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
LABELS: Final[dict[str, str]] = {
    "a": "all three lags",
    "nowk": "no week lag",
    "noday": "no day lag",
    "b": "no lags",
    "noorigin": "no origin lag",
    "f": "50/50 mixture of all three lags and no lags",
}
COLOURS: Final[dict[str, str]] = {
    "all three lags": theme.DATA_BLUE,
    "no week lag": theme.BRAND_ORANGE,
    "no day lag": theme.DATA_PURPLE,
    "no lags": theme.BLACK_1,
}
HOURS: Final[dict[str, int]] = {"6h": 6, "24h": 24, "48h": 48, "3d": 72, "5d": 120, "7d": 168}
BAND: Final[str] = "24-72 h"
LEVEL_PERCENT: Final[float] = 100.0 * (1.0 - 0.05 / 12)


def sweep_table() -> pl.DataFrame:
    """Return the sweep sums of the four arms on the two Primaries and two refits swept."""
    new = pl.read_parquet(fit.WORK_DIR / "sweep" / "sweep.parquet")
    old = pl.read_parquet(
        next(ML_EXPERIMENTS_DATA_DIR.glob("00465-*")) / "work" / "sweep" / "sweep.parquet"
    ).filter(
        pl.col("site").is_in(new["site"].unique()),
        pl.col("refit").is_in(new["refit"].unique()),
        pl.col("arm").is_in(["a", "b"]),
    )
    return pl.concat([new, old], how="diagonal_relaxed")


def sweep_figure() -> None:
    """Draw the median shift as a share of a 5% step, by step length and arm."""
    data = (
        sweep_table()
        .filter(
            pl.col("kind") == "step",
            pl.col("widening") == 1.0,
            pl.col("share") == 0.05,
            pl.col("band") == BAND,
        )
        .group_by("arm", "variant", "sign")
        .agg(pl.col("shift_sum").sum(), pl.col("n").sum())
        .with_columns(share=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * 0.05))
        .group_by("arm", "variant")
        .agg(pl.col("share").mean())
        .with_columns(
            hours=pl.col("variant").replace_strict(HOURS, return_dtype=pl.Int64),
            arm_label=pl.col("arm").replace(LABELS),
        )
    )
    chart = (
        alt.Chart(data.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hours:Q",
                title="Length of the injected step, ending at the forecast origin (hours)",
                axis=alt.Axis(values=list(HOURS.values())),
            ),
            y=alt.Y("share:Q", title="Median shift / step", scale=alt.Scale(domain=[0, 0.4])),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=220,
            title=alt.TitleParams(
                [
                    "A step of 5% of capacity in the last readings: how far the median forecast",
                    "moves, as a share of the step, at 24 to 72 hours lead (smaller is robust).",
                ],
                anchor="start",
                fontSize=11,
            ),
        )
    )
    chart.save(FIGURE_DIR / "figure_1_sweep_share.svg")


def contrast_figure() -> None:
    """Draw each arm's CRPS minus all three lags, on rows with no labelled event and after one."""
    rows = pl.read_parquet(fit.WORK_DIR / "scores" / "rows.parquet").filter(
        pl.col("widening") == 1.0
    )
    reference = rows.filter(pl.col("arm") == "a")
    sets = {
        "No labelled event": pl.col("in_event").not_() & pl.col("post48").not_(),
        "Within 7 days after an event": pl.col("post7d"),
    }
    out = []
    for arm in ("nowk", "noday", "noorigin", "f", "b"):
        for name, expr in sets.items():
            interval = metrics.month_bootstrap_contrast(
                treatment=rows.filter(pl.col("arm") == arm),
                reference=reference,
                metric="crps",
                where=expr & (pl.col("lead_hours") <= 240.0),
                level_percent=LEVEL_PERCENT,
                n_resamples=20_000,
            )
            out.append(
                {
                    "arm": LABELS[arm],
                    "rows": name,
                    "difference": interval["difference"],
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                }
            )
    data = pl.DataFrame(out).to_pandas()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "arm:N", title=None, sort=[LABELS[a] for a in ("nowk", "noday", "noorigin", "f", "b")]
        ),
        color=alt.Color("rows:N", legend=alt.Legend(title=None, orient="top")),
        yOffset="rows:N",
    )
    chart = (
        base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("difference:Q", title="CRPS minus all-three-lags CRPS (points of capacity)")
        )
    ).properties(
        width=WIDTH - 200,
        height=240,
        title=alt.TitleParams(
            [
                "Change in score against the forecast with all three lags, six Primaries, leads",
                "to 10 days (below zero is better; bars: 95% Bonferroni intervals, 12 contrasts).",
            ],
            anchor="start",
            fontSize=11,
        ),
    )
    chart.save(FIGURE_DIR / "figure_2_crps_contrasts.svg")


def main() -> None:
    """Write both figures."""
    sweep_figure()
    contrast_figure()


if __name__ == "__main__":
    main()
