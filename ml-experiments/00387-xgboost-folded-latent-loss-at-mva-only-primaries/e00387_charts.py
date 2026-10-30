"""Draw experiment 387's figures: Step 0 histogram, a sunny week at D12, the gate, the contrasts.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:ml-experiments/00387-* \
        uv run --no-sync python ml-experiments/00387-*/e00387_charts.py
"""

from datetime import timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00387_data as data
import e00387_fit as fit
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "xgboost-folded-latent-loss-at-mva-only-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00387-{SLUG}" / "work"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00387-{SLUG}"
WIDTH: Final[int] = 640
LEAD_BAND: Final[tuple[float, float]] = (12.0, 36.0)


def _chart(frame: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(frame)


def week_figure(*, mode: str, site: str, specs: dict[str, str], title: list[str], path: Path,
                first: str, last: str, truth: bool) -> None:
    """Draw one run's first 7 days of median forecasts, in points of capacity.

    The run is the one between `first` and `last` whose next 7 days hold the most ERA5 shortwave.
    """
    series = common.series_table().filter(pl.col("site") == site)
    arrays = common.load_series_arrays(series=series)[site]
    synthetic = fit.synthetic_for(mode=mode, arrays=arrays)
    grid = pl.DataFrame({"valid_time": arrays.valid_time,
                         "shortwave": data.shortwave_of(arrays=arrays),
                         "observed": (arrays.power if synthetic is None else synthetic.arrays.power) * 100})
    if synthetic is not None:
        grid = grid.with_columns(truth=pl.Series(synthetic.truth * 100))
    folder0 = fit.WORK / "preds" / mode.replace(":", "_") / specs["plain"]
    runs = (
        pl.concat([pl.read_parquet(f, columns=["power_fcst_init_time"]) for f in sorted(folder0.glob(f"{site}_*.parquet"))])
        .unique().sort("power_fcst_init_time")["power_fcst_init_time"].to_list()
    )
    lo, hi = (pl.Series([x]).str.to_datetime(time_zone="UTC")[0] for x in (first, last))
    best, best_sum = None, -1.0
    for run in (r for r in runs if lo <= r <= hi):
        window = grid.filter(pl.col("valid_time") > run, pl.col("valid_time") <= run + timedelta(days=7))
        total = float(window["shortwave"].sum())
        if total > best_sum:
            best, best_sum = run, total
    start = best
    end = start + timedelta(days=7)
    parts = []
    for label, spec in specs.items():
        folder = fit.WORK / "preds" / mode.replace(":", "_") / spec
        frame = pl.concat([pl.read_parquet(f) for f in sorted(folder.glob(f"{site}_*.parquet"))])
        frame = frame.filter(pl.col("power_fcst_init_time") == start, pl.col("valid_time") <= end)
        parts.append(frame.select("valid_time", value=pl.col("q50") * 100).with_columns(series=pl.lit(f"{label}: median (apparent power)")))
        if "folded" in label:
            parts.append(frame.select("valid_time", value=pl.col("latent_mean") * 100).with_columns(series=pl.lit(f"{label}: recovered signed power")))
    base = grid.filter(pl.col("valid_time") > start, pl.col("valid_time") <= end)
    parts.append(base.select("valid_time", value=pl.col("observed")).with_columns(series=pl.lit("metered apparent power")))
    if truth:
        parts.append(base.select("valid_time", value=pl.col("truth")).with_columns(series=pl.lit("true signed power")))
    frame = pl.concat(parts).with_columns(
        day=(pl.col("valid_time") - start).dt.total_minutes() / 1440
    )
    domain = list(frame["series"].unique(maintain_order=True))
    colours = [ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE, ocf.DATA_AMBER]
    chart = (
        _chart(frame.to_pandas())
        .mark_line(clip=True, aria=False)
        .encode(
            x=alt.X("day:Q", title="Days after the forecast was made", axis=alt.Axis(values=list(range(8)))),
            y=alt.Y("value:Q", title="Points of effective capacity"),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colours[: len(domain)]),
                            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=620)),
        )
        .properties(width=WIDTH, height=300, title=alt.TitleParams(text=title[0], subtitle=title[1:]))
    )
    zero = _chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    (chart + zero).save(path)


def histogram_figure(*, path: Path) -> None:
    """Draw how close the training-period readings get to zero at D12 and two references."""
    frame = pl.read_parquet(WORK / "step0.parquet").filter(
        pl.col("valid_time") < data.TRAIN_END, pl.col("reading").abs() < 10
    )
    frame = frame.with_columns(
        series=pl.when(pl.col("site") == "D12").then(pl.lit("D12 (MVA)"))
        .when(pl.col("site") == "D2").then(pl.lit("D2 (MVA)"))
        .when(pl.col("site") == "D10").then(pl.lit("D10 (signed MW, absolute value)"))
        .otherwise(pl.lit("other")),
        reading=pl.col("reading").abs(),
    ).filter(pl.col("series") != "other", pl.col("site") != "D3")
    chart = (
        _chart(frame.to_pandas())
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("reading:Q", bin=alt.Bin(step=0.5), title="Reading, points of effective capacity (below 10 shown)"),
            y=alt.Y("count():Q", title="Half-hours in the training period", scale=alt.Scale(type="symlog")),
            color=alt.Color("series:N", scale=alt.Scale(
                domain=["D12 (MVA)", "D2 (MVA)", "D10 (signed MW, absolute value)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=620)),
        )
        .properties(width=WIDTH, height=260, title=alt.TitleParams(
            text="D12's lowest readings stop at about 4 points; a signed meter reaches zero",
            subtitle=["Training readings before 2025-07-01 at or below 10 points of effective capacity.",
                      "Log-like count axis."]))
    )
    chart.save(path)


def contrast_figure(*, path: Path) -> None:
    """Draw each contrast by Primary with its 95% interval."""
    names = {"C1": "F2 (Q0 = 2) minus plain", "C3": "F2 (Q0 = 2) minus F0 (unfolded custom loss)",
             "F0_vs_plain": "F0 minus plain"}
    frames = []
    for key, label in names.items():
        f = pl.read_parquet(WORK / "scores" / f"by_primary_{key}.parquet")
        frames.append(f.select("label", "difference", "lower", "upper", contrast=pl.lit(label)))
    frame = pl.concat(frames).to_pandas()
    order = ["D12", "D8", "D22", "D19", "D2", "D3", "D4", "D21"]
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]
    base = _chart(frame).encode(
        y=alt.Y("label:N", sort=order, title="MVA-only Primary substation"),
        color=alt.Color("contrast:N", scale=alt.Scale(domain=list(names.values()), range=colours),
                        legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=620)),
        yOffset="contrast:N",
    )
    points = base.mark_point(filled=True, size=70).encode(
        x=alt.X("difference:Q", title="CRPS difference, points of capacity (below zero is better)"))
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")
    zero = _chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    (bars + points + zero).properties(
        width=WIDTH, height=360,
        title=alt.TitleParams(text="The folded loss does not lower the CRPS at any MVA-only Primary",
                              subtitle=["Dot: difference in mean CRPS over days 1 to 10, monthly refits. Line: 95% interval from",
                                        "resampling whole months. Dashed rule: no difference."]),
    ).save(path)


def main() -> None:
    """Draw every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    histogram_figure(path=DOCS / "figure_1_low_readings.svg")
    week_figure(
        mode="real", site="D12", first="2026-04-01", last="2026-06-25", truth=False,
        specs={"plain": "plain", "folded (Q0 = 2)": "f2_k1_m0_q2"},
        title=["D12, a sunny week: the folded model recovers a signed power but does not fit the readings better",
               "Median forecasts from one run in April to June 2026, the sunniest week of the runs scored."],
        path=DOCS / "figure_3_d12_sunny_week.svg",
    )
    week_figure(
        mode="syn:none:1.0", site="D10", first="2025-07-01", last="2025-09-20", truth=True,
        specs={"plain": "plain", "folded (Q0 = 2)": "f2_k1_m0_q2"},
        title=["Known answer at D10: the folded model finds the reverse flow that was injected",
               "Synthetic apparent power from signed D10 plus an injected solar plant, one run in July to September 2025, the sunniest week of the runs scored."],
        path=DOCS / "figure_2_known_answer_d10.svg",
    )
    contrast_figure(path=DOCS / "figure_4_contrasts.svg")


if __name__ == "__main__":
    main()
