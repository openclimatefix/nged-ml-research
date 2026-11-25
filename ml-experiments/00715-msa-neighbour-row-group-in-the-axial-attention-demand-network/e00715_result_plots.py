# ruff: noqa: E501
"""Experiment 715: plots 7 and 8 of the plan and one report-only extra figure, drawn from stored files, with no fit.

Usage, through `run.sh`, never `uv run` (set `DATA_PATH_INTERNAL` first, and put the frozen
experiment 713 folder on `EXTRA_PYTHONPATH`, as for the scorer):

    EXTRA_PYTHONPATH=<frozen 713 folder> SLOT_MEM=12G run.sh e00715_result_plots.py plot8
    EXTRA_PYTHONPATH=<frozen 713 folder> SLOT_MEM=12G run.sh e00715_result_plots.py plot7 --hosts D11,D19
    EXTRA_PYTHONPATH=<frozen 713 folder> SLOT_MEM=12G run.sh e00715_result_plots.py lead-bands

**Read-only.** The script reads the stored forecasts of M*, F4, and F4-R through the scorer's own
functions and `score/headline.json`. It writes only SVG files under `plots/` in this experiment's data
folder, and no fit, no GPU, and no write to any other folder happen.

- `plot8` is the plan's plot 8: N1 and N2 by Primary and by fold, with each Primary's number of
  neighbours in a second panel. Self-check: per fold, the mean of the paired differences over the six
  signed-MW Primaries, and over the eight MVA-only Primaries, equals `headline.json`'s `monthly` values.
- `plot7` is the plan's plot 7: the first forecast origin of the fold `--fold`, for one Primary of each
  group (`--hosts SIX_LABEL,EIGHT_LABEL`), F4 and M* (seed 0) against the cleaned power. **Choose the
  hosts by hand, among Primaries that host no metered generator.** The metadata table gives one
  Primary a note about generation, and the script refuses it. The table cannot identify the others.
- `lead-bands` is a report-only extra figure, not a plan plot: N1 and N2 by lead band. Self-check: the
  24 h to 240 h row reproduces `headline.json`, and the three bands' row counts sum to its row counts.

**Anonymisation.** Plot 8 and the extra figure aggregate Primaries, which are substations. Plot 7
shows one substation's own series under its D label with days counted from the origin. No axis
carries a calendar date, and the folds are numbered 1 to 8.
After running, optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline FILE`.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00715_arms as arms
import e00715_neighbours as nb
import e00715_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

sc = score.sc
ROOT: Final[Path] = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)
OUT: Final[Path] = ROOT / "plots"
HEADLINE: Final[Path] = ROOT / "score" / "headline.json"
METADATA: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
GROUP_NAMES: Final = {
    "six": "The six signed-MW Primaries",
    "eight": "The eight MVA-only Primaries",
}
N1_LABEL: Final = "N1: F4 minus M*"
N2_LABEL: Final = "N2: F4 minus F4-R"
CONTRASTS: Final = {"N1": N1_LABEL, "N2": N2_LABEL}
COLOURS: Final = {N1_LABEL: ocf.DATA_BLUE, N2_LABEL: ocf.BRAND_ORANGE}
DEFINITIONS: Final = [
    "F4: neural network with neighbour rows.",
    "M*: the baseline of experiment 713, without neighbour rows.",
    "F4-R: F4 with its neighbours rewired to other Primaries (a placebo).",
]
UNIT: Final = "CRPS difference in points of capacity; below zero means F4 forecasts better."
SCOPE_CELL: Final = [
    "Each cell: one Primary's mean CRPS difference in one fold (one month; the 8 folds are",
    "November 2025 to June 2026), for the forecast averaged over 2 seeds, on leads of 24 h",
    "up to but not including 240 h.",
]
# Hosts vetted for plot 7. No table records which Primary a metered generator connects to, so the
# evidence is circumstantial: experiment 722 found no metered generator inside these Primaries'
# service-area polygons and no `unresolved` generator fit, and NGED's monitor-direction file marks no
# reverse power flow. D8, D12, D22, D14, and D4 are never allowed.
ALLOWED_HOSTS: Final = {"six": ("D11",), "eight": ("D19", "D21")}
TOL: Final = 1e-9
WIDTH_PX: Final = CONTENT_WIDTH_PX
BAND_HEADLINE: Final = "24-240 h (headline)"


def load_headline() -> dict:
    """Return `score/headline.json` after checking that it belongs to this scorer."""
    headline = json.loads(HEADLINE.read_text())
    assert headline["level"] == score.LEVEL, (headline["level"], score.LEVEL)
    for key in CONTRASTS:
        for group in GROUP_NAMES:
            entry = headline[key][group]
            assert entry["reference"] == reference_of(key), (key, group)
            assert len(entry["monthly"]) == len(arms.ALL_FOLDS)
            assert all(math.isfinite(v) for v in entry["monthly"])
    return headline


def reference_of(key: str) -> str:
    """Return the reference arm of the contrast `key`."""
    return score.star() if key == "N1" else "F4-R"


def labels_by_id() -> dict[int, str]:
    """Return each `time_series_id`'s D label."""
    sites = sc.k578.sites()
    return dict(zip(sites["time_series_id"].to_list(), sites["label"].to_list(), strict=True))


def save(*, chart: alt.TopLevelMixin, name: str) -> Path:
    """Write the chart under `plots/` and return the path."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    chart.save(path)
    return path


def zero_rule(*, axis: str) -> alt.Chart:
    """Return a dashed rule at zero without data of its own, so that a facet can own the data."""
    rule = alt.Chart().mark_rule(color="#555555", strokeDash=[4, 3], aria=False)
    return rule.encode(**{axis: alt.datum(0)})  # ty: ignore[unresolved-attribute]


def plot8() -> Path:
    """Draw plot 8: N1 and N2 by Primary and by fold, and each Primary's number of neighbours."""
    headline = load_headline()
    groups = score.group_ids()
    label_of = labels_by_id()
    group_of = {i: g for g in GROUP_NAMES for i in groups[g].tolist()}
    rows = []
    for key, contrast_label in CONTRASTS.items():
        per_seed, averaged = score.pair(
            treatment="F4", reference=reference_of(key), months=arms.ALL_FOLDS
        )
        del per_seed
        band = sc.HEADLINE_BAND_HOURS
        kept = {
            name: frame.filter((pl.col("lead_hours") >= band[0]) & (pl.col("lead_hours") < band[1]))
            for name, frame in averaged.items()
        }
        del averaged
        both = sc.paired(treatment=kept["F4"], reference=kept[reference_of(key)], metric="crps")
        del kept
        both = both.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        months = sorted(both["month"].unique().to_list())
        assert months == sorted(arms.ALL_FOLDS), months
        for group in GROUP_NAMES:
            sub = both.filter(pl.col("time_series_id").is_in(groups[group].tolist()))
            by_month = sub.group_by("month").agg(pl.col("d").mean()).sort("month")["d"].to_numpy()
            expected = np.array(headline[key][group]["monthly"])
            assert np.allclose(by_month, expected, atol=TOL, rtol=0.0), (key, group)
            assert sub.height == headline[key][group]["n_rows"], (key, group)
        per_primary = both.group_by("time_series_id", "month").agg(pl.col("d").mean())
        del both
        for r in per_primary.iter_rows(named=True):
            if r["time_series_id"] not in group_of:
                continue
            rows.append(
                {
                    "contrast": contrast_label,
                    "primary": label_of[r["time_series_id"]],
                    "fold": months.index(r["month"]) + 1,
                    "difference": float(r["d"]),
                }
            )
    frame = pl.DataFrame(rows)
    order = [*sc.SIX, *sc.EIGHT]
    assert sorted(frame["primary"].unique().to_list()) == sorted(order)
    assert frame.height == 2 * 14 * 8, frame.height
    largest = float(frame["difference"].abs().max())  # ty: ignore[invalid-argument-type]
    limit = float(np.ceil(largest * 20) / 20)
    colour = alt.Color(
        "difference:Q",
        title="CRPS difference (points of capacity)",
        scale=alt.Scale(
            domain=[-limit, 0, limit], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(orient="top", gradientLength=200),
    )
    height = 14 * 17

    def heat(*, label: str, show_y: bool) -> alt.Chart:
        y_axis = alt.Axis() if show_y else alt.Axis(labels=False, ticks=False)
        chart = alt.Chart(frame.filter(pl.col("contrast") == label)).mark_rect(aria=False)
        return chart.encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("fold:O", title="Fold (1 to 8, in time order)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("primary:N", sort=order, title="Primary" if show_y else None, axis=y_axis),
            color=colour,
        ).properties(width=190, height=height, title=label)

    degrees = pl.DataFrame({"primary": order, "neighbours": [nb.DEGREES[p] for p in order]})
    bars = (
        alt.Chart(degrees)
        .mark_bar(color="#888888", aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("primary:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("neighbours:Q", title="Neighbours", axis=alt.Axis(values=[0, 2, 4, 6])),
        )
        .properties(width=70, height=height, title="Neighbours")
    )
    chart = alt.hconcat(
        heat(label=N1_LABEL, show_y=True), heat(label=N2_LABEL, show_y=False), bars
    ).resolve_scale(color="shared")
    chart = chart.properties(
        title=alt.TitleParams(
            "F4 minus M* and F4 minus F4-R, by Primary and fold",
            subtitle=[
                "Orange: F4 forecasts worse. Blue: F4 forecasts better. White: no difference.",
                f"Rows 1 to 6: the six signed-MW Primaries ({', '.join(sc.SIX)});",
                "rows 7 to 14: the eight MVA-only Primaries.",
                UNIT,
                *SCOPE_CELL,
                *DEFINITIONS,
            ],
        )
    )
    return save(chart=chart, name="plot_08_contrasts_by_primary_and_fold.svg")


def generation_noted_hosts() -> set[str]:
    """Return the D labels of the Primaries whose metadata note mentions generation."""
    meta = pl.read_parquet(METADATA)
    note = pl.col("information").fill_null("").str.to_lowercase().str.contains("generat")
    ids = meta.filter(note)["time_series_id"].to_list()
    label_of = labels_by_id()
    return {label_of[i] for i in ids if i in label_of}


def plot7(*, hosts: list[str], fold: str) -> Path:
    """Draw plot 7: the first forecast origin of `fold` for one Primary of each group, F4 against M*."""
    assert fold in arms.ALL_FOLDS, fold
    assert len(hosts) == 2, "give one D label of each group: --hosts D11,D19"
    assert hosts[0] in ALLOWED_HOSTS["six"], (
        f"{hosts[0]} is not vetted for the six: {ALLOWED_HOSTS}"
    )
    assert hosts[1] in ALLOWED_HOSTS["eight"], (
        f"{hosts[1]} is not vetted for the eight: {ALLOWED_HOSTS}"
    )
    refused = set(hosts) & generation_noted_hosts()
    assert not refused, f"{refused} has a metadata note about generation; choose another Primary"
    id_of = {v: k for k, v in labels_by_id().items()}
    truth = sc.truth_stack()
    scored = {}
    for name in ("F4", score.star()):
        raw = score.load_arm(arm=name, seeds=(0,), months=(fold,))
        scored[name] = sc.add_losses(frame=raw, truth=truth)
    origin = scored["F4"]["power_fcst_init_time"].min()
    panels = []
    for host, group in zip(hosts, GROUP_NAMES, strict=True):
        parts = []
        for name, label in (("F4", "F4"), (score.star(), "M* (baseline)")):
            sub = scored[name].filter(
                (pl.col("time_series_id") == id_of[host])
                & (pl.col("power_fcst_init_time") == origin)
            )
            assert sub.height == 14 * 48, (host, name, sub.height)  # no gap in the cleaned demand
            parts.append(
                sub.select(
                    days=pl.col("lead_hours") / 24.0,
                    median=pl.col("q50") * 100.0,
                    low=pl.col("q05") * 100.0,
                    high=pl.col("q95") * 100.0,
                    actual=pl.col("actual") * 100.0,
                ).with_columns(forecast=pl.lit(label))
            )
        frame = pl.concat(parts)
        x = alt.X(
            "days:Q",
            title="Days after the forecast origin",
            scale=alt.Scale(domain=[0, 14]),
            axis=alt.Axis(values=list(range(0, 15, 2))),
        )
        base = alt.Chart(frame).encode(x=x)
        band = (
            base.transform_filter(alt.datum.forecast == "F4")
            .mark_area(opacity=0.2, color=ocf.BRAND_ORANGE, aria=False)
            .encode(y=alt.Y("low:Q", title="Demand (% of capacity)"), y2="high:Q")  # ty: ignore[unresolved-attribute]
        )
        lines = base.mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y="median:Q",
            color=alt.Color(
                "forecast:N",
                title=None,
                scale=alt.Scale(
                    domain=["F4", "M* (baseline)"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        actual = (
            base.transform_filter(alt.datum.forecast == "F4")
            .mark_line(color="#555555", strokeDash=[2, 2], aria=False)
            .encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(band, lines, actual).properties(
                width=WIDTH_PX - 120, height=150, title=f"{GROUP_NAMES[group]}: Primary {host}"
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "One 14-day forecast against the cleaned demand, for one Primary of each group",
            subtitle=[
                "Solid lines: median forecast of F4 (orange, with its 5% to 95% band) and of M* (blue); seed 0.",
                "Dashed grey line: cleaned demand. Day 0 is the forecast origin (the first of the fold);",
                "no calendar date is shown. The day-ahead forecast is days 0.5 to 1.5 (leads of 12 h to 36 h).",
                *DEFINITIONS,
            ],
        )
    )
    return save(chart=chart, name="plot_07_forecast_example.svg")


def lead_bands() -> Path:
    """Draw the report-only extra figure: N1 and N2 by lead band with 98.75% intervals."""
    headline = load_headline()
    groups = score.group_ids()
    bands = {**score.BANDS, BAND_HEADLINE: sc.HEADLINE_BAND_HOURS}
    rows = []
    for key, contrast_label in CONTRASTS.items():
        per_seed, averaged = score.pair(
            treatment="F4", reference=reference_of(key), months=arms.ALL_FOLDS
        )
        counts = dict.fromkeys(GROUP_NAMES, 0)
        for band_name, band in bands.items():
            for group in GROUP_NAMES:
                c = sc.contrast(
                    per_seed=per_seed, averaged=averaged, treatment="F4", reference=reference_of(key),
                    group=group, groups=groups, level=score.LEVEL, band=band,
                )  # fmt: skip
                assert c["n_rows"] > 0, (key, band_name, group)
                if band_name == BAND_HEADLINE:
                    for field in ("point", "lower", "upper"):
                        assert abs(c[field] - headline[key][group][field]) < TOL, (
                            key,
                            group,
                            field,
                        )
                    assert c["n_rows"] == headline[key][group]["n_rows"], (key, group)
                else:
                    counts[group] += c["n_rows"]
                rows.append(
                    {
                        "contrast": contrast_label,
                        "group": GROUP_NAMES[group],
                        "band": band_name,
                        "point": float(c["point"]),
                        "lower": float(c["lower"]),
                        "upper": float(c["upper"]),
                    }
                )
        for group in GROUP_NAMES:
            assert counts[group] == headline[key][group]["n_rows"], (key, group, counts[group])
        del per_seed, averaged
    frame = pl.DataFrame(rows)
    assert frame.height == 2 * 4 * 2, frame.height
    base = alt.Chart().encode(
        y=alt.Y("band:O", sort=list(bands), title="Lead band (hours after the origin)"),
        yOffset=alt.YOffset("contrast:N"),
        color=alt.Color(
            "contrast:N",
            title=None,
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(orient="top"),
        ),
    )
    layer = alt.layer(
        zero_rule(axis="x"),
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower:Q", title="CRPS difference (points of capacity)"), x2="upper:Q"
        ),
        base.mark_point(filled=True, size=60, aria=False).encode(x="point:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH_PX - 220, height=130)
    chart = layer.facet(
        data=frame,
        row=alt.Row(
            "group:N",
            title=None,
            sort=list(GROUP_NAMES.values()),
            header=alt.Header(labelOrient="top", labelAnchor="start", labelAngle=0),
        ),
    ).properties(
        title=alt.TitleParams(
            "Report-only: F4 minus M* and F4 minus F4-R by lead band",
            subtitle=[
                UNIT,
                "Dot: difference. Line: 98.75% interval (the headline's level) from resampling whole months.",
                "Dashed line: no difference. Over 8 folds and 2 seeds.",
                "Report-only, not a counted contrast: only the 24-240 h row is the headline.",
                *DEFINITIONS,
            ],
        )
    )
    return save(chart=chart, name="extra_contrasts_by_lead_band_report_only.svg")


def main() -> int:
    """Draw one figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=["plot8", "plot7", "lead-bands"])
    parser.add_argument("--hosts", default="")
    parser.add_argument("--fold", default="2026-01")
    args = parser.parse_args()
    if args.figure == "plot8":
        path = plot8()
    elif args.figure == "plot7":
        path = plot7(hosts=[h for h in args.hosts.split(",") if h], fold=args.fold)
    else:
        path = lead_bands()
    print("wrote", path, "; now run: npx svgo@4 --multipass --precision=1 --final-newline", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
