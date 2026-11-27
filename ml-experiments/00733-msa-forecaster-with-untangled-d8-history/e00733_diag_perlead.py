# ruff: noqa: E501
"""Experiment 733: post hoc per-lead reading of the C4 contrast (report-only).

Splits the sign Brier score of Y0, YU, and the persistence rule B-hist by lead day 2 to 10, and the
paired contrast YU minus Y0 likewise. It reuses the scoring's rows, seeds, month-block and segment
bootstrap, and widening (`e00733_scoring_synth`, `e00733_score`), at the plan's level of 98.75%. The
intervals are not adjusted for nine lead days beyond that level. The reading changes no verdict.

Run it through `run.sh --go e00733_diag_perlead.py`; it writes `diagnostics/per_lead.json` and
`diagnostics/per_lead.svg` under the experiment's data folder.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Final

import e00733_core as core
import e00733_diag as diag
import e00733_paths as paths
import e00733_score as sc
import e00733_scoring_synth as synth
import e00733_synth as sy
import e00733_world as world
import numpy as np

# isort: split
import altair as alt
import e00729_figures as figures729
import polars as pl

SERIES: Final = {"Y0": "Y0", "YU": "YU", "B-hist": "persistence rule (B-hist)"}
"""The sign Brier series read by lead day, and the label of each."""
COLOURS: Final = {"Y0": figures729.GREY, "YU": figures729.BLUE, "B-hist": figures729.ORANGE}


def reading_dict(*, reading: Any) -> dict[str, float]:
    """Return a contrast reading as a point, an interval, and the number of rows."""
    return {
        "point": float(reading.reading.point), "lower": float(reading.reading.lower),
        "upper": float(reading.reading.upper), "n": int(reading.n_rows),
        "widening": float(reading.widening),
    }  # fmt: skip


def per_lead_readings(*, root: Path, n_resamples: int) -> dict[str, Any]:
    """Return the sign Brier readings by lead day, read from the stored test-bank archives.

    Args:
        root: The experiment's data folder.
        n_resamples: The bootstrap resamples of each reading.

    Returns:
        `by_lead` (lead day to a series' reading, plus `YU-Y0`), and `pooled` (the same readings on
        all scored rows, whose `YU-Y0` must equal C4 as scored).
    """
    rows = "test"
    arms = {
        a: synth.read_arm(root=root, arm=a, seeds=synth.seeds_held(root=root, arm=a, rows=rows), rows=rows)
        for a in ("Y0", "YU")
    }  # fmt: skip
    if any(v is None for v in arms.values()):
        msg = "Y0 and YU must hold every seed on the test bank"
        raise sc.ScoreError(msg)
    y0, yu = arms["Y0"], arms["YU"]
    assert y0 is not None
    assert yu is not None
    segments, _ = world.read_world(root=root, kind="test")
    index = sy.enumerate_samples(segments=segments)
    import e00733_adapters as adapters

    meta = world.read_meta(root=root)["constants"]
    base = synth.baseline_probabilities(
        segments=segments, index=index, clim=synth.clim_from_limits(root=root), d8_index=adapters.d8_index(),
        target_std=float(meta["std"]),
    )  # fmt: skip
    truth = yu["sign_truth"]
    keep = synth.scored_mask(lead_index=index.lead_index)[:, None] & np.isfinite(truth)
    srows = sc.synthetic_rows(
        seg=index.seg, origin_day=index.origin_day, seg_month=synth.segment_months(segments=segments)
    )  # fmt: skip
    probabilities = {"Y0": y0["p_neg"], "YU": yu["p_neg"], "B-hist": base["b_hist"]}
    brier = {
        name: synth.masked(values=(np.asarray(p) - truth) ** 2, keep=keep & np.isfinite(p))
        for name, p in probabilities.items()
    }
    row_lead = diag.lead_day_of_rows(lead_index=index.lead_index)

    def read_all(*, day: int | None) -> dict[str, Any]:
        def part(values: np.ndarray) -> np.ndarray:
            return (
                values
                if day is None
                else diag.restrict_to_lead_day(values=values, row_lead_day=row_lead, day=day)
            )

        out: dict[str, Any] = {}
        for name, values in brier.items():
            first = part(values)
            zero = np.where(np.isfinite(first), 0.0, np.nan)
            out[name] = reading_dict(
                reading=sc.synthetic_contrast(rows=srows, first=first, second=zero, n_resamples=n_resamples)
            )  # fmt: skip
        out["YU-Y0"] = reading_dict(
            reading=sc.synthetic_contrast(
                rows=srows,
                first=part(brier["YU"]),
                second=part(brier["Y0"]),
                n_resamples=n_resamples,
            )
        )
        return out

    return {
        "by_lead": {str(d): read_all(day=d) for d in core.SYNTHETIC_LEAD_DAYS},
        "pooled": read_all(day=None),
        "n_resamples": n_resamples,
    }


def per_lead_rows(*, readings: dict[str, Any]) -> pl.DataFrame:
    """Return one row per lead day and series: the point and the interval of its Brier score."""
    out = [
        {
            "lead_day": int(day),
            "series": name,
            **{k: record[name][k] for k in ("point", "lower", "upper")},
        }
        for day, record in readings["by_lead"].items()
        for name in (*SERIES, "YU-Y0")
    ]
    return pl.DataFrame(out)


def per_lead_chart(*, rows: pl.DataFrame) -> alt.VConcatChart:
    """Return the two panels: the Brier score of three forecasts by lead day, and YU minus Y0."""
    days = sorted(rows["lead_day"].unique().to_list())
    x = alt.X(
        "lead_day:Q", title="Lead day", axis=alt.Axis(values=days),
        scale=alt.Scale(domain=[days[0] - 0.5, days[-1] + 0.5]),
    )  # fmt: skip
    levels = rows.filter(pl.col("series") != "YU-Y0")
    names = list(SERIES)
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=[COLOURS[n] for n in names]),
        legend=alt.Legend(orient="top", title=None, labelExpr="datum.value == 'B-hist' ? 'persistence rule' : datum.value"),
    )  # fmt: skip
    top_base = alt.Chart(levels).encode(x=x, color=colour)  # ty: ignore[unresolved-attribute]
    top = (
        top_base.mark_line(aria=False).encode(
            y=alt.Y("point:Q", title="Sign Brier score (smaller is better)", scale=alt.Scale(zero=False))
        )  # ty: ignore[unresolved-attribute]
        + top_base.mark_circle(size=45, aria=False).encode(y="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=figures729.WIDTH_PX, height=170,
        title=alt.TitleParams(
            [
                "Sign Brier score of the forecast that the net flow is negative, by lead day",
                "Synthetic test days only. Dot: seed-mean score. Intervals are in the panel below.",
            ],
            anchor="start", fontSize=12, subtitleFontSize=11,
        ),
    )  # fmt: skip
    diff = rows.filter(pl.col("series") == "YU-Y0")
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    d_base = alt.Chart(diff).encode(x=x)  # ty: ignore[unresolved-attribute]
    bottom = (
        zero
        + d_base.mark_rule(color=figures729.BLUE).encode(y=alt.Y("lower:Q", title="YU minus Y0 (smaller is better)"), y2="upper:Q")  # ty: ignore[unresolved-attribute]
        + d_base.mark_circle(size=55, color=figures729.BLUE, aria=False).encode(y="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=figures729.WIDTH_PX, height=150,
        title=alt.TitleParams(
            [
                "Paired difference in sign Brier score, YU minus Y0 (YU has the history signals)",
                "Below the dashed line YU is better. Post hoc, not adjusted for nine lead days.",
            ],
            anchor="start", fontSize=12, subtitleFontSize=11,
        ),
    )  # fmt: skip
    return alt.vconcat(top, bottom).resolve_scale(color="independent")


def main(argv: list[str] | None = None) -> int:
    """Read the per-lead scores and write `per_lead.json` and `per_lead.svg`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-resamples", type=int, default=core.N_RESAMPLES_SYNTHETIC)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)
    paths.pin_threads()
    root = args.data_root or paths.data_root()
    folder = diag.diagnostics_folder(root=root)
    readings = per_lead_readings(root=root, n_resamples=args.n_resamples)
    readings["label"] = (
        "post hoc, report-only; intervals at the plan's level, not adjusted for 9 leads"
    )
    readings["level"] = core.LEVEL
    (folder / "per_lead.json").write_text(json.dumps(readings, indent=1))
    figures729.save_svg(
        chart=per_lead_chart(rows=per_lead_rows(readings=readings)), path=folder / "per_lead.svg"
    )
    for day, record in readings["by_lead"].items():
        r = record["YU-Y0"]
        print(
            f"lead day {day}: YU-Y0 {r['point']:+.5f} [{r['lower']:+.5f}, {r['upper']:+.5f}] n={r['n']}"
        )
    p = readings["pooled"]["YU-Y0"]
    print(f"pooled YU-Y0 {p['point']:+.7f} [{p['lower']:+.5f}, {p['upper']:+.5f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
