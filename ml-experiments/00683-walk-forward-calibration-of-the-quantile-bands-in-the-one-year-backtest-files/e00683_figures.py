# ruff: noqa: E501
"""Figures of experiment 683 (CPU only), written as SVG to the folder given as the first argument.

Run it from the repository root with `PYTHONPATH` set to the folders of experiments 668 and 683.
Substations only appear with identifiers; wind and solar appear only pooled.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import polars as pl
from e00683_calibrate import BAND_LABELS, EXPERIMENT_DIR, FIRST_SCORED_MONTH
from e00683_tables import ARM_NAMES, FILES, load, means, period

ARM_COLOURS = {
    "stored": "#7F7F7F",
    "fixed 1.0934": "#17E58F",
    "scale": "#306BFF",
    "scale + shift": "#FF4901",
}
FILE_TITLES = {
    "primaries_NRA": "Primaries, normal running",
    "primaries_prevailing": "Primaries, prevailing",
    "bsps": "Bulk supply points",
    "gsps": "Grid supply points",
    "wind": "Wind",
    "solar": "Solar (daylight)",
}
ARM_SCALE = alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values()))


def coverage_by_band(*, out: Path) -> None:
    """Headline: pooled q05-to-q95 coverage by lead band for each arm and file, months before July."""
    rows = []
    for file in FILES:
        m = means(frame=period(frame=load(file=file), which="dev"), by=["arm", "band"])
        rows.append(m.with_columns(file=pl.lit(FILE_TITLES[file])))
    frame = pl.concat(rows).with_columns(
        arm_name=pl.col("arm").replace_strict(ARM_NAMES),
        band_name=pl.col("band").replace_strict(dict(enumerate(BAND_LABELS))),
    )
    charts = []
    for title in FILE_TITLES.values():
        sub = frame.filter(pl.col("file") == title).to_pandas()
        line = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1, point=True)
            .encode(
                x=alt.X(
                    "band_name:N",
                    title="Lead band",
                    sort=list(BAND_LABELS),
                    axis=alt.Axis(labelAngle=-30),
                ),
                y=alt.Y(
                    "coverage:Q", title="q05-to-q95 coverage (%)", scale=alt.Scale(domain=[80, 96])
                ),
                color=alt.Color("arm_name:N", title="Arm", scale=ARM_SCALE),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"y": [90.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#000000", strokeWidth=1)
            .encode(y="y:Q")
        )
        charts.append((line + rule).properties(width=280, height=280, title=title))
    chart = alt.concat(*charts, columns=3)
    chart.save(out / "coverage_by_lead_band.svg")


def monthly_primaries(*, out: Path) -> None:
    """Pooled normal-running coverage of the 16 Primaries per origin month, with the 0.86 stop line."""
    m = means(frame=load(file="primaries_NRA"), by=["arm", "origin_month"]).with_columns(
        arm_name=pl.col("arm").replace_strict(ARM_NAMES)
    )
    line = (
        alt.Chart(m.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1, point=True)
        .encode(
            x=alt.X("origin_month:O", title="Origin month", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "coverage:Q", title="q05-to-q95 coverage (%)", scale=alt.Scale(domain=[60, 96])
            ),
            color=alt.Color("arm_name:N", title="Arm", scale=ARM_SCALE),
        )
    )
    stop = (
        alt.Chart(pl.DataFrame({"y": [86.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#000000", strokeWidth=1)
        .encode(y="y:Q")
    )
    nominal = (
        alt.Chart(pl.DataFrame({"y": [90.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeWidth=1)
        .encode(y="y:Q")
    )
    (line + stop + nominal).properties(width=620, height=300).save(
        out / "monthly_primaries_coverage.svg"
    )


def refit_widths(*, out: Path) -> None:
    """Days 10 to 14 band-width ratio at each month start for the eight Primaries, with seed spread."""
    steps = pl.read_csv(EXPERIMENT_DIR / "tables" / "refit_steps.csv")
    seeds = pl.read_csv(EXPERIMENT_DIR / "tables" / "refit_seed_spread.csv")
    spread = seeds.select(
        lo=pl.col("seed_diff_ratio").quantile(0.05), hi=pl.col("seed_diff_ratio").quantile(0.95)
    )
    lo, hi = float(spread["lo"][0]), float(spread["hi"][0])
    band = (
        alt.Chart(pl.DataFrame({"lo": [lo], "hi": [hi]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.15, color="#7F7F7F")
        .encode(y="lo:Q", y2="hi:Q")
    )
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#000000", strokeWidth=1)
        .encode(y="y:Q")
    )
    lines = (
        alt.Chart(steps.with_columns(series=pl.col("time_series_id").cast(pl.String)).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1, point=True)
        .encode(
            x=alt.X("refit_month:O", title="Month of the refit", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "width_ratio:Q",
                title="Band width, first 7 origins over last 7 before",
                scale=alt.Scale(domain=[0.5, 1.6]),
            ),
            color=alt.Color("series:N", title="Primary"),
        )
    )
    (band + one + lines).properties(width=620, height=300).save(out / "refit_width_ratio.svg")


def fans(*, out: Path) -> None:
    """Forecast fans before and after calibration for Primaries 17 and 13: 14 days and the first day."""
    origin = datetime(2026, 1, 14, 12, tzinfo=UTC)
    truth = pl.read_parquet(lib.FOLDER / "ground_truth_primaries.parquet")
    cap = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    arms = {
        0: lib.FOLDER / "primaries.parquet",
        2: EXPERIMENT_DIR / "predictions" / "arm2" / "primaries_NRA.parquet",
    }
    panels = []
    for tid in (17, 13):
        c = float(cap.filter(pl.col("time_series_id") == tid)["capacity"][0])
        obs = truth.filter(pl.col("time_series_id") == tid).select(
            "valid_time", y=pl.col("power") * 100 / c
        )
        for arm, path in arms.items():
            f = pl.scan_parquet(path).filter(
                (pl.col("time_series_id") == tid) & (pl.col("power_fcst_init_time") == origin)
            )
            if arm == 0:
                f = f.filter(pl.col("NRA_or_prevailing") == "NRA")
            f = f.collect().select(
                "valid_time",
                *[(pl.col(q) * 100 / c).alias(q) for q in ("q05", "q25", "q50", "q75", "q95")],
            )
            f = f.join(obs, on="valid_time", how="left").with_columns(
                hours=(pl.col("valid_time") - origin).dt.total_seconds() / 3600,
                arm=pl.lit(ARM_NAMES[arm]),
                primary=pl.lit(f"Primary {tid}"),
            )
            panels.append(f)
    frame = pl.concat(panels)
    for label, hours in (("14_days", 336.0), ("first_day", 24.0)):
        sub = frame.filter(pl.col("hours") <= hours).to_pandas()
        base = alt.Chart().encode(x=alt.X("hours:Q", title="Hours after the forecast origin"))
        band90 = base.mark_area(opacity=0.25, color="#306BFF").encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"
        )
        band50 = base.mark_area(opacity=0.35, color="#306BFF").encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        median = base.mark_line(strokeWidth=1, color="#306BFF").encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        actual = base.mark_line(strokeWidth=1, color="#FF4901").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        chart = (
            (band90 + band50 + median + actual)
            .properties(width=300, height=280)
            .facet(
                row=alt.Row("primary:N", title=None),
                column=alt.Column("arm:N", title=None, sort=["stored", "scale"]),
                data=sub,
            )
        )
        chart.save(out / f"fan_{label}.svg")


def main(out_dir: str) -> None:
    """Write every figure."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coverage_by_band(out=out)
    monthly_primaries(out=out)
    refit_widths(out=out)
    fans(out=out)
    _ = (FIRST_SCORED_MONTH, timedelta)


if __name__ == "__main__":
    main(sys.argv[1])
