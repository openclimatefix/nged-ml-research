# ruff: noqa: E501, C408
"""Experiment 482: the full 14-day horizon of one wind forecast run, for two contrasting runs.

Reads experiment 461's stored `member_plain` p99.9 quantile tables (3 seeds, 11 quantiles, every
scored lead from 0 to 351 hours, the 107,121 scored cases of experiment 303 in
`e00347_s4_data.scored_frame` order). The tables are the seed average before table P, because table
P is fitted to 24 to 240 hours only. For Farm A, B, and C, picks the run with the highest and the
run with the lowest mean actual power among the 53 Tuesday runs that have at least 650 of 702
unmasked half-hours, and draws the actual, the median, and the 50% and 90% bands against lead time
in days, as fractions of the p99.9 capacity. The x-axis carries lead time and no calendar date. The
charts carry no megawatts.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
source <paths of ml-experiments/*/ on PYTHONPATH>
uv run --no-sync python ml-experiments/00482-*/e00482_wind_14day.py
```
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from pathlib import Path  # noqa: E402
from typing import Final  # noqa: E402

import altair as alt  # noqa: E402
import e00347_s4_data as data347  # noqa: E402  # ty: ignore[unresolved-import]
import e00461_common as common461  # noqa: E402  # ty: ignore[unresolved-import]
import e00461_wind_fit as wind_fit  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import plotting.ocf_theme as ocf  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00482-best-models-performance-analysis-injected-shifts-and-wind-references"
    / "wind"
)
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
MIN_KEPT: Final[int] = 650
WIDTH: Final[int] = 640
LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}


def table() -> pl.DataFrame:
    """Return every scored case with its seed-averaged sorted quantiles in fractions of p99.9 capacity."""
    tables = [np.load(wind_fit.PARTS_DIR / f"member_plain_p999_h{s}.npy") for s in wind_fit.SEEDS]
    q = np.sort(np.mean(np.stack(tables), axis=0), axis=1)
    old = {
        r["site"]: float(r["effective_capacity_mw"])
        for r in __import__("e00230_common").site_table().iter_rows(named=True)
    }
    new = json.loads((common461.DATA_DIR / "wind_capacity.json").read_text())
    to_new = {f: float(new[f]["new_mw"]) / old[f] for f in new}
    frame = data347.scored_frame().select(
        "site",
        "masked",
        "lead_hours",
        init=pl.col("power_fcst_init_time"),
        actual=pl.col("actual_norm")
        / pl.col("site").replace_strict(to_new, return_dtype=pl.Float64),
    )
    return frame.with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])


def pick_runs(*, frame: pl.DataFrame, site: str) -> dict[str, pl.DataFrame]:
    """Return the highest-mean-actual and lowest-mean-actual full runs of `site`."""
    runs = (
        frame.filter((pl.col("site") == site) & ~pl.col("masked") & pl.col("actual").is_finite())
        .group_by("init")
        .agg(n=pl.len(), mean_actual=pl.col("actual").mean())
        .filter(pl.col("n") >= MIN_KEPT)
        .sort("mean_actual")
    )
    picks = {"calm": runs["init"][0], "stormy": runs["init"][-1]}
    return {
        kind: frame.filter(
            (pl.col("site") == site)
            & (pl.col("init") == init)
            & ~pl.col("masked")
            & pl.col("actual").is_finite()
        ).sort("lead_hours")
        for kind, init in picks.items()
    }


def chart(*, run: pl.DataFrame, farm: str, kind: str) -> alt.Chart:
    """Return the 14-day chart of one run."""
    d = run.with_columns(day=pl.col("lead_hours") / 24.0)
    x = alt.X(
        "day:Q",
        title="Lead time (days after the run starts)",
        axis=alt.Axis(values=list(range(0, 16, 2))),
    )
    y = alt.Y("v:Q", title="Power (fraction of capacity)")
    band90 = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.18, color=ocf.BRAND_ORANGE)
        .encode(x=x, y="q05:Q", y2="q95:Q")
    )
    band50 = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.30, color=ocf.BRAND_ORANGE)
        .encode(x=x, y="q25:Q", y2="q75:Q")
    )
    med = (
        alt.Chart(d.with_columns(v=pl.col("q50")))  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2)
        .encode(x=x, y=y)
    )
    act = (
        alt.Chart(d.with_columns(v=pl.col("actual")))  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BLACK_1, strokeWidth=1.5)
        .encode(x=x, y=y)
    )
    return (band90 + band50 + med + act).properties(
        width=WIDTH - 90,
        height=200,
        title=alt.TitleParams(
            [
                f"{farm}, a {kind} fortnight: the bands widen with lead time (black: actual; orange: median,",
                f"50% and 90% bands; {run.height} half-hours; before table P).",
            ],
            anchor="start",
        ),
    )


def main() -> None:
    """Write the six charts."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    frame = table()
    rows = []
    for site, farm in LABEL.items():
        for kind, run in pick_runs(frame=frame, site=site).items():
            c = chart(run=run, farm=farm, kind=kind)
            name = f"wind_14day_{kind}_{farm.replace(' ', '_')}"
            c.save(OUT / f"{name}.svg")
            c.save(OUT / "png" / f"{name}.png", scale_factor=1.5)
            cover90 = float(((run["actual"] >= run["q05"]) & (run["actual"] <= run["q95"])).mean())  # ty: ignore[invalid-argument-type]
            rows.append(dict(farm=farm, kind=kind, half_hours=run.height, cover90=cover90,
                             mean_actual=float(run["actual"].mean())))  # fmt: skip  # ty: ignore[invalid-argument-type]
    pl.DataFrame(rows).write_csv(OUT / "wind_14day_runs.csv")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
