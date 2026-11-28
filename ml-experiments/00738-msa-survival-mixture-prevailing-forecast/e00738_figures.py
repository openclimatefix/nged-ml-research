"""Result figures of experiment 738, written under the data folder only (private).

No figure shows a metered generator's series, a calendar date, or the real series of D8. The time
series plots take forecast frames from the prediction files, with axes in days from the shift's
onset. The injected panels draw each arm's median and the true injected power. The real-event plots
take the windows of NGED's logged events (the caller supplies them) and skip D8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import e00738_common as common
import e00738_fit as fit
import polars as pl
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN, DATA_PURPLE, DATA_SKY

WIDTH: Final[int] = 600
HEIGHT: Final[int] = 220
FIG_DIR: Final[str] = "figures"
ARM_COLOURS: Final[dict[str, str]] = {
    "M1": DATA_BLUE,
    "W0": DATA_GREEN,
    "LP": DATA_PURPLE,
    "LD": DATA_SKY,
    "LM": BRAND_ORANGE,
    "LR": "#8c564b",
    "actual": "#000000",
}


def _save(*, chart: alt.TopLevelMixin, name: str, out: Path | None = None) -> Path:
    path = (out if out is not None else common.OUT) / FIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    chart.save(str(path / f"{name}.svg"))
    return path / f"{name}.svg"


def loss_plots(*, month: str, seed: int, out: Path | None = None) -> list[Path]:
    """Draw the training and validation loss of each fitted arm of one fold and seed.

    Args:
        month: The fold month.
        seed: The seed.
        out: The output folder (`OUT` by default).

    Returns:
        The files written. The pass of the stored state is marked with a rule.
    """
    paths = []
    for arm in common.FIT_ARMS:
        log = fit.read_log(arm=arm, month=month, seed=seed, out=out)
        for i, stage in enumerate(log["stages"]):
            rows = [
                {"pass": k + 1, "loss": float(v), "set": name}
                for name, hist in (
                    ("training", stage["train_history"]),
                    ("validation", stage["val_history"]),
                )
                for k, v in enumerate(hist)
            ]
            base = (
                alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
                .mark_line()
                .encode(
                    alt.X("pass:Q", title="Pass"),
                    alt.Y(
                        "loss:Q",
                        title="Mixture negative log-likelihood",
                        scale=alt.Scale(zero=False),
                    ),
                    alt.Color(
                        "set:N", scale=alt.Scale(range=[DATA_BLUE, BRAND_ORANGE]), title=None
                    ),
                )
            )
            mark = (
                alt.Chart(alt.Data(values=[{"pass": stage["best_pass"]}]))  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeDash=[4, 3])
                .encode(x="pass:Q")
            )
            paths.append(
                _save(
                    chart=(base + mark).properties(
                        width=WIDTH, height=HEIGHT, title=f"{arm}, stage {i + 1}"
                    ),
                    name=f"loss_{arm}_stage{i + 1}_{month}_s{seed}",
                    out=out,
                )
            )
    return paths


def forest_plot(*, report: dict, out: Path | None = None) -> Path:
    """Draw C1, C2, C3, N1, N3, V1, and V2 with their 98.33% (95% for V) t intervals."""
    rows = [
        {
            "reading": k,
            "point": report[k]["point"],
            "lo": report[k]["t"][0],
            "hi": report[k]["t"][1],
        }
        for k in ("C1", "C2", "C3", "N1", "N3", "V1", "V2")
    ]
    base = alt.Chart(alt.Data(values=rows)).encode(
        alt.Y("reading:N", title=None, sort=[r["reading"] for r in rows])
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            alt.X("lo:Q", title="Difference in CRPS (points of capacity)"), alt.X2("hi:Q")
        )
        + base.mark_point(filled=True, color=BRAND_ORANGE).encode(alt.X("point:Q"))  # ty: ignore[unresolved-attribute]
        + alt.Chart(alt.Data(values=[{"zero": 0.0}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[3, 3])
        .encode(x="zero:Q")
    ).properties(width=WIDTH, height=HEIGHT)
    return _save(chart=chart, name="forest", out=out)


def stratum_plot(*, report: dict, out: Path | None = None) -> Path:
    """Draw each arm's CRPS difference from `M1` by stratum of the shift at the origin."""
    rows = [
        {"stratum": s, "arm": a, "difference": v}
        for s, d in report["by_stratum"].items()
        for a, v in d.items()
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            alt.X("stratum:N", title=None, sort=list(report["by_stratum"])),
            alt.Y("difference:Q", title="CRPS minus M1's (points of capacity)"),
            alt.XOffset("arm:N"),
            alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(ARM_COLOURS)[1:6],
                    range=[ARM_COLOURS[a] for a in list(ARM_COLOURS)[1:6]],
                ),
            ),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    return _save(chart=chart, name="strata", out=out)


def age_band_plot(*, report: dict, out: Path | None = None) -> Path:
    """Draw C1, C2, and C3 by the age of the shift at the origin."""
    rows = [
        {"band": b, "contrast": k, "point": r["point"]}
        for b, d in report["by_age_band"].items()
        for k, r in d.items()
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            alt.X(
                "band:N", title="Age of the shift at the origin", sort=list(report["by_age_band"])
            ),
            alt.Y("point:Q", title="Difference in CRPS (points of capacity)"),
            alt.XOffset("contrast:N"),
            alt.Color("contrast:N", scale=alt.Scale(range=[BRAND_ORANGE, DATA_BLUE, DATA_GREEN])),
        )
        .properties(width=WIDTH, height=HEIGHT)
    )
    return _save(chart=chart, name="age_bands", out=out)


def series_plot(
    *,
    frames: dict[str, pl.DataFrame],
    primary_id: int,
    origin: object,
    onset_us: int,
    end_us: int,
    name: str,
    out: Path | None = None,
) -> Path:
    """Draw each arm's median forecast of one origin and Primary, in days from the onset.

    Args:
        frames: Each arm's forecast frame (keys, `actual`, quantiles).
        primary_id: The `time_series_id` to draw (never D8's real series).
        origin: The `power_fcst_init_time` of the origin.
        onset_us: The shift's onset, in microseconds since the epoch.
        end_us: The shift's end.
        name: The file stem.
        out: The output folder (`OUT` by default).

    Returns:
        The file written.
    """
    rows = []
    for arm, f in frames.items():
        sel = f.filter(
            (pl.col("time_series_id") == primary_id) & (pl.col("power_fcst_init_time") == origin)
        )
        for t, q50 in zip(
            sel["valid_time"].dt.epoch("us").to_list(), sel["q50"].to_list(), strict=True
        ):
            rows.append({"day": (t - onset_us) / 8.64e10, "series": arm, "power": 100 * q50})
    first = next(iter(frames.values())).filter(
        (pl.col("time_series_id") == primary_id) & (pl.col("power_fcst_init_time") == origin)
    )
    rows += [
        {"day": (t - onset_us) / 8.64e10, "series": "actual", "power": 100 * a}
        for t, a in zip(
            first["valid_time"].dt.epoch("us").to_list(), first["actual"].to_list(), strict=True
        )
    ]
    rules = (
        alt.Chart(alt.Data(values=[{"day": 0.0}, {"day": (end_us - onset_us) / 8.64e10}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3])
        .encode(x="day:Q")
    )
    chart = (
        alt.Chart(alt.Data(values=rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            alt.X("day:Q", title="Days from the onset of the shift"),
            alt.Y("power:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
            alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                title=None,
            ),
        )
        + rules
    ).properties(width=WIDTH, height=HEIGHT)
    return _save(chart=chart, name=name, out=out)
