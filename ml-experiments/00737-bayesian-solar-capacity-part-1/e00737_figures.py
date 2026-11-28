"""Figures of experiment 737, written under the data folder only (private, never published).

The gallery shows synthetic quantities. The forward-model and fleet figures (6) are drawn before
the first scored fit, the four posterior figures come from the tuning seeds, and the one real-record
figure holds the median capacity path over the six Primaries as a multiple of each Primary's own
mean. No figure shows a metered series, a calendar date, a Primary's own path, or a capacity in
megawatts of a real Primary. Every axis counts hours, days of a week, or quarters.
"""

from __future__ import annotations

import time
from typing import Any, Final

import altair as alt
import numpy as np
import polars as pl
from e00737_common import GROWTH_SD_GRID, N_QUARTERS, OUT, PRIMARIES, time_log
from e00737_fit import (
    PrimaryModel,
    build_model,
    fit_arm,
    flat_prior,
    independent_prior,
    random_walk_prior,
)
from e00737_forward import PRIOR_FLEET, FleetParams, fleet_unit_output
from e00737_inputs import load_primary
from e00737_worlds import WorldSpec, build_world, draw_fleet
from plotting.ocf_theme import BRAND_ORANGE, DATA_BLUE, DATA_GREEN, DATA_PURPLE, DATA_SKY

WIDTH: Final[int] = 600
HEIGHT: Final[int] = 200
FIG_DIR: Final[str] = "figures"
FIGURE_PRIMARY: Final[str] = PRIMARIES[0]
"""The Primary whose sky and weather the synthetic figures use (the sky is public CAMS data)."""
ORIENTATIONS: Final[tuple[tuple[str, tuple[float, float, float]], ...]] = (
    ("east", (1.0, 0.0, 0.0)),
    ("south", (0.0, 1.0, 0.0)),
    ("west", (0.0, 0.0, 1.0)),
)
POSTERIOR_SEED: Final[int] = 737000
POSTERIOR_SIZE_MW: Final[float] = 2.5
JUNE_QUARTER_START: Final[int] = 14 * 91 * 48
"""A half-hour index in the second quarter of the fifth year, where the sample days are taken."""


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    path = OUT / FIG_DIR
    path.mkdir(parents=True, exist_ok=True)
    chart.save(str(path / f"{name}.svg"))


def pick_days(*, ghi: np.ndarray) -> tuple[int, int]:
    """Pick a clear and a cloudy day from the second quarter of a year.

    Args:
        ghi: The half-hourly irradiance of the grid.

    Returns:
        The first half-hour index of the brightest day, and of a day near the dullest tenth.
    """
    start = JUNE_QUARTER_START
    days = ghi[start : start + 91 * 48].reshape(91, 48).sum(axis=1)
    order = np.argsort(days)
    return start + int(order[-1]) * 48, start + int(order[len(order) // 10]) * 48


def _hours(*, n: int) -> np.ndarray:
    return (np.arange(n) + 1) / 2.0


def forward_figures() -> None:
    """Draw the three forward-model figures: orientations, clipping, and the temperature derate."""
    inputs = load_primary(site=FIGURE_PRIMARY)
    clear, cloudy = pick_days(ghi=inputs.sky.ghi_w_m2)
    sky = inputs.sky
    air, wind = inputs.weather.air_c, inputs.weather.wind_ms

    def day(
        *, start: int, fleet: FleetParams, air_c: np.ndarray | float | None = None
    ) -> np.ndarray:
        window = slice(start, start + 48)
        sub = type(sky)(
            half_hour_end_time=sky.half_hour_end_time[window],
            ghi_w_m2=sky.ghi_w_m2[window],
            dni_w_m2=sky.dni_w_m2[window],
            dhi_w_m2=sky.dhi_w_m2[window],
            zenith_deg=sky.zenith_deg[window],
            sun_azimuth_deg=sky.sun_azimuth_deg[window],
        )
        return fleet_unit_output(
            sky=sub,
            air_c=air[window] if air_c is None else np.full(48, air_c),
            wind_ms=wind[window] if air_c is None else np.full(48, 2.0),
            fleet=fleet,
        )

    rows = []
    for label, start in (("clear sky", clear), ("cloudy sky", cloudy)):
        for name, mix in ORIENTATIONS:
            fleet = FleetParams(mix=mix, tilt_deg=PRIOR_FLEET.tilt_deg, dc_ac_ratio=1.15)
            rows += [
                {"hour": float(h), "mw": float(v), "orientation": name, "sky": label}
                for h, v in zip(_hours(n=48), day(start=start, fleet=fleet), strict=True)
            ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of the day (UTC), daylight only",
                scale=alt.Scale(domain=[4, 20]),
            ),
            y=alt.Y("mw:Q", title="Output of 1 MW AC capacity (MW)"),
            color=alt.Color(
                "orientation:N", scale=alt.Scale(range=[DATA_BLUE, BRAND_ORANGE, DATA_GREEN])
            ),
            row=alt.Row("sky:N", title=None),
        )
        .properties(
            width=WIDTH,
            height=HEIGHT // 1,
            title="One megawatt of AC capacity, by panel orientation",
        ),
        name="forward_orientations",
    )
    rows = []
    for ratio in (1.0, 1.15, 1.5):
        fleet = FleetParams(mix=(0.0, 1.0, 0.0), tilt_deg=PRIOR_FLEET.tilt_deg, dc_ac_ratio=ratio)
        rows += [
            {"hour": float(h), "mw": float(v), "ratio": f"DC:AC {ratio}"}
            for h, v in zip(_hours(n=48), day(start=clear, fleet=fleet), strict=True)
        ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of the day (UTC), daylight only",
                scale=alt.Scale(domain=[4, 20]),
            ),
            y=alt.Y("mw:Q", title="Output of 1 MW AC capacity (MW)"),
            color=alt.Color(
                "ratio:N", scale=alt.Scale(range=[DATA_BLUE, BRAND_ORANGE, DATA_PURPLE])
            ),
        )
        .properties(
            width=WIDTH, height=HEIGHT, title="Clipping at the AC capacity, south-facing panels"
        ),
        name="forward_clipping",
    )
    rows = []
    for temperature in (5.0, 20.0, 35.0):
        rows += [
            {"hour": float(h), "mw": float(v), "air": f"{temperature:g} C air"}
            for h, v in zip(
                _hours(n=48),
                day(start=clear, fleet=PRIOR_FLEET, air_c=temperature),
                strict=True,
            )
        ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                title="Hour of the day (UTC), daylight only",
                scale=alt.Scale(domain=[4, 20]),
            ),
            y=alt.Y("mw:Q", title="Output of 1 MW AC capacity (MW)"),
            color=alt.Color("air:N", scale=alt.Scale(range=[DATA_SKY, BRAND_ORANGE, DATA_PURPLE])),
        )
        .properties(
            width=WIDTH, height=HEIGHT, title="Cell-temperature derate at three air temperatures"
        ),
        name="forward_derate",
    )


def fleet_figures(*, growth_sd: float) -> None:
    """Draw the three injected-fleet figures: growth paths, one week of output, and prior draws.

    Args:
        growth_sd: The growth standard deviation of the prior that figure 3 draws from.
    """
    from e00737_worlds import capacity_path

    rows = [
        {"quarter": q + 1, "mw": float(v), "path": f"{shape}, {size:g} MW"}
        for size in (1.0, 2.5, 5.0)
        for shape in ("ramp", "step", "accelerating")
        for q, v in enumerate(capacity_path(shape=shape, size_mw=size))
    ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("quarter:Q", title="Quarter (1 to 18)"),
            y=alt.Y("mw:Q", title="Injected AC capacity (MW)"),
            color=alt.Color("path:N", legend=alt.Legend(columns=3, orient="top")),
        )
        .properties(width=WIDTH, height=HEIGHT, title="The nine injected growth paths"),
        name="fleets_paths",
    )
    inputs = load_primary(site=FIGURE_PRIMARY)
    sky = inputs.sky
    start = JUNE_QUARTER_START
    weekly = sky.ghi_w_m2[start : start + 91 * 48].reshape(13, 7 * 48).sum(axis=1)
    bright_week = start + int(np.argmax(weekly)) * 7 * 48
    dull_week = start + int(np.argmin(weekly)) * 7 * 48
    fleets = {
        "prior's fleet": PRIOR_FLEET,
        "mismatched fleet 1": draw_fleet(seed=737101, primary_index=0, matched=False),
        "mismatched fleet 2": draw_fleet(seed=737102, primary_index=0, matched=False),
    }
    rows = []
    for label, week in (("brighter week", bright_week), ("duller week", dull_week)):
        window = slice(week, week + 7 * 48)
        sub = type(sky)(
            half_hour_end_time=sky.half_hour_end_time[window],
            ghi_w_m2=sky.ghi_w_m2[window],
            dni_w_m2=sky.dni_w_m2[window],
            dhi_w_m2=sky.dhi_w_m2[window],
            zenith_deg=sky.zenith_deg[window],
            sun_azimuth_deg=sky.sun_azimuth_deg[window],
        )
        for name, fleet in fleets.items():
            out = 5.0 * fleet_unit_output(
                sky=sub,
                air_c=inputs.weather.air_c[window],
                wind_ms=inputs.weather.wind_ms[window],
                fleet=fleet,
            )
            rows += [
                {"day": float(h) / 24.0 + 1.0, "mw": float(v), "fleet": name, "week": label}
                for h, v in zip(_hours(n=7 * 48), out, strict=True)
            ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line(clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Day of the week (1 to 7)", scale=alt.Scale(domain=[1, 8])),
            y=alt.Y("mw:Q", title="Output of a 5 MW fleet (MW)"),
            color=alt.Color(
                "fleet:N", scale=alt.Scale(range=[DATA_BLUE, BRAND_ORANGE, DATA_GREEN])
            ),
            row=alt.Row("week:N", title=None),
        )
        .properties(width=WIDTH, height=HEIGHT // 1, title="Output of a 5 MW fleet over one week"),
        name="fleets_week",
    )
    prior = random_walk_prior(growth_sd=growth_sd)
    cov = np.linalg.inv(prior.precision)
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(prior.mean, cov, size=20)
    rows = [
        {"quarter": q + 1, "mw": float(v), "draw": d}
        for d in range(20)
        for q, v in enumerate(np.asarray(draws)[d])
    ]
    _save(
        chart=alt.Chart(alt.Data(values=rows))
        .mark_line(opacity=0.5, color=DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("quarter:Q", title="Quarter (1 to 18)"),
            y=alt.Y("mw:Q", title="Capacity (MW)"),
            detail="draw:N",
        )
        .properties(width=WIDTH, height=HEIGHT, title="Twenty capacity paths drawn from the prior"),
        name="fleets_prior_draws",
    )


def posterior_world(
    *, model: PrimaryModel, spec: WorldSpec, growth_sd: float, ridge_scale: float
) -> pl.DataFrame:
    """Fit arms A, F, and Qt to a tuning world and to the real record, and return the increments.

    Args:
        model: The Primary's model inputs.
        spec: The tuning world's recipe.
        growth_sd: The growth standard deviation of arm A.
        ridge_scale: The multiplier of the baseline's ridge precision.

    Returns:
        One row per arm and quarter: the injected capacity, the fitted increment, and 80% bounds.
    """
    world = build_world(inputs=model.inputs, spec=spec)
    priors = {
        "A": random_walk_prior(growth_sd=growth_sd),
        "F": flat_prior(),
        "Qt": independent_prior(),
    }
    rows = []
    for arm, prior in priors.items():
        injected = fit_arm(model=model, power=world.power, prior=prior, ridge_scale=ridge_scale)
        control = fit_arm(
            model=model, power=model.inputs.power, prior=prior, ridge_scale=ridge_scale
        )
        increment = injected.capacity - control.capacity
        for q in range(N_QUARTERS):
            half = 1.2815515655446004 * injected.sd[q]
            rows.append(
                {
                    "quarter": q + 1,
                    "arm": arm,
                    "truth": float(world.truth[q]),
                    "fit": float(increment[q]),
                    "lo": float(increment[q] - half),
                    "hi": float(increment[q] + half),
                }
            )
    return pl.DataFrame(rows)


def _posterior_chart(*, frame: pl.DataFrame, title: str) -> alt.TopLevelMixin:
    base = alt.Chart(alt.Data(values=frame.to_dicts()))
    colours = alt.Scale(domain=["A", "F", "Qt"], range=[BRAND_ORANGE, DATA_BLUE, DATA_GREEN])
    band = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("quarter:Q", title="Quarter (1 to 18)"),
        y=alt.Y("lo:Q", title="Fitted capacity increment (MW)"),
        y2="hi:Q",
        color=alt.Color("arm:N", scale=colours),
    )
    line = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        x="quarter:Q", y="fit:Q", color=alt.Color("arm:N", scale=colours)
    )
    truth = base.mark_line(color="black", strokeDash=[4, 3]).encode(  # ty: ignore[unresolved-attribute]
        x="quarter:Q", y="truth:Q"
    )
    return (band + line + truth).properties(width=WIDTH, height=HEIGHT, title=title)


def posterior_figures(*, growth_sd: float, ridge_scale: float) -> None:
    """Draw the four posterior figures on tuning-seed worlds of one Primary.

    Args:
        growth_sd: The tuned growth standard deviation.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
    """
    start = time.time()
    model = build_model(inputs=load_primary(site=FIGURE_PRIMARY))
    for k, shape in enumerate(("ramp", "step", "accelerating")):
        spec = WorldSpec(FIGURE_PRIMARY, "tuning", POSTERIOR_SIZE_MW, shape, POSTERIOR_SEED + k)
        frame = posterior_world(
            model=model, spec=spec, growth_sd=growth_sd, ridge_scale=ridge_scale
        )
        _save(
            chart=_posterior_chart(
                frame=frame, title=f"Posterior of arms A, F, and Qt: {shape} fleet of 2.5 MW"
            ),
            name=f"posterior_{shape}",
        )
    frames = []
    for kind in ("cloud", "transfer"):
        spec = WorldSpec(FIGURE_PRIMARY, kind, POSTERIOR_SIZE_MW, "ramp", POSTERIOR_SEED)
        frames.append(
            posterior_world(
                model=model, spec=spec, growth_sd=growth_sd, ridge_scale=ridge_scale
            ).with_columns(world=pl.lit(kind))
        )
    both = pl.concat(frames)
    panels: list[Any] = [
        _posterior_chart(
            frame=both.filter(pl.col("world") == kind),
            title=f"Posterior on the {kind} world (2.5 MW ramp)",
        )
        for kind in ("cloud", "transfer")
    ]
    chart = alt.vconcat(*panels)
    _save(chart=chart, name="posterior_cloud_transfer")
    time_log(stage="figures", task="posterior", start=start)


def real_figure() -> None:
    """Draw the median capacity path over the Primaries as a multiple of each Primary's mean."""
    with np.load(OUT / "scores" / "real_path.npz") as f:
        paths = f["paths"]
    rows = [
        {
            "quarter": q + 1,
            "median": float(np.nanmedian(paths[:, q])),
            "lo": float(np.nanmin(paths[:, q])),
            "hi": float(np.nanmax(paths[:, q])),
        }
        for q in range(N_QUARTERS)
        if np.isfinite(paths[:, q]).any()
    ]
    base = alt.Chart(alt.Data(values=rows))
    band = base.mark_area(opacity=0.2, color=BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("quarter:Q", title="Quarter (1 to 18)"),
        y=alt.Y("lo:Q", title="Capacity as a multiple of the Primary's own mean"),
        y2="hi:Q",
    )
    line = base.mark_line(color=BRAND_ORANGE).encode(x="quarter:Q", y="median:Q")  # ty: ignore[unresolved-attribute]
    _save(
        chart=(band + line).properties(
            width=WIDTH,
            height=HEIGHT,
            title=f"Fitted capacity of {len(paths)} Primaries: median (line) and range (band)",
        ),
        name="real_capacity_path",
    )


def make_figures(*, part: str) -> None:
    """Draw one part of the gallery.

    Args:
        part: `forward`, `fleets`, `posterior`, or `real`.
    """
    import json

    tuned = OUT / "tuned.json"
    record = json.loads(tuned.read_text()) if tuned.exists() else {}
    growth_sd = record.get("growth_sd", GROWTH_SD_GRID[1])
    ridge_scale = record.get("ridge_scale", 1.0)
    if part == "forward":
        forward_figures()
    elif part == "fleets":
        fleet_figures(growth_sd=growth_sd)
    elif part == "posterior":
        posterior_figures(growth_sd=growth_sd, ridge_scale=ridge_scale)
    else:
        real_figure()
