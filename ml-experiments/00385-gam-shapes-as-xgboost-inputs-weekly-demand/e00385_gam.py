"""GAM columns for the weekly XGBoost model: stage-1 fitted value and partial effects (385).

Uses experiment 377's T-all fit (its design, penalty, and pinned capacity) at a cut, refitted here
from the cut's per-month Gram matrices so that each calendar month of the 24-month training window
can be left out in turn. A value for a training row of month m comes from the fit that leaves month
m out (out of fold); a value for a scored row comes from the fit on the whole window.

Columns, all in points of effective capacity:

- `gam_fitted`: the fit's net power (design times coefficients, minus the pinned solar output).
- `gam_calendar`: the daily profile, holiday, and tent terms (time of day and day type).
- `gam_season`: the seasonal harmonics.
- `gam_temperature`: the heating terms (the 64 pieces of the 16 heating hinges).
- `gam_lighting`: the lighting terms.

Set `E00377_SOURCE` to `control`, `mva`, or `d15` before importing, as experiment 377 requires.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_design as design  # ty: ignore[unresolved-import]
import e00377_run as run377  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import design_matrix, pinned_solar_points

COLUMNS: Final[tuple[str, ...]] = (
    "gam_fitted",
    "gam_calendar",
    "gam_season",
    "gam_temperature",
    "gam_lighting",
)
SHAPE_COLUMNS: Final[tuple[str, ...]] = COLUMNS[1:]
N_BASE: Final[int] = 268
"""T-all keeps B0's first 268 columns (profile, holiday, tent, harmonics) in place."""
TERM_SLICES: Final[dict[str, slice]] = {
    "gam_calendar": slice(0, 252),
    "gam_season": slice(252, 268),
    "gam_temperature": slice(268, 332),
    "gam_lighting": slice(332, 340),
}
"""Columns of `design.arm_matrix` under the `t_all` arm for each partial effect."""


@dataclass(frozen=True)
class CutModels:
    """The fits at one cut: the whole-window fit and one fit per left-out month.

    Attributes:
        cut: The cut.
        whole: The whole-window coefficients, shape (p,).
        left_out: The coefficients with each month left out, shape (months, p).
        months: The window's calendar months, `datetime64[M]`, matching the rows of `left_out`.
        pinned: The pinned capacity of the cut.
        clip: The export clip of the site and pinned capacity.
    """

    cut: datetime
    whole: np.ndarray
    left_out: np.ndarray
    months: np.ndarray
    pinned: object
    clip: object
    layout: object


def fit_cut(*, cache: FeatureCache, vintages: pl.DataFrame, site: str, cut: datetime) -> CutModels:
    """Fit T-all at `cut` on the whole window and with each month left out."""
    base = design.build_base(
        cache=cache, vintages=vintages, site=site, cut=cut, fitted=common.FITTED
    )
    arm = design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA)
    x = design.arm_matrix(
        cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
    )
    penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
    grams = design.MonthGrams.build(x=x, month_of_row=base.month_of_row, n_months=len(base.months))
    month_rhs = grams.cross_products(x=x, target=base.target)
    total_gram = grams.gram.sum(dim=0)
    total_rhs = month_rhs.sum(dim=0)
    batch_gram = torch.cat([total_gram[None], total_gram[None] - grams.gram])
    batch_rhs = torch.cat([total_rhs[None], total_rhs[None] - month_rhs])
    theta = design.solve(gram=batch_gram, rhs=batch_rhs, penalty=penalty)
    return CutModels(
        cut=cut,
        whole=theta[0],
        left_out=theta[1:],
        months=base.months,
        pinned=base.pinned,
        clip=common.clip_of(site=site, pinned=base.pinned),
        layout=base.layout,
    )


def run_values(
    *,
    driven: FeatureCache,
    site: str,
    models: CutModels,
    arm: design.ArmSpec,
    n_targets: int,
    thetas: np.ndarray,
) -> np.ndarray:
    """Return the five columns of one driven window for each coefficient vector.

    Args:
        driven: The window cache driven by one ensemble member.
        site: The substation's label.
        models: The cut's fits (for the pinned capacity and layout).
        arm: The T-all arm.
        n_targets: The window length.
        thetas: Coefficient vectors, shape (k, p).

    Returns:
        An array of shape (k, n_targets, 5), float64, `NaN` where an input is not finite.
    """
    index = np.arange(n_targets)
    features = driven.sites[site]
    x0 = design_matrix(cache=driven, features=features, layout=models.layout, index=index)
    x = design.arm_matrix(
        cache=driven, layout=models.layout, site=site, index=index, x0=x0, arm=arm
    )
    solar = pinned_solar_points(
        cache=driven, features=features, pinned=models.pinned, index=index, clip=models.clip
    )
    finite = np.isfinite(x).all(axis=1) & np.isfinite(solar)
    out = np.full((len(thetas), n_targets, len(COLUMNS)), np.nan)
    x_ok = x[finite]
    out[:, finite, 0] = thetas @ x_ok.T - solar[finite][None]
    for position, name in enumerate(SHAPE_COLUMNS, start=1):
        term = TERM_SLICES[name]
        out[:, finite, position] = thetas[:, term] @ x_ok[:, term].T
    return out


def tall_arm() -> design.ArmSpec:
    """Return experiment 377's T-all arm."""
    return design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA)


def load_site(*, site: str) -> run377.SiteContext:
    """Return the substation's stored inputs (experiment 377's loader)."""
    return run377.load_site(site=site)


def _window_times(*, ctx: run377.SiteContext, window: slice) -> pl.Series:
    """Return the valid times of a run's window as UTC datetimes."""
    return pl.Series(ctx.cache.end_times[window]).dt.replace_time_zone("UTC")


def window_for(*, ctx: run377.SiteContext, run_index: int):  # noqa: ANN201
    """Return experiment 377's `RunWindow` of one run."""
    from studies.dp_ens_weather import RunWindow

    return RunWindow.build(
        cache=ctx.cache, context=ctx.setup.context, run=ctx.setup.runs[run_index]
    )


def training_columns(
    *,
    ctx: run377.SiteContext,
    models: CutModels,
    arm: design.ArmSpec,
    run_indices: list[int],
) -> pl.DataFrame:
    """Return the out-of-fold columns of the control member's rows for the chosen runs.

    A row whose valid month is in the cut's window takes the fit that leaves that month out. Any
    other row takes the whole-window fit.
    """
    thetas = np.vstack([models.whole[None], models.left_out])
    frames = []
    for index in run_indices:
        try:
            window = window_for(ctx=ctx, run_index=index)
        except ValueError:
            continue  # the first week of runs has no analysis proxy yet; its rows get no column
        n = window.window.stop - window.window.start
        driven = window.driven_caches(
            context=ctx.setup.context, run=ctx.setup.runs[index], members=[0]
        )[0]
        values = run_values(
            driven=driven, site=ctx.site, models=models, arm=arm, n_targets=n, thetas=thetas
        )
        month = ctx.cache.end_times[window.window].astype("datetime64[M]")
        position = np.searchsorted(models.months, month)
        position = np.where(
            (position < len(models.months))
            & (models.months[np.minimum(position, len(models.months) - 1)] == month),
            position + 1,
            0,
        )
        chosen = values[position, np.arange(n)]
        frames.append(
            pl.DataFrame(
                {
                    "power_fcst_init_time": pl.Series(
                        np.full(
                            n, np.datetime64(ctx.setup.origins[index].replace(tzinfo=None), "us")
                        )
                    ).dt.replace_time_zone("UTC"),
                    "valid_time": _window_times(ctx=ctx, window=window.window),
                    **{c: chosen[:, k].astype(np.float32) for k, c in enumerate(COLUMNS)},
                }
            )
        )
    return pl.concat(frames)


def scored_columns(
    *,
    ctx: run377.SiteContext,
    models: CutModels,
    arm: design.ArmSpec,
    run_index: int,
    members: list[int],
) -> pl.DataFrame:
    """Return the whole-window fit's columns for each ensemble member of one scored run."""
    window = window_for(ctx=ctx, run_index=run_index)
    n = window.window.stop - window.window.start
    caches = window.driven_caches(
        context=ctx.setup.context, run=ctx.setup.runs[run_index], members=members
    )
    times = _window_times(ctx=ctx, window=window.window)
    origin = pl.Series(
        np.full(n, np.datetime64(ctx.setup.origins[run_index].replace(tzinfo=None), "us"))
    ).dt.replace_time_zone("UTC")
    frames = []
    for member, driven in zip(members, caches, strict=True):
        values = run_values(
            driven=driven,
            site=ctx.site,
            models=models,
            arm=arm,
            n_targets=n,
            thetas=models.whole[None],
        )[0]
        frames.append(
            pl.DataFrame(
                {
                    "power_fcst_init_time": origin,
                    "valid_time": times,
                    "ensemble_member": pl.Series(np.full(n, member, dtype=np.int32)),
                    **{c: values[:, k].astype(np.float32) for k, c in enumerate(COLUMNS)},
                }
            )
        )
    return pl.concat(frames)
