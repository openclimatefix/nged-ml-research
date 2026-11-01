"""Run the placebo-gated injection test of experiment 474, one Primary at a time.

For each Primary, injected farm and solar-shape rung, the script builds the monthly Gram matrices of
experiment 129's convex fit once, then solves one bounded least-squares problem per target series:
the base reading, the farm injected at 2 and 4 MW, the farm with a cloud-driven load, the
day-of-year profile alone, and 20 within-month shuffled placebos. The real targets carry a
month-block bootstrap. The plan is in
`docs/ml-experiments/00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md`.

    systemd-run --user --scope -p MemoryMax=10G -p OOMPolicy=continue \
        env PYTHONPATH=<folders of experiments 269 and 129> \
        uv run --no-sync python ml-experiments/00474-*/e00474_fit.py --sites D10
"""

import argparse
from typing import Final

import e00269_common as c269
import e00269_fit as f269
import e00474_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_gam import (
    GamSpec,
    derated_unit_shape,
    design_matrix,
    penalty_matrix,
    prepare_fit,
    row_chunks,
)

N_BOOTSTRAP: Final[int] = 200
READ_OUT_INDEX: Final[int] = c269.READ_OUT_COLUMNS - 1
"""Position of the read-out season-year in the cumulative capacity path."""


def solve_targets(
    *,
    cache,
    site: str,
    shape: np.ndarray,
    y_rows: np.ndarray,
    targets: np.ndarray,
    weights_by_target: list[np.ndarray],
    cloud_terms: bool = False,
):
    """Return per-target cumulative capacity paths, one array (B_t, n_seasons) per target.

    Args:
        cache: The cache.
        site: The Primary's label.
        shape: The unit shape.
        y_rows: The Primary's reading with `NaN` where a row is not trained on.
        targets: The target series, shape (T, grid), `NaN` where a row is not trained on.
        weights_by_target: Month weights for each target, shape (B_t, M).

    Returns:
        `(paths, index, layout)`.
    """
    shaped = common.with_shape(cache=cache, site=site, shape=shape, y=y_rows)
    spec = GamSpec(ridge=c269.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=shaped,
        site=site,
        spec=spec,
        cut=c269.CUT,
        window=c269.TRAINING_WINDOW,
        season_years=c269.SEASON_YEARS,
    )
    features = shaped.sites[site]
    months_of_rows = f269.row_months(cache=shaped, rows=index)
    months = np.arange(months_of_rows.min(), months_of_rows.max() + 1)
    slot = months_of_rows - months[0]
    p = layout.n_columns + (2 if cloud_terms else 0)
    z = common.cloud_regressors(cache=cache, site=site) if cloud_terms else None
    n_targets = targets.shape[0]
    gram = torch.zeros((len(months), p, p), dtype=torch.float64)
    rhs = torch.zeros((len(months), n_targets, p), dtype=torch.float64)
    for chunk in row_chunks(index=index):
        rows = index[chunk]
        x = torch.as_tensor(
            design_matrix(cache=shaped, features=features, layout=layout, index=rows),
            dtype=torch.float64,
        )
        if z is not None:
            x = torch.cat([x, torch.as_tensor(z[rows], dtype=torch.float64)], dim=1)
        yt = torch.as_tensor(targets[:, rows], dtype=torch.float64)
        for month in np.unique(slot[chunk]):
            members = torch.as_tensor(np.flatnonzero(slot[chunk] == month))
            xm = x[members]
            gram[month] += xm.T @ xm
            rhs[month] += yt[:, members] @ xm
    penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
    extra = p - layout.n_columns
    penalty_t = torch.zeros((p, p), dtype=torch.float64)
    penalty_t[: layout.n_columns, : layout.n_columns] = torch.as_tensor(penalty)
    shift = torch.zeros(p, dtype=torch.float64)
    shift[: layout.n_columns] = 0.5 * torch.as_tensor(cost, dtype=torch.float64)
    if extra:
        penalty_t[layout.n_columns :, layout.n_columns :] += c269.RIDGE * torch.eye(extra)
    constrained = np.concatenate([layout.constrained, np.zeros(extra, dtype=bool)])
    capacity = layout.slices["capacity"]
    paths = []
    for t in range(n_targets):
        weights = weights_by_target[t]
        w = torch.as_tensor(weights, dtype=torch.float64)
        g = torch.einsum("bm,mpq->bpq", w, gram) + penalty_t[None]
        r = torch.einsum("bm,mp->bp", w, rhs[:, t]) - shift[None]
        theta = f269.solve_in_batches(matrix=g, rhs=r, constrained=constrained)
        paths.append(np.cumsum(theta[:, capacity], axis=1))
    return paths, index, layout, months


def recovery_factor(*, shaped_cache, site: str, index: np.ndarray, series: np.ndarray) -> float:
    """Return experiment 269's ratio-of-99th-percentiles unit factor for one injected series."""
    features = shaped_cache.sites[site]
    derated = derated_unit_shape(features=features, index=index)
    in_year = shaped_cache.calendar.season_year[index] == c269.READ_OUT_SEASON_YEAR
    daylight = in_year & (derated > c269.DAYLIGHT_UNIT_SHAPE) & (series[index] > 0.0)
    return float(np.percentile(derated[daylight], 99) / np.percentile(series[index][daylight], 99))


def run_cell(
    *,
    cache,
    site: str,
    farm: str,
    rung: str,
    n_bootstrap: int,
    n_shuffles: int,
    cloud_terms: bool = False,
) -> list[dict]:
    """Return result rows for one Primary, injected farm and rung."""
    features = cache.sites[site]
    norm = c269.farm_norm(cache=cache, farm=farm)
    shape = common.rung_shape(cache=cache, site=site, rung=rung, farm=farm)
    shape = np.where(np.isfinite(norm), shape, np.nan)
    y_rows = features.y - 0.0 * norm
    profile = np.where(
        np.isfinite(norm), common.day_of_year_profile(cache=cache, series=norm), np.nan
    )
    series: list[tuple[str, float, int, np.ndarray, np.ndarray]] = []
    # (kind, mw, shuffle, injected series in unit of the farm's norm, extra additive load in points)
    zeros = np.zeros(len(norm))
    series.append(("base", 0.0, 0, zeros, zeros))
    for mw in common.INJECTION_MW:
        series.append(("real", mw, 0, norm, zeros))
    mw0 = common.INJECTION_MW[0]
    pts0 = common.points_from_mw(cache=cache, site=site, mw=mw0)
    series.append(("cloud", mw0, 0, norm, common.cloud_load(cache=cache, site=site, points=pts0)))
    series.append(("profile", mw0, 0, profile, zeros))
    for k in range(n_shuffles):
        placebo = common.shuffled_placebo(
            cache=cache, norm=norm, profile=profile, seed=1000 * k + 7
        )
        series.append(("placebo", mw0, k, placebo, zeros))
    targets = np.stack(
        [
            y_rows
            - common.points_from_mw(cache=cache, site=site, mw=mw) * np.nan_to_num(s, nan=0.0)
            + load
            for _, mw, _, s, load in series
        ]
    )
    targets = np.where(np.isfinite(norm)[None], targets, np.nan)
    # Month weights: bootstrap for base and real kinds, the point fit for the rest.
    probe = common.with_shape(cache=cache, site=site, shape=shape, y=y_rows)
    layout, index = prepare_fit(
        cache=probe,
        site=site,
        spec=GamSpec(ridge=c269.RIDGE, capacity="fitted"),
        cut=c269.CUT,
        window=c269.TRAINING_WINDOW,
        season_years=c269.SEASON_YEARS,
    )
    months_of_rows = f269.row_months(cache=probe, rows=index)
    months = np.arange(months_of_rows.min(), months_of_rows.max() + 1)
    counts = f269.month_block_counts(
        months=months,
        n_resamples=n_bootstrap,
        seed=2000
        + 37 * c269.ALL_SITES.index(site)
        + 7 * common.FARMS.index(farm)
        + common.RUNGS.index(rung),
    )
    point = np.ones((1, len(months)))
    weights = [counts if kind in ("base", "real", "cloud") else point for kind, *_ in series]
    paths, index, layout, months = solve_targets(
        cache=cache,
        site=site,
        shape=shape,
        y_rows=y_rows,
        targets=targets,
        weights_by_target=weights,
        cloud_terms=cloud_terms,
    )
    shaped = common.with_shape(cache=cache, site=site, shape=shape, y=y_rows)
    base_path = paths[0]
    rows: list[dict] = [
        {
            "site": site,
            "farm": farm,
            "rung": rung + ("+cloudterm" if cloud_terms else ""),
            "kind": "meta",
            "mw": 0.0,
            "shuffle": 0,
            "b": 0,
            "stat": "n_rows",
            "value": float(len(index)),
        }
    ]
    for t, (kind, mw, k, s, _) in enumerate(series):
        if kind == "base":
            continue
        pts = common.points_from_mw(cache=cache, site=site, mw=mw)
        factor = recovery_factor(shaped_cache=shaped, site=site, index=index, series=norm)
        pt = paths[t]
        base = base_path if len(pt) == len(base_path) else base_path[:1]
        delta = pt[:, READ_OUT_INDEX] - base[:, READ_OUT_INDEX]
        for b in range(len(delta)):
            for stat, value in (
                ("recovery", delta[b] / pts * factor),
                ("ratio", delta[b] / pts),
            ):
                rows.append(
                    {
                        "site": site,
                        "farm": farm,
                        "rung": rung + ("+cloudterm" if cloud_terms else ""),
                        "kind": kind,
                        "mw": mw,
                        "shuffle": k,
                        "b": b,
                        "stat": stat,
                        "value": float(value),
                    }
                )
    return rows


def main() -> None:
    """Write `work/ladder_<site>.parquet`."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.SIGNED_SITES))
    parser.add_argument("--farms", nargs="+", default=list(common.FARMS))
    parser.add_argument("--rungs", nargs="+", default=list(common.RUNGS))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--shuffles", type=int, default=common.N_PLACEBO_SHUFFLES)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--cloud-terms", action="store_true")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = c269.load_cache()
    for site in arguments.sites:
        results: list[dict] = []
        for farm in arguments.farms:
            for rung in arguments.rungs:
                try:
                    cell = run_cell(
                        cache=cache,
                        site=site,
                        farm=farm,
                        rung=rung,
                        n_bootstrap=arguments.bootstrap,
                        n_shuffles=arguments.shuffles,
                        cloud_terms=arguments.cloud_terms,
                    )
                except (ValueError, IndexError) as error:
                    print(f"{site} farm {farm} {rung}: skipped ({error})", flush=True)
                    continue
                results += cell
                print(f"{site} farm {farm} {rung}: {len(results)} rows", flush=True)
        pl.DataFrame(results).write_parquet(
            common.WORK_DIR / f"ladder_{site}{arguments.suffix}.parquet"
        )


if __name__ == "__main__":
    main()
