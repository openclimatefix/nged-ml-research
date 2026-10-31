# ruff: noqa: E501
"""Score the coherence of 51 ensemble member paths: energy and variogram scores, by route.

A route turns an arm's marginal forecast (51 quantiles per farm and half-hour) into 51 member paths:

- `ecc`: the member with the j-th lowest shortwave at a farm and half-hour takes the j-th quantile
  (ensemble copula coupling, ECC-Q);
- `schaake`: 51 observed past paths of the same layout, from days within 25 days of the same date
  one year earlier, set the rank of each member in every dimension (a Schaake shuffle; a dimension
  with no history is ranked at random);
- `independent`: the quantiles are permuted at random in each dimension (the floor);
- `memberwise`: the arm's own member paths (each member passed alone, at one fixed level).

A path vector is every scored half-hour of one run at the six farms ("joint", days 1 to 10), or the
half-hours of one run in one day band. The energy score is the mean distance of the members to the
observed vector minus half the mean distance between members; the variogram score of order 0.5
compares the observed and ensemble mean of `|x_i - x_j|^0.5` over same-farm pairs up to 6 hours
apart and same-time pairs across farms. Lower is better. Scores are in points of capacity (the
energy score) and in capacity units (the variogram score, per pair).

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00430-*/e00430_coherence.py ARM [ARM ...]
"""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_features as feats  # ty: ignore[unresolved-import]
import e00430_common as common
import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

DRAWS: Final[int] = 10_000
LO_PCT, HI_PCT = 0.5, 99.5
PAIR_WINDOW_HOURS: Final[float] = 6.0
SHIFT_DAYS: Final[tuple[int, ...]] = tuple(365 + d for d in range(-25, 26))
"""Schaake history: the same date one year earlier, plus or minus 25 days (51 observed paths)."""
OUT_DIR: Final[Path] = common.DATA_DIR / "scores"
ROUTES: Final[tuple[str, ...]] = ("independent", "ecc", "schaake", "memberwise")


def energy_score(*, members: np.ndarray, observed: np.ndarray) -> float:
    """Return the energy score of the members (shape (m, d)) against the observed vector (d,)."""
    to_observed = np.linalg.norm(members - observed[None, :], axis=1).mean()
    between = cdist(members, members).mean()
    return float(to_observed - 0.5 * between)


def variogram_score(
    *,
    members: np.ndarray,
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    order: float = 0.5,
) -> float:
    """Return the mean over the pairs of the squared gap between the observed and ensemble-mean variogram."""
    if len(first) == 0:
        return float("nan")
    seen = np.abs(observed[first] - observed[second]) ** order
    model = (np.abs(members[:, first] - members[:, second]) ** order).mean(axis=0)
    return float(np.mean((seen - model) ** 2))


def pairs(*, farm: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs (i < j): same farm up to 6 hours apart, or the same time at different farms."""
    same_farm = (farm[:, None] == farm[None, :]) & (
        np.abs(hours[:, None] - hours[None, :]) <= PAIR_WINDOW_HOURS
    )
    same_time = (hours[:, None] == hours[None, :]) & (farm[:, None] != farm[None, :])
    mask = np.triu(same_farm | same_time, k=1)
    first, second = np.nonzero(mask)
    return first, second


def load_rows(*, arm: str, seed: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return an arm's rows for one seed with their 51 values and 51 shortwave ranks."""
    tables, values, ranks = [], [], []
    for path in sorted((common.PRED_DIR / arm).glob(f"s{seed}_*.parquet")):
        tables.append(pl.read_parquet(path))
        z = np.load(path.with_name("paths_" + path.name.replace(".parquet", ".npz")))
        values.append(z["values"])
        ranks.append(z["rank"])
    table = pl.concat(tables)
    return table, np.concatenate(values), np.concatenate(ranks)


def observed_lookup() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return, per farm, the sorted observation times (microseconds) and the output."""
    out = {}
    for farm in c310.FARMS:
        series = feats.output_series(farm).sort("time")
        out[farm] = (
            series["time"].dt.epoch("us").to_numpy(),
            series["output"].to_numpy().astype(np.float64),
        )
    return out


def schaake_ranks(*, table: pl.DataFrame, seed: int) -> np.ndarray:
    """Return, per row, the member order (0 to 50) that the Schaake shuffle assigns, shape (rows, 51)."""
    rng = np.random.default_rng(900 + seed)
    lookup = observed_lookup()
    farms = table["time_series_id"].to_numpy()
    times = table["time"].dt.epoch("us").to_numpy()
    inits = table["init_time"].dt.epoch("us").to_numpy()
    out = np.zeros((len(table), common.N_MEMBERS), dtype=np.uint8)
    for init in np.unique(inits):
        rows = np.flatnonzero(inits == init)
        history = np.full((common.N_MEMBERS, len(rows)), np.nan)
        for k, shift in enumerate(SHIFT_DAYS):
            for farm in c310.FARMS:
                in_farm = farms[rows] == farm
                if not in_farm.any():
                    continue
                target = times[rows][in_farm] - int(timedelta(days=shift).total_seconds() * 1e6)
                t, y = lookup[farm]
                position = np.clip(np.searchsorted(t, target), 0, len(t) - 1)
                found = t[position] == target
                history[k, np.flatnonzero(in_farm)[found]] = y[position[found]]
        missing = np.isnan(history)
        history = np.where(missing, rng.random(history.shape), history) + 1e-9 * rng.random(
            history.shape
        )
        out[rows] = np.argsort(np.argsort(history, axis=0), axis=0).T.astype(np.uint8)
    return out


def route_members(
    *, route: str, values: np.ndarray, ranks: np.ndarray, schaake: np.ndarray | None, seed: int
) -> np.ndarray:
    """Return the (rows, 51) member values of a route."""
    if route == "memberwise":
        return values
    if route == "ecc":
        return np.take_along_axis(values, ranks.astype(np.int64), axis=1)
    if route == "schaake":
        assert schaake is not None
        return np.take_along_axis(values, schaake.astype(np.int64), axis=1)
    rng = np.random.default_rng(700 + seed)
    return rng.permuted(values, axis=1)


def group_scores(*, table: pl.DataFrame, members: np.ndarray) -> pl.DataFrame:
    """Return the energy and variogram scores of every group: each run jointly, and each run and day band."""
    farm_index = np.array([c310.FARMS.index(f) for f in table["time_series_id"].to_list()])
    hours = table["time"].dt.epoch("us").to_numpy() / 3.6e9
    observed = table["actual"].to_numpy()
    inits = table["init_time"].dt.epoch("us").to_numpy()
    bands = table["band"].to_numpy()
    out = []
    for init in np.unique(inits):
        rows = np.flatnonzero(inits == init)
        month = str(np.datetime64(int(init), "us").astype("datetime64[M]"))
        for name, selected in (
            ("joint", rows),
            *[(b, rows[bands[rows] == b]) for b in c310.BAND_ORDER],
        ):
            if len(selected) < 2:
                continue
            first, second = pairs(farm=farm_index[selected], hours=hours[selected])
            x = members[selected].T.astype(np.float64)
            y = observed[selected].astype(np.float64)
            out.append(
                {
                    "month": month,
                    "set": name,
                    "energy": 100.0 * energy_score(members=x, observed=y),
                    "variogram": variogram_score(members=x, observed=y, first=first, second=second),
                }
            )
    return pl.DataFrame(out)


def bootstrap_change(
    *, by_month_a: np.ndarray, by_month_b: np.ndarray
) -> tuple[float, float, float]:
    """Return the percentage change of a against b and its 99% interval over month blocks."""
    rng = np.random.default_rng(0)
    n = len(by_month_a)
    point = 100.0 * (by_month_a.mean() / by_month_b.mean() - 1.0)
    index = rng.integers(0, n, size=(DRAWS, n))
    draws = 100.0 * (by_month_a[index].mean(axis=1) / by_month_b[index].mean(axis=1) - 1.0)
    return float(point), float(np.percentile(draws, LO_PCT)), float(np.percentile(draws, HI_PCT))


def score_arm(*, arm: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return every route's group scores for the arm, averaged over the seeds, with a route column."""
    frames = []
    routes = ("memberwise",) if arm == "a2_memberwise" else ROUTES[:-1]
    for route in routes:
        per_seed = []
        for seed in seeds:
            table, values, ranks = load_rows(arm=arm, seed=seed)
            schaake = schaake_ranks(table=table, seed=seed) if route == "schaake" else None
            members = route_members(
                route=route, values=values, ranks=ranks, schaake=schaake, seed=seed
            )
            per_seed.append(
                group_scores(table=table, members=members).with_columns(seed=pl.lit(seed))
            )
        frames.append(
            pl.concat(per_seed)
            .group_by("month", "set")
            .agg(pl.col("energy").mean(), pl.col("variogram").mean())
            .with_columns(arm=pl.lit(arm), route=pl.lit(route))
        )
    return pl.concat(frames)


def summarise(*, scores: pl.DataFrame, reference: tuple[str, str]) -> pl.DataFrame:
    """Return, per arm, route, and set, the mean scores and the change against the reference arm and route."""
    ref_arm, ref_route = reference
    rows = []
    for (arm, route, band), part in scores.group_by("arm", "route", "set", maintain_order=True):
        ref = scores.filter(
            (pl.col("arm") == ref_arm) & (pl.col("route") == ref_route) & (pl.col("set") == band)
        )
        months = sorted(set(part["month"].to_list()) & set(ref["month"].to_list()))
        record: dict[str, float | str] = {"arm": arm, "route": route, "set": band}
        for metric in ("energy", "variogram"):
            a = np.array([part.filter(pl.col("month") == m)[metric][0] for m in months])
            b = np.array([ref.filter(pl.col("month") == m)[metric][0] for m in months])
            point, lo, hi = bootstrap_change(by_month_a=a, by_month_b=b)
            record |= {
                f"{metric}": float(a.mean()),
                f"{metric}_pct_vs_ref": point,
                f"{metric}_lo99": lo,
                f"{metric}_hi99": hi,
            }
        rows.append(record)
    return pl.DataFrame(rows)


def main() -> None:
    """Score the named arms (seeds 0 to 2), referenced to `g0_onepass` with ECC."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(14)
    scores = pl.concat([score_arm(arm=a, seeds=common.SEEDS_ABLATION) for a in sys.argv[1:]])
    table = summarise(scores=scores, reference=("g0_onepass", "ecc"))
    print(table)
    table.write_csv(OUT_DIR / ("coherence_" + "_".join(sys.argv[1:]) + ".csv"))


if __name__ == "__main__":
    main()
