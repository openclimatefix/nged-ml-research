# ruff: noqa: E501
"""Arm J4 of experiment 436: the member-anchored Schaake shuffle.

No fit. For each joint run, scenario k takes the observed path of the past date whose weather best
matches ensemble member k's forecast weather, and the 51 values at every node and half-hour are
ordered by those 51 paths (as in the Schaake shuffle `J3`). Writes `work/paths/J4/<date>.npy`.

- Weather state of member k: the daily means over days 1 to 10 of five series: the 2 m temperature
  and the shortwave radiation averaged over the 16 Primaries (the demand frames), and the 100 m wind
  speed of each of the three wind farms (the stored member weather of experiment 429). Each series
  is centred (over the 51 members, and over the candidate dates) and standardised by its
  standard deviation over the candidate dates.
- Weather state of a past date: the same five series from the stored analysis table of the nine
  generator sites (ERA5 weather with CAMS shortwave), with the temperature and shortwave averaged
  over the nine sites and the 100 m wind taken at the three wind farms. This is a regional proxy
  for the Primaries' own weather. The analysis table is observed weather, so a date at least 16 days
  before the run is long since available (ERA5 arrives about 5 days late).
- Candidates are the qualifying dates of the Schaake pool (`e00436_score.schaake_pool`). The 51
  members are matched to 51 distinct dates by the assignment that minimises the total distance,
  repeating the pool where it holds fewer than 51 dates.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00436-*/e00436_j4.py
```
"""

from datetime import UTC, datetime
from typing import Final

import e00303_common as c303  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00429_data as data429  # ty: ignore[unresolved-import]
import e00434_score as s434  # ty: ignore[unresolved-import]
import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

DAYS: Final[int] = 10
N_VARS: Final[int] = 2 + len(build.WIND_FARMS)
"""Temperature, shortwave, and 100 m wind at W1, W2, and W3."""
STEPS_PER_DAY: Final[int] = 48


def analysis_series() -> tuple[np.ndarray, datetime]:
    """Return the observed state series (5, n) on the half-hourly grid and the grid start."""
    sites = c303.site_table()
    wind_ids = {
        farm: int(sites.filter(pl.col("site") == farm)["time_series_id"][0])
        for farm in build.WIND_FARMS
    }
    frame = pl.read_parquet(
        c303.ANALYSIS_PATH,
        columns=["time_series_id", "valid_time", "temperature_2m", "shortwave", "wind_speed_100m"],
    )
    n = int((build.GRID_END - build.GRID_START) / build.STEP)
    out = np.full((N_VARS, n), np.nan)
    hourly = (
        frame.group_by("valid_time")
        .agg(pl.col("temperature_2m").mean(), pl.col("shortwave").mean())
        .sort("valid_time")
    )
    series = [
        hourly.select("valid_time", "temperature_2m"),
        hourly.select("valid_time", "shortwave"),
    ]
    for farm in build.WIND_FARMS:
        series.append(  # noqa: PERF401
            frame.filter(pl.col("time_series_id") == wind_ids[farm])
            .select("valid_time", "wind_speed_100m")
            .sort("valid_time")
        )
    for v, sub in enumerate(series):
        idx = ((sub["valid_time"] - build.GRID_START).dt.total_minutes() // 30).to_numpy()
        ok = (idx >= 0) & (idx < n)
        out[v, idx[ok]] = sub[sub.columns[1]].to_numpy()[ok]
        # hourly values stand for the half-hour on either side: fill by linear interpolation
        have = ~np.isnan(out[v])
        out[v] = np.interp(np.arange(n), np.flatnonzero(have), out[v, have])
    return out, build.GRID_START


def daily_means(*, series: np.ndarray) -> np.ndarray:
    """Return the daily means over the first `DAYS` days of series (5, 702): shape (5, DAYS)."""
    return series[:, : DAYS * STEPS_PER_DAY].reshape(series.shape[0], DAYS, STEPS_PER_DAY).mean(-1)


def member_state(
    *, init: datetime, wind: pl.DataFrame, wind_ws: np.ndarray, ids: list[int]
) -> np.ndarray:
    """Return each member's weather state (51, 5, DAYS)."""
    t_member, sw_member = s434.run_weather(init=init, ids=ids)  # (16, 702, 51)
    state = np.zeros((build.N_MEMBERS, N_VARS, DAYS))
    state[:, 0] = daily_means(series=t_member.mean(0).T)
    state[:, 1] = daily_means(series=sw_member.mean(0).T)
    for f, farm in enumerate(build.WIND_FARMS):
        rows = wind.filter((pl.col("power_fcst_init_time") == init) & (pl.col("site") == farm))
        step = np.rint(rows["lead_hours"].to_numpy() * 2).astype(int) - 1
        ws = np.full((build.N_STEPS, build.N_MEMBERS), np.nan)
        ws[step] = wind_ws[rows["row"].to_numpy()]
        for k in range(build.N_MEMBERS):
            col = ws[:, k]
            bad = np.isnan(col)
            if bad.any():
                col = np.interp(np.arange(col.size), np.flatnonzero(~bad), col[~bad])
            state[k, 2 + f] = daily_means(series=col[None])[0]
    return state


def main() -> None:
    """Write the J4 scenarios of every joint run."""
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]

    observed = np.load(score.WORK / "observed.npz")["observed"]
    analysis, _start = analysis_series()
    series = adapter.series_table()
    ids = series["time_series_id"].to_list()
    wind = data347.scored_frame().with_row_index("row")
    weather, _ = data429.load(set_name="scored")
    wind_ws = weather[:, :, build.WIND_100M].astype(np.float64)
    for file in sorted((score.WORK / "runs").glob("*.npz")):
        run = score.load_run(path=file)
        init = datetime.fromisoformat(f"{run['init']}+00:00").astimezone(UTC)
        out = score.scenario_path(arm="J4", init=init)
        if out.exists():
            continue
        pool = score.schaake_pool(init=init, observed=observed, kind=run["kind"])
        dates = [d for _, d, _ in pool]
        past = np.stack(
            [
                daily_means(
                    series=analysis[
                        :, build.grid_index(t=d) + 1 : build.grid_index(t=d) + 1 + build.N_STEPS
                    ]
                )
                for d in dates
            ]
        )  # (n, 5, DAYS)
        members = member_state(init=init, wind=wind, wind_ws=wind_ws, ids=ids)
        scale = past.reshape(len(dates), N_VARS, DAYS).std(axis=(0, 2))[None, :, None] + 1e-9
        # Remove each variable's mean (over the 51 members, and over the candidates) so that a
        # systematic bias between the forecast and the analysis does not decide the match.
        members = members - members.mean(axis=0, keepdims=True)
        past = past - past.mean(axis=0, keepdims=True)
        a = (members / scale).reshape(build.N_MEMBERS, -1)
        b = (past / scale).reshape(len(dates), -1)
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        # Repeat the candidate dates until there are at least 51 columns.
        reps = -(-build.N_MEMBERS // len(dates))
        wide = np.tile(cost, (1, reps))
        row, col = linear_sum_assignment(wide)
        chosen = [
            pool[col[np.flatnonzero(row == k)[0]] % len(dates)][2] for k in range(build.N_MEMBERS)
        ]
        ranks = score.ranks_of_paths(paths=chosen, init=init)
        path = s434.assign(values=run["values"].astype(np.float64), ranks=ranks)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, path.astype(np.float32))
        print(f"{init:%Y-%m-%d} written ({len(dates)} dates)", flush=True)


if __name__ == "__main__":
    main()
