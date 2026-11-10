"""Compare rules that cap the upper quantiles of series that exceed their table capacity (private).

uv run --no-sync python ml-experiments/00668-*/e00668_upper_quantile_rules.py

For every series of the five forecast files whose largest quantile exceeds the capacity of
`series_capacity.parquet`, writes to `<backtest folder>/clamp_fixes/`: the exceedance counts per
quantile column (`upper_exceedance.csv`), and the whole-year CRPS and q05 to q95 coverage of the
file and of four capping rules (`upper_rules.csv`). Nothing in the forecast files changes.

Rules: (a) cap every quantile at the table capacity; (b) cap at the larger of the table capacity and
the highest cleaned power before the first day of the origin's month; (c) cap at the 99.9th
percentile of the cleaned power in the 730 days before the first day of the origin's month; (d) cap
only `q95` and `q99` at the table capacity. A cap is a monotone `min`, so quantiles stay sorted.
Generators are labelled A to C (wind) and A to F (solar); no generator date is written.
"""

from typing import Final

import e00668_score_lib as lib
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.solar import zenith

RULES: Final[tuple[str, ...]] = (
    "file",
    "a_table_capacity",
    "b_max_prior_or_table",
    "c_p99.9_730d",
    "d_q95_q99_only",
)
WINDOW_DAYS: Final[int] = 730


def capped(
    *, q: np.ndarray, rule: str, cap: float, prior_max: np.ndarray, p999: np.ndarray
) -> np.ndarray:
    """Return the quantile matrix after applying one capping rule."""
    out = q.copy()
    if rule == "file":
        return out
    if rule == "a_table_capacity":
        return np.minimum(out, cap)
    if rule == "b_max_prior_or_table":
        return np.minimum(out, np.maximum(cap, prior_max)[:, None])
    if rule == "c_p99.9_730d":
        return np.minimum(out, p999[:, None])
    out[:, 9:] = np.minimum(out[:, 9:], cap)
    return out


def main() -> None:  # noqa: PLR0915
    """Write the exceedance and rule-comparison tables."""
    out_dir = lib.FOLDER / "clamp_fixes"
    out_dir.mkdir(exist_ok=True)
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    exceed_rows: list[dict] = []
    rule_rows: list[dict] = []
    letters = {"wind": "ABC", "solar": "ABCDEF"}
    for name in ("primaries", "bsps", "gsps", "wind", "solar"):
        forecast = pl.read_parquet(lib.FOLDER / f"{name}.parquet")
        truth = pl.read_parquet(lib.FOLDER / f"ground_truth_{name}.parquet")
        ids = sorted(forecast["time_series_id"].unique().to_list())
        cap_of = dict(
            capacity.filter(pl.col("time_series_id").is_in(ids))
            .select("time_series_id", "capacity")
            .iter_rows()
        )
        unit_of = dict(
            capacity.filter(pl.col("time_series_id").is_in(ids))
            .select("time_series_id", "power_unit")
            .iter_rows()
        )
        grid = lib.cleaned_grid(ids=ids)
        row_of = {t: i for i, t in enumerate(ids)}
        obs_max = {
            t: float(v)
            for t, v in truth.group_by("time_series_id").agg(pl.col("power").max()).iter_rows()
        }
        if name == "wind":
            idle = truth.filter(pl.col("research_flag").is_not_null())
            gi = lib.grid_index(times=idle["valid_time"])
            gr = idle["time_series_id"].replace_strict(row_of).to_numpy()
            score_grid = grid.copy()
            score_grid[gr, gi] = 0.0
        else:
            score_grid = grid
        sites = pv_sites().filter(pl.col("time_series_id").is_in(ids)) if name == "solar" else None
        outputs = ["NRA", "prevailing"] if name == "primaries" else [None]
        for output in outputs:
            part = (
                forecast
                if output is None
                else forecast.filter(pl.col("NRA_or_prevailing") == output)
            )
            for tid in ids:
                cap = float(cap_of[tid])
                sel = part.filter(pl.col("time_series_id") == tid)
                q = sel.select(lib.QCOLS).to_numpy().astype(np.float64)
                if q.max() <= cap:
                    continue
                label = (
                    f"{name.capitalize()} {letters[name][ids.index(tid)]}"
                    if name in letters
                    else f"{name[:-1].upper() if name != 'primaries' else 'Primary'} {tid}"
                )
                above = {c: int((q[:, k] > cap).sum()) for k, c in enumerate(lib.QCOLS)}
                valid_day = sel["valid_time"].dt.date().to_numpy()
                days_above = len(np.unique(valid_day[q[:, 10] > cap]))
                exceed_rows.append(
                    {
                        "series": label,
                        "output": output or "-",
                        "unit": unit_of[tid],
                        "rows": sel.height,
                        "capacity": cap,
                        "largest_ratio": float(q.max() / cap),
                        "observed_max_ratio": obs_max[tid] / cap,
                        "valid_days_with_q99_above": days_above,
                        **{f"rows_{c}_above": v for c, v in above.items()},
                    }
                )
                vidx = lib.grid_index(times=sel["valid_time"])
                if vidx.max() >= grid.shape[1]:
                    pad = int(vidx.max()) + 1 - grid.shape[1]
                    grid = np.pad(grid, ((0, 0), (0, pad)), constant_values=np.nan)
                    score_grid = np.pad(score_grid, ((0, 0), (0, pad)), constant_values=np.nan)
                r = row_of[tid]
                y = score_grid[r, vidx].copy()
                cutoff = lib.grid_index(times=sel["power_fcst_init_time"].dt.truncate("1mo"))
                prior_max = np.empty(len(y))
                p999 = np.empty(len(y))
                for c in np.unique(cutoff):
                    hist = grid[r, max(0, c - WINDOW_DAYS * 48) : c]
                    hist = hist[np.isfinite(hist)]
                    full = grid[r, :c]
                    full = full[np.isfinite(full)]
                    m = cutoff == c
                    prior_max[m] = full.max() if len(full) else -np.inf
                    p999[m] = np.quantile(hist, 0.999) if len(hist) else np.inf
                ok = np.isfinite(y)
                if sites is not None:
                    rec = sites.filter(pl.col("time_series_id") == tid).row(0, named=True)
                    mid = sel["valid_time"].dt.offset_by("-15m")
                    ok &= (
                        np.asarray(
                            zenith(stamps=mid, latitude=rec["latitude"], longitude=rec["longitude"])
                        )
                        < 90.0
                    )
                scale = 100.0 / cap
                for rule in RULES:
                    qr = capped(q=q, rule=rule, cap=cap, prior_max=prior_max, p999=p999)
                    res = lib.losses(y=y[ok] * scale, q=qr[ok] * scale)
                    rule_rows.append(
                        {
                            "series": label,
                            "output": output or "-",
                            "rule": rule,
                            "rows_scored": int(ok.sum()),
                            "rows_changed": int((qr != q).any(axis=1).sum()),
                            "crps": float(res["crps"].mean()),
                            "coverage_q05_q95": float(res["coverage"].mean()),
                            "share_obs_above_q99": float((y[ok] > qr[ok, 10]).mean() * 100.0),
                        }
                    )
                print(label, output, "done", flush=True)
    pl.DataFrame(exceed_rows).write_csv(out_dir / "upper_exceedance.csv")
    pl.DataFrame(rule_rows).write_csv(out_dir / "upper_rules.csv")
    print(pl.DataFrame(rule_rows))


if __name__ == "__main__":
    main()
