"""Experiment 507, N0 (CPU, before any fit): leak and boundary checks, label counts, shift shapes.

- `leaks`: for each screen month's lab table, that crossfit rows stop before the refit, that
  `further` rows start at it, and whether the level of the residual (reading minus lagless median,
  over the Primary's spread) jumps at the refit boundary or at a cross-fit block edge by more than it
  moves between adjacent 14-day windows elsewhere.
- `labels`: the number of labelled events and the confirmation months.
- `shapes`: for the labelled events with a measured deviation, how constant the deviation is over
  each event and how it correlates with the Primary's own daily demand shape.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_n0.py leaks
```
"""

import argparse
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00435_nra_actuals as nra  # ty: ignore[unresolved-import]
import e00507_common as c  # ty: ignore[unresolved-import]
import e00507_lagless as lab  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WINDOW: Final[int] = 14 * 48
OUT: Final = c.WORK_DIR / "n0"


def residual_series(*, month: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (residual over spread, source code 0/1, refit grid index) per Primary and half-hour."""
    spec = c.load_spec()
    table = pl.read_parquet(lab.table_path(month=month, seed=0))
    n_p, n = spec.real.shape
    q50 = np.full((n_p, n), np.nan, dtype=np.float32)
    src = np.full((n_p, n), -1, dtype=np.int8)
    p, g = table["primary"].to_numpy().astype(np.int64), table["grid"].to_numpy().astype(np.int64)
    q50[p, g] = table["q50"].to_numpy()
    src[p, g] = (table["source"] == "further").to_numpy().astype(np.int8)
    resid = (spec.real - q50) / spec.spread[:, None]
    return resid, src, int(common.grid_index(times=prep.refit_of(month=month)))


def window_mean(*, resid: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Return each Primary's mean residual over grid `[lo, hi)`."""
    return np.nanmean(resid[:, lo:hi], axis=1)


def leaks() -> None:
    """Check the table's time boundaries and print the level jumps."""
    rows = []
    for month in c.SCREEN_MONTHS:
        resid, src, refit = residual_series(month=month)
        first_cross = int(np.nonzero((src == 0).any(axis=0))[0].min())
        last_cross = int(np.nonzero((src == 0).any(axis=0))[0].max())
        first_further = int(np.nonzero((src == 1).any(axis=0))[0].min())
        assert last_cross < refit <= first_further, f"{month}: a crossfit row reaches the refit"
        jump = np.abs(
            window_mean(resid=resid, lo=refit, hi=refit + WINDOW)
            - window_mean(resid=resid, lo=refit - WINDOW, hi=refit)
        )
        # Usual movement between adjacent 14-day windows inside the cross-fit part.
        starts = np.arange(first_cross + WINDOW, refit - 2 * WINDOW, WINDOW)
        usual = np.stack(
            [
                np.abs(
                    window_mean(resid=resid, lo=s, hi=s + WINDOW)
                    - window_mean(resid=resid, lo=s - WINDOW, hi=s)
                )
                for s in starts
            ]
        )
        usual_q90 = np.nanquantile(usual, 0.9, axis=0)
        n_block = refit - first_cross
        edges = [first_cross + n_block * k // lab.lag.FOLDS for k in range(1, lab.lag.FOLDS)]
        block_jump = np.stack(
            [
                np.abs(
                    window_mean(resid=resid, lo=e, hi=e + WINDOW)
                    - window_mean(resid=resid, lo=e - WINDOW, hi=e)
                )
                for e in edges
            ]
        )
        rows.append(
            {"month": month, "crossfit_first": first_cross, "crossfit_last": last_cross,
             "further_first": first_further, "refit": refit,
             "refit_jump_median_over_primaries": float(np.nanmedian(jump)),
             "usual_q90_median_over_primaries": float(np.nanmedian(usual_q90)),
             "refit_jump_above_usual_q90": int(np.nansum(jump > usual_q90)),
             "block_edge_jump_median": float(np.nanmedian(block_jump)),
             "block_edge_above_usual_q90": int(np.nansum(block_jump > usual_q90[None, :]))}
        )  # fmt: skip
    table = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT / "leaks.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=220):
        print(table)
    print("leak check passed: no crossfit row at or after any refit; `further` starts at the refit")


def event_rows() -> tuple[pl.DataFrame, dict[int, float]]:
    """Return experiment 400's corrected event rows (one per event and Primary) and the measured steps."""
    events = pl.read_parquet(nra.EVENTS)
    valid = events.filter(
        pl.col("depth_mw").is_not_null() & (pl.col("depth_mw") != 0) & (pl.col("onset_idx") >= 0)
    )
    return events, nra.measured_steps(events=valid)


def month_of(*, idx: pl.Series) -> list[str]:
    """Return the calendar month label of each grid index."""
    times = common.grid_times()
    return [f"{t.year}-{t.month:02d}" for t in times.gather(idx.to_list())]


def labels() -> None:
    """Print the label counts and the confirmation months."""
    events, measured = event_rows()
    months = month_of(idx=events["onset_idx"])
    screen = set(c.SCREEN_MONTHS)
    confirm = sorted(
        {
            m for m in months
            if m not in screen and "2020-09" <= m < "2026-07"
        }
    )  # fmt: skip
    in_confirm = sum(m in confirm for m in months)
    six = events.filter(
        pl.col("label").is_in(list(c.SIGNED_SIX)) & pl.col("event_id").is_in(list(measured))
    )
    print(f"labelled event rows: {events.height}; with a measured deviation: {len(measured)} "
          f"({six.height} at the signed-MW six)")  # fmt: skip
    print(f"confirmation months ({len(confirm)}): {confirm}")
    print(f"labelled event rows in the confirmation months: {in_confirm}")


def shapes() -> None:
    """Report how constant the deviation is over each measured event, and its shape correlation."""
    spec = c.load_spec()
    resid, _, _ = residual_series(month="2026-06")
    events, measured = event_rows()
    sites = c.sites()
    rows = []
    for e in events.filter(pl.col("event_id").is_in(list(measured))).iter_rows(named=True):
        p = sites.index(e["label"])
        g0, g1 = int(e["onset_idx"]), int(e["end_idx"]) + 1
        base = np.nanmedian(resid[p, max(g0 - 7 * 48, 0) : g0] * spec.spread[p])
        dev = resid[p, g0:g1] * spec.spread[p] - base
        shape = spec.shape[p, g0:g1]
        ok = np.isfinite(dev)
        if ok.sum() < 8:
            continue
        d, s = dev[ok], shape[ok]
        mean = float(d.mean())
        corr = float(np.corrcoef(d, s)[0, 1]) if s.std() > 0 and d.std() > 0 else float("nan")
        rows.append(
            {"primary": e["label"], "event_id": e["event_id"], "hours": (g1 - g0) / 2.0,
             "mean_dev_mw": mean * spec.capacity_mw[p],
             "cv": float(d.std() / abs(mean)) if mean else float("nan"), "corr_with_demand": corr}
        )  # fmt: skip
    out = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT / "shapes.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(out)
    print(
        f"{out.height} events; median CV {out['cv'].median():.2f}; median correlation with own "
        f"demand shape {out['corr_with_demand'].median():.2f}; "
        f"share with correlation above 0.5: {(out['corr_with_demand'] > 0.5).mean():.2f}"
    )


def main() -> None:
    """Run one N0 check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("leaks", "labels", "shapes"))
    args = parser.parse_args()
    c.use_lab_tables()
    {"leaks": leaks, "labels": labels, "shapes": shapes}[args.command]()


if __name__ == "__main__":
    main()
