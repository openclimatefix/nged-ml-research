# ruff: noqa: RUF059, PERF401, B023
"""Experiment 565: the learnt D5 offset and the A10 embedding per Primary over the whole series.

Reads the last fold's (2026-06) pretrained weights, which hold one knot per month from 2019-09
(82 knots), for seeds 0 to 4. D5's knot is a scalar in standardised units; multiplied by the
Primary's target standard deviation it is a capacity fraction, and by the capacity it is MW or MVA.
A10's knot has 4 numbers; the first principal component of a Primary's 82 knots (pooled over seeds,
sign fixed so the newest knot is positive) stands for its embedding. The training targets of the
fold are not needed: the standard deviation comes from the cleaned power table.

Writes `offset_knots.csv` (one row per Primary, seed, month) and `offset_events.csv` (one row per
labelled event: the offset's change from onset to end, and the same change over event-free windows
of the same length at the same Primary) under `work/scores/`.
"""

from datetime import UTC, datetime

import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00565_design6 import W559, W

MONTH = "2026-06"
SEEDS = (0, 1, 2, 3, 4)
N_KNOTS = 82
START = datetime(2019, 9, 1, tzinfo=UTC)
SITES = pl.read_csv(
    next(W.parents[1].glob("00521*/site_map.csv"))
)  # primary, label, capacity_mw, group


def knot_date(i: int) -> datetime:
    """Return the first day of month `i` after 2019-09."""
    y, m = divmod(START.month - 1 + i, 12)
    return datetime(START.year + y, m + 1, 1, tzinfo=UTC)


def main() -> None:
    """Write the knots and the per-event movements."""
    clean = c.clean_stack()  # (primaries, grid) capacity fraction, NaN where flagged
    sd = np.nanstd(clean, axis=1)
    rows = []
    for seed in SEEDS:
        d5 = torch.load(W / "pretrained_D5/real" / f"{MONTH}_s{seed}.pt", map_location="cpu")[
            "knots"
        ][:, :, 0].numpy()
        a10 = torch.load(
            W559 / "pretrained_drift/real" / f"{MONTH}_s{seed}.pt", map_location="cpu"
        )["knots"].numpy()
        assert d5.shape == (16, N_KNOTS)
        for p in range(16):
            x = a10[p] - a10[p].mean(axis=0)
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            pc = x @ vt[0]
            if pc[-1] < 0:
                pc = -pc
            for i in range(N_KNOTS):
                rows.append(
                    {
                        "primary": p,
                        "seed": seed,
                        "month_index": i,
                        "month": knot_date(i).date().isoformat(),
                        "d5_standardised": float(d5[p, i]),
                        "d5_capacity_fraction": float(d5[p, i] * sd[p]),
                        "d5_mw": float(d5[p, i] * sd[p] * SITES["capacity_mw"][p]),
                        "a10_pc1": float(pc[i]),
                        "a10_pc1_explained": float(s[0] ** 2 / (s**2).sum()),
                    }
                )
    knots = pl.DataFrame(rows).join(SITES.select("primary", "label", "group"), on="primary")
    out = W / "scores"
    knots.write_csv(out / "offset_knots.csv")

    ev = c.labelled_events()
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    mean = (
        knots.group_by("primary", "month_index")
        .agg(pl.col("d5_capacity_fraction").mean())
        .sort("primary", "month_index")
    )
    rng = np.random.default_rng(0)
    res = []
    for p, spans in ev.items():
        off = mean.filter(pl.col("primary") == p)["d5_capacity_fraction"].to_numpy()
        busy = np.zeros(N_KNOTS, dtype=bool)
        pos = []
        for onset, end in spans:
            t0 = datetime.fromtimestamp((grid + onset * step) / 1e6, UTC)
            t1 = datetime.fromtimestamp((grid + end * step) / 1e6, UTC)
            m0 = (t0.year - START.year) * 12 + t0.month - START.month + (t0.day - 1) / 30
            m1 = (t1.year - START.year) * 12 + t1.month - START.month + (t1.day - 1) / 30
            pos.append((m0, m1))
            busy[max(int(m0) - 1, 0) : min(int(m1) + 2, N_KNOTS)] = True
        interp = lambda m: float(np.interp(m, np.arange(N_KNOTS), off))  # noqa: E731
        for m0, m1 in pos:
            length = m1 - m0
            starts = [
                s
                for s in np.arange(0, N_KNOTS - 1 - length, 0.25)
                if not busy[int(s) : int(s + length) + 2].any()
            ]
            ctrl = (
                [
                    abs(interp(s + length) - interp(s))
                    for s in rng.choice(starts, size=min(20, len(starts)), replace=False)
                ]
                if starts
                else []
            )
            res.append(
                {
                    "primary": p,
                    "label": SITES["label"][p],
                    "event_months": length,
                    "offset_change": interp(m1) - interp(m0),
                    "abs_change": abs(interp(m1) - interp(m0)),
                    "control_abs_change_mean": float(np.mean(ctrl)) if ctrl else None,
                    "m_start": m0,
                    "m_end": m1,
                }
            )
    pl.DataFrame(res).write_csv(out / "offset_events.csv")
    t = pl.DataFrame(res)
    print(
        t.height,
        "events; mean |change| events",
        t["abs_change"].mean(),
        "controls",
        t["control_abs_change_mean"].mean(),
    )
    print(
        t.filter(pl.col("event_months") >= 0.5).select(
            pl.len(), pl.col("abs_change").mean(), pl.col("control_abs_change_mean").mean()
        )
    )


if __name__ == "__main__":
    main()
