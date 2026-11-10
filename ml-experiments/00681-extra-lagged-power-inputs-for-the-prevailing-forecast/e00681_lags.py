# ruff: noqa: ANN001, D103, E501
"""Experiment 681, step 1: build the extra lagged-power blocks of the prevailing head (CPU).

For each fold month, for every (Primary, origin) row of experiment 667's head data (real rows and
bank rows), this writes three blocks of 14 residual lags, in fractions of capacity (the cleaned
power minus the Candidate's lag-free median, as the head's existing gap columns are):

- `true`: the Primary's own series (real rows: the cleaned power; bank rows: the power with the pass's
  injected shift, which is the series the target is built from);
- `placebo`: another Primary of the same metering group, not an electrical neighbour of the row's
  Primary, at the same times, from the cleaned power (no injected shift);
- `shifted7`: the `true` block with every input read 7 days earlier.

Column order (hours are before the origin; every input ends at origin minus 6 hours or earlier):
0-7 the residual at 6, 9, 12, 18, 24, 36, 48, 72 h; 8-10 the mean over the last 6, 12, 24 h ending at
origin minus 6 h; 11-13 the differences between consecutive trailing 24-hour means (the 24 h ending
at origin minus 6 h minus the 24 h before it, and so on, three differences).

Usage: python e00681_lags.py MONTH [--selftest]
"""

import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(next(Path(__file__).resolve().parents[1].glob("00667-*/prevailing"))))

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00650_bank as gen  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00667_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import p667_data as dat  # ty: ignore[unresolved-import]
import polars as pl

OUT: Final = (
    dat.ROOT.parent / "00681-extra-lagged-power-inputs-for-the-prevailing-forecast" / "lags"
)
DELAY: Final[int] = 12
"""The data delay in half-hours: the newest reading an input may use is the origin minus 6 hours."""
POINT_HOURS: Final = (6, 9, 12, 18, 24, 36, 48, 72)
MEAN_HOURS: Final = (6, 12, 24)
N_COLS: Final[int] = 14
SHIFT7: Final[int] = 7 * 48


def trailing_mean(
    *, csum: np.ndarray, ccount: np.ndarray, prim, end: np.ndarray, width: int
) -> np.ndarray:
    """Return the mean of finite values over the half-hours `(end - width, end]` (NaN if under half are finite)."""
    lo = end - width
    ok = lo >= 0
    s = csum[prim, end] - np.where(ok, csum[prim, np.clip(lo, 0, None)], 0.0)
    c = ccount[prim, end] - np.where(ok, ccount[prim, np.clip(lo, 0, None)], 0)
    return np.where(ok & (c >= width / 2), s / np.maximum(c, 1), np.nan).astype(np.float32)


def lag_block(
    *, series: np.ndarray, median: np.ndarray, prim: np.ndarray, oi: np.ndarray, back: int = 0
) -> np.ndarray:
    """Return the (rows, 14) residual lag block, reading nothing newer than `oi - DELAY - back`."""
    res = (series - median).astype(np.float64)
    fin = np.isfinite(res)
    csum = np.cumsum(np.where(fin, res, 0.0), axis=1)
    ccount = np.cumsum(fin, axis=1)
    end = (oi - DELAY - back).astype(np.int64)
    assert (end <= oi - DELAY).all(), "an input reads a half-hour newer than origin minus 6 hours"
    out = np.full((len(prim), N_COLS), np.nan, np.float32)
    for j, h in enumerate(POINT_HOURS):
        idx = oi - 2 * h - back
        out[:, j] = np.where(idx >= 0, res[prim, np.clip(idx, 0, None)], np.nan)
    for j, h in enumerate(MEAN_HOURS):
        out[:, 8 + j] = trailing_mean(csum=csum, ccount=ccount, prim=prim, end=end, width=2 * h)
    m = [
        trailing_mean(csum=csum, ccount=ccount, prim=prim, end=end - 48 * k, width=48)
        for k in range(4)
    ]
    for j in range(3):
        out[:, 11 + j] = m[j] - m[j + 1]
    return out


def placebo_map() -> np.ndarray:
    """Return, for each Primary, another Primary of the same metering group that is no electrical neighbour of it (fixed)."""
    nb_of, _cap, is_mw = c650.world()
    n = len(is_mw)
    out = np.zeros(n, np.int64)
    for p in range(n):
        cand = [
            q for q in range(n)
            if q != p and is_mw[q] == is_mw[p] and q not in nb_of[p] and p not in nb_of[q]
        ]  # fmt: skip
        assert cand, f"no placebo Primary for {p}"
        out[p] = cand[p % len(cand)]
    return out


def blocks(*, series_true, series_other, median, prim, oi, pmap) -> dict[str, np.ndarray]:
    return {
        "true": lag_block(series=series_true, median=median, prim=prim, oi=oi),
        "placebo": lag_block(series=series_other, median=median, prim=pmap[prim], oi=oi),
        "shifted7": lag_block(series=series_true, median=median, prim=prim, oi=oi, back=SHIFT7),
    }


def selftest(*, clean: np.ndarray, median: np.ndarray, prim: np.ndarray, oi: np.ndarray) -> None:
    """Perturbing every half-hour newer than origin minus 6 hours must not move any input.

    The rows are real origins, where the lag-free median is finite, so the test is not vacuous. A
    positive control perturbs the half-hour at origin minus 6 hours and expects the inputs to move.
    """
    rng = np.random.default_rng(0)
    pick = rng.choice(len(prim), size=min(200, len(prim)), replace=False)
    prim, oi = prim[pick].astype(np.int64), oi[pick].astype(np.int64)
    base = lag_block(series=clean, median=median, prim=prim, oi=oi)
    assert np.isfinite(base).mean() > 0.5, "the selftest rows have mostly missing inputs"
    moved = 0
    for k in range(len(prim)):
        bad = clean.copy()
        bad[prim[k], oi[k] - DELAY + 1 :] += 123.0
        got = lag_block(series=bad, median=median, prim=prim[k : k + 1], oi=oi[k : k + 1])
        assert np.array_equal(got, base[k : k + 1], equal_nan=True), "an input moved"
        ctl = clean.copy()
        ctl[prim[k], oi[k] - DELAY] += 123.0
        got = lag_block(series=ctl, median=median, prim=prim[k : k + 1], oi=oi[k : k + 1])
        moved += int(not np.array_equal(got, base[k : k + 1], equal_nan=True))
    assert moved > 0.9 * len(prim), "the positive control did not move the inputs"
    print("selftest passed: no input reads newer than origin minus 6 hours", flush=True)


def main(month: str, *, run_selftest: bool) -> None:
    ext.redirect()
    clean = k578.clean()
    median = dat.month_table(month=month).median
    if run_selftest:
        r0 = np.load(dat.HEAD_DATA / f"real_{month}.npz")
        selftest(clean=clean, median=median, prim=r0["org_prim"], oi=r0["org_oi"])
    pmap = placebo_map()
    nb_of, cap, is_mw = c650.world()
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    real = np.load(dat.HEAD_DATA / f"real_{month}.npz")
    for k, v in blocks(series_true=clean, series_other=clean, median=median,
                       prim=real["org_prim"].astype(np.int64), oi=real["org_oi"].astype(np.int64), pmap=pmap).items():  # fmt: skip
        out[f"real_{k}"] = v
    bank = np.load(dat.HEAD_DATA / f"bank_{month}.npz")
    ms = c650.month_start_idx(month=month)
    bp, bo, bq = (bank[k].astype(np.int64) for k in ("org_prim", "org_oi", "org_pass"))
    parts = {
        k: np.full((len(bp), N_COLS), np.nan, np.float32) for k in ("true", "placebo", "shifted7")
    }
    view = dat.generator_view(clean=clean, month=month)
    events_now = []
    assert set(bq.tolist()) == set(range(gen.N_TRAIN_PASSES)), "a training pass has no bank rows"
    for r in range(gen.N_TRAIN_PASSES):
        own, _mirror, events = gen.build(clean=view, month_idx=ms, month_start_idx=ms, r=r, nb_of=nb_of,
                                         capacity=cap, is_mw=is_mw, window_end_idx=dat.YEAR_END_IDX)  # fmt: skip
        events_now += [e.row() for e in events]
        n = clean.shape[1]
        fixed = np.where(np.isfinite(clean), clean + own[:, :n], clean)
        sel = bq == r
        for k, v in blocks(series_true=fixed, series_other=clean, median=median, prim=bp[sel], oi=bo[sel], pmap=pmap).items():  # fmt: skip
            parts[k][sel] = v
    for k, v in parts.items():
        out[f"bank_{k}"] = v
    stored = pl.read_parquet(dat.HEAD_DATA / f"events_{month}.parquet")
    assert pl.DataFrame(events_now).equals(stored), (
        "regenerated injected events differ from experiment 667's stored events"
    )
    share = (np.isfinite(out["real_true"]).mean(), np.isfinite(out["bank_true"]).mean())
    assert abs(share[0] - share[1]) < 0.1, (
        f"finite share differs between real and bank rows: {share}"
    )
    np.savez(OUT / f"lags_{month}.npz", **out)
    print(
        f"{month}: real {len(real['org_prim'])} origins, bank {len(bp)} origins; events match the stored table; finite share {np.isfinite(out['real_true']).mean():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], run_selftest="--selftest" in sys.argv)
