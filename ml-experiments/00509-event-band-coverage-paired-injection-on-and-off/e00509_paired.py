# ruff: noqa: E501, RUF046, ANN201
"""Experiment 509: coverage and width of the forecast bands with the lags injected and not injected.

Reads the row-level injected quantiles that experiment 482 stored for three donors (D5, D12, D14)
in five test months (15 clusters), joins the stored injection-off forecasts of experiments 454
(LL, S1, B-mix, K2-S1: three seed cells averaged level by level) and 470 (A-pool), and writes a
sufficient-statistics cube under `data/ml-experiments/00509-*/work/cube.npz`. Nothing is fitted.

Layout of one cell of the cube (last axis, `STAT_NAMES`): the row count, the count of actuals at or
below q01, q05, q95, and q99, the count inside the central 90% interval, the sum of the 90% width
and of the q05 and q95 pinball losses (fractions of capacity), 10 PIT bins, then two histograms
of the widening ratio (upper and lower tail) in steps of 0.01 up to `RATIO_MAX`.

```bash
export DATA_PATH_INTERNAL=<private data folder>
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00509-*/e00509_paired.py
```
"""

import re
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR
W482: Final[Path] = (
    ROOT / "00482-best-models-performance-analysis-injected-shifts-and-wind-references" / "work"
)
W454: Final[Path] = (
    ROOT / "00454-lag-trust-mixture-and-ensemble-constructs-for-lagged-and-lagless-heads" / "work"
)
W470: Final[Path] = ROOT / "00470-first-release-demand-forecaster-assembly-and-comparison" / "work"
WORK: Final[Path] = ROOT / "00509-event-band-coverage-paired-injection-on-and-off" / "work"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
DONORS: Final[tuple[str, ...]] = ("D5", "D12", "D14")
SCENARIOS: Final[tuple[int, ...]] = tuple(range(1, 9))
ARMS: Final[tuple[str, ...]] = ("LL", "S1", "B-mix", "K2-S1", "A-pool")
CLASSES: Final[tuple[str, ...]] = ("before", "newest_in", "older_in", "after")
BANDS: Final[dict[str, tuple[float, float]]] = {"h12_36": (12.0, 36.0), "d1_10": (24.0, 240.0)}
VARIANTS: Final[tuple[str, ...]] = ("injected", "clean")
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVEL_P: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
KEYS: Final[list[str]] = ["primary", "origin", "target"]
STEP: Final[float] = 0.01
RATIO_MAX: Final[float] = 6.0
N_RATIO: Final[int] = int(round(RATIO_MAX / STEP)) + 1
SCALAR_NAMES: Final[tuple[str, ...]] = (
    "n", "c01", "c05", "c95", "c99", "c90", "w90", "pin05", "pin95",
)  # fmt: skip
STAT_NAMES: Final[tuple[str, ...]] = (
    *SCALAR_NAMES, *(f"pit{i}" for i in range(10)), "hup", "hlo",
)  # fmt: skip
N_SCALAR: Final[int] = len(SCALAR_NAMES)
N_STAT: Final[int] = N_SCALAR + 10 + 2 * N_RATIO
REPRO_TOLERANCE: Final[float] = 1e-5


def clean_forecast(*, arm: str, month: str) -> pl.DataFrame:
    """Return the injection-off quantiles of an arm and month, keyed by `KEYS`."""
    if arm == "A-pool":
        frame = pl.read_parquet(W470 / "predictions" / "A-pool" / f"{month}_c0.parquet")
        return frame.select(*[pl.col(k).cast(pl.Int64) for k in KEYS], *LEVELS)
    cells = sorted((W454 / "predictions" / arm).glob(f"{month}_s*_w-1.parquet"))
    assert len(cells) == 3, (arm, month, len(cells))
    return average_cells(frames=[pl.read_parquet(c) for c in cells])


def average_cells(*, frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Average the 11 quantile levels of seed cells that share their keys and row order."""
    first = frames[0]
    for f in frames[1:]:
        for k in KEYS:
            assert np.array_equal(f[k].to_numpy(), first[k].to_numpy()), f"keys differ: {k}"
    mean = np.mean([f.select(LEVELS).to_numpy().astype(np.float64) for f in frames], axis=0)
    return pl.DataFrame(
        {
            **{k: first[k].cast(pl.Int64) for k in KEYS},
            **{name: mean[:, i] for i, name in enumerate(LEVELS)},
        }
    )


def pit(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the probability integral transform of `y` under the piecewise-linear CDF of `q`.

    Values below q01 get 0.005 and above q99 get 0.995.
    """
    k = (q <= y[:, None]).sum(axis=1)
    out = np.empty(len(y))
    out[k == 0] = 0.005
    out[k == len(LEVEL_P)] = 0.995
    mid = (k > 0) & (k < len(LEVEL_P))
    lo, hi = k[mid] - 1, k[mid]
    idx = np.nonzero(mid)[0]
    q_lo, q_hi = q[idx, lo], q[idx, hi]
    width = np.maximum(q_hi - q_lo, 1e-12)
    out[mid] = LEVEL_P[lo] + (LEVEL_P[hi] - LEVEL_P[lo]) * np.clip((y[idx] - q_lo) / width, 0, 1)
    return out


def ratio_bins(*, ratio: np.ndarray) -> np.ndarray:
    """Return histogram counts of `ratio` in steps of `STEP` (bin k holds ratio in ((k-1)STEP, k STEP])."""
    idx = np.clip(np.ceil(ratio / STEP - 1e-9), 0, N_RATIO - 1).astype(np.int64)
    return np.bincount(idx, minlength=N_RATIO).astype(np.float64)


def cell_stats(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the sufficient statistics (`STAT_NAMES`) of one group of rows."""
    out = np.zeros(N_STAT)
    n = len(y)
    if n == 0:
        return out
    q = np.sort(q, axis=1)
    med = q[:, 5]
    out[0] = n
    out[1] = (y <= q[:, 0]).sum()
    out[2] = (y <= q[:, 1]).sum()
    out[3] = (y <= q[:, 9]).sum()
    out[4] = (y <= q[:, 10]).sum()
    out[5] = ((y >= q[:, 1]) & (y <= q[:, 9])).sum()
    out[6] = (q[:, 9] - q[:, 1]).sum()
    out[7] = np.maximum(0.05 * (y - q[:, 1]), -0.95 * (y - q[:, 1])).sum()
    out[8] = np.maximum(0.95 * (y - q[:, 9]), -0.05 * (y - q[:, 9])).sum()
    out[N_SCALAR : N_SCALAR + 10] = np.bincount(
        np.minimum((pit(q=q, y=y) * 10).astype(np.int64), 9), minlength=10
    )
    up = (y - med) / np.maximum(q[:, 9] - med, 1e-9)
    lo = (med - y) / np.maximum(med - q[:, 1], 1e-9)
    base = N_SCALAR + 10
    out[base : base + N_RATIO] = ratio_bins(ratio=up)
    out[base + N_RATIO :] = ratio_bins(ratio=lo)
    return out


def groups(*, cls: np.ndarray, lead: np.ndarray):
    """Yield (class index, band name, row mask) for every class and lead band."""
    for ci, c in enumerate(CLASSES):
        for band, (lo, hi) in BANDS.items():
            yield ci, band, (cls == c) & (lead >= lo) & (lead <= hi)


def build() -> None:
    """Fill the cube cluster by cluster and write it, with the reproduction checks."""
    clusters = [(m, d) for m in MONTHS for d in DONORS]
    band_names = list(BANDS)
    cube = np.zeros(
        (
            len(clusters),
            len(SCENARIOS),
            len(ARMS),
            len(VARIANTS),
            len(CLASSES),
            len(band_names),
            N_STAT,
        )
    )
    checks: list[str] = []
    for month in MONTHS:
        clean = {arm: clean_forecast(arm=arm, month=month) for arm in ARMS}
        for donor in DONORS:
            ci_cluster = clusters.index((month, donor))
            for si, sid in enumerate(SCENARIOS):
                rows = pl.read_parquet(W482 / "rows" / f"{month}_s{sid}_{donor}.parquet")
                # keep the donor's own rows: the donor is the Primary of the unmirrored rows
                donor_id = int(
                    pl.read_parquet(W482 / "rows" / f"{month}_s1_{donor}.parquet")["primary"][0]
                )
                rows = rows.filter(pl.col("primary") == donor_id)
                y = rows["y_world"].to_numpy().astype(np.float64)
                cls = rows["cls"].to_numpy()
                lead = rows["lead_hours"].to_numpy().astype(np.float64)
                for ai, arm in enumerate(ARMS):
                    inj = (
                        rows.select([f"{arm}_{lv}" for lv in LEVELS]).to_numpy().astype(np.float64)
                    )
                    joined = rows.select(KEYS).join(
                        clean[arm], on=KEYS, how="left", maintain_order="left"
                    )
                    cln = joined.select(LEVELS).to_numpy().astype(np.float64)
                    missing = np.isnan(cln).any(axis=1)
                    if arm == "A-pool":
                        # A-pool has no forecast for some rows (experiment 482 dropped them from its scores)
                        keep = ~missing
                    else:
                        assert not missing.any(), f"{arm} {month} {donor} s{sid}: null join"
                        keep = np.ones(len(y), dtype=bool)
                    before = (cls == "before") & keep
                    diff = float(np.abs(inj[before] - cln[before]).max()) if before.any() else 0.0
                    assert diff < REPRO_TOLERANCE, (
                        f"{arm} {month} {donor} s{sid}: injected vs clean on before rows differ by {diff:.2e}"
                    )
                    checks.append(
                        f"{month} {donor} s{sid} {arm}: max before-row difference {diff:.2e}; kept {keep.sum()} of {len(y)}"
                    )
                    for vi, q in enumerate((inj, cln)):
                        for ci, band, mask in groups(cls=cls, lead=lead):
                            m = mask & keep
                            cube[ci_cluster, si, ai, vi, ci, band_names.index(band)] = cell_stats(
                                q=q[m], y=y[m]
                            )
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WORK / "cube.npz",
        cube=cube,
        clusters=np.array([f"{m}|{d}" for m, d in clusters]),
        scenarios=np.array(SCENARIOS),
        arms=np.array(ARMS),
        variants=np.array(VARIANTS),
        classes=np.array(CLASSES),
        bands=np.array(band_names),
    )
    (WORK / "reproduction_checks.txt").write_text("\n".join(checks) + "\n")
    print(
        f"wrote cube {cube.shape}; worst before-row difference "
        f"{max(float(re.search(r'difference (\S+);', c).group(1)) for c in checks):.2e}"
    )  # ty: ignore[unresolved-attribute]


def seed_cells() -> None:
    """Write per-seed coverage on the same rows, for the coverage minimum detectable effect.

    K2-S1, B-mix, S1, and LL use the three seed cells of experiment 454; A-pool uses the five seed
    arms A-s0 to A-s4 of experiment 470. Scored on the `newest_in` rows of the days 1 to 10 band
    of each scenario, pooled over the 15 clusters.
    """
    records = []
    for arm in ARMS:
        if arm == "A-pool":
            paths = {
                m: [W470 / "predictions" / f"A-s{s}" / f"{m}_c0.parquet" for s in range(5)]
                for m in MONTHS
            }
        else:
            paths = {
                m: sorted((W454 / "predictions" / arm).glob(f"{m}_s*_w-1.parquet")) for m in MONTHS
            }
        n_seeds = len(paths[MONTHS[0]])
        tallies = {(s, k): np.zeros(3) for s in SCENARIOS for k in range(n_seeds)}
        for month in MONTHS:
            seeds = [
                pl.read_parquet(p).select(*[pl.col(k).cast(pl.Int64) for k in KEYS], *LEVELS)
                for p in paths[month]
            ]
            for donor in DONORS:
                donor_id = int(
                    pl.read_parquet(W482 / "rows" / f"{month}_s1_{donor}.parquet")["primary"][0]
                )
                for sid in SCENARIOS:
                    rows = pl.read_parquet(
                        W482 / "rows" / f"{month}_s{sid}_{donor}.parquet"
                    ).filter(
                        (pl.col("primary") == donor_id)
                        & (pl.col("cls") == "newest_in")
                        & pl.col("lead_hours").is_between(24, 240)
                    )
                    y = rows["y_world"].to_numpy().astype(np.float64)
                    for k, frame in enumerate(seeds):
                        j = rows.select(KEYS).join(
                            frame, on=KEYS, how="left", maintain_order="left"
                        )
                        q = j.select(LEVELS).to_numpy().astype(np.float64)
                        ok = ~np.isnan(q).any(axis=1)
                        tallies[sid, k] += [
                            ok.sum(),
                            (y[ok] <= q[ok, 1]).sum(),
                            (y[ok] <= q[ok, 9]).sum(),
                        ]
        for sid in SCENARIOS:
            for k in range(n_seeds):
                n, c05, c95 = tallies[sid, k]
                records.append(
                    {
                        "arm": arm,
                        "scenario": sid,
                        "seed": k,
                        "n": int(n),
                        "cov_q05": c05 / n,
                        "cov_q95": c95 / n,
                    }
                )
    frame = pl.DataFrame(records)
    out = (
        frame.group_by("arm", "scenario")
        .agg(
            pl.col("n").first(),
            pl.col("cov_q05").std().alias("seed_sd_q05"),
            pl.col("cov_q95").std().alias("seed_sd_q95"),
        )
        .with_columns(mde_q05=2.8 * pl.col("seed_sd_q05"), mde_q95=2.8 * pl.col("seed_sd_q95"))
        .sort("arm", "scenario")
    )
    out.write_csv(WORK / "seed_mde.csv", float_precision=5)
    print(out)


if __name__ == "__main__":
    build()
    seed_cells()
