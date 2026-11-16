# ruff: noqa: E501, I001
"""Experiment 701, GAM side, step 3: the ensemble member arrays for the runs of 2026-07-01 to 2026-10-31.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:ml-experiments/00700-*:ml-experiments/00701-* \
    DATA_PATH_INTERNAL=<real data folder> uv run --no-sync python ml-experiments/00701-*/e00701_gam_ens.py check|build|gaps

The builder is `e00377_ens_inputs.read_run`, experiment 152's extraction: for each daily 00:00 UTC
run of the local ECMWF ENS archive (`<data>/NWP/nwp_model_id=ECMWF_ENS_0_25_degree`) it cuts the
51 members to each Primary's H3 cell (resolution 5), resamples them to half-hours with production's
`_upsample_nwp_to_half_hourly`, and keeps the valid times from the run's native origin (run plus 9
hours) to 351 hours later, which is 702 half-hours. The arrays are `temperature` and `shortwave`,
shape (runs, 51, 702), float32, as in experiment 111's and 152's `ens_store`.

- **`check`** rebuilds 10 runs of 2026-05 and 2026-06 at all 15 Primaries and prints the largest
  absolute difference from the stored arrays (and whether the `NaN` positions agree).
- **`build`** reads the 123 new runs with 8 processes and writes
  `<sandbox>/ml-experiments/00111-*/work/ens_store/<site>.npz` (the seven signed-MW Primaries) and
  `.../00152-*/work/ens_store/<site>.npz` (the eight MVA-only Primaries): the 821 stored runs
  followed by the 123 new runs, 944 in all. The stored files are only read.
- **`gaps`** counts `NaN` values in each new array by month, and the half-hours of each Primary's
  14-day contract window that hold no actual power reading, by origin month.
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta

import e00701_gam_extend377  # noqa: F401  (dates must be extended before e00377_ens_inputs imports e00377_ens)
import e00701_gam_extend as ext
import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_ens_inputs as ens_inputs  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

CONTROL = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
MVA = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SITES = (*CONTROL, *MVA)
STORED_STORE = {
    **dict.fromkeys(
        CONTROL,
        ext.REAL_DATA
        / "ml-experiments"
        / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
        / "work"
        / "ens_store",
    ),
    **dict.fromkeys(
        MVA,
        ext.REAL_DATA
        / "ml-experiments"
        / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries"
        / "work"
        / "ens_store",
    ),
}
SANDBOX_STORE = {
    **dict.fromkeys(
        CONTROL,
        ext.SANDBOX_ML
        / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
        / "work"
        / "ens_store",
    ),
    **dict.fromkeys(
        MVA,
        ext.SANDBOX_ML
        / "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries"
        / "work"
        / "ens_store",
    ),
}
SAMPLE_RUNS = [
    datetime(2026, m, d, tzinfo=UTC)
    for m, d in (
        (5, 5),
        (5, 12),
        (5, 19),
        (5, 26),
        (6, 2),
        (6, 9),
        (6, 16),
        (6, 22),
        (6, 26),
        (6, 30),
    )
]
N_STORED_RUNS = 821
WORKERS = 8
_TABLE: list[pl.DataFrame] = []


def _init() -> None:
    _TABLE.append(ens_inputs.site_table(sites=SITES))


def _read(run: datetime) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return ens_inputs.read_run(run=run, sites=_TABLE[0])


def new_runs() -> list[datetime]:
    """Return the daily runs from 2026-07-01 to 2026-10-31."""
    return [
        ext.FIRST_ORIGIN_MONTH + timedelta(days=i)
        for i in range((ext.LAST_RUN - ext.FIRST_ORIGIN_MONTH).days + 1)
    ]


def read_all(*, runs: list[datetime]) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Read runs with `WORKERS` processes, in order."""
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as pool:
        return list(pool.map(_read, runs, chunksize=4))


def stored_index(*, run: datetime) -> int:
    """Return a run's position in the stored arrays (they start on 2024-04-01)."""
    return (run - common.FIRST_RUN).days


def check() -> None:
    """Rebuild the sample runs and compare them with the stored arrays."""
    started = time.monotonic()
    results = read_all(runs=SAMPLE_RUNS)
    print(
        f"rebuilt {len(SAMPLE_RUNS)} runs at {len(SITES)} Primaries in {time.monotonic() - started:.0f} s"
    )
    worst = 0.0
    for site in SITES:
        with np.load(STORED_STORE[site] / f"{site}.npz") as stored:
            assert stored["temperature"].shape == (N_STORED_RUNS, 51, 702)
            parts = []
            for run, result in zip(SAMPLE_RUNS, results, strict=True):
                i = stored_index(run=run)
                assert stored["origins"][i] == np.datetime64(
                    (run + common.PUBLICATION_DELAY).replace(tzinfo=None), "us"
                )
                for name, new in zip(("temperature", "shortwave"), result[site], strict=True):
                    old = stored[name][i]
                    nan_equal = bool((np.isnan(old) == np.isnan(new)).all())
                    both = np.isfinite(old) & np.isfinite(new)
                    gap = float(np.abs(old[both] - new[both]).max())
                    assert nan_equal, (site, run, name)
                    assert new.dtype == old.dtype, (site, run, name, new.dtype, old.dtype)
                    assert new.shape == old.shape, (site, run, name, new.shape)
                    worst = max(worst, gap)
                    parts.append(gap)
        print(site, "max abs diff over 10 runs, temperature and shortwave:", max(parts))
    print("WORST max abs diff:", worst)


def build() -> None:
    """Write the 944-run arrays of every Primary into the sandbox."""
    runs = new_runs()
    assert len(runs) == 123
    assert common.run_dates()[-1] == runs[-1]
    assert len(common.run_dates()) == N_STORED_RUNS + 123
    started = time.monotonic()
    results = read_all(runs=runs)
    print(f"read {len(runs)} runs in {time.monotonic() - started:.0f} s")
    new_origins = np.array(
        [np.datetime64(common.origin_of(run=run).replace(tzinfo=None), "us") for run in runs]
    )
    for site in SITES:
        with np.load(STORED_STORE[site] / f"{site}.npz") as stored:
            arrays = {name: stored[name] for name in stored.files}
        temperature = np.stack([r[site][0] for r in results])
        shortwave = np.stack([r[site][1] for r in results])
        assert arrays["origins"][-1] + np.timedelta64(1, "D") == new_origins[0]
        out = {
            "origins": np.concatenate([arrays["origins"], new_origins]),
            "temperature": np.concatenate([arrays["temperature"], temperature]),
            "shortwave": np.concatenate([arrays["shortwave"], shortwave]),
        }
        assert out["temperature"].shape == (N_STORED_RUNS + 123, 51, 702)
        target = SANDBOX_STORE[site] / f"{site}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{site}.tmp.npz")
        np.savez(temporary, **out)  # ty: ignore[invalid-argument-type]
        temporary.rename(target)
        print(site, out["temperature"].shape, f"{time.monotonic() - started:.0f} s", flush=True)


def gaps() -> None:
    """Report `NaN` counts of the new arrays by month and missing actuals by origin month."""
    runs = new_runs()
    month_of = np.array([run.month for run in runs])
    for site in SITES:
        with np.load(SANDBOX_STORE[site] / f"{site}.npz") as stored:
            temperature, shortwave = (
                stored["temperature"][N_STORED_RUNS:],
                stored["shortwave"][N_STORED_RUNS:],
            )
        line = {
            int(m): int(
                np.isnan(temperature[month_of == m]).sum()
                + np.isnan(shortwave[month_of == m]).sum()
            )
            for m in (7, 8, 9, 10)
        }
        print(site, "NaN values in the new arrays by month:", line)
    cache = common.load_cache()
    print(
        "missing actual half-hours in each contract window (12:00 to 12:00 14 days later), summed over origins, by origin month:"
    )
    for site in (s for s in SITES if s != "D15"):
        present = cache.sites[site].present & np.isfinite(cache.sites[site].y)
        missing = dict.fromkeys((7, 8, 9, 10), 0)
        short = dict.fromkeys((7, 8, 9, 10), 0)
        for run in runs:
            origin = np.datetime64((run + timedelta(hours=12)).replace(tzinfo=None), "us")
            start = int(np.searchsorted(cache.end_times, origin, side="right"))
            stop = start + 672
            count = int((~present[start:stop]).sum())
            missing[run.month] += count
            short[run.month] += int(count > 0)
        print(site, "missing half-hours:", missing, "origins with a gap:", short)


if __name__ == "__main__":
    {"check": check, "build": build, "gaps": gaps}[sys.argv[1]]()
