# ruff: noqa: E501
"""Experiment 625: K-path cuts, anomaly-path Schaake shuffle, and path checks on stored scenarios.

Reads experiment 436's `work/runs`, `work/observed.npz`, and `work/paths/<arm>/<date>.npy` (51 paths
per arm; not recomputed), builds the SS-R arm (a Schaake shuffle that ranks anomaly paths, not
observed paths) and the K-path cuts, and writes five tables to this experiment's `work/` folder:

- `coherence.parquet`: energy (fair form) and variogram scores by block, as in experiment 436;
- `crps.parquet`: fair CRPS of each arm's paths against the actual, by family and window;
- `coverage.parquet`: coverage of the member spread at 50%, 80%, and 90%;
- `sky.parquet`: opposite-sky-state counts of neighbouring solar farms, by run and valid day;
- `acf.parquet`: lag-1 autocorrelation of the step-to-step change along a path.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00625-*/env.sh
OMP_NUM_THREADS=1 nice -n 19 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00625-*/e00625_score.py --workers 3
```
"""

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00434_score as s434  # ty: ignore[unresolved-import]
import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00625-coherent-members-from-quantiles-k-sweep-and-path-checks"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
WORK436: Final[Path] = score.WORK
N: Final[int] = build.N_NODES
S: Final[int] = build.N_STEPS
KIND: Final[np.ndarray] = build.KIND
KS: Final[tuple[int, ...]] = (5, 10, 20, 51)
CUT_ARMS: Final[tuple[str, ...]] = ("J0", "J2", "J3", "J5", "SSR")
FULL_ARMS: Final[tuple[str, ...]] = ("J6",)
"""J6 (member-wise) has no meaningful cut to K, so it is scored at 51 only."""
ANOMALY_LAGS: Final[tuple[int, ...]] = (7, 14, 21, 28)
"""Days before a path's start whose same clock time forms its baseline (same weekday)."""
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
FAMILIES: Final[dict[str, int]] = {"demand": 0, "wind": 1, "solar": 2}
COVER_LEVELS: Final[dict[str, float]] = {"50": 0.5, "80": 0.8, "90": 0.9}
SKY_PAIRS: Final[int] = 5
DAYLIGHT: Final[tuple[float, float]] = (10.0, 14.0)
ACF_WINDOWS: Final[dict[str, tuple[int, int]]] = {"h12_36": (23, 72), "d1_3": (47, 143)}
"""Step ranges (0 = lead 0.5 h) of the two windows the autocorrelation is computed over."""


def fair_energy_score(*, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the energy score per batch item with the pair term over distinct members only.

    Experiment 434's form averages the pair term over all M squared pairs, which biases a small M
    towards a spread that is too small. This form divides by M(M-1), so different M compare.
    """
    m = mask[:, None, :]
    members = x.shape[1]
    to_obs = np.sqrt((((x - y[:, None, :]) * m) ** 2).sum(-1)).mean(-1)
    pair = np.sqrt((((x[:, :, None, :] - x[:, None, :, :]) * m[:, :, None, :]) ** 2).sum(-1))
    score_ = to_obs - 0.5 * pair.sum((-1, -2)) / (members * (members - 1))
    return score_ / np.sqrt(mask.sum(-1))


s434.energy_score = fair_energy_score


def cut_paths(*, paths: np.ndarray, k: int) -> np.ndarray:
    """Return K paths (k, N, S) from an arm's 51 paths: every (51/K)-th member, re-valued at K levels.

    Each kept member keeps its rank among the K at every node and half-hour. Its value is
    the 51 sorted values interpolated at level rank / (K + 1) in the usual plotting-position form.
    """
    if k == paths.shape[0]:
        return paths
    sorted_values = np.sort(paths, axis=0)
    take = np.round(np.linspace(0, paths.shape[0] - 1, k)).astype(np.int64)
    ranks = np.argsort(np.argsort(paths[take], axis=0, kind="stable"), axis=0, kind="stable")
    pos = np.clip((ranks + 1) / (k + 1) * (paths.shape[0] + 1) - 1, 0, paths.shape[0] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, paths.shape[0] - 1)
    w = pos - lo
    return np.take_along_axis(sorted_values, lo, axis=0) * (1 - w) + np.take_along_axis(
        sorted_values, hi, axis=0
    ) * w


def anomaly_ssr(*, init: datetime, observed: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Return the SS-R paths (51, N, S): Schaake ranks of anomaly paths, applied to the 51 values.

    The anomaly of a past date is its observed path minus the mean of the same clock time on the
    four same-weekday dates 7 to 28 days before that date. The 51 dates, the 16-day buffer, and the gap filling are
    those of experiment 436's J3.
    """
    pool = score.schaake_pool(init=init, observed=observed, kind=kind)
    chosen = [pool[k % len(pool)] for k in range(s434.N_MEMBERS)]
    anomalies = []
    for _, date, path in chosen:
        i = build.grid_index(t=date) + 1
        base = np.stack(
            [observed[:, i - 48 * d : i - 48 * d + S] for d in ANOMALY_LAGS]
        )
        with np.errstate(all="ignore"):
            baseline = np.nanmean(base, axis=0)
        baseline = np.where(np.isnan(baseline), path.mean(axis=1, keepdims=True), baseline)
        anomalies.append(path - baseline)
    return score.ranks_of_paths(paths=anomalies, init=init)


def fair_crps(*, paths: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the fair CRPS (N, S) of K paths (K, N, S) against the actual (N, S)."""
    k = paths.shape[0]
    srt = np.sort(paths, axis=0)
    coeff = (2 * np.arange(1, k + 1) - k - 1).reshape(-1, 1, 1)
    spread = (coeff * srt).sum(0) * 2 / (k * (k - 1))
    return np.abs(paths - actual[None]).mean(0) - 0.5 * spread


def _acf1(*, path: np.ndarray) -> np.ndarray:
    """Return the lag-1 autocorrelation of the first difference along the last axis."""
    d = np.diff(path, axis=-1)
    a, b = d[..., :-1], d[..., 1:]
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a**2).sum(-1) * (b**2).sum(-1))
    return np.where(den > 1e-9, (a * b).sum(-1) / np.maximum(den, 1e-9), np.nan)


def sky_pairs(*, observed: np.ndarray, kind: np.ndarray) -> list[tuple[int, int]]:
    """Return the `SKY_PAIRS` solar-farm pairs with the highest correlation of daylight output.

    Uses observed half-hours from 2025-03-01 to 2025-09-01 only, before the first scored run.
    """
    solar = np.flatnonzero(kind == 2)
    first = build.grid_index(t=score.SCHAAKE_FIRST)
    last = build.grid_index(t=datetime(2025, 9, 1, tzinfo=UTC))
    idx = np.arange(first, last)
    hours = np.array([(build.GRID_START + i * build.STEP).hour for i in idx])
    sel = idx[(hours >= DAYLIGHT[0]) & (hours < DAYLIGHT[1])]
    corr = []
    for a in range(len(solar)):
        for b in range(a + 1, len(solar)):
            x, y = observed[solar[a], sel], observed[solar[b], sel]
            ok = ~np.isnan(x) & ~np.isnan(y)
            corr.append((float(np.corrcoef(x[ok], y[ok])[0, 1]), int(solar[a]), int(solar[b])))
    corr.sort(reverse=True)
    return [(a, b) for _, a, b in corr[:SKY_PAIRS]]


def sky_rows(
    *, label: str, k: int, paths: np.ndarray | None, actual: np.ndarray, thresholds: np.ndarray,
    pairs: list[tuple[int, int]], init: datetime, steps: np.ndarray, spread: np.ndarray,
) -> list[dict]:  # fmt: skip
    """Return, per valid day, the counts of opposite-sky-state triples for a set of paths.

    A triple is one path (or the actual, if `paths` is None), one pair of neighbouring solar farms,
    and one daylight half-hour; it is opposite when one farm is above its own q80 and the other
    below its own q20 of the stored 51 values.
    """
    lo, hi = thresholds[0], thresholds[1]
    days = np.array([(init + (int(i) + 1) * build.STEP).toordinal() for i in steps])
    rows = []
    for day in np.unique(days):
        cols = steps[days == day]
        count, total = 0, 0
        for a, b in pairs:
            if paths is None:
                ya, yb = actual[a, cols], actual[b, cols]
                ok = ~np.isnan(ya) & ~np.isnan(yb)
                opp = ((ya > hi[a, cols]) & (yb < lo[b, cols])) | (
                    (yb > hi[b, cols]) & (ya < lo[a, cols])
                )
                count += int((opp & ok).sum())
                total += int(ok.sum())
            else:
                pa, pb = paths[:, a, cols], paths[:, b, cols]
                opp = ((pa > hi[a, cols]) & (pb < lo[b, cols])) | (
                    (pb > hi[b, cols]) & (pa < lo[a, cols])
                )
                count += int(opp.sum())
                total += int(opp.size)
        rows.append(
            {"power_fcst_init_time": init, "day": int(day), "arm": label, "k": k,
             "opposite": count, "total": total, "spread": float(spread[days == day].mean())}
        )  # fmt: skip
    return rows


def process_run(*, file: str) -> dict[str, list[dict]]:  # noqa: PLR0912, PLR0915
    """Score every arm and every K on one run; return the rows of the five tables."""
    run = score.load_run(path=Path(file))
    init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
    observed = np.load(WORK436 / "observed.npz")["observed"]
    kind = run["kind"]
    actual = run["actual"].astype(np.float64)
    values = run["values"].astype(np.float64)
    day_tag = f"{init:%Y-%m-%d}"
    arms51: dict[str, np.ndarray] = {}
    for arm in (*CUT_ARMS, *FULL_ARMS):
        if arm == "SSR":
            ranks = anomaly_ssr(init=init, observed=observed, kind=kind)
            arms51[arm] = s434.assign(values=values, ranks=ranks)
            if not np.array_equal(np.sort(arms51[arm], axis=0), values.transpose(2, 0, 1)):
                msg = f"{day_tag}: SSR is not a re-ordering of the 51 values"
                raise ValueError(msg)
        else:
            arms51[arm] = np.load(WORK436 / "paths" / arm / f"{day_tag}.npy").astype(np.float64)
    out: dict[str, list[dict]] = {k: [] for k in ("coherence", "crps", "coverage", "sky", "acf")}
    masks = score.step_masks(init=init)
    have = ~np.isnan(actual)
    y0 = np.nan_to_num(actual)
    pairs = sky_pairs(observed=observed, kind=kind)
    q = np.quantile(values, [0.2, 0.8], axis=-1)
    day_steps = np.flatnonzero(masks["daylight"] & np.isin(np.arange(S), s434.window_steps(window="days_1_10")))
    spread_all = (np.quantile(values, 0.9, axis=-1) - np.quantile(values, 0.1, axis=-1))[kind == 2].mean(0)
    out["sky"].extend(
        sky_rows(label="OBS", k=0, paths=None, actual=actual, thresholds=q, pairs=pairs,
                 init=init, steps=day_steps, spread=spread_all[day_steps])
    )  # fmt: skip
    direct = fair_crps(paths=arms51["J0"], actual=y0)
    for k in KS:
        group = {a: cut_paths(paths=arms51[a], k=k) for a in CUT_ARMS}
        if k == 51:
            group.update({a: arms51[a] for a in FULL_ARMS})
        score.N_MEMBERS = k
        for row in score.rows_for_run(paths=group, actual=actual, kind=kind, init=init):
            out["coherence"].append({**row, "k": k})
        for arm, path in group.items():
            crps = fair_crps(paths=path, actual=y0)
            for window in s434.WINDOWS:
                steps = s434.window_steps(window=window)
                for fam, f in FAMILIES.items():
                    nodes = np.flatnonzero(kind == f)
                    ok = have[np.ix_(nodes, steps)]
                    if not ok.any():
                        continue
                    out["crps"].append(
                        {"power_fcst_init_time": init, "arm": arm, "k": k, "window": window,
                         "family": fam, "crps": float(crps[np.ix_(nodes, steps)][ok].mean()),
                         "direct": float(direct[np.ix_(nodes, steps)][ok].mean()), "n": int(ok.sum())}
                    )  # fmt: skip
                    for name, level in COVER_LEVELS.items():
                        lo_q = np.quantile(path, (1 - level) / 2, axis=0)
                        hi_q = np.quantile(path, 1 - (1 - level) / 2, axis=0)
                        inside = (actual >= lo_q) & (actual <= hi_q)
                        out["coverage"].append(
                            {"power_fcst_init_time": init, "arm": arm, "k": k, "window": window,
                             "family": fam, "level": name,
                             "coverage": float(inside[np.ix_(nodes, steps)][ok].mean()), "n": int(ok.sum())}
                        )  # fmt: skip
            out["sky"].extend(
                sky_rows(label=arm, k=k, paths=path, actual=actual, thresholds=q, pairs=pairs,
                         init=init, steps=day_steps, spread=spread_all[day_steps])
            )  # fmt: skip
            for wname, (a, b) in ACF_WINDOWS.items():
                acf = _acf1(path=path[:, :, a:b])  # (k, N)
                for fam, f in FAMILIES.items():
                    nodes = np.flatnonzero(kind == f)
                    out["acf"].append(
                        {"power_fcst_init_time": init, "arm": arm, "k": k, "window": wname,
                         "family": fam, "acf1": float(np.nanmean(acf[:, nodes]))}
                    )  # fmt: skip
    for wname, (a, b) in ACF_WINDOWS.items():
        seg = actual[:, a:b]
        good = ~np.isnan(seg).any(axis=1)
        acf = np.where(good, _acf1(path=np.nan_to_num(seg)), np.nan)
        for fam, f in FAMILIES.items():
            nodes = np.flatnonzero(kind == f)
            if good[nodes].any():
                out["acf"].append(
                    {"power_fcst_init_time": init, "arm": "OBS", "k": 0, "window": wname,
                     "family": fam, "acf1": float(np.nanmean(acf[nodes]))}
                )  # fmt: skip
    return out


def main() -> None:
    """Score every stored run, in parallel worker processes, and write the five tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = [str(f) for f in sorted((WORK436 / "runs").glob("*.npz"))[: args.max_runs]]
    tables: dict[str, list[dict]] = {k: [] for k in ("coherence", "crps", "coverage", "sky", "acf")}
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for file, result in zip(files, pool.map(process_run_kw, files), strict=True):
            for name, rows in result.items():
                tables[name].extend(rows)
            print(f"{Path(file).stem} scored", flush=True)
    for name, rows in tables.items():
        pl.DataFrame(rows).write_parquet(OUT / f"{name}{args.suffix}.parquet")


def process_run_kw(file: str) -> dict[str, list[dict]]:
    """Return `process_run` for one file (a positional wrapper for `pool.map`)."""
    return process_run(file=file)


if __name__ == "__main__":
    main()
