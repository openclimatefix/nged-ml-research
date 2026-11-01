"""Experiment 492: two constants per metering group in place of table P, on experiment 483's cache.

For each candidate and metering group, a lower factor scales the distance of q01 and q05 from the
median and an upper factor scales the distance of q95 and q99. Each factor is the grid value that
minimises the summed pinball loss of its two quantiles on the fitting rows (clean rows). Two fits:
`earlier` (each scored fold is fitted on the folds before it; 2025-10 is never scored) against
table P fitted the same way, and `lofo` (leave-one-fold-out) against table P leave-one-fold-out.

Reads `data/ml-experiments/00483-*/work/cache`; writes `data/ml-experiments/00492-*/work/`.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
PYTHONPATH=ml-experiments/00483-uncertainty-rerun-on-the-first-release-demand-candidates:\
ml-experiments/00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails:\
ml-experiments/00470-first-release-demand-forecaster-assembly-and-comparison \
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00492-*/e00492_two_constant.py
```
"""

from pathlib import Path
from typing import Final

import e00435_table_p_provisional as prov  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00483_tails as tails  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00492-two-constant-tail-widening-in-place-of-table-p-for-demand"
    / "work"
)
ARMS: Final[tuple[str, ...]] = ("A-pool", "A-gam-in", "K2-425A")
GROUPS: Final[tuple[str, ...]] = ("mva_eight", "signed_six", "report_only")
GRID: Final[np.ndarray] = np.round(np.arange(0.50, 1.801, 0.01), 2)
LOWER: Final[tuple[str, ...]] = ("q01", "q05")
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
TAU: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
RESAMPLES: Final[int] = 20_000
BONFERRONI_COUNT: Final[int] = 24
LIMIT_COVERAGE: Final[float] = 0.01
LIMIT_PINBALL: Final[float] = 0.05
SEED: Final[int] = 492


def pinball(*, y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Return the per-row pinball loss of quantile `q` at level `tau`."""
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)


def fit_constant(*, fit: pl.DataFrame, names: tuple[str, ...]) -> float:
    """Return the grid factor minimising the summed pinball loss of `names` on clean `fit` rows."""
    y = fit["actual"].to_numpy().astype(np.float64)
    q50 = fit["q50"].to_numpy().astype(np.float64)
    loss = np.zeros(len(GRID))
    for name in names:
        d = fit[name].to_numpy().astype(np.float64) - q50
        for i, f in enumerate(GRID):
            loss[i] += pinball(y=y, q=q50 + f * d, tau=TAU[name]).mean()
    return float(GRID[int(np.argmin(loss))])


def fit_constants(*, fit: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Return (lower, upper) factors per metering group from clean fitting rows."""
    clean = fit.filter(pl.col("cls") == "clean")
    return {
        g: (
            fit_constant(fit=part, names=LOWER),
            fit_constant(fit=part, names=UPPER),
        )
        for g in GROUPS
        if (part := clean.filter(pl.col("group") == g)).height
    }


def apply_constants(
    *, frame: pl.DataFrame, constants: dict[str, tuple[float, float]]
) -> pl.DataFrame:
    """Scale the lower and upper tail quantiles' distance from the median by the group's factors."""
    lo = pl.col("group").replace_strict({g: v[0] for g, v in constants.items()}, default=1.0)
    up = pl.col("group").replace_strict({g: v[1] for g, v in constants.items()}, default=1.0)
    return frame.with_columns(
        *[(pl.col("q50") + lo * (pl.col(q) - pl.col("q50"))).alias(q) for q in LOWER],
        *[(pl.col("q50") + up * (pl.col(q) - pl.col("q50"))).alias(q) for q in UPPER],
    )


def table_p(*, frame: pl.DataFrame, fit: pl.DataFrame) -> pl.DataFrame:
    """Apply table P fitted on the clean rows of `fit` to `frame`."""
    tagged = fit.with_columns(group_orig=pl.col("group"))
    factors = tails.fit_with_fallback(fit=tagged.filter(pl.col("cls") == "clean"), cold_class=False)
    return prov.apply_factors(frame=frame, factors=factors)


def scheme_frames(*, arm: str, scheme: str) -> tuple[dict[str, pl.DataFrame], dict]:
    """Return the held-out frame of each variant and the fitted constants per scored fold."""
    frame = tails.load(arm=arm)
    scored = tails.MONTHS[1:] if scheme == "earlier" else tails.MONTHS
    out: dict[str, list[pl.DataFrame]] = {"raw": [], "two_constant": [], "table_p": []}
    constants: dict = {}
    for month in scored:
        if scheme == "earlier":
            fit = frame.filter(pl.col("month").is_in(tails.MONTHS[: tails.MONTHS.index(month)]))
        else:
            fit = frame.filter(pl.col("month") != month)
        held = frame.filter((pl.col("month") == month) & (pl.col("cls") == "clean"))
        constants[month] = fit_constants(fit=fit)
        out["raw"].append(held)
        out["two_constant"].append(apply_constants(frame=held, constants=constants[month]))
        out["table_p"].append(table_p(frame=held, fit=fit).filter(pl.col("cls") == "clean"))
    return {k: pl.concat(v) for k, v in out.items()}, constants


def row_values(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row CRPS, coverage, and pinball loss (points) as experiment 483 scores them."""
    m = score470.row_metrics(frame=frame)
    return pl.DataFrame({k: v for k, v in m.items() if k == "crps" or k[:3] in ("cov", "pin")})


def interval(
    *, diff: np.ndarray, blocks: np.ndarray, level: float, rng: np.random.Generator
) -> tuple[float, float]:
    """Bound the mean of `diff` by a block bootstrap over `blocks` at `level` percent."""
    ids, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids)).astype(np.float64)
    pick = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    mean = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return float(np.percentile(mean, tail)), float(np.percentile(mean, 100.0 - tail))


def main() -> None:
    """Fit, score, and write the tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    level = 100.0 * (1.0 - 0.05 / BONFERRONI_COUNT)
    scores: list[pl.DataFrame] = []
    contrasts: list[dict] = []
    const_rows: list[dict] = []
    for arm in ARMS:
        for scheme in ("earlier", "lofo"):
            frames, constants = scheme_frames(arm=arm, scheme=scheme)
            for month, per_group in constants.items():
                for g, (lo, up) in per_group.items():
                    const_rows.append(
                        {
                            "arm": arm,
                            "scheme": scheme,
                            "month": month,
                            "group": g,
                            "lower": lo,
                            "upper": up,
                        }
                    )
            for variant, frame in frames.items():
                for name, part in tails.slices(frame=frame).items():
                    if name in ("real_event_nra", "real_event_metered"):
                        continue
                    for by in (["group"], ["group", "month"]):
                        s = tails.summarise(frame=part, by=by)
                        scores.append(
                            s.with_columns(
                                arm=pl.lit(arm), scheme=pl.lit(scheme), variant=pl.lit(variant),
                                slice=pl.lit(name), by=pl.lit("+".join(by)),
                            )
                        )  # fmt: skip
            two, tab = frames["two_constant"], frames["table_p"]
            assert two.select(tails.KEYS).equals(tab.select(tails.KEYS)), "row mismatch"
            for slice_name in ("clean", "clean_12_36h", "clean_cold"):
                m_two = tails.slices(frame=two)[slice_name]
                m_tab = tails.slices(frame=tab)[slice_name]
                for g in GROUPS:
                    a = m_two.filter(pl.col("group") == g)
                    b = m_tab.filter(pl.col("group") == g)
                    if a.is_empty():
                        continue
                    va, vb = row_values(frame=a), row_values(frame=b)
                    blocks = (a["month"] + "_" + a["day"].cast(pl.String)).to_numpy()
                    for col in va.columns:
                        diff = va[col].to_numpy() - vb[col].to_numpy()
                        low, high = interval(diff=diff, blocks=blocks, level=level, rng=rng)
                        relative = (
                            float(diff.mean() / vb[col].to_numpy().mean())
                            if col.startswith("pin") or col == "crps"
                            else None
                        )
                        contrasts.append(
                            {
                                "arm": arm, "scheme": scheme, "slice": slice_name, "group": g,
                                "metric": col, "two_constant": float(va[col].to_numpy().mean()),
                                "table_p": float(vb[col].to_numpy().mean()),
                                "diff": float(diff.mean()),
                                "low": low, "high": high, "n": a.height,
                                "relative": relative,
                            }
                        )  # fmt: skip
            print("done", arm, scheme, flush=True)
    pl.concat(scores, how="diagonal").write_parquet(WORK / "scores.parquet")
    pl.DataFrame(contrasts).write_csv(WORK / "contrasts.csv")
    pl.DataFrame(const_rows).write_csv(WORK / "constants.csv")


if __name__ == "__main__":
    main()
