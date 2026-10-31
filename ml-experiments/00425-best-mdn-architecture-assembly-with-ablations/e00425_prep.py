"""Fold rows for experiment 425: experiment 418's fold builder under the standing training recipe.

The fine-tuning rows are experiment 397's summary-scheme rows, as in `e00418_prep.build_ens_fold`,
with experiment 407's `build_train` cuts added. The ERA5-and-CAMS pretraining rows are experiment
418's, with the same cuts. Two cuts apply to both sets of rows:

- a D15 or D18 row is dropped if its valid time, or the time of its oldest (weekly) lag, is before
  2025-03-01;
- a row is dropped if its own half-hour, or the half-hour of any of its three origin lags, is a
  flagged exact zero (`substation_zero` in the cleaned power table).

The number of rows each cut removes is returned beside every fold so the fit can log it. With
`cut=False` neither cut applies; that is the 3-fit cross-check of the plan section "Arms" and
nothing else. `cleaned_scoring_context` returns the scoring actuals without the flagged half-hours.

Commands (run with `PYTHONPATH` set as `e00425_fit.py` shows):

- `check`: build one fold with and without the cuts and print the row counts the cuts removed.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00425-best-mdn-architecture-assembly-with-ablations"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder; every other experiment's folder is only read."""
MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
"""The 12 monthly folds of experiment 418's 12-fold result (its stored `seed_preds` months)."""
SHORT_HISTORY: Final[tuple[str, ...]] = fit407.SHORT_HISTORY
CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
US_PER_DAY: Final[int] = 86_400 * 1_000_000
US_PER_HOUR: Final[int] = 3_600 * 1_000_000


@dataclass(frozen=True)
class DroppedRows:
    """How many rows (otherwise kept by the fold's window) each cut removed."""

    short_history_start: int = 0
    flagged_zero: int = 0

    def __add__(self, other: DroppedRows) -> DroppedRows:
        """Return the sum of two counts."""
        return DroppedRows(
            short_history_start=self.short_history_start + other.short_history_start,
            flagged_zero=self.flagged_zero + other.flagged_zero,
        )

    def as_dict(self) -> dict[str, int]:
        """Return the counts as a JSON-ready mapping."""
        return {
            "short_history_start": self.short_history_start,
            "flagged_zero": self.flagged_zero,
        }


@dataclass
class EnsFold(prep418.EnsFold):
    """Experiment 418's fold rows, plus the rows the two cuts removed."""

    dropped: DroppedRows = DroppedRows()


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def apply_cuts(
    *,
    keep: np.ndarray,
    valid: np.ndarray,
    lead: np.ndarray,
    site: str,
    series_id: int,
    flagged: dict[int, np.ndarray],
) -> tuple[np.ndarray, DroppedRows]:
    """Return `keep` with the short-history start cut and the flagged-zero drop applied.

    Args:
        keep: Rows the fold's window already keeps (not modified).
        valid: Valid time of every row, microseconds since the epoch.
        lead: Lead of every row in hours; the lag ages follow from it.
        site: The Primary's label (`D15` and `D18` get the start cut).
        series_id: The Primary's `time_series_id`, the key into `flagged`.
        flagged: Each series' flagged exact-zero half-hours, from `flagged_times`.

    Returns:
        The reduced mask, and how many of the rows in `keep` each cut removed (the start cut
        first, the flagged-zero drop among the rows that remain).
    """
    keep = keep.copy()
    lag_us = (base.lag_ages_hours(lead_hours=lead) * US_PER_HOUR).astype(np.int64)
    start_cut = 0
    if site in SHORT_HISTORY:
        too_early = (valid - lag_us[:, 2]) < _us(t=CUT)
        start_cut = int((keep & too_early).sum())
        keep &= ~too_early
    flagged_zero = 0
    bad = flagged.get(series_id)
    if bad is not None:
        hit = np.isin(valid, bad)
        for j in range(lag_us.shape[1]):
            hit |= np.isin(valid - lag_us[:, j], bad)
        flagged_zero = int((keep & hit).sum())
        keep &= ~hit
    return keep, DroppedRows(short_history_start=start_cut, flagged_zero=flagged_zero)


def build_ens_fold(*, refit: datetime, cut: bool = True) -> EnsFold:
    """Return experiment 418's summary-scheme fold with the standing cuts applied.

    Without the cuts (`cut=False`) the rows equal `e00418_prep.build_ens_fold`. The standardiser
    and the per-Primary target statistics are fitted on the rows that remain.
    """
    _, mask, layout = base.feature_names(scheme="summary")
    flagged = fit407.flagged_times() if cut else {}
    dropped = DroppedRows()
    parts = []
    for row in base.primaries().iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        if cut:
            keep, counts = apply_cuts(
                keep=keep,
                valid=z["valid"],
                lead=z["lead"],
                site=row["site"],
                series_id=int(row["time_series_id"]),
                flagged=flagged,
            )
            dropped += counts
        if not keep.any():
            continue
        parts.append(
            {k: z[k][keep] for k in ("x", "y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(base.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return EnsFold(
        x=standardiser.apply(x=cat["x"][keep]),
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin_day=origin_day[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
        valid=cat["valid"][keep],
        dropped=dropped,
    )


def fold_pretrain(
    *, fold: EnsFold, refit: datetime, cut: bool = True
) -> tuple[prep418.PretrainSet, DroppedRows]:
    """Return a fold's ERA5-and-CAMS pretraining rows on the fold's input scale, and the cut counts.

    As `e00418_prep.fold_pretrain` with `noisy=False`, plus the two cuts of `apply_cuts`: a D15 or
    D18 pretraining row is dropped if its target or its weekly lag is before 2025-03-01, and any
    row whose target or lag half-hour is a flagged exact zero is dropped. The `std_*` columns take
    the fold's median ENS spread by lead band. A row is kept only if its valid time is at least 3.5
    hours before the refit and its origin is at or before it; the held-out weeks and their purge
    are the same absolute weeks as the fold's.
    """
    table, _ = prep418.spread_tables(fold=fold)
    _, std_cols = prep418.column_indices()
    cutoff_valid, cutoff_origin = _us(t=refit - regimes.REFIT_CUTOFF_AGE), _us(t=refit)
    flagged = fit407.flagged_times() if cut else {}
    dropped = DroppedRows()
    files = []
    total = 0
    for row in base.primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        origin = z["origin"]
        keep = (z["valid"] <= cutoff_valid) & (origin <= cutoff_origin)
        day = ((origin - _us(t=base.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
        held, purged = base.held_out_weeks(origin_day=day)
        keep &= ~purged
        if cut:
            keep, counts = apply_cuts(
                keep=keep,
                valid=z["valid"],
                lead=z["lead"],
                site=row["site"],
                series_id=int(row["time_series_id"]),
                flagged=flagged,
            )
            dropped += counts
        if keep.any():
            files.append((row["primary"], row["site"], keep, held))
            total += int(keep.sum())
    out = prep418.PretrainSet(
        x=np.empty((total, fold.x.shape[1]), dtype=np.float32),
        y=np.empty(total, dtype=np.float32),
        primary=np.empty(total, dtype=np.int64),
        is_val=np.empty(total, dtype=bool),
        lead=np.empty(total, dtype=np.float32),
    )
    start = 0
    for primary, site, keep, held in files:
        z = np.load(prep418.RAW_DIR / f"{site}.npz")
        n = int(keep.sum())
        raw, lead = z["x"][keep], z["lead"][keep]
        assert (z["valid"][keep] <= cutoff_valid).all()
        assert (z["origin"][keep] <= cutoff_origin).all()
        band = np.minimum((lead // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1)
        for a in range(0, n, prep418.CHUNK):
            b = min(a + prep418.CHUNK, n)
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            out.x[start + a : start + b] = chunk
        out.y[start : start + n] = z["y"][keep]
        out.primary[start : start + n] = primary
        out.is_val[start : start + n] = held[keep]
        out.lead[start : start + n] = lead
        start += n
    return out, dropped


def cleaned_scoring_context(*, series: pl.DataFrame) -> dict:
    """Return experiment 381's scoring context with the flagged exact-zero actuals removed.

    The flagged half-hours are the same ones the training rows drop. Prints how many actuals the
    removal took.
    """
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(fit407.FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    return ctx


def main() -> None:
    """Build one fold with and without the cuts and print what the cuts removed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check",))
    parser.add_argument("--month", default="2026-01")
    args = parser.parse_args()
    refit = base.refit_of(month=args.month)
    cut = build_ens_fold(refit=refit)
    uncut = build_ens_fold(refit=refit, cut=False)
    reference = prep418.build_ens_fold(refit=refit)
    assert uncut.x.shape == reference.x.shape, (uncut.x.shape, reference.x.shape)
    print(f"fold {args.month}: {len(cut.x):,} cut rows, {len(uncut.x):,} uncut rows")
    print(f"fine-tuning rows dropped: {cut.dropped.as_dict()}")
    pre, pre_dropped = fold_pretrain(fold=cut, refit=refit)
    print(f"pretraining rows kept {len(pre.x):,}; dropped: {pre_dropped.as_dict()}")


if __name__ == "__main__":
    main()
