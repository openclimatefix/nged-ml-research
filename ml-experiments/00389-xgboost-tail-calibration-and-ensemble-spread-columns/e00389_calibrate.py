"""Build the post-hoc tail arms T1, T2, and T3 for the weekly member and the weekly bundle.

Reads the raw (unwidened) pooled quantiles of two members at all 16 Primaries and writes, for each,
four arms to `work/arms/<member>_<arm>.parquet` (keys, `lead_hours`, the 11 quantiles as fractions
of effective capacity):

- `raw` (T1): factor 1.0.
- `base`: factor 1.15 about the median, the baseline.
- `t2`: a factor per Primary, per lead band, per side, refreshed at each weekly refit.
- `t3`: the baseline plus a shift of q01, q05, q95, and q99 per Primary and lead band.

The members are the weekly XGBoost member of experiment 87 (`W`, from its stored raw refits) and the
weekly bundle of experiment 381 (`B`, read only). Calibration at the refit at time `t` uses only
forecast rows whose valid time is at most 3.5 hours before `t` and at most 365 days before `t`, and
needs 60 distinct valid days for a Primary and lead band. Until then the factor is 1.15 and the
shift is 0. D8, D15, and D18 always use the donor, the mean of the seven other MVA-only
Primaries' values.

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
        DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00389-*/e00389_calibrate.py
"""

import dataclasses
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_assemble as asm  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUT: Final[Path] = DATA / "00389-xgboost-tail-calibration-and-ensemble-spread-columns" / "work"
STORED_BG2: Final[Path] = common.WORK_DIR / "arms" / "p_s6_bg2.parquet"
BUNDLE_DIR: Final[Path] = (
    DATA
    / "00381-simple-weekly-xgboost-demand-against-gam-stack"
    / "work"
    / "arm_preds"
    / "weekly"
    / "bundle_solar+sunny+hl1"
)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
Q: Final[tuple[str, ...]] = tuple(common.LEVEL_COLUMNS)
FIRST_REFIT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_VALID_AGE: Final[timedelta] = timedelta(hours=3.5)
WINDOW: Final[timedelta] = timedelta(days=365)
MIN_DAYS: Final[int] = 60
BAND_EDGES: Final[tuple[float, ...]] = (0.0, 24.0, 72.0, 168.0, 240.0, 1e9)
GRID: Final[np.ndarray] = np.round(np.arange(0.8, 1.6001, 0.05), 2)
BASE_FACTOR: Final[float] = common.SPREAD_FACTOR
DONOR_SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D12", "D19", "D21", "D22")
ALWAYS_DONOR: Final[tuple[str, ...]] = ("D8", "D15", "D18")
UPPER: Final[tuple[tuple[str, float], ...]] = (("q95", 0.95), ("q99", 0.99))
LOWER: Final[tuple[tuple[str, float], ...]] = (("q05", 0.05), ("q01", 0.01))
TAILS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)
N_REFITS: Final[int] = 52
MEDIAN: Final[int] = common.MEDIAN


@dataclasses.dataclass
class Table:
    """Calibration values per refit and lead band: upper factor, lower factor, four shifts."""

    upper: np.ndarray
    lower: np.ndarray
    shift: np.ndarray
    ok: np.ndarray


def band_of(*, lead: np.ndarray) -> np.ndarray:
    """Return the lead-band index of each lead in hours."""
    return np.searchsorted(np.asarray(BAND_EDGES[1:-1]), lead, side="right")


def refit_index(*, origin: np.ndarray) -> np.ndarray:
    """Return the weekly refit that forecasts each run: the refit date at or before the origin."""
    days = (origin.astype("datetime64[D]") - np.datetime64(FIRST_REFIT.date())).astype(np.int64)
    return np.minimum(days // 7, N_REFITS - 1)


def calibrate_site(*, frame: dict[str, np.ndarray], n_refits: int) -> Table:
    """Fit the factors and shifts of one Primary at every refit and lead band."""
    n_bands = len(BAND_EDGES) - 1
    table = Table(
        upper=np.full((n_refits, n_bands), BASE_FACTOR),
        lower=np.full((n_refits, n_bands), BASE_FACTOR),
        shift=np.zeros((n_refits, n_bands, len(TAILS))),
        ok=np.zeros((n_refits, n_bands), dtype=bool),
    )
    valid = frame["valid"]
    band = band_of(lead=frame["lead"])
    median = frame["q50"]
    actual = frame["actual"]
    wide = {
        name: median + BASE_FACTOR * (frame[name] - median) for name in ("q01", "q05", "q95", "q99")
    }
    day = valid.astype("datetime64[D]").astype(np.int64)
    for r in range(n_refits):
        refit = np.datetime64(FIRST_REFIT.replace(tzinfo=None) + timedelta(days=7 * r), "us")
        newest = refit - np.timedelta64(int(LAST_VALID_AGE.total_seconds() * 1e6), "us")
        oldest = refit - np.timedelta64(int(WINDOW.total_seconds() * 1e6), "us")
        in_window = (valid <= newest) & (valid >= oldest) & np.isfinite(actual)
        if not in_window.any():
            continue
        for b in range(n_bands):
            mask = in_window & (band == b)
            if len(np.unique(day[mask])) < MIN_DAYS:
                continue
            med, act = median[mask], actual[mask]
            up = []
            for name, _ in UPPER:
                spread = np.maximum(frame[name][mask] - med, 1e-6)
                up.append((act - med) / spread)
            lo = []
            for name, _ in LOWER:
                spread = np.maximum(med - frame[name][mask], 1e-6)
                lo.append((med - act) / spread)
            table.upper[r, b] = _combined(ratios=up, levels=tuple(v for _, v in UPPER), upper=True)
            table.lower[r, b] = _combined(ratios=lo, levels=tuple(v for _, v in LOWER), upper=False)
            for t, (name, level) in enumerate(TAILS):
                table.shift[r, b, t] = np.quantile(act - wide[name][mask], level)
            table.ok[r, b] = True
    return table


def _combined(*, ratios: list, levels: tuple[float, ...], upper: bool) -> float:
    """Return the factor that is best on the pooled ratios of the quantiles of one side."""
    ordered = [np.sort(r) for r in ratios]
    errors = np.zeros(len(GRID))
    for r, level in zip(ordered, levels, strict=True):
        if upper:
            share = np.searchsorted(r, GRID, side="right") / len(r)
        else:
            share = 1.0 - np.searchsorted(r, GRID, side="left") / len(r)
        errors += np.abs(share - level)
    return float(GRID[int(np.argmin(errors))])


def apply(
    *, frame: pl.DataFrame, tables: dict[str, Table], donor: Table, arm: str, label: str
) -> np.ndarray:
    """Return an arm's sorted quantiles (fractions) for one Primary's rows."""
    quantiles = frame.select(Q).to_numpy().astype(np.float64)
    median = quantiles[:, [MEDIAN]]
    if arm == "raw":
        return quantiles
    wide = median + BASE_FACTOR * (quantiles - median)
    if arm == "base":
        return wide
    r = refit_index(origin=frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy())
    b = band_of(lead=frame["lead_hours"].to_numpy())
    table = donor if label in ALWAYS_DONOR else tables[label]
    if arm == "t2":
        out = quantiles.copy()
        out[:, :MEDIAN] = median + table.lower[r, b][:, None] * (quantiles[:, :MEDIAN] - median)
        out[:, MEDIAN + 1 :] = median + table.upper[r, b][:, None] * (
            quantiles[:, MEDIAN + 1 :] - median
        )
        return np.sort(out, axis=1)
    out = wide.copy()
    for t, (name, _) in enumerate(TAILS):
        out[:, Q.index(name)] += table.shift[r, b, t]
    return np.sort(out, axis=1)


def donor_table(*, tables: dict[str, Table]) -> Table:
    """Return the mean of the donor Primaries' values."""
    donors = [tables[s] for s in DONOR_SITES]
    return Table(
        upper=np.mean([t.upper for t in donors], axis=0),
        lower=np.mean([t.lower for t in donors], axis=0),
        shift=np.mean([t.shift for t in donors], axis=0),
        ok=np.all([t.ok for t in donors], axis=0),
    )


def load_members() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return each member's raw quantiles joined to the actual, and the series label table."""
    stored = pl.read_parquet(STORED_BG2).select(*KEYS, "actual", "site", "capacity_mw")
    labels = stored.select("time_series_id", "site").unique()
    ctx = dataclasses.replace(
        asm.make_context(smoke=False, store=False), results_dir=OUT / "weights"
    )
    primary = ctx.sites_of(groups=("Primary",))
    w = asm.raw_member(ctx=ctx, member="W", offset="vintage", sites=primary)
    ids = labels["time_series_id"].to_list()
    bundle = pl.concat([pl.read_parquet(f) for f in sorted(BUNDLE_DIR.glob("*.parquet"))]).filter(
        pl.col("time_series_id").is_in(ids), pl.col("power_fcst_init_time") >= FIRST_REFIT
    )
    members = {"W": w, "B": bundle}
    s1_files = sorted((OUT / "anchor_preds" / "s1").glob("*.parquet"))
    if os.environ.get("E389_INCLUDE_S1") == "1":
        if len(s1_files) != N_REFITS * len(ids):
            msg = f"S1 is incomplete: {len(s1_files)} files of {N_REFITS * len(ids)}"
            raise FileNotFoundError(msg)
        members["S1"] = pl.concat([pl.read_parquet(f) for f in s1_files]).filter(
            pl.col("power_fcst_init_time") >= FIRST_REFIT
        )
    out = {}
    for name, raw in members.items():
        joined = (
            raw.select(*KEYS, *Q)
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0
            )
            .filter(pl.col("lead_hours") <= 336.0)
            .join(stored.select(*KEYS, "actual", "site"), on=KEYS, how="inner", validate="1:1")
            .sort("time_series_id", "power_fcst_init_time", "valid_time")
        )
        out[name] = joined
    return out, labels


def main() -> None:
    """Build and write every arm."""
    members, _ = load_members()
    (OUT / "arms").mkdir(parents=True, exist_ok=True)
    n_refits = N_REFITS
    for name, frame in members.items():
        print(f"{name}: {frame.height:,} rows, {frame['site'].n_unique()} Primaries", flush=True)
        tables = {}
        for label in sorted(frame["site"].unique().to_list()):
            part = frame.filter(pl.col("site") == label)
            arrays = {
                "valid": part["valid_time"].dt.replace_time_zone(None).to_numpy(),
                "lead": part["lead_hours"].to_numpy(),
                "actual": part["actual"].to_numpy().astype(np.float64),
                **{q: part[q].to_numpy().astype(np.float64) for q in Q},
            }
            tables[label] = calibrate_site(frame=arrays, n_refits=n_refits)
        donor = donor_table(tables=tables)
        for arm in ("raw", "base", "t2", "t3"):
            pieces = []
            for label in sorted(tables):
                part = frame.filter(pl.col("site") == label)
                q = apply(frame=part, tables=tables, donor=donor, arm=arm, label=label)
                pieces.append(
                    part.select(*KEYS, "lead_hours").hstack(
                        pl.DataFrame(q.astype(np.float32), schema=list(Q))
                    )
                )
            result = pl.concat(pieces)
            result.write_parquet(OUT / "arms" / f"{name}_{arm}.parquet")
            print(f"wrote {name}_{arm}: {result.height:,} rows", flush=True)


if __name__ == "__main__":
    main()
