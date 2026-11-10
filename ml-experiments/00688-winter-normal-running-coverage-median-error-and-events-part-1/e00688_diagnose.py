"""Winter coverage of the normal-running Primaries: median error and load transfers (CPU only).

Run it from the repository root with `DATA_PATH_INTERNAL` set to the data folder of the production
checkout and `PYTHONPATH` set to the folders of experiments 668 and 683 and of this experiment:

    PYTHONPATH=ml-experiments/00668-*:ml-experiments/00683-*:ml-experiments/00688-* \
        uv run --no-sync python ml-experiments/00688-*/e00688_diagnose.py

Reads (nothing is fitted): the stored normal-running quantiles of the 16 Primaries
(`primaries.parquet`, arm 0), the walk-forward scaled quantiles of experiment 683 (arm 2), and the
cleaned observed power. Scored rows are experiment 683's: finite cleaned observed power, origin
months 2025-12 to 2026-06.

Stages, applied to arm 0 and to arm 2 alike:

- `none`: the quantiles as they are.
- `events`: drops the half-hours whose valid time lies inside a labelled or verified load transfer
  of the same Primary (the step 1 mask).
- `shift`: adds to all 11 quantiles the median of (observed - q50) over the series' scored rows of
  the origin month (the step 2 oracle re-centring; it is computed on the rows it is scored on, so
  it is a diagnostic and never a forecast).
- `events+shift`: both, the shift computed on the rows that remain after the mask.
- `events+shift per band`: both, one shift per lead band (a sensitivity).

Writes tables (CSV) to `<data>/00688-*/tables/`, and prints every number the write-up quotes.
Intervals are 95% day-block bootstraps over forecast-origin days, with the picks shared across
stages, series, and arm sets.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00683_calibrate as cal  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00688-*"))
TABLES: Final[Path] = DATA / "tables"
DATA_683: Final[Path] = cal.EXPERIMENT_DIR
FIRST_MONTH: Final[str] = "2025-12"
LAST_MONTH: Final[str] = "2026-06"
WINTER: Final[tuple[str, ...]] = ("2025-12", "2026-01")
REST: Final[tuple[str, ...]] = ("2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
STAGES: Final[tuple[str, ...]] = (
    "none",
    "events",
    "shift",
    "events+shift",
    "events+shift per band",
)
SETS: Final[dict[str, str]] = {"stored": "arm 0", "scaled": "arm 2"}
N_BOOT: Final[int] = 2000
THRESHOLD: Final[float] = 0.86
PERIOD_START: Final[datetime] = datetime(2025, 12, 1, tzinfo=UTC)
PERIOD_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
SITE_MAP_GLOB: Final[str] = "00521-*/site_map.csv"
WORK_580: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00580-*")) / "work"
CANDIDATES_685: Final[Path] = (
    next(ML_EXPERIMENTS_DATA_DIR.glob("00685-*")) / "event_table" / "candidate_events.parquet"
)


def idx_of(when: datetime) -> int:
    """Return the half-hour index of an instant on the scoring grid."""
    return int((when - lib.GRID_START).total_seconds() // 1800)


def parse_idx(text: str) -> int:
    """Return the grid index of a `YYYY-MM-DD HH:MM` UTC text from experiment 580's tables."""
    return idx_of(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC))


def label_to_id() -> dict[str, int]:
    """Return the Primary label (such as `D16`) of each time series id, inverted."""
    site = pl.read_csv(next(ML_EXPERIMENTS_DATA_DIR.glob(SITE_MAP_GLOB)))
    return dict(zip(site["label"].to_list(), site["time_series_id"].to_list(), strict=True))


def event_spans(*, kind: str) -> pl.DataFrame:
    """Return the event spans `time_series_id`, `start`, `end` (grid half-hours) of one mask.

    `primary` is the step 1 mask: the labelled events of experiment 580's label table and the
    verified transfers, at both Primaries of each. `wide` adds the fitted spans of experiment 580
    and the candidate events of experiment 685, all clipped to the scored period by the caller.
    """
    ids = label_to_id()
    first, last = idx_of(PERIOD_START), idx_of(PERIOD_END)
    rows: list[dict] = []
    labelled = pl.read_parquet(WORK_580 / "events_labelled_status.parquet")
    for r in labelled.iter_rows(named=True):
        if r["label"] in ids:
            rows.append(
                {
                    "time_series_id": ids[r["label"]],
                    "start": r["label_onset_idx"],
                    "end": r["label_end_idx"] + 1,  # the label end is the last half-hour
                    "source": f"labelled event {r['event_id']}",
                }
            )
            if kind == "wide" and r["fit_onset"] and r["fit_end"]:
                rows.append(
                    {
                        "time_series_id": ids[r["label"]],
                        "start": parse_idx(r["fit_onset"]),
                        "end": parse_idx(r["fit_end"]),
                        "source": f"fitted span of event {r['event_id']}",
                    }
                )
    for r in pl.read_parquet(WORK_580 / "events_verified.parquet").iter_rows(named=True):
        for side in ("a", "b"):
            label = r[f"primary_{side}"]
            if label in ids:
                rows.append(
                    {
                        "time_series_id": ids[label],
                        "start": parse_idx(r[f"onset_{side}"]),
                        "end": parse_idx(r[f"end_{side}"]),
                        "source": f"verified transfer {r['transfer_id']}",
                    }
                )
    if kind == "wide":
        candidates = pl.read_parquet(CANDIDATES_685).filter(
            pl.col("time_series_id").is_in(list(ids.values()))
        )
        rows.extend(
            {
                "time_series_id": r["time_series_id"],
                "start": idx_of(r["onset"]),
                "end": idx_of(r["end"] or PERIOD_END),
                "source": "candidate event of experiment 685",
            }
            for r in candidates.iter_rows(named=True)
        )
    spans = pl.DataFrame(rows, schema_overrides={"time_series_id": pl.Int64})
    return spans.filter((pl.col("end") > first) & (pl.col("start") < last)).filter(
        pl.col("end") > pl.col("start")
    )


def event_mask(*, spans: pl.DataFrame, series: np.ndarray, vidx: np.ndarray) -> np.ndarray:
    """Return True for the rows whose valid time lies in a span of the same series."""
    inside = np.zeros(len(vidx), dtype=bool)
    for r in spans.iter_rows(named=True):
        inside |= (series == r["time_series_id"]) & (vidx >= r["start"]) & (vidx < r["end"])
    return inside


def load_rows() -> dict[str, np.ndarray]:
    """Return the scored rows of the normal-running Primaries as arrays in percent of capacity.

    Keys: `series` (time series id), `vidx`, `day` (origin day), `month` (origin month, string),
    `band`, `lead_h`, `y`, `stored` and `scaled` (both shape (rows, 11)), and `nonneg`.
    """
    stored = pl.read_parquet(lib.FOLDER / "primaries.parquet").filter(
        pl.col("NRA_or_prevailing") == "NRA"
    )
    scaled = pl.read_parquet(DATA_683 / "predictions" / "arm2" / "primaries_NRA.parquet")
    key = ["time_series_id", "valid_time", "power_fcst_init_time"]
    if not stored.select(key).equals(scaled.select(key)):
        msg = "experiment 683's arm 2 rows are not in the order of the stored file"
        raise ValueError(msg)
    ids = sorted(stored["time_series_id"].unique().to_list())
    index = {t: i for i, t in enumerate(ids)}
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    capacity = capacity.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")
    cap = capacity["capacity"].to_numpy().astype(np.float64)
    nonneg = np.array([u == "MVA" for u in capacity["power_unit"].to_list()])
    grid = cal.load_truth(name="primaries", ids=ids)
    row = stored["time_series_id"].replace_strict(index).to_numpy()
    vidx = lib.grid_index(times=stored["valid_time"])
    oidx = lib.grid_index(times=stored["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        pad = int(vidx.max()) + 1 - grid.shape[1]
        grid = np.pad(grid, ((0, 0), (0, pad)), constant_values=np.nan)
    lead_h = (vidx - oidx) / 2.0
    scale = 100.0 / cap[row]
    y = grid[row, vidx] * scale
    arrays = {
        "stored": stored.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None],
        "scaled": scaled.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None],
    }
    ok = np.isfinite(y) & np.isfinite(arrays["stored"]).all(axis=1)
    month = stored["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    keep = ok & (month >= FIRST_MONTH) & (month <= LAST_MONTH)
    return {
        "series": np.asarray(ids)[row][keep],
        "vidx": vidx[keep],
        "day": (oidx // 48)[keep],
        "month": month[keep],
        "band": cal.band_index(lead_h=lead_h)[keep],
        "lead_h": lead_h[keep],
        "y": y[keep],
        "stored": arrays["stored"][keep],
        "scaled": arrays["scaled"][keep],
        "nonneg": np.asarray(nonneg)[row][keep],
        "inwin": (vidx < idx_of(PERIOD_END))[keep],
    }


def shift_by_group(*, rows: dict[str, np.ndarray], use: np.ndarray, per_band: bool) -> np.ndarray:
    """Return the median of (observed - q50) of each row's group, computed on the rows in `use`.

    The group is the series and origin month, and also the lead band when `per_band` is set.
    `rows["resid"]` holds observed minus q50 of the quantile set being shifted. A group with no
    row in `use` gets a shift of zero.
    """
    month = month_codes(rows["month"])
    key = rows["series"].astype(np.int64) * 10_000 + month * 10
    if per_band:
        key = key + rows["band"]
    _, codes = np.unique(key, return_inverse=True)
    med = (
        pl.DataFrame({"c": codes[use], "r": rows["resid"][use]})
        .group_by("c")
        .agg(pl.col("r").median())
    )
    shift = np.zeros(int(codes.max()) + 1)
    shift[med["c"].to_numpy()] = med["r"].to_numpy()
    return shift[codes]


def month_codes(month: np.ndarray) -> np.ndarray:
    """Return an integer for each `YYYY-MM` string: its rank among the months present."""
    return np.unique(month, return_inverse=True)[1].astype(np.int64)


def stage_quantiles(
    *, rows: dict[str, np.ndarray], which: str, stage: str, evt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the quantiles of one stage and the mask of the rows that stage scores."""
    q = rows[which]
    use = rows["inwin"] & (~evt if stage.startswith("events") else True)
    if "shift" not in stage:
        return q, use
    rows["resid"] = rows["y"] - q[:, 5]
    d = shift_by_group(rows=rows, use=use, per_band=stage.endswith("per band"))
    out = q + d[:, None]
    out = np.where(rows["nonneg"][:, None], np.maximum(out, 0.0), out)
    return np.sort(out, axis=1), use


def daily_table(*, rows: dict[str, np.ndarray], evt: np.ndarray) -> pl.DataFrame:
    """Return per set, stage, series, origin day, and lead band the counts of the three outcomes."""
    parts: list[pl.DataFrame] = []
    for which in SETS:
        for stage in STAGES:
            q, use = stage_quantiles(rows=rows, which=which, stage=stage, evt=evt)
            y = rows["y"]
            below, above = y < q[:, 1], y > q[:, 9]
            frame = pl.DataFrame(
                {
                    "series": rows["series"],
                    "day": rows["day"],
                    "month": rows["month"],
                    "band": rows["band"],
                    "cover": (~below & ~above).astype(np.float64),
                    "below": below.astype(np.float64),
                    "above": above.astype(np.float64),
                    "use": use,
                }
            ).filter(pl.col("use"))
            parts.append(
                frame.group_by("series", "day", "month", "band")
                .agg(
                    pl.len().alias("n"),
                    pl.col("cover").sum(),
                    pl.col("below").sum(),
                    pl.col("above").sum(),
                )
                .with_columns(set=pl.lit(which), stage=pl.lit(stage))
            )
    return pl.concat(parts)


class Boot:
    """Day-block bootstrap of pooled coverage, with picks shared across every call."""

    def __init__(self, *, daily: pl.DataFrame, seed: int = 688) -> None:
        """Keep the per-day counts and the seed of the picks."""
        self.daily = daily
        self.seed = seed
        self._picks: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}

    def _days(self, months: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        if months not in self._picks:
            days = (
                self.daily.filter(pl.col("month").is_in(months))["day"].unique().sort().to_numpy()
            )
            rng = np.random.default_rng(self.seed + sum(map(hash_month, months)))
            self._picks[months] = (days, rng.integers(0, len(days), size=(N_BOOT, len(days))))
        return self._picks[months]

    def cell(
        self,
        *,
        which: str,
        stage: str,
        months: tuple[str, ...],
        series: int | None = None,
        band: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the point value (index 0) and the 2000 draws of each outcome share, in percent."""
        days, pick = self._days(months)
        sub = self.daily.filter(
            (pl.col("set") == which) & (pl.col("stage") == stage) & pl.col("month").is_in(months)
        )
        if series is not None:
            sub = sub.filter(pl.col("series") == series)
        if band is not None:
            sub = sub.filter(pl.col("band") == band)
        per_day = sub.group_by("day").agg(
            pl.col("n").sum(), pl.col("cover").sum(), pl.col("below").sum(), pl.col("above").sum()
        )
        index = {int(d): i for i, d in enumerate(days)}
        arrays = {k: np.zeros(len(days)) for k in ("n", "cover", "below", "above")}
        for r in per_day.iter_rows(named=True):
            for k, values in arrays.items():
                values[index[r["day"]]] = r[k]
        n_all = arrays["n"].sum()
        out: dict[str, np.ndarray] = {"n": np.array([n_all])}
        for k in ("cover", "below", "above"):
            point = 100.0 * arrays[k].sum() / max(n_all, 1.0)
            draws_n = arrays["n"][pick].sum(axis=1)
            draws = 100.0 * arrays[k][pick].sum(axis=1) / np.maximum(draws_n, 1.0)
            out[k] = np.concatenate([[point], draws])
        return out


def hash_month(month: str) -> int:
    """Return a stable integer for a month string."""
    return int(month[:4]) * 12 + int(month[5:])


def interval(draws: np.ndarray) -> tuple[float, float, float]:
    """Return the point value and the 95% interval of a `Boot.cell` array."""
    lo, hi = np.percentile(draws[1:], [2.5, 97.5])
    return float(draws[0]), float(lo), float(hi)
