# ruff: noqa: E501
"""Experiment 459: is the sum over a load-shift donor and its recipients unchanged by the event?

Nothing is fitted. The script reads the cleaned half-hourly power of the 16 Primaries, measures each
labelled event's step at the donor, at the donor plus its labelled recipients (G1), at G1 plus the
donor's graph neighbours (G2), and at all 16 Primaries (G3), and repeats the measurement on placebo
windows outside every labelled event. It then measures how stable each member's share of a group
sum is and how well sum times the trailing 30-day median share predicts a member.

```bash
PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00391-xgboost-lag-surprise-second-stage DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00459-*/e00459_conservation.py
```
"""

import json
import re
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00391_neighbours as neighbours  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00459-load-shift-conservation-check-donor-and-recipient-sum"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
LABELS: Final[Path] = (
    Path(__file__).parents[1]
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "sw_block_detector_label_table_corrected.csv"
)
R0_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00402-lag-gate-learned-weight-injected-load-shifts-part-1"
    / "work"
    / "arm_preds"
    / "r0_lagged"
)
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
HALF_HOURS_PER_DAY: Final[int] = 48
BASELINE_DAYS: Final[int] = 7
EDGE_HALF_HOURS: Final[int] = 4
"""The 2 hours cut from each end of an event's interior."""
SHORT_EVENT_HALF_HOURS: Final[int] = 12
"""Events under 6 hours use the whole span."""
MIN_DONOR_STEP_MW: Final[float] = 0.3
PLACEBOS_PER_EVENT: Final[int] = 20
LEAK_THRESHOLD: Final[float] = 0.25
NO_END_SPAN_DAYS: Final[int] = 7
"""A labelled onset with no end is treated as occupying this many days when placing placebos."""
MAX_LEAD_HOURS: Final[float] = 240.0
"""The protocol's horizon of 1 to 10 days."""
TRAILING_DAYS: Final[int] = 30
SEED: Final[int] = 459


def metering_group(*, donor: str) -> str:
    """Return `six`, `eight`, or `other` for the donor's metering group."""
    return "six" if donor in SIX else "eight" if donor in EIGHT else "other"


class Grid:
    """Cleaned power of the 16 Primaries on a dense half-hourly grid (NaN where missing)."""

    def __init__(self) -> None:
        """Read and grid the cleaned power of the 16 Primaries."""
        series = adapter.series_table()
        self.labels: list[str] = series["label"].to_list()
        self.capacity = dict(zip(series["label"], series["capacity_mw"], strict=True))
        ids = dict(zip(series["time_series_id"], series["label"], strict=True))
        power = (
            scan_power()
            .filter(pl.col("time_series_id").is_in(list(ids)))
            .select("time_series_id", "time", "power")
            .collect()
        )
        self.t0 = power["time"].min().replace(minute=0, second=0, microsecond=0)  # ty: ignore[unresolved-attribute]  # ty: ignore[missing-argument, no-matching-overload, unknown-argument]
        t_max = power["time"].max()
        self.n = int((t_max - self.t0).total_seconds() // 1800) + 1  # ty: ignore[unsupported-operator]
        self.x = np.full((self.n, len(self.labels)), np.nan)
        for column, label in enumerate(self.labels):
            one = power.filter(
                pl.col("time_series_id") == next(i for i, s in ids.items() if s == label)
            )
            position = ((one["time"] - self.t0).dt.total_seconds() // 1800).to_numpy().astype(int)
            self.x[position, column] = one["power"].to_numpy()
        self.column = {label: c for c, label in enumerate(self.labels)}

    def index(self, *, stamp: str) -> int:
        """Return the grid index of a `YYYY-MM-DD HH:MM` UTC stamp."""
        when = pl.Series([stamp]).str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC")[0]
        return int((when - self.t0).total_seconds() // 1800)


def deviation(*, grid: Grid, members: list[str], onset: int, stop: int) -> np.ndarray:
    """Return each member's actual minus its 7-day half-hour-of-day median baseline.

    Args:
        grid: The power grid.
        members: Primary labels.
        onset: Index of the window's first half-hour; the baseline is the 7 days before it.
        stop: Index one past the window's last half-hour.

    Returns:
        An array of shape `(stop - onset, len(members))`, NaN where a reading or baseline is missing.
    """
    columns = [grid.column[m] for m in members]
    before = grid.x[onset - BASELINE_DAYS * HALF_HOURS_PER_DAY : onset][:, columns]
    by_slot = before.reshape(BASELINE_DAYS, HALF_HOURS_PER_DAY, len(columns))
    with np.errstate(all="ignore"):
        valid = np.isfinite(by_slot).sum(axis=0) >= 3
        baseline = np.where(
            valid, np.nanmedian(np.where(np.isfinite(by_slot), by_slot, np.nan), axis=0), np.nan
        )
    slots = (np.arange(stop - onset)) % HALF_HOURS_PER_DAY
    return grid.x[onset:stop][:, columns] - baseline[slots]


def interior(*, onset: int, end: int) -> tuple[int, int]:
    """Return the interior's first index and one-past-last index."""
    if end - onset < SHORT_EVENT_HALF_HOURS:
        return onset, end
    return onset + EDGE_HALF_HOURS, end - EDGE_HALF_HOURS


def missing_share(*, grid: Grid, members: list[str], onset: int, end: int) -> float:
    """Return the share of interior half-hours where any member has no reading."""
    lo, hi = interior(onset=onset, end=end)
    block = grid.x[lo:hi][:, [grid.column[m] for m in members]]
    return float((~np.isfinite(block).all(axis=1)).mean())


def steps(*, grid: Grid, members: list[str], onset: int, end: int) -> dict[str, float]:
    """Return the median interior deviation of each member and of their sum (key `sum`)."""
    dev = deviation(grid=grid, members=members, onset=onset, stop=end)
    lo, hi = interior(onset=onset, end=end)
    inner = dev[lo - onset : hi - onset]
    out = {}
    for c, m in enumerate(members):
        col = inner[:, c]
        out[m] = float(np.nanmedian(col)) if np.isfinite(col).mean() >= 0.5 else np.nan
    total = inner.sum(axis=1)
    out["sum"] = float(np.nanmedian(total)) if np.isfinite(total).mean() >= 0.5 else np.nan
    return out


def read_events(*, grid: Grid, graph: dict[str, tuple[str, ...]]) -> list[dict]:
    """Return the labelled events with indices, donor, recipients, and the three groups."""
    import csv

    events = []
    with LABELS.open() as handle:
        for number, row in enumerate(csv.DictReader(handle)):
            named = list(dict.fromkeys(re.findall(r"D\d+", row["pair"])))
            donor = row["first_primary"]
            recipients = [n for n in named if n != donor]
            in16 = [r for r in recipients if r in grid.column]
            if not row["end_utc"] or donor not in grid.column:
                events.append({"row": number, "usable": False, "onset_utc": row["onset_utc"]})
                continue
            g1 = [donor, *in16]
            g2 = list(dict.fromkeys([*g1, *graph.get(donor, ())]))
            events.append(
                {
                    "row": number,
                    "usable": True,
                    "onset_utc": row["onset_utc"],
                    "end_utc": row["end_utc"],
                    "onset": grid.index(stamp=row["onset_utc"]),
                    "end": grid.index(stamp=row["end_utc"]),
                    "donor": donor,
                    "named_recipients": recipients,
                    "observed": len(in16) > 0,
                    "recipient_kind": "observed"
                    if in16
                    else ("outside the 16" if recipients else "none named"),
                    "mixed_units_g1": len({m in SIX for m in g1}) > 1,
                    "mixed_units_g2": len({m in SIX for m in g2}) > 1,
                    "g1": g1,
                    "g2": g2,
                    "g3": list(grid.labels),
                    "sign": int(row["corrected_sign_first"]),
                    "label_depth_mw": float(row["corrected_depth_first_mw"]),
                    "group": metering_group(donor=donor),
                }
            )
    return events


def forbidden_mask(*, grid: Grid, events: list[dict]) -> np.ndarray:
    """Return a mask of half-hours inside a labelled event or in the 7 days after one."""
    mask = np.zeros(grid.n, dtype=bool)
    tail = NO_END_SPAN_DAYS * HALF_HOURS_PER_DAY
    for e in events:
        onset = grid.index(stamp=e["onset_utc"])
        end = e["end"] if e["usable"] else onset + tail
        mask[max(onset, 0) : min(end + tail, grid.n)] = True
    return mask


def baseline_overlap(*, grid: Grid, events: list[dict], event: dict) -> bool:
    """Return whether the event's 7-day baseline overlaps another labelled event or its 7-day tail."""
    tail = NO_END_SPAN_DAYS * HALF_HOURS_PER_DAY
    lo = event["onset"] - BASELINE_DAYS * HALF_HOURS_PER_DAY
    for other in events:
        if other["row"] == event["row"]:
            continue
        start = grid.index(stamp=other["onset_utc"])
        end = other["end"] if other["usable"] else start + tail
        if start < event["onset"] and end + tail > lo:
            return True
    return False


def run_events(*, grid: Grid, events: list[dict], forbidden: np.ndarray) -> pl.DataFrame:
    """Measure every usable event and its placebos; return one row per event and placebo draw."""
    rng = np.random.default_rng(SEED)
    first_ok = BASELINE_DAYS * HALF_HOURS_PER_DAY + 1
    rows = []
    for e in events:
        if not e["usable"]:
            continue
        length = e["end"] - e["onset"]
        donor_step_event = None
        measured = {}
        for gname in ("g1", "g2", "g3"):
            s = steps(grid=grid, members=e[gname], onset=e["onset"], end=e["end"])
            measured[gname] = s
            if donor_step_event is None:
                donor_step_event = s[e["donor"]]
        base = {
            "row": e["row"], "kind": "event", "draw": -1, "donor": e["donor"], "group": e["group"],
            "observed": e["observed"], "recipient_kind": e["recipient_kind"],
            "mixed_units_g1": e["mixed_units_g1"], "mixed_units_g2": e["mixed_units_g2"],
            "baseline_overlaps_other_event": baseline_overlap(grid=grid, events=events, event=e),
            "onset_utc": e["onset_utc"], "end_utc": e["end_utc"],
            "sign": e["sign"], "label_depth_mw": e["label_depth_mw"],
            "recipients": ",".join(e["named_recipients"]),
            "g1_members": ",".join(e["g1"]), "g2_members": ",".join(e["g2"]),
            "donor_step_mw": donor_step_event,
            "recipient_steps_mw": ";".join(f"{m}={measured['g1'][m]:.3f}" for m in e["g1"][1:]),
        }  # fmt: skip
        rows.append(
            base
            | {f"{g}_step_mw": measured[g]["sum"] for g in ("g1", "g2", "g3")}
            | {
                "g1_missing_share": missing_share(
                    grid=grid, members=e["g1"], onset=e["onset"], end=e["end"]
                )
            }
        )
        found, attempts = 0, 0
        while found < PLACEBOS_PER_EVENT and attempts < 2_000:
            attempts += 1
            onset = int(rng.integers(first_ok, grid.n - length - 1))
            lo = onset - BASELINE_DAYS * HALF_HOURS_PER_DAY
            if forbidden[lo : onset + length].any():
                continue
            s = {
                g: steps(grid=grid, members=e[g], onset=onset, end=onset + length)
                for g in ("g1", "g2", "g3")
            }
            if not np.isfinite(s["g1"]["sum"]) or not np.isfinite(s["g1"][e["donor"]]):
                continue
            rows.append(
                base
                | {
                    "kind": "placebo",
                    "draw": found,
                    "baseline_overlaps_other_event": False,
                    "onset_utc": (grid.t0 + np.timedelta64(onset * 30, "m")).strftime(  # ty: ignore[unsupported-operator]
                        "%Y-%m-%d %H:%M"
                    ),
                    "end_utc": (grid.t0 + np.timedelta64((onset + length) * 30, "m")).strftime(  # ty: ignore[unsupported-operator]
                        "%Y-%m-%d %H:%M"
                    ),
                }
                | {f"{g}_step_mw": s[g]["sum"] for g in ("g1", "g2", "g3")}
                | {"placebo_donor_step_mw": s["g1"][e["donor"]]}
                | {
                    "g1_missing_share": missing_share(
                        grid=grid, members=e["g1"], onset=onset, end=onset + length
                    )
                }
            )
            found += 1
    frame = pl.DataFrame(rows, infer_schema_length=None)
    abs_donor = pl.col("donor_step_mw").abs()
    return frame.with_columns(
        *[(pl.col(f"{g}_step_mw").abs() / abs_donor).alias(f"{g}_leak") for g in ("g1", "g2", "g3")]
    )


def event_status(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the exclusion reason to each row, copied from its event."""
    ev = frame.filter(pl.col("kind") == "event").select(
        "row",
        reason=pl.when(pl.col("donor_step_mw").is_null() | pl.col("donor_step_mw").is_nan())
        .then(pl.lit("no donor step"))
        .when(pl.col("donor_step_mw").abs() < MIN_DONOR_STEP_MW)
        .then(pl.lit("donor step under 0.3 MW"))
        .when(pl.col("donor_step_mw").sign() != pl.col("sign"))
        .then(pl.lit("sign disagrees"))
        .otherwise(pl.lit("")),
    )
    return frame.join(ev, on="row")


def trailing_share(*, share: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Return, for each day-start, the median share over the 30 days before it (NaN if sparse)."""
    out = np.full(len(share) // HALF_HOURS_PER_DAY + 1, np.nan)
    window = TRAILING_DAYS * HALF_HOURS_PER_DAY
    for day in range(TRAILING_DAYS, len(out)):
        stop = day * HALF_HOURS_PER_DAY
        chunk = share[stop - window : stop][usable[stop - window : stop]]
        if chunk.size >= window // 3:
            out[day] = np.median(chunk)
    return out


def r0_rows(*, grid: Grid) -> pl.DataFrame:
    """Return experiment 392's R0 lagged forecast rows (stored by experiment 402) in MW."""
    frames = []
    for path in sorted(R0_DIR.glob("*.parquet")):
        label = path.name.split("_")[0]
        if label not in grid.column:
            continue
        frames.append(
            pl.read_parquet(
                path, columns=["power_fcst_init_time", "valid_time", "q50"]
            ).with_columns(label=pl.lit(label))
        )
    return (
        pl.concat(frames)
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds()
            / 3600
        )
        .filter(pl.col("lead_hours") <= MAX_LEAD_HOURS)
    )


def span_mask(*, grid: Grid, events: list[dict], member: str) -> np.ndarray:
    """Return a mask of half-hours inside a labelled event that names the member (no tail)."""
    mask = np.zeros(grid.n, dtype=bool)
    for e in events:
        if e["usable"] and member in e["g1"]:
            mask[e["onset"] : e["end"]] = True
    return mask


def harm_flags(*, grid: Grid, rows: pl.DataFrame, spans: np.ndarray) -> pl.Series:
    """Flag forecast rows whose target is outside every event but whose lags touch one.

    The lags are the readings 1, 2, and 7 days before the forecast origin's day, at the target's
    time of day (experiment 402's harm-row definition, applied here to the member's own events).
    """
    target = ((rows["valid_time"] - grid.t0).dt.total_seconds() // 1800).to_numpy().astype(int)
    origin = (
        ((rows["power_fcst_init_time"] - grid.t0).dt.total_seconds() // 1800).to_numpy().astype(int)
    )
    day_start = origin - origin % HALF_HOURS_PER_DAY
    touched = np.zeros(len(target), dtype=bool)
    for days in (1, 2, 7):
        lag = np.clip(
            day_start - days * HALF_HOURS_PER_DAY + target % HALF_HOURS_PER_DAY, 0, grid.n - 1
        )
        touched |= spans[lag]
    return pl.Series(touched & ~spans[np.clip(target, 0, grid.n - 1)])


def allocation(
    *, grid: Grid, events: list[dict], forbidden: np.ndarray, r0: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-(group, member) share stability and the monthly share table."""
    groups = {tuple(sorted(e["g1"])) for e in events if e["usable"] and e["observed"]}
    groups |= {tuple(sorted(e["g2"])) for e in events if e["usable"] and not e["observed"]}
    stamps = pl.datetime_range(  # ty: ignore[no-matching-overload]
        grid.t0,
        grid.t0 + np.timedelta64((grid.n - 1) * 30, "m"),  # ty: ignore[unsupported-operator]
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    stats, monthly = [], []
    for members in sorted(groups):
        columns = [grid.column[m] for m in members]
        block = grid.x[:, columns]
        total = block.sum(axis=1)
        floor = 0.1 * sum(grid.capacity[m] for m in members)
        ok = np.isfinite(total) & (total > floor)
        for c, m in enumerate(members):
            share = np.where(ok, block[:, c] / np.where(ok, total, 1), np.nan)
            usable = ok & ~forbidden
            med = trailing_share(share=share, usable=usable)
            share_med = np.repeat(med, HALF_HOURS_PER_DAY)[: grid.n]
            predicted = total * share_med
            err_mw = block[:, c] - predicted
            valid = ok & np.isfinite(err_mw) & ~forbidden
            cap = grid.capacity[m]
            pts = err_mw[valid] / cap * 100
            share_err_points = (share - share_med)[valid] * total[valid] / cap * 100
            one = pl.DataFrame(
                {"valid_time": stamps.filter(pl.Series(valid)), "oracle_abs_pts": np.abs(pts)}
            )
            valid_any = ok & np.isfinite(err_mw)
            one_any = pl.DataFrame(
                {
                    "valid_time": stamps.filter(pl.Series(valid_any)),
                    "oracle_any_abs_pts": np.abs(err_mw[valid_any]) / cap * 100,
                }
            )
            spans = span_mask(grid=grid, events=events, member=m)
            actual = pl.DataFrame({"valid_time": stamps, "actual": block[:, c]})
            harm = (
                r0.filter(pl.col("label") == m)
                .join(actual, on="valid_time")
                .join(one_any, on="valid_time")
                .with_columns(
                    r0_abs_pts=((pl.col("q50") * cap - pl.col("actual")).abs() / cap * 100)
                )
            )
            harm = harm.filter(harm_flags(grid=grid, rows=harm, spans=spans))
            ref = (
                r0.filter(pl.col("label") == m)
                .join(actual, on="valid_time")
                .join(one, on="valid_time")
                .with_columns(
                    r0_abs_pts=((pl.col("q50") * cap - pl.col("actual")).abs() / cap * 100)
                )
            )
            share_ok = share[usable & np.isfinite(share_med)]
            stats.append(
                {
                    "group": "+".join(members),
                    "member": m,
                    "metering": metering_group(donor=m),
                    "n_halfhours": int(valid.sum()),
                    "share_min": float(np.nanmin(share_ok)) if share_ok.size else np.nan,
                    "share_max": float(np.nanmax(share_ok)) if share_ok.size else np.nan,
                    "share_median": float(np.nanmedian(share_ok)) if share_ok.size else np.nan,
                    "share_err_p10_pts": float(np.percentile(share_err_points, 10))
                    if pts.size
                    else np.nan,
                    "share_err_p90_pts": float(np.percentile(share_err_points, 90))
                    if pts.size
                    else np.nan,
                    "oracle_median_abs_pts_all": float(np.median(np.abs(pts)))
                    if pts.size
                    else np.nan,
                    "harm_rows": harm.height,
                    "harm_oracle_median_abs_pts": float(harm["oracle_any_abs_pts"].median())  # ty: ignore[invalid-argument-type]
                    if harm.height
                    else np.nan,
                    "harm_r0_median_abs_pts": float(harm["r0_abs_pts"].median())  # ty: ignore[invalid-argument-type]
                    if harm.height
                    else np.nan,
                    "harm_paired_gain_median_pts": float(
                        (harm["r0_abs_pts"] - harm["oracle_any_abs_pts"]).median()  # ty: ignore[invalid-argument-type]
                    )
                    if harm.height
                    else np.nan,
                    "r0_rows": ref.height,
                    "oracle_median_abs_pts_on_r0_rows": float(ref["oracle_abs_pts"].median())  # ty: ignore[invalid-argument-type]
                    if ref.height
                    else np.nan,
                    "r0_median_abs_pts": float(ref["r0_abs_pts"].median())  # ty: ignore[invalid-argument-type]
                    if ref.height
                    else np.nan,
                }
            )
            if m in SIX:
                month = pl.DataFrame(
                    {"valid_time": stamps, "share": share, "usable": usable}
                ).filter(pl.col("usable"))
                monthly.append(
                    month.group_by(pl.col("valid_time").dt.month().alias("month"))
                    .agg(pl.col("share").median())
                    .with_columns(group=pl.lit("+".join(members)), member=pl.lit(m))
                )
    return pl.DataFrame(stats), pl.concat(monthly) if monthly else pl.DataFrame()


def main() -> None:
    """Run the whole check and write the tables."""
    WORK.mkdir(parents=True, exist_ok=True)
    grid = Grid()
    graph = {k: tuple(v) for k, v in neighbours.nged_neighbours()[0].items()}
    (WORK / "neighbours_nged.json").write_text(json.dumps(graph, indent=1))
    events = read_events(grid=grid, graph=graph)
    forbidden = forbidden_mask(grid=grid, events=events)
    print("grid half-hours", grid.n, "forbidden share", round(float(forbidden.mean()), 3))
    frame = event_status(frame=run_events(grid=grid, events=events, forbidden=forbidden))
    frame.write_parquet(WORK / "event_and_placebo_rows.parquet")
    print(
        "events",
        frame.filter(pl.col("kind") == "event").height,
        "placebo rows",
        frame.filter(pl.col("kind") == "placebo").height,
    )
    stats, monthly = allocation(
        grid=grid, events=events, forbidden=forbidden, r0=r0_rows(grid=grid)
    )
    stats.write_parquet(WORK / "allocation.parquet")
    monthly.write_parquet(WORK / "monthly_share.parquet")
    print("allocation rows", stats.height)


if __name__ == "__main__":
    main()
