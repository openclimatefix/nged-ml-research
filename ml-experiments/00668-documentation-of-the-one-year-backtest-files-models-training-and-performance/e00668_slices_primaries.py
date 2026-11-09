"""Score the Primaries file inside and around load-shift events and gate-on days.

PYTHONPATH not needed. Reads the per-row scores written by `e00668_score_demand.py primaries NRA`
and `... prevailing` (`scores_668/primaries_<kind>/rows.parquet`) and writes
`scores_668/primaries_slices/slices.csv` and `slice_events.csv` (private).

Four event sets, never mixed (the sets are also not nested):

- `long and deep`: events of experiment 653's event table (arm `real_events_masked`, as read by
  experiment 665) that last more than 7 days and step 8 noise units or more. None falls in the
  backtest year, so every slice of this set is empty.
- `event table, over 7 days, 2 units or more`: the same table, events that last more than 7 days and
  step 2 noise units or more (a descriptive proxy).
- `verified transfer`: experiment 580's verified D5 and D16 transfer, 2025-12-29 to 2026-01-13.
- `gate-on origin days`: origin days on which the G4 gate of experiment 592 is on, stored only for
  the five fold months of that experiment, beside the gate-off origin days of the same months.

For each set the slices are: rows inside an event, rows in the 7 days after the event ends, and
rows with no event and no event end in the previous 7 days ("outside"), for the same Primaries.
"""

from datetime import UTC, datetime

import e00668_score_lib as lib
import numpy as np
import polars as pl

R = lib.FOLDER.parent
SCORES = lib.FOLDER / "scores_668"
GRID_653 = datetime(2022, 1, 1, tzinfo=UTC)
SCORED_FROM_653 = 17520  # 2024-01-01 on the 653 grid, in half-hours
OFFSET_653 = int((GRID_653 - lib.GRID_START).total_seconds() // 1800)
DAY = 48
AFTER = 7 * DAY
VERIFIED = {
    "D5": (datetime(2025, 12, 29, tzinfo=UTC), datetime(2026, 1, 14, tzinfo=UTC)),
    "D16": (datetime(2025, 12, 29, tzinfo=UTC), datetime(2026, 1, 14, tzinfo=UTC)),
}
SYSTEMS = {
    "NRA": "Backtest, normal running",
    "prevailing": "Backtest, prevailing",
    "heur": "Manual heuristic",
    "clim": "Climatology",
}


def idx_of(when: datetime) -> int:
    """Return the half-hour grid index of an instant on the scoring grid."""
    return int((when - lib.GRID_START).total_seconds() // 1800)


def load_rows() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the joined per-row scores and the site map of the 16 Primaries."""
    site = pl.read_csv(next(R.glob("00521-*/site_map.csv")))
    out = None
    for kind in ("NRA", "prevailing"):
        r = pl.read_parquet(SCORES / f"primaries_{kind}" / "rows.parquet").filter(pl.col("ok"))
        keep = ["time_series_id", "valid_idx", "origin_idx", "lead_h"]
        r = r.select(
            *keep,
            pl.col("Model_crps").alias(f"{kind}_crps"),
            pl.col("Model_mae").alias(f"{kind}_mae"),
            *(
                [
                    pl.col("Manual heuristic_crps").alias("heur_crps"),
                    pl.col("Manual heuristic_mae").alias("heur_mae"),
                    pl.col("Climatology_crps").alias("clim_crps"),
                    pl.col("Climatology_mae").alias("clim_mae"),
                ]
                if kind == "NRA"
                else []
            ),
        )
        out = r if out is None else out.join(r, on=keep, how="inner")
    rows = out.join(site.select("time_series_id", "label", "primary"), on="time_series_id")
    return rows, site


def event_sets(*, site: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each event set as `label`, `start`, `end` (scoring-grid half-hours)."""
    table = pl.read_parquet(next(R.glob("00653-*/real_events_masked.parquet"))).filter(
        pl.col("start") >= SCORED_FROM_653
    )
    table = table.with_columns(
        label=pl.col("primary").replace_strict(
            dict(zip(site["label"].to_list(), site["label"].to_list(), strict=True)), default=None
        )
        if table["primary"].dtype == pl.String
        else pl.col("primary").replace_strict(
            dict(zip(site["primary"].to_list(), site["label"].to_list(), strict=True))
        ),
        start=pl.col("start") + OFFSET_653,
        end=pl.col("end") + OFFSET_653,
    ).with_columns(length=pl.col("end") - pl.col("start"))
    first, last = (
        idx_of(datetime(2025, 11, 1, tzinfo=UTC)),
        idx_of(datetime(2026, 11, 9, tzinfo=UTC)),
    )
    in_year = table.filter((pl.col("end") > first) & (pl.col("start") < last))
    sets = {
        "long and deep": in_year.filter((pl.col("length") > 7 * DAY) & (pl.col("units") >= 8)),
        "event table, over 7 days, 2 units or more": in_year.filter(
            (pl.col("length") > 7 * DAY) & (pl.col("units") >= 2)
        ),
        "verified transfer": pl.DataFrame(
            [
                {"label": k, "start": idx_of(a), "end": idx_of(b), "units": None}
                for k, (a, b) in VERIFIED.items()
            ]
        ),
    }
    return {k: v.select("label", "start", "end") for k, v in sets.items()}


def masks(*, rows: pl.DataFrame, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the boolean masks inside, after, and outside the events of one set."""
    inside = np.zeros(rows.height, dtype=bool)
    after = np.zeros(rows.height, dtype=bool)
    near_series = np.zeros(rows.height, dtype=bool)
    label = rows["label"].to_numpy()
    v = rows["valid_idx"].to_numpy()
    for e in events.iter_rows(named=True):
        own = label == e["label"]
        inside |= own & (v >= e["start"]) & (v < e["end"])
        after |= own & (v >= e["end"]) & (v < e["end"] + AFTER)
        near_series |= own
    return {
        "inside": inside,
        "7 days after": after & ~inside,
        "outside (same Primaries)": near_series & ~inside & ~after,
    }


def gate_masks(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the gate-on and gate-off masks over the origin days with a stored gate."""
    z = np.load(next(R.glob("00592-*/gate_state.npz")))
    on, gap = z["G4_on"], z["fresh_gap"]
    p = rows["primary"].to_numpy()
    day = rows["origin_idx"].to_numpy() // DAY
    day = np.minimum(day, on.shape[1] - 1)
    stored = np.isfinite(gap[p, day]) & (rows["origin_idx"].to_numpy() // DAY < on.shape[1])
    return {
        "gate-on origin days": stored & on[p, day],
        "gate-off origin days": stored & ~on[p, day],
    }


def main() -> None:
    """Write the slice tables."""
    rows, site = load_rows()
    day = (rows["origin_idx"].to_numpy() // DAY).astype(np.int64)
    sets = event_sets(site=site)
    slices: dict[tuple[str, str], np.ndarray] = {("all rows", "all"): np.ones(rows.height, bool)}
    counts: list[dict] = []
    for name, ev in sets.items():
        counts.append({"set": name, "events": ev.height, "series": ev["label"].n_unique()})
        for k, m in masks(rows=rows, events=ev).items():
            slices[(name, k)] = m
    for k, m in gate_masks(rows=rows).items():
        slices[("gate-on days of experiment 592 (G4)", k)] = m
    records = []
    for (name, part), mask in slices.items():
        n = int(mask.sum())
        for tag, system in SYSTEMS.items():
            for metric in ("crps", "mae"):
                v = rows[f"{tag}_{metric}"].to_numpy().astype(np.float64)
                if n == 0:
                    mean = lo = hi = float("nan")
                else:
                    draws = lib.day_bootstrap(day=day[mask], values={tag: v[mask]})[tag]
                    mean = float(v[mask].mean())
                    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
                records.append(
                    {
                        "set": name,
                        "part": part,
                        "system": system,
                        "metric": metric,
                        "mean": mean,
                        "lo": lo,
                        "hi": hi,
                        "rows": n,
                        "origin_days": len(np.unique(day[mask])) if n else 0,
                    }
                )
    out = SCORES / "primaries_slices"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_csv(out / "slices.csv")
    pl.DataFrame(counts).write_csv(out / "slice_events.csv")
    print(pl.DataFrame(counts), flush=True)


if __name__ == "__main__":
    main()
