"""Run E1-J-sym on the residual panel extended to 2026-10-31, and write the candidate event table.

Method: experiment 680's E1-J-sym on its real record, with no parameter changed. Three choices
keep the earlier events identical to experiment 680's, so that the window is the only difference:

- The all-Primaries median is removed as in experiment 680, but each Primary's scale (1.4826 times
  the median absolute deviation of its residual) is fixed on the first 78,816 half-hours (to the
  end of 2026-06-30), not on the longer record.
- The noise units and the slow-level penalties are frozen on the residual before 2026-07-01
  (on arrays cut at 78,816 half-hours), as experiment 680 froze them on the whole of its record.
- The slow level is refitted on the whole panel, as in experiment 680.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      env OMP_NUM_THREADS=1 uv run --no-sync python $P/00685-*/e00685_run.py
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00653_common as c653  # isort: skip
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OLD_LENGTH: Final[int] = 78816
NEW_LENGTH: Final[int] = 84720
EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00685-event-table-on-the-final-test-window-all-primaries"
)
E327_EXTENDED: Final[Path] = EXPERIMENT_DIR / "e327_extended"
E680_REAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
    / "real_events_E1-J-sym.parquet"
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
WINDOW_FIRST_INDEX: Final[int] = OLD_LENGTH
"""Half-hour index of 2026-07-01 00:00, the first half-hour of the final-test window."""

# The 653 modules read GRID_LENGTH and N_BLOCKS at import time, so move them first.
c653.GRID_LENGTH = NEW_LENGTH
c653.N_BLOCKS = NEW_LENGTH // c653.BLOCK
c653.E327 = E327_EXTENDED

from e00653_core import decode_world, freeze  # noqa: E402  # isort: skip
from e00653_segment import Event, Params  # noqa: E402  # isort: skip
from e00653_slow import to_half_hours  # noqa: E402  # isort: skip
from e00680_search import search_both_ends  # noqa: E402  # isort: skip


class ExtendedPanel(c653.Panel):
    """Experiment 653's panel on the extended grid, with scales fixed on the first 78,816 steps."""

    @staticmethod
    def _scale(x: np.ndarray) -> float:
        head = x[:OLD_LENGTH]
        d = np.nanmedian(np.abs(head - np.nanmedian(head)))
        return float(1.4826 * d) or 1.0


def write_extended_power() -> None:
    """Pad experiment 327's stored power with the window's cleaned power from `inputs/rows.parquet`."""
    rows = pl.read_parquet(
        EXPERIMENT_DIR / "inputs" / "rows.parquet",
        columns=["time_series_id", "time", "power", "present"],
    ).filter(pl.col("time_series_id").is_not_null() & pl.col("present"))
    out = E327_EXTENDED / "power"
    out.mkdir(parents=True, exist_ok=True)
    stored_dir = c653.ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
    for s in c653.SERIES_IDS:
        old = np.load(stored_dir / "power" / f"power_{s}.npy")
        if len(old) != OLD_LENGTH:
            msg = f"stored power for series {s} has {len(old)} half-hours"
            raise ValueError(msg)
        new = np.full(NEW_LENGTH, np.nan)
        new[:OLD_LENGTH] = old
        sub = rows.filter(pl.col("time_series_id") == s)
        idx = (
            (sub["time"].dt.replace_time_zone(None) - datetime(2022, 1, 1)).dt.total_minutes() // 30
        ).to_numpy()
        keep = (idx >= OLD_LENGTH) & (idx < NEW_LENGTH)
        new[idx[keep]] = sub["power"].to_numpy()[keep]
        np.save(out / f"power_{s}.npy", new)
        print(s, "window half-hours with power", int(np.isfinite(new[OLD_LENGTH:]).sum()))


def when(*, index: int) -> datetime:
    """UTC end time of a half-hour index."""
    return GRID_START + timedelta(minutes=30 * index)


def candidate_table(*, events: list[Event], sites_to_id: dict[str, int]) -> pl.DataFrame:
    """One row per decoded event, with the columns the brief asks for."""
    rows = []
    for e in events:
        if e.units < 2.0 or e.cls in ("routine", "ramp"):
            continue
        is_open = e.end is None
        partners = list(e.partners)
        involved = [e.primary, *partners]
        rows.append(
            {
                "primaries": ",".join(involved),
                "time_series_ids": ",".join(str(sites_to_id[p]) for p in involved),
                "primary": e.primary,
                "time_series_id": sites_to_id[e.primary],
                "class": "edge transfer" if e.cls.endswith("transfer") else "solo shift",
                "cls_raw": e.cls,
                "onset": when(index=e.start),
                "end": None if is_open else when(index=e.end),
                "open": is_open,
                "sign": "rise" if e.amp_mw > 0 else "fall",
                "depth_mw": e.amp_mw,
                "depth_noise_units": e.units,
                "confidence": e.confidence,
                "starts_before_window": e.start < WINDOW_FIRST_INDEX,
                "ends_after_window": is_open or e.end >= NEW_LENGTH,
                "in_tuning_years": e.start < c653.SCORED_FROM,
            }
        )
    return pl.DataFrame(rows)


def md(df: pl.DataFrame) -> str:
    """A markdown table of a small frame."""
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join(
        "| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in row) + " |\n"
        for row in df.rows()
    )
    return head + body


def write_summary(*, table: pl.DataFrame, out: Path, gate: dict) -> None:
    """SUMMARY.md: priority rows first, then counts per Primary and per month (window only)."""
    win = table.filter(pl.col("onset") >= datetime(2026, 7, 1, tzinfo=UTC))
    cols = [
        "primaries",
        "class",
        "sign",
        "onset",
        "end",
        "depth_mw",
        "depth_noise_units",
        "confidence",
    ]

    def show(df: pl.DataFrame) -> str:
        if df.height == 0:
            return "None found.\n"
        return md(df.select(cols).sort("onset"))

    def months(df: pl.DataFrame, lo: int, hi: int) -> pl.DataFrame:
        return df.filter(pl.col("onset").dt.month().is_between(lo, hi))

    transfer = win.filter(pl.col("class") == "edge transfer")
    pair = transfer.filter(
        pl.col("time_series_ids").str.split(",").list.sort().list.join(",") == "13,16"
    )
    lines = [
        "# Experiment 685 summary: candidate events on the final-test window",
        "",
        "Every row is a candidate, never a label: NGED's event labels are an incomplete dump. An",
        "edge transfer appears as two rows (the fall and the rise). Onset is the end time of the",
        "first affected half-hour (UTC). `time_series_id` 13 is D18, 16 is D14, 11 is D4, 18 is D16.",
        "",
        "## Priority 1: transfers between time_series_id 13 and 16, July and August 2026",
        show(months(pair, 7, 8)),
        "## Priority 2: any event at time_series_id 11, September 2026",
        show(months(win.filter(pl.col("time_series_id") == 11), 9, 9)),
        "## Priority 3: any event at time_series_id 18, October 2026",
        show(months(win.filter(pl.col("time_series_id") == 18), 10, 10)),
        "## Candidate events starting in the window, per Primary",
        md(
            win.group_by("primary", "time_series_id", "class").len().sort("time_series_id", "class")
        ),
        "",
        "## Candidate events starting in the window, per month",
        md(
            win.with_columns(month=pl.col("onset").dt.strftime("%Y-%m"))
            .group_by("month", "class")
            .len()
            .sort("month", "class")
        ),
        "",
        "## Reproduction gate against experiment 680 (events starting 2024-01-01 to 2026-06-30)",
        "```json",
        json.dumps(gate, indent=1),
        "```",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    """Decode the extended panel, check the earlier events against experiment 680, and write."""
    write_extended_power()
    p = json.loads((c653.WORK / "params.json").read_text())
    params = replace(Params(), **p)
    panel = ExtendedPanel()
    residual = panel.mask_common_mode(residual=panel.residual)
    truncated = os.environ.get("E685_TRUNCATE") == "1"
    if truncated:  # diagnostic: blank the window, which must reproduce experiment 680
        residual = {
            s: np.where(np.arange(NEW_LENGTH) < OLD_LENGTH, v, np.nan) for s, v in residual.items()
        }
    # Freeze on the pre-July residual with the old block count, so the noise units equal those of
    # experiment 680 (the slow-level design depends on the series length).
    c653.N_BLOCKS = OLD_LENGTH // c653.BLOCK
    try:
        frozen = freeze(
            panel=panel, residual={s: v[:OLD_LENGTH] for s, v in residual.items()}, upto=None
        )
    finally:
        c653.N_BLOCKS = NEW_LENGTH // c653.BLOCK
    events, slows = decode_world(
        panel=panel, residual=residual, frozen=frozen, arm="masked", params=params
    )
    xh = {s: residual[s] - to_half_hours(slow=slows[s]) for s in slows}
    table_events = search_both_ends(
        events=events, xh=xh, sigma=frozen.sigma, neighbours=panel.neighbours
    )
    info = pl.read_parquet(c653.SERIES_PATH).select("time_series_id", "site")
    sites_to_id = dict(zip(info["site"], info["time_series_id"], strict=True))
    table = candidate_table(events=table_events, sites_to_id=sites_to_id)
    out = EXPERIMENT_DIR / ("event_table_truncated_check" if truncated else "event_table")
    out.mkdir(exist_ok=True)
    table.sort("onset").write_parquet(out / "candidate_events.parquet")
    table.sort("onset").write_csv(out / "candidate_events.csv")

    # Reproduction gate: events that start before the window, in the scored years, against 680.
    stored = pl.read_parquet(E680_REAL).filter(pl.col("start") < WINDOW_FIRST_INDEX)
    mine = pl.DataFrame(
        [
            {
                "primary": e.primary,
                "start": e.start,
                "end": -1 if e.end is None else e.end,
                "amp_mw": e.amp_mw,
                "units": e.units,
                "cls": e.cls,
                "partners": ",".join(e.partners),
            }
            for e in table_events
            if c653.SCORED_FROM <= e.start < WINDOW_FIRST_INDEX
        ]
    )
    key = ["primary", "start", "cls", "partners"]
    both = stored.join(mine, on=key, how="inner", suffix="_new")
    only_stored = stored.join(mine, on=key, how="anti")
    only_mine = mine.join(stored, on=key, how="anti")
    gate = {
        "stored_events_before_window": stored.height,
        "matched_on_primary_start_class_partners": both.height,
        "only_in_stored": only_stored.height,
        "only_in_new": only_mine.height,
        "events_with_end_changed": both.filter(pl.col("end") != pl.col("end_new")).height,
    }
    (out / "reproduction_gate.json").write_text(json.dumps(gate, indent=1))
    only_stored.write_csv(out / "gate_only_in_stored.csv")
    only_mine.write_csv(out / "gate_only_in_new.csv")
    write_summary(table=table, out=out, gate=gate)
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
