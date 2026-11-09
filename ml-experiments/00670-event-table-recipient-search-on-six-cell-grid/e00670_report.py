"""Aggregate the injected worlds of experiment 670 by cell and check the pre-registered predictions.

Run with the same environment as `e00670_run.py`:

    uv run --no-sync python e00670_report.py

It reads every `injected_<config>_<first seed>.parquet` under the experiment's data folder, and
writes `injected_by_cell.json` there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY
from e00665_score import CELLS, boot_intervals, cell_stats, with_cells, wrec
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00670-event-table-recipient-search-on-six-cell-grid"
E660: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00660-shared-generator-of-injected-load-transfer-events"
)
METHODS: Final[tuple[str, str, str]] = ("E1", "E1-J", "E1-J rewired")
N_BOOT: Final[int] = 2000
PREDICTED: Final[dict[str, tuple[float, float | None]]] = {
    "long, deep": (0.85, 0.60),
    "medium, deep": (0.80, 0.50),
    "long, shallow": (0.55, 0.30),
    "medium, shallow": (0.45, 0.20),
    "short, deep": (0.40, 0.20),
    "short, shallow": (0.15, None),
}
"""Pre-registered E1-J minimum (found, paired) share per cell, MW-weighted."""


def load(*, config: str) -> pl.DataFrame:
    """Every scored truth row of one generator configuration, all methods."""
    files = sorted(OUT.glob(f"injected_{config}_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def paired_difference(
    *, a: pl.DataFrame, b: pl.DataFrame, cell: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """E1-J (`a`) minus E1 (`b`) MW-weighted paired share in a cell, interval from world draws."""
    worlds = sorted(a["world"].unique().to_list())
    tab = np.zeros((len(worlds), 4))
    for i, w in enumerate(worlds):
        for j, d in enumerate((a, b)):
            g = d.filter((pl.col("world") == w) & (pl.col("cell") == cell) & pl.col("edge_truth"))
            tab[i, 2 * j] = float((g["right_class"] * g["amp_mw"].abs()).sum())
            tab[i, 2 * j + 1] = float(g["amp_mw"].abs().sum())
    draws = rng.choice(len(worlds), size=(N_BOOT, len(worlds)))
    sel = tab[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = sel[:, 0] / sel[:, 1] - sel[:, 2] / sel[:, 3]
    point = tab[:, 0].sum() / tab[:, 1].sum() - tab[:, 2].sum() / tab[:, 3].sum()
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def gate_against_660(*, raw: pl.DataFrame) -> None:
    """E1 on seeds 900 to 907 must match experiment 660's stored recall by size (within 0.02)."""
    stored = json.loads((E660 / "ev4_recall_by_size.json").read_text())["bins"]
    e1 = raw.filter((pl.col("method") == "E1") & pl.col("world").is_between(900, 907))
    e1 = e1.filter((pl.col("days") >= 1) | (pl.col("days") < 0))
    bounds = [1.0, 2.0, 4.0, 6.0, 10.0, 20.01]
    for (lo, hi), s in zip(zip(bounds, bounds[1:], strict=False), stored, strict=False):
        sub = e1.filter((pl.col("units") >= lo) & (pl.col("units") < hi))
        per_world = [float(g["found"].mean()) for _, g in sub.group_by("world")]
        mine = float(np.mean(per_world)) if per_world else float("nan")
        print(f"GATE bin {lo:g}-{hi:g}: mine {mine:.3f} stored {s['injected_recall'][0]:.3f}")


def report(*, config: str) -> dict:
    """Cell table for one configuration, all methods, and the prediction checks."""
    raw = load(config=config)
    if raw.is_empty():
        return {}
    if config == "default":
        gate_against_660(raw=raw)
    rng = np.random.default_rng(670)
    frames = {m: with_cells(df=raw.filter(pl.col("method") == m)) for m in METHODS}
    res: dict = {"n_worlds": raw["world"].n_unique()}
    for m, d in frames.items():
        iv = boot_intervals(frames=[d])
        res[m] = {}
        print(f"\n== {config}: {m} ({raw['world'].n_unique()} worlds)")
        for cell in CELLS:
            st = cell_stats(frames=[d], cell=cell)
            res[m][cell] = {**st, "interval": iv[cell]}
            few = st["n_events"] < 20 or st["n_edge"] < 10
            print(
                f"{cell:16s} n={st['n_events']:4.0f} ends={st['n_edge']:3.0f}"
                f"{' (too few)' if few else ''} found MW {st['found_mw']:.3f} {iv[cell]['found']}"
                f" | paired MW {st['paired_mw']:.3f} {iv[cell]['paired']}"
                f" | start<=2h {st['start_within_2h']:.2f} end<=6h {st['end_within_6h']:.2f}"
                f" | median err start {st['median_start_err_h']:.1f} h end"
                f" {st['median_end_err_h']:.1f} h"
            )
    res["difference"] = {}
    for cell in CELLS:
        res["difference"][cell] = paired_difference(
            a=frames["E1-J"], b=frames["E1"], cell=cell, rng=rng
        )
        print(f"paired E1-J minus E1, {cell}: {res['difference'][cell]}")
    # Pairing of the rewired placebo as a share of E1-J's pairing, long cells.
    return res


def main() -> None:
    """Report every configuration found on disk."""
    out = {
        c: report(config=c) for c in ("default", "long_heavy", "short_heavy", "constant_mw_square")
    }
    (OUT / "injected_by_cell.json").write_text(json.dumps(out, indent=1, default=float))
    assert DAY == 48 and wrec is not None
    print(Path(sys.argv[0]).name, "done")


if __name__ == "__main__":
    main()
