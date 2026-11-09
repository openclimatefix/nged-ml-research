"""Aggregate the injected worlds of experiment 672 by cell and check the pre-registered predictions.

Run with the same environment as `e00672_run.py`:

    uv run --no-sync python e00672_report.py

It reads experiment 670's stored rows for E1, E1-J, and E1-J rewired, and every
`injected_<tag>_<config>_<first seed>.parquet` of this experiment, and writes `by_cell.json` in
this experiment's data folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00665_score import CELLS, with_cells  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00672-graph-model-that-finds-the-donor-of-a-load-transfer"
)
E670: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00670-event-table-recipient-search-on-six-cell-grid"
ARMS: Final[tuple[str, ...]] = ("G", "GR", "G0", "GS")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_BOOT: Final[int] = 2000
MEASURES: Final[tuple[str, ...]] = (
    "found",
    "paired",
    "donor_found",
    "donor_paired",
    "recipient_found",
    "recipient_paired",
)


def rows_of(*, measure: str, df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """The rows a measure is over and the true-or-false column it averages."""
    edge = df.filter(pl.col("edge_truth"))
    table = {
        "found": (df, "found"),
        "paired": (edge, "right_class"),
        "donor_found": (edge.filter(pl.col("amp_mw") < 0), "found"),
        "donor_paired": (edge.filter(pl.col("amp_mw") < 0), "right_class"),
        "recipient_found": (edge.filter(pl.col("amp_mw") > 0), "found"),
        "recipient_paired": (edge.filter(pl.col("amp_mw") > 0), "right_class"),
    }
    return table[measure]


def per_world(*, df: pl.DataFrame, cell: str, measure: str, worlds: list[int]) -> np.ndarray:
    """Per world, the megawatts of the measure's rows that count and all its megawatts."""
    rows, col = rows_of(measure=measure, df=df.filter(pl.col("cell") == cell))
    out = np.zeros((len(worlds), 2))
    for i, w in enumerate(worlds):
        g = rows.filter(pl.col("world") == w)
        wt = g["amp_mw"].abs()
        out[i] = (float((g[col].cast(pl.Float64) * wt).sum()), float(wt.sum()))
    return out


def share(*, tabs: list[np.ndarray], draw: np.ndarray | None = None) -> float:
    """Mean over seeds (tables) of the megawatt-weighted share, over drawn or all worlds."""
    vals = []
    for t in tabs:
        s = t if draw is None else t[draw]
        vals.append(s[:, 0].sum() / s[:, 1].sum() if s[:, 1].sum() > 0 else np.nan)
    return float(np.nanmean(vals))


def contrast(
    *, a: list[np.ndarray], b: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean share of `a` minus that of `b`, with a 95% interval from resampling whole worlds."""
    n = len(a[0])
    point = share(tabs=a) - share(tabs=b)
    d = [
        share(tabs=a, draw=rng.integers(0, n, n)) - share(tabs=b, draw=rng.integers(0, n, n))
        for _ in range(N_BOOT)
    ]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_670() -> pl.DataFrame:
    """Experiment 670's stored default-world rows with cells."""
    files = sorted(E670.glob("injected_default_*.parquet"))
    raw = pl.concat([pl.read_parquet(f) for f in files])
    return with_cells(df=raw).with_columns(world=pl.col("world").cast(pl.Int64))


def load_mine(*, arm: str, config: str = "default") -> list[pl.DataFrame]:
    """This experiment's rows for the seeds of one arm, one frame per seed."""
    out = []
    for k in SEEDS:
        files = sorted(OUT.glob(f"injected_{arm}_default_seed{k}_{config}_*.parquet"))
        if files:
            raw = pl.concat([pl.read_parquet(f) for f in files])
            out.append(with_cells(df=raw).with_columns(world=pl.col("world").cast(pl.Int64)))
    return out


def timing(*, frames: list[pl.DataFrame], cell: str) -> dict[str, float]:
    """Share of found events with the start within 2 hours and the end within 6 hours."""
    s_ok, e_ok = [], []
    for f in frames:
        g = f.filter((pl.col("cell") == cell) & pl.col("found"))
        s = g["start_err_hh"].to_numpy()
        e = g.filter(pl.col("end") > 0)["end_err_hh"].to_numpy()
        s_ok.append(float(np.mean(np.abs(s) <= 4)) if len(s) else np.nan)
        e_ok.append(float(np.mean(np.isfinite(e) & (np.abs(e) <= 12))) if len(e) else np.nan)
    return {"start_within_2h": float(np.nanmean(s_ok)), "end_within_6h": float(np.nanmean(e_ok))}


def main() -> None:
    """Cell tables per arm, contrasts against E1-J and G-0, and false transfers."""
    ref = load_670()
    worlds = sorted(ref["world"].unique().to_list())
    refs = {m: ref.filter(pl.col("method") == m) for m in ("E1", "E1-J", "E1-J rewired")}
    mine = {a: load_mine(arm=a) for a in ARMS}
    rng = np.random.default_rng(672)
    res: dict = {"n_worlds": len(worlds), "arms": {}, "contrast": {}, "false_transfers": {}}
    tabs: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    for name, frames in [*[(m, [d]) for m, d in refs.items()], *mine.items()]:
        if not frames:
            continue
        tabs[name] = {
            cell: {
                m: [per_world(df=f, cell=cell, measure=m, worlds=worlds) for f in frames]
                for m in MEASURES
            }
            for cell in CELLS
        }
        res["arms"][name] = {
            cell: {
                **{m: share(tabs=tabs[name][cell][m]) for m in MEASURES},
                **timing(frames=frames, cell=cell),
                "n_events": int(frames[0].filter(pl.col("cell") == cell).height),
                "n_edge_ends": int(
                    frames[0].filter((pl.col("cell") == cell) & pl.col("edge_truth")).height
                ),
            }
            for cell in CELLS
        }
    for name in ("G", "GR", "G0", "GS"):
        if name not in tabs:
            continue
        res["contrast"][f"{name} minus E1-J"] = {
            cell: {
                m: contrast(a=tabs[name][cell][m], b=tabs["E1-J"][cell][m], rng=rng)
                for m in MEASURES
            }
            for cell in CELLS
        }
    if "G" in tabs:
        for other in ("G0", "GS"):
            if other in tabs:
                res["contrast"][f"G minus {other}"] = {
                    cell: {
                        m: contrast(a=tabs["G"][cell][m], b=tabs[other][cell][m], rng=rng)
                        for m in MEASURES
                    }
                    for cell in CELLS
                }
    for name, frames in [*[(m, [d]) for m, d in refs.items()], *mine.items()]:
        if frames:
            w = [
                f.group_by("world").agg(
                    pl.col("world_edge_events").first(), pl.col("world_right_pairs").first()
                )
                for f in frames
            ]
            res["false_transfers"][name] = {
                "decoded_transfers": float(np.mean([x["world_edge_events"].sum() for x in w])),
                "match_no_injected_end": float(
                    np.mean([(x["world_edge_events"] - x["world_right_pairs"]).sum() for x in w])
                ),
            }
    (OUT / "by_cell.json").write_text(json.dumps(res, indent=1, default=float))
    for name, cells in res["arms"].items():
        for cell in CELLS[:3]:
            c = cells[cell]
            print(
                name,
                cell,
                {m: round(c[m], 3) for m in MEASURES},
                "n",
                c["n_events"],
                c["n_edge_ends"],
            )
    print(json.dumps(res["false_transfers"]))


if __name__ == "__main__":
    main()
