"""Aggregate experiment 680's injected worlds by cell and apply the pre-registered stage 1 rule.

Run with the environment of `e00680_run.py`:

    uv run --no-sync python e00680_report.py

It reads every `injected_<config>_<first seed>.parquet` under the experiment's data folder, and
writes `injected_by_cell.json` there. E1-J is also re-read from experiment 670's stored rows, to
check that the rerun reproduces them (the gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00665_score import (  # ty: ignore[unresolved-import]
    CELLS,
    boot_intervals,
    cell_stats,
    with_cells,
    wrec,
)
from e00670_report import paired_difference  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
)
E670: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00670-event-table-recipient-search-on-six-cell-grid"
METHODS: Final[tuple[str, str, str]] = ("E1-J", "E1-J-sym", "E1-J-sym rewired")
DEEP: Final[tuple[str, str]] = ("long, deep", "medium, deep")


def load(*, config: str) -> pl.DataFrame:
    """Every scored truth row of one configuration, all methods."""
    files = sorted(OUT.glob(f"injected_{config}_*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def gate(*, raw: pl.DataFrame) -> dict:
    """E1-J here must reproduce experiment 670's stored E1-J paired and found shares per cell."""
    old = pl.concat([pl.read_parquet(f) for f in sorted(E670.glob("injected_default_*.parquet"))])
    old = with_cells(df=old.filter(pl.col("method") == "E1-J"))
    new = with_cells(df=raw.filter(pl.col("method") == "E1-J"))
    out: dict = {}
    for cell in CELLS:
        a = cell_stats(frames=[old], cell=cell)
        b = cell_stats(frames=[new], cell=cell)
        ok = abs(a["paired_mw"] - b["paired_mw"]) <= 0.005 or (
            np.isnan(a["paired_mw"]) and np.isnan(b["paired_mw"])
        )
        out[cell] = {"stored": a["paired_mw"], "rerun": b["paired_mw"], "ok": bool(ok)}
        print(f"GATE {cell:16s} stored paired {a['paired_mw']:.4f} rerun {b['paired_mw']:.4f}")
    worlds_new = set(raw.filter(pl.col("method") == "E1-J")["world"].to_list())
    worlds_old = set(old["world"].to_list())
    out["same_worlds"] = worlds_new == worlds_old
    out["passed"] = out["same_worlds"] and all(out[c]["ok"] for c in CELLS)
    print("GATE passed:", out["passed"], "(same worlds:", out["same_worlds"], ")")
    return out


def report(*, config: str) -> dict:
    """Cell table for one configuration, the three methods, and the sym-minus-J contrast."""
    raw = load(config=config)
    if raw.is_empty():
        return {}
    res: dict = {"n_worlds": raw["world"].n_unique()}
    if config == "default":
        res["gate"] = gate(raw=raw)
    rng = np.random.default_rng(680)
    frames = {m: with_cells(df=raw.filter(pl.col("method") == m)) for m in METHODS}
    for m, d in frames.items():
        iv = boot_intervals(frames=[d])
        res[m] = {}
        print(f"\n== {config}: {m} ({res['n_worlds']} worlds)")
        for cell in CELLS:
            st = cell_stats(frames=[d], cell=cell)
            donors = d.filter(
                (pl.col("cell") == cell) & pl.col("edge_truth") & (pl.col("amp_mw") < 0)
            )
            recips = d.filter(
                (pl.col("cell") == cell) & pl.col("edge_truth") & (pl.col("amp_mw") > 0)
            )
            res[m][cell] = {
                **st,
                "interval": iv[cell],
                "donor_found_mw": wrec(df=donors, col="found") if donors.height else None,
                "recipient_found_mw": wrec(df=recips, col="found") if recips.height else None,
            }
            print(
                f"{cell:16s} n={st['n_events']:4.0f} ends={st['n_edge']:3.0f}"
                f" found {st['found_mw']:.3f} {iv[cell]['found']}"
                f" | paired {st['paired_mw']:.3f} {iv[cell]['paired']}"
            )
        w = (
            raw.filter(pl.col("method") == m)
            .group_by("world")
            .agg(pl.col("world_edge_events").first(), pl.col("world_right_pairs").first())
        )
        res[m]["edge_events"] = int(w["world_edge_events"].sum())
        res[m]["right_pairs"] = int(w["world_right_pairs"].sum())
        print(
            f"{m}: decoded edge events {res[m]['edge_events']},"
            f" correct ends {res[m]['right_pairs']}"
        )
    res["sym_minus_J"] = {}
    for cell in CELLS:
        res["sym_minus_J"][cell] = paired_difference(
            a=frames["E1-J-sym"], b=frames["E1-J"], cell=cell, rng=rng
        )
        print(f"paired E1-J-sym minus E1-J, {cell}: {res['sym_minus_J'][cell]}")
    if config == "default":
        diffs = [res["sym_minus_J"][c][0] for c in DEEP]
        branch = (
            "a: adopt E1-J-sym as the event table"
            if all(x >= 0.10 for x in diffs)
            else "b: weak mechanism, run part 2 with a prior of 20% or less"
            if all(x < 0.03 for x in diffs)
            else "c: run part 2"
        )
        res["stage1_branch"] = branch
        print("STAGE 1:", branch, diffs)
        res["verdict"] = verdict(res=res)
        print("VERDICT:", res["verdict"])
    return res


def verdict(*, res: dict) -> str:
    """The pre-registered verdict from the deep-cell differences and their intervals."""
    if not res["gate"]["passed"]:
        return "invalid"
    d = {c: res["sym_minus_J"][c] for c in DEEP}
    small = any(
        res["E1-J-sym"][c]["n_events"] < 20 or res["E1-J-sym"][c]["n_edge"] < 10 for c in DEEP
    )
    rewired_edge = res["E1-J-sym rewired"]["edge_events"]
    holds = rewired_edge <= res["E1-J-sym"]["edge_events"] / 3
    if small:
        return "inconclusive"
    if all(v[0] >= 0.10 for v in d.values()) and holds:
        return "supported"
    if any(v[0] >= 0.10 for v in d.values()) or all(v[0] >= 0.05 for v in d.values()):
        return "partly supported"
    if all(v[1] > 0 for v in d.values()):
        return "small gain"
    return "not supported"


def main() -> None:
    """Report every configuration found on disk."""
    out = {
        c: report(config=c) for c in ("default", "long_heavy", "short_heavy", "constant_mw_square")
    }
    (OUT / "injected_by_cell.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
