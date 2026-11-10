"""Aggregate experiment 682's injected worlds by cell and apply the pre-registered rules.

Run with the environment of `e00682_run.py`: `uv run --no-sync python e00682_report.py`. It reads
the `injected_<config>_<first seed>_<costs>.parquet` files under the experiment's data folder and
the stored rows of experiments 670 (E1, E1-J) and 680 (E1-J-sym), and writes `by_cell.json`.
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
    / "00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model"
)
E670: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00670-event-table-recipient-search-on-six-cell-grid"
E680: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
)
ARMS: Final[tuple[str, ...]] = ("E1-F", "E1-F rewired", "E1-F no edges", "E1-F all pairs")
DEEP: Final[tuple[str, str]] = ("long, deep", "medium, deep")
HYBRID_PAIRED: Final[dict[str, float]] = {"long, deep": 0.36, "medium, deep": 0.33}
"""Best of F-hyb and R-hyb in each deep cell, from experiment 678 (medium and deep approximate)."""


def stored(*, folder: Path, method: str, glob: str) -> pl.DataFrame:
    """One method's stored scored rows on the default worlds."""
    files = sorted(folder.glob(glob))
    df = pl.concat([pl.read_parquet(f) for f in files])
    return df.filter(pl.col("method") == method)


def load(*, config: str, costs: str) -> pl.DataFrame:
    """Scored rows of one configuration and one cost choice, all four arms."""
    files = sorted(OUT.glob(f"injected_{config}_*_{costs}.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def false_share(*, raw: pl.DataFrame) -> tuple[int, int, float]:
    """Decoded transfers, correct ends, and the share of decoded transfers beyond the ends."""
    w = raw.group_by("world").agg(
        pl.col("world_edge_events").first(), pl.col("world_right_pairs").first()
    )
    dec, right = int(w["world_edge_events"].sum()), int(w["world_right_pairs"].sum())
    return dec, right, 1 - right / dec if dec else float("nan")


def table(*, frames: dict[str, pl.DataFrame], raws: dict[str, pl.DataFrame]) -> dict:
    """Cell statistics, donor and recipient megawatts found, and false share for each arm."""
    res: dict = {}
    for m, d in frames.items():
        iv = boot_intervals(frames=[d])
        res[m] = {}
        for cell in CELLS:
            st = cell_stats(frames=[d], cell=cell)
            sub = d.filter((pl.col("cell") == cell) & pl.col("edge_truth"))
            don, rec = sub.filter(pl.col("amp_mw") < 0), sub.filter(pl.col("amp_mw") > 0)
            res[m][cell] = {
                **st,
                "interval": iv[cell],
                "donor_found_mw": wrec(df=don, col="found") if don.height else None,
                "recipient_found_mw": wrec(df=rec, col="found") if rec.height else None,
            }
        dec, right, fs = false_share(raw=raws[m])
        res[m]["decoded"], res[m]["right_ends"], res[m]["false_share"] = dec, right, fs
    return res


def verdict(*, res: dict) -> dict:
    """Stop rules and verdict of the plan, from the default worlds."""
    d = {c: res["diff_vs_E1-J"][c] for c in DEEP}
    both10 = all(v[0] >= 0.10 for v in d.values())
    n10 = sum(v[0] >= 0.10 for v in d.values())
    both05 = all(v[0] >= 0.05 for v in d.values())
    excl = all(v[1] > 0 for v in d.values())
    rew_ok = all(res["E1-F rewired"][c]["paired_mw"] <= 0.05 for c in DEEP)
    rew_half = any(
        res["E1-F rewired"][c]["paired_mw"] > 0.5 * res["E1-F"][c]["paired_mw"] for c in DEEP
    )
    allp_same = res["E1-F all pairs"]["false_share"] >= res["E1-F"]["false_share"] - 0.10
    close = n10 == 0
    topology_not_identified = rew_half or allp_same
    if close or topology_not_identified:
        stop = "close the edge-flow line at 15 Primaries"
    else:
        stop = "E1-F passes in at least one deep cell"
    if both10 and rew_ok and not topology_not_identified:
        v = "supported"
    elif n10 >= 1 or both05:
        v = "partly supported"
    elif excl:
        v = "small gain"
    else:
        v = "not supported"
    return {
        "stop_rule": stop,
        "verdict": v,
        "rewired_pairs_le_0.05": rew_ok,
        "rewired_over_half_of_E1-F": rew_half,
        "all_pairs_false_share_within_0.10": allp_same,
    }


def report(*, config: str, costs: str) -> dict:
    """Cell tables for one configuration, with contrasts against E1-J and E1-J-sym."""
    raw = load(config=config, costs=costs)
    if raw.is_empty():
        return {}
    rng = np.random.default_rng(682)
    raws = {m: raw.filter(pl.col("method") == m) for m in ARMS}
    refs = {}
    if config == "default":
        refs = {
            "E1-J": stored(folder=E670, method="E1-J", glob="injected_default_*.parquet"),
            "E1-J-sym": stored(folder=E680, method="E1-J-sym", glob="injected_default_*.parquet"),
            "E1": stored(folder=E670, method="E1", glob="injected_default_*.parquet"),
        }
    else:
        key = f"injected_{config}_*.parquet"
        refs = {
            "E1-J": stored(folder=E670, method="E1-J", glob=key),
            "E1-J-sym": stored(folder=E680, method="E1-J-sym", glob=key),
        }
    allraws = {**raws, **refs}
    frames = {m: with_cells(df=d) for m, d in allraws.items()}
    res: dict = {"n_worlds": raw["world"].n_unique(), "costs": costs}
    res.update(table(frames=frames, raws=allraws))
    for ref in ("E1-J", "E1-J-sym"):
        res[f"diff_vs_{ref}"] = {
            c: paired_difference(a=frames["E1-F"], b=frames[ref], cell=c, rng=rng) for c in CELLS
        }
    for m in ARMS:
        print(f"\n== {config} ({costs} costs): {m}")
        for c in CELLS[:6]:
            s = res[m][c]
            print(
                f"{c:16s} found {s['found_mw']:.3f} paired {s['paired_mw']:.3f}"
                f" {s['interval']['paired']}"
            )
        print(
            "decoded",
            res[m]["decoded"],
            "right ends",
            res[m]["right_ends"],
            "false",
            res[m]["false_share"],
        )
    for ref in ("E1-J", "E1-J-sym"):
        print(
            f"\nE1-F minus {ref}:",
            {c: [round(x, 3) for x in v] for c, v in res[f"diff_vs_{ref}"].items()},
        )
    if config == "default" and costs == "default":
        res["gate_no_edges_vs_E1"] = {
            c: [res["E1-F no edges"][c]["found_mw"], res["E1"][c]["found_mw"]] for c in CELLS[:6]
        }
        res["verdict"] = verdict(res=res)
        print(json.dumps(res["verdict"], indent=1))
    return res


def main() -> None:
    """Report every configuration and cost choice found on disk."""
    out: dict = {}
    for config in ("default", "constant_mw_square", "long_heavy", "short_heavy"):
        for costs in ("default", "constant_mw_square", "default_wide"):
            r = report(config=config, costs=costs)
            if r:
                out[f"{config}|{costs}"] = r
    (OUT / "by_cell.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
