"""Aggregate experiment 678's scored rows into per-cell shares, contrasts, and intervals.

Run with the environment of `e00678_run.py`: `uv run --no-sync python e00678_report.py`. Reads
`rows_*.parquet` from the experiment's data folder and writes `summary.json` there.
"""

# ruff: noqa: E501  # research code; long expressions kept as run

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00678_run import OUT

CELLS: Final[tuple[str, ...]] = (
    "long, deep",
    "medium, deep",
    "long, shallow",
    "medium, shallow",
    "short, deep",
    "short, shallow",
)
FAMILIES: Final[tuple[str, ...]] = (
    "E1-J", "E1", "E1-J rewired",
    "F-hyb", "R-hyb", "P-M-hyb", "P-C-hyb",
    "F-hyb rewired", "R-hyb rewired",
)  # fmt: skip
N_BOOT: Final[int] = 2000
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)


def family_of(method: str) -> tuple[str, int]:
    """Family name and seed of a method label such as `R-hyb rewired/2`."""
    if "/" in method:
        fam, seed = method.rsplit("/", 1)
        return fam, int(seed)
    return method, 0


def tables(
    *, df: pl.DataFrame
) -> dict[tuple[str, int, str, int], tuple[float, float, float, float]]:
    """(family, seed, cell, world) -> paired MW, edge MW, donor-found MW, donor MW."""
    edge = df.filter(pl.col("edge_truth") & (pl.col("units") >= 2.0))
    out: dict = {}
    for (method, cell, world), g in edge.group_by(["method", "cell", "world"]):
        fam, seed = family_of(method)
        w = g["amp_mw"].abs()
        donor = g["amp_mw"] < 0
        out[(fam, seed, cell, world)] = (
            float((g["right_class"].cast(pl.Float64) * w).sum()),
            float(w.sum()),
            float((g["found"].cast(pl.Float64) * w * donor.cast(pl.Float64)).sum()),
            float((w * donor.cast(pl.Float64)).sum()),
        )
    return out


def share(
    *, tab: dict, fam: str, seeds: list[int], cell: str, worlds: np.ndarray, col: int
) -> float:
    """Mean over seeds of the MW-weighted share of one column over a multiset of worlds."""
    vals = []
    for s in seeds:
        num = sum(tab.get((fam, s, cell, int(w)), (0, 0, 0, 0))[col] for w in worlds)
        den = sum(tab.get((fam, s, cell, int(w)), (0, 0, 0, 0))[col + 1] for w in worlds)
        vals.append(num / den if den else float("nan"))
    return float(np.nanmean(vals))


def main() -> None:
    """Print and store per-cell shares and contrasts with the reference E1-J."""
    df = pl.concat([pl.read_parquet(f) for f in sorted(OUT.glob("rows_*.parquet"))])
    worlds = np.array(sorted(df["world"].unique().to_list()))
    tab = tables(df=df)
    seeds = {f: sorted({k[1] for k in tab if k[0] == f}) for f in FAMILIES}
    rng = np.random.default_rng(5)
    boots = [rng.choice(worlds, len(worlds)) for _ in range(N_BOOT)]
    out: dict = {"n_worlds": len(worlds), "cells": {}, "false_transfers_per_primary_year": {}}
    for cell in CELLS:
        row: dict = {}
        for fam in FAMILIES:
            if not seeds[fam]:
                continue
            paired = share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=worlds, col=0)
            donor = share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=worlds, col=2)
            row[fam] = {"paired": paired, "donor_found": donor, "seeds": len(seeds[fam]),
                        "paired_per_seed": [share(tab=tab, fam=fam, seeds=[s], cell=cell, worlds=worlds, col=0) for s in seeds[fam]]}  # fmt: skip
            if fam != "E1-J":
                bs = [
                    share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=w, col=0)
                    - share(tab=tab, fam="E1-J", seeds=[0], cell=cell, worlds=w, col=0)
                    for w in boots
                ]
                row[fam]["paired_minus_e1j_ci"] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]  # fmt: skip
                row[fam]["paired_minus_e1j"] = paired - row["E1-J"]["paired"]
            lo = [
                share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=w, col=0) for w in boots
            ]
            row[fam]["paired_ci"] = [float(x) for x in np.nanpercentile(lo, [2.5, 97.5])]
        out["cells"][cell] = row
        print(
            cell, {f: (round(v["paired"], 3), round(v["donor_found"], 3)) for f, v in row.items()}
        )
    for method, g in df.group_by("method"):
        m = method[0]
        per_world = g.group_by("world").agg(
            pl.col("world_edge_events").first(), pl.col("world_right_pairs").first()
        )
        false = float((per_world["world_edge_events"] - per_world["world_right_pairs"]).sum())
        out["false_transfers_per_primary_year"][m] = false / (len(worlds) * 15 * YEARS)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                k: round(v, 2)
                for k, v in out["false_transfers_per_primary_year"].items()
                if "/0" in k or "/" not in k
            },
            indent=0,
        )
    )


if __name__ == "__main__":
    main()
