"""Experiment 676, Part B: per-cell shares, contrasts, and predictions G1 to G5.

    uv run --no-sync python e00676_report.py   (PYTHONPATH as in `e00676_train.py`)

Reads experiment 678's rows (E1-J, F-hyb, R-hyb) and this experiment's `scored/*_rows.parquet`,
and writes `summary.json` in the experiment's data folder. Shares are MW-weighted and intervals
resample whole worlds, as in experiment 678's report.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00676_worlds import ROOT
from e00678_report import CELLS, family_of, share, tables  # ty: ignore[unresolved-import]
from e00678_run import OUT as OUT678  # ty: ignore[unresolved-import]

REFS: Final = ("E1-J", "F-hyb", "R-hyb")
N_BOOT: Final[int] = 2000
YEARS: Final[float] = (GRID_LENGTH - SCORED_FROM) / (365.25 * DAY)
DEEP: Final = ("long, deep", "medium, deep")


def load() -> pl.DataFrame:
    """Rows of the reference families and of every scored network."""
    ref = pl.concat([pl.read_parquet(f) for f in sorted(OUT678.glob("rows_*.parquet"))])
    ref = ref.filter(pl.col("method").map_elements(lambda m: family_of(m)[0] in REFS, return_dtype=pl.Boolean))  # fmt: skip
    mine = [pl.read_parquet(f) for f in sorted((ROOT / "scored").glob("*_rows.parquet")) if "seed999" not in f.name]  # fmt: skip
    cols = ref.columns
    out = pl.concat([ref] + [m.select(cols) for m in mine])
    return out.unique(
        subset=["method", "world", "site", "start", "amp_mw", "kind"],
        keep="first",
        maintain_order=True,
    )


def main() -> None:
    """Print and store the results."""
    df = load()
    worlds = np.array(sorted(df["world"].unique().to_list()))
    tab = tables(df=df)
    fams = sorted({family_of(m)[0] for m in df["method"].unique().to_list()})
    seeds = {f: sorted({k[1] for k in tab if k[0] == f}) for f in fams}
    rng = np.random.default_rng(5)
    boots = [rng.choice(worlds, len(worlds)) for _ in range(N_BOOT)]
    out: dict = {"n_worlds": len(worlds), "cells": {}, "false_transfers_per_primary_year": {}, "start_err": {}}  # fmt: skip
    for cell in CELLS:
        row: dict = {}
        for fam in fams:
            p = share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=worlds, col=0)
            d = share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=worlds, col=2)
            ci = [float(x) for x in np.nanpercentile([share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=w, col=0) for w in boots], [2.5, 97.5])]  # fmt: skip
            row[fam] = {"paired": p, "donor_found": d, "paired_ci": ci, "seeds": seeds[fam],
                        "per_seed": [share(tab=tab, fam=fam, seeds=[s], cell=cell, worlds=worlds, col=0) for s in seeds[fam]]}  # fmt: skip
            for ref in ("E1-J", "R-hyb"):
                if fam != ref and ref in fams:
                    bs = [share(tab=tab, fam=fam, seeds=seeds[fam], cell=cell, worlds=w, col=0) - share(tab=tab, fam=ref, seeds=seeds[ref], cell=cell, worlds=w, col=0) for w in boots]  # fmt: skip
                    row[fam][f"minus_{ref}"] = [p - row[ref]["paired"] if ref in row else float("nan"), *[float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]]  # fmt: skip
        out["cells"][cell] = row
        print(cell, {f: round(v["paired"], 3) for f, v in row.items()})
    for method, g in df.group_by("method"):
        pw = g.group_by("world").agg(pl.col("world_edge_events").first(), pl.col("world_right_pairs").first())  # fmt: skip
        out["false_transfers_per_primary_year"][method[0]] = float((pw["world_edge_events"] - pw["world_right_pairs"]).sum()) / (len(worlds) * 15 * YEARS)  # fmt: skip
        rec = g.filter(pl.col("edge_truth") & (pl.col("amp_mw") > 0) & pl.col("right_class"))
        out["start_err"][method[0]] = [float(rec["start_err_hh"].abs().median()) if rec.height else float("nan"), rec.height]  # fmt: skip  # ty: ignore[invalid-argument-type]
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: round(v, 2) for k, v in out["false_transfers_per_primary_year"].items()}))
    print(out["start_err"])


if __name__ == "__main__":
    main()
