# ruff: noqa: E501
"""Experiment 679: the ratio of the probe curve's energy to the injected fleet's energy, per Primary.

The probe reads a capacity `c` (MW) by regressing the probed reading on the standard unit curve `u`.
The energy the probe attributes to the farm over the probed half-hours is therefore `c` times the
sum of `u`, and the energy injected is the injected capacity times the sum of the fleet shape. The
recovery in energy is the recovery in MW times `mean(u) / mean(fleet shape)`, both means taken over
the same probe cells (the probe dates' base half-hours), with `u` averaged over each cell's donors.
The fleet shape is the one experiment 662's world builder injects (farm rotation 0).

Usage: python e00679_energy.py   (writes `work/energy_factor.csv`)
"""

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import e00662_common as k  # ty: ignore[unresolved-import]
import e00662_probe as probe  # ty: ignore[unresolved-import]
import e00679_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

CUTOFF = "2026-06"


def main() -> None:
    """Build the fleet shapes as experiment 662's fit does and write each Primary's energy factor."""
    k.WORK = run.WORK
    clean = c.clean_stack()
    refit_t = c.month_start(month=CUTOFF)
    rows = data.build_fold(refit=refit_t, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
    world = k.WorldBuilder(
        primary=pre.primary, target_idx=pre.target_idx, x=pre.x, y=pre.y,
        standardiser=rows.standardiser, clean=clean,
    )  # fmt: skip
    frame = probe.load_frame(cutoff=CUTOFF)
    cap = k621.effective_capacity_mw()
    out = []
    for j, (site, p) in enumerate(k.index_of(sites=k.SIX).items()):
        pp = probe.PrimaryProbe(frame=frame, primary=p, site=site, cap=float(cap[p]))
        fleet = world.fleet(j=j, p=p, rotation=0)
        sel = world.primary == p
        shape_by_t = dict(zip(world.target_idx[sel].tolist(), fleet[sel].tolist(), strict=True))
        base = pp.probe["base"]
        t_idx = np.sort(frame["target_idx"][frame["primary"] == p])[
            base
        ]  # `base` indexes the Primary's rows sorted by time
        shape = np.array([shape_by_t.get(int(t), np.nan) for t in t_idx])
        ok = ~np.isnan(shape)
        cells = pp.cell
        # Mean over cells of (mean of u over the cell's donors) and of (mean fleet shape at the cell's base rows)
        u_cell = (
            pl.DataFrame({"cell": cells[ok], "u": pp.u_std[ok], "s": shape[ok]})
            .group_by("cell")
            .agg(u=pl.col("u").mean(), s=pl.col("s").mean())
        )
        u_mean, s_mean = float(u_cell["u"].mean()), float(u_cell["s"].mean())  # ty: ignore[invalid-argument-type]
        assert s_mean > 0.0, site
        factor = u_mean / s_mean
        out.append({"site": site, "energy_factor": factor, "n_cells": u_cell.height})
        print(site, round(factor, 3), u_cell.height, flush=True)
    pl.DataFrame(out).write_csv(run.WORK / "energy_factor.csv")


if __name__ == "__main__":
    main()
