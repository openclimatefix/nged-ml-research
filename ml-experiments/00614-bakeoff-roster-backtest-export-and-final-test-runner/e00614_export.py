# ruff: noqa: D103
"""Experiment 614: export the stored backtests of the three roster networks (CPU, no fit).

Layout, columns, units and caveats follow experiment 593's export (`e00593_export.py`, whose
`Context` this script reuses for the observed power, the load-shift flags and the units). Every
forecast is read from experiments 612 and 613's stored fits (`arms/*/work/fits`) and combined the
way
experiment 613 scored it: the mean of seeds 0 to 4 quantile by quantile for each Primary, origin
time,
and valid time, then sorted across the 11 levels. Nothing is refitted.

Networks (one Parquet file per network and fold month, named by the month of the forecast origin):

- `K1_<month>`: class 1. The lag network with synthetic shifts, E3, the offset D5, and the G4 gated
  band (the union band of the network and its lags-masked twin on the origin days where the gate is
  on or held for 7 days). `gate_on` is filled.
- `Candidate_<month>`: class 2. The network of experiment 610's P7 arm with 50% signal dropout,
  scaled about its median by experiment 613's per-month width factor so that its band is as wide as
  K1's (`width_factor` holds the factor). No gate.
- `K3_<month>`: class 3. The lag-free network with the offset D5. No gate.

No tail constants are applied to any network. The three files of a month hold identical keys (the
intersection of the three networks' stored rows).

Usage: python e00614_export.py [MONTH ...]
"""

import os
import sys
from pathlib import Path

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00593_export as x593  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT613 = ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds"
OUT = Path(
    os.environ.get("E00614_OUT", ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07")
)
"""The export folder; `E00614_OUT` redirects it (used for the seeds 0 and 1 cross-check of
experiment 612)."""
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS = tuple(int(x) for x in os.environ.get("E00614_SEEDS", "0 1 2 3 4").split())
KEYS = list(a.KEYS)
QCOLS = list(a.QCOLS)
MAX_LEAD_H = 240
IQ50 = a.IQ[0.5]
MODE = np.load(ROOT613 / "score" / "k1_d5" / "mode_g4.npy")
"""G4 gate state of experiment 613's K1 score folder: (16 Primaries, origin days), True where the
band is wide."""
WIDTH = dict(pl.read_csv(ROOT613 / "score" / "c2_width_factors.csv").iter_rows())
"""Experiment 613's band-width factor of the class 2 candidate, one per month of the forecast
origin."""


class RosterContext(x593.Context):
    """Experiment 593's context, with the gate state read from `mode_g4.npy` of experiment 613."""

    def _gate(self) -> None:
        self.gated = MODE


def load(*, arm: str, tag: str, month: str) -> pl.DataFrame:
    """Return the five-seed mean quantiles of one stored arm and month, leads under 240 h."""
    a.SEEDS = SEEDS
    return a._load(root=ROOT613 / "arms" / arm / "work", tag=tag, months=(month,)).filter(
        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() < 60 * MAX_LEAD_H
    )


def matrix(df: pl.DataFrame, *, prefix: str = "") -> np.ndarray:
    return np.sort(df.select([prefix + q for q in QCOLS]).to_numpy().astype(np.float32), axis=1)


def write(*, tbl: pl.DataFrame, name: str, month: str) -> None:
    path = OUT / f"{name}_{month}.parquet"
    tbl.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
    print(name, month, tbl.height, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def main(months: tuple[str, ...]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = RosterContext()
    sites = ctx.sites.select(
        "time_series_id",
        pl.col("label").alias("primary_name"),
        pl.col("group")
        .replace_strict(
            {
                "MVA-only eight": "eight MVA Primaries",
                "signed-MW six": "six MW Primaries",
                "D15 and D18": "report-only (D15 and D18, short history)",
            }
        )
        .alias("group"),
        pl.col("capacity_mw").alias("capacity"),
        "unit",
    ).sort("time_series_id")
    sites.write_csv(OUT / "substations.csv")
    for month in months:
        v = load(arm="k1_d5", tag="P5", month=month)
        m = load(arm="k1_d5", tag="P5-removed", month=month).rename({q: f"M_{q}" for q in QCOLS})
        c = load(arm="c2_e3", tag="P5", month=month).rename({q: f"C_{q}" for q in QCOLS})
        k3 = load(arm="k3_d5", tag="D5", month=month).rename({q: f"L_{q}" for q in QCOLS})
        for name, frame in (("K1", v), ("twin", m), ("candidate", c), ("K3", k3)):
            assert frame.height == frame.unique(KEYS).height, f"{name}: duplicate keys"
        both = v.join(m, on=KEYS).join(c, on=KEYS).join(k3, on=KEYS).sort(KEYS)
        print(
            month,
            "rows: K1",
            v.height,
            "twin",
            m.height,
            "candidate",
            c.height,
            "K3",
            k3.height,
            "common",
            both.height,
            flush=True,
        )
        base = both.select(KEYS)
        prim = base["time_series_id"].replace_strict(ctx.prim_of, return_dtype=pl.Int64).to_numpy()
        origin = (
            (base["power_fcst_init_time"].dt.epoch("us") - x593.k578.prep.GRID_START_US)
            // x593.k578.prep.STEP_US
        ).to_numpy()
        wide = ctx.gated[prim, origin // 48]
        qv, qm = matrix(both), matrix(both, prefix="M_")
        k1 = k597.assemble(v=qv, m=qm, wide=wide)
        qc = matrix(both, prefix="C_")
        factor = WIDTH[month]
        med = qc[:, IQ50 : IQ50 + 1]
        cand = (med + factor * (qc - med)).astype(np.float32)
        k3q = matrix(both, prefix="L_")
        write(tbl=ctx.table(df=base, q=k1, gate=True), name="K1", month=month)
        write(
            tbl=ctx.table(df=base, q=cand, gate=False).with_columns(
                width_factor=pl.lit(factor).cast(pl.Float32)
            ),
            name="Candidate",
            month=month,
        )
        write(tbl=ctx.table(df=base, q=k3q, gate=False), name="K3", month=month)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or MONTHS)
