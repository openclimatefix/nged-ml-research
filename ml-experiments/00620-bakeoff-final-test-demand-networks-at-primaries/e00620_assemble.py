# ruff: noqa: E501, ANN201, D103
"""Experiment 620: assemble the three networks' stored fits into the export tables (CPU, no fit).

Follows experiment 614's `e00614_export.py` (same seed mean, same sorting of quantiles, same union
band, same table layout), with three differences, all fixed before the test window is opened:

- The final fold has one fit at 2026-07-01, so the fits are `<month>_d0_s<seed>.parquet` only.
- The G4 gate state of the window is computed by `e00620_gate.g4_mode` from the K3 forecasts and
  the observed power, with the threshold frozen from the development months.
- The candidate's band is scaled about its median by the frozen factor of `freeze_config.json`.

`--dev-check` runs the same code on a development month from experiment 613's stored fits and its
stored gate state, and compares each table with experiment 614's export of that month.

Usage: python e00620_assemble.py 2026-07 | python e00620_assemble.py --dev-check 2026-06
"""

import json
import os
import sys
from pathlib import Path

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00593_export as x593  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00620_gate as gate  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HERE = Path(__file__).parent
R = ML_EXPERIMENTS_DATA_DIR / "00620-bakeoff-final-test-demand-networks-at-primaries"
S613 = next(ML_EXPERIMENTS_DATA_DIR.glob("00613-*"))
KEYS, QCOLS, IQ50 = list(a.KEYS), list(a.QCOLS), a.IQ[0.5]
MAX_LEAD_H = 240
SEEDS = (0, 1, 2, 3, 4)


def load(*, root: Path, tag: str, month: str, days: tuple[str, ...]) -> pl.DataFrame:
    """Return the five-seed mean quantiles of one stored arm, leads under 240 h."""
    parts = [
        pl.read_parquet(root / "fits" / tag / f"{month}_{d}_s{s}.parquet", columns=[*KEYS, *QCOLS])
        for d in days
        for s in SEEDS
    ]
    mean = pl.concat(parts).group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS])
    return mean.filter((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() < 60 * MAX_LEAD_H)


def matrix(df: pl.DataFrame, *, prefix: str = "") -> np.ndarray:
    return np.sort(df.select([prefix + q for q in QCOLS]).to_numpy().astype(np.float32), axis=1)


class Context(x593.Context):
    """Experiment 593's context with the gate state supplied by the caller."""

    mode: np.ndarray

    def _gate(self) -> None:
        self.gated = np.zeros((16, 1), bool)  # replaced by `assemble` before any table is built


def assemble(*, month: str, roots: dict[str, Path], days: tuple[str, ...], factor: float, mode: np.ndarray | None, out: Path) -> dict[str, pl.DataFrame]:
    ctx = Context()
    k1_root, cand_root, k3_root = roots["K1"], roots["Candidate"], roots["K3"]
    v = load(root=k1_root, tag="P5", month=month, days=days)
    m = load(root=k1_root, tag="P5-removed", month=month, days=days).rename({q: f"M_{q}" for q in QCOLS})
    c = load(root=cand_root, tag="P5", month=month, days=days).rename({q: f"C_{q}" for q in QCOLS})
    k3 = load(root=k3_root, tag="D5", month=month, days=days).rename({q: f"L_{q}" for q in QCOLS})
    for name, frame in (("K1", v), ("twin", m), ("candidate", c), ("K3", k3)):
        assert frame.height == frame.unique(KEYS).height, f"{name}: duplicate keys"
    both = v.join(m, on=KEYS).join(c, on=KEYS).join(k3, on=KEYS).sort(KEYS)
    print(month, "rows: K1", v.height, "twin", m.height, "candidate", c.height, "K3", k3.height, "common", both.height, flush=True)
    assert both.height == v.height == m.height == c.height == k3.height, "the four networks' rows differ"
    if mode is None:
        lag_free = load(root=k3_root, tag="D5", month=month, days=days)
        n_days = ctx.clean.shape[1] // 48
        gap = gate.fresh_gap(lag_free=lag_free, clean=ctx.clean, n_days=n_days)
        origin_days = sorted({int(x) for x in (lag_free["power_fcst_init_time"].dt.epoch("us").to_numpy() - gate.START_US) // gate.STEP_US // 48})
        mode, on, snap = gate.g4_mode(gap=gap, origin_days=origin_days)
        np.savez(out / f"gate_state_{month}.npz", gap=gap, mode=mode, on=on, snap=snap, thr=gate.frozen_threshold())
    ctx.gated = mode
    base = both.select(KEYS)
    prim = base["time_series_id"].replace_strict(ctx.prim_of, return_dtype=pl.Int64).to_numpy()
    origin = ((base["power_fcst_init_time"].dt.epoch("us") - gate.START_US) // gate.STEP_US).to_numpy()
    wide = mode[prim, origin // 48]
    k1 = k597.assemble(v=matrix(both), m=matrix(both, prefix="M_"), wide=wide)
    qc = matrix(both, prefix="C_")
    med = qc[:, IQ50 : IQ50 + 1]
    cand = (med + factor * (qc - med)).astype(np.float32)
    return {
        "K1": ctx.table(df=base, q=k1, gate=True),
        "Candidate": ctx.table(df=base, q=cand, gate=False).with_columns(width_factor=pl.lit(factor).cast(pl.Float32)),
        "K3": ctx.table(df=base, q=matrix(both, prefix="L_"), gate=False),
    }


def write(*, tables: dict[str, pl.DataFrame], month: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, tbl in tables.items():
        path = out / f"{name}_{month}.parquet"
        tbl.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
        print(name, month, tbl.height, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def dev_check(month: str) -> None:
    exp = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
    factor = dict(pl.read_csv(S613 / "score" / "c2_width_factors.csv").iter_rows())[month]
    out = R / "dev_check"
    out.mkdir(exist_ok=True)
    arms = S613 / "arms"
    roots = {"K1": arms / "k1_d5" / "work", "Candidate": arms / "c2_e3" / "work", "K3": arms / "k3_d5" / "work"}
    mode = np.load(S613 / "score" / "k1_d5" / "mode_g4.npy")
    tables = assemble(month=month, roots=roots, days=("d0", "d14"), factor=factor, mode=mode, out=out)
    for name, tbl in tables.items():
        old = pl.read_parquet(exp / f"{name}_{month}.parquet")
        j = tbl.join(old, on=KEYS, suffix="_o")
        diffs = {c: float((j[c] - j[c + "_o"]).abs().max()) for c in [*QCOLS, "observed_power", "nra_power"]}
        print(name, "rows", tbl.height, old.height, j.height, "max abs diff", max(diffs.values()), "gate_on equal", bool((j["gate_on"].fill_null(False) == j["gate_on_o"].fill_null(False)).all()))


def main(month: str) -> None:
    cfg = json.loads((HERE / "freeze_config.json").read_text())
    assert not (HERE / "final_test_620.lock").exists() or os.environ.get("E00620_PHASE") == "predict"
    roots = {"K1": R / "arms/k1/work", "Candidate": R / "P7_D50_e3/work", "K3": R / "arms/k3/work"}
    out = R / "export"
    write(tables=assemble(month=month, roots=roots, days=("d0",), factor=cfg["candidate_width_factor"], mode=None, out=out), month=month, out=out)


if __name__ == "__main__":
    if sys.argv[1] == "--dev-check":
        dev_check(sys.argv[2])
    else:
        main(sys.argv[1])
