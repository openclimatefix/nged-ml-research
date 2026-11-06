# ruff: noqa: E501, D103, D107
"""Experiment 593: package the stored backtests of the candidate normal-running-arrangement forecast.

No fit and no GPU. Every arm is a stored forecast of experiments 565, 572, 574, or 584 (or is built
from them with experiment 583's code), averaged over seeds 0 and 1 quantile by quantile, as
experiments 582 to 584 score them. One Parquet file is written per arm and fold month (named by the
month of the forecast origin) under `data/ml-experiments/backtest_export_2026-11-06/`.

Arms:

- `C0`: experiment 572's v3+D5 (`fits/V3D5`).
- `C1`: experiment 584's P5 recipe with lag-only synthetic load-shifts (`c1/work/fits/P5`).
- `C2`: C1 with the gated band: the union band of C1 and its lags-masked twin (`fits/P5-removed`)
  on the origin days where experiment 583's gate is on or held, C1 elsewhere.
- `C3`: the union band on every row.
- `E2`: the lag-free D5 network of experiment 565 (experiment 572 calls it E2).
- `P5plain`: experiment 574's P5 recipe without synthetic lags (`fits/P5`).

Usage: python e00593_export.py [MONTH ...]   (default: all five fold months)
"""

import sys
from pathlib import Path

import e00048_fit as fd  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k582  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "backtest_export_2026-11-06"
ROOT584 = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6"
MONTHS = c.MONTHS
QCOLS = list(a.QCOLS)
KEYS = list(a.KEYS)
GRID = (16, 121248)
SOURCES = {  # arm -> (folder holding `fits/`, fit tag)
    "C0": (ML_EXPERIMENTS_DATA_DIR / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work", "V3D5"),
    "C1": (ROOT584 / "c1/work", "P5"),
    "E2": (a.K565_W, "D5"),
    "P5plain": (ML_EXPERIMENTS_DATA_DIR / "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/work", "P5"),
}  # fmt: skip


def load(*, root: Path, tag: str, month: str) -> pl.DataFrame:
    """Return the seed-mean quantiles of one stored arm for one fold month, leads under 120 h."""
    return a._load(root=root, tag=tag, months=(month,)).filter(
        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
        < 60 * a.MAX_LEAD_H
    )


def sorted_q(df: pl.DataFrame, *, prefix: str = "") -> np.ndarray:
    return np.sort(df.select([prefix + q for q in QCOLS]).to_numpy().astype(np.float32), axis=1)


class Context:
    """Per-Primary lookups on the half-hour grid: power, events, gate, capacity."""

    def __init__(self) -> None:
        sm = k578.sites()
        series, arrays = fd._series_and_arrays()
        series = series.filter(pl.col("substation_type") == "Primary")
        self.sites = sm.join(
            series.select(pl.col("site").alias("label"), pl.col("units").alias("unit")),
            on="label",
            how="left",
        )
        assert self.sites["unit"].null_count() == 0
        self.prim_of = dict(
            zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True)
        )
        self.cap = sm["capacity_mw"].to_numpy().astype(np.float64)
        self.unit = np.array(self.sites["unit"].to_list())
        labels = sm["label"].to_list()
        self.clean = k578.clean()
        self.raw = np.stack([arrays[lab].power for lab in labels]).astype(np.float32)
        self.shift = k582.verified_shift(shape=self.clean.shape)
        self.reason = self._drop_reasons(sm=sm)
        self._events(sm=sm)
        self._gate()

    def _drop_reasons(self, *, sm: pl.DataFrame) -> np.ndarray:
        fl = pl.read_parquet(stale.FLAGGED_KEYS)
        out = np.full(self.clean.shape, "", dtype=object)
        idx = (
            (fl["valid_time"].dt.epoch("us") - k578.prep.GRID_START_US) // k578.prep.STEP_US
        ).to_numpy()
        prim = np.array([self.prim_of.get(t, -1) for t in fl["time_series_id"].to_list()])
        keep = (prim >= 0) & (idx >= 0) & (idx < out.shape[1])
        out[prim[keep], idx[keep]] = fl["drop_reason"].to_numpy()[keep]
        return out

    def _events(self, *, sm: pl.DataFrame) -> None:
        n_grid = self.clean.shape[1]
        self.verified_id = np.full(self.clean.shape, -1, np.int32)
        for r in k582.verified_events().iter_rows(named=True):
            self.verified_id[r["primary"], r["onset_idx"] : r["end_idx"]] = r["transfer_id"]
        ev = pl.read_parquet(k578.EVENTS)
        status = pl.read_parquet(k582.V580 / "events_labelled_status.parquet")
        verified_labels = set(
            status.filter(pl.col("final_class") == "verified transfer")["event_id"].to_list()
        )
        lab2p = dict(zip(sm["label"].to_list(), sm["primary"].to_list(), strict=True))
        self.label_id = np.full(self.clean.shape, -1, np.int32)
        self.label_ok = np.zeros(self.clean.shape, bool)
        near = np.zeros(self.clean.shape, bool)
        pad = 7 * 48
        for r in ev.iter_rows(named=True):
            p = lab2p.get(r["label"])
            if p is None:
                continue
            self.label_id[p, r["onset_idx"] : r["end_idx"]] = r["event_id"]
            self.label_ok[p, r["onset_idx"] : r["end_idx"]] = r["event_id"] in verified_labels
            near[p, max(r["onset_idx"] - pad, 0) : r["end_idx"] + pad] = True
        for r in k582.verified_events().iter_rows(named=True):
            near[r["primary"], max(r["onset_idx"] - pad, 0) : r["end_idx"] + pad] = True
        self.event_free = ~near
        assert self.event_free.shape[1] == n_grid

    def _gate(self) -> None:
        """Experiment 583's gate state on C1's table: True where the gate is on or held."""
        sc.setup("c1")
        t = c.X4Table()
        gi = c.gate_inputs(t=t)
        on, snap = c.gate_state(
            gap=gi["gap"], thr=gi["thr"], origin_days=gi["origin_days"], release_days=1
        )
        self.gated = c.hold_from(snap=snap) | on  # (16, origin day)
        self.t_event_free_share = float(t.free.mean())

    def table(self, *, df: pl.DataFrame, q: np.ndarray, gate: bool) -> pl.DataFrame:
        """Return the export table of one arm and month (forecasts and observations in MW or MVA)."""
        prim = df["time_series_id"].replace_strict(self.prim_of, return_dtype=pl.Int64).to_numpy()
        valid = (
            (df["valid_time"].dt.epoch("us") - k578.prep.GRID_START_US) // k578.prep.STEP_US
        ).to_numpy()
        origin = (
            (df["power_fcst_init_time"].dt.epoch("us") - k578.prep.GRID_START_US)
            // k578.prep.STEP_US
        ).to_numpy()
        cap = self.cap[prim]
        obs = self.clean[prim, valid]
        nra = obs + self.shift[prim, valid]
        reason = self.reason[prim, valid]
        vid, lid = self.verified_id[prim, valid], self.label_id[prim, valid]
        out = df.select(KEYS).with_columns(
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ).cast(pl.Float32),
            **{
                name: pl.Series(q[:, i].astype(np.float64) * cap).cast(pl.Float32)
                for i, name in enumerate(QCOLS)
            },
            observed_power=pl.Series(obs.astype(np.float64) * cap).cast(pl.Float32),
            observed_power_raw=pl.Series(self.raw[prim, valid].astype(np.float64) * cap).cast(
                pl.Float32
            ),
            drop_reason=pl.when(pl.Series(reason == "")).then(None).otherwise(pl.Series(reason.astype(str))),
            nra_power=pl.Series(nra.astype(np.float64) * cap).cast(pl.Float32),
            unit=pl.Series(self.unit[prim]),
            gate_on=(
                pl.Series(self.gated[prim, origin // 48]) if gate else pl.lit(None, dtype=pl.Boolean)
            ),
            in_verified_event=pl.Series(vid >= 0),
            event_id=pl.when(pl.Series(vid >= 0)).then(pl.Series(vid)).otherwise(None),
            labelled_event_id=pl.when(pl.Series(lid >= 0)).then(pl.Series(lid)).otherwise(None),
            label_verified=pl.when(pl.Series(lid >= 0)).then(pl.Series(self.label_ok[prim, valid])).otherwise(None),
            event_free=pl.Series(self.event_free[prim, valid]),
        )  # fmt: skip
        return out.sort(KEYS)


def write(*, tbl: pl.DataFrame, arm: str, month: str) -> None:
    path = OUT / f"{arm}_{month}.parquet"
    tbl.write_parquet(path, compression="zstd", compression_level=9, statistics=True)
    print(arm, month, tbl.height, f"{path.stat().st_size / 1e6:.1f} MB", flush=True)


def main(months: tuple[str, ...]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    ctx.sites.select(
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
    ).sort("time_series_id").write_csv(OUT / "substations.csv")
    for month in months:
        for arm, (root, tag) in SOURCES.items():
            df = load(root=root, tag=tag, month=month).sort(KEYS)
            write(tbl=ctx.table(df=df, q=sorted_q(df), gate=False), arm=arm, month=month)
            if arm != "C1":
                continue
            tw = load(root=root, tag="P5-removed", month=month)
            both = df.join(tw.rename({q: f"M_{q}" for q in QCOLS}), on=KEYS, how="inner").sort(KEYS)
            assert both.height == df.height, (both.height, df.height)
            v, m = sorted_q(both), sorted_q(both, prefix="M_")
            union = a.union_band(lag=v, masked=m)
            prim = (
                both["time_series_id"].replace_strict(ctx.prim_of, return_dtype=pl.Int64).to_numpy()
            )
            origin = (
                (both["power_fcst_init_time"].dt.epoch("us") - k578.prep.GRID_START_US)
                // k578.prep.STEP_US
            ).to_numpy()
            gated = ctx.gated[prim, origin // 48]
            c2 = np.where(gated[:, None], union, v)
            base = both.select(KEYS)
            write(tbl=ctx.table(df=base, q=c2, gate=True), arm="C2", month=month)
            write(tbl=ctx.table(df=base, q=union, gate=False), arm="C3", month=month)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or MONTHS)
