# ruff: noqa: E402, PT018, D103
"""Experiment 622: matched lag-free and lag-enabled demand networks at the 4 BSP/GSP series (GPU).

Wraps experiment 619's `e00619_fit.py` (experiment 574's arm P4E or P5 with experiment 591's shared
lag encoder, pooled per group of 2 series). Two switches, so that arms differ in one factor:

- `--lags off` replaces the day-lag and week-lag columns of every row (pre-training, fine-tuning,
  scoring) with their "absent" encoding, so the network never sees a power lag. Everything else,
  including the lag encoder's parameters, is identical to the lag-enabled arm.
- `--b1 on` adds experiment 569's thermal-lag and solar-gain columns (B1) through the weather
  encoder, as in experiment 574's arm P5, from the tables `e00622_b1.py` builds for the 4 series.

Both arms use the D5 output offset, the three-step schedule with a 3-day embargo, and frozen
encoders. Each arm pre-trains its own weights.

Usage: python e00622_fit.py {pretrain-wt|step2|fit|assemble} --group {bsp|gsp} --lags {on,off}
--b1 {on,off} [--months ...] [--seeds ...] [--limit N]
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("pretrain-wt", "step2", "fit", "assemble"))
_p.add_argument("--group", choices=("bsp", "gsp"), required=True)
_p.add_argument("--lags", choices=("on", "off"), required=True)
_p.add_argument("--b1", choices=("on", "off"), required=True)
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--limit", type=int, default=None)
MINE = _p.parse_args()
_argv = [
    "e00574_fit.py", MINE.command, "--arm", "P5" if MINE.b1 == "on" else "P4E",
    "--seeds", *map(str, MINE.seeds),
]  # fmt: skip
if MINE.months:
    _argv += ["--months", *MINE.months]
if MINE.limit is not None:
    _argv += ["--limit", str(MINE.limit)]
sys.argv = _argv

import e00510_common as c  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import e00611_pool as pool  # ty: ignore[unresolved-import]
import e00622_b1 as b1tables  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

NAME = b1tables.NAME
ROOT = b1tables.ROOT
TAG = f"J_{'LG' if MINE.lags == 'on' else 'LF'}_{'B1' if MINE.b1 == 'on' else 'NB'}"
pool.WORK = ROOT
W = pool.install(group=MINE.group)
b1tables.install()
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00574_fit as e  # ty: ignore[unresolved-import]

assert set(e.ARGS.months) <= set(c.MONTHS), e.ARGS.months
W.mkdir(parents=True, exist_ok=True)
e.WORK, e.W, e.W560 = ROOT, W, W
e.SPEC = dataclasses.replace(e.SPEC, tag=TAG, pretrain_from=TAG)
e.SPECS[TAG] = e.SPEC
assert e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen and not e.SPEC.wt
assert (e.cols569.width(arm="B1") if MINE.b1 == "on" else 0) == e.N_B1, e.N_B1
if MINE.b1 == "on":
    assert e.EX is not None and set(e.EX.rows) == {0, 1}
    assert [r["site"] for r in e.EX.rows.values()] == list(pool.GROUPS[MINE.group])


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = aux.AuxLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, kind="shared"
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model
_block_of = e.block_of


def block_of(**kwargs) -> np.ndarray:  # noqa: ANN003
    """Return the 8 lag columns; with `--lags off` the value, missing, and age say 'absent'."""
    blk = _block_of(**kwargs)
    if MINE.lags == "off":
        amax = lg.age_max(cap_steps=e.SPEC.cap_steps)
        for j in range(2):
            blk[:, 3 * j], blk[:, 3 * j + 1], blk[:, 3 * j + 2] = 0.0, 1.0, amax[j]
    return blk


e.block_of = block_of


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    import datetime as dt

    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        for seed in e.ARGS.seeds:
            parts = [
                W / "fits" / TAG / f"{month}_d{e.f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            first = frame["power_fcst_init_time"].min().date()  # ty: ignore[unresolved-attribute]
            last = frame["power_fcst_init_time"].max().date()  # ty: ignore[unresolved-attribute]
            assert f"{first:%Y-%m}" == month == f"{last:%Y-%m}", (month, seed, first, last)
            assert (
                first.day <= 2 and (last.replace(day=28) + dt.timedelta(days=4)).month != last.month
            ) or last.day >= 28, (month, seed, first, last)
            ids = pool.group_table()["time_series_id"].to_list()
            assert sorted(frame["time_series_id"].unique().to_list()) == sorted(ids), (month, seed)
            n_origins = frame.group_by("time_series_id").agg(
                pl.col("power_fcst_init_time").n_unique()
            )
            assert n_origins["power_fcst_init_time"].n_unique() == 1, (month, seed, n_origins)
            lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_hours()
            assert lead.max() <= 240 and lead.min() >= 0  # ty: ignore[unsupported-operator]
            e.f.atomic_frame(
                frame=frame, target=W / "seed_preds" / TAG / f"{month}_s{seed}.parquet"
            )
            print("wrote", TAG, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if MINE.command == "assemble":
        assemble()
    elif MINE.command == "pretrain-wt":
        e.pretrain_wt(device="cuda")
    elif MINE.command == "step2":
        e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    else:
        e.fit_arm(
            months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=MINE.limit, device="cuda"
        )
        (ROOT / f"checks_fit_{MINE.group}_{TAG}_s{'-'.join(map(str, MINE.seeds))}.json").write_text(
            json.dumps(e.CHECK, indent=1, default=str)
        )
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
