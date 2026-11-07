# ruff: noqa: E501, E402, PT018, D103
"""Experiment 619: the lag-enabled demand network at the 4 BSP/GSP series, pooled per group (GPU).

Wraps experiment 574's `e00574_fit.py` arm P4E (lags at least 24 h old, the D5 output offset,
three-step schedule with a 3-day embargo, frozen encoders) and swaps in experiment 591's shared lag
encoder with lag ID (arm A1, no auxiliary loss). Differences from the Primaries' K1 recipe: no B1
thermal-lag and solar-gain columns (experiment 437's stored values exist only for the Primaries), no
gate, no synthetic load-shifts, no E3 weighting. The harness is pointed at one group of 2 series by
`e00611_pool.install` with this experiment's folder as the work folder.

Usage: python e00619_fit.py {pretrain-wt|step2|fit|assemble} --group {bsp|gsp} [--months ...] [--seeds ...] [--limit N]
"""

import argparse
import dataclasses
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("pretrain-wt", "step2", "fit", "assemble"))
_p.add_argument("--group", choices=("bsp", "gsp"), required=True)
_p.add_argument("--months", nargs="+", default=None)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--limit", type=int, default=None)
MINE = _p.parse_args()
_argv = ["e00574_fit.py", MINE.command, "--arm", "P4E", "--seeds", *map(str, MINE.seeds)]
if MINE.months:
    _argv += ["--months", *MINE.months]
if MINE.limit is not None:
    _argv += ["--limit", str(MINE.limit)]
sys.argv = _argv

import e00510_common as c  # ty: ignore[unresolved-import]
import e00574_fit as e  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import e00611_pool as pool  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00619-bsp-gsp-lag-enabled-network-for-the-bake-off"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
TAG = "P4E"
pool.WORK = ROOT
assert set(e.ARGS.months) <= set(c.MONTHS), e.ARGS.months
W = pool.install(group=MINE.group)
W.mkdir(parents=True, exist_ok=True)
e.WORK, e.W, e.W560 = ROOT, W, W
e.SPEC = dataclasses.replace(e.SPEC, pretrain_from="LAGG")
e.SPECS[TAG] = e.SPEC
assert e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen and not e.SPEC.b1
assert e.N_B1 == 0 and not e.SPEC.wt


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = aux.AuxLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, kind="shared"
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


def assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
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
                first.day <= 2
                and (last.replace(day=28) + __import__("datetime").timedelta(days=4)).month
                != last.month
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
        (ROOT / f"checks_fit_{MINE.group}_s{'-'.join(map(str, e.ARGS.seeds))}.json").write_text(
            __import__("json").dumps(e.CHECK, indent=1, default=str)
        )
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
