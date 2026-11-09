# ruff: noqa: E402, E501, PT018, D103, D205
"""Experiment 669: the lag-enabled demand network with thermal-lag columns at the 4 BSP/GSP series,
one monthly fold at a time, with a lag cut of 3.5 hours before the network's 09:00 origin (GPU).

A copy of experiment 633's `e00633_node_fit.py` (the arm `lagged_b1` of experiment 622) with three
changes: the lag cap is experiment 572's `3.5h` (7 half-hours, so the newest power read is the
05:30 reading on the origin day, 6.5 hours before the contract's 12:00 origin), the fold is any of
the 12 months of the year (`e00669_fold`), and the forecast keeps leads to 351 hours.

Usage: python e00669_fit.py {pretrain-wt|step2|fit|assemble} --group {bsp|gsp} --months 2025-11 ... [--seeds 0 1]
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("pretrain-wt", "step2", "fit", "assemble"))
_p.add_argument("--group", choices=("bsp", "gsp"), required=True)
_p.add_argument("--months", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--limit", type=int, default=None)
MINE = _p.parse_args()
sys.argv = [
    "e00574_fit.py", MINE.command, "--arm", "P5", "--seeds", *map(str, MINE.seeds),
    "--months", *MINE.months,
]  # fmt: skip
if MINE.limit is not None:
    sys.argv += ["--limit", str(MINE.limit)]

import e00510_common as c  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import e00611_pool as pool  # ty: ignore[unresolved-import]
import e00633_common as k633
import e00669_common as k
import e00669_fold as fold
import polars as pl
import torch  # ty: ignore[unresolved-import]

ROOT = k.ROOT
pool.WORK = ROOT
W = pool.install(group=MINE.group)
import e00437_features as f437  # ty: ignore[unresolved-import]

f437.TRAILING_DIR = k633.ROOT / "inputs" / "trailing437_nodes"
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00574_fit as e  # ty: ignore[unresolved-import]

assert set(e.ARGS.months) <= set(k.MONTHS), e.ARGS.months
W.mkdir(parents=True, exist_ok=True)
e.WORK, e.W = ROOT, W  # e.W560 is set by fold.install (shared, read-only)
fold.install(e, months=e.ARGS.months, group=MINE.group)
e.SPEC = dataclasses.replace(e.SPEC, tag=k.TAG, pretrain_from=k.TAG, lags="3.5h")
e.SPECS[k.TAG] = e.SPEC
e.CAP_HOURS = e.SPEC.cap_steps * 0.5
assert e.SPEC.cap_steps == lg.CAP_3_5H == 7 and e.CAP_HOURS == 3.5
assert e.SPEC.sched == "3sE" and e.SPEC.frozen and not e.SPEC.wt and e.SPEC.b1 == "B1"
assert e.cols569.width(arm="B1") == e.N_B1, e.N_B1
assert e.EX is not None and set(e.EX.rows) == {0, 1}
assert [r["site"] for r in e.EX.rows.values()] == list(pool.GROUPS[MINE.group])


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = aux.AuxLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, kind="shared"
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


def assemble() -> None:
    """Concatenate each fold's forecast into one table per month and seed."""
    for month in e.ARGS.months:
        for seed in e.ARGS.seeds:
            part = W / "fits" / k.TAG / f"{month}_d0_s{seed}.parquet"
            assert part.exists(), part
            frame = pl.read_parquet(part)
            start = fold.month_start(month=month)
            days = (c.month_stop(month=month) - start).days
            assert frame["power_fcst_init_time"].n_unique() == days, (month, seed)
            ids = pool.group_table()["time_series_id"].to_list()
            assert sorted(frame["time_series_id"].unique().to_list()) == sorted(ids), (month, seed)
            n_origins = frame.group_by("time_series_id").agg(
                pl.col("power_fcst_init_time").n_unique()
            )
            assert n_origins["power_fcst_init_time"].n_unique() == 1, (month, seed, n_origins)
            per = frame.group_by("time_series_id", "power_fcst_init_time").len()["len"]
            assert per.min() == per.max() == 702, (month, seed, per.min(), per.max())
            lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_hours()
            assert lead.max() <= 351 and lead.min() >= 0  # ty: ignore[unsupported-operator]
            e.f.atomic_frame(
                frame=frame, target=W / "seed_preds" / k.TAG / f"{month}_s{seed}.parquet"
            )
            print("wrote", k.TAG, month, seed, frame.height, flush=True)


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
        tag = f"{MINE.group}_{k.TAG}_{'-'.join(MINE.months)}_s{'-'.join(map(str, MINE.seeds))}"
        (ROOT / "checks").mkdir(exist_ok=True)
        (ROOT / "checks" / f"fit_{tag}.json").write_text(json.dumps(e.CHECK, indent=1, default=str))
        print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
