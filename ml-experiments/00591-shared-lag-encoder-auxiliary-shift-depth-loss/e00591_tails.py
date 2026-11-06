# ruff: noqa: D103, E501, PT018, B023
"""Experiment 591: lag-following and tail metrics by phase on experiment 581's injected bank, for arms A0 to A5.

Experiment 581's `e00581_tails.event_table` patched, not copied. It reads an arm's injected forecasts
(`inject/`) and the same weights' forecasts with the observed lags (`fits/V3D5/`, the "uninjected
twin"). Our arms store the latter under `fits/P5/`, so a folder of symlinks per arm presents them under
the name the module expects. Arm `a0` is experiment 584's C1.

Usage: python e00591_tails.py ARM [ARM ...]   (writes `tails/events_<arm>.parquet`)
"""

import sys
from pathlib import Path

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00581_tails as t  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
C1 = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/c1/work"


def source(arm: str) -> Path:
    return C1 if arm == "a0" else ROOT / arm.upper() / "work"


def view(arm: str) -> Path:
    """Return the symlink folder that presents arm `arm` in experiment 581's layout."""
    v = ROOT / "tails_view" / arm / "work"
    (v / "fits").mkdir(parents=True, exist_ok=True)
    for dst, src in (
        (v / "fits" / "V3D5", source(arm) / "fits" / "P5"),
        (v / "inject", source(arm) / "inject"),
    ):
        if dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        assert dst.exists() and any(dst.iterdir()), dst
    return v


def main(arms: list[str]) -> None:
    sites = k578.sites()
    clean = k578.clean()
    out_dir = ROOT / "tails"
    out_dir.mkdir(parents=True, exist_ok=True)
    free_all = {
        m: pl.concat(
            [
                pl.read_parquet(
                    t.W565 / "fits" / "D5" / f"{m}_{d}_s{s}.parquet", columns=[*t.KEYS, *t.TAUS]
                )
                for d in ("d0", "d14")
                for s in t.SEEDS
            ]
        )
        .group_by(t.KEYS)
        .agg([pl.col(q).mean() for q in t.TAUS])
        for m in t.MONTHS
    }
    for arm in arms:
        target = out_dir / f"events_{arm}.parquet"
        if target.exists():
            continue
        v = view(arm)
        t.arm_root = lambda a, month="": v
        t.inject_dir = lambda a: v / "inject"
        tabs = []
        for m in t.MONTHS:
            tabs += [
                t.event_table(arm=arm, month=m, r=r, sites=sites, clean=clean, free=free_all[m])
                for r in range(bank.N_PASSES)
            ]
            print(arm, m, "done", flush=True)
        tmp = target.with_suffix(".tmp")
        pl.concat(tabs).write_parquet(tmp)
        tmp.rename(target)


if __name__ == "__main__":
    main(sys.argv[1:])
