"""Experiment 664: the per-member solar curves (arm d's physics), one table per cutoff.

Experiment 655 stored only the member average of each curve. This script recomputes the curve of
every one of the 51 ensemble members with experiment 655's physics (`e00655_curves._unit`: cell
temperature scale `s` fitted per cutoff, tilt 20 degrees, DC/AC 1.15), divides by experiment 655's
normalising constant (`Bank.norm["d"]`), and stores a (runs, 702, 51) float32 table per Primary.

Usage: python e00664_members.py build --cutoff 2026-06
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00634_common as k634  # ty: ignore[unresolved-import]
import e00655_curves as cv  # ty: ignore[unresolved-import]
import e00655_fit as fit655  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00664-pv-sum-of-distributions-spread-and-recalibration"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
MEMBERS: Final[Path] = WORK / "members"


def table_path(*, cutoff: str, primary: int) -> Path:
    """Return where one Primary's normalised member curves are stored for a cutoff."""
    return MEMBERS / cutoff / f"p{primary}.npy"


def build_one(*, cutoff: str, primary: int, s: float, norm: float) -> None:
    """Compute and store the normalised curve of every member for one Primary."""
    sw, t, w = cv.load_members(primary=primary)
    n_runs = sw.shape[0]
    z, a, doy = cv.geometry(primary=primary, run_index=np.arange(n_runs))
    out = np.empty(sw.shape, dtype=np.float32)
    for lo in range(0, n_runs, 20):
        hi = min(lo + 20, n_runs)
        zz, aa, dd = (x[lo:hi, :, None] for x in (z, a, doy))
        cell = cv._unit(
            ghi=sw[lo:hi].astype(np.float64), temp_air=t[lo:hi].astype(np.float64),
            wind=w[lo:hi].astype(np.float64), s=s, zenith_deg=zz, sun_az_deg=aa, doy=dd,
        )  # fmt: skip
        out[lo:hi] = (cell / norm).astype(np.float32)
    path = table_path(cutoff=cutoff, primary=primary)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, out)
    tmp.rename(path)


def main() -> None:
    """Build the member tables of one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--cutoff", required=True)
    a = parser.parse_args()
    s = json.loads(cv.scale_path(cutoff=a.cutoff).read_text())["s_mean"]
    norm = fit655.Bank(cutoff=a.cutoff).norm["d"]
    print(f"cutoff {a.cutoff}: s={s:.4f}, norm={norm:.5f}", flush=True)
    for primary in k634.six_indices().values():
        if table_path(cutoff=a.cutoff, primary=primary).exists():
            continue
        build_one(cutoff=a.cutoff, primary=primary, s=s, norm=norm)
        print(f"p{primary} done", flush=True)


if __name__ == "__main__":
    main()
