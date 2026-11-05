# ruff: noqa: E501, ANN001, ANN201, D103, PT018
"""Experiment 584: score arms C0 and C1 (and C2 to C4 built from them) with experiment 583's code.

No fit and no GPU. Experiment 583's scoring (`e00583_score.main`) reads one arm's stored forecasts from
`e00583_common.X3SYN` (fits `V3D5` and `V3D5-removed`, injected-bank folder `inject`). This script
builds a symlink view of each arm in that layout, points experiment 583's module constants at it, and
adds the width-matched controls as extra building blocks:

- `C4`: the arm's lag-network band scaled about its median by one constant factor, chosen so that its
  mean q05-q95 width on event-free rows at 24-120 h equals that of the gated band (experiment 583's A2).
- `C4u`: the same, matched to the always-on union band instead.

Outputs per arm under `data/ml-experiments/00584-*/score_<arm>/`: experiment 583's tables, plus
`row_arrays.npz` (quantiles and keys of every scored row, for the paired contrasts and CRPS).

Usage: python e00584_score.py ARM [FACTOR]   (ARM: c0 or c1; FACTOR overrides `C4u`'s factor, post hoc)
"""

import sys
from pathlib import Path

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00584-nra-candidate-pack-x6"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
ARMS = {
    "c0": {
        "fits": ML_EXPERIMENTS_DATA_DIR
        / "00572-two-offset-part-1-lags-versus-the-slow-output-offset/work/fits",
        "fit_tag": "V3D5",
        "inject": ML_EXPERIMENTS_DATA_DIR
        / "00581-robust-multiweek-lag-encoder-x3/arms/v3d5/work/inject",
    },
    "p5": {  # diagnostic: experiment 574's P5 without synthetic lags; the bank folder is a placeholder (its bank outputs are discarded)
        "fits": ML_EXPERIMENTS_DATA_DIR
        / "00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/work/fits",
        "fit_tag": "P5",
        "inject": ROOT / "c1/work/inject",
    },
    "c1": {
        "fits": ROOT / "c1/work/fits",
        "fit_tag": "P5",
        "inject": ROOT / "c1/work/inject",
    },
}
FACTORS: dict[str, float] = {}
OVERRIDE: float | None = None
"""Post hoc (experiment 584): a fixed factor for `C4u`, used to match the gated band's width on after-end rows (`C4w`)."""
_orig_building_blocks, _orig_combine, _orig_init = c.building_blocks, c.combine, c.X4Table.__init__


def link_view(*, arm: str) -> Path:
    """Return a folder laid out as experiment 583 expects, with symlinks to the arm's stored files."""
    spec = ARMS[arm]
    view = ROOT / f"view_{arm}"
    (view / "fits").mkdir(parents=True, exist_ok=True)
    for tag, src in (("V3D5", spec["fit_tag"]), ("V3D5-removed", f"{spec['fit_tag']}-removed")):
        dst = view / "fits" / tag
        if dst.is_symlink():
            dst.unlink()
        dst.symlink_to(spec["fits"] / src)
        assert dst.exists() and any(dst.iterdir()), dst
    inj = view / "inject"
    if inj.is_symlink():
        inj.unlink()
    inj.symlink_to(spec["inject"])
    assert any(inj.iterdir()), inj
    return view


def width90(q: np.ndarray) -> np.ndarray:
    return q[:, c.IQ[0.95]] - q[:, c.IQ[0.05]]


def scale_about_median(*, q: np.ndarray, factor: float) -> np.ndarray:
    med = q[:, c.IQ[0.5] : c.IQ[0.5] + 1]
    return med + factor * (q - med)


def init_with_factors(self) -> None:
    """Build the table, then fit the two width factors on event-free rows at 24-120 h."""
    _orig_init(self)
    gi = c.gate_inputs(t=self)
    od = gi["origin_days"]
    on, snap = c.gate_state(gap=gi["gap"], thr=gi["thr"], origin_days=od, release_days=1)
    hold = c.hold_from(snap=snap)
    near = s.dilated_near(near=gi["near"])
    rows = np.flatnonzero(self.free & ~near[self.prim, self.origin_day] & (self.band == 1))
    v, m = self.arms["V"][rows], self.arms["M"][rows]
    union = a.union_band(lag=v, masked=m)
    gated = (hold | on)[self.prim[rows], self.origin_day[rows]]
    a2 = np.where(gated[:, None], union, v)
    FACTORS["C4"] = float(width90(a2).mean() / width90(v).mean())
    FACTORS["C4u"] = float(width90(union).mean() / width90(v).mean())
    FACTORS["n_rows"] = len(rows)
    print("width factors", FACTORS, flush=True)


def building_blocks(*, v, m):
    blocks = _orig_building_blocks(v=v, m=m)
    blocks["C4"] = scale_about_median(q=v, factor=FACTORS["C4"])
    blocks["C4u"] = scale_about_median(q=v, factor=OVERRIDE or FACTORS["C4u"])
    return blocks


def combine(*, base, hold, on, frac=None):
    out = _orig_combine(base=base, hold=hold, on=on, frac=frac)
    out["C4"], out["C4u"] = base["C4"], base["C4u"]
    return out


def setup(arm: str, override: float | None = None) -> Path:
    """Point experiment 583's modules at arm `arm`'s stored forecasts and add the control bands; return the output folder."""
    global OVERRIDE  # noqa: PLW0603
    OVERRIDE = override
    view = link_view(arm=arm)
    suffix = f"_w{override}" if override else ""
    c.X3SYN, c.INJECT, c.WORK = view, view / "inject", ROOT / f"score_{arm}{suffix}"
    c.X4Table.__init__ = init_with_factors
    c.building_blocks, c.combine = building_blocks, combine
    s.BANK_PARQUET = next(
        (ML_EXPERIMENTS_DATA_DIR / "00581-robust-multiweek-lag-encoder-x3").glob(
            "bank_events_x3syn_*.parquet"
        )
    )
    return c.WORK


def main(arm: str, override: float | None = None) -> None:
    setup(arm, override)
    s.main()
    t = c.X4Table()
    np.savez(
        c.WORK / "row_arrays.npz",
        V=t.arms["V"], M=t.arms["M"], prim=t.prim, idx=t.idx, origin_idx=t.origin_idx,
        lead=t.lead, nra=t.nra, obs=t.obs, free=t.free, month=t.month, group=t.group_of_row,
        factors=np.array([FACTORS["C4"], FACTORS["C4u"]]),
    )  # fmt: skip
    print("saved arrays", t.n, flush=True)


if __name__ == "__main__":
    factor = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(sys.argv[1], factor)
