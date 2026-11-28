"""Report-only diagnostic of A1's misses on the 2025-11 injected rows (CPU only, no fit).

A1 is the age rule of `e00738_survival.age_rule_a1`. The diagnostic changes no bar and no threshold.
It rebuilds the rows of `task_a1_table` for one fold (the same shifts, the same origins) and writes
`a1/diagnostic_<month>.json` under the output folder, and prints the tables:

1. A1's miss rate by true age band crossed with the shift's depth in noise units.
2. The share of in-force rows, and of in-force depth-weighted mass, that A1 sees, by depth band.
3. The miss rate against the false-alarm share over the threshold grid.
4. The depth of the injected shifts, against the 66 hand-filtered real transfers of experiment 101.
5. How well a soft input (the deviation of the newest history rows from the reference, in noise
   units) separates in-force rows from real rows, by depth band, as the area under the ROC curve.

Run it as `python e00738_a1_diagnostic.py <month>` with `E00667_MONTH` and `E00667_PHASE=train`
set (`run.sh` does not start it; use the scope in the experiment's usual rules).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Final

import e00738_common as common
import numpy as np

DEPTH_BANDS_NOISE: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 1", 0.0, 1.0),
    ("1 to 2", 1.0, 2.0),
    ("2 to 4", 2.0, 4.0),
    ("over 4", 4.0, 1e9),
)
"""Depth bands in noise units: name, lower bound (inclusive), upper bound (exclusive)."""
AGE_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    *common.A1_BANDS,
    ("3 to 10", 3.0, 11.0),
)
N_SHIFT_PASSES: Final[int] = 10
Z_REPORTED: Final[float] = 3.5
"""The threshold that Phase 0 tuned on the real training rows of fold 2025-11 (report only)."""


def auc(*, positive: np.ndarray, negative: np.ndarray) -> float:
    """Return the probability that a random positive score exceeds a random negative one."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="stable")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    both = np.concatenate([positive, negative])
    # average the ranks of ties
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n_p = positive.size
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * negative.size))


def soft_score(*, hist_power: np.ndarray, visible: np.ndarray, rows: int = 3) -> np.ndarray:
    """Return the mean absolute deviation of the newest `rows` rows from the reference level.

    The reference level is the mean of the oldest 7 history rows, as in A1. The score is in the
    units of the input power (divide by the noise unit for noise units). Rows with fewer than
    `A1_MIN_REFERENCE_ROWS` reference rows score NaN.
    """
    import e00738_survival as surv

    means, present = surv.row_summaries(hist_power=hist_power, visible=visible)
    ref_rows = slice(common.N_HISTORY // 2, common.N_HISTORY)
    n_ref = present[:, ref_rows].sum(axis=1)
    ref = np.where(present[:, ref_rows], means[:, ref_rows], 0.0).sum(axis=1) / np.maximum(n_ref, 1)
    dev = np.abs(means[:, :rows] - ref[:, None])
    ok = present[:, :rows].all(axis=1) & (n_ref >= common.A1_MIN_REFERENCE_ROWS)
    return np.where(ok, dev.mean(axis=1), np.nan)


def band_mask(*, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return the half-open band `lo <= values < hi`."""
    return (values >= lo) & (values < hi)


def tables(
    *,
    inj: tuple[np.ndarray, np.ndarray],
    real: tuple[np.ndarray, np.ndarray],
    true_age: np.ndarray,
    depth_noise: np.ndarray,
    noise: float,
    z_main: float,
    grid: tuple[float, ...] = common.A1_Z_GRID,
) -> dict[str, Any]:
    """Return tables 1, 2, 3 and 5 from the history rows of the injected and real panels.

    Args:
        inj: The injected panel's (history power, visibility) per row.
        real: The real record's (history power, visibility) per row.
        true_age: Days from the onset of the in-force shift to the origin (-1 where none).
        depth_noise: The in-force shift's depth in noise units (0 where none).
        noise: The noise unit.
        z_main: The threshold of tables 1 and 2.
        grid: The thresholds of table 3.

    Returns:
        The tables as nested dicts.
    """
    import e00738_survival as surv

    def age(panel: tuple[np.ndarray, np.ndarray], z: float) -> np.ndarray:
        return surv.age_rule_a1(hist_power=panel[0], visible=panel[1], noise=noise, z=z)

    seen = age(inj, z_main) > 0
    in_force = true_age >= 0
    out: dict[str, Any] = {"z": z_main, "n_in_force_rows": int(in_force.sum())}
    cross: dict[str, dict[str, Any]] = {}
    for a_name, a_lo, a_hi in AGE_BANDS:
        in_age = band_mask(values=true_age, lo=a_lo, hi=a_hi) & in_force
        cross[a_name] = {}
        for d_name, d_lo, d_hi in (*DEPTH_BANDS_NOISE, ("all", 0.0, 1e9)):
            sel = in_age & band_mask(values=depth_noise, lo=d_lo, hi=d_hi)
            cross[a_name][d_name] = {
                "n": int(sel.sum()),
                "miss": float(1.0 - seen[sel].mean()) if sel.any() else None,
            }
    out["miss_by_age_and_depth"] = cross
    by_depth: dict[str, Any] = {}
    for d_name, d_lo, d_hi in (*DEPTH_BANDS_NOISE, ("all", 0.0, 1e9)):
        sel = in_force & band_mask(values=depth_noise, lo=d_lo, hi=d_hi)
        by_depth[d_name] = {
            "rows": int(sel.sum()),
            "share_of_in_force_rows": float(sel.sum() / max(in_force.sum(), 1)),
            "seen_share_of_rows": float(seen[sel].mean()) if sel.any() else None,
            "seen_share_of_depth_mass": float(
                depth_noise[sel & seen].sum() / depth_noise[sel].sum()
            )
            if sel.any()
            else None,
        }
    out["seen_by_depth"] = by_depth
    curve = []
    pooled = band_mask(values=true_age, lo=3.0, hi=11.0) & in_force
    deep = pooled & (depth_noise >= 1.0)
    for z in grid:
        s = age(inj, z) > 0
        curve.append(
            {
                "z": z,
                "false_alarm_real": float((age(real, z) > 0).mean()),
                "miss_3_10": float(1.0 - s[pooled].mean()) if pooled.any() else None,
                "miss_3_10_depth_at_least_1": float(1.0 - s[deep].mean()) if deep.any() else None,
                "miss_in_force": float(1.0 - s[in_force].mean()),
                "seen_share_of_depth_mass": float(
                    depth_noise[in_force & s].sum() / depth_noise[in_force].sum()
                ),
            }
        )
    out["curve"] = curve
    soft_inj = soft_score(hist_power=inj[0], visible=inj[1]) / noise
    soft_real = soft_score(hist_power=real[0], visible=real[1]) / noise
    neg = soft_real[np.isfinite(soft_real)]
    out["soft_input_auc_vs_real_rows"] = {
        d_name: {
            a_name: auc(
                positive=soft_inj[
                    in_force
                    & band_mask(values=true_age, lo=a_lo, hi=a_hi)
                    & band_mask(values=depth_noise, lo=d_lo, hi=d_hi)
                    & np.isfinite(soft_inj)
                ],
                negative=neg,
            )
            for a_name, a_lo, a_hi in (("3 to 10", 3.0, 11.0), ("11-15", 11.0, 16.0))
        }
        for d_name, d_lo, d_hi in DEPTH_BANDS_NOISE
    }
    return out


def depth_law(*, depth_share: np.ndarray, std: np.ndarray, noise: float) -> dict[str, Any]:
    """Return the share of shifts below each depth band, in noise units (table 4, injected side).

    Args:
        depth_share: Each shift's depth as a share of capacity.
        std: The standard deviation of the shift's Primary (share of capacity).
        noise: The noise unit in standardised units.
    """
    d = depth_share / std / noise
    return {
        "n_shifts": int(d.size),
        "share_by_band": {
            name: float(band_mask(values=d, lo=lo, hi=hi).mean())
            for name, lo, hi in DEPTH_BANDS_NOISE
        },
        "quantiles": {str(q): float(np.quantile(d, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }


def real_transfers(*, median_std: float, noise: float) -> dict[str, Any]:
    """Return the size of the 66 hand-filtered real transfers of experiment 101.

    The table gives each transfer's `share` (of the receiving load, which is not the share of
    capacity) and `moved_mw_p50`. The median Primary's standard deviation converts a share to noise
    units only roughly, so the figures are for comparison of scale.
    """
    import e00738_world as world
    import polars as pl
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    t = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / world.REAL_TRANSFERS_FILE)
    share = t["share"].drop_nulls().to_numpy()
    mw = t["moved_mw_p50"].drop_nulls().to_numpy()
    return {
        "n_transfers": int(t.height),
        "share_of_receiving_load_quantiles": {
            str(q): float(np.quantile(share, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "moved_mw_p50_quantiles": {
            str(q): float(np.quantile(mw, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "hours_quantiles": {
            str(q): float(np.quantile(t["hours"].to_numpy(), q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "share_under_1_noise_unit_if_share_were_capacity_share": float(
            (share / median_std / noise < 1.0).mean()
        ),
    }


def build(*, month: str) -> dict[str, Any]:
    """Build the rows of fold `month` as `task_a1_table` does, and return every table."""
    import e00713_train as tr  # ty: ignore[unresolved-import]
    import e00738_run as run
    import e00738_survival as surv
    import e00738_train as train
    import e00738_world as world

    data, tw, is_mw = run.load_fold(month=month, phase="train", device="cpu")
    z = (data.clean - data.target_mean[:, None]) / data.target_std[:, None]
    noise = surv.noise_unit_from_panel(z=z, cutoff_idx=data.cutoff_idx)

    def draw(pass_no: int) -> list:
        return [
            s
            for p in range(16)
            for s in world.renewal_shifts(
                space="train",
                fit_seed=0,
                month_idx=common.month_index(month=month),
                pass_no=pass_no,
                primary=p,
                window_start_idx=0,
                window_end_idx=data.cutoff_idx,
                is_mw=is_mw,
            )
        ]

    own_shifts = draw(0)
    own = world.build_own(clean=data.clean, shifts=own_shifts, is_mw=is_mw)
    sources = {
        "injected": train.world_source(
            base=tw.source,
            clean=data.clean,
            own=own,
            target_mean=data.target_mean,
            target_std=data.target_std,
            device="cpu",
        ),
        "real": tw.source,
    }
    origins = np.arange(48 * 20 + 24, data.cutoff_idx - 48 * 3, 48 * 3)
    prim, cc = np.meshgrid(np.arange(16), origins, indexing="ij")
    prim, cc = prim.ravel(), cc.ravel()
    samples = tr.SampleSet(primary=prim, c_idx=cc, is_val=np.zeros(len(prim), bool))
    hist = {}
    for name, src in sources.items():
        power, visible = [], []
        for start in range(0, len(prim), 512):
            sel = np.arange(start, min(start + 512, len(prim)))
            b = tr.batch_for(
                source=src, samples=samples, index=sel, lead=np.zeros(len(sel), int),
                ensemble=False, cut=10**9, dropout=None,
            )  # fmt: skip
            power.append(b.inputs.hist_power.numpy())
            visible.append(b.inputs.hist_visible.numpy())
        hist[name] = (np.concatenate(power), np.concatenate(visible))
    true_age = np.full(len(prim), -1.0)
    depth_noise = np.zeros(len(prim))
    by_prim = {p: [s for s in own_shifts if s.primary == p] for p in range(16)}
    for i, (p, c) in enumerate(zip(prim, cc, strict=True)):
        for s in by_prim[int(p)]:
            if s.onset_idx <= c < s.end_idx:
                true_age[i] = (c - s.onset_idx) / world.PER_DAY
                depth_noise[i] = s.depth / float(data.target_std[int(p)]) / noise
    z_main = Z_REPORTED
    result = tables(
        inj=hist["injected"], real=hist["real"], true_age=true_age, depth_noise=depth_noise,
        noise=noise, z_main=z_main,
    )  # fmt: skip
    shifts = [s for r in range(N_SHIFT_PASSES) for s in draw(r)]
    result["injected_depth_law"] = depth_law(
        depth_share=np.array([s.depth for s in shifts]),
        std=np.array([float(data.target_std[s.primary]) for s in shifts]),
        noise=noise,
    )
    result["real_transfers"] = real_transfers(
        median_std=float(np.median(data.target_std)), noise=noise
    )
    result["noise_unit"] = noise
    result["median_target_std"] = float(np.median(data.target_std))
    result["month"] = month
    return result


def main() -> None:
    """Build the tables for the month on the command line and write them to the output folder."""
    month = sys.argv[1]
    os.environ["E00667_MONTH"] = month
    os.environ["E00667_PHASE"] = "train"
    result = build(month=month)
    common.atomic_json(obj=result, path=common.OUT / "a1" / f"diagnostic_{month}.json")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
