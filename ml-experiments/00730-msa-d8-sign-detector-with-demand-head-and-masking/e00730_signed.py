"""Experiment 730: the signed read-outs R2 to R6 and table T1, as pure functions on arrays.

Experiment 729's refitted arms A, D, and D-sign store two-lobe forecasts. Arms S and SD* store
only P(X<0), so their signed forecast is the mixture of a lobe at +M and a lobe at -M with
weights (1 - P, P), each lobe with experiment 729's 0.5-point spread. The baselines P0, P0f, and
B1 come from experiment 729's `baseline_metrics`, called by the stage.
"""

from collections.abc import Mapping, Sequence

import e00730_paths  # noqa: F401

# isort: split
import e00729_baselines as base
import e00729_core as core729
import e00729_dist as dist
import e00729_score as score729
import e00730_core as core
import e00730_score as score
import numpy as np

T1_ARMS = ("S", "SD*", "D-sign", "A", "D")
FIGURE9_ARMS = ("P0f", "B1", "A", "D", "D-sign")
SIGNED_SUBSETS = ("all", "import", "reverse")


def lobes_of(*, pred: Mapping[str, np.ndarray], r: np.ndarray) -> dist.Lobes:
    """Return the two-lobe forecast of a stored prediction.

    Args:
        pred: A stored prediction: the lobes' keys `p`, `mu_i`, `s_i`, `mu_r`, `s_r`, or `p` alone.
        r: The reading in points of each row.

    Returns:
        The lobes: the stored ones, or the sign head's mixture around plus and minus `M`.
    """
    if "mu_i" in pred:
        return dist.Lobes(pred["p"], pred["mu_i"], pred["s_i"], pred["mu_r"], pred["s_r"])
    return base.point_lobes(
        xhat=base.reading_to_import(r=r), spread=core729.FLOOR_POINTS, p=np.asarray(pred["p"])
    )


def arm_signed_metrics(
    *, preds: Sequence[Mapping[str, np.ndarray]], rows: score729.Rows
) -> score729.RowMetrics:
    """Return the seed-averaged signed metrics of an arm from its stored predictions."""
    per_seed = [score729.lobe_metrics(lobes=lobes_of(pred=p, r=rows.r), x=rows.x) for p in preds]
    return score729.average_metrics(per_seed=per_seed)


def demand_climatology(*, climatology: base.Climatology, features: base.Features) -> np.ndarray:
    """Return the climatology's median demand at each row (the overall median in an empty cell)."""
    ids = climatology.cell_id(
        month=features.local_month, hh=features.local_hh, weekend=features.weekend
    )
    pooled = np.concatenate([v for v in climatology.cells.values() if len(v)])
    out = np.full(len(ids), float(np.median(pooled)))
    for i in np.unique(ids):
        sample = climatology.cells.get(int(i))
        if sample is not None and len(sample):
            out[ids == i] = float(np.median(sample))
    return out


def subset_table(
    *, rows: score729.Rows, metrics: Mapping[str, score729.RowMetrics], fields: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Return, for each arm, each of `fields` (`crps`, `mae`) on all, import, and reverse rows."""
    masks = score729.subset_masks(rows=rows)
    return {
        arm: {
            f: {
                s: float(getattr(m, f)[masks[s]].mean()) if masks[s].any() else float("nan")
                for s in SIGNED_SUBSETS
            }
            for f in fields
        }
        for arm, m in metrics.items()
    }


def signed_readout(
    *, rows: score729.Rows, metrics: Mapping[str, score729.RowMetrics], n_resamples: int
) -> dict:
    """Return R2 to R6, the signed part of R1, table T1, and the data of figure 9.

    Args:
        rows: The scored rows of the test bank.
        metrics: The signed metrics by name: the baselines `P0`, `P0f`, and `B1`, and the arms.
            A reading whose arms are missing is left out.
        n_resamples: The resamples of each reading.

    Returns:
        `R1_crps`, `R2` to `R6` (each a contrast dictionary; R4 and R5 add `limit` and `pass`),
        `T1`, and `figure9`.
    """
    masks = score729.subset_masks(rows=rows)
    m = metrics

    def read(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> dict[str, float]:
        reading = score.report_only_contrast(
            rows=rows, a=a, b=b, keep=keep, n_resamples=n_resamples
        )
        return score.contrast_dict(reading=reading)

    every = masks["all"]
    out: dict = {}
    if "S" in m and "D-sign" in m:
        out["R1_crps"] = read(m["S"].crps, m["D-sign"].crps, every)
    if "D" in m and "B1" in m:
        out["R2"] = read(m["D"].crps, m["B1"].crps, every)
    if "A" in m and "P0f" in m:
        out["R3"] = read(m["A"].crps, m["P0f"].crps, every)
    if "D" in m and "P0f" in m:
        for name, field, limit in (
            ("R4", "crps", core729.HARM_CRPS_POINTS),
            ("R5", "mae", core729.HARM_MAE_POINTS),
        ):
            reading = read(getattr(m["D"], field), getattr(m["P0f"], field), masks["import"])
            out[name] = {**reading, "limit": limit, "pass": bool(reading["point"] <= limit)}
    if "D-sign" in m and "D" in m:
        out["R6"] = read(m["D-sign"].crps, m["D"].crps, every)
    out["T1"] = subset_table(
        rows=rows, metrics={a: m[a] for a in T1_ARMS if a in m}, fields=("crps", "mae")
    )
    out["figure9"] = {
        a: v["crps"]
        for a, v in subset_table(
            rows=rows, metrics={a: m[a] for a in FIGURE9_ARMS if a in m}, fields=("crps",)
        ).items()
    }
    return out


def brier_by_subset(
    *, rows: score729.Rows, p_by_arm: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Return the Brier score of each arm on all, import, and reverse rows and by reading band."""
    subsets = {k: v for k, v in score729.subset_masks(rows=rows).items() if k in SIGNED_SUBSETS}
    subsets.update(score729.ambiguity_strata(rows=rows)["reading"])
    out: dict[str, dict[str, float]] = {}
    for arm, p in p_by_arm.items():
        per_row = score.row_brier(p=p, x=rows.x)
        out[arm] = {
            name: float(per_row[mask].mean()) if mask.any() else float("nan")
            for name, mask in subsets.items()
        }
    return out


def harm_detection_by_group(*, rows: score729.Rows, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Return the harm number and the detection number by world type and by reading band.

    The harm number of a group is the mean P(X<0) on its decidable import rows, and the detection
    number is the mean P(X<0) on its decidable reverse-flow rows (NaN where a group has none).
    """
    decidable = rows.r >= core.DECIDABLE_READING_POINTS
    groups = {
        f"world_{w}": rows.world == i for i, w in enumerate(core729.WORLD_TYPES)
    } | score729.ambiguity_strata(rows=rows)["reading"]
    harm, detection = {}, {}
    for name, mask in groups.items():
        imp = mask & decidable & (rows.x >= 0)
        rev = mask & decidable & (rows.x < 0)
        harm[name] = float(p[imp].mean()) if imp.any() else float("nan")
        detection[name] = float(p[rev].mean()) if rev.any() else float("nan")
    return {"harm": harm, "detection": detection}
