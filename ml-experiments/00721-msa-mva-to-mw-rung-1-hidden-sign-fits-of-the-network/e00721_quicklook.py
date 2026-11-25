# ruff: noqa: E501
"""Experiment 721: the quick look (Stage 3): `W2`, fold 2026-05, seed 0, arms `P0`, `SG-0`, and `SG-A`.

The quick look reads the wiring and prints every contrast labelled interim, compared with no bar. Its
only stop is the harm stop: if the signed CRPS of `SG-A` over the rows of either hidden Primary whose
signed truth is at or above zero exceeds the same Primary's `P0` by more than 0.20 points, the session
pauses the line and asks the manager, and the result is "stopped by the manager, no verdict". The
stop reads import rows because the reverse-flow rows are where an arm that can go negative gains over
`P0`.

Branch B's budget check (not the default) is the function `branch_b_decision`: the two interim `G1`
point estimates, from the patience-8 and the patience-30 fits, may differ by at most 0.20 points.
"""

from pathlib import Path

import e00721_core as core
import e00721_driver as drv
import e00721_fit as fit
import e00721_score as score
import polars as pl

QUICK_FOLD = core.MDE_FOLD
"""The quick look's fold, 2026-05."""


def mean_difference(
    *, a: pl.DataFrame, b: pl.DataFrame, restrict: pl.Expr | None, ids: list[int]
) -> tuple[float, int]:
    """Return `a` minus `b` as a mean CRPS difference and a row count.

    The difference is in points of the signed CRPS and is interim: it has no interval.
    """
    paired = score.pair_arms(
        a=a.filter(pl.col("time_series_id").is_in(ids)),
        b=b.filter(pl.col("time_series_id").is_in(ids)),
    )
    if restrict is not None:
        paired = paired.filter(restrict)
    if paired.height == 0:
        return float("nan"), 0
    return float(paired.select((pl.col("crps_a") - pl.col("crps_b")).mean()).item()), paired.height


def quick_look(*, data_dir: Path, hidden_ids: dict[str, int]) -> dict:
    """Return the quick look's interim contrasts and the harm stop, from the stored fits."""
    frames = {}
    for arm in ("P0", "SG-0", "SG-A"):
        df = drv.load_frame(
            data_dir=data_dir, job=fit.Job(arm=arm, world="W2", fold=QUICK_FOLD, seed=0)
        )
        if df is None:
            msg = f"the quick look needs {arm} in W2 on fold {QUICK_FOLD} at seed 0"
            raise FileNotFoundError(msg)
        frames[arm] = df
    import_rows = pl.col("actual_a") >= 0.0
    reverse_rows = pl.col("actual_a") < 0.0
    per_primary = {}
    for site, series_id in hidden_ids.items():
        diff, n = mean_difference(
            a=frames["SG-A"], b=frames["P0"], restrict=import_rows, ids=[series_id]
        )
        per_primary[site] = {
            "sg_a_minus_p0_import_points": diff,
            "n_rows": n,
            "harm": bool(diff > core.HARM_STOP_POINTS),
        }
    ids = list(hidden_ids.values())
    contrasts = {}
    for name, (a, b) in {
        "sg_a_minus_sg_0": ("SG-A", "SG-0"),
        "sg_a_minus_p0": ("SG-A", "P0"),
        "sg_0_minus_p0": ("SG-0", "P0"),
    }.items():
        for label, restrict in (("reverse", reverse_rows), ("all", None)):
            point, n = mean_difference(a=frames[a], b=frames[b], restrict=restrict, ids=ids)
            contrasts[f"{name}_{label}"] = {
                "point": point,
                "n_rows": n,
                "status": "interim, compared with no bar",
            }
    return {
        "fold": QUICK_FOLD, "seed": 0, "per_primary": per_primary, "contrasts": contrasts,
        "harm_stop": any(p["harm"] for p in per_primary.values()),
        "result_if_stopped": "stopped by the manager, no verdict",
    }  # fmt: skip


def branch_b_decision(*, g1_patience_8: float, g1_patience_30: float) -> dict:
    """Return Branch B's budget check.

    Decision 2 returns to the maintainer when the two interim `G1` points differ by more than 0.20
    points.
    """
    gap = abs(float(g1_patience_8) - float(g1_patience_30))
    return {"gap_points": gap, "return_to_maintainer": bool(gap > core.BRANCH_B_THRESHOLD_POINTS)}


def format_quick_look(*, result: dict) -> str:
    """Return the printed quick look (contrasts and counts only; it holds no series)."""
    lines = [f"quick look, fold {result['fold']}, seed {result['seed']} (interim, no bar)"]
    for name, body in result["contrasts"].items():
        lines.append(f"  {name}: {body['point']:+.3f} points on {body['n_rows']} rows")
    for site, body in result["per_primary"].items():
        lines.append(
            f"  harm check at {site}: SG-A minus P0 on import rows {body['sg_a_minus_p0_import_points']:+.3f} points -> {'HARM' if body['harm'] else 'ok'}"
        )
    lines.append(
        f"harm stop: {result['harm_stop']}"
        + (f" ({result['result_if_stopped']})" if result["harm_stop"] else "")
    )
    return "\n".join(lines)
