# ruff: noqa: E501
"""Experiment 514: the frozen probability tables per fold and daily cutoff, and the causality check.

For each fold the first cutoff (the refit) computes p for the whole history. Each later daily cutoff
recomputes the last `RECOMPUTE_STEPS` half-hours before its cut and takes older rows from the
previous cutoff's table. p is keyed by (Primary, grid index). Output per fold:
`p_{month}.npz` with `cutoffs` (grid indices), `p` of shape (cutoffs, 16, TABLE_STEPS) float32 over
the 730 days before each cutoff, and the mask-candidate list.

```bash
... e00514_tables.py build        # all four folds
... e00514_tables.py causal       # the perturbation check (printed before any other number)
```
"""

import argparse
import dataclasses
import pickle
import sys
import time
from pathlib import Path
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_detect as det  # ty: ignore[unresolved-import]
import e00514_train as train  # ty: ignore[unresolved-import]
import numpy as np

RECOMPUTE_STEPS: Final[int] = 45 * 48
TABLE_STEPS: Final[int] = 730 * 48
SHAPES_NAME: Final[str] = "shapes"


def p_for_series(
    *, model: train.Model, matrix: np.ndarray, edges: np.ndarray, scales: det.Scales,
    series: int, end: int, shape: np.ndarray | None, lo_candidate: int = 0,
) -> tuple[np.ndarray, det.Candidates, np.ndarray, np.ndarray]:  # fmt: skip
    """Return row p over [0, end), the candidates, their calibrated probabilities, and features."""
    cands = det.scan_series(x=matrix[series], sigma=scales.sigma[series], end=end, lo=lo_candidate)
    x = det.candidate_features(
        cands=cands, series=series, matrix=matrix, edges=edges, scales=scales, shape=shape
    )  # fmt: skip
    p_cand = model.candidate_p(x=x)
    raw = det.row_probability(cands=cands, p_cand=p_cand, n=end)
    p_cal = model.calibrate(raw=p_cand, shift=model.shift) if len(p_cand) else p_cand
    return model.calibrate(raw=raw, shift=model.shift), cands, p_cal, x


def fold_scales(*, fold: d514.Fold) -> det.Scales:
    """Return the scales fitted on the fold's history below its refit minus the cut."""
    return det.fit_scales(
        matrix=fold.partners(), mw_per_z=fold.mw_per_z, hist_end=fold.refit - d514.CUT_STEPS
    )


def p_at_cutoff(
    *, model: train.Model, fold: d514.Fold, scales: det.Scales, shapes: np.ndarray, cutoff_idx: int,
    previous: np.ndarray | None, full: bool, add: tuple[int, int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[dict]]:  # fmt: skip
    """Return the (16, n) p array at a cutoff, and the cutoff's open or closed candidate records.

    With `full` the whole history is scanned; otherwise only rows in the last `RECOMPUTE_STEPS`
    before the cut are replaced, and older rows are taken from `previous`.
    """
    end = cutoff_idx - d514.CUT_STEPS
    matrix = d514.series_at(fold=fold, end=end)
    if add is not None:
        lo, hi, amount = add
        matrix[:16, lo : min(hi, end)] += amount[:, None]
    edges = det.edge_matrix(matrix=matrix, scales=scales)
    n = matrix.shape[1]
    p = np.zeros((16, n), dtype=np.float32) if previous is None else previous.copy()
    p[:, end:] = 0.0
    records: list[dict] = []
    first = 0 if full else max(0, end - RECOMPUTE_STEPS)
    for i in range(16):
        p_row, cands, p_cand, x = p_for_series(
            model=model, matrix=matrix, edges=edges, scales=scales, series=i, end=end,
            shape=shapes[i].astype(np.float64), lo_candidate=max(0, first - det.OPEN_MAX),
        )  # fmt: skip
        p[i, first:end] = p_row[first:end]
        keep = (cands.start + cands.duration > first) & (p_cand >= 0.5)
        for j in np.flatnonzero(keep):
            f = dict(zip(det.FEATURES, x[j], strict=True))
            tier = (
                "A"
                if f["has_pair"] > 0 and f["symmetry"] <= np.log(det.SYMMETRY_TOLERANCE)
                else "B"
            )
            records.append({
                "primary": i, "start": int(cands.start[j]), "end": int(min(cands.start[j] + cands.duration[j], end)),
                "p": float(p_cand[j]), "tier": tier, "size_z": float(f["amplitude"]),
                "shape_corr": float(f["shape_corr"]), "open": bool(cands.is_open[j]),
                "valid_for_cutoff": cutoff_idx,
            })  # fmt: skip
    return p, records


def build_fold_tables(*, model: train.Model, month: str) -> dict:
    """Build a fold's p tables at every daily cutoff."""
    fold = d514.load_fold(month=month)
    shapes = d514.load_shapes()
    scales = fold_scales(fold=fold)
    cutoffs = [d514.cutoff_index(cutoff=c) for c in d514.daily_cutoffs(month=month)]
    stack = np.zeros((len(cutoffs), 16, TABLE_STEPS), dtype=np.float32)
    mask_rows: list[dict] = []
    previous: np.ndarray | None = None
    started = time.time()
    for k, c in enumerate(cutoffs):
        previous, records = p_at_cutoff(
            model=model, fold=fold, scales=scales, shapes=shapes, cutoff_idx=c, previous=previous, full=k == 0
        )  # fmt: skip
        mask_rows.extend(records)
        lo = max(0, c - TABLE_STEPS)
        stack[k, :, : c - lo] = previous[:, lo:c]
        if k == 0:
            print(f"{month}: first cutoff {time.time() - started:.0f} s", flush=True)
    return {"month": month, "cutoffs": np.array(cutoffs), "p": stack, "mask": mask_rows,
            "table_origin": [max(0, c - TABLE_STEPS) for c in cutoffs]}  # fmt: skip


def causal_check(*, model: train.Model, month: str, cutoffs: tuple[int, ...] = (3, 10, 20)) -> dict:
    """Perturb every half-hour at or after the cut and check that p does not change."""
    fold = d514.load_fold(month=month)
    shapes = d514.load_shapes()
    scales = fold_scales(fold=fold)
    days = d514.daily_cutoffs(month=month)
    result = {}
    previous = None
    for k in range(max(cutoffs) + 1):
        c = d514.cutoff_index(cutoff=days[k])
        p, _ = p_at_cutoff(
            model=model, fold=fold, scales=scales, shapes=shapes, cutoff_idx=c, previous=previous, full=k == 0
        )  # fmt: skip
        previous = p
        if k in cutoffs:
            # same cutoff, with the future of the fold replaced by noise, rebuilt from the first cutoff
            rng = np.random.default_rng(7)
            bad = dataclasses.replace(fold, z=fold.z.copy(), supply=fold.supply.copy())
            end = c - d514.CUT_STEPS
            bad.z[:, end:] = rng.normal(0, 5, size=bad.z[:, end:].shape)
            bad.supply[:, end:] = rng.normal(0, 5, size=bad.supply[:, end:].shape)
            prev2 = None
            for j in range(k + 1):
                cj = d514.cutoff_index(cutoff=days[j])
                prev2, _ = p_at_cutoff(
                    model=model, fold=bad, scales=scales, shapes=shapes, cutoff_idx=cj, previous=prev2, full=j == 0
                )  # fmt: skip
            assert prev2 is not None
            diff = float(np.abs(prev2[:, :end] - p[:, :end]).max())
            full_p, _ = p_at_cutoff(
                model=model, fold=fold, scales=scales, shapes=shapes, cutoff_idx=c, previous=None, full=True
            )  # fmt: skip
            old = slice(0, max(0, end - RECOMPUTE_STEPS))
            window_changed = int((np.abs(full_p[:, old] - p[:, old]) > 1e-6).sum())
            result[k] = {"older_than_window_rows_changed_by_full_recompute": window_changed, "cutoff_idx": c, "rows_checked": int(end * 16), "max_abs_difference": diff,
                         "rows_after_cut_nonzero": int((p[:, end:] != 0).sum())}  # fmt: skip
    return result


def main() -> None:
    """Dispatch the command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "causal"))
    parser.add_argument("--month", nargs="*", default=list(d514.FOLDS))
    args = parser.parse_args()
    model = pickle.loads((d514.WORK_DIR / "frozen_model.pkl").read_bytes())
    if args.command == "causal":
        out = causal_check(model=model, month=args.month[0])
        print("CAUSALITY CHECK (perturbed future vs original, whole chain):")
        for k, v in out.items():
            print(f"  cutoff day {k}: {v}")
        assert all(v["max_abs_difference"] == 0.0 for v in out.values()), (
            "p changed with the future"
        )
        return
    for month in args.month:
        path = d514.WORK_DIR / f"p_{month}.pkl"
        if path.exists():
            print(month, "exists")
            continue
        tables = build_fold_tables(model=model, month=month)
        tmp = Path(str(path) + ".tmp")
        tmp.write_bytes(pickle.dumps(tables, protocol=5))
        tmp.rename(path)
        print(month, "written", flush=True)


if __name__ == "__main__":
    sys.modules.setdefault("e00514_train", train)
    main()
