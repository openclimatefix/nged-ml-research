# ruff: noqa: E501
"""Experiment 514: checksums, the mask table, thresholds for 1/2/5/7% of rows, tiers, low-weight shares."""

import hashlib
import json
import pickle

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_real as real  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SHARES = (0.01, 0.02, 0.05, 0.07)


def main() -> None:
    """Write the mask table, the checksums, and the threshold and tier summary."""
    out: dict = {"folds": {}}
    frames, sums = [], []
    clean = d514.clean_primary_power()
    events = d514.event_mask(n=clean.shape[1])
    for month in d514.FOLDS:
        path = d514.WORK_DIR / f"p_{month}.pkl"
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        t = pickle.loads(path.read_bytes())
        mask = pl.DataFrame(t["mask"]).with_columns(pl.lit(month).alias("fold"))
        frames.append(mask)
        k = len(t["cutoffs"]) - 1
        c, origin = int(t["cutoffs"][k]), int(t["table_origin"][k])
        p = t["p"][k][:, : c - origin]
        valid = (
            np.isfinite(clean[:, origin:c])
            & (np.arange(origin, c) < c - d514.CUT_STEPS)[None, :]
            & ~events[:, origin:c]
        )
        values = p[valid]
        thresholds = {str(s): float(np.quantile(values, 1 - s)) for s in SHARES}
        achieved = {str(s): float((values >= thresholds[str(s)]).mean()) for s in SHARES}
        last = mask.filter(pl.col("valid_for_cutoff") == c)
        tier = {
            x: float((last["tier"] == x).sum() / (valid.any(axis=1).sum() * (730 / 365.25)))
            for x in ("A", "B")
        }
        out["folds"][month] = {
            "thresholds_for_flagged_share": thresholds, "achieved_share": achieved,
            "weight_below_half": {str(g): float((real.weights(p=values, g=g) < 0.5).mean()) for g in (1, 2)},
            "candidates_p_ge_half_per_primary_year_by_tier": tier,
            "open_candidates_p_ge_half": int(last["open"].sum()),
        }  # fmt: skip
        print(month, json.dumps(out["folds"][month]), flush=True)
    pl.concat(frames).write_parquet(d514.WORK_DIR / "mask_table.parquet")
    sums.append(
        f"{hashlib.sha256((d514.WORK_DIR / 'frozen_model.pkl').read_bytes()).hexdigest()}  frozen_model.pkl"
    )
    (d514.WORK_DIR / "sha256.txt").write_text("\n".join(sums) + "\n")
    (d514.WORK_DIR / "eval_finish.json").write_text(json.dumps(out, indent=2))
    print("\n".join(sums))


if __name__ == "__main__":
    main()
