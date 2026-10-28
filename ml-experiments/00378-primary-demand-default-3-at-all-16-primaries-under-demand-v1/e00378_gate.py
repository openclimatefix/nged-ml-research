# ruff: noqa: E501
"""Control gate: the port reproduces experiment 322's stored default 3 on the six Primaries.

The arms of `e00378_arms.py` run on experiment 319's own rows (the convex chain of experiment 317 at
the six Primaries, experiment 319's member responses, the three periods) instead of experiment 377's
stage-2 rows, with the same fold-month replay. The result is compared with the `d1_*` columns of
experiment 322's stored `scored_long_lead.parquet`, row by row, and by the mean quantile score at
168 to 240 hours.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320, 322, 377> E00377_SOURCE=control \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00378-*/e00378_gate.py
"""

import json

import e00319_arms as a319  # ty: ignore[unresolved-import]
import e00322_common as c322  # ty: ignore[unresolved-import]
import e00378_arms as arms
import e00378_common as common
import numpy as np
import polars as pl

QUANTILES = common.QUANTILES


def main() -> None:
    """Replay experiment 319's rows and compare default 3 with experiment 322's stored d1."""
    frame, delta = a319.load_rows()
    frame = frame.with_columns(group=pl.lit("mw"))
    scored = (frame["period"] == "scored").to_numpy()
    in_fit = np.ones(frame.height, dtype=bool)
    out, log = arms.replay(frame=frame, delta=delta, in_fit=in_fit, scored=scored, label="gate")
    stored = pl.read_parquet(c322.ARMS_DIR / "scored_long_lead.parquet")
    idx = np.flatnonzero(scored)
    keys = ["site", "power_fcst_init_time", "valid_time", "lead_hours"]
    if stored.height != len(idx) or not stored.select(keys).equals(frame[idx].select(keys)):
        msg = "experiment 322's stored rows do not line up with the rebuilt rows"
        raise ValueError(msg)
    mine = out["d3"][idx]
    theirs = stored.select([f"d1_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)[idx]
    ok = (~frame["switching_mask"]).to_numpy()[idx] & np.isfinite(actual)

    def score(q: np.ndarray) -> float:
        loss = np.maximum(
            common.LEVELS[None, :] * (actual[:, None] - q),
            (common.LEVELS[None, :] - 1.0) * (actual[:, None] - q),
        )
        return float(2.0 * loss[ok].mean())

    result = {
        "rows": int(len(idx)),
        "max_abs_quantile_difference": float(np.nanmax(np.abs(mine - theirs))),
        "quantile_score_port": score(mine),
        "quantile_score_stored_322": score(theirs),
        "difference_points": score(mine) - score(theirs),
        "passes_within_0.05": abs(score(mine) - score(theirs)) <= 0.05,
    }
    common.write_json_atomic(path=common.ARMS_DIR / "gate_322.json", content=result)
    common.write_json_atomic(path=common.ARMS_DIR / "gate_322_fit_log.json", content=log)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
