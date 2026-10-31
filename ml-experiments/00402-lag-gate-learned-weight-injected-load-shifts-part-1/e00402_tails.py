# ruff: noqa: E501
"""Experiment 402: upper-tail coverage and pinball loss on real rows, per arm and slice.

DATA_PATH_INTERNAL=... PYTHONPATH=<experiment 87 folder> uv run --no-sync python \
    ml-experiments/00402-*/e00402_tails.py
"""

from pathlib import Path
from typing import Final

import e00402_mixture as mix  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
)

LEVELS: Final[dict[str, int]] = {"q95": 9, "q99": 10}
ARMS: Final[dict[str, str]] = {
    "forecast with lags": "lags",
    "forecast without lags": "lagless",
    "50/50 mixture": "fixed 50/50",
    "constant mixture fitted on real labels": "const (real, per group)",
    "gate, full": "gate full (group)",
    "gate trained on real labels": "gate real (group)",
}
QUANTILE_COLUMNS: Final[list[str]] = [
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
]  # fmt: skip


def _stored(*, arm: str, site: str) -> pl.DataFrame:
    folder = next(ML_EXPERIMENTS_DATA_DIR.glob("00392-*")) / "work" / "arm_preds" / arm
    frame = pl.concat([pl.read_parquet(p) for p in sorted(folder.glob(f"{site}_s0_*.parquet"))])
    return frame.with_columns(
        pl.col("power_fcst_init_time", "valid_time").dt.replace_time_zone(None)
    )


def main() -> None:
    """Print coverage of q95 and q99 and the pinball loss per arm, group, and slice."""
    scores = pl.read_parquet(WORK / "gate_scores_r1.parquet").filter(pl.col("kind") == "real")
    rows = []
    for site in sorted(scores["site"].unique().to_list()):
        s = scores.filter(pl.col("site") == site)
        if s["group"][0] == "report":
            continue
        actual = pl.read_parquet(
            WORK / "worlds" / f"real_r1_{site}.parquet",
            columns=["power_fcst_init_time", "valid_time", "lead_hours", "actual"],
        )
        keys = ["power_fcst_init_time", "valid_time"]
        qa = _stored(arm="r1_lagged", site=site).select(
            *keys, *[pl.col(c).alias(f"a_{c}") for c in QUANTILE_COLUMNS]
        )
        qb = _stored(arm="r1_lagless", site=site).select(
            *keys, *[pl.col(c).alias(f"b_{c}") for c in QUANTILE_COLUMNS]
        )
        d = s.join(actual, on=[*keys, "lead_hours"]).join(qa, on=keys).join(qb, on=keys)
        d = d.drop_nulls(subset=["actual"])
        a = d.select([f"a_{c}" for c in QUANTILE_COLUMNS]).to_numpy().astype(np.float64)
        b = d.select([f"b_{c}" for c in QUANTILE_COLUMNS]).to_numpy().astype(np.float64)
        y = d["actual"].to_numpy().astype(np.float64)
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(y)
        d, a, b, y = d.filter(pl.Series(ok)), a[ok], b[ok], y[ok]
        pair = mix.Pair(q1=a, q2=b, actual=y)
        out = d.select("group", "lags_real", "target_real", "after_real")
        for label, col in ARMS.items():
            w = np.clip(np.nan_to_num(d[f"w::{col}"].to_numpy().astype(np.float64), nan=0.5), 0, 1)
            q = pair.quantiles(w=w)
            for name, i in LEVELS.items():
                level = float(name[1:]) / 100.0
                out = out.with_columns(
                    pl.Series(f"{label}|{name}|cover", (y <= q[:, i]).astype(float)),
                    pl.Series(
                        f"{label}|{name}|pinball",
                        100.0 * np.maximum(level * (y - q[:, i]), (level - 1.0) * (y - q[:, i])),
                    ),
                )
        rows.append(out)
    d = pl.concat(rows)
    slices = {
        "harm": (pl.col("lags_real") > 0) & ~pl.col("target_real"),
        "inside": pl.col("target_real"),
        "clean": (pl.col("lags_real") == 0) & ~pl.col("target_real") & ~pl.col("after_real"),
        "all": pl.lit(True),
    }
    for group in ("six", "eight"):
        for sname, expr in slices.items():
            g = d.filter((pl.col("group") == group) & expr)
            print(f"\\n{group} | {sname} | n={g.height}")
            for label in ARMS:
                print(
                    f"  {label}: "
                    + ", ".join(
                        f"{n} coverage {g[f'{label}|{n}|cover'].mean():.3f} pinball {g[f'{label}|{n}|pinball'].mean():.3f}"
                        for n in LEVELS
                    )
                )


if __name__ == "__main__":
    main()
