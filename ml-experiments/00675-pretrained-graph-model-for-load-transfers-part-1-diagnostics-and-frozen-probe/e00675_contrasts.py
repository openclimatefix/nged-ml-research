"""Pre-registered contrasts of experiment 675's probe, with world-resampled 95% intervals.

Reads the `found_*.parquet` and `*_recipient.parquet` files that `e00675_score.py` leaves in the
probe folder. Each contrast is the mean over seeds 0 to 2 (arm F has one fit, repeated). Intervals
resample the 40 scored worlds, the same worlds for both arms.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from e00675_pretrain import ROOT
from e00675_score import CELLS, WORLDS, auroc, share

PROBE = ROOT / "probe"
SEEDS = (0, 1, 2)


def found(arm: str, seed: int) -> pl.DataFrame:
    """Donor-found rows of an arm (arm F has one fit)."""
    return pl.read_parquet(PROBE / f"found_{arm}_seed{0 if arm == 'F' else seed}.parquet")


def main() -> None:
    """Print and store the contrasts."""
    rng = np.random.default_rng(1)
    worlds = np.array(WORLDS)
    boots = [rng.choice(worlds, len(worlds)) for _ in range(1000)]
    dfs = {(a, k): found(a, k) for a in ("P-M", "P-C", "R", "F") for k in SEEDS}
    out: dict = {}
    for c in CELLS:
        row: dict = {}

        def mean_share(arm: str, w: np.ndarray, c: str = c) -> float:
            return float(np.nanmean([share(df=dfs[(arm, k)], cell_name=c, worlds=w) for k in SEEDS]))

        for a, b in (("P-M", "R"), ("P-M", "F"), ("P-M", "P-C"), ("P-C", "R")):
            per_seed = [share(df=dfs[(a, k)], cell_name=c, worlds=worlds) - share(df=dfs[(b, k)], cell_name=c, worlds=worlds) for k in SEEDS]
            bs = [mean_share(a, w) - mean_share(b, w) for w in boots]
            row[f"{a} minus {b} donor MW found"] = {"mean": float(np.mean(per_seed)), "seeds": per_seed, "ci": [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]}  # fmt: skip
        row["donor found"] = {a: mean_share(a, worlds) for a in ("P-M", "P-C", "R", "F")}
        au = {}
        for arm in ("P-M", "P-C", "R", "F"):
            vals = []
            for k in SEEDS:
                rec = pl.read_parquet(PROBE / f"{arm}_seed{0 if arm == 'F' else k}_recipient.parquet").filter(pl.col("cell") == c)
                vals.append(auroc(y=rec["label"].to_numpy(), s=rec["score"].to_numpy()) if rec.height else float("nan"))
            au[arm] = vals
        row["recipient AUROC"] = {a: float(np.nanmean(v)) for a, v in au.items()}
        row["recipient AUROC P-M minus R"] = [float(x) for x in np.array(au["P-M"]) - np.array(au["R"])]
        row["n_donors"] = dfs[("R", 0)].filter(pl.col("cell") == c).height
        out[c] = row
        print(c, "n_donors", row["n_donors"], "found", {k: round(v, 3) for k, v in row["donor found"].items()}, "AUROC", {k: round(v, 3) for k, v in row["recipient AUROC"].items()})  # fmt: skip
        for k, v in row.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"   {k}: {v['mean']:+.3f} (95% {v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}), seeds {[round(x, 3) for x in v['seeds']]}")
    (PROBE / "contrasts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
