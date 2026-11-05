# ruff: noqa: ANN201, ANN202, D103, E501, PERF401, RUF028
"""Score experiment 562: real minus placebo, real minus M0, and placebo minus M0, per arm.

Reads the per-row CRPS that `e00562_rows.py` wrote and experiment 521's loader (the row sets, lead
bands, and cold-night flag). Intervals are 5-block month-block bootstrap (near min-to-max) at the
95% level and at the Bonferroni level for `TESTS` primary contrasts. A contrast is made only between
arms that share a seed set: `M0s0` for one seed, `M0s01` for two.
"""

import json

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W = ML_EXPERIMENTS_DATA_DIR / "00562-modular-lag-free-network-null-input-retry-with-placebos"
K = ["primary", "origin", "target"]
PAIRS = (("B1", "B1p"), ("B2", "B2p"), ("B3a", "B3ap"), ("B3b", "B3bp"), ("B4", "B4p"))
TESTS = 5 * 2 * 2
"""5 arms x 2 groups (eight, six) x 2 slices (all rows, cold nights) = 20 primary contrasts."""
SLICES = {"all": None, "cold_night": "cold"}


def table(*, arms: list[str]) -> pl.DataFrame:
    base = s.load()
    mine = pl.concat(
        [
            pl.read_parquet(
                W / "work/scores" / f"rows_{m}.parquet", columns=[*K, *[f"{a}__crps" for a in arms]]
            )
            for m in s.MONTHS
        ]
    )
    return base.join(mine, on=K, how="left", validate="1:1", maintain_order="left").with_columns(band=s.band_expr())  # fmt: skip


def main():
    schema = pl.read_parquet_schema(W / "work/scores" / f"rows_{s.MONTHS[0]}.parquet")
    arms = [a.removesuffix("__crps") for a in schema if a.endswith("__crps") and a.split("__")[0] in {*(x for p in PAIRS for x in p), "B4-oracle", "M0s0", "M0s01", "M0s1", "B1s0", "B1s1", "B1ps0", "B1ps1"}]  # fmt: skip
    t = table(arms=arms)
    need = [f"{a}__crps" for a in arms]
    t = t.with_columns(
        main16=pl.col("main16") & pl.all_horizontal([pl.col(c).is_finite() for c in need])
    )
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    for group in ("eight", "six"):
        part = s.row_set(table=t, name=group)
        for slice_name, flag in SLICES.items():
            sub = part if flag is None else part.filter(pl.col(flag))
            for arm in arms:
                absolute.append({"set": group, "slice": slice_name, "arm": arm, "n": sub.height, "crps": float(sub[f"{arm}__crps"].mean())})  # fmt: skip
            for real, plc in PAIRS:
                if real not in arms or plc not in arms:
                    continue
                ref = _reference(arm=real)
                pairs = [(real, plc, True), (real, ref, False), (plc, ref, False)]
                if real == "B4" and "B4-oracle" in arms:
                    pairs.append(("B4-oracle", ref, False))
                for a, b, primary in pairs:
                    d = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
                    st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                    contrasts.append({"set": group, "slice": slice_name, "contrast": f"{a} - {b}", "primary": primary, "tests": TESTS, **st})  # fmt: skip
    pl.DataFrame(absolute).write_csv(W / "scores_absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores_contrasts.csv")
    show = pl.DataFrame(contrasts).filter(pl.col("primary")).select("set", "slice", "contrast", "n", "difference", "low_bonferroni_month", "high_bonferroni_month", "by_fold")  # fmt: skip
    with pl.Config(tbl_rows=60, tbl_width_chars=200, fmt_str_lengths=120):
        print(show)
        print(pl.DataFrame(contrasts).filter(~pl.col("primary")).select("set", "slice", "contrast", "difference", "low_bonferroni_month", "high_bonferroni_month"))  # fmt: skip
    json.dump({"arms": arms}, open(W / "scores_arms.json", "w"))  # noqa: SIM115, PTH123


def _reference(*, arm: str) -> str:
    """Return the M0 reference sharing the arm's seed set: `M0s01` when seeds 0 and 1 are both assembled."""
    done = [(W / "work/seed_preds" / arm / f"{m}_s1.parquet").exists() for m in s.MONTHS]
    return "M0s01" if all(done) else "M0s0"


main()
