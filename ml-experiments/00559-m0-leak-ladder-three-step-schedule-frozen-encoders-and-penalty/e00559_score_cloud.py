# ruff: noqa: ANN201, D103, PTH123, SIM115
"""Score the limited-history cloud arms: A11c minus A11c-null on the 4 folds from 2025-12 only.

The cloud arm has no 2025-10 fold, so it is never compared with M0 or any full-history arm. Both
arms hold cloud as value 0 plus a missing flag before 2025-11-21. The output is not a counted test.
"""

import json

import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
W = DD / "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty"
MONTHS = ("2025-12", "2026-01", "2026-03", "2026-06")
K = ["primary", "origin", "target"]
POOLED = "24-120h"
TESTS = 28
ARMS = ("A11c", "A11c-null")
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]


def leak():
    out = []
    for arm in ARMS:
        for m in MONTHS:
            for day in range(14, 21):
                ctrl = pl.read_parquet(W / "work/fits" / arm / f"{m}_d{day}_s0.parquet").select(
                    *KEYS, c="q50"
                )
                for sign, tag in ((1, "p"), (-1, "n")):
                    p = W / "work/fits" / arm / f"{m}_d{day}_s0_stress{tag}.parquet"
                    if p.exists():
                        j = ctrl.join(pl.read_parquet(p).select(*KEYS, s="q50"), on=KEYS)
                        out.append(
                            {
                                "arm": arm,
                                "month": m,
                                "leak": float((sign * 100.0 * (j["s"] - j["c"])).mean()),  # ty: ignore[unsupported-operator]
                            }
                        )
    t = pl.DataFrame(out).group_by("arm", "month").agg(pl.col("leak").mean())
    return t.group_by("arm").agg(
        leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max()
    )


def main():
    s.MONTHS = MONTHS
    base = s.load()
    mine = pl.concat(
        [
            pl.read_parquet(
                W / "work/scores_cloud" / f"rows_{m}.parquet",
                columns=[*K, *[f"{a}__crps" for a in ARMS]],
            )
            for m in MONTHS
        ]
    )
    t = base.join(mine, on=K, how="left", validate="1:1", maintain_order="left")
    fin = pl.all_horizontal([pl.col(f"{a}__crps").is_finite() for a in ARMS])
    t = t.with_columns(main16=pl.col("main16") & fin.fill_null(False)).with_columns(
        band=s.band_expr()
    )
    chk = json.load(open(W / "checks_fit_A11c_s0-1.json"))
    rows_with_cloud = {
        m: v.get("fine_tune_rows_with_extra_values") for m, v in chk["months"].items()
    }
    d8 = next(p for p, lab in s.PRIMARY_LABELS.items() if lab == "D8")
    rng = np.random.default_rng(0)
    rows = []
    for name in ("eight", "six", "eight_cold", "six_cold", "eight_without_D8"):
        part = s.row_set(table=t, name="eight" if name.startswith("eight") else "six")
        if name.endswith("_cold"):
            part = part.filter(pl.col("cold"))
        if name == "eight_without_D8":
            part = part.filter(pl.col("primary") != d8)
        for band in (s.ALL, POOLED):
            sub = (
                part.filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120))
                if band == POOLED
                else s.in_band(part=part, band=band)
            )
            d = sub.select(
                "month", "day", d=pl.col("A11c__crps") - pl.col("A11c-null__crps")
            ).filter(pl.col("d").is_finite())
            st = s.block_stats(diff=d, rng=rng, tests=TESTS)
            rows.append({"set": name, "band": band, "contrast": "A11c - A11c-null", **st})
    pl.DataFrame(rows).write_csv(W / "scores/contrasts_cloud.csv")
    lk = leak()
    lk.write_csv(W / "scores/stress_leak_cloud.csv")
    json.dump(
        {"fine_tune_rows_with_cloud_by_fold": rows_with_cloud, "months": MONTHS},
        open(W / "scores/checks_cloud.json", "w"),
        indent=1,
    )
    print(rows_with_cloud)
    print(lk)
    with pl.Config(tbl_rows=40):
        print(pl.DataFrame(rows).select("set", "band", "difference", "low_month", "high_month"))


main()
