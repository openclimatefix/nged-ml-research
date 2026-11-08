# ruff: noqa: D103, E501
"""Experiment 648: gain over the reference by lead band and metering group, from the scored cells (CPU).

Writes `score/screen_bands.csv` (net, arm, band, group, gain in points of capacity; positive is better;
cells weighted by rows) and `score/screen_primaries.csv` (the same by Primary, 24-120 h).
"""

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
KEYS = ["prim", "label", "group", "month", "band"]


def cells(*, name: str) -> pl.DataFrame:
    return pl.read_csv(SCORE / name / "event_free_cells.csv").filter(pl.col("system") == "K2")


rows, prim_rows = [], []
for net in ("k1", "c2"):
    ref = cells(name=f"ref_{net}")
    for d in sorted(SCORE.iterdir()):
        if not d.name.startswith(f"{net}_") or not (d / "bank_summary.csv").exists():
            continue
        arm = d.name.removeprefix(f"{net}_")
        if arm == "613":
            continue
        j = ref.join(cells(name=d.name), on=KEYS, suffix="_a").with_columns(
            gain=100 * (pl.col("crps") - pl.col("crps_a"))
        )
        for (band, group), g in j.group_by(["band", "group"]):
            rows.append(
                {"net": net, "arm": arm, "band": band, "group": group,
                 "gain": float((g["gain"] * g["n_rows"]).sum() / g["n_rows"].sum())}  # ty: ignore[unsupported-operator]
            )  # fmt: skip
        for (label,), g in j.filter(pl.col("band") == "24-120 h").group_by(["label"]):
            prim_rows.append(
                {"net": net, "arm": arm, "label": label,
                 "gain": float((g["gain"] * g["n_rows"]).sum() / g["n_rows"].sum())}  # ty: ignore[unsupported-operator]
            )  # fmt: skip
pl.DataFrame(rows).write_csv(SCORE / "screen_bands.csv")
pl.DataFrame(prim_rows).write_csv(SCORE / "screen_primaries.csv")
b = (
    pl.DataFrame(rows)
    .pivot(on="band", index=["net", "arm", "group"], values="gain")
    .sort(["net", "arm", "group"])
)
with pl.Config(tbl_rows=80, float_precision=3):
    print(b)
