# ruff: noqa: E501, PLR0912, PLR0915, D103
"""Blend R-DW with the retrained K2-S1 by lead band, weights fitted walk-forward, and score it.

Inputs are stored (read-only, no fit): experiment 539's R-DW quantiles (seed average in
`predictions/R-DW/<month>_c0.parquet`) and the three cells of experiment 541's `K2-S1-origin-trained`
(K2-S1 with lag-free heads retrained on weather from the run issued at or before the forecast
origin). Each cell is blended with the R-DW seed average, `w * K2-S1 cell + (1 - w) * R-DW`, widened
by 1.15 about the median, and the three cells' CRPS are averaged, as experiment 541 scores K2-S1.
For each scored fold, `w` is the grid value that minimises the mean CRPS on the earlier folds, per
group and lead band. Copied from experiment 547's script.

```bash
source ml-experiments/00524*/env.sh
uv run --no-sync python ml-experiments/00549-*/e00549_blend.py
```
"""

import json
import math
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]

DD: Final[Path] = Path(build470.ML_EXPERIMENTS_DATA_DIR)
E539: Final[Path] = DD / "00539-l1d-plus-day-and-week-lags" / "work"
K2: Final[Path] = DD / "00541-k2-s1-rebuilt-on-origin-time-weather" / "work" / "cells" / "K2-S1-origin-trained"
OUT: Final[Path] = DD / "00549-k2-s1-blend-and-climatology-stored-check"
CELLS: Final[tuple[int, ...]] = (0, 1, 2)
MONTHS: Final[tuple[str, ...]] = s.MONTHS
SCORED: Final[tuple[str, ...]] = MONTHS[1:]
KEYS: Final[list[str]] = ["primary", "origin", "target"]
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
GRID: Final[np.ndarray] = np.round(np.linspace(0.0, 1.0, 11), 1)
WIDEN: Final[float] = 1.15
GROUPS: Final[dict[str, str]] = {"eight": "mva_eight", "six": "signed_six", "report": "report_only"}
POOLED: Final[str] = "24-120h"
POOLED_BANDS: Final[tuple[str, ...]] = ("24-48h", "48-72h", "72-120h")
TESTS: Final[int] = 28
CONTRASTS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("blend", "R-DW", True), ("blend", "K2-S1", True), ("blend", "A-pool", False),
    ("blend-half", "R-DW", False), ("K2-S1", "R-DW", False),
)  # fmt: skip


def blend_frame(*, left: pl.DataFrame, right: pl.DataFrame, weight: np.ndarray) -> pl.DataFrame:
    """Return `weight * right + (1 - weight) * left` on the quantile columns, widened by `WIDEN`.

    Here `left` is R-DW and `right` is one K2-S1 cell."""
    w = weight[:, None]
    mix = (
        w * right.select(list(LEVELS)).to_numpy() + (1.0 - w) * left.select(list(LEVELS)).to_numpy()
    )
    return widen(frame=pl.DataFrame(mix, schema=list(LEVELS), orient="row"), factor=WIDEN)


def metrics(*, quantiles: pl.DataFrame, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row CRPS and tail scores (points of capacity) of widened quantiles."""
    return score470.row_metrics(frame=quantiles.with_columns(actual=pl.Series(actual)))


def month_table(*, month: str) -> tuple[pl.DataFrame, pl.DataFrame, list[pl.DataFrame]]:
    """Return the month's rows, R-DW's quantiles, and the three K2-S1 cells' quantiles on them."""
    base = score454.month_base(month=month).select(*KEYS, "actual")
    stored = pl.read_parquet(
        E539 / "scores" / f"rows_{month}.parquet",
        columns=[*KEYS, "group", "cls", "lead_hours", "cold", "xmas", "main16",
                 "R-DW__crps", "A-pool__crps", "manual_heuristic__crps"],
    )  # fmt: skip
    rows = base.join(stored, on=KEYS, how="inner", validate="1:1", maintain_order="left")
    assert rows.height == base.height == stored.height, (rows.height, base.height, stored.height)
    q = pl.read_parquet(E539 / "predictions" / "R-DW" / f"{month}_c0.parquet")
    rdw = rows.select(*KEYS).join(q, on=KEYS, how="left", maintain_order="left")
    cells = []
    for c in CELLS:
        cell = pl.read_parquet(K2 / f"{month}_c{c}.parquet").select(*KEYS, *LEVELS)
        cells.append(rows.select(*KEYS).join(cell, on=KEYS, how="left", maintain_order="left"))
    return rows, rdw, cells


def mean_crps(*, left: pl.DataFrame, cells: list[pl.DataFrame], weight: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the per-row CRPS of the blend of `left` with each cell, averaged over the cells."""
    return np.mean(
        [metrics(quantiles=blend_frame(left=left, right=cell, weight=weight), actual=actual)["crps"] for cell in cells],
        axis=0,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores").mkdir(exist_ok=True)
    tables, grids = {}, {}
    for month in MONTHS:
        rows, left, cells = month_table(month=month)
        rows = rows.with_columns(
            band=s.band_expr(), month=pl.lit(month), day=(pl.col("origin") // 48).cast(pl.Int32)
        )
        finite = lambda frame: frame["q50"].is_finite().fill_null(value=False).to_numpy()  # noqa: E731
        ok = finite(left) & rows["actual"].is_finite().to_numpy()
        for cell in cells:
            ok &= finite(cell)
        assert (rows["target"] > rows["origin"]).all()
        rows = rows.with_columns(main16=pl.col("main16") & pl.Series(ok))
        left = left.fill_null(0.0)
        cells = [cell.fill_null(0.0) for cell in cells]
        actual = rows["actual"].to_numpy().astype(np.float64)
        grids[month] = np.stack(
            [mean_crps(left=left, cells=cells, weight=np.full(rows.height, w), actual=actual) for w in GRID],
            axis=1,
        )
        stored_crps = rows["R-DW__crps"].to_numpy().astype(np.float64)
        keep = ok & np.isfinite(stored_crps)
        err = float(np.abs(grids[month][keep, 0] - stored_crps[keep]).max())
        assert err < 1e-2, (month, "R-DW", err)
        k2_stored = pl.read_parquet(
            DD / "00541-k2-s1-rebuilt-on-origin-time-weather" / "work" / f"new_{month}.parquet",
            columns=[*KEYS, "K2-S1-origin-trained__crps"],
        )
        k2_row = rows.select(*KEYS).join(k2_stored, on=KEYS, how="left", maintain_order="left")
        k2_crps = k2_row["K2-S1-origin-trained__crps"].to_numpy().astype(np.float64)
        keep = ok & np.isfinite(k2_crps)
        err = float(np.abs(grids[month][keep, -1] - k2_crps[keep]).max())
        assert err < 1e-2, (month, "K2-S1", err)
        rows = rows.with_columns(**{"K2-S1__crps": pl.Series(grids[month][:, -1])})
        tables[month] = (rows, left, cells, actual)
        print(month, rows.height, "rows; both parents reproduced from the stored quantiles", flush=True)

    weights = []
    for i, month in enumerate(MONTHS):
        if i == 0:
            continue
        earlier = MONTHS[:i]
        assert all(m < month for m in earlier)
        for name, group in GROUPS.items():
            for band in s.BANDS:
                cols, n = [], 0
                for m in earlier:
                    rows = tables[m][0]
                    sel = (rows["main16"] & (rows["group"] == group) & (rows["band"] == band)).to_numpy()
                    if group == "signed_six":
                        sel &= (rows["cls"] == "clean").to_numpy()
                    cols.append(grids[m][sel])
                    n += int(sel.sum())
                means = np.concatenate(cols).mean(axis=0)
                weights.append(
                    {"fold": month, "set": name, "band": band, "weight": float(GRID[int(np.argmin(means))]),
                     "train_rows": n, "crps_at_weight": float(means.min()),
                     "crps_rdw": float(means[0]), "crps_k2": float(means[-1])}
                )  # fmt: skip
    wtab = pl.DataFrame(weights)
    wtab.write_csv(OUT / "scores" / "weights.csv")
    lookup = {(r["fold"], r["set"], r["band"]): r["weight"] for r in weights}

    outputs, example = [], None
    group_name = {v: k for k, v in GROUPS.items()}
    for month in SCORED:
        rows, left, cells, actual = tables[month]
        w = np.array([lookup[(month, group_name[g], b)] for g, b in zip(rows["group"].to_list(), rows["band"].to_list(), strict=True)])
        extra = {"weight": w, "blend__crps": mean_crps(left=left, cells=cells, weight=w, actual=actual),
                 "blend-half__crps": mean_crps(left=left, cells=cells, weight=np.full(rows.height, 0.5), actual=actual)}  # fmt: skip
        tails = [metrics(quantiles=blend_frame(left=left, right=cell, weight=w), actual=actual) for cell in cells]
        for k in ("pin_q01", "pin_q05", "pin_q95", "pin_q99", "cov_q01", "cov_q05", "cov_q95", "cov_q99"):
            extra[f"blend__{k}"] = np.mean([t[k] for t in tails], axis=0)
        for c, cell in enumerate(cells):
            extra[f"blend-cell{c}__crps"] = metrics(quantiles=blend_frame(left=left, right=cell, weight=w), actual=actual)["crps"]
        out = rows.hstack(pl.DataFrame({k: pl.Series(k, v) for k, v in extra.items()}))
        out.write_parquet(OUT / "scores" / f"rows_{month}.parquet")
        outputs.append(out)
        if example is None:
            i = int(np.flatnonzero((rows["group"] == "mva_eight").to_numpy() & (rows["band"] == "48-72h").to_numpy() & rows["main16"].to_numpy())[0])
            quant = blend_frame(left=left, right=cells[0], weight=w)
            example = {
                "fold": month, "primary": int(rows["primary"][i]), "origin_index": int(rows["origin"][i]),
                "target_index": int(rows["target"][i]), "lead_hours": float(rows["lead_hours"][i]),
                "weight_on_K2-S1": float(w[i]), "R-DW_q50": float(left["q50"][i]),
                "K2-S1_cell0_q50": float(cells[0]["q50"][i]), "blend_cell0_q50": float(quant["q50"][i]),
                "actual": float(actual[i]),
            }  # fmt: skip
            assert example["target_index"] > example["origin_index"] and example["lead_hours"] > 0
            print("example row:", example)
    table = pl.concat(outputs)

    rng = np.random.default_rng(0)
    arms = {"blend": "blend__crps", "blend-half": "blend-half__crps", "R-DW": "R-DW__crps", "K2-S1": "K2-S1__crps",
            "A-pool": "A-pool__crps", "manual_heuristic": "manual_heuristic__crps"}  # fmt: skip
    absolute, contrasts = [], []
    for name, group in GROUPS.items():
        part = table.filter(pl.col("main16") & (pl.col("group") == group))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        for band in (*s.BANDS, POOLED, s.ALL):
            sub = part.filter(pl.col("band").is_in(POOLED_BANDS)) if band == POOLED else s.in_band(part=part, band=band)
            if sub.height == 0:
                continue
            for arm, col in arms.items():
                absolute.append({"set": name, "band": band, "arm": arm, "n": sub.height,
                                 "crps": float(sub.filter(pl.col(col).is_finite())[col].mean())})  # fmt: skip
            for tail in ("q01", "q05", "q95", "q99"):
                for kind in ("pin", "cov"):
                    absolute.append({"set": name, "band": band, "arm": "blend", "n": sub.height,
                                     "metric": f"{kind}_{tail}", "value": float(sub[f"blend__{kind}_{tail}"].mean())})  # fmt: skip
            seedmat = (sub.group_by("month").agg([pl.col(f"blend-cell{c}__crps").mean() for c in CELLS])
                       .sort("month").select([f"blend-cell{c}__crps" for c in CELLS]).to_numpy())  # fmt: skip
            f, k = seedmat.shape
            resid = seedmat - seedmat.mean(axis=1, keepdims=True)
            mde = 1.96 * math.sqrt(2) * math.sqrt(float((resid**2).sum()) / (f * (k - 1))) / math.sqrt(k)
            for a, b, counted in CONTRASTS:
                d = sub.select("month", "day", d=pl.col(arms[a]) - pl.col(arms[b])).filter(pl.col("d").is_finite())
                contrasts.append({"set": name, "band": band, "contrast": f"{a} - {b}",
                                  "counted": counted and band in s.BANDS and name != "report",
                                  "tests": TESTS, "mde": mde, **s.block_stats(diff=d, rng=rng, tests=TESTS)})  # fmt: skip
    pl.DataFrame(absolute).write_csv(OUT / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(OUT / "scores" / "contrasts.csv")
    checks = {"scored_folds": list(SCORED), "first_fold_unscored": MONTHS[0], "rows_scored": table.height,
              "rows_main16_scored": int(table["main16"].sum()), "example_row": example,
              "weights_fitted_on_earlier_folds_only": True, "tests": TESTS, "month_blocks": len(SCORED)}  # fmt: skip
    (OUT / "scores" / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


main()
