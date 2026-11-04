# ruff: noqa: E501, PLR0912, PLR0915, D103
"""Blend L-1d and R-DW by lead band, weights fitted walk-forward by fold, and score the blend.

Inputs are stored (read-only, no fit): experiment 511's L-1d and experiment 539's R-DW quantiles
(seed averages in `predictions/<arm>/<month>_c0.parquet`, and the two seeds in `seed_preds`), and
experiment 539's per-row score tables for the row sets. The blend is quantile averaging on the 11
stored quantiles, `w * R-DW + (1 - w) * L-1d`, widened by 1.15 about the median (as both parents
are) before scoring. For each scored fold, `w` is the grid value that minimises the mean CRPS of
the widened blend on the earlier folds, per group and lead band.

```bash
source ml-experiments/00524*/env.sh
uv run --no-sync python ml-experiments/00547-*/e00547_blend.py
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
E511_SEEDS: Final[Path] = (
    DD
    / "00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining"
    / "work"
    / "seed_preds"
)
OUT: Final[Path] = DD / "00547-lag-light-and-lag-free-blend-by-lead-band-stored-forecasts"
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
    ("blend", "R-DW", True), ("blend", "L-1d", True), ("blend", "A-pool", False),
    ("blend-half", "R-DW", False), ("R-DW", "L-1d", False),
)  # fmt: skip


def blend_frame(*, left: pl.DataFrame, right: pl.DataFrame, weight: np.ndarray) -> pl.DataFrame:
    """Return `weight * right + (1 - weight) * left` on the quantile columns, widened by `WIDEN`."""
    w = weight[:, None]
    mix = (
        w * right.select(list(LEVELS)).to_numpy() + (1.0 - w) * left.select(list(LEVELS)).to_numpy()
    )
    return widen(frame=pl.DataFrame(mix, schema=list(LEVELS), orient="row"), factor=WIDEN)


def metrics(*, quantiles: pl.DataFrame, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row CRPS and tail scores (points of capacity) of widened quantiles."""
    return score470.row_metrics(frame=quantiles.with_columns(actual=pl.Series(actual)))


def month_table(*, month: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the month's rows (actuals, sets, stored scores) and the two parents' quantiles on them."""
    base = score454.month_base(month=month).select(*KEYS, "actual")
    stored = pl.read_parquet(
        E539 / "scores" / f"rows_{month}.parquet",
        columns=[
            *KEYS,
            "group",
            "cls",
            "lead_hours",
            "cold",
            "xmas",
            "main16",
            "L-1d__crps",
            "R-DW__crps",
            "A-pool__crps",
            "manual_heuristic__crps",
        ],
    )
    rows = base.join(stored, on=KEYS, how="inner", validate="1:1", maintain_order="left")
    assert rows.height == base.height == stored.height, (rows.height, base.height, stored.height)
    parents = []
    for arm in ("L-1d", "R-DW"):
        q = pl.read_parquet(E539 / "predictions" / arm / f"{month}_c0.parquet")
        parents.append(rows.select(*KEYS).join(q, on=KEYS, how="left", maintain_order="left"))
    return rows, parents[0], parents[1]


def seed_parent(
    *, arm: str, month: str, seed: int, rows: pl.DataFrame, sites: pl.DataFrame
) -> pl.DataFrame:
    """Return one seed's quantiles of an arm on the rows."""
    folder = (E511_SEEDS if arm == "L-1d" else E539 / "seed_preds") / arm
    return build470.on_rows(
        rows=rows,
        frame=build470.pool_stored(folder=folder, month=month, sites=sites, seeds=(seed,)),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores").mkdir(exist_ok=True)
    sites = build470.site_map()
    tables, grids = {}, {}
    for month in MONTHS:
        rows, left, right = month_table(month=month)
        rows = rows.with_columns(
            band=s.band_expr(), month=pl.lit(month), day=(pl.col("origin") // 48).cast(pl.Int32)
        )
        finite = lambda frame: frame["q50"].is_finite().fill_null(value=False).to_numpy()  # noqa: E731
        ok = finite(left) & finite(right) & rows["actual"].is_finite().to_numpy()
        rows = rows.with_columns(main16=pl.col("main16") & pl.Series(ok))
        left, right = left.fill_null(0.0), right.fill_null(0.0)
        actual = rows["actual"].to_numpy().astype(np.float64)
        # CRPS of every grid weight, for all rows (the fit reads the earlier folds' columns).
        grids[month] = np.stack(
            [
                metrics(
                    quantiles=blend_frame(left=left, right=right, weight=np.full(rows.height, w)),
                    actual=actual,
                )["crps"]
                for w in GRID
            ],
            axis=1,
        )
        for end, col in ((0, "L-1d__crps"), (len(GRID) - 1, "R-DW__crps")):
            stored_crps = rows[col].to_numpy().astype(np.float64)
            keep = ok & np.isfinite(stored_crps)
            err = float(np.abs(grids[month][keep, end] - stored_crps[keep]).max())
            assert err < 1e-2, (month, col, err)
        tables[month] = (rows, left, right, actual)
        print(month, rows.height, "rows; parents reproduced from the stored quantiles", flush=True)

    # Weights: per scored fold, group, and band, from the earlier folds only.
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
                    sel = (
                        rows["main16"] & (rows["group"] == group) & (rows["band"] == band)
                    ).to_numpy()
                    if group == "signed_six":
                        sel &= (rows["cls"] == "clean").to_numpy()
                    cols.append(grids[m][sel])
                    n += int(sel.sum())
                stack = np.concatenate(cols)
                means = stack.mean(axis=0)
                weights.append(
                    {
                        "fold": month,
                        "set": name,
                        "band": band,
                        "weight": float(GRID[int(np.argmin(means))]),
                        "train_rows": n,
                        "crps_at_weight": float(means.min()),
                        "crps_l1d": float(means[0]),
                        "crps_rdw": float(means[-1]),
                    }
                )
    wtab = pl.DataFrame(weights)
    wtab.write_csv(OUT / "scores" / "weights.csv")
    lookup = {(r["fold"], r["set"], r["band"]): r["weight"] for r in weights}

    # Score the blend (and the fixed half-half blend as a report-only reference) on the scored folds.
    outputs, example = [], None
    for month in SCORED:
        rows, left, right, actual = tables[month]
        group_name = {v: k for k, v in GROUPS.items()}
        w = np.array(
            [
                lookup[(month, group_name[g], b)]
                for g, b in zip(rows["group"].to_list(), rows["band"].to_list(), strict=True)
            ]
        )
        quant = blend_frame(left=left, right=right, weight=w)
        got = metrics(quantiles=quant, actual=actual)
        half = metrics(
            quantiles=blend_frame(left=left, right=right, weight=np.full(rows.height, 0.5)),
            actual=actual,
        )
        extra = {f"blend__{k}": v for k, v in got.items()} | {
            f"blend-half__{k}": v for k, v in half.items()
        }
        extra["weight"] = w
        for seed in (0, 1):
            ls = seed_parent(arm="L-1d", month=month, seed=seed, rows=rows, sites=sites).fill_null(
                0.0
            )
            rs = seed_parent(arm="R-DW", month=month, seed=seed, rows=rows, sites=sites).fill_null(
                0.0
            )
            extra[f"blend-seed{seed}__crps"] = metrics(
                quantiles=blend_frame(left=ls, right=rs, weight=w), actual=actual
            )["crps"]
            extra[f"L-1d-seed{seed}__crps"] = metrics(
                quantiles=widen(frame=ls, factor=WIDEN), actual=actual
            )["crps"]
        out = rows.hstack(pl.DataFrame({k: pl.Series(k, v) for k, v in extra.items()}))
        out.write_parquet(OUT / "scores" / f"rows_{month}.parquet")
        outputs.append(out)
        if example is None:
            i = int(
                np.flatnonzero(
                    (rows["group"] == "mva_eight").to_numpy()
                    & (rows["band"] == "48-72h").to_numpy()
                    & rows["main16"].to_numpy()
                )[0]
            )
            example = {
                "fold": month,
                "primary": int(rows["primary"][i]),
                "origin_index": int(rows["origin"][i]),
                "target_index": int(rows["target"][i]),
                "lead_hours": float(rows["lead_hours"][i]),
                "weight_on_R-DW": float(w[i]),
                "L-1d_q50": float(left["q50"][i]),
                "R-DW_q50": float(right["q50"][i]),
                "blend_q50": float(quant["q50"][i]),
                "actual": float(actual[i]),
            }
            assert example["target_index"] > example["origin_index"]
            assert example["lead_hours"] > 0
            print("example row:", example)
    table = pl.concat(outputs)

    rng = np.random.default_rng(0)
    arms = {
        "blend": "blend__crps",
        "blend-half": "blend-half__crps",
        "R-DW": "R-DW__crps",
        "L-1d": "L-1d__crps",
        "A-pool": "A-pool__crps",
        "manual_heuristic": "manual_heuristic__crps",
    }
    absolute, contrasts = [], []
    for name, group in GROUPS.items():
        part = table.filter(pl.col("main16") & (pl.col("group") == group))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        for band in (*s.BANDS, POOLED, s.ALL):
            sub = (
                part.filter(pl.col("band").is_in(POOLED_BANDS))
                if band == POOLED
                else s.in_band(part=part, band=band)
            )
            if sub.height == 0:
                continue
            for arm, col in arms.items():
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": arm,
                        "n": sub.height,
                        "crps": float(sub.filter(pl.col(col).is_finite())[col].mean()),
                    }
                )
            for tail in ("q01", "q05", "q95", "q99"):
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": "blend",
                        "n": sub.height,
                        "metric": f"pin_{tail}",
                        "value": float(sub[f"blend__pin_{tail}"].mean()),
                    }
                )
                absolute.append(
                    {
                        "set": name,
                        "band": band,
                        "arm": "blend",
                        "n": sub.height,
                        "metric": f"cov_{tail}",
                        "value": float(sub[f"blend__cov_{tail}"].mean()),
                    }
                )
            seedmat = (
                sub.group_by("month")
                .agg([pl.col(c).mean() for c in ("blend-seed0__crps", "blend-seed1__crps")])
                .sort("month")
                .select("blend-seed0__crps", "blend-seed1__crps")
                .to_numpy()
            )
            f, k = seedmat.shape
            resid = seedmat - seedmat.mean(axis=1, keepdims=True)
            mde = (
                1.96
                * math.sqrt(2)
                * math.sqrt(float((resid**2).sum()) / (f * (k - 1)))
                / math.sqrt(k)
            )
            for a, b, counted in CONTRASTS:
                d = sub.select("month", "day", d=pl.col(arms[a]) - pl.col(arms[b])).filter(
                    pl.col("d").is_finite()
                )
                contrasts.append(
                    {
                        "set": name,
                        "band": band,
                        "contrast": f"{a} - {b}",
                        "counted": counted and band in s.BANDS and name != "report",
                        "tests": TESTS,
                        "mde": mde,
                        **s.block_stats(diff=d, rng=rng, tests=TESTS),
                    }
                )
    pl.DataFrame(absolute).write_csv(OUT / "scores" / "absolute.csv")
    pl.DataFrame(contrasts).write_csv(OUT / "scores" / "contrasts.csv")
    checks = {
        "scored_folds": list(SCORED),
        "first_fold_unscored": MONTHS[0],
        "rows_scored": table.height,
        "rows_main16_scored": int(table["main16"].sum()),
        "example_row": example,
        "weights_fitted_on_earlier_folds_only": True,
        "tests": TESTS,
        "month_blocks": len(SCORED),
    }
    (OUT / "scores" / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


main()
