"""Experiment 435, step 3 (provisional): table P fitted on stored experiment 428 seed-0 forecasts.

Provisional because only seed 0 is on disk, only two condition classes exist for the injected
worlds (`inj_clean` and `inj_harm`, from experiment 428's flags), and each fold is fitted on the
earlier folds only (2025-10 has no earlier fold, so it is never scored). For each arm, metering
group, lead band, and tail side, one factor scales the distance of the quantile from the median so
that the coverage on the fitting rows equals the nominal level:

- upper tail (q95, q99): the factor is the nominal-level quantile of (actual - q50) / (q - q50);
- lower tail (q05, q01): the factor is the (1 - nominal)-quantile of (q50 - actual) / (q50 - q).

Rows where q equals q50 are dropped from the fit. Held-out scoring reports coverage and pinball
loss at the four tail quantiles raw, widened by 1.15, and after P, for the classes `clean` (world
0 rows outside real events), `inj_clean`, and `inj_harm`. Output:
`work/table_p_provisional_428.parquet`.

```bash
uv run --no-sync python ml-experiments/00435-*/e00435_table_p_provisional.py
```
"""

from pathlib import Path
from typing import Final

import polars as pl
from e00435_tailcore import (  # ty: ignore[unresolved-import]
    TAIL_LEVELS,
    WIDENING,
    clip_to,
    lead_band_expr,
    tail_table,
    widen,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
)
PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs"
    / "work"
    / "predictions"
)
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
ARMS: Final[tuple[str, ...]] = ("R", "R-inj")
LABELS: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D15",
    "D16",
    "D18",
    "D19",
    "D21",
    "D22",
)
GROUP_OF: Final[dict[str, str]] = {
    **dict.fromkeys(("D5", "D9", "D10", "D11", "D14", "D16"), "signed-MW six"),
    **dict.fromkeys(("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"), "MVA-only eight"),
    **dict.fromkeys(("D15", "D18"), "D15 and D18"),
}
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
MIN_FIT_ROWS: Final[int] = 200
"""A cell with fewer fitting rows keeps the factor 1.0 (no factor row)."""


def load(*, arm: str, month: str) -> pl.DataFrame:
    """Return one fold of one arm with `group`, `lead_band`, and `cls` columns."""
    parts: list[pl.DataFrame] = []
    for world in (0, 4, 5, 6):
        f = PRED_DIR / arm / f"{month}_s0_w{world}.parquet"
        frame = pl.read_parquet(f).with_columns(world=pl.lit(world))
        if world == 0:
            cls = (~pl.col("win_real") & ~pl.col("tgt_post")).replace_strict(
                {True: "clean", False: None}, return_dtype=pl.String
            )
        else:
            clean = (
                ~pl.col("win_inj") & ~pl.col("win_real") & ~pl.col("tgt_post") & ~pl.col("tgt_inj")
            )
            harm = pl.col("win_inj") & ~pl.col("tgt_post") & ~pl.col("tgt_inj")
            cls = pl.when(clean).then(pl.lit("inj_clean")).when(harm).then(pl.lit("inj_harm"))
        parts.append(frame.with_columns(cls=cls))
    return (
        pl.concat(parts)
        .filter(pl.col("cls").is_not_null() & pl.col("actual").is_finite())
        .with_columns(
            label=pl.col("primary").replace_strict(dict(enumerate(LABELS)), return_dtype=pl.String),
            lead_hours=pl.col("lead").cast(pl.Float64),
            **{c: pl.col(c).cast(pl.Float64) for c in ("actual", "q50", *TAIL_LEVELS)},
        )
        .with_columns(group=pl.col("label").replace_strict(GROUP_OF), lead_band=lead_band_expr())
        .filter(pl.col("lead_band").is_not_null())
    )


def fit_factors(*, fit: pl.DataFrame) -> pl.DataFrame:
    """Return one factor per (group, lead_band, tail quantile) from the fitting rows."""
    rows: list[dict] = []
    for (group, band), part in fit.group_by("group", "lead_band"):
        for name, tau in TAIL_LEVELS.items():
            spread = (pl.col(name) - pl.col("q50")).abs()
            if name in UPPER:
                ratio, level = (pl.col("actual") - pl.col("q50")) / spread, tau
            else:
                ratio, level = (pl.col("q50") - pl.col("actual")) / spread, 1.0 - tau
            use = part.filter(spread > 1e-6)
            if use.height < MIN_FIT_ROWS:
                continue
            rows.append(
                {
                    "group": group,
                    "lead_band": band,
                    "tail": name,
                    "factor": float(use.select(ratio.quantile(level)).item()),
                    "n_fit": use.height,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "group": pl.String, "lead_band": pl.String, "tail": pl.String,
            "factor": pl.Float64, "n_fit": pl.Int64,
        },
    )  # fmt: skip


def apply_factors(
    *,
    frame: pl.DataFrame,
    factors: pl.DataFrame,
    bounds: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its fitted factor, then clip."""
    out = frame
    for name in TAIL_LEVELS:
        f = factors.filter(pl.col("tail") == name).select("group", "lead_band", "factor")
        out = (
            out.join(f, on=["group", "lead_band"], how="left")
            .with_columns(
                clip_to(
                    expr=pl.col("q50")
                    + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")),
                    bounds=bounds,
                ).alias(name)
            )
            .drop("factor")
        )
    return out


def main() -> None:
    """Fit on earlier folds, score on the next fold, for each arm."""
    results: list[pl.DataFrame] = []
    factor_rows: list[pl.DataFrame] = []
    for arm in ARMS:
        folds = {m: load(arm=arm, month=m) for m in MONTHS}
        for i, month in enumerate(MONTHS[1:], start=1):
            fit_rows = pl.concat([folds[m] for m in MONTHS[:i]])
            held = folds[month]
            # One table per class: fit on the same class's earlier rows.
            for cls in ("clean", "inj_clean", "inj_harm"):
                part_fit = fit_rows.filter(pl.col("cls") == cls)
                part_held = held.filter(pl.col("cls") == cls)
                factors = fit_factors(fit=part_fit)
                factor_rows.append(
                    factors.with_columns(arm=pl.lit(arm), month=pl.lit(month), cls=pl.lit(cls))
                )
                variants = {
                    "raw": part_held,
                    "w1.15": widen(frame=part_held, factor=WIDENING),
                    "P": apply_factors(frame=part_held, factors=factors),
                }
                for variant, frame in variants.items():
                    results.append(
                        tail_table(frame=frame, by=("group", "lead_band")).with_columns(
                            arm=pl.lit(arm),
                            month=pl.lit(month),
                            cls=pl.lit(cls),
                            variant=pl.lit(variant),
                        )
                    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(results).write_parquet(WORK_DIR / "table_p_provisional_428.parquet")
    pl.concat(factor_rows).write_parquet(WORK_DIR / "table_p_factors_428.parquet")
    pooled = (
        pl.concat(results)
        .group_by("arm", "cls", "variant", "group", "lead_band")
        .agg(
            pl.col("n").sum(),
            *[(pl.col(f"cov_{q}") * pl.col("n")).sum() / pl.col("n").sum() for q in TAIL_LEVELS],
        )
        .sort("arm", "group", "lead_band", "cls", "variant")
    )
    pl.Config.set_tbl_rows(120)
    pl.Config.set_tbl_cols(20)
    print(pooled.filter(pl.col("group") == "signed-MW six", pl.col("lead_band") == "12-36h"))


if __name__ == "__main__":
    main()
