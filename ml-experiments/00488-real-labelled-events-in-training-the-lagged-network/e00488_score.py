"""Score the arms of experiment 488 against stored A0-inj and B-mix (CPU only, no fit).

Reads forecasts from `fit.PRED_DIR` (this experiment's folder, with read-only symlinks to experiment
404's stored `A0-inj` and `B-mix`). Row classes follow experiment 404's `report.load`: injected
scoring worlds 200 to 202 give `harm_inj`, `tgt_inj`, and `clean_inj`; the clean world (-1) gives
`harm_real`, `event` (target inside a padded real event), `after` (target in the 96 half-hours after
one), and `clean_real` (the rows with no labelled event; NGED's labels are incomplete). The `event`
class is also scored against experiment 435's NRA-corrected actuals at depth factors 0.75, 1.0,
and 1.25. Tail variants: raw and the 1.15 widening of experiment 435.

```bash
PYTHONPATH=<as run488.sh> DATA_PATH_INTERNAL=<private data> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00488-*/e00488_score.py cells
uv run --no-sync python ml-experiments/00488-*/e00488_score.py pool
```
"""

import argparse
from typing import Final

import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import e00488_fit as fit488  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

OUT_DIR: Final = fit488.WORK_DIR / "scores"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
CELLS: Final[tuple[tuple[str, int], ...]] = tuple(
    (m, s) for m in MONTHS for s in ((0, 1, 2) if m == "2026-06" else (0,))
)
NEW_ARMS: Final[tuple[str, ...]] = ("RealMix", "PT-FT", "RealMix-shuf")
REFERENCES: Final[tuple[str, ...]] = ("A0-inj", "B-mix")
WORLDS: Final[list[int]] = [-1, *score.SCORING_SEEDS]
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight", "report_only")
CLASSES: Final[tuple[str, ...]] = (
    "harm_inj", "tgt_inj", "clean_inj", "harm_real", "event", "after", "clean_real",
)  # fmt: skip
DEPTHS: Final[tuple[str, ...]] = ("0.75", "1.0", "1.25")
CONTRAST_COUNT: Final[int] = 12
RESAMPLES: Final[int] = 20000
MDE_MULTIPLIER: Final[float] = 2.8
TAILS: Final[tuple[str, ...]] = tuple(tailcore.TAIL_LEVELS)


def link_stored() -> None:
    """Link experiment 404's stored A0-inj and B-mix forecasts into this experiment's folder."""
    stored = fit488.STORED_DIR.parent / "predictions"
    fit488.fit.PRED_DIR.mkdir(parents=True, exist_ok=True)
    for arm in REFERENCES:
        link = fit488.fit.PRED_DIR / arm
        if not link.exists():
            link.symlink_to(stored / arm, target_is_directory=True)


def has(*, arm: str, month: str, seed: int) -> bool:
    """Return whether the arm has forecasts for every world of the cell."""
    return all(
        score.prediction_path(arm=arm, month=month, seed=seed, world=w).exists() for w in WORLDS
    )


def load_cell(*, arm: str, month: str, seed: int) -> pl.DataFrame:
    """Return an arm's forecasts for a cell over all worlds, with groups and classes."""
    frame = report.load(arm=arm, month=month, seed=seed, worlds=WORLDS)
    frame = frame.filter(tailcore.lead_band_expr().is_not_null())
    return frame.with_columns(
        cls=pl.when((pl.col("cls") == "tgt_real") & pl.col("tgt_real"))
        .then(pl.lit("event"))
        .when(pl.col("cls") == "tgt_real")
        .then(pl.lit("after"))
        .otherwise(pl.col("cls"))
    )


def widened(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with the 1.15 widening applied and the score recomputed."""
    typed = frame.cast(dict.fromkeys(("q50", *TAILS), pl.Float64))
    out = tailcore.widen(frame=typed, factor=tailcore.WIDENING)
    out = out.cast(dict.fromkeys(("q50", *TAILS), pl.Float32))
    return out.with_columns(crps=score.pinball_points(frame=out))


def tail_row(*, frame: pl.DataFrame, prefix: str) -> dict[str, float]:
    """Return coverage and pinball loss at the four tail quantiles, prefixed."""
    table = tailcore.tail_table(
        frame=frame.cast({"actual": pl.Float64}).with_columns(all=pl.lit(1)), by=("all",)
    )
    row = table.row(0, named=True)
    return {f"{prefix}{k}": v for k, v in row.items() if k not in {"n", "all"}}


def cell_rows(*, arm: str, ref: str, month: str, seed: int, real: np.ndarray) -> list[dict]:
    """Return contrast rows of one arm against one reference for a cell."""
    ref0 = load_cell(arm=ref, month=month, seed=seed)
    arm0 = load_cell(arm=arm, month=month, seed=seed)
    keys = ["world", "primary", "origin", "target"]
    if not ref0.select(keys).equals(arm0.select(keys)):
        msg = f"{arm} rows differ from {ref} in {month} seed {seed}"
        raise ValueError(msg)
    clusters = report.cluster_ids(frame=ref0, real=real)
    shares = np.load(fit488.SHARE_PATH)
    out = []
    for variant in ("raw", "widen1.15"):
        a_all = arm0 if variant == "raw" else widened(frame=arm0)
        r_all = ref0 if variant == "raw" else widened(frame=ref0)
        for group in GROUPS:
            for cls in CLASSES:
                mask = ((ref0["group"] == group) & (ref0["cls"] == cls)).to_numpy()
                if not mask.any():
                    continue
                depths = ("metered", *DEPTHS) if cls == "event" else ("metered",)
                for depth in depths:
                    a, r = a_all.filter(pl.Series(mask)), r_all.filter(pl.Series(mask))
                    if depth != "metered":
                        p = r["primary"].to_numpy().astype(np.int64)
                        g = np.clip(r["target"].to_numpy().astype(np.int64), 0, 121247)
                        adj = pl.Series(r["actual"].to_numpy() - shares[f"f{depth}"][p, g])
                        a = a.with_columns(actual=adj.cast(pl.Float32))
                        r = r.with_columns(actual=adj.cast(pl.Float32))
                        a = a.with_columns(crps=score.pinball_points(frame=a))
                        r = r.with_columns(crps=score.pinball_points(frame=r))
                    cl = clusters[mask]
                    point, low, high = report.contrast(a=a, b=r, cluster=cl)
                    many = len(np.unique(cl)) > 1
                    drop = (
                        report.contrast(a=a, b=r, cluster=cl, drop_largest=True)[0]
                        if many
                        else np.nan
                    )
                    primaries = a["primary"].to_numpy()
                    diff = a["crps"].to_numpy() - r["crps"].to_numpy()
                    loo = [diff[primaries != p].mean() for p in np.unique(primaries)]
                    out.append(
                        {
                            "variant": variant, "arm": arm, "ref": ref, "group": group, "cls": cls,
                            "depth": depth, "month": month, "seed": seed, "rows": int(mask.sum()),
                            "events": len(np.unique(cl)),
                            "arm_score": float(a["crps"].mean()),  # ty: ignore[invalid-argument-type]
                            "ref_score": float(r["crps"].mean()),  # ty: ignore[invalid-argument-type]
                            "diff": point, "low": low, "high": high, "diff_drop_largest": drop,
                            "loo_primary_min": float(min(loo)) if len(loo) > 1 else np.nan,
                            "loo_primary_max": float(max(loo)) if len(loo) > 1 else np.nan,
                            **(tail_row(frame=a, prefix="arm_") if depth == "metered" else {}),
                            **(tail_row(frame=r, prefix="ref_") if depth == "metered" else {}),
                        }
                    )  # fmt: skip
    return out


def cells(*, arms: list[str]) -> None:
    """Write one row per variant, arm, reference, group, class, depth, and cell."""
    link_stored()
    real = score.real_event_mask()
    rows = []
    for month, seed in CELLS:
        for arm in arms:
            if not has(arm=arm, month=month, seed=seed):
                continue
            for ref in REFERENCES:
                if has(arm=ref, month=month, seed=seed):
                    rows += cell_rows(arm=arm, ref=ref, month=month, seed=seed, real=real)
        print(f"{month} seed {seed} done", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT_DIR / "cells.parquet")


def month_bootstrap(*, values: np.ndarray, level: float, seed: int = 0) -> tuple[float, float]:
    """Return the percentile interval of the mean of month values, resampling months."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(RESAMPLES, len(values)))
    means = values[draws].mean(1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def pool() -> None:
    """Pool the cells over months (seeds averaged first) and write the contrast table."""
    cell = pl.read_parquet(OUT_DIR / "cells.parquet")
    months = cell.group_by("variant", "arm", "ref", "group", "cls", "depth", "month").agg(
        pl.col("diff").mean(),
        pl.col("diff_drop_largest").mean(),
        pl.col("arm_score").mean(),
        pl.col("ref_score").mean(),
        pl.col("events").first(),
        pl.col("rows").first(),
        pl.col("loo_primary_min").mean(),
        pl.col("loo_primary_max").mean(),
    )
    level = 1.0 - 0.05 / CONTRAST_COUNT
    rows = []
    keys = ("variant", "arm", "ref", "group", "cls", "depth")
    for key, part in months.group_by(*keys, maintain_order=True):
        d = part["diff"].to_numpy()
        dd = part["diff_drop_largest"].to_numpy()
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        lo, hi = month_bootstrap(values=d, level=0.95)
        lob, hib = month_bootstrap(values=d, level=level)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)), "months": len(d),
                "events_total": int(part["events"].sum()), "rows_total": int(part["rows"].sum()),
                "arm_score": float(part["arm_score"].mean()),  # ty: ignore[invalid-argument-type]
                "ref_score": float(part["ref_score"].mean()),  # ty: ignore[invalid-argument-type]
                "mean_of_months": float(d.mean()), "sign_negative": int((d < 0).sum()),
                "low95": lo, "high95": hi, "low_bonf": lob, "high_bonf": hib,
                "drop_largest_mean": float(np.nanmean(dd)) if len(dd) else np.nan,
                "loo_primary_min": float(part["loo_primary_min"].min()),  # ty: ignore[invalid-argument-type]
                "loo_primary_max": float(part["loo_primary_max"].max()),  # ty: ignore[invalid-argument-type]
                "mde_80pct_month_spread": (
                    MDE_MULTIPLIER * sd / np.sqrt(len(d)) if len(d) > 1 else np.nan
                ),
            }
        )  # fmt: skip
    table = pl.DataFrame(rows).sort("variant", "group", "cls", "depth", "ref", "arm")
    table.write_csv(OUT_DIR / "pooled.csv", float_precision=3)
    print(f"Bonferroni contrast count {CONTRAST_COUNT}, level {level:.5f}")
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(30)
    print(table.filter(pl.col("variant") == "raw"))


def seed_spread() -> None:
    """Print the 2026-06 seed spread of A0-inj and B-mix scores (seeds 0 to 2) for the MDE."""
    link_stored()
    rows = []
    for arm in REFERENCES:
        for seed in (0, 1, 2):
            frame = load_cell(arm=arm, month="2026-06", seed=seed)
            for (group, cls), part in frame.group_by("group", "cls"):
                rows.append(
                    {"arm": arm, "seed": seed, "group": group, "cls": cls,
                     "score": float(part["crps"].mean())}  # ty: ignore[invalid-argument-type]
                )  # fmt: skip
    table = (
        pl.DataFrame(rows)
        .group_by("arm", "group", "cls")
        .agg(pl.col("score").mean().alias("mean"), pl.col("score").std().alias("sd_seed"))
        .with_columns(
            mde_5_month_pool=MDE_MULTIPLIER * np.sqrt(2.0) * pl.col("sd_seed") / np.sqrt(5.0)
        )
        .sort("group", "cls", "arm")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(OUT_DIR / "seed_spread_2026-06.csv", float_precision=4)
    pl.Config.set_tbl_rows(100)
    print(table)


def main() -> None:
    """Run one scoring step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells", "pool", "seed-spread"))
    parser.add_argument("--arms", nargs="+", default=[*NEW_ARMS, *(f"{a}+LL" for a in NEW_ARMS)])
    args = parser.parse_args()
    if args.command == "cells":
        cells(arms=args.arms)
    elif args.command == "pool":
        pool()
    else:
        seed_spread()


if __name__ == "__main__":
    main()
