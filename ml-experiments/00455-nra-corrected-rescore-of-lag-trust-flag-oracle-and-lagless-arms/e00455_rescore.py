"""Rescore rows inside real load-shift events against NRA-corrected actuals (CPU only, no fit).

Experiment 404's audit (`e00404_audit.py nra`) rescored B0, B1, B3, B4c, B6, S1, and A0-inj against
experiment 435's NRA-corrected actuals. This script adds the lagless head and the three flag-oracle
mixtures of experiment 445 (O-hard, O-soft, O-floor) to that rescore. It reads stored forecasts
only. Rows are the real-event rows (`tgt_real` or `tgt_post`: the target lies inside a padded real
event or in the 96 half-hours after one), at the signed-MW six and the MVA-only eight, with the
metered actual and with the NRA-corrected actual at depth factors 0.75, 1.0, and 1.25.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights:\
ml-experiments/00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails \
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00455-*/e00455_rescore.py cells
uv run --no-sync python ml-experiments/00455-*/e00455_rescore.py pool
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00404_report as report  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00435_tailcore as tailcore  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_404: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "predictions"
)
DATA_445: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors"
    / "work"
    / "predictions"
)
NRA_SHARES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00455-nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms"
)
REFERENCE: Final[str] = "B-mix"
ARM_DIRS: Final[dict[str, tuple[Path, str]]] = {
    "B-mix": (DATA_404, "B-mix"),
    "A0-inj": (DATA_404, "A0-inj"),
    "S1": (DATA_404, "S1"),
    "B3": (DATA_404, "B3"),
    "B6": (DATA_404, "B6"),
    "lagless": (DATA_445, "O-lagless"),
    "O-hard": (DATA_445, "O-hard"),
    "O-soft": (DATA_445, "O-soft"),
    "O-floor": (DATA_445, "O-floor"),
}
"""Arm label, then the folder and subfolder that hold its forecasts."""
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
CELLS: Final[tuple[tuple[str, int], ...]] = tuple(
    (m, s) for m in MONTHS for s in ((0, 1, 2) if m == "2026-06" else (0,))
)
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight")
DEPTHS: Final[tuple[str, ...]] = ("metered", "0.75", "1.0", "1.25")
CONTRAST_COUNT: Final[int] = (len(ARM_DIRS) - 1) * 3 * len(GROUPS)
"""Pre-registered contrasts: 8 arms against B-mix, at 3 NRA depth factors, in 2 groups."""
RESAMPLES: Final[int] = 20000
MDE_MULTIPLIER: Final[float] = 2.8
"""Normal-approximation multiplier for 80% power at a two-sided 5% test (experiment 404's)."""


def load_arm(*, arm: str, month: str, seed: int) -> pl.DataFrame:
    """Return the arm's real-world forecasts for a cell, scored, with its group."""
    folder, name = ARM_DIRS[arm]
    frame = pl.read_parquet(folder / name / f"{month}_s{seed}_w-1.parquet")
    frame = frame.filter(tailcore.lead_band_expr().is_not_null())  # leads 0 to 240 h, as in 404
    return frame.with_columns(crps=score.pinball_points(frame=frame))


def cell_scores(*, month: str, seed: int, real: np.ndarray, shares: dict[str, np.ndarray]) -> list:
    """Score every arm on the cell's real-event rows at every depth; return one dict per arm."""
    ref = load_arm(arm=REFERENCE, month=month, seed=seed)
    groups = report.group_of(sites=report.fit.prep.primaries()["site"].to_list())
    ref = ref.with_columns(world=pl.lit(-1), group=pl.Series(groups[ref["primary"].to_numpy()]))
    mask_inside = (ref["tgt_real"] | ref["tgt_post"]).to_numpy()
    clusters = report.cluster_ids(frame=ref, real=real)
    out = []
    for group in GROUPS:
        mask = mask_inside & (ref["group"] == group).to_numpy()
        if not mask.any():
            continue
        primary = ref["primary"].to_numpy()[mask].astype(np.int64)
        grid = np.clip(ref["target"].to_numpy()[mask].astype(np.int64), 0, 121247)
        keys = ref.filter(pl.Series(mask)).select("primary", "origin", "target")
        for arm in ARM_DIRS:
            if not (ARM_DIRS[arm][0] / ARM_DIRS[arm][1] / f"{month}_s{seed}_w-1.parquet").exists():
                continue  # B6 has fit seed 0 only
            frame = load_arm(arm=arm, month=month, seed=seed).filter(pl.Series(mask))
            if not frame.select("primary", "origin", "target").equals(keys):
                msg = f"{arm} rows differ from {REFERENCE} in {month} seed {seed}"
                raise ValueError(msg)
            for depth in DEPTHS:
                actual = frame["actual"].to_numpy().astype(np.float64)
                if depth != "metered":
                    actual = actual - shares[depth][primary, grid]
                rescored = frame.with_columns(actual=pl.Series(actual).cast(pl.Float32))
                crps = rescored.select(crps=score.pinball_points(frame=rescored))["crps"]
                out.append(
                    {
                        "arm": arm, "group": group, "month": month, "seed": seed, "depth": depth,
                        "rows": int(mask.sum()), "crps": crps.to_numpy(),
                        "cluster": clusters[mask],
                    }
                )  # fmt: skip
    return out


def cells() -> None:
    """Write one row per arm, group, depth, and cell, with the contrast to B-mix."""
    real = score.real_event_mask()
    npz = np.load(NRA_SHARES)
    shares = {"0.75": npz["f0.75"], "1.0": npz["f1.0"], "1.25": npz["f1.25"]}
    rows = []
    for month, seed in CELLS:
        scored = cell_scores(month=month, seed=seed, real=real, shares=shares)
        refs = {(r["group"], r["depth"]): r for r in scored if r["arm"] == REFERENCE}
        for r in scored:
            ref = refs[(r["group"], r["depth"])]
            diff = r["crps"] - ref["crps"]
            codes, inverse = np.unique(r["cluster"], return_inverse=True)
            totals = np.abs(np.bincount(inverse, weights=diff, minlength=len(codes)))
            keep = inverse != int(np.argmax(totals))
            rows.append(
                {
                    "arm": r["arm"], "group": r["group"], "month": r["month"], "seed": r["seed"],
                    "depth": r["depth"], "rows": r["rows"], "events": len(codes),
                    "score": float(r["crps"].mean()), "ref_score": float(ref["crps"].mean()),
                    "diff": float(diff.mean()),
                    "diff_drop_largest": float(diff[keep].mean()) if len(codes) > 1 else np.nan,
                }
            )  # fmt: skip
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
    cell = pl.read_parquet(OUT_DIR / "cells.parquet").filter(pl.col("arm") != REFERENCE)
    months = cell.group_by("arm", "group", "depth", "month").agg(
        pl.col("diff").mean(),
        pl.col("diff_drop_largest").mean(),
        pl.col("score").mean(),
        pl.col("ref_score").mean(),
        pl.col("events").first(),
    )
    bonferroni_level = 1.0 - 0.05 / CONTRAST_COUNT
    rows = []
    for key, part in months.group_by("arm", "group", "depth", maintain_order=True):
        d = part["diff"].to_numpy()
        dd = part["diff_drop_largest"].to_numpy()
        w = part["events"].to_numpy().astype(float)
        sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
        lo95, hi95 = month_bootstrap(values=d, level=0.95)
        lob, hib = month_bootstrap(values=d, level=bonferroni_level)
        lod, hid = month_bootstrap(values=dd[~np.isnan(dd)], level=0.95)
        lodb, hidb = month_bootstrap(values=dd[~np.isnan(dd)], level=bonferroni_level)
        rows.append(
            {
                "arm": key[0], "group": key[1], "depth": key[2], "months": len(d),
                "events_total": int(w.sum()), "arm_score": float(part["score"].mean()),  # ty: ignore[invalid-argument-type]
                "bmix_score": float(part["ref_score"].mean()),  # ty: ignore[invalid-argument-type]
                "mean_of_months": float(d.mean()),
                "event_weighted": float((d * w).sum() / w.sum()),
                "low95": lo95, "high95": hi95, "low_bonf": lob, "high_bonf": hib,
                "drop_largest_mean": float(np.nanmean(dd)),
                "drop_low95": lod, "drop_high95": hid,
                "drop_low_bonf": lodb, "drop_high_bonf": hidb,
                "mde_80pct": MDE_MULTIPLIER * sd / np.sqrt(len(d)) if len(d) > 1 else np.nan,
            }
        )  # fmt: skip
    table = pl.DataFrame(rows).sort("group", "depth", "arm")
    table.write_csv(OUT_DIR / "pooled.csv", float_precision=3)
    print(f"Bonferroni contrast count: {CONTRAST_COUNT}, level {bonferroni_level:.5f}")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    print(table)


def main() -> None:
    """Run one step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells", "pool"))
    args = parser.parse_args()
    if args.command == "cells":
        cells()
    else:
        pool()


if __name__ == "__main__":
    main()
