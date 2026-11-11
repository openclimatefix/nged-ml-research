# ruff: noqa: E501, D103
"""Experiment 693, stage 0: score the no-retrain ceilings and print the gate (private CSV and report).

Reads `<root>/P7_D50_<fold>_stage0/work/stage0/<fold>_r<pass>.parquet`, written by
`e00693_cand.py stage0`. Each row is one scored forecast on an injected pass of bank L, in phase P1,
P2, or P3, with the seed-mean and per-seed scores of A0-clean (observed lags), A0 (injected lags),
and A0-mask (injected lags with every flagged lag masked). Scores are in points of capacity; the
truth is the unshifted observed power.

The gate: seed-mean CRPS(A0) minus CRPS(A0-clean) on phases P1 plus P2, all folds pooled. Under 0.3
points, the no-retrain-arm retraining is not run. Intervals resample clusters of (fold, pass,
Primary), stratified by fold, 2,000 resamples.

Usage: python e00693_stage0_report.py [FOLD ...]
"""

import sys
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final = ML_EXPERIMENTS_DATA_DIR / "00693-oracle-transfer-flag-in-the-candidate-part-1"
FOLDS: Final = ("2025-11", "2026-02", "2026-05")
VARIANTS: Final = ("clean", "inj", "mask")
GATE: Final[float] = 0.3
N_BOOT: Final[int] = 2000


def load(*, folds: tuple[str, ...]) -> pl.DataFrame:
    frames = []
    for fold in folds:
        files = sorted((ROOT / f"P7_D50_{fold}_stage0" / "work" / "stage0").glob("*.parquet"))
        assert files, fold
        frames.extend(
            pl.read_parquet(f).with_columns(
                fold=pl.lit(fold), pass_=pl.lit(int(f.stem.split("_r")[1]))
            )
            for f in files
        )
    return pl.concat(frames)


def cluster_table(*, rows: pl.DataFrame, diff: pl.Expr) -> pl.DataFrame:
    """Return per (fold, pass, Primary) cluster the sum of `diff` and the row count."""
    return rows.group_by("fold", "pass_", "prim").agg(s=diff.sum(), n=pl.len())


def boot(*, clusters: pl.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (mean, low, high): pooled mean per row and its 95% cluster-bootstrap interval, stratified by fold."""
    by_fold = [g.select("s", "n").to_numpy() for _, g in clusters.group_by("fold")]
    point = sum(g[:, 0].sum() for g in by_fold) / sum(g[:, 1].sum() for g in by_fold)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = n = 0.0
        for g in by_fold:
            pick = g[rng.integers(0, len(g), len(g))]
            s += pick[:, 0].sum()
            n += pick[:, 1].sum()
        draws[b] = s / n
    return float(point), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def follow(*, rows: pl.DataFrame, a: str, b: str) -> float:
    """Return the share of the day-lag shift the median moves by: sum(change * shift) / sum(shift ** 2) over rows with a day-lag shift."""
    r = rows.filter(pl.col("shift_day") != 0)
    change = (r[f"{a}_med"] - r[f"{b}_med"]).to_numpy()
    shift = r["shift_day"].to_numpy()
    return float((change * shift).sum() / (shift**2).sum())


def main() -> None:
    folds = tuple(sys.argv[1:]) or FOLDS
    rows = load(folds=folds)
    rng = np.random.default_rng(693)
    lines = [
        f"folds {folds}; rows {rows.height:,}; passes per fold {rows.group_by('fold').agg(n=pl.col('pass_').n_unique()).to_dicts()}"
    ]
    sets = {"P1+P2": [1, 2], "P1": [1], "P2": [2], "P3": [3]}
    tab = []
    for name, phases in sets.items():
        r = rows.filter(pl.col("phase").is_in(phases))
        if r.is_empty():
            continue
        out = {"phase": name, "rows": r.height}
        for metric in ("crps", "ae"):
            for v in VARIANTS:
                out[f"{v}_{metric}"] = float(r[f"{v}_{metric}"].mean())
        for label, (a, b) in {
            "A0 - A0-clean": ("inj", "clean"),
            "A0 - A0-mask": ("inj", "mask"),
            "A0-mask - A0-clean": ("mask", "clean"),
        }.items():
            mean, lo, hi = boot(
                clusters=cluster_table(rows=r, diff=pl.col(f"{a}_crps") - pl.col(f"{b}_crps")),
                rng=rng,
            )
            out[f"crps {label}"] = f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        for seed in (0, 1, 2):
            if f"inj_crps_s{seed}" in r.columns:
                out[f"seed{seed} A0 - A0-clean"] = float(
                    (r[f"inj_crps_s{seed}"] - r[f"clean_crps_s{seed}"]).mean()
                )
        tab.append(out)
    table = pl.DataFrame(tab, infer_schema_length=None)
    for fold in folds:
        r = rows.filter((pl.col("fold") == fold) & pl.col("phase").is_in([1, 2]))
        lines.append(
            f"fold {fold} P1+P2 rows {r.height:,}: A0 - A0-clean {float((r['inj_crps'] - r['clean_crps']).mean()):+.3f}"
        )
    lines.append(
        f"copied share of the day-lag shift, A0 {follow(rows=rows, a='inj', b='clean'):.4f}, A0-mask {follow(rows=rows, a='mask', b='clean'):.4f}"
    )
    gate = table.filter(pl.col("phase") == "P1+P2")
    point = float(gate["crps A0 - A0-clean"][0].split()[0])
    lines.append(
        f"STAGE 0 GATE: CRPS(A0) - CRPS(A0-clean) on P1+P2 = {point:+.3f} points; gate {GATE}; the no-retrain-arm retraining {'RUNS' if point >= GATE else 'IS NOT RUN'}"
    )
    out = ROOT / "scores"
    out.mkdir(exist_ok=True)
    table.write_csv(out / "stage0_summary.csv")
    (out / "stage0_report.md").write_text(
        "\n".join(lines) + "\n\n" + table.to_pandas().to_string() + "\n"
    )
    print("\n".join(lines))
    with pl.Config(tbl_cols=-1, tbl_rows=-1, fmt_str_lengths=60):
        print(table)


if __name__ == "__main__":
    main()
