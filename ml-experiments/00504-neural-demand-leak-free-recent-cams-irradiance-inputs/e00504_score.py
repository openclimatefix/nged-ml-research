# ruff: noqa: E501
r"""Score experiment 504's CAMS arms against K3, the three-seed pool of the stored arm A.

Uses experiment 470's scorer, tail script, and curtailment script on its rows, pointed at this
experiment's folder (the pattern of experiments 489 and 494). K3 is the quantile average of seeds
0 to 2 of experiment 425's arm A; each CAMS arm is pooled the same way from its three seeds.

| Comparison | Arms | Counted |
|---|---|---|
| `main` | `A-cams`, `A-cams-real`, `A-cams-lag-only` | all three get Bonferroni intervals (count 12); only `A-cams` is gated |
| `checks` | `A-cams-mean`, `A-cams-stale1` (prediction only, no refit) | descriptive |

Commands: `build`, then per comparison `rows`, `baseline` (minimum detectable effect), `stats`,
`tails`, `curtailment`, each with `--comparison`.

```bash
export PYTHONPATH=$(ls -d $PWD/ml-experiments/*/ | tr '\\n' ':')
uv run --no-sync python ml-experiments/00504-*/e00504_score.py build
uv run --no-sync python ml-experiments/00504-*/e00504_score.py rows --comparison main
```
"""

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_prep as prep397  # ty: ignore[unresolved-import]
import e00454_build as build454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as curtailment470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00470_tails as tails470  # ty: ignore[unresolved-import]
import e00504_cams as cams_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00504-neural-demand-leak-free-recent-cams-irradiance-inputs" / "work"
)
E425_SEED_PREDS: Final[Path] = heads470.E425 / "seed_preds"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
TESTS: Final[int] = 12
"""Three arms times two groups times CRPS and q99 pinball loss (count 12, as experiment 494)."""
RESAMPLES: Final[int] = 100_000
"""Raised from the scorer's 20,000: at count 12 each tail of a 20,000-draw interval holds about 40."""
FIVE: Final[tuple[str, ...]] = build470.MONTHS
COMPARISONS: Final[dict[str, tuple[str, ...]]] = {
    "main": ("A-cams", "A-cams-real", "A-cams-lag-only"),
    "checks": ("A-cams-mean", "A-cams-stale1"),
}
CONTROLS: Final[dict[str, str]] = {"main": "K3", "checks": "A-cams"}
"""The arm each comparison's candidates are contrasted with (the checks are read against A-cams)."""
JULY: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time on or after this is scored (the ml-experiment skill)."""
D12: Final[str] = "D12"
ARM_FOLDERS: Final[dict[str, Path]] = {
    "K3": E425_SEED_PREDS / "A",
    **{a: WORK_DIR / "seed_preds" / a for a in (*COMPARISONS["main"], *COMPARISONS["checks"])},
}


def configure(*, comparison: str) -> None:
    """Point experiment 470's scorer, tails, and curtailment scripts at one comparison."""
    arms = COMPARISONS[comparison]
    heads470.WORK_DIR = WORK_DIR
    score470.MONTHS = FIVE
    curtailment470.MONTHS = FIVE
    build454.MONTHS = FIVE
    score470.SCORE_DIR = WORK_DIR / "scores" / comparison
    control = CONTROLS[comparison]
    score470.CANDIDATES = arms
    score470.CONTROLS = (control,)
    score470.DESCRIPTIVE = ()
    score470.REFERENCES = ()
    score470.MAIN_ARMS = (*arms, control)
    score470.CELLS = {}
    score470.CONTRASTS = tuple((a, control) for a in arms)
    score470.GATING_CRPS = ("eight", "six_clean")
    score470.GATING_Q99 = ("eight", "six_clean")
    score470.SLICES = {
        **score470.SLICES,
        "d12": ("mva_eight", "d12"),
        "eight_no_d12": ("mva_eight", "no_d12"),
    }
    d12 = int(prep397.primaries().filter(pl.col("site") == D12)["primary"][0])
    original_slice = score470.slice_filter
    original_context = score470.month_context

    def slice_filter(
        *, table: pl.DataFrame, name: str, basis: str = "metered"
    ) -> tuple[pl.DataFrame, str]:
        """Return a slice's rows; adds D12 alone and the eight without D12."""
        kind = score470.SLICES[name][1]
        if kind in ("d12", "no_d12"):
            part = table.filter(pl.col("group") == "mva_eight")
            keep = pl.col("primary") == d12 if kind == "d12" else pl.col("primary") != d12
            return part.filter(keep), ""
        return original_slice(table=table, name=name, basis=basis)

    def month_context(*, month: str, sites: pl.DataFrame, ctx: dict) -> pl.DataFrame:
        """Return experiment 470's month rows without valid times on or after 2026-07-01."""
        frame = original_context(month=month, sites=sites, ctx=ctx).filter(
            pl.col("valid_time") < JULY
        )
        assert frame["valid_time"].max() < JULY  # ty: ignore[unsupported-operator]
        return frame

    score470.slice_filter = slice_filter  # ty: ignore[invalid-assignment]
    score470.month_context = month_context  # ty: ignore[invalid-assignment]
    score470.TESTS = TESTS
    score470.RESAMPLES = RESAMPLES

    def masks(*, wide: pl.DataFrame) -> dict[str, pl.Series]:
        """Return the row-set masks: every arm present; the same set without the report-only group."""
        have = pl.all_horizontal([pl.col(f"{a}__q50").is_finite() for a in score470.MAIN_ARMS])
        frame = wide.select(
            main16=have,
            gam14=have & (pl.col("group") != "report_only"),
            ref14=have & (pl.col("group") != "report_only"),
        )
        return {c: frame[c] for c in frame.columns}

    score470.masks = masks  # ty: ignore[invalid-assignment]
    tails470.ARMS = (*arms, control)
    curtailment470.ARMS = (*arms, control)
    curtailment470.CANDIDATES = (*arms, control)
    curtailment470.REFERENCES = ("manual_heuristic",)
    curtailment470.COMPARISONS = (
        *((a, control) for a in arms),
        *((a, "manual_heuristic") for a in arms),
        (control, "manual_heuristic"),
    )


def cams_for(*, row: dict) -> cams_mod.SiteCams:
    """Return a Primary's CAMS arrays (for the report script)."""
    return cams_mod.site_cams(
        site=row["site"], latitude=float(row["latitude"]), longitude=float(row["longitude"])
    )


def link_stored() -> None:
    """Symlink the stored arms of experiment 470 that the scorer reads (singles and references)."""
    source = (
        ML_EXPERIMENTS_DATA_DIR
        / "00470-first-release-demand-forecaster-assembly-and-comparison"
        / "work"
        / "predictions"
    )
    target = WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in ("manual_heuristic", "A-pool", *[f"A-s{s}" for s in build470.POOL_SEEDS]):
        link = target / arm
        if not link.exists():
            link.symlink_to(source / arm, target_is_directory=True)


def build() -> None:
    """Write each arm's pooled seeds 0 to 2 on experiment 454's rows (one cell, raw quantiles)."""
    sites = build470.site_map()
    for month in FIVE:
        _, keys = build454.load_heads(month=month)
        rows = pl.DataFrame({k: keys[k] for k in build470.KEYS})
        for arm, folder in ARM_FOLDERS.items():
            if not all((folder / f"{month}_s{s}.parquet").exists() for s in SEEDS):
                print("skipped", arm, month, "(no networks)", flush=True)
                continue
            frame = build470.pool_stored(folder=folder, month=month, sites=sites, seeds=SEEDS)
            joined = build470.on_rows(rows=rows, frame=frame)
            assert joined["q50"].null_count() == 0, f"{arm} {month} misses rows"
            build470.write_arm(arm=arm, month=month, cell=0, frame=joined)
            print("wrote", arm, month, joined.height, flush=True)


def mde_three_seeds(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 428's minimum detectable effect for a pool of 3 seeds (k = 3).

    The seed spread is the spread of the single-seed scores `A-s0` to `A-s2` around their fold
    mean; the effect is `1.96 * sqrt(2) * sd / sqrt(3)`, the scorer's formula at k = 3.
    """
    records = []
    for label, group in {"eight": "mva_eight", "six_clean": "signed_six"}.items():
        part = table.filter((pl.col("group") == group) & pl.col("main16"))
        if group == "signed_six":
            part = part.filter(pl.col("cls") == "clean")
        values = np.array(
            [
                [part.filter(pl.col("month") == m)[f"A-s{s}__crps"].mean() for s in SEEDS]
                for m in score470.MONTHS
            ]
        )
        resid = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((resid**2).sum() / (len(values) * (len(SEEDS) - 1)))
        records.append(
            {"group": label, "folds": len(values), "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(len(SEEDS))}
        )  # fmt: skip
    return pl.DataFrame(records)


def baseline() -> None:
    """Print the three-seed minimum detectable effect before any contrast."""
    if not all((score470.SCORE_DIR / f"rows_{m}.parquet").exists() for m in score470.MONTHS):
        score470.rows_command()
    minimum = mde_three_seeds(table=score470.load_rows())
    score470.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    minimum.write_csv(score470.SCORE_DIR / "mde_k3.csv")
    print(minimum)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build", "rows", "baseline", "stats", "tails", "curtailment")
    )
    parser.add_argument("--comparison", choices=list(COMPARISONS), default="main")
    args = parser.parse_args()
    configure(comparison=args.comparison)
    link_stored()
    {
        "build": build,
        "rows": score470.rows_command,
        "baseline": baseline,
        "stats": score470.stats_command,
        "tails": tails470.main,
        "curtailment": curtailment470.main,
    }[args.command]()


if __name__ == "__main__":
    main()
