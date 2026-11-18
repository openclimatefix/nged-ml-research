r"""The two gates of experiment 707, run before any long-window fit.

**Gate 1, `predict`: experiment 425's stored networks reproduce its stored forecasts.** The check
loads each stored fitted state of arms `A` and `R` (`work/models/<arm>/<month>/s<seed>/`),
forecasts the month with this experiment's copied code, and compares the quantiles with experiment
425's stored per-seed forecast (`work/seed_preds/<arm>/<month>_s<seed>.parquet`). It passes if the
largest absolute difference is at most 1e-4 of capacity and both files hold the same keys. It also
loads each stored pre-trained state (`work/pretrained/A/<month>_s<seed>.pt`) into the network and
prints how far fine-tuning moved the weights, which shows that the two stored states belong to the
same chain.

**Gate 2, `refit`: this experiment's refits of `A_24` and `R_24` agree with the stored arms.** The
refits are fitted by `e00707_fit.py --gate`. For each arm and fold, the check scores the seed
average of the refit and the seed average of the stored arm, and compares the two scores with the
standard error of the seed-to-seed spread. The score is the equal-weight mean over the 16
Primaries of twice the mass-weighted pinball loss of the 11 raw quantiles (the CRPS approximation of
`common.pinball_quantile_score`), in points of capacity, over leads up to 240 hours. The difference
of the two seed-average scores must lie within two standard errors, where the standard error is
`sqrt(sd_refit**2 / n_refit + sd_stored**2 / n_stored)` and each `sd` is the sample standard
deviation of the per-seed scores. Experiment 425 set only `torch.manual_seed`, so GPU kernels are
not deterministic, and a fixed tolerance of 0.005 points would be tighter than the seed noise (the
stored seed-mean difference has a standard deviation of 0.052 points per fold). A failure on fold
2026-06 prints `STOP`: the plan's stop rule applies.

The exit status is 1 if a check fails, so a script can chain the gate before a fit.

**A passing run leaves a marker.** A gate that passes on every gate fold (and is not a smoke run)
writes its name and the time to `arms.GATE_MARKER`; a gate that fails removes its own entry.
`e00707_fit.py` refuses every arm with an `S_ens` stage unless the marker holds both gates, or
`--no-gate-marker` is passed.

```bash
# environment as in e00707_fit.py
uv run --no-sync python ml-experiments/00707-*/e00707_gate.py predict --device cuda
uv run --no-sync python ml-experiments/00707-*/e00707_fit.py --gate --workers 1
uv run --no-sync python ml-experiments/00707-*/e00707_gate.py refit
```
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_net as net  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

STORED_ARMS: Final[tuple[str, ...]] = ("A", "R")
"""Experiment 425's arm folders that the gate reads."""
PREDICTION_TOLERANCE: Final[float] = 1e-4
"""Largest absolute quantile difference, as a fraction of capacity, that gate 1 accepts."""
STANDARD_ERRORS: Final[float] = 2.0
MAX_LEAD_HOURS: Final[float] = 240.0
KEYS: Final[list[str]] = list(common.KEYS)
STOP_MONTH: Final[str] = "2026-06"


@dataclass(frozen=True)
class RefitResult:
    """Gate 2's reading of one arm and fold."""

    arm: str
    month: str
    refit_score: float
    stored_score: float
    refit_sd: float
    stored_sd: float
    standard_error: float
    passed: bool


def stored_state(*, arm: str, month: str, seed: int) -> Path:
    """Return the folder of one of experiment 425's stored fitted states."""
    return arms.STORED_425_DIR / "models" / arm / month / f"s{seed}"


def stored_forecast(*, arm: str, month: str, seed: int) -> Path:
    """Return experiment 425's stored per-seed raw forecast."""
    return arms.STORED_425_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def compare_forecasts(*, new: pl.DataFrame, stored: pl.DataFrame) -> tuple[float, float, bool]:
    """Return the largest and mean absolute quantile difference, and whether the keys match."""
    new = new.sort(KEYS)
    stored = stored.sort(KEYS)
    same_keys = new.height == stored.height and new.select(KEYS).equals(stored.select(KEYS))
    if not same_keys:
        return float("nan"), float("nan"), False
    gap = np.abs(
        new.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
        - stored.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    )
    return float(gap.max()), float(gap.mean()), True


def predict_gate(*, months: list[str], seeds: list[int], device: str) -> bool:
    """Run gate 1 on the named folds and seeds; return whether every check passed."""
    ok = True
    for arm in STORED_ARMS:
        for month in months:
            for seed in seeds:
                folder = stored_state(arm=arm, month=month, seed=seed)
                fitted = net.load_fitted(folder=folder, device=device)
                scale = net.load_scale(folder=folder)
                new = net.forecast_month(fitted=fitted, scale=scale, month=month, device=device)
                stored = pl.read_parquet(stored_forecast(arm=arm, month=month, seed=seed))
                worst, mean, same_keys = compare_forecasts(new=new, stored=stored)
                passed = same_keys and worst <= PREDICTION_TOLERANCE
                ok &= passed
                print(
                    f"predict {arm} {month} seed {seed}: {new.height:,} rows, keys "
                    f"{'match' if same_keys else 'DIFFER'}, max |diff| {worst:.2e}, "
                    f"mean {mean:.2e}: "
                    f"{'ok' if passed else 'FAIL'}",
                    flush=True,
                )
                if arm == "A":
                    ok &= pretrained_check(month=month, seed=seed, fitted=fitted, device=device)
    return ok


def pretrained_check(*, month: str, seed: int, fitted: net.Fitted, device: str) -> bool:
    """Load the stored pre-trained state into the network and print its distance from the fit."""
    path = arms.STORED_425_DIR / "pretrained" / "A" / f"{month}_s{seed}.pt"
    state = torch.load(path, map_location=device)
    fine_tuned = fitted.model.state_dict()
    if state.keys() != fine_tuned.keys() or any(
        state[k].shape != fine_tuned[k].shape for k in state
    ):
        print(f"pretrained A {month} seed {seed}: the keys or shapes differ from the fit: FAIL")
        return False
    moved = max(float((state[k].float() - fine_tuned[k].float()).abs().max()) for k in state)
    print(
        f"pretrained A {month} seed {seed}: loads; fine-tuning moved a weight by up to {moved:.3f}"
    )
    return moved > 0.0


def seed_average(
    *, folder: Path, month: str, seeds: tuple[int, ...]
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Return the seed average in quantile space, keyed and sorted, and the per-seed frames.

    Raises:
        ValueError: If a seed's file is missing or its keys differ from the first seed's.
    """
    frames = []
    for seed in seeds:
        path = folder / f"{month}_s{seed}.parquet"
        if not path.exists():
            msg = f"missing {path}"
            raise ValueError(msg)
        frames.append(pl.read_parquet(path).sort(KEYS))
    for frame in frames[1:]:
        if not frame.select(KEYS).equals(frames[0].select(KEYS)):
            msg = f"{folder} {month}: a seed's keys differ from the first seed's"
            raise ValueError(msg)
    values = np.mean(
        [f.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64) for f in frames], axis=0
    )
    averaged = (
        frames[0]
        .select(KEYS)
        .hstack(pl.DataFrame(np.sort(values, axis=1), schema=list(common.LEVEL_COLUMNS)))
    )
    return averaged, frames


def score_points(*, forecast: pl.DataFrame) -> float:
    """Return the equal-weight Primary mean of the CRPS approximation, in points, up to 240 hours.

    The actual is the observed power over capacity at the valid time; a half-hour with no reading is
    skipped.
    """
    _series, arrays = common.series_and_arrays()
    site_of = dict(
        zip(
            prep.primaries()["time_series_id"].to_list(),
            prep.primaries()["site"].to_list(),
            strict=True,
        )
    )
    per_site = []
    for series_id, part in forecast.group_by("time_series_id"):
        site = site_of[int(series_id[0])]
        index = np.asarray(common.grid_index(times=part["valid_time"]))
        actual = common.take(values=arrays[site].power, index=index).astype(np.float64)
        lead = (
            part["valid_time"] - part["power_fcst_init_time"]
        ).dt.total_minutes().to_numpy() / 60.0
        keep = np.isfinite(actual) & (lead <= MAX_LEAD_HOURS)
        quantiles = part.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)[keep]
        score = common.pinball_quantile_score(quantiles=quantiles, actual=actual[keep])
        per_site.append(float(score.mean()) * 100.0)
    return float(np.mean(per_site))


def refit_reading(*, arm: str, month: str, seeds: tuple[int, ...], out_dir: Path) -> RefitResult:
    """Return gate 2's reading for the refit of `A_24` or `R_24` on one fold.

    Args:
        arm: `A_24` or `R_24`.
        month: The fold.
        seeds: The seeds both sets hold.
        out_dir: This experiment's `work` folder.
    """
    spec = arms.ARMS[arm]
    assert spec.stored_folder is not None
    refit_avg, refit_seeds = seed_average(
        folder=out_dir / "seed_preds" / spec.folder, month=month, seeds=seeds
    )
    stored_avg, stored_seeds = seed_average(
        folder=arms.STORED_425_DIR / "seed_preds" / spec.stored_folder, month=month, seeds=seeds
    )
    if not refit_avg.select(KEYS).equals(stored_avg.select(KEYS)):
        msg = f"{arm} {month}: the refit and the stored arm hold different keys"
        raise ValueError(msg)
    refit_each = np.array([score_points(forecast=f) for f in refit_seeds])
    stored_each = np.array([score_points(forecast=f) for f in stored_seeds])
    refit_score = score_points(forecast=refit_avg)
    stored_score = score_points(forecast=stored_avg)
    refit_sd, stored_sd = float(refit_each.std(ddof=1)), float(stored_each.std(ddof=1))
    se = float(np.sqrt(refit_sd**2 / len(seeds) + stored_sd**2 / len(seeds)))
    passed = abs(refit_score - stored_score) <= STANDARD_ERRORS * se
    print(
        f"refit {arm} {month}: seed-average score refit {refit_score:.4f}, "
        f"stored {stored_score:.4f}, "
        f"gap {refit_score - stored_score:+.4f}; per-seed sd refit {refit_sd:.4f}, stored "
        f"{stored_sd:.4f}; standard error {se:.4f}; limit {STANDARD_ERRORS:.0f} SE "
        f"{STANDARD_ERRORS * se:.4f}: {'ok' if passed else 'FAIL'}",
        flush=True,
    )
    return RefitResult(
        arm=arm,
        month=month,
        refit_score=refit_score,
        stored_score=stored_score,
        refit_sd=refit_sd,
        stored_sd=stored_sd,
        standard_error=se,
        passed=passed,
    )


def refit_gate(*, months: list[str], seeds: tuple[int, ...], out_dir: Path) -> bool:
    """Run gate 2 on the named folds; print `STOP` if a stop-month reading fails."""
    results = []
    for arm in ("A_24", "R_24"):
        for month in months:
            try:
                results.append(refit_reading(arm=arm, month=month, seeds=seeds, out_dir=out_dir))
            except ValueError as error:
                print(f"refit {arm} {month}: not checked ({error}): FAIL")
                return False
    for month in months:
        by_arm = {r.arm: r for r in results if r.month == month}
        if {"A_24", "R_24"} <= by_arm.keys():
            new = by_arm["A_24"].refit_score - by_arm["R_24"].refit_score
            old = by_arm["A_24"].stored_score - by_arm["R_24"].stored_score
            print(f"refit {month}: A minus R, refit {new:+.4f}, stored {old:+.4f} (information)")
    failed = [r for r in results if not r.passed]
    if any(r.month == STOP_MONTH for r in failed):
        print(f"STOP: a refit misses the stored arm on fold {STOP_MONTH}; report to the manager")
    return not failed


def record_gate(*, gate: str, passed: bool) -> None:
    """Record a gate's result in `arms.GATE_MARKER`: its time if it passed, else no entry."""
    marker: dict[str, str] = {}
    if arms.GATE_MARKER.exists():
        marker = json.loads(arms.GATE_MARKER.read_text())
    if passed:
        marker[gate] = datetime.now(UTC).isoformat(timespec="seconds")
    else:
        marker.pop(gate, None)
    arms.GATE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    temporary = arms.GATE_MARKER.with_name(arms.GATE_MARKER.name + ".tmp")
    temporary.write_text(json.dumps(marker, indent=1))
    temporary.replace(arms.GATE_MARKER)


def main() -> None:
    """Run gate 1, gate 2, or both."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("gate", choices=("predict", "refit", "all"))
    parser.add_argument("--months", nargs="+", default=list(arms.GATE_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="read the refits from work/smoke")
    args = parser.parse_args()
    ok = True
    on_gate_folds = set(arms.GATE_MONTHS) <= set(args.months) and not args.smoke
    if args.gate in ("predict", "all"):
        passed = predict_gate(months=args.months, seeds=args.seeds or [0], device=args.device)
        ok &= passed
        if on_gate_folds:
            record_gate(gate="predict", passed=passed)
    if args.gate in ("refit", "all"):
        out_dir = arms.WORK_DIR / "work" / ("smoke" if args.smoke else "")
        seeds = tuple(args.seeds) if args.seeds else arms.FIVE_SEEDS
        passed = refit_gate(months=args.months, seeds=seeds, out_dir=out_dir)
        ok &= passed
        if on_gate_folds:
            record_gate(gate="refit", passed=passed)
    print("GATE PASSED" if ok else "GATE FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
