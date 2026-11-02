# ruff: noqa: E501
r"""Pretrain, fine-tune, and forecast the arms of experiment 526 (placebo repeat of the trailing windows).

Adds files to experiment 515's harness and never edits it. Everything this script writes goes to this
experiment's own `work/` folder, because experiment 515's scorer lists arms by scanning its own
`work/seed_preds/*`.

Arms (`ARM_SPEC`):

- `W0`, `W-T`, `W-S`, `W-X`, `W-null`: as in experiment 515 (`e00515_fit.ARM_COLUMNS`).
- `W-Tnull`: the three `W-T` columns taken from the shuffled other day of experiment 515's placebo
  (the same donor draw as `W-null`, so it is `W-null` restricted to W-T's three columns).
- `W-noise3`: three frozen independent standard-normal columns (a column-count control). The columns
  are drawn once per fold and stage and are the same for every training seed.
- `W-seam`: W-T with persistence fill before the origin, fine-tuned from W-T's pretrained weights.

Forecasts are the 11 quantiles of experiment 515 plus `mean_forecast`, the mixture mean
`sum(exp(log_w) x mean)` as a fraction of capacity (experiment 510's `forecast_frame` has no mean).
Seeds 0 to 2 of the experiment 515 arms are read from experiment 515 (and experiment 513's `P0`) and
never refitted, except by the `refit-mean` command, which fine-tunes them again from the stored
pretrained weights to obtain the mean column.

Commands: `fit` (pretrain what is missing, fine-tune, forecast), `refit-mean`, `checks`.
`--stage 1` to `--stage 6` select the planned run order; without it `--arms`, `--months`, and
`--seeds` select the work. A forecast that exists is never rewritten, so more seeds can be added later.

```bash
. <env script>; PYTHONPATH=<folders of 515, 513, 510, 448, 437, 428, 422, 418, 404, 402, 397, 87 ...> \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    /usr/bin/time -v uv run --no-sync python ml-experiments/00526-*/e00526_fit.py fit --stage 1
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_features as feat  # ty: ignore[unresolved-import]
import e00515_fit as f515  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00526-lagless-demand-network-trailing-weather-window-placebo-repeat"
    / "work"
)
E513_ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00513-lagless-demand-network-whole-day-price-shape-at-training-time"
    / "work"
)
E515_ARMS: Final[tuple[str, ...]] = ("W0", "W-T", "W-S", "W-X", "W-null")
"""Arms whose seeds 0 to 2 experiment 515 stored."""
E515_SEEDS: Final[tuple[int, ...]] = f515.SEEDS
SEAM_ARM: Final[str] = f515.SEAM_ARM
SEAM_PARENT: Final[str] = f515.SEAM_PARENT
SPEC_NULL_T: Final[str] = "W-Tnull"
NOISE_ARM: Final[str] = "W-noise3"
NOISE_WIDTH: Final[int] = 3
NOISE_SEED: Final[int] = 526
STAGE_CODE: Final[dict[str, int]] = {"pre": 0, "ft": 1, "test": 2}
ARMS: Final[tuple[str, ...]] = (*E515_ARMS, SPEC_NULL_T, NOISE_ARM)
ALL_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ARMS_ALL_BUT_SEAM: Final[tuple[str, ...]] = (
    "W0",
    "W-T",
    "W-S",
    "W-X",
    "W-null",
    "W-Tnull",
    "W-noise3",
)
WINTER: Final[tuple[str, ...]] = f515.MONTHS
SUMMER: Final[tuple[str, ...]] = ("2025-05", "2025-07", "2025-09", "2026-04")
"""Summer folds: the refit cutoff is the first day of the month; July 2026 stays unused."""
MEAN_REFIT_ARMS: Final[tuple[str, ...]] = ("W0", "W-T", "W-null")
BASE_ARM: Final[dict[str, str]] = {SPEC_NULL_T: "W-null", NOISE_ARM: "W0"}
"""The experiment 515 arm whose rows and placebo draw `Fold` must build for each new arm."""
EXTENSION_SEEDS: Final[tuple[int, ...]] = (5, 6, 7, 8, 9, 10)
"""Seeds 5 to 10 take W0, W-T, and W-Tnull on the winter folds to 11 seeds (approved in this PR)."""
STAGES: Final[dict[int, list[dict]]] = {
    1: [{"arms": ["W0", "W-T", "W-null"], "months": WINTER, "seeds": (3, 4)}],
    2: [{"arms": [SPEC_NULL_T, NOISE_ARM], "months": WINTER, "seeds": ALL_SEEDS}],
    3: [{"arms": ["W0", "W-T", SPEC_NULL_T], "months": WINTER, "seeds": EXTENSION_SEEDS}],
    4: [{"arms": ["W-S", "W-X"], "months": WINTER, "seeds": (3, 4)}],
    5: [{"arms": [*ARMS_ALL_BUT_SEAM], "months": SUMMER, "seeds": ALL_SEEDS}],
    6: [{"arms": [SEAM_ARM], "months": WINTER, "seeds": (3, 4)}],
}  # fmt: skip
"""Run order: core 5 seeds (1, 2), the 11-seed extension (3), W-S and W-X (4), summer folds (5), W-seam (6).

The mean refit is stage 7 (`refit-mean`)."""


def own_pretrained(*, arm: str, month: str, seed: int) -> Path:
    """Return where this experiment stores an arm's pretrained weights."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def pretrained_source(*, arm: str, month: str, seed: int) -> Path | None:
    """Return the existing pretrained weights of an arm, fold, and seed, or None.

    Order: this experiment's own folder, experiment 513's `P0` (W0 on the three reused folds), then
    experiment 515's folder (seeds 0 to 2 of its arms).
    """
    own = own_pretrained(arm=arm, month=month, seed=seed)
    if own.exists():
        return own
    if arm in E515_ARMS and seed in E515_SEEDS:
        if arm == "W0" and month in f515.REUSED_BASE:
            old = E513_ROOT / "pretrained" / "P0" / f"{month}_s{seed}.pt"
        else:
            old = f515.pretrained_path(arm=arm, month=month, seed=seed)
        if old.exists():
            return old
    return None


def stored_forecast(*, arm: str, month: str, seed: int) -> Path:
    """Return the forecast file for an arm, fold, and seed, in experiment 515 or in this folder."""
    if arm in (*E515_ARMS, SEAM_ARM) and seed in E515_SEEDS:
        if arm == "W0" and month in f515.REUSED_BASE:
            return E513_ROOT / "seed_preds" / "P0" / f"{month}_s{seed}.parquet"
        return f515.prediction_path(arm=arm, month=month, seed=seed)
    return own_forecast(arm=arm, month=month, seed=seed)


def own_forecast(*, arm: str, month: str, seed: int) -> Path:
    """Return where this experiment writes an arm's forecast (quantiles and the mixture mean)."""
    return W / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def refit_forecast(*, arm: str, month: str, seed: int) -> Path:
    """Return where the mean refit of an experiment 515 arm is written."""
    return W / "mean_refit" / arm / f"{month}_s{seed}.parquet"


class Fold(f515.Fold):
    """Experiment 515's fold with the `W-Tnull` and `W-noise3` arms added."""

    def __init__(
        self, *, month: str, clean: np.ndarray, need_pre: bool, arms: list[str], seam: bool
    ) -> None:
        """Build the rows and the raw window columns, mapping the new arms to 515's builders."""
        self._noise: dict[tuple[str, int], np.ndarray] = {}
        mapped = sorted({BASE_ARM.get(a, a) for a in arms})
        super().__init__(month=month, clean=clean, need_pre=need_pre, arms=mapped, seam=seam)

    def noise(self, *, stage: str, n_rows: int) -> np.ndarray:
        """Return the frozen standard-normal columns of a stage (the same for every training seed)."""
        key = (stage, NOISE_WIDTH)
        if key not in self._noise:
            month_code = int(self.month.replace("-", ""))
            rng = np.random.default_rng([NOISE_SEED, month_code, STAGE_CODE[stage], NOISE_WIDTH])
            self._noise[key] = rng.standard_normal((n_rows, NOISE_WIDTH)).astype(np.float32)
        out = self._noise[key]
        if len(out) != n_rows:
            msg = f"noise for {stage} has {len(out)} rows, asked for {n_rows}"
            raise ValueError(msg)
        return out

    def matrix(self, *, arm: str, stage: str, x: np.ndarray, seam: bool = False) -> np.ndarray:
        """Return the stage's inputs with the arm's standardised window columns appended."""
        if arm == NOISE_ARM:
            return np.concatenate([x, self.noise(stage=stage, n_rows=len(x))], axis=1)
        if arm == SPEC_NULL_T:
            raw = self.raw[(stage, "null")]
            index = [feat.ALL_COLUMNS.index(name) for name in feat.ARM_COLUMNS["T"]]
            scaled = (raw[:, index] - self.mean[index]) / self.std[index]
            scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
            return np.concatenate([x, scaled], axis=1)
        return super().matrix(arm=arm, stage=stage, x=x, seam=seam)


def mixture_mean(
    *, model: torch.nn.Module, test: data.TestRows, rows: data.Rows, device: str
) -> np.ndarray:
    """Return the mixture mean of each test row as a fraction of capacity, `sum(exp(log_w) x mean)`."""
    parts = []
    mu = torch.from_numpy(rows.target_mean).to(device)
    sd = torch.from_numpy(rows.target_std).to(device)
    for a in range(0, len(test.x), 65_536):
        x = torch.from_numpy(test.x[a : a + 65_536]).to(device)
        p = torch.from_numpy(test.primary[a : a + 65_536]).to(device)
        with torch.no_grad():
            log_w, mean, _ = c.base.mixture_parts(out=model(x, p).double(), components=c.COMPONENTS)
            z = (log_w.exp() * mean).sum(-1)
        parts.append((z * sd[p] + mu[p]).float().cpu().numpy())
    return np.concatenate(parts)


def _tensor_set(*, x: np.ndarray, rows: data.Rows, y: np.ndarray, w: np.ndarray, primary: np.ndarray, is_val: np.ndarray, device: str) -> dict[str, torch.Tensor]:  # fmt: skip
    return f515._tensor_set(x=x, y=y, w=w, primary=primary, is_val=is_val, rows=rows, device=device)


def pretrain(*, fold: Fold, arm: str, seed: int, device: str) -> str:
    """Pretrain an arm from scratch on the fold's ERA5-and-CAMS rows and store the weights."""
    assert fold.pre is not None
    pre, rows = fold.pre, fold.rows
    x = fold.matrix(arm=arm, stage="pre", x=pre.x)
    t = _tensor_set(x=x, rows=rows, y=pre.y, w=np.ones(len(pre.y)), primary=pre.primary, is_val=pre.is_val, device=device)  # fmt: skip
    started = time.monotonic()
    torch.manual_seed(seed)
    model = c.new_model(outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)).to(
        device
    )
    epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device)  # fmt: skip
    r448.atomic_save(
        state=model.state_dict(), target=own_pretrained(arm=arm, month=fold.month, seed=seed)
    )
    del t, model, x
    torch.cuda.empty_cache()
    return f"  pretrain {arm} {fold.month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s"


def fine_tune_and_forecast(
    *, fold: Fold, arm: str, parent: str, seed: int, seam: bool, out: Path, device: str
) -> str:
    """Fine-tune from the parent arm's pretrained weights and write quantiles and the mixture mean."""
    started = time.monotonic()
    rows = fold.rows
    source = pretrained_source(arm=parent, month=fold.month, seed=seed)
    if source is None:
        msg = f"no pretrained weights for {parent} {fold.month} s{seed}"
        raise FileNotFoundError(msg)
    x = fold.matrix(arm=parent, stage="ft", x=rows.x, seam=seam)
    t = _tensor_set(x=x, rows=rows, y=rows.y, w=rows.weight, primary=rows.primary, is_val=rows.is_val, device=device)  # fmt: skip
    model = c.new_model(outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)).to(
        device
    )
    c.load_pretrained(model=model, path=source)
    model.to(device)
    epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device)  # fmt: skip
    test = fold.test
    scored = data.TestRows(
        x=fold.matrix(arm=parent, stage="test", x=test.x, seam=seam), primary=test.primary,
        origin_idx=test.origin_idx, target_idx=test.target_idx, lead=test.lead,
        origin=test.origin, valid=test.valid,
    )  # fmt: skip
    frame = c.forecast_frame(model=model, outlier=False, test=scored, select=np.ones(len(test.x), dtype=bool), rows=rows, widen=1.0, device=device)  # fmt: skip
    frame = frame.with_columns(
        mean_forecast=pl.Series(mixture_mean(model=model, test=scored, rows=rows, device=device), dtype=pl.Float32)
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    del t, model
    torch.cuda.empty_cache()
    return f"  fit {arm} {fold.month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(rows.y):,} rows"


def run(*, jobs: list[dict], device: str, refit_mean: bool) -> None:
    """Pretrain what is missing, then fine-tune and forecast each job's arms, folds, and seeds.

    Args:
        jobs: Dicts with `arms`, `months`, and `seeds`.
        device: A torch device.
        refit_mean: Fine-tune the experiment 515 forecasts again (from stored pretrained weights
            only) into `mean_refit/`, and print the largest quantile difference from the stored ones.
    """
    clean = c.clean_stack()
    for job in jobs:
        for month in job["months"]:
            todo = []  # (arm, seed, parent, seam, output)
            for arm in job["arms"]:
                for seed in job["seeds"]:
                    if refit_mean:
                        out = refit_forecast(arm=arm, month=month, seed=seed)
                    else:
                        out = stored_forecast(arm=arm, month=month, seed=seed)
                    if out.exists():
                        continue
                    parent = SEAM_PARENT if arm == SEAM_ARM else arm
                    todo.append((arm, seed, parent, arm == SEAM_ARM, out))
            if not todo:
                continue
            needs_pre = [
                (p, s)
                for _, s, p, _, _ in todo
                if pretrained_source(arm=p, month=month, seed=s) is None
            ]
            needs_pre = sorted(set(needs_pre))
            if refit_mean and needs_pre:
                msg = f"refit-mean needs stored pretrained weights, missing: {needs_pre[:3]}"
                raise FileNotFoundError(msg)
            started = time.monotonic()
            fold = Fold(month=month, clean=clean, need_pre=bool(needs_pre), arms=sorted({p for _, _, p, _, _ in todo}), seam=any(t[3] for t in todo))  # fmt: skip
            days = fold.windows.days["ens"]
            print(
                f"{month}: feeder {time.monotonic() - started:.0f} s; placebo days without donor per draw {days.unmatched}; widened {days.widened}; rows without any donor {fold.windows.unfilled}",
                flush=True,
            )
            for parent, seed in needs_pre:
                print(pretrain(fold=fold, arm=parent, seed=seed, device=device), flush=True)
            for arm, seed, parent, seam, out in todo:
                print(fine_tune_and_forecast(fold=fold, arm=arm, parent=parent, seed=seed, seam=seam, out=out, device=device), flush=True)  # fmt: skip
                if refit_mean:
                    old = pl.read_parquet(stored_forecast(arm=arm, month=month, seed=seed))
                    new = pl.read_parquet(out)
                    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
                    both = old.join(new, on=keys, how="inner", suffix="_new")
                    diff = max(
                        float((both[q] - both[f"{q}_new"]).abs().max())
                        for q in ("q01", "q50", "q99")
                    )  # ty: ignore[invalid-argument-type]
                    print(f"    refit vs stored {arm} {month} s{seed}: {both.height}/{old.height} rows matched, max |difference| in q01, q50, q99 = {diff:.6f}", flush=True)  # fmt: skip
            del fold


def checks(*, month: str) -> None:
    """Print the sanity checks of the new arms on one fold: widths, scale, NaN, and the placebo match."""
    clean = c.clean_stack()
    fold = Fold(
        month=month,
        clean=clean,
        need_pre=True,
        arms=["W-T", "W-null", SPEC_NULL_T, NOISE_ARM],
        seam=False,
    )
    print(
        f"{month}: {len(fold.rows.y):,} fine-tuning rows, {len(fold.test.x):,} test rows, {len(fold.pre.x):,} pretraining rows"
    )  # ty: ignore[unresolved-attribute]
    for arm in ("W-T", SPEC_NULL_T, NOISE_ARM, "W-null"):
        for stage, base in (("pre", fold.pre.x), ("ft", fold.rows.x), ("test", fold.test.x)):  # ty: ignore[unresolved-attribute]
            extra = fold.matrix(arm=arm, stage=stage, x=base)[:, base.shape[1] :]
            print(f"  {arm} {stage}: {extra.shape[1]} columns, mean {extra.mean():+.3f}, sd {extra.std():.3f}, NaN {int(np.isnan(extra).sum())}")  # fmt: skip
    for stage, base in (("ft", fold.rows.x), ("test", fold.test.x)):
        tnull = fold.matrix(arm=SPEC_NULL_T, stage=stage, x=base)[:, base.shape[1] :]
        null = fold.matrix(arm="W-null", stage=stage, x=base)[:, base.shape[1] :]
        index = [feat.ALL_COLUMNS.index(n) for n in feat.ARM_COLUMNS["T"]]
        print(f"  {stage}: W-Tnull equals the temperature columns of W-null: {bool(np.array_equal(tnull, null[:, index]))}")  # fmt: skip
        again = fold.matrix(arm=NOISE_ARM, stage=stage, x=base)[:, base.shape[1] :]
        print(f"  {stage}: noise columns repeat exactly on a second call: {bool(np.array_equal(again, fold.noise(stage=stage, n_rows=len(base))))}")  # fmt: skip


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "refit-mean", "checks"))
    parser.add_argument("--stage", type=int, choices=sorted(STAGES))
    parser.add_argument("--arms", nargs="+", choices=[*ARMS, SEAM_ARM])
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "checks":
        checks(month=(args.months or [WINTER[0]])[0])
        return
    if args.command == "refit-mean":
        jobs = [{"arms": list(MEAN_REFIT_ARMS), "months": list(args.months or WINTER), "seeds": list(args.seeds or E515_SEEDS)}]  # fmt: skip
        run(jobs=jobs, device=args.device, refit_mean=True)
        return
    if args.stage is not None:
        jobs = STAGES[args.stage]
    else:
        jobs = [{"arms": args.arms, "months": args.months, "seeds": args.seeds}]
    run(jobs=jobs, device=args.device, refit_mean=False)


if __name__ == "__main__":
    main()
