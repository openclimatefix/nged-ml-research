# ruff: noqa: E501
r"""Pretrain, fine-tune, and forecast the arms of experiment 515 (trailing weather windows).

Commands: `time` (one fold and seed, every stage timed), `pretrain`, `fit`, `both`, `checks`.

All arms are the lag-free network of experiment 448's `L0PT` with experiment 510's `L-M` recipe
(monthly refit, 182.6-day weight), pretrained in experiment 513's loop. The window columns are
appended to the 37 non-lag columns, so each arm is pretrained from scratch.

- `W0`: no added column (the control; experiment 513's `P0` is reused for three folds).
- `W-T`, `W-S`, `W-X`, `W-ALL`, `W-ALLnoDH`: the columns of `e00515_features.ARM_COLUMNS`.
- `W-null`: the nine `W-ALL` columns taken from a shuffled other day (`e00515_placebo`).
- `W-seam`: fine-tunes and scores W-T with persistence fill before the origin, starting from W-T's
  pretrained weights (ERA5 windows have no seam). It adds no pretraining.

```bash
. <env script>; OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00515-*/e00515_fit.py time
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_features as feat
import e00515_placebo as placebo
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = feat.WORK_DIR
ALL: Final[tuple[str, ...]] = feat.ALL_COLUMNS
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "W0": (),
    "W-T": feat.ARM_COLUMNS["T"],
    "W-S": feat.ARM_COLUMNS["S"],
    "W-X": feat.ARM_COLUMNS["X"],
    "W-null": ALL,
    "W-ALL": ALL,
    "W-ALLnoDH": tuple(c_ for c_ in ALL if c_ not in feat.ARM_COLUMNS["DH"]),
}
CORE_ARMS: Final[tuple[str, ...]] = ("W0", "W-T", "W-S", "W-X", "W-null")
SEAM_ARM: Final[str] = "W-seam"
SEAM_PARENT: Final[str] = "W-T"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MONTHS: Final[tuple[str, ...]] = (
    "2024-11", "2024-12", "2025-01", "2025-02", "2025-03",
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03",
)  # fmt: skip
REUSED_BASE: Final[tuple[str, ...]] = ("2025-12", "2026-01", "2026-03")
"""Folds whose W0 forecasts are experiment 513's `P0` (same network, loop, rows, and seeds)."""
KEY_SHIFT: Final[int] = 22


class EnsembleLookup:
    """All Primaries' ensemble-row window tables for one seam variant, searchable by (origin, valid)."""

    def __init__(self, *, seam: str) -> None:
        """Load the 16 tables, sorted by key, as float32 arrays."""
        self.keys: dict[int, np.ndarray] = {}
        self.values: dict[int, np.ndarray] = {}
        for row in prep.primaries().iter_rows(named=True):
            t = pl.read_parquet(feat.table_path(site=row["site"], seam=seam))
            origin = (t["origin_us"].to_numpy() - c.prep422.GRID_START_US) // c.prep422.STEP_US
            valid = (t["valid_us"].to_numpy() - c.prep422.GRID_START_US) // c.prep422.STEP_US
            key = origin.astype(np.int64) * (1 << KEY_SHIFT) + valid.astype(np.int64)
            order = np.argsort(key)
            self.keys[int(row["primary"])] = key[order]
            self.values[int(row["primary"])] = t.select(ALL).to_numpy()[order].astype(np.float32)

    def get(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the nine columns of each (primary, origin, valid) row and a mask of rows found."""
        out = np.full((len(primary), len(ALL)), np.nan, dtype=np.float32)
        found = np.zeros(len(primary), dtype=bool)
        want = origin_idx.astype(np.int64) * (1 << KEY_SHIFT) + valid_idx.astype(np.int64)
        for p in np.unique(primary):
            sel = np.nonzero(primary == p)[0]
            keys = self.keys[int(p)]
            pos = np.clip(np.searchsorted(keys, want[sel]), 0, len(keys) - 1)
            hit = keys[pos] == want[sel]
            out[sel[hit]] = self.values[int(p)][pos[hit]]
            found[sel[hit]] = True
        return out, found


class Windows:
    """The nine raw window columns of any row set, real or placebo, for one fold."""

    def __init__(self, *, fit_until_idx: int) -> None:
        """Load the ensemble tables and the ERA5 prefix sums, and draw the placebo days."""
        self.ens = EnsembleLookup(seam="previous")
        self.dense = {
            int(r["primary"]): feat.DenseWindows(row=r)
            for r in prep.primaries().iter_rows(named=True)
        }
        first_run = int(common.grid_index(times=prep.FIRST_RUN)) + 5 * placebo.STEPS_PER_DAY
        holdout = int(common.grid_index(times=common.HOLDOUT))
        self.days = {
            "era5": placebo.DayStructure(fit_until_idx=fit_until_idx),
            "ens": placebo.DayStructure(fit_until_idx=fit_until_idx, allowed=(first_run, holdout)),
        }
        self.unfilled = 0

    def real(
        self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray,
        ens: EnsembleLookup | None = None,
    ) -> np.ndarray:  # fmt: skip
        """Return raw columns: `kind` is `"era5"` (pretraining rows) or `"ens"` (fine-tuning and test rows)."""
        if kind == "era5":
            out = np.empty((len(primary), len(ALL)), dtype=np.float32)
            for p in np.unique(primary):
                sel = primary == p
                out[sel] = self.dense[int(p)].columns(
                    origin_idx=origin_idx[sel], valid_idx=valid_idx[sel]
                )
            return out
        values, found = (ens or self.ens).get(
            primary=primary, origin_idx=origin_idx, valid_idx=valid_idx
        )
        if not found.all():
            msg = f"{int((~found).sum())} ensemble rows have no window columns"
            raise ValueError(msg)
        return values

    def shuffled(
        self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:  # fmt: skip
        """Return the placebo columns: the donor day's, with fallback donors for missing runs (NaN if none)."""
        out = np.full((len(primary), len(ALL)), np.nan, dtype=np.float32)
        todo = np.ones(len(primary), dtype=bool)
        for attempt in range(len(placebo.SEEDS)):
            shift = self.days[kind].shifts(valid_idx=valid_idx, attempt=attempt)
            usable = todo & np.isfinite(shift)
            if not usable.any():
                continue
            s = np.where(usable, shift, 0).astype(np.int64)
            o2, v2 = origin_idx + s, valid_idx + s
            inside = usable & (o2 >= 0) & (v2 < len(self.days[kind].day_id))
            idx = np.nonzero(inside)[0]
            if kind == "era5":
                out[idx] = self.real(
                    kind="era5", primary=primary[idx], origin_idx=o2[idx], valid_idx=v2[idx]
                )
                todo[idx] = False
            else:
                values, found = self.ens.get(
                    primary=primary[idx], origin_idx=o2[idx], valid_idx=v2[idx]
                )
                out[idx[found]] = values[found]
                todo[idx[found]] = False
        self.unfilled += int(todo.sum())
        return out


def pretrained_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's pretrained weights live."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def prediction_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's forecasts live."""
    return W / "seed_preds" / arm / f"{month}_s{seed}.parquet"


class Fold:
    """One fold's rows, test rows, pretraining rows, and standardised window columns per arm."""

    def __init__(
        self, *, month: str, clean: np.ndarray, need_pre: bool, arms: list[str], seam: bool
    ) -> None:
        """Build the rows and the raw (real and placebo) window columns once."""
        self.month = month
        refit = c.month_start(month=month)
        self.rows = data.build_fold(refit=refit, clean=clean)
        self.test = data.build_test(month=month, standardiser=self.rows.standardiser)
        self.pre = (
            r448.pretrain_rows(rows=self.rows, refit=refit, clean=clean) if need_pre else None
        )
        fit_until = int((c.us(t=refit) - c.prep422.GRID_START_US) // c.prep422.STEP_US)
        windows = Windows(fit_until_idx=fit_until)
        self.windows = windows
        needs_real = any(a not in {"W0", "W-null"} for a in arms) or seam
        needs_null = "W-null" in arms
        self.raw: dict[tuple[str, str], np.ndarray] = {}  # (stage, "real"|"null") -> (n, 9)
        sets = {"ft": (self.rows, "ens"), "test": (self.test, "ens")}
        if self.pre is not None:
            sets["pre"] = (self.pre, "era5")
        for stage, (rows, kind) in sets.items():
            args = {
                "primary": rows.primary,
                "origin_idx": rows.origin_idx,
                "valid_idx": rows.target_idx,
            }
            if needs_real:
                self.raw[(stage, "real")] = windows.real(kind=kind, **args)
            if needs_null:
                self.raw[(stage, "null")] = windows.shuffled(kind=kind, **args)
        self.seam_raw: dict[str, np.ndarray] = {}
        if seam:
            persistence = EnsembleLookup(seam="persistence")
            for stage in ("ft", "test"):
                rows = sets[stage][0]
                self.seam_raw[stage] = windows.real(
                    kind="ens", primary=rows.primary, origin_idx=rows.origin_idx,
                    valid_idx=rows.target_idx, ens=persistence,
                )  # fmt: skip
            del persistence
        train = ~self.rows.is_val
        reference = self.raw[("ft", "real")] if ("ft", "real") in self.raw else windows.real(
            kind="ens", primary=self.rows.primary, origin_idx=self.rows.origin_idx, valid_idx=self.rows.target_idx
        )  # fmt: skip
        self.mean = reference[train].mean(axis=0)
        self.std = np.maximum(reference[train].std(axis=0), 1e-6)

    def matrix(self, *, arm: str, stage: str, x: np.ndarray, seam: bool = False) -> np.ndarray:
        """Return the stage's inputs with the arm's standardised window columns appended."""
        columns = ARM_COLUMNS[arm]
        if not columns:
            return x
        if seam:
            raw = self.seam_raw[stage]
        else:
            raw = self.raw[(stage, "null" if arm == "W-null" else "real")]
        index = [ALL.index(name) for name in columns]
        scaled = (raw[:, index] - self.mean[index]) / self.std[index]
        scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
        return np.concatenate([x, scaled], axis=1)


def _tensor_set(
    *, x: np.ndarray, y: np.ndarray, w: np.ndarray, primary: np.ndarray, is_val: np.ndarray,
    rows: data.Rows, device: str,
) -> dict[str, torch.Tensor]:  # fmt: skip
    z = (y - rows.target_mean[primary]) / rows.target_std[primary]
    return {
        "x": torch.from_numpy(x).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(w.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device),
        "v": torch.from_numpy(is_val).to(device),
    }


def _fine_tune_and_forecast(
    *, fold: Fold, arm: str, parent: str, seed: int, seam: bool, device: str
) -> str:
    """Fine-tune an arm from `parent`'s pretrained weights and write its forecasts."""
    started = time.monotonic()
    rows = fold.rows
    x = fold.matrix(arm=parent, stage="ft", x=rows.x, seam=seam)
    t = _tensor_set(
        x=x,
        y=rows.y,
        w=rows.weight,
        primary=rows.primary,
        is_val=rows.is_val,
        rows=rows,
        device=device,
    )
    model = c.new_model(outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)).to(
        device
    )
    c.load_pretrained(model=model, path=pretrained_path(arm=parent, month=fold.month, seed=seed))
    model.to(device)
    epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device)  # fmt: skip
    test = fold.test
    scored = data.TestRows(
        x=fold.matrix(arm=parent, stage="test", x=test.x, seam=seam), primary=test.primary,
        origin_idx=test.origin_idx, target_idx=test.target_idx, lead=test.lead,
        origin=test.origin, valid=test.valid,
    )  # fmt: skip
    frame = c.forecast_frame(model=model, outlier=False, test=scored, select=np.ones(len(test.x), dtype=bool), rows=rows, widen=1.0, device=device)  # fmt: skip
    out = prediction_path(arm=arm, month=fold.month, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    del t, model
    torch.cuda.empty_cache()
    return f"  fit {arm} {fold.month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(rows.y):,} rows"


def run(
    *,
    months: list[str],
    seeds: list[int],
    arms: list[str],
    do_pretrain: bool,
    do_fit: bool,
    device: str,
) -> None:
    """Pretrain and/or fine-tune and forecast each arm (and W-seam), fold, and seed (skips outputs that exist)."""
    clean = c.clean_stack()
    for month in months:
        month_arms = [a for a in arms if not (a == "W0" and month in REUSED_BASE)]
        todo_pre = [(a, s) for a in month_arms if a != SEAM_ARM for s in seeds if do_pretrain and not pretrained_path(arm=a, month=month, seed=s).exists()]  # fmt: skip
        todo_fit = [(a, s) for a in month_arms if a != SEAM_ARM for s in seeds if do_fit and not prediction_path(arm=a, month=month, seed=s).exists()]  # fmt: skip
        todo_seam = [s for s in seeds if SEAM_ARM in month_arms and do_fit and not prediction_path(arm=SEAM_ARM, month=month, seed=s).exists()]  # fmt: skip
        if not todo_pre and not todo_fit and not todo_seam:
            continue
        t0 = time.monotonic()
        used = [a for a, _ in (*todo_pre, *todo_fit)]
        fold = Fold(
            month=month,
            clean=clean,
            need_pre=bool(todo_pre),
            arms=used or [SEAM_PARENT],
            seam=bool(todo_seam),
        )
        print(
            f"{month}: feeder {time.monotonic() - t0:.0f} s; placebo days without donor per draw {fold.windows.days['ens'].unmatched} (ensemble) and {fold.windows.days['era5'].unmatched} (ERA5); days that needed a wider group {fold.windows.days['ens'].widened} and {fold.windows.days['era5'].widened}; rows without any donor {fold.windows.unfilled}",
            flush=True,
        )
        rows = fold.rows
        for arm, seed in todo_pre:
            assert fold.pre is not None
            pre = fold.pre
            x = fold.matrix(arm=arm, stage="pre", x=pre.x)
            t = _tensor_set(
                x=x,
                y=pre.y,
                w=np.ones(len(pre.y)),
                primary=pre.primary,
                is_val=pre.is_val,
                rows=rows,
                device=device,
            )
            started = time.monotonic()
            torch.manual_seed(seed)
            model = c.new_model(
                outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device)  # fmt: skip
            r448.atomic_save(
                state=model.state_dict(), target=pretrained_path(arm=arm, month=month, seed=seed)
            )
            print(
                f"  pretrain {arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del t, model, x
            torch.cuda.empty_cache()
        for arm, seed in todo_fit:
            print(
                _fine_tune_and_forecast(
                    fold=fold, arm=arm, parent=arm, seed=seed, seam=False, device=device
                ),
                flush=True,
            )
        for seed in todo_seam:
            print(
                _fine_tune_and_forecast(
                    fold=fold, arm=SEAM_ARM, parent=SEAM_PARENT, seed=seed, seam=True, device=device
                ),
                flush=True,
            )
        del fold


def checks(*, month: str) -> None:
    """Print the sanity checks of one fold: columns found, placebo coverage, standardised ranges, no NaN."""
    clean = c.clean_stack()
    fold = Fold(
        month=month, clean=clean, need_pre=True, arms=["W-T", "W-S", "W-X", "W-null"], seam=True
    )
    print(
        f"{month}: {len(fold.rows.y):,} fine-tuning rows, {len(fold.test.x):,} test rows, {len(fold.pre.x):,} pretraining rows"  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    print(
        f"placebo: days without a donor per draw {fold.windows.days['ens'].unmatched} (ensemble), widened {fold.windows.days['ens'].widened}; rows without any donor {fold.windows.unfilled}"
    )
    for arm in ("W-T", "W-S", "W-X", "W-null"):
        for stage, base in (("pre", fold.pre.x), ("ft", fold.rows.x), ("test", fold.test.x)):  # ty: ignore[unresolved-attribute]
            m = fold.matrix(arm=arm, stage=stage, x=base)
            extra = m[:, base.shape[1] :]
            print(
                f"  {arm} {stage}: {extra.shape[1]} columns, mean {extra.mean():+.3f}, sd {extra.std():.3f}, min {extra.min():+.2f}, max {extra.max():+.2f}, NaN {int(np.isnan(extra).sum())}"
            )
    for stage, base in (("ft", fold.rows.x), ("test", fold.test.x)):
        real = fold.matrix(arm="W-T", stage=stage, x=base)[:, base.shape[1] :]
        seam = fold.matrix(arm="W-T", stage=stage, x=base, seam=True)[:, base.shape[1] :]
        print(
            f"  W-T seam variants {stage}: mean absolute difference {np.abs(real - seam).mean():.4f}"
        )
    same = float(np.mean(fold.raw[("ft", "null")] == fold.raw[("ft", "real")]))
    print(f"  share of placebo entries equal to the real entries (fine-tuning): {same:.4f}")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("time", "pretrain", "fit", "both", "checks"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--arms", nargs="+", default=[*CORE_ARMS, SEAM_ARM], choices=[*ARM_COLUMNS, SEAM_ARM]
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "checks":
        checks(month=args.months[0])
        return
    if args.command == "time":
        run(
            months=args.months[:1],
            seeds=args.seeds[:1],
            arms=args.arms,
            do_pretrain=True,
            do_fit=True,
            device=args.device,
        )
        return
    run(months=args.months, seeds=args.seeds, arms=args.arms, do_pretrain=args.command in ("pretrain", "both"), do_fit=args.command in ("fit", "both"), device=args.device)  # fmt: skip


if __name__ == "__main__":
    _ = ML_EXPERIMENTS_DATA_DIR
    main()
