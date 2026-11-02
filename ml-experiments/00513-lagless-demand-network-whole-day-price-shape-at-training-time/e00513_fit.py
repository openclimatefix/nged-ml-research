# ruff: noqa: E501
r"""Pretrain, fine-tune, and forecast the arms of experiment 513 (whole-day shape as a training-time input).

Commands: `time` (one fold and seed, every stage timed), `pretrain`, `fit`.

Arms (all the lag-free network of experiment 448's `L0PT`, monthly refit, 182.6-day weight, as the
`L-M` arm of experiment 510):

- `P0`: no shape columns, pretrained here with the same loop as the other arms. Experiment 448's `L0PT`
  weights are not reused, because experiment 448 trained them with a different loop.
- `PO`: the ten shape features and the missing flag of the Market Index price of the target
  half-hour's local day, at training and scoring (hindsight, never deployable).
- `PP`: the same columns from a seeded other day of the same calendar month, day type, and length.
- `PN`: the same ten features of national demand minus wind of the target local day.

The shape columns are appended to the 37 non-lag columns, so the 48-wide network is pretrained from
scratch with the same rows and loop as `L0PT`.

```bash
. <env script>; OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00513-*/e00513_fit.py time
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00513_data as shapes
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00513-lagless-demand-network-whole-day-price-shape-at-training-time"
    / "work"
)
SOURCE_OF: Final[dict[str, str | None]] = {
    "P0": None,
    "PO": "price",
    "PP": "placebo",
    "PN": "national",
}
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MONTHS: Final[tuple[str, ...]] = c.MONTHS


def pretrained_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's pretrained weights live (every arm, `P0` included, is pretrained here)."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def _prepare(*, month: str, clean: np.ndarray, with_pretrain_rows: bool) -> dict:
    """Return the fold's rows, shape tables, test rows, and (optionally) the pretraining rows, with shape columns appended."""
    refit = c.month_start(month=month)
    rows = data.build_fold(refit=refit, clean=clean)
    test = data.build_test(month=month, standardiser=rows.standardiser)
    fit_until = int((c.us(t=refit) - c.prep422.GRID_START_US) // c.prep422.STEP_US)
    tables = shapes.ShapeTables(fit_until_idx=fit_until)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean) if with_pretrain_rows else None
    return {"rows": rows, "test": test, "tables": tables, "pre": pre}


def _x(
    *, x: np.ndarray, target_idx: np.ndarray, tables: shapes.ShapeTables, source: str | None
) -> np.ndarray:
    """Return the inputs with the arm's shape columns appended."""
    if source is None:
        return x
    return np.concatenate([x, tables.columns(source=source, target_idx=target_idx)], axis=1)


def _tensor_set(
    *,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    primary: np.ndarray,
    is_val: np.ndarray,
    rows: data.Rows,
    device: str,
) -> dict[str, torch.Tensor]:
    z = (y - rows.target_mean[primary]) / rows.target_std[primary]
    return {
        "x": torch.from_numpy(x).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(w.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device),
        "v": torch.from_numpy(is_val).to(device),
    }


def run(
    *,
    months: list[str],
    seeds: list[int],
    arms: list[str],
    do_pretrain: bool,
    do_fit: bool,
    device: str,
) -> None:
    """Pretrain and/or fine-tune and forecast each arm, fold, and seed (skips outputs that exist)."""
    clean = c.clean_stack()
    for month in months:
        todo_pre = [
            (a, s) for a in arms for s in seeds
            if do_pretrain and not pretrained_path(arm=a, month=month, seed=s).exists()
        ]  # fmt: skip
        todo_fit = [
            (a, s) for a in arms for s in seeds
            if do_fit and not (W / "seed_preds" / a / f"{month}_s{s}.parquet").exists()
        ]  # fmt: skip
        if not todo_pre and not todo_fit:
            continue
        t0 = time.monotonic()
        prep = _prepare(month=month, clean=clean, with_pretrain_rows=bool(todo_pre))
        rows, test, tables, pre = prep["rows"], prep["test"], prep["tables"], prep["pre"]
        print(
            f"{month}: feeder {time.monotonic() - t0:.0f} s; shape tables: price-missing days {int(tables.price_missing.sum())}, placebo-unmatched {tables.unmatched}",
            flush=True,
        )
        for arm, seed in todo_pre:
            source = SOURCE_OF[arm]
            x = _x(x=pre.x, target_idx=pre.target_idx, tables=tables, source=source)
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
            target = pretrained_path(arm=arm, month=month, seed=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            r448.atomic_save(state=model.state_dict(), target=target)
            print(
                f"  pretrain {arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del t, model
            torch.cuda.empty_cache()
        for arm, seed in todo_fit:
            source = SOURCE_OF[arm]
            started = time.monotonic()
            x = _x(x=rows.x, target_idx=rows.target_idx, tables=tables, source=source)
            t = _tensor_set(
                x=x,
                y=rows.y,
                w=rows.weight,
                primary=rows.primary,
                is_val=rows.is_val,
                rows=rows,
                device=device,
            )
            model = c.new_model(
                outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            c.load_pretrained(model=model, path=pretrained_path(arm=arm, month=month, seed=seed))
            model.to(device)
            epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device)  # fmt: skip
            test_x = _x(x=test.x, target_idx=test.target_idx, tables=tables, source=source)
            scored = data.TestRows(x=test_x, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx, lead=test.lead, origin=test.origin, valid=test.valid)  # fmt: skip
            frame = c.forecast_frame(model=model, outlier=False, test=scored, select=np.ones(len(test.x), dtype=bool), rows=rows, widen=1.0, device=device)  # fmt: skip
            out = W / "seed_preds" / arm / f"{month}_s{seed}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            frame.write_parquet(tmp)
            tmp.rename(out)
            print(
                f"  fit {arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(rows.y):,} rows",
                flush=True,
            )
            del t, model
            torch.cuda.empty_cache()
        del prep, rows, test, pre


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pretrain", "fit", "both"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(SOURCE_OF), choices=list(SOURCE_OF))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        months=args.months, seeds=args.seeds, arms=args.arms,
        do_pretrain=args.command in ("pretrain", "both"),
        do_fit=args.command in ("fit", "both"), device=args.device,
    )  # fmt: skip


if __name__ == "__main__":
    main()
