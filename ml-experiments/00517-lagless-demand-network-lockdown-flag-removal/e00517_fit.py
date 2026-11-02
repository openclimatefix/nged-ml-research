# ruff: noqa: E501
r"""Pretrain, fine-tune, and forecast the arms of experiment 517 (is the lockdown flag still needed?).

Commands: `pretrain`, `fit`, `both`. The loop is experiment 513's `P0` loop (experiment 510's `L-M`
recipe: monthly refit, 182.6-day weight, pretraining on ERA5 and CAMS rows), so the three arms differ
only in the flag column and the pretraining rows.

Arms:

- `C0`: the 0/1 `covid` input kept (37 non-lag columns), all pretraining rows. The control.
- `C1`: the `covid` column removed (36 columns), all pretraining rows kept.
- `C2`: the `covid` column removed and the pretraining rows with the flag set (2020-03-23 to
  2021-07-19) dropped, from the training and the early-stopping rows alike.

The fine-tune rows (ensemble forecasts from 2024-04) hold no flagged row, so `C2` differs from `C1`
only in pretraining. `run` asserts this for every fold.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00517-*/e00517_fit.py both
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00517-lagless-demand-network-lockdown-flag-removal" / "work"
)
ARMS: Final[tuple[str, ...]] = ("C0", "C1", "C2")
DROP_COLUMN: Final[dict[str, bool]] = {"C0": False, "C1": True, "C2": True}
DROP_LOCKDOWN_ROWS: Final[dict[str, bool]] = {"C0": False, "C1": False, "C2": True}
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MONTHS: Final[tuple[str, ...]] = c.MONTHS


def pretrained_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's pretrained weights live (every arm is pretrained here)."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def covid_index(*, rows: data.Rows) -> int:
    """Return the position of the `covid` column among the non-lag inputs."""
    return rows.columns.index("covid")


def select_columns(*, x: np.ndarray, arm: str, covid: int) -> np.ndarray:
    """Return the inputs with the `covid` column removed for the arms that drop it."""
    if not DROP_COLUMN[arm]:
        return x
    return np.delete(x, covid, axis=1)


def _tensor_set(
    *, x: np.ndarray, y: np.ndarray, w: np.ndarray, primary: np.ndarray, is_val: np.ndarray,
    rows: data.Rows, device: str,
) -> dict[str, torch.Tensor]:  # fmt: skip
    z = (y - rows.target_mean[primary]) / rows.target_std[primary]
    return {
        "x": torch.from_numpy(np.ascontiguousarray(x)).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(w.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device),
        "v": torch.from_numpy(is_val).to(device),
    }


def run(  # noqa: PLR0915
    *, months: list[str], seeds: list[int], arms: list[str], do_pretrain: bool, do_fit: bool,
    device: str,
) -> None:  # fmt: skip
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
        refit = c.month_start(month=month)
        rows = data.build_fold(refit=refit, clean=clean)
        test = data.build_test(month=month, standardiser=rows.standardiser)
        pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean) if todo_pre else None
        covid = covid_index(rows=rows)
        fine_tune_flagged = int((rows.x[:, covid] == 1).sum())
        test_flagged = int((test.x[:, covid] == 1).sum())
        assert fine_tune_flagged == 0, "a fine-tune row carries the lockdown flag"
        assert test_flagged == 0, "a scored row carries the lockdown flag"
        flagged = int((pre.x[:, covid] == 1).sum()) if pre is not None else -1
        print(
            f"{month}: feeder {time.monotonic() - t0:.0f} s; pretraining rows "
            f"{0 if pre is None else len(pre.y):,}, flagged {flagged:,}; fine-tune rows "
            f"{len(rows.y):,}, flagged {fine_tune_flagged}; scored rows {len(test.x):,}, flagged {test_flagged}",
            flush=True,
        )
        for arm, seed in todo_pre:
            assert pre is not None
            assert flagged > 0 or not DROP_LOCKDOWN_ROWS[arm], "no flagged pretraining row to drop"
            keep = (
                pre.x[:, covid] == 0 if DROP_LOCKDOWN_ROWS[arm] else np.ones(len(pre.y), dtype=bool)
            )
            x = select_columns(x=pre.x[keep], arm=arm, covid=covid)
            t = _tensor_set(
                x=x, y=pre.y[keep], w=np.ones(int(keep.sum())), primary=pre.primary[keep],
                is_val=pre.is_val[keep], rows=rows, device=device,
            )  # fmt: skip
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
                f"  pretrain {arm} {month} s{seed}: {int(keep.sum()):,} rows, {x.shape[1]} columns, {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del t, model
            torch.cuda.empty_cache()
        for arm, seed in todo_fit:
            started = time.monotonic()
            x = select_columns(x=rows.x, arm=arm, covid=covid)
            t = _tensor_set(
                x=x, y=rows.y, w=rows.weight, primary=rows.primary, is_val=rows.is_val,
                rows=rows, device=device,
            )  # fmt: skip
            model = c.new_model(
                outlier=False, n_inputs=x.shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            c.load_pretrained(model=model, path=pretrained_path(arm=arm, month=month, seed=seed))
            model.to(device)
            epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device)  # fmt: skip
            scored = data.TestRows(x=select_columns(x=test.x, arm=arm, covid=covid), primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx, lead=test.lead, origin=test.origin, valid=test.valid)  # fmt: skip
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
        del rows, test, pre


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pretrain", "fit", "both"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        months=args.months, seeds=args.seeds, arms=args.arms,
        do_pretrain=args.command in ("pretrain", "both"),
        do_fit=args.command in ("fit", "both"), device=args.device,
    )  # fmt: skip


if __name__ == "__main__":
    main()
