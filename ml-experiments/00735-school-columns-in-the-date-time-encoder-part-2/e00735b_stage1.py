# ruff: noqa: E501, PT018
"""Experiment 735 part 2: stage 1 (ERA5 and CAMS pre-training) with the 8 columns, and shared helpers.

README. `pretrain_arm` is a copy of `e00574_fit.pretrain_wt` (experiment 574, the function that
experiment 677's `e00677_k1.py pretrain` calls), changed in three places that the comments mark with
`PART 2`: the pre-training matrix has 8 more trailing columns, which hold the arm's school and coast
columns computed from each row's valid time (grid start plus `target_idx` times the grid step, the
construction that `pretrain_b1_columns` uses for its valid times) and Primary; the network, tables,
and loss history come from the caller; and a smoke run caps the epochs. Every other line is
experiment 574's. The module takes the patched module of experiment 574 as an argument (`e`), because
that module parses the command line when it is imported, so this file never imports it.

The module also holds `History` (per-epoch mean training and validation loss, as part 1's runner
records it) and `NewColumnTables` (the arm's `table_new` on a device, cached by grid length).
PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import e00574_model as mod  # ty: ignore[unresolved-import]
import e00735b_columns as cols  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

STAGE_1_FIRST_VALID: Final = datetime(2019, 12, 1, tzinfo=UTC)
"""The first valid time that a stage-1 row may have (the start of the ERA5 pre-training record)."""


class History:
    """Per-epoch mean training and validation negative log-likelihood, recorded from the row loss."""

    def __init__(self) -> None:
        """Start empty."""
        self.rows: list[dict[str, float]] = []
        self.in_val = False
        self.tr: torch.Tensor | None = None
        self.va: torch.Tensor | None = None
        self.tr_n = 0
        self.va_n = 0

    def reset(self) -> None:
        """Forget the recorded epochs."""
        self.rows = []
        self.in_val = False
        self.tr = self.va = None
        self.tr_n = self.va_n = 0

    def add(self, *, nll: torch.Tensor) -> None:
        """Add one batch's row losses; training batches have gradients on, validation batches do not."""
        d = nll.detach().double()
        if torch.is_grad_enabled():
            if self.in_val:
                self.close()
            self.tr = d.sum() if self.tr is None else self.tr + d.sum()
            self.tr_n += d.numel()
        else:
            self.in_val = True
            self.va = d.sum() if self.va is None else self.va + d.sum()
            self.va_n += d.numel()

    def close(self) -> None:
        """Close the epoch that is open."""
        assert self.tr is not None and self.va is not None
        self.rows.append(
            {
                "train_nll_unweighted": float(self.tr) / self.tr_n,
                "val_nll": float(self.va) / self.va_n,
            }
        )
        self.in_val = False
        self.tr = self.va = None
        self.tr_n = self.va_n = 0

    def finish(self) -> tuple[list[dict[str, float]], int]:
        """Return the epochs and the 1-based epoch that experiment 574's early-stopping rule stores."""
        if self.in_val:
            self.close()
        best, chosen = float("inf"), -1
        for i, r in enumerate(self.rows):
            if r["val_nll"] < best - 1e-5:
                best, chosen = r["val_nll"], i
        return self.rows, chosen + 1


def write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    tmp.rename(path)


def check_table_matches_rows(
    *, table: torch.Tensor, block: np.ndarray, primary: np.ndarray, valid_us: np.ndarray,
    first_us: int, step_us: int, n_rows: int = 1000, seed: int = 735,
) -> None:  # fmt: skip
    """Raise `AssertionError` unless the lag table equals the row columns on random rows.

    The lag path reads `table[primary, grid index]`, so on every checked row it must equal the
    row path's block: the same grid index (valid time minus grid start over the step, with no
    one-step offset), the same Primary, the same columns.

    Args:
        table: The arm's `table_new`, shape (Primaries, grid, columns).
        block: The row path's columns, shape (rows, columns).
        primary: The rows' Primary indices.
        valid_us: The rows' valid times in microseconds since the epoch.
        first_us: The first grid time in microseconds since the epoch.
        step_us: The grid step in microseconds.
        n_rows: How many random rows to check.
        seed: The seed of the random choice.
    """
    assert len(block) == len(primary) == len(valid_us) > 0, (
        len(block),
        len(primary),
        len(valid_us),
    )
    pick = np.random.default_rng(seed).choice(
        len(block), size=min(n_rows, len(block)), replace=False
    )
    offset = np.asarray(valid_us, dtype=np.int64)[pick] - first_us
    assert (offset % step_us == 0).all(), "a valid time is off the grid"
    idx = offset // step_us
    assert idx.min() >= 0 and idx.max() < table.shape[1], (idx.min(), idx.max(), table.shape)
    got = table[
        torch.from_numpy(np.asarray(primary)[pick]).to(table.device).long(),
        torch.from_numpy(idx).to(table.device),
    ]
    assert np.array_equal(got.cpu().numpy(), block[pick]), (
        "the lag table differs from the row columns"
    )


class NewColumnTables:
    """The arm's `table_new` (Primaries, grid, 8) on a device, built once per grid length."""

    def __init__(
        self, *, arm: cols.ArmB, first_us: int, step_us: int, device: str,
        c_true: np.ndarray, c_permuted: np.ndarray,
    ) -> None:  # fmt: skip
        """Set the arm, the grid, the device, and the closeness arrays."""
        self.arm, self.first_us, self.step_us, self.device = arm, first_us, step_us, device
        self.c_true, self.c_permuted = c_true, c_permuted
        self._cache: dict[int, torch.Tensor] = {}

    def get(self, *, n_grid: int) -> torch.Tensor:
        """Return the table for a grid of `n_grid` half-hours."""
        if n_grid not in self._cache:
            table = cols.grid_table(
                arm=self.arm, first_us=self.first_us, step_us=self.step_us, n_grid=n_grid,
                c_true=self.c_true, c_permuted=self.c_permuted,
            )  # fmt: skip
            self._cache[n_grid] = torch.from_numpy(table).to(self.device)
        return self._cache[n_grid]


def pretrain_arm(  # noqa: PLR0915
    *, e: Any, months: list[str], seeds: list[int], device: str, arm: cols.ArmB,
    c_true: np.ndarray, c_permuted: np.ndarray, history_dir: Path, smoke: bool, code_hash: str,
) -> None:  # fmt: skip
    """Pre-train the arm's network on each fold's ERA5 and CAMS rows, with the arm's 8 columns.

    Args:
        e: The patched module of experiment 574 (`e00574_fit`), with `new_model`, `W`, and
            `row_nll` set by the runner.
        months: The refit months.
        seeds: The seeds.
        device: `cuda` (or `cpu` for a test).
        arm: The arm; R1 appends nothing.
        c_true: Closeness per Primary index.
        c_permuted: The placebo closeness per Primary index.
        history_dir: Where the per-epoch loss history of each seed is written.
        smoke: Cap every pre-training at one epoch.
        code_hash: The code hash that is recorded beside each history file.
    """
    assert e.SPEC.pretrain_from != "V3D5", "this arm uses the stored pre-training of experiment 572"
    c, k, r448, d560, lg = e.c, e.k, e.r448, e.d560, e.lg
    clean = c.clean_stack()
    first_us, step_us = c.prep422.GRID_START_US, c.prep422.STEP_US
    first_new = mod.b1_cols(n_b1=e.N_B1).stop
    hist = History()
    inner_row_nll = e.row_nll

    def row_nll_recorded(*, out: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        nll = inner_row_nll(out=out, z=z, p=p)
        hist.add(nll=nll)
        return nll

    for month in months:
        todo = [s for s in seeds if not (e.pretrained_dir() / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        e.PRETRAINING["on"] = True
        mr = k.build_month(month=month, clean=clean)  # sets CTX["stats"] and the ENS-side lag table
        e.PRETRAINING["on"] = False
        refit_t = c.month_start(month=month)
        pre = r448.pretrain_rows(rows=mr.base, refit=refit_t, clean=clean)
        assert pre.x.shape[1] == 37
        latest = datetime.fromtimestamp((pre.target_idx.max() * step_us + first_us) / 1e6, UTC)
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
            flush=True,
        )
        blk = e.block_of(
            clean=clean, primary=pre.primary, target_idx=pre.target_idx, origin_idx=pre.origin_idx
        )
        ecols, origin = d560.raw_table(source="pre", cache=e.W560 / "expected")
        table_pre = torch.from_numpy(
            d560.standardised_table(cols=ecols, standardiser=mr.base.standardiser)
        ).to(device)
        del ecols
        e.CTX["table_pre"] = table_pre
        n_in = e.N_LAG_INPUTS + (1 if e.SPEC.columns else 0) + e.N_B1 + arm.width
        x_gpu = torch.empty((len(pre.x), n_in), dtype=torch.float32, device=device)
        for a in range(0, len(pre.x), 1_000_000):
            x_gpu[a : a + 1_000_000, :37] = torch.from_numpy(pre.x[a : a + 1_000_000]).to(device)
        if e.SPEC.columns:
            tau = e.WT.rows(
                kind="era5", primary=pre.primary, origin_idx=pre.origin_idx, valid_idx=pre.target_idx, null=False,
            )  # fmt: skip
            x_gpu[:, e.N_LAG_INPUTS] = torch.from_numpy(tau[:, 0]).to(device)
        if e.SPEC.b1:
            extra_b1 = e.pretrain_b1_columns(month=month, pre=pre)
            assert extra_b1.shape == (len(pre.x), e.N_B1) and np.isfinite(extra_b1).all()
            x_gpu[:, mod.b1_cols(n_b1=e.N_B1)] = torch.from_numpy(extra_b1).to(device)
            del extra_b1
        # >>> PART 2 BEGIN (the arm's 8 columns from each row's valid time and Primary)
        if arm.columns:
            valid_us = first_us + pre.target_idx.astype(np.int64) * step_us
            assert valid_us.min() >= STAGE_1_FIRST_VALID.timestamp() * 1e6, (
                "a stage-1 row is earlier than 2019-12-01",
                datetime.fromtimestamp(valid_us.min() / 1e6, UTC),
            )
            block = cols.extra_columns(
                valid_us=valid_us,
                primary=pre.primary,
                arm=arm,
                c_true=c_true,
                c_permuted=c_permuted,
            )  # raises if a valid time is outside the school calendar
            cols.check_block(arm=arm, block=block, valid_us=valid_us, kind="train")
            check_table_matches_rows(
                table=NewColumnTables(
                    arm=arm, first_us=first_us, step_us=step_us, device=device,
                    c_true=c_true, c_permuted=c_permuted,
                ).get(n_grid=table_pre.shape[1]),
                block=block, primary=pre.primary, valid_us=valid_us, first_us=first_us, step_us=step_us,
            )  # fmt: skip
            x_gpu[:, first_new : first_new + arm.width] = torch.from_numpy(block).to(device)
            e.CHECK["months"].setdefault(month, {})["new_columns_pretrain"] = dict(
                zip(arm.names, (block != 0).mean(axis=0).round(4).tolist(), strict=True)
            )
            print(month, "stage-1 new columns, non-zero shares", e.CHECK["months"][month]["new_columns_pretrain"], flush=True)  # fmt: skip
            del block, valid_us
        # <<< PART 2 END
        stub = np.zeros((min(2_000_000, len(blk)), e.N_LAG_INPUTS), np.float32)
        stub[:, 37:] = blk[: len(stub)]
        ex = lg.check_lag_ages(
            x=stub, origin_us=first_us + pre.origin_idx[: len(stub)] * step_us,
            primary=pre.primary[: len(stub)], origin_tab=np.asarray(origin), cap_hours=e.CAP_HOURS, label=f"{month} pretrain",
        )  # fmt: skip
        del stub
        print(ex, flush=True)
        e.CHECK["months"][month]["lag_example_pretrain"] = ex
        z = (pre.y - mr.base.target_mean[pre.primary]) / mr.base.target_std[pre.primary]
        t = {
            "z": torch.from_numpy(z.astype(np.float32)).to(device), "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(pre.primary).to(device), "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        n_primaries = len(mr.base.target_mean)
        origin_idx = pre.origin_idx
        del pre, mr
        for seed in todo:
            started = time.monotonic()
            for a in range(0, len(blk), 1_000_000):
                x_gpu[a : a + 1_000_000, 37 : e.N_LAG_INPUTS] = torch.from_numpy(
                    blk[a : a + 1_000_000]
                ).to(device)
            share = e.apply_dropout(x=x_gpu, origin_idx=origin_idx, seed=seed)
            torch.manual_seed(seed)
            model = e.new_model(n_primaries=n_primaries, stage="pre").to(device)
            hist.reset()
            e.row_nll = row_nll_recorded
            try:
                epochs, best = e.train(
                    model=model, x=x_gpu, z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    lr=c.PRETRAIN_LR, max_epochs=1 if smoke else c.MAX_EPOCHS, fresh_lr=None,
                    frozen=False, anchor=None, lam=0.0, device=device,
                )  # fmt: skip
            finally:
                e.row_nll = inner_row_nll
            rows, stored_epoch = hist.finish()
            assert len(rows) == epochs, (len(rows), epochs)
            assert abs(rows[stored_epoch - 1]["val_nll"] - best) < 1e-4, (
                rows[stored_epoch - 1],
                best,
            )
            write_json(
                path=history_dir / f"{month}_s{seed}.json",
                payload={
                    "arm": arm.name, "stage": 1, "code_hash": code_hash, "epochs_run": epochs,
                    "stored_epoch": stored_epoch, "val_nll": best, "history": rows,
                },
            )  # fmt: skip
            target = e.pretrained_dir() / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {arm.name} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, dropout {share}",
                flush=True,
            )
        del t, x_gpu, blk
        e.CTX.pop("table_pre", None)
        torch.cuda.empty_cache()
