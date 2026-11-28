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

import hashlib
import json
import time
from collections.abc import Callable
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


TABLE_PREFIX_STEPS: Final = 10_000
"""Spacing, in grid steps, of the prefixes of `table_new` that the stage tag hashes."""
GATE_ROWS: Final = 2048
"""Rows in the gate's training batch and validation batch."""
GATE_SEED: Final = 735
"""Seed of the random choice of the gate's rows."""


def _sha(*, array: np.ndarray) -> str:
    """Return the hex sha256 of an array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def table_tags(*, model: Any) -> dict[str, Any]:
    """Return the hashes of the model's `table_new`: its length, the hash of the whole table, and of its prefixes.

    The stage-1 table and the stage-2 and stage-3 tables may differ in length, but each is a prefix
    of the same deterministic table, so the tag also hashes every prefix whose length is a multiple
    of `TABLE_PREFIX_STEPS`.
    """
    table = getattr(model, "table_new", None)
    if table is None:
        return {"n_grid": 0, "full": "none", "prefix": {}}
    array = table.detach().cpu().numpy()
    n_grid = int(array.shape[1])
    prefixes = {
        str(k): _sha(array=array[:, :k])
        for k in range(TABLE_PREFIX_STEPS, n_grid + 1, TABLE_PREFIX_STEPS)
    }
    return {"n_grid": n_grid, "full": _sha(array=array), "prefix": prefixes}


def datetime_sha(*, model: Any) -> str:
    """Return the sha256 of the whole state of `model.datetime`: every parameter (weights and biases of both layers), by name."""
    digest = hashlib.sha256()
    for name, value in sorted(model.datetime.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def write_stage_tag(*, path: Path, arm: cols.ArmB, model: Any) -> None:
    """Write the arm, the `table_new` hashes, and the hash of the date-time encoder's whole state beside a stage-1 checkpoint."""
    write_json(
        path=path,
        payload={"arm": arm.name, "table_new": table_tags(model=model), "datetime_state_sha256": datetime_sha(model=model)},
    )  # fmt: skip


def check_stage_tag(*, path: Path, arm: cols.ArmB, model: Any) -> None:
    """Raise `AssertionError` unless the model's `table_new` and date-time encoder state match the stage-1 tag.

    The whole table is compared when its length equals the tag's; otherwise the longest prefix whose
    length is a multiple of `TABLE_PREFIX_STEPS` and at most the shorter length is compared.
    """
    assert path.exists(), ("the stage-1 tag is missing", str(path))
    tag = json.loads(path.read_text())
    assert tag["arm"] == arm.name, (tag["arm"], arm.name)
    now, then = table_tags(model=model), tag["table_new"]
    assert (now["n_grid"] == 0) == (then["n_grid"] == 0), (
        "table_new is present in only one of the two"
    )
    if now["n_grid"]:
        if now["n_grid"] == then["n_grid"]:
            assert now["full"] == then["full"], "table_new differs from stage 1's"
        k = min(now["n_grid"], then["n_grid"]) // TABLE_PREFIX_STEPS * TABLE_PREFIX_STEPS
        assert k > 0, (now["n_grid"], then["n_grid"])
        assert now["prefix"][str(k)] == then["prefix"][str(k)], (
            "table_new differs from stage 1's over the first",
            k,
        )
    assert tag["datetime_state_sha256"] == datetime_sha(model=model), (
        "the date-time encoder differs from stage 1's"
    )


def _recorder(*, seen: list[np.ndarray]) -> Any:
    """Return a forward pre-hook that appends the tensor reaching the module."""

    def hook(_module: Any, args: tuple) -> None:
        seen.append(args[0].detach().cpu().numpy().copy())

    return hook


def gate_stage(  # noqa: PLR0915
    *, model: Any, x: torch.Tensor, primary: torch.Tensor, is_val: torch.Tensor,
    valid_us: np.ndarray, arm: cols.ArmB, stage: int, month: str, seed: int, out_dir: Path,
    first_us: int, step_us: int, c_true: np.ndarray, c_permuted: np.ndarray, n_rows: int = GATE_ROWS,
) -> dict[str, Any]:  # fmt: skip
    """Check, on real rows, the tensors that reach the date-time encoder, and write them to a JSON file.

    For one training batch and one validation batch of `n_rows` random rows, a forward pre-hook on
    `model.datetime` reads the encoder's input on the three paths (the row, the day lag, the week
    lag). An arm with the columns must show the arm's 8 columns (`extra_columns` at the row's valid
    time, and at the grid times in `x[:, 43]` and `x[:, 44]`, zeros where the index is below 0) in
    input columns 21 to 28. An arm without them must show an encoder input of 21 columns plus the
    embedding. The file `checks_stage_<stage>_<month>_s<seed>.json` holds, for each batch, path, and
    column, the number of distinct values and the shares of ones and of non-zero values.

    Args:
        model: The network, with its tables attached.
        x: The rows' input matrix (all rows of the stage).
        primary: The rows' Primary indices.
        is_val: True for validation rows.
        valid_us: The rows' valid times in microseconds since the epoch.
        arm: The arm.
        stage: 1, 2, or 3.
        month: The refit month.
        seed: The seed.
        out_dir: The folder of the JSON file.
        first_us: The first grid time in microseconds since the epoch.
        step_us: The grid step in microseconds.
        c_true: Closeness per Primary index.
        c_permuted: The placebo closeness per Primary index.
        n_rows: Rows per batch.

    Returns:
        The payload that was written.
    """
    assert len(valid_us) == len(x) == len(primary) == len(is_val), (
        len(valid_us),
        len(x),
        len(primary),
        len(is_val),
    )
    width, emb_dim = model.datetime[0].in_features, model.embedding.embedding_dim
    assert width == 21 + arm.width + emb_dim, ("encoder input width", width, arm.width, emb_dim)
    rng = np.random.default_rng(GATE_SEED)
    was_training = model.training
    model.eval()
    batches: dict[str, Any] = {}
    try:
        for batch, mask in (("train", ~is_val), ("val", is_val)):
            pool = torch.nonzero(mask).squeeze(1).cpu().numpy()
            if not len(pool):
                batches[batch] = {"n_rows": 0}
                continue
            rows = np.sort(rng.choice(pool, size=min(n_rows, len(pool)), replace=False))
            idx_t = torch.from_numpy(rows).to(x.device)
            seen: list[np.ndarray] = []
            hook = model.datetime.register_forward_pre_hook(_recorder(seen=seen))
            try:
                with torch.no_grad():
                    model(x[idx_t], primary[idx_t])
            finally:
                hook.remove()
            assert len(seen) == 3, len(seen)  # the row, the day lag, the week lag
            xb = x[idx_t].cpu().numpy()
            pb = primary[idx_t].cpu().numpy()
            vb = np.asarray(valid_us)[rows]
            paths: dict[str, Any] = {}
            for k, name in enumerate(("row", "day lag", "week lag")):
                inp = seen[k]
                assert inp.shape == (len(rows), width), (name, inp.shape)
                entry: dict[str, Any] = {"input_width": int(inp.shape[1]), "columns": []}
                if arm.columns:
                    new = inp[:, 21 : 21 + arm.width]
                    if k == 0:
                        when = vb
                        present = np.ones(len(rows), dtype=bool)
                    else:
                        lag = xb[:, 43 + k - 1].astype(np.int64)
                        present = lag >= 0
                        when = np.where(present, first_us + lag * step_us, vb)
                    want = cols.extra_columns(
                        valid_us=when, primary=pb, arm=arm, c_true=c_true, c_permuted=c_permuted
                    )
                    want[~present] = 0.0
                    entry["equals_expected"] = bool(np.array_equal(new, want))
                    assert entry["equals_expected"], (
                        arm.name,
                        stage,
                        name,
                        "the encoder's new columns differ from the arm's columns",
                    )
                    for j, col in enumerate(arm.names):
                        v = new[:, j]
                        entry["columns"].append(
                            {
                                "column": col, "distinct": len(np.unique(v)),
                                "min": float(v.min()), "max": float(v.max()),
                                "share_one": float((v == 1.0).mean()), "share_nonzero": float((v != 0.0).mean()),
                            }
                        )  # fmt: skip
                    if name == "row":
                        entry["per_primary_mean"] = {
                            str(p): new[pb == p].mean(axis=0).tolist() for p in np.unique(pb)
                        }
                        entry["school_sample"] = (
                            new[:, cols.SCHOOL_SLICE].astype(np.int8).tolist()
                            if batch == "train"
                            else []
                        )
                    if batch == "train" and name == "row":
                        assert all(c["distinct"] > 1 for c in entry["columns"]), (
                            arm.name,
                            stage,
                            "a column is constant",
                            entry["columns"],
                        )
                else:
                    entry["equals_expected"] = None
                paths[name] = entry
            row_hash = hashlib.sha256(
                vb.astype(np.int64).tobytes() + pb.astype(np.int64).tobytes()
            ).hexdigest()
            batches[batch] = {"n_rows": len(rows), "row_hash": row_hash, "paths": paths}
    finally:
        model.train(was_training)
    payload = {
        "arm": arm.name, "stage": stage, "month": month, "seed": seed, "n_new": arm.width,
        "encoder_input_width": width, "embedding_dim": emb_dim, "batches": batches,
    }  # fmt: skip
    write_json(path=out_dir / f"checks_stage_{stage}_{month}_s{seed}.json", payload=payload)
    return payload


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
    checks_dir: Path, check_b1_stats: Callable[[str], None],
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
        checks_dir: Where the stage-1 gate writes its JSON file of each seed.
        check_b1_stats: Called with the month right after the B1 statistics are written; raises if they differ from experiment 677's.
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
            check_b1_stats(month)  # the statistics are written by `pretrain_b1_columns`
            x_gpu[:, mod.b1_cols(n_b1=e.N_B1)] = torch.from_numpy(extra_b1).to(device)
            del extra_b1
        # >>> PART 2 BEGIN (the arm's 8 columns from each row's valid time and Primary)
        valid_us = first_us + pre.target_idx.astype(np.int64) * step_us
        if arm.columns:
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
            del block
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
        prim_t = t["p"]
        for seed in todo:
            started = time.monotonic()
            for a in range(0, len(blk), 1_000_000):
                x_gpu[a : a + 1_000_000, 37 : e.N_LAG_INPUTS] = torch.from_numpy(
                    blk[a : a + 1_000_000]
                ).to(device)
            share = e.apply_dropout(x=x_gpu, origin_idx=origin_idx, seed=seed)
            torch.manual_seed(seed)
            model = e.new_model(n_primaries=n_primaries, stage="pre").to(device)
            gate_stage(
                model=model, x=x_gpu, primary=prim_t, is_val=t["v"], valid_us=valid_us, arm=arm, stage=1,
                month=month, seed=seed, out_dir=checks_dir, first_us=first_us, step_us=step_us,
                c_true=c_true, c_permuted=c_permuted,
            )  # fmt: skip
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
            write_stage_tag(path=target.with_suffix(".tag.json"), arm=arm, model=model)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain {arm.name} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, dropout {share}",
                flush=True,
            )
        del t, x_gpu, blk, valid_us
        e.CTX.pop("table_pre", None)
        torch.cuda.empty_cache()
