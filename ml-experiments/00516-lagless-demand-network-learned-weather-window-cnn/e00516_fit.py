# ruff: noqa: E501
r"""Pretrain, fine-tune, and forecast the arms of experiment 516 (window encoder).

Commands: `time` (one fold and seed, every stage timed), `pretrain`, `fit`, `both`, `checks`.

Arms: `C-CNN` (24 h window centred on the valid time), `T-CNN` (24 h ending at the valid time), and
`C-CNN-null` (C-CNN reading the shuffled-day windows of experiment 515's placebo rule). Everything
else is experiment 515's recipe: the lag-free network of experiment 448's `L0PT` with experiment
510's `L-M` recipe (monthly refit, 182.6-day weight), pretrained in experiment 513's loop, on
experiment 515's ten winter folds and 3 seeds. Each arm is pretrained from scratch.

**Rows carry an index, not a window.** The network input is the 37 lag-free columns plus one column
holding the row's position in its stage's table. `WindowedNetwork.forward` looks the windows up on
the GPU from per-Primary dense arrays, so no (rows, 7, 48) array is ever materialised.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00516-*/e00516_fit.py time
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00515_fit as fit515  # ty: ignore[unresolved-import]
import e00516_cnn as cnn
import e00516_window as win
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00516-lagless-demand-network-learned-weather-window-cnn" / "work"
)
ARMS: Final[dict[str, str]] = {"C-CNN": "centred", "T-CNN": "trailing", "C-CNN-null": "centred"}
NULL_ARMS: Final[tuple[str, ...]] = ("C-CNN-null",)
OFFSETS: Final[dict[str, np.ndarray]] = {
    "centred": win.CENTRED_OFFSETS,
    "trailing": win.TRAILING_OFFSETS,
}
MONTHS: Final[tuple[str, ...]] = fit515.MONTHS
SEEDS: Final[tuple[int, ...]] = fit515.SEEDS
N_RUNS: Final[int] = 821
FIRST_RUN_ORIGIN: Final[int] = int(
    common.grid_index(times=prep.FIRST_RUN.replace(hour=9))
)  # the origin of the first daily run, 09:00 UTC
FWD_WIDTH: Final[int] = win.HORIZON - win.ORIGIN_OFFSET + 1


class WindowStore:
    """Per-Primary dense weather arrays on the GPU, and the gather of any row's 7 x 48 window."""

    def __init__(self, *, device: str, fit_until_idx: int) -> None:
        """Build ERA5 grids A and B, the ensemble history, and the ensemble run arrays."""
        self.device = device
        size = common.grid_times().len()
        times = common.grid_times()
        on_hour = (times.dt.minute() == 0).to_numpy()
        hour = np.where(on_hour, times.dt.hour().to_numpy(), -1)
        primaries = prep.primaries()
        n_p = primaries.height
        n_v = cnn.N_VARIABLES
        grid_a = np.empty((n_p, n_v, size), dtype=np.float32)
        grid_b = np.empty_like(grid_a)
        history = np.full((n_p, n_v, size), np.nan, dtype=np.float32)
        forward = np.full((n_p, n_v, N_RUNS, FWD_WIDTH), np.nan, dtype=np.float32)
        self.has_run = np.zeros((n_p, N_RUNS), dtype=bool)
        names = [f"mean_{v}" for v in prep418.WEATHER]
        for row in primaries.iter_rows(named=True):
            p = int(row["primary"])
            dense = prep418.dense_weather(series_id=int(row["time_series_id"]), site=row["site"])
            for v in range(n_v):
                series = np.where(on_hour, dense[:, v], np.nan).astype(np.float64)
                a, b = win.era5_step_grids(hourly=series, hour_of_day=hour)
                grid_a[p, v], grid_b[p, v] = a, b
            summary = (
                prep.load_summary(series_id=int(row["time_series_id"]))
                .select(
                    pl.col("power_fcst_init_time").dt.epoch("us").alias("origin"),
                    pl.col("valid_time").dt.epoch("us").alias("valid"),
                    *names,
                )
                .sort("origin", "valid")
            )
            oi = (summary["origin"].to_numpy() - c.prep422.GRID_START_US) // c.prep422.STEP_US
            vi = (summary["valid"].to_numpy() - c.prep422.GRID_START_US) // c.prep422.STEP_US
            run = (oi - FIRST_RUN_ORIGIN) // win.HALF_HOURS_PER_DAY
            j = vi - oi
            self.has_run[p, np.unique(run)] = True
            first_day = (j >= 1) & (j <= win.HALF_HOURS_PER_DAY)
            if np.unique(vi[first_day]).size != int(first_day.sum()):
                msg = "two runs' first days overlap on the grid"
                raise ValueError(msg)
            for v, name in enumerate(names):
                values = summary[name].to_numpy().astype(np.float32)
                history[p, v, vi[first_day]] = values[first_day]
                keep = (j >= 1) & (j < FWD_WIDTH)
                forward[p, v, run[keep], j[keep]] = values[keep]
        before = np.arange(size) < fit_until_idx
        stack = grid_a[:, :, before]
        self.mean = torch.from_numpy(np.nanmean(stack, axis=(0, 2))).to(device)[None, :, None]
        self.std = torch.from_numpy(np.maximum(np.nanstd(stack, axis=(0, 2)), 1e-6)).to(device)[
            None, :, None
        ]
        self.grid_a = torch.from_numpy(grid_a).to(device)
        self.grid_b = torch.from_numpy(grid_b).to(device)
        self.history = torch.from_numpy(history).to(device)
        self.forward = torch.from_numpy(forward).to(device)
        self.size = size
        self.var = torch.arange(n_v, device=device)[None, :, None]

    def window(
        self, *, kind: str, primary: torch.Tensor, origin: torch.Tensor, valid: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:  # fmt: skip
        """Return standardised windows (batch, 7, 48); NaN positions become 0 (the training mean)."""
        pos = (valid[:, None] + offsets[None, :]).clamp(0, self.size - 1)
        p3, v3, pos3 = primary[:, None, None], self.var, pos[:, None, :]
        if kind == "era5":
            cut = (origin - win.ORIGIN_OFFSET + win.EARLY_HORIZON)[:, None, None]
            raw = torch.where(pos3 <= cut, self.grid_a[p3, v3, pos3], self.grid_b[p3, v3, pos3])
        else:
            run = ((origin - FIRST_RUN_ORIGIN) // win.HALF_HOURS_PER_DAY).clamp(0, N_RUNS - 1)
            j = pos - origin[:, None]
            after = (j >= 1) & (j < FWD_WIDTH)
            fwd = self.forward[p3, v3, run[:, None, None], j.clamp(0, FWD_WIDTH - 1)[:, None, :]]
            raw = torch.where(after[:, None, :], fwd, self.history[p3, v3, pos3])
        scaled = (raw - self.mean) / self.std
        return torch.nan_to_num(scaled, nan=0.0)


class Stage:
    """One stage's window source: row table positions to windows, real or shuffled-day."""

    def __init__(
        self, *, store: WindowStore, kind: str, primary: np.ndarray, origin: np.ndarray,
        valid: np.ndarray, offsets: np.ndarray, device: str,
    ) -> None:  # fmt: skip
        """Hold the row tables on the GPU."""
        self.store, self.kind = store, kind
        self.primary = torch.from_numpy(primary.astype(np.int64)).to(device)
        self.origin = torch.from_numpy(origin.astype(np.int64)).to(device)
        self.valid = torch.from_numpy(valid.astype(np.int64)).to(device)
        self.offsets = torch.from_numpy(offsets.astype(np.int64)).to(device)

    def __call__(self, rid: torch.Tensor) -> torch.Tensor:
        """Return the windows of rows `rid`."""
        return self.store.window(
            kind=self.kind, primary=self.primary[rid], origin=self.origin[rid],
            valid=self.valid[rid], offsets=self.offsets,
        )  # fmt: skip


class WindowedNetwork(torch.nn.Module):
    """The lag-free network with a window encoder; the last input column is the row's table position."""

    def __init__(self, *, n_base: int, n_primaries: int) -> None:
        """Build the encoder and the mixture network for `n_base` columns plus the encoding."""
        super().__init__()
        self.encoder = cnn.WindowEncoder()
        self.mdn = c.new_model(
            outlier=False, n_inputs=n_base + cnn.OUT_DIM, n_primaries=n_primaries
        )
        self.source: Stage | None = None

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the raw mixture outputs for rows whose last column is their table position."""
        assert self.source is not None
        rid = x[:, -1].round().long()
        encoded = self.encoder(self.source(rid))
        return self.mdn(torch.cat([x[:, :-1], encoded], dim=-1), primary)


def with_id(*, x: np.ndarray) -> np.ndarray:
    """Append the row's table position as the last column (exact in float32 below 2^24 rows)."""
    if len(x) >= 1 << 24:
        msg = "too many rows for a float32 index"
        raise ValueError(msg)
    return np.concatenate([x, np.arange(len(x), dtype=np.float32)[:, None]], axis=1)


def pretrained_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's pretrained weights live."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def prediction_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's forecasts live."""
    return W / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def donor_rows(
    *, windows: fit515.Windows, kind: str, primary: np.ndarray, origin: np.ndarray,
    valid: np.ndarray, store: WindowStore,
) -> tuple[np.ndarray, np.ndarray, int]:  # fmt: skip
    """Return the donor day's origin and valid index per row (experiment 515's placebo rule).

    A row takes the first of the three draws whose donor run exists (ensemble rows) or whose shifted
    row lies inside the grid (ERA5 rows). A row with no donor keeps its own times and is counted;
    the scorer's tables show the count.
    """
    o_out, v_out = origin.copy(), valid.copy()
    todo = np.ones(len(primary), dtype=bool)
    for attempt in range(len(fit515.placebo.SEEDS)):
        shift = windows.days[kind].shifts(valid_idx=valid, attempt=attempt)
        s = np.where(np.isfinite(shift), shift, 0).astype(np.int64)
        o2, v2 = origin + s, valid + s
        ok = todo & np.isfinite(shift) & (o2 >= 0) & (v2 < store.size)
        if kind == "ens":
            run = (o2 - FIRST_RUN_ORIGIN) // win.HALF_HOURS_PER_DAY
            inside = ok & (run >= 0) & (run < N_RUNS)
            have = np.zeros(len(primary), dtype=bool)
            have[inside] = store.has_run[primary[inside], run[inside]]
            ok = inside & have
        o_out[ok], v_out[ok] = o2[ok], v2[ok]
        todo &= ~ok
    return o_out, v_out, int(todo.sum())


class Fold:
    """One fold's rows (experiment 515's `Fold` with no window columns) and window stages per arm kind."""

    def __init__(self, *, month: str, clean: np.ndarray, need_pre: bool, device: str) -> None:
        """Build rows, the window store, and the real and placebo stages."""
        self.month = month
        self.base = fit515.Fold(
            month=month, clean=clean, need_pre=need_pre, arms=["W0"], seam=False
        )
        fit_until = int(
            (c.us(t=c.month_start(month=month)) - c.prep422.GRID_START_US) // c.prep422.STEP_US
        )
        self.store = WindowStore(device=device, fit_until_idx=fit_until)
        self.device = device
        self.stages: dict[tuple[str, str, bool], Stage] = {}
        self.no_donor = 0
        tables = {"ft": (self.base.rows, "ens"), "test": (self.base.test, "ens")}
        if self.base.pre is not None:
            tables["pre"] = (self.base.pre, "era5")
        self.tables = tables
        for variant, offsets in OFFSETS.items():
            for stage, (rows, kind) in tables.items():
                self.stages[(stage, variant, False)] = Stage(
                    store=self.store, kind=kind, primary=rows.primary, origin=rows.origin_idx,
                    valid=rows.target_idx, offsets=offsets, device=device,
                )  # fmt: skip
        for stage, (rows, kind) in tables.items():
            o2, v2, missing = donor_rows(
                windows=self.base.windows, kind=kind, primary=rows.primary,
                origin=rows.origin_idx, valid=rows.target_idx, store=self.store,
            )  # fmt: skip
            self.no_donor += missing
            self.stages[(stage, "centred", True)] = Stage(
                store=self.store, kind=kind, primary=rows.primary, origin=o2, valid=v2,
                offsets=OFFSETS["centred"], device=device,
            )  # fmt: skip

    def stage(self, *, arm: str, stage: str) -> Stage:
        """Return the window stage of an arm."""
        return self.stages[(stage, ARMS[arm], arm in NULL_ARMS)]


def _tensors(*, x: np.ndarray, y: np.ndarray, w: np.ndarray, primary: np.ndarray, is_val: np.ndarray, rows: data.Rows, device: str) -> dict[str, torch.Tensor]:  # fmt: skip
    return fit515._tensor_set(x=x, y=y, w=w, primary=primary, is_val=is_val, rows=rows, device=device)  # fmt: skip


def pretrain_one(*, fold: Fold, arm: str, seed: int, device: str) -> str:
    """Pretrain an arm on the fold's ERA5-and-CAMS rows and save the weights."""
    started = time.monotonic()
    pre = fold.base.pre
    assert pre is not None
    rows = fold.base.rows
    x = with_id(x=pre.x)
    t = _tensors(x=x, y=pre.y, w=np.ones(len(pre.y)), primary=pre.primary, is_val=pre.is_val, rows=rows, device=device)  # fmt: skip
    torch.manual_seed(seed)
    model = WindowedNetwork(n_base=pre.x.shape[1], n_primaries=len(rows.target_mean)).to(device)
    model.source = fold.stage(arm=arm, stage="pre")
    epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.PRETRAIN_LR, outlier=False, prior=False, device=device)  # fmt: skip
    r448.atomic_save(state=model.state_dict(), target=pretrained_path(arm=arm, month=fold.month, seed=seed))  # fmt: skip
    del t, model, x
    torch.cuda.empty_cache()
    return f"  pretrain {arm} {fold.month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s"  # fmt: skip


def fine_tune_one(*, fold: Fold, arm: str, seed: int, device: str) -> str:
    """Fine-tune an arm from its pretrained weights and write its forecasts."""
    started = time.monotonic()
    rows, test = fold.base.rows, fold.base.test
    x = with_id(x=rows.x)
    t = _tensors(x=x, y=rows.y, w=rows.weight, primary=rows.primary, is_val=rows.is_val, rows=rows, device=device)  # fmt: skip
    model = WindowedNetwork(n_base=rows.x.shape[1], n_primaries=len(rows.target_mean)).to(device)
    model.load_state_dict(torch.load(pretrained_path(arm=arm, month=fold.month, seed=seed), map_location="cpu"))  # fmt: skip
    model.source = fold.stage(arm=arm, stage="ft")
    epochs, best = c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device)  # fmt: skip
    model.source = fold.stage(arm=arm, stage="test")
    scored = data.TestRows(x=with_id(x=test.x), primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx, lead=test.lead, origin=test.origin, valid=test.valid)  # fmt: skip
    frame = c.forecast_frame(model=model, outlier=False, test=scored, select=np.ones(len(test.x), dtype=bool), rows=rows, widen=1.0, device=device)  # fmt: skip
    out = prediction_path(arm=arm, month=fold.month, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    del t, model
    torch.cuda.empty_cache()
    return f"  fit {arm} {fold.month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(rows.y):,} rows"  # fmt: skip


def run(*, months: list[str], seeds: list[int], arms: list[str], do_pretrain: bool, do_fit: bool, device: str) -> None:  # fmt: skip
    """Pretrain and/or fine-tune each arm, fold, and seed (skips outputs that exist)."""
    clean = c.clean_stack()
    for month in months:
        todo_pre = [(a, s) for a in arms for s in seeds if do_pretrain and not pretrained_path(arm=a, month=month, seed=s).exists()]  # fmt: skip
        todo_fit = [(a, s) for a in arms for s in seeds if do_fit and not prediction_path(arm=a, month=month, seed=s).exists()]  # fmt: skip
        if not todo_pre and not todo_fit:
            continue
        t0 = time.monotonic()
        fold = Fold(month=month, clean=clean, need_pre=bool(todo_pre), device=device)
        print(f"{month}: feeder {time.monotonic() - t0:.0f} s; placebo rows without a donor {fold.no_donor}", flush=True)  # fmt: skip
        for arm, seed in todo_pre:
            print(pretrain_one(fold=fold, arm=arm, seed=seed, device=device), flush=True)
        for arm, seed in todo_fit:
            print(fine_tune_one(fold=fold, arm=arm, seed=seed, device=device), flush=True)
        del fold
        torch.cuda.empty_cache()


def checks(*, month: str, device: str) -> None:
    """Print window sanity checks on a fold: no NaN share, scale, and ERA5 against ensemble windows."""
    clean = c.clean_stack()
    fold = Fold(month=month, clean=clean, need_pre=True, device=device)
    print(f"{month}: placebo rows without a donor {fold.no_donor}")
    for stage in ("pre", "ft", "test"):
        for arm in ARMS:
            s = fold.stage(arm=arm, stage=stage)
            rid = torch.randint(0, len(s.primary), (20_000,), device=device)
            w = s(rid)
            print(f"  {arm} {stage}: mean {w.mean():+.3f}, sd {w.std():.3f}, exactly 0 {float((w == 0).float().mean()):.4f}")  # fmt: skip
    # the same (primary, origin, valid) from ERA5 and from the ensemble: the gap is forecast error
    s_ft = fold.stage(arm="C-CNN", stage="ft")
    rid = torch.randint(0, len(s_ft.primary), (20_000,), device=device)
    ens = s_ft(rid)
    era = fold.store.window(kind="era5", primary=s_ft.primary[rid], origin=s_ft.origin[rid], valid=s_ft.valid[rid], offsets=s_ft.offsets)  # fmt: skip
    print("  ERA5 minus ensemble window, standardised, per variable (mean, sd):")
    for v, name in enumerate(prep418.WEATHER):
        d = era[:, v] - ens[:, v]
        print(f"    {name}: {d.mean():+.3f} {d.std():.3f}")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("time", "pretrain", "fit", "both", "checks"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "checks":
        checks(month=args.months[0], device=args.device)
    elif args.command == "time":
        run(months=args.months[:1], seeds=args.seeds[:1], arms=args.arms[:1], do_pretrain=True, do_fit=True, device=args.device)  # fmt: skip
    else:
        run(months=args.months, seeds=args.seeds, arms=args.arms, do_pretrain=args.command in ("pretrain", "both"), do_fit=args.command in ("fit", "both"), device=args.device)  # fmt: skip


if __name__ == "__main__":
    main()
