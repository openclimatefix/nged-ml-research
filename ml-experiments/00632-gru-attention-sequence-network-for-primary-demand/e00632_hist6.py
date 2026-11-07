# ruff: noqa: E501, D103, E402, PLR0915, E731, PT018
"""Experiment 632: arm P2-hist6, P2's per-row network with a lag encoder that reads 6 daily lags.

**What differs from P2-pre.** P2-pre (experiment 591's `AuxLagMdn(kind="shared")`, pre-training stage,
D5 output offset, B1 columns) reads two lags: the newest same-half-hour reading at least 24 h before
the origin, and the newest same-weekday one. P2-hist6 keeps the weather encoder, datetime encoder,
Primary embedding, D5 offset, B1 columns, and mixer widths of that network, and replaces the lag
encoder's input by 6 same-half-hour readings: the newest reading at least 24 h before the origin
(the same reading P2-pre's day lag is) and the 5 readings 24 h, 48 h, ..., 120 h before it. All 6 are
within the 7 days before the origin (asserted). The shared encoder reads one lag at a time (value,
missing flag, age, the expected-weather embeddings at that lag's own time, the Primary embedding, and
an 8-number slot embedding), and the activated outputs of the available lags are mean-pooled.

**Training.** The P2 pre-training recipe: experiment 574's `pretrain_wt` rows
(`e00448_run.pretrain_rows`, ERA5 and CAMS pseudo-runs, valid time at least the refit cutoff age
before the refit, purged and held-out weeks), the stored fold statistics, experiment 574's `train`
(AdamW, learning rate `PRETRAIN_LR`, batch 8,192, early stopping on held-out weeks with patience 8,
at most 60 epochs), and experiment 558's block dropout on whole origin days.

**Test rows** are `e00632_fold.test_rows` (09:00 UTC origins, leads to 168 h); the predictions have
the keys and columns of `predictions/P2-pre/`.

Usage: python e00632_hist6.py MONTH [MONTH ...] [--seeds 0 1 2]
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from typing import Final

_parser = argparse.ArgumentParser()
_parser.add_argument("months", nargs="+")
_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
MINE = _parser.parse_args()
# experiment 574's module reads its own command line when imported (as experiments 591 and 613 do)
sys.argv = ["e00574_fit.py", "pretrain-wt", "--arm", "P5", "--months", *MINE.months]

import e00397_models as base  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00545_model as m545  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00559_model as m  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import e00569_columns as cols569  # ty: ignore[unresolved-import]
import e00572_lags as lg  # ty: ignore[unresolved-import]
import e00574_fit as e574  # ty: ignore[unresolved-import]
import e00591_model as aux591  # ty: ignore[unresolved-import]
import e00632_data as d
import e00632_fold as fold_mod
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00397_prep import INPUT_CLIP_SD, primaries  # ty: ignore[unresolved-import]
from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

ARM: Final[str] = "P2-hist6"
N_LAGS: Final[int] = 6
"""Same-half-hour readings per row: the newest one at least 24 h before the origin, and 5 older."""
SEVEN_DAYS_STEPS: Final[int] = 7 * lg.STEPS
N_B1: Final[int] = fold_mod.N_B1
TAU_COL: Final[int] = 37
B1_COLS: Final[slice] = slice(38, 38 + N_B1)
LAG_FIRST: Final[int] = 38 + N_B1
"""Lag columns: 6 values, 6 missing flags, 6 ages, then the grid index of the newest reading."""
N_INPUTS: Final[int] = LAG_FIRST + 3 * N_LAGS + 1
VALUE_COLS: Final[slice] = slice(LAG_FIRST, LAG_FIRST + N_LAGS)
MISSING_COLS: Final[slice] = slice(LAG_FIRST + N_LAGS, LAG_FIRST + 2 * N_LAGS)
AGE_COLS: Final[slice] = slice(LAG_FIRST + 2 * N_LAGS, LAG_FIRST + 3 * N_LAGS)
NEWEST_INDEX_COL: Final[int] = LAG_FIRST + 3 * N_LAGS
SLOT_DIM: Final[int] = aux591.LAG_ID_DIM
BYTES_PER_GPU_GB: Final[float] = 8 / 48


def lag_grid_indices(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, cap_steps: int = lg.CAP_24H
) -> np.ndarray:
    """Return the (rows, 6) grid indices of the 6 lags: newest first, each 24 h older than the last.

    The newest is the latest reading of the target's half-hour at least `cap_steps` before the
    origin. All 6 are asserted to lie between 7 days and `cap_steps` half-hours before the origin.
    """
    newest, _ = lg.lag_indices(target_idx=target_idx, origin_idx=origin_idx, cap_steps=cap_steps)
    idx = newest[:, None] - lg.STEPS * np.arange(N_LAGS)[None, :]
    origin = np.asarray(origin_idx)[:, None]
    assert (idx <= origin - cap_steps).all(), "a lag is younger than the cap"
    assert (idx >= origin - SEVEN_DAYS_STEPS).all(), "a lag is older than 7 days"
    return idx


def hist6_block(
    *, clean: np.ndarray, primary: np.ndarray, target_idx: np.ndarray, origin_idx: np.ndarray,
    stats: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Return the 19 lag columns: 6 standardised values, 6 missing flags, 6 ages, newest grid index.

    Values are cleaned power over capacity (flagged half-hours are NaN, so they are missing) and are
    standardised with the fold's day-lag mean and standard deviation, as P2-pre's day lag is. A
    missing lag has value 0, flag 1, and the missing-age code of P2-pre's day lag.
    """
    idx = lag_grid_indices(target_idx=target_idx, origin_idx=origin_idx)
    out = np.zeros((len(primary), 3 * N_LAGS + 1), dtype=np.float32)
    amax = lg.age_max(cap_steps=lg.CAP_24H)[0]
    for j in range(N_LAGS):
        raw = np.where(idx[:, j] >= 0, clean[primary, np.clip(idx[:, j], 0, None)], np.nan)
        miss = ~np.isfinite(raw)
        z = np.clip((raw - stats[0, 0]) / stats[0, 1], -INPUT_CLIP_SD, INPUT_CLIP_SD)
        age = (target_idx - idx[:, j]) * 0.5 / lags.LEAD_SCALE_HOURS
        out[:, j] = np.where(miss, 0.0, z)
        out[:, N_LAGS + j] = miss
        out[:, 2 * N_LAGS + j] = np.where(miss, amax, age)
    out[:, 3 * N_LAGS] = idx[:, 0]
    return out


class Hist6Mdn(aux591.AuxLagMdn):
    """P2's network (stage `pre`) whose lag encoder reads 6 lags and mean-pools them."""

    def __init__(self, *, n_primaries: int, n_knots: int) -> None:
        """Build experiment 591's network, then add a 6-slot embedding and narrow the mixers by one lag."""
        super().__init__(
            n_primaries=n_primaries, stage="pre", n_knots=n_knots, n_b1=N_B1, kind="shared"
        )
        self.lag_id = nn.Embedding(N_LAGS, SLOT_DIM)
        wide = 2 * m545.ENCODER_OUT + aux591.m560.LAG_OUT + m545.EMBEDDING
        self.mixer_pre = base.mlp(
            n_in=wide, hidden=m545.MIXER_HIDDEN, layers=2, n_out=3 * m545.COMPONENTS,
            dropout=m545.DROPOUT,
        )  # fmt: skip
        base.init_head(head=self.mixer_pre, components=m.COMPONENTS)
        del self.mixer_ft, self.head

    def pooled_lag_embedding(
        self, *, x: torch.Tensor, primary: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        """Return the mean over the available lags of the encoder's activated outputs.

        A row with no available lag takes the mean over all 6 (each carries the missing flag).
        """
        assert self.table is not None, "attach the expected-input table first"
        newest = x[:, NEWEST_INDEX_COL].long()
        total = torch.zeros(len(x), aux591.m560.LAG_OUT, device=x.device)
        weight = torch.zeros(len(x), 1, device=x.device)
        available = x[:, MISSING_COLS] < 0.5
        keep_all = ~available.any(dim=1, keepdim=True)
        for j in range(N_LAGS):
            idx = newest - lg.STEPS * j
            row = self.table[primary, idx.clamp(min=0)].float()
            row = torch.where((idx >= 0)[:, None], row, torch.zeros_like(row))
            hw = nn.functional.gelu(
                self.weather(
                    torch.cat([row[:, :7], emb, torch.zeros(len(x), N_B1, device=x.device)], dim=-1)
                )
            )
            hd = nn.functional.gelu(self.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
            own = torch.stack(
                [
                    x[:, LAG_FIRST + j],
                    x[:, LAG_FIRST + N_LAGS + j],
                    x[:, LAG_FIRST + 2 * N_LAGS + j],
                ],
                dim=-1,
            )
            slot = self.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))
            out = nn.functional.gelu(self.lag[0](torch.cat([own, hw, hd, emb, slot], dim=-1)))
            w = (available[:, j : j + 1] | keep_all).float()
            total = total + w * out
            weight = weight + w
        return total / weight

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the 9 raw outputs, with the D5 offset added to the component means."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(
            self.weather(torch.cat([x[:, m.WEATHER_COLS], emb, x[:, B1_COLS]], dim=-1))
        )
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        hl = self.pooled_lag_embedding(x=x, primary=primary, emb=emb)
        out = self.mixer_pre(torch.cat([hw, hd, hl, emb], dim=-1))
        off = self.offset(x[:, TAU_COL], primary)
        k = m.COMPONENTS
        return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


def n_knots_of(*, month: str) -> int:
    """Return the number of D5 knots of the fold, as experiment 632's Seq counts them."""
    refit = fold_mod.month_start(month=month)
    return int((refit.year - 2019) * 12 + refit.month - 9) + 1


def apply_block_dropout(*, x: torch.Tensor, origin_idx: np.ndarray, seed: int) -> dict:
    """Mask lags in place per experiment 558: whole origin days, all 6 lags or only the newest."""
    mask_all, mask_newest = lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)
    ma = torch.from_numpy(mask_all).to(x.device)
    mn = torch.from_numpy(mask_newest).to(x.device)
    amax = float(lg.age_max(cap_steps=lg.CAP_24H)[0])
    for rows, j_range in ((ma, range(N_LAGS)), (mn, range(1))):
        for j in j_range:
            x[rows, LAG_FIRST + j] = 0.0
            x[rows, LAG_FIRST + N_LAGS + j] = 1.0
            x[rows, LAG_FIRST + 2 * N_LAGS + j] = amax
    return {
        "share_all_lags_masked": float(mask_all.mean()),
        "share_newest_lag_only_masked": float(mask_newest.mean()),
        "share_newest_lag_missing": float((x[:, LAG_FIRST + N_LAGS] > 0.5).float().mean()),
    }


def assert_training_lags_not_leaky(
    *, blk: np.ndarray, origin_idx: np.ndarray, n: int = 2_000_000
) -> None:
    """Assert for training rows that every available lag is at least 24 h and at most 7 days before its origin."""
    newest = blk[:n, NEWEST_INDEX_COL - LAG_FIRST].astype(np.int64)
    origin = origin_idx[:n]
    for j in range(N_LAGS):
        idx = newest - lg.STEPS * j
        assert (idx <= origin - lg.CAP_24H).all() and (idx >= origin - SEVEN_DAYS_STEPS).all()


def pretrain_month(*, month: str, seeds: list[int], clean: np.ndarray, device: str) -> list[dict]:
    """Pre-train P2-hist6 for the month's fold on every seed; return one log record per seed."""
    stats = fold_mod.fold_stats(month=month, clean=clean)
    refit = fold_mod.month_start(month=month)
    rows = e574.data.build_fold(refit=refit, clean=clean)
    pre = r448.pretrain_rows(rows=rows, refit=refit, clean=clean)
    assert pre.x.shape[1] == 37
    latest = datetime.fromtimestamp((pre.target_idx.max() * STEP_US + GRID_START_US) / 1e6, UTC)
    assert latest <= refit - e574.c.regimes.REFIT_CUTOFF_AGE, (latest, refit)
    n_rows = len(pre.x)
    print(f"{month}: pretraining rows {n_rows:,}; latest valid {latest.isoformat()}", flush=True)
    blk = np.concatenate(
        [
            hist6_block(
                clean=clean, primary=pre.primary[a : a + 2_000_000],
                target_idx=pre.target_idx[a : a + 2_000_000],
                origin_idx=pre.origin_idx[a : a + 2_000_000], stats=stats.lag_stats,
            )
            for a in range(0, n_rows, 2_000_000)
        ]
    )  # fmt: skip
    assert_training_lags_not_leaky(blk=blk, origin_idx=pre.origin_idx)
    ex = cols569.Extras(arm="B1", proximity={})
    stored = np.load(fold_mod.P2_DIR / "stats" / "B1" / f"{month}.npz")
    ex.mean, ex.std = stored["mean"], stored["std"]
    x_gpu = torch.empty((n_rows, N_INPUTS), dtype=torch.float32, device=device)
    for a in range(0, n_rows, 500_000):
        sl = slice(a, a + 500_000)
        x_gpu[sl, :37] = torch.from_numpy(pre.x[sl]).to(device)
        raw_b1 = ex.raw_pretrain(
            primary=pre.primary[sl], origin_idx=pre.origin_idx[sl], target_idx=pre.target_idx[sl],
            shift=None,
        )  # fmt: skip
        b1 = ex.scale(raw=raw_b1)
        assert b1.shape[1] == N_B1 and np.isfinite(b1).all()
        x_gpu[sl, B1_COLS] = torch.from_numpy(b1).to(device)
    tau = e574.extra.DriftColumns.tau(valid_us=GRID_START_US + pre.target_idx * STEP_US)
    x_gpu[:, TAU_COL] = torch.from_numpy(tau.astype(np.float32)).to(device)
    cols, _ = d560.raw_table(source="pre", cache=e574.W560 / "expected")
    table = torch.from_numpy(d560.standardised_table(cols=cols, standardiser=rows.standardiser)).to(
        device
    )
    del cols
    z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
    t_z = torch.from_numpy(z.astype(np.float32)).to(device)
    t_w = torch.ones(n_rows, device=device)
    t_p = torch.from_numpy(pre.primary).to(device)
    t_v = torch.from_numpy(pre.is_val).to(device)
    origin_idx = pre.origin_idx
    n_val = int(pre.is_val.sum())
    n_primaries = len(rows.target_mean)
    del pre, rows
    records = []
    for seed in seeds:
        target = d.ROOT / "models" / ARM / f"{month}_s{seed}.pt"
        started = time.monotonic()
        x_gpu[:, LAG_FIRST:] = torch.from_numpy(blk).to(device)
        share = apply_block_dropout(x=x_gpu, origin_idx=origin_idx, seed=seed)
        torch.manual_seed(seed)
        model = Hist6Mdn(n_primaries=n_primaries, n_knots=n_knots_of(month=month)).to(device)
        model.table = table
        epochs, best = e574.train(
            model=model, x=x_gpu, z=t_z, w=t_w, p=t_p, is_val=t_v, seed=seed,
            lr=c510.PRETRAIN_LR, max_epochs=c510.MAX_EPOCHS, fresh_lr=None, frozen=False,
            anchor=None, lam=0.0, device=device,
        )  # fmt: skip
        fit_seconds = time.monotonic() - started
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target)
        records.append(
            {
                "arm": ARM, "month": month, "seed": seed, "epochs": epochs, "best_val_nll": best,
                "fit_seconds": fit_seconds, "train_sequences": n_rows - n_val,
                "val_sequences": n_val, "dropout": share,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )  # fmt: skip
        print(f"  {ARM} {month} s{seed}: {records[-1]}", flush=True)
        predict_month(model=model, month=month, seed=seed, clean=clean, table=table, device=device)
        model.to("cpu")
        del model
    del x_gpu, t_z, t_w, t_p, t_v, table
    torch.cuda.empty_cache()
    return records


def predict_month(
    *, model: Hist6Mdn, month: str, seed: int, clean: np.ndarray, table: torch.Tensor, device: str
) -> None:  # fmt: skip
    """Write `predictions/P2-hist6/<month>_s<seed>.parquet` with P2-pre's keys and columns.

    The test table is the dense ERA5 and CAMS array standardised with the fold's statistics, as for
    the P2-pre comparator; `table` (the pseudo-run table of pre-training) is not used.
    """
    del table
    raw = d.dense_raw()
    cap = fold_mod.capacity_of_primary()
    stats = fold_mod.fold_stats(month=month, clean=clean)
    rows = fold_mod.test_rows(raw=raw, clean=clean, s=stats, month=month, device=device)
    b1 = fold_mod.b1_columns(month=month, rows=rows)
    blk = hist6_block(
        clean=clean, primary=rows["primary"], target_idx=rows["target_idx"],
        origin_idx=rows["origin_idx"], stats=stats.lag_stats,
    )  # fmt: skip
    assert (blk[:, NEWEST_INDEX_COL - LAG_FIRST] <= rows["origin_idx"] - lg.CAP_24H).all()
    x = np.concatenate([rows["x37"], rows["tau"][:, None], b1, blk], axis=1).astype(np.float32)
    assert x.shape[1] == N_INPUTS
    test_table = d.standardised_inputs(raw=raw, fold=fold_mod.fold_for_seq(s=stats), device=device)
    model.table = test_table
    model.eval()
    out_q = np.empty((len(x), 11), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(x), 100_000):
            sl = slice(a, a + 100_000)
            out = model(
                torch.from_numpy(x[sl]).to(device), torch.from_numpy(rows["primary"][sl]).to(device)
            )
            out_q[sl] = fold_mod.quantiles_mw(
                out=out, primary=rows["primary"][sl], s=stats, capacity=cap
            )
    to_time = lambda idx: pl.from_epoch(
        pl.Series(GRID_START_US + idx * STEP_US), time_unit="us"
    ).dt.replace_time_zone("UTC")
    ids = primaries().sort("primary")["time_series_id"].to_numpy()[rows["primary"]]
    frame = pl.DataFrame(
        {
            "time_series_id": ids,
            "power_fcst_init_time": to_time(rows["origin_idx"]),
            "valid_time": to_time(rows["target_idx"]),
            "lead_hours": rows["lead"],
            **{f"q{round(t * 100):02d}": out_q[:, i] for i, t in enumerate(base.LEVELS)},
        }
    )  # fmt: skip
    target = d.ROOT / "predictions" / ARM / f"{month}_s{seed}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)
    print(f"{ARM} {month} s{seed}: {frame.height} rows", flush=True)
    model.table = None
    del test_table
    torch.cuda.empty_cache()


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(BYTES_PER_GPU_GB)
    clean = d.clean_power()
    log = d.ROOT / "logs" / f"{ARM}.jsonl"
    log.parent.mkdir(exist_ok=True)
    for month in MINE.months:
        seeds = [
            s
            for s in MINE.seeds
            if not (d.ROOT / "predictions" / ARM / f"{month}_s{s}.parquet").exists()
        ]
        if not seeds:
            continue
        for record in pretrain_month(month=month, seeds=seeds, clean=clean, device="cuda"):
            with log.open("a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
