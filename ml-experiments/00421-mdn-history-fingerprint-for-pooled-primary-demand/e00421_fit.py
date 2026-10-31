"""Fit, forecast, and probe the history-fingerprint arms of experiment 421.

Every arm is experiment 397's pooled plain network (setting `g09`, summary scheme) on its five
screen folds, monthly refits, with the D15 and D18 start cut of experiment 399 and one global
target scale. The arms differ in two switches (the Primary embedding, the 7-day history encoder)
and one input choice (the three origin lags):

- `E3g` embedding, three origin lags; `N3g` no embedding, three origin lags;
- `NH` no embedding, three origin lags, encoder; `EH` embedding, three origin lags, encoder;
- `NH0` no embedding, encoder, no origin lags; `NHs` as NH but each Primary reads the history of
  the next Primary (index plus one, modulo 16) on the same run day.

`--held-out-group g` leaves out of training every Primary whose index has remainder `g` modulo 4
and scores only those Primaries. An arm with an embedding then receives the mean embedding of the
Primaries it trained on for each left-out Primary.

Commands: `check` (history alignment, run before any fit), `history` (write the history table),
`fit` (fit, forecast, probe). The fit stops cleanly when `MemAvailable` falls to 12 GB or below.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00406-pooled-mixture-density-network-without-embedding-and-with-short-history:\
ml-experiments/00421-mdn-history-fingerprint-for-pooled-primary-demand \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00421-*/e00421_fit.py fit --arms NH --device cuda
```
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00406_fit as fit406  # ty: ignore[unresolved-import]
import e00421_models as models  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SETTING: Final[fit.Setting] = fit.setting_by_name(name="g09")
SCHEME: Final[str] = "summary"
N_GROUPS: Final[int] = 4
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "signed": ("D5", "D9", "D10", "D11", "D14", "D16"),
    "mva": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    "short": ("D15", "D18"),
}
"""The metering groups of the demand-v2 protocol, by site label."""
MIN_AVAILABLE_GB: Final[float] = 12.0
"""The stop line: the `available` column of `free -g`."""
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00421-mdn-history-fingerprint-for-pooled-primary-demand" / "work"
)
HISTORY_PATH: Final[Path] = OUT_DIR / "history_table.npy"
N_RUN_DAYS: Final[int] = (prep.LAST_RUN - prep.FIRST_RUN).days + 1


@dataclass(frozen=True)
class Arm:
    """One arm: which switches are on."""

    name: str
    embedding: bool
    history: bool
    lags: bool
    shuffled: bool = False


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("E3g", embedding=True, history=False, lags=True),
        Arm("N3g", embedding=False, history=False, lags=True),
        Arm("NH", embedding=False, history=True, lags=True),
        Arm("NHs", embedding=False, history=True, lags=True, shuffled=True),
        Arm("EH", embedding=True, history=True, lags=True),
        Arm("NH0", embedding=False, history=True, lags=False),
    )
}
"""The six new arms; experiment 406's `pooled_emb` is the stored E406."""


def available_gb() -> float:
    """Return `MemAvailable` from `/proc/meminfo` in GB (the `available` column of `free -g`)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    msg = "MemAvailable not found in /proc/meminfo"
    raise RuntimeError(msg)


# --- the history table -----------------------------------------------------------------------


def build_history_table() -> np.ndarray:
    """Return power over capacity for the 336 half-hours up to each run's newest usable reading.

    Shape (Primaries, run days, 336), oldest half-hour first, NaN where unobserved or flagged.
    For D15 and D18 every half-hour before the 2025-03-01 start cut is NaN. The newest half-hour is
    grid index `origin - NEWEST_AGE`, exactly the reading `lag_origin` uses.
    """
    series = prep.primaries()
    arrays = common.load_power_arrays(series=series)
    origins = pl.Series(
        [prep.FIRST_RUN + timedelta(days=d) + common.PUBLICATION_DELAY for d in range(N_RUN_DAYS)]
    ).dt.cast_time_unit("us")
    origins = origins.dt.replace_time_zone("UTC") if origins.dtype == pl.Datetime("us") else origins
    newest = np.asarray(common.grid_index(times=origins)) - common.NEWEST_AGE
    index = newest[:, None] - np.arange(models.HISTORY_STEPS - 1, -1, -1)[None, :]
    if index.min() < 0:
        msg = "a history window starts before the grid"
        raise ValueError(msg)
    cut_index = common.grid_index(times=fit406.CUT)
    rows = []
    for row in series.iter_rows(named=True):
        window = arrays[row["site"]].power[index].astype(np.float32)
        if row["site"] in fit406.SHORT_HISTORY:
            window[index < cut_index] = np.nan
        rows.append(window)
    return np.stack(rows)


def history_table() -> np.ndarray:
    """Return the cached history table, writing it first if it does not exist."""
    if not HISTORY_PATH.exists():
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = HISTORY_PATH.with_name("history_table.tmp.npy")
        np.save(temporary, build_history_table())
        temporary.rename(HISTORY_PATH)
    return np.load(HISTORY_PATH)


def history_inputs(*, table: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """Return the encoder's inputs (Primaries, run days, 2, 336): standardised value and flag."""
    missing = np.isnan(table)
    value = np.clip((table - mean) / std, -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
    value = np.where(missing, 0.0, value)
    return torch.from_numpy(
        np.stack([value, missing.astype(np.float32)], axis=2).astype(np.float32)
    )


# --- the fold --------------------------------------------------------------------------------


@dataclass
class Fold:
    """A fold's training rows (see `prep.FoldTrain`) with the arm's column choice and scale."""

    train: prep.FoldTrain
    x: np.ndarray
    scale_mean: float
    scale_std: float
    seen: np.ndarray
    history: torch.Tensor


def build_fold(*, refit: datetime, arm: Arm, left_out: frozenset[int], table: np.ndarray) -> Fold:
    """Return a fold's training rows with one global target scale.

    Experiment 406's `build_train` supplies the rows and the start cut. Rows of the `left_out`
    Primaries are dropped, and the input standardiser is fitted again on the remaining rows, so
    that no statistic of a left-out Primary reaches the network.
    """
    fold = fit406.build_train(refit=refit, history_months=0, cut_primaries=None)
    seen = np.array([p for p in range(len(fold.target_mean)) if p not in left_out])
    keep = np.isin(fold.primary, seen)
    if left_out:
        # `fold.x` is already standardised with all 16 Primaries; undo and refit on the seen ones.
        std = fold.standardiser
        raw = fold.x[keep] * std.std + std.mean
        train_rows = ~fold.is_val[keep]
        # missing lags were set to 0 after standardisation; restore NaN from the missing flags
        names, mask, layout = prep.feature_names(scheme=SCHEME)
        lag_cols = [names.index(n) for n in ("lag_origin", "lag_day", "lag_week")]
        flag_cols = [names.index(f"{n}_missing") for n in ("lag_origin", "lag_day", "lag_week")]
        for c, f in zip(lag_cols, flag_cols, strict=True):
            raw[raw[:, f] > 0.5, c] = np.nan
        new_std = prep.Standardiser.fit(x=raw[train_rows], mask=mask)
        x_all = new_std.apply(x=raw)
        standardiser = new_std
    else:
        x_all = fold.x
        standardiser = fold.standardiser
    y = fold.y[keep]
    train_rows = ~fold.is_val[keep]
    # The global scale uses the rows that train the network (not the early-stopping weeks).
    mean, std = float(y[train_rows].mean()), max(float(y[train_rows].std()), 1e-3)
    layout = fold.layout
    x = x_all if left_out else x_all[keep]
    if not arm.lags:
        lag_slice = layout.slices["lags"]
        x = np.delete(x, np.arange(lag_slice.start, lag_slice.stop), axis=1)
    sub = prep.FoldTrain(
        x=x,
        y=y,
        primary=fold.primary[keep],
        weight=fold.weight[keep],
        is_val=fold.is_val[keep],
        lead=fold.lead[keep],
        origin_day=fold.origin_day[keep],
        standardiser=standardiser,
        target_mean=np.full(len(fold.target_mean), mean),
        target_std=np.full(len(fold.target_std), std),
        layout=layout,
    )
    return Fold(
        train=sub,
        x=x,
        scale_mean=mean,
        scale_std=std,
        seen=seen,
        history=history_inputs(
            table=np.roll(table, -1, axis=0) if arm.shuffled else table, mean=mean, std=std
        ),
    )


# --- training --------------------------------------------------------------------------------


def make_network(*, arm: Arm, n_inputs: int, n_primaries: int) -> models.HistoryMdn:
    """Return an untrained network for an arm."""
    return models.HistoryMdn(
        n_inputs=n_inputs,
        n_primaries=n_primaries,
        use_embedding=arm.embedding,
        use_history=arm.history,
        hidden=SETTING.hidden,
        layers=SETTING.layers,
        components=SETTING.components,
        dropout=SETTING.dropout,
    )


def _forward(
    *,
    model: models.HistoryMdn,
    x: torch.Tensor,
    primary: torch.Tensor,
    day: torch.Tensor,
    history: torch.Tensor | None,
) -> torch.Tensor:
    """Return the raw outputs, gathering each row's history by (Primary, run day)."""
    h = history[primary, day] if history is not None else None
    return model(x, primary, h)


def train_network(
    *, model: models.HistoryMdn, fold: Fold, arm: Arm, seed: int, device: str
) -> tuple[models.HistoryMdn, int, float, float]:
    """Train as `e00397_fit.train_network` does, plus the history gather; return the best network.

    Args:
        model: The untrained network.
        fold: The fold's training rows.
        arm: The arm, which says whether the network reads the history.
        seed: The seed of the batch order.
        device: The torch device.

    Returns:
        The network, the epochs run, the best validation negative log-likelihood, and the peak GPU
        memory in GB.
    """
    t = fold.train
    z = ((t.y - fold.scale_mean) / fold.scale_std).astype(np.float32)
    x = torch.from_numpy(t.x).to(device)
    zt = torch.from_numpy(z).to(device)
    w = torch.from_numpy(t.weight).to(device)
    primary = torch.from_numpy(t.primary).to(device)
    day = torch.from_numpy(t.origin_day).to(device)
    is_val = torch.from_numpy(t.is_val).to(device)
    history = fold.history.to(device) if arm.history else None
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=SETTING.lr, weight_decay=SETTING.weight_decay
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    k = SETTING.components
    for epoch in range(fit.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), fit.BATCH):
            b = order[start : start + fit.BATCH]
            out = _forward(model=model, x=x[b], primary=primary[b], day=day[b], history=history)
            nll = base.mixture_nll(out=out, z=zt[b], components=k)
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(
                            out=_forward(
                                model=model, x=x[b], primary=primary[b], day=day[b], history=history
                            ),
                            z=zt[b],
                            components=k,
                        )
                        for b in val_idx.split(fit.BATCH * 4)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= fit.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    return model, epochs, best, peak


def set_unseen_embeddings(
    *, model: models.HistoryMdn, seen: np.ndarray, by_group: bool = False
) -> None:
    """Give every Primary outside `seen` the mean embedding of the Primaries in `seen`.

    With `by_group`, the mean is taken over the seen Primaries in the same metering group (the
    signed-MW six, the MVA-only eight, or D15 and D18), and falls back to the all-seen mean if the
    group has no seen Primary.
    """
    if model.embedding is None:
        return
    sites = prep.primaries()["site"].to_list()
    groups = {s: g for g, labels in GROUPS.items() for s in labels}
    with torch.no_grad():
        weight = model.embedding.weight
        overall = weight[torch.as_tensor(seen, device=weight.device)].mean(dim=0)
        unseen = [p for p in range(weight.shape[0]) if p not in set(seen.tolist())]
        for p in unseen:
            same = [int(q) for q in seen if groups[sites[int(q)]] == groups[sites[p]]]
            if by_group and same:
                weight[p] = weight[torch.as_tensor(same, device=weight.device)].mean(dim=0)
            else:
                weight[p] = overall


# --- forecasting -----------------------------------------------------------------------------


def forecast_month(
    *,
    model: models.HistoryMdn,
    fold: Fold,
    arm: Arm,
    month: str,
    score_primaries: set[int],
    device: str,
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for the Primaries in the set."""
    k = SETTING.components
    history = fold.history.to(device) if arm.history else None
    layout = fold.train.layout
    pieces = []
    for origin in prep.test_origins(month=month):
        run_day = (origin - common.PUBLICATION_DELAY - prep.FIRST_RUN).days
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme=SCHEME, standardiser=fold.train.standardiser
        ):
            p = int(row["primary"])
            if p not in score_primaries:
                continue
            if not arm.lags:
                lag_slice = layout.slices["lags"]
                x = np.delete(x, np.arange(lag_slice.start, lag_slice.stop), axis=1)  # noqa: PLW2901
            xb = torch.from_numpy(x).to(device)
            pb = torch.full((len(xb),), p, dtype=torch.int64, device=device)
            db = torch.full((len(xb),), run_day, dtype=torch.int64, device=device)
            with torch.no_grad():
                out = _forward(model=model, x=xb, primary=pb, day=db, history=history)
                log_w, mean, std = base.mixture_parts(out=out.double(), components=k)
                q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std).cpu().numpy()
            q = np.sort(q * fold.scale_std + fold.scale_mean, axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


# --- the probe -------------------------------------------------------------------------------


def _probe_accuracy(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    classes: int,
    month_tolerance: bool = False,
) -> float:
    """Fit a linear softmax classifier on the GPU tensors and return the test accuracy.

    With `month_tolerance`, a prediction within one calendar month (cyclically) counts as correct.
    """
    mu, sd = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    a, b = (x_train - mu) / sd, (x_test - mu) / sd
    torch.manual_seed(0)
    lin = torch.nn.Linear(a.shape[1], classes).to(a.device)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(a), y_train)
        loss.backward()
        opt.step()
    pred = lin(b).argmax(1)
    if month_tolerance:
        gap = (pred - y_test).abs()
        return float(((gap <= 1) | (gap >= classes - 1)).float().mean())
    return float((pred == y_test).float().mean())


def probe(
    *,
    model: models.HistoryMdn,
    fold: Fold,
    arm: Arm,
    refit: datetime,
    device: str,
    table: np.ndarray,
) -> dict[str, float]:
    """Return the probe accuracies of an encoder arm (see the plan's "The probe")."""
    held_day, purged_day = prep.held_out_weeks(origin_day=np.arange(N_RUN_DAYS))
    last_day = (refit - prep.FIRST_RUN).days  # run days before the refit are in the window
    first_day = max(0, last_day - prep.WINDOW_DAYS)
    days = np.arange(first_day, last_day)
    seen = fold.seen
    sites = prep.primaries()["site"].to_list()
    cut_day = (fit406.CUT - prep.FIRST_RUN).days + models.HISTORY_STEPS // 48
    pairs = np.array(
        [
            (p, d)
            for p in seen
            for d in days
            if sites[int(p)] not in fit406.SHORT_HISTORY or d >= cut_day
        ]
    )
    primary_of = pairs[:, 0]
    day_of = pairs[:, 1]
    month = np.array(
        [(prep.FIRST_RUN + timedelta(days=int(d))).month - 1 for d in day_of], dtype=np.int64
    )
    train_mask = ~held_day[day_of] & ~purged_day[day_of]
    test_mask = held_day[day_of]
    primary_label = np.searchsorted(np.sort(seen), primary_of)
    tensors = fold.history.to(device)
    p_t = torch.from_numpy(primary_of).to(device)
    d_t = torch.from_numpy(day_of).to(device)
    with torch.no_grad():
        hist = tensors[p_t, d_t]
        trained = model.encode(hist)
        torch.manual_seed(12345)
        random_encoder = models.HistoryEncoder().to(device)
        random = random_encoder(hist)
    raw = hist.flatten(start_dim=1)
    # Reference (a): the three origin lags at the first valid time of each run, with their flags.
    lags = _first_lead_lags(fold=fold, pairs=pairs, device=device)

    y_p = torch.from_numpy(primary_label).to(device)
    y_m = torch.from_numpy(month).to(device)
    tr, te = torch.from_numpy(train_mask).to(device), torch.from_numpy(test_mask).to(device)
    n_p = len(seen)
    out: dict[str, float] = {"chance_primary": 1.0 / n_p, "chance_month": 1.0 / 12.0}
    for name, feats in (("encoder", trained), ("raw_history", raw), ("random_encoder", random)):
        out[f"primary_{name}"] = _probe_accuracy(
            x_train=feats[tr], y_train=y_p[tr], x_test=feats[te], y_test=y_p[te], classes=n_p
        )
        out[f"month_{name}"] = _probe_accuracy(
            x_train=feats[tr], y_train=y_m[tr], x_test=feats[te], y_test=y_m[te], classes=12
        )
        out[f"month_within_one_{name}"] = _probe_accuracy(
            x_train=feats[tr],
            y_train=y_m[tr],
            x_test=feats[te],
            y_test=y_m[te],
            classes=12,
            month_tolerance=True,
        )
    if lags is not None:
        out["primary_three_lags"] = _probe_accuracy(
            x_train=lags[tr], y_train=y_p[tr], x_test=lags[te], y_test=y_p[te], classes=n_p
        )
        out["month_within_one_three_lags"] = _probe_accuracy(
            x_train=lags[tr],
            y_train=y_m[tr],
            x_test=lags[te],
            y_test=y_m[te],
            classes=12,
            month_tolerance=True,
        )
    out["n_probe_train"] = float(int(train_mask.sum()))
    out["n_probe_test"] = float(int(test_mask.sum()))
    return out


def _first_lead_lags(*, fold: Fold, pairs: np.ndarray, device: str) -> torch.Tensor | None:
    """Return the lag columns at the first valid time of each (Primary, run day), or None.

    The rows of a fold carry the lag columns in the layout's `lags` slice. For each pair the row
    with the smallest lead is used. Pairs with no row (a run that is not in the training window) get
    zeros, and the arm without lags has no such columns.
    """
    lag_slice = fold.train.layout.slices["lags"]
    if fold.x.shape[1] != fold.train.layout.n_inputs:
        return None
    t = fold.train
    key = t.primary.astype(np.int64) * 10_000 + t.origin_day
    order = np.lexsort((t.lead, key))
    key_sorted = key[order]
    first = order[np.r_[True, key_sorted[1:] != key_sorted[:-1]]]
    lookup = dict(zip(key[first].tolist(), first.tolist(), strict=True))
    out = np.zeros((len(pairs), lag_slice.stop - lag_slice.start), dtype=np.float32)
    for i, (p, d) in enumerate(pairs):
        j = lookup.get(int(p) * 10_000 + int(d))
        if j is not None:
            out[i] = t.x[j, lag_slice]
    return torch.from_numpy(out).to(device)


# --- running ---------------------------------------------------------------------------------


def left_out_primaries(*, group: int | None) -> frozenset[int]:
    """Return the Primaries left out of training: every fourth by index, or none."""
    if group is None:
        return frozenset()
    return frozenset(range(group, len(prep.primaries()), N_GROUPS))


def tag_of(*, arm: str, group: int | None) -> str:
    """Return an arm's folder name."""
    return arm if group is None else f"{arm}_ho_g{group}"


def run(
    *, arms: list[str], group: int | None, seeds: list[int], months: list[str], device: str
) -> int:
    """Fit each arm on each fold and seed; return 3 if the memory stop line was reached, else 0."""
    left_out = left_out_primaries(group=group)
    table = history_table()
    score_set = set(left_out) if group is not None else set(range(len(prep.primaries())))
    for month in months:
        refit = prep.refit_of(month=month)
        for arm_name in arms:
            arm = ARMS[arm_name]
            fold: Fold | None = None
            tag = tag_of(arm=arm_name, group=group)
            for seed in seeds:
                target = OUT_DIR / "seed_preds" / tag / f"{month}_s{seed}.parquet"
                info_path = OUT_DIR / "info" / tag / f"{month}_s{seed}.json"
                alt_path = OUT_DIR / "seed_preds" / f"{tag}_gm" / f"{month}_s{seed}.parquet"
                needs_alt = group is not None and arm.embedding
                if info_path.exists() and (alt_path.exists() or not needs_alt):
                    continue
                if available_gb() <= MIN_AVAILABLE_GB:
                    print(f"stop line: {available_gb():.1f} GB available", flush=True)
                    return 3
                started = time.monotonic()
                if fold is None:
                    fold = build_fold(refit=refit, arm=arm, left_out=left_out, table=table)
                    counts = {int(p): int((fold.train.primary == p).sum()) for p in fold.seen}
                    print(f"fold {month} {tag}: {len(fold.train.x):,} rows; {counts}", flush=True)
                    if left_out:
                        names = prep.primaries()["site"].to_list()
                        print("left out:", [names[p] for p in sorted(left_out)], flush=True)
                torch.manual_seed(seed)
                model = make_network(
                    arm=arm, n_inputs=fold.x.shape[1], n_primaries=len(prep.primaries())
                ).to(device)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model, epochs, val, peak = train_network(
                    model=model, fold=fold, arm=arm, seed=seed, device=device
                )
                set_unseen_embeddings(model=model, seen=fold.seen)
                save_dir = OUT_DIR / "models" / tag / month
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / f"s{seed}.pt")
                q = forecast_month(
                    model=model,
                    fold=fold,
                    arm=arm,
                    month=month,
                    score_primaries=score_set,
                    device=device,
                )
                common.atomic_write(frame=q, target=target)
                if group is not None and arm.embedding:
                    set_unseen_embeddings(model=model, seen=fold.seen, by_group=True)
                    q2 = forecast_month(
                        model=model,
                        fold=fold,
                        arm=arm,
                        month=month,
                        score_primaries=score_set,
                        device=device,
                    )
                    common.atomic_write(frame=q2, target=alt_path)
                info: dict[str, object] = {
                    "epochs": epochs,
                    "val_nll": val,
                    "peak_gpu_gb": peak,
                    "seconds": time.monotonic() - started,
                    "scale_mean": fold.scale_mean,
                    "scale_std": fold.scale_std,
                }
                if arm.history and not arm.shuffled and group is None:
                    info["probe"] = probe(
                        model=model, fold=fold, arm=arm, refit=refit, device=device, table=table
                    )
                info_path.parent.mkdir(parents=True, exist_ok=True)
                info_path.write_text(json.dumps(info))
                print(
                    f"{tag} {month} seed {seed}: {info['seconds']:.0f} s, {epochs} epochs, "
                    f"peak GPU {peak:.1f} GB",
                    flush=True,
                )
    return 0


def check_alignment() -> None:
    """Assert that the history ends at `lag_origin` and no history value is from the future."""
    table = history_table()
    names, _, _ = prep.feature_names(scheme=SCHEME)
    col = names.index("lag_origin")
    for p, row in enumerate(prep.primaries().iter_rows(named=True)):
        z = np.load(prep.RAW_DIR / SCHEME / f"{row['site']}.npz")
        day = ((z["origin"] - fit406._us(t=prep.FIRST_RUN)) // fit406.US_PER_DAY).astype(np.int64)
        ok = (day >= 0) & (day < N_RUN_DAYS)
        last = table[p, day[ok], -1]
        lag = z["x"][ok, col]
        same = np.isnan(last) == np.isnan(lag)
        both = ~np.isnan(last) & ~np.isnan(lag)
        close = np.isclose(last[both], lag[both], atol=1e-4)
        if row["site"] in fit406.SHORT_HISTORY:
            after_cut = z["origin"][ok] >= fit406._us(t=fit406.CUT) + 8 * fit406.US_PER_DAY
            same = same | ~after_cut
        if not same.all():
            msg = f"{row['site']}: missing flags differ from lag_origin on {(~same).sum()} rows"
            raise AssertionError(msg)
        if not close.all():
            msg = f"{row['site']}: history[-1] differs from lag_origin on {(~close).sum()} rows"
            raise AssertionError(msg)
        print(f"{row['site']}: {len(lag):,} rows, last history value equals lag_origin", flush=True)
    print("alignment check passed")


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    sub.add_parser("history")
    f = sub.add_parser("fit")
    f.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    f.add_argument("--held-out-group", type=int, default=None, choices=range(N_GROUPS))
    f.add_argument("--seeds", nargs="+", type=int, default=None)
    f.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    f.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "fit" and args.seeds is None:
        args.seeds = [0, 1] if args.held_out_group is not None else [0, 1, 2]
    if args.command == "check":
        check_alignment()
    elif args.command == "history":
        print(history_table().shape)
    else:
        sys.exit(
            run(
                arms=args.arms,
                group=args.held_out_group,
                seeds=args.seeds,
                months=args.months,
                device=args.device,
            )
        )


if __name__ == "__main__":
    main()
