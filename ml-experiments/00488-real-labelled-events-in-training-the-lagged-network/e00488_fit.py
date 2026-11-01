"""Fit and forecast the real-labelled-event training regimes of experiment 488.

The base network is experiment 404's A0-inj (a plain mixture density network on three lags, trained
on the injected world of seed 100 + n). Only the training rows change between arms. Real labelled
events come from experiment 400's `events_corrected.parquet`; the normal-running-arrangement (NRA)
correction of a half-hour inside an event is experiment 435's `nra_correction_share.npz` (the step
as a fraction of capacity, at depth factors 0.75, 1.0, and 1.25). Forward-only: an event is
available to the fit of scored month m when it ends at least 14 days before m starts.

Arms:

- `repro`: the A0-inj recipe refitted with this script's loop (the reproduction check).
- `RealMix`: rows whose target lies inside an available event are removed and re-added with the
  NRA-corrected target and the real (observed) lags, repeated so that they are 2% of training rows.
- `RealMix-shuf`: as RealMix, but each event's lag pattern and depth are injected into a random
  clean period of the same length (same Primary, training months); the target stays the clean
  actual.
- `PT-FT`: the stored A0-inj network fine-tuned for 2 epochs at one tenth of the learning rate on
  the corrected event rows plus an equal number of rows with no labelled event from the same months.
- `RealMix-d075`, `RealMix-d125`: RealMix with the correction scaled by 0.75 and 1.25.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00488-*/e00488_fit.py fit --arm RealMix --month 2026-06
```
"""

import argparse
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00488-real-labelled-events-in-training-the-lagged-network" / "work"
)
STORED_DIR: Final[Path] = fit.MODEL_DIR
"""Experiment 404's stored networks (read only)."""
fit.MODEL_DIR = WORK_DIR / "models"
fit.PRED_DIR = WORK_DIR / "predictions"
EVENTS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
    / "work"
    / "events_corrected.parquet"
)
SHARE_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
ARMS: Final[tuple[str, ...]] = (
    "repro",
    "RealMix",
    "RealMix-shuf",
    "PT-FT",
    "RealMix-d075",
    "RealMix-d125",
)
DEPTH_KEY: Final[dict[str, str]] = {"RealMix-d075": "f0.75", "RealMix-d125": "f1.25"}
FORWARD_GAP_STEPS: Final[int] = 14 * 48
REAL_SHARE: Final[float] = 0.02
FT_EPOCHS: Final[int] = 2
FT_LR_FACTOR: Final[float] = 0.1
SHUF_TRIES: Final[int] = 400
MIN_EVENTS: Final[int] = 5


def available_events(*, month: str) -> pl.DataFrame:
    """Return the real events that ended at least 14 days before the scored month starts."""
    refit = int(np.asarray(common.grid_index(times=pl.Series([prep.refit_of(month=month)])))[0])
    first = int(np.asarray(common.grid_index(times=pl.Series([prep.FIRST_RUN])))[0]) + 19
    events = pl.read_parquet(EVENTS_PATH).filter(pl.col("depth_mw").is_not_null())
    sites = prep.primaries()["site"].to_list()
    return events.filter(
        (pl.col("end_idx") + FORWARD_GAP_STEPS <= refit) & (pl.col("end_idx") >= first)
    ).with_columns(primary=pl.col("label").replace_strict(sites, list(range(len(sites)))))


def event_span_mask(*, events: pl.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """Return the (primaries, grid) mask of the events' unpadded spans."""
    mask = np.zeros(shape, dtype=bool)
    for e in events.iter_rows(named=True):
        mask[e["primary"], e["onset_idx"] : e["end_idx"] + 1] = True
    return mask


def weighted_epochs(
    *,
    net: torch.nn.Module,
    pool: fit.Pool,
    ridx: torch.Tensor,
    y: torch.Tensor,
    lags: torch.Tensor,
    scalers: fit.Scalers,
    epochs: int,
    lr: float,
    device: str,
    tag: str,
) -> list[float]:
    """Train `net` on rows `pool[ridx]` with targets `y` and lags `lags`; return epoch losses."""
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(ridx)
    losses = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            b = order[start : start + fit.BATCH]
            rows = ridx[b]
            primary = pool.primary[rows]
            log_w, mean, std = net(pool.base[rows], lags[b], primary)
            z = (y[b] - tm[primary]) / ts[primary]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = pool.weight[rows]
            loss = (nll * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(b)
        losses.append(total / n)
        print(f"{tag} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    return losses


def oversample(*, unique: torch.Tensor, total: int, gen: torch.Generator) -> torch.Tensor:
    """Return indices (into `unique`) that make the event rows `REAL_SHARE` of the new total."""
    rest = total - len(unique)
    want = int(np.ceil(REAL_SHARE * rest / (1.0 - REAL_SHARE)))
    if want <= len(unique):
        return torch.arange(len(unique))
    full, extra = divmod(want, len(unique))
    pick = torch.randperm(len(unique), generator=gen)[:extra]
    return torch.cat([torch.arange(len(unique)).repeat(full), pick])


def shuffled_power(
    *,
    power: np.ndarray,
    code: np.ndarray,
    padded: np.ndarray,
    share: np.ndarray,
    events: pl.DataFrame,
    rng: np.random.Generator,
    low: dict[int, int],
    high: dict[int, int],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Inject each event's lag pattern into a random clean period of its Primary.

    Returns the new power array and, per event, (event row, primary, new start, length). An event
    with no clean period found after `SHUF_TRIES` draws is skipped.
    """
    out = power.copy()
    placed = []
    busy = padded | (code != 0) | ~np.isfinite(power)
    for k, e in enumerate(events.iter_rows(named=True)):
        p, length = e["primary"], e["end_idx"] - e["onset_idx"] + 1
        pattern = share[p, e["onset_idx"] : e["end_idx"] + 1]
        span = high[p] - low[p] - length - 96
        if span <= 0:
            continue
        for _ in range(SHUF_TRIES):
            s = low[p] + 48 + int(rng.integers(0, span))
            if not busy[p, s - 48 : s + length + 48].any():
                out[p, s : s + length] += pattern
                busy[p, s - 48 : s + length + 48] = True
                placed.append((k, p, s, length))
                break
    return out, placed


def fit_arm(*, arm: str, month: str, seed: int, device: str = "cuda") -> Path:  # noqa: PLR0915
    """Fit one arm for a scored month and seed; save the network; return the path."""
    started = time.monotonic()
    train, _test, scalers, arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=2
    )
    assert train is not None
    events = available_events(month=month)
    six = events.filter(pl.col("label").is_in(score.SIX_SIGNED)).height
    print(f"{month}: {events.height} available events ({six} at the signed-MW six)", flush=True)
    world, code = fit.world_power(seed=100 + seed)
    observed, _ = fit.world_power(seed=None)
    shape = observed.shape
    share_file = np.load(SHARE_PATH)
    share = share_file[DEPTH_KEY.get(arm, "f1.0")].astype(np.float32)
    span = event_span_mask(events=events, shape=shape)
    padded = score.real_event_mask()
    kw = {"pool": train, "lag_stats": lag_stats, "device": device}
    lags_world, lag_grid = fit.lag_features(power=torch.tensor(world, device=device), **kw)
    n = len(train)
    target_np = train.target.cpu().numpy()
    primary_np = train.primary.cpu().numpy()
    in_event = torch.from_numpy(span[primary_np, np.clip(target_np, 0, shape[1] - 1)]).to(device)
    unique = torch.nonzero(in_event).squeeze(1)
    print(
        f"{arm} {month}: {n:,} rows, {len(unique):,} with target inside an available event "
        f"({100 * len(unique) / n:.2f}%)",
        flush=True,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    path = fit.MODEL_DIR / arm / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    net = fit.build_network(arm="A0-inj", n_base=train.base.shape[1], n_trust=0).to(device)
    y_corr = train.y[unique] - torch.from_numpy(
        share[primary_np[unique.cpu().numpy()], target_np[unique.cpu().numpy()]]
    ).to(device)
    info: dict = {"events": events.height, "events_six": six, "event_rows_unique": len(unique)}
    if arm == "repro":
        ridx, y, lags = torch.arange(n, device=device), train.y, lags_world
        epochs, lr = fit.EPOCHS, fit.LEARNING_RATE
    elif arm.startswith("RealMix") and arm != "RealMix-shuf":
        lags_obs, _ = fit.lag_features(power=torch.tensor(observed, device=device), **kw)
        keep = torch.nonzero(~in_event).squeeze(1)
        sel = oversample(unique=unique, total=n, gen=gen).to(device)
        add, y_add = unique[sel], y_corr[sel]
        ridx = torch.cat([keep, add])
        y = torch.cat([train.y[keep], y_add])
        lags = torch.cat([lags_world[keep], lags_obs[add]])
        info["event_rows_added"] = len(add)
        epochs, lr = fit.EPOCHS, fit.LEARNING_RATE
    elif arm == "RealMix-shuf":
        rng = np.random.default_rng(seed)
        low = {p: int(target_np[primary_np == p].min()) for p in np.unique(primary_np)}
        high = {p: int(target_np[primary_np == p].max()) for p in np.unique(primary_np)}
        shuf, placed = shuffled_power(
            power=world, code=code, padded=padded, share=share_file["f1.0"], events=events,
            rng=rng, low=low, high=high,
        )  # fmt: skip
        info["events_placed"] = len(placed)
        lags_shuf, _ = fit.lag_features(power=torch.tensor(shuf, device=device), **kw)
        keep = torch.nonzero(~in_event).squeeze(1)
        factor = len(oversample(unique=unique, total=n, gen=gen)) / len(unique)
        spans = events.select("primary", "onset_idx", "end_idx").rows()
        picks = []
        for k, p, s, length in placed:
            _p, onset, end = spans[k]
            own = int(((primary_np == p) & (target_np >= onset) & (target_np <= end)).sum())
            new = np.flatnonzero((primary_np == p) & (target_np >= s) & (target_np < s + length))
            if own and len(new):
                picks.append(new[rng.integers(0, len(new), size=round(own * factor))])
        add = torch.from_numpy(np.concatenate(picks)).to(device)
        ridx = torch.cat([keep, add])
        y = torch.cat([train.y[keep], train.y[add]])
        lags = torch.cat([lags_world[keep], lags_shuf[add]])
        info["event_rows_added"] = len(add)
        epochs, lr = fit.EPOCHS, fit.LEARNING_RATE
    elif arm == "PT-FT":
        saved = torch.load(STORED_DIR / "A0-inj" / f"{month}_s{seed}.pt", weights_only=False)
        assert np.allclose(saved["standardiser_mean"], scalers.standardiser.mean, equal_nan=True)
        assert np.allclose(saved["target_mean"], scalers.target_mean)
        net.load_state_dict(saved["state"])
        lags_obs, _ = fit.lag_features(power=torch.tensor(observed, device=device), **kw)
        lag_in = padded[primary_np[:, None], np.clip(lag_grid.cpu().numpy(), 0, shape[1] - 1)]
        labelled = padded[primary_np, np.clip(target_np, 0, shape[1] - 1)] | lag_in.any(1)
        month_of = target_np // 1461
        clean_pool = np.flatnonzero(~labelled)
        wanted = np.bincount(month_of[unique.cpu().numpy()], minlength=month_of.max() + 1)
        rng = np.random.default_rng(seed)
        clean_pick = []
        for m, count in enumerate(wanted):
            cand = clean_pool[month_of[clean_pool] == m]
            if count and len(cand):
                clean_pick.append(cand[rng.integers(0, len(cand), size=count)])
        clean = torch.from_numpy(np.concatenate(clean_pick)).to(device)
        ridx = torch.cat([unique, clean])
        y = torch.cat([y_corr, train.y[clean]])
        lags = torch.cat([lags_obs[unique], lags_world[clean]])
        info["finetune_clean_rows"] = len(clean)
        epochs, lr = FT_EPOCHS, fit.LEARNING_RATE * FT_LR_FACTOR
    else:
        raise ValueError(arm)
    losses = weighted_epochs(
        net=net, pool=train, ridx=ridx, y=y, lags=lags, scalers=scalers, epochs=epochs, lr=lr,
        device=device, tag=f"{arm} {month} s{seed}",
    )  # fmt: skip
    seconds = time.monotonic() - started
    torch.save(
        {
            "state": net.state_dict(), "arm": arm, "month": month, "seed": seed,
            "n_base": train.base.shape[1], "target_mean": scalers.target_mean,
            "target_std": scalers.target_std, "standardiser_mean": scalers.standardiser.mean,
            "standardiser_std": scalers.standardiser.std,
            "standardiser_mask": scalers.standardiser.mask, "lag_stats": lag_stats,
            "trust_mean": None, "trust_std": None, "losses": losses, "rows": len(ridx),
            "seconds": seconds, "g0": arrays.g0, "info": info,
        },
        path,
    )  # fmt: skip
    print(f"{arm} {month} s{seed}: {len(ridx):,} rows, {seconds:.0f} s, {info} -> {path}")
    return path


@contextlib.contextmanager
def mixture_source(*, arm: str) -> Iterator[None]:
    """Make `score.forecast(arm="B-mix")` mix the lagless table with `arm`'s network."""
    original = score.load_network

    def load(*, arm: str, month: str, seed: int, device: str):  # noqa: ANN202
        return original(
            arm=source if arm == "A0-inj" else arm, month=month, seed=seed, device=device
        )

    source = arm
    score.load_network = load
    try:
        yield
    finally:
        score.load_network = original


def predict(*, arm: str, month: str, seed: int, device: str = "cuda") -> None:
    """Forecast the clean world and the three scoring worlds with the arm, alone and mixed."""
    real = score.real_event_mask()
    for world in (-1, *score.SCORING_SEEDS):
        frame = score.forecast(
            arm=arm, month=month, seed=seed, world=world, device=device, real=real
        )
        out = score.prediction_path(arm=arm, month=month, seed=seed, world=world)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out)
        with mixture_source(arm=arm):
            mixed = score.forecast(
                arm="B-mix", month=month, seed=seed, world=world, device=device, real=real
            )
        out = score.prediction_path(arm=f"{arm}+LL", month=month, seed=seed, world=world)
        out.parent.mkdir(parents=True, exist_ok=True)
        mixed.write_parquet(out)
        print(f"{arm} {month} s{seed} w{world}: {frame.height:,} rows", flush=True)


def main() -> None:
    """Fit or forecast an arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "both"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command in ("fit", "both"):
        fit_arm(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    if args.command in ("predict", "both"):
        predict(arm=args.arm, month=args.month, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
