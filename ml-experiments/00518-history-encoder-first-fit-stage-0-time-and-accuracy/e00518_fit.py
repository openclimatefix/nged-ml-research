"""Experiment 518: fit the stage-0 history encoder H on fold 2026-01, then score it.

One H, one seed, on injected worlds of experiment 507's generator (families A and B), with the
reading-class head dropped. Commands (all through `run.sh`):

- `selftest`: CPU, synthetic data, no private files. Checks the model, the split, the labels, the
  losses, an overfit on a toy world, the scoring functions, and the abort rule.
- `counts`: CPU. Prints the window counts of the split, the share of windows with a step in force at
  the newest reading and the genuine share among them, and the share of half-hours masked by the
  apparent-power rule. Run before the fit.
- `timing`: CPU. Times the world generator and one training step, and extrapolates.
- `fit`: trains H (the only command that should use the GPU) and writes the log and the best state.
- `score`: scores the saved H on held-out worlds 200 to 202 and on planted steps.

```bash
ml-experiments/00518-*/run.sh ml-experiments/00518-*/e00518_fit.py selftest
```
"""

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00507_generator as gen  # ty: ignore[unresolved-import]
import e00518_data as data  # ty: ignore[unresolved-import]
import e00518_model as model  # ty: ignore[unresolved-import]
import e00518_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00518-history-encoder-first-fit-stage-0-time-and-accuracy"
)
MONTH: Final[str] = "2026-01"
TRAIN_SEED_BASE: Final[int] = 1000
VALIDATION_WORLDS: Final[tuple[int, ...]] = (150, 151, 152)
TEST_WORLDS: Final[tuple[int, ...]] = (200, 201, 202)
EPOCH_WINDOWS: Final[int] = 16384
BATCH: Final[int] = 128
LEARNING_RATE: Final[float] = 2e-3
MAX_EPOCHS: Final[int] = 60
MIN_EPOCHS: Final[int] = 10
PATIENCE: Final[int] = 8
ABORT_SECONDS: Final[float] = 2 * 3600.0
"""The plan's cap: abort if one fit extrapolates above 2 GPU-hours."""
CUT_PRIMARIES: Final[tuple[str, ...]] = ("D15", "D18")
CUT_DATE: Final[str] = "2025-03-01"


@dataclass(frozen=True)
class Context:
    """Everything a fit or a score needs about one fold.

    Attributes:
        spec: The generator spec (the real readings and the Primaries' properties).
        med: The lagless median (primaries, grid).
        split: The training, validation, and test windows.
        groups: `six`, `eight`, or `other` per Primary.
        weights: The family D sampling weights of the training windows.
    """

    spec: gen.Spec
    med: np.ndarray
    split: data.Split
    groups: np.ndarray
    weights: np.ndarray


def build_context(*, month: str = MONTH) -> Context:
    """Load the private data for the fold: readings, ERA5 lab median, labelled events, the split."""
    import calendar

    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00435_nra_actuals as nra  # ty: ignore[unresolved-import]
    import e00507_common as c  # ty: ignore[unresolved-import]
    import e00507_lagless as lab  # ty: ignore[unresolved-import]

    spec = c.load_spec()
    sites = c.sites()
    table = pl.read_parquet(lab.table_path(month=month, seed=0))
    med = np.full(spec.real.shape, np.nan, dtype=np.float32)
    med[table["primary"].to_numpy().astype(np.int64), table["grid"].to_numpy().astype(np.int64)] = (
        table["q50"].to_numpy()
    )
    labelled = np.zeros(spec.real.shape, dtype=bool)
    for e in pl.read_parquet(nra.EVENTS).iter_rows(named=True):
        if e["label"] in sites and e["onset_idx"] >= 0 and e["end_idx"] >= e["onset_idx"]:
            p = sites.index(e["label"])
            lo = max(int(e["onset_idx"]) - data.EVENT_PADDING, 0)
            labelled[p, lo : int(e["end_idx"]) + data.EVENT_PADDING + 1] = True
    cut = int(common.grid_index(times=pl.Series([CUT_DATE]).str.to_datetime(time_zone="UTC")[0]))
    first = np.array([cut if s in CUT_PRIMARIES else 0 for s in sites])
    refit = int(common.grid_index(times=prep.refit_of(month=month)))
    year, mon = (int(x) for x in month.split("-"))
    half_hours = calendar.monthrange(year, mon)[1] * gen.STEPS_PER_DAY
    candidates = data.candidate_windows(spec=spec, med=med, labelled=labelled, first_history=first)
    split = data.split_by_time(windows=candidates, refit=refit, month_half_hours=half_hours)
    return Context(
        spec=spec,
        med=med,
        split=split,
        groups=c.group_of_primary(),
        weights=data.sampling_weights(spec=spec, med=med, windows=split.train),
    )


def should_abort(*, elapsed: float, epochs_done: int, max_epochs: int, limit: float) -> str | None:
    """Return the reason to abort a fit, or None.

    The fit aborts when the time spent exceeds `limit`, or, from the third epoch, when the time per
    epoch extrapolated to `max_epochs` exceeds `limit`.
    """
    if elapsed > limit:
        return f"elapsed {elapsed / 3600:.2f} h exceeds the {limit / 3600:.2f} h cap"
    if epochs_done >= 3 and elapsed / epochs_done * max_epochs > limit:
        projected = elapsed / epochs_done * max_epochs / 3600
        return f"extrapolated {projected:.2f} h exceeds the {limit / 3600:.2f} h cap"
    return None


def to_tensors(*, arrays: data.Arrays, device: str) -> dict[str, torch.Tensor]:
    """Move one set of arrays to the device."""
    return {
        "x": torch.from_numpy(arrays.x).to(device),
        "step": torch.from_numpy(arrays.step_true).to(device),
        "valid": torch.from_numpy(arrays.valid).to(device),
        "g": torch.from_numpy(arrays.genuine_target).to(device),
        "gv": torch.from_numpy(arrays.genuine_valid).to(device),
    }


def batch_loss(
    *,
    net: model.HistoryEncoder,
    t: dict[str, torch.Tensor],
    index: torch.Tensor,
) -> model.Losses:
    """Return the stage-0 losses of the rows `index` of a tensor set."""
    out = net(t["x"][index])
    return model.stage0_loss(
        out=out,
        step_true=t["step"][index],
        valid=t["valid"][index],
        genuine_target=t["g"][index],
        genuine_valid=t["gv"][index],
    )


@torch.no_grad()
def validation_loss(*, net: model.HistoryEncoder, t: dict[str, torch.Tensor]) -> dict[str, float]:
    """Return the mean validation losses; `step` is the step head's risk plus its size loss."""
    net.eval()
    n = len(t["x"])
    sums = {"step": 0.0, "size": 0.0, "genuine": 0.0, "plain_step": 0.0}
    for lo in range(0, n, 256):
        index = torch.arange(lo, min(lo + 256, n), device=t["x"].device)
        loss = batch_loss(net=net, t=t, index=index)
        for key in sums:
            sums[key] += float(getattr(loss, key)) * len(index)
    out = {key: value / n for key, value in sums.items()}
    out["l_u"] = out["step"] + out["size"]
    return out


def fit(
    *,
    ctx: Context,
    device: str,
    out_dir: Path | None,
    max_epochs: int = MAX_EPOCHS,
    epoch_windows: int = EPOCH_WINDOWS,
    limit: float = ABORT_SECONDS,
    seed: int = 0,
    emit: Callable[[str], None] = print,
) -> dict[str, float | int | str]:
    """Train H; return a summary. Raises `RuntimeError` if the abort rule fires.

    Each epoch draws `epoch_windows` training windows (family D weighted) into a fresh world, on
    one feeder thread that prepares the next epoch while the device trains on this one. Early
    stopping reads the validation block, scored in three fixed worlds, for `PATIENCE` epochs after
    the best, and never before `MIN_EPOCHS`.
    """
    torch.manual_seed(seed)
    net = model.HistoryEncoder().to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    val_parts = [
        data.arrays_of(
            spec=ctx.spec,
            ev=gen.sample_world(spec=ctx.spec, seed=w),
            med=ctx.med,
            windows=ctx.split.validation,
        )
        for w in VALIDATION_WORLDS
    ]
    val = to_tensors(
        arrays=data.Arrays(
            x=np.concatenate([a.x for a in val_parts]),
            step_true=np.concatenate([a.step_true for a in val_parts]),
            valid=np.concatenate([a.valid for a in val_parts]),
            genuine_target=np.concatenate([a.genuine_target for a in val_parts]),
            genuine_valid=np.concatenate([a.genuine_valid for a in val_parts]),
            masked_mva=0,
            observed=0,
        ),
        device=device,
    )

    def make(epoch: int) -> tuple[data.Arrays, float]:
        started = time.monotonic()
        arrays = data.epoch_arrays(
            spec=ctx.spec,
            med=ctx.med,
            windows=ctx.split.train,
            weights=ctx.weights,
            epoch=epoch,
            size=epoch_windows,
            base_seed=TRAIN_SEED_BASE,
        )
        return arrays, time.monotonic() - started

    start = time.monotonic()
    best, best_epoch, best_state = float("inf"), -1, None
    rows: list[dict[str, float]] = []
    with ThreadPoolExecutor(max_workers=1) as feeder:
        pending = feeder.submit(make, 0)
        for epoch in range(max_epochs):
            waited = time.monotonic()
            arrays, gen_seconds = pending.result()
            wait_seconds = time.monotonic() - waited
            pending = feeder.submit(make, epoch + 1)
            t = to_tensors(arrays=arrays, device=device)
            net.train()
            order = torch.randperm(len(t["x"]), device=device)
            running = 0.0
            for lo in range(0, len(order) - BATCH + 1, BATCH):
                loss = batch_loss(net=net, t=t, index=order[lo : lo + BATCH])
                optimiser.zero_grad(set_to_none=True)
                loss.total.backward()
                optimiser.step()
                running += float(loss.total)
            val_loss = validation_loss(net=net, t=val)
            elapsed = time.monotonic() - start
            row = {
                "epoch": epoch + 1,
                "train_total": running / max(len(order) // BATCH, 1),
                **{f"val_{k}": v for k, v in val_loss.items()},
                "gen_seconds": gen_seconds,
                "feeder_wait_seconds": wait_seconds,
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            emit(
                f"epoch {epoch + 1:3d}: train {row['train_total']:.4f}, val L_u "
                f"{val_loss['l_u']:.4f} (plain {val_loss['plain_step']:.4f}), genuine "
                f"{val_loss['genuine']:.3f}, gen {gen_seconds:.1f} s, wait {wait_seconds:.1f} s, "
                f"elapsed {elapsed:.0f} s"
            )
            if val_loss["l_u"] < best:
                best, best_epoch = val_loss["l_u"], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            reason = should_abort(
                elapsed=elapsed, epochs_done=epoch + 1, max_epochs=max_epochs, limit=limit
            )
            if reason is not None:
                raise RuntimeError(f"fit aborted: {reason}")
            if epoch + 1 >= MIN_EPOCHS and epoch + 1 - best_epoch >= PATIENCE:
                break
    assert best_state is not None
    summary: dict[str, float | int | str] = {
        "epochs": len(rows),
        "best_epoch": best_epoch,
        "best_val_l_u": best,
        "val_stops_falling_before_epoch_10": int(best_epoch < 10),
        "elapsed_seconds": rows[-1]["elapsed_seconds"],
        "parameters": model.parameter_count(model=net),
    }
    if out_dir is not None:
        (out_dir / "models").mkdir(parents=True, exist_ok=True)
        target = out_dir / "models" / f"h_stage0_{MONTH}_seed0.pt"
        torch.save(best_state, target.with_suffix(".tmp"))
        target.with_suffix(".tmp").replace(target)
        pl.DataFrame(rows).write_csv(out_dir / "fit_log.csv")
        (out_dir / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def predict(
    *, net: model.HistoryEncoder, arrays: data.Arrays, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the estimated level (windows, 672) in spreads, and the genuine log-odds (windows,)."""
    net.eval()
    levels, logits = [], []
    for lo in range(0, len(arrays.x), 256):
        out = net(torch.from_numpy(arrays.x[lo : lo + 256]).to(device))
        levels.append(out.level.cpu().numpy())
        logits.append(out.genuine_logit.cpu().numpy())
    return np.concatenate(levels), np.concatenate(logits)


Predictor = Callable[[data.Arrays], tuple[np.ndarray, np.ndarray]]


def score(
    *,
    ctx: Context,
    predictor: Predictor,
    worlds: tuple[int, ...] = TEST_WORLDS,
) -> dict[str, Any]:
    """Score a predictor on the test windows of held-out worlds, and on real windows.

    Returns the numbers as nested dictionaries keyed by group (`six`, `eight`, `other`, `all`).
    """
    windows = ctx.split.test
    n = len(windows)
    groups = ctx.groups[windows.primary]
    event_parts, lag_parts, auc_parts, ceiling_parts = [], [], [], []
    for w in worlds:
        ev = gen.sample_world(spec=ctx.spec, seed=w)
        arrays = data.arrays_of(spec=ctx.spec, ev=ev, med=ctx.med, windows=windows)
        level, genuine_logit = predictor(arrays)
        events = sc.true_events(ev=ev, p=windows.primary, d=windows.day, step_true=arrays.step_true)
        scored = sc.score_events(level=level, events=events, step_true=arrays.step_true)
        event_parts.append(scored.with_columns(pl.lit(w).alias("world")))
        ceiling_parts.append(
            sc.score_events(level=arrays.step_true, events=events, step_true=arrays.step_true)
        )
        lag = sc.lag_errors(level=level, step_true=arrays.step_true, valid=arrays.valid)
        lag_parts.append(
            lag.with_columns(
                pl.Series("window", np.arange(n)),
                pl.lit(w).alias("world"),
                pl.Series("group", groups),
            )
        )
        gv = arrays.genuine_valid
        auc_parts.append((genuine_logit[gv], arrays.genuine_target[gv]))
    events = pl.concat(event_parts).with_columns(
        pl.col("window").map_elements(lambda i: groups[i], return_dtype=pl.String).alias("group")
    )
    ceiling = pl.concat(ceiling_parts).with_columns(events["group"])
    lag = pl.concat(lag_parts)
    empty = gen.Events(
        family=np.zeros((ctx.spec.real.shape[0], ctx.spec.days, gen.SLOTS), dtype=np.int8),
        start=np.zeros_like(gen.sample_world(spec=ctx.spec, seed=0).start),
        end=np.zeros_like(gen.sample_world(spec=ctx.spec, seed=0).end),
        size=np.zeros((ctx.spec.real.shape[0], ctx.spec.days, gen.SLOTS), dtype=np.float32),
        source=np.full((ctx.spec.real.shape[0], ctx.spec.days, gen.SLOTS), -1, dtype=np.int16),
        ramp=np.zeros((ctx.spec.real.shape[0], ctx.spec.days, gen.SLOTS), dtype=np.int16),
        stair=np.zeros((ctx.spec.real.shape[0], ctx.spec.days, gen.SLOTS), dtype=np.int16),
    )
    real = data.arrays_of(spec=ctx.spec, ev=empty, med=ctx.med, windows=windows)
    real_level, _ = predictor(real)
    real_flag = sc.flagged(level=real_level)
    result: dict[str, Any] = {}
    genuine_all = np.concatenate([a for a, _ in auc_parts])
    target_all = np.concatenate([b for _, b in auc_parts])
    for name in ("all", "six", "eight", "other"):
        e = events if name == "all" else events.filter(pl.col("group") == name)
        top = ceiling if name == "all" else ceiling.filter(pl.col("group") == name)
        if e.height == 0:
            result[name] = {"events": 0}
            continue
        lg = lag if name == "all" else lag.filter(pl.col("group") == name)
        sel = np.ones(n, dtype=bool) if name == "all" else groups == name
        found = {
            f"found_{int(h)}h": sc.cluster_mean(
                values=(e["onset_error_hours"].to_numpy() <= h), cluster=e["window"].to_numpy()
            )
            for h in sc.FOUND_HOURS
        }
        missed = sc.cluster_mean(
            values=~(e["onset_error_hours"].to_numpy() <= 24.0), cluster=e["window"].to_numpy()
        )
        hit = e.filter(pl.col("onset_error_hours") <= 24.0)
        touched = lg.filter(pl.col("touched"))
        clean = lg.filter(pl.col("usable") & ~pl.col("touched"))
        false_rate = sc.cluster_mean(values=real_flag[sel], cluster=np.arange(n)[sel])
        recall = found["found_2h"]
        result[name] = {
            "events": e.height,
            **{k: str(v) for k, v in found.items()},
            "missed": str(missed),
            **{
                f"ceiling_found_{int(h)}h": float((top["onset_error_hours"] <= h).mean())
                for h in sc.FOUND_HOURS
            },
            "size_error_median": float(hit["size_error"].median()) if hit.height else float("nan"),
            "size_error_within_25pc": float((hit["size_error"] <= 0.25).mean())
            if hit.height
            else float("nan"),
            "lag_error_repaired": float(touched["repaired"].mean())
            if touched.height
            else float("nan"),
            "lag_error_unrepaired": float(touched["unrepaired"].mean())
            if touched.height
            else float("nan"),
            "lag_error_clean_rows": float(clean["repaired"].mean())
            if clean.height
            else float("nan"),
            "false_flag_rate_real_windows": str(false_rate),
            "planted_precision_at_2pc": sc.precision_at_rate(
                recall=recall.mean, false_rate=false_rate.mean
            ),
        }
    result["genuine_head_auc"] = sc.rank_auc(score=genuine_all, label=target_all)
    result["genuine_head_windows"] = len(target_all)
    return result


def load_net(*, out_dir: Path, device: str) -> model.HistoryEncoder:
    """Load the saved best H."""
    net = model.HistoryEncoder().to(device)
    net.load_state_dict(torch.load(out_dir / "models" / f"h_stage0_{MONTH}_seed0.pt"))
    return net


def counts(*, ctx: Context) -> None:
    """Print the split sizes, the in-force shares, and the masked share (before any fit)."""
    s = ctx.split
    print(
        f"windows: train {len(s.train):,}, validation {len(s.validation):,}, test {len(s.test):,}"
    )
    for name in ("six", "eight", "other"):
        print(
            f"  {name}: train {int((ctx.groups[s.train.primary] == name).sum()):,}, "
            f"test {int((ctx.groups[s.test.primary] == name).sum()):,}"
        )
    ev = gen.sample_world(spec=ctx.spec, seed=TRAIN_SEED_BASE)
    any_step, genuine = data.in_force_shares(ev=ev, windows=s.train)
    print(
        f"training windows with a step in force at the newest reading: {any_step:.3f}; "
        f"family B share among them: {genuine:.3f} (plan target about 0.6; 507 measured 0.79)"
    )
    arrays = data.arrays_of(spec=ctx.spec, ev=ev, med=ctx.med, windows=s.test)
    print(
        f"test windows: apparent-power rule masks {arrays.masked_mva / arrays.observed:.4%} of "
        f"observed half-hours; genuine-head labels on {arrays.genuine_valid.mean():.3f} of windows"
    )


def timing(*, ctx: Context) -> None:
    """Time the generator and one CPU training step; extrapolate to a GPU-free estimate."""
    started = time.monotonic()
    arrays = data.epoch_arrays(
        spec=ctx.spec,
        med=ctx.med,
        windows=ctx.split.train,
        weights=ctx.weights,
        epoch=0,
        size=2048,
        base_seed=TRAIN_SEED_BASE,
    )
    per_window = (time.monotonic() - started) / 2048
    print(
        f"generator: {per_window * 1000:.2f} ms per window on one core; one epoch of "
        f"{EPOCH_WINDOWS} windows takes {per_window * EPOCH_WINDOWS:.1f} s of feeder time"
    )
    torch.set_num_threads(1)
    net = model.HistoryEncoder()
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    t = to_tensors(arrays=arrays, device="cpu")
    started = time.monotonic()
    for i in range(5):
        loss = batch_loss(net=net, t=t, index=torch.arange(i * BATCH, (i + 1) * BATCH))
        optimiser.zero_grad()
        loss.total.backward()
        optimiser.step()
    step = (time.monotonic() - started) / 5
    steps = EPOCH_WINDOWS // BATCH
    print(
        f"training step on one CPU core: {step:.2f} s per batch of {BATCH}; "
        f"{steps} steps per epoch = {step * steps:.0f} s on the CPU (an upper bound for a GPU)"
    )
    print(
        f"at most {MAX_EPOCHS} epochs: feeder "
        f"{per_window * EPOCH_WINDOWS * MAX_EPOCHS / 3600:.2f} h, "
        f"CPU training {step * steps * MAX_EPOCHS / 3600:.2f} h (the GPU should be far below this)"
    )


def selftest() -> None:  # noqa: PLR0915
    """Check the model, split, labels, losses, scoring, and abort rule on a synthetic world."""
    torch.manual_seed(0)
    torch.set_num_threads(1)
    net = model.HistoryEncoder()
    count = model.parameter_count(model=net)
    print(f"H has {count:,} parameters (plan: about 100,000)")
    assert 70_000 <= count <= 130_000, count
    out = net(torch.zeros(3, model.CHANNELS, gen.HISTORY))
    assert out.step_logit.shape == (3, 672)
    assert out.size.shape == (3, 672)
    assert out.genuine_logit.shape == (3,)
    # Causality is not required (the window is all history), but a change at the newest reading must
    # reach the genuine head, which reads the last token.
    x = torch.zeros(1, model.CHANNELS, gen.HISTORY)
    x2 = x.clone()
    x2[:, 0, -8:] = 1.0
    assert (net(x).genuine_logit - net(x2).genuine_logit).abs().max() > 0
    # Data on a synthetic spec whose median is the noise-free signal.
    spec = gen.synthetic_spec(primaries=4, days=200)
    t = np.arange(spec.real.shape[1])
    med = np.stack([0.5 + 0.2 * np.sin(2 * np.pi * (t + 5 * p) / 48) for p in range(4)]).astype(
        np.float32
    )
    labelled = np.zeros(spec.real.shape, dtype=bool)
    labelled[1, 3000:3100] = True
    first = np.zeros(4, dtype=np.int64)
    cand = data.candidate_windows(spec=spec, med=med, labelled=labelled, first_history=first)
    touched_days = np.flatnonzero(
        (gen.newest_of(day=np.arange(spec.days)) + 48 >= 3000 - 0)
        & (gen.newest_of(day=np.arange(spec.days)) - 671 - 48 <= 3100)
    )
    assert not np.isin(cand.day[cand.primary == 1], touched_days).any(), "a labelled event leaked"
    refit = gen.origin_of(day=np.array(150))
    split = data.split_by_time(windows=cand, refit=int(refit), month_half_hours=30 * 48)
    n_train_newest = gen.newest_of(day=split.train.day)
    n_val_newest = gen.newest_of(day=split.validation.day)
    n_test_origin = gen.origin_of(day=split.test.day)
    assert n_train_newest.max() < n_val_newest.min() - 14 * 48 + 1, "train and validation overlap"
    assert n_val_newest.max() < refit - 24 * 48, "validation inside the embargo"
    assert n_test_origin.min() >= refit
    assert n_test_origin.max() < refit + 30 * 48
    print(
        f"split: {len(split.train)} train, {len(split.validation)} validation, "
        f"{len(split.test)} test windows; embargo and gap hold"
    )
    ev = gen.sample_world(spec=spec, seed=7)
    arrays = data.arrays_of(spec=spec, ev=ev, med=med, windows=split.train)
    again = data.arrays_of(spec=spec, ev=ev, med=med, windows=split.train)
    assert np.array_equal(arrays.x, again.x), "arrays are not deterministic"
    # Step labels equal the shifted reading minus the stored reading, in spreads, at the signed
    # Primaries; and the input surprise equals the label plus the unshifted surprise.
    p, d = split.train.primary, split.train.day
    g = data.history_grid(day=d)
    shifted = gen.reading(
        spec=spec, ev=ev, p=np.repeat(p, 672), d=np.repeat(d, 672), g=g.reshape(-1), mode="shifted"
    ).reshape(len(p), 672)
    signed = ~spec.mva_only[p]
    want = (shifted - spec.real[p[:, None], g]) / spec.spread[p][:, None]
    assert np.allclose(arrays.step_true[signed], want[signed], atol=1e-4)
    base = (spec.real[p[:, None], g] - med[p[:, None], g]) / spec.spread[p][:, None]
    got = arrays.x[:, 0] * model.SURPRISE_SCALE
    clipped = np.clip(base + want, -model.SURPRISE_CLIP, model.SURPRISE_CLIP)
    assert np.allclose(got[signed], clipped[signed], atol=1e-3), "surprise channel is wrong"
    in_force, genuine = data.in_force_shares(ev=ev, windows=split.train)
    labelled_share = arrays.genuine_valid.mean()
    print(
        f"toy world: step in force at newest in {in_force:.3f} of windows; B share {genuine:.3f}; "
        f"genuine-head labels on {labelled_share:.3f}; apparent-power rule masks "
        f"{arrays.masked_mva / max(arrays.observed, 1):.3%}"
    )
    assert labelled_share <= in_force + 1e-9
    # Genuine labels: a B step in force at the newest has target 1, an A step with onset inside 0.
    in_a, in_b = gen.in_force_at_newest(ev=ev)
    only_b = in_b[p, d] & ~in_a[p, d]
    assert (arrays.genuine_target[only_b & arrays.genuine_valid] == 1).all()
    assert (arrays.genuine_target[in_a[p, d] & arrays.genuine_valid] == 0).all()
    # Losses: with prior 0 the PU risk is the plain cross-entropy; gradients are finite.
    logit = torch.randn(8, 672, requires_grad=True)
    pos = torch.rand(8, 672) < 0.3
    valid = torch.rand(8, 672) < 0.95
    risk0, plain0 = model.pu_step_loss(logit=logit, positive=pos, valid=valid, prior=0.0)
    assert torch.allclose(risk0, plain0)
    risk, _ = model.pu_step_loss(logit=logit, positive=pos, valid=valid, prior=0.03)
    risk.backward()
    assert logit.grad is not None
    assert torch.isfinite(logit.grad).all()
    big = torch.full((8, 672), 20.0)
    clamped, _ = model.pu_step_loss(logit=big, positive=pos, valid=valid, prior=0.5)
    assert torch.isfinite(clamped)
    assert clamped >= 0
    # Overfit: the loss falls and held-out worlds are read better than a zero predictor.
    train_arr = data.epoch_arrays(
        spec=spec,
        med=med,
        windows=split.train,
        weights=np.ones(len(split.train)),
        epoch=0,
        size=512,
        base_seed=500,
    )
    tens = to_tensors(arrays=train_arr, device="cpu")
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    first_loss = last_loss = 0.0
    started = time.monotonic()
    for step in range(120):
        index = torch.randint(0, 512, (64,))
        loss = batch_loss(net=net, t=tens, index=index)
        opt.zero_grad()
        loss.total.backward()
        opt.step()
        if step < 10:
            first_loss += float((loss.step + loss.size).detach()) / 10
        if step >= 110:
            last_loss += float((loss.step + loss.size).detach()) / 10
    print(f"toy fit: L_u {first_loss:.3f} -> {last_loss:.3f} in {time.monotonic() - started:.0f} s")
    assert last_loss < 0.8 * first_loss, "the encoder does not learn the toy world"
    # Scoring: perfect level is perfect; zero level misses everything and removes no lag error.
    ctx = Context(
        spec=spec,
        med=med,
        split=split,
        groups=np.array(["six", "six", "eight", "eight"]),
        weights=np.ones(len(split.train)),
    )
    perfect = score(
        ctx=ctx, predictor=lambda a: (a.step_true.copy(), np.zeros(len(a.x))), worlds=(200, 201)
    )
    zero = score(
        ctx=ctx,
        predictor=lambda a: (np.zeros_like(a.step_true), np.zeros(len(a.x))),
        worlds=(200, 201),
    )
    allp, allz = perfect["all"], zero["all"]
    assert allp["ceiling_found_2h"] > 0.8, allp["ceiling_found_2h"]
    assert abs(float(allp["found_2h"].split()[0]) - allp["ceiling_found_2h"]) < 1e-3
    assert allp["size_error_median"] < 1e-6
    assert allp["lag_error_repaired"] < 1e-6
    assert allp["false_flag_rate_real_windows"].startswith("0.000")
    assert allz["missed"].startswith("1.000"), allz["missed"]
    assert abs(allz["lag_error_repaired"] - allz["lag_error_unrepaired"]) < 1e-9
    print(f"scoring: perfect found within 2 h {allp['found_2h']}; zero missed {allz['missed']}")
    assert abs(sc.precision_at_rate(recall=1.0, false_rate=0.0) - 1.0) < 1e-9
    assert abs(sc.precision_at_rate(recall=0.5, false_rate=0.1) - 0.01 / (0.01 + 0.098)) < 1e-9
    assert sc.rank_auc(score=np.array([0.1, 0.9, 0.5, 0.4]), label=np.array([0, 1, 1, 0])) == 1.0
    # The abort rule.
    assert should_abort(elapsed=100, epochs_done=2, max_epochs=60, limit=7200) is None
    assert should_abort(elapsed=400, epochs_done=3, max_epochs=60, limit=7200) is not None
    assert should_abort(elapsed=7300, epochs_done=1, max_epochs=60, limit=7200) is not None
    assert should_abort(elapsed=300, epochs_done=3, max_epochs=60, limit=7200) is None
    # The epoch generator is a fresh world per epoch and repeatable per epoch.
    w = np.ones(len(split.train))
    a0 = data.epoch_arrays(
        spec=spec, med=med, windows=split.train, weights=w, epoch=0, size=64, base_seed=9
    )
    a0b = data.epoch_arrays(
        spec=spec, med=med, windows=split.train, weights=w, epoch=0, size=64, base_seed=9
    )
    a1 = data.epoch_arrays(
        spec=spec, med=med, windows=split.train, weights=w, epoch=1, size=64, base_seed=9
    )
    assert np.array_equal(a0.x, a0b.x)
    assert not np.array_equal(a0.x, a1.x)
    print("selftest passed")


def main() -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest", "counts", "timing", "fit", "score"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return
    ctx = build_context()
    if args.command == "counts":
        counts(ctx=ctx)
    elif args.command == "timing":
        timing(ctx=ctx)
    elif args.command == "fit":
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        print(json.dumps(fit(ctx=ctx, device=args.device, out_dir=EXPERIMENT_DIR), indent=2))
    else:
        net = load_net(out_dir=EXPERIMENT_DIR, device=args.device)
        result = score(ctx=ctx, predictor=lambda a: predict(net=net, arrays=a, device=args.device))
        (EXPERIMENT_DIR / "t1_score.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
