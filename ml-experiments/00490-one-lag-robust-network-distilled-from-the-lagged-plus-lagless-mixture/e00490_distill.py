"""Fit a lagged network that is trained towards a mixture teacher, and forecast with it.

The student is experiment 404's A0-inj network (three origin lags, no lagless input, no surprise
inputs) on the injected world of its fit seed, refitted monthly with experiment 397's recency
weights for 6 epochs. What changes is the loss. Each row's loss is the recency weight times

    0.5 * negative log likelihood of the true target
    + 0.5 * negative log likelihood, under the student, of M samples from the teacher's mixture

where the second term is a Monte Carlo estimate of the forward divergence from teacher to student.
Arms:

- `D-K2`: teacher K2-S1 of experiment 454 (two S1 networks and two lagless forecasts of the seed
  pair of cell `c`: c = 0 seeds 0 and 1, c = 1 seeds 0 and 2, c = 2 seeds 1 and 2), teacher term on
  every row.
- `D-Bmix`: teacher B-mix (A0-inj seed `c` and the lagless table of seed `c`), teacher term on
  every row.
- `D-K2-corr`: teacher K2-S1, teacher term only on rows where one of the three lags lies in an
  injected event or a padded real event; other rows use the true target alone.
- `D-0`: no teacher term; reproduces the stored A0-inj recipe through this script.

Teacher mixtures come from forward passes of the stored networks of experiment 404 on the student's
own training rows and world. The teacher networks are in sample on those rows; only the lagless
table is cross-fitted. A separate random generator draws the teacher samples, so the main sequence
(initialisation, batch order) matches experiment 404's A0-inj fit of the same seed.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00490-*/e00490_distill.py fit --arm D-K2 \
    --month 2026-06 --seed 0
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00490-one-lag-robust-network-distilled-from-the-lagged-plus-lagless-mixture"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
PRED_DIR: Final[Path] = WORK_DIR / "predictions"
STORED_MODEL_DIR: Final[Path] = fit.MODEL_DIR
"""Experiment 404's saved networks (the teachers); `fit.MODEL_DIR` is repointed at `MODEL_DIR`."""
ARMS: Final[tuple[str, ...]] = ("D-K2", "D-Bmix", "D-K2-corr", "D-0")
PAIRS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))
TEACHER_WEIGHT: Final[float] = 0.5
TEACHER_SAMPLES: Final[int] = 8
BATCH_TEACHER: Final[int] = 16384
type Mixture = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def use_student_folders() -> None:
    """Point experiment 404's fit and score modules at this experiment's folders."""
    fit.MODEL_DIR = MODEL_DIR
    fit.PRED_DIR = PRED_DIR


def _check_same_scalers(*, saved: dict, scalers: fit.Scalers, lag_stats: np.ndarray) -> None:
    """Stop unless a teacher network standardises its inputs and target as the student does."""
    assert np.allclose(saved["standardiser_mean"], scalers.standardiser.mean)
    assert np.allclose(saved["standardiser_std"], scalers.standardiser.std)
    assert np.allclose(saved["target_mean"], scalers.target_mean)
    assert np.allclose(saved["target_std"], scalers.target_std)
    assert np.allclose(saved["lag_stats"], lag_stats)


@torch.no_grad()
def lagged_head(
    *,
    arm: str,
    month: str,
    seed: int,
    pool: fit.Pool,
    power_np: np.ndarray,
    scalers: fit.Scalers,
    lag_stats: np.ndarray,
    device: str,
) -> Mixture:
    """Return a stored network's mixture (z units) on the rows of `pool` and the world `power_np`.

    `arm` is `A0-inj` or `S1`; S1 reads the surprise at each lag against the lagless median of its
    own seed's table.
    """
    saved = torch.load(
        STORED_MODEL_DIR / arm / f"{month}_s{seed}.pt", map_location=device, weights_only=False
    )
    _check_same_scalers(saved=saved, scalers=scalers, lag_stats=lag_stats)
    net = fit.build_network(arm=arm, n_base=saved["n_base"], n_trust=rows_mod.N_TRUST).to(device)
    net.load_state_dict(saved["state"])
    net.eval()
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    lags, lag_grid = fit.lag_features(
        pool=pool, power=torch.tensor(power_np, device=device), lag_stats=lag_stats, device=device,
        surprise=fit.surprise_array(power=power_np, table=table)
        if arm == fit.SURPRISE_ARM
        else None,
    )  # fmt: skip
    metered = torch.tensor(fit.metered_indicator(), device=device)
    parts = []
    for start in range(0, len(pool), BATCH_TEACHER):
        idx = torch.arange(start, min(start + BATCH_TEACHER, len(pool)), device=device)
        parts.append(
            fit.forward(
                net=net, pool=pool, idx=idx, lags=lags, stack=None, lag_grid=lag_grid,
                scalers=scalers, metered=metered,
            )
        )  # fmt: skip
    return tuple(torch.cat([p[i] for p in parts]) for i in range(3))  # ty: ignore[invalid-return-type]


def lagless_head(
    *, month: str, seed: int, pool: fit.Pool, scalers: fit.Scalers, device: str
) -> Mixture:
    """Return the cross-fitted lagless table of `seed` as a mixture (z units) on `pool`'s rows."""
    arrays = fit.table_arrays(table=pl.read_parquet(lagless.table_path(month=month, seed=seed)))
    raw, present = fit.lagless_in_network_units(
        arrays=arrays,
        primary=pool.primary.cpu().numpy(),
        grid=pool.target.cpu().numpy(),
        mean=scalers.target_mean,
        std=scalers.target_std,
    )
    assert present.all(), f"lagless seed {seed} misses rows of {month}"
    return models.mixture_parts(
        out=torch.tensor(raw, dtype=torch.float32, device=device), components=models.COMPONENTS
    )


def mix_equal(*, heads: list[Mixture]) -> Mixture:
    """Return the equal-weight mixture of the heads (log weights, means, standard deviations)."""
    k = len(heads)
    return (
        torch.cat([h[0] - float(np.log(k)) for h in heads], dim=1),
        torch.cat([h[1] for h in heads], dim=1),
        torch.cat([h[2] for h in heads], dim=1),
    )


def teacher_mixture(
    *, teacher: str, month: str, seed: int, pool: fit.Pool, power_np: np.ndarray,
    scalers: fit.Scalers, lag_stats: np.ndarray, device: str,
) -> Mixture:  # fmt: skip
    """Return the teacher's mixture (z units) for the rows of `pool` in the world `power_np`.

    `teacher` is `K2-S1` (cell `seed` of experiment 454: the S1 networks and lagless tables of
    `PAIRS[seed]`) or `B-mix` (A0-inj of `seed` and the lagless table of `seed`).
    """

    def lagged(*, arm: str, seed: int) -> Mixture:
        return lagged_head(
            arm=arm, month=month, seed=seed, pool=pool, power_np=power_np, scalers=scalers,
            lag_stats=lag_stats, device=device,
        )  # fmt: skip

    def lagless_of(seed: int) -> Mixture:
        return lagless_head(month=month, seed=seed, pool=pool, scalers=scalers, device=device)

    if teacher == "K2-S1":
        assert seed < len(PAIRS), "K2-S1 cells exist for seeds 0 to 2"
        pair = PAIRS[seed]
        heads = [lagged(arm="S1", seed=s) for s in pair] + [lagless_of(s) for s in pair]
    else:
        heads = [lagged(arm="A0-inj", seed=seed), lagless_of(seed)]
    return mix_equal(heads=heads)


def teacher_term(
    *,
    student: Mixture,
    teacher: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """Return, per row, the student's negative log likelihood of samples from the teacher.

    The mean over `TEACHER_SAMPLES` draws from the teacher's mixture estimates the cross entropy,
    which differs from the forward divergence from teacher to student by the teacher's own entropy.
    """
    t_logw, t_mean, t_std = teacher
    rows = len(t_logw)
    component = torch.multinomial(
        t_logw.exp(), TEACHER_SAMPLES, replacement=True, generator=generator
    )
    noise = torch.randn(rows, TEACHER_SAMPLES, device=t_logw.device, generator=generator)
    z = torch.gather(t_mean, 1, component) + torch.gather(t_std, 1, component) * noise
    log_w, mean, std = student
    log_pdf = -0.5 * ((z[..., None] - mean[:, None, :]) / std[:, None, :]) ** 2 - torch.log(
        std[:, None, :]
    )
    log_q = torch.logsumexp(log_w[:, None, :] + log_pdf, dim=-1) - 0.5 * float(np.log(2 * np.pi))
    return -log_q.mean(dim=1)


def fit_student(*, arm: str, month: str, seed: int, device: str, stride: int = 2) -> Path:
    """Fit one student arm on one scored month and seed; save it; return the file."""
    started = time.monotonic()
    use_student_folders()
    train, _test, scalers, arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    power_np, code = fit.world_power(seed=100 + seed)
    event_mask = fit.event_mask_of(code=code, real=score.real_event_mask(), device=device)
    power = torch.tensor(power_np, device=device)
    lags, lag_grid = fit.lag_features(
        pool=train,
        power=power,
        lag_stats=lag_stats,
        device=device,  # ty: ignore[invalid-argument-type]
    )
    teacher = None
    if arm != "D-0":
        teacher = teacher_mixture(
            teacher="B-mix" if arm == "D-Bmix" else "K2-S1", month=month, seed=seed,
            pool=train, power_np=power_np, scalers=scalers, lag_stats=lag_stats, device=device,
        )  # fmt: skip
        print(f"{arm} {month} s{seed}: teacher forward {time.monotonic() - started:.0f} s")
    corrupted = _oracle_any(
        event_mask=event_mask,
        primary=train.primary,
        lag_grid=lag_grid,  # ty: ignore[unresolved-attribute]
    )
    teacher_rows = corrupted.float() if arm == "D-K2-corr" else torch.ones_like(corrupted).float()
    metered = torch.tensor(fit.metered_indicator(), device=device)
    torch.manual_seed(seed)
    net = fit.build_network(arm=arm, n_base=train.base.shape[1], n_trust=rows_mod.N_TRUST).to(
        device
    )  # ty: ignore[unresolved-attribute]
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    generator = torch.Generator(device=device).manual_seed(1000 + seed)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)  # ty: ignore[invalid-argument-type]
    losses = []
    for epoch in range(fit.EPOCHS):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            idx = order[start : start + fit.BATCH]
            out = fit.forward(
                net=net, pool=train, idx=idx, lags=lags, stack=None, lag_grid=lag_grid,  # ty: ignore[invalid-argument-type]
                scalers=scalers, metered=metered,
            )  # fmt: skip
            log_w, mean, std = out
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]  # ty: ignore[unresolved-attribute]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            if teacher is None:
                per_row = nll
            else:
                t_batch = tuple(t[idx] for t in teacher)
                term = teacher_term(student=out, teacher=t_batch, generator=generator)  # ty: ignore[invalid-argument-type]
                share = TEACHER_WEIGHT * teacher_rows[idx]
                per_row = (1.0 - share) * nll + share * term
            w = train.weight[idx]  # ty: ignore[unresolved-attribute]
            loss = (per_row * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
        print(f"{arm} {month} s{seed} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    path = MODEL_DIR / arm / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(), "arm": arm, "month": month, "seed": seed,
            "n_base": train.base.shape[1],  # ty: ignore[unresolved-attribute]
            "target_mean": scalers.target_mean, "target_std": scalers.target_std,
            "standardiser_mean": scalers.standardiser.mean,
            "standardiser_std": scalers.standardiser.std,
            "standardiser_mask": scalers.standardiser.mask, "lag_stats": lag_stats,
            "trust_mean": None, "trust_std": None, "losses": losses, "rows": n,
            "corrupted_rows": int(corrupted.sum()),
            "seconds": time.monotonic() - started, "g0": arrays.g0,
        },
        path,
    )  # fmt: skip
    print(f"{arm} {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


def _oracle_any(
    *, event_mask: torch.Tensor, primary: torch.Tensor, lag_grid: torch.Tensor
) -> torch.Tensor:
    """Return True for rows where any of the three lags lies in an event."""
    size = event_mask.shape[1]
    inside = (lag_grid >= 0) & (lag_grid < size)
    safe = lag_grid.clamp(0, size - 1)
    return (event_mask[primary[:, None], safe] & inside).any(dim=1)


def predict(*, arm: str, month: str, seed: int, worlds: list[int], device: str) -> None:
    """Forecast a scored month on the scoring worlds and the clean world with a saved student."""
    use_student_folders()
    real = score.real_event_mask()
    for world in worlds:
        frame = score.forecast(
            arm=arm, month=month, seed=seed, world=world, device=device, real=real
        )
        target = PRED_DIR / arm / f"{month}_s{seed}_w{world}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        print(f"{arm} {month} s{seed} w{world}: {frame.height:,} rows -> {target}")


def main() -> None:
    """Fit or forecast a student."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, nargs="*", default=[-1, *score.SCORING_SEEDS])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "fit":
        fit_student(arm=args.arm, month=args.month, seed=args.seed, device=args.device)
    else:
        predict(
            arm=args.arm, month=args.month, seed=args.seed, worlds=args.worlds, device=args.device
        )


if __name__ == "__main__":
    main()
