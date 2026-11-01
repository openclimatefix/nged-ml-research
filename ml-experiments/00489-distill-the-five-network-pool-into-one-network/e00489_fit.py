# ruff: noqa: E501
"""Fine-tune one student network to match experiment 425's five-network pool (A-pool).

The teacher of a fold is the same-month fine-tuned arm A networks of experiment 425 (seeds 0 to 4,
`work/models/A/<month>/s<seed>`), all trained before the fold's test month, so the scored actuals
never reach the student. The teacher target of a row is the average of the five networks' 11
quantiles in standardised units (the quantile average that A-pool serves). Every teacher is in
sample on the fine-tuning rows; the 0.5 true-target mix and the held-out-week coverage log in
`work/logs/` handle that.

Each student starts from the stored pretrained arm A network of the same seed
(`00425-*/work/pretrained/A/<month>_s<seed>.pt`) and is fine-tuned at 2e-4 on the ENS rows of
experiment 425's fold, with experiment 397's loop shape (batch 16,384, at most 40 epochs, patience
6, gradient clip 1.0). The loss of a row is, in standardised units,

    w * [NLL(y) + alpha * P(q_student, q_teacher) + (1 - alpha) * P(q_student, y)]

where `w` is the recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)` of the fold, `NLL` is the
student's own mixture likelihood on the true target, and `P` is the pinball loss summed over the 11
levels (against the teacher's 11 quantiles as atoms of the scorers' level masses in the first term,
and against the true value in the second). The student's
quantiles are differentiable through the implicit function theorem. Early stopping sees the same
loss on the held-out weeks, unweighted.

Arms (`work/seed_preds/<arm>/<month>_s<seed>.parquet`, raw quantiles as in experiment 425):

- `Dpool50`: alpha 0.5;
- `Dpool80`: alpha 0.8;
- `DpoolERA5`: alpha 0.5, plus one batch per step of the ERA5-and-CAMS rows that pretrained the
  teacher, with the teacher-quantile term only (no true target) at equal row weights.

```bash
source .e489env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00489-*/e00489_fit.py --months 2025-10 --seeds 0
```
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Path = (
    ML_EXPERIMENTS_DATA_DIR / "00489-distill-the-five-network-pool-into-one-network" / "work"
)
TEACHER_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
STUDENT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
FINE_TUNE_LR: Final[float] = fit425.FINE_TUNE_LR
LEVELS: Final[torch.Tensor] = torch.tensor(models.LEVELS, dtype=torch.float32)
MASSES: Final[torch.Tensor] = torch.tensor(common.ATOM_MASSES, dtype=torch.float32)
FIT_CHUNK: Final[int] = 65_536


@dataclass(frozen=True)
class Arm:
    """One student arm: the weight of the teacher term, and whether ERA5 rows are added."""

    name: str
    alpha: float
    era5: bool


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("Dpool50", alpha=0.5, era5=False),
        Arm("Dpool80", alpha=0.8, era5=False),
        Arm("DpoolERA5", alpha=0.5, era5=True),
    )
}


def teacher_networks(*, month: str, fold: prep.EnsFold) -> list[fit397.Fitted]:
    """Return the five stored fine-tuned arm A networks of a month, after checking the fold matches."""
    nets = []
    for seed in TEACHER_SEEDS:
        fitted, standardiser, t_mean, t_std = heads470.load_network(month=month, seed=seed)
        assert np.allclose(standardiser.mean, fold.standardiser.mean, atol=1e-5)
        assert np.allclose(standardiser.std, fold.standardiser.std, atol=1e-5)
        assert np.allclose(t_mean, fold.target_mean)
        assert np.allclose(t_std, fold.target_std)
        nets.append(fitted)
    return nets


@torch.no_grad()
def teacher_quantiles(
    *, nets: list[fit397.Fitted], x: np.ndarray, primary: np.ndarray, device: str
) -> np.ndarray:
    """Return the mean over the networks of their 11 sorted standardised quantiles, per row."""
    out = np.empty((len(x), len(models.LEVELS)), dtype=np.float32)
    k = nets[0].setting.components
    for start in range(0, len(x), FIT_CHUNK):
        xb = torch.from_numpy(x[start : start + FIT_CHUNK]).to(device)
        pb = torch.from_numpy(primary[start : start + FIT_CHUNK]).to(device)
        total = torch.zeros(len(xb), len(models.LEVELS), dtype=torch.float64, device=device)
        for net in nets:
            raw = fit397.forward(model=net.model, x=xb, primary=pb).double()
            log_w, mean, std = models.mixture_parts(out=raw, components=k)
            q = models.mixture_quantiles(log_w=log_w, mean=mean, std=std)
            total += torch.sort(q, dim=1).values
        out[start : start + FIT_CHUNK] = (total / len(nets)).float().cpu().numpy()
    return out


def student_quantiles(*, out: torch.Tensor, components: int) -> torch.Tensor:
    """Return the mixture's 11 quantiles with gradients, by the implicit function theorem.

    The bisection runs without gradients. With `F(q; theta)` the mixture CDF, the derivative of the
    solution of `F = level` is `dq/dtheta = -dF/dtheta / f(q)`, built here as `q0 - (F - F.detach()) / f`.
    """
    log_w, mean, std = models.mixture_parts(out=out, components=components)
    with torch.no_grad():
        q0 = models.mixture_quantiles(
            log_w=log_w.double(), mean=mean.double(), std=std.double()
        ).float()
    w = log_w.exp()
    z = (q0[..., None] - mean[:, None, :]) / std[:, None, :]
    cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z / float(np.sqrt(2.0))))).sum(-1)
    with torch.no_grad():
        pdf = (
            w[:, None, :] * torch.exp(-0.5 * z**2) / (std[:, None, :] * float(np.sqrt(2.0 * np.pi)))
        ).sum(-1)
    return q0 - (cdf - cdf.detach()) / pdf.clamp_min(1e-4)


def pinball(*, q: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the pinball loss of the quantiles `q` summed over the levels, against a point `target`."""
    d = target - q
    return torch.maximum(levels * d, (levels - 1.0) * d).sum(-1)


def pinball_to_teacher(
    *, q: torch.Tensor, teacher: torch.Tensor, levels: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    """Return the expected pinball loss of `q` against the teacher's distribution, summed over levels.

    The teacher's distribution is its 11 quantiles as atoms of probability `masses` (the mass of
    each level, as the scorers use). A pinball loss against one teacher value is asymmetric, so it
    pulls a tail quantile of the student away from the teacher's; against the whole distribution it
    is a proper score, minimised when each student quantile equals the teacher's at that level.
    """
    d = teacher[:, None, :] - q[:, :, None]
    loss = torch.maximum(levels[None, :, None] * d, (levels[None, :, None] - 1.0) * d)
    return (loss * masses[None, None, :]).sum(-1).sum(-1)


def row_loss(
    *,
    out: torch.Tensor,
    z: torch.Tensor,
    teacher: torch.Tensor,
    alpha: float,
    components: int,
    levels: torch.Tensor,
    masses: torch.Tensor,
) -> torch.Tensor:
    """Return NLL(y) + alpha * pinball(q, teacher) + (1 - alpha) * pinball(q, y) per row."""
    nll = models.mixture_nll(out=out, z=z, components=components)
    q = student_quantiles(out=out, components=components)
    return (
        nll
        + alpha * pinball_to_teacher(q=q, teacher=teacher, levels=levels, masses=masses)
        + (1.0 - alpha) * pinball(q=q, target=z[:, None], levels=levels)
    )


def train_student(
    *,
    model: torch.nn.Module,
    setting: fit397.Setting,
    t: dict[str, torch.Tensor],
    alpha: float,
    era: dict[str, torch.Tensor] | None,
    seed: int,
) -> tuple[int, float]:
    """Fine-tune in place with experiment 397's loop shape and the composite loss; return epochs, best."""
    torch.manual_seed(seed)
    device = t["x"].device
    levels = LEVELS.to(device)
    masses = MASSES.to(device)
    k = setting.components
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    era_train = torch.nonzero(~era["v"]).squeeze(1) if era is not None else None
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(fit397.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), fit397.BATCH):
            b = order[start : start + fit397.BATCH]
            out = fit397.forward(model=model, x=t["x"][b], primary=t["p"][b])
            loss = (
                t["w"][b]
                * row_loss(
                    out=out,
                    z=t["z"][b],
                    teacher=t["q"][b],
                    alpha=alpha,
                    components=k,
                    levels=levels,
                    masses=masses,
                )
            ).sum() / t["w"][b].sum()
            if era is not None:
                assert era_train is not None
                eb = era_train[torch.randint(len(era_train), (fit397.BATCH,), device=device)]
                out_e = fit397.forward(model=model, x=era["x"][eb], primary=era["p"][eb])
                q_e = student_quantiles(out=out_e, components=k)
                loss = (
                    loss
                    + pinball_to_teacher(
                        q=q_e, teacher=era["q"][eb], levels=levels, masses=masses
                    ).mean()
                )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit397.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        row_loss(
                            out=fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]),
                            z=t["z"][b],
                            teacher=t["q"][b],
                            alpha=alpha,
                            components=k,
                            levels=levels,
                            masses=masses,
                        )
                        for b in val_idx.split(fit397.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= fit397.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


@torch.no_grad()
def coverage(
    *, model: torch.nn.Module, k: int, t: dict[str, torch.Tensor], rows: torch.Tensor
) -> dict:
    """Return the share of true targets below each tail quantile of a network, on the given rows."""
    below = []
    for b in rows.split(fit397.BATCH * 2):
        raw = fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]).double()
        log_w, mean, std = models.mixture_parts(out=raw, components=k)
        q = torch.sort(models.mixture_quantiles(log_w=log_w, mean=mean, std=std), dim=1).values
        below.append((t["z"][b].double()[:, None] <= q).float())
    share = torch.cat(below).mean(0)
    return {
        f"q{int(100 * lev):02d}": float(share[i])
        for i, lev in enumerate(models.LEVELS)
        if lev in (0.05, 0.95, 0.99)
    }


def run_month(*, month: str, arms: list[Arm], seeds: list[int], device: str) -> None:
    """Build the fold and the teacher targets once, then fit each missing arm and seed."""
    todo = [
        (a, s)
        for a in arms
        for s in seeds
        if not final_path(arm=a.name, month=month, seed=s).exists()
    ]
    if not todo:
        return
    refit = base.refit_of(month=month)
    fold = prep.build_ens_fold(refit=refit)
    stored = json.loads((fit425.OUT_DIR / "logs" / f"{month}.json").read_text())
    assert len(fold.x) == stored["fine_tune_rows"], "fold rows differ from experiment 425's"
    assert int(fold.is_val.sum()) == stored["fine_tune_val_rows"], "held-out weeks differ"
    nets = list(teacher_networks(month=month, fold=fold))
    for n in nets:
        n.model.to(device)
    started = time.monotonic()
    teacher_q = teacher_quantiles(nets=nets, x=fold.x, primary=fold.primary, device=device)
    print(
        f"fold {month}: {len(fold.x):,} rows, teacher pass {time.monotonic() - started:.0f} s",
        flush=True,
    )
    t = fit425.tensors(
        x=fold.x,
        y=fold.y,
        primary=fold.primary,
        weight=fold.weight,
        is_val=fold.is_val,
        fold=fold,
        device=device,
    )
    t["q"] = torch.from_numpy(teacher_q).to(device)
    era = None
    if any(a.era5 for a, _ in todo):
        pre, _ = prep.fold_pretrain(fold=fold, refit=refit)
        era_q = teacher_quantiles(nets=nets, x=pre.x, primary=pre.primary, device=device)
        era = {
            "x": torch.from_numpy(pre.x).to(device),
            "p": torch.from_numpy(pre.primary).to(device),
            "q": torch.from_numpy(era_q).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }
        del pre, era_q
        print(f"  ERA5 rows {len(era['x']):,}", flush=True)
    log_path = WORK_DIR / "logs" / f"{month}.json"
    log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
    val_rows = torch.nonzero(t["v"]).squeeze(1)
    torch.manual_seed(0)
    train_rows = torch.nonzero(~t["v"]).squeeze(1)
    train_rows = train_rows[torch.randperm(len(train_rows), device=device)[:400_000]]
    k = nets[0].setting.components
    log["teacher_s0_coverage"] = {
        "held_out_weeks": coverage(model=nets[0].model, k=k, t=t, rows=val_rows),
        "training_rows": coverage(model=nets[0].model, k=k, t=t, rows=train_rows),
    }
    for arm, seed in todo:
        started = time.monotonic()
        torch.manual_seed(seed)
        model = fit425.new_model(arm=fit425.ARMS["A"], fold=fold).to(device)
        model.load_state_dict(
            torch.load(
                fit425.pretrained_path(arm=fit425.ARMS["A"], cut=True, month=month, seed=seed),
                map_location=device,
            )
        )
        setting = replace(fit425.SETTING, lr=FINE_TUNE_LR)
        epochs, best = train_student(
            model=model,
            setting=setting,
            t=t,
            alpha=arm.alpha,
            era=era if arm.era5 else None,
            seed=seed,
        )
        fitted = fit397.Fitted(
            model=model, setting=setting, arm="plain_pooled", seed=seed, epochs=epochs, val_nll=best
        )
        fit397.save_fits(
            fits={None: fitted},
            fold=fold,
            target=WORK_DIR / "models" / arm.name / month / f"s{seed}",
        )
        q, _ = fit397.forecast_month(
            fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
        )
        common.atomic_write(frame=q, target=final_path(arm=arm.name, month=month, seed=seed))
        seconds = round(time.monotonic() - started)
        log[f"{arm.name}_s{seed}"] = {
            "epochs": epochs,
            "val_loss": best,
            "seconds": seconds,
            "coverage_held_out_weeks": coverage(model=model, k=k, t=t, rows=val_rows),
            "coverage_training_rows": coverage(model=model, k=k, t=t, rows=train_rows),
        }
        print(
            f"{arm.name} {month} s{seed}: {epochs} epochs, val loss {best:.4f}, {seconds} s",
            flush=True,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=1))
        tmp.replace(log_path)
    del t, era
    torch.cuda.empty_cache()


def final_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where a student's raw quantile forecast is stored."""
    return WORK_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def main() -> None:
    """Fit the named arms on the named months and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months", nargs="+", default=["2026-06", "2026-01", "2025-10", "2026-03", "2025-12"]
    )
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(STUDENT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", type=int, default=0, help="epochs cap; outputs under work/smoke")
    args = parser.parse_args()
    if args.smoke:
        global WORK_DIR  # noqa: PLW0603
        WORK_DIR = WORK_DIR / "smoke"
        fit397.MAX_EPOCHS = args.smoke
    for month in args.months:
        run_month(
            month=month, arms=[ARMS[a] for a in args.arms], seeds=args.seeds, device=args.device
        )


if __name__ == "__main__":
    main()
