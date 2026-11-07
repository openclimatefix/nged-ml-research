# ruff: noqa: E501, ANN001
"""Shared pieces of experiment 624: paths, the paired twin rows, and the training loops.

Every real row of the six MW Primaries gets two twin rows that share its inputs. The *physics twin*
has an extra +1 MW of PV (the unit curve driven by the row's own CAMS shortwave) subtracted from its
target and trains its own embedding vector (index 16 + j for the j-th MW Primary). The *profile
twin* has an extra +1 MW of a farm's calendar-mean shape (no sky dependence) subtracted from its
target, and trains embedding index 22 + j. The twins add 12 embedding vectors to the network and no
other parameter, so the pairing of an injected and an uninjected row is carried by the embedding.
"""

from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00623_common as k  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00624-pv-probe-contrastive-injection-arms"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFFS: Final = k.CUTOFFS
SEEDS: Final = k.SEEDS
CONDITIONS: Final[tuple[str, ...]] = (
    "control",
    "farm_mw4_r0",
    "farm_mw4_r1",
    "phys_mw4",
    "profile_mw4_r0",
    "shuffled_mw4_r0",
    "cloudlin_strong",
)
"""The held-out worlds each network is trained in (a subset of experiment 621's)."""
N_REAL: Final[int] = 16
PHYS_BASE: Final[int] = 16
PROFILE_BASE: Final[int] = 22
N_EMBED: Final[int] = 28
TWIN_MW: Final[float] = 1.0
PAIR_SCALE: Final[float] = 0.05
"""Scale (in standardised target units) of the paired-difference term's squared error."""
PAIR_WEIGHT: Final[float] = 1.0


def weights_path(*, family: str, cutoff: str, condition: str, seed: int) -> Path:
    """Return where a network of the given family (`a` or `c`) is stored."""
    return WORK / f"pretrained_{family}" / cutoff / f"{condition}_s{seed}.pt"


def plain_weights_path(*, cutoff: str, condition: str, seed: int) -> Path:
    """Return where experiment 621's plain network (16 embedding vectors) is stored."""
    return k.PLAIN_WORK / "pretrained" / cutoff / f"{condition}_s{seed}.pt"


class Fold:
    """The pretraining rows of one cutoff, the kept-row mask, and the injector."""

    def __init__(self, *, cutoff: str) -> None:
        """Build the rows exactly as experiment 623's `fit` does."""
        clean = c.clean_stack()
        refit_t = c.month_start(month=cutoff)
        self.rows = data.build_fold(refit=refit_t, clean=clean)
        self.pre = r448.pretrain_rows(rows=self.rows, refit=refit_t, clean=clean)
        self.inj = k.Injector(
            primary=self.pre.primary,
            target_idx=self.pre.target_idx,
            x=self.pre.x,
            standardiser=self.rows.standardiser,
        )
        self.keep = ~self.inj.gap
        self.six = k.six_indices()
        self.cap = k.effective_capacity_mw()

    def combined(self, *, condition: str, device: str) -> dict[str, torch.Tensor]:
        """Return the real rows and both twins of the six Primaries as GPU tensors, for one world."""
        pre, keep = self.pre, self.keep
        tgt_mean, tgt_std = self.rows.target_mean[pre.primary], self.rows.target_std[pre.primary]
        change = self.inj.change_points(condition=k.parse(name=condition))
        y_world = pre.y.astype(np.float64) + change
        z_real = ((y_world - tgt_mean) / tgt_std)[keep]
        dz_phys = (self.inj.change_points(condition=k.parse(name="phys_mw1")) / tgt_std)[keep]
        dz_prof = (self.inj.change_points(condition=k.parse(name="profile_mw1_r0")) / tgt_std)[keep]
        primary = pre.primary[keep]
        j_of = {p: j for j, p in enumerate(self.six.values())}
        src = np.nonzero(np.isin(primary, list(j_of)))[0]
        j = np.array([j_of[p] for p in primary[src]])
        n = int(keep.sum())
        row_idx = np.concatenate([np.arange(n), src, src])
        emb = np.concatenate([primary, PHYS_BASE + j, PROFILE_BASE + j])
        partner = np.concatenate([primary, primary[src], primary[src]])
        z = np.concatenate([z_real, z_real[src] + dz_phys[src], z_real[src] + dz_prof[src]])
        dz = np.concatenate([np.zeros(n), dz_phys[src], dz_prof[src]])
        is_twin = np.concatenate([np.zeros(n, bool), np.ones(2 * len(src), bool)])
        is_val = np.concatenate([pre.is_val[keep], pre.is_val[keep][src], pre.is_val[keep][src]])
        t = lambda a, dtype: torch.from_numpy(np.ascontiguousarray(a)).to(dtype).to(device)  # noqa: E731
        return {
            "x": torch.from_numpy(pre.x[keep]).to(device),
            "row_idx": t(row_idx, torch.long), "emb": t(emb, torch.long), "partner": t(partner, torch.long),
            "z": t(z, torch.float32), "dz": t(dz, torch.float32), "is_twin": t(is_twin, torch.bool), "is_val": t(is_val, torch.bool),
        }  # fmt: skip


def mixture_mean(*, out: torch.Tensor) -> torch.Tensor:
    """Return the mixture mean (standardised units) of each row."""
    log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
    return (torch.softmax(log_w, dim=-1) * mean).sum(-1)


def train(
    *, model, d: dict[str, torch.Tensor], seed: int, lr: float, pair_weight: float, device: str
) -> tuple[int, float]:
    """Experiment 554's loop over the real rows and both twins, with the paired-difference term."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~d["is_val"]).squeeze(1)
    val_idx = torch.nonzero(d["is_val"]).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    x, ri, emb, partner, z, dz, tw = (
        d["x"],
        d["row_idx"],
        d["emb"],
        d["partner"],
        d["z"],
        d["dz"],
        d["is_twin"],
    )
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            xb = x[ri[b]]
            out = model(xb, emb[b])
            loss = base.mixture_nll(out=out, z=z[b], components=m.COMPONENTS).mean()
            if pair_weight > 0:
                t = tw[b]
                if bool(t.any()):
                    mean_real = mixture_mean(out=model(xb[t], partner[b][t]))
                    diff = mixture_mean(out=out[t]) - mean_real
                    loss = loss + pair_weight * (((diff - dz[b][t]) / PAIR_SCALE) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        val = float(validation_nll(model=model, d=d, idx=val_idx))
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def validation_nll(*, model, d: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    """Return the mean negative log-likelihood on the given rows."""
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                base.mixture_nll(
                    out=model(d["x"][d["row_idx"][i]], d["emb"][i]),
                    z=d["z"][i],
                    components=m.COMPONENTS,
                )
                for i in idx.split(c.BATCH * 2)
            ]
        ).mean()
