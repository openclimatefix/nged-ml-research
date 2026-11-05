# ruff: noqa: E501, E402, PLR0912, ANN001, ANN201, PLR0915, PT018, ANN202, D103
r"""Experiment 567: pretrain, fine-tune, and forecast the neighbour lag-encoder networks.

Rows, cutoffs, recency weights, early stopping, lag columns, block dropout, the expected-input table
and the weather and lag-time assertions are experiment 560's (imported through experiment 566, which
also installs 560's patches on experiment 511's harness). New here: synthetic load-shift worlds that
change the lag columns of the ERA5/CAMS pretraining rows and of the ensemble fine-tuning rows (the
targets stay clean), neighbour lag columns, and the auxiliary flag head.

Commands: `worlds` (CPU), `pretrain`, `coarse`, `select` (rate and auxiliary-weight choice).
Run with the PYTHONPATH of every experiment folder, e.g.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00567-*/e00567_fit.py pretrain --arms N0-r7 N1-r7
```
"""

import argparse
import dataclasses
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_argv = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("worlds", "pretrain", "coarse", "select"))
parser.add_argument("--arms", nargs="+", default=[])
parser.add_argument("--months", nargs="+", default=None)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--aux-weight", type=float, default=0.1)
ARGS = parser.parse_args(_argv[1:])
sys.argv = [_argv[0], "step2"]  # experiment 566 parses its own arguments at import

import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00560_data as d  # ty: ignore[unresolved-import]
import e00566_fit as f566  # ty: ignore[unresolved-import]
import e00567_model as m  # ty: ignore[unresolved-import]
import e00567_world as w  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.argv = _argv
b = f566.b
DD = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = w.WORK
W: Final[Path] = WORK / "work"
NRA_SHARES: Final[Path] = f566.NRA_SHARES
FRESH_LR: Final[float] = 5e-4
HALF_LIFE: Final[float] = 90.0
DAY_US: Final[int] = c.DAY_US
MONTHS: Final[list[str]] = ARGS.months or list(c.MONTHS)
VALIDATION_DAYS: Final[int] = 60
AUX_WEIGHTS: Final[tuple[float, ...]] = (0.05, 0.1, 0.3)
CHUNK: Final[int] = 1_000_000


@dataclass(frozen=True)
class Arm:
    """How an arm builds its rows and network."""

    tag: str
    kind: str
    rate: str
    use_nb: bool
    shared: bool = True
    aux: bool = False
    graph: str = "real"
    corrected: bool = False


def make_arm(*, tag: str) -> Arm:
    """Parse a tag such as `N0-r7`, `N1-r21`, `N2-r7`, `N4-r7`, `N1placebo-r7`, `N1corr-r7`, `N1aux-r7`."""
    kind, rate = tag.split("-")
    assert rate in w.RATES, tag
    table = {
        "N0": {"use_nb": False},
        "N1": {"use_nb": True},
        "N2": {"use_nb": True, "shared": False},
        "N4": {"use_nb": True, "graph": "pairs"},
        "N1placebo": {"use_nb": True, "graph": "placebo"},
        "N1corr": {"use_nb": True, "corrected": True},
        "N1aux": {"use_nb": True, "aux": True},
    }
    return Arm(tag=tag, kind=kind, rate=rate, **table[kind])


class Topology:
    """The neighbour graph and its three versions (real, placebo, event pairs), the held-out split, and worlds' inputs."""

    def __init__(self) -> None:
        import e00402_inject as inj  # ty: ignore[unresolved-import]
        import e00402_worlds as worlds402  # ty: ignore[unresolved-import]

        self.ctx = worlds402.load_context()
        self.labels = list(self.ctx.labels)
        assert self.labels == c.sites()
        self.pool = worlds402.make_pool(ctx=self.ctx)
        self.graph = w.neighbour_graph(ctx=self.ctx)
        self.held = w.held_out_pairs(graph=self.graph)
        self.train_graph, self.test_graph = w.split_graph(graph=self.graph, held=self.held)
        table = pl.read_csv(worlds402.LABELS)
        self.index = {
            "real": w.neighbour_index(graph=self.graph, labels=self.labels),
            "placebo": w.placebo_index(graph=self.graph, labels=self.labels),
            "pairs": w.event_pair_index(labels=self.labels, table=table),
        }
        self.real_share = w.real_share_of_days(table=table, n_primaries=len(self.labels))
        self.inj = inj


TOPO: Topology | None = None


def topo() -> Topology:
    global TOPO
    if TOPO is None:
        TOPO = Topology()
    return TOPO


def corrected_clean() -> np.ndarray:
    """Return the clean series with experiment 435's NRA step (depth factor 1.0) removed."""
    shares = np.load(NRA_SHARES)["f1.0"]
    return (c.clean_stack() - shares).astype(np.float32)


def cut_index(*, month: str) -> int:
    """Return the grid index of the month's first cutoff."""
    return int((c.us(t=c.month_start(month=month)) - c.prep422.GRID_START_US) // c.prep422.STEP_US)


def train_world(*, arm: Arm, month: str, seed: int, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the arm's training world for a fold and seed (events end before the first cutoff)."""
    t = topo()
    spec = w.WorldSpec(kind="train", rate=arm.rate, month=month, seed=seed, gain="corr" if arm.corrected else "mixed")
    return w.build_world(spec=spec, clean=clean, ctx=t.ctx, pool=t.pool, graph=t.train_graph, last=cut_index(month=month))


def heldout_world(*, kind: str, month: str, seed: int, clean: np.ndarray, first: int, last: int, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Return a held-out-pair world (`exact` or `imperfect`) over [first, last)."""
    t = topo()
    spec = w.WorldSpec(kind=kind, rate="r21", month=month, seed=seed, gain=f"held{tag}")
    return w.build_world(spec=spec, clean=clean, ctx=t.ctx, pool=t.pool, graph=t.test_graph, last=last, first=first)


# --- lag columns from a world ---------------------------------------------------------------------


def nb_block(*, world, primary, target_idx, origin_idx, lead, nbr, extra_days=0) -> np.ndarray:
    """Return the 28 neighbour columns (4 slots of 6, then the 4 neighbour indices) for rows."""
    out = np.zeros((len(primary), 28), dtype=np.float32)
    out[:, 24:] = -1.0
    for j in range(m.K):
        nid = nbr[primary, j]
        present = nid >= 0
        blk = b.block_of(
            clean=world, primary=np.clip(nid, 0, None), target_idx=target_idx, origin_idx=origin_idx,
            lead=lead, extra_days=extra_days,
        )[:, :6]  # fmt: skip
        blk[~present, 0], blk[~present, 1], blk[~present, 2] = 0.0, 1.0, d.AGE_MAX_DAY
        blk[~present, 3], blk[~present, 4], blk[~present, 5] = 0.0, 1.0, d.AGE_MAX_WEEK
        out[:, 6 * j : 6 * j + 6] = blk
        out[:, 24 + j] = np.where(present, nid, -1)
    return out


def absent_nb(*, nb: np.ndarray) -> np.ndarray:
    """Return the neighbour columns with every lag value removed (the neighbours' identities stay)."""
    out = nb.copy()
    for j in range(m.K):
        out[:, 6 * j], out[:, 6 * j + 1], out[:, 6 * j + 2] = 0.0, 1.0, d.AGE_MAX_DAY
        out[:, 6 * j + 3], out[:, 6 * j + 4], out[:, 6 * j + 5] = 0.0, 1.0, d.AGE_MAX_WEEK
    return out


def own_block(*, world, primary, target_idx, origin_idx, lead, extra_days=0) -> np.ndarray:
    """Return the own lag columns 37-44 (6 values plus the 2 lag-time indices) for rows."""
    return b.block_of(clean=world, primary=primary, target_idx=target_idx, origin_idx=origin_idx, lead=lead, extra_days=extra_days)


def slots_of(*, primary: np.ndarray, day_idx: np.ndarray, week_idx: np.ndarray, nbr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows x 10) slot Primary and lag index arrays: own day, own week, then each neighbour's day and week."""
    sp = np.empty((len(primary), m.SLOTS), dtype=np.int64)
    si = np.empty_like(sp)
    sp[:, 0] = sp[:, 1] = primary
    si[:, 0], si[:, 1] = day_idx, week_idx
    for j in range(m.K):
        nid = nbr[primary, j]
        sp[:, 2 + 2 * j] = sp[:, 3 + 2 * j] = nid
        si[:, 2 + 2 * j], si[:, 3 + 2 * j] = day_idx, week_idx
    return sp, si


def lag_columns(*, world, code, clean, arr: dict, nbr, aux: bool, extra_days=0) -> dict:
    """Return the own columns (n, 8), neighbour columns (n, 28), and, if `aux`, the flag and fraction labels, in chunks."""
    n = len(arr["primary"])
    own = np.empty((n, 8), dtype=np.float32)
    nbc = np.empty((n, 28), dtype=np.float32)
    flag = np.empty((n, m.SLOTS), dtype=np.int8) if aux else None
    frac = np.empty((n, m.SLOTS), dtype=np.float16) if aux else None
    for a in range(0, n, 2_000_000):
        s = slice(a, a + 2_000_000)
        kw = dict(target_idx=arr["target_idx"][s], origin_idx=arr["origin_idx"][s], lead=arr["lead"][s])
        own[s] = own_block(world=world, primary=arr["primary"][s], extra_days=extra_days, **kw)
        nbc[s] = nb_block(world=world, primary=arr["primary"][s], nbr=nbr, extra_days=extra_days, **kw)
        if aux:
            sp, si = slots_of(primary=arr["primary"][s], day_idx=own[s, 6].astype(np.int64), week_idx=own[s, 7].astype(np.int64), nbr=nbr)
            flag[s], frac[s] = w.aux_labels(world=world, clean=clean, code=code, slot_primary=sp, slot_idx=si)
    return {"own": own, "nb": nbc, "flag": flag, "frac": frac}


def mask_neighbours(*, x: torch.Tensor, ma: torch.Tensor, md: torch.Tensor) -> None:
    """Apply experiment 558's block dropout to the neighbour lag columns, in place (the same masks as the own lags)."""
    for j in range(m.K):
        c0 = m.NB0 + 6 * j
        for use, col, age in ((ma | md, c0, d.AGE_MAX_DAY), (ma, c0 + 3, d.AGE_MAX_WEEK)):
            x[use, col], x[use, col + 1], x[use, col + 2] = 0.0, 1.0, float(age)


def dropout567(*, x: torch.Tensor, flag: torch.Tensor | None, origin_idx: np.ndarray, seed: int) -> dict:
    """Experiment 560's block dropout of the own lags, extended to the neighbour columns and the flag labels."""
    share = b.apply_dropout(x=x, origin_idx=origin_idx, seed=seed)
    ma, md = b.lags.block_dropout_masks(origin_idx=origin_idx, seed=seed)
    ma, md = torch.from_numpy(ma).to(x.device), torch.from_numpy(md).to(x.device)
    mask_neighbours(x=x, ma=ma, md=md)
    if flag is not None:
        day, week = ma | md, ma
        flag[day, 0] = -1
        flag[week, 1] = -1
        for j in range(m.K):
            flag[day, 2 + 2 * j] = -1
            flag[week, 3 + 2 * j] = -1
    return share


class NbGpuMonth(b.LagGpuMonth):
    """The month's rows on the GPU; the lag columns and flag labels follow the current world."""

    aux_flag: torch.Tensor | None = None
    aux_frac: torch.Tensor | None = None

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)  # own-lag dropout is applied here
        idx = refit.index
        origin_idx = (self.mr.origin_us[idx] - c.prep422.GRID_START_US) // c.prep422.STEP_US
        ma, md = b.lags.block_dropout_masks(origin_idx=origin_idx, seed=b.CTX["seed"])
        ma, md = torch.from_numpy(ma).to(t["x"].device), torch.from_numpy(md).to(t["x"].device)
        mask_neighbours(x=t["x"], ma=ma, md=md)
        if self.aux_flag is not None:
            sel = torch.from_numpy(idx).to(self.device)
            flag = self.aux_flag[sel].clone()
            day, week = ma | md, ma
            flag[day, 0] = -1
            flag[week, 1] = -1
            for j in range(m.K):
                flag[day, 2 + 2 * j] = -1
                flag[week, 3 + 2 * j] = -1
            t["aux_flag"], t["aux_frac"] = flag, self.aux_frac[sel]
        return t


k.GpuMonth = NbGpuMonth


# --- rows -----------------------------------------------------------------------------------------

_build_month_560, _build_test_560 = k.build_month, b.data.build_test


def widen73(*, x45: np.ndarray, nb: np.ndarray | None = None) -> np.ndarray:
    """Return the rows with the 28 neighbour columns appended (absent unless `nb` is given), in chunks."""
    out = np.empty((len(x45), m.N_INPUTS), dtype=np.float32)
    for a in range(0, len(x45), CHUNK):
        out[a : a + CHUNK, : x45.shape[1]] = x45[a : a + CHUNK]
    if nb is None:
        out[:, 45:] = 0.0
        for j in range(m.K):
            out[:, 45 + 6 * j + 1] = 1.0
            out[:, 45 + 6 * j + 2] = d.AGE_MAX_DAY
            out[:, 45 + 6 * j + 4] = 1.0
            out[:, 45 + 6 * j + 5] = d.AGE_MAX_WEEK
        out[:, 69:] = -1.0
    else:
        out[:, 45:] = nb
    return out


def build_month(*, month, clean):
    """Experiment 560's rows (assertions, example rows), widened to 73 columns with absent neighbours."""
    mr = _build_month_560(month=month, clean=clean)
    mr.x = widen73(x45=mr.x)
    return mr


def row_arrays(mr) -> dict:
    """Return the arrays that address the lag grid for a month's rows."""
    return {
        "primary": mr.primary,
        "target_idx": mr.target_idx,
        "origin_idx": (mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US,
        "lead": mr.lead,
    }


k.build_month = build_month


class Cell:
    """One fold: rows with world lag columns, GPU copy, test rows and their scoring variants."""

    def __init__(self, *, month: str, device: str, corrected: bool = False) -> None:
        self.month, self.device, self.corrected = month, device, corrected
        self.clean = corrected_clean() if corrected else c.clean_stack()
        self.mr = k.build_month(month=month, clean=self.clean)
        self.g = k.GpuMonth(mr=self.mr, device=device)
        self.test = b.data.build_test(month=month, standardiser=self.mr.base.standardiser)
        self.arr = row_arrays(self.mr)
        self.v45 = dict(b.CTX["variants"])
        self.mu_t = torch.from_numpy(self.mr.base.target_mean).to(device)
        self.sd_t = torch.from_numpy(self.mr.base.target_std).to(device)
        self.n_primaries = len(self.mr.base.target_mean)
        self.world_key: tuple | None = None

    def apply_world(self, *, arm: Arm, seed: int) -> None:
        """Rewrite the rows' own and neighbour lag columns (and flag labels) from the arm's training world."""
        key = (arm.rate, arm.graph, arm.aux, seed, arm.corrected)
        if self.world_key == key:
            return
        t = topo()
        power, code = train_world(arm=arm, month=self.month, seed=seed, clean=self.clean)
        cols = lag_columns(world=power, code=code, clean=self.clean, arr=self.arr, nbr=t.index[arm.graph], aux=arm.aux)
        for a in range(0, len(self.g.x), CHUNK):
            s = slice(a, a + CHUNK)
            self.g.x[s, 37:45] = torch.from_numpy(cols["own"][s]).to(self.device)
            self.g.x[s, 45:] = torch.from_numpy(cols["nb"][s]).to(self.device)
        self.g.aux_flag = torch.from_numpy(cols["flag"]).to(self.device) if arm.aux else None
        self.g.aux_frac = torch.from_numpy(cols["frac"]).to(self.device) if arm.aux else None
        share = float(((power != self.clean) & np.isfinite(power))[:, : cut_index(month=self.month)].mean())
        print(f"{self.month} s{seed} {arm.tag}: world rate {arm.rate}, share of Primary half-hours before the cutoff with a changed reading {share:.4f}", flush=True)
        self.world_key = key

    def select(self, *, cut: datetime):
        r = c.select_rows(rows=self.mr, cutoff=cut, half_life_days=HALF_LIFE)
        assert r.report["rows_past_valid_cut"] == 0 and r.report["rows_past_origin_cut"] == 0
        return r

    def fit(self, *, arm: Arm, cut: datetime, seed: int, aux_weight: float) -> torch.nn.Module:
        """Fine-tune the arm at `cut` from its pretrained weights."""
        self.apply_world(arm=arm, seed=seed)
        refit = self.select(cut=cut)
        b.CTX.update(on=False, shift=None, sign=None, block=None, seed=seed, cut=cut)
        t = self.g.refit_tensors(refit=refit)
        model = new_net(arm=arm, n_primaries=self.n_primaries, stage="ft")
        model.table = b.CTX["table_ft"]
        b.load_pretrained(model=model, path=pretrained_path(arm=arm, month=self.month, seed=seed))
        model.to(self.device)
        aux = (t["aux_flag"], t["aux_frac"]) if arm.aux else None
        train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
              lr=c.FINE_TUNE_LR, fresh_lr=b.MIXER_LR, device=self.device, aux=aux, aux_weight=aux_weight)  # fmt: skip
        del t
        return model

    def variants(self, *, arm: Arm, held: dict) -> dict[str, np.ndarray]:
        """Return the test inputs to score: as observed, removed, aged by 1 to 3 days, neighbour lags removed, held-out worlds."""
        nbr = topo().index[arm.graph]
        test = self.test
        kw = dict(primary=test.primary, target_idx=test.target_idx, origin_idx=test.origin_idx, lead=test.lead)
        nb_real = nb_block(world=self.clean, nbr=nbr, **kw)
        out = {"base": widen73(x45=test.x, nb=nb_real)}
        for name in ("removed", "aged1", "aged2", "aged3"):
            extra_days = int(name[-1]) if name.startswith("aged") else 0
            nb = nb_block(world=self.clean, nbr=nbr, extra_days=extra_days, **kw)
            if name == "removed":
                nb = absent_nb(nb=nb)
            out[name] = widen73(x45=self.v45[name], nb=nb)
        out["nbremoved"] = widen73(x45=test.x, nb=absent_nb(nb=nb_real))
        for name, (power, _code) in held.items():
            x45 = test.x.copy()
            x45[:, 37:45] = own_block(world=power, **kw)
            out[name] = widen73(x45=x45, nb=nb_block(world=power, nbr=nbr, **kw))
        return out

    def forecast(self, *, model, arm: Arm, lo: int, hi: int, held: dict) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        """Forecast the window with the lags as observed and with each scoring variant."""
        model.table = b.CTX["table_ft"]
        vs = self.variants(arm=arm, held=held)
        frames = {
            n: b._window_frame(model=model, test=dataclasses.replace(self.test, x=v), mr=self.mr, lo=lo, hi=hi, flag=False, device=self.device)
            for n, v in vs.items()
        }
        return frames.pop("base"), frames


# --- training -------------------------------------------------------------------------------------


def new_net(*, arm: Arm, n_primaries: int, stage: str) -> torch.nn.Module:
    """Return an untrained network for the arm."""
    return m.NeighbourLagMdn(n_primaries=n_primaries, stage=stage, use_nb=arm.use_nb, shared=arm.shared, aux=arm.aux)


def pretrained_path(*, arm: Arm, month: str, seed: int) -> Path:
    p = W / "pretrained" / arm.tag / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def model_path(*, tag: str, month: str, key: str, seed: int) -> Path:
    return W / "models" / tag / f"{month}_{key}_s{seed}.pt"


def aux_loss(*, logit, frac, flag, target, pos_weight: float) -> torch.Tensor:
    """Binary cross-entropy with a positive-class weight on the labelled slots, plus a smooth-L1 term on the signed fraction."""
    keep = flag >= 0
    n = keep.sum().clamp(min=1)
    y = flag.clamp(min=0).float()
    weight = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * weight
    reg = torch.nn.functional.smooth_l1_loss(frac, target.float(), reduction="none")
    return ((bce + reg) * keep).sum() / n


def train(*, model, x, z, w, p, is_val, seed: int, lr: float, fresh_lr: float | None, device: str, aux=None, aux_weight: float = 0.0):
    """Experiment 566's loop; with `aux` = (flag, fraction) labels the loss adds the auxiliary head's term."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    mixer = list(model.mixer_ft.parameters()) if fresh_lr is not None else []
    mixer_ids = {id(q) for q in mixer}
    groups = [{"params": [q for q in model.parameters() if id(q) not in mixer_ids], "lr": lr}]
    if mixer:
        groups.append({"params": mixer, "lr": fresh_lr})
    optimiser = torch.optim.AdamW(groups, weight_decay=c.WEIGHT_DECAY)
    pos_weight = 1.0
    if aux is not None:
        flags = aux[0][train_idx]
        n_pos, n_neg = int((flags == 1).sum()), int((flags == 0).sum())
        pos_weight = float(min(50.0, max(1.0, n_neg / max(n_pos, 1))))
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            bi = order[start : start + c.BATCH]
            if aux is not None and aux_weight > 0:
                out, logit, frac = model.forward_aux(x[bi], p[bi])
                nll = b.base.mixture_nll(out=out, z=z[bi], components=b.m545.COMPONENTS)
                extra = aux_loss(logit=logit, frac=frac, flag=aux[0][bi], target=aux[1][bi], pos_weight=pos_weight)
                loss = (w[bi] * nll).sum() / w[bi].sum() + aux_weight * extra
            else:
                nll = b.row_nll(model=model, x=x[bi], p=p[bi], z=z[bi])
                loss = (w[bi] * nll).sum() / w[bi].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([b.row_nll(model=model, x=x[i], p=p[i], z=z[i]) for i in val_idx.split(c.BATCH * 2)]).mean())
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


# --- commands -------------------------------------------------------------------------------------


def cmd_worlds() -> None:
    """Build every training and held-out world the arms need (CPU only)."""
    t = topo()
    clean = c.clean_stack()
    cclean = corrected_clean()
    print(f"real share of Primary-days with a labelled event (Primary named): {t.real_share:.4f}", flush=True)
    for tag in ARGS.arms:
        arm = make_arm(tag=tag)
        for month in MONTHS:
            for seed in ARGS.seeds:
                started = time.monotonic()
                power, code = train_world(arm=arm, month=month, seed=seed, clean=cclean if arm.corrected else clean)
                n = cut_index(month=month)
                share = w.share_of_days_with_event(code=code, first=n - 730, last=n)
                print(f"{tag} {month} s{seed}: share of Primary-days with a synthetic event in the 730 days before the cutoff {share:.4f} ({time.monotonic() - started:.0f} s)", flush=True)


def cmd_pretrain() -> None:
    """Pretrain each arm on ERA5 and CAMS rows whose lags come from the arm's world, per month and seed."""
    t = topo()
    for month in MONTHS:
        arms = [make_arm(tag=a) for a in ARGS.arms]
        todo = [(a, s) for a in arms for s in ARGS.seeds if not (W / "pretrained" / a.tag / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        for corrected in sorted({a.corrected for a, _ in todo}):
            clean = corrected_clean() if corrected else c.clean_stack()
            rows = b.data.build_fold(refit=refit_t, clean=clean)
            b.CTX["stats"] = d.lag_stats(rows=rows, clean=clean)
            pre = b.r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
            latest = datetime.fromtimestamp((pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC)
            assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
            print(f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})", flush=True)
            cols, origin = d.raw_table(source="pre", cache=b.W / "expected")
            b.CTX["table_pre"] = torch.from_numpy(d.standardised_table(cols=cols, standardiser=rows.standardiser)).to("cuda")
            del cols
            x_gpu = torch.empty((len(pre.x), m.N_INPUTS), dtype=torch.float32, device="cuda")
            for a in range(0, len(pre.x), CHUNK):
                x_gpu[a : a + CHUNK, : b.m.N_BASE] = torch.from_numpy(pre.x[a : a + CHUNK]).to("cuda")
            arr = {"primary": pre.primary, "target_idx": pre.target_idx, "origin_idx": pre.origin_idx, "lead": pre.lead}
            z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
            tens = {"z": torch.from_numpy(z.astype(np.float32)).to("cuda"), "w": torch.ones(len(pre.y), device="cuda"),
                    "p": torch.from_numpy(pre.primary).to("cuda"), "v": torch.from_numpy(pre.is_val).to("cuda")}  # fmt: skip
            n_primaries = len(rows.target_mean)
            col_key: list = [None, None]
            for arm, seed in sorted([(a, s) for a, s in todo if a.corrected == corrected], key=lambda r: (r[1], r[0].rate, r[0].graph, r[0].aux)):
                started = time.monotonic()
                power, code = train_world(arm=arm, month=month, seed=seed, clean=clean)
                key = (arm.rate, arm.graph, arm.aux, seed, arm.corrected)
                if col_key[0] != key:
                    col_key[0], col_key[1] = key, lag_columns(world=power, code=code, clean=clean, arr=arr, nbr=t.index[arm.graph], aux=arm.aux)
                cols = col_key[1]
                prep_s = time.monotonic() - started
                # row-by-row assertions: the neighbour's lag time is the own lag time, at least 24 h before the origin
                lag_t = c.prep422.GRID_START_US + cols["own"][:, 6].astype(np.int64) * c.prep422.STEP_US
                origin_us = c.prep422.GRID_START_US + arr["origin_idx"] * c.prep422.STEP_US
                ok = cols["own"][:, 6] >= 0
                assert (lag_t[ok] <= origin_us[ok] - 24 * 3_600_000_000).all(), "a pretraining lag is within 24 h of the origin"
                for a in range(0, len(cols["own"]), CHUNK):
                    s = slice(a, a + CHUNK)
                    x_gpu[s, 37:45] = torch.from_numpy(cols["own"][s]).to("cuda")
                    x_gpu[s, 45:] = torch.from_numpy(cols["nb"][s]).to("cuda")
                flag = torch.from_numpy(cols["flag"]).to("cuda") if arm.aux else None
                frac = torch.from_numpy(cols["frac"]).to("cuda") if arm.aux else None
                share = dropout567(x=x_gpu, flag=flag, origin_idx=pre.origin_idx, seed=seed)
                torch.manual_seed(seed)
                model = new_net(arm=arm, n_primaries=n_primaries, stage="pre").to("cuda")
                model.table = b.CTX["table_pre"]
                epochs, best = train(model=model, x=x_gpu, z=tens["z"], w=tens["w"], p=tens["p"], is_val=tens["v"], seed=seed,
                                     lr=c.PRETRAIN_LR, fresh_lr=None, device="cuda",
                                     aux=(flag, frac) if arm.aux else None, aux_weight=ARGS.aux_weight)  # fmt: skip
                target = W / "pretrained" / arm.tag / f"{month}_s{seed}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.rename(target)
                print(f"  pretrain {arm.tag} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s (rows {prep_s:.0f} s), dropout {share}", flush=True)
                del model, flag, frac
            del x_gpu, tens, pre, rows
            b.CTX.pop("table_pre", None)
            torch.cuda.empty_cache()


def cmd_coarse() -> None:
    """Fit each arm once per fold at the first cutoff and forecast the month under every scoring variant."""
    for month in MONTHS:
        arms = [make_arm(tag=a) for a in ARGS.arms]
        for corrected in sorted({a.corrected for a in arms}):
            todo = [a for a in arms if a.corrected == corrected]
            cell = Cell(month=month, device="cuda", corrected=corrected)
            cut = c.month_start(month=month)
            lo, hi = c.us(t=cut), c.us(t=c.month_stop(month=month))
            first_day = cut_index(month=month) - 14 * 48
            held = {}
            for seed in ARGS.seeds:
                for kind, name in (("exact", "hx"), ("imperfect", "hi")):
                    held[(name, seed)] = heldout_world(kind=kind, month=month, seed=seed, clean=cell.clean, first=first_day, last=cell.test.target_idx.max().item() + 1, tag=f"fold{month}")
            for arm in todo:
                for seed in ARGS.seeds:
                    out = W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                    if out.exists():
                        continue
                    started = time.monotonic()
                    mfile = model_path(tag=arm.tag, month=month, key="coarse", seed=seed)
                    if mfile.exists():
                        model = new_net(arm=arm, n_primaries=cell.n_primaries, stage="ft")
                        model.load_state_dict(torch.load(mfile, map_location="cpu"))
                        model = model.to("cuda").eval()
                    else:
                        model = cell.fit(arm=arm, cut=cut, seed=seed, aux_weight=ARGS.aux_weight)
                        mfile.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), mfile.with_suffix(".tmp"))
                        mfile.with_suffix(".tmp").rename(mfile)
                    frame, extra = cell.forecast(model=model, arm=arm, lo=lo, hi=hi, held={n: v for (n, s), v in held.items() if s == seed})
                    for name, fr in extra.items():
                        b._atomic_frame(frame=fr, target=W / "seed_preds" / f"{arm.tag}-{name}" / out.name)
                    b._atomic_frame(frame=frame, target=out)
                    print(f"{arm.tag} {month} s{seed}: {frame.height} rows, {time.monotonic() - started:.0f} s", flush=True)
                    del model
            del cell
            torch.cuda.empty_cache()
    c.write_json(obj=b.CHECK, path=WORK / f"checks_coarse_{'-'.join(ARGS.arms)}.json")


if __name__ == "__main__":
    {"worlds": cmd_worlds, "pretrain": cmd_pretrain, "coarse": cmd_coarse}[ARGS.command]()
