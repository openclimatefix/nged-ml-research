# ruff: noqa: E501, E402, PLR0912, ANN001, ANN201, PLR0915, PT018
r"""Experiment 568: pretraining, step 2, and the coarse-refit arms T0 to T5.

Rows, cutoffs, early stopping, seeds, the origin-time weather assertions, the training loop, and the
injected-shift world are experiments 511's and 559's (imported, which installs 559's assertions on
experiment 511's harness; the arm A1 is loaded only for that). The base recipe is experiment 559's A1
schedule: ERA5/CAMS pretraining, a step 2 that trains a fresh mixer and the encoders on every row up
to the fold's first cutoff, then a fine-tune with the 90-day half-life.

Arms: T0 MLP mixer; T1 attention with the multi-variable lead-and-spread token; T2 attention without it
(lead on the query token); T3 attention with 2 m temperature spread alone; T4 T0 with the spread block
appended; T5 T1 with the spread block shuffled between days of the same Primary and month.

Commands: `pretrain` (attention encoders, one per fold and seed), `step2`, `coarse` (one fine-tune per
fold and seed at the first cutoff, scored over the whole month), `stress` (injected 7-day shift,
fine-tuned at day 14 on metered and shifted targets, seed 0 only), `attention` (query-to-token weights).
Run with the PYTHONPATH of every experiment folder.
"""

import argparse
import copy
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

_argv = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("pretrain", "step2", "coarse", "stress", "attention"))
parser.add_argument("--arms", nargs="+", default=["T0", "T0z", "T1", "T2", "T3", "T4", "T5"])
parser.add_argument("--months", nargs="+", default=None)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ARGS = parser.parse_args(_argv[1:])
sys.argv = [_argv[0], "fit", "--arm", "A1"]  # experiment 559 parses its own arguments at import

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import e00559_fit as f559  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00568_model as model_lib
import e00568_spread as spread_lib
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.argv = _argv
NAME: Final[str] = "00568-attention-mixer-modular-network-with-lead-and-spread-token"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
W: Final[Path] = WORK / "work"
W554: Final[Path] = f559.W554
MONTHS: Final[list[str]] = ARGS.months or list(c.MONTHS)
FRESH_SEED: Final[int] = f559.FRESH_SEED
FRESH_LR: Final[float] = f559.FRESH_LR
UNIFORM: Final[float] = f559.UNIFORM_HALF_LIFE
HALF_LIFE: Final[float] = 90.0
LEAD_BANDS: Final[list[tuple[str, float, float]]] = [
    ("0-6 h", 0, 6), ("6-24 h", 6, 24), ("24-48 h", 24, 48), ("48-72 h", 48, 72),
    ("72-120 h", 72, 120), ("120-240 h", 120, 240), ("240-336 h", 240, 1e9),
]  # fmt: skip
CHECK: dict = {"months": {}}


@dataclass(frozen=True)
class Arm:
    """How an arm builds its network."""

    tag: str
    kind: str  # "attn" or "mlp"
    spread: str = "multi"  # attention: multi, temp, or none
    with_block: bool = False  # mlp: append the spread block
    placebo: bool = False
    zero_stored: bool = False  # mlp: zero M0's 7 stored spread columns (a spread-free MLP)


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("T0", "mlp"),
        Arm("T0z", "mlp", zero_stored=True),
        Arm("T1", "attn"),
        Arm("T2", "attn", spread="none"),
        Arm("T3", "attn", spread="temp"),
        Arm("T4", "mlp", with_block=True),
        Arm("T5", "attn", placebo=True),
    )
}
_T1_PARAMS: Final[int] = model_lib.n_parameters(
    model_lib.AttentionMdn(n_primaries=16, stage="ft", spread="multi").mixer_ft
)
HIDDEN: Final[dict[bool, int]] = {
    wb: model_lib.matched_hidden(target=_T1_PARAMS, with_block=wb) for wb in (False, True)
}


for _wb, _h in HIDDEN.items():
    _n = model_lib.n_parameters(model_lib.MlpMdn(n_primaries=16, stage="ft", hidden=_h, with_block=_wb).mixer_ft)
    assert abs(_n / _T1_PARAMS - 1) < 0.10, (_wb, _n, _T1_PARAMS)


def new_net(*, arm: Arm, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return an untrained network of the arm's kind."""
    if arm.kind == "mlp":
        return model_lib.MlpMdn(n_primaries=n_primaries, stage=stage, hidden=HIDDEN[arm.with_block], with_block=arm.with_block, zero_stored=arm.zero_stored)
    return model_lib.AttentionMdn(n_primaries=n_primaries, stage=stage, spread=arm.spread)


def pretrained_path(*, arm: Arm, month: str, seed: int) -> Path:
    """Return the pretrained weights: this experiment's attention encoders, or experiment 554's M0."""
    p = (W / "pretrained" / "ATT" if arm.kind == "attn" else W554 / "pretrained" / "M0") / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def step2_path(*, arm: Arm, month: str, seed: int) -> Path:
    """Return the arm's saved step-2 weights."""
    return W / "step2" / arm.tag / f"{month}_s{seed}.pt"


def model_path(*, arm: Arm, month: str, key: str, seed: int) -> Path:
    """Return a saved fine-tuned model."""
    return W / "models" / arm.tag / f"{month}_{key}_s{seed}.pt"


def atomic_save(*, obj, target: Path) -> None:
    """Write a state dict atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(target)


def fresh_mixer(*, model: torch.nn.Module) -> None:
    """Give the model's fine-tuning mixer a seed-12345 initialisation (experiment 554's recipe)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FRESH_SEED)
        fresh = model.make_mixer_ft()  # ty: ignore[call-non-callable]
    model.mixer_ft.load_state_dict(fresh.state_dict())  # ty: ignore[unresolved-attribute]


def load_pretrained_fresh(*, model: torch.nn.Module, path: Path) -> None:
    """Load pretrained weights without the fine-tuning mixer, then give the model a fresh one."""
    raw = torch.load(path, map_location="cpu")
    state = {a: v for a, v in raw.items() if not a.startswith("mixer_ft.")}
    missing = model.load_state_dict(state, strict=False)
    assert all(a.startswith("mixer_ft.") for a in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    fresh_mixer(model=model)


# --- the spread block on top of experiment 559's row builders -------------------------------------

BLOCKS: dict[str, spread_lib.SpreadBlock] = {}
_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Experiment 559's month rows (assertions included) with the 7 spread-block columns appended."""
    mr = _build_month(month=month, clean=clean)
    sb = spread_lib.SpreadBlock(month=month, mr=mr, placebo=False)
    origin_idx = ((mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)
    block = sb.block(x37=mr.x, standardiser=mr.base.standardiser, primary=mr.primary,
                     origin_idx=origin_idx, valid_idx=mr.target_idx, describe=True)  # fmt: skip
    mr.x = np.concatenate([mr.x, block], axis=1)
    BLOCKS[month] = sb
    CHECK["months"].setdefault(month, {})["train_example"] = sb.example
    print(month, "spread block |", sb.example, flush=True)
    return mr


def build_test(*, month, standardiser):
    """Experiment 559's test rows (assertions included) with the 7 spread-block columns appended."""
    test = _build_test(month=month, standardiser=standardiser)
    sb = BLOCKS[month]
    block = sb.block(x37=test.x, standardiser=standardiser, primary=test.primary,
                     origin_idx=test.origin_idx, valid_idx=test.target_idx, describe=True)  # fmt: skip
    CHECK["months"][month]["test_example"] = sb.example
    print(month, "test spread block |", sb.example, flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, block], axis=1))


k.build_month, data.build_test = build_month, build_test


def placebo_rows(*, month: str, mr, test):
    """Return copies of the month's rows and test rows with the spread block shuffled between days."""
    sb = BLOCKS[month]
    sb.placebo = True
    try:
        origin_idx = ((mr.origin_us - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)
        mr2 = copy.copy(mr)
        mr2.x = np.concatenate([mr.x[:, :37], sb.block(x37=mr.x[:, :37], standardiser=mr.base.standardiser,
                                primary=mr.primary, origin_idx=origin_idx, valid_idx=mr.target_idx, describe=True)], axis=1)  # fmt: skip
        CHECK["months"][month]["placebo_train_example"] = sb.example
        te = test.x
        test2 = dataclasses.replace(test, x=np.concatenate([te[:, :37], sb.block(x37=te[:, :37], standardiser=mr.base.standardiser,
                                    primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx, describe=True)], axis=1))  # fmt: skip
        CHECK["months"][month]["placebo_test_example"] = sb.example
    finally:
        sb.placebo = False
    return mr2, test2


class Cell:
    """One fold: its rows, test rows, and GPU copies (real and placebo spread blocks)."""

    def __init__(self, *, month: str, device: str) -> None:
        """Build the month once."""
        self.month, self.device = month, device
        self.clean = c.clean_stack()
        self.mr = k.build_month(month=month, clean=self.clean)
        self.test = data.build_test(month=month, standardiser=self.mr.base.standardiser)
        self.g = k.GpuMonth(mr=self.mr, device=device)
        self.placebo: tuple | None = None
        self.n_primaries = len(self.mr.base.target_mean)

    def rows_for(self, *, arm: Arm):
        """Return (month rows, GPU month, test rows) with the arm's spread block."""
        if not arm.placebo:
            return self.mr, self.g, self.test
        if self.placebo is None:
            mr2, test2 = placebo_rows(month=self.month, mr=self.mr, test=self.test)
            self.placebo = (mr2, k.GpuMonth(mr=mr2, device=self.device), test2)
        return self.placebo[0], self.placebo[1], self.placebo[2]

    def refit(self, *, arm: Arm, cut: datetime, half_life: float, shift: np.ndarray | None = None,
              sign: int = 0, block: tuple[int, int] | None = None):  # fmt: skip
        """Return the cutoff's selected rows and their GPU tensors."""
        mr, g, _ = self.rows_for(arm=arm)
        refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=half_life)
        assert refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
        shift_z = None
        if shift is not None and block is not None:
            inj = (mr.valid_us[refit.index] >= block[0]) & (mr.valid_us[refit.index] < block[1]) & ~refit.is_val
            prim = mr.primary[refit.index]
            shift_z = np.where(inj, sign * shift[prim] / mr.base.target_std[prim], 0.0)
        return refit, g.refit_tensors(refit=refit, shift_z=shift_z)


def pretrain(*, device: str) -> None:
    """Pretrain the attention encoders and the three-token attention mixer on ERA5 and CAMS rows."""
    clean = c.clean_stack()
    for month in MONTHS:
        todo = [s for s in ARGS.seeds if not (W / "pretrained" / "ATT" / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        latest = datetime.fromtimestamp((pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC)
        assert latest <= refit_t - c.regimes.REFIT_CUTOFF_AGE, (latest, refit_t)
        print(f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})", flush=True)
        x = np.concatenate([pre.x, np.zeros((len(pre.x), 7), dtype=np.float32)], axis=1)
        z = (pre.y - rows.target_mean[pre.primary]) / rows.target_std[pre.primary]
        t = {"x": torch.from_numpy(x).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
             "w": torch.ones(len(pre.y), device=device), "p": torch.from_numpy(pre.primary).to(device),
             "v": torch.from_numpy(pre.is_val).to(device)}  # fmt: skip
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = new_net(arm=ARMS["T1"], n_primaries=len(rows.target_mean), stage="pre").to(device)
            epochs, best = f559.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                                      lr=c.PRETRAIN_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False,
                                      anchor=None, lam=0.0, device=device)  # fmt: skip
            atomic_save(obj=model.state_dict(), target=W / "pretrained" / "ATT" / f"{month}_s{seed}.pt")
            print(f"  pretrain ATT {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s", flush=True)
        del t, pre, rows
        torch.cuda.empty_cache()


def step2(*, device: str) -> None:
    """Train and save each arm's step-2 weights: the encoders and a fresh mixer on every row up to the first cutoff."""
    for month in MONTHS:
        todo = [(a, s) for a in ARGS.arms for s in ARGS.seeds if not step2_path(arm=ARMS[a], month=month, seed=s).exists()]
        if not todo:
            continue
        cell = Cell(month=month, device=device)
        cut = c.month_start(month=month)
        for tag, seed in todo:
            arm = ARMS[tag]
            refit, t = cell.refit(arm=arm, cut=cut, half_life=UNIFORM)
            assert float(t["w"].min()) > 0.999, "step 2 must have no recency weighting"
            started = time.monotonic()
            model = new_net(arm=arm, n_primaries=cell.n_primaries).to(device)
            load_pretrained_fresh(model=model, path=pretrained_path(arm=arm, month=month, seed=seed))
            model.to(device)
            epochs, best = f559.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                                      lr=c.FINE_TUNE_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=FRESH_LR, frozen=False,
                                      anchor=None, lam=0.0, device=device)  # fmt: skip
            atomic_save(obj=model.state_dict(), target=step2_path(arm=arm, month=month, seed=seed))
            c.write_json(obj={**refit.report, "epochs": epochs, "val_nll": best, "rows": len(refit.index)},
                         path=W / "logs" / "step2" / tag / f"{month}_s{seed}.json")  # fmt: skip
            print(f"step 2 {tag} {month} s{seed}: {len(refit.index):,} rows, {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s", flush=True)
            del t, model
            torch.cuda.empty_cache()
        del cell
        torch.cuda.empty_cache()


def fine_tune(*, cell: Cell, arm: Arm, cut: datetime, seed: int, shift=None, sign: int = 0, block=None) -> torch.nn.Module:
    """Fine-tune from the arm's step-2 weights at one cutoff."""
    refit, t = cell.refit(arm=arm, cut=cut, half_life=HALF_LIFE, shift=shift, sign=sign, block=block)
    model = new_net(arm=arm, n_primaries=cell.n_primaries).to(cell.device)
    sp = step2_path(arm=arm, month=cell.month, seed=seed)
    assert sp.exists(), f"run the step2 command first: {sp}"
    model.load_state_dict(torch.load(sp, map_location="cpu"))
    model.to(cell.device)
    f559.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
               lr=c.FINE_TUNE_LR, max_epochs=c.MAX_EPOCHS, fresh_lr=None, frozen=False,
               anchor=None, lam=0.0, device=cell.device)  # fmt: skip
    del t
    return model


def window(*, cell: Cell, arm: Arm, model: torch.nn.Module, lo: int, hi: int) -> pl.DataFrame:
    """Return the forecast of test origins in [lo, hi)."""
    mr, _, test = cell.rows_for(arm=arm)
    frame = f511.window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=cell.device)
    assert frame is not None
    return frame


def coarse(*, device: str) -> None:
    """Fit each arm once per fold at the first cutoff and forecast the whole month."""
    for month in MONTHS:
        todo = [(a, s) for a in ARGS.arms for s in ARGS.seeds
                if not (W / "seed_preds" / a / f"{month}_s{s}.parquet").exists()]  # fmt: skip
        if not todo:
            continue
        cell = Cell(month=month, device=device)
        cut = c.month_start(month=month)
        lo, hi = c.us(t=cut), c.us(t=c.month_stop(month=month))
        for tag, seed in todo:
            arm = ARMS[tag]
            started = time.monotonic()
            model = fine_tune(cell=cell, arm=arm, cut=cut, seed=seed)
            atomic_save(obj=model.state_dict(), target=model_path(arm=arm, month=month, key="coarse", seed=seed))
            frame = window(cell=cell, arm=arm, model=model, lo=lo, hi=hi)
            f511.atomic_frame(frame=frame, target=W / "seed_preds" / tag / f"{month}_s{seed}.parquet")
            print(f"{tag} {month} s{seed}: {frame.height} rows, {time.monotonic() - started:.0f} s", flush=True)
            del model
            torch.cuda.empty_cache()
        del cell
        torch.cuda.empty_cache()
    c.write_json(obj=CHECK, path=WORK / f"checks_coarse_{'-'.join(ARGS.arms)}.json")


def stress(*, device: str) -> None:
    """Injected 7-day shift: fine-tune at day 14 on metered targets and on targets shifted up and down (seed 0)."""
    for month in MONTHS:
        todo = [a for a in ARGS.arms if not (W / "stress" / a / f"{month}_n.parquet").exists()]
        if not todo:
            continue
        cell = Cell(month=month, device=device)
        start = c.month_start(month=month)
        end = start + timedelta(days=f511.STRESS_END_DAY)
        shift = c.shift_sizes(clean=cell.clean, cutoff=end)
        block = (c.us(t=end - timedelta(days=f511.STRESS_BLOCK_DAYS)), c.us(t=end) - c.CUTOFF_AGE_US)
        lo, hi = c.us(t=end), c.us(t=end + timedelta(days=7))
        for tag in todo:
            arm = ARMS[tag]
            for key, sign in (("ctrl", 0), ("p", 1), ("n", -1)):
                out = W / "stress" / tag / f"{month}_{key}.parquet"
                if out.exists():
                    continue
                started = time.monotonic()
                model = fine_tune(cell=cell, arm=arm, cut=end, seed=0, shift=shift if sign else None, sign=sign, block=block if sign else None)
                f511.atomic_frame(frame=window(cell=cell, arm=arm, model=model, lo=lo, hi=hi), target=out)
                print(f"stress {tag} {month} {key}: {time.monotonic() - started:.0f} s", flush=True)
                del model
                torch.cuda.empty_cache()
        del cell
        torch.cuda.empty_cache()


def attention(*, device: str) -> None:
    """Mean attention weight from the query token to each token, by lead band, for the fitted attention arms."""
    for month in MONTHS:
        cell = Cell(month=month, device=device)
        for tag in [a for a in ARGS.arms if ARMS[a].kind == "attn"]:
            arm = ARMS[tag]
            _, _, test = cell.rows_for(arm=arm)
            lead_h = (test.valid - test.origin) / 3.6e9
            for seed in ARGS.seeds:
                out = W / "attention" / tag / f"{month}_s{seed}.json"
                path = model_path(arm=arm, month=month, key="coarse", seed=seed)
                if out.exists() or not path.exists():
                    continue
                model = new_net(arm=arm, n_primaries=cell.n_primaries).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                model.mixer_ft.record = True  # ty: ignore[unresolved-attribute]
                res = {}
                for name, a, b in LEAD_BANDS:
                    sel = np.nonzero((lead_h >= a) & (lead_h < b))[0][::20]
                    acc = []
                    for i in np.array_split(sel, max(1, len(sel) // 65_536)):
                        if len(i) == 0:
                            continue
                        x = torch.from_numpy(test.x[i]).to(device)
                        p = torch.from_numpy(test.primary[i]).to(device)
                        with torch.no_grad():
                            model(x, p)
                        acc.append(model.mixer_ft.last_weights.mean(0).cpu().numpy() * len(i))  # ty: ignore[unresolved-attribute]
                    res[name] = (np.sum(acc, axis=0) / max(1, len(sel))).tolist() if acc else None
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"tokens": list(model.mixer_ft.kinds) , "by_band": res}, indent=1))  # ty: ignore[unresolved-attribute]
                print("attention", tag, month, seed, flush=True)
        del cell
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("MLP hidden widths for parameter-count matching:", HIDDEN, "T1 mixer parameters:", _T1_PARAMS, flush=True)
    {"pretrain": pretrain, "step2": step2, "coarse": coarse, "stress": stress, "attention": attention}[ARGS.command](device="cuda")
