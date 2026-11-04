# ruff: noqa: ANN001, ANN201, ANN204, D101, D102, D103, D107, E501, F841, RUF005
"""Experiment 543: 511's daily-refit L-1d recipe with auxiliary output heads (F1, F2, F3).

Inputs are exactly L-1d's (37 lag-free columns and the Primary embedding). The last layer is widened
by the auxiliary outputs, which are zero-initialised, so the pretrained ERA5/CAMS weights are reused
and the fine-tune starts as L-1d. The forecast reads only the first 3 x 3 mixture outputs.

- F1: three regression outputs for the standardised newest-observed, day-ago, and week-ago power
  (experiment 397's lag value columns; training targets only, never inputs), masked when missing.
- F2: one logit for "power at the target time is below -0.005 of capacity", masked at the eight
  MVA-only Primaries (their sign is not metered).
- F3: both.

Usage: python e00543_fit.py {fit|assemble} --arm F1|F2|F3 [--stress]
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = (
    ML_EXPERIMENTS_DATA_DIR / "00543-lag-free-network-auxiliary-recent-level-and-reverse-flow-heads"
)
W = WORK / "work"
SITE_MAP = (
    ML_EXPERIMENTS_DATA_DIR
    / "00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/site_map.csv"
)
REVERSE_EPS = 0.005
ARM_HEADS = {"F1": ("level",), "F1-w01": ("level",), "F2": ("reverse",), "F3": ("level", "reverse")}
N_AUX = {"level": 3, "reverse": 1}
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble"))
parser.add_argument("--arm", choices=list(ARM_HEADS) + ["F1-w01"], required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
LAMBDA = 0.1 if ARGS.arm == "F1-w01" else 0.5  # F1-w01 is a post hoc check of the loss weight
ARM = f.Arm(ARGS.arm, "d", 90.0)
HEADS = ARM_HEADS[ARGS.arm]
NAUX = sum(N_AUX[h] for h in HEADS)

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
EXTRA = list(LAG)  # all 9 lag columns are read at build time, then removed from the inputs
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
NEW_NAMES = [NAMES[i] for i in EXTRA]
POS = {n: N0 + j for j, n in enumerate(NEW_NAMES)}
AUX_VALUE = ["lag_origin", "lag_day", "lag_week"]
assert all(n in POS for n in AUX_VALUE), (AUX_VALUE, list(POS))
SIGNED = {
    int(r["primary"]): r["group"] != "MVA-only eight"
    for r in pl.read_csv(SITE_MAP).iter_rows(named=True)
}
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


def layout():
    keep = list(KEEP0) + EXTRA
    return keep, [*COLS0, *NEW_NAMES], MASK[np.array(keep)]


data.non_lag_layout = layout


class AuxMdn(base.PlainMdn):
    """`PlainMdn` plus `NAUX` auxiliary outputs; `forward` returns the 9 mixture outputs only."""

    def __init__(self, *, n_inputs: int, n_primaries: int) -> None:
        super().__init__(
            n_inputs=n_inputs,
            hidden=c.HIDDEN,
            layers=c.LAYERS,
            components=c.COMPONENTS,
            dropout=c.DROPOUT,
            n_primaries=n_primaries,
        )
        last = self.net[-1]
        assert isinstance(last, torch.nn.Linear)
        head = torch.nn.Linear(last.in_features, last.out_features + NAUX)
        with torch.no_grad():
            head.weight.zero_()
            head.weight[: last.out_features] = last.weight
            head.bias.zero_()
            head.bias[: last.out_features] = last.bias
            if "reverse" in HEADS:
                head.bias[-1] = -5.0  # rare event: start near 0.7% probability
        self.net[-1] = head

    def forward_all(self, x, primary):
        out = super().forward(x, primary)
        return out[..., : 3 * c.COMPONENTS], out[..., 3 * c.COMPONENTS :]

    def forward(self, x, primary=None):
        return self.forward_all(x, primary)[0]


def new_model(*, outlier: bool, n_inputs: int, n_primaries: int) -> torch.nn.Module:
    assert not outlier
    return AuxMdn(n_inputs=n_inputs, n_primaries=n_primaries)


def load_pretrained(*, model, path):
    """Load the 37-input pretrained weights into the widened network (auxiliary rows stay zero)."""
    state = torch.load(path, map_location="cpu")
    if any(key.startswith("net.net.") or key == "net.embedding.weight" for key in state):
        state = {key[4:]: v for key, v in state.items()}
    own = model.state_dict()
    last = max(
        int(key.split(".")[1])
        for key in state
        if key.startswith("net.") and key.endswith(".weight")
    )
    for suffix in ("weight", "bias"):
        key = f"net.{last}.{suffix}"
        wide = own[key].clone()
        wide[: state[key].shape[0]] = state[key]
        state[key] = wide
    model.load_state_dict(state)


c.new_model = new_model
c.load_pretrained = load_pretrained
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


def lag_time_us(*, mr_valid, age_scaled):
    return mr_valid - (age_scaled.astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9).astype(
        np.int64
    )


def check_rows(*, x, valid_us, origin_us, lead, label):
    """Assert row by row: lag ages match an independent recomputation; lags precede origin minus delay."""
    ages = prep.lag_ages_hours(lead_hours=lead.astype(np.float64))
    delay_us = int(common.NEWEST_AGE * 30 * 60 * 1e6)
    example = None
    for j, tag in enumerate(("origin", "day", "week")):
        stored = x[:, POS[f"lag_{tag}_age"]].astype(np.float64) * prep.LEAD_SCALE_HOURS
        assert np.allclose(stored, ages[:, j], atol=1e-3), (
            f"{label} {tag} age differs from independent recomputation"
        )
        t_lag = valid_us - (ages[:, j] * 3.6e9).astype(np.int64)
        assert (t_lag <= origin_us - delay_us).all(), (
            f"{label} {tag} lag observed after origin minus data delay"
        )
        if example is None:
            i = int(np.argmax(ages[:, j] == ages[:, j].min()))
            example = (
                f"origin-lag of row {i}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, "
                f"lag observed {datetime.fromtimestamp(t_lag[i] / 1e6, UTC)} (<= origin - {common.NEWEST_AGE * 0.5} h)"
            )
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    return example


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    mr = _build_month(month=month, clean=clean)
    ex = check_rows(
        x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train"
    )
    pub = common.PUBLICATION_DELAY
    runs = np.unique(mr.origin_us)
    miss = [
        r
        for r in runs
        if not (
            prep.SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - int(pub.total_seconds() * 1e6)) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss, (
        f"{month}: {len(miss)} training origins have no weather-run summary file one publication delay earlier"
    )
    val = np.stack([mr.x[:, POS[n]] for n in AUX_VALUE], axis=1).astype(np.float32)
    ok = np.stack([mr.x[:, POS[f"{n}_missing"]] < 0.5 for n in AUX_VALUE], axis=1)
    ok &= np.abs(val) < prep.INPUT_CLIP_SD - 1e-6  # clipped values are not targets
    # the raw lag matrices may hold flagged half-hours: keep a target only where the cleaned power at the lag time is observed and equal
    lag_idx = np.stack(
        [
            np.rint(
                (
                    lag_time_us(mr_valid=mr.valid_us, age_scaled=mr.x[:, POS[f"{n}_age"]])
                    - c.prep422.GRID_START_US
                )
                / c.prep422.STEP_US
            )
            for n in AUX_VALUE
        ],
        axis=1,
    ).astype(np.int64)
    ref = clean[mr.primary[:, None], np.clip(lag_idx, 0, clean.shape[1] - 1)]
    agree = (
        np.abs(
            val * mr.base.standardiser.std[[POS[n] for n in AUX_VALUE]]
            + mr.base.standardiser.mean[[POS[n] for n in AUX_VALUE]]
            - ref
        )
        < 1e-3
    )
    ok_raw = ok.copy()
    ok &= np.isfinite(ref)
    mr.aux = np.where(ok, val, np.nan).astype(np.float32)
    mr.aux_time = np.stack(
        [lag_time_us(mr_valid=mr.valid_us, age_scaled=mr.x[:, POS[f"{n}_age"]]) for n in AUX_VALUE],
        axis=1,
    )
    mr.aux_std = np.array([mr.base.standardiser.std[POS[n]] for n in AUX_VALUE], dtype=np.float32)
    mr.signed = np.array([SIGNED[int(p)] for p in range(len(mr.base.target_mean))])
    mr.x = np.ascontiguousarray(mr.x[:, :N0])
    assert mr.x.shape[1] == N0
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {
        "train_rows": len(mr.y),
        "train_origins": len(runs),
        "lag_example_train": ex,
        "aux_present_share": [float(ok[:, j].mean()) for j in range(3)],
        "aux_raw_present_share": [float(ok_raw[:, j].mean()) for j in range(3)],
        "aux_equals_cleaned_power_share_of_kept": [
            float(agree[ok[:, j], j].mean()) for j in range(3)
        ],
        "reverse_label_share_signed_rows": float(
            ((mr.y < -REVERSE_EPS) & mr.signed[mr.primary]).sum()
            / max(1, mr.signed[mr.primary].sum())
        ),
        "reverse_label_rows": int(((mr.y < -REVERSE_EPS) & mr.signed[mr.primary]).sum()),
    }
    print(month, "train checks passed;", ex, flush=True)
    return mr


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    ex = check_rows(
        x=test.x, valid_us=test.valid, origin_us=test.origin, lead=test.lead, label=f"{month} test"
    )
    pub = common.PUBLICATION_DELAY
    for o in np.unique(test.origin):
        origin = datetime.fromtimestamp(o / 1e6, UTC)
        run = origin - pub
        fr = pl.read_parquet(common.demand_frame_path(run=run))
        assert (fr["nwp_init_time"] <= fr["power_fcst_init_time"]).all()
        assert (fr["power_fcst_init_time"] == origin).all()
        assert (prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet").exists()
        CHECK["origins_weather_checked"] += 1
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed;", CHECK["months"][month]["weather_example"], flush=True)
    return dataclasses.replace(test, x=np.ascontiguousarray(test.x[:, :N0]))


k.build_month, data.build_test = build_month, build_test


class AuxGpuMonth(k.GpuMonth):
    def __init__(self, *, mr, device, flag=None):
        super().__init__(mr=mr, device=device, flag=flag)
        self.aux = torch.from_numpy(mr.aux).to(device)
        self.mean = torch.from_numpy(mr.base.target_mean.astype(np.float32)).to(device)
        self.std = torch.from_numpy(mr.base.target_std.astype(np.float32)).to(device)
        self.signed = torch.from_numpy(mr.signed).to(device)

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        idx = torch.from_numpy(refit.index).to(self.device)
        t["a"] = self.aux[idx].clone()
        t["mean"], t["std"], t["sg"] = self.mean[t["p"]], self.std[t["p"]], self.signed[t["p"]]
        return t


k.GpuMonth = AuxGpuMonth


def aux_loss(*, aux_out, t, b, w):
    """Return the weighted auxiliary loss: each head is normalised by the weight of its own present terms."""
    total = torch.zeros((), device=aux_out.device)
    col = 0
    if "level" in HEADS:
        tgt = t["a"][b]
        m = torch.isfinite(tgt).float()
        err = ((aux_out[:, :3] - torch.nan_to_num(tgt)) ** 2) * m
        total = total + (w[:, None] * err).sum() / (w[:, None] * m).sum().clamp_min(1.0)
        col = 3
    if "reverse" in HEADS:
        y = t["z"][b] * t["std"][b] + t["mean"][b]
        label = (y < -REVERSE_EPS).float()
        m = t["sg"][b].float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            aux_out[:, col], label, reduction="none"
        )
        total = total + (w * m * bce).sum() / (w * m).sum().clamp_min(1.0)
    return total


def train_aux(*, model, t, seed, lr, device):
    """Experiment 510's loop (weighted NLL, early stopping on the served NLL) plus the weighted auxiliary loss."""
    torch.manual_seed(seed)
    is_val = t["v"]
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        val_idx = val_idx[
            torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS].to(device)
        ]
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            main, aux_out = model.forward_all(t["x"][b], t["p"][b])
            nll = base.mixture_nll(out=main, z=t["z"][b], components=c.COMPONENTS)
            w = t["w"][b]
            loss = (w * nll).sum() / w.sum() + LAMBDA * aux_loss(aux_out=aux_out, t=t, b=b, w=w)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        base.mixture_nll(
                            out=model(t["x"][ch], t["p"][ch]), z=t["z"][ch], components=c.COMPONENTS
                        )
                        for ch in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
            )
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


def fine_tune(*, arm, g, month, cut, seed, device, prev_state, shift=None, sign=0, block=None):
    import time

    mr = g.mr
    started = time.monotonic()
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=arm.half_life)
    shift_z, injected = None, 0
    if shift is not None and block is not None:
        inj = (
            (mr.valid_us[refit.index] >= block[0])
            & (mr.valid_us[refit.index] < block[1])
            & ~refit.is_val
        )
        prim = mr.primary[refit.index]
        shift_z = np.where(inj, sign * shift[prim] / mr.base.target_std[prim], 0.0)
        injected = int(inj.sum())
    t = g.refit_tensors(refit=refit, shift_z=shift_z)
    if shift is not None and block is not None:
        # shift the lag-valued auxiliary targets that were observed inside the injected block, as 538 shifted lag inputs
        times = mr.aux_time[refit.index]
        inblock = (times >= block[0]) & (times < block[1])
        delta = np.where(
            inblock, sign * shift[mr.primary[refit.index]][:, None] / mr.aux_std[None, :], 0.0
        ).astype(np.float32)
        t["a"] = t["a"] + torch.from_numpy(delta).to(device)
    feeder = time.monotonic() - started
    model = c.new_model(
        outlier=False, n_inputs=mr.x.shape[1], n_primaries=len(mr.base.target_mean)
    ).to(device)
    day = f.day_of(month=month, cut=cut)
    t0 = time.monotonic()
    c.load_pretrained(model=model, path=f.pretrained_file(arm=arm, month=month, seed=seed))
    model.to(device)
    epochs, best = train_aux(model=model, t=t, seed=seed, lr=c.FINE_TUNE_LR, device=device)
    log = {
        "name": f"{arm.tag}/{month}_d{day}_s{seed}",
        "epochs": epochs,
        "val_nll": best,
        "feeder_seconds": round(feeder, 2),
        "train_seconds": round(time.monotonic() - t0, 2),
        "injected_rows": injected,
        **refit.report,
    }
    del t
    return model, log


f.fine_tune = fine_tune


def assemble():
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [
                W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(
                frame=frame, target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    else:
        f.fit_arm(
            arm=ARM,
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
        out = WORK / f"checks_{ARM.tag}{'_stress' if ARGS.stress else ''}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {k_: v for k_, v in CHECK.items() if k_ != "months"})
