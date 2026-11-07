# ruff: noqa: ANN001, ANN201, ANN003, E402, E501, D102, D103, D107, PT018
"""Experiment 646: run K1 or K3 once per ensemble member and pool the 51 outputs (GPU).

A wrapper over experiment 613's K1 recipe (`e00613_arm.py`: experiment 591's arm A1 with the E3
refit rule and lag-only synthetic shifts) and K3 recipe (`e00613_free.py`: experiment 565's arm D5
with E3). Both load the stored fits of experiments 612 and 613 when a model file exists, so mode
`b` refits nothing. Mode `c` fine-tunes on single-member rows (a random member per row on every
access) from a step-2 chain refitted the same way.

- Inference (both modes): each test row's input is run 51 times, with the seven mean-weather
  columns replaced by one member's; the 51 mixtures (3 components each) are pooled with equal
  weights into 153 components and the 11 quantiles are read from the pooled distribution function.
- `--spread run` keeps the run's own standard deviation in columns 7 to 13 (B1, C1); `zero` sets
  them to 0 (B0, C0), in training as in testing.
- `--identity` replaces the members by the stored ensemble mean to test the pooling code: the
  quantiles must then equal the stored single-pass fits.

Usage: python e00646_arm.py NET NAME {fit|step2|pretrain-check} [--mode b|c] [--spread run|zero]
       [--months ...] [--seeds ...] [--identity]
Outputs: `data/ml-experiments/00646-*/arms/NAME/work/{fits,models,step2}`.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("net", choices=("k1", "k3"))
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "step2"))  # step2: K1 only (K3 has no step 2)
_p.add_argument("--mode", choices=("b", "c"), default="b")
_p.add_argument("--spread", choices=("run", "zero"), default="run")
_p.add_argument("--months", nargs="+", default=["2025-10"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0])
_p.add_argument("--identity", action="store_true")
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
assert not (MINE.identity and MINE.mode == "c")
assert not (MINE.command == "step2" and MINE.net == "k3"), "K3 is the one-step recipe: no step 2"

E3_EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
if MINE.net == "k1":
    sys.argv = [
        "e00591_arm.py",
        "A1",
        MINE.command,
        "--months",
        *MINE.months,
        "--seeds",
        *map(str, MINE.seeds),
    ]
else:
    sys.argv = [
        "e00565_fit.py",
        MINE.command,
        "--arm",
        "D5",
        "--months",
        *MINE.months,
        "--seeds",
        *map(str, MINE.seeds),
    ]

import json
import time

import e00397_models as base  # ty: ignore[unresolved-import]
import e00646_members as mm
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00646-per-ens-member-neural-demand-networks-pooled-quantiles"
OUT = ROOT / "arms" / MINE.name
STORED = {
    "k1": [
        ML_EXPERIMENTS_DATA_DIR
        / "00612-nra-combined-networks-three-class-roster-completion/arms/k1_d5/work",
        ML_EXPERIMENTS_DATA_DIR
        / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k1_d5/work",
    ],
    "k3": [
        ML_EXPERIMENTS_DATA_DIR
        / "00612-nra-combined-networks-three-class-roster-completion/arms/k3_d5/work",
        ML_EXPERIMENTS_DATA_DIR
        / "00613-nra-roster-contrasts-with-intervals-five-seeds/arms/k3_d5/work",
    ],
}[MINE.net]
TAG = "P5" if MINE.net == "k1" else "D5"

if MINE.net == "k1":
    import e00591_arm as arm  # ty: ignore[unresolved-import]

    e = arm.e
    assert arm.LAMBDA == 0.0
    assert e.EMBARGO_DAYS == 3
    c, k, f, data = e.c, e.k, e.f, e.data
    OUTDIR = e  # the module holding W
    e.WORK, e.W = OUT, OUT / "work"
    CTX = e.CTX
    if MINE.command == "fit":
        e.row_nll = arm.row_nll
        k.build_month, k.GpuMonth = arm.build_month_syn, arm.AuxGpuMonth
        e.EMBARGO_DAYS = E3_EMBARGO_DAYS
        if MINE.mode == "c":  # fine-tune from this arm's own member-trained step 2
            arm.CHAIN = OUT / "work"
    else:  # step 2 as the stored chain made it: plain rows, no E3, pre-trained weights from 591's chain
        (OUT / "work").mkdir(parents=True, exist_ok=True)
        link = OUT / "work" / "pretrained_P2"
        if not link.exists():
            link.symlink_to(arm.CHAIN / "pretrained_P2")
else:
    import e00565_fit as e  # ty: ignore[unresolved-import]

    c, k, f, data = e.c, e.k, e.f, e.data
    W565 = e.W
    e.pretrained_dir = lambda: W565 / "pretrained_D5" / e.SPEC.columns
    e.WORK, e.W = OUT, OUT / "work"
    CTX = {}

# E3: no rows from the newest 7 days before the cutoff, half weight for the 7 days before that.
_select_rows = c.select_rows


def select_rows(**kwargs):
    refit = _select_rows(**kwargs)
    rows, cutoff = kwargs["rows"], kwargs["cutoff"]
    age_days = (c.us(t=cutoff) - rows.valid_us[refit.index]) / c.DAY_US
    window = (age_days >= E3_EMBARGO_DAYS) & (age_days < E3_EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
    weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
    if MINE.net == "k1":
        return c.Refit(
            refit.index, refit.y, weight, refit.is_val, {**refit.report, "embargo_days": 7}
        )
    keep = age_days >= E3_EMBARGO_DAYS
    return c.Refit(
        refit.index[keep],
        refit.y[keep],
        weight[keep],
        refit.is_val[keep],
        {**refit.report, "embargo_days": 7},
    )


if MINE.command == "fit":
    c.select_rows = select_rows

# Reuse the stored fits of experiments 612 and 613 (mode b): link their model files, never write them.
if MINE.mode == "b" and MINE.command == "fit":
    models = e.W / "models" / TAG
    models.mkdir(parents=True, exist_ok=True)
    for src in STORED:
        for p in (src / "models" / TAG).glob("*.pt"):
            if not (models / p.name).exists():
                (models / p.name).symlink_to(p)

# --- test rows: the 51 members of every row -------------------------------------------------
_build_test = data.build_test
SPREAD_ZERO = MINE.spread == "zero"


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    values, n_avail = mm.member_values(
        primary=test.primary, origin_us=test.origin, valid_us=test.valid
    )
    assert (n_avail == mm.N_MEMBERS).all(), f"{month}: a test row has no 51-member frame"
    test.members = mm.standardise(values=values, standardiser=standardiser)
    mean_back = test.members.astype(np.float32).mean(axis=1)
    # Members are clipped at 5 standard deviations one by one, so the mean of the clipped members
    # differs from the clipped mean in the rare rows where a member reaches the clip (precipitation,
    # wind). The median gap is the alignment test; the five smooth columns must match everywhere.
    gap_all = np.abs(mean_back - test.x[:, : mm.N_WEATHER])
    gap = float(gap_all[:, :5].max())
    assert gap < 0.05 and float(np.median(gap_all)) < 1e-3, (
        f"{month}: member mean differs from the stored mean by {gap:.3f} sd"
    )
    print(
        month,
        f"member weather aligned: max |mean of members - stored mean| = {gap:.4f} sd",
        flush=True,
    )
    if MINE.net == "k1":
        CTX["variants"] = {}
    return test


data.build_test = build_test
if MINE.net == "k1":
    e.data.build_test = build_test

# --- pooled forecast frame -------------------------------------------------------------------
CHUNK = 2048


def forecast_frame(*, model, outlier, test, select, rows, widen, device):
    assert not outlier
    assert widen == 1.0
    table = prep_primaries()
    idx = np.nonzero(select)[0]
    parts = []
    for a in range(0, len(idx), CHUNK):
        sl = idx[a : a + CHUNK]
        n = len(sl)
        x = torch.from_numpy(test.x[sl]).to(device)
        p = torch.from_numpy(test.primary[sl]).to(device)
        xe = x[:, None, :].expand(n, mm.N_MEMBERS, x.shape[1]).clone()
        if MINE.identity:
            pass  # members = the stored mean: the base input as it is
        else:
            xe[:, :, : mm.N_WEATHER] = torch.from_numpy(test.members[sl]).to(device).float()
        if SPREAD_ZERO:
            xe[:, :, mm.N_WEATHER : 2 * mm.N_WEATHER] = 0.0
        with torch.no_grad():
            out = model(xe.reshape(n * mm.N_MEMBERS, -1), p.repeat_interleave(mm.N_MEMBERS))
            log_w, mean, std = base.mixture_parts(
                out=out.double().reshape(n, mm.N_MEMBERS, -1), components=3
            )
            log_w = (log_w - np.log(mm.N_MEMBERS)).reshape(n, -1)
            q = base.mixture_quantiles(
                log_w=log_w, mean=mean.reshape(n, -1), std=std.reshape(n, -1)
            )
        mu = torch.from_numpy(rows.target_mean).to(device)[p][:, None]
        sd = torch.from_numpy(rows.target_std).to(device)[p][:, None]
        q = torch.sort(q * sd + mu, dim=1).values
        parts.append(q.float().cpu().numpy())
    q = np.concatenate(parts)
    frame = pl.DataFrame(
        {
            "primary": test.primary[idx],
            "power_fcst_init_time": test.origin[idx],
            "valid_time": test.valid[idx],
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )
    frame = frame.join(table, on="primary", how="left", maintain_order="left").drop("primary")
    levels = pl.DataFrame(q, schema=list(c.common.LEVEL_COLUMNS), orient="row")
    return frame.select(
        pl.col("time_series_id").cast(pl.Int32), "power_fcst_init_time", "valid_time"
    ).hstack(levels)


def prep_primaries():
    return c.prep.primaries().select("primary", "time_series_id")


c.forecast_frame = forecast_frame

# --- training on single-member rows (mode c) ---------------------------------------------------
if MINE.mode == "c":
    _Gpu = k.GpuMonth

    class MemberGpuMonth(_Gpu):
        """The month's rows on the GPU, with every row's 51 members (or its control member) beside them."""

        def __init__(self, *, mr, device, flag=None) -> None:
            super().__init__(mr=mr, device=device, flag=flag)
            order = np.argsort(
                mr.origin_us, kind="stable"
            )  # rows of one run together: each frame is read once
            self.M = torch.empty(
                (len(order), mm.N_MEMBERS, mm.N_WEATHER), dtype=torch.float16, device=device
            )
            n_avail = np.empty(len(order), dtype=np.int64)
            for a in range(0, len(order), 400_000):
                sl = order[a : a + 400_000]
                values, na = mm.member_values(
                    primary=mr.primary[sl], origin_us=mr.origin_us[sl], valid_us=mr.valid_us[sl]
                )
                self.M[torch.from_numpy(sl).to(device)] = torch.from_numpy(
                    mm.standardise(values=values, standardiser=mr.base.standardiser)
                ).to(device)
                n_avail[sl] = na
                del values
            self.N = torch.from_numpy(n_avail).to(device)
            print(
                f"{mr.month}: member rows {len(n_avail):,}, with 51 members {int((n_avail == 51).sum()):,}",
                flush=True,
            )

        def refit_tensors(self, *, refit, shift_z=None):
            t = super().refit_tensors(refit=refit, shift_z=shift_z)
            idx = torch.from_numpy(refit.index).to(self.device)
            t["x"] = mm.MemberX(
                x=t["x"], members=self.M[idx], n_avail=self.N[idx], zero_spread=SPREAD_ZERO
            )
            return t

    k.GpuMonth = MemberGpuMonth

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if MINE.command == "step2":
        e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")  # K1 only
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    (OUT / f"log_{MINE.command}.json").write_text(
        json.dumps({"args": vars(MINE), "seconds": round(time.monotonic() - started, 1)}, indent=1)
    )
