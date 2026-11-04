# ruff: noqa: E501
r"""Experiment 546: distil the day-ago and week-ago lag teacher R-DW (experiment 539) into a lag-free student.

The student is experiment 511's L-1d recipe (37 lag-free inputs, the same pretrained weights, the
same rows, recency weights, early stopping, seeds, and scoring windows). It is fine-tuned on a loss
that mixes two terms, each weighted 0.5: the mixture negative log-likelihood of the real target, and
the mixture negative log-likelihood of one draw per row from the teacher's own mixture
(a Monte Carlo estimate of the cross-entropy of the teacher's forecast under the student, which is
the teacher-to-student KL divergence up to a constant). The teacher is experiment 539's R-DW network
fitted at the same month, daily cutoff, and seed; it sees the six lag inputs in training only. The
student sees no lag input at any time. The teacher is read from experiment 539's saved weights, so
nothing here refits it.

Row-by-row assertions (inherited from experiment 539): every weather run is issued at or before its
forecast origin, and every lag the teacher reads was observed at or before the origin minus the data
delay. The student input list is asserted to contain no lag column.

Usage: python e00546_fit.py {fit|assemble} [--stress] [--months ...] [--seeds ...] [--limit N]
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00546-teacher-student-distillation-into-lag-free-demand-network"
)
W: Final[Path] = WORK / "work"
TEACHER_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00539-l1d-plus-day-and-week-lags" / "work" / "models" / "R-DW"
)
TEACHER_LAG_INDEX: Final[list[int]] = [1, 2, 4, 5, 7, 8]
"""Indices of the day-ago and week-ago lag inputs among the nine lag inputs (experiment 539's R-DW)."""
TEACHER_WEIGHT: Final[float] = 0.5
"""Weight of the teacher term; the real-target term has weight 1 - TEACHER_WEIGHT."""

parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("fit", "assemble"))
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=None)
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
if ARGS.seeds is None:
    ARGS.seeds = [0] if ARGS.stress else [0, 1]  # experiment 539 saved stress teachers for seed 0 only
ARM = f.Arm("S-DW", "d", 90.0)

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
EXTRA = [LAG[i] for i in TEACHER_LAG_INDEX]
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
assert set(KEEP0).isdisjoint(LAG), "student input list contains a lag column"
NEW_NAMES = [NAMES[i] for i in EXTRA]
POS = {n: N0 + j for j, n in enumerate(NEW_NAMES)}
VALUE = [n for n in NEW_NAMES if not n.endswith(("_missing", "_age"))]
CTX: dict = {"on": False}
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


def layout():
    """Return the teacher's column layout: the 37 lag-free columns, then the six teacher lag columns."""
    keep = list(KEEP0) + EXTRA
    return keep, [*COLS0, *NEW_NAMES], MASK[np.array(keep)]


data.non_lag_layout = layout
f.W = W
f.model_file = lambda *, arm, month, day, seed, stress="": W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
f.n_inputs = lambda *, arm, mr: N0  # the student sees the first 37 columns only


def check_rows(*, x, valid_us, origin_us, lead, label):
    """Row by row: lag ages equal an independent recomputation, every lag was observed <= origin - data delay."""
    ages = prep.lag_ages_hours(lead_hours=lead.astype(np.float64))
    delay_us = int(common.NEWEST_AGE * 30 * 60 * 1e6)
    example = None
    for j, tag in enumerate(("origin", "day", "week")):
        nm = f"lag_{tag}_age"
        if nm not in POS:
            continue
        stored = x[:, POS[nm]].astype(np.float64) * prep.LEAD_SCALE_HOURS
        assert np.allclose(stored, ages[:, j], atol=1e-3), f"{label} {tag} age differs from independent recomputation"
        t_lag = valid_us - (ages[:, j] * 3.6e9).astype(np.int64)
        assert (t_lag <= origin_us - delay_us).all(), f"{label} {tag} lag observed after origin minus data delay"
        if example is None:
            i = int(np.argmax(ages[:, j] == ages[:, j].min()))
            example = (
                f"{tag} lag of row {i}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, "
                f"target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, "
                f"lag observed {datetime.fromtimestamp(t_lag[i] / 1e6, UTC)} (<= origin - {common.NEWEST_AGE * 0.5} h)"
            )
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    return example


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month, clean):
    """Build the month's rows with the teacher's layout and run the row-by-row checks."""
    mr = _build_month(month=month, clean=clean)
    ex = check_rows(x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train")
    runs = np.unique(mr.origin_us)
    pub_us = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    miss = [r for r in runs if not (prep.SUMMARY_DIR / f"init={datetime.fromtimestamp((r - pub_us) / 1e6, UTC):%Y-%m-%d}.parquet").exists()]
    assert not miss, f"{month}: {len(miss)} training origins have no weather-run summary file 9 h earlier"
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = {"train_rows": len(mr.y), "train_origins": len(runs), "lag_example_train": ex}
    print(month, "train checks passed;", ex, flush=True)
    return mr


def build_test(*, month, standardiser):
    """Build the month's test rows and assert the weather run is at or before the origin on every origin."""
    test = _build_test(month=month, standardiser=standardiser)
    ex = check_rows(x=test.x, valid_us=test.valid, origin_us=test.origin, lead=test.lead, label=f"{month} test")
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
    CHECK["months"][month]["lag_example_test"] = ex
    CHECK["months"][month]["weather_example"] = (
        f"target {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, "
        f"weather run {datetime.fromtimestamp(test.origin[0] / 1e6, UTC) - pub} (nwp_init_time <= power_fcst_init_time asserted for all rows of every test origin)"
    )
    print(month, "test checks passed;", ex, "|", CHECK["months"][month]["weather_example"], flush=True)
    return test


k.build_month, data.build_test = build_month, build_test


def lag_delta(*, x_cols, valid_us, prim, std, ctx):
    """Return the standardised shift to add to each teacher lag value column (rows x len(VALUE))."""
    shift, sign, block = ctx["shift"], ctx["sign"], ctx["block"]
    out = np.zeros((len(valid_us), len(VALUE)), dtype=np.float32)
    for j, nm in enumerate(VALUE):
        tag = nm.replace("lag_", "")
        t = valid_us - (x_cols[f"lag_{tag}_age"].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9).astype(np.int64)
        inblock = (t >= block[0]) & (t < block[1]) & (x_cols[f"lag_{tag}_missing"] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / std[POS[nm]], 0.0)
    return out


class LagGpuMonth(k.GpuMonth):
    """GPU month whose refit tensors carry the injected shift in the teacher's lag columns (stress runs)."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        if CTX["on"]:
            idx = refit.index
            names = [n for n in POS if n.endswith(("_age", "_missing"))]
            cols = {n: self.mr.x[idx, POS[n]] for n in names}
            d = lag_delta(x_cols=cols, valid_us=self.mr.valid_us[idx], prim=self.mr.primary[idx], std=self.mr.base.standardiser.std, ctx=CTX)
            for j, nm in enumerate(VALUE):
                v = t["x"][:, POS[nm]] + torch.from_numpy(d[:, j]).to(t["x"].device)
                t["x"][:, POS[nm]] = v.clamp(-prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        return t


k.GpuMonth = LagGpuMonth


def teacher_file(*, month: str, day: int, seed: int, stress: str) -> Path:
    """Return experiment 539's saved R-DW weights for the same month, cutoff day, seed, and world."""
    return TEACHER_DIR / f"{month}_d{day}_s{seed}{stress}.pt"


def sample_teacher(*, out: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Draw one standardised target per row from the teacher's mixture, using a dedicated generator.

    The dedicated generator keeps the student's row order and dropout masks on the same random
    stream as an L-1d fit with the same seed.
    """
    log_w, mean, std = base.mixture_parts(out=out, components=c.COMPONENTS)
    comp = torch.multinomial(log_w.exp(), 1, generator=gen)
    m, s = mean.gather(1, comp).squeeze(1), std.gather(1, comp).squeeze(1)
    return m + s * torch.randn(m.shape, device=m.device, dtype=m.dtype, generator=gen)


def train_distill(
    *, student: torch.nn.Module, teacher: torch.nn.Module, t: dict[str, torch.Tensor], seed: int, device: str
) -> tuple[int, float, float]:
    """Experiment 510's loop (`c.train`) with the teacher-draw term added to the training loss.

    Early stopping scores the student on the real target alone, as `c.train` does, so the stopping
    rule is the one L-1d used. Returns the epoch count, the best validation NLL, and the mean
    training value of the teacher term in the last epoch.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    x, z, w, p, is_val = t["x"], t["z"], t["w"], t["p"], t["v"]
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(student.parameters(), lr=c.FINE_TUNE_LR, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs, last_teacher_term = float("inf"), None, 0, 0, float("nan")
    for epoch in range(c.MAX_EPOCHS):
        student.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        terms = []
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            xb, pb = x[b], p[b]
            with torch.no_grad():
                z_teacher = sample_teacher(out=teacher(xb, pb), gen=gen)
            out = student(xb[:, :N0], pb)
            nll_real = base.mixture_nll(out=out, z=z[b], components=c.COMPONENTS)
            nll_teacher = base.mixture_nll(out=out, z=z_teacher, components=c.COMPONENTS)
            nll = (1.0 - TEACHER_WEIGHT) * nll_real + TEACHER_WEIGHT * nll_teacher
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), c.GRAD_CLIP)
            optimiser.step()
            terms.append(nll_teacher.mean().detach())
        last_teacher_term = float(torch.stack(terms).mean())
        student.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [base.mixture_nll(out=student(x[i][:, :N0], p[i]), z=z[i], components=c.COMPONENTS) for i in val_idx.split(c.BATCH * 2)]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in student.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    student.load_state_dict(best_state)
    student.eval()
    return epochs, best, last_teacher_term


def fine_tune(
    *, arm, g, month, cut, seed, device, prev_state, shift=None, sign=0, block=None
):  # fmt: skip
    """Return a fine-tuned student and its log for one cutoff (restart from the pretrained weights)."""
    import time

    assert not arm.warm
    mr = g.mr
    started = time.monotonic()
    CTX.update(on=shift is not None, shift=shift, sign=sign, block=block)
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=arm.half_life)
    shift_z, injected = None, 0
    if shift is not None and block is not None:
        inj = (mr.valid_us[refit.index] >= block[0]) & (mr.valid_us[refit.index] < block[1]) & ~refit.is_val
        shift_z = np.where(inj, sign * shift[mr.primary[refit.index]] / mr.base.target_std[mr.primary[refit.index]], 0.0)
        injected = int(inj.sum())
    t = g.refit_tensors(refit=refit, shift_z=shift_z)
    feeder = time.monotonic() - started
    day = f.day_of(month=month, cut=cut)
    sfx = "" if shift is None else f"_stress{'p' if sign > 0 else 'n'}"
    tfile = teacher_file(month=month, day=day, seed=seed, stress=sfx)
    assert tfile.exists(), f"teacher weights missing: {tfile}"
    n_prim = len(mr.base.target_mean)
    teacher = c.new_model(outlier=False, n_inputs=t["x"].shape[1], n_primaries=n_prim).to(device)
    teacher.load_state_dict(torch.load(tfile, map_location=device))
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    student = c.new_model(outlier=False, n_inputs=N0, n_primaries=n_prim).to(device)
    c.load_pretrained(model=student, path=f.pretrained_file(arm=arm, month=month, seed=seed))
    student.to(device)
    t0 = time.monotonic()
    epochs, best, teacher_term = train_distill(student=student, teacher=teacher, t=t, seed=seed, device=device)
    log = {
        "name": f"{arm.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
        "teacher_term_last_epoch": teacher_term, "teacher_file": str(tfile),
        "feeder_seconds": round(feeder, 2), "train_seconds": round(time.monotonic() - t0, 2),
        "injected_rows": injected, **refit.report,
    }  # fmt: skip
    del t, teacher
    return student, log


f.fine_tune = fine_tune
_window_frame = f.window_frame


def window_frame(*, model, test, mr, lo, hi, flag, device):
    """Forecast with the student: it reads the first 37 columns only; stress runs shift nothing it reads."""
    student_test = dataclasses.replace(test, x=test.x[:, :N0])
    return _window_frame(model=model, test=student_test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)


f.window_frame = window_frame


def assemble():
    """Concatenate each daily window's forecast into one table per month and seed."""
    for month in c.MONTHS:
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            parts = [W / "fits" / ARM.tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet" for cut in cuts]
            if not all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                print("skip (incomplete)", month, seed)
                continue
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            f.atomic_frame(frame=frame, target=W / "seed_preds" / ARM.tag / f"{month}_s{seed}.parquet")
            print("wrote", ARM.tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble()
    else:
        f.fit_arm(arm=ARM, months=ARGS.months, seeds=ARGS.seeds, stress=ARGS.stress, limit=ARGS.limit, device="cuda")
        out = WORK / f"checks_{ARM.tag}{'_stress' if ARGS.stress else ''}_{'-'.join(ARGS.months)}.json"
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {k_: v for k_, v in CHECK.items() if k_ != "months"})
