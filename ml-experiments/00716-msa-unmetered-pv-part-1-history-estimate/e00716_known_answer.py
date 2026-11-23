# ruff: noqa: E501, PT018, PT017, PLR0915, E731, ANN003
"""Experiment 716: known-answer checks KF6 and KF9 to KF12, KF15, and the early-stopping pre-check (WP5b; design section 8.2).

Usage: run.sh e00716_known_answer.py --selftest
       run.sh e00716_known_answer.py fit-m WORLD SEED [--device cuda]    # arm M, hiding the world's sky
       run.sh e00716_known_answer.py precheck                            # the early-stopping pre-check, before any PV fit
       run.sh e00716_known_answer.py fit-arms WORLD SEED [--device cuda] # the three gating arms (refuses before a passed pre-check)
       run.sh e00716_known_answer.py finish                              # the checks, the gate file (reads work/wiring/*.json)

Order (design 8.2): `fit-m WH 0` and `fit-m WT 0`, then `precheck`. A failed pre-check goes up one rung:
`e00716_world_rung.py --after N`, then `fit-m WH 0`, `fit-m WT 0` and `precheck` again. Only after a pass
are the other `fit-m` jobs (seeds 1 and 2, and world WN) and every `fit-arms` job allowed.

This file is implemented and has not been run on real data: the work package's brief forbids any GPU
use and any fit. `--selftest` runs the decision logic on synthetic numbers and one smoke pass of a
gating fit on a tiny synthetic world on the CPU.

**The world.** `e00716_worlds.build_world` makes the power at the rung that `work/world_rung.json`
names. The weather tokens keep the real ERA5-and-CAMS arrays except for one hidden sky: the shortwave
column, every shortwave-derived column, and the longwave column are zero in the history rows of world
WH and in the target row of world WT (`HIDDEN_COLUMNS`, asserted against the names of experiment 632's
columns). Arm `M` is trained from scratch under experiment 713's amended known-answer budget (stage 1,
at most 400 passes, patience 30) with experiment 713's own `train_stage`. Each gating arm starts from the
state of `M` of the same seed and trains for 60 passes with no early stopping: it uses
`e00716_train.train_stage` at the settings that the design lists (no ensemble target row, no `p0_sd`
in WT, shifts of experiment 581's `synthetic_source` in training, validation on the unshifted source,
token weights 1, trunk rate 2e-4, stage 2's cut), and records every metric after every pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00632_data as d632  # ty: ignore[unresolved-import]
import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_arms as arms
import e00716_batch as pvb
import e00716_model as pvm
import e00716_pv as pv
import e00716_train as pvt
import e00716_worlds as wd
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HIDDEN_COLUMNS: Final = {
    "shortwave": 0, "longwave": 1, "sw_mean_3h": 15, "sw_mean_6h": 16, "sw_mean_24h": 17,
    "csi_change_3h": 18, "cams_prev_day_mean": 19, "cams_prev_day_csi": 20,
}  # fmt: skip
"""The weather-token columns that carry the sky: the 7 ERA5-and-CAMS columns (shortwave 0, longwave 1), then experiment 632's 12 valid-time B1 columns (index 7 + k), then the two CAMS day columns."""
WORLD_ARMS: Final = {"WH": ("Z", "H", "H-S"), "WT": ("Z", "HT", "HT-S"), "WN": ("Z", "H", "HT")}
SEEDS: Final = (0, 1, 2)
GATING_PASSES: Final = 60
GATING_CHECK_PASSES: Final = 5
M_PASSES, M_PATIENCE = 400, 30
THRESHOLDS: Final = {
    "gap_fraction": 0.5,
    "kf10_abs": 0.005,
    "kf12_max": 0.005,
    "kf15_range": (-1.3, -0.7),
    "noise_factor": 10.0,
}
PATIENCE_713: Final = tr.PATIENCE


def check_hidden_columns() -> None:
    """Assert that `HIDDEN_COLUMNS` name the shortwave-derived columns of experiment 632's layout and experiment 713's CAMS columns."""
    b1 = d632.B1_GRID_NAMES
    assert HIDDEN_COLUMNS["sw_mean_3h"] == mdl.N_WEATHER + b1.index("sw_mean_3h")
    assert HIDDEN_COLUMNS["sw_mean_6h"] == mdl.N_WEATHER + b1.index("sw_mean_6h")
    assert HIDDEN_COLUMNS["sw_mean_24h"] == mdl.N_WEATHER + b1.index("sw_mean_24h")
    assert HIDDEN_COLUMNS["csi_change_3h"] == mdl.N_WEATHER + b1.index("csi_change_3h")
    assert [
        HIDDEN_COLUMNS["cams_prev_day_mean"],
        HIDDEN_COLUMNS["cams_prev_day_csi"],
    ] == bt.CAMS_COLUMNS
    assert d632.N_WEATHER == mdl.N_WEATHER == 7 and len(set(HIDDEN_COLUMNS.values())) == len(
        HIDDEN_COLUMNS
    )


class WorldSource(pvb.PvSource):
    """A world source that zeroes the hidden sky columns in the history rows (`hide="history"`) or the target row (`hide="target"`)."""

    def __init__(self, *, source: bt.Source, tables: pvb.PvTables, hide: str) -> None:
        """Hold the source and tables in world mode."""
        super().__init__(source=source, tables=tables, world=True)
        assert hide in {"history", "target", "none"}, hide
        self.hide = hide
        self.cols = sorted(HIDDEN_COLUMNS.values())

    def with_power_in(self, *, power_in: torch.Tensor) -> WorldSource:
        """Return the same world with `power_in` as the history power (stage 3's synthetic shifts)."""
        return WorldSource(
            source=self.source.with_power_in(power_in=power_in), tables=self.tables, hide=self.hide
        )

    def assemble(self, **kwargs) -> pvb.PvAssembled:
        """Assemble as `PvSource` does, then hide the sky."""
        batch = super().assemble(**kwargs)
        if self.hide == "history":
            batch.inputs.hist_weather[..., self.cols] = 0.0
        elif self.hide == "target":
            batch.inputs.top_weather[..., self.cols] = 0.0
        return batch


class HidingSource(bt.Source):
    """Experiment 713's source with the hidden sky columns zeroed in the history rows or the target row, for training arm `M` in a world.

    Arm `M` is trained "in each world" (design 8.2), and each world hides one part of the sky, so `M`
    must never see the part its world hides.
    """

    def __init__(self, *, source: bt.Source, hide: str) -> None:
        """Copy the arrays of `source` and set the part of the sky to hide."""
        super().__init__(
            power_z=source.power_z, dense_weather=source.dense_weather, dense_datetime=source.dense_datetime,
            day_type=source.day_type, ens=source.ens, power_in=source.power_in,
        )  # fmt: skip
        assert hide in {"history", "target", "none"}, hide
        self.hide = hide
        self.cols = sorted(HIDDEN_COLUMNS.values())

    def with_power_in(self, *, power_in: torch.Tensor) -> HidingSource:
        """Return the same hiding source with `power_in` as the history power."""
        return HidingSource(source=super().with_power_in(power_in=power_in), hide=self.hide)

    def assemble(self, **kwargs) -> bt.Assembled:
        """Assemble as `bt.Source` does, then hide the sky."""
        batch = super().assemble(**kwargs)
        if self.hide == "history":
            batch.inputs.hist_weather[..., self.cols] = 0.0
        elif self.hide == "target":
            batch.inputs.top_weather[..., self.cols] = 0.0
        return batch


# ---------------------------------------------------------------- the PV tables of a world


def world_tables(
    *, world: wd.PvWorld, dense: wd.Dense4, arm: str, kind: str, target_std: np.ndarray, device: str, shifted: tuple[np.ndarray, np.ndarray] | None = None,
) -> pvb.PvTables:  # fmt: skip
    """Return the PV tables of `arm` in world `kind`.

    `p` is `s u` over the world's own `target_std`, clipped to 0..10, for the four Primaries (other
    rows are absent). `H-S` and `HT-S` read the yield curve of the shifted sky (`shifted` is the pair
    (ghi, temperature) from `pv.shifted_sky`, 28 days earlier); `H-S` shifts the history estimate and
    `HT-S` the target row only. `WN` reads `0.30 u`, and its world has no dip (`build_world(s_override=0)`). `H` fills `hist_p`; `HT` and `T` fill the target table too, with `p0_sd = 0`.
    """
    n_grid = world.y.shape[1]
    n_days = n_grid // lay.STEPS_PER_DAY
    spec_hist = arm in {"H", "H-S", "HT", "HT-S"}
    spec_top = arm in {"HT", "HT-S", "T"}
    u = world.u
    if arm in {"H-S", "HT-S"}:
        assert shifted is not None
        zero = np.zeros(n_grid)
        u = np.stack([
            pv.yield_curve(
                ghi_mean=shifted[0][i], ghi_sd=zero, cos_zenith=dense.cos_zenith[i], sin_azimuth=dense.sin_azimuth[i],
                cos_azimuth=dense.cos_azimuth[i], doy=dense.doy, temperature_c=shifted[1][i],
            )[0]
            for i in range(u.shape[0])
        ])  # fmt: skip
    size = np.full(u.shape[0], 0.30) if kind == "WN" else world.s
    p_top = np.clip(size[:, None] * u / target_std[:, None], 0.0, 10.0)
    # `HT-S` shifts the target-row estimate only: its history rows are real (design section 5), so they read the world's own `u`
    p_hist = (
        np.clip(size[:, None] * world.u / target_std[:, None], 0.0, 10.0)
        if arm == "HT-S"
        else p_top
    )
    hist_p = np.zeros((16, n_grid), dtype=np.float32)
    hist_ok = np.zeros((16, n_grid), dtype=bool)
    top_p0 = np.zeros((16, n_days, lay.STEPS_PER_DAY * lay.N_LEAD_DAYS, 2), dtype=np.float32)
    top_filled = np.zeros((16, n_days, lay.STEPS_PER_DAY * lay.N_LEAD_DAYS), dtype=bool)
    rel = np.arange(lay.STEPS_PER_DAY * lay.N_LEAD_DAYS)
    for pos, prim in enumerate(wd.PRIMARIES):
        if spec_hist:
            hist_p[prim], hist_ok[prim] = p_hist[pos], True
        if spec_top:
            idx = (np.arange(n_days) * lay.STEPS_PER_DAY + 24)[:, None] + rel[None, :]
            valid = idx < n_grid
            vals = np.where(valid, p_top[pos][np.clip(idx, 0, n_grid - 1)], 0.0)
            top_p0[prim, :, :, 0] = vals
            top_filled[prim] = valid
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return pvb.PvTables(
        hist_p=to(hist_p), hist_ok=to(hist_ok), top_p0=to(top_p0), top_filled=to(top_filled), first_day=0,
        run_us=to(np.zeros((16, n_days), dtype=np.int64)), exception_days=frozenset(),
    )  # fmt: skip


def world_standardisation(*, world: wd.PvWorld, cutoff_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-Primary mean and standard deviation of the world's power over the training region."""
    train = world.y[:, :cutoff_idx]
    return train.mean(axis=1), train.std(axis=1)


def to_tr_world(
    *, world: wd.PvWorld, template: bt.Source, tables: pvb.PvTables, hide: str, cutoff_idx: int, device: str,
) -> tuple[tr.World, np.ndarray, np.ndarray]:  # fmt: skip
    """Return experiment 713's `World` of the synthetic power, with a `WorldSource`, and the world's mean and standard deviation (16,) (zero and one for absent Primaries)."""
    mean4, std4 = world_standardisation(world=world, cutoff_idx=cutoff_idx)
    n_grid = world.y.shape[1]
    clean = np.full((16, n_grid), np.nan, dtype=np.float32)
    mean, std = np.zeros(16), np.ones(16)
    for pos, prim in enumerate(wd.PRIMARIES):
        clean[prim] = world.y[pos]
        mean[prim], std[prim] = mean4[pos], std4[pos]
    z = torch.from_numpy(((clean - mean[:, None]) / std[:, None]).astype(np.float32)).to(device)
    source = bt.Source(
        power_z=z,
        dense_weather=template.dense_weather,
        dense_datetime=template.dense_datetime,
        day_type=template.day_type,
        ens=None,
    )
    world713 = tr.World(
        source=WorldSource(source=source, tables=tables, hide=hide), cutoff_idx=cutoff_idx, clean=clean, target_mean=mean, target_std=std, month="2026-01", placebo_pool=None,
    )  # fmt: skip
    return world713, mean, std


# ---------------------------------------------------------------- scoring


def predict_medians(
    *,
    model: torch.nn.Module,
    source: pvb.PvSource,
    c_idx: np.ndarray,
    primary: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, actual, target index) per (origin, lead day, column) in capacity shares, on the unshifted source with every PV flag as the arm sets it."""
    model.eval()
    meds, acts, idxs = [], [], []
    samples = tr.SampleSet(primary=primary, c_idx=c_idx, is_val=np.zeros(len(primary), bool))
    with torch.no_grad():
        for lead in range(lay.N_LEAD_DAYS):
            for start in range(0, len(primary), 512):
                sel = np.arange(start, min(start + 512, len(primary)))
                b = pvt.batch_for(
                    source=source,
                    samples=samples,
                    index=sel,
                    lead=np.full(len(sel), lead),
                    cut=10**9,
                )
                q = mdl.standardised_quantiles(out=model(b.inputs).out)[..., 5].cpu().numpy()
                p = primary[sel]
                meds.append(q * std[p][:, None] + mean[p][:, None])
                acts.append(b.target.cpu().numpy() * std[p][:, None] + mean[p][:, None])
                idxs.append(b.target_idx.cpu().numpy())
    return np.concatenate(meds), np.concatenate(acts), np.concatenate(idxs)


def maes(*, med: np.ndarray, act: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    """Return the MAE of the median on all finite target tokens and on the solar block (UTC half-hours 12 to 29)."""
    ok = np.isfinite(act)
    err = np.abs(med - act)
    utc = idx % lay.STEPS_PER_DAY
    solar = ok & (utc >= wd.SOLAR_HALF_HOURS_UTC[0]) & (utc <= wd.SOLAR_HALF_HOURS_UTC[1])
    return {"all": float(err[ok].mean()), "solar": float(err[solar].mean())}


def best_pass(*, val_history: list[float], patience: int = PATIENCE_713) -> int:
    """Return the 1-based pass that experiment 713's rule (improvement `val < best - 1e-5`, patience 8) keeps."""
    best, best_i, stale = float("inf"), 0, 0
    for i, v in enumerate(val_history, start=1):
        if v < best - 1e-5:
            best, best_i, stale = v, i, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return max(best_i, 1)


def check_metric(*, per_pass: list[float], best: int, width: int = GATING_CHECK_PASSES) -> float:
    """Return the median over the `width` evaluated passes ending at the best-validation pass (fewer if the best pass comes earlier)."""
    lo = max(0, best - width)
    return float(np.median(per_pass[lo:best]))


def gating_metric(*, runs: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Return, per arm and metric (`solar`, `all`), the median over the seeds of each seed's check metric.

    `runs[f"{arm}:{seed}"]` holds `val_history` and `per_pass` (a list of `{"all", "solar"}` dicts).
    """
    out: dict[str, dict[str, float]] = {}
    arms_seen = sorted({k.split(":")[0] for k in runs})
    for arm in arms_seen:
        out[arm] = {}
        for metric in ("solar", "all"):
            vals = []
            for k, run in runs.items():
                if k.split(":")[0] != arm:
                    continue
                vals.append(
                    check_metric(
                        per_pass=[m[metric] for m in run["per_pass"]],
                        best=best_pass(val_history=run["val_history"]),
                    )
                )
            out[arm][metric] = float(np.median(vals))
    return out


def kf_results(
    *, kind_metrics: dict[str, dict[str, dict[str, float]]], gaps: dict[str, float]
) -> dict:
    """Return KF9 to KF12 from the per-world metrics (`kind_metrics[world][arm][metric]`) and the references' `gap` per world."""
    th = THRESHOLDS
    wh, wt, wn = kind_metrics["WH"], kind_metrics["WT"], kind_metrics["WN"]
    kf9 = wh["H-S"]["solar"] - wh["H"]["solar"]
    kf10 = abs(wh["H-S"]["solar"] - wh["Z"]["solar"])
    kf11 = wt["HT-S"]["solar"] - wt["HT"]["solar"]
    kf12 = {"H": wn["H"]["all"] - wn["Z"]["all"], "HT": wn["HT"]["all"] - wn["Z"]["all"]}
    return {
        "KF9": {"value": kf9, "need": th["gap_fraction"] * gaps["WH"], "pass": bool(kf9 >= th["gap_fraction"] * gaps["WH"])},  # ty: ignore[unsupported-operator]
        "KF10": {"value": kf10, "need": th["kf10_abs"], "pass": bool(kf10 <= th["kf10_abs"])},  # ty: ignore[unsupported-operator]
        "KF11": {"value": kf11, "need": th["gap_fraction"] * gaps["WT"], "pass": bool(kf11 >= th["gap_fraction"] * gaps["WT"])},  # ty: ignore[unsupported-operator]
        "KF12": {"values": kf12, "need": th["kf12_max"], "pass": bool(all(v <= th["kf12_max"] for v in kf12.values()))},  # ty: ignore[unsupported-operator]
    }  # fmt: skip


def kf15(*, c_by_primary: dict[int, float]) -> dict:
    """Return KF15 (secondary, never gating): the learned `c` of each of the four Primaries of WT lies between -1.3 and -0.7."""
    lo, hi = THRESHOLDS["kf15_range"]  # ty: ignore[not-iterable]
    return {
        "c": c_by_primary,
        "in_range": {p: bool(lo <= v <= hi) for p, v in c_by_primary.items()},
        "all_in_range": bool(all(lo <= v <= hi for v in c_by_primary.values())),
    }


def early_stopping_precheck(
    *, nll_naive: float, nll_oracle: float, m_val_history: list[float]
) -> dict:
    """Return the early-stopping pre-check: the references' difference in negative log-likelihood against 10 times the validation noise of arm `M`.

    The noise is the standard deviation of the last 10 values of arm `M`'s validation loss in the same world.
    """
    last = np.asarray(m_val_history[-10:], dtype=np.float64)
    noise = float(last.std(ddof=1)) if len(last) > 1 else float("inf")
    need = THRESHOLDS["noise_factor"] * noise  # ty: ignore[unsupported-operator]
    diff = nll_naive - nll_oracle
    return {"difference": diff, "noise": noise, "need": need, "pass": bool(diff >= need)}


def gaussian_nll_standardised(*, errors: np.ndarray, std: np.ndarray) -> float:
    """Return the mean Gaussian negative log-likelihood of `errors` (n_tokens,) after dividing each by its Primary's `std`, with the error's own standard deviation."""
    z = errors / std
    var = float(np.mean(z**2))
    return 0.5 * float(np.log(2 * np.pi * var)) + 0.5


# ---------------------------------------------------------------- one fit job (real worlds; not run in this work package)


@dataclass
class GatingJob:
    """The inputs of one (world, seed) job."""

    kind: str
    seed: int
    rung: wd.Rung
    device: str


def run_gating_arm(
    *, arm: str, start_state: dict, world713: tr.World, mean: np.ndarray, std: np.ndarray, seed: int, passes: int, samples: tr.SampleSet, held: tr.SampleSet,
    train_source: pvb.PvSource | None, device: str,
) -> dict:  # fmt: skip
    """Train one gating arm from `start_state` for `passes` passes with no early stopping and record the metrics after every pass.

    The settings mirror the real fits (design 8.2): the encoders are frozen as `M*`'s are, the constant
    divisors `r_h` and `r_t` are calibrated on the arm's own training samples before the first step,
    stage 2's cut and token weights apply, there is no ensemble target row (the world's own source),
    `p0_sd` is zero, shifts are in training only, and validation reads the unshifted source. The trunk
    trains at 2e-4 and `proj_s` at the stage 2 rate of 5e-4, as in experiment 713.
    """
    spec = arms.ARMS[arm]
    model = pvm.PvNet(arm=spec.net, seed=seed).to(device)  # ty: ignore[invalid-argument-type]
    model.load_m_star(state=copy.deepcopy(start_state))
    model.freeze_encoders()
    scale: dict[str, float] = {}
    if model.pv.any_path:
        cut = tr.target_end(world=world713, stage=2)
        scale = model.calibrate_scale(
            batches=pvt.calibration_batches(
                world=world713, samples=samples, source=world713.source, cut=cut
            )  # ty: ignore[invalid-argument-type]
        )
    per_pass: list[dict] = []

    def on_pass(_passes: int, net: torch.nn.Module) -> None:
        state = torch.get_rng_state()
        med, act, idx = predict_medians(
            model=net,
            source=world713.source,
            c_idx=held.c_idx,
            primary=held.primary,
            mean=mean,
            std=std,
        )  # ty: ignore[invalid-argument-type]
        per_pass.append(maes(med=med, act=act, idx=idx))
        torch.set_rng_state(state)

    log = pvt.train_stage(
        model=model, world=world713, stage=2, samples=samples, seed=seed, encoders="frozen", max_passes=passes, patience=10**6, on_pass=on_pass, train_source=train_source,
    )  # fmt: skip
    return {
        "val_history": list(log.val_history),
        "per_pass": per_pass,
        "scale": scale,
        "a": None if model.pv_a is None else model.pv_a.detach().cpu().tolist(),
        "c": None if model.pv_c is None else model.pv_c.detach().cpu().tolist(),
    }


PATH_OVERRIDE: dict[str, object] = {}
"""The self-test points the work folder and the chosen rung at a temporary folder through this dictionary."""


def work_root() -> Path:
    """Return part 1's `work/` folder."""
    if "root" in PATH_OVERRIDE:
        return Path(str(PATH_OVERRIDE["root"]))
    return arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR, arm="Z") / "work"


def work_dir() -> Path:
    """Return `work/known_answers/` of part 1's data folder."""
    return work_root() / "known_answers"


# ---------------------------------------------------------------- self-test


def selftest() -> int:
    """Check the decision logic on synthetic numbers, the hidden columns, and one smoke pass of a gating fit on a tiny CPU world."""
    check_hidden_columns()
    print("hidden sky columns: ", HIDDEN_COLUMNS)
    assert (
        best_pass(
            val_history=[1.0, 0.9, 0.95, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        )
        == 2
    )
    assert best_pass(val_history=[1.0 - 0.01 * i for i in range(20)]) == 20
    assert check_metric(per_pass=[5, 4, 3, 2, 1, 0, 9, 9], best=6) == 2.0, "median of passes 2 to 6"
    assert check_metric(per_pass=[3.0, 1.0, 2.0], best=2) == 2.0, (
        "fewer than 5 passes when the best pass is early"
    )
    fake: dict[str, dict] = {}
    for world, (z, h, hs) in {
        "WH": (0.20, 0.10, 0.202),
        "WT": (0.20, 0.10, 0.22),
        "WN": (0.20, 0.20, 0.20),
    }.items():
        runs = {}
        for seed in SEEDS:
            for arm, v in zip(WORLD_ARMS[world], (z, h, hs), strict=True):
                runs[f"{arm}:{seed}"] = {
                    "val_history": [1.0, 0.9, 0.8, 0.8, 0.8],
                    "per_pass": [{"solar": v + 0.001 * i, "all": v} for i in range(5)],
                }
        fake[world] = gating_metric(runs=runs)
    res = kf_results(kind_metrics=fake, gaps={"WH": 0.02, "WT": 0.02})
    assert all(res[k]["pass"] for k in ("KF9", "KF10", "KF11", "KF12")), res
    assert not kf_results(kind_metrics=fake, gaps={"WH": 0.30, "WT": 0.30})["KF9"]["pass"]
    print(
        "KF9 to KF12 decision logic: ok",
        {k: round(v.get("value", 0.0), 4) for k, v in res.items() if "value" in v},
    )
    pre = early_stopping_precheck(
        nll_naive=0.50, nll_oracle=0.30, m_val_history=[0.9 + 0.001 * (i % 3) for i in range(30)]
    )
    assert (
        pre["pass"]
        and early_stopping_precheck(
            nll_naive=0.31, nll_oracle=0.30, m_val_history=[0.9 + 0.01 * (i % 3) for i in range(30)]
        )["pass"]
        is False
    )
    rng = np.random.default_rng(0)
    nll = gaussian_nll_standardised(errors=rng.normal(0, 0.05, size=20000), std=np.full(20000, 0.5))
    assert abs(nll - (0.5 * np.log(2 * np.pi * 0.01) + 0.5)) < 0.02
    k15 = kf15(c_by_primary={2: -1.0, 5: -0.9, 9: -1.2, 14: -1.4})
    assert k15["in_range"][14] is False and not k15["all_in_range"]
    print("early-stopping pre-check, Gaussian NLL, and KF15: ok")
    check_order()
    smoke()
    print("selftest passed")
    return 0


def check_order() -> None:
    """B5: no PV gating fit and no later `M` fit may run before a passed pre-check, and the ladder steps up after a failure."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "known_answers").mkdir()
        PATH_OVERRIDE.update(root=root, rung=4)
        try:
            for call in (
                lambda: fit_arms(kind="WH", seed=0, device="cpu"),
                lambda: fit_m(kind="WH", seed=1, device="cpu"),
                lambda: fit_m(kind="WN", seed=0, device="cpu"),
            ):
                try:
                    call()
                except AssertionError as err:
                    assert "precheck" in str(err), err
                else:
                    raise AssertionError("B5: a fit ran before the early-stopping pre-check")
            write_json(obj={"rung": 4, "pass": False}, path=precheck_file(rung=4))
            try:
                fit_arms(kind="WH", seed=0, device="cpu")
            except AssertionError as err:
                assert "step up" in str(err) and "--after 4" in str(err), err
            else:
                raise AssertionError("B5: a PV gating fit ran after a failed pre-check")
            write_json(obj={"rung": 4, "pass": True}, path=precheck_file(rung=4))
            require_precheck(rung=4)
            try:
                require_precheck(rung=5)
            except AssertionError:
                pass
            else:
                raise AssertionError("B5: a pre-check at rung 4 must not release a fit at rung 5")
        finally:
            PATH_OVERRIDE.clear()
    print(
        "B5: fit-arms and the later fit-m jobs refuse before a passed pre-check, and a failed pre-check names the step up to the next rung"
    )


def smoke() -> None:
    """One gating pass of arm `HT` in a tiny synthetic world on the CPU: the tables, the hidden sky, the unshifted validation, and the metrics."""
    rng = np.random.default_rng(0)
    n_days = 170
    n_grid = 48 * n_days
    steps = np.arange(n_grid)
    hour = (steps % 48) / 2.0
    elevation = np.sin(np.pi * np.clip((hour - 4.0) / 16.0, 0.0, 1.0))
    cosz = np.clip(0.9 * elevation, 0.0, 1.0) - 0.2 * (elevation <= 0)
    sky = np.repeat(rng.uniform(0.2, 1.0, size=(4, n_days)), 48, axis=1)
    dense = wd.Dense4(
        ghi=1000.0 * np.clip(cosz, 0, None)[None] * sky, temperature_c=np.tile(8.0 + 6 * np.sin(steps / 48 / 58.0), (4, 1)),
        dew_c=np.tile(4.0 + 3 * np.sin(steps / 48 / 40.0), (4, 1)), precip=np.zeros((4, n_grid)), cos_zenith=np.tile(cosz, (4, 1)),
        sin_azimuth=np.tile(np.sin(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)), cos_azimuth=np.tile(np.cos(np.pi + (hour - 12) / 12 * np.pi / 2), (4, 1)),
        day_type=rng.integers(0, 8, n_grid), half_hour=(steps % 48).astype(np.int64), doy=(1.0 + (steps // 48) % 365),
    )  # fmt: skip
    world = wd.build_world(dense=dense, rung=wd.RUNGS[3])
    cutoff = 48 * 100
    template = bt.Source(
        power_z=torch.zeros(16, n_grid), dense_weather=torch.randn(16, n_grid, 21).half(), dense_datetime=torch.randn(16, n_grid, 21).half(),
        day_type=torch.randint(0, 8, (n_grid,)), ens=None,
    )  # fmt: skip
    _, std4 = world_standardisation(world=world, cutoff_idx=cutoff)
    tables = world_tables(
        world=world, dense=dense, arm="HT", kind="WT", target_std=std4, device="cpu"
    )
    w713, mean, std = to_tr_world(
        world=world,
        template=template,
        tables=tables,
        hide="target",
        cutoff_idx=cutoff,
        device="cpu",
    )
    src = w713.source
    assert isinstance(src, WorldSource)
    batch = src.assemble(
        primary=torch.tensor([2, 5]),
        c_idx=torch.tensor([48 * 40 + 24, 48 * 41 + 24]),
        lead_day=torch.tensor([3, 4]),
    )
    assert (
        float(batch.inputs.top_weather[..., sorted(HIDDEN_COLUMNS.values())].abs().max()) == 0.0
    ), "the target-row sky must be hidden"
    assert float(batch.inputs.hist_weather[..., 0].abs().max()) > 0.0, (
        "the history sky must stay visible in a WT world"
    )
    assert float(batch.inputs.top_pv[..., 0].max()) > 0.0 and bool(batch.inputs.top_pv_ok.all()), (
        "the target PV must be on in WT"
    )
    assert float(batch.inputs.top_pv[..., 1].abs().max()) == 0.0, "p0_sd must be zero in WT"
    samples = tr.make_samples(world=w713, stage=1, placebo=False, primaries=wd.PRIMARIES)
    samples = tr.SampleSet(
        primary=samples.primary[::6], c_idx=samples.c_idx[::6], is_val=samples.is_val[::6]
    )
    held_c = np.arange(cutoff // 48, cutoff // 48 + 10) * 48 + 24
    held = tr.SampleSet(
        primary=np.repeat(np.asarray(wd.PRIMARIES), len(held_c)),
        c_idx=np.tile(held_c, 4),
        is_val=np.zeros(4 * len(held_c), bool),
    )
    start = mdl.MsaNet().state_dict()
    tr.BATCH = 64
    shifted = tr.synthetic_source(world=w713, samples=samples)
    out = run_gating_arm(
        arm="HT",
        start_state=start,
        world713=w713,
        mean=mean,
        std=std,
        seed=0,
        passes=1,
        samples=samples,
        held=held,
        train_source=shifted,
        device="cpu",
    )
    assert len(out["per_pass"]) == 1 and np.isfinite(out["per_pass"][0]["solar"]), out
    assert set(out["scale"]) == {"hist", "top"} and all(
        v > 0 and abs(v - 1.0) > 1e-3 for v in out["scale"].values()
    ), (
        "B2: the gating arm must calibrate the constant divisors r_h and r_t before its first step",
        out["scale"],
    )
    # B6: HT-S shifts the target row only; its history rows are real
    sh = pv.shifted_sky(raw=pseudo_raw(dense=dense))
    t_ht = world_tables(
        world=world, dense=dense, arm="HT", kind="WT", target_std=std4, device="cpu"
    )
    t_hts = world_tables(
        world=world,
        dense=dense,
        arm="HT-S",
        kind="WT",
        target_std=std4,
        device="cpu",
        shifted=(sh[0], sh[1]),
    )
    t_hs = world_tables(
        world=world,
        dense=dense,
        arm="H-S",
        kind="WH",
        target_std=std4,
        device="cpu",
        shifted=(sh[0], sh[1]),
    )
    t_h = world_tables(world=world, dense=dense, arm="H", kind="WH", target_std=std4, device="cpu")
    assert torch.equal(t_hts.hist_p, t_ht.hist_p), "B6: HT-S history rows must be the real estimate"
    assert not torch.equal(t_hts.top_p0, t_ht.top_p0), (
        "B6: HT-S target row must be the shifted estimate"
    )
    assert not torch.equal(t_hs.hist_p, t_h.hist_p), "B6: H-S history rows must be shifted"
    # B3: world WN has no dip, and the estimate 0.30 u is of PV that does not exist
    wn = wd.build_world(dense=dense, rung=wd.RUNGS[3], s_override=0.0)
    wh = wd.build_world(dense=dense, rung=wd.RUNGS[3])
    assert float(np.abs(wn.s).max()) == 0.0 and float(np.abs(wh.s).min()) > 0.0

    def slope_on_u(w: wd.PvWorld) -> float:
        """Least-squares slope of the part of the power that the level and anomaly leave over, on `u`."""
        left = (w.y - w.known - (w.level + w.anomaly) * w.d).reshape(-1)
        u = w.u.reshape(-1)
        return float((left * u).sum() / (u * u).sum())

    assert abs(slope_on_u(wn)) < 0.02 and slope_on_u(wh) < -0.3, (
        "B3: WN must carry no dip",
        slope_on_u(wn),
        slope_on_u(wh),
    )
    tables_n = world_tables(
        world=wn, dense=dense, arm="H", kind="WN", target_std=std4, device="cpu"
    )
    assert float(tables_n.hist_p.max()) > 0.0, "WN's estimate must still be non-zero"
    # B4: arm M's source hides the same sky as its world
    plain = bt.Source(power_z=torch.zeros(16, n_grid), dense_weather=torch.ones(16, n_grid, 21).half(), dense_datetime=torch.ones(16, n_grid, 21).half(), day_type=torch.zeros(n_grid, dtype=torch.long), ens=None)  # fmt: skip
    keys = {
        "primary": torch.tensor([2]),
        "c_idx": torch.tensor([48 * 40 + 24]),
        "lead_day": torch.tensor([0]),
        "ensemble": False,
    }
    cols = sorted(HIDDEN_COLUMNS.values())
    for hide, hist_zero, top_zero in (
        ("history", True, False),
        ("target", False, True),
        ("none", False, False),
    ):
        b = HidingSource(source=plain, hide=hide).assemble(**keys)
        assert (float(b.inputs.hist_weather[..., cols].abs().max()) == 0.0) == hist_zero, (
            "B4",
            hide,
        )
        assert (float(b.inputs.top_weather[..., cols].abs().max()) == 0.0) == top_zero, ("B4", hide)
    print(
        "B2 calibration, B3 WN without a dip, B4 hiding source for arm M, and B6 HT-S history rows: ok"
    )
    print(
        f"smoke: one gating pass of HT in a tiny WT world, solar MAE {out['per_pass'][0]['solar']:.3f}"
    )


def pseudo_raw(*, dense: wd.Dense4) -> np.ndarray:
    """Return a (4, G, 28) table with the shortwave, temperature, and solar-geometry columns that `pv.shifted_sky` reads."""
    raw = np.zeros((dense.ghi.shape[0], dense.ghi.shape[1], 28), dtype=np.float32)
    raw[:, :, pv.DRAW_COLUMNS["ghi"]] = dense.ghi
    raw[:, :, pv.DRAW_COLUMNS["temperature"]] = dense.temperature_c
    raw[:, :, pv.DRAW_COLUMNS["cos_zenith"]] = dense.cos_zenith
    raw[:, :, pv.DRAW_COLUMNS["sin_azimuth"]] = dense.sin_azimuth
    raw[:, :, pv.DRAW_COLUMNS["cos_azimuth"]] = dense.cos_azimuth
    return raw


def write_json(*, obj: object, path: Path) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def chosen_rung() -> int:
    """Return the rung that `work/world_rung.json` names, and refuse when no rung passed the `gap` pre-check."""
    if "rung" in PATH_OVERRIDE:
        return int(str(PATH_OVERRIDE["rung"]))
    record = json.loads((work_root() / "world_rung.json").read_text())
    assert record["rung"] is not None, (
        "no rung passed the gap pre-check: F7 stops before any PV fit"
    )
    return int(record["rung"])


def m_files(*, kind: str, seed: int, rung: int) -> tuple[Path, Path]:
    """Return the state file and the log file of arm `M` of a world, a seed, and a rung."""
    stem = f"M_{kind}_s{seed}_r{rung}"
    return work_dir() / f"{stem}.pt", work_dir() / f"{stem}.json"


def precheck_file(*, rung: int) -> Path:
    """Return the file that records the early-stopping pre-check at a rung."""
    return work_dir() / f"precheck_rung{rung}.json"


def read_checked(*, path: Path, files: tuple[str, ...] | None = None) -> dict:
    """Read a stored JSON artefact and refuse it when its code record is missing, from a dirty tree, or from other code (`files` limits the comparison to those modules)."""
    record = json.loads(path.read_text())
    problems = arms.artefact_problems(what=path.name, record=record, files=files)
    assert not problems, "; ".join(problems)
    return record


WORLD_FILES: Final = ("e00716_worlds.py", "e00716_world_rung.py")
"""`world_rung.json` depends on these two modules alone."""


def require_precheck(*, rung: int) -> None:
    """Refuse to continue unless the early-stopping pre-check has passed at `rung` (the rung is decided before any PV fit)."""
    path = precheck_file(rung=rung)
    assert path.exists(), f"run the `precheck` command first: {path.name} is missing"
    record = read_checked(path=path)
    assert record["rung"] == rung and record["pass"], (
        f"the early-stopping pre-check failed at rung {rung}: step up with `e00716_world_rung.py --after {rung}`"
    )


@dataclass
class WorldContext:
    """What every command of a world needs: the real fold's arrays, the dense weather, and the world."""

    data: object
    dense: wd.Dense4
    world: wd.PvWorld
    std4: np.ndarray
    hide: str
    rung: wd.Rung


def world_context(
    *, kind: str, rung_number: int, device: str, fold: str = "2026-01"
) -> WorldContext:
    """Build world `kind` at `rung_number` on the real fold's weather (experiment 713's pipeline).

    World WN has no dip (`s = 0`), and the job asserts it.
    """
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00716_world_rung as wr

    rung = wd.RUNGS[rung_number - 1]
    arms.install_own_cache()
    pipe = inp.load_pipeline(month=fold, phase="train")
    data = inp.build_fold_data(pipe=pipe, device=device)
    dense = wr.load_dense4()
    world = wd.build_world(dense=dense, rung=rung, s_override=0.0 if kind == "WN" else None)
    if kind == "WN":
        assert float(np.abs(world.s).max()) == 0.0, "world WN must carry no dip"
    check_hidden_columns()
    _, std4 = world_standardisation(world=world, cutoff_idx=data.cutoff_idx)
    return WorldContext(
        data=data, dense=dense, world=world, std4=std4, hide=HIDE_OF_WORLD[kind], rung=rung
    )  # fmt: skip


HIDE_OF_WORLD: Final = {"WH": "history", "WT": "target", "WN": "none"}


def m_world(*, ctx: WorldContext, device: str) -> tr.World:
    """Return experiment 713's `World` for arm `M` in this world: the synthetic power, and the same hidden sky as the world's other arms."""
    tables_z = world_tables(
        world=ctx.world, dense=ctx.dense, arm="Z", kind="WT", target_std=ctx.std4, device=device
    )
    w_m, _, _ = to_tr_world(
        world=ctx.world, template=ctx.data.source, tables=tables_z, hide="none", cutoff_idx=ctx.data.cutoff_idx, device=device,  # ty: ignore[unresolved-attribute]
    )  # fmt: skip
    plain = w_m.source.source  # ty: ignore[unresolved-attribute]
    return tr.World(
        source=HidingSource(source=plain, hide=ctx.hide), cutoff_idx=w_m.cutoff_idx, clean=w_m.clean, target_mean=w_m.target_mean,
        target_std=w_m.target_std, month=w_m.month, placebo_pool=None,
    )  # fmt: skip


def fit_m(*, kind: str, seed: int, device: str) -> Path:
    """Train arm `M` of one world and seed from scratch (stage 1, experiment 713's amended budget) on a source that hides the world's sky, and store its state and log.

    Seed 0 of worlds WH and WT is fitted first, to run the pre-check at the rung of `world_rung.json`.
    Every other (world, seed) waits for the pre-check to pass at that rung, so the rung is decided
    before any later fit.
    """
    rung_number = chosen_rung()
    state_path, log_path = m_files(kind=kind, seed=seed, rung=rung_number)
    if state_path.exists() and log_path.exists():
        print(f"{state_path.name}: done, skipped", flush=True)
        return state_path
    if not (kind in {"WH", "WT"} and seed == 0):
        require_precheck(rung=rung_number)
    ctx = world_context(kind=kind, rung_number=rung_number, device=device)
    bare = m_world(ctx=ctx, device=device)
    assert isinstance(bare.source, HidingSource) and bare.source.hide == HIDE_OF_WORLD[kind]
    torch.manual_seed(seed)
    m = mdl.MsaNet().to(device)
    samples0 = tr.make_samples(world=bare, stage=1, placebo=False, primaries=wd.PRIMARIES)
    m_log = tr.train_stage(
        model=m, world=bare, stage=1, samples=samples0, seed=seed, encoders="random", max_passes=M_PASSES, patience=M_PATIENCE,
    )  # fmt: skip
    state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, state_path)
    write_json(
        obj={"kind": kind, "seed": seed, "rung": rung_number, "passes": m_log.passes, "best_val": m_log.best_val, "cap_hit": m_log.cap_hit, "val_history": list(m_log.val_history), **arms.code_record()},
        path=log_path,
    )  # fmt: skip
    return state_path


def precheck(*, device: str = "cpu") -> int:
    """Run the early-stopping pre-check at the rung of `world_rung.json`, from arm `M` of seed 0 in WH and WT, and write `precheck_rung<N>.json`.

    The check reads no PV fit. On a failure it prints the step to the next rung.
    """
    del device
    import e00716_world_rung as wr

    rung_number = chosen_rung()
    dense = wr.load_dense4()
    world = wd.build_world(dense=dense, rung=wd.RUNGS[rung_number - 1])
    cutoff = wr.fold_cutoff_idx(fold="2026-01")
    _, std4 = world_standardisation(world=world, cutoff_idx=cutoff)
    out: dict = {}
    for kind in ("WH", "WT"):
        _, log_path = m_files(kind=kind, seed=0, rung=rung_number)
        assert log_path.exists(), f"fit M first: {log_path.name} is missing"
        nll = wd.heldout_nll(
            world=world, dense=dense, kind=kind, cutoff_idx=cutoff, target_std=std4
        )
        history = read_checked(path=log_path)["val_history"]
        out[kind] = early_stopping_precheck(
            nll_naive=nll["naive"], nll_oracle=nll["oracle"], m_val_history=history
        )
    passed = all(v["pass"] for v in out.values())
    write_json(
        obj={"rung": rung_number, "pass": passed, "kinds": out, **arms.code_record()},
        path=precheck_file(rung=rung_number),
    )
    print(json.dumps(out, indent=1, default=float))
    if not passed:
        print(
            f"PRE-CHECK FAILED at rung {rung_number}: run `e00716_world_rung.py --after {rung_number}`, then fit M again at the new rung",
            flush=True,
        )
        return 1
    return 0


def fit_arms(*, kind: str, seed: int, device: str) -> Path:
    """Train the three gating arms of one world and seed from the stored arm `M`, and write `work/known_answers/<kind>_s<seed>.json`.

    Refuses unless the early-stopping pre-check passed at the chosen rung.
    """
    rung_number = chosen_rung()
    require_precheck(rung=rung_number)
    path = work_dir() / f"{kind}_s{seed}.json"
    if path.exists():
        print(f"{path.name}: done, skipped", flush=True)
        return path
    state_path, log_path = m_files(kind=kind, seed=seed, rung=rung_number)
    assert state_path.exists() and log_path.exists(), f"fit M first: {state_path.name} is missing"
    ctx = world_context(kind=kind, rung_number=rung_number, device=device)
    import e00716_world_rung as wr

    sh_ghi, sh_temp, _, _ = pv.shifted_sky(raw=pseudo_raw(dense=ctx.dense))
    del wr
    start = torch.load(state_path, map_location="cpu")
    m_log = read_checked(path=log_path)
    out: dict = {
        "kind": kind, "seed": seed, "rung": rung_number, "arms": {}, **arms.code_record(),
        "M": {k: m_log[k] for k in ("passes", "best_val", "cap_hit", "val_history")},
    }  # fmt: skip
    cutoff = ctx.data.cutoff_idx  # ty: ignore[unresolved-attribute]
    held_c = np.arange(cutoff // 48, cutoff // 48 + wd.HELD_OUT_DAYS) * 48 + 24
    held = tr.SampleSet(
        primary=np.repeat(np.asarray(wd.PRIMARIES), len(held_c)),
        c_idx=np.tile(held_c, len(wd.PRIMARIES)),
        is_val=np.zeros(len(wd.PRIMARIES) * len(held_c), bool),
    )
    for arm in WORLD_ARMS[kind]:
        shifted = (sh_ghi, sh_temp) if arm.endswith("-S") else None
        tables = world_tables(
            world=ctx.world, dense=ctx.dense, arm=arm, kind=kind, target_std=ctx.std4, device=device, shifted=shifted,
        )  # fmt: skip
        w713, mean, std = to_tr_world(
            world=ctx.world, template=ctx.data.source, tables=tables, hide=ctx.hide, cutoff_idx=cutoff, device=device,  # ty: ignore[unresolved-attribute]
        )  # fmt: skip
        samples = tr.make_samples(world=w713, stage=1, placebo=False, primaries=wd.PRIMARIES)
        train_source = tr.synthetic_source(world=w713, samples=samples)
        out["arms"][arm] = run_gating_arm(
            arm=arm, start_state=start, world713=w713, mean=mean, std=std, seed=seed, passes=GATING_PASSES, samples=samples, held=held,
            train_source=train_source, device=device,
        )  # fmt: skip
        print(
            kind,
            seed,
            arm,
            "pass-60 solar MAE",
            out["arms"][arm]["per_pass"][-1]["solar"],
            flush=True,
        )
    write_json(obj=out, path=path)
    return path


WIRING_REQUIRED: Final = (
    "KF1",
    "KF2",
    "KF3",
    "KF4",
    "KF5",
    "KF6",
    "KF7",
    "KF8a",
    "KF8b",
    "KF13",
    "KF13b",
    "KF14",
)
"""The wiring checks whose records `finish` reads from `work/wiring/` (each check script writes its own). KF8a is the gradient half (`check_model`) and KF8b the optimiser-group half (`check_train`)."""


def read_wiring_records() -> tuple[dict[str, dict], list[str]]:
    """Return the stored wiring records and the names that are missing, failed, or written by other code."""
    records: dict[str, dict] = {}
    bad: list[str] = []
    for name in WIRING_REQUIRED:
        path = arms.wiring_file(data_dir=ML_EXPERIMENTS_DATA_DIR, name=name)
        if not path.exists():
            bad.append(f"missing wiring record {name}")
            continue
        rec = json.loads(path.read_text())
        records[name] = rec
        bad += arms.wiring_problems(name=name, record=rec)
    return records, bad


def finish() -> int:
    """Aggregate the jobs, run the decision logic, and write `known_answers_passed.json` only if every gating check passes."""
    root = work_root()
    rung_record = read_checked(path=root / "world_rung.json", files=WORLD_FILES)
    chosen = str(rung_record["rung"])
    require_precheck(rung=int(chosen))
    for kind in WORLD_ARMS:
        for seed in SEEDS:
            read_checked(path=m_files(kind=kind, seed=seed, rung=int(chosen))[1])
    gaps = {k: rung_record["results"][chosen]["gaps"][k]["gap"] for k in ("WH", "WT")}
    metrics, jobs, fixed = {}, {}, {}
    for kind in WORLD_ARMS:
        runs = {}
        for seed in SEEDS:
            job = read_checked(path=work_dir() / f"{kind}_s{seed}.json")
            assert job["rung"] == int(chosen), (
                f"{kind} seed {seed} was fitted at rung {job['rung']}, not {chosen}"
            )
            jobs[f"{kind}:{seed}"] = job
            for arm, res in job["arms"].items():
                runs[f"{arm}:{seed}"] = res
        metrics[kind] = gating_metric(runs=runs)
        fixed[kind] = {
            a: {m: float(np.median([runs[f"{a}:{s}"]["per_pass"][-1][m] for s in SEEDS])) for m in ("solar", "all")}
            for a in WORLD_ARMS[kind]
        }  # fmt: skip
    results = kf_results(kind_metrics=metrics, gaps=gaps)
    results["secondary_pass60"] = kf_results(kind_metrics=fixed, gaps=gaps)
    cs = np.median([jobs[f"WT:{s}"]["arms"]["HT"]["c"] for s in SEEDS], axis=0)
    results["KF15"] = kf15(c_by_primary={prim: float(cs[prim]) for prim in wd.PRIMARIES})
    pre = json.loads(precheck_file(rung=int(chosen)).read_text())
    results["early_stopping_precheck"] = pre["kinds"]
    records, bad = read_wiring_records()
    results["wiring_records"] = {
        k: {"pass": v.get("pass"), "code_hash": v.get("code_hash")} for k, v in records.items()
    }
    gating = ("KF9", "KF10", "KF11", "KF12")
    failed = [k for k in gating if not results[k]["pass"]] + bad
    results["failed"] = failed
    write_json(obj=results, path=root / "known_answers_last_run.json")
    print(json.dumps(results, indent=1, default=float))
    if failed:
        print("gate file not written; failed:", failed)
        return 1
    write_json(
        obj={**results, "code": arms.code_record()},
        path=arms.gate_file(data_dir=ML_EXPERIMENTS_DATA_DIR),
    )
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("fit-m", "precheck", "fit-arms", "finish"))
    parser.add_argument("world", nargs="?", choices=tuple(WORLD_ARMS))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "fit-m":
        fit_m(kind=args.world, seed=args.seed, device=args.device)
        return 0
    if args.command == "fit-arms":
        fit_arms(kind=args.world, seed=args.seed, device=args.device)
        return 0
    if args.command == "precheck":
        return precheck()
    return finish()


if __name__ == "__main__":
    sys.exit(main())
