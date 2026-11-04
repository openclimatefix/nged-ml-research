# ruff: noqa: E501
r"""Fit and forecast the arms of experiment 553: R-DW plus W-T windows, trained with lag dropout.

Both arms are experiment 550's `RDW-T` (experiment 539's R-DW, which is experiment 511's L-1d recipe
plus the day-ago and week-ago power lags as value, missing flag, and age, with experiment 515's three
trailing temperature windows) with experiment 544's R-DROP added: in a fraction of the fine-tuning
rows the six lag inputs are set to "absent" (value 0, missing flag 1, age at the fold's training
maximum). The dropout is drawn once per daily refit, as in experiment 544.

- `RDWT-D30`: dropout fraction 0.30 (experiment 544's rate).
- `RDWT-D15`: dropout fraction 0.15.

Each arm is scored with the lags present (arm tag) and with every row's lags absent (tag plus
`-absent`). Pretraining weights are the lag-free W-T weights of experiments 515, 526, 548, and 553
(`e00553_pretrain.py`). The lag and window columns start with zero first-layer weights. This script
copies the fit code of experiment 550 and the dropout code of experiment 544. Widening by 1.15 is
applied at scoring.

Commands: `fit` (daily fits and forecasts), `assemble` (one table per month and seed).
"""

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f  # ty: ignore[unresolved-import]
import e00515_features as feat  # ty: ignore[unresolved-import]
import e00515_fit as f515  # ty: ignore[unresolved-import]
import e00515_placebo as placebo  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00553-lag-light-release-candidate-lag-dropout-with-trailing-temperature-windows"
)
W: Final[Path] = ROOT / "work"
E515_PRE: Final[Path] = f515.W / "pretrained"
E550_PRE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00550-lag-light-network-trailing-temperature-windows-with-placebo"
    / "work"
    / "pretrained"
)
E526_PRE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00526-lagless-demand-network-trailing-weather-window-placebo-repeat"
    / "work"
    / "pretrained"
)
E548_PRE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
    / "work"
    / "pretrained"
)
PRETRAINED_FOLDER: Final[dict[str, str]] = {"RDWT-D30": "W-T", "RDWT-D15": "W-T"}
DROP_FRACTION: Final[dict[str, float]] = {"RDWT-D30": 0.30, "RDWT-D15": 0.15}
EXTRA_IDX: Final[list[int]] = [1, 2, 4, 5, 7, 8]
"""Indices of the day-ago and week-ago lag inputs among the nine lag inputs (experiment 539's R-DW)."""
COLUMNS: Final[tuple[str, ...]] = feat.ARM_COLUMNS["T"]
N_WINDOW: Final[int] = len(COLUMNS)
CHUNK: Final[int] = 2_000_000
EMBEDDING: Final[int] = 8

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=("fit", "assemble"))
parser.add_argument("--arm", choices=list(PRETRAINED_FOLDER), required=True)
parser.add_argument("--stress", action="store_true")
parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args()
ARM_TAG: Final[str] = ARGS.arm
PLACEBO: Final[bool] = False
SEAM: Final[bool] = False
ARM = f.Arm(ARM_TAG, "d", 90.0)

NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
EXTRA = [LAG[i] for i in EXTRA_IDX]
KEEP0, COLS0, MASK0 = data.non_lag_layout()
N0 = len(KEEP0)
assert N0 == 37
NEW_NAMES = [NAMES[i] for i in EXTRA]
POS = {n: N0 + j for j, n in enumerate(NEW_NAMES)}
VALUE = [n for n in NEW_NAMES if not n.endswith(("_missing", "_age"))]
CTX: dict = {"on": False}
STATE: dict = {}
CHECK: dict = {"train_rows": 0, "test_rows": 0, "origins_weather_checked": 0, "months": {}}


def layout() -> tuple[list[int], list[str], np.ndarray]:
    """Return the 37 lag-free columns followed by the six day-ago and week-ago lag columns."""
    keep = list(KEEP0) + EXTRA
    return keep, [*COLS0, *NEW_NAMES], MASK[np.array(keep)]


data.non_lag_layout = layout


def load_pretrained(*, model: torch.nn.Module, path: Path) -> None:
    """Load a 37 + 3 window + 8 embedding pretrained network; the six lag inputs get zero first-layer weights."""
    state = torch.load(path, map_location="cpu")
    if any(key.startswith("net.net.") or key == "net.embedding.weight" for key in state):
        state = {key[4:]: v for key, v in state.items()}
    w = state["net.0.weight"]
    assert w.shape[1] == N0 + N_WINDOW + EMBEDDING, w.shape
    new = torch.zeros(w.shape[0], w.shape[1] + len(EXTRA), dtype=w.dtype)
    new[:, :N0] = w[:, :N0]
    new[:, N0 + len(EXTRA) : N0 + len(EXTRA) + N_WINDOW] = w[:, N0 : N0 + N_WINDOW]
    new[:, N0 + len(EXTRA) + N_WINDOW :] = w[:, N0 + N_WINDOW :]
    state["net.0.weight"] = new
    model.load_state_dict(state)


c.load_pretrained = load_pretrained


def pretrained_file(*, arm: f.Arm, month: str, seed: int) -> Path:
    """Return the pretrained weights of the arm's window kind: this experiment's, else 548's, 526's, then 515's."""
    folder = PRETRAINED_FOLDER[arm.tag]
    for root in (W / "pretrained", E548_PRE, E526_PRE, E515_PRE):
        path = root / folder / f"{month}_s{seed}.pt"
        if path.exists():
            return path
    msg = f"no {folder} pretrained weights for {month} seed {seed}"
    raise FileNotFoundError(msg)


f.W = W
f.pretrained_file = pretrained_file
f.model_file = lambda *, arm, month, day, seed, stress="": (
    W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
)


class WindowColumns:
    """The three window columns of any row set, standardised with the fold's real statistics.

    The columns are real, or the placebo's donor days.
    """

    def __init__(self, *, month: str, mr: c.MonthRows) -> None:
        """Load the 16 ensemble tables, draw the placebo days, and fit the statistics.

        The mean and standard deviation come from the real training rows.
        """
        self.lookup = f515.EnsembleLookup(seam="previous")
        self.arm_lookup = f515.EnsembleLookup(seam="persistence") if SEAM else self.lookup
        self.index = [feat.ALL_COLUMNS.index(n) for n in COLUMNS]
        self.windows = f515.Windows.__new__(f515.Windows)
        self.windows.ens = self.lookup
        self.windows.unfilled = 0
        refit = c.month_start(month=month)
        fit_until = int((c.us(t=refit) - c.prep422.GRID_START_US) // c.prep422.STEP_US)
        first_run = int(common.grid_index(times=prep.FIRST_RUN)) + 5 * placebo.STEPS_PER_DAY
        holdout = int(common.grid_index(times=common.HOLDOUT))
        self.windows.days = {
            "ens": placebo.DayStructure(fit_until_idx=fit_until, allowed=(first_run, holdout))
        }
        base = mr.base
        real = self.chunked(
            kind="real", primary=base.primary, origin_idx=base.origin_idx, valid_idx=base.target_idx
        )
        train = ~base.is_val
        self.mean = real[train].mean(axis=0)
        self.std = np.maximum(real[train].std(axis=0), 1e-6)
        del real

    def chunked(
        self, *, kind: str, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the raw window columns in chunks, so the nine-column table never sits in memory for all rows."""
        out = np.empty((len(primary), N_WINDOW), dtype=np.float32)
        lookup = self.arm_lookup if kind == "arm" else self.lookup
        for a in range(0, len(primary), CHUNK):
            sl = slice(a, a + CHUNK)
            args = {
                "primary": primary[sl],
                "origin_idx": origin_idx[sl],
                "valid_idx": valid_idx[sl],
            }
            if kind in ("real", "arm"):
                values, found = lookup.get(**args)
                if not found.all():
                    msg = f"{int((~found).sum())} rows have no window columns"
                    raise ValueError(msg)
            else:
                values = self.windows.shuffled(kind="ens", **args)
            out[sl] = values[:, self.index]
        return out

    def scaled(
        self, *, primary: np.ndarray, origin_idx: np.ndarray, valid_idx: np.ndarray
    ) -> np.ndarray:
        """Return the standardised columns of the arm (non-finite entries set to 0, as experiment 515 does)."""
        raw = self.chunked(
            kind="null" if PLACEBO else "arm",
            primary=primary,
            origin_idx=origin_idx,
            valid_idx=valid_idx,
        )
        out = (raw - self.mean) / self.std
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

    def donor_shift_report(self, *, valid_idx: np.ndarray) -> dict:
        """Return the share of rows whose placebo donor day lies after the row's own day.

        The report also holds the share of rows without a donor.
        """
        shift = self.windows.days["ens"].shifts(valid_idx=valid_idx, attempt=0)
        return {
            "share_donor_later": float(np.nanmean(shift > 0)),
            "share_no_donor_first_draw": float(np.isnan(shift).mean()),
        }


def idx_of(*, us: np.ndarray) -> np.ndarray:
    """Return the half-hour grid index of microsecond timestamps."""
    return ((us.astype(np.int64) - c.prep422.GRID_START_US) // c.prep422.STEP_US).astype(np.int64)


def check_rows(
    *, x: np.ndarray, valid_us: np.ndarray, origin_us: np.ndarray, lead: np.ndarray, label: str
) -> str:
    """Check, row by row, the lag ages, the lag observation times, and the target time.

    Lag ages equal the independent recomputation, every lag was observed no later than the origin
    minus the data delay, and the target is after the origin.
    """
    ages = prep.lag_ages_hours(lead_hours=lead.astype(np.float64))
    delay_us = int(common.NEWEST_AGE * 30 * 60 * 1e6)
    example = ""
    for j, tag in enumerate(("origin", "day", "week")):
        name = f"lag_{tag}_age"
        if name not in POS:
            continue
        stored = x[:, POS[name]].astype(np.float64) * prep.LEAD_SCALE_HOURS
        assert np.allclose(stored, ages[:, j], atol=1e-3), (
            f"{label} {tag} age differs from independent recomputation"
        )
        t_lag = valid_us - (ages[:, j] * 3.6e9).astype(np.int64)
        assert (t_lag <= origin_us - delay_us).all(), (
            f"{label} {tag} lag observed after origin minus data delay"
        )
        if not example:
            i = int(np.argmax(ages[:, j] == ages[:, j].min()))
            example = (
                f"{tag} lag of row {i}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, target {datetime.fromtimestamp(valid_us[i] / 1e6, UTC)}, "
                f"lag observed {datetime.fromtimestamp(t_lag[i] / 1e6, UTC)} (<= origin - {common.NEWEST_AGE * 0.5} h)"
            )
    assert (valid_us > origin_us).all(), f"{label}: target not after origin"
    return example


def assert_runs_exist(*, origin_us: np.ndarray, label: str) -> str:
    """Assert every origin has the weather-run summary file issued 9 h earlier, and print one example."""
    pub_us = int(common.PUBLICATION_DELAY.total_seconds() * 1e6)
    runs = np.unique(origin_us)
    miss = [
        r
        for r in runs
        if not (
            prep.SUMMARY_DIR
            / f"init={datetime.fromtimestamp((r - pub_us) / 1e6, UTC):%Y-%m-%d}.parquet"
        ).exists()
    ]
    assert not miss, f"{label}: {len(miss)} origins have no weather-run summary file 9 h earlier"
    i = len(origin_us) // 2
    return f"{label}: origin {datetime.fromtimestamp(origin_us[i] / 1e6, UTC)}, weather run {datetime.fromtimestamp((origin_us[i] - pub_us) / 1e6, UTC)} (summary file exists for all {len(runs)} origins)"


_build_month, _build_test = k.build_month, data.build_test


def build_month(*, month: str, clean: np.ndarray) -> c.MonthRows:
    """Return experiment 511's month rows with the lag and window columns appended.

    Every row is checked.
    """
    mr = _build_month(month=month, clean=clean)
    ex = check_rows(
        x=mr.x, valid_us=mr.valid_us, origin_us=mr.origin_us, lead=mr.lead, label=f"{month} train"
    )
    ex_run = assert_runs_exist(origin_us=mr.origin_us, label=f"{month} train")
    CTX["age_max"] = {n: float(mr.x[:, POS[n]].max()) for n in POS if n.endswith("_age")}
    win = WindowColumns(month=month, mr=mr)
    STATE["win"], STATE["month"] = win, month
    extra = win.scaled(
        primary=mr.primary, origin_idx=idx_of(us=mr.origin_us), valid_idx=mr.target_idx
    )
    info = {
        "train_rows": len(mr.y),
        "lag_example_train": ex,
        "weather_example_train": ex_run,
        "window_mean": win.mean.tolist(),
        "window_std": win.std.tolist(),
    }
    if PLACEBO:
        info["placebo_train"] = win.donor_shift_report(valid_idx=mr.target_idx)
    CHECK["train_rows"] += len(mr.y)
    CHECK["months"][month] = info
    print(month, "train checks passed;", ex, "|", ex_run, flush=True)
    return dataclasses.replace(mr, x=np.concatenate([mr.x, extra], axis=1))


def build_test(*, month: str, standardiser: object) -> data.TestRows:
    """Return the month's test rows with the lag and window columns, every test origin checked."""
    test = _build_test(month=month, standardiser=standardiser)
    assert STATE["month"] == month, "window statistics belong to another month"
    win: WindowColumns = STATE["win"]
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
    extra = win.scaled(primary=test.primary, origin_idx=test.origin_idx, valid_idx=test.target_idx)
    CHECK["test_rows"] += len(test.x)
    CHECK["months"][month]["lag_example_test"] = ex
    if PLACEBO:
        CHECK["months"][month]["placebo_test"] = win.donor_shift_report(valid_idx=test.target_idx)
    else:
        # real windows: the table row of (origin, valid) holds the run issued at the origin and earlier runs only
        assert (test.valid > test.origin).all()
    CHECK["months"][month]["window_example_test"] = (
        f"row 0: origin {datetime.fromtimestamp(test.origin[0] / 1e6, UTC)}, target {datetime.fromtimestamp(test.valid[0] / 1e6, UTC)}, window columns {extra[0].tolist()}"
    )
    CHECK["months"][month]["unfilled_placebo_rows"] = int(win.windows.unfilled)
    print(month, "test checks passed;", ex, flush=True)
    return dataclasses.replace(test, x=np.concatenate([test.x, extra], axis=1))


k.build_month, data.build_test = build_month, build_test


def lag_delta(
    *, x_cols: dict, valid_us: np.ndarray, prim: np.ndarray, std: np.ndarray, ctx: dict
) -> np.ndarray:
    """Return the standardised shift to add to each lag value column (rows x len(VALUE))."""
    shift, sign, block = ctx["shift"], ctx["sign"], ctx["block"]
    out = np.zeros((len(valid_us), len(VALUE)), dtype=np.float32)
    for j, nm in enumerate(VALUE):
        tag = nm.replace("lag_", "")
        t = valid_us - (
            x_cols[f"lag_{tag}_age"].astype(np.float64) * prep.LEAD_SCALE_HOURS * 3.6e9
        ).astype(np.int64)
        inblock = (t >= block[0]) & (t < block[1]) & (x_cols[f"lag_{tag}_missing"] < 0.5)
        out[:, j] = np.where(inblock, sign * shift[prim] / std[POS[nm]], 0.0)
    return out


def absent_value(*, name: str) -> float:
    """Return the standardised input that marks a lag as absent: value 0, missing flag 1, age at the fold's maximum."""
    if name.endswith("_missing"):
        return 1.0
    return CTX["age_max"][name] if name.endswith("_age") else 0.0


class LagGpuMonth(k.GpuMonth):
    """Experiment 511's GPU month with the injected shift also added to the lag value columns."""

    def refit_tensors(self, *, refit: object, shift_z: object = None) -> dict:
        """Return the refit tensors, with the stress shift applied to the lag inputs inside the block."""
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        if CTX["on"]:
            idx = refit.index  # ty: ignore[unresolved-attribute]
            names = [n for n in POS if n.endswith(("_age", "_missing"))]
            cols = {n: self.mr.x[idx, POS[n]] for n in names}
            d = lag_delta(
                x_cols=cols,
                valid_us=self.mr.valid_us[idx],
                prim=self.mr.primary[idx],
                std=self.mr.base.standardiser.std,
                ctx=CTX,
            )
            for j, nm in enumerate(VALUE):
                v = t["x"][:, POS[nm]] + torch.from_numpy(d[:, j]).to(t["x"].device)
                t["x"][:, POS[nm]] = v.clamp(-prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
            CTX["n_shifted"] = int((d != 0).any(axis=1).sum())
        rng = np.random.default_rng([CTX["seed"], int(CTX["cut"].timestamp())])
        drop = torch.from_numpy(rng.random(len(refit.index)) < DROP_FRACTION[ARM_TAG]).to(
            t["x"].device
        )
        for n in NEW_NAMES:
            t["x"][drop, POS[n]] = absent_value(name=n)
        CTX["n_dropped"] = int(drop.sum())
        return t


k.GpuMonth = LagGpuMonth
_fine_tune, _window_frame = f.fine_tune, f.window_frame


def fine_tune(**kw: object) -> tuple:
    """Fine-tune, recording whether the stress shift is on."""
    CTX.update(
        on=kw.get("shift") is not None,
        shift=kw.get("shift"),
        sign=kw.get("sign"),
        block=kw.get("block"),
        seed=kw["seed"],
        cut=kw["cut"],
    )
    model, log = _fine_tune(**kw)
    if CTX["on"]:
        log["rows_with_shifted_lag"] = CTX.get("n_shifted")
    log["rows_with_lags_dropped"] = CTX.get("n_dropped")
    return model, log


def window_frame(
    *,
    model: object,
    test: data.TestRows,
    mr: c.MonthRows,
    lo: int,
    hi: int,
    flag: object,
    device: str,
) -> pl.DataFrame:
    """Forecast the test window, with the stress shift applied to the lag inputs inside the block."""
    if CTX["on"]:
        names = [n for n in POS if n.endswith(("_age", "_missing"))]
        cols = {n: test.x[:, POS[n]] for n in names}
        d = lag_delta(
            x_cols=cols,
            valid_us=test.valid,
            prim=test.primary,
            std=mr.base.standardiser.std,
            ctx=CTX,
        )
        x = test.x.copy()
        for j, nm in enumerate(VALUE):
            x[:, POS[nm]] = np.clip(
                x[:, POS[nm]] + d[:, j], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD
            )
        test = dataclasses.replace(test, x=x)
    frame = _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    CTX["absent_frame"] = None
    if not CTX["on"] and frame is not None:
        x = test.x.copy()
        for n in NEW_NAMES:
            x[:, POS[n]] = absent_value(name=n)
        CTX["absent_frame"] = _window_frame(
            model=model,
            test=dataclasses.replace(test, x=x),
            mr=mr,
            lo=lo,
            hi=hi,
            flag=flag,
            device=device,
        )
    return frame


_atomic_frame = f.atomic_frame


def atomic_frame(*, frame: pl.DataFrame, target: Path) -> None:
    """Write the forecast, and with it the twin forecast made with every lag absent."""
    _atomic_frame(frame=frame, target=target)
    if CTX.get("absent_frame") is not None and target.parent.name == ARM_TAG:
        _atomic_frame(
            frame=CTX["absent_frame"],
            target=target.parent.parent / f"{ARM_TAG}-absent" / target.name,
        )


f.fine_tune, f.window_frame, f.atomic_frame = fine_tune, window_frame, atomic_frame


def assemble(months: list[str], seeds: list[int]) -> None:
    """Concatenate each month's daily forecasts into one table per month and seed, for both twins."""
    for tag in (ARM_TAG, f"{ARM_TAG}-absent"):
        for month in months:
            cuts = k.cutoffs(month=month, cadence="d")
            for seed in seeds:
                parts = [
                    W / "fits" / tag / f"{month}_d{f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in cuts
                ]
                twin = lambda p: (p.parent.parent / ARM_TAG / p.name).with_suffix(".empty")  # noqa: E731
                assert all(
                    p.exists()
                    or p.with_suffix(".empty").exists()
                    or (tag.endswith("-absent") and twin(p).exists())
                    for p in parts
                ), (month, seed)
                frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                _atomic_frame(
                    frame=frame, target=W / "seed_preds" / tag / f"{month}_s{seed}.parquet"
                )
                print("wrote", tag, month, seed, frame.height, flush=True)


if __name__ == "__main__":
    if ARGS.command == "assemble":
        assemble(ARGS.months, ARGS.seeds)
    else:
        f.fit_arm(
            arm=ARM,
            months=ARGS.months,
            seeds=ARGS.seeds,
            stress=ARGS.stress,
            limit=ARGS.limit,
            device="cuda",
        )
        out = (
            ROOT
            / f"checks_{ARM_TAG}{'_stress' if ARGS.stress else ''}_{'-'.join(ARGS.months)}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(CHECK, indent=1, default=str))
        print("CHECKS PASSED", {key: v for key, v in CHECK.items() if key != "months"})
