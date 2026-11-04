# ruff: noqa: E501
r"""Fit the stress cells of experiment 556: how much of an injected step reaches the next forecasts.

A fine-tune of the lag-free network depends only on its cutoff, half-life, pretrained weights, seed,
and (for the two controls) its rule, so the arms differ only in which cutoffs they refit at and in
which rows or weights they use. This script reuses experiment 511's rows, network, and loop.

Commands:

- `check`: reproduce a stored fit of experiment 511 (L-1d, 2026-01, day 10) from the stored weights,
  and fit it again here; print the maximum median difference of each. Print the look-ahead audit.
- `time`: fit one stress cell end to end and print the seconds.
- `cells`: fit the cells given by `--arms`, `--months`, `--kinds`.

A cell is (arm, month, scenario, sign, origin day, seed). Its output is one parquet file of
sign-adjusted median shifts (in points of capacity), summed by Primary, lead band, and origin day
after the origin, for every refit of the arm's chain.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00556-lag-free-finetune-absorption-of-load-shift-events-and-unlearn-time"
    / "work"
)
E511: Final[Path] = k.WORK_DIR
E510: Final[Path] = k.E510_WORK
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-6h": (0.0, 6.0), "6-12h": (6.0, 12.0), "12-24h": (12.0, 24.0), "24-48h": (24.0, 48.0),
    "48-72h": (48.0, 72.0), "72-120h": (72.0, 120.0), "120-240h": (120.0, 240.0),
}  # fmt: skip
"""Experiment 521's lead bands in hours (lower bound inclusive, upper exclusive, the last includes 240)."""
DEPTH_FRACTION: Final[float] = 0.10
CLIP_C: Final[float] = 3.0
SPIKE_MADS: Final[float] = 10.0
CHAIN_DAYS: Final[int] = 14
WINDOW_DAYS: Final[int] = 7
SEED: Final[int] = 0
ORIGIN_DAYS: Final[tuple[int, ...]] = (7, 14)
"""Refit origins in days from the month's first day: week 2 and week 3."""
ADAPT_ORIGIN_DAY: Final[int] = 0


@dataclass(frozen=True)
class Arm:
    """How an arm refits."""

    tag: str
    cadence: int
    half_life: float = 90.0
    embargo_days: int = 0
    clip: float | None = None


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("F1d", 1),
        Arm("F3d", 3),
        Arm("F7d", 7),
        Arm("F1d-182", 1, half_life=182.6),
        Arm("F3d-E3", 3, embargo_days=3),
        Arm("F3d-R", 3, clip=CLIP_C),
    )
}


@dataclass(frozen=True)
class Scenario:
    """What is injected into the fine-tuning targets.

    `kind` is `step` (L days before the origin), `lasting` (from the origin onward), `stuck`
    (24 h), `dropout` (6 h to zero), or `spike` (6 h of 10 residual MADs). `depth` is a share of
    capacity, or None for 3 standard deviations of the Primary's 7-day mean target.
    """

    kind: str
    length_days: float = 0.0
    depth: float | None = DEPTH_FRACTION
    mirrored: bool = False
    chain: bool = True

    @property
    def name(self) -> str:
        """Return a file-name-safe label."""
        d = "sd3" if self.depth is None else f"p{round(100 * self.depth)}"
        m = "_mir" if self.mirrored else ""
        return f"{self.kind}{self.length_days:g}_{d}{m}"


def day_us(days: float) -> int:
    """Return days as microseconds."""
    return int(days * c.DAY_US)


def cut_of(*, month: str, day: int) -> datetime:
    """Return the cutoff `day` days after the month's first day (negative before it)."""
    return c.month_start(month=month) + timedelta(days=day)


class Cell:
    """The month's rows and test rows on the GPU, and the fits that read them."""

    def __init__(self, *, month: str, device: str) -> None:
        """Build the month once."""
        self.month, self.device = month, device
        self.clean = c.clean_stack()
        self.mr = k.build_month(month=month, clean=self.clean)
        self.g = k.GpuMonth(mr=self.mr, device=device)
        self.test = data.build_test(month=month, standardiser=self.mr.base.standardiser)
        self.last_day = (k.cutoffs(month=month, cadence="d")[-1] - c.month_start(month=month)).days
        self.sd = {}
        self.mu_t = torch.from_numpy(self.mr.base.target_mean).to(device)
        self.sd_t = torch.from_numpy(self.mr.base.target_std).to(device)
        self.models: dict[tuple, torch.nn.Module] = {}
        self.q50_cache: dict[tuple, np.ndarray] = {}
        self.lead_h = ((self.test.valid - self.test.origin) / 3.6e9).astype(np.float32)
        self.band = np.full(len(self.lead_h), -1, np.int8)
        for i, (lo, hi) in enumerate(BANDS.values()):
            last = i == len(BANDS) - 1
            self.band[(self.lead_h >= lo) & ((self.lead_h <= hi) if last else (self.lead_h < hi))] = i

    # ---- selection and injection -------------------------------------------------------------
    def select(self, *, arm: Arm, cut: datetime) -> c.Refit:
        """Return the arm's rows at a cutoff (half-life weights, with the embargo applied)."""
        r = c.select_rows(rows=self.mr, cutoff=cut, half_life_days=arm.half_life)
        if arm.embargo_days:
            keep = self.mr.valid_us[r.index] <= c.us(t=cut) - day_us(arm.embargo_days)
            r = c.Refit(index=r.index[keep], y=r.y[keep], weight=r.weight[keep], is_val=r.is_val[keep], report=r.report)
        return r

    def injection(
        self, *, r: c.Refit, scen: Scenario, origin: datetime, sign: int, mad: np.ndarray | None
    ) -> np.ndarray:
        """Return the additive shift (capacity fraction) of each selected row."""
        mr = self.mr
        o = c.us(t=origin)
        valid = mr.valid_us[r.index]
        prim = mr.primary[r.index]
        tgt = mr.target_idx[r.index]
        end = o - c.CUTOFF_AGE_US
        if scen.kind == "step":
            inj = (valid >= o - day_us(scen.length_days)) & (valid < end + 1)
        elif scen.kind == "lasting":
            inj = valid >= o
        elif scen.kind == "stuck":
            inj = (valid >= end - day_us(1)) & (valid < end + 1)
        else:
            inj = (valid >= end - day_us(0.25)) & (valid < end + 1)
        inj &= ~r.is_val
        signs = np.where(np.arange(len(mr.base.target_mean)) % 2 == 0, 1, -1) if scen.mirrored else np.ones(len(mr.base.target_mean), int)
        sgn = (sign * signs)[prim]
        if scen.depth is None:
            size = c.shift_sizes(clean=self.clean, cutoff=origin)[prim]
        else:
            size = np.full(len(prim), scen.depth)
        shift = sgn * size
        if scen.kind == "stuck":
            start_idx = (end - day_us(1) - c.prep422.GRID_START_US) // c.prep422.STEP_US + 1
            held = np.nan_to_num(self.clean[prim, start_idx], nan=0.0)
            ok = np.isfinite(self.clean[prim, start_idx])
            shift = np.where(ok, held - r.y, 0.0)
        elif scen.kind == "dropout":
            shift = -r.y
        elif scen.kind == "spike":
            assert mad is not None
            shift = SPIKE_MADS * mad[prim]
        _ = tgt
        return np.where(inj, shift, 0.0).astype(np.float32), inj

    # ---- fitting and forecasting ---------------------------------------------------------------
    def new(self) -> torch.nn.Module:
        """Return a network loaded with the fold's pretrained weights."""
        m = c.new_model(outlier=False, n_inputs=self.mr.x.shape[1], n_primaries=len(self.mr.base.target_mean)).to(self.device)
        c.load_pretrained(model=m, path=f510.pretrained_path(v=f510.VARIANTS["hl90"], month=self.month, seed=SEED))
        return m.to(self.device)

    def median_rows(self, *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor) -> np.ndarray:
        """Return the model's median (capacity fraction) for rows on the GPU."""
        out = []
        for a in range(0, len(x), 262_144):
            q = c.serving_quantiles(model=model, x=x[a : a + 262_144], p=p[a : a + 262_144], outlier=False)
            q = torch.sort(q, dim=1).values[:, 5]
            out.append((q * self.sd_t[p[a : a + 262_144]] + self.mu_t[p[a : a + 262_144]]).float().cpu().numpy())
        return np.concatenate(out)

    def residual_mad(self, *, ref: torch.nn.Module, r: c.Refit, cut: datetime) -> np.ndarray:
        """Return each Primary's residual MAD (capacity fraction) of `ref` over the 90 days before `cut`."""
        sel = self.mr.valid_us[r.index] >= c.us(t=cut) - day_us(90)
        idx = torch.from_numpy(r.index[sel]).to(self.device)
        med = self.median_rows(model=ref, x=self.g.x[idx], p=self.g.p[idx])
        res = r.y[sel] - med
        prim = self.mr.primary[r.index][sel]
        out = np.zeros(len(self.mr.base.target_mean))
        for p in range(len(out)):
            e = res[prim == p]
            out[p] = float(np.median(np.abs(e - np.median(e)))) if len(e) else 0.0
        return out

    def fit(
        self, *, arm: Arm, cut: datetime, scen: Scenario | None, origin: datetime | None, sign: int,
        ref: torch.nn.Module | None, mad_model: torch.nn.Module | None,
    ) -> torch.nn.Module:  # fmt: skip
        """Fit one fine-tune: the arm's rows at `cut`, with the scenario's injection if given."""
        r = self.select(arm=arm, cut=cut)
        mad = None
        if scen is not None and scen.kind == "spike":
            assert mad_model is not None and origin is not None
            mad = self.residual_mad(ref=mad_model, r=self.select(arm=arm, cut=origin), cut=origin)
        shift = None
        y_seen = r.y
        if scen is not None:
            assert origin is not None
            shift, _ = self.injection(r=r, scen=scen, origin=origin, sign=sign, mad=mad)
            y_seen = r.y + shift
        weight = r.weight
        if arm.clip is not None:
            assert ref is not None
            m = self.residual_mad(ref=ref, r=c.Refit(r.index, y_seen, r.weight, r.is_val, r.report), cut=cut)
            idx = torch.from_numpy(r.index).to(self.device)
            med = self.median_rows(model=ref, x=self.g.x[idx], p=self.g.p[idx])
            prim = self.mr.primary[r.index]
            z = np.abs(y_seen - med) / np.maximum(m[prim], 1e-6)
            weight = (r.weight * np.minimum(1.0, arm.clip / np.maximum(z, 1e-9))).astype(np.float32)
        refit = c.Refit(r.index, r.y, weight, r.is_val, r.report)
        shift_z = None if shift is None else shift / self.mr.base.target_std[self.mr.primary[r.index]]
        t = self.g.refit_tensors(refit=refit, shift_z=shift_z)
        model = self.new()
        c.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=SEED, lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=self.device)  # fmt: skip
        del t
        return model

    def window_rows(self, *, cut: datetime) -> np.ndarray:
        """Return the indices of test rows whose origin falls in the 7 days from `cut`."""
        lo, hi = c.us(t=cut), c.us(t=cut) + day_us(WINDOW_DAYS)
        return np.nonzero((self.test.origin >= lo) & (self.test.origin < hi))[0]

    def q50(self, *, model: torch.nn.Module, rows: np.ndarray) -> np.ndarray:
        """Return the median forecast (capacity fraction) of test rows."""
        x = torch.from_numpy(self.test.x[rows]).to(self.device)
        p = torch.from_numpy(self.test.primary[rows]).to(self.device)
        return self.median_rows(model=model, x=x, p=p)

    # ---- controls ------------------------------------------------------------------------------
    def stored_path(self, *, arm: Arm, cut: datetime) -> Path | None:
        """Return a stored weights file for an unmodified arm at a cutoff, or None."""
        if arm.embargo_days or arm.clip is not None or arm.half_life != 90.0:
            return None
        day = (cut - c.month_start(month=self.month)).days
        cands = [E511 / "models" / "L-1d" / f"{self.month}_d{day}_s{SEED}.pt", E511 / "models" / "L-3d" / f"{self.month}_d{day}_s{SEED}.pt"]
        if day in k.WEEKLY_OFFSETS:
            cands.append(E510 / "models" / "hl90" / f"{self.month}_w{k.WEEKLY_OFFSETS.index(day)}_s{SEED}.pt")
        return next((p for p in cands if p.exists()), None)

    def control(self, *, arm: Arm, cut: datetime) -> torch.nn.Module:
        """Return the matched uninjected fit at a cutoff (stored weights, else fitted once and saved)."""
        key = (arm.tag if (arm.embargo_days or arm.clip is not None) else f"hl{arm.half_life}", cut)
        if key in self.models:
            return self.models[key]
        path = self.stored_path(arm=arm, cut=cut)
        if path is None:
            day = (cut - c.month_start(month=self.month)).days
            path = WORK / "models" / "control" / key[0] / f"{self.month}_d{day}_s{SEED}.pt"
        if path.exists():
            m = self.new()
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
        else:
            ref = None
            if arm.clip is not None:
                ref = self.control(arm=ARMS["F3d"], cut=cut - timedelta(days=arm.cadence))
            m = self.fit(arm=arm, cut=cut, scen=None, origin=None, sign=0, ref=ref, mad_model=None)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
        self.models[key] = m
        if len(self.models) > 80:
            self.models.pop(next(iter(self.models)))
        return m

    def control_q50(self, *, arm: Arm, cut: datetime, rows: np.ndarray) -> np.ndarray:
        """Return the control's median over the window rows (cached)."""
        key = (arm.tag if (arm.embargo_days or arm.clip is not None) else f"hl{arm.half_life}", cut)
        if key not in self.q50_cache:
            self.q50_cache[key] = self.q50(model=self.control(arm=arm, cut=cut), rows=rows)
        return self.q50_cache[key]


def run_cell(
    *, cell: Cell, arm: Arm, scen: Scenario, sign: int, origin_day: int, out: Path
) -> int:  # fmt: skip
    """Fit a cell's chain and write the shift table; return the number of fits."""
    origin = cut_of(month=cell.month, day=origin_day)
    n_chain = (CHAIN_DAYS // arm.cadence + 1) if (scen.chain and scen.kind != "lasting") else 1
    if scen.kind == "lasting":
        n_chain = (cell.last_day - origin_day) // arm.cadence + 1
    cuts = [origin + timedelta(days=j * arm.cadence) for j in range(n_chain)]
    cuts = [x for x in cuts if (x - c.month_start(month=cell.month)).days <= cell.last_day]
    records = []
    nfits = 0
    prev_state: dict[datetime, torch.nn.Module] = {}
    mad_model = cell.control(arm=arm, cut=origin) if scen.kind == "spike" else None
    # F3d-R needs the plain F3d fit three days earlier under the same injection.
    f3 = ARMS["F3d"]
    for j, cut in enumerate(cuts):
        ref = None
        if arm.clip is not None:
            pre = cut - timedelta(days=arm.cadence)
            if pre not in prev_state:
                if pre < origin:
                    prev_state[pre] = cell.fit(arm=f3, cut=pre, scen=scen if scen.kind != "lasting" else None, origin=origin, sign=sign, ref=None, mad_model=mad_model)
                    nfits += 1
            ref = prev_state[pre]
        model = cell.fit(arm=arm, cut=cut, scen=scen, origin=origin, sign=sign, ref=ref, mad_model=mad_model) if arm.clip is None else cell.fit(arm=arm, cut=cut, scen=scen, origin=origin, sign=sign, ref=ref, mad_model=mad_model)
        nfits += 1
        if arm.clip is None and arm.tag == "F3d":
            prev_state[cut] = model
        if arm.clip is not None:
            # the reference for the next cutoff is the plain F3d fit at this cutoff
            prev_state[cut] = cell.fit(arm=f3, cut=cut, scen=scen, origin=origin, sign=sign, ref=None, mad_model=mad_model)
            nfits += 1
        rows = cell.window_rows(cut=cut)
        if len(rows) == 0:
            continue
        s = cell.q50(model=model, rows=rows)
        ctl = cell.control_q50(arm=arm, cut=cut, rows=rows)
        signs = np.where(np.arange(len(cell.mr.base.target_mean)) % 2 == 0, 1, -1) if scen.mirrored else np.ones(len(cell.mr.base.target_mean), int)
        prim = cell.test.primary[rows]
        delta = (sign * signs[prim]) * 100.0 * (s - ctl)
        origin_off = ((cell.test.origin[rows] - c.us(t=origin)) // c.DAY_US).astype(np.int16)
        records.append(
            pl.DataFrame({"j": j, "refit_off": (cut - origin).days, "primary": prim, "band": cell.band[rows], "origin_off": origin_off, "shift": delta})
            .group_by("j", "refit_off", "primary", "band", "origin_off")
            .agg(pl.col("shift").sum().alias("sum"), pl.len().alias("n"))
        )
        del model
    frame = pl.concat(records).with_columns(arm=pl.lit(arm.tag), month=pl.lit(cell.month), scenario=pl.lit(scen.name), sign=pl.lit(sign), origin_day=pl.lit(origin_day))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").rename(out)
    return nfits


SCENARIOS: Final[dict[str, Scenario]] = {
    s.name: s
    for s in (
        *[Scenario("step", length) for length in (1.0, 3.0, 7.0, 14.0)],
        Scenario("step", 7.0, depth=None),
        Scenario("step", 3.0, mirrored=True),
        Scenario("lasting"),
        Scenario("stuck", chain=False),
        Scenario("dropout", chain=False),
        Scenario("spike", chain=False),
    )
}
FIRST_ONLY: Final[tuple[str, ...]] = ("step1_p10", "step14_p10", "step7_sd3", "step3_p10_mir")
SKIPPED: Final[dict[str, tuple[str, ...]]] = {
    "F1d-182": ("step1_p10", "step14_p10", "step7_sd3", "step3_p10_mir", "stuck0_p10", "dropout0_p10", "spike0_p10"),
    "F3d-E3": ("stuck0_p10", "dropout0_p10", "spike0_p10"),
    "F3d-R": ("stuck0_p10", "dropout0_p10", "spike0_p10"),
}
"""Cells left out to keep the job near 4 GPU-hours."""


def cell_list(*, kinds: list[str]) -> list[tuple[Scenario, int, tuple[int, ...]]]:
    """Return (scenario, sign, origin days) for the requested scenario names."""
    out = []
    for name in kinds:
        scen = SCENARIOS[name]
        signs = (0,) if scen.kind in ("stuck", "dropout", "spike") else (1, -1)
        origins = (ADAPT_ORIGIN_DAY,) if scen.kind == "lasting" else ORIGIN_DAYS
        for sgn in signs:
            out.append((scen, sgn if sgn else 1, origins))
    return out


def main() -> None:
    """Run a command."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("check", "time", "cells"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=list(c.MONTHS))
    ap.add_argument("--kinds", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.command == "check":
        check(device=a.device)
        return
    if a.command == "time":
        month = "2026-01"
        cell = Cell(month=month, device=a.device)
        for arm_tag, scen_name in (("F1d", "step3_p10"), ("F3d-R", "step3_p10")):
            t0 = time.monotonic()
            n = run_cell(cell=cell, arm=ARMS[arm_tag], scen=SCENARIOS[scen_name], sign=1, origin_day=14, out=WORK / "time_test" / f"{arm_tag}.parquet")
            print(f"{arm_tag} {scen_name}: {n} fits, {time.monotonic() - t0:.1f} s", flush=True)
        return
    for month in a.months:
        cell = Cell(month=month, device=a.device)
        for arm_tag in a.arms:
            arm = ARMS[arm_tag]
            for scen, sign, origins in cell_list(kinds=a.kinds):
                if scen.name in SKIPPED.get(arm.tag, ()):
                    continue
                for od in origins:
                    out = WORK / "cells" / arm.tag / f"{month}_{scen.name}_s{sign:+d}_o{od}.parquet"
                    if out.exists():
                        continue
                    sc = scen
                    if scen.name in FIRST_ONLY:
                        sc = Scenario(scen.kind, scen.length_days, scen.depth, scen.mirrored, chain=False)
                    t0 = time.monotonic()
                    n = run_cell(cell=cell, arm=arm, scen=sc, sign=sign, origin_day=od, out=out)
                    print(f"{arm.tag} {month} {scen.name} sign {sign:+d} origin {od}: {n} fits, {time.monotonic() - t0:.0f} s", flush=True)
        del cell
        torch.cuda.empty_cache()


def check(*, device: str) -> None:
    """Reproduce a stored fit and print the look-ahead audit of the chain's rows."""
    month = "2026-01"
    cell = Cell(month=month, device=device)
    cut = cut_of(month=month, day=10)
    arm = ARMS["F1d"]
    stored = cell.control(arm=arm, cut=cut)
    rows = cell.window_rows(cut=cut)
    q_stored = cell.q50(model=stored, rows=rows)
    fresh = cell.fit(arm=arm, cut=cut, scen=None, origin=None, sign=0, ref=None, mad_model=None)
    q_fresh = cell.q50(model=fresh, rows=rows)
    stored_fc = pl.read_parquet(E511 / "fits" / "L-1d" / f"{month}_d10_s0.parquet")
    print("rows", len(rows), "stored-vs-fresh max |median diff| (points)", float(np.abs(q_fresh - q_stored).max() * 100), "mean", float(np.abs(q_fresh - q_stored).mean() * 100))
    print("stored forecast file rows", stored_fc.height)
    # Look-ahead audit over the chain of a step scenario
    r = cell.select(arm=ARMS["F3d-E3"], cut=cut)
    print("E3 rows", len(r.index), "latest valid", datetime.fromtimestamp(cell.mr.valid_us[r.index].max() / 1e6), "cut", cut)
    for arm in ARMS.values():
        rr = cell.select(arm=arm, cut=cut)
        past_valid = int((cell.mr.valid_us[rr.index] > c.us(t=cut) - c.CUTOFF_AGE_US).sum())
        past_origin = int((cell.mr.origin_us[rr.index] > c.us(t=cut)).sum())
        print(f"{arm.tag}: rows {len(rr.index):,}, past valid cut {past_valid}, past origin cut {past_origin}")
    i = int(rr.index[0])
    print("example row: valid", datetime.fromtimestamp(cell.mr.valid_us[i] / 1e6), "origin", datetime.fromtimestamp(cell.mr.origin_us[i] / 1e6), "cut", cut)


if __name__ == "__main__":
    main()
