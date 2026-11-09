# ruff: noqa: E501, PLR0911
"""Shared pieces of experiment 656: paths, the 24 conditions, and the MVA-world target.

Each condition changes the pretraining target of the Primaries it covers. For the six signed-MW
Primaries the apparent-power target is `|S'| = sqrt((P - I)^2 + Q^2)`, with `P` the cleaned signed
power, `I` the injected power, and `Q` a reactive-power model. For the five MVA Primaries that do
not fold (D2, D3, D4, D19, D21), `P = 0.97 |S|` and `Q = 0.243 |S|` (a "native world"). The three
MVA Primaries that fold (D8, D12, D22), D15 and D18 keep their real targets in every condition.
All values are in points of the Primary's effective capacity.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import SunAndSky, ac_power_mw, plane_of_array_w_m2
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00656-pv-probe-through-the-mva-fold-at-eight-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
CUTOFFS: Final[tuple[str, ...]] = ("2026-06", "2026-03")
"""2026-06 first (stop rules 3 and 4 read it before 2026-03 runs)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SIX: Final[tuple[str, ...]] = pv.SIGNED_SIX
NATIVE: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21")
"""The MVA Primaries whose midday flow never nears zero."""
FOLDERS: Final[tuple[str, ...]] = ("D8", "D12", "D22")
"""The MVA Primaries that fold (report-only)."""
FARMS: Final[str] = "ABCF"
DAY_US: Final[int] = 86_400_000_000
PF_Q_SHARE: Final[float] = 0.25
"""PF arm: Q = 0.25 x the median signed power of the same half-hour of the week (power factor 0.97)."""
FLOOR_Q_SHARE: Final[float] = 0.10
"""Floor arm: Q = 0.10 x the median sunny-midday power."""
NATIVE_P_SHARE: Final[float] = 0.97
NATIVE_Q_SHARE: Final[float] = 0.243
LEVELS: Final[dict[str, float]] = {"l05": 0.5, "l1": 1.0, "l2": 2.0, "l6": 6.0}
"""Injection level L: AC capacity as a multiple of the Primary's median sunny-midday reading."""
SUNNY_KT: Final[float] = 0.7
CLOUD_SHARE: Final[float] = 0.30

CONDITIONS: Final[tuple[str, ...]] = (
    # signed arm
    "sig_twin", "sig_fleet_l1_r0", "sig_fleet_l6_r0",
    # PF arm
    "pf_twin", "pf_fleet_l05_r0", "pf_fleet_l1_r0", "pf_fleet_l1_r1", "pf_fleet_l2_r0",
    "pf_fleet_l2_r1", "pf_fleet_l6_r0", "pf_fleet_l6_r1", "pf_phys_l1", "pf_phys_l6",
    "pf_clim", "pf_boxcar", "pf_sunload", "pf_cloud",
    # floor arm
    "floor_twin", "floor_fleet_l1_r0", "floor_fleet_l2_r0", "floor_fleet_l6_r0", "floor_clim",
    # zero reactive power arm
    "q0_twin", "q0_fleet_l2_r0",
)  # fmt: skip
PATTERN: Final = re.compile(
    r"(?P<arm>sig|pf|floor|q0)_(?P<kind>twin|fleet|phys|clim|boxcar|sunload|cloud)"
    r"(_(?P<level>l05|l1|l2|l6))?(_r(?P<rot>\d))?"
)
PLACEBO_LEVEL: Final = "l2"
"""Placebos are sized to the energy of the L2 fleet."""


@dataclass(frozen=True)
class Condition:
    """A parsed condition name."""

    name: str
    arm: str  # sig, pf, floor, q0
    kind: str  # twin, fleet, phys, clim, boxcar, sunload, cloud
    level: float
    rotation: int


def parse(*, name: str) -> Condition:
    """Parse a condition name such as `pf_fleet_l2_r1` or `floor_clim`."""
    match = PATTERN.fullmatch(name)
    assert match is not None, name
    level_key = match.group("level")
    level = LEVELS[level_key] if level_key else 0.0
    return Condition(
        name, match.group("arm"), match.group("kind"), level, int(match.group("rot") or 0)
    )


def weights_path(*, cutoff: str, condition: str, seed: int) -> Path:
    """Return where a pretrained network is stored."""
    return WORK / "pretrained" / cutoff / f"{condition}_s{seed}.pt"


def frame_path(*, cutoff: str) -> Path:
    """Return where a cutoff's probe frame is stored."""
    return WORK / "frames" / f"{cutoff}.npz"


def index_of(*, sites: tuple[str, ...]) -> dict[str, int]:
    """Return the row-index of each named Primary in the 16-Primary order."""
    names = [r["site"] for r in prep.primaries().iter_rows(named=True)]
    return {s: names.index(s) for s in sites}


# --- physics curves with free tilt, azimuth, and DC-to-AC ratio -----------------------------


def doy_of(*, times_us: np.ndarray) -> np.ndarray:
    """Return the day of the year of each time."""
    return (
        pl.Series(times_us).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.float64)
    )


def plant_curve(
    *, ghi: np.ndarray, x: np.ndarray, doy: np.ndarray, temp: np.ndarray, tilt: float, azimuth: float, dc_ac: float,
) -> np.ndarray:  # fmt: skip
    """Return the derated AC output per MW of AC capacity of one plant (clipped at AC capacity)."""
    cos_zenith = x[:, 14].astype(np.float64)
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    sun_az = (
        np.degrees(np.arctan2(x[:, 15].astype(np.float64), x[:, 16].astype(np.float64))) % 360.0
    )
    ghi = np.where(cos_zenith > 0.0, np.clip(ghi, 0.0, None), 0.0)
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=doy)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(len(ghi), dtype="datetime64[us]"), ghi_w_m2=ghi,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"], dtype=np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"], dtype=np.float64)),
        zenith_deg=zenith, sun_azimuth_deg=sun_az,
    )  # fmt: skip
    poa = plane_of_array_w_m2(sky=sky, tilt_deg=tilt, azimuth_deg=azimuth, tracker=False)
    unit = ac_power_mw(poa_w_m2=poa, dc_capacity_mw=dc_ac, ac_capacity_mw=1.0)
    cell = temp + pv.CELL_RISE_K * unit
    return unit * np.clip(1.0 - pv.DERATING_PER_KELVIN * (cell - pv.REFERENCE_CELL_C), 0.0, None)


BASIS: Final[tuple[tuple[float, float], ...]] = ((15.0, 1.1), (35.0, 1.1), (35.0, 1.3), (35.0, 1.5))
"""The K = 4 basis curves of the fold model: (tilt, DC-to-AC ratio), azimuth 180."""


def basis_curves(
    *, ghi: np.ndarray, x: np.ndarray, times_us: np.ndarray, temp: np.ndarray
) -> np.ndarray:
    """Return the K = 4 basis unit curves, shape (rows, 4), each over the standard 99th percentile."""
    doy = doy_of(times_us=times_us)
    u99 = pv.u99()
    return np.stack(
        [plant_curve(ghi=ghi, x=x, doy=doy, temp=temp, tilt=t, azimuth=180.0, dc_ac=r) / u99 for t, r in BASIS],
        axis=1,
    )  # fmt: skip


def physics_mixture(
    *, ghi: np.ndarray, x: np.ndarray, times_us: np.ndarray, temp: np.ndarray, rng: np.random.Generator
) -> np.ndarray:  # fmt: skip
    """Return a six-component physics mixture (tilt 15 or 35, azimuth 120/180/240, DC/AC 1.0 to 1.4)."""
    doy = doy_of(times_us=times_us)
    weights = rng.dirichlet(np.full(6, 2.0))
    total = np.zeros(len(ghi))
    for w in weights:
        tilt = float(rng.choice([15.0, 35.0]))
        az = float(rng.choice([120.0, 180.0, 240.0]))
        ratio = float(rng.uniform(1.0, 1.4))
        total += w * plant_curve(
            ghi=ghi, x=x, doy=doy, temp=temp, tilt=tilt, azimuth=az, dc_ac=ratio
        )
    return total


# --- the world builder ----------------------------------------------------------------------


def _week_median(*, clean_row: np.ndarray, cut_idx: int) -> np.ndarray:
    """Return the median of a Primary's cleaned power per half-hour of the week (length 336)."""
    idx = np.arange(cut_idx)
    v = clean_row[:cut_idx]
    ok = np.isfinite(v)
    frame = pl.DataFrame({"slot": idx[ok] % 336, "v": v[ok].astype(np.float64)})
    med = frame.group_by("slot").agg(m=pl.col("v").median()).sort("slot")
    out = np.zeros(336)
    out[med["slot"].to_numpy()] = med["m"].to_numpy()
    return out


class WorldBuilder:
    """Builds the pretraining target of each condition for one cutoff's row set."""

    def __init__(
        self, *, primary: np.ndarray, target_idx: np.ndarray, x: np.ndarray, y: np.ndarray,
        standardiser: object, clean: np.ndarray,
    ) -> None:  # fmt: skip
        """Hold the rows' keys and every quantity the conditions share."""
        self.primary, self.target_idx, self.x, self.y = primary, target_idx, x, y.astype(np.float64)
        self.t_us = k621.time_us(target_idx=target_idx)
        self.idx6 = index_of(sites=SIX)
        self.idxn = index_of(sites=NATIVE)
        self.cap = k621.effective_capacity_mw()
        self.ghi, self.temp = k621.raw_weather(x=x, standardiser=standardiser)
        self.series = {f: p5.farm_norm_series(farm=f) for f in FARMS}
        self.unique, self.inverse = np.unique(self.t_us, return_inverse=True)
        self.covered = np.isin(primary, list(self.idx6.values()) + list(self.idxn.values()))
        self.gap = self._gap_mask()
        cut_idx = int(target_idx.max()) + 1
        self.week_median = {
            p: _week_median(clean_row=clean[p], cut_idx=cut_idx) for p in self.idx6.values()
        }
        self.slot = target_idx % 336
        self._cache: dict[tuple, np.ndarray] = {}
        hour = (self.t_us // 3_600_000_000) % 24
        self.slice_midday = (hour >= 10) & (hour <= 15)
        self.median_midday = self._sunny_midday_median()

    def _gap_mask(self) -> np.ndarray:
        gap = np.zeros(len(self.unique), dtype=bool)
        for f in FARMS:
            for salt in (None, 0):
                gap |= np.isnan(
                    p9.series_at(times_us=self.unique, series=self.series[f], shuffle_salt=salt)
                )
        return gap[self.inverse] & self.covered

    def _sunny_midday_median(self) -> dict[int, float]:
        """Median sunny-midday reading (points) per covered Primary: signed power for the six, |S| for the five."""
        clear = k621.haurwitz_w_m2(cos_zenith=self.x[:, 14].astype(np.float64))
        kt = self.ghi / np.maximum(clear, 1.0)
        hour = (self.t_us // 3_600_000_000) % 24
        months = pl.Series(self.t_us).cast(pl.Datetime("us")).dt.month().to_numpy()
        sel = (
            (kt >= SUNNY_KT)
            & (hour >= 10)
            & (hour <= 15)
            & (months >= 4)
            & (months <= 8)
            & ~self.gap
        )
        return {
            p: float(np.median(self.y[sel & (self.primary == p)]))
            for p in [*self.idx6.values(), *self.idxn.values()]
        }

    # -- shapes (all indexed over every row, zero outside the Primary)

    def _farm(self, *, farm: str) -> np.ndarray:
        key = ("farm", farm)
        if key not in self._cache:
            self._cache[key] = p9.series_at(
                times_us=self.unique, series=self.series[farm], shuffle_salt=None
            )
        return self._cache[key]

    def fleet(self, *, j: int, p: int, rotation: int, physics_only: bool = False) -> np.ndarray:
        """Return the fleet shape of Primary `p`: half the energy a real farm, half a physics mixture.

        The shape is scaled so that its 99th percentile over daylight rows is 1.
        """
        key = ("fleet", p, rotation, physics_only)
        if key not in self._cache:
            sel = self.primary == p
            rng = np.random.default_rng(656 + 7919 * p)
            mix = (
                physics_mixture(
                    ghi=self.ghi[sel],
                    x=self.x[sel],
                    times_us=self.t_us[sel],
                    temp=self.temp[sel],
                    rng=rng,
                )
                / pv.u99()
            )
            if physics_only:
                shape = mix
            else:
                farm = np.nan_to_num(
                    self._farm(farm=FARMS[(j + rotation) % len(FARMS)])[self.inverse[sel]]
                )
                shape = 0.5 * farm + 0.5 * mix * (farm.sum() / max(float(mix.sum()), 1e-9))
            lit = shape > 0.01
            shape = shape / max(float(np.percentile(shape[lit], 99)), 1e-9)
            full = np.zeros(len(self.primary))
            full[sel] = shape
            self._cache[key] = full
        return self._cache[key]

    def climatology(self, *, j: int, p: int) -> np.ndarray:
        """Return the multi-year mean of the fleet shape by day of the year and half-hour, smoothed over +/-7 days."""
        key = ("clim", p)
        if key not in self._cache:
            sel = self.primary == p
            shape = self.fleet(j=j, p=p, rotation=0)[sel]
            doy = doy_of(times_us=self.t_us[sel]).astype(int)
            hh = ((self.t_us[sel] % DAY_US) // 1_800_000_000).astype(int)
            table = (
                pl.DataFrame({"doy": doy, "hh": hh, "v": shape})
                .group_by(["doy", "hh"])
                .agg(m=pl.col("v").mean())
            )
            grid = np.zeros((367, 48))
            grid[table["doy"].to_numpy(), table["hh"].to_numpy()] = table["m"].to_numpy()
            smooth = sum(np.roll(grid, off, axis=0) for off in range(-7, 8)) / 15.0
            full = np.zeros(len(self.primary))
            full[sel] = smooth[doy, hh]  # ty: ignore[not-subscriptable]
            self._cache[key] = full
        return self._cache[key]

    def boxcar(self, *, p: int, day_energy: float, cap: float) -> np.ndarray:
        """Return a load-shift boxcar drop (positive = demand falls).

        On 30% of April-to-September days a drop of 3 or 4 half-hours starts between 08:00 and 17:00 UTC.
        Each event removes at most `day_energy` (points x half-hours) in all, the energy of one average
        April-to-September day of the L2 fleet, and its height never exceeds `cap` (the Primary's median
        sunny-midday reading), because a load transfer cannot remove more than the load.
        """
        key = ("boxcar", p)
        if key not in self._cache:
            sel = np.nonzero(self.primary == p)[0]
            t = self.t_us[sel]
            day = t // DAY_US
            hh = ((t % DAY_US) // 1_800_000_000).astype(int)
            months = pl.Series(t).cast(pl.Datetime("us")).dt.month().to_numpy()
            rng = np.random.default_rng(6560 + p)
            shape = np.zeros(len(sel))
            for d in np.unique(day):
                if not (4 <= months[day == d][0] <= 9) or rng.random() > 0.3:
                    continue
                start = int(rng.integers(16, 30))
                width = int(rng.integers(3, 5))
                shape[(day == d) & (hh >= start) & (hh < start + width)] = 1.0
            full = np.zeros(len(self.primary))
            full[sel] = min(day_energy / 3.5, cap) * shape
            self._cache[key] = full
        return self._cache[key]

    def cloud(self, *, p: int, mean: float) -> np.ndarray:
        """Return a linear cloud-following load (positive = demand rises) averaging `mean` over daylight rows."""
        key = ("cloud", p)
        if key not in self._cache:
            sel = self.primary == p
            clear = k621.haurwitz_w_m2(cos_zenith=self.x[:, 14].astype(np.float64))
            lit = clear > 10.0
            cloud = np.where(lit, np.clip(1.0 - self.ghi / np.maximum(clear, 1e-9), 0.0, 1.0), 0.0)
            day = self.t_us // DAY_US
            frame = pl.DataFrame(
                {"p": self.primary, "d": day, "t": self.target_idx, "cloud": cloud, "lit": lit}
            )
            frame = frame.unique(subset=["p", "t"], keep="first")
            daily = frame.filter(pl.col("lit")).group_by(["p", "d"]).agg(r=pl.col("cloud").mean())
            joined = pl.DataFrame({"p": self.primary, "d": day}).join(
                daily, on=["p", "d"], how="left"
            )
            response = np.where(lit, joined["r"].fill_null(0.0).to_numpy(), 0.0) * sel
            scale = mean / max(float(response[sel & lit].mean()), 1e-9)
            self._cache[key] = response * scale
        return self._cache[key]

    # -- targets

    def injection(self, *, cond: Condition, p: int, j: int) -> np.ndarray:
        """Return the injected power `I` over all rows, in points (positive lowers net demand)."""
        m = self.median_midday[p]
        n = len(self.primary)
        sel = self.primary == p
        if cond.kind == "twin":
            return np.zeros(n)
        if cond.kind == "fleet":
            return cond.level * m * self.fleet(j=j, p=p, rotation=cond.rotation)
        if cond.kind == "phys":
            return cond.level * m * self.fleet(j=j, p=p, rotation=0, physics_only=True)
        l2 = LEVELS[PLACEBO_LEVEL] * m * self.fleet(j=j, p=p, rotation=0)
        if cond.kind == "clim":
            return LEVELS[PLACEBO_LEVEL] * m * self.climatology(j=j, p=p)
        if cond.kind == "boxcar":
            _, first = np.unique(self.target_idx[sel], return_index=True)
            t_u = self.t_us[sel][first]
            months = pl.Series(t_u).cast(pl.Datetime("us")).dt.month().to_numpy()
            summer = (months >= 4) & (months <= 9)
            n_days = max(len(np.unique(t_u[summer] // DAY_US)), 1)
            return self.boxcar(p=p, day_energy=float(l2[sel][first][summer].sum()) / n_days, cap=m)
        if cond.kind == "sunload":
            phys = self.fleet(j=j, p=p, rotation=0, physics_only=True)
            fleet = self.fleet(j=j, p=p, rotation=0)
            mid = sel & self.slice_midday
            k_scale = float(np.median(fleet[mid & (fleet > 0.3)])) / max(
                float(np.median(phys[mid & (phys > 0.3)])), 1e-9
            )
            return -LEVELS["l6"] * m * k_scale * phys
        if cond.kind == "cloud":
            return -self.cloud(p=p, mean=CLOUD_SHARE * LEVELS[PLACEBO_LEVEL] * m)
        raise AssertionError(cond.kind)

    def reactive(self, *, cond: Condition, p: int) -> np.ndarray:
        """Return `Q` over all rows, in points, of a six-Primary under the condition's reactive-power arm."""
        if cond.arm == "q0":
            return np.zeros(len(self.primary))
        if cond.arm == "floor":
            return np.full(len(self.primary), FLOOR_Q_SHARE * abs(self.median_midday[p]))
        return PF_Q_SHARE * np.abs(self.week_median[p][self.slot])

    def target(self, *, cond: Condition) -> np.ndarray:
        """Return the pretraining target (points) of every row under the condition."""
        y = self.y.copy()
        for j, p in enumerate(self.idx6.values()):
            sel = self.primary == p
            inj = self.injection(cond=cond, p=p, j=j)
            if cond.arm == "sig":
                y[sel] = self.y[sel] - inj[sel]
            else:
                q = self.reactive(cond=cond, p=p)
                y[sel] = np.sqrt((self.y[sel] - inj[sel]) ** 2 + q[sel] ** 2)
        if cond.arm == "pf":
            for j, p in enumerate(self.idxn.values()):
                sel = self.primary == p
                inj = self.injection(cond=cond, p=p, j=len(self.idx6) + j)
                y[sel] = np.sqrt(
                    (NATIVE_P_SHARE * self.y[sel] - inj[sel]) ** 2
                    + (NATIVE_Q_SHARE * self.y[sel]) ** 2
                )
        return y
