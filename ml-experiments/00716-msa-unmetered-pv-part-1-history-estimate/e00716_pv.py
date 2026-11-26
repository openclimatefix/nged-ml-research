# ruff: noqa: E501, PT018
"""Experiment 716: the unmetered PV estimate `PV̂ = C(i, f) × u(i, t)` (design section 3, WP1).

`C` is the as-of AC capacity of Primary `i` for fold `f` in megawatts, and `u` is the physics yield
curve of experiment 405 (`e00405_pv.unit_shape` divided by `u99()`). The network reads
`p = PV̂ / capacity(i) / target_std(i, f)`, clipped to 0 to 10.

This module holds the as-of capacity table, the dense history estimate over the whole grid, the ENS
target-row table, and the three placebo builders: `H-S` (the sky of 28 days earlier), `HT-S` (the ENS
run of 28 days earlier), and `H-X` (the other Primary's estimate). It reads no neural network.

Dense-table column order (read from `e00632_data.dense_raw`): weather columns 0 to 6 hold the
CAMS shortwave at column 0 and the ERA5 2 m temperature at column 2, and columns 7 to 27 are the
date-time columns of experiment 397's rows 14 to 34, so column 7 is the cosine of the zenith and
columns 8 and 9 are the sine and cosine of the azimuth (`e00634_common.unit_curve` reads the
same columns 14, 15, and 16 of a Candidate row). `check_dense_columns` checks the order against
the Candidate's rows.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SIX: Final = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The signed-MW Primaries of every counted contrast."""
EIGHT: Final = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The MVA-only Primaries of every counted contrast."""
ABSENT: Final = ("D15", "D18")
"""Primaries with no capacity: `p` is zero and flagged absent in every arm."""
MVA_RESCALED: Final = ("D2", "D3", "D4", "D19", "D21")
"""The Primaries whose `L0` capacity is rescaled by `factor / 0.7858` (experiment 673)."""
FOLD_CUTOFFS_649: Final = ("2025-07", "2025-10", "2026-03", "2026-06")
"""The cutoffs that experiment 649 stores (one calibration factor per cutoff)."""
CUTOFFS_662: Final = ("2026-03", "2026-06")
"""The cutoffs that experiment 662 stores; the factors of both come from data to 2026-06."""
P_CLIP: Final = 10.0
"""Twice the power clip: `p` is clipped to 0 to this value."""
SHIFT_DAYS: Final = 28
"""The shift of the placebos `H-S` and `HT-S`."""
K_CAP: Final = 1.5
"""The cap of the clear-sky index of a placebo."""
LOW_SUN_W_M2: Final = 50.0
"""Below this donor clear-sky irradiance the placebo uses the donor day's daylight ratio."""
PARTNERS: Final[dict[str, str]] = {
    "D12": "D14", "D22": "D11", "D5": "D19", "D10": "D21", "D2": "D8", "D16": "D3", "D9": "D4",
}  # fmt: skip
"""The seven pairs of the `H-X` placebo: rank `r` of experiment 673's PV share with rank `r + 7`."""
GRID_STEPS_PER_DAY: Final = 48
DRAW_COLUMNS: Final = {
    "ghi": 0,
    "temperature": 2,
    "cos_zenith": 7,
    "sin_azimuth": 8,
    "cos_azimuth": 9,
}
"""The columns of the dense raw table (`e00632_data.dense_raw`) that the yield curve reads."""

NAME_649: Final = "00649-pv-capacity-table-season-year-from-stored-probes"
NAME_662: Final = "00662-pv-probe-through-the-mva-fold-matched-level-redesign"
CapacitySource = Literal["649", "662"]


def partner_table() -> dict[str, str]:
    """Return the symmetric `H-X` partner of every Primary that has one (14 entries)."""
    out = dict(PARTNERS)
    out.update({v: k for k, v in PARTNERS.items()})
    assert len(out) == 14 and set(out) == set(SIX) | set(EIGHT), sorted(out)
    return out


@dataclass(frozen=True)
class AsOfCapacity:
    """The as-of capacity of one Primary and fold, and what it looks ahead at.

    Attributes:
        site: The Primary's label (`D5`).
        fold: The fold month (`2025-11`).
        mw: The AC capacity in megawatts, 0 where absent.
        source: The experiment of the table, `649` or `662`, or `None` where absent.
        cutoff: The stored cutoff the value was read from.
        lookahead_probes: The probes' neural networks were trained to a cutoff after the fold's.
        lookahead_factors: The calibration factor or the 0.7858 rescale comes from data after the fold's
            cutoff.
    """

    site: str
    fold: str
    mw: float
    source: str | None
    cutoff: str | None
    lookahead_probes: bool
    lookahead_factors: bool


def latest_cutoff(*, stored: tuple[str, ...], fold: str) -> tuple[str, bool]:
    """Return the latest stored cutoff at or before `fold`, and whether that rule had to look ahead.

    The fold's cutoff is the start of the fold month. When no stored cutoff is at or before it, the
    earliest stored cutoff is returned and the flag is true.
    """
    at_or_before = [c for c in stored if c <= fold]
    if at_or_before:
        return max(at_or_before), False
    return min(stored), True


def as_of_capacity(
    *, site: str, fold: str, table_649: pl.DataFrame, table_662: pl.DataFrame
) -> AsOfCapacity:
    """Return the as-of capacity of `site` for `fold` by the rule of design section 3.3.

    The six read experiment 649's `cal_p50` at the latest stored cutoff at or before the fold. D2, D3,
    D4, D19, and D21 read experiment 662's `L0` `cal_p50 × factor / 0.7858`, D8 and D22 its `L0`
    `cal_p50`, and D12 its `FA` `cal_p50`, all at the latest stored cutoff of experiment 662 at or
    before the fold. Experiment 662 stores 2026-03 and 2026-06 only, and its factors come from data to
    2026-06, so every eight value of a fold before 2026-06 looks ahead at the factors, and the folds
    before 2026-03 look ahead at the probes as well. D15 and D18 have no capacity.
    """
    if site in ABSENT:
        return AsOfCapacity(site, fold, 0.0, None, None, False, False)
    if site in SIX:
        cutoff, ahead = latest_cutoff(stored=FOLD_CUTOFFS_649, fold=fold)
        rows = table_649.filter(
            (table_649["probe_year"] == 2025)
            & (table_649["cutoff"] == cutoff)
            & (table_649["site"] == site)
        )
        assert rows.height == 1, (site, cutoff, rows.height)
        return AsOfCapacity(site, fold, float(rows["cal_p50"][0]), "649", cutoff, ahead, ahead)
    assert site in EIGHT, site
    cutoff, ahead = latest_cutoff(stored=CUTOFFS_662, fold=fold)
    method = "FA" if site == "D12" else "L0"
    rows = table_662.filter(
        (table_662["cutoff"] == cutoff)
        & (table_662["site"] == site)
        & (table_662["method"] == method)
    )
    assert rows.height == 1, (site, cutoff, method, rows.height)
    row = rows.row(0, named=True)
    scale = row["factor"] / 0.7858 if site in MVA_RESCALED else 1.0
    return AsOfCapacity(
        site, fold, float(row["cal_p50"] * scale), "662", cutoff,
        ahead, fold < "2026-06",
    )  # fmt: skip


def read_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read experiment 649's and 662's stored capacity tables (the only files this module reads)."""
    t649 = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / NAME_649 / "work" / "capacity_table.parquet")
    t662 = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR / NAME_662 / "work" / "analysis" / "capacity_table.csv"
    )
    return t649, t662


def capacity_table(*, fold: str, sites: list[str]) -> list[AsOfCapacity]:
    """Return the as-of capacity of every Primary of `sites` (in the 16-Primary order) for `fold`."""
    t649, t662 = read_tables()
    return [as_of_capacity(site=s, fold=fold, table_649=t649, table_662=t662) for s in sites]


def probe_table_at_cutoff(
    *, cutoff: str, table_649: pl.DataFrame, table_662: pl.DataFrame
) -> dict[str, float]:
    """Return the `probe` column of `e00673_deliverable.probe_table` at `cutoff`.

    The function is a copy with a `cutoff` argument.

    Source: `e00673_deliverable.probe_table`, lines 255 to 308 (experiment 673), whose cutoff is the
    literal `2026-06`. The six read `cal_p50` and the eight read `cal_p50 × rescale`, with the rescale
    `factor / 0.7858` for D2, D3, D4, D19, and D21 and 1 for D8, D22, and D12 (method `FA` for D12).
    """
    out: dict[str, float] = {}
    for site in (*SIX, *EIGHT):
        if site in SIX:
            r = table_649.filter(
                (table_649["probe_year"] == 2025)
                & (table_649["cutoff"] == cutoff)
                & (table_649["site"] == site)
            ).row(0, named=True)
            out[site] = float(r["cal_p50"])
        else:
            method = "FA" if site == "D12" else "L0"
            r = table_662.filter(
                (table_662["cutoff"] == cutoff)
                & (table_662["site"] == site)
                & (table_662["method"] == method)
            ).row(0, named=True)
            out[site] = float(
                r["cal_p50"] * (r["factor"] / 0.7858 if site in MVA_RESCALED else 1.0)
            )
    return out


def lookahead_statement(*, caps: list[AsOfCapacity]) -> list[str]:
    """Return the printed look-ahead of every Primary of one fold (design KF5)."""
    lines = []
    for c in caps:
        if c.source is None:
            lines.append(f"{c.fold} {c.site}: no capacity, p is zero and flagged absent")
            continue
        lines.append(
            f"{c.fold} {c.site}: {c.mw:.3f} MW from experiment {c.source} cutoff {c.cutoff}; "
            f"probes look ahead: {c.lookahead_probes}; factors look ahead: {c.lookahead_factors}"
        )
    return lines


# ------------------------------------------------------------------ the yield curve


def day_of_year(*, grid_start_us: int, n_grid: int, step_us: int = 1_800_000_000) -> np.ndarray:
    """Return the day of the year (1 to 366, float) of every grid stamp.

    The calculation is the same as in `e00634_common.unit_curve`.
    """
    stamps = grid_start_us + np.arange(n_grid, dtype=np.int64) * step_us
    return pl.Series(stamps).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy().astype(np.float64)


def yield_curve(
    *, ghi_mean: np.ndarray, ghi_sd: np.ndarray, cos_zenith: np.ndarray, sin_azimuth: np.ndarray,
    cos_azimuth: np.ndarray, doy: np.ndarray, temperature_c: np.ndarray, u99: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Return the mean and standard deviation of the yield curve `u`: `unit_shape / u99`.

    `u99` defaults to the stored constant (0.879) of experiment 405.
    """
    import e00405_pv as pv  # ty: ignore[unresolved-import]

    mean, sd = pv.unit_shape(
        ghi_mean=ghi_mean, ghi_sd=ghi_sd, cos_zenith=cos_zenith, sin_azimuth=sin_azimuth,
        cos_azimuth=cos_azimuth, doy=doy, temperature_c=temperature_c,
    )  # fmt: skip
    scale = pv.u99() if u99 is None else u99
    return mean / scale, sd / scale


def share_of_capacity(*, capacity_mw: float, u: np.ndarray, power_capacity_mw: float) -> np.ndarray:
    """Return `PV̂ / capacity` for one Primary: the as-of capacity times `u`, over the power's divisor."""
    return capacity_mw * u / power_capacity_mw


def to_p(*, share: np.ndarray, target_std: float | np.ndarray) -> np.ndarray:
    """Return `p = share / target_std`, clipped to 0 to 10, as float32. No mean is subtracted."""
    return np.clip(share / target_std, 0.0, P_CLIP).astype(np.float32)


@dataclass
class DenseEstimate:
    """The history estimate over the whole grid.

    Attributes:
        share: (16, G) float32, `PV̂ / capacity` (no standardisation), 0 where absent.
        ok: (16, G) bool, whether the estimate exists.
    """

    share: np.ndarray
    ok: np.ndarray


def dense_share(
    *, raw: np.ndarray, doy: np.ndarray, capacities_mw: np.ndarray, power_capacity_mw: np.ndarray,
    ghi_override: np.ndarray | None = None, temperature_override: np.ndarray | None = None,
    u99: float | None = None,
) -> DenseEstimate:  # fmt: skip
    """Build `PV̂ / capacity` for every Primary and grid stamp from the dense raw table (CAMS history).

    Args:
        raw: (16, G, 28) the raw dense table; column 0 shortwave, 2 temperature, 7 to 9 geometry.
        doy: (G,) day of the year.
        capacities_mw: (16,) the as-of capacity, 0 for an absent Primary.
        power_capacity_mw: (16,) the divisor of the power (`effective_capacity_mw`).
        ghi_override: (16, G) a replacement shortwave (the `H-S` placebo), else column 0 of `raw`.
        temperature_override: (16, G) a replacement temperature, else column 2 of `raw`.
        u99: The scaling constant, default the stored one.

    Returns:
        The share and the flag; a stamp with a non-finite shortwave or temperature is not `ok`.
    """
    n_primary, n_grid = raw.shape[:2]
    share = np.zeros((n_primary, n_grid), dtype=np.float32)
    ok = np.zeros((n_primary, n_grid), dtype=bool)
    for i in range(n_primary):
        if capacities_mw[i] <= 0.0:
            continue
        ghi = np.asarray(
            raw[i, :, DRAW_COLUMNS["ghi"]] if ghi_override is None else ghi_override[i],
            dtype=np.float64,
        )
        temp = np.asarray(
            raw[i, :, DRAW_COLUMNS["temperature"]]
            if temperature_override is None
            else temperature_override[i],
            dtype=np.float64,
        )
        finite = np.isfinite(ghi) & np.isfinite(temp)
        u, _ = yield_curve(
            ghi_mean=np.where(finite, ghi, 0.0), ghi_sd=np.zeros(n_grid),
            cos_zenith=np.asarray(raw[i, :, DRAW_COLUMNS["cos_zenith"]], dtype=np.float64),
            sin_azimuth=np.asarray(raw[i, :, DRAW_COLUMNS["sin_azimuth"]], dtype=np.float64),
            cos_azimuth=np.asarray(raw[i, :, DRAW_COLUMNS["cos_azimuth"]], dtype=np.float64),
            doy=doy, temperature_c=np.where(finite, temp, 0.0), u99=u99,
        )  # fmt: skip
        share[i] = np.where(
            finite,
            share_of_capacity(
                capacity_mw=capacities_mw[i], u=u, power_capacity_mw=power_capacity_mw[i]
            ),
            0.0,
        )
        ok[i] = finite
    return DenseEstimate(share=share, ok=ok)


def p_dense(*, estimate: DenseEstimate, target_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the network's input `p` (16, G) and its `ok` flag from a dense estimate."""
    p = to_p(share=estimate.share, target_std=target_std[:, None])
    return np.where(estimate.ok, p, 0.0).astype(np.float32), estimate.ok


# ------------------------------------------------------------------ placebo builders


def clear_sky(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return the Haurwitz clear-sky irradiance (W/m2) from the cosine of the zenith (`studies.baselines`)."""
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    return haurwitz_w_m2(apparent_zenith_deg=zenith)


@dataclass
class SkyTransfer:
    """The result of `transfer_sky`: the new mean and spread, and where a rule bound."""

    mean: np.ndarray
    sd: np.ndarray
    cap_binds: np.ndarray
    low_sun_binds: np.ndarray


def transfer_sky(
    *, donor_mean: np.ndarray, donor_sd: np.ndarray | None, clear_donor: np.ndarray,
    clear_own: np.ndarray, cap: float = K_CAP, low_sun: float = LOW_SUN_W_M2,
) -> SkyTransfer:  # fmt: skip
    """Move the sky of a donor half-hour onto the sun position of the token.

    One clear-sky function is used on both sides.

    `k` is the donor's irradiance over the Haurwitz irradiance at the donor's own zenith, capped at
    `cap`. Where the donor's clear-sky irradiance is under `low_sun` W/m2, `k` is the donor day's
    daylight ratio of summed irradiance to summed clear-sky irradiance, with the day taken as the
    last axis of the arrays (one row of 48 half-hours). The spread is scaled by the same factor as
    the mean. With the donor equal to the token, the result equals the donor's own values wherever
    neither rule binds.

    Args:
        donor_mean: (..., 48) irradiance of the donor half-hours.
        donor_sd: (..., 48) spread of the donor half-hours, or None.
        clear_donor: (..., 48) Haurwitz irradiance at the donor's zenith.
        clear_own: (..., 48) Haurwitz irradiance at the token's zenith.
        cap: The cap of `k`.
        low_sun: The donor clear-sky threshold.

    Returns:
        The new mean and spread and the two masks.
    """
    daylight = clear_donor >= low_sun
    safe = np.where(daylight, clear_donor, 1.0)
    k_raw = np.where(daylight, donor_mean / safe, 0.0)
    day_clear = np.where(daylight, clear_donor, 0.0).sum(axis=-1, keepdims=True)
    day_ghi = np.where(daylight, donor_mean, 0.0).sum(axis=-1, keepdims=True)
    ratio = np.where(day_clear > 0, day_ghi / np.maximum(day_clear, 1e-9), 0.0)
    low = (~daylight) & (clear_own > 0)
    k_unc = np.where(daylight, k_raw, np.broadcast_to(ratio, k_raw.shape))
    k = np.minimum(k_unc, cap)
    new_mean = k * clear_own
    shrink = np.where(k_unc > 0, k / np.maximum(k_unc, 1e-12), 1.0)
    if donor_sd is None:
        new_sd = np.zeros_like(new_mean)
    else:
        sd_index = np.where(daylight, donor_sd / safe, 0.0) * shrink
        sd_low = (
            np.where(
                day_clear > 0,
                np.where(daylight, donor_sd, 0.0).sum(axis=-1, keepdims=True)
                / np.maximum(day_clear, 1e-9),
                0.0,
            )
            * shrink
        )
        new_sd = np.where(daylight, sd_index, sd_low) * clear_own
    return SkyTransfer(
        mean=new_mean, sd=new_sd, cap_binds=daylight & (k_raw > cap), low_sun_binds=low,
    )  # fmt: skip


def shifted_sky(
    *, raw: np.ndarray, shift_days: int = SHIFT_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Return the `H-S` shortwave and temperature of every Primary and stamp: the sky of `shift_days` earlier.

    Each token at grid index `t` reads the sky of `t - 48 × shift_days` with its own sun position:
    `ghi = k(t - 28 d) × clear(t)`. The temperature is the donor's. Tokens whose donor lies before
    the grid start keep their own sky and are listed by the returned mask. The day of a token is the
    block of 48 grid indices that contains it, so the donor day is the block of the donor stamp.

    Returns:
        The shortwave (16, G), the temperature (16, G), the mask of tokens with no donor (16, G),
        and the mask where the cap or the low-sun rule bound (16, G).
    """
    n_primary, n_grid = raw.shape[:2]
    n_blocks = n_grid // GRID_STEPS_PER_DAY
    used = n_blocks * GRID_STEPS_PER_DAY
    shift = shift_days * GRID_STEPS_PER_DAY
    ghi = np.array(raw[:, :, DRAW_COLUMNS["ghi"]], dtype=np.float64)
    temp = np.array(raw[:, :, DRAW_COLUMNS["temperature"]], dtype=np.float64)
    cosz = np.asarray(raw[:, :, DRAW_COLUMNS["cos_zenith"]], dtype=np.float64)
    out_ghi, out_temp = ghi.copy(), temp.copy()
    missing = np.zeros((n_primary, n_grid), dtype=bool)
    binds = np.zeros((n_primary, n_grid), dtype=bool)
    clear = clear_sky(cos_zenith=cosz)
    for i in range(n_primary):
        donor = np.full(used, -1, dtype=np.int64)
        idx = np.arange(used)
        donor[idx >= shift] = idx[idx >= shift] - shift
        has = donor >= 0
        missing[i, :used] = ~has
        missing[i, used:] = True
        d = np.where(has, donor, 0)
        d_ghi = ghi[i, d].reshape(n_blocks, GRID_STEPS_PER_DAY)
        d_clear = clear[i, d].reshape(n_blocks, GRID_STEPS_PER_DAY)
        own_clear = clear[i, :used].reshape(n_blocks, GRID_STEPS_PER_DAY)
        res = transfer_sky(
            donor_mean=np.nan_to_num(d_ghi), donor_sd=None, clear_donor=d_clear, clear_own=own_clear
        )
        new = res.mean.reshape(-1)
        keep = has & np.isfinite(ghi[i, d])
        out_ghi[i, :used] = np.where(keep, new, ghi[i, :used])
        out_temp[i, :used] = np.where(has & np.isfinite(temp[i, d]), temp[i, d], temp[i, :used])
        binds[i, :used] = (res.cap_binds | res.low_sun_binds).reshape(-1) & keep
    return out_ghi, out_temp, missing, binds


def donor_origin_day(
    *, origin_day: np.ndarray, first_ens_day: int, shift_days: int = SHIFT_DAYS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the `HT-S` donor origin day of each sample and a flag for the exception.

    The donor is the run issued 28 days earlier. For an origin in the first 28 days of the ENS record
    the donor is the run 28 days later, which is issued after the sample's origin and carries no
    information about the sample's target day. The flag marks those origins.
    """
    day = np.asarray(origin_day, dtype=np.int64)
    early = day - shift_days < first_ens_day
    return np.where(early, day + shift_days, day - shift_days), early


def host_partner_index(*, sites: list[str]) -> np.ndarray:
    """Return the `H-X` partner's index in the 16-Primary order for each Primary, -1 where none (D15, D18)."""
    table = partner_table()
    return np.array([sites.index(table[s]) if s in table else -1 for s in sites], dtype=np.int64)


def partner_p(
    *, share: np.ndarray, ok: np.ndarray, partner: np.ndarray, target_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the `H-X` input: each host reads its partner's share of capacity over the host's own `target_std`.

    Args:
        share: (16, G) `PV̂ / capacity` of each Primary.
        ok: (16, G) its flag.
        partner: (16,) partner index per host, -1 for none.
        target_std: (16,) the hosts' standard deviations.

    Returns:
        `p` (16, G) and its flag; a host without a partner gets zeros and a false flag.
    """
    has = partner >= 0
    src = np.where(has, partner, 0)
    p = to_p(share=share[src], target_std=target_std[:, None])
    flag = ok[src] & has[:, None]
    return np.where(flag, p, 0.0).astype(np.float32), flag


# ------------------------------------------------------------------ the ENS target row


@dataclass
class TargetEstimate:
    """The ENS target-row estimate in the (Primary, origin day, position) layout of `EnsTensors`.

    Attributes:
        p0: (16, n_days, 672, 2) float32, `p0_mean` and `p0_sd`.
        filled: (16, n_days, 672) bool.
    """

    p0: np.ndarray
    filled: np.ndarray


def target_row_pv(
    *, ghi: np.ndarray, ghi_sd: np.ndarray, temperature_c: np.ndarray, geometry: np.ndarray,
    doy: np.ndarray, share_scale: np.ndarray, target_std: np.ndarray, u99: float | None = None,
) -> np.ndarray:  # fmt: skip
    """Return `(N, 2)` `p0_mean`, `p0_sd` for N target rows of Candidate (float32 source) values.

    Args:
        ghi: (N,) ENS-mean shortwave in W/m2, un-standardised from the float32 rows.
        ghi_sd: (N,) ENS-spread shortwave.
        temperature_c: (N,) ENS-mean 2 m temperature.
        geometry: (N, 3) cosine of the zenith, sine and cosine of the azimuth (columns 14 to 16 of a row).
        doy: (N,) day of the year.
        share_scale: (N,) `capacity_mw / power_capacity_mw` of each row's Primary, 0 if absent.
        target_std: (N,) the row's Primary's `target_std`.
        u99: Scaling constant, default the stored one.
    """
    mean, sd = yield_curve(
        ghi_mean=ghi, ghi_sd=ghi_sd, cos_zenith=geometry[:, 0], sin_azimuth=geometry[:, 1],
        cos_azimuth=geometry[:, 2], doy=doy, temperature_c=temperature_c, u99=u99,
    )  # fmt: skip
    out = np.stack(
        [
            to_p(share=share_scale * mean, target_std=target_std),
            np.clip(share_scale * sd / target_std, 0.0, P_CLIP).astype(np.float32),
        ],
        axis=1,
    )
    return out * (share_scale > 0)[:, None]


def un_standardise(*, x: np.ndarray, column: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return one column of standardised Candidate rows in raw units, as float64 from the float32 rows."""
    return x[:, column].astype(np.float64) * std[column] + mean[column]


def scatter_target(
    *, values: np.ndarray, primary: np.ndarray, origin_idx: np.ndarray, target_idx: np.ndarray,
    first_day: int, n_days: int, n_steps: int, n_primary: int = 16, native_offset_steps: int = 6,
) -> TargetEstimate:  # fmt: skip
    """Scatter per-row `(N, 2)` values into the table layout of experiment 713's `ens_table`.

    A row's origin is the native origin index; its contract origin is `native_offset_steps` later, and
    its day index is `contract // 48 - first_day`, its position `target_idx - contract`.
    """
    c_idx = origin_idx + native_offset_steps
    day = c_idx // GRID_STEPS_PER_DAY - first_day
    rel = target_idx - c_idx
    keep = (rel >= 0) & (rel < n_steps) & (day >= 0) & (day < n_days)
    p0 = np.zeros((n_primary, n_days, n_steps, 2), dtype=np.float32)
    filled = np.zeros((n_primary, n_days, n_steps), dtype=bool)
    p0[primary[keep], day[keep], rel[keep]] = values[keep]
    filled[primary[keep], day[keep], rel[keep]] = True
    return TargetEstimate(p0=p0, filled=filled)


def shifted_target(
    *, mean: np.ndarray, sd: np.ndarray, cos_zenith_donor: np.ndarray, cos_zenith_own: np.ndarray,
) -> SkyTransfer:  # fmt: skip
    """Return the `HT-S` sky of a donor run moved to the target half-hours' own sun positions.

    The arrays are (..., 48): one target-row day of the donor run (mean and spread of the ENS
    shortwave at the donor's valid times) and the two zeniths. The same clear-sky-index transfer is
    applied to the mean and the spread.
    """
    return transfer_sky(
        donor_mean=mean, donor_sd=sd, clear_donor=clear_sky(cos_zenith=cos_zenith_donor),
        clear_own=clear_sky(cos_zenith=cos_zenith_own),
    )  # fmt: skip


def check_dense_columns(
    *,
    raw: np.ndarray,
    x_rows: np.ndarray,
    grid_index: np.ndarray,
    primary: np.ndarray,
    atol: float = 1e-4,
) -> str:
    """Check the dense table's geometry columns against Candidate rows at the same stamps (KF4, column order).

    `raw[primary, grid_index, 7:10]` must equal `x_rows[:, 14:17]`.
    """
    dense = np.asarray(raw[primary, grid_index, 7:10], dtype=np.float64)
    rows = x_rows[:, 14:17].astype(np.float64)
    err = float(np.abs(dense - rows).max())
    assert err <= atol, f"dense geometry columns differ from the Candidate rows by {err}"
    return f"dense geometry columns equal the Candidate rows to {err:.2e}"


def check_dense_weather(
    *,
    raw: np.ndarray,
    x_rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    grid_index: np.ndarray,
    primary: np.ndarray,
    min_ghi_corr: float = 0.7,
    min_temp_corr: float = 0.9,
) -> str:
    """Check that the dense table's shortwave and temperature columns are the Candidate rows' quantities.

    This is KF4, the column identity check.

    The dense table holds the observed weather (ERA5 and CAMS) and the rows hold the numerical weather
    prediction at the row's lead, so the two cannot be equal. They must still correlate strongly at the
    same stamps, which catches a swapped or shifted column. The check cannot show that the dense values
    are right, only that the columns hold the quantity their index says.
    """
    ghi_rows = np.clip(x_rows[:, 0].astype(np.float64) * std[0] + mean[0], 0.0, None)
    temp_rows = x_rows[:, 2].astype(np.float64) * std[2] + mean[2]
    ghi_dense = np.asarray(raw[primary, grid_index, 0], dtype=np.float64)
    temp_dense = np.asarray(raw[primary, grid_index, 2], dtype=np.float64)
    ok = np.isfinite(ghi_dense) & np.isfinite(temp_dense)
    c_ghi = float(np.corrcoef(ghi_rows[ok], ghi_dense[ok])[0, 1])
    c_temp = float(np.corrcoef(temp_rows[ok], temp_dense[ok])[0, 1])
    assert c_ghi >= min_ghi_corr, f"dense shortwave correlates {c_ghi:.3f} with the rows'"
    assert c_temp >= min_temp_corr, f"dense temperature correlates {c_temp:.3f} with the rows'"
    return f"dense weather columns correlate with the Candidate rows' at the same stamps: shortwave {c_ghi:.3f}, temperature {c_temp:.3f}"


# ------------------------------------------------------------------ the raw ENS target table and the target estimates

N_ENS_STEPS: Final = 14 * GRID_STEPS_PER_DAY
"""Half-hours of the ENS target table per origin day (672)."""


@dataclass
class TargetRaw:
    """The raw ENS target-row quantities of the Candidate's rows.

    The tensors follow the (Primary, origin day, position) layout of `EnsTensors`.

    Attributes:
        values: (16, n_days, 672, 4) float32: ENS-mean shortwave, ENS-spread shortwave (W/m2), 2 m
            temperature (C), and the cosine of the zenith, from the float32 rows.
        azimuth: (16, n_days, 672, 2) float32: the sine and cosine of the azimuth.
        filled: (16, n_days, 672) bool.
        first_day: The day number (grid index // 48) of day index 0.
        run_us: (16, n_days) int64: the issue time of the 00Z run behind each row, 0 where absent.
    """

    values: np.ndarray
    azimuth: np.ndarray
    filled: np.ndarray
    first_day: int
    run_us: np.ndarray


def target_raw_from_rows(
    *, x: np.ndarray, primary: np.ndarray, origin_idx: np.ndarray, target_idx: np.ndarray,
    origin_us: np.ndarray, mean: np.ndarray, std: np.ndarray, first_day: int, n_days: int,
    native_offset_steps: int = 6, ensemble_lead_us: int = 9 * 3_600_000_000,
) -> TargetRaw:  # fmt: skip
    """Scatter the Candidate's standardised rows into a raw target table.

    The shortwave, its spread, and the temperature are un-standardised in float64 from the float32
    rows (experiment 713's device table is float16, which cannot meet the 1e-6 check of KF4), and the
    geometry columns 14 to 16 are read as they are, as `e00634_common.unit_curve` does.
    """
    origin_us_int = (
        np.asarray(origin_us).astype("datetime64[us]").astype(np.int64)
        if np.asarray(origin_us).dtype.kind == "M"
        else np.asarray(origin_us).astype(np.int64)
    )
    assert (origin_us_int % (24 * 3_600_000_000) == 9 * 3_600_000_000).all(), (
        "every native origin is 09:00 UTC (as experiment 713 asserts); the ENS run is the 00Z run of the day"
    )
    c_idx = origin_idx + native_offset_steps
    day = c_idx // GRID_STEPS_PER_DAY - first_day
    rel = target_idx - c_idx
    keep = (rel >= 0) & (rel < N_ENS_STEPS) & (day >= 0) & (day < n_days)
    n_primary = 16
    values = np.zeros((n_primary, n_days, N_ENS_STEPS, 4), dtype=np.float32)
    azimuth = np.zeros((n_primary, n_days, N_ENS_STEPS, 2), dtype=np.float32)
    filled = np.zeros((n_primary, n_days, N_ENS_STEPS), dtype=bool)
    run_us = np.zeros((n_primary, n_days), dtype=np.int64)
    p, d, r = primary[keep], day[keep], rel[keep]
    xk = x[keep]
    values[p, d, r, 0] = np.clip(un_standardise(x=xk, column=0, mean=mean, std=std), 0.0, None)
    values[p, d, r, 1] = np.clip(un_standardise(x=xk, column=7, mean=mean, std=std), 0.0, None)
    values[p, d, r, 2] = un_standardise(x=xk, column=2, mean=mean, std=std)
    values[p, d, r, 3] = xk[:, 14]
    azimuth[p, d, r] = xk[:, 15:17]
    filled[p, d, r] = True
    run_us[p, d] = origin_us[keep] - ensemble_lead_us
    return TargetRaw(
        values=values, azimuth=azimuth, filled=filled, first_day=first_day, run_us=run_us
    )


def merge_target_raw(*, base: TargetRaw, other: TargetRaw) -> TargetRaw:
    """Return `base` with the filled entries of `other` written over it.

    In use, the scored rows are written over the training rows.
    """
    assert base.first_day == other.first_day
    take = other.filled
    values = np.where(take[..., None], other.values, base.values)
    azimuth = np.where(take[..., None], other.azimuth, base.azimuth)
    run_us = np.where(take.any(axis=-1), other.run_us, base.run_us)
    return TargetRaw(
        values=values,
        azimuth=azimuth,
        filled=base.filled | take,
        first_day=base.first_day,
        run_us=run_us,
    )


def target_estimate(
    *, raw: TargetRaw, doy_of_day: np.ndarray, share_scale: np.ndarray, target_std: np.ndarray,
    donor_day: np.ndarray | None = None, u99: float | None = None,
) -> TargetEstimate:  # fmt: skip
    """Return the target-row estimate `p0_mean`, `p0_sd` for every filled row of `raw`.

    Args:
        raw: The raw target table.
        doy_of_day: (n_days, 672) day of the year of each target half-hour of each origin day.
        share_scale: (16,) `capacity_mw / power_capacity_mw`, 0 for an absent Primary.
        target_std: (16,) the fold's standard deviations.
        donor_day: (n_days,) the donor origin day index of the `HT-S` placebo, or None for the real run.
            A row whose donor row is missing is not filled in the result.
        u99: Scaling constant, default the stored one.
    """
    n_primary, n_days = raw.filled.shape[:2]
    p0 = np.zeros((n_primary, n_days, N_ENS_STEPS, 2), dtype=np.float32)
    filled = np.zeros_like(raw.filled)
    for i in range(n_primary):
        if share_scale[i] <= 0.0:
            continue
        days = np.nonzero(raw.filled[i].any(axis=1))[0]
        if len(days) == 0:
            continue
        own = raw.values[i, days]  # (D, 672, 4)
        ghi, sd, temp, cosz = (
            own[..., 0].astype(np.float64),
            own[..., 1].astype(np.float64),
            own[..., 2].astype(np.float64),
            own[..., 3].astype(np.float64),
        )
        ok = raw.filled[i, days]
        if donor_day is not None:
            src = donor_day[days]
            valid = (src >= 0) & (src < n_days)
            src_c = np.clip(src, 0, n_days - 1)
            ok = ok & raw.filled[i, src_c] & valid[:, None]
            donor = raw.values[i, src_c]
            blocks = (len(days), 14, GRID_STEPS_PER_DAY)
            res = shifted_target(
                mean=donor[..., 0].astype(np.float64).reshape(blocks), sd=donor[..., 1].astype(np.float64).reshape(blocks),
                cos_zenith_donor=donor[..., 3].astype(np.float64).reshape(blocks), cos_zenith_own=cosz.reshape(blocks),
            )  # fmt: skip
            ghi, sd = res.mean.reshape(ghi.shape), res.sd.reshape(sd.shape)
            temp = donor[..., 2].astype(np.float64)
        flat = ok.reshape(-1)
        az = raw.azimuth[i, days].reshape(-1, 2).astype(np.float64)
        geometry = np.stack([cosz.reshape(-1), az[:, 0], az[:, 1]], axis=1)
        values = np.zeros((flat.size, 2), dtype=np.float32)
        values[flat] = target_row_pv(
            ghi=ghi.reshape(-1)[flat], ghi_sd=sd.reshape(-1)[flat], temperature_c=temp.reshape(-1)[flat],
            geometry=geometry[flat], doy=doy_of_day[days].reshape(-1)[flat],
            share_scale=np.full(int(flat.sum()), share_scale[i]), target_std=np.full(int(flat.sum()), target_std[i]), u99=u99,
        )  # fmt: skip
        p0[i, days] = values.reshape(len(days), N_ENS_STEPS, 2)
        filled[i, days] = ok
    return TargetEstimate(p0=p0, filled=filled)


GRID_START_US: Final = 1_567_296_000_000_000
"""2019-09-01 00:00 UTC, the grid start of experiment 422 (checked in `e00713_check_inputs.py`)."""
STEP_US: Final = 1_800_000_000


def target_doy(*, first_day: int, n_days: int) -> np.ndarray:
    """Return the day of the year (n_days, 672) of every target half-hour of every origin day.

    The target half-hour at position `r` of origin day `d` has grid index `48 d + 24 + r`, where `d`
    counts days from the grid start and 24 is the contract origin at 12:00 UTC.
    """
    day = (first_day + np.arange(n_days, dtype=np.int64))[:, None]
    grid = day * GRID_STEPS_PER_DAY + 24 + np.arange(N_ENS_STEPS, dtype=np.int64)[None, :]
    stamp_us = GRID_START_US + grid * STEP_US
    out = pl.Series(stamp_us.reshape(-1)).cast(pl.Datetime("us")).dt.ordinal_day().to_numpy()
    return out.astype(np.float64).reshape(stamp_us.shape)


def monthly_mean_temperature(
    *, raw: np.ndarray, cutoff_idx: int, window_days: int = 730
) -> np.ndarray:
    """Return the temperature (16, G) with each token's value replaced by a monthly mean.

    The replacement is the Primary's mean 2 m temperature of the same calendar month.

    The mean is taken over the `window_days` before `cutoff_idx` (the fold's training days). This is the
    temperature of the CAMS-rule read-out (design section 7.5), because the history estimate otherwise
    reads ERA5 temperature, which a live service would not have.
    """
    n_primary, n_grid = raw.shape[:2]
    stamps = GRID_START_US + np.arange(n_grid, dtype=np.int64) * STEP_US
    month = pl.Series(stamps).cast(pl.Datetime("us")).dt.month().to_numpy()
    lo = max(0, cutoff_idx - window_days * GRID_STEPS_PER_DAY)
    out = np.zeros((n_primary, n_grid), dtype=np.float64)
    for i in range(n_primary):
        temp = np.asarray(raw[i, :, DRAW_COLUMNS["temperature"]], dtype=np.float64)
        window = temp[lo:cutoff_idx]
        window_month = month[lo:cutoff_idx]
        means = np.array(
            [
                np.nanmean(window[window_month == m]) if (window_month == m).any() else np.nan
                for m in range(1, 13)
            ]
        )
        out[i] = means[month - 1]
    return out
