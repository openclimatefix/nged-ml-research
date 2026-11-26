# ruff: noqa: E501, PT018
"""Experiment 716: the three synthetic worlds and the NumPy ridge references.

This module is WP5a (design section 8.2). Everything here is NumPy on the CPU and reads no neural
network. A world is experiment 713's world W0 (the real grid, the real ERA5-and-CAMS weather, and
the real calendar of four Primaries, with power replaced by a known function plus noise of standard
deviation 0.01) with four additions: a hidden daytime level `L_b d(t)`, a daily daytime anomaly
`e_day d(t)`, a PV-like dip `-s u(t)`, and (for the networks only) one part of the sky hidden from
the weather tokens. This module builds the power and the two reference estimators; the driver
`e00716_world_rung.py` runs them rung by rung.

**One ridge per Primary and target half-hour, pooled over lead days.** The design asks for a ridge
regression fitted separately for each of the four Primaries. A single linear model cannot give a
daytime level that scales with the target half-hour's `d`, so each Primary's ridge here is a family
of 48 regressions, one per target half-hour of the row (grid column), which share one penalty chosen
by cross-validation. Every regression pools the 14 lead days of an origin: the history features of an
origin are shared by its 14 lead days, which lets the Gram matrix of the history features be built
once per origin and the eigendecomposition be shared by all 48 regressions (`block_ridge`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

import e00713_layout as lay  # ty: ignore[unresolved-import]
import e00713_train as tr  # ty: ignore[unresolved-import]
import e00716_pv as pv
import numpy as np

PRIMARIES: Final = (2, 5, 9, 14)
"""Indices (in the 16-Primary order) of the four Primaries of experiment 713's worlds."""
S_BASE: Final = (0.30, 0.40, 0.50, 0.60)
"""The dip's size `s` at the four Primaries at rungs 1 to 3, in shares of capacity."""
LEVEL_SD: Final = 0.10
NOISE_SD: Final = 0.01
HELD_OUT_DAYS: Final = 236
"""Validation origins: the 236 days after the cutoff, the longest window the dense weather table allows (design addendum 1)."""
N_DRAWS: Final = 8
"""World draws 0 to 7 (`build_world(seed=k)`): a rung passes only if the mean `gap` over them and draw 0's own `gap` both reach `GAP_MIN`."""
EARLY_DAYS: Final = 60
"""The first 60 validation days, the window of the original design: KF9 and KF11 are also printed on days 1 to 60 and 61 to 236 (secondary)."""
GAP_MIN: Final = 0.010
VALIDATION_LEADS: Final = (0, 1, 3, 6, 9, 13)
"""Experiment 713's `VALIDATION_LEAD_DAYS` (zero-based lead-day indices)."""
PENALTIES: Final = (1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 10_000_000.0)
"""Candidate ridge penalties, added to the Gram matrix of standardised features pooled over leads."""
CV_BLOCKS: Final = 3
SOLAR_HALF_HOURS_UTC: Final = (12, 29)
"""The solar block: half-hours stamped 06:00 to 14:30 UTC (grid columns 12 to 29 of a UTC day), the definition of the stored 713 solar analysis."""
N_LEADS: Final = lay.N_LEAD_DAYS


@dataclass(frozen=True)
class Rung:
    """One rung of the ladder of design section 8.2 (cumulative settings).

    Attributes:
        number: 1 to 5.
        redraw_days: How often the hidden level `L_b` is redrawn.
        e_sd: Standard deviation of the daily anomaly `e_day`.
        s_scale: The factor on `S_BASE` (1 at rungs 1 to 3, 2 at rungs 4 and 5).
    """

    number: int
    redraw_days: int
    e_sd: float
    s_scale: float


RUNGS: Final = (
    Rung(1, 30, 0.03, 1.0),
    Rung(2, 90, 0.03, 1.0),
    Rung(3, 90, 0.015, 1.0),
    Rung(4, 90, 0.015, 2.0),
    Rung(5, 90, 0.0075, 2.0),
)


@dataclass
class Dense4:
    """The real dense weather and calendar of the four Primaries, each (4, G) unless stated.

    Attributes:
        ghi: Shortwave irradiance (W/m2, the CAMS column).
        temperature_c: 2 m temperature in degrees Celsius.
        dew_c: 2 m dew point.
        precip: Precipitation.
        cos_zenith: Cosine of the solar zenith.
        sin_azimuth: Sine of the azimuth.
        cos_azimuth: Cosine of the azimuth.
        day_type: (G,) day type 0 to 7.
        half_hour: (G,) local half-hour of the day, which indexes the table `w`.
        doy: (G,) day of the year.
    """

    ghi: np.ndarray
    temperature_c: np.ndarray
    dew_c: np.ndarray
    precip: np.ndarray
    cos_zenith: np.ndarray
    sin_azimuth: np.ndarray
    cos_azimuth: np.ndarray
    day_type: np.ndarray
    half_hour: np.ndarray
    doy: np.ndarray


def dense4_from_raw(
    *, raw: np.ndarray, day_type: np.ndarray, half_hour: np.ndarray, doy: np.ndarray
) -> Dense4:
    """Select the four Primaries from the raw dense table (16, G, 28) and name the columns.

    Columns 0 to 6 are shortwave, longwave, 2 m temperature, dew point, pressure, precipitation, and
    10 m wind speed; columns 7 to 9 are the cosine of the zenith and the sine and cosine of the azimuth.
    A temperature table in kelvin is converted, as experiment 713's known-answer script does.
    """
    sel = np.asarray(PRIMARIES)

    def col(c: int) -> np.ndarray:
        return np.nan_to_num(np.asarray(raw[sel, :, c], dtype=np.float64), nan=0.0)

    temperature, dew = col(2), col(3)
    if float(np.median(temperature)) > 100.0:
        temperature, dew = temperature - 273.15, dew - 273.15
    return Dense4(
        ghi=col(0), temperature_c=temperature, dew_c=dew, precip=col(5), cos_zenith=col(7),
        sin_azimuth=col(8), cos_azimuth=col(9), day_type=day_type, half_hour=half_hour, doy=doy,
    )  # fmt: skip


def check_validation_weather_finite(
    *, raw: np.ndarray, b1_grid: np.ndarray, cams: np.ndarray, day_code: np.ndarray, cutoff_idx: int
) -> str:
    """Assert that every input the networks read is finite over the whole validation window.

    This is design addendum 1, F11. `dense4_from_raw` and `b1_scale` turn a missing value into 0,
    which would read as night. The window runs from the oldest history token of the first validation
    origin to the last target token of the last one, and covers all 28 dense columns of the four
    Primaries, the 12 B1 grid columns, and the 2 CAMS day columns.
    """
    first_origin = (cutoff_idx // 48) * 48 + 24
    last_origin = first_origin + 48 * (HELD_OUT_DAYS - 1)
    lo = first_origin - lay.STEPS_PER_DAY * lay.MAX_AGE
    hi = last_origin + lay.STEPS_PER_DAY * lay.N_LEAD_DAYS
    assert hi <= raw.shape[1], (
        f"the validation window ends at {hi}, past the dense table's {raw.shape[1]}"
    )
    sel = np.asarray(PRIMARIES)
    for name, block in (
        ("dense weather (28 columns)", np.asarray(raw[sel, lo:hi, :])),
        ("B1 grid columns", np.asarray(b1_grid[sel, lo:hi])),
        ("CAMS day columns", np.asarray(cams)[sel][:, day_code[lo:hi]]),
    ):
        assert bool(np.isfinite(block).all()), f"non-finite {name} inside the validation window"
    return f"every dense, B1, and CAMS input is finite from index {lo} to {hi} ({HELD_OUT_DAYS} validation days)"


def draw_table(*, n_day_types: int = 8) -> np.ndarray:
    """Return experiment 713's fixed 8 x 48 table `w`: uniform on -0.10 to +0.10, seed 0."""
    return np.random.default_rng(0).uniform(-0.10, 0.10, size=(n_day_types, 48))


@dataclass
class PvWorld:
    """One world's power and the quantities the references and the networks read.

    Each quantity is (4, G) unless stated.

    Attributes:
        rung: The rung's settings.
        y: The power over capacity (known function, hidden level, anomaly, dip, and noise).
        known: The known W0 function (`0.5 + 0.02 max(0, 15.5 - T) + w`), which the references
            subtract.
        d: The shape `d(t)` of the hidden level.
        u: The yield curve `u(t)` of the real CAMS irradiance.
        s: (4,) the dip's size per Primary.
        level: The hidden daytime level `L_b` at every grid index (before `d`).
        anomaly: The daily anomaly `e_day` at every grid index (before `d`).
        extra: Further arrays that a rung adds, keyed by name.
    """

    rung: Rung
    y: np.ndarray
    known: np.ndarray
    d: np.ndarray
    u: np.ndarray
    s: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def resid(self) -> np.ndarray:
        """Return the power minus the known W0 function: `(L_b + e_day) d - s u + noise`."""
        return self.y - self.known


def yield_and_shape(*, dense: Dense4) -> tuple[np.ndarray, np.ndarray]:
    """Return the yield curve `u` and the shape `d` of the hidden level.

    The curve `u` is the yield curve of the real CAMS irradiance, and `d` is the clear-sky yield curve
    over its daily maximum.
    """
    n_primary, n_grid = dense.ghi.shape
    u = np.zeros((n_primary, n_grid))
    d = np.zeros((n_primary, n_grid))
    zero = np.zeros(n_grid)
    clear = pv.clear_sky(cos_zenith=dense.cos_zenith)
    n_days = n_grid // lay.STEPS_PER_DAY
    for i in range(n_primary):
        args = {
            "ghi_sd": zero, "cos_zenith": dense.cos_zenith[i], "sin_azimuth": dense.sin_azimuth[i],
            "cos_azimuth": dense.cos_azimuth[i], "doy": dense.doy, "temperature_c": dense.temperature_c[i],
        }  # fmt: skip
        u[i], _ = pv.yield_curve(ghi_mean=dense.ghi[i], **args)  # ty: ignore[invalid-argument-type]
        clear_yield, _ = pv.yield_curve(ghi_mean=clear[i], **args)  # ty: ignore[invalid-argument-type]
        block = clear_yield[: n_days * lay.STEPS_PER_DAY].reshape(n_days, lay.STEPS_PER_DAY)
        peak = block.max(axis=1, keepdims=True)
        shaped = np.where(peak > 1e-9, block / np.where(peak > 1e-9, peak, 1.0), 0.0)
        d[i, : n_days * lay.STEPS_PER_DAY] = shaped.reshape(-1)
    return np.clip(u, 0.0, None), d


def build_world(
    *, dense: Dense4, rung: Rung, seed: int = 0, s_override: float | None = None
) -> PvWorld:
    """Return the world of `rung`: `y = known + (L_b + e_day) d - s u + noise`.

    `L_b` is normal with standard deviation 0.10 and redrawn every `rung.redraw_days` days (with a
    random offset per Primary); `e_day` is normal with standard deviation `rung.e_sd`, one draw per UTC
    day; the noise has standard deviation 0.01. `s_override` replaces the dip size `s` of every Primary
    (world WN passes 0: the power has no dip, so the estimate `0.30 u` is of PV that does not exist).
    """
    n_primary, n_grid = dense.ghi.shape
    rng = np.random.default_rng([seed, rung.number, 7])
    w = draw_table()
    known = np.zeros((n_primary, n_grid))
    for i in range(n_primary):
        known[i] = (
            0.5
            + 0.02 * np.maximum(0.0, 15.5 - dense.temperature_c[i])
            + w[dense.day_type, dense.half_hour]
        )
    u, d = yield_and_shape(dense=dense)
    s = (
        np.asarray(S_BASE) * rung.s_scale
        if s_override is None
        else np.full(n_primary, float(s_override))
    )
    day = np.arange(n_grid) // lay.STEPS_PER_DAY
    n_days = int(day.max()) + 1
    level = np.zeros((n_primary, n_grid))
    anomaly = np.zeros((n_primary, n_grid))
    for i in range(n_primary):
        offset = int(rng.integers(rung.redraw_days))
        blocks = rng.normal(0.0, LEVEL_SD, size=(n_days + offset) // rung.redraw_days + 2)
        level[i] = blocks[(day + offset) // rung.redraw_days]
        anomaly[i] = rng.normal(0.0, rung.e_sd, size=n_days)[day]
    noise = rng.normal(0.0, NOISE_SD, size=(n_primary, n_grid))
    y = known + (level + anomaly) * d - s[:, None] * u + noise
    return PvWorld(rung=rung, y=y, known=known, d=d, u=u, s=s, level=level, anomaly=anomaly)


# ---------------------------------------------------------------- origins


def origin_sets(
    *, world: PvWorld, cutoff_idx: int, primaries: tuple[int, ...] = PRIMARIES
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the training and validation origins as `(primary position, contract index)` arrays.

    Training origins are experiment 713's stage 1 sample set (`make_samples` with its held-out and
    purged weeks) of the world's finite power, excluding the held-out weeks; validation origins are the
    `HELD_OUT_DAYS` days after the cutoff.
    """
    from types import SimpleNamespace

    n_grid = world.y.shape[1]
    clean = np.full((16, n_grid), np.nan)
    for pos, prim in enumerate(primaries):
        clean[prim] = world.y[pos]
    fake = SimpleNamespace(source=None, cutoff_idx=cutoff_idx, clean=clean)
    samples = tr.make_samples(world=fake, stage=1, placebo=False, primaries=primaries)  # ty: ignore[invalid-argument-type]
    keep = ~samples.is_val
    held_mask = samples.is_val
    pos_of = {p: i for i, p in enumerate(primaries)}
    train_pos = np.array([pos_of[int(p)] for p in samples.primary[keep]], dtype=np.int64)
    train_c = samples.c_idx[keep].astype(np.int64)
    days = np.arange(cutoff_idx // 48, cutoff_idx // 48 + HELD_OUT_DAYS)
    val_c = np.tile(days * 48 + 24, len(primaries)).astype(np.int64)
    val_pos = np.repeat(np.arange(len(primaries)), len(days)).astype(np.int64)
    assert int(val_c.max()) + 48 * N_LEADS <= n_grid, "validation targets leave the grid"
    held_pos = np.array([pos_of[int(p)] for p in samples.primary[held_mask]], dtype=np.int64)
    return {
        "train": (train_pos, train_c),
        "val": (val_pos, val_c),
        "held": (held_pos, samples.c_idx[held_mask].astype(np.int64)),
    }


# ---------------------------------------------------------------- features


@dataclass
class Features:
    """The features and targets of one Primary.

    The history features are `h` (n, p), the target-token features are `z` (n, 14, 48, k), and the
    targets are `y` (n, 14, 48).

    Attributes:
        h: (n, p) float64, unstandardised.
        z: (n, 14, 48, k) float64, column 0 the constant.
        y: (n, 14, 48) the residual target.
        c: (n,) contract indices.
        solar: (48,) bool, the solar-block columns.
    """

    h: np.ndarray
    z: np.ndarray
    y: np.ndarray
    c: np.ndarray
    solar: np.ndarray


def solar_columns() -> np.ndarray:
    """Return which of the 48 target columns fall in the UTC solar block.

    The columns start at the 12:00 UTC contract origin.
    """
    utc_half_hour = (24 + np.arange(48)) % 48
    return (utc_half_hour >= SOLAR_HALF_HOURS_UTC[0]) & (utc_half_hour <= SOLAR_HALF_HOURS_UTC[1])


def build_features(
    *, world: PvWorld, dense: Dense4, pos: int, c: np.ndarray, kind: str, reference: str
) -> Features:
    """Return the features of one Primary's origins `c`.

    The features are for world `kind` (`WH` or `WT`) and `reference` (`naive` or `oracle`). The
    history features are shared by the 14 lead days of an origin. The lists follow design
    section 8.2; two readings are made explicit: "d of each history token" and "the product of
    the range with d" are taken at the history token (14 x 48 each), and the target row's `d`
    and visible weather are taken at each target token.
    """
    assert kind in {"WH", "WT"} and reference in {"naive", "oracle"}
    steps = np.arange(48)
    ages = lay.history_ages()
    hi = c[:, None, None] - 48 * ages[None, :, None] + steps[None, None, :]
    visible = hi <= (c - lay.NEWEST_VISIBLE_BEFORE_CONTRACT)[:, None, None]
    resid = world.resid[pos]
    hist_resid = np.where(visible, resid[hi], 0.0).reshape(len(c), -1)
    d_hist = world.d[pos][hi]
    su_hist = (world.s[pos] * world.u[pos])[hi]
    temp_h = dense.temperature_c[pos][hi]
    t_range = temp_h.max(axis=2) - temp_h.min(axis=2)
    range_d = (t_range[:, :, None] * d_hist).reshape(len(c), -1)
    dew_mean = dense.dew_c[pos][hi].mean(axis=2)
    precip_mean = dense.precip[pos][hi].mean(axis=2)
    if kind == "WH":
        parts = [hist_resid, dew_mean, precip_mean, d_hist.reshape(len(c), -1), t_range, range_d]
        if reference == "oracle":
            parts.append(su_hist.reshape(len(c), -1))
    else:
        parts = [hist_resid, su_hist.reshape(len(c), -1), t_range, range_d]
    h = np.concatenate(parts, axis=1)
    ti = c[:, None, None] + 48 * np.arange(N_LEADS)[None, :, None] + steps[None, None, :]
    d_t, u_t, su_t = world.d[pos][ti], world.u[pos][ti], (world.s[pos] * world.u[pos])[ti]
    cols = [np.ones_like(d_t)]
    if kind == "WH":
        cols += [u_t, d_t]
    else:
        dew_t, prec_t = dense.dew_c[pos][ti], dense.precip[pos][ti]
        cols += [
            dew_t, prec_t, dense.temperature_c[pos][ti], dense.cos_zenith[pos][ti], dense.sin_azimuth[pos][ti],
            dense.cos_azimuth[pos][ti], d_t, d_t * dew_t.mean(axis=2, keepdims=True), d_t * prec_t.mean(axis=2, keepdims=True),
        ]  # fmt: skip
        if reference == "oracle":
            cols.append(su_t)
    z = np.stack(cols, axis=-1)
    return Features(h=h, z=z, y=resid[ti], c=c, solar=solar_columns())


# ---------------------------------------------------------------- the block ridge


@dataclass
class Piece:
    """The sums of one set of origins that a block ridge needs (additive over origins).

    Attributes:
        a: (p, p) `L * H'H` (every origin counts for 14 lead days).
        b: (48, p, k) `H' sum_l Z`.
        c: (48, k, k) `sum Z'Z`.
        r: (48, p) `H' sum_l Y`.
        q: (48, k) `sum Z'Y`.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    r: np.ndarray
    q: np.ndarray

    def __sub__(self, other: Piece) -> Piece:
        """Return the sums with `other`'s origins removed."""
        return Piece(
            self.a - other.a, self.b - other.b, self.c - other.c, self.r - other.r, self.q - other.q
        )

    def __add__(self, other: Piece) -> Piece:
        """Return the sums of both sets of origins."""
        return Piece(
            self.a + other.a, self.b + other.b, self.c + other.c, self.r + other.r, self.q + other.q
        )


def piece_of(*, hs: np.ndarray, z: np.ndarray, y: np.ndarray) -> Piece:
    """Return the block-ridge sums of origins.

    The origins have standardised history features `hs` (n, p), `z` (n, 14, 48, k), and `y` (n, 14,
    48).
    """
    zsum = z.sum(axis=1)  # (n, 48, k)
    ysum = y.sum(axis=1)  # (n, 48)
    b = np.stack([hs.T @ zsum[:, h] for h in range(48)])  # (48, p, k)
    r = np.stack([hs.T @ ysum[:, h] for h in range(48)])  # (48, p)
    flat = [z[:, :, h].reshape(-1, z.shape[-1]) for h in range(48)]
    c = np.stack([f.T @ f for f in flat])
    q = np.stack([f.T @ y[:, :, h].reshape(-1) for h, f in enumerate(flat)])
    return Piece(a=N_LEADS * (hs.T @ hs), b=b, c=c, r=r, q=q)


@dataclass
class Fit:
    """The eigendecomposition of a piece's Gram matrix.

    Any penalty's solution follows cheaply from the eigendecomposition.
    """

    piece: Piece
    evals: np.ndarray
    evecs: np.ndarray
    vt_b: np.ndarray  # (48, p, k) V'B
    vt_r: np.ndarray  # (48, p) V'r


def prepare(*, piece: Piece) -> Fit:
    """Eigendecompose the Gram matrix `a` once and project `b` and `r` onto its eigenvectors."""
    evals, evecs = np.linalg.eigh(piece.a)
    evals = np.clip(evals, 0.0, None)
    vt_b = np.einsum("pq,hpk->hqk", evecs, piece.b, optimize=True)
    vt_r = np.einsum("pq,hp->hq", evecs, piece.r, optimize=True)
    return Fit(piece=piece, evals=evals, evecs=evecs, vt_b=vt_b, vt_r=vt_r)


def solve(*, fit: Fit, penalty: float, z_ridge: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Return the weights `w` (48, p) and `v` (48, k) at `penalty`.

    The weights `w` are those of the history features and `v` those of the target features. The
    system per half-hour is `[[A + lam I, B], [B', C + eps I]] [w; v] = [r; q]`, solved by
    eliminating `w` with the eigendecomposition of `A`.
    """
    inv = 1.0 / (fit.evals + penalty)  # (p,)
    pc = fit.piece
    k = pc.c.shape[-1]
    # (A + lam)^-1 B = V diag(inv) V'B ; the Schur complement S = C + eps I - B' (A + lam)^-1 B
    ab = fit.vt_b * inv[None, :, None]  # (48, p, k) in the eigenbasis
    schur = (
        pc.c + z_ridge * np.eye(k)[None] - np.einsum("hpk,hpj->hkj", fit.vt_b, ab, optimize=True)
    )
    rhs_v = pc.q - np.einsum("hpk,hp->hk", ab, fit.vt_r, optimize=True)
    v = np.linalg.solve(schur, rhs_v[..., None])[..., 0]
    # w = (A + lam)^-1 (r - B v) in the eigenbasis, then back to the feature basis
    inner = fit.vt_r - np.einsum("hpk,hk->hp", fit.vt_b, v, optimize=True)
    w = np.einsum("pq,hq->hp", fit.evecs, inner * inv[None, :], optimize=True)
    return w, v


def predict(*, hs: np.ndarray, z: np.ndarray, w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the predictions (n, 14, 48): `hs @ w[h] + z[., l, h] @ v[h]`."""
    return (hs @ w.T)[:, None, :] + np.einsum("nlhk,hk->nlh", z, v)


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardise the columns of `train` and apply the same statistics to `others`.

    A constant column becomes zero.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    scale = np.where(std > 1e-12, 1.0 / np.where(std > 1e-12, std, 1.0), 0.0)
    return tuple((x - mean) * scale for x in (train, *others))


@dataclass
class ReferenceResult:
    """One reference's scores on the validation origins of one Primary.

    Attributes:
        penalty: The cross-validated penalty.
        cv_mae: The cross-validation MAE of each candidate penalty.
        err_solar: Absolute errors on the solar-block validation tokens (flattened).
        err_all: Absolute errors on all validation tokens.
        resid_sd: The validation residual standard deviation on the solar block.
        err_full: Absolute errors on every validation lead of every token, or None if not kept.
    """

    penalty: float
    cv_mae: dict[float, float]
    err_solar: np.ndarray
    err_all: np.ndarray
    resid_sd: float
    err_full: np.ndarray | None = None


def run_reference(
    *,
    train: Features,
    val: Features,
    penalties: tuple[float, ...] = PENALTIES,
    blocks: int = CV_BLOCKS,
) -> ReferenceResult:
    """Fit one reference on one Primary's training origins and score it on the validation origins."""
    hs_train, hs_val = standardise(train.h, val.h)
    n = len(hs_train)
    z_mean = train.z.reshape(-1, train.z.shape[-1]).mean(axis=0)
    z_std = train.z.reshape(-1, train.z.shape[-1]).std(axis=0)
    z_scale = np.where(z_std > 1e-12, 1.0 / np.where(z_std > 1e-12, z_std, 1.0), 0.0)
    z_scale[0], z_mean[0] = 1.0, 0.0  # the constant stays a constant
    z_train, z_val = (train.z - z_mean) * z_scale, (val.z - z_mean) * z_scale
    edges = np.linspace(0, n, blocks + 1).astype(int)
    pieces = [
        piece_of(hs=hs_train[a:b], z=z_train[a:b], y=train.y[a:b]) for a, b in pairwise(edges)
    ]
    total = pieces[0]
    for pc in pieces[1:]:
        total = total + pc
    lead_val = list(VALIDATION_LEADS)
    cv = dict.fromkeys(penalties, 0.0)
    for j, (a, b) in enumerate(pairwise(edges)):
        fit = prepare(piece=total - pieces[j])
        for lam in penalties:
            w, v = solve(fit=fit, penalty=lam)
            pred = predict(hs=hs_train[a:b], z=z_train[a:b], w=w, v=v)
            cv[lam] += float(np.abs(pred - train.y[a:b])[:, lead_val].sum()) / (
                n * len(lead_val) * 48
            )
    best = min(cv, key=lambda lam: cv[lam])
    fit = prepare(piece=total)
    w, v = solve(fit=fit, penalty=best)
    err = np.abs(predict(hs=hs_val, z=z_val, w=w, v=v) - val.y)
    return ReferenceResult(
        penalty=best, cv_mae=cv, err_solar=err[:, :, val.solar].reshape(-1), err_all=err.reshape(-1),
        resid_sd=float(err[:, :, val.solar].std()), err_full=err,
    )  # fmt: skip


@dataclass
class GapResult:
    """The `gap` of one world at one rung."""

    kind: str
    gap: float
    gap_per_primary: list[float]
    mae_naive: float
    mae_oracle: float
    penalties: dict = field(default_factory=dict)


def gap_of_world(
    *, world: PvWorld, dense: Dense4, kind: str, cutoff_idx: int, log: bool = True
) -> GapResult:
    """Return the `gap` of world `kind` at `world.rung`.

    The `gap` is the naive reference's MAE of the median minus the oracle's, on solar-block
    validation tokens pooled over the four Primaries.
    """
    sets = origin_sets(world=world, cutoff_idx=cutoff_idx)
    naive_err, oracle_err, per_primary, penalties = [], [], [], {}
    for pos in range(len(PRIMARIES)):
        res = {}
        for reference in ("naive", "oracle"):
            tpos, tc = sets["train"]
            vpos, vc = sets["val"]
            feats_train = build_features(
                world=world, dense=dense, pos=pos, c=tc[tpos == pos], kind=kind, reference=reference
            )
            feats_val = build_features(
                world=world, dense=dense, pos=pos, c=vc[vpos == pos], kind=kind, reference=reference
            )
            res[reference] = run_reference(train=feats_train, val=feats_val)
            del feats_train, feats_val
        naive_err.append(res["naive"].err_solar)
        oracle_err.append(res["oracle"].err_solar)
        per_primary.append(float(res["naive"].err_solar.mean() - res["oracle"].err_solar.mean()))
        penalties[PRIMARIES[pos]] = {r: res[r].penalty for r in res}
        if log:
            print(
                f"  {kind} rung {world.rung.number} Primary {PRIMARIES[pos]}: gap {per_primary[-1]:.4f}, penalties {penalties[PRIMARIES[pos]]}",
                flush=True,
            )
    mae_n, mae_o = float(np.concatenate(naive_err).mean()), float(np.concatenate(oracle_err).mean())
    return GapResult(
        kind=kind,
        gap=mae_n - mae_o,
        gap_per_primary=per_primary,
        mae_naive=mae_n,
        mae_oracle=mae_o,
        penalties=penalties,
    )


def heldout_nll(
    *, world: PvWorld, dense: Dense4, kind: str, cutoff_idx: int, target_std: np.ndarray
) -> dict[str, float]:
    """Return the Gaussian negative log-likelihood of the naive and the oracle reference.

    The tokens are those that experiment 713's `validation_loss` averages. The tokens are the
    held-out origins' target tokens at lead days 1, 2, 4, 7, 10, and 14, all 48 half-hours,
    unweighted, in the world's standardised units (errors divided by each Primary's
    `target_std`), each reference as a Gaussian with its own error standard deviation. The
    references are the ridge references of `gap_of_world`, scored on the held-out origins
    instead.
    """
    sets = origin_sets(world=world, cutoff_idx=cutoff_idx)
    errors: dict[str, list[np.ndarray]] = {"naive": [], "oracle": []}
    for pos in range(len(PRIMARIES)):
        tpos, tc = sets["train"]
        hpos, hc = sets["held"]
        for reference in ("naive", "oracle"):
            train = build_features(
                world=world, dense=dense, pos=pos, c=tc[tpos == pos], kind=kind, reference=reference
            )
            held = build_features(
                world=world, dense=dense, pos=pos, c=hc[hpos == pos], kind=kind, reference=reference
            )
            res = run_reference(train=train, val=held)
            assert res.err_full is not None
            errors[reference].append(
                res.err_full[:, list(VALIDATION_LEADS)].reshape(-1) / target_std[pos]
            )
    out = {}
    for reference, parts in errors.items():
        z = np.concatenate(parts)
        out[reference] = 0.5 * float(np.log(2 * np.pi * np.mean(z**2))) + 0.5
    return out
