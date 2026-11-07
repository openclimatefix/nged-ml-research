# ruff: noqa: E501
"""Experiment 616: fit the donor regression to every stored probe response and write the result tables.

For each fold, arm (D5 or M0), Primary, and probe date, the probed response `y` (MW, one value per
donor and slot) is regressed on experiment 269's unit PV curve `u` and the donor's temperature, with
one intercept per slot. The capacity is minus the slope on `u`. Variants:

- `T`: `u` and temperature (the headline).
- `noT`: `u` only.
- `cloud`: `u`, temperature, the half-hour cloud index `cs`, and the donor day's mean cloud index `cd`.
- `tm`: variant `T` on donors within 2 degrees Celsius of the slot's median temperature.
- `P_profile`: variant `T` with `u` replaced by the unit curve at the climatological sky (the mean
  irradiance of the same slot over donors in the same 7-day bin of offset from the probe date); it
  carries the seasonal trend of the sky and no day-specific cloud. It is not a null, because the
  real response also follows the seasonal trend; `J_real` and `J_profile` fit `u` and the
  climatological `u` together and read the slope on each.
- `P_shuffle`: variant `T` with `u` replaced by the unit curve at a sky shuffled across donor days in
  the same slot; a placebo that has the right distribution and no link to the response.

Intervals: the seed spread (10 stored networks), and a day-block bootstrap that resamples whole
donor days within each probe date (200 resamples, shared across seeds).

Usage: python e00616_analyse.py FOLD [FOLD ...]
"""

import sys
from datetime import date
from typing import Final

import e00616_probe as probe
import e00616_regress as reg
import numpy as np
import polars as pl
from studies.pv_plant_fit import ResponseTable

OUT = reg.OUT
BOOTSTRAPS: Final[int] = 200
ARMS: Final[tuple[str, ...]] = ("D5", "M0")
VARIANTS: Final[tuple[str, ...]] = (
    "T",
    "noT",
    "cloud",
    "tm",
    "P_profile",
    "P_shuffle",
    "J_real",
    "J_profile",
)
DATE_SETS: Final[dict[str, tuple[date, date]]] = {
    "SY2024_dates": (date(2025, 4, 1), date(2025, 6, 30)),
    "SY2025_dates": (date(2025, 7, 1), date(2026, 6, 30)),
}
"""The probe dates that fall in each of experiment 474's season-years (July to June)."""
SITE_ORDER: Final[tuple[str, ...]] = probe.MW_PRIMARIES


def one_hot(*, group: np.ndarray) -> np.ndarray:
    """Return the dense one-hot matrix of integer group ids."""
    out = np.zeros((len(group), int(group.max()) + 1))
    out[np.arange(len(group)), group] = 1.0
    return out


def weighted_fit(*, y: np.ndarray, x: np.ndarray, onehot: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the slope vectors (regressors, seeds) of the weighted slot-demeaned regression of `y` (rows, seeds)."""
    denom = onehot.T @ w
    ok = denom > 0
    mean_x = (onehot.T @ (w[:, None] * x)) / np.where(ok, denom, 1.0)[:, None]
    mean_y = (onehot.T @ (w[:, None] * y)) / np.where(ok, denom, 1.0)[:, None]
    xd = x - onehot @ mean_x
    yd = y - onehot @ mean_y
    a = xd.T @ (w[:, None] * xd)
    b = xd.T @ (w[:, None] * yd)
    return np.linalg.solve(a + 1e-12 * np.eye(len(a)), b)


def group_table(*, rows: pl.DataFrame, table: ResponseTable, seed: int = 0) -> pl.DataFrame:
    """Add the real, profile-placebo, and shuffled-placebo unit curves and the temperature-matched flag."""
    rng = np.random.default_rng(seed)
    r = reg.add_regressors(rows=rows, table=table)
    day_of_year = pl.col("donor_us").cast(pl.Datetime("us")).dt.ordinal_day()
    probe_doy = pl.col("date").dt.ordinal_day()
    r = r.with_columns(offset=(day_of_year - probe_doy).cast(pl.Int32), row=pl.int_range(pl.len()))
    r = r.with_columns(
        g_clim=pl.col("g_in")
        .mean()
        .over(["primary", "date", "slot", (pl.col("offset") / 7.0).round().cast(pl.Int32)]),
        t_med=pl.col("temp").median().over(["primary", "date", "slot"]),
    )
    shuffled = r["g_in"].to_numpy().copy()
    keys = r.select("primary", "date", "slot").with_row_index("i")
    for (idx,) in (
        keys.group_by(["primary", "date", "slot"], maintain_order=True)
        .agg(pl.col("i"))
        .select("i")
        .iter_rows()
    ):
        shuffled[idx] = shuffled[rng.permutation(idx)]
    r = r.with_columns(g_shuf=pl.Series(shuffled))
    for name, col in (("u_prof", "g_clim"), ("u_shuf", "g_shuf")):
        u = reg.unit_curve(
            table=table,
            g=r[col].to_numpy(),
            elevation=r["elev_probe"].to_numpy(),
            temp=r["temp"].to_numpy(),
        )
        r = r.with_columns(pl.Series(name, u))
    return r.with_columns(tm=((pl.col("temp") - pl.col("t_med")).abs() <= 2.0))


def design(*, g: pl.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the regressor matrix of a variant and the row mask it uses."""
    t = g["temp"].to_numpy()
    cols = {
        "T": [g["u"].to_numpy(), t],
        "noT": [g["u"].to_numpy()],
        "cloud": [g["u"].to_numpy(), t, g["cs"].to_numpy(), g["cd"].to_numpy()],
        "tm": [g["u"].to_numpy(), t],
        "P_profile": [g["u_prof"].to_numpy(), t],
        "P_shuffle": [g["u_shuf"].to_numpy(), t],
        "J_real": [g["u"].to_numpy(), g["u_prof"].to_numpy(), t],
        "J_profile": [g["u"].to_numpy(), g["u_prof"].to_numpy(), t],
    }[variant]
    mask = g["tm"].to_numpy() if variant == "tm" else np.ones(len(g), dtype=bool)
    return np.stack(cols, axis=1), mask


def fold_results(*, month: str) -> pl.DataFrame:
    """Return the capacity draws of a fold: every arm, Primary, date, variant, bootstrap resample, and seed.

    `boot` 0 is the fit on the real donors; `boot` 1 to 200 are day-block bootstrap resamples.
    """
    rows = pl.read_parquet(OUT / f"probe_{month}.parquet")
    y_all = np.load(OUT / f"probe_{month}_y.npy")
    table = reg.unit_table()
    rows = group_table(rows=rows.with_row_index("global_row"), table=table)
    rng = np.random.default_rng(616)
    meta: list[tuple[int, date, str, str]] = []
    blocks: list[np.ndarray] = []
    for (primary, d), g in rows.group_by(["primary", "date"], maintain_order=True):
        idx = g["global_row"].to_numpy()
        slot = np.unique(g["slot"].to_numpy(), return_inverse=True)[1]
        onehot = one_hot(group=slot)
        days, day_id = np.unique(g["day"].to_numpy(), return_inverse=True)
        boot = rng.multinomial(len(days), np.full(len(days), 1.0 / len(days)), size=BOOTSTRAPS)
        for a, arm in enumerate(ARMS):
            y = y_all[a][:, idx].T.astype(np.float64)  # rows, seeds
            for variant in VARIANTS:
                x, mask = design(g=g, variant=variant)
                oh = onehot[mask]
                weights = [np.ones(int(mask.sum()))] + [
                    b[day_id][mask].astype(np.float64) for b in boot
                ]
                col = 1 if variant == "J_profile" else 0
                block = np.stack(
                    [-weighted_fit(y=y[mask], x=x[mask], onehot=oh, w=w)[col] for w in weights]
                )
                blocks.append(block)  # (boot, seed)
                meta.append((primary, d, arm, variant))
    n_boot, n_seed = blocks[0].shape
    frame = pl.DataFrame(
        {
            "primary": np.repeat([m[0] for m in meta], n_boot * n_seed),
            "date": np.repeat(
                np.array([m[1] for m in meta], dtype="datetime64[D]"), n_boot * n_seed
            ),
            "arm": np.repeat([m[2] for m in meta], n_boot * n_seed),
            "variant": np.repeat([m[3] for m in meta], n_boot * n_seed),
            "boot": np.tile(np.repeat(np.arange(n_boot), n_seed), len(meta)),
            "seed": np.tile(np.arange(n_seed), len(meta) * n_boot),
            "capacity_mw": np.concatenate([b.ravel() for b in blocks]).astype(np.float32),
        }
    )
    return frame.with_columns(fold=pl.lit(month))


def main() -> None:
    """Write the capacity-draw table of each fold."""
    for month in sys.argv[1:]:
        draws = fold_results(month=month)
        draws.write_parquet(OUT / f"draws_{month}.parquet")
        print(month, draws.shape, flush=True)


if __name__ == "__main__":
    main()
