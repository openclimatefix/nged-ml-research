# ruff: noqa: E501, E402, PLR0915, PERF401, PT018, D107
"""Rows for experiment 552: ICON-EU and ECMWF inputs at four issue times a day.

`pack` writes two private caches under `WORK_DIR`: the ICON-EU runs as one array, and the ECMWF
summary rows (experiment 397's per-Primary `summary` files) of 2026-01-20 to 2026-06-30.
`make_rows` then builds, for a list of issue times, every row whose inputs were all available at
the issue time: the latest ICON-EU run with `init + ICON_DELAY <= issue`, and the latest ECMWF run
with `origin <= issue` (ECMWF origins sit 9 h after the 00 UTC run, the project's convention).
"""

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00552-icon-eu-four-runs-a-day-lag-free-demand" / "work"
)
RUN_DIR: Final[Path] = (
    Path(__import__("os").environ["DATA_PATH_INTERNAL"]) / "studies" / "icon_eu_multirun" / "runs"
)
VARIABLES: Final[tuple[str, ...]] = (
    "downward_direct_short_wave_radiation_flux_surface",
    "downward_diffuse_short_wave_radiation_flux_surface",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
)
H: Final[int] = 3_600_000_000
DAY: Final[int] = 24 * H
ICON_DELAY: Final[int] = 4 * H
"""Assumed hours between an ICON-EU run's init time and the run being usable (stated in the page)."""
ECM_OFFSET: Final[int] = 9 * H
MAX_LEAD_H: Final[int] = 78
ECM_FIRST: Final[datetime] = datetime(2026, 1, 20, tzinfo=UTC)
HOLDOUT_US: Final[int] = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1e6)
ISSUE_HOURS: Final[tuple[int, ...]] = (0, 6, 12, 18)
ICON_FEATURES: Final[int] = 6


def _us(t: datetime) -> int:
    return int(t.timestamp() * 1_000_000)


def pack() -> None:
    """Write the ICON-EU array and the ECMWF rows."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RUN_DIR.glob("*.parquet"))
    inits, arrays = [], []
    for f in files:
        d = pl.read_parquet(f).sort("primary", "lead_time")
        n_lead = d["lead_time"].n_unique()
        assert n_lead == MAX_LEAD_H + 1 and d["primary"].n_unique() == 16, f
        arr = np.stack(
            [d[v].to_numpy().reshape(16, n_lead) for v in VARIABLES], -1
        )  # primary, lead, var
        inits.append(d["init_time"][0].replace(tzinfo=UTC))
        arrays.append(arr.transpose(1, 0, 2).astype(np.float32))  # lead, primary, var
    arr = np.stack(arrays)
    np.savez(WORK_DIR / "icon.npz", icon=arr, init_us=np.array([_us(t) for t in inits]))
    print("icon", arr.shape, "NaN share by variable", np.isnan(arr).mean((0, 1, 2)).round(4))
    keep_cols, _, _ = data.non_lag_layout()
    out = {}
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        sel = (z["origin"] >= _us(ECM_FIRST)) & (z["valid"] < HOLDOUT_US)
        order = np.lexsort((z["valid"][sel], z["origin"][sel]))
        p = row["primary"]
        out[f"x{p}"] = z["x"][sel][order][:, keep_cols]
        out[f"o{p}"] = z["origin"][sel][order]
        out[f"v{p}"] = z["valid"][sel][order]
    np.savez(WORK_DIR / "ecm.npz", **out)


@dataclass
class Rows:
    """Rows of a list of issue times; `x_ecm` is raw (unstandardised), `icon` is raw."""

    x_ecm: np.ndarray
    icon: np.ndarray
    primary: np.ndarray
    issue: np.ndarray
    valid: np.ndarray
    ecm_origin: np.ndarray
    icon_init: np.ndarray


class Sources:
    """The packed ICON-EU array and ECMWF rows."""

    def __init__(self) -> None:
        z = np.load(WORK_DIR / "icon.npz")
        self.icon, self.init = z["icon"], z["init_us"]
        e = np.load(WORK_DIR / "ecm.npz")
        self.ecm = {p: (e[f"x{p}"], e[f"o{p}"], e[f"v{p}"]) for p in range(16)}

    def make_rows(self, *, issues: np.ndarray) -> Rows:
        """Return every row available at each issue time (microseconds), up to the ICON-EU run's 78 h horizon (measured from the run's init time)."""
        parts: list[dict[str, np.ndarray]] = []
        for t in issues:
            k = np.searchsorted(self.init + ICON_DELAY, t, side="right") - 1
            if k < 0 or t - self.init[k] > 2 * DAY:
                continue
            origin = ((t - ECM_OFFSET) // DAY) * DAY + ECM_OFFSET
            for p in range(16):
                x, o, v = self.ecm[p]
                a, b = (
                    np.searchsorted(o, origin, side="left"),
                    np.searchsorted(o, origin, side="right"),
                )
                if a == b:
                    continue
                vv = v[a:b]
                m = (vv >= t) & (vv <= self.init[k] + MAX_LEAD_H * H) & (vv < HOLDOUT_US)
                if not m.any():
                    continue
                valid = vv[m]
                h = (valid - self.init[k]) / H
                lo = np.floor(h).astype(int)
                hi = np.minimum(lo + 1, MAX_LEAD_H)
                if (lo > MAX_LEAD_H).any():
                    continue
                w = (h - lo)[:, None]
                f = (1 - w) * self.icon[k, lo, p] + w * self.icon[k, hi, p]
                parts.append(
                    {
                        "x_ecm": x[a:b][m],
                        "icon": f,
                        "primary": np.full(int(m.sum()), p),
                        "issue": np.full(int(m.sum()), t),
                        "valid": valid,
                        "ecm_origin": np.full(int(m.sum()), origin),
                        "icon_init": np.full(int(m.sum()), self.init[k]),
                    }
                )
        cat = {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}
        rows = Rows(**cat)
        assert (rows.ecm_origin <= rows.issue).all(), "ECMWF run issued after the issue time"
        assert (rows.icon_init + ICON_DELAY <= rows.issue).all(), (
            "ICON-EU run issued after the issue time"
        )
        assert (rows.valid >= rows.issue).all()
        assert not np.isnan(rows.icon).any(), "NaN in ICON-EU inputs"
        return rows


def icon_features(raw: np.ndarray) -> np.ndarray:
    """Return shortwave (direct + diffuse), 2 m temperature, dew point, surface pressure, precipitation, 10 m wind speed."""
    return np.column_stack(
        [
            raw[:, 0] + raw[:, 1],
            raw[:, 2],
            raw[:, 3],
            raw[:, 4],
            raw[:, 5],
            np.hypot(raw[:, 6], raw[:, 7]),
        ]
    ).astype(np.float32)


def target_idx(valid: np.ndarray) -> np.ndarray:
    """Return the half-hour grid index of each valid time."""
    return ((valid - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)


def print_example(rows: Rows) -> None:
    """Print one row's timing, as the standing rule requires."""
    i = len(rows.valid) // 2
    f = lambda us: datetime.fromtimestamp(us / 1e6, UTC).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    print(
        f"example row: issue {f(rows.issue[i])}, ICON-EU run init {f(rows.icon_init[i])} "
        f"(usable {f(rows.icon_init[i] + ICON_DELAY)}), ECMWF origin {f(rows.ecm_origin[i])}, valid {f(rows.valid[i])}"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["pack"]:
        pack()
