# ruff: noqa: E501
"""Experiment 646: the 51 ensemble members' weather inputs for the demand networks' rows.

The networks of experiments 559 to 613 read the ensemble mean in columns 0 to 6 of the standardised
input and the ensemble standard deviation in columns 7 to 13. This module builds, for any set of
rows (a Primary, a forecast origin, a valid time), the seven weather columns of each of the 51
members, in the networks' own units (precipitation through `log1p`, as experiment 397's
`feature_matrix`) and standardised with the fold's mean-column statistics. Where the 51-member
frame of a run is not stored (runs before 2025-07-02), only the control member (index 0) is
available, read from experiment 397's `member` matrices; `n_avail` says how many members a row has.
"""

from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

N_MEMBERS = 51
N_WEATHER = len(prep.WEATHER)
KEY_SHIFT = 1 << 52
CLIP_SD = prep.INPUT_CLIP_SD
_PRECIP = prep.WEATHER.index("precipitation")
_FRAME_CACHE: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}


def _transform(values: np.ndarray) -> np.ndarray:
    """Return raw weather values as the network's input units (precipitation through log1p)."""
    out = values.astype(np.float64, copy=True)
    out[..., _PRECIP] = np.log1p(np.clip(out[..., _PRECIP], 0, None))
    return out


def run_frame(*, origin_us: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sorted keys, members array (keys, 51, 7)) of the run behind an origin, or None.

    The key is `time_series_id * 2**52 + valid_time in microseconds`. The run is the origin minus
    the 9-hour publication delay.
    """
    if origin_us in _FRAME_CACHE:
        return _FRAME_CACHE[origin_us]
    run = datetime.fromtimestamp(origin_us / 1e6, UTC) - common.PUBLICATION_DELAY
    path = common.demand_frame_path(run=run)
    if not path.exists():
        _FRAME_CACHE[origin_us] = None
        return None
    ids = prep.primaries()["time_series_id"].to_list()
    frame = (
        pl.read_parquet(
            path, columns=["time_series_id", "valid_time", "ensemble_member", *prep.WEATHER]
        )
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id", "valid_time", "ensemble_member")
    )
    assert frame.height % N_MEMBERS == 0, (path, frame.height)
    members = frame["ensemble_member"].to_numpy().reshape(-1, N_MEMBERS)
    assert (members == np.arange(N_MEMBERS)).all(), f"{path}: members are not 0..50 in every group"
    values = _transform(frame.select(prep.WEATHER).to_numpy()).reshape(-1, N_MEMBERS, N_WEATHER)
    sid = frame["time_series_id"].to_numpy().reshape(-1, N_MEMBERS)[:, 0].astype(np.int64)
    valid = (
        frame["valid_time"].dt.epoch("us").to_numpy().reshape(-1, N_MEMBERS)[:, 0].astype(np.int64)
    )
    keys = sid * KEY_SHIFT + valid
    assert (np.diff(keys) > 0).all(), f"{path}: duplicate or unsorted keys"
    if len(_FRAME_CACHE) > 4:
        _FRAME_CACHE.pop(next(iter(_FRAME_CACHE)))
    _FRAME_CACHE[origin_us] = (keys, values.astype(np.float32))
    return _FRAME_CACHE[origin_us]


def control_rows(*, site: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the control member's (sorted key, weather) rows of a Primary from experiment 397's matrix.

    The key is `origin * 2**20 + (valid - origin) / 1 800 000 000` (the half-hour lead index), which
    is unique per row. Columns 0 to 6 of the matrix are the unstandardised weather.
    """
    z = np.load(prep.RAW_DIR / "member" / f"{site}.npz")
    key = (z["origin"].astype(np.int64) // 1_800_000_000) * (1 << 20) + (
        z["valid"] - z["origin"]
    ) // 1_800_000_000
    order = np.argsort(key)
    return key[order], z["x"][order][:, :N_WEATHER].astype(np.float32), order


def member_values(
    *, primary: np.ndarray, origin_us: np.ndarray, valid_us: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return unstandardised member weather (rows, 51, 7) float32 and the members available per row.

    Rows whose run has a stored 51-member frame get all 51 members. Other rows get the control
    member in slot 0 and `n_avail` 1; their slots 1 to 50 repeat slot 0, so a draw from any slot
    returns the control member. A row found in neither source raises.
    """
    table = prep.primaries()
    out = np.empty((len(primary), N_MEMBERS, N_WEATHER), dtype=np.float32)
    n_avail = np.ones(len(primary), dtype=np.int16)
    filled = np.zeros(len(primary), dtype=bool)
    sid_of = dict(zip(table["primary"].to_list(), table["time_series_id"].to_list(), strict=True))
    for o in np.unique(origin_us):
        sel = np.nonzero(origin_us == o)[0]
        frame = run_frame(origin_us=int(o))
        if frame is None:
            continue
        keys, values = frame
        want = (
            np.array([sid_of[int(p)] for p in primary[sel]], dtype=np.int64) * KEY_SHIFT
            + valid_us[sel]
        )
        pos = np.searchsorted(keys, want)
        pos = np.minimum(pos, len(keys) - 1)
        hit = keys[pos] == want
        out[sel[hit]] = values[pos[hit]]
        n_avail[sel[hit]] = N_MEMBERS
        filled[sel[hit]] = True
    for p in np.unique(primary[~filled]):
        site = table.filter(pl.col("primary") == int(p))["site"][0]
        key_sorted, weather, _ = control_rows(site=site)
        sel = np.nonzero((primary == p) & ~filled)[0]
        want = (origin_us[sel].astype(np.int64) // 1_800_000_000) * (1 << 20) + (
            valid_us[sel] - origin_us[sel]
        ) // 1_800_000_000
        pos = np.minimum(np.searchsorted(key_sorted, want), len(key_sorted) - 1)
        if not (key_sorted[pos] == want).all():
            raise ValueError(
                f"{site}: {int((key_sorted[pos] != want).sum())} rows have no member weather"
            )
        out[sel] = weather[pos][:, None, :]
    return out, n_avail


def standardise(*, values: np.ndarray, standardiser: prep.Standardiser) -> np.ndarray:
    """Return member weather standardised with the fold's mean-column statistics (columns 0 to 6), float16."""
    mu = standardiser.mean[:N_WEATHER].astype(np.float32)
    sd = standardiser.std[:N_WEATHER].astype(np.float32)
    z = np.clip((values - mu) / sd, -CLIP_SD, CLIP_SD)
    return np.nan_to_num(z, nan=0.0).astype(np.float16)


class MemberX:
    """A row-indexable input whose weather columns are one random member per row on every access.

    `x[b]` returns the base rows with columns 0 to 6 replaced by the drawn member's standardised
    weather. With `zero_spread`, columns 7 to 13 read zero (as experiment 613's `drop_spread`).
    """

    def __init__(
        self, *, x: torch.Tensor, members: torch.Tensor, n_avail: torch.Tensor, zero_spread: bool
    ) -> None:
        """Hold the base rows (rows, columns), members (rows, 51, 7) float16, and n_avail (rows,)."""
        assert len(x) == len(members) == len(n_avail)
        self.x, self.members, self.n_avail, self.zero_spread = x, members, n_avail, zero_spread

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.x)

    @property
    def device(self) -> torch.device:
        """Return the device of the base rows."""
        return self.x.device

    def __getitem__(self, b: torch.Tensor) -> torch.Tensor:
        """Return the rows `b` with a fresh member drawn for each."""
        out = self.x[b].clone()
        draw = (torch.rand(len(out), device=out.device) * self.n_avail[b]).long()
        out[:, :N_WEATHER] = self.members[b, draw].float()
        if self.zero_spread:
            out[:, N_WEATHER : 2 * N_WEATHER] = 0.0
        return out
