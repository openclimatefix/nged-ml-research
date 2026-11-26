# ruff: noqa: E501
"""Experiment 732: the truth arrays of an evaluation set (numpy only).

In the order that `predict` writes predictions.

The scoring stage needs the targets, the delta of each pair, the sky, and the BSP step of every window,
and needs no torch to rebuild them: the windows come from `make_eval_batch`, the same function that
`e00732_train.predict` feeds to the network.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import e00732_paths  # noqa: F401

# isort: split
import e00732_arms as arms
import e00732_overlays as ov
import numpy as np

CHUNK = 256
ROW_KEY_NAMES = ("row_seg", "row_date", "row_hh")
"""The names under which a stored prediction file holds the segment, the date, and the half-hour of each row."""


@dataclass
class Truth:
    """The arrays of an evaluation set, each `(P, 48)` unless noted.

    Attributes:
        t: The plain targets `(P, 8, 48)`.
        ti: The injected targets `(P, 8, 48)` (None without an amount).
        pair: 1 where the window has a pair `(P,)`.
        delta: The scored delta.
        cosz: The cosine of the solar zenith angle.
        csi: The clear-sky index.
        dbsp: The BSP step channel (scaled; zero where the meter has no data).
        keep_raisable: The raisable half-hours of the plain window.
        picks: The (segment, day) of each window.
        unreliable: The unreliable S1 solar half-hours `(P, 48)`.
        keys: The (segment, date as days since 1970-01-01, half-hour) of each half-hour `(3, P, 48)`.
        r_points: The reading of the window in points `(P, 48)` (NaN where the reading is not shown).
    """

    t: np.ndarray
    ti: np.ndarray | None
    pair: np.ndarray
    delta: np.ndarray
    cosz: np.ndarray
    csi: np.ndarray
    dbsp: np.ndarray
    keep_raisable: np.ndarray
    picks: list[tuple[int, int]]
    unreliable: np.ndarray | None = None
    keys: np.ndarray | None = None
    r_points: np.ndarray | None = None

    def flat_keys(self) -> np.ndarray:
        """Return the row keys `(3, N)` on the valid half-hours: segment, date, half-hour.

        Raises:
            ValueError: When the truth holds no keys.
        """
        if self.keys is None:
            msg = "this truth holds no row keys"
            raise ValueError(msg)
        return self.keys[:, self.valid()]

    def valid(self) -> np.ndarray:
        """Return the valid half-hours `(P, 48)`."""
        return self.t[:, arms.T["valid"]] > 0.5

    def flat(self, *, values: np.ndarray) -> np.ndarray:
        """Return `(P, 48)` values on the valid half-hours, in the order of experiment 729's `Rows`."""
        return values[self.valid()]

    def row(self, *, name: str) -> np.ndarray:
        """Return one target row `(P, 48)` of the plain targets."""
        return self.t[:, arms.T[name]]


def truth(
    *,
    prepared: Sequence[arms.Prepared],
    index: ov.PairIndex | None = None,
    amount: float | None = None,
    eps: float | None = None,
    picks: list[tuple[int, int]] | None = None,
) -> Truth:
    """Return the truth arrays of every scored day of `prepared`.

    The arrays include the injected member's targets when an amount is given.
    """
    picks = arms.all_picks(prepared=prepared) if picks is None else picks
    spec = arms.eval_spec(inject=amount is not None, raised=eps is not None)
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("t", "ti", "pair", "delta", "cosz", "csi", "dbsp", "raisable", "r")
    }
    for i in range(0, len(picks), CHUNK):
        b = arms.make_eval_batch(
            prepared=prepared,
            picks=picks[i : i + CHUNK],
            spec=spec,
            index=index,
            amount=amount,
            eps=eps,
        )
        parts["t"].append(b.t)
        if b.ti is not None:
            parts["ti"].append(b.ti)
        parts["pair"].append(b.pair)
        parts["delta"].append(b.delta)
        scored = slice(arms.MID, arms.MID + 48)
        parts["cosz"].append(b.x[:, arms.ROW["cosz"], scored])
        parts["csi"].append(b.x[:, arms.ROW["csi"], scored])
        parts["dbsp"].append(b.x[:, arms.ROW["d_bsp"], scored])
        parts["raisable"].append(b.keeps[:, arms.K["raisable"]] > 0.5)
        shown = b.x[:, arms.ROW["r_mask"], scored] > 0.5
        parts["r"].append(
            np.where(
                shown,
                b.x[:, arms.ROW["r"], scored].astype(np.float64) * arms.arms729.R_SCALE,
                np.nan,
            )
        )
    cat = {k: np.concatenate(v) if v else None for k, v in parts.items()}
    return Truth(
        t=cat["t"],  # ty: ignore[invalid-argument-type]
        ti=cat["ti"],
        pair=cat["pair"],  # ty: ignore[invalid-argument-type]
        delta=cat["delta"],  # ty: ignore[invalid-argument-type]
        cosz=cat["cosz"],  # ty: ignore[invalid-argument-type]
        csi=cat["csi"],  # ty: ignore[invalid-argument-type]
        dbsp=cat["dbsp"],  # ty: ignore[invalid-argument-type]
        keep_raisable=cat["raisable"],  # ty: ignore[invalid-argument-type]
        picks=picks,
        keys=row_keys(prepared=prepared, picks=picks),
        r_points=cat["r"],
        unreliable=np.stack([arms.unreliable_day(p=prepared[i], day=d) for i, d in picks])
        if picks
        else None,
    )  # ty: ignore[invalid-argument-type]


def row_keys(
    *, prepared: Sequence[arms.Prepared], picks: list[tuple[int, int]]
) -> np.ndarray | None:
    """Return the (segment, date, half-hour) key of every half-hour of the windows `(3, P, 48)`.

    The segment is the position in the bank's list and the date is the number of days since
    1970-01-01, the keys of experiment 729's `Rows`. The result is None when a segment holds no dates.
    """
    if not picks:
        return None
    out = np.zeros((3, len(picks), 48), dtype=np.int64)
    for k, (i, d) in enumerate(picks):
        dates = prepared[i].dates
        if dates is None:
            return None
        out[0, k] = prepared[i].source_index
        out[1, k] = int(
            (dates[d].astype("datetime64[D]") - np.datetime64("1970-01-01", "D")).astype(int)
        )
        out[2, k] = np.arange(48)
    return out
