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
        k: [] for k in ("t", "ti", "pair", "delta", "cosz", "csi", "dbsp", "raisable")
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
    )  # ty: ignore[invalid-argument-type]
