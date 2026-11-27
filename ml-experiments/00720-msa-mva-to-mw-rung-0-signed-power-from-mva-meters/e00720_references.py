# ruff: noqa: E501, PT018
"""Experiment 720: the two earlier references, scored on the same injected worlds as the new read-outs.

Experiment 506's gated solar rule calls a half-hour reverse when the PV estimate `p` (the host's own
estimate plus the injected capacity times the unit shape `u`) exceeds the dark-hour demand level of
the same month and weekday class, on rows with `u >= 0.3`. Experiment 674's day labeller assigns a
class to each Primary-day (`reverse`, `no_reverse`, `unknown`, or `not_scorable`) from stored day
tables. The day tables belong to experiment 674's own injection (levels 0, 1, 2, and 3 of the same
definition); this module reads their stored labels, reads nothing new, and compares them on the
days both cover. The functions here are pure, and `selftest` checks them on synthetic arrays.
"""

import sys
from pathlib import Path
from typing import Final

import e00720_sign as sign
import numpy as np

U_GATE: Final[float] = 0.3
MIN_DARK_ROWS: Final[int] = 50


def dark_demand_levels(
    *, v: np.ndarray, u: np.ndarray, month: np.ndarray, weekend: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Return the per-day dark-hour demand level of experiment 506's rule, on `(days, 48)` arrays.

    Args:
        v: Magnitude of the reading with the floor removed, `(days, 48)`.
        u: Unit PV shape.
        month: Calendar month of each day, `(days,)`.
        weekend: 1 where the day is a Saturday or a Sunday, `(days,)`.
        rows: Indices of the reference days that the level is learned from.

    Returns:
        `(days,)` the median of `v` in the dark half-hours (`u < 0.01`) of the reference days of the
        same month and weekday class, falling back to the weekday class alone, then to NaN.
    """
    key = month * 2 + weekend
    dark = np.zeros(v.shape, dtype=bool)
    dark[rows] = (u[rows] < 0.01) & np.isfinite(v[rows])
    out = np.full(len(v), np.nan)
    by_wk = {
        w: float(np.median(v[dark & (weekend == w)[:, None]]))
        for w in (0, 1)
        if (dark & (weekend == w)[:, None]).sum() > MIN_DARK_ROWS
    }
    for c in np.unique(key):
        sel = dark & (key == c)[:, None]
        if sel.sum() > MIN_DARK_ROWS:
            out[key == c] = float(np.median(v[sel]))
        else:
            wk = int(c % 2)
            out[key == c] = by_wk.get(wk, np.nan)
    return out


def gated_rule_called(*, p: np.ndarray, u: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Return the rule's call per half-hour: `p > demand` on rows with `u >= 0.3`, NaN elsewhere (0 or 1)."""
    ok = (u >= U_GATE) & np.isfinite(p) & np.isfinite(demand)[:, None]
    called = (p > demand[:, None]).astype(float)
    return np.where(ok, called, np.nan)


def rule_metrics(
    *,
    called: np.ndarray,
    x_true: np.ndarray,
    u: np.ndarray,
    dead_band: float = sign.DEAD_BAND_POINTS,
) -> dict[str, float]:
    """Return recall and false-reverse rate of hard calls on the rows `evaluate_sign` would score."""
    ok = np.isfinite(called) & np.isfinite(x_true) & (u >= U_GATE) & (np.abs(x_true) >= dead_band)
    pos = x_true[ok] < 0.0
    c = called[ok] > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return {
        "recall": float((c & pos).sum() / n_pos) if n_pos else float("nan"),
        "false_reverse": float((c & ~pos).sum() / n_neg) if n_neg else float("nan"),
        "n_reverse": float(n_pos),
        "n_import": float(n_neg),
    }


def day_call_from_rho(*, rho: np.ndarray, u: np.ndarray, min_rows: int = 2) -> np.ndarray:
    """Return a day-level call from the sign probabilities.

    The call is true when at least `min_rows` gated half-hours have `rho > 0.5` (the labeller's own
    count is 2).
    """
    gated = (u >= U_GATE) & np.isfinite(rho)
    return ((rho > 0.5) & gated).sum(axis=1) >= min_rows


def day_truth(
    *,
    x_true: np.ndarray,
    u: np.ndarray,
    dead_band: float = sign.DEAD_BAND_POINTS,
    min_rows: int = 2,
) -> np.ndarray:
    """Return whether a day has at least `min_rows` gated half-hours below minus the dead band."""
    return ((x_true < -dead_band) & (u >= U_GATE)).sum(axis=1) >= min_rows


def day_scores(
    *, call: np.ndarray, truth: np.ndarray, label: np.ndarray | None = None
) -> dict[str, float]:
    """Return recall, false-reverse rate, and counts over days.

    Args:
        call: Boolean day calls (the new read-out), or, with `label`, ignored for the rows whose label is
            `unknown` or `not_scorable` (those days are counted, not scored).
        truth: Boolean day truth.
        label: Experiment 674's class per day (`reverse`, `no_reverse`, `unknown`, `not_scorable`). With
            it, `call` is replaced by `label == "reverse"` and unknown or not-scorable days are excluded.

    Returns:
        A dictionary with `recall`, `false_reverse` (NaN when its denominator is zero), `n_days`
        (the scored days), `n_unknown`, and `n_not_scorable`.
    """
    if label is not None:
        scorable = np.isin(label, ["reverse", "no_reverse"])
        call = label == "reverse"
        n_unknown = float((label == "unknown").sum())
        n_not_scorable = float((label == "not_scorable").sum())
    else:
        scorable = np.ones(len(call), dtype=bool)
        n_unknown = n_not_scorable = 0.0
    t, c = truth[scorable], call[scorable]
    return {
        "recall": float((c & t).sum() / t.sum()) if t.any() else float("nan"),
        "false_reverse": float((c & ~t).sum() / (~t).sum()) if (~t).any() else float("nan"),
        "n_days": float(scorable.sum()),
        "n_unknown": n_unknown,
        "n_not_scorable": n_not_scorable,
    }


def read_674_labels(*, work: Path | None = None, arm: str = "B"):  # noqa: ANN201
    """Read experiment 674's stored known-answer day labels (read-only).

    The columns are `site`, `level`, `date`, and `cls`, and experiment 674's own truth columns
    `any_negative` and `crossing`.
    """
    import polars as pl

    if work is None:
        import e00674_features as feat  # ty: ignore[unresolved-import]

        work = feat.WORK
    return pl.read_parquet(work / "known_answer_days.parquet").select(
        "site", "level", "date", "any_negative", "crossing", cls=f"cls_{arm}"
    )


def selftest() -> int:
    """Check the reference functions on synthetic arrays with known answers."""
    rng = np.random.default_rng(5)
    nd = 120
    u = np.tile(np.clip(np.sin(np.pi * (np.arange(48) - 10) / 28.0), 0, 1), (nd, 1)) * rng.uniform(
        0.5, 1, (nd, 1)
    )
    month = np.tile(np.arange(1, 13), nd // 12 + 1)[:nd]
    weekend = (np.arange(nd) % 7 >= 5).astype(int)
    demand_true = 20.0 + 3.0 * weekend
    x = (
        demand_true[:, None] - 30.0 * u * (np.arange(nd) % 3 == 0)[:, None]
    )  # every third day exports
    v = np.abs(x)
    p = 30.0 * u * (np.arange(nd) % 3 == 0)[:, None]  # the estimate is large only where the PV is
    lvl = dark_demand_levels(v=v, u=u, month=month, weekend=weekend, rows=np.arange(nd))
    assert np.allclose(lvl, demand_true)
    called = gated_rule_called(p=p, u=u, demand=lvl)
    m = rule_metrics(called=called, x_true=x, u=u)
    assert m["recall"] > 0.99 and m["false_reverse"] < 0.01, m
    # a rule that has a capacity twice too small never reverses
    called_low = gated_rule_called(p=0.3 * p, u=u, demand=lvl)
    assert rule_metrics(called=called_low, x_true=x, u=u)["recall"] < 0.05
    truth = day_truth(x_true=x, u=u)
    assert 10 < truth.sum() <= (np.arange(nd) % 3 == 0).sum()
    rho = (x < -2.0).astype(float)
    ds = day_scores(call=day_call_from_rho(rho=rho, u=u), truth=truth)
    assert ds["recall"] == 1.0 and ds["false_reverse"] == 0.0
    label = np.where(truth, "reverse", "no_reverse").astype(object)
    label[:5] = "unknown"
    ls = day_scores(call=truth, truth=truth, label=label)
    assert ls["n_unknown"] == 5.0 and ls["n_days"] == nd - 5
    print("PASS references: dark level, gated rule, day labels, and day scores")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
