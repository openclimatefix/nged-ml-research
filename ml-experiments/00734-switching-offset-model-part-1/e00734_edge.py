"""The edge check of experiment 734's tuning: did an arm's chosen value sit on its grid's edge?

Every arm is tuned over a grid. A chosen value at the lowest or the highest grid value of its
parameter is an edge hit when the four-cell objective is still better there than at the next grid
value inward by at least `MIN_GAIN`, because the best value may then lie outside the grid. A
smaller gain is a plateau. The tau of an arm is an edge hit at the grid floor unless the rate of
calls binds there, because a lower tau would then be ineligible. The functions are pure and are
called by `choose` for every arm in the same way, so no arm is tuned harder than another.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

MIN_GAIN: Final[float] = 0.005
"""Smallest objective gain towards the edge that counts as 'still improving'."""
RATE_TOL: Final[float] = 1e-9
"""Relative tolerance for 'the called rate equals the limit'."""


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def parameter_edge(
    *,
    arm: str,
    name: str,
    grid: Sequence[float],
    chosen: float,
    objective_by_value: dict[float, float],
) -> dict:
    """Edge check of one parameter against the objective profile over its grid.

    Args:
        arm: The arm's name.
        name: The parameter's name.
        grid: The grid values of the parameter.
        chosen: The chosen value (a member of the grid).
        objective_by_value: For each grid value, the best objective among the eligible settings
            that use it (NaN when none is eligible).

    Returns:
        A record with the position of the chosen value, whether it is on an edge, the objective at
        the chosen value and at the next value inward, and `hit`.
    """
    values = sorted({float(v) for v in grid})
    pos = min(range(len(values)), key=lambda i: abs(values[i] - chosen))
    edge = "lowest" if pos == 0 else "highest" if pos == len(values) - 1 else None
    if len(values) < 2:
        edge = None
    inward = None if edge is None else values[1] if edge == "lowest" else values[-2]
    obj_c = objective_by_value.get(values[pos], float("nan"))
    obj_n = float("nan") if inward is None else objective_by_value.get(inward, float("nan"))
    gain = obj_c - obj_n if _finite(obj_c) and _finite(obj_n) else float("nan")
    ineligible = [
        side
        for side, k in (("lower", pos - 1), ("higher", pos + 1))
        if 0 <= k < len(values) and not _finite(objective_by_value.get(values[k], float("nan")))
    ]
    hit = edge is not None and _finite(gain) and gain >= MIN_GAIN
    return {
        "arm": arm,
        "parameter": name,
        "grid": values,
        "chosen": values[pos],
        "position": f"{pos + 1} of {len(values)}",
        "edge": edge,
        "inward_neighbour": inward,
        "objective_at_chosen": obj_c,
        "objective_at_neighbour": obj_n,
        "gain_towards_edge": gain,
        "hit": hit,
        "ineligible_neighbours": ineligible,
        "classification": (
            "edge hit"
            if hit
            else "rate-binding: the next value " + " and ".join(ineligible) + " is ineligible"
            if ineligible
            else "on an edge, plateau"
            if edge is not None
            else "interior"
        ),
        "note": (
            ""
            if edge is None
            else "inward neighbour ineligible"
            if not _finite(gain)
            else "plateau"
            if gain < MIN_GAIN
            else ""
        ),
    }


def tau_edge(
    *,
    arm: str,
    grid: Sequence[float],
    chosen: float,
    big_rate: float,
    big_limit: float,
    all_rate: float,
    all_limit: float,
) -> dict:
    """Edge check of a threshold tau: the grid floor is a hit unless a called rate binds there.

    Args:
        arm: The arm's name.
        grid: The tau grid.
        chosen: The chosen tau.
        big_rate: The rate of calls of 2 noise units or more at the chosen tau.
        big_limit: E1-J-sym's rate of such calls.
        all_rate: The rate of all calls at the chosen tau.
        all_limit: E1-J-sym's rate of all calls.

    Returns:
        A record like `parameter_edge`'s, with `rate_bound` saying that a lower tau is ineligible.
    """
    values = sorted({float(v) for v in grid})
    pos = min(range(len(values)), key=lambda i: abs(values[i] - chosen))
    edge = "lowest" if pos == 0 else "highest" if pos == len(values) - 1 else None
    bound = big_rate >= big_limit * (1 - RATE_TOL) or all_rate >= all_limit * (1 - RATE_TOL)
    return {
        "arm": arm,
        "parameter": "tau",
        "grid": values,
        "chosen": values[pos],
        "position": f"{pos + 1} of {len(values)}",
        "edge": edge,
        "rate_bound": bool(bound),
        "hit": edge is not None and not bound,
        "classification": (
            "interior"
            if edge is None
            else "rate-binding floor: a called rate binds, so a lower tau is ineligible"
            if bound
            else "edge hit"
        ),
        "note": "a called rate binds here, so a lower tau is ineligible" if bound else "",
    }


def profile(*, table: Sequence[dict], key: str, objective: str = "objective") -> dict[float, float]:
    """For each value of `key`, the best objective among the eligible rows that use it."""
    out: dict[float, float] = {}
    for row in table:
        value = float(row[key])
        obj = row[objective] if row.get("eligible") else float("nan")
        if value not in out or (_finite(obj) and (not _finite(out[value]) or obj > out[value])):
            out[value] = obj if _finite(obj) else out.get(value, float("nan"))
    return out


def summary_lines(*, checks: Sequence[dict]) -> list[str]:
    """One line per parameter, with EDGE HIT where the check fired."""
    lines = []
    for c in checks:
        base = f"{c['arm']:7s} {c['parameter']:9s} chosen {c['chosen']:g} ({c['position']})"
        if c["hit"] and c.get("final_extension"):
            extra = (
                f"tuned to the edge of the extended grid (the {c['edge']} value; "
                "the grid is not extended again)"
            )
        elif c["hit"]:
            extra = (
                f"EDGE HIT at the {c['edge']} value"
                + (
                    f"; objective {c['objective_at_chosen']:.4f} against "
                    f"{c['objective_at_neighbour']:.4f} at {c['inward_neighbour']:g}"
                    if "objective_at_chosen" in c
                    else ""
                )
                + "; choose never extends a grid by itself"
            )
        elif c["edge"] is not None:
            extra = f"on the {c['edge']} edge, no hit ({c['classification']})"
        else:
            extra = c["classification"]
        lines.append(f"{base}: {extra}")
    return lines


def equal_effort_lines(*, checks: Sequence[dict]) -> list[str]:
    """The line that asks for a decision when PELT or H-nbr hit an edge, as the other arms may."""
    hits = [c for c in checks if c["hit"] and c["arm"] in ("PELT", "H-nbr")]
    return (
        [
            "EQUAL EFFORT: PELT/H-nbr edge hit, needs a decision: "
            + ", ".join(f"{c['arm']} {c['parameter']}" for c in hits)
        ]
        if hits
        else []
    )
