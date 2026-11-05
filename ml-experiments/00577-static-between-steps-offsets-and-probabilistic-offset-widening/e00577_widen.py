# ruff: noqa: E501
"""Experiment 577: widen the network's mixture for the offset's uncertainty (Q1, Q2, Q3, Q-const).

Every function works in standardised units on the three main mixture components
(`log_w`, `mean`, `std`, each (rows, 3)). A widening adds a variance to every component, because
a component N(mu + o, sigma squared) with o drawn from N(0, v) is N(mu, sigma squared + v) exactly.
"""

from typing import Final

import e00577_model_variants as mv  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

CTX: dict = {"month": None, "day": None, "seed": None}
"""The fold, refit day, and seed of the model being used (set before each forecast)."""
Q1_SEEDS: Final[tuple[int, ...]] = (5, 6, 7, 8, 9)
EXTRA_WEIGHT_CAP: Final[float] = 0.05
_Q1_CACHE: dict = {}


def q1_variance(*, w565: object, month: str, day: int, device: str) -> torch.Tensor:
    """Return each Primary's variance across seeds 5 to 9 of the newest knot (standardised units squared)."""
    key = (month, day)
    if key not in _Q1_CACHE:
        knots = torch.stack(
            [
                torch.load(w565 / "models" / "D5" / f"{month}_d{day}_s{s}.pt", map_location="cpu")[  # ty: ignore[unsupported-operator]
                    "knot_new"
                ][:, 0].float()  # ty: ignore[unsupported-operator]
                for s in Q1_SEEDS
            ]
        )
        _Q1_CACHE[key] = knots.var(dim=0, unbiased=True)
    return _Q1_CACHE[key].to(device).double()


def days_since_cutoff(*, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return each row's valid time in days after the refit cutoff (zero before it)."""
    n_monthly = getattr(model, "n_monthly", None) or model.n_knots  # ty: ignore[unresolved-attribute]
    after_month_start = (x[:, mv.TAU_COL].double() - float(n_monthly - 1)) * mv.MONTH_DAYS  # ty: ignore[unsupported-operator]
    return (after_month_start - model.cut_offset_days).clamp(min=0.0)  # ty: ignore[unsupported-operator, unresolved-attribute]


def apply(
    *, post: str, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor, log_w: torch.Tensor,
    mean: torch.Tensor, std: torch.Tensor, picks: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Return the mixture after the arm's widening."""
    if post == "none":
        return log_w, mean, std
    if post == "var":
        v = model.extra_variance(x, p).double()  # ty: ignore[call-non-callable]
        return log_w, mean, torch.sqrt(std**2 + v[:, None])
    if post == "q1":
        from studies.sources import ML_EXPERIMENTS_DATA_DIR

        w565 = ML_EXPERIMENTS_DATA_DIR / "00565-drift-embedding-variants-six-designs" / "work"
        v = q1_variance(w565=w565, month=CTX["month"], day=CTX["day"], device=str(x.device))[p]
        return log_w, mean, torch.sqrt(std**2 + v[:, None])
    if post == "qconst":
        scale = torch.tensor(picks["qconst"], dtype=torch.float64, device=x.device)[p]
        return log_w, mean, std * scale[:, None]
    if post in {"q3", "q3b"}:
        stats = picks[post]
        rate, step = float(stats["rate_per_day"]), float(stats["step_scale"])
        days = days_since_cutoff(model=model, x=x)
        step_z = step / model.target_std[p]  # ty: ignore[not-subscriptable, unresolved-attribute]
        v = model.extra_variance(x, p).double() + rate * step_z**2 * days  # ty: ignore[call-non-callable]
        std = torch.sqrt(std**2 + v[:, None])
        weight = (rate * days).clamp(max=EXTRA_WEIGHT_CAP)
        centre = (log_w.exp() * mean).sum(dim=-1, keepdim=True)
        log_w = torch.cat(
            [log_w + torch.log1p(-weight)[:, None], torch.log(weight.clamp(min=1e-12))[:, None]],
            dim=-1,
        )
        return log_w, torch.cat([mean, centre], dim=-1), torch.cat([std, step_z[:, None]], dim=-1)
    raise ValueError(post)
