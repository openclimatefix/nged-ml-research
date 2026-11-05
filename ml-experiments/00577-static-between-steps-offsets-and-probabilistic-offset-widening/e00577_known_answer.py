# ruff: noqa: E501
"""Experiment 577: the known-answer check of the offset penalties, on experiment 573's cached network outputs.

Two worlds, each added to the cleaned power of the six Primaries that experiment 573 injects:

- `K0`: nothing added, so the true injected offset is zero.
- `Kl`: two persistent steps of 10% to 20% of capacity per Primary (sign random, dates between
  2022-01-01 and 2026-05-31, at least 180 days apart).

Arms (each fits only the offset, on the cached nine mixture outputs of the frozen network, as in
experiment 573): `D5` (monthly knots, squared first-difference penalty, weight 10), `O1` (monthly
knots, L1 total variation), `O2` (daily knots, L1 total variation, zero-order hold), and `O3`
(experiment 573's daily hard-concrete gates started from `O2`). The L1 weights are 300, 1000, and
3000 nats per standardised unit.

Reported per fold, world, and arm: the offset's root-mean-square error in static periods (days more
than 10 days from any injected step; for `K0` every day), and for `Kl` the recovered step size and
date of each step that the experiment-573 extraction finds within 3 days, with the step-size bias.
Prints a table and writes `known_answer.json`.

Usage: python e00577_known_answer.py [--months ...]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00573_steps as s  # ty: ignore[unresolved-import]
import e00573_world as world  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00577-static-between-steps-offsets-and-probabilistic-offset-widening"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "known_answer.json"
STEP_SIZE: Final = (0.10, 0.20)
STATIC_MARGIN: Final = 10
TV_SWEEP: Final = (1000.0, 3000.0, 10000.0, 30000.0)
GATE: Final = 100.0


def large_steps(*, primary: int) -> list[tuple[int, float]]:
    """Return a Primary's two persistent steps `(day, signed size as a capacity fraction)` in world Kl."""
    rng = np.random.default_rng([577, primary])
    first = world.day_of(t=datetime(2022, 1, 1, tzinfo=UTC)) + int(rng.integers(0, 600))
    last = world.day_of(t=datetime(2026, 5, 31, tzinfo=UTC))
    second = first + 180 + int(rng.integers(0, max(1, last - first - 180)))
    return [
        (day, float(rng.uniform(*STEP_SIZE) * rng.choice([-1.0, 1.0]))) for day in (first, second)
    ]


def injection_large(*, grid: int) -> np.ndarray:
    """Return world Kl's added component (16 Primaries, `grid` half-hours), a fraction of capacity."""
    out = np.zeros((16, grid), dtype=np.float32)
    days = np.arange(grid) // world.PER_DAY
    for p in world.INJECTED:
        for day, size in large_steps(primary=p):
            out[p, days >= day] += size
    return out


def fit_o1(*, fold: s.Fold, z: torch.Tensor, lam: float) -> np.ndarray:
    """Return O1's daily path (16, days): monthly knots, linear interpolation, L1 total variation."""
    knots = torch.zeros(16, fold.n_knots, device=fold.out.device, requires_grad=True)
    tau = fold.tau.clamp(0.0, float(fold.n_knots - 1))
    lo = tau.floor().long().clamp(max=fold.n_knots - 2)
    frac = tau - lo.to(tau.dtype)
    n = float(len(z))

    def loss() -> torch.Tensor:
        rows = (1 - frac) * knots[fold.primary, lo] + frac * knots[fold.primary, lo + 1]
        pen = lam * (knots[:, 1:] - knots[:, :-1]).abs().sum()
        return (s.nll_sum(fold=fold, z=z, offset_rows=rows) + pen) / n

    s._optimise(params=[knots], loss_fn=loss, steps=s.STEPS)
    with torch.no_grad():
        days = np.datetime64("2019-09-01T12:00:00", "us") + np.arange(fold.n_days)[:, None].astype(
            "timedelta64[D]"
        ).astype("timedelta64[us]")
        days = days.ravel()
        start = days.astype("datetime64[M]")
        lo_t, hi_t = start.astype("datetime64[us]"), (start + 1).astype("datetime64[us]")
        t_np = (start - np.datetime64("2019-09", "M")).astype(np.float64) + (days - lo_t) / (
            hi_t - lo_t
        )
        t_day = (
            torch.from_numpy(t_np.astype(np.float32))
            .to(fold.out.device)
            .clamp(0.0, float(fold.n_knots - 1))
        )
        l2 = t_day.floor().long().clamp(max=fold.n_knots - 2)
        f2 = t_day - l2.float()
        return ((1 - f2)[None] * knots[:, l2] + f2[None] * knots[:, l2 + 1]).cpu().numpy()


def truth_path(*, world_name: str, n_days: int) -> np.ndarray:
    """Return the injected offset per Primary and day (capacity fraction); zero in K0."""
    out = np.zeros((16, n_days))
    if world_name == "Kl":
        for p in world.INJECTED:
            for day, size in large_steps(primary=p):
                out[p, day:] += size
    return out


def score(*, fold: s.Fold, path: np.ndarray, world_name: str, baseline: np.ndarray | None) -> dict:
    """Return the static-period RMSE (capacity fraction) and the recovered steps of a fitted path."""
    truth = truth_path(world_name=world_name, n_days=fold.n_days)
    cap = path * fold.std[:, None]
    first = np.array([int(np.argmax(fold.rows_per_day[p] > 0)) for p in range(16)])
    limit = fold.span_end_day - s.END_MARGIN
    # The real series' own drift is in every fit, so Kl is scored against K0's fit plus the truth.
    target = truth + (baseline * fold.std[:, None] if baseline is not None else 0.0)
    sq, count = 0.0, 0
    for p in world.INJECTED:
        steps = [d for d, _ in large_steps(primary=p)] if world_name == "Kl" else []
        static = np.ones(fold.n_days, dtype=bool)
        for d in steps:
            static[max(d - STATIC_MARGIN, 0) : d + STATIC_MARGIN + 1] = False
        static[: first[p]] = False
        static[limit + 1 :] = False
        err = (cap[p] - target[p])[static]
        sq, count = sq + float((err**2).sum()), count + int(static.sum())
    res: dict = {"static_rmse": float(np.sqrt(sq / max(count, 1)))}
    if world_name == "Kl":
        found_all, bias = [], []
        for p in world.INJECTED:
            found = [(d, sz) for d, sz in s.extract(path=path[p], std=fold.std[p]) if d <= limit]
            for day, size in large_steps(primary=p):
                if day > limit or not s.supported(fold=fold, primary=p, day=day):
                    continue
                hit = [
                    (d, sz)
                    for d, sz in found
                    if abs(d - day) <= s.TOL_DAYS and np.sign(sz) == np.sign(size)
                ]
                if hit:
                    best = min(hit, key=lambda t: abs(t[1] - size))
                    found_all.append(
                        {
                            "primary": p,
                            "day": day,
                            "true": size,
                            "found": best[1],
                            "found_day": best[0],
                        }
                    )
                    bias.append((abs(best[1]) - abs(size)) / abs(size))
                else:
                    found_all.append(
                        {"primary": p, "day": day, "true": size, "found": None, "found_day": None}
                    )
        res["steps"] = found_all
        res["recovered"] = sum(1 for r in found_all if r["found"] is not None)
        res["scored_steps"] = len(found_all)
        res["mean_size_bias"] = float(np.mean(bias)) if bias else None
    return res


def main() -> None:
    """Fit every arm in K0 and Kl on each fold and write the scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=list(s.MONTHS))
    args = parser.parse_args()
    out: dict = {}
    for month in args.months:
        fold = s.load_fold(month=month, device="cuda")
        grid = data.grid_length()
        inj = {"K0": np.zeros((16, grid), dtype=np.float32), "Kl": injection_large(grid=grid)}
        paths: dict[str, dict[str, np.ndarray]] = {}
        for w, add in inj.items():
            z = s.world_z(fold=fold, inj=add, device="cuda")
            p = {"D5": s.fit_s0(fold=fold, z=z)}
            for lam in TV_SWEEP:
                p[f"O1@{lam:g}"] = fit_o1(fold=fold, z=z, lam=lam)
                p[f"O2@{lam:g}"] = s.fit_s1(fold=fold, z=z, lam=lam)
            p[f"O3@3000/{GATE:g}"] = s.fit_s2(fold=fold, z=z, lam=GATE, init=p["O2@3000"])
            paths[w] = p
        out[month] = {}
        for w in inj:
            out[month][w] = {
                arm: score(
                    fold=fold,
                    path=path,
                    world_name=w,
                    baseline=paths["K0"][arm] if w == "Kl" else None,
                )
                for arm, path in paths[w].items()
            }
            for arm, r in out[month][w].items():
                print(month, w, arm, {k: v for k, v in r.items() if k != "steps"}, flush=True)
        del fold
        torch.cuda.empty_cache()
    OUT.write_text(
        json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )


if __name__ == "__main__":
    main()
