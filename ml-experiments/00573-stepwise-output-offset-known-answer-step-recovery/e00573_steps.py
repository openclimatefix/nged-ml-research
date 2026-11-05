# ruff: noqa: E501
"""Experiment 573, Step 1: fit the offsets S0, S1, and S2 on the frozen network's cached outputs and score step recovery.

Usage: python e00573_steps.py {sweep|run} [--months ...] [--worlds ...]

Every arm fits only its offset: the nine mixture outputs of each pre-training row are cached (see
`e00573_cache.py`), the offset is added to the three component means (columns 3 to 5) in
standardised units, and the loss is the mixture negative log-likelihood of the world's targets.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00573_world as world  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00573-stepwise-output-offset-known-answer-step-recovery"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
COMPONENTS: Final = 3
PATHS: Path = WORK / "paths_sweep"
"""Where the offset paths are saved; `main` points it at the command's own folder."""
S0_LAMBDA: Final = 10.0
"""Experiment 565's penalty weight on the mean squared first difference of the monthly knots."""
TV_SWEEP: Final = (300.0, 1000.0, 3000.0)
"""Nats of negative log-likelihood per standardised unit of total variation (S1)."""
GATE_SWEEP: Final = (30.0, 100.0, 300.0)
"""Nats per expected open gate (S2)."""
STEPS: Final = 1000
SUBSAMPLE: Final = 0.25
"""Share of each fold's pre-training rows the offset fits use (a fixed random subset; the rows of one half-hour differ only in lead)."""
INIT_JUMP: Final = 0.001
"""S2 starts with a gate open at every day where S1's offset changes by 0.1% of capacity or more."""
LR: Final = 0.02
DETECT_PER_DAY: Final = 0.001
"""A day's change of the offset counts toward a step when it is at least 0.1% of capacity."""
MERGE_GAP: Final = 2
MIN_STEP: Final = 0.01
TOL_DAYS: Final = 3
TOL_SIZE: Final = 0.25
END_MARGIN: Final = 10
"""Truth steps in the last 10 days of a fold's span are not scored: the offset has too few rows after them."""
HC_BETA, HC_GAMMA, HC_ZETA = 2.0 / 3.0, -0.1, 1.1


@dataclass
class Fold:
    """One fold's cached rows on the GPU."""

    month: str
    out: torch.Tensor
    y: np.ndarray
    primary: torch.Tensor
    primary_np: np.ndarray
    target_idx: np.ndarray
    day: torch.Tensor
    tau: torch.Tensor
    mean: np.ndarray
    std: np.ndarray
    n_days: int
    n_knots: int
    span_end_day: int
    rows_per_day: np.ndarray


def load_fold(*, month: str, device: str) -> Fold:
    """Load a fold's cache."""
    z = np.load(WORK / "cache" / f"{month}.npz")
    keep = np.sort(np.random.default_rng(573).permutation(len(z["y"]))[: int(SUBSAMPLE * len(z["y"]))])
    primary = z["primary"][keep].astype(np.int64)
    target_idx = z["target_idx"][keep].astype(np.int64)
    day = target_idx // world.PER_DAY
    year, mon = (int(p) for p in month.split("-"))
    n_knots = (year - 2019) * 12 + (mon - 9) + 1
    us = target_idx.astype("int64") * 1_800_000_000
    t64 = (np.datetime64("2019-09-01T00:00:00", "us") + us.astype("timedelta64[us]"))
    start = t64.astype("datetime64[M]")
    lo, hi = start.astype("datetime64[us]"), (start + 1).astype("datetime64[us]")
    tau = ((start - np.datetime64("2019-09", "M")).astype(np.float64) + (t64 - lo) / (hi - lo)).astype(np.float32)
    n_days = int(z["target_idx"].max() // world.PER_DAY) + 1
    rows_per_day = np.zeros((16, n_days))
    np.add.at(rows_per_day, (primary, day), 1.0)
    return Fold(
        month=month, out=torch.from_numpy(z["out"][keep]).to(device), y=z["y"][keep],
        primary=torch.from_numpy(primary).to(device), primary_np=primary, target_idx=target_idx,
        day=torch.from_numpy(day).to(device), tau=torch.from_numpy(tau).to(device),
        mean=z["target_mean"].astype(np.float64), std=z["target_std"].astype(np.float64),
        n_days=int(z["target_idx"].max() // world.PER_DAY) + 1, n_knots=n_knots, span_end_day=int(z["target_idx"].max() // world.PER_DAY), rows_per_day=rows_per_day,
    )  # fmt: skip


def world_z(*, fold: Fold, inj: np.ndarray, device: str) -> torch.Tensor:
    """Return the world's standardised targets: the real target plus the injection, MVA Primaries clipped at zero."""
    add = inj[fold.primary_np, fold.target_idx]
    y = fold.y.astype(np.float64) + add
    mva = np.isin(fold.primary_np, list(world.MVA_INJECTED))
    y = np.where(mva, np.maximum(y, 0.0), y)
    z = (y - fold.mean[fold.primary_np]) / fold.std[fold.primary_np]
    return torch.from_numpy(z.astype(np.float32)).to(device)


def nll_sum(*, fold: Fold, z: torch.Tensor, offset_rows: torch.Tensor) -> torch.Tensor:
    """Return the summed mixture negative log-likelihood with the row offset added to the component means."""
    k = COMPONENTS
    out = torch.cat(
        [fold.out[:, :k], fold.out[:, k : 2 * k] + offset_rows[:, None], fold.out[:, 2 * k :]], dim=-1
    )
    return base.mixture_nll(out=out, z=z, components=k).sum()


def _optimise(*, params: list[torch.Tensor], loss_fn, steps: int) -> float:  # noqa: ANN001
    """Run Adam with a cosine-decayed rate; return the final loss."""
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR / 10)
    last = float("nan")
    for _ in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        last = float(loss)
    return last


def fit_s0(*, fold: Fold, z: torch.Tensor) -> np.ndarray:
    """Return S0's daily path (16, days): monthly knots, linear interpolation, squared first-difference penalty."""
    knots = torch.zeros(16, fold.n_knots, device=fold.out.device, requires_grad=True)
    tau = fold.tau.clamp(0.0, float(fold.n_knots - 1))
    lo = tau.floor().long().clamp(max=fold.n_knots - 2)
    frac = tau - lo.to(tau.dtype)
    n = float(len(z))

    def rows(k: torch.Tensor) -> torch.Tensor:
        return (1 - frac) * k[fold.primary, lo] + frac * k[fold.primary, lo + 1]

    def loss() -> torch.Tensor:
        pen = S0_LAMBDA * ((knots[:, 1:] - knots[:, :-1]) ** 2).mean()
        return nll_sum(fold=fold, z=z, offset_rows=rows(knots)) / n + pen

    _optimise(params=[knots], loss_fn=loss, steps=STEPS)
    with torch.no_grad():
        days = (np.datetime64("2019-09-01T12:00:00", "us") + np.arange(fold.n_days)[:, None].astype("timedelta64[D]").astype("timedelta64[us]")).ravel()
        start = days.astype("datetime64[M]")
        lo, hi = start.astype("datetime64[us]"), (start + 1).astype("datetime64[us]")
        t_np = (start - np.datetime64("2019-09", "M")).astype(np.float64) + (days - lo) / (hi - lo)
        t_day = torch.from_numpy(t_np.astype(np.float32)).to(fold.out.device).clamp(0.0, float(fold.n_knots - 1))
        l2 = t_day.floor().long().clamp(max=fold.n_knots - 2)
        f2 = t_day - l2.float()
        path = (1 - f2)[None] * knots[:, l2] + f2[None] * knots[:, l2 + 1]
    return path.cpu().numpy()


def fit_s1(*, fold: Fold, z: torch.Tensor, lam: float) -> np.ndarray:
    """Return S1's daily path (16, days): daily knots held constant, L1 total-variation penalty."""
    knots = torch.zeros(16, fold.n_days, device=fold.out.device, requires_grad=True)
    n = float(len(z))

    def loss() -> torch.Tensor:
        pen = lam * (knots[:, 1:] - knots[:, :-1]).abs().sum()
        return (nll_sum(fold=fold, z=z, offset_rows=knots[fold.primary, fold.day]) + pen) / n

    _optimise(params=[knots], loss_fn=loss, steps=STEPS)
    return knots.detach().cpu().numpy()


def fit_s2(*, fold: Fold, z: torch.Tensor, lam: float, init: np.ndarray) -> np.ndarray:
    """Return S2's daily path (16, days): sum of hard-concrete gated jumps, penalised by the expected number of open gates."""
    dev = fold.out.device
    assert init is not None, "S2 starts from S1's thresholded jumps (pure gradient start from closed gates did not open any gate)"
    jump = np.diff(init, axis=1, prepend=init[:, :1]) * fold.std[:, None]
    big = np.abs(jump) >= INIT_JUMP
    log_alpha = torch.from_numpy(np.where(big, 2.0, -3.0).astype(np.float32)).to(dev)
    amp = torch.from_numpy((np.where(big, jump, 0.0) / fold.std[:, None]).astype(np.float32)).to(dev)
    log_alpha.requires_grad_(True)
    amp.requires_grad_(True)
    torch.manual_seed(573)
    n = float(len(z))
    offset_const = HC_BETA * float(np.log(-HC_GAMMA / HC_ZETA))

    def gates(*, noisy: bool) -> torch.Tensor:
        if noisy:
            u = torch.rand_like(log_alpha).clamp(1e-4, 1 - 1e-4)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / HC_BETA)
        else:
            s = torch.sigmoid(log_alpha)
        return (s * (HC_ZETA - HC_GAMMA) + HC_GAMMA).clamp(0.0, 1.0)

    def path(*, noisy: bool) -> torch.Tensor:
        return torch.cumsum(gates(noisy=noisy) * amp, dim=1)

    def loss() -> torch.Tensor:
        pen = lam * torch.sigmoid(log_alpha - offset_const).sum()
        return (nll_sum(fold=fold, z=z, offset_rows=path(noisy=True)[fold.primary, fold.day]) + pen) / n

    _optimise(params=[log_alpha, amp], loss_fn=loss, steps=STEPS)
    with torch.no_grad():
        return path(noisy=False).cpu().numpy()


def extract(*, path: np.ndarray, std: float, per_day: float = DETECT_PER_DAY, min_step: float = MIN_STEP) -> list[tuple[float, float]]:
    """Return the steps `(day, signed size in capacity fraction)` of one Primary's daily offset path.

    Runs of consecutive days whose change is at least `per_day` of capacity in the same direction
    (a gap of up to `MERGE_GAP` days is allowed) are merged; a merged run with total size at least
    `min_step` is a step, located at the day where half of its total change has happened.
    """
    delta = np.diff(path, prepend=path[0]) * std
    steps: list[tuple[float, float]] = []
    d, n = 0, len(delta)
    while d < n:
        if abs(delta[d]) < per_day:
            d += 1
            continue
        sign = np.sign(delta[d])
        end, last = d, d
        while end + 1 < n and end + 1 - last <= MERGE_GAP:
            end += 1
            if np.sign(delta[end]) == sign and abs(delta[end]) >= per_day:
                last = end
            elif np.sign(delta[end]) == -sign and abs(delta[end]) >= per_day:
                break
        run = delta[d : last + 1]
        total = float(run[np.sign(run) == sign].sum())
        cum = np.cumsum(np.where(np.sign(run) == sign, run, 0.0))
        mid = int(np.searchsorted(np.abs(cum), abs(total) / 2.0))
        if abs(total) >= min_step:
            steps.append((float(d + mid), total))
        d = last + 1
    return steps


SUPPORT_ROWS: Final = 100
"""A step is scored only when the Primary has at least 100 cached rows in each of the 5 days before and after it (about a quarter of the typical 400)."""


def supported(*, fold: Fold, primary: int, day: int) -> bool:
    """Return whether the fold has rows on both sides of the day for the Primary."""
    r = fold.rows_per_day[primary]
    before, after = r[max(day - 5, 0) : day].sum(), r[day + 1 : day + 6].sum()
    return bool(before >= SUPPORT_ROWS and after >= SUPPORT_ROWS)


def match(*, truth: list[tuple[int, float]], found: list[tuple[float, float]]) -> tuple[int, int]:
    """Return (truth steps recovered, found steps that match a truth step): same sign, date within 3 days, size within 25%."""
    used: set[int] = set()
    hit = 0
    for day, size in truth:
        for j, (fd, fs) in enumerate(found):
            if j in used:
                continue
            if np.sign(fs) == np.sign(size) and abs(fd - day) <= TOL_DAYS and abs(fs - size) <= TOL_SIZE * abs(size):
                used.add(j)
                hit += 1
                break
    return hit, len(used)


def score_steps(*, fold: Fold, path: np.ndarray, truth: dict[int, list[tuple[int, float]]]) -> dict:
    """Score a path against a world's change points on the injected Primaries."""
    limit = fold.span_end_day - END_MARGIN
    rec_all = rec_big = n_all = n_big = n_found = n_true_found = 0
    for p in world.INJECTED:
        found = [(d, s) for d, s in extract(path=path[p], std=fold.std[p]) if d <= limit]
        tru = [(d, s) for d, s in truth.get(p, []) if d <= limit and supported(fold=fold, primary=p, day=d)]
        hit, ok = match(truth=tru, found=found)
        big = [(d, s) for d, s in tru if abs(s) >= 0.03]
        hit_big, _ = match(truth=big, found=found)
        rec_all, n_all, rec_big, n_big = rec_all + hit, n_all + len(tru), rec_big + hit_big, n_big + len(big)
        n_found, n_true_found = n_found + len(found), n_true_found + ok
    return {
        "recall_all": rec_all / max(n_all, 1), "recall_3pct": rec_big / max(n_big, 1),
        "precision": n_true_found / max(n_found, 1), "n_truth": n_all, "n_truth_3pct": n_big,
        "n_found": n_found, "n_recovered": rec_all, "n_recovered_3pct": rec_big,
    }  # fmt: skip


def score_clean(*, fold: Fold, path: np.ndarray) -> dict:
    """Count steps of 1% or more per Primary-year on the six injected and on the ten untouched Primaries (world K0).

    Each Primary's years run from its first cached day to ten days before the span ends.
    """
    limit = fold.span_end_day - END_MARGIN
    out = {}
    for name, ps in (("injected_six", world.INJECTED), ("untouched_ten", [p for p in range(16) if p not in world.INJECTED])):
        n, years = 0, 0.0
        for p in ps:
            first = int(np.argmax(fold.rows_per_day[p] > 0))
            n += len([1 for d, _ in extract(path=path[p], std=fold.std[p]) if first <= d <= limit])
            years += world.offset_years(start_day=first, end_day=limit)
        out[name] = {"steps": n, "primary_years": years, "per_primary_year": n / years}
    return out


def score_ramp(*, fold: Fold, path: np.ndarray) -> dict:
    """K-S3 per injected Primary: the recovered end value within 25% of truth, and at least 5 steps of 0.3% or more inside the ramp."""
    rows = []
    for p in world.INJECTED:
        first, last, size = world.ramp_points(primary=p)
        if last + 30 > fold.span_end_day or not (supported(fold=fold, primary=p, day=first) and supported(fold=fold, primary=p, day=last)):
            continue
        rec = float(path[p, last + 30] - path[p, max(first - 30, 0)]) * fold.std[p]
        steps = [s for d, s in extract(path=path[p], std=fold.std[p], per_day=0.0005, min_step=0.003) if first - 15 <= d <= last + 15]
        rows.append({
            "primary": p, "truth_end": size, "recovered_end": rec,
            "end_ok": abs(rec - size) <= 0.25 * abs(size), "n_steps": len(steps), "steps_ok": len(steps) >= 5,
        })  # fmt: skip
    return {"per_primary": rows, "pass_share": float(np.mean([r["end_ok"] and r["steps_ok"] for r in rows])) if rows else float("nan")}


def run_one(*, fold: Fold, world_name: str, device: str, tvs: tuple[float, ...], gates: tuple[float, ...]) -> dict:
    """Fit S0, S1 at each total-variation weight, and S2 at each gate weight started from each S1; return the scores."""
    grid = data.grid_length()
    inj = world.injection(world=world_name, grid=grid)
    z = world_z(fold=fold, inj=inj, device=device)
    truth = world.truth_steps(world=world_name)
    result: dict = {"month": fold.month, "world": world_name}

    def record(*, key: str, path: np.ndarray, started: float) -> None:
        res: dict = {"seconds": round(time.monotonic() - started, 1)}
        if world_name == "Ks":
            res.update(score_steps(fold=fold, path=path, truth=truth))
        elif world_name == "K0":
            res.update(score_clean(fold=fold, path=path))
        else:
            res.update(score_ramp(fold=fold, path=path))
        result[key] = res
        np.save(PATHS / f"{fold.month}_{world_name}_{key.replace('/', '_')}.npy", path.astype(np.float32))
        print(fold.month, world_name, key, {k: v for k, v in res.items() if k != "per_primary"}, flush=True)

    started = time.monotonic()
    record(key="S0", path=fit_s0(fold=fold, z=z), started=started)
    for tv in tvs:
        started = time.monotonic()
        p1 = fit_s1(fold=fold, z=z, lam=tv)
        record(key=f"S1@{tv:g}", path=p1, started=started)
        for gate in gates:
            started = time.monotonic()
            record(key=f"S2@{tv:g}/{gate:g}", path=fit_s2(fold=fold, z=z, lam=gate, init=p1), started=started)
    return result


def main() -> None:
    """Run the penalty sweep (first fold) or the full Step 1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("sweep", "run"))
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--worlds", nargs="+", default=list(world.WORLDS))
    parser.add_argument("--tv", type=float, nargs="+", default=list(TV_SWEEP))
    parser.add_argument("--gate", type=float, nargs="+", default=list(GATE_SWEEP))
    args = parser.parse_args()
    global PATHS  # noqa: PLW0603
    PATHS = WORK / f"paths_{args.command}_r2"
    PATHS.mkdir(parents=True, exist_ok=True)
    (WORK / f"step1_{args.command}_r2").mkdir(parents=True, exist_ok=True)
    device = "cuda"
    for month in args.months:
        fold = load_fold(month=month, device=device)
        for w in args.worlds:
            target = WORK / f"step1_{args.command}_r2" / f"{month}_{w}.json"
            if target.exists():
                continue
            res = run_one(fold=fold, world_name=w, device=device, tvs=tuple(args.tv), gates=tuple(args.gate))
            target.write_text(json.dumps(res, indent=1))
        del fold
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
