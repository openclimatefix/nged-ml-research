# ruff: noqa: E501
"""Experiment 575: fit S0, S1, and S2 on cached outputs of the frozen network, in worlds Kl and Kp, and score recovery.

Reuses experiment 573's cache, fold loader, offset fits, step extractor, and penalty grid unchanged
(`e00573_steps`, `e00573_cache`); only the worlds (`e00575_world`) and the scoring differ.

Usage: python e00575_run.py [--months ...] [--worlds ...] [--seeds ...]

K0 false-step rates are not produced here: the report recounts them from experiment 573's saved K0 paths.
"""

import argparse
import json
import time
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00573_steps as s  # ty: ignore[unresolved-import]
import e00575_world as w
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00575-stepwise-offset-large-step-and-mirrored-pair-recovery"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
SCORES_DIR: Final = WORK / "scores"
TV: Final = (3000.0, 1000.0)
GATE: Final = (300.0, 30.0)
"""The penalty grid of experiment 573's full run (`e00573_report.ARMS`), unchanged."""
BIG: Final = 0.20
TOL_BIG: Final = 1
FALSE_MIN: Final = 0.05


def tol_days(*, size: float) -> int:
    """Return the date tolerance: 1 day for a step of 20% of capacity or more, else 3."""
    return TOL_BIG if abs(size) >= BIG else s.TOL_DAYS


def match_changes(
    *, truth: list[tuple[int, float]], found: list[tuple[float, float]]
) -> list[bool]:
    """Return, for each truth change `(day, size)` in order, whether a not-yet-used found step matches it.

    A match has the same sign, a date within `tol_days`, and a size within 25%.
    """
    used: set[int] = set()
    hit = []
    for day, size in truth:
        got = False
        for j, (fd, fs) in enumerate(found):
            if (
                j not in used
                and np.sign(fs) == np.sign(size)
                and abs(fd - day) <= tol_days(size=size)
                and abs(fs - size) <= s.TOL_SIZE * abs(size)
            ):
                used.add(j)
                got = True
                break
        hit.append(got)
    return hit


def residual_sd(*, fold: s.Fold, z0: torch.Tensor) -> dict[int, float]:
    """Return per injected Primary the standard deviation (fraction of capacity) of the daily mean of the offset-free residual."""
    log_w, mean, _ = base.mixture_parts(out=fold.out, components=s.COMPONENTS)
    mix = (log_w.exp() * mean).sum(-1)
    res = ((z0 - mix).float().cpu().numpy()) * fold.std[fold.primary_np]
    out = {}
    for p in w.INJECTED:
        sel = fold.primary_np == p
        total = np.bincount(fold.day.cpu().numpy()[sel], weights=res[sel], minlength=fold.n_days)
        count = np.bincount(fold.day.cpu().numpy()[sel], minlength=fold.n_days)
        ok = count >= s.SUPPORT_ROWS // 5
        out[p] = float(np.std(total[ok] / count[ok])) if ok.any() else float("nan")
    return out


def score_world(*, fold: s.Fold, path: np.ndarray, world: str, seed: int) -> dict:
    """Score a path: one record per truth event, and false-step counts on the untouched Primaries."""
    limit = fold.span_end_day - s.END_MARGIN
    truth = w.truth_changes(world=world, seed=seed)
    changes: dict[int, list[tuple[int, float]]] = {}
    for p, evs in truth.items():
        changes[p] = []
        for e in evs:
            changes[p] += [(e.onset, e.size), (e.end, -e.size)]
    found = {
        p: [(d, sz) for d, sz in s.extract(path=path[p], std=fold.std[p]) if d <= limit]
        for p in w.INJECTED
    }
    hits = {
        p: dict(zip(changes[p], match_changes(truth=changes[p], found=found[p]), strict=True))
        for p in w.INJECTED
    }
    records = []
    donor_size = {e.pair_id: e.size for evs in truth.values() for e in evs if e.role == "donor"}
    for p, evs in truth.items():
        for e in evs:
            scored = (
                e.end <= limit
                and s.supported(fold=fold, primary=p, day=e.onset)
                and s.supported(fold=fold, primary=p, day=e.end)
            )
            records.append({
                "primary": p, "pair_id": e.pair_id, "role": e.role, "onset": e.onset, "end": e.end,
                "duration": e.end - e.onset, "size": e.size, "donor_size": donor_size.get(e.pair_id, e.size), "scored": scored,
                "hit_onset": hits[p][(e.onset, e.size)], "hit_return": hits[p][(e.end, -e.size)],
            })  # fmt: skip
    clean_ps = [p for p in range(16) if p not in w.INJECTED]
    ten_years = ten_big = ten_small = 0.0
    for p in clean_ps:
        first = int(np.argmax(fold.rows_per_day[p] > 0))
        steps = [
            (d, sz)
            for d, sz in s.extract(path=path[p], std=fold.std[p], min_step=0.01)
            if first <= d <= limit
        ]
        ten_small += len(steps)
        ten_big += sum(abs(sz) >= FALSE_MIN for _, sz in steps)
        ten_years += w.w573.offset_years(start_day=first, end_day=limit)
    return {
        "events": records,
        "untouched_ten": {"years": ten_years, "steps_1pct": ten_small, "steps_5pct": ten_big},
    }


def run_arms(*, fold: s.Fold, z: torch.Tensor, world: str, seed: int, paths_dir) -> dict:  # noqa: ANN001
    """Fit every arm of the grid and score it."""
    out: dict = {}

    def record(*, key: str, path: np.ndarray, started: float) -> None:
        res = score_world(fold=fold, path=path, world=world, seed=seed)
        res["seconds"] = round(time.monotonic() - started, 1)
        out[key] = res
        np.save(
            paths_dir / f"{fold.month}_{world}{seed}_{key.replace('/', '_')}.npy",
            path.astype(np.float32),
        )
        sc = [e for e in res["events"] if e["scored"]]
        print(
            fold.month,
            world,
            seed,
            key,
            "scored",
            len(sc),
            "onset hits",
            sum(e["hit_onset"] for e in sc),
            "untouched 5%",
            res["untouched_ten"]["steps_5pct"],
            flush=True,
        )

    started = time.monotonic()
    record(key="S0", path=s.fit_s0(fold=fold, z=z), started=started)
    for tv in TV:
        started = time.monotonic()
        p1 = s.fit_s1(fold=fold, z=z, lam=tv)
        record(key=f"S1@{tv:g}", path=p1, started=started)
        for gate in GATE:
            started = time.monotonic()
            record(
                key=f"S2@{tv:g}/{gate:g}",
                path=s.fit_s2(fold=fold, z=z, lam=gate, init=p1),
                started=started,
            )
    return out


def main() -> None:
    """Fit and score the worlds, one JSON per fold, world, and seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=list(s.MONTHS))
    parser.add_argument("--worlds", nargs="+", default=["Kl", "Kp"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(w.SEEDS)))
    args = parser.parse_args()
    paths_dir = WORK / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    (WORK / "scores").mkdir(parents=True, exist_ok=True)
    grid = data.grid_length()
    for month in args.months:
        fold = s.load_fold(month=month, device="cuda")
        z0 = s.world_z(fold=fold, inj=np.zeros((16, grid), dtype=np.float32), device="cuda")
        sd_file = WORK / "scores" / f"{month}_residual_sd.json"
        if not sd_file.exists():
            sd_file.write_text(json.dumps(residual_sd(fold=fold, z0=z0)))
        for world in args.worlds:
            for seed in args.seeds:
                target = WORK / "scores" / f"{month}_{world}{seed}.json"
                if target.exists():
                    continue
                inj = w.injection(world=world, grid=grid, seed=seed)
                z = s.world_z(fold=fold, inj=inj, device="cuda")
                res = run_arms(fold=fold, z=z, world=world, seed=seed, paths_dir=paths_dir)
                tmp = target.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                )
                tmp.rename(target)
        del fold
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
