# ruff: noqa: E501
"""Experiment 721, Stage 0: the checks that run before any fit, on the CPU, and the files they write.

Stage 0 runs in four steps, in the order of the page:

1. The test of the flag on the stored curves of experiment 716's known-answer worlds: the flag fires
on
   the curves that were still falling and stays quiet on the noise world.
2. The reachability check of the synthetic world: a sky-blind rule that knows the noise-free demand
   exactly must leave an area at least 0.10 plus 3 standard errors below 1. When it fails, a 5-point
   day offset of demand noise is added and the check runs once more; a second failure is a stop.
3. The noise check of the margin: 0.10 must be at least 3 standard errors of the area estimate.
4. The record of the stage in which the sign head trains (`core.SIGN_HEAD_STAGE`, Branch A).

The module writes `synthetic_world.json` (the day offset the synthetic world uses) and `stage0.json`
(the table that the summary copies, with the stops of verdict row 0). It reads no fit, no GPU, and no
farm series, and prints no generator value.
"""

import datetime
from pathlib import Path

import e00721_core as core
import e00721_fit as fit
import e00721_real as real
import e00721_score as score
import e00721_worlds as wd
import numpy as np


def known_answer_folder(*, data_dir: Path) -> Path:
    """Return the folder of experiment 716's stored known-answer curves."""
    found = sorted((data_dir / "ml-experiments").glob("00716-*/work/known_answers"))
    if not found:
        msg = f"experiment 716's known-answer curves are not under {data_dir}/ml-experiments"
        raise FileNotFoundError(msg)
    return found[0]


def flag_test(*, data_dir: Path) -> dict:
    """Return the test of the flag: it fires on `WH` and `WT`'s still-falling curves, and not on `WN`'s."""
    import json

    folder = known_answer_folder(data_dir=data_dir)

    def flagged(*, world: str, seed: int, arm: str) -> bool:
        body = json.loads((folder / f"{world}_s{seed}.json").read_text())
        return core.stopped_while_falling(val_history=body["arms"][arm]["val_history"]).flagged

    fires = [flagged(world="WH", seed=s, arm="H") for s in (0, 1, 2)] + [
        flagged(world="WT", seed=0, arm="HT")
    ]
    quiet = [not flagged(world="WN", seed=s, arm=a) for s in (0, 2) for a in ("H", "HT")]
    return {
        "fires_on_falling": fires,
        "quiet_on_noise": quiet,
        "pass": bool(all(fires) and all(quiet)),
    }


def scored_span(*, fold: str, n_grid: int) -> tuple[int, int]:
    """Return the grid index range `[start, stop)` of the half-hours that the margin scores in a fold.

    The margin scores history tokens at or after the fold's cutoff (the first half-hour of the fold's
    month), where the fit phase held every value at NaN, up to the first half-hour of the next month.
    """
    from e00422_prep import GRID_START_US, STEP_US  # ty: ignore[unresolved-import]

    year, month = (int(part) for part in fold.split("-"))
    first = datetime.datetime(year, month, 1, tzinfo=datetime.UTC)
    nxt = datetime.datetime(year + month // 12, month % 12 + 1, 1, tzinfo=datetime.UTC)

    def index(when: datetime.datetime) -> int:
        return int((int(when.timestamp()) * 1_000_000 - GRID_START_US) // STEP_US)

    return index(first), min(index(nxt), n_grid)


def exporter_tokens(*, parts: dict, span: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Return the pooled tokens of the synthetic exporters that the margin uses (flattened arrays).

    Args:
        parts: The synthetic world's arrays (`real.synthetic_inputs`).
        span: The grid index range `[start, stop)` of the tokens to keep, or `None` for the whole grid of
            experiment 87. The checks of Stage 0 pass the span that the margin scores.

    Returns:
        The arrays `demand_mean`, `reading`, `x_true`, `u`, `day`, and `pv_capacity`, one entry per token.
    """
    syn, four, exporters = parts["syn"], list(parts["four"]), list(parts["exporters"])
    rows = [four.index(e) for e in exporters]
    n = syn.x.shape[1]
    keep = np.zeros(n, dtype=bool)
    start, stop = (0, real.N_GRID_87) if span is None else span
    keep[start : min(stop, real.N_GRID_87)] = True
    day = np.broadcast_to(np.arange(n) // wd.STEPS_PER_DAY, (len(rows), n))
    sel = lambda a: a[rows][:, keep].ravel()  # noqa: E731
    return {
        "demand_mean": sel(syn.demand_mean), "reading": sel(syn.reading), "x_true": sel(syn.x),
        "u": sel(parts["u"][exporters]), "day": sel(day),
        "pv_capacity": sel(np.broadcast_to(syn.capacity_share[:, None], (len(four), n))),
    }  # fmt: skip


def run_stage0(*, data_dir: Path, seeds: tuple[int, ...] = (0, 1)) -> dict:
    """Run the four steps and write `synthetic_world.json` and `stage0.json` under the data root."""
    root = fit.data_root(data_dir=data_dir)
    flags = flag_test(data_dir=data_dir)
    gi = real.load_grid_inputs(
        fold=core.SYNTH_PIPELINE_FOLD, seed=0, phase="train", with_farms=False
    )
    span = scored_span(fold=core.SYNTH_PIPELINE_FOLD, n_grid=real.N_GRID_87)
    offset, reach_rows = 0.0, []
    for offset in (0.0, 5.0):
        rows = []
        for seed in seeds:
            parts = real.synthetic_inputs(gi=gi, seed=seed, day_offset_points=offset)
            tok = exporter_tokens(parts=parts, span=span)
            rows.append(wd.reachability_check(**tok))
        reach_rows.append(
            {"day_offset_points": offset, "per_seed": rows, "pass": all(r["pass"] for r in rows)}
        )
        if reach_rows[-1]["pass"]:
            break
    reach_ok = bool(reach_rows[-1]["pass"])
    day_counts = []
    for seed in seeds:
        parts = real.synthetic_inputs(gi=gi, seed=seed, day_offset_points=offset)
        tok = exporter_tokens(parts=parts, span=span)
        ok = (tok["u"] >= core.U_MIN) & (np.abs(tok["x_true"]) * 100.0 >= core.DEAD_BAND_POINTS)
        rev = tok["x_true"] < 0.0
        day_counts.append(
            (len(np.unique(tok["day"][ok & rev])), len(np.unique(tok["day"][ok & ~rev])))
        )
    noise = score.noise_check(
        n_days_pos=float(min(c[0] for c in day_counts)),
        n_days_neg=float(min(c[1] for c in day_counts)),
    )
    stage0 = {
        "flag_test": flags, "reachability": reach_rows, "day_offset_points": offset, "noise_check": noise,
        "sign_head_stage": core.SIGN_HEAD_STAGE, "branch": "A",
        "stops": {
            "stage0_flag_test_failed": not flags["pass"],
            "reachability_failed_after_noise": not reach_ok,
            "noise_check_failed": not noise["pass"],
        },
    }  # fmt: skip
    fit.atomic_write_json(path=root / real.SYNTH_WORLD_FILE, body={"day_offset_points": offset})
    fit.atomic_write_json(path=root / "stage0.json", body=stage0)
    return stage0
