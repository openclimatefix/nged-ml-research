"""Stage 0: the stopped-while-falling flag on the stored curves of experiments 713 and 716
(read-only).

The tests skip when the private data folder is absent. They print no loss value that belongs to a
generator
series: a validation curve is a loss of the demand network at Primary substations.
"""

import json
import os
from pathlib import Path

import e00721_core as core
import numpy as np
import pytest

DATA = Path(os.environ.get("DATA_PATH_INTERNAL", "data")) / "ml-experiments"


def _known_answers() -> Path:
    found = sorted(DATA.glob("00716-*/work/known_answers"))
    if not found:
        pytest.skip("the stored curves of experiment 716 are not on this machine")
    return found[0]


def _curve(*, world: str, seed: int, arm: str) -> list[float]:
    body = json.loads((_known_answers() / f"{world}_s{seed}.json").read_text())
    return body["arms"][arm]["val_history"]


@pytest.mark.parametrize(
    ("world", "seed", "arm"), [("WH", 0, "H"), ("WH", 1, "H"), ("WH", 2, "H"), ("WT", 0, "HT")]
)
def test_flag_fires_on_the_known_still_falling_curves(world: str, seed: int, arm: str) -> None:
    assert core.stopped_while_falling(val_history=_curve(world=world, seed=seed, arm=arm)).flagged


@pytest.mark.parametrize(("seed", "arm"), [(0, "H"), (0, "HT"), (2, "H"), (2, "HT")])
def test_flag_is_quiet_on_the_noise_world(seed: int, arm: str) -> None:
    assert not core.stopped_while_falling(
        val_history=_curve(world="WN", seed=seed, arm=arm)
    ).flagged


def test_every_stored_curve_is_printed_with_its_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rows = []
    for path in sorted(_known_answers().glob("W[HNT]_s[0-2].json")):
        for arm, body in json.loads(path.read_text())["arms"].items():
            if arm == "Z":
                continue
            flag = core.stopped_while_falling(val_history=body["val_history"])
            rows.append((path.stem, arm, flag.flagged, round(flag.gap / max(flag.noise, 1e-12), 2)))
    with capsys.disabled():
        print("\nstored curves, flag, gap in noise units:")
        for row in rows:
            print("  ", *row)
    assert len(rows) == 18


def _plateau_flag_rate(*, curves: list[list[float]], n_draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    flags = []
    for h in curves:
        stop8, _ = core.replay_stop(val_history=h, patience=core.REAL_FOLD_PATIENCE)
        best = int(np.argmin(h[:stop8]))
        post = np.asarray(h[best + 1 : stop8], dtype=np.float64)
        if post.size < 3:
            continue
        mean, sd = float(post.mean()), float(post.std(ddof=1))
        for _ in range(n_draws):
            tail = rng.normal(mean, sd, core.CONTINUATION_CAP_PASSES)
            flags.append(core.stopped_while_falling(val_history=[*h[:stop8], *tail]).flagged)
    return float(np.mean(flags)) if flags else float("nan")


def test_false_flag_rate_on_flat_plateaus_of_713s_real_fold_curves() -> None:
    """Continue each stored stage-1 curve of 713 as a flat plateau at its post-best mean and
    spread."""
    curves = []
    for f in sorted(DATA.glob("00713-*/work/logs/*/*_fit_stage1.json")):
        h = (json.loads(f.read_text()).get("first") or {}).get("val_history")
        if h and len(h) > core.REAL_FOLD_PATIENCE + 3:
            curves.append([float(x) for x in h])
    if len(curves) < 10:
        pytest.skip("the stored real-fold curves of experiment 713 are not on this machine")
    assert _plateau_flag_rate(curves=curves, n_draws=20, seed=0) <= core.PLATEAU_FALSE_FLAG_CEILING
