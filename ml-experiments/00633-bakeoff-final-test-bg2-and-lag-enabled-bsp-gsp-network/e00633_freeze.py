# ruff: noqa: D103, E501
"""Experiment 633: freeze the BG2 arms before the final-test window is opened.

Writes `freeze_part1.json`: the SHA-256 hash of every booster, the training-row record of each,
the hash of the 123 weather frames, the code files' hashes, the arms' recipes, and the result of the
timing assertions below. Nothing of the window is forecast or scored at this point.

Timing assertions, run on all 123 origins from 2026-07-01 to 2026-10-31 without reading any power:

- every frame's weather run is the 00:00 UTC run of the origin's day, and the origin is that run plus 9 hours;
- every valid time is after the origin and at most 360 hours after it;
- for every origin and every valid time, the three lag indices are at most the newest usable half-hour
  (the origin minus 3.5 hours), computed with experiment 87's `origin_lags` on a dummy power array;
- no booster's training rows extend past 2026-06-30 20:30 (valid time) or the run of 2026-06-29 09:00 (origin).

`e00633_bg2.py lookahead` adds the perturbation test on synthetic power. `predict.sh` refuses to start
unless `--verify` passes and the lock file does not exist.

Usage: python e00633_freeze.py [--verify]
"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00633_common as k
import numpy as np
import polars as pl

HERE = Path(__file__).parent
CONFIG = HERE / "freeze_part1.json"
LOCK = HERE / "final_test_633_part1.lock"
CODE = ("e00633_common.py", "e00633_frames.py", "e00633_bg2.py", "e00633_freeze.py", "e00633_score.py", "predict_part1.sh", "gpu.sh")
ARMS = ("bg2_single_fit",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artefacts() -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for arm in ARMS:
        folder = k.BOOSTERS / arm
        out[arm] = {p.name: p for p in sorted(folder.glob("*.ubj"))} | {p.name: p for p in sorted(folder.glob("*.json"))}
    out["frames"] = {p.name: p for p in sorted(k.FRAMES.glob("init=*.parquet"))}
    return out


def timing_assertions() -> int:
    k.redirect()
    n = 0
    for run in k.runs():
        origin = run + common.PUBLICATION_DELAY
        frame = pl.read_parquet(k.FRAMES / f"init={run:%Y-%m-%d}.parquet", columns=["nwp_init_time", "power_fcst_init_time", "valid_time", "lead_hours"])
        assert frame["nwp_init_time"].unique().to_list() == [run], run
        assert frame["power_fcst_init_time"].unique().to_list() == [origin], run
        assert (frame["valid_time"] > origin).all() and frame["lead_hours"].max() <= 360.0, run
        target = np.asarray(common.grid_index(times=frame["valid_time"].unique().sort()))
        o = np.full(target.shape, int(common.grid_index(times=origin)))
        common.origin_lags(power=np.zeros(int(common.grid_index(times=k.GRID_END)) + 1, np.float32), target=target, origin=o)  # raises on a lag younger than origin - 3.5 h
        n += 1
    for arm in ARMS:
        for p in sorted((k.BOOSTERS / arm).glob("*.json")):
            rec = json.loads(p.read_text())
            assert datetime.fromisoformat(rec["last_valid_time"]) <= k.CUTOFF - timedelta(hours=3.5), rec
            assert datetime.fromisoformat(rec["last_run_origin"]) <= k.CUTOFF, rec
    return n


def upstream_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(HERE.parent.glob("*/*.py")):
        h.update(p.relative_to(HERE.parent).as_posix().encode() + sha(p).encode())
    return h.hexdigest()


def build() -> dict:
    if LOCK.exists():
        raise SystemExit("the lock exists: the test window has been opened")
    n = timing_assertions()
    assert n == 123, n
    return {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip(),
        "train_data_ends_before": "2026-07-01",
        "test_origins": ["2026-07-01", "2026-10-31"],
        "n_origins_timing_checked": n,
        "arms": {
            "bg2_single_fit": "experiment 87's configuration n1h (weekly XGBoost member of BG2), one refit at 2026-07-01 00:00 UTC, seed 0, 500 rounds, eta 0.05, half-life 0.5 years, cleaned power, D15 and D18 cut before 2025-03-01, 51-member mixture, spread factor 1.15 about the median, lags from the newest usable half-hour (origin minus 3.5 h)",
        },
        "artefacts": {name: {rel: sha(p) for rel, p in files.items()} for name, files in artefacts().items()},
        "upstream_code_digest": upstream_digest(),
        "code": {name: sha(HERE / name) for name in CODE if (HERE / name).exists()},
    }


def verify() -> None:
    cfg = json.loads(CONFIG.read_text())
    for name, files in artefacts().items():
        got = {rel: sha(p) for rel, p in files.items()}
        assert got == cfg["artefacts"][name], f"{name}: artefacts differ from the freeze config"
    for name, h in cfg["code"].items():
        assert sha(HERE / name) == h, f"{name} changed since the freeze"
    assert not LOCK.exists(), "the lock exists"
    assert upstream_digest() == cfg["upstream_code_digest"], "an experiment module changed since the freeze"
    print("freeze config verified:", {name: len(v) for name, v in cfg["artefacts"].items()})


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        CONFIG.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", CONFIG)
