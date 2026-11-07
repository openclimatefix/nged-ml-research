# ruff: noqa: E501, ANN201
"""Experiment 620: freeze the three demand networks before the final-test window is opened.

Writes `freeze_config.json` (artefact paths and SHA-256 hashes, the code files' hashes, the frozen
gate threshold, neighbour list and candidate width factor, and the refit schedule), and runs the
final-test runner's timing assertions (`e00614_final_runner.origin_rows`: weather run = origin
minus 9 h, the origin stamped on every row, every power lag at least 24 h old) on the extended
inputs for all 123 origins from 2026-07-01 to 2026-10-31. Nothing is predicted and no forecast of the
window exists at this point. `e00620_predict.py` refuses to start unless `freeze_config.json`
verifies and the lock file does not exist.

Usage: python e00620_freeze.py [--verify]
"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00614_final_runner as runner  # ty: ignore[unresolved-import]
import e00620_gate as gate  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HERE = Path(__file__).parent
NAME = "00620-bakeoff-final-test-demand-networks-at-primaries"
R = ML_EXPERIMENTS_DATA_DIR / NAME
E607 = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
S613 = next(ML_EXPERIMENTS_DATA_DIR.glob("00613-*"))
CONFIG = HERE / "freeze_config.json"
LOCK = HERE / "final_test_620.lock"
SEEDS = (0, 1, 2, 3, 4)
CODE = (
    "e00620_fold.py", "e00620_extend.py", "e00620_k1.py", "e00620_k3.py", "e00620_cand.py",
    "e00620_gate.py", "e00620_tables.py", "e00620_freeze.py", "e00620_assemble.py", "predict.sh", "gpu.sh",
)  # fmt: skip


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artefacts() -> dict[str, dict[str, Path]]:
    """Return, per network, the files whose bytes define it."""
    s = [f"_s{k}" for k in SEEDS]
    return {
        "K1": {
            **{f"pretrained_P2/2026-07{x}.pt": R / f"chains/k1/work/pretrained_P2/2026-07{x}.pt" for x in s},
            **{f"step2/P5/2026-07{x}.pt": R / f"chains/k1/work/step2/P5/2026-07{x}.pt" for x in s},
            "stats/B1/2026-07.npz": R / "chains/k1/work/stats/B1/2026-07.npz",
            **{f"models/P5/2026-07_d0{x}.pt": R / f"arms/k1/work/models/P5/2026-07_d0{x}.pt" for x in s},
        },
        "Candidate": {
            **{f"tables/T_2026-07{x}.npz": R / f"tables/T_2026-07{x}.npz" for x in s},
            **{f"models/P5/2026-07_d0{x}.pt": R / f"P7_D50_e3/work/models/P5/2026-07_d0{x}.pt" for x in s},
        },
        "K3": {
            **{f"pretrained_D5/real/2026-07{x}.pt": R / f"chains/k3/work/pretrained_D5/real/2026-07{x}.pt" for x in s},
            **{f"models/D5/2026-07_d0{x}.pt": R / f"arms/k3/work/models/D5/2026-07_d0{x}.pt" for x in s},
        },
        "Inputs": {
            **{f"raw/summary/{p.name}": p for p in sorted((R / "inputs/raw/summary").glob("*.npz"))},
            "cams_demand_points_extended.parquet": R / "inputs/cams_demand_points_extended.parquet",
            **{f"trailing437/{p.name}": p for p in sorted((R / "inputs/trailing437").glob("*.parquet"))},
            **{f"w560/{p.name}": p for p in sorted((R / "inputs/w560").rglob("*.npy"))},
            **{f"frames/{p.name}": p for p in sorted((R / "inputs/frames").glob("*.parquet"))},
            "observed.parquet": E607 / "observed.parquet",
            "era5_features.parquet": E607 / "era5_features.parquet",
            **{p.name: p for p in sorted((E607 / "ens_summary").glob("init=2026-0[7-9]-*.parquet"))},
            **{p.name: p for p in sorted((E607 / "ens_summary").glob("init=2026-10-*.parquet"))},
        },
    }  # fmt: skip


def timing_assertions() -> int:
    """Run the runner's per-origin timing rules on the extended inputs; return the number of origins."""
    observed = pl.read_parquet(E607 / "observed.parquet", columns=["time_series_id", "valid_time", "power"])
    n = 0
    for k in range((runner.FINAL_LAST - runner.FINAL_FIRST).days + 1):
        origin = runner.FINAL_FIRST + timedelta(days=k) + runner.PUBLICATION_DELAY
        runner.origin_rows(origin=origin, ens_dir=E607 / "ens_summary", observed=observed)
        n += 1
    return n


def upstream_digest() -> str:
    """Return one hash over every Python file of every experiment folder, which the networks import."""
    h = hashlib.sha256()
    for p in sorted(HERE.parent.glob("*/*.py")):
        h.update(p.relative_to(HERE.parent).as_posix().encode() + sha(p).encode())
    return h.hexdigest()


def build() -> dict:
    if LOCK.exists():
        raise SystemExit("the lock exists: the test window has been opened")
    widths = pl.read_csv(S613 / "score" / "c2_width_factors.csv")
    n_origins = timing_assertions()
    assert n_origins == 123, n_origins
    return {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip(),
        "train_data_ends_before": "2026-07-01",
        "test_origins": ["2026-07-01", "2026-10-31"],
        "n_origins_timing_checked": n_origins,
        "seeds": list(SEEDS),
        "refit_schedule": "one fit at the cutoff 2026-07-01 (day 0), no refits during the window; the extended inputs are read only to build features for origins from 2026-07-01",
        "g4_threshold_share_of_capacity": gate.frozen_threshold().tolist(),
        "g4_threshold_source": "experiment 592 gate_state.npz (dev months only)",
        "g4_neighbours": {str(k): v for k, v in gate.neighbours().items()},
        "g4_hold_days_after_release": gate.c.HOLD_DAYS,
        "candidate_width_factor": float(widths["factor"].mean()),
        "candidate_width_factor_source": "mean of experiment 613's five dev-month factors (c2_width_factors.csv)",
        "candidate_embargo": "E3 is the fixed refit rule (7-day embargo, then half weight for 7 days) applied at the single fit; no embargo trigger exists, so the candidate is scored with E3 only",
        "artefacts": {net: {rel: sha(p) for rel, p in files.items()} for net, files in artefacts().items()},
        "upstream_code_digest": upstream_digest(),
        "code": {name: sha(HERE / name) for name in CODE if (HERE / name).exists()},
    }


def verify() -> None:
    cfg = json.loads(CONFIG.read_text())
    for net, files in artefacts().items():
        got = {rel: sha(p) for rel, p in files.items()}
        assert got == cfg["artefacts"][net], f"{net}: artefacts differ from the freeze config"
    for name, h in cfg["code"].items():
        assert sha(HERE / name) == h, f"{name} changed since the freeze"
    assert not LOCK.exists(), "the lock exists"
    live = json.loads(json.dumps({
        "g4_threshold_share_of_capacity": gate.frozen_threshold().tolist(),
        "g4_neighbours": {str(k): v for k, v in gate.neighbours().items()},
        "g4_hold_days_after_release": gate.c.HOLD_DAYS,
        "candidate_width_factor": float(pl.read_csv(S613 / "score" / "c2_width_factors.csv")["factor"].mean()),
    }))
    for k, v in live.items():
        assert cfg[k] == v, f"{k} differs from the freeze config"
    assert upstream_digest() == cfg["upstream_code_digest"], "an experiment module changed since the freeze"
    print("freeze config verified:", {k: len(v) for k, v in cfg["artefacts"].items()})


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        CONFIG.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", CONFIG)
