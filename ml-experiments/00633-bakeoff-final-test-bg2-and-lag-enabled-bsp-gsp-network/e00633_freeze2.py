# ruff: noqa: D103, E501
"""Experiment 633, Part 2: freeze the lag-enabled network at the BSP/GSP series before the window is opened.

Writes `freeze_part2.json`: the SHA-256 hash of every pre-trained, step-2, and fine-tuned network
weight file of the final fold (2026-07, seeds 0 and 1, groups bsp and gsp), the standardiser files,
the node inputs (123 ensemble summaries, the raw matrices, the thermal-lag tables, the demand
frames), the observed-power and ERA5 files, the code files, and the Part 1 forecasts' absence. The
dev-check (experiment 622's stored 2026-06 weights reproduce its stored forecast exactly on the
extended inputs) is recorded in the page, not here. `predict_part2.sh` refuses to start unless
`--verify` passes and the lock file does not exist.

Usage: python e00633_freeze2.py [--verify]
"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import e00633_common as k

HERE = Path(__file__).parent
CONFIG = HERE / "freeze_part2.json"
LOCK = HERE / "final_test_633_part2.lock"
GROUPS = ("bsp", "gsp")
TAG = "J_LG_B1"
CODE = (
    "e00633_common.py", "e00633_fold.py", "e00633_nodes.py", "e00633_tables.py", "e00633_node_fit.py",
    "e00633_score_nodes.py", "e00633_freeze2.py", "predict_part2.sh", "gpu.sh",
)  # fmt: skip


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artefacts() -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for g in GROUPS:
        w = k.ROOT / "work" / g
        out[f"weights_{g}"] = {
            p.relative_to(w).as_posix(): p
            for pat in (
                f"pretrained_{TAG}/2026-07_s?.pt",
                f"step2/{TAG}/2026-07_s?.pt",
                f"models/{TAG}/2026-07_d0_s?.pt",
                "stats/B1/2026-07.npz",
            )
            for p in sorted(w.glob(pat))
        }
    for name, sub in (
        ("ens_summary_nodes", "inputs/ens_summary_nodes"),
        ("raw", "inputs/raw"),
        ("trailing437_nodes", "inputs/trailing437_nodes"),
        ("frames", "frames"),
    ):
        root = k.ROOT / sub
        out[name] = {
            p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file()
        }
    out["inputs"] = {
        p.name: p for p in (k.E607 / "observed.parquet", k.E607 / "era5_features.parquet")
    }
    return out


def upstream_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(HERE.parent.glob("*/*.py")):
        h.update(p.relative_to(HERE.parent).as_posix().encode() + sha(p).encode())
    return h.hexdigest()


def build() -> dict:
    assert not LOCK.exists(), "the lock exists: the Part 2 window has been opened"
    arts = artefacts()
    assert all(len(arts[f"weights_{g}"]) == 7 for g in GROUPS), {
        g: len(arts[f"weights_{g}"]) for g in GROUPS
    }
    for g in GROUPS:  # no window forecast may exist yet
        assert not list((k.ROOT / "work" / g / "fits" / TAG).glob("2026-07_d0_s?.parquet")), g
    return {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "train_data_ends_before": "2026-07-01",
        "test_origins": ["2026-07-01", "2026-10-31"],
        "arm": "experiment 622's lagged_b1 (D5 offset, lag-enabled, thermal-lag columns), pooled per group of 2 series, seeds 0 and 1, one fit at 2026-07-01, power lags at least 24 h old",
        "comparators": ["BG2-single-fit (e00633_bg2.py, Part 1 recipe)", "climatology"],
        "artefacts": {n: {r: sha(p) for r, p in f.items()} for n, f in arts.items()},
        "upstream_code_digest": upstream_digest(),
        "code": {n: sha(HERE / n) for n in CODE if (HERE / n).exists()},
    }


def verify() -> None:
    cfg = json.loads(CONFIG.read_text())
    for name, files in artefacts().items():
        got = {rel: sha(p) for rel, p in files.items()}
        assert got == cfg["artefacts"][name], f"{name}: artefacts differ from the freeze config"
    for name, h in cfg["code"].items():
        assert sha(HERE / name) == h, f"{name} changed since the freeze"
    assert not LOCK.exists(), "the lock exists"
    assert upstream_digest() == cfg["upstream_code_digest"], "an experiment module changed"
    print("freeze config verified:", {n: len(v) for n, v in cfg["artefacts"].items()})


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        CONFIG.write_text(json.dumps(build(), indent=1) + "\n")
        print("wrote", CONFIG)
