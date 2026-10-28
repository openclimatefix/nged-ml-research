"""Re-run experiment 152's engine for the 8 MVA-only Primaries into this experiment's folder.

The engine of experiment 152 is imported unchanged. This runner redirects its output paths to this
experiment's work folder (forecasts, fits, ensemble store, and calibration), so no file under
experiment 152's folder changes. The ensemble store and the calibration are copied from experiment
152's folder first, which saves re-reading the raw ensemble files. The tuning, replicate, and scored
periods are all written. Each site's frame is written atomically by the engine.

    PYTHONPATH=ml-experiments/00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00374-*/e00374_rerun.py --workers 4
"""

import argparse
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Final

import e00152_common as common  # ty: ignore[unresolved-import]
import e00152_engine as engine  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-stage-two-at-eight-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00374-{SLUG}" / "work"
DONORS: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21", "D22")
ARMS: Final[dict[str, str]] = {"N": "naive", "A": "meter", "F_fold": "meter"}
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def redirect() -> None:
    """Point the engine's writable folders at this experiment's work folder."""
    common.OWN_STORE_DIR = WORK / "ens_store"
    common.OWN_CALIBRATION_DIR = WORK / "calibration"
    engine.FITS_DIR = WORK / "fits"
    engine.PREDS_DIR = WORK / "preds"


def copy_inputs() -> None:
    """Copy experiment 152's ensemble store and calibration, if not already copied."""
    for name in ("ens_store", "calibration"):
        source = common.WORK_DIR / name
        target = WORK / name
        if source.exists() and not target.exists():
            shutil.copytree(source, target)


def one_site(site: str) -> tuple[str, dict[str, int]]:
    """Run one site and return its row counts per period."""
    redirect()
    os.environ["OMP_NUM_THREADS"] = "2"
    cache = common.load_cache()
    return site, engine.run_site(
        site=site,
        variant="real",
        cache=cache,
        kinds=("naive", "meter"),
        arms=ARMS,
        donors=DONORS,
        periods=PERIODS,
    )


def main() -> None:
    """Run the sites in a spawned pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="*", default=list(common.MVA_ONLY_PRIMARIES))
    arguments = parser.parse_args()
    copy_inputs()
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers) as pool:
        for site, counts in pool.imap_unordered(one_site, arguments.sites):
            print("done", site, counts, flush=True)


if __name__ == "__main__":
    main()
