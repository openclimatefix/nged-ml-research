"""Experiment 701, GAM side: build the sandbox data folder that the GAM stages write to.

    PYTHONPATH=ml-experiments/00701-* DATA_PATH_INTERNAL=<real data folder> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_sandbox.py

`<ROOT>/gam/sandbox` is a data folder of its own (set `DATA_PATH_INTERNAL` to it to run a GAM
stage). It holds symlinks to every top-level entry of the real data folder except
`ml-experiments`, and inside `ml-experiments`:

- **copies** of the folders of experiments 377 and 378 taken from experiment 700's sandbox, so the
  stored forecasts, fits, and member files of origin months to 2026-06 are the ones experiment 700
  scored (its re-run of 2025-11 to 2026-06 at 678 half-hours included). The stages here append the
  months 2026-07 to 2026-10 and write nowhere else;
- **real folders** `00111-*/work` and `00152-*/work` whose entries are symlinks to the stored
  entries, except `ens_store`, an empty real folder that `e00701_gam_ens.py build` fills with the
  stored arrays plus the 123 new runs;
- symlinks to every other experiment folder, read only.

Nothing is written into experiment 700's sandbox or into any stored folder.
"""

import shutil
from pathlib import Path

import e00701_gam_extend as ext

E700_SANDBOX: Path = (
    ext.REAL_DATA
    / "ml-experiments"
    / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
    / "sandbox"
    / "ml-experiments"
)
COPIED = (
    "00377-demand-v1-first-run-feasibility",
    "00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1",
)
SPLIT = (
    "00111-differentiable-physics-6-ens-forecast-engine-against-bg2",
    "00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries",
)


def main() -> None:
    """Create the sandbox, refusing to touch an existing one."""
    assert not ext.SANDBOX.exists(), f"{ext.SANDBOX} exists; delete it by hand to rebuild"
    real_ml = ext.REAL_DATA / "ml-experiments"
    ext.SANDBOX_ML.mkdir(parents=True)
    for entry in sorted(ext.REAL_DATA.iterdir()):
        if entry.name != "ml-experiments":
            (ext.SANDBOX / entry.name).symlink_to(entry)
    for entry in sorted(real_ml.iterdir()):
        target = ext.SANDBOX_ML / entry.name
        if entry.name in COPIED or entry.name in SPLIT or entry.name.startswith("00701-"):
            continue
        target.symlink_to(entry)
    for name in COPIED:
        shutil.copytree(E700_SANDBOX / name, ext.SANDBOX_ML / name, symlinks=True)
        print("copied", name)
    for name in SPLIT:
        work = ext.SANDBOX_ML / name / "work"
        work.mkdir(parents=True)
        for entry in sorted((real_ml / name).iterdir()):
            if entry.name != "work":
                (ext.SANDBOX_ML / name / entry.name).symlink_to(entry)
        for entry in sorted((real_ml / name / "work").iterdir()):
            if entry.name != "ens_store":
                (work / entry.name).symlink_to(entry)
        (work / "ens_store").mkdir()
        print("split", name)
    print("sandbox ready:", ext.SANDBOX)


if __name__ == "__main__":
    main()
