"""Score arm A on the four nodes against experiment 415's N1 and experiment 386's references.

This reuses experiment 415's `score_nodes` unchanged (shared half-hours, seed-averaged quantiles,
month-block bootstrap, tails) with arm `A` read from this experiment's `seed_preds` and arm `N1`
read from experiment 415's. Only the arm list and the contrasts are replaced.

```bash
. <the environment of e00475_fit.py>
uv run --no-sync python ml-experiments/00475-*/e00475_score.py
```
"""

from pathlib import Path

import e00415_prep as prep415  # ty: ignore[unresolved-import]
import e00415_score as s415  # ty: ignore[unresolved-import]
import e00475_prep as prep  # ty: ignore[unresolved-import]

ARM_DIRS: dict[str, Path] = {
    "N1": prep415.WORK_DIR / "seed_preds",
    "A": prep.WORK_DIR / "seed_preds",
}


def _seed_files(*, arm: str) -> dict[int, list[Path]]:
    """Return an arm's forecast files grouped by seed, from the arm's own folder."""
    out: dict[int, list[Path]] = {}
    for f in sorted((ARM_DIRS[arm] / arm).glob("*_s*.parquet")):
        out.setdefault(int(f.stem.rsplit("_s", 1)[1]), []).append(f)
    return out


def main() -> None:
    """Score the nodes."""
    s415.seed_files = _seed_files  # ty: ignore[invalid-assignment]
    s415.prep.WORK_DIR = prep.WORK_DIR  # write the scores here, never into experiment 415's folder
    s415.prep.ARMS = ("N1", "A")
    s415.CONTRASTS = (
        ("A", "N1"),
        ("A", "bg2"),
        ("A", "xgb_control_w"),
        ("N1", "bg2"),
        ("A", "manual_heuristic"),
        ("A", "climatology"),
    )
    s415.ARMS_TO_SCORE[:] = ["N1", "A"]
    s415.score_nodes()


if __name__ == "__main__":
    main()
