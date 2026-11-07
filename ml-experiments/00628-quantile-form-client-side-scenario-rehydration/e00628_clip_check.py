"""Experiment 628: how often solar actuals and stored q95 values exceed 100% of capacity.

Backs the `data-processing.md` entry on not clipping a solar farm's forecast at 100%. Daylight rows
are those whose stored median is above zero. Prints the share of daylight rows whose stored q95
exceeds 100, the share of scored actuals above 100, and the q95 exceedance with and without the
clip at 100, over the 6 solar farms and the 44 runs.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00628-*/env.sh
uv run --no-sync python ml-experiments/00628-*/e00628_clip_check.py
```
"""

from datetime import UTC, datetime

import e00436_score as score  # ty: ignore[unresolved-import]
import e00628_score as s628  # ty: ignore[unresolved-import]
import numpy as np


def main() -> None:
    """Print the three shares."""
    solar = np.flatnonzero(s628.FAMILIES["solar"] == s628.KIND)
    day = q95_over = actual_over = scored = hit_open = hit_clip = 0
    for file in sorted((score.WORK / "runs").glob("*.npz")):
        run = score.load_run(path=file)
        init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
        actual = s628.masked_actual(run=run)[solar]
        q11 = np.load(s628.OUT / "q11" / f"{init:%Y-%m-%d}.npy").astype(np.float64)[solar]
        daylight = q11[:, :, 5] > 0
        q95 = q11[:, :, 9]
        ok = ~np.isnan(actual) & daylight
        day += int(daylight.sum())
        q95_over += int((daylight & (q95 > 100)).sum())
        scored += int(ok.sum())
        actual_over += int((ok & (actual > 100)).sum())
        hit_open += int((ok & (actual > q95)).sum())
        hit_clip += int((ok & (actual > np.minimum(q95, 100))).sum())
    print(f"daylight rows with stored q95 > 100: {100 * q95_over / day:.1f}%")
    print(f"scored daylight actuals > 100: {100 * actual_over / scored:.1f}%")
    print(f"q95 exceedance, unclipped {100 * hit_open / scored:.1f}%")
    print(f"q95 exceedance, clipped at 100 {100 * hit_clip / scored:.1f}%")


if __name__ == "__main__":
    main()
