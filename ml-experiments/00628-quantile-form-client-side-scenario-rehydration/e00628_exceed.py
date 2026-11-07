"""Experiment 628: exceedance of actuals under the client's analytic quantile function.

The empirical quantile of 1,000 draws is noisy, and where a family has a point mass at zero a
noisy positive quantile turns every actual of zero into a hit, which the stored knots do not do.
This script scores the quantile function itself instead: the quantile of each tail rule at each
level, computed by `client_rehydrate.invert`, against the actual. At the four knot levels all rules
equal the stored quantiles by construction; the informative levels are those past the knots. Writes
`work/exceed_analytic.parquet`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00628-*/env.sh
uv run --no-sync python ml-experiments/00628-*/e00628_exceed.py
```
"""

from datetime import UTC, datetime
from pathlib import Path

import client_rehydrate as client  # ty: ignore[unresolved-import]
import e00436_score as score  # ty: ignore[unresolved-import]
import e00628_score as s628  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

LEVELS = (0.001, 0.002, 0.005, 0.01, 0.05, 0.95, 0.99, 0.995, 0.998, 0.999)


def main() -> None:
    """Write the analytic exceedance table."""
    rows = []
    for file in sorted((score.WORK / "runs").glob("*.npz")):
        run = score.load_run(path=file)
        init = datetime.fromisoformat(str(run["init"]) + "+00:00").astimezone(UTC)
        actual = s628.masked_actual(run=run)
        q11 = np.load(s628.OUT / "q11" / f"{init:%Y-%m-%d}.npy").astype(np.float64)
        usable = (q11[:, :, 10] - q11[:, :, 0]) > 1e-6
        for tail in s628.TAILS:
            quant = {}
            for level in LEVELS:
                u = np.full((1, s628.S), level)
                quant[level] = np.stack(
                    [
                        client.invert(
                            q=q11[n],
                            u=u,
                            tail=tail,
                            bounds=None if np.isnan(s628.BOUNDS[n, 0]) else tuple(s628.BOUNDS[n]),
                        )[0]
                        for n in range(s628.N)
                    ]
                )
            rows.extend(
                s628.exceed_rows(
                    label=f"Q11-{tail}", quant=quant, actual=actual, usable=usable, init=init,
                    extra={"coupling": "all", "tail": tail},
                )
            )  # fmt: skip
    pl.DataFrame(rows).write_parquet(Path(s628.OUT) / "exceed_analytic.parquet")


if __name__ == "__main__":
    main()
