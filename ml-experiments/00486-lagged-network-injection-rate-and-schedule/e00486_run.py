"""Run experiment 404's pooled contrasts and encroachment sweep on experiment 486's arms.

This script redirects experiment 404's pooling and sweep code to experiment 486's folders (see
`e00486_score.point_at_486`) and then calls its entry points unchanged. Output goes to this
experiment's `work/` folder.

```bash
PYTHONPATH=<as e00486_fit.py shows> \
    uv run --no-sync python ml-experiments/00486-*/e00486_run.py pool \
    --arms R05 R50 Cur-down Cur-up A0-inj --reference B-mix
uv run --no-sync python ml-experiments/00486-*/e00486_run.py sweep --month 2026-06 --seed 0 \
    --arms A0-inj R05 R50 Cur-down Cur-up
```
"""

import sys

import e00404_pool as pool404  # ty: ignore[unresolved-import]
import e00404_sweep as sweep404  # ty: ignore[unresolved-import]
import e00486_score as score486  # ty: ignore[unresolved-import]


def main() -> None:
    """Dispatch to experiment 404's `pool` or `sweep` entry point after redirecting its folders."""
    command = sys.argv.pop(1)
    if command == "sweep":
        sys.argv.insert(1, "run")
    score486.point_at_486()
    {"pool": pool404.main, "sweep": sweep404.main}[command]()


if __name__ == "__main__":
    main()
