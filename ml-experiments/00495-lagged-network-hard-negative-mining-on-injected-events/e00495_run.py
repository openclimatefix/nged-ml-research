"""Run experiment 404's pooled contrasts and encroachment sweep on experiment 495's arms.

This script redirects experiment 404's pooling and sweep code to experiment 495's folders (see
`e00495_score.point_at_495`) and then calls its entry points unchanged. Output goes to this
experiment's `work/` folder.

```bash
PYTHONPATH=<as e00495_fit.py shows> \
    uv run --no-sync python ml-experiments/00495-*/e00495_run.py pool \
    --arms HN A0-inj --reference B-mix
uv run --no-sync python ml-experiments/00495-*/e00495_run.py sweep --month 2026-06 --seed 0 \
    --arms A0-inj HN
```
"""

import sys

import e00404_pool as pool404  # ty: ignore[unresolved-import]
import e00404_sweep as sweep404  # ty: ignore[unresolved-import]
import e00495_score as score495  # ty: ignore[unresolved-import]


def main() -> None:
    """Dispatch to experiment 404's `pool` or `sweep` entry point after redirecting its folders."""
    command = sys.argv.pop(1)
    if command == "sweep":
        sys.argv.insert(1, "run")
    score495.point_at_495()
    {"pool": pool404.main, "sweep": sweep404.main}[command]()


if __name__ == "__main__":
    main()
