# ruff: noqa: E501
"""Experiment 701: run experiment 87's `e00087_fit_demand.py` on the 123 runs 2026-07-01 to 2026-10-31.

The script imports `e00701_common`, which points experiment 87's constants at this experiment's
sandbox (see its docstring), and then calls experiment 87's `main()` with the arguments given.
Experiment 87's fit logic, parameters, seeds, training windows, and file layout are unchanged. The
predictions go to `work/anchor_preds/<config>[_o|_oheld]/<site>_s0_<refit>.parquet` in this
experiment's private data folder, and the wall-clock seconds of each job to `work/gpu_hours.jsonl`.
A file that exists is skipped, so a stopped run resumes.

    PYTHONPATH=<folders of experiments 87, 33, 48, 701> \
        uv run --no-sync python ml-experiments/00701-*/e00701_fit_demand.py \
        --config n1h --regime monthly [--offset] [--workers 6] [--cores 12]

`--config` is one of `n1h` (the weekly and monthly regimes, `--regime` picks one), `frozen`, `an`,
and `pt`; `--offset` fits the offset twin at the five offset Primaries. `--dry-gpu-hours` times two
real fits. Do not pass `--smoke` or `--repeat` (the smoke test of this experiment is
`e00701_smoke.py`).
"""

import e00087_fit_demand as fit  # ty: ignore[unresolved-import]
import e00701_common  # noqa: F401

if __name__ == "__main__":
    fit.main()
