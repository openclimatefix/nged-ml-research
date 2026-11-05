# ruff: noqa: D103, D301, E501
"""Experiment 580: known-answer check of the sign convention of the corrected target and lags.

A synthetic Primary has a daily-shaped baseline. A transfer adds `depth` (fraction of capacity,
positive = the Primary carries more) between `onset` and `end`. The check confirms three things:

1. The correction used by experiments 435, 566, 578 (`corrected = observed + shift`, with
   `shift = -f1.0` and `f1.0 = +depth_share` inside the event, as written by `e00435_nra_actuals.py`)
   recovers the baseline exactly, so it removes the event rather than doubling it.
2. This experiment's `correct()` and the fitted depth of `step_series` agree with that convention.
3. A wrong-sign label (`-depth`) doubles the event, which is the failure the audit looks for.

```bash
PYTHONPATH=ml-experiments/00580-verified-load-shift-event-set-and-label-audit:ml-experiments/00510-... \\
    uv run --no-sync python ml-experiments/00580-*/e00580_check_sign.py
```
"""

import e00580_verify as v  # ty: ignore[unresolved-import]
import numpy as np

N = 48 * 60
ONSET, END = 48 * 20, 48 * 35
DEPTH = 0.3


def synthetic() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N)
    base = 0.5 + 0.2 * np.sin(2 * np.pi * (t % 48) / 48)
    obs = base.copy()
    obs[ONSET:END] += DEPTH
    return base.astype(np.float32), obs.astype(np.float32)


def main() -> None:
    base, obs = synthetic()
    # (1) the stored-correction path of experiment 435: f1.0 = +share inside the event; shift = -f1.0
    f10 = np.zeros(N, np.float32)
    f10[ONSET:END] += DEPTH
    shift = -f10  # `e00578_common.corrected_shift`
    assert np.allclose(obs + shift, base, atol=1e-6), "435 correction does not recover the baseline"
    # (2) this experiment's correction and fitted step
    assert np.allclose(v.correct(observed=obs, depth=DEPTH, onset=ONSET, end=END), base, atol=1e-6)
    s = v.step_series(x=obs, w=3 * 48)
    assert np.isclose(s[ONSET], DEPTH, atol=1e-4), s[ONSET]
    assert np.isclose(s[END], -DEPTH, atol=1e-4), s[END]
    # (3) a wrong-sign label doubles the event
    wrong = obs - shift  # label depth of the opposite sign
    assert np.isclose(wrong[ONSET:END].mean() - base[ONSET:END].mean(), 2 * DEPTH, atol=1e-5)
    # event 146 pattern: a label of +depth over a span where the telemetry has no step gives a corrected series
    flat = base.copy()
    assert (flat + shift)[ONSET:END].mean() < base[ONSET:END].mean() - 0.9 * DEPTH
    print("sign convention checks passed")


if __name__ == "__main__":
    main()
