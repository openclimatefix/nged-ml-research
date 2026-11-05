# Experiment 578: scripts

Run from the repository root with `DATA_PATH_INTERNAL` set and `source ml-experiments/00524*/env.sh`.
The write-up is `docs/ml-experiments/00578-obvious-load-shift-gate-on-lags-follow-ratio/result.md`.

1. `e00578_step0.py` defines the long-and-deep events from the labels (CPU).
2. `run.sh pretrain|fit|assemble` (the wrapper `e00578_fit.py`) fits and re-forecasts the v3+D5 network
   of experiment 572, with a corrected-lag twin. `pipeline.sh` runs the whole chain.
3. `e00578_follow2.py`, `e00578_level.py`, and `e00578_nra.py` compute the follow ratio, the level
   share, and the other metrics per event. `e00578_gate.py` runs the descriptive gates.
4. `e00578_figures.py` and `e00578_nra.py` draw the figures.
