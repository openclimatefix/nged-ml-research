---
id: "00668"
slug: documentation-of-the-one-year-backtest-files-models-training-and-performance
title: >-
  What do the one-year backtest files hold, how were the five models behind them built and trained,
  and how well do they forecast?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
  - lagged-power-and-error-correction
status: planned
kind: other
size: >-
  Medium. (1) It stores no forecasts: the page reads the five backtest files and the observed power
  and writes figures and tables; (2) the page has one defensible structure, set in the plan below;
  (3) it spans the code of experiments 611, 620, 633, 652, 654, 659, 666, 667 and the BSP and GSP
  backtest, so every model section is checked against the code by a fact-check reviewer; (4) it
  touches no serving path or degradation rule; (5) it changes no contract.
hypothesis: >-
  Not a hypothesis test. The page documents the five one-year backtest files so that an engineer can
  re-implement each model and a colleague at NGED can read the files.
prediction: >-
  None. The page reports descriptive scores only.
decision_rule: >-
  None. The page is finished when the fact check, the numbers review and the prose review have no
  open finding.
benchmark: >-
  Climatology beside every model, and the manual heuristic or the earlier research default where
  one exists.
row_set_id: backtest_year_2025_11_to_2026_10
metric: >-
  CRPS in points of capacity, normalised mean absolute error of the median, q05 to q95 coverage, by
  lead band and by month.
---
This page is planned. It will document the five Parquet files in
`data/ml-experiments/backtest_year_2025-11_to_2026-10/`.

## Outline

- **Take-home message and headline figure:** score against lead time for each model, with
  climatology.
- **The files:** the five files, every column and type, the period, the horizon, the data delays,
  the meaning of `NRA_or_prevailing` and of the quantile columns, units and row counts.
- **One section per model**, each with a bulleted Method (inputs, architecture, loss, training
  data, pre-training, refits and embargo, seeds, capacity rule, missing-input handling,
  hyper-parameters with the file and experiment page each came from): Primaries, BSPs and GSPs,
  wind, solar.
- **Performance per model:** metrics tables by lead band and by month, with climatology and the
  manual heuristic or earlier default; time-series figures; score against lead time.
- **Caveats:** descriptive backtest that overlaps the development months and the final test; ENS
  history starts 2024-04-01; days 10 to 14 untested for the demand networks; solar's upper band on
  very sunny days; incomplete shift labels.
- **Anonymisation:** generators appear as Wind A to C and Solar A to F, with no coordinates, no
  calendar dates, and no time series ids.
