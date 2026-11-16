---
id: "00701"
slug: gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10
title: >-
  How do the GAM and the XGBoost blend score on the final-test origin months 2026-07 to 2026-10,
  beside the neural networks of experiments 620 and 677?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  Moderate. (1) Changes what is stored: yes, extended input tables, rebuilt forecasts, and scores,
  all private. (2) Touches the production serving path: no. (3) Touches a degradation rule: no. (4)
  Admits more than one defensible design: yes (which XGBoost blend, how to extend the inputs), so
  the choices are fixed in Method. (5) Spans code whose callers cannot be named without searching:
  yes (experiments 87, 377, 378, 607, 620, 700). The scripts import those experiments' modules.
hypothesis: >-
  Descriptive, not a hypothesis test. The run extends experiment 700's scores of the GAM and the
  XGBoost blend from the origin months 2025-11 to 2026-06 to 2026-07 to 2026-10, and so scores
  both systems on the final-test window that experiment 620 scored for the neural networks.
prediction: >-
  Written before the run. P1: the GAM and the XGBoost blend score below the manual heuristic on
  each of the 14 forecast days, as they did over 2025-11 to 2026-06. P2: step 8 with monthly refits
  scores below both on pooled MAE. P3: the extended inputs equal the stored inputs on every shared
  row. The prediction fails if P1 or P3 fails.
decision_rule: >-
  Descriptive, scored once on the final-test window with no tuning. No claim of benefit is made, and
  no production code changes.
benchmark: >-
  The manual heuristic, climatology, and steps 5 and 8 of experiment 677, on the same rows.
row_set_id: final_test_origins_2026_07_01_to_2026_10_31
metric: >-
  Normalised mean absolute error of the median forecast, continuous ranked probability score, and
  threshold-weighted CRPS, in points of capacity; q05 to q95 and q01 to q99 coverage.
code: ml-experiments/00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10/
---
Planned. The write-up follows the run.
