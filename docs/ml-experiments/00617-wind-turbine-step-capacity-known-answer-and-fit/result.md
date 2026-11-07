---
id: "00617"
slug: wind-turbine-step-capacity-known-answer-and-fit
title: >-
  Can a hidden Markov model find whole-turbine capacity steps at the 3 metered wind farms, in a
  known-answer test and in the real histories?
theme: generator-forecasting
also_relevant_to: []
status: planned
kind: other
date: 2026-11-07
size: >-
  Medium. Answers to the five triggers: (1) it stores only private daily series and fitted
  posteriors; (2) more than one defensible design exists, and the plan in the data folder chose one;
  (3) it reuses the plant summary of experiment 347 and the series of experiment 453; (4) it touches
  no serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  The effective capacity of a metered wind farm moves in whole-turbine steps. A hidden Markov model
  on the daily upper envelope of power in the saturation range of the power curve recovers injected
  turbine outages of 2 or more turbines lasting 14 days or longer.
prediction: >-
  Written before any run. At W1 and W2, recall is at least 80% for 2-turbine outages of 14 days or
  more. At W3 recall is under 50% at 14 days, because W3 has about 43 informative half-hours a month.
decision_rule: >-
  Known-answer pass bars: recall at least 80% (outages of 2 or more turbines, 14 days or more, W1
  and W2); at most 1 false step a year on permuted history; recovered step within half a turbine in
  at least 80% of detected steps; recovered turbine size within 15% in at least 70% of runs; 90%
  interval covering the true capacity on 85% to 95% of injected days. Stop the wind line if W1
  recall is under 50% or false steps exceed 3 a year. The real fit runs only if W1 and W2 pass.
benchmark: >-
  Experiment 347's all-row daily-ratio test, which detected a one-turbine loss in 0 to 2% of blocks.
row_set_id: wind_saturation_rows_to_2026_06_30
metric: Recall, false steps a year, step-size error, turbine-size error, interval coverage.
---

# Wind turbine-step capacity: known-answer test, then real fit

Planned. The results replace this page when the run finishes.
