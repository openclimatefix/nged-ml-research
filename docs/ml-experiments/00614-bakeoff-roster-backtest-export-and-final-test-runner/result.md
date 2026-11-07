---
id: "00614"
slug: bakeoff-roster-backtest-export-and-final-test-runner
title: >-
  Can the three roster networks be exported as stored backtests that reproduce experiments 612 and
  613, and can a final-test runner be built that scores each frozen candidate once?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Answers to the five triggers: (1) it stores Parquet exports and a runner script in the
  private data folder and the repository, and writes no new network weights if the stored seeds
  suffice; (2) one defensible design for the export (the seed means that experiments 612 and 613
  scored), and one for the runner; (3) it spans code whose callers cannot be named without
  searching (experiments 593, 597, 605, 607, 612, and 613); (4) it touches no serving path and no
  degradation rule; (5) it changes no contract. The export reads stored forecasts only. The runner
  is tested on a development month and is not run on the final test.
hypothesis: >-
  The stored forecasts of experiments 612 and 613 can be repackaged for the three roster networks
  (class 1 K1, class 2 candidate, class 3 K3) over the five development months so that
  event-free q05-q95 coverage recomputed from the exported files matches the coverage those
  experiments printed, and a runner can check the timing rules row by row, score each frozen
  candidate exactly once, and refuse to run a second time.
prediction: >-
  Written before any export. P1: every recomputed event-free coverage cell (Primary group, lead
  band, network) is within 0.003 of experiments 612 and 613's stored cells. P2: the forecasts are
  stored to 240 h for all three networks, so no network falls back to 0-120 h. P3: the runner's
  development-month test passes both timing assertions on every row and the second invocation
  refuses to run.
decision_rule: >-
  Descriptive. The export passes if P1 holds in every cell; any mismatch above 0.003 is reported
  with its cell. The runner passes if the development-month test with a dummy configuration writes
  Parquet forecasts, records the candidate hashes in the lock file, and refuses a second run. The
  experiment adopts nothing, tests no hypothesis about forecast skill, and so applies no Bonferroni
  correction (count: 0) and no minimum detectable effect.
benchmark: >-
  The stored event-free cells of experiments 612 and 613 (`score/*/event_free_cells.csv`).
row_set_id: experiment_470_intersection_v1
metric: >-
  Event-free q05-q95 coverage by Primary group and lead band; rows breaking each timing assertion.
---
**Plan, written before any export.** This experiment packages the three roster networks' stored
five-seed backtests for the five development months, and builds a runner for the one final test on
origins from 2026-07-01 to 2026-10-31. The runner is not run. July 2026 onwards is untouched.
