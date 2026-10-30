---
id: "00398"
slug: load-shift-event-rescore-without-flagged-zero-actuals
theme: production-reference-and-best-guess
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v1
size: >-
  simple. Stored: small result tables only, in this experiment's private folder; no forecast is
  written and no model is fitted. Designs: one defensible design, because the stored per-row score
  files already carry every slice flag, so dropping the flagged keys and re-aggregating with the
  existing bootstrap is the whole method. Callers: the script imports the scorers of experiments
  379, 381, and 391 and nothing imports it. Production serving path and degradation rule: not
  touched (this repository has neither).
hypothesis: >-
  The flagged exact-zero Primary half-hours that the scoring actuals of experiments 377, 379, 381,
  and 391 still contain (the `substation_zero` rule of 2026-10-06 flags them in the production table)
  do not change any published headline figure to its printed precision, and change no arm ranking
  or sign of an arm contrast interval.
prediction: >-
  Written before any score is read. P1. Every published headline CRPS and tail figure moves by less
  than 0.005 points of capacity, because the flagged half-hours are about 38 of 257,000 unique
  half-hours. P2. No arm contrast interval changes sign and no ranking changes. P3. The largest
  change is at D21, which holds 13 of the 38 flagged keys. P4. Experiment 379's unit check
  (finite in both sources only) cannot see the gap, because a flagged actual is finite.
decision_rule: >-
  Descriptive, no adoption. The result is "no effect" if every figure agrees to its printed
  precision and no contrast interval or ranking changes. Any figure, interval sign, or ranking that
  moves is listed and sent to the manager and the Discuss session. The threshold is zero, because
  the experiment changes nothing in production.
benchmark: >-
  The published headline figures of experiments 379, 381, and 391, and the numbers of the stored
  per-row score files they were built from.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, threshold-weighted CRPS, and the pinball loss at q01, q05, q95, and q99, in points of
  effective capacity, from `studies.eval_protocol`.
code: ml-experiments/00398-load-shift-event-rescore-without-flagged-zero-actuals/
---

# Experiment 398: do the flagged zero actuals change any published demand score?

## Plan

**The question.** The scoring actuals of experiments 377, 379, 381, and 391 come from experiment
103's `rows.parquet`, which is built without the `drop_reason` filter. The production table flags
38 unique Primary half-hours in their scored rows as `substation_zero`, each with an actual of
exactly 0. This experiment drops those half-hours from the scored rows and reports how far each
published headline figure moves. Training rows and lag inputs of these experiments are already
clean, so only the scoring actuals are affected.

**Method.**

- Build the set of flagged keys (Primary, valid time) from experiment 87's `observed.parquet`
  (flagged half-hours held as nulls), and state its build date and the count per Primary for each
  file rescored. The counts depend on the scored set, so each file gets its own count.
- Drop the flagged keys from the stored per-row score files: experiment 379's `rows_stack.parquet`
  and `rows_components.parquet`, experiment 391's `scores/rows.parquet`, and experiment 395's
  `b_rows_*.parquet` (which also carry experiment 381's monthly arms). Re-aggregate them with the
  existing bootstrap. No scorer is rerun and no forecast is rebuilt.
- Experiment 381's `final_scores.parquet` and its weekly bundle have no row-level scores. Rescore
  the weekly bundle from its stored forecasts with the flagged keys dropped, using experiment 381's
  `build_context` and `score_arms`.
- Report old value, new value, and change for each published headline figure at the slices the
  pages report, and the Primaries that hold flagged keys (D21, D8, D22, D9, D12, D18, D16) apart.
- Show that experiment 379's unit check cannot detect the gap, and that the rescore does.
- Experiments 395, 388, 392, and 396 report their own checks and are not repeated.

**Compute.** At most 2 cores, `MemoryMax=6G`, no GPU, no fit. The 2.4 GB components file is read
lazily and filtered before any collection. Estimated wall time is about 1 hour, and the job stops
and goes back to the manager if the first aggregation pass projects over 2 hours.
