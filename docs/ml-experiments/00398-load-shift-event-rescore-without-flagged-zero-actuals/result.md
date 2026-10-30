---
id: "00398"
slug: load-shift-event-rescore-without-flagged-zero-actuals
theme: production-reference-and-best-guess
also_relevant_to:
  - lsc-detectors
status: finished
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
verdict: partly supported
production_status: not-for-production
adopted_in: none
best_arm: not applicable
headline: >-
  Dropping the 38 flagged exact-zero half-hours lowers the days 1 to 10 CRPS by at most 0.0023
  points at the signed-MW six and 0.008 points at the MVA-only eight for BG2, the control arms,
  and the GAM stack. No arm contrast interval changes sign or crosses zero, and no ranking moves.
lessons:
  - >-
    A flagged zero actual lowers every arm's error when it is dropped, and lowers it by nearly the
    same amount for every arm, so contrasts between arms barely move.
advice:
  - >-
    Score against the cleaned actuals (flagged half-hours dropped) in every new experiment, because
    the old cache inflates absolute scores by up to 0.035 points at D21 (per-Primary table).
  - >-
    Compare absolute scores of experiments 379, 381, 391, and 395 with each other freely, but do not
    compare them with a score built on the cleaned actuals without the 0.002 to 0.008 point shift
    (headline table).
take_home_message: >-
  The 38 half-hours that the production table flags as exact zeros made the published demand
  scores slightly too high, by at most 0.008 points of capacity on average. They changed no arm
  ranking and no contrast, so none of the published conclusions needs correcting.
---

# Experiment 398: do the flagged zero actuals change any published demand score?

**The flagged zero actuals change no published conclusion.** Four earlier experiments (379, 381,
391, and 395) scored their forecasts against half-hourly readings that still contained 38 half-hours
which the production table marks as faulty zeros. We removed those half-hours from the stored
scores, recomputed every headline figure, and compared old with new. The average error falls by at
most 0.0023 points of capacity at the six Primaries with signed readings and 0.008 points at the
eight MVA-only Primaries. Within each file every arm falls by nearly the same amount, so every
contrast between arms agrees with the published one. The verdict is "partly supported" because
the hypothesis asked for agreement to the printed precision, and the MVA-only figures move in the
second decimal place.

## Headline figures, days 1 to 10 (points of effective capacity)

Old is the published scoring set, new has the flagged half-hours dropped. Pooled over all rows of
each group.

| Experiment, arm | Group | CRPS old | CRPS new | Change | Pinball q01 change | Pinball q99 change | Threshold-weighted CRPS change |
|---|---|---|---|---|---|---|---|
| 379, BG2 | signed-MW six | 3.0160 | 3.0138 | -0.0023 | -0.0021 | -0.0000 | +0.0000 |
| 379, BG2 | MVA-only eight | 3.7429 | 3.7352 | -0.0077 | -0.0051 | -0.0001 | +0.0000 |
| 391, control (b0) | signed-MW six | 3.1234 | 3.1211 | -0.0023 | -0.0022 | -0.0000 | +0.0000 |
| 391, control (b0) | MVA-only eight | 7.6831 | 7.6778 | -0.0053 | -0.0009 | -0.0002 | +0.0000 |
| 391, surprise (n1) | signed-MW six | 3.2293 | 3.2271 | -0.0023 | -0.0022 | -0.0000 | +0.0000 |
| 391, surprise (n1) | MVA-only eight | 7.6789 | 7.6740 | -0.0049 | -0.0011 | -0.0002 | +0.0000 |
| 395, control (m_control) | signed-MW six | 3.1132 | 3.1110 | -0.0023 | -0.0022 | -0.0000 | +0.0000 |
| 395, control (m_control) | MVA-only eight | 3.7843 | 3.7763 | -0.0079 | -0.0056 | -0.0001 | +0.0000 |

- **Coverage at q01, q05, q95, and q99** changes by at most 0.0043 (as a fraction) in any group, for
  every arm of the four files.
- **Other slices** (lead bands, winter, Economy 7 step, and the rest) change by at most 0.018 points
  in CRPS. Two slices change more. The load-shift event rows at the MVA-only eight fall from 5.197
  to 4.952 (-0.245) for BG2, because 70 of 16,478 rows are dropped. The error of the median for the
  reference forecasts in experiment 391's sunny daylight slice falls by up to 0.080. The cold-night
  slice does not change at all.
- **The short-history group** (D15 and D18) is report-only. In experiment 379's component file it
  falls by up to 0.185 points because that file holds 239 flagged keys at those two Primaries (D15
  120, D18 119); no published headline uses it.

## Contrasts and rankings

- **No contrast interval changes sign or crosses zero.** We recomputed 81 paired contrasts (99%
  interval, 5,000 resamples, month-resampled): experiment 379's 14 arms against BG2, experiment
  391's five arms against the control (all rows, lag inside an event, lag outside), and experiment
  395's seven monthly arms against the control (CRPS and error of the median), each at the signed-MW
  six and MVA-only eight. The largest change in a point estimate is 0.0008 points, and in an
  interval bound 0.0017 points.
- **The weekly bundle of experiment 381** was rescored from its stored forecasts. Bundle minus the
  GAM stack at the signed-MW six is +0.0749 old and +0.0749 new, at the MVA-only eight with D8
  -0.3609 and -0.3612, and without D8 -0.1967 and -0.1966. Bundle minus weekly baseline, BG2, manual
  heuristic, and climatology move by at most 0.0011 points. The largest single-Primary change for
  the bundle is -0.035 points at D21.
- **No arm ranking changes** within experiments 379, 391, and 395 at the headline slice, because
  every arm of one file loses the same few rows and falls by nearly the same amount.

## Flagged half-hours per Primary

The flagged keys are the 2,035 `substation_zero` rows of the cleaned power table between 2024-04-01
and 2026-07-01, each with a power of exactly 0 (build date: the table as read on 2026-10-30). The
counts depend on the scored rows of each file, so each file has its own.

| File | Unique flagged half-hours dropped | Per Primary |
|---|---|---|
| 379 stack and BG2 (`rows_stack`), 395 (`b_rows_*`) | 38 | D21 13, D8 7, D22 5, D9 5, D12 4, D18 3, D16 1 |
| 379 components (`rows_components`) | 276 | D15 120, D18 119, D21 13, D8 9, D22 5, D9 5, D12 4, D16 1 |
| 391 (`scores/rows`) | 15 | D8 9, D9 5, D16 1 |
| 381 weekly bundle (scored from forecasts) | not counted by key | its scoring actuals lose 1,179 of 630,528 half-hours across all 16 series and the whole window |

At the Primaries that hold the flagged keys, BG2's days 1 to 10 CRPS in experiment 379's stack file
falls by 0.035 at D21 (2.377 to 2.343), 0.022 at D18, 0.017 at D22, 0.011 at D9, 0.005 at D12, 0.004
at D8, and 0.003 at D16. The other nine Primaries do not change.

## Checks

- **Experiment 379's unit check cannot see the gap.** It compares the stored actual of experiment 87
  with the cache actual on rows where both are finite, and a flagged actual is a finite 0.0 in both
  sources, so the maximum difference is zero. The rescore finds the gap because it joins to the
  production table's `drop_reason`, not to another copy of the same cache.
- **The median forecast minus the mean forecast and the error of the mean forecast** are not
  recomputed: the experiment changes the scored rows, not the forecasts, and the stored losses
  already carry the quantile score.
- **Triggers fired:** none. No control failed, no result contradicts its prediction, and no nested
  arm beats the arm that contains it. The prediction P1 (every headline moves by less than 0.005)
  was missed at the MVA-only eight, where the change is up to 0.008; P2, P3 (D21 largest), and P4
  held.
- **Cost:** the rescore reads and filters stored per-row scores and recomputes contrasts; no scorer
  was rerun except experiment 381's own weekly bundle.

## Caveats and what to do next

- **The 38 half-hours are not the whole cache.** Experiment 388's scorer, which covers the full
  scored year for every Primary including D15 and D18, finds 622 unique flagged half-hours, 389 of
  them at D18 and 196 at D15. Experiments built on that scored set, and experiments 392, 395's Part
  F, and 396, report their own check columns and are not repeated here.
- **Interval resamples:** the 99% intervals use 5,000 resamples, not the 20,000 of the headline
  protocol, to keep the job inside its time limit. All widths agree with the published figures to
  the printed precision.
- **A fix in `src/studies/dp_features.py` or experiment 103's builder** (dropping flagged half-hours
  when the cache is built) would stop new experiments scoring against faulty zeros. We did not make
  it, because it changes the shared cache that open experiments read. The maintainer decides.
