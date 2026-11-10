---
id: "00691"
slug: causal-detector-features-plus-extra-lags-in-the-prevailing-head
title: >-
  Does a prevailing head that reads both the causal detector features and the extra lagged power
  inputs react to a load shift sooner than a head that reads only one of them?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  Simple to Medium. (1) It stores one new set of head weights and predictions per month and seed
  (10 fits, about 9,000 parameters each) and no new table; (2) one design is defensible, because
  the arm is the union of two arms that experiment 687 already defined; (3) it imports the code of
  experiment 687 and nothing outside this folder imports it; (4) it touches no serving path or
  degradation rule; (5) it changes no contract.
hypothesis: >-
  The detector features of experiment 687 (F1) and the extra lagged power block of experiment 681
  (L1) carry partly different information, so a head that reads both (FL1) reacts to the start of a
  load shift sooner than the better of the two single-input heads.
prediction: >-
  Written before any fit or score. P1 (enters the verdict): at the start of an injected shift of
  the held-out family, FL1 covers half of the new level at least 0.5 days sooner than the better
  of F1 and L1, and the 95% interval of the paired contrast over events excludes zero. P2 (enters
  the verdict): the scaled in-force CRPS of FL1 improves on B0 by 0.30 points of capacity or more
  (the bar of experiment 687, whose single arms reached 0.12 for F1 and 0.25 for L1). P3 (a veto):
  the false-alarm rate on event-free injected-bank rows rises by at most 1 percentage point
  against B0, as in experiment 687.
decision_rule: >-
  Support needs P1 and P2 to hold and P3 not to fail. Stop rule, fixed before any score is read:
  if the reaction gain over the better single arm is under 0.5 days, or its interval includes
  zero, the result says plainly that features and lags overlap (they carry the same information),
  and no new feature is proposed. The paired contrasts FL1 minus F1, FL1 minus L1, and FL1 minus
  B0 are reported with 95% intervals (bootstrap over events for reaction time, over the 5
  development months for CRPS). The 0.5-day and 0.30-point thresholds are chosen as the smallest
  gain that would justify a head that has to compute both blocks at serving time; the lag block
  costs 16 inputs and the feature block 90.
benchmark: >-
  Arms B0, L1, and F1 of experiment 687, loaded from its stored fits (same code, seeds, banks, and
  folds, so they are the arms a refit would give), and K1 as context.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  Reaction time in days to cover half the new level after an injected shift starts and ends;
  in-force and event-free scaled CRPS; false-alarm rate on event-free rows; q05 to q95 coverage;
  q01, q05, q95, and q99 pinball loss; avoidable-curtailment cost; on real confirmed events, the
  CRPS and the departure of the head from K1.
---
Throughout, K1 is the frozen two-output demand neural network, and the small head is the neural
network that corrects its prevailing output (experiment 652's R-haz head). B0 is that head with no
extra inputs, L1 adds the extra lagged power block (experiment 681), F1 adds the causal detector
features D1 to D6 (experiment 687), and FL1 adds both.

## Plan

- **Question.** Experiment 687 found that F1 reacts 1.59 days [1.00, 2.26] sooner than B0 at the
  start of a shift and L1 1.25 days sooner, with F1 minus L1 at -0.33 [-0.72, +0.07]. No arm read
  both. This experiment fits that arm.
- **Arms.** B0, L1, F1 (loaded from 687's stored fits), and FL1: B0 plus F1's feature columns (day
  lag, week lag, and 6-hour cutoff) plus L1's 16 lag columns, with the larger of F1's age and L1's
  age opening the survival gate, and dropout on the extra columns as in F1 and L1. An FL1 arm with
  the explicit flag (F2's extra column) is not run.
- **Rows and data.** The banks, folds, five scored months (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06), two seeds, and cleaned power of experiment 687. Data before 2026-07-01 only (asserted).
  No refit cadence applies: one small head is fitted per scored month.
- **Measures.** As in experiment 687, with the real confirmed events reported first: the CRPS and
  the departure from K1 on the verified D5 and D16 transfer of December 2025, then the injected
  held-out bank.
- **Compute.** 10 fits on 4 CPU processes, about 5 minutes (experiment 687's 100 fits took about
  25 minutes). No GPU. Fits saved: 90 of 100.
- **Reviews.** A Sonnet code review before the fits, a Sonnet check of the write-up against the
  printed report, and a Sonnet prose review.
