---
id: "00378"
slug: primary-demand-default-3-at-all-16-primaries-under-demand-v1
title: >-
  How does Primary demand research default 3, ported to all 16 Primaries, score against BG2, the
  manual heuristic, and climatology under the demand-v1 protocol?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts, fits, and small result files in this experiment's private folder only;
  experiments 87, 111, 152, 317, 319, 322, 374, 376, and 377 are read and never written. Production
  serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (refit the long-lead weights per Primary or pooled per group); the choice is fixed below. Callers:
  the scripts copy and import the folders of experiments 317, 319, 322, and 377, whose callers are
  all in those folders; no change under `packages/` or `src/`.
hypothesis: >-
  Primary demand research default 3 ported to all 16 Primaries is not worse than BG2 by more than
  0.20 points of effective capacity at days 1 to 10 at the six signed-MW Primaries, and it beats
  Primary demand research default 1 extended to 240 hours (experiment 377) at 168 to 240 hours at
  both groups.
prediction: >-
  Written before any score is read. P1. Signed-MW six, default 3 minus BG2 on the shared rows, days
  1 to 10, pooled: between -0.20 and +0.20 points. P2. Default 3 minus default 1 extended at 168 to
  240 hours is below zero at the signed-MW six (experiment 322 found -0.064 - 0.075 = about -0.14
  points against default 1) and between -0.20 and 0 at the MVA-only eight. P3. MVA-only eight
  days 1 to 10 minus BG2: between +0.30 and +0.90 points with D8, between -0.05 and +0.25 without.
  P4. The control gate holds: the port reproduces experiment 322's stored default 3 scores on the
  six within 0.05 points at 0.5 to 240 hours and at 168 to 240 hours.
decision_rule: >-
  Descriptive, no adoption. Contrasts registered before any score is read, CRPS in points of
  effective capacity on the shared rows, by the paired month-resampled bootstrap of
  `studies.eval_protocol` at 99% (20,000 resamples) with 95% beside it. C1: default 3 minus BG2,
  days 1 to 10 (BG2 only to 168 hours), signed-MW six pooled. C2: the same for the MVA-only eight,
  with and without D8, labelled "reactive power assumed zero". C3: default 3 minus default 1
  extended (experiment 377), 168 to 240 hours, per group. C4: default 3 minus climatology and minus
  the manual heuristic, per group. D15 and D18 are report-only and enter no tally. One verdict per
  group.
benchmark: >-
  BG2, climatology, and `manual_heuristic` as stored by experiment 87 on `fold_a_daily_gap_v1`;
  Primary demand research default 1 extended to 240 hours (experiment 377).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, mean absolute error of the median, and
  twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 240 hours.
code: ml-experiments/00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1/
---

# Primary demand research default 3 at all 16 Primaries under demand-v1 (plan)

This page is the plan. The write-up replaces it when the scores exist.

## What depends on the six-Primary set

- **Chain below 168 hours.** Default 1's stack comes from experiment 376 and 377 at all 16
  Primaries, so nothing is ported here.
- **Solar summary S (experiment 319).** Fitted per scored month on the trailing 365 days of
  origins, per Primary (`WINDOW_DAYS` in `e00319_common.py`). The loops over `SCORED_SITES` (D5,
  D9, D10, D11, D14, D16) become the 16 Primaries. Each MVA-only Primary needs its own fitted S on
  its rule-chosen fit (experiment 374), with the naive fit as a second arm.
- **Heuristic blend (experiments 319 and 322).** The convex weight is fitted per lead band and
  season group on trailing 365 days, per Primary. The daylight gate (sun above 5 degrees) needs no
  fit. Inputs: the stored `manual_heuristic` forecasts of experiment 87 at all 16, and the ENS
  temperature store of experiment 111 at each site (needed only by the D3 reference arm, which is
  not ported).
- **Hand-filtered event lists.** Scoring excludes the hand-filtered events; the 16-Primary lists
  come from the demand-v1 protocol module. Any Primary without a list is reported.
- **Fold set.** Experiment 322 used the six-month-plus tuning and replicate periods; demand-v1 uses
  the contiguous-month folds, so every fit is refitted for each scored month on earlier months
  only, as experiment 377 does.

## Method

- **Stage A (build).** Copy experiment 377's fold, stage-2 and adapter code, then add the
  experiment 319 summary S and the experiment 322 daylight blend at 168 to 240 hours, with
  `SCORED_SITES` replaced by the 16 Primaries.
- **Stage B (control gate).** Run the port at the six signed-MW Primaries on experiment 322's own
  folds and compare with its stored default 3 scores. Pass: within 0.05 points at 0.5 to 240 hours
  and at 168 to 240 hours. Do not run the other ten Primaries until this passes.
- **Stage C (all 16, demand-v1 folds).** MVA-only eight use the rule-chosen fit of experiment 374
  (main arm) and the naive fit (second arm). D8 uses the donor spread and is the labelled
  exception. D15 and D18 are report-only.
- **Stage D (scoring).** Score through `studies.eval_protocol` with the contrasts above, with and
  without D8, plus per-Primary and leave-one-out contrasts beside C1 and C2.

## Resources and rules

- Sonnet only. An Opus audit only with the manager's approval; a Sonnet audit before merge.
- 12 cores, 16 GB per job; `uptime` first. No fetch or egress. No NGED generator named.

## What the reconnaissance found

- **Stored inputs exist for 15 Primaries.** Experiment 377 stores per-fold stage-2 forecasts (11
  quantiles, `mean`, `actual`, `run`, `lead_hours` to 240 hours) under `work/control/arms/a0/dp_preds_s2/`
  (the signed-MW seven, D18 included) and `work/mva/arms/a0/dp_preds_s2/` (the MVA-only eight, D8
  with an own and a donor file). Experiment 377 did not fit D15.
- **The solar summary S needs member-level solar responses that nobody has stored for 15
  Primaries.** Experiment 319 computed them with experiment 184's engine for six Primaries only.
  The engine's inputs exist for all 15: experiment 111's calibrations for the signed-MW seven and
  experiment 152's for the MVA-only eight; experiment 377 stores the monthly solar fits
  (`tall_fits/monthly`) for both groups. Stage A therefore runs the 184 member pass through 377's
  copies of the modules (`E00377_SOURCE=control|mva`).
- **Experiment 377's forecasts differ from experiment 317's chain.** Experiment 377 refits stage 2
  per month, so the control gate compares the port on the six against experiment 322's stored
  scores at the six Primaries and judges the difference as the effect of the folds, not as a bug,
  only when the unchanged-fold run (S and blend on experiment 317's chain) reproduces 322 first.

## Assumptions

- **MVA-only fit.** Experiment 377's `a0` arm uses the naive fit at all eight MVA-only Primaries,
  including D12 (confirmed by the 377 worker). The rule-chosen fit of experiment 374 was not run at
  240 hours, so the main arm needs a re-run of experiment 377's stage 1 and stage 2 with experiment
  374's fits; that re-run is stage C2 and starts after stage C on the naive fit has been scored.
- **Below 168 hours.** Experiment 377's stage 2 has no two-number solar summary S at any lead
  (confirmed by the 377 worker). Default 1 includes S below 168 hours (experiments 202 and 251), so
  the brief's "default 1 stack below 168 h" needs S applied there too. Stage A therefore also
  stores member responses at 0.5 to 168 hours and applies experiment 202's per-band fit
  (`e00202_arms.fit_factors`) to those leads. The control gate compares the six against
  experiment 322's stored scores at 0.5 to 240 hours.

- **Comparison with experiment 377.** Experiment 377's "default 1 extended to 240 hours" lacks S
  at every lead, so it is not default 1. The default 1 comparison here (C3) is the faithful one:
  default 3 minus default 1 means the S layer below 168 hours, S at 168 to 240 hours, and the
  daylight blend. Experiment 377's stack is reported as a third reference, "the stack without S".

- **Tails (maintainer, 2026-10-28).** The tails matter most. Every table reports pinball loss and
  coverage at q01, q05, q95, and q99 beside CRPS and twCRPS, per group and per Primary, and the
  take-home names the tail result. Rows where the solar summary or the heuristic is missing (no
  member, no heuristic median) fall back to the stage-2 quantiles; width and coverage are reported
  on those rows against the rest.

## Open questions for the manager

- None. Experiment 377's stored forecasts are in place; its scores and PR are still to come.
