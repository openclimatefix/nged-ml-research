---
id: "00577"
slug: static-between-steps-offsets-and-probabilistic-offset-widening
title: >-
  Do offsets that stay constant between sparse steps, and offsets whose uncertainty widens the
  forecast, improve the output-side slow offset (D5) of the modular lag-free demand network?
theme: retraining-and-training-history
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights and forecasts under this experiment's private data folder only;
  experiments 565, 573, and 559 are read only. More than one defensible design: three
  constant-between-steps offsets and five ways to carry offset uncertainty, which is the question.
  Callers: the scripts import experiments 454, 470, 510, 511, 521, 559, 565, and 573. Serving path
  and degradation rule: not touched (this repository holds none).
hypothesis: >-
  The learnt offsets of D5 (experiment 565) wander from month to month, although a substation's
  true offset should stay constant for months and then step. A penalty that allows only sparse
  steps (a total-variation penalty on the first differences, or gated jumps) therefore gives a
  steadier offset without costing accuracy. Separately, the offset is the least certain part of the
  forecast, so adding its uncertainty to each mixture component's variance, and a small
  heavy-tailed component for a step that happened since the refit, improves the tails and the
  calibration beyond what a constant widening does.
prediction: >-
  Written before any score is read. Wander: D5's offsets change by less than 0.2% of capacity in at
  most 50% of event-free months, and a total-variation arm raises that share to at least 70%
  (fails if it rises by under 20 percentage points). Each total-variation arm (O1 monthly, O2
  daily, O3 gated) is within the minimum detectable effect (MDE) of D5 on CRPS, and at least one
  gains 0.03 or more. Known answer: in the zero-offset world O1 or O2 cuts the offset's root mean
  square error in static periods by at least 40% (fails under 20%). Q1 (spread of the offset
  across seeds) changes CRPS by under 0.02 and the 90% coverage by under 0.005. Q2 (variational
  newest knot) is within the MDE of D5. Q3 (jump-risk widening) improves the summed 1% and 99%
  pinball loss by at least 2% against D5 and by at least 1% against the constant-widening placebo
  (Q-const), and cuts the 98% coverage shortfall by a quarter. Q4 (best O arm with the best
  widening) is within the MDE of the better of its parts.
decision_rule: >-
  A gain counts when the Bonferroni month-block interval (5 blocks, near the minimum to maximum of
  the five folds) excludes zero, and it is at least 0.03 CRPS points of capacity for an arm that
  adds a penalty, a trained variance, or a mixture component. Bonferroni count 22 (Design 1: 3 arms
  x 2 groups = 6; Design 2: CRPS of Q1 to Q4 x 2 groups = 8, Q2 and Q3 against Q-const x 2 groups
  = 4, tail loss of Q2 and Q3 x 2 groups = 4). The threshold of 0.03 pays for one extra penalty or
  one stored variance per Primary. Static-ness, calibration, sharpness, known-answer errors, slices,
  and the per-Primary table are descriptive.
benchmark: >-
  D5 of experiment 565 (coarse refits on days 0 and 14; stored seeds 0 and 1 and fresh seeds 5 to
  9) as O0 and Q0, with the coarse M0 of experiment 565 as a second reference.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 24-120 h at the eight MVA Primaries and the six MW
  Primaries; summed 1% and 99% pinball loss; coverage of the 80%, 90%, and 98% central intervals.
---

*Plan only: this page is the pre-registered plan; the write-up replaces it when the fits are
scored.* The full brief is `briefs/offset_tv_probabilistic_brief.md` in the private data folder; the
decisions below fill in what the brief leaves open.

## Design choices that the brief leaves open

- **Cadence and rows.** Coarse refits on days 0 and 14 of each of the five folds, the harness of
  experiment 565's D5 arm, with the same pretraining rows (ERA5 and CAMS from 2019-09), seeds, and
  scoring rows. Forecasts are widened by 1.15 about the median, as every stored arm is; the raw
  (unwidened) CRPS of the Q arms is a sensitivity row.
- **O1 (monthly L1 total variation).** Same network as D5 with the penalty
  `lambda_tv x sum |first difference of knots| / N_pretraining_rows`, lambda_tv in {300, 1000,
  3000} nats per standardised unit (the sweep of experiment 573). The value is picked on fold 1's
  pretraining early-stopping NLL (seed 0): the largest within 0.002 of the best. All three are
  scored on fold 1 and reported.
- **O2 (daily L1 total variation, zero-order hold).** One knot per day; fine-tuning trains the
  newest 28 daily knots (a step inside the last four weeks can be learnt) and fixes the older ones;
  later days read the newest knot. The same lambda grid.
- **O3 (monthly gated jumps).** Knots are the cumulative sum of `gate x amplitude` per month, with
  hard-concrete gates (as S2 of experiment 573) and a penalty on the expected number of open
  gates; lambda_gate in {30, 100, 300}, picked on fold 1 as for O1. Fine-tuning trains the newest
  gate and amplitude.
- **Q2 (variational newest knot).** In fine-tuning the newest knot of the best O arm's network (D5
  itself for Q2 alone) has a Gaussian posterior with mean m and standard deviation s. The
  evidence lower bound uses one reparameterised draw per row and a KL term to a prior centred on
  the previous knot with standard deviation tau, divided by the number of fine-tuning rows. tau
  is picked on fold 1's training days from {0.05, 0.1, 0.2} standardised units by early-stopping
  NLL. Forecasts use m and add s squared to each component's variance.
- **Q3 (jump-risk widening).** Q2 plus, at valid time t, an added offset variance
  `rate x step_scale^2 x days_since_cutoff` and one extra mixture component of weight
  `min(0.05, rate x days_since_cutoff)` and standard deviation `step_scale`, centred on the
  mixture mean. Rate and step scale come from fold 1's training days only: from the jumps of O2's
  pretrained path if O2 passes its static-ness check, otherwise from the depth quartiles of the
  labelled events before fold 1.
- **Q-const (placebo).** Each component's standard deviation is multiplied by one constant per
  Primary, the mean ratio of Q2's widened standard deviation to the plain one on fold 1's
  fine-tuning rows, applied to all folds.
- **Q1.** The standard deviation across seeds 5 to 9 of each Primary's newest knot at each fold and
  cutoff, added to each component's variance for every seed's forecast.

## Standing practices followed and departures

- **Followed:** monthly-or-coarser refits (the coarse harness), cleaned data, the D15 and D18 cut
  before 2025-03, saved weights, reuse of stored fits (O0), a day-ahead and a 14-day example plot,
  tails and cold nights, `draft` PR, the STANDING RULE on weather timing (asserted row by row, one
  example printed per arm), no lag column (asserted by the input count), the newest knot fitted at
  the refit cut (asserted by the cut checks of experiment 565's harness).
- **Departure:** July 2026 stays unused; the five folds are the five of experiment 565.
- **Earlier advice used:** carry slow change as an offset on the component means (experiment 565),
  and do not expect a free offset to find steps of 2% to 6% of capacity (experiment 573).

## Order, cost, and stop rules

Known-answer check and Q1 first (nearly free), then O1 to O3 on seeds 0 and 1, then Q2, Q3,
Q-const, then Q4. The brief's ceiling is 3 GPU-h; the manager is told before it is exceeded. Cuts,
in order: the passers' extra seeds, O3, Q4.
