---
id: "00577"
slug: static-between-steps-offsets-and-probabilistic-offset-widening
title: >-
  Do offsets that stay constant between sparse steps, and offsets whose uncertainty widens the
  forecast, improve the output-side slow offset (D5) of the modular lag-free demand network?
theme: retraining-and-training-history
also_relevant_to:
  - calibration-and-uncertainty
status: finished
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
  adds a penalty, a trained variance, or a mixture component. Bonferroni count 28 (22 pre-registered
  plus 6 for Q5, the scale offset added during the run; Design 1: 3 arms
  x 2 groups = 6; Design 2: CRPS of Q1 to Q4 x 2 groups = 8, Q2 and Q3 against Q-const x 2 groups
  = 4, tail loss of Q2 and Q3 x 2 groups = 4). The threshold of 0.03 pays for one extra penalty or
  one stored variance per Primary. Static-ness, calibration, sharpness, known-answer errors, slices,
  and the per-Primary table are descriptive.
benchmark: >-
  D5 of experiment 565 (coarse refits on days 0 and 14; stored seeds 0 and 1 and fresh seeds 5 to
  9) as O0 and Q0, with the coarse M0 of experiment 565 as a second reference.
verdict: refuted
best_arm: >-
  D5 itself. No arm beats D5 at both groups: the best point estimates are O2 on 5 fresh seeds
  (-0.008 at the eight MVA Primaries, +0.054 at the six MW Primaries) and Q4-O2 on seeds 0 and 1
  (-0.045 and +0.015), none with a Bonferroni interval that excludes zero.
headline: >-
  A steadier offset (total-variation or gated-jump penalty) costs no accuracy but gains none, and
  carrying the offset's uncertainty into the mixture does not improve CRPS, tails, or calibration;
  the one pre-registered widening that carries the jump risk (Q3) makes the forecast clearly worse.
lessons: >-
  Three of the pre-registered predictions failed: no total-variation arm reached the 70% share of
  static month pairs (best 60%), no arm gained 0.03 CRPS, and Q3 worsened CRPS by 0.15 to 0.18
  instead of improving the tails. The known-answer check showed that a larger penalty cuts the
  offset's error in a world with no true offset but also biases recovered step sizes low, so the
  penalty trades a calmer path for smaller steps.
take_home_message: >-
  The slow offset (D5) is already as good as these refinements can make it on this data. Making the
  offset steadier or wider does not help the forecast, so keep D5 as it is; the penalty and widening
  ideas are not worth carrying into the next model release.
advice:
  - >-
    Keep the D5 offset with its squared first-difference penalty; the total-variation and gated-jump
    penalties did not improve CRPS (Results).
  - >-
    Do not add a jump-risk mixture component or a variance growing with days since the refit: Q3 as
    pre-registered worsened CRPS by 0.15 to 0.18 and made the 90% interval about 11 points wider
    (Results).
  - >-
    Judge a penalty by CRPS and by recovered step sizes together; a penalty that cuts the offset's
    error in a zero-offset world also shrinks real steps (Known-answer table).
production_status: not-for-production
adopted_in: none
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 24-120 h at the eight MVA Primaries and the six MW
  Primaries; summed 1% and 99% pinball loss; coverage of the 80%, 90%, and 98% central intervals.
---

**We asked two questions about the slow offset (D5) that the modular lag-free demand network adds to
the forecast's mean: does an offset that stays constant between sparse steps forecast better, and
does carrying the offset's uncertainty into the forecast improve it?** We tested three
total-variation or gated-jump penalties (O1 to O3) and six ways to carry the uncertainty (Q1 to Q5
and a placebo), on the same 16 Primary substations (14 scored) and five test months as [experiment
565](../00565-drift-embedding-variants-six-designs/result.md), with 5 seeds for the arms that showed
a point gain and 2 for the rest. Neither idea improved the forecast. A steadier offset costs no
accuracy, and widening the forecast by the offset's uncertainty changes CRPS by under 0.03 points of
capacity.

**Verdict: refuted.** Take-home: D5 is already as good as these refinements make it, so the next
step is to keep D5 unchanged. Every interval below is a Bonferroni month-block interval over 28
contrasts (22 pre-registered plus 6 for Q5) with 5 blocks, which is near the minimum to maximum of
the five folds. The rest of this page says "interval excludes zero" only where it does.

![Each arm's CRPS minus D5's at 24 to 120 h](contrasts_vs_d5.svg)

## Results

- **Absolute CRPS of D5 at 24-120 h (seeds 0 and 1).** 4.651 points of capacity at the eight MVA
  Primaries and 3.376 at the six MW Primaries. On seeds 5 to 9, D5 is 4.646 and 3.475.
- **Steadier offsets (O1 to O3), seeds 0 and 1, CRPS minus D5 (minimum detectable effect, MDE).** O1
  +0.017 at the eight (MDE 0.118) and +0.021 at the six (0.120); O2 -0.042 (0.128) and +0.018
  (0.229); O3 +0.118 (0.178) and -0.025 (0.149). No interval excludes zero. Penalties picked on fold
  1's pretraining NLL: O1 lambda 3000, O2 3000, O3 1000.
- **O2 on fresh seeds 5 to 9.** CRPS minus D5 is -0.008 at the eight (interval -0.068 to +0.082, MDE
  0.102, 2 of 5 folds positive) and +0.054 at the six (-0.085 to +0.357, MDE 0.123, 2 of 5
  positive). O2 narrows the 90% interval by 0.9 points of capacity (eight) and 1.6 (six), and the
  80% interval covers 2.5 and 3.9 points less than D5's (interval excludes zero). Q4-O2 on the same
  seeds: -0.010 and +0.067.
- **Offset-uncertainty widening, seeds 0 and 1.** Q1 (spread across seeds) is -0.000 and -0.001;
  Q-const (placebo) is -0.000 and -0.001; Q2 (variational newest knot) is +0.028 and +0.000; Q4-O2
  is -0.045 and +0.015. None meets the 0.03 threshold, and no interval excludes zero.
- **Q3 (jump-risk widening) as pre-registered is worse.** CRPS is +0.183 at the eight (0.128 to
  0.217) and +0.152 at the six (0.085 to 0.199), the summed 1% and 99% pinball loss rises by 0.158
  and 0.201, and the 90% interval is 11 points of capacity wider; all 5 folds are positive. Q3b, the
  same widening with statistics taken from O2's own jumps, is a **post hoc** variant and is neutral.
- **Q5 (slow per-Primary scale offset).** CRPS minus D5 is +0.007 (MDE 0.158) and +0.014 (0.130);
  the shuffled-across-Primaries placebo is +0.000 and +0.033, so the learnt scale knots are not
  better than shuffled ones. The knots do differ across Primaries and drift over time, and agree
  across seeds (correlation above 0.93), see the figure below.
- **Slices (24-120 h, CRPS minus D5, O2 / Q4-O2).** Leave-D8-out at the eight: -0.053 / -0.057. D8
  alone: +0.032 / +0.033. D14 alone at the six: +0.002 / +0.055. Rows with no labelled event: -0.053
  / -0.060 (eight), +0.018 / +0.015 (six). Cold nights (3 folds): -0.093 / -0.091 (eight) and -1.091
  / -1.118 (six, interval -2.072 to +0.424, 1 of 3 folds positive). Christmas week (1 fold only):
  -0.430 / -0.286 (eight) and +0.349 / +0.262 (six). No slice is shown to improve.
- **The offsets are not much steadier.** The share of month pairs without a labelled event whose
  offset moves by under 0.2% of capacity is 37% for D5, 60% for O1, 40% for O2, and 36% for O3. O1
  gains 23 points, short of the 70% target.

![Learnt offsets by arm, one line per Primary](offsets_by_arm.svg)

![Q5's learnt scale knots per Primary](scale_knots.svg)

![A day-ahead and a 14-day example forecast](example_forecasts.svg)

## Known-answer check

The penalties were fitted on [experiment
573](../00573-stepwise-output-offset-known-answer-step-recovery/result.md)'s cached network outputs
with nothing added (K0) and with two persistent steps of 10% to 20% of capacity per Primary (Kl).
Static-period RMSE is in percent of capacity, averaged over five folds.

| Arm | K0 RMSE | Kl RMSE | Steps recovered | Mean step-size bias |
|---|---|---|---|---|
| D5 | 2.89 | 1.33 | 8 of 45 | -52% |
| O1, lambda 3000 | 2.77 | 0.85 | 19 of 45 | -19% |
| O1, lambda 30000 | 0.87 | 7.59 | 10 of 45 | -33% |
| O2, lambda 3000 | 1.81 | 8.24 | 40 of 45 | -70% |
| O2, lambda 30000 | 0.47 | 10.58 | 0 of 45 | n/a |

The pre-registered prediction (O1 or O2 cuts the static-period RMSE by at least 40% in K0) fails at
the picked penalty (4% for O1, 37% for O2) and holds only at lambda 10000 or more for O2 and 30000
for O1. A larger penalty cuts K0's error but recovers fewer steps and shrinks the ones it recovers.

## Checks and caveats

- **Triggers fired.** Trigger 2 (Q3's result contradicted its prediction) and trigger 1 (the 40%
  known-answer prediction failed for O1). Neither result is surprising once the known-answer table
  is read, so no separate audit was run.
- **Preliminary penalty grid.** The first grid of penalties was too weak (validation NLL 0.36 and
  0.40 against D5's 0.325) and was replaced; its fits are kept in the private data folder and are
  not used here.
- **Not run.** The absorbed-share and stress-run fits, and the raw (unwidened, factor 1.0)
  sensitivity: every score includes the 1.15 widening about the median that all stored arms use.
- **Event labels are incomplete,** so the "no labelled event" slice may hold unlabelled events, and
  the static-ness shares are lower bounds on how often the offset moved for a real reason.
- **Compute.** About 2.6 of the 3 GPU-hours; the extra seeds were run for O2 and Q4-O2 only.
- **Weights.** Pretrained and fine-tuned weights for every arm are in `models/` and `pretrained_*`
  under this experiment's private data folder; the final candidate is D5 of experiment 565.
