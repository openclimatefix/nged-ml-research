---
id: '00369'
slug: d8-cross-site-step-rate-against-stack-gap
title: >-
  Does a Primary's rate of generator-like steps predict how much the best-guess stack trails BG2
  across the 14 scored Primaries?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - calibration-and-uncertainty
  - lsc-detectors
status: planned
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00369-d8-cross-site-step-rate-against-stack-gap/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (a MW or a capacity-share threshold, a window of the scored year or of all history), so
  the plan fixes the detector of experiment 361 unchanged and reports the alternatives as post hoc
  checks. Callers: the scripts import experiments 252 and 361 and call reviewed `studies.*` modules
  without changing them. Asset graph: not touched. No model is fitted, so it runs in minutes.
hypothesis: >-
  The best-guess stack trails BG2 most at the Primaries that show the most generator-like steps, so
  D8's gap is part of a pattern across Primaries and not only a property of D8's residual. If the
  hypothesis holds, a step detector's rate is a cheap flag for the Primaries where the stack's
  uncertainty bands need widening.
prediction: >-
  Written before any score or step count is read. (P1) The Spearman rank correlation, over the 14
  scored Primaries, between a site's step rate (steps of at least 1.5 MW and 6 hours per month in the
  scored year, rises and falls together) and its stack-minus-BG2 quantile-score gap is at least 0.5.
  (P2) D8, D12, and D22, the three sites where the stack trails BG2 by most in experiment 252, are
  all among the five sites with the highest step rate. (P3) On time-shuffled copies, every site's
  step rate falls to under half its real rate.
decision_rule: >-
  No model is fitted, so the threshold for porting is zero: the result is a diagnostic. `supported`
  when P1 and P2 both hold. `partly supported` when exactly one holds. `not supported` when neither
  holds. P3 is a placebo check: if it fails, the detector is reading the weekly and seasonal shape
  of demand and no verdict is drawn from P1 and P2. Fourteen points make a weak test: the write-up
  gives a 95% interval from resampling sites, and says that an interval spanning zero leaves the
  correlation unproven. Control gate: the stored D8 gap, stack (experiment 252 arm A0) minus BG2,
  reproduces +3.619 points before any correlation is read.
benchmark: >-
  BG2, as stored by experiment 87. The stack is experiment 252 arm A0 at D2, D3, D4, D8, D12, D19,
  D21, and D22, and experiment 155 arm C (recipe R alone) at D5, D9, D10, D11, D14, and D16
row_set_id: fold_a_daily_gap_v1
metric: >-
  Spearman rank correlation of step rate against stack-minus-BG2 quantile score (twice the mean
  pinball loss over 11 quantiles, in points of effective capacity); rank of D8, D12, and D22 by step
  rate; step rate on shuffled copies
code: ml-experiments/00369-d8-cross-site-step-rate-against-stack-gap/
---

# Does a Primary's rate of generator-like steps predict how much the best-guess stack trails BG2?

Planned. The write-up follows the run.
