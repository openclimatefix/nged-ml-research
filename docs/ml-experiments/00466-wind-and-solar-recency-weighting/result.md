---
id: "00466"
slug: wind-and-solar-recency-weighting
title: >-
  Does weighting recent training rows more raise the quantile score of the wind and solar networks?
theme: generator-forecasting
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: change
protocol: wind-v1
date: 2026-11-01
size: >-
  medium. Stored: predictions of 15 wind fits and 84 solar fits under
  data/ml-experiments/00466-wind-and-solar-recency-weighting/ (private). Serving path: not touched.
  Degradation rule: not touched. Designs: a per-row weight on the training loss is the one design
  tested; a training window that starts after a forecast-model upgrade is not testable (below).
  Callers: the folder's scripts, which import experiments 310, 429, 430, and 461. One Sonnet code
  review before the first scored fit, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  If each training row's loss is weighted by 0.5 to the power of its age at the refit divided by a
  half-life, then the quantile score of the wind network `member_plain` and the solar networks
  `g0_onepass` and `a2_member` improves, because farms change efficiency slowly and each upgrade of
  the ECMWF ensemble shifts the statistics of the inputs.
prediction: >-
  Written before any fit. No half-life improves the score by the margin at any technology. The
  prior is low: experiment 443 found no gain from half-lives of 0.5 and 1 year for solar
  `g0_onepass` (+0.00% and -0.11%), and experiment 464 found that scaling by trailing efficiency
  helps no farm. Wind trains on 12.5 months, so only the 9-month half-life (weights 0.40 to 1) and
  the 4.5-month half-life (0.09 to 1) differ from equal weights. Solar trains on 17 to 26 months,
  so 2 years gives weights 0.47 to 1 and 3 years gives 0.60 to 1. The Cycle 49r1 indicator has
  no training signal for Cycle 50r1, so it can only show whether Cycle 49r1 mattered.
decision_rule: >-
  An arm passes at a technology (or at one series, reported separately) if its score in MW, after
  table P and experiment 451's clip, is lower than the equal-weight reference by more than the
  margin (wind 0.10 points of capacity, applied per series as 0.10 percent of its p99.9 capacity
  in MW; solar 1% of the pooled reference score), with the upper bound of the 95% month-block
  interval of (arm minus reference) below zero, a tail harm of at most 0.02 on the pinball loss at
  q95 and q99 and at most 0.5 points on their coverage, and a gain in a majority of scored months.
  If no half-life passes at the wind farms and none at the solar series, the networks are
  insensitive to recency within the tested range and the line closes. A half-life that passes at
  some series and not others is reported per series, not pooled. The margins are the ones
  experiments 450 and 443 set; a half-life costs nothing in production (a training-time weight),
  so the threshold only guards against a gain inside seed noise. The minimum detectable effect is
  stated before any score. The Cycle 49r1 indicator is a diagnostic and is not adopted.
benchmark: >-
  Experiment 461's p99.9 fits with equal weights: wind `member_plain` (5 seeds, refitted here with
  all weights 1 as a gate, which reproduces experiment 461's stored forecasts bit for bit) and
  solar `g0_onepass` and `a2_member` (3 seeds, 4 quarterly refits, stored by experiment 461).
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score in MW (mean over the 11 levels) at true days 1 to 10, with coverage and pinball
  loss at q01, q05, q95, and q99, per series, with the 12 to 36 hour lead band reported separately.
code: ml-experiments/00466-wind-and-solar-recency-weighting/
---

**Take-home: not yet known.** Planned.

## Question

Farms change efficiency slowly, and each upgrade of the ECMWF ensemble forecast model shifts the
statistics of the inputs. Weighting recent training rows more could therefore lower the quantile
score of the wind and solar networks, for no production cost beyond a training-time weight.

## Arms

- **Weight.** Each training row's loss is multiplied by `0.5 ** (age / half_life)`, with age from
  the row to the fit's cutoff, rescaled to a mean of 1 over the rows the loss sees. Early stopping
  is unweighted.
- **Wind `member_plain`, Farm A to C.** One fit per seed on the 2024-04-01 to 2025-04-15 hindcast
  with the cutoff 2025-04-16, as experiments 429 and 461 do, scored on 53 Tuesday runs. Half-lives:
  none (reference), 9 months, and 4.5 months. 3 arms times 5 seeds is 15 fits.
- **Solar `g0_onepass`, Solar A to F.** Experiment 430's four quarterly refits (from 2025-09) and 3
  seeds. Half-lives 0.5, 1, 2, and 3 years: 4 times 3 times 4 is 48 fits. The 0.5 and 1 year arms of
  experiment 443 are refitted at the p99.9 capacity, so every arm shares one base.
- **Solar `a2_member`.** Half-lives 2 and 3 years: 2 times 3 times 4 is 24 fits.
- **Cycle 49r1 indicator (diagnostic, solar `g0_onepass`).** One more 0/1 input that is 1 for
  ensemble runs initialised from 2024-11-12. 3 seeds times 4 refits is 12 fits. The stored ensemble
  history starts 2024-04-01, so only Cycles 49r1 (2024-11-12) and 50r1 (2026-05-12) fall inside it.
  The indicator has no training signal for Cycle 50r1.
- **Dropped.** A training window that starts after an upgrade is not tested, because a start after
  Cycle 49r1 leaves 5 months of wind rows and 10 to 19 months of solar rows. The 3-year and
  1.5-year wind half-lives are dropped, because 12.5 months of training make them nearly equal
  weighting.

## Method

- **Capacity.** p99.9 of each series' earlier power throughout (experiment 453's recommendation, the
  base of experiment 461). Experiment 443's p99 forecasts are not mixed in.
- **Reuse.** Experiment 461's stored equal-weight p99.9 forecasts are the solar references. Wind is
  refitted with equal weights so that every wind arm comes from one code path; the gate against
  experiment 461's stored forecasts is exact.
- **Scoring.** Experiment 461's scoring (table P, experiment 451's clip recomputed per capacity,
  experiment 210's idle-band mask for wind) in MW, which is the decision unit. The
  cleaned-zero rescore of experiment 432 is not applied.
- **Contrasts.** Each arm against the equal-weight reference, per series and pooled, per scored
  month and per scored year, with month-block bootstrap intervals, the Bonferroni count as a number,
  and absolute scores beside every contrast.
- **Production cost.** None for a half-life (a training-time weight). A retrain per ECMWF cycle if
  the indicator were adopted.
