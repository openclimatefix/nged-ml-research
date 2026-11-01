---
id: "00466"
slug: wind-and-solar-recency-weighting
title: >-
  Does weighting recent training rows more raise the quantile score of the wind and solar networks?
theme: generator-forecasting
also_relevant_to:
  - retraining-and-training-history
status: finished
kind: change
protocol: wind-v1
date: 2026-11-01
size: >-
  medium. Stored: predictions of 15 wind fits and 84 solar fits under
  data/ml-experiments/00466-wind-and-solar-recency-weighting/ (private). Serving path: not touched.
  Degradation rule: not touched. Designs: a per-row weight on the training loss is the one design
  tested; a training window that starts after an ECMWF ensemble upgrade is not testable (below).
  Callers: the folder's scripts, which import experiments 310, 429, 430, and 461. One Sonnet code
  review before the first scored fit, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  If each training row's loss is weighted by 0.5 to the power of its age at the refit divided by a
  half-life, then the quantile score of the wind neural network `member_plain` and the solar neural
  networks
  `g0_onepass` and `a2_member` improves, because farms change efficiency slowly and each upgrade of
  the ECMWF ensemble shifts the statistics of the inputs.
prediction: >-
  Written before any fit. No half-life improves the score by the margin at any technology. The
  prior is low: experiment 443 found no gain from half-lives of 0.5 and 1 year for solar
  `g0_onepass` (+0.00% and -0.11%), and experiment 464 found that scaling by trailing efficiency
  helps no farm. Wind trains on 12.5 months, so only the 9-month half-life (weights 0.38 to 1) and
  the 4.5-month half-life (0.15 to 1) differ from equal weights. Solar trains on 17 to 26 months,
  so 2 years gives weights 0.47 to 1 and 3 years gives 0.60 to 1. The Cycle 49r1 indicator has
  no training signal for Cycle 50r1, so it can only show whether Cycle 49r1 mattered.
decision_rule: >-
  An arm passes at a technology (or at one series, reported separately) if its score in MW, after
  table P and experiment 451's clip, is lower than the equal-weight reference by more than the
  margin (wind 0.10 points of capacity, applied per series as 0.10 percent of its p99.9 capacity
  in MW; solar 1% of the pooled reference score), with the upper bound of the 95% month-block
  interval of (arm minus reference) below zero, a tail harm of at most 0.02 on the pinball loss at
  q95 and q99 and at most 0.5 points on their coverage, and a gain in a majority of scored months.
  If no half-life passes at the wind farms and none at the solar series, the neural networks are
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
verdict: not supported
best_arm: >-
  Solar `g0_onepass` with a 2-year half-life: pooled score 0.6646 MW against 0.6655 MW with equal
  weights (-0.14%, 95% interval -0.36% to +0.08%), a seventh of the 1% margin. No wind arm and no
  `a2_member` arm improves.
headline: >-
  No half-life improves the wind or solar score by the margin. Wind scores 0.22% to 0.25% worse
  with shorter half-lives, solar `g0_onepass` 0.14% better at 2 and 3 years (inside seed noise),
  and `a2_member` 0.14% to 0.17% worse.
lessons: >-
  Within the half-lives that the 12.5 to 26 months of training allow, the wind and solar neural
  networks
  are insensitive to recency, so the line closes. A Cycle 49r1 indicator also changes nothing
  pooled.
take_home_message: >-
  Weighting recent training rows more does not improve the wind or solar forecasts. The wind
  network scores 0.22% to 0.25% worse with half-lives of 9 and 4.5 months, and the best solar arm
  gains 0.14%, which is less than the difference between two fits of the same network. The
  wind and solar neural networks are insensitive to recency within the range that the short history
  allows, so this line closes.
advice:
  - >-
    Train the wind and solar networks with equal row weights; no half-life from 4.5 months to 3
    years passes the margin (figure 1).
  - >-
    Do not test a recency weight or an upgrade cutoff again until the ECMWF ensemble history is
    longer than 2 years, because the weights now differ little from equal weights.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00466-wind-and-solar-recency-weighting/
---

**Take-home: weighting recent training rows more does not improve the wind or solar forecasts, so
the line closes. The wind neural network scores 0.22% to 0.25% worse with half-lives of 9 and 4.5
months.
The best solar arm gains 0.14%, which is less than the difference between two fits of the same
network and a seventh of the 1% margin.**

Farms change efficiency slowly, and each upgrade of the ECMWF ensemble shifts the statistics of the
inputs, so recent rows might train a better network. We multiplied each training row's loss by
`0.5 ** (age / half_life)` and refitted the wind neural network `member_plain` and the solar neural
networks
`g0_onepass` and `a2_member` at the p99.9 capacity of experiment 461. Farm A to C and Solar A to F
are the anonymised series of experiments 429 and 430. The training history is 12.5 months for wind
and 17 to 26 months for solar, so only the shorter half-lives differ much from equal weights.

## Figures

![Pooled score change against equal weights, every arm](figure_1_score_change.svg)

![Pooled score change by scored month](figure_2_monthly_change.svg)

## Results

Contrasts are arm minus the equal-weight reference in MW after table P (a per-series tail
correction) and experiment 451's clip (an upper bound),
pooled over a technology's series, so a positive change is worse. Rows are the half-hours at true
days 1 to 10 (lead 24 to 240 hours) that experiment 210's mask keeps for wind, and all scored
half-hours for solar. Intervals are 95% month-block bootstrap intervals over 11 scored months.

- **No arm passes.** The margin is 1% of the pooled reference score for solar (about 0.0067 MW) and
  0.10 points of capacity for wind. The best arm gains 0.0009 MW.
- **Wind `member_plain`** (5 seeds, reference 1.1744 MW). Half-life 9 months: +0.0026 MW (+0.22%;
  -0.0008 to +0.0060), better in 3 of 11 months. Half-life 4.5 months: +0.0029 MW (+0.25%; -0.0033
  to +0.0088), better in 5 of 11 months. Per farm, 9 months: Farm A +0.15%, Farm B +0.44%, Farm C
  -0.01%.
- **Solar `g0_onepass`** (3 seeds, 4 refits, reference 0.6655 MW). Half-life 0.5 year: +0.11%
  (-0.0024 to +0.0039 MW). At 1 year: -0.04% (-0.0032 to +0.0020). At 2 years: -0.14% (-0.0024 to
  +0.0005), better in 7 of 11 months. At 3 years: -0.14% (-0.0020 to +0.0002), better in 7 of 11
  months. At 2 years, Solar C is the one series whose 95% interval excludes zero in the favourable
  direction (2 years: -0.20%, -0.0028 to
  -0.0001 MW, better in 9 of 11 months), and its adjusted interval does not.
- **Solar `a2_member`** (reference 0.6627 MW). At 2 years: +0.17% (-0.0010 to +0.0032 MW). At 3
  years:
  +0.14% (-0.0005 to +0.0030). Solar B at 2 years is +0.32% worse with an interval that excludes
  zero (+0.0004 to +0.0043 MW).
- **Minimum detectable effect** (mean gap between two single-seed fits of the reference, stated
  before any arm was scored): wind 0.0077 MW, `g0_onepass` 0.0050 MW, `a2_member` 0.0036 MW. Every
  pooled contrast is smaller than the wind and `g0_onepass` values.
- **Cycle 49r1 indicator (diagnostic, `g0_onepass`).** Pooled +0.01% (-0.0023 to +0.0025 MW). Solar
  A improves 0.36% in 9 of 11 months (-0.0036 to +0.0002 MW), and Solar E worsens 0.20%. The
  indicator has no training signal for Cycle 50r1, and the indicator is not adopted.
- **Tails after table P, pooled.** Coverage is the share of actuals at or below the quantile. Wind,
  equal weights to 9 months:
  share above q99 0.88% to 0.86%; q95 coverage 95.13% for both; q99 pinball loss 0.0641 to 0.0643
  MW.
  Wind at 4.5 months: share above q99 0.94%, q99 pinball 0.0646 MW. Solar `g0_onepass`, equal
  weights to 2 years: share above q99 1.41% to 1.37%; q95 coverage 94.58% to 94.82%; q99 pinball
  loss 0.0247 for both. `a2_member` equal weights to 3 years: share above q99 1.50% to 1.48%. No
  half-life arm changes a pooled tail coverage by more than 0.4 points or a pinball loss by more
  than
  0.0009 (wind q95 at 9 months); the Cycle 49r1 indicator moves `g0_onepass` q95 coverage by 0.65
  points.
- **Bonferroni count.** 57 intervals (arms times pooled and per series); at the adjusted level no
  contrast excludes zero in the favourable direction.
- **Rows and seeds.** Equal across arms: wind 54,899 half-hours (Farm A 18,831, B 18,403, C
  17,665); solar 345,510 half-hours per arm (A 60,210, B 60,039, C 60,885, D 54,864, E 49,770,
  F 59,742). Wind 5 seeds; solar 3.
- **The 12 to 36 hour band.** The scored rows start at lead 24 hours, so the band holds only leads
  24 to 36 hours (wind 3,361 half-hours, solar 18,014). The scores for that band were not separated,
  because no
  pooled contrast is near the margin.

## Caveats

- **Reproduction gates.** With all weights 1 the wind fit (seed 0) and the solar `g0_onepass` fit
  (seed 0, refit 2025-09) reproduce experiment 461's stored forecasts bit for bit. The four other
  wind reference seeds are copies of experiment 461's p99.9 fits.
- **Short history.** Wind trains on 12.5 months, so a half-life of 3 years or 1.5 years is nearly
  equal weighting, and those arms were dropped. A training window that starts after Cycle 49r1 was
  not tested, because it leaves 5 months of wind rows and 10 to 19 months of solar rows. Cycle
  50r1 (2026-05-12) falls inside the scored months, and no month's contrast shows a step there
  (figure 2); the figures do not mark it.
- **Scored years.** Wind spans 2025-08 to 2026-06 and solar 2025-10 to 2026-08, so the contrast by
  month stands in for the contrast by year.
- **Experiment 443's p99 forecasts are not reused.** Experiment 443's 0.5-year and 1-year arms were
  refitted at p99.9 here,
  and agree with its p99 result: neither gains.
- **Cleaned-zero rescore.** Experiment 432's cleaned-zero row set was not scored for wind.
- **Production cost.** None for a half-life, which is a training-time weight. An indicator would
  need a retrain at each ECMWF cycle.

## Checks

- **Reviews.** A fresh Sonnet reviewed the code before the first fit and found no defect; it
  asked for three low items (a stale docstring, a check on unknown labels, and an assertion), all
  fixed.
- **Median bias.** The pooled mean of the median forecast minus the actual after table P is -0.614
  MW for wind with equal weights and with 9 months, and +0.115 MW (equal weights) and +0.109 MW (3
  years) for `g0_onepass`, so the bias does not drive any contrast.
- **Audit triggers.** None fired: no contrast is near the margin, so no audit was run.
