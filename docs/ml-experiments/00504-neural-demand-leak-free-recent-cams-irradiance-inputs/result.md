---
id: "00504"
slug: neural-demand-leak-free-recent-cams-irradiance-inputs
title: >-
  Do satellite irradiance values that a live service could have at forecast time lower the pooled
  demand neural network's error at the eight Primaries whose meters report apparent power only?
theme: demand-inputs
also_relevant_to:
  - solar-disaggregation
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fine-tuned neural networks and forecasts under this experiment's private data
  folder only; experiments 425, 449, 470, 480, and 494 are read and never written. Serving path and
  degradation rule: not touched (this repository holds neither). Designs: more than one would test
  the hypothesis (which days feed the columns, and whether to re-pretrain). The plan fixes
  whole-day columns, a 36-hour availability rule, and fine-tuning from the stored pretrained
  weights before any score is read. Callers: the fit script imports experiments 397 and 425 and
  patches their input builders. The scoring script imports experiments 470 and 494.
hypothesis: >-
  Giving the pooled demand neural network (experiment 425's arm A, which already reads the ensemble
  forecast of shortwave radiation) three extra inputs built only from Copernicus Atmosphere
  Monitoring Service (CAMS) satellite irradiance that a live service would have at the forecast
  origin lowers its continuous ranked probability score (CRPS) at days 1 to 10 at the eight
  Primaries that report only apparent power in megavolt-amperes (MVA): D2, D3, D4, D8, D12, D19,
  D21, and D22. The reason is that recent observed sunshine says how much embedded solar generation
  hides in those readings, which the ensemble forecast does not.
prediction: >-
  Written before any score is read. P1. A-cams lowers pooled CRPS at days 1 to 10 at the eight by
  0.05 to 0.15 points of effective capacity against K3 (the quantile average of the stored arm A
  seeds 0 to 2). The XGBoost model of experiment 480 gained 0.355 points from realised CAMS
  columns, but it had no irradiance input of its own, so a much smaller neural gain is expected.
  P2. The gain is largest at D12, on sunny half-hours (CAMS clear-sky index at least 0.7, 10:00 to
  15:00), and at days 2 to 5, where three-day-old observed cloudiness still carries information and
  the ensemble shortwave is no longer near-perfect (a hypothesis, not a measurement). P3. The
  realised-CAMS arm (A-cams-real, an upper bound) gains more than A-cams. P4. The one-number
  cloudiness arm (A-cams-lag-only) gains between 0 and A-cams. P5. At the signed-MW six and on cold
  winter nights, A-cams loses no more than 0.02 points of CRPS, and its q99 pinball loss is within
  5% of K3's at both groups.
decision_rule: >-
  1. Cadence: monthly refits at five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0
  to 2, fine-tuned from experiment 425's stored pretrained weights of the same month and seed.
  2. A-cams and K3 are each pooled by the quantile average of their seeds 0 to 2 at each fold.
  3. The gate is A-cams minus K3, CRPS at days 1 to 10 at the eight, with a Bonferroni interval
  from an origin-day block bootstrap within months (20,000 resamples would leave about 40 draws in
  each tail at a count of 12, so the scorer is run with 100,000). The bootstrap departs from the
  brief's month-block bootstrap, which has only 5 blocks, and matches experiment 494.
  4. The Bonferroni count is 12 (3 arms x 2 groups x 2 metrics, CRPS at days 1 to 10 and q99
  pinball loss), as in experiment 494; the brief's count of arms x groups x lead bands becomes
  arms x groups x metrics because days 1 to 10 is the only lead band counted. All three arms get
  Bonferroni intervals, and only A-cams is gated.
  5. The minimum detectable effect (MDE, experiment 428's definition with k = 3 seeds and five
  folds, from the spread of the single-seed scores of the stored arm A) is printed before any
  contrast; experiment 449 gives about 0.08 at the eight.
  6. The leak-free columns pass when the gate interval lies wholly below zero and the point
  estimate is at least MDE below zero. The threshold is MDE, not zero, because the change adds a
  CAMS feed and a daily lookup to serving, which the live service does not have today.
  7. A gain whose interval lies wholly below zero but whose point estimate is smaller than the MDE
  is reported as a small gain.
  8. If the interval includes zero and its lower end is above -0.053 points (15% of experiment
  480's 0.355), the page reports no detectable gain at this design, with fine-tuning only, so the
  neural network may already carry most of the irradiance information the XGBoost model lacked.
  If the interval includes zero and its lower end is at or below -0.053, the result is
  inconclusive.
  9. A "no loss" claim is an equivalence claim, made for the signed-MW six (CRPS and q99 pinball
  loss) and the eight (q99 pinball loss) only when the upper Bonferroni bound is at most 0.02
  points. Where the printed MDE of the group exceeds 0.02, as the queue-order check expects at the
  six, the claim is reported as inconclusive without being tested.
  10. Descriptive and not gated: A-cams-real and A-cams-lag-only, D12 alone and the eight without
  D12, lead bands, sunny-midday rows, cold winter nights, the 12 to 36 hour band, tails,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, a prediction-only check with the new
  columns set to their training mean, and a prediction-only check with the columns taken from
  one more day earlier.
  11. Audit trigger 4 of the `ml-experiment` skill: if the eight without D12 flips the gate
  verdict, the audit runs before merge.
  12. Every score is selection-biased, because the five folds that choose the candidate also score
  it. Rows with a valid time on or after 2026-07-01 are not scored; July 2026 stays unused, and any
  later July scoring starts 2026-07-02.
  13. The experiment does not touch the sign problem of the MVA-only Primaries and leaves the
  decision on the neural MVA experiment to the maintainer.
benchmark: >-
  K3, the quantile average of arm A seeds 0 to 2 of experiment 425 (stored, same pretrained
  weights, same rows). Also shown: the stored five-seed pool and seed 0 of arm A, `manual_heuristic`,
  and the climatology.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at days 1 to 10; q01, q05, q95, and q99 coverage and pinball
  loss; avoidable-curtailment cost at the signed-MW six.
---

**This page is the plan; it is rewritten with the results.** We ask whether the demand neural
network forecasts better when it can see how sunny the last few wholly observed days were, as
measured by satellite. We give the network three such inputs, fine-tune it, and compare its error
with that of the stored network without them, at the eight Primary substations whose meters report
apparent power and so hide embedded solar generation. The result is not yet known.

## Plan

**The XGBoost gain from realised satellite irradiance is an upper bound.**
[Experiment 480](../00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/result.md)
gave the XGBoost demand model two columns of realised irradiance from the
[CAMS radiation service](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries)
and lowered CRPS at days 1 to 10 by 0.355 points at the eight and 0.476 at the signed-MW six. The
columns hold realised irradiance at every lead, and CAMS arrives about a day late. The twin
[experiment 408](../00408-xgboost-signed-label-and-cams-columns-at-mva-only-primaries/result.md)
found 0.359 at the eight, with 15 of 16 Primaries improved (the interval at D15 includes zero).
Experiment 504 asks how much of that gain survives for a neural network that already reads the
ensemble shortwave, when only values a live service could have are used. The gain concerns the
magnitude score, not the sign of reverse flow.

**The CAMS columns use only whole days that are at least 36 hours old at the origin.** Origins are
09:00 UTC. The CAMS table (`cams_demand_points.parquet`: hourly irradiance and clear-sky irradiance
per Primary, hour-ending stamps, 2019-09 to 2026-09-10) has no availability stamp, so the delay
cannot be measured from it, and the plan assumes 36 hours. For an origin at 09:00 UTC on date D0,
the cutoff is 21:00 UTC on D0-2, and the latest wholly available day is D0-3. That day ends at
00:00 UTC on D0-2, 57 hours before the origin. The
[lag-features study](../../studies/forecasts/lag-features.md) found the whole previous
day available at 09:00 UTC after February 2026 (two days before that), so serving can be one to two
days fresher than this plan. The one-day-earlier check measures sensitivity to staleness. The same
rule applies to every training row of the fine-tune, which never sees realised CAMS values, and the
fold's training end bounds both the pretrained weights and the fine-tune rows.

**Three columns are built from those days.** An hour-ending stamp belongs to the hour before it, so
an hour stamped 00:00 UTC is the last hour of the previous day. Half-hourly irradiance comes from
`half_hour_sun_and_sky`, as in experiment 480. Column (a) is the CAMS irradiance at the target
half-hour's time of day on day D. Column (b) is the mean of (a)'s quantity over days D, D-1, and
D-2 at the same time of day. Column (c) is the sum of the day's CAMS irradiance divided by the sum
of its clear-sky irradiance, a cloudiness state shared by all leads of an origin.

**The experiment fits 45 neural networks.** Five folds, three seeds, and three arms give 45
fine-tunes. Arm K3 is the stored arm A and needs no fit. A-cams adds columns (a) to (c).
A-cams-real adds experiment 408's realised unit photovoltaic shape and clear-sky index of the
target half-hour (an upper bound that serving cannot supply, report only). A-cams-lag-only adds
column (c) only.

**Fine-tuning from the stored weights may fail to learn the new columns.** Each arm's neural
network is experiment 425's pooled plain network. Its weights are loaded from the stored
pretrained arm A network of the same month and seed, with zero-initialised first-layer weights
inserted for the new inputs (before the Primary embedding). The network is then fine-tuned on the
fold's ensemble (ENS) rows at a learning rate of 2e-4 with experiment 425's loop, and the new
columns are standardised on the fold's training rows. A null result may therefore mean the columns
were not learned in the fine-tune and not that they carry no information, and the page says so.
Masked training (a random 20% of rows with the columns missing, as the fallback for a late feed)
is not run here, because it would need a fourth arm (15 more fine-tunes, about 0.15 to 0.25
GPU-hours). The prediction-only check with training-mean columns measures that loss instead.

**Rows, cleaning, and standing practices follow experiments 470 and 494.** Rows are experiment
470's intersection, with flagged exact-zero half-hours dropped (the count is printed per fold), the
D15 and D18 start cut of 2025-03-01, and widening 1.15. The weather inputs are the ensemble mean and
spread. Monthly refits, cleaned data, saved neural networks, stored predictions as the control,
tails, avoidable-curtailment cost, and example forecasts all follow the skill. The screen on a few
folds is these five folds, and the calendar inputs are experiment 425's unchanged. The plan follows
the advice of [experiment 494](../00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md)
(report an interval across folds beside the origin-day interval, and reuse stored networks) and
departs from none. Load-shift event plots are not needed, because the experiment does not score
events.

**The write-up will carry a serving table and a Checks list.** The serving table lists a CAMS feed
that arrives about a day late (not ingested today), one stored daily value per Primary for column
(c) and three per half-hour for (a) and (b), unchanged neural networks per pass, and the proposed
fallback of setting the columns to their training mean and widening the bands. The Checks list
gives the median forecast minus the mean forecast per site, the error of the mean forecast beside
each CRPS and q99 contrast, and which audit triggers fired.

**Compute is 0.43 to 0.75 GPU-hours.** One fine-tune takes 35 to 60 GPU seconds
([experiment 481](../00481-seed-averaging-diminishing-returns-for-demand-and-wind/result.md)), and
the first fit is timed and reported. One GPU process under 8 GB and one feeder core run at a time,
and `free -g` shows more than 25 GB available before each launch.

**Reviews.** A Sonnet agent reviews the plan, another reviews the code before the first fit, a
third checks the write-up against the printed report, and a fourth reviews the prose.
