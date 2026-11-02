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
  medium. Stored: fine-tuned networks and forecasts under this experiment's private data folder
  only; experiments 425, 449, 470, 480, and 494 are read and never written. Serving path and
  degradation rule: not touched (this repository holds neither). Designs: more than one would test
  the hypothesis (which days feed the columns, and whether to re-pretrain), so the plan fixes
  whole-day columns, a 36-hour availability rule, and fine-tuning from the stored pretrained
  weights before any score is read. Callers: the fit script imports experiments 397 and 425 and
  patches their input builders; the scoring script imports experiments 470 and 494.
hypothesis: >-
  Giving the pooled demand network (experiment 425's arm A, which already reads the ensemble
  forecast of shortwave radiation) three extra inputs built only from Copernicus Atmosphere
  Monitoring Service (CAMS) irradiance that a live service would have at the forecast origin lowers
  its continuous ranked probability score (CRPS) at days 1 to 10 at the eight MVA-only Primaries
  (D2, D3, D4, D8, D12, D19, D21, D22), because recent observed sunshine says how much embedded
  solar generation hides in the apparent-power readings, which the ensemble forecast does not.
prediction: >-
  Written before any score is read. P1. A-cams lowers pooled CRPS at days 1 to 10 at the eight by
  0.08 to 0.20 points of effective capacity against the three-seed pool of the stored arm A (K3).
  The XGBoost model of experiment 480 gained 0.355 points from realised CAMS columns, but it had no
  irradiance input of its own, so a smaller neural gain is expected. P2. The gain is largest at D12,
  on sunny half-hours (CAMS clear-sky index at least 0.7, 10:00 to 15:00), and at days 2 to 5,
  where the ensemble shortwave is least certain. P3. The realised-CAMS arm (A-cams-real, an upper
  bound) gains more than A-cams. P4. The one-number cloudiness arm (A-cams-lag-only) gains between 0
  and A-cams. P5. At the signed-MW six, A-cams changes CRPS by less than 0.10 points in either
  direction, and q99 pinball loss changes by less than 0.02 points at both groups.
decision_rule: >-
  Cadence: monthly refits, five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0 to 2,
  fine-tuning from experiment 425's stored pretrained weights of the same month and seed, with
  zero-initialised weights for the new columns. The gate is A-cams minus K3 (the quantile average
  of the stored arm A seeds 0 to 2), CRPS at days 1 to 10 at the eight, origin-day block bootstrap
  within months, 20,000 resamples, Bonferroni count 12 (3 arms x 2 groups x 2 metrics, CRPS days 1
  to 10 and q99 pinball loss; one lead band is counted, days 1 to 10). The minimum detectable
  effect (MDE, experiment 428's definition with k = 3 seeds, five folds) is printed before any
  contrast. A-cams adopts the leak-free columns when its Bonferroni interval at the eight lies
  wholly below -MDE. A gain whose interval lies below zero but not below -MDE is small but real and
  is reported as a small gain. If the interval includes zero and its lower end is above -0.05 (a
  gain of 0.15 of experiment 480's 0.355 would be 0.053), the page reports that the neural network
  already carries most of the irradiance information the XGBoost model lacked; if the interval
  includes -0.05, the result is inconclusive. A "no loss" claim at the six, on cold winter nights,
  or in q99 pinball loss is an equivalence claim, made only when the upper Bonferroni bound is at
  most max(0.02, MDE of that group) and reported as inconclusive when it is not; the printed MDE is
  larger than 0.02 at every group. Descriptive and not counted: A-cams-real and A-cams-lag-only,
  D12 alone and the eight without D12, lead bands, sunny-midday rows, cold winter nights, the 12 to
  36 hour band, tails, avoidable-curtailment cost at f = 0.9, 0.75, and 0.5, a prediction-only
  check with the new columns set to their training mean, and a prediction-only check with the
  columns taken from one more day earlier. Selection bias: every score is selection-biased because
  the five folds that choose the candidate also score it; July 2026 stays unused (any later July
  scoring starts 2026-07-02). The experiment does not touch the sign problem of the MVA-only
  Primaries and leaves the decision on the neural MVA experiment to the maintainer.
benchmark: >-
  K3, the quantile average of arm A seeds 0 to 2 of experiment 425 (stored, same pretrained
  weights, same rows). Also shown: the stored five-seed pool and seed 0 of arm A, `manual_heuristic`,
  and the climatology.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at days 1 to 10; q01, q05, q95, and q99 coverage and pinball
  loss; avoidable-curtailment cost at the signed-MW six.
---

**This page is the plan; it is rewritten with the results.** The question is whether the demand
neural network forecasts better when it can see how sunny the last few whole days were, as measured
by satellite, at the eight Primaries whose meters report apparent power and so hide embedded solar
generation.

## Plan

- **Why this experiment.** [Experiment 480](../00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/result.md)
  gave the XGBoost demand model two columns of realised CAMS irradiance and lowered CRPS at days 1 to
  10 by 0.355 points at the eight and 0.476 at the signed-MW six. That gain is an upper bound,
  because the columns hold realised irradiance at every lead and CAMS arrives about a day late. The
  twin [experiment 408](../00408-xgboost-signed-label-and-cams-columns-at-mva-only-primaries/result.md)
  found 0.359 at the eight with 15 of 16 Primaries improved (D15's interval includes zero). This
  experiment asks how much of that survives for a neural network that already reads the ensemble
  shortwave, when only values a live service could have are used. The gain concerns the magnitude
  score, not the sign of reverse flow.
- **Leak-free columns (whole days only).** Origins are 09:00 UTC. The CAMS table
  (`cams_demand_points.parquet`, hourly irradiance and clear-sky irradiance per Primary, hour-ending
  stamps, 2019-09 to 2026-09-10) has no availability stamp, so the delay cannot be measured from it.
  The plan assumes 36 hours, conservatively: for an origin T, a day D is available when its end
  (D+1 00:00 UTC) is at or before T minus 36 hours, so D is the third day before the origin's date.
  The lag-features study found the whole previous day available at 09:00 UTC, so serving can be one
  to two days fresher than this plan; the one-day-earlier check below measures the sensitivity to
  staleness. Column (a): the CAMS irradiance (W/m2, half-hourly, `half_hour_sun_and_sky`) at the
  target half-hour's time of day on day D. Column (b): the mean of (a)'s quantity over days D, D-1,
  and D-2 at the same time of day. Column (c): the daily CAMS irradiance sum of day D divided by the
  daily clear-sky irradiance sum of day D (a cloudiness state shared by all leads of the origin).
- **Arms (5 folds x 3 seeds, 45 fine-tunes).** (i) K3, stored arm A, no fit. (ii) A-cams: arm A plus
  columns (a) to (c). (iii) A-cams-real: arm A plus experiment 408's realised columns, the unit PV
  shape and clear-sky index of the target half-hour itself (an upper bound that serving cannot
  supply, report only). (iv) A-cams-lag-only: arm A plus column (c) only.
- **Fine-tuning.** Each arm's network is experiment 425's pooled plain network. Its first-layer
  weights are loaded from the stored pretrained arm A network of the same month and seed, with
  zero-initialised weight columns inserted for the new inputs (before the Primary embedding), then
  fine-tuned on the fold's ENS rows at a learning rate of 2e-4 with experiment 425's loop. The new
  columns are standardised on the fold's training rows. A null may mean "not learned in the
  fine-tune" rather than "no information"; the page says so.
- **Rows and cleaning.** Experiment 470's intersection, cleaned data (flagged exact-zero half-hours
  dropped; the count is printed per fold), the D15 and D18 start cut of 2025-03-01, widening 1.15.
  Weather inputs are the ensemble mean and spread, as in experiment 425.
- **Serving cost, to be tabled in the write-up.** A CAMS feed that arrives about a day late (not
  ingested today), one stored daily value per Primary for column (c) and three per half-hour for (a)
  and (b), networks per pass unchanged. Proposed fallback when the feed is late: set the columns to
  their training mean and widen the bands; the prediction-only check measures that loss, and a
  network trained with a random 20% of rows masked is a follow-up and not run here.
- **Compute.** One fine-tune takes 35 to 60 GPU seconds (experiments 481 and 494), so 45 fits take
  0.43 to 0.75 GPU-hours. One GPU process under 8 GB, one feeder core, `free -g` above 20 GB before
  each launch.
- **Standing practices followed.** Monthly refits, cleaned data, the D15 and D18 cut, networks saved
  under the experiment's private folder, stored predictions reused as the control, tails and
  avoidable-curtailment cost reported, example forecasts plotted. Load-shift event plots are not
  needed because the experiment does not score events.
- **Reviews.** Sonnet plan review, Sonnet code review before the first fit, Sonnet results review,
  and Sonnet write-up and prose reviews.
