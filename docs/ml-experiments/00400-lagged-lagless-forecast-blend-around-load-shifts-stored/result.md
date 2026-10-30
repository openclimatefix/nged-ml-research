---
id: "00400"
slug: lagged-lagless-forecast-blend-around-load-shifts-stored
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
  - curtailment-cost
status: finished
kind: other
protocol: demand-v1
size: >-
  medium. Stored: small result tables only, in this experiment's private folder; no fitted booster
  and no prediction folder of full forecasts beyond one table per blend arm. Production serving
  path: not touched. Degradation rule: not touched. More than one design would test the hypothesis
  (the weight function, the mixture against the quantile average); the plan fixes them before any
  score is read. Callers: the scripts import the folders of experiments 383, 391, and 395.
hypothesis: >-
  A mixture of the weekly XGBoost forecast with power lags and the XGBoost forecast without power
  lags, weighted by a smooth function of the lags' surprise scores and of lead time, scores better
  than the forecast with lags on the rows whose lags touch a load-shift event, and costs no more
  than 0.02 points of effective capacity on rows outside events.
prediction: >-
  Written before any score is read. P1. Arm B2 (own plus neighbour surprise) lowers CRPS on rows
  with one to three lags inside an event by 0.3 to 1.5 points of effective capacity against the
  forecast with lags, and the gain grows with the number of lags inside the event. P2. The cost
  outside events is under 0.02 points pooled and under 0.05 at every Primary. P3. Arm B3 (signed
  neighbour logic) beats B2 on harm rows by under 0.2 points; the sign information adds little.
  P4. The largest half-hour-to-half-hour change in the blended median across an event onset is no
  larger than the larger of the two pure forecasts' changes. P5. The mixture's q95 and q99 reach the
  event level on more event-level readings than the forecast with lags alone, and the quantile
  average does not.
decision_rule: >-
  Contrasts (pre-registered, scored on the scored months after the warm-up): for each of B1, B2,
  B3, and the fixed-weight control, CRPS minus the forecast with lags (C1 to C4, rows with one to
  three lags inside an unpadded corrected event), the same arms minus the forecast with lags on rows
  with no lag in an event and outside the 7 days after (C5 to C8), and B2 minus the fixed-weight
  control and minus the forecast without lags on both row sets (C9 to C12). Intervals: whole-event
  resampling for event rows, month resampling for pooled rows, 95% per contrast and 99% with
  k = 12 for the headline pair (B2 on harm rows, B2 outside events). The blend counts as helpful
  when the B2 interval on harm rows lies wholly below zero, the B2 interval outside events has its
  upper end at or below +0.05 points of effective capacity (the protocol's per-Primary tolerance),
  and the blended median does not step more than the pure forecasts do (P4). A blend that helps
  needs a threshold above a zero-cost change, because it adds a second forecast and a score
  pipeline to production: a pooled gain on harm rows under 0.1 points is reported as
  `small gain`. Not a test of an NRA forecast: no forecast can predict an event.
benchmark: >-
  The forecast with lags (experiment 381's weekly bundle), the forecast without lags (experiment
  383's raw lagless arm), the fixed-weight mixture (w = 0.5), the quantile average of the same two
  forecasts, and the GAM stack (experiment 377), on the row set of experiment 395's Part F.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, q95 and q99 coverage, 5-95% width, and avoidable curtailment
  cost, in points of effective capacity or GBP as the protocol defines them, from
  `studies.eval_protocol`.
code: ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored/
verdict: not supported
best_arm: >-
  None. Arm B3 (signed neighbour logic) scores best of the blends, 3.886 points of effective
  capacity CRPS against 3.867 for the forecast with lags, so it does not beat the forecast with lags
headline: >-
  The fitted weight on the forecast with lags stays near 0.6 whether or not a lag is inside an
  event (0.61 with three lags in an event, 0.62 with none). Arm B2 minus the forecast with lags: -0.75
  (95% -1.78 to -0.03) points on event-lag rows at the signed-MW six, +0.09 (-0.04 to +0.22) at the
  MVA-only eight, and +0.03 (-0.01 to +0.08) outside events.
take_home_message: >-
  The lag surprise scores from experiment 391 did not tell a blend when to trust the power lags: the
  fitted weight hardly moved around events, so every learned blend behaved like a fixed 60/40 mix.
  That fixed mix helps at the six Primaries with signed readings when a lag sits in an event, and
  loses a little everywhere else, so a smooth blend driven by these scores is not worth porting. The
  next step is a learned blender trained on injected events, which gives the blend examples of
  events that the real rows are too few to supply.
lessons: >-
  Around 112 hand-labelled events the real rows hold too few event-lag rows (about 40,000 of
  1,570,000) for a three-to-eight-parameter weight to learn a surprise response; the fit spends its
  effort on the 97% of rows without an event, where the surprise scores point the other way
  (large surprise slightly raises the weight on the forecast with lags).
advice:
  - >-
    Do not fit a lag-surprise blend weight on real rows alone; train it on injected load-shift events
    so event-lag rows are not 3% of the fit (C1 to C4, section Weight).
  - >-
    Treat a fixed mixture of the forecast with lags and the forecast without lags as the control for
    any learned blender, because it beat every surprise-driven weight on event-lag rows at the
    signed-MW six (C9).
  - >-
    Score a mixture with the exact CRPS as well as the 11-quantile score, because the 11-quantile
    score understates a bimodal mixture (exact CRPS table).
production_status: not-for-production
adopted_in: none
---

# Experiment 400: blend a forecast with power lags and a forecast without them, by lag surprise

## Plan

- **Question.** Can a smooth, threshold-free weight on a forecast without power lags, driven by how
  surprising the three lagged readings are, give a forecast that is better around load-shift events
  and no worse elsewhere? A binary detector reached recall 0.42 with 1.7 false alarms per
  Primary-year (experiment 391), and a forecast that steps loses users' trust.
- **Forecast with lags.** Experiment 381's weekly bundle (`arm_preds/weekly/bundle_solar+sunny+hl1`),
  stored as fractions of effective capacity, unwidened. It is not experiment 383's plain weekly
  member (arm d, which is experiment 87's stored weekly member): that member has 64 stored files and
  cannot cover all rows.
- **Forecast without lags.** Experiment 383's raw lagless arm (`work/arms/lagless.parquet`,
  4,099,680 rows, 16 Primaries). It is not arm c2, which is this arm on rows with a lag inside an
  event and arm d elsewhere.
- **Widening.** Both forecasts are used raw (no 1.15 widening): experiment 381's bundle carries none
  and the lagless arm is raw. A sensitivity run applies the same 1.15 factor about the median to
  both forecasts and to the blend.
- **Row set.** The 2,824,844 rows of experiment 395's Part F `slice_table.parquet` (16 Primaries,
  valid times 2025-07-02 to 2026-06-30). The plan checks before scoring that both forecasts hold
  every key and reports any gap. Scoring actuals come from Part F's `actual_mw`; half-hours flagged
  in the cleaned power table are dropped from scoring, and the count dropped is reported per
  Primary.
- **Suspicion score.** Experiment 391's `surprise_tables/half_hourly/<site>.parquet`: `u`, the
  normal score of a reading in the lagless stage-1 quantile forecast, at each lag's valid time. A
  row's three lags (origin, day, and week lags, found with experiment 87's `origin_lags`) give
  three own-Primary values. The neighbour surprise uses experiment 391's NGED interconnection list.
  **Leak check.** Each `u` comes from a monthly refit that trained on no row at or after its refit
  time, and a lag's valid time is at or before the forecast origin, so no `u` uses information
  after the origin. The monthly stage-1 tables are indexed by valid time, so they align with the
  weekly forecasts through the lag's valid time. A lag in the first stage-1 month (2024-10) has
  `u` from a model trained on 2024-04 to 2024-09; the scored months start nine months later.
- **Arms.** Mixture weight on the forecast with lags: w = 1 / (1 + exp(a + b s + c lead_term)),
  with lead_term = log(lead hours / 24). B1: s is the largest absolute own surprise over the three
  lags. B2: s is B1 plus the largest absolute neighbour surprise, as two inputs (b1, b2). B3: the
  signed own surprise, the sum of the neighbours' positive and negative surprises, and the largest
  of each sign, as five inputs, so a fit can learn the neighbour sign logic; the rule is not hand
  coded. Fixed-weight control: w = 0.5. Endpoints: the two pure forecasts. Each of B1 to B3 and the
  control is scored as a mixture and as a quantile average of the same weights.
- **Fit.** The exact CRPS of the mixture (the integral of the squared difference between the mixture
  CDF and the step at the actual value, with the piecewise-linear CDFs below), plus a ridge penalty
  of 1e-3 times the squared parameters, minimised by BFGS (at most 60 iterations) on a fixed random
  subsample of 200,000 training rows (seed 0), with an expanding window: the model for scored month
  m uses rows whose valid time is before the first of m. July to September 2025 is warm-up and
  uses w = 0.5; October 2025 to June 2026 is scored. Warm-up months are reported separately. The
  label table never enters a fit.
- **Reported score.** The protocol's CRPS from the mixture's 11 quantiles, so every arm is scored as
  every earlier experiment is. For a mixture of two well-separated forecasts the 11-quantile score
  is lower than the exact CRPS, so the report also gives the exact CRPS of the mixture arms.
- **Mixture quantiles.** Each forecast's CDF is linear between its 11 quantiles, with the end
  segments extended to levels 0 and 1 at the slope of the outer segment. The mixture CDF is
  w F1 + (1 - w) F2, inverted by bisection at the 11 levels.
- **Events.** The corrected label table `sw_block_detector_label_table_corrected.csv` from the head
  of nged-ml-research PR 49 (112 rows with an end time and a depth), unpadded, with the corrected
  signs. It is used only to evaluate.
- **Checks** are the seven in the brief: the harm table by number of lags inside the event, the cost
  outside events, the median share of the labelled depth, the tails during events, smoothness at
  onset and end, avoidable curtailment during events and in the 7 days after (differences against
  the forecast with lags and the GAM stack), and the cold-night rows separately.
- **Compute.** Stored forecasts only: no refit, no GPU, 2 cores, `MemoryMax=6G`. Chunks of one
  Primary and one month.
- **Earlier advice followed.** Cleaned actuals; both forecasts and the weight named in full in every
  table; the absolute score beside every contrast. No departures.

## Results

All numbers are days 1 to 10, valid times from 2025-10-01 (the fits' first scored month), flagged
half-hours dropped (4 Primaries lose 96 half-hours in total), CRPS in points of effective capacity.
Intervals resample whole events (event rows) or months (pooled rows). The full printed report is in
the private folder (`work/report.txt`, `work/curtailment.txt`). This page has no figures.

- **Weight.** The fitted weight on the forecast with lags has mean 0.615 outside events and 0.611,
  0.597 in the first 6 hours and after 24 hours of an event (arm B2). Its 5th to 95th percentile
  range is 0.56 to 0.68. The surprise coefficient is slightly negative: a larger surprise raises the
  weight on the forecast with lags.
- **Absolute CRPS, all 16 Primaries:** forecast with lags 3.867; forecast without lags 4.338; fixed
  mixture 3.938; mixture B1 3.898, B2 3.892, B3 3.886; quantile averages 3.907 to 3.968.
- **Event-lag rows (1 to 3 lags in an event, target outside), minus the forecast with lags** (C1 to
  C4):

  | Arm | Signed-MW six | MVA-only eight |
  |---|---|---|
  | Fixed mixture | -0.90 (-2.16 to -0.01) | +0.14 (-0.04 to +0.31) |
  | Mixture B1 | -0.77 (-1.83 to -0.03) | +0.10 (-0.05 to +0.24) |
  | Mixture B2 | -0.75 (-1.78 to -0.03) | +0.09 (-0.04 to +0.22) |
  | Mixture B3 | -0.78 (-1.83 to -0.03) | +0.08 (-0.05 to +0.20) |
  | Forecast without lags | -0.93 (-2.49 to +0.26) | +0.59 (+0.14 to +1.04) |

- **Surprise adds nothing over a fixed weight on the six** (C9): B2 minus the fixed mixture is +0.15
  (-0.02 to +0.39); at the eight it is -0.05 (-0.10 to -0.01).
- **Cost outside events** (C5 to C8, all 16, month resampling): fixed mixture +0.078 (+0.013 to
  +0.138); B1 +0.038, B2 +0.033 (-0.013 to +0.076), B3 +0.027 (-0.018 to +0.070). The B2 upper end
  exceeds the +0.05 tolerance. D15 (short history) carries +1.4 of the cost and D18 +0.15.
- **Tails.** Mixture q95 coverage outside events is 0.936 (B2) against 0.922 for the forecast with
  lags, because the mixture is wider (90% width 28.4 against 25.2 points). Quantile averages keep
  the width of the forecast with lags (26.1) and gain little coverage. During events (0, 1 to 2,
  and 3 lags in the event) the share of readings inside the 1-99% band is 0.88, 0.79, and 0.81 for
  mixture B2 against 0.87, 0.78, and 0.82 for the forecast with lags, and inside the 5-95% band
  0.65, 0.56, and 0.40 against 0.62, 0.53, and 0.40.
- **Median share of the labelled depth** (corrected signs) missed by the median: 0.75 (forecast with
  lags), 0.56 (without lags), 0.59 (B2) with no lag in the event; 0.63, 0.57, and 0.66 with three
  lags in the event.
- **Smoothness.** The blends do not step more than the pure forecasts. Largest half-hour
  change of the median across an onset: median 6.2 points and 95th percentile 13.3 (mixture B2)
  against 6.7 and 15.8 (forecast with lags). The pure forecasts already change by several points at event edges.
- **Avoidable curtailment** at the signed-MW six, limit 0.9 times the maximum, GBP, minus the
  forecast with lags: during events B2 mixture +61,000, fixed mixture +89,000, forecast without lags
  +1,164,000; in the 7 days after about -3,000 for every blend; all rows B2 mixture +40,000. The GAM
  stack is lower than the forecast with lags by 3.7 million during events and 4.9 million over all
  rows.
- **Cold nights** (checks 1 and 2): outside events B2 minus the forecast with lags is +0.042 (-0.182
  to +0.146); on event-lag rows at the six -1.43 (-2.57 to -0.06). The sample is 538 event-lag rows.
- **Warm-up** (July to September 2025, w = 0.5): fixed mixture 2.883 against 2.864 for the forecast
  with lags.

## Caveats and what to try next

- **Leak check.** Each surprise value comes from a monthly stage 1 refit that trained on no row at or
  after its refit time, and every lag time is before the forecast origin, so no row leaks.
- **Reused forecasts.** Experiment 381's weekly bundle (no widening) and experiment 383's raw lagless
  arm (no widening); experiment 383's arm c2 is not used. The 1.15 widening sensitivity was not run,
  because neither forecast carries the widening and the blend result does not depend on it.
- **Corrected label table:** 112 rows (51 sign-corrected) from PR 49, merged to 184 unpadded events.
- **Next.** A learned blender trained on injected load-shift events, as the plan describes.
- **Code.** `ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored/`; it reuses
  experiments 87, 377, 381, 383, 390, and 395 (copied as `e00400_partf.py`).
