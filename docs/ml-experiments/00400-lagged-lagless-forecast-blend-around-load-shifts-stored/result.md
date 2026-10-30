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
  capacity CRPS against 3.867 for the forecast with lags, so no blend beats the forecast with lags
headline: >-
  The fitted weight on the forecast with lags stays near 0.6 whether or not a lag is inside an
  event (0.61 with three lags in an event, 0.62 with none). Arm B2 minus the forecast with lags:
  -0.75 (95% -1.78 to -0.03) points on event-lag rows at the six Primaries with signed readings,
  +0.09 (-0.04 to +0.22) at the eight with MVA-only readings, and +0.03 (-0.01 to +0.08) outside
  events.
take_home_message: >-
  The lag surprise scores from experiment 391 did not tell a blend when to trust the power lags:
  the fitted weight hardly moved around events, so every learned blend behaved like a mix of about
  60% forecast with lags and 40% forecast without lags. A fixed 50/50 mix helps on event-lag rows at
  the six Primaries with signed readings and costs a little elsewhere, so a smooth blend driven by
  these scores is not worth porting. The next step is a learned blender trained on injected events,
  which supplies the event examples that the real rows are too few to give.
lessons: >-
  Event-lag rows are 41,000 of 1,573,000 scored rows (2.6%), too few for a weight with three to
  eight parameters to learn a surprise response. The fit is dominated by rows with no lag in an
  event, where a larger surprise score slightly raises the weight on the forecast with lags.
advice:
  - >-
    Do not fit a lag-surprise blend weight on real rows alone; train it on injected load-shift events
    so event-lag rows are more than 3% of the fit (event-lag table, section Results).
  - >-
    Include a fitted constant-weight mixture (intercept and lead time only) beside the fixed 50/50
    mixture as the control for any learned blender, because the fixed 50/50 mixture and the learned
    weights differ in level as well as in surprise response (control comparison, section Results).
production_status: not-for-production
adopted_in: none
---

# Experiment 400: blend a forecast with power lags and a forecast without them, by lag surprise

## Plan

- **Question.** Can a smooth, threshold-free weight on a forecast without power lags, driven by how
  surprising the three lagged readings are, give a forecast that is better around load-shift events
  and no worse elsewhere? A binary detector reached recall 0.42 with 1.7 false alarms per
  Primary-year (experiment 391). A forecast that steps loses the trust of NGED's operators.
- **Forecast with lags.** Experiment 381's weekly bundle (`arm_preds/weekly/bundle_solar+sunny+hl1`),
  stored as fractions of effective capacity, unwidened. It is not experiment 383's plain weekly
  member (arm d, which is experiment 87's stored weekly member): that member has 64 stored files and
  cannot cover all rows.
- **Forecast without lags.** Experiment 383's raw lagless arm (`work/arms/lagless.parquet`,
  4,099,680 rows, 16 Primaries). Arm c2 of experiment 383 is not used: it applies this arm only on
  rows with a lag inside an event and arm d elsewhere.
- **Widening.** Both forecasts are used raw (no 1.15 widening): experiment 381's bundle carries none
  and the lagless arm is raw. The 1.15 sensitivity run planned here was not run (see Caveats).
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
  table; the absolute score beside every contrast. Departures are listed under Caveats.

## Results

**The blend gives no usable gain, and the surprise scores do not drive the weight.** Every number
below is for days 1 to 10, valid times from 2025-10-01 (the first scored month), CRPS (continuous
ranked probability score) in points of effective capacity. The six Primaries with signed readings
are D5, D9, D10, D11, D14, and D16. The eight with MVA-only readings are D2, D3, D4, D8, D12, D19,
D21, and D22. Intervals resample whole events (event rows) or months (pooled rows) and are at 95%.
The decision rule asked for 99% intervals at k = 12 and a pooled contrast on event-lag rows; neither
was computed, so the verdict rests on the failed cost condition below and on the absence of any
gain at the eight. The printed reports are in the private folder (`work/report.txt` and
`work/curtailment.txt`). This page has no figures.

- **Flagged-zero handling.** Flagged exact-zero half-hours are dropped from the scoring actuals and
  from the fits. The scored rows lose 96 half-hours (D12 8, D18 24, D22 40, D8 24).
- **Widening.** Both forecasts are raw. Experiment 381's weekly bundle and experiment 383's raw
  lagless arm carry no 1.15 widening.
- **Label table.** The corrected table `sw_block_detector_label_table_corrected.csv` from nged-ml-research
  PR 49: 112 labelled pairs of Primaries (51 with a corrected sign), expanded to 184 per-Primary
  unpadded events. It is used only to evaluate, never as an input.
- **The weight barely moves around events.** In arm B2 the mean weight on the forecast with lags is
  0.615 outside events, 0.611 in the first 6 hours of an event, and 0.597 after 24 hours. Its 5th to
  95th percentile range is 0.56 to 0.68. The fitted coefficients on surprise and on the log of lead
  time are both negative, so a larger surprise and a longer lead each raise the weight on the
  forecast with lags.
- **Absolute CRPS, all 16 Primaries.** Forecast with lags 3.867; forecast without lags 4.338; fixed
  50/50 mixture 3.938; mixture B1 3.898, B2 3.892, B3 3.886; quantile averages 3.907 to 3.968.
- **Event-lag rows** (1 to 3 lags in an event, target outside), arm minus forecast with lags (C1 to
  C4):

  | Arm | Six with signed readings | Eight with MVA-only readings |
  |---|---|---|
  | Fixed 50/50 mixture | -0.90 (-2.16 to -0.01) | +0.14 (-0.04 to +0.31) |
  | Mixture B1 | -0.77 (-1.83 to -0.03) | +0.10 (-0.05 to +0.24) |
  | Mixture B2 | -0.75 (-1.78 to -0.03) | +0.09 (-0.04 to +0.22) |
  | Mixture B3 | -0.78 (-1.83 to -0.03) | +0.08 (-0.05 to +0.20) |
  | Forecast without lags | -0.93 (-2.49 to +0.26) | +0.59 (+0.14 to +1.04) |

- **Control comparison (C9).** The control is a fixed 50/50 mixture that was not fitted, and no
  fitted constant-weight arm was run, so the weight level and the surprise response are not
  separated. On event-lag rows, mixture B2 minus the fixed 50/50 mixture is +0.15 (-0.02 to +0.39) at
  the six and -0.05 (-0.10 to -0.01) at the eight. Outside events it is -0.046 (-0.065 to -0.024)
  over all 16 Primaries.
- **Cost outside events** (C5 to C8, all 16 Primaries, month resampling): fixed 50/50 mixture +0.078
  (+0.013 to +0.138); B1 +0.038, B2 +0.033 (-0.013 to +0.076), B3 +0.027 (-0.018 to +0.070). The B2
  upper end exceeds the +0.05 tolerance, and the pooled cost is dominated by D15, which has a short
  history: B2 minus the forecast with lags is +1.40 points at D15, +0.15 at D18, and +0.10 at D22,
  with 9 of 16 Primaries below zero.
- **Tails.** Mixture B2 has q95 coverage 0.936 outside events against 0.922 for the forecast with
  lags, because the mixture is wider (90% width 28.4 against 25.2 points). The quantile averages
  keep the width of the forecast with lags (26.1) and gain little coverage. During events with 0, 1
  to 2, and 3 lags in the event, the share of readings inside the 1-99% band is 0.88, 0.79, and 0.81
  for mixture B2 and 0.87, 0.78, and 0.82 for the forecast with lags. For the 5-95% band it is 0.65,
  0.56, and 0.40 for mixture B2 and 0.62, 0.53, and 0.40 for the forecast with lags.
- **Depth missed by the median.** The share of each event's labelled depth that the forecast median
  misses, with the corrected signs, is 0.75 (forecast with lags), 0.56 (forecast without lags), and
  0.59 (mixture B2) with no lag in the event. With three lags in the event it is 0.63, 0.57, and
  0.66.
- **Smoothness.** The blends do not step more than the pure forecasts. The typical (median over
  events) largest half-hour change of the forecast median at an onset is 6.2 points for mixture B2
  against 6.7 for the forecast with lags, and the 95th percentile is 13.3 against 15.8. The pure
  forecasts already change by several points at event edges.
- **Avoidable curtailment** (limit 0.9 times each Primary's maximum observed power), arm minus the
  forecast with lags. Only four of the six Primaries with signed readings returned a cost during
  events and in the 7 days after (D5, D10, D14, D16; D9 and D11 returned "no cost" from the
  protocol function), and D16 supplies most of it. During events mixture B2 is +£61,000 and the fixed
  50/50 mixture +£89,000, against a total of £6.56 million for the forecast with lags (about 1%,
  within noise), and the forecast without lags is +£1.16 million. In the 7 days after events every
  blend is between -£3,800 and -£3,000, and the GAM stack is +£112,000. Over all rows mixture B2 is
  +£40,000. The GAM stack is £3.7 million below the forecast with lags during events and £4.9
  million below it over all rows.
- **Cold nights.** Outside events mixture B2 minus the forecast with lags is +0.042 (-0.182 to
  +0.146). On event-lag rows at the six Primaries with signed readings it is -1.43 (-2.57 to -0.06),
  from 538 rows.
- **Warm-up** (July to September 2025, weight 0.5): fixed 50/50 mixture 2.883 against 2.864 for the
  forecast with lags.

## Predictions

- **P1 (gain of 0.3 to 1.5 points on event-lag rows, growing with lags).** Met at the six only (B2
  -0.75); not met at the eight (+0.09); the gain does not grow steadily with the number of lags.
- **P2 (cost under 0.02 pooled and under 0.05 at every Primary).** Not met: +0.033 pooled, and D15,
  D18, and D22 exceed 0.05.
- **P3 (B3 beats B2 by under 0.2 points).** Met: -0.78 against -0.75 at the six.
- **P4 (no larger step than the pure forecasts).** Met.
- **P5 (mixture tails reach the event level more often).** Not tested as written; the band shares
  above are the nearest measure and show 0.01 to 0.03 differences.

## Caveats and what to try next

- **Leak check.** Each surprise value comes from a monthly stage 1 refit that trained on no row at or
  after its refit time, and every lag time is before the forecast origin, so no row leaks.
- **Departures from the plan.** The 1.15 widening sensitivity run was not run, so the relative
  result under widening is untested. The exact CRPS of each mixture was computed
  (`work/exact_crps.parquet`) but is not tabulated here. The 99% intervals and the pooled
  event-lag contrast were not computed.
- **Next.** A learned blender trained on injected load-shift events, with a fitted constant-weight
  mixture as its control.
