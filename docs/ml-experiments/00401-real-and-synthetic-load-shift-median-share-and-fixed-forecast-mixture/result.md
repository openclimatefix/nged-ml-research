---
id: "00401"
slug: real-and-synthetic-load-shift-median-share-and-fixed-forecast-mixture
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: small result tables only, in this experiment's private folder; no fitted booster
  and no full prediction folder beyond one table per mixture arm. Production serving path: not
  touched. Degradation rule: not touched. More than one design would test each hypothesis (the
  synthetic-step placement, the weight function); the plan fixes them before any score is read.
  Callers: the scripts import the folders of experiments 391, 395, and 400 (and 381 and 383
  through them).
hypothesis: >-
  Job 1. When a synthetic step of the corrected labelled depth is injected into clean periods at the
  same Primary and the same lag status as each real labelled event, the plain monthly control's
  median still follows the step by much less than it follows the real event, so the gap between 36.8%
  (experiment 395 Part E) and 0.94 (Part F) is not an artefact of the depth or the rows. Job 2. A
  fixed-weight mixture of the weekly forecast with power lags and the forecast without them scores
  within 0.1 points of effective capacity of experiment 400's learned blend B2 on rows with lags in
  an event, at the six Primaries with signed readings.
prediction: >-
  Written before any score is read. P1. With depth, Primary, and lag status matched, the synthetic
  median share at 3 lags stays under 0.6 (it was 0.368 in Part E), so at least half of the gap to
  the real 0.94 remains. P2. The real 0-lag share of 0.52 sits within 0.15 of the share that a
  forecast blind to the event would show, 1 minus the measured step over the labelled depth, taken
  from the lagless forecast on the same rows. P3. The synthetic 0-lag share is within 0.05 of zero.
  P4. The fixed mixture with w = 0.5 minus B2 on event-lag rows at the six Primaries is between -0.3
  and +0.3 points, and w = 0.3 or 0.7 is worse than w = 0.5 on those rows by less than 0.3 points.
  P5. The per-lead weight chosen on training months beats the best fixed weight on all rows by under
  0.03 points. P6. The mixture of distributions beats the quantile average on rows with a lag in an
  event by 0.05 to 0.4 points.
decision_rule: >-
  Job 1 is descriptive: there is no verdict word beyond what the pre-registered predictions say.
  The matched synthetic share and the real share are each a median over events of the per-event
  median over rows (the Part H convention), with a 95% interval from resampling events; the
  two-sided comparison is "the difference excludes zero". Job 2 contrasts (pre-registered,
  CRPS in points of effective capacity, scored on the months experiment 400 scores): for each of the
  three fixed weights and the per-lead weight, as a mixture of distributions and as a quantile
  average, raw and widened by 1.15 (24 arms), minus the forecast with lags (C1 to C24, rows with one
  to three lags inside an unpadded corrected event; whole-event resampling), the same arms minus the
  forecast with lags outside events and outside the 7 days after (C25 to C48; month resampling),
  and the arms minus B2 on both row sets (C49 to C96). 95% intervals per contrast, 99% with k = 12
  for the headline pair (w = 0.5 mixture minus B2 on event-lag rows at the six Primaries, and the
  same arm minus the forecast with lags outside events). "A fixed mixture does as well as B2" holds
  when the headline interval's upper end is at or below +0.1 points on event-lag rows at the six
  and at or below +0.05 points outside events (the protocol's per-Primary tolerance); the audit
  checks that claim before any adoption sentence. A fixed mixture adds a second forecast to
  production, so a pooled gain on event-lag rows under 0.1 points is `small gain`.
benchmark: >-
  Job 2: the forecast with lags (experiment 381's weekly bundle), the forecast without lags
  (experiment 383's raw lagless arm), the learned blend B2 and the w = 0.5 mixture (experiment 400),
  and the GAM stack (experiment 377), on experiment 395's Part F rows. Job 1: experiment 381's
  monthly control and experiment 391's plain monthly control b0 boosters.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Job 1: the median share of the step followed, 1 minus (actual minus median forecast) over the
  signed depth (Part H), and the response share, the change in the median forecast over the injected
  depth (Part E). Job 2: CRPS estimated from the 11 quantiles (exact CRPS for the mixture arms),
  q95 and q99 coverage, 5-95% width, the Part F harm table, smoothness, cost outside events, rows
  on cold nights, and avoidable curtailment cost, in points of effective capacity or GBP as the
  protocol defines them, from `studies.eval_protocol`.
code: ml-experiments/00401-real-and-synthetic-load-shift-median-share-and-fixed-forecast-mixture/
---

# Experiment 401: real and synthetic load-shift median share, and a fixed mixture of two forecasts

## Plan

Two stored-forecast jobs from the load-shift Discuss session (maintainer-approved 2026-10-30). Both
re-score or re-predict stored forecasts; no booster is refitted. Sonnet only. Protocol `demand-v1`.
All numbers are interim until the Sonnet audit.

### Job 1: why do the real-event and synthetic-step medians disagree?

- **The two numbers.** Experiment 395 Part E predicts experiment 391's plain monthly control b0 with
  a synthetic step added to the first `k` lagged-power inputs: the median moves by 1.3%, 17.3%, and
  36.8% of the step at `k` = 1, 2, 3. Part F scores real labelled events with the corrected signs
  and finds the median at 0.52, 0.83, and 0.94 of the event reading with 0, 1 or 2, and 3 lags in the
  event.
- **The two quantities differ in definition, and the plan reports both on both arms.** Part E's
  response share is `(q50 with the step - q50 without) / step`. Part F's share is
  `1 - (actual - q50) / step`. The second share equals 1 when the median matches the observed
  reading, whatever the cause, and it equals `1 - measured step / labelled step` when the forecast
  ignores the event.
- **(a) Same events.** The real arm is experiment 381's monthly control (rows of Part F; it scores
  0.60, 0.83, 0.94). The synthetic arm uses b0's stored boosters: for each corrected event with a
  measurable depth and at least 10 rows in a lag-status group, take every clean scored row of the
  same Primary (no target in a padded event, no lag in an event, not flagged) in the months where
  boosters exist, add the corrected signed depth (as a fraction of the Primary's capacity) to the
  `k` newest lagged inputs and to the actual reading, where `k` = 0, 1, 2 (reported separately and
  as "1 or 2"), and 3. The row sample keeps the lead distribution of the event's real rows (up to
  5,000 rows per event and `k`, fixed seed). The comparison is the per-event median share, then the
  median over events, for the real and the synthetic arm on the same event list, using both share
  definitions on the synthetic arm.
- **The weekly bundle's boosters are not stored** (only experiment 391's `n1`, `b0`, and later
  boosters exist), so the comparison runs on the monthly control for both arms, as the brief allows.
  The first step checks that b0's predictions on the clean rows match experiment 381's monthly
  control to within a stated tolerance; if they do not, the page says so and uses b0 alone for both
  arms.
- **(b) Why 0.52 at 0 lags.** Four small calculations on the real rows with 0 lags in the event:
    1. *Labelled against measured depth.* Recompute the share with the event's measured step
       (`measured_depth_first_mw` in the corrected table) in place of the labelled depth. The Part F
       tables report a measured step of 0.63 times the labelled depth.
    2. *A blind forecast.* Compute the share of experiment 383's raw lagless forecast (no power
       lags at all) on the same rows. A forecast blind to the event shows `1 - measured / labelled`
       rather than 0, so the lagless share is the zero-information reference.
    3. *Forecast noise.* Take clean rows of the same Primaries with a zero-size step and compute the
       same share: its median and its spread show how much forecast error moves the per-event median.
    4. *The 48-hour dilution.* Recompute the real share on the first 48 hours, the first 6 hours, and
       the rest of each event separately, and on events by length class, to separate the dilution of
       a deepening step from the events' own shape.
- **Reported:** a table of real and matched-synthetic shares by lag status with intervals, the
  four calculations in one table, and two figures (real against synthetic share by lag status; the
  zero-information reference against the 0-lag share).

### Job 2: fixed mixture of the forecast with lags and the forecast without lags

- **Inputs.** The forecast with lags is experiment 381's weekly bundle; the forecast without lags is
  experiment 383's raw lagless arm (`work/arms/lagless.parquet`, not arm c2). The scored rows, the
  label table, and the flagged-zero handling are experiment 400's (Part F slice table; flagged
  half-hours tagged and dropped; the count per scored set is stated on the page).
- **Arms.** (1) Fixed weight `w` on the forecast with lags in {0.3, 0.5, 0.7}. (2) A per-lead
  weight: for each scored month, for each lead day 1 to 10, the `w` in {0.1, 0.2, ..., 0.9} that
  minimises the exact CRPS of the mixture on the earlier months' rows (flagged half-hours dropped,
  pooled over Primaries); the page reports whether the chosen weight rises or falls with lead. (3)
  Each weight function as a mixture of distributions (CDF mixing, experiment 400's `e00400_mixture`)
  and as an average of quantiles. Each arm is written raw and widened by 1.15, where "widened by 1.15"
  means each quantile moves from the arm's median by 1.15 times its distance (the convention of
  experiment 383's `widen`). The 1.15 arms are the widened versions of the raw arms, not new fits.
  B2 and the `w` = 0.5 mixture come from experiment 400's tables, and 400's B2 is raw only; the page
  says so wherever B2 is compared with a widened arm.
- **Reported, per group (signed-MW six, MVA-only eight) and per Primary:** the Part F harm table
  (CRPS minus the forecast with lags by lag count inside an event), the cost outside events,
  smoothness (largest half-hour change in the median across an event onset), avoidable curtailment
  cost during events, in the 7 days after, and on all rows, as differences against the forecast with
  lags and the GAM stack, and cold-night rows (the coldest 5% of valid times at each Primary). Tail
  calibration (0.01, 0.05, 0.95, 0.99 pinball loss and coverage) sits beside CRPS.
- **Audit.** The claim "a fixed mixture does as well as B2" goes to the Sonnet audit before any
  adoption sentence is written.

### Process and sizing

- **Stored.** Small tables under the private `work/` folder; nothing new in git beyond scripts and
  the page.
- **Production serving path / degradation rule.** Not touched.
- **More than one design.** Yes (synthetic placement; weight function): the plan fixes both.
- **Callers.** The scripts import experiments 391, 395, and 400.
- **Cadence and compute.** Prediction-only with b0's monthly boosters; stored forecasts only for Job
  2; no GPU; at most 2 cores; `MemoryMax=6G`; `systemd-run --user --scope -p OOMPolicy=continue`.
- **Advice followed.** Train on injected events, not real rows alone (experiment 400): Job 2 only
  re-weights stored forecasts and does not fit a surprise response.
- **Caveats stated in advance.** Weekly-bundle boosters are not stored. Experiment 395 is unmerged
  (PR 49); the corrected label table is copied from its head. Experiment 400's learned blend is raw
  only.
