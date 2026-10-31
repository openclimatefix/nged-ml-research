---
id: "00428"
slug: day-window-convolution-attention-encoder-replacing-lagged-power-inputs
title: >-
  Can a small convolutional network over whole days of recent power, read by attention, replace the
  hand-made lagged power inputs and stop the forecast following load-shift events?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks, forecasts, sweep tables, and attention logs in this experiment's
  private folder only; every earlier experiment is read and never written. More than one design
  would test the hypothesis (which days enter, how a day is cut, how the encoder is trained), so
  each choice is fixed in this plan before any score is read. Production serving path: not touched
  (this repository holds none). Degradation rule: not touched. Callers: the scripts import the
  folders of experiments 87, 397, 402, 422, and 392 and the `studies` evaluation code.
hypothesis: >-
  A pooled mixture density network that has no hand-made lagged power input and instead reads the
  raw half-hourly power of whole recent days through a one-dimensional convolutional network and
  an attention layer (whose query is the half-hour being forecast) forecasts as well as the pooled
  plain network with the three origin lags on rows outside load-shift events, forecasts better on
  harm rows (rows whose target is outside every event but whose recent days are inside one), and
  follows a step added to the recent days by a much smaller share of the step, with wider bands.
prediction: >-
  Written before any fit and before any score is read. All numbers are CRPS in points of effective
  capacity at days 1 to 10, or shares of a step. P1. On clean rows, the one-day encoder (E1) minus
  the lagged reference trained on the same injected data (R-inj) lies between 0.00 and +0.15 at
  the signed-MW six, because E1 has no weekly lag; the three-day encoder (E3) minus R-inj lies
  between -0.05 and +0.05. P2. On harm rows, E3 minus R-inj lies between -0.30 and -0.10 at the
  signed-MW six and within +-0.10 at the MVA-only eight (experiment 402 found a flag oracle gains
  nothing there). P3. In the encroachment sweep (step of 5% of capacity, 24 to 72 h leads, raw
  bands), R follows 0.15 to 0.45 of the step at three lags (the XGBoost model of experiment 392
  followed 0.38 to 0.41; a neural network may follow less), R-inj follows 0.05 to 0.30, and E3
  follows at most 0.05 of the step at every k, in both variants. P4. The 5%-95% band of E3 is at
  least 1.10 times wider after the step than before at k = 3 (variant 2). P5. The attention mass
  on the "no day" token rises by at least 0.15 on rows whose days contain the step, against the
  same rows without it. P6. E2 minus E1 on clean rows is between -0.10 and -0.02 (the day of the
  week matching the target carries the weekly pattern); E3 minus E2 lies within +-0.03. P7. The
  lag-masked control follows exactly 0 (a check of the pipeline, not a result). P8. B-mix (the
  fixed 50/50 mixture of the lagged and lagless networks) follows about half of R's share and
  loses 0.05 to 0.20 CRPS points on clean rows against R.
decision_rule: >-
  Screen, four folds (test months 2026-06, 2026-01, 2025-10, 2026-03), then five (adding 2025-12),
  three seeds (0, 1, 2) per fold, monthly refits only. Every number is interim and carries no 99%
  claim. Contrasts, each read per group (signed-MW six; MVA-only eight; D15 and D18 report-only)
  on CRPS at days 1 to 10 at the 1.15 widening, on three row sets (clean rows, harm rows, rows
  whose target is inside a labelled event): C1 E1 minus R-inj; C2 E2 minus E1; C3 E3 minus E2; C4
  E3 minus R-inj; C5 E3 minus B-mix; C6 R-inj minus R (does training on injected events alone help
  the lagged network). The smallest gain worth porting is 0.10 CRPS points on harm rows with a
  loss of at most 0.05 on clean rows, because the encoder adds a network module to the production
  training loop. A fold counts as a win for a contrast when the point estimate is below zero by
  more than the minimum detectable effect. Harm rows here means the injected-world harm rows; the
  real harm rows are reported beside them and do not decide. A fold is lost when the point
  estimate is above zero by more than the minimum detectable effect. Advance to the fifth fold,
  and to experiment 404's trust inputs as extra tokens, if E3 beats R-inj on harm rows at the
  signed-MW six in at least 3 of 4 folds with a pooled gain of at least 0.10, loses at most 0.05
  on clean rows, and in the sweep follows no more than half of R-inj's share at k = 3 in variant
  2. Stop, and report which of the three conditions failed, if E3 loses on harm rows in at least 3
  of 4 folds. In every other case (for example 2 wins and 2 losses) the page reports the screen as
  inconclusive and the maintainer decides; E2 is reported but never decides. The sweep goal is a
  share near zero with wider bands: success is a median share of at most 0.05 at every k up to 3
  in both variants, with a band-width ratio of at least 1.10 at k = 3. The minimum detectable
  effect is computed before any contrast is read, separately for each row set and group: for
  R-inj, the sample standard deviation (12 fold-by-seed values, each seed's CRPS minus the fold's
  three-seed mean, 8 degrees of freedom) times 1.96 times the square root of 2 divided by the
  square root of 3. It covers seed noise only, so it is a lower bound, as in experiments 419 and
  425. No multiplicity correction is applied to the screen. A contrast smaller than that is
  "inside the noise".
benchmark: >-
  R (the pooled plain mixture density network with the three origin lags, experiment 397's
  `plain_pooled` recipe at setting g09, with the Primary embedding, the ensemble mean and spread
  once, and the hand-made calendar columns, refitted here on clean data under the D15 and D18
  cut), R-inj (the same network trained on the injected worlds), L0 (the same network with no
  lagged input), and B-mix (the fixed 50/50 mixture of R-inj and L0). The monthly XGBoost recipe
  (experiment 389 arm S0M), BG2, `manual_heuristic`, and `climatology` are absolute-skill
  references on the half-hours they share, read from their stored forecasts.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles (the protocol's), coverage and pinball
  loss at q01, q05, q95, and q99, the avoidable-curtailment cost, raw and at the 1.15 widening; the
  encroachment sweep of experiment 392 (median shift as a share of the step, 5%-95% band-width
  ratio, coverage of clean and event-level readings); the attention weights per day.
code: ml-experiments/00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs/
---

*Status: planned. No fit has run and no score has been read.*

## Question

CRPS is the continuous ranked probability score. The signed-MW six and the MVA-only eight are
the two groups of Primary substations by metering (signed megawatt readings, or unsigned
megavolt-ampere magnitudes); D15 and D18 are two short-history Primaries.

**We test whether a network that reads whole days of raw power can learn how much to trust them.**
A load-shift event moves demand between neighbouring Primary substations for hours or weeks, so
the power an hour, a day, or a week ago can be abnormal. Today's networks take three hand-made
lagged readings. Experiment 392 found an XGBoost model follows about 1%, 15% to 20%, and 38% to 41%
of a step added to the first one, two, and three lags, with no band widening. This experiment
replaces the lags (and every window or surprise statistic built from them) with a small
one-dimensional convolutional network over whole days, followed by attention. The idea is the
maintainer's (2026-10-31). The attention design follows [Bahdanau et al.
(2015)](https://arxiv.org/abs/1409.0473) (a query scores each position of a sequence) and the
convolution follows [Bai et al. (2018)](https://arxiv.org/abs/1803.01271) (a stack of convolutions
as a sequence encoder).

## Review against the latest evidence

**The brief was checked against the pages merged since it was written** (experiments 392, 397, 402,
403, 421, 422, 424 and 426, and the open plans of experiments 404, 425, and 427, read on 2026-10-31).

- **The reference is the pooled plain network.** Experiment 425's plan names the ENS-only pooled
  plain network (R) as the reference; and this plan uses the same recipe.
- **Experiment 404's harness has not merged.** Its injected-event generator is in the draft PR 58. This
  plan uses experiment 402's generator (`e00402_inject.inject`), which is merged and which 404's
  generator follows. The scripts of experiment 404 are not imported.
- **The sweep's k is defined for whole days** (manager, 2026-10-31); see "Encroachment sweep".
- **Standing practices followed.** Monthly refits; cleaned data with the flagged and start-cut
  half-hours dropped (experiment 422's `cleaned_arrays`), reported per fold; the D15 and D18 cut at
  2025-03-01; every network saved in the data folder; stored forecasts reused as references; tails
  and the avoidable-curtailment cost beside CRPS; example forecasts and event plots; draft PR.

## Design

**Inputs.** The network sees the same weather (ensemble mean and spread, once), solar, calendar,
holiday, lead-time, and Primary-embedding inputs as R, and no lag column of any kind. Its only view
of recent power is up to three day windows of 48 half-hours:

- **D1, the day before the origin:** the 48 half-hours ending at the newest usable reading (the
  origin minus 3.5 hours, the rule of `common.origin_lags`).
- **D2, the matching day of week:** 48 half-hours centred on the most recent earlier half-hour with
  the target's day of week and time of day that is at least 3.5 hours before the origin (the
  weekly lag of R), the half-hours after the newest usable reading masked as unobserved.
- **D3, two weeks earlier:** the window of D2 shifted back by 336 half-hours.

Each window position carries five channels: the power over effective capacity (standardised with
the training rows' mean and standard deviation, clipped at 5 standard deviations, 0 where
unobserved), an observed flag, sin and cos of the position's UTC time of day, and its age before the
origin divided by 336 hours. Missing, flagged, and start-cut half-hours are unobserved.

**Encoder.** Three one-dimensional convolutions (kernel 5, padding 2, widths 16, 32, and 32, GELU)
applied to each window with shared weights, giving 48 position vectors of width 32 per window. No
pooling, so position survives. A learned day-type embedding (3 vectors) is added to each position
vector, so attention knows which window a position is from.

**Attention.** One head. The query is a linear map of the lead features, the solar, calendar, and
holiday columns, the ensemble weather summary, and the Primary embedding. Keys and values are linear
maps of the position vectors. A learned "no day" key and value token is always present, so a
window set that is entirely unobserved, or distrusted, can be ignored; the softmax runs over all
positions of all windows shown plus the token. The context vector (width 32) is concatenated with
the query inputs and passed to the same head as R (two hidden layers of 64, dropout 0.1, three
Gaussians, g09), so the head is unchanged and only the input route differs. About 13,000 more
parameters than R's 8,400.

**Training.** Experiment 397's loop: weighted negative log-likelihood, AdamW, early stopping on a
held-out week in every twelve (experiment 422's purge), monthly refits on the 24 months before the
test month. Every arm except R and L0 is trained on injected power: in every epoch each training row
draws, with equal probability, the clean series or one of three injected worlds (seeds 100 to 102)
for the power in its windows (and for R-inj's lags); the target is always the clean series. Worlds
come from `e00402_inject.inject` over 2024-05-01 to 2026-07-01 at 10% coverage (the generator of
experiment 402 with `FIRST_DAY` moved earlier), with the label table's padded real events excluded as
base periods. The event pool uses all labelled events including those after a fold's refit, and
each depth is a fraction of the donor's mean absolute power over its whole series; the pool
supplies only depths and durations, and both are listed as caveats. Three quarters of the
training rows are injected, which is more than real events, so clean-row contrasts partly measure
that training mix. Power windows are gathered on the GPU from dense per-Primary arrays, so no
window is stored.

## Arms

| Arm | Recent-power input | Trained on |
|---|---|---|
| R | the three origin lags | clean power |
| R-inj | the three origin lags | clean and injected power |
| L0 | none (lagless) | clean power |
| B-mix | fixed 50/50 mixture of R-inj and L0 distributions (not an average of quantiles) | no fit |
| E1 | encoder over D1 | clean and injected |
| E2 | encoder over D1 and D2 | clean and injected |
| E3 | encoder over D1, D2, and D3 | clean and injected |
| E3-trust | E3 plus experiment 404's trust inputs as extra tokens | only if the advance rule is met |

## Encroachment sweep

**Prediction-only, on saved networks, on the screen's clean rows.** A row is clean when its target is
finite and outside the padded real-event mask and the 48 hours after an event, every observed
half-hour of its three windows is outside the padded mask, and R's three lags are finite (experiment
392's `clean_rows`, extended to whole days). Every arm is swept on the same rows: every third daily
origin of each test month, leads 0.5 to 240 hours.

- **Steps.** Add a step D to the affected half-hours, both signs, of 2.5% and 5% of effective
  capacity, and of 10% of the Primary's trailing 30-day mean power before the origin. MVA-only
  readings are clipped at zero after the step, and rows where the clip changed any shifted value are
  dropped from the share (for all arms alike).
- **k for R, as in experiment 392.** k = 1 shifts `lag_origin`; k = 2 also `lag_same_time_latest_day`;
  k = 3 also `lag_same_time_latest_week`.
- **k for the encoders, two variants.** Variant 1 shifts single half-hours, in R's order: k = 1 the
  newest observed half-hour of D1 (R's `lag_origin`); k = 2 also the half-hour of D1 at the
  target's time of day (R's day lag); k = 3 also the centre half-hour of D2 (R's week lag); k = 4
  (E3 only) also the centre half-hour of D3. Variant 1 is the like-for-like counterpart of R's three
  single lags, and k up to 3 is compared with R. Variant 2 shifts every observed half-hour of whole
  windows: k = 1 shifts D1; k = 2 also D2; k = 3 also D3. Variant 2 is the event that fills a day,
  and R has no counterpart, so the comparison of variant 2 with R is harsher on the encoders.
- **Raw and widened.** The headline is raw (experiment 392's headline); the same table is repeated
  with the 1.15 widening about the median.
- **Lead bands.** 0-24, 24-72, 72-168, and 168-240 hours; the 24-72 hour band is the headline.
- **Reported.** The median shift of q50 as a share of the step (+1 is the full step, whichever the
  sign); the ratio of the 5%-95% band width after the step to the width before; coverage of the
  clean reading and of the event-level reading (actual plus step) by the 5%-95% band; the attention
  mass on D1, D2, D3, and the "no day" token, for every k and variant. Three seeds per fold.
- **Lag-masked control.** Every arm is also evaluated in evaluation mode (dropout off) with all
  its recent-power inputs masked, the step added, and the mask applied after the step; the
  forecast must equal the unshifted masked forecast to within 1e-6 (share 0). A larger difference
  is a bug.

## Evaluation rows

- **Clean rows (real data, no injection).** The scored months' rows whose recent days are outside
  every padded real event; the benchmark for loss outside events.
- **Harm rows (injected worlds).** Test-month rows from three held-out injected worlds (seeds 200
  to 202, never seen in training) whose target is outside every event and whose windows contain an
  injected event; clean rows of the same worlds are reported beside them.
- **Event rows (real).** Rows whose target is inside a padded labelled event of the corrected label
  table, and real harm rows (target outside, windows inside). The unlogged-event caveat applies:
  NGED may have events missing from its logs, so the clean-lag rows may hold events no label
  covers and the real-event slices undercount.

## Compute

**The fits are 6 arms by 4 folds by 3 seeds, 72 fits, plus a fifth fold if the screen advances.**
Experiment 397 timed the plain network at about a minute. The encoder is heavier because it reads
up to 144 positions per row. A first timing of E3 on the GPU took about 3 minutes per pass over the
4.7 million training rows of the 2026-01 fold, so every arm trains on a random 20% of the training
rows in each epoch (up to 60 epochs, patience 8), and early stopping reads a fixed random 200,000
of the held-out rows. A single CPU core was not timed, because the GPU epoch is already the
cheaper route. One GPU process under 8 GB with one feeder core, `MemoryMax=8G`,
`OMP_NUM_THREADS=1`, `free -g` checked before each launch (stop line 12 GB available). Fitted
networks are saved under the experiment's data folder. No stored fit is reused, because R's recipe
differs (the cut, the injected data). Stored forecasts of experiments 389, 87, and the baselines
are read as references only.

## Second screen: a surprise channel

**Arm E3-surp gives the encoder a sixth channel, the surprise.** The surprise of a half-hour is
its observed power minus the median of the cross-fitted lagless forecast for the same half-hour
(experiment 404's table, K = 6 blocks, never an in-sample lagless forecast), divided by the width
of that forecast's 5%-95% band, clipped at 5, and 0 where either is unobserved. The windows
otherwise match E3. It runs as a second screen when experiment 404's table exists, on the same
four folds, rows, and seeds, with the prediction written before any score is read: E3-surp minus
E3 on harm rows at the signed-MW six lies between -0.20 and -0.03 CRPS points, and E3-surp follows
at most half of E3's share of the step in the sweep at k = 3. The table must cover each refit
without the scored month; if it does not, the arm waits.

## Design iterations

The maintainer wants at least two design iterations of the tokens and the attention after the first
screen, before any separate lag-trust evaluation runs on the saved networks. Each iteration is a
change to the encoder (for example a second head, a window of a different length, a learned
position code, or the trust inputs of experiment 404 as extra tokens), registered with its own
prediction before its scores are read, and scored on the same folds and rows. This page records
the first screen only; each iteration adds an arm and a dated section.

## Checks, reviews, and what the page will report

- **Departures from experiment 397's recipe, found by the code review.** Every arm (R included)
  trains on a random 20% of the rows per epoch with batch 8,192 (experiment 397: full epochs, batch
  16,384, patience 6), and standardises the three lags with one pooled mean and standard deviation
  of the cleaned targets (experiment 397: one per lag column). Scores use the 1.15 widening; CRPS
  is the mean of the three seeds' scores, and the tail pinball and coverage are computed from the
  seed-averaged quantiles. Rows with a valid time on or after 2026-07-01 are not scored.
- **The avoidable-curtailment cost is not computed at the screen.** The harm-row sets have no node
  table; it is computed for the clean rows if the screen advances.

- **Reviews.** A fresh Sonnet reviews this plan and then the code before the first fit; a fresh
  Sonnet re-derives the headline numbers; a fresh Sonnet reviews the write-up and its prose. The PR
  body records what each found.
- **Reported per group and per Primary:** CRPS, q01, q05, q95, q99 coverage and pinball loss, cold
  winter nights, the avoidable-curtailment cost, absolute CRPS beside every contrast with its
  interval, and the minimum detectable effect.
- **Figures.** CRPS by lead for every arm and the references (headline); sweep share and band ratio
  against k; attention weight per day and per position; at least six real-event time series plots
  across both metering groups; a day-ahead and a 14-day example forecast. Generators are
  anonymised.
- **Audit checks.** The median forecast minus the mean forecast per site; the error of the mean
  forecast beside each quantile-score contrast; which audit triggers fired.

## What it can and cannot say

Four months give wide intervals: the screen can stop the idea or rank it for the queue, and cannot
support a claim of benefit. The injected worlds are synthetic; the real-event rows are few. The
encoder sees days of raw power at the forecast origin only, so it uses nothing a live service
would lack (CAMS and ERA5 are not used at all).
