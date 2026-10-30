---
id: "00419"
slug: pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports
title: >-
  Do the solar summary, the sunny flag, and a different row-weight half-life, which helped the
  XGBoost model, also help the pooled mixture density network?
theme: production-reference-and-best-guess
also_relevant_to:
  - retraining-and-training-history
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: forecasts, fitted networks, and small result tables in this experiment's private
  folder only; experiments 381, 397, 406, and 407 are read and never written. Designs: more than one
  defensible (how the solar summary is computed from the ensemble, the sunny threshold, the
  half-lives), so every arm is registered. Callers: the scripts import experiments 87, 48, 377, 381,
  397, and 407 and the `studies` evaluation code, whose callers sit in those folders. Serving path
  and degradation rule: not touched (this repository holds neither).
hypothesis: >-
  The XGBoost model's gain from a two-column solar summary (experiment 381: -0.051 points at the
  signed-MW six on the scored year) carries over to the pooled plain neural network, which has the
  solar geometry but no cloud-aware summary, and the 0.25-year half-life that lowered the score at
  D15 and D18 and on cold nights in experiment 407 does so again with new random seeds.
prediction: >-
  Written before any score is read. The reference is `ctrl_w115`, the pooled plain network widened
  by 1.15 about the median, stored by experiment 407 (CRPS 3.277 at the signed-MW six and 3.858 at
  the MVA-only eight). S1: the solar summary (`sol`) lowers the CRPS at days 1 to 10 at the
  signed-MW six by at least 0.02 points and lowers it on sunny April-to-June half-hours by at least
  0.05 points. S2: the sunny flag alone (`sunny`) moves the CRPS by less than 0.02 points at both
  groups (experiment 381 found -0.002). S3: `sol_sunny` is within 0.02 points of `sol` at both
  groups. S4: the 0.25-year half-life retested with new seeds (`hl91r`) again lowers the CRPS at D15
  and D18 by at least 0.10 points and on cold nights by at least 0.05 points, and raises it by no
  more than 0.10 points at the signed-MW six (experiment 407 found -0.183, -0.162, and +0.073). S5:
  `sol_hl91` lowers the CRPS at D15 and D18 by at least 0.10 points against `sol`. S6: `bundle`
  (the solar summary, the sunny flag, and a 1-year half-life) is within 0.03 points of `sol_sunny`
  at both groups (experiment 407 found +0.009 for the 1-year half-life alone).
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight; D15 and D18, report-only) on CRPS
  at days 1 to 10, with the month-block bootstrap of `studies.eval_protocol` at 95% and 20,000
  resamples. A gain counts only if the interval lies wholly below zero and the point estimate is
  below minus the minimum detectable effect, which is 2 times the standard deviation, across the 15
  fold-and-seed pairs, of the difference between two fits of the same configuration with different
  seeds (computed from the 407 `ctrl` fits and the 406 `pooled_emb` fits and written on this page
  before any new arm is scored). The five screen months give wide intervals, so every number is
  interim and a result ranks changes for further testing without supporting a claim of benefit. The
  hypothesis is supported if S1 holds and counts as a gain; partly supported if S1 fails but S4
  holds; not supported if both fail. An arm is carried forward if its prediction holds and its
  interval does not lie wholly above zero. The trailing-temperature arm (`temp`) runs only if
  experiment 396's arm W1 reports a gain that outlasts its noise floor.
benchmark: >-
  The widened pooled plain network `ctrl_w115` and the 0.25-year half-life arm `hl91` of experiment
  407, the per-Primary network of experiment 406, and experiment 381's XGBoost bundle, all on the
  same half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 336
  hours; the headline band is days 1 to 10.
code: ml-experiments/00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports/
---

**Take-home message (to be written when the result is known).**

This page is the plan. It is committed before any neural network is fitted, so that its commit order
proves the predictions came first.

## Question and design

**We add one XGBoost input change at a time to the pooled plain neural network and compare each
change with the same neural network widened by 1.15.** Experiment 381 found that two columns holding
the daily mean of the ensemble's shortwave radiation, today and yesterday, lowered the XGBoost
model's CRPS by 0.051 points at the signed-MW six. The pooled plain neural network of [experiment
397](../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md) has the solar
geometry (cosine of the zenith, sine and cosine of the azimuth) and the ensemble mean and spread of
shortwave at the valid time, but not that daily summary and not a cloud-aware sunny flag.

- **The neural networks.** The pooled plain network of experiment 397 (setting `g09`, summary
  scheme, an 8-wide Primary embedding, three Gaussian components) on its five screen folds (test
  months 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06), refitted monthly, three seeds averaged in
  quantile space, with the same scoring rows. Every arm is widened by 1.15 about the median on q01,
  q05, q95, and q99, as the reference is.
- **`ref`.** The reference `ctrl_w115` stored by experiment 407 (not refitted).
- **`sol`.** Adds `shortwave_day_mean` and `shortwave_prev_day_mean` (experiment 381's arm A2),
  computed from the ensemble-mean shortwave of the same NWP run over the local day of the valid
  time and over the previous local day, and standardised like the other weather inputs.
- **`sunny`.** Adds `sunny_flag` and `sunny_times_shortwave` (experiment 381's arm A4): the flag is
  1 where the ensemble-mean shortwave exceeds 0.6 times the Haurwitz clear-sky irradiance.
- **`sol_sunny`.** Adds both groups.
- **`hl91r`.** The reference with row weights that halve every 0.25 years, fitted again with seeds
  3, 4, and 5 (experiment 407's `hl91` used seeds 0, 1, and 2). The 0.25-year half-life is carried
  from experiment 407.
- **`sol_hl91`.** `sol` with the 0.25-year half-life carried from experiment 407.
- **`bundle`.** `sol_sunny` with row weights that halve every 1 year: experiment 381's weekly
  bundle, here refitted monthly.
- **`temp`.** Adds the trailing means of the 2-metre temperature over 6, 24, 72, and 168 hours and
  the matching heating degrees below 15.5 degrees Celsius (experiment 396's arm W1). It is
  registered here and is fitted only if experiment 396 reports that W1 beats its noise floor.
- **The start cut.** D15 and D18 lose every training row before 2025-03-01, and every row whose
  weekly lag reaches back before that date (as in experiments 406 and 407). The scoring rows start
  in 2025-10, so no scoring actual or inference input is affected.
- **Not in this experiment.** A 1-year half-life alone, because experiment 407's `hl365` already
  tested it (+0.009 at the signed-MW six); an explicit solar term, which experiment 416 owns.

## Components adopted from earlier results

- **Adopted.** One pooled network with a Primary embedding (experiment 403); the ensemble mean and
  spread passed through the network once (experiments 397 and 394); monthly refits; the 1.15
  widening and the 0.5-year half-life of the control (experiment 407: no tail arm and no other
  half-life beat the widened control at the signed-MW six); and the D15 and D18 start cut
  (experiment 399).
- **Carried for retest.** The 0.25-year half-life, because it lowered the score at D15 and D18 and
  on cold nights in experiment 407 although its interval at the signed-MW six included zero.
- **Not adopted.** Attention over input groups (experiment 397: the gate stayed within 0.01 of
  uniform), the differentiable physics network (experiment 413: worse than the plain network by
  0.17 and 0.23 points), and any tail arm (experiment 407).

## Standing practices followed, and the deviations

- **Followed.** Monthly refits; every fitted network saved in the private data folder; stored
  forecasts reused as references and never refitted; a few-fold screen with every number labelled
  interim; tails (coverage and pinball loss at q01, q05, q95, and q99, on all half-hours and on
  sunny low-demand half-hours) and the avoidable-curtailment cost reported beside the CRPS; example
  forecast charts; a fresh Sonnet plan review and a fresh Sonnet code review before the first fit.
- **Cleaned data.** Training rows drop every half-hour that the cleaned power table flags as an
  exact zero (`substation_zero`) and every row whose lag reaches one, and the scorer drops flagged
  half-hours from the actuals of every arm including the references. The number of rows dropped is
  printed for each fold. The reference was fitted with the same rule in experiment 407.
- **Departure: the sanity contrast.** Experiment 407's `ctrl` differs from experiment 406's pooled
  plain network by -0.021 points at the signed-MW six, which is the size of the change that the
  cleaned training rows made. Differences below that size are not read as effects.

## Fit counts, time, and memory

- **Fits.** 6 arms by 5 folds by 3 seeds, so 90 fits, and 15 more for `temp` if it runs. The
  widenings need no refit.
- **Time.** About 40 to 50 seconds per fit on the GPU, so about 1.5 hours in total.
- **Memory.** One GPU process with at most 8 GB of host memory and one feeder core. The scorer runs
  in passes of a few arms each, because holding every arm's rows at once needs more than 14 GB.

## Checks the write-up reports

- Absolute CRPS of every arm beside the reference and experiment 406's per-Primary network, per
  group and per Primary.
- Coverage and pinball loss at the four outer levels, on all half-hours and on sunny low-demand
  half-hours (the sunny half-hours whose reference median is in the lowest quarter for the Primary).
- The CRPS on sunny April-to-June half-hours, on cold nights, and on Economy 7 step hours.
- The cost of avoidable curtailment at the scaled export limit, if the scorer prints it.
- Example forecasts for D10 (signed-MW) and D4 (MVA-only): a day-ahead forecast and a full 14-day
  forecast from one run, in winter (made in January) and summer (made in June), of the best arm.

## Caveats registered before the scores

- **The solar summary is built from the ensemble mean.** Experiment 381 built it from the control
  member, so a difference from that experiment's effect may come from the mean.
- **Experiment 381 chose its arms on the year it scored**, so its -0.051 is an upper bound.
- **Experiment 416 is testing an explicit solar term** and found no gain at 5 months; a gain from
  `sol` is not evidence about an explicit physical term.

## Counting towards the demand-v2 stop rule

**This experiment does not count.** It tests input changes on an existing model, not a new demand
model, so no idea's count rises whatever the verdict.

## Problems caused by the move

None so far.
