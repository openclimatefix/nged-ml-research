---
id: "00421"
slug: mdn-history-fingerprint-for-pooled-primary-demand
title: >-
  Can the pooled mixture density network tell which Primary it is forecasting, and forecast a
  Primary it never saw, from the last 7 days of that Primary's power alone?
theme: hierarchy-and-pooling
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: forecasts, fitted networks, history tables, and probe scores in this experiment's
  private folder only; experiments 397 and 406 are read and never written. Production serving path:
  not touched (this repository holds none). Degradation rule: not touched. Designs: more than one
  defensible (the length of history, the encoder, whether the target is rescaled per Primary), so
  the arms register each. Callers: the scripts import experiments 87, 48, 397, and 406 and the
  `studies` evaluation code, whose callers sit in those folders.
hypothesis: >-
  A 1D convolutional encoder over the last 7 days of a Primary's own demand gives the pooled plain
  mixture density network enough information to infer which Primary it forecasts, so that the
  network without a learned Primary embedding recovers most of the embedding's gain, and a Primary
  held out of training is forecast better than by a network given a placeholder embedding.
prediction: >-
  Written before any fit and before any score is read. P1 (target scale control). Experiment 406's
  network with no embedding lost 0.789 points to the network with the embedding at the signed-MW
  six, but its target was standardised with each Primary's own mean and standard deviation, which
  a network with no Primary input cannot invert. With one global target scale, E3g (embedding,
  three origin lags) lands within 0.05 points of experiment 406's pooled network with the
  embedding (U0) at both groups. P2. The no-embedding network with three origin lags (N3g) loses
  less than half of that 0.789 to E3g at the signed-MW six (N3g minus E3g at most +0.39) and less
  than half of 0.809 at the MVA-only eight (at most +0.40), because the lags carry the level.
  P3. The 7-day encoder (NH, no embedding, three origin lags plus the encoder) scores at least 0.15
  points below N3g at the signed-MW six. P4. NH is within +0.10 points of EH (the embedding and the
  encoder) at both groups. P5. EH is no more than 0.05 points below E3g at either group (once the
  identity is known, the 7 days add little). P6 (probe). A linear classifier on the encoder's
  32-wide output recovers the Primary of an origin with an accuracy of at least 0.90 on held-out
  weeks, against at most 0.50 for the same classifier on the three origin lags, and recovers the
  month of the year to within one month at least 0.60 of the time. P7. NH0 (no embedding, the
  encoder but no origin lags) is within 0.05 points of NH at both groups, because experiment 388
  found that dropping the newest reading changes the days 1 to 10 score by -0.022 and +0.001. P8
  (held-out Primaries, run only if the advance rule below holds). For a Primary left out of
  training, NH scores at least 0.20 points below E3g given the mean embedding of the other
  Primaries, at the signed-MW six.
decision_rule: >-
  Each contrast is read per group (signed-MW six; MVA-only eight with and without D8; D15 and D18
  report-only) on CRPS at days 1 to 10 (days 11 to 14 descriptive), paired by fold and seed, with
  the month-block bootstrap of `studies.eval_protocol` at a 95% level Bonferroni-corrected across
  the eight planned contrasts (99.375%), 20,000 resamples. Planned contrasts. C1 E3g minus U0. C2
  N3g minus E3g. C3 NH minus N3g. C4 EH minus E3g. C5 NH minus EH. C6 NH0 minus NH. C7 (held-out
  Primaries, six and eight) NH minus E3g with the mean embedding. C8 (held-out Primaries) NH minus
  N3g. The five screen months give wide intervals, so every number is interim and no claim of
  benefit rests on it. The hypothesis is supported if P3 and P6 both hold, partly supported if
  exactly one holds, and not supported if neither holds. The held-out stage (C7, C8) advances only
  if the probe recovers the Primary at accuracy 0.50 or higher on held-out weeks, or C3 is at most
  -0.10 points at the signed-MW six; otherwise it is skipped and the page says that no fingerprint
  was found. The smallest gain worth porting the encoder into a research default is 0.03 points at
  the signed-MW six with no loss above +0.02 at the MVA-only eight (C4), because the encoder adds a
  module and a per-origin history table to the pipeline. No adoption is sought from this
  experiment.
benchmark: >-
  Experiment 406's pooled network with the embedding (U0, `pooled_emb`) and with no embedding
  (`pooled_noemb`), and its per-Primary network, on the same half-hours; BG2, `manual_heuristic`,
  and `climatology` as absolute-skill references; experiment 397's stored forecasts of the monthly
  XGBoost control.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and the avoidable-curtailment cost, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00421-mdn-history-fingerprint-for-pooled-primary-demand/
---

**Plan: does the last week of a Primary's power identify the Primary to a pooled neural network?**
The status is `planned`. Nothing below has been run.

## Question and design

**We test whether recent power plays the part of a language model's context.** The maintainer's
idea is that the 7 days before the forecast origin tell a pooled network which Primary it is
looking at, and where in the seasonal cycle that Primary sits, without a learned Primary embedding.
The maintainer does not propose dropping the embedding. The experiment asks whether the history
could replace it, which matters for the 1,300-substation case, where a new substation has no
embedding of its own.

- **The base network.** Experiment 397's pooled plain mixture density network (setting `g09`,
  3 Gaussian components, ECMWF ENS mean and spread passed through once), on the five screen folds
  of experiment 397 (test months 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06), monthly refits,
  with the D15 and D18 start cut of experiment 399, as in [experiment
  406](../00406-pooled-mixture-density-network-without-embedding-and-with-short-history/result.md).
- **The history.** For each origin, the 336 half-hours (7 days) of power over effective capacity
  that end at the newest usable reading (grid index `origin - NEWEST_AGE`, the same reading as
  `lag_origin`). Every run is daily at 00:00 UTC, so every history ends at the same time of day. The
  history comes from the same cleaned grid as the three origin lags (`observed.parquet` of
  experiment 87, with the flagged `substation_zero` half-hours as nulls), so the lags and the
  history agree. Values are standardised with the fold's training rows and clipped at 5 standard
  deviations. A missing half-hour is set to 0 and flagged in a second channel, so the encoder sees
  a gap and not a zero. A check before any fit asserts that the last history value equals
  `lag_origin` on every row, and that no history index is later than the newest usable reading.
- **The encoder.** Two 1D convolution layers (kernel 5, 16 channels, stride 2, GELU) over the two
  channels (value and missing flag), then the flattened output (84 steps by 16 channels) through one
  linear layer to 32 values with a LayerNorm. The flattened output keeps the time of day, which a
  global pooling would lose. The 32 values join the network's other inputs.
- **The target scale.** Experiment 406's targets were standardised per Primary, so a network with
  no Primary input could not invert the scale, and its 0.789-point loss mixes "no identity" with
  "no scale". All new arms therefore use one global target mean and standard deviation, fitted on
  the fold's training rows, and the arms with a letter `g` in the name carry that change.
- **Experiment 404 does not make any arm redundant.** Experiment 404 (plan on [PR #58 (Experiment
  404: a lagless module and a calibration module with continuous trust weights for the lagged power
  inputs)](https://github.com/openclimatefix/nged-ml-research/pull/58), no fit run yet) attends over
  three lag tokens and a lagless-forecast token (arm B3). Its arm B7 adds a 7-day CNN, but only on
  top of the trust head and the cross-fitted lagless forecast, and it is the lowest priority. The
  arms below ask the narrower question of identity, so they run now.

## Arms

Names say which components each arm carries: `g` is the global target scale, `3` is the three
origin lags (`lag_origin`, `lag_same_time_latest_day`, `lag_same_time_latest_week`, each with its
age and missing flag), `H` is the 7-day encoder, `E` is the 8-wide Primary embedding, and `N` is no
embedding.

- **U0 (stored).** Experiment 406's pooled network with the embedding and per-Primary targets
  (`pooled_emb`). Zero fits; read with `experiment_store.read_predictions`.
- **E3g.** Embedding, three origin lags, global target. The control for everything below.
- **N3g.** No embedding, three origin lags, global target. Experiment 406's `pooled_noemb` made
  scale-fair.
- **NH.** No embedding, three origin lags, and the 7-day encoder. The arm the hypothesis names.
- **EH.** Embedding, three origin lags, and the 7-day encoder.
- **NH0.** No embedding, the 7-day encoder, and no origin lags. Tests whether the history makes the
  lags redundant, as experiment 388's arm E suggests.
- **Held-out stage (advance rule above).** Arms N3g, NH, E3g, and EH, each fitted four times per
  fold and seed with one quarter of the Primaries (every fourth, by index) left out of training
  entirely, and scored only on that quarter. E3g and EH have no embedding for an unseen Primary and
  receive the mean embedding of the Primaries they trained on (a placeholder, named in the
  tables). Left-out Primaries are also scored against `climatology` and `manual_heuristic`.

**The probe.** After each fit of NH, EH, and NH0, a linear softmax classifier is trained on the
frozen 32-wide encoder output of every training origin to predict the Primary (16 classes) and the
month of the year (12 classes), and scored on the held-out early-stopping weeks (purged exactly as
in experiment 397, so no probe row sits within 14 days before or 7 days after a held-out week). It
is also scored on the origins of the test month for the Primary label. Reference probes use (a) the
three origin lags with their missing flags, (b) the raw 336-value history, and (c) the encoder at
random initialisation. A chance level is printed beside each accuracy.

## Components adopted from earlier results, and not adopted

- **Adopted.** Pooled network with an 8-wide embedding and the ensemble mean and spread passed
  through once ([experiment
  397](../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md) and
  [experiment 403](../00403-pooled-versus-per-primary-mixture-density-network/result.md)); the
  three origin lags, with no extra weekly analogues ([experiment
  388](../00388-xgboost-lagged-power-column-sets-weekly-demand/result.md): six weekly analogues
  raised the CRPS by 0.137 points);
  the D15 and D18 start cut ([experiment
  399](../00399-xgboost-short-history-start-cut-d15-d18/result.md)); the Primary embedding in every
  arm that can use one ([experiment
  406](../00406-pooled-mixture-density-network-without-embedding-and-with-short-history/result.md):
  dropping it cost 0.789 points); the 1.15 widening is reported beside the raw scores, and the
  Gaussian 3-component head is kept ([experiment
  407](../00407-pooled-mixture-density-network-tail-arms-and-recency-weights/result.md): no tail arm
  beat the widened pooled network).
- **Missing-lag handling.** [Experiment
  410](../00410-mdn-stale-weather-and-delayed-power-rescore/result.md) found that the pooled network
  loses 4 to 10 points when the three newest-reading lags are missing, so the encoder's missing
  channel matters. Each write-up table of the encoder arms includes the share of origins whose
  history has any gap.
- **Not adopted.** Attention over input groups (experiment 397: the gate stayed within 0.01 of
  uniform). The lag-trust architecture, the explicit PV term, ERA5 and CAMS pretraining, and the
  input ports (experiments 404, 405, 418, and 419, none merged), because
  adopting an unmerged component would tie this experiment to a result that may not hold.
  Combining a winner of this experiment with those components belongs to the best-architecture
  assembly.

## Standing practices

- **Followed.** Monthly refits; cleaned data (see "The history"; the training rows are experiment
  397's rows, so flagged exact-zero targets stay in training as in experiment 406, and the scorer
  drops them from the actuals and prints how many); the D15 and D18 start cut in the training rows,
  lags, history, and scoring actuals; every fitted network saved in the private folder; stored
  forecasts reused as references and never refitted; a screen of five folds with every number
  labelled interim; tails (coverage and pinball loss at q01, q05, q95, and q99, raw and widened by
  1.15) and the avoidable-curtailment cost beside CRPS; example forecast charts (a day-ahead
  forecast and a 14-day forecast, for one signed-MW Primary and one MVA-only Primary); a fresh
  Sonnet plan review and a fresh Sonnet code review before the first fit.
- **Lower-tail check.** The tables report the lower tail (q01 and q05) on sunny, low-demand
  half-hours too, because curtailment depends on it.
- **Departure.** The training rows are experiment 397's rows, as in experiment 406, so the fits stay
  paired with the stored controls. Experiment 398 found that dropping flagged half-hours moved the
  CRPS by at most 0.008 points in four earlier experiments.
- **D8.** Any D8 figure carries the caveat "it is unproven whether the generator explains the steps
  and the gap".

## Fit counts, time, and memory

- **Main stage.** 5 new arms (E3g, N3g, NH, EH, NH0) by 5 folds by 3 seeds is 75 pooled fits, and
  U0 adds none.
- **Held-out stage.** 4 arms by 4 quarters by 5 folds by 2 seeds is 160 pooled fits (seeds 0 and 1
  only, because each fit trains on 12 Primaries). It runs only if the advance rule holds.
- **Probes.** 45 probe sets (3 encoder arms by 5 folds by 3 seeds), each a few seconds on the GPU.
- **Time.** Experiment 406 fitted one pooled plain network in about 50 s on the GPU. The encoder
  adds convolutions over 336 steps for every row, so the estimate is 60 to 120 s per fit, and the
  whole experiment is expected to take 4 to 8 hours. The first task after the fit slot is granted
  is to time one fit of each arm and replace this estimate.
- **Memory.** One GPU process under 8 GB of GPU memory, one feeder core, `OMP_NUM_THREADS=1`, and
  `systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue`. The history table is
  16 Primaries by about 820 origin days by 336 half-hours in float16 (about 9 MB) plus a missing
  mask, gathered per batch by (Primary, origin day). The fit script logs the peak GPU memory.
- **Stop line.** The fit script reads `MemAvailable` from `/proc/meminfo` before each fit and exits
  cleanly, leaving finished fits in place, when it falls to 12 GB or below. Every fit writes its own
  file atomically, so a stopped run resumes where it stopped.

## Checks the write-up reports

- Absolute CRPS of every arm beside U0, experiment 406's per-Primary network, BG2,
  `manual_heuristic`, and `climatology`, per group and per Primary.
- The median forecast minus the mean forecast per Primary, the error of the mean forecast beside
  every quantile-score contrast, and which audit triggers fired.
- The probe accuracies with chance levels, a confusion table of the Primaries the probe confuses,
  and the share of origins whose history has a gap.
- Example forecasts: one day-ahead forecast and one 14-day forecast for one signed-MW Primary and
  one MVA-only Primary, from NH and from E3g.
- **Unlogged load-shift events.** NGED says load-shift events may be missing from its logs. The
  history of an origin can sit inside an unlogged event, so the encoder may read an event as a
  property of the Primary. The write-up states that the training and scoring rows can hold events no
  label covers.

## Counting towards the demand-v2 stop rule

**This experiment does not count.** It tests where a pooled network gets its sense of identity, not
a new demand model, so no idea's count rises whatever the verdict.

## Problems caused by the move

None so far.
