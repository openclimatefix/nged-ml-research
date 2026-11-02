---
id: "00510"
slug: lagless-demand-network-weekly-finetune-and-recency-weights
title: >-
  Can a demand network that reads no lagged power, fine-tuned every week with a recency weight, get
  close to K2-S1 on ordinary rows and avoid the load-shift problem?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: the weekly and monthly fine-tuned networks, their raw quantile forecasts, and the
  stress-run forecasts, under this experiment's private data folder only; every earlier experiment is
  read and never written. Serving path: not touched (this repository holds none). Degradation rule:
  not touched. Designs: more than one defensible (what a weekly fine-tune trains on, how the
  injected shift is defined, how the outlier component is parameterised), so the plan fixes each
  before any score is read. Callers: this folder's scripts import experiments 397, 425, 428, 448,
  and 470, and the `studies` evaluation code. One Sonnet plan review, one Sonnet code review before
  the first fit, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  A demand network with no lagged-power input, pretrained once on ERA5 and CAMS rows and then
  fine-tuned every week on ensemble-forecast (ENS) rows, closes part of the gap to the lagged
  network K2-S1 on rows with no labelled event, because a weekly refit learns the most recent
  level that the lags would otherwise carry. A 90-day recency weight closes more of the gap, at the
  price of learning a recent load-shift event as normal; dropping labelled events from the
  fine-tune, or a latent outlier component that is discarded at serving, reduces that leak.
prediction: >-
  Written before any score is read. P1: L-W (weekly) beats L-M (monthly) by 0.02 to 0.08 points of
  CRPS at the six and 0.01 to 0.05 at the eight, less than the 0.13 that experiment 48 found for
  XGBoost with lags. P2: L-Ww (weekly, 90-day half-life) beats L-W by under 0.05 points at both
  groups. P3: no arm gets within 0.1 points of K2-S1 at the six (the gap to close is 0.23 to 0.32).
  P4: the injected-shift leak of L-Ww is larger than L-W's, and L-Ww-D and L-Ww-O each cut it by less
  than the minimum detectable effect.
decision_rule: >-
  See "Pre-registered rules" in the plan. Limits of 0.03 and 0.05 sit below the minimum detectable
  effect (0.086 at the eight and 0.170 at the six with 2 seeds and 5 folds, experiment 494), so a
  pass at those limits reads "not shown worse" and a pass at -0.03 can only read "interval excludes
  zero". Bonferroni count 30 CRPS tests (60 with q99 pinball loss).
benchmark: >-
  K2-S1 (stored, experiment 470) and A-pool (experiment 425 arm A, five seeds averaged), on the same
  rows; BG2, the lagless average of experiment 472, climatology, and `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
---

**Planned; no score has been read.** The plan below is the one the brief and the queue-order check
fixed on 2026-11-02, with the corrections the check made.

## Plan

### Question

Can a demand network that reads no lagged power at forecast time, pretrained once on ERA5 and CAMS
rows and then fine-tuned every week or every month on ENS rows, match K2-S1 and A-pool on rows
with no labelled event while avoiding the load-shift problem entirely?

**The gap to close is 7% to 10% at the signed-MW six.** On ordinary rows the lag-free average of
experiment 472 scores 3.622 and a single-seed lag-free network about 3.71, against 3.387 for
K2-S1: 0.23 to 0.32 points. (The brief's earlier "4% to 5%" is not what experiment 482 shows.)

### Base network (checked before this plan)

- **L0PT is arm A minus the lag columns.** Experiment 448's `L0PT` weights and experiment 425's arm A
  are the same `PlainMdn` of experiment 397 (setting `g09`: hidden 64, 2 layers, 3 components,
  dropout 0.1, 8-wide Primary embedding), built from the same `summary` feature scheme. Arm A takes
  the scheme's 46 columns plus the embedding (first-layer weight 64 x 54). L0PT takes the 37 columns
  left after dropping the scheme's nine lag columns (`lag_origin`, `lag_day`, `lag_week`, their
  `_missing` flags, and their `_age` columns) plus the embedding (64 x 45). Every other weight has
  the same shape. Arm A's own pretrained weights have the lag columns and do not load.
- **Pretraining rows.** L0PT folds exist for 2025-10, 2026-01, 2026-03, and 2026-06 (5 seeds each).
  Fold 2025-12 is pretrained here (seeds 0 and 1, about 46 s each) with experiment 448's
  `pretrain_rows` and loop.

### Design

- **Folds and rows.** The five experiment 449 folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) on
  the rows of `experiment_470_intersection_v1`. July 2026 is unused. The D15 and D18 start cut
  (2025-03-01) and the flagged exact-zero drop apply (cleaned data). Reported per fold.
- **Standing practices followed.** Cleaned data; D15 and D18 cut; saved networks; stored forecasts
  reused as references; tails and curtailment cost; example forecasts and the 14-day plot; load-shift
  event plots (at least 6, both metering groups); the process order of the ml-experiment skill.
  Not followed: the monthly-only rule, because the question is the refit cadence.
- **Arms (pooled over the 16 Primaries; two seeds, 0 and 1; widened by 1.15 unless said):**
    - **L-M:** fine-tuned monthly with experiment 425's recency weight, `0.5 ** (age_days /
      182.6)`. The queue-order check found that 425's recipe already weights rows with a half-life
      of 182.6 days, so the "unweighted" arms of the brief are weighted at 182.6 days.
    - **L-W:** fine-tuned weekly with the same 182.6-day weight.
    - **L-Ww:** fine-tuned weekly with a 90-day half-life. L-Ww against L-W is the one recency test.
      L-Mw (monthly, 90 days) is dropped: experiment 407 tested a 0.25-year half-life with the pooled
      network (+0.073 at the six, +0.025 at the eight, not better).
    - **L-Ww-D:** L-Ww on rows minus the half-hours inside labelled events (see "Appended arms").
    - **L-Ww-O:** L-Ww with a latent outlier component (see "Appended arms").
    - **Stress runs** of L-W and L-Ww (see "Stress run").
- **Weekly refit rule and leak check.** For scored month M the refits are at 00:00 UTC on days 1, 8,
  15, 22, and 29 of M (five refits; the last covers the days left in the month). Each refit
  fine-tunes the month's L0PT weights on rows with valid time at most the refit minus 3.5 h and
  origin at most the refit (experiment 425's `build_ens_fold` rule, with the refit set to the
  week's first origin), over the 24 months before the refit. The early-stopping held-out weeks
  (the absolute weeks of experiment 422's `purge_masks`) must end before the refit as well. The
  forecast of a scored week uses only its own refit. A script prints, for every refit, the latest
  training valid time and origin and the number of rows past the cut (it must be 0) before any
  score is computed. Week 1 of each month is the monthly refit, so L-M and the first week of L-W
  share one fit per seed. The month's standardiser, target mean and standard deviation, and
  spread table are fitted once at the monthly refit and kept for the month's weekly refits, so the
  pretrained weights see the input scale they were pretrained on.
- **Fits.** L-M and L-W: 5 folds x 5 refits x 2 seeds = 50 fits, of which 10 are shared monthly
  fits. L-Ww: 50 fits. L-Ww-D and L-Ww-O seed 0 only: 25 fits each, plus 5 re-pretrainings with 4
  components (seed 0, about 46 s each) and 5 matching seed-0 pretrainings for the L-Ww control.
  Stress runs: 10 fits for L-W and L-Ww (one refit week per fold, one seed, both signs), and 10 for
  L-Ww-D and L-Ww-O. About 165 fine-tunes at 30 to 31 s plus 12 pretrainings at 46 s: about 1.7 GPU
  hours. The exact fit count and GPU-hours are reported to the manager before the fits run.
  One GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`) and one feeder core. Each month's
  rows are built once and masked by cutoff.
- **References (stored, no refit):** K2-S1, the lag-free average of experiment 472, A-pool (five
  seeds, and its two-seed version), BG2, `manual_heuristic`, and climatology.

### Stress run

- **The shift.** 3 times the standard deviation of a Primary's 7-day mean target (printed per
  Primary), added to the training target of the 7 days before the refit, excluding the
  early-stopping weeks, both signs. One refit week per fold (the week after the injection, chosen
  before any score is read as week 3 of each month), one seed. L-W and L-Ww (and L-Ww-D, L-Ww-O)
  are refit on the same injected rows, with matched uninjected controls from the main run.
- **What it measures.** The leak is the CRPS and the signed bias of the following week's forecast
  against the matched uninjected fit. The effective weight of 7 days in a window of about 730 is
  small (about 5% at the 90-day half-life), so the leak may fall below the minimum detectable
  effect. A lag-free network is not exposed to a bad lag at forecast time; the training targets are
  exposed, and this arm measures that.

### Appended arms

- **L-Ww-D (drop labelled events).** Fine-tune on L-Ww's rows minus the half-hours inside NGED's
  labelled events, padded by 2 days either side. Only events whose label is known before the refit
  count: an event that started before the cutoff and ends after it is dropped from its start to
  the cutoff, with no use of its future end, and the 2-day pad applies only before the cutoff. The
  rows leave the training set and the early-stopping week alike. NGED's labels are hindsight labels,
  unavailable in real time at a weekly refit, and they are incomplete, so unlabelled events stay in.
  Report the share of rows dropped per fold and, per scored week, whether it contains or follows an
  event. The injected shift is unlabelled by construction, so L-Ww-D cannot drop it.
- **L-Ww-O (latent outlier component, dropped at serving).** A fourth Gaussian component whose mean
  is the mixture's mean (weights-averaged main mean), whose scale is the widest main scale times (3 +
  softplus(learned offset)), and whose weight has a Beta(3, 97) prior (mean 3%) that enters the
  fine-tune loss only. The fourth component changes the output head, so the arm is re-pretrained with
  4 components (5 folds, seed 0) and L-Ww gets a seed-0 control pretrained the same way. At serving
  the outlier component is dropped and the main weights are renormalised. The 1.15 widening no
  longer applies: the widening sweep 1.0, 1.05, 1.10, and 1.15 runs per arm, with q95 and q99. The
  learned outlier weight is reported by row class (rows with no labelled event, rows inside labelled
  events, cold winter nights with their count, Christmas week).
- **Seeds.** Seed 0 is a screen for L-Ww-D and L-Ww-O. Any arm that passes runs at seed 1 (about 25
  more fits) before an adoption claim. Comparisons are paired with seed-0 L-Ww and with the
  two-seed L-Ww; the minimum detectable effect comes from the two-seed arms and is indicative only.
- **Held, not run:** a two-sided step repair of the training targets (held until experiment 507
  reports), a penalty holding the fine-tune near the pretrained weights, and a robust 28-day median
  level adjustment.

### Scoring

- **Scorer.** The experiment 470 scorer, tail script, and curtailment script pointed at this
  folder, as in experiment 494. Predictions are raw quantiles in experiment 425's seed-prediction
  layout, widened by 1.15 (L-Ww-O: by the sweep).
- **Minimum detectable effect, stated before any score:** 0.086 at the eight and 0.170 at the six
  (2 seeds, 5 folds, experiment 494), recomputed from this experiment's own seed spread and printed.
- **Headline contrasts:** each arm minus K2-S1 and minus A-pool; L-W minus L-M; L-Ww minus L-W; on
  rows with no labelled event, at the signed-MW six and the MVA-only eight separately (CRPS in
  points of effective capacity, days 1 to 10 and the 12 to 36 hour band). Bonferroni count 30 CRPS
  tests (60 with q99 pinball loss). The weekly-against-monthly contrast is reported as a 2-seed
  contrast with a fold interval beside the origin-day interval.
- **Also:** real events (NRA-corrected actuals headline, metered secondary; the events of
  experiments 435 and 482, 184 labelled events in the scored months) with the event count and the
  minimum detectable effect, including weeks that contain or follow an event; cold winter nights;
  Christmas week (report-only); q01, q05, q95, q99 coverage and pinball loss; the widening sweep;
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six; D15 and D18 report-only. NGED's
  labels are incomplete, so rows with no labelled event may hold unlabelled events.
- **Serving cost per arm:** networks per serving pass (1), fits per month (about 1 for monthly,
  about 5 for weekly), stored tables (none), and the added work of a weekly refit.

### Pre-registered rules

- **Weekly beats monthly** if L-W minus L-M at the eight or at the six has a Bonferroni interval
  that excludes zero with a point estimate at most -0.03, q99 pinball loss within +0.02, and no
  worse on labelled-event rows beyond the minimum detectable effect. Otherwise stop; keep monthly.
- **A recency weight helps** if L-Ww beats L-W by at least the minimum detectable effect and the
  stress arm loses no more than 0.05 points against L-W given the same injected rows. Otherwise the
  90-day weight is not adopted.
- **A lag-free candidate exists** if the best arm is within 0.03 of K2-S1 on rows with no labelled
  event at both groups ("not shown worse", not equivalent) and not worse than K2-S1 on
  labelled-event rows and in the load-shift stress tests. If so, say it has 1 network per serving
  pass and no load-shift exposure at forecast time, and leave the first-release choice to the
  maintainer. If no arm gets within 0.1 of K2-S1, say the lags carry information a refit cannot
  replace at these leads, and close the line.
- **L-Ww-D or L-Ww-O is worth adopting over L-Ww** only if the injected-shift leak falls by at least
  the minimum detectable effect, rows with no labelled event stay within 0.03 of L-Ww, cold winter
  nights are no worse than +0.05 against L-Ww, and q99 pinball loss stays within +0.02. If L-Ww-O's
  learned outlier weight on cold winter nights is more than twice its weight on other rows with no
  labelled event (leave-one-fold-out, with a count), the component is absorbing weather and the arm
  fails. If both arms fail to cut the leak by the minimum detectable effect, close them and name the
  hindsight target repair as untested.

### Caveats named in advance

- Weights and standardisers of the pretraining are the month's, not refreshed per week.
- The 2025-12 fold has no stored L0PT weights; the 2-seed pretraining here is new.
- Lag-free rows need no lag-age cut, so the rows differ slightly from experiment 425's arm A rows
  (more rows survive); the page prints the row counts of both.
- Intervals ignore dependence across days of one fold.

### Series

Neural simplification series; runs alongside experiment 507 (writes only to its own folder).
