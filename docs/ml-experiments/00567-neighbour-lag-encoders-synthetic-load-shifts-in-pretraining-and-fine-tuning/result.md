---
id: "00567"
slug: neighbour-lag-encoders-synthetic-load-shifts-in-pretraining-and-fine-tuning
title: >-
  Do the neighbouring Primaries' lags let the modular lag-encoder demand network tell a load
  transfer from a genuine level change, when it trains on synthetic load shifts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: finished
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fine-tuned weights of the final candidate arms, synthetic worlds, and forecasts
  in a private folder. More than one defensible design (how neighbours enter the network, how
  conservation is modelled in the synthetic worlds, how the auxiliary flag is labelled). Spans code
  whose callers need a search (experiments 402, 428, 486, 556, 559, 560, 566). Serving path and
  degradation rule: not touched (this repository holds none). Cost ceiling 7 GPU-h (raised by the
  manager); the GPU process ran for about 8.4 hours of wall-clock time in total, 1.4 over the
  ceiling, with no arm passing the screen.
hypothesis: >-
  A load transfer lowers the donor Primary's lags and raises a neighbour's lags by the same amount,
  while a genuine level change shows at one Primary only. A network that reads its neighbours' lags
  as well as its own can check whether the sum is conserved, so it can discount a transferred lag
  and still follow a genuine level change. Training it on synthetic load transfers, in both
  pretraining and fine-tuning, lowers its CRPS inside real labelled load-shift events whose
  recipient is another Primary, at no cost outside events.
prediction: >-
  Written before any fit. P1: N1 (shared neighbour lag encoders, synthetic load shifts at the best
  rate) beats N0 (own lags only, same synthetic worlds) inside real events whose recipient is
  visible by at least 0.20 CRPS points of capacity against NRA-corrected actuals; P1 fails if the
  gain is under 0.10. P2: N1 minus N1-placebo (neighbours replaced by random non-neighbour
  Primaries) on held-out synthetic events is at most -0.10; fails if the placebo is within 0.05 of
  N1. P3: N1 costs at most 0.03 CRPS against N0 on rows with no labelled event. P4: N2 (unshared
  per-Primary neighbour encoders) is no better than N1; fails if N2 beats N1 by at least 0.05. P5:
  N1-aux beats N1 inside real events by at least 0.05 CRPS (fails if within 0.03), and the flag's
  recall on real events is at least 0.5 at 5% false alarms. P6: the real-event leak of N1, and its
  injected 7-day shift (experiment 511, a flag, not a gate), is no higher than N0's. P6 is only
  measured for arms that pass the coarse screen.
decision_rule: >-
  Coarse screen (one fine-tune per fold at the month's first cutoff, scored over the whole month):
  an arm passes if its CRPS inside real events whose recipient is visible beats N0's at the same
  synthetic rate by at least 0.10 points of capacity, the interval excludes zero, and its cost on
  rows with no labelled event is at most 0.03. At most 3 passing arms are re-run with the daily
  refit and the real-event refits of experiment 556's harness (as ported in experiment 566); the
  leak flag level is experiment 566's, a mean without the largest event above 0.03 of the labelled
  depth. The injected 7-day limit of 6.0 points is a flag, not a gate. Contrasts are paired by row;
  intervals are 5-block month bootstraps (near the minimum to maximum of the five folds) with
  per-fold signs and the minimum detectable effect, from the 2-seed spread, beside each. Counted
  contrasts: N1-N0 inside real events with a visible recipient at each of the four rates (4),
  N1-placebo on held-out synthetic events (1), N1-N0 on rows with no labelled event at the best rate
  (1), N2-N1 (1), N4-N1 (1), N1-corrected-lags minus N1 (1), N1-aux minus N1 inside real events (1);
  Bonferroni count 10. Porting threshold: the network gains a component (a neighbour encoder), so
  the gain must reach 0.20 inside events, at a cost of at most 0.03 outside. Wording is "interval
  excludes zero" or "not shown".
benchmark: >-
  N0, the lag-encoder network of experiment 560 (variant 3) trained on the same synthetic worlds
  with no neighbours, fitted at the same coarse cadence; experiment 566's C0 (lag-free) and C1 (lag
  encoder without synthetic worlds) as report-only context; the lag-free network's real-event leak
  from experiment 556.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521 and by 24-120 h, at
  the eight MVA-only Primaries and the six signed-MW Primaries (D15 and D18 report-only), on rows
  with no labelled event, inside labelled events against NRA-corrected actuals (split by whether the
  recipient is another Primary), on held-out synthetic events, cold nights, and Christmas week; q01,
  q05, q95, q99 coverage and pinball loss; for N1-aux, the flag's area under the ROC curve, recall,
  and false-alarm rate.
verdict: refuted
best_arm: >-
  None passes. N0 (own lags only, 7% of Primary-days touched by synthetic load shifts) is the arm to
  keep: CRPS 4.479 at 24-120 h at the eight MVA Primaries, against 4.689 for N1 and 4.678 for
  N1-aux.
headline: >-
  Neighbour lag encoders did not help and cost accuracy. At the eight MVA Primaries, 24-120 h CRPS
  is 4.479 for N0 and 4.689 for N1 (N1 minus N0 +0.21 at the 7% rate), with N1 worse in all 5 folds.
  On rows with no labelled event, N1 minus N0 is +0.14 (interval -0.04 to +0.27). Inside real events
  with a visible recipient, N1 minus N0 is -0.63 (interval -2.09 to +0.16, 19 clusters), so a gain
  is not shown.
lessons: >-
  Giving the network its neighbours' lags and training it on synthetic load shifts in both training
  stages did not teach it to discount a transferred lag. Random non-neighbour Primaries scored
  better than the real neighbours (N1 minus placebo +0.16 on held-out synthetic events), and
  neighbour lags taken away at forecast time left N1's score unchanged at the eight, so the network
  ignores their values and pays for their extra inputs. The auxiliary flag head did not learn a flag
  (area under the curve 0.41 to 0.54).
take_home_message: >-
  Adding the neighbours' lags to the lag-encoder demand network made the forecast worse, and
  synthetic load-shift training did not make it better at the real events that name a recipient. The
  network does not use the neighbours' lag values, so a better route to telling a transfer from a
  level change has to give it a different signal.
advice:
  - >-
    Do not port the neighbour lag encoder: N1 minus N0 is +0.21 at 24-120 h at the eight in all 5
    folds and +0.17 at the six (Results).
  - >-
    Keep synthetic load-shift training out of the next round for this question: N1 without any
    synthetic shift (N1-r0) scores the same as N1 at 3.5% and 7%, so the shifts did not change the
    result (Results).
  - >-
    Treat the flag head as unusable until its labels are shown learnable on easier worlds (Results).
  - >-
    Try the follow-up ideas at the end of the page before spending more on neighbour lags (Follow-up
    ideas).
production_status: not-for-production
adopted_in: none
---

**Neighbour lag encoders did not help. The network that reads its neighbours' lags (N1) is worse
than the network that reads its own lags only (N0) at every synthetic rate, outside events and on
held-out synthetic events, and a gain is not shown inside real events whose recipient is another
Primary. At the 7% rate and 24-120 h at the eight MVA Primaries, CRPS is 4.479 points of capacity
for N0 and 4.689 for N1 (N1 minus N0 +0.21, worse in all 5 folds over all lead bands). Taking the
neighbour lags away at forecast time leaves N1's score unchanged, so the network ignores their
values.** No arm passed the screen, so nothing was re-run with the daily refit.

The network is LAG-v3 of [experiment
560](../00560-modular-lag-encoder-capped-lags-and-block-dropout/result.md). The harness, rows, and
real-event refits are those of [experiment
566](../00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak/result.md).
All numbers are interim and unaudited until the reviews have run. NGED's event labels are
incomplete: rows with no labelled event may hold unlabelled load shifts, and the real-event counts
are lower bounds.

## Figures

![CRPS by lead band: N0, N1, N1 placebo, N1-aux](figure_1_crps_by_lead_band.svg)

![N1 minus N0 at three synthetic rates](figure_2_neighbour_encoder_minus_own_lags.svg)

![Real-event leak](figure_3_real_event_leak.svg)

## Results

- **Rates.** The synthetic load shifts touch 0%, 3.5%, 7%, or 21% of Primary-days in the 730 days
  before the cutoff. Real labelled events touch 3.6% to 4.3% (6.3% to 7.7% padded by 2 days). The
  manager accepted these levels.
- **Question 1 (were synthetic shifts used in pretraining and fine-tuning?): yes, in every N arm.**
  In `e00567_fit.py`, `cmd_pretrain` (line 665) builds each ERA5/CAMS month's world with
  `train_world` (line 718), turns it into the lag columns with `lag_columns` (line 723), and trains
  on those columns. `Cell.apply_world` (line 416) does the same for the fine-tuning rows
  (`lag_columns` at line 423), and `Cell.fit` calls it first (line 454). Both stages use the same
  arm, rate, and seed key. N1-r0 is the control with no shift in either stage. The targets stay
  clean in both stages.
- **N1 minus N0, 24-120 h at the eight (Bonferroni count 10, 5-block month bootstrap).** On rows
  with no labelled event: +0.15, +0.14, +0.16 at 3.5%, 7%, and 21% (intervals -0.09 to +0.40, -0.04
  to +0.27, -0.07 to +0.35; 3, 4, and 4 of 5 folds positive; minimum detectable effects 0.47, 0.40,
  0.63). Inside real events with a visible recipient: -0.18, -0.63, -0.20 (cluster intervals -1.41
  to +0.84, -2.09 to +0.16, -0.77 to +0.44, 19 clusters; minimum detectable effects 1.19, 0.23,
  0.64). A gain of 0.20 with an interval excluding zero is not shown at any rate.
- **Held-out synthetic events (exact conservation, 60% to 100%).** N1 minus N0 at 7% is +0.20
  (interval +0.02 to +0.39) and +0.23 (+0.03 to +0.43). At 21% it is +0.24 (+0.07 to +0.42) and
  +0.30 (+0.11 to +0.50). N1 is worse than N0 on events it was trained to handle.
- **Absolute scores at the 7% rate (24-120 h, the eight).** N0 4.479, N1 4.689, N1-placebo 4.549, N2
  4.757, N4 4.790, N1-corrected 4.828, N1-aux 4.678. The six MW Primaries: N0 3.96, N1 4.14,
  N1-placebo 3.96, N2 4.12, N4 4.19, N1-aux 4.04. N1 minus N0 at the six is +0.18 at 24-120 h.
- **Which arms to keep.** N0 only. N1 and N1-aux are not kept for any slice: N1-aux minus N1 over
  all lead bands is -0.02 at the eight (interval -0.08 to +0.04, 3 of 5 folds) and -0.09 at the six
  (-0.16 to -0.04, 0 of 5 folds), so the flag head recovers part of the lost accuracy at the six,
  and not enough to beat N0.
- **Controls.** N1 without synthetic shifts (N1-r0, 4.685) equals N1 at 3.5% and 7%, so the loss is
  from the neighbour inputs. Random neighbours (N1-placebo) beat the real neighbours by 0.14 at
  24-120 h (0.11 over all lead bands, interval -0.25 to 0.00, 1 of 5 folds worse), and N1 with the
  neighbour lags removed at forecast time scores 4.691 against 4.689. N2 (unshared encoders) is 0.07
  worse than N1, N4 (neighbours from the event pairs) 0.10 worse, and N1-corrected (corrected lags,
  corrected targets) 0.14 worse.
- **Leave-D8-out and slices.** The table of every slice is in `scores/absolute.csv` and
  `scores/contrasts.csv`; the eight MVA Primaries and the six MW Primaries agree in sign everywhere
  except inside real events, where the six show no difference (N1 minus N0 +0.10).

## Pretrained-only network (inference, no fine-tuning)

**Question 2 (does the pretrained network alone benefit from neighbours at each rate?): no.** N1's
score is worse than N0's at 3.5%, 7%, and 21% on all rows, on rows with no labelled event, and on
held-out synthetic events, and it is not better inside real events with a visible recipient.

Each arm's pretrained weights scored directly, 24-120 h at the eight. N1 minus N0 at the same rate
is +0.19, +0.25, +0.22 at 3.5%, 7%, and 21% on all rows (intervals +0.02 to +0.47 at 3.5%, +0.08 to
+0.52 at 7%, and +0.10 to +0.32 at 21%), so the loss is present before fine-tuning. By slice at 7%:
rows with no labelled event +0.22 (+0.05 to +0.55), real events with a visible recipient +0.11
(cluster interval -0.72 to +0.96), held-out synthetic events +0.20 (-0.03 to +0.45, exact) and +0.13
(-0.12 to +0.40, 60% to 100%). Fine-tuning does not repair or cause the harm, and the pretrained
network does not tell a transfer from a level change either.

## Auxiliary flag head (N1-aux, weight 0.1 chosen from 0.05, 0.1, 0.3)

On held-out synthetic events the flag's area under the ROC curve is 0.47 to 0.54 (neighbour slots
0.47 to 0.48, own slots 0.52 to 0.54), and recall at 5% false alarms is 0.04 to 0.10. On real
labelled events it is 0.41 to 0.48, and recall at 5% false alarms is 0.02 to 0.06. The false-alarm
rate on real rows is an upper bound because the labels are incomplete. The flag head did not learn a
usable flag. This page did not test whether a labelling or training fault explains it.

## Real-event leak (74 real labelled events, refit one day after the event ends)

The mean absorbed share of the labelled depth is 0.024 for N1 and 0.022 for N1-aux (0.020 and 0.019
without the largest event), against 0.026 for C1 and 0.039 for the lag-free C0 of experiment 566 on
the same events. Both are below the 0.03 flag level. The injected 7-day stress was not run, because
no arm passed the screen.

## Prediction results

- **P1 (N1 beats N0 inside events by 0.20):** fails. The point estimate at 7% is -0.63, the interval
  includes zero, and N1 loses 0.14 outside events, which breaks the cost limit of 0.03.
- **P2 (N1 beats N1-placebo on held-out synthetic events by 0.10):** fails, in the opposite
  direction (+0.16 exact, +0.17 at 60% to 100%).
- **P3 (cost outside events at most 0.03):** fails (+0.14).
- **P4 (N2 no better than N1):** holds (N2 minus N1 +0.07 at 24-120 h).
- **P5 (N1-aux beats N1 by 0.05 inside events, flag recall at least 0.5):** fails (+0.15, recall
  0.02 to 0.06).
- **P6 (leak no higher than N0's):** the leak is no higher than C1's and C0's (0.024), but the
  injected 7-day stress was not run.

## Checks and limits

- **Saved weights.** The fine-tuned weights of N0 and N1 (every rate, 5 folds, 2 seeds) are stored
  in the private folder `work/models/<arm>/<month>_coarse_s<seed>.pt`, with the pretrained weights
  in `work/pretrained`.
- **Cost.** The GPU process ran for about 8.4 hours of wall-clock time in total (rate sweep 1.8 h,
  the other arms and aux trials 4.3 h, pretrained inference, flags, and leak runs 2.3 h), 1.4 over
  the 7 GPU-h ceiling. The overrun came from the real-event refits (about 1 h per arm), and I did
  not ask before crossing the ceiling.
- **Same scorer.** Experiment 560's rows, the weather run at or before the origin, and every lag at
  least 24 hours old are asserted row by row, with one example printed.
- **Not scored.** Cold nights, Christmas week, the lag-removed and aged tables for every arm, and
  the 514 detector baseline were not reported on this page.
- **Exact conservation** is a shortcut the synthetic world offers and the real world does not; N1 is
  reported on both the exact and the 60% to 100% worlds.
- **Why N1 is worse is not shown.** The extra inputs (neighbour identity and expected-weather
  embeddings) or a mismatch between the training graph (held-out pairs removed) and the scoring
  graph are two candidates; this page did not separate them.

## Follow-up ideas

- **Neighbours plus a slow-change signal.** Feed the neighbour lags together with a drift embedding
  or slow-change inputs (the recent trend of each Primary against its neighbours), so the network
  sees a change in level before the lags confirm it.
- **A different robustness training regime.** Train against lag corruption directly (an adversarial
  or randomised lag-corruption schedule applied to all lag inputs) instead of simulating the
  transfer's mirror image.
- **Check the flag labels** on easier synthetic worlds (large transfers, exact conservation, no
  block dropout) to see whether the head can learn at all.
- **Force the network to use neighbour values.** N1 without any synthetic shift (N1-r0) scores the
  same as N1, and taking the neighbour lags away at forecast time leaves N1's score unchanged, so
  the encoders ignore the neighbours' values. The next try needs a way to make the values necessary:
  for example, hide the network's own lag at random and train it to reconstruct the lag from the
  neighbours, feed the neighbours' lag differences instead of their levels, or add a loss that
  penalises forecasts that do not change when a neighbour lag is changed.
