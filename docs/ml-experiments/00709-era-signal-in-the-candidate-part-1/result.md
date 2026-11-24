---
id: "00709"
slug: era-signal-in-the-candidate-part-1
title: >-
  Does an era signal that tells the Candidate network when a Primary's behaviour changed for good
  improve the forecast in the weeks after a permanent change, without carrying load shifts forward?
theme: retraining-and-training-history
also_relevant_to:
  - lsc-detectors
  - switching-labels-in-the-forecast
  - lagged-power-and-error-correction
status: finished
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts, network weights for each arm, fold, and seed, and
  detector tables; (2) more than one design defensibly tests the hypothesis (the era signal's entry
  point, the detector, and the era-or-switch classifier); (3) it spans the Candidate's training and
  injection code from experiments 559, 572, 574, 591, 613, 660, 667, 687, and 696; (4) it touches
  no serving path or degradation rule; (5) it changes no contract. Full set of reviews: plan, code,
  results, write-up, prose.
hypothesis: >-
  A discrete era signal, computed only from data before the forecast origin and fed only for shifts
  classed as eras, lowers the error of the normal-running-arrangement (NRA) forecast in the weeks
  after a permanent change at a Primary, without carrying load shifts into the NRA forecast and
  without raising the error where no shift happened.
prediction: >-
  Written before any fit. C1a (oracle O minus A0-aug, NRA output, injected era events, origins 0 to
  28 days after the switch) is 5 to 15 points of capacity better for steps of 25% and 1 to 5 points
  for steps of 10%. C1b (O's NRA output minus A0-prev, the prevailing output) is 0.2 to 1.5 points
  better. The best learned arm keeps 30% to 60% of C1b. C7 (carrying a load shift forward as an era)
  is within +-0.1 for the rule and soft classifiers and 1 point or more worse for the arm with no
  classifier. C4 and C5 are within +-0.03, and C6 is under 0.3 points.
decision_rule: >-
  Supported: C1b and C2 pass their thresholds (0.5 points for C1b; a kept share of at least 0.5 of
  the ceiling, with L-best's NRA output better than A0-prev by more than 0.25 points, for C2), with
  intervals that exclude zero, and every veto holds. Partly supported: C1b passes and C2 does not.
  Not supported: C1b's interval does not exclude zero in the gain direction. A failed C7 or
  carried-depth veto gives "not supported for adoption" whatever C2 shows. Vetoes: C4 above 0.03,
  C5 outside +-0.05, C7 above 0.1, carried depth and copied share above A0's by more than 0.02. The
  machinery check stops the run if C1a is under 1 point for steps of 25%. Intervals use month-block
  bootstrap at the Bonferroni level for nine contrasts.
benchmark: >-
  A0 (the Candidate refitted in the same batch), A0-aug, A0-prev (A0's prevailing output),
  experiments 667, 668, and 677 as context, climatology, and the manual heuristic.
row_set_id: experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06
metric: >-
  CRPS in points of capacity, both outputs, sliced by distance from the switch, by the state of the
  shift at the origin (undetected, unknown, classified correctly, classified wrongly), and on stable
  rows; normalised MAE of the median; q05-q95 coverage; q01, q05, q95, and q99 pinball loss; the
  four load-shift scenarios; cold winter nights; avoidable-curtailment cost; detection and
  classification delay.
verdict: inconclusive
best_arm: >-
  No arm is best on every reading. On the real rows `none` (the Candidate without an era signal)
  scores 3.967 points of capacity on all rows, and no learned arm differs from it (best `soft`,
  difference -0.003, 99.44% interval -0.046 to +0.047). On injected eras the oracle, which reads the
  true dates, lowers the error by 7.4 points of capacity (-8.7 to -6.1), and `all` by 2.9, `soft` by
  2.0, `stat` by 0.6, and `hmm` by 0.3.
headline: >-
  An era signal lowers the error after an injected era: the oracle by 7.4 points of capacity, and the
  learned arms `all`, `soft`, `stat`, and `hmm` by 2.9, 2.0, 0.6, and 0.3 (7 to 28 days after the
  onset, 128 events). The arms that help most also raise the error after an injected switching
  event, at the Primary that loses the demand (`all` by 3.9 points, 99.44% interval +2.4 to +5.5).
  On the real rows of four test months no learned arm lowers the error on all rows, and at D8 only
  `hmm` lowers it with an interval that excludes zero (-0.064).
lessons: >-
  Phase B's curriculum never survived early stopping, so `aug` equalled `none`. Phase B2 mixed the
  injected world into every training batch, and its one arm that met the pre-registered rule at 95%
  (`b2-aug`) fails at 99.44%, while the placebo moved the same way. The contrasts against the
  prevailing output (C1b and C2) were not scored because the prevailing output was not stored. In
  the one case we examined (`all` at D8 in 2026-01), a detector carried a temporary excursion
  forward as an era.
take_home_message: >-
  Telling the demand neural network when a substation's behaviour changed for good helps a lot after
  an injected change, but the versions that help most also make the forecast worse after a temporary
  load move, because they treat the move as permanent. On real months the network with an era signal
  forecast no better than the network without one. The experiment therefore cannot say whether an
  era signal is worth adopting until a version that separates permanent changes from temporary
  moves is tested against the prevailing forecast.
advice:
  - >-
    Do not carry a learned era signal forward until it passes the veto on switching events:
    every learned arm that lowered the error after eras raised it at the donor Primary of an
    injected switching event, `all` by 3.9 points, `soft` by 0.34, `hmm` by 0.24, and `stat` by
    0.11 (bank, switching events).
  - >-
    Score any new era signal on injected eras and on injected switching events together, because
    the arms that lowered the error most after eras (`all`, `soft`) raised it most after switching
    events (bank table).
  - >-
    Store the prevailing output of every arm in the next era experiment, so that the contrast
    against the prevailing forecast (C1b) can be scored (verdict paragraph).
  - >-
    Judge a curriculum arm against a placebo arm that receives the same training rows: in phase B2
    the placebo lowered the all-rows error by 0.037 points, about three-quarters of the 0.050 of
    the arm trained on injected eras with no signal (section 27).
production_status: not-for-production
adopted_in: none
---
**We asked whether telling a demand-forecasting neural network when a substation's behaviour changed
for good improves its forecast.** The answer is yes after an injected change, and not shown on the
real months we scored. The network is the Candidate, the modular network of [the final models
page](../final-models.md) that forecasts demand at the 16 Primary substations ("Primaries") of one
electricity network operator for up to 14 days ahead. An era is a sudden change that lasts, such as
a large generator or solar farm connecting to a Primary. A switching event is a temporary move of
demand from one Primary to a neighbour, which the forecast must not carry forward. We gave the
Candidate an era signal built only from data before the forecast origin (the time at which the
forecast is issued), in several forms: fixed rules (`rule`, `soft`), a hidden Markov model (`hmm`),
a statistical detector (`stat`), a version that turns every detected shift into an era (`all`), and
an oracle that reads the true dates and is an upper bound, not a deployable arm. We compared each
with the same network without a signal (`none`). We scored four test months (2025-11, 2026-01,
2026-04, and 2026-06) with two training seeds, on the real series and on real series into which we
injected synthetic eras and switching events. All numbers are interim: the plan pre-registered five
seeds, and we ran two because experiment 667, which supplies the first two of the Candidate's three
training stages, stores them for seeds 0 and 1 only.

**Verdict: inconclusive.** The decision rule rests on two contrasts, C1b and C2, which compare each
arm with that arm's prevailing output (the forecast the network makes without the era signal). We
did not score C1b or C2, because we did not store the prevailing output of any arm. The scored
contrasts point two ways. After injected eras, five of the seven era signals lower the continuous
ranked probability score (CRPS, lower is better, in points of the Primary's effective capacity) with
a 99.44% interval that excludes zero in at least one window: `oracle`, `all`, `soft`, `stat`, and
`hmm`. After injected switching events, every learned arm that helps after eras raises the CRPS at
the donor Primary (the Primary that loses the demand), which fails the pre-registered veto on
switching events. On the real rows, no learned arm lowers the CRPS
on all rows. An audit was required (see Checks), and it reproduced the bank table.

**On the real rows, no learned arm lowers the error on all rows, and at D8 only `hmm` lowers it with
an interval that excludes zero.** Differences are from `none`, with 99.44% intervals (the
pre-registered level for nine contrasts) that resample whole fold months, so four fold months and
two seeds support each interval. D8 is the Primary where a solar farm connected on 2025-03-24.

| Arm | All rows | D8 | Other Primaries | D8, 14 days after a stat alarm |
|---|---|---|---|---|
| `none` (CRPS) | 3.967 | 7.686 | 3.719 | 7.252 |
| `oracle` | +0.003 [-0.018, +0.044] | -0.047 [-0.107, +0.015] | +0.007 | -0.088 [-0.193, +0.005] |
| `soft` | -0.003 [-0.046, +0.047] | -0.060 [-0.223, +0.053] | +0.001 | -0.025 |
| `hmm` | +0.071 [-0.025, +0.205] | -0.064 [-0.141, -0.010] | +0.080 | -0.107 [-0.363, +0.028] |
| `rule` | +0.012 [-0.029, +0.071] | +0.053 | +0.009 | -0.029 |
| `stat` | +0.030 [-0.019, +0.124] | +0.325 [+0.056, +0.695] | +0.010 | +0.077 |
| `all` | +0.102 [-0.019, +0.289] | +1.366 [-0.095, +4.970] | +0.018 | -0.027 |
| `placebo` | +0.019 [-0.069, +0.116] | -0.004 | +0.021 | -0.010 |

**After injected eras, five arms lower the error, and the arms that help most do the most harm after
injected switching events.** The injected bank holds 16 passes per fold month, where a pass is one
injection run over the panel with two eras and two switching events. Four fold months therefore give
128 events of each kind, the same events for both seeds and for every arm, and the CRPS of an event
is the mean over its rows and over the two seeds. An era is scored against the injected power for
forecast origins 0 to 7 and 7 to 28 days after its onset. A switching event is scored at its donor
Primary against the power without the switch, from its onset to 7 days after its end or the end of
the fold month, whichever comes first (only 48 of the 128 events end at least 7 days before the
month ends). The recipient Primary is not in the table. Differences
are from `none`, with 99.44% percentile intervals that resample whole passes (a fold month and a
pass index), because the events of one pass share a panel.

| Arm | Era, 0 to 7 days | Era, 7 to 28 days | Switching event, donor, onset to end + 7 days or month end |
|---|---|---|---|
| `none` (CRPS) | 10.971 | 10.751 | 3.355 |
| `oracle` | -7.115 [-8.375, -5.881] | -7.386 [-8.714, -6.090] | -0.083 [-0.132, -0.038] |
| `all` | -1.527 [-2.384, -0.743] | -2.858 [-4.253, -1.550] | +3.901 [+2.430, +5.538] |
| `soft` | -0.839 [-1.427, -0.323] | -2.047 [-3.119, -1.026] | +0.337 [+0.081, +0.695] |
| `stat` | -0.060 [-0.136, +0.021] | -0.639 [-1.165, -0.221] | +0.112 [+0.018, +0.261] |
| `hmm` | +0.001 [-0.147, +0.141] | -0.343 [-0.531, -0.161] | +0.242 [+0.137, +0.379] |
| `rule` | -0.015 [-0.091, +0.077] | -0.053 [-0.151, +0.061] | +0.022 [-0.061, +0.182] |
| `placebo` | -0.059 [-0.114, +0.000] | -0.044 [-0.123, +0.044] | -0.016 [-0.039, +0.005] |

**The oracle lowers the error after eras by about 7 points of capacity, and `all` lowers it by 2.9
points at the price of raising it by 3.9 points at the donor Primary of a switching event.** The
oracle's figure pools injected steps of different sizes, so it is not the pre-registered C1a. The
arm `all` turns every detected shift into an era, so it carries a switching event forward at the
donor. An audit that scored the recipient Primary instead (the Primary that gains the demand, 96
events with a metered recipient) found that `all` lowers the recipient's CRPS by 0.03 points, so the
harm is confined to the donor. The placebo is within 0.06 points of `none` everywhere. The
pre-registered veto on switching events (C7 above 0.1, or its interval end above 0.2) fails for
`all`, `soft`, `hmm`, and `stat`, and only `rule` passes it, so no arm that gains after eras can be
called supported under the screen's rule. The two decision rules in this page differ on a failed C7:
section 11 gives "partly supported" when C2 passes, and section 20 gives "not supported for
adoption" whatever C2 shows. Section 20 refines the earlier sections, so its rule governs.

**In phase B2, the arm trained on injected eras with no signal (`b2-aug`) meets the pre-registered
rule at 95% and fails it at 99.44%.** Phase B2 mixes the curriculum (the injected world) into every
training batch, and its four arms run on the four test months with two seeds. At 95%, `b2-aug`
lowers the all-rows CRPS by 0.050 (interval -0.103 to -0.001) and changes the touched Primaries (the
Primaries that received an injected era) by +0.008. At 99.44% the all-rows interval is -0.109 to
+0.005. The placebo moves the same way (-0.037, 95% interval -0.127 to +0.096), and the injected
rows receive a second noise draw, which could explain both moves without any era effect (we did not
test this). The curriculum world held 2 eras in 6 of the 8 fold-and-seed pairs and 4 in the other 2,
so 6 of the 8 pairs are weak tests of the curriculum, and the contrast `aug` against `none` is not
decisive (section 27).

**At D8, `e-stat` lowers the error in phase E although `stat` raised it in the screen months, so the
sign of its effect at one Primary is not stable.** Phase E is a robustness check at one Primary: it
has about 6 forecast origins per month and wide intervals, and its oracle reads the scored months.
It scores the folds 2025-04 to 2025-06 with seeds 0 and 1, and differences are from `e-none` (CRPS
8.718 at D8 and 3.717 on all rows), with 95% month-block intervals. At D8, `e-stat` is -0.569
[-1.339, -0.242], `e-rule` -0.170 [-0.425, +0.049], `e-placebo` -0.110 [-0.295, +0.049], and
`e-oracle` +0.374 [-0.399, +1.420]. On all rows only `e-rule` has an interval that excludes zero
(+0.021 [+0.000, +0.042]). A series-origin may hold fewer than 672 forecast rows (14 days of
half-hours), because the stored rows keep only half-hours with an observed target. These numbers are
never pooled with the four screen months.

**The `all` arm's D8 outlier comes from one fold month and is consistent with a temporary
excursion.** D8's daily mean power was about 7.5 MW from 2026-01-19 to 2026-01-25 against about 4 MW
before and after, with no mirror shift in its three neighbours. The `all` arm turned the excursion
into an era. Leaving out fold 2026-01 changes the `all` contrast at D8 from +1.366 to -0.018.
Section 26 has the details and the maintainer's open questions.

## Checks

- **Median minus mean forecast per site, and the error of the median forecast beside every CRPS
  contrast:** not computed. The scorers print CRPS only, so the normalised mean absolute error of
  the median that the evaluation protocol requires is missing, and trigger 3 cannot be cleared.
- **Triggers that fired:** trigger 1 (a known-answer test) required an audit of the bank table,
  because the bank is the known-answer test of the signal. A Sonnet agent that did not write the
  experiment re-derived the bank contrasts from the raw forecast files with its own code. All
  contrasts matched the table to three decimals, 0 events differed between arms, and no left-out
  site or fold flipped a sign of the era contrasts of `soft` and `all`, the switching contrast of
  `all` and `hmm`, or the D8 contrasts of `hmm` and `stat`. Trigger 2 applies to phase E, where
  `stat` changes sign against the screen months and `e-oracle` raises the D8 error. Trigger 4 found
  two fragile contrasts: the `all` contrast at D8 depends on fold 2026-01 alone, and `hmm` on eras 0
  to 7 days changes sign when any of 6 sites or fold 2025-11 or 2026-04 is left out. The `soft` D8
  contrast is driven by fold 2026-01 (-0.199), and the `hmm` all-rows contrast by fold 2025-11
  (+0.216).
- **Scored Primary of a switching event:** the table scores the donor. At the recipient, `all`
  changes the CRPS by -0.029, so the sign of its harm depends on which Primary is scored.
- **Controls:** the placebo is within 0.06 points of `none` on injected eras and switching events,
  as a control that carries no information should be.
- **`aug` against `none`:** the two agree to 3 decimals, because stage 3 always keeps its first
  epoch in phase B (section 23). Every phase B curriculum contrast is void.
- **Intervals:** the real-row intervals resample 4 fold months, so they are wide, and an interval at
  99.44% is wider than the 95% interval that the B2 rule and phase E use.
- **Not scored:** C1b, C2, C4 to C7 except the switching-event reading above, the state slices
  (undetected, unknown, classified), the bank's classification scores, cold winter nights,
  avoidable-curtailment cost, and the four load-shift scenarios.

## Method

- **Architecture (all arms):** the Candidate network of experiments 667 and 677, with frozen stage 1
  and stage 2 weights from experiment 667 for seeds 0 and 1, and a stage 3 that sees the era signal
  as extra input columns.
- **Era signals:** the oracle reads the true era dates and the injected switching events, `rule` and
  `soft` use fixed thresholds on shifts, `hmm` uses a hidden Markov model fitted before 2025-01-01,
  `stat` uses the statistical detector of phase A, and `all` turns every detected shift into an era.
  The placebo moves the oracle's change vector to other dates. Every signal at an origin uses data
  up to the day before the origin.
- **Training regime:** stage 3 is fitted per fold, seed, and arm on the pre-fold history, with
  patience 8 and the best state restored. The curriculum injects eras and switching events into the
  training panel. Phase B2 reads the injected world with probability 0.5 per row and epoch.
- **Measurement:** CRPS in points of effective capacity against the cleaned half-hourly power, at
  the contract's 12:00 UTC origin, on experiment 677's monthly-refit rows for 2025-11 to 2026-06 and
  on the phase E row set for 2025-04 to 2025-06. Differences are paired, with a bootstrap that
  resamples whole fold months. The injected bank is scored by a percentile bootstrap over passes,
  and each switching event is scored at its donor Primary only.
  The bank scorer maps an event's host panel index to the Primary's time series identifier through
  the site map. Its bootstrap resamples passes, and the 16 passes of a fold month share one trained
  network per seed, so the interval does not include the noise of fitting the network.
- **Changes in phase B2 and phase E:** phase B2 changes the stage 3 rule to mix the curriculum into
  every batch and runs four arms; phase E runs five arms on three earlier months.

## Plan as registered

The sections below are the plan as registered before the first fit, with the interim amendments of
sections 22 to 27.

This experiment asks whether the Candidate network (the modular network of
[the final models page](../final-models.md)) forecasts better after a permanent change at a Primary
when it is told the Primary's era. The maintainer asked for it by name: "revisiting the idea that
substations experience these eras" (2026-11-19). An era is a discrete, rare, sudden, and persistent
regime, such as a large generator or solar farm connecting. A load shift is a temporary move of
demand
between neighbouring Primaries, and the NRA forecast must exclude it. The two look alike in recent
data, so the plan includes arms that classify each shift.

**Phase E costs compute only.** Phase E (three extra folds, 2025-04 to 2025-06, to score the network
across D8's 2025-03-24 solar connection, and the conditional pre-training arm O-enc-pre) needs no
purchase and no data fetch beyond what the workstation holds. It is excluded from the committed arms
until the maintainer approves it through the manager.

**Earlier advice.** The plan follows the advice that a step-capable offset cannot date a step
(experiments 573, 575, and 577) by using an explicit detector and an explicit state input, and the
advice to test the era signal with lags present (experiment 572) by building every arm on the full
Candidate. It departs from experiment 565's advice to feed a drift embedding to the mixer only,
because the design also tests an entry through the encoders (arm O-enc).

**Terms.** NRA is the normal running arrangement: the electricity network's configuration without
any load shift. The Candidate (class `PermLagMdn`,
`ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbmodel.py`) outputs the
NRA forecast. The prevailing output is a second forecast that keeps the conditions at the forecast
origin: a survival head mixes the Candidate's quantiles with a shifted copy, weighted by the
probability that a shift in progress persists (`docs/ml-experiments/final-models.md`). A switching
event (a load shift) moves demand between electrical neighbours, usually for days to weeks. An era
change is a permanent change in a substation's own net demand, such as a generator connecting. CRPS
is the continuous ranked probability score in points of capacity, lower is better.

## Summary of arms and contrasts

| Arm | Kind | Shift dates and classes come from | What changes against the Candidate |
|---|---|---|---|
| A0 | Control | none | Nothing; refitted in the same batch |
| A0-aug | Control | none | Stage 3 training rows also hold injected persistent steps, with no era input |
| O | Cheating upper bound | True dates and true classes (era table, verified switching events, the injector) | Era encoder into `mixer_ft`, era anchor, lag translation, era row weights; switches removed from lags and from training rows |
| P | Placebo | O's true era switches and true switch removals, each permuted to a random Primary and a random date in the same window, keeping the count and the class of each | O's recipe with uninformative dates; read O minus P as the value of the true dates |
| O-enc | Where the signal enters | As O | O, plus the era vector into the weather encoder and the date-time encoder through zero-initialised first-layer columns |
| L-all | Learned, no classifier | Online change-point detector; every detected shift is treated as an era | The era package of O, with detected dates |
| L-rule | Learned, rule classifier | The same detector, then a three-state rule (`unknown`, `era`, `switch`) from the neighbour mirror, the weather response, and duration | Era package for class `era`; removal for class `switch`; no action while `unknown` |
| L-soft | Learned, trained classifier | The same detector, then a small neural classifier that outputs a probability of `era` | The era vector and anchor scaled by the probability; removal scaled by one minus the probability |
| L-hmm | Learned, recurrent states | Sticky hidden-Markov model, forward filter; duration and recurrence are the only evidence of class | Learned vector per (Primary, state); a state counts as an era after 28 days of dwell |
| L-stat | Learned, statistics detector | Step changes in 12 distribution statistics of the Primary's own raw power over trailing 28-day windows, each change compared with the same change one year earlier; then L-rule's three-state rule with two shape-evidence terms | The era package of O with detected dates, plus a learned vector per (Primary, era id), with one shared vector for an era id first confirmed after the training cut |

All ten arms (L-stat replaces L-R) change stage 3 only and share stage 1 and stage 2 weights with
A0, seed by seed.
Reference rows, not arms: A0-prev (A0's prevailing output, 16 head refits), the stored Candidate
forecasts of experiments 667 and 668, climatology, and NGED's manual heuristic.

| Contrast | Arms and output | Rows and truth | Reads as |
|---|---|---|---|
| C1a | O minus A0-aug, NRA output | Injected era events, origins 0 to 28 days after the switch, observed power | Does a perfect signal make the NRA output adopt a new era? (machinery check) |
| C1b | O's NRA output minus A0-prev | Same rows | Does the ceiling beat the existing forecast of current conditions? **The deciding ceiling.** |
| C2a | L-best minus O, NRA output | Same rows, including the days when the shift is undetected or `unknown` | How much of the ceiling the learned signal gives up (shares weights with O, so precise). |
| C2 | Share of the ceiling kept: 1 − C2a / (A0-prev − O) | Same rows | **The verdict.** Passes at a share of 0.5 or more with a bootstrap interval on the ratio that excludes zero, and L-best's NRA output minus A0-prev below −0.25 (head noise of A0-prev stated, with the seed pairing below) |
| C3 | O-enc minus O | Same rows, split by step kind | Does the era vector help more inside the encoders than at the mixer? |
| C4 | L-best minus A0, NRA output | Stable rows | Harm where no shift happened (veto) |
| C5 | P minus A0-aug | Stable rows and post-switch rows | Placebo equivalence (veto) |
| C6 | O minus A0-aug | Retrained worlds, origins 28 to 90 days after an era switch | Does the signal still help once a refit has seen the new era? |
| C7 | L-best minus A0, NRA output | Injected switching events, inside the event and 7 days after its end, against the power without the injected shift | Does the era signal carry a switching event forward as an era? (veto) |
| C8 | L-best minus L-all, NRA output | Same rows as C7 | What the classification is worth |

## 1. Hypothesis and prediction

**Hypothesis: a discrete era signal, computed only from data before the forecast origin and fed only
for shifts classed as eras, lowers the error of the NRA forecast in the weeks after a permanent
change at a Primary, without carrying switching events into the NRA forecast and without raising the
error where no shift happened.**

**Prediction, written before any fit.** The Candidate is trained to ignore a shifted lag, because a
load shift is not part of the normal running arrangement: experiment 699 measured that the NRA
output copies 0.075 to 0.127 of a lag-only shift. A new era is part of the normal running
arrangement, so after an era switch the NRA output stays near the old level until refits learn the
new level. C1a is therefore predicted large: 5 to 15 points for a step of 25% of capacity and 1 to 5
points for a step of 10%. C1b is predicted between 0.2 and 1.5 points, because the prevailing output
already follows the conditions at the origin. C2 is predicted to keep 30% to 60% of C1b, limited by
the days a shift spends undetected or `unknown`. C7 is predicted inside ±0.1 for L-rule and L-soft,
and C8 predicted as a gain of 1 point or more, because L-all carries every switching event forward.
C3 is predicted inside its noise for level steps and as a gain only for the solar-shaped kind. C4
and C5 are predicted inside ±0.03, and C6 under 0.3 points.

## 2. Three states: the slow level, the era, and the switching event

**The slow level is the Candidate's per-Primary output offset, and the offset stays as built.** The
offset is piecewise-linear between monthly knots from 2019-09, is added to the three mixture means,
and carries a penalty of 10 times the mean squared first difference of the knots
(`LagOffsetMdn.penalty`, `e00572_model.py`). The offset models gradual change: demand growth, slow
growth of rooftop solar, and efficiency. Experiments 573 and 575 found that an offset cannot date a
step.

**An era is a discrete index per Primary that almost never changes.** Each half-hour inherits the
era of the half-hour before. A switch of era is sudden, large, and assumed permanent. Every learned
arm encodes that prior: a constant hazard of one shift per 730 days in the change-point detector, a
self-transition probability of 1 − 1/180 per day in the hidden-Markov model, and a minimum size
below which a change is never an era.

**A switching event is a temporary shift that belongs to neither state.** The event has an onset, a
depth, and an end, and the NRA forecast must not contain the event.

**What the forecast does in each state of a shift:**

| State at the origin | NRA output | Prevailing output | Slow level and training rows |
|---|---|---|---|
| Undetected | Unchanged | The existing survival head follows the shift from its own lagged inputs | Unchanged |
| `unknown` (alarm fired, class undecided) | Unchanged: the shift is not carried, and the era encoder reads an `unknown` flag and the shift's age | As above | Rows since the estimated onset are held out of stage 3 until the class is known |
| `era` | Era vector on; era anchor added to the means; lags and gap signals from before the onset translated to the new level | Follows through the Candidate's quantiles | Penalty break at the onset; earlier-era rows weighted by 0.1 |
| `switch` | The estimated depth is subtracted from lags and gap signals inside the event; no era vector and no anchor | The existing survival head carries the event | Rows whose valid time lies inside the event are dropped from stage 3; the era index and the offset are untouched |

**A wrong `era` class is reversed.** If the level returns to within tolerance of the earlier era
inside 120 days, the shift is re-classed as a switching event, the era vector and anchor are
withdrawn, and the rows are dropped from the next refit.

**The era anchor re-anchors the slow level without resetting the knots, and the anchor applies only
where no stage-1 knot already carries the new level.** In the Candidate, stage 1 trains every knot,
and in stages 2 and 3 the older knots are a frozen buffer (`knots_old`) while only the newest knot
(`knot_new`) trains (`LagOffsetMdn`, `e00572_model.py`). An anchor added to stage-3 rows whose
stage-1 knots already hold the new level would count the step twice. The rule is therefore: for a
stage-3 row, the anchor of an era is added only when the era's onset lies after the stage-1 cut, or
when the row's valid time lies after the newest knot (where `tau` is clamped to the newest knot).
For an era whose onset lies inside the stage-1 history, as at D8's two era onsets, the old knots
after the onset already carry the level, the anchor is zero, and the era vector alone enters. A
known-answer check injects an era into stage-1 rows, with its anchor, and asserts that the stage-3
output does not shift by twice the step. The first-difference penalty is unchanged: dropping the
pair of the last old knot and the new knot would remove the only constraint on `knot_new`. A
detector restart whose change is under the minimum size leaves the era index unchanged and is left
to the offset.

## 3. What separates an era from a switching event

**Both look like a sudden level shift in the recent data, and four kinds of evidence separate
them.**

- **The neighbours' mirror image.** A switching event moves load between electrical neighbours, so a
  neighbour shows the opposite shift and the sum is conserved. An era changes the substation's own
  net demand, and no neighbour moves. The classifier reuses the causal mirror statistic of
`ml-experiments/00687-causal-detector-features-for-the-prevailing-forecast-heads/e00687_features.py`,
  which reads no input later than 6 hours before the origin. Two limits apply: a neighbour may be
  unmetered, and experiment 456 found that a mirror test passed at shifted dates as often as at true
  dates.
- **Duration and reversal.** The verified events of experiment 580 last 7 to 60 days and revert. The
  probability of `era` therefore rises with every day a shift persists, from a prior built on the
  labelled durations. Duration alone decides slowly, and experiment 101 found that 80.5% of logged
  half-hours lie in closures of more than 60 days.
- **The weather response and the daily shape.** A new solar farm adds a term that follows radiation
  and is zero at night. A transferred load has the daily shape of demand and follows temperature. A
  gas engine follows neither. The classifier regresses the daily change in each hour band on the
  day's radiation and temperature. The weather is the ensemble mean of the freshest run published by
  the origin; ERA5 is used only for days at least 7 days before the origin, because ERA5 arrives
  about 5 days late.
- **Existing causal detectors of switching.** The lag gate G4 of experiment 592 (fires a median of 1
  day after onset, with 0.20 false triggers per Primary-year) and the step statistics of experiment
  687 are inputs. The labels of experiment 41 and the verified set of experiment 580 are built with
  hindsight, so those labels score the classifier and never feed a forecast.

**L-rule sums the evidence as log-odds and holds three states.** The class stays `unknown` until the
summed log-odds pass a threshold in either direction. The weights and thresholds are set once on
injected shifts in 2019-09 to 2024-12 at Primaries other than D8, and are then frozen.

**L-soft trains a small neural classifier on the same evidence plus the outputs of gate G4 and of
the features of experiment 687.** The classifier is a network with one hidden layer of 16 units,
trained on injected eras and injected switching events in the same tuning period, and outputs a
calibrated probability of `era` that is refreshed daily. The forecast uses the probability as a
weight, so L-soft has no `unknown` state. For scoring, L-soft's shift counts as unclassified while
the probability lies between 0.2 and 0.8.

**L-hmm uses duration and recurrence only.** A state that the filter leaves within 28 days was a
switching event, and a state held for 28 days becomes an era. L-hmm tests how far the cheapest
evidence goes, at substations with no metered neighbour.

## 4. The era table and the known cases (built before any fit)

**The first script builds one table of known eras, and the experiment reads no era date from
prose.** The file is `data/ml-experiments/<id>-<slug>/era_table.parquet`, written by a committed
script that reads the Embedded Capacity Register with `studies.ecr_reader.read_sheet` (the pattern
of `ml-experiments/00191-dp-price-sensitive-assets-in-the-history/e00191_ecr.py`, because `read_ecr`
drops `Date Connected`). Columns: `primary`, `era_index`, `register_date`, `telemetry_date`,
`technology`, `size_mw_register`, `size_mw_telemetry`, `size_mw_recollection`, `source`,
`confidence`, and `note`. The table holds no customer name.

**D8's rows use the register's dates where the register gives one.**

| Era | Register row | Note |
|---|---|---|
| 1 | Gas engine, registered under a different Primary | D8's reading stopped including the engine's output on 2023-08-15 (experiment 194). |
| 2 | Solar farm, 8.91 MW, connected 2025-03-24 | Telemetry date to be measured in stage 0 |
| not an era | Solar, 0.7 MW, no connection date | **Unknown.** The row cannot define an era until a date is found. |

**The oracle switches on the telemetry date where the era table has one.** The reading that the
network forecasts changes on 2023-08-15, so this design switches the oracle's era on
`telemetry_date` where the table has one and on `register_date` otherwise, and reports the
register-date version as a sensitivity. The classifier's score counts a detection within 30 days of
either date as a match.

**The maintainer recalls a solar farm of about 700 kW in 2025, and the register's dated 2025 row is
8.91 MW.** The plan resolves the gap in three steps: re-read both solar rows from the workbook and
print them; estimate the midday change on clear days in April to August 2025 against the same months
of 2024 from D8's own reading, which separates sizes 13 times apart; and ask the maintainer through
the manager. Every size the network reads is measured from telemetry, so the experiment does not
wait for the answer.

**Other known eras and known switching events.** Register connections of 1 MW or more since 2019-09
at the other Primaries (for example D5) enter the table as `register only`. The fade at D8
from 2026-03 (experiment 697) enters as `telemetry only, unresolved`. The known switching events are
the labelled set of experiment 41 (`labelled_set.parquet`, loaded by `e00086_labels.py`) and the
verified events of experiment 580 (`events_verified.parquet`).

## 5. The signal, row by row

**Every signal at a forecast row is a function of power readings up to the end of the UTC day before
the origin day.** The Candidate forecasts at 09:00 UTC on origin day D and relabels the origin to
12:00 UTC. The detector and classifier state used at origin D is the state after day D − 1, whose
last reading ends 9 hours before 09:00. A training row with origin o carries the state after day o −
1, never a hindsight label, so one function builds training rows and forecast rows. Arm O is the one
exception and is labelled non-causal.

**Detector input.** For each Primary and UTC day, six numbers: the mean residual in four hour bands
(night, morning, midday, evening), and the day's 5th and 95th percentile of the half-hourly
residual. The residual is power over capacity minus a month-by-half-hour climatology
(`ClimatologyForecaster`), trained on the 730 days that end 28 days before the start of the day's
month. Each feature is scaled by a robust spread from the first 365 days.

**The detector is online Bayesian change-point detection** ([Adams and MacKay
(2007)](https://arxiv.org/abs/0710.3742)): an independent Normal mean and variance per feature
within a segment, a constant hazard of 1/730 per day, and a run length truncated at 730 days. An
alarm fires on the first day when the posterior mass on run lengths under 28 days has exceeded a
threshold on 3 consecutive days and the change in segment mean is at least 3 scaled units in one
feature and at least 5% of capacity. The threshold is tuned to at most 0.5 false alarms per
Primary-year on clean history.

**The columns written for each (Primary, origin day):** the class state (`none`, `unknown`, `era`,
`switch`, or L-soft's probability), the shift's age in days (log-scaled, capped at 365), the
six-number change vector (mean of the features since the estimated onset minus the previous
segment's mean), the anchor, and the estimated depth of a switching event. The change vector and the
anchor update daily for 28 days after an `era` class and are then frozen until the next era switch,
so the era signal is piecewise-constant after a settling window.

**Two delays are printed for every known and injected shift: the detection delay (alarm day minus
onset) and the classification delay (class day minus alarm day).** The printed example has this form
(the values show the layout only and are not measurements):

```text
primary=D8 onset=2026-02-01 truth=era kind=level size=-0.25
day_processed  p(run<28d)  alarm  log_odds_era  state     first_origin_with_this_state
2026-02-04     0.93        no     -             none      -
2026-02-06     0.99        yes    +0.4          unknown   2026-02-07 09:00 UTC
2026-02-15     1.00        yes    +3.1          era       2026-02-16 09:00 UTC
detection_delay_days=6  classification_delay_days=9
origins 02-01..02-06 undetected | 02-07..02-15 unknown | 02-16 onwards era
```

**Arm L-hmm** follows the stickiness idea of [Fox et al.
(2011)](https://doi.org/10.1214/10-AOAS395): three Gaussian states per Primary over the six
features, fitted by expectation-maximisation on days before the fold's training cut, with the
self-transition probability fixed at 1 − 1/180. The signal is the forward-filtered state
probabilities after day D − 1, with no smoothing.

**Arms O and P.** O reads the true onset, the true class, and the true size (for D8, the hindsight
mean over the era). P draws the same number of era switches per Primary at uniform random dates at
least 28 days from any true shift, each carrying a permuted true change vector, with a fixed seed.

**A perturbation test proves the cut before any fit.** The check changes every reading at or after
00:00 of day D, at the Primary and at its neighbours, and asserts that no signal value at origin D
changes, for 200 random (Primary, origin) pairs per arm, in the style of `check_state_cut` in
`e00696_checks.py`.

## 6. Target weather comes from the run issued at or before the origin

**Every stage 2, stage 3, and forecast row reads the ECMWF ensemble (ENS) 00Z run of the origin day,
which the Candidate treats as available at 09:00 UTC.** On every row, `nwp_init_time` is 00:00 UTC
of the origin day, 9 hours before the native origin of 09:00 UTC and 12 hours before the contract
origin of 12:00 UTC. The weather columns before the origin use the ensemble mean of the freshest run
published by the origin. Stage 1 trains on ERA5 and CAMS reanalysis rows, which never reach a
forecast row.

**Worked example.** Contract origin 2026-02-10 12:00 UTC, valid time 2026-02-13 17:30 UTC: the row's
weather is the ENS run of 2026-02-10 00:00 UTC at step 89.5 hours, and the lead from the contract
origin is 77.5 hours. The scoring script asserts `nwp_init_time <= power_fcst_init_time − 3 h` on
every scored row and prints one row per arm, reusing `check_origin_time_weather` from
`e00696_score.py`.

**No reference uses experiment 404's lag-free scores above 24 hours, which are void.** Lag-free
references come from experiment 541 (rebuilt K2-S1: 5.400 at the eight MVA-only Primaries and 3.936
at the six signed-MW Primaries, at 120 to 240 hours). Experiment 531 has no merged result page, so
its numbers are not quoted. Candidate references come from experiments 667, 668, and 677.

## 7. How the signal enters the network, and the frozen encoders

**The mixer-entry arms add columns to `mixer_ft` at stage 3 only, with zero initial weights.** A new
class `EraPermLagMdn(PermLagMdn)` holds an era encoder (9 inputs: `unknown` flag, class weight, age,
six changes; 16 hidden GELU units; 8 outputs) and widens `mixer_ft[0]` by 8 inputs with
`widen_first_layer` from `e00696_model.py`. A zero era vector reproduces `PermLagMdn`, which a check
asserts to 1e-6. The anchor is one more column read by index and added to the means, the pattern of
`OutputOffsetDrift`.

**Within a fold the encoders are trained once, in stage 1, on rows from 2019-09, and are frozen in
stages 2 and 3; the chain repeats stage 1 at each monthly fold (`chain.sh` of experiment 667).** An
era input learnt in stage 1 would need a label on every row back to 2019-09. The design handles the
constraint in two steps.

- **O-enc avoids stage 1.** The era vector enters the weather encoder and the date-time encoder
  through a separate zero-initialised `nn.Linear(8, 64, bias=False)` per encoder, whose output is
  added to that encoder's first-layer pre-activation. **The two layers are registered on the network
  outside `model.weather` and `model.datetime`.** (`widen_first_layer` replaces the first layer's
  weight with one `nn.Parameter`, and the `frozen` filter in `train()` of `e00559_fit.py` leaves
  every parameter of those two modules out of the optimiser, so widened columns would stay zero and
  O-enc would equal O.) A check asserts that the layers' weights move after one epoch of stage 3.
  Every pre-trained encoder weight stays frozen. Labels are needed only on stage 3 rows, and O-enc
  stays paired with A0. The lag-side calls of the two encoders pass a zero era vector, which is a
  stated limit.
- **A full pre-training arm (O-enc-pre) is conditional on C3.** Its historic labels are causal by
  the as-of rule: the forward detector and classifier run once over the whole record, and a stage 1
  row at time t reads the state after the day before t, which also lies before the fold's cut.
  O-enc-pre shares no weights with A0, so its contrast carries the unpaired noise of section 11.

**The curriculum teaches the two classes apart.** The Candidate's recipe already injects lag-only
shifts into stage 3: the lag moves and the target does not, which is a switching event with the NRA
target. The curriculum adds persistent steps to half of the training passes: the target and the lags
both move, with sizes uniform in ±5% to ±30% of capacity and a flat, midday-weighted, or
evening-weighted band profile. The detector and classifier run on each injected series, so the
learned arms train on signals with real delays, real `unknown` periods, and real misclassifications.
A0-aug has the persistent steps and no signal.

## 8. Arms that were dropped, and why

- **A piecewise-constant embedding trained end to end with a penalty on the number of switches.**
  Experiments 573, 575, and 577 found that penalised step offsets date at most 23% to 32% of steps,
  and a knot learnt in training cannot switch after the training cut.
- **A soft latent state from recent residuals.** The Candidate's lag encoder already reads 60 gap
  signals with half-lives of 1, 3, and 7 days, so A0 is that arm.
- **Era row weights as an arm of their own, and training on the current era only.** The weights stay
  inside the era package. Stage 3's 90-day half-life already discounts old rows, and C6 says whether
  the package helps after a refit.
- **Penalised offline segmentation (PELT, [Killick et al.
  (2012)](https://doi.org/10.1080/01621459.2012.737745)).** Run daily on an expanding window, PELT
  is the same kind of arm as the change-point detector. Stage 0 uses a penalised least-squares
  segmentation in hindsight as a cross-check.
- **L-R (reset only).** Stage 0 found that L-rule classed 3 of 64 steps of 25% as `era` by day 28,
  so
  an arm that acts only on L-rule's `era` class would differ from A0 on almost no row. L-stat takes
  its place.
- **The hindsight switching labels as a forecast input.** The labels are not available at the
  origin, so the labels score the classifier only.
- **The weather encoder and the date-time encoder as separate arms, and the gas-engine state.** The
  split is worth a fit only if C3 shows a gain, and experiment 696 tested the engine state.

## 9. Known-answer tests and the injected bank

**Stage 0 scores the detector and the classifiers on CPU, before any fit.** The known eras are D8's
two register connections and the other register rows of section 4. The known switching events are
the experiment 41 labelled set and the experiment 580 verified set. The injected shifts have known
dates, sizes, and classes. For each classifier the report prints the confusion table (era, switch,
unknown, undetected) at 7, 14, 28, and 60 days after onset, the two delay distributions, and the
share of switching events ever carried as an era with the days carried.

**Stage 0 gates,** at no more than 0.5 false alarms per Primary-year:

- **Detection:** median delay at most 7 days for era steps of 25%, and at most 21 days for steps of
  10%; D8's 2023-08-15 change alarmed within 14 days. Earlier detectors alarmed at D8 65 and 113
  days late (experiment 328).
- **Classification of eras:** at least 70% of injected era steps of 25% classed `era` by day 28, and
  both D8 connections classed `era` within 28 days of the alarm.
- **Classification of switches:** at most 10% of injected switching events, and at most 15% of the
  verified events of experiment 580, ever classed `era`.

**Two cautions on the real switching labels.** The verified events were selected for a mirrored
shift, so the mirror evidence is circular on that set and the score is optimistic. Experiment 565's
erratum records that telemetry confirms few of the experiment 41 labels, so a miss on that set may
be a wrong label. Experiment 295 labelled 42% of 14 to 60 day load shifts as states, which is the
failure these gates guard against.

**The injected bank holds both classes.** Each of 8 passes per fold injects shifts at 8 host
Primaries on the first day of the scored month, so every event has 28 days before the next monthly
refit. Four hosts get an era and four get a switching event.

- **Era kinds:** a level step of −10% of capacity, a level step of −25%, a level step of +15% (a
  load connects), and a solar-shaped reduction with a peak of 25% of capacity that follows the
  reanalysis radiation at the valid time. The solar-shaped kind is absent from the curriculum.
- **Switching kinds:** a transfer with the daily shape of the donor's demand, mirrored at an
  electrical neighbour so the pair's sum is conserved, reverting after a duration drawn from the
  durations of the verified events, built with the shared generator of experiment 660. One in four
  has its mirror hidden (the neighbour is treated as unmetered), and one in four lasts 60 to 120
  days. Those two are the hard cases, reported apart.
- **Truth:** an era event is scored against the injected power for both outputs. A switching event
  is scored against the power without the injected shift for the NRA output, and against the
  injected power for the prevailing output.

**The retrained worlds test days 28 to 90 and the refit's handling of a switching event.** Two
worlds each inject 4 eras and 4 switching events on one date (2025-12-01 and 2026-02-01), and the
three folds after each date are refitted on the world's data.

**D8's real era switches are descriptive.** Both register connections fall before the first scored
month, 2025-11, so stage 0 tests them on the detector and classifier alone. Scoring the network
across 2025-03-24 needs three extra folds (2025-04 to 2025-06) on at most 14 months of ensemble
history, because the stored ensemble summaries start on 2024-04-01. Those folds are an optional last
phase.

## 10. Placebo and stress readings

**P is trained and scored with random-date era signals, and C5 tests that P equals A0-aug.** Two
readings need no extra fit. The false-alarm stress feeds random-date era switches to the fitted
networks of O and L-best and reports the harm per false era. The O-date reading feeds O's network
the true date and class with the causal size, which splits O's lead over L-best into the part from
delay and the part from size estimation.

## 11. Scoring, noise, and the decision rule

**Rows are sliced by the origin's distance from the onset, and by the state of the shift at the
origin.** The distance slices are: the 14 days before the onset with a valid time after the onset
(reported, in no contrast); days 0 to 7; days 7 to 28; days 28 to 90 (retrained worlds); and stable
rows, 90 days or more after any shift and outside events. **The state slices are scored separately
for every learned arm: undetected, `unknown`, classified correctly, and classified wrongly.** Every
table reports both outputs, CRPS, the normalised mean absolute error of the median, coverage and
pinball loss at q01, q05, q95, and q99, the four load-shift scenarios of the `ml-experiment` skill,
cold winter nights, and the avoidable-curtailment cost. The row set is experiment 677's
monthly-refit rows, valid times 2025-11-01 to 2026-06-30.

**The minimum detectable effect comes from measured refit noise.** Nine contrasts at a family level
of 95% give a two-sided z of 2.77, and 80% power adds 0.84, so the detectable effect is 3.61
standard errors.

| Case | Noise of one refit (source) | Refits per arm | Standard error | Detectable effect |
|---|---|---|---|---|
| Arms sharing stages 1 and 2, all rows | 0.014 (699, within-seed contrasts) | 10 | 0.0044 | 0.016 from seeds; about 0.03 with month sampling (699's shared-weights interval, −0.036 to +0.020) |
| Arms sharing no weights, all rows | 0.087 (699) | 10 | 0.039 | 0.14 |
| Arms sharing no weights, D8 alone | 0.233 per seed-month (696) | 10 seeds, 8 months | measured | about 0.10 (696) |
| Prevailing heads on both sides, rows inside a shift | 0.395 (693) | 10 | 0.177 | 0.64 |
| A network output against A0-prev (heads on one side) | 0.395 | 16 | 0.099 | 0.36 |
| Arms sharing stages 1 and 2, post-shift rows | unknown, measured blind | 10 | noise / 3.16 | 1.14 times the noise |

**A noise step runs before any contrast is read.** The step prints, per slice, the standard
deviation over seeds of each paired contrast and no mean. If the post-shift noise is above 0.22,
which would put the detectable effect above 0.25 points, the seeds rise from 10 to 16. Experiment
706's interval on windows that contain a real shift (−0.317 to +1.622) shows that a few real events
resolve nothing under about 1 point, which is why the confirmation bank holds 512 events.

**Thresholds.** The demand protocol's minimum gain for a new fitted step or input is 0.02 points on
the headline. At 0.5 era switches per Primary-year, the 28 days after a switch are about 3.8% of
rows, so 0.02 points over all rows is 0.5 points in those rows. The threshold is 0.5 points for C1b
and 0.25 points for C2.

**Verdict.**

- **Supported:** C1b and C2 pass their thresholds with intervals that exclude zero, and every veto
  holds.
- **Partly supported:** C1b passes and C2 does not (a perfect signal helps and detection or
  classification is the limit), or C2 passes and a veto on switching events fails.
- **Not supported:** C1b's interval does not exclude zero in the gain direction, which means the
  prevailing output already does what an era signal would do.
- **Vetoes:** C4 above 0.03, or its upper interval end above 0.05; C5 outside ±0.05 on stable rows;
  C7 above 0.1, or its upper interval end above 0.2; the share of an injected switching event's
  depth left in the NRA median 7 days after the event ends more than 0.02 above A0's share; the
  copied share of a lag-only shift on experiment 581's bank more than 0.02 above A0's. A failed veto
  on stable rows or the placebo makes the verdict `inconclusive` pending an audit.
- **Machinery check:** if C1a is under 1 point for steps of 25%, the era signal is not reaching the
  output, and the run stops for an audit.

Intervals come from `losses_for_bootstrap` and `bootstrap_difference_at_level` with blocks of (fold
month, Primary), at the Bonferroni level for nine contrasts, with the interval over seeds beside
each.

## 12. Phases and gates

| Phase | Work | Gate to the next phase |
|---|---|---|
| A | Era table, detector, classifiers, stage 0 report (CPU, 2 cores) | Stage 0 gates; a learned arm whose gate fails is scored and labelled as failing the gate |
| B | Screen: 4 folds (2025-11, 2026-01, 2026-04, 2026-06), 5 seeds, all 10 arms, bank only; numbers labelled interim | C1a machinery check; L-best is the learned arm with the lowest CRPS on era events among the arms that pass the screen's C7 veto |
| C | Confirmation: 8 folds, 10 seeds, A0, A0-aug, O, P, L-best, L-all, and O-enc if the screen's C3 favours O-enc; 16 head refits for A0-prev | C1b decides whether phase D runs |
| D | Retrained worlds for C6 and the refit-side vetoes | none |
| E (optional) | D8 folds 2025-04 to 2025-06; O-enc-pre | Manager's agreement |

## 13. The stop rule, and what is new against each counted experiment

**The findings count 9 `not supported` or `refuted` verdicts on this idea in the seven named themes,
and 11 with the two experiments of the gross-demand-simulation theme.** Both counts exceed three, so
a further attempt needs the maintainer's approval. The manager states that the maintainer's named
request stands as that approval, and the plan quotes the request: "revisiting the idea that
substations experience these eras" (maintainer, 2026-11-19, via the manager's chat). If this
experiment ends `not supported`, the write-up recommends that no further era-signal experiment runs
without a new approval.

| Id | Verdict | What that experiment tried | What is new here |
|---|---|---|---|
| 294 | not supported | An offset per half-hour that D8's gas engine runs, in the additive-model stack, from the last settled day's engine state | The signal is the era (the engine is installed), not the engine's running state; the model is the neural network; the signal is persistent, so a day-old value is still valid |
| 295 | not supported | Hindsight segmentation of regimes, with no forecast scored against a model input; 42% of load shifts labelled as states | Forward-only detection with the delay scored at the origin; an explicit era-or-switch classifier with an `unknown` state; the signal feeds the network |
| 328 | not supported | Detect a change, then add regime-specific effects to an additive model; both detectors alarmed too late | An oracle arm separates the value of the signal from the detector; a six-feature daily detector gated in stage 0 before any fit; scoring by state of the shift |
| 361 | not supported | An as-of step detector for the engine's on/off spells at D8, with the steps removed from the training target | Eras, not spells; the target is untouched and the signal is an input; switching events are the only shifts removed |
| 363 | not supported | One penalised level per stretch between detected steps, which absorbed the solar terms and harmed D2 and D4 | Few eras under a persistence prior and a minimum size, not many stretches; an anchor fitted with the offset present; a veto on stable rows at every Primary |
| 571 | not supported | A smooth monthly drift embedding and the offset, in synthetic worlds, one fold, two seeds, no intervals | A piecewise-constant discrete signal; 10 seeds with intervals; rows after a switch scored apart; real D8 dates |
| 573 | not supported | A daily-knot or gated-jump offset asked to find steps of 2% to 6% as a by-product of the fit | Dating is done by an explicit detector on telemetry, and the network is told the state |
| 575 | not supported | The same offsets at label-sized steps; the offset moved about half the level | The explicit state input that experiment 575's own advice asks for; a persistent-step curriculum that teaches the network to move the whole level |
| 577 | refuted | Offsets constant between sparse steps, scored on all real rows; widening for jump risk made the forecast worse | Scoring on the weeks after a shift, where the effect lives; an era input, not a new form of the offset |
| 695 | not supported | The engine's inferred output removed from D8's target and lags, with a climatology added back | No target cleaning; the level of the era is the signal, not the engine's output |
| 696 | not supported | The engine's state given to the pre-training mixer only, and dropped at fine-tuning and at forecast time | The signal is present in stage 3 and at forecast time, enters the encoders through stage 3 columns, and every arm is paired with the control |

**The classification of a shift as era or switching event is new against all eleven.**

## 14. Scaling to 2,500 substations without labels

**No arm that would be ported needs a label, and the cost per substation is small.** The detector
reads only the substation's own power and a climatology: one day's update costs 730 run lengths
times 6 features, so 2,500 substations take well under a second per day in NumPy. The classifiers
add a regression on a few weeks of daily values and, where a neighbour is metered, the mirror
statistic. The era encoder is shared, so L-rule and L-soft add no parameter per substation, and
L-hmm adds 12. The bank's hidden-mirror cases measure what is lost at a substation with no metered
neighbour. The curriculum needs no real labels. The oracle arm and the era table do not scale and
are not meant to.

## 15. Compute

**Slot A has 8 cores, and the GPU share is not yet set, so the figures below are estimates to
replace with a one-fold timing pilot.** Experiment 667 records about 0.5 GPU-hours per fold for the
Candidate chain at 2 seeds. The split assumed is 9 minutes per seed for stage 1, stage 2, and the
table, and 6 minutes per stage 3 fit.

| Phase | Stage 1 and 2 chains | Stage 3 fits | Head fits | GPU job-hours |
|---|---|---|---|---|
| B | 12 (seeds 2 to 4; seeds 0 and 1 are read from experiment 677's chains if stored) | 200 | 0 | about 22 |
| C | 52 | 420 new (7 arms) | 128 | about 52 |
| D | 60 | 300 (5 arms) | 0 | about 39 |
| Bank inference, 8 passes per fold | | | | 15 to 25, to be measured |
| Total | 124 | 920 | 128 | about 130 to 140 |

**Each job needs about 14 GB of host memory (experiment 699) and about 5 GB of GPU memory
(experiment 696 peaked at 33 GB with 6 to 7 jobs).** Four jobs at once need about 64 GB of host
memory and 20 to 25 GB of GPU memory, and give about 35 hours of wall-clock time. The brief caps
each command at 16 GB and asks for the manager's agreement before the first fit. The proposed
ceiling is 150 GPU job-hours, and the run stops between folds at the ceiling. If the share is small,
cut phase E first, then phase D, then L-R and L-hmm from phase B. Phase A needs 2 cores and no GPU.

## 16. Data hygiene and anonymisation

- **Cleaned data everywhere:** `substation_zero` half-hours are dropped from training rows, lags,
  detector features, and scoring actuals, and D15 and D18 rows before 2025-03-01 are dropped.
- **Every detector and classifier setting is fixed on data before 2025-01, at Primaries other than
  D8,** before D8 or any scored month is examined.
- **Whether stage 1 rows include D8 before 2023-08-15 is unknown.** The implementer reads
  `e00667_k1.py` and prints the answer. Stage-3-only arms are unaffected.
- **No metered generator's series is published, with or without its name or identifier.** The
  solar-shaped injection is built from reanalysis radiation, and the engine's record is not
  used. Charts of D8 show the substation's own reading, which may be named, and never overlay a
  generator's output. Connection dates and sizes define eras and may be printed. The
  pre-anonymisation engine label on older pages is not copied.
- **Unlogged events:** NGED's logs may miss events, so stable rows may hold unlabelled shifts, and
  the harm table is optimistic.

## 17. Risks

- **The prevailing output may already capture the gain,** in which case C1b fails and the result is
  a clean `not supported`.
- **A long or unmirrored switching event may be classed as an era.** C7, the carried-depth veto, and
  the hard cases of the bank measure the harm, and the reversal rule limits the duration.
- **A real era may be classed as a switching event and erased from training rows.** The state slices
  report the cost of that error.
- **The injected switching events come from the generator of experiment 660, and the curriculum's
  lag-only shifts from the Candidate's recipe,** so a classifier may learn the generator. The real
  labelled sets in stage 0 are the check.
- **The curriculum may teach A0-aug to follow lags.** A0 and A0-aug are both reported.
- **Injected steps are simpler than a real connection.** A gas engine runs on a price-driven pattern
  that the four era kinds do not copy.
- **Experiment 565 found that an embedding added to the offset removed the offset's gain.** The
  anchor rule is the answer, and C4 would show the conflict.
- **The noise figures come from experiment 677's step 5 network and from prevailing heads,** not
  from the Candidate's stage 3, which is why the noise step comes first. A0 is refitted in the batch
  for every seed, because a control fitted in another batch scored 0.083 points apart in experiment
  696.

## 18. Order of implementation

**All code goes in `ml-experiments/<id>-era-signal-in-the-candidate-part-1/`, with the id from
`reserve_experiment`.** The suggested theme is `retraining-and-training-history`, with
`lsc-detectors`, `switching-labels-in-the-forecast`, and `lagged-power-and-error-correction` in
`also_relevant_to`, and `kind: other`. Each script gets an Opus code review before its first run.

1. `e<id>_era_table.py`: build and print the era table and the two sets of known switching events.
2. `e<id>_features.py`: daily detector features with the as-of cut.
3. `e<id>_bocpd.py` and `e<id>_hmm.py`: the two detectors, forward-only.
4. `e<id>_classify.py`: the mirror, duration, and weather-response evidence (importing
   `e00687_features.py` and the gate of experiment 592), the rule classifier, and the trained
   classifier.
5. `e<id>_inject.py`: the era injector, the switching injector built on
   `ml-experiments/00660-shared-generator-of-injected-load-transfer-events/`, and separate seeds for
   the tuning worlds, the curriculum, the bank, and the retrained worlds.
6. `e<id>_stage0.py`: the detector and classifier report on known and injected cases, with the
   gates.
7. `e<id>_signal.py`: the columns per (Primary, origin day) for every arm, with lag translation,
   switch removal, and the anchor.
8. `e<id>_model.py`: `EraPermLagMdn(PermLagMdn)`, reusing `widen_first_layer` (`e00696_model.py`),
   the read-by-index pattern of `OutputOffsetDrift` (`e00565_model_variants.py`), the mixer-only
   pattern of `ModularMdnDrift` (`e00559_model_drift.py`), the zero-padding `load_state_dict` of
   `PermLagMdn`, and an override of `LagOffsetMdn.penalty` that skips knot pairs around an era
   switch.
9. `e<id>_cand.py`: a copy of `e00693_cand.py` (the Candidate with extra columns and an injected
   bank, reading experiment 677's chains without writing to them) with `--era
   {none,aug,oracle,placebo,oracle-enc,all,rule,soft,hmm,reset}`.
10. `e<id>_checks.py`: the identity check, the signal cut, the row alignment, the origin-time
    weather assertion, and a synthetic test that a known anchor moves the output; no fit starts
    until `checks.json` passes, as in `e00696_chain.sh`.
11. `e<id>_noise.py`, `e<id>_score.py`, `e<id>_verdict.py`, and `e<id>_figures.py`, with the chain
    and GPU shell scripts wrapped in `systemd-run`.

## 19. What was verified, and what was not

**Checked against source files (by the design sub-agent):** the definitions of NRA, the Candidate,
the prevailing output, and a load shift in `final-models.md`, including that the prevailing
quantiles are a mixture of the Candidate's quantiles and a shifted copy; the classes `PermLagMdn`,
`StateAuxLagMdn`, `widen_first_layer`, `ModularMdnDrift`, `LagOffsetMdn` and its penalty,
`mixer_pre`, and `mixer_ft`; that `chain.sh` of experiment 667 repeats pre-training per monthly
fold; the origin and weather timing; the noise figures of experiments 693 (0.395), 696 (0.233, about
0.10), and 699 (0.087, 0.014, copied shares 0.075 to 0.127); the title, hypothesis, verdict, and
headline of each of the 11 counted experiments, read from each `result.md`; the engine row
in `printed_ecr.txt`; the 2023-08-15 entry in `data-processing.md`; the
0.02 minimum gain in the demand protocol; experiment 667's 0.5 GPU-hours per fold; the ensemble
summaries starting 2024-04-01; and the two DOIs, which resolve at doi.org to Fox, Sudderth, Jordan,
and Willsky (2011), "A sticky HDP-HMM with application to speaker diarization", and Killick,
Fearnhead, and Eckley (2012), "Optimal Detection of Changepoints With a Linear Computational Cost".

**Not checked:** the workbook rows for the 8.91 MW and 0.7 MW solar entries (taken from the findings
file); the feature names inside `e00687_features.py` beyond its constants, and the interface of gate
G4 in experiment 592's folder; the interface of experiment 660's generator; whether experiment 677's
chains store stage 2 weights for seeds 0 and 1 of the Candidate; the split of chain time between
stages; whether stage 1 holds D8's rows before 2023-08-15; the figures quoted from experiments 101,
456, 565, 580, and 592, which come from the findings file; the stop-rule count, which is the
findings file's judgement. `e00693_cand.py`'s docstring names `e00693_bank`, and no such file exists
in that folder: the bank code is inside `e00693_cand.py` and `e00581_bank`.

## 20. Further design decisions (these refine earlier sections)

The sections above state the arms and the two largest decisions (the encoder entry in section 7 and
the anchor rule in section 2). The list below states the remaining decisions.

1. **One cut everywhere.** Recompute the mirror statistic of experiment 687 and gate G4 of
   experiment 592 at the end of day D − 1, so the perturbation test, which changes every reading
   from 00:00 of day D, holds for all inputs.
2. **Era encoder identity.** The encoder output is multiplied by an indicator that the state is not
   `none`. The check asserts the identity after training, on rows whose state is `none`.
3. **C7 rows.** The bank is inference-only on one fold's network, so C7 and the carried-depth veto
   use only switching events that end within 21 days of the injection. Longer events are scored in
   the retrained worlds.
4. **Duration bands.** The gates and C7 are reported for events under 60, 60 to 120, and over 120
   days. The bank gains transfers longer than 120 days.
5. **Verified set size.** The verified set of experiment 580 holds 27 events, 4 with onsets in the
   scored window. State n = 27 and a binomial interval for the 15% gate. The duration prior, the
   injector's durations, and L-soft's training exclude those 4 events.
6. **Signed power at MVA-only Primaries.** Injected steps and the solar shape are built in signed
   power and then passed through an absolute value at MVA hosts. The telemetry estimate of D8's
   solar size states how it handles reverse power flow (`Monitor Direction.csv` marks D8 "Yes").
7. **D15 and D18.** Their state is `none` where the climatology and spread do not exist, and they
   stay out of the host pool.
8. **Winner's curse.** The phase C bank uses fresh injection seeds and hosts, and C2 is also
   reported on the four folds the screen did not use.
9. **Decision rule.** C1b or C2 with an interval that excludes zero below its threshold is `small
   gain`. A failed C7 or carried-depth veto gives "not supported for adoption", whatever C2 shows.
10. **Learning the injector.** The bank, the curriculum, and the classifier's training use different
    generator settings and seeds, and C2 is also reported on the out-of-curriculum solar-shaped
    kind. The claim is scoped to "injected level steps" unless the solar-shaped kind also passes.
11. **Curriculum harm.** Add L-best minus A0-aug and A0-aug minus A0 beside C4 and C7.
12. **D8 fade and network-level D8 test.** D8 from 2026-02-15 to 2026-05-31 leaves the stable rows,
    and the fade is reported as a descriptive real case scored on the network. The network-level
    test across D8's 2025-03-24 connection stays in optional phase E, and the plan says so plainly:
    the stored ensemble summaries start on 2024-04-01, so three extra folds with 14 months of
    history are possible, and the maintainer decides whether to fund them.
13. **Bank hosts.** An era host is never the mirror partner of, or a neighbour of, a switching pair
    in the same pass. A Primary with no metered neighbour hosts only the hidden-mirror kind.
14. **Stop-rule table corrections.** Row 295: it scored a latest-regime offset after a step (2.2
    points better), and a trailing 60-day mean residual did as well, so the trailing residual offset
    and experiment 706's look-back offset become no-fit reference rows on the bank. Row 571: the
    drift embedding recovered 96.5% of the oracle's gain on injected steps, the nearest earlier
    evidence and positive but synthetic. The novelty claim reads: "a causal classifier whose classes
    drive the forecast is new against all eleven" (experiment 295 classed persistent against
    temporary shifts with hindsight).
15. **Seeds and compute.** Raising seeds from 10 to 16 adds about 336 stage-3 fits and 40 GPU-hours.
    The plan cuts L-hmm and L-R from phase C first, then phase D, to stay under the 150 GPU job-hour
    ceiling.
16. **Stored chains.** Seeds 0 and 1 of stage 1 and stage 2 are read from experiment 667's chains,
    not 677's.
17. **Era rate.** The register gives about 4 connections of 1 MW or more since 2019-09 across 16
    Primaries, about 0.04 per Primary-year, so 0.5 points in the 28 days after a switch is about
    0.002 points on the headline. The plan states the headline-equivalent gain from the era table's
    rate and counts telemetry-only regime changes separately.
18. **Seed pairing for C2.** A0-prev's 16 head refits pair with the 10 network seeds by seed index
    (seeds 0 to 9), with the six extra head refits used only for the head-noise estimate. In folds
    2025-11 and 2025-12 the prevailing rows come from the unfitted fallback head, so those folds
    carry no head noise and are reported apart.
19. **Minor.** The copied share of a lag-only shift is 0.075 on experiment 581's bank and 0.024 on
    experiment 660's bank (0.127 is experiment 699's no-injection arm). The noise figures 0.087 and
    0.014 are standard deviations of MAE, and 0.014 rests on 3 seeds. The weather assertion tightens
    to `nwp_init_time` equal to 00:00 UTC of the origin day. "Clean history" for false-alarm tuning
    means outside labelled events and on surrogate series. The solar-shaped injection uses CAMS or a
    perturbed field rather than the ERA5 field the classifier reads. C5 passes when both interval
    ends lie inside ±0.05. C6 states its minimum detectable effect from its 8 eras. Prose fixes:
    conclusions in bold leads, no pronoun-opening sentences, no money metaphor, and expand ECMWF,
    ERA5, CAMS, PELT, and GELU on first use.

## 21. Stage 0 results and the L-stat arm (interim)

**Stage 0 ran on 2026-11-19 and the numbers below are interim.** The maintainer asked for arm
L-stat:
detect eras from 28-day summary statistics of raw power, find steps in those statistics, give each
step an era id, and feed the id to the network. The design of L-stat is in this section and in the
arm table. It replaces L-R, so the arm count and the compute budget stay level.

**The perturbation test found one real leak, which is fixed.** Experiment 687's noise unit and the
G4 threshold read past the origin for neighbours with an incomplete window. The rule and soft
classifiers were refitted, and the cut test passes for 200 pairs and every arm.

**The daily-feature change-point detector detects few small steps.** It needs a step of 3 noise
units
in one feature. At D8 one unit is 9% to 13% of capacity, so it detects 13 of 64 injected steps of
10% and 14 of 64 solar-shaped steps, and misses D8's 2025 solar connection. It detects D8's gas
change
18 days after 2023-08-15 (gate: 14 days). Injected steps of 25% pass the detection gate (34 of 64
within 7 days).

**L-stat detects both D8 changes and rarely mistakes a load shift for an era.**

- D8 gas change: alarm 10 days after 2023-08-15, classed `era` 9 days later (gate passed).
- D8 solar change: alarm 33 days after the register date (2025-03-24) and 21 days after the
  telemetry date fitted on the same statistic (2025-04-05). The gate passes on the telemetry date,
  which rests on the same statistic as the alarm, so the pass is labelled as such. The earlier
  telemetry estimate (2025-06-10) was 9 weeks later and is replaced.
- Verified switching events (experiment 580, 15 transfers outside the scored window): 0 ever classed
  `era` (Wilson 95% interval 0.00 to 0.20). Experiment 41 labelled events: 2 of 93 (0.02).
- Injected bank, `era` class by day 28: L-stat 79 of 190 era events, against 13 for L-rule.
  L-stat's `era` class on injected switching events is 5 of 211 at day 28.
- False alarms: 0.31 per Primary-year on clean history at threshold 4.0 (limit 0.5).

**Expectation, recorded before any fit:** L-stat keeps 10% to 30% of the ceiling on steps of 25% and
under 10% on steps of 10%, because at most about 13 of the 28 scored days follow an `era` class
(the state needs a dwell of 14 days). C7 is within +-0.1.

**Plan changes from the L-stat design.**

- L-stat is eligible as L-best. Two descriptive readings sit outside the nine contrasts: L-stat
  minus L-rule on injected era events (NRA output), and L-stat with its era-id table zeroed at
  inference, on stable rows and on rows after an era.
- Era id 0 means "no era yet" and carries no vector. An id first confirmed after the training cut
  uses one shared "new era" vector.
- Anchor rule for every learned arm: the anchor equals the level change minus the offset's own
  change between the knot before the onset and the newest knot, clipped between zero and the full
  change. The known-answer check gains a case with an era confirmed after a refit. Phases B and C
  never exercise that case, and phase D does.
- Phase B gains 20 stage 3 fits (about 2 GPU job-hours) and about 1 hour of bank inference, and
  loses L-R. An irregular-engine era kind exists in stage 0 only.

## 22. Amendments made while running phase B (interim)

**Phase B runs on 2 seeds, not the pre-registered 5.** Experiment 667 stores the stage 1 and stage 2
weights for seeds 0 and 1 only, so seeds 2 to 4 would need new stage 1 and stage 2 chains. The
manager accepted seeds 0 and 1 on 2026-11-19. The screen's intervals come from 2 seeds across
4 folds, so the intervals over seeds are wide and every phase B number stays interim. The nine
contrasts keep their Bonferroni level, and the write-up states the 2-seed caveat beside each
interval.

**The curriculum holds 4 eras per fold and seed, not the planned 8.** Each injected pass touches up
to 6 of the 13 eligible Primaries, so at most 2 passes with disjoint touched Primaries fit. Their
onsets lie 120 and 100 days before the cut, not 120 to 60 days. The one-fold pilot logged
`alt_passes 2` and `alt_eras 4`. The curriculum arms therefore see fewer injected eras than
designed, and the write-up counts that as a limitation of every curriculum arm.

**The oracle's curriculum switch vectors are read with hindsight.** `hindsight_vector` averages up
to 120 days from the onset, so a long injected switching event that starts 100 to 120 days before
the cut can read telemetry from the scored month. Only the oracle arms are affected, and they are
labelled non-causal.

**The bank's oracle and placebo hold the injected events only.** The bank overlays each world at
the touched Primaries, so the real record would have to be placed inside the 30-day bank window,
which the separation rule forbids.

## 23. First interim result: stage 3 keeps its first epoch, so the curriculum is void in phase B

**Every contrast that involves the curriculum (the injection of eras) is void in phase B, and the
arm A0-aug equals A0 by construction.** Stage 3 stops after 8 epochs without a gain in the
validation loss and keeps the best state. In fold 2025-11, seed 0, every arm ran 9 epochs and kept
the state of the first epoch, so the odd epochs, which read the injected world, never survive the
selection. A0 and A0-aug have the same validation loss to 16 digits, and their forecasts differ by
at most 4e-6 in the median. The curriculum world also held only 2 eras in this fold, not the 4 of
the one-fold pilot and not the 8 planned. The signal arms learn the era path from one epoch on the
real record, so phase B still tests the signal on the real record but not the curriculum.

**The first numbers are interim, from one fold and one seed, and carry no interval.** The CRPS in
points of effective capacity, over all rows, is 4.183 for A0, 4.178 for the oracle arm (D8: 7.440
against 7.546), and 4.264 for the placebo (+0.08, mostly on the other Primaries). Nothing here
supports or refutes the hypothesis.

## 24. Phase B2 (proposed): mix the curriculum into every batch

**Phase B2 keeps phase B's selection rule and mixes the injected world into every batch, for four
arms.** A fixed number of epochs was rejected in the Opus review, because stage 3 fine-tunes the
stage 2 weights at a low learning rate and every arm kept its first epoch. A fixed run of 4 epochs
would give every arm, A0 included, a worse baseline than the production recipe, and the last update
would be made on the injected world. The arms are A0-aug, the oracle, the placebo, and L-stat. A0
never reads the injected world, so A0 under B2 is phase B's A0, and B2 reuses those fits.

**Rule.** In every epoch, each training row in a batch reads the curriculum tensors with probability
0.5, and its plain values otherwise. The coin comes from a separate CPU generator seeded with
`seed * 1000 + epoch`, so the row order is phase B's. Selection is unchanged: patience 8, best state
restored, loss on plain validation rows. The per-epoch validation loss is logged. The odd-epoch
alternation of phase B is retired, and the curriculum acts from the first step, so selection cannot
strip it out.

**Contrasts.** Each B2 arm is paired against A0 from phase B on the same rows: A0-aug minus A0
(lag-only
learning from injected eras), oracle minus A0, placebo minus A0, and L-stat minus A0. The write-up
also reports CRPS on the touched Primaries separately, as a harm check. It says that the selection
rule changed after an interim look and why, gives the realised number of eras per fold, and says
that
for the signal arms the mixing is partly a noise augmentation, because the injected rows get a
second noise draw.

**B2 runs on phase B's cached curriculum worlds.** The world held 2 eras in fold 2025-11, not the
planned 8, because each pass touches up to 6 of 13 eligible Primaries and the passes must be
disjoint. The cache is keyed by fold and seed, so B2 reuses the same worlds, and `alt_eras` is
logged
for every fit. A fold with fewer than 4 eras is reported as a weak test of the curriculum.

**Decision rule, fixed before the first B2 fit.** A B2 arm helps when its difference from A0 on all
rows is below 0 with the bootstrap interval (blocks of fold month and Primary, 95%) excluding zero,
and the difference on the touched Primaries is not above +0.05 points. An arm whose interval
contains zero is reported as not distinguishable from A0. Every B2 number is interim until all 4
folds and both seeds are scored.

**Cost.** 4 arms on 4 folds and 2 seeds is 32 fits, about 9 to 10 minutes each at 2 concurrent jobs,
so about 2.5 hours of wall-clock time after phase B finishes. Phase B's aug result is void.

## 25. Phase E: a one-substation robustness check on D8, reported apart

**Phase E is a robustness check on one substation, and its numbers are never pooled with the screen
folds'.** The maintainer approved it after seeing the corrected scope. It runs the arms none,
oracle,
placebo, stat, and rule (5 arms) on the folds 2025-04, 2025-05, and 2025-06 with seeds 0 and 1,
which
is 30 fits, and it scores the rows against its own row set id,
`experiment_709_phase_e_monthly_refit_rows_valid_2025_04_to_2025_06`. Its arms are named `e-<arm>`.

**The history is shorter, and D8's solar connection sits at the start of the test months.** The
ensemble summaries start on 2024-04-01, so the 2025-04 fold trains on about 12 months of ensemble
history, against about 19 for the first screen fold. D8's 8.91 MW solar farm connected on
2025-03-24,
a week before the first test month, so the first fold's training rows hold almost no days after the
connection, and the fold tests how an era signal behaves when the new era is a week old. The
hidden-Markov models were fitted on days before 2025-01-01, so they read no day of these folds.

**Phase E scores every fifth day's forecast, not every day's.** Experiment 667 keeps experiment
397's
stored rows before 2025-07-01, which hold only every fifth day's run, so each E fold has about 6
forecast origins in its month, against about 30 for a screen fold. The 7-day embargo and the 7-day
down-weight window cover one or two runs in the training rows near the cut. The intervals are wider
for that reason. Rebuilding the rows from experiment 607's daily summaries would give daily origins
but change the recipe relative to the screen folds, so the plan keeps the stored rows. The daily
frames rebuilt in `frames_e/` serve only the timing check, and the rebuild is optional.

**The oracle in phase E reads the scored months.** The oracle's change vector for D8's solar farm
averages up to 120 days after its onset (2025-04-05 in the era table), which reaches into the test
months of the three E folds. The E oracle is reported as an arm that knows the scored months, and
the placebo, which moves the same vector to other dates, carries part of the same information.

**The extra work is outside the screen's pipeline, and every piece is a copy in this experiment's
folder.** The daily demand frames of 2025-04-01 to 2025-06-30 are rebuilt by `e00709_e_frames.py`
from the archive with experiment 48's `build_run`, into `frames_e/` of this experiment's data
folder;
a rebuild of 2025-05-29 matched a stored snapshot frame on all its rows. Six new stage 1, stage 2,
and table chains (3 folds, 2 seeds) run through `e00709_e_chain.sh`, which writes to `e_667/` of
this
experiment's data folder and reads experiment 667's extended inputs only. Each changed or new script
gets an Opus review before its first run. Phase E starts after phase B and B2 end.

**The first phase E pilot wrote 61 stand-in frames into experiment 667's folder, and they await the
maintainer's approval to delete.** Experiment 667's `fold.install` writes a one-row stand-in demand
frame for each run that has none. The pilot chain for fold 2025-04 wrote 61 of them
(`init=2025-03-01` to `init=2025-04-30`, 1,057 bytes each) into experiment 667's `inputs/frames`,
and the chain's guard stopped the run. None of the files replaced a real frame, because the folder
held no 2025 file before. No other experiment script reads the folder. `e00709_e_stub.py` now writes
the stand-ins under `e_667/inputs/frames` of this experiment's data folder, and the guard still
fails the chain if a file lands in experiment 667's folder. The deletion command and the list of
files are in `briefs/eras_709_handover3.md` (private).

**Phase E's contract rows relax one assertion, and every phase E table says so.** The stored rows of
experiment 667 before 2025-07-01 keep only half-hours with an observed target, so a series-origin
holds between 637 and 672 forecast rows. Experiment 667's `contract_rows` asserts exactly 672 (14
days of half-hours) per series-origin. The copy `e00709_e_contract.py` asserts at most 672 and keeps
every other assertion. An Opus code review verified that the relaxation lets no row with a wrong
origin or valid time through, that the scorer computes CRPS only where the target is observed, and
that all five arms are scored on the same rows (`losses_for_bootstrap` raises otherwise). The
caption of every phase E table reads: "about 6 origins per month, wide intervals, one-substation
robustness check; the oracle reads the scored months; series-origins may hold fewer than 672 rows
because the stored rows keep only half-hours with an observed target".

## 26. Open questions for the maintainer, and one outlier at a single substation (interim)

**Whether phase E should use daily origins, and whether any learned arm is worth carrying on, are
open questions that this page does not decide.** The stored rows give about 6 origins per month, and
rebuilding daily rows from experiment 607's daily summaries would change the recipe against the
screen folds. For the second question, the arms that have lowered the error at D8 are the oracle
(not distinguishable from zero at 99.44%), `hmm` in the screen months, and `e-stat` in phase E.

**The `all` arm's D8 outlier in fold 2026-01 is consistent with a temporary excursion that `all`
treated as an era.** The `all` arm turns every detected shift into an era. At D8 it flags an episode
from 2026-01-19 to 2026-02-02, with the era starting on 2026-01-21. D8's daily mean power is about
7.5 MW from 2026-01-19 to 2026-01-25, against about 4 MW before and after. The summed daily mean of
D8's three electrical neighbours (D3, D5, and D14) stays between 20 and 25 MW with no mirror-image
change, so the excursion is not conserved load moving between neighbours. D8 has no labelled
switching event in 2026. The arm `all` has no classifier to reject the shift, so it carries the
excursion forward as a persistent era.

## 27. Phase B2 result: one arm passes the rule at 95% only (interim)

**Only `b2-aug` meets the pre-registered B2 decision rule, and only at the 95% level.** The table
compares each B2 arm with phase B's `none` on the same
rows (4 fold months, seeds 0 and 1). The 95% intervals resample whole fold months only, so each
rests on 4 months. CRPS differences are in points of effective capacity.

| Arm | All rows | D8 | Touched Primaries |
|---|---|---|---|
| `b2-aug` | -0.050 [-0.103, -0.001] | -0.007 | +0.008 [-0.089, +0.198] |
| `b2-placebo` | -0.037 [-0.127, +0.096] | +0.001 | -0.011 |
| `b2-oracle` | +0.009 [-0.010, +0.032] | -0.058 [-0.076, -0.036] | +0.026 |
| `b2-stat` | +0.009 [-0.060, +0.118] | +0.386 [+0.071, +0.629] | +0.041 |

**The curriculum contrast `aug` against `none` is not decisive.** The `b2-aug` interval only just
excludes zero. `b2-placebo` moves the same way with an interval that contains zero, which is what a
noise augmentation would do, because the injected rows get a second noise draw. The curriculum world
also held 2 eras in 6 of the 8 fold-and-seed pairs and 4 in the other 2 (seed 0 of folds 2026-04 and
2026-06), so 6 of the 8 pairs are weak tests of the curriculum (a fold with fewer than 4 eras is a
weak
test, section 24). The `b2-oracle` gain at D8 repeats phase B's, and `b2-stat` is worse at D8 as in
phase B.

## Files

- `ml-experiments/00709-era-signal-in-the-candidate-part-1/e00709_era_table.py`,
  `e00709_features.py`, `e00709_statdetect.py`, `e00709_bocpd.py`, `e00709_hmm.py`,
  `e00709_classify.py`, `e00709_stage0.py`, `e00709_probe_hosts.py`: phase A, which builds the era
  table of known cases, the detectors, the era-or-switch classifiers, and the stage 0 report.
- `e00709_signal.py`, `e00709_inject.py`, `e00709_model.py`: the era signal tables per arm, the
  injection of eras and switching events, and the network's signal input.
- `e00709_cand.py`: the stage 3 fit, predict, and bank commands of the Candidate for one arm, fold,
  and seed (`--rule es` for phase B, `--rule b2` for phase B2, and the `IS_E` branches for phase E).
- `e00709_chain.sh`: runs the train, predict, and bank phases for a list of arms, folds, and seeds
  with capped memory.
- `e00709_checks.py`: the synthetic, signal, alignment, and weather checks that gate the chain.
- `e00709_score.py`: scores stored forecasts on the real rows (all rows, D8, other Primaries, the 14
  days after a stat alarm, the touched Primaries); `e00709_bank_score.py` scores the injected bank
  on eras and switching events; `e00709_stats.py` holds shared statistics.
- `e00709_e_chain.sh`, `e00709_e_gpu.sh`, `e00709_e_k1.py`, `e00709_e_cand.py`,
  `e00709_e_frames.py`, `e00709_e_stub.py`, `e00709_e_contract.py`: the phase E copies of experiment
  667's chain, which write to `e_667/` of the data folder, redirect stand-in frames there, and relax
  the rows-per-origin assertion.
- Earlier experiments read: [experiment
  667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md) (stage 1
  and 2 weights, extended inputs), and experiments 559, 572, 574, 591, 613, 660, 687, and 696 for
  the injection and the Candidate code.
- Data (private, not committed): `data/ml-experiments/00709-era-signal-in-the-candidate-part-1/`
  holds `era_table.parquet`, `known_events.parquet`, `signals/` (one table per arm), `fits/`
  (network weights), `predictions/` (80 phase B sets, 32 `b2-` sets, and 30 `e-` sets, each a seed's
  quantile forecasts), `bank/` (injected-bank forecasts and event tables), `scores/`
  (`interim_phase_b_real_rows.csv`, `interim_b2_real_rows.csv`, `interim_b2_rows.csv`,
  `interim_phase_e_rows.csv`,
  `interim_bank_rows.csv`), `checks.json`, and `e_667/` (phase E chains).
- Briefs and reviews: the findings file, the design, and the design review were written under
  `data/ml-experiments/briefs/eras_embedding/` (private).
