---
id: "00507"
slug: history-encoder-first-test-oracle-repair-and-total-variation
title: >-
  Is there enough to gain from repairing shifted power readings, rather than down-weighting them, to
  justify a learned 14-day history encoder?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v2
date: 2026-11-02
size: >-
  complex. Stored: about 100 small saved neural networks, their forecast tables, and injected
  worlds, all under the experiment's private data folder. Serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: more than one design would
  defensibly test the hypothesis (how a repair is scored, the injection generator, the
  total-variation repair), so the plan fixes each. Callers: the scripts import experiments 404,
  428, 454, and 472, and the callers of that chain cannot be named without searching. Reviews: one
  Sonnet plan review, one Sonnet code review before any number is used, one Sonnet results review,
  and two Sonnet write-up reviews. The brief asks for Sonnet only.
hypothesis: >-
  Repairing the shifted readings in the last 14 days of power, instead of down-weighting them as
  experiments 402 to 482 did, leaves enough forecast gain to justify a learned history encoder. A
  perfect (oracle) repair of the three lag readings improves the lag correction of experiment 472
  (on the ERA5 lab anchor) on rows where a shift is in force. A non-learned repair (total-variation
  denoising of
  the 14-day residual) keeps less than half of that gain once genuine level changes are in the
  data, and a learned encoder can reach the accuracy that the oracle's noise curve demands.
prediction: >-
  Written before any fit. Scores are CRPS in points of effective capacity at the signed-MW six on
  held-out injected worlds. P1. The retrained oracle correction beats the retrained plain
  correction on injected harm and target rows by at least 0.3 points (the brief's prediction; the
  stop threshold of rule (a) is the injected-row minimum detectable effect, which the plan expects
  to be near 0.05 to 0.15). P2. Total-variation repair keeps under half of that gain on the
  new-generator worlds, because it erases genuine level changes (60% of the steps in force at the
  origin). P3. The encoder's onset error, size error, and share of steps missed fall in the part of
  the noise curve that keeps at least half of the oracle gain. Any of the three can fail.
decision_rule: >-
  Fixed in the section "Decision rule", below, before any score is read. In order: (a) stop the
  whole line if the retrained oracle correction does not beat the retrained plain correction on
  injected harm and target rows by more than the injected-row minimum detectable effect (from the
  3-seed spread of the retrained plain control) with an interval excluding zero; (b) promote the
  non-learned repair and stop the neural encoder only if total-variation repair keeps at least half
  of the oracle gain on the old worlds and on the new-generator worlds with genuine changes (lower
  end of the interval on the paired difference of TV gain and half the oracle gain at or above
  zero), is not shown worse than the plain correction by more than
  0.03 points on rows with no labelled event at the six, and is not shown worse by more than 0.02
  points on q99 pinball loss; (c) stop the neural encoder if the encoder's errors fall where the
  noise curve keeps under half of the oracle gain, or its validation step loss stops falling
  before epoch 10. The smallest gain worth porting is the injected-row minimum detectable effect,
  because the repair adds a convex solve per Primary and a 672-half-hour history store.
benchmark: >-
  The retrained plain correction (experiment 472's recipe, 32 units, penalty 1e-2, fit seeds 0 to 2)
  on the same ERA5 and CAMS lab lagless table. Every contrast is lab against lab. The stored
  ENS-based AvgAnchor and K2-S1 forecasts are not contrasted.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from 11 quantiles, by row
  class and group, as in experiment 472's report; q01, q05, q95, and q99 coverage and pinball loss.
code: >-
  ml-experiments/00507-history-encoder-first-test-oracle-repair-and-total-variation/
verdict: partly supported
best_arm: >-
  Oracle repair of the three lag readings on the new generator worlds: CRPS lower than the plain
  correction by 0.59 points of capacity at the six Primaries that report signed MW (95% interval
  0.51 to 0.67; minimum detectable effect 0.09) and by 0.73 at the eight that report MVA only
  (0.66 to 0.79).
headline: >-
  Interim and unaudited. A perfect repair of shifted lag readings improves the score on rows where
  a shift touches a lag (by 0.2 to 0.7 CRPS points of capacity, depending on the rows and worlds).
  The one non-learned repair tried, total-variation denoising, was run on the new generator worlds
  only and did not help: 0.045 points worse than no repair at the six, 0.012 better at the eight
  (2% of the oracle gain). The learned encoder was not tried, because the line is paused.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Interim and unaudited. If the readings of the last 14 days could be repaired exactly, the
  forecast would improve by 0.2 to 0.7 CRPS points of capacity on rows where a shift touches a
  lag, so a repair is worth something. The cheap repair we tried, total-variation denoising, kept
  at most 2% of that gain, but its tuning picked a setting that barely repairs anything, so the
  test cannot say whether any non-learned repair could work. A learned repair was not tested.
lessons: >-
  The oracle ceiling is real but flattered by the lab setting. The single non-learned repair we
  tried failed, and was tuned by a criterion whose best answer was almost no repair. The learned
  encoder question is untouched.
advice:
  - >-
    Do not rely on total-variation denoising of the 14-day residual tuned to minimise the mean lag
    error over all windows, because it repaired almost nothing; a setting tuned only on windows
    with a temporary shift has not been tried (tuning section).
  - >-
    Score a repair on rows whose lag positions a shift touches, and split them into temporary and
    genuine shifts, because the oracle gain is 1.25 to 1.58 points on temporary-shift rows and only
    0.24 on genuine-shift rows (oracle table).
  - >-
    Report a repair's effect on rows with no shift touching a lag, because the oracle retrained on
    exact lags was 0.09 points worse there at the six and 0.05 worse at the eight (clean-row
    table).
  - >-
    Run the oracle noise curve on the new generator worlds before building a learned history
    encoder, because this page measured only the ceiling and one non-learned repair.
code: >-
  ml-experiments/00507-history-encoder-first-test-oracle-repair-and-total-variation/
---

**This page is interim and unaudited: it reports a perfect repair of shifted lag readings (an
oracle) and one non-learned repair, and no learned encoder.** We asked whether the power readings
of the last 14 days need repairing, or whether down-weighting them (as experiments 402 to 482 did)
is enough. We injected shifts into the readings of 14 Primary substations: temporary load transfers
(family A), and genuine level changes that move the true power (family B). We retrained the lag
correction of experiment 472, a small neural network that adjusts the forecast using the last
three lag readings, and scored it with the lags repaired in three ways: not repaired ("plain"),
repaired by an oracle that knows the injected shift, and repaired by total-variation denoising of
the 14-day residual (the reading minus the forecast of a model that has no power input).

**The learned encoder (rung T1) was not run because the line is paused, not because a kill rule
fired.** The oracle improved the score, so the stop rule (a) did not fire. The non-learned repair
did not keep half of the oracle's gain, so rule (b), which would have promoted it in place of the
encoder, did not fire either. The next step would have been the noise curve and one encoder fit.

## Results

**Scores are CRPS in points of capacity (lower is better), on the four screen folds** (2026-06,
2026-01, 2025-10, 2026-03), three fit seeds each, on held-out injected worlds 200 to 202. "The
six" are the six Primaries that report signed MW, and "the eight" are the eight that report MVA
only. The minimum detectable effect (MDE) is experiment 454's formula: 2.8 times the mean standard
deviation of paired seed differences, over the square root of 4 folds. Intervals are the exact 95%
month-block resamples of the four fold means, a coarse resample. The weather is ERA5 reanalysis
and CAMS in both history and forecast, with one seed.

![Change in CRPS against the plain correction on rows where a shift touches a lag](figure_1_gain_over_plain.svg)

| Rows and worlds | Group | Oracle minus plain | Total-variation minus plain |
|---|---|---|---|
| Old worlds (experiment 428): harm and target rows with an injected shift | six | -0.214 (-0.314 to -0.132), MDE 0.014 | not run |
| Old worlds: same rows | eight | -0.466 (-0.581 to -0.309), MDE 0.223 | not run |
| New generator worlds: rows where a shift touches a lag | six | -0.589 (-0.674 to -0.505), MDE 0.089 | +0.045 (-0.006 to +0.139) |
| New generator worlds: same rows | eight | -0.727 (-0.793 to -0.661), MDE 0.088 | -0.012 (-0.018 to -0.005) |

- **Rule (a) did not fire: repairing the lags helps.** At the six, the old-world oracle gain was
  15.6 times the MDE, with an interval excluding zero. At the eight it was 2.1 times the MDE. The
  zero-lag control (experiment 472's correction with all lag inputs set to zero) was worse than
  plain by 0.49 (six) and 0.91 (eight) points on the old worlds, and by 2.43 and 2.65 on the new
  worlds, so the lag readings carry information.
- **The oracle gain sits on temporary shifts.** On new-world rows where a temporary (family A) shift
  touches a lag, the oracle gain was 1.25 (six) and 1.58 (eight) points. Where only a genuine
  (family B) shift touches a lag, it was 0.24 and 0.24. The oracle that removes a constant level
  per temporary shift (oracle-constant) gained 0.57 (six) and 0.68 (eight) over plain, close to the
  exact oracle, so most of the gain needs only the right level.
- **The oracle's gain depends on how it is applied.** Training on shifted lags and scoring on exact
  lags (oracle re-prediction) gained 0.18 (six) and 0.21 (eight) on the old worlds, against 0.21
  and 0.47 for the retrained oracle. The old-world gain at the eight roughly halves.
- **Prediction P1 (an oracle gain of at least 0.3 points) was met on the new worlds and at the
  eight on the old worlds,** and was missed at the six on the old worlds (0.21).
- **Total-variation repair did not keep half of the oracle gain, on the new generator worlds.** The
  paired difference between the repair's loss and the halfway point to the oracle was +0.34 (0.25
  to 0.45) at the six and +0.35 (0.31 to 0.39) at the eight; a negative value would have meant the
  repair keeps more than half. At the six the repair was worse than plain by 0.045 points. At the
  eight it kept 1.6% of the oracle gain (0.012 of 0.727). With 60% of the steps genuine, instead
  of the generator's 79%, the paired difference was +0.37 and +0.38, so the result did not change
  at that share. The repair was not run on the old worlds.
- **Rows with no shift touching a lag.** The repair's change in CRPS against plain was +0.028
  (-0.000 to +0.084) at the six and 0.000 at the eight, and in q99 pinball loss +0.011 (-0.000 to
  +0.031) at the six and +0.000 at the eight. The rule's bound applies to rows with no labelled
  event on the old worlds, which were not scored for this repair, so the bound was not tested. The
  oracle, retrained on exact lags, was 0.09 points worse than plain on these rows at the six
  (0.016 to 0.185) and 0.05 worse at the eight (0.03 to 0.09). We have not tested why.

## Tuning, and why the repair result is weak

![Mean absolute error of the repaired lags against the exact lags, for each total-variation setting](figure_2_tv_tuning.svg)

**The tuning picked the weakest repair, so this test cannot show whether any non-learned repair
works.** On a separate tuning world (origins in 2024, no scored month), all 30 settings (6 weights
times 5 values of tau) gave a larger mean lag error than no repair (0.124 spreads, where a spread
is the Primary's residual standard deviation). The best setting, weight 16 and tau 5, the largest
of each, gave 0.127. The criterion averages over all tuning windows, most of which have no shift
touching a lag, and any repair adds noise to those windows. A setting tuned only on windows with
a temporary shift in force was not tried, because the plan fixed the criterion before any score
was read. The take-home message therefore does not claim that total-variation denoising cannot
tell a temporary shift from a genuine level change; this test never examined that.

## Outcome against the plan

| Item | Outcome |
|---|---|
| Rule (a): stop if the oracle does not beat plain by more than the MDE | Did not fire (old worlds, both groups) |
| Rule (b): promote total-variation repair if it keeps at least half the oracle gain | Did not fire (new worlds); not run on old worlds |
| Rule (c): stop the encoder if its errors fall where the noise curve keeps under half the gain | Not evaluated: no encoder, no noise curve |
| P1: oracle gain of at least 0.3 points | Partly met (above) |
| P2: total-variation repair keeps under half | Met on the new worlds; weak evidence (tuning) |
| P3: encoder errors fall in the useful part of the noise curve | Not tested |

## Limits

- **The lab setting flatters detection.** ERA5 knows the weather that happened, so the reading
  minus the forecast is smaller than at serving. A single-seed null is informative only when far
  from the MDE, and the six-Primary total-variation result (+0.045, MDE 0.089) is not far from it.
- **The NGED event labels are incomplete,** so counts of events are lower bounds.
- **Generator.** The generator makes 79% of the steps in force at the origin genuine, not the
  planned 60%. The 79% follows from the shares and durations chosen (derived 78%).
- **Code review.** A code review found 12 problems before any number was read. Seven were fixed:
  the oracle now uses the cleaned readings, events are a function of seed, Primary, and day, the
  size cap applies to the final step, circuit events draw a fraction of 0.3 to 1, the self-test
  tolerance is relative, an N0 slice no longer wraps, and the injection spread comes from the
  earliest table. Not fixed: the oracle and plain fit seeds are not paired by world, which makes
  the seed spread in the MDE approximate; the row count was not asserted against experiment 404's
  scoring rows. The review's other findings were fixed or do not change a number.
- **N0.** Of the 34 events with a measured deviation, the median coefficient of variation of the
  deviation is 0.44, and 29 have a correlation above 0.5 with the Primary's own daily demand, so a
  flat step describes most real events poorly. The generator draws shapes in the planned thirds
  regardless (tables in the private `work/n0/` folder).
