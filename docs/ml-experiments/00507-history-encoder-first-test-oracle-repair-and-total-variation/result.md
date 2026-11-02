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
  ENS-based AvgAnchor and K2-S1 forecasts are not contrasted (second set of edits, 2026-11-02).
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from 11 quantiles, by row
  class and group, as in experiment 472's report; q01, q05, q95, and q99 coverage and pinball loss.
code: >-
  ml-experiments/00507-history-encoder-first-test-oracle-repair-and-total-variation/
verdict: partly supported
best_arm: >-
  Oracle repair of the three lag readings on the new generator worlds: CRPS lower than the plain
  correction by 0.59 points of capacity at the signed-MW six (95% interval 0.51 to 0.67; minimum
  detectable effect 0.09) and by 0.73 at the MVA-only eight (0.66 to 0.79).
headline: >-
  A perfect repair of shifted lag readings helps (0.2 to 0.7 CRPS points on rows where a shift
  touches a lag), but total-variation denoising, the cheap non-learned repair, kept 0 to 2% of that
  gain, far below the half that would have made a neural encoder unnecessary. No encoder was fitted.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  If the readings of the last 14 days could be repaired exactly, the forecast of rows hit by a
  shifted reading would improve by 0.2 to 0.7 CRPS points of capacity, so the repair is worth
  something. The cheap repair we tried, total-variation denoising, recovered almost none of it
  because it cannot tell a temporary shift from a genuine level change. The next test is whether a
  learned repair can do better, and its first check is whether it can reach the accuracy the oracle
  noise curve demands.
lessons: >-
  The oracle ceiling is real but lab-flattered. The non-learned repair we tuned does nothing
  useful, so the question of a learned encoder stays open and is not answered by this page.
advice:
  - >-
    Do not use total-variation denoising of the 14-day residual to repair shifted lags, because it
    kept 0 to 2% of the oracle gain on the new generator worlds (rule (b) section).
  - >-
    Score any repair against both oracles and the zero-lag control on rows whose lag positions are
    touched, not on all rows, because the oracle gain on all rows is diluted by 70% of rows that
    no shift touches (oracle gain table).
  - >-
    Report a repair's effect on rows with no shift touching a lag, because an oracle trained on
    exact lags was 0.09 points worse there at the six (clean-row table).
  - >-
    Before building a learned history encoder, run the noise curve on the new generator worlds, as
    the plan describes, because this page measured only the ceiling and the non-learned repair.
code: >-
  ml-experiments/00507-history-encoder-first-test-oracle-repair-and-total-variation/
---

**A perfect (oracle) repair of shifted lag readings helps the forecast, but a cheap non-learned
repair gets almost none of that help.** We asked whether the power readings of the last 14 days
need repairing, or whether down-weighting them (as experiments 402 to 482 did) is enough. We
injected shifts into the readings (temporary load transfers, and genuine level changes that move
the true power), retrained the lag correction of experiment 472, and scored it with each lag set
repaired in turn: by an oracle that knows the injected shift, and by total-variation denoising of
the 14-day residual. The oracle improved the score on rows where a shift touches a lag. The
denoising improved it by at most 2% of the oracle's gain. The rule that stops the line at the
non-learned repair therefore did not fire, and the learned encoder (rung T1) was not built or
fitted in this experiment.

## Results

All scores are CRPS in points of capacity (lower is better), on the four screen folds
(2026-06, 2026-01, 2025-10, 2026-03), three fit seeds, held-out injected worlds 200 to 202. The
lagless forecast is ERA5 and CAMS for both history and forecast, one seed. Intervals are exact
month-block resamples of the four fold means.

![Change in CRPS against the plain correction on rows where a shift touches a lag](figure_1_gain_over_plain.svg)

| Rows and worlds | Group | Oracle minus plain | Total-variation minus plain |
|---|---|---|---|
| Injected harm and target rows, old worlds | six | -0.214 (-0.314 to -0.132), MDE 0.014 | not run |
| Injected harm and target rows, old worlds | eight | -0.466 (-0.581 to -0.309), MDE 0.223 | not run |
| Rows where a shift touches a lag, new worlds | six | -0.589 (-0.674 to -0.505), MDE 0.089 | +0.045 (-0.006 to +0.139) |
| Rows where a shift touches a lag, new worlds | eight | -0.727 (-0.793 to -0.661), MDE 0.088 | -0.012 (-0.018 to -0.005) |

- **Rule (a) passed.** At the six, the oracle gain on the old worlds was 15 times the minimum
  detectable effect, with an interval excluding zero. The same held at the eight (2.1 times).
  The zero-lag control (experiment 472's lag-free correction) was worse than plain by 0.49 (six) and
  0.91 (eight) points, so the lag readings carry information that repair can recover.
- **Prediction P1 was met only partly.** The plan predicted an oracle gain of at least 0.3 points.
  The old worlds gave 0.21 at the six (below) and 0.47 at the eight. The new worlds gave 0.59 and
  0.73.
- **Oracle-constant** (a single mean level removed per temporary shift) gained 0.57 (six) and 0.68
  (eight), so most of the oracle gain needs only the right level, not the right shape.
- **Rule (b) fired.** Total-variation repair, tuned on a separate world, kept 0 to 2% of the
  oracle gain. The paired difference between the repair's loss and the halfway point to the oracle
  was +0.34 (0.25 to 0.46) at the six and +0.35 (0.31 to 0.39) at the eight, where a negative value
  would have meant that the repair keeps more than half. Re-weighting the rows so that 60% of the
  steps are genuine level changes (the generator makes 79%) gave +0.37 and +0.38, so the conclusion
  does not depend on that share. The repair was not run on the old worlds, because failing on the
  new worlds already fails the rule.
- **Not shown worse on rows with no shift touching a lag.** The repair's point estimates there
  were +0.028 CRPS points at the six and 0.000 at the eight, below the 0.03 bound, and +0.011 and
  +0.000 on q99 pinball loss, below the 0.02 bound. The oracle, retrained on exact lags, was
  0.09 points worse than plain on those rows at the six (+0.016 to +0.185), probably because it
  saw no shifted readings in training.

![Mean absolute error of the repaired lags against the exact lags, for each total-variation setting](figure_2_tv_tuning.svg)

## Why the repair failed

- **The tuning picked the weakest repair.** On the tuning world (origins in 2024, no scored month),
  all 30 settings (6 weights times 5 values of tau) gave a larger mean lag error than no repair at
  all (0.124 spreads). The best setting (weight 16, tau 5) gave 0.127, so it repaired almost nothing.
  The mean lag error is dominated by the 68% of windows with no shift touching a lag, where any
  repair adds noise.
- **A repair tuned on the lag error of all windows may under-repair.** We did not run a setting
  tuned only on windows with a temporary shift in force, because the plan fixed the tuning
  criterion before any score was read. A non-learned repair that does better is not ruled out.

## What we did not do, and what limits the result

- **No encoder was fitted and the noise curve was not run.** Both belong to rungs after rule (b),
  which stopped the plan at the non-learned repair. The trained-encoder question stays open.
- **The lab setting flatters detection.** ERA5 knows the weather that happened, so the lagless
  surprise is smaller than at serving. A single-seed null is informative only when far from the
  minimum detectable effect; the repair's null at the six (+0.045, MDE 0.089) is not far from it.
- **The NGED event labels are incomplete,** so "rows with no labelled event" may hold unlabelled
  events, and counts of events are lower bounds.
- **Code review 1 found 12 problems before any number was read;** nine were fixed (the oracle now
  uses the cleaned readings, window events are a function of seed, Primary, and day, the size cap
  applies to the final step, the circuit fraction is drawn, and the self-test tolerance is
  relative). Unfixed: the oracle and plain fit seeds are not paired by world, and the row count was
  not asserted against experiment 404's scoring rows.
- **The generator makes 79% of the steps in force genuine, not the planned 60%.** This follows
  from the shares and durations chosen (self-test: derived 78%, simulated 79%), and the 60%
  re-weighting above shows it does not matter here.
- **N0.** The leak checks and month-boundary check are in `work/n0/` (private). The 34 events with
  a measured deviation: a median coefficient of variation of 0.44, 29 of the 34 with a correlation
  above 0.5 with the Primary's own daily demand shape, so a flat step is a poor description of most
  real events; the generator draws shapes in the planned thirds regardless.

## Next

The repair question is open, not closed: the oracle gain is large, and the one non-learned repair
we tried fails. A learned encoder is the obvious next test, with its error placed on the oracle's
noise curve first. This page does not commit the project to that experiment.
