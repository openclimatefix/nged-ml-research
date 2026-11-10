---
id: "00690"
slug: slow-level-state-on-primaries-13-and-16-july-august
title: >-
  Does the slow-level state track the July to August level change at Primaries 13 and 16?
theme: lsc-detectors
also_relevant_to: []
status: finished
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private tables under
  data/ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august/ only.
  (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: one defensible
  design, the masked frozen seasonal slow level of the edge-state addendum, as coded in experiment
  653. (5) Callers: this folder's scripts, which import the folders of experiments 653 and 685;
  nothing imports them. CPU only, 2 cores.
hypothesis: >-
  The slow level of the edge-state addendum (a smooth seasonal level fitted with candidate windows
  masked, then frozen) follows the July to August 2026 change in level that the stored
  normal-running forecasts miss at Primaries 13 and 16.
prediction: >-
  At both ids 13 and 16 the level estimate crosses half of the July to August residual change
  within 14 days of the residual's own half-crossing, and adding it to q50 cuts the median absolute
  error of July to September by at least 25%, while the two control Primaries move by less than 5%.
  Expected to fail, because the design masks long deviations on purpose.
decision_rule: >-
  Supported only if both ids meet both conditions and both controls move by less than 5%. If either
  id misses, the slow-level state does not explain the miss and the miss stays recorded as
  unexplained; no new method is proposed.
benchmark: >-
  The stored normal-running q50 of the one-year backtest (primaries.parquet, NRA rows), with no
  correction.
row_set_id: >-
  one-year backtest normal-running rows of primaries.parquet, one row per half-hour (the row with
  the smallest lead), 2026-05-01 to 2026-09-30; not comparable with the leaderboard
metric: >-
  Weekly median of observed power minus q50; correlation and lag of the weekly level estimate with
  it; days from the start to half of the change; median absolute error and q05 to q95 coverage in
  July to September with and without the level added.
code: ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august/
verdict: partly supported
best_arm: >-
  Masked frozen slow level (the main arm): it crossed half of the change within 3 days of the
  residual's own crossing at ids 13 and 16 and cut the median absolute error by 50.3% (95%
  interval -59.9 to -34.8) at id 13 and 51.3% (-55.8 to -35.3) at id 16, in hindsight.
headline: >-
  The slow-level state follows the July to August level change at Primaries 13 and 16 (weekly
  correlation 0.66 and 0.87), but it spreads the change over months, and a control Primary moved
  by 5.5% and the other by 7.5%, above the 5% pre-registered.
lessons: >-
  A smooth level fitted in hindsight on a related residual follows most of the level miss at these
  two Primaries, so the miss behaves like a slow drift rather than a step that needs an event. The
  level estimate rises from May at Primary 13 and so passes half of its own change before July,
  which makes the 14-day timing test easy to pass. The added level is a hindsight diagnostic and
  never a forecast.
take_home_message: >-
  The level miss at Primaries 13 and 16 in July and August is tracked by a smooth slow level: adding
  it to q50 halves the median absolute error at both Primaries, in hindsight and not as a forecast.
  The same level moves the two control Primaries by 5.5% and 7.5%, above the 5% limit set in
  advance, so the verdict is partly supported.
advice:
  - >-
    Record the level miss at ids 13 and 16 as tracked in hindsight by a slow level, not as
    explained, because the level change is half the residual change and NGED's event labels are
    incomplete.
  - >-
    Do not read the 14-day timing pass as proof of a step: the level estimate ramps over months
    (see the figure of the two target Primaries).
production_status: not-for-production
adopted_in: none
---

**The slow-level state of the edge-state addendum tracks the July to August level change at
Primaries 13 and 16, so the miss is a slow drift.** The stored normal-running forecasts miss the
observed power by +0.84 MW at Primary 13 (D18, mean power 2.1 MW before the change) and by -1.09 MW
at Primary 16 (D14, 5.9 MW). The masked frozen level crossed half of that change 3 days before (id
13) and 2 days before (id 16) the residual's own crossing, and adding it to q50 halves the median
absolute error at both Primaries. The level moved the median absolute error of two control Primaries
by 5.5% (a fall, with an interval that includes zero) and 7.5% (a rise), which is above the
5% limit, so the verdict is *partly supported*. The stop rule did not fire, because neither id
missed.

**What it narrows.** The level miss is a gradual drift that a smooth level absorbs, not an
unexplained step. This is a hindsight diagnostic on a lab setting: the level was fitted on the whole
record, including the window, from a different forecast's residual. July to September 2026 lies in
the final-test window and is read descriptively only; no choice is made from it.

![Weekly median of observed power minus the stored q50, with the weekly 5th to 95th percentile.](figure_miss.svg)

![The 7-day mean of the stored forecast's miss and the three slow-level estimates at the two target Primaries.](figure_level_targets.svg)

![The residual and the three slow-level estimates at the two control Primaries.](figure_level_controls.svg)

![Change in median absolute error from adding the level to q50, with 95% intervals.](figure_error_change.svg)

## Results

- **Arms and data.** Arm 1: observed `power` minus stored `q50` on normal-running rows (one row per
  half-hour, the smallest lead). Arm 2: `fit_slow` of experiment 653 (masked, frozen) on experiment
  685's lag-free residual panel (ERA5 and CAMS weather at the target time, so a lab setting), noise
  units frozen on the residual before 2026-07-01. S1 (Huber, no mask) and S0 (30-day random walk)
  are the addendum's other arms, reported beside it. No forecast model was fitted, so no retraining
schedule applies.
- **Controls, by the rule fixed in the plan:** ids 26 and 17 (stored q05 to q95 coverage 0.89 and
  0.91 in July, 0.91 and 0.86 in August). Only the 15 panel ids were candidates, because the state
  needs the panel.
- **Timing (main arm).** Half of the change is passed at day 0 (id 13) and day 3 (id 16), counted
  from 2026-07-01, against day 3 and day 5 for the residual. At id 13 the level estimate rises from
  May, so the crossing is early. Level change is +0.41 MW at id 13 (residual +0.84) and -0.64 MW at
  id 16 (residual -1.09).
- **Agreement.** Weekly correlation of level with residual: 0.66 at id 13 and 0.87 at id 16 (S0:
  0.89 and 0.90). Best daily lag: 28 days at id 13 (the edge of the search) and 6 days at id 16.
- **Median absolute error, July to September, level added (main arm).** Id 13: 0.78 to 0.39 MW,
  -50.3% (95% day-block interval -59.9 to -34.8). Id 16: 0.83 to 0.40 MW, -51.3% (-55.8 to -35.3).
  Controls: id 26 -5.5% (-9.9 to +1.3); id 17 +7.5% (+1.6 to +14.0).
- **q05 to q95 coverage, same rows.** Id 13: 36% to 51%. Id 16: 49% to 66%. Control 26: 85% to 86%.
  Control 17: 90% to 87%. The level does not restore 90% coverage.
- **Variants.** The level change is half the residual change at id 13 (+0.41 against +0.84 MW), and
  the mean level added there is +0.70 MW, so an offset between the two forecasts enters. The post
  hoc variant that adds only the change from the 2026-05-01 to 2026-06-14
  mean gives -44.0% and -55.1% at ids 13 and 16 and +8.5% and -1.2% at the controls, with coverage
50% and 70% at ids 13 and 16. The random-walk
  arm S0 gives -52.9% and -59.9%, and +0.7% and +14.4% at the controls.
- **Caveats.** The level is a smooth spline, so it ramps over months and misses the step on
  2026-07-01 and the fall from late August. The residual panel belongs to a different forecast from
  the stored one, so a constant offset between them enters the "level" variant. Control crossing
  days are not meaningful because their changes are under 0.4 MW. NGED's event labels are an
  incomplete dump, so "no transfer" in experiment 685 is a lower bound. The 5% limit on the controls
  was met by neither control, so the main arm fails the pre-registered test on that count.
- **Checks.** No new forecast was scored, so the median-minus-mean and mean-forecast audit checks do
  not apply, and no audit trigger fired.
- **What to try next.** Fit the level causally (data before each origin only) to see whether it
  could be used live; this is a proposal for the maintainer.

## Method

The code is in `ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august/`
(`e00690_run.py`, `e00690_figures.py`); it imports the folders of experiments 653 and 685.
Intermediate tables are in the experiment's data folder (private, not committed): `weekly_miss.csv`,
`daily_level_and_residual.parquet`, `results.json`, and `control_choice.csv`. Bootstrap: 7-day
blocks, 2,000 draws, paired. Review: one fresh Sonnet agent reviewed the code before the run.
