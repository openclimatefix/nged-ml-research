---
id: "00694"
slug: gas-engine-state-tracker-part-1
title: >-
  Can the on/off state of the gas engine at D8 be tracked from the substation's raw half-hourly
  reading alone, with the number of units and the unit size inferred?
theme: gross-demand-simulation
also_relevant_to:
  - lsc-detectors
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  anything else (new method, no new data). Stored: private tables and fitted parameters under
  data/ml-experiments/00694-gas-engine-state-tracker-part-1/ only; no forecasts. More than one
  design is defensible (difference pairing, a count hidden Markov model on differences, a grid
  filter on the level and the unit count), so all are arms. Callers: this folder's scripts only;
  they read `studies.power.scan_power`, experiment 194's stored `g2_window.parquet`, and
  experiment 196's stored residual panel for the host series.
hypothesis: >-
  The engine at D8 (3 identical units of about 2 MW) can be tracked by walking forwards through the
  raw half-hourly reading with a hidden-state model of the number of units running, with no demand
  forecast, no price, and no weather. D8's reading is apparent power (MVA, never negative), so a
  model that lets the reading fold, y = sqrt((L - k u)^2 + q^2) for demand level L, k units
  running, and unit size u, tracks the engine better than a model of signed steps.
prediction: >-
  Written after a short look at D8's differences and before any model exists. (P1) Fitted on D8's
  reading from 2023-09 to 2025-02 without the engine's own record, the folded grid filter (M2)
  selects K = 3 units and u between 1.7 and 2.3 MW. (P2) On the known answers, M2 reaches a
  half-hour F1 of at least 0.80 for "any unit running", with recall at least 0.80 at 2 or 3 units
  and at least 0.50 at 1 unit, precision at least 0.80, and a median switch-time error of at most 1
  half-hour, under both injection rules below. (P3) On half-hours with no injection, on D8 before
  the engine was connected and on each host series, M2 flags at most 3% of half-hours and at most
  one active day a month. (P4) M2 beats the pairing rule (M0) by at least 0.10 in F1. (P5) On real
  D8 data the folded filter predicts held-out readings better than the same filter without the
  fold. The pairing rule's 1.5 MVA threshold and experiment 196's 3% and one-day bars are
  re-used on purpose, because experiment 196 failed them, and P3 asks whether a tracker that scales
  its noise to each half-hour of the day passes where a residual detector did not.
decision_rule: >-
  Fixed before any score is read. Unsupervised fitting (K, u, switching probabilities) uses only
  the reading, never the truth, and every other hyperparameter is fixed in this plan, so no
  parameter is tuned on a known answer. Every segment is fitted on its own reading. Planned
  contrasts, each read as a difference of pooled half-hour F1 for "any unit running" from a
  cluster bootstrap (10,000 resamples) at 98.3% (Bonferroni for three): C1, M2 minus M0; C2, M2
  minus M1 (a count model on differences); C3, M2 with a 2-half-hour minimum dwell minus M2 with
  the geometric dwell the model has by default, read on F1. A cluster is one schedule-week of the
  injected schedule (7 per schedule), resampled jointly across all host segments, then host
  segments are resampled within each cluster. The pooled schedules are the engine's record and 4
  semi-Markov schedules built from it (below); the record alone is also reported. Smallest gain
  worth having for C1: +0.10 F1 with the interval above zero, because M2 adds a grid filter and
  about ten parameters to a threshold rule, and the inventory project would carry that only for a
  gain visible in a plot. C2 and C3 change one term of a model that exists, so their threshold is
  zero. "Active day" is a UTC day with at least 4 flagged half-hours. Verdict, in order: `invalid`
  if Phase A cannot reproduce experiment 194's counts exactly. `not supported` if P2 fails under
  either rule on recall at 2 or 3 units or on precision, or if two or more P3 entries fail, or if
  the false-alarm run (stop rule 2) fires. `supported` if P1 to P5 hold and C1 clears +0.10.
  `partly supported` if P2 and P3 hold and any of P1, P4, or P5 fails, or C1 is positive but under
  +0.10 (then `small gain` for C1 alone). `inconclusive` if M0 alone meets P2 and P3 (the simple
  rule is then the answer and C1 is undefined), or if M2's fitted u moves by more than 0.5 MW
  between 2023-09 to 2025-02 and 2025-03 to 2026-02 on D8. 2026-03 onward is descriptive, because
  the shortfall fades there. Stop rules. (1) Stop before Phase B if the rebuilt known-answer table
  does not have exactly 2,054 half-hours. (2) After the false-alarm run on the un-injected host
  segments and D8 before the engine, and before any contrast is scored, stop if M2 flags more than
  3% on 3 or more of the 8 hosts. (3) Stop at 2 hours of wall-clock compute.
benchmark: >-
  M0, the difference-pairing rule, on the same segments. Experiment 196's residual detector is a
  reference only, because its injections differ and it used a demand forecast.
row_set_id: none (states are scored on injected half-hours and on un-injected history; nothing is forecast)
metric: >-
  Half-hour F1, precision, and recall for "any unit running" and by units running; spell recall;
  switch-time error in half-hours; estimated K and u against the truth; share of un-injected
  half-hours flagged and active days a month; held-out log-likelihood of M2 with and without the
  fold on D8.
code: ml-experiments/00694-gas-engine-state-tracker-part-1/
---

## Plan

**This is the first experiment of an exploratory series on reading embedded generators out of a
substation's raw reading.** It has a data look (Phase A) and four small trackers (Phase B), all
judged on known answers. It forecasts nothing, so the leaderboard, the climatology row, the
normalised MAE, and the load-shift scenarios do not apply. The series' longer aim is an inventory
of the embedded generation on the GB grid that combines the asset registers (FiT, REPD, ECR) with
what substation power flow shows. The maintainer asked for a hidden semi-Markov model that walks
forwards through spikes of the half-hourly reading, with minimum-on and minimum-off priors as an
ambitious extra, and with no external signals.

## Step 0

- **Stored:** private parquet tables, fitted parameters as JSON, and figures; no predictions.
- **More than one defensible design:** yes, so the trackers are arms.
- **Callers:** this folder's scripts only.
- **Size:** anything else. Opus 5.5 reviewed the plan, a fresh Sonnet agent reviews the code
  before the first run, and a fresh Sonnet agent reviews the write-up and its prose. Audit
  triggers 1 and 7 are likely to fire, so an audit follows if either does.

## What earlier work and a first look showed

- **The engine has three units of about 2 MW.** Experiment 194's engine record ("Gas A") sits at
  0, about 2, 4, and 6 MW. The maintainer's recollection of three units holds, and 1 MW does not.
  Phase A must reproduce experiment 194's count of half-hours at 0, 1, 2, and 3 units exactly
  (1,704, 137, 133, and 80 of 2,054), and its spell count.
- **D8's reading is apparent power.** A first look found merged steps near 5 MVA from 2023-09,
  and falls that shrink as the level before them shrinks. A folded reading `|L - k u|` would do
  that and would hide the engine where demand is near 3 MW. Confounders (fewer units running at
  night) also fit, so Phase A includes a sign test: from a level below 3 MVA, an engine start
  should show as a rise.
- **Gas A's spells are short.** The median is 5 half-hours, so a minimum dwell is a contrast.
- **Experiment 196's failure shapes this plan.** Its residual detector over-flagged D8 from
  2025-03 and failed false alarms at D5, D10, and D11. The mechanism tested here is a noise scale
  per half-hour of the day, estimated causally, so an 11 kV solar step changes the noise and not
  the state.

## Phase A: the data look

**Phase A draws the raw reading and its differences in four eras and fits no model.** The eras are
pre-engine (2022-01 to 2023-06-12), net (2023-09 to 2025-02), net with 11 kV solar (2025-03-24 to
2026-02), and 2026-03 onward. The engine's own record covers 2023-07-03 to 2023-08-15.

1. **Figure 1:** D8's half-hourly reading, one 14-day panel per era, shared MVA axis.
2. **Figure 2:** histograms of the lag-1 difference and the lag-2 difference with log counts, one
   column per era; and merged steps (consecutive same-sign differences above 4 trailing spreads).
3. **Figure 3:** merged fall size against the level before it, binned, per era, with the curves
   `L - |L - k u|` for k = 1, 2, 3 at u = 2 MW.
4. **Figure 4:** the sign test (counts of rises and falls above 1.5 MVA from a level below 3 MVA
   and from 8 to 10 MVA, per era); the dwell time from a fall to the next rise; and rises and
   falls by hour of day.
5. **Figure 5:** the engine record's histogram of output levels only, labelled "Gas A". No time
   series of Gas A is plotted, because its dates and size could identify it beside the register.

## Phase B: four trackers on the raw reading

**Every arm reads only its own segment of the series.** Noise is scaled by a causal robust spread:
1.4826 times the median absolute lag-1 difference over the trailing 28 days, by half-hour of day,
rounded to one of 8 log-spaced levels. The spread says how much demand moves in a half-hour. It
predicts no level.

- **M0, difference pairing.** Merge consecutive same-sign differences above 4 spreads; open a
  spell at a merged fall of at least 1.5 MVA; close it at the next merged rise of at least half
  that fall. It gives a binary state only.
- **M1, count hidden Markov model on differences.** State k in 0 to K. The lag-1 difference is
  `-(k[t] - k[t-1]) u` plus Student-t noise (4 degrees of freedom) at the spread. Any jump in k is
  allowed, as experiment 196 advises. Forward-backward, O(T K^2).
- **M2, grid filter on (level, units).** L lies on a 0.1 MVA grid from 0 to 22 MVA. L is a random
  walk with Student-t steps (4 degrees of freedom) at the spread. k is a Markov chain with a full
  transition matrix. The reading is `sqrt((L - k u)^2 + q^2)` plus Student-t noise (4 degrees of
  freedom, scale 0.08 MVA), with q fixed at 0.14 MVA (experiment 196's reactive floor). The
  forward pass is one banded multiplication along L and one mix over k per half-hour: O(T K G),
  G = 221. The filtered state is the online result and the smoothed state comes from the backward
  pass. This is a hidden Markov model with an explicit level, in the line of
  [Hart (1992)](https://doi.org/10.1109/5.192069) and
  [Kolter and Jaakkola (2012)](https://proceedings.mlr.press/v22/zico12.html).
- **M2-dwell.** M2 with the state expanded to (k, age), where age is 1 or at least 2: after k
  changes, k cannot change again for one half-hour. This is the smallest explicit-duration model
  (in the sense of [Yu (2010)](https://doi.org/10.1016/j.artint.2009.11.011)). Longer minimum
  durations are a later experiment.
- **M2-signed (diagnostic only).** M2 with `L - k u` and no fold, for P5.
- **Fitting K and u, without truth.** Per segment, profile the likelihood over K in {0, 1, 2, 3}
  and u from 0.5 to 3.5 MW in steps of 0.25, with a fixed prior transition matrix (stay 0.9), pick
  by the Bayesian information criterion, then run 5 rounds of expectation-maximisation on the
  transition matrix. K = 0 is the answer "no engine", which is what keeps false alarms low. A
  half-hour is flagged when the smoothed posterior of "k at least 1" exceeds 0.5. Code is numpy,
  scipy, and Polars only.

## Known answers

**Two injection rules, both run everywhere.** The engine record is subtracted from a host's
reading through the meter in two ways. The folded rule is experiment 196's: real power is
`sqrt(y^2 - q^2)`, the output g is subtracted, and the reading is `sqrt((real - g)^2 + q^2)`. The
signed rule is `max(y - g, 0.01)`, which is the generative model of M1 and M2-signed. The folded
rule favours M2 by construction, so P2 and the contrasts must hold under both rules, and the
power-factor-0.95 variant of the folded rule is reported without gating.

- **K1 (D8 itself).** `|y - g|` on D8 in 2023-07-03 to 2023-08-15, where D8 includes the engine,
  using the folded rule only. The fold is imposed there, so K1 is reported, and does not decide.
- **Host segments.** Each segment is 28 days of pre-roll, 120 days of history, and the 44-day
  schedule injected from day 38 of the 120. Hosts: D3, D4, D11, D14, D16, D2, D10, and D8 before
  the engine (2022-01 to 2023-06 only). Each host has 3 segments before 2025-03-24 and 3 after
  2025-04-01, except D8. The engine's own record is one schedule.
- **Synthetic schedules.** 4 semi-Markov schedules of 44 days with K = 3, u = 2 MW, built by
  resampling the record's runs (a unit level and a duration) with replacement, seeds 0 to 3. 4
  reported-only schedules test whether K and u are recovered: u = 1 and u = 3 MW (seeds 4 and 5)
  and K = 1 and K = 2 (seeds 6 and 7).
- **False alarms.** The 28 days of pre-roll and the days before and after each injected window
  are un-injected, and are scored for flagged share and active days; 28 days of pre-roll are
  excluded from the segment's flagged-day count to avoid the spread's warm-up.
- **Real D8, descriptive and P5.** Monthly running share, K, and u per era beside experiment 194's
  left-tail share. P5: fit u, the transition matrix, and q's floor on odd months of 2023-09 to
  2025-02, then score the filtered one-step predictive log-likelihood of M2 and M2-signed on the
  even months, with parameters frozen.

**Standing practices.** Followed: cleaned data everywhere (with the count of dropped half-hours per
series), a draft pull request, fitted parameters saved. Not applicable: refit cadence, screening,
tail coverage, forecast plots, load-shift scenarios. The research default does not apply.

## Earlier advice

The earlier `advice` lists were filtered for engine, D8, and injection items; 37,705 lines are too
many to read in full. **Followed:** use the engine's record as a known answer only for the 44
days, and treat D8 as gross before 2023-08-15 (194); let k jump several units, fit noise per
regime, gate in the noisy regime, and gate on both pf 1 and the 0.90 to 0.95 variant (196); report
night rows (199); inject before trusting a detector (361). **Departed from:** the advice to stop
refining engine detectors (199), because a model of the folded reading is new; and the theme's
rule that an asset list starts from the register, because this experiment asks what the reading
alone shows.

## Files and figures

- `e00694_common.py`: paths, eras, spread, loaders, injections, and metrics.
- `e00694_trackers.py`: M0, M1, M2, M2-dwell, M2-signed.
- `e00694_look.py`: Phase A tables and figures 1 to 5.
- `e00694_run.py`: segments, fitting, false-alarm run, scoring, bootstrap, P5; writes
  `printed_report.txt`.
- `e00694_charts.py`: result figures: one injected segment with the arms' states; F1 per arm with
  intervals; flagged share per host; D8's inferred running share per month.

**Compute:** CPU only, about 5 CPU-minutes per fit of a segment, under 2 hours on 24 cores.

## What the next experiment should answer

Written after the results.
