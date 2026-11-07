---
id: "00617"
slug: wind-turbine-step-capacity-known-answer-and-fit
title: >-
  Can a hidden Markov model find whole-turbine capacity steps at the 3 metered wind farms, in a
  known-answer test and in the real histories?
theme: generator-forecasting
also_relevant_to: []
status: finished
kind: other
date: 2026-11-07
size: >-
  Medium. Answers to the five triggers: (1) it stores only private daily series and fitted
  posteriors; (2) more than one defensible design exists, and the plan in the data folder chose one;
  (3) it reuses the plant summary of experiment 347 and the series of experiment 453; (4) it touches
  no serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  The effective capacity of a metered wind farm moves in whole-turbine steps. A hidden Markov model
  on the daily upper envelope of power in the saturation range of the power curve recovers injected
  turbine outages of 2 or more turbines lasting 14 days or longer.
prediction: >-
  Written before any run. At W1 and W2, recall is at least 80% for 2-turbine outages of 14 days or
  more. At W3 recall is under 50% at 14 days, because W3 has about 43 informative half-hours a month.
decision_rule: >-
  Known-answer pass bars: recall at least 80% (outages of 2 or more turbines, 14 days or more, W1
  and W2); at most 1 false step a year on permuted history; recovered step within half a turbine in
  at least 80% of detected steps; recovered turbine size within 15% in at least 70% of runs; 90%
  interval covering the true capacity on 85% to 95% of injected days. Stop the wind line if W1
  recall is under 50% or false steps exceed 3 a year. The real fit runs only if W1 and W2 pass.
benchmark: >-
  Experiment 347's all-row daily-ratio test, which detected a one-turbine loss in 0 to 2% of blocks.
row_set_id: wind_saturation_rows_to_2026_06_30
metric: Recall, false steps a year, step-size error, turbine-size error, interval coverage.
verdict: refuted
best_arm: >-
  Daily-envelope hidden Markov model with a daily failure probability of 0.001; detects 31% of
  outages of 2 or more turbines lasting 14 days or more at W1, against a bar of 80%.
headline: >-
  On injected outages the estimator detects 31% (W1), 32% (W2), and 14% (W3) of outages of 2 or more
  turbines lasting 14 days or more, and raises 3.5 to 6 false steps a year on permuted history. Both
  stop rules fired, so the real-history fit was not run.
lessons:
  - >-
    The limit is the number of informative days, not the model: only about 20% of days at W1 and W2
    have 6 or more half-hours in the saturation range, and recall is 64% to 74% only for 60-day
    outages.
  - >-
    Knowing the turbine count from a public register did not raise recall at W1 (34% with the count
    fixed, 33% inferred).
advice:
  - >-
    Do not fit a turbine-step capacity from the saturation-speed daily envelope of W1 to W3; it
    detects under half of 14-day outages and raises false steps (figure 1, figure 4).
  - >-
    Before trying again, find a signal that is observed on most days, such as a turbine availability
    log from NGED, because ERA5 saturation-speed days are too few (informative days).
  - >-
    Expect no help from knowing the turbine count; the count-fixed arm scored the same as the
    inferred count (fixed rating).
take_home_message: >-
  A model that looks for whole-turbine drops in the daily top of wind-farm power cannot find
  outages shorter than about two months, and it raises 3 to 6 false alarms a year when the data
  hold no outage. The cause is too few windy-enough days, not the model, so the effective capacity
  of the metered wind farms cannot be tracked this way.
production_status: not-for-production
adopted_in: none
---
**The turbine-step estimator is too blunt to use: it finds 14-day outages of 2 or more turbines
only about a third of the time at W1 and W2, and it raises false steps every year.** The cause is the
data. Only about 20% of days at W1 and W2 have 6 or more half-hours with wind in the saturation range
of the power curve, so a 14-day outage is often seen on one or two days. This narrows the search:
turbine-step capacity needs a signal observed on most days, and a longer or better wind-speed record
cannot supply it. The real-history fit (part B) was not run, because both stop rules fired. Nothing
is adopted. July 2026 onwards is not used.

**Terms.** The hidden Markov model has one hidden state, the number of turbines out of service. Its
daily input is the 90th percentile of power over the half-hours whose ERA5 100 m wind speed lies
between the rated speed plus the blur and 22 m/s (at most the cut-out speed minus 3 m/s), from the
plant summary of
[experiment 347](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/result/)
and the series of
[experiment 453](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00453-effective-capacity-upper-bound-data-look/result/).
"Recall" is the share of injected outages whose recovered step is at least half the injected step.

## Figures

![Recall by outage length](figure_1_recall_by_duration.svg)

*Figure 1. Share of injected outages detected, by outage length. The dashed line is the 80% bar. No
farm reaches it at 2 turbines out; W1 reaches 64% (2 turbines) and 74% (3 turbines) only at 60 days.*

![Recovered step by outage length](figure_3_recovered_vs_injected.svg)

*Figure 3. Recovered step size, in turbines, of detected outages. A correct recovery equals the
number of turbines injected (2 or 3). Recovered steps sit below it.*

![False steps on permuted history](figure_4_false_steps.svg)

*Figure 4. False steps a year on permuted history, by daily failure probability. The dashed lines mark
the pass bar (1 a year) and the stop rule (3 a year). Every bar is above the stop rule at W1 and W2.*

![One known-answer run at W1](figure_5_example_run.svg)

*Figure 5. One run at W1: six injected 60-day outages of 3 turbines (orange), the recovered deficit
(blue line) and its 90% interval (light band). Two outages are recovered, four are missed, and
several outages are invented.*

## Results

- **Design.** 405 known-answer cells (3 farms, 3 failure probabilities, 3 turbine counts of 6, 10,
  and 14, 3 outage sizes, 5 lengths), 3 runs each of 15 outages, so 18,225 injected outages. The
  failure probability a is chosen as the one with false steps under 1 a year and the best recall;
  none qualified, so the smallest (0.001) was used.
- **Recall, 2 or more turbines, 14 days or more (bar 80%).** W1 31%, W2 32%, W3 14%. Counting only
  outages with an informative day inside them, W1 44%, W2 50%, W3 23%; 71%, 65%, and 60% of outages
  have such a day.
- **False steps (bar 1 a year, stop 3 a year).** Mean 3.5 (W1), 3.5 (W2), 2.4 (W3) a year at a=0.001
  on days permuted at random, rising to 5.8, 6.0, 4.3 at a=0.01. The drift placebo raises 3.7 to 4.1
  steps a year and ends 10% above truth at a=0.001 (error bar 3%).
- **Step size.** Of detected outages, 66% (W1), 67% (W2), and 75% (W3) are within half a turbine;
  the recovered step is 0.24 to 0.36 turbines too small on average.
- **Turbine count.** The count is recovered within 15% in 10% (W1), 32% (W2), and 10% (W3) of runs
  (bar 70%). The median estimate is 13 to 16 turbines at W1 and W2 and 11.6 at W3 for every true
  count, so the data do not identify it.
- **Interval coverage.** The 90% interval covers the true deficit on 28% of outage days and 59% to
  84% of all days (bar 85% to 95%).
- **Register and fixed count.** The public Renewable Energy Planning Database matches W1 and W3 by
  position and capacity, and W2 ambiguously. The counts stay in the private data folder. At W1,
  with the count fixed, recall (2 or 3 turbines, 14 to 60 days) was 34%, against 33% with the count
  inferred, so the count is not the limit.
- **Stop rules.** W1 recall is below 50% and false steps exceed 3 a year, so the wind line stops
  here. The real fit, and the check against the ensemble analysis proxy, were not run.

## Caveats

- **The test is kind in four ways.** All 15 outages in a run have one size; the base series has real
  outages of about 10% or more removed; the ceiling is known to the estimator up to injection; and
  the permutation placebo uses the uncleaned series, which is the harsher null.
- **The tuning is on the real series.** The failure probability was chosen on the same days that
  part B would fit.
- **Turbine counts of 6, 10, and 14** replace the plan's 1.5 to 3.5 MW turbines, which do not fit the
  two smaller farms.
- **Missing real events.** NGED's logs may omit real outages, so the base series may hold unlabelled
  ones.

## Checks

- A fresh Sonnet reviewer read the code before the run. Its findings on the ceiling, the placebo
  base, step counting, block permutation, the window off-by-one, and coverage on observed days were
  fixed; same-size events and the generous detection rule are recorded above as caveats.
- Code: `ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit/`. Private outputs:
  `data/ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit/`.
