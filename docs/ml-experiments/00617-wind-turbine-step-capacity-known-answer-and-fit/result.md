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
  Daily-envelope hidden Markov model with a daily failure probability of 0.001; it detects 31% of
  injected outages of 2 or more turbines lasting 14 days or more at W1, against a bar of 80%.
headline: >-
  On injected outages the estimator detects 31% (W1), 32% (W2), and 14% (W3) of outages of 2 or more
  turbines lasting 14 days or more, and raises 3.5 false steps a year at W1 and W2 (2.4 at W3) on
  permuted history, against a bar of 1. The W1 recall stop rule (under 50%) and the false-step stop
  rule (over 3 a year at W1) both fired, so the real-history fit was not run.
lessons:
  - >-
    The limit is the number of informative days, not the model: only 21% of days at W1, 20% at W2,
    and 11% at W3 have 6 or more half-hours in the saturation range, and recall reaches 64% (2
    turbines) or 74% (3 turbines) at W1 only for 60-day outages.
  - >-
    Knowing the turbine count from a public register did not measurably raise recall at W1 (34%
    with the count fixed, 33% inferred; 3 runs per cell, so the 1-point gap is noise).
advice:
  - >-
    Do not fit a turbine-step capacity from the saturation-speed daily envelope of W1 to W3; it
    detects under half of 14-day outages and raises false steps (figures 1 and 4).
  - >-
    Before trying again, look for a signal observed on most days, such as a turbine availability
    log from NGED, because ERA5 saturation-speed days are too few (Results, recall).
  - >-
    Expect no help from knowing the turbine count: recall with the count fixed matched recall with
    the count inferred (Results, register).
take_home_message: >-
  A hidden Markov model that looks for whole-turbine drops in the daily top of wind-farm power
  cannot reliably find outages shorter than about two months, and it raises 2 to 4 false alarms a
  year on data that hold no outage. The cause is too few days with wind strong enough to show full
  output, not the model, so the effective capacity of the metered wind farms cannot be tracked this
  way.
production_status: not-for-production
adopted_in: none
---
**A hidden Markov model that looks for whole-turbine outages in the daily top of wind-farm power
finds 14-day outages of 2 or more turbines only about a third of the time at W1 and W2, and it
raises 2 to 4 false steps a year on data that hold no outage.** The cause is the data. Only about
20% of days at W1 and W2 have 6 or more half-hours with wind strong enough for the farm to reach
full output, so a 14-day outage is often seen on one or two days. A signal that is observed on most
days is therefore needed to track turbine-step capacity. The test injected artificial outages into
the real daily series and checked whether the model found them (a known-answer test). The fit to the
real histories (part B of the plan) was not run, because the stop rules fixed in advance both fired.
Nothing is adopted, and no data from July 2026 onwards is used.

**Terms.** The model's hidden state is the number of turbines out of service. Its daily input is
the 90th percentile of power over the half-hours that count as informative. A half-hour is
informative when the wind speed from ERA5 (the European Centre for Medium-Range Weather Forecasts'
reanalysis) at 100 m is above the rated speed plus the blur, and at most 22 m/s (below the cut-out
speed minus 3 m/s). The rated speed, the blur, and the cut-out speed come from the plant summary of
[experiment 347](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/result/).
The blur is the width of the smoothing that experiment 347 fitted to the wind speed. The power
series comes from
[experiment 453](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00453-effective-capacity-upper-bound-data-look/result/).
"Recall" is the share of injected outages whose recovered step is at least half the injected step.
A "false step" is a rise of one or more turbines out that lasts 3 days or more.

## Figures

![Recall by outage length](figure_1_recall_by_duration.svg)

*Figure 1. Share of injected outages detected, by outage length. The dashed line is the 80% bar. No
farm reaches it in either panel; W1 reaches 64% (2 turbines) and 74% (3 turbines) at 60 days.*

![Recovered step by outage length](figure_3_recovered_vs_injected.svg)

*Figure 3. Recovered step size, in turbines, of detected outages with 2 or 3 turbines injected.
Boxes span the 25th to 75th percentile, the white line is the median, and the whiskers are the
minimum and maximum. The dashed line is the injected size. Boxes are missing where no outage of
that length was detected.*

![False steps on permuted history](figure_4_false_steps.svg)

*Figure 4. False steps a year on permuted history, by daily failure probability. The dashed lines
mark the pass bar (1 a year) and the stop rule (3 a year). W1 and W2 are above the stop rule at
every probability; W3 is below it at 0.001.*

![One known-answer run at W1](figure_5_example_run.svg)

*Figure 5. One run at W1: six injected 60-day outages of 3 turbines (black), the recovered
deficit (blue line), and its 90% interval (light band). Under the detection rule (a step of at
least half the injected size), 1 of the 6 outages is recovered; several outages are invented.*

## Results

- **Design.** 405 known-answer cells (3 farms, 3 failure probabilities, 3 turbine counts of 6, 10,
  and 14, 3 outage sizes, 5 lengths), 3 runs each, up to 15 outages a run. The events table holds
  17,496 outages, because long outages leave room for fewer than 15 in the record. The failure
  probability a was to be chosen as the one with false steps under 1 a year and the best recall;
  none qualified, so the smallest (0.001) was used.
- **Recall, 2 or more turbines, 14 days or more (bar 80%).** W1 31%, W2 32%, W3 14%. Counting only
  outages with an informative day inside them, W1 44%, W2 50%, W3 23%; 71%, 65%, and 60% of
  outages have such a day. At 60 days, recall at W2 is 52% and 51% (2 and 3 turbines) and at W3 31%
  and 20%.
- **False steps (bar 1 a year, stop rule 3 a year).** At a=0.001 on days permuted at random: 3.5
  (W1), 3.5 (W2), and 2.4 (W3) a year. At a=0.01: 5.8, 6.0, and 4.3. The drift placebo raises 3.7
  to 4.1 steps a year at a=0.001, and the estimated final capacity is 10% to 11% above the true
  value (bar 3%).
- **Step size.** Of detected outages, 66% (W1), 67% (W2), and 75% (W3) are within half a turbine of
  the injected size. The recovered step is 0.24 to 0.36 turbines too small on average.
- **Turbine count.** The count is recovered within 15% in 10% (W1), 32% (W2), and 10% (W3) of runs
  (bar 70%); only W2 with 14 true turbines passes, at 71%. The median estimate is 16 at W1 for
  every true count, 13 to 14.5 at W2, and 11.6 at W3, so the data do not identify the count.
- **Interval coverage.** The 90% interval covers the true deficit on 27% to 29% of outage days and
  on 59% to 84% of all days (bar 85% to 95%).
- **Register and fixed count.** The public Renewable Energy Planning Database matches W1 and W3 by
  position and capacity, and W2 ambiguously. The counts stay in the private data folder. At W1,
  recall for 2 and 3 turbines at lengths 14 to 60 days was 34% with the count fixed and 33% with the
  count inferred, so knowing the count did not help.
- **Stop rules.** W1 recall is under 50% and W1 false steps exceed 3 a year, so the wind line stops
  here. The real fit, and the check against the ensemble analysis proxy (the ECMWF ensemble
  analysis that the forecaster reads), were not run.

## Caveats

- **The known-answer test is easier than the real problem in four ways.** All outages in one run
  have one size; real outages of about 10% or more were removed from the base series; the ceiling
  (the full-farm output) is estimated from the series with the injected outages; and the detection
  rule accepts a recovered step of half the injected size. The permutation placebo uses the
  unfiltered series, which is the harsher null.
- **The tuning is on the real series.** The failure probability was chosen on the same days that
  part B would fit.
- **Turbine counts of 6, 10, and 14** replace the plan's turbine sizes of 1.5 to 3.5 MW, which do not
  fit the two smaller farms.
- **Missing real events.** NGED's logs may omit real outages, so the base series may hold unlabelled
  outages.

## Checks

- A fresh Sonnet reviewer read the code before the run. The ceiling, the placebo base, step
  counting, block permutation, the window off-by-one, and coverage on observed days were fixed;
  same-size outages and the detection rule are recorded above as caveats.
- A fresh Sonnet reviewer then checked the numbers, figures, and prose; its findings were applied.
- Code: `ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit/`. Private outputs:
  `data/ml-experiments/00617-wind-turbine-step-capacity-known-answer-and-fit/`.
