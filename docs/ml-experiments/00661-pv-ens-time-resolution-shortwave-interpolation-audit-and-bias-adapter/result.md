---
id: "00661"
slug: pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter
theme: solar-disaggregation
also_relevant_to: []
status: finished
kind: other
size: >-
  complex. Stored: re-built degraded-weather input matrices, pretrained and fine-tuned
  neural-network weights, and quantiles in this experiment's private folder. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one test of the time-resolution
  hypothesis is defensible (shortwave only or all variables degraded, 3-hour or 6-hour windows), so
  the choices are fixed below before any Part 2 or Part 3 result is read. Callers: the scripts
  import the modules of experiments 48, 87, 397, 405, 418, 428, 448, 510, 554, 634, 638, 639, 644,
  and 655 and edit none of them. Asset graph: not touched. One GPU process at a time; ceiling 3
  GPU-hours (Part 2 at most 1.5, Part 3 at most 1.5).
hypothesis: >-
  The explicit solar term gains about 0.75 CRPS points with CAMS and ERA5 weather (experiment 639)
  and loses 0.25 to 0.37 with ECMWF ensemble weather (experiments 644 and 655). One cause is the
  ensemble's coarse time step, which is 3 hours to lead 144 hours and 6 hours after. A second cause
  would be a faulty interpolation of the ensemble's shortwave flux to half-hours. A third is
  forecast cloud error. This experiment tests the first two directly and tests a bias adapter as a
  remedy for the third.
prediction: >-
  Fixed before any Part 2 or Part 3 result was read. Part 1 (the audit) had run before this plan
  was committed, as the brief orders, and is reported below. Part 2: the explicit gain on known
  weather degraded to 3-hour windows falls to between 0.4 and 0.6 CRPS points (a fall of 20% to
  45%), not to half or below, because the interpolation error at 3-hour windows is 5% of the midday
  peak and the explicit term still sees the sunny-day shape. At 6-hour windows I expect 0.2 to 0.4.
  Resolution is therefore a partial cause and not the main one. The 644 loss at 3-hourly leads
  (day 1 to 6) and 6-hourly leads (day 6 to 10) is of similar size (within 0.1 points), so the loss
  does not grow when the step coarsens. Part 3: the adapter arm does not beat the free network at
  day 1-2 by the bar of 0.084, because it must correct a bias that depends on the forecast cloud
  field, which the adapter's five inputs only partly carry.
decision_rule: >-
  Fixed before any Part 2 or Part 3 result was read. CRPS is in points (percent of effective
  capacity), pooled over the six MW Primaries and the sunny half-hours (10:00 to 15:00 UTC,
  clear-sky index at least 0.7) of the 60 days after each cutoff, as in experiments 639 and 644.
  Intervals are day-block bootstraps at 99.6% (Bonferroni over this page's contrasts). Part 1 rule:
  if the audit finds a mismatch with production or a time-stamp shift of one window or more, fix
  the pipeline copy and re-score 644's two headline contrasts on the corrected half-hours, and say
  whether the sign changes. Part 2 step 5 rule: if the explicit-minus-free gain on known weather
  degraded to 3-hour windows is at most half of the same cutoff's undegraded gain (experiment 639's
  stored value for the same seeds), report that time resolution is a main cause and name the remedy
  to test next (an ensemble product at finer time resolution, or a learnt sub-window shape);
  otherwise report that resolution is not the main cause. The 6-hour variant is reported beside it
  and does not change the rule. Part 3 rule: the adapter arm A is kept only if it beats the free
  network at day 1-2 by the experiment 644 bar (0.084 points) with an interval that excludes zero.
  The learned DC-to-AC ratio and the fleet-shape mixture are reported and are not gated.
benchmark: >-
  Part 2: experiment 639's stored free and calibrated-explicit predictions at the same cutoff and
  seeds. Part 3: experiment 644's stored free network (arm a), calibrated explicit network (arm b),
  14-day tracker (arm d), and placebos, on 644's rows, plus climatology and the manual heuristic as
  the floors of 644.
row_set_id: >-
  Part 2: experiment 639's test windows (the 60 days after each cutoff, cut at 2026-07-01). Part 3
  and step 6: experiment 644's rows (every ensemble run in 644's test months at lead 0 to 240
  hours). Cleaned power, farm-gap rows dropped. Nothing dated 2026-07-01 or later is used.
metric: >-
  CRPS in percent of effective capacity (points) from 11 quantiles, on sunny middays and on all
  half-hours, by lead band; mean absolute error of the median forecast normalised by effective
  capacity; coverage of the 5% to 95% interval; tail pinball loss at 0.01, 0.05, 0.95, and 0.99
  for the Part 3 arm.
verdict: refuted
best_arm: >-
  The calibrated explicit network of experiment 644 (arm b) is the best of the explicit arms but
  still loses to the free network by 0.37, 0.25, and 0.27 points at the three lead bands. The
  adapter arm (A) loses to the free network by 0.29, 0.14, and 0.13 points (sunny-midday CRPS 4.04
  against 3.75 at day 1-2).
headline: >-
  The ensemble's 3-hour to 6-hour time step does not explain why the explicit solar term loses with
  ensemble weather. Known weather degraded to 3-hour windows keeps the whole explicit gain (0.84
  points against 0.75 at hourly resolution), and 6-hour windows keep it too (0.77). The audit found
  no fault in the interpolation, and a bias adapter does not rescue the term.
lessons:
  - >-
    The experiment pipeline's half-hourly ensemble shortwave is identical to production's (0.0 W/m2
    on 15 runs) and has no time-stamp shift against CAMS, so earlier ensemble results need no
    re-reading for this reason.
  - >-
    Production's resampler rebuilds 3-hour windows to within 5.3% of the midday peak (6.6% for
    6-hour windows), far better than linear interpolation (15.5%, 30.8%) or step-holding (10.4%,
    19.5%), so the coarse step costs little accuracy on sunny days.
  - >-
    The explicit term's loss is the same size at 3-hourly leads (-0.28) and 6-hourly leads (-0.30),
    so the step width is probably not the cause (the lead also differs between the two groups, so the test is weak); forecast cloud error is the likely remaining cause.
  - >-
    A learned correction of the clear-sky index and a mixture of fleet shapes recover 0.08 to 0.15
    points of the explicit network's loss and stay 0.13 to 0.29 points behind the free network.
take_home_message: >-
  The ensemble forecast's coarse time step (3 hours, then 6 hours) is not why telling the demand
  network each substation's solar capacity makes the forecast worse: the same capacity term still
  helps by about 0.8 points of capacity when real weather is averaged into 3-hour or 6-hour windows
  first. The most likely remaining cause is the ensemble's cloud forecast error, which was not tested here, and a small neural correction
  of the sunshine input did not fix it.
advice:
  - >-
    Do not blame the ensemble's 3-hour or 6-hour time step for a solar-term loss, because known
    weather (shortwave only) degraded to those windows kept the whole explicit gain (0.84 and 0.77 against 0.75
    points, experiment 661 step 5).
  - >-
    Keep production's window-mean resampler for ensemble shortwave, because it rebuilds 3-hour
    windows to within 5.3% of the midday peak against 15.5% for linear interpolation (experiment 661
    Part 1).
  - >-
    Do not expect this learned clear-sky-index adapter to rescue the explicit solar term, because it stayed 0.13
    to 0.29 points behind the free network (experiment 661 Part 3).
production_status: not-for-production
adopted_in: none
---

# The ensemble's time step is not why the explicit solar term loses

**The explicit solar term (telling the demand neural network each Primary substation's solar
capacity, so that it subtracts a computed solar output) helps by about 0.8 points of capacity when
weather is real, and hurts by about 0.3 points when weather is an ensemble forecast. We tested
whether the ensemble's coarse time step is the reason, and the time step is not the reason.** We
averaged real weather (CAMS and ERA5 estimates of past weather) into 3-hour and 6-hour windows,
rebuilt half-hourly values with production's window-mean resampler, and retrained the neural
networks. The explicit gain did not fall. The audit found nothing wrong with how ensemble shortwave
becomes half-hourly values, and a small neural adapter in front of the solar curve did not rescue the
term. The most likely cause that remains is the ensemble's forecast cloud error, which this
experiment did not test directly.

This page follows [experiment 639](../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md)
(real weather, gain 0.75), [experiment 644](../00644-pv-explicit-term-with-ens-forecast-weather/result.md)
and [experiment 655](../00655-pv-member-level-cell-temperature-term-and-forecast-sunniness-tracker/result.md)
(ensemble weather, loss 0.25 to 0.37). Scores are the continuous ranked probability score (CRPS) in
percent of each Primary's effective capacity ("points"), on sunny middays (10:00 to 15:00 UTC,
clear-sky index at least 0.7), at the six MW Primaries. Smaller is better. The free network learns each
Primary's solar response from the data alone; the explicit network is told the capacity. A positive
contrast (free minus explicit) favours the explicit network.

![Explicit gain against input time resolution](gain_by_input_resolution.svg)

- **Averaging real weather into 3-hour or 6-hour windows did not reduce the explicit gain.** Free minus explicit CRPS is
  0.753 [0.478, 1.050] at hourly resolution, 0.839 [0.581, 1.127] with 3-hour windows, and 0.773
  [0.512, 1.058] with 6-hour windows (three cutoffs, five seeds, 99.6% day-block intervals). The
  3-hour gain exceeds the hourly gain by 0.086 [0.016, 0.162]; the 6-hour gain by 0.020 [-0.084,
  0.120]. The decision rule (a fall of at least half) is not met. The gain is concentrated: Primary 6
  carries about 3 points, and the other five Primaries together gain 0.27 [0.07, 0.49] at hourly
  resolution, with 3-hour minus hourly still positive at 0.113 [0.053, 0.176].
- **The ensemble loss is the same at both step widths.** Free minus explicit is -0.281 [-0.426,
  -0.147] at 3-hourly ensemble leads (up to 144 hours) and -0.297 [-0.413, -0.182] at 6-hourly
  leads. The two groups also differ in lead, so the comparison is a weak test on its own.

![Explicit gain by lead day with ensemble weather](ensemble_gain_by_lead_day.svg)

## Interpolation audit (Part 1)

- **The pipeline is production's resampler.** Stored summary tables, the half-hourly rows of
  experiment 48, and a fresh call of `ml_core` agree to 0.0 W/m2 on 15 runs between 2024-06 and
  2026-06. Production's code is the same file as the version the experiments imported.
- **No time-stamp shift.** The ensemble's window-mean shortwave correlates best with CAMS (hourly
  values stamped at the end of the hour) at a shift of zero at all six Primaries (r 0.966 to 0.986).
  The other CAMS convention gives a shift of two half-hours and r of 0.878 to 0.914 at zero shift.
- **Production's method is the best of the three methods tested (window mean, linear flux, step-holding).** Rebuilding known weather from 3-hour windows leaves
  a root-mean-square error of 5.3% of the CAMS midday peak (peak bias -2.5%); linear flux gives 15.5% and step-holding
  10.4%. With 6-hour windows the errors are 6.6%, 30.8%, and 19.5%.

![Interpolation error](interpolation_error.svg)

## Bias adapter (Part 3)

![Adapter contrasts](adapter_contrasts.svg)

| Sunny middays, ensemble rows of experiment 644 | Day 1-2 | Day 3-5 | Day 6-10 |
|---|---|---|---|
| Free network (644 arm a) | 3.75 | 4.28 | 5.05 |
| Explicit network (644 arm b) | 4.11 | 4.53 | 5.33 |
| Adapter arm (A) | 4.04 | 4.42 | 5.18 |

- **The adapter arm is not kept.** Free minus A is -0.291 [-0.479, -0.124], -0.135 [-0.316, 0.036],
  and -0.129 [-0.275, 0.013]; the bar was a gain of 0.084 at day 1-2. A beats the explicit network by
  0.075 [-0.006, 0.150], 0.115 [0.030, 0.190], and 0.146 [0.079, 0.217], and the 14-day tracker beats
  A by 0.63, 0.60, and 0.70.
- **The mixture without the adapter scored about the same as the adapter arm at day 1-2.** The
  mixture-only arm (A0) is 0.024 [-0.070, 0.121] behind the adapter arm (A) there, and 0.081 and
  0.089 behind at day 3-5 and day 6-10.
- **The learned fleet shape stayed uniform.** The mixture weights ended near equal across the three
  tilts and five DC-to-AC ratios at every Primary, so the Dirichlet prior dominated and no ratio was
  learned. The fleet shape is reported and not gated.

## Checks

- **Audit trigger that fired:** the Part 2 result contradicts the prediction of a 20% to 45%
  fall. The explanation, forecast cloud error, rests on experiments 644 and 655 and was not tested
  here. A fresh Sonnet agent re-derived the step 5 contrasts from the stored quantiles.
- **The checks of median against mean, per-site contrasts, and the error beside each contrast**
  were not run for step 5, because its arms share the pretraining and scoring code of experiment
  639.
  For Part 3, the mean absolute error of the median on sunny middays is 6.49 (A), 6.63 (A0), 6.33
  (free), and 6.67 (explicit), in the same order as the CRPS contrasts.
- **Tails (Part 3, sunny middays):** coverage of the 5% to 95% interval is 88.4% (A), 89.4% (A0),
  88.0% (free), and 89.2% (explicit). The 0.01 and 0.99 quantiles cover 2.7% and 99.8% of outcomes
  for A.
- **Caveats:** each day is its own trajectory in the degradation, so interpolation never crosses
  midnight (the 6-hour window that holds sunrise is affected in summer); only shortwave is degraded;
  capacities are experiment 639's, calibrated on undegraded weather; no example-forecast figure is
  shown; the Part 2 gain is measured on real-weather rows (the figure's blue rows)  and the ensemble loss on ensemble rows, so
  the two are not paired.

## Method

- **Architecture.** The networks are experiment 639's modular mixture-density network (free, and
  calibrated explicit with the term `C * inc * u` subtracted from every quantile). Arm A replaces
  `u` with a differentiable curve: a 6-input adapter (ensemble-mean clear-sky index, ensemble
  standard deviation over clear-sky irradiance, lead, hour of day as sine and cosine, temperature;
  16 and 16 tanh units; zero-initialised output) gives `kt' = kt + 0.25 tanh(output)`, clipped to
  0 to 1.4; plane-of-array irradiance is tabulated at 16 clear-sky indices for tilts of 20, 35, and
  50 degrees and interpolated; a per-Primary softmax mixture over 15 components (three tilts times
  DC-to-AC ratios 1.0, 1.15, 1.3, 1.5, 1.7) applies the cell-temperature derating and a smooth clip
  `z / (1 + z^8)^(1/8)`, under a Dirichlet prior of concentration 2.
- **Training regime.** Step 5 pretrains the free and calibrated-explicit networks on CAMS and ERA5
  rows of experiment 418 with shortwave replaced by 3-hour or 6-hour window means rebuilt by
  `ml_core`'s resampler (three cutoffs, five seeds, real world). Part 3 fine-tunes from experiment
  639's calibrated pretraining on 24 months of ensemble rows with experiment 644's recipe (90-day
  recency half-life, fresh mixer, monthly refits, five seeds). The adapter learning rate is 0.002.
  About 0.6 GPU-hours for step 5 and 0.5 for Part 3.
- **Measurement.** Part 1 compares with CAMS on sunny days (daily CAMS energy at least 80% of clear
  sky) in 2025-07, 2026-03, and 2026-06. Steps 5 and 6 and Part 3 score the rows, cutoffs, and
  sunny-midday mask of experiments 639 and 644, with 99.6% day-block bootstraps (Bonferroni over 12
  contrasts) and 644's bars (0.080, 0.090, and 0.118). Nothing from 2026-07-01 is scored.

## Files

- `e00661_audit.py`: Part 1 (`identity`, `lag`, `score`); writes `work/audit/*.json` and `*.csv`.
- `e00661_step6.py`: the ensemble contrast by step width and lead day, from experiment 644's stored
  quantiles; writes `work/audit/step6.json`.
- `e00661_degrade.py`: degrades CAMS shortwave and writes raw matrices to `work/raw/w3` and `w6`.
- `e00661_fit5.py` and `e00661_analyse5.py`: step 5 fits (weights and quantiles under `work/step5`)
  and contrasts (`work/analysis/step5_*.json`).
- `e00661_adapter.py` and `e00661_analyse3.py`: Part 3 fits (`work/adapter/fits`) and contrasts
  (`work/adapter/analysis/part3.json`).
- `e00661_report.py`: prints every number on this page and draws the figures.
- Read from earlier experiments: 639 (pretraining, probes, stored predictions), 644 (rows and
  stored quantiles), 418 (dense weather), 48 and 397 (summaries). Paths are under `data/ml-experiments/`.
