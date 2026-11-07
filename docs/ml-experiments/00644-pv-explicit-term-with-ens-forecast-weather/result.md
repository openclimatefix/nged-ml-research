---
id: "00644"
slug: pv-explicit-term-with-ens-forecast-weather
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  complex. Stored: fine-tuned neural-network weights and test-window quantiles in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  way to feed the explicit solar term from ensemble weather is defensible (summary or members, and
  how the 14-day tracker is dated), so the choices are fixed below before any result is read.
  Callers: the scripts import the modules of experiments 510, 554, 634, 638, 639, and 643 and edit
  none of them. Asset graph: not touched. One GPU process; ceiling 4 GPU-hours.
hypothesis: >-
  The sunny-midday gain of experiment 639's explicit solar term over the free network survives when
  the weather inputs are ECMWF ensemble forecasts at lead 1 to 10 days instead of CAMS and ERA5,
  and it is more than the gain of a bias correction that uses recent errors.
prediction: >-
  Fixed before any result was read. The explicit network (arm b) beats the free network (arm a) on
  sunny middays at every lead band (day 1-2, 3-5, 6-10) by at least the bar, with a smaller gain at
  day 6-10 than at day 1-2 because forecast cloud error grows with lead and swamps the capacity
  term. My expectation is 0.3 to 0.6 CRPS points at day 1-2 and 0.1 to 0.3 at day 6-10. I expect
  the 14-day tracker (d) to stay competitive in the real months (experiment 643 found it 0.48
  points ahead), so (b) against the best of (d) and (e) is the contrast most likely to fail. The
  explicit network loses no more than 0.02 points over all half-hours. In the step world the
  explicit network beats the free network and the tracker after the step.
decision_rule: >-
  Fixed before any result was read. CRPS is in points: percent of effective capacity, pooled over
  the six MW Primaries, the three cutoffs (2025-07, 2026-03, 2026-06), and the sunny half-hours
  (10:00 to 15:00 UTC, clear-sky index at least 0.7) of the 60 days after each cutoff, averaged
  over seeds. A contrast is the control's CRPS minus the explicit network's, so a positive value
  favours the explicit network. Intervals are day-block bootstraps at 99% (Bonferroni-corrected
  over the planned contrasts), over seeds as the study skill requires. The bars are experiment
  639's minimum detectable effects, 0.067 (real months) and 0.181 (step world), because the added
  step is a capacity table that someone must update, so it needs a threshold above zero, and no
  larger production threshold is set here. Contrast (i): arm b against arm a, and against the best
  of d and e (the lower sunny-midday CRPS of the two, picked per lead band on the same rows), by
  lead band (day 1-2, 3-5, 6-10); a gain is an interval that excludes zero with a point estimate of
  at least the bar. Contrast (ii): over all half-hours, b minus a is no worse than +0.02. Contrast
  (iii): in the step world, b beats a and d after the step by at least the step-world bar.
  Placebos: capacity permuted across the six Primaries (scaled by 0.5 and 1.5 as experiment 643
  advised, beside the plain swap) and a shuffled-day PV term must each lose to b by at least the
  bar. Recommendation: if (i) holds at every lead band and (ii) holds, recommend b as a release
  candidate and name its cost (the capacity table and its update cadence). If (i) holds only at day
  1-2, say so, because NGED care about 1 to 10 days. If neither b nor d beats a by the bar, stop the
  explicit-PV line. If the capacity-swap placebo fails again, report that the gain does not depend
  on the right capacity.
benchmark: >-
  Arm a (free network, ensemble inputs) and the floors, climatology and the manual heuristic, scored
  on the same rows. Experiment 639's and 643's hindcast results are context only, since they use
  different weather inputs.
row_set_id: >-
  Experiment 639's test windows (60 days after 2025-07, 2026-03, and 2026-06; 30 days at 2026-06),
  with every ensemble run in those windows at lead 24 to 240 hours rather than the first run per
  valid time, so a row is one Primary, valid time, and run. Cleaned power, farm-gap rows dropped.
  Nothing dated 2026-07-01 or later is used.
metric: >-
  CRPS in points of effective capacity from the 11 quantiles of each mixture, on sunny middays and
  on all half-hours, by lead day; MAE of the median forecast; q05-q95 coverage; the
  avoidable-curtailment-cost proxy of earlier studies.
---

# Does the explicit solar term still help when the weather is a forecast?

*Plan, written before any code runs. The write-up replaces this page when the fits finish.*

We ask whether telling a demand neural network each Primary substation's solar capacity still lowers
its error when the weather inputs are ensemble forecasts. [Experiment
639](../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md) found a gain on sunny
middays with known weather (CAMS and ERA5). [Experiment
643](../00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations/result.md)
showed the gain beats two fixed bias corrections but not a correction from the previous 14 days'
errors. This experiment repeats the comparison with the weather NGED would have at forecast time.

## What the plan fixes

- **Weather inputs.** The cached ensemble tables from [experiment
  607](../00607-bakeoff-extend-cached-inputs-to-2026-10-31/result.md) hold, per run, Primary, and
  valid time, the ensemble mean and standard deviation of each weather column, not the 51 members.
  No new download is allowed, so the networks read these summaries, as the demand networks of
  experiment 397 do. The physics curve of the explicit term is computed from the ensemble-mean
  shortwave, and the brief's "sampled once per ensemble member and epoch" cannot be done. Instead,
  each epoch draws the capacity once per Primary from its probe spread (experiment 639's rule), and
  the ensemble standard deviation of shortwave enters the network through its spread columns. This
  is a stated departure from the brief. If the stored 51-member frames of 2025-07 onwards turn out
  to be cheap to read, a member-level check of arm b is added as a labelled extra.
- **Training.** Each network keeps experiment 639's and 643's ERA5/CAMS pretraining (stored
  weights are reused where present), then fine-tunes on the 24 months of ensemble rows before the
  cutoff (the "ft" stage of the modular network, experiment 510's recipe) with the same gross-target
  construction as experiment 639: the target is `y + C(t) * inc * u(t)`, and the same term is
  subtracted from every forecast quantile. The ensemble rows start in 2024-04, so the cutoff
  2025-07 fine-tunes on 15 months.
- **Arms.** (a) free; (b) explicit with the calibrated season-year capacity (experiment 639's
  `cal`); (c) explicit with the raw probed capacity (`raw`); (d) the 14-day tracker; (e) the
  experiment 643 bias-shifted free network (`fbias`); (f) the experiment 474 R4 table as a fixed
  capacity (`r4`); (g) climatology and the manual heuristic. Placebos: `perm` (plus the 0.5 and
  1.5 scalings) and `shuf`.
- **The 14-day tracker.** Arm d is arm a's forecast plus the mean error of arm a's own day-1 and
  day-2 forecasts over the sunny middays of the 14 days before the run's origin
  (`power_fcst_init_time`, run time plus 9 hours). The error history uses only forecasts and metered
  power available at the origin. It adds no fit, needs no capacity, and is defined here before any
  ensemble-weather score is read.
- **Scoring.** Sunny middays and all half-hours, by lead day 1 to 10 (each reported), in the real
  months and in the step world (experiment 638's ordinary-spring step); CRPS, MAE of the median,
  q05-q95 coverage, and the curtailment-cost proxy; five seeds with intervals; leave-one-Primary-out
  for every contrast, because experiment 643's gain rested on one Primary.
- **Compute.** About 60 pretrained networks already exist (experiments 639 and 643). The new work
  is fine-tuning: seven fit arms (a, b, c, e, f and the two placebos), five seeds, three cutoffs,
  two worlds. A smoke fit fixes the time per fine-tune before the batch starts. If the projection
  exceeds 4 GPU-hours, stage 1 runs seeds 0 to 2 in the real months for all arms and in the step
  world for a, b, d, and e only; stage 2 adds seeds 3 and 4 as the ceiling allows, and the write-up
  states which cells ran.

## Standing practices

Follows: monthly refit cadence (a single refit per cutoff window), cleaned power everywhere, the
D15/D18 start cut (not relevant at the six MW Primaries), saved models, reuse of stored networks,
tails and curtailment cost reported, day-ahead and 14-day forecast plots, draft pull request. Does
not follow: the four-month screen (the three cutoffs are the pre-registered design). The advice of
experiment 643 followed: compare against the 14-day tracker, report the gain without the largest
contributing Primary, scale the capacity-swap placebo.

## Departures from the brief

- **Summary weather, not members** (above).
- **The tracker is the ensemble-forecast version of experiment 643's.** Experiment 643 corrected a
  free network by its own mean error over the 14 days before each test day. Here the same rule
  applies, with three conditions: the history holds only days whose date is before the run's origin
  date; each history error is the free network's forecast from the last run issued before that
  day (its lead 24 to 48 hours slice) against cleaned metered power; and the shift applies to
  sunny-midday half-hours only, as in 643, so contrast (ii) is unaffected by it. In the first 14
  days after a cutoff the history is incomplete, and the days with fewer than 3 history days get
  no shift. The history never includes a day the network fine-tuned on, because fine-tuning stops
  at the cutoff.
- **Extra placebos** (0.5 and 1.5 scalings), taken from experiment 643's advice.
- **The control in contrast (i) is the lower of d and e per lead band.** The bootstrap resamples days
  and re-picks the control in each resample, so the interval includes the selection.
- **The bars** (0.067 and 0.181) are experiment 639's. The first analysis step re-derives them per lead
  band from the seed-split null on these rows, and the larger of the old and new value is the bar.
- **Number of contrasts.** Contrast (i): 2 controls x 3 lead bands x 2 worlds = 12, (ii): 1, (iii): 2,
  placebos: 2 x 2 = 4. The Bonferroni level is set for 19 contrasts.
- **A pooled pass whose leave-one-Primary-out fails is reported as not a pass.** If the tracker beats
  the free network but the explicit network does not, the explicit-PV line stops and the tracker is
  named the candidate.

## Leakage and row rules

- Fine-tuning rows are filtered by valid time before the cutoff, not by run time, so no run at
  lead up to 10 days carries a target after the cutoff. The early-stopping split is by day.
- Lead is counted from `power_fcst_init_time` (run time plus 9 hours), 24 to 240 hours. Rows below
  24 hours are not scored, which is a limit on the day-1 result.
- Sunny middays use realised clear-sky index (as in 639). The write-up adds two further slices: all
  middays, and middays whose ensemble-mean clear-sky index is at least 0.7, because conditioning on
  the realised sky favours no arm but hides the forecast's sunniness error at long lead.
- The ensemble-mean shortwave curve is smoother than the mean of member curves, and at day 6 to 10
  it approaches climatology, so the shuffled-day placebo is weak by construction there. The
  write-up says so where it reports the placebo by lead band.
- The cutoff 2025-07 fine-tunes on 15 months of ensemble rows and the others on 24. The write-up shows
  the contrasts per cutoff beside the pooled value.
- Only the ensemble summary tables with shortwave and temperature are used (the cloud-cover
  summary starts later than the 2025-07 cutoff).
- The fine-tune stage reuses the pretrained weather and datetime encoders; its mixer is new, because
  it takes the spread and lead columns.
- In the step world the 4 MW step starts in 2023-04 and the ensemble rows start in 2024-04, so the
  fine-tune sees only post-step data and the contrast tests whether the explicit term carries the
  capacity change from pretraining to fine-tuning, not a step inside the fine-tune.
- All arms, including the tracker, `fbias`, and the floors, are scored on the same rows (every run
  at lead 24 to 240 hours).

## Compute go/no-go

Fit arms: a, b, c, f, `perm` (plain, 0.5, 1.5), and `shuf`, which is 8 fine-tunes per seed, cutoff,
and world. A smoke fit fixes the minutes per fine-tune. If the full grid projects above 3
GPU-hours, stage 1 runs seeds 0 to 2 in the real months for all arms and in the step world for a,
b, and f only, with every second run subsampled for fine-tuning if the projection is still above 3
GPU-hours. The step-world placebos are then untested and the write-up says so.
