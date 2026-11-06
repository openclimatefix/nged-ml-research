---
id: "00609"
slug: bakeoff-wind-and-solar-design
title: >-
  How should the final bake-off score the best forecasts of the 3 metered wind farms and the 6
  metered solar farms, and what would it cost?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
status: done
kind: other
date: 2026-11-06
size: >-
  simple, by the five triggers: nothing is stored; one deliverable (a written design); no callers,
  because no script is written; the serving path and the degradation rules are not touched in this
  repository; one defensible design for each choice, with the alternatives listed. One Sonnet
  write-up review.
hypothesis: >-
  The wind and solar part of the final bake-off can be run from stored fitted artefacts and a small
  number of refits for under 8 GPU-h in total, and every scoring choice can be fixed before any
  candidate is scored on the final test.
prediction: >-
  Written before reading the experiment pages. The whole wind and solar part costs under 2 GPU-h
  (the most thorough design, at 3.1, did not meet this),
  because the earlier fits took seconds to about a minute each. The experiment 605 gaps force at
  least two scoring versions for the solar farms and one for the wind farms.
decision_rule: >-
  Stop and ask the manager if the projected cost exceeds 8 GPU-h. Otherwise report a ranked list of
  at most three designs.
benchmark: none, because no weather model or forecast is fitted or scored.
row_set_id: none, because nothing is scored.
metric: GPU-hour projection, and the list of scoring rules.
verdict: inconclusive
code: none (a written design; all numbers are read from earlier pages and from file listings)
best_arm: none
headline: >-
  The wind and solar bake-off would cost about 0.4 GPU-h in its cheapest design and about 3 GPU-h
  in its most thorough, against a ceiling of 8 GPU-h. Every earlier stored forecast covers
  only the development period, so every candidate has to be re-run for the final test.
take_home_message: >-
  Fitting is cheap for wind and solar: one wind head refit takes about 25 seconds per seed and one
  solar
  refit about 1 minute per seed. The expensive part is the data. The ensemble inputs of both
  technologies must be extended to 2026-10-31 by experiment 607 first, and the nine generators
  have had no exact-zero reading since 2026-03-26. We therefore score the generators in two
  ways, on the rows present and on a version that fills night half-hours of the solar farms with
  zero, and we do not compare final-test scores with development scores unless the same filter
  was applied to both.
lessons:
  - >-
    The stored wind and solar forecasts cannot be reused for the final test. They cover the
    development period, and the wind default is stored only for 53 Tuesday runs.
advice:
  - >-
    Check the newest date of the ensemble farm-point tables and member arrays that wind default 2
    and the solar network read, before anything else, because experiment 605 did not list them
    (Dependencies).
  - >-
    Fix the filter for exact-zero readings and apply it to the development months as well as the
    final test (Missing readings).
  - >-
    Report the scores of Solar E on their own row, and never inside a pooled solar number
    (Missing readings).
production_status: not-for-production
adopted_in: none
---

**The wind and solar part of the bake-off is cheap to fit and costly to prepare: about 0.4 GPU-h
for the cheapest design and about 3 GPU-h for the most thorough, against a ceiling of 8 GPU-h.**
The fit times come from the logs of earlier experiments and are estimates, so the first worker
must time one fit of each kind before relying on them. No weather model was fitted and no forecast
was scored for this page. The page is a design that a later worker can follow once the list of
release candidates is frozen.

The bake-off scores each frozen candidate once, on forecasts issued from 2026-07-01 to 2026-10-31
(the final test, opened once). The five development months are 2025-10, 2025-12, 2026-01, 2026-03,
and 2026-06. July 2026 is the first month of the final test. Nothing here is adopted for production,
and
the generators appear only as W1 to W3 (wind) and Solar A to F (solar).

## 1. Candidates, stored artefacts, and cost

**Every candidate must be re-run for the final test, because no stored forecast reaches past
2026-08.** The first table lists the candidates. "Stored" means a file under
`data/ml-experiments/` that holds the forecasts or fitted weights for the development period.

| Technology | Candidate | What it is | Stored | Needs for the final test |
|---|---|---|---|---|
| Wind | Wind default 2 | [Experiment 347](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/result/), arm y2: the frozen physics plant, run on each of the 51 ensemble members, then a three-component mixture-density head; capacity at p99.9 of earlier power ([experiment 461](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00461-wind-and-solar-retrain-with-p99-9-effective-capacity/result/)) | Plant files (`plant_stage3.json`), ensemble summaries, and `predictions_y2.parquet` for 53 Tuesday runs | Plant summaries for the new runs (CPU), a head refit (GPU), and a daily-origin run for the development months |
| Wind | Wind default 1 | ERA5-trained XGBoost on the 51 members plus the offset table of [experiment 259](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00259-wind-plant-plus-quantile-offset-head/result/) | Predictions for 53 Tuesday runs (experiments 303 and 347) | A refit on CPU |
| Wind | `climatology` | Per-farm climatology | Not stored as forecasts | Build on CPU |
| Wind | Persistence, hours 0 to 6 | Last reading held | Not stored | Build on CPU |
| Solar | `a2_member` | The 51-member encoder network of [experiment 430](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations/result/), refitted once a year ([experiment 493](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00493-solar-annual-against-quarterly-refits/result/) and [experiment 498](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00498-solar-e-annual-refit-schedule-only-read/result/)) | Weights `annual_a2_member_s0.pt` to `s5.pt` and predictions, six seeds | One annual refit per seed |
| Solar | `a2_member` quarterly | The same network refitted every quarter | Predictions of experiment 430 (seeds 0 to 5, four refits) | Two refits per seed in the test window |
| Solar | GAM reference | The additive-curve arm `a3_gam_full` of [experiment 272](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00272-metered-farm-weather-adapter-gam-vs-neural/result/) | Predictions, three seeds, per farm | A refit on CPU |
| Solar | XGBoost reference | The XGBoost reference of experiment 243 with the floors of experiment 224 | Predictions under `00243-*/predictions/` | A refit on CPU |
| Solar | `manual_heuristic` and `climatology` | The two baselines of the solar protocol | Not stored as forecasts | Build on CPU |

**Two stored sets do not match the bake-off's rows.** The wind default was scored on 53 Tuesday
runs, not on daily origins, so the development months need a daily-origin run of the plant
summaries. The solar forecasts were scored over 2025-09 to 2026-08 on refits dated 2025-09,
2025-12, 2026-03, and 2026-06, so the stored solar forecasts cover the five development months only
where those refits apply. The stored annual forecasts come from a single refit dated 2025-09, so
they match that rule even less well. The worker must check the refit dates against the coarse-refit
rule of
the common brief (day 0 and day 14 of each test month) and refit where the dates differ.

**Fit costs are small.** The numbers below are read from logs, and the totals are estimates.

| Fit | Cost per fit | Source |
|---|---|---|
| Solar `a2_member`, one refit, one seed | 29 to 61 s wall time (24 s of GPU time) | Logs of experiment 430 and the cost table of experiment 493 |
| Solar one-pass network, one refit, one seed | 8 s of GPU time | Experiment 493 |
| Wind head, one seed, one refit | about 25 s | Log of experiment 347 stage 4 (the 45 timed head fits took 1,102 s in all) |
| Wind plant summaries, one farm, hindcast and scored sets | about 96 s of CPU | `inputs_W1.log` of experiment 347 |
| XGBoost, GAM, `climatology`, `manual_heuristic` | CPU only, minutes | Earlier pages |

**The projection has three designs, ranked.**

| Rank | Design | GPU-h | Notes |
|---|---|---|---|
| 1 | Final test only. Solar: one annual refit and two quarterly refits, six seeds. Wind: one head refit at 2026-07-01, five seeds. Stored forecasts stand in for the development months where their rows match. | about 0.4 | The cheapest design: 18 solar fits at 61 s and 5 wind fits at 25 s, plus 30% for data loading. |
| 2 | Design 1, plus re-running the five development months with the same code (day 0 and day 14 refits, two seeds for wind and six for solar), so that the development and final-test numbers come from one script. | about 1.9 | Adds 60 solar fits and 20 wind head fits (10 refits). Catches a mismatch between the stored forecasts and the new code before the test is opened. |
| 3 | Design 2, plus a wind head refit every quarter inside the test window, and ten seeds for Solar E ([experiment 498](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00498-solar-e-annual-refit-schedule-only-read/result/) needed ten seeds to resolve a 1% contrast). | about 3.1 | Adds 1 wind refit and 4 more solar seeds for 13 refits. The extra seeds matter only if a contrast at one farm decides a ranking. |

**All three designs are below the 8 GPU-h ceiling, so no stop is needed.** The estimate counts
the longest fit time seen (61 s) for every solar fit and 25 s for every wind head fit, and adds 30%
for data loading. The first worker confirms it with one timed fit of each kind.

## 2. Scoring rules

**Every candidate is scored by the quantile score over 11 levels in points of capacity, on the
same rows, with the same capacity.** The levels are q01, q05, q10, q25, q40, q50, q60, q75, q90,
q95, and q99. The continuous ranked probability score (CRPS) is twice the mean pinball loss over
those levels. Capacity is the p99.9 rule of [experiment 461](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00461-wind-and-solar-retrain-with-p99-9-effective-capacity/result/)
for the wind farms. For the solar farms the capacity follows
[experiment 505](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00505-rolling-p99-9-effective-capacity-for-wind-and-solar/result/):
the old capacity at Solar A and Solar E, where the rolling and all-history p99.9 rules failed (Solar
A on score, Solar E on the tail test), and the worker chooses the rule for the other four farms from
experiments 461 and 505
and records the rule. The scoring code reads each capacity from one file, so no candidate can use
another.

| Metric | Definition | Why |
|---|---|---|
| Quantile score | Mean pinball loss over 11 levels, in points of capacity | The headline of both protocols |
| CRPS | Twice the mean pinball loss | Comparable with the demand pages |
| Interval coverage | Share of rows with the observed power inside q05 to q95; target 0.90 | Calibration |
| Tail calibration | Share of rows above q99 (target 0.01) and below q01 (target 0.01); the 0.99 and 0.01 pinball losses | [Experiment 347](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/result/) found 2.3% above q99, so the worker reports both shares for every candidate |
| Threshold-weighted CRPS | Weight on observed power above 80% of capacity (this page's own choice; no earlier page sets a threshold) | The part of the distribution that matters for curtailment |

**No curtailment-cost metric exists for generators yet.** The avoidable curtailment cost of
[experiment 601](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00601-nra-curtailment-cost-and-tail-calibration-of-assembled-candidates/result/)
is defined for demand nodes with an import limit, and experiment 347 lists a generator version as
"not computed". The design therefore reports the 0.99 pinball loss and the threshold-weighted CRPS
instead. The maintainer must choose whether to define a generator curtailment cost before the
test is opened; this page does not choose.

**Every metric is reported in every slice below, and every slice row shows its half-hour count.**
Intervals resample whole calendar months of origin, as in the earlier pages. The final test has
only 4 months, so intervals will be wide. The page states the minimum detectable effect beside
every contrast and counts the contrasts, so that the Bonferroni level is a number. Bonferroni
count: the number of contrasts is the number of candidates compared with each technology's default,
times the 3 lead bands, and the worker computes it from the frozen list.

| Slice | Definition |
|---|---|
| Lead band | 0 to 24 h, 24 to 120 h, 120 to 240 h; the headline is the whole range 0 to 240 h |
| Month | Each of 2026-07, 2026-08, 2026-09, 2026-10; the development months beside them |
| Season | Summer and autumn from the final test; winter and spring from the development months. The final test holds no winter and no spring, so no season is compared across the two periods |
| Hour of day | Each hour 0 to 23 UTC; daylight half-hours only for the solar farms |
| Very sunny days | Days on which the observed solar output reaches 90% of the clear-sky output on at least half the daylight half-hours (the clear-sky definition of `solar-v1` is not spelled out on its page, so the worker states the rule used) |
| High-wind days | Days on which the observed wind output of at least 2 of the 3 wind farms exceeds 80% of capacity on any half-hour |
| Near a data gap | Half-hours within 24 hours of a gap of at least 3 hours in the same series |
| Series | Every farm on its own row; the pooled number is a mean over farms, not over rows |

**Slices use the observed power only to label days, never as a forecast input.**

## 3. Missing readings and the experiment 605 gaps

**The nine generators have had no exact-zero reading since 2026-03-26, so the final test cannot be
compared with the development months unless both are filtered the same way.** [Experiment 605](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00605-bakeoff-final-test-data-check/result/)
found that the last exact zero at all nine series falls between 2026-03-23 and 2026-03-26. Solar
farms read zero in 47% to 68% of half-hours in the six months before March, and wind farms in 0.3%
to 4.8%. The cause is unknown, and Solar E also misses about 70% of its four middle daylight hours
in July and October 2026.

**Three versions of each score are reported, and a fourth is reported for the development months.**

| Version | Rows kept | Comparable with the development months? |
|---|---|---|
| A, rows present | Every half-hour that has a reading | Only after the development months are filtered the same way (version A-dev) |
| B, night filled | Version A plus the solar night half-hours (sun elevation at or below 0 degrees) added with an observed power of 0 | Yes for solar, with the development months scored as they were before |
| C, calm filled (wind only) | Version A plus wind half-hours that have no reading and whose ensemble-mean 100 m wind speed is below the plant's cut-in speed, added with an observed power of 0 | Yes for wind, as a sensitivity run only, because the fill rule uses a weather forecast |
| A-dev | The development months with every exact-zero reading dropped | The match for version A |

**Version A is the headline, and A-dev is its development-period match.** Dropping zeros removes
the easy half-hours, so version A scores are higher than the scores in the earlier pages and must
never be set beside them. Version B is a bound: night half-hours are scored with zero forecasts and
zero actuals, so the night fill lowers the score and is reported separately, never pooled with
daylight
half-hours. The solar protocol already scores daylight half-hours only, so version A and
version B differ only through the dawn and dusk half-hours that read exactly zero.

**Solar E is reported on its own row, and every pooled solar number is shown with and without it.**
Its missing midday readings are the largest gap in the final test, and the missing readings remove
the half-hours
that carry most of its score. A pooled solar mean that includes Solar E is not comparable with the
development months.

**What is comparable with the development months.**

- **Comparable:** version B for solar, and A against A-dev for both technologies.
- **Not comparable:** version A against the development scores of earlier pages; any pooled solar
  number that includes Solar E; and any season comparison.
- **Open question:** whether the source stopped reporting zeros or the farms stopped reading zero.
  The manager's question to NGED is listed in the risks.

## 4. Plots

**Every plot follows the standing plot rules of the common brief and the `dataviz` skill.** The
plots are anonymised: the axis shows the day of the week (1 to 7, repeated for the second week)
and never a calendar date. Plots are drawn from the saved forecasts, not from refits.

| Group | Panels | Stretch |
|---|---|---|
| Wind farms W1 to W3, one figure per farm | Two panels, each 280 px or taller: the default 2 median with its q05 to q95 band in one panel, and the same for wind default 1 in the other. Observed power in black; no other mark is black or grey | A 14-day high-wind stretch, a 14-day calm stretch, and a 14-day stretch around a gap |
| Solar farms A to F, one figure per farm | Two panels: `a2_member` annual with its band, and `a2_member` quarterly with its band | A 14-day very sunny stretch, a 14-day cloudy stretch, and a 14-day stretch around a gap; Solar E gets two gap stretches (July and October) |
| Score by lead band, month, and hour | Bars with 95% intervals, one panel per technology | The whole final test |
| Coverage and tail calibration | Points against the 0.90 and 0.01 targets, one row per candidate | The whole final test |

**Each figure has a legend outside the data area, a plain caption saying what to look at, and a
gate or gap strip checked against the data.** The gap strip is a thin bar under the panel and is
drawn from the rows with no reading. A strip that is on across the whole axis is a bug until
checked against the stored rows. A fresh Sonnet reviewer opens every figure rendered to PNG and
answers the figure-review checklist before the pull request is marked ready.

## 5. Outputs

**Forecasts are saved as Parquet in the layout of `backtest_export_2026-11-06/`, one file per
candidate and month, with a README.** The README states the candidates, the seeds, the refit dates,
the version of the missing-reading rule, and the caveats of section 3. The files hold:

- `series` (W1 to W3 or Solar A to F), `power_fcst_init_time`, `valid_time`, and `lead_hours`;
- `q01` to `q99`, in points of capacity and in the generator's unit (MW);
- `observed_power` (null where there is no reading) and `observed_filled` (the version B or C
  value);
- `sun_elevation_degrees`, `reading_present`, and `near_gap` (true within 24 hours of a gap of at
  least 3 hours).

No file holds a generator's name, its identifier, or its location. The generator files are private:
they stay under `data/` and are not published.

## 6. Dependencies and risks

**Dependencies.**

- **Experiment 607 (extend the cached inputs to 2026-10-31) must finish first.** Experiment 605
  found the ensemble summary, wind-component, and cloud-cover tables ending with the run of 2026-06-30.
- **The inputs of the two networks are not on experiment 605's list of 11 cached inputs.** Wind
  default 2 reads the 51-member farm-point tables that experiment 303's builders write, and
  `a2_member` reads the member arrays of experiment 430 (`member_arrays`, 402 MB). The first
  worker checks the newest date of both and adds them to experiment 607's list if they stop early.
- **The experiment 605 gaps** (section 3) fix the scoring rules. The filter must be agreed before
  the first candidate is scored.

**Risks.**

- **Capacity can leak.** Wind recomputes the p99.9 capacity at each refit, and the solar capacity is
  fixed or rolling according to the rule the worker records. A capacity computed from power that
  includes the test months would leak, so the worker
  asserts that every capacity uses only readings from before the refit.
- **Only 4 test months and 3 wind farms.** The intervals are wide. Experiment 347 found a gain of
  0.23 points with a 99.75% interval that included zero. The bake-off may not separate close
  candidates, and the page reports ties as ties.
- **Upper tail of wind default 2.** 2.3% of observations exceeded q99 in experiment 347. The worker
  reports this share for every candidate, and it is not a pass or fail rule here.
- **Solar E.** Solar E has the fewest scored rows, and ten seeds were needed to resolve one
  contrast.
  Design 3 adds those seeds; designs 1 and 2 do not.
- **One question for NGED.** Whether the source stopped reporting exact-zero readings on about
  2026-03-26. The answer decides whether version A or version B is the headline.
- **Frozen list.** The candidates must be frozen before the first score is read. Any change after
  that is a new bake-off.

## Checks

- **Median forecast minus mean forecast, and the mean-forecast error beside each quantile-score
  contrast:** not applicable, because this page reads no forecast.
- **Triggers fired:** none of the seven audit triggers applies to a design with no control, no
  contrast, and no earlier result to contradict.
- **Sources read:** the pages of experiments 347, 461, 464, 466, 493, 498, 499, 505, 506, 508,
  605, and 607, the solar evaluation protocol, and the file listings under
  `data/ml-experiments/` of experiments 272, 347, 430, 461, and 493. The fit times are read from
  the logs of experiments 347 and 430 and the cost table of experiment 493.
- **Pages not changed:** no other page or script is edited.

## What the next experiment should answer

Whether a fit, a score, and a plot of the three wind farms and six solar farms run end to end on
the final test with the rules above, once experiment 607 has extended the inputs.
