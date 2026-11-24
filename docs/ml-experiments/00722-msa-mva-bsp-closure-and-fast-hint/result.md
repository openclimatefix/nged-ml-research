---
id: "00722"
slug: msa-mva-bsp-closure-and-fast-hint
title: >-
  Does the meter of a group of Primaries, minus the other Primaries and the outside generators,
  show the sign of the flow through one Primary that reports apparent power?
theme: solar-disaggregation
also_relevant_to:
  - switching-labels-in-the-forecast
status: finished
kind: other
size: >-
  Anything else. (1) It stores topology tables, regression tables, and verdict files, all private;
  (2) more than one design defensibly tests the hypothesis (the meter above the group, or the
  within-day change of the reading against the sky), so the design note orders them as separate
  rungs; (3) it reads the site map of experiment 521, the engine state of experiment 697, and the
  frame of experiment 39, and the callers of those files cannot be named without searching the
  repository; (4) it touches no serving path or degradation rule; (5) it changes no contract. Full
  set of reviews: plan, code (Opus, before the first run), results, write-up, prose.
hypothesis: >-
  The signed meter above a group of Primaries (a bulk supply point, BSP) shows the physical flow of
  each child with the right sign and size. Subtracting every other child and every outside
  generator from the meter therefore leaves a residual that moves with the left-out child, by a
  slope of at least +1 when the child imports and of at most -1 when the child exports, for every
  reactive floor. The residual's 2-hour change follows an independent 44-day record of one
  generator below D8 with a slope near -1, and follows the hidden signed flow of D15 when D15's
  signed series is folded into a magnitude. Where those two known answers pass, the same residual
  separates export from import at D12 in a joint regression with a term for the sky.
prediction: >-
  Written before any fit. The generator slope lies within -1 +/- 0.3. D15's closure slope lies
  within 0.7 to 1.3, and D15's sign is recovered with an area under the curve above 0.9 on large
  hourly changes, of which under half are large enough to score. At D12 the calibration
  coefficient is positive and larger than 1, the test coefficient is below -1 with a wide interval,
  and D3's coefficient stays near +1. The experiment can fail on any of these.
decision_rule: >-
  Job 2 (rung U2a) is `not supported` if any readable test fails its bar, `supported` if at least
  one test is readable and every readable test and its placebo pass, and `inconclusive` if no test
  is readable. Only `not supported` stops job 3. Job 3 (rung U2b) is `supported` when the
  calibration bar, the reverse-flow test bar, and the three controls pass, `not supported` when the
  calibration bar or a control fails or the test refutes reverse flow, and `inconclusive` when a
  fit is void or the test is unresolved. Every bar and every threshold is a named `Final` constant
  of `e00722_core.py`, set before any real data are read. The experiment ports nothing, so no
  smallest gain applies. A failed rung is closed with no variant.
benchmark: >-
  Two known answers at BSP A (an independent 44-day generator record, and D15's own signed series
  folded to a magnitude), one placebo for each (a circular shift of the record by 7 and 14 days, and
  D15's reading shifted by 7 days), and D3 as a closure control.
row_set_id: >-
  not applicable (no forecast is scored): the jobs read past half-hours, and no valid time on or
  after 2026-07-01 is read.
metric: >-
  Slopes of the residual's change on the change of a child's series, with 95% intervals from 1,000
  bootstrap replicates that resample whole days; the area under the receiver operating characteristic
  curve for the sign; and detection limits of 2.8 bootstrap standard errors.
verdict: inconclusive
production_status: not-for-production
adopted_in: none
headline: >-
  Job 1 failed its cloud-class audit, so job 3 and the farm-pair rung did not run. Job 2 is
  inconclusive: the engine test is void on its detection limit (slope -0.85, interval -1.16 to
  -0.39), the folded D15 tests have no scorable hour, and the D15 closure slope is 2.02 (1.72 to
  2.36) against a bar of 0.7 to 1.3.
lessons: >-
  The BSP meter closes at D15 in dark hours (0.96) but not in daylight (2.53), the D15 sign test
  cannot be scored because D15 is too small, and the cloud class fails on hourly CAMS. Four rules of
  the specification need restating before any rerun.
take_home_message: >-
  We could not yet tell whether the meter above a group of substations shows which way power flows
  through one of them. The meter does follow a substation's flow at night, but in daylight it moves
  about twice as much as the substation does, because solar output nearby moves with the sky, and
  the substation we hoped to use for the sign test is too small to score. Four rules of the plan
  need rewriting before the experiment is repeated.
advice:
  - >-
    Do not rerun the BSP closure test with its current bars: restate the D15 sign test, the
    closure bar for daylight, the two-placebo rule, and the cloud class first (sections "What
    happened in job 2" and "Post hoc diagnosis").
  - >-
    Pre-register a size check for a signed-sign test before any run: compare the largest hourly
    change of the child with the size threshold, because D15's largest change (2.4 MW) is under half
    of its threshold (5.8 MW) (post hoc diagnosis).
  - >-
    Test closure on dark hours, or add a term for generation under the same sky, when the left-out
    child is small against the BSP's unmetered remainder (post hoc diagnosis, closure by hour).
  - >-
    Judge a placebo against its empirical null over many shifts, not two intervals that must both
    include zero, because the two-interval rule fails about 10% of true nulls (post hoc diagnosis,
    placebo distribution).
  - >-
    Check the time resolution of the CAMS clear-sky index before defining a cloud class from its
    half-hour changes, because the index changes once an hour (post hoc diagnosis, audit failure).
---

**We asked whether a meter above a group of substations can tell us which way power flows through
one of them, and this run could not answer.** Eight of the 16 Primary substations report apparent
power (MVA), which is a magnitude, so a Primary that exports solar output shows a rise in its
reading and not a negative value. The signed meter at the bulk supply point (BSP) above a group of
Primaries carries the sign. We subtracted the readings of the other Primaries and the output of the
metered generators outside every Primary, and asked whether the remainder moves with the Primary we
left out. We tested this against two answers we know independently: a 44-day record of one
generator below D8, and D15, a Primary whose signed series we fold into a magnitude ourselves.
Nothing here forecasts. The tests that could be read neither passed nor failed their bars: the
generator test was too noisy to read, the D15 sign test had no hour large enough to score, and the
D15 closure slope came out at 2.0 against an expected 1.

**The pre-registered verdict is `inconclusive`, and the experiment stops after job 2.** Job 1 ran
and failed its cloud-class audit. Job 2 ran and found no readable test. Job 3, the regression at
BSP A that would have tested reverse flow at D12, did not run. The farm-and-host rung that the
specification calls U1 did not run either. This page reports those pre-registered outcomes first
and then a post hoc diagnosis, which is labelled as such and changes no verdict.

![For each condition of the cloud-class audit, the value at each site as a dot, with the
passing range shaded. Condition 3 fails at all 16 Primaries, condition 2 at all 6 farms, and
condition 4 at the pooled sites. Condition 1 passes at all 6 farms.](figure_1.svg)

## What happened in job 1 (pre-registered outcomes)

**The cloud-class audit failed on three of its four conditions, so job 3 and the farm-pair rung
were skipped by rule.** The audit classes each day as clear, intermittent, or overcast from the
clear-sky index of the Copernicus Atmosphere Monitoring Service (CAMS) alone, and the figure shows
every value.

| Condition | Bar | Value | Result |
|---|---|---|---|
| 3: share of intermittent days, each of 16 Primaries | 0.2 to 0.6 | 0.105 to 0.130 | Fails at all 16 |
| 2: share of clear days with a large change in farm output, each of 6 farms | At most 0.05 | 0.634 to 0.836 | Fails at all 6 |
| 1: share of days that are clear, each of 6 farms | At least 0.7 | 0.77 to 1.00 | Passes at all 6 |
| 4: share of days that change class when the elevation limit moves from 10° to 15°, all sites pooled | At most 0.1 | 0.131 | Fails |

**Twenty-one of the 36 pairs of a host Primary and a solar farm are under the same sky, and none
of the 21 involves an unmetered farm.** A pair is same-sky when the half-hour changes of the CAMS
index at the two sites correlate at 0.5 or more. All 6 farms are metered generators under a BSP, so
no pair counts towards the rule of at least 4 same-sky pairs of unmetered farms. The farm-pair rung
is therefore `inconclusive` by its own rule, and the metered pairs are reported apart.

**Only D8 has an add-back candidate, so the form of D8's series is unresolved and the other three
Primaries need no form.** D3, D12, and D22 have no subtracted generator inside their service-area
polygons (status `not_needed`). D8 has one (status `unresolved`), and job 2 therefore reports both
forms for it as a non-binding variant.

**The engine record covers every half-hour of its 44 days at the BSP meter, and 84% of its changes
have a readable residual.** Those readable changes hold 90% of the squared change. The engine test
is not void on coverage, because the coverage is 100% against the bar of 90%.

**The remaining checks of job 1 passed or are report-only.** The stamps of all 18 series with a
signal align at lag 0. No row on or after the holdout date was read. The generator counts match the
design at both BSPs (8 at BSP A and 5 at BSP B). Five generators are classed differently under the
second, joint fit, and two change label between D8's eras, and neither has a bar. The switching logs
could not be read because a module failed to import, which is also report-only.

## What happened in job 2 (pre-registered outcomes)

**The U2a verdict is `inconclusive`, because every test is void.** A test is void when 2.8
bootstrap standard errors exceed its bar.

| Test | Bar | Result | Outcome |
|---|---|---|---|
| Engine record, leave-out form | Interval inside -1.3 to -0.7 and excluding 0, detection limit under 0.5 | Slope -0.85 (-1.16 to -0.39), limit 0.54 | Void |
| Engine, 7-day and 14-day circular shifts | Intervals include 0 | 0.28 (0.04 to 0.61) and 0.17 (-0.17 to 0.58) | Report-only, since the test is void |
| D15 closure | Interval inside 0.7 to 1.3, detection limit under 0.3 | Slope 2.02 (1.72 to 2.36) on 7,113 hourly rows over 413 days | Void |
| D15 sign | Area under the curve at least 0.85, at least 200 scorable hours | 0 scorable hours | Void |
| D15 placebo | Interval includes 0.5, at least 200 scorable hours | 0 scorable hours | Void |

**The engine slope is near the expected -1, and the test is void only because 44 days are few.**
The standard error is 0.194, so the detection limit of 0.54 is just over the bar of 0.5. The
all-children form gives -0.84. D8's second variant, which also subtracts the unresolved generator,
gives a readable test: its slope is -0.81 (-1.07 to -0.45), and its interval fails the band on its
upper end. That variant is non-binding, and it would have produced a `not supported` verdict had it
bound.

**The D15 sign test and its placebo have zero scorable hours, and the reach of D15's hours over
the size threshold is 0.0.** No hourly change of D15 reaches twice the noise scale of the residual.
The D15 closure is void on its detection limit (0.46 against 0.3), and its point estimate is above
the band. The report-only D10 closure under BSP B is 7.50 (7.18 to 7.81), with 3 scorable hours.
The run did not stop job 3 on a failed test, because no test failed.

## Post hoc diagnosis (not pre-registered)

**The diagnosis below was written after the results and changes no verdict.** An Opus agent re-derived
the numbers with plain numpy, and the sections that follow state its findings. It found no code
defect: its independent closure slope of 2.02195 on 7,113 rows matches the run's 2.02195 to 15
figures. It attributes the anomalies to four flaws in the specification, which are the subject of
the last section.

**D15 is too small for its sign to show through the noise of BSP A.** D15's largest hourly change
after de-meaning is 2.4 MW. The size threshold is twice a noise scale of 2.89 MW, which is 5.8 MW.
The 99th percentile of D15's change is 1.26 MW. A threshold that left 200 hours would sit at about
0.3 noise scales, where the ratio of the residual's change to D15's change is mostly noise. The sign
test therefore has no power at D15 under any threshold that keeps it meaningful. The specification
predicted that under half of D15's hours would score, and did not check that against D15's size.

**The meter follows D15 at night and exaggerates it in daylight.** Splitting the closure slope by
hour of day gives the table below. The 2.8 standard-error limit is 0.45 for the dark D15 hours, so
that interval would still be void under the bar.

| Child | Hours | Slope | 95% interval | Rows |
|---|---|---|---|---|
| D15 | Dark | 0.96 | 0.62 to 1.27 | 2,793 |
| D15 | Lit (sun at least 10° at the hour's end) | 2.53 | 2.08 to 3.02 | 3,227 |
| D10 | Dark | 1.35 | 1.25 to 1.44 | 9,223 |
| D10 | Lit | 8.75 | 8.38 to 9.09 | 8,656 |

The excess sits in the BSP meter, not in the subtraction. Regressed on D15's change, the BSP A meter
has slope 2.60, the other children's sum about 0.47, and the subtracted generator 0.01, so no child
is subtracted twice. A slope above 1 means that the unmetered remainder of the BSP co-moves with a
child small enough to be swamped by it. Two daytime mechanisms fit and these data cannot separate
them: generation under the same sky that D15's reading does not include, and a partial add-back of
embedded solar inside D15's own series. The non-binding variant that adds the unresolved generators
lowers the slope to 1.58, but it risks counting one generator twice, so 1.58 is not a cleaner
estimate.

**The 7-day placebo of 0.28 is a tail draw from a null centred on 0, so the placebo rule alone would
have failed the engine test by chance.** Rolling the engine record by every whole number of days
from 3 to 40 gives 38 placebo slopes with mean +0.007 and standard deviation 0.169. The 7-day value
is the largest of the 38. The main slope of -0.85 is 5.0 placebo standard deviations from 0. The rule
needs both the 7-day and 14-day intervals to include 0, and two nearly independent placebos fail
together by chance about 10% of the time. With a readable main slope, this run would therefore have
returned `not supported` and stopped job 3 on a chance draw.

**The audit failure comes from the time resolution of the CAMS clear-sky index, which changes once an
hour.** At D15's CAMS point the mean absolute change of the index is 0.106 on half-hours ending at
:30 and exactly 0 on half-hours ending on the hour. The index is held constant within each hour in
the frame this experiment reads. Condition 3 fails because only about 13% of days reach 6 fast
changes, although 67% of April-to-September days have a daily mean index in the intermittent band,
and the median such day has 4 changes. Condition 2 fails because the output of a farm on a clear
day ramps with the sun, and 64% to 70% of its large changes fall at sun elevations under 30°.
Condition 4 fails because the class boundaries are knife-edges on a coarse count. The audit stopped
job 3 and the farm-pair rung exactly as designed.

**Three reporting faults in the code do not change any outcome.** The row `d15_sign_with_crossing`
reports the number of finite hours (7,111) where its area is computed on no rows. The docstrings
say that the all-children slope equals the leave-out slope minus 1, which holds only for a child
that always imports, and D15 exports on 77% of half-hours after both 2025 transformers start. The
size-threshold diagnostics (noise scale, threshold, and largest change) are not written to the run
log, and the CAMS resolution is not reported by job 1.

## Checks

- **Median minus mean forecast, per site:** not applicable, because the experiment scores no
  forecast.
- **Error of the mean or median forecast beside the quantile-score contrasts:** not applicable, for
  the same reason.
- **Audit triggers:** two fired, since a control failed (the 7-day placebo lies outside its bar)
  and a result contradicts its prediction (the D15 closure slope of 2.02 against 0.7 to 1.3). An
  Opus diagnostic served as the audit, and the section "Post hoc diagnosis" gives its findings:
  it found no code defect, it explained each trigger by a data fact or a specification flaw, and
  the three reporting faults above are triaged as fixes before any rerun. The diagnostic is the
  private plain-text file `e722_u2a_diagnosis.md` in the `briefs/msa_architecture/` folder under
  `data/ml-experiments/`, so this page does not link it.

## What the next experiment should answer

**A rerun of job 2 under the current rules would reproduce the same void tests, so the next
experiment should settle four rules first.** The post hoc diagnosis names them:

- the D15 sign test needs a signed child whose swings are large against the BSP's remainder, or it
  is dropped;
- the closure bar needs to bind on dark hours, or to include a term for generation under the same
  sky;
- the placebo rule needs an empirical null over many shifts, or a joint test, in place of two
  intervals that must both include 0;
- the cloud class needs half-hourly CAMS or a definition that suits hourly steps, and condition 2
  needs the daily arc of the sun taken out.

A follow-on experiment with restated rules is planned. It will be labelled post hoc, because its
rules follow the results of this one, and this page makes no further commitment about it.

## What ran

**Job 1 (rung U0) has no hypothesis and holds no fit.** It joins each Primary to its BSP, classes
every metered generator, checks the engine record's coverage, the stamp alignment, the gaps, the
switching-log windows, the era-split table, and the holdout, runs the cloud-class audit from the sky
alone, and counts the same-sky pairs of hosts and farms.

**Job 2 (rung U2a) tests the BSP pathway against the two known answers, and job 3 (rung U2b), a
joint regression at BSP A with a term for the sky, did not run.** The bars of job 2 are in the
table of the results above, and the decision rule is in the front matter.

## Method

- **Architecture:** no neural network and no hidden-state model: every test is an ordinary
  least-squares regression through the origin on series de-meaned at matched quiet times, with the
  regressor's noise left in.
- **Architecture:** the residual that leaves Primary `i` out is the BSP meter minus the readings of
  every other child and plus the output of each generator that sits outside every Primary or is
  added back, and it never subtracts the reading of Primary `i`.
- **Architecture:** a metered generator's output is subtracted only where the add-back regression of
  job 1 puts the generator outside every Primary or in a Primary's added-back series, and never
  twice. A generator is an add-back candidate at an unsigned Primary only if it lies inside the
  Primary's service-area polygon.
- **Architecture:** D15's signed series enters job 2 only as the folded regressor `|X|` and as the
  truth for the sign, and the engine record enters as a 2-hour change.
- **Training regime:** nothing is trained. Each job runs once on one core, the seed of every
  bootstrap is fixed, and job 1 took 7.9 core-seconds and job 2 took 1.2 core-seconds.
- **Measurement:** matched quiet times are half-hours with the same local half-hour, calendar month,
  and day type (weekday against weekend or bank holiday) that lie at least 6 half-hours from a
  change in the engine state.
- **Measurement:** every interval is the 2.5th to 97.5th percentile of 1,000 bootstrap replicates
  that resample whole days, with the same resampled days shared by every child of one BSP.
- **Measurement:** the detection limit of a test is 2.8 bootstrap standard errors, and a test over
  its limit is void. The engine test is void on BSP-meter coverage only under 90% of its 44 days.
- **Measurement:** a passed sign test with a void placebo counts as void, the verdict of the two
  forms of an unresolved Primary is their common verdict or else `inconclusive`, and `inconclusive`
  also results when no test is readable.
- **Measurement:** no row on or after the holdout date was read, and the stdout of every job holds
  ratios, counts, shares, and intervals only.

## Files

- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_core.py`: the named constants, the
  bootstrap, the matched-time de-meaning, and the decision rules.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_u0.py`: job 1, which wrote the
  checks, the class audit, the same-sky pairs, and the era-split table below.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_u2a.py`: job 2, which wrote the
  results table and the verdict file below.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_u2b.py`: job 3 (it did not run).
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_sky.py`, `e00722_data.py`,
  `e00722_d8core.py`, `e00722_guard.py`, `e00722_common.py`, and `e00722_testkit.py`: the cloud
  class and audit, the loaders, the D8 era handling, the holdout guard and output paths, the shared
  helpers, and the synthetic-array helpers of the tests in `tests/studies/bsp_closure_722/`.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_figure.py`: draws `figure_1.svg`
  from the class-audit table.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/run.sh`: the launch wrapper, which runs no
  job without `--go`.
- `data/ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/u0_checks.json`,
  `u0_class_audit.csv`, `u0_same_sky_pairs.csv`, `u0_era_split.csv`, `u2a_results.csv`,
  `u2a_verdict.json`, and the run logs: the aggregate outputs that this page quotes (written by the
  jobs). Dated detail is under `PRIVATE/` there.
- `data/ml-experiments/briefs/msa_architecture/e722_u2a_diagnosis.md`: the private diagnosis of
  the post hoc section (read only, never published).
