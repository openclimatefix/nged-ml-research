---
id: "00723"
slug: msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc
title: >-
  Does the half-hourly output of solar farms under the same sky label the export stretches of a
  Primary that reports apparent power, in worlds where a farm is injected?
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  Anything else. (1) It stores aggregate regression and verdict tables, all private; (2) more than
  one design defensibly tests the hypothesis (the meter above the group, a half-hourly sky sensor,
  or a finer satellite product), and the amendment orders them; (3) it reads the site map of
  experiment 521, the generator record of experiment 722, and the frame of experiment 39, whose
  callers cannot be named without searching the repository; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Reviews: plan (the amendment's own Opus design
  review), code (Opus, before the first run), results, write-up, prose.
hypothesis: >-
  POST HOC. Every test of this experiment was written after the outputs of experiment 722 and its
  diagnosis had been read. Within a stretch of daylight half-hours between two near-zero readings,
  the concordance of the fast change of a Primary's reading with the fast change of a half-hourly
  sky sensor, built from the output of solar farms that sit under the same sky, separates the
  stretches on which the Primary exports from the stretches on which it imports. The separation
  has an area under the receiver operating characteristic curve of at least 0.85 in worlds where a
  metered farm is injected into a host Primary that reports a signed series, and the separation
  survives a transfer check on real import stretches. If job W2 is supported, the same statistic
  passes two controls: D12's anchor stretches and D3's post-bounce stretches.
prediction: >-
  Written before any fit, and written after experiment 722's results were known. With the farm
  sensor, the pooled area is 0.85 to 0.95 at injection level 2.0 and 0.75 to 0.88 at injection
  level 1.0. The CAMS sensor scores 0.60 to 0.75. The level 1.0 bar is the likeliest miss, and the
  transfer check on real import stretches is the second likeliest miss. In job W3, more than 60% of
  the evidence-giving stretches at D12 read as export. The pooled fold ratio of D10 on export hours
  has a standard error of 2.6 or more, so it cannot be read. The experiment can fail on any of
  these. A miss of the level 1.0 bar gives `not supported`, which closes the within-day hint for
  every sensor, so the predicted likeliest miss is also a predicted closure of the idea.
decision_rule: >-
  POST HOC; every bar below is copied from the private design amendment named in the page's
  decision section, and none was in experiment 722's plan. JOB W0 has no hypothesis and no
  verdict; it decides which tests can be read. The power table is a read-out: after erratum ruling
  3 no slope in jobs W2 or W3 carries a bar. The predicted standard error is 1.25 times the naive
  figure, and the factor applies to hour-to-hour changes only. The sensor audit has three
  conditions. (1) Resolution, report only: the share of hour-ending half-hours on which the CAMS
  clear-sky index does not change (100% expected), and the share of hour-ending half-hours on
  which `u_clear` does not change (0% expected). The arc term is void if the second share exceeds 1%
  at any one site (the job writes the share of every site), and job W2 then builds no world and is
  `inconclusive`, row 2 below. (2) Arc removal, per farm: the correlation of the farm's `ΔZ̃` with
  `Δu_clear` over daylight
  half-hours has a size of at most 0.1. A farm that fails is dropped from every sensor. (3)
  Coherence, per farm and per host sensor: for each host, and each farm of that host's sensor, the
  correlation of the farm's `ΔZ̃` with the mean `ΔZ̃` of the other farms of the same host's
  sensor, on daylight half-hours between April and September. A farm under 0.5 is dropped from
  that host's sensor and stays in other hosts' sensors. A host's sensor exists when at least 2
  farms remain, and job W2 is `inconclusive` (void before run) when fewer than 4 qualifying pairs
  remain. The value 0.5 is a sanity floor, and the power that job W2 needs sets a second, derived
  bar. For jointly normal changes with correlation `ρ`, the expected concordance is `(2 / π) ×
  arcsin(ρ)`, and a stretch with `n` usable pairs has a sampling standard deviation near `sqrt((1
  − C²) / n)`, which is 0.2 to 0.35 for 8 to 20 pairs. A point estimate of the pooled area clears
  0.85 with 80% probability when the true area is 0.88, at the standard error of 0.036. A true
  area of 0.88 needs a correlation between the sensor's and the reading's fast changes of 0.43 at
  8 pairs, 0.36 at 12, and 0.28 at 20. The need is set at 0.45. The dilution of a pair and level
  is `d = s_F / sqrt(s_F² + s_H²)`, where `s_F` is the standard deviation of the injected output's
  fast change and `s_H` that of the host's reading. A pair and level is "powered" when the
  injected farm's coherence with the host's sensor, times `d`, is at least 0.45. Job W2 is `void
  before run` when fewer than 4 pairs are powered at level 2.0. JOB W2 bars (farm sensor binds;
  the injected farm never enters its own world's sensor; the bars bind on days with exactly one
  post-bounce stretch, with the transfer mask; that stretch is scored when it holds at least 8
  usable pairs). Alignment: The cross-correlation of the reading's and the sensor's fast changes
  on anchor stretches peaks at lag 0. Readable: Bootstrap standard error of the pooled area at
  most 0.036; at least 4 powered pairs on at least 2 hosts; at least 30 stretches in each class.
  Area, export against import stretches, pooled: Point estimate at least 0.85, and lower end of
  the 95% interval above 0.75; at least 0.80 at level 1.0 alone, point estimate. The level 1.0
  point estimate is `not readable` (row 3 below) when either class at level 1.0 holds fewer than
  30 scorable stretches. Real sensor minus the sensor shifted by 28 days: At least 0.15 in area. The
  shifted sensor shifts only the fast change `ΔZ̃`, keeps the real sensor's index change, and is
  read on exactly the stretches and pairs of the area, so the placebo has no void state of its own.
  False export evidence at un-injected D5 and D10: At most 2% of true import stretches. The
  false-export bar at un-injected D5 and D10 passes when at most `floor(0.02 × n)` of the `n`
  scorable true import stretches give export evidence, and is `void` (not readable, row 3) under 30.
  The count of each host is reported beside the pooled bar, with no bar of its own. Transfer check
  on real import stretches: At un-injected D9, D11, D14, and D16, at least 50%
  of true import stretches have `C` below −0.25; void under 100 stretches or with a bootstrap
  standard error above 0.05. D15 dead band: At least 90% of D15's stretches give no evidence,
  binding only under the condition of section 5.3 of the amendment: when job 1 classes the farm
  inside D15's polygon "outside every metered Primary" and the farm is not an add-back candidate at
  D15, the bar binds as pre-registered, and when the class is embedded or unresolved, or the farm is
  an add-back candidate, the bar is report-only. The D15 dead band is `void` and neutral
  under 30 stretches, and the page reports the share among scorable stretches beside the
  every-stretch share. Recall on D10's real export stretches: At least 0.5 on 30 or more scorable
  stretches; `void` under 30, in which case the verdict rests on the other bars and the page says
  "no bar on real export was read". A recall under 0.5 on 30 or more scorable stretches gives `not
  supported`. Every share of stretches in jobs W2 and W3 has scorable stretches as its
  denominator, except the D15 dead band, which keeps every stretch. Calibration of the likelihood
  ratio: No bin of at least 100 stretches off the diagonal by more than 0.1. The plan page defines
  no likelihood ratio, so this bar is void by construction and neutral in every verdict, and the
  page lists it beside the verdict. The bootstrap
  resamples calendar days with replacement in each of 1,000 draws, and every pair, level, and host
  takes its stretches from the drawn days. A standard error of 0.036 gives a detection limit of
  0.10, the gap between the bar of 0.85 and the record's 0.75. Job W2 verdict: The first row that
  matches gives the verdict. Row 1, `invalid`: A correctness check fails: the holdout guard, the
  day count of ruling 2, a known-answer test on synthetic arrays, or the alignment check together
  with a time-stamp defect that a failing test shows. Row 2, `inconclusive` (void before run); no
  injected world is built: After the sensor audit, fewer than 4 qualifying pairs keep a sensor, or
  fewer than 4 pairs are powered at level 2.0, or the powered pairs sit on fewer than 2 hosts, or
  the arc term is void (a failed sensor audit). Row 3, `inconclusive` (not readable): Job W2 ran,
  and the standard error of the pooled area exceeds
  0.036, or either class holds fewer than 30 scorable stretches, or the alignment check fails and
  no time-stamp defect is found. Row 4, `supported`: Every binding bar passes with the farm
  sensor. Row 5, `partly supported`: Every binding bar passes except the transfer check (failed or
  void), the D15 dead band where that bar binds, or both. Row 6, `not supported`: Otherwise. The
  pooled area is scored on every qualifying pair at both levels. The powered pairs decide row 2
  and form one reported sub-pool with no bar. The D10 recall bar and the calibration bar are
  neutral when void, and the page lists every void bar beside the verdict. A failed sensor audit
  is `inconclusive`. `inconclusive` and `not supported` are never rerun. A rerun is the same
  attempt only when four conditions hold, and each job reruns at most once without the manager's
  approval: (1) a test on synthetic data, or a check that reads no bar statistic, fails before the
  fix and passes after the fix; (2) the fix changes no constant, population, sensor, or bar; (3)
  an Opus review confirms the fix; (4) both sets of outputs are kept, and the page reports both.
  Every read-out that reads the population of a bar runs after its job's verdict file is written.
  JOB W3 runs only when job W2 is `supported`. Controls, with bars: D12's anchor stretches on every
  day, where night import fixes the sign, and D3's post-bounce stretches on days with exactly one
  post-bounce stretch. A control passes when at most
  `floor(0.02 × n)` of its `n` scorable stretches give export evidence, and is `void` under 30
  scorable stretches. D12's dark-hour fold ratio is a read-out. Read-outs, no bar. Verdict: the
  first row that matches gives the verdict. (1) A control read on 30 or more scorable stretches
  fails: `a control fails, the hint is withdrawn at D12`, labelled "transfers not masked" when the
  switching logs gave no mask. (2) A control is `void`, or the switching logs gave no mask:
  `void`. (3) Otherwise: `controls pass, the read-outs stand`. The read-outs of the former job W1
  can withhold and can never admit: an export-hours interval at D10 above 0, or a blind
  meter-sensor score at folded D10 below 0.5 in area, contradicts the corrected identity of the
  amendment's section 2, and the daylight BSP factor may then not be proposed in any form. No size
  bar binds anywhere. The experiment ports nothing, so no smallest gain applies. Named constants
  (each a `Final` of `e00723_core.py`, set before any real data are read): 0.85, 0.80, 0.75, 0.15,
  0.036, 0.05, 0.10 (usable-pair size of the sky index change), 0.2 points (usable-pair size of
  the reading's fast change), 0.25 (dead band of `C`), 8 (minimum usable pairs), 0.5 (coherence
  floor, transfer share, and recall), 0.45 (powered-pair need), 0.1 (arc removal), 1% (arc term
  void), 90% (D15 dead-band share), 2% (false export evidence and the floor rule, `floor(0.02 ×
  n)`), 4 (pairs), 2 (hosts), 30 and 100 (stretch counts), 413 (days of the later span), 32
  (worlds of job W2), 1 core-hour (cap of each job), 0.1 (calibration bin), 1.25 (standard error
  factor), 2.8 (detection limit in standard errors), 1,000 (bootstrap draws), 28 days (placebo
  shift), 91 days (trailing capacity window), 1 hour (transfer mask margin), levels 1.0 and 2.0,
  and a reactive floor of 2 points.
benchmark: >-
  The injected farm's own signed truth in worlds built from signed hosts (D10 folded, D15 folded,
  and the six signed-MW Primaries (D5, D9, D10, D11, D14, and D16), un-injected); the CAMS sensor
  on hour-ending rows as the reference sensor; the sensor shifted by 28 days as the placebo; and
  D15, D12's anchor stretches, and D3 as controls.
row_set_id: >-
  not applicable (no forecast is scored): the jobs read the later span, the calendar days from the
  first day on which both transformers of 2025 have readings to 2026-06-30; job W0 asserts 413
  days, and another count is a code defect. No valid time on or after 2026-07-01 is read.
metric: >-
  The area under the receiver operating characteristic curve of the concordance `C` for export
  against import stretches, with 95% intervals from 1,000 bootstrap draws that resample whole
  days; shares of stretches that give export evidence; and detection limits of 2.8 bootstrap
  standard errors. Hour-to-hour regression slopes appear only as read-outs.
---

**POST HOC: we ask whether the half-hourly output of solar farms can show which way power flows
through a Primary substation that reports only a magnitude, in a repeat of experiment 722 whose
every test was written after 722's results were read.** A Primary substation steps the voltage down
to 11 kV. Eight of the 16 Primaries report apparent power in megavolt-amperes (MVA), which has no
sign, so a Primary that exports solar output shows a rise in its reading and not a negative value.
Where a Primary reports real power in megawatts (MW), its sign is known. Primaries are labelled D3,
D8, D10, and so on. A bulk supply point (BSP) is the substation that feeds a group of Primaries, and
its meter reports signed power. The two BSPs are BSP A and BSP B.

**Experiment 722 was inconclusive, and the diagnosis that followed it found four flaws in the plan
and no defect in the code.** Experiment 722 tested whether the BSP meter, minus the other Primaries,
shows the sign of one Primary's flow. The test of the meter's closure slope, which is the slope of
the hourly change of the meter's remainder on the hourly change of the left-out Primary, read about
2.0 at D15 where perfect cancellation gives 1, and the cloud-class audit failed. This experiment
drops the meter route as a source of any label. Experiment 723 tests instead one use of the sky:
within a stretch of daylight half-hours, does the reading of a Primary move against the sunshine
(the
Primary imports, so more sun lowers its reading) or with it (the Primary exports, so more sun raises
its reading)?

**The terms below are used in every job.**

- **Fold:** an MVA reading is a magnitude, the square root of the squared signed flow plus the
  squared reactive floor. "Folded D10" is D10's signed series passed through the same formula, with
  the sign hidden from the method under test.
- **Sky sensor:** a half-hourly series that moves with the sunshine over a Primary. The farm sensor
  is the mean output, as a share of capacity, of the metered solar farms under the same sky as the
  Primary and outside the Primary's own service-area polygon. The meter sensor is the negative of
  the BSP residual's change. The CAMS sensor is the hourly Copernicus Atmosphere Monitoring Service
  estimate at the Primary's location, and serves as the reference.
- **Same-sky pair:** a solar farm and a host Primary whose CAMS series correlate at 0.5 or more.
  Experiment 722's job 1 found 21 such pairs among 36 pairs of a farm and a host Primary. A pair
  qualifies for job W2 when the host's sensor keeps at least 2 farms after the injected farm is
  removed and the farm lies outside the host's polygon. The amendment counts 12 qualifying pairs,
  on three hosts (D10, D14, and D16).
- **Stretch and concordance `C`:** a stretch is a run of daylight half-hours between two near-zero
  readings, inside which the sign cannot change. `C` is the share of usable half-hour pairs in a
  stretch on which the reading and the sky move the same way, minus the share on which the reading
  and the sky move opposite ways. `C` above +0.25 is export evidence, `C` below −0.25 is import
  evidence, and a stretch in between, or with fewer than 8 usable pairs, gives no evidence.
- **Closure slope and fold ratio:** the closure slope is the slope of the hourly change of the
  residual that leaves one Primary out on that Primary's hourly change. The fold ratio is the same
  slope with the Primary's reading folded. The amendment shows that the fold ratio has the sign of
  the flow times the sign of one plus the co-movement of the rest of the residual with the Primary,
  and that the fold ratio's size identifies nothing.
- **Reactive floor and points:** the reactive floor is the smallest reading that a magnitude meter
  shows when the real flow is 0. A point is 1% of the Primary's capacity, and the floor is given in
  points.
- **Post-bounce stretch and anchor stretch:** a post-bounce stretch is a stretch that does not join
  the night, and an anchor stretch is a stretch that does, so night import fixes the sign of an
  anchor stretch.
- **Usable pair:** a pair of consecutive daylight half-hours on which the sky index changes by at
  least 0.10 and the reading's fast change is at least 0.2 points.
- **BSP residual:** the BSP meter minus every other child of the BSP.
- **Injection level:** the multiple of the host's median absolute signed reading on sunny middays
  before 2025-07-01 at which a farm's output is added. A sunny midday is a half-hour that starts
  from
  10:00 to 14:00 UTC, from April to September, with a clear-sky index of at least 0.7. The injected
  series is the farm's recorded output divided by the 99th percentile of the farm's own daylight
  output, with night set to zero, negative values set to zero, and a daylight dropout set to a
  gap, times the level, times that median (the definition of experiment 720).
- **Void:** a test is void when its detection limit, 2.8 bootstrap standard errors, exceeds its
  bar, because a slope or area the size of the bar could not be told from zero. A void test is
  reported as void and never read as a pass.

**The experiment has three jobs, and no job reads a Primary's sign from the BSP meter.** Job W0
decides which tests can be read and has no hypothesis. Job W2 scores the concordance in worlds where
a metered solar farm's output is added to a host Primary that reports a signed series. Job W3 runs
only if job W2 is supported, and checks the statistic against two controls: D12's anchor stretches,
where night import fixes the sign, and D3's post-bounce stretches. D12 is the Primary whose sign is
unknown. Every number on the page counts either half-hours, stretches, farm-Primary pairs, or
calendar days, and each table says which.

## Which rules changed from experiment 722, and why

**Four flaws in the specification of experiment 722 changed the rules. Two bars were moved after
their failure, and one group of bars was removed after its failure.** The amendment says so plainly
and so does this page. A rule changed after a result is seen is post hoc, so the page reports
experiment 722's verdict as it stood, and any pass here is a pass of the new rule only.

| Rule in experiment 722 | Rule in experiment 723 | Reason, from the amendment |
|---|---|---|
| D15 sign test and its placebo | Dropped, with no replacement test at D10 | A pooled slope reads only D15's known majority sign (D15 exports on 77% of half-hours). The sign score that deployment needs is per hour or per day, and an hour carries 0.016 nats and a day an area of 0.59. D10's export hours are too few (77 hours on 36 days) for a slope to be read |
| Closure bar of 0.7 to 1.3 at D15 | No size bar binds anywhere; dark-hour closure is a read-out | The slope is the sign times one plus the co-movement times a positive factor, so the size identifies nothing. A bar of 0.3 would need about 935 days at D15 |
| Generator-record bar: slope interval inside −1.3 to −0.7, and two placebos of 7 and 14 days | A one-sided permutation rule over circular day shifts, for future records only; the detection-limit gate of 0.5 is kept | The size band and the two-placebo rule were removed after the result was known. The two-placebo rule fails a true result 9.75% of the time. The 44-day record stays `void`, so the experiment reruns no generator fit and the reading gates nothing |
| Cloud class with audit conditions 3 and 4 (intermittent days 20% to 60%, at most 10% changing class) | Class dropped; the usable-pair rule scores every post-bounce stretch with at least 8 usable pairs | The two conditions failed in experiment 722 (10% to 13% and 13%). Hourly CAMS cannot define a class, because the clear-sky index changes once an hour. The scored population changes, and the page says so |
| Audit condition 2 (farm output against the sun's arc) | Arc-corrected condition 2, and a per-farm and per-host-sensor coherence condition | Condition 2 failed at every farm, because clear-day output ramps with the sun |
| Pair rule: a farm under the same sky as an unmetered host | Metered farms outside the host's polygon are admitted as injected farms, and the farm sensor replaces the CAMS index in the usable-pair rule | All six farms are metered under BSP A or BSP B, so the old rule finds no pair. The farm sensor never reads the BSP meter, so no megawatt is counted twice |
| Pair count of 17 on four hosts (first amendment) | 12 pairs on three hosts, and a bar of 0.036 on the standard error of the pooled area from a whole-day bootstrap | The first amendment had not applied the polygon test |
| Job W1 of the first amendment (D10 export-hours slope) | Withdrawn as a test; a read-out inside job W2 with no bar | The detection limit of 7 to 11 would sit against a bar near 8 |
| Three controls at D12 in job W3 | Two controls: D12's anchor stretches and D3's post-bounce stretches; D12's dark-hour fold ratio is a read-out | Erratum 1: a lower end above +0.5 is a size bar on a BSP statistic, which the amendment excludes |
| A failed sensor audit is `invalid` | A failed sensor audit is `inconclusive`; `invalid` means a defect of the code or of an input | Erratum 1: the two verdicts overlapped, and a rerun of a failed audit would be a second attempt |
| Daylight BSP factor of version 2 | Switched off under every outcome | No test here can admit it |
| Switching logs unread | Read with `engine="openpyxl"` and turned into a mask | Job 1 failed on a missing module; the repair adds no dependency |

**The amendment adds four bars that experiment 722 did not have.** The four bars are the interval
condition on the area (the lower end of the 95% interval above 0.75), the readable bar (a standard
error of at most 0.036), the transfer check on real import stretches, and the powered-pair bar
derived from the correlation that the area bar needs.

## The jobs

**Job W0 computes no statistic that depends on the direction of a co-movement between a sky
sensor, or a Primary's own leave-out residual, and the reading of a Primary that enters a bar of job
W2 or W3.** Counts, standard deviations of one series, farm-to-farm correlations, and counts that
depend only on the sizes of two changes are allowed. Job W0's estimate is under 0.01 core-hours.

| Output of job W0 | What it holds |
|---|---|
| Power table | For each fitted slope of jobs W2 and W3: the row count and the standard deviations of the regressor and of the response, with the standard error predicted as 1.25 times the naive figure. The table is a read-out, because no slope in jobs W2 or W3 carries a bar |
| Counts at D10 | The export hours and export days, with the base stated, as a recount of the review's 77 hours on 36 days |
| Sensor audit | The three conditions of the decision rule (condition 1 with the share of every site, so that one broken `u_clear` shows), the polygon-filtered pair table, the powered pairs with each pair's count of reference rows for the injection level, D12's and D15's sensor membership, and job 1's class for the farm inside D15's polygon, as a class only, with whether the farm is an add-back candidate at D15 |
| Share of pairs that pass | The share of daylight half-hour pairs that pass the usable-pair rule under each sensor, because a mean over 2 or more farms moves less than CAMS at one point |
| Switching logs | Each sheet's name, column names, and row count; for each sheet, the columns chosen as substation, start, and end, the rows read, the rows with a parsed start, and the rows matched to a Primary; one window per logged event and Primary (the logged start to end, or the local calendar day when no end is logged; a date with no time is read by the per-row rules of the ruling on switching-log times); the mask of the amendment. A time with no time zone is read as Europe/London. A sheet that has the substation and start columns and rows but matches no Primary stops the job as an input defect |
| Sky's share at D12, read-out | The squared correlation of the change of the residual that leaves D12 out with D12's farm sensor on lit hours, written as `r²` only and never as a correlation or a slope. A squared correlation carries no direction. The amendment bounds the share at 74% from dark hours alone |

**Job W2 scores the concordance in injected worlds.** Each of the 12 qualifying pairs is injected at
levels 1.0 and 2.0 into a host that reports a signed series, folded with a reactive floor of 2
points and within-half-hour averaging, as in the amendment's version 2. D10 folded, D15 folded, and
the six signed-MW Primaries (D5, D9, D10, D11, D14, and D16), un-injected, complete the worlds. The
farm sensor binds, the meter sensor is scored with the same statistics and with no verdict, and the
CAMS sensor is the reference. A pass with the meter sensor alone opens nothing. The page also
reports the pooled area without D14, which holds half of the 12 pairs. Job W2's read-outs, which
carry no bar, change no verdict, and run after the verdict file is written, are the pooled fold
ratio of D10 on export hours and on import hours, the blind score of the meter sensor at folded D10,
and the dark-hour closure slopes. A read-out that did not withhold may not later be cited as
evidence for a hint. Job W2 has an estimate of 0.05 core-hours and a cap of 1.

**Job W2 ends with two fits at D15, after the verdict file is written.** The two fits are lit-hour
closure fits with a second regressor, and D15 enters the dead-band bar, so they run last. Fit A adds
the farm sensor of D15. Fit B adds the output change of the farm inside D15's polygon instead.
Neither fit changes a verdict. Fit A has three pre-stated readings. An interval of the slope on
D15's change inside 0.6 to 1.5 means that same-sky generation outside D15 explains the excess. A
lower end above 2.0 means an add-back inside D15's series, or generation that the farm sensor does
not follow. Anything else is unresolved. Fit B supports an add-back when the slope falls inside 0.6
to 1.5 and the coefficient on the farm's megawatts lies between −1 and 0. The readings decide how
the page describes D15, and can forbid the design of rung U4 from citing D15's lit-hour slope. The
readings license nothing.

**Job W3 runs only when job W2 is `supported`, and is not a further attempt.** Job W3's two controls
are in the decision rule. Job W3's read-outs have no bar and run after the verdict file is written:
the shares of D12's one-stretch post-bounce days, April to September, on which `C` is above +0.25,
below −0.25, and in the dead band, for the farm sensor and the meter sensor, and the pooled fold
ratio of D12 on the same days with and without the farm sensor, and D12's dark-hour fold ratio. The
pooled fold ratio at D12 is never a label for a day, a stretch, or a half-hour, and never becomes a
prior. Job W3 has an estimate of 0.01 core-hours and a cap of 1.

## What each verdict changes

**The stop rule: this is the 20th experiment that produces or opens a gate to a sign label, and it
runs once.** A test is a statistic with a bar or pattern stated in advance, of which at least one
outcome admits: the outcome produces or opens a gate to a sign label, admits a hint or factor, sets
a prior, or licenses a later design to cite the statistic as evidence for a sign. A read-out
describes or withholds, and only an outcome that admits can add a false label to the record. Two
counters use this definition: experiments that hold at least one test that produces or opens a gate
to a sign label (the count of 18, 19th, and 20th below), and tests that use BSP closure for a sign
(the "no fifth test" rule). A read-out that did not withhold may not later be cited as evidence for
a hint. The design note for version 2 lists 18 earlier experiments by number: 79, 152, 358, 362,
365, 366, 370, 387, 408, 412, 417, 420, 424, 427, 506, 508, 543, and 674. Experiment 722 is the
19th. The stored BSP run is not counted. Experiment 722 was the fourth use of BSP closure for a
sign, and this experiment makes no fifth test of BSP closure: its D15 fits and its pooled fold
ratios are read-outs. **[Decided by the manager]** Experiment 723 is the 20th experiment and
experiment 724 is the 21st,
regardless of which runs first.

**Under every outcome, rung U3 and the build of the untangler proceed, the daylight BSP factor stays
switched off, and no third experiment on either idea starts without the maintainer's recorded
approval.** Rung U3 is the hindsight switching model of experiment 720 with its known-answer rungs,
and the untangler is the model that would write a probabilistic series of past signed power for each
MVA Primary. The fast gain `a` is the gain between 0 and 1 on the fast PV term of the untangler's
emission, and with `a` = 0 the untangler ignores half-hour cloud.

| Job W2 | Job W3 | The within-day hint | The untangler |
|---|---|---|---|
| `supported` | Controls pass | Admitted for the next design of the hint rung | The rung gains one arm with `a` free and the farm sensor as the fast PV term. The first buildable version is unchanged |
| `supported` | A control fails | Withdrawn at D12 and D22; the injected-world result stays on record | `a` = 0 at D12 and D22 |
| `supported` | `void` | Paused | `a` = 0 |
| `partly supported` | Does not run | Paused: the hint labels farm-driven export and is unproven for distributed PV | `a` = 0 |
| `not supported` | Does not run | Closed for every sensor; no variant with other thresholds, sensors, or levels runs | `a` = 0 for good; the sign at D12 rests on the PV estimate and the structural hints |
| `invalid` | Does not run | A defect of the code or of an input is fixed and the job reruns as the same attempt, under the four rerun conditions of the decision rule | `a` = 0 |
| `inconclusive` (including a failed sensor audit) | Does not run | Paused | `a` = 0 |

**A closed idea runs in no variant again, and a paused idea runs again only on data that do not
exist today.** An example is a longer later span once the holdout is released, and a rerun needs the
maintainer's recorded approval.

## Compute and pre-run gates

**The experiment needs CPU only, and its estimate is under 0.1 core-hours in total.** The estimates
scale from the 7.9 and 1.2 core-seconds of experiment 722's two jobs: 0.01 for job W0, 0.05 for job
W2, and 0.01 for job W3. Each job has a cap of 1 core-hour. Each job runs inside one `systemd-run
--user --scope` at a time, on one pinned core, with `MemoryMax=12G`, `MemorySwapMax=0`,
`OOMPolicy=continue`, one thread for BLAS and Polars, `CUDA_VISIBLE_DEVICES=""`, and a check for 25
GB of free memory before launch.

**The timing gate counts real units of work.** Jobs W0 and W3 make one elapsed-time check each. Job
W2 times the first world, scored with all four sensor variants (farm, meter, CAMS, and the sensor
shifted by 28 days), and the first 10 bootstrap draws on the first world's stretches repeated to the
expected size of the pooled population. It projects the whole job: `seconds per world × 32 worlds +
seconds per draw × 1,000`, then five more pooled bootstraps in the read-outs, a second scoring of
every world for the unmasked read-out, and 7 closure-fit bootstraps of 1,000 draws each. The job
stops when the projection exceeds the cap, and checks the elapsed time again before the pooled
bootstrap. The job writes the number of worlds it scored, the number it planned, and the name and
reason of every world it skipped, and a test asserts that no world is scored twice. A projection
over the cap before a job starts is reported to
the manager, who may raise the cap, and writes no verdict. A stop after the job has started returns
`inconclusive`.

- **Gate 1:** a fresh Opus agent that did not write the code reviews it before any first run that
  produces a number the page quotes. The reporting fixes below go in with the new jobs.
- **Gate 2:** the session sends the manager the message "fits starting" before each job, and waits
  for the reply.
- **Gate 3:** the maintainer or the manager has ruled on decision 1 below.
- **Gate 4:** both plan pages match erratum 1, and the must-fix items of the review of the erratum
  are applied.

**Three guards apply to every job.**

- **Row 1 reads the known-answer tests:** job W2 runs this folder's tests on synthetic arrays
  before it starts, and a failing test makes the verdict `invalid`. The run log holds the count of
  tests and the code hash.
- **A rerun keeps both sets of outputs:** a job moves its earlier outputs into `attempt_<job>_<n>/`
  before it writes new ones. Each job writes an attempt marker with its attempt number before it
  reads any input, and refuses to start when the marker exists (the earlier attempt crashed or
  ended with any verdict, `invalid` included), unless an environment variable names the job and the
  attempt that the manager approved.
- **A pair with no injection scale is counted, not dropped:** a pair whose host has fewer than 200
  reference rows with a finite injected series for the injection level gives no world. Job W0
  counts such pairs, job W2 writes the name, the reason, and the count of reference rows of every
  skipped world to the verdict file, and the number of worlds scored is compared with the number
  planned.

**Three reporting fixes apply to every job.**

- **R1, the row count of every statistic:** each row of a results table reports `n_rows` from the
  mask that the statistic used, with the count of days. The verdict file holds the count at each
  filter stage: finite, daylight, outside a transfer window, usable pairs, scorable stretches, and
  stretches in each class.
- **R2, identities with their conditions:** a docstring states `Δm = s × c × ΔX` only "within one
  sign", and the fold ratio `s × (1 + κ) × g` only under the linearity assumption of the amendment's
  section 2.1.
- **R3, an empty population shows its cause:** the run log holds, for each sensor and host, the two
  thresholds of the usable-pair rule, the share of daylight pairs that pass each threshold alone and
  both together, the 99th percentile and the largest value of each change, and the count of
  scorable stretches.

**The experiment follows these earlier advice items:** do not rerun a closure test with its old
bars, check the size of a sign test before any run (job W0's power table, now a read-out), judge a
placebo against a null distribution (the permutation rule), check the CAMS time resolution before
defining a cloud class (the audit's first condition), and test closure on dark hours. It departs
from none.

## Rulings before the run

**The design lead ruled on every reading below before any run, and none of the rulings changes a
threshold, a bar, or a verdict word.** The rulings answer the first and second Opus code reviews
of experiment 723, which found that the plan page left each reading open. Each reading is marked
`RULED` in the code. The readings that stay marked `PROVISIONAL` are the ones that the reviews
judged unable to flip a verdict, apart from one low-risk case: the missing-value rule of the
overcast days.

- **[Design lead] Void arc term:** the arc term is void if any one site's share of hour-ending
  half-hours on which `u_clear` does not change exceeds 1%. A site whose share cannot be computed
  (NaN), for example a site with no lit hour-ending half-hour, also voids the arc term, and an
  empty set of sites is void. Job W0 writes the share of every site.
  When the arc term is void, job W2 builds no world and the verdict is `inconclusive`, row 2 (a
  failed sensor audit).
- **[Design lead] Placebo population:** the placebo shifts only the fast change `ΔZ̃` by 28 days and
  keeps the real sensor's index change. The placebo is read on exactly the stretches and pairs of
  the area, so it has no void state.
- **[Design lead] Injection level:** the reference rows and the farm series are experiment 720's
  definition. A reference row is a half-hour that starts from 10:00 to 14:00 UTC, from April to
  September, before 2025-07-01, with a clear-sky index of at least 0.7, and a pair needs at least
  200 such rows. The injected series is experiment 720's `farm_norm`: the farm's recorded output
  divided by the 99th percentile of the farm's own daylight output (a sun zenith under 75
  degrees, before the holdout), zero at night, with negative values clipped to zero and a
  daylight dropout (an output of 0 or less with CAMS irradiance above 100 W/m²) set to a gap. The
  series is multiplied by the level and by the host's median absolute signed reading on the
  reference rows. The host's series is a gap wherever the farm's series is a gap, as in experiment
  720; experiment 720 also masked the gaps of all three of its farms in every world of a site, and
  this experiment masks only the injected farm's own gaps, because the two levels of one pair share
  one farm. The code copies `farm_norm` from the folder of experiment 720 (see the Files section).
- **[Design lead] D15 dead band:** the bar binds only when job 1 classes the farm inside D15's
  polygon outside every metered Primary and the farm is not an add-back candidate at D15. In every
  other case the bar is report-only. Under experiment 722's class definitions the bar is
  report-only in practice: a farm inside D15's polygon whose class is `outside_or_added_back` is
  also an add-back candidate at D15 whenever the farm is subtracted, so the binding condition holds
  only for a farm that is not subtracted. A reader should not take the D15 dead band for a live
  test.
- **[Design lead] Calibration bar:** the bar is void by construction, because the plan page defines
  no likelihood ratio and nobody defines one after the fact. The code records the bar as void, and
  the page lists it beside the verdict.
- **[Design lead] Controls of job W3:** D12's anchor stretches are scored on every day, and D3's
  post-bounce stretches are scored on days with exactly one post-bounce stretch.
- **[Design lead] Switching-log times:** job W0 reads every cell of the workbook as text and parses
  each cell on its own, so one text cell such as "ongoing" in a column of dates becomes a missing
  time, is counted, and cannot make the sheet unreadable. A time with no time zone is read as
  Europe/London. A time that carries a zone is converted to UTC. The workbook program turns every
  date into a date and time at midnight, so two per-row rules read a date with no time. First, an
  end at exactly local 00:00 on a row whose end is not after a start on the same local date is local
  24:00 of that date, so a same-day start never falls after its end. An end at 00:00 on a later date
  than the start stays at 00:00. Second, a start with no logged end covers the local calendar day of
  the start, from local 00:00 to local 24:00, whether the start is timed or not. A start or end
  that parses to a year outside 2000 to 2100 (a two-digit year such as "25" reads as year 25) is an
  error: it is counted as implausible and treated as a missing time. Windows whose end still
  precedes their start are counted and listed in the check file and are not masked, and the file
  also counts the ends that do not parse and the windows that lie wholly outside the grid.
- **[Design lead] False-export bar:** the bar stays pooled over un-injected D5 and D10, as the
  single count in the decision rule reads, and the count of each host is reported beside it with no
  bar.
- **[Design lead] Pairs with no injection scale:** a pair whose injection scale is not finite is
  logged and counted, and is never dropped without a message.
- **[Design lead] Farm that reads negative:** job W0 writes, for each qualifying pair, whether the
  farm's median recorded output on the reference rows is 0 or less, and job W2 treats such a pair
  as non-qualifying and logs the count, because the injection would otherwise add import where the
  world is meant to gain generation.
- **[Design lead] Reruns:** job W0, job W2, and job W3 each write an attempt marker and an attempt
  number before they read any input. A later start of the same job is refused, whether the earlier
  attempt crashed, ended `invalid`, or ended with any other verdict, unless the manager sets
  `E723_RERUN_APPROVED_BY_MANAGER` to the job and the number of the attempt to start, for example
  `w0:2`. Any other value refuses, so one approval permits one rerun, and the marker records the
  approval. A start that is refused or fails the launch check writes its run log to a separate
  file named `run_log_<job>_refused_<time>.json`, and never overwrites the run log of an earlier
  attempt. The folder of an archived attempt is named `attempt_<job>_<n>`, so each job's attempts
  are numbered on their own.
- **[Design lead] Days dropped by a farm's gaps:** a gap or dropout in the injected farm's series on
  a lit half-hour drops the whole day from the injected world, because the world has no reading
  there. Job W2 writes, for each injected farm, the number of days of the later span that its gaps
  drop, beside the number of lit days, in the verdict file, so that a reader can tell a level 1.0 bar
  left unreadable by dropouts from one left unreadable by too few export stretches.
- **[Design lead] No qualifying pair:** job W0 writes `w0_checks.json` and the pair table, with
  every column present, when no pair qualifies, so that job W2's row 2 records the outcome.
- **[Design lead] Timing gate:** the projection covers the whole of job W2, including the second
  scoring for the unmasked read-out and the bootstraps of the read-outs.
- **[Design lead] Sensor on hour-ending rows:** the CAMS sensor uses the hourly change of the
  clear-sky index and of the sky's fast change, and the reading's fast change is summed the same
  way,
  because the index of experiment 39's frame is held inside each hour. The CAMS sensor has about
  half as many candidate pairs per stretch as the farm sensor, so its area is read on its own,
  smaller set of scorable stretches, which is a different population from the farm sensor's. The
  prediction of 0.60 to 0.75 for the CAMS sensor refers to that population. The CAMS area enters no
  bar and no verdict.

## Decisions for the maintainer or the manager

**Ten decisions belong to the maintainer or the manager, and neither the amendment nor its erratum
takes any of them. The manager has since decided the count in row 1 and the rule in row 7, and the
rows that carry no marker stay open.** The design amendment is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_design_v3_amendment_v2.md`, its Opus
design review is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_v3_amendment_review.md`, the ruling on
each finding of the review is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_v3_triage.md`, and experiment 722's
diagnosis is `data/ml-experiments/briefs/msa_architecture/e722_u2a_diagnosis.md`. The erratum that
corrects the amendment, and governs where the two disagree, is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_erratum_1.md`, and the Opus review of
the erratum is `data/ml-experiments/briefs/msa_architecture/erratum_and_plans_review.md`. Version 2
of the amendment itself has had no Opus review.

| No. | Decision | Recommendation |
|---|---|---|
| 1 | [Maintainer or manager] Whether this experiment runs, as the 20th experiment, in the trimmed form of jobs W0, W2, and W3. **[Decided by the manager]** The count: experiment 723 is the 20th experiment and experiment 724 is the 21st | Run it: the maintainer asked for the idea by name, and hourly CAMS cannot test it |
| 2 | The stop rule (a test is a statistic with at least one outcome that admits) and the outcome table above, which replace the stop rules of the two earlier rungs in version 2 | Adopt both |
| 3 | Flaw 1: drop the D15 sign test and its placebo, with no replacement test at D10 | Drop them |
| 4 | Flaw 2: remove every size bar and keep dark-hour closure as a read-out | Remove the bars |
| 5 | [Maintainer] Flaw 3: adopt the permutation rule for future generator records only, with the gate of 0.5 retained. A separate question is whether future tests set the detection limit against half the gap, as now, or against the full gap | Adopt the rule. The erratum recommends that the claim decides: a claim of presence against a point null takes the full gap, and a test with an outcome that states a size takes half the gap. No gate changes for a record whose result has been read |
| 6 | Flaw 4 and the within-day hint: remove the day class and audit conditions 3 and 4, adopt the farm sensor as the binding sensor, and admit metered farms outside the host's polygon as injected farms. The farms' output is a private regressor, and no page shows a farm's series | Adopt, and run job W2 on the farm sensor (option 3 of the amendment's section 6) |
| 7 | [Maintainer] The form of the daylight hints in the later rung: "one factor at most", or one factor with a latent sky measured by two sensors. Needed only if job W2 is `supported`. **[Decided by the manager]** "One factor at most" | "One factor at most". The latent-sky form needs the meter sensor, which is the paused daylight BSP factor under another name, and the meter sensor also holds the child's own flow, so it is not a second independent measurement of the sky |
| 8 | The CAMS 15-minute download of 22 or 154 requests | Decline for now; revisit only if the farm sensor fails its audit |
| 9 | [Maintainer] Whether the build of the first buildable untangler may follow its known-answer test directly, because the hint rung has no hint to test with the BSP factor off and `a` = 0 | Yes. Version 2 gates the build on a rung that is `supported` or `partly supported`, where `partly supported` means that no hint is kept and the model passes alone, and the known-answer test is that case with an empty list of hints. If this experiment later admits the within-day hint, the rung runs then as a paired contrast |
| 10 | The switching-log repair: `engine="openpyxl"` in this experiment's own code, with no change to `pyproject.toml` | Adopt the repair |

**Two further points are decided, and four decisions stay open.** The verdict word `not built`
**[Decided by the manager]** applies to experiment 724 where its job 5 stops, and experiment 723
never uses it: the verdict words of this page are the six of the decision rule. The four open
decisions below are shared with experiment 724, and the wording of each follows that page. Where a
decision belongs to experiment 724's build alone, the line says so and takes no position for this
experiment.

- **11. The power-factor alternatives [Maintainer; open].** Version 2 allows a reported alternative
  to the assumed reactive floor. Experiment 723 folds with a reactive floor of 2 points and reads no
  power-factor alternative.
    - *Design lead's recommendation:* report the alternatives of experiment 724 (the positive-flow
      mean under `Q` = 0 and under a power factor of 0.97) as two labelled alternatives with no bar,
      which changes no threshold. Experiment 723's fold keeps the reactive floor of 2 points.
    - *Opus reviewer's view:* the code review of this experiment records no position on the
      power-factor alternatives, and the review of experiment 724 asks only that the page list the
      two alternatives with no bar. This agrees with the recommendation.
- **12. The compute cap [Manager; open].** Experiment 723's estimate is under 0.1 core-hours in
  total, and each job has a cap of 1 core-hour. The cap of 5 core-hours for the series is experiment
  724's.
    - *Design lead's recommendation:* the erratum recommends 5 core-hours for the series in place of
      3, with a cap on each job. Experiment 723 keeps its cap of 1 core-hour for each job.
    - *Opus reviewer's view:* the code review found that the timing gate projected only part of job
      W2, because it left out the second scoring of every world for the unmasked read-out and the
      bootstraps of the read-outs. The gate now projects the whole job, and the review judged a real
      overrun of the cap of 1 core-hour against an estimate of 0.05 unlikely. This agrees with the
      recommendation.
- **13. The rows of the scored worlds [Manager; open].** Experiment 723 scores the later span of 413
  calendar days, from the first day on which both transformers of 2025 have readings, and not the
  whole fitted span of experiment 724.
    - *Design lead's recommendation:* take the erratum's whole-span rows for experiment 724, which
      change no threshold set before data were read, and keep this experiment's later span.
    - *Opus reviewer's view:* the review of experiment 724 finds the whole-span scoring "sound" and
      asks for the span to be defined, which this page does through the later span's day count of
      413. This agrees with the recommendation.
- **14. The placebo bar [Manager; open].** Experiment 723's bar is the real sensor's area minus the
  area of the sensor shifted by 28 days, at least 0.15, with the placebo read on exactly the
  stretches and pairs of the area.
    - *Design lead's recommendation:* the erratum states the placebo of experiment 724 as a paired
      interval above 0 with no size. The recommendation that changes no threshold set before data
      were read is to keep the sizes that were registered: 0.15 here and 0.1 in experiment 724.
    - *Opus reviewer's view:* the code review found that the first code read the placebo on
      different stretches from the area, so that a void placebo gave `not supported`, a state that
      the page does not have, and recommended the same stretches and no void state. The review of
      the
      erratum states that the placebo "is read on the same stretches as the area". This agrees with
      the recommendation.

**The amendment rejects one suggestion of its review, and this experiment makes no request to
NGED.** The review recommended asking NGED for signed or reactive readings at D12. The maintainer
has said that NGED holds no information that the project does not already have.

## Method

### Architecture

- **Nothing is trained.** Every job is a regression, a correlation, or a count on stored series, and
  no neural network or hidden-state model is fitted.
- **The inputs are the power table through `studies.power.scan_power()`, the frame and sun
  elevation of experiment 39, the CAMS files of experiment 412, the site map of experiment 521, and
  the outputs of experiment 722's job 1,** with nothing on or after 2026-07-01. The experiment's own
  code imports experiment 722's loaders by `PYTHONPATH` and never copies them.
- **The remainder that leaves a Primary out is the BSP meter minus every other child, plus the
  generators that job 1 classed as outside every metered Primary.** It never subtracts the Primary's
  own reading.
- **The farm sensor of Primary `i` is the mean output of the same-sky farms, less every farm inside
  `i`'s polygon, as a share of capacity.** Its index is `k^Z = Z / (c_Z × u_clear)`, where `c_Z` is
  the 95th percentile of `Z / u_clear` over the daylight half-hours of the trailing 91 days, and its
  fast change removes the day's arc term.
- **Farms carry the labels F1 to F6 only, and any row with a date stays under `PRIVATE/`.**

### Training regime

- **No training, so no seeds beyond the bootstrap seed, which is fixed.** Each job runs once, on one
  core, with its own output file written atomically.
- **Each job's estimate is at most 0.05 core-hours and its cap is 1 core-hour,** and the
  experiment's total estimate is under 0.1.
- **The switching-log workbook is read with `engine="openpyxl"`,** and a change of Primary `i` is
  dropped when either end lies inside a window of `i` or within 1 hour of a window's start or end.
  Every statistic carries the mask, and the same statistic without it is a read-out. When no window
  can be built, every statistic is labelled "transfers not masked".
- **No farm series is shown, and a page that publishes a posterior that used the farm sensor says
  that the posterior used farm output.**

### Measurement

- **The scored rows are the post-bounce stretches that hold at least 8 usable pairs,** on days with
  exactly one post-bounce stretch, scoring that stretch when it holds at least 8 usable pairs, in
  the later span of 413 days.
- **The truth is the signed series of each host minus the injected farm's output,** and the
  statistic is the area under the receiver operating characteristic curve of `C`, pooled over pairs,
  levels, and hosts.
- **Every interval is the 2.5th to 97.5th percentile of 1,000 bootstrap draws that resample calendar
  days,** with every pair, level, and host taking its stretches from the drawn days.
- **The experiment scores no forecast, so no climatology row, mean absolute error, or leaderboard
  row applies.** It reads no load-shift event, and no standing practice that concerns fitted
  forecasts applies.
- **The verdicts are those of the decision rule, and the read-outs of the former job W1 can withhold
  and can never admit.**

## Files

The code is written and has not been run. The page lists each file as written when the results are
published.

- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_core.py`: the named
  constants, the bootstrap, the concordance and usable-pair rules, the permutation rule, and the
  decision rules.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w0.py`: job W0,
  which writes the power table, the sensor audit, the polygon-filtered pair table, the switching-log
  windows, and the sky-share read-out, with `r²` only.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w2.py`: job W2,
  which builds the injected worlds, scores the concordance, writes the results table and the
  verdict file, and, after the verdict file, runs the two D15 fits and the read-outs.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w3.py`: job W3,
  which runs the two controls and the read-outs only after job W2 is `supported`.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_world.py`:
  the sensors, the folding and injection, the stretch scoring, and the closure
  fits, as functions of arrays.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_score.py`:
  the 32 worlds, the scoring of each world, and the evidence counts.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_guard.py`:
  the output folders, the anonymisation guard, and the run log with the input
  fingerprints.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_common.py`:
  the launch check, the refusal of a job whose earlier job has not passed, and
  the loading of the sensors and readings.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/tests/`:
  unit tests on synthetic data, skipped without `--run-studies`.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/run.sh`: the launch
  wrapper, which runs no job without `--go`.
- `ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/e00720_rung0b.py`: the
  definition of the injection level's reference rows and of the injection's size (function
  `build_probe_world`), copied into `e00723_world.py` and not imported, because the function needs
  experiment 720's panel and fold objects. The folder is on the branch `msa-mva-to-mw-rung-0`.
- `ml-experiments/00506-mva-sign-evidence-dip-depth-and-gated-solar-rule/e00506_data.py`: the
  function `farm_norm`, which experiment 720 calls to build the injected farm series (power over the
  farm's own daylight 99th percentile, night zero, negative values clipped, daylight dropouts as
  gaps). It is copied into `e00723_world.py` as `farm_norm` and applied to the arrays of experiment
  39's frame, and is not imported, because the original reads the frame by path.
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/`: the loaders, the holdout guard, and the
  shared helpers of [experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md), imported
  and not edited.
- `data/ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/`: the aggregate
  outputs that the page will quote, written by the jobs. Dated detail goes in the `PRIVATE/`
  subfolder.
- `data/ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/u0_same_sky_pairs.csv`,
  `u0_checks.json`, and `u2a_results.csv`: the aggregates of experiment 722 that this experiment
  reads.
- `data/ml-experiments/briefs/msa_architecture/mva_untangling_design_v3_amendment_v2.md`: the design
  that this plan follows (read only, never published).
- `data/ml-experiments/briefs/msa_architecture/mva_untangling_erratum_1.md`: the erratum that
  governs where it disagrees with the amendment (read only, never published).
