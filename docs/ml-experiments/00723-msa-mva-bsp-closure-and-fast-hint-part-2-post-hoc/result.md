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
  passes three controls at D12.
prediction: >-
  Written before any fit, and written after experiment 722's results were known. With the farm
  sensor, the pooled area is 0.85 to 0.95 at injection level 2.0 and 0.75 to 0.88 at injection
  level 1.0. The CAMS sensor scores 0.60 to 0.75. The level 1.0 bar is the likeliest miss, and the
  transfer check on real import stretches is the second likeliest miss. In job W3, more than 60% of
  the evidence-giving stretches at D12 read as export. The pooled fold ratio of D10 on export hours
  has a standard error of 2.6 or more, so it cannot be read. The experiment can fail on any of
  these.
decision_rule: >-
  POST HOC; every bar below is copied from the private design amendment named in the page's
  decision section, and none was in experiment 722's plan. JOB W0 has no hypothesis and no
  verdict; it decides which tests can be read. A bar whose predicted detection limit exceeds the
  bar is `void before run` and is not fitted. The predicted standard error is 1.25 times the naive
  figure, and the factor applies to hour-to-hour changes only. The sensor audit has three
  conditions. (1) Resolution, report only: the share of hour-ending half-hours on which the CAMS
  clear-sky index does not change (100% expected), and the share of hour-ending half-hours on which
  `u_clear` does not change (0% expected). The arc term is void if the second share exceeds 1%.
  (2) Arc removal, per farm: the correlation of the farm's `ΔZ̃` with `Δu_clear` over daylight
  half-hours has a size of at most 0.1. A farm that fails is dropped from every sensor. (3)
  Coherence, per farm and per host sensor: for each host, and each farm of that host's sensor, the
  correlation of the farm's `ΔZ̃` with the mean `ΔZ̃` of the other farms of the same host's sensor,
  on daylight half-hours between April and September. A farm under 0.5 is dropped from that host's
  sensor and stays in other hosts' sensors. A host's sensor exists when at least 2 farms remain, and
  job W2 is `invalid` only when fewer than 4 qualifying pairs remain. The value 0.5 is a sanity
  floor, and the power that job W2 needs sets a second, derived bar. For jointly normal changes
  with correlation `ρ`, the expected concordance is `(2 / π) × arcsin(ρ)`, and a stretch with `n`
  usable pairs has a sampling standard deviation near `sqrt((1 − C²) / n)`, which is 0.2 to 0.35
  for 8 to 20 pairs. A point estimate of the pooled area clears 0.85 with 80% probability when the
  true area is 0.88, at the standard error of 0.036. A true area of 0.88 needs a correlation
  between the sensor's and the reading's fast changes of 0.43 at 8 pairs, 0.36 at 12, and 0.28 at
  20. The need is set at 0.45. The dilution of a pair and level is `d = s_F / sqrt(s_F² + s_H²)`,
  where `s_F` is the standard deviation of the injected output's fast change and `s_H` that of the
  host's reading. A pair and level is "powered" when the injected farm's coherence with the host's
  sensor, times `d`, is at least 0.45. Job W2 is `void before run` when fewer than 4 pairs are
  powered at level 2.0. JOB W2 bars (farm sensor binds; the injected farm never enters its own
  world's sensor; the bars bind on days with exactly one post-bounce stretch, with the transfer
  mask). Alignment: The cross-correlation of the reading's and the sensor's fast changes on anchor
  stretches peaks at lag 0. Readable: Bootstrap standard error of the pooled area at most 0.036;
  at least 4 powered pairs on at least 2 hosts; at least 30 stretches in each class. Area, export
  against import stretches, pooled: Point estimate at least 0.85, and lower end of the 95% interval
  above 0.75; at least 0.80 at level 1.0 alone, point estimate. Real sensor minus the sensor
  shifted by 28 days: At least 0.15 in area. False export evidence at un-injected D5 and D10: At
  most 2% of true import stretches. Transfer check on real import stretches: At un-injected D9,
  D11, D14, and D16, at least 50% of true import stretches have `C` below −0.25; void under 100
  stretches or with a bootstrap standard error above 0.05. D15 dead band: At least 90% of D15's
  stretches give no evidence; when the farm inside D15's polygon is classed "outside every metered
  Primary" the bar binds as pre-registered, and when the class is embedded or unresolved the bar is
  report-only. Recall on D10's real export stretches: At least 0.5; no bar under 30 scorable
  stretches. Calibration of the likelihood ratio: No bin of at least 100 stretches off the diagonal
  by more than 0.1. The bootstrap resamples calendar days with replacement in each of 1,000 draws,
  and every pair, level, and host takes its stretches from the drawn days. A standard error of 0.036
  gives a detection limit of 0.10, the gap between the bar of 0.85 and the record's 0.75. Job W2
  verdict: `invalid` when fewer than 4 qualifying pairs keep a sensor, or the alignment check
  fails. `inconclusive` when the test is `void before run` or not readable. `supported` when every
  binding bar passes with the farm sensor. `partly supported` when every bar passes except the
  transfer check or the D15 dead band, or when the transfer check is void. `not supported`
  otherwise. JOB W3 runs only when job W2 is `supported`. Controls, with bars: on D12's anchor
  stretches, where night import fixes the sign, at most 2% of stretches give export evidence. At D3,
  at most 2% of post-bounce stretches give export evidence. D12's dark-hour fold ratio has a lower
  end above +0.5, if job W0 predicts a detection limit under 1. Read-outs, no bar. Verdict:
  `controls pass, the read-outs stand`; `a control fails, the hint is withdrawn at D12`; or `void`
  when the controls cannot be read or the switching logs gave no mask. The read-outs of the
  former job W1 can withhold and can never admit: an export-hours interval at D10 above 0, or a
  blind meter-sensor score at folded D10 below 0.5 in area, contradicts the corrected identity of
  the amendment's section 2, and the daylight BSP factor may then not be proposed in any form. No
  size bar binds anywhere. The experiment ports nothing, so no smallest gain applies. Named
  constants (each a `Final` of `e00723_core.py`, set before any real data are read): 0.85, 0.80,
  0.75, 0.15, 0.036, 0.05, 0.10 (usable-pair size of the sky index change), 0.2 points (usable-pair
  size of the reading's fast change), 0.25 (dead band of `C`), 8 (minimum usable pairs), 0.5
  (coherence floor, detection-limit gate, transfer share, and recall), 0.45 (powered-pair need),
  0.1 (arc removal), 1% (arc term void), 90% (D15 dead-band share), 2% (false export evidence), 4
  (pairs), 2 (hosts), 30 and 100 (stretch counts), 0.1 (calibration bin), 1.25
  (standard error factor), 2.8 (detection limit in standard errors), 1,000 (bootstrap draws), 28
  days (placebo shift), 91 days (trailing capacity window), 1 hour (transfer mask margin), levels
  1.0 and 2.0, and a reactive floor of 2 points.
benchmark: >-
  The injected farm's own signed truth in worlds built from signed hosts (D10 folded, D15 folded,
  and the six un-injected signed hosts); the CAMS sensor on hour-ending rows as the reference
  sensor; the sensor shifted by 28 days as the placebo; and D15, D12's anchor stretches, and D3 as
  controls.
row_set_id: >-
  not applicable (no forecast is scored): the jobs read the later span, the 413 days up to the
  holdout of 2026-07-01 in which both transformers of 2025 have readings, and no valid time on or
  after 2026-07-01 is read.
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
drops the meter route as a source of any label. It tests instead one use of the sky: within a
stretch of daylight half-hours, does the reading of a Primary move with the sunshine (the Primary
imports) or against it (the Primary exports)?

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
  stretch on which the reading and the sky move the same way, minus the share on which they move
  opposite ways. `C` above +0.25 is export evidence, `C` below −0.25 is import evidence, and a
  stretch in between, or with fewer than 8 usable pairs, gives no evidence.
- **Closure slope and fold ratio:** the closure slope is the slope of the hourly change of the
  residual that leaves one Primary out on that Primary's hourly change. The fold ratio is the same
  slope with the Primary's reading folded. The amendment shows that the fold ratio has the sign of
  the flow times the sign of one plus the co-movement of the rest of the residual with the Primary,
  and that its size identifies nothing.
- **Void:** a test is void when its detection limit, 2.8 bootstrap standard errors, exceeds its
  bar, because a slope or area the size of the bar could not be told from zero. A void test is
  reported as void and never read as a pass.

**The experiment has three jobs, and no job reads a Primary's sign from the BSP meter.** Job W0
decides which tests can be read and has no hypothesis. Job W2 scores the concordance in worlds where
a metered solar farm's output is added to a host Primary that reports a signed series. Job W3 runs
only if job W2 is supported, and checks the statistic against three controls at D12, the Primary
whose sign is unknown. Every number on the page counts either half-hours, stretches, farm-Primary
pairs, or calendar days, and each table says which.

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
| Daylight BSP factor of version 2 | Switched off under every outcome | No test here can admit it |
| Switching logs unread | Read with `engine="openpyxl"` and turned into a mask | Job 1 failed on a missing module; the repair adds no dependency |

**The amendment adds four bars that experiment 722 did not have.** They are the interval condition
on the area (the lower end of the 95% interval above 0.75), the readable bar (a standard error of
at most 0.036), the transfer check on real import stretches, and the powered-pair bar derived from
the correlation that the area bar needs.

## The jobs

**Job W0 computes no product of a Primary's reading with a residual or a sensor, and no test
statistic, before it decides which tests can be read.** Its estimate is under 0.01 core-hours.

| Output of job W0 | What it holds |
|---|---|
| Power table | For each fitted slope of jobs W2 and W3: the row count and the standard deviations of the regressor and of the response, with the standard error predicted as 1.25 times the naive figure. A bar whose detection limit exceeds the bar is `void before run` |
| Counts at D10 | The export hours and export days, with the base stated, as a recount of the review's 77 hours on 36 days |
| Sensor audit | The three conditions of the decision rule, the polygon-filtered pair table, the powered pairs, D12's and D15's sensor membership, and job 1's class for the farm inside D15's polygon, as a class only |
| Share of pairs that pass | The share of daylight half-hour pairs that pass the usable-pair rule under each sensor, because a mean over 2 or more farms moves less than CAMS at one point |
| Switching logs | Each sheet's name, column names, and row count; one window per logged event and Primary (the logged start to end, or the calendar day when no end is logged); the mask of the amendment |
| Sky's share at D12, read-out | The squared correlation of the change of the residual that leaves D12 out with D12's farm sensor on lit hours. The amendment bounds the share at 74% from dark hours alone |
| Add-back fit at D15, read-out | The lit-hour closure fit with a second regressor. Fit A adds the farm sensor of D15. Fit B adds the output change of the farm inside D15's polygon instead |

**Fit A has three pre-stated readings, and neither fit changes a verdict.** An interval of the slope
on D15's change inside 0.6 to 1.5 means that same-sky generation outside D15 explains the excess. A
lower end above 2.0 means an add-back inside D15's series, or generation that the farm sensor does
not follow. Anything else is unresolved. Fit B supports an add-back when the slope falls inside 0.6
to 1.5 and the coefficient on the farm's megawatts lies between −1 and 0. The readings decide how
the page describes D15 and whether the design of the later rung may cite D15's lit-hour slope.

**Job W2 scores the concordance in injected worlds.** Each of the 12 qualifying pairs is injected at
levels 1.0 and 2.0 into a host that reports a signed series, folded with a reactive floor of 2 points
and within-half-hour averaging, as in the amendment's version 2. D10 folded, D15 folded, and the six
un-injected signed hosts complete the worlds. The farm sensor binds, the meter sensor is scored
against the same bars and reported apart, and the CAMS sensor is the reference. A pass with the
meter sensor alone opens nothing. The page also reports the pooled area without D14, which holds half
of the 12 pairs. Its read-outs, which carry no bar and change no verdict, are the pooled fold ratio
of D10 on export hours and on import hours, the blind score of the meter sensor at folded D10, and
the dark-hour closure slopes. Job W2 takes under 0.05 core-hours.

**Job W3 runs only when job W2 is `supported`, and is not a further attempt.** Its three controls are
in the decision rule. Its read-outs have no bar: the shares of D12's one-stretch post-bounce days,
April to September, on which `C` is above +0.25, below −0.25, and in the dead band, for the farm
sensor and the meter sensor, and the pooled fold ratio of D12 on the same days with and without the
farm sensor. The pooled fold ratio at D12 is never a label for a day, a stretch, or a half-hour, and
never becomes a prior. Job W3 takes under 0.01 core-hours.

## What each verdict changes

**The stop rule: this is the 20th experiment that produces or gates a sign label, and it runs
once.** The design note for version 2 lists 18 earlier experiments by number: 79, 152, 358, 362,
365, 366, 370, 387, 408, 412, 417, 420, 424, 427, 506, 508, 543, and 674. Experiment 722 is the
19th. The stored BSP run is not counted. Experiment 722 was the fourth use of BSP closure for a sign,
and this experiment makes no fifth test of BSP closure.

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
| `invalid` | Does not run | A code defect is fixed and the job reruns as the same attempt; a failed sensor audit pauses the idea | `a` = 0 |
| `inconclusive` | Does not run | Paused | `a` = 0 |

**A closed idea runs in no variant again, and a paused idea runs again only on data that do not exist
today.** An example is a longer later span once the holdout is released, and a rerun needs the
maintainer's recorded approval.

## Compute and pre-run gates

**The experiment needs CPU only and under 0.1 core-hours in total.** The estimates scale from the
7.9 and 1.2 core-seconds of experiment 722's two jobs: 0.01 for job W0, 0.05 for job W2, and 0.01
for job W3. Each job runs inside one `systemd-run --user --scope` at a time, on one pinned core,
with `MemoryMax=12G`, `MemorySwapMax=0`, `OOMPolicy=continue`, one thread for BLAS and Polars,
`CUDA_VISIBLE_DEVICES=""`, and a check for 25 GB of free memory before launch. The timing gate of the
first-slice specification applies.

- **Gate 1:** a fresh Opus agent that did not write the code reviews it before any first run that
  produces a number the page quotes. The reporting fixes R1 to R3 of experiment 722's diagnosis go in
  with the new jobs.
- **Gate 2:** the session sends the manager the message "fits starting" before each job, and waits
  for the reply.
- **Gate 3:** the maintainer or the manager has ruled on decision 1 below.

**The experiment follows these earlier advice items:** do not rerun a closure test with its old bars,
check the size of a sign test before any run (job W0's power table), judge a placebo against a
null distribution (the permutation rule), check the CAMS time resolution before defining a cloud
class (the audit's first condition), and test closure on dark hours. It departs from none.

## Decisions for the maintainer or the manager

**Ten decisions belong to the maintainer or the manager, and the amendment takes none of them.** The
design amendment is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_design_v3_amendment_v2.md`, its Opus
design review is `data/ml-experiments/briefs/msa_architecture/mva_untangling_v3_amendment_review.md`,
the ruling on each finding of the review is
`data/ml-experiments/briefs/msa_architecture/mva_untangling_v3_triage.md`, and experiment 722's
diagnosis is `data/ml-experiments/briefs/msa_architecture/e722_u2a_diagnosis.md`. Version 2 of the
amendment has had no Opus review yet.

| No. | Decision | Recommendation |
|---|---|---|
| 1 | Whether this experiment runs, as the 20th experiment, in the trimmed form of jobs W0, W2, and W3 | Run it: the maintainer asked for the idea by name, and hourly CAMS cannot test it |
| 2 | The stop rule and the outcome table above, which replace the stop rules of the two earlier rungs in version 2 | Adopt both |
| 3 | Flaw 1: drop the D15 sign test and its placebo, with no replacement test at D10 | Drop them |
| 4 | Flaw 2: remove every size bar and keep dark-hour closure as a read-out | Remove the bars |
| 5 | Flaw 3: adopt the permutation rule for future generator records only, with the gate of 0.5 retained. A separate question is whether future tests set the detection limit against half the gap, as now, or against the full gap | Adopt the rule; the amendment names no preference on the gap and keeps half the gap as now |
| 6 | Flaw 4 and the within-day hint: remove the day class and audit conditions 3 and 4, adopt the farm sensor as the binding sensor, and admit metered farms outside the host's polygon as injected farms. The farms' output is a private regressor, and no page shows a farm's series | Adopt, and run job W2 on the farm sensor (option 3 of the amendment's section 6) |
| 7 | The form of the daylight hints in the later rung: "one factor at most", or one factor with a latent sky measured by two sensors. Needed only if job W2 is `supported` | The amendment states no recommendation; the choice can wait for the result |
| 8 | The CAMS 15-minute download of 22 or 154 requests | Decline for now; revisit only if the farm sensor fails its audit |
| 9 | Whether the build of the first buildable untangler may follow its known-answer test directly, because the hint rung has no hint to test with the BSP factor off and `a` = 0 | The amendment states no recommendation; its section 9 proceeds as if the answer is yes |
| 10 | The switching-log repair: `engine="openpyxl"` in this experiment's own code, with no change to `pyproject.toml` | Adopt the repair |

**The amendment rejects one suggestion of its review, and this experiment makes no request to NGED.**
The review recommended asking NGED for signed or reactive readings at D12. The maintainer has said
that NGED holds no information that the project does not already have.

## Method

### Architecture

- **Nothing is trained.** Every job is a regression, a correlation, or a count on stored series, and
  no neural network or hidden-state model is fitted.
- **The inputs are the power table through `studies.power.scan_power()`, the frame and sun elevation
  of experiment 39, the CAMS files of experiment 412, the site map of experiment 521, and the
  outputs of experiment 722's job 1,** with nothing on or after 2026-07-01. The experiment's own code
  imports experiment 722's loaders by `PYTHONPATH` and never copies them.
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
- **Each job's cost is under 0.05 core-hours,** and the experiment's total is under 0.1.
- **The switching-log workbook is read with `engine="openpyxl"`,** and a change of Primary `i` is
  dropped when either end lies inside a window of `i` or within 1 hour of a window's start or end.
  Every statistic carries the mask, and the same statistic without it is a read-out. When no window
  can be built, every statistic is labelled "transfers not masked".
- **No farm series is shown, and a page that publishes a posterior that used the farm sensor says
  that the posterior used farm output.**

### Measurement

- **The scored rows are post-bounce stretches with at least 8 usable pairs,** on days with exactly
  one such stretch, in the later span of 413 days.
- **The truth is the signed series of each host minus the injected farm's output,** and the
  statistic is the area under the receiver operating characteristic curve of `C`, pooled over pairs,
  levels, and hosts.
- **Every interval is the 2.5th to 97.5th percentile of 1,000 bootstrap draws that resample calendar
  days,** with every pair, level, and host taking its stretches from the drawn days.
- **The experiment scores no forecast, so no climatology row, mean absolute error, or leaderboard row
  applies.** It reads no load-shift event, and no standing practice that concerns fitted forecasts
  applies.
- **The verdicts are those of the decision rule, and the read-outs of the former job W1 can withhold
  and can never admit.**

## Files

Planned files only; no code exists yet, and the page lists each file as written when the results are
published.

- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_core.py`: the named
  constants, the bootstrap, the concordance and usable-pair rules, the permutation rule, and the
  decision rules.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w0.py`: job W0,
  which writes the power table, the sensor audit, the polygon-filtered pair table, the switching-log
  windows, and the two read-outs.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w2.py`: job W2,
  which builds the injected worlds, scores the concordance, and writes the results table and the
  verdict file.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/e00723_w3.py`: job W3,
  which runs the three controls and the read-outs only after job W2 is `supported`.
- `ml-experiments/00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc/run.sh`: the launch
  wrapper, which runs no job without `--go`.
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
  that governs this plan (read only, never published).
