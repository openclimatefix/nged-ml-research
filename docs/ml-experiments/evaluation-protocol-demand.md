# Evaluation protocol for Primary demand, version 1

**This page fixes how every experiment on Primary substation demand is scored, so that results from
different experiments can be compared.** An experiment records `protocol: demand-v1` in its front
matter. A change to any metric, fold, series list, slice, interval, or tolerance on this page makes
a new version. The superseded text moves to `evaluation-protocol-archive/` as `demand-v1.md`, and
the table of versions below gains a row. The protocols for [solar](evaluation-protocol-solar.md) and
[wind](evaluation-protocol-wind.md) have their own version numbers, so a change here does not move
theirs. The [research default rule](research-default.md) defines the research default and the log of
its versions. This page defines how a change to it is scored and when the change is adopted.

| Version | Date | Change |
|---|---|---|
| `demand-v1` | 2026-10-28 | First version |

## What is scored

**The series are the 16 Primaries, in three groups that are always reported separately.** The three
groups answer different questions, so no headline pools them.

| Group | Primaries | What the group tests |
|---|---|---|
| Signed-MW | D5, D9, D10, D11, D14, D16 | The demand model, because the meter records the sign of the flow |
| MVA-only | D2, D3, D4, D8, D12, D19, D21, D22 | The demand model plus the effect of hidden reverse flow, because the meter records apparent power with no sign |
| Short-history | D15, D18 | Report-only: no adoption rests on this group. No experiment we found fitted the GAM stack at D15 |

**D8 is a labelled exception.** D8 has no tuning month and borrows the second-stage spread of six
donor Primaries (D2, D3, D4, D19, D21, and D22). A forecast for D8 that uses donor spread is allowed
and is labelled as such in every table.

**Every number for the MVA-only group carries the label "reactive power assumed zero".** The rule
that picks the fit fails at a power factor of 0.97, and the reactive power of the real MVA-only
meters is unknown.

**The forecasts compared are the forecast under test, the current research default, and three fixed
references.** The references are BG2 (the XGBoost-based production-style forecast),
`manual_heuristic`, and `climatology`. The current research default and each
reference are regenerated on the folds and Primaries of this protocol, under new stored-forecast
names, because the existing stored forecasts cover six Primaries on another row set and are never
overwritten. Each reference is never retuned. Earlier scores of the references, which stop at 168
hours and use other row sets, are not comparable with scores under this protocol.

**The lead range is 0.5 to 240 hours, and every experiment scores all of it.** NGED's horizon is 1
to 10 days. Days 11 to 14 are reported where the forecast exists.

**The first run under this protocol produces the forecasts for 168 to 240 hours that the references
lack.** The stored demand forecasts stop at 168 hours, so BG2 has never been scored past 168 hours,
and BG2, the GAM stack, `manual_heuristic`, and `climatology` need new runs at those leads. The ML
MANAGER schedules this run. Until it has finished, no experiment scores the 168 to 240 hour leads,
and the days 8 to 10 and 11 to 14 slices are empty.

## The three fixed benchmarks

**Each family keeps three benchmarks whose recipes never change: `manual_heuristic`, `climatology`,
and one frozen XGBoost-based reference.** The rows grow as months arrive, and the benchmarks are
extended with the same recipe and never retuned. Every `change` page reports the research default's
absolute score against the three benchmarks, with intervals, because a chain of small contrasts can
hide a drift in absolute skill. Diagnostic pages need not report them.

**For Primary demand the frozen reference is BG2.** BG2 is the stored arm `p_s6_bg2` of [experiment
87](00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md). BG2 is a quantile blend of
five forecasts, with an explicit solar offset at the 5 of 16 Primaries that have a clean solar
capacity estimate. The five forecasts are a weekly-refitted XGBoost model, a monthly-refitted
XGBoost model, an analogue model (the experiment 87 page does not state its type), an XGBoost model
pooled by substation type, and the frozen stack of experiment 48. For the bulk and grid supply
points, BG2 is the weekly-refitted XGBoost model alone (`b_weekly`).

**Production does not run BG2.** Production serves a much weaker XGBoost model, stored as
`xgboost_baseline`. Experiment 87 scored it at 10.536 points on the Primary quantile score, against
5.260 for [BG2](glossary.md#bg2). A page that names BG2 as a benchmark says that production does not
run BG2.

**The GAM stack trails BG2 on sunny April-to-June half-hours.** [Experiment
251](00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) found the stack worse
than BG2 in April, May, and June (+0.162, +0.220, and +0.054 points), and at D5 on sunny half-hours.
Every Primary-demand `change` page reports this finding until a version of the research default
closes the gap.

## Folds

**Folds are contiguous months with a training window that ends before the first forecast origin of
the scored month.** Fold k trains on data up to the end of month M(k), then scores every daily
forecast run whose origin date falls in month M(k)+1, at every lead up to 240 hours. The publication
lag of the power data counts as part of the training cut. The row set is every daily ensemble run,
at the 00:00 UTC origin, in the scored period. It includes the runs that `fold_a_every5d_v1` holds,
and it gets a new `row_set_id` when the first experiment under this protocol stores its forecasts.
The publication lag of the power data is the lag in production, and `studies.eval_protocol` states
its value.

**The experimenter chooses the training regime and states it in the plan.** An expanding window and
a sliding window (for example, the last 24 months) are both allowed. The regime must be the same for
the forecast under test and for its comparison arm, unless the regime is the change under test.

**A fold needs at least 12 months of training history before it is scored.** The GAM has seasonal
terms, and a fold that has never seen a winter measures extrapolation, not the change under test.
Each Primary's report says which months were excluded by this rule.

**The scored period is 2025-07 to 2026-06, with a replicate period of 2024-07 to 2025-06
for the Primaries that have one.** Only a Primary with at least 12 months of history before the
replicate period has a replicate. Five long-history Primaries qualified at 2024-07-01 in
experiment 103, so the other Primaries have no replicate, and an adoption that needs a replicate
cannot rest on them. Rows inside hand-filtered events are excluded, as in the research default's row
set, except that a detected step day at a Primary like D8 is never excluded.

**Every Primary in a group is scored over the same months, so that a pooled group score averages
the same rows.** A Primary whose history is too short for a month is left out of that month for its
whole group, or the pooled score is labelled with the months it covers. D15 and D18 first have 12
months of history on 2025-03-01 (experiment 103), so they are scored from March 2026 for about four
months, with no replicate. A 95% month bootstrap on four months under-covers, so their numbers are
reported only and do not count in the tally of Primaries where the forecast under test wins. D9's
history starts on 2023-10-01, so it has no replicate months before 2024-10.

**The months that the 12-month rule excludes are reported as their own labelled slice.** Cold-start
months are where a longer fit history helps most, and a gain there is otherwise invisible.

**Stage-2 and filter forecasts for the tuning period are stored for every fold, except at D8.** The
GAM stack's
second stage and the Kalman filter need forecasts from before each scored month, and a missing store
blocks a rescore. Every experiment stores its per-run forecasts and its stage columns.

**No experiment scores a valid time on or after 2026-07-01.** This is an honour-system rule. A
replicate that scores later valid times is labelled "tuning-adjacent" and is not read as held-out
evidence. The rule truncates the last June origins, whose leads run past 2026-07-01. A physical lock
on the held-out data is a separate, later piece of work.

## Metrics

**The three scored metrics are the mean absolute error of the median, the continuous ranked
probability score (CRPS), and the tail-weighted CRPS (twCRPS).** CRPS is estimated as the quantile
score over 11 levels, which is the headline metric of the research default rule. The ensemble CRPS
of `studies.ensemble_scores` is a different estimator and is not used here. Pass or fail is judged
in points of effective capacity, where a point is 1% of a Primary's capacity, and every table shows
megawatts beside it, as the substation cards do. The headline metric is CRPS.

**Four tail measures and the curtailment cost are reported and have no pass or fail.** The tail
measures are pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles, per group and
per Primary. The curtailment cost is the existing avoidable-curtailment-cost measure.

**The implementing code is the package `studies.eval_protocol`.** The package lives in
`src/studies/eval_protocol/`, with the modules `metrics.py` and `slices.py`, and a separate pull
request adds it. Slice masks and the twCRPS threshold are defined there. The definitions are those
of the substation cards (sunny by clear-sky index, sunny by unit shape, cold nights at 3 °C or
below, and bank holidays). The twCRPS threshold is fixed per Primary and named in
`studies.eval_protocol`. The docstrings of the package are the definition, and a change to a mask
or a threshold makes a new version of this protocol. The package defines the
slices with the ERA5 2 m temperature at each series' location, and the ENS forecasts are scored
over those slices. The slice of a half-hour is therefore the same for every forecast, because it
depends on an estimate of the weather that happened and not on any forecast. ERA5 is an estimate of
past weather, so a half-hour near a threshold can sit in the wrong slice, equally for every arm.

**Experiment code must import the metric definitions from `studies.eval_protocol`.** A hard-coded
copy of a metric is a defect, because copies are how a changed definition stops applying to some
experiments.

## Slices

**Each metric is reported over the slices below, and every slice row shows its half-hour count.**
A slice with few half-hours is labelled "thin", as the substation cards do.

| Kind | Slices |
|---|---|
| Lead time | Day 1 (0 to 24 hours), day 2, days 3 to 4, days 5 to 7, days 8 to 10, days 11 to 14. Days 1 to 10 is the headline band |
| Time of day | Night, morning, midday, and evening, in local time, with the hour boundaries stated in `studies.eval_protocol` |
| Condition | All days; sunny by clear-sky index; sunny by unit shape; sunny April to June; the months excluded by the 12-month rule; winter (December to February); winter nights; cold nights (3 °C or below); Economy 7 step hours; Christmas (24 December to 1 January); bank holidays |
| Load-shift events | During each hand-labelled load-shift (switching) event; in the 7 days after each event ends |
| Group | Signed-MW, MVA-only, and short-history, each separately |
| Series | Every Primary on its own row |

**The two load-shift slices have no pass or fail.** The row set of the main score excludes the rows
inside hand-filtered events, so the scores during the events and in the 7 days after each event
ends are reported separately. Seven logged transfers are too few for an interval, per [experiment
277](00277-lsc-curtailment-metric-on-existing-forecasts/result.md). The length of 7 days is a choice
of `demand-v1`, which the maintainer may change in a later version. The event list is the
hand-filtered list and never the output of a detector.

**The tolerance for "sunny daylight half-hours" applies to the sunny-by-clear-sky-index slice.** The
other sunny slices are reported.

**The verdict names the lead band it rests on.** A verdict that rests on days 1 to 2 says so,
because NGED's horizon is 1 to 10 days.

## Reporting in every write-up

**Each write-up reports the load-shift slices beside the main score.** The write-up shows the score
during the events and the score in the 7 days after each event ends, for the forecast under test
and for the current research default, with the half-hour count of each slice. The write-up draws no
pass or fail from them.

**Each write-up carries a per-Primary table, the worst Primary, and a leave-one-out contrast.** The
table lists every Primary's paired difference and its 95% interval, sorted by gap. Beside the pooled
score the write-up shows the worst Primary and the number of Primaries where the forecast under
test wins. For each pooled group contrast, it also shows two leave-one-out contrasts: without the
Primary with the largest absolute paired difference, and without the Primary with the worst score.
The sign of a pooled contrast can change when one Primary is left out.

**Intervals are 95% intervals from the paired bootstrap that resamples whole calendar months.** Any
statistic across Primaries states the number of Primaries, because a correlation across 14 or 16
sites is weak evidence.

**The write-up names the research default's version and the protocol version.** For example,
"default plus x versus Primary demand research default 3, scored under `demand-v1`". The
`result.md` take-home stays at one or two paragraphs with figures, and the full metric table sits in
a collapsed section.

**A change aimed at one series or slice names at least two placebo Primaries in its plan.** A
placebo fails when it moves in the same direction as the target, beyond its own 95% interval. A
change with a failed placebo cannot be adopted as a general change.

## When a change is adopted

**These conditions apply only when an experiment asks to adopt a change.** Most experiments do not
ask. A change is adopted only when all six conditions hold. A change aimed at named Primaries
follows the targeted path below, and the other conditions still apply to it.

1. **Gain.** The 99% month-block interval of the contrast (default plus x, minus default) on the
CRPS of days 1 to 10 lies wholly on the better side of zero, and the point estimate is at least the
minimum gain for the change's tier. The contrast is read for the signed-MW group and for the
MVA-only group separately, and each group's verdict is stated on its own. With k pre-registered
arms, the interval is read at the level 1 − 0.01/k. The short-history group takes no part.
2. **No harm to a group.** The signed-MW group passes the gain condition. The MVA-only group stays
within the tolerance of its headline, with D8 and without D8, and is not required to show a gain.
Each protected slice stays within its tolerance in each group, judged by a one-sided 95%
non-inferiority bound. A breach at a group makes the verdict `rejected`.
3. **No harm to a Primary.** A Primary whose point estimate is worse than 0.05 points makes the
   verdict `needs-human`, with the Primary named. The interval is not tested at each Primary,
   because 16 one-sided tests would breach by chance more often than not.
4. **Replication.** The gain has the same sign in the replicate period at the Primaries that have
   one, and the replicate estimate is at least half of the smaller of the main estimate and the
   minimum gain. Where the minimum gain is zero, the condition reduces to the same sign. Leaving
   out one series at a time is a check for changes tuned on one series and does not stand in for a
   second period: a change with no second period stays a candidate. A change that fails replication
   is not retried on the same replicate period with a changed x, unless the change is pre-registered
   as a new experiment and the failed attempt is counted. A change that meets this condition
   satisfies audit trigger 7, and a candidate that fails it never becomes a base for later work.
5. **Controls.** The experiment's own controls passed, the placebos did not fail, and no audit
   trigger in the `ml-experiment` skill is unresolved. The audit's findings are triaged in the pull
   request before the adoption.
6. **Serving.** The change uses only inputs available at forecast time. If the change would need a
   serving-path change to port, the page says so.

**A change aimed at named Primaries has a targeted path.** The gain is judged on the named Primaries
instead of the group, because a gain at one Primary is diluted by 8 or 16 in a pooled score. The
placebos must stay non-inferior, and the change is adopted only as an automatic per-substation
choice. The choice follows a rule fixed at training time from earlier out-of-sample scores alone,
and the experiment scores the rule's output, never the better arm at each Primary in hindsight.

**The tolerances and minimum gains are fixed here and tweaked only by a new version.** A tolerance
is the harm a protected group may suffer before an adoption fails.

| Protected group | Tolerance, in points |
|---|---|
| Headline of each group (CRPS, days 1 to 10), including the MVA-only group with and without D8 | 0.02 |
| The 0.01 and 0.99 pinball tails, the worse end scored | 0.05 |
| Cold winter nights | 0.05 |
| Sunny daylight half-hours | 0.05 |
| twCRPS | 2% relative |
| Each lead band | 0.05 |
| Each Primary, on the point estimate | 0.05 |
| Per-Primary coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles | Flagged when the 95% month-resampled interval of the coverage excludes the nominal level; a flag does not fail the change |

| Tier of the change | Minimum gain, in points |
|---|---|
| A changed setting, an added column, or a changed loss | 0 |
| A new fitted step, a new input, or a new data source | 0.02 |
| A new head | 0.03 |

**The adoption outcome is `adopted`, `rejected`, or `needs-human`.** `adopted` sets `promoted_to`
in the front matter, and the outcomes `rejected` and `needs-human` appear in the first line of
`result.md` beside the existing `verdict`, with `production_status: candidate`.

**A neutral change passes a group's harm test with probability Φ(τ/SE − 1.645),** where τ is the
tolerance and SE is the standard error. A tolerance of 2 standard errors passes a neutral change 64%
of the time, and a change that truly harms the group by exactly the tolerance passes 5% of the time.
The per-Primary rule is a flag because 16 such tests would breach by chance far more often than not.

**When the metrics disagree, the verdict is decided in this order.** A breach of condition 2 makes
the verdict `rejected`, unless the change targets a different slice and the maintainer overrules.
A Primary breach of condition 3 makes the verdict `needs-human`. A CRPS gain with the median error
moving the other way beyond tolerance gives `needs-human`. `needs-human` is also the fallback for
any case this list does not cover.

**The first experiment under this protocol rescoring the references.** Experiments 374 and 376 used
one fixed fit and a tuning month, not monthly folds, so their scores are not comparable with scores
under this protocol.

**A screening arm is allowed.** Monthly retraining of 16 Primaries over 12 scored months costs about
384 stack fits per change for two arms. An experiment's plan may screen with fewer folds or
Primaries, and says so. Only a full run under this protocol supports an adoption.

**The cumulative harm budget is 2 times the group's tolerance, measured against the family's first
version.** The no-harm condition compares each adoption only with the previous version, so n
adoptions that each cost up to the tolerance could cost n times the tolerance in total. At each
adoption, the current research default must also stay within the budget against the first version,
using the first version's stored forecasts. The retest rules of the [research default
rule](research-default.md#the-rebase-rule-for-stale-results) still apply.

**A change that wins on the gain condition but fails another condition is recorded as a
candidate, not adopted.** Candidates are retested when the missing evidence arrives, or when a
rebase run is available, as the [research default rule](research-default.md) describes.

**A targeted adoption on a subgroup uses a short list of admissible subgroups, set in advance.** The
list is cold nights, the Economy 7 step, and sunny daylight. A targeted adoption on a subgroup also
needs no harm on the family headline.

**An experiment may hold several default-plus-x arms, and at most one of them is adopted.** Each arm
is read at the level 1 − 0.01/k, and the others become candidates that rebase on the new version. A
bundle of two components is adopted as one change only when it was pre-registered as one change and
its leave-one-out arms show neither component harmful.

**Only a pre-registered arm can be adopted.** A post hoc arm becomes a candidate that needs a fresh
experiment. When an experiment tests k variants of one change (tilt 20 and tilt 27, for example),
the best variant needs replication on its own after the Bonferroni read.

**An experiment that evaluates a blend reports each constituent model family alone beside the
blend.** The single-family table comes first, and the page states the blend's added value over the
best single family, with an interval. A blend is adopted into the research default only with that
table present. The first production update is a relatively simple model implemented as one model
family, for example XGBoost only, while blends stay of interest.

**The default plus x is produced on the research default's full row set.** Its stored forecasts then
serve as the next version without a separate regeneration job.

**The ablation that follows every fifth adoption drops a component only on evidence.** A component
is dropped only when the interval of its removal lies wholly on the better side, or when its
removal is non-inferior within the tolerance and the component sits in a higher complexity tier.

**The nominal 99% interval covers less than 99% on short periods.** A percentile bootstrap that
resampled independent normal months with a nominal 99% interval covered 96.5% of the time at 12
months, 92.0% at 6 months, and 83.7% at 4 months (1,500 trials of 2,000 resamples each). The real
procedure also resamples seeds and has autocorrelated months, so the figures are indicative. For
periods under 12 months, add a t test on the monthly paired differences or use a studentised
bootstrap. Adoption contrasts use 99% whatever the number of arms, as a stated exception to the
[interval-level convention](glossary.md#interval-level-convention), and a `change` page says "the
99% interval lies wholly on the better side of zero" in place of the `study` skill's
"statistically significant at the 5% level".

**The recorded gain of each version is an upper bound.** Winners are selected on the scored year, so
the recorded estimate overstates the true gain. In a simulation, a true gain of 2 standard errors
was recorded on average as 2.7. The log therefore also reports the cumulative gain on the replicate
period, which carries no selection bias, beside the scored-year sum.

## The holdout

**The frozen period is the existing holdout: valid times on or after 2026-07-01.** No experiment
scores those valid times, which are kept for a finalist. The `release-model` skill scores the
holdout once per release, on the sign of the effect only, and a month may confirm one release only.

**The holdout holds under four months and no winter on 2026-10-23.** It covers July to October 2026,
and the months from November to February arrive by about March 2027. Until then, a change that acts
in winter (the heating curve, cold nights, the Economy 7 step) is adopted on the replicate year
alone, and its log entry reads "winter-unconfirmed". The holdout confirms its sign in March 2027 or
later.

**The same-months shrinkage check pauses a family's adoptions only on evidence of shrinkage.** The
check compares the holdout months with the same calendar months of the scored year (July to October
2025 for the current holdout). Adoptions in the family pause when the one-sided 90% upper bound of
the holdout months' cumulative gain lies below half of the cumulative gain on those same scored-year
months. A month read for a decision, or for a pause investigation, joins the scored months, and the
holdout rolls forward as new months arrive. In a simulation of a 12-month scored year and a
4-month holdout, the check paused the family after 29% of false adoptions, 7% of adoptions with a
true gain of 2 standard errors, and 3% of adoptions with a true gain of 3 standard errors. The
check understates the chance of a pause for a change whose effect is seasonal.

## Retraining

**Retraining is monthly for the GAM stack, and weekly only for BG2 and the production forecast.**
There is no separate screening tier, because monthly retraining is already the R&D practice.

## Known-answer tests

**The protocol requires tests on synthetic series whose correct scores are known.** The pull
request that adds `studies.eval_protocol` adds them, and they run in continuous integration.
They cover each metric, the month bootstrap, the folds (including the 12-month rule and the training
cut), and the slice masks. A change to the evaluation code that breaks a known answer fails the
build.

## What this page does not cover

- A physical lock that stops research sessions reading the held-out actuals.
- Logging scores to MLflow, which a follow-on pull request adds.
- Wind and solar, whose own protocols are linked at the top of this page.
- The bulk and grid supply points. Their row set is `fold_a_daily_gap_v1`, their lead range is to
  be named with their first research default, and no adoption happens until then.
- Scores before `demand-v1`. Those scores used `fold_a_daily_gap_v1` on six Primaries, a scored
  year of 2025-07-02 to 2026-06-30, a replicate year of 2024-25, and a lead range of 0.5 to 168
  hours.
