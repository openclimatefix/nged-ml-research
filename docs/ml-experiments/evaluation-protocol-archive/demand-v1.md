<!-- Frozen copy of demand-v1, kept so that the solar and wind protocols can link a pinned text. -->

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

## Folds

**Folds are contiguous months with a training window that ends before the first forecast origin of
the scored month.** Fold k trains on data up to the end of month M(k), then scores every daily
forecast run whose origin date falls in month M(k)+1, at every lead up to 240 hours. The publication
lag of the power data counts as part of the training cut. The row set is every daily ensemble run,
at the 00:00 UTC origin, in the scored period. It includes the runs that `fold_a_every5d_v1` holds,
and it gets a new `row_set_id` when the first experiment under this protocol stores its forecasts.
The publication lag of the power data is the lag in production, and the shared module states its
value.

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

**Slice masks and the twCRPS threshold are defined in the shared metric module.** The definitions
are those of the substation cards (sunny by clear-sky index, sunny by unit shape, cold nights at
3 °C or below, and bank holidays). The twCRPS threshold is fixed per Primary and named in the
module. The module's docstrings are the definition, and a change to a mask or a threshold makes a
new version of this protocol. The pull request that adds the module names the temperature source and
the curtailment-cost function.

**Experiment code must import the metric definitions from one module.** The pull request that
implements this protocol names the module. A hard-coded copy of a metric is a defect, because
copies are how a changed definition stops applying to some experiments.

## Slices

**Each metric is reported over the slices below, and every slice row shows its half-hour count.**
A slice with few half-hours is labelled "thin", as the substation cards do.

| Kind | Slices |
|---|---|
| Lead time | Day 1 (0 to 24 hours), day 2, days 3 to 4, days 5 to 7, days 8 to 10, days 11 to 14. Days 1 to 10 is the headline band |
| Time of day | Night, morning, midday, and evening, in local time, with the hour boundaries stated in the shared module |
| Condition | All days; sunny by clear-sky index; sunny by unit shape; sunny April to June; the months excluded by the 12-month rule; winter (December to February); winter nights; cold nights (3 °C or below); Economy 7 step hours; Christmas (24 December to 1 January); bank holidays |
| Group | Signed-MW, MVA-only, and short-history, each separately |
| Series | Every Primary on its own row |

**The tolerance for "sunny daylight half-hours" applies to the sunny-by-clear-sky-index slice.** The
other sunny slices are reported.

**The verdict names the lead band it rests on.** A verdict that rests on days 1 to 2 says so,
because NGED's horizon is 1 to 10 days.

## Reporting in every write-up

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

**The cumulative harm budget and the retest rules on the research default rule page still apply.**
The budget is 2 times a group's tolerance, measured against the family's first version.

## Retraining

**Retraining is monthly for the GAM stack, and weekly only for BG2 and the production forecast.**
There is no separate screening tier, because monthly retraining is already the R&D practice.

## Known-answer tests

**The protocol requires tests on synthetic series whose correct scores are known.** The pull
request that implements the shared metric module adds them, and they run in continuous integration.
They cover each metric, the month bootstrap, the folds (including the 12-month rule and the training
cut), and the slice masks. A change to the evaluation code that breaks a known answer fails the
build.

## What this page does not cover

- A physical lock that stops research sessions reading the held-out actuals.
- Logging scores to MLflow, which a follow-on pull request adds.
- Wind and solar, whose own protocols are linked at the top of this page.
