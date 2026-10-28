# Evaluation protocol for Primary demand, version 1

**This page fixes how every experiment on Primary substation demand is scored, so that results from
different experiments can be compared.** An experiment records `protocol: demand-v1` in its front
matter. A change to any metric, fold, series list, slice, interval, or tolerance on this page makes
a new version. The superseded text moves to `evaluation-protocol-archive/` as `demand-v1.md`, and the
table of versions below gains a row. The protocols for [solar](evaluation-protocol-solar.md) and
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
| Signed-MW | D5, D9, D10, D11, D14, D16 | The architecture, because the meter records the sign of the flow |
| MVA-only | D2, D3, D4, D8, D12, D19, D21, D22 | The architecture plus the cost of hidden reverse flow, because the meter records apparent power with no sign |
| Short-history | D15, D18 | Behaviour on a short history; the GAM stack was never fitted at D15 |

**Every number for the MVA-only group carries the label "reactive power assumed zero".** The rule
that picks the fit fails at a power factor of 0.97, and the reactive power of the real MVA-only
meters is unknown.

**The forecasts compared are the forecast under test and four fixed references.** The references
are BG2 (the XGBoost-based production-style forecast), the best-guess GAM stack of the current
research default, `manual_heuristic`, and `climatology`. Each reference is rescored under this
protocol when the protocol starts, and is never retuned.

**The lead range is 0.5 to 240 hours, and every experiment scores all of it.** NGED's horizon is 1
to 10 days. Days 11 to 14 are reported where the forecast exists.

## Folds

**Folds are contiguous months with a training window that ends before the first forecast origin of
the scored month.** Fold k trains on data up to the end of month M(k), then scores every daily
forecast run whose valid times fall in month M(k)+1. The publication lag of the power data counts
as part of the training cut. The row set is every daily ensemble run in the scored period, and it
gets a new `row_set_id` when the first experiment under this protocol stores its forecasts.

**The experimenter chooses the training regime and states it in the plan.** An expanding window and
a sliding window (for example, the last 24 months) are both allowed. The regime must be the same for
the forecast under test and for its comparison arm, unless the regime is the change under test.

**A fold needs at least 12 months of training history before it is scored.** The GAM has seasonal
terms, and a fold that has never seen a winter measures extrapolation, not the change under test.
Each Primary's report says which months were excluded by this rule.

**The scored periods are 2025-07-02 to 2026-06-30, and a replicate period of 2024-07 to 2025-06.**
Rows inside hand-filtered events are excluded, as in the research default's row set.

**Stage-2 and filter forecasts for the tuning period are stored for every fold.** The GAM stack's
second stage and the Kalman filter need forecasts from before each scored month, and a missing store
blocks a rescore. Every experiment stores its per-run forecasts and its stage columns.

**No experiment scores a valid time on or after 2026-07-01.** This is an honour-system rule. A
replicate that scores later valid times is labelled "tuning-adjacent" and is not read as held-out
evidence. A physical lock on the held-out data is a separate, later piece of work.

## Metrics

**The three scored metrics are the mean absolute error of the median, the continuous ranked
probability score (CRPS), and the tail-weighted CRPS (twCRPS).** CRPS is estimated with the
quantile score over 11 levels. Pass or fail is judged in points of effective capacity, where a point
is 1% of a Primary's capacity, and every table shows megawatts beside it, as the substation cards
do. The headline metric is CRPS.

**Four tail measures and the curtailment cost are reported and have no pass or fail.** The tail
measures are pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles, per group and
per Primary. The curtailment cost is the existing avoidable-curtailment-cost measure.

**Experiment code imports the metric definitions from one module.** A hard-coded copy of a metric is
a defect, because copies are how a changed definition stops applying to some experiments.

## Slices

**Each metric is reported over the slices below, and every slice row shows its half-hour count.**
A slice with few half-hours is labelled "thin", as the substation cards do.

| Kind | Slices |
|---|---|
| Lead time | Day 1 (0 to 24 hours), day 2, days 3 to 4, days 5 to 7, days 8 to 10, days 11 to 14. Days 1 to 10 is the headline band |
| Condition | All days; sunny by clear-sky index; sunny by unit shape; sunny April to June; winter (December to February); winter nights; cold nights (3 °C or below); Economy 7 step hours; Christmas (24 December to 1 January); bank holidays |
| Group | Signed-MW, MVA-only, and short-history, each separately |
| Series | Every Primary on its own row |

**The verdict names the lead band it rests on.** Several results so far stopped at short leads, so
a verdict that rests on days 1 to 2 says so.

## Reporting in every write-up

**Each write-up carries a per-Primary table, the worst Primary, and a leave-one-out contrast.** The
table lists every Primary's paired difference and its 95% interval, sorted by gap. Beside the pooled
score the write-up shows the worst Primary and the number of Primaries where the forecast under
test wins. For each pooled MVA-only contrast, it also shows the contrast without the one dominant
Primary, because the sign of a pooled contrast has flipped when one Primary was left out.

**Intervals are 95% intervals from the paired bootstrap that resamples whole calendar months.** Any
statistic across Primaries states the number of Primaries, because a correlation across 14 or 16
sites is weak evidence.

**The write-up names the research default's version and the protocol version.** For example,
"default plus x versus Primary demand research default 3, scored under `demand-v1`". The
`result.md` take-home stays at one or two paragraphs with figures, and the full metric table sits in
a collapsed section.

**A change aimed at one series or slice names at least two placebo Primaries in its plan.** The
write-up records whether the placebos moved. A verdict that ignores a failed placebo is
overconfident.

## When a change is adopted

**This section replaces the five adoption conditions on the research default rule page, and applies
only when an experiment asks to adopt a change.** Most experiments do not ask. A change is adopted
only when all six conditions hold.

1. **Gain.** The 95% interval of the contrast (default plus x, minus default) on the CRPS of days 1
   to 10 lies wholly on the better side of zero, and the point estimate is at least the minimum gain
   for the change's tier. With k pre-registered arms, the interval is read at the level 1 − 0.05/k.
2. **No harm to a group.** The signed-MW group passes, and the MVA-only group stays within
   tolerance. Each group's verdict is stated separately, and no verdict is one word across groups.
3. **No harm to a protected slice or a Primary.** Each protected slice and each Primary stays within
   its tolerance, judged by a one-sided 95% non-inferiority bound. A breach at any Primary makes the
   verdict `needs-human`, with the Primary named, and never a silent `adopted`. A change that helps
   one Primary and is neutral elsewhere is a candidate for an automatic per-substation choice.
4. **Replication.** The gain has the same sign in the replicate period, and the replicate estimate
   is at least half of the smaller of the main estimate and the minimum gain. A change that fails
   replication is not retried on the same replicate period with a tweaked x, unless the tweak is
   pre-registered as a new experiment and the failed attempt is counted.
5. **Controls.** The experiment's own controls passed, the placebos are recorded, and no audit
   trigger in the `ml-experiment` skill is unresolved.
6. **Serving.** The change uses only inputs available at forecast time.

**The tolerances and minimum gains are fixed here and tweaked only by a new version.** A tolerance
is the harm a protected group may suffer before an adoption fails.

| Protected group | Tolerance, in points |
|---|---|
| Pooled headline (CRPS, days 1 to 10) | 0.02 |
| The 0.01 and 0.99 pinball tails, the worse end scored | 0.05 |
| Cold winter nights | 0.05 |
| Sunny daylight half-hours | 0.05 |
| twCRPS | 2% relative |
| Each lead band, and each Primary | 0.05 |

| Tier of the change | Minimum gain, in points |
|---|---|
| A changed setting, an added column, or a changed loss | 0 |
| A new fitted step, a new input, or a new data source | 0.02 |
| A new head | 0.03 |

**When the metrics disagree, the verdict is decided in this order.** A breach of condition 2 or 3
makes the verdict `rejected`, unless the change targets a different slice and the maintainer
overrules. A CRPS gain with the median error moving the other way beyond tolerance gives
`needs-human`. `needs-human` is also the fallback for any case this list does not cover.

**The cumulative harm budget and the retest rules on the research default rule page still apply.**
The budget is 2 times a group's tolerance, measured against the family's first version.

## Retraining

**The research default's retraining is monthly for the GAM stack, and weekly only for BG2 and the
production forecast.** There is no separate screening tier, because monthly retraining is already
the R&D practice and the GAM has shown little gain from weekly refits.

## Known-answer tests

**The protocol is tested on synthetic series whose correct scores are known.** The tests run in
continuous integration. They cover each metric, the month bootstrap, the folds (including the
12-month rule and the training cut), and the slice masks. A change to the evaluation code that
breaks a known answer fails the build.

## What this page does not cover

- A physical lock that stops research sessions reading the held-out actuals.
- Logging scores to MLflow, which a follow-on pull request adds.
- Wind and solar, whose own protocols are linked at the top of this page.
