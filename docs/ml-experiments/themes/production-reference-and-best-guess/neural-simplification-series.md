# Simplifying the neural demand, wind, and solar forecasters: series result

**Of the four demand input and output prunings tested, only dropping the 7 ensemble-spread inputs
was equivalent to the baseline network at both metering groups, and that pruning loses on the rows
where the forecast tails matter, so the final assembly step was not run.** The pruning is
[experiment 496](../../00496-simplify-demand-network-input-and-output-pruning/result.md). Scores
are CRPS (continuous ranked probability score, in points of effective capacity; lower is better).
On rows with no labelled event, the change was +0.012 at the eight and -0.043 at the six, both
inside the equivalence limit. On cold winter nights at the six, the network without spread inputs
scored 0.187 points worse, and on the six's rows with a labelled event it scored 0.408 points
worse. The whole-sample equivalence claim does not show those losses.

**Terms.** A Primary is a primary substation. "The eight" are the 8 Primaries whose telemetry is in
megavolt-amperes only, and "the six" are the 6 Primaries whose telemetry is in signed megawatts
(MW). NGED's event labels are incomplete, so "rows with no labelled event" may still hold
unlabelled events. A seed is one random initialisation of a neural network, a refit is a fresh
training run on newer data, and fine-tuning continues training a pretrained network on archived
ensemble weather forecasts. q95 and q99 are the 95% and 99% points of the forecast distribution,
and coverage is the share of actuals at or below a quantile. Table P is the recalibration table of
widening factors per metering group, lead band, and tail quantile from
[experiment 483](../../00483-uncertainty-rerun-on-the-first-release-demand-candidates/result.md).
Solar A to F are the six anonymised metered solar farms.

**Scope.** Most numbers come from the five demand months of experiment 449 (2025-10, 2025-12,
2026-01, 2026-03, 2026-06). Experiment 492 used four of those folds, and the solar experiments 493
and 498 used 12 scored months. Every score is selection-biased, and the demand results are interim.

## Serving table

**The candidate serving settings below need fewer networks and refits than the baselines. They keep
the inputs and the recalibration table that the experiments did not show to be removable.** No page
records these settings as adopted in a production release.

| Family | Candidate setting | Evidence |
|---|---|---|
| Demand: networks per serving pass | 2 seeds | [Experiment 481](../../00481-seed-averaging-diminishing-returns-for-demand-and-wind/result.md): a second seed scores 0.035 points better than one; later seeds gain at most 0.011 each (5 months). |
| Demand: refits | Pretrain once, fine-tune monthly | [Experiment 494](../../00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md): equivalent to a fresh refit at the eight and the six over 5 months; saves about 42 of the 80 GPU seconds per seed-refit. |
| Demand: inputs | Keep the 7 ensemble-spread inputs | [Experiment 496](../../00496-simplify-demand-network-input-and-output-pruning/result.md): dropping them scores +0.187 on cold winter nights at the six. |
| Demand: recalibration | Keep table P | [Experiment 492](../../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md): two constants match table P at the eight only. [Experiment 500](../../00500-rolling-conformal-recalibration-of-the-demand-tails/result.md): none of the three pre-registered rolling conformal rules passes the q95 floor at the six. |
| Wind: networks per serving pass | 1 seed | [Experiment 481](../../00481-seed-averaging-diminishing-returns-for-demand-and-wind/result.md): one seed meets the pre-registered rule; a second seed gains 0.041 points. |
| Solar: refits | Annual, not quarterly | [Experiment 498](../../00498-solar-e-annual-refit-schedule-only-read/result.md): for the encoder at Solar E, annual refits score +0.26% (10 seeds), inside the 1% limit. [Experiment 493](../../00493-solar-annual-against-quarterly-refits/result.md): both networks tested pass at the 1% limit. |
| Solar: network design | Keep the 51-member encoder (`a2_member`) | [Experiment 493](../../00493-solar-annual-against-quarterly-refits/result.md): the one-pass network with annual refits scores 1.28% worse than the encoder with quarterly refits at Solar E, mostly from the network design. |

## Tried and not adopted

**Each of these was a candidate simplification, and none was shown to lose nothing.**

- **Distilling the five-network pool into one student network** ([experiment
  489](../../00489-distill-the-five-network-pool-into-one-network/result.md)): not supported. At
  the eight the students score within 0.006 of a single seed (4.762 to 4.770, against 4.764 for the
  single seed and 4.680 for the pool).
- **Distilling the lagged-plus-lagless mixture into one lag-robust network** ([experiment
  490](../../00490-one-lag-robust-network-distilled-from-the-lagged-plus-lagless-mixture/result.md)):
  not supported. On rows with no labelled event at the six, the student scores 3.79 points against
  3.51 for the mixture of one lagged network and the lagless forecast (B-mix), a gap of 0.28.
- **Averaging weights during fine-tuning** ([experiment
  491](../../00491-weight-averaging-in-demand-network-fine-tuning/result.md)): gains almost
  nothing. At the eight, an exponential moving average scores 4.760 and a stochastic weight average
  4.757, against 4.762 for one ordinary network and 4.680 for the pool.
- **Two constants per metering group** ([experiment
  492](../../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md)): matches
  table P at the eight; at the six, q95 coverage is 0.011 to 0.013 lower.
- **Two constants per farm for wind and solar** ([experiment
  499](../../00499-two-constant-tail-widening-in-place-of-table-p-for-wind-and-solar/result.md)):
  inconclusive. The pre-registered rule needs the minimum detectable effect to lie below each
  limit, and it does not, except for wind q99 coverage. Wind coverage stays within 0.004 of table P
  at every tail, and solar per-farm constants within 0.005, but the stored year is too short to
  show equivalence.
- **Rolling conformal recalibration** ([experiment
  500](../../00500-rolling-conformal-recalibration-of-the-demand-tails/result.md)): not supported.
  With buffers of 7, 14, and 28 days, q95 coverage at the six stays between 0.874 and 0.913, below
  the 0.93 floor. A rule added after the scores were seen, which shifts all 11 quantiles by the
  median error of the last 28 days, brings q95 to 0.935 and q99 to 0.988, but most of its CRPS
  gain comes from one Primary.
- **Three other demand prunings** ([experiment
  496](../../00496-simplify-demand-network-input-and-output-pruning/result.md)): none equivalent at
  both the eight and the six. The CRPS change at the eight and the six is +0.095 and +0.229 for
  merging the four holiday columns into one day-type column, +0.152 and +0.082 for one Gaussian
  instead of three, and +0.079 and +0.050 for dropping the weekly lag. Dropping the weekly lag also
  scored 0.167 worse on cold winter nights at the six.
