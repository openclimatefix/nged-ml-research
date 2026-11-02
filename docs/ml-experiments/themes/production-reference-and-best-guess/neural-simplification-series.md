# Simplifying the neural demand, wind, and solar forecasters: series result

**The series found one simplification that held on the whole scored sample, and it loses where the
tails matter, so the series stops and the serving table below stays as it is.** The simplification
is dropping the 7 ensemble-spread inputs of the demand neural network ([experiment
496](../../00496-simplify-demand-network-input-and-output-pruning/result.md)). On the whole scored
sample its change in CRPS (continuous ranked probability score, in points of effective capacity;
lower is better) was inside the equivalence limit. On cold winter nights at the 6 Primary
substations whose telemetry is in signed megawatts ("the six"), it scored 0.187 points worse, and on
those 6 Primaries' rows with a labelled event it scored 0.408 points worse. The other pruning arms
were not shown equivalent. The step that would assemble the passing simplifications into one network
is therefore not run.

NGED's event labels are incomplete, so "rows with no labelled event" may still hold unlabelled
events. "The eight" are the 8 Primaries whose telemetry is in megavolt-amperes only. Every number
below is scored on the folds of the experiment it comes from, mostly the five months of experiment
449 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), so each is selection-biased and interim.

## Serving table

**The first-release demand, wind, and solar forecasters serve with fewer networks and refits than
their baselines, and keep the inputs and the recalibration table that the experiments did not show
to be removable.**

| Family | Decision | Evidence |
|---|---|---|
| Demand: networks per serving pass | 2 seeds (K = 2) | [Experiment 481](../../00481-seed-averaging-diminishing-returns-for-demand-and-wind/result.md): a second seed scores 0.035 points better than one; later seeds gain at most 0.011 each (5 months). |
| Demand: refits | Pretrain once, fine-tune monthly | [Experiment 494](../../00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md): equivalent to a fresh refit at the eight and the six; saves about 42 of the 80 GPU seconds per seed-refit. |
| Demand: inputs | Keep the 7 ensemble-spread inputs | [Experiment 496](../../00496-simplify-demand-network-input-and-output-pruning/result.md): dropping them scores +0.187 on cold winter nights at the six. |
| Demand: recalibration | Keep table P | [Experiment 492](../../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md): two constants match table P at the eight only. [Experiment 500](../../00500-rolling-conformal-recalibration-of-the-demand-tails/result.md): no rolling conformal rule passes the q95 floor at the six. |
| Wind: networks per serving pass | 1 seed (K = 1) | [Experiment 481](../../00481-seed-averaging-diminishing-returns-for-demand-and-wind/result.md): one seed meets the pre-registered rule; a second seed gains 0.041 points. |
| Solar: refits | Annual, not quarterly | [Experiment 493](../../00493-solar-annual-against-quarterly-refits/result.md) and [experiment 498](../../00498-solar-e-annual-refit-schedule-only-read/result.md): annual refits score +0.26% at Solar E, inside the 1% limit. |
| Solar: member encoder | Keep `a2_member` | [Experiment 493](../../00493-solar-annual-against-quarterly-refits/result.md): the one-pass network with annual refits scores 1.3% worse at Solar E, mostly from the network design. |

Table P is the recalibration table of widening factors per metering group, lead band, and tail
quantile, described in [experiment
483](../../00483-uncertainty-rerun-on-the-first-release-demand-candidates/result.md).

## Tried and not adopted

**Each of these was a candidate simplification, and none was shown to lose nothing.**

- **Distilling the five-network pool into one network** ([experiment
  489](../../00489-distill-the-five-network-pool-into-one-network/result.md)): not supported. At the
  eight the students score within 0.006 of a single seed (4.762 to 4.770, against 4.764 for the
  single seed and 4.680 for the pool).
- **Distilling the lagged-plus-lagless mixture into one lag-robust network** ([experiment
  490](../../00490-one-lag-robust-network-distilled-from-the-lagged-plus-lagless-mixture/result.md)):
  not supported. On rows with no labelled event at the six, the student scores 3.79 points against
  3.51 for the mixture B-mix, a gap of 0.28.
- **Averaging weights during fine-tuning** ([experiment
  491](../../00491-weight-averaging-in-demand-network-fine-tuning/result.md)): gains almost nothing.
  At the eight, an exponential moving average scores 4.760 and a stochastic weight average 4.757,
  against 4.762 for one ordinary network and 4.680 for the pool.
- **Two constants per metering group** ([experiment
  492](../../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md)): matches
  table P at the eight; at the six, q95 coverage is 0.011 to 0.013 lower.
- **Two constants per farm for wind and solar** ([experiment
  499](../../00499-two-constant-tail-widening-in-place-of-table-p-for-wind-and-solar/result.md)):
  inconclusive. Wind coverage is within 0.004 of table P at every tail, and solar per-farm constants
  are within 0.005, but the stored year is too short for the minimum detectable effect to fall below
  each limit.
- **Rolling conformal recalibration** ([experiment
  500](../../00500-rolling-conformal-recalibration-of-the-demand-tails/result.md)): not supported.
  At the six, q95 coverage stays between 0.874 and 0.913, below the 0.93 floor, for every buffer
  length.
- **Merging the four holiday columns into one day-type column, one Gaussian instead of three, and
  dropping the weekly lag** ([experiment
  496](../../00496-simplify-demand-network-input-and-output-pruning/result.md)): none shown
  equivalent. The CRPS change at the eight and the six is +0.095 and +0.229 for the holiday columns,
  +0.152 and +0.082 for one Gaussian, and +0.079 and +0.050 for the weekly lag.
