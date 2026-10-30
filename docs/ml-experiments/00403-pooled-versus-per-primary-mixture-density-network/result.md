---
id: "00403"
slug: pooled-versus-per-primary-mixture-density-network
title: >-
  Does one pooled mixture density network beat one network per Primary for Primary demand, and where
  does the gain fall?
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
status: done
kind: other
protocol: demand-v2
size: >-
  simple. Stored: a small score table in this experiment's private folder; experiment 397's stored
  forecasts are read and never written. Production serving path: not touched (this repository holds
  none). Degradation rule: not touched. Designs: one defensible (a re-score of two stored forecast
  sets on identical rows). Callers: the script imports experiments 377, 381, and 87 and the
  `studies` evaluation code, whose callers sit in those folders. No fit runs.
hypothesis: >-
  One network trained across all 16 Primaries scores lower CRPS than one network per Primary, with
  the same inputs, rows, folds, and seeds, and the gain is largest at the Primaries with the
  shortest history.
prediction: >-
  Written before the contrast is computed. The pooled plain network is already known to score
  0.12 to 0.24 points lower than the per-Primary network in experiment 397's group means, so P1 is
  that pooled A0 minus the per-Primary network is negative at the signed-MW six and at the
  MVA-only eight, with a month-block 95% interval that excludes zero at the signed-MW six. P2: the
  gain correlates negatively with months of history across the 14 countable Primaries (a larger gain
  where the history is shorter), with a correlation below -0.3. P3: D15 and D18 gain more than the
  median countable Primary. P4: the gain does not concentrate: the best three Primaries carry less
  than 60% of the summed gain.
decision_rule: >-
  The contrast is pooled A0 minus the per-Primary plain network on CRPS at days 1 to 10, per group
  (signed-MW six; MVA-only eight, with and without D8; D15 and D18 report-only) and per Primary, by
  the month-block bootstrap of `studies.eval_protocol` at 95% (20,000 resamples). The five screen
  months give wide intervals, so every number is labelled interim and no claim of benefit rests on
  it. Pooling is supported if the signed-MW six and the MVA-only eight both have a negative
  difference with an interval below zero, and at least 10 of the 14 countable Primaries gain. A
  Primary-level interval that includes zero is reported as such. The smallest gain worth acting on
  is any gain whose interval excludes zero, because a pooled network costs less to train than many
  per-Primary networks and so needs no gain to pay for its complexity.
benchmark: >-
  The per-Primary plain network of experiment 397 stage 1, the pooled plain network of stage 2, and
  the stored monthly XGBoost forecasts of experiment 389, all on the same half-hours.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
verdict: supported
best_arm: >-
  Pooled A0 (one network across 16 Primaries, uniform input-group weights): CRPS 3.351 at the
  signed-MW six and 3.791 at the MVA-only eight, against 3.536 and 3.952 for the per-Primary plain
  network (days 1 to 10, five screen months).
headline: >-
  Pooled A0 minus the per-Primary plain network is -0.185 (95% interval -0.300 to -0.061) at the
  signed-MW six and -0.161 (-0.272 to -0.089) at the MVA-only eight, and 11 of the 14 countable
  Primaries gain. The best three Primaries carry 60% of the summed gain, and the gain does not
  correlate with months of history (-0.02). At D15 and D18 the pooled A0 loses 0.34 points together
  (D15 worse by 1.66, D18 better by 0.98), with an interval from -1.8 to +2.7.
lessons: >-
  Pooling helps in this screen, but not for the reason predicted: the Primaries with the shortest
  history do not gain more, and one short-history Primary (D15) loses badly under A0 while it gains
  under the pooled plain network.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  One neural network trained across all 16 Primaries forecasts demand better than 16 separate
  networks: at the six signed-direction substations the error score falls by 0.185 points of
  capacity (about 5%), at the eight apparent-power-only substations by 0.161 points (about 4%), and
  11 of 14 long-history substations gain. The gain does not grow where the history is shorter, and
  at the two short-history substations the result is mixed. These are five test months, so every
  interval is interim.
advice:
  - >-
    Prefer one pooled network to one network per Primary for further demand work, because it lowered
    the CRPS at both substation groups and costs one fit instead of 16.
  - >-
    Do not expect pooling to rescue substations with under 12 months of history, because the gain
    there was mixed (D18 gained 0.98 points, D15 lost 1.66 under A0).
  - >-
    Repeat the contrast on the full year before relying on it, because five test months give wide
    intervals at most single Primaries.
code: ml-experiments/00403-pooled-versus-per-primary-mixture-density-network/
---

**Take-home message: one pooled neural network beats one network per Primary at both substation
groups.** Pooled A0 (the pooled network with uniform input-group weights) lowers the CRPS at days 1
to 10 by 0.185 points of capacity at the six signed-MW Primaries (95% interval -0.300 to -0.061) and
by 0.161 points at the eight MVA-only Primaries (-0.272 to -0.089), and 11 of the 14 countable
Primaries gain. **The gain does not grow where the history is shorter** (correlation -0.02), so the
mechanism the prediction named is not what carries it. At D15 and D18 the result is mixed: D18 gains
0.98 points and D15 loses 1.66 under A0, while the pooled plain network gains at both. The verdict is
`supported (interim)`, because the pre-registered rule held on five test months.

![Pooling gain at each Primary](figure_1_pooling_gain.svg)

## Results

CRPS at days 1 to 10, mean over Primaries with equal weight, in points of capacity.

| Group | Per-Primary plain | Pooled plain | Pooled A0 | A0 minus per-Primary (95% interval) |
|---|---|---|---|---|
| Signed-MW six | 3.536 | 3.295 | 3.351 | -0.185 (-0.300 to -0.061) |
| MVA-only eight | 3.952 | 3.881 | 3.791 | -0.161 (-0.272 to -0.089) |
| MVA-only seven (no D8) | 3.336 | 3.294 | 3.227 | -0.109 (-0.124 to -0.090) |
| D15 and D18 (report-only) | 10.494 | 9.278 | 10.835 | +0.336 (-1.815 to +2.709) |

- Per Primary, the largest gains under A0 are D10 (-0.647, interval -0.798 to -0.478), D8 (-0.528),
  and D18 (-0.980). D2 (-0.073) and D21 (-0.131) gain with intervals that exclude zero.
- D11 (+0.017), D5 (+0.030), D3 (+0.103), and D15 (+1.663) do not gain, and none of their intervals
  excludes zero.
- The pooled plain network gains more than A0 at the signed-MW six (-0.241 against -0.185) and less
  at the MVA-only eight (-0.071 against -0.161).

## Predictions and decision rule

- **P1 held.** Both groups have negative differences with intervals below zero.
- **P2 failed.** The correlation of the gain with months of history is -0.02, not below -0.3.
- **P3 failed.** D15 loses under A0, so D15 and D18 do not both gain more than the median Primary.
- **P4 failed narrowly.** The best three Primaries carry 60% of the summed gain, and the prediction
  required under 60%.
- **The decision rule is met:** both group intervals lie below zero and 11 of 14 Primaries gain.

## Limits

- Five test months (2025-10, 2025-12, 2026-01, 2026-03, and 2026-06) give few resamples for the
  month-block bootstrap, so every interval is interim.
- A0 differs from the per-Primary network in two ways: pooling, and the five-token input encoder.
  The pooled plain network separates them: it gains 0.241 and 0.071 points at the two groups.
- The forecasts and scores are those of experiment 397; nothing was refitted.

## Problems caused by the move

None.
