# Load-shift labels in the forecast: research plan

**One proposal in this theme is waiting: experiment 712, which tests era labels made without
hindsight.** Experiment 693 found no gain from a perfect transfer flag,
but its refits were too noisy to rule out a gain or a loss of up to about half a point of capacity.
A repeat needs at least 10 refits per arm and fold, a placebo drawn from an independent world, and
real training rows given a matched share of non-zero flags.

**Experiment 00116 (differentiable physics, NRA) also bears on this theme.** It fits flows on the
interconnection edges between Primary substations and scores them on injected transfers; see the
[Network research plan](../differentiable-physics-network/plan.md).

## Priority table

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 712 | Do labels for permanent changes in demand, made by detectors and a language model from data available at the time, lower the demand-forecasting neural network's error after a permanent change without raising it after a temporary move of load between substations? | High (maintainer approval, 2026-11-20) | The plan's pull request must merge, and then the ML experiment manager must give the go. Stage 2 waits for the result of stage 1, which tests whether the detectors alone alarm early enough. | Planned |

## Planned experiments

<!-- planned-table -->
