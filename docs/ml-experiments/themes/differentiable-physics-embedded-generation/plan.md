# Differentiable physics: embedded generation: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Wind node | A typed wind node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| Dispatchable node | A typed dispatchable-generator node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| Solar node | A typed solar-farm node beside the flat generator node | Proposed | A flat node that stays off where no generator exists (experiment 110 found it reads 12 to 35 points at all five decision substations) | Not started |
| 265, gas engine bands | Can an outside prior and a causal visibility switch make D8's widened bands honest in every quarter, and do they harm Primaries with no engine? | Done | None | Finished (not supported): In this year of forecasts, the other five Primaries' residuals did not predict D8's errors, so a starting estimate from them left D8's bands too narrow in the first quarter, and returning to normal widening after the engine's drops faded made them too narrow again. See [experiment 265](../../00265-dp-gas-engine-bands-prior-and-visibility-switch/result.md). |
| 352, capacity response surface | How much does the sunny-half-hour score of Primary demand research default 3 move with the solar capacity, with and without the Kalman filter? | Done | None | Finished (partly supported): The demand forecast's score on sunny half-hours moved by only 0.010 points (-0.104 to +0.088) when the assumed solar capacity was 20% too low, but rose by 0.124 points when it was 20% too high and by 0.358 when it was 40% too high, so a capacity estimate above the pin is the risk and a lower capacity estimate is not. See [experiment 352](../../00352-primary-demand-default-3-capacity-response-surface/result.md). |
| 409, solar term with a day-of-year home | Can the explicit solar term read the sky once the network has a day-of-year by hour term and the known-answer test injects only the farm's day-to-day anomaly, and does it then lower the net-demand CRPS? | Done | Experiment 405 (merged, inconclusive) | Finished (not supported): Adding an explicit solar-power term to the neural network did not reliably lower the net-demand error at the six signed-direction substations: the change was about -0.06 points of capacity (about 2%), with an interval that includes zero. See [experiment 409](../../00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/result.md). |
| 477, neural PV module placebo gate | Does the PV module of the net demand neural network recover a placebo-gated capacity per Primary, and what does the explicit split cost and gain on sunny half-hours? | Done | Experiments 416 (merged) and 474 | Finished: placebo gate failed |
| N13, PV orientation mix | Does replacing the single south-facing solar shape with three non-negative capacities (south, east, west) keep the model convex and lower the summer morning and evening error? | Dropped | None. [Experiment 73](../../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) and [experiment 77](../../00077-pvdis-6b-fitted-solar-shape-in-arm-am/result.md) found the mix unidentified or immaterial, and experiment 114 dropped it | Dropped |
| [674](../../00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/result.md) | Can a per-Primary day labeller tell whether an apparent-power Primary was exporting, and does a soft or a hard PV prior help? | Done | Nothing | Finished (inconclusive): A day labeller that reads the lowest reading at the edges of the sun's path and a W-shape score finds injected solar export (64% of days at 1.5 to 3 times the night import, 31% below 3 times) but misses 54 of 59 real reverse-flow days, which are shallow and short. |

## N13: PV orientation mix

**Experiment 114 dropped N13, so the design below is not queued.** It stays as the design for
any later experiment on solar orientation.

**The proposal gives the convex model three non-negative solar capacities, one each for south, east,
and west, in place of one south-facing unit shape.** The model stays convex, and an east-west array
should get a larger east and west share. The shares count as identified only if they pass the
coverage rule of experiment 113, which asks that 80% intervals cover the injected truth 70 to 90% of
the time. The experiment passes if the shares are identified and the score does not get worse. The run
takes minutes on the CPU. It was first considered for the same bundle as the interaction-term and
heating-threshold experiment, which ran without it as experiment 114.

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
