# Generator forecasting: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 338 | Does a non-neural column keep the gain of the neural adapter's effective wind speed in the wind forecast? | 1 | nothing | Finished (not supported): A lookup table of wind-speed corrections keeps almost none (5%) of the gain that the neural adapter gave the wind forecast, and boosted trees keep about half (0.051 of 0.099 points of capacity, scored at 1 to 10 day leads). See [experiment 338](../../00338-wind-adapter-column-without-neural-network/result.md). |
| 307 | Which property of the ensemble's wind speed is wrong, judged against the wind that the farms' own power implies? | 2 | nothing | Finished (not supported): The ensemble's low wind-power forecast comes mostly from wind speeds that are too slow, by about 1.1 m/s on average and more at longer leads. See [experiment 307](../../00307-wind-farm-power-as-wind-sensor/result.md). |
| 276 | Why does XGBoost beat the physics core on the three metered wind farms, and is the metering noise the reason? | 1 | nothing | Finished (refuted): XGBoost beats the physics model of the wind farms because it corrects the wind speed using the wind direction, the 10 m wind, and the hour of day. See [experiment 276](../../00276-wind-xgboost-vs-physics-gap-diagnostics/result.md). |
| 444 | Which label-free detectors find idle half-hours at the three metered wind farms at any wind speed, without flagging genuine low output? | 1 | nothing | Finished (not supported); hand labels are the next step |
| 450 | Does table P meet the wind upper-tail rule, and does changing any one setting of the plain wind network, or ERA5 pretraining, lower its score by 0.10 points? | 1 | nothing | Finished (not supported); Farm A upper tail is next |
| 451 | Why does Farm A's upper tail stay above the 1.5% limit after table P? | 1 | nothing | Finished (partly supported); clip at the farm's largest output |
| 453 | Do the metered wind and solar series hold implausibly high values, and which percentile should set effective capacity? | 1 | nothing | Finished (partly supported); p99.9 of the earlier power |
| 461 | Does retraining the wind and solar networks with effective capacity at p99.9 of earlier power keep their score and bring the upper tail inside 1.5%? | 1 | 453 | Finished (partly supported); p99.9 helps wind, marginal for solar |
| 466 | Does weighting recent training rows more raise the quantile score of the wind and solar networks? | 1 | 461 | Finished (not supported); no half-life helps, line closed |
| 473 | What makes Generators X, Y, and Z hard to forecast, and can any of them be forecast from weather and calendar? | 1 | nothing | Finished (refuted); stage 2 for Generator X, and Generator Y only with folds before mid-2024, awaits the manager |
| 617 | Can a hidden Markov model find whole-turbine capacity steps at the metered wind farms from the daily upper envelope of power? | 1 | 453 | Finished (refuted); too few informative days, wind line stops |
| 659 | Does one simple pooled network, with or without a fixed physics-lite power input, forecast the metered wind and solar farms as well as the current defaults? | 1 | nothing | Finished (partly supported): A small shared neural network that sees only the average and spread of the ensemble weather forecasts is no worse than the current, more elaborate defaults for the 3 wind farms and the 6 solar farms over the 5 development months. See [experiment 659](../../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md). |
| [666](../../00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) | What does a one-year backtest of the simple shared wind network S and the simple shared solar network PLP forecast for the 3 metered wind farms and the 6 metered solar farms? | Done | Nothing | Finished (inconclusive): The two simple networks produce complete, sorted 11-quantile forecasts to 14 days for 3 wind farms and 6 solar farms, and wind S matches experiment 659 on the rows both cover. |

## Planned experiments

<!-- planned-table -->
