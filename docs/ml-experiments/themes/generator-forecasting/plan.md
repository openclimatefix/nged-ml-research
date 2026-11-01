# Generator forecasting: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 338 | Does a non-neural column keep the gain of the neural adapter's effective wind speed in the wind forecast? | 1 | nothing | Planned |
| 307 | Which property of the ensemble's wind speed is wrong, judged against the wind that the farms' own power implies? | 2 | nothing | Planned |
| 276 | Why does XGBoost beat the physics core on the three metered wind farms, and is the metering noise the reason? | 1 | nothing | Planned; running diagnostics D1 to D10 |
| 444 | Which label-free detectors find idle half-hours at the three metered wind farms at any wind speed, without flagging genuine low output? | 1 | nothing | Finished (not supported); hand labels are the next step |
| 450 | Does table P meet the wind upper-tail rule, and does changing any one setting of the plain wind network, or ERA5 pretraining, lower its score by 0.10 points? | 1 | nothing | Finished (not supported); Farm A upper tail is next |
| 451 | Why does Farm A's upper tail stay above the 1.5% limit after table P? | 1 | nothing | Finished (partly supported); clip at the farm's largest output |
| 453 | Do the metered wind and solar series hold implausibly high values, and which percentile should set effective capacity? | 1 | nothing | Finished (partly supported); p99.9 of the earlier power |
| 461 | Does retraining the wind and solar networks with effective capacity at p99.9 of earlier power keep their score and bring the upper tail inside 1.5%? | 1 | 453 | Finished (partly supported); p99.9 helps wind, marginal for solar |
| 466 | Does weighting recent training rows more raise the quantile score of the wind and solar networks? | 1 | 461 | Planned; low prior (experiments 443 and 464 found no gain) |

## Planned experiments

<!-- planned-table -->
