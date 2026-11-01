# Lagged power and error correction: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 400 | Does a smooth blend of the forecast with power lags and the forecast without them, weighted by lag surprise, beat the forecast with lags around load shifts at no cost elsewhere? | High (maintainer approved 2026-10-30) | Nothing | Planned |
| 404 | Can one neural mixture density network with a lagless module and a calibration module learn a continuous trust weight for each lagged power input, so that lags inside a load-shift event do not harm the forecast? | High (maintainer, weekend programme 2026-10-30) | Slot from the manager; experiment 402's injected-event generator if it exists | Planned |
| 402 | Can a gate trained on injected load shifts set the weight of a mixture of the forecast with power lags and the forecast without them, with no NGED labels? | High (maintainer approved 2026-10-30) | Experiment 400 merged | Finished: partly supported, see experiment 402 |
| 411 | By how much would the forecast with power lags improve on harm rows if the lags inside each real load-shift event were replaced by clean values (the most any lag-trust method can gain)? | High (batch 0 of experiment 404) | Nothing | Planned |
| 414 | Post hoc: does filling the gaps in the stage-1 median, so that more load-shift events are scored, change the counterfactual-lag gain on harm rows? | High (manager, after experiment 411) | Nothing | Planned |
| 428 | Can a convolutional network over whole days of recent power, read by attention, replace the hand-made lagged power inputs and stop the forecast following load-shift events? | High (maintainer, 2026-10-31) | Nothing | Done (S1, the surprise at the three lags, carries forward) |
| 445 | If a detector knew exactly which recent lags lie inside a load-shift event, how much would a lagged and lagless forecast mixture gain (the O-rule ceiling, no fits)? | High (neural Discuss session, 2026-10-31) | Experiment 404 merged | Planned |
| 447 | Does changing any one setting of the S1 demand network (width, depth, embedding, components, dropout, weight decay, learning rate and schedule, batch size, optimiser), or a random search over the three most influential, lower its quantile score by the agreed margin? | High (neural Discuss session, 2026-11-01) | Nothing | Done (not supported: no single setting beats S1; see experiment 447) |
| 459 | Is the sum over a load-shift donor and its recipient Primaries unchanged by the event, so that a forecast of the group sum split by a slowly changing share could work? | High (load-shift Discuss session, 2026-11-01) | Nothing | Finished: refuted, see experiment 459 |

## Planned experiments

<!-- planned-table -->
