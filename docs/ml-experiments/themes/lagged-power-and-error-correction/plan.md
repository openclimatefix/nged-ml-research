# Lagged power and error correction: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 400 | Does a smooth blend of the forecast with power lags and the forecast without them, weighted by lag surprise, beat the forecast with lags around load shifts at no cost elsewhere? | High (maintainer approved 2026-10-30) | Nothing | Planned |
| 402 | Can a gate trained on injected load shifts set the weight of a mixture of the forecast with power lags and the forecast without them, with no NGED labels? | High (maintainer approved 2026-10-30) | Experiment 400 merged | Finished: partly supported, see experiment 402 |
| 411 | By how much would the forecast with power lags improve on harm rows if the lags inside each real load-shift event were replaced by clean values (the most any lag-trust method can gain)? | High (batch 0 of experiment 404) | Nothing | Planned |
| 414 | Post hoc: does filling the gaps in the stage-1 median, so that more load-shift events are scored, change the counterfactual-lag gain on harm rows? | High (manager, after experiment 411) | Nothing | Planned |
| 428 | Can a convolutional network over whole days of recent power, read by attention, replace the hand-made lagged power inputs and stop the forecast following load-shift events? | High (maintainer, 2026-10-31) | Nothing | Done (S1, the surprise at the three lags, carries forward) |
| 440 | Does reading the surprise in a window around each lag, and against what the neighbouring Primaries did, help the lag network tell a load-shift event from a genuine level shift? | High (maintainer, 2026-10-31) | Experiment 439 (S1 confirmation) for the base | Planned |

## Planned experiments

<!-- planned-table -->
