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
| 458 | Can a weight per lead band, a prequential weight, or a shrunk level offset on the lagless forecast beat the fixed 50/50 mixture of the lagged and lagless forecasts (zero fits)? | High (neural Discuss session, 2026-11-01) | Nothing | Finished: not supported, see experiment 458 |
| 462 | Can a lag network that sees the neighbours learn to follow a level change that no neighbour mirrors, when its training worlds contain genuine level changes with and without a mirror (row 4 of the experiment 404 review, G3)? | High (neural Discuss session, 2026-11-01) | Row 3 reported; NRA final list | Done (inconclusive by design: the optimal mirror gap is 0.054 to 0.075, below the seed noise; see experiment 462) |
| 507 | Is there enough to gain from repairing shifted power readings, rather than down-weighting them, to justify a learned 14-day history encoder? | High (neural Discuss session, 2026-11-02) | Nothing | Interim: oracle repair helps; one total-variation repair does not; learned encoder not run (line paused) |
| 550 | Do trailing temperature windows still help the lag-light demand network once the day-ago and week-ago power lags are in? | High (manager, 2026-11-04) | Nothing | Planned |
| 567 | Do the neighbouring Primaries' lags let the modular lag-encoder demand network tell a load transfer from a genuine level change, when it trains on synthetic load shifts? | High (manager, 2026-11-05) | Experiments 559, 560, and 566 merged | Planned |
| 589 | Why does the P5 forecast cover Primary D5 poorly before the December 2025 load-shift, and is that one Primary or a property of the P5 recipe? | High (manager, 2026-11-06) | Nothing | Planned |
| 595 | Does the fine-tuned demand network of C1 (P5 plus lag-only synthetic shifts) or A1 (shared lag encoder) absorb less of a load-shift in its training power than plain P5's 0.219 of the depth, so that the proposed bar L4 (at most 0.10) can be met? | High (manager, 2026-11-06) | Nothing | Finished: partly supported, see experiment 595 |
| 597 | Do C1 and A1 with the G4 gated band, assembled as single forecasts, hold the normal running arrangement inside the q05-q95 band on 0.90 of rows from day 4 of a load-shift to its end? | High (manager, 2026-11-06) | Nothing | Finished: supported, see experiment 597 |
| 600 | Do C1, A1, and their G4 assembled forecasts still hold the normal running arrangement inside the q05-q95 band at 120-240 h ahead, where NGED's horizon reaches 10 days? | High (manager, 2026-11-06) | Nothing | Planned |
| 602 | What does each encoder of the modular demand network (datetime, weather, power lag) contribute to its forecast, on the December 2025 load-shift, on injected load-shifts, and on an event-free fortnight? | High (maintainer, 2026-11-06) | Nothing | Finished: partly supported, see experiment 602 |
| 608 | Which parts of the combined lag-and-gate, lag-without-gate, and lag-free demand neural networks are needed, when each part is removed alone? | High (maintainer, 2026-11-06) | Nothing | Planned |
| 647 | Can a small departure head on the frozen K1 neural network forecast the arrangement now in force better than K1 shifted by its own recent error? | High (maintainer, 2026-11-07) | Nothing | Finished: partly supported, see experiment 647 |
| 650 | Does a departure head trained on shifts with random ends, varied shapes, and varied depths beat K1 shifted by its own decayed recent error, once band widening is removed? | High (manager, 2026-11-08) | Nothing | Finished: partly supported, see experiment 650 |
| 652 | Does a departure head with an explicit survival output beat the best forecast that does not know when a shift ends, once each forecast is scored under its own rules? | High (maintainer, 2026-11-08) | Nothing | Finished: not supported, see experiment 652 |
| 654 | Does one network with an NRA output and a prevailing output, trained end to end with a learned end detector, beat a frozen K1 plus a separate survival head? | High (maintainer, 2026-11-09) | Nothing | Planned |

## Planned experiments

<!-- planned-table -->
