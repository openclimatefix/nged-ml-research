# Lagged power and error correction: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 400 | Does a smooth blend of the forecast with power lags and the forecast without them, weighted by lag surprise, beat the forecast with lags around load shifts at no cost elsewhere? | High (maintainer approved 2026-10-30) | Nothing | Planned |
| 404 | Can one neural mixture density network with a lagless module and a calibration module learn a continuous trust weight for each lagged power input, so that lags inside a load-shift event do not harm the forecast? | High (maintainer, weekend programme 2026-10-30) | Slot from the manager; experiment 402's injected-event generator if it exists | Planned |

## Planned experiments

<!-- planned-table -->
