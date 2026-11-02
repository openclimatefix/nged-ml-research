# Demand inputs: research plan

## Priority table

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 437 | Do trailing and forward weather windows, calendar and tariff columns, and national signals lower the error of the pooled neural network? | High (maintainer request, 2026-10-31) | Nothing | Done: no bundle met the bar; thermal lag (bundle T) is the only candidate for a retest on more folds |
| 442 | Does thermal lag (bundle T) lower the pooled neural network's error on cold winter nights, on six winter-weighted folds? | High (brief, 2026-10-31) | Nothing; fits wait for a free GPU slot | Planned |
| 515 | Do trailing weather windows (temperature, shortwave, their 24 h product) lower the error of a lag-free demand network that cannot see recent power, on ten winter folds? | High (maintainer request, 2026-11-02) | Experiments 511 and 513 scored | Done |
| 517 | Is the 0/1 lockdown flag still needed by the lag-free demand network now that pretraining reaches back to 2019? | Medium (brief, 2026-11-02) | Nothing | Planned |

## Done experiments

<!-- planned-table -->
