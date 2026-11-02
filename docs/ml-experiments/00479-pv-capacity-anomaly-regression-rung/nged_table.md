# Unmetered solar capacity at each Primary, anomaly regression (NGED table)

Season-year July 2025 to June 2026, in megawatts. Estimator: anomaly regression with the two cloud
terms. No row is validated, because the estimator recovers only 0.73 to 0.83 of an injected farm
(see the experiment page). The intervals are the 10th to 90th percentile of a month-block bootstrap
and do not include that bias, so the true capacity is likely higher.

| Primary | Status | AC capacity, MW (10th / 50th / 90th) | DC capacity at 1.25, MW (10th / 50th / 90th) | Experiment 474 AC median, MW |
|---|---|---|---|---|
| D2 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 6.6 / 7.0 / 7.3 | 8.3 / 8.7 / 9.2 | 6.9 |
| D3 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 2.7 / 2.9 / 3.0 | 3.4 / 3.6 / 3.7 | 4.2 |
| D4 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 0.7 / 1.1 / 1.6 | 0.9 / 1.4 / 2.0 | 0.8 |
| D5 | export limit: the clip-free solar term does not represent it; unvalidated | 3.1 / 3.2 / 3.2 | 3.8 / 4.0 / 4.1 | 3.1 |
| D8 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 2.7 / 3.2 / 3.6 | 3.4 / 4.0 / 4.6 | 8.6 |
| D9 | signed MW reading; unvalidated | 2.0 / 2.2 / 2.3 | 2.5 / 2.7 / 2.9 | 2.2 |
| D10 | signed MW reading; unvalidated | 2.5 / 2.5 / 2.6 | 3.1 / 3.2 / 3.3 | 2.9 |
| D11 | signed MW reading; unvalidated | 3.1 / 3.2 / 3.4 | 3.9 / 4.1 / 4.2 | 3.2 |
| D12 | reverse flow: the estimate is negative and is not a capacity; unvalidated | -3.4 / -3.0 / -2.6 | -4.2 / -3.8 / -3.3 | - |
| D14 | signed MW reading; unvalidated | 3.6 / 3.7 / 3.8 | 4.5 / 4.6 / 4.8 | 3.7 |
| D15 | generator case: not run | - | - | - |
| D16 | signed MW reading; unvalidated | 3.5 / 3.6 / 3.7 | 4.4 / 4.5 / 4.6 | 3.1 |
| D18 | history only from 2025-03-01; report-only; unvalidated | 2.0 / 2.0 / 2.1 | 2.5 / 2.6 / 2.6 | 1.8 |
| D19 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 1.6 / 1.7 / 1.9 | 2.0 / 2.2 / 2.4 | 1.8 |
| D21 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 1.5 / 1.5 / 1.6 | 1.9 / 1.9 / 2.0 | 2.0 |
| D22 | MVA-only: apparent power, reactive power assumed zero; unvalidated | 1.6 / 1.7 / 1.8 | 2.0 / 2.1 / 2.3 | 1.5 |
