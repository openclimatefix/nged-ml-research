# Gross-demand simulation: research plan

**No experiment in this theme has been agreed to run.** The options below are queued for the
maintainer to choose; the project is not starting new experiments unprompted. The reasoning is on
[the asset-inventory simulator page](asset-inventory-simulator.md).

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| - | None agreed | - | The maintainer's decision | - |

## Options

| Option | Question | Cost | Depends on |
|---|---|---|---|
| 1. Known-answer test of the inventory fit (first) | Does the fit recover the capacities and running-time parameters of listed, injected assets, and give near-zero capacity to a listed asset that is absent? | CPU only; about a day of one-core fits (a guess) | Nothing; the test is described under [The smallest test](asset-inventory-simulator.md#the-smallest-test) |
| 2. Battery scheduler as a forward model | Do several scheduling profiles (price taker, tariff window, charge only), run forward with a state of charge, reproduce the recorded output of Battery A? | Low to medium: reuses `studies.battery_dispatch` and `studies.battery_templates` | Option 1 for the joint fit; Battery A's record for the check |
| 3. Gas-engine plant equations | Can equations with ramping time and minimum on-duration reproduce the known spells of Gas A and the D8 engine's change? | Medium: no such model exists, and the running data is 44 days for Gas A | Nothing for the equations; option 1 for use in the fit |
| 4. Long-horizon scenario calibration | Issued a month ahead, are the scenarios' tail quantiles calibrated, and narrower than a block bootstrap of the Primary's own history? | Medium to high | Options 1 to 3 |
| 5. Days-1-to-10 what-if for a dormant generator | Does a simulated turn-on reproduce a known start or stop within 30%? | Medium | Option 3 |

## Planned experiments

<!-- planned-table -->
