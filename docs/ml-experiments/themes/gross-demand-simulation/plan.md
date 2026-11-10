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
| 1. Known-answer test of the inventory fit (first) | Does the fit recover the capacities and running-time parameters of listed, injected assets, and give near-zero capacity to a listed asset that is absent? | CPU only; one-core fits | Nothing; the test is described under [The smallest test](asset-inventory-simulator.md#the-smallest-test) |
| 2. Battery scheduler as a forward model | Do several scheduling profiles (price taker, tariff window, charge only), run forward with a state of charge, reproduce the recorded output of Battery A? | Low to medium: reuses `studies.battery_dispatch` and `studies.battery_templates` | Option 1 for the joint fit; Battery A's record for the check |
| 3. Gas-engine plant equations | Can equations with ramping time and minimum on-duration reproduce the known spells of Gas A and the D8 engine's change? | Medium: no such model exists, and the running data is 44 days for Gas A | Nothing for the equations; option 1 for use in the fit |
| 4. Long-horizon scenario calibration | Issued a month ahead, are the scenarios' tail quantiles calibrated, and narrower than a block bootstrap of the Primary's own history? | Medium to high | Options 1 to 3 |
| 5. Days-1-to-10 what-if for a dormant generator | Does a simulated turn-on reproduce a known start or stop within 30% (a proposed threshold)? | Medium | Option 3 |
| 6. Language-model loop on the register match and the simulator (an idea) | Does a large language model (LLM) that matches Primary signals to the registers and edits the simulator's assets, parameters, and operating profiles recover the known answers better than the plain register-constrained fit? | Unknown; the loop would add LLM calls to the CPU-only fits of option 1 | Option 1 first: the plain fit and its known-answer test must exist before the loop has anything to be compared against |

## Option 6 in detail: a language-model loop

**Option 6 is an idea, not agreed work, and no experiment has been started.** The maintainer's
thought is that an LLM could help with the fuzzy matching between the signals in a Primary's time
series and the very incomplete registers (the Embedded Capacity Register and the other registers on
[the asset-inventory page](asset-inventory-simulator.md#what-the-register-can-supply)). The LLM could
also make adjustments to the simulator's assets, parameters, and operating profiles, so
that the simulator's statistical properties match the observed Primary data as closely as possible.

**The nearest evidence is one reviewer on one task, and it does not settle whether the loop would
work.** In [experiment
692](../../00692-event-table-reviewer-calibration-gate/result.md), a Claude Sonnet reviewer judged a
figure of each candidate load transfer from the event table. It missed the pre-registered precision
gate: 0.45 against 0.80. The reviewer's precision was 0.75, still under 0.80, after the 22 recurring
candidates were removed from the calibration set, a post-hoc reading. Those candidates are events in
the real record that the injected truth leaves out. The experiment did not test an LLM editing a simulator.

**The asset-inventory page's own risks apply to the loop.** The risks are circularity and bands
doing the work (see [Risks](asset-inventory-simulator.md#risks)), and fits left free that invent
generators (see
[that section](asset-inventory-simulator.md#fits-left-free-to-add-generators-report-generators-that-are-absent-or-unconfirmed)).
A loop that edits assets freely could also invent generators to match the statistics. The
controls the loop would need, as options:

- **Register-constrained asset types:** the LLM may only choose among asset types that the register
  lists for the Primary.
- **Held-out months:** the match is scored on months the loop has not seen.
- **A fixed budget of edits:** each run has a set number of changes.
- **Every move logged:** each edit and the reason given for it are stored.
- **The same known-answer test as option 1:** real metered output added to the readings, with an
  absent decoy asset and an unlisted asset.

**The loop passes only if it matches option 1's plain fit on the known-answer test, and a proposed
stop rule ends it otherwise.** The prediction that could fail: the loop recovers each listed,
injected asset's capacity within 20% and the engine's median spell length within 30%, as in the
smallest test, and gives the absent decoy asset under 0.5 MW at 5 or more of the 6 Primaries.
Stop if the decoy gets 1 MW or more at 3 or more of the 6 Primaries, the stop rule of
[the smallest test](asset-inventory-simulator.md#the-smallest-test).

## Planned experiments

<!-- planned-table -->
