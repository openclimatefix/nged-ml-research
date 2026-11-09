---
id: "00656"
slug: pv-probe-through-the-mva-fold-at-eight-primaries
theme: solar-disaggregation
also_relevant_to: []
status: done
kind: other
size: >-
  complex. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  observation model and probe regression is defensible (a linear probe, a one-sided probe, and a
  fold model), so all three are fixed below with every pass bar before any recovery is read.
  Callers: this folder's scripts import experiments 405, 409, 448, 510, 554, 621, and 634 and edit
  none of them. Asset graph: not touched. About 1.0 to 1.6 GPU-hours (ceiling 3), or CPU if one fit
  takes under 2 minutes.
hypothesis: >-
  The donor-day probe of experiment 621, with its linear regression replaced by a fold model that
  knows apparent power is the length of a real-power part and a reactive-power part, reads the
  capacity of unmetered solar at the Primaries metered in megavolt-amperes. It reads it at the five
  Primaries whose midday flow never nears zero (Tier A), and through the fold where the midday flow
  crosses zero (Tier B).
prediction: >-
  Fixed from the Opus design of 2026-11-09 before any recovery is read. Tier A passes: converting
  the six megawatt Primaries to megavolt-amperes at power factor 0.97 costs the probe only a few
  per cent of its recovery. Tier B passes for export-dominated worlds (injection of 6 times the
  midday reading) but not for worlds near the vertex of the fold (injection of 1 to 2 times).
decision_rule: >-
  Every bar of the Tier A and Tier B tables below passes or fails as written, with no rescue tuning
  after a failure. Tier A gates a capacity table for the five Primaries that do not fold. Tier B
  decides whether capacity can be read through the fold, and the three Primaries that fold (D8, D12,
  D22) stay report-only whatever Tier B finds. Recovery is always in megawatts of AC capacity, of the
  injected network minus its uninjected twin, over the injected megawatts. No threshold for acting
  on a probe capacity in production is set here.
benchmark: >-
  The uninjected twin of each condition (same cutoff, seed, reactive-power arm, and rows).
  Experiment 479's anomaly rung, experiment 474's rung R4, and experiment 267's register floor for
  the real-data capacities, as context only.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, every half-hour with the sun at 5.6 degrees elevation or
  more.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day probe; recovery as a share of the
  injected MW and of the injected energy; day-block bootstrap 80% interval widened by the
  leave-one-farm-out spread.
production_status: not-for-production
adopted_in: none
verdict: inconclusive
best_arm: >-
  The fold model with a night prior (FA), post hoc with wider starting capacities: 0.74 to 0.89 of
  an injected fleet's megawatts at 0.5 to 6 times the Primary's median sunny-midday reading, against
  -0.80 for the linear probe at 6 times (see the post hoc table).
headline: >-
  Bar A1 failed, so stop rule 3 ended the run after the 2026-06 cutoff, with 4 of 11 pre-registered
  bars passed. Bar A1 compares two different injection levels, and the 6-times failure comes from
  the fit's starting capacities. Median recovery of the injected megawatts over the six MW
  Primaries: linear probe 0.68, 0.53, and -0.11 at 0.5, 1, and 2 times the midday reading; fold
  model 0.76, 0.70, and 0.79.
lessons:
  - >-
    The linear probe cannot read capacity once solar output makes the apparent-power reading fold:
    it reads 0.53 at 1 times the midday reading and -0.11 at 2 times, against 0.76 for the same
    farm in signed megawatts.
  - >-
    The fold model reads an injected fleet at 0.70 to 0.82 of its megawatts up to 2 times the
    midday reading, as much as the signed world gives (ratio 1.09 at 1 times), and reads the real
    D12 at +5.3 MW where the linear probe reads -3.0 MW.
  - >-
    The pre-registered start grid for the fold fit (up to 3 times the midday reading) cannot reach
    a 6-times capacity; starting at 6 and 8 times, post hoc, gives 0.88 and a sun-rising-load
    placebo of -0.11.
  - >-
    The five MVA Primaries that do not fold behave like the six MW Primaries: the linear probe
    reads 0.78 of an injected fleet there, and the leave-one-Primary-out factor is 0.68 with a
    spread of 0.01.
take_home_message: >-
  We tested, on six substations metered in signed megawatts, whether a probe of a trained neural
  network can find a solar farm that we added to the data. To mimic the eight substations metered in
  megavolt-amperes, we first converted the flow to apparent power, which never goes negative. A
  simple linear probe finds two thirds of a small farm and fails once the farm's output can exceed
  the substation's demand. A probe that models the fold finds 0.7 to 0.8 of farms up to twice the
  midday demand, and 0.88 of a six-times farm after we widened its starting capacities. A capacity
  table looks possible for the five MVA Primaries that never fold. The three that do fold need a
  second try with a second cutoff.
advice:
  - >-
    Read capacity at an MVA Primary that never folds from the linear probe divided by 0.68, and
    state the power-factor-0.97 assumption beside the number (rows A2 and A4 of the bar table).
  - >-
    Use the fold model, not the linear or one-sided linear probe, wherever the midday reading nears
    zero (the 2-times point of the recovery figure).
  - >-
    Start the fold model's capacity search at up to 8 times the midday reading, and judge its
    recovery at 6 times only with those starts (post hoc table).
  - >-
    Train a signed twin at every injection level before using a fold-loss ratio as a stop rule (row
    A1 of the bar table).
  - >-
    Do not size an injection by the midday reading at D5 or D10, where that reading is small (row A1
    of the bar table).
---

# Can a solar-capacity probe still work when the substation meter reports apparent power?

**The run is inconclusive: the pre-registered gate failed after one cutoff, but the failure came
from a flaw in bar A1 and from the starting capacities of the fold fit.** The linear probe from
experiment 621 loses most of its signal once an added solar farm can push a substation's flow
through zero. A probe that models the fold recovered 0.76 to 0.82 of an added fleet's megawatts at
0.5 to 2 times the midday reading, and 0.88 at 6 times only after the starting capacities were
widened following the failure (post hoc).

We asked whether the unmetered solar capacity behind a substation can be read when the meter reports
apparent power (megavolt-amperes, which never goes negative) rather than signed megawatts. Six
Primary substations report signed megawatts. We converted their flow to synthetic apparent power,
added a solar fleet of known size to the data, and trained a neural network on the result. The probe
then holds the calendar inputs of that neural network fixed, feeds it the weather of other days, and
regresses its output on a solar-output curve. The slope is the capacity it reads, in megawatts of
alternating current (AC). We compare the capacity read from a neural network trained with the fleet
against its twin trained without the fleet. Only the 2026-06 cutoff ran: 24 conditions with 3 seeds
each is 72 neural networks, at 23 seconds per fit on the graphics processing unit (GPU).

**Terms used below.**

- **MW Primaries and MVA Primaries.** The six MW Primaries (D5, D9, D10, D11, D14, D16) are metered
  in signed megawatts. The eight MVA Primaries are metered in apparent power: D2, D3, D4, D19, and
  D21 never come near zero flow on a sunny midday, and D8, D12, and D22 do (they "fold").
- **The fold.** Apparent power is the length of a real-power part and a reactive-power part. When
  solar output exceeds demand, the reading falls to the reactive floor and then rises again, so a
  linear fit reads the wrong sign.
- **Midday reading.** A Primary's median flow over sunny half-hours from 10:00 to 15:00 UTC in April
  to August, in MW.
- **Injection level.** The AC capacity of the added fleet as a multiple of the midday reading (0.5×,
  1×, 2×, 6×). A fleet is half a real metered farm's output shape and half a physics mixture.
- **Recovery.** The capacity read from the neural network trained with the fleet, minus the capacity
  read from its twin, divided by the fleet's AC capacity. The values below are medians over seeds
  and Primaries.
- **Probes.** L0 is the linear probe of experiment 621. L1 is L0 on rows well above the reactive
  floor. F4 is the fold model with four solar-output curves, F1 with one, and FA is F4 with a prior
  on each probe cell's demand from the night reading.
- **Worlds.** In the power-factor (PF) world the reactive power is 0.25 times the median real power
  of that half-hour of the week (power factor 0.97). In the floor world the reactive power is a
  constant. In the native world the five non-folding MVA Primaries take the same transformation.
- **Tier A and Tier B.** Tier A gates a capacity table for the five MVA Primaries that do not fold.
  Tier B asks whether capacity can be read through the fold, where D8, D12, and D22 stay
  report-only.

![Recovery by injection level and probe, six MW Primaries](recovery_by_level.svg)

**Figure: look at the right half.** Orange and green (the linear probes) fall below zero at 2× and
6×. Blue (the fold model, pre-registered starting capacities) holds near 0.7 to 0.8 up to 2×, then
also falls at 6×, which the post hoc section explains. Dots are single Primaries, so D5 and D10 sit
lowest at 0.5× and 1×.

**The pre-registered gate failed: 4 of 11 bars passed (2026-06 cutoff only).** The bars are as fixed
in the plan commit.

| Bar | Result | Pass |
|---|---|---|
| A1 Fold-free loss: ratio of PF 0.5× recovery to the signed-world recovery at 1×, for L0 and F4, in 0.90 to 1.10 at 5 of 6 | L0 ratios 0.56, 0.90, 0.44, 0.87, 0.88, 0.91 for D5, D9, D10, D11, D14, D16; F4 0.47 to 1.28 | fail |
| A2 Native worlds: recovery within ±0.12 of the six's median (0.68) at 4 of 5 | 4 of 5 (D3 0.83 is outside; D2, D4, D19, D21 are 0.73 to 0.78) | pass |
| A3 Placebos: 95th percentile of capacity read under 0.25 of the real 0.5× capacity read | climatological 6 of 6 MW Primaries and 5 of 5 MVA Primaries; boxcar 5 of 6 and 5 of 5 | pass |
| A4 Leave-one-Primary-out factor in 0.55 to 0.95, spread under 0.15 | 0.677 to 0.683 at all six | pass |
| A5 Interval coverage at least 60% | 0.18 | fail |
| B1 Four worlds (PF 1×, PF 2×, floor 1×, floor 2×) each have FA calibrated recovery in 0.75 to 1.25 at 4 of 6 | 4, 5, 1, and 5 of 6 respectively | fail |
| B2 PF 6×: FA calibrated recovery in 0.75 to 1.25 at 4 of 6 | 2 of 6 (values 0.55 to 6.78) | fail |
| B3 At PF 2×, FA's median absolute error from 1 is at least 0.10 below L1's | 0.152 against 0.234, gap 0.08 | fail |
| B4 Sun-rising-load placebo: FA under 0.25 of the 6× recovery at 5 of 6, and crossing flags on under 10% of cells | 6 of 6; flags on 0.0% of cells | pass |
| B5 Interval coverage at least 60% at PF 1× and 2× | 0.17 and 0.14 | fail |
| B6 Crossing map: balanced accuracy at least 0.80 at 1× and 2×, false crossings at most 10% at 0.5× | 0.73 and 0.94; 7.6% | fail |

- **A1's denominator is a design flaw.** No signed world was trained at 0.5×, so the code divides by
  the signed world at 1×. The linear probe's recovery falls from 0.68 to 0.53 between those levels
  in the PF world and stays at 0.76 to 0.79 in the signed world. At the same level (1×), the ratio
  of PF to signed recovery is 0.69 for L0, 0.89 for L1, and 1.09 for FA.
- **A3 is a one-sided bar.** The linear probe reads the climatological placebo at -0.21, which is
  far from zero and still passes, because a negative reading is under 0.25.
- **D5 and D10 cause several bars to fail.** Their midday readings are 0.88 and 1.2 MW, so a 0.5×
  injection is 0.44 and 0.60 MW, close to the seed spread.
- **Intervals are narrow.** A5 and B5 fail because the bootstrap (40 draws over days) carries no
  variation between neural networks, and the leave-one-Primary-out factor spread (0.01) does not
  widen the interval to include it.

## Post hoc: widening the fold fit's starting capacities fixes the failure at 6×

The audit found that the fold fit sits on its zero-capacity bound at almost every MW Primary in the
6× worlds. The fit started at 0.25, 1, and 3 times the midday reading, and a true capacity of 6
times lies beyond every start. Starting also at 6 and 8 times, post hoc and with no retraining,
gives:

| Condition (median over seeds and the six MW Primaries) | FA | F4 | F1 |
|---|---|---|---|
| Fleet 0.5× | 0.74 | 0.67 | 0.70 |
| Fleet 1× | 0.70 | 0.64 | 0.72 |
| Fleet 2× | 0.80 | 0.78 | 0.71 |
| Fleet 6× | 0.88 | 0.83 | 0.72 |
| Physics mixture 6× | 0.94 | 0.87 | 0.78 |
| Sun-rising load (placebo; no solar added) | -0.11 | -0.13 | -0.09 |

FA at 6× reads 0.99 at D11 and 0.63 at D5, and the five native MVA Primaries read 0.91. No bar is
re-scored with these rows.

![Capacity read from worlds with no added solar farm, and the 0.5× fleet for reference](placebos.svg)

**Figure: the cloud-following load is the placebo that fails.** The fold model reads it at 0.62 of
its size and the linear probe at 0.44, so neither probe separates solar output from demand that
follows the clouds. Dots at the same level as the reference row would count as a read-out of real
solar.

## Real-data capacity (uninjected 2026-06 neural networks, raw)

For the five MVA Primaries that do not fold, the linear probe's raw capacity divided by the 0.68
factor of bar A4 gives a calibrated column, which assumes power factor 0.97. The fold-model column
is raw, with the pre-registered starting capacities.

| Primary | Linear raw (MW) | Linear divided by 0.68 (MW) | Fold model FA raw (MW) |
|---|---|---|---|
| D2 | 5.2 | 7.6 | 8.0 |
| D3 | 2.1 | 3.0 | 3.0 (up to 4.8) |
| D4 | 1.0 | 1.4 | 3.2 (1.1 to 3.6) |
| D19 | 1.1 | 1.7 | 2.1 |
| D21 | 1.2 | 1.8 | 1.4 |
| D8 (report only) | 1.8 | 2.6 | 2.1 |
| D12 (report only) | -3.0 | not defined | 5.3 |
| D22 (report only) | 1.3 | 2.0 | 1.7 |

![Capacity read from the uninjected neural networks at fourteen Primaries](capacity_at_fourteen_primaries.svg)

**Figure: compare orange with blue at D12.** The linear probe reads -3.0 MW there, as [experiment
479](../00479-pv-capacity-anomaly-regression-rung/result.md) did on the raw reading, and the fold
model reads +5.3 MW. D12 is the only Primary whose probed reading rises with sun; the other 13 fall.
Sites labelled report only fold. The estimates are raw and uncalibrated.

![Where the fold model flags a crossing, at D12 and D22](crossing_map.svg)

**Figure: D12 is flagged at midday in most months and D22 is not flagged.** The map is unvalidated
because bar B6 failed. Grey cells have no flagged crossing, and blank cells have no probe cell.

## Deviations from the plan

- **Fold fit.** Three starting capacities (0.25, 1, and 3 times the midday reading) in place of a
  profile over 121 capacities; 40 bootstrap draws in place of 200; 40 donor rows per probe cell. The
  post hoc starts at 6 and 8 times are labelled above.
- **A1.** Compared with the signed world at 1×, as above. Stop rule 3 (stop if A1 fails) fired as
  pre-registered.
- **Fleet.** The physics half of the fleet is driven by the Primary's own sky, with no offset to a
  neighbouring pixel. The boxcar placebo is capped at the Primary's midday reading per event,
  because an event sized to the daily energy of the 2× fleet exceeded the load.
- **L1 probe.** The floor estimate in the one-sided probe's threshold is capped at half the midday
  reading.
- **Native worlds** apply only in the PF world.
- **Not run.** The 2026-03 cutoff, the signed-world energy conversion, and the example S(u) curves
  beyond their class (falling at 13 Primaries, rising at D12).

## Caveats

- The reactive power of the real meters is unknown, and power factor 0.97 is an assumption.
- NGED's event labels are incomplete, so the clean rows may hold unlabelled events.
- D8's flow regime changed in 2025, and the neural network has no year input, so it averages over
  the change.
- Nothing here is replicated: one cutoff and one set of farms.

## Checks

- **Forecast checks:** the experiment scores no forecast, so the median-minus-mean and
  quantile-score checks do not apply.
- **Audit triggers that fired:** a control failed (A1), the result contradicted its prediction for
  Tier B at 6× (explained by the audit), and a follow-up would depend on the unreplicated 0.79 at
  2×. A leave-one-site-out flip and a nested-arm win did not fire. The audit report is
  `e656_audit.md` in the private briefs folder.
- **Audit triage:** accepted finding 1 (the post hoc table), accepted finding 3 (the A1 caveat), and
  accepted the trigger list (finding 5). Finding 2 reproduced the headline numbers, and finding 4
  found no bug in the world construction.
- **Code review (Sonnet):** accepted the boxcar cap, the sun-rising-load scaling, the arm-specific
  night reading, the basis-curve normalisation, and native worlds in the PF world only. Rejected the
  profile-likelihood deviation as a defect, and listed it above.
- **Write-up review:** accepted the A2 and rounding corrections, the missing captions, and the
  figure fixes.

## What the next experiment should answer

Does the fold model with a night prior keep its 0.7 to 0.9 recovery at a second cutoff and a second
set of farms, with a signed twin at every level, starting capacities out to 8 times, and an
injection that is not sized by D5's small midday reading? Does it still pass the sun-rising-load
placebo, and can anything separate solar output from the cloud-following load that it misreads?
