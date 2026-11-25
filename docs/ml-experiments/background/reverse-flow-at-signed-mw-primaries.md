# Reverse flow at the signed-MW Primaries

**Only D15 has reverse flow as its usual state, and D10 has small midday reverse flow that is growing
year on year.** Reverse flow means a half-hour in which the signed power reading, in megawatts (MW),
is below zero, so that the Primary exports power to the grid above it. This page records, for the
eight Primaries that NGED meters in signed MW, how often and how deeply reverse flow occurs, and what
that means for training a detector that recovers the sign of a reading that has been folded into a
magnitude. The Primaries metered only in MVA are covered by [experiment
79](../00079-pvdis-7-mw-at-mva-only-primaries/result.md) and [experiment
110](../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md), and the units
are explained on the [NGED power telemetry page](nged-power-telemetry.md).

## What was counted

- **The eight signed-MW Primaries are D5, D9, D10, D11, D14, D15, D16, and D18.** The count uses the
  cleaned power store, every row, with hours in local time on the start of each half-hour (exploration
  on 2026-11-25; 8 Primaries; provisional).
- **Depth is in points of capacity.** A depth of 10 points is 10% of the Primary's effective capacity.
- **A folded reading is the magnitude of the apparent power, which is never negative.** A
  reverse-flow half-hour is ambiguous after folding when its absolute MW is below 1 MVA, or below 0.3
  MVA, because a small reading looks the same as a small forward flow.
- **The drivers named below are inferences from the time of day and the season.** No generator is
  identified.

## How often, and how deep

| Primary | Half-hours | Reverse flow | Share of half-hours | Depth, median / 99th percentile (points) | Usual hour | Under 1 MVA | Under 0.3 MVA |
|---|---|---|---|---|---|---|---|
| D5 | 121,492 | 19 | 0.02% | 2.8 / 8.5 | 13:00 | 100% | 79% |
| D9 | 52,749 | 0 | 0% | none | none | none | none |
| D10 | 121,611 | 317 | 0.26% | 3.2 / 14.3 | 13:00 | 99% | 68% |
| D11 | 121,673 | 0 | 0% | none | none | none | none |
| D14 | 120,052 | 0 | 0% | none | none | none | none |
| D15 | 26,700 | 20,386 | 76.35% | 30.4 / 90.2 | all day | 55% | 12% |
| D16 | 99,471 | 0 | 0% | none | none | none | none |
| D18 | 28,360 | 48 | 0.17% | 10.6 / 33.0 | 11:00 | 85% | 35% |

The last two columns are shares of the Primary's reverse-flow half-hours. Expressed as shares of all
half-hours, the ambiguous cases are 0.02% and 0.01% at D5, 0.26% and 0.18% at D10, 42% and 9% at
D15, and 0.14% and 0.06% at D18.

## Primaries with no reverse flow

- **D9, D11, D14, and D16 never read below zero.** Their lowest readings are above zero, so a folded
  reading and a signed reading are the same number at these Primaries (exploration on 2026-11-25; 4
  Primaries; provisional).
- **D11 and D16 show a midday dip from solar PV that does not reach zero.** At D16, the median midday
  reading in 2026 is below the median night reading. The dip is inferred from the daily profile alone.

## D15: reverse flow as the usual state

- **76% of D15's half-hours are reverse flow, in a profile that is flat through the day.** The share
  is 69% to 83% in every hour except 16:00 to 21:00, where it falls to 58% to 74%. A flat all-day
  profile is consistent with base-load generation and not with solar PV.
- **Reverse flow at D15 is deep.** The median is 30 points and the 99th percentile is 90 points.
  The longest runs last 784, 647, and 485 half-hours.
- **The share is 87% in 2025 and 69% in 2026.** The series starts in 2025, so both years are short, and
  the 2025 figure covers only part of a year.
- **Folding makes D15 badly ambiguous.** More than half of its reverse-flow half-hours, 55%, read
  below 1 MVA, and 12% read below 0.3 MVA.

## D10: small and growing midday reverse flow

- **D10's reverse flow is 0.26% of half-hours and sits between 10:00 and 15:00, peaking at 13:00,
  from March to August.** The pattern is consistent with solar PV.
- **The yearly count rises from 0 half-hours in 2022 to 27, 55, 108, and 127 in 2023 to 2026.** The
  share of half-hours rises from 0.15% to 0.81% over the same years, and 2026 is a partial year.
- **Reverse flow at D10 is shallow, with one outlier.** The median depth is 3.2 points and the 99th
  percentile is 14.3 points. One isolated reading reaches 94 points, probably a switching event.
- **Folding hides nearly all of it.** Of D10's 317 reverse-flow half-hours, 315 read below 1 MVA.

## D5 and D18: a handful of half-hours

- **D5 has 19 reverse-flow half-hours in about seven years, all between 10:00 and 15:00.** They fall
  in April, August, and September of 2021 and 2023, with a depth of at most 8.6 points, so they
  are consistent with solar PV on days of low demand.
- **D18 has 48 reverse-flow half-hours, all in a short episode in 2025, and none in 2026.** The
  episode spans every hour of the day, so the readings do not match a solar profile. The cause is not
  identified.

## What this means for training a sign detector

- **Few real reverse-flow examples exist outside D15.** D10, D5, and D18 together hold 384 reverse-flow
  half-hours, of which 98% (375) sit under 1 MVA, where the folded reading carries almost no signal
  about the sign.
- **D15 is the one Primary with enough reverse-flow half-hours to train or test on, and it supplies
  one regime.** That regime is deep, all-day reverse flow, consistent with base-load generation. It has
  a short history, so any split by time leaves few weeks on each side.
- **The regimes are separate: D15 shows all-day reverse flow, and D10 and D5 show only midday
  reverse flow.** A detector validated on D15 or D10 says little about a Primary with both patterns.
- **Real examples are too few to score a detector, so evidence must come from synthetic reverse flow
  or from other sources.** Experiment 79 found no reverse flow at 7 of the 8 MVA-only Primaries, and
  experiment 110 fitted a power factor only where reverse flow exceeded 0.5% of half-hours. The signed-MW
  Primaries add D15 and, weakly, D10 to that set.
