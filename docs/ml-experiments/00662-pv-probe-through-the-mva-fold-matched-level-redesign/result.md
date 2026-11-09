---
id: "00662"
slug: pv-probe-through-the-mva-fold-matched-level-redesign
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  complex. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: the test
  design is fixed by the maintainer-approved redesign of experiment 656's test, so one design is
  defensible and every change is listed below. Callers: this folder's scripts copy experiment 656's
  and import experiments 397, 405, 409, 428, 448, 510, 554, and 621 and edit none of them. Asset
  graph: not touched. About 1.4 GPU-hours (ceiling 2) and 8 wall-clock hours.
hypothesis: >-
  The donor-day probe of experiment 621, with its linear regression replaced by a fold model,
  reads the capacity of unmetered solar at the eight Primaries metered in megavolt-amperes.
  Experiment 656 could not settle this, because its bar A1 compared two injection levels, its fold
  fit started too low to reach a 6-times farm, and its intervals ignored the variation between
  trained networks. This experiment repeats the test with those three faults removed.
prediction: >-
  Fixed before any recovery is read. Matched-level bar A1 passes at 0.5, 1, and 2 times, because
  experiment 656 found the linear probe's recovery close to level-free in the signed world (0.76
  at 1 times, 0.80 at 6 times) and the PF world differs from it by a 3% power-factor loss at
  0.5 times. Tier B passes at 1, 2, and 6 times once the fit starts at up to 8 times the midday
  reading, because experiment 656's post hoc wide-start fit read 0.70 to 0.88 there. Fewer than
  10% of fold fits end on a bound. Interval coverage reaches 60% once between-seed variation is
  added to the day-block bootstrap.
decision_rule: >-
  Every bar of the tables below passes or fails as written, with no rescue tuning after a failure.
  Tier A gates a capacity table for the five Primaries that do not fold. Tier B decides whether
  capacity can be read through the fold, and the three Primaries that fold (D8, D12, D22) stay
  report-only whatever Tier B finds. Recovery is always in megawatts of AC capacity of the injected
  network minus its uninjected twin, over the injected megawatts. No threshold for acting on a
  probe capacity in production is set here.
benchmark: >-
  The uninjected twin of each condition (same cutoff, seed, reactive-power arm, and rows). The
  signed-megawatt world of the same farm at the same level, for the matched-level bar A1.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, every half-hour with the sun at 5.6 degrees elevation or
  more.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day probe; recovery as a share of the
  injected MW; day-block bootstrap 80% interval widened by the between-seed spread and the
  leave-one-farm-out spread.
---

# Can the solar-capacity probe read through the apparent-power fold, with a fair test?

**Status: planned. The specification is `briefs/pv_mva_bounce_redesign_brief.md` of 2026-11-09
(private folder), approved by the maintainer, and every bar below is fixed before any number is
read.**

Experiment 656 asked whether the unmetered solar capacity behind a substation can be read when the
meter reports apparent power (MVA, never negative) rather than signed real power (MW). It stopped
after one cutoff because bar A1 failed, and its audit found three faults in the test. This
experiment repeats the test with those faults removed.

## What changes from experiment 656

- **A1 compares like with like.** Signed-megawatt worlds are trained at 0.5, 1, and 2 times, so the
  PF recovery of a farm is divided by the signed recovery of the same farm at the same level.
- **The fold fit starts at 0.25, 1, 3, 6, and 8 times the midday reading.** The bounds are
  unchanged. The share of fits that end on a bound is reported.
- **Injection floor.** Level L means max(L times the Primary's median sunny-midday reading, 2.0 MW)
  of AC capacity. At D5 and D10, whose midday readings are 0.89 and 1.2 MW, the 0.5-times and
  1-times worlds therefore inject the same 2.0 MW. The write-up states the injected megawatts of
  every Primary.
- **Intervals include network variation.** 5 seeds at 2026-06 and 3 at 2026-03. The interval of a
  recovery is the day-block bootstrap half-width and 1.2816 times the between-seed standard
  deviation, summed in quadrature, then widened by the leave-one-farm-out factors.
- **A3 is two-sided.** A placebo passes when the 95th percentile of its absolute read is under 0.25
  of the real recovery.
- **Both cutoffs, with a real-data table.** The uninjected twins of all eight MVA Primaries are
  read at 2026-03 and 2026-06, raw and calibrated by the same-level leave-one-farm-out factor.

Unchanged from experiment 656: the network (experiment 554's modular mixture-density network),
the rows, the fleet shapes, the three reactive-power arms, the probe, the three regressions (L0,
L1, and the fold models F4, F1, and FA), and bars A2 to A5 and B1 to B6.

## Design in brief

- **Conditions.** 26 conditions: the signed arm (twin and fleets at 0.5, 1, 2, and 6 times), the PF
  arm (twin; real-farm fleets at 0.5, 1, 2, and 6 times, the last three with two farm rotations;
  physics fleets at 1 and 6 times; four placebos), the floor arm (twin; fleets at 1, 2, and 6
  times; climatological placebo), and the zero-reactive-power arm (twin; fleet at 2 times).
- **Fits.** 26 conditions times 5 seeds at 2026-06 plus the same times 3 seeds at 2026-03 is about
  208 fits at about 23 s on the GPU, 1.4 GPU-hours.
- **Day-block bootstrap.** 40 draws, on the PF twin and the PF real-farm fleets.

## Pass bars (fixed before any read-out)

Tier A (gates the capacity table for D2, D3, D4, D19, and D21):

| Bar | Pass when |
|---|---|
| A1 Matched-level fold-free loss | at each of 0.5, 1, and 2 times, for each of L0, L1, and FA: the PF recovery divided by the signed recovery of the same farm at the same level is in 0.90 to 1.10 at 5 of the 6 MW Primaries |
| A2 Native worlds | PF arm, L0.5, native: raw L0 recovery within ±0.12 of the six's PF L0.5 recovery at 4 of the 5 |
| A3 Placebos | climatological and boxcar placebos: 95th percentile of the absolute L0 read under 0.25 of the real L0.5 recovery at 5 of 6, and at 4 of the 5 in native worlds |
| A4 Calibration factor | leave-one-farm-out factor (L0.5, PF, L0) in 0.55 to 0.95 at 5 of 6, with largest minus smallest under 0.15 |
| A5 Coverage | the calibrated, widened nominal 80% interval covers the injected capacity in at least 60% of L0.5 injections (six and native worlds pooled) |

Tier B (decides whether capacity can be read through the fold; D8, D12, and D22 stay report-only):

| Bar | Pass when |
|---|---|
| B1 Crossing worlds | L1 and L2: calibrated recovery of FA in 0.75 to 1.25 at 4 of 6, in both the PF arm and the floor arm |
| B2 Export-dominated | L6, PF arm: calibrated recovery of FA in 0.75 to 1.25 at 4 of 6 |
| B3 Fold beats truncation | at L2, PF arm: median absolute (calibrated recovery − 1) of FA at least 0.10 below L1's; otherwise L1 is the method and the fold model is dropped |
| B4 Sign placebo | the sun-rising-load world: the absolute FA read is under 0.25 of the absolute L6 real recovery at 5 of 6, and the crossing flag is set on under 10% of cells |
| B5 Coverage | at least 60% in L1 and L2 (PF arm) |
| B6 Crossing map (gates publication of the map, not the capacity) | balanced accuracy of the per-cell crossing call against the true sign at least 0.80 in L1 and L2, and false crossings on at most 10% of L0.5 cells |
| C1 Start grid (diagnostic) | under 10% of FA fits end on a bound (all fold capacities zero, or most cell levels at their lower bound) |

At D5, whose midday reading is 0.89 MW, the 0.5-, 1-, and 2-times worlds all inject the 2.0 MW
floor, and the load-shift boxcar's height stays capped at the midday reading. A1 divides a ratio
only where the signed recovery exceeds 0.05 in magnitude.

The zero-reactive-power arm, the cloud-following load (reported as its read over the real-farm
read at 2 times), the shape mix, and K = 1 against K = 4 are report-only rows. A bar is evaluated
on the 2026-06 cutoff first, and the pooled cutoffs are reported beside it.

## Stop rules

1. **Timing.** If one GPU fit takes over 60 s, drop the zero-reactive-power and floor arms; over 90
   s, also drop the 2026-03 cutoff.
2. **After 2026-06.** If A1 fails, only the four twins of 2026-03 are fitted (for the real-data
   table) and no 2026-03 injection runs. If any PF-arm placebo exceeds 0.25 at 2 or more
   Primaries, stop Tier B.
3. **Tier B early exit.** If B1's median calibrated recovery at 2026-06 is below 0.5 or above 1.5,
   skip the 2026-03 Tier B worlds.
4. **Ceilings.** 2 GPU-hours and 8 wall-clock hours; at most 4 CPU processes; one GPU process at a
   time, after checking the GPU for experiments 654, 658, 659, and 661.

## Standing practices

Cleaned data everywhere; D15 and D18 are not used here; the experiment scores no forecast, so the
forecast-score practices (climatology row, tails, example forecasts, load-shift plots) do not
apply. Anonymisation: solar farms appear only as Solar A to F.
