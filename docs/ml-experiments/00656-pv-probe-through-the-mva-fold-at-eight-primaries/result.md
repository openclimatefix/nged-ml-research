---
id: "00656"
slug: pv-probe-through-the-mva-fold-at-eight-primaries
theme: solar-disaggregation
also_relevant_to: []
status: planned
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
---

# Can the PV probe read capacity through the MVA fold at the eight MVA Primaries?

**Status: planned. The specification is the Opus design `briefs/pv_mva_bounce_opus_answer.md` of
2026-11-09 (private folder), sections 2 and 3, and every bar below is copied from it before any
number is read.**

We ask whether the unmetered solar capacity behind a substation can be read from a substation
whose meter reports apparent power (MVA, never negative) rather than signed real power (MW). We
turn the six signed-MW Primaries into synthetic MVA Primaries with a known solar farm added, and
check how much of it the probe finds.

## Design in brief

- **Observation model.** The apparent power is the length of a real-power part, demand minus
  `C·u`, and a reactive-power part `Q`. Three probes are compared: the linear probe of experiment
  621 (L0), a one-sided linear probe that drops rows near the fold (L1), and a fold model (F) with
  four unit curves, plus F with a prior on demand from night readings (F_anchored).
- **Known answer.** At the six MW Primaries, the target becomes `|S'| = sqrt((P − I)² + Q²)`, with
  `P` the cleaned signed MW, `I` a fleet-shaped injection, and `Q` from one of three reactive-power
  arms (power factor 0.97 on demand, constant floor, zero). Injection levels are L0.5, L1, L2, and
  L6 times the Primary's median sunny-midday reading. The five MVA Primaries that do not fold
  (D2, D3, D4, D19, D21) receive the same world through `P = 0.97·|S|`, `Q = 0.243·|S|`.
- **Placebos.** Climatological profile, load-shift boxcar, sun-rising load, cloud-following load.
- **Conditions.** 24 conditions × 3 seeds × 2 cutoffs (2026-03, 2026-06) = 144 fits, 2026-06 first.

## Pass bars (fixed before any read-out)

Bars are relaxed in the same way as experiment 621's after its result: a calibrated figure sits
beside the raw one, intervals are widened by the leave-one-farm-out spread, coverage must reach 60%,
and placebos must stay under 0.25.

Tier A (gates the capacity table for D2, D3, D4, D19, and D21):

| Bar | Pass when |
|---|---|
| A1 Fold-free loss | PF arm, L0.5: median fold-loss ratio (arms F and L0) in 0.90 to 1.10 at 5 of the 6 MW Primaries |
| A2 Native worlds | PF arm, L0.5, native: raw recovery within ±0.12 of the six's PF L0.5 recovery at 4 of the 5 |
| A3 Placebos | climatological and boxcar placebos: 95th percentile under 0.25 of the real L0.5 recovery at 5 of 6, and at 4 of the 5 in native worlds |
| A4 Calibration factor | leave-one-farm-out factor (L0.5, PF) in 0.55 to 0.95 at 5 of 6, with largest minus smallest under 0.15 |
| A5 Coverage | the calibrated, widened nominal 80% interval covers the injected capacity in at least 60% of L0.5 injections (six and native worlds pooled) |

Tier B (decides whether capacity can be read through the fold; D8, D12, and D22 stay report-only):

| Bar | Pass when |
|---|---|
| B1 Crossing worlds | L1 and L2: calibrated recovery of F_anchored in 0.75 to 1.25 at 4 of 6, in both the PF arm and the floor arm |
| B2 Export-dominated | L6, PF arm: calibrated recovery of F_anchored in 0.75 to 1.25 at 4 of 6 |
| B3 Fold beats truncation | at L2, PF arm: median abs(calibrated recovery − 1) of F_anchored at least 0.10 below L1's; otherwise L1 is the method and F is dropped |
| B4 Sign placebo | the sun-rising-load world: F_anchored reads under 0.25 of the L6 real recovery at 5 of 6, and the crossing posterior exceeds 0.5 on under 10% of cells |
| B5 Coverage | at least 60% in L1 and L2 (PF arm) |
| B6 Crossing map (gates publication of the map, not the capacity) | balanced accuracy of the per-cell crossing call against the true sign at least 0.80 in L1 and L2, and false crossings on at most 10% of L0.5 cells |

The Q0 arm, the cloud-following load, the shape mix, and K = 1 against K = 4 are report-only rows.
The Bonferroni count is the 11 bars above; each is a pass-count bar rather than an interval, so no
interval level is corrected.

## Stop rules

1. **Stage 0 (probe the stored networks, no fit).** Stop if, at 3 or more of the five Primaries
   that do not fold, the seed standard deviation of L0's C on the stored networks exceeds 0.5 MW, or
   S(u) is not monotone falling.
2. **Timing.** If one GPU fit takes over 60 s, drop the Q0 and floor arms (17 conditions); over
   90 s, also drop the 2026-03 cutoff.
3. **After 2026-06.** If A1 fails, stop. If any PF-arm placebo exceeds 0.25 at 2 or more Primaries,
   stop Tier B.
4. **Tier B early exit.** If B1's median calibrated recovery at 2026-06 is below 0.5 or above 1.5,
   skip the 2026-03 Tier B worlds.
5. **Ceilings.** 3 GPU-hours and 8 wall-clock hours; at most 4 CPU processes.

## Standing practices

Cleaned data everywhere; D15 and D18 are not used here; the experiment scores no forecast, so the
forecast-score practices (climatology row, tails, example forecasts, load-shift plots) do not
apply. Monthly refit is replaced by the two fixed cutoffs of experiment 621. Anonymisation: solar
farms appear only as Solar A to F.
