---
id: "00679"
slug: pv-mva-conversion-loss-reactive-floor-at-half-level
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights, probe results and a loss table in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: the test is
  experiment 662's, with two reactive-power arms moved to a new level, so one design is defensible.
  Callers: this folder's script imports experiment 662's fit and probe modules and edits none of
  them. Asset graph: not touched. About 0.4 GPU-hours (9 networks) and 2 wall-clock hours.
hypothesis: >-
  NGED say to assume power factor 1 with a small always-present reactive floor. At the injection
  level that matches the five MVA Primaries that never fold (0.5 times the midday reading), the
  conversion loss of the linear probe (signed-world recovery minus apparent-power-world recovery) is
  smaller under a reactive-floor world than under the power-factor 0.97 world that experiments 656,
  662, and 671 used.
prediction: >-
  Fixed before any recovery is read. The floor arm's median conversion loss of the linear probe L0
  at 0.5 times is under 5 percentage points, because experiment 673 measured 10 points at 1 times
  in the floor arm (0.63 against 0.78 signed) and 25 points in the PF arm (0.53), and the loss
  shrinks as the injected farm gets smaller relative to the reactive floor. The zero-reactive arm's
  loss is below the floor arm's.
decision_rule: >-
  On the median over the six MW Primaries of L0's conversion loss at 0.5 times in the reactive-floor
  arm: under 5 points, recommend that the NGED capacity page's calibration for the five MVA
  Primaries use the signed-world factor 0.68 alone, with the loss stated as negligible; 5 to 12
  points, keep experiment 671's wording; above 12 points, say the PF-arm figure was not an
  overestimate. No rescue tuning. The threshold is a reporting threshold, not a porting threshold,
  because nothing here is ported to production.
benchmark: >-
  Signed-world L0 recovery of the same farm at the same level, and the PF arm's loss at 0.5 times
  from experiment 662.
row_set_id: >-
  Not a forecast score. Pretraining rows and probe dates are experiment 662's at the 2026-06 cutoff
  (development data only; nothing dated 2026-07-01 or later).
metric: >-
  Recovery in MW and in energy, per Primary and as the median over the six MW Primaries, with the
  day-block bootstrap widened by the between-seed spread (experiment 662's interval method).
---

# Planned: how much capacity does the probe lose when the substation reports apparent power with a small reactive floor?

**This page is a plan. It holds no result yet.**

- **Question.** Experiments 656, 662, and 671 measured the loss of the linear probe's recovery under
  apparent power mainly in the power-factor arm (reactive power 0.25 times the median real power of
  the same half-hour of the week). NGED's statement is power factor 1 with a small always-present
  reactive floor. Experiment 673 has the reactive-floor arm only at 1 and 2 times and the
  zero-reactive arm only at 2 times. This experiment adds both arms at 0.5 times, the level that
  matches the five non-folding MVA Primaries (D2, D3, D4, D19, and D21).
- **Arms (2026-06 cutoff only, seeds 0 to 2, the six MW Primaries).**
  - Reactive-floor arm at 0.5 times: `floor_fleet_l05_r0`, new, against the reused `floor_twin`.
  - Zero-reactive arm at 0.5 and 1 times: `q0_fleet_l05_r0` and `q0_fleet_l1_r0`, new, against the
    reused `q0_twin`.
  - Signed world at 0.5 and 1 times, and the PF arm at 0.5 and 1 times: experiment 662's networks,
    reused for seeds 0 to 2.
  - Probes L0, L1, and FA, as in experiment 671. Same injection (fleet, 2.0 MW floor) as experiment
    662.
- **Reused code.** `ml-experiments/00662-pv-probe-through-the-mva-fold-matched-level-redesign/`
  (fit, probe, and analysis), edited not at all.
- **Native worlds at the five MVA Primaries in the floor arm.** Experiment 662's target builder
  makes native worlds only in the PF arm, so the floor-arm native read is not supported without
  editing 662's code and is not run.
- **Tails and curtailment.** The experiment scores no forecast, so the tail and curtailment reports
  do not apply.
- **Weather.** The networks use ERA5 and CAMS weather at training and probe time, as in experiment
  662. The probe is a diagnostic of a trained network and is not a forecast, so forecast-time
  availability does not apply.
- **Reviews.** A fresh Sonnet agent reviews the code before the first fit, another checks the
  results, and a third reviews the write-up.
