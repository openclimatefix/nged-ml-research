---
id: "00671"
slug: pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: probe results and figures only, in this experiment's private folder. Serving
  path: not touched. Degradation rule: not touched. Designs: Part 1 is a re-scoring of stored
  experiment 662 probe outputs with a denominator fixed here; Part 2 re-probes stored networks, and
  the two fixed-reactive-term settings are listed below, so one design is defensible. Callers: the
  scripts import experiment 662's modules and edit none of them. Asset graph: not touched. CPU
  only, at most 2 processes while the maintainer's backtests run; ceiling 4 wall-clock hours.
hypothesis: >-
  Experiment 662's bar A1 failed because its denominator was the fold model's recovery on a
  signed target that never folds, where the fold model collapses. With the linear probe's
  signed-world recovery as the denominator, the apparent-power conversion costs no recovery at
  Primaries where the floor does not apply. The linear and fold probes disagree at D3 and D4 on
  the real twins because the fold model reads extra capacity from its reactive term, not because
  the curve is folded.
prediction: >-
  Fixed before any number is read. A1' passes for L0 and L1 at 0.5 times and fails for them at 1
  and 2 times, because experiment 662's linear probe falls to -0.12 at 2 times in the PF world
  while staying at 0.78 in the signed world. A1' for FA passes at 0.5 and 1 times and is uncertain
  at 2 times. D2 to D21 curves are monotone falling or flat, not V-shaped, and at D3 and D4 more
  than half of the fold model's extra capacity over the linear probe disappears when the reactive
  terms are fixed at zero.
decision_rule: >-
  Fixed before any number is read. A1' is the PF recovery of probe X (L0, L1, FA) divided by the
  signed-world recovery of the linear probe L0 at the same Primary and level (0.5, 1, 2 times), per
  Primary, median over seeds, with the signed-world denominator kept only where its absolute value
  is at least 0.05. A1' passes for a probe at a level when at least 5 of the 6 MW Primaries lie in
  0.90 to 1.10. If A1' passes for FA at all three levels, the fold model has no fold-free loss and
  experiment 662's failure was the denominator. If it passes only for L0, recommend L0 for the five
  Primaries that do not fold and FA only for D12-type exporters. If it passes for neither,
  quantify the loss of recovery in percentage points by Primary. A second reading divides FA's PF
  recovery by L0's signed recovery and reads the ratio as a bias statement. Part 2 ends with a
  one-table recommendation per Primary: probe, capacity range with the calibration assumption,
  confidence (high, provisional, report-only). No bar is changed after a number is read.
benchmark: >-
  The linear probe L0 in the signed world of the same farm and level (Part 1). The known-answer
  recoveries in native PF worlds at D2, D3, D4, D19, and D21 (A2 rows of experiment 662) and the
  injected farms at D5 and D10 (Part 2).
row_set_id: >-
  Experiment 662's stored 2026-06 probe outputs and networks (five seeds; development data only,
  nothing dated 2026-07-01 or later).
metric: >-
  Recovery ratios (A1'), effective AC capacity in MW per Primary and probe, and the share of the
  fold model's extra capacity that comes from its reactive terms.
production_status: not-for-production
---

# Does the fold model lose recovery, and which probe should be trusted at D2, D3, D4, D19, and D21?

**Plan only; no number has been read.** Experiment 662's matched-level bar A1 divided the fold
model's apparent-power recovery by the fold model's own signed-world recovery. On a signed target
that never folds, the fold model collapses (ratios 2.1 to 3.8 at 2 times), so A1 compared a probe
with itself under different conditions. This experiment rescores A1 against the linear probe's
signed-world recovery and explains why the linear and fold probes disagree at D3 and D4.

## Part 1: A1' with a fair denominator

- **Contrast.** For X in L0, L1, and FA at 0.5, 1, and 2 times: PF recovery of X over signed-world
  recovery of L0, per MW Primary, 2026-06, median over seeds and the two farm rotations where they
  exist. Bar: 0.90 to 1.10 at 5 of 6.
- **Second reading.** FA(PF) over L0(signed), read as a bias statement.
- **Inputs.** Experiment 662's stored probe results; nothing is re-probed. D5 and D10 are reported
  separately from D9, D11, D14, and D16, because 662's audit says every injected world at D5 and
  D10 sits past the fold.

## Part 2: D3 and D4

- **(a)** S(u) from one probe cell per Primary at D2, D3, D4, D19, D21, with the linear and fold
  fits overlaid, from the stored uninjected networks (seed 0). D5 and D10 use an injected network
  (`pf_fleet_l05_r0`, 2.0 MW).
- **(b)** FA refitted with the reactive terms fixed: arm Z has q0 = q1 = 0; arm N has q0 = 0 and
  q1 = 0.25, the power-factor 0.97 reactive share of the native worlds, which is also the share a
  night-time reading implies. The share of the extra capacity that comes from the reactive terms is
  (FA free minus FA with Z) over (FA free minus L0).
- **(c)** Both probes' capacities beside the A2 known-answer recoveries in native PF worlds.
- **(d)** A trust statement per Primary.

## Method

- **Cadence and compute.** No network is trained. CPU only; at most 2 processes. Scripts
  `e00671_*.py` import experiment 662's modules and edit none of them.
- **Weather.** The probes use experiment 662's donor-day weather (ERA5 and CAMS rows of experiment
  554); no weather is used at lag times.
- **Anonymisation.** No generator series appears.
