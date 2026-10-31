---
id: "00444"
slug: wind-idle-detectors-label-free-bake-off
title: >-
  Which label-free detectors find idle half-hours at the three metered wind farms, at any wind
  speed, without flagging genuine low output?
theme: generator-forecasting
also_relevant_to: []
status: planned
kind: other
date: 2026-10-31
size: >-
  medium. Stored: tables and aggregate figures under
  data/ml-experiments/00444-wind-idle-detectors-label-free-bake-off/ only; no forecast is written.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one pseudo-label
  construction is defensible, so this plan fixes the construction, the gates, and the decision rule
  before any score is read. Callers: this folder's scripts only, which read experiment 210's
  `joined.parquet`, `report.json` and `mask.parquet`, and import experiment 431's feature code.
  One Sonnet plan review, one Sonnet code review before scoring, one Sonnet results review, and
  two Sonnet write-up reviews. No fit beyond a logistic regression and a binned power curve on
  CPU.
hypothesis: >-
  A detector that reads only the farm's own power series (D1, texture), possibly smoothed by a
  two-state duration model (D4), finds the idle half-hours that the experiment 210 mask misses
  (idle readings at moderate and high wind), without flagging genuine low output. The wind-based
  residual detector D3 finds the same half-hours but cannot tell them from genuine low output at
  high wind, so it is only useful in combination with D1. No label-free evidence can show that a
  flagged half-hour at high wind is an outage and not an ERA5 error.
prediction: >-
  Written before any score is read. P1: at Farm B and Farm C, D1 and D4 pass the gate below on
  the injection and synthetic-low-output pseudo-labels. P2: at Farm A, D1 passes the
  injection recall gate but fails the false-drop gate on synthetic low output (its genuine low
  output and idle floor overlap in level, and the scaled synthetic rows have shrunken
  differences, which makes the test hard). P3: D4 raises recall over D1 by at least 5 percentage
  points for 3-hour to 12-hour blocks and does not for 96-hour blocks. P4: D3 flags 90% or more
  of synthetic low output at ERA5 wind of 6 m/s or more, so it fails the false-drop gate on its
  own; the combination D1 AND D3 passes it. P5: the experiment 210 mask recalls under 20% of
  injected idle blocks shorter than 12 hours and almost all blocks of 12 hours or more. P6: at
  Farm B, at least 70% of rows with negative power at ERA5 wind above 1.5 times cut-in are flagged
  by D1 (the sign is not an input to D1).
decision_rule: >-
  Fixed before any score is read. **Rows and folds.** All half-hours from 2019-09-17 to 2026-09-20
  in experiment 210's `joined.parquet` (ERA5 100 m wind, power, capacity), farms W1, W2, W3
  (written Farm A, Farm B, Farm C). Two folds by calendar-year parity as in experiment 210:
  tune on odd years and test on even years, and tune on even years and test on odd years; the two
  test halves are pooled and every interval resamples calendar-year blocks (2,000 draws), or
  injected blocks where the unit is a block. **Pseudo-labels (none is ground truth).** Sure idle
  (SI): ERA5 below 2 m/s and power inside experiment 210's idle band. Sure generating (SG): ERA5
  at least 6 m/s and power 3% to 8% of capacity. Synthetic low output (G0s): real runs of 6 to 48
  hours with ERA5 at least 6 m/s and power at least 15% of capacity, rescaled by one factor per
  run so that the run's mean power is drawn uniformly from 2% to 6% of capacity (the rescale also
  shrinks the half-hour differences, so G0s is hard for a texture detector; stated as a
  limitation). Injected idle (INJ): blocks of the farm's own calm floor (SI rows of the same
  calendar year, pasted as consecutive pieces of 12 half-hours in random order) replace the
  reading in 10 replicates per fold; each replicate holds 4 blocks per cell of block length
  {3, 12, 24, 48, 96 hours} by ERA5 bin {moderate: 1.5 times the used cut-in to 8 m/s; high: above
  8 m/s} wherever the original block was contiguous, had no in-band reading within 6 hours, and
  had every half-hour at ERA5 at least 1.5 times cut-in; blocks are at least 24 hours apart; a
  cell with fewer than 10 placed blocks per fold is reported as thin. Rolling features are
  recomputed on the injected series, and the ±12 hours around a block are excluded from
  false-drop counts. Further label-free checks: cut-in curve (flagged share by 1 m/s ERA5 bin
  from real data); cut-out rows (ERA5 at least 22 m/s and power in the idle band, counts reported,
  no interval if under 30 rows); Farm B negative-power rows; agreement of D1 with D3 (2 by 2
  table); Farm B against Farm C agreement. **Candidates.** R0, the experiment 210 mask as
  ported (ERA5 below the fitted threshold, or an unbroken in-band run of 12 hours or more at ERA5
  above 1.5 times the used cut-in). R1, the level-only rule (power inside the idle band). I2, the
  run-length rule (in-band, ERA5 above 1.5 times cut-in, unbroken for at least 24 hours; the
  weekday covariate is optional and is not used). D1, texture: a class-balanced logistic
  regression (L2, C = 1) on experiment 431's level-free statistics only (rolling standard
  deviation of first differences, lag-1 and lag-2 autocorrelation, and share of repeated values,
  over 3-hour and 24-hour trailing and centred windows; no ERA5, no level, no sign), trained per
  farm on the tune years with SI as the positive class against SG plus G0s. D1a, the single
  statistic with the best level-matched AUC in experiment 431 (3-hour centred standard deviation of
  first differences at Farm B; 24-hour centred share of repeats at Farm C; the same statistic as
  Farm C at Farm A), as a simple-rule reference. D2 is not a separate detector: it splits D1's
  flags into I1 (ERA5 below cut-in) and I2 candidates (ERA5 above 1.5 times cut-in) and reports the
  census of each. D3, the residual rule: observed 6-half-hour centred mean power divided by the
  expected power from a 0.5 m/s-binned ERA5 median power curve fitted on the other years' rows
  (cross-fitted); flag if expected power is at least 15% of capacity and the ratio is at most
  0.10 (sweep: 0.05 and 0.20). D1&D3, flagged by both. D4, a two-state Viterbi smoothing of D1's
  log-odds with a stay probability of 0.95 per half-hour (sweep: 0.90, 0.98), with a bias offset
  chosen on the tune years like D1's threshold. D5 (capacity change points), D6 (neural day
  labeller), and anything needing hand labels are not run. **Thresholds.** D1, D1a, and D4's
  threshold or offset is set per farm on the tune years so that the false-drop rate on
  (SG plus G0s, equal weight) is 5%. R0, R1, I2, and D3 have no threshold to tune. **Gate (per
  farm, test folds pooled).** A candidate passes strictly if: (a) pooled recall over injected
  blocks (all 10 cells, equal weight per cell) is at least 0.80; (b) the false-drop rate on SG plus
  G0s (equal weight) is at most 0.05, and passes leniently if it is at most 0.08; (c) its recall
  exceeds R0's recall at that farm by at least 0.10 on the same injected blocks. A farm-level
  verdict for D1 or D4 is `survives` (strict at that farm), `survives leniently`, or `fails`. The
  family verdict is `supported` if D1 or D4 survives strictly at two or more farms, `partly
  supported` if it survives at one farm strictly or at two or more leniently, and `not supported`
  otherwise. **Reported beside the gate:** recall by block length and wind bin; false drop on SG
  and on G0s separately; the flag rate on clearly generating rows (power at least 10% of capacity
  and ERA5 at least 1.5 times cut-in, outside the ±12-hour margin); recall on cut-out rows and on
  Farm B negative-power rows; level-matched AUC on SG against SI (as experiment 431 defined it);
  paired recall differences with cluster-bootstrap intervals over injected blocks; the minimum
  detectable effect, taken as 2.8 times the bootstrap standard error of a paired recall
  difference, and the same for false-drop differences from the year-block bootstrap; and the
  expected width of a recall interval from 300 hand-labelled farm-days (100 per farm) given the
  prevalence of D4-flagged farm-days at ERA5 above 1.5 times cut-in. **What this cannot answer
  (written to the page):** injected blocks are easier than real idle episodes (an exact copy of
  the calm floor, hard edges, no ambiguity at the ends), so injection recall is an upper bound;
  G0s depends on the rescale; SG and G0s define genuine output by ERA5 and power only; and no
  pseudo-label can show that a flagged half-hour at moderate or high wind is a whole-farm outage
  and not an ERA5 error or a partly available farm. A hand-labelled sample is the only route to
  those, and the page says what it would add. No gain threshold applies, because no forecast is
  scored.
  **Amendment after the plan review, before any code or score exists.** (1) Injected floor pieces
  come from the opposite-parity years (the floor drifts across years, so this is a transfer test),
  and recall on injected blocks is a sanity floor, not evidence of separation: R1 and I2 recall
  them by construction. (2) The false-drop gate uses three negative sets with equal weight: SG, SGm
  (real rows with ERA5 from 1.5 times cut-in to 6 m/s and power 3% to 8% of capacity), and G0s;
  it is a calibration and transfer check, and the verdict rests on SGm and G0s as well as SG. SG
  false drop is also reported excluding rows that D3 flags. (3) Gate (c) is read at matched false
  drop: R0's false-drop rate is reported, and D1's recall is also read at R0's false-drop rate.
  (4) The gated candidates are D1 with the defaults above and D4 with stay probability 0.95, each
  judged on its own per farm; sweeps (D3 0.05 and 0.20, D4 0.90 and 0.98, D4 posterior thresholds
  0.3, 0.5, 0.7) are sensitivity only; D1&D3 is reported and does not enter the verdict. There is
  one gate (the strict one); the 0.08 figure is reported descriptively. The family verdict counts
  farms where D1 passes, and separately where D4 passes. (5) Cells with fewer than 10 placed blocks
  per fold are dropped from the recall denominator, and the dropped cells are listed. (6) Recall
  intervals resample calendar-year blocks, because replicates reuse source locations. (7) The idle
  band is refitted on the tune years only in each fold; D3's power curve uses the tune years only;
  scoring drops rows within 24 hours of a calendar-year boundary. (8) Recall is also reported
  without the half-hours adjacent to a pasted join. (9) The D1 stop rule and the I2 weekday test
  were run in experiment 431 and are not repeated; the D1 and D3 disagreement slice (G4) is
  reported.
benchmark: >-
  The experiment 210 mask (R0) and the level-only rule (R1), on the same pseudo-labels. No
  forecast is scored.
row_set_id: >-
  all half-hours of the three wind farms from 2019-09-17 to 2026-09-20 with a power reading, as in
  experiments 210 and 431; not comparable with the leaderboard
metric: >-
  Recall on injected idle blocks and false-drop rate on sure-generating plus synthetic low-output
  rows, with calendar-year-block and injected-block bootstrap intervals.
code: ml-experiments/00444-wind-idle-detectors-label-free-bake-off/
production_status: not-for-production
adopted_in: none
---

Planned. This page is the pre-registration for the label-free part of Stage 1 of the wind idle-noise
detector design. Results, figures, and the take-home message are added after the scoring run.

## Plan notes

- **Standing practices followed.** Cleaned data: the three customer-generator series are never
  flagged by the production cleaning rule, so no `substation_zero` rows are dropped. No forecast is
  fitted, so cadence, saved boosters, and the 5-day ERA5 delay do not apply; ERA5 is a
  hindsight-only input to D2, D3, R0, I2, and to the pseudo-labels, and D1 and D4 read the power
  series only, so they are usable as-of.
- **Reused code.** Experiment 431's `add_features` (imported with that folder on `PYTHONPATH`),
  experiment 210's report (cut-in, idle band, 12-hour outage rule).
- **Not asked of anyone.** No hand labels are requested.
- **Anonymisation.** Farm A, Farm B, and Farm C only. Published figures show aggregates (recall,
  false-drop rates, shares by wind bin); no farm-day, no capacity, no raw series.
- **Compute.** CPU only, one process, `OMP_NUM_THREADS=1`, `MemoryMax=8G`.
