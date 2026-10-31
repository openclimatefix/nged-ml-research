---
id: "00444"
slug: wind-idle-detectors-label-free-bake-off
title: >-
  Which label-free detectors find idle half-hours at the three metered wind farms, at any wind
  speed, without flagging genuine low output?
theme: generator-forecasting
also_relevant_to: []
status: finished
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
verdict: not supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  None passes the gate. Closest of the two gated candidates: D4 (texture, smoothed) at Farm A,
  recall 0.72 (0.70 to 0.75) at a false-drop rate of 0.058. The experiment 210 mask recalls every
  injected block of 12 hours or more and none of 3 hours, at false-drop rates of 0.07, 0.20, and
  0.44.
headline: >-
  The texture rules tested (D1, and D4 which smooths D1) find injected idle blocks of 12 hours or
  more (recall 0.74 to 0.90 across the three farms) but miss 3-hour blocks (0.08 to 0.33), and they
  flag 6% to 12% of the genuine and synthetic low-output half-hours. Neither passes the
  pre-registered gate at any farm. The experiment 210 mask recalls every injected block of 12 hours
  or more, at higher false-drop rates (0.20 at Farm B, 0.44 at Farm C). The pseudo-labels cannot
  show whether a flagged half-hour at moderate wind is an outage or an error in the ERA5 wind.
lessons: >-
  Injected recall flatters every candidate, because the pasted floor is an exact copy of calm-night
  readings and the level-only rule and the experiment 210 outage rule recall it automatically, yet
  the texture rules reach only 0.58 to 0.72. Tightening D1's false-drop rate from 10% to 1% costs
  little recall at Farm A (0.73 to 0.68) and much at Farm B and Farm C (0.61 to 0.42, 0.75 to
  0.40).
take_home_message: >-
  The texture rules tested here, which read only a wind farm's own power readings, find pasted idle
  stretches of 12 hours or more but miss stretches of 3 hours, and they flag 6% to 12% of the
  genuine and synthetic low-output half-hours. The existing experiment 210 mask also finds every
  long stretch, and flags more genuine low output at two farms, so these rules do not clearly
  improve on it. None of the pseudo-labels can show whether a flagged stretch at moderate wind is
  a farm outage or an error in the weather estimate, so hand labels or outage records from the
  network operator are the way to settle that.
advice:
  - >-
    Keep the experiment 210 mask for idle stretches of 12 hours or more until a detector beats it
    on hand labels, because the texture rules tested here did not pass the gate (gate table).
  - >-
    Treat idle stretches of 3 hours at moderate and high wind as missed by the texture rules tested
    (recall 0.08 to 0.33), and do not claim a detector for them without hand labels (recall by
    block length).
  - >-
    Use the combination of D1 and the power-curve residual rule D3 as a flag with a low false-drop
    rate on the three low-output sets (0.1% to 1.5%), not as a mask, because its recall is 0.44 to
    0.59 (gate table).
  - >-
    Judge any later idle detector on injected blocks, synthetic low output, and real low output at
    wind below 6 m/s together, because each alone passes or fails a detector automatically
    (false-drop figure).
---

**The texture rules tested here do not clearly improve on the experiment 210 mask.** The texture
detector D1 is a logistic regression on the shape of a wind farm's power readings (how much
consecutive half-hours differ and how often they repeat), ignoring the level. D4 smooths D1's
flags across neighbouring half-hours. Recall is the share of injected idle half-hours a detector
flags. False drop is the share of genuine or synthetic low-output half-hours it flags wrongly.
D1 finds injected idle blocks of 12 hours or more (recall 0.74 to 0.90 across the three farms) but
misses 3-hour blocks (0.08 to 0.33), with false-drop rates of 0.06 to 0.12. The experiment 210 mask
(an ERA5 wind threshold plus a rule for unbroken 12-hour runs of low readings) recalls every
injected block of 12 hours or more. Neither D1 nor D4 passes the pre-registered gate (recall at
least 0.80 and false drop at most 0.05) at any farm, so the verdict is `not supported`.

**The question and what it narrows.** The design for a wind idle-noise detector (Stage 1) asks for
a detector that finds idle half-hours at any wind speed, which the ERA5 wind threshold of
experiment 210 cannot do. ERA5 is the European Centre for Medium-Range Weather Forecasts
fifth-generation reanalysis, an estimate of past weather. This experiment runs every part of
Stage 1 that needs no human label, on three kinds of pseudo-label (none of them ground truth):
idle blocks pasted into generating periods, low output that is genuine (real) or synthetic
(rescaled), and a few real events (storm shutdown, negative readings). The tests run here did not
settle whether the detectors work on real idle stretches, so the next step is hand labels or outage
records from the network operator (National Grid Electricity Distribution, NGED).

![Recall of injected idle blocks against false-drop rate on low output, one panel per farm. The
dashed lines are the gate (recall 0.80, false drop 0.05).](figure_1_headline.svg)

![Recall by length of injected block](figure_2_recall_by_length.svg)

![False-drop rate on each set of low output](figure_3_false_drop.svg)

![Share of real half-hours flagged, by ERA5 wind speed](figure_4_flagged_by_wind.svg)

## Results

All numbers are interim scored output of this experiment. The gate: recall at least 0.80, false
drop at most 0.05, and recall at least 0.10 above the experiment 210 mask's at matched false drop.
Recall is the equal-weight mean over block-length by wind-bin cells; false drop is the equal-weight
mean over the real sets and the synthetic set. Test years only, with every fitted value tuned on
the other parity of years. Intervals resample calendar-year blocks.

| Farm | Candidate | Recall (95% interval) | False drop | Passes |
|---|---|---|---|---|
| A | D1 | 0.72 (0.70 to 0.75) | 0.061 | no |
| A | D4 | 0.72 (0.70 to 0.75) | 0.058 | no |
| A | Experiment 210 mask | 0.75 | 0.066 | not gated |
| B | D1 | 0.58 (0.53 to 0.63) | 0.076 | no |
| B | D4 | 0.59 (0.54 to 0.63) | 0.061 | no |
| B | Experiment 210 mask | 0.75 | 0.200 | not gated |
| C | D1 | 0.66 (0.59 to 0.73) | 0.115 | no |
| C | D4 | 0.66 (0.57 to 0.75) | 0.115 | no |
| C | Experiment 210 mask | 0.71 | 0.441 | not gated |

- **Which parts of the gate fail.** Recall is below 0.80 at all three farms for D1 and D4, and the
  false drop is above 0.05 at all three. At the mask's own false-drop rate, D1 does not exceed the
  mask's recall by 0.10 at any farm. At Farm A the two rates match (0.72 against 0.75). At Farm C
  D1 reaches recall 0.84 at false drop 0.38, against the mask's 0.71 at 0.44. At Farm B D1 was not
  measured at the mask's rate of 0.20, because its false-drop sweep stops at 0.13 (recall 0.61).
- **Short blocks account for the recall shortfall.** D1's recall for blocks of 12 hours or more is
  0.90 at Farm A, about 0.74 at Farm B, and 0.79 at Farm C, against 0.19, 0.08, and 0.33 for 3-hour
  blocks. D4 changes recall by 0.00 to 0.01 (its minimum detectable effect over D1 is 0.005 to
  0.023), so smoothing adds nothing here.
- **The experiment 210 mask recalls long blocks automatically.** It recalls every injected block of
  12 hours or more and none of 3 hours, because the pasted blocks are exactly the kind of unbroken
  low-reading run its rule drops (prediction P5). Its false-drop rate is high at Farm B and Farm C
  (0.20 and 0.44), because the synthetic low output is also an unbroken run of low readings at
  high wind. The mask therefore cannot tell a rescaled run from an outage. A real outage may differ
  from a rescaled run in ways these pseudo-labels do not show.
- **The residual rule D3 cannot separate idle from low output at high wind.** D3 compares the
  reading with the power the ERA5 wind implies. It flags 69% to 81% of the synthetic low output
  (false drop 0.23 to 0.38 in total) and recalls 0.56 to 0.64 of the injected blocks. Requiring both
  D1 and D3 gives false drops of 0.1% to 1.5% and recall of 0.44 to 0.59. That combination does not
  enter the verdict.
- **Level-matched area under the receiver operating characteristic curve (AUC) of D1's score**
  (sure-idle against sure-generating rows, test years): 0.92 (0.88 to 1.00) at Farm A, 0.71 (0.66
  to 0.77) at Farm B, and 0.74 (0.69 to 0.79) at Farm C.
- **Real data at high wind.** D1 flags 3.8%, 7.3%, and 20.3% of half-hours with ERA5 wind above
  1.5 times the used cut-in at Farm A, B, and C (the mask flags 0.5%, 0.7%, and 7.4%). At Farm C
  the flags form 86 runs of 24 hours or more and 40 of 48 hours or more, the longest 263 hours.
  D1 also flags 2.8%, 4.6%, and 10.1% of half-hours with power of at least 10% of capacity at high
  wind, so a share of the flags at Farm C are false. Of the 7,270 high-wind half-hours that D1
  flags at Farm B, 3,069 are also flagged at Farm C.
- **Farm B negative readings (evidence of non-generation).** At high wind D1 flags 41% of the 713
  negative readings (D4 35%, the mask 10%). Prediction P6 required 70%, so P6 fails. Over all winds
  D1 flags 57% and the mask 75%.
- **Storm shutdown.** Only 3, 0, and 4 half-hours have ERA5 wind of 22 m/s or more with power in
  the idle band at Farm A, B, and C, which is too few for any rate.
- **Predictions.** P1 fails (neither D1 nor D4 passes at Farm B or Farm C). P2 fails in its
  mechanism: Farm A fails the gate on recall and on the real sure-generating rows (false drop
  0.089), not on the synthetic low output (0.032). P3 fails (D4 raises recall by at most 0.01). P4
  holds in part: D3 flags 69% to 81% of the synthetic low output, short of 90%, and fails alone,
  while D1 and D3 together pass the false-drop gate but fail recall. P5 holds. P6 fails.

## What a hand-labelled sample would add

- **The question the pseudo-labels did not settle** is whether a flagged half-hour at moderate or
  high wind is a whole-farm outage, a partly available farm, or an ERA5 error at a 25 km cell. The
  pseudo-labels here come from the same readings and ERA5 wind that the detectors read.
- **Precision of a recall estimate.** D4 flags at least 6 half-hours at high wind on 8%, 16%, and
  28% of farm-days at Farm A, B, and C. These shares are upper bounds on idle prevalence, because
  D4 also flags some generating rows. A random sample of 100 farm-days per farm would then hold at
  most about 8, 16, and 28 idle days, giving half-widths on a recall of 0.8 of at least about 0.28,
  0.20, and 0.15. A sample of 100 days drawn from days that a detector flags would give about 0.08.
- **Minimum detectable effects.** Paired recall differences smaller than 0.04 to 0.11 (D1 against
  the mask, 2.8 times the standard error over seven year blocks) cannot be told from zero. The
  intervals of D1 minus the mask include zero at Farm A (-0.05 to 0.00) and Farm C (-0.13 to 0.02),
  and exclude it in the mask's favour at Farm B.

## Caveats

- **Injection flatters recall.** The pasted floor is an exact copy of calm-night readings from the
  opposite parity of years, with hard joins every 12 half-hours. Recall excluding the half-hours
  next to a join differs by at most 0.012. The 3-hour and 24-hour windows reach past a join, so
  that exclusion understates the join's effect.
- **The false-drop sets are not independent of the detectors.** The synthetic low output shrinks
  the differences between half-hours along with the level, which should make it look more idle.
  D1 is also trained on part of that set. The real sets (power 3% to 8% of capacity at ERA5 wind of
  6 m/s or more, or of 1.5 times cut-in to 6 m/s) contain any idle rows at high wind that the study
  seeks. The set from 1.5 times cut-in to 6 m/s is empty at Farm A, whose cut-in used is 4.5 m/s.
- **Tune-year thresholds transfer imperfectly.** D1's threshold gives a 5% false drop on the tune
  years and 6% to 12% on the test years. A sweep of D1 at false-drop targets of 1%, 2%, and 10%
  shows that tightening does not recover the gate at Farm B or Farm C.
- **Cut-in and ERA5 are not refitted per fold.** The used cut-in comes from experiment 210's report
  (fitted on all years) and defines the high-wind rows and the mask's outage rule. The idle band,
  wind threshold, power curve, logistic fit, and thresholds are fitted on the tune years only.
- **Seven cells had fewer than 10 blocks per fold** and were dropped from the recall average: the
  48-hour and 96-hour moderate-wind cells at Farm A and Farm B, and the 24-hour, 48-hour, and
  96-hour moderate-wind cells at Farm C. Ten replicates re-pick the same locations, so intervals
  resample calendar years, of which there are about seven.
- **One texture rule was tested.** D1 is one logistic regression on one set of statistics, and D1a
  (the single best statistic of experiment 431) is in the stored tables only. Other texture rules
  may do better.
- **Checks.** No forecast is scored, so the median-minus-mean, tail, and curtailment checks do not
  apply, and none of the audit triggers applies. The three customer-generator series are never
  flagged by the production cleaning rule, so no `substation_zero` rows were dropped.

## What to try next

- **Offer hand labels or ask NGED** for outage dates and a reactive-power channel; the tests run
  here did not settle the question.
- **Keep the experiment 210 mask** for the training and evaluation row sets for now.
- **Stage 5 of the design (capacity change points)** was not run here and is the next label-free
  item.

## Experiment details

- **Code:** `ml-experiments/00444-wind-idle-detectors-label-free-bake-off/` (`e00444_bakeoff.py`,
  `e00444_summarise.py`, `e00444_charts.py`), importing experiment 431's feature code
  (`ml-experiments/00431-wind-idle-data-look/`) and experiment 210's wind-threshold fit
  (`ml-experiments/00210-wind-idle-noise-mva-artefact/`). Data: the experiment's folder under
  `data/ml-experiments/` (private).
- **Cadence, weather, and cleaning:** no forecast and no refit. ERA5 100 m wind (research-only,
  about 5 days late) is used by the pseudo-labels, D3, and the mask; D1 and D4 read the power
  series only and are usable as-of.
- **Reviews:** one Sonnet plan review (nine findings, all accepted and written into the plan before
  any code), one Sonnet code review before scoring (ten findings; eight fixed, one documented, one
  rejected as harmless), one Sonnet numbers review (sixteen findings, applied), and one Sonnet
  prose review (applied).
