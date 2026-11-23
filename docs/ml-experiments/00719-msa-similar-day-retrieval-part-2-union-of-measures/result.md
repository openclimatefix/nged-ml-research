---
id: "00719"
slug: msa-similar-day-retrieval-part-2-union-of-measures
title: >-
  Does the axial-attention network forecast a Primary better when its history rows mix the
  weather-nearest, most recent, and same-weekday past days?
theme: lagged-power-and-error-correction
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts and network weights for about 70 fits (part 2 of two);
  (2) more than one retrieval rule defensibly tests the hypothesis, so a first experiment tests the
  first rule; (3) it copies training code from experiment 713, and the callers of that code cannot
  be listed without searching the repository; (4) it touches no serving path or degradation rule;
  (5) it changes no contract. Full set of reviews: plan, code (Opus, each new script, twice), results,
  write-up, prose.
hypothesis: >-
  A union of the 5 weather-nearest, 5 most recent, and 4 same-weekday blocks of the last 379 days
  lets the axial-attention demand network of experiment 713 forecast days 2 to 10 with a lower
  continuous ranked probability score (CRPS) than the last 14 days, because the three measures
  supply the level, the weekly shape, and the weather response together.
prediction: >-
  Written before any fit. B1 (U minus M*) lies between -0.16 and +0.06 points of capacity at the six
  signed-MW Primaries and between -0.25 and +0.16 at the eight MVA-only Primaries, and B2 (U minus
  U-A) is below zero in both groups. The predicted verdict is `inconclusive` or `small gain`. The
  prediction fails if B1 lies outside those ranges, or if B2 is at or above zero in a group.
decision_rule: >-
  Two counted contrasts, B1 (U minus M*) and B2 (U minus U-A), each at the six and at the eight, so
  four counted readings, each read at 98.75% from a month-block bootstrap, with experiment 713's
  three guards. The smallest gain worth porting is 0.02 points of capacity. Group readings and
  verdicts follow experiment 718's table, which is experiment 713's table with B1 in place of H1 and
  B2 in place of H2. "Harmful placebo" means that U-A minus M* reads "loss" or "clear loss" in a
  group. U27 advances to the four other folds only if U27 minus U27-A is below zero and U27 minus M*
  is at most zero in both groups on the screen folds, and the manager agrees to the hours. If both
  parts of F9 end `inconclusive` and the point estimates of A1 and B1 are at or above zero in both
  groups, the line closes as for `not supported`. Every verdict, headline, and take-home message
  states that experiment 713's known-answer gate was waived and that every score is a ceiling for a
  live forecast.
benchmark: >-
  M* (experiment 713's selected arm, M1, stored seeds 0 and 1 of the screen folds); P (experiment
  713's placebo); the Candidate (P7+D50 of experiment 667); the month-by-half-hour climatology; the
  manual heuristic and BG2 where stored rows cover the scored keys.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity on leads from 24 h up to but not including 240 h after the 12:00 UTC
  origin, with the mean absolute error of the median forecast normalised by capacity beside it; q01
  and q99 pinball loss; cold winter nights; the read-out weights by age and by measure; the four
  load-shift scenarios.
---

**We ask whether the axial-attention network of experiment 713 forecasts a Primary substation's
demand better when its 14 history rows are a union of the 5 weather-nearest, the 5 most recent, and
the 4 same-weekday blocks of the last 379 days, instead of the last 14 days.** This is part 2 of
follow-up F9. Part 1
([experiment 718](../00718-msa-similar-day-retrieval-part-1-weather-distance/result.md)) tests the
weather distance alone over 91 days and fixes the arms, the code, and the decision table that this
page reuses. This page is the plan: no neural network has been fitted, and every number below is a
design value or a figure from an earlier experiment.

**The honest expectation is a tie or a small gain.** In the CPU-only exploration of F9, a plain mean
of the union's rows beat a plain mean of the last 14 days by 0.80 points of MAE at 15 rows, and beat
the best single measure by 0.18. The union was the best of 64 settings reported on the evaluation
folds, on weather that a live service would not hold for the youngest blocks, so the 0.18 is a
hypothesis. Experiment 713's interim screen puts `M*` only 0.13 points (the six) and 0.31 points
(the eight) ahead of 14 random days of the past year.

## Why this runs

**This experiment runs on the maintainer's instruction, as part 1 does.** On 2026-11-23 the
maintainer approved F8 and F9 ("yes, please run these autonomously at the appropriate time.") and,
relayed by the manager, said: "I'd be generous to the similar days idea. I'm optimistic that the MSA
will be able to extract information from similar days that a simpler approach can't." The rules of
[experiment 718](../00718-msa-similar-day-retrieval-part-1-weather-distance/result.md) on experiment
713's verdict, on the confirmation obtained from the maintainer, and on the stop rule apply to this
page unchanged. Experiment 713's section 1.8 closes the "28-row layout" after `not supported`, and
`U27` is a different layout (1 target row and 27 history rows, where experiment 713's layout has 14
target rows and 14 history rows). The maintainer's instruction replaces the manager's earlier
condition that arm `R` as specified would not run.

## What is compared

| Arm | History rows | Rows | Seeds | Role |
|---|---|---|---|---|
| `M*` | Ages 2 to 15 | 15 | Stored | Control; never refitted |
| `U` | 5 weather-nearest, 5 most recent, 4 same-weekday blocks, ages 2 to 379 | 15 | 0, 1 (then 2 to 4) | Treatment |
| `U-A` | The 5 most recent rows of `U`, and 9 random blocks whose ages match the other 9 rows of `U` | 15 | As `U` | Counted placebo |
| `U27` | Ages 2 to 15, plus 7 weather-nearest and 6 same-weekday blocks of ages 16 to 379 | 28 | 0, 1 on 4 folds | Secondary: the largest affordable layout |
| `U27-A` | Ages 2 to 15, plus 13 age-matched random blocks | 28 | 0, 1 on 4 folds | Secondary placebo of `U27` |
| `UX` | `U`, with calendar-nearest blocks of earlier winters for targets from 20 December to 5 January | 15 | 0, 1 on 2 folds | Secondary: the Christmas switch |

**The counted contrasts are B1 (`U` minus `M*`) and B2 (`U` minus `U-A`), each at the six and at the
eight.** B1 compares the union's rows with the last 14 days at equal tokens. B2 compares similar
days with equally old random days beside the same 5 recent rows. `U-A` randomises the weekday rows
as well as the weather rows, so B2 measures weather similarity and weekday similarity together.
`U` minus `P`, `U27` minus `M*`, `U27` minus `U27-A`, `U27` minus `U`, and the Christmas readings
are pre-registered and report-only. The 27 history rows of `U27` contain the 14 history rows of
`M*`, so `U27` minus `M*` is a container against its nested arm.

**Each measure fills its rows in a fixed order, and no block repeats inside a sample.** `U` takes the
5 weather-nearest eligible blocks of ages 2 to 379, then the 5 youngest eligible blocks not yet
chosen, then the 4 youngest same-weekday blocks not yet chosen. The weather distance, the
eligibility rule, the tie rule, the age buckets, and the full-recipe training are those of
[experiment 718](../00718-msa-similar-day-retrieval-part-1-weather-distance/result.md). The
placebo `U-A` keeps the 5 recency rows and draws each of the other 9 rows uniformly from its age
class and from outside the rows of `U`, where ages 2 to 15 form one class and the buckets of the
age table form the others.

## Leakage and the weather look-ahead

**The rules of experiment 718 hold, with the pool widened to 379 days.** Selection reads only the
target row's ENS weather, the calendar, ages, whether a block holds power at least 27 h old, and the
pool blocks' ERA5 and CAMS weather, which is a lab shortcut for blocks of ages 2 to 5. A
prediction-only read-out, the selection swap, rebuilds the row tables with the stored ENS table for
blocks of ages 2 to 5 and ERA5 from age 6, and reports the share of rows that change and B1 under the
swap. The tokens still read ERA5, so the swap measures the selection's dependence on the shortcut
and does not make a live score.

## Decision rule and verdict

**The decision rule is that of
[experiment 718](../00718-msa-similar-day-retrieval-part-1-weather-distance/result.md), with B1 and
B2.** Each part has four counted readings, each interval is read at 98.75%, the smallest gain worth
porting is 0.02 points of capacity, and the seed count is 5 for `U` and `U-A` if experiment 713's
phases 8 and 9 run and 2 otherwise. The reachable thresholds are the same: a null part ends
`inconclusive` or `small gain`, and "no detectable difference" is out of reach at the eight.

**Screen, audit, and stop rules.** The screen is experiment 713's four folds with seeds 0 and 1,
labelled interim. If B2 is positive in both groups on the screen, audit trigger 1 fires before the
other four folds run. The part stops, and is written up from the screen as `not supported`, if B1
on the screen is above +0.15 in both groups, or above +0.10 in both and positive in at least 3 of 4
folds in both. A stop of part 1 does not stop part 2. A verdict of `not supported` or `refuted` in
both parts closes similar-day retrieval of history rows in the network on the 16-Primary panel,
unless `U27` minus `U27-A` is below zero at 95%, which the page says it could not read if `U27-A`
was cut. The exploration tuned on the four screen folds, so the other four folds (2025-12, 2026-02,
2026-04, 2026-05) give the only reading clean of that selection, printed at 95% and report-only.

## Christmas

**No MSA network runs in the live service, and the decision to port any arm lies outside this
experiment.** Arm `UX` replaces the 5 weather rows of `U` by the 5 blocks of earlier winters
nearest in calendar distance, for targets from 20 December to 5 January. In the exploration, the
plain mean of calendar-nearest blocks scored 7.72% against 11.77% for the last 7 days on 24 to 26
December (6 winters). `UX` is fitted for folds 2025-12 and 2026-01 with seeds 0 and 1, and its
readings (`UX` minus `U`, `UX` minus `M*`, `U` minus `M*`) are report-only and one winter. The
pointer "calendar rows help at Christmas" needs `UX` minus `U` on the blocks of 24 to 26 December at
most -0.30 in both groups, with the interval below zero and the same sign for both seeds, and
larger in size than the refit noise that `UX` minus `U` shows outside the window. The pointer is
never a live decision.

## Known-answer checks and worlds

**Part 2 reuses the wiring checks KR1 to KR9 and the worlds of experiment 718.** KS2 needs `U-A`
minus `U` in world WS to be at least half the gap that a NumPy reference computes for the union's
rows. The gating fits and the checks run once, under experiment 718's code, and this page records
the result.

## Compute

**The plan for part 2 is about 9.5 GPU-process-hours with seeds 0 and 1, within the shared ceiling of
23.5 for both parts.** The `U27` fits take about 1.9 times the time of a 15-row fit and an estimated
5.2 GB, and phase 0 of experiment 718 measures both. If the peak exceeds 5.8 GB, `U27` and `U27-A` run
with two half-batches of 128 per optimiser step, with the loss of each half-batch normalised by the
weight sum of the full 256 samples. Order of cuts when the plan is tight: `U27-A`, then `U27`, then
seed 1 of `UX`. The fits of `U` and `U-A` are never cut.

## Disclosures

- **Experiment 713's known-answer gate failed KA3 and KA12 and was waived by the maintainer on
  2026-11-23.**
- **The union was chosen from 64 settings reported on the evaluation folds, with means of rows and
  not with the network.** All four of the exploration's folds are this experiment's screen folds.
  The 5, 5, and 4 split of `U` and the 14, 7, and 6 split of `U27` were not among the 64 settings.
- **The pool blocks' weather is ERA5 and CAMS, a lab shortcut,** and every score of `U`, `U27`, and
  `UX` is a ceiling for a live forecast. The follow-up F8 removes the shortcut from the tokens, and
  a combined arm needs F8's design to include the mixed rule.
- **A null result may come from slow learning, and the Candidate's frozen encoders may not separate
  two days' weather well enough for attention to match the days.**
- **The Christmas reading is one winter,** report-only.

## Method

The experiment is planned and has no fitted result yet, so these bullets describe the design.

- **Architecture and training regime:** those of experiment 718, with rows chosen by the union and
  the 28-row layout for `U27`.
- **Measurement:** CRPS in points of capacity on experiment 713's row set, month-block bootstrap
  intervals, no valid time on or after 2026-07-01 scored.
- **Standing practices followed:** monthly refit; cleaned data; fitted states saved outside git;
  stored forecasts reused as controls; a screen on four folds first; forecast example plots.

## Reviews

An Opus agent reviewed the design before this page was written, and the design was revised after it.
Still to come: an Opus code review of each new script before any fit, a second review of the fixed
code, the known-answer gate, a result review, an audit if a trigger fires, a numbers review, and a
prose review.

## Files

- `data/ml-experiments/briefs/msa_architecture/f9_design.md`, `f9_design_review.md`, and
  `f9_design_triage.md` (read, private): the design that this page condenses.
- `ml-experiments/00719-msa-similar-day-retrieval-part-2-union-of-measures/`: this part's scorer,
  figures, and report, which import the shared code of experiment 718. Written and unit-checked on
  the CPU, with no fit run yet.
- Read from experiment 713 ([pull request 402](https://github.com/openclimatefix/nged-ml-research/pull/402)):
  the stored states and forecasts of `M*` and `P`, and the scorer.
- Written by this experiment (to be created): fitted states under
  `data/ml-experiments/00719-msa-similar-day-retrieval-part-2-union-of-measures/work/` and
  predictions under `predictions/`, never committed.
