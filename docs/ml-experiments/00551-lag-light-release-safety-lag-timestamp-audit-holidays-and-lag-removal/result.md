---
id: "00551"
slug: lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal
title: >-
  Is the lag-light demand network safe to release: do its lag timestamps look ahead, does it hold
  up in Christmas week and on bank holidays, and does it survive removed or late power readings?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
size: >-
  medium. Stored: new score tables and one audit file under this experiment's private data folder
  only. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Designs: one defensible design per part, because the briefs fix the audit, the slices, and the
  perturbations. Callers: the scripts import experiments 087, 397, 428, 454, 470, 510, 511, 521,
  and 544 (and read 539's stored weights and predictions). One fresh Sonnet code review before the
  first scoring run, no further review unless it flags many problems.
hypothesis: >-
  The lag-light demand network R-DW (experiment 539) and its dropout variant R-DROP (experiment 544)
  read only power readings that existed at the forecast origin, are no worse than L-1d in Christmas
  week, and degrade gracefully when readings are removed or late.
prediction: >-
  Written before any score is read. P1 (audit): zero rows where a lag's source time is later than
  the origin minus 3.5 h, where the age column differs from the origin-to-source gap, or where the
  missing flag disagrees with a missing value. P2 (calendar): R-DW is no worse than L-1d by more
  than 0.10 CRPS in Christmas week at the eight. P3 (removal): R-DROP with every reading removed is
  within 0.15 CRPS of L-1d at the eight; R-DW without dropout loses more than 0.5 when its readings
  are removed; the part fails if R-DROP loses more than 0.3.
decision_rule: >-
  P1 fails on a single violation, which voids the R-DW and R-DROP headlines and stops the line. P2
  and P3 are read as point values, because a slice of a few days gets no interval claim; any slice
  with fewer than 10 days shows per-year signs only. Intervals, where shown, are month-block
  bootstraps over 5 blocks (near the minimum to maximum range of the folds). The minimum detectable
  effect is L-1d's two-seed spread, and the Bonferroni count is stated as a number in the page.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425), R-DW (experiment 539), R-DROP (experiment 544),
  and the lag-free recipe with trailing temperature windows (experiment 548, arm E), all read from
  their stored forecasts or stored weights.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries; D15
  and D18 report-only; July 2026 unused.
verdict: partly supported
best_arm: >-
  R-DROP. At the eight, 24-120 h, CRPS minus L-1d's in points of capacity: R-DROP with readings
  present -0.085, with every reading removed -0.021 (interval -0.057 to +0.009); R-DW -0.188 with
  readings present and +0.201 with every reading removed (interval -0.047 to +0.379).
headline: >-
  The lag audit found no look-ahead. R-DW is worse than L-1d by 0.171 CRPS in Christmas week at the
  eight, which breaks the 0.10 limit (nine target days, one winter). R-DROP stays within 0.15 of
  L-1d with every reading removed (-0.021 at 24-120 h). R-DW with every reading removed loses 0.201
  to L-1d at 24-120 h at the eight, less than the 0.5 predicted.
lessons: >-
  The lag inputs are built from readings at least 3.5 h old in every row, so the large gain of the
  lag-light network is not look-ahead. The week-ago reading is an ordinary week in Christmas week,
  and R-DW (but not R-DROP) pays for it. Training with 30% of rows with no readings makes the
  network safe to run without readings, at the cost of most of its gain.
take_home_message: >-
  No lag input looks ahead of the forecast origin, so the lag-light networks' gains are genuine.
  R-DW is 0.17 points of capacity worse than the lag-free network in Christmas week at the eight,
  past the 0.10 limit, and it loses 0.2 when its readings are removed. R-DROP does not have either
  weakness but keeps under half of the gain. The next step is to decide whether R-DW with a
  Christmas-week fallback or R-DROP goes into the release candidate.
advice:
  - >-
    Do not release R-DW unchanged: it is worse than L-1d by 0.17 in Christmas week at the eight and by
    0.30 at the six (figure 1).
  - >-
    Release a lag network only with a switch that removes the readings: R-DROP loses nothing against
    L-1d with every reading removed, R-DW loses 0.2 at the eight and 0.36 at the six (figure 2).
  - >-
    Score any lag network on Christmas week and each bank holiday before release; a network that
    wins on all rows can lose on those days (figure 1).
  - >-
    Late readings (1 to 3 extra days) cost R-DW at most 0.04 and R-DROP at most 0.01 at 24-120 h, so
    a late feed needs no special fallback at the eight (figure 2).
production_status: candidate
adopted_in: none
---
 **No lag input looks ahead of the forecast origin. R-DW fails the Christmas-week limit, and R-DROP
passes every pre-registered test but keeps less of the gain.** The question was whether the
lag-light demand networks R-DW (experiment 539) and R-DROP (experiment 544) are safe to put in a
release. All numbers are interim and unaudited, in points of capacity; lower CRPS is better. Nothing
was refitted.

Verdict: partly supported. P1 held, P2 failed, and P3 held for R-DROP and failed (in the safe
direction) for R-DW.

## Figures

![CRPS minus L-1d's in calendar slices](figure_1_calendar_slices.svg)

![CRPS minus L-1d's with readings removed or late](figure_2_lag_removal_and_late_reading.svg)

## Results

- **P1, lag-timestamp audit: zero violations.** The audit recomputed each lag's source half-hour
  from its definition, for the day-ago, week-ago, and newest reading, in 34.8 million rows
  (104,481,990 lag checks): the raw summary matrices of all 16 Primaries, and each of the five
  folds' training and test tables. It found 0 sources later than the origin minus 3.5 h, 0 age
  columns that differ from the target minus the source time, 0 missing flags that disagree with the
  cleaned power, and 0 values that differ from the cleaned power (tolerance 1e-5 on raw values, 1e-4
  on standardised tables inverted with the fold's own standardiser; 981,447 rows clipped at 5
  standard deviations are skipped in the built tables and are covered by the raw pass). One row of
  every lag and lead band is printed in `audit_lag_timestamps.json`; for example, a 2026-01-01
  09:00Z origin with target 2026-01-03 05:00Z has its day-ago lag from 2026-01-01 05:00Z (origin
  minus 4.0 h) and its week-ago lag from 2025-12-27 05:00Z. The same standing weather rule holds:
  the 2026-01 test example uses weather run 2026-01-01 00:00Z for origin 2026-01-01 09:00Z, and the
  check that `nwp_init_time` is at or before the origin passed for every test origin.
- **One data-cleaning gap, not look-ahead.** The raw lag matrices still hold D15 and D18 readings
  from before 2025-03-01, which the cleaned power drops (261,287 lag checks). Those two Primaries
  are report-only, so no headline number changes.
- **P2, Christmas week: failed at the eight.** R-DW minus L-1d is +0.171 over 9 target days (24
  December 2025 to 1 January 2026; 5 of 9 days worse; +0.170 and +0.284 in the two folds that hold
  them), above the 0.10 limit and below the 0.2 stop rule. A-pool is +0.160, R-DROP -0.097, and the
  lag-free recipe of experiment 548 -0.065. At the six R-DW is +0.299. With one winter and a few
  days no interval is shown, and the minimum detectable effect (0.12) is the seed spread of all
  rows, so it understates the noise of a slice.
- **Bank holidays.** Christmas Day, Boxing Day, and New Year's Day windows (3 days each) are worse
  for R-DW by +0.36, +0.37, and +0.19 at the eight. Good Friday and Easter Monday windows reach only
  long leads (3 to 10 days), where R-DW is better (-0.37 and -0.85), so they say little about the
  short leads.
- **Cold nights and rows with no labelled event.** R-DW is better than L-1d at the eight (-0.209 and
  -0.117), and worse at the six on cold nights (+0.178). R-DROP is better or equal in both.
- **P3, lag removal (eight, 24-120 h, minimum detectable effect 0.116, Bonferroni count 28).**
  R-DROP with every reading removed minus L-1d is -0.021 (interval -0.057 to +0.009, 5 month blocks,
  near min to max; 1 of 5 folds worse), inside the 0.15 limit in every band (largest upper end
  +0.035). It reproduces experiment 544's stored R-DROP-absent forecasts exactly. R-DW with every
  reading removed is +0.201 (-0.047 to +0.379; 4 of 5 folds worse), +0.390 against its own
  readings-present forecast, so the predicted loss over 0.5 did not occur (largest band 6-12 h,
  +0.342). At the six R-DW is +0.355 and R-DROP +0.029.
- **Partial removal and late reading.** Removing only the day-ago or only the week-ago reading
  leaves R-DW at -0.052 and -0.058 against L-1d. Readings 1, 2, and 3 days late leave R-DW at
  -0.173, -0.163, and -0.152 and R-DROP at -0.081, -0.079, and -0.075, so ageing costs R-DW at most
  0.04 and R-DROP at most 0.01 against their readings-present forecasts. Removing 25% of rows at
  random leaves R-DW at -0.090 and R-DROP at -0.069.
- **Release checklist.** Passed: lag-timestamp and weather audits; removal with R-DROP within 0.15;
  late reading within 0.1. Failed: R-DW in Christmas week (0.10 limit).

## Caveats

- NGED's logs may miss load-shift events, so rows with no labelled event may hold unlabelled events
  and the slices undercount them.
- Christmas week is one winter (9 target days), so a per-year sign cannot be read. D15 and D18 are
  report-only and July 2026 is unused. The six-Primary holiday windows are signed-MW rows with no
  labelled event.
- Part 3 scores the stored networks with their readings altered at inference only. A network trained
  to expect late readings may do better.

## What the next experiment should answer

Does R-DW with the week-ago reading replaced by a holiday-aware reading, or with its readings
switched off on public holidays, remove the Christmas-week loss and keep the gain?

## Code

`ml-experiments/00551-*/`: `e00551_audit.py` (Part 1), `e00551_calendar.py` (Part 2),
`e00551_removal.py`, `e00551_rows.py`, and `e00551_score.py` (Part 3), `e00551_figures.py`. They
import the folders of experiments 087, 397, 428, 454, 470, 510, 511, 521, and 544, and read
experiments 539, 544, and 548's stored weights and scores. The Sonnet code review found no
look-ahead; its tolerance, reproduction, and gating findings were applied, and its caveats on the
Easter windows and the slice minimum detectable effect are in the text.
