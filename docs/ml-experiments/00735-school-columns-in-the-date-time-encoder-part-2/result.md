---
id: "00735"
slug: school-columns-in-the-date-time-encoder-part-2
title: >-
  Does giving the date-time encoder of the step-5 demand neural network the school-holiday and
  coast-closeness columns, in all three training stages, lower the neural network's error around
  school holidays?
theme: demand-inputs
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  Anything else. (1) The experiment stores about 96 fitted neural networks (4 arms, 8 monthly
  refits, 3 seeds), each with per-row forecasts and a re-pre-trained encoder, all private. (2) More
  than one design defensibly tests the hypothesis (columns in the encoder, in the mixer, or in
  both), so the plan fixes one design and names the others as not tested. (3) The experiment
  patches the pre-training code of experiments 574 and 677 and the runner of part 1, whose callers
  cannot be named without searching; it touches no serving path or degradation rule.
hypothesis: >-
  Part 1 gave the step-5 neural network the school-holiday columns at the mixer only, so the
  network could learn them only from the ENS rows after 2024-04-01, about 5 independent school
  holidays in the held-in months. The encoder pre-training reaches back to 2019-12-01 (6 to 7
  years of school holidays, depending on the refit month). Giving the date-time encoder the
  school-holiday and coast-closeness columns in stage 1 (ERA5 and CAMS pre-training), and keeping
  them as encoder inputs in stages 2 and 3, lowers the neural network's CRPS in the weeks around
  school holidays at both metered groups. A school calendar moved 21 days later, and a coast
  closeness permuted between Primaries, give no such gain.
prediction: >-
  Written before any fit. CRPS, arm minus reference, in points of capacity (a point is 1% of a
  Primary's capacity), days 1 to 10, on the rows of the school windows (7 days before, the holiday
  block, and 7 days after) of the 8 held-in refit months. P1: the real-columns arm (SKE) minus the
  re-pre-trained control (R1) is -0.03 to -0.10 at the six signed-MW Primaries and at the eight
  MVA-only Primaries. P2: SKE minus its shifted-calendar twin (SKE-S21) is -0.02 to -0.08 at both
  groups. P3: SKE minus its permuted-coast twin (SKE-CP) is 0.00 to -0.05 at the 4 coastal
  Primaries (D2, D4, D9, and D19), with zero the likeliest value. P4: the change on all rows of
  the held-in months is within -0.01 to +0.01. P5: R1 minus R0 (the stored control) is within
  -0.05 to +0.05, the noise of re-pre-training. The likeliest miss is P1, because part 1's school
  columns scored +0.05 and +0.08 worse, and the extra pre-training years may not carry the
  Lincolnshire signal into the Primaries' demand. The likeliest verdict is `inconclusive`, because
  part 1's intervals on the same 8 months were 0.2 to 0.3 points wide, and this experiment scores
  the same 8 months with about 5 independent school holidays.
decision_rule: >-
  A contrast is a paired difference of CRPS (11-level quantile score, points of capacity) at days 1
  to 10 on the school-window rows of the 8 held-in refit months (2025-11 to 2026-06, valid times
  before 2026-07-01), averaged over seeds 0 to 2, with part 1's two-level bootstrap (20,000 draws:
  refit months, then seeds inside a month, then 7-day blocks of valid dates), read at 99% (95%
  Bonferroni over 5 readings: C1 and C2 at the six and at the eight, C3 at the 4 coastal Primaries).
  C1 is SKE minus R1, C2 is SKE minus SKE-S21, and C3 is SKE minus SKE-CP; the four new arms use
  identical stage 1 seeds and initialisations, so the three contrasts pair. A gain needs the
  interval wholly below zero, a point estimate of at most -0.02, a negative seed-mean difference in
  at least 6 of the 8 refit months, and a negative difference in at least 18 of the 24 fits. A
  reading wholly above zero is harm. Placebo readings, at 95% (SKE-S21 minus R1 and SKE-CP minus
  R1): a placebo fails when its interval lies wholly below zero. School claim at a group: C1 and C2
  are gains there, SKE-S21 minus R1 has not failed there, and the no-harm contrast (SKE minus R1 on
  all rows of the 8 held-in months) has a one-sided 95% upper bound of at most +0.02. Coast claim:
  C3 is a gain at the 4 coastal Primaries, SKE-CP minus R1 has not failed, the no-harm contrast has
  an upper bound of at most +0.02 in each group, and C3 at the 3 inland Primaries (D11, D10, and
  D22) does not move in the same direction beyond its own 95% interval. Verdict, first rule that
  applies: (1) `refuted` when C1 and C2 are both harm at the same group; (2) `supported` when the
  school claim holds at both groups and the coast claim holds; (3) `partly supported` when the
  school claim holds at one or both groups or the coast claim holds; (4) `small gain` when C1 or C3
  has an interval wholly below zero with a point estimate between 0 and -0.02; (5) `not supported`
  when the intervals of C1, C2 at both groups and of C3 lie inside -0.05 to +0.05; (6)
  `inconclusive` otherwise. SKE minus R0 (the stored control) is a descriptive reading that links
  the result to part 1. This experiment scores the same 8 months as part 1 and cannot return
  `supported` unless the effect is larger than the predicted range.
benchmark: >-
  R1 (the control re-pre-trained through this experiment's code with the unchanged 21 encoder
  columns), and R0, experiment 677's stored `k1_5w_sp1` forecasts (arm 9a of the step-5 chain), as
  a descriptive reference behind a replay gate; part 1's school-column arm S as a descriptive
  reference; the manual heuristic and climatology from experiment 677's joint pass.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS (headline) and normalised MAE of the median, in points of capacity, with the signed error of
  the median, q01, q05, q95, and q99 coverage and pinball loss, by holiday window, group, and
  Primary, at days 1 to 10 (headline) and in every demand-v2 lead band.
production_status: not-for-production
---

> **Plan, written before any fit. Status `planned`.** Nothing below has been run.

**This is part 2 of [experiment 735 part
1](../00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1/result.md).** Part 1 found
that school-holiday columns at the step-5 neural network's mixer did not lower its error, and
listed "school columns in the date-time encoder" under "Not done". Part 2 asks the same question
with the columns placed where the neural network has years of data to learn them: the date-time
encoder, which is pre-trained on ERA5 and CAMS rows from 2019-12-01 and then frozen.

## What the maintainer required

**The school-holiday columns and the coast-closeness columns must reach every training stage, not
only the last** (maintainer, 2026-11-28, via the manager). The plan therefore gives both column
groups to the date-time encoder in stage 1 (pre-training), stage 2 (encoders frozen, fresh mixer),
and stage 3 (fine-tuning). Part 1 added them in stages 2 and 3 only.

## Design

**The date-time encoder takes 8 more input columns, in all three stages.** The date-time encoder
reads columns 14 to 34 of the 37-column row (21 solar-geometry, calendar, and holiday columns,
`e00559_model.DATETIME_COLS`), plus the Primary's 8-number embedding. The new arms append 8 columns
to the encoder's input, which widens its first layer from 29 to 37 inputs:

- **School columns S1 to S3** (part 1's definition): `school_off` (on every date of a holiday
  block, weekends included), `school_first_weekend` (the first Saturday and Sunday of a block),
  and `school_half_term` (inside a block of at most 10 days). Every column uses the local date
  (Europe/London) of the stamp minus 15 minutes.
- **Coast columns K1 to K5** (part 1's definition): K1 `c = exp(-d / 5 km)`, where `d` is the
  distance in kilometres from the Primary to the coast, then `c` times `school_off`,
  `school_first_weekend`, `bank_holiday`, and a Saturday-or-Sunday flag. `c` is static per Primary
  and exists for all 16 Primaries. K4 and K5 recompute the bank-holiday and weekend flags from the
  stamp minus 15 minutes, so they differ from the encoder's existing `bank_holiday` and
  `day_of_week_local` columns, which use the stamp itself (part 1 documents the difference).
- **The 8 columns are not standardised.** They are 0 or 1, or `c` (0.003 to 0.87) times 0 or 1, so
  they stay on one scale in every stage. Stage 1 trains the encoder from scratch with paired
  initialisation: R1 draws the 29 original first-layer input columns from the seed, the new arms
  copy those weights, and draw the 8 new columns' weights from a generator seeded 735 (uniform
  within plus or minus 1 over the square root of 37), so the arms of one seed differ only in the
  new columns.
- **The 8 columns enter at every place the encoder is called.** The step-5 neural network passes
  two inputs through the date-time encoder: the row itself, and the day lag and week lag, whose
  weather and date-time columns are read at each lag's own valid time (`e00574_model._expected`,
  called by `e00591_model.lag_embedding`). The 8 columns are deterministic functions of the valid
  time and the Primary, so the lag path takes them at the lag's valid time. A zero fill would give
  the frozen encoder a different input distribution at lag time than at row time, so the plan
  rules it out.
- **Stages 2 and 3 pass the 8 columns through the frozen encoder.** The encoder weights stay frozen,
  so the columns reach the mixer through the encoder's output. Part 1's mixer columns are not
  repeated, because a column both in the frozen encoder and in a zero-initialised mixer would mix
  two mechanisms.
- **The weather-by-coast columns W1 to W4 of part 1's arm SKW are not tested in part 2.** They read
  the ensemble forecast's daily maximum temperature and mean shortwave, which have no counterpart
  in the ERA5 and CAMS rows of stage 1. This is a departure from part 1's SKW arm.

**The school calendar covers the whole stage 1 record.** Stage 1 rows have valid times from
2019-12-01. The calendar has three parts:

- Term dates for 2019-20 and 2020-21, parsed from Branston Community Academy's published term-date
  PDFs (a Lincolnshire academy, so a proxy for the county calendar), in
  `lincolnshire_term_dates_2019_21.csv`. The academy's summer term ends 16 July in both years,
  while the county's 2021-22 term ends 22 July, so the July holiday block may start a few days
  early. The fetch lineage is in the experiment's data folder.
- Term dates for 2021-22 to 2025-26, from experiment 11's `lincolnshire_term_dates.csv`,
  unchanged.
- Terms 1 and 2 of 2026-27, from part 1's file, unchanged.

**The Covid closures of 2020 and early 2021 are a caveat, not a column.** The columns say what the
published term dates say, not whether pupils attended. Stage 1 may therefore learn `school_off`
against lockdown demand in those months. The page states this limit and does not model it.

## The arms

| Arm | In words | Stages 1, 2, and 3 read |
|---|---|---|
| R0 | the stored control: experiment 677's `k1_5w_sp1` forecasts, behind a replay gate | 21 encoder columns |
| R1 | the control re-pre-trained through this experiment's code, with the 3 seeds and the initialisation shared with the three arms below | 21 encoder columns |
| SKE | school and coast columns in the date-time encoder | S1 to S3, K1 to K5 |
| SKE-S21 | placebo for the school signal: the school calendar read 21 days earlier in S1 to S3 and in the school products K2 and K3, in all three stages | shifted S1 to S3, K2, K3; real K1, K4, K5 |
| SKE-CP | placebo for the coast signal: the coast closeness permuted between Primaries (part 1's permutation, seed 735) in K1 to K5, in all three stages | real S1 to S3; permuted K1 to K5 |

**Every new arm re-runs stage 1 for each refit month and seed.** The frozen encoder is a different
neural network once its first layer has 37 inputs, so the control's stored stage 1 weights cannot
be reused. R1 repeats stage 1 with the 29-input encoder, which makes C1 a paired contrast with the
same code, seeds, and initialisation as SKE. R0 stays as a descriptive link to part 1: R1 minus R0
(P5) shows how much re-pre-training alone moves the score.

**The placebos are defined so that each contrast isolates one signal.** SKE-S21 shifts the school
calendar everywhere it enters: S1 to S3, and the products K2 and K3. C2 (SKE minus SKE-S21)
therefore removes the school signal including its coast interaction, and leaves the coast
closeness itself (K1) and its bank-holiday and weekend products (K4, K5) real. SKE-CP permutes `c`
in all of K1 to K5, so K2 and K3 take the permuted `c` times the real school columns. C3 (SKE minus
SKE-CP) removes the coast signal and leaves the school calendar real.

## Planned contrasts and checks

- **C1:** SKE minus R1 at the six signed-MW Primaries and at the eight MVA-only Primaries.
- **C2:** SKE minus SKE-S21 at both groups (the school signal).
- **C3:** SKE minus SKE-CP at the 4 coastal Primaries (the coast signal), with the same contrast
  at the 3 inland Primaries as a placebo.
- **Descriptive:** SKE minus R0, R1 minus R0, part 1's arm S, and the manual heuristic.
- **Same rows as part 1:** the 8 held-in refit months 2025-11 to 2026-06 with part 1's windows,
  scored windows, cleaned actuals, and the D15 and D18 cut before 2025-03-01, so that the two parts
  compare. The four months 2026-07 to 2026-10 decide nothing in part 1 and are not run in part 2,
  which saves about a third of the compute and pays for R1.
- **Departures from part 1's rule:** 5 readings instead of 6, so the intervals are read at 99%
  instead of 99.17%, because the SKW, SE, and SEP readings (C4, C5) are not run. `refuted` needs C1
  and C2 together, because C1 includes the noise of one re-pre-training draw per seed.
- **The "every stage" gate (`e00735b_check_stages.py`):** for each arm and each stage (1, 2, 3), the
  check builds a batch of the rows the stage trains on, and a batch of its day-lag and week-lag
  rows, and asserts that the 8 new columns are in the encoder's input and are non-constant (more
  than one distinct value, and for the school columns both 0 and 1 present). The check also
  asserts that the shifted arm's school columns differ from the real arm's on at least 5% of rows,
  that the permuted arm's coast columns differ from the real arm's at 14 of the 16 Primaries, and
  that R0 and R1 have none of the columns. The check prints a table of arm, stage, input path (row,
  day lag, week lag), column, distinct values, and share of ones, and the table goes in the
  write-up.
- **Look-ahead:** the calendar is fixed in advance and the coast closeness is static, so no new
  column reads a weather product or any value after the forecast origin. The published 2026-27
  term dates are known in advance of every refit that reads them. A test checks that the columns
  of a row depend only on its valid time and its Primary.
- **Replay gate:** part 1's R0 replay, run through part 1's pipeline in the gate month (2026-03,
  seeds 0 to 2), must reproduce experiment 677's stored forecasts with a maximum quantile
  difference of at most 5e-4 of capacity (part 1's rule) before any new arm runs. The gate does
  not exercise part 2's runner, so the first check of the part 2 code on real files is the
  smoke test, and R1 minus R0 (P5) is the descriptive check afterwards.
- **Calendar check (data-validation):** the parsed calendar has ordered terms, no overlap, and
  holiday gaps of 16 to 19 days at Christmas and Easter and 43 to 51 days in summer. The check ran
  on 2026-11-28 and its output goes in the write-up.

## GPU-process-hours

| Item | Arm-months | Hours |
|---|---|---|
| Smoke test (4 arms, one epoch each, validation rows) | n/a | 0.35 |
| Replay gate | 1 | 0.10 |
| R1, SKE, SKE-S21, SKE-CP, each in 8 refits | 32 | 5.07 |
| **Expected total** | | **5.52** |

**Each arm-month takes about 9.5 minutes of GPU process time.** Part 1's stages 2 and 3 and the
prediction pass took about 4.5 minutes (part 1's run took 5.83 hours for 73 arm-months). Stage 1
adds about 3.5 minutes for 3 seeds (46 to 71 s per seed in experiment 677's logs, not re-measured)
and about 1.5 minutes of row building for the wider input. The smoke test runs one epoch of each
stage, so it does not measure stage 1; the first full arm-month measures it, and the projection is
updated then.

**The stop line is 6.5 GPU-process-hours, agreed with the manager.** If the measured time takes the
projection over 6.5 hours, SKE-CP is dropped from the months from 2026-06 backwards in that order,
and the page says so. The phase runner stops at 6.5 hours and asks.

## Compute and memory

- One GPU process at a time, on CPU cores 16 to 23, with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` set to 1, every Python command inside `systemd-run --user --scope -p
  MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue`.
- Seeds, months, and arms write their own files atomically, so a crashed job reruns alone.
- Scoring streams one month at a time. Expected peak memory is under 8 GB (part 1's scoring peak).

## Reviews, in order

- Sonnet plan review (done 2026-11-28; the findings are triaged in the pull request).
- Sonnet writes the code, then one Opus code review limited to the diff of the changed encoder,
  calendar, and gate code, then the smoke test, then the code hash is frozen. A re-review of fixed
  code is a Sonnet review, because the Opus allowance is one review.
- After the scored run: a Sonnet result review that re-derives the headline contrasts, a Sonnet
  numbers review, and a Sonnet prose review.
- The standing practices follow part 1: monthly refits, cleaned data, the D15 and D18 cut, fitted
  models saved, tails and avoidable-curtailment cost reported, and examples of day-ahead and 14-day
  forecasts.

## Advice from earlier experiments

The plan follows part 1's advice to score against a shifted twin trained in the same refits (C2 and
C3), and to read the first weekend of a school holiday first (reported as a window). It departs from
part 1's mixer placement on purpose, because that placement is what part 1 already tested. Part 1's
advice to design a scored period with more than 5 independent school holidays is not followed: the
same 8 months are scored so that the two parts compare, and the page states that the intervals will
be about as wide as part 1's.

## Not done in this experiment

- The weather-by-coast columns W1 to W4 (stage 1 has no matching weather).
- The columns at the mixer, or at both the encoder and the mixer.
- The months 2026-07 to 2026-10.
- A school calendar that models the Covid closures, or the East Midlands and Yorkshire school
  calendars that visitors to the coast follow.

## Files

- `ml-experiments/00735-school-columns-in-the-date-time-encoder-part-2/lincolnshire_term_dates_2019_21.csv`:
  term dates for 2019-20 and 2020-21 (written by this experiment, read by the calendar code).
- `data/ml-experiments/00735-school-columns-in-the-date-time-encoder-part-2/raw/`: the fetched pages
  and `LINEAGE.md` (private, not committed).
- Part 1's code and data, reused: see [part
  1](../00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1/result.md).
