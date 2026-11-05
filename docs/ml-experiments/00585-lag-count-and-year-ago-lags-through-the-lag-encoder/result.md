---
id: "00585"
slug: lag-count-and-year-ago-lags-through-the-lag-encoder
title: >-
  Does the modular demand network forecast better with a different number of power lags, or with a
  power reading from one year earlier, when both go through the lag encoder?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted weights, forecasts, and scores in the private data folder only. More than
  one defensible design: how the extra lags reach the lag encoder (flat concatenation, a median, or
  attention), and how the year-ago lags are built. Callers: the code copies experiments 572 and 581
  and imports 397, 422, 428, 510, 511, 558, 559, 560, and 565 (more than two callers, found by
  search). Serving path and degradation rule: not touched (this repository holds none). Cost:
  ceiling 3 GPU-h; projected about 1.5.
hypothesis: >-
  The lag encoder of the modular network (experiment 572's v3+D5, the control N2) reads one
  day-old and one week-old reading of a Primary's power. A reading from 364 days earlier (the same
  weekday and half-hour) carries information that the two recent lags cannot at long leads, where
  the week lag is 2 to 3 weeks old, and on holidays that repeat each year. The manual heuristic
  already uses last year's values.
prediction: >-
  Written before any fit or score. Y1 (control plus one year-ago lag) lowers CRPS at 120-240 h by at
  least 0.05 points of effective capacity at one group (fails if within 0.03 of the control; a gain
  between 0.03 and 0.05 is a small gain and is not kept). Y1 does not help in Christmas and New Year,
  because 364 days before 25 December is Boxing Day; Y1c lowers CRPS there by at least 0.10 at one
  group (fails if within the MDE). Bank holidays that fall on a Monday align under Y1. Y1 does not improve cold winter
  nights (within the minimum detectable effect, MDE). Y3 beats Y1 by under 0.03. A "within 0.03" claim is withdrawn wherever the printed MDE exceeds 0.03. N1 (week lag only)
  is worse than the control by at least 0.10 at 0-24 h (fails if within the MDE). N3 (an added
  second weekly lag) is within 0.03 of the control. N5 (experiment 581's X3-med, stored) is within 0.03 of
  the control, as experiment 581 found. Masking the year-ago lag where it lies in a verified
  load-shift event (Y1-mask) changes CRPS by under 0.03 on all rows.
decision_rule: >-
  The maintainer's keep rule: an arm is kept if it improves all rows at one group by at least that
  contrast's MDE with a Bonferroni interval excluding zero, or improves one pre-registered slice
  (cold winter nights, Christmas and New Year, Easter, bank holidays) by at least that slice's MDE.
  A leak or slice cost is a flag, not a gate, unless it actively hurts: worse than the control by
  more than the larger of 0.03 and the slice MDE with the interval excluding zero, or the harmful
  sign in 4 of 5 folds. Bonferroni count 36 (N1, N3, N5, Y1, Y1c, and Y3 against the control, times 2
  groups, times the 3 lead bands); 48 if N5Y1 runs (adding N5Y1 against N5 and against Y1). The
  slice route is a second keep route that is not Bonferroni-corrected across slices (4 slices, 2
  groups, per arm), and the write-up labels it that way. Y3 against Y1 and Y1 against Y1-mask are
  descriptive. The
  MDE (two seeds) is printed beside every contrast. Y1-mask is descriptive. The threshold is the
  MDE because each kept arm adds inputs and a data dependency (a year of power history).
benchmark: >-
  The control N2 is experiment 572's v3+D5, with its stored forecasts and weights (seeds 0 and 1,
  no refit). N5 is experiment 581's stored X3-med forecasts. References: experiment 565's D5
  (lag-free), the climatology, and the manual heuristic.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at 0-24 h, 24-120 h, and 120-240 h at the eight MVA Primaries
  and the six MW Primaries (D15 and D18 report-only), with the tail pinball loss and coverage, and
  the avoidable-curtailment cost.
---
**This page is a plan; no fit has run.** It is written before any score, so the commit order proves
the hypothesis came first.

## Plan

- **Base recipe.** Experiment 572's v3+D5 (the control N2), refit on days 0 and 14 of each of the
  five fold months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seeds 0 and 1, monthly cadence,
  cleaned data, D15 and D18 cut before 2025-03-01. The brief named the P5 recipe of experiment 574;
  the plan departs from it because experiment 581 already stored X3-med on the v3+D5 recipe, which
  makes N5 free, and porting the lag block into experiment 574's much larger fit would add a second
  change. Whether a kept arm survives on P5 is the next experiment.
- **Arms.** N1 (week lag only: the day-lag columns hold the absent code), N2 (control), N3 (day,
  week, and a second weekly lag one week further back; a fixed 14-day lag would coincide with the
  week lag at many leads), N5 (experiment 581's X3-med, stored: a median and spread of 5 weekly
  lags, a different layout from the brief's shared encoder), Y1 (control plus a lag at 364 days,
  weekday-aligned), Y1c (control plus a lag at 365 days, the same calendar date, so it aligns
  Christmas and New Year but not the weekday), Y3 (control plus lags at 357, 364, and 371 days, each
  in its own input slot, no aggregation). N1 also zeroes the day lag's expected-weather embeddings.
  N5Y1 runs only if N5 or Y1 clears the keep rule.
- **Lag construction.** Every extra lag has a value, a missing flag, and an age, and goes through
  the same lag encoder together with the expected-weather embeddings at the lag's own time, like the
  day and week lags. A year-ago lag is the cleaned power at `target - 364 days` (a lag of 357 or 371
  days is a whole number of weeks, so the weekday matches; the UTC half-hour grid shifts the local
  hour by one during the 1 to 2 weeks around each clock change). Every lag is at least 24 h older
  than the forecast origin (asserted). Power comes from the cleaned power stack that experiments
  572 and 581 use, which starts in 2019-09, so the earliest usable fold month is 2025-10 (a year
  earlier is inside the stack); D15 and D18 lose the year-ago lag before 2026-03 because of the
  2025-03-01 cut. The expected-weather table for fine-tuning (ENS summaries) starts on 2024-04-01,
  so year-ago lag times before 2024-04-01 get zero weather embeddings in fine-tuning; the share of
  such rows is printed per fold. Because the lag encoder is frozen in fine-tuning, the year-ago arms
  see zero weather embeddings on those fine-tuning rows but real ones at scoring, so their result is
  partly confounded; the write-up says so and reports the share per fold.
- **Block dropout.** Each extra lag is masked on whole origin days by its own draw (experiment 558's
  all-lags probability, a separate random stream per lag), and also whenever experiment 558's
  all-lags mask fires.
- **Y1-mask (inference only, no refit).** Experiment 580's event set covers only the period it verified, so events outside it stay unlabelled and Y1-mask understates contamination. Y1's saved weights forecast again with the year-ago lag set
  to the absent code wherever it lies inside a verified load-shift event of experiment 580. The
  contrast Y1 against Y1-mask on rows whose year-ago lag lies in an event measures contamination.
- **Injected bank.** The lag-following measures of experiment 581 run on the injected bank for every
  fitted arm (the control and N5 reuse 581's stored results). The bank shifts only recent lags, so it
  shows whether the extra lags dilute following; contamination of the year-ago lag itself is read
  from Y1-mask.
- **Row-key checks.** The code asserts that the stored control, the stored N5, and every new arm
  have identical row keys on the five months and refit days 0 and 14, and that the stored checks
  files show seeds 0 and 1 and the same recipe. Lag-age checks cover every extra lag (at least 24 h
  before the origin, weather run at or before the lag time, an exact 364- or 365-day offset, nothing
  after 2026-06-30).
- **Scoring.** Experiment 572's scorer, patched as experiment 581 did, plus the lead band 120-240 h,
  and extra slices: Christmas and New Year (24 December to 2 January), Easter (Good Friday to Easter
  Monday), bank holidays of England and Wales, and cold winter nights. False-alarm cost on event-free rows, plots of CRPS by lead and by event type for every arm, and
  per-Primary gain tables are included. Tail calibration (q01, q05,
  q95, q99) and the avoidable-curtailment cost against the manual heuristic are reported beside CRPS.
- **Standing practices followed.** Monthly refits; cleaned data; the D15 and D18 cut; fitted weights
  saved in the experiment's data folder; stored control forecasts reused; example forecasts (a
  day-ahead forecast and a full 14-day horizon, with and without the year-ago lag, including
  Christmas) plotted; the unlogged-event caveat stated. Not followed: the few-fold screen, because
  the control exists only on the five scored months.
- **Process.** Sonnet writes the code; a fresh Sonnet agent reviews the plan, then the code before
  the first fit; results, write-up, and prose each get a fresh Sonnet review.
- **Compute.** One GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`), at most 4 GPU
  processes machine-wide, fits start only when `free -g` shows more than 20 GB available. Five fitted
  arms (N1, N3, Y1, Y1c, Y3), each with its own pre-training (the lag encoder is frozen in
  fine-tuning, so every arm pre-trains per month and seed), plus the optional N5Y1. Reusing the
  control and N5 saves two arms. Projected 2 GPU-h, ceiling 3.
- **Earlier advice followed.** Judge lag changes on the injected bank and on days 4 to the end of an
  event ([experiment 581](../00581-robust-multiweek-lag-encoder-x3/result.md)); report absolute
  scores beside contrasts.
