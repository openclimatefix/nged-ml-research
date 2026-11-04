---
id: "00542"
slug: correction-notes-for-pages-using-experiment-404-look-ahead-weather
title: >-
  Which earlier pages used experiment 404's look-ahead weather, and which headlines depend on it?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
date: 2026-11-04
size: >-
  simple. No fit, no GPU, and nothing stored; the experiment reads merged pages and their scripts.
  Designs: one. Callers: none. Production serving path and degradation rule: not touched.
hypothesis: >-
  Every merged page whose gap to K2-S1 used experiment 404's freshest-run weather table has a
  headline that depends on it above 24 h of lead, and the dependants of that table that use
  experiment 425A's weather do not.
prediction: >-
  Written before reading the scripts. Most of the 14 dependants use 404's table, and the headline
  of each depends on it only where it compares a lagless forecast, B-mix, or K2-S1 in absolute terms
  above 24 h of lead.
decision_rule: >-
  Descriptive. A page gets a correction note when its headline compares a forecast that contains
  experiment 404's lagless table with a forecast that does not, or states the size of the gain a lagged
  input adds over that table, above 24 h of lead.
benchmark: none
row_set_id: none
metric: none
code: none
verdict: supported
best_arm: >-
  Not an arm contest. 13 of the 14 dependants use 404's table; 1 (481) uses it for one comparison
  only.
headline: >-
  17 merged pages now carry a dated correction note: 9 whose gap to K2-S1 used the look-ahead
  weather, and 8 of the 14 dependants of 404's table.
take_home_message: >-
  Almost every earlier lagged-power page rests on experiment 404's lagless forecast, which used a
  look-ahead weather feed. Above 24 h of lead the lagless forecast, B-mix, and K2-S1 on those pages
  look better than a forecast made at the origin can be. Comparisons between variants that share the
  same lagless table are unaffected, and so is everything at 0 to 24 h. Experiments 531 and 532 and
  the rebuilt K2-S1 will restate the numbers.
lessons: >-
  A table of forecasts that is shared by many experiments needs an origin-time check of its weather
  inputs before the first experiment reads it.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Read the origin-time weather check of experiments 531 and 532 before quoting any score above 24
    h of lead from K2-S1, B-mix, or the lagless table of experiment 404 (table below).
  - >-
    Build lag-free or lagged networks on experiment 425A's weather tables, which experiment 532
    found taken from the run published at the forecast origin, until the rebuilt K2-S1 exists
    (table below).
---

**Almost every earlier lagged-power page rests on experiment 404's lagless forecast, whose weather
came from the freshest run at each target's valid time (a look-ahead).** Above 24 h of lead the
lagless forecast, B-mix, and K2-S1 on those pages look better than a forecast made at the origin
can be. Experiments 531 and 532 confirmed the look-ahead and found experiment 425A's weather clean.
This experiment lists which pages are affected, adds a dated correction note to the 17 whose
headline depends on the table, and leaves the other 6 unchanged. It fitted and scored nothing.

## Pages whose gap to K2-S1 used the look-ahead weather

Each of these has a correction note at the top of its `result.md`.

| Experiment | Note |
|---|---|
| 468, 470, 510, 511, 521, 523, 524, 525, 519 | Gap to K2-S1 above 24 h voided; 0 to 24 h unchanged; comparisons without K2-S1 unaffected. |
| 527, 529, 530 | Not on `main`: the pages are not merged, so there is no page to correct. The same note belongs on each when its page merges. |

## Dependants of experiment 404's table and of experiment 425A

"404 table" means the cross-fitted lagless table built by `e00404_lagless.py` (or a copy in
`e00445_lagless.py`), which takes each target's weather from the freshest run published before the
target's valid time. "425A" means the rows of experiment 397's summary tables, which experiment 532
checked row by row against the run published at the forecast origin.

| Experiment | Weather source | Headline depends on it? | Note |
|---|---|---|---|
| 445 | 404 table (own copy, same rule) | No: mixture rules compared with a 50/50 mixture of the same table | None |
| 455 | 404 table (rescore of 445's arms) | No: same contrast as 445 | None |
| 460 | 404 table | No: AUC of the spread and lag-dropout network against S1 on shared rows | None |
| 463 | 404 table | No: learned weight against 50/50 on the same table | None |
| 467 | 404 table | Yes: gain over the lagless anchor and gaps to B-mix and K2-S1 | Added |
| 469 | 404 table | No: learned weight against 50/50 on the same table | None |
| 472 | 404 table | Yes: same as 467 | Added |
| 481 | 425A for seed averaging; 404 table for the S1 plus lagless-head comparison | Yes, for the S1 plus lagless-head comparison only | Added |
| 482 | 404 table (through experiment 454's stored forecasts) | Yes, for K2-S1 against A-pool above 24 h | Added |
| 486 | 404 table | Yes: B-mix's lead over the lagged network | Added |
| 488 | 404 table | Yes: the 50/50 mixture stays ahead | Added |
| 490 | 404 table | Yes: the lagless forecast carries information the raw readings do not | Added |
| 495 | 404 table | Yes: B-mix's lead over the lagged network | Added |
| 509 | 404 table (through experiments 454 and 482's stored forecasts) | No: each forecast is compared with itself with the injection off | None |

## Checks

- The classification reads each page's scripts and take-home message, not the stored forecasts, so a
  page marked "No" is a judgement about its contrast, not a rescore.
- No weather input was recomputed in this experiment. Experiment 532's row-by-row assertion covers
  397's summary tables (the feature source of 425A, A-pool, and 428) and not 425's ERA5 and CAMS
  pretraining rows, which are not forecasts.
- Triggers in "Audit before merge": none fired, because nothing was scored.

## What the next experiment should answer

The rebuilt K2-S1 on origin-time weather should restate the gaps above 24 h of lead and the
lagged-versus-lagless gain on the pages marked "Yes".
