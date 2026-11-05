---
id: "00563"
slug: pooled-primary-bsp-gsp-data-check
title: >-
  Are there enough clean bulk and grid supply point series to test one neural network pooled over all
  Primaries, bulk supply points, and grid supply points?
theme: hierarchy-and-pooling
also_relevant_to: []
status: done
kind: other
date: 2026-11-04
size: >-
  simple, by the five triggers: nothing is stored beyond one metadata table read; one design (a
  count and a quality table); the callers are none, because no script is written; the serving
  path and the degradation rules are not touched in this repository. One Sonnet write-up review.
hypothesis: >-
  At least 8 bulk supply point (BSP) and grid supply point (GSP) series in NGED's data are clean
  for 2 years before each fold cutoff and have a weather location, so that a pooled neural network
  (arm M2 of the modular-architecture item) can learn from more series than the 16 Primaries.
prediction: >-
  Written before the count. At least 8 series pass. This was a guess, because the brief did not
  know the number of series.
decision_rule: >-
  The brief's own stop rule: fewer than 8 clean series with a weather mapping means stop and
  report, with no GPU fit.
benchmark: none, because no model is fitted.
row_set_id: none, because no model is scored.
metric: Count of series, first and last timestamp, and share of zero and flagged rows.
verdict: inconclusive
code: none
best_arm: none
headline: >-
  NGED's metadata holds 4 bulk supply point (BSP) and grid supply point (GSP) series, not 8: 2 BSPs
  and 2 GSPs, all signed MW. The pooled neural network test (M2) was stopped before any fit.
take_home_message: >-
  NGED's metadata holds four bulk and grid supply point series. Adding them to the 16 Primaries
  gives 20 series, too few to test whether many extra series help. By the hierarchy file, each
  grid series should include one bulk series, so the four are not independent. The test was not
  run. Experiment 415 already fitted a pooled neural network at the 16 Primaries and these four
  series.
lessons:
  - >-
    In this check, the number of BSP and GSP series in NGED's metadata (4) fell below the 8 the
    experiment needed.
advice:
  - >-
    Do not plan an experiment that needs 8 or more bulk or grid supply point series, because
    the metadata holds 4 (this page, table of series).
production_status: not-for-production
adopted_in: none
---

**The check stopped the experiment: NGED's metadata holds 4 bulk supply point (BSP) and grid
supply point (GSP) series, below the 8 the experiment brief required.** No neural network was
fitted and no GPU time was used. By the hierarchy file, each GSP's power should include one BSP's
power. I did not compare the power series themselves.

## The four series

All four are signed MW "Raw Flow" series in `metadata.parquet`. History is from the raw power
table, and zero and flagged shares are from the cleaned table.

| Series id | Kind | Substation | First reading | Last reading | Zero share | Flagged share |
|---|---|---|---|---|---|---|
| 1 | BSP | Boston 33kV | 2021-11-29 | 2026-11-04 | 0.05% | 0.05% |
| 2 | BSP | Skegness 33kV | 2019-09-17 | 2026-11-04 | 0.004% | 0.004% |
| 27 | GSP | Walpole 132kV | 2019-09-14 | 2026-11-04 | 0.12% | 0.12% |
| 28 | GSP | Bicker Fen 132kV | 2019-09-11 | 2026-11-04 | 0.89% | 0.89% |

- Every flagged row is a zero row, so no other cleaning rule fired on these series.
- Series 2, 27, and 28 have at least 2 years of history before any cutoff from 2021-10 on, and
  series 1 from 2023-12 on. I did not check the gaps inside that history against each of the five
  folds of experiment 511's row set.

## Parent-child map

The hierarchy file received from NGED on 2026-10-15 lists 17 rows under Boston BSP (the BSP and 16
children) and 13 rows under Skegness BSP (the BSP and 12 children). Boston BSP sits under Walpole
GSP and Skegness BSP under Bicker Fen GSP. So each GSP should include one BSP and the BSP's
children, and the two GSP series are probably close to the BSP series. The residual of GSP minus BSP
was not measured.

## Weather mapping

Each of the four series has a latitude, a longitude, and a resolution 5 H3 cell in the metadata,
and each cell is present in `data/h3_grid_weights.parquet`. A weather location therefore exists
for all four without a fetch.

## What this means

- **M2 is not run.** Four series do not meet the bar of 8, and the stop rule said to halt.
- **Pooling at these four series already exists.** Experiment 415 fitted one neural network on the
  16 Primaries and its four nodes, which are these four series, so the step was already taken at a
  small scale.
- **No other bulk or grid series exist** in NGED's metadata, so the arms M2-a to M2-c cannot be
  filled by waiting for a download. A reduced four-series version would need the maintainer's
  request.

## Checks

- Leakage: not applicable, because nothing was fitted.
- Origin-time weather assertion: not applicable, because nothing was fitted.
- Generators: none are shown.
