---
id: "00562"
slug: modular-lag-free-network-null-input-retry-with-placebos
title: >-
  Do inputs that earlier networks found useless help the modular lag-free demand network, once each
  is compared with a shuffled-day placebo of the same width?
theme: demand-inputs
also_relevant_to:
  - retraining-and-training-history
status: finished
kind: other
protocol: demand-v2
size: >-
  medium, close to complex. Stored: fitted network weights and window forecasts for each arm under
  this experiment's private data folder; no earlier experiment's files are written. Serving path:
  not touched (this repository holds none). Degradation rule: not touched. More than one design is
  defensible (whether the new columns are also pretrained, where each column group enters the
  network, whether the privileged price is present per row or per day), so the plan fixes each
  before any score is read. Callers: this folder's scripts import experiments 397, 428, 437, 448,
  510, 511, 513, 515, and 545. One Sonnet review of the plan and the code together before the first
  fit, one Sonnet review of the write-up against the printed report, and one Sonnet prose review.
hypothesis: >-
  The modular lag-free network of experiment 545 (arm M0) has a weather encoder and a datetime
  encoder, so it may extract signal from four groups of inputs that earlier networks found null or
  harmful: thermal-lag and solar-gain windows (B1), look-ahead weather windows (B2), school-term and
  national-signal columns (B3, one arm for each), and a whole-day price shape that is present only
  while training (B4). Each real arm is compared with a shuffled-day placebo of the same number of
  columns, so a gain counts only if it exceeds what extra columns give regardless of content.
prediction: >-
  Written before any score is read. P1: B1 and B2, real minus placebo on cold winter nights at the
  eight MVA-only Primaries, is at most -0.05 points of CRPS; the prediction fails if it is above
  -0.02. P2: school-term (B3a) is within 0.03 of its placebo on all rows (null again); it fails if
  the real arm gains 0.03 or more. P3: B4 scored without price beats plain M0 by at least 0.03 at
  the
  six signed-MW Primaries (experiment 513's oracle gave -0.048); it fails if the gain is within
  0.02.
decision_rule: >-
  An arm passes the gate if real minus placebo is at most -0.03 points of CRPS on all rows at the
  eight or at the six, with a Bonferroni interval that excludes zero. The threshold of 0.03 is the
  brief's, set because every added input column needs a feed in production. Passing arms, and any
  arm whose cold-night real minus placebo is at most -0.05, are re-run on the best lag-encoder
  variant of experiment 560. B4 additionally needs the without-price score to beat plain M0.
benchmark: >-
  Plain M0 (experiment 545's stored predictions, seeds 0 and 1, same rows and same pretrained
  weights) and each arm's own placebo. L-1d, A-pool, and manual_heuristic are report-only
  references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only Primaries and the six signed-MW
  Primaries, on all rows and on cold winter nights, with D15 and D18 report-only.
verdict: not supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  None passed the gate. B1 (thermal-lag and solar-gain columns) on cold nights at the six signed-MW
  Primaries, 3.381 points of CRPS against 3.612 for its placebo and 3.569 for plain M0.
headline: >-
  No input group beat its shuffled-day placebo by 0.03 points on all rows. Thermal-lag and
  solar-gain columns (B1) gained 0.231 [0.194 to 0.281] points over the placebo on cold nights at the
  six, in 3 months, and lost 0.049 on all rows at the eight.
lessons: >-
  The modular network does not find use for school-term columns, national signals, look-ahead
  windows, or a training-only price shape in a one-year test. The one gain is thermal lag on cold
  nights at the six, which a later test on more cold-night months must confirm.
take_home_message: >-
  Giving the modular lag-free demand network extra inputs that earlier networks found useless did
  not help over a whole year: no group beat a shuffled-day control of the same width by the
  threshold. The exception is the thermal-lag and solar-gain group (B1), which lowered the error on
  cold winter nights at the six signed-MW substations by about 0.2 points of capacity (3 months
  only), but raised it on all rows at the eight. The next step is a larger cold-night test of that
  group on the best lagged network.
advice:
  - >-
    Do not add school-term, national-signal, look-ahead, or training-only price-shape columns to the
    modular lag-free network, because none beat a shuffled-day control of the same width (figure 1).
  - >-
    Test thermal-lag and solar-gain columns (B1) on more cold-night months and on the best lag-encoder
    network before any port, because the gain is only on cold nights at the six in 3 months (figure
    1, Checks).
  - >-
    Add a shuffled-day control of the same width to every test of new input columns, because the
    control moved cold-night scores by up to 0.04 points by itself (Checks).
---

*Status: finished. Every number is interim and unaudited.*

**Take-home message.** None of the extra inputs helped over a whole year. We gave a neural network
that forecasts substation demand (the modular lag-free network of [experiment
545](../00545-modular-lag-free-demand-network-weather-and-datetime-encoders/result.md)) five groups
of inputs that earlier networks found useless, and compared each with a control of the same width
whose columns came from a different day. One group, thermal-lag and solar-gain columns, lowered the
error on cold winter nights at the six signed-MW substations, by 0.231 points of effective capacity
against its control. The same group raised the error on all rows at the eight MVA-only substations.
The next step is a larger cold-night test of that group on the best lagged network (experiment 560,
variant v3). The verdict is **not supported**.

## What was done

- **Network and folds.** Arm M0 of experiment 545, refitted daily, on experiment 511's five folds
  (July 2026 unused), 2 seeds for arms B1, B2, B3b, and 1 seed for the rest. Experiment 554's
  pretrained M0 weights were read, never written.
- **How the columns enter.** The new columns are seen only during fine-tuning on ensemble forecasts,
  through input units initialised to zero, so each arm starts as M0. Experiment 437 pretrained each
  bundle on ERA5 and CAMS rows, so this is a different test. Experiments 559 and 560 also read
  experiment 554's M0 weights.
- **Arms.** B1: thermal-lag and solar-gain columns of [experiment
  437](../00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) (14).
  B2: forward windows from the same run (6). B3a: school-term columns (5). B3b: national signals of
  the previous day (7). B4: the whole-day price shape of [experiment
  513](../00513-lagless-demand-network-whole-day-price-shape-at-training-time/result.md), present in
  a seeded 50% of fine-tuning rows with a flag and absent at scoring (12). Each arm has a
  shuffled-day control of the same width; B4-oracle scores B4's networks with price present.
- **Weather timing.** Every weather input comes from the run issued at or before the forecast
  origin. The fit script asserts that for every row, and the log prints one example row per fold.
  Experiment 437's causality check (later runs deleted, identical result) passed on Primary D2.

## Results

![Real minus placebo CRPS for each arm, with Bonferroni intervals](figure_1_real_minus_placebo.svg)

**No arm passes the gate.** The gate was real minus placebo of at most -0.03 on all rows, with a
Bonferroni interval excluding zero (20 primary contrasts, 5-month block bootstrap, near min-to-max).
Differences are CRPS points of effective capacity; negative favours the real columns.

- **B1:** all rows at the eight +0.049 [+0.007, +0.122]; at the six +0.015 [-0.056, +0.167]. On cold
  nights, 3 months: eight -0.059 [-0.157, +0.023], six -0.231 [-0.281, -0.194], with all 3 months
  negative at the six. B1 minus M0 on cold nights at the six is -0.188 [-0.211, -0.126], so the gain
  holds against plain M0 too.
- **B2:** all rows +0.017 (eight) and +0.005 (six). Cold nights at the eight -0.007 [-0.065, +0.012]
  and at the six -0.031 [-0.073, -0.012]. The 1-seed cold-night result at the eight (-0.037) did not
  survive the second seed.
- **B3a (school term, 1 seed):** all rows -0.027 [-0.238, +0.126] at the eight and -0.007 at the six.
- **B3b (national signals):** all rows -0.002 and +0.004; cold nights at the eight +0.026
  [+0.021, +0.031] (worse than its control).
- **B4 (price shape, scored without price):** all rows -0.008 and -0.006 against its control. Against
  plain M0 (1 seed) it is -0.004 at the eight and +0.010 at the six. B4-oracle against M0: -0.011
  (eight) and -0.001 (six), so even a real price shape at scoring gains little.
- **Absolute CRPS, all rows, eight and six:** M0 4.751 and 3.480 (seeds 0 and 1); B1 4.758 and 3.478;
  B2 4.747 and 3.474; B3b 4.726 and 3.468. Cold nights at the six: M0 3.569, B1 3.381, B1 control
  3.612.
- **Minimum detectable effect** (seed spread of two M0 seeds, 5 months): 0.087 (eight) and 0.104
  (six) on all rows, 0.106 and 0.227 on cold nights. Every cold-night contrast except B1 at the six
  sits near or under this, so the cold-night results are suggestive only.

**Predictions.** P1 (B1 and B2 cold nights at the eight at most -0.05): B1 met the point value
(-0.059) with an interval including zero, and B2 did not (-0.007). P2 (school term within 0.03 of its
control): held on all rows. P3 (B4 beats M0 by 0.03 at the six): failed.

## Checks

- **Trigger 5 (a control moves the score):** the shuffled-day controls moved cold-night scores
  against M0 by +0.043 (B1 control) and +0.026 (B2 control) at the six, with intervals that
  exclude zero. We did not test an explanation (extra uninformative inputs may add variance on the
  few cold-night rows), and no audit ran. B1's gain holds against both the control and M0.
- **Not run:** median minus mean forecast per site (the stored forecasts have no mean column), the
  error of the mean forecast beside the quantile score, leave-D8-out, the injected-shift leak check,
  and the lag-encoder re-run on experiment 560's v3 variant. Experiment 560 merged during this
  experiment. The re-run needs new harness code, and only one slice supports it.
- **Placebo limits:** controls take a donor day's columns, which can lie after the row's own origin;
  rows with no donor (clock-change days, the first 5 days of runs) take the column mean (about 0.4%
  of 2026-01 rows); B4's control has no 3-day minimum separation; the two CAMS columns of B1 use the
  valid-day shift though they are constant within a run.
- **Availability at serving:** B1 and B2 need ECMWF runs and, for two B1 columns, the CAMS feed. B3a
  is a fixed calendar. B3b needs daily Elexon and NESO feeds that production does not read today.
  NGED's event logs may be incomplete, so rows with no labelled event can hold unlabelled events.

## What the next experiment should answer

Does B1 lower cold-night error at the six on more than 3 cold months, and on the best lagged network
(experiment 560, variant v3), without the loss on all rows at the eight?
