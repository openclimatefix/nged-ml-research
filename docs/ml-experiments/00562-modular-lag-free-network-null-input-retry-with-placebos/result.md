---
id: "00562"
slug: modular-lag-free-network-null-input-retry-with-placebos
title: >-
  Do inputs that earlier networks found useless help the modular lag-free demand network, once each
  is compared with a shuffled-day placebo of the same width?
theme: demand-inputs
also_relevant_to:
  - retraining-and-training-history
status: planned
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
---

*Status: planned. Every number that experiment 562 produces is interim and unaudited.*

## Plan

**Question.** Earlier demand networks found four groups of inputs null or harmful. [Experiment
437](../00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md) found that
7 of 8 arms with thermal-lag, solar-gain, look-ahead, or calendar columns were worse by 0.016 to
0.059 points at the six. [Experiment
442](../00442-thermal-lag-cold-night-test-on-six-winter-folds/result.md) found cold-night gains of
-0.086 (six) and -0.014 (eight), both with intervals that include zero. [Experiment
513](../00513-lagless-demand-network-whole-day-price-shape-at-training-time/result.md) found an
oracle price shape worth -0.048 at the six and +0.037 at the eight. The modular network of
[experiment
545](../00545-modular-lag-free-demand-network-weather-and-datetime-encoders/result.md) reads weather
and calendar columns through separate encoders, which might extract signal that one multilayer
network missed.

**Design.**

- **Network.** Arm M0 of experiment 545 (`ModularMdn`), refitted daily with a 90-day recency
  half-life, on experiment 511's five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06; July 2026
  unused), 2 seeds. The pretrained M0 weights are experiment 554's, read-only.
- **How a column group enters.** Each new column group is appended to the 37 inputs and read by one
  encoder: weather-derived columns (B1, B2) by the weather encoder, calendar and price columns (B3,
  B4) by the datetime encoder. The first layer of that encoder gains one input unit per new column,
  initialised to zero, so each arm starts at M0's function. **The new columns are therefore seen
  only
  during ENS fine-tuning, not during ERA5 and CAMS pretraining.** This lets one set of pretrained
  weights serve every arm, and it is a departure from experiment 437, which pretrained each bundle;
  the write-up states it as a caveat.
- **Arms.**
    - B1: thermal-lag and solar-gain columns (experiment 437's T and S bundles, 14 columns), rebuilt
      on the half-hourly interpolated series with experiment 515's window rule.
    - B2: look-ahead windows (experiment 437's F bundle, 6 columns): forward means of temperature
      and
      shortwave over 6 and 24 hours after the valid time, and the share of each window the run
      covers.
    - B3a: five school-term columns. B3b: seven national-signal columns of the local day before the
      origin (price mean, maximum and 16:00 to 19:00 mean; Demand Flexibility Service flag; national
      demand; embedded solar; embedded wind).
    - B4: the 10 whole-day price-shape columns and the missing flag of experiment 513, plus a
      presence flag. In 50% of fine-tuning rows (drawn once per row, seeded) the flag is 1 and the
      real shape is present; in the others the flag is 0 and the shape is 0. At scoring the flag is
      0, so the forecast never needs price. A diagnostic arm, B4-oracle, scores the same fitted
      networks with the real shape present (flag 1). It is not a candidate.
    - A placebo for each arm (B1p, B2p, B3ap, B3bp, B4p) of the same width: every column is taken
      from a donor day with the same calendar month, day type, and length, at least 3 days away, on
      the same side of the fold's refit (experiment 515's `DayStructure`), with the donor's origin
      and
      valid time both moved by the whole-day shift. B4p uses experiment 513's shuffled shape (donor
      day of the
      same calendar month and day type, not the same day) with the same presence flags.
- **Gate.** All arms run on M0 first. A passing arm re-runs on the best lag-encoder variant of
  experiment 560 (2 seeds). That re-run waits for experiment 560's scores.
- **Standing rule: origin-time weather.** Every weather input for a target comes from the run issued
  at or before the forecast origin. The fit script asserts it row by row for each bundle and prints
  one example row per bundle with its source run, origin, and valid time. B1 columns use only the
  freshest earlier run before the origin (the analysis proxy); B2 columns use the same run, after
  the
  origin, which exists at the origin.
- **Availability at serving**, stated per column group in the write-up. A column that production
  could not read at the origin is dropped, except the price shape, which never enters at scoring.
  B1 and B2 need ECMWF runs (already ingested); B1's CAMS previous-day columns need the CAMS feed
  (published at 00 UTC); B3a is a fixed calendar; B3b needs daily Elexon and NESO feeds that
  production does not read today.

**Statistics.** **Seed 0 alone is a screening pass:** it has no seed noise estimate, so its
intervals cover the month-block resampling only, and the minimum detectable effect comes from seed 1
where that runs. The gate applies to the 10 all-rows contrasts; the Bonferroni level of 0.05/20
counts the cold-night contrasts too. Contrasts are real minus placebo for each arm, at the eight and
the six, on all
rows and on cold nights (as defined in experiment 526's scorer). That is 5 arms x 2 groups x 2
slices
= **20 primary contrasts**, so the Bonferroni level is 0.05/20. Placebo-minus-M0 contrasts and the
B4 without-price-minus-M0 contrast are reported beside them. A 5-block month bootstrap (near
min-to-max with 5 blocks), per-fold signs, and the minimum detectable effect from the seed spread
sit beside each contrast. D15 and D18 are report-only. The injected-shift leak check (experiment
511's
stress world) is run for each passing arm. A real-event leak check waits for the maintainer's
primary
measure. Rows with no labelled event means `tgt_real` false.

**Placebo caveats, to state in the write-up.** Donors can lie after the row's own origin (the
placebo is not causal, only the real arms are); rows without a donor (clock-change days, the first 5
days of runs) take the column mean, counted per fold and slice; B3b donors use one shift per origin
day, but the two CAMS columns of B1 are constant within a run yet use the valid-day shift.

**Standing practices followed.** Daily refit with a 90-day half-life (the harness of experiment
511; the weekly and monthly cadences belong to other questions); cleaned data; D15 and D18 rows
before 2025-03-01 cut; every fitted network saved in the private data folder; stored predictions
reused (M0 from experiment 545). Earlier advice followed: report the injected-shift leak and a
leave-D8-out table beside the CRPS gain (experiment 545's advice), and use a column-count-matched
shuffled-day placebo (experiment 526).

**Cost.** One arm of 5 folds x 2 seeds is about 300 daily fine-tunes, which took about 0.85
GPU-hours
in experiment 545. Ten arms (5 real, 5 placebo) are therefore about 8.5 GPU-hours, above the 5
GPU-hours the brief allows. The plan runs seed 0 of all ten arms first (about 4.3 GPU-hours), and
runs seed 1 only for arms that pass the gate or are near it, unless the manager approves the full
cost.

## What the next experiment should answer

*Written after the results.*
