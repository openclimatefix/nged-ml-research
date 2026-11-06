---
id: "00589"
slug: p5-coverage-on-d5-before-the-december-load-shift
title: >-
  Why does the P5 forecast put the power of Primary D5 inside its 90% band on under half of the rows
  before the December 2025 load-shift, and is that one Primary or a property of the P5 recipe?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: nothing in the first two stages (tables and figures from stored forecasts, no fit);
  the optional third stage is 5 seeds x 1 fold month x 2 embargo settings of the daily fine-tune
  (about 10 s each), written to the private data folder. More than one design could test the
  hypothesis (a bias, a width, an embargo, or an earlier unflagged shift can each explain the
  coverage). Spans experiments 572, 574, 584, whose code and stored forecasts it reads. Serving path
  and degradation rule: not touched (this repository holds none). Ceiling 1 GPU-h.
hypothesis: >-
  The low coverage of the P5 recipe at D5 before the December 2025 load-shift is mainly a downward
  bias of the median at that one Primary (observed power above q95), not a narrow band in general,
  and P5 does not under-cover on event-free rows at the other Primaries.
prediction: >-
  Written before any score. P1: on event-free rows 24-120 h, pooled q05-q95 coverage of the observed
  power is within 0.03 of C0's for P5 and C1 at both metering groups, with D5 the only Primary more
  than 0.15 below C0. P2: at D5 in the 2025-12 fold before the onset, the median of P5 is below the
  observed power on most rows (share above 0.70), and the share of observed power above q95 is at
  least 0.40. P3 (only if stage 3 runs): shortening the embargo from 3 days to 1 day raises D5's
  coverage before the event by at least 0.10. The prediction fails if D5's coverage gap to C0 is
  matched by a gap of similar size at three or more other Primaries on event-free rows.
decision_rule: >-
  Descriptive for the stage 1 and 2 tables (no test; the shares carry month-block bootstrap
  intervals over the five fold months, and the single D5 fold has one block, so no interval).
  Stage 3 contrasts (P5 with a 1 day embargo minus P5, and v3+D5 as a reference) use five seeds on
  one fold month, so each interval is a seed range, not a test. Bonferroni count: 2 (the
  coverage contrast at D5 before the onset, and the same contrast on event-free rows of the other
  Primaries, for the embargo change). Minimum detectable effect printed beside every contrast. The
  verdict names the cause: a one-Primary effect, a property of P5, or noise.
benchmark: >-
  C0, the v3+D5 forecast of experiment 572 (stored), P5 of experiment 574 (stored), and C1 of
  experiment 584 (stored).
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of observed power inside q05-q95 (and q01-q99), the shares above q95 and below q05, the
  median minus the observed power, and the mean q05-q95 width, by Primary, fold month, and lead band
  (0-24 h and 24-120 h).
production_status: not-for-production
adopted_in: none
---

**Planned, not run.** Plan: (1) tabulate coverage by Primary, month, and lead band for C0, P5, and C1
from the stored forecasts of experiments 572, 574, and 584, on rows with no labelled or verified
event within 7 days; (2) diagnose D5 in the 2025-12 fold (median bias, width, embargo, the weather
and thermal-lag columns, and the 19 days of gate-on before the onset); (3) only if stage 2 points at
the embargo or the schedule, a five-seed test on the 2025-12 fold with a 1 day embargo.
