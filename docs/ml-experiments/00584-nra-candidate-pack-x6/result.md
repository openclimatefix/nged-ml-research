---
id: "00584"
slug: nra-candidate-pack-x6
title: >-
  Does the best normal-running-arrangement forecast we can assemble (the embargoed three-step
  recipe trained with lag-only synthetic load-shifts, plus a wider band while a load-shift is
  suspected) put the normal running arrangement inside its band on 0.90 of rows after a load-shift
  ends, and is that more than extra width?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one new fine-tune per fold month, refit day, and seed (20 small neural-network
  fits, private data folder), plus tables and figures. More than one design could test the
  hypothesis; the arms follow the maintainer's brief. Spans experiments 574, 581, 582, and 583,
  whose code this experiment wraps. Serving path and degradation rule: not touched (this repository
  holds none). Cost: well under the 1.5 GPU-h ceiling (fits of about 10 seconds each, plus inference
  on the injected bank).
hypothesis: >-
  Training the embargoed three-step recipe of experiment 574 (P5) on lag inputs that carry synthetic
  load-shifts, and keeping a wider band while a gate on the lags suspects a load-shift, puts the
  normal running arrangement (NRA) inside the q05-q95 band on at least 0.90 of rows after a
  load-shift ends, on the injected bank, and the gain is more than the extra width of the band.
prediction: >-
  Written before any score. P1: on injected 15 and 30 day events, 24-120 h, 0-7 days after the end,
  C2 (C1 plus the gated band) covers the NRA on at least 0.90 of rows. P2: C4 (C1 widened by a
  constant factor to the mean width of C2 on event-free rows) covers the NRA on at most 0.85 of
  those rows, i.e. at least 0.05 less than C2. P3: C1 loses at most 0.08 points of capacity of CRPS
  to C0 at 24-120 h, and C2 and C3 add under 0.001 points of pinball loss at q05 and q95 on
  event-free rows. P4: on the verified December pair (2 Primary-events, descriptive), C2 covers the
  NRA after the end on more rows than C0 (0.40) and C1 (0.59). The prediction fails if C4 comes
  within 0.03 of C2.
decision_rule: >-
  Every event metric is descriptive (event-block bootstrap; leave-one-Primary-out range on the
  verified events). Counted tests: 16 (the contrasts C1 minus C0, C2 minus C1, C3 minus C1, and C4
  minus C2, times the eight MVA Primaries and the six MW Primaries, times the q05-q95 and the q01-q99
  band's pinball loss on event-free rows at 24-120 h). Each contrast uses a five-block month
  bootstrap at the Bonferroni level, and the half-width of that interval is printed beside it as
  the minimum detectable effect. The 0-24 h band is reported without a test. A band rule is worth
  porting only if its event-free extra pinball loss is no more than the minimum detectable effect,
  its coverage after the end is 0.90 or more on the injected bank, and its coverage exceeds the
  width-matched control C4 by at least 0.05.
benchmark: >-
  C0, the v3+D5 forecast of experiment 572 (stored), and the X3-syn results of experiments 581 to
  583 as references.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and q01-q99 by phase (before, days 0-3, day 4 to the end, 0-7 days
  after, 7-14 days after), one-sided NRA share, pinball loss at q01, q05, q95, q99, width, CRPS in
  points of capacity, and on event-free rows the width inflation and false triggers per
  Primary-year.
---

*Planned; the write-up replaces this page.*

- **Arms.** C0: v3+D5 (experiment 572, stored). C1: experiment 574's P5 schedule (step 2 stored,
  embargoed daily fine-tune re-run) with lag-only synthetic load-shifts on 5% of fine-tune time
  (experiment 581's recipe), refits on days 0 and 14, seeds 0 and 1. C2: C1 with experiment 583's
  A2 gated band. C3: C1 with the always-on union band. C4: C1 widened by a constant factor so its
  mean width on event-free rows equals C2's; C4-w (post hoc) matches the mean width of C2 on the
  after-end rows instead; C4-r places C2's widening on random days.
- **Standing practices followed.** Monthly-style coarse refits (days 0 and 14), cleaned data, the D15
  and D18 cut, July 2026 unused, weather from the run at or before origin minus 9 h, lags at least
  24 h old, anonymised generators. Fits saved as `.pt` in the private data folder.
