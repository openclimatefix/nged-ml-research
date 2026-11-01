---
id: "00452"
slug: pretraining-cap-and-surprise-inputs-on-the-425-arm-a-recipe
title: >-
  Do the surprise inputs and a longer pretraining cap improve experiment 425's pretrained pooled
  mixture density network for Primary demand?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored in this experiment's private folder only: 50 pretrained and 50 fine-tuned
  networks (two new arms, five folds, seeds 0 to 4). Experiments 404, 425, 428, 448, and 449 are
  read and never written. More than one design is defensible (which arms isolate the cap from the
  surprise inputs), so the plan fixes each choice before any score is read. Production serving
  path: not touched (this repository holds none). Degradation rule: not touched. Callers: the
  scripts copy the fit and score code of experiment 449 (not yet on `main`) and import the folders
  of experiments 397, 404, 418, 422, 428, and the `studies` evaluation code.
hypothesis: >-
  Experiment 449's audit found that experiment 425's arm A (a pretrained pooled mixture density
  network without surprise inputs) beats PT1-pool5 (the same network with the surprise inputs), but
  the two used different fit loops and arm A averages five seeds' quantiles, and all 55 of 449's
  pretraining runs stopped at the 60-epoch cap. If pretraining was too short, then raising the cap
  to 150 epochs improves the network with the surprise inputs (A-S1-long) and the network without
  them (A-long), and the surprise inputs add to the longer pretraining.
prediction: >-
  Written before any fit and before any score is read. P1. A-S1-long beats PT1 (cap only, C1) by
  0.02 to 0.08 points at the eight MVA-only Primaries, because the 60-epoch cap stopped every
  pretraining run before its validation loss stopped improving. P2. A-long matches A-S1-long within
  0.03 points (C3), so the cap and not the surprise inputs does the work, because 449 found the
  surprise inputs added little beyond 425's recipe once the 2026-01 fold is set aside. P3. Neither
  new arm beats experiment 425's arm A by the minimum detectable effect at the eight. The prediction
  could fail in either direction: a longer cap could overfit the ERA5-and-CAMS pretraining rows and
  worsen cold nights.
decision_rule: >-
  Four folds (2025-10, 2026-01, 2026-03, 2026-06) and five folds (adding 2025-12), seeds 0 to 4,
  monthly refits, 1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity
  (negative is a gain), on the half-hours every arm and reference shares (579,044 on four folds,
  713,323 on five). Gating rows are the eight MVA-only Primaries, cold winter nights, and Christmas
  week (22 to 31 December, one fold; report-only if its interval spans zero), each with a
  month-block bootstrap and Bonferroni intervals over 26 tests. Report-only: D15 and D18 on real
  rows (83,458 rows on four folds, minimum detectable effect 1.18). Secondary: all 16 Primaries and
  the signed-MW six (five-fold minimum detectable effect 0.34, so unreadable). The scorer prints
  every minimum detectable effect before any contrast is read. Adopt A-S1-long over PT1 only if its
  eight-Primary CRPS improvement is at least the minimum detectable effect, its cold-night and
  Christmas-week scores do not worsen by more than 0.02, and its q99 pinball loss does not worsen
  by more than 0.02. Adopt A-S1-long or A-long over A-ref on the same terms, with the loop confound
  stated. If A-long matches A-S1-long, the cap and not the surprise inputs does the work. If no
  arm beats A-ref, experiment 425's arm A (and its pool of five) stays the candidate and demand
  pretraining stops.
benchmark: >-
  A-ref (experiment 425's arm A, seeds 0 to 4, stored), A-S1 (experiment 449's PT1, seeds 0 to 4,
  stored), BG2, S1, and `manual_heuristic` as stored references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99;
  cold winter nights; Christmas week; avoidable-curtailment cost at f = 0.9, 0.75, 0.5 at the
  signed-MW six. Per group and per Primary.
verdict: null
production_status: not-for-production
---

## Plan

**Question.** Does the longer pretraining cap, the surprise inputs, or both improve the pooled plain
mixture density network that experiment 425 assembled, on the rows that decide the first release?

**Arms.** All are pooled over the 16 Primaries with the one-pass ensemble mean and spread, an 8-wide
Primary embedding, and a 3-component Gaussian mixture. Every new fit uses experiment 428's fit loop
(0.2 of the rows per epoch, patience 8, injected worlds), as experiment 449 did.

- **A-ref (stored).** Experiment 425's arm A, run in experiment 397's fit loop.
- **A-S1 (stored).** Experiment 449's `PT1`, 60-epoch pretraining cap, seeds 0 to 4, five folds.
  Also scored as a five-seed quantile average, so it is seed-matched to A-ref.
- **A-S1-long (new).** `PT1`'s recipe with the pretraining cap raised to 150 epochs.
- **A-long (new).** The same loop and cap as A-S1-long without the surprise inputs.

**Reused, not rebuilt.** The 25 ERA5 lagless tables of experiment 449 for pretraining, and
experiment 404's frozen cross-fit tables (seeds 0 to 5) for fine-tuning and test.

**Clean contrasts.** A-long minus A-S1: surprise inputs plus cap. A-S1-long minus A-S1: cap only.
A-S1-long minus A-long: surprise inputs. A-S1-long and A-long minus A-ref are confounded with the fit
loop (397 against 428). A-long minus A-ref mixes loop and cap.

**Compute.** 100 networks (25 pretraining and 25 fine-tuning runs per arm). One pretraining run is
timed first; the manager receives the fit count and GPU-hours before the remaining fits.

## Result

To be written after the fits.
