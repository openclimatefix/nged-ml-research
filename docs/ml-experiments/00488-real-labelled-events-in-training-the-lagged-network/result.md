---
id: "00488"
slug: real-labelled-events-in-training-the-lagged-network
title: >-
  Do real labelled events, mixed into training or used to fine-tune, improve the lagged network?
theme: lagged-power-and-error-correction
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
kind: other
size: >-
  medium. Stored: predictions and models under the experiment's data folder only. Serving path: not
  touched. Degradation rule: not touched. Designs: one per arm, fixed by the accepted brief and its
  queue-order check. Callers: this folder's scripts, which import experiments 404, 428, 435, and
  455 without editing them. Asset graph: not touched. A diagnostic only: it needs event labels that
  NGED will not supply at about 1,300 substations.
hypothesis: >-
  Adding the real labelled load-shift events from months before the scored month to training, with
  their targets corrected to the normal running arrangement (NRA), or fine-tuning the stored A0-inj
  network on them, improves the lagged network on rows inside real events scored against NRA-corrected
  actuals, and on real harm rows, beyond what injected events already give.
prediction: >-
  Written before any score is read. P1: RealMix or PT-FT improves the NRA-corrected event rows at the
  signed-MW six by 0.05 to 0.2 points of effective capacity against A0-inj. This is below the minimum
  detectable effect (MDE), so it is reported as "within the MDE of A0-inj", never as a pass; a gain
  above the MDE, or a loss beyond it, falsifies P1. P2: RealMix-shuf (real shape and depth, injected
  into a random clean period) does not improve on A0-inj, so any gain comes from the real events'
  timing and content. P3: no arm beats B-mix on rows with no labelled event. P4: the real-event
  target rows are already 3.45% of the 2026-06 training rows, above the 2% target, so RealMix
  differs from A0-inj only by the corrected target.
decision_rule: >-
  Fixed before any score is read. Contrasts (k = 12, Bonferroni, level 1 - 0.05/12): RealMix,
  PT-FT, and RealMix-shuf, each minus A0-inj and minus B-mix, on injected harm rows (worlds 200 to
  202, where events are many) and on NRA-corrected real-event rows at the six (depth 1.0
  headline; 0.75 and 1.25 as sensitivities). Real harm rows, rows with no labelled event, and
  the eight MVA-only Primaries are descriptive. The MDE is stated from two sources before any
  contrast is read: the seed spread of A0-inj and B-mix at 2026-06 (seeds 0 to 2), and experiment
  404's month-to-month spread (0.4 to 1.0 points on harm rows, 0.3 to 0.9 inside events, 0.3 to
  0.8 on rows with no labelled event). Experiment 439's clean-row seed MDE (0.061) is the bar for
  equivalence on rows with no labelled event. A gain under the MDE is reported as "within the MDE
  of A0-inj". Stop rule: if the real-event interval for RealMix and PT-FT excludes no gain and
  the point estimate is under the MDE, the result is that injected events carry what the real
  labels add. A gain is worth porting only as a label-free recipe; the write-up states whether the
  real lags or the corrected target carries it.
benchmark: >-
  Stored A0-inj and B-mix (50/50 mixture of A0-inj with the cross-fitted lagless table) of
  experiment 404, same months, rows, and seeds; plus a refit of the A0-inj recipe with this
  experiment's loop (arm `repro`) as the reproduction check.
row_set_id: >-
  Experiment 404's scored rows: five months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), seed 0,
  and seeds 0 to 2 at 2026-06 for the reference noise floor; the clean world and the injected
  worlds 200 to 202; flagged zeros dropped.
metric: >-
  Mass-weighted pinball (CRPS-like score) in points of effective capacity by row class and group,
  with q01, q05, q95, and q99 coverage and pinball loss, avoidable curtailment cost, and the
  encroachment sweep.
---

**Status: planned.** The experiment asks whether the real labelled events add anything to the
injected events the lagged network (A0-inj of experiment 404) is already trained on. It is a
diagnostic: NGED will supply no event labels at about 1,300 substations, so a gain cannot ship
without a label-free way to find events.

## Plan

- **Forward-only.** The fit for scored month m uses real events that ended at least 14 days before
  m starts. Events available at the six: 20 (2025-10), 28 (2025-12), 29 (2026-01), 38 (2026-03),
  43 (2026-06); no month has fewer than 5, so none is dropped. All 16 Primaries: 64, 84, 88, 106,
  113.
- **Arms.** `RealMix`: training rows whose target lies inside an available event get the
  NRA-corrected target (actual minus the labelled step) and the real lags; the uncorrected copies are
  removed. `PT-FT`: the stored A0-inj network, fine-tuned for 2 epochs at one tenth of the learning
  rate on the corrected event rows plus an equal number of rows with no labelled event from the
  same months. `RealMix-shuf`: each event's lag pattern and depth is injected into a random clean
  period of the same length in the same Primary, with the clean target. `RealMix-d075` and
  `RealMix-d125` repeat RealMix with the correction scaled by 0.75 and 1.25, at 2026-06 only.
- **Fits.** 18: three arms × five months, two depth sensitivities, one reproduction. One fit takes
  about 78 s on the GPU (about 0.022 GPU-hours), so the fits cost about 0.4 GPU-hours. Monthly
  refits, seed 0.
- **Oversampling.** The real-event target rows are 3.45% of 2026-06 training rows, above the
  brief's 2% target, so RealMix keeps each corrected row once and does not oversample.
- **Reused.** Stored A0-inj and B-mix forecasts and the A0-inj models of experiment 404 (read
  only), the injected worlds of experiment 428, and the NRA correction of experiment 435.
- **Caveats to state.** NGED's event labels are incomplete, so rows with no labelled event may hold
  unlabelled events and real-event slices undercount. Depth error is the main risk of the corrected
  target.
