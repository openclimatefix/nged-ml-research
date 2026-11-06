---
id: "00598"
slug: nra-gate-against-signals-as-inputs-at-equal-band-width
title: >-
  Does a neural network that reads the gate's signals as inputs cover the normal running arrangement
  as well as the hand-built gate does, at the same band width?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the triggers: (1) it stores new fitted network weights and forecasts in the
  private data folder; (2) more than one design could test the question (the plan fixes one and
  lists the others as departures); (3) it spans code whose callers cannot be named without
  searching (experiments 574, 581, 584, 591, 592, and 597); (4) it touches no serving path and no
  degradation rule; (5) it changes no contract. The plan review is one Sonnet agent, the code review
  one Sonnet agent, and the write-up review one Sonnet agent.
hypothesis: >-
  A neural network that is fine-tuned on synthetic lag-only load-shifts and is given the gate's
  signals as inputs can set its own band width as well as the hand-built G4 gate sets it, so the
  gate is not needed.
prediction: >-
  Written before any score. P1: arm S reaches arm R's day-4-to-end and after-end coverage within
  0.01, at R's event-free band width within 2% (fails if S is 0.02 or more below R). P2: S beats arm
  W (R's band widened by a constant factor to R's mean width on event rows) by at least 0.03 after
  the end. P3: S's follow ratio is at most 0.08. P4: arm S-p (the signals replaced by noise of the
  same variance) is at least 0.02 below S, otherwise the inputs carry nothing. P5: S's event-free
  CRPS is within 0.03 points of capacity of A1's.
decision_rule: >-
  P1 and P2 both hold: prefer S, because it has no hand-tuned threshold, no neighbour rule, and no
  false-trigger rate; say that one real pair and the step from synthetic to real events remain
  unproven. S is 0.03 or more below R from day 4 to the end in the first fold pair (2025-10 and
  2025-12): run only S-g next. S-g also fails: keep the gate and stop.
benchmark: >-
  Arm R, which is A1 + G4 from experiment 597 (stored forecasts, export in
  `backtest_export_candidates_2026-11-06`).
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 and on the safe side of q05 or q95, by phase; median follow ratio;
  band width; event-free CRPS at 0-24 h and 24-120 h; tail pinball.
---

**Planned, no score read yet.** We ask whether a neural network that reads the signals the G4 gate
uses can learn to widen its own band during a load-shift, so that the hand-built G4 gate is not
needed. A load-shift moves part of one Primary substation's demand onto a neighbour for days or
weeks. The comparison is at equal band width, because a wider band alone raises coverage.

## Plan

- **Arms (all P5 recipe with A1 wiring, fine-tuned on lag-only synthetic shifts mirrored at one
  neighbour).** R: A1 + G4 band (stored from [experiment
  597](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00597-nra-assemble-c1-and-a1-with-g4-gate-as-single-forecasts/result/)).
  W: A1 with the band widened by a constant factor to R's mean width on event rows. S: signals as
  inputs, no gate. S-g: S plus the G4 state bit. S-p: S with noise of the same variance in place of
  the signals.
- **Reuse.** The A1 pre-training and step-2 weights of [experiment
  591](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00591-shared-lag-encoder-auxiliary-shift-depth-loss/result/)
  start every S arm, with the new input columns zero-initialised, so each S arm begins as A1. Only
  the fine-tune is new, which is why the GPU estimate is below the brief's 3 GPU-h ceiling.
- **Cadence.** Coarse refits (day 0 and day 14), as in experiments 584 and 591; seeds 0 and 1; five
  development months 2025-10, 2025-12, 2026-01, 2026-03, 2026-06, first fold pair first. D15 and D18
  are cut before 2025-03-01 and reported only.
- **Where the signals come from (a departure from the brief's wording).** The brief takes the
  disagreement from the lags-masked twin's median. Stored lag-free forecasts exist only for the
  five test months, so training rows would have none. Instead, a frozen copy of the fold's A1
  step-2 network, run with its lags masked, gives a lag-free median and band for every grid time and
  Primary, in training and test months alike (the table `T`). Every signal is then a lookup on `T`
  plus the observed power, as the G4 gate does with E2: the disagreement of each lag with `T` at the
  lag's own time, its sustain count, the fresh residual (observed power minus `T` up to the origin
  minus 4.0 h, never as a lag), the lag ages, the neighbour's disagreement, and `T`'s q05 and q95.
  `T` is in-sample on training rows and out-of-sample on test rows; this is a caveat that S-p tests.
- **No lookahead.** The disagreements and the sustain count use lags at least 24 h old. The fresh
  residual uses power up to the origin minus 4.0 h. A training row's signals use only its own
  synthetic lags and earlier rows.
- **Counted tests.** Seven: P1 (two phases), P2, P3, P4, P5 (two groups). The Bonferroni level and
  the minimum detectable effect are printed beside each contrast.
- **Stop rule.** As in the decision rule. One GPU process at a time under 8 GB; ask the manager
  before crossing 3 GPU-h.
- **Maintainer's example event (2026-11-06).** Plots of forecast behaviour during and after
  load-shifts use the December 2025 pair (D5 and D16), which the maintainer dates "2025-12-30 01:00
  to 2026-01-14 11:30". One figure per Primary shows 14 days before to 14 days after, with dashed
  vertical lines at those two times, for R, W, S, S-g, and S-p. The verified window in [experiment
  580](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/result/)
  is 2025-12-29 to 2026-01-13, about a day earlier. The scoring script checks the readings to see
  which window matches the step, and the caption says which, and does not choose silently. The
  predictions are unchanged.
- **Time-series plot style (maintainer, 2026-11-06).** Every time-series figure has panels at least
  280 px high and at most two panels per figure. Each panel shows one arm's median and q05-q95 band
  with the observed power and the NRA, and a thin strip under the panel marks when the gate is on.
  A legend outside the data area names each line, band, shade, and bar. The five arms (R, W, S,
  S-g, S-p) therefore go in separate figures or small multiples of one arm each, never on one panel.
- **Further figure rules (maintainer, 2026-11-06).** The NRA line is not drawn on time-series plots,
  because it is an estimate, and each caption says coverage is scored against that estimate.
  Observed power is black, no other mark is black or grey, and the load-shift shade is a pale colour
  that no arm uses. A gate strip is checked against the stored gate state before it is published.
  Before the pull request is marked ready, a fresh Sonnet reviewer opens each figure rendered to PNG
  and answers the six-question figure checklist in writing; the pull request body records "figure
  review: passed" with the date.
- **Changes after the Sonnet plan review (all 12 findings accepted; the review is in the private
  briefs folder as `598_plan_review.md`).**
    - **Equal width.** Each S arm also gets a width-matched control: its band rescaled about the
      median so its mean q95 - q05 on R's G4-on-or-held origin days equals R's, one factor per fold
      and seed, printed with its row count. The factor is fitted on the injected bank's 24-120 h
      rows, which are the rows P1 and P2 read. P1 and P2 are read on the rescaled S as well as on
      S, and count as passed only if S's own event-row width is within 5% of R's or the rescaled S
      also passes. W uses the same factor rule applied to R.
    - **Training signals are in-sample.** Cross-fitting the frozen network is too expensive, so a
      pre-fit check prints, per Primary, the standard deviation of the fresh residual and of the lag
      disagreement on training rows against test rows. Where the training-to-test ratio is below
      0.8, Gaussian noise is added to the training-row signals to match the test standard
      deviation, and the write-up says so.
    - **The table.** `T` holds the lag-free median by Primary and grid time, taken from the freshest
      origin that covers it. The lags-masked q05 and q95 are per row (Primary, origin, valid time),
      computed by one masked inference pass over the rows used. The build asserts that no entry
      used for an origin comes from an NWP run issued after that origin.
    - **Signals in training.** The synthetic shift is added to the signal inputs (lag disagreement,
      fresh residual, neighbour disagreement) and not to the target, as `FreshGap.gap(shifts=...)`
      does in experiment 592. The sustain count and the neighbour disagreement are precomputed per
      row from earlier origins only and are zero where history is missing, with an assertion. A
      known-answer row checks the shift reaches each signal.
    - **Reference parity.** A fifth reference, G4 computed on `T` instead of E2, separates the gate
      from the change of expectation. Arm S-p (noise signals, same fine-tune and columns) is the
      second fine-tune of the A1 wiring with no usable signal, so it serves as the parity arm.
    - **Outcome bands.** Gap of S to R in day-4-to-end coverage: under 0.01 passes, 0.01 to 0.03 is
      inconclusive (S-g is not run), 0.03 or more fails (run S-g as the stop rule says).
    - **Detectability.** The minimum detectable effect of every prediction is printed before the
      fits are scored; a prediction whose margin is below its minimum detectable effect is reported
      as untestable. P5's margin is fixed at 0.03 points of capacity of event-free CRPS (S minus A1,
      both metering groups), set before any score; its interval resamples the fold months as
      blocks. R's and S-p's follow ratios are printed beside P3. Contrasts counted: seven, as
      above. S-g, the target-side absorption score (experiment 595's method, for R and S), and
      the real pair are descriptive and uncounted.
- **Standing practices followed.** Monthly-equivalent coarse refits, cleaned data, the D15 and D18
  cut, saved weights, reuse of stored forecasts, tails and avoidable-curtailment cost, load-shift
  event plots (at least 6, both metering groups), example forecasts, and the unlogged-event caveat.

Reviews: one Sonnet plan review, one Sonnet code review before the first fit, one Sonnet write-up
review (details in the pull request).
