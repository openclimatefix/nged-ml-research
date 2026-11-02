---
id: "00507"
slug: history-encoder-first-test-oracle-repair-and-total-variation
title: >-
  Is there enough to gain from repairing shifted power readings, rather than down-weighting them, to
  justify a learned 14-day history encoder?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
date: 2026-11-02
size: >-
  complex. Stored: about 100 small saved neural networks, their forecast tables, and injected
  worlds, all under the experiment's private data folder. Serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: more than one design would
  defensibly test the hypothesis (how a repair is scored, the injection generator, the
  total-variation repair), so the plan fixes each. Callers: the scripts import experiments 404,
  428, 454, and 472, and the callers of that chain cannot be named without searching. Reviews: one
  Sonnet plan review, one Sonnet code review before any number is used, one Sonnet results review,
  and two Sonnet write-up reviews. The brief asks for Sonnet only.
hypothesis: >-
  Repairing the shifted readings in the last 14 days of power, instead of down-weighting them as
  experiments 402 to 482 did, leaves enough forecast gain to justify a learned history encoder. A
  perfect (oracle) repair of the three lag readings improves the lag correction of experiment 472
  (on the ERA5 lab anchor) on rows where a shift is in force. A non-learned repair (total-variation denoising of
  the 14-day residual) keeps less than half of that gain once genuine level changes are in the
  data, and a learned encoder can reach the accuracy that the oracle's noise curve demands.
prediction: >-
  Written before any fit. Scores are CRPS in points of effective capacity at the signed-MW six on
  held-out injected worlds. P1. The retrained oracle correction beats the retrained plain
  correction on injected harm and target rows by at least 0.3 points (the brief's prediction; the
  stop threshold of rule (a) is the injected-row minimum detectable effect, which the plan expects
  to be near 0.05 to 0.15). P2. Total-variation repair keeps under half of that gain on the
  new-generator worlds, because it erases genuine level changes (60% of the steps in force at the
  origin). P3. The encoder's onset error, size error, and share of steps missed fall in the part of
  the noise curve that keeps at least half of the oracle gain. Any of the three can fail.
decision_rule: >-
  Fixed in the section "Decision rule", below, before any score is read. In order: (a) stop the
  whole line if the retrained oracle correction does not beat the retrained plain correction on
  injected harm and target rows by more than the injected-row minimum detectable effect (from the
  3-seed spread of the retrained plain control) with an interval excluding zero; (b) promote the
  non-learned repair and stop the neural encoder only if total-variation repair keeps at least half
  of the oracle gain on the old worlds and on the new-generator worlds with genuine changes (lower
  end of the interval on the paired difference of TV gain and half the oracle gain at or above zero), is not shown worse than the plain correction by more than
  0.03 points on rows with no labelled event at the six, and is not shown worse by more than 0.02
  points on q99 pinball loss; (c) stop the neural encoder if the encoder's errors fall where the
  noise curve keeps under half of the oracle gain, or its validation step loss stops falling
  before epoch 10. The smallest gain worth porting is the injected-row minimum detectable effect,
  because the repair adds a convex solve per Primary and a 672-half-hour history store.
benchmark: >-
  The retrained plain correction (experiment 472's recipe, 32 units, penalty 1e-2, fit seeds 0 to 2)
  on the same ERA5 and CAMS lab lagless table. Every contrast is lab against lab. The stored
  ENS-based AvgAnchor and K2-S1 forecasts are not contrasted (second set of edits, 2026-11-02).
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of effective capacity from 11 quantiles, by row
  class and group, as in experiment 472's report; q01, q05, q95, and q99 coverage and pinball loss.
code: >-
  ml-experiments/00507-history-encoder-first-test-oracle-repair-and-total-variation/
---

**This page is the plan; the write-up replaces it when the experiment finishes.** The plan follows
the neural Discuss session's brief
`briefs/history_encoder_first_test_brief.md` with its edits of 2026-11-02, the queue-order check
`briefs/queue_order_check_history_encoder_answer.md`, and sections 2.2 to 2.8 of the final design
`briefs/history_encoder_debate_3_opus_reply.md`. Rungs R1 and later are never started.

## Question

Do the readings of the last 14 days need repairing, or is it enough to down-weight them? A perfect
(oracle) repair that gains nothing ends the line. A cheap non-learned repair that keeps half of the
oracle's gain replaces the neural encoder. Otherwise one encoder fit says whether a learned repair
can reach the accuracy the oracle's noise curve demands.

## Design

### Lab setting

- **Weather everywhere is ERA5 and CAMS, never ensemble forecasts**, on the history side and on the
  forecast side. The lagless forecast is experiment 449's cross-fitted ERA5 and CAMS lagless
  module, one seed (seed 0). The lab surprise (reading minus lagless median) is smaller than at
  serving, because ERA5 knows the weather that happened, so detection is flattered. The page says
  so, and says that a single-seed encoder or lagless null is informative only when it lies far
  from the minimum detectable effect.
- **The stored ERA5 tables of experiment 449 are rebuilt, not read as they are.** They hold only
  the 5%, 50%, and 95% quantiles and only the cross-fitted half-hours before each refit, so they
  have no 9-number mixture for the correction's anchor and no forecast for the scored month. The
  rebuild uses experiment 449's code and the same cross-fit blocks, saves the mixture, and adds one
  "further" module fitted on every half-hour before the refit that forecasts the scored month and
  the 62 days after it (7 small lagless fits per screen month, 28 in all). Nothing at or after the
  refit enters any fit.
- **The 30-day hold-back step is dropped.**

### What the correction reads

- **Correction network.** Experiment 472's 32-unit network, penalty 1e-2, on the single-seed
  3-component ERA5 lagless anchor, reading the three origin lags (value and the surprise against the
  anchor median) and two lead features. There are no missing-flag or age inputs because family C is
  dropped. The scripts import experiment 472's and 404's modules and copy only what must change.
- **Folds.** The four screen folds, 2026-06, 2026-01, 2025-10, and 2026-03, each trained on the 24
  months before (one fit per fold, seed `s` trained on injected world `100 + s`). Only the tiny
  correction fits use 3 seeds (they give the minimum detectable effect); the lagless module and
  the encoder are single-seed. No confirmation month is read. July 2026 stays unused. 2025-12 is
  "not yet scored by a rolling-origin refit" and is not used.
- **Scoring rows and target.** Experiment 404's rows (`experiment_404_scoring_rows`). On the old
  worlds the target is the clean actual (`test.y`, the stored reading) on every row, including rows
  whose target lies inside an injected event, because an injected shift is a metering artefact. On
  the new-generator worlds the target is the stored reading plus the family B step in force at the
  target time, because a genuine level change moves the true power; the training target is the same.
- **Row classes.** On the old worlds, experiment 472's `tgt_inj`, `harm_inj`, and `clean_inj`;
  "injected harm and target rows" is the union of the first two. On the new-generator worlds,
  `temp` (the row's window has a family A step), `gen` (family B), and `none`; "step rows" is
  `temp` plus `gen`.

### Arms

| Arm | Lags the correction trains and scores on | Role |
|---|---|---|
| Plain | shifted | Retrained lab reference |
| Oracle-exact | the exact injected shift removed (the stored reading, or on new worlds the stored reading plus the genuine step) | Ceiling |
| Oracle-constant | a constant removed: the step's mean over the history (capacity fraction), so a shape that varies over the day leaves a residual | Prices a simple constant repair |
| Oracle-reprediction | trained shifted, scored on exact lags | Brief arm (a), no retrain |
| Noise curve | exact lags degraded | Nine points, below |
| TV | total-variation-repaired | Non-learned repair |
| Z1e-2 | lags zeroed | Experiment 472's lag-free control |

- **Oracle repair on the old worlds** subtracts the exactly known injected shift, so the oracle lag
  is the stored reading (those worlds are the stored reading plus an additive shift). On the new
  worlds the oracle removes family A and keeps family B. Real events that nobody injected stay in
  every arm's lags.
- **Data path.** `forecast` takes a `lag_world` and a `flag_world` separately; `fit_one` takes a
  preloaded pool so many fits share one data load; on the new worlds lag values come from a per-row
  evaluation of the window's events (the next section). A unit test checks that the changed code,
  fed the old worlds and no repair, reproduces a plain fit.
- **Retrained plain, three seeds.** Its 3-seed spread gives the injected-row minimum detectable
  effect, stated before any contrast is read: 2.8 times the mean over fold pairs of the standard
  deviation of paired seed differences, divided by the square root of 4 (experiment 454's formula
  with 4 folds).
- **Noise curve, on the new-generator worlds** (the event table gives exact onsets and sizes).
  Size errors of 10% and 50%; onset errors of 2 and 24 hours; 10% and 50% of steps missed; the
  joint point (25% size error, 6 hour onset error, 30% steps missed); and the 25%, 6 hour, and 30%
  mid-points on each axis (9 points). Each is the oracle repair degraded, then retrained (4 folds
  by 3 seeds is 12 small fits per point, about 35 seconds each, in 3 parallel processes).

### The injection generator (new code, `e00507_generator.py`)

- **Window-level sampling.** A window is one (Primary, origin) pair; origins are daily at 09:00 UTC.
  Its events are a pure function of (world seed, Primary, origin), so rows that share an origin
  share an injection. 45% of windows get family A, 25% get family B, and 30% get neither. Worlds G100
  to G102 train (one per seed, as experiment 454's advice asks), G200 to G202 are held out with
  disjoint seeds, and G110 is a tuning world.
- **Family A, temporary load shift** (1 to 3 events). Durations: 50% sampled from the labelled
  quantiles of experiment 402's pool, 50% log-uniform from 1 hour to 14 days. Sizes are 0.5 to 3
  times the Primary's lagless residual spread, either sign, capped at 5.4 MW. Onsets run from 14
  days before the history to its last half-hour.
- **Family B, genuine level change** (1 event). The same size, sign, shape, and onset
  distributions as family A, with the onset inside the history and no end.
- **Family D, hard negatives.** Windows where the lagless median moves most (top decile of its
  14-day range), oversampled three times. They receive families A and B at the same rates as every
  other window.
- **Shapes, both families**, in equal thirds: constant MW; proportional to the Primary's own demand;
  and a circuit profile, a fraction (0.3 to 1) of another Primary's demand profile at the same time
  of day, scaled to the drawn size. Of each, 30% ramp over 1 to 4 half-hours and 20% are two-stage
  staircases. No slow ramps. N0's shape table (below) may change the thirds; the change is printed
  and recorded before any score is read.
- **No family C** (bad readings), no reading-class head, and no missing, flagged-zero, or age
  channel: NGED's telemetry is very clean once experiment 399's start cut removes the first months
  of a new meter.
- **At the eight (MVA only):** inject as |MVA + s|; mask half-hours where the anchor's median is
  below twice |s| from the step loss and report their share. No signed claim is made at the eight.
  The old-world generator skips windows below 0.2 of capacity at the eight; the difference is
  stated beside every number at the eight.
- **Terms.** The history is the 672 half-hours ending at the newest usable reading, `origin -
  NEWEST_AGE` (7 half-hours of telemetry delay). A step is "in force at the origin" if it covers
  that last half-hour.
- **Self-test (printed before any total-variation or encoder number is read), with acceptance
  criteria.** It prints the share of windows with a step in force at the origin (within 5 points of
  the value derived in the docstring from the duration mix); the share of family B among them
  (expected about 60%, within 5 points; an arithmetic result of the chosen shares, not a finding);
  and, for events whose onset lies inside the history, a two-sample test (Kolmogorov-Smirnov, level
  0.01) that ramp length, size, sign, and onset hour of day match between families A and B. A
  failing self-test stops the experiment until the generator is fixed. Unit tests cover step
  algebra, determinism, and that no repaired lag reads a half-hour after the newest usable reading.

### N0, before any fit (CPU)

- Leak checks of the final design's point 2.6c: no training-window row of a test month came from a
  lagless network that saw that month, using the `source` column of the stored lagless tables.
- The month-boundary check of the stored tables: whether the switch from `crossfit` to `further`
  at a month start moves the mean residual by more than the residual's usual month-to-month
  spread (a faked level step).
- The number of labelled events and the confirmation months (every other month with a labelled
  event, at least 12 months of history, before July 2026).
- **Shift-shape table and plot for the 34 events with a measured deviation.** For each event, the
  spread of the deviation over the event relative to its mean, and the correlation of the deviation
  with the Primary's own daily demand shape and with a flat profile. The count (about 5 at the six),
  the table, and the plot set the shape thirds. No score is computed.
- The generator's self-test, as above.
- **Actions.** A leak-check failure stops the experiment and is reported to the manager (the
  stored lagless tables would be unusable for the residual). A month-boundary step larger than the
  residual's usual month-to-month spread restricts the total-variation window to one source (or
  masks the boundary) and is reported.
- The hold-back cost step is dropped (no stored table was fitted with a hold-back, and stored
  tables are already out of sample for the history).

### T0: the oracle ceiling (rule (a)), then the rest only if (a) passes

1. Retrain the plain control and the oracle-exact and oracle-constant corrections (4 folds by 3
   seeds each, 36 fits, about 20 minutes), re-predict the plain correction on exact lags, and score all of them on
   worlds 200 to 202. All fits share one data load and one GPU process under 8 GB.
2. Print the injected-row minimum detectable effect. Then read the contrast oracle minus plain.
3. If rule (a) fires, the line ends here at about 0.3 GPU-hours (36 fits and the re-prediction). The page is still written, with
   the oracle-gain interval, the retrained plain control, and the minimum detectable effect, so the
   null is informative.
4. Otherwise run the noise curve (nine points) and build the new-generator worlds.

### T0-TV: the non-learned repair (only if rule (a) passes)

- **Repair.** For each origin, take the 672-half-hour residual (ending at `origin - NEWEST_AGE`) of the reading against the lab lagless
  median and denoise it by total variation (batched solver on the GPU). The normal level is the
  median of the denoised level. Where the denoised level departs from the normal level by more
  than tau spreads (the Primary's residual spread), subtract the departure.
- **Tuning.** The criterion is the mean absolute error of the repaired lag against the oracle
  lag, over rows with a temporary step in force, on tuning world G110. G110 uses only origins in
  months of the training window that are not among the four scored months, and no correction fit
  trains on those origins' injections. The grid is 6 total-variation weights times 5 values of tau
  (30 solves of the tuning set), printed in full. Worlds G200 to G202 are never used for tuning.
- **Scoring.** The 32-unit correction is retrained on repaired lags (4 folds by 3 seeds), with
  Z1e-2 beside it, on the old worlds and on the new-generator worlds with family B. Plain and
  oracle are retrained on the new-generator training worlds too, so the three arms see the same
  generator.

### T1: one encoder fit (only if rule (b) did not fire)

- Stage 0 only: the history encoder H alone on fold 2026-01, one seed, losses L_u (spike-and-slab
  step loss with the non-negative positive-unlabelled estimator on the spike), L_cls, L_g, and L_tv
  of the final design's section 2.4. About 100,000 parameters. No stage 1, no R1 or later rung.
- Read onset within 2 hours, size error, share of steps missed, and the planted-step precision test
  (held-out injected steps planted at 2% of windows into real windows) on worlds G200 to G202, the
  same worlds as the noise curve, and place the three errors on the noise curve.
- **Timing first.** One epoch of one encoder fit is timed and the total is reported to the manager
  before T1 runs. Abort and message the manager and the neural Discuss session if one fit
  extrapolates above 2 GPU-hours. Budget: T0 0.3, noise curve 0.6 (108 fits across 3 parallel
  processes), new-generator and TV retrains 0.5 (about 48 fits), T1 at most 1.6, in GPU-hours of
  wall-clock, so at most 3. The total-variation solves and the tuning grid are CPU work and are
  timed and reported.

## Decision rule

All scores are CRPS in points of effective capacity (the `pinball_points` scorer of experiment
472), lower is better, at the signed-MW six, on injected harm and target rows (the union of
`harm_inj` and `tgt_inj`, with its own seed-spread minimum detectable effect), with a month-block
interval over the 4 folds (3 seeds averaged in each fold; the interval is the exact enumeration of
the 256 resamples of 4 blocks, with the four monthly contrasts also printed). Differences are
written "arm minus plain"; gain is plain minus arm.

1. **Rule (a), stop the whole line.** The oracle gain is below the injected-row minimum detectable
   effect, or its interval includes zero, on the old worlds. The new-generator worlds are scored
   too. If the old worlds pass and the new-generator worlds do not, the line continues to T0-TV and
   the page reports both.
2. **Rule (b), promote the non-learned repair.** Test the difference (TV gain minus half the oracle
   gain) with the paired interval: it passes when the lower end is at or above zero, on the old
   worlds and on the new-generator worlds pooled at the generator's own mix (about 60% genuine
   among steps in force). The ratio TV gain over oracle gain is reported only when the oracle
   gain's lower end exceeds the minimum detectable effect, because the ratio is unstable otherwise.
   The genuine-only difference is reported beside. The clean-row clause is a point estimate of TV
   minus plain of at most 0.03 points on rows with no labelled event at the six, on the
   real world (-1), reported with its interval and labelled "not shown worse", and the q99 clause is
   a point estimate of at most +0.02 points on the single-level q99 pinball loss (formula and
   the plain correction's baseline printed, with the absolute and relative margin). The brief's "within 0.03" is a point
   estimate, so the plan keeps it one; both limits sit far below the clean-row minimum detectable
   effect of 0.237 and prove no equivalence. If the point estimate passes and the interval does
   not, the page says "point estimate passes, not shown", and T1 still runs.
3. **Rule (c), stop the neural encoder.** T1's onset error (reported as a distribution in hours,
   from which the shares within 2, 6, and 24 hours follow), size error, and share of steps missed
   are placed on the noise curve of the same new-generator worlds G200 to G202. The noise curve is
   measured on those worlds only (their event table gives the exact onsets and sizes). The curve's
   axis values are interpolated linearly in the error, and "under half the oracle gain" needs the
   curve's point to lie below half the oracle gain by more than the seed spread. The worst of the
   three axes decides. The rule also fires if validation step loss stops falling before epoch 10.
   T1 is one fold and one seed, so the page calls the result a single-fit reading.
4. **Contrasts and levels.** Four decision contrasts at the six: C1 oracle minus plain (old
   worlds), C2 TV against half the oracle gain (old worlds), C3 the same on the new-generator
   worlds, C4 TV minus plain on rows with no labelled event. The same four at the eight, which
   are report-only: a result at the eight that contradicts the six is flagged and does not change
   the rule. Bonferroni count 8, interval level 1 - 0.05/8, stated as numbers, with the 95% level
   beside it.
5. **Not units-free.** The signed-MW six and the MVA-only eight are never pooled. D15 and D18 are
   trained after the 2025-03-01 cut and are report-only.

## Scores, caveats, and what this cannot show

- **Scores.** CRPS and absolute scores beside every contrast, tail calibration (q01, q05, q95, q99
  coverage and pinball loss), the 12 to 36 hour band, the share of an unmirrored 5% step followed
  at k = 3, rows with no labelled event, equal rows across arms. The write-up includes at least two
  example forecasts and the load-shift plots required by the common brief where scoring allows.
- **What this cannot show.** The test reads injected worlds. A pass shows headroom, not that real
  shifts behave like injected ones. Inside real events the minimum detectable effect is 0.38 at
  the six (experiment 454). NGED's event labels are incomplete, so "rows with no labelled event"
  may hold real events nobody logged.
- **Serving cost of a repair.** A repair needs a 672-half-hour history store per Primary and, for
  total variation, one convex solve per Primary per forecast (well under a millisecond), plus the
  anchor's lagless forecasts for the history half-hours (one lagless seed). The encoder adds one
  network at serving.
- **Anonymisation.** Demand series only: Primaries may be named (D codes), no generator series
  appear.

## Reuse and advice followed

- **Stored forecasts reused:** none as contrasts. Experiment 449's cross-fit code is reused to
  rebuild the ERA5 lab tables. Every correction fit is new because its training lags change.
- **Advice followed:** give each fit seed its own injected training world (experiment 454); judge a
  correction on injected held-out shifts and real harm rows together (experiment 472); include the
  lag-zeroed control Z1e-2 beside a corrected arm (experiment 472).
- **Cadence:** one fit per screen fold (24-month window), the screen design of the common brief.
  A winner needs the full evaluation before any release claim.
