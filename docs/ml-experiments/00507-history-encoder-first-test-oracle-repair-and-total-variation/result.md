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
  (AvgAnchor) on rows where a shift is in force. A non-learned repair (total-variation denoising of
  the 14-day residual) keeps less than half of that gain once genuine level changes are in the
  data, and a learned encoder can reach the accuracy that the oracle's noise curve demands.
prediction: >-
  Written before any fit. Scores are CRPS in points of effective capacity at the signed-MW six on
  held-out injected worlds. P1. The retrained oracle correction beats the retrained plain
  AvgAnchor on injected harm and target rows by at least 0.3 points (the brief's prediction; the
  stop threshold of rule (a) is the injected-row minimum detectable effect, which the plan expects
  to be near 0.05 to 0.15). P2. Total-variation repair keeps under half of that gain on the
  new-generator worlds, because it erases genuine level changes (60% of the steps in force at the
  origin). P3. The encoder's onset error, size error, and share of steps missed fall in the part of
  the noise curve that keeps at least half of the oracle gain. Any of the three can fail.
decision_rule: >-
  Fixed in the section "Decision rule", below, before any score is read. In order: (a) stop the
  whole line if the retrained oracle correction does not beat the retrained plain AvgAnchor on
  injected harm and target rows by more than the injected-row minimum detectable effect (from the
  3-seed spread of the retrained plain control) with an interval excluding zero; (b) promote the
  non-learned repair and stop the neural encoder only if total-variation repair keeps at least half
  of the oracle gain on the old worlds and on the new-generator worlds with genuine changes (lower
  end of the interval on the ratio at or above 0.5), is not shown worse than AvgAnchor by more than
  0.03 points on rows with no labelled event at the six, and is not shown worse by more than 0.02
  points on q99 pinball loss; (c) stop the neural encoder if the encoder's errors fall where the
  noise curve keeps under half of the oracle gain, or its validation step loss stops falling
  before epoch 10. The smallest gain worth porting is the injected-row minimum detectable effect,
  because the repair adds a convex solve per Primary and a 672-half-hour history store.
benchmark: >-
  Retrained plain AvgAnchor (experiment 472's recipe, seeds 0 to 2, the same fit code), and the
  stored AvgAnchor forecasts of experiment 472 as a reproduction check. Stored LL-ens3 for the
  anchor. K2-S1 is not scored on the held-out injected worlds because it has no forecasts on
  worlds 200 to 202.
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

### What the correction reads

- **Correction network.** Experiment 472's AvgAnchor: a 32-unit network on the 9-component anchor
  (the equal-weight mixture of lagless seeds 0 to 2), reading the three origin lags (value,
  missing flag, age), the surprise of each lag against the anchor's median, and two lead features.
  Penalty 1e-2. The scripts import experiment 472's `e00472_arm.py` and copy only what must change.
- **Folds.** The four screen folds, 2026-06, 2026-01, 2025-10, and 2026-03, each trained on the 24
  months before (experiment 472's recipe, 24-month window, one fit per fold, seed `s` = network
  seed, trained on injected world `100 + s`). No confirmation month is read. July 2026 stays
  unused. 2025-12 is "not yet scored by a rolling-origin refit" and is not used here.
- **Scoring rows and target.** Experiment 404's rows (`experiment_404_scoring_rows`). On the old
  worlds the target is the clean actual (`test.y`, the stored reading) on every row, including the
  rows whose target half-hour lies inside an injected event, because an injected shift is a
  metering artefact and the true power did not change. On the new-generator worlds the target is the
  clean actual plus the family B step in force at the target time, because a genuine level change
  moves the true power. Both statements are printed by the scoring script.
- **Row classes.** Experiment 472's classes: `tgt_inj` (target inside an injected event),
  `harm_inj` (target outside, one of the three day windows inside), and `clean_inj`. "Injected
  harm and target rows" is the union of the first two.

### Arms

| Arm | Lags the correction trains on | Lags it reads at scoring | Role |
|---|---|---|---|
| Plain | shifted (world `100 + s`) | shifted (worlds 200 to 202) | Retrained control, same recipe as the stored AvgAnchor |
| Oracle | exact (world -1, the stored readings) | exact | Ceiling |
| Oracle-reprediction | shifted | exact | The stored AvgAnchor correction re-predicted on exact lags (brief arm (a)) |
| Noise curve | exact lags degraded | degraded | Ten points, below |
| TV | total-variation-repaired | total-variation-repaired | Non-learned repair |
| Z1e-2 | lags zeroed | lags zeroed | Experiment 472's lag-free control, for the same recipe |

- **Oracle repair, per family.** On the old worlds (injected load shifts and spikes, codes 1 to 3)
  the oracle lag is the stored reading (world -1), because those worlds are the stored reading
  plus an additive shift. On the new-generator worlds the oracle removes families A and C (the
  temporary shifts and the corruptions, restoring dropouts, stuck readings, delays, and missing
  runs to the stored reading) and keeps family B, so the oracle lag is the stored reading plus the
  genuine step in force at the lag time. Real events that nobody injected stay in every arm's lags.
- **Data path.** `forecast` takes a `lag_world` and a `flag_world` separately, so the oracle arms
  read lags from the repaired source and row-class flags from worlds 200 to 202. `fit_one` takes a
  preloaded pool, so many fits share one data load. A unit test checks that the changed code,
  given the old worlds and no repair, reproduces the stored AvgAnchor forecasts.
- **Training target.** On the new-generator training worlds the target is the stored reading plus
  the family B step at the target time (`target_power`), for every arm, as it is for scoring. On
  the old worlds it is the stored reading.
- **Retrained plain, three seeds.** The retrained plain control uses seeds 0 to 2 and the stored
  recipe, so it should reproduce the stored AvgAnchor; the script prints the difference. The
  3-seed spread of its scores gives the injected-row minimum detectable effect, stated before any
  contrast is read: 2.8 times the mean over fold pairs of the standard deviation of paired seed
  differences, divided by the square root of 4 (experiment 454's formula, with 4 folds).
- **Noise curve, on the new-generator worlds.** Size errors of 10% and 50%; onset errors of 2 and 24
  hours; 10% and 50% of steps missed; plus the joint point (25% size error, 6 hour onset error,
  30% steps missed) that brief edit 4 asks for, and the 25%, 6 hour, and 30% mid-points on each
  axis (9 points in all). Each is the oracle repair degraded using the event table, then
  retrained (4 folds by 3 seeds is 12 small fits per point, about 35 seconds each, run in 3 parallel
  processes). Degrading by event table means the old worlds are not used for the curve.

### The injection generator (new code, `e00507_generator.py`)

- **Family A, temporary load shift** (45% of windows, 1 to 3 events). Durations: 50% sampled from
  the labelled quantiles of experiment 402's pool, 50% log-uniform from 1 hour to 14 days. Sizes
  are 0.5 to 3 times the Primary's lagless residual spread, either sign, capped at 5.4 MW. Onsets
  run from 14 days before the window to its last half-hour.
- **Family B, genuine level change** (25% of windows). The same size, sign, shape, and onset
  distributions as family A, with onsets inside the window and no end (the step persists through
  the horizon of 480 half-hours).
- **Shapes, both families.** Constant MW or proportional to demand in equal shares. Of each, 30%
  ramp over 1 to 4 half-hours and 20% are two-stage staircases. No slow ramps.
- **Family C, corrupted readings** (30% of windows, independent): dropout to zero, stuck readings,
  spikes, scale glitches, missing runs, and telemetry delays of 0, 12, 24, or 48 half-hours, by
  the definitions of experiments 465, 486, and 410.
- **Family D, hard negatives:** windows where the lagless median moves most (top decile of its
  14-day range), oversampled three times. They receive families A to C at the same rates as every
  other window.
- **At the eight (MVA only):** inject as |MVA + s|; mask the half-hours where the anchor's median is
  below twice |s| from the step loss and report their share. No signed claim is made at the eight.
  The old-world generator (experiment 402) skips windows below 0.2 of capacity at the eight; the
  difference is stated beside every number at the eight.
- **Grid-level worlds.** The generator writes worlds in the format of experiment 428
  (`power[primary, grid]` and `code[primary, grid]`), so the reused fit and scoring code reads them
  unchanged. Events are placed on the dense half-hourly grid per Primary as a Poisson process whose
  rates are set so that the shares above hold per origin. It also writes `target_power` (the stored
  reading plus the family B steps and nothing else), `step_temp` and `step_gen` (the exact step in
  force, in capacity fractions, per half-hour), and an event table (Primary, family, onset, end,
  size, shape). Worlds G100 to G102 train (one per seed, as in experiment 454's advice); G200 to
  G202 are held out, with seeds disjoint from every training world; G110 is a tuning world.
- **Terms.** The history is the 672 half-hours ending at the newest usable reading, which is
  `origin - NEWEST_AGE` (7 half-hours of telemetry delay, as in `e00087_common`). A step is "in
  force at the origin" if the history's last half-hour lies inside its span.
- **Self-test (printed before any total-variation or encoder number is read), with acceptance
  criteria.** It prints the share of origins with a step in force (target 60% to 80% of origins at
  which family A or B places any step, within 5 points of the value derived in the script's
  docstring from the duration mix); the share of family B among them (expected 60%, within 5
  points; this is arithmetic on the chosen shares and not a finding); and, for events whose onset
  lies inside the history, a two-sample test (Kolmogorov-Smirnov, level 0.01) that ramp length,
  size, sign, and onset hour of day match between families A and B. A failing self-test stops the
  experiment until the generator is fixed. Unit tests cover step algebra, the corrupted-reading
  definitions, determinism, and that a repaired lag never reads a half-hour after the newest usable
  reading.

### N0, before any fit (CPU)

- Leak checks of the final design's point 2.6c: no training-window row of a test month came from a
  lagless network that saw that month, using the `source` column of the stored lagless tables.
- The month-boundary check of the stored tables: whether the switch from `crossfit` to `further`
  at a month start moves the mean residual by more than the residual's usual month-to-month
  spread (a faked level step).
- The number of labelled events, the confirmation months (every other month with a labelled event,
  at least 12 months of history, before July 2026), and the demoted real-event reading: the count
  of the 34 events with a measured deviation (about 5 at the six) and one plot, no score.
- The generator's self-test, as above.
- **Actions.** A leak-check failure stops the experiment and is reported to the manager (the
  stored lagless tables would be unusable for the residual). A month-boundary step larger than the
  residual's usual month-to-month spread restricts the total-variation window to one source (or
  masks the boundary) and is reported.
- The hold-back cost step is dropped (no stored table was fitted with a hold-back, and stored
  tables are already out of sample for the history).

### T0: the oracle ceiling (rule (a)), then the rest only if (a) passes

1. Retrain the plain control and the oracle correction (4 folds by 3 seeds each, 24 fits, about 15
   minutes), re-predict the stored AvgAnchor correction on exact lags, and score all of them on
   worlds 200 to 202. All fits share one data load and one GPU process under 8 GB.
2. Print the injected-row minimum detectable effect. Then read the contrast oracle minus plain.
3. If rule (a) fires, the line ends here at about 0.3 GPU-hours (24 fits and the re-prediction). The page is still written, with
   the oracle-gain interval, the retrained plain control, and the minimum detectable effect, so the
   null is informative.
4. Otherwise run the noise curve (ten points) and build the new-generator worlds.

### T0-TV: the non-learned repair (only if rule (a) passes)

- **Repair.** For each origin, take the 672-half-hour residual (ending at `origin - NEWEST_AGE`) of the reading against LL-ens3's
  median and denoise it by total variation (batched solver on the GPU). The normal level is the
  median of the denoised level. Where the denoised level departs from the normal level by more
  than tau spreads (the Primary's residual spread), subtract the departure. Zero, stuck, and
  missing readings are set to missing by the rules of experiments 435 and 410 before the solve.
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
   minus plain AvgAnchor of at most 0.03 points on rows with no labelled event at the six, on the
   real world (-1), reported with its interval and labelled "not shown worse", and the q99 clause is
   a point estimate of at most +0.02 points on the single-level q99 pinball loss (formula and
   AvgAnchor's baseline printed, with the absolute and relative margin). The brief's "within 0.03" is a point
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
   worlds, C4 TV minus AvgAnchor on rows with no labelled event. The same four at the eight, which
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
  anchor's lagless forecasts for the history half-hours (3 lagless seeds). The encoder adds one
  network at serving.
- **Anonymisation.** Demand series only: Primaries may be named (D codes), no generator series
  appear.

## Reuse and advice followed

- **Stored forecasts reused:** experiment 472's AvgAnchor and experiment 454's LL-ens3 tables (as
  references and for the reproduction check). Every correction fit is new because its training
  lags change.
- **Advice followed:** give each fit seed its own injected training world (experiment 454); judge a
  correction on injected held-out shifts and real harm rows together (experiment 472); include the
  lag-zeroed control Z1e-2 beside a corrected arm (experiment 472).
- **Cadence:** one fit per screen fold (24-month window), the screen design of the common brief.
  A winner needs the full evaluation before any release claim.
