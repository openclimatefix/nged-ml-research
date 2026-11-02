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

- **Oracle repair** subtracts the exactly known injected shift from each lag, so the oracle lag is
  the stored reading at that half-hour (the injected worlds are the stored reading plus an
  additive shift). Real events that nobody injected stay in every arm's lags, and the page says so.
- **Retrained plain, three seeds.** The retrained plain control uses seeds 0 to 2 and the stored
  recipe, so it should reproduce the stored AvgAnchor; the script prints the difference. The
  3-seed spread of its scores gives the injected-row minimum detectable effect, stated before any
  contrast is read: 2.8 times the mean over fold pairs of the standard deviation of paired seed
  differences, divided by the square root of 4 (experiment 454's formula, with 4 folds).
- **Noise curve.** Size errors of 10%, 25%, and 50%; onset errors of 2, 6, and 24 hours; 10%, 30%,
  and 50% of steps missed; plus one joint point (25% size error, 6 hour onset error, 30% steps
  missed). Each is the oracle repair degraded, then retrained (3 seeds by 4 folds is 12 small fits
  per point, about 35 seconds each).

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
- **Window-level injection with deterministic seeds.** A window is one (Primary, origin). Its
  injection is a pure function of (world seed, Primary, origin), so rows that share an origin
  share an injection and any worker regenerates it. Worlds G100 to G102 train; worlds G200 to G202
  are held out and use a seed range disjoint from every training world.
- **Self-test (printed before any total-variation or encoder number is read):** the share of
  windows with a step in force at the origin; the share of family B among those windows (expected
  about 60%, an arithmetic result of the chosen shares, not a finding); and a two-sample test that
  ramp length, size, sign, and onset hour have the same distribution in families A and B. Unit
  tests cover step algebra, conservation of the corrupted-reading definitions, and the seeding.

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
- The hold-back cost step is dropped (no stored table was fitted with a hold-back, and stored
  tables are already out of sample for the history).

### T0: the oracle ceiling (rule (a)), then the rest only if (a) passes

1. Retrain the plain control and the oracle correction (4 folds by 3 seeds each, 24 fits, about 15
   minutes), re-predict the stored AvgAnchor correction on exact lags, and score all of them on
   worlds 200 to 202. All fits share one data load and one GPU process under 8 GB.
2. Print the injected-row minimum detectable effect. Then read the contrast oracle minus plain.
3. If rule (a) fires, the line ends here at about 0.3 GPU-hours. The page is still written, with
   the oracle-gain interval, the retrained plain control, and the minimum detectable effect, so the
   null is informative.
4. Otherwise run the noise curve (ten points) and build the new-generator worlds.

### T0-TV: the non-learned repair (only if rule (a) passes)

- **Repair.** For each window, take the 672-half-hour residual of the reading against LL-ens3's
  median and denoise it by total variation (batched solver on the GPU). The normal level is the
  median of the denoised level. Where the denoised level departs from the normal level by more
  than tau spreads (the Primary's residual spread), subtract the departure. Zero, stuck, and
  missing readings are set to missing by the rules of experiments 435 and 410 before the solve.
- **Tuning.** The total-variation weight and tau are chosen on new-generator worlds from the screen
  folds that are not worlds G200 to G202 (worlds G100 to G102 held-out halves by Primary and
  month), never on the scoring worlds.
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
  extrapolates above 2 GPU-hours. Total cost at most 3 GPU-hours.

## Decision rule

All scores are CRPS in points of effective capacity at the signed-MW six, lower is better, on
injected harm and target rows, with a month-block interval over the 4 folds (3 seeds averaged in
each fold). Differences are written "arm minus plain".

1. **Rule (a), stop the whole line.** The oracle minus plain contrast is not below the negative of
   the injected-row minimum detectable effect, or its 95% interval includes zero.
2. **Rule (b), promote the non-learned repair.** Let gain be plain minus arm. The ratio (TV gain)
   over (oracle gain), on the same rows, has a 95% interval (month-block bootstrap of the ratio of
   means) whose lower end is at least 0.5, both on the old worlds and on the new-generator worlds
   pooled at the generator's own mix (about 60% genuine among steps in force). The genuine-only
   ratio is reported beside it. The repair is also not shown worse than plain AvgAnchor by more
   than 0.03 points of capacity on rows with no labelled event at the six (the upper end of the
   interval for TV minus AvgAnchor is at most 0.03), and not shown worse by more than 0.02 points
   on q99 pinball loss. "Not shown worse" is not equivalence; both limits sit well below the
   clean-row minimum detectable effect of 0.237. If the point ratio is at least 0.5 and its lower
   end is below 0.5, the page says "point estimate passes, not shown", and T1 still runs.
3. **Rule (c), stop the neural encoder.** The encoder's three errors land where the noise curve
   keeps under half the oracle gain, or its validation step loss stops falling before epoch 10.
   When the three errors land in different regions, the worst region decides.

The Bonferroni count is 8 (4 decision contrasts at the six, 4 at the eight), stated as a number.
The signed-MW six and the MVA-only eight are scored separately and never pooled. D15 and D18 are
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
