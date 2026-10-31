---
id: "00446"
slug: honest-bands-and-tail-calibration-on-the-s1-network
title: >-
  Do scale-only bands trained on ambiguous jumps, and tail-calibrating changes, beat a recalibration
  table on the S1 network?
theme: calibration-and-uncertainty
also_relevant_to:
  - lagged-power-and-error-correction
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
date: 2026-10-31
size: >-
  complex (Brief B and Brief C in one batch). Stored: 128 fitted neural networks, plus 12 for U1n
  with
  trimmed durations at 2% (24 with the 5% rate), 12 for U4 if its stop rule did not fire, and up to
  36 for 5% U2, U2i, and U2c only if U1n at 5% matches held-out coverage better, their forecasts,
  and sweep tables in this experiment's private folder only;
  experiment 428's, 439's, and 404's folders are read and never written. More than one design would
  test the hypothesis (how the training worlds are built, how the scale path is wired, which tail
  change), so each choice is fixed in this plan before any score is read. Production serving path:
  not touched (this repository holds none). Degradation rule: not touched. Callers: the scripts
  import the folders of experiments 428, 404, 441, and 392 and the `studies` evaluation code.
hypothesis: >-
  The forecast targets the normal running arrangement (NRA), so a corrupted recent-power reading
  removes information and never moves the target. On the S1 network (three lagged power readings
  plus the surprise at each), (B) a scale-only path trained on worlds in which some jumps are
  genuine NRA level changes widens the bands where a jump is ambiguous, without widening them on
  clean rows, and beats a stratified recalibration table (table P) laid over the unmodified
  network; and (C) a fitted change to the tails (Student-t components, a pinball term in the loss,
  a tail fine-tune, or a temperature-dependent tail weight) brings clean-row tail coverage into
  tolerance, on cold winter nights too, better than table P alone.
prediction: >-
  Written before any fit and before any score is read. Brief B. P1. U2-clean (the scale-only path
  trained on clean-target worlds) keeps a band-width ratio within 0.02 of 1.00 on injected harm
  rows. P2. U2 raises the band-width ratio at a jump's first day by at least 0.05 and keeps it
  under 1.03 after 3 days, read on the held-out world with injected genuine level changes (defined
  in the design choices below). P3. U2i adds under 0.01 of q99 coverage beyond U2. P4. The mean
  scale factor on clean rows stays at or under 1.03 for U2, U2i, and U2c. Brief C (author's
  guesses, each about one chance in three of holding). P5. At the signed-MW six, at least one of
  T1, T2, and T3 lifts clean-row q95 coverage from S1's 0.920 to 0.93 or more (the tolerance),
  without raising clean-row CRPS by more than 0.03. P6. T5 (the temperature-dependent tail weight)
  is the only tail arm among T1 to T5 (T4 included) that lifts q95 coverage on cold winter nights
  by 0.05 or more relative to the base with P (read against the raw base until P exists). P7. No
  tail arm brings q95 and q99 coverage on cold winter nights to 0.92 and 0.975 in every fold.
decision_rule: >-
  Four folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 2, monthly refits, as in
  experiments 428 and 439. Every number is interim and carries no 99% claim. Each arm is scored raw
  and at the 1.15 widening; the rules that read table P are applied when experiment 435 delivers P,
  from the stored forecasts. Brief B arm passes if, after P, (a) it meets the conditional-coverage
  tolerances of experiment 435 on every class where R-inj with P fails; (b) clean-row CRPS is at
  most 0.03 points and tail pinball at most 0.02 points above R-inj with P, at both groups; (c) on
  classes where R-inj with P already passes, its 90% band-width ratio is at most 1.05; and (d) on
  real harm rows and real event rows (against NRA-corrected actuals), q95 coverage is not lower
  than R-inj's with P (event-cluster interval, with and without the largest event). Rule (d) can
  veto and cannot pass an arm alone. (e) On cold winter nights, q95 coverage is at least 0.92 and
  q99 at least 0.975, and neither is worse than R-inj's. An arm whose q99 pinball on real harm rows
  exceeds R-inj's by more than 0.02 is reported as failing that rule; rule (d) can veto it and
  cannot pass it alone. A scale factor whose mean on clean rows exceeds 1.03 counts as
  widening everywhere and is reported as failing (b)'s intent. If no arm passes, P is the mechanism.
  Brief C arm passes if, after P refitted on its own validation months, per metering group at days 1
  to 10 and in the 12 to 36 hour band, q95 coverage lies in 0.93 to 0.97, q99 in 0.98 to 0.995, q05
  in 0.03 to 0.07, and q01 in 0.005 to 0.02; it meets more of those cells than its base with P;
  clean-row CRPS is at most 0.03 points and lower-tail pinball at most 0.02 points above its base
  with P; no Primary of the six or the eight has q95 coverage under 0.90 (D14, D15, and D18
  reported separately); and on cold winter nights q95 coverage is at least 0.92 and q99 at least
  0.975 per group, with CRPS on all other rows within 0.02 points and clean-row tail pinball within
  0.02 points of the base. A method that fixes the overall tails and leaves cold nights below
  target does not pass. The tail limit of +0.02 on clean rows is measured against S1 as well as
  against the base. An arm whose q99 pinball on real harm rows exceeds R-inj's by more than 0.02 is
  reported as failing that rule; rule (d) can veto it and cannot pass an arm alone. The smallest
  gain worth porting is the pass rule itself for T2 and T5, which
  change only the loss and cost nothing to port; U1n, U2, U2i, and U2c add a training-world
  generator and a scale path to the training loop, and T1 and T3 add a component family and a
  fine-tune stage, so each of those must pass the rule in full. Intervals are month-and-day block
  bootstrap intervals, and event-cluster intervals for real events. The minimum detectable effect
  is experiment 428's (1.96 times the square root of 2, times the seed standard deviation of the
  reference, divided by the square root of 3), computed before any contrast is read.
benchmark: >-
  R-inj, S1, and B-mix (the fixed 50/50 mixture of the R-inj and L0 distributions, not an average
  of quantiles; no fit) from experiment 428's and 439's stored forecasts, with L0 and R as
  secondary references, all raw, at 1.15, and (when experiment 435 delivers P) after P.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Coverage and pinball loss at q01, q05, q95, and q99 for each arm, group, lead band, row set, and
  the cold-winter-night slice; the continuous ranked probability score from the 11 quantiles; the
  band-width ratio (width of the 5% to 95% band on the row set over its width on clean rows); the
  mean scale factor on clean rows; the avoidable-curtailment cost; the encroachment sweep of
  experiment 392 with its masked-lag control. Raw and at the 1.15 widening.
code: ml-experiments/00446-honest-bands-and-tail-calibration-on-the-s1-network/
---

**Status: planned. No code has run and no score has been read.** The verdict is not yet run.

## Question

**Two questions share one batch of fits, because both ask whether a trained change can do what a
lookup table of widening factors might do alone.** The demand forecast is a neural network that
returns 11 quantiles. Its upper bands are too narrow in two places. First, when the recent power
readings it was given are untrusted (they may sit inside a load-shift event, or be missing or
stale), the bands do not widen, because a network fitted to a clean target has no reason to widen
(the argument is in the design note of the uncertainty programme). Second, on clean rows the upper
tail covers too little: S1's q95 and q99 cover 0.920 and 0.985 of clean readings at the signed-MW
six (experiment 439), and on cold winter nights the pooled plain network's cover only 0.76 and 0.83
(experiment 441).

**Experiment 435 is fitting table P, which widens the bands by a factor per class of input
condition, from stored forecasts and with no new fit.** Table P does not exist yet. This experiment
asks whether a fitted change does better than P laid over the same network: Brief B changes the
training worlds and adds a path that can only widen (arms U1n, U2, U2i, U2c); Brief C changes the
tails on clean rows (arms T1 to T5). Both ride on S1, the network that experiment 439 confirmed.
"Better than P" cannot be decided until P exists, so the plan scores every arm raw and at 1.15 now,
and applies the "after P" rules to the stored forecasts when experiment 435 delivers P.

**Definitions used throughout.** The signed-MW six are the Primary substations metered in signed
megawatts, and the MVA-only eight are metered in unsigned megavolt-amperes; D15 and D18 are
reported separately. A clean row has recent power outside every logged load-shift event. A harm row
has a target outside every event but recent power inside one (injected harm rows hold simulated
events added to clean power; real harm rows hold logged ones). A real event row has its target
inside a logged event. A cold winter night is defined as in experiment 441: solar elevation below
0, local month November to March, and ENS-mean 2 m temperature at or below the Primary's 10th
percentile of winter-night ENS temperatures in the 730 days before the fold's refit.

**The cold-winter-night slice is read on fewer folds than every fold.** Experiment 441's mask table
has cold-winter-night rows only for 2025-12, 2026-01, and 2026-03. It has none for 2025-10 or
2026-06, and 2025-12 is not a test fold of experiment 428. Every rule that reads the
cold-winter-night slice is therefore judged on the folds that have rows, not on all folds.

## Decisions already made

- **One experiment, 446, hosts Brief B and Brief C as one batch.** It runs in experiment 428's
  harness, with module-global redirection of the output folders as experiments 438 to 440 did.
  Experiment 428's folders are read and never written. The base network is S1.
- **Shared reference arms and rows.** R-inj, S1, and B-mix are read from experiment 428's stored
  forecasts (seeds 0 to 2) and 439's (seeds 3 to 5, used only where an arm needs extra seeds). All
  new arms share the held-out worlds (world 0 is the real clean series; worlds 4 to 6 are
  experiment 428's injected ones) and the row sets of experiment 428.
- **Order.** Brief B is fitted and scored first. Brief C's base is the Brief B arm with the lowest
  raw clean-row CRPS among arms that meet the 1.03 scale limit and the coverage tolerances on raw
  and 1.15 forecasts; otherwise S1. The brief says R-inj; the plan uses S1 because S1 is the
  confirmed base (experiment 439). Brief C therefore starts after Brief B is scored.
- **Cadence.** Monthly refits on the 24 months before each test month (the research default), four
  test months, three seeds. A candidate that passes is re-checked on more folds before any release
  claim, and the release claim needs the 12 folds of experiment 425.
- **Wind.** The per-wind-speed-band copy of T2 runs only if experiment 435 shows P cannot fix the
  wind upper tail. It is not in this experiment, so the wind rule that
  flags 1.5% of rows above q99 is not applied here either.

## Arms and fit counts

**One fit is one network for one arm, one test month, and one seed (4 folds x 3 seeds = 12 fits per
arm).**

| Brief | Arm | What changes from S1 | Fits |
|---|---|---|---|
| B | U1n at 2% | S1 trained on worlds that add genuine NRA level changes to lags and target together, at 2% of training rows | 12 |
| B | U1n at 5% | The same worlds at 5% of training rows | 12 |
| B | U2 | U1n at 2% plus a scale-only path (factor 1 + softplus(g) on every component's standard deviation) | 12 |
| B | U2i | U2 plus the 90% and 98% interval score on corrupted rows | 12 |
| B | U2c | U2 plus the coverage-matching penalty (improper; labelled) | 12 |
| B | U2-clean | The scale-only path trained on clean-target worlds, as experiment 428 planned (read from experiment 428 if already fitted; otherwise fit here) | 12 (0 if read) |
| B | U4 (conditional) | Experiment 404's B1, normal lagged head, trained on U1n worlds | 12 |
| C | T1 | Student-t mixture components, degrees of freedom learned and at least 2.5 | 12 |
| C | T2 | Negative log-likelihood plus 0.2 times the pinball loss at q01, q05, q95, and q99 | 12 |
| C | T3 | Trunk frozen, head fine-tuned for a few epochs on the four tail pinball losses | 12 |
| C | T4 | 2 extra seeds of the base, 5 mixtures pooled | 8 |
| C | T5 | T2 with a tail weight that depends on the ENS-mean temperature | 12 |

**Totals.** Brief B has 72 fits (6 arms of 12), Brief C has 56, so the batch is 128 fits, plus 12
for U1n with trimmed durations at 2% (24 with the 5% rate), 12 for U4 if experiment 404's stop rule
did not fire, and up to 36 for 5% U2, U2i, and U2c only if U1n at 5% matches held-out coverage
better. U3 (20 masked passes with a mask probability that rises with the lag's surprise) is
prediction only and has no fit; it is run if U1n's token dropout does not match the masking, which
means U3's 90% band-width ratio on injected harm rows is not within 0.02 of U1n-dropout's.

**The seeds, the lagless tables, and the cost.** Seeds 0 to 2 use experiment 404's frozen
cross-fitted lagless tables. S1-sized fits took about 30 seconds each in experiment 439 (24 fits in
about 11 minutes), and the briefs allow 1 to 3 minutes. The batch is therefore about 1 to 6 GPU
hours, in one GPU process under 8 GB plus one feeder core, and Brief B comes first.

**Design choices fixed before any score is read.**

- **U1n's worlds.** Experiment 428's three injected worlds (seeds 100 to 102) plus genuine NRA level
  changes, in which depth and duration are drawn from the ranges in the label table and the shift
  enters the lags and the target together. Token dropout 0.2 on lags and spread, and 8 origin hours.
  The 2% rate runs first. The scale-only arms (U2, U2i, U2c) train on the 2% worlds; the 5% rate is
  a sweep of U1n only, and a 5% rate for U2 is run only if the 5% U1n arm is closer to the held-out
  world's coverage than the 2% arm. The held-out world is a world with injected genuine level
  changes, built the same way as U1n's training worlds but with different draws, scored on
  clean-target rows for the load-shift worlds. It is new in this experiment and held-out for all
  arms; P2 and the "closer to the held-out world's coverage" rule read it. The write-up states that
  the true rate of genuine level changes is unmeasured.
- **Duration.** The load-shift session reported a median event duration near 48 hours and about
  10% of events above 7 days (unverified). The code checks both figures against the corrected label
  table
  (`sw_block_detector_label_table_corrected.csv`; the sign was wrong for 51 of 112 rows before the
  correction) before the first fit. For U1n only, the duration distribution is swept (the label
  table's own, and a version with the upper tail trimmed) beside the rate.
- **U2's scale path.** The surprise magnitude, lag ages, missing flags, and lead feed a factor
  1 + softplus(g), initialised at 1, with a penalty on the mean log factor on clean rows. The
  stored inputs hold no weather-age column, so the scale path has 11 inputs: 3 absolute surprises,
  3 lag ages, 3 missing flags, and the 2 lead columns (`lead` and `lead_sqrt`, in place). The scale
  path adds its own inputs; the location path is S1's network, unchanged (it takes the signed
  surprise directly), and receives no gradient from the scale-only loss terms. U2i and U2c compute
  their interval-score and coverage-penalty terms with the mixture weights, means, and base spread
  detached, so only the factor g receives gradient from them.
- **U2-clean.** This arm is experiment 428's planned scale-only arm: the U2 network on worlds
  without U1n's genuine level changes. Running it beside U2 makes the training worlds the only
  difference. Experiment 428's earlier widening path (E3-surp-w) learned no widening on the clean
  target; U2-clean is the control that tests whether the same holds on the S1 base.

## Earlier advice followed and departed from

- **Followed.** S1 is the base (experiments 428 and 439). Q99 pinball loss on real harm rows is
  reported beside every arm (experiment 439's advice). Every slice carries its seed-spread minimum
  detectable effect, and a slice with fewer than 20 dates is not called a test (experiment 441's
  advice). The sweep steps the same lags in every arm (experiment 428's advice).
- **Departed from.** Experiment 428 advised "do not train a widening path on the clean target". U2
  does not do that: U2 trains on worlds in which the jump is ambiguous, and U2-clean repeats the
  clean-target design as the control.

## Handling the S1 tail caveat

**Experiment 439 found S1's q99 pinball loss on real harm rows 0.026 higher than R-inj's, above the
0.02 limit.** Three rules handle it.

1. Every table reports q95 and q99 pinball loss and coverage of S1 beside R-inj, for every row set.
2. Every Brief C tail arm is scored on real harm rows against both S1 and R-inj as comparators.
   (The arms are fitted once, on the Brief C base. The R-inj comparison is a second reference in
   the table, not a second fit.)
3. The clean-row tail pinball limit of +0.02 is measured against S1 as well as against the base.

## Slices, row sets, and actuals

- **Row sets** as in experiment 428: clean, injected harm, real harm, and real event rows, for the
  signed-MW six, the MVA-only eight, and D15 and D18 (report-only). Real event and real harm rows
  are reported with and without the largest event, with event-cluster intervals.
- **Cold-winter-night slice.** A named slice in every coverage table and every pass rule, defined as
  in experiment 441 (its mask code is imported, not copied). Targets: q95 coverage at least 0.92 and
  q99 at least 0.975 per group (the accepted tolerances of 0.03 and 0.015 from nominal), with
  month-block intervals. The table first lists the row and day counts and the minimum detectable
  effect from the seed spread, then each arm per fold. For orientation, experiment 441's counts on
  its scored set were 11,793 rows on 21 days at the six and 16,958 rows on 28 days at the eight,
  in 3 of the 4 folds; the counts here are recomputed on this experiment's rows. Its minimum
  detectable effects on CRPS were 0.12 to 0.23, so a gain smaller than that is not detectable on
  the slice.
- **NRA-corrected actuals.** Real event rows are scored against the metered reading minus the
  measured deviation of the event at the donor and at each recipient, where experiment 402's step 2
  redo measured one, and against the labelled depth otherwise. The labelled-depth fallback is also
  scaled by 0.75 and 1.25, centred on the measured deviation where it exists. The page states how
  many events have a measured deviation and how many were dropped for lack of any depth. The
  metered flow is a secondary report. Experiment 435 builds these actuals; this experiment reads
  them if they exist before the first score and otherwise builds them with the same code.
- **Unlogged events.** NGED says some load-shift events are missing from its logs. Rows labelled
  clean can hold events no label covers, so every clean-row and harm-row result is a lower bound
  on the effect of events, and the real event rows undercount. The true rate of genuine level
  changes (new connections, a generator curtailed for days) is unmeasured.
- **Rate and duration.** A genuine level change and a load-shift event cannot be told apart at the
  start of a jump, so the band width that U2 learns depends on the injected rate and duration.
  The write-up reports the 2% and 5% rates side by side, and both duration distributions for U1n
  only.

## What can only be evaluated after table P

**Every rule that compares against "R-inj with P" or "the base with P" waits for experiment 435.**
Until P exists, the page reports each rule on raw and 1.15 forecasts and labels the result interim.

- Brief B (a), the classes where R-inj with P fails, which are not known until P exists.
- Brief B (b), (c), and (d), whose comparator is R-inj with P.
- Brief C's coverage cells after P, refitted on each arm's own validation months, and its
  comparison with "more cells than the base with P".
- Brief C's clean-row CRPS and lower-tail pinball limits against the base with P.
- Every cold-winter-night rule that reads "after P", including Brief B (e) and P6. The
  cold-winter-night stratum of table P is requested from experiment 435 (Brief A), and these rules
  wait for it.

Rules that can be read now, on raw and 1.15 forecasts: the predictions P1 to P7, the mean scale
factor on clean rows, the coverage cells before P, the cold-winter-night coverage targets before P,
the encroachment sweep, and the limits against S1. All quantile forecasts are stored (11
quantiles, every arm, every world, every seed), so the "after P" pass is a free re-score.

## Encroachment sweep

**Every arm, including the references, is swept with experiment 392's protocol and its masked-lag
control.** The code copies the pattern of `e00428_sweep.py` (and of `e00440_sweep.py`, which sits on
the unmerged branch `exp-440-windowed-surprise` and is therefore copied, not imported). A step of
2.5% and 5% of effective capacity, and of 10% of the Primary's trailing 30-day mean power, is added
to the recent power of clean rows (both signs), and the forecast distribution is compared with the
unshifted one: median shift as a share of the step, band-width ratio, and coverage. The same lags
are stepped in every arm. After the step, every input that is a function of the lags is rebuilt,
including the surprise. The masked-lag control hides all recent power after the step, and its
largest difference from the unshifted masked forecast must be below 1e-6.

## Standing practices

- **Cleaned data** everywhere (flagged `substation_zero` half-hours dropped; the count per fold is
  reported), and the D15 and D18 start cut at 2025-03-01.
- **Fits saved** (state files in the experiment's private data folder, never in git).
- **Stored forecasts reused:** R-inj, S1, B-mix, L0, and R from experiments 428 and 439. Fitted
  here: every arm in the table above. Reuse saves 4 arms x 12 fits for the references (B-mix needs
  no fit).
- **Tails and cost reported** beside the CRPS: q01, q05, q95, and q99 coverage and pinball loss per
  group and per Primary, cold winter nights, and the avoidable-curtailment cost on the scaled limit
  (experiment 84's pattern).
- **Example forecasts:** a day-ahead forecast (median, 50% and 90% bands, actual) and the full
  10-day horizon of one forecast (the stored forecasts reach 240 hours), for S1 and the arms that
  pass, so the reader sees whether the bands widen with lead.
- **Load-shift plots:** at least 6 time series plots across both metering groups, marking each
  event's start and end and showing the forecast response of S1, U2, and the best Brief C arm. The
  unlogged-event caveat above is stated beside them. Metered generators are anonymised.
- **Compute:** one GPU process under 8 GB plus one feeder core. Scripts run as `systemd-run --user
  --scope -p MemoryMax=16G -p OOMPolicy=continue`, with the multiprocessing "spawn" start method.
  The first minutes of the first run check `uptime` and `nvidia-smi`.

## Process and reviews

**The first commit contains this planned `result.md` and the row in the theme's Research plan
page.** No pull request is opened at plan time. Sonnet writes the code. A fresh Sonnet agent that
did not write it reviews the code before the first fit (leakage, the fit stopping at each fold's
`train_end`, the injected-world construction, the scoring, the match with experiment 428's recipe,
and that every quantile forecast is stored), and a second fresh Sonnet reviews the fixed code if
the first flagged many problems. Then the fits run, a fresh Sonnet reviews the results, the
write-up is written, and a fresh Sonnet reviews the write-up against the printed report, and a
second fresh Sonnet runs the prose review. A fresh Sonnet reviews this plan first, as the size
requires.

**Audit triggers to watch.** (1) A control fails: the masked-lag control not at 0, or U2-clean
widening on injected harm rows beyond P1. (2) A result contradicts its prediction or an earlier
experiment: S1 or R-inj reproduced here differing from experiment 439 or 428. (3) The quantile
score and the error of the median point in different directions. (4) Leaving one site out flips a
contrast's sign. (5) A nested arm beats the arm that contains it: U2 beating U2i on its own
corrupted-row interval score, or U2 failing to match U1n on clean rows. (6) A headline interval
more than twice or under half as wide as the same contrast's in experiments 428 or 439. (7) A later
design depends on an unreplicated headline. If any fires, the manager starts a fresh audit before
the prose review. The write-up has the "Checks" list: median minus mean forecast per site, the
error of the median beside each quantile contrast, and which triggers fired.

## Open items

- Experiment 435 has delivered no table P, no NRA-corrected actuals, and no per-Primary diagnostic
  yet, so every "after P" rule is deferred.
- The 5% rate and the duration sweep multiply fits only for U1n; whether the scale-only arms are
  also run at 5% is decided by the rule in the design choices above.

## Problems from the move to nged-ml-research

To be filled in during the run. None so far.
