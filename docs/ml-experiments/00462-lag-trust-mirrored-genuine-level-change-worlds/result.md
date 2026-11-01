---
id: "00462"
slug: lag-trust-mirrored-genuine-level-change-worlds
title: >-
  Can a lag network that sees the neighbours learn to follow a level change that no neighbour
  mirrors, when its training worlds contain genuine level changes?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
date: 2026-11-01
size: >-
  medium to complex (row 4 of the Opus review of experiment 404, merged with G3 of the NRA
  discussion's final list). Stored: 16 fitted neural networks for the first stage (S3-amb and
  S3-rw-amb, 4 folds, 2 seeds), 8 more only if that pair separates (S3-attn-amb), their forecasts,
  and sweep tables, in this experiment's private folder only; experiments 404's, 428's, 439's,
  440's, and 446's folders are read and never written. More than one design would test the
  hypothesis (what a mirror looks like, how large the genuine share is, how the attention arm reads
  its tokens), so each choice is fixed in this plan before any score is read. Callers: the scripts
  import the folders of experiments 402, 428, 440, and 446 and the `studies` evaluation code. This
  repository holds no production serving path or degradation rule.
hypothesis: >-
  Every training world so far (experiments 402, 404, 440, and 446) keeps a clean target under every
  injected jump, so discounting every jump is optimal and a neighbour mirror carries no information.
  When the training worlds also contain genuine level changes (the lags and the target move
  together), some with the load-conserving mirror at 1 or 2 neighbours that a load shift leaves and
  some without, a lag network given the neighbours' surprise (S3) can use the mirror, and a network
  given randomised neighbours (S3-rw) cannot. A pass would show only that a network can use a mirror
  when the training world makes the mirror informative. It would not be evidence about real
  Primaries, because no labelled genuine level change exists in the real data.
prediction: >-
  Written before any fit and before any score is read. P1. S3-amb scores lower CRPS than S3-rw-amb
  on held-out injected level-change rows by at least 0.03 points of effective capacity at the
  signed-MW six (author's guess, about one chance in three: experiment 440 found S3 and S3-rw about
  equal in worlds without genuine changes). P2. The share of a k = 3 step the forecast follows is
  higher without a mirror than with one for S3-amb, and the two shares are equal for S3-rw-amb. P3.
  The Bayes-optimal shares of the training world (printed by the generator's test, below) bound
  what any arm can reach, and they sit above 0.05 with a mirror and below 0.5 without one, so the
  pre-registered sweep thresholds are probably unreachable at a 2% genuine share. P4. S3-amb costs
  0.03 or more of clean-row CRPS against U1n-2 (experiment 440 found S3 +0.044 against S1).
decision_rule: >-
  Four folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 and 1, monthly refits, as in
  experiments 428 and 446. Every number is interim and carries no 99% claim. The ambiguous-world
  rule of the brief: the second part of the maintainer's hope is supported if (a) S3-amb beats
  S3-rw-amb on held-out injected level-change rows and on injected load-shift harm rows by more than
  the larger of 0.03 and the minimum detectable effect; (b) no arm loses more than 0.03 CRPS points
  on rows with no labelled event, or more than 0.02 on q99 pinball loss, against U1n-2 (the same
  ambiguous worlds; the cost against S1 is reported beside it as a stated production cost); and (c)
  the paired sweep shows a share of the step followed of at most 0.05 with a mirror and at least 0.5
  without a mirror at k = 3. S3-attn-amb is fitted only if (a) holds for the pair on level-change
  rows by half of the margin or more; if S3-attn-amb passes where S3-amb does not, attention over
  the lags is the first learned-trust arm to pass a control. If no arm passes, the learned-trust line
  closes and B-mix or its row-3 successor stays. The adoption gate for the forecast is the real-row
  checks (no loss beyond 0.03 on rows with no labelled event, no tail harm above 0.02), never a pass
  on injected rows alone. The Bonferroni count is 3 contrasts (S3-amb minus S3-rw-amb on level-change
  rows, on load-shift harm rows, and S3-amb minus U1n-2 on rows with no labelled event).
benchmark: >-
  U1n-2 of experiment 446 (S1 trained on the same recipe of ambiguous worlds; stored networks,
  forecast only on the new held-out worlds), S1, R-inj, and B-mix of experiments 428 and 404
  (stored forecasts on the real rows).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles in points of effective capacity,
  coverage and pinball loss at q01, q05, q95, and q99 (raw and at the 1.15 widening), the share of a
  paired step the forecast follows, and the avoidable-curtailment cost at the signed-MW six.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00462-lag-trust-mirrored-genuine-level-change-worlds/
---

*This page is the plan, committed before any code ran. The write-up replaces it.*

## Question

**The experiment asks whether a lag network can learn to follow a jump in recent power that no
neighbour mirrors, and to discount a jump that a neighbour mirrors.** The lag network (S1) is a
neural network given three earlier readings of a Primary's power and the surprise at each reading.
S3 adds the neighbouring Primaries' surprise at the same times. In every training world so far a
jump never moved the target, so a network learned to discount every jump and the neighbours had
nothing to add. This experiment adds genuine level changes to the training worlds, where the target
moves with the lags, and leaves a mirror at 1 or 2 neighbours on half of them.

**What a pass means.** A pass shows only that a network can use a mirror when the training world
makes the mirror informative. Experiment 456 found that real mirrors are not evidence of a load
shift at the thresholds it tested, and no labelled genuine level change exists in the real data, so a
pass on injected rows says nothing about real Primaries. The real-row checks are the adoption gate.
NGED's event labels are incomplete: rows with no labelled event may hold unlabelled events, so every
such result is stated as "rows with no labelled event", never "clean rows", and the page repeats the
unlogged-event caveat.

## Design

- **Genuine level change.** The generator reuses experiment 446's `draw_events` (so the own events
  are the same events as U1n-2's: seeds 300 to 303, 2% of each Primary's observed half-hours, depths
  and durations from the corrected label table). It then flips a fair coin per event. On a heads
  event it chooses 1 or 2 of the Primary's neighbours at random and adds
  `-step × capacity_p / k / capacity_q` to each neighbour `q` over the same half-hours, exactly the
  signature of a donor and its recipients in experiment 402's load-shift generator. On a tails
  event no neighbour changes. Lags and target move together at the Primary and at every recipient.
  MVA-only Primaries clip at zero in the lags and the target alike.
- **Why that mirror.** The mirror is the load-conserving one. A load shift in the experiment 428
  worlds (a donor with 1 or 2 recipients in 70% of events, none in 30%) leaves the same signature,
  so a mirrored genuine change is indistinguishable from a load shift by its neighbours and differs
  only in what the target does. That is the case in which the mirror carries information only
  through base rates, which is what the maintainer's hope needs to test. It is a design choice taken
  here; the neural Discuss session has not confirmed the reading of "a mirror at the recipients".
- **Base rates are printed.** The generator's test prints, for the realised training worlds, the
  share of jump rows (a lag inside a jump) whose target moved with a mirror and without one. Those
  are the Bayes-optimal shares followed, and the sweep thresholds of the brief are compared with
  them.
- **Placeholder rate.** The genuine share stays at the brief's 2% placeholder, the rate of
  U1n-2. Experiment 456's scan cannot set a rate. The realised share of half-hours inside a genuine
  change at a Primary or a recipient is printed and is higher than 2% (about 2% plus the recipients'
  half-hours). S1-amb is therefore U1n-2 only approximately: U1n-2 trains on worlds with fewer
  genuine-change half-hours. The cost of an exact S1 on these worlds is 8 fits, and the plan asks the
  manager whether to add them.
- **Held-out worlds.** Three worlds on the real clean series (the real rows of worlds 7 to 9), drawn
  with seeds 500 to 502 and the 3% held-out rate of experiment 446, with the same mirror rule.
  Their events are never seen in training. They hold level-change rows with a mirror, level-change
  rows without one, and (from the experiment 428 worlds 4 to 6) load-shift harm rows.
- **Arms.**
    - **S3-amb:** S3 of experiment 440 (S1 plus 24 window statistics per Primary and 30
      neighbour columns) trained on the ambiguous worlds with experiment 446's recipe: token
      dropout 0.2 on each lag and on each lag's surprise, origin of the lags drawn 0 to 7 hours
      before the usual one. A dropped lag also zeroes its window and neighbour columns.
    - **S3-rw-amb:** the same with random neighbours that keep each Primary's degree, in the
      features only. The mirror in the worlds still reaches the real neighbours.
    - **S3-attn-amb (conditional):** a single multi-head attention layer over the 3 lag tokens and a
      lagless token (the non-lag inputs). Each lag token carries the lag, its missing flag and age,
      the surprise, the own window statistics, and the neighbour columns of that lag. No
      down-weight-only constraint. Fitted only if the pair separates (rule above).
    - **References, no fits:** U1n-2 (forecast only on the new held-out worlds, from experiment 446's
      stored networks), S1 and R-inj (forecast only on the new held-out worlds, from experiment
      428's), B-mix, S1, R-inj, and 440's S3 and S3-rw (stored forecasts on the real rows).
- **Recipe.** Experiment 428's rows, folds, fit loop, and frozen cross-fitted lagless tables of
  experiment 404 (seeds 0 and 1). Monthly refits, as always in research. The Primaries D15 and D18
  are cut before 2025-03-01 as in experiment 428's rows (stated in the write-up). Fitted
  networks are saved in this experiment's data folder.
- **Paired sweep.** On the new held-out worlds' clean real rows, a step of k times the lag standard
  deviation (k = 1, 2, 3) is added to the lags of the first k lags with a mirror (the load-conserving
  signature at 1 or 2 neighbours, which shifts the neighbour window columns and neighbour-relative
  surprise) and without one, and the share of the step the median moves is measured, as in
  experiment 440's sweep.
- **Scored rows.** Experiment 446's row sets on the real rows (world 0: rows with no labelled event,
  real harm, real events against NRA-corrected actuals as the headline and metered actuals as
  secondary, each with and without the longest event) and on the held-out worlds (gen rows with a
  level change in force at a lag, split by mirror and no mirror, plus the load-shift harm rows of
  worlds 4 to 6). Equal-row set across arms, checked in code.

## Compute

Fit count: 2 arms × 4 folds × 2 seeds = **16 fits** for the first stage, about 4 graphics-processing-
unit hours at 0.25 h per fit (to be timed on one fit first; the brief's "8 fits, about 2
graphics-processing-unit hours" reads as one arm). S3-attn-amb adds 8 fits only if the pair
separates. S1-amb is the stored U1n-2 (12 fits saved). One process under 8 GB (`MemoryMax=8G`,
`OMP_NUM_THREADS=1`, 1 feeder core).

## Standing practices

Monthly refits (not weekly); cleaned data; the D15 and D18 cut; saved networks; stored forecasts
reused as references; screen not used (a mechanism test with four folds); tails reported at q01,
q05, q95, q99; example forecasts and event plots on the page; draft pull request. Reviews: a fresh
Sonnet reviews this plan, the code before the first fit, the results, and the write-up.
