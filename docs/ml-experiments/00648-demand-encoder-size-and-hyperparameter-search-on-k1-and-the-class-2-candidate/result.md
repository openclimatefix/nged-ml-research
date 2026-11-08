---
id: "00648"
slug: demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate
title: >-
  Does a larger or differently shaped part of the K1 or class 2 candidate demand network lower its
  quantile score by the agreed margin, now that the lag encoder also reads neighbour summaries?
theme: lagged-power-and-error-correction
also_relevant_to:
  - calibration-and-uncertainty
  - production-reference-and-best-guess
status: planned
kind: change
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for
  about 20 arms in the private data folder only, and reads the stored K1 and candidate forecasts of
  experiments 612 and 613 without writing them; (2) more than one design could test the question
  (one factor at a time, a random search, or a grid), so this plan fixes one factor at a time
  because experiment 447 did and a search is deferred until a factor passes; (3) it spans code whose
  callers cannot be named without searching (experiments 545, 559, 560, 572, 574, 591, 606, 610,
  612, and 613); (4) it touches no serving path and no degradation rule; (5) it changes no
  contract. Ceiling 8 GPU-hours (process hours), one GPU process at a time; ask before crossing.
  Sonnet writes and reviews; no Opus.
hypothesis: >-
  Experiment 447 found the S1 demand network insensitive to its settings. K1 and the class 2
  candidate differ from S1: the lag encoder reads two lags with a learnt lag ID, and the candidate
  adds neighbour summaries. Their parts may no longer sit at a flat optimum, so changing the width
  or depth of the lag encoder, the width of the neighbour-summary encoder, the number of mixture
  components, the hidden width, the dropout, the learning rate, or the maximum number of epochs
  lowers days 1 to 10 event-free CRPS by more than the margin without making the network follow an
  injected load-shift more.
prediction: >-
  Written before any fit and before any score is read. P1. Most one-factor arms change the pooled
  event-free CRPS by less than the margin (as in experiment 447). P2. A wider lag encoder (128) and
  a wider hidden layer (128) are the likeliest to help the candidate, each by 0.02 to 0.08 points of
  capacity, because the candidate's lag encoder carries more inputs than S1's. P3. A bigger lag
  encoder raises the follow ratio of an injected shift by 0.01 to 0.05, so one of the two guards may
  fail first. P4. Five mixture components change CRPS by under 0.03 points and widen the upper tail.
  P5. Dropout 0.0 and a learning rate of 1e-3 hurt. The prediction that could fail: at least one
  arm passes the screen and again passes on 5 folds by 3 seeds. If none passes the screen, the
  experiment stops and runs no search.
decision_rule: >-
  Screen folds: test months 2026-06, 2026-01, 2025-10; seed 0; the same monthly daily-refit recipe
  as
  K1 and the candidate of experiment 613, changed in one setting per arm. The reference is the same
  network at its current setting, refitted on the same folds and seed in this experiment (the
  stored seed 0 fits of experiments 612 and 613 are used only to check the refit reproduces them to
  four decimals). Contrast = arm minus reference in event-free CRPS (points of capacity, 24 to 120
  h, pooled over the 14 Primaries, D15 and D18 cut before 2025-03-01), with a month-and-day block
  bootstrap 95% interval. Margin = the larger of 0.064 points scaled by the ratio of the reference
  network's pooled CRPS to S1's on the same folds (experiment 447's margin), and 1.96 x sqrt(2) x
  the standard deviation across K1's stored seeds 0 to 4 on the three folds; this is computed from
  stored forecasts and committed in this file before any arm is scored. An arm passes the screen
  when its pooled gain is at least the margin, it improves at least 2 of the 3 folds, the gain
  holds with D15 and D18 left out, the follow ratio of an injected shift rises by at most 0.03, the
  safe-side share (at q95 and q05) falls by at most 0.03, and no pooled q95 or q99 pinball loss
  rises by more than 0.02 points. Only a factor with a passing arm goes on to 5 folds (add 2026-03
  and 2025-12) by seeds 0 to 2, against the reference refitted on the same folds and seeds; it
  passes when its pooled gain is at least the margin, its 95% interval excludes zero, it improves
  at least 3 of 5 folds, and both guards hold. If no arm passes the screen the experiment stops
  and reports; it runs no random search. The smallest gain worth porting is the margin, because
  every setting is a constant in the training script and costs nothing in production. Interim; no
  99% claim.
benchmark: >-
  K1 and the class 2 candidate (P7_D50_e3 at K1's band width) of experiments 612 and 613, refitted
  at
  their current settings on the screen folds as the reference of every contrast.
row_set_id: experiment_470_intersection_v1
metric: >-
  Event-free CRPS at 24-120 h and 120-240 h by lead band and by Primary, pooled and without D15 and
  D18; q01, q05, q95, and q99 pinball loss and coverage; follow ratio and safe-side share on the
  injected bank; absorption not scored.
production_status: not-for-production
adopted_in: none
---

## Plan

**Question.** Experiment 447 changed one setting of the S1 demand network at a time and found no
gain above 0.064 points. K1 and the class 2 candidate are different networks (a shared lag encoder
with a lag ID, and neighbour summaries in the candidate), so this experiment repeats the screen on
them. It follows the advice of experiment 447 to look at inputs and the lag-trust design before
more tuning, and departs from it only because the maintainer asked for a new look after the lag
encoder gained neighbour summaries.

**Networks.** K1 is the lag network with the G4 gated band; the class 2 candidate is K2 plus
neighbour summaries and 50% signal dropout from
[experiment 610](../00610-nra-neighbour-summary-lag-encoder-inputs-iteration/result.md), scaled to
K1's band width ([experiment 613](../00613-nra-roster-contrasts-with-intervals-five-
seeds/result.md)).
Both are refitted from pre-training through step 2 and the daily fine-tune, with the changed setting
applied at every stage.

**One-factor arms (default in bold).** Each arm changes one module constant before the network is
built, through a wrapper around experiment 612's and 613's arm scripts.

| Factor | Levels | Applies to |
|---|---|---|
| Lag-encoder width (`LAG_HIDDEN`) | 32, **64**, 128 | K1, candidate |
| Lag-encoder depth (`layers`) | **1**, 2 | K1, candidate |
| Neighbour-summary encoder width (`BOTTLENECK_HIDDEN`) | 8, **16**, 32, 64 | candidate |
| Mixture components (`COMPONENTS`) | **3**, 5 | K1, candidate |
| Hidden width of the encoders and the mixer (`ENCODER_HIDDEN`, `MIXER_HIDDEN`) | **64**, 128, 256, 384 | K1, candidate |
| Dropout (`DROPOUT`) | 0.0, **0.1**, 0.2 | K1, candidate |
| Learning rate (all stages) | x0.5, **x1**, x2 | K1, candidate |
| Maximum epochs and patience | **current**, both x2 | K1, candidate |

That is 16 arms on the candidate and 14 on K1, each with 3 folds at seed 0. If one fold-seed fit
chain takes long enough that all 30 arms exceed the 8 GPU-hour ceiling, the arms are dropped in this
order until the projection fits: K1 arms for the neighbour-summary-free factors that the candidate
arm already covers least (dropout, maximum epochs), then the hidden width 384, then the learning-
rate
x2 arm. The projection is made from the measured time of the first arm and recorded before the
second arm starts.

**Standing practices followed.** Monthly refits with daily fine-tunes as in experiment 613, cleaned
data, the D15 and D18 cut before 2025-03-01, fitted weights saved in the private data folder,
stored experiment 612 and 613 forecasts reused only as a reproduction check, tail and cost
reporting, a draft pull request, and the example-forecast and event plots in the write-up. The
weather source at lag times is the one experiments 612 and 613 use, unchanged; the code review names
it in the PR body, and the write-up says whether it exists at forecast time.

**Process.** Plan commit and draft pull request; a fresh Sonnet code review before the first fit;
a free step-0 script that computes the margin from K1's stored seeds and commits it; the screen; a
fresh Sonnet agent that re-derives the headline numbers; the write-up; one Sonnet prose review.
Chat updates go to the manager only: plan pull request open, fits started, fits finished, pull
request merged.

## Margin (fixed before any arm is scored)

`e00648_margin.py` reads experiment 613's stored event-free cells for seed sets 0 to 1 and 2 to 4 on
the three screen months, 24-120 h, all Primaries. The seed standard deviation comes from the
disagreement of the two seed-set means.

- K1: CRPS 3.009 points of capacity, seed standard deviation 0.0407, minimum detectable effect for
  one seed against one seed 0.1129, experiment 447's margin scaled 0.0640. **Margin 0.113 points.**
- Class 2 candidate: CRPS 3.056 points, standard deviation 0.0393, minimum detectable effect 0.1089,
  scaled 0.0650. **Margin 0.109 points.**
- Caveat: the two seed sets differ by 0.064 in 2025-10 and by under 0.016 in the other two months, so
  the standard deviation rests on one noisy month. The margin is a lower bound on what a single seed
  can detect and is not tuned afterwards.

## Results

Not yet run.
