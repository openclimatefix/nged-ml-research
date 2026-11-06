---
id: "00610"
slug: nra-neighbour-summary-lag-encoder-inputs-iteration
title: >-
  Can the demand network's lag encoder keep following a load-shift this little (follow at most 0.05)
  while its event-free forecast stays as good as the A1 network's, by changing the span of the
  neighbour summaries, building the gap signals out of fold, dropping the signals during training,
  or adding a label head?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for
  about 20 arms in the private data folder; (2) more than one design could test the question (the
  plan fixes the arms and lists the departures); (3) it spans code whose callers cannot be named
  without searching (experiments 560, 574, 581, 584, 591, 592, 597, 598, and 606); (4) it touches
  no serving path and no degradation rule; (5) it changes no contract. Reviews: one Sonnet agent
  for the code, one for the write-up against the printed report, one for the prose, and one fresh
  Sonnet agent that opens every rendered figure. Ceiling 4 GPU-h; ask before crossing.
hypothesis: >-
  In experiment 606, summaries over all electrical neighbours (arm P7) cut the median forecast's
  follow of a load-shift from 0.119 (A1) to 0.035, but raised the event-free CRPS by 0.021 at the
  eight MVA Primaries and by 0.068 at the six MW Primaries. The cost is caused partly by a
  mismatch between training and test signals (the signals of the training rows come from a
  network that saw those rows), and partly by a network that has learnt to depend on signals that
  are sometimes uninformative. Building the signals out of fold, dropping them on part of the
  training rows, or passing them through one scalar lowers the cost without lowering the benefit.
prediction: >-
  Written before any score. Reference values are P7's from experiment 606 (follow 0.035; event-free
  CRPS minus A1 0.021 at the eight MVA Primaries and 0.068 at the six MW Primaries). Cross-fitted
  signals (CF) lower the six-MW cost by at least 0.03 and keep follow at most 0.05. Signal dropout
  on 30% (D30) or 50% (D50) of the training rows lowers the six-MW cost by at least 0.02 each and
  raises follow by less than 0.02. The scalar bottleneck (BN) lowers the six-MW cost by at least
  0.02 and keeps follow at most 0.05. The label head at weight 0.1 (X1) or 0.3 (X3) changes follow
  by at least 0.01 (either direction) and leaves event-free CRPS within the minimum detectable
  effect. A box window of 3 or 7 days follows within 0.015 of P7, and a window of 1 day alone
  follows at least 0.02 more than P7. No single change reaches the keep rule at both groups; the
  best combination of the changes that helped does. The placebo (neighbours shuffled between
  Primaries) follows at least 0.02 more than the best arm.
decision_rule: >-
  Keep rule (from the brief): an arm is a keeper if follow is at most 0.05 and the event-free CRPS
  is within the printed minimum detectable effect of A1 at both the eight MVA Primaries and the six
  MW Primaries (all rows), and the coverage at the gate's band width is no more than 0.01 below the
  gate's (R). Event metrics are descriptive (event-block bootstrap), as in experiments 584 and 598.
  Twenty counted contrasts (8 span arms and 10 stage 2 arms against P7 on follow, the combination
  against P7, and the placebo against the best arm), Bonferroni alpha 0.05/20 = 0.0025, with the
  minimum detectable effect beside each. A leak or slice cost is a flag unless it actively hurts.
benchmark: >-
  A1 (experiment 591, stored forecasts) for the keep rule; P7 (experiment 606, stored forecasts)
  for the contrasts; R (A1 plus the G4 gate, experiment 597) for the coverage at the gate's width.
row_set_id: experiment_470_intersection_v1
metric: >-
  Follow, share of the NRA inside q05-q95 by phase (day 4 to the end, 0 to 7 days after), band width,
  event-free CRPS at 24-120 h at the eight MVA and the six MW Primaries, with the minimum
  detectable effect beside each contrast.
---

## Plan

**Standing practices followed.** Refit at day 0 and day 14 of each fold month (the coarse cadence
of experiments 584, 591, 598, and 606), cleaned data everywhere, the D15 and D18 cut before
2025-03-01, every fitted network saved in the data folder, stored forecasts of A1, R, and P7
reused as controls, the first fold pair (2025-10 and 2025-12) screened before the other months,
seeds 0 and 1, and July 2026 unopened. The code is a copy of experiment 606's arm, model, signal,
and scoring modules (not imported across worktrees), with the changes listed below.

**Arms** (every arm starts from experiment 591's A1 step-2 weights and its shared lag encoder, and
differs only in step 3, as in 606). Inputs, per lag: P7 is A1's lag encoder inputs plus the per-lag
gap (lag minus the lag-free forecast's median at the lag's own time), three exponentially weighted
averages of the gap (half-lives 1, 3, and 7 days), the expected-missing flag, and summaries over all
electrical neighbours (sum, mean, minimum, and maximum of the gap and of each average, the sum of
the Primary's own value and the neighbours' sum, the neighbour count, and a "no neighbour" flag).

| Stage | Arm | Definition |
|---|---|---|
| 0 | A1, P7, R | stored forecasts, not refitted |
| 1 | B1, B3, B7, B14 | P7 with the three weighted averages replaced by one trailing mean of the gap over the last 1, 3, 7, or 14 days up to the lag's time |
| 1 | B1-14 | the four trailing means together (1, 3, 7, 14 days) |
| 1 | B1-28 | the four trailing means plus a 28-day mean |
| 1 | E1, E7 | P7 with only the weighted average of half-life 1 day, or only 7 days |
| 2 | CF | the best stage 1 inputs, with the lag-free forecast table built out of fold (below) |
| 2 | D30, D50 | the best stage 1 inputs; on 30% or 50% of training rows, all signal columns take the "no information" value (zero, the missing flags set to 1, the neighbour "none" flag set to 1) |
| 2 | BN | the best stage 1 inputs reach the lag encoder only through one learnt scalar (a 1-hidden-layer network with 16 units, read by both lags) |
| 2 | X1, X3 | the best stage 1 inputs plus a head on each lag-encoder output that predicts whether the lag is inside a synthetic shift (cross-entropy) and the shift depth as a share of capacity (squared error on shifted lags), added to the loss at weight 0.1 or 0.3 |
| 2 | O1 | compact summaries: only the neighbours' mean and the own-plus-neighbours sum |
| 2 | O2 | adds the share of neighbours whose gap has the opposite sign and exceeds 5% of capacity, for the gap and each average |
| 2 | O3 | signals for the week lag only (the day lag reads no signals) |
| 2 | O4 | dropout of 50% on the neighbour columns only (own columns always kept) |
| 3 | COMBO | the stage 2 changes that individually met the prediction, together |
| 3 | placebo | the best arm with each Primary reading another Primary's neighbours (a fixed derangement) |
| 3 | widened | the best arm rescaled about its median to the gate's band width (scoring only) |
| 3 | winner on 3 more months | the best arm on 2026-01, 2026-03, and 2026-06 |

**Cross-fitting.** In experiment 606 the lag-free table `T` is the lags-masked median of the step-2
network, which has seen the training rows, so the training gaps are smaller than the gaps at test
time. CF builds `T` for the training rows from networks trained without them. The training window of
step 2 is split into three contiguous blocks by valid time; for each block, a copy of the step-2
recipe (the same pre-trained weights, fresh mixer, 20 epochs at most) is trained on the other two
blocks and predicts the lags-masked median on the block. Test rows use the full step-2 network.
Cost: three extra step-2 trainings per fold month and seed (each on two thirds of the rows). If a
training takes more than 15 minutes, CF runs on one seed only and the page says so.

**Screening rule (to save GPU).** Every arm is fitted and its event-free CRPS scored. An arm is
injected (the 15 and 30 day bank) only if its event-free CRPS minus A1's is below P7's at both
groups. Arms that fail are reported with their event-free CRPS only.

**Target-side bank.** The winner is also scored on the target-side bank with a mirrored neighbour
(experiment 595's method), because the lag-only bank flatters neighbour inputs. If the GPU ceiling
does not leave time for it, the page says that the check was not run.

**Label head check.** For X1 and X3, the head's "active" output is also read on the observed lags
around the real December 2025 pair (D5 and D16), descriptively; the head is never trained on real
events.

**Plots.** Every time-series plot follows the standing plot rules (at least 280 px per panel, one
forecast arm per panel, a legend, the maintainer's example event of 2025-12-30 01:00 to 2026-01-14
11:30, the NRA left off the lines). A fresh Sonnet reviewer opens every rendered figure.
