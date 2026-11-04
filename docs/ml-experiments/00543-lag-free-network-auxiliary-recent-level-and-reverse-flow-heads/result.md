---
id: "00543"
slug: lag-free-network-auxiliary-recent-level-and-reverse-flow-heads
title: >-
  Can an auxiliary head that predicts the recent power level, or the reverse flow, close the gap
  between the lag-free demand network and the lagged forecast without lag inputs at forecast time?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. Stored: fitted network weights and window forecasts under this experiment's private data
  folder only; earlier experiments' predictions are read, never written. Designs: more than one is
  defensible (which quantity the recent-level head predicts, the loss weight), so the plan fixes
  each before any score is read. Callers: this folder's scripts import experiments 397, 425, 428,
  510, and 511 and the `studies` evaluation code. Serving path and degradation rule: not touched.
  One Sonnet code review before the first fit, one Sonnet review of the write-up against the
  printed report, one Sonnet prose review.
hypothesis: >-
  The lag-free network trails the lagged network (A-pool) because it never sees recent power. A
  second output that predicts the recent power level (the newest observed value and the day-ago
  and week-ago values, used as training targets only and never as inputs) makes the shared hidden
  layers carry level information, so the lag-free network closes part of the gap without any lag
  input at forecast time.
prediction: >-
  Written before any score is read. F1 (recent-level head) closes at least half of the L-1d minus
  A-pool gap at 24 to 72 h at the eight (L-1d minus A-pool is +0.288 at 24-48 h and +0.254 at
  48-72 h, so F1 minus A-pool at most +0.144 and +0.127), without raising the injected-shift leak
  above 4.49 + 0.5 points of capacity. It may fail: the newest value used as an input (experiment
  538's R1) closed only 24% to 28%. F2 (reverse-flow head) changes the eight by less than the
  minimum detectable effect, because the eight are MVA-only and carry no sign; F2 is expected to
  matter, if at all, at the six. F3 (both heads) is no better than F1 at the eight.
decision_rule: >-
  F1 counts as supported if its paired contrast against A-pool and against L-1d, pooled over
  24-72 h at the eight, has a Bonferroni interval excluding zero in the predicted direction and
  closes at least half of the gap, and the leak stays within 4.99 points of capacity. Cost of
  porting: an extra output head and a training loss, no new data source and no serving input, so
  the threshold for a gain is an interval excluding zero. The other contrasts are report-only.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425), R-DW (experiment 539), R1 and R9 (experiment
  538), and the rebuilt K2-S1 (experiment 541) when stored, on experiment 511's rows; BG2,
  climatology, and `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight (MVA-only Primaries) and the six (signed-MW
  Primaries), D15 and D18 report-only, and the injected-shift leak of the 511 stress world.
---

# Auxiliary heads for the lag-free network (planned)

**Plan, committed before any fit or score.** Base recipe: experiment 511's L-1d (ERA5 and CAMS
pretrained, ENS fine-tuned, daily refit, 90-day half-life, 2 seeds, 5 folds of 511). Each arm adds
outputs to the last layer of the network and a loss on them. The inputs are the 37 lag-free columns
and the Primary embedding, exactly as in L-1d. July 2026 is not scored and not tuned on.

- **F1, recent-level head.** Three regression outputs predict the standardised value of the newest
  observed power at the origin, the day-ago power, and the week-ago power (the three lag value
  columns of experiments 397 and 538), masked when the lag is missing. These lags are training
  targets only. Loss: mean squared error, weight 0.5 relative to the forecast loss, fixed before any
  score is read and not tuned.
- **F2, reverse-flow head.** One logit predicts whether the power at the target time is below
  -0.005 of capacity. The label exists only where the sign is metered: it is masked at the eight.
  Loss: binary cross-entropy, weight 0.5. Reverse flow is rare at the six (0.00% to 0.26% of
  half-hours), so F2 is expected to be nearly inert; the label counts are reported.
- **F3.** Both heads.
- **Weather inputs.** Every weather input of a row comes from the run issued at or before the
  forecast origin. The code asserts this row by row and prints one example row.
- **Leak.** The injected-shift stress world of experiment 511 (7-day block, both signs, seed 0)
  shifts the forecast target and the lag-valued auxiliary targets together.
