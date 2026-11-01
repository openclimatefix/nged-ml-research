---
id: "00487"
slug: era5-and-cams-replay-rows-during-demand-network-fine-tuning
title: >-
  Does mixing ERA5-and-CAMS rows into the fine-tuning batches improve the demand network on cold
  winter nights and in the tails?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
date: 2026-11-01
size: >-
  medium. Stored: 45 fine-tuned networks (the state files and one 11-quantile forecast per fit) in
  the experiment's private folder. More than one design would defensibly test the question (how a
  replay share is defined, which rows count as cold nights), so the plan fixes each before any
  score is read. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched. Callers: the scripts import experiments 397, 418, 425, 454, and 470, whose
  callers sit in those folders. Compute: 45 fits plus one reproduction fit, about 65 minutes of
  wall time with one GPU process under 3 GB, one feeder core. One Sonnet plan review, one Sonnet
  code review before the first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  Fine-tuning experiment 425's arm A on the ensemble (ENS) rows alone may make the network forget
  what its ERA5-and-CAMS pretraining taught it about rare cold winter nights, which the ENS history
  (from 2024-04) holds few of. Replaying ERA5-and-CAMS rows during fine-tuning, especially
  cold-night
  rows, lowers the CRPS on cold winter nights.
prediction: >-
  Written before any score is read. P1: Rcold lowers cold-night CRPS by at least 0.05 points of
  effective capacity at the eight and at the six; because the cold-night minimum detectable effect
  is likely above 0.05, a gap below the printed cold-night minimum detectable effect is called
  inconclusive. P2: R25 and R50 leave CRPS at the eight and at the six on rows with no labelled
  event equivalent to R0, meaning the 27-test Bonferroni interval of the contrast lies inside plus
  or minus the minimum detectable effect for three-seed means (0.065 at the eight and 0.100 at the
  six in experiment 470, recomputed and printed before any contrast). P2 is reported, never a gate:
  the interval may be too wide to satisfy it, and failing equivalence is not evidence of an effect .
  P3: no arm raises q99 pinball loss by
  more than 0.02 points at the eight or at the six on rows with no labelled event. Experiment 452
  found that a longer pretraining did not help, so the null is the likely outcome.
decision_rule: >-
  Score = CRPS in points of effective capacity, 1.15 widening unless stated, three seeds averaged
  per arm, month-block bootstrap (month by origin day, 20,000 resamples) with Bonferroni intervals.
  The count of tested contrasts is 27, fixed before any score is read: 3 contrasts (R25, R50, and
  Rcold, each against R0) times 9 tests (CRPS on 7 row sets: the eight; the six on rows with no
  labelled event; the six on real-event rows against normal-running-arrangement actuals; cold winter
  nights at the eight and at the six; the 12 to 36 hour band at the eight and at the six; plus the
  q99 pinball loss at the eight and at the six on rows with no labelled event). Christmas week, D15
  and D18, the other tail quantiles, the widening sweep, and curtailment are descriptive and enter
  no count. The minimum detectable effect (experiment 428's formula on the seed spread of arm A's
  five single seeds) is printed first, at the eight, the six, and on cold nights at each. An arm is
  a candidate if (a) the Bonferroni interval of its cold-night CRPS change excludes zero in the
  improving direction at both the eight and the six, with a point estimate of at least 0.05 points,
  and (b) the upper end of the Bonferroni interval of its CRPS change at the eight and at the six on
  rows with no labelled event is at most +0.02. Equivalence (P2) never makes an arm a candidate.
  Stop rule, judged on Rcold only: close the idea (the network is not forgetting the rare rows) only
  if the printed cold-night minimum detectable effect is at most 0.05 and the upper end of the
  Bonferroni interval of Rcold's cold-night CRPS change is above -0.05 at both the eight and the
  six; otherwise report the result as inconclusive and do not close it The smallest gain worth
  porting is 0.05 points on cold nights without loss elsewhere:
  replay needs the ERA5-and-CAMS rows at every monthly refit, so replay adds a stored table and
  roughly 1.3 to 2 times the fine-tuning time.
benchmark: >-
  R0, which is experiment 425's stored arm A fit at seeds 0 to 2 on the same folds (not refitted
  except once to prove it reproduces), and experiment 470's A-pool (five-seed quantile average) as
  context.
row_set_id: >-
  The `gam14` row set of experiment 470 (the eight and the six; D15 and D18 are report-only on
  `main16`).
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss, and
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six, per metering group and slice.
verdict: inconclusive
best_arm: >-
  None. R25 (25% of each step's loss weight on ERA5-and-CAMS rows) scores 3.441 CRPS points at the
  signed-MW six on rows with no labelled event, against 3.517 for R0, but the gain is below the
  0.112 minimum detectable effect and comes mostly from one fold.
headline: >-
  Replaying ERA5-and-CAMS rows did not lower cold-night CRPS at the eight (R25 -0.012, R50 +0.026,
  Rcold +0.047 points; cold-night minimum detectable effect 0.169). At the six, R25 and R50 lower
  CRPS on rows with no labelled event by 0.075 and 0.088 points, 3 to 4 times the single-seed
  spread but below the minimum detectable effect of 0.112, and 2026-01 alone supplies about half of
  it. No arm raises q99 pinball loss by more than 0.003 points.
lessons: >-
  Replay of ERA5-and-CAMS rows is not a fix for cold nights at the eight. At the six it brings a
  three-seed fine-tune to about the level of a five-seed pool, which suggests replay acts as a
  regulariser. A cold-night-only replay pool (Rcold) is worse than a plain one.
take_home_message: >-
  Mixing ERA5-and-CAMS rows back into the fine-tuning batches does not help the cold winter nights
  at the eight MVA-only Primaries, and replaying only cold nights makes the eight slightly worse.
  At the signed-MW six, plain replay lowers CRPS by about 2.5% (to the level of a five-seed pool),
  but the gain is below the noise floor of three seeds and one fold supplies half of it. Cold nights
  stay badly under-covered at the six whatever the replay.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not add cold-night-only replay rows to the fine-tuning: Rcold raised CRPS at the eight by
    0.037 points (Bonferroni interval 0.013 to 0.065) and did not lower it on cold nights (Figure 1).
  - >-
    Test plain 25% replay again on more seeds and on folds other than 2026-01 before using it at
    the six: its 0.075-point gain is below the 0.112 minimum detectable effect and 2026-01 supplies
    0.237 of it (Figure 2).
  - >-
    Compare any replay or weighting arm with a five-seed pool as well as the three-seed reference,
    because R25 and R50 at the six land at A-pool's level (3.441 and 3.429 against 3.451).
  - >-
    Expect cold nights at the six to stay under-covered (q95 coverage 0.79 to 0.80, target 0.95)
    under every fine-tuning regime tried here; fix them in the recalibration, not in the fine-tune
    (absolute table).
---

---

**Replaying ERA5-and-CAMS rows does not fix cold winter nights at the eight, and its small gain at
the six is within the noise.** Fine-tuning experiment 425's arm A with 25% or 50% of each step's
loss on ERA5-and-CAMS pretraining rows left CRPS at the eight MVA-only Primaries unchanged, and
replaying only cold-night rows (Rcold) made the eight slightly worse. At the signed-MW six, plain
replay lowered CRPS on rows with no labelled event by 0.075 to 0.088 points (about 2.5%), which
brings a three-seed fine-tune to the level of a five-seed pool. That gain is below the 0.112-point
minimum detectable effect and 2026-01 supplies about half of it, so this experiment is inconclusive
and no arm is a candidate. All scores are selection-biased, because the five folds that choose the
candidate also score it. July 2026 is unused.

**Question and answer.** The question was whether the network forgets rare cold nights during
fine-tuning on the ensemble (ENS) rows, which start in 2024-04. The answer at the eight is no sign
of forgetting. The experiment narrows replay to a possible regulariser at the six and rules out
cold-night-only replay.

![CRPS change against R0 with Bonferroni intervals](figure_1_crps_contrasts_against_r0.svg)

Figure 1 shows each replay arm minus R0 on the seven CRPS row sets, with 27-test Bonferroni
intervals. Below zero is better.

![CRPS change by fold](figure_2_crps_change_by_fold.svg)

Figure 2 shows the same differences by fold for the eight and the six on rows with no labelled
event. The six's gain comes mainly from the 2026-01 fold.

**Arms and design.**

- **Arms:** all fine-tune the stored arm A pretrained weights of experiment 425 (learning rate
  2e-4, seeds 0 to 2, folds 2025-10, 2025-12, 2026-01, 2026-03, 2026-06). R0 is experiment 425's
  stored fit. R25 and R50 put 25% and 50% of each step's loss weight on ERA5-and-CAMS rows. Rcold
  does as R25 but draws replay rows only from cold nights (sun more than 5 degrees below the
  horizon, 2 m temperature of 3 degrees Celsius or below; about 4.7% of the replay rows). This is a
  deviation from the brief's decile definition, chosen to match the scored slice. The slice is
  called "cold nights" because it covers every month.
- **Step and loss:** an epoch is one pass over the ENS training rows in batches of 16,384 ENS rows
  (R0's steps). Each step adds 5,461 (R25, Rcold) or 16,384 (R50) replay rows, drawn with
  replacement. Step loss is `(1 - r)` times the recency-weighted ENS loss plus `r` times the
  unweighted replay loss. Early stopping reads the unweighted ENS held-out weeks only. This is a
  loss share, not the brief's row share per batch.
- **Replay rows:** the fold's pretraining rows with the held-out weeks and purge excluded and both
  standing cuts applied; the same 46 inputs, with the `std_*` columns set to the fold's median ENS
  spread by lead band. ERA5-and-CAMS weather is not available at forecast time; it enters training
  only.
- **Fits and cost:** 45 fits plus one reproduction fit, about 65 minutes with one GPU process
  (peak 3.2 GB, one feeder core). Predictions, weights and logs are in the experiment's private
  folder `work/`. Code reused from experiments 397, 418, 425, 454, and 470.
- **Reproduction check:** the R0 refit (2026-01, seed 0) matches the stored fit: largest quantile
  difference 1e-5, 10 epochs each, CRPS 4.5462 at the eight and 4.8854 at the six for both.
  Passed.
- **Production cost:** serving is unchanged. The monthly refit needs the ERA5-and-CAMS table and
  1.33 times (R25) or 2 times (R50) the rows per epoch.

**Minimum detectable effect** (printed before any contrast; points of capacity). Five single seeds
of arm A give a seed standard deviation of 0.052 (eight), 0.070 (six, no labelled event), 0.105
(cold nights, eight), and 0.147 (cold nights, six). For two arms of three seeds each the minimum
detectable effect is 0.083, 0.112, 0.169, and 0.236.

**Headline numbers** (CRPS in points of effective capacity, 1.15 widening; difference against R0
with the 27-test Bonferroni interval; Bonferroni count 27).

| Row set | R0 | R25 − R0 | R50 − R0 | Rcold − R0 |
|---|---|---|---|---|
| Eight, all rows | 4.764 | +0.006 (−0.023, 0.037) | +0.031 (−0.012, 0.078) | +0.037 (0.013, 0.065) |
| Six, no labelled event | 3.517 | −0.075 (−0.106, −0.048) | −0.088 (−0.133, −0.048) | −0.049 (−0.083, −0.017) |
| Six, real events (NRA) | n/a | +0.053 (−0.030, 0.132) | +0.034 (−0.081, 0.139) | +0.042 (−0.126, 0.206) |
| Eight, cold nights | 4.286 | −0.012 (−0.068, 0.046) | +0.026 (−0.058, 0.107) | +0.047 (−0.024, 0.120) |
| Six, cold nights | 5.955 | −0.100 (−0.190, 0.002) | −0.112 (−0.237, 0.016) | −0.025 (−0.128, 0.063) |
| Eight, 12 to 36 h | n/a | −0.019 (−0.064, 0.028) | +0.013 (−0.058, 0.091) | +0.012 (−0.025, 0.053) |
| Six, 12 to 36 h | n/a | −0.083 (−0.127, −0.040) | −0.104 (−0.169, −0.042) | −0.025 (−0.068, 0.017) |

- **Context:** A-pool (five-seed average) scores 4.680 (eight), 3.451 (six), 4.161 and 5.931 (cold
  nights, eight and six). R25 and R50 at the six (3.441, 3.429) match A-pool.
- **Predictions:** P1 (Rcold lowers cold-night CRPS by 0.05 at both) failed at the eight (+0.047)
  and is inconclusive at the six. P2 holds at the eight (R25 and R50 intervals lie inside ±0.083)
  and fails at the six (an improvement, so not equivalence). P3 holds: q99 pinball loss changes
  by at most +0.003 points (eight) and −0.012 (six).
- **Stop rule:** not met, because the cold-night minimum detectable effect (0.169 and 0.236) is
  above 0.05. The idea is not closed; the cold-night result is inconclusive.
- **Candidates:** none. Rcold fails the +0.02 non-inferiority gate at the eight (upper bound
  0.065); R25 and R50 do not lower cold-night CRPS with an interval excluding zero at both groups.
- **Validation loss:** replay lowers the ENS held-out negative log-likelihood (2026-01: R25 0.355,
  R0 0.364 to 0.369) but not CRPS at the eight.

**Tails** (raw coverage of the 1.15-widened bands, targets q95 0.95, q99 0.99, q05 0.05, q01 0.01).
At the eight, R0 has q95 0.945 and q99 0.989; the replay arms move by at most 0.007. At the six,
q95 rises from 0.918 (R0) to 0.925 to 0.928 (R25, R50). On cold nights at the six every arm stays
far below target (q95 0.787 to 0.804, q99 0.871 to 0.877), and Rcold raises cold-night q01 coverage
at the eight to 0.009 and at the six to 0.068 (target 0.01), so its lower tail worsens. Full
tables are in `work/scores/absolute.csv`.

**Avoidable-curtailment cost** at the six, in GBP million summed over the five scored folds, rows
with no labelled event, scaled limit f. R0: 1.87 (f = 0.9), 7.07 (0.75), 21.70 (0.5); manual
heuristic at f = 0.75 is 12.88. Paired difference against R0 with a 95% (not Bonferroni) interval:
at f = 0.9, R25 −0.29, R50 −0.41, Rcold −0.39, each statistically significant at the 5% level; at
f = 0.75 and 0.5 the intervals mostly include zero (f = 0.5: R25 −1.38, significant). The totals
carry wide intervals (R0 at f = 0.9: 0.18 to 4.90), as in experiment 470.

**Checks.**

- **Audit triggers:** two fired. Trigger 4: the six's CRPS gain comes mostly from 2026-01 (R25
  −0.237, R50 −0.326 in that fold; the other four folds average about −0.04 for both), so leaving
  that fold out probably removes the interval. Trigger 2: P1 failed with no tested explanation
  beyond the cold-night noise floor. No headline verdict rests on the six's gain.
- **Not run:** the median-minus-mean check and the mean-forecast error contrast, the load-shift
  event plots (no event contrast is claimed), and the example probabilistic forecast plots (the
  forecasts are fine-tunes of experiment 425's arm A, whose examples are on its page).
- **Caveats:** NGED's event labels are incomplete, so rows with no labelled event may hold
  unlabelled events. Three seeds and five folds give wide seed noise. The block bootstrap does not
  include seed variance. Rcold draws about 4 rows per cold-night row per epoch, so memorisation of
  the pool is possible and is not visible to the ENS-only early stopping.

**What to try next.** Repeat plain 25% replay with five seeds on a second set of folds before using
it at the six. Do not pursue cold-night-only replay. Leave the cold-night under-coverage at the six
to recalibration.
