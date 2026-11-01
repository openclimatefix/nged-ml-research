---
id: "00448"
slug: s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool
title: >-
  Does adding pretraining, the B-mix output, and a five-seed pool, one at a time and then together,
  lower the S1 network's CRPS, and does the whole beat its parts and the references?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: about 190 fitted neural networks (listed under Compute), their forecasts, and
  the score tables, in this experiment's private folder only; experiments 404, 425, 428, 439, 446
  are read and never written. More than one design is defensible (which components enter the
  ladder, which leave-one-out arms get fits, whether pretraining sees the lag inputs), so the arms
  below fix each choice before any score is read. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Callers: the scripts import the folders of
  experiments 425, 428, 439, 446, 420, and the `studies` evaluation code.
hypothesis: >-
  The S1 network (pooled plain mixture density network, Primary embedding, ensemble mean and spread
  once, hand-made calendar columns, monthly refits, three origin lags plus the surprise at each lag)
  improves with each of three components added one at a time (ERA5-and-CAMS pretraining then
  ensemble fine-tuning; the fixed 50/50 B-mix of the lagged S1 and lagless output distributions; a
  five-seed pool), and the network with the first and third together (pretrained, pooled) scores
  lower CRPS at days 1 to 10 than every one of its parts and than the monthly XGBoost recipe.
prediction: >-
  Written before any fit and before any score is read. P1. Pretraining lowers S1's CRPS by 0.03 to
  0.10 points at the signed-MW six and by 0.08 to 0.20 at the MVA-only eight (experiment 425, a
  network with no lag inputs: -0.050 and -0.124). P2. B-mix lowers CRPS on rows with clean lags by
  under 0.03 and lowers q99 pinball on rows with a lag jump (experiment 404 found B-mix the
  strongest lag-trust design). P3. The five-seed pool lowers CRPS by 0.04 to 0.07 (experiment 446
  T4: -0.056, mostly ensembling), whether it is applied to S1 or to the pretrained network. P4. The
  full assembly (pretrained, B-mix, five-seed pool) beats S1 with the pool alone by 0.02 to 0.08.
  P5. Removing the Primary embedding from the assembled network raises CRPS by 0.5 to 1.0 points
  (experiment 425: +0.851 and +0.837); removing the surprise inputs raises it by 0.05 to 0.15
  (experiment 428's S1 against R-inj: -0.106); removing pretraining raises it by the size of P1.
  P6. The assembled network beats the monthly XGBoost recipe at both groups and trails BG2 at D15
  and D18 by less than experiment 425 found (7.970 against 7.457).
decision_rule: >-
  Four screen folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 4, monthly refits,
  1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity, per metering
  group (signed-MW six; MVA-only eight; D15 and D18 report-only) and per Primary. Every screen
  number is interim and carries no 99% claim. Contrasts C1 to C4 are paired by fold and seed with
  the month-block bootstrap of `studies.eval_protocol`, at a level Bonferroni-corrected over the
  contrasts: C1 pretrained S1 minus S1; C2 B-mix minus S1; C3 pool of 5 seeds minus a single S1
  seed (reported beside the pool of 3 seeds, so that ensembling is separated from the component);
  C4 the assembly minus the best single component. Each of C1 and C2 adds a stage to the pipeline,
  so its smallest gain worth porting is 0.03 points at the signed-MW six, the threshold of
  experiment 425; the pool adds no model and its threshold is zero, with the production cost
  (passes per forecast) reported. A component advances to the final test if its interval is wholly
  below the threshold at the signed-MW six or at the MVA-only eight and it is not worse than +0.02
  at the other group or in clean-row tail pinball. The three leave-one-out arms are read as
  checks: a component whose removal does not raise CRPS is reported as not supported inside the
  assembly. The minimum detectable effect is 1.96 times the square root of 2, times the seed
  standard deviation of S1 over its six stored seeds, divided by the square root of 5, printed
  before any contrast is read. The final test (5 folds by 3 seeds, adding 2025-12) runs once, on
  the advancing components only, against every stored reference, and is not used to tune.
benchmark: >-
  S1 seeds 0 to 5 from experiments 428 and 439 (raw forecasts, 1.15); R-inj; the monthly XGBoost
  recipe, BG2, `manual_heuristic`, and `climatology` as read by experiment 425; the unfitted B-mix
  of experiment 446 for seed 0. Table P of experiment 435 is applied as a last stage only if the
  435 page has merged.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99;
  cold winter nights; sunny half-hours; avoidable-curtailment cost; clean rows and rows with a lag
  jump, with and without the largest event; the encroachment sweep of experiment 392 for the final
  arm. Per group and per Primary.
---

**Take-home message (plan, not a result): this experiment tests which of the cheap add-ons to the
S1 network earn their place, one at a time and together, and what the combination costs in
production.** S1 already holds the ensemble mean and spread, the Primary embedding, the calendar
columns, monthly refits, and the lag-and-surprise inputs, so the new components are pretraining,
B-mix, and the seed pool. All numbers will be interim, from four test months, until the final test.

## Question

Experiment 425 assembled a lag-free pooled network from the components that won earlier. Since
then the neural Discuss session chose S1 (confirmed by experiment 439) as the base. This experiment
adds the remaining supported components to S1 one at a time, then together, and compares with the
references.

## Plan against the latest evidence

- **Base.** S1, experiment 428 (seeds 0 to 2) and 439 (seeds 3 to 5), four folds. Stored networks
  and forecasts are reused; no S1 fit is repeated for seeds 0 to 4.
- **Already in S1, so no ladder step.** ENS mean and spread once (397, 403), the Primary embedding
  (406, 421), hand-made calendar columns (423), monthly refits, the D15 and D18 cut. Their value is
  read from the leave-one-out arms.
- **Components added.** Pretraining (418, 425); B-mix (404, 446); the seed pool (446 T4); the
  export gate (420, 424) at the eight, applied to the forecast; table P (435) only after merge.
- **Optional arm.** The event-tail option (U1n-2 of 446), reported beside the others, never the
  default. It exists for seeds 0 to 2 and is not refitted.
- **Excluded**, as the brief lists: the explicit solar term, the shared calendar encoder, the
  shapes-and-responses network, the capacity prior, the 381 solar bundle, the widened head,
  the thermal-lag, solar-gain, look-ahead, and tariff bundles, E3 and day-window encoders, S2, S3,
  trust heads and attention over lags, the Student-t, tail-loss, and tail fine-tune arms, and the
  scale-only widening path.
- **Unlisted or unsupported components.** Experiment 422 (origin-hour variation) and 419's solar
  arm are excluded because their pages do not support them.

## Arms

| Arm | Content | Fits |
|---|---|---|
| S1 | stored reference, single seed | 0 (stored) |
| S1-pool5 | five-seed pool of S1 mixtures | 0 (stored) |
| S1-pool3 | three-seed pool | 0 (stored) |
| S1+PT | S1 inputs, ERA5-and-CAMS pretraining, ENS fine-tune | 20 pretrain + 20 fine-tune |
| S1+Bmix | 50/50 mixture of S1 and L0 distributions | 16 (L0, seeds 1 to 4) |
| S1+gate | signed-output export gate at the eight | 0 |
| S1+PT+Bmix+pool5 | assembly | 0 beyond the above |
| LOO-embedding | assembly without the Primary embedding | 20 + 20 |
| LOO-surprise | assembly with the surprise inputs removed (R-inj with PT) | 20 + 20 |
| LOO-PT | S1 with Bmix and pool5 | 0 |
| Event-tail | U1n-2 as stored (seeds 0 to 2) | 0 |

## Compute

- **Fits.** About 100 new fits for the screen (each pretrain about 45 s, each S1 fine-tune a few
  minutes; experiment 428's S1 fits ran 60 epochs on 3.9 million rows) and about 40 for the final
  test (5 folds by 3 seeds, pretrained and its removal-checks). Expect 10 to 14 GPU hours.
- **Slot.** One GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`,
  `free -g` before each launch (stop at 12 GB available). A second process (the demand
  hyperparameter sweep, experiment 447) runs in parallel and is not touched.
- **Production cost.** The pool of 5 costs five forward passes per series per forecast; with 51
  ensemble members this is 255 passes, reported.

## Standing practices followed

Monthly refits; cleaned data (flagged and start-cut half-hours dropped, counted per fold); the
D15 and D18 cut at 2025-03-01 applied; every network saved in the data folder; stored forecasts
reused; tails, cold nights, and avoidable-curtailment cost reported; example forecasts and event
plots in the write-up; draft PR.
