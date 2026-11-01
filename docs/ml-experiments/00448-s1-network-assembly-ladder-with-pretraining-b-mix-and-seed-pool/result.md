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
  complex. Stored: 156 new fitted neural networks for the screen and 72 more for the final test
  (table under Compute), their forecasts, and the score tables, in this experiment's private
  folder only; experiments 404, 425, 428, 439, 446 are read and never written. More than one
  design is defensible (which components enter the ladder, which leave-one-out arms get fits, how
  pretraining treats the lag inputs), so this plan fixes each choice before any score is read.
  Production serving path: not touched (this repository holds none). Degradation rule: not
  touched. Callers: the scripts import the folders of experiments 425, 428, 439, 446, and the
  `studies` evaluation code.
hypothesis: >-
  The S1 network (pooled plain mixture density network, Primary embedding, ensemble mean and
  spread once, hand-made calendar columns, monthly refits, three origin lags plus the surprise at
  each lag) improves with each of three components added one at a time (ERA5-and-CAMS pretraining
  then ensemble fine-tuning; the fixed 50/50 B-mix of the S1 and lagless output distributions; a
  five-seed pool), and the network with pretraining and the pool together scores lower CRPS at
  days 1 to 10 than every one of its parts and than the monthly XGBoost recipe.
prediction: >-
  Written before any fit and before any score is read. P1. Pretraining lowers S1's CRPS by 0.03 to
  0.10 points at the signed-MW six and by 0.08 to 0.20 at the MVA-only eight (experiment 425, a
  network with no lag inputs: -0.050 and -0.124). P2. B-mix lowers CRPS on clean-lag rows by under
  0.03 and lowers q99 pinball loss on rows with a lag jump (experiment 404 found B-mix the
  strongest lag-trust design), so it passes only through its harm-row rule. P3. The five-seed pool
  lowers CRPS by 0.04 to 0.07 (experiment 446 T4: -0.056; the three-seed pool of S1 gave -0.042),
  mostly by ensembling. P4. The assembly PT+pool5 beats S1-pool5 by 0.02 to 0.08. P5. Removing the
  Primary embedding from the assembled network raises CRPS by 0.5 to 1.0 points (experiment 425:
  +0.851 and +0.837); removing the surprise inputs raises it by 0.05 to 0.15 (experiment 428's S1
  against R-inj: -0.106); removing pretraining raises it by the size of P1. P6. The assembled
  network beats the monthly XGBoost recipe at both groups.
decision_rule: >-
  Four screen folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 4, monthly
  refits, 1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity (a
  negative difference is a gain), per metering group (signed-MW six; MVA-only eight; D15 and D18
  report-only) and per Primary. Every screen number is interim and carries no 99% claim. Contrasts
  are paired by fold and seed with the month-block bootstrap of `studies.eval_protocol`, at a
  level Bonferroni-corrected over the contrasts. C1 pretrained S1 minus S1. C2 B-mix minus S1. C3
  pool of 5 seeds minus a single S1 seed (reported beside the pool of 3 seeds, so that ensembling
  is separated from the component). C4 the assembly minus the single best component, where the
  best component is fixed from the C1 to C3 point estimates at the signed-MW six before C4 is
  read, and held for the final test. Pretraining and B-mix each add a stage to the pipeline, so
  the smallest gain worth porting is 0.03 points at the signed-MW six (experiment 425's
  threshold); the pool adds no model and its threshold is zero, with its production cost reported.
  Pretraining advances if its C1 interval lies wholly below -0.03 at the signed-MW six or at the
  MVA-only eight, and is not above +0.02 CRPS at the other group. B-mix advances only through harm
  rows: on real lag-jump rows (event-cluster interval, with and without the largest event) its q99
  pinball loss is lower than S1's with an interval wholly below zero, and its clean-row CRPS is no
  more than +0.03 above S1's; it is a named option, not a default, whatever it scores. The pool
  advances if C3 is wholly below zero. Tail check units: pinball loss at q95 and q99 in points of
  effective capacity, clean rows, no more than +0.02 above S1. The three leave-one-out arms
  (embedding, surprise inputs, pretraining) and LOO-Bmix are read as checks: a component whose
  removal does not raise CRPS is reported as not supported inside the assembly. The minimum
  detectable effect is computed per contrast from the variance of the paired fold-by-seed
  differences (S1 seeds 0 to 5), printed before any contrast is read. The final test (5 folds by 3
  seeds, adding 2025-12) runs once, on the advancing components only, against every stored
  reference, and is not used to tune. The winning configuration of the experiment 447 sweep, if it
  has reported, is confirmed inside the assembled network at the end as one extra arm.
benchmark: >-
  S1 seeds 0 to 5 from experiments 428 and 439 (raw forecasts, 1.15); R-inj; the monthly XGBoost
  recipe, BG2, `manual_heuristic`, and `climatology` as read by experiment 425. Table P of
  experiment 435 is a last stage run only if the 435 page has merged when the screen finishes; if
  it has not, the page says so and reports raw and 1.15-widened scores only.
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

- **Base.** S1, experiments 428 (seeds 0 to 2) and 439 (seeds 3 to 5), four folds, on the same
  injected worlds. Stored networks and forecasts are reused; no S1 fit is repeated for the four
  screen folds.
- **Already in S1, so no ladder step.** ENS mean and spread once (397, 403), the Primary embedding
  (406, 421), hand-made calendar columns (423), monthly refits, the D15 and D18 cut. Their value
  is read from the leave-one-out arms.
- **How pretraining treats the lag inputs.** Experiment 425's stored pretrained networks have no
  lag inputs, and no lagless forecast exists for the ERA5-and-CAMS rows, so S1's surprise inputs
  cannot be built there. The pretrained S1 is therefore pretrained on the ERA5-and-CAMS rows with
  S1's input layout, the lag and surprise columns held at their standardised mean (zero), and then
  fine-tuned on ensemble rows with the lag and surprise columns live. A smoke test checks that the
  pretrained and fine-tuned S1 has the stored S1's input width and parameter count (8,585).
- **B-mix.** Mixture of S1's output distribution and the lagless network L0's, 50/50 as in
  experiment 446. L0 is stored for seed 0 only, so L0 is fitted for seeds 1 to 4 (16 fits).
  Experiment
  404's cross-fitted lagless tables are not used, because they cover training-period rows only.
  B-mix is a named option; the maintainer has not approved it for the first release.
- **Export gate (420, 424).** The gate is a trained component scored on signed output, and this
  ladder scores unsigned MVA, so CRPS cannot rank it. It is not a ladder arm. The page reports
  experiments 420 and 424's results beside the ladder as a descriptive item.
- **Table P (435).** Run as a last stage only if the 435 page has merged by the end of the screen.
- **Optional arm.** The event-tail option (U1n-2 of 446), as stored for seeds 0 to 2, reported
  beside the others and never the default.
- **Excluded**, as the brief lists: the explicit solar term, the shared calendar encoder, the
  shapes-and-responses network, the capacity prior, the 381 solar bundle, the widened head,
  the thermal-lag, solar-gain, look-ahead, and tariff bundles, E3 and day-window encoders, S2, S3,
  trust heads and attention over lags, the Student-t, tail-loss, and tail fine-tune arms, and the
  scale-only widening path. Experiment 422 (origin-hour variation) is excluded because no page
  supports it.

## Arms and the claimed gain of each component

| Arm | Content | Supporting experiment and claimed gain | New fits (screen) |
|---|---|---|---|
| S1 | stored reference, single seed | 428, 439: -0.106 against R-inj | 0 |
| S1-pool3, S1-pool5 | pool of 3 or 5 seed mixtures | 446 T4: -0.056 (3-seed pool: -0.042) | 0 |
| S1+PT | pretrained, then fine-tuned S1 | 425: -0.050 (six), -0.124 (eight), no lags | 40 (20 + 20) |
| S1+Bmix | 50/50 mixture of S1 and L0 | 404, 446: lag-jump tails | 16 (L0 seeds 1 to 4) |
| PT+pool5 | assembly without B-mix | this plan | 0 |
| PT+Bmix+pool5 | assembly with B-mix | this plan | 0 |
| LOO-embedding | assembly without the Primary embedding | 425: +0.85 | 40 |
| LOO-surprise | PT network with R-inj inputs | 428: +0.106 | 40 |
| LOO-PT | S1 with pool5 | as C1 | 0 |
| LOO-Bmix | PT+pool5 | as C2 | 0 |
| Event-tail | U1n-2 as stored, seeds 0 to 2 | 446: q99 on level changes 0.909 to 0.983, +0.076 CRPS | 0 |

## Compute

| Stage | Fits |
|---|---|
| Screen: PT (20 pretrain + 20 fine-tune), L0 seeds 1 to 4, LOO-embedding, LOO-surprise | 136 |
| Final test, fold 2025-12 only (seeds 0 to 4): S1, L0, PT, and the advancing leave-one-out arms | up to 40 |
| Final test, sweep-winner arm from experiment 447, 5 folds by 3 seeds, if it has reported | up to 32 |

- **Time.** Experiment 425's pretrain took about 45 s per network; S1 fits ran 60 epochs on 3.9
  million rows. Expect 10 to 14 GPU hours in all; the first fit is timed and the estimate corrected
  in the PR.
- **Slot.** One GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`,
  `free -g` before each launch (stop at 12 GB available). The sweep worker's separate GPU process
  is not touched.
- **Production cost.** The pool of 5 costs five forward passes per series per forecast; with 51
  ensemble members this is 255 passes, reported.

## Figures

The headline chart is CRPS by lead for the assembly, the ablations, BG2, the monthly XGBoost
recipe, `manual_heuristic`, and `climatology`. Further figures: tail coverage per arm, a
per-Primary table, cold winter nights, example day-ahead and 14-day forecasts, and six lag-jump
event plots across both metering groups with the unlogged-event caveat.

## Standing practices followed

Monthly refits; cleaned data (flagged and start-cut half-hours dropped, counted per fold); the
D15 and D18 cut at 2025-03-01 applied; every network saved in the data folder; stored forecasts
reused; tails, cold nights, and avoidable-curtailment cost reported; example forecasts and event
plots in the write-up; draft PR.
