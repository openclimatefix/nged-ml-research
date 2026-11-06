---
id: "00603"
slug: nra-lag-perturbation-in-pretraining-and-split-mixer
title: >-
  Does perturbing the lagged power in pre-training and step 2, and keeping the first half of the
  mixer through every training stage, make the demand network follow a load-shift less than the
  network of experiment 584 (C1)?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: 6 new pre-trainings, 8 new step-2 fits, and 8 new fine-tunes per fold month, refit
  day and seed (all small neural networks in the private data folder), plus tables and figures.
  More than one design could test the hypothesis, and the arms follow the maintainer's brief. The
  experiment wraps the code of experiments 574 and 584. Serving path and degradation rule: not
  touched (this repository holds none). Compute estimate 5 to 7 GPU-h against an 8 GPU-h ceiling (ask before crossing).
hypothesis: >-
  Showing the network synthetic lag-only load-shifts during pre-training and step 2, and not only in
  the step-3 fine-tune, and keeping the first half of the mixer (stage 1, which reads the weather,
  datetime and lag embeddings) frozen from pre-training onwards, makes the network trust unusual
  lagged power less. Coverage of the normal running arrangement (NRA) after an injected load-shift
  rises, and less of a load-shift in the training power is absorbed at the next refit.
prediction: >-
  Written before any score. P1: at least one of J1, J2, and J3 raises the share of rows with the NRA
  inside the q05-q95 band, 0 to 7 days after the end of an injected 15 or 30 day load-shift, by 0.02
  or more over J0 (the stored C1), with follow (the share of injected depth that the median
  follows) no higher than 0.108. P2: J3 absorbs at least 0.03 less of an injected shift's depth,
  7 days after its end, than J0's 0.168 (experiment 595 method), because stage 1 is frozen. P3: the
  event-free CRPS and event-free coverage of every arm stay within the minimum detectable effect of
  J0 at the eight MVA Primaries and the six MW Primaries (D15 and D18 reported only). P4: J3 minus
  J3c is positive on coverage after the end, so the perturbation matters beyond the split. P5: a raw
  lag-encoder path that bypasses stage 1 (J4 and J5) lets the freshly fitted stage 2 re-read the lag
  embedding directly, so J4 and J5 absorb no less than J3 (experiment 595 method), and J4b, which
  has no raw lag path, keeps J3's absorption (within 0.03). P6: J4 and J5 cover the NRA after the
  end on no more rows than J3 and follow at least as much; J4b matches J3 on both. Known
  risk: pre-training sees ERA5 and CAMS analysis weather, whose expectations are sharper than an
  ensemble's, so a distrust learnt there may not transfer to the ensemble stages.
decision_rule: >-
  Every event metric is descriptive (event-block bootstrap), as in experiment 584. Counted tests:
  stated in the write-up as a number, with the minimum detectable effect beside every contrast. A
  perturbation or split is worth porting only if it passes P1 and P3 together. Passing P2 is
  reported separately.
benchmark: >-
  J0, the stored C1 forecasts of experiment 584, and the stored absorption of experiment 595.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the NRA inside q05-q95 by phase (before, days 0-3, day 4 to the end, 1-7 days after),
  safe-side share, follow, band width, absorption by phase, CRPS at 0-24 and 24-120 h.
---

# Planned

## Arms

All arms start from experiment 584's recipe (C1) and share its rows, refit days 0 and 14, seeds 0
and 1, and five development test months. The core is the concatenation `[hw, hd, hl, emb]` of the
weather, datetime, and lag encoder outputs and the Primary embedding (width `wide`). The synthetic
shifts are experiment 581's: 5% of time, lag-only, depths 15/30/50% of capacity, lengths 3/7/15/30
days, random sign, clean targets.

- **J0**: stored C1 (perturbation in step 3 only; unsplit mixer).
- **J1**: unsplit mixer; perturbation in pre-training and step 3.
- **J2**: unsplit mixer; perturbation in pre-training, step 2, and step 3.
- **J3**: split mixer, perturbation in all three stages. Stage 1 is `Linear(wide, 64)`, GELU,
  dropout, learnt in pre-training and frozen (and held in evaluation mode) in steps 2 and 3. Stage 2
  is a fresh `mlp(hidden 64, 1 hidden layer, 9 outputs)` on `[h1, spread, lead]` in steps 2 and 3
  (on `h1` alone in pre-training). The total depth equals the unsplit mixer's.
- **J3c**: as J3, perturbation in step 3 only.
- **J4**: as J3, but stage 2 reads `[h1, hw, hd, hl, spread, lead]`, the raw outputs of the three
  encoders beside stage 1's output.
- **J4b**: as J4, without the lag encoder: stage 2 reads `[h1, hw, hd, spread, lead]`.
- **J5**: as J3, with a residual stage 1: `h1 = core + g(core)`, where `g` is `Linear(wide, 64)`,
  GELU, dropout, `Linear(64, wide)`, so the dimensions match and the raw embeddings reach stage 2
  through the residual path. Stage 2 reads `[h1, spread, lead]` (width `wide` + 9).

J3, J3c, J4, J4b, and J5 perturb as stated and freeze stage 1 after pre-training. J4, J4b, and J5
perturb in all three stages.

## Stop rule

A pilot on one fold month first checks that perturbing in pre-training changes the network's lag
sensitivity (lag-ablation sensitivity before and after). J3 and J3c run whatever the pilot shows.
If J1 and J2 show no change in lag sensitivity or coverage against J0 in the pilot, J4, J4b, and
J5 run only if J3 beats J3c on absorption or coverage.

## Compute

Ceiling 8 GPU-h in total (ask before crossing); up to 3 GPU processes at a time (8 GB each,
`MemoryMax=8G`, `OMP_NUM_THREADS=1`) while `free -g` shows over 20 GB available. Estimate 5 to 7
GPU-h, dominated by the load-shift absorption runs (experiment 595 method) for J3, J4, J4b, and J5.

Plan only; the write-up replaces this page when the fits are scored. The arms and the compute
estimate are in the draft pull request.
