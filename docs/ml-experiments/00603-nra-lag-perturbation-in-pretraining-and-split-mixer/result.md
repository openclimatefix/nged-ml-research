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
  medium. Stored: 3 new pre-trainings, 4 new step-2 fits, and 4 new fine-tunes per fold month, refit
  day and seed (all small neural networks in the private data folder), plus tables and figures.
  More than one design could test the hypothesis, and the arms follow the maintainer's brief. The
  experiment wraps the code of experiments 574 and 584. Serving path and degradation rule: not
  touched (this repository holds none). Compute estimate 3 to 4 GPU-h against a 6 GPU-h ceiling.
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
  J3c is positive on coverage after the end, so the perturbation matters beyond the split. Known
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

Plan only; the write-up replaces this page when the fits are scored. The arms and the compute
estimate are in the draft pull request.
