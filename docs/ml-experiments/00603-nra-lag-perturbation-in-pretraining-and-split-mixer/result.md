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
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. Stored: pre-trainings, step-2 fits, and fine-tunes of 7 arms for one fold month
  (2025-12), small neural networks in the private data folder, plus tables and one figure. Cut by
  the manager to a one-month go/no-go. Serving path and degradation rule: not touched (this
  repository holds none). Under 2 GPU-h used against an 8 GPU-h ceiling.
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
verdict: not supported
best_arm: >-
  none: no arm beat J0 (the stored C1 of experiment 584) by 0.01 on coverage on day 4 to the end
  or 0-7 days after the end, at either the eight MVA Primaries or the six MW Primaries (2025-12
  only).
headline: >-
  Showing the network synthetic load-shifts in pre-training, with or without a split mixer, did
  not raise coverage of the normal running arrangement after an injected load-shift. Coverage was
  at or below J0's on day 4 to end and 0-7 days after the end in every arm, and the bands were
  narrower in all arms except J3c (December 2025, descriptive).
lessons: >-
  The tighter bands probably explain the lower coverage, because the median's error against the
  normal running arrangement fell by at most 0.007 (0.0578 for J3 against 0.0643 for J0, 0-7 days
  after the end, eight MVA Primaries); this was not tested. Load-shift absorption at the refit
  (experiment 595 method) was not measured.
take_home_message: >-
  Perturbing the lagged power earlier in training (pre-training, step 2) and freezing the first
  half of the mixer did not help: the band around the forecast got narrower and the normal running
  arrangement fell outside it on more rows after an injected load-shift, in the one fold month we
  ran. The extra paths that let the fresh output stage read the raw encoder outputs (J4, J4b, J5)
  did not rescue it. The next step for coverage is the width of the band, not where the lag
  perturbation sits in the training schedule.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not move the synthetic lag-only load-shifts into pre-training or step 2 of the P5 recipe to
    raise coverage after a load-shift; keep them in the fine-tune only (experiment 603 pilot,
    December 2025).
  - >-
    Do not freeze a first mixer stage after pre-training to make the demand network trust lags
    less; coverage fell at the six MW Primaries (experiment 603, J3 against J3c).
---

**Take-home message.** Showing the demand network synthetic load-shifts earlier in training, and
freezing the first half of its mixer, did not make the forecast band hold the normal running
arrangement on more rows after a load-shift. It held it on fewer rows. The bands were narrower, and the
median followed the shifted lags only slightly less, so the network is less cautious without being
much less misled.
This is a one-month go/no-go, so it stops the idea rather than measuring it finely. Verdict: not
supported.

We asked whether the demand network of experiment 584 (C1), which sees synthetic load-shifts only
in its last training step, would behave better if it saw them from the start. We also asked whether
splitting its mixer (the layers that combine the encoder outputs) into a kept first half and a
refitted second half, and letting the second half read the raw encoder outputs, would help. We added
fake load-shifts to the lagged power of every Primary in December 2025, and measured how often the
normal running arrangement (the power without the shift) stayed inside the forecast's q05-q95 band.
The method follows
[experiment 584](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00584-nra-candidate-pack-x6/result/),
which builds on the synthetic shifts of
[experiment 581](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00581-robust-multiweek-lag-encoder-x3/result/)
and the three-step recipe of
[experiment 574](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result/).

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

## Coverage by arm

![Coverage of the normal running arrangement by arm, phase, and metering group](coverage_by_arm.svg)

*Share of rows with the normal running arrangement inside the q05-q95 band, on injected 15 and 30 day
load-shifts, 24-120 h ahead, December 2025. Each arm has three dots (one per phase). Look at the
orange and blue dots: every arm sits at or below J0 at the six MW Primaries. The 0.90 target is the
nominal coverage. Pilot month only, so no intervals.*

## Results (pilot month 2025-12, descriptive)

- **P1 fails.** No arm raises coverage by 0.02 or more over J0 (day 4 to end: eight MVA Primaries J0
  0.893, J1 0.877, J2 0.852, J3 0.883, J3c 0.876, J4 0.873, J4b 0.857, J5 0.872; six MW Primaries J0
  0.830, J1 0.791, J2 0.755, J3 0.729, J3c 0.792, J4 0.744, J4b 0.693, J5 0.742). 0-7 days after the
  end: eight MVA J0 0.912 against 0.886 to 0.910 for the others; six MW J0 0.905 against 0.787 to
  0.890.
- **Bands are narrower.** Width of the 90% band at 24-120 h on day 4 to end, eight MVA Primaries:
  J0 0.267, J3 0.232; six MW Primaries: J0 0.209, J3 0.160.
- **Follow proxy.** The median's absolute error against the normal running arrangement (`ae50`) is
  only slightly lower for J2 to J5 (0-7 days after the end, eight MVA: J0 0.0643, J3 0.0578). The
  stored scorer has no follow ratio, so P1's follow clause was not tested.
- **P3 (event-free).** Event-free coverage at the six MW Primaries falls from 0.890 (J0) to 0.867
  (J3) and 0.854 (J4); at the eight MVA Primaries it ranges from 0.893 (J5) to 0.911 (J4). The 90% band's
  loss is lower than J0's for J3, J4b, and J5 at both groups.
- **P4 (J3 minus J3c).** Not positive: J3 0.883 against J3c 0.876 on day 4 to end at the eight MVA
  Primaries, but 0.729 against 0.792 at the six MW Primaries.
- **P2, P5, and P6 (absorption).** Not tested. The manager's cut stopped the run after the pilot,
  and no arm passed the filter that earns a refit-absorption run.
- **Later phase.** 7-14 days after the end, J1 (0.949) and J3c (0.961) exceed J0 (0.914) at the
  eight MVA Primaries. That phase is outside the decision rule.

## Checks

- Median minus mean forecast per site: not computed.
- The error of the median forecast beside the coverage contrast: `ae50` above.
- Triggers: trigger 2 (a result contradicting its own prediction) fired for P1 and no explanation
  has been tested beyond the narrower bands above, so a Sonnet audit is the manager's call.
  Triggers 1 and 3 to 7 did not fire.

## Caveats

- One fold month (December 2025), two seeds, injected shifts only, no intervals and no Bonferroni
  count. The fold pair 2025-10 and 2025-12 was not run, because no arm survived the filter.
- The real December 2025 pair, the tails, and the avoidable-curtailment cost were not scored.
- Pre-training sees ERA5 and CAMS analysis weather, so a distrust of lags learnt there may not
  transfer to the ensemble stages. The pilot cannot separate that from the other explanations.
- The pilot's pre-training perturbed the day lag of 3.6% of finite pre-training rows.
- Code reused from experiments 574, 581, 583, and 584.

## What to try next

- Widen the band where the lags look unusual (the gated band of experiment 584) rather than moving
  the perturbation earlier in training.
