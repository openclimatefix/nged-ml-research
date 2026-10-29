---
id: "00378"
slug: primary-demand-default-3-at-all-16-primaries-under-demand-v1
title: >-
  How does Primary demand research default 3, ported to all 16 Primaries, score against BG2, the
  manual heuristic, and climatology under the demand-v1 protocol?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts, fits, and small result files in this experiment's private folder only;
  experiments 87, 111, 152, 317, 319, 322, 374, 376, and 377 are read and never written. Production
  serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (refit the long-lead weights per Primary or pooled per group); the choice is fixed below. Callers:
  the scripts copy and import the folders of experiments 317, 319, 322, and 377, whose callers are
  all in those folders; no change under `packages/` or `src/`.
hypothesis: >-
  Primary demand research default 3 ported to all 16 Primaries is not worse than BG2 by more than
  0.20 points of effective capacity at days 1 to 10 at the six signed-MW Primaries, and it beats
  Primary demand research default 1 extended to 240 hours (experiment 377) at 168 to 240 hours at
  both groups.
prediction: >-
  Written before any score is read. P1. Signed-MW six, default 3 minus BG2 on the shared rows, days
  1 to 10, pooled: between -0.20 and +0.20 points. P2. Default 3 minus default 1 extended at 168 to
  240 hours is below zero at the signed-MW six (experiment 322 found -0.064 - 0.075 = about -0.14
  points against default 1) and between -0.20 and 0 at the MVA-only eight. P3. MVA-only eight
  days 1 to 10 minus BG2: between +0.30 and +0.90 points with D8, between -0.05 and +0.25 without.
  P4. The control gate holds: the port reproduces experiment 322's stored default 3 scores on the
  six within 0.05 points at 0.5 to 240 hours and at 168 to 240 hours.
decision_rule: >-
  Descriptive, no adoption. Contrasts registered before any score is read, CRPS in points of
  effective capacity on the shared rows, by the paired month-resampled bootstrap of
  `studies.eval_protocol` at 99% (20,000 resamples) with 95% beside it. C1: default 3 minus BG2,
  days 1 to 10 (BG2 only to 168 hours), signed-MW six pooled. C2: the same for the MVA-only eight,
  with and without D8, labelled "reactive power assumed zero". C3: default 3 minus default 1
  extended (experiment 377), 168 to 240 hours, per group. C4: default 3 minus climatology and minus
  the manual heuristic, per group. D15 and D18 are report-only and enter no tally. One verdict per
  group.
benchmark: >-
  BG2, climatology, and `manual_heuristic` as stored by experiment 87 on `fold_a_daily_gap_v1`;
  Primary demand research default 1 extended to 240 hours (experiment 377).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, mean absolute error of the median, and
  twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to 240 hours.
code: ml-experiments/00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1/
verdict: partly supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  signed-MW group: default 3 minus BG2 is -0.12 points at days 1 to 10 (99% interval -0.31 to +0.06)
headline: >-
  Verdicts per group, for reactive power zero, at days 1 to 10 on the demand-v1 protocol. Six
  signed-MW Primaries: supported (default 3 minus BG2 -0.12, -0.31 to +0.06; the registered margin
  was an upper end below +0.20). Eight MVA-only Primaries: partly supported (+0.63, +0.35 to +0.94
  with D8; +0.16, -0.02 to +0.35 without D8, all 99%). Default 3 beats default 1 in both groups
  (-0.03 signed-MW, -0.02 MVA-only, intervals below zero). D15 and D18: report only.
take_home_message: >-
  Default 3 scores level with BG2 at the six Primaries that report signed megawatts, and the solar
  summary and daylight blend lower its error from 6 to 10 days ahead by about 0.12 points against
  default 1. At the eight Primaries that report only apparent power it trails BG2 by 0.16 points
  without Marsh Lane (D8) and 0.63 with it. The tails are close to nominal at the signed-MW six and
  at the MVA-only seven without D8, and far off at D8, where the readings fall below the 5% quantile
  29% of the time. This holds for reactive power zero only.
lessons:
  - >-
    The solar summary below 168 h and the daylight blend beyond it lower the score by 0.03 points
    at days 1 to 10 and 0.12 at days 8 to 10 at the signed-MW six, and the gain is in the same
    direction at all 14 countable Primaries.
  - >-
    The MVA-only tails fail because of D8: without it, coverage at q01, q05, q95, and q99 is 0.021,
    0.070, 0.946, and 0.985, and with it 0.041, 0.097, 0.918, and 0.965.
  - >-
    Default 3 and BG2 differ in the tails: BG2's lower tail is too wide (q01 coverage 0.005 to 0.010)
    and default 3's is close to nominal at the signed-MW six.
advice:
  - >-
    Use default 3 as the base for 7 to 10 day demand experiments at the signed-MW Primaries, and
    compare any new arm with BG2 on `fold_a_daily_gap_v1` at days 1 to 10.
  - >-
    Treat D8 as its own problem: it contributes 0.48 of the 0.63 MVA-only gap to BG2 and fails every
    tail by a wide margin.
  - >-
    Run the rule-chosen fit of experiment 374 at 240 hours (stage C2) before reading the MVA-only
    gap to BG2 as final.
date: 2026-10-29
---

# Primary demand research default 3, ported to all 16 Primaries, scored against BG2 under demand-v1

**Default 3 scores level with BG2 at the six Primaries that report signed megawatts and trails it
at the eight that report only apparent power, almost all of that at Marsh Lane (D8).** A Primary is
a 33 kV to 11 kV substation, labelled D2 to D22 here. Default 3 is "Primary demand research default
3": the generalised additive model (GAM) stack of experiment 377 with two-number solar summary S
below 168 hours, and at 168 to 240 hours both S and a blend with the manual heuristic on daylight
half-hours (sun above 5 degrees). A score is the quantile score (twice the mean pinball loss over 11
quantiles, which equals the continuous ranked probability score, CRPS, up to the estimation from 11
quantiles), in points of effective capacity (percent of a Primary's capacity). A negative
difference means default 3 is better. BG2 is the best-guess forecast of experiment 87, which
production does not run. "MVA-only" Primaries report apparent power in megavolt-amperes (MVA) and no
signed megawatts (MW); every result there assumes reactive power is zero and uses the naive fit of
experiment 377. twCRPS is the threshold-weighted CRPS of the protocol.

![Figure 1: default 3 minus BG2 at days 1 to 10, per Primary and pooled](headline.svg)

![Figure 2: default 3 minus default 1 at days 1 to 10, per Primary and pooled](minus_default_1.svg)

**Signed-MW six: supported.** Default 3 minus BG2 at days 1 to 10, pooled over the six, is -0.12 (99%
interval -0.31 to +0.06, 20,000 resamples of whole months). The registered margin was an upper end
below +0.20, and +0.06 is below it. Days 8 to 10 give -0.14 (-0.33 to +0.01). Default 3 minus default
1 is -0.03 (-0.06 to -0.02) at days 1 to 10 and -0.12 (-0.19 to -0.05) at days 8 to 10.

**MVA-only eight: partly supported.** Default 3 minus BG2 is +0.63 (+0.35 to +0.94) with D8 and
+0.16 (-0.02 to +0.35) without. Prediction P3 (+0.30 to +0.90 with D8; -0.05 to +0.25 without) holds
on both ranges. D8 alone is +3.97 (95% interval +2.29 to +5.81), because D8 uses the donor spread in
every arm here. Default 3 minus default 1 is -0.02 (-0.05 to -0.00) with D8 and -0.02 (-0.04 to
-0.01) without.

**D15 and D18 (short history, report only).** Default 3 minus BG2 is -0.00 (-0.93 to +0.70) over 6
months. D15 scores worse than default 1 (+0.62, -0.20 to +1.32) because it takes the signed-MW fits
of S and the blend, and its readings differ from those Primaries.

## Results (quantile score in points of effective capacity; lower is better)

- **Absolute scores, days 1 to 10 (default 3, default 1, BG2, manual heuristic, climatology).**
  Signed-MW: D5 3.99, 4.04, 3.94, 4.35, 4.48; D9 2.45, 2.50, 2.51, 2.92, 2.99; D10 3.21, 3.25, 3.45,
  4.16, 4.31; D11 2.41, 2.42, 2.57, 3.41, 3.55; D14 2.18, 2.20, 2.39, 3.01, 3.47; D16 3.15, 3.18,
  3.24, 3.80, 4.20. MVA-only: D2 2.46, 2.47, 2.30, 2.86, 2.93; D3 1.66, 1.67, 1.75, 2.11, 2.33; D4
  2.43, 2.45, 2.51, 2.80, 2.83; D8 11.61, 11.65, 7.64, 8.54, 9.76; D12 7.47, 7.52, 6.75, 7.92, 8.83;
  D19 3.28, 3.29, 3.18, 3.79, 3.93; D21 2.34, 2.35, 2.38, 2.77, 2.91; D22 4.08, 4.10, 3.75, 4.71,
  5.47. Short history: D15 13.67, 13.05, 13.35, 22.59, 46.20; D18 2.44, 2.47, 2.65, 4.02, 7.11.
- **Per-Primary default 3 minus BG2 (95%, 2,000 resamples).** The interval excludes zero in default
  3's favour at D11 (-0.16), D14 (-0.21), and D18 (-0.21). It excludes zero against default 3 at D12
  (+0.72), D22 (+0.33), and D8 (+3.97).
- **Primaries won, days 1 to 10 (D15 and D18 not counted, denominator 14).** Against BG2 default 3
  wins 8 and loses 6; against the manual heuristic and against climatology, it wins 14 and loses 0.
  Against default 1 it scores lower at all 14, and the interval excludes zero at 10 of them.
- **Against the other references, pooled, days 1 to 10 (99%).** Minus the manual heuristic: -0.71
  (-1.06 to -0.39) signed-MW, -0.02 (-0.38 to +0.31) MVA-only with D8, -0.46 (-0.78 to -0.22)
  without D8. Minus climatology: -0.94 (-1.33 to -0.57), -0.46 (-0.91 to -0.09), -0.79 (-1.27 to
  -0.41).
- **Against experiment 377's stack without S (99%).** Default 3 minus the stack without S is -0.05
  (-0.09 to -0.03) signed-MW and -0.02 (-0.05 to -0.00) MVA-only at days 1 to 10, and -0.12 and -0.08
  at days 8 to 10. Default 1, here, is the stack with S below 168 h and the stack without S above.
- **Leave one Primary out (95%).** Default 3 minus BG2 at the signed-MW six stays between -0.10 and
  -0.15 whichever Primary is left out, with intervals that exclude zero only when D5 or D9 is left
  out. The MVA-only gap stays between +0.62 and +0.74 whichever Primary but D8 is left out, and is
  +0.16 (+0.02 to +0.30) without D8. Default 3 minus default 1 is negative with an interval below
  zero in every leave-one-out case in both groups.
- **Median error beside the CRPS contrasts (99%).** The mean absolute error of the median points
  the same way as CRPS in every row: default 3 minus BG2 -0.22 (-0.55 to +0.12) signed-MW, +0.41
  (+0.01 to +0.80) MVA-only with D8; default 3 minus default 1 -0.04 (-0.09 to -0.01) and -0.04
  (-0.07 to -0.01); minus the manual heuristic -1.28 and -0.72.
- **Tail calibration, days 1 to 10 (share of readings at or below each quantile, mean over the
  group; ideal 0.01, 0.05, 0.95, 0.99).** Signed-MW default 3 0.018, 0.058, 0.947, 0.983; BG2 0.010,
  0.045, 0.932, 0.975. MVA-only default 3 0.041, 0.097, 0.918, 0.965; BG2 0.005, 0.042, 0.955,
  0.995. MVA-only without D8, default 3: 0.021, 0.070, 0.946, 0.985. D8 alone: 0.183, 0.291, 0.721,
  0.826. D15 and D18: 0.034, 0.077, 0.867, 0.947. The lower tail at the signed-MW six is a little too
  low for q01 at D16 (0.036) and D5 (0.032). Pinball loss at q01, q05, q95, q99 (default 3 against
  BG2, signed-MW): 0.245 against 0.222, 0.739 against 0.686, 0.793 against 0.882, 0.262 against
  0.323. At the MVA-only eight the pinball loss at q05 is 1.27 against 0.86 for BG2, mostly D8
  (4.36).
- **twCRPS, days 1 to 10, mean over the group (default 3, BG2).** Signed-MW 0.324, 0.355; MVA-only
  0.206, 0.195.
- **Replicate months.** None: demand-v1 has no replicate period, and the fits of S and the blend
  fall back to the identity in the first fold month (2025-04) and are fitted on the trailing 365 days
  of earlier runs afterwards.

## Checks

- **Control gate.** The port reproduces experiment 322's stored default 3 at 168 to 240 hours on
  experiment 319's own rows at the six Primaries: 3.07596 points against 3.07596 (difference
  2e-10, largest single-quantile difference 1e-5), well inside the 0.05-point gate. The gate runs
  the same replay as the 16-Primary run, on experiment 322's rows and fold structure.
- **Median against mean forecast.** The median minus the mean is 0.00 at every Primary: the stored
  `mean` is the corrected mean around which stage 2 centres the quantiles, so the median equals it
  by construction and no independent mean-forecast contrast exists. The error of the median is
  reported beside the CRPS contrasts above, and agrees in direction in every row.
- **Triggers.** Trigger 1 did not fire (control gate passed). Trigger 2 did not fire (predictions P1
  to P4 hold). Trigger 3 did not fire (median error and CRPS agree in direction). Trigger 4 did not
  fire (no leave-one-out case flips a sign; the interval at the signed-MW six excludes zero in two
  leave-one-out cases, and the headline 99% interval does not). Trigger 5 did not fire. Trigger 6 did
  not fire (the pooled signed-MW interval is 0.37 wide against 0.42 in experiment 377). Trigger 7
  applies, because later experiments will use default 3 as their base; a fresh Sonnet audit is
  linked in the pull request.

## Caveats and what to try next

- Reactive power is assumed zero at the MVA-only Primaries, and the naive fit of experiment 377 is
  used at all eight. The rule-chosen fit of experiment 374 at 240 hours (stage C2) was not run.
- D8 uses the donor spread in every arm here, so its row differs from the own-residual arm in
  experiment 377 (3.97 against +1.61 for the stack without S against BG2). It is unproven whether the
  generator explains the steps and the gap at D8.
- The fits of S and of the blend are pooled within a group (signed-MW six, MVA-only seven) and
  applied to D8 (MVA-only fit), D15, and D18 (signed-MW fit). The first fold month has no fit rows
  and falls back to the identity.
- The sunny-half-hour, Christmas, and load-shift slices of experiment 377 were not recomputed here.
- Experiment 378 reads the stage-2 forecasts of experiment 377 and its stored ensemble arrays, so
  an error there carries into every number above.
