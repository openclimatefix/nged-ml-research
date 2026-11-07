---
id: "00646"
slug: per-ens-member-neural-demand-networks-pooled-quantiles
title: >-
  Does running the demand networks once per ensemble member and pooling the 51 outputs beat giving
  them the ensemble mean and spread?
theme: calibration-and-uncertainty
also_relevant_to:
  - weather-products-and-blending
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores fitted fine-tune weights and forecasts for about eight arms in the private
  data folder only; (2) more than one design could test the question (spread input kept or zeroed,
  training members resampled each epoch or fixed, pooling by mixture or by quantile average), so
  this plan fixes one primary design and reports the others as post hoc; (3) it spans code whose
  callers cannot be named without searching (experiments 397, 559, 560, 574, 591, 597, 608, 610,
  612, 613, and 614); (4) it touches no serving path and no degradation rule; (5) it changes no
  contract. Ceiling 6 GPU-hours (process hours); ask before crossing.
hypothesis: >-
  Running K1 and K3 once per ensemble member, and pooling the 51 member outputs into one mixture,
  improves the continuous ranked probability score (CRPS), the upper-tail coverage, and the
  event-free scores over the networks that read the ensemble mean and spread, without raising the
  share of an injected load-shift that the forecast follows.
prediction: >-
  Written before any score. P1: arm B (stored weights applied to each member, no refit) is worse
  than the seed-matched arm A by at least 0.05 points of capacity in CRPS. The weather encoder was
  pre-trained on single ERA5 and CAMS estimates, so it may accept a member, but the mixer was fitted
  on means and spreads, so the mixer is where B is expected to fail, most in B0 (zero spread). P2:
  arm C's event-free CRPS is lower than arm A's by 0.01 to 0.06 points at days 6 to 10, where the
  members differ most, and differs from A's by less than 0.03 points in either direction at days 1
  and 2; a rise of 0.03 points or more at days 1 and 2 is a miss. Experiment 397 found the member
  scheme within 0.05 points of the summary scheme for a plain network, and experiment 192 found no
  gain from member temperatures on cold nights, so the expected result is a gain below the bar.
  P3: arm C raises q95 and q99 coverage by 0.01 or more at days 6 to 10. P4: arm C's follow ratio is
  within 0.03 of arm A's.
decision_rule: >-
  Comparator: the ungated seed-matched arm A (K2 seed 0 for K1, K3 seed 0 for K3; seeds 0 to 2 once
  seeds 1 and 2 run). Score: event-free CRPS pooled over the Primaries without D15 and D18, days 1
  to 10; every other cut is descriptive. Material gain: the arm's CRPS is lower than A's by at
  least 0.0006 of capacity (0.06 points, the minimum detectable effect that experiment 591
  printed and experiment 613 used) in at least two of three fold groups, and the paired
  month-block interval of the pooled difference over the five months excludes zero. A fold group
  is one season group of months (2025-10 and 2025-12; 2026-01 and 2026-03; 2026-06 alone). The
  width factor is the same for the arm and for A (see Width). The follow ratio is measured on the
  injected bank, for A and for the arm, only for an arm that meets the CRPS condition; a follow
  ratio more than 0.03 above A's removes the claim. Variant choice: on seed 0, B0 versus B1 and C0
  versus C1 are chosen by pooled event-free CRPS on all five months, and the chosen variant is the
  one carried to seeds 1 and 2. Stop rule: if no seed 0 arm B or C meets the CRPS condition for
  K1 or K3, the experiment stops and reports. Families for Bonferroni: (1) 4 arm-versus-A CRPS
  contrasts at seed 0 (B and C, K1 and K3), (2) the same 4 on 3 seeds if run; the count is
  reported beside each. The experiment adopts nothing.
benchmark: >-
  Arm A: the seed-matched ungated forecasts from experiments 612 and 613's stored fits (the decision
  comparator), and experiment 614's export (five-seed mean, K1 gated) as a reference only.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS, MAE of the median, twCRPS, q05 to q95 coverage, and upper-tail exceedance, by lead band
  (day 1, day 2, days 3 to 5, days 6 to 10) and by Primary, pooled and without D15 and D18, on all
  rows and on event-free rows; injected-shift follow ratio and safe-side share as in experiment 613.
---

**Plan only. The page is rewritten with the results when the arms have run.**

## Step 0: how the ensemble weather enters today

**K1, the class 2 candidate, and K3 read the ensemble as a mean and a spread, never as members.**
The network applied once per forecast run gets, for each valid time, the 51 members' mean of seven
weather columns (shortwave, longwave, temperature, dew point, surface pressure, log precipitation,
10 m wind speed) in columns 0 to 6 of the 37 lag-free inputs. The members' population standard
deviation of the same seven columns sits in columns 7 to 13. Both come from the per-run summary
files that experiment 397 built (`ens_summary/init=<date>.parquet`), from the stored 51-member
frames from 2025-07-01 and from an archive rebuild before that.

**The weather encoder sees only the mean.** The standard deviations reach only the fine-tune
mixer, which also gets the two lead columns. K1 adds the day lag and week lag encoders and the
gated band G4; K3 has no lags.

**Training is in three steps** (`e00559_model.py`, `e00574_fit.py`):

- pre-training on ERA5 and CAMS rows, where the weather columns hold a single estimate and the
  spread columns do not exist, so the pre-training mixer ignores spread;
- step 2, which fits a fresh mixer (`mixer_ft`) on every ensemble row up to the fold's first
  cutoff, with the pre-trained encoders frozen; this mixer reads the seven spread columns and the
  two lead columns, so it has seen means and spreads;
- the ENS fine-tune on mean-and-spread rows from the 365 daily runs, refitted on days 0 and 14 of
  each test month with the 7-day embargo E3 for K1 (K3 keeps experiment 613's setting, checked
  before the first fit).

**Consequence for the plan.** Only the pre-trained weather and datetime encoders (ERA5 and CAMS,
single estimates) are free of ensemble spread, so arm C reuses the pre-trained weights and refits
step 2 and the ENS fine-tune on member rows. The K1 fine-tune also uses the lag-only synthetic
load-shift rows and the B1 columns of experiment 591's chain; arm C keeps both. Experiment 574 found
extra relative-spread columns unhelpful, so C0 against C1 also tests whether the base seven spread
columns are worth keeping. The "365 daily runs" and "about 10 seconds a fit" figures are taken from
the logs of experiment 613 (one K1 fit took 7 to 11 seconds) and are checked in the code review.

## Arms

- **A.** Current K1 and K3 from the roster export, scored as stored (K1 with the G4 gate). The
  seed-matched control is the ungated seed 0 forecast (K2) from experiments 612 and 613's fits.
- **B.** Per-member inference only: the stored weights applied to the 51 members' inputs, pooled.
  B1 gives each member the run's own standard deviation in the spread columns, so the inputs keep
  their trained format. B0 sets the spread columns to zero. The result says whether a network
  trained on summaries can take a single member at all.
- **C.** Per-member training: refit the ENS fine-tune on rows with one member's seven weather
  columns, drawing a different member for each row each epoch. C1 keeps the run's standard deviation
  as the spread input, C0 sets it to zero for training and test. Both pool the 51 outputs at
  inference. K1 and K3, seed 0 first, then seeds 1 and 2 for the winning variant. Runs before
  2025-07-01 have only the control member stored, so they train on it as BG2 and experiment 397
  do. This is a limit on arm C.

**Pooling** follows experiment 397: an equal-weight mixture of the 51 members' mixture components,
with the 11 quantiles read from the pooled cumulative distribution function, then the D5 offset as
fitted; the offset was fitted on mean inputs, which is an assumption for arm B. **Width.** The
pooled mixture is wider than a single pass, so each arm, A included, gets one width factor per
Primary group (six MW, eight MVA) and lead band, about the median, chosen to bring the q05 to q95
coverage closest to 0.90 on earlier months only; 2025-10 has no earlier month and uses the factor
chosen on 2025-12 (flagged as lookahead for that month alone). This replaces the fixed 1.15 of BG2
and gives every arm the same calibration freedom. The unscaled numbers are reported beside the
scaled ones, and P3's coverage is read on the unscaled bands. K1's gate is not applied in the primary
reading (the K2 reading); a secondary reading unions the pooled band with the pooled lags-removed
band on the stored gate days if the primary reading shows a gain.

## Scoring and standing practices

- **Rows.** The five development months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) and the
  `experiment_470_intersection_v1` rows. The July to October 2026 window is never opened. Event-free
  rows use experiment 637's frozen detector, with the verified-event flags as a second reading.
- **Cleaned data and the short-history cut** follow the standing practices, since the arms reuse
  experiments 612 and 613's rows. D15 and D18 rows before 2025-03-01 are dropped.
- **Tails and cost.** q01, q05, q95, and q99 coverage and pinball loss, cold winter nights, and the
  avoidable-curtailment cost are reported beside CRPS, per Primary and pooled.
- **Injected shifts.** The follow ratio and safe-side share are measured on experiment 613's
  injected bank for arm A and for any arm meeting the CRPS condition, because the bank multiplies
  inference by 51 and arm C needs its own refits on shifted rows (counted in the compute below).
- **Cadence.** Monthly refits (days 0 and 14 of each month), as experiments 612 and 613 used.
- **Compute.** Inference is 51 times the rows of arm A's, and the input build is the cost, not the
  forward pass. Arm C refits take about 10 seconds each: 5 months, 2 refits, 2 networks, 2 variants
  is 40 fits for seed 0, plus step 2 refits and, for a qualifying arm, the injected-bank refits. The
  estimate is under 3 GPU-hours in total, against the 6 GPU-hour ceiling; `nvidia-smi` is checked
  before each run and one process runs at a time.
- **Seeds.** Seed 0 first, then seeds 1 and 2 for the chosen variant, three seeds in total.
- **Training rows.** Runs before 2025-07-01 hold only the control member, so the 2025-10 and
  2025-12 fold group trains mostly on control-member rows; results are tabulated by fold group so
  this limit is visible.
- **Intervals.** Month-block bootstrap over the five months and the Primaries, as experiment 613.
  The headline chart shows each arm's difference from A with its interval; at most two panels.
- **Event-free rows.** Experiment 637's detector was applied to the development months by
  experiment 645, so it applies here; the verified-event flags are the second reading.
- **Reviews.** One Sonnet plan review, one Sonnet code review before the first number that will be
  quoted, one Sonnet write-up review against the printed report, one Sonnet prose review.

## Earlier work this plan follows

- Experiment 397 ran a plain network on the control member and applied it to 51 members, and found
  it within 0.05 points of the summary scheme. This plan trains on members, which 397 did not.
- Experiments 620 and 633 run BG2's XGBoost once per member and pool 51 outputs with a spread factor
  of 1.15, and experiment 125 treated members one at a time; arm C is the neural analogue.
- Experiments 187 and 192 ran the 51 members through part of the demand model (the solar term, and
  the temperature term on cold nights) and found a 2.2% daylight gain and no cold-night gain.
- Advice from earlier results: report a score for every member as well as for the member average
  when testing a weather correction, and train a calibrator on pooled members or leave it out.

## Fold groups

The brief asked for "at least two of the three fold pairs". The five development months have no
natural three pairs, so this plan reads them as three season groups (autumn and early winter,
winter and spring, summer). Experiment 612 names only the first fold pair (2025-10 and 2025-12), which
matches the first group here; neither 612 nor 613 defines the others. The manager confirmed this
reading before any score.
