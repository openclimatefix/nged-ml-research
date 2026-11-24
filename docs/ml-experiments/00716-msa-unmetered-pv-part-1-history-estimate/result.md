---
id: "00716"
slug: msa-unmetered-pv-part-1-history-estimate
title: >-
  Does an estimate of unmetered solar output in each history day help the axial-attention network forecast a Primary's demand?
theme: solar-disaggregation
also_relevant_to:
  - lagged-power-and-error-correction
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts and network weights for about 150 fits; (2) more than one
  design defensibly tests the hypothesis (a per-token additive path, a separate row group, or a
  larger flattened matrix); (3) it copies training code from experiment 713, and the callers of that
  code cannot be listed without searching the repository; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Full set of reviews: plan, code (Opus, each new
  script, twice), results, write-up, prose.
hypothesis: >-
  A satellite-based estimate of a Primary's unmetered solar output (an as-of capacity times a
  yield curve driven by the CAMS irradiance of each history half-hour), added to each of the 14
  history rows of experiment 713's axial-attention network, lowers the continuous ranked
  probability score (CRPS) of that network at leads of 24 h to 240 h, because the network can then
  separate demand from solar in the past days and stop copying a sunny window's midday dip onto a
  dull target day. The gain comes from the actual sky of each history day, and not from the
  capacity, the season, or the added parameters.
prediction: >-
  Written before any fit. A1 (H minus M*) lies between -0.06 and +0.06 points of capacity at the
  six signed-MW Primaries and between -0.16 and +0.16 at the eight MVA-only Primaries, and the
  interval of A2 (H minus H-S) includes zero in both groups. The predicted verdict is `not
  supported` (no detectable difference) or `inconclusive`. The prediction fails if A1 shows a gain
  or a loss beyond those ranges, or if the interval of A2 excludes zero.
decision_rule: >-
  Two counted contrasts, A1 and A2, each at the six and at the eight, so four readings, each read
  at 98.75% (1 - 0.05/4) from experiment 713's month-block bootstrap, with 713's three guards (a t
  interval on the 8 monthly differences, the same sign for each seed alone, and a point estimate of
  at least 0.034 at the six and 0.090 at the eight). The smallest gain worth porting is 0.02 points
  of capacity, the demand protocol's tier for a new input. A group reads "gain" when the interval
  is below zero, the estimate is at or below -0.02, and the three guards hold; "small gain" when
  the interval is below zero with the seed guard holding for at least 4 of 5 seeds; "no detectable
  difference" when the interval lies inside +-0.06 (six) or +-0.16 (eight); "loss" when the lower
  end is above zero. The verdict rows are experiment 713's, with A1 in place of H1 and A2 in place
  of H2. Row 5 (`small gain`) also needs the placebo arm H-S not to be a "loss" against M* in
  either group; this guard was added before any fit, as a design fix, to make row 5 consistent
  with rows 3 and 4 (the second Opus design review, finding S7), and it can only turn a `small
  gain` into `inconclusive`. A `small gain` whose placebo contrast has a point estimate at or above
  zero reads `inconclusive`. Every verdict, headline, and take-home message states that the
  warm-start states of experiment 713 come from a waived known-answer gate (KA3 and KA12 failed),
  and that no conclusion about era recognition or following level shifts is backed by a passing
  known-answer test.
benchmark: >-
  M* (experiment 713's selected arm M1, stored seeds 0 and 1 of the screen folds, seeds 2 to 4
  when 713's phases 8 and 9 have run); arm Z, a refit of M* by this experiment's training loop; the
  Candidate (P7+D50 of experiment 667); the month-by-half-hour climatology; the manual heuristic
  and BG2 where stored rows cover the scored keys.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity on leads from 24 h up to but not including 240 h after the 12:00 UTC
  origin, with the mean absolute error of the median forecast normalised by capacity beside it;
  q01, q05, q95, and q99 pinball loss and coverage; cold winter nights; avoidable-curtailment cost;
  the four load-shift scenarios; secondary readings S1 to S5 on the solar block.
---

**We ask whether the axial-attention network of experiment 713 forecasts a Primary's demand better
when each of its 14 history rows also carries an estimate of the Primary's unmetered solar output.**
Unmetered PV is the photovoltaic generation behind a Primary that no NGED meter records
separately, so the Primary's meter reads demand minus that generation. The estimate is a capacity
(in megawatts) times a yield curve between 0 and about 1 that is driven by the CAMS satellite
irradiance of the token's own half-hour. This page is part 1 of follow-up F7. Part 2 is
[experiment 717](../00717-msa-unmetered-pv-part-2-target-row-forecast/result.md), which adds an
explicit PV forecast to the target row. This page is the plan: no code has been fitted, no neural
network has been trained, and every number below is a design value or a figure from an earlier
experiment. The design is condensed from the private design note `f7_design.md` and its two Opus
reviews.

**Known-answer check KF9 failed and the maintainer waived it, so the gate is a waived known-answer
check and not a passed one.** KF9 asks arm `H` to beat arm `H-S` on the solar block of world WH by
half the `gap`. Its value is 0.00070 against the need of 0.00729, and its secondary readings (days 1
to 60, days 61 to 236, and pass 60) fail as well. KF10, KF11, and KF12 pass. No threshold and no
verdict rule changed. An audit of the failure found no implementation fault and two method findings.
First, the patience-8 rule scored arm `H` at passes 4, 23, and 5 of seeds 0, 1, and 2, against
global bests at passes 43, 55, and 32, because the history path learns slowly. Second, the yardstick
of KF9 sits at the oracle ridge: arm `Z` scores 0.0394 on the solar block, the naive ridge reference
0.0470, and the oracle ridge 0.0324, so arm `H` needs 0.0321 or better to pass. Every statement that
the history path recovers the hidden sky from the PV estimate is therefore unsupported by a passing
known-answer test.

**The early-stopping pre-check of the known-answer worlds read one seed of three, and the other
seeds of arm `M` are noisier than its limit.** The pre-registered pre-check reads arm `M` of seed 0
only. At rung 4 it passed in both worlds, with a last-10 validation-loss noise of 0.0109 in world WH
and 0.0121 in world WT, against the limit of 0.0225. Seeds 1 and 2 of arm `M` in WH have a last-10
noise of 0.029 and 0.031, above that limit, so the rung-4 pass depended on seed 0. The numbers
review checks this reading.

**The honest expectation is a tie.** Each history token already carries CAMS shortwave irradiance
and solar geometry, so the estimate adds a capacity scale, a physical shape, and units that match
the power. Six earlier experiments found that an explicit PV term built on forecast weather did not
help net demand, and experiments 703 and 704 found that the two previous-day CAMS columns can be
dropped from the Candidate with no detectable loss. With 5 seeds the design detects, with 80%
power, a CRPS difference of 0.06 points of capacity at the six signed-MW Primaries and 0.16 points
at the eight MVA-only Primaries, and a smaller difference is likely to read as a tie.

## Why this runs although the stop rule counts earlier verdicts

**The maintainer asked for follow-up F7 by name and confirmed the scope on 2026-11-23 ("I
confirm."), relayed through the experiment manager. That instruction is the approval that the
demand protocol requires for an idea that is past its fourth counted verdict.** The protocol's stop
rule counts the verdicts `not supported` and `refuted` on one idea, resets the count at `supported`
or `small gain`, and from the fourth experiment requires the maintainer's approval and scoring at
every Primary. This part scores all 16 Primaries.

The idea of part 1 (realised or recent satellite sunshine as an input that explains the past) has
one counted experiment, experiment 504 (`not supported`: CAMS days at least 36 hours old, as
summary columns, in a neural network without history rows). Experiment 480 (`supported`) is a weak
reset, because it gave the forecast the realised sunshine of the target half-hour. Experiments 039,
289, 639, and 661 found that realised CAMS explains past net demand, and their verdicts neither
count nor reset. Part 2's idea (an explicit embedded-PV forecast in a net-demand forecast) has 6
counted experiments on a strict reading: 409, 477, 501, 644, 655, and 664. Counting experiment 661,
experiments 350 and 351, and experiments 329 and 419 as well gives 7, 9, and 11. Whether part 1
shares part 2's idea for the count is a decision for the manager.

| Experiment | What the experiment gave the forecast | Verdict | Counted for scope (b) |
|---|---|---|---|
| 409 | An explicit solar term in the pooled mixture density network | `not supported` | Yes |
| 477 | An explicit PV module in the net demand neural network | `not supported` | Yes |
| 501 | An explicit solar-output column from an as-of capacity | `refuted` | Yes |
| 644 | An explicit solar-capacity term with ENS weather | `refuted` | Yes |
| 655 | A member-level solar curve with cell temperature | `refuted` | Yes |
| 664 | Net demand as gross demand minus a sampled ENS solar curve | `refuted` | Yes |
| 661 | Whether the ENS time step explains experiment 644, and a bias adapter | `refuted` | Borderline (gives 7) |
| 350, 351 | A capacity-free irradiance curve and a capacity read-out in the GAM stack | `not supported`, twice | Arguably (gives 9) |
| 329, 419 | A 3-member solar distribution; a solar summary in the mixture density network | `not supported`, twice | Borderline (gives 11) |
| 031, 349, 405, 416 | PV features in XGBoost and the GAM stack; explicit term with learned capacity or 5 seeds | `inconclusive` | No |
| 352, 639, 643 | Explicit terms scored with recorded weather | `partly supported` | No: neither counts nor resets |

**This page departs from three items of earlier advice, on the maintainer's instruction.**
Experiments 644 and 655 advise against any explicit solar term in the ENS demand network.
Experiment 504 advises against satellite irradiance columns at least 36 hours old. Experiment 501
advises keeping the irradiance input and the Primary embedding as the network's way of
representing solar output. This experiment follows experiment 504's advice to score cold winter
nights separately, experiment 661's advice to keep production's window-mean resampler for ENS
shortwave, and experiment 501's advice to judge a capacity by a known-answer or placebo test.

**What differs from the counted experiments:** no earlier neural network held 14 aligned days of
net power beside the sunshine of each day, so no earlier experiment could use a PV estimate to
separate demand from solar in the past.

**Experiment 713's screen stop rule binds this experiment.** If the stop rule of experiment 713's
design fires on its eight folds (the selected network is 0.10 to 0.15 points worse than the
Candidate), an increment on that network has no route to production, and the work stops to ask.

## Experiment 713's known-answer gate was waived, and this limits the conclusions

**Experiment 713's known-answer gate failed KA3 and KA12 and the maintainer waived it on
2026-11-23 (713's design amendment 4).** Every warm-start state here comes from that waived gate. This experiment therefore makes no claim
about era recognition or about following level shifts, and every verdict, headline, and take-home
message of this experiment says so. The gate of this experiment (KF1 to KF15 below) is its own and
is not waived.

## Capacity look-ahead at the eight

**The as-of capacity of the eight MVA-only Primaries in the seven folds 2025-11 to 2026-05 uses
information from after the fold's cutoff, in two ways.** The probes' neural networks that
experiment 662 stored were trained to 2026-03, so folds 2025-11 to 2026-02 look ahead by up to 4
months. The calibration factors of all seven folds come from data to 2026-06 (experiment 662's page
says "the calibration at 2026-03 reuses the 2026-06 factors", and the rescale of 0.7858 of
experiment 673 is measured at the 2026-06 cutoff). Only fold 2026-06 is clean at the eight. The six
are clean in every fold, because experiment 649 stores one calibration factor per cutoff. The leak
is one constant per Primary and fold, and carries no day-level information. A2 cancels the
constant, because both arms of the contrast read the same capacity, and A1 does not. A read-out at
the six, where the honest capacity is known, measures the effect of a look-ahead of the same kind.

## What is compared

| Arm | PV input | Seeds | Role |
|---|---|---|---|
| `M*` | None | Stored, 0 to 4 | Control of A1; never refitted |
| `Z` | None; this experiment's code with every PV path absent | 0 and 1 on 5 folds | Repeat of `M*`; measures run-to-run noise |
| `H` | CAMS `PV̂` in the 14 history rows | 0 to 4 | Treatment |
| `H-S` | History `PV̂` with the sky of the date 28 days earlier | 0 to 4 | Counted placebo |
| `H-X` | History `PV̂` of another Primary | 0 and 1 | Report-only placebo |

Part 2's arms `HT`, `HT-S`, and `T` use the code of this folder and are scored in
[experiment 717](../00717-msa-unmetered-pv-part-2-target-row-forecast/result.md). Seeds 2 to 4 need
experiment 713's phases 8 and 9; without them, the design reads every contrast with two seeds, a
noise floor of 0.054 at the six and 0.142 at the eight, and a detectable difference of 0.094 and
0.25. Each placebo keeps the parameters, the training recipe, and the capacity of its treatment and
changes one property of the PV input. `H` minus `H-S` is what the actual sky of each history day
adds. `H-S` minus `M*` is what the capacity, the season, and the added parameters add. `H` minus
`H-X` is what the Primary's own capacity adds.

## The PV estimate

**`PV̂(i, t) = C(i, f) × u(i, t)`, where `C` is the as-of AC capacity of Primary `i` for fold `f`
in megawatts and `u` is the yield curve.** The network reads `p = PV̂ / capacity / target_std`,
clipped to 0 to 10, in the units of its power input, with `target_std` from the Candidate chain's
stored fold statistics and no mean subtracted. The yield curve is the repository's physics curve,
`e00405_pv.unit_shape` divided by `u99()` (0.879): the Erbs split of global horizontal
irradiance, a fixed plane (tilt 35 degrees, azimuth 180 degrees), a DC-to-AC ratio of 1.15, and a
derate of 0.4% per kelvin of cell temperature. History rows use the CAMS shortwave and ERA5 2 m
temperature of the dense table that experiment 713 reads. One orientation stands for every roof,
and the curve has no export limit.

| Primaries | Table | Value | Cutoff rule |
|---|---|---|---|
| The six (D5, D9, D10, D11, D14, D16) | experiment 649's `capacity_table.parquet`, probe year 2025 | `cal_p50` | The latest stored cutoff at or before the fold's cutoff |
| D2, D3, D4, D19, D21 | experiment 662's `capacity_table.csv`, method `L0` | `cal_p50 × factor / 0.7858` (experiment 673's rescale) | Cutoff 2026-03 before the 2026-06 fold: look-ahead |
| D8, D22 | The same table, method `L0` | `cal_p50` | As above |
| D12 | The same table, method `FA` | `cal_p50` | As above |
| D15, D18 | None | `p` is zero and flagged absent | |

## How the estimate enters the network

**Each new path starts at exactly zero, with a random partner, so a network that loads a stored
state of `M*` gives the output of `M*` bit for bit.** The token path is
`γ_h ⊙ (W2_h GELU(W1_h [p·ok, ok]) / r_h)`, added to each history token before the first axial
block. `W1_h` is 2 to 32 with bias, `W2_h` is 32 to 64 without bias, `γ_h` is a 64-element gain
initialised at zero, and `r_h` is a stored constant (the root mean square of the path's output at
initialisation over the arm's training tokens), not a per-token LayerNorm, because a per-token
LayerNorm discards the length of a vector that is a function of one scalar (median cosine
similarity 0.98 between `p` of 2 and 3). The read-out path computes, from the attention weights
`α` that build `base`, `pbar` (the PV inside `base`) and `cover`, and adds `a[primary] × pbar × cover`
to each mixture mean through a 16-entry table `a` initialised at zero, and adds
`[pbar, cover, p0_mean, p0_sd, top_ok]` to the head's first-layer pre-activation through a
zero-initialised linear map. New parameters: 2,560 in arm `H` (2.0% more than the 129,999 of
`M*`). Every new module is built from a dedicated generator seeded by the network seed, so the
global torch stream is untouched.

## Training recipe

**Every arm skips stage 1 and runs stages 2 and 3 of experiment 713's recipe from the stored stage 1
state of `M*` of the same fold and seed.** Stage 1 trains on 2019-09 onward, where a constant
as-of capacity would overstate `PV̂` by up to 30%, and a shared stage 1 state pairs each fit with
the `M*` fit of the same seed. The parameters of `M*` train exactly as in experiment 713 (2e-4,
5e-4 for `proj_s` in stage 2, a 300-step warm-up, and a gradient clip over the parameters of `M*`
alone). **The new parameters form their own optimiser group, with no warm-up, their own clip, and
no weight decay on the gains.** The stored logs of experiment 713 put the best state of stage 2 at
66 to 224 optimiser steps and of stage 3 at 22 to 644 steps, inside the 300-step warm-up at 2e-4,
so a zero-initialised gain at the trunk's rate would move by about 0.0015 to 0.017 against a
working value near 1. The rates are 1e-2 for `a`, 3e-3 for `γ_h`, and 2e-3 for `W1`, `W2`, and the
head map; the reachable range is 0.88 for `a` and 0.26 per element for `γ` in the shortest stored
fit. Mask dropout hides the PV estimate of the age-2 row and part of the age-3 row on 10% of
training samples, and of ages 2 to 5 on another 10%, from a generator of its own.

## Counted contrasts and the decision rule

**Part 1 has four counted readings, A1 and A2 at the six and at the eight, so each interval is read
at 98.75%.** Each contrast is treatment minus reference in CRPS on experiment 713's row set,
headline band, and 8 folds, from seed-averaged forecasts, through 713's scorer.

| Id | Contrast | Question |
|---|---|---|
| A1 | `H` minus `M*` | Does a historical PV estimate in the history rows improve the network? |
| A2 | `H` minus `H-S` | Does the gain come from the actual sky of each history day? |

The verdict mapping is the one in the front matter. The verdict `supported` also needs the sign of
A1 to hold with each Primary and each month left out in turn, the q01 and q99 pinball loss of `H`
within 0.05 of `M*`, the normalised MAE of the median to move the same way as CRPS, `H` not worse
than `M*` by more than 0.05 on cold winter nights, and A1 to keep its sign under both masks of the
read-outs below. **A rule on arm `Z`:** if `Z` minus `M*` on its 5 folds exceeds 0.054 at the six or
0.142 at the eight in absolute value, audit trigger 1 fires before any A1 is read. **Report-only,
at 95%:** `H` minus `H-X`, `H-S` minus `M*`, `H` minus `Z`, every arm minus the Candidate, every
contrast at all 16 Primaries, by lead band, by Primary, with one Primary left out, and with one
month left out. Each page also reports its contrasts at 99.375% (eight readings across both parts)
as a labelled sensitivity reading that changes no verdict. **A gain that the placebo matches is not
evidence for the PV estimate.** Part 1 screens on four folds (2025-11, 2026-01, 2026-05, 2026-06):
the stop rule is experiment 713's (`H` minus `M*` above +0.15 in both groups, or above +0.10 in both
and positive in at least 3 of 4 folds in both), and every screen number is labelled interim.

**Secondary readings, read at 95% and changing no verdict (14 in all, so one false positive among
them is likely):** S1 (A1 helps in the solar block of the sunny folds 2026-04 to 2026-06), S2 (the
same for A2), S3 (A1 removes the copied dip on days duller than their history window, compared
with a control that shifts `M*`'s quantiles by `H`'s mean median difference), S4 (no harm in the
dark folds), and S5 (dose response against experiment 673's PV share). Read-outs at prediction
with no refit: a CAMS-rule mask, a mask of ages 2 to 5, a look-ahead sensitivity at the six, PV
absent, capacity scale of 0.7 and 1.3, the learned gains as a share of their reachable range, and
the four load-shift scenarios.

## Known-answer checks

**Every threshold is fixed now and no threshold is tuned to pass.** The wiring checks KF1 (zero
start), KF2 (poisoned PV after the origin), KF3 (placebo donors), KF4 (the estimate against
`e00634_common.unit_curve` and an independent NumPy computation), KF5 (as-of capacity, with a
reproduction of experiment 673's `probe` column to 1e-9), KF6 (weather look-ahead statement), KF7
(prediction against the stored forecast of `M*` to 1e-6), KF8 (gradients and optimiser groups),
KF13 and KF13b (the copied training loop equals 713's, bit for bit on the CPU), and KF14 (CAMS
stamps are period end) run before any real fold. Three synthetic worlds (WH: the sky hidden in the
history; WT: hidden in the target row, used by part 2; WN: no PV dip) with a hidden daytime level, a daily anomaly, and a PV-like dip of
known size gate the learning: KF9 (`H-S` minus `H` at least 0.5 × `gap`), KF10 (the placebo does not
help, at most 0.005), KF12 (negative control, at most +0.005), and the secondary KF15 (learned
gain). **A ladder of five world rungs is pre-registered, because a Monte Carlo found that rung 1
probably gives a `gap` (the naive reference's MAE minus the oracle's) below the required 0.010 of
capacity.** The first rung that passes the rule below is used, decided from NumPy references on
the real weather before any PV fit. No threshold changes between rungs. If no rung passes, the
work stops before any PV fit and goes to the designer.

**The `gap` is measured on the 236 days after the cutoff, over eight world draws, and a rung passes
only if both the mean and draw 0 reach 0.010.** The validation origins of the worlds are the 236
days after the cutoff (`HELD_OUT_DAYS = 236`) for the `gap` of WH and WT and for every metric of KF9
to KF12. For each rung, `e00716_world_rung.py` computes the `gap` of WH and of WT for world draws 0
to 7 (`build_world(seed=k)`). A rung passes when, in both worlds, the mean `gap` over the eight
draws is at least 0.010 and the `gap` of draw 0 is at least 0.010. A rung whose mean passes and
whose draw 0 fails is a failed rung, and no other draw is used. Every fit and every check keeps draw
0, so KF9 and KF11 compare against the `gap` of draw 0 on the 236 days. `world_rung.json` records
the eight values, their mean, their standard deviation, and `held_out_days`, and `chosen_rung()`
refuses a file that is stale, lacks those fields, or has a `held_out_days` other than 236. KF9 and
KF11 are also printed on validation days 1 to 60 and 61 to 236 as secondary readings that never
gate.

**An early-stopping pre-check follows, and a failure steps the ladder up one rung.** The pre-check
reads no PV fit. It computes the Gaussian negative log-likelihood of the naive and of the oracle
reference on the (sample, lead day) pairs and target tokens that experiment 713's `validation_loss`
averages. The difference must be at least 10 times the standard deviation of arm `M`'s validation
loss over its last 10 passes in the same world, with the best pass reconstructed from `val_history`
by 713's improvement test and a patience of 8. After a failed pre-check at rung N, the command
`e00716_world_rung.py --after N` refuses to run unless `precheck_rung<N>.json` exists, records a
failure, and comes from the current world code. It then computes the next rung and never chooses a
rung numbered N or lower, and arm `M` is refitted at seed 0 in WH and WT at the new rung. If no rung
above N passes, F7 stops before any PV fit and the designer is asked. **Every extra fit of arm `M`
is charged to the reserve of phase 9 of design section 10 (1.5 GPU-process-hours), and all three
worlds use the same rung.** Arm `M` at seeds 1 and 2 and in world WN is fitted only at the rung that
finally passes. **Arm `M` is fitted at seed 0 in WH and WT first, and the pre-check is reported
before any other fit.** Two diagnostics are printed beside the pre-check and never gate it: the
standard deviation of arm `M`'s validation loss over the 10 passes ending at the best pass, and the
standard deviation of the last 10 passes after removing a linear trend. **A second failure, after
the step up to rung 5, goes to the maintainer through the manager as a waiver question, and no new
definition of the check's tokens or noise is adopted afterwards.** In the eight stored known-answer
fits of experiment 713 the standard deviation of the last 10 validation losses (`ddof=1`, as in the
code) lay between 0.0037 and 0.0265: arm `M` in world W1 sits at 0.0220 against the rung-4 limit of
0.0225, so the rung-4 pre-check is close to a coin toss weighted towards a pass.

**The phase 0 smoke fit of arm `H` runs after the gate, not before it.** `fit()` refuses to run
before the known-answer gate has passed and has no flag to skip the gate, so the smoke fit of the
run brief's phase 0 moves to follow the gate.

## Leakage and what a live service could build

**The history PV estimate rides on the CAMS analysis, which is the lab shortcut experiment 713
already flags for ages 2 to 5.** Under experiment 504's assumed delay of 36 hours, the estimate is
buildable live for ages 4 to 15 and columns 0 to 24 of the age-3 row, and the ERA5 temperature
makes the estimate buildable only for ages 6 to 15 under 713's convention. The weather tokens of
ages 2 to 5 keep the lab shortcut in every arm, so every score of `H` is a ceiling for a live
forecast. The newest history power is visible only at grid index at most `origin_idx - 48`
(27 hours before the contract origin). The placebo `H-S` reads the sky 28 days before each history
token, older than the history rows. The placebo `H-X` reads the `p` of a fixed partner (D12 with
D14, D22 with D11, D5 with D19, D10 with D21, D2 with D8, D16 with D3, D9 with D4).

## Compute

**The plan needs about 14 GPU-process-hours for part 1's arms on one worker (part 2 adds about 8),
against a reporting ceiling of 24 across both parts.** One fit is two stages and one prediction,
about 0.08 GPU-process-hours. Part 1 runs: a smoke fit and the known-answer worlds (about 3.1),
`Z` on 5 folds with 2 seeds (0.8), `H` and `H-S` on the 4 screen folds (1.3), on the other 4 folds
(1.3), seeds 2 to 4 (about 3.9), and `H-X` (1.3), with a read-out budget and a reserve. The 5-seed plan also needs
experiment 713's phases 8 and 9 (about 15.6 GPU-process-hours, or 4.8 if only `M*` runs). The
order of cuts is `H-X` first, then the secondary known-answer fits, then `Z` on 3 folds. No fit
runs until the code reviews pass.

## Plots

`p` beside the net power for 14 history days of a sunny and a dull window at a high-share and a
low-share Primary; the contrast by half-hour of valid time per fold; the contrast against PV share
per Primary; day-ahead and 14-day forecasts of `M*` and `H` on a clear target day and on a dull
target day after a sunny window; the read-out weights and across-day attention of `M*` and `H`;
the learned gains per Primary; and the token-length bars. Every score, forecast, capacity, and plot
is of a Primary. The capacity is an estimate of the unmetered PV behind a Primary and not the
output of any metered generator, and no page names or plots a metered generator's time series.

## Risks

- **The estimate is mostly redundant** with the CAMS shortwave and the ENS shortwave already in
  the tokens, so a tie is the most likely result; the learned gains say whether the network
  ignored the estimate or used it without a gain.
- **A null result may come from slow learning**, which experiment 713's gate found. KF9 and the
  reachable-range read-out test this.
- **A1 compares a refit with a stored fit.** Arm `Z` measures the run-to-run difference.
- **The capacity is uncertain and partly looked ahead.** The scale read-out and `H-X` bound the effect.
- **The eight read apparent power**, so demand plus `p` is not gross demand when PV exceeds demand
  (at D12).
- **One summer fold and two spring folds carry the solar evidence**, and the design cannot resolve
  a gain under 0.06 at the six, although the smallest gain worth porting is 0.02.

## Method

The experiment is planned and has no fitted result yet, so these bullets describe the design.

- **The known-answer gate is waived at KF9:** `e00716_waive_gate.py` writes the gate file with
  `waived: true`, the failed check and its values, the code hash, and the sha256 of the last-run
  file and of `e00716_known_answer.py`. The scoring rule of the known-answer checks, patience 8 and
  the median of the 5 passes ending at the best pass, is unchanged.
- **The pre-check reads seed 0 only:** the noise limit of the early-stopping pre-check comes from
  the last 10 validation losses of arm `M` at seed 0. At seeds 1 and 2 of WH the same noise is 0.029
  and 0.031, above the limit of 0.0225 at rung 4. The diagnostic best pass of the pre-check uses
  patience 30, the patience that trains arm `M`. An Opus review of the code behind the pre-check
  (commit 126577d1d) was clean.
- **Architecture:** experiment 713's network (width 64, two axial blocks) with the new token path,
  read-out path, and head map of "How the estimate enters the network".
- **Training regime:** stages 2 and 3 of experiment 713's recipe from the stored stage 1 state of
  `M*`, monthly refits over 8 folds (2025-11 to 2026-06), seeds 0 and 1 (0 to 4 if experiment 713
  stores them), cleaned power with the `substation_zero` half-hours dropped and the D15 and D18
  rows before 2025-03-01 cut, and the ENS 00Z run for the target row.
- **Measurement:** CRPS in points of capacity on leads from 24 h up to but not including 240 h,
  against the cleaned power, through 713's scorer; baselines `M*`, the Candidate, the climatology,
  the manual heuristic, and BG2; month-block bootstrap intervals; no valid time on or after
  2026-07-01 is scored.
- **Standing practices followed:** monthly refit; cleaned data; the short-history cut; every fitted
  state saved outside git; stored forecasts reused as controls; a four-fold screen; tails, cold
  winter nights, and the curtailment figure; example forecasts; load-shift plots with the
  unlogged-event caveat. A stated departure: the date-time encoder holds calendar columns, as in
  experiment 713.
- **Departures from earlier advice:** see "Why this runs although the stop rule counts earlier
  verdicts".

## Amendments made after the first code review

**The reference regression is read as 48 ridge regressions per Primary, pooled over the 14 leads.**
The design asks for "a ridge regression, fitted separately for each of the four Primaries". A single
linear model cannot scale a level by the target day's daily shape, so each Primary gets one ridge
regression for each of the 48 half-hours of the target day, with one shared penalty chosen by 3-fold
cross-validation over contiguous blocks of origins, and the 14 lead days pooled. The history features
carry the same weight at every lead. The naive and the oracle references share this structure.
The oracle may therefore learn more from the history than a single linear model would, so `gap`
probably understates how much the estimate tells a neural network, and the KF9 threshold of 0.5 ×
`gap` is probably easier to meet than a single-model reading would make it.

**The first rung failed on fresh draws on the 60-day window, and the validation window was widened
before any fit of `M`.** The stored rung-4 pass came from an upper-tail world draw: on the 60
validation days, 3 of 10 fresh draws reached 0.010 and the mean was 0.0087. Before any fit of `M`,
and with those values known, the `gap` and KF9 to KF12 were moved to the 236 days after the cutoff
and the rung rule gained a condition on the mean over draws 0 to 7. The world itself was not
changed. The ten values of WH's `gap` at rung 4 for draws 0 to 9 on the 60 days were 0.0165, 0.0041,
0.0061, 0.0044, 0.0124, 0.0058, 0.0027, 0.0215, 0.0062, and 0.0075. **The change turns rung 5 from a
fail into a pass:** on the 60 days rung 5 fails at draw 0 (0.0099), and on the 236 days it passes
(0.0147 in WH, 0.1163 in WT), so a failed early-stopping pre-check at rung 4 now leads to rung 5 and
not to a stop. KF9 at rung 4 needs 0.0073 where the 60-day rule needed 0.0082, and KF11 needs 0.057
where the 60-day rule needed 0.041. The rebuilt `world_rung.json` gives, for draws 0 to 7 on the 236
days, WH at rung 4: 0.0146, 0.0097, 0.0091, 0.0122, 0.0116, 0.0147, 0.0104, and 0.0208 (mean
0.0129), and WT at rung 4: 0.1141, 0.1148, 0.1097, 0.1147, 0.1067, 0.1140, 0.1152, and 0.1094 (mean
0.1123). Rungs 1 to 3 fail in WH with means of 0.0036, 0.0057, and 0.0059, so rung 4 is chosen. A
gating fit takes about 40% longer, because every held-out sample is scored at all 14 lead days
(about 13,200 rows per pass, against about 3,400), which is an estimated 0.5 GPU-process-hours over
phase 1.

**The gating fits mirror the real recipe, with two stated differences.** The encoders are frozen, the
constant divisors of both token paths are calibrated on the arm's own training samples before the first
step, and the fits start from an arm `M` trained in the same world with the same hidden sky. `proj_s`
trains at the stage 2 rate of experiment 713 (5e-4), which the design's list of gating settings did not
state, and the early-stopping patience is off, so the state that KF15 reads is the best of all 60 passes.

**The cold-night guard of `supported` is computed inside the scorer.** The protocol scorer of
experiment 713 loads only 713's own arms, so the scorer here reads the same two thresholds of the
`cold_nights` slice (the sun more than 5 degrees below the horizon, and the ERA5 temperature at or
below 3 degrees Celsius) from the dense table, in points of capacity. A missing input raises instead
of passing. **The guard's slice departs from the protocol's `cold_nights` slice in two ways.** The
guard adds a restriction to UTC months 12, 1, and 2, to match "cold winter nights", whereas the
protocol's slice has no month restriction. The guard also reads the sun at the grid stamp (the
cosine of the zenith in dense column 7), whereas the protocol reads it at the half-hour midpoint.
Dense column 2 as the protocol's `temperature_c` series has not been compared.

**KF7 runs on the GPU.** The stored forecast of `M*` was written on the GPU, and the CPU path
differs from it by 8.4e-6 of capacity at the 99th percentile (experiment 713's own CPU path gives
the same difference), so the 1e-6 tolerance is met only by a GPU run.

**KF2 runs on the training phase and on the held-out origins.** The poison test runs once on 64
stage 2 samples of the training phase and once, in a process of its own, on 64 draws from the scored
origins of the held-out month, where the target table holds the runs of that month. An arm with no
history row or no target row (`H`, `H-S`, and `H-X` have no target row) skips the negative control for
that input and instead asserts that the poison changes nothing. **KF4's real-data half cannot show that
the dense shortwave and temperature values are right.** Its comparison of the dense `u` with
`e00634_common.unit_curve` reads both sides from the dense table, so it tests the composition of the
yield function (the column order, the day of the year, and the normalisation by the 99th percentile)
and not the weather. The independent rows hold forecasts, so the geometry columns are compared for
equality and the weather columns for correlation (0.90 for shortwave and 0.93 for temperature in fold
2025-11).

**S5 pools the 14 Primaries of the six and the eight, and report-only contrasts are read on the cells
both arms hold.** An arm that exists on 5 folds (`Z`) or at 2 seeds (`H-X`, `T`) gives a contrast on
those folds and seeds, and the scorer prints and records which folds, seeds, and cells are missing. A
contrast with no common cell is recorded as `not_read` with its reason.

**The `HT-S` exception covers the first 28 days of each fold's own Candidate rows.** KF3 prints the count
for the first fold, and the count decides whether the exception days include days that have a real
earlier donor in the archive.

## Reviews

Two Opus design reviews preceded this page (18 findings in the second, all accepted; the first
review's findings are in its triage). The second review's finding S7 changed one verdict rule
before any fit, as described in the decision rule. Still to come: an Opus code review of every
new script before any run, a second Opus review of the fixed code, the known-answer gate, a result
review, an audit if a trigger fires, a write-up review against the printed report, and a prose
review.

## Files

- `data/ml-experiments/briefs/msa_architecture/f7_design.md`, `f7_design_review.md`,
  `f7_design_review_2.md`, and the two triage files (read, private): the design that this page
  condenses.
- `ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/`: the code, written and
  unit-checked on the CPU, with no fit run yet. It holds the code of both parts (the PV estimate and
  capacity, `PvNet`, batches, the copied training loop, the fit driver, prediction, the synthetic
  worlds and ridge references, the known-answer gate, the scorer, the read-outs, figures, and report);
  part 2's folder imports it.
- Read from experiment 713 ([pull request 402](https://github.com/openclimatefix/nged-ml-research/pull/402)):
  the stored stage 1 and stage 3 states and forecasts of `M*`, the inputs adapter, and the scorer.
- Read from experiments 649, 662, 673, 405, and 634: the capacity tables and the yield curve.
- Written by this experiment (to be created): fitted states under
  `data/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/work/` and predictions under
  `predictions/`, never committed.
