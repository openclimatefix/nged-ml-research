---
id: "00710"
slug: ens-only-one-shot-training-recipe
title: >-
  Can the demand network train on ECMWF ENS alone, in one stage or with a recency-weighted final
  stage, and forecast as well as the 3-stage recipe that pre-trains on ERA5 and CAMS?
theme: retraining-and-training-history
also_relevant_to:
  - weather-products-and-blending
  - switching-labels-in-the-forecast
status: planned
kind: change
protocol: demand-v2
size: >-
  Complex. (1) It stores about 380 fitted networks in the minimal plan, with per-row forecasts and
  two injected-event forecasts per network; (2) more than one design defensibly tests the hypothesis
  (the shape, mechanism, and half-life of the recency weight, and the injection state); (3) it
  reuses the training code of experiments 425 and 707, the injection code of experiments 581 and
  699, and the scoring code of experiments 587 and 707, whose callers cannot be named without
  searching; (4) it touches no serving path or degradation rule; (5) it changes no contract. Full
  set of reviews: plan, code, results, write-up, prose.
hypothesis: >-
  A pooled mixture density network trained only on the 47 to 58 months of ECMWF ENS history (window
  F of experiment 707), with synthetic switching events injected into its lag inputs and a recency
  weight on the final stage or throughout one stage, forecasts demand as well as the 3-stage
  reference that first pre-trains on ERA5 and CAMS, within 0.02 points of CRPS.
prediction: >-
  Written before any fit, as pre-registration. C1 (the best ENS-only recipe `B*-i` minus the
  reference `M1-i`) is between -0.02 and +0.03 points of CRPS at the six signed-MW Primaries, and
  between 0 and +0.04 at the eight MVA-only Primaries, where the reference held a lead of 0.041 in
  experiment 707. The placebo contrast C5 is below zero at the eight. Injection raises CRPS on
  ordinary rows by under 0.02 (I1 to I3) and lowers the slump share (I4 to I6).
decision_rule: >-
  Non-inferiority margin 0.02 points of CRPS, the demand protocol's tolerance for a headline. C1 is
  read per group on its interval [lower, upper] at 99.17% (6 readings, family R), and the first test
  that holds gives the verdict. Lower limit above +0.02: refuted. Upper limit below +0.02:
  supported (and beats the reference where the upper limit is below 0). Lower limit above 0: not
  supported. Otherwise inconclusive. A supported verdict also needs the point estimate of C1 below
  +0.02 on seeds 0 to 2 and on seeds 3 and 4 separately, the upper limit on the eight without D8
  below +0.02, and the one-sided 95% bounds on the tails, lead bands, cold nights, and sunny
  half-hours within 0.05. The experiment is supported when both groups are, partly supported when
  one is, and otherwise the worse of the two group verdicts. A claim that a recency weight helps
  needs the upper limits of C4 and of the placebo contrast C5 both below 0.
benchmark: >-
  The reference recipe `M1`, which is arm `A_F` of experiment 707 (ERA5 and CAMS pre-training, then
  ENS, then a recency-weighted fine-tune), refitted with injection (`M1-i`) and stored without it
  (`M1-n`); the ENS-only arms `M2` (`R_F`) and `M3`; climatology, the manual heuristic, BG2, and
  `S0M` as reference rows.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) at lead days 1 to 10 in points of
  effective capacity, with the normalised MAE of the median beside it, and, per Primary group, the
  sunny half-hours, cold nights, the slump share into injected and real switching events, and the
  half-hours after an event.
production_status: not-for-production
adopted_in: none
---
**This page is a pre-registration. No network has been fitted, and every number below is quoted
from an earlier experiment, with its source named.** The status is `planned`. Everything on the page
was written before any fit.

## Plan

### What we are asking, and what we will decide

**We ask whether the demand network needs the ERA5 and CAMS pre-training stage at all.** Today's
reference recipe trains the pooled mixture density network in three stages: first on ERA5
reanalysis and CAMS solar estimates of past weather, then on the archive of ECMWF ENS forecasts,
then on the last 730 days of ENS with a recency weight. Dropping the first stage would remove two
weather sources and a code path from any future production recipe. [Experiment
707](../00707-ens-backfill-pretraining-gain-against-ens-history-length/result.md) found that the
first stage still helped at the eight MVA-only Primaries (-0.041 points of CRPS) and not clearly at
the six signed-megawatt Primaries (+0.019).

**We will fit several ENS-only recipes and compare each with the reference.** The recipes differ in
where and how a recency weight enters: a separate final stage, a loss weight over one stage, a
schedule over epochs, a hard window, a floor, a sampling probability, weight averaging, a cyclic
schedule, and a half-life per Primary. Every recipe is fitted with and without synthetic switching
events injected into its lag inputs, and every recipe is scored on five kinds of half-hour: all of
them, sunny ones, cold nights, the slump into switching events, and the half-hours after an event.

**We will decide whether the best ENS-only recipe forecasts within 0.02 points of CRPS of the
reference, in each of the two Primary groups.** The experiment adopts nothing. The result tells the
maintainer whether to drop ERA5 and CAMS pre-training from the production recipe, and which recency
weight to use if so. The result applies to the pooled network of experiment 707, not to the
production Candidate network, which is modular and fine-tunes daily with a 90-day half-life.

**Protocol line.** This is a comparison of ENS-only recipes with the reference recipe `M1`, which is
arm `A_F` of experiment 707, scored under the demand protocol `demand-v2`
(`protocol: demand-v2` in the front matter), on row set `fold_a_daily_gap_v1`. The reference is a
stored research recipe of experiment 707, so Primary demand research default 3 (the GAM stack and
heuristic blend of experiment 322) is context here, not the benchmark.

### Why the stop rule allows the attempt

**A plain count of the theme's Experiments table gives three `not supported` verdicts on recency
weighting (experiments 331, 407, and 466), and one direct `not supported` verdict on one-shot ENS
training (experiment 519).** After three, the stop rule asks for the maintainer's approval, so the
threshold is reached. The related verdicts on a different pairing of weather products (experiments 3
and 9) are listed and not counted. Experiment 519 trained the lag-free network and gave it ENS rows
during pre-training. Experiment 3 and experiment 9 compared ERA5 and CAMS training with ENS
training of the first stage of a different model.

**The maintainer approved the attempt on 2026-11-19, through the manager.** The request reads:
"train in one shot ... using just the long history of ECMWF ENS ... find an optimal recipe", with
recency-weighted arms. The maintainer later noted that experiments 331, 407, and 466 tested other
models or targets, and that the half-life arms H1 and H2 of experiment 407 are the only arm-level
result on the demand network.

**A `not supported` or `refuted` verdict here counts once against each of the two ideas, recency
weighting and one-shot ENS training.** The write-up then says what a further attempt would change.

### What is new against each earlier verdict

| Experiment | What it tested (pre-registration, from the experiment's own page) | What is new here |
|---|---|---|
| [331](../00331-primary-demand-recency-weights-on-gam-and-kalman/result.md) | A GAM with a Kalman filter at Primaries, half-life against equal weights: -0.025 points of CRPS (95% interval -0.032 to -0.017) | A neural network, ENS weather, a window back to 2021, weight schedules, injection, and four extra metrics |
| [407](../00407-pooled-mixture-density-network-tail-arms-and-recency-weights/result.md) | Arms H1 and H2 of the pooled network: fine-tune half-lives of 1 year (+0.009, -0.015 to +0.033 at the six) and 0.25 years (+0.073, -0.003 to +0.140), on five screen months and a 24-month window. This counts as one arm-level result at most | Twelve folds, the long window, the half-life varied inside a one-shot stage as well as in the fine-tune, other weight shapes, schedules, injection, and the four metrics |
| [466](../00466-wind-and-solar-recency-weighting/result.md) | Wind and solar networks with 12.5 to 26 months of history; best solar -0.14% (95% interval -0.36% to +0.08%) | Primary demand, a history of 47 to 58 months, and every other change above |
| [519](../00519-ens-share-in-lagless-network-pretraining-ratio-sweep/result.md) | ENS rows in the pre-training of the lag-free network: the forecast got worse at the eight, by 0.06 to 0.22 points | The pooled network with lag inputs, ENS as the only weather source in every stage, and the ENS history reaching back to 2021 |

**The reference's third stage already is a recency-weighted ENS fine-tune.** The reference `M1`
(`A_F` of experiment 707) trains on ERA5 and CAMS with equal weights, then on ENS from 2021-07-20
with equal weights, then on the last 730 days of ENS with weight `0.5 ** (age / 0.5 years)`. The
arms here change three properties of that last step: the half-life, the shape of the weight
(exponential, hard window, exponential with a floor, or a schedule over epochs), and the mechanism
(a loss weight, a sampling probability, a separate stage, or a weight per Primary). No arm
introduces a recency weight where none existed. The stored fine-tune also caps the window at 730
days, so a 2021 row never carries weight in the reference.

### Arms

**The notation is `h` for the half-life in years and `w = 0.5 ** (age / h)` for the exponential
weight, where `age` is the refit time minus the row's valid time, in days.** "From scratch" means a
fresh network and a fresh AdamW at learning rate 5e-4. A stage that follows another stage uses 2e-4.
Batch size (16,384), weight decay (1e-3), dropout (0.1), gradient clip (1.0), patience (6), and the
1.15 widening of the quantile bands are those of experiment 707 in every arm.

**The maintainer's arms.**

| Arm | Recipe | Difference from the reference `M1` |
|---|---|---|
| `M1` | ERA5 and CAMS stage, then `S_ens(F)`, then the 730-day fine-tune at `h` = 0.5 | The reference; arm `A_F` of experiment 707 |
| `M2` | `S_ens(F)`, then the fine-tune | Drops the ERA5 and CAMS stage; arm `R_F` of experiment 707 |
| `M3` | `S_ens(F)` only | Drops the ERA5 and CAMS stage and the recency-weighted stage |
| `W-h0.5`, `W-h1`, `W-h2` | One stage from scratch on window F with an exponential loss weight over the whole window, `h` = 0.5, 1, and 2 years | One stage with recency weight and no 730-day cap; `M3` is the `h` = infinity end of the series |
| `S` | One stage from scratch. Epoch 1 has equal weights, epochs 2 to 6 use `h` = 8, 4, 2, 1, and 0.5 years, and every later epoch uses `h` = 0.5 | The weight is a function of the epoch |

**The creative arms differ in kind from the maintainer's arms.**

| Arm | Kind | Recipe |
|---|---|---|
| `HW730`, `HW365` | Hard window | One stage from scratch with equal weights on the last 730 or 365 days |
| `FLR` | Exponential with a floor | One stage with `w = 0.1 + 0.9 * 0.5 ** (age / 0.5)` |
| `SMP` | Sampling probability | One stage in which each batch is drawn with probability proportional to `0.5 ** (age / 0.5)` and the loss is unweighted |
| `RFE` | Recent-only final epoch | `S_ens(F)`, then exactly one epoch on the last 365 days at 2e-4 |
| `EMA` | Weight averaging | The recipe of `M2` with an exponential moving average (decay 0.999 per step) of the network weights over the fine-tune |
| `CYC` | Warm restarts | Three cycles of 4 epochs, with equal weights, `h` = 2, and `h` = 0.5, and a cosine learning-rate schedule in each cycle |
| `PP` | Weight per Primary | One stage with half-life `h_p` per Primary, chosen from {0.5, 1, 2, infinity} on tuning folds 2024-10 and 2025-01, with each Primary's total weight share fixed at its equal-weight value |
| `FTh90d`, `FTh1` | Half-life of the separate stage | The recipe of `M2` with a fine-tune half-life of 90 days or 1 year |

**The `W` arms change each Primary's share of the total weight as well as the recency.** D9 has no
power before 2023-08-23, and D15 and D18 lose rows before 2025-03-01, so a short-history Primary
gains weight share under a short half-life. Contrast C4 therefore mixes the recency effect with the
change in Primary mix, and `PP` fixes the shares. The sources of the methods are [Loshchilov and
Hutter (2017)](https://arxiv.org/abs/1608.03983) for warm restarts, [Polyak and Juditsky
(1992)](https://doi.org/10.1137/0330046) and [Izmailov et al.
(2018)](https://arxiv.org/abs/1803.05407) for weight averaging, and [Byrd and Lipton
(2019)](https://arxiv.org/abs/1812.03372) for testing a sampler beside a loss weight.

**Every arm is fitted in two injection states.** In `-n`, no synthetic events enter training. In
`-i`, synthetic events enter every training stage. The stored forecasts of experiment 707 hold no
injection, so they serve as `M1-n` and `M2-n`, and the state after `M2-i`'s equal-weight stage is
`M3-i` with no extra fit.

### Row set, folds, and standing practices

**Every arm is scored on row set `fold_a_daily_gap_v1`, on the 12 monthly folds 2025-07 to 2026-06
of experiment 707, with the same Primaries and the same cleaning.** The six signed-MW Primaries (D5,
D9, D10, D11, D14, D16) and the eight MVA-only Primaries (D2, D3, D4, D8, D12, D19, D21, D22) are
scored separately. D15 and D18 are reported and not in any contrast. No valid time on or after
2026-07-01 is scored. The ENS window is window F of experiment 707, which starts on 2021-07-20 and
excludes the lockdown rows.

**Four tuning folds precede the scored year, so a choice made on them reads no scored row.** The
tuning folds are the test months 2024-10, 2025-01, 2025-03, and 2025-06. They name `B*` (the
ENS-only recipe with the lowest tuning CRPS on 2025-03 and 2025-06), `O*` (the best one-stage
recipe), and `h_T` (the half-life of the best `W` arm). Among recipes within 0.01 points of the
best, the recipe with the fewest stages, then the fewest hyper-parameters, wins, with one-stage
recipes preferred.

**The experiment follows these standing practices.**

- **Monthly refits,** at 00:00 UTC on the first day of each test month.
- **Cleaned data everywhere:** the half-hours that the `substation_zero` rule flags are dropped from
  training, lags, inference inputs, and scoring, and the count dropped per fold is reported.
- **The noisy start of D15 and D18 is cut:** every row before 2025-03-01 is dropped.
- **Every fitted network is saved** in the experiment's data folder, never in git.
- **Stored forecasts are reused first:** `M1-n` and `M2-n` are the stored `A_F` and `R_F` of
  experiment 707, and the reference rows come from stored forecasts.
- **A screen precedes the scored year,** and the four tuning folds play that part. Every number from
  them is labelled interim.
- **Tails and cold nights are reported** beside CRPS, with the q05 and q95 coverage.
- **Example forecasts are plotted:** a day-ahead forecast and a full 14-day forecast of each
  Primary group, and at least six event plots across both metering groups, with the unlogged-event
  caveat. Generators are anonymised.
- **The four load-shift scenarios are reported** (all rows; rows outside events and at least 7 days
  after one; rows inside events; the 7 days after an event ends).
- **The normalised MAE of the median** is reported in every metric table.
- **The pull request is opened as a draft** and marked ready after the reviews are triaged.

**The experiment departs from one practice.** The minimal plan screens on four tuning folds with two
seeds and not on one fold per season, because the tuning folds must precede every scored month.

### Weather at the forecast origin

**Every row of every arm reads weather issued at or before its forecast origin.** An ENS training
row takes the member mean and spread of one 00Z run, and its origin is that run's initialisation
time plus 9 hours. The code asserts this on every row, and a row enters a fold only if its valid
time is at least 3.5 hours before the refit. A scored row reads the 00Z run issued 9 hours before
its origin. The three lag inputs are power readings at least 3.5 hours old at the origin. The
network has no lag-free head, so no score reads experiment 404's voided lag-free scores above 24
hours. The `M1` ERA5 and CAMS rows read analysed weather at their valid time, in training only, as
in experiment 707, and ERA5 arrives about 5 days late. The cold-night and sunny labels are analysis
values at the valid time, applied identically to every arm, and never reach a network.

**The implementation prints one training row and one scored row before any fit.** The print gives
the Primary, the run's initialisation time, the origin, the valid time, the lead, the times of the
three lags, the row's age at the refit, and its weight under each weight rule.

**A recency weight is a function of one quantity, the row's age at the refit.** One network serves
every origin of its test month, and the refit time is at or before every scored origin of that
month. A weight never reads the target, the forecast error, the weather, or a row after the refit.

### Loss, steps, and selection

**Every arm minimises a weighted negative log-likelihood normalised inside each batch,
`(w * nll).sum() / w.sum()`.** The loss does not change when every weight is multiplied by a
constant. Every new stage is selected on one set, `V_sel`, with an unweighted loss: the held-out
weeks inside the last 730 days, which carry unshifted lags in both injection states. Training losses
are never compared across arms. The report prints the effective rows per batch, because the
effective share is about 0.32 for `h` = 0.5 over 4.5 years and about 0.63 for the reference's
fine-tune (pre-registration, from the weight formula).

**No ENS-only arm may take more steps than the reference may.** A one-stage arm on window F has an
epoch limit of 59, so its step cap equals the cap of `M2` (about 37,760 steps at fold 2026-01). The
report prints, per arm, the steps used beside the cap, the GPU seconds, and the share of fits whose
final stage stopped at its epoch limit. An arm that stopped at the limit in more than a quarter of
its fits is labelled "not converged".

### Injection of synthetic switching events

**The generator is `synthetic_clean` of experiment 581, unchanged.** For each Primary it draws
events until 5% of the half-hours are covered, with a length of 3, 7, 15, or 30 days, a depth of
0.15, 0.30, or 0.50 of capacity, a random sign, and a random start. The shift is added to the
power that feeds the three lag inputs and never to the target, so the target stays at the normal
running arrangement (NRA).

**Each (fold, network seed) has one world, which every stage and every epoch of every `-i` arm with
that seed reads.** The world seed is 7,100,000 + 1,000 times the network seed + YYYYMM. The
multiplier of 1,000 keeps the seeds of different folds apart, and the range (7,302,410 to
7,306,606) does not overlap the ranges of experiment 699. The arms of one fold and seed therefore
differ in the weights and the schedule only.

**An injected event carries the weight of the rows whose lags it shifts.** A lag-only event has no
loss term of its own, a separate weight would mix the recency contrast with the injection contrast,
and the realised share of shifted rows under a short window may be small. Every `-i` fit logs the
unweighted and the weighted share of rows with a shifted lag, per Primary.

### Contrasts

**All contrasts are treatment minus reference in the `-i` state, on seed-averaged forecasts, with a
month-block bootstrap of 100,000 resamples, read for the six and for the eight.** Each family has
its own Bonferroni level.

| Family and level | Contrast | Question |
|---|---|---|
| R, 99.17% (6 readings) | C1 (headline): `B*-i` minus `M1-i` | Does the best ENS-only recipe match the 3-stage reference? |
| | C2: `M2-i` minus `M1-i` | Experiment 707's C1 with the sign reversed, refitted with injection |
| | C3: `O*-i` minus `M2-i` | Does one stage match the two-stage ENS-only recipe? |
| W, 99.375% (8 readings) | C4: `W-h_T-i` minus `M3-i` | Does a recency weight beat equal weights in one stage? |
| | C5 (placebo): `W-h_T-i` minus `P-i` | Is that gain a gain from recency? |
| | C6: `S-i` minus `W-h0.5-i` | Does a schedule beat a constant weight with the same final half-life? |
| | C7: `HW730-i` minus `M3-i` | Does discarding rows older than 730 days harm the forecast? |
| I, 99.58% (12 readings) | I1 to I3: `X-i` minus `X-n` for `X` in `M1`, `M2`, `B*`, on CRPS | Does injection harm the forecast on ordinary rows? |
| | I4 to I6: the same three, on the slump share of the injected bank | Does injection reduce the slump? |

**The family levels are those of the full plan and stay fixed when the extension below is not
run.** The minimal plan reads fewer contrasts at the same, more conservative levels.

**A secondary contrast, `B*-n` minus `M1-n`, repeats C1 without injection.** It has its own verdict
line and does not change the experiment's verdict. The minimum detectable effect is stated before
the run: experiment 707's C1 interval at 99.69% had a half-width of about 0.026 at the six and 0.017
at the eight, so the half-widths at 99.17% are about 0.024 and 0.015 (rescaled from experiment
707). A `supported` verdict at the six therefore needs a point estimate below about -0.004, and at
the eight below about +0.005.

**Every other difference is descriptive and printed at 95%.** That covers each recipe's `-i` minus
`-n` on every metric, each `W` arm and each creative finalist minus `M3` and minus `M1`, `SMP` minus
`W-h0.5`, per-Primary and per-lead-band tables, each contrast without fold 2026-01, and a t test on
the 12 monthly differences.

### Decision rule

**The margin is 0.02 points of CRPS, because ERA5 and CAMS pre-training is a fitted stage with two
weather sources, so an ENS-only recipe needs to match the reference and not to beat it.** C1 is read
per group on [lower, upper], and the first test that holds gives the verdict.

1. **Lower above +0.02:** the ENS-only recipe is worse by more than the margin. `refuted`.
2. **Upper below +0.02:** the ENS-only recipe matches the reference. `supported`.
3. **Lower above 0:** the ENS-only recipe is worse, and a loss larger than the margin is not
    excluded. `not supported`.
4. **Otherwise:** `inconclusive`, with the interval's width printed.

**Two guards protect a `supported` verdict against seed noise and against harm on a protected
slice.** Experiment 707's audit found +0.024 at the six between two seed sets of one stored arm,
which is as large as the margin. A group's `supported` therefore needs the point estimate of C1
below +0.02 on seeds 0 to 2 and on seeds 3 and 4 separately, the upper limit of C1 on the eight
without D8 below +0.02, and the one-sided 95% bound of C1 within 0.05 on the 0.01 and 0.99 pinball
tails, on each lead band, on cold nights, and on `sunny_clear_sky_index`. If the realised clear-sky
index cannot be built, the protocol has no protected sunny slice, and `sunny_unit_shape` stays
descriptive.

**The six get three extra readings, because experiment 707's placebo failed there (C5 -0.011, -0.055
to +0.050).** The page makes no claim that ERA5 and CAMS weather helps or does not help the six.
The stored `R_F` scored slightly better than `A_F` at the six (2.821 against 2.840, experiment 707),
so the page also reports `B*-i` minus `M2-i`, and a `supported` at the six needs that point estimate
below +0.02. C2 shows whether the +0.019 of experiment 707's C1 repeats.

**A claim about recency needs the placebo.** The page says "a recency weight helps" at a group only
when the upper limits of both C4 and C5 are below 0. When C4 passes and C5 fails, the page says that
the weighted arm scored better and that weights unrelated to age did as well. The injection rule is
that injection "does no harm" for a recipe when the upper limit of its CRPS contrast (I1 to I3) is
below +0.02, and "reduces the slump" when the upper limit of its slump contrast (I4 to I6) is below
0.

### Known-answer tests and the placebo

**All of these run before the first fit of the tuning phase, and a failure stops the experiment.**

1. **The new loop equals the old loop when its features are off:** per-step losses agree to 1e-6
   against `train_network` of experiment 707 on one thread, one seed, 200,000 rows, and two epochs.
2. **A refit with the new loop matches the stored `R_F`:** on folds 2026-06 and 2026-01 with seeds 0
   to 2, within two standard errors of the seed spread, per group, with the comparison logged.
3. **The stored forecasts are reproduced by prediction only** from the saved networks to 1e-4, and
   `M3-n` loads each stored `S_ens(F)` state with the scale of the matching `R_F` fit.
4. **The weight rules give known values:** `W-h0.5` equals the stored `fold.weight` to 1e-6, `h` =
   infinity gives all ones, `HW730` keeps exactly the rows of window `W24` (5,608,723 rows at fold
   2026-01, experiment 707's log), epoch 6 of `S` equals `W-h0.5`, a factor of 10 on every weight
   changes no loss, and the sampler's frequencies match its weights.
5. **An empty world changes nothing:** with no event, the rebuilt lag columns equal the cached
   columns bit for bit in each of the four copied functions. With a world, only the three lag value
   columns differ, the last changed half-hour lies before the refit, and the held-out rows are
   unchanged.
6. **The slump share has a known answer:** a forecaster whose median copies the 1-day lag scores 1
   on the rows whose 1-day lag is shifted, and a forecaster that ignores the lags scores 0, for both
   signs.
7. **Experiment 699's known-answer table still reproduces** (MAE 7.124 and 7.165, copied shares
   0.123 and 0.075, from experiment 699), which shows the imported bank and share functions are
   intact.

**The placebo `P` gives each row the weight of `W-h_T` after the ages are permuted among whole
calendar months.** A seed fixed by the fold draws the permutation, a row takes the age of the
matching offset in the permuted month (clipped to the month's end), the partial first month of
window F keeps its own ages, and all Primaries share the permutation. The placebo keeps the weight
distribution, the effective rows, and the block structure in time, and removes the link between
weight and recency. A failed placebo fires audit trigger 1. The placebo tests recency and season
adjacency together. The experiment does not rerun experiment 707's weather-shuffle placebo, because
no contrast here claims a pre-training gain.

### Injected-event scoring and metrics

**The bank of injected events for scoring is `contaminate` of experiment 581, separate from the
training world.** Fold `k` (0 to 11) is forecast with pass `k` and pass `k + 12`, so each Primary
meets each of the 12 combinations of depth and length with both signs over the year. Every arm,
state, and seed uses the same two passes. The stored networks are forecast with shifted lags and no
refit.

**Every metric is reported for every arm in both states, for the six and for the eight, with a 95%
month-block interval and the number of half-hours behind it.** A subset under 100 half-hours is
marked thin, with the number of months that hold any row of the subset.

| Metric | Definition |
|---|---|
| CRPS and normalised MAE of the median | As experiment 707, over all shared half-hours |
| Sunny half-hours | CRPS and MAE on `sunny_clear_sky_index`, and descriptively on `sunny_unit_shape` |
| Cold nights | CRPS and MAE on `cold_nights` (sun more than 5 degrees below the horizon, ERA5 temperature at or below 3 degrees Celsius) |
| Slump share, injected events | `sign(d) * (q50 with shifted lags - q50 with unshifted lags) / abs(d)`, with `d` the depth as a share of capacity, averaged within each event and then over events; 0 for a forecast that keeps to the NRA, 1 for one that takes up the whole step. Also by days since onset, as in experiment 587 |
| NRA error, injected events | Mean of `sign(d) * (q50 with shifted lags - NRA actual)` in points of capacity, shifted minus unshifted, and the CRPS difference against the NRA actual |
| After an event | The slump share and the NRA error for origins in the 7 days after the event ends |
| Real events | The same formulas on the verified transfers of experiment 580; the December D5 and D16 pair is the real-event result, and the May events are flagged as unusable (experiment 582); figures per event, no interval |
| Labelled-event slices | CRPS and MAE on `event_rows` and `event_following_7_days`, as experiment 707's scorer reports them |

### Phases, fit counts, and the pre-registered extension

**The experiment commits to the minimal plan, about 380 fits and 8 to 10 GPU-hours, which is
comparable with the 8 GPU-hours of experiment 707.** The minimal plan holds three groups of fits.

- **Phase T, tuning folds, `-i` state:** the 15 ENS-only recipes (`M2`, the three `W` arms, `S`, the
  ten creative arms) plus `M1-i`, on the four tuning folds with seeds 0 and 1: 128 fits, labelled
  interim. `M3` needs no fit.
- **Phase S, scored folds, `-i` state:** `M1`, `M2`, `W-h_T`, the placebo `P`, and `B*` (and `O*`
  if it differs) on 12 folds and seeds 0 to 2, at most 180 fits. The stored `M1-n` and `M2-n` give
  the `-n` state of `M1` and `M2`.
- **Phase C, seeds 3 and 4:** `M1-i`, `M2-i`, and `B*-i` on 12 folds, 72 fits. Phase C is never cut,
  because the decision rule reads C1 on seeds 3 and 4 separately.

**The minimal plan reads C1 to C5, I1, I2, I4, and I5.** `HW730`, `S`, `W-h0.5`, `W-h1`, and `W-h2`
get tuning scores in the minimal plan, and a recipe that is not a scored arm gets no verdict.

**The extension is pre-registered and starts only after the minimal plan's results are reviewed and
the manager gives the word.** The full plan has about 960 fits and 19 to 25 GPU-hours (pre-registered
estimate from the design; the upper figure allows for the bank forecasts). The extension adds these
fits to the minimal plan, and no fit of the extension changes what the minimal plan has already
read.

- **The 120 interim `-n` fits of Phase T** (every recipe on the tuning folds without injection), for
  the per-recipe injection contrast on the tuning folds.
- **Phase S for the other core arms:** `S`, `W-h0.5`, `W-h1`, `W-h2`, and `HW730` in the `-i` state,
  and the `-n` state of the new recipes, which opens C6, C7, I3, I6, and the secondary contrast
  `B*-n` minus `M1-n`.
- **The two creative finalists** in Phase S, and `O*` if it lies outside them, and the `-n` seeds 3
  and 4 of `B*` (24 fits).
- **Cut order if the budget shrinks:** the `-n` state of `W-h1`, `W-h2`, `S`, and `HW730` first,
  then `W-h1-i`, then the `-n` seeds 3 and 4 of `B*`.

### Stop conditions and invalid conditions

**The experiment stops, and reports to the manager, in five cases.** A known-answer test fails. An
origin, cut, or injection guard raises. The one-fold timing pilot puts the total above the GPU-hours
the maintainer approves, or a fit above 16 GB of host memory. On the tuning folds, `M2-i` scores
more than 0.10 points worse than `M1-i` or than `M2-n`, which would point to a fault in the new loop.
Or the measured share of rows with a shifted lag lies outside the range fixed after the pilot. The
experiment is `invalid` if any row reads a run issued after its origin, a training row passes the
refit, a world changes a half-hour at or after the refit, arms are scored on different rows, or a
valid time on or after 2026-07-01 is scored.

### Compute and code

**All new code goes in `ml-experiments/00710-ens-only-one-shot-training-recipe/`, and nothing under
`src/` or `packages/` changes.** The scripts import experiments 707, 581, and 699 through
`PYTHONPATH`, and no script of those experiments is edited. The new code holds a training loop with
epoch-dependent weights, a sampler, a learning-rate schedule, and weight averaging; the weight
rules; the injection of a (fold, seed) world into copies of `build_fold`, `ens_rows`,
`fold_pretrain`, and `run_inputs`; the bank forecasts; the scorer; and the known-answer checks.
Every script gets a code review before its first run.

**The compute plan is checks on one core, a one-fold timing pilot, and a request to the manager.**
The pilot runs fold 2026-01, seed 0, arms `M1-i` and `W-h0.5-i`, with both bank passes, under
`MemoryMax=16G`, and records wall-clock, peak host memory, peak GPU memory, and the share of shifted
rows. Fits run only after the manager replies with cores and GPU memory. The expected request is 3
streams on 4 cores with one thread per stream and about 5 GPU gigabytes per stream (experiment 707's
unverified estimate). The workstation's GPU is used, so no money is spent.

### Risks

- **Seed noise is as large as the margin at the six,** so C1 at the six may end `inconclusive`.
- **The 12-month bootstrap undercovers,** so the t tests on the monthly differences are printed
  beside each interval.
- **A world is one draw per (fold, seed),** so the injection family rests on at most five draws per
  fold.
- **Short effective windows hold few events,** so a half-life of 0.5 years may teach injection
  little.
- **Only the December D5 and D16 pair is a clean real-event test,** so the real-event figures are
  descriptive.
- **`B*`, `O*`, and `h_T` are chosen on four tuning months with two seeds,** so the choice may
  differ from the one the scored year would make.
- **One-stage arms train at 5e-4 throughout and the reference ends at 2e-4,** so `W` minus `M2` mixes
  the weight path with the final learning rate.
- **The window length and the ENS resolution upgrade of 2023-06-27 are confounded** above 24 months
  (experiment 707), and `HW730` minus `M3` carries the confound.
- **The stored `-n` arms and the new `-n` arms come from two code paths,** which known-answer tests 1
  to 3 bound.

### Open questions for the maintainer

1. **Are 19 to 25 GPU-hours (about 960 fits) acceptable for the full plan?** The minimal plan needs
   8 to 10.
2. **Should the window start on 2020-10-15 (window C, lockdown rows kept with a `covid` input)
   instead of 2021-07-20?** Window F reuses the stored `A_F` and `R_F`.
3. **Is it right that an injected event takes its row's weight?**
4. **Is one world per (fold, seed), shared by every stage and epoch, acceptable?**
5. **Is the `-i` state the right state for the headline?**
6. **Is a margin of 0.02 the right bar for "matches"?** A stricter reading would ask the ENS-only
   recipe to beat the reference at the eight.
7. **Should a winning recipe be retested on the Candidate network in a later experiment?**

## Method

To be completed with the results.

## Files

To be completed with the results. The code will be in
`ml-experiments/00710-ens-only-one-shot-training-recipe/` and the private data in
`data/ml-experiments/00710-ens-only-one-shot-training-recipe/`. The design, its review, and the
findings are private files under `data/ml-experiments/briefs/ens_one_shot/`.
