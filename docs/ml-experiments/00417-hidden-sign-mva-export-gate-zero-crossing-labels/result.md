---
id: "00417"
slug: hidden-sign-mva-export-gate-zero-crossing-labels
title: >-
  Can a Primary-level export gate, or labels taken from the zero crossings of the daily reading,
  stop the signed mixture density network from inventing reverse flow at MVA-only Primaries?
theme: production-reference-and-best-guess
also_relevant_to:
  - solar-disaggregation
  - differentiable-physics-embedded-generation
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks (state files), per-row forecasts of the hidden pair, ERA5
  temperature arrays, and day-level diagnostic tables, all in this experiment's private folder.
  Designs: more than one is defensible (the Opus design note lists eight, A to H, and this plan
  fixes the order and the stop rule before any score is read). Callers: the code imports
  experiments 87, 397, 405, and 412 and copies experiment 412's fold builder with the changes listed
  under "Code". Serving path and degradation rule: not touched (this repository holds neither).
hypothesis: >-
  The false reverse flow that the signed network of experiment 412 produced at MVA-only Primaries
  came from a capacity fitted on the rows it labels, so a constraint that fixes the sign with
  information the capacity cannot fit from the same rows removes the false calls and keeps the skill.
  Two constraints are tested: a once-per-Primary export gate in front of the folded likelihood
  (design A, with a hard high-irradiance row gate H layered on it) and training labels taken from
  the two daily zero crossings of an export day (design C).
prediction: >-
  Written before any score is read. Z1 (zero-fit diagnostics, level 2.0): the day-level W-shape
  score (design C), the Sunday-and-holiday contrast (D), and the sunny-versus-cloudy slope (E) each
  separate injected from no-injection days or Primary-months with an area under the receiver
  operating characteristic curve of at least 0.9 for C and at least 0.8 for D and E. At level 1.0
  the areas fall, and at least one stays at 0.8 or above. The bars
  (P1 for design A, P3 for design C) are read on rotation r1, level 2.0, reactive floor 2 points,
  in the injected setting, pooled over the hidden pair D5 and D10: balanced accuracy of the sign at
  least 0.85 pooled and at least 0.70 at each hidden Primary, false-reverse rate at most 5%, and a
  margin of at least 0.1 balanced accuracy over the shuffled-sky arm; in the no-injection setting,
  false reverse at most 1% at D5. Prediction for A: it passes the D5 placebo bar (by construction,
  because a shut gate truncates the density at zero) and fails the 5% bar at the pair in June,
  which would show that a Primary-level gate fixes the placebo and not the pair (experiment 408's
  rule-only labels gave 3.8% to 6.1% at level 2.0). The hypothesis is refuted for A if the gate
  opens at neither injected Primary or shuts at one of them. P2 (A with H): removes at least half
  of the false calls of S1c-ref at the injected pair. P3 (C): passes all bars, the 5% bar included.
  P4 (power factor 0.97, reactive power proportional to the Primary's own demand): reported with no
  bar, because experiment 366's rule failed there; the W-shape area under the curve stays at or
  above 0.8.
decision_rule: >-
  The bars of P1 apply unchanged to A, A with H, and C, on rotation r1 only. Stop rule, registered:
  stop the signed-MW work if A (with or without H) and C both fail the bars on rotation r1 (S1c-ref,
  A, and C make about 20 fits; A with H and the power-factor fits bring the most to 32), or if every
  zero-fit diagnostic gives an area under the curve below 0.8 between level-2.0 injected days and
  no-injection days. Stopping means: publish unsigned MVA forecasts at the eight, label D12 alone
  "inferred, heavy reverse flow only", count the result as the eighth failed treatment, and add no
  further capacity priors. Design B is a follow-up, run only after the stop decision and only if A
  passes the placebo and leaves the pair above 5%. Passing means running rotations r2 and r3 for the
  winning design as a follow-up experiment, not here. No porting threshold applies, because the
  experiment is a lab test and nothing here ports to production.
benchmark: >-
  S1c refitted on the CAMS-and-ERA5 inputs (the reference of this experiment), S1c with the sky
  shuffled, and the no-injection placebo. Experiment 412's numbers are quoted for context only,
  because its demand module read the ECMWF ENS mean and spread.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Balanced accuracy, recall, and false-reverse rate of the predicted sign (P(X < 0) above 0.5 on
  gate-open half-hours with CAMS clear-sky index at least 0.7), the false-reverse rate on sunny
  summer middays with neither injected nor real reverse flow, signed pinball loss of the 11
  quantiles in points of effective capacity, and, for the diagnostics, area under the receiver
  operating characteristic curve.
date: 2026-10-30
code: ml-experiments/00417-hidden-sign-mva-export-gate-zero-crossing-labels/
take_home_message: >-
  Written when the experiment finishes.
production_status: not-for-production
adopted_in: none
---

**Status: planned. No fit has run and no diagnostic score has been read.** This plan follows the
Opus design note of 2026-10-30 (private folder: `briefs/neural_mva_new_ideas_answer.md`), which
proposes eight constraints (A to H) against false reverse flow at the eight Primaries whose meter
reads apparent power (MVA) only.

## Question and plan in plain words

**Some Primaries report only the size of the power flowing, not its direction, so a forecast cannot
say whether the flow is demand or solar generation pushing power back up the network.** Experiment
412 trained a neural network to infer the direction from the sunshine. With the direction hidden at
two Primaries where it is known, the network called reverse flow on 17% of the half-hours that had
none, and on 18% to 47% of sunny half-hours when no solar farm was added at all. This experiment
tests two constraints that decide the direction with information the network cannot fit from the
same readings: one yes-or-no decision per Primary about whether it exports at all, and training
labels from the two daily times at which the flow passes through zero.

## Evidence the plan follows

- **Experiment 412 merged as not supported (pull request 69).** The capacity prior was inert: 13
  nats against a loss over 5.6 million rows moved the capacities by under 0.2 points, and a prior at
  full weight pulled the capacity to the prior centre and removed the skill with the false calls
  (balanced accuracy 0.54 to 0.61, within 0.04 of the shuffled sky). This plan adds no capacity
  prior and keeps the registered, divided prior, whose gradient share the fit script prints in every
  arm that has it.
- **Experiment 408 is still a draft plan (pull request 64), with an interim block.** Its rule-only
  diagnostics (refit 2026-01-01) show the rule's gate opening on real readings at D10 and D12 only,
  so it stays closed at D5, and the rule's labels reaching recall 0.64 to 0.79 with a false-reverse
  rate of 3.8% to 6.1% at level 2.0. Design A copies that rule, so P1's 5% bar sits inside the range
  the rule already reaches, and the plan says so in the prediction.
- **Experiment 416 is a draft plan (pull request 73).** It tests an explicit solar term inside the
  demand network at the signed Primaries and has no results. Nothing in this plan depends on it.
- **Experiment 409's learned capacities were 0.40 to 0.50 of the studies page at five Primaries**
  (quoted by the design note), so design G (borrowed capacity) is not run.
- **Standing rules of 2026-10-30 applied.** Monthly refits only; cleaned data (experiment 397's
  rows) with the dropped count per fold reported; D15 and D18 cut before 2025-03-01; every fitted
  network saved as a state file; draft pull request; the seven-step review order (code review before
  the first fit, results review, write-up review); a day-ahead and a 14-day example forecast are
  not drawn, because this experiment scores the sign of reverse flow and not the forecast (stated as
  a departure).
- **Weather inputs are CAMS and ERA5 only, realised, in laboratory conditions; no ECMWF ENS.** CAMS
  gives the unit PV shape `u` and the clear-sky index `k` as in experiment 412. ERA5 gives
  temperature at each Primary (`era5_features.parquet` of the stage 1 pilot, stamped every 30 minutes to 2026-07-31,
  interpolated linearly to the half-hour's midpoint), the demand module's only weather input. ERA5 shortwave
  serves only as a cross-check of CAMS in the diagnostics. The day-ahead behaviour of every design
  stays untested.

## Code

All code is in `ml-experiments/00417-hidden-sign-mva-export-gate-zero-crossing-labels/`.

- `e00417_inputs.py` copies experiment 412's `site_arrays` and `build_fold`, and changes three
  things. The demand module's input columns are the irradiance-free columns of experiment 405 with
  every ensemble mean and spread column removed and one ERA5 temperature column added (standardised
  on the training rows). `Setting` gains a power factor with demand-driven reactive power, as in
  experiment 366: the reading is `sqrt(P'^2 + (tan(phi) y)^2 + Q^2)`, with `y` the Primary's own
  signed power before the farm is injected and `tan(phi)` from
  `studies.dp_operator.tan_phi_from_power_factor` (power factor 1 is experiment 412's reading). The
  reactive term follows the demand, so it fills the midday dip, which is what defeated experiment
  366's rule. The lags use the same definition. The fold also carries each row's valid time and `k`.
- `e00417_models.py` imports experiment 412's `SignedMdn`, mixture functions, and quantiles. It adds
  a negative log-likelihood with a per-row mask for where the folded density applies and an optional
  signed label for labelled rows.
- `e00417_gates.py` holds the Primary-level gate (design A: experiment 366's rule as experiment 408
  fits it, a Spearman correlation of the reading with `u` above +0.3 on sunny summer-midday training
  rows, or a 1st percentile below 3 points of capacity, which is 1 point above the reactive floor,
  because every folded reading is at least 2 points and only the correlation branch can fire. The
  +0.3 comes from experiment 366's results on these same Primaries, so the gate is not a clean
  out-of-sample choice) and the high-irradiance gate H (`u_hi(h) = D_ref(h) / C_max` with `D_ref(h)`
  the median reading at UTC hour `h` on overcast summer training rows (`k` below 0.3) and `C_max`
  the largest studies-page capacity (an external figure), in points, among the four signed Primaries
  that are not hidden, never the hidden pair; the gate is `u >= max(0.3, u_hi(h))` with `u` the
  scaled shape).
- `e00417_diagnostics.py` runs the zero-fit diagnostics of C, D, and E, and the zero-crossing
  labeller of design C.
- `e00417_fit.py` fits and forecasts, `e00417_score.py` scores (copying experiment 412's metrics and
  adding the capacity and the prior's gradient share), and `e00417_figures.py` draws the figures.

## Zero-fit diagnostics (run first)

**The diagnostics read the folded readings at the six signed-MW Primaries before any network is
trained.** Rows are unique valid times before 2025-07-01 (the window that sets the injection level),
with the minimum of `k` over `t0 ± 5 h` at least 0.7 on the day for C (a clear day) and `k` at
least 0.7 at the half-hour for D and E. Settings:
level 2.0 (reactive floor 2 points), level 1.0, level 2.0 at power factor 0.97, and no injection.
The no-injection group is D5, D9, D11, D14, and D16, and D10 is reported on its own, because its 130
real reverse half-hours make it no clean placebo.

- **C, the W-shape score.** On each sunny day, with `t0` the solar noon: the left minimum is the
  smallest reading in `t0 - 5 h` to `t0 - 1 h`, the right minimum the smallest in `t0 + 1 h` to
  `t0 + 5 h`, and the centre maximum the largest in `t0 - 1 h` to `t0 + 1 h`. The score is the smaller
  of the two differences (centre maximum minus each minimum), in points of capacity. It is positive
  when the reading crosses zero twice and negative on a day with one smooth midday dip. The area
  under the curve compares injected days with no-injection days, per Primary and pooled. The
  labeller calls the half-hours between the two minima reverse flow on days whose score exceeds a
  threshold, which is the 99th percentile of the score over the no-injection days of the four
  visible signed Primaries (D9, D11, D14, D16), so the threshold uses no information from the hidden
  pair. In a fit the threshold is recomputed per refit from the rows `build_fold` selects (up to the
  refit minus the cutoff age, inside the 730-day window), and its value is reported per refit. The
  diagnostics also report the labeller's recall and false-positive rate at that threshold on D5's
  injected and no-injection days.
- **D, the Sunday-and-holiday contrast.** For each clear Sunday or bank holiday, the difference
  between its midday reading (10 to 14 UTC) and the mean midday reading of clear Tuesdays to
  Thursdays within 14 days, after removing a linear ERA5-temperature term fitted on those days. The
  area under the curve compares the differences of injected and no-injection pairs. The sign of
  the same difference at night (22 to 04 UTC) is reported beside it as a control.
- **E, the sunny-versus-cloudy slope.** In each Primary-month, the slope of the midday reading on
  `k` after removing a mean for each half-hour of the day and ISO week, with ERA5 temperature as a
  covariate. The area under the curve compares injected and no-injection Primary-months.
- **Reported with each area:** the number of units behind it (days, pairs, or Primary-months), and
  its 95% interval from a bootstrap that resamples whole days within each Primary (E: Primary-months
  within a Primary). The bars (0.9 for C, 0.8 for D and E) and the stop rule's 0.8 apply to the
  pooled point estimate, and the per-Primary areas are reported beside it. D's temperature term is
  fitted separately for each Primary.

## Fits (after the code review, with the manager's slot)

**Every arm runs on experiment 412's harness: rotation r1 (D5 and D10 hidden), level 2.0 with
Q = 2 points, test months 2026-03 and 2026-06, one seed, one GPU process.** One arm is 6 fits:
injected, no-injection placebo, and shuffled sky, each in both months. The power-factor setting
adds 2 fits (injected, both months).

| Order | Arm | What it is | Fits |
|---|---|---|---|
| 1 | S1c-ref | Experiment 412's S1c on the CAMS-and-ERA5 inputs | 6 + 2 |
| 2 | A | S1c with the Primary-level gate: a Primary whose gate is shut trains and forecasts with `X >= 0` (the truncated density everywhere) | 6 + 2 |
| 3 | A+H | A with the high-irradiance row gate H on the Primaries whose gate is open | 6 + 2 |
| 4 | C | S1c with the zero-crossing labels as signed targets on labelled rows | 6 + 2 |

Design B (demand frozen off the gate) runs only if A passes the placebo and leaves the within-pair
false-reverse rate above 5%. Designs D and E enter the fits only if their diagnostics pass and A
and C fail. Designs F and G are not run.

- **Gate decisions are printed before any fit.** Design A differs from S1c-ref only where the gate
  is shut, so A at a hidden Primary whose gate is open is the same fit as S1c-ref and its result is
  copied from S1c-ref (same seed), not refitted. A shut gate at D5 in the no-injection setting gives
  zero false reverse by construction, which tests the gate and not the network, and the write-up
  says so. A cannot lower the false-reverse rate at a Primary where the gate is open. A+H differs
  from A only through the row gate H, and is fitted in full.
- **Gate timing.** Design A's gate is computed once per Primary per refit from the folded readings
  in the training window (rows before the refit), using the real `u` and `k`, also in the
  shuffled-sky arm.
- **Labels.** Design C's labels are computed once per Primary per refit from the training window
  only. Only days whose W score exceeds the threshold are labelled: reverse between the two minima,
  forward outside them. Labelled rows use the signed density at `y' = +-sqrt(max(r^2 - Q0^2, 0))`
  with no folding. Every other row, including sunny days below the threshold, keeps the folded
  likelihood with experiment 412's `u >= 0.3` gate, so moderate export is never taught `X > 0`.
- **Compute.** Experiment 412 timed a fit at 29 to 83 seconds, 45 seconds to 3 minutes with the fold
  build, so 32 fits take 25 minutes to 1.6 hours on one GPU process under 8 GB with one feeder
  core, run as `systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue`, with `free -g`
  checked before each launch and the run stopped if available memory falls below 12 GB (the
  manager's limit for this experiment). Refits are monthly. No stored forecast is reused, because
  experiment 412 read ENS inputs. State files go to `models/<arm>/` in this experiment's private
  folder.

## Scoring

**The scoring is experiment 412's, unchanged:** the headline slice is gate-open half-hours (`u >=
0.3`) with `k` at least 0.7, a predicted reverse flow is `P(X < 0) > 0.5`, the false-reverse rate
has as its denominator the sunny summer-midday half-hours with no injected and no real reverse flow,
and every recall is reported with its half-hour count. The prior penalty's share of the capacity
gradient is printed for every arm. Tail calibration (coverage and pinball loss of the 0.01, 0.05,
0.95, and 0.99 signed quantiles) is reported per Primary beside the sign scores, and a
signed-median-against-truth example is drawn at D5 and D10. The reference's absolute numbers may
differ from experiment 412's, because the inputs differ. The first three arms' numbers are interim
(two months, one seed, point estimates only).

## Code review before the first fit

A fresh Sonnet reviewer found no leakage and no blocking problem. Accepted: the W-score centre
window was ±0.5 h and is now ±1 h as planned (the diagnostics were re-run); `bars` now scores the
registered bars. Noted, not changed: the capacity prior centre is the median studies-page capacity
of all six signed Primaries, the hidden pair included, as in experiment 412 (external information,
not the hidden sign, and the prior is inert); the ERA5 file is stamped every 30 minutes, so the
midpoint interpolation averages two samples; a Primary with fewer than 200 sunny summer-midday
training rows gets a shut gate by default, and the row count is stored beside the decision; the
bootstrap resamples injected and no-injection days separately although they are paired, and D10
enters the injected pool only; the 1st-percentile branch of the gate does fire (at D5, D8, and D10,
whose readings sit at the 2-point floor), contrary to an earlier reading of the plan.

## Departures from the design note

- **The H arm is A with H layered on top**, not H alone, and the D5 placebo "should still fail with
  H alone" is not tested.
- **Design C omits the note's CAMS filter that discards a minimum coinciding with a dip in CAMS
  irradiance.** The day filter (`k` at least 0.7 across `t0 ± 5 h`) removes cloudy days instead.
- **Design A reuses S1c-ref's fits** where the gate is open (see "Fits").
- **Power factor 0.97 uses reactive power proportional to the Primary's own demand**, as experiment
  366 did, with the 2-point floor kept.
- **Designs B, D, and E run only as follow-ups after the stop decision.**
- **Tail calibration and example plots follow `common.md`;** day-ahead and 14-day forecast graphs
  are not drawn, because the experiment scores the sign.

## Risks

- **The shuffled sky keeps the daylight profile** (experiment 405), so it may score above chance; P1
  asks for a margin of 0.1 and not for chance.
- **A Primary-level gate decides once, so a wrong decision costs the whole Primary.** The
  diagnostic of the gate at the injected pair and the placebo is reported before the fit.
- **Reactive power at power factor 0.97 raises the readings the likelihood treats as real power**,
  which experiment 366's rule could not survive; the 2 power-factor fits report how far design A
  and C degrade.
- **The ERA5 temperature is realised weather**, available about 5 days late, and the CAMS inputs
  about 1 day late, so every result is a laboratory result.
- **Moderate export (level 1.0) may give no W shape**, so design C may only work at heavy export.
  Level 1.0 is run in the diagnostics only.
- **Unlogged load-shift events** may sit in the readings, as NGED says its logs are incomplete.

## Problems caused by the move

None so far.
