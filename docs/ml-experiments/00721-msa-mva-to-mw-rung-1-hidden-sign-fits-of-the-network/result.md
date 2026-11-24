---
id: "00721"
slug: msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network
title: >-
  Can the axial-attention demand network learn the sign of a Primary's power flow from a magnitude
  reading, when the sign of two signed Primaries is hidden and a real solar farm's output is added
  to them?
theme: solar-disaggregation
also_relevant_to:
  - demand-inputs
status: planned
kind: other
size: >-
  Complex. (1) The experiment stores about 53 fits (network weights and per-row forecasts, all
  private) for two injected worlds and one synthetic world; (2) more than one design defensibly
  tests the hypothesis (a gross-demand read-out, a learned sign head, and a hindsight posterior),
  and the plan ranks them; (3) the experiment copies training code from experiments 713 and 716 and
  injection code from experiment 720, whose callers cannot be named without searching the
  repository; (4) the experiment touches no serving path or degradation rule; (5) the experiment
  changes no contract. Reviews: plan (Opus), code (Opus, twice, before the first fit), results,
  write-up, prose.
hypothesis: >-
  A learned sign head, which reads each history token of the axial-attention demand network and is
  trained on the folded readings of signed Primaries, lowers the signed continuous ranked
  probability score (CRPS) on the reverse-flow rows of two Primaries whose sign is hidden and into
  which a real solar farm's recorded output is injected. The comparison is against the same network
  without the sign head (`SG-0`). The reason is that gross demand at a half-hour of the day and day
  type does not depend on the sky once calendar and temperature are given, so the sky separates
  export half-hours from import half-hours. The gain comes from the actual sky: the same network
  given the sky of 28 days earlier does not gain.
prediction: >-
  Written before any fit, and before the code of this experiment exists. `SG-0` recovers 0.70 to
  0.95 of the gap between `P0` and `S` on the reverse-flow rows of `W2`, because experiment 420's
  output-only recipe already scored the same as its arm A on those rows. `SG-A` minus `SG-0` on
  those rows lies between -0.80 and +0.40 points of capacity, so G1 (a gain of at least 0.404
  points) is the likeliest miss, and G3 (no invented reverse flow at D5 with the gate forced open)
  is the second likeliest, because three earlier fits invented reverse flow at D5. `SG-A` passes the
  synthetic world on sign accuracy, and fails it on the shifted-sky margin or the false-call rate
  with a probability near one in three. The predicted verdict is `not supported` or `inconclusive`.
  The prediction fails if `SG-A` passes G1, G2, and G3.
decision_rule: >-
  Counted readings G1, G2, and G3, each read at 98.33% (1 - 0.05 / 3) from a day-block bootstrap on
  calendar days, paired on valid time, with `studies.bootstrap.bootstrap_difference_at_level`. G1:
  signed CRPS on the reverse-flow rows of `W2` (rows whose signed truth is below zero, injected or
  real), `SG-A` minus `SG-0`, has a point estimate at or below -MDE and an upper interval end below
  zero, where MDE is the larger of 0.404 points of effective capacity (experiment 420's rotation r1
  figure) and the absolute difference between two seeds of `SG-0` on fold 2026-05, printed before
  any contrast is read. G2: `SG-A` minus its shifted-sky placebo on the same rows has the same bar.
  G3, no harm in `W0`: signed CRPS over all rows of the hidden pair, `SG-A` minus `P0`, at most
  +0.05 points pooled and at most +0.20 at either Primary, read on the plain `W0` fits and on the
  `SG-A-open` fits (the export gate forced open). A synthetic world precedes the real worlds. A
  treatment passes it when its sign accuracy outside a 2-point dead band is at least 0.95 at the two
  exporting synthetic Primaries, the treatment calls reverse flow on under 1% of sunny midday
  half-hours at the two other Primaries, and its sign accuracy under the real sky exceeds its
  accuracy under the shifted sky by at least 0.1 (the margin of rung 0b's stop rule). Verdict: the
  first row that matches. (1) `invalid`: a correctness check fails (the holdout guard, the fold
  point of the loss, the origin-time weather assertion, the no-two-worlds figure guard, or a
  known-answer test on synthetic arrays). (2) The synthetic world fails: the line stops, and the
  verdict is `inconclusive` when any fit of the failing treatment hit the 400-pass cap, and `not
  supported` when every fit stopped by patience. (3) G3 fails, or the quick look's harm stop fires:
  `not supported`, and the line stops under experiment 420's registered rule of no further gates or
  capacity priors. (4) `supported`: G1, G2, and G3 pass. (5) `not supported` for the history sign:
  G1 passes against `P0` only, and the output-only result of `SG-0` is reported. (6) `inconclusive`:
  every other combination, including a G1 or G2 miss whose treatment fits in `W2` are all flagged
  `stopped while falling`, reported with the counts. `S` and `FL` carry no bar. G4, the folded CRPS
  at the eight MVA-only Primaries from the same fits, and every share of the gap from `P0` to `S`
  carry no bar. A rerun is the same attempt only when a check that reads no bar statistic fails
  before the fix and passes after the fix, the fix changes no constant, an Opus review confirms the
  fix, and both outputs are kept. `inconclusive` and `not supported` are never rerun. Named
  constants (each a `Final` of `e00721_core.py`, set before any real data are read): 0.404, 0.05,
  0.20, 0.95, 0.1, 1%, 2 points, 28 days, 0.3 (the gate's `u_min`), 400 and 30 (synthetic passes and
  patience), 98.33%, 2.0 (injection level), and 0.20 (harm limit at the quick look).
benchmark: >-
  `P0` (the network on the folded readings, read as signed) as the low reference; `S` (signed
  readings given) as the ceiling; `SG-0` (experiment 420's output-only recipe on this network) as
  the control that isolates the history sign; the shifted-sky placebo of each treatment; experiment
  420's stored arm A forecasts on the rows they share with this experiment, stating the
  intersection; the stored `M*` of experiment 713 and F7's stored forecasts where stored rows cover
  the scored keys.
row_set_id: >-
  backtest_2025_11_to_2026_06_daily_12utc_before_holdout, restricted to the four folds 2025-12,
  2026-04, 2026-05, and 2026-06 (experiment 713's row set, leads from 24 h up to but not including
  240 h after the 12:00 UTC origin), and to the hidden pair D5 and D10 for G1 to G3. Fold 2025-12 is
  the dark fold, kept for the no-harm reading. No valid time on or after 2026-07-01 is read.
metric: >-
  Signed CRPS in points of capacity (twice the mean pinball loss over 11 quantiles) on the
  reverse-flow rows of the hidden pair, and on all rows of the hidden pair; the folded CRPS at the
  eight MVA-only Primaries and at the hidden pair, with the mean absolute error of the median
  forecast normalised by capacity beside each; sign accuracy and area under the receiver operating
  characteristic curve of the history-token sign posterior, with a 2-point dead band; the share of
  the gap from `P0` to `S` that each treatment recovers; q01 coverage on reverse-flow rows. Sign
  accuracy and area are descriptive in the real worlds and carry bars only in the synthetic world.
---

**We ask whether a forecasting neural network can learn the direction of power flow through a
Primary substation whose meter reports only a magnitude.** A Primary substation steps the voltage
down to 11 kV. Eight of the 16 Primaries report apparent power in megavolt-amperes (MVA), a
magnitude with no sign, so a Primary that exports solar output shows a rise in its reading and not
a negative value. We test the idea where the answer is known. We hide the sign of two Primaries
that do report a signed value in megawatts (MW), add the recorded output of a real solar farm to
each so that they export on sunny middays, and compare a network that has a sign-inference
component with the same network without the component. The score is the signed forecast error on
the half-hours when the true flow is an export. Every score is a ceiling, because the history rows
read recorded weather that a live forecast would not have. This page is the plan: nothing has been
fitted, and the code does not exist yet.

**Rung 1 is the second rung of a ladder that starts with experiment 720.** Rung 0 (experiment 720)
tests, on the CPU and without a neural network, whether the sign can be recovered in hindsight.
Rung 1 (this experiment) asks the same question inside the forecasting network that the
experiments 713 to 719 build. Rung 2 would repeat the test on more Primaries and score the eight
MVA-only Primaries against the stored network. This page covers rung 1 only.

**The terms below are used on the whole page.**

- **MSA network and `M*`:** the axial-attention demand network of experiment 713, and its selected
  arm `M1`.
- **The six signed-MW Primaries:** D5, D9, D10, D11, D14, and D16.
- **The eight MVA-only Primaries:** D2, D3, D4, D8, D12, D19, D21, and D22.
- **The hidden pair:** D5 and D10, the two signed-MW Primaries whose sign is hidden in rung 1
  (rotation r1 of experiment 420). D10 holds real reverse flow on 0.76% of its half-hours, and D5 is
  the Primary at which three earlier fits invented reverse flow.
- **Folded reading:** the magnitude `sqrt(X² + Q²)` of the signed power `X`, with a reactive floor
  `Q` of 2 points. A point is 1% of the Primary's capacity. "Hidden" means that the network sees
  only the folded reading.
- **Reverse-flow rows:** the scored half-hours of the hidden pair on which the signed truth is
  below zero, injected or real, as in experiment 420.
- **Injection level 2.0:** a real solar farm's recorded output, added at twice the host's median
  absolute reading on sunny summer middays before 2025-07-01. The farm's series is a private input
  and appears on no page.
- **Worlds `W2` and `W0`:** `W2` folds the hidden pair after the injection. `W0` folds the same
  pair with no injection, and is the placebo that must show no invented reverse flow.
- **Export gate:** the per-Primary switch of experiment 420, set once per refit from the readings,
  that allows a Primary to export at all.
- **Sign posterior `ρ`:** the probability that a history token exported, from the gross-demand
  read-out or from the learned head.
- **Signed CRPS:** the continuous ranked probability score of the 11 signed quantiles against the
  signed truth, in points of capacity. A score below another score is better.

## Where this experiment sits in the queue

**Rung 1 depends on F7's `HT` code, which is reviewed and running, and not on the five-seed verdict
of F7.** The `HT` code supplies the PV estimate `p` in the history rows, the target-row forecast
`p_0`, the as-of capacity table, and the 28-day shifted-sky placebo. Experiment 716 and 717's code
is imported from their folders on `PYTHONPATH` and not copied. **[Decided by the manager]**

**Rung 1 sits ahead of F9 in the GPU queue: F4, then F7, then rung 1, then F9.** The earlier design
note placed rung 1 behind F4 and F9 and after an F7 verdict. The Opus review of the queue found that
F9 answers no question directly and has the weakest prior, and that rung 1 is the direct test of
the always-megawatt forecast. **[Decided by the manager]**

**With the decisions below ruled and the first Opus code review passed, this plan is ready to start
its synthetic world.** No step before the synthetic world waits for rung 0b, for F7's five-seed
verdict, or for F9. The next action is the code, which a Sonnet agent writes from this page,
followed by the Opus review. Nothing on this page has been run.

| Prerequisite | State on 2026-11-24 | Blocks the synthetic world? |
|---|---|---|
| F7's `HT` code (experiments 716 and 717) | Reviewed; part 1's fits are running | No: the code is imported as it stands |
| F7's `HT` forecasts on the real folds | Not stored yet: the stored forecasts are arms `H` and `Z` on folds 2025-11, 2026-01, 2026-05, and 2026-06, seed 0 | No: only the `S` reference of `W0` reads them, and falls back to a fit |
| Experiment 720's injection code | Written and tested on synthetic worlds; unmerged | No: imported on `PYTHONPATH` |
| Rung 0b's real run | Not run; waits for the maintainer's choice between the two options of the rung 0b note | No: only the `SG-E` block waits for it |
| Experiment 713's known-answer gate | Waived by the maintainer; every verdict carries the waiver sentence | No |

## What this plan changes from the design note

**The design note (`mva_to_mw_design.md`, section 5.2) fixes the arms, the readings G1 to G4, and 71
fits at about 14 GPU-process-hours.** Four orders from the manager, applied before any data, change
the plan. Each change is listed with its reason.

| Design note | This plan | Reason |
|---|---|---|
| Synthetic fits: not stated, with the harness of experiment 713's gate in mind | 400 passes, patience 30, stage 1 only | Finding C1 of the F4 audit (next section) |
| `S` fitted in `W0` on 4 folds | Reuse F7's stored `HT` forecasts where stored, with one from-scratch `S` fit as a recipe check | The two recipes differ only by the warm start; saves 4 fits (about 0.8 GPU-process-hours) |
| `SG-E`, `SG-E-S`, and `SG-E-open` in the first wave (about 16 fits and 4 synthetic fits) | Out of the first wave; reconsidered if rung 0b reads `partly supported` | Rung 0b's synthetic worlds show the anchored read-out losing to a reading-blind reference |
| Second seed of the winner only; no rule for further seeds | Quick look first; seeds 2 to 4 only when the futility test does not fire, and outside the projection | The Opus review of the queue: month-to-month variation, not seed noise, sets the interval width |
| `SG-A`'s farm injection at the visible Primaries stated as a condition | Built in the first code review, or `SG-A` leaves rung 1 and the plan stops before `W2` | With `SG-E` out, `SG-A` is the only treatment |

## The training budget, checked against the F4 audit's finding C1

**The synthetic worlds train at up to 400 passes with a patience of 30, and not at the 150 passes
and patience 8 that failed the F4 gate.** The F4 audit's finding C1 is that the known-answer
harness of experiment 715 used experiment 713's budget from before its amendment 3. The W3 fits then
stopped at passes 42 to 54 while the validation loss was still falling by about 0.05 every 10
passes. On world W0, 713's arm `M` needed 350 passes, and under 715's cap the same pair of fits
stopped at pass 143 and at the cap of 150. Experiment 713's amendment 3 raised the synthetic budget
to 400 and 30 for this reason, and experiment 716's arm `M` of its known-answer worlds uses the
same pair.

| Fits | Passes | Patience | Why |
|---|---|---|---|
| Synthetic world of rung 1 (stage 1 only) | At most 400 | 30 | A hard stop gate must not fail for a harness reason, as the F4 gate did |
| `W2` and `W0` real folds | Experiment 713's real-fold budget, with its rerun at the larger cap | 8 | `P0` and `S` stay comparable with `M*` and F7; the fit prints its stopping pass |

**The real folds keep experiment 713's budget, and a flag marks any fit that may have stopped
early.** A real fit is flagged "stopped while falling" when its validation loss over the last 10
passes before the stop fell by more than the 10-pass noise of that fit's own validation loss. The
flag changes no number, and verdict row 6 uses it. The audit of F7's arm `H` (experiment 716's KF9)
found the same failure on real folds, where patience 8 picked passes 4, 23, and 5 while the path
was still learning at pass 60. Every fit's stopping pass, and the learned gain of each new path as
a share of its reachable range, are printed and appear in the write-up.

**The cost of the larger synthetic budget is inside the design's figure of 0.2 GPU-process-hours
per fit.** Experiment 713's gate, at 400 passes and patience 30, took 1,985 s for all its fits
(run log, second rerun). The figure of 0.2 covers a rerun at the pass cap. If the first synthetic
fit measures a different time, the manager is told before the second fit starts.

## Worlds, arms, and folds

**Three worlds are fitted, and each treatment is tested in the order synthetic, `W2`, `W0`.**

- **Synthetic world (stop gate).** Gross demand is a fixed function of calendar and temperature,
  `X = D - C u`, and the reading is `sqrt(X² + Q²)`. Two synthetic Primaries export on clear
  middays and two never do. The exporting pair holds the real export share of D12's history
  windows, 70% to 93% of days from March to September, because a world with balanced windows
  cannot show the failure of a read-out that starts from "every row imported".
- **`W2` (injected).** Folds 2026-04, 2026-05, and 2026-06 are sunny, and fold 2025-12 is dark and
  tests for harm. Every arm of `W2` reads the same readings. Only the loss of `SG-A` reads the true
  sign of the four visible Primaries (D9, D11, D14, and D16), into which the two other farms'
  recorded output is injected so that `SG-A` has positive labels.
- **`W0` (no injection).** The same four folds, with the hidden pair folded and nothing added.

**The arms are listed with their role, and every arm of the first wave fits from scratch.** The
fits load no stage-1 state and no frozen encoder of the Candidate, because both were trained on the
hidden pair's signed readings. The exception is the reused `S` of `W0`, described below.

| Arm | What it is | Role | In the first wave |
|---|---|---|---|
| `P0` | `M*` recipe on the folded readings, output read as signed | Reference: the present network, a low bar | Yes |
| `S` | `M*` recipe with F7's `HT` inputs on the signed readings | Ceiling: the sign is given | Yes |
| `FL` | `M*` with the folded loss and a free signed mixture, no PV term (`W2`, one fold, seed 0) | Negative control; the symmetry of the folded loss is known | Yes |
| `SG-0` | `HT` with the sky-blind head, the gate, the folded loss, the target term `-c_i p_0`, and `ρ` fixed at 0 | Experiment 420's output-only recipe on this network: the control that isolates the history sign | Yes |
| `SG-A` | `SG-0` plus the learned sign head | Treatment | Yes |
| `SG-A-S` | `SG-A` with `p` and `p_0` from the sky 28 days earlier (`W2` only) | Placebo: the sky must carry the gain | Yes |
| `SG-A-open` | `SG-A` in `W0` with the gate forced open at the hidden pair (one seed, four folds) | Tests a gate where the ungated network fails | Yes |
| `SG-E`, `SG-E-S`, `SG-E-open` | `SG-0` plus the anchored gross-demand read-out, with its placebo and forced-open fit | Treatment 1, held back | No: see below |

**The learned sign head reads each history token after the axial trunk.** For a history token `x`
of a gate-open Primary, `ρ = e_i × 1[u ≥ u_min] × sigmoid(w · LayerNorm(x) + b)`, where `e_i` is the
gate, `u_min` is 0.3, and `b` starts at -4, so an untrained head gives `ρ` near 0.02. In training, a
sample from a signed Primary has its history rows folded with probability 0.5, the target loss
stays signed, and a cross-entropy loss between `ρ` and the true sign is added. D15 stays out of
`SG-A`'s supervision; a report with and without D15 is a read-out and carries no bar.

**The likelihood gate `1[u ≥ u_min]` reads the real sky in the placebos, as in experiment 412, so a
placebo keeps some real-sky information that the margin of G2 must absorb.** The write-up says so.

### The `SG-E` family is held back, and rung 0b decides whether it returns

**Experiment 720's rung 0b has not been run on real data, and its synthetic worlds say the anchored
read-out loses.** In 10 synthetic exporter worlds, including worlds with the PV estimate off by 25%
a day and demand with a 5-point day offset, the anchored read-out lost to the reading-blind
reference by 0.07 to 0.12 in area under the curve, with every upper interval end at or below +0.017.
The hindsight read-out beat the reading-blind reference in all 10 worlds. The `SG-E` family repeats
the anchored read-out inside the network and would cost about 20 fits (4.0 GPU-process-hours).

**This plan reconsiders `SG-E` if rung 0b reads `partly supported`, and its recommendation on the
evidence available now is to keep `SG-E` out.** The rule is written before any data:

- **Rung 0b reads `supported`, or its second arm (the PV estimate as the prior) passes its bar:**
  the `SG-E` family is added as a second block, and the cost is stated to the manager first, because
  the total then exceeds the cap (next section).
- **Rung 0b reads `partly supported`:** `SG-E` is reconsidered before any `SG-E` fit. The default is
  that `SG-E` stays out, G3's forced-open check stays on `SG-A-open`, and the page says why. Whether
  `SG-E` returns is decision 6.
- **Rung 0b has not been read when rung 1's `W0` finishes:** `SG-E` stays out, and the page says
  that its absence is untested.

**Any new read-out, such as a hindsight posterior (`SG-H` of addendum 1), needs an amendment before
data and its own review.** Neither is in this plan. Addendum 1 recommends neither before rung 0e's
first known-answer score has been read.

### The `S` arm of `W0` reuses F7's stored forecasts

**In `W0`, `S` reads the real signed readings, so F7's arm `HT` already holds the same forecast
inputs.** F7 fits `HT` from a warm start at `M*`'s stage 1, and rung 1's `P0`, `SG-0`, and `SG-A`
fit from scratch. The two recipes of `S` differ only by that warm start, because stage 1 of `M*`
also saw the signed readings. The plan therefore reads `S` for `W0` from F7's stored `HT` forecasts
(seed 0) on every fold where the forecasts are stored and the row keys match, and fits `S` only for
a fold where the forecasts are not stored. The stored forecasts of `W2` cannot be reused, because
the injection changes the readings.

- **Recipe check.** One from-scratch `S` fit in `W0` on fold 2026-06, seed 0, is compared with the
  stored `HT` forecast of the same fold. A difference above the seed difference of `SG-0` on fold
  2026-05 is reported as a caveat on every "share of the gap from `P0` to `S`" figure.
- **State of the stored forecasts.** On 2026-11-24 the stored forecasts are arms `H` and `Z`, not
  `HT`, on folds 2025-11, 2026-01, 2026-05, and 2026-06. Only folds 2026-05 and 2026-06 overlap
  rung 1's folds. F7's phase 5 stores `HT` for eight folds. If phase 5 has finished when `W0` is
  scored, folds 2025-12 and 2026-04 are reused too, and otherwise `S` is fitted there (four fits at
  most, about 0.8 GPU-process-hours).

## Counted readings, verdict rows, and the stop rule

**The three counted readings are G1, G2, and G3, and G4 is a report.** The decision rule in the
front matter states each bar. In words:

- **G1, the gain from the history sign.** `SG-A` minus `SG-0`, signed CRPS on the reverse-flow rows
  of `W2`. Treatment minus `P0` is reported and carries no bar, because experiment 420 found the
  ungated network scored the same as its arm A on those rows (6.99 against 6.98), so any arm that
  can go below zero beats `P0`.
- **G2, the sky carries the gain.** `SG-A` minus `SG-A-S` on the same rows, with the same bar.
- **G3, no invented reverse flow.** In `W0`, `SG-A` minus `P0` over all rows of the hidden pair, on
  the plain fits and on `SG-A-open`. The write-up reports each gate decision per world and fold,
  because the gate is expected to stay shut at the un-injected pair, and a shut gate passes G3 by
  construction.
- **G4, the eight MVA-only Primaries.** The folded CRPS of each arm at the eight, from the same
  fits, with its interval and no bar. Two seeds detect a difference of 0.25 points only, and 0.090
  points is the five-seed noise floor.

**The smallest gain worth porting is not set here, because this experiment ports nothing.** A
`supported` verdict allows rung 2 to be planned. Rung 2 sets its own threshold against the cost of
adding the sign head to the network.

**The experiment can fail in every row.** A miss of G1 is the likeliest outcome, and the page then
reports the output-only result of `SG-0` as the finding. The verdict reads the treatment that wins
G1 on seed 0, and the page says that the choice was made on the scored rows. A pre-registered gate
stays pre-registered: a reading chosen after the scores were seen is labelled post hoc and changes
no verdict.

**Count and stop rule.** The design note counts the signed-MW line's earlier tests in experiments
152, 372, 374, 387, 408, 412, 417, 420, 424, and 427, and the sign-experiment count of the later
amendment lists 18 earlier experiments, with 722 the 19th, 723 the 20th, and 724 the 21st. **This
page says honestly that rung 1 is another experiment that opens a gate to a sign label.** Its
`supported` verdict admits a treatment to rung 2 and so to a signed forecast at D12, which is an
outcome that the stop rule's definition counts. Experiment 420 tested the same kind of statistic in
the same kind of world. The recommendation is that rung 1 is the 22nd counted experiment. The
decision belongs to the maintainer (decision 1).

## GPU projection

**The first wave is 53 fits and about 10.6 GPU-process-hours, inside the cap of about 14
GPU-process-hours that the standing approval covers.** The figure of 0.2 GPU-process-hours per fit
is the design note's, from experiment 713's measured solo fits of 0.09 to 0.19 and a margin for
predictions, fold builds, and reruns at the pass cap. The first fit measures the real figure.

| Arm | Synthetic (2 seeds) | `W2` (4 folds, seed 0) | `W0` (4 folds, seed 0) | Second seed | Fits | GPU-process-hours |
|---|---|---|---|---|---|---|
| `P0` | 0 | 4 | 4 | 0 | 8 | 1.6 |
| `S` | 0 | 4 | 1 (recipe check) | 0 | 5 | 1.0 |
| `SG-0` | 2 | 5 (includes seed 1 on fold 2026-05) | 4 | 0 | 11 | 2.2 |
| `SG-A` | 2 | 4 | 4 | 8 (both worlds) | 18 | 3.6 |
| `SG-A-S` | 2 | 4 | none | 0 | 6 | 1.2 |
| `SG-A-open` | none | none | 4 | 0 | 4 | 0.8 |
| `FL` | none | 1 | none | 0 | 1 | 0.2 |
| **First wave** | **6** | **22** | **17** | **8** | **53** | **10.6** |

| Item | Fits | GPU-process-hours | Condition |
|---|---|---|---|
| `S` fitted in `W0` where F7's `HT` is not stored | up to 4 | up to 0.8 | Total 11.4 |
| `SG-E` family (synthetic 4, `W2` 8, `W0` 8) | 20 | 4.0 | Rung 0b opens it; total 14.6 to 15.4, which needs the manager's approval |
| Seeds 2 to 4 of `SG-0` and `SG-A` on `W2` (24 fits) and of `SG-A-S` (12 fits) | 36 | 7.2 | Outside the projection: the futility test does not fire, and the manager approves |
| The design note's figure for comparison | 71 | about 14 | Fits `SG-E` and `SG-A`, and `S` in `W0` |

**The compute rules of the slot hold for every fit.**

- **Slot and queue.** The GPU slot comes from the manager. One worker fits at a time, because four
  concurrent fits gave no gain. The first wave takes about 10.6 hours of wall clock. The queue is
  F4, F7, rung 1, F9. Rung 0b and the CPU work run beside rung 1 and use no GPU.
- **Memory per fit.** The host cap is `MemoryMax=12G`. The references are experiment 713's gate
  fits (GPU peak 5,090 MiB, host peak 6.3 GiB) and experiment 715's gate fits (GPU 10.6 to 11.7
  GB, host peak 7.1 GB). Rung 1's first fit measures its own peak and the cap is set from it.
- **Every command** runs in `systemd-run --user --scope` with `MemoryMax`, `MemorySwapMax=0`, and
  `OOMPolicy=continue`, through the `.venv/bin/python` interpreter and never `uv run` inside the
  scope, with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `POLARS_MAX_THREADS`,
  and `RAYON_NUM_THREADS` set to 1, after a check that `free -g` shows at least 25 GB available. No
  test runs with `pytest -n`.
- **Messages to the manager.** The session sends "fits starting" before each family of fits (the
  synthetic world, the quick look, the rest of `W2`, `W0`, the second seed) and waits for the reply.
  The session sends "last fit ended" after each family. A fit that hits its pass cap is reported
  with its stopping pass.

## Stages, the quick look, and the seeds

**Each stage ends before the next starts, and only the stages' stop rules can end the line early.**

1. **Stage 0, no GPU.** Code, unit tests on synthetic arrays, and the Opus code review. A second
   Opus review follows the fixes, before any fit that produces a number the page quotes.
2. **Stage 1, the synthetic world.** `SG-0`, `SG-A`, and `SG-A-S`, two seeds each. A failure stops
   the line under verdict row 2.
3. **Stage 2, the quick look.** `W2`, fold 2026-05, seed 0: `P0`, `S`, `SG-0`, and `SG-A`, and
   `SG-0` with seed 1 on the same fold, which sets the MDE of G1 before any contrast is read. The
   quick look reads the wiring, the stopping passes, and the learned gains, and prints every
   contrast labelled interim. Its only stop is harm: if `SG-A`'s signed CRPS over all rows of
   either hidden Primary exceeds the same Primary's `P0` by more than 0.20 points, the session
   stops and asks the manager. A one-fold null cannot be read, and no fold is cut from a counted
   reading.
4. **Stage 3, the rest of `W2`, then `W0`.** All four folds, seed 0.
5. **Stage 4, the second seed.** `SG-A` in both worlds on all four folds. The second seed is a
   replication that carries no bar of its own.
6. **Stage 5, seeds 2 to 4, only after a futility test.** The futility test fires when the point
   estimate of G1 (`SG-A` minus `SG-0`, pooled over four folds and two seeds) is at or above zero,
   because a null reading has no route to a gain that three more seeds could find. The test is
   written before any treatment score is read. When the test fires, or when the two-seed verdict is
   `supported` or `not supported`, the page is written at two seeds. Seeds 2 to 4 run only when the
   test does not fire and the two-seed verdict is `inconclusive`, and they need the manager's
   approval because they lie outside the projection.

## Leakage and origin-time weather

**The target row reads only the ensemble forecast issued at or before the forecast origin, and a
test asserts the rule row by row.** Experiment 404's lag-free heads used weather issued after the
origin, which voided every comparison above 24 h of lead. The brief therefore prints one example
origin, its target valid time, and the issue time of the ensemble run it reads.

**History rows read recorded weather (CAMS and ERA5), which arrives about a day late.** The use of
recorded weather is a lab shortcut, as in experiments 713 and 716, and every number on the page is a
ceiling for a live forecast. Experiment 424 lost 7.2 points of signed CRPS on reverse-flow rows when
ensemble weather replaced recorded weather. The scale of `p` is fixed per fold from a source outside
the scored rows, and the write-up repeats the contrast with the injected capacity perturbed by D12's
interval, because the injected capacity is an oracle input. The as-of capacity of the eight MVA-only
Primaries carries F7's stated look-ahead in seven of eight folds; it affects G4 only.

## Decisions

**Decisions the manager has ruled on are marked [Decided by the manager], and open decisions are
marked [Maintainer] with one recommendation each.**

- **A. Rung 1 depends on F7's `HT` code, not on F7's verdict, and runs ahead of F9 [Decided by the
  manager].**
- **B. The training budget is checked against finding C1 and stated above [Decided by the
  manager].** The synthetic budget is 400 passes with patience 30.
- **C. `S` in `W0` reuses F7's stored forecasts, and `SG-E` is reconsidered if rung 0b reads
  `partly supported` [Decided by the manager].**
- **D. Seeds follow the staging rule: a quick look first, and seeds 2 to 4 only when the futility
  test does not fire [Decided by the manager].**
- **E. The cap is about 14 GPU-process-hours [Decided by the manager].** A rung 1 that needs more
  asks first.

| No. | Open decision | Recommendation |
|---|---|---|
| 1 | [Maintainer] Whether rung 1 is the 22nd counted sign experiment, and whether a recorded approval is needed because the design note says rung 1 needs its own, while the standing approval names F2 to F9 | Count it as the 22nd, and record an approval that names the design note, because a `supported` verdict opens rung 2 and a signed forecast at D12 |
| 2 | [Maintainer] The real-fold budget: experiment 713's recipe (patience 8) with the "stopped while falling" flag, or patience 30 | Keep 713's recipe so that `P0` and `S` stay comparable with `M*` and F7, and report the flag |
| 3 | [Maintainer] The size of the sign margin of the synthetic world (0.1 in sign accuracy, borrowed from rung 0b) and of the harm stop of the quick look (0.20, borrowed from G3) | Keep both, because each is an existing bar set before any data |
| 4 | [Maintainer] Whether `SG-A` leaves rung 1 when the farm injection at the visible Primaries is not built in the first code review | Build the injection, because `SG-A` is the only treatment while `SG-E` is out |
| 5 | [Maintainer] Whether `SG-H` (addendum 1's hindsight posterior) joins rung 1 | No, until rung 0e's first known-answer score has been read |
| 6 | [Maintainer] Whether `SG-E` returns if rung 0b reads `partly supported` | Keep `SG-E` out, because the synthetic worlds say the anchored read-out loses and the 4.0 GPU-process-hours would put the total over the cap |

**The contract decisions stay with a human in the production repository.** They are the unit label
of a signed forecast at an MVA series, the sign convention, the clamp at zero for the eight, the
scoring of a signed forecast against an unsigned observation, and the effective capacity at D12.
This experiment changes no contract and makes no request to NGED.

## Reviews and pre-run gates

- **Gate 1:** a fresh Opus agent that did not write the code reviews it before any first fit, and a
  second fresh Opus agent reviews the fixed code before the `W2` fits. The reviews check leakage,
  the fold point of the loss at zero megawatts, the history-power clip in `v` units, the origin-time
  weather, the injection (no two worlds of one Primary on shared dates in any figure), the
  holdout, and a match with the recipe copied from experiments 713 and 716.
- **Gate 2:** "fits starting" to the manager before each family, and a reply.
- **Gate 3:** decisions 1 to 6 are ruled, or the manager rules that the defaults stand.
- **Gate 4:** the first synthetic fit has measured its time per pass and its memory.

**The experiment follows these standing practices and advice:** reuse stored predictions as controls
and references before fitting (`S` in `W0`), save every fitted model outside git, screen before the
full reading (the quick look), report the mean absolute error of the median forecast beside every
CRPS, and report the tails. The experiment departs from refitting monthly only in that every arm of
rung 1 is fitted once per fold. The write-up carries the "Checks" list: the median minus the mean
forecast per site, the error of the median beside every quantile-score contrast, and which audit
triggers fired.

## Method

### Architecture

- **The network is the axial-attention demand network of experiment 713,** with the 14-row by
  48-column token grid of history rows and the target row, the three-component Gaussian mixture
  head, and the read-out of a weighted mean of history power added to the mixture means.
- **The gate-open Primaries read F7's `HT` inputs:** `p` in each history token, `p_0` in the target
  row, and per-Primary gains `a` and `c` that start at zero. The `SG` arms zero the shortwave
  column, the B1 solar-gain columns, and the solar-geometry columns of the tokens that feed the
  head, so the sky enters only through `p`, `p_0`, and `ρ`. Calendar and the time-of-day embedding
  stay.
- **The loss at a gate-open Primary is the folded likelihood,** a log-sum-exp over 3 components and
  2 signs, with the censored branch below `Q`. It folds about zero megawatts and not about zero in
  standardised units. The mixture means are `G_out + head - c_i p_0`, and the 11 folded quantiles
  come from the bisection of experiment 397. Each folded quantile goes through `sqrt(q² + Q²)`
  before the scorer.
- **`SG-A` adds the learned sign head** of the section "Worlds, arms, and folds", and `SG-0` fixes
  `ρ` at 0. `P0` and `S` use the signed loss and no gate. `S` adds `HT`'s inputs to `P0`.
- **No forecast uses information after its origin:** the target-row weather is the ensemble run
  issued at or before the origin, and the check above asserts it.

### Training regime

- **Training data:** the 14 Primaries, with the hidden pair folded in stage 1 too (stage 1 covers
  2019-09 onwards, and the farm frame of experiment 39 runs from 2019-09-17 to 2026-09-10, with 87%
  to 96% of half-hours present per whole year). A half-hour is dropped when any farm has a gap, and
  the write-up reports the dropped share per fold. Cleaned data are used everywhere, and the noisy
  start of the short-history Primaries D15 and D18 before 2025-03-01 is cut.
- **Injection:** the injected reading is built from a within-half-hour interpolation of the signed
  power (6 sub-steps, averaging `sqrt(P'² + Q²)`), so crossings read as D12's do. The injected
  capacity in `p` is the oracle capacity, as experiment 720's code does, with 0.7 and 1.3 times
  that capacity as a sensitivity. The injection code is experiment 720's, imported and not edited.
- **Fits:** each arm fits from scratch on each world and fold, in three stages, with experiment
  713's optimiser, schedule, and real-fold budget (patience 8, with a rerun at the larger cap). The
  synthetic fits run stage 1 only, at up to 400 passes with patience 30. Seeds are 0 and 1 (and 2 to
  4 only after the futility test). Folds are 2025-12, 2026-04, 2026-05, and 2026-06, refitted
  monthly.
- **GPU hours:** about 10.6 GPU-process-hours for the first wave, with a cap of about 14.
- **Saved outputs:** every fitted network's state file under the experiment's data folder, never in
  git, and per-row forecasts through `experiment_store.write_predictions`.

### Measurement

- **Scored rows:** the hidden pair's rows on experiment 713's row set, in the four folds, for G1 to
  G3, and the eight MVA-only Primaries' rows for G4. Leads run from 24 h up to but not including
  240 h after the 12:00 UTC origin. The scored valid times stop before 2026-07-01.
- **Truth:** the signed truth of each hidden Primary (the real signed reading, less the injected
  farm's output in `W2`) for the signed CRPS, and the folded reading for the folded CRPS.
- **Metrics:** the signed CRPS and the mean absolute error of the median forecast, the sign
  accuracy and area of `ρ` on the hidden pair's history tokens with a 2-point dead band, the q01
  coverage on reverse-flow rows (0.062 in experiment 420 against a nominal 0.01), D10's real
  reverse-flow rows in `W0` alone, D12's crossing-day minimum beside the injected one, and the score
  of experiment 420's stored arm A forecasts on the rows they share, stating the intersection.
- **Baselines and intervals:** `P0`, `S`, `SG-0`, and the placebos; the month-by-half-hour
  climatology (repeated once per seed) for the folded scores. Intervals are the day-block
  bootstrap of the decision rule, at 98.33%.
- **Plots:** attention maps of `SG-A` on a sunny and a dull history window; day-ahead and 14-day
  forecasts against what happened, shown as the net folded reading only; the sign posterior `ρ`
  against the true sign, by pass and by month; learning curves with the stopping pass of each fit;
  and the synthetic world's sign accuracy under the real and the shifted sky.
- **Anonymisation:** no figure or table shows any two of the host's reading, the injected reading,
  and the signed truth for the same dates, because the difference is the farm's output. A figure
  of an injected Primary shows the net reading only. No farm series, no generator state, and no
  inferred unit count appears, and D8 appears only inside pooled counts of the eight.

## Files

The code is not written. The page lists each file as written when the results are published.

- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_core.py`: the
  named constants, the bars, and the verdict rows.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_worlds.py`: the
  synthetic world and the builders of `W2` and `W0`, using experiment 720's injection.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_arms.py`: the
  arms, the sign head, and the gate.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_fit.py`: the
  fit driver, with the stage order, the budgets, and the stopping-pass flags.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_score.py`: the
  scoring of G1 to G4 and the read-outs.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/e00721_report.py`:
  the printed report, the verdict file, and the figures.
- `ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/run.sh`: the launch
  wrapper with the slot rules.
- `ml-experiments/00713-msa-axial-attention-demand-network-part-1/`: the network, the training
  loop, and the scorer of experiment 713 (unmerged), imported and not edited.
- `ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/` and the folder of experiment
  717: F7's PV estimate and `HT` inputs (unmerged), imported and not edited.
- `ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/`: the injection and the
  folding code of rung 0 (unmerged), imported and not edited.
- `ml-experiments/00039-pv-disaggregation-cams-injection/` and the farm helpers of experiment 405:
  the farm frame, read through experiment 412's input builder.
- `data/ml-experiments/00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/`: the aggregate
  outputs the page will quote, the network states, and the predictions. Dated detail goes in a
  `PRIVATE/` subfolder.
- `data/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/work/fits/`: F7's stored
  forecasts, read by the `S` reference of `W0` (read only).
- `data/ml-experiments/briefs/msa_architecture/mva_to_mw_design.md` and its two addenda: the
  design this plan follows (read only, never published).
