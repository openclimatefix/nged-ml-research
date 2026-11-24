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
  Complex. (1) The experiment stores 59 to 67 fits (network weights and per-row forecasts, all
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
  ungated reference scored the same as its arm A on those rows (6.99 against 6.98). `SG-A` minus
  `SG-0` on those rows lies between -0.80 and +0.40 points of capacity, so G1 (a gain of at least
  0.404 points) is the likeliest miss, and G3 (no invented reverse flow at D5, with the gate forced
  open) is the second likeliest, because three earlier fits invented reverse flow at D5. `SG-A`
  passes the synthetic world on sign accuracy and on the false-call rate, and its sign accuracy is
  higher under the real sky than under the shifted sky. The probability that `SG-A` fails one of
  the synthetic world's two bars is near one in three. The predicted verdict is `not supported` or
  `inconclusive`. The prediction fails if `SG-A` passes G1, G2, and G3, or if either of the two
  ranges above misses.
decision_rule: >-
  Counted readings G1, G2, and G3. Each is read at 98.33% (1 - 0.05 / 3) from a block bootstrap
  with blocks of 10 consecutive origin days, paired on valid time, with
  `studies.bootstrap.bootstrap_difference_at_level`, and each reads seeds 0 and 1 together, a choice
  fixed here before any fit. G1: signed CRPS on the reverse-flow rows of `W2` (rows whose signed
  truth is below zero, injected or real), `SG-A` minus `SG-0`, has a point estimate at or below
  -MDE and an upper interval end below zero. MDE is the minimum detectable effect, the larger of
  0.404 points of effective capacity (experiment 420's rotation r1 figure) and the absolute
  difference between the two seeds of `SG-0` on fold 2026-05, taken as the signed CRPS on the
  reverse-flow rows of the hidden pair pooled over both Primaries; the MDE is written to a file
  that refuses to be overwritten before any contrast is computed. G2: `SG-A` minus its shifted-sky
  placebo on the same rows has the same bar. G3, no harm in `W0`: signed CRPS over all rows of the
  hidden pair, `SG-A` minus `P0`, has an upper interval end at or below +0.05 points pooled over the
  two Primaries and at or below +0.20 at each Primary, read on the plain `W0` fits (seeds 0 and 1)
  and on the `SG-A-open` fits (the export gate forced open, seed 0, paired with `P0` seed 0). All
  four G3 limits bind the upper end, because harm must be excluded. A synthetic world precedes the
  real worlds. A treatment passes it when, in both of its seeds, its sign accuracy outside a 2-point
  dead band is at least 0.95 at the two exporting synthetic Primaries, and, with the export gate
  forced open, it calls reverse flow on under 1% of sunny midday half-hours at the two other
  Primaries. Its sign accuracy under the real sky and under the shifted sky is reported, and no
  verdict row binds on the gap between them until the maintainer sets the margin (decision 7).
  A fit is flagged `stopped while falling` by the rule of the section on the training budget, and a
  fit that stops before pass 10 is flagged `stopped early`, which counts as flagged. Verdict: the
  first row that matches, and every combination of readings reaches exactly one row. (1) `invalid`:
  a correctness check fails (the holdout guard, the fold point of the loss, the origin-time weather
  assertion, the no-two-worlds figure guard, or a known-answer test on synthetic arrays). (2) The
  synthetic world fails: the line stops, and the verdict is `inconclusive` when any fit of the
  failing treatment hit the 400-pass cap, and `not supported` when every fit stopped by patience.
  (3) G3 fails on the plain fits or on `SG-A-open`: `not supported`, and the line stops under
  experiment 420's registered rule of no further gates or capacity priors. (4) `supported`: G1, G2,
  and G3 pass. (5) `inconclusive`: G1 or G2 misses, and at least one fit of either arm of the
  missed contrast in `W2` is flagged, with the counts reported. (6) `not supported`: G1 misses,
  and no fit of either arm of the G1 contrast is flagged; the output-only result of `SG-0` is
  reported. (7) `inconclusive`: every remaining combination, which is G1 passing with G2 missing
  and no fit flagged. The quick look's harm stop gives no verdict: it pauses the line and asks the
  manager (Stage 2). Treatment minus `P0` on the reverse-flow rows, `S`, `FL`, G4 (the folded CRPS
  at the eight MVA-only Primaries from the same fits), and every share of the gap from `P0` to `S`
  are read-outs with no bar, and no verdict row rests on them. The futility test decides only
  whether seeds 2 to 4 run. Their result is a pre-registered secondary read, reported beside the
  verdict, and it cannot change the verdict from `inconclusive` to `supported` or change any other
  verdict, so no decision is read twice. A rerun is the same attempt only when a check that reads no
  bar statistic fails before the fix and passes after the fix, the fix changes no constant, an Opus
  review confirms the fix, and both outputs are kept. `inconclusive` and `not supported` are never
  rerun. Named constants (each a `Final` of `e00721_core.py`, set before any real data are read):
  0.404, 0.05, 0.20, 0.95, 1%, 2 points, 28 days, 0.3 (the gate's `u_min`), 400 and 30 (synthetic
  passes and patience), 98.33%, 2.0 (injection level), 10 days (the block length), 10 passes (the
  flag window), and 0.20 (the harm limit at the quick look). The sign margin of the synthetic world
  is not yet a constant (decision 7).
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
  arm `M1`. The page uses `P0`, `S`, `SG-0`, `SG-A`, and the other arm names as the arm table below
  defines them.
- **F4, F7, and F9:** the follow-up experiments on the MSA network that share its GPU queue. F4 adds
  neighbouring substations' rows to the network (experiment 715). F7 adds an estimate of the
  unmetered solar power in each history row (experiments 716 and 717). F9 adds similar days from the
  past (experiment 718 and its second part).
- **`HT`:** F7's arm that gives the network the PV estimate `p` in the history rows and the
  target-row PV forecast `p_0`.
- **The Candidate:** the class 2 Candidate demand network of
  [experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md),
  whose weather columns and B1 columns (14 trailing-temperature and solar-gain columns) the MSA
  network reuses.
- **MDE:** the minimum detectable effect, the smallest gain that a reading can tell from noise.
- **q01 coverage:** the share of true values that fall below the forecast's 1% quantile. A reliable
  forecast scores 0.01.
- **The six signed-MW Primaries:** D5, D9, D10, D11, D14, and D16.
- **The eight MVA-only Primaries:** D2, D3, D4, D8, D12, D19, D21, and D22.
- **The hidden pair:** D5 and D10, the two signed-MW Primaries whose sign is hidden in rung 1
  (rotation r1 of experiment 420). D10 holds real reverse flow on 0.76% of its half-hours, and D5 is
  the Primary at which three earlier fits invented reverse flow. The other four signed-MW Primaries
  (D9, D11, D14, and D16) are the visible Primaries.
- **Folded reading:** the magnitude `sqrt(X² + Q²)` of the signed power `X`, with a reactive floor
  `Q` of 2 points. A point is 1% of the Primary's capacity. "Hidden" means that the network sees
  only the folded reading.
- **Reverse-flow rows:** the scored half-hours of the hidden pair on which the signed truth is
  below zero, injected or real, as in experiment 420.
- **Injection level 2.0:** a real solar farm's recorded output, added at twice the host's median
  absolute reading on sunny summer middays before 2025-07-01. The farm's series is a private input
  and appears on no page.
- **Worlds `W2` and `W0`:** `W2` folds the hidden pair after the injection. `W0` folds the same
  pair with no injection, and is the placebo that must show no invented reverse flow. Both worlds
  carry the same injection of the two other farms' output at the four visible Primaries, in the
  readings of every arm.
- **Export gate:** the per-Primary switch of experiment 420, set once per refit from the readings,
  that allows a Primary to export at all.
- **Sign posterior `ρ`:** the probability that a history token exported, from the gross-demand
  read-out or from the learned head.
- **Signed CRPS:** the continuous ranked probability score of the 11 signed quantiles against the
  signed truth, in points of capacity. A score below another score is better.

## Where this experiment sits in the queue

**Rung 1 depends on F7's `HT` code, which is reviewed and running, and not on the five-seed verdict
of F7.** The `HT` code supplies the PV estimate `p` in the history rows, the target-row forecast
`p_0`, the as-of capacity table, and the 28-day shifted-sky placebo. The code of experiments 716 and
717 is imported from their folders on `PYTHONPATH` and not copied. **[Decided by the manager]**

**Rung 1 sits ahead of F9 in the GPU queue: F4, then F7, then rung 1, then F9.** The earlier design
note placed rung 1 behind F4 and F9 and after an F7 verdict. The Opus review of the queue found that
F9 answers no question directly and has the weakest prior, and that rung 1 is the direct test of
the always-megawatt forecast. **[Decided by the manager]**

**With decisions 1 to 6 ruled and the first Opus code review passed, this plan is ready to start its
synthetic world.** No step before the synthetic world waits for rung 0b, for F7's five-seed
verdict, or for F9. The next action is the code, which a Sonnet agent writes from this page,
followed by the Opus review. Nothing on this page has been run.

| Prerequisite | State on 2026-11-24 | Blocks the synthetic world? |
|---|---|---|
| F7's `HT` code (experiments 716 and 717) | Reviewed; part 1's fits are running | No: the code is imported as it stands |
| F7's `HT` forecasts on the real folds | Not stored: the stored forecasts are arms `H` and `Z`, which cannot stand in for `S` | No: `S` in `W0` is fitted from scratch on four folds |
| Experiment 720's injection code | Written and tested on synthetic worlds; unmerged | No: imported on `PYTHONPATH` |
| Rung 0b's real run | Not run; waits for the maintainer's choice between the two options of the rung 0b note | No: only the `SG-E` block waits for it |
| Experiment 713's known-answer gate | Waived by the maintainer; every verdict carries the waiver sentence | No |

## What this plan changes from the design note

**The design note (`mva_to_mw_design.md`, section 5.2) fixes the arms, the readings G1 to G4, and 71
fits at about 14 GPU-process-hours.** Four orders from the manager and the Opus review of this plan,
applied before any data, change the plan. Each change is listed with its reason.

| Design note | This plan | Reason |
|---|---|---|
| Synthetic fits: not stated, with the harness of experiment 713's gate in mind | 400 passes, patience 30, stage 1 only | Finding C1 of the F4 audit (next section) |
| `S` fitted in `W0` on 4 folds | Unchanged: `S` is fitted from scratch on all four folds of `W0`. F7's `HT` forecasts, if stored by then, are a descriptive comparison only | No `HT` forecast is stored today, and F7's recipe differs from rung 1's (see "The `S` arm of `W0`") |
| `SG-E`, `SG-E-S`, and `SG-E-open` in the first wave (about 16 fits and 4 synthetic fits) | Out of the first wave; reconsidered only if the anchored read-out passes both of its pre-registered bars in experiment 720 | Rung 0b's synthetic worlds show the anchored read-out losing to a reading-blind reference |
| Second seed of the winner only; no rule for further seeds | Seeds 0 and 1 are read together for G1 to G3. A quick look comes first. Seeds 2 to 4 run only when the futility test does not fire, outside the projection, and give a secondary read | The Opus review of the queue: month-to-month variation, not seed noise, sets the interval width; the two-seed reading is the only counted one, so no decision is read twice |
| `SG-A`'s farm injection at the visible Primaries stated as a condition | Built in the first code review, in the readings of every arm in both worlds, or `SG-A` leaves rung 1 and the plan stops before `W2` | With `SG-E` out, `SG-A` is the only treatment, and an injection that only `SG-A` saw would not isolate the sign head |
| Injected capacity in `p` estimated by experiment 662's probe on the folded readings | The oracle capacity (the farm's true capacity), as experiment 720's code does, with 0.7 and 1.3 times that capacity as the one sensitivity rule | Experiment 720's injection code takes the true capacity. The oracle capacity is a look-ahead input, so every score is a ceiling for that reason as well as for the recorded weather |
| Day-block bootstrap (not specified) | Blocks of 10 consecutive origin days | Origins up to 10 days apart share weather and error, so one-day blocks would understate the variance |

## The training budget, checked against the F4 audit's finding C1

**The synthetic worlds train at up to 400 passes with a patience of 30, and not at the 150 passes
and patience 8 that failed the F4 gate.** The F4 audit's finding C1 is that the known-answer
harness of experiment 715 used experiment 713's budget from before its amendment 3. The W3 fits then
stopped at passes 42 to 54 while the validation loss was still falling by about 0.05 every 10
passes. On experiment 713's synthetic world W0, 713's arm `M` needed 350 passes, and under 715's cap
the same pair of fits stopped at pass 143 and at the cap of 150. Experiment 713's amendment 3 raised
the synthetic budget to 400 and 30 for this reason, and experiment 716's arm `M` of its
known-answer worlds uses the same pair.

| Fits | Passes | Patience | Why |
|---|---|---|---|
| Synthetic world of rung 1 (stage 1 only) | At most 400 | 30 | A hard stop gate must not fail for a harness reason, as the F4 gate did |
| `W2` and `W0` real folds | Experiment 713's real-fold budget, with its rerun at the larger cap | 8 | Experiment 713's budget is the recipe the manager ruled on (decision 2); the fit prints its stopping pass |

**The real folds keep experiment 713's budget, and a flag marks every stage that may have stopped
early.** Experiment 716's known-answer gate (F7's KF9, in synthetic worlds) shows why the flag is
needed. Its arm `H` trained 60 passes with no early stopping, and patience 8, applied afterwards,
picked passes 4, 23, and 5 of the three seeds, where the best of the 60 passes were 43, 55, and 32.
The audit of that gate found the PV history path still learning at pass 60. Every `SG` arm uses the
same path, so G1 is read under a budget that is known to under-train the history path. No counted
reading compares a rung 1 arm with `M*` or with F7, so the budget does not need to match theirs for
G1 to G3. The flag is the only guard, and it is therefore a formula.

**The flag is defined per stage, with these constants (`e00721_core.py`), and a known-answer test on
a synthetic loss curve checks it.**

- **Stages tested.** Each real fit has three stages, and each stage has its own stopping pass. The
  flag is computed for all three stages and printed per stage. A fit is flagged when any one of its
  three stages is flagged. The write-up states, once the code has been read, in which stage the sign
  head and the `HT` gains `a` and `c` train. Until then every stage counts.
- **The window.** The stopping pass is the last pass the stage ran, whether patience or the pass cap
  ended it. The window is the 10 passes that end at the stopping pass.
- **The fall and the noise.** The fall is the mean validation loss over the first 5 passes of the
  window minus the mean over the last 5. The noise is the sample standard deviation of the 10
  validation losses in the window. Neither the F4 audit nor the F7 audit states a noise measure, so
  the page uses this standard deviation.
- **The flag.** A stage is flagged `stopped while falling` when the fall exceeds the noise. A stage
  that stopped before pass 10 has no window and is flagged `stopped early`. Both flags count as
  flagged.
- **Effect.** The flag changes no number. Verdict rows 5, 6, and 7 use it.

Every fit's stopping pass per stage, its flags, and the learned gain of each new path as a share of
its reachable range are printed and appear in the write-up.

**The cost of the larger synthetic budget is inside the design's figure of 0.2 GPU-process-hours
per fit.** Experiment 713's amended gate, at 400 passes and patience 30, took 4,027 s for 8 stage-1
fits plus its checks, about 0.14 GPU-process-hours per fit (run log of experiment 713). The earlier
1,985 s of its second rerun used 150 passes and patience 8, so the page does not cite it. The figure
of 0.2 covers a rerun at the pass cap. If the first synthetic fit measures a different time, the
manager is told before the second fit starts.

## Worlds, arms, and folds

**Three worlds are fitted, and each treatment is tested in the order synthetic, `W2`, `W0`.**

- **Synthetic world (stop gate).** Gross demand is a fixed function of calendar and temperature,
  `X = D - C u`, and the reading is `sqrt(X² + Q²)`. Two synthetic Primaries export on clear
  middays and two never do. The exporting pair holds the real export share of D12's history
  windows, 70% to 93% of days from March to September, because a world with balanced windows
  cannot show the failure of a read-out that starts from "every row imported". The stop
  gate reads both seeds of each treatment, and a treatment passes only when both seeds pass. The
  non-exporter check forces the export gate open at the two non-exporting Primaries, as rung 0b's
  source check does at D5, because a shut gate would pass the check by construction. `SG-0` is
  fitted in the synthetic world as a reference for the signed CRPS on the exporters' reverse-flow
  rows. It has no sign accuracy, because its `ρ` is fixed at 0, and it carries no bar.
- **`W2` (injected).** Folds 2026-04, 2026-05, and 2026-06 are sunny, and fold 2025-12 is dark and
  tests for harm. Every arm of `W2` reads the same readings, which include the two other farms'
  recorded output injected into the four visible Primaries. Only the loss of `SG-A` reads the true
  sign of the visible Primaries. The injection is in the readings of every arm and is not added for
  `SG-A` alone, so the contrast of G1 isolates the sign head.
- **`W0` (no injection at the hidden pair).** The same four folds, with the hidden pair folded and
  nothing added to it. The four visible Primaries carry the same injection as in `W2`, so that
  `SG-A` has positive labels in `W0` as well and G3 tests the same treatment as G1.

**The arms are listed with their role, and every arm of the first wave fits from scratch.** The
fits load no stage-1 state and no frozen encoder of the Candidate, because both were trained on the
hidden pair's signed readings.

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

### The `SG-E` family is held back, and the anchored read-out's two bars decide whether it returns

**Experiment 720's rung 0b has not been run on real data, and its synthetic worlds say the anchored
read-out loses.** In 10 synthetic exporter worlds, including worlds with the PV estimate off by 25%
a day and demand with a 5-point day offset, the anchored read-out lost to the reading-blind
reference by 0.07 to 0.12 in area under the curve, with every upper interval end at or below +0.017.
The hindsight read-out beat the reading-blind reference in all 10 worlds. The `SG-E` family repeats
the anchored read-out inside the network and would cost about 20 fits (4.0 GPU-process-hours).

**This plan reconsiders `SG-E` only when the anchored read-out passes both of its pre-registered
bars in experiment 720, and its recommendation on the evidence available now is to keep `SG-E`
out.** Experiment 720 pre-registers two bars for the anchored read-out in rung 0b: a stop rule (an
area under the curve with the real sky at least 0.1 above the shifted sky, with under 1% false
reverse flow at D5) and a references bar (the anchored read-out beats the reading-blind and the
hour-of-day references, with the lower end of the paired interval above 0 on the pooled area). The
rule keys to these two bars and not to experiment 720's verdict word, because experiment 720's
verdict also depends on rungs 0a, 0c, and 0d. The rule is written before any data:

- **Both bars pass:** the `SG-E` family is added as a second block, and the cost is stated to the
  manager first, because the total then exceeds the cap (next section). Experiment 720 reads
  `supported` in this case when its other rungs also pass. A `partly supported` verdict never meets
  this condition, because that verdict requires the anchored read-out to fail a bar.
- **Either bar fails, whatever word experiment 720's verdict carries:** `SG-E` stays out, G3's
  forced-open check stays on `SG-A-open`, and the page says why. Experiment 720 reads `partly
  supported` when the anchored read-out fails and the unanchored control passes, or when the
  anchored read-out passes its stop rule and does not beat a reference. Neither case reopens `SG-E`,
  and the verdicts `not supported`, `inconclusive`, and `invalid` also keep `SG-E` out. Whether
  `SG-E` returns on any other ground is decision 6.
- **Rung 0b has not been read when rung 1's `W0` finishes:** `SG-E` stays out, and the page says
  that its absence is untested.

**Any new read-out, such as a hindsight posterior (`SG-H` of addendum 1) or a prior built from the
PV estimate, needs an amendment before data and its own review.** Neither `SG-H` nor any other new
read-out is in this plan, and no read-out that was never pre-registered reopens `SG-E`. Addendum 1
recommends no hindsight posterior before rung 0e's first known-answer score has been read.

### The `S` arm of `W0` is fitted from scratch on all four folds

**In `W0`, `S` reads the real signed readings and is fitted from scratch on each of the four folds,
because no `HT` forecast is stored today.** On 2026-11-24 the stored forecasts of F7 are arms `H`
and `Z`, not `HT`, on folds 2025-11, 2026-01, 2026-05, and 2026-06, seed 0. Arms `H` and `Z` cannot
stand in for `S`. The stored folder changes as F7 runs, so the page keeps this snapshot's date and
the scoring script re-reads the folder when it scores.

**F7's stored `HT` forecasts, if F7's phase 5 has stored them when `W0` is scored, are a descriptive
comparison and replace no fit.** Their recipe differs from rung 1's `S` in four ways. F7 loads the
stored stage 1 of `M*` for the same fold and seed, freezes the encoders, and fits stages 2 and 3
only, while rung 1's `S` fits all three stages from scratch. Rung 1 drops every half-hour at which
any farm has a gap, and F7 drops none. The seeds of stage 1 and the code hashes differ. The row keys
match (`backtest_2025_11_to_2026_06_daily_12utc_before_holdout`), and the comparison states the
intersection of the scored keys after rung 1's farm-gap drop. A share of the gap from `P0` to `S`
in `W0` therefore compares two from-scratch arms, as the same share in `W2` does.

## Counted readings, verdict rows, and the stop rule

**The three counted readings are G1, G2, and G3, and G4 is a report.** The decision rule in the
front matter states each bar. Each counted reading is read at 98.33% from a block bootstrap with
blocks of 10 consecutive origin days, paired on valid time, and reads seeds 0 and 1 together. In
words:

- **G1, the gain from the history sign.** `SG-A` minus `SG-0`, signed CRPS on the reverse-flow rows
  of `W2`. The bar needs a point estimate at or below -MDE and an upper interval end below zero. The
  MDE is the larger of 0.404 points and the difference between the two seeds of `SG-0` on fold
  2026-05, taken as the signed CRPS on the reverse-flow rows of the hidden pair, pooled over both
  Primaries. The MDE file is written, and refuses to be overwritten, before any contrast is
  computed. Experiment 420 derived 0.404 as 2.8 bootstrap standard deviations (0.144 each) over 14
  Primary-month blocks, at a 95% two-sided level with 80% power. The same recipe at 98.33% gives
  about 3.23 standard deviations, or about 0.47. The page keeps 0.404 as a fixed constant, so G1 is
  read at a floor lower than the power its own level calls for.
- **Treatment minus `P0`, a read-out.** The signed CRPS of each arm minus that of `P0` on the
  reverse-flow rows is reported with its interval and carries no bar, because experiment 420 found
  the ungated reference scored the same as its arm A on those rows (6.99 against 6.98), so any arm
  that can go below zero beats `P0`. No verdict row rests on it.
- **G2, the sky carries the gain.** `SG-A` minus `SG-A-S` on the same rows, with the same bar.
- **G3, no invented reverse flow.** In `W0`, `SG-A` minus `P0` over all rows of the hidden pair, on
  the plain fits (seeds 0 and 1) and on `SG-A-open` (seed 0, paired with `P0` seed 0). The upper end
  of the 98.33% interval must be at or below +0.05 points pooled over the two Primaries, and at or
  below +0.20 at each Primary. The design note says "within +0.05 points of `P0` pooled" and
  "within +0.20 at either Primary", and does not say whether the limit binds the point estimate or
  the upper interval end. This page binds the upper end, because harm must be excluded and not only
  unseen. The write-up reports each gate decision per world and fold, because the gate is expected
  to stay shut at the un-injected pair, and a shut gate passes G3 by construction.
- **G4, the eight MVA-only Primaries.** The folded CRPS of each arm at the eight, from the same
  fits, with its interval and no bar. G4 is read at seed 0, and the seed-1 fits are reported beside
  it where they exist. The design note's two-seed detectable difference of 0.25 points therefore
  does not apply to G4, and this page states no one-seed figure.

**The verdict table is strictly first-match.** Row 5 sits ahead of rows 6 and 7 because a fit that
stopped while its validation loss was still falling must make a G1 or G2 miss `inconclusive`, and
not `not supported`. A fit that is flagged cannot show that its treatment has no gain, because the
budget may have hidden the gain. Rows 6 and 7 read only unflagged contrasts. A G1 miss in which
only some of the contrast's fits are flagged reaches row 5, because one flagged fit is enough to
leave the miss unexplained.

**The smallest gain worth porting is not set here, because this experiment ports nothing.** A
`supported` verdict allows rung 2 to be planned. Rung 2 sets its own threshold against the cost of
adding the sign head to the network.

**The experiment can fail in every row.** A miss of G1 is the likeliest outcome, and the page then
reports the output-only result of `SG-0` as the finding. The verdict reads the treatment that is
`SG-A`, the only treatment in the first wave, on seeds 0 and 1 together. A pre-registered gate
stays pre-registered: a reading chosen after the scores were seen is labelled post hoc and changes
no verdict.

**The seeds are fixed in advance, and no decision is read twice.** Seeds 0 and 1 give the verdict.
The futility test decides only whether seeds 2 to 4 run. Their result is a pre-registered secondary
read, reported beside the verdict. It cannot change an `inconclusive` verdict to `supported`, and it
cannot change any other verdict. A `supported` or a `not supported` verdict at two seeds is final,
and seeds 2 to 4 do not run.

### The count of sign experiments

**The honest count of experiments that open a gate to a sign label puts rung 1 between the 23rd
and the 25th.** The erratum's definition of a test is a statistic with a bar stated in advance, at
least one outcome of which produces or gates a sign label, admits a hint or factor, sets a prior,
or licenses a later design to cite the statistic as evidence for a sign. Two lists of earlier
experiments exist, and they differ.

- **The design note's list** has 10 experiments: 152, 372, 374, 387, 408, 412, 417, 420, 424, and
  427.
- **The later amendment's list** has 18: 79, 152, 358, 362, 365, 366, 370, 387, 408, 412, 417, 420,
  424, 427, 506, 508, 543, and 674. It lacks experiments 372 and 374, which the design note counts.
  It numbers experiment 722 the 19th, experiment 723 the 20th, and experiment 724 the 21st.
- **Experiment 720 meets the definition and is in neither list.** The bars of its rung 0b gate
  `SG-E` here, and its rung 0e writes a sign posterior.
- **The reconciled count before experiment 722** is the 18 of the amendment, plus 372, 374, and
  720, which is 21. With experiment 722 that is 22, so rung 1 is the 23rd if neither experiment 723
  nor experiment 724 runs first, and the 25th if both run first. Experiment 723 is the 20th of the
  amendment's numbering only if it is approved.

**Rung 1 is another experiment that gates a sign label, and the manager ruled "count it as the
22nd, count honestly".** The 22nd is rung 1's place in the amendment's own numbering (18, then
experiments 722, 723, and 724). The page states the honest count, 23rd to 25th, as a range for the
maintainer, and the stop rule's approval stands. A `supported` verdict admits a treatment to rung 2,
and so to a signed forecast at D12, which the stop rule's definition counts. Experiment 420 tested
the same kind of statistic in the same kind of world.

## GPU projection

**The first wave is 59 to 67 fits, about 11.8 to 13.4 GPU-process-hours, inside the cap of about 14
GPU-process-hours that the manager set (decision E).** Rung 1's own recorded approval is decision 1.
The figure of 0.2 GPU-process-hours per fit is the design note's, from experiment 713's measured
solo fits of 0.09 to 0.19 and a margin for predictions, fold builds, and reruns at the pass cap. The
first fit measures the real figure.

| Arm | Synthetic (2 seeds) | `W2` (4 folds, seed 0) | `W0` (4 folds, seed 0) | Seed 1 | Fits | GPU-process-hours |
|---|---|---|---|---|---|---|
| `P0` | 0 | 4 | 4 | 4 (`W0`, for G3) | 12 | 2.4 |
| `S` | 0 | 4 | 4 | 0 | 8 | 1.6 |
| `SG-0` | 2 | 4 | 4 | 3 (`W2`, folds other than 2026-05, for G1) plus 1 (fold 2026-05) | 14 | 2.8 |
| `SG-A` | 2 | 4 | 4 | 8 (both worlds) | 18 | 3.6 |
| `SG-A-S` | 2 | 4 | none | 4 (`W2`, for G2) | 10 | 2.0 |
| `SG-A-open` | none | none | 4 | 0 | 4 | 0.8 |
| `FL` | none | 1 | none | 0 | 1 | 0.2 |
| **First wave, all three readings at two seeds** | **6** | **21** | **20** | **20** | **67** | **13.4** |

**The range of the first wave comes from how many of the three counted readings are read at two
seeds, and the verdict rule reads all three at seeds 0 and 1.** The fits above are the 53 fits of
this page's earlier projection (10.6 GPU-process-hours), plus three groups that projection left out.

| Item | Fits | GPU-process-hours | Running total |
|---|---|---|---|
| The 53 fits of the earlier projection | 53 | 10.6 | 10.6 |
| `S` fitted on all four folds of `W0` (no stored `HT`; the recipe-check fit is one of the four) | 3 | 0.6 | 11.2 |
| `SG-0` seed 1 on the three other `W2` folds (G1 at two seeds) | 3 | 0.6 | 11.8 |
| `SG-A-S` seed 1 on four `W2` folds (G2 at two seeds) and `P0` seed 1 on four `W0` folds (G3 at two seeds) | 8 | 1.6 | 13.4 |

**The planned total is the top of the range, 13.4.** The floor of 11.8 applies only if G2 and G3 were
read at seed 0 alone, which the verdict rule above does not do. The rule would need an amendment
before data to allow it. The evidence goes to the manager before any rerun uses the synthetic
world's sign margin (decision 7).

| Item | Fits | GPU-process-hours | Condition |
|---|---|---|---|
| A rerun of the synthetic world, once, if it fails once (design note section 6.1) | 6 | 1.2 | Total 13.0 to 14.6; a total above 14 asks the manager first |
| `SG-E` family (synthetic 4, `W2` 8, `W0` 8) | 20 | 4.0 | Rung 0b's anchored read-out passes both bars; total 15.8 to 17.4, which needs the manager's approval |
| Seeds 2 to 4 of `SG-0` and `SG-A` on `W2` (24 fits) and of `SG-A-S` (12 fits) | 36 | 7.2 | Outside the projection: the futility test does not fire, and the manager approves |
| The design note's figure for comparison | 71 | about 14 | Fits `SG-E` and `SG-A`, and `S` in `W0` |

**The compute rules of the slot hold for every fit.**

- **Slot and queue.** The GPU slot comes from the manager. One worker fits at a time, because four
  concurrent fits gave no gain. The first wave takes about 13.4 hours of wall clock at most. The
  queue is F4, F7, rung 1, F9. Rung 0b and the CPU work run beside rung 1 and use no GPU.
- **Memory per fit.** The host cap is `MemoryMax=12G`. The references are experiment 713's gate
  fits (GPU peak 5,090 MiB, host peak 6.3 GiB) and experiment 715's gate fits (GPU 10.6 to 11.7
  GB, host peak 7.1 GB). The GPU-memory share is the manager's slot, and the first fit measures its
  own peak. Before the second fit starts, the host cap is reset from the measured peak.
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

1. **Stage 0, no GPU.** Code, unit tests on synthetic arrays (including the known-answer test of the
   flag), and the Opus code review. A second Opus review follows the fixes, before any fit that
   produces a number the page quotes.
2. **Stage 1, the synthetic world.** `SG-0`, `SG-A`, and `SG-A-S`, two seeds each. A failure stops
   the line under verdict row 2.
3. **Stage 2, the quick look.** `W2`, fold 2026-05, seed 0: `P0`, `S`, `SG-0`, and `SG-A`, and
   `SG-0` with seed 1 on the same fold, which sets the MDE of G1 before any contrast is read. The
   MDE file is written first, and it refuses to be overwritten. The quick look reads the wiring,
   the stopping passes, and the learned gains, and prints every contrast labelled interim. Its only
   stop is harm, and the harm stop gives no verdict: if `SG-A`'s signed CRPS over the rows of either
   hidden Primary whose signed truth is at or above zero exceeds the same Primary's `P0` by more
   than 0.20 points, the session pauses the line and asks the manager. The stop reads import rows
   because the reverse-flow rows are where an arm that can go negative gains over `P0`, and an
   all-rows contrast would hide harm. A one-fold point estimate without an interval cannot be read
   as a verdict, and no fold is cut from a counted reading.
4. **Stage 3, the rest of `W2`, then `W0`.** All four folds, seed 0.
5. **Stage 4, the second seed.** Seed 1 of `SG-A` in both worlds, of `SG-0` and `SG-A-S` in `W2`,
   and of `P0` in `W0`, all on four folds. The verdict reads seeds 0 and 1 together from this stage
   on, and the stage's fits carry no bar of their own apart from the pooled readings.
6. **Stage 5, seeds 2 to 4, only after a futility test.** The futility test fires when the point
   estimate of G1 (`SG-A` minus `SG-0`, pooled over four folds and two seeds) is at or above zero,
   because a null reading has no route to a gain that three more seeds could find. The test is
   written before any treatment score is read. When the test fires, or when the two-seed verdict is
   `supported` or `not supported`, the page is written at two seeds. Seeds 2 to 4 run only when the
   test does not fire and the two-seed verdict is `inconclusive`, and they need the manager's
   approval because they lie outside the projection. Their result is the pre-registered secondary
   read, and it cannot change the verdict.

## Leakage and origin-time weather

**The target row reads only the ensemble forecast issued at or before the forecast origin, and a
test asserts the rule row by row.** Experiment 404's lag-free heads used weather issued after the
origin, which voided every comparison above 24 h of lead. The brief therefore prints one example
origin, its target valid time, and the issue time of the ensemble run it reads.

**History rows read recorded weather (CAMS and ERA5), which arrives about a day late.** The use of
recorded weather is a lab shortcut, as in experiments 713 and 716. Every number on the page is a
ceiling for a live forecast. Experiment 424 lost 7.2 points of signed CRPS on reverse-flow rows
when ensemble weather replaced recorded weather. The oracle injected capacity (see the table
"What this plan changes from the design note") is a look-ahead input, so every score is a ceiling
for that reason too. The write-up repeats the contrast with the injected capacity at 0.7 and 1.3
times the oracle value, which is the one sensitivity rule.

**The hidden pair's own PV capacity in `p` is re-estimated from the folded readings, and never read
from the signed readings.** F7's as-of capacity table, for the six signed-MW Primaries, may come
from a probe on their signed readings (experiment 673 read the signed Primaries' capacity that
way). A capacity estimated from the hidden sign would leak the sign into `p`. Rung 1 therefore
takes the capacity of D5 and D10 from [experiment 662](../00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md)'s
probe on the folded readings, with data before the fold's start only, and takes the as-of table
only for the other Primaries. The scale of `p` at every Primary is fixed per fold from these two
sources, both outside the scored rows. The as-of capacity of the eight MVA-only Primaries carries
F7's stated look-ahead in seven of eight folds, and it affects G4 only.

**The holdout guard names the farm frame.** The row set ends before 2026-07-01, and the loader clips
the farm frame, which runs to 2026-09-10, so that no valid time on or after 2026-07-01 is read.

## Decisions

**Decisions the manager has ruled on are marked [Decided by the manager], and open decisions are
marked [Maintainer] with one recommendation each.**

- **A. Rung 1 depends on F7's `HT` code, not on F7's verdict, and runs ahead of F9 [Decided by the
  manager].**
- **B. The training budget is checked against finding C1 and stated above [Decided by the
  manager].** The synthetic budget is 400 passes with patience 30.
- **C. `SG-E` is reconsidered only when the anchored read-out passes both of its pre-registered bars
  in experiment 720 [Decided by the manager].**
- **D. The verdict reads seeds 0 and 1 together, and seeds 2 to 4 follow the futility rule and give
  a secondary read [Decided by the manager].**
- **E. The cap is about 14 GPU-process-hours [Decided by the manager].** A rung 1 that needs more
  asks first.
- **1. Counting and approval [Decided by the manager].** Rung 1 is counted as a sign experiment: the
  ruling is "count it as the 22nd, count honestly". The recorded approval of rung 1 falls under the
  standing exception. The honest count is stated as the range 23rd to 25th (section "The count of
  sign experiments").
- **2. The real-fold budget [Decided by the manager].** Experiment 713's recipe (patience 8) with
  the "stopped while falling" flag. G1 is therefore read under a budget known to under-train the
  history path, and the flag is the only guard.
- **3. The borrowed bars [Decided by the manager].** The harm stop of 0.20 (read on import rows) and
  the level of 98.33% are accepted, because each is pre-registered on this page and covered by the
  Opus plan review. The third borrowed bar, the sign margin of the synthetic world, is decision 7.

**The sign margin of the synthetic world cannot be derived from the sources, so it is decision 7
and no verdict row binds on it.** The derivation was attempted in four steps.

1. Rung 0b's pre-registered margin is 0.1 in area under the receiver operating characteristic curve,
   with the real sky above the shifted sky (experiment 720's decision rule, and row V1 of addendum
   1). It is the only margin in the sources, and its unit is area.
2. Experiment 720's synthetic worlds report area and the false-reverse rate. They report no sign
   accuracy under the real and under the shifted sky, so no gap in accuracy units exists to read.
3. A difference of 0.1 in area does not convert to a difference in accuracy outside the dead band,
   because that accuracy depends on the share of exporting half-hours (70% to 93% of days here) and
   on the threshold.
4. A noise floor for the accuracy gap needs the number of tokens that the synthetic world scores.
   That number is set by the synthetic code, which does not exist yet.

| No. | Open decision | Recommendation |
|---|---|---|
| 4 | [Maintainer] Whether `SG-A` leaves rung 1 when the farm injection at the visible Primaries is not built in the first code review | Build the injection, because `SG-A` is the only treatment while `SG-E` is out |
| 5 | [Maintainer] Whether `SG-H` (addendum 1's hindsight posterior) joins rung 1 | No, until rung 0e's first known-answer score has been read |
| 6 | [Maintainer] Whether `SG-E` returns on any ground other than the two bars of experiment 720 | No, because the synthetic worlds say the anchored read-out loses and the 4.0 GPU-process-hours would put the total over the cap |
| 7 | [Maintainer] The margin by which the sign accuracy outside the 2-point dead band under the real sky must exceed the accuracy under the shifted sky, in the synthetic world, in sign-accuracy units | 0.10, because it is 4.6 times the standard deviation (0.022) of a difference of two independent accuracies of 0.95 on 200 tokens, the smallest cell that experiment 720's rules treat as scoreable. The number is a proposal and not a source's bar. Until the maintainer sets it, the margin is reported and gates nothing. The synthetic accuracies under both skies go to the manager before any rerun uses the margin |

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
- **Gate 3:** decisions 1 to 3 are decided (above). Decisions 4 to 6 are ruled, or the manager rules
  that their defaults stand. Decision 7 stays open, and no verdict row binds on it.
- **Gate 4:** the first synthetic fit has measured its time per pass and its memory, and the memory
  cap is reset from the measured peak before the second fit.

**The experiment follows these standing practices and advice:** save every fitted model outside
git, screen before the full reading (the quick look), report the mean absolute error of the median
forecast beside every CRPS, and report the tails. The write-up carries the "Checks" list: the median
minus the mean forecast per site, the error of the median beside every quantile-score contrast, and
which audit triggers fired.

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

- **Training data:** the 16 Primaries, with the hidden pair folded in stage 1 too (stage 1 covers
  2019-09 onwards, and the farm frame of experiment 39 runs from 2019-09-17 to 2026-09-10, with 87%
  to 96% of half-hours present per whole year). A half-hour is dropped when any farm has a gap, and
  the write-up reports the dropped share per fold. Cleaned data are used everywhere, and the noisy
  start of the short-history Primaries D15 and D18 before 2025-03-01 is cut.
- **Injection:** the injected reading is built from a within-half-hour interpolation of the signed
  power (6 sub-steps, averaging `sqrt(P'² + Q²)`), so crossings read as D12's do. The injected
  capacity in `p` is the oracle capacity, as experiment 720's code does, with 0.7 and 1.3 times
  that capacity as the one sensitivity. The capacity of the hidden pair's own host PV in `p` comes
  from experiment 662's probe on the folded readings (section "Leakage and origin-time weather").
  The injection code is experiment 720's, imported and not edited.
- **Fits:** each arm fits from scratch on each world and fold, in three stages, with experiment
  713's optimiser, schedule, and real-fold budget (patience 8, with a rerun at the larger cap). The
  synthetic fits run stage 1 only, at up to 400 passes with patience 30. Seeds 0 and 1 give the
  verdict (and seeds 2 to 4 run only after the futility test, for a secondary read). Folds are
  2025-12, 2026-04, 2026-05, and 2026-06, refitted monthly.
- **GPU hours:** about 11.8 to 13.4 GPU-process-hours for the first wave (planned: 13.4), with a
  cap of about 14.
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
  climatology (repeated once per seed) for the folded scores. Intervals are the block
  bootstrap of the decision rule (blocks of 10 consecutive origin days), at 98.33%.
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
  forecasts, read only as a descriptive comparison with `S` in `W0` if F7's phase 5 has stored `HT`
  by then.
- `data/ml-experiments/briefs/msa_architecture/mva_to_mw_design.md` and its two addenda: the
  design this plan follows (read only, never published).
