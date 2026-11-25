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
  Complex. (1) The experiment stores 64 fits (network weights and per-row forecasts, all
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
  passes the synthetic world on sign accuracy and on the false-call rate, and its area under the
  curve is higher under the real sky than under the shifted sky. The probability that `SG-A` fails
  one of the synthetic world's bars is near one in three. The predicted verdict is `not supported`
  or `inconclusive`. The prediction fails if `SG-A` passes G1, G2, and G3, or if either of the two
  ranges above misses.
decision_rule: >-
  Counted readings G1, G2, and G3. Each is read at 98.33% (1 - 0.05 / 3) from a two-stage block
  bootstrap, and each reads seeds 0 and 1 together, a choice fixed here before any fit. The blocks
  are 10 consecutive valid dates within a fold, on which the pairing already runs. The resampler is
  written in `e00721_score.py`. Each resample draws one seed, then draws folds with replacement,
  then draws blocks with replacement within each drawn fold, because each fold is a separate fit
  whose blocks share its offset. `studies.bootstrap.bootstrap_difference_at_level` draws flat units
  from one `month` column and cannot do this, so no counted reading uses it, and no file under
  `src/studies/` changes. The point estimate is the mean over the two seeds, and the interval
  describes one seed drawn per resample, so it is conservative for the two-seed estimate. A reading
  with fewer than 20 resampling units, which is every reading here, widens its interval about the
  point estimate by the ratio of the Student t quantile to the normal quantile at 98.33%, with B - 1
  as the degrees of freedom, and by the square root of B / (B - 1), with B the Kish effective number
  of blocks. Each counted reading uses 20,000 resamples and a fixed generator seed (both named
  constants below), and the power check of G3 runs once with them. The folds are not counted again
  as degrees of freedom, because the two-stage resampler already carries the fold-level clustering,
  and the page applies no fold-count correction on top of the two-stage draw. The resampler and the
  widening were fixed before any rung 1 data were read. G1: signed CRPS on the reverse-flow rows of
  `W2` (rows whose signed truth is below zero, injected or real), `SG-A` minus `SG-0`, has a point
  estimate at or below -MDE and an upper interval end below zero. MDE is the minimum detectable
  effect, the larger of 0.404 points of effective capacity (experiment 420's rotation r1 figure) and
  the absolute difference between the two seeds of `SG-0` on fold 2026-05, taken as the signed CRPS
  on the reverse-flow rows of the hidden pair pooled over both Primaries. The MDE is written to a
  file that refuses to be overwritten before any contrast is compared with a bar. G2: `SG-A` minus
  its shifted-sky placebo on the same rows has the same bar. G3, no harm in `W0`: signed CRPS over
  all rows of the hidden pair, `SG-A` minus `P0`, read on the plain `W0` fits (seeds 0 and 1) and on
  the `SG-A-open` fits (the export gate forced open, seed 0, paired with `P0` seed 0), against three
  limits on each of the two readings, which makes six limit checks. The recommended limits are +0.10
  points pooled over the two Primaries and +0.20 at each Primary, each read against the point
  estimate (decision 8, Option A). The maintainer ruled on 2026-11-24, in the manager's chat, for
  Option A with these limits. Each limit is read against the point estimate. A synthetic world
  precedes the real worlds. `SG-A` passes it when, in both seeds, its sign accuracy outside a
  2-point dead band is at least 0.95 at the two exporting synthetic Primaries, and, with the export
  gate forced open, it calls reverse flow on under 1% of sunny midday half-hours at the two other
  Primaries. Its area under the receiver operating characteristic curve for the sign posterior `ρ`
  must also exceed that of `SG-A-S` by at least 0.10 (rung 0b's own bar, `V1_AUC_GAP`), on distinct
  history tokens of the two exporting Primaries whose real shortwave likelihood `u` is at least 0.3
  and which lie outside the dead band, in each seed. Two checks run before any synthetic fit, and
  the maintainer has confirmed the margin (decision 7). The reachability check requires a sky-blind
  rule that knows the noise-free demand exactly to leave an area at least 0.10 plus 3 standard
  errors below 1. When it fails, a 5-point day offset of demand noise is added to the world and the
  check is run once more. The noise check requires 0.10 to be at least 3 standard errors of the area
  difference, by the correlated form of Hanley and McNeil with an assumed correlation of 0 between
  the two areas, at the number of distinct days and at assumed areas of 0.95 and 0.85. An area
  computed on fewer than 30 tokens of either class is `not evaluable`. A stage is flagged `stopped
  while falling` by the rule of the section on the training budget, which reads the flags of `SG-A`
  alone. Verdict: the first row that matches, and every combination of readings reaches exactly one
  row. (0) Stopped, no verdict: later fits stay unrun by design because the manager halts the line
  (at the quick look's harm stop, or by declining a family of fits), a check before the first fit
  fails (the Stage 0 test of the flag, the power check of G3, the reachability check after the
  demand noise has been added, or the noise check of the margin), the maintainer stops the line when
  Branch B returns decision 2, the farm injection at the visible Primaries is not built (decision
  4), the manager declines further `SG-A` fits at the continuation floor, an area of the synthetic
  world is `not evaluable`, or decision 1, 2, 3, 7, or 8 is still unconfirmed when the first wave
  would start. A fit run without its continuation, after the manager allows it at the floor stop,
  counts as flagged in rows 7 to 9, because its stage cannot be shown to have converged. A
  correctness failure found at any time is recorded in the first line of the report. (1) `invalid`:
  a correctness check fails (the holdout guard, the fold point of the loss, the origin-time weather
  assertion, the no-two-worlds figure guard, or a known-answer test on synthetic arrays). (2) The
  synthetic world fails: the line stops before any real-fold fit, and the verdict is `inconclusive`
  when any fit of `SG-A` hit the 400-pass cap, and `not supported` when every fit stopped by
  patience. (3) `failed`: a fit raised and was not rerun under the rerun rule, or a fit that a
  counted reading needs is missing although no stop of row 0 or row 2 applies, and no row above
  applies. (4) `not supported`: G3 fails on the plain fits or on `SG-A-open`, and the line stops
  under experiment 420's registered rule of no further gates or capacity priors. (5) `supported`:
  G1, G2, and G3 pass. (6) `inconclusive`: G3 is `unresolved`. (7) `inconclusive`: G3 passes, G1 or
  G2 misses, and at least one counted stage of an `SG-A` fit in `W2` is flagged, with the counts
  reported. (8) `not supported`: G3 passes, G1 misses, and no counted stage of an `SG-A` fit in `W2`
  is flagged; the output-only result of `SG-0` is reported. (9) `inconclusive`: G3 passes, G1
  passes, G2 misses, and no counted stage of an `SG-A` fit in `W2` is flagged, whether or not the
  fits of `SG-0` or `SG-A-S` are flagged. The comparators' flags are never read, because an
  under-trained comparator only makes a contrast easier to pass. Treatment minus `P0` on the
  reverse-flow rows, `S`, G4 (the folded CRPS at the eight MVA-only Primaries from the same fits),
  and every share of the gap from `P0` to `S` are read-outs with no bar, and no verdict row rests on
  them. The futility test decides only whether seeds 2 to 4 run. Their result is a pre-registered
  secondary read, reported beside the verdict, and it cannot change any verdict, so no decision is
  read twice. A rerun is the same attempt only when a fit raised or a check that reads no bar
  statistic fails before the fix and passes after the fix, the fix changes no constant, an Opus
  review confirms the fix, and both outputs are kept. `inconclusive` and `not supported` are never
  rerun. Named constants (each a `Final` of `e00721_core.py`, set before any real data are read):
  0.404; 0.10 pooled and 0.20 per Primary (the limits of G3, decision 8, Option A); 0.95 (sign
  accuracy); 1%; 2 points (the dead band) and 2 points (the reactive floor `Q`), each with its own
  name; 28 days; 0.3 (the gate's `u_min` and the likelihood floor of the margin); 400 and 30
  (synthetic passes and patience); 98.33%; 2.0 (injection level); 10 days (the block length); 20
  (the minimum number of resampling units); 10 passes (the noise window); 150 passes (the cap of the
  continuation); 0.5 and 0.05 (the flag's noise multiple and share of descent); 2% (the plateau
  false-flag ceiling of the Stage 0 test); 0.80 (the power of the G3 check); 0.10 (the area margin,
  which the maintainer confirmed on 2026-11-24); 30 tokens (the smallest class); 0.20 (the harm
  limit at the quick look); 0.20 (Branch B's threshold on the interim G1 point estimates, a separate
  constant fixed before any data); 3 (the standard errors that the reachability and noise checks
  require); 0.95 and 0.85 (the assumed areas of the noise check) and 0 (its assumed correlation); 5
  points (the demand-noise day offset); 60 and 120 passes with patience 8 (the real-fold budget);
  0.7 and 1.3 (the capacity sensitivity); 0.5 (the probability that a supervised sample's history
  rows are folded); 1.0 (`SIGN_CE_WEIGHT`, the weight of the sign head's cross-entropy in the stage 1
  loss); -4 (the head's starting bias); 20,000 (the bootstrap's resamples) and 721 (its
  generator seed); and 0.6 GPU-process-hours (the continuation floor). The 6 sub-steps of the
  injection's interpolation come from experiment 720's code.
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

- **MSA network and `M*`:** the axial-attention demand network of experiment 713, in `M1`, the arm
  of experiment 713 that was selected as the reference network. The page uses `P0`, `S`, `SG-0`,
  `SG-A`, and the other arm names as the arm table below defines them.
- **F4, F7, and F9:** the follow-up experiments on the MSA network that share its GPU queue. F4 adds
  neighbouring substations' rows to the network (experiment 715). F7 adds an estimate of the
  unmetered photovoltaic (PV) power in each history row (experiments 716 and 717). F9 adds similar
  days from the past (experiment 718 and its second part).
- **`HT`:** F7's arm that gives the network the PV estimate `p` of the unmetered
  solar power in the history rows and the target-row PV forecast `p_0`.
- **The Candidate:** the class 2 Candidate demand network of [experiment
  667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), whose
  weather columns and B1 columns (14 trailing-temperature and solar-gain columns) the MSA network
  reuses.
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
  The visible Primaries D11 and D16 carry the same farms as the hidden pair D5 and D10 (by experiment
  506's farm map), so the sign head learns the sign of farms whose output it later reads at the hidden
  pair. A hidden-sign result is therefore a ceiling on what a sign head learns where no labelled
  Primary shares a farm with the hidden pair.
- **Export gate:** the per-Primary switch of experiment 420, set once per refit from the readings,
  that allows a Primary to export at all.
- **Sign posterior `ρ`:** the probability that a history token exported, from the gross-demand
  read-out or from the learned head.
- **Signed CRPS:** the continuous ranked probability score of the 11 signed quantiles against the
  signed truth, in points of capacity. A score below another score is better.

## Where this experiment sits in the queue

**Experiment 720's whole-span hindsight posterior failed both of its bars on injected generator
schedules, and the stop rule closed its D8 branch.** The hindsight model scored 0.123 F1 below the
reading-blind reference M2 (interval -0.213 to -0.054), and its false-alarm rate at its worst host
was 59.2% against a bar of 3%. Rung 1 tests a different mechanism, a learned sign head with its own
baselines `P0` and `SG-0`, so no bar on this page changes. The failure is a prior that hindsight
gains over a plain reference are hard to obtain.

**Rung 1 depends on F7's `HT` code, which is reviewed and running, and not on the five-seed verdict
of F7.** The `HT` code supplies the PV estimate `p` in the history rows, the target-row forecast
`p_0`, the as-of capacity table, and the 28-day shifted-sky placebo. The code of experiments 716 and
717 is imported from their folders on `PYTHONPATH` and not copied. **[Decided by the manager]**

**Rung 1 sits ahead of F9 in the GPU queue: F4, then F7, then rung 1, then F9.** The earlier design
note placed rung 1 behind F4 and F9 and after an F7 verdict. The Opus review of the queue found that
F9 answers no question directly and has the weakest prior, and that rung 1 is the direct test of
the always-megawatt forecast. **[Decided by the manager]**

**The first wave may run only after the maintainer has settled G3 and the margin (decisions 7 and
8), and after both Opus code reviews.** The first wave is the code-works check, the synthetic world
at seed 0, and the quick look (about 1.2 to 1.3 GPU-process-hours under Branch A and about 1.7 to
2.0 under Branch B, one seed, one real fold). No step before the synthetic world waits for rung 0b,
for F7's five-seed verdict, or for F9. The next action is the code, which a Sonnet agent writes from
this page, followed by two Opus reviews. Nothing on this page has been run.

| Prerequisite | State on 2026-11-24 | Blocks the synthetic world? |
|---|---|---|
| F7's `HT` code (experiments 716 and 717) | Reviewed; part 1's fits are running | No: the code is imported as it stands |
| F7's `HT` forecasts on the real folds | Not stored: the stored forecasts are arms `H` and `Z`, which cannot stand in for `S` | No: `S` in `W0` is fitted from scratch on four folds |
| Experiment 720's injection code | Written and tested on synthetic worlds; unmerged | No: imported on `PYTHONPATH` |
| Rung 0b's real run | Not run; waits for the maintainer's choice between the two options of the rung 0b note | No: only the `SG-E` block waits for it |
| Experiment 713's known-answer gate | Waived by the maintainer; every verdict carries the waiver sentence | No |

## What this plan changes from the design note

**The design note (`mva_to_mw_design.md`, section 5.2) fixes the arms, the readings G1 to G4, and 71
fits at about 14 GPU-process-hours.** This plan makes the changes below, each with its reason.

| Design note | This plan | Reason |
|---|---|---|
| Synthetic fits: not stated, with the harness of experiment 713's gate in mind | 400 passes, patience 30, stage 1 only, for `SG-A` and `SG-A-S` | Finding C1 of the F4 audit (next section). `SG-0` has no sign accuracy, so its synthetic fits gate nothing and are not run |
| `S` fitted in `W0` on 4 folds | Unchanged: `S` is fitted from scratch on all four folds of `W0`. F7's `HT` forecasts, if stored by then, are a descriptive comparison only | No `HT` forecast is stored, and F7's recipe differs from rung 1's (see "The `S` arm of `W0`") |
| `SG-E`, `SG-E-S`, and `SG-E-open` in the first wave (about 16 fits and 4 synthetic fits) | Out of the first wave; reconsidered only if the anchored read-out passes both of its pre-registered bars in experiment 720 | Rung 0b's synthetic worlds show the anchored read-out losing to a reading-blind reference |
| Second seed of the winner only; no rule for further seeds | Seeds 0 and 1 are read together for G1 to G3. A quick look comes first. Seeds 2 to 4 run only when the futility test does not fire, outside the projection, and give a secondary read | Month-to-month variation, not seed noise, sets the interval width (the Opus review of the queue); the two-seed reading is the only counted one, so no decision is read twice |
| `SG-A`'s farm injection at the visible Primaries stated as a condition | Built in the first code review, in the readings of every arm in both worlds, or `SG-A` leaves rung 1 and the plan stops before `W2` | With `SG-E` out, `SG-A` is the only treatment, and an injection that only `SG-A` saw would not isolate the sign head |
| Injected capacity in `p` estimated by experiment 662's probe on the folded readings | The oracle capacity (the farm's true capacity), as experiment 720's code does, with 0.7 and 1.3 times that capacity as the one sensitivity rule | Experiment 720's injection code takes the true capacity. The oracle capacity is a look-ahead input, so every score is a ceiling for that reason as well as for the recorded weather |
| Bootstrap resampling unit not stated | Blocks of 10 consecutive valid dates, resampled fold first and then block within the fold by a resampler written in `e00721_score.py`, with the interval widened for the small number of blocks | Each fold is a separate fit, and a block of origins overlaps the next block in valid time. About 11 blocks in `W2` and 15 in `W0` are too few for a percentile interval at 98.33% |
| `FL` (the folded loss with a free signed mixture) as a negative control | Not run | The symmetry of the folded loss is known, and `FL` gates nothing |

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
| `W2` and `W0` real folds | Experiment 713's real-fold budget: at most 60 passes per stage, raised to 120 passes for a stage that hits the cap of 60 | 8 | Experiment 713's budget is the recipe that decision 2 keeps; the fit prints its stopping pass |

**The real folds keep experiment 713's budget for the scored checkpoint, and a measurement of the
learning that the budget cut off guards G1.** Experiment 716's known-answer gate (F7's KF9, in
synthetic worlds) shows why. Its arm `H` trained 60 passes with no early stopping, and patience 8,
applied afterwards, picked passes 4, 23, and 5 of the three seeds, where the best of the 60 passes
were 43, 55, and 32. The audit of that gate found the PV history path still learning at pass 60.
Every `SG` arm uses the same path, so G1 is read under a budget that is known to under-train the
history path. No counted reading compares a rung 1 arm with `M*` or with F7, so the budget does not
need to match theirs for G1 to G3. A rule that reads the shape of the patience-8 curve alone cannot
tell a plateau from a slow fall, because the truncated curve does not hold the information. The
guard therefore continues training and measures the lost learning directly.

**Stage 0 states which stage trains the sign head, before any flag is computed.** The design note
puts the hidden pair's folding in stage 1 and fixes no stage for the sign head, and experiment 716
trains F7's gains `a` and `c` at the stage 2 rate. The page pre-registers the default that the sign
head trains in stage 1, with the trunk, because the head reads trunk tokens and its cross-entropy
loss is added to the stage 1 loss. The first Opus code review reads the code and records the stage,
and one of two guards then applies.

- **Branch A, the sign head trains in stage 1: the combined flag on stage 1.** The scored checkpoint
  of every score and contrast is the one at the patience-8 stop, as in experiment 713 (decision 2).
  In the same run, stage 1 of an `SG-A` fit in `W2` keeps training after the patience-8 stop until
  patience 30 ends it, with a cap of 150 passes in total, and the run logs the patience-8 best and
  the patience-30 best validation loss. The gap is the patience-8 best minus the patience-30 best.
  The noise is the sample standard deviation of the validation loss over the last 10 passes of the
  patience-8 run (all the passes it ran, when it ran fewer than 10). The noise measure is the page's
  own. The stage's descent is its first-pass validation loss minus the patience-30 best. Stage 1 is
  flagged `stopped while falling` when the gap exceeds 0.5 times the noise and is at least 5% of the
  descent. Experiment 713's stored stage-1 curves, continued as flat plateaus with their own
  post-best noise, give a modelled false-flag rate of about 0.7% per stage-1 stage, so about 5.5%
  across the eight stage-1 tests of `SG-A` in `W2` (four folds, two seeds). An `SG-A` fit is flagged
  when its stage 1 is flagged, and either seed's flag sets the flag. Stages 2 and 3 are not read by
  any verdict row. The gains `a` and `c`, which experiment 716 trains at the stage 2 rate, are
  therefore unguarded, and their learned gains are reported and carry no bar. `SG-0` carries the
  same `HT` inputs, `a`, and `c` at the same budget, so any under-training of them is shared by both
  arms of G1 and does not favour `SG-A`.
- **Branch B, the sign head trains in stage 2 or 3: a direct budget check in the quick look.** No
  loss rule has a tolerable false-flag rate on stages 2 and 3, because those stages start near their
  optimum and their descents are tiny. The quick look instead fits `SG-A`, and `SG-0`, a second time
  on fold 2026-05 at seed 0, with patience 30 and a cap of 150 passes in the stage that trains the
  sign head. Every later stage is refitted from that stage's patience-30 state, so each of the two
  extra fits has a G1 point estimate of its own. The quick look compares the two budgets on the
  held-out folded loss of the hidden pair and on the two interim G1 point estimates. If those two
  estimates differ by more than 0.20 points, decision 2 returns to the maintainer before the rest of
  `W2` runs, and the result is row 0 if the maintainer stops the line. The 0.20 is a constant fixed
  here before any data, and it is not derived from the MDE, which is written later. This reading is
  interim and carries no bar, so it adds no look at the verdict. The two extra fits cost about 0.5
  to 0.8 GPU-process-hours. Rows 7 to 9 then read no flag, and the verdict rows treat every `SG-A`
  fit as unflagged.

**Both branches keep the scored chain untouched.** Experiment 713's `train_stage` draws each pass's
order from the seed, the stage, and the epoch, and holds a constant rate after warm-up, so passes
after the patience-8 stop do not change the passes before it. The patience-8 best state is the state
that is saved, scored, and passed to the next stage, and the continuation's better state is saved
apart and never passed on. `SG-A` and `SG-0` then start stage 2 from a stage 1 of the same budget,
and G1 compares heads and not budgets. The code-works check asserts that the scored `SG-A` state
equals the state of a run with no continuation. The rerun at 120 passes and the continuation are
one run, so a stage that hit the first cap is not trained twice.

- **The validation loss.** The validation loss that early stopping and the flag read is the folded
  loss at the hidden pair of every `SG` arm and of `P0`, computed against the folded reading and
  never against the signed truth, so that no checkpoint is selected on the hidden sign. Experiment
  713's validation split (lead days 1, 2, 4, 7, 10, and 14 before the cutoff, with the stage
  embargo) is inherited unchanged.
- **The test in Stage 0.** The flag is run on every stored learning curve of experiments 713, 715,
  and 716 that runs past its patience-8 stop, before any fit. The test requires the combined flag to
  fire on experiment 716's arm `H` in world WH at seeds 0, 1, and 2 and on its arm `HT` in world WT
  at seed 0, and not to fire on its arms in world WN at seeds 0 and 2. A second test continues each
  of 713's stored real-fold stage-1 curves as a flat plateau at its own post-best mean, with
  Gaussian noise of its own post-best spread, and requires a false-flag rate of 2% or less per
  stage. Both tests can fail, and a failure stops the line before any fit with no verdict (row 0). A
  code fix that changes no constant may rerun the test once, and both outputs are kept. Every other
  stored curve is printed with its flag for the write-up.
- **Where the constants came from.** The constants 0.5, 0.05, and the 2% ceiling were chosen on the
  same stored curves that the test above then checks, so the test checks the code and is not
  independent evidence for the rule. Its margins are thin. Arm `HT` in world WT at seed 0 fires at a
  share of descent of 0.061 against 0.05. Arm `H` in world WN at seed 1 sits at 0.048 and 0.66 noise
  units, just under both thresholds. The plan leaves it out of the quiet set and records no reason
  for doing so, so the quiet set is chosen after the fact, and the test does not cover that curve.
  The known-answer curves are single-stage fits that train the `HT` gains with the trunk, so they
  stand in for stage 1 only if `a` and `c` train there.
Every fit's stopping pass per stage, its flag, and the learned gain of each new path as a share of
its reachable range are printed and appear in the write-up.

**The continuation adds GPU time, which the projection counts.** In the 160 stored real-fold fits of
experiment 713, a stage ran 12.1 s per pass in stage 1, 4.0 s in stage 2, and 3.9 s in stage 3, and
a whole fit took 641 s on average (0.178 GPU-process-hours; the median is 567 s). A stage-1
continuation from a converged stop needs 22 more passes (30 minus 8), which is 266 s, or 0.07
GPU-process-hours, for each of the 8 `SG-A` fits in `W2`: 0.6 in all. A stage that is still learning
keeps resetting patience and runs to the cap of 150 passes. For a stage-1 stop at the mean of 38
passes, that is 112 extra passes, or 0.38 GPU-process-hours per fit and 3.0 for the eight. A
continuation of all three stages would cost 0.68 per fit and 5.4 for the eight, which is why the
continuation runs only in the stage that Stage 0 fixes.

**The floor of 0.6 is enforced between fits, because the session cannot pause a running fit to ask
the manager.** A started continuation runs until patience 30 or the cap of 150 passes ends it.
Before each `SG-A` fit in `W2` starts, the session adds up the continuation time of the fits that
have finished. When that sum has reached 0.6 GPU-process-hours, the line stops at that fit, and the
session asks the manager before the next fit starts. A fit whose continuation is running when the
sum passes the floor finishes, and the manager is told afterwards. The worst case before the manager
is asked is about 0.98 GPU-process-hours of continuation (the floor plus one running continuation),
a total of about 14.0. If the manager allows further `SG-A` fits without the continuation, each of
those fits counts as flagged in rows 7 to 9, fixed here before data, because its stage cannot be
shown to have converged. If the manager declines the fits, the result is row 0. The spend above the
floor is the row "The continuation above its floor" of the table of conditional items below.
**The cost of the larger synthetic budget is inside the design's figure of 0.2 GPU-process-hours
per fit.** Experiment 713's amended gate, at 400 passes and patience 30, took 4,027 s for 8 stage-1
fits plus its checks, about 0.14 GPU-process-hours per fit (run log of experiment 713). The figure
of 0.2 covers a rerun at the pass cap. If the first synthetic fit measures a different time, the
manager is told before the second fit starts.

## Worlds, arms, and folds

**Three worlds are fitted, and each treatment is tested in the order synthetic, `W2`, `W0`.**

- **Synthetic world (stop gate).** Gross demand is a fixed function of calendar and temperature,
  `X = D - C u`, and the reading is `sqrt(X² + Q²)`. Two synthetic Primaries export on clear
  middays and two never do. The exporting pair holds the real export share of D12's history
  windows, 70% to 93% of days from March to September, because a world with balanced windows
  cannot show the failure of a read-out that starts from "every row imported". The stop
  gate reads both seeds, and `SG-A` passes only when both seeds pass. The non-exporter check forces
  the export gate open at the two non-exporting Primaries, as rung 0b's source check does at D5,
  because a shut gate would pass the check by construction. The synthetic world fits `SG-A` and
  `SG-A-S` only, because `SG-0` has no sign accuracy (its `ρ` is fixed at 0) and gates nothing.
- **`W2` (injected).** Folds 2026-04, 2026-05, and 2026-06 are sunny, and fold 2025-12 is dark and
  tests for harm. Every arm of `W2` reads the same readings, which include the two other farms'
  recorded output injected into the four visible Primaries. Only the loss of `SG-A` reads the true
  sign of the visible Primaries. The injection is in the readings of every arm and is not added for
  `SG-A` alone, so the contrast of G1 isolates the sign head.
- **`W0` (no injection at the hidden pair).** The same four folds, with the hidden pair folded and
  nothing added to it. The four visible Primaries carry the same injection as in `W2`, so that
  `SG-A` has positive labels in `W0` as well and G3 tests the same treatment as G1.

**The arms are listed with their role, and every arm fits from scratch.** The
fits load no stage-1 state and no frozen encoder of the Candidate, because both were trained on the
hidden pair's signed readings.

| Arm | What it is | Role | In the plan |
|---|---|---|---|
| `P0` | `M*` recipe on the folded readings, output read as signed | Reference: the present network, a low bar | Yes |
| `S` | `M*` recipe with F7's `HT` inputs on the signed readings | Ceiling: the sign is given | Yes |
| `SG-0` | `HT` with the sky-blind head, the gate, the folded loss, the target term `-c_i p_0`, and `ρ` fixed at 0 | Experiment 420's output-only recipe on this network: the control that isolates the history sign | Yes |
| `SG-A` | `SG-0` plus the learned sign head | Treatment | Yes |
| `SG-A-S` | `SG-A` with `p` and `p_0` from the sky 28 days earlier (`W2` only) | Placebo: the sky must carry the gain | Yes |
| `SG-A-open` | `SG-A` in `W0` with the gate forced open at the hidden pair (one seed, four folds) | Tests a gate where the ungated network fails | Yes |
| `SG-E`, `SG-E-S`, `SG-E-open` | `SG-0` plus the anchored gross-demand read-out, with its placebo and forced-open fit | Treatment 1, held back | No: see below |

**The learned sign head reads each history token after the axial trunk.** For a history token `x`
of a gate-open Primary, `ρ = e_i × 1[u ≥ u_min] × sigmoid(w · LayerNorm(x) + b)`, where `e_i` is the
gate, `u_min` is 0.3, and `b` starts at -4, so an untrained head gives `ρ` near 0.02. In training, a
sample from a signed Primary has its history rows folded with probability 0.5, the target loss
stays signed, and a cross-entropy loss between `ρ` and the true sign is added. The
supervision set is D9, D11, D14, and D16 with their injected farms. D5 and D10 are excluded from the
sign supervision in every world and stage, and D15 stays out of it too; a report with and without
D15 is a read-out and carries no bar.

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
out.** Experiment 720 pre-registers two bars for the anchored read-out in rung 0b: an area bar (an
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
  anchored read-out passes its area bar and does not beat a reference. Neither case reopens `SG-E`,
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

## Counted readings and verdict rows

**The three counted readings are G1, G2, and G3, and G4 is a report.** The decision rule in the
front matter states each bar. Each counted reading is read at 98.33% from a block bootstrap and
reads seeds 0 and 1 together. In words:

- **The bootstrap.** The two-stage resampler is written in `e00721_score.py`.
  `studies.bootstrap.bootstrap_difference_at_level` draws its units uniformly from one flat list
  keyed by its `month` column, so relabelling `month` with a fold-and-block identifier would give a
  flat draw over fold-blocks, which is about 0.77 times as wide as the two-stage draw on the
  seed-pair null (decision 8). No counted reading uses the function, and no file under
  `src/studies/` changes. Each resample draws one seed, then draws folds with replacement, then
  draws blocks of 10 consecutive valid dates with replacement within each drawn fold, across both
  hidden Primaries. The folds are separate fits whose blocks share an offset, and a block of origins
  overlaps the next block in valid time, so blocks on the valid date are the unit on which the
  pairing already runs. The interval describes a one-seed contrast and is conservative for the point
  estimate, which is the mean over the two seeds. Before the reverse-flow restriction, the three
  sunny folds of `W2` fall in about 11 blocks, and `W0` falls in 15 (4 per fold, and 3 in fold
  2026-06 because valid times stop before 2026-07-01), with a Kish effective number of 13.8. Both
  counts are under the minimum of 20 resampling units that the page requires, so each interval is
  widened about the point estimate. The first factor is the ratio of the Student t quantile to the
  normal quantile at 98.33%, with B - 1 degrees of freedom and B the Kish effective number of
  blocks, because the reverse-flow rows gather in the sunny blocks: about 1.22 at a Kish B of 10.1
  (the 11 blocks of `W2`) and 1.15 at a Kish B of 13.8 (the 15 blocks of `W0`). The second factor is
  the square root of B / (B - 1), about 1.05 at 10.1 and 1.04 at 13.8. The two factors together are
  about 1.29 in `W2` and 1.19 in `W0`, and both are planning figures, because the counted readings
  use the Kish number of their own reverse-flow rows. The folds are not counted again as degrees of
  freedom, because the two-stage resampler already carries the fold-level clustering, and the page
  applies no fold-count correction on top of the two-stage draw. On the seed-pair null such a
  correction would change the interval by about 3%. Each reading uses 20,000 resamples and the
  generator seed 721, and the power check of G3 runs once with them, because the 98.33% tails of the
  resample move by up to 0.1 points at D5 between runs of 4,000 resamples. The write-up reports the
  Kish number of the reverse-flow rows, and states beside every interval that the June valid dates
  of folds 2026-05 and 2026-06 share weather outcomes across two fits. The resampler and the
  widening were fixed before any rung 1 data were read.
- **G1, the gain from the history sign.** `SG-A` minus `SG-0`, signed CRPS on the reverse-flow rows
  of `W2`. The bar needs a point estimate at or below -MDE and an upper interval end below zero. The
  MDE is the larger of 0.404 points and the difference between the two seeds of `SG-0` on fold
  2026-05, taken as the signed CRPS on the reverse-flow rows of the hidden pair, pooled over both
  Primaries. The MDE file is written, and refuses to be overwritten, before any contrast is
  compared with a bar. The quick look prints interim contrasts and compares none with a bar.
  Experiment 420 derived 0.404 as 2.8 bootstrap standard deviations (0.144 each) over 14
  Primary-month blocks, at a 95% two-sided level with 80% power. The same recipe at 98.33% gives
  about 3.23 standard deviations, or about 0.47. The page keeps 0.404 as a fixed constant. That MDE
  sits below the 0.47 that the same recipe gives at 98.33%, so G1's point-estimate bar is easier to
  pass than its level implies. The interval bar is unchanged.
- **Treatment minus `P0`, a read-out.** The signed CRPS of each arm minus that of `P0` on the
  reverse-flow rows is reported with its interval and carries no bar, because experiment 420 found
  the ungated reference scored the same as its arm A on those rows (6.99 against 6.98), so any arm
  that can go below zero beats `P0`. No verdict row rests on it.
- **G2, the sky carries the gain.** `SG-A` minus `SG-A-S` on the same rows, with the same bar.
- **G3, no invented reverse flow, is an open decision.** In `W0`, `SG-A` minus `P0` over all rows of
  the hidden pair, on the plain fits (seeds 0 and 1) and on `SG-A-open` (seed 0, paired with `P0`
  seed 0). There are three limits on each of the two readings, which is six limit checks: pooled
  over the two Primaries, at D5, and at D10. The recommended limits are +0.10 points pooled and
  +0.20 at D5 and at D10. The design note says "within +0.05 points of `P0` pooled" and "within
  +0.20 at either Primary", and does not say whether a limit binds the point estimate or the upper
  interval end. Experiment 420's own check, B4, reads point estimates (its write-up reports "+0.03
  at D5"), and the design note copied the numbers and not the reading. The stored seed pairs show
  that the pooled limit and the reading both matter (decision 8). The maintainer has ruled for
  Option A, so the verdict rows read G3 under that reading; the first wave still waits for the rest
  of Gate 3. The six checks combine as follows: G3 fails when any check shows harm, is `unresolved`
  when no check shows harm and any point estimate is above its limit, and passes otherwise. The
  write-up reports each gate decision per world and fold, because the gate is expected to stay shut
  at the un-injected pair, and a shut gate passes G3 by construction.
- **G4, the eight MVA-only Primaries.** The folded CRPS of each arm at the eight, from the same
  fits, with its interval and no bar. G4 is read at seed 0, and the seed-1 fits are reported beside
  it where they exist. The design note's two-seed detectable difference of 0.25 points therefore
  does not apply to G4, and this page states no one-seed figure.

**The verdict table is strictly first-match, and it reads in this order.** Row 0 holds every stop
that gives no verdict. A stop leaves later fits unrun by design, so row 0 comes before the row for a
missing fit, and fits that a stop of row 0 or row 2 leaves unrun never fall in row 3. The stops are
the manager's halt (the quick look's harm stop, or a declined family of fits), a failed check before
the first fit (the Stage 0 test of the flag, the power check of G3, the reachability check after the
demand noise has been added, or the noise check of the margin), a decision of the maintainer (1, 2,
3, 7, or 8) that is still unconfirmed when the first wave would start, the maintainer's stop when
Branch B returns decision 2, a farm injection at the visible Primaries that is not built (decision
4), the manager's decline of further `SG-A` fits at the continuation floor, and an area of the
synthetic world that is `not evaluable`. A correctness failure found at any time is recorded in the
first line of the report, even when row 0 gives the verdict. Row 2 (the synthetic world fails) is
read before row 3 (`failed`), because a failed synthetic world leaves the `W2` and `W0` fits unrun,
and the stop is the verdict. Row 3 holds a fit that raised and was not rerun, or a needed fit that
is missing although no stop applies. Row 7 sits ahead of rows 8 and 9 because a stage of `SG-A` that
stopped while its validation loss was still falling must make a G1 or G2 miss `inconclusive`, and
not `not supported`. Rows 7 to 9 read the flags of `SG-A` alone, because an under-trained `SG-0` or
`SG-A-S` only makes its contrast easier to pass, so a miss with a flagged comparator is a cleaner
negative and not an excuse. Row 6 sits ahead of rows 7 to 9 so that an `unresolved` G3 reads
`inconclusive` whatever G1 and G2 say. Row 9 holds one case, G1 passing and G2 missing with `SG-A`
unflagged, and `SG-0` or `SG-A-S` may be flagged.

**Every combination of readings reaches exactly one row.** With every fit run and no stop, G3
passes, fails, or is `unresolved`. When G3 fails, all 8 cells of G1, G2, and the flag of `SG-A` fall
in row 4, and when G3 is `unresolved`, all 8 fall in row 6. When G3 passes, the 8 cells fall in row
5 (2 cells: both bars pass, flagged or not), row 7 (3 cells: a miss with a flag), row 8 (2 cells: G1
misses without a flag, whatever G2 reads), and row 9 (1 cell). Under Branch B no fit is flagged, so
row 7 cannot be reached, by design.

**The smallest gain worth porting is not set here, because this experiment ports nothing.** A
`supported` verdict allows rung 2 to be planned. Rung 2 sets its own threshold against the cost of
adding the sign head to the network.

**The experiment can fail in every row.** A miss of G1 is the likeliest outcome, and the page then
reports the output-only result of `SG-0` as the finding. The verdict reads `SG-A`, the only
treatment in the first wave, on seeds 0 and 1 together. A pre-registered gate stays pre-registered:
a reading chosen after the scores were seen is labelled post hoc and changes no verdict.

**The seeds are fixed in advance, and no decision is read twice.** Seeds 0 and 1 give the verdict.
The futility test decides only whether seeds 2 to 4 run. Their result is a pre-registered secondary
read, reported beside the verdict. It cannot change an `inconclusive` verdict to `supported`, and it
cannot change any other verdict. A `supported` or a `not supported` verdict at two seeds is final,
and seeds 2 to 4 do not run. Seeds 2 to 4 also do not run after a row 7 `inconclusive`, because
the same patience-8 budget would flag the same stage again. They run only after a row 6 or a
row 9 `inconclusive` that passes the futility test, and they cost 7.2 GPU-process-hours. Month to
month variation, not seed noise, sets the interval width (the Opus review of the queue), so the
extra seeds narrow the seed-noise part of the interval only, and the secondary read informs the
design of rung 2.

### The count of sign experiments

**Rung 1 is the 21st or the 22nd experiment in the amendment's numbering, and the 24th or the 25th
in the reconciled count.** The amendment's rule on sign experiments counts a test as a statistic
with a bar stated in advance, at least one outcome of which produces or gates a sign label, admits a
hint or factor, sets a prior, or licenses a later design to cite the statistic as evidence for a
sign. Two lists of earlier experiments exist, and they differ. This snapshot dates from 2026-11-24
and is updated when the plan is approved.

- **The design note's list** has 10 experiments: 152, 372, 374, 387, 408, 412, 417, 420, 424, and
  427.
- **The later amendment's list** has 18: 79, 152, 358, 362, 365, 366, 370, 387, 408, 412, 417, 420,
  424, 427, 506, 508, 543, and 674. It lacks experiments 372 and 374, which the design note counts.
  It numbers experiment 722 the 19th, experiment 723 the 20th if experiment 723 is approved, and
  experiment 724 the 21st.
- **Experiment 720 meets the definition and is in neither list.** The bars of its rung 0b gate
  `SG-E` here, and its rung 0e writes a sign posterior.

**The two numberings follow from the lists.** Experiment 722 has run, and experiment 723 has stored
`W0` outputs and is done, so it is the 20th. In the amendment's numbering, rung 1 is then the 21st
if experiment 724 has not run, and the 22nd if it has. The reconciled count adds experiments 372,
374, and 720 to the amendment's 18, so rung 1 is the 24th if experiment 724 has not run and the 25th
if it has. The manager ruled "count it as the 22nd, count honestly" for rung 1, which is the
amendment's numbering with experiment 724 run first. The page states the range 24th to 25th for the
maintainer. A `supported` verdict admits a treatment to rung 2, and so to a signed forecast at D12,
which the amendment's rule on sign experiments counts. Experiment 420 tested the same kind of
statistic in the same kind of world.

## GPU projection

**The plan is 64 fits and about 13.6 GPU-process-hours, under the cap of about 14, and the first
wave is about 1.2 to 1.3 of them under Branch A and about 1.7 to 2.0 under Branch B.** The figure of
0.2 GPU-process-hours per fit is the design note's, from experiment 713's measured solo fits of 0.09
to 0.19 and a margin for predictions, fold builds, and reruns at the pass cap. Experiment 713's 160
stored real-fold fits took 0.178 GPU-process-hours on average, so the expected spend is lower: 64
fits at 0.178, plus the code-works check and the continuation floor, is about 12.2. The first fit
measures the real figure. The 13.6 is 64 fits at 0.2 (12.8), the code-works check (0.2), and the
continuation floor of Branch A (0.6). The 8 probe pretraining jobs add about 0.07, which the
margin of the 14 absorbs. Branch B replaces the continuation with two extra fits of 0.5
to 0.8 GPU-process-hours, for 13.5 to 13.8 in all.

**The waves run in this order, so that the first results arrive early and each wave can stop the
line.** One seed and one real fold come first. Running the quick look before synthetic seed 1 risks
0.6 GPU-process-hours on a line that the synthetic world then stops, which the maintainer's wish for
fast results justifies.

| Wave | What runs | Fits | GPU-process-hours | Running total |
|---|---|---|---|---|
| 1 | Code-works check: one tiny run that trains, scores, and writes to a scratch folder, plus one full-size real batch that measures the GPU and host memory peaks and checks the scored-state assertion. No number from it is quoted | 0 counted | 0.2 | 0.2 |
| 1 | Synthetic world, seed 0: `SG-A`, `SG-A-S`, on the pipeline of fold 2026-03, trained before 2026-03-01 and scored on March to June 2026 (about 0.02 GPU-process-hours more scoring per fit than a one-month span, re-measured at the code-works check; the projection stays) | 2 | 0.4 | 0.6 |
| 1 | Quick look: `W2`, fold 2026-05, seed 0: `P0`, `SG-0`, `SG-A`, with the continuation (Branch A: 0.6 to 0.7) or the budget check (Branch B: 1.1 to 1.4, from 0.6 plus the two extra fits of 0.5 to 0.8). Nothing reads G3 or the flag yet | 3, plus 2 in Branch B | 0.6 to 0.7 (A); 1.1 to 1.4 (B) | 1.2 to 1.3 (A); 1.7 to 2.0 (B) |
| 2 | Synthetic world, seed 1, then the rest of `W2`, seed 0: `P0` 3, `S` 4, `SG-0` 3, `SG-A` 3, `SG-A-S` 4 | 19 | 3.8 | 5.0 to 5.1 (A); 5.5 to 5.8 (B) |
| 3 | `W0`, seed 0: `P0`, `S`, `SG-0`, `SG-A`, and `SG-A-open`, 4 folds each | 20 | 4.0 | 9.0 to 9.1 (A); 9.5 to 9.8 (B) |
| 4 | Seed 1: `SG-0` 4 (`W2`), `SG-A` 8 (both worlds), `SG-A-S` 4 (`W2`), `P0` 4 (`W0`), after the manager's reply to "fits starting" | 20 | 4.0 | 13.0 to 13.1 (A); 13.5 to 13.8 (B) |
| | The continuation floor on the 7 other `SG-A` fits of `W2` that carry it (Branch A only; 3 in wave 2, 4 in wave 4) | 0 | 0.5 | about 13.6 (A) |

**The 64 fits are the fits that the three counted readings need.** The synthetic world needs `SG-A`
and `SG-A-S` at two seeds (4 fits). Seed 0 of `W2` needs `P0`, `S`, `SG-0`, `SG-A`, and `SG-A-S` on
four folds (20 fits). Seed 0 of `W0` needs the same arms but `SG-A-S`, plus `SG-A-open` (20 fits).
Seed 1 needs `SG-0` and `SG-A-S` on four `W2` folds for G1 and G2, `SG-A` on four folds of each
world, and `P0` on four `W0` folds for G3 (20 fits), because `experiment_store.losses_for_bootstrap`
refuses arms with different seeds. `S` in `W0` is fitted because no `HT` forecast is stored: the
stored forecasts are arms `H` and `Z`, which cannot stand in for `S`.

**A synthetic rerun, the `SG-E` block, the continuation above its floor, and seeds 2 to 4 come on
top, and each asks the manager first.**

| Item | Fits | GPU-process-hours | Condition |
|---|---|---|---|
| A rerun of the synthetic world, once, if a synthetic fit raises or a check that reads no bar statistic fails | 4 | 0.8 | Total about 14.4 |
| The continuation above its floor, if a stage keeps learning to the 150-pass cap (Branch A, stage 1 only) | 0 | up to 2.4 | Total up to about 16.0 |
| `SG-E` family (synthetic 4, `W2` 8, `W0` 8) | 20 | 4.0 | The anchored read-out passes both bars; total about 17.6 |
| Seeds 2 to 4 of `SG-0` and `SG-A` on `W2` (24 fits) and of `SG-A-S` (12 fits) | 36 | 7.2 | The futility test does not fire after a row 6 or row 9 `inconclusive`, and the manager approves |
| The design note's figure for comparison | 71 | about 14 | Fits `SG-E` and `SG-A`, and `S` in `W0` |

**The rare cases are inside the margin of the figure.** A first-run stage that hits the 60-pass cap
is rerun at 120 passes, and 4 of experiment 713's 160 stage-1 fits did. The 0.2 per fit covers those
reruns, and the per-fit mean of 0.178 leaves 0.02 for them. The capacity sensitivity (0.7 and 1.3
times the oracle capacity) is prediction-only and refits nothing.

**The compute rules of the slot hold for every fit.**

- **Slot and queue.** The GPU slot comes from the manager. One worker fits at a time, because four
  concurrent fits gave no gain. The queue is F4, F7, rung 1, F9. Rung 0b and the CPU work run
  beside rung 1 and use no GPU. The plan takes about 13.6 GPU-process-hours, and its wall-clock time
  is longer whenever an F4 or F7 process shares the GPU.
- **GPU processes and memory.** At most two GPU processes run at once across all agents. Each
  process has 6 GB of GPU memory, and a raise needs the maintainer. The total is 36 GB. Rung 1
  shares the GPU with at most one F4 or F7 process. The code enforces the 6 GB with
  `torch.cuda.set_per_process_memory_fraction`, because an F4 process was recorded at 8.2 GB against
  5.04 GB measured. The planning figure is F7's real-fit peak of 3.4 to 3.5 GB (`peak_gpu_mb` in
  experiment 716's logs), and F4's real-fit peak is 5.05 to 5.18 GB, so rung 1 should fit inside
  6 GB. Experiment 715's gate fits peaked at 10.6 to 11.7 GB, which breaks the cap, so a fit of that
  size would not run. Experiment 713's gate fits peaked at 5,090 MiB on the GPU and 6.3 GiB on the
  host. The code-works check measures the peak on one full-size real batch, and the first real fit
  measures it again.
- **Checks before each family of fits.** `uptime` shows a load of 24 or less (otherwise wait), and
  the `nvidia-smi` totals leave room for 6 GB. `run.sh` repeats both checks. No multiprocessing or
  `joblib` runs in a capped scope.
- **Host memory per fit.** The host cap is `MemoryMax=12G`, and the first fit measures its own peak
  and resets the cap before the second fit.
- **Every command** runs in `systemd-run --user --scope` with `MemoryMax`, `MemorySwapMax=0`, and
  `OOMPolicy=continue`, through the `.venv/bin/python` interpreter and never `uv run` inside the
  scope, with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `POLARS_MAX_THREADS`,
  and `RAYON_NUM_THREADS` set to 1, after a check that `free -g` shows at least 25 GB available. No
  test runs with `pytest -n`.
- **Messages to the manager.** The session sends "fits starting" before each family of fits (the
  code-works check, the synthetic world, the quick look, the rest of `W2`, `W0`, the second seed)
  and waits for the reply. The session sends "last fit ended" after each family. A fit that hits
  its pass cap is reported with its stopping pass.
- **The capacity probe.** Experiment 662's probe, re-run per fold and per world for D5 and D10 (8
  runs), pretrains a network on the GPU for about 30 seconds per run (about 4 minutes in all, or about
  0.07 GPU-process-hours) and runs its regression on the CPU. Experiment 662 reports about 8
  wall-clock hours for its whole design, which bounds the probe's cost, and the first probe run is
  timed and the manager told before the second.

## Stages, the quick look, and the seeds

**Each stage ends before the next starts, and only the stages' own stopping rules can end the line
early.**

1. **Stage 0, no GPU.** Code, unit tests on synthetic arrays, and two Opus code reviews. **Both
   reviews come before any script that reads real data runs**, which includes the world builders
   and the per-fold re-run of experiment 662's capacity probe, and before any fit. The first review
   records the stage in which the sign head trains (branch A or branch B of the training budget).
   Any script that reads a file that the synthetic tests do not exercise, such as the real farm
   tables, the folded readings, and the probe's inputs, gets a read-only smoke test in a capped
   scope that prints no values. The tests include the flag's tests on stored curves, the power check
   of G3 (below), the reachability check and the noise check of the margin, and a test of the
   two-stage resampler and the widening. **The power check of G3 computes the joint probability that
   all six limit checks pass at zero harm.** It uses the seed-pair null: two disjoint groups of
   experiment 713's stored `M1` seeds at D5 and D10 over the four `W0` folds, scored by this page's
   recipe. The check requires the joint probability to be at least 0.80, the power of experiment
   420's minimum-detectable-effect recipe. If it fails under the rule the maintainer picks (decision
   8), no fit runs until the maintainer amends G3, and the result is "stopped before any fit, no
   verdict". A failed test of the flag gives the same result.
2. **Stage 1, the code-works check.** A tiny run proves the code trains, scores, and writes, and one
   full-size real batch measures the peaks. It quotes no number.
3. **Stage 2, the synthetic world.** `SG-A` and `SG-A-S`: seed 0 first, then seed 1 only if seed 0
   passed. A failure stops the line under verdict row 2.
4. **Stage 3, the quick look.** `W2`, fold 2026-05, seed 0: `P0`, `SG-0`, and `SG-A`. The quick look
   reads the wiring, the stopping passes, and the learned gains, and prints every contrast labelled
   interim and compared with no bar. Its only stop is harm, and the harm stop gives no verdict: if
   `SG-A`'s signed CRPS over the rows of either hidden Primary whose signed truth is at or above
   zero exceeds the same Primary's `P0` by more than 0.20 points, the session pauses the line and
   asks the manager, and the result is "stopped by the manager, no verdict". The stop reads import
   rows because the reverse-flow rows are where an arm that can go negative gains over `P0`, and an
   all-rows contrast would hide harm. A one-fold point estimate without an interval cannot be read
   as a verdict, and no fold is cut from a counted reading.
5. **Stage 4, the rest of `W2`, then `W0`.** Seed 0 on all four folds of each world.
6. **Stage 5, the second seed, after the manager's reply to "fits starting".** `SG-0` seed 1 on fold
   2026-05 runs first and writes the MDE file, which refuses to be overwritten. Then come seed 1 of
   `SG-A` in both worlds, of `SG-0` and `SG-A-S` in `W2`, and of `P0` in `W0`, all on four folds.
   The verdict reads seeds 0 and 1 together from this stage on, and the stage's fits carry no bar
   of their own apart from the pooled readings.
7. **Stage 6, seeds 2 to 4, only after a futility test.** The futility test fires when the point
   estimate of G1 (`SG-A` minus `SG-0`, pooled over four folds and two seeds) is at or above zero,
   because a null reading has no route to a gain that three more seeds could find. The test is
   written before any treatment score is read. Seeds 2 to 4 run only after a row 6 or row 9
   `inconclusive` whose futility test does not fire, and they need the manager's approval because
   they lie outside the projection. Their result is the pre-registered secondary read, and it
   cannot change the verdict.

## Leakage and origin-time weather

**The target row reads only the ensemble forecast issued at or before the forecast origin, and a
test asserts the rule row by row.** Experiment 404's lag-free heads used weather issued after the
origin, which voided every comparison above 24 h of lead. The brief therefore prints one example
origin, its target valid time, and the issue time of the ensemble run it reads.

**History rows read recorded weather: CAMS (the Copernicus Atmosphere Monitoring Service's solar
radiation estimates) and ERA5 (the reanalysis of past weather by the European Centre for
Medium-Range Weather Forecasts, ECMWF), which arrive about a day late.** The use of recorded weather
is a lab shortcut, as in experiments 713 and 716. Every number on the page is a ceiling for a live
forecast. Experiment 424 lost 7.2 points of signed CRPS on reverse-flow rows when ensemble weather
replaced recorded weather. The oracle injected capacity (see the table "What this plan changes from
the design note") is a look-ahead input, so every score is a ceiling for that reason too. The
write-up repeats the contrast with the injected capacity at 0.7 and 1.3 times the oracle value,
which is the one sensitivity rule. The sensitivity is prediction-only: the fitted networks predict
again with `p` rescaled, and nothing is refitted, so it adds no GPU-process-hours beyond the margin
of 0.2 per fit.

**The hidden pair's own PV capacity in `p` is re-estimated from the folded readings, and never read
from the signed readings.** F7's as-of capacity table, for the six signed-MW Primaries, may come
from a probe on their signed readings (experiment 673 read the signed Primaries' capacity that way).
A capacity estimated from the hidden sign would leak the sign into `p`. Rung 1 therefore takes the
capacity of D5 and D10 from [experiment
662](../00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md)'s probe on the folded
readings, with data before the fold's start only, and takes the as-of table only for the other
Primaries. The network that the probe reads is trained on rows in which D5 and D10 appear only as
folded readings, so no signed reading of either Primary reaches the probe's weights. The code review
checks this. The probe's pretraining runs on the GPU and its regression on the CPU, once per fold
and world for D5 and D10 (8 runs), and its cost is in the GPU projection. The scale of `p` at every
Primary is fixed per fold from these two sources, both outside the scored rows. The as-of capacity
of the eight MVA-only Primaries carries F7's stated look-ahead in seven of eight folds, and it
affects G4 only.

**The holdout guard names the farm frame.** The row set ends before 2026-07-01, and the loader clips
the farm frame, which runs to 2026-09-10, so that no valid time on or after 2026-07-01 is read.

## Decisions

**Decisions the manager has ruled on are marked [Decided by the manager], decisions that the
maintainer has ruled on say so, and open decisions are marked [Maintainer] with one recommendation
each.** The standing MSA approval reserves "changes to thresholds or verdict rules" for the
maintainer, so Gate 3 stays closed for every decision that sets a threshold or a verdict rule until
the maintainer confirms it.

- **A. Rung 1 depends on F7's `HT` code, not on F7's verdict, and runs ahead of F9 [Decided by the
  manager].**
- **B. The training budget is checked against finding C1 and stated above [Decided by the
  manager].** The synthetic budget is 400 passes with patience 30.
- **C. `SG-E` is reconsidered only when the anchored read-out passes both of its pre-registered bars
  in experiment 720 [Decided by the manager].**
- **D. The verdict reads seeds 0 and 1 together, and seeds 2 to 4 follow the futility rule and give
  a secondary read [Decided by the manager].**
- **E. The cap is about 14 GPU-process-hours [Decided by the manager].** A rung 1 that needs more
  asks first. This plan needs about 13.6, with the continuation above its floor, a synthetic rerun,
  the `SG-E` block, and seeds 2 to 4 all excluded and each asked of the manager first.
- **1. Approval and counting.** The maintainer's approval of rung 1 reached the manager as a relay
  through the MSA discussion session on 2026-11-24, in these words: "Rung 1 (hidden-sign fits, about
  14 GPU-process-hours) IS covered by the standing MSA approval; it still needs your GPU projection
  and 'fits starting' line to the manager before each fit family; follows the queue (F4, F7, F9,
  then rung 1) and keeps every standing rule". The manager later changed the queue to F4, F7, rung
  1, F9. A ruling that names an order may not cover a different order, so the maintainer is asked to
  confirm the reorder as well as the ruling. The ruling covers "about 14" GPU-process-hours, and
  this plan needs about 13.6. The relayed text is saved in no brief: a search of the private briefs
  found the quoted words only inside plan reviews 2 and 3, so this page is its only record until the
  manager saves it with its time. The manager has asked the maintainer to confirm the relayed
  ruling, and on 2026-11-24 the maintainer approved rung 1 at about 14 GPU-process-hours in the
  manager's chat. On 2026-11-24 the maintainer approved the queue reorder (F4, F7, rung 1, F9) in
  the manager's chat. On the sign-experiment count the manager ruled "count it as the 22nd, count
  honestly", and the section "The count of sign experiments" states the range.
- **2. The real-fold budget [Decided by the manager; confirmed by the maintainer on 2026-11-24].**
  Experiment 713's recipe (patience 8) scores the checkpoint, and a measurement of the learning that
  the budget cut off guards G1 (the training budget section). G1 is read under a budget known to
  under-train the history path.
- **3. The borrowed bars [Decided by the manager; confirmed by the maintainer on 2026-11-24].** The
  harm stop of 0.20 (read on import rows) and the level of 98.33% are accepted, because each is
  pre-registered on this page and covered by the Opus plan reviews. The third borrowed bar, the area
  margin of the synthetic world, is decision 7.

| No. | Open decision | Recommendation |
|---|---|---|
| 4 | [Maintainer] Whether `SG-A` leaves rung 1 when the farm injection at the visible Primaries is not built in the first code review | Build the injection, because `SG-A` is the only treatment while `SG-E` is out |
| 5 | [Maintainer] Whether `SG-H` (addendum 1's hindsight posterior) joins rung 1 | No, until rung 0e's first known-answer score has been read |
| 6 | [Maintainer] Whether `SG-E` returns on any ground other than the two bars of experiment 720 | No, because the synthetic worlds say the anchored read-out loses and the 4.0 GPU-process-hours would put the total over the cap |
| 7 | [Maintainer, confirmed 2026-11-24] The margin of 0.10 in area under the curve, `SG-A` minus `SG-A-S`, with the four parts below | Pre-register it as defined below. It is rung 0b's own bar in its own unit. Gate 3 stays closed until the maintainer confirms |
| 8 | [Maintainer, ruled 2026-11-24 for Option A] How G3 is read and where its limits sit (the evidence tables and two options below) | Option A with +0.10 pooled and +0.20 at each Primary; the measured joint pass rate at zero harm is 0.90 (15 of 15 plain, 18 of 20 open). Gate 3 stays closed, so no fit runs and no verdict row binds on G3 until the maintainer rules |

**The margin of the synthetic world is rung 0b's own bar in its own unit, with four parts.** Rung
0b's pre-registered margin is 0.10 in area under the receiver operating characteristic curve, with
the real sky above the shifted sky (experiment 720's decision rule, row V1 of addendum 1, and
`V1_AUC_GAP` in its code). Rung 1 reuses that bar for the sign posterior `ρ`: the area for `SG-A`
minus the area for `SG-A-S` is at least 0.10, on distinct history tokens of the two exporting
synthetic Primaries whose real shortwave likelihood `u` is at least 0.3 and which lie outside the
2-point dead band, in each of the two seeds. Area does not depend on the share of exporting tokens,
and the restriction to `u` of 0.3 or more removes the information that the placebo keeps through the
likelihood gate. A margin in sign-accuracy units is not pre-registered, because the token count is
unknown, each history token is scored several times, and the accuracy of the shifted-sky arm is
expected to sit below 0.95. Two checks, a rule for a failed check, and the maintainer's confirmation
are written before any synthetic fit.

1. **A reachability check.** The synthetic gross demand is a noise-free function of calendar and
   temperature, and the `SG` arms keep the calendar and the folded reading in the tokens that feed
   the head. Given the demand exactly, the two candidate signs of a reading are separated wherever
   only one of them fits, so `SG-A-S` could read the sign with no sky at all. The check computes the
   area of a sky-blind rule that knows the demand exactly, on the margin's token set, and requires
   it to leave at least 0.10 plus 3 standard errors below 1. If the check fails, demand noise (a
   5-point day offset, as experiment 720's harder worlds used) is added to the world before any fit
   and the check is run once more. Only when the noisy world also fails does the line stop (row 0).
2. **A correlated-difference standard error.** The two areas are computed on the same tokens, so
   the standard error of their difference uses the correlated form of [Hanley and McNeil
   (1983)](https://doi.org/10.1148/radiology.148.3.6878708) or, once scores exist, of [DeLong et al.
   (1988)](https://doi.org/10.2307/2531595), and not the single-area formula of [Hanley and McNeil
   (1982)](https://doi.org/10.1148/radiology.143.1.7063747). Both correlated forms need a
   correlation between the two areas that no data give before a fit, so the check before any fit
   assumes a correlation of 0 (the independent form, which is conservative for positively
   correlated areas). It is computed with the class counts set to the numbers of distinct days that
   hold an export token and an import token, because tokens within a day share that day's sky, and
   at assumed areas of 0.95 and 0.85. The margin of 0.10 must be at least 3 standard errors.
3. **The outcome when a check fails.** No synthetic fit runs until the maintainer amends the
   margin or the world, and the result is "stopped before any fit, no verdict" (row 0). A rerun of
   the noise check after the reachability check has added demand noise is not an amendment. An area
   computed on fewer than 30 tokens of either class (experiment 720's minimum) is `not evaluable`,
   and the result is also row 0, because the margin cannot be read.
4. **The maintainer's confirmation.** The margin sets a threshold that a verdict row reads, so Gate
   3 keeps it closed until the maintainer confirms the margin.

**Decision 8, the reading of G3 and its limits, depends on stored seed pairs of two networks of the
same recipe.** Experiment 713 stores `M1` forecasts for 5 seeds on every fold. At D5 and D10, on the
four `W0` folds (2025-12, 2026-04, 2026-05, and 2026-06), with leads of 24 h up to 240 h after the
12:00 UTC origin and valid times before 2026-07-01 (98,856 paired rows, 15 blocks of 10 valid
dates), two disjoint seed groups of the same recipe stand for "treatment" and `P0`, so the true harm
is zero. The null is computed with this page's own recipe: valid-date blocks, a fold-then-block
resample with one seed per resample, and the widening of the section "Counted readings and verdict
rows" about the two-seed point estimate (Kish B of 13.8, 12.8 degrees of freedom).

| Null reading | Splits | Upper-end passes | Point estimates over a limit (at +0.05 pooled) | Upper end minus point estimate (pooled, D5, D10) |
|---|---|---|---|---|
| Plain: two seeds against two | 15 | 0 | 3 | 0.34, 0.62, 0.24 |
| Open: one seed against one | 20 | 0 | 4 | 0.27, 0.52, 0.21 |

- **The upper-end reading cannot pass.** The upper end sits 0.27 to 0.34 points above the point
  estimate pooled, against a limit of +0.05 or +0.10, so G3 passed in 0 of 35 null readings, and
  `supported` would be unreachable.
- **A point-estimate fail bar stops the line on noise.** The two-seed point estimate has a standard
  deviation of about 0.046 points pooled across splits (0.067 for one seed against one), so +0.05 is
  about one standard deviation. Seven of the 35 null readings put a point estimate above a limit.
- **Experiment 420 read its own check, B4, on point estimates.** B4 reads "within +0.05 points of
  `P0` pooled, with no single Primary above +0.20", and the write-up of experiment 420 reports point
  estimates against it.

Two options exist, and both are pre-registered before any fit that reads `W0`. The table counts the
null readings under Option A at two pooled limits, with +0.20 at each Primary.

| Pooled limit | Plain: pass / fail / unresolved | Open: pass / fail / unresolved | Joint pass rate |
|---|---|---|---|
| +0.05 | 12 / 0 / 3 | 16 / 0 / 4 | 0.72 (product 0.64) |
| +0.10 | 15 / 0 / 0 | 18 / 0 / 2 | 0.90 (product 0.90) |

- **Option A (recommended): read G3 as experiment 420 read B4, with +0.10 pooled and +0.20 at each
  Primary.** G3 passes when every point estimate is within its limit, with the interval reported. G3
  fails only when harm is shown: the lower end of the widened interval is above zero and the point
  estimate is above its limit. Otherwise G3 is `unresolved`. The joint pass rate is counted directly
  over the 60 assignments of the five stored seeds to the roles that the two readings share (`P0`
  seed 0, `P0` seed 1, `SG-A-open`, and the two `SG-A` seeds). It is 0.90 at +0.10 pooled (the plain
  reading passes in 15 of 15 and the open reading in 18 of 20) and 0.72 at the design note's +0.05.
  The product of the plain and the open pass rates is 0.90 and 0.64, and it is a lower bound,
  because the two readings share `P0` seed 0. The power check computes the direct count. At +0.05
  the joint rate of 0.72 still misses the 0.80 that the power check requires, so +0.05 would stop
  the line, and at the recommended limits the power check passes at 0.90.

- **Option B: keep the upper-end reading with wider limits.** The figures in this option are the
  writer's reading of the null, and no source states them. They come from bootstrap runs of 20,000
  resamples, and they move a little between runs, because with 4 folds the fold-level draw has few
  distinct outcomes and the 98.33% upper end at D5 lands on a jump. A limit equal to the mean gap
  above would pass about half of the null readings on that check. The limit that 80% of the plain
  null readings meet on each check alone is about 0.42 points pooled, 0.78 at D5, and 0.32 at D10,
  and the 95th percentiles are about 0.50, 0.97, and 0.40. Limits of +0.45 pooled, +0.85 at D5, and
  +0.40 at D10 give a joint pass rate of 0.74 to 0.78 over three runs, which misses 0.80. Limits of
  +0.50 pooled, +1.00 at D5, and +0.42 at D10 give 0.89 to 0.95 over three runs, so D5's +0.20 would
  have to rise fivefold and D10's would double. The maintainer would have to accept limits of this
  width as "no harm".

The power check of Stage 0 restates the 0.80 target of experiment 420's recipe as the joint
probability over all six limit checks, computed from the seed-pair null under the option the
maintainer picks. The maintainer ruled for Option A on 2026-11-24, so the verdict rows read G3
under that reading.

**The contract decisions stay with a human in the production repository.** They are the unit label
of a signed forecast at an MVA series, the sign convention, the clamp at zero for the eight, the
scoring of a signed forecast against an unsigned observation, and the effective capacity at D12.
This experiment changes no contract and makes no request to NGED.

## Reviews and pre-run gates

- **Gate 1:** a fresh Opus agent that did not write the code reviews it, and a second fresh Opus
  agent reviews the fixed code. Both reviews come before any script that reads real data runs
  (including the world builders and the per-fold re-run of experiment 662's capacity probe) and
  before the first fit, the synthetic fits among them. The reviews check leakage, the fold point of
  the loss at zero megawatts, the history-power clip in `v` units, the origin-time weather, the
  injection (no two worlds of one Primary on shared dates in any figure), the holdout, a match with
  the recipe copied from experiments 713 and 716, and the stage in which the sign head trains. Any
  script that reads a file that the synthetic tests do not exercise (the real farm tables, the
  folded readings, and the probe's inputs) gets a read-only smoke test in a capped scope that prints
  no values, before the script's first real run.
- **Gate 2:** "fits starting" to the manager before each family, and a reply.
- **Gate 3:** the maintainer has confirmed decisions 1, 2, 3, 7, and 8, which are the relayed
  approval and the queue reorder, a training budget, bars, a margin, and the reading of G3 that
  verdict rows read. On 2026-11-24 the maintainer confirmed the approval (decision 1), the margin
  (decision 7), and Option A for G3 (decision 8). On 2026-11-24 the maintainer also confirmed the
  queue reorder (decision 1), the real-fold budget with patience 8 (decision 2), and the borrowed
  bars (decision 3). Gate 3 is therefore open from the quota reset on 2026-11-25. The first wave may
  run only after both Opus code reviews. Decisions 4 to 6 are ruled, or the manager rules that their
  defaults stand.
- **Gate 4:** the code-works check and the first synthetic fit have measured their time per pass
  and their memory, and the memory cap is reset from the measured peak before the second fit.

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
- **GPU hours:** about 13.6 GPU-process-hours for the plan under Branch A (64 fits at 0.2, the
  code-works check, and the continuation floor) and 13.5 to 13.8 under Branch B, against a cap of
  about 14, and about 12.2 at experiment 713's measured per-fit mean. The first wave is about 1.2 to
  1.3 under Branch A and about 1.7 to 2.0 under Branch B.
- **Saved outputs:** every fitted network's state file under the experiment's data folder, never in
  git, and per-row forecasts through `experiment_store.write_predictions`.

### Choices fixed in the code

- **Day offset:** the synthetic world's 5-point day offset is the standard deviation of a per-day
  offset of the gross demand.
- **Censor rule:** a reading within 1e-4 of `Q` takes the probability mass inside `|X| <= Q`, and
  every other reading takes the density of both signs with the Jacobian `r / |X|`.
- **Block counting:** blocks of 10 consecutive valid dates are numbered from all of a fold's rows
  before the reverse-flow restriction, so a block with no reverse-flow row stays a resampling unit
  with a zero count.
- **Sign head training:** the sign head trains in stage 1 only, with the trunk (Branch A). Its
  cross-entropy enters the stage 1 loss with the weight `SIGN_CE_WEIGHT` of 1.0, and the head's
  parameters sit in experiment 716's new-parameter group for stage 1 only. In stages 2 and 3 the head
  is frozen: it takes no sign step, no cross-entropy, and no gradient, and none of its parameters is
  in an optimiser group.
  The signed-expected history still reads the frozen head's `ρ` values in stages 2 and 3, but `ρ` is
  detached there, so the frozen head passes no gradient to the trunk through `ρ`.
- **Probe jobs:** the probe's pretraining job for each world and fold runs on the GPU (about 30
  seconds each, about 4 minutes for the 8 jobs, in a 6 GB process) and the regression that reads
  the capacity runs on the CPU. The probe runner sequences before the `W2` and `W0` fits.
- **Synthetic world on the real grid:** the pipeline of fold 2026-03 supplies the real weather, the
  Primaries D9 and D11 export, and D2 and D3 do not. All four train on the folded reading and the true
  sign of all four trains the head. Each process serves one fold month, because experiment 713's
  pipeline reads its fold month once.
- **Folded loss at D15 and D18:** the page names D15 and D18 in neither list, so they train on
  their stored readings with the plain likelihood, with no gate and no sign labels.
- **Sky blinding:** besides the shortwave, the solar-gain columns, and the solar-geometry columns,
  the longwave and precipitation columns and their ensemble spreads are blanked at a gate-open Primary,
  because both carry cloud.
- **Sign read-outs:** the sign accuracy uses every distinct token outside the dead band, with no
  restriction on `u`, because the page states none, and the bar stays as stated. A report-only
  read-out gives the same accuracy at `u` of at least 0.3, because night tokens are trivially right
  and inflate the pre-registered figure. The area uses tokens with `u` of at least 0.3. The
  false-call rate counts sunny midday half-hours (10:00 to 14:00 UTC with a clear-sky index of at
  least 0.7) of the two non-exporting Primaries, with the gate forced open.
- **Widening:** a reading widens its interval only when it has fewer than 20 resampling units.
- **Sign head read-out base:** the history power that the read-out averages is the signed expectation
  `(1 - 2ρ) v`, and its variance includes the sign variance `4ρ(1 - ρ) v²`.
- **Primary gate:** the per-Primary export gate is computed in NumPy from the training window's rows
  alone (the rule of experiment 417, design A).
- **Gap rows:** every signed Primary drops the half-hours at which any farm has a gap, in both
  worlds.
- **Synthetic fold and scored span:** the synthetic fit trains on data before 2026-03-01 (after the
  purge) and scores the never-trained tokens of March to June 2026, all before the 2026-07-01
  holdout, whose guard is kept. A one-month span cannot pass the noise check, because the check
  needs about 92 distinct days per class and a month holds at most 30. The distinct days per class
  with `u` of at least 0.3 outside the dead band, as export days and import days, are 21 and 24 in
  March, 28 and 29 in April, 29 and 30 in May, and 29 and 29 in June, so 107 and 112 over the four
  months, which gives 3 standard errors of 0.092 or less against the margin of 0.10. Stage 0 runs
  the reachability check and the noise check on exactly that scored set, with each exporter's own
  capacity, and the margin and the noise-check rule are unchanged. The synthetic fit replaces the
  power with synthetic power, so the real folds 2026-04 to 2026-06 and their training windows are
  untouched. The extra cost is about 3 more months of scoring (prediction) in each of the 4
  synthetic fits and about 1% less training data, an estimate of about 0.1 GPU-process-hours in
  all that the code-works check re-measures; the projection stays at about 13.6 until then.
- **Synthetic Primaries:** D9 and D11 export and D2 and D3 never export. The world uses the real
  grid of the fold 2026-03 pipeline, the real calendar with England's bank holidays, the real 2 m
  temperature, and the real shortwave likelihood `u`, in experiment 713's fixed demand table `w`.
  The page does not say which Primaries or which fold, so these are choices.
- **Synthetic sign labels:** the true sign of all four synthetic Primaries trains the sign head, and
  all four train on the folded reading. The page does not state this, so it is a choice. In `W2` and
  `W0` only the four visible signed Primaries carry a sign label, and the sign cross-entropy has
  weight 1.0, a value that the page does not state either.
  The two non-exporting synthetic Primaries carry labels too, but their export gate is 0 in training,
  so their labels give the head no gradient. They read the sky in training, and the check of the
  false-call rate blinds them to the sky when it forces their gate open at prediction.
- **Process split:** `e00721_cli.py` runs the fit, the predict, and the score phases as separate
  processes. The score phase writes the MDE file from the seed difference of `SG-0` in `W2` on fold
  2026-05, then runs the power check of G3 over experiment 713's stored `M1` seeds 0 to 4.

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
  bootstrap of the decision rule (blocks of 10 consecutive valid dates within a fold), at 98.33%.
- **Plots:** attention maps of `SG-A` on a sunny and a dull history window; day-ahead and 10-day
  forecasts against what happened (leads stop before 240 h), drawn for the hidden pair in `W0` and
  for the synthetic Primaries, never as a dated series at an injected Primary; the sign posterior
  `ρ` against the true sign as pooled rates by month or by hour at the injected Primaries, on
  aggregate or relative axes; learning curves with the stopping pass of each fit; and the synthetic
  world's area under the curve under the real and the shifted sky.
- **Anonymisation:** no figure or table shows any two of the host's reading, the injected reading,
  and the signed truth for the same dates, because the difference is the farm's output. A figure of
  an injected Primary shows the net reading only, and no dated plot of the sign posterior or of a
  forecast sits beside the net folded reading at an injected Primary. No farm series, no generator
  state, and no inferred unit count appears, and D8 appears only inside pooled counts of the eight.

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
  `PRIVATE/` subfolder. The first job creates the folder with `Path.mkdir`, and no script calls
  `reserve_experiment`, because the manager reserved the id.
- `data/ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/work/fits/`: F7's stored
  forecasts, read only as a descriptive comparison with `S` in `W0` if F7's phase 5 has stored `HT`
  by then.
- `data/ml-experiments/briefs/msa_architecture/mva_to_mw_design.md` and its two addenda: the
  design this plan follows (read only, never published).
