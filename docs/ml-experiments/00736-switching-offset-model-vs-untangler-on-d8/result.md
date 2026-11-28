---
id: "00736"
slug: switching-offset-model-vs-untangler-on-d8
title: >-
  Does a probabilistic model of switched load find more injected load transfers at D8 than a neural
  network's step detector?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: finished
date: 2026-11-27
kind: other
size: >-
  Medium. (1) Stored: private decode tables, call tables, and score tables under
  data/ml-experiments/00736-switching-offset-model-vs-untangler-on-d8/ only; no neural network is
  trained or run forward (the step heads' outputs on the test overlay are read from experiment
  732's stored predictions), and the offset model is not refitted (its tuned values are read from
  experiment 734's params.json). (2) Designs: more than one design defensibly tests the question
  (decode the offset model on the same transfers injected into D8's real record, run the step
  heads on experiment 734's worlds, or chain the step heads' windows), so the choice and every
  handicap are fixed below. (3) Callers: this folder's scripts, which import the folders of
  experiments 653, 660, 732, and 734 read-only; nothing imports them. (4) Serving path and
  degradation rule: not touched. (5) Contracts: not touched. CPU only, no GPU time, about 2.4
  CPU-hours, with a stop at 8 CPU-hours, on 1 core. Reviews: plan, code, numbers, and prose.
hypothesis: >-
  On the load transfers that experiment 732 injected into D8's own pre-engine days, the offset model
  H-nbr of experiment 734 finds a larger share of the transfers that last 1 day to 4 days and move
  4 step noise units or more (1.42 MW or more) than the best recalibrated step head of experiment
  732 (U-noBSP, the head with the higher validation F1), by the porting threshold of 0.15 or more.
  A start is found when a call has the same sign and starts within 6 hours of the injected start.
prediction: >-
  Written before any offset-model decode exists. The head's number was computed first, from stored
  files (see "The head's number was read before the plan was frozen"): U-noBSP finds 12 of the 21
  counted transfers (0.571). The counted cell holds 21 injected transfers (24 to 96 hours, 4 step
  noise units or more, in the 6 held-out months that lie inside experiment 653's record). P1
  (counted, C1): the hypothesis fails, and the head wins. H-nbr finds 0.20 to 0.45 of the 21 starts
  (5 to 9 transfers), so C1 (H-nbr minus U-noBSP) lies between -0.37 and -0.12. The reason is that
  D8 has the largest 6-hour block noise unit of the 15 Primaries (1.02 MW), so a transfer of 1.4 to
  4.2 MW is only 1.4 to 4 block noise units, below the 2 block noise units at which experiment 734
  scored its shallow cells. P2: in the 29 transfers of 6 to 24 hours and 4 step noise units or
  more, H-nbr finds at least 0.20 less than U-noBSP. P3: when a found event must have both a start
  call and an opposite-sign end call within 6 hours, U-noBSP's share in the counted cell falls by
  0.20 or more, and H-nbr's falls by 0.10 or less. P4: on starts moved 7 days later (the placebo),
  the share matched by chance is at most 0.08 for U-noBSP and at most 0.05 for H-nbr. P5: in the
  counted cell, H-nbr finds at least 1 more of the 11 transfers whose partner a neighbour mirrors
  than H-own does, and at most 1 more of the other 10. Maintainer-side estimate that the hypothesis
  holds (verdict `supported`): about 10%. The rest sits in `not supported` (about 50%),
  `inconclusive` (about 30%), `refuted` (about 5%), `partly supported` (about 3%), and `small gain`
  (about 2%, a verdict the sign test below makes nearly impossible). The resolution is coarse: 21
  transfers resolve a difference of shares only to about 0.25, so `refuted` needs both intervals
  below zero and is unlikely.
decision_rule: >-
  Fixed before any offset-model decode. One counted contrast, C1: the share of the 21 transfers of
  24 to 96 hours and 4 step noise units or more whose injected start has a matching call, H-nbr
  minus U-noBSP, paired on the same transfers. Each transfer falls in one cell of a table of paired
  counts: found by both (a), only by U-noBSP (b), only by H-nbr (c), by neither (d), with a+b+c+d =
  21. C1 equals (c - b) / 21, a number that moves in steps of 1/21 = 0.048. C1 counts when an exact
  two-sided sign test on the b+c discordant transfers gives p below 0.05, which needs at least 6
  discordant transfers split 6 to 0 or more unevenly. The page also reports two 95% percentile
  intervals for C1, one resampling the 8 draws (worlds) and one resampling the 6 calendar months,
  10,000 draws each; both are crude for 21 paired outcomes, so they decide only the choice between
  `not supported` and `inconclusive`. Veto V1: H-nbr's false edge calls per 100 scored base days
  (a call of D8 that matches no injected start or end edge within 6 hours) is no more than 1.10
  times U-noBSP's, on D8 only, with the same day count for both detectors. Verdict, one row per
  outcome. `supported`: c exceeds b, C1 counts, and V1 passes (C1 is then at least 6/21 = 0.29,
  above the porting threshold). `partly supported`: c exceeds b, C1 counts, and V1 fails.
  `refuted`: b exceeds c and C1 counts. `small gain`: cannot occur, because any gain that passes
  the sign test is at least 6 transfers (0.29); a gain of 1 to 3 transfers does not count and
  falls under `not supported` or `inconclusive`. `not supported`: C1 does not count, and the upper
  end of the wider of the two intervals is below +0.15. `inconclusive`: C1 does not count, and that
  upper end is +0.15 or more; also any result whose counted cell holds fewer than 20 transfers
  after the drops for gaps, and any result for which more than 2 of the 6 months hold no
  discordant transfer (the month-resampled interval then fails). The tests for fewer than 20
  transfers and for more than 2 empty months come before the sign test, so a result that passes
  the sign test with discordant transfers in only 3 of the 6 months is `inconclusive`. The porting
  threshold is +0.15, which is a gain of at least 4 of the 21 transfers (4/21 = 0.19; 3/21 = 0.14
  falls short). The
  threshold is above zero because the offset model needs a continuous record of 6-hour residual
  means, a panel of the 15 Primaries for its shared term, a demand forecast that uses no lagged
  power, and priors tuned on synthetic worlds, and each world takes about 15 CPU-minutes to decode
  (an estimate, see "Stages and compute"). The head, in contrast, reads a 48-hour window with no
  stored history, and a gain of 3 transfers or fewer does not justify those extra requirements.
  Gates G1 to G4 under "Gates" must pass first, or the verdict is `invalid`. The match rule, the
  tolerance, and the thresholds are fixed on this page before any offset-model decode is read; the
  recalibration report's `frozen_thresholds.json` fixes the head's map and threshold (digest below),
  and the 6-hour tolerance, the 2-hour merge, the sign rule, and the larger-transfer-first
  assignment are not in that file and were not checked on the validation overlay.
benchmark: >-
  U-noBSP, the step head of experiment 732 without the bulk supply point (BSP) input, with the
  isotonic map and the F1-rule threshold that the recalibration report froze on the validation
  overlay (chosen because it had the higher raw-map validation F1 score, the harmonic mean of
  precision and recall, 0.614 against 0.523; the isotonic F1 rule gives 0.606), read from experiment
  732's stored test predictions. U* (the head with the BSP input) is a reference row. H-own (the
  offset model without the shared term and neighbour evidence) is a reference row.
row_set_id: >-
  experiment 732's test bank (8 held-out months of D8's pre-engine days, 8 draws each), restricted
  to the 6 months that lie inside experiment 653's record: 288 injected transfers, 104 of 4 step
  noise units or more; not comparable with the leaderboard
metric: >-
  Share of injected transfers whose start has a matching call (same sign, within 6 hours), by
  duration band, for transfers of 4 step noise units or more (D8's step noise unit in experiment 732
  is 3.42 points, which is 0.355 MW, so 4 units are 13.67 points, or 1.42 MW; one point is 0.10379
  MW), with world-resampled and month-resampled 95% intervals; the same share under a rule that also
  needs an end call; false edge calls per 100 scored base days; share matched by chance on shifted
  starts.
verdict: inconclusive
best_arm: >-
  None. The offset model H-nbr found the start of 13 of the 21 counted transfers (0.619) and the
  step head U-noBSP found 12 of the 21 (0.571), so C1 is +0.048, which the sign test cannot tell
  from zero (p = 1.0). The step head is the better detector of transfers under 24 hours (see the
  table of shares by duration band under Result).
headline: >-
  On the 21 injected transfers of 24 to 96 hours and at least 1.42 MW, the offset model found the
  start of 13 and the step head found 12. The two detectors found the same 9, the step head alone
  found 3, the offset model alone found 4, and neither found 5, so the difference of +0.048 (1 of 21
  transfers) is inside the noise. The offset model found 2 of the 29 transfers of 6 to 24 hours and
  0 of the 54 transfers under 6 hours, against 17 of 29 and 29 of 54 for the step head. The
  pre-registered porting threshold of +0.15 was not reached.
lessons: >-
  The offset model finds about as many long transfers as the step head, within the noise. The offset
  model found only 2 of 29 transfers of 6 to 24 hours and 0 of 54 under 6 hours, against 17 of 29
  and 29 of 54 for the step head. The likely causes are its 6-hour blocks, its slow states, and D8's
  large block noise unit (see Limits), and the experiment did not test any of them. The step head
  raises about twice as many false calls on D8's own base days (2.70 against 1.42 per 100 days), so
  the offset model is the quieter detector. Requiring an end call as well as a start call lowers
  both shares by 2 of 21 transfers (the step head 12 to 10, the offset model 13 to 11). The
  21-transfer counted cell resolves a difference of shares only to about 0.25, so a larger bank of
  long transfers is needed before anyone can rank the two detectors on long transfers.
take_home_message: >-
  On the substation D8, a probabilistic model of switched load finds about as many long load
  transfers as a neural network that looks for steps in demand, and the probabilistic model almost
  never finds the short transfers, so the probabilistic model does not replace the neural network.
  The experiment could not say which detector is better on long transfers, because only 21 such
  transfers were available. The open question is whether the two detectors together find more
  transfers than either alone.
advice:
  - >-
    Do not replace the step head with the offset model for transfers under 24 hours at D8, because
    the offset model found 2 of 29 transfers of 6 to 24 hours and 0 of 54 under 6 hours, against
    17 of 29 and 29 of 54 for U-noBSP (table of shares by duration band).
  - >-
    Build a bank with many more than 21 injected transfers of 24 hours or more before ranking two
    detectors on long transfers, because 21 transfers resolve a difference of shares only to about
    0.25 and the sign test needs at least 6 discordant transfers split 6 to 0 (C1).
  - >-
    Excuse only calls near injected edges large enough for a detector to see, or count excused calls
    separately, because veto V1 excuses a call near an injected edge of any size, including the edge
    of a 0.3 MW transfer that neither detector can see (see the limit "Veto V1 excuses a call near
    an injected edge of any size").
  - >-
    Read the 6-hour and the 12-hour tolerance side by side when comparing a block-based detector
    with a half-hourly detector, because the post hoc 12-hour row moves the offset model from 13 to
    15 found and the step head stays at 12 (post hoc rows).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00736-switching-offset-model-vs-untangler-on-d8/
---

**We asked whether two detectors of load transfers, built in two earlier experiments of this
repository, find the same synthetic transfers at one substation.** The substation, D8, is a Primary
substation, the kind that steps voltage down towards customers. A transfer is the switching of part
of a circuit between two Primaries, which moves demand for hours to days. The first detector is the
untangling network of experiment 732, a neural network that reads two days of D8's electricity
readings at a time and gives, for each half-hour, the probability of a step in demand (its step
head). The second is the offset model of experiment 734, a probabilistic model of D8's whole history
that infers, for every 6-hour block, how much of the demand is switched power, which is power moved
by a transfer. We added the same synthetic transfers of known start, end, and size to D8's record
and asked each detector which starts it finds. The measure is the share of the 21 injected
transfers, lasting 24 to 96 hours and moving at least 1.42 MW, whose start each detector finds. The
plan registered that cell as the headline because it matches the part of the medium cell of
experiment 665 (transfers of more than 1 day and up to 7 days) that lasts 4 days or less. A detector
finds a start when it calls a step of the same sign within 6 hours of the injected start.

**The offset model and the step head found about the same number of long transfers, so the verdict
is `inconclusive`.** The offset model found 13 of the 21 and the step head found 12, a difference of
1 transfer that the experiment cannot tell from chance. The step head found far more of the shorter
transfers: 17 of 29 transfers of 6 to 24 hours and 29 of 54 transfers under 6 hours, against 2 and 0
for the offset model.

## Result

**The verdict is `inconclusive`: on the 21 counted transfers the offset model and the step head find
about the same number, and 21 transfers are too few to say which is better.** The counted cell holds
the 21 injected transfers that last 24 to 96 hours and move at least 4 step noise units, all in the
6 held-out months. A step noise unit is D8's noise scale for steps in experiment 732 (0.355 MW), so
4 units are 1.42 MW. C1 is the share found by the offset model H-nbr minus the share found by the
step head U-noBSP (the step head without the input from the bulk supply point), paired on the same
transfers. C1 is +0.048, which is 1 transfer in 21. The exact two-sided sign test on the 7 transfers
that only one detector found gives p = 1.0, so C1 does not count (C1 counts when the sign test gives
p below 0.05). The 95% interval is -0.25 to +0.25 when the 8 draws are resampled and -0.21 to +0.29
when the 6 months are resampled. A draw is one random placement of the injected transfers, and a
world is one draw of all 6 months merged into one record, so there are 8 worlds. The pre-registered
rule gives `inconclusive` because C1 does not count and the upper end of the wider interval, the one
resampled by draw (+0.25), is above the porting threshold of +0.15, the gain in share that would
justify adopting the offset model.

| Counted cell, 21 transfers | Found by U-noBSP | Not found by U-noBSP |
|---|---|---|
| Found by H-nbr | 9 | 4 |
| Not found by H-nbr | 3 | 5 |

**The gap rule, the gates, and veto V1 all pass.** No transfer was dropped for a gap in the
residual, so the cell holds 21. One of the 6 months holds no transfer that only one detector found,
which is within the rule's limit of 2. The four gates G1 to G4 all passed: the head's stored results
reproduce, the injection is exact, the offset model's decode reproduces, and the overlays reproduce.
Veto V1, which requires the offset model's false-call rate to be at most 1.10 times the step head's,
passes (below). The numbers in this section are from the final scoring of the 8 worlds; the head's
12 of 21 equals the interim number read before the plan was frozen.

**The offset model finds about as many long transfers as the step head, within the noise, and finds
almost no short transfers.** The table gives the share of injected transfers of at least 4 step
noise units whose start has a matching call, by duration band, for the four detectors that were
scored. The offset model with no shared term and no neighbour evidence is H-own, and U\* is the step
head with the bulk supply point input. Both reference rows are uncounted.

| Duration | Transfers | U-noBSP | U\* | H-own | H-nbr |
|---|---|---|---|---|---|
| under 6 hours | 54 | 29 (0.54) | 23 (0.43) | 0 (0.00) | 0 (0.00) |
| 6 to under 24 hours | 29 | 17 (0.59) | 16 (0.55) | 2 (0.07) | 2 (0.07) |
| 24 to 96 hours (counted) | 21 | 12 (0.57) | 9 (0.43) | 10 (0.48) | 13 (0.62) |

**The offset model is the quieter detector on D8's own days, so veto V1 passes.** V1 counts the
calls that match no injected start or end edge within 6 hours, on the 1,480 base days (D8's own days
before any transfer is injected, 185 days in each of the 8 draws). U-noBSP made 40 false calls in
168 calls on base days (2.70 per 100 days), and H-nbr made 21 false calls in 39 events (1.42 per 100
days). H-nbr's rate is 0.53 times the step head's, against the veto's limit of 1.10. Real D8 events
inside the base days count as false calls for both detectors.

**Requiring an end call lowers both shares by 2 transfers, and the chance-matched share (the
placebo) is near zero.** The placebo moves every injected start 7 days later, onto time where no
transfer was injected, and counts the calls that match a moved start.

| Secondary row | U-noBSP | H-nbr |
|---|---|---|
| Counted transfers found with both a start call and an end call | 10 of 21 | 11 of 21 |
| Starts matched by chance, counted cell (placebo, 14 starts kept) | 0 | 0 |
| Starts matched by chance, all transfers of at least 4 step noise units (63 starts kept) | 0 | 1 |

**Three of the five pre-registered predictions hold, one holds in half, and P1, the prediction about
long transfers, fails.** P1 predicts that the step head wins, and P1 fails in a direction that
favours the offset model.

| Prediction | Registered | Result | Outcome |
|---|---|---|---|
| P1 | H-nbr finds 5 to 9 of the 21, so C1 is between -0.37 and -0.12 | H-nbr finds 13, so C1 is +0.048 | Fails |
| P2 | In the 29 transfers of 6 to 24 hours, H-nbr finds at least 0.20 less than U-noBSP | 0.07 against 0.59, a gap of 0.52 | Holds |
| P3 | With an end call also required, U-noBSP's counted share falls by 0.20 or more and H-nbr's by 0.10 or less | U-noBSP falls by 0.095 (12 to 10), H-nbr by 0.095 (13 to 11) | Half holds: H-nbr's fall (0.095) is under 0.10, but U-noBSP's fall (0.095) is under 0.20, so the 0.20 half fails |
| P4 | Chance-matched share at most 0.08 for U-noBSP and 0.05 for H-nbr | 0 of 14 and 0 of 63 for U-noBSP, 0 of 14 and 1 of 63 (0.016) for H-nbr | Holds |
| P5 | H-nbr finds at least 1 more of the 11 mirrored transfers than H-own, and at most 1 more of the other 10 | H-nbr 6 against H-own 4 of 11, and H-nbr 7 against H-own 6 of 10 | Holds |

**The step head found more of the 11 mirrored transfers, and the offset model found more of the 10
unmirrored ones.** A mirrored transfer is one that experiment 732 mirrored at its partner Primary
(D3, D5, or D14), by adding the equal and opposite change there. Of the 11 counted mirrored
transfers, both detectors found 5, only the step head 3, only the offset model 1, and neither 2. Of
the 10 counted transfers with no mirror, both found 4, only the step head 0, only the offset model
3, and neither 3. Each split holds too few transfers for a test. Of the 21 counted transfers, 2 sit
in segments with neither a simulated engine array nor a simulated solar array added to D8's record,
where both detectors read the same signal. Each detector found both of those 2 transfers.

**Post hoc, a tolerance of 12 hours moves the offset model ahead by 3 transfers, which is still
inside the noise.** This row was not pre-registered, because the plan fixed the tolerance at 6
hours. At 12 hours both detectors find 10 transfers, only the step head finds 2, only the offset
model finds 5, and neither finds 4, so the step head finds 12 and the offset model finds 15. C1 is
then +0.143 (3 of 21), which is below the +0.15 threshold, and the sign test on 2 against 5
discordant transfers gives p = 0.45. The pre-registered verdict is unchanged.

## Checks

- **The checks that compare a median forecast and a mean forecast with the metered values do not
  apply,** because no forecast is scored.
- **One audit trigger fires, for P1, and no audit has been run.** The trigger "a result contradicts
  its prediction" fires for P1, because the offset model found more transfers than the plan
  expected. The small cell and the tuning years listed below are the explanations on offer. The
  trigger "a failed control" does not fire, because the placebo is near zero. No other trigger
  applies, because no quantile score, leave-one-out, nested arm, or sister interval is involved.
- **All four gates passed.** G1 reproduced the head's 304 calls and its recall of 0.490 for
  transfers of at least 4 step noise units, and the 12 of 21. G2 found the injected megawatts equal
  to the plan's values (worst relative error 5.8e-8), found no injected half-hour with a zero
  addition in 9,063 half-hours, and asserted that the offset model's panel index is the bank's
  half-hour plus 1, so transfers are not placed 30 minutes early. G3 reproduced the 131 stored
  events of experiment 734's un-injected decode exactly. G4 reproduced experiment 732's overlay
  checksum, its 384 transfers, and its 768 edge rows.
- **The world design that ran was Design A with the simulated engine and solar arrays added, so both
  detectors read the same signal.**

## Limits that apply to this result

**Five handicaps of the plan apply to the result, and each is listed here so that a reader can
discount the number.**

- **The head's recall was computed before the plan was frozen.** The step head's 12 of 21 came
  from stored files before the offset model was decoded. Only the offset model's side was
  predicted blind.
- **Experiment 734's tuning years overlap the test months and favour the offset model.** All 8 test
  months fall before the end of the years that experiment 734 tuned on. The tuning touched the
  priors, the time constant, the noise units, the block scales, and the shared-term loadings. No
  injected transfer was seen in tuning, and the step heads never trained on those months. A win by
  the offset model would have carried this caveat, and the offset model's 13 against 12 carries it
  too.
- **Veto V1 excuses a call near an injected edge of any size.** A call within 6 hours of an injected
  edge of the right sign counts as not false even when that transfer moves 0.3 MW, a transfer that
  neither detector can see. Both detectors get the same excuse. Each rate is therefore lower than
  its unexcused rate, and the veto tests the ratio.
- **The `small gain` verdict row cannot be reached with 21 transfers.** Any gain that passes the
  sign test is at least 6 transfers (0.29), so a gain of 1 to 3 transfers can only land in
  `inconclusive` or `not supported`.
- **The placebo and the real rows do not cover exactly the same starts.** Each world merges one draw
  of each of the 6 months into one record. The placebo drops every moved start that crosses a month
  end, lands inside a transfer, or meets another drop rule, so only 177 of the 288 starts remain in
  the placebo. Transfers near a month's end are scored in the real rows.

**Six further limits apply, and the last one concerns the registered text itself.**

- **The counted cell holds only 21 transfers, so the experiment resolves a difference of shares only
  to about 0.25.**
- **The transfers last 96 hours or less, so the experiment says nothing about the long, deep
  transfers on which experiment 734 scored the offset model.**
- **The size band is below the offset model's design range, so the weak showing on short transfers
  may reflect D8's large block noise unit as well as the offset model.** At D8, 4 step noise units
  are 1.39 block noise units (1.42 MW against 1.02 MW), below the 2 block noise units where
  experiment 734 scored its shallow cells.
- **Both detectors read the future of the event (see "Look-ahead"), so the result bears on labelling
  a historical record and not on real-time use.**
- **The experiment tests one substation.**
- **The registered text has three inconsistencies, and the Result follows the decision rule's own
  table.** First, the prediction says that a transfer of 1.4 to 4.2 MW is 1.4 to 4 block noise
  units, "below the 2 block noise units" of experiment 734's shallow cells, but that range straddles
  2 block noise units. Second, the prediction says that `refuted` needs both intervals below zero,
  while the decision rule defines `refuted` by the sign test (b exceeds c and C1 counts). Third, the
  prediction calls `small gain` nearly impossible, while the decision rule says that `small gain`
  cannot occur.

## Compute

**The experiment used 0.91 CPU-hours, against the plan's estimate of 2.4 and its stop at 8
CPU-hours.** The time log sums the start-to-end time of every job on one core. The 16 decodes (8
worlds, 2 arms) took 0.82 CPU-hours, with H-own taking about 2.1 minutes and H-nbr about 4.0 minutes
per decode (3.1 minutes on average), and G3's un-injected decode took 4.1 minutes. The plan's
estimate of 15 CPU-minutes per world, and its worst case of 27.6 minutes per decode, were far too
high. The plan had taken experiment 734's median of 27.6 wall-minutes per scoring job as a guide to
the time of one decode. Peak memory was 0.57 GB, against the plan's 1.5 GB.

## What the result means for choosing a detector

**The offset model cannot replace the step head on D8, because the step head sees the transfers
under 24 hours that the offset model does not.** On the transfers of 24 to 96 hours the two
detectors are tied, with the offset model giving fewer false calls. Of the 21 counted transfers, 16
were found by at least one detector (9 by both, 3 by only the step head, and 4 by only the offset
model), against 13 for the offset model, the better single detector. A union of the two detectors
would also carry both detectors' false calls. The result leaves open whether a rule that uses both
detectors, where the step head gives short calls and the offset model long ones, finds more
transfers than either alone.

## The plan as registered

**The two earlier experiments cannot be compared as they stand, because they measured different
rows.** Experiment 732 scored 8 held-out months of D8's real days with simulated transfers of 1 to
96 hours, one 48-hour window at a time. Experiment 734 scored the 15 Primaries over a continuous
record from 2022 to mid-2026, with transfers of hours to months. The offset model cannot be decoded
on one 48-hour window, and the step head cannot see more than 48 hours. The experiment put both
detectors on the same transfers.

**Everything from here to "Method" is the plan as it was frozen before the offset model was decoded,
and is kept unchanged.** The plan's "Open questions before the first run" were settled by the code
review, and the world design that ran is stated above.

## Design

**The offset model is decoded on experiment 732's transfers, added to D8's record.** Three designs
could put both detectors on identical transfers.

- **Design A (chosen).** Take the 288 transfers of experiment 732's test bank that fall in the 6
  held-out months inside experiment 653's record. Build each of the 8 worlds from D8's power plus
  the segment's synthetic engine and solar output plus the transfers, so that the offset model reads
  the same signal as the step head (experiment 732's test segments add simulated engine or solar
  output in most of the 64 segments, and the head reads that sum together with the synthetic sky).
  Add the mirror to the partner Primary when experiment 732 mirrored the transfer (D3, D5, or D14),
  where the mirror is the equal and opposite change added at the partner. Decode the 15-Primary
  panel with H-own and H-nbr, with the tuned values of experiment 734. Read the step heads' outputs
  on the same transfers from experiment 732's stored test predictions. The code review checked that
  experiment 734's `apply_additions` and `WorldDecoder(power=...)` accept the engine and solar
  arrays of experiment 732's overlay segments (`engine` and `solar`), and they do. Had they not, the
  worlds would have held transfers alone, the offset model would have read a cleaner series than the
  head, and the "Every handicap" list would have said so.
- **Design B (rejected).** Run U-noBSP and U* on experiment 734's D8 worlds. Experiment 734
  injected events into D8's record in its scored years, which lie later in the record than the
  heads' training days, and the heads were trained only on pre-engine days, with a synthetic sky and
  a BSP meter that experiment 732's banks supply and experiment 734's worlds do not. The sizes
  also differ: experiment 732's transfers are 0.31 to 4.2 MW, and experiment 734's shallow cell
  starts at 2.05 MW at D8. The test banks of experiment 732 are the rows that the heads' stored
  predictions cover, and no other rows have heads' outputs.
- **Design C (rejected).** Chain experiment 732's daily windows into months and decode only the
  heads' output. Experiment 732's stored predictions are already read this way, and chaining windows
  gives the offset model no continuous record.

**The 6 held-out months inside experiment 653's record hold 288 injected transfers, and all of them
last 96 hours or less.** The two oldest of the 8 held-out months fall before experiment 653's record
begins (2022-01-01), so their 96 transfers have no residual to decode. The durations are
log-uniform from 1 to 96 hours, so the long cell of experiment 665 (more than 7 days) is empty, and
the question is about the part of experiment 665's medium cell (transfers of more than 1 day and up
to 7 days) that lasts 4 days or less. The table counts the 104 transfers of 4 step noise units or
more (13.67 points, 1.42 MW), which is the size band where experiment 732's recalibrated recall is
reported. All 288 transfers have a start edge on a scored half-hour, and all 288 have an end edge on
one; the counts below are start edges.

| Duration | Injected transfers of at least 4 step noise units | With a mirror at D3, D5, or D14 |
|---|---|---|
| under 6 hours | 54 | 24 |
| 6 to under 24 hours | 29 | 12 |
| 24 to 96 hours (counted) | 21 | 11 |

**Every handicap is stated before the run.**

- **The step head sees 48 hours and calls edges, not events.** The head has no memory beyond its
  window, so a transfer of 24 hours or more reaches the head as two separate windows. The plan
  scores the start edge alone for both detectors (the generous reading for the head), and scores the
  rule that needs both ends as an uncounted secondary row.
- **The offset model works on 6-hour blocks.** The offset model cannot place a start more finely
  than 6 hours, which is why the one start tolerance is 6 hours (12 half-hours, one block) for both
  detectors. The tolerance is wider than the 1 half-hour that the recalibration report used for the
  head, so the head's 6-hour number is higher than the report's.
- **The size band sits below the offset model's design range.** 4 step noise units of experiment
  732 is 1.39 block noise units at D8 (1.42 MW against 1.02 MW), and experiment 734 scored nothing
  under 2 block noise units (2.05 MW). A weak result for the offset model here reflects D8's large
  noise unit and small transfers, and does not bear on the long, deep transfers of experiment 734.
- **The offset model was tuned on these years, in six ways.** Experiment 734 tunes on the years
  before 2024-01-01 (`TUNE_BLOCKS` is the number of blocks up to its scored years), and all 8 test
  months precede 2024-01-01, with 6 of them inside experiment 734's record from 2022. The months
  touch the priors, the time constant tau, the noise units, the block scales, and the shared-term
  loadings and switches, though none of the injected transfers has been seen. The step heads never
  trained on those months. This tuning favours the offset model, and a win by the offset model
  carries that caveat.
- **The step head's validation overlay was also the networks' early-stopping set.** The recalibration
  report states this caveat, and the head's frozen threshold is therefore tuned on rows that the
  networks also stopped on.
- **The two detectors read the same signal, because Design A could add the engine and solar
  arrays.** Experiment 732's test segments add simulated engine or solar output in most of the 64
  segments, which the head reads. Had Design A been unable to add the arrays, only the transfers
  would have distinguished the worlds. A secondary row scores the counted transfers that sit in
  segments with neither engine nor solar output, where both detectors read the same signal, and
  reports how many such transfers there are (the reviewer's count is 2 of the 21).
- **The counted contrast has a half-hour offset to get right.** On the valid half-hours of
  segments with neither engine nor solar output, D8's value in the offset model's panel equals the
  reading in points times 0.10379 MW only when the panel index is the bank half-hour plus 1, so a
  builder that writes the additions at the bank's half-hour puts every transfer 30 minutes early.
  The world builder uses the +1 index, G2 asserts it, and the known-answer test of the match runs
  through it.
- **Both detectors read the future.** See "Look-ahead".
- **Twenty-one transfers are few.** The counted contrast resolves a difference of shares only to
  about 0.25, so the plan expects a verdict of `not supported` or `inconclusive`. A cell with fewer
  than 20 transfers after the drops for gaps gives `inconclusive`.
- **The head's side was not blind.** The recalibration test rows (recall 0.490 and F1 0.531 for
  U-noBSP, at a tolerance of 1 half-hour on all 8 months) were seen before this plan, and the
  head's number below was computed before the plan was frozen. Only the offset model's side is
  predicted.

**The head's number was read before the plan was frozen, and the offset model had not been
decoded.** The numbers in this paragraph came from a read-only script on stored files, run before
any offset-model decode. It took the mean of the three seeds' `step_p` and `size` in
`test_transfer.npz` for U-noBSP (94,080 half-hours), applied the frozen isotonic map and F1-rule
threshold (0.5111), merged calls of the same sign less than 2 hours apart, matched each call to
an injected start edge with the match rule below, and counted the counted cell of 21 transfers.

| Quantity | Value |
|---|---|
| U-noBSP calls on all 8 months before merging | 304 (the recalibration report's number) |
| Calls after the merge, in all 8 months | 244 |
| Counted cell: transfers found, tolerance 6 hours | 12 of 21 (0.571) |
| Counted cell: transfers found, tolerance 1 half-hour (reference) | 11 of 21 |
| Transfers of 6 to 24 hours and 4 step noise units or more found | 17 of 29 |
| Transfers under 6 hours and 4 step noise units or more found | 29 of 54 |

The 12 found transfers split by month as 1 of 1, 4 of 5, 1 of 2, 0 of 3, 3 of 5, and 3 of 5, from
the oldest of the 6 months to the newest. Placing a merged call at its first half-hour or at its
largest step probability gives the same 12. The head's size output is signed (its sign agrees
with the injected sign at 0.95 of the injected edges, and the head's output is positive when D8
gains load), so the sign rule can be applied to both detectors. The files read had these SHA-256
digests, and G1 checks all six prediction files and the truth cache against them:

- `predictions/U-noBSP_unet_s0/test_transfer.npz` of experiment 732:
  `5d001cbd78ccbb524aa5797d8ddb3a1afe154b4f9544fe028ba0ec908a820669`
- `predictions/U-noBSP_unet_s1/test_transfer.npz`:
  `e967ca20a284223627d51e9d1ddf31b05ee20b16cdd68b4b5c92137765002795`
- `predictions/U-noBSP_unet_s2/test_transfer.npz`:
  `db376858834aa8299162cea197e087e246608d8a1975f77446129bdd1fb22d25`
- `predictions/Ustar_unet_s0/test_transfer.npz`, the reference arm U*:
  `022aa08770290805381fdf6015866c05096a408ee6b0fabed47e348d75f8a378`
- `predictions/Ustar_unet_s1/test_transfer.npz`:
  `b4dd121238a475263cd78f4bc6859a2a6773d6c90528ac5a1ff00bb8f24c202a`
- `predictions/Ustar_unet_s2/test_transfer.npz`:
  `9a7b9856b6dab568fa7c2d3ff95317ba8697dad4dbf4d25c8b16281a0c0673bc`
- experiment 734's `params.json`, the tuned values of the offset model (pinned in
  `e00736_common.py`, and checked against the `params.sha256` that experiment 734 stored beside it):
  `d7b3c57cb43c21fcb9569b66cadc5d4b50eacf1ad98395fc67a94cef37896120`
- the recalibration report's `frozen_thresholds.json` (written 16:36:17 UTC on 2026-11-27, before the
  report's `stage_b_test.py` first read the test overlay at 16:36:55):
  `49e81b90bc600dd27097efea831fbb209c5747a72d2419264d7758686325cc95`
- the diagnostics' `truth_cache.npz` (the injected edges' start, size, and duration):
  `33e935a3cb8f52e8670f9584067906c3470204364e22099768a46c9e91054cfe`

The recalibration report's folder, `common.py`, `stage_b_test.py`, and the JSON are not committed.
The experiment's own `e00736_heads.py` copies the match and isotonic code and reproduces this table
before the offset model is decoded, so the files named above become the experiment's own
read-only inputs.

**A step head's calls become event calls by the start edge alone, and found is the same for both
detectors.** A head call is a half-hour at which the mean of the three seeds' step probability,
after the frozen isotonic map, reaches the frozen F1-rule threshold. Calls of the same sign less than
2 hours apart merge into one call, and the sign of the call is the sign of the step head's size
output (positive when D8 gains load).

**An offset-model call is the start of an event in its event table.** The event table is the H-nbr
(or H-own) table with its sign, which is the sign of the event's `amp_mw` read as positive when D8
gains load (the code review checked the convention against experiment 734's table). A transfer is
found when a call of the same sign lies within 6 hours of its injected start, and one call matches
at most one transfer, with the larger transfer assigned first. The step head's predicted size and
the offset model's `amp_mw` do not enter the match. A match to an injected end edge counts as a
match for V1's false-call count, so a call near either injected edge is not false.

**The secondary rule also needs an end call.** The head needs an opposite-sign call within 6
hours of the injected end. The offset model needs an event whose end lies within 6 hours of the
injected end. With about 0.155 calls a day, a chance match to a given start is about 0.04.

**Both detectors are restricted to D8 and to scored base days for the false-call count.** The day
count for V1 is the same for both detectors: 185 base days per draw times 8 draws, which is 1,480
draw-days in the 6 months that lie inside experiment 653's record, for the head's calls and for the
offset model's D8 events alike. Real D8 events in the base days count as false calls for both
detectors. The false-call count compares each call with the injected edges of all 288 transfers,
including the transfers that the drops for gaps remove from the share found.

**Veto V1 excuses a call near an injected edge of any size, and both detectors are treated
alike.** A call within 6 hours of an injected edge of the right sign is not false, even when that
transfer moves only 0.3 MW, which neither detector can see. Both detectors gain the same excuse, so
each detector's false-call rate is lower than its rate on unexcused calls, and V1 tests the ratio
of the two rates.

## Predictions, gates, and the placebo

**The numbered predictions are in the front matter, and four gates must pass before any verdict
counts.**

- **G1, the head's rows reproduce.** Re-scoring the stored test predictions with the frozen
  thresholds, on all 8 months and with the recalibration report's one-half-hour match, gives
  U-noBSP's 304 calls and a recall of 0.490 on 4 step noise units and more. The same script then
  reproduces the 12 of 21 above under this plan's match rule.
- **G2, the injection is exact.** In every injected half-hour with a reading, the added megawatts
  equal the transfer's mean level, from `Transfer.mean_levels()` in 64-bit floats, times 0.10379
  MW per point, to a relative tolerance of 1e-6 (experiment 732's overlay arrays `t` and `mirror`
  are 32-bit floats, so their relative error is about 6e-8). D8 is an MVA Primary in experiment
  734 (`is_mva` is true), so the offset model's panel gets the addition `|p + add|` of
  `apply_additions`, and the head's reading gets experiment 732's refold with the reactive floor,
  which moves the reading by about the transfer times x/r rather than the transfer itself. G2
  checks each of the two against its own rule, and checks that every mirrored partner moves by minus
  the jitter factor times the transfer. In the transfers-only worlds, G2 also checks that D8 has no
  addition outside the injected half-hours. G2 also asserts the +1 index of the half-hour offset above.
- **G3, the decode reproduces.** The decode of the 15-Primary panel with no addition gives
  experiment 734's stored un-injected H-nbr event table
  (`uninjected/H-nbr_events.parquet`, 131 events at 12 Primaries, 28 of them at D8), exactly
  on the primary, start, and end of every event, and to a relative tolerance of 1e-6 on `amp_mw`
  and `confidence`, with an equal grade, compared over every Primary in the table. The decode also
  stops when the compiled Viterbi kernel of experiment 734 was not the one used.
- **G4, the overlays reproduce.** The regenerated test overlays reproduce experiment 732's digest
  `4c57546a9278dc0d0a6e3b2803d880b3205ae441dea7725681ab168000c30c6f` and its 384 transfers and 768
  edge rows over the 8 months.

**A placebo moves every injected start 7 days later, onto time where no transfer was injected, and
scores each detector's calls against the moved starts.** The placebo gives the share of starts that
each detector matches by chance, and P4 predicts it. Starts whose moved time lands inside a
transfer, within 6 hours of any real start or end edge of the segment (the edges of transfers that
the drops for gaps remove count too), off a scored half-hour, across a month end, or in a gap are
dropped from the placebo, and the page reports how many starts remain. The one-to-one match of the
placebo still draws on the moved starts of every scored transfer, dropped starts included, so a
dropped start can still take a call away from a kept start of the same sign.

## Synthetic world gallery

**The gallery shows only synthetic quantities.** D8's readings in MW are private, and no figure
shows a metered series, a date, or a real residual. The plan has 19 figures before the first
decode, which is 1 histogram figure, 9 injected additions, and 9 detector rows.

- **The injected transfers.** Histograms of the 288 durations and sizes, with the 24-hour and
  4-step-noise-unit lines, and the counts of the table above (1 figure).
- **Nine injected additions in MW, one per duration and shape cell.** Constant, demand-shaped, and
  solar-shaped additions (3 shapes) at about 6 hours, 30 hours, and 90 hours (3 durations), each
  drawn alone with its mirror at the partner, on an axis of hours from the start with no date (9
  figures).
- **Nine detector rows.** For the same nine transfers, the head's step probability and the
  offset model's posterior of the switched power, in noise units, each against the injected truth.
  The decoded offset is shown on an axis shifted to zero, so no D8 level is visible (9 figures).

The smoke check on the data files asserts that the injected addition is nonzero in every
half-hour of every transfer (the failure that silenced one earlier run's solar output in 37.5% of
its test rows, in experiment 729), that every read partner carries a nonzero mirror, and that all
288 transfers are present in the 8 worlds, and that in every segment with an engine or solar
array the array is nonzero at the half-hours where it is written into D8's addition (the check
reports a segment's index and never a value). The write-up repeats the gallery with the results
added.

## Look-ahead

**Neither detector forecasts, and both read the future of the event, so the comparison is a
hindsight comparison.** The step head reads a 48-hour window that holds the 24 hours before and the
24 hours after the half-hour it scores: D8's folded reading, the sky inputs of the simulated
segment, the BSP meter, and the three neighbours' readings. The offset model reads the whole record
from 2022 to mid-2026. Its forward-backward decode uses blocks before and after every block, and
its residual uses ERA5 (the European Centre for Medium-Range Weather Forecasts' reanalysis of past
weather) and CAMS (the Copernicus Atmosphere Monitoring Service's estimates of past solar
radiation) and a demand forecast that uses no lagged power, as experiment 734 defines. No input of
either detector is a weather forecast at a forecast origin. The result bears on whether the offset
model can label transfers in a historical record, and not on whether either detector could run in
real time.

## Stages and compute

**The compute slot is 1 core and about 2.4 CPU-hours, and the estimate is uncertain.** Experiment
734's time log gives a median of 27.6 wall-minutes for each of its 40 `score-default` jobs and 4.1
wall-minutes for an uninjected full-record decode of one arm. The plan's 15 CPU-minutes per world
with H-own and H-nbr is an estimate from those figures and from about 12 CPU-minutes for each world's
H arms, and is not a measurement. If one decode of one arm takes 27.6 minutes, the 16 decodes cost
about 7.4 CPU-hours (16 decodes of 27.6 minutes), which is the worst case. The manager set the
stop at 8 CPU-hours, so the worst case stays below it. The first world's measured time replaces
the estimate, and the plan asks the manager before passing the 8 CPU-hour stop. The runner also
stops when the finished decode tasks project more than 8 CPU-hours.

| Stage | Work | Expected CPU-hours |
|---|---|---|
| 0 | Build the 8 worlds (draw k of the 6 months, merged, no month overlaps another) and run G2 | 0.1 |
| 1 | Read the step heads' stored test predictions, apply the frozen maps, and run G1 | under 0.1 |
| 2 | G3, then decode 8 worlds with H-own and H-nbr | about 2 |
| 3 | Match, placebo, intervals, and figures | 0.2 |

**The stages total about 2.4 CPU-hours.** Peak memory is about 1.5 GB (experiment 734's measured
figure per worker, and the forward messages of one Primary in 32-bit floats take 340 MB), and the
scope cap is 4 GB. Stage 1 loads one arm's 94,080 half-hours (about 2,000 days of 48 half-hours) in
32-bit floats, and the load takes under 1 GB. Each stage writes one file per world, atomically,
with a time log.

## Run order and reviews

**The experiment runs in this order, and no stage runs before its review marker exists.**

1. A fresh Sonnet agent that did not write the plan reviews the plan, and a fresh Sonnet agent that
   did not write the code reviews the code (leakage, the match rule, the +1 index, the sign
   convention, and a match with the recipe the code copies). A second fresh Sonnet agent reviews
   the fixed code if the first review flags many problems.
2. After the code review, a person or the manager runs `review-mark`, which writes the hash of the
   reviewed code (the scripts, `run.sh`, and `run_workers.sh`) into `review_stage1.ok`. Every stage
   from `world` onward refuses to start unless the marker holds the current code hash, and
   `freeze-code` refuses unless the code hash equals the marker's content. Stage 0 needs no marker.
   A done marker of a decode task holds the code hash too, and a re-run ignores a marker of other
   code.
3. A real-environment smoke test runs stage 0 and one world's decode into a scratch folder
   (`.claude/scratch/`) that `E00736_OUT` names, with nothing written to the experiment's data
   folder (the runner refuses `E00736_OUT` unless `E00736_SCRATCH=1` is set or the path has a
   `scratch` component, prints the output folder at the start of every stage, and stores it in
   `scores.json`). The smoke test has no validation mode, because this design has test rows only,
   and none of its numbers is used to tune anything: every parameter is frozen before it runs.
   The test reads the real input files and takes the world's call tables through the scoring
   code.
4. The runs happen inside `systemd-run --user --scope` with a memory cap of 4 GB and 1 thread for
   each numerical library, after the ML experiment manager approves the workstation run.
5. A Sonnet agent checks the write-up against the printed report, and a second Sonnet agent runs the
   prose review.

## Standing practices, advice, and scenarios

- **Followed:** cleaned data with the `substation_zero` half-hours dropped (the count is reported);
  stored rows reused as benchmark and as the head's outputs; a synthetic world gallery and a smoke
  check; a per-phase time log; no stage past its review gate; the hash of every edited reviewed
  script recorded; mid-day onsets in the known-answer test of the match (a transfer starting at
  13:20 and one at 00:10, built through the +1 index); a placebo beside the contrast.
- **Not applicable:** monthly refits, model saving, training-loss plots, and screening folds,
  because nothing is trained; tail coverage of a forecast, because no forecast is scored; the D15
  and D18 cut, because D8 is not one of them.
- **The four load-shift scenarios do not apply,** because no forecast is scored. The experiment
  scores detection of injected transfers, and the share found is the headline.
- **Advice followed:** use U-noBSP, the head without the BSP input, and ask for neighbour mirrors
  only where they can be read (experiment 732's advice items on the BSP input and the neighbour
  mirror); compare with a stored reference that the same rows produced and read a cell only with 20
  events or more (experiment 734's rule, from experiment 665; the counted cell holds 21, and fewer
  than 20 after drops gives `inconclusive`); report shares found and shares paired separately
  (experiment 653, hence the secondary rule); expect a test of one extra input at one Primary to
  resolve only a coarse gain, and plan the threshold above that (experiment 733's advice on the
  half-width of its confidence interval for a change at one Primary, which this page's C1 is not).
- **Advice departed from:** experiment 734's tolerance of 12 hours and its intersection-over-union
  rule, which become 6 hours and a start-only match here, because the head has no intervals. This
  experiment's reading at a tolerance of 12 hours is reported as a post hoc row, and labelled so.

## Open questions before the first run

- **D8's residual series.** Whether the real D8 series of experiment 653 on the 6 months equals the
  folded reading that experiment 732 used (the code review checked that both are MVA and equal to
  rounding).
- **Whether the world builder can add the engine and solar arrays.** The code review settled it
  before any decode, and the "Every handicap" list was updated to match.
- **Gaps in the residual.** Transfers whose start block has no residual at D8 are dropped from both
  detectors, and the count is reported, together with the discordant counts per month.

## Method

### Architecture

- The step head is the U-net of experiment 732 with three seeds, read from stored predictions
  (`predictions/<arm>_s<k>/test_transfer.npz`), with no network run and none retrained.
- The offset model is experiment 734's hidden Markov model on a grid of 41 offset levels, with
  slow states, a shared term across Primaries, and (H-nbr) neighbour evidence. Tuned values come
  from experiment 734's `params.json`, read unchanged.
- The residual is observed cleaned power minus the forecast of the lag-free convex demand model of
  experiment 130, as experiment 734 builds it; blocks are 6-hour means.

### Training regime

- Nothing is trained or tuned. The injected transfers are those of experiment 732's test bank,
  re-drawn by its own overlay code (the same seed and segments), so the sizes, starts, shapes, and
  partners are experiment 732's.
- Each of the 8 worlds merges one draw (draw 0 to draw 7) of every one of the 6 months. A world is
  D8's power plus the segment's synthetic engine and solar output plus the draw's additions, with
  the partners' power plus the mirrors, all written at the +1 half-hour index.

### Measurement

- The scored rows are the 288 transfers of the 6 months, and the counted cell is the 21 of 24 to 96
  hours and 4 step noise units or more.
- The metric is the share of injected starts matched, by the rule above. The secondary rows are the
  other duration bands, the both-ends rule, the counted transfers in segments with neither engine nor
  solar output, the shifted-start placebo, and false edge calls per 100 scored base days.
- The page reports the table of paired counts (found by both, only by U-noBSP, only by H-nbr, by
  neither), the exact sign test on the discordant transfers, and the discordant counts per month.
  Intervals resample the 8 draws and the 6 months, 10,000 draws each. Every number before the final
  scoring is interim.
- One step noise unit is 3.42 points (0.355 MW), so 4 step noise units are 13.67 points, or 1.42
  MW. One point is one unit of the folded reading (0.10379 MW). One block noise unit at D8 is 1.02
  MW. The jitter factor is the factor by which a partner's mirror differs from the transfer
  (1.0 for an exact mirror). The page writes "step noise unit" and "block noise unit" in full, because
  the two units differ.

## Files

- `ml-experiments/00736-switching-offset-model-vs-untangler-on-d8/`:
  `e00736_world.py` re-draws experiment 732's test overlay and builds the 8 worlds;
  `e00736_heads.py` copies the match and isotonic code of the recalibration report, reads the stored
  head predictions, applies the frozen maps, and reproduces the head's table above;
  `e00736_decode.py` decodes the worlds with experiment 734's code; `e00736_score.py` matches,
  intervals, placebo, and figures.
- Read only: `ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/e00732_overlays.py`
  and its bank; experiment 732's `data/ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/predictions/`
  (`step_p`, `size`, `row_seg`, `row_date`, `row_hh` of `test_transfer.npz`) and `limits.json`
  (noise units); the recalibration report's `frozen_thresholds.json`, `common.py`, and
  `stage_b_test.py` under `data/ml-experiments/briefs/msa_architecture/e_untangler_recal/`, and
  the diagnostics' `truth_cache.npz` under `.../e_untangler_diag/`; experiment 734's code,
  `params.json`, and `uninjected/` tables; experiment 653's `noise_units.json`; experiment 660's
  generator.
- Written: `data/ml-experiments/00736-switching-offset-model-vs-untangler-on-d8/` (never committed).
