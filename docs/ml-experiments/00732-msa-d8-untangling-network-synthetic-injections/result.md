---
id: "00732"
slug: msa-d8-untangling-network-synthetic-injections
title: >-
  Can one neural network split D8's unsigned reading into the sign of the power, the output of
  unmetered solar panels in megawatts, and the steps of load transfers, without giving a worse
  probability of reverse flow than the sign-only network of experiment 730?
theme: solar-disaggregation
also_relevant_to:
  - gross-demand-simulation
  - switching-labels-in-the-forecast
status: finished
kind: other
size: >-
  Anything else. (1) The experiment stores 24 fitted neural networks, three synthetic overlay
  tables, D8's neighbour input table, two fitted rules for load-transfer steps, per-half-hour
  predictions, and score tables, all private; (2) more than one design defensibly tests the
  hypothesis (with or without the contrastive terms, the neighbour inputs, or the BSP input, with
  shared or stopped gradients, and on the U-net or the bidirectional LSTM), so each is an arm with
  a pre-registered role; (3) the experiment reads experiment 730's banks, margin, limits, and
  stored predictions, and the stored power series of D8's three electrical neighbours, and imports
  the code of experiments 722, 729, and 730, whose callers cannot be named without searching the
  repository; (4) the experiment touches no serving path or degradation rule; (5) the experiment
  changes no contract. Reviews, in order: this Opus plan, a Sonnet plan review, Sonnet code, a
  fresh Opus code review before any first run, a real-file smoke test, then the results, write-up,
  and prose reviews. An Opus review of this revised plan comes before any further code.
hypothesis: >-
  On held-out synthetic days built from D8's own pre-engine readings, one U-net that reads D8's
  reading, the sky, the BSP steps, and the readings of D8's three electrical neighbours, and that
  learns three tasks (the probability that D8's signed power is negative, the output of the
  injected solar panels, and the load-transfer steps), gives a Brier score of the probability of
  reverse flow no worse than experiment 730's sign-only arm S by more than experiment 730's margin,
  and raises no more false alarms on import half-hours than the harm checks allow. The solar
  output and the transfer steps are judged by their own reading words, and do not set the verdict.
prediction: >-
  Written before any fit or score exists, in four parts, one for each question. (a) The sign: row 5
  (`inconclusive`) is the likeliest verdict, row 4 (`supported`) the next, and row 3 less likely. In
  experiment 730 every arm with an extra output had a worse point estimate than S, and the margin m
  (0.00122) is about the half-width of experiment 730's own C1 interval (0.00117), so
  non-inferiority needs a point estimate of C1 at or below about zero. C1, U* minus experiment 730's
  S in the Brier score of P(X<0), has a point estimate between -0.0003 and +0.0008, and C6 (U* minus
  S+) and R4 (S+ minus S) each have an interval that includes zero, so the extra tasks do not
  detectably improve the sign. U* passes H1 pooled and at 3 MVA and above, and is not detectably
  worse than S in the band of 1 to 3 MVA. The prediction of the sign fails if C1's lower end is
  above zero. (b) The solar output: the solar word is "reasonable". The recovery lies at 0.93 to
  1.03 of each injected amount, and at 0.8 to 1.1 on reverse-flow half-hours; the error on
  solar-world sun-up half-hours is under half the mean solar target; U* beats the sky-only rule
  B-PV0; and C4 is a gain of at least 2 points of capacity. The likeliest miss is the recovery on
  reverse-flow half-hours. (c) Load transfers: the transfer word is "works". C5 is a gain over the
  rule B-TN, and U* reaches an F1 of at least 0.6 on steps of 5 points and more. C7 (the neighbour
  inputs) is a gain in the step Brier score, and the whole of that gain sits on transfers with a
  mirror at a neighbour that the neural network reads. Steps under about 4 noise units are not
  detectable with or without the neighbours. Recall on solar-shaped transfers is within 0.1 of
  recall on constant transfers, because both have flat edges at any hour. The neighbour inputs make
  no detectable difference to the sign on plain rows (R13). (d) The BSP steps: R19 has an interval
  that includes zero and a point at or above zero, so the BSP word is "no detectable difference",
  and "hurts" is the next likeliest word. On experiment 729's banks the BSP steps added to sunshine
  lost 0.149 points, experiment 124 found that a BSP term worsened clean forecasts at every weight,
  and experiment 722 found that the BSP moves 2.53 times as much as the Primary in lit hours. R19's
  half-width is about 0.003 to 0.004, an estimate, so R19 cannot detect a small effect.
decision_rule: >-
  Every bar and constant is a named `Final` of `e00732_core.py`, fixed before any fit or score
  exists. Seven contrasts are counted at 99.286% (Bonferroni, 1 - 0.05 / 7), from experiment 729's
  two-way bootstrap with Kish widening; the section "Contrasts" defines them. The margin m is
  experiment 730's, 0.00122. Non-inferiority (NI) holds when C1's upper end is at most +m. The harm
  limits are formulas evaluated on the test bank's truth before any fit. Verdict, first row that
  matches: (0) stopped; (1) `invalid`; (2) `failed`; (3) `not supported`, a harm check fails; (4)
  `supported`, NI; (5) `inconclusive`, C1's upper end above +m and its lower end at or below zero;
  (6) `not supported`, every other case. Every row from 3 to 6 prints C6 and R4 beside it. Four
  reading words sit beside the verdict and do not change it: the solar word (from eight bars), the
  transfer word (from C5 and an F1 bar), the neighbour word (from C7), and the BSP word (from the
  report-only reading R19, on the test months that the BSP meter covers). The harm reading HB (U*
  minus S in the band of 1 to 3 MVA) decides row 3 through a one-sided lower bound at 99.286% and
  sits outside the Bonferroni count. The section "Verdict table" gives every row in
  full. The experiment ports nothing, so no smallest gain applies beyond m.
benchmark: >-
  Experiment 730's sign-only U-net S (stored folders `S_unet_s0` to `S_unet_s2`) for the sign, and
  its bidirectional LSTM S (`S_bilstm_s0` to `S_bilstm_s2`) for the architecture reading; P0 (the
  reading read as signed import) for the harm checks; for the solar output, B-PV2 (experiment 730's
  chosen demand-output arm SD*, stored as `SD-R_unet_s0` to `SD-R_unet_s2`, with its demand output
  minus the signed reading implied by its P(X<0), clipped at zero and set to zero at night) and
  B-PV1 (the same with D8's demand climatology and S's P(X<0)); for the transfer steps, B-TN and
  B-T, logistic regressions on the steps of the reading and of the synthetic BSP meter in noise
  units, fitted on the validation overlay; for the solar output also B-PV0, a rule fitted on the
  validation bank that reads the sky alone. B-TN adds the neighbours' steps, sign-aligned with D8's
  step, and the count of neighbours that step the same way as D8, the feature of experiment 70.
row_set_id: >-
  e00730_synthetic_d8_v1 for the sign and the solar output: experiment 730's 94,080 scored
  half-hours (8 held-out calendar months of D8's pre-engine base days, 8 synthetic segment draws
  each, the middle 24 hours of a 48-hour window). e00732_overlay_d8_v1 for the transfer steps and
  the injection recovery: the same windows carrying this experiment's fixed test overlays. The BSP
  reading R19 reads the rows of e00730_synthetic_d8_v1 that fall in the two test months that the BSP
  meter covers. Real checks read the same months without injection, the metered engine record's
  days, the days of logged closures at D8, and D8's eras up to 2026-06-30. No valid time on or after
  2026-07-01 is read. No forecast is made.
metric: >-
  The Brier score and the log loss of P(X<0) on all, import, and reverse-flow half-hours and by
  reading band; the harm means of H1 pooled and by band; the mean absolute error of the solar output
  in points (and MW) on sun-up half-hours, by world type and source; the recovery ratio and the mean
  and median absolute injection error at injected amounts of 5, 20, and 50 points; the night mean
  and night share; the shares of monotonicity violations and of responsive half-hours; the Brier
  score, precision, recall, and F1 of the transfer step, by band of size in noise units, by
  duration, by shape, and by partner, and the error of the step size; the normalised mean absolute
  error of the median of the signed distribution, for continuity.
verdict: supported
best_arm: >-
  S+ has the lowest Brier score of the probability of reverse flow (0.00721) and U* (0.00787) is the
  best arm that also gives a solar output and transfer steps. U* is no worse than experiment 730's
  S (0.00853) by more than the margin: C1 is -0.00066 (-0.00223 to +0.00071), the margin m being
  0.00122. All numbers are interim.
headline: >-
  A neural network that also learned the solar output and the load-transfer steps gave a
  probability of reverse flow no worse than experiment 730's sign-only network (C1 -0.00066,
  interval
  -0.00223 to +0.00071, margin 0.00122), so the verdict is row 4, `supported` (non-inferior). The
  extra tasks did not improve the sign: U* minus the sign-only network S+ trained on the same
  windows (C6) is +0.00065 (+0.00004 to +0.00137). The solar output on synthetic days beats every
  rule (mean absolute error 3.31 points of capacity against 6.93 for the best rule) but misses two
  of eight bars, so its word is "not shown". The transfer head finds 7.7% of the simulated steps of
  5 points or more. The neighbours (C7) and the BSP steps (R19) made no detectable difference
  overall. All numbers are interim.
lessons: >-
  The plan named row 5 as the likeliest verdict and row 4 as the next, and row 4 came out. A
  sign-only network on the same windows (S+) scored 0.00131 lower than experiment 730's S, so the
  apparent gain over S comes from this experiment's training windows and inputs and not from the
  extra heads. The extra heads cost the sign about 0.0007, whether or not their gradients reach
  the shared backbone (U-SG against U*). The contrastive terms make the solar head obey the
  physical rules (monotonicity violations fall from 33% to 0.7%) and lower the injection error,
  and a heavier plain solar loss lowers the solar error more. Transfer detection is weak: a recall
  of 7.7%, and 22% for the largest steps. The neighbours help only where the mirror is exact, and
  the BSP input worsens detection. On real days the solar output follows the sky but does not
  stay at zero before the solar farm existed. All numbers are interim.
take_home_message: >-
  A single neural network that also estimates solar output and spots load transfers gives a
  probability of reverse flow that is as good as the simple sign-only network, but not better, so
  the extra tasks should not be added to improve the sign. On synthetic days it estimates the solar
  output much better than the rules, but it finds only about 8% of the simulated load transfers,
  and on real days it still outputs solar where there was no solar farm. All numbers are interim.
advice:
  - >-
    Use the sign-only U-net trained on this experiment's windows and inputs (S+) as the reference
    for
    the probability of reverse flow at D8, because it scored lower than every arm with extra heads
    (C6, R4).
  - >-
    Do not add solar, step, or size heads to a sign neural network in order to improve the sign,
    because the
    extra heads cost about 0.0007 of Brier score whether or not their gradients reach the shared
    backbone (C6, R1, R2).
  - >-
    Keep the injection, night, and monotone terms in any network that outputs solar power, because
    they cut monotonicity violations from 33% to 0.7% and lower the injection error (C3, R9).
  - >-
    Stop a fit with a solar head on a score that includes the solar error, because stopping on the
    sign left the solar error higher than later epochs in two arms (loss curves, figure 8).
  - >-
    Test a solar head on real days where no solar farm exists before trusting it, because the head
    passed all synthetic bars but outputs solar before the solar farm existed (P2, P4).
  - >-
    Redesign the transfer detector before relying on it, because the head finds 7.7% of simulated
    steps of 5 points or more, and drop the BSP input from it, because the arm without that input
    finds twice as many steps (R22, step table).
  - >-
    Feed neighbour readings to a transfer detector only where an exact mirror can be read, because
    the gain from the neighbours sits on exact mirrors alone (C7, R16).
production_status: not-for-production
adopted_in: none
---

**We asked whether one neural network can take a day of D8's meter readings and give three answers
at each half-hour: the probability that power flows out of D8 towards the grid, the output of the
solar panels below D8, and whether a block of customers has been switched to or from another
substation.** D8 is a primary substation (a "Primary") whose meter reports apparent power, a
magnitude with no sign. A gas engine and a solar farm connect below D8, so power sometimes flows in
reverse, and the meter shows that export as a positive reading. No meter records the three answers
on a real day, so we trained the neural network on synthetic days with a known answer: D8's real
readings from before the engine was connected, plus a simulated engine, simulated solar output, and
simulated switched blocks of customers (a "load transfer"), folded into a magnitude the way the
meter does. We tested on synthetic days that training never saw, and ran checks of physical sense on
real days.

**The neural network also read the meters of D8's three neighbouring Primaries (D3, D5, and D14) and
the meter of the bulk supply point above D8, because a load transfer shows as a mirror image at a
neighbour.** When customers move from a neighbouring Primary to D8, the demand at D8 steps up and
the demand at the neighbour steps down by the same amount, and the simulated transfers change the
neighbours' readings in the same way.

**The verdict rests on one question: does learning the extra answers make the probability of reverse
flow worse?** Experiment 730 (a sign-only neural network against a neural network that also
predicted D8's demand) found that a demand output made that probability slightly worse. This
experiment compares the new neural network, called U\*, with experiment 730's best arm, the
sign-only neural network S, on the same 94,080 test half-hours. The page gives the result first,
then the synthetic world, the gallery of drawings that checks the simulators, the method, and the
files.

## Result

*Every number on this page comes from the first and only scoring of the test half-hours, and is
interim until the reviews of the results and the prose are complete. No number is read from memory:
each comes from `scores/verdict.json`, `scores/readings.json`, `scores/real.json`,
`scores/timing.json`, `scores/tables.json`, or a fit record, all under
`data/ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/`.*

### The four questions in plain terms

**(a) The extra training tasks did not improve the probability of reverse flow, and the sign-only
neural network does that one job at least as well.** The measure is the Brier score of the
probability of reverse flow (the mean squared difference between the probability and what happened,
0 or 1; lower is better) over the 94,080 test half-hours. U\* scored 0.00787, and experiment 730's
sign-only neural network S scored 0.00853. The difference (C1) is -0.00066, with an interval of
-0.00223 to +0.00071, so there is no detectable difference. The comparison that isolates the extra
tasks uses the arm S+, a sign-only neural network trained on the same windows and inputs as U\*. S+
scored 0.00721, and U\* minus S+ (C6) is +0.00065 (+0.00004 to +0.00137), so the extra tasks made
the probability slightly worse, by a margin that just excludes zero. The apparent gain over S comes
from this experiment's training windows and inputs, not from the extra tasks: S+ minus S (R4) is
-0.00131 (-0.00299 to +0.00002).

**(b) The neural network separates solar output from the reading on synthetic days to a useful
degree, but it fails the pre-registered rule for the word "reasonable".** On sun-up half-hours its
mean absolute error is 3.31 points of D8's capacity, against 6.93 for the best rule built from
experiment 730's output (C4: -3.62, interval -5.58 to -2.14) and 16.80 for a rule that reads the sky
alone. Six of the eight pre-registered solar bars pass. The two that fail are the recovery of a
5-point injection (0.83 of the injected amount, against a bar of 0.9 to 1.1) and the output at night
(a mean of 0.065 points against 0.05, and 1.02% of night half-hours above 0.5 points against 0.5%).
The solar word is therefore "not shown", because the rule withholds "partly" and "reasonable" when
either recovery bar fails. On real days the sky check passes and the night check passes, but the two
checks that ask for no solar output where no solar farm exists, and for no solar output that follows
the engine, fail.

**(c) The neural network barely detects switching events, even on synthetic days where it knows
every step.** At a threshold of 0.5 it flagged 70 half-hours, 66% of them right, and found 7.7% of
the 594 simulated transfer steps of 5 points or more (F1 0.139, against a bar of 0.6). It beats the
logistic rule that reads the neighbours (B-TN) in Brier score (C5: -0.00059, interval -0.00091 to
-0.00032), so the transfer word is "works only in Brier score". Steps of 8 noise units and more
(about 27 points of capacity and above) are found 22% of the time, and steps under 4 noise units
essentially never. The neighbours' readings made no detectable difference overall (C7: -0.00007,
interval -0.00024 to +0.00009). The only split with a gain is a transfer whose mirror at a neighbour
the neural network reads exactly (R16: -0.00063, interval -0.00129 to -0.00015).

**(d) Adding the steps of the bulk supply point's meter did not help the direction of flow.** The
Brier score of U\* minus that of U-noBSP (the same neural network without the BSP input), on the two
test months that the BSP meter covers (23,808 half-hours, 462 of them reverse flow), is -0.000003
(-0.00162 to +0.00210), so the BSP word is "no detectable difference". The interval is wide, so the
result does not rule out a small effect. The BSP input did detectably worsen the step detection in
two partner cases (R22).

**Against experiment 730, the harm that experiment found is smaller here and does not reach the
verdict.** In experiment 730 the arm with a demand output scored 0.00105 worse than S (interval
-0.00011 to +0.00222) and failed the false-alarm check H1. In this experiment the extra tasks score
0.00065 worse than the same-window sign-only neural network S+ (C6), a smaller difference in the
same direction, and U\* passes H1 pooled and at 3 MVA and above. In the band of 1 to 3 MVA the mean
probability of reverse flow on import half-hours is 0.0894 for U\*, 0.0828 for S, and 0.0717 for S+,
against a limit of 0.083. The verdict calls this band "at the calibrated boundary", because the
lower end of the harm reading HB (+0.00657, interval -0.01052 to +0.02846) is not above zero.
Experiment 730's verdict was row 3 (`not supported`); this experiment's is row 4.

### The verdict is `supported` (verdict row 4): U\* is no worse than S by more than the margin

**C1's upper end (+0.00071) is below the margin m of 0.00122, so the sign is non-inferior to
experiment 730's S.** The margin is experiment 730's: the larger of 0.02 times the share of
reverse-flow half-hours (0.0364) and twice the standard deviation over seeds of S's validation Brier
score. The verdict word "supported" means non-inferiority, and says nothing about an improvement:
C1's interval includes zero, so the two neural networks cannot be told apart. U\* passes the harm
checks H1 pooled (a mean probability of 0.00473 on import half-hours of at least 1 MVA, against a
limit of 0.0071) and at 3 MVA and above (0.00150 against 0.0042), and passes H2 on real held-out
days before the engine was connected (a mean of 0.00012 and no half-hour above 0.5).

**The plan predicted row 5 (`inconclusive`) as the likeliest verdict and row 4 as the next; the
result is row 4.** The plan predicted C1 between -0.0003 and +0.0008, and C1 is -0.00066, a little
better than the predicted range. It predicted that C6 and R4 would each have an interval that
includes zero. R4's interval includes zero, only just (upper end +0.00002), and C6's does not (lower
end +0.00004), so that part of the prediction failed narrowly. The plan's other predictions fared as
follows:

- **The solar word was predicted "reasonable"; it is "not shown".** The recovery lies at 0.93 and
  0.96 of the injected amount at 20 and 50 points, inside the predicted 0.93 to 1.03, and at 0.83 at
  5 points, outside it. The recovery on reverse-flow half-hours is 0.997, inside the predicted 0.8
  to 1.1, so the likeliest miss named by the plan did not occur.
- **The absolute-error, sky-alone, and demand-output bars passed as predicted.** U\*'s error on
  sun-up half-hours of solar worlds is 4.89 points against a bar of 15.5, it beats the sky-only rule
  B-PV0 (R23: -14.09), and C4 is a gain of 3.62 points against the predicted 2.
- **The transfer word was predicted "works"; it is "works only in Brier score".** C5 is a gain, as
  predicted, but the F1 is 0.139 against a predicted 0.6 or more.
- **C7 was predicted to be a gain; it is not detectable.** The plan also predicted that steps under
  about 4 noise units are undetectable, which holds, and that steps of 4 noise units and more are
  detectable, which does not hold: no size band reaches both a recall and a precision of 0.5.
- **The neighbours' effect on the sign on plain rows (R13) was predicted to be undetectable, and it
  is:** -0.00088 (-0.00211 to +0.00012). Recall on solar-shaped transfers is 0.035 above recall on
  constant transfers, inside the predicted 0.1.
- **The BSP word was predicted "no detectable difference", and it is.**

### Every arm's Brier score and harm check

**S+, the sign-only neural network trained on this experiment's windows, has the lowest Brier score
of every arm on all rows.** The table gives each arm's Brier score on all half-hours, on import
half-hours, on reverse-flow half-hours, and on half-hours with a reading of 1 to 3 MVA. H1 is the
mean probability of reverse flow on import half-hours of at least 1 MVA, pooled and by band, and
passes at or below its limit. Detection is the mean probability on reverse-flow half-hours of at
least 1 MVA, and passes at or above 0.75. P0 reads the reading as signed import, so its probability
of reverse flow is 0 and its Brier score is the share of reverse-flow half-hours. The seeds 0 to 2
are averaged.

| Arm | Brier, all | Brier, import | Brier, reverse flow | Brier, 1 to 3 MVA | H1 pooled (limit 0.0071) | H1, 1 to 3 MVA (limit 0.083) | H1, 3 MVA and above (limit 0.0042) | Detection (bar 0.75) |
|---|---|---|---|---|---|---|---|---|
| P0 | 0.03643 | 0.00000 | 1.00000 | 0.2506 | n/a | n/a | n/a | n/a |
| S (experiment 730) | 0.00853 | 0.00370 | 0.1361 | 0.0634 | 0.00453 | 0.0828 | 0.00154 | 0.842 |
| S+ | 0.00721 | 0.00332 | 0.1102 | 0.0501 | 0.00345 | 0.0717 | 0.00085 | 0.866 |
| U\* | 0.00787 | 0.00342 | 0.1256 | 0.0555 | 0.00473 | 0.0894 | 0.00150 | 0.845 |
| U-NC | 0.00847 | 0.00408 | 0.1245 | 0.0654 | 0.00643 | 0.1169 | 0.00222 | 0.839 |
| U-NCw | 0.00832 | 0.00413 | 0.1192 | 0.0592 | 0.00590 | 0.1079 | 0.00201 | 0.847 |
| U-SG | 0.00778 | 0.00358 | 0.1190 | 0.0568 | 0.00455 | 0.0889 | 0.00134 | 0.857 |
| U-N | 0.00875 | 0.00336 | 0.1513 | 0.0662 | 0.00547 | 0.1004 | 0.00185 | 0.805 |
| U-noBSP | 0.00743 | 0.00378 | 0.1041 | 0.0512 | 0.00451 | 0.0887 | 0.00130 | 0.876 |
| S-L (experiment 730) | 0.00909 | 0.00403 | 0.1430 | 0.0700 | 0.00527 | 0.0992 | 0.00169 | 0.814 |
| U\*-L | 0.00818 | 0.00350 | 0.1321 | 0.0626 | 0.00487 | 0.0970 | 0.00136 | 0.832 |

![Brier score of each arm by group of half-hours](fig01_brier_by_group.svg)

**Figure 1 shows the arms separating only on reverse-flow half-hours and on the band of 1 to 3
MVA.** Every arm sits near zero on import half-hours and at 3 MVA and above. On reverse-flow
half-hours the scores run from 0.104 (U-noBSP) to 0.151 (U-N).

**The sign calibration bar passes for every arm.** The reliability term of each arm is at most
0.00010, against a bar of 0.00364. The largest gap between a bin's mean probability and its observed
share is 0.091 for U\* and 0.086 for S+, against a bar of 0.1 (figure 2).

![Reliability of the probability of reverse flow](fig04_reliability.svg)

**Figure 2 shows every arm close to the diagonal.** The largest departures are in the bins between
0.05 and 0.2, where some arms lie above the diagonal and others below it.

### The extra tasks do not improve the sign, and sharing gradients is not the cause

**The seven counted contrasts are read at 99.286% (Bonferroni, 1 - 0.05 / 7), and a negative
difference favours the first arm.** The intervals come from the two-way bootstrap of [experiment
729](../00729-msa-d8-neural-untangler-synthetic-known-answers/result.md) (the first neural network
trained on synthetic days with a known answer) with 20,000 resamples, over blocks of 7 consecutive
base dates and over segment draws. The Kish effective numbers of blocks and segments (36.9 and 64.0
for the sign contrasts) are above 20, so no interval is widened.

| Contrast | What is compared | Difference and interval | Reading |
|---|---|---|---|
| C1 | U\* minus S (730), Brier score of P(X<0) | -0.00066 (-0.00223 to +0.00071) | no detectable difference |
| C2 | U\* minus U-NC, solar mean absolute error (points) | -0.070 (-0.313 to +0.160) | no detectable difference |
| C3 | U\* minus U-NC, mean absolute injection error (points) | -0.487 (-0.619 to -0.372) | gain |
| C4 | U\* minus B-PV2, solar mean absolute error (points) | -3.618 (-5.580 to -2.143) | gain |
| C5 | U\* minus B-TN, Brier score of the step event | -0.00059 (-0.00091 to -0.00032) | gain |
| C6 | U\* minus S+, Brier score of P(X<0) | +0.00065 (+0.00004 to +0.00137) | loss |
| C7 | U\* minus U-N, Brier score of the step event | -0.00007 (-0.00024 to +0.00009) | no detectable difference |

![The sign contrasts](contrasts_sign.svg)

**Figure 3 shows C1 straddling zero inside the margin, and C6 just to the right of zero.** The sign
contrasts C1 and C6 and the report-only readings R1, R2, R4, R5, R13, R19, and R21 share one axis,
in Brier score.

**The stop-gradient arm U-SG shows that shared gradients are not what costs the sign.** In U-SG the
solar, step, and size heads read the backbone through a stop-gradient, so their losses train only
their own linear layers. U-SG minus S+ (R1) is +0.00057 (+0.000002 to +0.00129), and U\* minus U-SG
(R2) is +0.00008 (-0.00068 to +0.00071). U-SG and U\* therefore cost the sign about the same, and
the cost is not shown to depend on the shared gradients. The solar head of U-SG learned almost
nothing (a mean absolute error of 13.46 points, and a recovery of 0.20 to 0.27), so the sign cost of
U-SG is not the cost of a working solar head.

**The contrastive terms improve the sign a little.** U\* minus U-NC (R5, the same neural network
without the injection, night, and monotone terms) is -0.00060 (-0.00126 to -0.00006). Without those
terms the extra heads cost the sign more: U-NC scores 0.00847, against 0.00721 for S+. The
report-only readings are not counted in the Bonferroni count, and each is read at the same level.

**The bidirectional LSTM gives the same picture as the U-net.** U\*-L minus S-L (R3, experiment
730's bidirectional LSTM) is -0.00091 (-0.00226 to +0.00017), and U\*-L scores 0.00818 against
0.00787 for U\*.

**The log loss points the same way as the Brier score for C1, C6, and R4.** The seed-averaged log
loss on all half-hours is 0.02806 for S, 0.02337 for S+, and 0.02578 for U\*, so U\* minus S is
-0.00228, U\* minus S+ is +0.00241, and S+ minus S is -0.00469 (point differences from
`scores/tables.json`).

### Report-only readings

**The report-only readings enter no verdict row and no Bonferroni count, and each is read at the
same level (99.286%).** A negative difference favours the first arm. HB is the harm reading of the
band of 1 to 3 MVA, and it sits outside the count because it can only lower a verdict.

| Reading | What is compared | Difference and interval | Reading word | Half-hours |
|---|---|---|---|---|
| R1 | U-SG minus S+, sign | +0.00057 (+0.000002 to +0.00129) | loss | 94,080 |
| R2 | U\* minus U-SG, sign | +0.00008 (-0.00068 to +0.00071) | no detectable difference | 94,080 |
| R3 | U\*-L minus S-L (730), sign | -0.00091 (-0.00226 to +0.00017) | no detectable difference | 94,080 |
| R4 | S+ minus S (730), sign | -0.00131 (-0.00299 to +0.00002) | no detectable difference | 94,080 |
| R5 | U\* minus U-NC, sign | -0.00060 (-0.00126 to -0.00006) | gain | 94,080 |
| R6 | U\* minus B-PV1, solar error | -6.960 (-9.094 to -5.160) | gain | 34,765 |
| R7 | U\*-L minus U\*, solar error | +0.105 (-0.071 to +0.292) | no detectable difference | 34,765 |
| R8 | U\* minus S+, sign on the transfer overlay | +0.00085 (+0.00024 to +0.00173) | loss | 94,080 |
| R9 | U\* minus U-NC, share of monotonicity violations | -0.321 (-0.401 to -0.244) | gain | 35,008 |
| R10 | U\* minus U-NCw, solar error | +0.481 (+0.294 to +0.680) | loss | 34,765 |
| R11 | U\* minus U-NCw, injection error | -0.141 (-0.232 to -0.051) | gain | 97,116 |
| R12_solar_only | C4 on solar-only rows | -0.081 (-1.699 to +1.918) | no detectable difference | 7,340 |
| R12_no_injection | C4 on no-injection rows | -0.001 (-0.579 to +0.439) | no detectable difference | 5,275 |
| R13 | U\* minus U-N, sign | -0.00088 (-0.00211 to +0.00012) | no detectable difference | 94,080 |
| R14_mirror | U\* minus U-N, sign at transfers with a mirror the neural network reads | -0.00078 (-0.00808 to +0.00239) | no detectable difference | 23,908 |
| R14_no_mirror | U\* minus U-N, sign at transfers without such a mirror | -0.00150 (-0.00482 to +0.00048) | no detectable difference | 17,756 |
| R15 | U-N minus B-T, step event | -0.00081 (-0.00118 to -0.00050) | gain | 94,080 |
| R19 | U\* minus U-noBSP, sign, two BSP test months | -0.000003 (-0.00162 to +0.00210) | no detectable difference | 23,808 |
| R21 | U\* with the BSP input hidden at scoring minus U\*, sign, two BSP test months | +0.00025 (-0.00136 to +0.00196) | no detectable difference | 23,808 |
| R23 | U\* minus B-PV0, solar error on solar-world rows | -14.088 (-21.235 to -8.316) | gain | 19,375 |
| HB | U\* minus S, mean P(X<0) on import half-hours of 1 to 3 MVA | +0.00657 (-0.01052 to +0.02846) | no detectable difference | 3,289 |

### The solar head recovers an injected output on synthetic days, with two bars missed

**The solar head's output rises by the injected amount at 20 and 50 points and falls short at 5
points.** The injection test adds a known extra solar output δ, taken from another synthetic site's
solar output on the same day, to a window, and asks whether the solar head's output rises by δ (the
recovery ratio is the sum of the rise over the sum of δ, over sun-up half-hours where δ is at least
5% of the injected amount a). The table gives the seed-averaged result for each arm with a solar
head.

| Arm | Recovery, a = 5 (bar 0.9 to 1.1) | Recovery, a = 20 | Recovery, a = 50 | Recovery in reverse flow (bar 0.8 to 1.2) | Absolute error, solar worlds (points; bar 15.5) | Night mean (points; bar 0.05) | Night share above 0.5 (bar 0.5%) | Monotonicity violations (bar 5%) | Response to the sky (bar 50%) |
|---|---|---|---|---|---|---|---|---|---|
| U\* | 0.834 | 0.932 | 0.965 | 0.997 | 4.89 | 0.065 | 1.02% | 0.74% | 91.1% |
| U-NC | 0.707 | 0.880 | 0.941 | 0.938 | 5.42 | 0.211 | 4.70% | 32.81% | 28.6% |
| U-NCw | 0.779 | 0.951 | 0.991 | 1.033 | 4.58 | 0.147 | 2.34% | 34.82% | 25.5% |
| U-SG | 0.201 | 0.216 | 0.271 | 0.852 | 23.78 | 0.258 | 1.79% | 4.94% | 10.6% |
| U\*-L | 0.805 | 0.911 | 0.952 | 1.006 | 5.08 | 0.061 | 1.05% | 0.41% | 94.0% |

**U\* passes six of eight bars, and the two misses are small.** The bar of 15.5 points in the
absolute-error column is half the mean solar target of 30.9 points on sun-up half-hours of solar
worlds, set before any fit. U\* has a mean absolute error of 4.89 points there. The eighth bar,
better than the sky alone, is R23: -14.09 (-21.24 to -8.32), a gain. The seventh, better than the
demand output, is C4. U\*-L passes the same six bars and misses the same two (its night mean is
0.061 and its 5-point recovery is 0.805).

**The contrastive terms are what make the solar head obey the physical rules.** U-NC and U-NCw,
which lack those terms, fail the monotonicity bar (33% and 35% violations against 5%) and the
response bar (29% and 25% against 50%), and U\* passes both (0.74% and 91%). The mean absolute
injection error is 0.49 points lower for U\* than for U-NC (C3: -0.487, interval -0.619 to -0.372),
and 0.14 points lower than for U-NCw (R11). The overall solar error is not detectably different from
U-NC (C2: -0.070, interval -0.313 to +0.160), and is 0.48 points higher than for U-NCw (R10: +0.481,
interval +0.294 to +0.680), so a heavier plain solar loss lowers the error but not the physical
violations.

**The table splits the solar error by what a window holds.** Each cell is the mean absolute error on
sun-up half-hours, in points. S1 and S2 are the two sources of simulated solar output (S1 is a
metered farm's record, shown only as an aggregate; S2 is a satellite-driven solar model).

| Arm | No injection | Engine only | Solar only | Engine and solar | Source S1 | Source S2 |
|---|---|---|---|---|---|---|
| U\* | 0.95 | 1.51 | 4.35 | 5.28 | 5.17 | 4.90 |
| U-NC | 0.52 | 0.96 | 4.94 | 5.74 | 5.70 | 5.38 |
| U-SG | 0.00 | 0.72 | 24.15 | 23.09 | 20.04 | 24.91 |
| U\*-L | 0.86 | 1.56 | 4.43 | 5.51 | 5.24 | 5.15 |
| B-PV1 | 4.30 | 10.97 | 7.93 | 13.70 | 10.86 | 12.07 |
| B-PV2 | 0.96 | 8.05 | 4.41 | 10.18 | 7.20 | 8.48 |

**U\* gives false solar output of 0.95 points on days with no injection and 1.51 points on days with
an engine and no solar.** The rules B-PV1 and B-PV2, which subtract the signed reading implied by a
demand estimate, attribute the engine to the solar output (10.97 and 8.05 points on engine-only
days, against 1.51 for U\*). In worlds with solar and an engine, U\* scores 5.28 points against
10.18 for B-PV2. On the solar-only rows the gain over B-PV2 is not detectable (R12: -0.081, interval
-1.70 to +1.92), and the same holds on the no-injection rows (-0.001, interval -0.58 to +0.44), so
the gain of C4 sits on the rows that hold an engine.

**The recovery depends on the sign of the plain member.** U\*'s pooled recovery is 0.94 to 0.95 on
half-hours where the plain member imports, 0.997 where it is in reverse flow, and 0.86 to 0.90 where
the injection changes the sign (table T5 in `scores/tables.json`). A head that mistook demand for
solar output would give a ratio near -1 in reverse flow, so U\* does not do that.

![The solar error, the night output, and the response to the sky](fig07_solar_checks.svg)

**Figure 4 shows the night output, the monotonicity violations, and the response to a raised sky for
each arm.** U\* and U\*-L have a night mean just above the bar of 0.05 points, and U-NC and U-NCw
sit far from it. The rows draw B-PV0, B-PV1, and B-PV2 at zero night output by construction.

![Recovery of an injected solar output](fig06_recovery.svg)

**Figure 5 shows the rise of the solar head against the injected output, on import half-hours and on
reverse-flow half-hours, for each arm.** Points on the dotted diagonal are exact recovery. U\*, U-N,
U-NC, U-NCw, and U\*-L follow the diagonal, U-SG stays near zero on import half-hours, and the
5-point panels hold only a few points because the injected output is small.

### Transfer detection works in Brier score and finds few of the steps

**The step event is rare (767 half-hours of the 94,080 hold a transfer edge), so a Brier score of
0.0072 is only a little below the 0.0082 of a neural network that always says no.** The table gives
each arm's step Brier score on the transfer overlay and its precision, recall, and F1 at a threshold
of 0.5 on the 594 steps of 5 points and more. A flagged half-hour within one half-hour of a true
edge counts as a hit.

| Arm | Brier, step event | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Calls | Size error (points) |
|---|---|---|---|---|---|---|
| B-T | 0.00808 | 0.000 | 0.000 | 0.000 | 0 | n/a |
| B-TN | 0.00779 | 0.622 | 0.039 | 0.073 | 37 | n/a |
| U-N | 0.00727 | 0.714 | 0.067 | 0.123 | 56 | 4.98 |
| U-NC | 0.00714 | 0.730 | 0.077 | 0.140 | 63 | 4.69 |
| U-NCw | 0.00719 | 0.770 | 0.079 | 0.144 | 61 | 4.67 |
| U-SG | 0.00828 | 0.000 | 0.000 | 0.000 | 0 | 11.86 |
| U\* | 0.00720 | 0.657 | 0.077 | 0.139 | 70 | 4.73 |
| U-noBSP | 0.00686 | 0.738 | 0.157 | 0.258 | 126 | 4.45 |
| U\*-L | 0.00719 | 0.610 | 0.042 | 0.079 | 41 | 4.40 |

**The step head of U\* has a precision of 66% and a recall of 7.7%.** B-T, the rule without
neighbours, never flags a half-hour at 0.5, and B-TN flags 37 and finds 3.9%. The step-size head has
a mean absolute error of 4.73 points on true edges. The step probability is calibrated low: figure 6
shows that the recall falls steeply as the threshold rises from 0.1.

**Recall rises with the size of the step but stays low.** The table gives the recall at 0.5 by the
size of the transfer in D8's noise units (1.4826 times the median absolute step statistic of the
training base days, 3.42 points for D8). The counts of true steps are 265, 206, 198, and 98.

| Arm | Under 2 | 2 to 4 | 4 to 8 | 8 and more |
|---|---|---|---|---|
| Arm | under 2 | 2 to 4 | 4 to 8 | 8 and more |
|---|---|---|---|---|
| B-TN | 0.000 | 0.000 | 0.015 | 0.204 |
| U-N | 0.000 | 0.005 | 0.106 | 0.184 |
| U\* | 0.000 | 0.010 | 0.111 | 0.224 |
| U-noBSP | 0.000 | 0.053 | 0.258 | 0.316 |

**No size band is detectable by the pre-registered rule, which asks for a recall and a precision of
0.5 together.** The detectability statement for U\* and for U-N is therefore empty: the plan's
expectation that steps of 4 noise units and more are detectable is not met. U-noBSP, which has no
BSP input, finds more steps (an F1 of 0.258 and a recall of 0.157 against 0.139 and 0.077 for U\*),
and the BSP input detectably worsens the step Brier score at transfers with an exact mirror and at
transfers to a Primary under another BSP (R22: +0.00115 and +0.00079). The BSP input is no help to
transfer detection in this design.

![Precision and recall of the step output](fig08_transfer_steps.svg)

**Figure 6 shows precision rising and recall falling as the step threshold rises, and the recall of
each arm by size band.** U-noBSP (black) has the highest recall at every threshold and in every size
band. The U-SG and B-T arms sit at zero.

**The neighbours help only where the mirror is exact and read.** C7 pools every partner. The table
splits C7 and C5 by partner case (R16) and by shape (R17), and the sign and the step by world type
and by BSP input (R20 and R22). R16 and R22 name partner cases as `read_exact` (a mirror at D3, D5,
or D14 of the same size), `read_jittered` (the same with the size changed by a factor of 0.7 to
1.3), `unread_bsp_a` (a Primary under BSP A that the neural network does not read), and `other_bsp`
(a Primary under another BSP).

| Reading | Difference and interval | Reading word | Half-hours |
|---|---|---|---|
| R16_other_bsp | +0.00021 (-0.00042 to +0.00085) | no detectable difference | 9,487 |
| R16_read_exact | -0.00063 (-0.00129 to -0.00015) | gain | 16,781 |
| R16_read_jittered | -0.00037 (-0.00129 to +0.00038) | no detectable difference | 7,127 |
| R16_unread_bsp_a | +0.00055 (-0.00014 to +0.00148) | no detectable difference | 8,269 |
| R17_C5_constant | -0.00155 (-0.00253 to -0.00078) | gain | 23,181 |
| R17_C5_demand | -0.00140 (-0.00270 to -0.00036) | gain | 9,739 |
| R17_C5_solar | -0.00144 (-0.00303 to -0.00022) | gain | 8,744 |
| R17_C7_constant | -0.00014 (-0.00064 to +0.00031) | no detectable difference | 23,181 |
| R17_C7_demand | -0.00030 (-0.00109 to +0.00031) | no detectable difference | 9,739 |
| R17_C7_solar | -0.00007 (-0.00079 to +0.00069) | no detectable difference | 8,744 |
| R20_both | +0.00011 (-0.01157 to +0.01626) | no detectable difference | 5,952 |
| R20_engine | -0.00014 (-0.00363 to +0.00278) | no detectable difference | 8,928 |
| R20_solar | +0.00009 (-0.00084 to +0.00139) | no detectable difference | 5,952 |
| R22_other_bsp | +0.00079 (+0.00009 to +0.00168) | loss | 9,487 |
| R22_read_exact | +0.00115 (+0.00056 to +0.00193) | loss | 16,781 |
| R22_read_jittered | +0.00085 (-0.00004 to +0.00193) | no detectable difference | 7,127 |
| R22_unread_bsp_a | +0.00082 (-0.00013 to +0.00189) | no detectable difference | 8,269 |

**The three transfer shapes (constant, demand-shaped, and solar-shaped) all give a gain for C5 and
none for C7.** The recall on solar-shaped transfers is 0.117 for U\* and for U-N, and the recall on
constant transfers is 0.082 for U\* and 0.070 for U-N (R18).

![The seven counted contrasts, in groups of one unit each: transfer step](contrasts_step.svg)

**Figure 7 shows C5 left of zero and C7 and R15 against the same axis.** U-N minus B-T (R15) is
-0.00081 (-0.00118 to -0.00050), so the transfer head beats the rule even without the neighbours.

### Load-transfer scenarios

**The four scenarios that the `ml-experiment` skill requires score every arm worse inside a transfer
than outside one.** The table gives the sign and step Brier scores of U\*, S+, and U-N and the solar
error of U\*. Experiment 730's S was never run on the transfer overlay, so S has no row. The
transfer overlay of the test bank holds 384 simulated transfers. The scenarios are all half-hours,
half-hours outside a transfer and at least 7 days after one ended, half-hours inside a transfer, and
half-hours within 7 days after a transfer ended. The synthetic transfers are not logged events, so
the caveat that real logs miss events does not apply.

| Scenario | Half-hours | U\* sign | S+ sign | U\* solar error (points) | U\* step | U-N step |
|---|---|---|---|---|---|---|
| Scenario | Rows | U\* sign | S+ sign | U\* solar error (points) | U\* step | U-N step |
|---|---|---|---|---|---|---|
| all | 94,080 | 0.00838 | 0.00753 | 3.49 | 0.00720 | 0.00727 |
| outside | 19,247 | 0.00602 | 0.00523 | 3.07 | 0.00030 | 0.00033 |
| inside | 12,065 | 0.01092 | 0.00909 | 3.97 | 0.05387 | 0.05470 |
| after | 62,768 | 0.00861 | 0.00793 | 3.53 | 0.00035 | 0.00028 |

### Real days: two of four checks pass

**On real days the sky check and the night check pass, and the two checks of "no solar output where
there is none" fail.** No real-day figure shows a level of the real solar output. P1 reads the night
half-hours of every era against the night bar. P2 reads the held-out days from before the engine,
which have no solar farm, and asks that the solar head's output stays at or below 1 point on average
and that at most 1% of sun-up half-hours exceed 5 points; it fails, with 2.1% of those 4,376
half-hours above 5 points. P3 reads the days of the solar and fade eras: the Spearman correlation
between the daily sum of the solar output and the daily mean of the clear-sky index is 0.54 over 465
days (bar 0.5), and the mean midday output on clear days is 4.2 times that on cloudy days (bar 2);
it passes. P4 asks that the solar output does not rise on half-hours when the metered engine runs;
it fails, and prints a pass or fail word only. H2 passes (mean probability of reverse flow 0.00012
on 11,760 pre-engine half-hours, and none above 0.5).

![Real-day checks](real_checks.svg)

**K3 is void and K1 and K2 were not scored.** K3 (the share of logged closure edges flagged within 2
hours) could not read any edge; the section "Disclosures" says why. On the 245 held-out pre-engine
days, U\* flagged 0.0 steps a day and U-N 0.004 steps a day, with no closure day excluded. K1 and K2
(experiment 730's checks on the metered engine record's days) are not in this experiment's real-day
output, and the section "Disclosures" lists them as not done.

### The fits stopped on the sign, and the solar head was still improving in some arms

**Every arm stops early on the validation Brier score of the sign, and training took 2.41
GPU-process-hours.** The table gives each fitted arm, its architecture, the number of weights, the
epochs run by seeds 0, 1, and 2, the epoch of the stored state file for each seed, and the hours of
fit time summed over the three seeds.

| Arm | Architecture | Weights | Epochs run (seeds 0 / 1 / 2) | Epoch stored | Hours of fit time |
|---|---|---|---|---|---|
| S+ | unet | 1,067,777 | 125 / 109 / 117 | 100 / 84 / 92 | 0.17 |
| U\* | unet | 1,067,876 | 168 / 196 / 124 | 143 / 171 / 99 | 0.44 |
| U-N | unet | 1,065,956 | 85 / 150 / 112 | 60 / 125 / 87 | 0.15 |
| U-NC | unet | 1,067,876 | 81 / 105 / 125 | 56 / 80 / 100 | 0.26 |
| U-noBSP | unet | 1,067,876 | 135 / 194 / 145 | 110 / 169 / 120 | 0.41 |
| U-NCw | unet | 1,067,876 | 120 / 136 / 134 | 95 / 111 / 109 | 0.32 |
| U-SG | unet | 1,067,876 | 99 / 96 / 91 | 74 / 71 / 66 | 0.25 |
| U\*-L | bilstm | 548,052 | 162 / 120 / 153 | 137 / 95 / 128 | 0.41 |

![Training and validation loss by epoch](loss_curves.svg)

**Figure 8 shows seed 0 of each arm, with the training loss (solid) and the validation loss (dashed)
of each term and of their weighted sum, against the epoch, and the stored epoch as a dotted line.**
The validation loss of every term is above the training loss throughout: for U\*, seed 0, the
weighted sum is 0.245 at the stored epoch on validation against 0.105 on training. The validation
total of U\* is flat after the stored epoch (0.245 at epoch 143, 0.245 at epoch 168), so the stop on
the sign did not leave the total unfinished, but the validation solar error of some arms is lower
later than at the stored epoch (seed 0 of U-NCw: 3.39 points at the stored epoch 95 and 2.84 at
epoch 113; U-NC: 4.07 at epoch 56 and 3.57 at epoch 70). The solar head of U-NCw and U-NC was
therefore still improving when the sign stopped the fit. The sign validation Brier score of the
stored epoch is the best of its fit by construction. The panel of S+ shows the sign and the total
only, because S+ has no other term.

### Single windows

**The 23 windows below each show one scored day, picked by a fixed rule from the test overlays, with
no calendar date.** Each figure shows, from the top, the signed truth and the reading, the
probability of reverse flow of U\*, the solar target and U\*'s solar output, and the true transfer
edges with U\*'s step probability. The titles repeat in the figure files. The windows come from
engine-only, S2, and no-injection segments, so no metered farm is shown, and two windows the plan
listed for real days (W24 and W25) were never drawn, because K3 is void. The probability panel shows
U\* alone: the plan also asked for S+ and S (730), which the series stage did not draw.

**W1 to W4: engine and solar worlds, each the window with the median depth of reverse flow in its
group.**

![W1: an evening engine run](series_W1.svg)

*W1: U\* gives the probability of reverse flow a peak of about 0.65 in the evening run, and a false
solar output of about 3 points (the target is zero). The step event at about 8 hours is not
flagged.*

![W2: a morning engine run](series_W2.svg)

*W2: the morning run gives a peak probability of about 0.45 and false solar output of about 4
points.*

![W3: solar output alone](series_W3.svg)

*W3: the probability of reverse flow rises to 1 over the reverse-flow span, and the solar output
follows the solar target closely.*

![W4: engine and solar together](series_W4.svg)

*W4: the probability reaches 1 over the reverse-flow span and the solar output follows the target,
with the output a few points low around 10 hours.*

**W5 to W8: hard windows.** W7 and W8 are the windows on which U\* gains most and loses most against
S in Brier score.

![W5: a window within 0.3 MVA of zero](series_W5.svg)

*W5: the reading stays near zero for at least 4 half-hours, and the probability stays between 0.5
and 0.9 all day. The solar head outputs a hump of about 15 points while the target is zero.*

![W6: the import window with the largest mean probability](series_W6.svg)

*W6: the import window with the largest mean probability of reverse flow on decidable rows reaches
about 0.6 around 14 hours.*

![W7: where U\* gains most against S](series_W7.svg)

*W7: the window on which U\* gains most against experiment 730's S has a small reverse flow at
night; the probability stays at 0.4 to 0.5 over the night.*

![W8: where U\* loses most against S](series_W8.svg)

*W8: the window on which U\* loses most has a reading that varies about zero, and the probability is
high through the day.*

**W9 to W11: injection pairs at 5, 20, and 50 points.** Each shows the solar output of the plain
window, of the injected window, and the line of the plain output plus δ.

![W9: injection at 5 points](series_W9.svg)

*W9: the injected window's output follows the line of the plain output plus δ closely, with a small
shortfall in the morning.*

![W10: injection at 20 points](series_W10.svg)

*W10: at 20 points the injected output rises above the line of the plain output plus δ, by about 1.5
points at midday.*

![W11: injection at 50 points](series_W11.svg)

*W11: at 50 points the injected output is about 27 points at midday against 24.5 on the line, an
over-response of about 10% in this window; the pooled recovery at 50 points is 0.965.*

**W12 and W13: the two largest solar errors.**

![W12: engine only, the largest solar output](series_W12.svg)

*W12: in an engine-only window U\* outputs a solar hump of about 28 points at midday, against a
target of zero.*

![W13: the clear S2 window with the largest solar error](series_W13.svg)

*W13: U\* outputs about 95 points at midday against a solar target of about 15, because the reverse
flow is attributed to solar output.*

**W14 to W19: six load-transfer windows.** Two add load, two remove load, one is inside a solar
spell, and one has a partner under another BSP.

![W14: adds load](series_W14.svg)

*W14: the transfer adds load; four true edges appear in the evening and the step probability stays
near zero, so U\* misses them. The sign probability reaches about 0.9 over the reverse flow at
midday.*

![W15: adds load, a second window](series_W15.svg)

*W15: two true edges late in the day get a step probability near zero.*

![W16: removes load](series_W16.svg)

*W16: a transfer that removes load; the true edge at about 7 hours gets a step probability near
zero.*

![W17: removes load, a second window](series_W17.svg)

*W17: a second transfer that removes load; the true edge at about 13 hours gets a step probability
near zero.*

![W18: inside a solar spell](series_W18.svg)

*W18: a transfer inside a solar spell; the window looks the same by eye as W16 and W20, so the three
rules pick one window.*

![W19: a partner under another BSP](series_W19.svg)

*W19: a transfer to a Primary under another BSP; the four true edges between 17 and 23 hours get a
step probability near zero.*

**W20 to W23: shape, mirror, and the transfers of most interest.**

![W20: a solar-shaped transfer with a mirror](series_W20.svg)

*W20: a solar-shaped transfer with a mirror at a neighbour; the step probability reaches about 0.25
near 21 hours and stays near zero at the true edge at about 7 hours.*

![W21: a solar-shaped transfer without a mirror](series_W21.svg)

*W21: a solar-shaped transfer without a mirror; the four true edges in the evening are missed, and
the window looks the same by eye as W14.*

![W22: the mirrored transfer with the largest gain over U-N](series_W22.svg)

*W22: the mirrored transfer on which U\* gains most over U-N in the step Brier score; U\* gives a
step probability of about 0.55 at one of the true edges and a lower one at the others.*

![W23: the transfer of at least 4 noise units that U\* misses by the widest margin](series_W23.svg)

*W23: the true edge has a step probability of about 0.2 at the edge, so U\* misses it at the
threshold of 0.5.*

**The windows show the pattern of the scores.** The sign probability follows reverse flow, the solar
output follows the solar target except where the reading is near zero or an engine runs, and the
step probability reaches 0.5 rarely. Attention maps do not apply, because the U-net has no attention
layer.

### Checks

- **The median forecast minus the mean forecast does not apply,** because the neural networks output
  a probability of reverse flow and a solar output, not a quantile forecast.
- **The error of the median forecast beside the quantile-score contrasts:** the solar output is a
  point output, and its mean absolute error (the normalised error of a median in percent of
  capacity) is the metric of C2 to C4, printed beside its root mean squared error in the solar
  table. The sign has no median, so no normalised error of the median is reported for it, and the
  plan's request for one is not met.
- **Audit triggers 2 and 5 fired, and no audit has run.** Trigger 5 fired because S+, a nested arm
  with fewer outputs, scores better than U\* by a margin whose interval excludes zero (C6). Trigger
  2 fired because C6 and the solar word differ from the prediction. The stored evidence is the
  stop-gradient arm (R1 and R2) and the arms without the contrastive terms (R5), which place the
  cost on the extra heads and not on the shared gradients. Trigger 3 did not fire: the Brier and
  log-loss points agree for C1, C6, and R4. Trigger 4 does not apply (the test bank is synthetic).
  Trigger 6 does not fire: C1's half-width (0.00115) matches experiment 730's C1 (0.00117). An
  independent audit is left to the manager.

### What the result means for later experiments

**A sign-only neural network is enough for the probability of reverse flow, and the extra heads do
not belong in it.** S+ is the best arm on every sign score, and U\* is the best arm that also gives
a solar output.

**The solar output works on synthetic days, and the contrastive terms are what make it obey the
physical rules, but the real-day checks do not yet support it.** The next experiment on the solar
output should keep the contrastive terms, raise the supervised solar weight (U-NCw's error is lower
than U\*'s), stop each fit on a score that includes the solar error, and read the real-day checks P2
and P4 as the targets, because the head outputs solar where no solar farm exists.

**The switching detector needs a different design.** A transfer detector that finds 8% of simulated
steps is not usable. The neighbours help only at an exact mirror, and the BSP input worsens
detection, so the next experiment on transfers should drop the BSP input (U-noBSP found twice as
many steps), and read the switching logs first, because the transfer generator has no realism gate.

## Disclosures

**Each item below departed from the plan or limits what the page can claim.** No bank and no bar
changed, and `CODE_VERSION` stayed 2.

- **A zero-fill rule excludes unreliable solar targets.** Experiment 729's solar simulator keeps a
  day of the metered farm's output when the farm has at least 40 of 48 readings, and it fills each
  missing half-hour with zero, so the banks hold zeros that are unknown outputs. A half-hour of that
  source is "unreliable" when its output is zero, the same day has positive output elsewhere, the
  cosine of the zenith angle is at least 0.2, and the clear-sky index is present. The first version
  required positive output on both sides of the zero, and a read-only diagnostic before any fit
  showed that zero-filled tails at the edges of a day fail the solar-target check in the same way,
  so the rule was extended to those edge tails. The unreliable half-hours leave the solar loss, the
  injection loss, the solar bars, and the solar contrasts: 243 test half-hours are excluded.
- **The unreliable-share bound was raised from 5% to 7%, with the maintainer's approval, before any
  fit.** The line stops when any bank's unreliable share of sun-up half-hours of solar worlds
  exceeds the bound. Training seed 0 holds 4.21% (the five training banks 3.84% to 4.61%), the
  validation bank 5.68%, and the test bank 3.49%. The validation bank exceeded the first bound of
  5%.
- **The first gate attempt was an overlay attempt, not a verdict.** After the first run of the
  overlay stage the overlay digest key changed, and the stage recorded `invalid_attempt1.json` and
  `summary_attempt1.json`, then the gate ran again (`gate_attempts.json` lists versions 1 and 2). No
  fit or score existed at the time.
- **K3 is void.** The sheets of the switching log load, but the header sits in the first data row
  and the cells hold personal names, so no column could be identified without guessing, and no
  column was guessed. The reader returned no logged edge, so K3's hit rate reads 0 edges and is
  void; the false-step count of 245 days therefore excludes no closure day.
- **K1 and K2 were not scored.** The real-day stage computes H2, P1 to P4, and K3 only, although the
  stored predictions for the K1 and K2 half-hours exist. The page makes no claim from the metered
  engine record except P4's pass or fail word.
- **`fastexcel` is installed in the environment but not declared in `pyproject.toml`.** The Excel
  reader of the switching log needed it.
- **The timing projection was 6.21 GPU-process-hours against the plan's 3.1, and the fits took
  2.41.** The timing gate measured 3.41 seconds an epoch for U\* against the plan's 1.6, because
  building a window with its transfer overlay, mirror, injected member, and raised-sky member takes
  2.12 seconds an epoch on one CPU thread. The projection assumed every fit ran to the cap of 300
  epochs, and every fit stopped between 81 and 196 epochs. The 2.41 hours are the sum of the 24 fit
  records' `seconds`; the timing gate took 0.025 hours, and the time of inference was not recorded
  separately. The projection of 6.21 hours was under the stop line of 16 hours, so no arm was
  dropped.
- **Two starts of the fit chain failed, and both were fixed with no bank, bar, or constant
  changed.** The first start failed on all six fits of S+ and U\* with an `AttributeError`, a call
  to a GPU memory-cap function that does not exist. The second start fitted the three S+ seeds and
  failed at the first check of a U\* neural network, where a CUDA tensor was passed to numpy. The
  third start skipped the finished fits and ran the rest. The 24 fits that scored come from the
  third start and the S+ fits of the second.
- **NaN cosines of the zenith angle count as sun down in the B-T and B-TN rules.** The real banks
  hold NaN cosines, which turned the step rules into NaN; the fix was set before any score existed
  (the test-score marker is dated after it).
- **Series windows W24 and W25 were never drawn,** because K3 is void.
- **Plan items not done.** The shifted-mix bank (table T4) and the recovery by tercile of the BSP
  coupling and with the BSP channel hidden in the injection overlay (the last part of table T5) were
  not computed; table T2 has no row for S, because S was never run on the transfer overlay; the
  probability panel of the windows shows U\* alone; the normalised mean absolute error of the median
  of the sign was not reported; no audit ran; and the plan's figures 3 (harm), 5 (solar error by
  world), 9 (scenarios), and 12 (BSP input) appear as tables here, although their drawings are among
  the stage's files.
- **Table T3 (the ambiguity read-out by reading band) and the split of table T5 by sign are in
  `scores/tables.json`; the T5 split is summarised in the solar section and T3 is not reproduced.**
- **The synthetic transfer generator has no realism gate,** so the transfer word and the neighbour
  word describe the synthetic world only.

## The synthetic world and the arms

### Terms

- **Signed power X** is D8's real power, positive for import and negative for reverse flow.
- **The reading R** is what D8's meter reports: R = sqrt(X² + Q²), where Q is the reactive floor,
  the reactive power that keeps the reading above zero when X crosses zero. **Folding** is the loss
  of the sign.
- **A point** is 1% of D8's effective capacity of 10.379 MVA, the normaliser of
  [data-processing](../data-processing.md). One point is 0.104 MW at power factor 1.
- **P(X<0)** is the probability that a half-hour's signed power is negative. **The solar output** is
  the output of the simulated solar panels below D8 in a synthetic window, in points, and **the
  solar head** is the output of the neural network that estimates it.
- **A load transfer** moves a block of customers between D8 and another Primary for a span of time,
  so D8's demand steps by a roughly constant amount at the start and steps back at the end. **A
  transfer step** is the half-hour that holds the start or the end.
- **The bulk supply point (BSP)** above D8 is BSP A, which feeds D8 and eight other Primaries. Its
  meter reports signed MW, and **the BSP steps** are its half-hour changes. A transfer to a Primary
  under BSP A leaves the BSP meter unchanged, and a transfer to a Primary under another BSP moves
  it.
- **D8's electrical neighbours** are D3, D5, and D14, listed as interconnected with D8 by
  [experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md) (multi-day
  drops at D8) and [experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md) (era
  signals). D3 is metered in MVA and D5 and D14 in signed MW.
- **A noise unit** of a series is 1.4826 times the median absolute step statistic of the series over
  the training base days, where the step statistic is the mean over the next 2 hours minus the mean
  over the previous 2 hours, after the daily shape is removed. D8's noise unit is 3.42 points.
- **A base day** is a UTC day of D8's real reading before the engine was connected. **A segment** is
  one calendar month of base days carrying one continuous injection draw. **Import rows** have X at
  or above 0, **reverse-flow rows** have X below 0, and **decidable rows** have a reading of at
  least 1 MVA. **Sun-up half-hours** have a cosine of the solar zenith angle above 0.1, and **night
  half-hours** have one at or below 0.

### The arms

**Eight arms are fitted with seeds 0 to 2 (24 fits), and two arms of experiment 730 are read from
stored predictions.** Every arm hides one random span of the reading, 1 to 24 hours long, on half of
its training windows.

| Arm | Architecture | Outputs | Contrastive terms | Role |
|---|---|---|---|---|
| U\* | U-net | Sign, solar, step, size | Injection, night, monotone | The headline arm |
| U-NC | U-net | Sign, solar, step, size | None | Control of C2 and C3 |
| U-NCw | U-net | Sign, solar, step, size | None, supervised solar weight 0.08 | Report only: weight-matched control |
| S+ | U-net | Sign | None | Control of C6: sign-only on this experiment's windows |
| U-SG | U-net | As U\* | As U\* | Report only: no gradient from the solar, step, and size terms reaches the backbone |
| U\*-L | Bidirectional LSTM | As U\* | As U\* | Report only: the architecture |
| U-N | U-net | As U\* | As U\* | Control of C7: no neighbour channels |
| U-noBSP | U-net | As U\* | As U\* | Report only: no BSP channel |
| S (730) | U-net | Sign | None | Control of C1: stored predictions of experiment 730 |
| S-L (730) | Bidirectional LSTM | Sign | None | Control of R3: stored predictions of experiment 730 |

**Three rules give the solar output without a solar head, and two logistic regressions give the step
event.** B-PV2 subtracts the signed reading implied by experiment 730's demand-output arm from its
demand output (the counted reference of C4). B-PV1 does the same with D8's demand climatology. B-PV0
predicts the solar target from the sky alone, with a least-squares fit on the validation bank. B-T
and B-TN are logistic regressions fitted on the validation overlay on step statistics in noise
units, and B-TN adds the neighbours' sign-aligned steps, the largest opposite-sign step, and the
count of neighbours that step the same way as D8 (the cues of [experiment
65](../00065-sw2-14-combined-single-series-and-neighbour-score/result.md), the mirror, and
[experiment 70](../00070-sw2-19-same-sign-neighbour-count/result.md), the same-sign count). B-T has
5 weights and B-TN 10.

### The loss

**The loss of one training triple (the window, its injected member, and its raised-sky member) is
the sum of seven terms.** The sign term is binary cross-entropy on X < 0 on the plain member (weight
1.0), the step term is binary cross-entropy on the step event (1.0), and five terms of weight 0.02
each are the Huber losses (scale 1 point) of the solar output and of the step size, the injection
term (the rise of the solar output between the two members must equal δ), the night term (mean
output at night), and the monotone term (a raised clear-sky index must not lower the output). U-NC
drops the last three, and U-NCw drops them and gives the solar term a weight of 0.08. The injection
term follows the contrastive term of [experiment
624](../00624-pv-probe-contrastive-injection-arms/result.md), at a smaller weight, with the same
function for both members and a bounded Huber loss.

### The verdict table

**The first row that matches is the verdict.** NI means C1's upper end is at most +m (0.00122).

| Row | Condition | Verdict |
|---|---|---|
| 0 | The line stops before C1 is read | Stopped, no verdict |
| 1 | A correctness check fails | `invalid` |
| 2 | A fit that a counted reading needs raised and was not rerun | `failed` |
| 3 | H1 pooled or at 3 MVA and above fails, H2 fails, or the band of 1 to 3 MVA is above its limit with the lower bound of HB above zero | `not supported` |
| 4 | NI | `supported` |
| 5 | C1's upper end above +m and its lower end at or below zero | `inconclusive` |
| 6 | Every other case | `not supported` |

**Four reading words sit beside the verdict and change no row.** The solar word is "reasonable" when
all eight solar bars pass, "partly" when the two recovery bars, the absolute bar, and the bar
against the sky alone pass and another fails, and "not shown" otherwise. The transfer word is
"works" when C5 is a gain and the F1 at 0.5 on steps of 5 points and more is at least 0.6, and
"works only in Brier score" when C5 is a gain and the F1 bar fails. The neighbour word reads C7 and
the BSP word reads R19.

## Synthetic world gallery

**The gallery shows the output of every synthetic generator that this experiment's windows use.**
Experiment 730 drew the panels G1 to G9 (engine days, solar output, the BSP meter, the fold, base
days, masks, realism, and component checks) from the same digest-pinned banks, and its page holds
them: [experiment 730's
gallery](../00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md#synthetic-world-gallery).
The panels G10 to G21 below were drawn from this experiment's overlays before any fit, and the
session recorded in `PRIVATE/gallery/panel_check.json` that each panel shows what its written list
requires (12 of 12 pass). Every window carries hours from the start of day 1 and no calendar date,
and every published window comes from an engine-only, S2, or no-injection segment. G17, the pairs
built from the metered farm's source, is in the private folder only.

**G10: load transfers at D8 change the truth by the drawn load, and the BSP steps mark both edges
only for a transfer to a Primary under another BSP.** Each panel has 3 draws; dotted lines mark the
edges.

![G10: a transfer that adds load on an import day](gallery_g10_adds_load.svg)

*The truth and the reading step up at the first edge and back at the second, and the BSP steps stay
flat, as they should for a transfer inside BSP A.*

![G10: a transfer that removes load in a solar spell](gallery_g10_removes_load_in_solar_spell.svg)

*The reverse flow deepens inside the transfer and the reading folds at the new crossings; the BSP
steps are flat.*

![G10: a transfer to a Primary under another BSP](gallery_g10_other_bsp.svg)

*The BSP meter's steps mark both edges, and the step the transfer adds to the BSP meter is non-zero
only at the edges.*

![G10: a transfer of 2 days or more in a 5-day window](gallery_g10_long.svg)

*The long transfer holds its level for 2 days or more, and a partial half-hour at an edge shows a
step smaller than the drawn size.*

**G11: each contrastive pair differs only on sun-up half-hours.** The re-folded reading equals the
original wherever δ is zero.

![G11: pairs at 5 points](gallery_g11_pairs_5.svg)

![G11: pairs at 20 points](gallery_g11_pairs_20.svg)

![G11: pairs at 50 points](gallery_g11_pairs_50.svg)

*Each panel gives δ, the truth and the injected truth, the reading and the re-folded reading, and
the BSP meter and the injected BSP meter. The pair differs only where δ is above zero.*

**G12: the raised sky is above the original on every raisable half-hour and equal elsewhere, with
the reading unchanged.**

![G12: the raised sky on a clear and a cloudy day](gallery_g12_raised_sky.svg)

**G13: the target stack is non-empty for the worlds that carry it and aligned with its cause.** The
solar target, the step events, and the step sizes of one window of each world type, with the hidden
span shaded.

![G13: the solar target, step events, and step sizes](gallery_g13_targets.svg)

**G14: every histogram of a drawn transfer parameter lies inside its registered range.** The panel
shows the number of transfers per 30 days, the duration, the size, and the share of partners under
BSP A for the training overlay of seed 0 and for the test overlay, and the settled transfer size
after the donor rule by direction, partner, and duration.

![G14: the drawn transfer parameters](gallery_g14_parameters.svg)

**G15: a window with an engine run, solar output, a load transfer, and an extra injection shows
every component.** The injected member differs from the plain one.

![G15: everything at once](gallery_g15_everything.svg)

**G16: the share of windows with a sibling pair is above 95% in every training month.** Months are
numbered in time order.

![G16: pairing coverage by training month](gallery_g16_pairing.svg)

**G18: the partner's series moves by the opposite of D8's change, the other two neighbours' series
do not move, and no neighbour moves when the partner is one that the neural network does not read.**
Each draw shows D8's truth and reading and the three neighbours before the fold and after it, each
without the overlay (thin) and with it (thick), with both edges marked. Each neighbour is drawn as a
multiple of its own median.

![G18: a transfer with partner D3](gallery_g18_partner_D3.svg)

![G18: a transfer with partner D5](gallery_g18_partner_D5.svg)

![G18: a transfer with partner D14](gallery_g18_partner_D14.svg)

![G18: a transfer with a partner that the neural network does not read](gallery_g18_unread.svg)

**G19: the three transfer shapes have the same T₀ and the same solar target.** The constant transfer
holds the transferred load flat, the demand-shaped transfer follows the daily demand shape, and the
solar-shaped transfer follows the sky.

![G19: the three transfer shapes](gallery_g19_shapes.svg)

**G20: the neighbour table covers each neighbour, the drawn sizes straddle 4 noise units, and the
exact mirrors lie on the diagonal.** The coverage panel is by month for information, and the gate
applies the bound of 90% to the whole table (D3 1.000, D5 1.000, D14 0.973). The jittered mirrors in
a second colour scatter around the diagonal.

![G20: the neighbour table and the noise units](gallery_g20_neighbour_table.svg)

**G21: one built input tensor shows every channel moving where it should and flat where it should
not.** The hidden span is shaded, and the neighbour channels are zero inside it. The second figure
shows the same window as built for U-N (neighbour channels absent, drawn as zero) and for U-noBSP
(BSP step and mask zero).

![G21: every input channel of one training window](gallery_g21_tensor.svg)

![G21: the same window for U-N and U-noBSP](gallery_g21_variants.svg)

## Method

**Architecture:**

- U\* reads experiment 730's arm S inputs over a 48-hour window from 12:00 UTC on the day before the
  scored day to 12:00 UTC on the day after, and scores the middle 24 hours: the reading in points
  divided by 50 and its presence mask; the CAMS clear-sky index (clipped to 0 to 1.5), the clear-sky
  irradiance divided by 1,000, the cosine of the solar zenith angle, and a CAMS presence mask at D8;
  the half-hour change of the synthetic BSP meter divided by 20 and its presence mask; and the local
  half-hour, day of week, bank-holiday flag, and the ERA5 2 m temperature, 10 m wind speed, and log
  precipitation at D8's grid cell. No input reads the month or the day of the year.
- U\* also reads 12 neighbour channels: for each of D3, D5, and D14, the scaled folded reading, its
  half-hour change, that change times the sign of D8's change of reading, and a presence mask.
- The backbone is the U-net of experiment 729 (a convolutional neural network after [Ronneberger et
  al. (2015)](https://doi.org/10.1007/978-3-319-24574-4_28)) with four levels of 32, 64, 128, and
  256 channels, kernels of width 5, group normalisation, GELU activations, and skip connections,
  with 1.07 million weights. The bidirectional LSTM of U\*-L has 0.55 million weights.
- U\* outputs, at each of the middle 24 hours' half-hours, a sign logit, a non-negative solar output
  in points (a softplus), a transfer-step logit, and a step size in points, each a linear layer on
  the backbone's last hidden layer.
- The weather is the observed ERA5 and CAMS weather at the scored times, which is legitimate because
  this experiment reconstructs the past and makes no forecast. Any later use of these neural
  networks inside a forecast must read the weather issued at or before the forecast origin.
- U-SG, U-NC, U-NCw, S+, U-N, U-noBSP, and U\*-L differ from U\* as the arms table says.

**Training regime:**

- The training windows come from experiment 730's training banks (synthetic D8 days built from D8's
  pre-engine base days, with a simulated engine and solar output from two sources, in the manner of
  [Kelly and Knottenbelt (2015)](https://doi.org/10.1145/2821650.2821672) and randomised in the
  spirit of [Tobin et al. (2017)](https://doi.org/10.1109/IROS.2017.8202133)), with this
  experiment's load-transfer overlay drawn once per seed and an extra solar injection drawn for
  every window and epoch (log-uniform from 2 to 60 points).
- The load transfers are synthetic: 0 to 6 per 30 days, 1 hour to 4 days long, 3 to 40 points in
  size, constant (60%), demand-shaped (20%), or solar-shaped (20%), and mirrored at D3, D5, or D14
  in 60% of draws, with the mirror's size jittered by a factor of 0.7 to 1.3 in 30% of the mirrored
  draws. A transfer that gives load to D8 is mirrored by the partner's equal and opposite change.
- The neighbours' readings are the real cleaned readings of D3, D5, and D14 on each base day, with
  D5 and D14 folded to a magnitude.
- The loss is the sum of seven terms with the weights above, and the training uses AdamW with a
  learning rate of 1e-3 (halved after 10 epochs with no gain), weight decay of 1e-4, batch size 256,
  gradient clipping at 1.0, epochs of 8,000 plain windows, and early stopping after 25 epochs
  without a gain in the validation Brier score of P(X<0), capped at 300 epochs. Seeds 0 to 2 set the
  initial weights and the training bank. There is no refit schedule and no forecast.
- The fits ran on one GPU process of at most 6 GB (peak 2.30 GB, U\*-L) and 1 CPU thread each, 2.41
  GPU-process-hours in total.

**Measurement:**

- The sign and the solar output are scored on experiment 730's 94,080 test half-hours (8 held-out
  calendar months of D8's pre-engine base days, 8 synthetic segment draws each, the middle 24 hours
  of a 48-hour window; the months are listed by rule in experiment 729's private folder). The
  injection recovery and the monotonicity are scored on fixed test overlays of the same windows
  (injections of 5, 20, and 50 points, and a raised sky of 20%), the transfer steps on the transfer
  overlay, and the BSP contrast on the two test months that the BSP meter covers.
- The baselines are experiment 730's stored S for the sign, the rules B-PV0, B-PV1, and B-PV2 for
  the solar output, and the logistic regressions B-TN and B-T for the transfer steps. P0 is the
  reference of the harm checks.
- Seven contrasts are counted at 99.286%, from experiment 729's two-way bootstrap with Kish widening
  and 20,000 resamples; the report-only readings, the harm reading HB, and the bar reading R23 are
  read at the same level outside the count.
- The metrics are the Brier score and log loss of P(X<0), the harm means H1 and H2, the detection
  mean, the reliability term, the mean absolute error of the solar output in points of capacity, the
  recovery ratio, the Brier score, precision, recall, and F1 of the step event, and the mean
  absolute error of the step size.
- The scored rows drop 243 half-hours of unreliable solar targets (see "Disclosures").
- The standing practices that apply to a reconstruction are followed: cleaned data, saved neural
  networks, stored predictions as controls. Monthly refits, the D15 and D18 cut, the tail coverage,
  and the forecast plots do not apply, because no forecast is made.

## Files

**The code imports the merged modules of experiments 722, 729, and 730 unchanged and adds modules
for the neighbours, the overlays, the heads, the losses, the baselines, and the scores.**
`ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/e00732_paths.py` puts the
folders of experiments 730, 729, and 722 on `sys.path`. No file of an earlier experiment is edited.

**Reused unchanged from [experiment
730](../00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md)
(`ml-experiments/00730-msa-d8-sign-detector-with-demand-head-and-masking/`):** `e00730_core.py`,
`e00730_arms.py` (the channel stack and the window builder), `e00730_masks.py`, `e00730_score.py`
(the Brier score, harm, detection, and calibration), `e00730_real.py`, `e00730_train.py`, and
`run.sh`.

**Reused unchanged from [experiment
729](../00729-msa-d8-neural-untangler-synthetic-known-answers/result.md):** `e00729_sim.py`,
`e00729_banks.py`, `e00729_nets.py` (the U-net and the bidirectional LSTM), `e00729_score.py` (the
contrasts and the Kish numbers), `e00729_baselines.py`, and `e00729_data.py`.

**Reused unchanged from [experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md) (the
closure of MVA meters and the BSP):** `e00722_data.py` (`load_power`) and `e00722_core.py`.

**New in `ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/`:**

- `e00732_core.py`: every named constant and bar, the arms, the gallery's written list, and the
  verdict rule.
- `e00732_neighbours.py`: the neighbour table of D3, D5, and D14, the coverage gate, the noise
  units, and the 12 neighbour channels.
- `e00732_overlays.py`: the transfer generator, the re-fold, the contrastive pairing, the raised
  sky, and the component checks.
- `e00732_arms.py`, `e00732_nets.py`, `e00732_losses.py`, and `e00732_truth.py`: the arm
  specifications and tensor checks, the four heads, the seven loss terms, and the targets.
- `e00732_train.py` and `e00732_fit.py`: the fit loop with its per-epoch loss history, the timing
  gate, and the fit chain.
- `e00732_baselines.py`: B-PV0, B-PV1, B-PV2, B-T, and B-TN.
- `e00732_score.py`, `e00732_real.py`, `e00732_leakage.py`, and `e00732_report.py`: the readings,
  the real-day checks, the leakage assertions, and the score, real, and report stages that write the
  verdict.
- `e00732_gallery.py`, `e00732_figures.py`, `e00732_result_figures.py`, `e00732_series_figures.py`,
  `e00732_stages.py`, `e00732_cli.py`, `e00732_solar_diag.py`, and `run.sh`: the gallery, the
  figures, the windows, the stage orchestration, the command line, and the launch wrapper.
- `e00732_tables.py` (read only): the absolute scores of every arm from the stored predictions,
  written to `scores/tables.json`.
- `e00732_contrast_groups.py` (read only): the three figures of the contrasts by unit, from
  `scores/readings.json`.
- `tests/`: tests on synthetic arrays; those that need `torch` skip in CI.

**Data, under `data/ml-experiments/00732-msa-d8-untangling-network-synthetic-injections/` (private,
never committed):**

- `limits.json`, `banks_check.json`: the harm limits, the noise units, and the bank digests, written
  before any fit.
- `neighbours/` and `overlays/`: the neighbour table, and the test, validation, and training
  overlays with `summary.json` (digests, component checks, and unreliable shares).
- `fits/` and `networks/`: one JSON per fit (loss history of every term, epochs, and stored epoch)
  and the state files.
- `predictions/`: the outputs of each fit on each set, with the row keys.
- `scores/`: `readings.json` (every contrast), `verdict.json`, `real.json`, `timing.json`,
  `bt_fit.json`, `tables.json`, `gate_attempts.json`, and `invalid_attempt1.json`.
- `figures/` and `PRIVATE/`: the figures copied beside this page, and the private gallery, series,
  and logs.
- Read only, from [experiment
  730](../00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md): the banks,
  `margin.json`, `limits.json`, and the stored predictions of S, S-L, and the demand-output arm.
  From [experiment 729](../00729-msa-d8-neural-untangler-synthetic-known-answers/result.md):
  `PRIVATE/splits.md`. D8's cleaned reading, the BSP A meter, the cleaned power of D3, D5, and D14,
  and the switching logs through experiments 722 and 729's loaders.
