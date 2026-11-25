---
id: "00729"
slug: msa-d8-neural-untangler-synthetic-known-answers
title: >-
  Can a neural network trained on synthetic days with a known answer turn a day of D8's unsigned
  apparent-power reading into a calibrated probabilistic estimate of D8's signed power, and which
  extra input carries the information?
theme: solar-disaggregation
also_relevant_to:
  - gross-demand-simulation
status: finished
verdict: not supported
kind: other
size: >-
  Anything else. (1) It stores about 50 fitted neural networks, synthetic training and test banks,
  per-half-hour predictions, and score tables, all private; (2) more than one design defensibly
  tests
  the hypothesis (three neural-network architectures, three output heads, and a ladder of inputs),
  so
  all are arms with a pre-registered selection rule; (3) it imports experiment 694's tracker as a
  baseline and reads the stored frame of experiment 39 and the engine window of experiment 194,
  whose callers cannot be named without searching the repository; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Reviews, in order: this Opus plan, a Sonnet plan
  review, Sonnet code, a fresh Opus code review before any first run, a real-file smoke test, then
  the results, write-up, and prose reviews.
hypothesis: >-
  A neural network that reads 48 hours of D8's apparent-power reading together with satellite
  sunshine, the half-hour steps of the bulk supply point meter above D8, and the calendar inputs
  and coincident observed temperature, wind speed, and precipitation of a demand forecast, and that
  is trained only on synthetic days built from D8's own readings before its gas engine was
  connected, returns for each half-hour of the middle
  24 hours a two-lobe distribution of D8's signed power. On held-out synthetic days the neural
  network's signed continuous ranked probability score (CRPS) beats the best rule that does not
  learn from the day's shape, and on real days before the engine was connected the neural network
  does not invent reverse flow where there is none.
prediction: >-
  Written before any code exists. Arm D on the chosen architecture beats the better of the two
  demand-climatology rules (B1, B2) by 1 to 4 points of D8's capacity in signed CRPS over all
  synthetic test half-hours, and almost all of the gain sits on reverse-flow half-hours. On the
  input
  ladder, adding sunshine (C3) and adding the bulk supply point steps (C4) each lower the signed
  CRPS by at least 0.3 points. The C3 gain rests on solar source S1, whose output shares no
  satellite product with the sunshine input. The C4 gain sits mostly in the worlds with an engine.
  Adding the calendar, temperature, wind-speed, and precipitation inputs (C5) changes the signed
  CRPS by under 0.2 points. The
  quantile head (C6) has an 11-quantile pinball loss at least 0.2 points higher than the two-lobe
  head. The 48-hour window has a signed CRPS lower than the 24-hour window (C11). The sign-only arm
  D-sign, which outputs one probability of reverse flow per half-hour and fixes its two lobes at
  plus
  and minus the reading with the reactive floor removed, is a simpler function that may receive less
  supervisory signal. D-sign scores a signed CRPS higher than arm D (C13, a loss) by 0.01 to 0.10
  points, always below the 0.25-point floor of the margin m, and its reliability term of P(X<0) is
  within 0.005 of arm D's. This sentence was fixed before any fit exists. The harm checks
  pass: arm D's signed CRPS exceeds P0f's by 0.01 to 0.08 points on import half-hours, against a
  limit of 0.12 points. The likeliest miss is the sign calibration bar on solar-only worlds. The
  second likeliest is the lobe coverage bar on half-hours within 3 points of zero. The prediction
  fails if C1 does not show a gain, or if the ladder order differs.
decision_rule: >-
  Every bar and constant below is a named `Final` of `e00729_core.py`, fixed before any fit or score
  exists. Headline contrast C1 is arm D minus B* (the better of B1 and B2 on the validation months)
  in signed CRPS over all synthetic test half-hours, read at 99.583% (Bonferroni, 1 - 0.05 / 12)
  from a bootstrap that resamples blocks of 7 consecutive base dates and the segment draws. The
  margin m is the larger of 0.25 points and twice the standard deviation over seeds 0 to 4 of arm
  D's validation-month signed CRPS, written once to a file that refuses to be overwritten before any
  test contrast is read. Harm limits, derived from the synthetic world's own mix in the section
  "Verdict table": H1, arm D minus P0f (P0's point forecast given the same 0.5-point lobe-width
  floor) on synthetic import half-hours, at most +0.12 points of signed CRPS and at most +0.25
  points of normalised mean absolute error of the median; H2, the same two limits on the real
  held-out base days, with at most 0.5% of those half-hours at P(X<0) above 0.5. H2 and R1 are weak
  checks. Verdict, first row that matches: (0) stopped, no verdict; (1) `invalid`; (2) `failed`; (3)
  `not supported`, H1 or H2 fails; (4) `supported`, C1 at or below -m with an upper end below zero,
  and the sign calibration and lobe coverage bars pass; (5) `partly supported`, C1 passes and one of
  those two bars fails; (6) `small gain`, C1's upper end below zero and its point above -m; (7)
  `inconclusive`, C1's interval includes zero and its half-width is at least m; (8) `not
  supported`, every other case. The section "Verdict table" gives every row in full. The experiment
  ports nothing, so no smallest gain applies beyond m. C13 (D-sign minus arm D) is a thirteenth
  reading that enters no verdict row and is not among the 12 that set the Bonferroni level.
benchmark: >-
  P0 (the reading read as signed import) and P0f (P0's point forecast given the same lobe-width
  floor as the neural networks, the reference for the harm checks); B1 (a demand climatology of D8's
  base days that sets the probability of reverse flow to the world's injected share times the
  probability that demand lies below the reading); B2 (the same rule with a regression of base
  demand
  on calendar and temperature);
  experiment 694's M2 tracker on the engine-only test segments; arm A (the reading alone) as the
  no-hint neural network.
row_set_id: >-
  e00729_synthetic_d8_v1: the 48 half-hours of each UTC day of 8 held-out calendar months of D8's
  base days, each carried by 8 independent synthetic segment draws, scored in the middle of a
  48-hour
  window. Real checks read the same months without injection, the metered engine record's days, and
  D8's eras up to 2026-06-30. No valid time on or after 2026-07-01 is read. No forecast is made.
metric: >-
  Signed CRPS in points of D8's capacity (exact for the two-lobe distribution), the 11-quantile
  pinball loss, and the mean absolute error of the median normalised by D8's capacity, each on all,
  import, and reverse-flow half-hours; the reliability term of the Brier score of P(X<0); lobe
  coverage; q05 and q95 coverage; the engine head's mean absolute error and running F1 on the
  metered engine record; false-alarm shares on real days.
best_arm: >-
  No arm. Arm D-quantile has the lowest import score of the neural networks (0.227 points against
  P0f's 0.118), and arm D, the headline arm, scores +0.258 points above P0f on import rows against a
  limit of 0.12. Against B* on all rows, arm D is 33.878 points better (interval -40.018 to
  -27.593).
headline: >-
  Arm D beats the demand-climatology rule B* by 33.878 points of D8's capacity in signed CRPS on
  synthetic days (interval -40.018 to -27.593), but it scores 0.258 points above the point reading
  P0f on synthetic import half-hours, against a pre-registered limit of 0.12 points, so the
  verdict is not supported.
lessons: >-
  The neural network beats the demand-climatology rule B1 by 33.878 points, almost all of it on
  import half-hours where B1 puts a probability near 0.40 on reverse flow, and the gain is measured
  on a synthetic world only. The two-lobe head with its 0.5-point floor scores worse than the
  reading on import half-hours, and so do the sign-only and 11-quantile heads, although both score
  lower than arm D on the point means. The BSP steps did not lower the score (C4 is +0.149 points).
  The sign calibration bar fails on four bins, and no real signed truth exists for D8. Every number
  is interim.
take_home_message: >-
  A neural network trained on synthetic days finds the direction of D8's power flow much better
  than rules built from D8's past demand, but the neural network adds error on the half-hours when
  D8 imports power, and D8 imports on almost all half-hours. The harm check set in advance fails,
  so the experiment does not support using this neural network, and no real signed record exists to
  confirm the gain. The sign-only and quantile versions score lower than the headline neural
  network on the point means and still fall short of simply reading the meter on import
  half-hours.
advice:
  - >-
    Check the harm on import half-hours against the reading with the same lobe floor before reading
    any reverse-flow gain, and fail the arm if the excess passes the limit (H1).
  - >-
    Do not read a gain over a demand climatology as evidence for the neural network, because
    B1 scores 34.664 points over all rows and any neural network beats it by more than 30 points
    (C1, Table 3).
  - >-
    Test an output head other than the two-lobe head with a 0.5-point floor on the same synthetic
    banks before choosing a head, because the sign-only head and the 11-quantile head score lower
    than arm D on the point means of all, import, and reverse-flow rows (C6, C13, Table 3).
  - >-
    Do not expect the bulk supply point steps to lower the score when they are added to the
    sunshine inputs of a synthetic-day neural network, because C4 is +0.149 points (interval 0.068
    to 0.455).
  - >-
    Report the calibration of P(X<0) by reading band, because arm D over-states reverse flow in the
    bins from 0.1 to 0.8 and no arm is sure of the sign where the reading is under 1 MVA (the
    ambiguity read-out, Table 5).
production_status: not-for-production
adopted_in: none
---

**We ask whether a neural network can turn a day of D8's unsigned reading into an estimate of D8's
signed power, with honest uncertainty.** D8 is a primary substation (a "Primary"), which steps the
voltage down to 11 kV. D8's meter reports apparent power in megavolt-amperes (MVA), a magnitude with
no sign. A gas engine and a large solar farm now connect below D8, so on some half-hours power may
flow out of D8 towards the grid (reverse flow), and the meter shows that export as a positive
reading. No meter tells us D8's signed power on any real day, so we cannot train on real answers.
We therefore build synthetic days with a known answer: we take D8's real readings from before the
engine was connected, when D8 only imported, subtract a simulated engine and a simulated solar
output, and fold the result into a magnitude the way the meter does. The neural network learns to
undo the fold. We then test the neural network on synthetic days that training never saw, on real
days before the engine where the answer is the reading itself, and on the real days when the
engine's output was metered.

**The page began as the plan, written before any code existed, and the plan sections below are kept
as registered.** The design follows the denoising autoencoder and the other neural networks that
[Kelly and Knottenbelt (2015)](https://doi.org/10.1145/2821650.2821672) used to split a household's
total power into its appliances, and this experiment trains those neural networks on randomised
synthetic data in the spirit of domain randomisation ([Tobin et al.
(2017)](https://doi.org/10.1109/IROS.2017.8202133)).

**We tested a neural network that rebuilds the sign of D8's power from the magnitude the meter
shows.** The neural network learned only from synthetic days, for which the answer is known. On
held-out synthetic days, the neural network scored far better than the best rule that ignores the
shape of the day, mostly because the rule gives ordinary import half-hours a high probability of
reverse flow. On the half-hours where D8 imports power, which are almost all half-hours, the neural
network scored worse than simply reading the meter, by more than the limit we set before the first
fit. The rules of the plan therefore give the verdict `not supported`. Every number below is
interim.

## Result

**Superseded first run.** The first run measured every reading on this page with the S2 solar worlds
silent: the satellite-sky rule of `farm_set` never passed, so no day of source S2 held any solar
output, and 24 of the 64 test-bank segments (37.5% of the test rows) were solar or engine-and-solar
worlds with no solar output. The headline readings of that run were verdict row 3 and an H1 excess
of +0.258 points, and both are superseded. Every other number on this page was measured on the same
silent worlds and is superseded too, and the manager ruled that the whole chain is rerun from the
banks, starting at gate attempt 1 with the registered simulator ranges. The superseded outputs move
to `superseded_run_1` in the experiment's private data folder before the rerun, and no stage of the
rerun reads them.

**The code changed after the first run, and the table gives each file's git blob before and after.**
The commit before the changes is `c9d963946`. The changes are these: the satellite sky of a farm-day
now counts when 40 of 48 (83%) of its sun-up half-hours are finite, instead of 40 half-hours of the
whole day; the five simulator ranges that the first run re-tuned are back at their registered values
(solar capacity 5 to 150 points); every stage refuses to run unless the data folder carries a run
marker that matches the code, and the gate refuses a third attempt. The tests and the real-file
smoke test gained checks of the real `farm_set`, of an S2 draw on clear days, and of the guard.

| File | Blob before | Blob after |
|---|---|---|
| `e00729_cli.py` | edf396f8e0 | 7f4dd1e69c |
| `e00729_core.py` | 18c72b6967 | a1b3ef7d71 |
| `e00729_data.py` | 9b1006eca3 | 4f835d466e |
| `e00729_sim.py` | 34aae11579 | 81e1263b3c |
| `e00729_stages.py` | 9245f09b57 | b4db685dbf |
| `smoke_real_files.py` | 828d71b22a | ab8cc661ab |
| `synthetic_world.py` | 0ad4b2de69 | de09a505a5 |
| `test_e00729_core.py` | 25e335e00a | 254ae0021d |
| `test_e00729_farm_set.py` | new | 00bae6565b |
| `test_e00729_run_guard.py` | new | 8eb655f929 |
| `test_e00729_sim.py` | 1f2ed1895d | 54af40f44d |

**These sections hold numbers from the superseded run and are rewritten after the rerun.** They are
"Result" and its subsections, "Time series of the estimates" (figures S1 to S12), "The event generator
against real data" (the ranges in the caption of figure E1, and the findings under figures E3 and E4),
"The solar and engine heads against the true series", "Checks", the realism gate and timing records
under "The realism gate before any fit" and "Compute and gates", the figures 1 to 9, and the front
matter (verdict and status). The registered plan sections keep their text.

**The verdict is `not supported`, and verdict row 3 sets it: the harm check on import half-hours
(H1) fails while the real-day check (H2) passes.** On the synthetic test half-hours where D8 imports
power, arm D scores a signed continuous ranked probability score (CRPS) of 0.376 points and the
reference P0f scores 0.118 points. The
excess is +0.258 points against the pre-registered limit of 0.12 points. The normalised error of the
median is also 0.379 points above P0f's, against its limit of 0.25 points. Both limits fail. The
manager ruled that no simulator range is re-tuned and no neural network is refitted after this
reading.

### What this means

**Within the synthetic world the neural network scores far better than the demand-climatology rule
B1, and the neural network adds error where D8 only imports.** C1, arm D minus B\*, is −33.878
points of D8's capacity (interval −40.018 to −27.593). Almost all of that gain sits on import
half-hours: import rows give 33.65 points of it and reverse-flow rows 0.23 points. C1 is large
because B1 puts a mean P(X<0) of 0.398 on ordinary half-hours above 3 MVA (Table 5), where the
observed share is 0.011. On reverse-flow half-hours arm D scores 18.385 points against 28.550 for
B\*. Each half-hour of arm D carries two Gaussian lobes with a spread of at least 0.5 points, and
P0f carries the same floor, so the excess over P0f measures what the neural network adds on import
half-hours. This experiment did not test why the neural network adds that error. On the point means,
the sign-only arm D-sign and the 11-quantile arm D-quantile score lower than arm D on all rows,
import rows, and reverse-flow rows, yet their import scores (0.265 and 0.227) remain above P0f's
0.118. Only C6 and C13 have intervals, and both are on all rows. No interval exists on import or
reverse-flow rows, and arm D averages 5 seeds where D-sign and D-quantile average 3. Both
comparisons are report-only and are not a verdict reading. The point means do not show which head to
use.

**The result has four limits.**

- **The gains are measured on the synthetic world only.** The simulators are our descriptions of the
  engine and the solar farm, and the realism gate checks two shares and a bulk supply point (BSP)
  slope, not the whole distribution of the days.
- **No real signed truth exists for D8.** The two real checks that have an answer, H2 and K1, each
  cover one narrow case: days before the engine was connected, and the days of the metered engine
  record.
- **The two-lobe head with its 0.5-point floor loses on import half-hours.** Arm D scores 0.258
  points above P0f there, and every other neural network also scores above P0f.
- **The experiment does not say what to do next.** The options left open are a different output
  head, a different loss weight on import half-hours, and accepting the harm in exchange for the
  gain on reverse flow. The result review chooses among these options.

### The bars and harm checks

**Table 1 gives each bar's reading: H1 and the sign calibration bar fail, and the other checks
pass.** Source: the "Bars and harm checks" section of the
printed report, and `verdict.json`.

*Table 1. Bars and harm checks.*

| Check | Reading | Limit | Result |
|---|---|---|---|
| H1, arm D minus P0f on synthetic import rows, signed CRPS | +0.258 points | at most 0.12 | fail |
| H1, the same, normalised error of the median | +0.379 points | at most 0.25 | fail |
| H1, false-sign share on synthetic import rows | 0.00908 | none stated | printed only |
| H2, arm D minus P0f on real held-out base days, signed CRPS | +0.024 points | at most 0.12 | pass |
| H2, the same, normalised error of the median | +0.114 points | at most 0.25 | pass |
| H2, share of half-hours with P(X<0) above 0.5 | 0.0 | at most 0.005 | pass |
| Sign calibration, reliability term | 0.000507 | at most 0.01 | pass |
| Sign calibration, largest gap in a bin of at least 200 half-hours | 0.130 | at most 0.1 | fail |
| Lobe coverage | 0.987 | at least 0.85 | pass; mean lobe width 3.245 points |
| Reachability, reverse-flow half-hours | 2,019 in all, 303 in the BSP months | at least 2,000 in all and 200 in the BSP months | pass |

**The reachability count clears its limit by 19 half-hours.** The 2,019 reverse-flow half-hours are
under half the plan's projection of about 4,400, and the 303 in the BSP months are under a third of
the projected 1,100.

**The sign calibration bar fails on four bins, and row 3 sits before the rows that read it.** Four
bins of at least 200 half-hours have a gap above 0.1: the bins of forecast probability 0.2 to 0.3
(739 half-hours, mean forecast 0.249, observed share 0.129, gap 0.120), 0.3 to 0.4 (700, 0.348,
0.219, gap 0.130), 0.5 to 0.6 (890, 0.549, 0.440, gap 0.108), and 0.6 to 0.7 (362, 0.639, 0.536,
gap 0.103). Figure 6 shows the bins. In the bins from 0.1 to 0.8 the observed share is below the
mean forecast, so the neural network over-states the probability of reverse flow in that range.

### The 12 counted contrasts and the report-only contrast

**Only C1 sets a verdict word, and C1's interval lies far below −m (−0.250 points).**
Table 2 lists every contrast. A negative difference is a gain of the first arm. C6 is read on the
pinball loss and the others on the signed CRPS. The BSP contrasts C4 and C12 are read on the two BSP
test months, and by world type. Source: the "Contrasts" table of the printed report.

*Table 2. The contrasts, in points of D8's capacity.*

| Contrast | Arms | Reading | Point | Lower | Upper |
|---|---|---|---|---|---|
| C1 | D minus B\* (verdict) | all | -33.878 | -40.018 | -27.593 |
| C2 | A minus P0f | all | -1.677 | -5.502 | 0.102 |
| C3 | B minus A | all | -0.370 | -1.024 | -0.018 |
| C3 | B minus A | source_S1 | -2.574 | -8.256 | 0.628 |
| C3 | B minus A | source_S2 | -0.025 | -0.126 | 0.078 |
| C4 | C minus B | bsp_all | 0.149 | 0.068 | 0.455 |
| C4 | C minus B | bsp_engine | 0.128 | 0.033 | 0.408 |
| C4 | C minus B | bsp_solar | 0.203 | -0.297 | 2.767 |
| C4 | C minus B | bsp_both | 0.153 | -0.157 | 0.594 |
| C5 | D minus C | all | -0.059 | -0.197 | 0.035 |
| C6 | D-quantile minus D (pinball loss) | all | -0.307 | -0.629 | -0.124 |
| C7 | E minus D | all | -0.056 | -0.119 | -0.010 |
| C8 | F minus D | all | 0.006 | -0.097 | 0.123 |
| C9 | G1 minus D | all | -0.112 | -0.234 | -0.023 |
| C10 | G2 minus D | all | 0.008 | -0.075 | 0.111 |
| C11 | D-24 minus D | all | -0.014 | -0.133 | 0.073 |
| C12 | H minus A | bsp_all | 0.100 | -0.025 | 0.255 |
| C12 | H minus A | bsp_engine | 0.077 | -0.163 | 0.252 |
| C12 | H minus A | bsp_solar | 0.083 | -0.574 | 1.087 |
| C12 | H minus A | bsp_both | 0.172 | -0.172 | 0.900 |
| C13 (report only) | D-sign minus D | all | -0.340 | -0.749 | -0.099 |

**At least eight pre-registered predictions did not hold.** The plan predicted that almost all of
C1's gain sits on reverse-flow half-hours, and the opposite holds (see "What this means"). C1 is a
gain of 33.878 points against a predicted 1
to 4 points, because B1 scores 34.664 points over all rows (Table 3), far worse than every neural
network, so the size of C1 depends on a weak yardstick. The prediction that the D-sign reliability
term is within 0.005 of arm D's is not read in the report. C4, the BSP steps added to arm B, is
+0.149
points (interval 0.068 to 0.455), a loss, against a predicted gain of at least 0.3 points. C6 shows
the 11-quantile head 0.307 points better than the two-lobe head on the pinball loss, where the plan
predicted it at least 0.2 points worse. C13 shows D-sign 0.340 points better than arm D (interval
−0.749 to −0.099), where the plan predicted a loss of 0.01 to 0.10 points. C11 is −0.014 points with
an interval that includes zero, so the predicted advantage of the 48-hour window is not shown. The
H1 excess was predicted at 0.01 to 0.08 points and is 0.258. C3, sunshine added to arm A, is −0.370
points (interval −1.024 to −0.018), a gain of the predicted size, but its S1 reading is −2.574
points with an interval of −8.256 to 0.628, so the prediction that the gain rests on source S1 is
not shown. C5 is −0.059 points, inside the predicted range of under 0.2.

![The 12 counted contrasts and the report-only C13](figure_5_contrasts.svg)

*Figure 1. Each contrast of Table 2 at the Bonferroni level, with the margin m marked.*

### Scores of every arm and baseline

**Every arm beats B\* by more than 30 points over all rows, and no neural network beats P0f on
import rows.** Table 3 gives the signed CRPS and the normalised error of the median, in points of
D8's capacity, on all rows, import rows, and reverse-flow rows. P0 and P0f use the signed reading as
a point forecast. P0f has the same 0.5-point lobe floor as the neural networks. M2 is scored on all
94,080 rows, including solar worlds, although the plan scored M2 on the engine-only segments because
M2 has no solar term (a departure from the plan; the M2 figures are as scored). Source: the "Scores"
table of the printed report.

*Table 3. Signed CRPS and normalised error of the median, in points of D8's capacity.*

| System | CRPS, all | CRPS, import | CRPS, reverse | Median error, all | Median error, import | Median error, reverse |
|---|---|---|---|---|---|---|
| P0 | 2.580 | 0.015 | 112.687 | 2.580 | 0.015 | 112.687 |
| P0f | 2.673 | 0.118 | 112.405 | 2.580 | 0.015 | 112.687 |
| B1 | 34.664 | 34.806 | 28.550 | 59.211 | 59.913 | 29.078 |
| B2 | 39.190 | 39.384 | 30.836 | 68.282 | 69.137 | 31.561 |
| B\* | 34.664 | 34.806 | 28.550 | 59.211 | 59.913 | 29.078 |
| M2 | 3.097 | 0.891 | 97.837 | 3.545 | 1.142 | 106.728 |
| A | 0.996 | 0.378 | 27.552 | 1.398 | 0.385 | 44.872 |
| B | 0.626 | 0.354 | 12.307 | 0.774 | 0.399 | 16.890 |
| C | 0.845 | 0.339 | 22.587 | 1.068 | 0.344 | 32.125 |
| D | 0.786 | 0.376 | 18.385 | 0.985 | 0.394 | 26.371 |
| H | 0.889 | 0.369 | 23.229 | 1.077 | 0.319 | 33.637 |
| D-quantile (Dq) | 0.470 | 0.227 | 10.927 | 0.589 | 0.301 | 12.951 |
| D-24 | 0.772 | 0.417 | 16.027 | 0.964 | 0.475 | 21.985 |
| E | 0.730 | 0.343 | 17.362 | 0.884 | 0.346 | 23.989 |
| F | 0.791 | 0.421 | 16.679 | 0.952 | 0.437 | 23.069 |
| G1 | 0.674 | 0.306 | 16.497 | 0.831 | 0.323 | 22.668 |
| G2 | 0.794 | 0.412 | 17.213 | 0.936 | 0.392 | 24.290 |
| D-sign | 0.445 | 0.265 | 8.174 | 0.450 | 0.207 | 10.894 |

**On import rows P0f scores 0.118 points and no neural network scores below 0.227.** D-quantile
(0.227), D-sign (0.265), and G1 (0.306) score lowest, and arm D scores 0.376. On reverse-flow rows
D-sign scores 8.174 points, D-quantile 10.927, and arm D 18.385, against 28.550 for B\*, 97.837 for
M2, and 112.405 for P0f.

![Signed CRPS of the arms and baselines](figure_4_scores.svg)

*Figure 2. Signed CRPS of P0f, B\*, and the arms A, B, C, D, and H on all, import, and reverse-flow
rows
(smaller is better).*

**The q05 and q95 coverage of every neural network is at least 0.94, consistent with the plan's note
that the lobe floor makes coverage easy.** Source: the "q05 and q95 coverage" line of the printed
report. Arm D has q05 coverage of 0.996 and q95 coverage of 0.995. D-quantile has the lowest q95
coverage of the arms, 0.943.

### The architecture screen, B\*, and the margin

**The bidirectional long short-term memory neural network (bilstm) was chosen on the validation
months, and
B1 was chosen as B\*.** Source: the header of the printed report, and `verdict.json`. B1 is the
demand climatology that sets the probability of reverse flow to the world's injected share times the
probability that demand lies below the reading. The margin m is 0.250 points, the floor of its rule.
On the validation months, arm D scores 0.868 points with the bilstm, 0.941 with the U-net, and 3.083
with the denoising autoencoder (3 seeds each; source: `selection.json`). Figure 3 shows the scores.

![Validation signed CRPS of the three architectures](figure_9_architectures.svg)

*Figure 3. Mean signed CRPS on the validation months for the bilstm, the denoising autoencoder, and
the U-net; the smaller is better.*

### The realism gate and the timing gate

**The realism gate passed at its second attempt, after the one permitted re-tune of the simulator
ranges.** Source: "Realism gate" in the printed report. Table 4 gives the share checks. The BSP-step
regression passed on all three parts (slope overlap, standard error inside, standard deviation
inside), with 8 finite fits, 8 synthetic stretches used of 19, and 1 real stretch of 1,488
half-hours. The real slope lies 7.427 standard deviations below the placebo slopes
(`real_slope_z_vs_placebo` = −7.427). The shares pass narrowly: the synthetic share over the real
share is 0.59 for net daily minima, 0.61 for solar daily minima, and 0.56 for solar half-hours,
against a lower edge of 0.5. The gate checks these shares and the BSP step slope, and it checks no
other feature of the synthetic days.

*Table 4. Real and synthetic shares in the realism gate.*

| Series | Real daily-minimum share | Synthetic | Real half-hour share | Synthetic |
|---|---|---|---|---|
| Net | 0.172 | 0.101 | 0.012 | 0.012 |
| Solar | 0.536 | 0.328 | 0.045 | 0.026 |

![The realism gate](figure_3_gate.svg)

*Figure 4. Share of days with a daily minimum below 1.0 MVA, synthetic against real, for the net and
solar series.*

**The timing gate projects 1.102 GPU-process-hours at the epoch cap and 0.386 at early stopping,
against a budget of 10.0.** Source: "Timing gate" in the printed report. The gate measured 0.350
seconds per epoch for the denoising autoencoder, 0.409 for the U-net, and 0.437 for the bilstm. The
50 fits spent 1.107 GPU-process-hours in all, the sum of the `seconds` of their fit records.

### Real-day checks

**H2, K1, and K2 pass; R2 and R3 are reports with no bar.** Source: `verdict.json`, key `real`. All
numbers in this list are aggregates.

- **K1, the engine head against the metered engine record:** the head's mean absolute error is 0.432
  times that of an all-zero engine (bar: at most 0.5), its running F1 is 0.781 against 0.678 for M2
  (bar: at least M2's), and arm D's signed CRPS on those days is 1.370 points against 32.113 for
  B\*. The simulator's ranges were set from published descriptions of the same record, so K1 is not
  independent of the simulator.
- **K2, the BSP placebo:** the share of half-hours with P(X<0) above 0.5 is 0.0 (bar: at most
  0.005), and the engine head's median exceeds 10 points on 0.00616 of half-hours (bar: at most
  0.02).
- **R2, re-fold consistency:** the share of half-hours whose re-folded median lies within 2 points
  of the reading is 0.930 in the net era, 0.795 in the solar era, 0.723 in the fade era, and 0.999
  on the K2 days. The neural network is trained to re-fold, so R2 is nearly guaranteed.
- **R3, the share of half-hours with P(X<0) above 0.5 by era:** 0.040 in the net era, 0.157 in the
  solar era, and 0.217 in the fade era. Figure 5 gives the share by UTC hour. The reading says where
  the neural network places reverse flow and does not show that reverse flow occurred.

![Real eras by UTC hour](figure_8_eras.svg)

*Figure 5. Real eras by UTC hour: the share of half-hours with P(X<0) above 0.5, beside the share of
half-hours with a reading below 1.0 MVA.*

### The shifted-mix read-out and the ambiguity read-out

**The shifted-mix read-out shows no sign that arm D memorised the training mix.** On the shifted
bank, which has an injected share of 0.50 against 0.85 in the test banks, arm D scores a signed CRPS
of 0.741 points with a
reliability term of 0.0000943, a largest bin gap of 0.0863, and 0.0218 of half-hours with P(X<0)
above 0.5. B\* scores 13.405 points with a reliability term of 0.0747 and a largest gap of 0.437.
Source: `verdict.json`, key `shifted`.

**Where the reading is under 1 MVA, no arm is sure of the sign, and B\* is the arm that is confident
and wrong.** Table 5 gives B\*, arm D, and D-sign by reading band, from the "Ambiguity read-out"
table of the printed report. "Confident share" is the share of half-hours with P under 0.1 or over
0.9. In the band under 0.3 MVA, 0.463 of the half-hours are reverse flow. B\* is confident on 0.810
of those half-hours with a mean P of 0.096, so B\* is confidently wrong. Arm D has a mean P of 0.529
and is
confident on 0.003. D-sign has a mean P of 0.482. By the truth's size, the observed share of reverse
flow is 0.437 where |X| is under 1 times the reactive floor Q̂ of 0.1 MVA, 0.474 for 1 to 3, 0.402
for 3 to 10, and 0.017 above 10. The ambiguity read-out has no bar.

*Table 5. The ambiguity read-out by reading band.*

| System | Reading band | Rows | Mean P(X<0) | Observed share | Reliability | Confident share |
|---|---|---|---|---|---|---|
| B\* | under 0.3 MVA | 326 | 0.096 | 0.463 | 0.1930 | 0.810 |
| B\* | 0.3 to 1 MVA | 911 | 0.020 | 0.405 | 0.1485 | 1.000 |
| B\* | 1 to 3 MVA | 2880 | 0.020 | 0.222 | 0.0408 | 1.000 |
| B\* | over 3 MVA | 89963 | 0.398 | 0.011 | 0.2223 | 0.212 |
| D | under 0.3 MVA | 326 | 0.529 | 0.463 | 0.0189 | 0.003 |
| D | 0.3 to 1 MVA | 911 | 0.514 | 0.405 | 0.0208 | 0.002 |
| D | 1 to 3 MVA | 2880 | 0.326 | 0.222 | 0.0139 | 0.124 |
| D | over 3 MVA | 89963 | 0.014 | 0.011 | 0.0002 | 0.985 |
| D-sign | under 0.3 MVA | 326 | 0.482 | 0.463 | 0.0162 | 0.055 |
| D-sign | 0.3 to 1 MVA | 911 | 0.435 | 0.405 | 0.0094 | 0.095 |
| D-sign | 1 to 3 MVA | 2880 | 0.248 | 0.222 | 0.0023 | 0.461 |
| D-sign | over 3 MVA | 89963 | 0.011 | 0.011 | 0.0001 | 0.992 |

![Reliability of P(X<0) for arm D](figure_7_reliability.svg)

*Figure 6. The reliability diagram of arm D's P(X<0): forecast probability against the observed
share of reverse flow.*

### Reviews and pre-run gates

**The planner ran `e00729_plan_calcs.py` before the Opus plan review, the review re-derived the
numbers the script produced, and no verdict row changed.** The plan paragraph that records the
breach is unchanged in "The plan as registered". Figures 1, 2, and 6 of the planned list were not
written by the run, and figures 3, 4, 5, 7, 8, and 9 are in this section. The Opus result review
(the run's first review of this section) found no leakage and confirmed verdict row 3.

## Time series of the estimates

**Twelve figures show single 48-hour windows from the stored predictions, and each day is picked by
a fixed rule.** The script `e00729_series_figures.py` (stage `series`) reads the stored test-bank
predictions of each arm at seed 0, and the real held-out days for the last two figures. It fits
nothing and re-scores nothing. The middle 24 hours of each window are the scored day, and the 12
hours either side (grey blocks) are context that the neural networks read. The top row holds the
true signed power (black, synthetic days only), the folded reading (grey), and each arm's median
with a 5% to 95% band. The truth and the reading are drawn wide beneath the arm lines. The row below
holds each arm's P(X<0), with a dashed line at 0.5. The x axis is the hour of the day in UTC, and no
figure carries a calendar date. Each figure shows one day at seed 0, so none is a sample of days or
a seed mean, and every number is interim.

**The numbers in the paragraphs below come from a read-only snippet over the stored test-bank
predictions.** The snippet reuses the picking functions of the script and prints, for each synthetic
figure's day, the lowest truth, the lowest median of each arm, the highest P(X<0), the number of
half-hours with P(X<0) above 0.5, and the mean width of the 5% to 95% band. The real-day paragraphs
describe what the plots show. Medians are in points of D8's capacity.

**The figure index lists the figures in the order of this section.**

- Figure S1: An evening engine run that reverses the flow.
- Figure S2: A morning engine run that reverses the flow.
- Figure S3: Solar output alone reverses the flow at midday.
- Figure S4: Engine and solar output together.
- Figure S5: A day on which the reading comes close to zero and the sign is hard to tell.
- Figure S6: A plain import day with no injection.
- Figure S7: The import day on which arm D does worst against P0f.
- Figure S8: A month with real BSP steps: arm C beside arm B.
- Figure S9: The 11-quantile head beside the two-lobe head.
- Figure S10: The sign-only head beside the two-lobe head.
- Figure S11: A real day before the engine was connected, typical false alarm.
- Figure S12: A real day before the engine was connected, largest false alarm.

**On the evening engine day, the medians of arms A, B, and D stay close to zero while the truth
drops to −7.8 points.** The truth is below zero on 5 half-hours. The median of arm A goes below zero
on 3 of them (lowest −1.0), arm B on 2 (lowest −3.3), and arm D on 1 (lowest −0.3). P(X<0) peaks at
0.56 for arm A, 0.58 for arm B, and 0.51 for arm D, and is above 0.5 on 3, 2, and 1 half-hours.
Outside the engine run all three arms sit on the reading with P(X<0) near zero. The band of arm D is
3.8 points wide on average.

![An evening engine run that reverses the flow](series_01_evening_engine.svg)

*Figure S1. An evening engine run that reverses the flow. Rule for the day: an engine-only day with
reverse flow whose engine output is centred between 17:00 and 23:00 UTC; the day with the median
depth of reverse flow. Arms at seed 0; band 5% to 95%.*

**On the morning engine day, the arms follow the direction of the run but not its depth.** The truth
falls to −16.0 points on 11 half-hours. The lowest medians are −1.7 for arm A, −8.4 for arm B, and
−4.8 for arm D. P(X<0) is above 0.5 on 9 half-hours for arms A and D and on 6 for arm B, and peaks
at 0.67 for arm D. Arm B's P(X<0) falls back below 0.5 before the run ends.

![A morning engine run that reverses the flow](series_02_morning_engine.svg)

*Figure S2. A morning engine run that reverses the flow. Rule for the day: an engine-only day with
reverse flow whose engine output is centred between 06:00 and 12:00 UTC; the day with the median
depth of reverse flow. Arms at seed 0; band 5% to 95%.*

**On the solar-only day, no arm puts its median or P(X<0) on the reverse flow.** The truth reaches
−74.0 points on 11 half-hours. The lowest medians are +3.3, +10.6, and +11.6 for arms A, B, and D,
so no median goes below zero, and P(X<0) peaks at 0.43, 0.30, and 0.43. The band of arm D is 26.4
points wide on average and reaches below −100 near noon, so the band covers the truth while the
median and P(X<0) do not show it.

![Solar output alone reverses the flow at midday](series_03_solar_midday.svg)

*Figure S3. Solar output alone reverses the flow at midday. Rule for the day: a solar-only day with
reverse flow whose solar output is centred between 10:00 and 14:00 UTC; the day with the median
depth of reverse flow. Arms at seed 0; band 5% to 95%.*

**On the engine-and-solar day, the arms flag reverse flow on most of the right half-hours and also
raise P(X<0) on import half-hours.** The truth reaches −53.3 points on 10 half-hours. The medians of
arms B and D reach −67.3 and −59.9, and the median of arm A reaches −2.4. P(X<0) is above 0.5 on 9,
9, and 10 half-hours for arms A, B, and D. Near 14:00 the medians of arms B and D dip to about −60
where the truth is about −15, deeper than the truth at that half-hour. The mean P(X<0) on the import
half-hours of the day is 0.22, 0.17, and 0.21.

![Engine and solar output together](series_04_engine_and_solar.svg)

*Figure S4. Engine and solar output together. Rule for the day: an engine-and-solar day with reverse
flow in which both outputs reach at least 2 points; the day with the median depth of reverse flow.
Arms at seed 0; band 5% to 95%.*

**On the ambiguous day, arm B puts its median at −74.3 points for about two hours although the truth
never goes below −1.9.** The day has 4 reverse-flow half-hours. The P(X<0) of arm B peaks at 0.70
and is above 0.5 on 7 half-hours, against 4 and 3 half-hours for arms A and D. The band of arm D is
37.7 points wide on average and reaches below −90 around 13:00. Near a reading of zero, P(X<0) is
the mass of two lobes at plus and minus the reading, so a value near 0.5 there follows from the
geometry, whatever an arm has learned about the sign.

![A day on which the reading comes close to zero and the sign is hard to tell](series_05_ambiguous.svg)

*Figure S5. A day on which the reading comes close to zero and the sign is hard to tell. Rule for
the day: a day with at least 4 half-hours of reading under 0.3 MVA; the day with the median number
of such half-hours. Arms at seed 0; band 5% to 95%.*

**On the typical no-injection import day, no arm invents reverse flow.** The truth and the reading
stay above 57.0 points. The P(X<0) of arm D peaks at 0.00 and its mean is 0.001, and the P(X<0) of
arm A peaks at 0.01. The band of arm D is 1.7 points wide on average, close to the 1.6 points of
P0f, which is the dashed line under the others. This day is the median of its rule, so it shows the
typical case and not the worst.

![A plain import day with no injection](series_06_plain_import.svg)

*Figure S6. A plain import day with no injection. Rule for the day: an all-import day of the no-
injection worlds; the day with the median of arm D's largest P(X<0) of the day. Arms at seed 0; band
5% to 95%.*

**On the import day with the largest harm, arms B and D put their medians at about −70 points where
the median of arm A stays near +72.** The truth never goes below +2.7 points. The medians
of arms B and D reach −72.8 and −70.1 on 10 half-hours, with P(X<0) up to 0.94 and 0.83. The median
of arm A reaches only −1.0, with P(X<0) above 0.5 on 2 half-hours. P0f keeps its median on the
reading with P(X<0) at 0.0, and the band of arm D is 36.9 points wide on average against 1.6 for
P0f. The medians of arms B and D are roughly the mirror image of the median of arm A.

![The import day on which arm D does worst against P0f](series_07_largest_harm.svg)

*Figure S7. The import day on which arm D does worst against P0f. Rule for the day: a day of the
test bank on which every half-hour is import; the day with the largest mean of arm D's signed CRPS
minus P0f's. Arms at seed 0; band 5% to 95%.*

**On the BSP-month day, arm C keeps its median near zero where arm B swings to −93.6 points.** The
truth reaches −16.9 points on 6 half-hours. The lowest median of arm C is −5.3. P(X<0) is above 0.5
on 9 half-hours for arm C and 10 for arm B, with a mean on the reverse-flow half-hours of 0.60 for
arm C and 0.42 for arm B. The mean P(X<0) on the import half-hours of the day is 0.14 for arm C and
0.19 for arm B. The bottom row shows steps of the BSP meter that reach about ±100 points.

![A month with real BSP steps: arm C beside arm B](series_08_bsp_steps.svg)

*Figure S8. A month with real BSP steps: arm C beside arm B. Rule for the day: an engine or engine-
and-solar day in a BSP month with reverse flow; the day with the median depth of reverse flow. Arms
at seed 0; band 5% to 95%.*

**On the evening engine window, the 11-quantile head and the two-lobe head give nearly the same
picture.** The median of D-quantile reaches −1.5 points against −0.3 for arm D, and P(X<0) peaks at
0.65 against 0.51. The mean band is 3.3 points wide for D-quantile and 3.8 for arm D. The mean
P(X<0) on import half-hours is 0.002 for D-quantile and 0.009 for arm D. This is one window at seed
0, so it does not rank the heads.

![The 11-quantile head beside the two-lobe head](series_09_quantile_head.svg)

*Figure S9. The 11-quantile head beside the two-lobe head. Rule for the day: the day of Figure S1.
Arms at seed 0; band 5% to 95%.*

**On the ambiguous window, D-sign gives a much narrower band than arm D and nearly the same
P(X<0).** The P(X<0) of D-sign peaks at 0.57 against 0.60 for arm D. The mean band is 9.7 points
wide for D-sign and 37.7 for arm D. The median of D-sign goes below zero on 2 half-hours and that of
arm D on 3. D-sign places its lobes at plus and minus the reading with the reactive floor removed,
and its P(X<0) is the mass of the mixture below zero, which goes to 0.5 by construction where the
reading is under about 0.15 MVA.

![The sign-only head beside the two-lobe head](series_10_sign_head.svg)

*Figure S10. The sign-only head beside the two-lobe head. Rule for the day: the day of Figure S5.
Arms at seed 0; band 5% to 95%.*

**On the typical real day, no arm raises P(X<0) visibly above zero.** The plotted reading stays
between about 60 and 92 points, and the three arms follow it. The P(X<0) lines lie on the zero line
on the whole day.

![A real day before the engine was connected, typical false alarm](series_11_real_median.svg)

*Figure S11. A real day before the engine was connected, typical false alarm. Rule for the day: a
real day before the engine was connected; the day with the median of arm D's largest P(X<0) of the
day. Arms at seed 0; band 5% to 95%.*

**On the real day with the largest false alarm of arm D, P(X<0) stays under about 0.1 and the
context holds a stretch of zero reading.** The reading in the 12 hours before the scored day is a
flat zero until about the scored day's first half-hour, and then jumps to about 80 points. The
P(X<0) of arms A and D is highest in the first half-hours of the scored day, at about 0.1 for arm A
and about 0.05 for arm D, and falls to near zero after. The largest false alarm of arm D on this day
therefore comes where the window starts after a run of zero reading, and it never reaches 0.5.

![A real day before the engine was connected, largest false alarm](series_12_real_largest.svg)

*Figure S12. A real day before the engine was connected, largest false alarm. Rule for the day: a
real day before the engine was connected; the day with the largest of arm D's largest P(X<0) of the
day. Arms at seed 0; band 5% to 95%.*

## The event generator against real data

**Three figures are published: the simulated engine days, the hour-of-day run probability against
the real record, and the ramp sizes against the real record.** The script `e00729_event_figures.py`
(stage `events`) draws 400 engines from the simulator with fixed seeds and no network output. Each
draw has 30 invented days that start on a Monday, with the 400 starts spread over the 52 weeks of a
year. Each draw takes the UK clock offset of each day and has Saturday and Sunday as its quiet days,
so the simulated hour-of-day curves shift by an hour in summer as the real engine's clock may. The
stage reads the real engine record and changes no file that it reads.

**The real engine series stay private.** The metered engine and the metered solar farms are NGED
generators, so no published figure shows a real series, a real size, a real date, or a real count.
The five aggregates are shares. Each divides the output by the series' own reference output, so no
absolute size can be recovered. Each share is rounded to the nearest 5%, for the real record and the
simulated draws alike, so the bar heights do not reveal the record's length or the count in a bin.
The rounded shares of one chart sum to about 1, and can differ from 1 by the rounding. The four bar
charts use coarse bins, and a bin with fewer than 5 real events is merged with its neighbour, so no
bin shows a single real run. The hour-of-day chart cannot merge hours, so it leaves a gap in the real
line at each real hour with 1 to 4 running half-hours. The same normalisation and rounding apply to
the simulated draws, and the simulated bins follow the real merges. A chart is published only when at
least 100 real events lie behind it (running half-hours for the hour-of-day chart, and runs, steps,
blocks of 7 days, or days for the others); otherwise the stage leaves the chart out and the page
omits it.

**Definitions of the five aggregates.** The reference output is the 99th percentile of running
output, and running output is positive output of at least 10% of the 99.9th percentile of positive
output, so small positive readings while idle cannot set the reference. A half-hour is running when
the output is at least 15% of the reference output. A run is a stretch of running half-hours, and
its length counts only where the run starts and ends inside a stretch of complete days. A ramp is a
change of at least 5% of the reference output between two neighbouring half-hours, and the top ramp
bin is 30% and above. A start is the first half-hour of a run, and the starts are counted in blocks
of 7 complete consecutive days. The daily energy is the day's output summed, as a share of the
reference output for 24 hours; the lowest bin holds every day under 0.5%, including days of net
import. An engine that never runs counts its days with no run, no start, and zero energy. Weekdays
and weekends come from the calendar, and the simulated draws treat bank holidays as weekdays.

**Figure E1 shows twelve kinds of simulated engine day.** Each panel is three consecutive days from
the randomised simulator, picked by a fixed rule: the first draw that fits the kind of day, preferring
a draw that no earlier kind used. The label describes the first day of the panel. The caption names
the ranges from which the simulator draws, and the panels hold simulated output only.

![Twelve kinds of simulated engine day](events_01_engine_days.svg)

*Figure E1. Twelve kinds of simulated engine day, in points of D8's capacity, three days per panel,
with the hour of day in UTC. Ranges drawn from: 1 to 4 units of 10 to 50 points each (at most 120
points in all); evening, morning, and midday run starts and lengths; ramps of 0 to 4 half-hours;
weekend factor 0.1 to 0.8. The full ranges are in the figure subtitle.*

**Figure E2, the simulated solar days, is not published.** In the first run's stored test bank, the
stage found no simulated day of source S2 (the PV model driven by CAMS) with solar output, so the
figure had nothing to show. The rerun is expected to fill it. Source S1 is the metered farms'
output, and its days stay private.

**Three of the four bar charts are left out because too few real events lie behind them.** The
run-length chart, the starts-per-week chart, and the daily-energy chart each fell under the floor of
100 real events. The page gives no count. The hour-of-day chart and the ramp chart passed the floor
and follow.

![Share of half-hours running, by hour of day, real record and simulated draws](events_03_hour_of_day.svg)

*Figure E3. Share of half-hours with the engine running, by UTC hour, for weekdays (top) and weekends
(bottom). Orange is the real record and blue is the pooled simulated draws, both rounded to 5%. A gap
in the orange line is an hour with 1 to 4 running half-hours in the real record.*

**On weekdays, the real engine runs in a larger share of half-hours between 04:00 and 15:00 UTC than
the pooled simulated draws do, and both lines peak between 15:00 and 20:00 UTC.** The orange line
stands at 0.15 to 0.35 from 04:00 to 15:00 UTC, with a peak of 0.35 near 07:00, while the blue line
stands at 0.05 to 0.2 over the same hours. Both lines reach about 0.35 to 0.4 between 15:00 and 19:00
UTC. The blue line is the mean of 400 draws, so the figure does not show whether individual draws
cover the real morning share.

**On weekends, the real record shows almost no running, and the pooled simulated draws show more.**
The orange line is at 0 where it is drawn, with gaps where the real hour is blank, and the blue line
stands at 0.05 to 0.2 with its peak between 16:00 and 19:00 UTC. The figure shows the pooled mean,
so it does not show whether the lowest weekend factor of the simulator (0.1 of the weekday
probability) covers the real weekends.

![Share of half-hour steps in three size bins, real record and simulated draws](events_05_ramp.svg)

*Figure E4. Share of half-hour steps of at least 5% of the reference output in three size bins. Orange
is the real record and blue is the pooled simulated draws, both rounded to 5%.*

**The real record has more steps of 30% or more of the reference output than the simulated draws
have, and fewer steps of 15% to 30%.** The real shares are 0.30, 0.25, and 0.45 for the bins 0.05 to
0.15, 0.15 to 0.3, and 0.3 and above, and the simulated shares are 0.30, 0.35, and 0.35. The two
distributions agree in the smallest bin, and the difference of 0.10 in each of the other two bins
is larger than the 5% rounding of the figure.

## The solar and engine heads against the true series

**The solar head of arm G1 and the engine head of arm G2 are scored against the true synthetic
output, and the stage that does so has not run yet.** The script `e00729_component_scores.py`
(stage `components`) reads the stored test predictions of arms G1 and G2 at seeds 0 to 2, the stored
test bank, and the first stored training bank. It fits nothing, scores no new network output, and
reads no real file. Every head is a Gaussian with a mean and a standard deviation per half-hour, so
the 90% interval is the mean plus and minus 1.645 standard deviations, with its lower end clipped at
zero. Errors are in points of D8's capacity.

**The table will hold the following measures for each head, as the mean over seeds 0 to 2 with the
smallest and largest seed beside it.** The numbers are interim until the stage has run, and none is
written here.

- The mean absolute error over all half-hours, over the half-hours where the true output is not
  zero, and over each of the four world types.
- The mean absolute error of two predictors that need no network, for context: the all-zero
  predictor, and a climatology predictor that takes the mean true output of each (season,
  half-hour) cell of the first training bank.
- The correlation of the head mean with the true output, and the coverage of the 90% interval.
- For the engine head, the F1 of the running flag, which is the head mean above 10 points, against
  the true label of a running engine (true engine output above zero), and so the same flag as K1.
- For the solar head, the mean and the largest output at night, which should be about zero, and the
  share of (segment, half-hour) groups in which the head rises with the CAMS clear-sky index, with
  the same share for the true output beside it. That share is descriptive and has no bar.

**The figures show four days of each head and a binned comparison of head and truth.** Each day is
picked by a fixed rule that the caption states, and no day carries a calendar date. The solar days
of source S1 are built from the metered farms' output, so they go to a private folder and are not
published; a solar day of source S2 may be published. The binned plots show the mean head output and
its 10th to 90th percentile in bins of true output, and keep only bins of at least 100 half-hours.

**A count of the stored banks, made while preparing this section, found no simulated day of source S2
with solar output.** In the first training bank, 194 segments of the solar and both worlds use
source S2, and none of them holds any solar output. The same holds for the 8 such segments of the
validation bank and the 24 such segments of the test bank. Every base day of those segments carries
the flag that the day took no data. The solar head was therefore scored on days of source S1 only,
in effect. The cause was the satellite-sky rule of `farm_set`, which is fixed for the rerun (see
"Superseded first run").

## Checks

**Two audit triggers fired, and no audit has run.** Trigger 2 fired because C1, C4, C6, and C13
depart from the prediction, and the weak yardstick B1 explains the size of C1 only. Trigger 5 fired
because arm B (0.626 points over all rows) scores below the arms C and D that contain its inputs
(0.845 and 0.786), which is also the reading of C4. The result review decides whether to commission
an audit. The median forecast minus the mean forecast per site does not apply, because D8 is one
series. The normalised error of the median sits beside the signed CRPS of every arm in Table 3.

**Anonymisation check for D8.** The page and its six figures hold aggregates only. They show no
engine size, no connection date, no distance, no register entry, no series identifier, and no
generator value. The real engine record enters only as the K1 and K2 aggregates above. The figures
show shares by era and by UTC hour, and no figure carries a calendar date. A synthetic generator
series may be shown, and none is shown here.

## The plan as registered

**Every section from here to "Method" is the plan, written before any fit.** Its sentences about the
future describe what the code was written to do.

## Terms

- **Signed power X** is D8's real power in megawatts (MW), positive for import and negative for
  reverse flow.
- **The reading R** is what D8's meter reports: R = sqrt(X² + Q²), where Q is the reactive floor,
  the reactive power that keeps the reading above zero when X crosses zero. **Folding** is the
  name for this loss of the sign.
- **A point** is 1% of D8's effective capacity of 10.379 MVA, the normaliser that
  [data-processing](../data-processing.md) already uses for D8.
- **The bulk supply point (BSP)** above D8 is BSP A, the substation that feeds D8 and eight other
  Primaries. Its meter reports signed MW. **The BSP steps** are the half-hour changes of that meter.
- **A base day** is a UTC day of D8's real reading in the pre-engine era. **An injection** is a
  simulated engine output E and solar output S subtracted from the base day, so that X = base − E −
  S. **A segment** is one calendar month of base days carrying one continuous injection draw.
- **A world type** says what a segment holds: no injection, engine only, solar only, or engine and
  solar.
- **The two-lobe distribution** is the neural network's output for each half-hour: a probability of
  reverse flow, and a Gaussian magnitude for each of the two lobes (one at or above zero, one
  below). **P(X<0)** is the distribution's mass below zero.
- **The eras of D8** are the pre-engine era, the net era (engine only), the solar era (engine and
  the large solar farm), and the fade era, as [experiment
  697](../00697-gas-engine-state-tracker-part-4-running-patterns/result.md) defines them. This page
  uses the era labels only.
- **The metered engine record** is the record of D8's engine output that experiments 194, 694, and
  697 use. Its length is on the [experiment
  194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
  page. Experiments 694 and 697 call the record "Gas A" and describe it as a separate engine. A step
  test for this plan shows that the record is D8's engine (next section).

## What we know before the experiment

**D8's reading never falls below 1.0 MVA in the pre-engine era, and falls almost to zero once the
engine connects.** A read-only check written for this plan counted the pre-engine days with at least
40 of 48 half-hours, and none has a half-hour below 1.0 MVA; the median daily minimum is 5.7 MVA. In
the net era, 17% of days have a daily minimum below 1.0 MVA, and the lowest daily minima are 0.03
MVA. In the solar era the share rises to 56% of days, and to 87% to 97% in May to August. The lowest
half-hours sit in a one-half-hour V-shaped dip. That dip is what folding through zero predicts, but a
net flow that approaches zero without crossing it would give the same dip, so the reading alone
cannot show reverse flow.

**The reactive floor Q at D8 is small: the reading reaches 0.01 to 0.03 MVA.** The real reading does
not stay clear of zero, so the simulator draws Q from 0 to 0.3 MVA (0 to 3 points). The upper end
covers the 0.14 MVA that experiment 694 assumed. Treating a pre-engine base reading as signed power
introduces an error of at most Q² / (2R), which is under 0.02 MVA for Q of 0.3 MVA and the
pre-engine minimum of about 3 MVA.

**Reverse flow is plausible in the net era as well as in the solar era.** The maintainer's working
idea was that D8 reverses only once both the engine and the solar farm are connected. The check does
not support "only after both", because the reading already reaches its floor in the engine-only era.
The synthetic worlds therefore contain reverse flow from the engine alone, from solar alone, and
from both.

**The metered engine record is D8's engine, and its output steps in the BSP meter but not in D8's
reading.** A step test regressed the half-hour changes of each meter on the changes of the record.
The BSP A meter falls by less than 1 MW per MW of engine output, far outside 36 placebo shifts. D8's
reading does not move, because on the record's days D8's reading is gross demand with no engine in
it, as [data-processing](../data-processing.md) records. Two consequences follow for this plan:

- On the record's days the neural network can be scored against the record only after the record is
  subtracted from D8's reading and folded (check K1 below). The same days without that subtraction
  are a placebo with a real engine in the BSP meter and none in D8's reading (check K2).
- The BSP's response is below 1, so the simulator scales the engine and solar terms it adds to the
  BSP by a coefficient β drawn from 0.6 to 1.0. That range is pre-registered here and is never
  refitted. The range comes from the same record that the realism gate and check K1 read, so the
  three are not independent (see "The realism gate before any fit").

**The BSP steps are therefore a strong, real hint for the engine.** Three earlier pages agree.
[Experiment
194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)
placed the engine at D8 from the timing of D8's shortfall and warned that the register's Primary
column does not name the meter that sees a generator. [Experiment
722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md) found a slope of the BSP residual's 2-hour
change on the record's change of -0.85 (95% interval -1.16 to -0.39), close to the expected -1 but
void on its detection limit of 0.54 against a bar of 0.5. Experiment 723's plan keeps that reading
void and replaces the two-placebo rule with a permutation rule over circular day shifts for future
records. The half-hour step test of this plan reads the same relation and finds it far outside its
placebos. The test that found the BSP too noisy was a different one: the level of the BSP residual
at half-hours when D8's reading is near its floor, which cannot show a reverse flow of a few MW.

**Only the last few months of the base carry BSP steps.** The BSP A meter covers a short span of the
base. Arms C and H and every arm after them read the BSP steps where the meter exists and a mask
channel elsewhere. A few months of base days carry real BSP steps: two of them are test months,
one is a validation month, and the rest are training months. The BSP contrasts (C4 and C12) are
read on the two test months that the meter covers.

## The synthetic world

### Base days and the split

**The base is D8's cleaned pre-engine reading, cut at the earlier of two boundaries.** The first is
the end of the pre-engine era label. The second is the engine's connection date in the private
constants file, read through `ml-experiments/engine_constants.py`. The page never states which
boundary binds or how many days the cut removes. Half-hours that the `substation_zero` rule flags
are dropped, and the count dropped per split is reported. Days within 24 hours of a logged closure at
D8 are dropped if the switching logs load, and the count is reported.

**Whole calendar months are held out, because D8's demand has changed since the pre-engine era.** A
rule chooses the months, `e00729_core.py` applies the rule to the private base calendar, and neither
this page nor the repository's code lists a month. The explicit list is in `PRIVATE/splits.md` of the
experiment's data folder. The rule has four steps:

- A base month is a calendar month with at least 20 base days.
- A base month that overlaps the fixed lockdown window of [data-processing](../data-processing.md)
  is always a training month, and never counts when the next step numbers the months.
- The remaining base months fall in two groups, numbered from 1 in time order: the months in which
  the BSP A meter has data on at least 20 base days, and all other months.
- In both groups, every third month from the second is a test month. A validation month is the last
  month of the BSP group, and every sixth month from the first in the other group. Every other
  base month is a training month.

The rule gives 8 test months, 2 of them in the BSP group, and 4 validation months, interleaved with
training months in time order, and every season holds at least one test month.

| Split | Rule | Role |
|---|---|---|
| Test | Every third month from the second in each group | Every counted reading |
| Validation | The last month (BSP group); every sixth month from the first (other group) | Early stopping, the architecture choice, the choice of B\*, and the margin m |
| Training | Every other base month, and every month in the lockdown window | Fitting only |

Training drops every base day within 2 days of a test or validation month, because a window spans 48
hours. A validation month that borders a test month also drops its days within 2 days of the test
month.

### The engine simulator

**The engine simulator draws its parameters from wide ranges set from published descriptions, never
from a fitted real value.** The descriptions are those of experiments
[694](../00694-gas-engine-state-tracker-part-1/result.md),
[697](../00697-gas-engine-state-tracker-part-4-running-patterns/result.md), and
[159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md): the engine runs more
on weekday evenings, often starts on weekday mornings, runs for spells of a few half-hours, reaches
most of its level in the first running half-hour, and runs on for days at a time in multi-day
spells. Each segment draws one parameter set:

| Parameter | Range |
|---|---|
| Number of units | 1 to 4 |
| Size of one unit | 10 to 50 points; total capped at 120 points |
| Share of days with no run | 0.05 to 0.5 |
| Weekday evening run (start 15:00 to 17:30 local, 2 to 6 hours) | probability 0.3 to 0.9 a day |
| Weekday morning run (start 05:30 to 08:30 local, 1 to 8 hours) | probability 0.1 to 0.6 a day |
| Midday run (start 10:00 to 14:00 local, 1 to 4 hours) | probability 0 to 0.3 a day |
| Weekend and bank-holiday factor on each run probability | 0.1 to 0.8 |
| Multi-day spell (2 to 6 days, at part or full output) | 0 to 0.5 per 30 days |
| Units running in a run | 1 to all, drawn per run |
| Output of a running unit | 0.6 to 1.0 of its size |
| Ramp to full output | 0 to 4 half-hours |
| Trip of one unit, per running half-hour | probability 0 to 0.02, lasting 1 to 4 half-hours |
| Start and stop times | continuous, so a half-hour that holds a switch carries a partial mean |

The simulator has no price input, so the engine's link with the system price (experiment 697's Q2)
is not reproduced. Runs carry no season, because experiment 697 found no winter-summer difference.

### The solar simulator

**Solar output comes from two sources, so that the sunshine input never equals the driver of the
injected solar.** Each segment draws one source:

- **Source S1 (half of the segments):** the normalised output of one of the metered solar farms A
  to F on the same base day, times a drawn capacity. The farms lie 5 to 25 km from D8, so their sky
  differs from the sky at D8 as a real embedded farm's would.
- **Source S2 (the other half):** the satellite irradiance of the Copernicus Atmosphere Monitoring
  Service (CAMS) at one of the farm sites, not at D8, on the same base day, through a plane-of-array
  solar model with a drawn tilt (10° to 40°), azimuth (120° to 240°), and ratio of direct-current to
  alternating-current rating (1.0 to 1.4, with clipping), times a drawn capacity and a multiplicative
  noise with a half-hour autocorrelation of 0.9 and a standard deviation of 0.05.

A segment with solar draws a capacity log-uniform from 60 to 400 points. Solar output is never
negative and is zero when the sun is below the horizon. The stored CAMS index changes once an hour
([experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md)), so S2 and every CAMS input
interpolate the hourly index linearly to half-hours.

**Source S1 needs metered farm data, and a base day without it takes source S2 for that day.** A
read-only check counted the base days on which a farm has at least 40 of 48 half-hours of data. At
least one of the six farms has data on about 80% of base days, and no farm has data on about 19% to
25% of base days in each split. A segment that draws S1 uses one farm at a time: on a base day on
which the drawn farm lacks data, the segment takes another farm that has data, and on a base day on
which no farm has data, the segment takes source S2 at the segment's capacity for that day. The
code flags each such day, leaves flagged days out of the S1 rows of every by-source table, and
reports the count of flagged days per split. About 1 in 5 of the days in S1 segments is flagged.

**The sunshine input at D8 and the injected solar must not share a CAMS cell.** Arm B reads CAMS at
D8, and source S2 builds the injected solar from CAMS at a farm site, so a gain of arm B on S2 could
come from the shared satellite product and not from the sky that D8 sees. Three rules limit that
reading:

- The code asserts that the CAMS series at the injection's farm site differs from the CAMS series at
  D8. A read-only check found that no farm's daytime series equals D8's: the median gap is 4.5% to
  11% of D8's value, depending on the farm.
- The write-up reports arm B's gain (C3) separately for S1 and S2. The C3 prediction rests on S1,
  because the metered farm output shares no satellite processing with the sunshine input, so the S1
  gain is the evidence that the sunshine input helps. The S2 gain is an upper bound, because the
  injection and the input both come from one satellite product.
- Both sources are reported in every by-source table, with the S1-fallback days left out of S1.

### Folding, the BSP steps, and the world mix

**Each segment folds its signed truth with its own reactive floor.** The segment draws Q from 0 to
0.3 MVA, and each half-hour multiplies Q by a factor from 0.8 to 1.0. The reading is R = sqrt(X² +
Q²).

**The synthetic BSP meter is the real BSP A meter of the same base day plus the injection's
effect.**
It is BSP_real − β (E + S), with β drawn per segment from 0.6 to 1.0, so the BSP noise from the other
eight Primaries and the outside generators is real and has the same structure in training and test.
The neural network reads the half-hour change of this series.

**The test world mix is fixed:** no injection 0.15, engine only 0.30, solar only 0.20, and engine
and solar 0.35, so 85% of test segments carry an injection. The share with no injection gives the
neural network real pre-engine days whose answer is the reading itself.

**The training mix is randomised, so that C1 does not measure knowledge of the fixed 85%.** Each
bank
of 32 segment draws (one month, one seed) draws its own share of no-injection segments from a uniform
range of 0.05 to 0.60, which makes the injected share of a bank range from 0.40 to 0.95. The other
three world types share the remainder in the proportions 0.30 : 0.20 : 0.35. The test mix of 0.15
and the shifted mix below (0.50) both lie inside the range. The baselines B1 and B2 receive the
test world's injected share as a prior (see "Baselines"), so the neural network has no
advantage in knowing the mix.

**Training draws come from fixed banks, and test draws are shared by every arm.** Each training
month gets 32 segment draws per seed, from generator seed 7290 plus the seed. Each
validation and test month gets 8 segment draws from a separate generator seed space (729,000 and
up), built once, hashed, and read by every arm and baseline. Test draws never enter training.

**Every arm draws half of each epoch's windows from the training months that carry real BSP steps,
and half from all other training months.** A few training months carry real BSP steps, and the BSP
channel is empty everywhere else, so without this sampler arms C and later would see a filled BSP
channel on a small share of windows. Every arm uses the same sampler, including arms A and B, which
ignore the BSP channel, so that no contrast changes the training mix.

### The realism gate before any fit

**Before any neural network is fitted, the synthetic readings must look like D8's real readings in
the eras they imitate, judged on folded aggregates only.** The gate compares the training banks'
engine-only worlds with the real net era, and the engine-and-solar worlds with the real solar era:

- the share of days whose daily minimum is below 1.0 MVA (real: 0.17 in the net era and 0.56 in the
  solar era) must lie within 0.5 to 2.0 times the real share;
- the share of half-hours below 1.0 MVA must lie within 0.5 to 2.0 times the real share.

**The gate also checks the synthetic BSP steps against the real relation on the metered engine
record, with steps matched in size.** On the real days, the half-hour steps of the BSP A meter are
regressed on the record's steps, with circular shifts of 3 to 20 whole days as placebos. The
regression gives a real slope, the slope's standard error, and the residual standard deviation. The
same regression runs on the engine-only synthetic test segments, cut into stretches as long as the
record. A regression on larger steps has a smaller standard error, so the two sides are compared at
equal step sizes: both sides are split into five bins of the absolute size of the record's step, the
bins are the quintiles of the real record, and each synthetic stretch is weighted so that every bin
holds the real record's share of half-hours. The gate passes when the real slope's 95% interval
overlaps the simulated range of -β (-1.0 to -0.6), and the real standard error and the real residual
standard deviation each lie inside the 5th to 95th percentile of the size-matched synthetic
stretches. The second condition checks that the synthetic BSP noise around an engine step is neither
cleaner nor noisier than the real noise.

**The gate is not independent evidence about the BSP simulator.** The range of β was set from the
same record that the gate tests, and check K1 reads the same record. A pass therefore shows that the
simulator's BSP noise has the right size around an engine step, and does not validate β. The record
holds an engine and no solar farm, so β for solar has no real validation at all. The write-up reports
the BSP gain (C4 and C12) separately for engine-only worlds, solar-only worlds, and engine-and-solar
worlds, so that a reader can see how much of the gain rests on the solar value of β.

**If the gate fails, the simulator's ranges may be re-tuned once, and the re-tune is recorded.** The
gate reads no neural-network output and no target, so a re-tune cannot tune on a score. A second
failure stops the line (verdict row 0). The test banks must also hold at least 2,000 reverse-flow
half-hours outside a dead band of 2 points, and at least 200 in the two BSP test months, or the line
stops at row 0. The plan's simulation expects about 4,400 and about 1,100, so the counts have a
margin of about 2 and 5 (section "Verdict table").

**Gate attempt 1 failed its share checks and passed its BSP-step checks.** The first run of the gate
on the training bank of seed 0 gave these shares. In engine-only worlds against the real net era,
the share of days with a daily minimum below 1.0 MVA was 0.144 against 0.172 (ratio 0.84, inside
the range), and the share of half-hours below 1.0 MVA was 0.042 against 0.0119 (ratio 3.5, outside
0.5 to 2.0). In engine-and-solar worlds against the real solar era, the share of days was 0.215
against 0.536 (ratio 0.40, outside), and the share of half-hours was 0.049 against 0.045 (ratio 1.1,
inside). The BSP-step conditions passed, with 7 finite synthetic fits. The synthetic engine therefore
holds the reading at the floor for too many consecutive half-hours in the net era, and too few
solar-era days reach the floor. The gate read no network output and no score.

**The one permitted re-tune changed five ranges, chosen offline from 129 candidate range sets.**
`e00729_gate_calibration.py` builds small synthetic banks for each candidate, on a training-bank
seed that no fit and no gate uses, and prints the four shares (days and half-hours, engine-only
and engine-and-solar worlds) as ratios to the real shares. It ran on the CPU, read D8's base
readings to build the synthetic worlds, and read no network output and no score, because no fit
existed. It tried 8 candidates, then 73 (spells, solar capacity, and ramps), then 48 (engine
frequency and size, and solar capacity), and the set that kept every ratio furthest inside 0.5 to
2.0 was taken. The ranges changed from attempt 1 are these:

| Range | Attempt 1 | Re-tune |
|---|---|---|
| Share of days with no run | 0.2 to 0.7 | 0.05 to 0.5 |
| Multi-day spells per 30 days | 0 to 2 | 0 to 0.5 |
| Length of a multi-day spell | 2 to 19 days | 2 to 6 days |
| Ramp to full output | 0 to 2 half-hours | 0 to 4 half-hours |
| Solar capacity (log-uniform) | 5 to 150 points | 60 to 400 points |

Every other range, and every registered bar, is unchanged. Spells at constant output held the
reading at the floor for long stretches, so they are rarer and shorter, and the engine runs on more
days; larger solar capacities make more solar-era days cross the floor. The offline shares of the
chosen set were 0.70 and 1.2 times the real net-era shares (days, half-hours) and 0.64 and 0.63 times
the real solar-era shares. A second failure of the gate ends the line at verdict row 0.

**The gate passed on attempt 2, and the gate has now used both of its runs.** The 129 candidate
range sets were tried offline against the real era shares, so the re-tune used the calibration of
the real shares and no network output and no score. In the net era, the share of days with a floor
reading was 0.101 in the synthetic worlds against 0.172 in the real data (0.59 times), and the
share of half-hours was 0.0124 against 0.0119 (1.04 times). In the solar era, the day share was
0.328 against 0.536 (0.61 times), and the half-hour share was 0.0255 against 0.0454 (0.56 times).
The solar-era half-hour share sits near the 0.5 lower bound. The BSP-step regression passed and
was not void, with 8 finite synthetic fits (the minimum is 5) from 8 synthetic stretches (19
stretches before the size-matched selection) and 1 real stretch. The reachability counts were 2,019
stretches, of which 303 are BSP stretches. The gate's pass is a reading-level realism check only.
It shows that the synthetic readings match the real readings in these shares and in the BSP-step
slope, and it says nothing about whether a network trained on the synthetic worlds will untangle
real engines.

## Arms and architectures

### The input ladder

**The ladder adds one input at a time, from arm A to arm D, and each other arm changes arm D or arm
A
in one way.** Every window is 48 hours long, from 12:00 UTC on the day before the scored day to
12:00 UTC on the day after, and every arm scores the middle 24 hours. The clock is implicit in the
window's position in every arm.

| Arm | Inputs | Output |
|---|---|---|
| A | The reading R and its missing-value mask | Two-lobe |
| B | A plus the CAMS clear-sky index, clear-sky irradiance, and cosine of the solar zenith angle at D8 | Two-lobe |
| C | B plus the synthetic BSP steps and a BSP mask | Two-lobe |
| D | C plus local half-hour (sine and cosine), day of week, bank-holiday flag, and ERA5 2 m temperature, 10 m wind speed, and precipitation at D8's grid cell | Two-lobe |
| H | A plus the synthetic BSP steps and a BSP mask, with no CAMS fields | Two-lobe |
| D-quantile | D | 11 quantiles, pinball loss |
| D-24 | D on a 24-hour window holding only the scored day | Two-lobe |
| E | D, trained with half of the windows masked over a random 1 to 24 hours of the reading (with a mask channel), scored unmasked | Two-lobe |
| F | D plus the folded reading of the same segment 7 and 14 days earlier, each with a mask | Two-lobe |
| G1 | D | Two-lobe plus a solar-output head |
| G2 | D | Two-lobe plus an engine-output head |
| D-sign | D | One probability of reverse flow, with the two lobes fixed at plus and minus the reading with the reactive floor removed |

**The ladder is cumulative, so the BSP gain depends on the order of the inputs.** In arm C the BSP
steps join an input set that already holds sunshine, and the steps may carry less information once
sunshine is known. Arm H adds the BSP steps to arm A with no sunshine, so that C12 (H minus A) is the
BSP gain in the other order. A BSP gain that shows in C12 and not in C4 means that sunshine already
carries the information. Neither order is a result about which input matters most.

**Arm D carries the coincident observed weather because this is a reconstruction of the past, not
a forecast.** The temperature, the 10 m wind speed, and the precipitation are the hour's own ERA5
values, and no forecast origin exists, so the origin-time weather rule of the forecasting
experiments does not apply. The three values are ERA5's at D8's grid cell, the cell that gives the
temperature. The wind speed is the stored speed of the 10 m wind (the square root of the sum of the
squares of its eastward and northward components, checked against the two component files), and the
precipitation is ERA5's total precipitation. The stored hourly values cover every hour from
2019-09-01 to 2026-07-31, with no gap and no missing value, so every base day has all three. The
wind speed is standardised with the mean and standard deviation over the training days, and the
precipitation enters as the logarithm of one plus its millimetres an hour, divided by 0.5. No arm
reads the month or the day of the year, so no arm can memorise the calendar. Every arm that
extends arm D (D-quantile, D-24, E, F, G1, G2, and D-sign) carries the same three weather inputs.

**Arm F's earlier days carry their own engine and solar output, never a copy of the scored day's.**
The segment is simulated as one continuous month, so the day 7 days earlier holds that day's own
engine runs and sky, drawn from the same parameter set. A lag that falls outside the window's own
month is masked, with the same rule in training and test. The lags share the segment's parameter
set, which makes them more informative than the lags of a real engine whose behaviour drifts, so C8
is an upper bound on the gain from lags.

### Three neural-network architectures

**Three architectures are fitted on arms A and D, and the best on the validation months carries
every other arm.**

- **Denoising autoencoder**, after Kelly and Knottenbelt: a 1-D convolution of 8 filters (width 4),
  a dense layer to a 128-unit bottleneck, a dense layer back to the window length, a 1-D
  convolution, and the output heads at every half-hour.
- **1-D U-net**, after [Ronneberger et al. (2015)](https://doi.org/10.1007/978-3-319-24574-4_28):
  four levels of 32, 64, 128, and 256 channels, kernels of width 5, group normalisation, GELU
  activations, and skip connections, about 1 million weights.
- **Bidirectional long short-term memory neural network**, after the recurrent neural network of
  Kelly and Knottenbelt: a 1-D convolution of 16 filters, two bidirectional layers of 128 units, and
  the output heads at every half-hour.

**The architecture rule is fixed here:** the lowest mean over seeds 0 to 2 of arm D's signed CRPS on
the validation months wins, and an architecture within 0.125 points (half of the margin's floor) of
the lowest wins if it has fewer weights. The test scores of all three architectures on arms A and D
are reported, and none of them enters a counted contrast.

**The validation months do four jobs, and their scores are optimistic for the fitted networks.** The
months pick the architecture, stop each fit early, choose B\*, and set the margin m. Early stopping
picks the epoch that scores best on the same months, so a validation score is lower than the same
fit's score on months it never touched. The test months carry every counted reading for this reason.

### The output heads

**The two-lobe head outputs, at each half-hour, a logit for the sign and a mean and a spread for
each
lobe.** The import lobe is a Gaussian with a positive mean and the reverse-flow lobe a Gaussian with
a negative mean, in the manner of a mixture density network ([Bishop
(1994)](https://publications.aston.ac.uk/id/eprint/373/)). Each spread is 0.5 points plus a softplus,
so no lobe collapses. The loss is the negative log likelihood. The signed CRPS of the mixture is
computed exactly ([Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235)).

**The 0.5-point floor sets a lower limit on every score.** A Gaussian lobe of spread 0.5 points
whose
mean is exact scores a CRPS of 0.5 × (2 × 0.399 − 0.564) = 0.117 points. No arm can score below about
0.12 points on a half-hour where its lobe is right, and the harm checks therefore use P0f (see
"Baselines") and not P0.

**The sign head outputs one logit per half-hour and is trained by binary cross-entropy on the event
X < 0.** The probability P is the sigmoid of the logit. The signed distribution is the two-lobe
mixture with weights (1 − P, P), an import lobe at +M and a reverse-flow lobe at −M, where M =
sqrt(max(R² − Q̂², 0)) is the reading R with the reactive floor Q̂ of 0.1 MVA removed (the
magnitude that P0 uses). Both lobes have the fixed spread of 0.5 points, the same floor as every
other neural network and as P0f. Arm D-sign is therefore scored by the same signed CRPS, the same
pinball loss, and the same reliability of P(X<0) as every other arm. The head leaves the magnitude
to the reading and learns only the sign, so the contrast C13 shows what learning the magnitude
adds to learning the sign.

**D-sign's P(X<0) is the mixture's mass below zero, so it sits near 0.5 where the reading is
small.** The sigmoid of the logit is the weight P of the reverse-flow lobe, and the scored
probability of reverse flow is the mass of the whole mixture below zero. Where the reading is under
about 0.15 MVA, the two lobes lie within a few spreads of zero and overlap, so the mass below zero
is pulled toward 0.5 whatever P is. That pull shapes the lowest reading bands of the ambiguity
read-out and the reliability of P(X<0) that the write-up reports beside C13.

**The quantile head outputs the 11 quantiles 0.05, 0.1, 0.2 to 0.9 in steps of 0.1, and 0.95**,
sorted, and is trained by the pinball loss ([Koenker and Bassett
(1978)](https://doi.org/10.2307/1913643)). Its CRPS is computed from the piecewise-linear
distribution through its quantiles.

**The auxiliary heads of G1 and G2 output a Gaussian for the solar output and for the engine output
at each half-hour**, each trained by negative log likelihood with a weight of 0.5 beside the main
loss.

## Baselines

**Five rules that learn nothing from a day's shape set the bar for the neural networks.**

- **P0** reads the reading as signed import: X = sqrt(max(R² − Q̂², 0)), with Q̂ = 0.1 MVA, as a point
  distribution.
- **P0f** is P0's point X̂ given the same lobe-width floor as the neural networks: a Gaussian
  centred on X̂ with a spread of 0.5 points. P0f is the reference for the harm checks H1 and H2 and
  for contrast C2, so that the floor adds the same CRPS to the neural network and to its reference.
  P0 itself is reported in every table.
- **B1** uses the fact that the injected output is never negative: if demand D is below the reading,
  X = +R is impossible, so P(X<0) = π × P(D < R), where π is the share of segments that carry an
  injection in the world being scored (0.85 in the test banks). The prior π is the world-mix
  knowledge that a fixed-mix neural network would absorb in training, and giving it to B1 and B2
  keeps C1 from rewarding that knowledge. P(D < R) comes from the month-by-half-hour
  climatology of base demand (`baseline_forecasters.climatology.ClimatologyForecaster`, with local
  month, half-hour, and weekend flag) fitted on the training base days. The two lobes sit at ±R'.
- **B2** is B1 with a linear regression of base demand on local half-hour, day of week, bank
  holiday, and temperature, and a Gaussian residual.
- **M2**, experiment 694's grid filter, runs on the engine-only test segments with 28 days of
  injected history before each month, and its posterior of demand level minus engine output gives
  P(X<0). M2 has no solar term, so it is scored on the engine-only segments only, as a description.

**B\* is the better of B1 and B2 on the validation months**, chosen before any test score is read.
Both baselines use the prior π of the bank being scored.

## Scoring

**Every counted reading pairs arms on the same synthetic test half-hours.** The rows are the 48
half-hours of each test base day, for each of the 8 draws of its segment. Rows are split into
import (X ≥ 0) and reverse flow (X < 0), and a dead band of 2 points around zero is reported apart.
Every table also reports the scores by world type, by solar source (with the S1-fallback days left
out of S1), and by daylight (09:00 to 13:00 UTC) against the other hours, following experiment
721's advice.

**Intervals come from a two-way bootstrap that resamples date blocks and segment draws.** Within
each
test month, a resample draws whole blocks of 7 consecutive base dates with replacement and, drawn
independently, whole segments (the 8 draws of that month) with replacement. A resampled cell pairs
one drawn block with one drawn segment and carries all 48 half-hours of each date in the block. The
point estimate is the mean over seeds 0 to 2, and each resample uses the seed-averaged losses. Each
counted reading uses 20,000 resamples and generator seed 729. Drawing the two factors independently
makes the interval count the variance between segment draws as well as the variance between dates,
and counts their interaction twice, so the interval is slightly conservative.

**The Kish effective number of this design is the product of the Kish numbers of its two factors.**
A resampled cell has weight a × b, where a counts how often its block was drawn and b how often its
segment was drawn. The Kish number is (Σw)² / Σw², and for product weights (Σw)² / Σw² equals
((Σa)² / Σa²) × ((Σb)² / Σb²). The page reports the two factors beside each interval. Below 20 of
either factor, the interval is widened about its point by the ratio of the Student t quantile (with
that factor's Kish number minus 1 degrees of freedom, at the contrast's Bonferroni level) to the
normal quantile, using the smaller factor, as in [experiment
721](../00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/result.md). The eight test months
hold about 33 blocks and 64 segments, so no interval on all test months is widened. The two BSP test
months hold about 8 blocks and 16 segments, so the BSP contrasts are widened by about 1.46 (the
ratio of the t quantile with 7 degrees of freedom to the normal quantile, 2.865).

**Twelve contrasts are counted, and each is read at 99.583% (Bonferroni across the 12).** A
negative difference is a gain. C1 carries the verdict. The other eleven each read "gain", "loss", or
"no detectable difference" at a threshold of zero, because nothing here is ported. A thirteenth
contrast, C13, is report-only: it enters no verdict row and no bar, so it adds nothing to the
Bonferroni count, which stays at 12. C13 is read at the same 99.583% level, which makes its interval
wider than a 95% interval, and it is described and never counted. C6 is read on the
11-quantile pinball loss, defined as twice the mean of the pinball losses over the 11 levels (in
points, comparable with CRPS), for the two-lobe head's quantiles and for the quantile head. Every
other contrast is read on signed CRPS.

| Contrast | Difference | Rows |
|---|---|---|
| C1 (verdict) | D minus B\* | All synthetic test rows |
| C2 | A minus P0f | All |
| C3 | B minus A | All, and by solar source |
| C4 | C minus B | The two BSP test months, and by world type |
| C5 | D minus C | All |
| C6 | D-quantile minus D, on the 11-quantile pinball loss of both | All |
| C7 | E minus D | All |
| C8 | F minus D | All |
| C9 | G1 minus D | All |
| C10 | G2 minus D | All |
| C11 | D-24 minus D | All |
| C12 | H minus A | The two BSP test months, and by world type |
| C13 (report only) | D-sign minus D, on signed CRPS, so a positive difference is a loss of the sign-only arm | All |

**The margin m is set from the spread between seeds before any test contrast is read.** It is the
larger of 0.25 points and twice the standard deviation over seeds 0 to 4 of arm D's signed CRPS on
the validation months. Seeds 3 and 4 are fitted for arm D alone. With 3 seeds, a sample standard
deviation falls below half the true value with probability 0.22 (2 degrees of freedom); with 5
seeds, with probability 0.09 (4 degrees of freedom). The floor of 0.25 points absorbs a true seed
standard deviation of up to 0.125 points, and the standard deviation raises m only where the seeds
disagree by more. The validation months are optimistic (see "Three neural-network architectures"),
so the standard deviation measured there understates the spread on test months, and the floor
matters more than the standard deviation. 0.25 points is about 26 kW, below which a difference in
D8's reconstructed power is smaller than the reactive floor's effect on the reading. The value is
written to `margin.json`, which refuses to be overwritten.

**A shifted-mix read-out tests whether arm D has learned the training mix.** A second test bank is
built once from a separate seed space with 50% of its segments carrying no injection (an injected
share of 0.50, against 0.85 in the test banks), with the same months and draws. The read-out reports,
for arm D and for B\* with the prior set to 0.50, the signed CRPS, the reliability term and diagram
of P(X<0), and the share of half-hours with P(X<0) above 0.5. A neural network that had memorised the
injected share would over-predict reverse flow in this bank, and its reliability term would rise
well above its value on the test banks. The read-out enters no verdict row. The harm limit does not
change: it is derived from the test mix (section "Verdict table"), and the shifted bank has a
lower reverse-flow share, so it only makes harm easier to detect.

**An ambiguity read-out reports where P(X<0) is uncertain and whether it is calibrated there.** It
covers every arm with a probability output (every arm except D-quantile) and B\*, on the synthetic
test rows only, because the truth is known only there. Two groups of strata split the rows. The
first group splits the rows by the reading's level, in four bands: under 0.3 MVA (the top of the
simulated reactive power), 0.3 to 1.0 MVA, 1.0 to 3.0 MVA, and over 3.0 MVA. The second group splits
the rows by the truth's magnitude |X| in multiples of the reactive floor Q̂ of 0.1 MVA, in four
bands: under 1, 1 to 3, 3 to 10, and over 10. Each stratum reports its row count, its mean P(X<0),
the observed share of reverse flow, the reliability term of the Brier score over 10 bins, the
sharpness (the mean of 2 |P − 0.5|, which is 0 at P = 0.5 and 1 at certainty), and the share of rows
with P under 0.1 or over 0.9. The read-out sets no bar and enters no verdict row. A probability
that serves the user well sits near 0.5, with a mean forecast close to the observed share, where the
reading cannot tell the sign (a truth under Q̂), and sits near 0 or 1 where the reading is clear.

**Two bars decide between `supported` and `partly supported`.**

- **Sign calibration:** the reliability term of the Brier score of arm D's P(X<0), outside the dead
  band, is at most 0.01, and no bin of at least 200 half-hours sits off the diagonal by more than
  0.1 (the bars of experiment 724's plan). The reliability term is the mean squared gap between a
  forecast probability and the observed frequency, so 0.01 is a root-mean-square gap of 0.1, the same
  size as the bin bar. A calibrated forecast's term is about 0.0013 or less from sampling noise alone
  (10 bins, probabilities at worst 0.5, 64 segments of 30 days counted as independent), so the bar is
  about 8 times the noise.
- **Lobe coverage:** the 5% to 95% range of the lobe on the true side covers the truth on at least
  85% of half-hours. The floor makes coverage easy: in the plan's simulation, P0's magnitude with the
  floor alone covers the true magnitude on 99.8% of half-hours, and on 97.7% of reverse-flow
  half-hours. The 85% bar therefore detects a collapsed or biased lobe and says nothing about
  sharpness, and the write-up reports the mean width of the range beside the coverage. A lobe that
  covers more often is not penalised, because the reading fixes the magnitude closely.

## Real-data checks

**Five checks run on real days, and each reports aggregates only.**

- **R1, false alarms on real pre-engine days, a weak check.** The real test months, with no
  injection, have a known answer: X equals the reading. H2 reads arm D's signed CRPS minus P0f's on
  these half-hours and the share of half-hours with P(X<0) above 0.5. These days never read below
  1.0 MVA, so the neural network meets no reading near the fold. A pass shows that the neural network
  does not invent reverse flow far from zero, and says nothing about readings near zero. R1 and H2
  are weak checks.
- **K1, the engine head against the metered engine record.** On the record's days, the record's
  output is subtracted from D8's reading and folded with Q̂, and the real BSP meter, which already
  holds the engine's step, is read unchanged. Arm G2's engine head (the equal mixture of seeds 0 to
  2) is scored against the record. Bars: the head's median has a mean absolute error at most 0.5
  times that of an all-zero engine; and its F1 score for "running" (head median above 10 points,
  against the record's own running label) is at least M2's F1 on the same injected days. Arm D's
  signed CRPS on these days is reported beside B\*'s. The check reads a real engine schedule, real
  demand, and the real BSP response, so it is the one real known answer for the engine.
- **K2, the BSP placebo on the same days.** D8's reading is read unchanged, so it holds no
  engine, while the BSP meter steps with the engine. Bars: at most 0.5% of half-hours have P(X<0)
  above 0.5, and G2's engine head median exceeds 10 points on at most 2% of half-hours.
- **R2, re-fold consistency on the real net, solar, and fade eras, a report with no bar.** Arm D's
  output is folded with Q̂ and compared with the reading, and the report gives the share of
  half-hours on which the folded median lies within 2 points of the reading in each era, and on the
  K2 days. The neural network is trained to re-fold, so R2 is nearly guaranteed and enters no verdict
  row.
- **R3, descriptions with no bar.** The share of half-hours with P(X<0) above 0.5 by era and month
  and by era and local hour, set beside the share of days whose daily minimum is below 1.0 MVA; the
  share of predicted sign changes that sit next to a half-hour below 1.0 MVA, set beside the same
  share in the synthetic test; and the level of the matched BSP residual on half-hours with P(X<0)
  above 0.5, a level test that the plan's reverse-flow check found too noisy to show a few MW. The
  BSP steps, which carry the engine clearly, are validated by the realism gate and used by K1.

**K1 and K2 are counted readings that change no verdict word.** Each is reported as passes, fails,
or void. K1 is void if fewer than 200 of the record's half-hours have the engine running. The F1 bar
has little power: 200 running half-hours in spells of about 8 half-hours make about 25 independent
spells, so a proportion such as recall has a standard error near 0.10, and the bar separates two F1
scores only where they differ by about 0.2 or more. The simulator's ranges were set from published
descriptions of this same record, so K1 is not wholly independent of the simulator, and the write-up
says so.

## Verdict table

**The first row that matches is the verdict, and every row is reachable.** Row 3 sits before the
gain
rows on purpose: a reconstruction that harms the import half-hours is not usable, however well it
scores on reverse flow.

| Row | Condition | Verdict |
|---|---|---|
| 0 | The line stops before C1 is read: the manager halts it; the timing projection exceeds the budget and the manager declines; the realism gate fails after its one re-tune; the reachability counts are short; or a review gate has not passed when a fit would start | Stopped, no verdict (`inconclusive` in the front matter) |
| 1 | A correctness check fails (the leakage checks below, the holdout guard, the private cut, or a known-answer unit test of the fold, the CRPS, or the bootstrap) | `invalid` |
| 2 | A fit that a counted reading needs raised and was not rerun under the rerun rule (arm D-sign is no such fit) | `failed` |
| 3 | H1 or H2 fails | `not supported` |
| 4 | C1 at or below −m with an upper end below zero, and the calibration and lobe bars pass | `supported` |
| 5 | C1 as in row 4, and the calibration or lobe bar fails | `partly supported` |
| 6 | C1's upper end below zero and its point above −m | `small gain` |
| 7 | C1's interval includes zero and its half-width is at least m | `inconclusive` |
| 8 | Every other case | `not supported` |

**A failed or missing D-sign fit makes C13 `void` and leaves the verdict untouched.** Arm D-sign is
report-only, so no counted reading needs its fit. When any seed of arm D-sign has no fit record,
the scoring stage reports C13 as `void` (no point and no interval), leaves arm D-sign out of the
tables and the ambiguity read-out, and reads the verdict rows from the other arms alone. A failed
D-sign fit is therefore not row 2.

**The harm checks compare arm D with P0f on import half-hours, because the lobe-width floor
penalises every arm on every row.** A Gaussian lobe with a spread of 0.5 points scores a CRPS of
0.117 points even when its mean is exact, whereas P0, a point distribution, scores only its absolute
error, near 0.02 points on import half-hours in the plan's simulation. A limit set against P0 would
fail every arm by about 0.1 points before the neural network had learned anything. P0f gives P0's
point forecast the same floor and scores 0.119 points on import half-hours in the simulation, so
arm D's excess over P0f measures only what the neural network adds on import half-hours: probability
placed on reverse flow, and a lobe centred away from P0's point.

**The limit is half of the margin's floor, spread over the import share of the synthetic world.**
The smallest gain that C1 counts as support is m, at least 0.25 points over all rows. Harm of h
points on every import half-hour lowers the overall gain by h × (1 − s), where s is the share of
reverse-flow half-hours. The limit lets the harm consume at most half of the smallest countable gain:
h ≤ 0.125 / (1 − s). In the synthetic world's own mix (no injection 0.15, engine only 0.30, solar
only 0.20, engine and solar 0.35), `e00729_plan_calcs.py` draws 3,000 segments from the parameter
tables of this page on D8's base readings, and s = 0.051 (0.059 in engine-only worlds, 0.010 in
solar-only worlds, 0.091 in engine-and-solar worlds). The limit is then 0.125 / 0.949 = 0.132 points.
On the real held-out base days every half-hour is import, so s = 0 and the limit is 0.125 points.
One limit serves both checks, rounded down to 0.12 points.

**The code review re-derived s as 0.059, and the limit of 0.12 points stays.** The review ran the
simulator's own engine on invented base demand and an independent clear-sky solar model, and found s
= 0.059 overall (0.084 in engine-only worlds, 0.007 in solar-only worlds, 0.092 in engine-and-solar
worlds), against the script's 0.051. The script's engine is simpler: runs start at UTC rather than
local time, stop at midnight, use one output share for all units, and have no ramps or trips. The
formula 0.125 / (1 − s) is at least 0.125 for any s at or above zero, so the registered 0.12 is
stricter than the formula for every s: it is the formula's smallest value rounded down to two
decimals, which keeps the check no looser than the formula even where s is larger than the script
found. The same review found 5.5% of half-hours as reverse flow outside the dead band, against the
script's 4.7%, so the reachability counts below have more margin than the plan states.

**The limit is fixed before any fit and before any score, and it does not depend on the simulation.**
The training mix is randomised, and the limit is derived on the fixed test mix, which scores the
harm checks. The one data input is D8's base readings, which the calculation script read to estimate
s, and no score exists. The simulation is approximate: solar output is a clear-sky shape times a
random cloud factor, and engine runs follow the parameter table without trips or ramps. A share s
between 0.03 and 0.10 would move the formula only between 0.129 and 0.139 points, so the rounded 0.12
holds across that range. The same simulation gives 4.7% of half-hours as reverse flow outside the
dead band of 2 points, which is about 4,400 of the 92,000 half-hours of the test banks (8 months ×
30 days × 8 draws × 48 half-hours) and about 1,100 of the 23,000 in the two BSP test months.

**The second harm limit is on the normalised mean absolute error of the median.** Arm D's error on
import half-hours may exceed P0's by at most 0.25 points, the margin's floor, because a median error
smaller than the margin is below the smallest difference that the plan counts. P0's own error is
near 0.02 points in the simulation. H1 and H2 each fail if either limit fails.

**A rerun is the same attempt only when a fit raised, or a check that reads no bar failed, the fix
changes no constant, and the Opus reviewer confirms the fix.** `inconclusive` and `not supported`
are never rerun. Every number is labelled interim until the counted reading it belongs to is read.

## Leakage checks

**Each of these checks is an assertion in the code, and a failure is verdict row 1.**

- No base day sits in two splits, no training day lies within 2 days of a test or validation month,
  and no validation day lies within 2 days of a test month.
- No test or validation month overlaps the fixed lockdown window.
- The test, validation, and shifted-mix banks come from a seed space that no training bank uses, and
  their hashes match across all arms.
- Solar source S1 reads a farm on the scored base day only, S2 reads CAMS at a farm site, never at
  D8, and the CAMS series of that farm site differs from the CAMS series at D8.
- The synthetic BSP is built the same way in every split, from the same base day's real meter.
- No simulator parameter lies outside the ranges on this page, and no simulator code reads the
  metered engine record or the private constants file.
- The private cut is applied, and no day of the metered engine record is a base day.
- No valid time on or after 2026-07-01 is read, and no arm reads the month or the day of the year.
- The ERA5 wind speed and precipitation are the coincident hourly values of the scored hours at
  D8's grid cell, the same legitimate reconstruction input as the temperature, and each input row
  of every arm is on a list of allowed rows that holds no calendar-position row.
- The architecture, B\*, and m are chosen on validation months only, before any test score exists.

## Compute and gates

**The fits are small, and a timing gate measures their cost before the first fit.** One GPU process
of at most 6 GB runs at a time, with one data worker, on the core set that the manager assigns,
inside `systemd-run --user --scope` with MemoryMax 12G, MemorySwapMax 0, and one BLAS thread. The
plan has 12 arms (A, B, C, D, H, D-quantile, D-24, E, F, G1, G2, D-sign) and 50 fits:

- 18 fits in the architecture screen (3 architectures, arms A and D, 3 seeds);
- 30 fits in the ladder (the 10 arms other than A and D, 3 seeds each);
- 2 further fits of arm D, for seeds 3 and 4, which set the margin m.

The total of 50 holds only if the ladder reuses the fits of arms A and D from the screen, on the
architecture that the validation rule chooses.

**The GPU projection is about 1.17 GPU-process-hours at the epoch cap, from a timing gate that
measured 0.35 to 0.44 seconds an epoch.** The timing gate runs 5 epochs of each architecture on arm
D and projects the GPU-process-hours from the measured time per epoch. At those times, the 47 fits
of the plan before arm D-sign take 1.10 GPU-process-hours at the cap of 200 epochs. Arm D-sign adds 3
fits, which the projection prices at the slowest architecture's 0.44 seconds an epoch (D-sign's
single output makes it no slower), so the 50 fits take about 1.17 GPU-process-hours at the cap
(1.10 plus 3 × 200 × 0.44 / 3600 = 0.07). An early stop near 70 epochs takes about 0.41. The plan's
first projection assumed 3 seconds an epoch and gave 2.7 to 7.8 GPU-process-hours, and 10.2 if the
bidirectional LSTM were slow; the measured times are about seven times shorter. The default budget
is 10 GPU-process-hours. The manager owns the GPU slots and may set another budget, and a
projection above the budget goes to the manager before any fit.

**The CPU work adds about 2 hours on one core.** These are estimates from the plan's simulation,
which drew 3,000 segments in a few seconds, and the timing gate measures them:

- Generating the training, validation, and test banks for 5 seeds takes under 15 minutes.
- Scoring 50 fits on test banks of about 92,000 half-hours each, and the bootstrap of 20,000
  resamples over the cell means of 13 contrasts, take about 30 minutes.
- M2's baseline runs on one CPU core in under an hour.

**The gates run in the maintainer's order:** this plan, a Sonnet plan review, Sonnet code, a fresh
Opus code review before any first run, a real-file smoke test of every loader that reads a file the
synthetic-array tests do not exercise (D8's reading, the BSP meter, the frame of experiment 39, the
ERA5 temperature, wind speed, and precipitation, the bank holidays, and the metered engine record),
then a "fits starting" line to the manager with the projection and the cores before each family of
fits. Every fitted neural network
is saved in the experiment's data folder.

## Reviews and pre-run gates

**Disclosed breach of the first-run rule: the planner ran `e00729_plan_calcs.py` before any Opus
code review, and every number it produced is unreviewed.** The script applies the split rule to D8's
private base calendar and writes the month lists to the private `PRIVATE/splits.md`. It then
simulates 3,000 synthetic segments on D8's pre-engine readings. From the simulation it prints the
reverse-flow share (all half-hours, outside the 2-point dead band, and by world type), the dead-band
share, P0's median error on import half-hours, P0f's CRPS on import half-hours, and the lobe
coverage of the floor alone. The script also counts the share of base days on which at least 1 and
at least 3 metered farms have data, and prints the identical share and the median gap between the
CAMS series at each farm and at D8. Last, it prints the arithmetic of the bootstrap widening, the
seed floor, the GPU projection, and the harm limit. It reads the private engine-connection date to
apply the cut but prints no engine size, date, series identifier, or per-day real value, and it
prints split counts and shares only. The harm limit of 0.12 points, the simulated import share, the
cell-gap range of 4.5% to 11%, and every other number it produced stay unreviewed until the Opus
code review of the experiment's code passes. The first Opus code review has since re-derived the
import share and the sunshine-cell gap method (see "The limit is half of the margin's floor"); the
fixes it asked for are in the code, and a re-review of the fixes is still open. No further script
runs before that re-review.

## Standing practices and earlier advice

**The plan follows the standing practices that apply to a reconstruction.** It uses cleaned data and
reports the dropped half-hours, saves every fitted neural network, screens the architectures on four
validation months, reports q05 and q95 coverage, and plots example days. Monthly refits, the D15 and
D18 cut, stored-prediction controls, cold winter nights, and the avoidable-curtailment cost do not
apply, because no forecast is made. The experiment is not about load-shift events, so the four
load-shift scenarios do not apply; days near a logged closure are dropped instead.

**The plan follows six items of earlier advice and departs from one.**

- It checks import rows before reverse-flow gains (721), as H1 and H2.
- It splits daylight from the other hours (721).
- It interpolates the hourly CAMS index before using its half-hour changes (722).
- It scores the engine on the real record alone as well as on synthetic data (694), as K1.
- It builds synthetic engine runs with ramps, part load, and units much smaller than the engine's
  (694).
- It does not read the tracker's spell lengths as engine behaviour (697): the simulator's spell
  ranges are wide and not taken from tracker labels.
- It departs from 694's advice to build synthetic schedules from the real record's switching
  structure, because the record is now a test set; the simulator uses published descriptions as
  ranges.

## Decisions that need the maintainer

**The plan holds no open decision.** Arm D carries ERA5 wind speed and precipitation from the files
already stored, and the sign-only arm D-sign is a report-only contrast.

## Figures planned

**Every figure is safe to publish: synthetic series in detail, real series as aggregates only.** No
figure shows the metered engine record, an engine-head output on a real day, or a date of the engine.
Real-day panels carry no calendar dates.

1. D8's reading by era: the share of days with a daily minimum below 1.0 MVA by month, and the
   lower tail of the reading (aggregates).
2. A gallery of six synthetic segments drawn with solar source S2: base demand, engine output, solar
   output, signed truth, folded reading, and synthetic BSP steps, on days 1 and 2.
3. The realism gate: synthetic against real shares of near-zero days and half-hours.
4. Signed CRPS of every arm and baseline on all, import, and reverse-flow rows, as staggered dots on
   striped rows (`dot_chart` of experiment 700).
5. The 12 counted contrasts and the report-only C13, with their intervals and the margin m.
6. Example test windows: the two-lobe distribution as a band and P(X<0) as a strip against the
   truth, for an import-only, an engine-reversal, a solar-reversal, and a combined day.
7. The reliability diagram of P(X<0).
8. Real eras: the share of half-hours with P(X<0) above 0.5 by local hour and era, beside the share
   below 1.0 MVA.
9. The architecture screen on the validation months.

## Choices of the plan

- **The reactive floor is drawn from 0 to 0.3 MVA,** because D8's real reading reaches 0.01 to 0.03
  MVA, and a floor that keeps the reading clear of zero does not fit it.
- **The engine head's real check subtracts the record from D8's reading (K1),** because D8's reading
  is gross demand on the record's days; the unchanged days become a placebo (K2).
- **The BSP coefficient β is drawn from 0.6 to 1.0,** because the step test found a response below 1.
- **Arm D carries wind speed and precipitation,** from the stored ERA5 features of D8's grid cell,
  because this is a reconstruction and coincident observed weather is legitimate.
- **Arm D-sign is report-only,** because it enters no verdict row, and the Bonferroni level stays at
  12 counted contrasts.
- **The ladder is cumulative (A, then B, C, D)** and the variants E, F, G1, G2, the quantile head,
  the 24-hour window, and the sign head are each added to arm D, so that every contrast changes one
  input. Arm H adds the BSP steps to arm A, so that the BSP gain is read in both orders.
- **The demand-climatology baseline is B1,** which turns the climatology into a sign probability
  through the rule that injected output is never negative.
- **The harm checks use P0f and not P0,** because the lobe-width floor adds 0.117 points to every
  arm and not to P0.
- **The test and validation months follow a rule and are listed only in a private file,** because a
  list of months, with the era cut, could let a reader recompute withheld dates.

## Choices fixed in the code

**Where the plan was ambiguous or inconsistent, the code takes the simplest reading, listed here.**

- **Source S1 falls back to source S2 on a base day with no metered-farm output,** and the segment
  records which days fell back, so that the S1 leakage check skips them.
- **Solar output is zero whenever the sun is below the horizon** (zenith of 90 degrees or more), so
  that noise in the CAMS fields cannot create night-time output.
- **The hourly CAMS index becomes half-hourly by linear interpolation** with weights 0.75 and 0.25,
  and an hour whose neighbour is missing keeps its own value.
- **A training day within 2 days of a test or validation month is dropped,** and so is a validation
  day within 2 days of a test month; test days are never dropped. A month with too few days is
  dropped whole.
- **Lockdown months are always training months,** whatever position the rule would give them.
- **A lag input that falls outside the month, or on a missing half-hour, is zero with a zero mask.**
- **Masking hides one span of 1 to 24 hours** in both reading rows of arm E's training windows, and
  evaluation windows are never masked.
- **Half of each epoch's windows come from segments with real BSP steps** in every arm, and from
  all segments when none has them.
- **The training mix of worlds is drawn once per bank and month,** with the no-injection share
  uniform between 0.05 and 0.60 and the rest split in the test mix's proportions.
- **The quantile arm Dq is scored by its piecewise-linear CRPS,** computed from its 11 quantiles.
- **P0's lobes use a spread of 1e-6 points, P0f's use the 0.5-point floor,** and both take the
  reading after removing the 0.1 MVA reactive floor.
- **The timing gate projects the 32 ladder fits at the slowest architecture's time per epoch,**
  because the chosen architecture is not known at that point.
- **The ERA5 wind speed and precipitation are read from the stored hourly features of D8's grid
  cell, the file that holds the temperature.** The wind speed is instantaneous, so it takes
  the temperature's path exactly: the same placement of the hourly values on the half-hour grid as
  experiment 39's frame uses, then the same linear interpolation, and a test pins the two to equal
  values. The precipitation is an accumulation over the hour that ends at its stamp, so each
  half-hour takes the accumulation of the hour that contains it, in millimetres an hour (the file's
  kilograms per square metre per second times 3,600). A missing value would enter as zero without a
  mask, as the temperature does, and the stored record has none over the base years.
- **The banks are rebuilt with the two weather fields and keep their readings and digests.** A
  segment now stores the wind speed and the precipitation, so a bank built before this change
  cannot be loaded by the new code. The weather fields enter no random draw and no digest, so the
  rebuilt banks hold the same readings, truths, engines, and solar outputs, and every bank digest
  is unchanged. The realism gate reads only the readings, so the gate does not run again.
- **The `banks` stage prints whether the digests are unchanged.** It copies the earlier
  `banks/summary.json` to `banks/summary_before.json` before it builds, and logs "digests unchanged
  from the earlier summary" with true, false, or none (no earlier summary), so a rebuild after a
  change that should not touch a draw proves it.
- **The sign head's reading row is a target row, never an input.** The target stack holds the
  folded reading as its fifth row so that the sign head's decoder can place its two lobes, and no
  input window reads that row.
- **The ambiguity read-out's bands are code constants:** the reading bands at 0.3, 1.0, and 3.0
  MVA, and the truth-magnitude bands at 1, 3, and 10 times the 0.1 MVA reactive floor.
- **The margin file refuses to be overwritten,** so that m is fixed once, before any test score.
- **M2 runs on a continuous series of folded readings, with the causal spread,** and its running
  probability is multiplied by the climatology's sign probability.
- **Every evaluation bank has its own seed space, and its worlds are assigned by quota.** The
  validation bank takes root 729,250, the test bank 729,000, and the shifted bank 729,500, so no two
  banks share a draw. Each bank's world types are assigned by quota across the whole bank, so the
  realised mix is the planned mix up to rounding (10, 19, 13, and 22 segments of 64 for the test
  bank), and the code then deals the segments out month by month, so each test month, and with
  it each BSP test month, gets its share of every world type. B1 and B2 take the bank's realised
  share of injected segments as the prior pi; M2 has no such prior.
- **The BSP half of the realism gate uses stretches of one month, not the record's length.** The
  record spans more days than one synthetic segment, which is one month. The code cuts the real
  record and every engine-only synthetic segment into stretches of the longest length that a
  segment can hold; the real standard error and residual standard deviation are the medians over the
  real stretches, and the real slope's interval comes from the whole record. No registered bar
  changes. The zero steps of the record form their own size bin, and the five quintile bins are
  taken over its non-zero steps, because most steps are zero. Real and synthetic standard errors
  are compared as the standard error times the square root of the number of finite step pairs,
  because a stretch with more pairs has a smaller standard error. The gate counts only finite
  synthetic fits, records that count, and reads `void` (which does not pass) below 5 of them, a
  code choice and not a registered bar: the 5th and 95th percentiles of fewer fits are not a spread
  that can judge the real value.
- **K1's running label is the record's own `units >= 1`.** Half-hours without a record row are left
  out of K1, and the result holds the error ratio and the F1 scores only, never the raw mean of the
  record or a count of running half-hours.
- **The switching logs are read with the column heuristic of experiment 722.** A closure runs from
  its start to its end, and the days within 24 hours of either are dropped from the base. Only a
  missing or unreadable workbook is tolerated, and the summary records the status (`loaded`,
  `missing`, or `unreadable`) and whether any closure day fell on a base day. The banks stage
  raises a row-1 error when the split derived with the closure days differs from the private file,
  or when the file is missing, and the smoke test prints the flag with and without closure days.
- **The code computes B1's climatology directly.** The sign prior needs the empirical distribution of
  base demand in each (month, half-hour, weekend) cell, so the code builds it from the training days
  without smoothing over neighbouring cells, and does not call `ClimatologyForecaster`.
- **M2 starts at the first day of each month, without the plan's 28 days of injected history.** The
  simulated segments hold no earlier days, so M2 is weaker as a benchmark than the plan states, and
  K1's F1 bar is easier to meet.
- **A failure record counts only for a fit that a counted reading needs and that has no fit
  record.** A sanctioned rerun that succeeds removes the stale record.
- **The first test score is marked by a file** that the scoring stage creates once; the choice
  files of the architecture, B*, and the margin must be older than it, and B* is written once.
- **M2 reads each segment as one continuous series.** A missing half-hour is filled by linear
  interpolation before the tracker runs, and M2 is scored on every test segment, with the engine-only
  segments as the listed benchmark.
- **A segment shorter than 28 days gives M2's causal spread its own median.** Experiment 694's
  causal spread fails on a series shorter than 28 days, because a lag of 28 days reaches before
  its first half-hour. A local copy skips those lags and equals 694's function on a series of 28
  days or more (a test checks 28 and 60 days). Where fewer than 7 earlier days exist on a shorter
  series, the spread is 1.4826 times the series' own median absolute lag-1 difference, at least 0.03
  MVA, and is then quantised as before. The fill uses days after the half-hour it fills, so it mildly
  favours M2, which is conservative against the neural networks. The K1 pseudo-segments are longer
  than 28 days and pass through unchanged.
- **A steps series is NaN across a gap of days.** The first half-hour of a day whose previous day is
  not in the record has no step.
- **The engine size cap can push a unit below the 10-point draw range.** The code records each
  segment's smallest unit and the banks stage logs the smallest value over all segments.
- **A fit that raises is recorded and not retried.** It writes `fits/<name>.failed.json`, and the
  scoring stage then reads verdict row 2. A correctness error inside the scoring stage reads row 1.
- **The bank-level leakage checks run on every segment of every evaluation bank,** and the CAMS check
  compares D8's half-hourly series with each farm's on the same interpolation, failing when more
  than half of the daytime half-hours are identical.
- **Printed output holds counts, shares, and flags only,** never a month, a date, an engine size, a
  series identifier, or a register value, and the months come from the rule applied to the private
  base calendar when a stage runs.

## Method

**Architecture:**

- The carrying arm is arm D on the architecture that the validation rule chooses, with the two-lobe
  head; the inputs, layers, and heads of each architecture are in "Three neural-network
  architectures" and "The output heads".
- Inputs are in points of D8's capacity, the CAMS fields are normalised by clear-sky values, and
  temperature and wind speed are standardised on the training days.
- Every arm reads a 48-hour window and is scored on its middle 24 hours, except D-24.

**Training regime:**

- The training data are synthetic segments built on D8's pre-engine base days in the training
  months, from banks of 32 draws per month per seed, with the world mix, simulators, and fold above.
- Half of each epoch's windows come from the training months that carry real BSP steps, in every
  arm.
- An epoch is 8,000 windows; the optimiser is AdamW with a learning rate of 1e-3, weight decay of
  1e-4, batch size 256, and gradient clipping at 1.0; the learning rate halves after 10 epochs with
  no gain; training stops after 25 epochs with no gain in validation signed CRPS, or at 200 epochs.
- Seeds 0, 1, and 2 set the initial weights and the training bank of each fit; seeds 3 and 4 are
  fitted for arm D alone, to set the margin m.

**Measurement:**

- The scored rows, metrics, two-way bootstrap, Kish widening, margin, contrasts, harm limits, real
  checks, and verdict table are in the sections above.
- Every table reports signed CRPS, the 11-quantile pinball loss, and the normalised mean absolute
  error of the median, on all, import, and reverse-flow rows.
- The M2 baseline fills the causal spread of a segment shorter than 28 days with the segment's own
  median lag-1 spread (see "Choices fixed in the code"), which mildly favours M2.

## Files

**The code follows the pattern of experiment 721.** Experiment 729 reuses the structure of the
reviewed code in `ml-experiments/00721-*` (its path helpers, `run.sh`, test configuration, and smoke
test) by copying the pattern and imports nothing from 721. It imports the folders of experiments 694
(M2 and its trackers), 700, 697, 39, and 194 only through `e00729_paths.py`, which puts them on
`sys.path` and skips the dependent tests when a folder is absent. No file of those experiments is
edited.

- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/e00729_plan_calcs.py`
  (written, run once before its code review; see "Reviews and pre-run gates"): the plan's
  calculations. It applies the split rule to the private base calendar and writes
  `PRIVATE/splits.md`, simulates the synthetic world to find the reverse-flow share and the floor
  scores, counts the metered farms with data on each base day, and prints the arithmetic of the
  bootstrap widening, the seed floor, and the GPU projection. It prints aggregates only.
- `e00729_paths.py`: the data folder, `PRIVATE` folder, thread pinning, and the sibling folders.
- `e00729_core.py`: every named constant and bar on this page, the split rule (which derives the
  months from the private base calendar), the world mixes, the margin file, and the verdict rule.
- `e00729_data.py`: loaders for D8's cleaned reading, the BSP A meter, the CAMS and temperature
  fields of experiment 39's frame, the ERA5 wind speed and precipitation at D8's grid cell, the
  metered farms' output, the bank holidays, the metered engine record, and the private cut.
- `e00729_sim.py`: the engine and solar simulators, the fold, and the synthetic BSP.
- `e00729_banks.py`: builds, hashes, saves, and loads the training, validation, test, and shifted
  banks, and holds the functions of the realism gate.
- `e00729_arms.py`: the 12 arms, their input channels, windows, lags, masking, and the sampler.
- `e00729_nets.py` and `e00729_train.py`: the three architectures, the heads and losses, the fits,
  the timing gate, and the 50-fit schedule.
- `e00729_dist.py`: the exact mixture CRPS, the quantiles, and the quantile CRPS.
- `e00729_baselines.py`: P0, P0f, B1, B2, and a wrapper of experiment 694's M2.
- `e00729_score.py`: the scored rows, the metrics, the two-way bootstrap, the Kish widening, the
  bars, the harm and reachability checks, and the ambiguity read-out.
- `e00729_real.py`: R1, K1, K2, R2, and R3.
- `e00729_leakage.py`: the leakage assertions of verdict row 1.
- `e00729_gate_calibration.py`: the offline calibration behind the one permitted re-tune of the
  simulator ranges; it prints share ratios only and reads no network output and no score.
- `e00729_component_scores.py`: the component table and the head figures (stage `components`). It
  reads stored predictions of arms G1 and G2 and the stored banks, and writes solar days of source S1
  only to `PRIVATE/components`.
- `e00729_event_figures.py`: the event-generator figures (stage `events`). It draws simulated engine
  and solar days, and compares aggregates of the simulated draws with aggregates of the real engine
  record. It writes real series only to `PRIVATE/event_generator`.
- `e00729_stages.py`, `e00729_cli.py`, `e00729_report.py`, `e00729_figures.py`, and `run.sh`: the
  stage orchestration (including the `figures` stage, which writes figures 1, 2, and 6), the command
  line, the printed report (which writes the other figures, among them figure 8), the figure code,
  and the launch wrapper.
- `ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/tests/`: tests on synthetic
  arrays only; the tests that need `torch` skip in CI. `test_e00729_stages_end_to_end.py` runs the
  stages (banks, gate, tiny CPU fits, margin, scoring, verdict, figures) on invented arrays, with
  the day table, the metered record, and the private cut replaced. `smoke_real_files.py` opens each
  real file once and prints flags only; it is run by hand once, after the code review.
- `data/ml-experiments/00729-msa-d8-neural-untangler-synthetic-known-answers/`: written. It will
  hold `banks/`, `networks/`, `predictions/`, `scores/`, `margin.json`, `verdict.json`, the run
  logs, and a `PRIVATE/` folder for anything dated, which already holds `splits.md` (the month
  lists). The two read-only checks written for this plan, `reverse_flow_check.md` and
  `engine_metering_check.md`, are there too and are never published.
- `data/ml-experiments/00039-pv-disaggregation-cams-injection/work/frame.parquet`: read. The CAMS
  index, clear-sky irradiance, zenith, and 2 m temperature at D8 and at the farm sites, from
  [experiment 39](../00039-pv-disaggregation-cams-injection/result.md).
- `data/studies/per_study/stage1_pilot/demand/inputs/era5_features.parquet`: read. The hourly ERA5
  10 m wind speed and total precipitation at D8's grid cell, the file and the cell that give the
  temperature of experiment 39's frame.
- The metered engine record: read, from the window table of [experiment
  194](../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md).
  Never published with dates.
- `data/ml-experiments/private/engine_constants.json`: read through
  `ml-experiments/engine_constants.py` for the private cut only. Never committed.
