---
id: "00613"
slug: nra-roster-contrasts-with-intervals-five-seeds
title: >-
  Do five seeds give intervals tight enough to show that the class 2 candidate can replace the
  gate, that O2 is equivalent to D5, and that dropping E3 changes absorption?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v2
size: >-
  Complex. Answers to the five triggers: (1) it stores fitted network weights and forecasts for
  about six arms in the private data folder only; (2) more than one design could test the question
  (seed averaging against per-seed contrasts), so this plan fixes the seed mean and checks it
  against two disjoint seed sets; (3) it spans code whose callers cannot be named without searching
  (experiments 565, 574, 577, 584, 591, 592, 595, 597, 599, 601, 608, 610, and 612); (4) it touches
  no serving path and no degradation rule; (5) it changes no contract. Ceiling 10 GPU-hours
  (process hours); ask before crossing.
hypothesis: >-
  Experiment 612 left three claims resting on point estimates inside the tolerance. With seeds 0 to
  4 on the five development months, the interval of each contrast lies inside the tolerance (0.01
  coverage, 0.03 absorption, the printed minimum detectable effect in CRPS) for (1) the class 2
  candidate against K1, (2) O2 against D5 in K1, K2, and K3, and (3) E3 against the 3-day embargo
  on coverage.
prediction: >-
  Written before any score. P1: the whole interval of the candidate minus K1 lies inside 0.01 for
  day-4-to-end and after-end coverage at K1's band width. P2: O2 minus D5 lies inside 0.01 for
  event-free coverage in K1 and K2 but not in K3 at 120-240 h (experiment 612 saw -0.017 there). P3:
  E3 lowers absorption by at least 0.03 against the 3-day embargo (experiment 599 saw -0.045 to
  -0.062). P4: the cell-level bootstrap intervals shrink by less than a third against experiment
  612's, because the unit of resampling is still the 16 Primaries and 5 months.
decision_rule: >-
  A contrast is "equivalent" only if its whole Bonferroni interval lies inside the tolerance
  (coverage 0.01, follow 0.01, absorption 0.03, event-free CRPS 0.0006, the minimum detectable
  effect of experiment 591). The class 2 candidate can replace the G4 gate if its day-4-to-end and
  after-end coverage contrasts with K1 are equivalent and its event-free CRPS contrast with K2 is
  equivalent. O2 is an equivalent swap for D5 if every counted coverage and CRPS contrast is
  equivalent. Anything else is reported as inconclusive, better, or worse, with the side the
  interval falls on. The Bonferroni count is stated as a number beside every family. The experiment
  adopts nothing.
benchmark: >-
  K1, K2, and K3 of experiment 612 (D5 offset), and the candidate of experiment 612, all rescored
  on seeds 0 to 4.
row_set_id: experiment_470_intersection_v1
metric: >-
  Event-free q05-q95 coverage and CRPS at 24-120 h and 120-240 h; coverage, follow, and safe-side
  share of the normal running arrangement by phase on the injected bank; absorption; coverage 0-7
  days after the end.
verdict: partly supported
best_arm: >-
  Class 1: K1 with E3 (7-day embargo and half weight on the next 7 days). Class 2: the candidate at
  K1's band width, injected coverage within 0.0097 of K1. Class 3: K3 (lag-free), unchanged.
headline: >-
  Five seeds did not make the contrasts tight enough to settle two of the three claims. The class 2
  candidate at K1's band width covers injected load-shifts like K1 (day 4 to the end -0.004, interval
  -0.0097 to +0.0011; 0-7 days after the end -0.003, -0.009 to +0.003), and its safe-side share is
  equivalent (+0.001), so it can replace the gate on coverage. Its event-free CRPS is not shown
  equivalent in 3 of 4 cells. O2 against D5 is equivalent in none of the 30 counted event-free
  contrasts, because their half-widths are 0.02 to 0.06 in coverage. The 3-day embargo is equivalent
  to E3 on every injected contrast, and E3 lowers absorption by 0.045 to 0.062, but the interval
  reaches 0.02, short of the 0.03 tolerance.
lessons: >-
  More seeds do not narrow an interval whose width comes from the 16 Primaries and the 5 months. A
  test of event-free equivalence at a 0.01 coverage tolerance needs more Primaries or more months,
  not more seeds.
take_home_message: >-
  The class 2 candidate can stand in for the gate on injected load-shifts, because its coverage and
  safe-side share match K1's within 0.01. It follows far less of a shift, and it covers more event-free
  rows only because its band is wider. Five seeds did not tighten the event-free comparisons enough to
  prove that O2 equals D5, so that swap rests on the point estimates and on explainability.
advice:
  - >-
    Replace the G4 gate with the class 2 candidate scaled to K1's band width only if a wider
    event-free band is acceptable: injected coverage and safe-side share are within 0.01 of K1's, and
    event-free coverage is 0.02 to 0.03 higher (class 2 results bullet).
  - >-
    Do not add seeds to tighten event-free contrasts between networks; their intervals are set by the
    16 Primaries and 5 months, and seeds 0 to 4 gave half-widths of 0.02 to 0.06 in coverage (O2
    results bullet).
  - >-
    Keep E3: against the 3-day embargo it costs at most 0.01 on event-free coverage at the eight MVA
    Primaries, and lowers absorption by 0.045 to 0.062 (E3 results bullet).
  - >-
    Measure absorption with more than 12 injected designs before using a 0.03 tolerance; the
    intervals here reach 0.02 to 0.05 on each side (E3 results bullet).
production_status: candidate
adopted_in: none
---

**Experiment 612 left three claims resting on point estimates inside the tolerance. This experiment
refitted the arms behind them on three more seeds and replaced the point estimates with intervals.**
We asked whether three networks can be swapped without changing the forecast by more than a stated
tolerance. The three swaps are the class 2 candidate for the gate, the steadier offset O2 for the
slow offset D5, and the 3-day refit embargo for the E3 rule. We refitted each network on seeds 2 to
4, joined them to [experiment 612](../00612-nra-combined-networks-three-class-roster-completion/result.md)'s
seeds 0 and 1, and scored the mean of the five seeds' quantiles on the five development months
(2025-10, 2025-12, 2026-01, 2026-03, and 2026-06). The class 2 candidate keeps its injected-load-shift
coverage and loses nothing on safe-side share, so it can replace the gate on those measures. The two
other swaps stay unproven, because five seeds left the intervals as wide as before.

**Terms.** A Primary is a primary substation: the eight MVA Primaries are metered in megavolt-amperes
and the six MW Primaries in megawatts. K1 is the lag neural network with the G4 gated band (a band
that widens when a load-shift is detected). K2 is the same fits without the gate. K3 is the lag-free
neural network. The class 2 candidate is K2 plus neighbour summaries and 50% signal dropout from
[experiment 610](../00610-nra-neighbour-summary-lag-encoder-inputs-iteration/result.md), scaled
about its median so its band is as wide as K1's. E3 is a 7-day embargo on the newest rows of the
daily refit, with half weight on the 7 days before that. Follow is the share of an injected
load-shift's depth that the median forecast tracks. Absorption is the share the median forecast takes
up when the network is refitted on rows that contain the shift. Safe-side share is the share of rows
where the normal running arrangement (NRA, the load pattern without the load-shift) lies on the safe
side of the band's edge. Coverage is scored against an estimate of the NRA. A contrast is equivalent
only if its whole Bonferroni interval lies inside the tolerance: 0.01 for coverage, follow, and
safe-side share, 0.03 for absorption, and 0.0006 for event-free continuous ranked probability score
(CRPS), the minimum detectable effect of [experiment 591](../00591-shared-lag-encoder-auxiliary-shift-depth-loss/result.md).
"Mde" is half the interval width.

## Figures

![Class 2 candidate minus K1](intervals_class2.svg)

Each dot is the class 2 candidate at K1's band width minus K1 (CRPS: minus K2), with its Bonferroni
interval (12 counted contrasts, plus 2 for absorption). The blue band is the tolerance. Look at the
top panel, where both injected intervals lie inside the band and every event-free interval lies
above it, and at the bottom panel, where absorption is lower but the intervals touch the band.

![O2 minus D5](intervals_o2.svg)

Each dot is the O2 offset minus the D5 offset, tail constants on both (30 counted contrasts). Look
at how wide every event-free interval is compared with the band.

![3-day embargo minus E3](intervals_e3.svg)

Each dot is the 3-day embargo minus E3 (12 counted contrasts, plus 2 for absorption). Look at the
injected panels, where every interval lies inside the band, and at absorption, where E3 absorbs less.

![D5, K1 and the class 2 candidate](event_D5_k1_c2.svg)

![D16, K1 and the class 2 candidate](event_D16_k1_c2.svg)

The December 2025 load-shift at Primaries D5 and D16, one network per panel, day-ahead forecasts
from 14 days before the start to 14 days after the end. The dashed lines are the start and end dates
the maintainer gave. The pale shade is the window verified in [experiment 580](../00580-verified-load-shift-event-set-and-label-audit/result.md),
about one day earlier. The blue strip under K1 is the gate: it is on from about 1 January, not over
the whole axis. Coverage is scored against an estimate of the NRA, which these panels do not draw.

![A quiet week, K1 and the candidate](quiet_week_D2_k1_c2.svg)

![A cold winter night, K1 and the candidate](cold_night_D2_k1_c2.svg)

![A very sunny day, K1 and the candidate](sunny_day_D5_k1_c2.svg)

Three further situations at the same scale. Look at whether the candidate's band stays as wide as
K1's when the gate is off.

## Results

- **Seeds and rows.** Seeds 0 and 1 are experiment 612's stored fits. Seeds 2 to 4 are new fits,
  with pre-training and step-2 weights fitted afresh. A fit of one cell on seed 0 in the new chain
  matched the stored fit to 6e-7 in the median (seed 1 differs from seed 0 by 0.02). K3 with O2 had
  pre-training for seeds 0, 1, and 5 to 9 only, so every K3 contrast between O2 and D5 uses seeds 0,
  1, 5, 6, and 7 for both offsets. The same contrasts on seeds {0, 1} alone and on seeds {2, 3, 4}
  alone (K3: {5, 6, 7}) keep every sign. The five-seed intervals are inside the tolerance for the
  injected class 2 coverage contrasts, while the two seed subsets alone are inconclusive on them.
- **Class 2 candidate against K1** (12 counted contrasts; 2 more for absorption; band width set from
  the candidate's own month, as in experiment 612). Day 4 to the end: coverage -0.0043 (-0.0097 to
  +0.0011; mde 0.0054), safe-side share +0.0014 (-0.0040 to +0.0068), follow -0.0595 (-0.0657 to
  -0.0537, better). 0-7 days after the end: coverage -0.0033 (-0.0090 to +0.0027). Levels: K1 covers
  0.926 and 0.918, the candidate 0.921 and 0.915. Event-free coverage is higher by 0.019 to 0.030
  (all four cells above the tolerance) because the band is wider everywhere. Event-free CRPS minus
  K2: +0.0003 and +0.0005 at 24-120 h (intervals -0.0002 to +0.0006 and +0.0002 to +0.0010),
  0.0000 and +0.0003 at 120-240 h (-0.0006 to +0.0006, and +0.0001 to +0.0006); one cell (six MW,
  120-240 h) is equivalent. With each month's width set from the other four months, the injected
  contrasts move by at most 0.001.
- **Absorption, candidate against K1 and K2** (experiment 599's E3 arm, 12 designs, seeds 0 and 1,
  2 counted contrasts). Day 4 to the end: -0.056 (-0.089 to -0.029). 7 days after the end: -0.050
  (-0.074 to -0.029). The candidate absorbs less, but both upper ends sit just above -0.03, so the
  verdict is inconclusive.
- **O2 against D5** (30 counted contrasts; four months with tail constants). No event-free
  contrast is equivalent. Half-widths are 0.021 to 0.057 in coverage and 0.001 to 0.002 in CRPS,
  against 0.038 and 0.0017 for the matching cells of experiment 612, so seeds 2 to 4 did not
  narrow them. K1 injected 0-7 days after the end: O2 covers 0.012 more (0.007 to 0.017); K2:
  +0.020 (0.013 to 0.026, better). K1 follow: -0.009 (-0.011 to -0.007; lower is better). K3 at the
  eight MVA Primaries: coverage -0.011 (24-120 h) and -0.016 (120-240 h), intervals reaching +0.010
  and +0.005.
- **E3 against the 3-day embargo** (12 counted contrasts; five months). Equivalent on all four
  injected contrasts: coverage -0.0040 (-0.0061 to -0.0018) and +0.0052 (+0.0034 to +0.0071), follow
  +0.0021, safe-side share -0.0029. Event-free, equivalent at the eight MVA Primaries at 24-120 h
  (coverage -0.0032, CRPS +0.0001) and inconclusive in the other six cells, with the six MW
  Primaries 0.009 more covered without E3 (0.0004 to 0.019 at 24-120 h). Absorption (experiment
  599's E0 and E3 arms, 12 designs, 2 counted contrasts): the 3-day embargo absorbs 0.062 more from
  day 4 to the end (0.020 to 0.115) and 0.045 more in the 7 days after (0.020 to 0.078). The
  direction is clear and the size is inconclusive against 0.03.
- **Predictions.** P1 held (injected coverage intervals inside 0.01). P2 failed (O2 is not
  equivalent in K1 or K2 event-free). P3 held for the point estimates and not for the interval ends.
  P4 held: the intervals shrank by less than a third (K1 six MW, 24-120 h: half-width 0.038 against
  0.0385).

## Table of recommended networks for the maintainer

| Class | Recommended network | Evidence now | One-line recommendation |
|---|---|---|---|
| 1 | K1 with E3, O2 or D5, tail constants | E3 matches the 3-day embargo on every injected contrast (intervals inside 0.01); O2 is not shown equal to D5 (half-widths 0.02 to 0.06) | Keep E3. Choose O2 for explainability and accept an unproven swap, or keep D5 to stay inside the evidence. |
| 2 | Candidate at K1's band width | Injected coverage -0.004 (-0.0097 to +0.0011); safe-side share +0.001; follow 0.051 against 0.110; event-free band wider by 0.02 to 0.03 coverage | It can replace the gate on injected coverage and safe-side share. Its event-free CRPS is not shown equivalent in 3 of 4 cells. |
| 3 | K3 with tail constants | O2 against D5 on seeds 0, 1, 5, 6, 7: coverage -0.011 to -0.016 at the eight MVA Primaries, intervals include 0 | Keep D5 until a test with more Primaries shows O2 is equal. |

## Checks

- **Median minus mean forecast.** Not computed: these forecasts are stored as 11 quantiles, with no
  mean.
- **Median error beside the quantile scores.** Follow (a median measure) and coverage point the same
  way in every contrast, so the quantile-score and median-error contrasts agree.
- **Audit triggers.** None fired. P2 failed, and the explanation (resampling over 16 Primaries and 5
  months, not over seeds) was tested: the half-width did not change with five seeds.

## Not run

- Absorption of K1 and the candidate on seeds 2 to 4: the 12-design absorption fits were not
  repeated, so absorption keeps seeds 0 and 1.
- K1 and K3 absorption with O2.
- July 2026 onwards is untouched.

## Reproduction notes

- **Code reused (copied into this folder).** Experiment 612's folder (K1, K2, K3, candidate, tail
  constants, scoring, tables), which copies experiments 565, 574, 577, 584, 591, 592, 595, 597,
  599, 601, and 610. New in this folder: `e00613_link.py` (joins seeds 0 and 1 to the new seeds),
  `e00613_widths.py` (the band-width factors), `e00613_roster.py` (the contrasts), and the
  `CHAIN613` and `--seeds` options.
- **Fits and compute.** About 4 GPU process-hours, under the ceiling of 10. Fits are in the
  private data folder.
- **Caveats.** NGED's logs may omit load-shift events, so the event counts are lower bounds.
  Coverage is scored against an estimate of the normal running arrangement. The injected bank
  changes only the lag columns, which flatters lag networks. Nothing is adopted for production.
