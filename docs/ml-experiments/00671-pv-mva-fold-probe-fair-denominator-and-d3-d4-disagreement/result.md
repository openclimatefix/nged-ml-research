---
id: "00671"
slug: pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement
theme: solar-disaggregation
also_relevant_to: []
status: finished
kind: other
size: >-
  medium. Stored: probe results and figures only, in this experiment's private folder. Serving
  path: not touched. Degradation rule: not touched. Designs: Part 1 is a re-scoring of stored
  experiment 662 probe outputs with a denominator fixed here; Part 2 re-probes stored networks, and
  the two fixed-reactive-term settings are listed below, so one design is defensible. Callers: the
  scripts import experiment 662's modules and edit none of them. Asset graph: not touched. CPU
  only, at most 2 processes while the maintainer's backtests run; ceiling 4 wall-clock hours.
hypothesis: >-
  Experiment 662's bar A1 failed because its denominator was the fold model's recovery on a
  signed target that never folds, where the fold model collapses. With the linear probe's
  signed-world recovery as the denominator, the apparent-power conversion costs no recovery at
  Primaries where the floor does not apply. The linear and fold probes disagree at D3 and D4 on
  the real twins because the fold model reads extra capacity from its reactive term, not because
  the curve is folded.
prediction: >-
  Fixed before any number is read. A1' passes for L0 and L1 at 0.5 times and fails for them at 1
  and 2 times, because experiment 662's linear probe falls to -0.12 at 2 times in the PF world
  while staying at 0.78 in the signed world. A1' for FA passes at 0.5 and 1 times and is uncertain
  at 2 times. D2 to D21 curves are monotone falling or flat, not V-shaped, and at D3 and D4 more
  than half of the fold model's extra capacity over the linear probe disappears when the reactive
  terms are fixed at zero.
decision_rule: >-
  Fixed before any number is read. A1' is the PF recovery of probe X (L0, L1, FA) divided by the
  signed-world recovery of the linear probe L0 at the same Primary and level (0.5, 1, 2 times), per
  Primary, median over seeds, with the signed-world denominator kept only where its absolute value
  is at least 0.05. A1' passes for a probe at a level when at least 5 of the 6 MW Primaries lie in
  0.90 to 1.10. If A1' passes for FA at all three levels, the fold model has no fold-free loss and
  experiment 662's failure was the denominator. If it passes only for L0, recommend L0 for the five
  Primaries that do not fold and FA only for D12-type exporters. If it passes for neither,
  quantify the loss of recovery in percentage points by Primary. A second reading divides FA's PF
  recovery by L0's signed recovery and reads the ratio as a bias statement. Part 2 ends with a
  one-table recommendation per Primary: probe, capacity range with the calibration assumption,
  confidence (high, provisional, report-only). No bar is changed after a number is read.
benchmark: >-
  The linear probe L0 in the signed world of the same farm and level (Part 1). The known-answer
  recoveries in native PF worlds at D2, D3, D4, D19, and D21 (A2 rows of experiment 662) and the
  injected farms at D5 and D10 (Part 2).
row_set_id: >-
  Experiment 662's stored 2026-06 probe outputs and networks (five seeds; development data only,
  nothing dated 2026-07-01 or later).
metric: >-
  Recovery ratios (A1'), effective AC capacity in MW per Primary and probe, and the share of the
  fold model's extra capacity that comes from its reactive terms.
production_status: not-for-production
adopted_in: none
verdict: partly supported
best_arm: >-
  For the five MVA Primaries that do not fold: the linear probe L0 (equal to the fold model with its
  reactive terms fixed), scaled by the native-world recovery at 0.5 times.
headline: >-
  A1' fails for all three probes. Where the floor does not apply (D9, D11, D14, D16), the linear
  probe loses 5 to 12 points of recovery at 0.5 times, 22 to 25 at 1 times, and 75 to 94 at 2
  times; the fold model reads within -6 to +18 points of the signed-world linear recovery at 0.5 to
  2 times. At the five MVA Primaries 82 to 94% of the fold model's extra capacity at D2, D3, D4, and
  D19 disappears when its reactive terms are fixed.
lessons:
  - >-
    The apparent-power conversion costs the linear probe recovery even where the floor does not
    apply, in proportion to the injected size, and costs the fold model little at the median
    but with a spread of -6 to +18 points between Primaries.
  - >-
    The fold model's extra capacity over the linear probe at D2, D3, D4, and D19 comes from its two
    free reactive terms: fixing them at zero returns the linear probe's capacity to within 0.5 MW.
  - >-
    At D4 the free fold fit is bimodal across seeds (1.1 MW at two seeds, 3.1 to 4.0 MW at three),
    because the reactive floor sits at zero or at 3.5 MW.
  - >-
    Past the fold (D5 and D10, injected 2.0 MW), the capacity read depends more on how the reactive
    terms are set than on the probe: recovery ranges from 0.0 to 1.16 across the fold variants.
advice:
  - >-
    Use the linear probe, calibrated by the native-world 0.5-times recoveries (0.63 to 0.82), for
    D2, D3, D4, D19, and D21, and state the fold model's free-reactive capacity only as an upper
    bound.
  - >-
    Do not read a capacity from the fold model with free reactive terms at one seed; report the
    spread over seeds.
take_home_message: >-
  Turning a substation's flow into apparent power does cost the linear solar probe some recovery
  even where the flow never nears zero, and the cost grows with the size of the solar farm. The
  fold model avoids most of that loss, but its extra capacity at D2, D3, D4, and D19 comes almost
  entirely from two free reactive-power terms. For those five Primaries the linear probe, scaled by
  the recovery measured on injected farms, is the better-supported reading, with the fold model's
  reading as an upper bound.
---

# Does the fold model lose recovery with a fair denominator, and which probe should be trusted at D2, D3, D4, D19, and D21?

**Converting a flow to apparent power costs the linear probe recovery even where the flow never
nears zero, and the fold model's extra capacity at D3 and D4 comes from its free reactive terms,
not from a folded curve.** The fair-denominator test A1' fails for all three probes (decision-rule
branch 3: quantify the loss). Fixing the two reactive terms of the fold model at zero returns the
linear probe's capacity at D2, D3, D4, and D19 to within 0.5 MW. For D2, D3, D4, D19, and D21 we
recommend the linear probe, scaled by the recovery measured on injected farms, with the fold
model's free-reactive reading as an upper bound. Both parts re-use experiment 662's stored networks
(2026-06 cutoff, 5 seeds) and its probe outputs; no network was trained.

![A1': PF recovery over the signed-world linear recovery, by probe, level, and Primary](a1_prime.svg)

**Figure: L0 sits near 0.9 only at 0.5 times, L1 also at 1 times, and the fold model scatters
around 1.**
Each dot is one Primary at one level. D5 and D10 sit past the fold in every injected world (2.0 MW
floor), so their ratios are shown apart.

![S(u) of one probe cell per Primary with the linear and fold fits](s_of_u.svg)

**Figure: every curve is monotone falling or flat, not V-shaped.** D3 and D4 are nearly flat
(under 2 MVA across the range of u), and the three fits lie on top of each other in this cell.
The disagreement between the probes therefore comes from the fits across cells, not from the curve
class in one cell.

## Part 1: A1' at the six MW Primaries (median over seeds, first farm rotation)

Terms: L0 is the linear probe, L1 the linear probe on rows above the reactive floor, and FA the
fold model with a night prior (all as in experiment 662). "PF" is the apparent-power world with
power factor 0.97. A ratio is valid where the signed-world denominator is at least 0.05.

- **Bar.** 0.90 to 1.10 at 5 of 6. It passes for no probe at any level. Valid ratios in the bar:
  L0 2, 0, 0; L1 2, 2, 0; FA 3, 2, 2 of 6 at 0.5, 1, 2 times.
- **Loss in points of recovery (PF minus signed L0), D9, D11, D14, D16.**

| Probe | 0.5 times | 1 times | 2 times |
|---|---|---|---|
| L0 | -5 to -12 | -22 to -25 | -75 to -94 |
| L1 | -6 to -12 | -2 to -9 | -78 to -95 |
| FA | -6 to +10 | -4 to +18 | +1 to +13 |

- **Pre-registered predictions.** The plan predicted that A1' passes for L0 and L1 at 0.5 times
  and for FA at 0.5 and 1 times. All three failed (L0 2 of 6, L1 2 of 6, FA 3 and 2 of 6), so the
  first hypothesis (no loss where the floor does not apply) is refuted for L0 and L1 and not
  supported for FA. The prediction for Part 2 (curves not V-shaped; most of the extra capacity gone
  with the reactive terms fixed) held.
- **Deviation.** The plan said median over seeds and both farm rotations; only the first rotation
  is used, because the signed arm has no second rotation.
- **Signed-world L0 denominators** are 0.73 to 0.82 at all six Primaries and levels.
- **D5 and D10** (2.0 MW at every level, past the fold): L0 and L1 lose 79 to 115 points; FA loses
  53 to 57 points at D5 and 6 to 12 points at D10.
- **Bias reading (FA over signed L0).** FA's ratio is 0.85 to 1.22 at D9, D10, D11, D14, and D16
  and 0.30 to 0.33 at D5. FA therefore recovers about as much of the signed-world capacity as the
  linear probe does in the signed world, except at D5, and overshoots by up to 22% at D11 and D16.
- **Reading against the decision rule.** Neither FA nor L0 passes, so the apparent-power conversion
  costs recovery even where the floor does not apply. The cost is 5 to 12 points for the linear
  probe at 0.5 times and 22 to 25 points at 1 times.

## Part 2: D2, D3, D4, D19, D21 (and D5, D10)

Terms: a "twin" is the network trained on the real data with no injected farm. "q = 0" fixes both
reactive terms of FA at zero; "q1 = 0.25" fixes the reactive share at the power-factor 0.97 value.
The midday reading is the median flow on sunny half-hours from 10:00 to 15:00 UTC, April to August.
The "A2 rows" are experiment 662's recoveries in native apparent-power worlds.

| Primary | Midday reading (MW) | L0 (MW) | FA free (MW) | FA, q = 0 | FA, q1 = 0.25 | Share of FA's extra gone with q = 0 |
|---|---|---|---|---|---|---|
| D2 | 8.84 | 5.20 | 7.95 (7.4 to 8.2) | 5.70 | 6.05 | 82% |
| D3 | 6.62 | 2.06 | 4.44 (2.7 to 4.7) | 2.26 | 2.37 | 91% |
| D4 | 7.09 | 0.98 | 3.09 (1.1 to 4.0) | 1.10 | 1.15 | 94% |
| D19 | 2.56 | 1.13 | 2.13 (1.3 to 2.4) | 1.28 | 1.36 | 85% |
| D21 | 2.94 | 1.13 | 1.35 (1.2 to 1.8) | 1.26 | 1.32 | 42% |

- **Real twins, raw capacity**, median over 5 seeds (min to max for the free fold fit).
- **Reactive floor of the free fit.** The free fit's q0 is 3.3 to 5.7 MW at D2, 2.1 to 5.1 MW at
  D3, and 0 or 3.5 to 3.8 MW at D4, where two of five seeds find no floor and read 1.1 MW.
- **Known-answer recoveries in native PF worlds (experiment 662, A2 rows).** The linear probe
  recovers 0.63 to 0.82 at 0.5 times and 0.51 to 0.64 at 1 times, and is negative at 2 times
  (-0.10 to -0.19). The fold model recovers 0.49 to 0.97 at 0.5 times, 0.54 to 0.91 at 1 times, and
  0.73 to 0.92 at 2 times. The calibrated linear capacities are 0.2 to 0.5 times the midday
  reading at D3 and D4, where the linear probe recovers 0.74 to 0.82 at 0.5 times, and about 0.5 to
  0.9 times it at D2, D19, and D21, where the linear recovery falls to 0.51 to 0.63 at 1 times.
- **Known answer at D5 and D10 (injected 2.0 MW, past the fold).** Recovery per probe: L0 -0.34 and
  -0.03; FA free 0.24 and 0.65; FA with q = 0 1.16 and 0.00; FA with q1 = 0.25 0.01 and 1.07. No
  probe reads both correctly, so past the fold the reactive-term setting decides the answer.

## Recommendation

Capacity ranges divide the raw linear capacity by the native-world 0.5-times recoveries (0.63 to
0.82) of experiment 662; the "upper bound" divides the free fold fit by 0.71, the median recovery
of the fold model at 1 times at the six MW Primaries (a different level from the linear divisor,
because the fold model's 0.5-times recovery at D3, D4, and D19 is lower). Both assume power factor
0.97.

| Primary | Probe to use | Capacity range (MW) | Upper bound, free fold (MW) | Confidence |
|---|---|---|---|---|
| D2 | L0 | 6.3 to 8.3 | 11.2 | provisional |
| D3 | L0 | 2.5 to 3.3 | 6.2 | provisional |
| D4 | L0 | 1.2 to 1.6 | 4.3 | provisional |
| D19 | L0 | 1.4 to 1.8 | 3.0 | provisional |
| D21 | L0 (the fold model's median is 0.2 MW higher) | 1.4 to 1.8 | 1.9 | provisional |
| D5, D10 | none | not defined | not defined | report-only |

- **Why L0 for D3, D4, and D19.** Fixing the reactive terms at zero or at the power-factor
  value returns the linear capacity, so the extra 1.0 to 2.4 MW of the free fold fit at D3, D4,
  and D19 is a reactive-term effect that the injected farms give no sign of supporting. The fold
  model recovers less than the linear probe at 0.5 times at D3, D4, and D19 (0.68, 0.55, 0.49
  against 0.82, 0.74, 0.63).
- **What stays open.** The injection tests difference a twin against a fleet, so they cannot show
  whether the twin's own absolute read of 3 to 4 MW is true. A real reactive floor of 3 to 5 MW at
  D3 and D4 would also produce the free fit's answer.

## Caveats

- One cutoff (2026-06) and one network architecture; the 2026-03 networks are not re-probed.
- The reactive power of the real meters is unknown, and power factor 0.97 is an assumption.
- D5 and D10 have midday readings near 1 MW, so a 2.0 MW floor puts them past the fold at every
  level.
- NGED's event labels are incomplete, so the clean rows may hold unlabelled events.
- The S(u) lines use one cell and seed 0; the linear line omits the temperature term.

## Checks

- **Forecast checks:** the experiment scores no forecast, so the median-minus-mean and
  quantile-score checks do not apply.
- **Code review (Sonnet):** no wrong-number bugs. It asked for the plan to say that the first farm
  rotation only is used (done) and noted that the temperature term is omitted from the plotted
  linear line (stated).
- **Compute:** CPU only, 2 processes; Part 2 took about 15 minutes.
- **Reproduce:** `e00671_a1.py`, then `e00671_part2.py` (twice, for the twin and the injected
  networks), then `e00671_report.py`, in this experiment's folder.

## What the next experiment should answer

Does a reactive floor of 3 to 5 MW exist at D3 and D4? A test needs the real meters' active and
reactive power, which the meters may record.
