# Differentiable physics (demand, calendar, and weather): state and next steps

**The convex model of Primary demand is a readable alternative to lagless XGBoost, but none of the
terms added to it has a confirmed gain.** The convex model fits a daily profile, holiday terms, a
heating term, a lighting term, and a solar term by convex optimisation, a method that has a single
best answer. "Lagless" means the XGBoost model receives no recent readings of the substation's own
power. This page is a selection of the theme's experiments.

**GAM-K is the version of the convex model that the series chose as its candidate.** GAM is short
for generalised additive model, a model that adds one smooth curve per effect. GAM-K is a GAM
fitted on 24 months of history, with no drift term, with the solar capacity pinned (fixed to a value
estimated beforehand), and with a Kalman filter. The Kalman filter corrects each forecast with its
recent errors.

**None of this theme's terms appears in the final models.** The Primary demand forecast in the
[final models](../../final-models.md) is a neural network, and whether it already captures the
winter night bias is untested.

**Scores are quantile scores unless stated, in points of effective capacity.** One point is 1% of a
substation's capacity, and a lower score is better, so a negative contrast favours the added term.
Experiment 63 and Experiment 122 score mean absolute error of the median instead. The experiments
use past weather unless stated, so each gain is a ceiling for the weather forecasts used in production.
Verdict words (`supported`, `partly supported`, `small gain`, `not supported`, `inconclusive`) are
defined in the [glossary](../../glossary.md#pre-registration-decision-rule-and-verdict-words).
Experiment 126 adds three labels. `undecidable` means the interval is too wide to read, `tie` means
the arms differ by 0.02 points or less, and `not powered` means the known-answer test could not show
that the effect is detectable.

## What works

**The Kalman filter makes GAM-K better than both plain models in each of three winters.**
[Experiment
122](../../00122-differentiable-physics-three-winter-backtest-gam-k/result.md) (verdict
`supported`) scored mean absolute error. GAM-K beat the plain GAM by 1.83 to 3.06 points and lagless
XGBoost by 2.00 to 3.10 points in all three winters, and every contrast met the 1-point bar. Two of
the winters were partly in the data the models were fitted on. After subtracting each model's
winter-mean error, 5 of 6 contrasts miss the 1-point bar, so most of the gain is level tracking.

**Splitting each heating slope between the two halves of its hour band lowered the filtered score by
0.039 points.** [Experiment
114](../../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md) gives an
interval of -0.070 to -0.014 and a gain at 5 of 5 substations in both years, with the score falling
from 2.255 to 2.216. About 38% of the gain appears on warm rows where every heating term is zero,
and without substation D5 the estimate is -0.030, which is `small gain`. The verdict is `partly
supported`.

## What is inconclusive, does not work, or is unproven

**A winter night offset raised the 0.95 quantile's coverage on cold winter nights from 80% to 93%,
but the result is `inconclusive`.** In [Experiment
137](../../00137-differentiable-physics-e7-economy-7-night-step-term-in-the-forecast/result.md), a
term fitted on the day-ahead forecast's own past errors cut the mean error on those rows from 2.5 to
0.2 points and lowered the score by 0.484 points (99% interval -0.777 to -0.169) on 365 headline
rows. Over all rows the gain is only -0.032 points, and the term worsened the 22:00 to 23:00
half-hours. A plain winter night offset with no temperature does as well, so the experiment found a
winter night bias and not a step that grows with cold. The shuffled-night placebo gained as much as
the term, so the test could not separate the term from chance, and the data covered one winter.

**Fitting night levels inside the model beat the winter night offset by 0.02 to 0.04 points, which
is `inconclusive`.** [Experiment
281](../../00281-economy7-night-step-smooth-in-temperature/result.md) found the gain on all rows and
on winter nights in both years. The gain in the two hours after the step starts changes sign
between the two years. A step that grows with the cold added almost nothing over free night levels.

**Day-type interactions, a per-site heating threshold, and a learned year cycle add nothing.** In
Experiment 114, letting heating or lighting differ between weekdays and weekends moved the score by
+0.002 and -0.0001 points, and a per-site heating threshold moved it by -0.012 points with an
interval spanning zero. In [Experiment
126](../../00126-differentiable-physics-s1-cheap-screens-added-to-gam-k/result.md), the learned year
cycle (W-basis, a periodic curve fitted in place of the fixed seasonal terms) made the score 0.091
points worse (interval 0.025 to 0.149) and was worse in every month, site, and year.

**None of 13 low-compute screens showed a trusted gain.** Experiment 126 labelled seven screens
`undecidable`, two `tie`, two `not powered`, one `worse`, and one `inconclusive`. Lighting from solar
elevation (E-sol) came closest, at -0.036 points on dark winter evenings with an interval of -0.122
to +0.024. G-skf, a Kalman filter fitted to the residuals, found injected load transfers 4 to 7
times as often as the plain filter at 25% and 40% transfers, but failed its power gate, so it is a
detector only.

**A smooth heating curve, a thermal lag, and wind each left the score within 0.02 points of the
straight-line heating response.** [Experiment
279](../../00279-dp-heating-curve-monotone-lagged-temperature-and-wind/result.md) (`not supported`)
tested these on cold nights. A small neural network lowered the cold-night error by 0.13 points, but
the gain came from one cold spell and failed a known-answer test.
[Experiment 413](../../00413-differentiable-physics-demand-network-part-1/result.md) (`not
supported`) built a neural network from readable shared parts, and it scored 0.17 and 0.23 points
worse than the plain neural network at six MW and eight MVA Primaries.

**A neural network that reads the weather around a substation gained 0.062 and 0.035 points, and about
two-thirds of the gain survived another date's weather, which suggests quantile recalibration.**
[Experiment
119](../../00119-differentiable-physics-enc-weather-patch-encoder-on-ens/result.md) scored one year
in each direction.
The known-answer gate failed, so the verdict is `inconclusive`.

**More history is not shown to help, even with a drift term.** In [Experiment
104](../../00104-differentiable-physics-2-simple-structural-model/result.md), all years of history
without drift scored 0.25 points worse than 12 months. With a drift term, the contrast between all
years and 12 months was -0.04 (interval -0.30 to +0.21), and the all-years model scored 2.78 against
2.82 for 12 months. Four of five positive controls (synthetic tests of an effect known to exist)
failed, so the verdict is `inconclusive`. The same experiment's 12-month convex model beat lagless
XGBoost by 0.25 points (post hoc, chosen after seeing results). Drift is covered by the theme
[Differentiable physics: long history and
drift](../differentiable-physics-long-history-and-drift/index.md).

**A hindsight solar term lowered the error by 0.322 points, short of the 0.35-point threshold.**
[Experiment 63](../../00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md) compared a linear fit
of baseline, solar, and load-transfer terms with the frozen Flexpectation v1 XGBoost baseline, in
mean absolute error. The interval is -0.437 to -0.207, with sunshine and capacity taken from the
past. The injection checks failed for reasons of test design, so the verdict is `inconclusive`. The
load-transfer terms were barely exercised (2 event-substation pairs).

## Open questions

- **Whether a seasonal night offset removes the night bias is unknown.** Experiment 137 cannot
  separate it from a cold-driven step with one winter of data, and Experiment 281 answers only part
  of the question.
- **Whether the half-band heating split survives weather forecasts and monthly refits is unknown.**
  Experiment 114 used past temperature only.
- **Whether the weather encoder's gain is a recalibration of the quantiles is untested directly.**
  Experiment 119 suggests so through the wrong-date copy.

## Next-experiment options

**An option is to test lighting from solar elevation again on forecast weather.** The source is
Experiment 126's "What to try next". The test would use the ECMWF ensemble (ENS) weather forecast,
the earlier winters, and a row set that narrows the interval below 0.03 points.

**An option is to test a seasonal night offset on a second winter (2026-27).** The source is
Experiment 137's advice. It needs at least 50 cold-night rows per substation and a placebo that
shuffles temperature within each calendar month.

**An option is to confirm the half-band heating split on ENS weather with monthly refits.** The
source is Experiment 114's advice, before any port into GAM-K.

**An option is to run a plain recalibration of GAM-K's quantiles first.** The source is Experiment
119's "What to try next" and the research plan. The recalibration would use the centre box's extra
fields (dew point, pressure, precipitation, and member spread). The plan queues a weather encoder
retest on convective and frontal days after it, because Experiment 119 found the surrounding boxes
add nothing.

## Parked

**The per-site heating threshold, the learned year cycle, and the weather-patch encoder are
parked.** Experiments 114, 126, and 119 found no trusted gain from them.

**Load-shift event detection is paused until January.** The state at the pause is in
[the load-shift detector pause page](../lsc-detectors/state-at-january-pause.md). The unmetered PV
(photovoltaic) capacity work is paused until January as well.

The background to the models is in [Differentiable physics:
background](../differentiable-physics-background.md).
