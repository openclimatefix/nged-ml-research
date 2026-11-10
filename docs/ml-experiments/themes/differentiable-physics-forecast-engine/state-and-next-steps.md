# Differentiable physics, forecast engine: state and next steps

**A demand forecast built from physical terms beat the production-style XGBoost forecast (BG2) by
0.136 points over one year at six MW Primary substations, but only from November to March.** From
April to June the same forecast is worse than BG2. The year is July 2025 to June 2026, the only
year with BG2 forecasts, and the gain's 95% interval (-0.277 to -0.005) just excludes zero.

A *Primary* is a primary substation. A *point* is 1% of a substation's effective capacity, and the
*quantile score* is twice the mean pinball loss over 11 quantiles, in points. Negative differences
between scores are better. Verdict words such as `inconclusive`, `partly supported`, `small gain`,
`undecidable`, `tie`, and `not powered` are defined in the
[glossary](../../glossary.md#pre-registration-decision-rule-and-verdict-words). BG2 is the recipe of
[Experiment 87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md). The
[final-models page](../../final-models.md) uses none of this theme's models for Primary demand.

The *convex demand model* sums physical terms: daily profile, holidays, heating, lighting, and
unmetered rooftop solar PV (photovoltaic generation). *GAM-K* is a generalised additive model (GAM),
meaning the convex demand model, plus a Kalman filter on its recent errors. *Recipe R* subtracts
0.75
times the mean error of the last 14 days. *Arm D* is GAM-K with a winter night offset
and recipe R ([Experiment
155](../../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md)). The *stack*
is arm D, a small neural-network correction of the median (*H*), and a two-number summary of the
ensemble's solar response (*S*: mean and spread per half-hour). *ENS* is the ECMWF 51-member
ensemble
forecast and *ERA5* is the ECMWF estimate of past weather.

## What works

**On ENS weather, the Kalman filter lowers the convex model's score by 0.151 points.** The gain is
0.392 points at leads under 24 hours ([Experiment
111](../../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)). The same
filter lowers lagless XGBoost by about the same amount (-0.154, nearly all at day 14).

**On ERA5 weather, much of the filter's gain is a level correction.** In mean absolute error of the
median, the filter's gain was 1.8 to 3.1 points. After debiasing, GAM-K stays ahead by only 0.35 to
1.05 points, and 5 of 6 contrasts would miss the 1-point bar ([Experiment
122](../../00122-differentiable-physics-three-winter-backtest-gam-k/result.md)).

**GAM-K cannot be told apart from BG2 at five substations.** It scored 2.80 against 2.78
(-0.109 to +0.157).

**The correction network and the solar summary add to each other.** Together they lower the quantile
score by 0.136 points (5%), 99% of the sum of their separate gains ([Experiment
220](../../00220-dp-bg-stage4-assembly-with-drop-one/result.md)).

**Over the one scored year, the stack beats BG2 and the manual heuristic.** The stack beats BG2 by
0.136 points (9 of 12 months, a negative estimate at all six substations, two intervals excluding
zero) and the manual heuristic by 0.818 points (12 of 12 months) ([Experiment
251](../../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md), [Experiment
228](../../00228-dp-bg-rolling-origin-stack-against-bg2/result.md)).

**Coasting the filter through a reading gap beats the uncorrected model.** Coasting (running the
filter without new readings, with widened bands) scored 0.25 to 0.44 points better at every gap
length (3, 6, 24, and 72 hours). Against dropping only the filter the result is `undecidable`: all
four
intervals include zero (3 hours -0.147, -0.321 to +0.030; 72 hours +0.013, -0.152 to +0.203)
([Experiment 205](../../00205-dp-gamk-gap-robustness-coast-or-frozen-gam/result.md)).

## What does not work

**Most additions to the filter and the fit gave no usable gain.** Of 13 low-compute additions, 7
were `undecidable`, 2 a `tie`, 2 `not powered`, and 1 (a learned year cycle on the
W-basis, a set of smooth weather functions) clearly worse
([Experiment 126](../../00126-differentiable-physics-s1-cheap-screens-added-to-gam-k/result.md)).

**Fitting the demand model through the filter made the forecast worse.** Tying the fit back to the
two-stage fit removed the damage and the gain ([Experiment
135](../../00135-differentiable-physics-n10-joint-kalman-and-demand-fit/result.md)).

**A drift term gains about 0.19 points without the filter, and
almost nothing with it.** With the filter it worsens the score by 0.031 points on 24 months (99%
interval +0.008 to +0.059) and by 0.011 on all years. The earlier 0.40-point gain was on a weaker
base without ensemble
weather, offset, or recipe R ([Experiment
170](../../00170-dp-bg-stage3-drift-and-holiday-scoring/result.md)).

**Recipe R carries the arm D gain, not the winter night offset.** Recipe R changes the score by
-0.037 (99% interval -0.062 to -0.016). The offset adds +0.002 (-0.004 to +0.009) on top of it,
and a placebo offset gained as much (`inconclusive`).

**Running the error table around each of the 51 members did not improve the tails** ([Experiment
125](../../00125-differentiable-physics-n8-ens-members-through-gam-k-and-the-spread-around-them/result.md)).

**The cold-night bands of arm D are too narrow, and the stack's correction overshoots.** Arm D's
95% quantile covered 0.73 to 0.78 of cold nights ([Experiment
192](../../00192-dp-ens-members-temperature-cold-nights/result.md)). The stack's coverage was 0.965
and BG2's 0.781 ([Experiment
251](../../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md)).

**The disagreement between 20 independently trained correction networks did not predict the error
(correlation 0.00), but the result is not conclusive.** The experiment's own check failed: the
spread fell 8% under a shifted input ([Experiment
244](../../00244-dp-deep-ensemble-heads-cold-night-bands/result.md)).

## Where the stack is weaker than BG2

**From April to June 2026 the stack is worse than BG2, and the gap is localised but not explained.**
The gap is a solar-term gap at D5 and D9 on daylight half-hours, plus a night-time gap at D10
([Experiment 266](../../00266-summer-gap-diagnostics-solar-term/result.md)). The correction network,
the solar summary, the filter, and the ensemble sunshine forecast do not cause it. At D5
the excess sits on clear midday half-hours (+1.67 points), and at D9 it spreads across the day
([Experiment 299](../../00299-summer-gap-timing-night-level-and-error-correction/result.md)).

**Two tests of what BG2 knows about sunshine were refuted.** Banning sunshine interactions in
XGBoost removed 2.7% of BG2's advantage ([Experiment
305](../../00305-summer-gap-what-bg2-knows-that-arm-d-does-not/result.md)). Adding solar-by-heating
and solar-by-weekend columns closed 0.4% ([Experiment
306](../../00306-summer-gap-solar-interactions-with-temperature-and-day-type/result.md)).

**Past weather explains about a third of the excess loss on the unlabelled worst days.** The worst
5% of days carry about 42% of the excess loss, and load-shift events are the largest labelled group
([Experiment 172](../../00172-dp-error-budget-worst-days/result.md), `inconclusive`). Of the 72
worst days, 37 carry no label. Past weather removes 31.5% of their excess loss ([Experiment
174](../../00174-dp-unlabelled-worst-days-weather-shared-or-single-site/result.md), `partly
supported`). The loss could come from events at one substation. None of six other data sources
explains the days ([Experiment
177](../../00177-dp-worst-days-other-data-sources/result.md)).

## Open questions

- **The cause of the stack's April to June gap at D5, D9, and D10 is unknown.**
- **The cause of the unlabelled worst days is unknown.**
- **Whether the ranking against BG2 holds in other years and at all 16 Primaries is unknown.** Only
  July 2025 to June 2026 holds BG2 forecasts, at six substations.
- **Whether a special-day term is needed at Christmas is unknown.** Christmas error was 0.97 to
  1.22 times a weekday on fits with at least 180 training days (a post hoc subset), but the
  pre-registered ratios reached 1.91 and the test failed its positive control ([Experiment
  173](../../00173-dp-christmas-more-years-and-sites/result.md), [Experiment
  228](../../00228-dp-bg-rolling-origin-stack-against-bg2/result.md)).
- **Which demand line the production port should follow is unresolved.** The [research default
  log](../../research-default-log.md) names the stack as the first Primary demand research default.
  The final-models page documents the neural-network Candidate (the Primary demand forecast on that
  page), with no Kalman filter or convex model. No experiment of this theme compares the two.

## Options for next experiments

Each option comes from the [research plan](plan.md) or from the "What to try next" section of the
named experiment.

1. **An option is to test the D9 solar shape: an export clip, a morning-and-afternoon solar shape,
   and a month-varying night term at D10** (Experiments 266 and 299). Question: does a change to the
   solar term close the spring gap to BG2? A D5 clip adds nothing once the correction network is in
   ([Experiment
   251](../../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md)).
2. **An option is to run the margin arms at weekly cadence** (plan page). Question: does the convex
   forecast help lagged XGBoost when XGBoost refits weekly, as production does? Starting XGBoost
   from
   the convex forecast changed nothing in Experiment 111 (-0.023, -0.217 to +0.183). Dependency: GPU
   budget.
3. **An option is to fetch the 2021 to 2024 ensemble history from Source Cooperative** (plan page).
   Question: do three more years give XGBoost a fair training history? Dependency: the maintainer's
   approval and an AWS virtual machine.
4. **An option is to mask load-shift readings from the filter** (Experiment 205). Question: does
   masking avoid the 14-point loss that unmasked readings caused on one date?
5. **An option is to ablate the correction network's inputs one at a time, or to fit a
   gradient-boosted model with pinball loss** (Experiment 209's next steps, `not supported`: era
   splits changed the score by under 0.003). Dependency: none.

## Parked

**Only tail calibration and the coherence pass (CAL) are postponed.** Four further methods in the
plan page's queue have their own dependencies:

- **N7** (paired residuals at load-shift events) waits for nothing and needs minutes of CPU.
- **Schaake shuffle against ensemble copula coupling** (two ways to restore dependence between
  quantile forecasts) is a coherence test and waits for the Experiment 125 member forecasts.
- **Tail quantile regression on filtered residuals** is independent of CAL and needs stored
  residuals.
- **Conditional Pareto tails** (a heavy-tailed fit to residual extremes) wait for tail quantile
  regression.

Load-shift event detection and unmetered PV capacity work are paused until January and are not
options on this page.

The [experiments page](experiments.md) lists every experiment of the theme.
