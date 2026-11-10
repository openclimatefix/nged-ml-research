# Differentiable physics, embedded generation: state and next steps

**Only a clamp at zero on the quantiles of the eight MVA Primaries reached the final models, and the
evidence does not call for adding more.** Embedded generation is generation connected behind a
Primary substation. A Primary substation steps voltage down from 33 kV to 11 kV
([glossary](../../glossary.md#primary-substation)). This series tested four kinds of embedded
generation: unmetered rooftop solar, a solar export limit, a gas engine, and the loss of the sign of
power at meters that report apparent power (MVA, which has no sign). The tests found one clear gain,
a ceiling on the solar output at one substation. They also found that the solar capacity read from
the data cannot be validated. The [final models](../../final-models.md) have no explicit term for
unmetered solar, no export clip, and no gas-engine term.

D5 to D22 are anonymised labels for Primaries. BG2 is the best-guess forecast
([glossary](../../glossary.md#bg2)). The metric is the quantile score
([glossary](../../glossary.md#quantile-score-and-pinball-loss)), in points of capacity (percent of a
Primary's reference power) unless a unit is given, and a negative change is an improvement. Verdict
words such as `inconclusive`, `partly supported`, and `not supported` are defined in the
[glossary](../../glossary.md#pre-registration-decision-rule-and-verdict-words). PV means
photovoltaic (solar) generation. GAM means generalised additive model. The stack is the demand
forecast built from a convex GAM, a Kalman filter, and correction stages. This page is a selection;
the [experiments page](experiments.md) lists every experiment.

## What works

**Capping the solar output that the forecast subtracts at 60% of the held capacity helps at D5 and
nowhere else.** The held capacity is the fixed solar capacity (the pin) that the demand model
subtracts, and the cap (the clip) mimics an export limit. [Experiment
180](../../00180-dp-sunny-daylight-fix-d5-solar-shape/result.md) cut D5's loss to BG2 on sunny
half-hours from +0.202 MW to +0.069 MW. [Experiment
214](../../00214-dp-pv-clip-ceiling-as-of-and-screen/result.md) showed that 97% of the gain remains
when the cap is chosen from earlier data only (0.129 MW of 0.133 MW). Among the other five Primaries
tested, none shows a clear gain, and D9 cannot be told apart from no gain (0.03 MW better, interval
-0.076 to +0.007 MW).

**A ceiling learned from the substation's own recent data closes most of the clip's gain.**
[Experiment 247](../../00247-dp-learned-export-cap-for-the-solar-clip/result.md) found that the
learned ceiling closes 96% (interval 62% to 104%) of the hand-set clip's gain at D5. One year of
data cannot show that it reaches the pre-registered 80%, and at D9 the share is 93% with an interval
of 70% to 189%.

## What did not work

**Correction layers on top of the clip add nothing reliable.** [Experiment
232](../../00232-dp-pv-clip-with-correction-heads/result.md) found that the stored correction heads
(small neural networks that shift the demand forecast) cancel most of the clip's gain.

**Fitting through the apparent-power meter at Stickney (D12) scored better on one substation, and
the result is `inconclusive`.** [Experiment
152](../../00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries/result.md) folded the forecast
quantiles at zero, because reverse flow (power flowing from the substation back into the grid) is
hidden when the meter reports apparent power. The fold scored 0.40 points better on D12's sunny
middays and did no harm at the other seven apparent-power Primaries. The verdict is `inconclusive`
because the whole-year interval includes zero and the known-answer test (injecting a known signal
and checking that the fit recovers it) failed at two injection settings. [Experiment
372](../../00372-mva-fold-rule-chosen-fit-at-all-primaries/result.md) was `not supported`: its D12
gain of 0.57 points on sunny half-hours cannot be told from zero.

**Solar capacity read from each Primary's own power cannot yet be shown to be a calibrated
measurement.** [Experiment
129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md) gave 3.3 to 4.5 MW
of AC capacity per Primary at five Primaries, with a spread over test scenarios and not a calibrated
80% interval. [Experiment 241](../../00241-pv-capacity-calibrated-interval/result.md) got within 10%
of injected capacity at 5 of 5 Primaries, but its intervals cover only 59% to 79% of injected farms,
and 0% to 2% when demand follows cloud. [Experiment
477](../../00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits/result.md) found that
a neural-network PV module reads a placebo (the farm's daylight profile with a shuffled sky) as
strongly as the real farm, so its placebo gate failed.

**Reading the capacity from the data did not improve the forecast.** [Experiment
139](../../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) was `refuted`: a
capacity read from the Primary's own power made sunny half-hours 0.11 points worse. [Experiment
351](../../00351-primary-demand-capacity-physical-shape-readout/result.md) was `not supported`: a
capacity from the clear-sky output curve made them 0.065 points worse (interval +0.017 to +0.126).
[Experiment 349](../../00349-primary-demand-default-3-exp-269-capacity-in-the-stack/result.md)
scored 0.047 points worse with the estimate of Experiment 269 (interval -0.029 to +0.135).

**A capacity above the pin hurt, and a capacity below it did not.** [Experiment
352](../../00352-primary-demand-default-3-capacity-response-surface/result.md) found that a capacity
20% too low moved the sunny-half-hour score by 0.010 points, while 20% too high made it 0.124 points
worse. The curve is flat from 0.8 to 1.0 times the pin. The experiment advises against refining the
capacity inside that range.

**An explicit solar term did not beat letting the neural network read the sunshine forecast.**
[Experiment 409](../../00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/result.md) found a
change of about -0.06 points of capacity, with an interval including zero. [Experiment
416](../../00416-mdn-explicit-solar-term-better-powered-follow-up/result.md) found the two within
0.03 points of continuous ranked probability score (CRPS, [glossary](../../glossary.md#crps)) over
12 months. The explicit term was nominally 0.23 points worse on sunny middays.

**None of the gas-engine terms tried beat plain band widening at one Primary.** [Experiment
264](../../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md) found that widening the bands from
past residuals lowers the quantile score by 9% to 19%. An engine mixture did no better than plain
widening, because the mixture also fits a running engine at two Primaries that have none.
[Experiment 294](../../00294-regime-aware-model-at-d8-engine-state-term/result.md) found that an
engine term using the last settled day's state made the score 10.3% worse. The engine has not been
modelled from the readings, and NGED's own record of the engine's output would be the data to test
an engine model.

## What is known about the apparent-power meters

**The rule plus the stage-2 corrections trails BG2 by 0.53 points at the eight MVA Primaries.**
[Experiment 374](../../00374-mva-fold-stage-two-at-eight-primaries/result.md) found that the rule
with the stage-2 corrections (the stack's last correction stage) scores no worse than the stack's
arm A1 at those eight Primaries.

**A dip towards zero at midday is a screen and not a sign label.** [Experiment
506](../../00506-mva-sign-evidence-dip-depth-and-gated-solar-rule/result.md) found that at power
factor 0.97, the threshold that finds 89% of injected sign changes also flags 34% of days with none.

**A day labeller finds only 5 of 59 real reverse-flow days at the six MW Primaries.** [Experiment
674](../../00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/result.md) was `inconclusive`.

**Reverse flow at D12 is possible, and no experiment has proved it.** Experiment 152 fitted reverse
flow on 100% of D12's sunny middays. Experiment 506 audited that: D12's midday median is 81.6 points
against 15.6 in dark hours, so its midday reading never nears zero. A crossing, if any, would be in
the shoulders of the day. [Experiment
370](../../00370-mva-fold-pv-capacity-and-eight-primaries/result.md) found reverse flow at D12 and
at none of the other seven. [Experiment
679](../../00679-pv-mva-conversion-loss-reactive-floor-at-half-level/result.md) found that if the
meters follow NGED's power factor of 1 plus a small reactive floor, the conversion loss is about 0.5
points (80% interval -1 to 8), so the 0.97 loss was probably an overestimate.

## Open questions

**Whether D8, D12, and D22 ever export is unknown.** NGED's signed power at D12 and D8 would settle
the question without a fit, as Experiment 674 notes.

**Whether the gas engine at one Primary can be known before the half-hour it affects is unknown.**
Experiment 294 names a forecast of the engine's state and NGED's own record of running hours.

**Whether any capacity estimate beats the pin is unknown.** The experiments above tested the convex
stage out to 7 days only, and 7 to 10 days is untested.

## Next-experiment options

**An option is to stop here for production.** This is the page's own judgement. The final models
need none of these parts, and the series has found no gain that a final model could use.

**An option is to run the sign test on real signed days.** The source is Experiment 674's "What to
try next": calibrate the midday dip on real days where they exist, and ask NGED for a few weeks of
signed power at D12 and D8. The dependency is data from NGED.

**An option is to test the capacity estimate with a cap.** The source is Experiments 349 and 352's
"What to try next": cap any estimate at the pin, and test a per-Primary export clip at D5 and D9.

**An option is to build typed generator nodes once a flat node stays off where no generator
exists.** The source is the [research plan](plan.md). A flat node is one generator term shared by
all substations. [Experiment
110](../../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md) found that the
flat node read 12 to 35 points at all five decision Primaries (the five Primaries on which the
pre-registered tests are scored).

## Parked

**Orientation mix for rooftop solar is dropped.** The research plan records that Experiments 73 and
77 found the mix unidentified or immaterial, and that Experiment 114 dropped it.

**Unmetered solar capacity and load-shift event detection are paused until January.** The
[load-shift detector state page](../lsc-detectors/state-at-january-pause.md) describes the pause.
