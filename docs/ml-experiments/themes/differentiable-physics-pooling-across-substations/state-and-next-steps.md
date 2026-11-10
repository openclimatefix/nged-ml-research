# Differentiable physics, pooling across substations: state and next steps

**Fitting one differentiable-physics demand model across several Primary substations helped a
substation with 4 weeks to 3 months of history. At full history (2 years) pooling lost a little, and
no gain in this theme passed a known-answer test.** The differentiable-physics pooling work is not in
the [final models](../../final-models.md): the Primaries' Candidate there is a neural network pooled
across all 16 Primaries
([Experiment 403](../../00403-pooled-versus-per-primary-mixture-density-network/result.md)). The
[glossary](../../glossary.md) defines Primary (primary substation), BSP (bulk supply point), GSP (grid
supply point), and MVA (megavolt-ampere, apparent power). The theme is unfinished.

Scores are quantile scores, in points of capacity (1% of a substation's capacity), and a lower score
is better. A contrast is the score of one arm minus the score of another, so a negative contrast
favours the first arm. The verdict words (`supported`, `partly supported`, `inconclusive`, and the
rest) are defined in the
[glossary](../../glossary.md#pre-registration-decision-rule-and-verdict-words). The decision
substations are the five Primaries D5, D10, D11, D14, and D16, on which the pre-registered tests were
scored. A partly pooled fit shares curves across substations but lets each substation depart from them
(shrinkage). Onboarding means forecasting a substation that has just joined. A known-answer test fits
a model to synthetic data with a known truth.

## What works

**A pooled neural-network demand model gives a new substation a usable forecast from 4 weeks of
readings.** The model was fitted across 15 Primaries. On the five decision substations it scored 4.18
after onboarding on 4 weeks, against 15.13 for the same model fitted to those 4 weeks alone (-10.95,
98.33% interval -15.12 to -7.28). With 3 months it scored 4.38 against 5.47 (-1.09, -2.01 to -0.34).
On six held-out Primaries including D9 (21 months of history), the 4-week scores were 4.04 and 14.38.
The verdict is `inconclusive` because a known-answer check failed. The per-site neural comparator was
weak: pooled 4-week onboarding only ties a climatology fitted on the full history (+0.12, -0.33 to
+0.57) and is worse than the manual heuristic (+0.70, +0.23 to +1.19).
([Experiment 105](../../00105-differentiable-physics-3-pooled-and-per-site-demand-model/result.md)).

**The pooled convex model N keeps the onboarding gain and loses 0.03 points at full history.** On the
five MW Primaries, N scored 4.35 against 7.60 for a per-site convex model on 4 weeks (-3.25, 99%
interval -5.20 to -1.59, better at 5 of 5). With 2 years of
history N was 0.03 points worse (+0.012 to +0.056, inside the 0.10 margin). Experiment 105's neural
model lost 0.385 points there, which suggests the neural fit caused most of that loss. The verdict is
`inconclusive`, because the known-answer test failed its pre-registered definition
([Experiment 109](../../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md)).

**On synthetic substations with a known answer, sharing the heating curve beat fitting each
substation alone.** On 4 winter weeks, the pooled fit's heating error was below no pooling and
complete pooling on 5 of 5 seeds, and close to a fair oracle (a reference estimator given the true
form of the curve). The sign of a substation's own heating offset was right at 6 or more of 8
substations on only 2 of 5 seeds,. The verdict is `partly supported`
([Experiment 154](../../00154-differentiable-physics-n0-gate-2-heating-error-against-truth/result.md)).
A June-weeks rerun was `inconclusive`
([Experiment 150](../../00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds/result.md)).

**Pooling the shared curves in the generalised additive model (GAM) forecast engine helped a
substation with 3 months of history.** The forecast is GAM-K, a GAM with a Kalman filter on recent
errors. On six Primaries, a substation with 3 months of history scored 3.01 instead of 3.64 (-0.629,
99% interval -1.639 to -0.091), against 2.84 for the same substation with 24 months of its own
history. The gain held at all five decision substations and in all 12 months of the scored
year. The verdict is
`partly supported`
([Experiment 158](../../00158-dp-partly-pooled-gam-shapes-heating-and-calendar-curves/result.md)).

## What does not work

**Pooling loses a little at 2 years of history, and no gain here passed a known-answer test.** At five
Primaries with 2 years of history, the pooled neural network scored 3.42 against 3.04 for per-site
fits (+0.385, 98.33% interval +0.178 to +0.689, worse at 5 of 5). The pooled convex model N lost 0.03
points. At 24 months, complete pooling of the GAM lost 0.29, and partly pooling it changed the score
by +0.004 (-0.006 to +0.014) (Experiment 158).

**Fitting the pooled neural network exactly made the forecast worse.** The exact fit (a variable
projection, which solves the linear weights in closed form) reached a training objective 16,000 to
27,000 nats (units of log-likelihood) lower, yet scored 5.49 against 3.27 for the gradient fit. It
forecast 20 to 280 points too high on four days of spring 2026, because the exact solve made the
last layer of the interaction neural network (which lets heating and calendar effects interact) very
large. With that network off, the exact fit matched the gradient fit within 0.03 points
([Experiment 112](../../00112-differentiable-physics-vp-variable-projection-fit-of-pooled-model/result.md)).

**Adding the BSP reading to the Primaries' fit made the clean forecast worse.** At every weight above
zero it worsened clean-forecast error by 0.385 to 4.6 points, against a limit of 0.023. An exactly
specified synthetic system lost accuracy in the same way, so the cause is untested.

**The BSP reading was a weak monitor of metering errors.** The closure audit checks whether a BSP's
reading equals the sum of its Primaries' readings plus an unmetered remainder (the closure relation).
A fit trained on earlier rows found a 25% scale error lasting 30 days in 50% of
injections and false-alarmed on 61% of later windows. The verdict is `not supported`
([Experiment 124](../../00124-differentiable-physics-h-bsp-closure-audit-and-bsp-hint/result.md)).

**Pooling only the heating curve did not help a substation with 3 months of history.** It scored 0.507
points worse than no pooling, with an interval that includes zero (-0.073 to +1.024). The arms also
differed in ridge (a penalty on large coefficients), so the test cannot say which shared curves carry
the gain of Experiment 158.

## Open questions

**Whether the convex pooled model N passes a known-answer test is unknown.** The gate of Experiment
109 failed as pre-registered, and Experiment 150 was `inconclusive`. Every onboarding gain above is
a screening result.

## Next-experiment options

**No option in this theme is a prerequisite for the production port.** The options are in the
[research plan's](plan.md) priority order, not the
[hierarchy and pooling page's](../hierarchy-and-pooling/state-and-next-steps.md), which lists options
5 to 7 in its own order.

1. **An option is a drift term, a slowly changing demand level, trained from 2019-09 (plan Part 4).**
   The question is whether it beats the same model on its best shorter window. The
   [forecast-engine page](../differentiable-physics-forecast-engine/state-and-next-steps.md) cites
   [Experiment 170](../../00170-dp-bg-stage3-drift-and-holiday-scoring/result.md) finding drift
   worsened the score by 0.03 once the Kalman filter was in. Experiment 105's arm that used all years
   of history plus a drift term scored 2.784 against 3.072 on five substations.
2. **An option is the MVA observation layer (plan Part 5).** The question is whether a model of how a
   meter that reads apparent power relates to real power helps MW sites and recovers MW at MVA sites.
   It waits for a better capacity fit.
3. **An option is a scale-up of the pooled model to every v1 series it can take (plan N9).** It
   waits for nothing and needs 3 to 5 GPU hours.
4. **An option is to treat the BSP reading as an extra observation in the Kalman filter of each
   Primary (plan).** The question is whether the closure relation, with its unmetered remainder as
   state noise, helps the Primaries' forecasts. Experiment 124 also proposes
   using the residual as a separate monitor outside the fit.
5. **An option is to compare a direct forecast at a BSP or GSP with the sum of the Primary
   forecasts, and to reconcile the two (plan).** It depends on the closure audit.
   The hierarchy page holds the evidence so far
   ([Experiment 204](../../00204-dp-bsp-full-bottom-up-with-signed-mva-children/result.md),
   [Experiment 386](../../00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost/result.md), and
   [Experiment 426](../../00426-mdn-neighbouring-primaries-graph-and-bottom-up-design/result.md)).
6. **An option is to enter metered generators as export-positive nodes at the BSP (plan).**
   The hierarchy page cites
   [Experiment 223](../../00223-dp-bsp-generators-as-children-and-double-count-test/result.md), which
   puts separate unmetered solar and battery dispatch first.
7. **An option is to estimate unmetered wind at the BSP and GSP from the residual (plan).** It waits
   for the closure audit.
8. **An option is to give the heating curve and the calendar families separate penalties in the
   partly pooled GAM (Experiment 158's next step).**
9. **An option is to rerun the pooled exact fit with stronger penalties, or to compare the pooled
   model without its interaction neural network (Experiment 112's closing section).**

## Parked

**Onboarding a substation with little history is parked.** The
[hierarchy and pooling page](../hierarchy-and-pooling/state-and-next-steps.md) parks it, and the
maintainer's priority is the production port. Parked with it are the pooled-for-new, per-site-for-mature
design, and
pooling the heating columns of
[Experiment 114's](../../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md)
model split four ways by day type, and the real-substation test of the 3-month gain.

**Pooling for a substation with 2 years of history, adding the BSP reading to the Primaries' loss, and
learning a substation's own heating offset from 4 weeks are parked.** "What does not work" gives the
results.

**Load-shift event detection and unmetered PV capacity are paused until January.** Any pooled
experiment that needs either waits with them.
