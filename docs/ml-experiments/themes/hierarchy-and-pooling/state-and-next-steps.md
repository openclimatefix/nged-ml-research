# Hierarchy and pooling: state and next steps

**Pooling many series in one neural network lowered the error at the Primaries, but pooling in an
XGBoost model did not, and neighbour or hierarchy information has not yet improved any forecast.**
Terms such as CRPS (continuous ranked probability score), BSP (bulk supply point), GSP (grid supply
point), and Primary (primary substation) are defined in the [glossary](../../glossary.md). One
neural network across all 16 Primaries beat 16 separate neural networks by 0.185 points of capacity
at the six MW Primaries and 0.161 points at the eight MVA Primaries
([Experiment 403](../../00403-pooled-versus-per-primary-mixture-density-network/result.md)). The
embedding of the Primary identifier carries that gain, and the gain is larger for a Primary with
little history
([Experiment 406](../../00406-pooled-mixture-density-network-without-embedding-and-with-short-history/result.md)).
Pooling the 4 supply points with the Primaries beat a neural network fitted at the Primaries alone
by 4.1 and 15.2 points
([Experiment 415](../../00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points/result.md)).
For the production port, one lag-enabled neural network per group forecasts the 2 bulk supply points
and the 2 grid supply points, as [Final models](../../final-models.md) documents.

## What works

**One XGBoost model per substation type or per capacity band scores within the pre-registered margin
of one XGBoost model per series.** On the 16 Primaries, three or four pooled XGBoost models score
0.15 points of capacity worse than 20 per-series XGBoost models (5.47 against 5.32, 99.44% interval
+0.03 to +0.26, margin 0.30). The saved models shrink from 435 MB to 73 to 94 MB. The result comes
from one validation year and 20 series, so it says nothing about 2,500 substations
([Experiment 24](../../00024-pooling-groupings-demand-xgboost/result.md)).

**One pooled neural network across the 16 Primaries scores lower CRPS than 16 separate neural
networks.** The gain is 0.185 points at the six MW Primaries (95% interval -0.300 to -0.061) and
0.161 points at the eight MVA Primaries (-0.272 to -0.089), and 11 of the 14 Primaries with long
history gain (Experiment 403).

**The Primary embedding makes the pooling work, and pooling helps most where one Primary has little
history.** A pooled neural network without the embedding scores 0.57 points worse than separate
neural networks at the six MW Primaries, and the embedding turns that into a gain of 0.22. With one
Primary's history cut to 3 months, the pooled neural network is 1.15 points better at the six MW
Primaries (Experiment 406).

**One neural network per group forecasts the bulk and grid supply points better than a climatology
and than the weekly XGBoost forecast at the bulk supply points.** The pooled neural network scored
0.47 points of capacity lower than the weekly XGBoost forecast at the bulk supply points in all 4
test months, and tied it at the grid supply points with very wide bands (Experiment 415). The
lag-enabled neural network of the production port scores a CRPS of 4.27 at the bulk supply points
and 10.52 at the grid supply points at days 0 to 1, against 7.23 and 18.69 for climatology
([Final models](../../final-models.md#demand-at-the-bsps-and-gsps)).

**Giving the demand neural network the previous day's and week's power helps a little at the bulk
supply points.** When the lag-enabled and lag-free neural networks differ only in the lags, the lags
lower the error at days 1 to 3 by 0.11 points of capacity at the 2 bulk supply points, a gain that
passes the pre-registered bar. The gain fades to zero by about day 6. At the 2 grid supply points
the gain is 0.73 points but is below the minimum detectable effect, so it is not shown
([Experiment 622](../../00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/result.md)).
Experiment 622 supersedes Experiment 619, whose offsets were unmatched and whose gain was smaller
than 5 development months could resolve
([Experiment 619](../../00619-bsp-gsp-lag-enabled-network-for-the-bake-off/result.md)).

## What does not work

**The series identifier cannot be dropped from a pooled XGBoost model, and numeric fingerprints of a
substation add nothing to it.** Dropping the identifier raised the score of the type pools by 0.82
and of the all-series pool by 0.85 points (Experiment 24). Adding location or training-window
statistics changed the score by +0.00 (C1 and C2, both within plus or minus 0.05), and adding
recent-window statistics changed it by +0.02 (C4, 99.375% interval -0.05 to +0.10).

**A fingerprint did not carry a substation held out of the pool, except for recent-window
statistics.** A substation held out of the pool was forecast worse with training-window statistics
than with no fingerprint (+1.39, +0.99 to +1.91), from one seed and 8 held-out Primaries.
Recent-window statistics alone lowered the held-out score by 0.23 (-0.29 to -0.16) in a post hoc arm
added after the first run
([Experiment 32](../../00032-substation-fingerprints-pooled-demand/result.md)).

**One XGBoost model for all 20 series falls outside the margin.** The model scores 0.29 points worse
than per-series XGBoost models on the 16 Primaries (+0.19 to +0.42), and 0.80 points worse on the 2
bulk supply points alone (Experiment 24).

**Adding the 4 supply points to the neural network's training rows did not improve the 16
Primaries.** The change was smaller than the experiment could detect, and its sign differed between
the 8 MVA Primaries (+0.026) and the 6 MW Primaries (-0.054). A fair test of whether pooling helps
needs many more extra series than NGED's data holds
([Experiment 564](../../00564-pooled-primary-bsp-gsp-m2-reduced/result.md)).

**Forecasts of the series beneath a bulk supply point did not lower its score.** Summing the
Primary and generator forecasts beneath Boston and Skegness changed the score by +0.04 and +0.19
points, and the intervals include zero
([Experiment 36](../../00036-hierarchy-aware-bsp-gsp-forecasts/result.md)). Putting every child
Primary into the sum changed the score by +0.005 points at Skegness and -0.011 at Boston, with
intervals that include zero
([Experiment 204](../../00204-dp-bsp-full-bottom-up-with-signed-mva-children/result.md)).

**A scaled sum of the Primary forecasts scored worse than a neural network that sees the node's own
readings.** The sum scored 34% to 60% higher error than the neural network
([Experiment 426](../../00426-mdn-neighbouring-primaries-graph-and-bottom-up-design/result.md)).

**The recent errors of interconnected neighbours predicted under 2% of a Primary's own day-ahead
error.** Adding true neighbours' lag residuals lowered the held-out squared error by 0.01% against
own lags only (Experiment 426).

**Adding the bulk supply point reading to the differentiable-physics fit of the Primaries made the
forecast worse.** The clean-forecast error rose by 0.4 to 4.6 points at every weight tried
([Experiment 124](../../00124-differentiable-physics-h-bsp-closure-audit-and-bsp-hint/result.md)).

**The first-release recipe did not clearly beat the plain pooled neural network at the supply
points.** At the bulk supply points the difference was inside the noise (-0.14 and +0.09). At the
grid supply points the recipe was about 0.5 points better, close to the refit noise of about 0.4
([Experiment 475](../../00475-recipe-on-bulk-and-grid-supply-points-against-pooled-plain-network/result.md)).

**A 0/1 metering-type input did not help.** At the 8 MVA Primaries it made the score worse by 0.038
points, as a randomly permuted flag did. At the 6 MW Primaries it helped by 0.030 points, a gain too
small to port
([Experiment 525](../../00525-metering-type-flag-lagless-demand-network/result.md)).

## Open questions

**Pooling is unsettled at hundreds of series.** All pooled results use at most 20 series
(Experiments 24 and 564).

**A substation never seen by the pool is tested only for XGBoost models.** Experiment 32 used one
seed and 8 Primaries. The write-up suggests, without testing it, that 7 distinct fingerprints act as
7 identifiers.

**The largest observed miss at a grid supply point looks like a load shift.** The demand at Walpole
stepped up on 2026-10-19 while the median fell, and the neural network has no neighbour inputs and no
load-shift component ([Final models](../../final-models.md#demand-at-the-bsps-and-gsps)). Whether
neighbours help inside load-shift events is untested. Experiment 426 did not measure it, and
[Experiment 404](../../00404-mdn-lagless-and-calibration-module-trust-weights/result.md) has no
neighbour inputs.

**The 90% band at the supply points is too narrow.** The band covers 0.867 of readings at the bulk
supply points and 0.877 at the grid supply points, against 0.90
([Final models](../../final-models.md#demand-at-the-bsps-and-gsps)).

## Next experiments, as options

**For the production port, no experiment in this theme is a prerequisite.** The bulk-and-grid neural
network and its retraining recipe are specified in
[Final models](../../final-models.md#demand-at-the-bsps-and-gsps). The options below are ordered by
how directly they bear on the port. None is started. Options 2 to 6 come from the
[differentiable-physics pooling plan](../differentiable-physics-pooling-across-substations/plan.md).
The closure audit is Experiment 124's fit of how much of each bulk supply point's reading its
metered children and generators explain, with the rest as an unmetered remainder.

1. **An option is to compare the 3.5-hour lag cap with a longer cap on matched refit schedules.**
   The question is which cap scores better at the supply points. A fit costs about 2 GPU-hours for
   the one-year run
   ([Final models](../../final-models.md#demand-at-the-bsps-and-gsps)). Nothing blocks it, and Final
   models lists it as open.

2. **An option is to compare a top-down with a bottom-up forecast at the bulk and grid supply
   points.** The question is whether a direct forecast beats the sum of the Primary forecasts plus
   generators, and whether reconciling the two helps. Experiment 204 found the bottom-up forecast
   with every child Primary matched the top-down forecast (+0.1% and -0.15%, intervals include
   zero). Experiment 386 found the generalised additive model stack 1.0 points (bulk supply points)
   and 2.0 points (grid supply points) worse than the weekly XGBoost forecast
   ([Experiment 386](../../00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost/result.md)). The
   topology would enter as a soft hint, because the unmetered remainder makes an exact sum wrong.
3. **An option is to use the bulk supply point reading as an observation in the Kalman filters of
   the Primaries.** The question is whether the closure relation, with its unmetered remainder as
   state noise, helps the Primaries' filters. The experiment runs on the CPU in about 2 hours and
   depends on the closure audit of Experiment 124.
4. **An option is to treat metered generators as export-positive nodes at the bulk supply point.**
   The question is whether the closure relation then holds better at the 2 bulk supply points.
   Experiment 223 found that the remainders move with the generators but not one-for-one, and it
   recommends separating unmetered solar and battery dispatch first
   ([Experiment 223](../../00223-dp-bsp-generators-as-children-and-double-count-test/result.md)).
   Any figure would show the generators anonymised.
5. **An option is to test unmetered wind at the bulk and grid supply points.** The question is
   whether the closure residual follows the weather forecast's wind speed. The test depends on the
   closure audit and on separating the wind part of the residual from the solar part.
6. **An option is a pooled differentiable-physics model on every v1 series, including MVA
   substations.** The question is whether full-history skill and capacity identification improve.
   The experiment needs 3 to 5 GPU hours. The design differs from the pooled neural network.

## Parked

**Neighbour graph neural networks are parked, because the signal they would pass along is
near zero.** Experiment 426 found no neighbour-error signal, so no graph neural network was fitted.

**Pooling for onboarding substations with little history is parked.** The pooled model forecast a
substation with 4 weeks of history far better than a per-site copy (4.04 against 14.38), and worse
at 2 full years (3.42 against 3.04)
([Experiment 105](../../00105-differentiable-physics-3-pooled-and-per-site-demand-model/result.md)).

**Load-shift event detection is paused until January.** The pause is described on the
[load-shift detectors state page](../lsc-detectors/state-at-january-pause.md). Any neighbour test
inside load-shift events waits with it.

**Wider fingerprint designs are parked.** Experiment 32 found no gain for series seen in training,
and recent-window statistics alone helped a held-out substation by 0.23 in a post hoc test.
