# Lagged power and error correction: state and next steps

**Recent power readings (lags) improve the demand forecast at short and medium leads, and the final
demand networks read them.** The Primary substation demand network (the Candidate, a neural network
described in [final models](../../final-models.md)) reads the day-ago and week-ago readings, each at
least 24 hours old, plus neighbour and gap signals. Its prevailing head, which forecasts the
arrangement now in force, reads 14 extra lagged inputs at least 6 hours old
([Experiment 681](../../00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md)). The
bulk and grid supply point network reads lags at least 3.5 hours old. The wind and solar networks
read no power lag ([wind and solar page](../generator-forecasting/state-and-next-steps.md)).

**Scores are CRPS in points of capacity unless stated, and lower is better.** CRPS is the continuous
ranked probability score, and one point of capacity is 1% of a series' capacity; the
[glossary](../../glossary.md) defines both, and also MVA, MW, N1, BG2, and the frozen XGBoost model.
A contrast is written as the first forecast minus the second, so a negative contrast favours the
first. "Worse by 0.36" always means the score rose by 0.36. The early experiments used 16 Primaries,
as stated for each result. The experiments on load-shift events (transfers of demand between
neighbouring Primaries) are in the [load-shift summary](load-shift-summary.md). Load-shift detection
is paused until January, so the options below concern the lagged inputs themselves.

## What works

**Day-ago and week-ago power closes most of the gap between a forecast that sees recent power and
one that does not.** At 1 to 5 days ahead for the eight MVA Primaries, the network without recent
power scored 0.2 to 0.3 worse than the network with it. Adding only the day-ago and week-ago
readings closed about four fifths of that gap, a point estimate with wide intervals ([Experiment
540](../../00540-lag-gap-what-closes-the-gap-to-lag-enabled-models/result.md)). A lag encoder (the
part of the neural network that reads lagged power) using readings at least 24 hours old then beat
the network without lags by 0.07 to 0.15 at 1 to 5 days, as point estimates against the lag-free
reference M0 ([Experiment
560](../../00560-modular-lag-encoder-capped-lags-and-block-dropout/result.md)).

**The slow output offset does not replace lags at the eight MVA Primaries.** The offset is a slowly
changing level shift added to the network's output. The lag network with the offset beat the offset
alone by 0.224 (interval +0.006 to +0.394) at the eight, and by 0.049 at the six MW Primaries
against the lag-free network with the offset ([Experiment
572](../../00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md)). That experiment
also tested 3.5-hour lags at the Primaries. They beat the lag-free network by 0.257 at the eight and
0.159 at the six at 0 to 24 hours, but beat 24-hour lags by only 0.070 and 0.055. These are
descriptive contrasts.

**One lag-light network for all 14 Primaries is a defensible choice.** A lag-light network reads
only the day-ago and week-ago lags. It scored 0.081 better than the lag-free recipe at the six MW
Primaries over 1 to 5 days, so choosing a recipe per group gave up accuracy ([Experiment
557](../../00557-per-group-lag-recipe-choice-stored-forecasts/result.md)).

**No lag input is newer than the forecast origin.** A timestamp audit of 104 million lag checks
found 0 violations ([Experiment
551](../../00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal/result.md)).

**Extra lagged inputs make the prevailing output react to a load shift sooner.** Fourteen extra
lagged inputs made the prevailing output cover half of a new load level about 20 hours sooner (95%
interval 16.3 to 24.8) over 1,152 simulated shifts, and 120 hours sooner at Primary D5 in one
verified event ([Experiment
681](../../00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md)). The final
models include them.

**In the early two-stage XGBoost design, recent power in the second stage beat recent power in the
first.** For 16 Primaries up to 48 hours ahead, the stage 2 calibrator with lags scored 5.31, stage
1b with a lag-free calibrator 6.14, and a calibrator without lags 7.11 ([Experiment
5](../../00005-lag-placement-stage1b-vs-stage2-demand/result.md)).

**Removing the three recent-power inputs from BG2 made Primary scores worse by 0.36.** The figure
holds once one substation's commissioning months were cut ([Experiment
102](../../00102-simp1-simplifying-the-bg2-best-guess-before-production/result.md)).

## What does not work

**The 14-day correction lowered the Primary score by about 1 point in the early design, and no
variant beat the plain additive form.** The correction adds the mean error of the last 14 days (see
the [glossary](../../glossary.md)). It scored -0.99 (interval -1.82 to -0.18) with power 2 days old
([Experiment 21](../../00021-error-correction-variants-bsp-gsp-demand/result.md)).

**The 14-day correction made the score worse at the supply points and for solar.** At 4 bulk and
grid supply point series it made the score 0.36 worse (interval includes zero) ([Experiment
21](../../00021-error-correction-variants-bsp-gsp-demand/result.md)) and 0.305 worse in BG2 (interval
0.060 to 0.594), so BG2 serves the supply points without it ([Experiment
87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)). On 6 solar farms and 3
wind farms it made the CRPS worse by 0.437 for solar (interval +0.215 to +0.664) and by 0.413 for
wind (interval -0.003 to +0.924, which includes zero) ([Experiment
23](../../00023-lagged-power-and-error-correction-solar-wind/result.md)).

**Extra lag columns, summaries, and year-ago lags added no resolvable gain.** None of five added
lag column sets lowered the XGBoost CRPS for days 1 to 10 ([Experiment
388](../../00388-xgboost-lagged-power-column-sets-weekly-demand/result.md)). A year-ago lag changed CRPS
by +0.019 at the eight MVA Primaries over 24 to 120 hours (interval includes zero), although it
lowered the Christmas and New Year error at the six MW Primaries by 0.53 at 120 to 240 hours, in
one December ([Experiment
585](../../00585-lag-count-and-year-ago-lags-through-the-lag-encoder/result.md)).

**Random lag dropout did not widen the band where a lag is missing.** The q05-to-q95 band covered
0.799 of masked half-hours for the plain model and 0.802 with dropout at 10% of training rows
(0.836 at 30%), against 0.90 nominal. On synthetic transfers, masking lags without dropout made the
score 1.36 points worse than the plain model, and 30% dropout brought that to -0.02 (-0.35 to
+0.38). Dropout also made the score worse by up to 0.046 on unmasked rows, above the 0.02 limit. The
verdict was inconclusive ([Experiment
100](../../00100-drop1-random-lag-dropout-widens-intervals/result.md)).

**Detector features added to extra lags did not make the prevailing output react sooner.** The head
with both reacted 1.75 days sooner than the head of experiment 652, but only 0.16 days sooner than
extra lags alone (interval -0.45 to +0.14) ([Experiment
691](../../00691-causal-detector-features-plus-extra-lags-in-the-prevailing-head/result.md); the
features alone are in [Experiment
687](../../00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md)). Both were
not supported.

## Where the pages disagree

**The long-lead advantage that [Experiment
521](../../00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/result.md) reported for K2-S1
came from a look-ahead in its weather.** K2-S1 is a mixture of two lagged heads and two lag-free
heads, and 521 placed it ahead of the daily lag-free network at 5 to 10 days. Once the lag-free
heads read only weather known at the forecast origin, K2-S1 scored 0.43 (eight) and 0.41 (six) worse
than its own earlier score at 120 to 240 hours. Against the daily lag-free network, K2-S1 is worse
by point estimate in every lead band, but the intervals include zero except for the eight at 12 to
24 hours ([Experiment
541](../../00541-k2-s1-rebuilt-on-origin-time-weather/result.md)). The lag gain is therefore a
short-lead and medium-lead gain, 0.2 to 0.3 at 24 to 120 hours in 540.

## Open questions

**Three Primaries (D4, D12, and D19) carry 85% of the lag gain at the eight MVA Primaries, and the
reason is unknown.** The gain is as large at night as in sunny half-hours ([Experiment
576](../../00576-lag-gap-by-half-hour-type-eight-mva-and-six-mw-primaries/result.md)).

**Lags may be unsafe in a cold, holiday, or event week.** The lag-light network without dropout
scored 0.171 worse than the lag-free network in Christmas week at the eight MVA Primaries, past the
0.10 limit set in advance, on 9 target days in one winter. The variant with dropout stayed within
0.15 when its readings were removed ([Experiment
551](../../00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal/result.md)).

**A longer embargo lowers the load-shift leak of lagged networks, and the proposed limit is not yet
met.** The share of a shift that the median forecast carries fell from 0.160 to 0.114 (interval
0.073 to 0.154, which includes the proposed limit of 0.10) with a 7-day embargo plus half weight on
the next 7 days ([Experiment
599](../../00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption/result.md)).
The [load-shift summary](load-shift-summary.md) holds the rest.

## Ranked next experiments (options)

1. **An option is to find what D4, D12, and D19 share.** [Experiment
   576](../../00576-lag-gap-by-half-hour-type-eight-mva-and-six-mw-primaries/result.md) names it as
   the next step. The answer would show which Primaries need lags.
2. **An option is to run a lag-free head that also sees the lead time.** [Experiment
   541](../../00541-k2-s1-rebuilt-on-origin-time-weather/result.md) names it as the untested next
   step for the long-lead gap.
3. **An option is to judge the trailing temperature windows on cold winter nights with more winter
   folds.** [Experiment
   550](../../00550-lag-light-network-trailing-temperature-windows-with-placebo/result.md) found a
   gain of about 0.10 there, and names the extra folds as the next step.
4. **An option is to test 3.5-hour lags in the Candidate.** [Final models](../../final-models.md)
   lists the question as open.

## Parked

**Load-shift event detection and lag-trust methods are paused until January.** The 38 load-shift
experiments are in the [summary](load-shift-summary.md). The planned end-to-end two-output network
with an end detector (experiment 654 in the [research plan](plan.md)) is load-shift work and is
paused with them.

**The 14-day correction for supply points, solar, and wind is parked.** Of the experiments read for
this page, none shows a gain there, and the final networks use no such correction.
