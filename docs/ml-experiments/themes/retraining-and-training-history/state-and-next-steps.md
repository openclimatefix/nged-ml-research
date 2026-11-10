# Retraining and training history: state and next steps

**For the Primary demand forecast, refitting more often than monthly gains a small, real amount, and
the numbers say daily for the lag-free neural network and weekly for the XGBoost model.** Neither
gain is large. Daily refitting of the lag-free neural network scored 0.065 points of CRPS better
than weekly refitting, about 1.3%. Weekly refitting of the XGBoost model scored 0.20 to 0.24 points
of capacity below the stack on two sets of runs, but three substations carry much of that gain.
[Final models](../../final-models.md) describes what the first release does, which is a monthly
refit followed by a fine-tune. The daily and weekly advice on this page is a different cadence.

**Terms.** [Refit](../../glossary.md#refit-and-rolling-refit), [stack](../../glossary.md#stack),
[N1](../../glossary.md#n1), the [14-day correction](../../glossary.md#14-day-correction), [D
codes](../../glossary.md#d-codes), [quantile score](../../glossary.md#quantile-score-and-pinball-loss),
[CRPS](../../glossary.md#crps) (continuous ranked probability score), and [points of
capacity](../../glossary.md#capacity-and-points-of-capacity) are defined in the glossary. The eight
MVA Primaries are the eight Primary substations measured in MVA only, and the six MW Primaries are
the six Primary substations measured in signed MW. K1 and K2-S1 are lagged demand neural networks,
which take recent power as inputs. A negative contrast is an improvement.

## What works

**Refit the lag-free demand neural network every day rather than every week.** The daily refit
scored 4.854 CRPS points at the eight and 3.493 at the six, against 4.920 and 3.555 for the weekly
refit. The gains were 0.065 and 0.061 points, with Bonferroni intervals that exclude zero, and
twice-weekly refitting gained 0.036 and 0.041 ([experiment
511](../../00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining/result.md)). The
neural network gets about 0.02 points worse for each day since its refit, so most of the gain is
freshness. A daily fine-tune takes about 3 seconds. Weekly refitting had already beaten monthly
refitting by 0.17 to 0.19 points ([experiment
510](../../00510-lagless-demand-network-weekly-finetune-and-recency-weights/result.md)).

**Refitting the XGBoost model for Primaries weekly, and dropping the 14-day correction from it, scored
better than the stack.** Weekly refitting scored 0.20 below the stack in [experiment
48](../../00048-weekly-retraining-confirmation-and-extension-demand-generators/result.md) (99.44%
interval -0.40 to -0.035, 292 daily runs at 16 Primaries) and 0.240 below it in [experiment
81](../../00081-r11-control-member-second-validation-year-demand/result.md) (99.17% interval -0.360
to -0.145, 274 control-member runs at 13 Primaries). Monthly refitting already scored 0.20 below the
stack in [experiment 34](../../00034-retraining-cadence-and-cheap-proxies-demand/result.md), and the
case for weekly over monthly is only 0.04 to 0.13 points. Experiment 48 (292 runs) covers N1 frozen
at its fit date, and experiment 81 covers its own 274 runs, so both results stand.

**Pretraining the neural network on ERA5 and CAMS weather, then fine-tuning on ensemble weather,
lowered the error.** ERA5 is the ECMWF reanalysis, CAMS is the Copernicus Atmosphere Monitoring
Service, and ENS is the ECMWF ensemble forecast. The gain was about 0.06 points (2.1%) at the six
MW Primaries and 0.16 points (4.2%) at the
eight MVA Primaries ([experiment 418](../../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md)).
Pretraining once and fine-tuning each month scored within 0.03 points of pretraining at every refit
and halves the graphics processing unit (GPU) time ([experiment
494](../../00494-simplify-demand-network-refit-cadence-and-pretrain-once/result.md)).

**The 14-day correction helps an XGBoost model fitted once.** It lowered a frozen model's score by
1.45 points in experiment 48 and supplied most of the stack's gain over N1 ([experiment
19](../../00019-primary-demand-stack-vs-manual-heuristic/result.md)).

## What does not work

**The 14-day correction stops helping a refitted XGBoost model.** It raised a weekly model's score
by 0.23 (99.375% interval +0.03 to +0.44) in experiment 34. In experiment 48 the 7-day correction
raised it by 0.16 and the 14-day correction by 0.08, both with intervals that include zero.

**Shorter recency windows, mixed pretraining, and ENS rows in pretraining did not help the neural
network.** Half-lives of 30 and 60 days did not reach the -0.05 bar against 90 days and raised the
leak of an injected load shift from 6.1 to 9.5 points of capacity, and pretraining on ERA5, CAMS,
and ENS rows together scored 0.062 (eight MVA Primaries) and 0.045 (six MW Primaries) behind
(experiment 511). Any ENS share in pretraining made the eight MVA Primaries 0.06 to 0.22 points
worse, steeply from 50%: 10% scored +0.076, 25% +0.059, 50% +0.105, 75% +0.183, and ENS only +0.220
([experiment 519](../../00519-ens-share-in-lagless-network-pretraining-ratio-sweep/result.md)). A
daily warm start scored worse than a daily restart (experiment 511).

**Replaying ERA5 and CAMS rows during fine-tuning did not fix cold winter nights.** Cold-night CRPS
at the eight MVA Primaries moved by -0.012, +0.026, and +0.047 points for the three replay arms,
against a minimum detectable effect of 0.169 points ([experiment
487](../../00487-era5-and-cams-replay-rows-during-demand-network-fine-tuning/result.md)). The small
gain at the six MW Primaries came mostly from one month.

**Smooth recency weights on the generalised additive model added nothing over a 24-month training
window**
([experiment 331](../../00331-primary-demand-recency-weights-on-gam-and-kalman/result.md)). Averaging
the quantiles of several XGBoost models gained 0.04 to 0.10 points, and a four-model version fell
short of its threshold in experiment 81.

## What is uncertain

**The XGBoost weekly gain is fragile.** In experiment 48 the contrast without D8, D12, and D5 was
-0.08 (99.44% interval -0.24 to +0.10). In experiment 81, D8 alone gained 2.05 points against the
stack. In experiment 355, weekly refitting lowered D8's score by only 0.045 against monthly refitting
(99% interval -0.118 to +0.001), so the 2.05 compares weekly refitting with the stack, and the 0.045
compares weekly with monthly refitting.

**The monthly-refit gain was not distinguishable from zero in its pre-registered test.** Experiment
34's pre-registered interval for monthly refitting included zero (-0.44 to +0.08). Experiment 81's
weekly gain in fold C (the 2022-23 winters) was -0.080 with an interval from -0.187 to +0.051.

**Experiments 48 and 81 disagree on the 14-day correction's value to a frozen model.** Experiment
48 found a gain of 1.45 points, and experiment 81 found +0.091 (-0.005 to +0.187) on its own runs.

**Older training years score worse, and the cause is not found.** One substation drives most of the
gap, and the gap was 1.71 points with ERA5 weather in both years, so the weather product is not the
main cause ([experiment
95](../../00095-age2-old-year-penalty-with-the-same-weather-product-demand/result.md)). Removing each
substation's trailing 365-day level removed 42% of the gap with all 13 series, and none of it without
the largest substation ([experiment
93](../../00093-age1-old-training-year-after-level-normalisation-demand/result.md)).

**A network fitted long before the forecast month loses accuracy, and the size of the loss is
uncertain.** A network fitted 4 weeks earlier scored about 3% worse, about 10% at 16 weeks, and about
6% at 26 weeks, with wide intervals, and D15 and D18 (two of the D codes) lost several times more ([experiment
640](../../00640-demand-fit-age-refit-k1-k3/result.md)). The two Christmas weeks confirm the weekly
gain at the six MW Primaries but not at the eight MVA Primaries ([experiment
512](../../00512-lagless-weekly-finetune-december-2024-sensitivity/result.md)).

**The lag-free neural network stays behind the lagged neural network.** After a daily refit it was
still 0.091 (eight MVA Primaries) and 0.071 (six MW Primaries) behind K2-S1 (experiment 511).

## Options for what to do next

1. **An option is to port the daily fine-tune of the lag-free neural network, taken from experiment
   511's next step.** The question is whether the fine-tune, at about 3 seconds a day, fits the
   serving path. [Final models](../../final-models.md) refits monthly with a fine-tune and applies
   experiment 599's rule, which drops the newest 7 days of rows and halves the weight of the next 7
   days. A daily cadence would have to keep that rule. The option needs engineering time and no new
   experiment.
2. **An option is to test refitting every 4 to 8 weeks for K1, taken from experiment 640's next
   step.** Experiment 640 judged this worth running, and found weekly refitting not shown to be worth
   more for K1. The option needs a set of refits, and the D15 and D18 histories need separate
   treatment.
3. **An option is to test an ENS-share variant aimed at the tails, taken from experiment 519's next
   step.** Experiment 519 found the 99th-percentile loss fell by up to 0.03 points, but only with
   the source flag set at fine-tuning. The option needs a few GPU fits, and the seed noise of 0.06
   to 0.08 points limits what it can show.
4. **An option is to test a longer ENS history.** The question is whether the backfill (2021-03 to
   2024-03) gives an extra validation fold or improves the single-stage or stage 2 models. It is
   option P16 in the [Research plan](plan.md), and it needs a cloud machine in `us-west-2`, which
   the maintainer must approve ([issue #959 (Extend ECMWF ENS training
   history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959)).
5. **An option is to find the cause of the old-year penalty, taken from experiment 95's next step.**
   The question is whether solar growth, heating, or changes in the electricity network drive it.
   Experiment 95 suggests them without testing any. The option needs a diagnostic experiment.

## What is parked

**The research plan's other planned work is on hold.** Option 4 above waits for an approved cloud
fetch. Load-shift event detection, and absorbing load-shifts in a refit, are paused until January
(see [Load-shift event detection: state at the January
pause](../lsc-detectors/state-at-january-pause.md)).
