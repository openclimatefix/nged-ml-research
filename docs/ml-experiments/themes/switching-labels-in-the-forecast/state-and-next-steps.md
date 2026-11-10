# Load-shift labels in the forecast: state and next steps

**Telling the Primary demand forecast about load shifts has not improved it on real data, and the
final demand models use no label table or detector.** On injected load shifts, labels recover most
of the lost accuracy only when the megawatts (MW) moved are known exactly. The [final models
page](../../final-models.md) describes how the final models cope without labels. The [theme
page](index.md) holds the full results, and the [state of the event
detector](../lsc-detectors/state-at-january-pause.md) records where the detector work paused.

**A load shift is a transfer of demand between neighbouring Primary substations for hours or days.**
The normal running arrangement (NRA) is the electricity network configuration without any load
shift. A lag is an earlier power reading used as an input. RO-NC is the refit-offset neighbour-count
detector, which flags a load shift from a 24-hour step in a Primary's forecast error (see the
[detector page](../lsc-detectors/index.md)). The gap is the extra error of a forecast on lags that a
shift has changed, against the same forecast on unchanged lags.

**Scores are quantile scores, which are twice the mean pinball loss over 11 quantile levels.** They
are in points of effective capacity (1% of a substation's capacity), and a positive change is a
worse forecast. NGED's event labels are incomplete, so every count of real events is a lower bound.
Unless a paragraph says otherwise, results come from 16 Primary series and 292 daily forecast runs.
Verdict words in backticks, such as `inconclusive`, follow the [glossary's
definitions](../../glossary.md#pre-registration-decision-rule-and-verdict-words).

## What we know

**On real data, no label-based change lowered the score by an amount the data can show.** Dropping
labelled load-shift half-hours from the training targets changed the score by +0.004 (98.33%
interval -0.026 to +0.025), and only 22 of 82 labelled events fall in the scored year ([Experiment
75](../../00075-sw2-11-nra-forecast-with-labelled-switching-masks/result.md), `inconclusive`).

**A perfect detector could lower the quantile score by little in the year tested.** In the band of
events that the event labeller (a script that marks load-shift events from NGED's logs and the power
series) finds most reliably, a perfect detector would lower the quantile score by 0.42% (95%
interval -0.47% to 1.96%), against a 3% threshold. These events touch 3 of the 16 series. Over the
22 events visible in the power data, the bound is 2.35% (-0.22% to 5.73%), an interval that includes
zero and straddles the threshold ([Experiment
44](../../00044-sw2-2-switching-share-of-best-guess-score/result.md), `refuted`).

**Only the true MW recovered most of the gap on injected load shifts.** With the true MW and a
forecast trained on NGED's logged shifts, 83% (70% to 91%) of the gap closes. With the MW from RO-NC
as built, 16% (-17% to 67%) closes, which cannot be told from zero ([Experiment
88](../../00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md),
`inconclusive`). These shares rest on 2,452 forecast origins inside injected transfers (4 onset
months, 13% of origins).

**Taking one MW per half-hour from the largest overlapping step recovered more, but the choice was
made after seeing the results.** The loss comes from overlapping detector intervals that add up (21%
of marked cases). The lag correction alone closed 48% (18% to 71%) of the gap, and the RO-NC-trained
booster that also drops those half-hours from its training targets closed 43% (17% to 71%)
(Experiment 88).

**A calibrated probability of a lag inside a load shift did not beat the detector's yes-or-no
flag.** On injected transfers, the probability lowered the Brier score (the mean squared error of a
probability forecast) by 3.3% (97.5% interval -7.6% to +1.6%), against a 20% requirement. Its area
under the curve against NGED's logs (where 0.5 is chance) was 0.53, so the experiment advises
against a band-widening rule built on it ([Experiment
91](../../00091-sw2-26-calibrated-probability-of-abnormal-arrangement/result.md), `inconclusive`).

**Feeding the detector's alarm into the GAM-K Kalman filter did not help.** GAM-K is the generalised
additive demand model whose Kalman filter corrects each forecast with its recent errors. The alarm
helped inside a transfer and hurt for 72 hours after it ended. The alarm came 48 hours after the
step, which was no better than a random placebo ([Experiment
141](../../00141-switching-detector-fed-into-gam-k/result.md), `not supported`).

**After the alarm, subtracting the detected MW closes much of the gap, but the weather alone closes
more.** Subtracting closed 71% (25% to 85%) of the gap to a forecast fed the true NRA series, and
the weather model alone closed 78%. Counting missed transfers, subtracting gained nothing and
masking gained 21% ([Experiment
151](../../00151-nra-forecast-rescored-against-the-true-nra-series/result.md), `partly supported`).

**A second Kalman filter state for the transfer added nothing over masking.** It closed 78% of the
gap and masking the readings inside the alarm closed 81% ([Experiment
153](../../00153-latent-nra-state-in-the-gam-k-filter/result.md), `partly supported`).

## What does not work

**Hiding lags that lie inside a load shift made the forecast worse.** Setting them to missing
worsened the score by 0.409 (+0.281 to +0.517) on the 352,765 rows with a lag inside an abnormal
running arrangement (the union of RO-NC and the logs). ([Experiment
89](../../00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/result.md), `refuted`).

**Giving the forecast the unusualness of each lag worsened the score by 0.355 (+0.209 to +0.513)**
([Experiment 90](../../00090-sw2-28-lag-unusualness-inputs-all-rows-and-switching/result.md),
`refuted`).

**Three times as many synthetic shifts in training did not help.** The score changed by +0.079
(-0.031 to +0.153) ([Experiment
96](../../00096-sw2-23-synthetic-switches-injected-into-training/result.md), `inconclusive`).

**Random lag dropout in training did not widen the band enough.** The 5% to 95% band covered 0.799
on rows with a missing lag for the plain forecast and 0.836 with 30% of training rows hidden (the
best arm), against 0.90 nominal. The band width did not change ([Experiment
100](../../00100-drop1-random-lag-dropout-widens-intervals/result.md), `inconclusive`).

**Every label experiment on this theme used hindsight labels (labels known only after the event), so
nothing here is ready for the serving path.**

## What the final models do instead

**The final demand models for Primaries contain no label table and no detector input.** They use
three substitutes.

- **Injected shifts in training.** The lag encoder, the part of the neural network that reads the
  lagged power, trains on random synthetic shifts covering 5% of each Primary's rows.
- **A refit embargo.** The newest 7 days of training rows are held back and the 7 days before them
  are down-weighted by half. This cut the share of a shift leaked into the median forecast 7 days
  after it ends from 0.16 to 0.114 ([Experiment
  599](../../00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption/result.md),
  `partly supported`).
- **A prevailing output.** A second output mixes the NRA quantiles with a shifted copy, and extra
  lagged inputs make it react to a new shift 20.4 hours sooner on injected events, below the 24
  hours set in advance. Which extra-lag set to port is open, because the 8-input set scored worst
  inside the 2 real events that can be read ([Experiment
  681](../../00681-extra-lagged-power-inputs-for-the-prevailing-forecast/result.md), `small gain`).

**The embargo's leak of 0.114 is above the proposed 0.10 bar, and the six MW Primaries lose 1.3
points of q05 to q95 coverage (Experiment 599).**

**Detector features for the prevailing output were not adopted.** The features reacted 1.59 days
sooner than the earlier head, but extra lagged power alone reacted 1.25 days sooner ([Experiment
687](../../00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md), `partly
supported`). Its reaction times are not comparable with Experiment 681's, which used different
events and folds.

**The prevailing output is better overall, and December 2025 is the exception.** It scored 0.17
points better than the NRA output for origins in 2026-01 to 2026-06 and 0.42 better in 2026-07 to
2026-10 ([Experiment
668](../../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)).
After the one verified shift, at D5 and D16 in December 2025, it scored worse than the NRA output
for 7 days after the shift ended. December 2025 is a fallback month (B-haz fallback head).

## Open questions

- **Whether the detector's MW, with each shift counted once and ended promptly, recovers more than
  the 43% of the gap is unknown.** Only a post hoc rule has been scored.
- **Whether any label improves the forecast on real shifts is unknown.** The logged events in one
  year are too few (Experiment 100).

## Next-experiment options

**Each option below comes from the "What to try next" section of the experiment named.** The final
models come first and load-shift event detection is paused until January, so options that need a
better detector wait for it.

1. **An option is to re-run Experiment 88 with the de-duplicated MW.** The rerun would keep the
   largest of overlapping same-sign intervals and let the MW decay after the end (from Experiment
   88). It depends on the detector work.
2. **An option is to replace a masked lag with a forecast of the NRA** (from Experiment 89).
3. **An option is to train synthetic shifts with the detector's errors** (from Experiment 96).
4. **An option is to widen the band where a lag is missing, with an indicator column** (from
   Experiment 100).
5. **An option is a Kalman-filter correction that decays over the typical transfer duration** (from
   Experiment 141).
6. **An option is a detector that alarms sooner, paired with a forecast that drops recent power
   while a transfer is suspected** (from Experiment 151).
7. **An option is a gated embargo that starts only when a gate fires** (from Experiment 599).

## What is parked

**Hindsight-label results are parked until a causal detector is better, which is January work.** The
[detector state page](../lsc-detectors/state-at-january-pause.md) records why.

**Experiment 116 (differentiable physics, edge flows between Primaries) is `inconclusive`.** Its
known-answer gate failed: the fit cannot see transfers shorter than 7 to 9 days, and it found 0.22
of the injected MW over 1 day. The [network research
plan](../differentiable-physics-network/plan.md) holds its follow-ups, a drift term and a topology
check.

The [experiments page](experiments.md) lists every experiment in the theme.
