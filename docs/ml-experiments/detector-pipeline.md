# How the history detectors fit together

**Four detectors read each [Primary substation's](glossary.md#primary-substation) historical power
series in a fixed order, and their outputs serve two forecasting neural networks and the reports we
send to National Grid Electricity Distribution (NGED).** The detectors estimate the sign of the
power, the capacity of the unmetered solar panels, the power moved by switching, and the level
shifts and spikes that none of the first three estimates explain. Both forecasters predict a
Primary's power for the coming days from its last 14 days of power, weather, and calendar. The MSA
forecaster is the axial-attention neural network of experiments 713 to 733, and its name, MSA, comes
from the multiple sequence alignment of AlphaFold 2 ([experiment
713](00713-msa-axial-attention-demand-network-part-1/result.md), axial-attention demand network).
The Candidate is the demand neural network of [experiment
677](00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md) (built up one
component at a time) and [experiment
699](00699-injection-in-every-training-stage-of-the-candidate-demand-network/result.md) (synthetic
load shifts added to the training data). No detector reads a forecaster's output.

**This page records the design the maintainer approved, and every number on it is interim.** The
reverse-flow numbers come from D8 alone, the Primary where the sign work started. Two experiments
the design depends on are not yet published: experiment 734 (an unsupervised Bayesian offset model
of switching events) is running, and experiment 733 (the MSA forecaster reading D8's untangled
history) is planned.

## What the meters hide

**At some Primaries the meter hides the direction of the power, and at every Primary the meter hides
the output of the solar panels connected below it.** Of the Primaries the MSA experiments score, six
are metered in signed [megawatts](glossary.md#mw-and-mva) (MW), which show the direction of the
power. Eight are metered in megavolt-amperes (MVA), which give the size of the apparent power and
not its direction. Solar panels and other generators connected below a Primary can push power back
up the electricity network. An MVA meter reads that reverse flow as a positive value, so at an
MVA-metered Primary the sign of the power has to be estimated. D8 is one of the MVA-metered
Primaries, and its power sometimes flows in reverse.

## Why we detect switching events

**Detecting switching events serves three uses: undoing lagged power, scoring the forecast with the
events removed, and reporting the events to NGED.** A switching event is a period when an operator
moves part of one Primary's load onto a neighbour, so the Primary runs outside its [normal running
arrangement](glossary.md#abnormal-running-arrangement-ara-and-normal-running-arrangement-nra) (NRA).

- **Undo lagged power.** Lagged power is the past power readings the forecaster takes as inputs.
  Inside a switching event the 14-day history is distorted. The forecaster needs to know how much
  power was moved before it can forecast the NRA.
- **Score the NRA forecast with the events removed.** A forecast of the NRA should not be marked
  wrong for missing a transfer it was never meant to predict.
- **Report events to NGED.** NGED cannot easily extract switching instructions from its operational
  systems, so a table of detected events is useful to NGED in its own right.

**A level shift caused by the weather or the calendar is not a switching event.** The forecaster's
weather and date-time encoders model weather and calendar shifts. The detectors should therefore
leave weather and calendar shifts in the series.

## The stages, in order

**Each Primary runs through five stages, and each stage reads only the outputs of earlier stages.**

1. **Sign of the power (MVA-metered Primaries only).** An MVA meter reports the size of the apparent
   power and not its direction. The sign comes from a stand-alone neural network: the sign-only
   U-net (a convolutional neural network with skip connections) of [experiment
   732](00732-msa-d8-untangling-network-synthetic-injections/result.md) (one untangling neural
   network for D8). The sign network outputs the probability of reverse flow for each half-hour. A
   half-hour with a probability between 0.05 and 0.95 is flagged ambiguous and treated as missing by
   the later stages.
2. **Unmetered solar capacity.** A Bayesian fit with a physical forward model turns irradiance from
   the Copernicus Atmosphere Monitoring Service (CAMS) and temperature and wind speed from the ERA5
   reanalysis into solar output per unit of capacity. Capacity is held constant within each quarter.
   The prior (the belief the fit holds before it reads the data) favours growth from one quarter to
   the next. The DC:AC ratio (the panels' direct-current rating over the inverters'
   alternating-current rating) is a fixed prior, because [experiment
   73](00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) (orientation mix and DC:AC ratio)
   could not pin the ratio down. Wind capacity may follow later.
3. **Switched offset.** Experiment 734's offset model reads a residual: the signed power, minus a
   baseline built from weather and calendar inputs without any past power, minus the stage 2 solar
   estimate. The offset model outputs quantiles of the switched megawatts in each 6-hour block, and
   an event table with a probability, an evidence grade, and an 80% interval on each event's size.
4. **One feedback round.** Stage 2 is refitted with the called events masked, and stage 3 runs
   again. The round stops there and reports how far each output moved.
5. **Residual report.** Stage 3 grades each level change by two cues: a mirrored step at a
   neighbour, and a return to the old level. The report lists the permanent level jumps and gain
   changes that show neither cue, graded "neither". A half-hour layer looks for spikes of 1 to 6
   hours in what remains, because the 6-hour blocks of stage 3 cannot see them. The spike layer is
   untested.

**The sign network has to run before the offset model wherever the power flows in reverse.** A
change from import to export reads on an MVA meter as an upward step, and the offset model would
call that step a transfer.

**The sign network stays a separate, frozen neural network.** In experiment 732, the neural network
that also learned solar output and transfer steps scored a Brier score (the mean squared error of a
probability) 0.00065 worse than a sign-only neural network trained on the same windows (interval
+0.00004 to +0.00137). The extra tasks did not help the sign. A separate neural network also keeps
the sign's own harm checks independent of changes to the other stages.

## Why a pipeline and not one joint model

**The four products work at different time scales and rest on different evidence, so a pipeline lets
each stage be tested on its own known answers.** The time scales are a half-hour for the sign, a
quarter for solar capacity, a 6-hour block for the switched offset, and a half-hour again for
spikes.

**Three test results would make a joint model of solar and switching the better design:**

- on synthetic series with simulated solar growth, the offset model calls more false switching
  events than on event-free time;
- the capacity estimate moves outside its own 80% interval between the fits with and without the
  called events masked; or
- the feedback round of stage 4 does not settle.

## What each stage hands on

**Every output carries its own uncertainty, and every "unexplained" label names the baseline it is
measured against.**

| Product | Value | Uncertainty |
|---|---|---|
| Sign | probability of reverse flow per half-hour | calibrated probability; ambiguous flag |
| Solar | capacity per quarter; MW per half-hour | 80% interval on each |
| Switched offset | switched MW per 6-hour block; event table | seven quantiles, probability of zero, an "unknown" flag |
| Residual report | level jump, gain change, spike | interval on size; grade "neither" |

## Two versions of every product

**Each product comes in a whole-record version and a past-only version, because the whole-record
version reads data after the half-hour it labels.** The whole-record version suits the evaluation
masks and the reports sent to NGED. The past-only version is recomputed with the record cut at each
forecast origin, and only the past-only version may enter a forecaster's history inputs. Feeding the
whole-record version into history inputs would leak the future into the forecast. A look-ahead of
the same kind, weather issued after the forecast origin, voided every comparison above 24 h of lead
in [experiment 404](00404-mdn-lagless-and-calibration-module-trust-weights/result.md) (lag-free
mixture density network), as [experiment
721](00721-msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network/result.md) records.

**The past-only version is less accurate.** In [experiment
730](00730-msa-d8-sign-detector-with-demand-head-and-masking/result.md) (D8 sign detector with a
demand output), the past-only variant of the chosen arm scored a Brier score of 0.01049, against
0.00958 for the whole-record variant of the same arm. ERA5 and CAMS are published about 5 days late,
so the past-only version can use ERA5 and CAMS only for history rows older than about 5 days.

## How the MSA forecaster uses the outputs

**In the design, the evaluation masks come first, and the history inputs wait until experiment 733
reports.** Both uses are planned, and neither exists in a published MSA experiment yet.

**The design reports every score on three masks: confirmed switching, ambiguous, and clean.** The
masks use the whole-record version, so the NRA forecast is scored with the switching events removed.
Reporting all three stops a detector's mistakes from silently flattering the score.

**The history inputs give the forecaster what it needs to undo lagged power inside a switching
event.** Each of the 14 history rows would carry the past-only probability of reverse flow, the
solar estimate in MW, the switched-offset quantiles, and the probability of zero offset. With those
inputs the forecaster can tell how far a lagged reading sits from the NRA, and forecast the NRA.

**The forecaster should ride through a switching event but follow a slow level shift that is not a
switching event.** The MSA forecaster already follows shifts in recent power more than the Candidate
does: 7 to 28 days after an injected shift began, it followed 0.225 more of the shift's depth at the
six signed-MW Primaries (interval +0.171 to +0.278; [experiment
726](00726-msa-candidate-follow-share-on-injected-shift-bank/result.md), follow share on injected
shifts). Recent power alone does not settle which kind of shift is under way. In [experiment
714](00714-msa-era-identifiability-audit-and-frozen-probe/result.md) (permanent change or temporary
move), 6 summaries of a Primary's last 14 days separated a permanent level change from a temporary
load move with an AUROC (area under the receiver operating characteristic curve) of only 0.63 at 7
days after the onset.

**The history inputs can improve the forecast by only a little, and a wrong mask can do much more
harm.** A perfect switching detector would lower the best guess's quantile score by 0.42% (-0.47% to
1.96%; [experiment 44](00044-sw2-2-switching-share-of-best-guess-score/result.md), switching share
of the best guess's score). One wrongly masked interval of 60 days raised the tail error outside
logged switching by 57% (0.4% to 134%; [experiment 38](00038-switching-mask-value/result.md),
switching mask value). The gap between a gain of 0.42% and a harm of 57% is why, in the design, the
forecaster follows a level change graded "neither" by default.

**Cleaning and scoring must not use the same detector.** If one detector both cleans a forecaster's
training targets and supplies the masks the forecaster is scored on, that detector's misses flatter
the score.

## Feedback loops to guard against

**Seven loops could let one stage's error reinforce another's:**

- **Solar growth read as a level shift or a transfer.** Test: experiment 734's synthetic series with
  simulated solar growth, with and without the stage 2 solar term.
- **A transfer absorbed by the solar fit.** Test: capacity with and without the called events masked
  (stage 4).
- **A sign error read as a step.** Guard: the ambiguous flag.
- **A forecaster's error used as a detector baseline.** A forecaster that follows a shift removes
  the shift from its own error, so the detector would miss the shift. Guard: the stage 3 baseline
  uses no past power.
- **Cleaning and scoring with the same detector.** Guard: report all three masks.
- **Solar capacities setting the sign network's simulated solar while the sign feeds the solar
  fit.** Guard: in each round, freeze either the sign network or the solar fit.
- **Weather read as an unexplained shift.** A cold spell or a holiday shifts a baseline for weeks
  and then returns. Experiment 734's shared neighbourhood term targets regional weather. A single
  industrial Primary whose load does not follow the weather stays a risk.

## How solid each piece is today

- **Reverse flow: solid on synthetic days, unmeasured on real reverse flow, at D8 only.** In
  experiment 730 the sign-only arm scored a Brier score of 0.00853 against 0.03643 for reading every
  half-hour as import, and was calibrated where the reading is 1 to 3 MVA (mean probability 0.2479
  against an observed share of 0.2506). The sign-only arm of experiment 732 scored 0.00721. The real
  days on which the sign is known held no reverse-flow half-hours, so no real reverse-flow half-hour
  has been scored. [Experiment 729](00729-msa-d8-neural-untangler-synthetic-known-answers/result.md)
  (D8 untangler on synthetic days) is invalid, because a bug silenced half of the simulated solar
  farms.
- **Switching: weak so far.** The rule detector [RO-NC](glossary.md#ro-nc) found 0.357 of the moved
  power of 35 real transfers lasting over 1 day ([experiment
  101](00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). The event-table
  method E1-J-sym of [experiment
  680](00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md)
  (symmetric search) found 0.515 of the injected megawatts of long, deep transfers, and paired both
  ends of a transfer for 0.379 of those megawatts. The transfer head of experiment 732 found 7.7% of
  the simulated steps of 5 [points of capacity](glossary.md#capacity-and-points-of-capacity) or
  more. Experiment 734 has not reported.
- **Solar: solid where the meter shows the sign, weak at MVA-metered Primaries.** At the six
  signed-MW Primaries, a linear probe recovered 0.76 to 0.80 of an injected solar fleet's megawatts
  in [experiment 656](00656-pv-probe-through-the-mva-fold-at-eight-primaries/result.md) (solar probe
  through the MVA fold), as [experiment
  662](00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md) (matched-level
  redesign) records. With the same Primaries' power turned into synthetic apparent power, the fold
  model of experiment 662 recovered 0.73 to 0.90. The fold model reads the solar term through the
  fold by which an MVA meter turns negative power into a positive reading. At real MVA-metered
  Primaries, 82 to 94% of the fold model's extra capacity disappeared when the fold model's
  reactive-power terms were fixed ([experiment
  671](00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement/result.md), fair
  denominator). Solar capacity at MVA-metered Primaries therefore depends on the sign stage. On real
  days before a solar farm existed, the solar output of experiment 732's neural network still stayed
  above zero. A solar estimate gave no detectable forecast gain in the MSA forecaster ([experiment
  716](00716-msa-unmetered-pv-part-1-history-estimate/result.md), history estimate; [experiment
  717](00717-msa-unmetered-pv-part-2-target-row-forecast/result.md), target-row forecast).
- **Spikes: untested.** No experiment published here has scored the detection of a spike of a few
  hours.
- **Scale beyond D8: paused** until NGED's new data arrive.

## The maintainer's current build order

**The maintainer's current plan builds the stages in this order, and the plan may change as results
arrive:**

1. Finish experiment 734 (the Bayesian offset model).
2. Export the sign network's probability of reverse flow for D8's history, in both versions.
3. Fit the Bayesian solar capacity: first at the signed-MW Primaries, with a test that injects a
   known solar fleet, then at the MVA-metered Primaries on the signed series.
4. Run the feedback round.
5. Build the spike layer and the residual report.
6. Build the evaluation masks, then the past-only history inputs once experiment 733 reports.
