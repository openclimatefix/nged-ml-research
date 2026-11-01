---
id: "00459"
slug: load-shift-conservation-check-donor-and-recipient-sum
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - hierarchy-and-pooling
status: finished
kind: other
protocol: none
size: >-
  medium. Stored: tables and figures only, nothing fitted. More than one design would test the
  hypothesis (the group definition, the baseline, the interior window), so the plan fixes them
  before any result is read. Callers: the script imports the folders of experiments 377 (series
  table) and 391 (neighbour graph). Production serving path and degradation rule: not touched.
hypothesis: >-
  When NGED moves load from one Primary substation to another, most of the donor's step is still
  visible in the sum over the donor and its labelled recipients, so a forecast of that group sum,
  split by a slowly changing share, would be immune to a transfer between members.
prediction: >-
  Written before any result is read. P1. At the six Primaries with signed readings, for events
  with an observed recipient, the median leak share of group G1 is at most 0.25, at least 70% of
  those events have a leak share of 0.25 or less, and the placebo median is above 0.5. P2. The
  share of events with an unobserved recipient is 20 to 40%, and their G2 leak share is above 0.5.
  P3. The oracle allocation error (true group sum times the trailing 30-day median share) is under
  half the median absolute error of experiment 392's R0 lagged XGBoost forecast at the six.
decision_rule: >-
  The group definition for the group-sum forecast (idea 2) is fixed as G1 (the donor and its
  labelled recipients that are among the 16 Primaries) before any result is read. Idea 2 is worth
  fitting only if P1 and P3 both hold at the six. Stop rule: if the median G2 leak share at the
  six exceeds 0.5, or fewer than 50% of events with an observed recipient have a leak share of 0.25
  or less, the group sum is not conserved enough and idea 2 is closed. If P1 holds only for G3, the
  whole system is conserved and the groups are too small to be useful. The MVA-only eight are
  reported and are not part of the rule. Threshold: no code is ported, and the check only gates a
  16 to 64 fit experiment, so the threshold is the prediction above.
benchmark: >-
  A placebo: the same statistic on windows of the same duration and the same group, placed at
  random onsets outside every labelled event and the 7 days after it, 20 per event. The placebo is
  windows outside labelled events, not clean windows, because NGED's event labels are incomplete.
  For allocation error, experiment 392's R0 lagged XGBoost forecast.
row_set_id: none
metric: >-
  Leak share (the absolute step of the group sum divided by the absolute donor step) for groups
  G1, G2, and G3; and the oracle allocation error in points of effective capacity.
code: ml-experiments/00459-load-shift-conservation-check-donor-and-recipient-sum/
verdict: refuted
best_arm: >-
  none (no fit; group G1 leaks a median of 0.53 of the donor step at the six, not distinguishable
  from the placebo with 15 events)
headline: >-
  At the six Primaries with signed-MW readings, the median G1 leak share is 0.53 against a placebo
  median of 0.54 (15 events with an observed recipient); 13% of those events have a leak share of
  0.25 or less. The stop rule fires, so the group-sum forecast (idea 2) is closed.
lessons: >-
  With 15 events at the six, the sum over a donor and its labelled recipients is not distinguishable
  from the same sum on windows outside the labelled events. The oracle share-split forecast has no
  lower error than experiment 392's R0 lagged forecast on harm rows at the six.
take_home_message: >-
  When NGED moves load from one Primary substation to another, the sum over the donor and its
  labelled recipients still steps by about half of the donor's step, as it does in windows with no
  logged event. A forecast of that sum split by a share therefore cannot absorb a transfer, and even
  with the true sum known, the split is no more accurate than the lagged XGBoost forecast of
  experiment 392, so the group-sum forecast does not reach its threshold.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not run the group-sum forecast fits for Primary substations: at the six the sum over the
    donor and its labelled recipients leaks a median 0.53 of the donor step, and the oracle
    share-split gain over the R0 lagged forecast on harm rows is -0.10 points (P1, P3).
  - >-
    Treat a labelled event's named recipient as an unreliable mirror of the donor's step: the
    donor's step is only partly matched in the group sum (leak share 0.53 at the six, against a
    placebo 0.54).
  - >-
    Draw placebo windows beside any step statistic taken from logged events, and judge the
    statistic against the placebo spread (10th to 90th percentile of the G1 leak share is 0.09 to
    1.9 at the six).
---

# Experiment 459: does a load shift leave the sum over the donor and its recipients unchanged?

**When NGED moves load between Primary substations, the sum over the donor and its labelled
recipients still steps by about half of the donor's step, as it does in windows with no logged
event.** That sum is therefore not conserved by the event, and a forecast of the sum split by a
slowly changing share cannot absorb a transfer. The pre-registered stop rule fired on both its
tests, and the oracle share-split forecast is no more accurate than experiment 392's R0 lagged
XGBoost forecast, so the group-sum forecast does not reach its threshold. With 15 events at the six
the evidence is thin. NGED's event labels are incomplete, so every count is a lower bound and every
placebo window may hold an unlogged event.

**Terms.** NGED's label table names a donor Primary and usually a recipient Primary for each event;
the step is the median change in a series' power over the event's interior, from its 7-day
half-hour-of-day baseline. G1 is the donor plus its labelled recipients among the 16 Primaries, G2
is G1 plus the donor's neighbours in NGED's interconnection table, and G3 is all 16 Primaries. The
leak share is the absolute step of the group's sum divided by the absolute step of the donor: 0
means the transfer cancels inside the group, and 1 means no cancellation. The six Primaries (D5,
D9, D10, D11, D14, D16) report signed MW, the eight (D2, D3, D4, D8, D12, D19, D21, D22) report
MVA magnitude only, and D15 and D18 are reported apart. A placebo window has the event's duration
and group at a random onset outside every labelled event and the 7 days after it (20 per event).
The oracle forecast is given the true group sum and splits it by the trailing 30-day median share.
A harm row is a forecast row whose target lies outside every labelled event but whose 1, 2, or
7-day lag lies inside one. The R0 lagged forecast is experiment 392's XGBoost forecast with
power lags, refitted on plain targets by experiment 402 (the stored forecast read here). Points
are percentage points of a Primary's effective capacity.

## Figures

![Six events: deviation of the donor, the first recipient, and the group sum](figure_events.svg)

If load were conserved, the black sum line in figure 1 would stay flat through the shaded event.

![Histogram of the G1 leak share for labelled events and placebo windows, at the six and at the eight](figure_leak_histogram.svg)

![Each Primary's trailing share of its group sum over time](figure_share_over_time.svg)

## Results

- **P1 (fails).** P1 predicted a median G1 leak share of at most 0.25 at the six for events with a
  recipient among the 16, against a placebo above 0.5. Observed: 0.53 (15 events) against a
  placebo median of 0.54, and 13% of events at 0.25 or less (placebo 30%).
- **Stop rule (fires).** The median G2 leak share at the six is 1.73 (28 events with a valid sum
  out of 31; limit 0.5), and 13% of observed-recipient events have a G1 leak share of 0.25 or less
  (limit 50%). A leak share above 1 means the sum moves more than the donor alone.
- **P2 (holds).** Of the 112 events, 31 (28%) name no recipient, inside the predicted 20 to 40%.
  By donor group the share is 24 of 46 at the six (52%), 4 of 54 at the eight (7%), and 3 of 12
  for D15 and D18. At the six, the median G2 leak share of these events is 1.21 (14 valid of 16),
  above the predicted 0.5.
- **P3 (fails).** The oracle share-split error has a median of 3.91 points at the six, against
  3.31 for the R0 lagged forecast on the same rows (ratio 1.2; 30 group-and-Primary pairs, one
  with a ratio under the predicted 0.5).
- **Harm rows (fail).** The oracle's median paired gain over the R0 lagged forecast is −0.10
  points at the six (18 pairs, 26,786 rows); the rule needs at least 0.1. Per Primary the gain is
  +0.21 at D10, −0.47 at D11, and −0.36 at D14; D5, D9, and D16 have no harm rows, because the
  R0 forecast covers four test months. The oracle uses the true group sum at the target time, so
  its gain is a ceiling for a trailing-share split.
- **At the eight, leak is worse than placebo.** Median G1 leak share 0.89 against 0.40, and 12%
  at 0.25 or less (placebo 38%). MVA magnitudes are not conserved when flow reverses, and 19 of the
  34 events mix MW and MVA readings.
- **At all 16 Primaries (G3),** the median leak share is 5.1 at the six, with a valid sum for only
  16 of 31 events, because the sum needs all 16 readings in the same half-hour.

| Pre-registered contrast (G1, observed recipient) | Events | Estimate | Interval |
|---|---|---|---|
| Six: median leak minus placebo median | 15 | −0.01 | −0.27 to 0.91 |
| Six: share at most 0.25 minus placebo share | 15 | −0.16 | −0.38 to 0.14 |
| Eight: median leak minus placebo median | 34 | 0.49 | 0.23 to 0.65 |
| Eight: share at most 0.25 minus placebo share | 34 | −0.26 | −0.39 to −0.11 |

Intervals resample events with their placebo windows, at 98.75% so that the four contrasts hold
together at 95%. The smallest median-leak contrast the placebo spread can detect is 0.18 at the
six and 0.16 at the eight, so the six's interval, much wider than that, cannot separate the events
from the placebo. Leaving out one donor moves the six's median-leak contrast between −0.21 and
0.34, and dropping the largest event gives 0.07.

- **Share stability.** The trailing share moves slowly (figure 3): the month-to-month range of a
  six Primary's share has a median of 0.06 and a maximum of 0.38 (D14 with D18). Multiplied by the
  group sum, the one-day-ahead share error has 80% band edges between −14 and +11 points at the
  six, against a median absolute error of 2.9 to 4.2 points for the R0 lagged forecast.
- **Dead meters.** Placebo windows at the six with more than 5% missing half-hours in the group
  (a gap proxy for a dead or flagged meter, not a direct test) have a median G1 leak share of
  0.85 against 0.59 for the rest, from only 11 windows. That is too few to conclude, and the
  windows give no sign of a shield.
- **What an operator would see.** The forecast for a Primary would read "the group's demand times
  this Primary's usual share". The group sum is not smooth through logged events: in figure 1 the
  black sum swings as much as, or more than, the donor.

## Caveats

- **Incomplete labels.** Placebo windows are outside labelled events, not clean windows.
- **Exclusions.** Of the 112 events, 37 are excluded (22 with a donor step under 0.3 MW, 8 with no
  measurable donor step, 7 whose measured sign disagrees with the corrected sign), leaving 75
  (31 at the six, 38 at the eight, 6 for D15 and D18).
- **Event baselines.** Of the 75 events, 52 (24 of 31 at the six) have a 7-day baseline that
  overlaps an earlier labelled event or its 7-day tail, because the same donors repeat. Placebo
  baselines never do. Placebo onsets are uniform over the record, not matched in season.
- **Scope.** Experiment 456's candidate steps are not used. The brief's 12 to 36 hour band is not
  reported, because the oracle does not depend on lead.

## What to try next

- Treat a labelled event as a property of one Primary's own series, as experiments 400 and 402 do.
- A forecast that borrows the neighbours' lag surprise (as experiment 402's gate does) stays open.

## Code and data

- Code: `ml-experiments/00459-load-shift-conservation-check-donor-and-recipient-sum/`. It reuses
  experiment 377's series table, experiment 391's neighbour graph, experiment 395's corrected label
  table, and the R0 lagged forecast stored by experiment 402.
- Tables: `table_groups.csv`, `table_contrasts.csv`, `table_allocation.csv`. The per-event and
  placebo rows, with the monthly shares, are in the private work folder.
