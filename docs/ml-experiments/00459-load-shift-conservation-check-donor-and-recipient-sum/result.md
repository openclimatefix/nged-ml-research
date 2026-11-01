---
id: "00459"
slug: load-shift-conservation-check-donor-and-recipient-sum
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - hierarchy-and-pooling
status: planned
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
  its fits only if P1 and P3 both hold at the six. Stop rule: if the median G2 leak share at the
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
---

# Experiment 459: does a load shift leave the sum over the donor and its recipients unchanged?

**Status: planned. No result has been read.** NGED's event labels are incomplete, so every count
below is a lower bound and every placebo window may contain an unlogged event.

## Plan

- **Question.** When NGED moves load from one Primary to another, how much of the donor's step is
  still visible in the sum over the donor and its recipients, and how stable is each Primary's
  share of that sum outside events?
- **Events.** The 112 events of the corrected label table of experiment 395 that have an end and a
  depth. The donor is the first-named Primary. Experiment 456's candidate steps do not exist yet,
  so the unlogged-event set is left out and named as a gap.
- **Baseline and step.** The same half-hour-of-day median over the 7 days before onset is
  subtracted from each member's cleaned power. The step is the median deviation from 2 hours after
  onset to 2 hours before the end (the whole span if the event is under 6 hours).
- **Groups.** G1 is the donor and its labelled recipients among the 16 Primaries. G2 is G1 plus all
  graph neighbours of the donor in `neighbours_nged.json` (rebuilt from NGED's interconnection
  table by experiment 391's function). G3 is all 16 Primaries. Group G1 is the pre-stated group for
  idea 2.
- **Six and eight.** An event belongs to the six (D5, D9, D10, D11, D14, D16) or the eight (D2, D3,
  D4, D8, D12, D19, D21, D22) by its donor. Events donated by D15 or D18 are reported apart. A
  group that mixes MW and MVA readings is flagged.
- **Exclusions.** Events with a donor step under 0.3 MW, or whose measured sign disagrees with the
  corrected sign, are listed, counted, and left out of the headline.
- **Placebo.** 20 random onsets per event, outside every labelled event and its 7 days after, same
  duration and group.
- **Allocation stability.** Per group and Primary: the share outside events, the 30-day trailing
  median share, its one-day-ahead error, and the oracle allocation error beside experiment 392's R0
  lagged XGBoost forecast on the same rows. Share by month for the six.
- **Contrasts (k = 4).** Median G1 leak share minus the placebo's, at the six and at the eight; and
  the share of events with leak at most 0.25 minus the placebo's, at the six and at the eight.
  Intervals resample events (95%, Bonferroni k = 4), with leave-one-Primary-out and
  largest-event-dropped checks.
- **Compute.** CPU only, 2 cores, no fits. Cleaned power through `studies.power.scan_power`.
- **Standing practices not followed.** None apply: nothing is fitted, so retraining cadence, boosters,
  and curtailment cost do not arise. Curtailment cost is not computed (no forecast is scored).
