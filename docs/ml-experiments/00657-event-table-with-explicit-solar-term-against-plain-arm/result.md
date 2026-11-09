---
id: "00657"
slug: event-table-with-explicit-solar-term-against-plain-arm
title: >-
  Does subtracting an explicit solar term from the Primaries' residuals reduce false events at sunny
  midday without losing injected events?
theme: lsc-detectors
also_relevant_to:
  - solar-disaggregation
status: planned
date: 2026-11-09
kind: other
size: >-
  medium. (1) Stored: one small private residual panel and event tables under
  data/ml-experiments/00657-event-table-with-explicit-solar-term-against-plain-arm/ only. (2) Serving
  path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for
  the solar curve, the capacity, and the gain, so each is fixed below before any score is read. (5)
  Callers: this folder's scripts import the folder of experiment 653 for the decoder, the generator,
  and the matching rules, and read the stored outputs of experiments 6, 649, and 653; nothing
  imports them. CPU only, at most 2 cores.
hypothesis: >-
  Subtracting an explicit solar term (the capacity of unmetered solar behind a Primary from
  experiment 649, times a physics curve built from recorded ERA5 irradiance, times a gain fitted on
  the tuning years) from the six megawatt Primaries' residuals lowers the rate of false events on
  sunny middays, and loses no more than 0.03 of the injected megawatts that the plain arm of
  experiment 653 finds.
prediction: >-
  Written before any score is read. P1: the rate of decoded events per sunny-midday Primary-year
  at the six megawatt Primaries, in rows more than 14 days from any labelled or verified event,
  falls by 20% or more against the plain arm. P2: MW-weighted recall of injected events of 4 noise
  units or more that last 1 day or more, at the same six Primaries, changes by -0.03 or more
  (a fall of at most 0.03). P3: the rate of decoded events on non-sunny days at those Primaries
  changes by less than 10% in either direction. P4: a placebo that gives each Primary another
  Primary's capacity lowers the sunny-midday rate by at most half as much as the real capacity does.
  P5: the normalised mean absolute error of the residual, in the sunny-midday half-hours at those
  Primaries, falls by 3% or more.
decision_rule: >-
  Fixed before any score is read. The solar arm is supported if P1, P2, and P4 hold. Penalties,
  thresholds, and the generator are those of experiment 653, with no retuning, so any change is the
  solar term's. The gain is fitted on 2022 and 2023 only, and the scored years are 2024-01-01 to
  2026-06-30. The final-test window from 2026-07-01 is never read. Contrasts are printed beside a
  neutral placebo (capacity swapped between Primaries, which leaves the decoder and every
  non-solar row unchanged) and at equal width (the same decoded-event count, by lowering the
  confidence threshold of the arm that has fewer events). The porting threshold is zero, because
  the table is a research output and changes no production code.
benchmark: >-
  The plain arm of experiment 653 (masked frozen seasonal slow level, no solar term), run on the
  same injected worlds and the same real record.
row_set_id: >-
  experiment 653's grid: 15 Primaries, half-hourly, 2022-01-01 to 2026-06-30, cleaned power. The
  solar term applies to the six Primaries metered in megawatts (D5, D9, D10, D11, D14, D16); the
  other 9 are identical in both arms. Not comparable with the leaderboard.
metric: >-
  Events per Primary-year in the event-free stratum, split into sunny-midday days and other days;
  MW-weighted recall of injected events by duration band; normalised MAE of the median forecast in
  four scenarios; logged shifts matched.
code: ml-experiments/00657-event-table-with-explicit-solar-term-against-plain-arm/
---

**This plan tests whether a physics solar term, with the capacity from experiment 649, removes the
sunny-midday false events that the plain arm of experiment 653 decodes.** Experiment 653 decoded
5.4 events per Primary-year in rows far from any logged event, and unmetered solar behind the six
megawatt Primaries leaves a sunny-day shape in the residual. We subtract that shape and re-run the
same decoder on the same injected worlds and the same real record. Nothing is tuned again.

## Plan

- **Solar curve.** For each of the six megawatt Primaries, the unit curve `u(t)` is the ERA5
  surface shortwave irradiance (the `shortwave` column of experiment 6's feature table, interpolated
  from hourly to half-hourly), divided by its 99th percentile over daylight half-hours at that
  Primary in 2022 and 2023. It is zero at night.
- **Capacity.** `C` is the calibrated 50th-percentile AC capacity of experiment 649 (probe cutoff
  2026-06, no cloud template): the April-to-September 2025 value for dates from 2025-01-01, and
  the April-to-September 2024 value for earlier dates. The table describes a summer half-year, so
  the same `C` is used in every month, and the gain below absorbs the winter difference.
- **Gain.** For each Primary, `g` is the slope of a Huber regression of the plain residual on
  `C·u(t)` over daylight half-hours of 2022 and 2023, with half-hours inside logged or verified
  events left out. The solar-arm residual is the plain residual minus `g·C·u(t)`. A `g` of zero
  reproduces the plain arm, and `g` is clipped to 0 to 1.5.
- **Arms.** Plain (experiment 653's residual), solar (the arm above), and the placebo (each Primary
  gets another Primary's `C`, in a fixed cycle, with its own `g` refitted).
- **Decoder.** Experiment 653's, with its `params.json`, noise units, and slow level refitted on each
  arm's residual. The noise unit is recomputed per arm, and the page prints both.
- **Injected worlds.** The six worlds of experiment 653 with the same seeds, added to each arm's
  residual after the solar term is subtracted. The solar term does not touch the injected events.
- **Sunny-midday days.** A day is sunny-midday at a Primary when at least 70% of its half-hours from
  10:00 to 15:00 UTC have a clearness index (ERA5 shortwave divided by the Haurwitz clear-sky
  value) of at least 0.7. Decoded events are counted in the event-free stratum (rows more than 14
  days from any labelled or verified event), by whether the event's start or end falls on such a day.
  The denominator is the number of sunny-midday Primary-years, and likewise for other days.
- **Measures.** Events per Primary-year in each stratum; recall of injected events by duration band
  and size; the normalised MAE of the median forecast in the four scenarios of experiment 653 (all
  rows, quiet rows, inside events, the 7 days after an event), before and after subtracting the slow
  level and the decoded events, for all 15 Primaries and for the six; logged shifts matched at the
  strict and loose rules. Intervals resample whole worlds and whole months.
- **Time series.** The D5 and D16 pair of December 2025 to January 2026 (D5 and D16 are both
  megawatt Primaries), 14 days before to 7 days after, with the K1 day-ahead median, the plain
  residual, the solar-term residual, and the decoded events. D5 and D16 are anonymised Primary
  names, so no generator is involved.
- **Event-free stratum.** Printed per arm, beside the event rows.

## Known weaknesses, accepted

- **The demand model already sees irradiance.** The stored forecast comes from experiment 130's
  demand model, whose inputs include ERA5 and CAMS irradiance, so it may already remove most of
  the solar shape. The fitted `g` then shrinks towards zero and the arms differ little. That
  outcome is a result, not a failure.
- **The capacity table is for one summer half-year,** so it fixes the shape's size but not its
  seasonal trend, and it cannot follow a capacity change inside the record.
- **One irradiance product.** The term uses recorded ERA5 irradiance, so it is a hindsight product
  that cannot be made live. A forecast-weather version is out of scope (experiment 644 found that
  ensemble weather removed the gain of an explicit term).

## Files

Written when the run finishes.
