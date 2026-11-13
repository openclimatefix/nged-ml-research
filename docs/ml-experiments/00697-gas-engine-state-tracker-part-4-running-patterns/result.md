---
id: "00697"
slug: gas-engine-state-tracker-part-4-running-patterns
title: >-
  Does D8's gas engine run to a schedule that follows the working week, and what else regular can
  be read from the tracker's labels?
theme: gross-demand-simulation
also_relevant_to:
  - lsc-detectors
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  Anything else (no forecast, no new data, new analysis code). Stored: one private half-hour table
  per series, tracker labels for 36 control jobs, six tables, and private dated figures. More
  than one design is defensible (which labels, which control, which bins), so the choices are fixed
  below. Callers: this folder's scripts, which import experiment 694's trackers and loaders and
  experiment 695's states and engine module.
hypothesis: >-
  D8's gas engine runs to a schedule that follows the working week: it runs more on weekday
  evenings than on weekend evenings, by more than the tracker's errors alone would show.
prediction: >-
  (P1) In the tracker's labels at D8, the share of half-hours flagged as running from 16:00 to
  19:00 local time is higher on weekdays than on weekends. The same difference is computed on
  control series that carry an injected engine with no weekday pattern. D8's difference exceeds
  each control's difference by 0.10 or more, in the net era and in the solar era. (P1b) On D8's
  own readings from before the engine, with a patternless engine injected, the labels show a
  weekday-minus-weekend difference below 0.10. The recorded engine's own difference is not part
  of the prediction, because earlier experiments already show it.
decision_rule: >-
  D is D8's weekday-minus-weekend share minus a null control's, with a 95% interval from
  independent week-block resampling of the two terms. Tested in this order. Not supported: D8's
  own difference has an interval that includes zero in both eras. Supported: the interval of D
  lies above zero against every null control in both eras, and every point estimate of D is 0.10
  or more. Partly supported: the interval of D lies above zero against every null control in one
  era only, or in both eras with a point estimate under 0.10. Inconclusive: D8's own difference
  excludes zero but the interval of D includes zero against a null control in both eras, so
  tracker error is not excluded. No smallest worthwhile gain applies, because the experiment
  changes no production code. Every other question is exploratory and does not change the verdict word.
benchmark: none (no forecast is scored)
row_set_id: none
metric: running share of half-hours, differences of shares, spell lengths in half-hours
code: ml-experiments/00697-gas-engine-state-tracker-part-4-running-patterns/
---

**This file is a plan in the shape of a planned `result.md`, and nothing in it has run.** The
experiment asks whether the tracker's labels of D8's gas engine show a regular pattern, and whether
the pattern could be an error of the tracker. D8 is a Primary substation and may be named. "Gas A"
is the anonymised label of the one engine record we hold, and the page shows it only as
aggregates over the whole record.

## What earlier experiments left open

**Only profiles by hour and monthly totals of the tracker's labels have been printed, and no figure
shows them by weekday, season, price, or spell length.** [Experiment
694](../00694-gas-engine-state-tracker-part-1/result.md) built the tracker (a grid filter on the
demand level and the number of engine units running) and labelled D8's whole history. [Experiment
695](../00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/result.md)
stored the tracker's probabilities and its causal version. Neither experiment checked a pattern in
the labels against what the tracker would produce on a series with a known engine and no such
pattern.

## Four facts that shape the design

- **Power stamps mark the end of a half-hour and price stamps mark the start.** Every stamp that
  comes from a power series is moved to the start of its half-hour (`time` minus 30 minutes) before
  the price join, the gas-day join, and the conversion to local time. The authority is [the
  data-processing page](../data-processing.md). Every hour-of-day axis therefore sits 30 minutes
  earlier than in experiment 194.
- **The record of Gas A and the tracker's labels share almost no time.** The tracker's F1 score
  comes from injecting the record into other Primaries. The record is used as aggregates and as
  the source of injected days, and no pattern is passed on the record alone.
- **The tracker's unit count is a block size.** At D8 the tracker's "running" is closer to "two or
  more units of Gas A", so the record is read at "1 or more units" and at "2 or more units".
- **Gas A's unit count cannot be read at an edge.** The `units` column is the output divided by
  1.97 MW and rounded, and the reading behaves as a half-hour mean. No question is asked of unit
  counts, and the page says that the number of units running at D8 cannot be read from D8's reading.

## Label sets, eras, and time

- **R (record):** Gas A's stored 44-day record, read at "1 or more units" and "2 or more units"
  with levels taken from the output in MW.
- **Ts (smoothed):** experiment 694's M2 `flag`, eras `net` (2023-09-01 to 2025-02-28), `solar`
  (2025-03-24 to 2026-02-28), and `fade` (2026-03-01 to 2026-06-30). The days from 2025-03-01 to
  2025-03-23 are in no era and are dropped. **Tf (filtered):** the same file's M2 `filt`.
  **Ta:** experiment 695's stored probabilities, used as a mean probability that does not depend on
  the 0.5 threshold. **Tc:** experiment 695's fixed-lag states, in the table of labels only.
- **Eras are never pooled.** The net and solar eras carry the verdict. The fade era is shown and
  never supports a pattern, because D8's shortfall disappears in the fade era while flags persist.
- **Local time.** A half-hour start becomes `Europe/London` time. The axis is the local half-hour
  (0 to 47) and the weekday is that of the local date. The six clock-change days are left out of
  hour-of-day plots. A **cell** is one local half-hour in one weekday class in one calendar month.
- **Red band.** The red band of NGED's Distribution Use of System (DUoS) charges for half-hourly
  metered low-voltage and high-voltage customers in East Midlands is 16:00 to 19:00 local time,
  Monday to Friday, and weekends are never red. NGED's Statements of Use of System Charges for
  2023-24, 2024-25, and 2025-26 give the same hours. D8's substation is in the East Midlands licence
  area: D8's panel key is the `time_series_id` of a primary substation in the stored metadata
  (`NGED/metadata.parquet`), and that row's `licence_area` is `EMids`. The match of panel key to
  `time_series_id` is inferred from the key type and one matching row, and is not documented in the
  panel. The band is the evening block, local half-hours 32 to 37. The 2025-26 statement says
  "Monday to Friday (Including Bank Holidays)", and the 2023-24 and 2024-25 statements do not state
  whether bank holidays are charged as weekdays. T6 therefore splits weekday evenings into ordinary
  days and bank holidays as a descriptive check, and does not change Q1's weekday class. The
  statements are the source for the hours and for the wording on bank holidays.

## Controls

**A control is a real reading with a known engine injected by experiment 694's folded rule.** Days
of Gas A's record are drawn at random with replacement for every local day of the control period.

- **Null truth:** days are drawn ignoring the weekday, so the injected engine has no weekday
  pattern, no season, and no link to weather or price except through the half-hour of the day. Any
  such pattern in the labels of a null control is tracker error. The null control is what lets
  question Q1 fail.
- **Weekday-class truth:** days are drawn within the weekday class, as in experiment 695. The ratio
  of the labels' Q1 to the truth's Q1 gives the attenuation.
- **C1:** D8 before the engine, null and weekday-class truth, seeds 0 to 2 (6 runs). **C1s:** C1
  with one constant subtracted from the reading so that its weekday daytime level matches the net
  era's, null truth, net and solar levels, seeds 0 to 2 (6 runs). **C2:** hosts D2, D3, and D4, net
  and solar spans, null and weekday-class truth, seeds 0 and 1 (24 runs), which together make 36
  control jobs. **C3:** experiment 694's stored false-alarm pass, read and not rerun. **C4:**
  experiment 695's stored host states, read to reproduce its F1 score of 0.785 and precision of
  0.998.
- **What the controls cannot show:** no control has D8's 11 kV solar or the rise in noise at D8
  after 2025-03-24, and the injected engine has Gas A's mix of levels, not the block the tracker
  fits at D8. A reading of "not explained by tracker error" holds only for the demand shapes the
  controls contain.

## Figures, tables, and questions

- **Four figures on the page.** Figure 1 is a calendar heat map (local half-hour by weekday,
  running share). Figure 2 is dwell: survival curves and hazards of on-spells and off-spells.
  Figure 3 is running share against system price within cell. Figure 4 is Gas A's output levels and
  ramp shape. Each caption states
  which pattern a tracker artefact could produce.
- **Eight exploratory figures in a gallery** (G1 to G8): running days, start and stop clock times,
  daily price shape and the red band, spark spread and gas price, weather through the null control,
  holidays and clock-change days, night share by month, and season.
- **Six tables (T1 to T6)** and seven questions. Q1 (weekday share minus weekend share, 16:00 to
  19:00 local) decides the verdict. Q2, Q3, and Q5 to Q8 are exploratory, are read at 95%, and are
  reported with the note that about 30 such intervals are shown. Every interval resamples whole ISO
  weeks of local dates within an era (2,000 draws, percentile method). A spell belongs to the week
  in which it starts.
- **Reading of an exploratory question.** "Not explained by the controls" needs an interval that
  excludes zero with one sign in both eras and excludes zero against every null control. "Tracker
  error not excluded" means that a null control reproduces the pattern. "No pattern found" means
  that D8's interval includes zero in both eras. A result chosen after the tables are seen is
  labelled post hoc.

## Anonymisation

**Gas A appears only as aggregates over the whole record: by local half-hour, weekday class, price
third or fifth, spell length, and output level.** No figure or sentence shows a week, a day, or a
date of Gas A. No time series of D8's tracked engine state is published, and the dated day-by-day
raster and example weeks stay in the private data folder. Every driver plot uses bins that hold
20 or more days, and every mark sets `aria=False`.

## Signals added after the design

**Ten open data tables, from nine groups of signals, were added after the design was written and
were not pre-registered as separate questions.** The descriptive panels in the gallery (G9 and G10)
show them. No panel changes the Q1 verdict rule. The engine is described only as a gas engine with a
long-term capacity contract, and no evidence of combined heat and power was found.

- **Within-day price.** Elexon's Market Index Data from the APX provider only, from 2023-06. The
  index is the half-hourly price of the within-day market and is not the day-ahead auction. The
  half-hours with price 0 and volume 0 are treated as missing. The N2EX provider returns 0.00 in
  2023 and is not used. The held day-ahead price is NESO's N2EX hourly price, so the two series are
  compared where they overlap and are not expected to match.
- **Short Term Operating Reserve (STOR).** NESO's committed windows. Whether the engine holds a STOR
  agreement is unknown, so a match with a STOR window is a coincidence of timing and not a cause.
  The auction results are downloaded without unit or company names.
- **System stress.** Elexon's system warnings (margin, capacity market, and demand flexibility
  notices). There are only seven capacity market notices in 3 years, so the panel's intervals are
  wide.
- **Forecasts at the time of decision.** Elexon's indicated imbalance forecast (one vintage a day,
  published at about 09:45 UTC) and NESO's day-ahead demand forecast.
- **Demand and weather.** NESO's historic demand files for 2023 to 2026 (national demand, embedded
  wind, and embedded solar), Elexon's FUELHH transmission wind for the first year, and the
  national Carbon Intensity API.
- **Calendar.** The gov.uk list of bank holidays, checked against the `holidays` package.

Each panel shows the running share in the top third of the signal minus the bottom third, within a
cell, with the week-block interval, for D8's net and solar eras and for the null control. A signal
that the injected control engine cannot respond to should show no pattern in the control. A
lookup of the NESO Enduring Auction Capability (EAC) results asks only for the technology "Gas
Reciprocating Engines", counts rows per post code and service, and filters to post codes within 5
km of the supply area's centre on this machine, so NESO is never sent a location. It prints counts
only, and its rows stay in the private data folder. In the current file, one post code within 5 km
of the supply area's centre lists gas reciprocating engines. The match is on place and technology
only and does not confirm the identity of the engine.

## Standing practices

The experiment uses cleaned data, reads stored tracker states, and scores no forecast, so the
practices for forecasts do not apply. The calendar is the subject, so the "no calendar
memorisation" rule is departed from on purpose. No valid time on or after 2026-07-01 is read.
Compute is CPU only, 8 workers of one thread each, under 2 hours in all. The only downloads are
the small open tables of the section on signals added after the design, with no login and no
spend.

## Files

- `ml-experiments/00697-gas-engine-state-tracker-part-4-running-patterns/e00697_*.py`: the scripts
  that the finished write-up describes (common helpers, self-test, audit, engine draw, controls,
  frame, statistics, figures, private figures, verdict) and the extra-signal scripts
  (`e00697_signals_download.py`, `e00697_signals.py`, `e00697_lookups.py`).
