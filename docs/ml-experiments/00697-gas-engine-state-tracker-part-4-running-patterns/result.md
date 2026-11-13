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
status: done
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
verdict: supported
best_arm: >-
  Not applicable (no forecast is scored). D8's weekday-minus-weekend running share from 16:00 to
  19:00 is +0.392 (95% interval +0.323 to +0.464) in the net era and +0.256 (+0.156 to +0.365) in
  the solar era, and the difference from the six null controls is +0.202 or more in every case.
headline: >-
  The tracker's labels of D8's gas engine show a higher running share on weekday evenings than on
  weekend evenings, by more than a patternless injected engine produces. In the net era the
  difference is +0.392, and the smallest difference from a null control is +0.337. In the solar era
  the difference is +0.256, and the smallest difference from a null control is +0.202.
lessons: >-
  The weekday-evening pattern survives six null controls, but the controls cannot show that the
  tracker never reads D8's own evening demand as the engine. The running share also rises with the
  half-hourly price within a cell (Q2), a link that both D8 and the one metered engine record
  (Gas A) show. The tracker's share of one-half-hour spells is higher than the null controls'
  (Q3a), so spell lengths cannot be read as engine behaviour.
take_home_message: >-
  The recorded demand at D8 behaves as if its gas engine runs much more on weekday evenings
  (16:00 to 19:00) than on weekends, and more in dear half-hours, and a test engine with no weekday
  habit does not produce that. The evidence comes from the tracker's guesses and not from a meter
  on the engine, so the pattern is a lead for the next experiment, which builds a simple model of
  the engine's running, and not a result that a forecast can use yet.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Run a question about price against the host-series controls (C2), not the controls built on
    D8's own history, because D8's history before the engine ends before half-hourly prices start
    (Q2 and the controls section).
  - >-
    Read the tracker's labels at D8 as a test of a known engine only after injecting an engine with
    no weekday pattern into the same series and subtracting the control's difference (Q1, P1b).
  - >-
    Do not read spell lengths or the share of one-half-hour spells from the tracker's labels as
    engine behaviour, because the null controls show the same shapes (Q3a, figure 2).
  - >-
    Test any weekday-evening or price link in a plant model against the one metered engine record
    (Gas A) at 2 or more units, and against D8's tracker labels, before relying on it (Q1, Q2).
---

**This experiment asks whether the gas engine behind D8, one of the 16 Primary substations that this
research forecasts, runs to a schedule that follows the working week.** The engine is a gas engine
with a long-term capacity contract. No meter on the engine is available at D8, so we read the engine
from D8's half-hourly reading with the **tracker** of [experiment
694](../00694-gas-engine-state-tracker-part-1/result.md), a filter that infers from the reading
alone how many of the engine's units are running in each half-hour. We take **running share** to
mean the share of half-hours in a group that the tracker flags as running. **Q1** is the
weekday-minus-weekend running share from 16:00 to 19:00 local time. That window is the **red band**,
the hours in which NGED's Distribution Use of System charges for half-hourly metered customers in
the East Midlands, where D8 sits, are highest on Monday to Friday. No evidence of combined heat and
power was found.

**The tracker could produce a weekday pattern by itself, so we measured the pattern against
controls.** A **null control** is a real reading with an engine injected into it. The injected
output is experiment 694's method of adding a recorded engine to a reading, applied to days of the
one metered engine record we hold (called Gas A here, 44 days long), drawn at random with no regard
to the weekday. Any weekday pattern in the tracker's labels of a null control is tracker error. We
used three kinds of host reading: D8 before its engine, D8 before its engine with a constant
subtracted so that its weekday daytime level matches D8's level in the net era and in the solar era
(the **level-matched control**), and the readings of three other Primaries (D2, D3, and D4) over the
same two periods. The **net era** is 2023-09-01 to 2025-02-28 and the **solar era** is 2025-03-24 to
2026-02-28. D8's reading changes between them because 11 kV solar generation began. Six null
controls result, each run with 2 or 3 random seeds for the drawn days.

**The verdict is `supported`.** D8's Q1 is +0.392 (95% interval +0.323 to +0.464) in the net era and
+0.256 (+0.156 to +0.365) in the solar era. D is D8's Q1 minus a null control's Q1, with independent
resampling of whole calendar weeks for the two terms. Against all six null controls in both eras the
interval of D lies above zero and every point estimate is 0.10 or more. The smallest D is +0.337
(+0.260 to +0.416) in the net era, and +0.202 (+0.097 to +0.311) in the solar era, both against the
level-matched control for the solar era. The pre-registered gate is met. The second pre-registered
check, P1b, also passes: the null controls' labels show a Q1 of at most +0.054, below the limit of
0.10.

**Two limits keep the finding from being a result about the engine itself.** First, the shares come
from the tracker. D8's own demand is highest on weekday evenings, and the null controls cannot show
that the tracker never reads part of that demand as the engine. Second, the one metered engine
record (Gas A) agrees in direction only. At 2 or more units Gas A's weekday-minus-weekend share is
+0.233 (+0.057 to +0.442), but the record holds 7 weeks, 12 weekend days, and 3 of those weekend
days with the engine running, so its interval is a rough one.

## Questions beyond the verdict

**Q1 decides the verdict. Every other question is exploratory, was listed in the plan, and does not
change the verdict word.** About 30 such intervals are shown at 95% without a correction. A reading
of 1 means that D8's pattern excludes zero with one sign in both eras and excludes zero against
every null control, a reading of 2 means that a null control reproduces it, and a reading of 3 means
that D8's interval includes zero. Q5 is descriptive.

| Question | D8, net era | D8, solar era | Reading |
|---|---|---|---|
| Q1: weekday minus weekend, 16:00 to 19:00 | +0.392 (+0.323 to +0.464) | +0.256 (+0.156 to +0.365) | Supported (the verdict) |
| Q2: dearest price third minus cheapest third, within cell | +0.284 (+0.251 to +0.316) | +0.237 (+0.190 to +0.285) | 1, not explained by the controls |
| Q2q: the same with quarter cells | +0.303 (+0.273 to +0.334) | +0.250 (+0.190 to +0.302) | 1, not explained by the controls |
| Q3: hazard at 1 to 2 half-hours minus at 3 to 8 | +0.007 (-0.014 to +0.027) | +0.006 (-0.016 to +0.028) | 3, no pattern found |
| Q3a: hazard at 1 half-hour | +0.125 (+0.098 to +0.151) | +0.095 (+0.072 to +0.120) | 2, tracker error not excluded: the null controls show a larger value |
| Q6: winter-half minus summer-half weekday evening | +0.011 (-0.064 to +0.083) | not asked | 3, no pattern found |
| Q7: running day after a running day minus after an idle day | +0.133 (-0.032 to +0.294) | not estimable | 3, no pattern found |

**The running share rises with the half-hourly price within a cell, and the controls do not explain
it.** The price is the half-hourly system price (Q2), and a cell is one local half-hour in one
weekday class in one calendar month. Gas A shows the same direction at 1 or more units (+0.123,
+0.038 to +0.256) and at 2 or more units (+0.094, +0.019 to +0.212). Q2 and Q2q are compared with
the host controls only, because D8's history before the engine ends before half-hourly prices
start. The price peak, the red band, and D8's own demand peak all fall on weekday evenings, and this
experiment cannot separate them.

**Q5 is +0.074 (+0.031 to +0.123) and is descriptive.** Q5 is the share of half-hours between 00:00
and 05:00 that the tracker flags, in the 6 months after 2025-03-24 minus the same months a year
earlier. The host controls show -0.006 to -0.003 for the same contrast, so the rise is not a
property of the tracker on other Primaries. D8's night readings changed when its solar generation
began, and this experiment does not say whether the engine ran more at night.

## Figures

**Figure 1 shows the weekday evening block in D8's labels and not in the null control's.** Each
panel is a weekday by local half-hour heat map of the running share, for the summer half and the
winter half of each era. A tracker artefact could produce the evening block if the tracker reads
D8's weekday evening demand peak as the engine, which the null control, built on D8's reading
before the engine, makes less likely.

![The calendar heat map of the running share](f1_calendar_heat_map.svg)

**Figure 2 shows spell lengths that look alike in D8's labels, in Gas A, and in a control's labels,
so they say little about the engine.** The panels show the survival curve and the hazard (the share
of spells of length n that end at n) of on-spells and off-spells of 1 to 48 half-hours. A tracker
artefact could produce any shape, because the tracker's stay prior sets the hazard.

![Survival curves and hazards of spells](f2_dwell.svg)

**Figure 3 shows the running share rising with the price in D8's labels and in Gas A, and staying
flat in the null control.** The left panel uses price fifths over the whole period, and the right
panel uses price thirds within a cell. A tracker artefact could produce an overall rise, because the
price and D8's demand both peak on weekday evenings. The null control's flatness answers that only
for the demand shapes that the control contains.

![Running share against price](f3_price.svg)

**Figure 4 shows that Gas A runs at about 1, 2, or 3 times 1.97 MW and reaches its first full level
within two half-hours in 94% of the 54 spells.** Left: output in running half-hours, with rules at
multiples of 1.97 MW. Centre: output as a share of the level reached. Bottom: starts and stops by
local clock time. No panel for D8 is drawn, because the tracker places its edges where D8's reading
steps. The tracker's fitted unit count at D8 is a block size and not a count of units that D8's
reading could show.

![Gas A output levels and ramps](f4_gas_a_levels_and_ramps.svg)

Nine more exploratory figures (G1 to G6 and G8 to G10) are on [the gallery page](gallery.md). G7,
the night share by month, and the dated rasters stay in the private data folder.

## Checks

- **The null controls agree with the injected truth to a degree that varies by host.** The ratio of
  the labels' Q1 to the truth's Q1 in the weekday-class controls (the attenuation) is 0.745 at D8's
  history before the engine and 0.687 to 0.821 at D3 and D4, and 0.149 (net era) and 0.221 (solar
  era) at D2. The F1 score of the tracker against the injected truth is 0.54 to 0.73 at D2 and 0.62
  to 0.65 for the solar level-matched control, against 0.80 to 0.90 elsewhere. Every D above is
  computed against D2's null control too.
- **P1b:** the largest Q1 in the labels of the null controls built on D8's history is +0.054, below
  0.10.
- **The audit passed with 0 failed checks.** The audit asserts the row counts, the era lengths, the
  flagged shares of 0.268, 0.384, and 0.262, the 1.75, 2.25, and 3.5 MW unit sizes, and 0 flags of
  253,440 half-hours in the stored false-alarm pass (experiment 694; 44 segments at 8 Primaries).
  The stored host reproduction (experiment 695, host D4) gives an F1 score of 0.785 and a precision
  of 0.998.
- **Label agreement.** D8's smoothed labels agree with experiment 695's stored probabilities on
  0.999 of half-hours in the net era, 0.989 in the solar era, and 0.867 in the fade era. Smoothed
  and filtered labels agree on 0.958, 0.825, and 0.798.
- **The fade era is never read as a pattern.** D8's shortfall disappears in that era while flags
  persist: the smoothed labels flag 0.262 of half-hours and the filtered labels flag 0.150, and Q1
  is +0.383 on the smoothed labels and +0.030 (-0.051 to +0.105) on the filtered labels. The fade
  era carries no part of the verdict.
- **The timestamp shift is checked.** A power stamp marks the end of a half-hour and a price stamp
  marks the start, so every power-derived stamp is moved 30 minutes earlier before any join. The
  audit reproduces the five price-fifth shares of experiment 194 at the end-stamp join (0.089 to
  0.259) and gives the corrected shares at the start-stamp join (0.057 to 0.281). Every hour-of-day
  axis sits 30 minutes earlier than in experiment 194.
- **Bank holidays are weekdays in Q1.** The weekday class follows the local date, and the 2025-26
  charging statement states that its band includes bank holidays while the two earlier statements do
  not. Weekday evenings that are bank holidays are too few to read: 5 in each of the net and solar
  eras, with running shares of 0.533 and 0.867 against 0.841 and 0.806 on ordinary weekday evenings.
- **Median minus mean forecast and the audit triggers do not apply,** because no forecast is scored.
  No audit trigger from the experiment skill fired: no control failed, and no result contradicted
  its prediction.
- **Weather and signal coverage.** Half-hours missing a price are dropped and never filled. The
  Elexon within-day price index (the APX provider's half-hourly Market Index Data) is not the
  day-ahead auction, and 20 half-hours with price 0 and volume 0 are treated as missing. The
  controls built on D8's history before the engine carry no price, because that history ends in
  June 2023.

## What the next experiment should answer

**The next experiment should build equations for a gas engine's running and test them on Gas A and
on D8's labels.** The equations should include a minimum on-time and a ramp time, and ask whether a
simple rule reproduces the weekday-evening share and the rise with price. The experiment should add
a series with the same red band and no engine, to separate the band from the evening demand peak.

## Method

**Architecture and inputs:**

- The tracker is experiment 694's M2 grid filter on D8's apparent-power reading, with a demand level
  and a number of engine units running. The eras fit different unit sizes: 1.75 MW (net), 2.25 MW
  (solar), and 3.5 MW (fade).
- The label sets are Ts (the smoothed flag), Tf (the filtered flag), Ta (experiment 695's stored
  probabilities, as a mean probability), and Tc (the fixed-lag states, in the table of labels only).
- A half-hour start becomes `Europe/London` time, the weekday is that of the local date, and the six
  clock-change days are left out of hour-of-day plots.
- The drivers are the half-hourly system price, the daily gas price, the spark spread (the system
  price minus the gas price divided by 0.40), the day-ahead price from 2025-09, national wind, and
  local temperature, wind speed at 100 m, and irradiance (carried by grid index and never re-joined
  by time).

**Controls and injected truth:**

- A control injects Gas A's recorded days by experiment 694's folded rule at a half-hour grid offset
  of 40,944, with null truth (days drawn with no regard to the weekday) or weekday-class truth (days
  drawn within the weekday class).
- C1 is D8's reading before its engine (3 seeds, null and weekday-class truth, 6 runs). C1s is C1
  with a constant of 0.093 MVA (net) or 3.019 MVA (solar) subtracted (3 seeds, null truth, 6 runs).
  C2 is the hosts D2, D3, and D4 in the net and solar spans (2 seeds, null and weekday-class truth,
  24 runs). In all, 36 jobs run at 8 workers in about 200 seconds on CPU.
- C3 reads experiment 694's stored false-alarm pass. C4 reads experiment 695's stored host states.
- Every control seed differs: the seed is the draw seed plus the job seed plus a per-series and a
  per-control offset.

**Measurement:**

- Q1 is the weekday-minus-weekend running share from 16:00 to 19:00 local. The verdict reads Ts,
  with Tf and Ta shown beside it.
- Every interval resamples whole ISO weeks of local dates within an era (2,000 draws, percentile
  method). D uses independent resamples for D8 and the control.
- Bins with fewer than 20 distinct days are dropped, and the heat-map floor is 15.
- The verdict rule, written before the run, is tested in this order: not supported (D8's own Q1
  includes zero in both eras), supported (D above zero against every null control in both eras, with
  every point estimate at 0.10 or more), partly supported, and inconclusive (the default). P1b is
  the second check. No gain threshold applies, because no forecast is scored and the experiment
  changes no production code.
- Gas A is shown only as aggregates over the whole record: by local half-hour, weekday class, price
  third or fifth, spell length, and output level. D8's dated rasters, example weeks, and the night
  share by month stay in the private data folder.

**Signals added after the design (post hoc):**

- Ten open data tables from nine groups of signals were added after the design and show in G9 and
  G10: Elexon's within-day price index, NESO's Short Term Operating Reserve (STOR) windows and
  auction results (without unit or company names), Elexon's system warnings, the indicated imbalance
  forecast, NESO's day-ahead demand forecast, NESO's historic demand, Elexon's transmission wind,
  the national Carbon Intensity API, and the gov.uk bank holidays. Whether the engine holds a STOR
  agreement is unknown.
- A lookup of NESO's Enduring Auction Capability results asks only for the technology "Gas
  Reciprocating Engines" and filters by distance on this machine. It found one post code within 5
  km of D8's supply area, which does not confirm the engine's identity, and it stays private.

## Files

- `ml-experiments/00697-gas-engine-state-tracker-part-4-running-patterns/e00697_common.py`: helpers
  for the stamp shift, local time, the gas-day join, spells, hazards, week-block intervals, and the
  injected-day draw.
- `e00697_selftest.py`: 16 known-answer tests that read no stored data apart from one check of the
  level-matched constants.
- `e00697_audit.py`: the data audit, which writes `printed/audit.txt` and a private file of two
  checks of the record.
- `e00697_engine.py`, `e00697_controls.py`: the injected-day draw and the 36 control runs, each
  written to its own file in the data folder.
- `e00697_frame.py`, `e00697_drivers.py`: build one half-hour table per label set.
- `e00697_analysis.py`, `e00697_stats.py`: Q1 to Q8, the differences D, the tables T1 to T6, and
  `stats.json`.
- `e00697_verdict.py`: applies the decision rule and writes `printed/verdict.txt`.
- `e00697_figures.py`, `e00697_private_figures.py`: the published figures and the private dated
  ones.
- `e00697_signals_download.py`, `e00697_signals.py`, `e00697_lookups.py`: the extra signals, their
  panels, and the register lookup.
- `printed/`: the audit, the tables, the verdict, and the signal checks, as printed by the scripts.
- `data/ml-experiments/00697-gas-engine-state-tracker-part-4-running-patterns/`: private. It holds
  the 36 control files (`controls/`), the half-hour frames (`frames/`), `stats.json`, the downloaded
  signals (`signals/`), the lookup rows (`lookups/`), and the private figures.
- Read, from earlier work: experiment 694's tracker and stored false-alarm pass, and experiment
  695's stored probabilities and host states ([experiment
  695](../00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/result.md)).
