---
id: "00665"
slug: event-table-recall-by-length-and-depth
title: >-
  How much of an injected load-shift event does each event-table method find, split by how long the
  event lasts and how deep the step is?
theme: lsc-detectors
also_relevant_to: []
status: finished
date: 2026-11-09
kind: other
size: >-
  medium. (1) Stored: private score tables (JSON) and figures under
  data/ml-experiments/00665-event-table-recall-by-length-and-depth/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: the cut points between the six types
  are the only choice that is defensible either way, and they are fixed below before any score is
  read. (5) Callers: this folder's scripts, which read the stored outputs of experiments 653, 658,
  254, and 580 and import the folder of experiment 653 for the panel; nothing imports them. The
  maintainer's brief limits the work to CPU, 2 cores, and no new fit or network training.
hypothesis: >-
  The penalised segmentation of experiment 653 and the neural network of experiment 658 find most
  of the injected megawatts of events that are both long and deep, the kind a human operator would
  see at once, and the 0.64 and 0.52 headline recalls are low because they average in short and
  shallow events of which neither method finds more than a quarter of the megawatts.
prediction: >-
  Written before any score is read. P1a: in the long and deep cell (more than 7 days, 8 noise units
  or more), experiment 653's MW-weighted found share is 0.70 or more. P1b: experiment 658's second
  look finds 0.60 or more in the same cell. P2: for experiment 653, in each of the three lengths
  the deep cell finds a larger MW-weighted share than the shallow cell (direction only). P3: in both
  short cells (1 day or less), both methods find less than 0.25 (the stored 0.24 for events of 3
  hours to 1 day is close to this limit). P4: the correctly paired share of injected edge-transfer
  ends is 0.30 or less in every cell for both methods, including the long and deep cell. P5: on
  the real record, at least 70% of the events that experiment 653 calls per Primary-year fall in
  the three shallow cells, and fewer than 1 event per Primary-year falls in the long and deep cell.
decision_rule: >-
  Fixed before any score is read. The experiment re-scores stored outputs and fits nothing. Types:
  three lengths times two depths. Long is more than 7 days, and includes events with no end
  (stored as `days` of -1 and `end` of -1: permanent shifts and open events). Medium is more than
  1 day up to 7 days. Short is 1 day or less, down to the 3-hour shortest shift. Deep is 8 noise
  units or more. Shallow is 2 units up to but not including 8. A noise unit is a Primary's typical
  difference between successive 6-hour block means, as in experiment 653. The cut points change
  only with a stated reason on the page. Rows: injected events of kind edge, solo,
  permanent_transfer, or permanent_shift, of 2 noise units or more, in both signs, with every
  share weighted by the absolute megawatts, in experiment 653's default spec and masked arm and in
  experiment 658's second look (three seeds, the decoder setting each seed chose on validation
  worlds, arm `net`). Injected pulses (30 minutes to 2.5 hours) and the hard negatives fall in no
  cell. The pulses get a seventh row, reported apart, because experiment 653 never found a pulse
  shorter than its 6-hour block. Found is the stored `found` flag (right Primary and sign, start
  within 12 hours, and for an edge transfer the right donor with at least half the recipient
  megawatts matched). Paired is the stored `right_class` over the injected edge-transfer ends
  (`edge_truth` rows). Reproduction gate: on the rows above with 4 units or more and `days` of 1 or
  more or of -1, the re-scored figures must equal the stored headlines (experiment 653: 227 events,
  0.639 found, 0.179 paired; experiment 658 second look, mean of three seeds: 0.52 and 0.17,
  within 0.01). If a gate fails, stop and report `invalid`. A cell with fewer than 20 events or 10
  edge-transfer ends is too few to read, and a prediction that needs it counts as not holding. The
  verdict is `supported` if P1a, P1b, P2, and P5 hold; `partly supported` if three or more of the
  six predictions hold and the supported condition does not; and `not supported` otherwise. The
  porting threshold is zero, because the result is a research table and changes no production code.
benchmark: >-
  Experiment 653's default arm against experiment 658's second look, cell by cell. RO-NC and LR_b20
  (the label-free and label-trained detectors of experiment 254) are scored only on the real record.
row_set_id: >-
  experiment 653's injected worlds (6 worlds, default spec, scored years 2024-01-01 to
  2026-06-30) and its real record of 15 Primaries; not comparable with the leaderboard
metric: >-
  MW-weighted share of injected events found, and of injected edge-transfer ends paired correctly,
  per cell, with 95% intervals from resampling whole worlds (optimistic with 6 worlds, so the
  direction of a contrast is tested and no gap is called significant); start and end timing error;
  logged shifts matched and events called per Primary-year on the real record, per cell.
code: ml-experiments/00665-event-table-recall-by-length-and-depth/
verdict: partly supported
best_arm: >-
  experiment 653 default in the long and deep cell: 0.67 of injected megawatts found (95% interval
  0.60 to 0.76); experiment 658 second look 0.65 (0.56 to 0.74)
headline: >-
  Both methods find about two thirds of the megawatts of injected events that last more than 7 days
  and step 8 noise units or more, but only 0.31 to 0.50 when the step is shallower and almost none
  of the events of a day or less. Pairing stays at 0.18 or less for long events. On the real record
  the long and deep cell is almost empty: the table calls 1 event there in 2.5 years.
lessons: >-
  The headline recalls of 0.64 and 0.52 hide a split. Depth matters at every length (deep beats
  shallow by 0.1 to 0.3 for the table), and length matters more (events of a day or less are lost).
  The real record holds few events in the cell where both methods work, so the table's use lies in
  the shallow, long and medium events, where it finds about half of the injected megawatts.
take_home_message: >-
  Where the injected events are long (more than 7 days) and deep (8 noise units or more), the event
  table of experiment 653 finds 0.67 of the megawatts and the neural network of experiment 658
  finds 0.65, so the 0.64 and 0.52 headlines were pulled down by shallow and short events. Events of
  a day or less are almost never found (0.01 to 0.26), and no cell pairs the two ends of a
  transfer well (0.42 at best, from 14 ends). On the real record, none of the 77 logged shifts of
  more than 1 day falls in the long and deep cell, so the cell where the methods work is the one the
  record fills least.
advice:
  - >-
    Report recall for events of a day or less apart from longer events, because the table of
    experiment 653 finds 0.26 or less of their megawatts and the 0.64 headline mixes them in
    (recall by cell).
  - >-
    Read a pairing score per cell, over at least 20 edge-transfer ends, before drawing a conclusion,
    because the cells with 14 or 20 ends spanned 0.13 to 0.66 (paired share by cell).
production_status: not-for-production
adopted_in: none
---

**Where an injected load shift is both long and deep, the event table and the neural network find
about two thirds of its megawatts; when it is shallow or lasts a day or less, they find much
less.** A Primary substation is the substation that steps electricity down to a local distribution
network, and the project forecasts the demand at each Primary. Operators sometimes switch part of a
circuit from one Primary to a neighbour, which moves demand between the two (a load shift).
[Experiment 653](../00653-edge-state-event-table-penalised-segmentation/result.md) built a table of
load shifts with a penalised segmentation of the residual (the observed cleaned power minus a demand
forecast that uses no lagged power), and
[experiment 658](../00658-edge-state-event-table-neural-network/result.md) tried a recurrent neural
network. Both reported one recall figure (0.64 and 0.52 of the injected megawatts). We re-scored
their stored outputs on 6 types of injected event: 3 lengths times 2 depths. No model was fitted and
no network was trained.

In the long and deep cell (more than 7 days, 8 noise units or more) the table found 0.67 of the
injected megawatts (95% interval 0.60 to 0.76) and the network 0.65 (0.56 to 0.74). Neither method
paired the two ends of a transfer well in any cell: 0.42 was the best, from only 14 ends. On the
real record the long and deep cell is almost empty. The verdict is *partly supported*: 3 of the 6
predictions held (P1b, P2, and P5), and P1a, P3, and P4 failed.

## Figures

![Injected megawatts found and edge-transfer ends paired, in the six types](figure_recall_by_cell.svg)

![D5 and D16, 15 December 2025 to 21 January 2026: power and decoded events](figure_event_series_D5_D16.svg)

![Events called in event-free time, per Primary-year, in each cell and for each method](figure_event_free_stratum.svg)

## Results

- **Terms.** A *noise unit* is a Primary's typical difference between successive 6-hour block means
  (0.16 to 1.02 MW across the 15 Primaries). *Deep* is 8 noise units or more and *shallow* is 2 to
  under 8. *Long* is more than 7 days (including events with no end), *medium* is more than 1 day up
  to 7 days, and *short* is 1 day or less. A *world* is one copy of the residual panel with
  synthetic events added; there are 6. *Found* and *paired* are defined in the decision rule.
  *LR_b20* and *RO-NC* are the label-trained and label-free detectors of
  [experiment 254](../00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/result.md).
  *Seed* is one random start of network training; there are 3.
- **Reproduction gates passed.** Experiment 653: 227 events, 0.639 found, 0.179 paired. Experiment
  658 second look, mean of 3 seeds: 0.521 and 0.172. The 77 logged shifts: 8 strict and 17 loose
  matches for the table, and 19 and 24 for LR_b20 (the stored values).
- **Injected events, MW-weighted found share (95% interval from resampling the 6 worlds, which is
  optimistic; n is events, with injected edge-transfer ends in brackets).**

    | Cell | n | Experiment 653 | Experiment 658 second look |
    |---|---|---|---|
    | Long, deep | 51 (24) | 0.67 (0.60 to 0.76) | 0.65 (0.56 to 0.74) |
    | Long, shallow | 193 (96) | 0.50 (0.44 to 0.55) | 0.31 (0.26 to 0.36) |
    | Medium, deep | 27 (14) | 0.65 (0.43 to 0.84) | 0.61 (0.41 to 0.77) |
    | Medium, shallow | 121 (59) | 0.52 (0.40 to 0.65) | 0.32 (0.21 to 0.44) |
    | Short, deep | 37 (20) | 0.26 (0.15 to 0.33) | 0.06 (0.01 to 0.09) |
    | Short, shallow | 106 (42) | 0.16 (0.04 to 0.28) | 0.01 (0.00 to 0.02) |

    Unweighted, the long and deep cell is 0.55 (table) and 0.57 (network). The three seeds of the
    network give 0.72, 0.66, and 0.56 in that cell. No cell has fewer than 20 events or 10 ends, so
    none is marked too few to read, but the medium and deep cell (27 events, 14 ends) has intervals
    about 0.4 wide.
- **Paired share of injected edge-transfer ends (table; network).** Long, deep 0.13 (0.00 to 0.34);
  0.18 (0.03 to 0.35). Long, shallow 0.15; 0.08. Medium, deep 0.42 (0.13 to 0.66); 0.38 (0.08 to
  0.57). Medium, shallow 0.12; 0.05. Short, deep 0.03; 0.00. Short, shallow 0.04; 0.00. P4 (0.30 or
  less in every cell) fails in the medium and deep cell, from 14 ends.
- **Timing, long and deep cell (a miss counts as a failure).** The start is within 2 hours for 0.55
  (table) and 0.54 (network) of events, and the end within 6 hours for 0.52 for both. The median
  error among found events is 0.0 to 0.5 hours for both. In the long and shallow cell the network's
  median end error is 7.3 hours against the table's 1.0, and its end is within 6 hours for 0.16
  against 0.29. The median end error excludes 13 found events whose end error is missing, so it is
  optimistic.
- **Pulses.** Injected pulses of 30 minutes to 2.5 hours (221 events of 2 noise units or more) were
  found at 0.00 of the megawatts by both methods.
- **Predictions.** P1a fails: 0.67 against 0.70 or more, with an interval that reaches 0.76. P1b
  holds: 0.65 against 0.60 or more. P2 holds in direction for the table at all 3 lengths (0.67
  against 0.50, 0.65 against 0.52, 0.26 against 0.16), though the intervals of the medium cells
  overlap. P3 fails: the table finds 0.26 in the short and deep cell, against less than 0.25. P4
  fails (above). P5 holds: of the table's 62 real events in a cell (by its stored units), 56 (90%)
  are in a shallow cell, and 1 is in the long and deep cell.
- **Real record, events per Primary-year in each cell (scored 2024-01-01 to 2026-06-30; 36.0
  Primary-years with a reading; the cell uses each method's own stored units, and for LR_b20 and
  RO-NC the depth measured in the residual panel).**

    | Method | Long, deep | Long, shallow | Medium, deep | Medium, shallow | Short, deep | Short, shallow |
    |---|---|---|---|---|---|---|
    | Experiment 653 (246 events) | 0.03 | 0.56 | 0.11 | 0.83 | 0.03 | 0.11 |
    | Experiment 658 second look, seed 0 (18) | 0.00 | 0.27 | 0.05 | 0.08 | 0.00 | 0.00 |
    | LR_b20 (78 alarms) | 0.08 | 0.24 | 0.24 | 0.67 | 0.05 | 0.05 |
    | RO-NC (189 alarms) | 0.08 | 0.91 | 0.24 | 1.82 | 0.11 | 0.13 |

    Seeds 1 and 2 of the network give 0.19 and 0.32 long, shallow events per Primary-year and put
    0.13 and 0.08 in the medium and deep cell. Events or alarms under 2 noise units fall in no
    cell: 181 of the table's 246 events have a measured depth under 2 units, so the cell counts of
    the table depend on the depth measure (with the residual-panel measure the table has 3, 22, 4,
    35, 0, and 1 events in the six cells, in the order above). Alarms and table events are different
    objects, so these rates are not like for like.
- **Logged shifts of more than 1 day in the scored years (77).** By the residual-panel depth: 0
  long and deep, 7 long and shallow, 0 short, 7 medium and deep, 19 medium and shallow, 43 under 2
  noise units, and 1 unmeasured. Matched strictly (start within 2 hours), the table found 1 of 7
  long and shallow, 3 of 7 medium and deep, and 2 of 19 medium and shallow; the three network seeds
  found 2, 0, and 2 of 7, then 2, 4, and 3 of 7, then 0 of 19 in each; LR_b20 found 3, 6, and 7; RO-NC
  found 3, 6, and 5. LR_b20 and RO-NC match more logged medium shifts than either method here.
  LR_b20 was trained on the labels.
- **Event-free time (27.6 Primary-years).** Events called per Primary-year in the long and deep
  cell: 0.04 for the table, 0.00 for the network, 0.00 for LR_b20 and RO-NC. In the medium and
  shallow cell: 0.69 for the table, 0.01 to 0.07 for the network, 0.43 for LR_b20, and 1.45 for
  RO-NC. NGED's logs miss events, so these are upper bounds on false events.
- **Verified pair (D5, D16).** Both methods decode D5 as a step of -1.1 MW from 30 December 2025
  to 14 January 2026, 24 hours late at both ends, and D16 as +2.0 to +2.1 MW to 14 January 2026.
  The table starts D16 2.6 days early and the network 9.5 hours late. Neither pairs them. The
  event is long (15 days) and, at 6.6 and 7.3 noise units, just under the deep cut of 8.

## Checks

- **Median minus mean forecast, tail calibration, curtailment cost, normalised MAE of the median,
  and the four scenarios:** not applicable, because no forecast is scored.
- **Audit triggers.** Trigger 1 (a control fails): the reproduction gates passed. Trigger 2 (a
  result contradicts its prediction) fired for P1a, P3, and P4. P1a misses by 0.03 with an interval
  that reaches 0.76, P3 misses by 0.01, and P4's miss rests on 14 edge-transfer ends with an
  interval of 0.13 to 0.66, so the explanation is small cells and not a method error. Triggers 3 to
  7 do not apply (no forecast contrast, no nested arm, no sister experiment on the same rows, and
  no later design resting on this result). The numbers review re-derived every figure from the
  stored files.
- **Unlogged events.** NGED's event logs are an incomplete one-off dump, so clean rows may hold
  events no label covers. Matched-shift counts are lower bounds, and the event-free rates are upper
  bounds on false events.
- **Lab shortcuts.** The injected events are added to the stored residual and not to the power
  before the demand model is refitted, so injected recall is a ceiling for real events. The demand
  model trained on every row, so earlier real events leak into the residual's baseline.

## Reviews and triage

- **Plan review (Sonnet).** 15 findings, 4 of them high: the sign and weight of the injected
  shares, the coding of permanent events, the reproduction gate's row filter, and the exclusion of
  pulses. All were accepted and written into the plan before the scoring. The verdict rule, the
  pairing wording, the intervals, the event-free definition, and the prose findings were accepted
  too.
- **Code review (Sonnet).** 8 findings and no bug that changed an injected-world number. It ran
  after the first scoring run, not before it, which the plan said would be the order. Accepted and
  fixed: the per-Primary-year denominator now counts only half-hours with a reading (36.0
  Primary-years, where the nominal figure is 37.5, so the real-record rates rose by about 4%); the
  rows in no cell are split into under 2 noise units and unmeasured depth; the depth of the table's
  events is also measured in the residual panel and shown beside the stored units; the event-free
  spans are checked to end after they start. Accepted as page text: the missing end errors and
  the first-seed counts.
- **Numbers review and prose review (Sonnet):** see the end of this section after they ran.

## What the next experiment should answer

**The next experiment should ask whether a residual with more history, or a method that reads the
shallow events, can lift the 0.50 and 0.31 found in the long and shallow cell.** Both methods
already find about two thirds of the long and deep megawatts, but the real record holds few such
events, so the gain from the cell where they work is small. The hardest and most common real events
are shallow and medium, and the logged shifts there are matched better by LR_b20 and RO-NC than by
the table or the network.

## Method

**Architecture and algorithm.**

- **Experiment 653's table (arm `masked`, spec `default`).** The input is the observed cleaned power
  minus the forecast of the lag-free convex demand model of
  [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), as stored by
  [experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md), for 15 Primaries
  on a half-hourly grid from 2022-01-01 to 2026-06-30, with the median across Primaries removed.
  A seasonal slow level per Primary is fitted with candidate events masked and then frozen. An
  exact dynamic programme ([Viterbi, 1967](https://doi.org/10.1109/TIT.1967.1054010)) decodes
  6-hour blocks into 25 states (no shift, and 12 log-spaced amplitudes of 1 to 20 noise units in
  each sign) with a Student-t emission. Events are paired across neighbouring Primaries by sign,
  start, end, and share. Experiment 653's page has the full description.
- **Experiment 658's network (second look, arm `net`).** A bidirectional recurrent neural network
  reads all 15 Primaries' residuals and the neighbour links. It has 32 hidden units, dropout 0.3,
  weight decay 0.01, and Gaussian noise on the inputs. It is trained on experiment 653's injector
  over 300 worlds of 2022-01-01 to 2023-06-30, with a decoder tuned on validation worlds of 2023
  (boost 10, cut-off 0.5, on cost 0.5). Experiment 658's page has the full description.
- **This experiment.** No fit. It assigns each stored injected event, logged shift, event, and
  alarm to a cell by its length and depth, and recomputes the scores per cell.

**Training regime.**

- **None here.** Experiments 653 and 658 trained nothing in the first case and 3 seeds of the
  network in the second; this experiment uses their stored outputs. It ran on 1 CPU core in a few
  minutes, with no GPU.

**Measurement.**

- **Injected events.** The 6 worlds, default spec, scored years 2024-01-01 to 2026-06-30, of
  experiment 653's injector. Rows are events of kind edge, solo, permanent_transfer, or
  permanent_shift of 2 noise units or more, in both signs, weighted by absolute megawatts.
  Hard negatives are excluded and pulses are reported apart.
- **Cells.** Length from the injected `days` (-1 means no end, counted long); depth from the
  injected `units`. The cut points are 1 day, 7 days, and 8 noise units.
- **Found, paired, timing.** As in the decision rule of this page. Intervals are 2000 resamples of
  whole worlds, with the 3 seeds averaged in each draw.
- **Real record.** The table's and network's stored events of the scored years, with the
  cell from stored units. LR_b20 and RO-NC alarms of experiment 254 (duplicates removed) and the
  logged shifts of experiment 580 get their depth from the residual panel: the absolute difference
  between the median residual over 3 days after the onset (or to the end if sooner) and over the 3
  days before, in noise units, needing 48 readings in each window. Logged shifts are matched when a
  method's start lies within 4 half-hours (strict) or 48 half-hours (loose) at the same Primary.
- **Event-free time.** Half-hours with a reading that lie more than 14 days from every labelled or
  verified event of experiment 580, per Primary. An event counts when none of its half-hours lies
  in the excluded time.
- **Standing practices.** Cleaned data only; D15 and D18 lose every row before 2025-03-01; no row
  from 2026-07-01 onward is read (the script asserts it).

## Files

- `ml-experiments/00665-event-table-recall-by-length-and-depth/e00665_score.py`: stage `injected`
  scores the 6 cells for both methods, runs the reproduction gates, and writes
  `injected_by_cell.json`; stage `real` scores the real record and the event-free time and writes
  `real_by_cell.json`.
- `ml-experiments/00665-event-table-recall-by-length-and-depth/e00665_figures.py`: draws the three
  figures from the two JSON files, the residual panel, and the stored events.
- Imported code, read only: the folder of
  [experiment 653](../00653-edge-state-event-table-penalised-segmentation/result.md)
  (`e00653_common.py` for the panel and grid, `e00653_inject.py` for the 2019 offset).
- `data/ml-experiments/00665-event-table-recall-by-length-and-depth/`: `injected_by_cell.json` (per
  cell and method: counts, shares, intervals, timing; plus the pulse row) and `real_by_cell.json`
  (events per Primary-year, logged shifts matched, event-free rates), written here, never committed.
- `data/ml-experiments/00653-edge-state-event-table-penalised-segmentation/`:
  `injected_truth_scored.parquet` (one row per injected event and arm, with `found`, `right_class`,
  `start_err_hh`, `end_err_hh`, `units`, `days`), `real_events_masked.parquet`, and
  `noise_units.json` (read only).
- `data/ml-experiments/00658-edge-state-event-table-neural-network/`:
  `injected_v2main_seed{0,1,2}.parquet`, `costs_v2main_seed{0,1,2}.json`, and
  `real_events_v2main_seed{0,1,2}.parquet` (read only).
- `data/ml-experiments/00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/`:
  `alarms_LR_b20.parquet` and `alarms_RO_b80.parquet` (onset, end, series; read only).
- `data/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/work/`:
  `events_labelled_status.parquet` and `events_verified.parquet` (read only).
- `data/ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/`: stored power and
  forecast arrays behind the residual panel (read only).
