---
id: "00665"
slug: event-table-recall-by-length-and-depth
title: >-
  How much of an injected load-shift event does each event-table method find, split by how long the
  event lasts and how deep the step is?
theme: lsc-detectors
also_relevant_to: []
status: planned
date: 2026-11-09
kind: other
size: >-
  medium. (1) Stored: private score tables and figures under
  data/ml-experiments/00665-event-table-recall-by-length-and-depth/ only. (2) Serving path: not
  touched. (3) Degradation rule: not touched. (4) Designs: the cut points between the six types
  are the only defensible-either-way choice, and they are fixed below before any score is read.
  (5) Callers: this folder's scripts, which read the stored outputs of experiments 653, 658, 254,
  and 580 and import the folder of experiment 653 for the panel; nothing imports them. The
  maintainer's brief limits the work to CPU, 2 cores, and no new fit or network training.
hypothesis: >-
  The penalised segmentation of experiment 653 and the neural network of experiment 658 find most
  of the injected megawatts of events that are both long and deep, the kind a human operator would
  see at once, and the 0.64 and 0.52 headline recalls are low because they average in short and
  shallow events that neither method can see.
prediction: >-
  Written before any score is read. P1: in the long and deep cell (more than 7 days, 8 noise units
  or more), experiment 653's MW-weighted found share is 0.70 or more, and experiment 658's second
  look is 0.60 or more. P2: in each of the three lengths, the deep cell finds a larger MW-weighted
  share than the shallow cell, for experiment 653. P3: in both short cells (1 day or less), both
  methods find 0.25 or less. P4: the correctly paired share of injected edge transfers is 0.30 or
  less in every cell for both methods, including the long and deep cell. P5: on the real record,
  at least 70% of the events that experiment 653 calls per Primary-year fall in the three shallow
  cells, and fewer than 1 event per Primary-year falls in the long and deep cell.
decision_rule: >-
  Fixed before any score is read. The experiment is a re-scoring of stored outputs and fits
  nothing. Types: three lengths times two depths. Long is more than 7 days, and includes events
  with no end (permanent shifts and open events). Medium is more than 1 day up to 7 days. Short is
  1 day or less (the injected events last 3 hours or more). Deep is 8 noise units or more.
  Shallow is 2 units up to but not including 8. A noise unit is a Primary's typical difference
  between successive 6-hour block means, as in experiment 653. The cut points change only with a
  stated reason on the page. Rows: injected positive events (kinds edge, solo, permanent_transfer,
  permanent_shift) of 2 noise units or more in experiment 653's default spec and masked arm, and
  in experiment 658's second look (three seeds, the costs each seed chose on validation worlds,
  arm net). Reproduction gate: the re-scored all-cell figures must equal the stored headlines
  (653: 0.639 found and 0.179 paired on events of 4 units or more lasting a day or more; 658
  second look: 0.52 and 0.17). If a gate fails, stop and report `invalid`. A cell with fewer than
  20 injected events or fewer than 10 injected edge transfers is marked too few to read, and no
  contrast is drawn from it. The verdict is `supported` if P1, P2, and P5 hold and at least one
  of P3 and P4 holds, `partly supported` if three or four of the five predictions hold, and `not
  supported` otherwise. The porting threshold is zero, because the result is a research table
  and changes no production code.
benchmark: >-
  Experiment 653's default arm against experiment 658's second look, cell by cell. RO-NC (the
  RO_b80 alarms of experiment 254) and LR_b20 are scored only where their stored outputs allow,
  which is the real record.
row_set_id: >-
  experiment 653's injected worlds (6 worlds, default spec, scored years 2024-01-01 to
  2026-06-30) and its real record of 15 Primaries; not comparable with the leaderboard
metric: >-
  MW-weighted share of injected events found, and of injected edge transfers paired correctly, per
  cell, with 95% intervals from resampling whole worlds; start and end timing error; logged
  shifts matched and events called per Primary-year on the real record, per cell.
code: ml-experiments/00665-event-table-recall-by-length-and-depth/
---
**This page is the plan, written before any score is read.** The question: the headline recalls of
experiments 653 and 658 (0.64 and 0.52 of injected megawatts) mix events that a human operator
would see at once with events no method can see. We re-score the stored injected-world outputs in
six types (three lengths times two depths) to find out where each method works and where it does
not. No model is fitted and no network is trained.

## Plan

- **Injected worlds.** Read `injected_truth_scored.parquet` of experiment 653 (default spec, masked
  arm) and `injected_v2main_seed{0,1,2}.parquet` of experiment 658 (arm `net`, the costs in
  `costs_v2main_seed*.json`). Both score the same 6 worlds with the same truth table.
- **Per cell and method:** the number of injected events; MW-weighted found share (an event found
  at the right Primary and sign with its start within 12 hours, and for an edge transfer the right
  donor); the correctly paired share, reported separately and only over injected edge transfers;
  the share with the start within 2 hours and the end within 6 hours, a miss counting as a
  failure; the median absolute timing error among found events.
- **Intervals.** 95% intervals from 2000 resamples of whole worlds. For experiment 658 each draw
  averages the three seeds.
- **Real record, scored years.** Events called per Primary-year per cell by experiment 653 and by
  each seed of experiment 658 (real events carry their noise units and ends). Logged shifts of more
  than 1 day matched, strictly (start within 4 half-hours) and loosely (48), per cell. A logged
  shift's length is its logged length. Its depth is not stored, so it is measured in the residual
  panel: the absolute difference between the residual median over the first 3 days of the shift
  (or the whole shift if shorter) and over the 3 days before the onset, in noise units. RO-NC and
  LR_b20 are scored the same way from their stored alarms (their depth is measured in the same
  way, and they cannot be scored on the injected worlds because no injected-world alarms are
  stored).
- **Event-free stratum.** Decoded events more than 14 days from every labelled or verified event,
  per method and per cell.
- **Figures.** The verified D5 and D16 transfer, 14 days before the start to 7 days after the end,
  with the decoded events of each method; the event-free stratum per method; the recall and paired
  share per cell with intervals (two panels, at least 280 px each).
- **Standing practices.** Cleaned data only (the stored panel flags `substation_zero`; D15 and D18
  lose every row before 2025-03-01, as in experiment 653); nothing from the final-test window
  (2026-07-01 onward) is read; no generator series is drawn, so there is nothing to anonymise
  beyond the Primaries' names, which are substations.
- **Process.** One Sonnet plan or code review before the scoring script runs, one Sonnet review of
  the numbers, and one Sonnet prose review. The experiment is a CPU re-score of a few minutes on 1
  to 2 cores, so the compute estimate is under 0.1 core-hours.
- **Reuse of earlier work.** Reads the stored tables of experiments 653, 658, 254, and 580, and
  imports the panel loader of experiment 653.
