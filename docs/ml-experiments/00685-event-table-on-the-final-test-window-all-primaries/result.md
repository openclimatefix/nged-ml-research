---
id: "00685"
slug: event-table-on-the-final-test-window-all-primaries
title: >-
  Which load-shift candidates does the fixed event table find at the 15 Primaries from 2026-07-01
  to 2026-10-31?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: finished
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private tables under
  data/ml-experiments/00685-event-table-on-the-final-test-window-all-primaries/
  only. (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: none, the
  method is fixed as experiment 680's. (5) Callers: this folder's scripts, which import the folders
  of experiments 130, 653, 670, and 680; nothing imports them. CPU only, 2 cores.
hypothesis: >-
  None. This is a data product: the event table of experiment 680, run unchanged on the residual
  panel extended to 2026-10-31, so that the one-year-backtest flags at four Primaries can be set
  beside the events the table can see.
prediction: >-
  None. No detector is tuned, no method is chosen, and no model is scored on the window.
decision_rule: >-
  None. Gate: the 2026-06-01 refit cut must reproduce experiment 327's stored forecast, and the
  pipeline with the window blanked must reproduce experiment 680's stored real-record events.
benchmark: >-
  Experiment 680's stored real-record event table (E1-J-sym), 249 events starting 2024-01-01 to
  2026-06-30.
row_set_id: >-
  experiment 653's real record of 15 Primaries extended from 2026-06-30 to 2026-10-31; not
  comparable with the leaderboard
metric: >-
  Candidate events per Primary and per month; depth in megawatts and in noise units; events that
  match experiment 680's table.
code: ml-experiments/00685-event-table-on-the-final-test-window-all-primaries/
verdict: inconclusive
best_arm: >-
  Not applicable: one method, no arms.
headline: >-
  The event table finds 13 candidate events (14 rows) starting from 2026-07-01 and none of the
  transfers
  between time_series_id 13 and 16 that would explain the one-year-backtest flags in July and
  August, so the flags at
  those Primaries are not explained by an event the table can see.
lessons: >-
  A level offset that persists for months, such as the flagged id 13 and 16 pattern, does not
  appear as an event in this table. Adding 4 months of data changes 62 of 249 earlier events,
  mostly small solo shifts, so the table changes when the panel is extended.
take_home_message: >-
  From 2026-07-01 to 2026-10-31 the event table finds 13 candidate events (14 rows) at the 15
  Primaries and no transfer between time_series_id 13 and 16 in July or August, nothing at id 11 in
  September,
  and one open fall at id 18 from 2026-10-04. The one-year-backtest flags at ids 13, 16, 11, and 18
  are therefore not explained by events the table can see. A level offset in the lag-free forecast
  remains a candidate explanation that this table does not test. One transfer from id 13 to id 16
  opens on 2026-10-26 and is still open at the
  end of the window. Every row is a candidate, never a label.
advice:
  - >-
    Treat the event table's rows from before 2026-07-01 as provisional when the panel is extended,
    because 62 of 249 earlier events change (see Reproduction gate).
  - >-
    Look for a level offset, not an event, when a Primary is outside its forecast band for months
    and the table shows nothing there (see Headline rows).
production_status: not-for-production
adopted_in: none
---

**From 2026-07-01 to 2026-10-31 the event table of experiment 680, run unchanged on 15 Primary
substations, finds 14 rows (13 events, because an edge transfer takes two rows).** None of them
explains the one-year backtest's flags at four Primaries. A *Primary substation* steps electricity
down to a local distribution network. A *load shift* is a lasting change in a Primary's demand with
no change in the weather, and an *edge transfer* is a load shift that moves between two neighbouring
Primaries. The one-year backtest flagged `time_series_id` 13 (D18) as above its 95th-percentile
forecast on 69% and 73% of half-hours, and 16 (D14) as below its 5th percentile on 75% and 78%.
Flags at id 11 (D4) in September and id 18 (D16) in October
were also reported.

We ran the event table on the *residual*, which is the observed power minus the forecast of a convex
demand model that uses no lagged power ("lag-free"), extended to the end of 2026-10-31. The table
finds no edge transfer between ids 13 and 16 in July or August, and
no event at id 11 in September.

**Every row is a candidate, never a label.** NGED's event labels are an incomplete dump, so a clean
half-hour may hold an event that no label covers, and an absence here is not proof that nothing
happened. **The final-test window 2026-07-01 to 2026-10-31 was read for this event table only, on
the maintainer's authorisation.** No detector was tuned on it, no method was chosen with it, and no
forecasting model was scored on it. *Noise unit* has the meaning given in Method.

## Headline rows

| Question | Result |
|---|---|
| Transfer between id 13 (D18) and id 16 (D14), July and August 2026 | None found. |
| Any event at id 11 (D4), September 2026 | None found. |
| Any event at id 18 (D16), October 2026 | One solo fall of 0.75 MW (2.7 noise units) from 2026-10-04 15:30 UTC, open at 2026-10-31. |
| Open edge transfer from id 13 to id 16 | From 2026-10-26 17:30 UTC D18 falls 1.31 MW (8.2 noise units) and D14 rises 0.71 MW (2.2 noise units); both are open at 2026-10-31. |

The 14 rows starting in the window are 12 solo shifts (5 in July, 3 in September, 4 in October) and
the two rows of the one edge transfer (the per-Primary and per-month counts are in the private
`SUMMARY.md`).

## Reproduction gate

- **Refit cut.** The 2026-06-01 cut of the lag-free forecast refitted here matches
  [experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md)'s stored forecast
  with a maximum difference of 0 MW at all 15 series.
- **Window blanked.** With the residual after 2026-06-30 set to missing, the pipeline reproduces
  experiment 680's 249 stored real-record events exactly (key, end, depth, and class).
- **Window included: 62 of 249 earlier events change.** Of the 249 earlier events, 187 keep
  their Primary, start, class, and partners. The other 62 are replaced by 51 new ones, mostly small
  solo shifts spread over every Primary. The monthly refits and the median removal are ruled out as
  the cause. The slow level is refitted on the whole panel, so 4 more months of data move the
  baseline and the end-of-record handling; no run separated those two, so that cause is an inference
  by
  elimination. Among the 56 earlier events of 2 noise units or more that are neither ramps nor
  routine
  steps, 47 keep their key and move by a median of 0.05 noise units (largest 1.1), and none of the
  47 has a changed end (22 of the 187 keyed matches in the full set do).
  The other 9 are dropped and 21 are added, mostly near the 2-unit threshold. Six of the added
  events
  are 4 noise units or more. Treat events from before July as possibly changed by the extension.

## Checks

- **Weather at forecast time.** The lag-free forecast uses ERA5 2 m temperature and CAMS irradiance,
  both estimates of past weather. The table is a hindsight product and cannot be made live.
- **Last half-hour.** The forecast for 2026-10-31 23:30 is missing because ERA5 ends at 23:00 on
  that
  day. Hours after 2026-06-30 are ERA5T in ARCO-ERA5, whose values are preliminary and may be
  revised.
- **Window power.** Each Primary has 5,784 to 5,904 of 5,904 half-hours with a cleaned reading in
  the
  window. Half-hours flagged by the `substation_zero` cleaning rule are missing.
- **Code review.** One fresh Sonnet code review ran before the run. It found a blocking indexing
  error, a dependence of the noise units on the series length, and a wrong `dark` flag on half-hour
  rows (a flag marking half-hours without daylight); all three were fixed. Every refit cut fitted.

## Method

- **Unchanged from experiment 680.** E1-J-sym is [experiment
  653](../00653-edge-state-event-table-penalised-segmentation/result.md)'s event table plus two
  searches, with no parameter changed.
- **Input.** The residual is the cleaned power minus the forecast of the lag-free convex demand
  model of [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), as stored by
  experiment
  327 (arm C, trained on every half-hour that has a reading, refitted on the 1st of each month on a
  24-month window). Here the forecast is refitted for the cuts 2026-07-01, 08-01, 09-01, and 10-01,
  using experiment 327's own code path on a feature cache extended to 2026-10-31. The median
  residual
  across the 15 Primaries is removed before segmentation, with each Primary's scale fixed on the
  data
  before 2026-07-01. D18 loses every row before 2025-03-01.
- **Noise unit and slow level.** One noise unit is the robust spread of the difference between
  successive 6-hour block means of the residual before 2026-07-01, frozen. The slow level is a
  seasonal level per Primary fitted with a
  [Huber (1964)](https://doi.org/10.1214/aoms/1177703732) loss, as in experiment 653, and refitted
  on
  the whole extended panel.
- **Segmentation.** Each Primary's residual, less the slow level, is cut into 6-hour block means. An
  exact dynamic programme decodes 25 states per block (no shift, and 12 amplitudes log-spaced from 1
  to 20 noise units in each sign) with a Student-t emission, using the
  [Viterbi (1967)](https://doi.org/10.1109/TIT.1967.1054010) algorithm, then
  refines each start and end to the half-hour. Events are classed as ramps, routine steps,
  permanent or open shifts, and paired across
  neighbouring Primaries (the graph of
  [experiment 383](../00383-load-shift-xgboost-oracle-first-part-1/result.md)) as in experiment 653.
- **Recipient search (E1-J,
  [experiment 670](../00670-event-table-recipient-search-on-six-cell-grid/result.md)).**
  From each unpaired fall of 4 noise units or more, look at every connected Primary for a rise of 2
  noise units or more starting near the fall, and accept up to three recipients whose combined rise
  conserves the fall under experiment 653's loose conservation cost.
- **Donor search (E1-J-sym, experiment 680).** From each rise of 4 noise units or more still
  unpaired, look at every connected Primary for a fall of 2 noise units or more that starts up to 36
  half-hours before the rise (widened by 6 half-hours on each side for start error), with a
  recipient
  share of 0.3 to 2.0 and a conservation cost of at most 2.5.
- **Output table.** One row per event of 2 noise units or more that is not a ramp or routine step:
  the Primaries involved, class (edge transfer or solo shift), onset and end (end-of-half-hour
  stamps in UTC, or open), depth in megawatts and noise units, confidence, and whether the event
  starts before the window or ends after it. An edge transfer appears as two rows.
- **Compute.** CPU only, 2 cores. The ERA5 extension is 2 m temperature only for August to October
  2026 (1.6 MB), read from the ARCO-ERA5 store (Google's public copy of ERA5) with the stage-1 box
  script.

## Files

- `ml-experiments/00685-.../e00685_fetch_era5.py`: runs the stage-1 ERA5 box fetch with its span set
  to 2026-08-01 to 2026-10-31.
- `e00685_inputs.py`: extends experiment 103's row table, the ERA5 temperature, and the CAMS
  irradiance to 2026-10-31.
- `e00685_refit.py`: refits experiment 327's arm C at the cuts 2026-06-01 to 2026-10-01 (the first
  is the gate) and overrides, in that process only, the three constants in the feature-cache and
  fitting code that exclude the window.
- `e00685_run.py`: extends the power panel, removes the common median, runs E1-J-sym, checks the
  earlier events against experiment 680, and writes the tables.
- Private tables, not in git:
  `data/ml-experiments/00685-event-table-on-the-final-test-window-all-primaries/event_table/`
  (`candidate_events.parquet` and `.csv`, `SUMMARY.md`, `reproduction_gate.json`, and the lists of
  events only in the stored or only in the new table), and `event_table_truncated_check/` for the
  window-blanked run. Inputs and forecasts: `inputs/`, `e327_extended/`, `era5_box_extension/` in
  the
  same folder.
