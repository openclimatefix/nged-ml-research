---
id: "00680"
slug: event-table-reconstruction-from-edge-flows-part-1-symmetric-search
title: >-
  Does a search that also starts from each unpaired rise, and looks for the matching fall at a
  neighbouring Primary, pair more injected load-transfer megawatts than the one-way search of
  experiment 670?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private score tables (parquet and JSON) and figures under
  data/ml-experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/
  only. (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: the donor
  search has one obvious mirror of experiment 670's search, whose thresholds are fixed below
  before any score is read; the later edge-flow model (part 2) admits more than one design and
  gets its own plan. (5) Callers: this folder's scripts, which import the folders of experiments
  653, 660, 665, and 670; nothing imports them. CPU only, 2 cores.
hypothesis: >-
  Experiment 675 found that the donor is the bottleneck of experiment 670's search: E1-J finds
  0.73 of the recipient megawatts in the long and deep cell but only 0.44 of the donor megawatts,
  and 0.87 of the recipient megawatts it found but left unpaired belong to transfers whose donor
  it had not found. A donor search that starts from each unpaired rise of 4 noise units or more,
  and looks at each connected Primary for a fall of 2 noise units or more that starts up to 36
  half-hours before the rise, can recover those donors, and so pairs at least 0.10 more of the
  injected transfer megawatts than E1-J in both the long and deep cell and the medium and deep
  cell.
prediction: >-
  Written before any score is read, as the difference in MW-weighted paired share, E1-J-sym minus
  E1-J, on the 40 default worlds. Long and deep: at least +0.10 (E1-J is 0.29). Medium and deep:
  at least +0.10 (E1-J is 0.25). Long and shallow: at least +0.03. Medium and shallow: at least
  +0.02. Short cells: no worse than -0.03. Donor megawatts found in the long and deep cell: at
  least 0.54 (E1-J is 0.44). A rewired graph gives a third or fewer of the real record's decoded
  transfers. On the real record E1-J-sym keeps the D5 to D16 transfer in the right direction.
  Maintainer-side estimate that the +0.10 bar is met in both deep cells: about 35%.
decision_rule: >-
  Fixed before any score is read. Worlds, residual, cells, and the found and paired definitions are
  those of experiment 670 (and so of experiment 665). Intervals resample whole worlds. A cell with
  fewer than 20 events or 10 edge-transfer ends is too small to read, and a prediction that needs
  it counts as not holding. Stage 1 rule, applied to the point estimate of E1-J-sym minus E1-J
  in the paired share. (a) If it is +0.10 or more in both the long and deep and the medium and deep
  cells, E1-J-sym becomes the event table, the page says so, the event-table summary page is
  updated, and the edge-flow reconstruction model (part 2) is run only if the maintainer asks.
  (b) If it is under +0.03 in both deep cells, the either-end-seeds mechanism is weak, and part 2
  is run all the same, because it also tests the change-cost objective, with a stated prior of 20%
  or less. (c) Otherwise part 2 is run. Verdict. `supported` if both deep differences reach +0.10
  and the rewired graph holds. `partly supported` if one deep difference reaches +0.10, or both
  reach +0.05. `small gain` if both deep differences have intervals excluding zero but neither
  reaches +0.05. `not supported` otherwise. The porting threshold is zero, because the result is
  a research table that changes no production code. A reproduction gate must pass first, or the
  verdict is `invalid`. Gate. Re-running E1-J on the 40 default worlds reproduces experiment 670's
  stored paired shares per cell within 0.005.
benchmark: >-
  E1-J, experiment 670's event table with its recipient search, on the same 40 default worlds
  (seeds 900 to 939, scored years 2024-01-01 to 2026-06-30), read from experiment 670's stored
  rows and re-run here as the gate. F-hyb, R-hyb, and 676-G are reported from experiments 676 and
  678 as references only.
row_set_id: >-
  experiment 660's injected worlds (seeds 900 onward, default configuration, scored years
  2024-01-01 to 2026-06-30) and experiment 653's real record of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted share of injected events found and of injected edge-transfer ends paired correctly,
  per cell, with 95% intervals from resampling whole worlds; donor and recipient megawatts found;
  decoded transfers and correct ends; on the real record, decoded transfers per Primary-year, the
  verified D5 and D16 transfer, and logged shifts matched.
code: ml-experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/
production_status: not-for-production
adopted_in: none
---

**This is the plan of part 1 of a two-part line. Part 2 would reconstruct the event table from
flows on the edges between Primaries; part 1 tests the cheaper mechanism that part 2 would also
rely on.** The learnt-graph line was closed at 15 Primaries by experiments 672 and 676. Part 2 is a
different model: a reconstruction of the residual by flows on edges, not a learnt neural network.

## Question

A *Primary substation* steps electricity down to a local distribution network. A *load transfer*
moves demand from one Primary (the *donor*) to a neighbour (the *recipient*). Experiment 670's
search starts at a large fall and looks for the rise at a neighbour. That search finds the
recipient end often (0.73 of the long and deep megawatts) and the donor end rarely (0.44).
This experiment adds the mirror search: it starts at each rise that is still unpaired and looks for
the fall at a neighbour.

## Arms

- **E1-J** is experiment 670's event table with its recipient search. It is re-run as the gate.
- **E1-J-sym** is E1-J, then the donor search from each unpaired rise of 4 noise units or more. The
  donor must be a fall of 2 noise units or more that starts from 42 half-hours before to 6
  half-hours after the rise. Among the connected Primaries, the one whose step best conserves
  the rise (experiment 653's loose conservation cost, share 0.3 to 2.0) is accepted.
- **E1-J-sym rewired** is E1-J-sym on a degree-preserving rewired graph (the draw of 30 that keeps
  the fewest real links, as in experiment 670). It is the placebo.

References from stored rows: E1 (experiment 670), F-hyb, R-hyb (experiment 678), and 676-G.

## Plan

- Worlds: the 40 default worlds of experiment 670 (seeds 900 to 939). The thresholds are those
  of experiment 670 mirrored, and are fixed in `e00680_search.py` before any score is read. No
  tuning is done.
- Cells, found and paired definitions, and false-pairing counts as in experiment 670. The
  other three generator configurations (long heavy, short heavy, constant-MW square) are run as a
  robustness check.
- Real record: decoded transfers per Primary-year, the verified D5 and D16 transfer, the 8
  transfers E1-J calls, and the 77 logged shifts longer than 1 day, as in experiment 670.
- Standing practices: cleaned data, the D15 and D18 cut and the final-test window exclusion are
  inherited from experiment 653's panel. No XGBoost model is fitted, so retraining cadence, saved
  boosters, and the forecast graphs do not apply. NGED's event labels are an incomplete dump, so
  every count of real events is a lower bound. No NGED generator is named.
- Compute: about 7 seconds per world on one core, so under 10 minutes in all. Run with
  `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue` and one thread per process.
- Reviews: one fresh Sonnet code review before the first scored run, one Sonnet review of the
  results, and one Sonnet prose review before merge. No Opus review.

## Stop rules

The stage 1 rule above. Part 2 (the edge-flow reconstruction model, spec in the maintainer's
brief) is planned in its own result page once part 1 reports.
