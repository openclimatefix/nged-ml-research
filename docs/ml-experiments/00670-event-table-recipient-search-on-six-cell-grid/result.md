---
id: "00670"
slug: event-table-recipient-search-on-six-cell-grid
title: >-
  Does a search for the recipient of each donor step let the event table pair injected load
  transfers, split by how long the transfer lasts and how deep its step is?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-09
kind: other
size: >-
  medium. (1) Stored: private score tables (parquet and JSON) and figures under
  data/ml-experiments/00670-event-table-recipient-search-on-six-cell-grid/ only. (2) Serving path:
  not touched. (3) Degradation rule: not touched. (4) Designs: the recipient search has more than
  one defensible form, so its thresholds are fixed below before any score is read. (5) Callers:
  this folder's scripts, which import the folders of experiments 653, 660, and 665; nothing
  imports them. The maintainer's brief limits the work to CPU and 2 cores, with no network
  training.
hypothesis: >-
  Experiment 653's event table pairs few injected load transfers (0.18 of the megawatts at the two
  ends) because each end must clear the full detection penalty on its own. A search at each
  connected Primary for a recipient step of opposite sign, started within 36 half-hours of a deep
  donor step, needs only the small recipient step to be present, and so pairs at least 0.6 of the
  injected megawatts in transfers that are both long (over 7 days) and deep (8 noise units or
  more), the bar that experiment 653 set for any model of edges.
prediction: >-
  Written before any score is read, per cell of the six types of experiment 665 (long, medium, and
  short times deep and shallow), for E1-J, MW-weighted. Found and paired shares at least: long and
  deep 0.85 and 0.60; medium and deep 0.80 and 0.50; long and shallow 0.55 and 0.30; medium and
  shallow 0.45 and 0.20; short and deep 0.40 and 0.20; short and shallow 0.15 found, with no
  prediction for pairing. E1-J pairs at least 0.25 more than E1 in the long and deep cell and in
  the medium and deep cell. On the real record E1-J pairs the verified transfer between D5 and
  D16 in the right direction with both ends within 12 hours of the verified starts. A rewired
  graph gives a third or fewer of the real record's edge events.
decision_rule: >-
  Fixed before any score is read. Worlds come from experiment 660's generator, injected into the
  power; the residual is the injected power minus the lag-free forecast with the common mode
  removed. Rows, cells, and the found and paired definitions are those of experiment 665 (events of
  2 noise units or more, of kind edge, solo, permanent_transfer, or permanent_shift, both signs,
  weighted by absolute megawatts; long is over 7 days or no end, medium is over 1 day up to 7 days,
  short is 1 day or less; deep is 8 noise units or more). Intervals resample whole worlds. A cell
  with fewer than 20 events or 10 edge-transfer ends is too small to read, and a prediction that
  needs it counts as not holding. Stop rule, applied to E1-J's paired share with the world-resampled
  interval's point estimate. (a) If it is 0.60 or more in both the long and deep and the medium and
  deep cells, E1-J becomes the event table. (b) If it is under 0.30 in the long and deep cell, the
  pairing of the two ends of a transfer is reported as unidentifiable at 15 Primaries, and the
  graph model (X2, GPU) is not run. (c) Otherwise the report says that X2 is the next step. X2 is
  not started here. Verdict. `supported` if the long and deep and the medium and deep cells both
  meet their found and paired predictions, both differences over E1 reach 0.25, and the rewired
  graph holds. `partly supported` if at least 8 of the 15 individual predictions hold (6 found
  shares, 5 paired shares, 2 differences, the D5 and D16 pair, and the rewired graph) and the
  supported condition does not. `not supported` otherwise. The porting threshold is zero, because
  the result is a research table that changes no production code. A reproduction gate must pass
  first, or the verdict is `invalid`. Gate 1 (injected). E1 on the 8 default worlds of seeds 900 to
  907 reproduces experiment 660's stored recall by size within 0.02 per size bin. Gate 2 (real
  record). E1's events on the real record in the scored years equal the 246 stored by experiment
  653.
benchmark: >-
  E1, experiment 653's event table in its default arm (masked slow level, tuned penalties of that
  experiment), on the same worlds. The label-trained detector LR_b20 of experiment 254 is reported
  on the real record only, from experiment 665's stored counts.
row_set_id: >-
  experiment 660's injected worlds (seeds 900 onward, default configuration, scored years
  2024-01-01 to 2026-06-30) and experiment 653's real record of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted share of injected events found (any detection) and of injected edge-transfer ends
  paired correctly, per cell, with 95% intervals from resampling whole worlds; start and end timing
  error; on the real record, logged shifts matched and events called per Primary-year.
code: ml-experiments/00670-event-table-recipient-search-on-six-cell-grid/
---

*This page is the plan. The result is written after the run.*

## The question

The load-shift event table of [experiment 653](../00653-edge-state-event-table-penalised-segmentation/result.md)
finds an injected transfer's two ends one at a time and then pairs them, and it paired 0.18 of the
injected megawatts. [Experiment 665](../00665-event-table-recall-by-length-and-depth/result.md)
showed that no cell of event length and depth paired better than 0.42, from 14 ends. This
experiment asks whether the event table pairs better when, after it has found a deep donor step, it
searches the connected Primaries for the recipient step. The result decides whether the graph
neural network proposed for the next experiment (X2) has anything to improve on.

## Arms

- **E1.** Experiment 653's event table in its default arm.
- **E1-J.** E1, plus a search at each connected Primary for an upward step of 2 noise units or
  more that starts from 6 half-hours before to 42 half-hours after the start of each unpaired
  downward donor step of 4 noise units or more. A set of up to 3 recipients is accepted when its
  steps conserve the donor's step under experiment 653's cost, with the share of the donor's step
  between 0.3 and 2.0, and the cost at most 2.5. The search window of 0 to 36 half-hours is
  experiment 660's recipient offset, and the 6 half-hours before and after allow for the error in
  a refined start. The step size is the median after the start (to the donor's end, at most 3
  days) minus the median over the 3 days before.
- **E1-J rewired.** The same search, and E1's own pairing, on a degree-preserving rewired
  connection graph (one rewiring per world, 100 on the real record). Its pairing of injected
  transfers should fall to about zero, and it is the placebo for the real record's edge events.

## Plan

- **Worlds.** 40 worlds of the default configuration of experiment 660 (seeds 900 to 939), and
  10 each of its `long_heavy`, `short_heavy`, and `constant_mw_square` configurations as
  sensitivity arms. The default worlds hold about 3 events per world in the long and deep cell, so
  40 worlds are needed for 20 or more edge ends there.
- **Panel.** Experiment 653's panel from 2022 (the 2019-onward cross-fitted panel is not built):
  this experiment trains nothing, the scored years are the same either way, and the panel's other
  use, a longer training history for a neural network, belongs to X2. The page says so.
- **Real record.** Events called per Primary-year, logged shifts matched (start within 2 and within
  24 hours) per cell, the event-free stratum, the verified D5 and D16 transfer, and edge events
  under 100 rewirings. LR_b20's matches per cell come from experiment 665's stored file.
- **Realness.** Experiment 660's generator scored 0.55 on its realness classifier (the gate is
  under 0.65). X1 does not rerun it, because the generator is unchanged. The version that reads
  features at decoded times is the gate of X2.
- **Standing practices.** Clean data, with D18 cut before 2025-03-01 (experiment 660's panel); no
  fits; no read of any half-hour from 2026-07-01 onward; NGED's logs are an incomplete one-off
  dump, so matched counts are lower bounds; generators are anonymised.
- **Compute.** CPU only, 1 core per process and at most 2 processes, about 6 seconds per world.
  No GPU. No fitted model is saved because none is fitted.
- **Earlier advice.** The plan follows the advice of experiment 665 to report cells with at least
  20 events and 10 ends, and to report shifts of a day or less apart.

## Reviews

The plan review, code review, results review, and prose review run in the order the
`ml-experiment` skill gives; the section is filled in at the end.
