---
id: "00672"
slug: graph-model-that-finds-the-donor-of-a-load-transfer
title: >-
  Does a graph neural network find the donor of an injected load transfer, and pair it with its
  recipients, better than experiment 670's search at connected Primaries?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-09
kind: other
size: >-
  complex. (1) Stored: private injected-world banks, network weights, score tables, and figures
  under data/ml-experiments/00672-graph-model-that-finds-the-donor-of-a-load-transfer/ only. (2)
  Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is
  defensible for the donor objective, the decoder, and the placebo graph, so each is fixed below
  before any score is read. (5) Callers: this folder's scripts, which import the folders of
  experiments 653, 657, 658, 660, 665, and 670; nothing imports them. GPU, behind the maintainer's
  one-year backtests; ceiling 8 GPU-hours.
hypothesis: >-
  Experiment 670 found that the event table with a recipient search (E1-J) pairs 0.29 (0.20 to
  0.40) of the injected megawatts in the long and deep cell, and that two losses limit it: the
  event table finds only 0.44 of the donor megawatts, and the search pairs only 0.22 of the
  recipient megawatts although it finds 0.73. A graph neural network that reads every Primary's
  residual and the electricity network's links together, and is trained on experiment 660's
  injected worlds with an objective for the donor itself, finds the donor of a transfer by the
  matching recipient step at a linked Primary, which a per-Primary detector cannot see when the
  donor step is small. It therefore finds more donor megawatts and pairs at least 0.10 more than
  the search in the long and deep and in the medium and deep cells.
prediction: >-
  Written before any score is read; each is a mean over 3 seeds, and a prediction fails if any seed
  is more than 0.05 worse than its threshold. Cells are those of experiment 665, MW-weighted, in
  experiment 660's default worlds. P1 (the pass bar, set against E1-J's 0.29 in the long and deep
  cell): G pairs at least 0.10 more than E1-J in the long and deep cell and in the medium and deep
  cell, and at least 0.10 more in the long and shallow cell; at least 0.05 more in the medium and
  shallow and the short and deep cells. P2 (the donor objective): G finds at least 0.10 more of the
  donor megawatts of transfers than E1-J in the long and deep and in the medium and deep cells (the
  baseline is E1-J's own donor-found share in each cell, re-measured from experiment 670's stored
  rows; E1 finds 0.44 of the long and deep donor megawatts). P2 tests G as built, with the donor
  head and its loss. It does not separate the donor objective from the graph, because no arm drops
  the donor loss; the decode weight `lambda` is the only ablation of the donor head. P3 (found
  means any injected event of the cell, donors and recipients together): G's found share minus
  E1-J's is at least +0.03 in the long and deep, medium and deep, and +0.08 in the long and shallow
  cell, +0.05 in the medium and shallow cell, and no worse than -0.03 in the short and deep and
  short and shallow cells; G's absolute paired share is at least 0.70 (long and deep), 0.60 (medium
  and deep), and 0.40 (long and shallow). P4: G minus G-0 in found share is at least +0.05 in the
  long and shallow and medium and shallow cells, and within 0.03 in the long and deep cell. P5: for
  70% of found events in the long and deep and medium and deep cells the start is within 2 hours
  and the end within 6 hours of the truth. P6: on the real record G pairs the verified D5 and D16
  transfer with D5 as the donor, both starts within 6 hours of the verified starts, and each end
  between 1.2 and 2.2 MW. P7: on the real record G-R decodes a third or fewer of G's edge events in
  the long and deep and the medium and deep cells. P8: G's events per Primary-year on the real
  record are within a factor of 2 of experiment 653's, at 13 or more Primaries. P9: the
  cross-injector drop (in-injector minus cross-injector found share, long and deep and medium and
  deep cells) is 0.05 or less in both directions. P10: the arm with the explicit solar term (GS) is
  within 0.03 of G in found, donor found, and paired shares in the long and deep and medium and
  deep cells, because experiment 657 found that the term changed neither the residual nor the
  recall.
decision_rule: >-
  Fixed before any score is read. Worlds come from experiment 660's generator, injected into the
  power of experiment 660's panel (2022-01-01 to 2026-06-30); the residual is the injected power
  minus the lag-free forecast with the common mode removed, as in experiment 670. Training worlds
  start events in 2022-01-01 to 2023-06-30 and show the network nothing later; validation worlds
  start events up to 2023-12-31, and the loss counts only blocks from 2023-07-01; scored worlds are
  the 40 default worlds of experiment 670 (seeds 900 to 939) with events scored from 2024-01-01.
  Training seeds start at 100000 and validation seeds at 105000, so the two sets and the scored
  seeds (900 to 939, and 1000 to 1009 for the cross-injector test) are disjoint. The lag-free
  forecast that makes the residual is experiment 327's, fitted before 2024 as in experiment 670.
  The final-test window from 2026-07-01 is never read. Decoder costs (onset cost, none boost, donor
  weight, link cut-off) are chosen on validation worlds from a grid frozen in `e00672_run.py`, by
  the highest found share at no more than 6 events per Primary-year on the real record from
  2023-07-01 to 2023-12-31, the months of the validation worlds (experiment 670's E1-J decodes
  about 6 events per Primary-year there as well; both are caps on false events, and the verified D5
  and D16 transfer of December 2025 and the D3 and D22 transfer of October 2024 fall outside it).
  Intervals resample whole worlds. A cell with fewer than 20 events or 10 edge-transfer ends is too
  small to read. A prediction that needs such a cell is reported as `unreadable`, and an unreadable
  cell counts neither for nor against stop rules (a) to (c) and the verdict; if the long and deep
  or the medium and deep cell is unreadable, the verdict is `inconclusive`. Stop rule (the spec's,
  applied to the 3-seed means): the graph line is closed at 15 Primaries, and the next step waits
  for the fleet-scale adjacency of about 1,161 Primaries, if any of these holds. (a) G misses the
  pairing margin of +0.10 in the long and deep or the medium and deep cell. (b) The cross-injector
  drop is above 0.05. (c) G-R reaches half or more of G's real edge events. X3 (history length) is
  started only if none of (a) to (c) holds, and not by this experiment. Verdict. `supported` if P1
  holds in the long and deep and the medium and deep cells, P2 holds, and none of (a) to (c) holds.
  `partly supported` if P2 holds but P1 does not, or P1 holds in only one of the two cells, or P1
  holds but one of (b) and (c) fails. `not supported` otherwise. P3 to P10 do not enter the
  verdict; the write-up lists every prediction that failed. If G-R reaches more than a third but
  under half of G's real edge events, P7 fails and the graph line stays open. The porting threshold
  is zero, because the result is a research table that changes no production code. A reproduction
  gate must pass first, or the verdict is `invalid` and no network is trained: E1-J's rows in all
  six cells, re-scored here from experiment 670's stored worlds must reproduce experiment 670's
  headline cells within 0.01.
benchmark: >-
  E1-J, experiment 670's event table with a recipient search, on the same 40 default worlds (seeds
  900 to 939), from experiment 670's stored scored rows. E1 and the label-trained detector LR_b20
  are reported on the real record from experiments 670 and 665.
row_set_id: >-
  experiment 660's injected worlds (seeds 900 to 939, default configuration, scored years
  2024-01-01 to 2026-06-30) and experiment 653's real record of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted share of injected events found, of injected edge-transfer donor megawatts found, and
  of edge-transfer ends paired correctly, per cell, with 95% intervals from resampling whole
  worlds; start and end timing error; on the real record, edge events per cell, logged shifts
  matched, events per Primary-year, and the verified D5 and D16 transfer. No forecast is scored, so
  the four load-shift scenarios, tail calibration, and the curtailment-cost metric do not apply.
code: ml-experiments/00672-graph-model-that-finds-the-donor-of-a-load-transfer/
---

*This page is the plan. The result is written after the run.*

## The question

[Experiment 670](../00670-event-table-recipient-search-on-six-cell-grid/result.md) gave the event
table of [experiment 653](../00653-edge-state-event-table-penalised-segmentation/result.md) a
search at connected Primaries for the recipient of each donor step. That event table (E1-J) paired
0.29 of the injected megawatts in the long and deep cell, against 0.09 without the search. Two
losses limit it: E1 finds only 0.44 of the donor megawatts (E1-J pairs 0.36), and the search pairs
only 0.22 of the recipient megawatts although it finds 0.73. E1-J also decodes false transfers (69%
of its decoded transfers match no injected end, against 57% for E1). This experiment asks whether a
graph neural network, trained to find the donor and reading the recipient evidence at the linked
Primaries, improves both ends, by at least 0.10 of paired megawatts over E1-J.

## Arms

- **G.** The donor-finding network on the real connection graph (below).
- **G-R.** The same network trained and run on a degree-preserving rewired graph, the placebo (one
  rewiring per seed, the one of 300 draws that keeps the fewest real links, because experiment
  670's best of 30 kept 16% of the real links; the share kept is reported).
- **G-0.** The same network with no links and no neighbour layer: each Primary sees only itself.
- **GS.** G with the explicit solar term added to the residual at the six Primaries metered in
  megawatts: known weather from CAMS and ERA5 and the capacity table of
  [experiment 649](../00649-pv-capacity-table-season-year-from-stored-probes/result.md), built by
  [experiment 657](../00657-event-table-with-explicit-solar-term-against-plain-arm/result.md).
  Experiment 657 tested this term inside E1 and found that it left the residual and the recall
  unchanged; GS repeats the test inside the network. The capacity table's 2026-06 cut-off and the
  gain fitted on 2022 and 2023 sit partly inside the scored years, as in experiment 657, so a gain
  would be an upper bound.
- **Reference rows.** E1-J and E1 from experiment 670's stored scored rows on the same worlds.

Each arm trains 3 seeds, on the `default` generator configuration. Two further networks, each one
seed, give the cross-injector test: G trained on the
`constant_mw_square` configuration of experiment 660 and G trained on its `shaped_transfers`
configuration. Each is scored on 10 worlds (seeds 1000 to 1009) of both configurations, so the
test reads in-injector minus cross-injector found share in both directions. P9 is a one-seed
reading.

## Plan

- **Network.** Experiment 658's recurrent encoder, shared by the 15 Primaries, a neighbour-mean
  layer, and a link head, on the 6-hour block grid of experiment 653. A **donor head** is added:
  per Primary, the chance that the Primary is the donor of a transfer with a metered recipient. The
  link head also reads the two Primaries' donor logits, so the recipient state is conditioned on
  the donor. The loss is experiment 658's plus a weighted cross-entropy on the donor flag (donor
  blocks weigh 5, about 2% of blocks). The graph specification's loss table is not used: 658's
  loss is the one already shown to train on these worlds.
- **Decoder.** Experiment 658's Viterbi decode and link pairing, with one addition: the decoder
  moves `lambda` times the donor probability of each block's 'none' mass to 'shift, down'
  (`lambda` 0 is the ablation of the donor head's role in decoding; 1 is the donor-aware decode,
  chosen on validation worlds from {0, 1}).
- **Resolution.** Blocks are 6 hours, as in experiment 658, not the 1-hour blocks the
  specification priced at about 10 times the cost; boundaries are refined to the half-hour by
  experiment 653's refinement. If the timing prediction (P5) fails for this reason, the write-up
  says so.
- **Ends reported separately.** As experiment 670's advice asks, every cell reports the donor end
  and the recipient end separately (found and paired), and the false transfers (decoded transfers
  that match no injected end) beside recall.
- **Donor recall.** The share of the megawatts of injected donor events (negative step, an edge
  transfer with a metered recipient, 4 noise units or more) that the decoded table finds, by the
  matching rule of experiment 653, reported beside pairing for every cell.
- **Panel.** Experiment 653's panel from 2022 (no 2019-onward panel exists on disk, and a fetch
  is outside this experiment's scope); the history-length question belongs to X3.
- **Real record.** As in experiment 670: events per Primary-year, logged shifts matched, edge
  events per cell, the verified D5 and D16 transfer, and the event-free stratum, for G, G-R, G-0,
  and GS.
- **Compute.** 14 networks of about 2400 steps each. One network is timed first. If one takes more
  than 30 minutes, the step count is cut to 1200 for all 14 networks. Decoder tuning and scoring run
  on CPU while the GPU trains the next network, and the ceiling of 8 GPU-hours counts training time
  only.
  One GPU process at a time, at most 4 CPU cores, `MemoryMax=16G` per script. The one-year backtests
  hold priority: a fit pauses when a backtest worker needs the GPU.
- **Definitions.** A cell is one of six groups of injected events by length (long over 7 days or no
  end, medium over 1 up to 7 days, short up to 1 day) and depth (deep 8 noise units or more,
  shallow 2 up to 8). An event is found when the decoded table has an event at the right Primary
  and sign starting within 12 hours; an end is paired when the decoded table links it to the right
  other end; an edge event is a decoded transfer. Shares are weighted by absolute megawatts.
- **Inherited leaks, stated.** The noise units (experiment 653's, fitted before 2024-01-01) saw the
  validation months, and the injected start hours and weekdays come from NGED's logged onsets up to
  2026-07, so the scored window has shaped both. Both are shared with experiment 670, so the
  contrast with E1-J is unaffected. The GS arm also carries the capacity table (cut-off 2026-06) and
  a gain fitted before 2024-01-01, so a GS gain would be an upper bound.
- **Caveat.** NGED's logs are an incomplete one-off dump, so matched logged shifts are lower
  bounds.

## Reviews

The plan review, code review, results review, and prose review run in the order the
`ml-experiment` skill gives; the section is filled in at the end.
