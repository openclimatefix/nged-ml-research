---
id: "00682"
slug: event-table-reconstruction-from-edge-flows-part-2-change-cost-model
title: >-
  Does choosing edge flows by a fixed cost per change pair more injected load-transfer megawatts
  than the searches of experiments 670 and 680, without adding more false transfers?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: finished
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private score tables (parquet and JSON) and figures under data/ml-
  experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/ only. (2)
  Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: a reconstruction by
  edge flows has more than one defensible form, so its costs, grid, and tuning rule are fixed
  below before any score is read. (5) Callers: this folder's scripts, which import the folders of
  experiments 653, 660, 665, 670, and 680; nothing imports them. CPU only, 2 cores.
hypothesis: >-
  Experiment 680 found that searching from either end of a transfer raises the paired share by
  +0.09 (long and deep) but adds 442 decoded transfers for 63 correct ends. A model that decides
  every pair jointly, by the saving in change cost against the misfit of one common flow, with the
  recipient share bounded by a prior, accepts only pairs whose two steps are explained better by
  one flow than by two steps that stand alone. It therefore pairs at least 0.10 more of the
  injected transfer megawatts than E1-J in both the long and deep and the medium and deep cells,
  with a share of decoded transfers that match no injected end no higher than E1-J's.
prediction: >-
  Written before any score is read, MW-weighted paired share on the 40 default worlds. Long and
  deep: E1-F at least 0.39 (E1-J + 0.10), and at least 0.03 above the better of F-hyb (0.36) and
  R-hyb (0.35). Medium and deep: at least 0.35, and at least 0.03 above the better hybrid (about
  0.33). Donor megawatts found in the long and deep cell at least 0.54. Long and shallow: E1-J +
  0.05. Medium and shallow: E1-J + 0.03. Short cells: no worse than E1-J - 0.03. E1-F minus
  E1-J-sym at least +0.05 paired in both deep cells. E1-F minus E1-F all pairs: a false-transfer
  share at least 0.10 lower in both deep cells, at a paired share no more than 0.03 lower. E1-F
  rewired pairs 0.05 or less in both deep cells and calls a third or fewer of E1-F's real-record
  transfers. E1-F no edges has a found share within 0.03 of E1-F's in the deep cells. On the real
  record the verified D5 to D16 transfer is one flow with D5 as the donor, both starts within 6
  hours of the verified onsets (D5 2025-12-29 01:00, D16 about 15 hours later), the end within 12
  hours of 2026-01-13 11:30, 1.2 to 2.2 MW at each end, and a share from 0.8 to 2.0. The author of
  the spec put the chance that E1-F beats E1-J by +0.10 in both deep cells at about 35%, and the
  chance that it beats the better hybrid by +0.10 at about 15%.
decision_rule: >-
  Fixed before any score is read. Worlds, residual, cells, and the found and paired definitions
  are those of experiments 670 and 680. Intervals resample whole worlds. A cell with fewer than 20
  events or 10 edge-transfer ends is too small to read. Tuning. The change costs c_edge (2, 4, 6,
  8, 12) and delta (0.5, 1, 2, 4, 8) are chosen on 60 validation worlds whose events start from
  2023-07-01 to 2023-12-31 (seeds 20000 to 20059, readings after 2023-12-31 hidden). The chosen
  pair maximises the pooled paired share of the two deep cells among the pairs whose false-
  transfer share (one minus correct ends over decoded transfers) is no higher than E1-J's on the
  same worlds, with ties broken towards larger delta. If no pair meets the constraint, the pair
  with the highest paired share is used and the page says so. Stop rules, applied to the point
  estimate on the scored worlds. (a) Close the edge-flow line at 15 Primaries if E1-F misses E1-J
  + 0.10 paired in both deep cells; (b) close it as well if E1-F rewired pairs more than half as
  much as E1-F in a deep cell, or E1-F all pairs matches E1-F on the false-transfer share (the
  topology is not identified). (c) If E1-F passes in at least one deep cell, a learnt noise model
  (ceiling 1 GPU-hour) may be planned, and must add +0.03 paired in that cell. (d) If E1-F passes
  in one deep cell only, use E1-F for that cell's lengths and E1-J-sym for the rest. Verdict.
  `supported` if both deep differences from E1-J reach +0.10 and the topology checks hold. `partly
  supported` if one reaches +0.10, or both reach +0.05. `small gain` if both deep intervals
  exclude zero but neither reaches +0.05. `not supported` otherwise. The porting threshold is
  zero, because the result is a research table. Gate. The experiment reads E1-J and E1-J-sym from
  experiments 670 and 680 and recomputes E1 from the same code, so the gate is that E1-F no edges
  reproduces experiment 670's E1 found share per cell within 0.005. A cross-injector check tunes
  on the constant-MW square worlds and scores the default worlds, and the reverse; the paired-
  share drop must be 0.05 or less.
benchmark: >-
  E1-J and E1-J-sym on the same 40 default worlds (seeds 900 to 939), from the stored rows of
  experiments 670 and 680; F-hyb, R-hyb, and 676-G from experiments 678 and 676 as references.
row_set_id: >-
  experiment 660's injected worlds (seeds 900 onward, default configuration, scored years
  2024-01-01 to 2026-06-30) and experiment 653's real record of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted share of injected events found and of injected edge-transfer ends paired correctly,
  per cell, with 95% intervals from resampling whole worlds; donor and recipient megawatts found;
  decoded transfers and correct ends; on the real record, decoded transfers per Primary-year, the
  verified D5 and D16 transfer, the eight transfers E1-J calls, and logged shifts matched.
code: ml-experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/
verdict: not supported
best_arm: >-
  E1-J-sym of experiment 680 remains the best of the methods compared in experiments 670 to 682:
  0.38 of long and deep megawatts paired. E1-F pairs 0.33 (95% interval 0.23 to 0.46), +0.041
  (-0.041 to 0.124) over E1-J
headline: >-
  Choosing edge flows by a fixed cost per change pairs 0.33 of the long and deep injected transfer
  megawatts and 0.27 of the medium and deep ones, +0.04 and +0.03 over E1-J and below the
  symmetric search of experiment 680, and it accepts 2990 transfers with 329 correct ends. A graph
  that connects every pair of Primaries pairs 0.32 and 0.26, so the cost objective gains almost
  nothing from knowing the real edges.
lessons: >-
  With the real edges, 89% of the transfers E1-F decodes have no correct end, and letting every
  pair of Primaries connect pairs 0.32 against 0.33. Change costs from 0 to 64 left the share of
  such transfers at 0.88 to 0.90, so the cost does not choose between a true pair and a coincident
  pair. Whatever separates them has to come from the misfit term or from information that the
  residuals lack, such as which neighbour received the load.
take_home_message: >-
  Explaining pairs of steps by one flow on an edge, chosen by a fixed cost per change, does not
  find the donor and recipient of an injected load transfer better than the simple searches: it
  pairs 0.33 of the long and deep megawatts against 0.29 for the one-way search and 0.38 for the
  two-way search. It accepts almost every pair that roughly conserves megawatts, 2990 transfers
  for 329 correct ends, and a graph that connects any two Primaries pairs nearly as much as the
  real neighbours. Within the 15 Primaries tested, this cost objective gains almost nothing from
  knowing the real edges, and the edge-flow line closes here.
advice:
  - >-
    Do not expect a change cost to make a flow model selective: across the 25 cost pairs of the
    plan and two post hoc grids (costs from 0 to 64), the share of decoded transfers without a
    correct end stayed between 0.88 and 0.90 on validation worlds (cost grids).
  - >-
    Compare any graph method with the same method on an all-pairs graph, because here the
    all-pairs graph paired 0.32 against 0.33 for the real edges (all-pairs arm).
production_status: not-for-production
adopted_in: none
---

**Choosing edge flows by a fixed cost per change does not pair injected load transfers better than
the simple searches of experiments 670 and 680: it pairs 0.33 of the long and deep transfer
megawatts, against 0.29 and 0.38, and it decodes 2990 transfers of which only 329 ends are
correct.** A *Primary substation* steps electricity down to a local distribution network. A *load
transfer* moves part of one Primary's demand (the *donor*) onto a neighbour (the *recipient*). An
*edge* joins two neighbouring Primaries, and an *edge flow* is the megawatts that move along it.
Experiments 672 and 676 closed the neural-network graph-learning approach at 15 Primaries; E1-F uses
no neural network.

The terms of the line are these. **E1** is the event table of [experiment
653](../00653-edge-state-event-table-penalised-segmentation/result.md), a list of steps at each
Primary. **E1-J** adds a search for a recipient from each large fall ([experiment
670](../00670-event-table-recipient-search-on-six-cell-grid/result.md)), and **E1-J-sym** adds a
search for a donor from each rise ([experiment
680](../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md)). A
*noise unit* is a Primary's usual difference between successive 6-hour means. The long, medium, and
short cells are over 7 days, 1 to 7 days, and 1 day or less. *Deep* means 8 noise units or more, and
*shallow* means 2 to 8. *Found* means an injected event has a decoded event at the right Primary,
sign, and start. *Paired* means the decoded donor and recipient are the right two Primaries. F-hyb
and R-hyb are the donor-probe hybrids of experiment 678.

**E1-F decides every pair of opposite steps at two connected Primaries by cost.** E1-F compares the
saving in change cost with the misfit of one common flow (the donor's step and the recipient's step
must roughly match, with a prior on their ratio), and takes the largest savings first. Either end
can propose the other. The verdict is *not supported*: stop rule (a) of the decision rule closes the
edge-flow line at 15 Primaries, because E1-F misses E1-J + 0.10 in both deep cells. Stop rule (b)
also trips, because the all-pairs graph's false-transfer share is only 0.068 below E1-F's, inside
the 0.10 tolerance.

![Paired share per cell for E1-J, E1-J-sym, E1-F, E1-F on all pairs, and E1-F
rewired](figure_paired_by_cell.svg)

![Decoded transfers and correct ends per arm](figure_decoded_transfers.svg)

## Results (40 default worlds)

- **Gate passed.** E1-F with no edges has the same found share as experiment 670's E1 in every cell,
  to four decimals (long and deep 0.488).
- **Paired share:** long and deep: E1-F 0.327 (0.228 to 0.455), E1-J 0.286, E1-J-sym 0.379. Medium
  and deep: E1-F 0.272 (0.163 to 0.403), E1-J 0.245, E1-J-sym 0.316.
- **E1-F minus E1-J (95% interval):** long and deep +0.041 (-0.041 to 0.124); medium and deep +0.027
  (-0.048 to 0.113); long and shallow +0.097 (0.037 to 0.150); medium and shallow +0.090 (0.032 to
  0.154). **E1-F minus E1-J-sym:** long and deep -0.053 (-0.113 to 0.000); medium and deep -0.045
  (-0.120 to 0.029).
- **Donor megawatts found,** long and deep: E1-F 0.491, E1-J 0.441, E1-J-sym 0.481.
- **Decoded transfers and correct ends.** The false-transfer share is one minus the correct ends
  over the decoded transfers. E1-F decodes 2990 transfers with 329 correct ends (share 0.89); E1-J
  788 and 220 (0.72); E1-J-sym 1230 and 283 (0.77).
- **Placebos.** E1-F on a rewired graph pairs 0.022 and 0.053 in the two deep cells (2310 transfers,
  44 correct ends). E1-F on all pairs pairs 0.316 and 0.257 (7099 transfers, 301 correct ends, share
  0.96). E1-F on the real edges is therefore only 0.011 and 0.015 above the all-pairs graph, and its
  false-transfer share is 0.068 lower, under the 0.10 predicted.
- **Costs.** The tuning rule chose `c_edge` 2 and `delta` 8. None of the 25 pairs met the constraint
  of a false-transfer share no higher than E1-J's (0.75 on the validation worlds), so the highest
  paired share was used. All 25 pairs gave a share of 0.876 to 0.884.
- **Cross-injector check.** Tuning on constant-MW square worlds chose the same costs, so the drop is
  zero by construction. On 20 constant-MW square worlds E1-F minus E1-J is +0.036 (-0.171 to 0.303)
  long and deep and +0.302 (0.161 to 0.473) medium and deep, the one interval that excludes zero. On
  long-heavy worlds the differences are +0.003 and +0.040, and on short-heavy worlds +0.020 and
  +0.101; those intervals include zero.

## Post hoc grids (not pre-registered)

Because the planned grid gave nearly the same false-transfer share at every pair, two more grids
were run on the same validation worlds, and the best pair of each was scored on the 40 default
worlds.

- **Wide grid:** `c_edge` 8, 16, or 32 and `delta` 8, 16, 32, or 64. No pair met the constraint
  (best share 0.875). The chosen pair (`c_edge` 8, `delta` 64) pairs 0.349 and 0.293 in the two deep
  cells (+0.063 and +0.048 over E1-J, intervals -0.016 to 0.148 and -0.019 to 0.135), with 3008
  decoded transfers and 348 correct ends. In E1-F a larger cost for a step that stands alone raises
  the saving from merging, so this grid ran in the direction that cannot restrict merges.
- **Low grid:** `c_edge` 0, 0.5, or 1 and `delta` 0, 0.1, or 0.25. No pair met the constraint
  (shares 0.884 to 0.897). The chosen pair (`c_edge` 0, `delta` 0.25) pairs 0.314 and 0.257, with
  2858 decoded transfers and 307 correct ends, and the all-pairs graph pairs 0.302 and 0.174.

Costs from 0 to 64 therefore left the false-transfer share between 0.88 and 0.90. The costs of E1-F
do not separate a true pair from a coincident pair. The misfit term, the compatibility rule, and the
candidate steps were not varied.

## Real record

- **Decoded transfers** (2024-01 to 2026-06, 36.0 Primary-years): E1-F 17; E1-J 8; E1-F on the
  stored rewired graph 4, with a median of 6 over 100 rewired graphs; E1-F on all pairs 30. NGED's
  event labels are an incomplete dump, so every count of real events is a lower bound.
- **Verified D5 to D16 transfer (December 2025):** one flow with D5 as the donor (-1.08 MW) and D16
  as the recipient (+1.28 MW, a share of 1.19), the same pair as E1-J found. Both starts are late
  (D5 by 24 hours, D16 by 12.5 hours), and both ends are 24 hours late, so the pre-registered bars
  of 6 hours at the start and 12 hours at the end do not hold. The donor amplitude is under the 1.2
  to 2.2 MW range.
- **Logged shifts longer than 1 day (77 in the scored years):** 10 have an event start within 2
  hours (E1-J: 8, from experiment 670) and 21 within 24 hours (E1-J: 18).
- **Event-free time** (27.6 Primary-years more than 14 days from a logged or verified event): E1-F
  calls three transfers; E1-F on all pairs calls 11.

## Pre-registered predictions

- **Held:**
    - Long and shallow E1-J + 0.05: +0.097.
    - Medium and shallow E1-J + 0.03: +0.090.
    - Short cells no worse than E1-J - 0.03.
    - Paired share on all pairs no more than 0.03 below E1-F: 0.011 and 0.015 below.
    - E1-F on the stored rewired graph calls a third or fewer of E1-F's real-record transfers: 4
      of 17. The median over 100 rewired graphs is 6, just over a third.
    - D5 to D16: one flow with D5 as the donor, a share of 1.19, and D16 at 1.28 MW.
- **Did not hold:**
    - Long and deep paired 0.39 or more: 0.327. Medium and deep 0.35 or more: 0.272.
    - Three points above the better hybrid in both deep cells (about 0.36 and 0.33).
    - Donor megawatts found 0.54 or more: 0.491.
    - E1-F minus E1-J-sym +0.05: -0.053 and -0.045.
    - False-transfer share 0.10 below all pairs: 0.068 below.
    - Rewired pairs 0.05 or less: 0.022 and 0.053.
    - No-edges found share within 0.03 of E1-F: 0.039 and 0.044 below.
    - D5 to D16 start and end timing, and the donor amplitude of 1.2 MW or more.

## Method

- **Code:**
  `ml-experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/`,
  importing the folders of experiments 653, 660, 665, 670, and 680. The panel, residual, worlds,
  cells, and the found and paired definitions are unchanged. E1-J and E1-J-sym are read from the
  stored rows of experiments 670 and 680.
- **Steps and costs.** E1-F uses E1's steps of 2 noise units or more, plus the opposite step that
  each proposes at each connected Primary. A step that stands alone costs `c_edge + delta`. An edge
  flow costs `c_edge` plus 1 for each recipient beyond the first. The misfit is half the squared
  distance, in noise units, from the best common amplitude and share (a share between 0.3 and 2.0 on
  a 24-point grid, with a normal prior on its logarithm of mean `log 0.8` and standard deviation
  0.5). E1-F does not build the segment-boundary search with end moves, ramps, or a routine-step
  cost, and uses E1's own segment boundaries.
- **Tuning.** The costs were tuned on 60 validation worlds (seeds 20000 to 20059, events from
  2023-07-01 to 2023-12-31, readings after 2023-12-31 hidden).
- **Reviews.** A fresh Sonnet code review ran before the first scored run. The review found that a
  flow made only of proposed steps crashed `reconstruct`, that tuning counted correct ends
  differently for E1-J and E1-F, that validation truth ran past the hidden readings, that the fitted
  share was not stored, and that the fallback was not recorded; all were fixed. A bug in the
  real-record run (noise units from the wrong residual) was found afterwards and fixed before the
  real-record numbers above. A fresh Sonnet agent reviewed the results and the write-up; it found
  two wrong cost-grid figures, a wrongly marked prediction, a misleading description of the
  all-pairs comparison, and an unsupported conclusion from the wide grid, and all were corrected.
- **Compute:** CPU only, 2 cores, under 1 hour. The final-test window from 2026-07-01 is never read.
  No NGED generator is named.

## What the next experiment should answer

The remaining gap is information that the power of the 15 Primaries may not carry: which neighbour
received the load. One option is a source of that information, such as NGED's records of which
circuits were switched. This experiment does not test whether another model fitted to the same
residuals would do better.

## Files

- `figure_paired_by_cell.svg` and `figure_decoded_transfers.svg`: the figures above.
- Private tables:
  `data/ml-experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/`.
