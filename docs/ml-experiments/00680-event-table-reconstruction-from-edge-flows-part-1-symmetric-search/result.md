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
status: finished
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private score tables (parquet and JSON) and figures under data/ml-
  experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/ only. (2)
  Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: the donor search has
  one obvious mirror of experiment 670's search, whose thresholds are fixed below before any score
  is read; the later edge-flow model (part 2) admits more than one design and gets its own plan.
  (5) Callers: this folder's scripts, which import the folders of experiments 653, 660, 665, and
  670; nothing imports them. CPU only, 2 cores.
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
  Fixed before any score is read. Worlds, residual, cells, and the found and paired definitions
  are those of experiment 670 (and so of experiment 665). Intervals resample whole worlds. A cell
  with fewer than 20 events or 10 edge-transfer ends is too small to read, and a prediction that
  needs it counts as not holding. Stage 1 rule, applied to the point estimate of E1-J-sym minus
  E1-J in the paired share. (a) If it is +0.10 or more in both the long and deep and the medium
  and deep cells, E1-J-sym becomes the event table, the page says so, the event-table summary page
  is updated, and the edge-flow reconstruction model (part 2) is run only if the maintainer asks.
  (b) If it is under +0.03 in both deep cells, the either-end-seeds mechanism is weak, and part 2
  is run all the same, because it also tests the change-cost objective, with a stated prior of 20%
  or less. (c) Otherwise part 2 is run. Verdict. `supported` if both deep differences reach +0.10
  and the rewired graph holds. `partly supported` if one deep difference reaches +0.10, or both
  reach +0.05. `small gain` if both deep differences have intervals excluding zero but neither
  reaches +0.05. `not supported` otherwise. The porting threshold is zero, because the result is a
  research table that changes no production code. A reproduction gate must pass first, or the
  verdict is `invalid`. Gate. Re-running E1-J on the 40 default worlds reproduces experiment 670's
  stored paired shares per cell within 0.005.
benchmark: >-
  E1-J, experiment 670's event table with its recipient search, on the same 40 default worlds
  (seeds 900 to 939, scored years 2024-01-01 to 2026-06-30), read from experiment 670's stored
  rows and re-run here as the gate.
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
verdict: partly supported
best_arm: >-
  E1-J-sym on the long and deep cell: 0.38 of injected transfer megawatts paired (95% interval
  0.28 to 0.50), against 0.29 for E1-J (0.20 to 0.40); the difference is +0.094 (0.043 to 0.151)
headline: >-
  Starting the search from each unpaired rise as well as from each fall raises the paired share of
  injected transfer megawatts by 0.094 in the long and deep cell and 0.071 in the medium and deep
  cell, short of the predicted +0.10 in both. The symmetric search adds 442 decoded transfers (788
  to 1230) for 63 more correct ends (220 to 283), and a rewired graph removes the gain.
lessons: >-
  The donor end is only partly recoverable by searching from the recipient. Donor megawatts found
  in the long and deep cell rise from 0.44 to 0.48, not to the predicted 0.54, which suggests most
  of the gain is pairing of donors the event table already held. Most of the added transfers match
  no injected end (63 correct ends for 442 added transfers).
take_home_message: >-
  Searching from each unpaired rise for the matching fall at a neighbouring Primary pairs 0.38 of
  the injected transfer megawatts in the long and deep cell, against 0.29 for the one-way search,
  a gain of 0.09. The gain of 0.09 disappears on a rewired graph (0.057 paired), but its interval
  includes zero in one of three other generator configurations. It falls just short of the 0.10
  needed to adopt the search, and the search adds 442 decoded transfers, most of which match no
  injected end. The donor is still found for only 0.48 of its megawatts, so the joint
  reconstruction model of part 2 has room to add something, and the share of decoded transfers
  that match no injected end is the number to watch.
advice:
  - >-
    Compare any new pairing method with E1-J-sym rather than E1-J alone, because the symmetric
    search adds +0.09 paired share in the long and deep cell (paired share table).
  - >-
    Report decoded transfers beside correct ends, because E1-J-sym adds 442 decoded transfers for
    63 more correct ends (decoded transfers).
production_status: not-for-production
adopted_in: none
---

**Searching from each unpaired rise for the matching fall at a neighbouring Primary pairs 0.38 of
the injected transfer megawatts in the long and deep cell, against 0.29 for the one-way search of
experiment 670. The gain of 0.09 is just under the 0.10 needed to adopt the search, and the search
adds 442 decoded transfers, most of which match no injected end.** A *Primary substation* steps
electricity down to a local distribution network. A *load transfer* moves part of one Primary's
demand (the *donor*) onto a neighbour (the *recipient*). Experiments 672 and 676 closed the
neural-network graph-learning approach at 15 Primaries; the search in this experiment uses no neural
network. A *noise unit* is a Primary's usual difference between successive 6-hour means. The long,
medium, and short cells are over 7 days, 1 to 7 days, and 1 day or less, and *deep* means 8 noise
units or more, as in experiment 670. *Found* means an injected event has a decoded event at the
right Primary, sign, and start; *paired* means the decoded donor and recipient are the right two
Primaries.

**E1-J-sym adds the mirror search: start at each unpaired rise, and look for the matching fall at a
neighbour.** [Experiment 670](../00670-event-table-recipient-search-on-six-cell-grid/result.md)'s
search (E1-J) starts at a large fall and looks for the rise at a neighbour. [Experiment
675](../00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/result.md)
found that the donor is the bottleneck. This experiment adds the mirror search (E1-J-sym): it starts
at each rise of 4 noise units or more that is still unpaired, and looks for a fall of 2 noise units
or more that starts up to 36 half-hours before that rise (the window is widened by 6 half-hours on
each side to allow for start error). The verdict is *partly supported*: both deep-cell differences
exceed +0.05, neither reaches +0.10. The pre-registered stage 1 rule (adopt at +0.10 in both deep
cells, otherwise continue) therefore sends the line on to part 2, a model that reconstructs both
Primaries' residuals jointly from flows on the edges between Primaries.

![Paired share per cell for E1-J, E1-J-sym, and E1-J-sym on a rewired
graph](figure_paired_by_cell.svg)

## Results

- **Gate passed.** E1-J, re-run here on the same 40 default worlds (seeds 900 to 939), reproduces
  experiment 670's paired share in every cell to four decimals.
- **Paired share, E1-J-sym minus E1-J (95% interval from resampling worlds):** long and deep +0.094
  (0.043 to 0.151); medium and deep +0.071 (0.025 to 0.139); long and shallow +0.076 (0.038 to
  0.117); medium and shallow +0.028 (0.004 to 0.061); short and deep +0.003 (0.0 to 0.011); short
  and shallow +0.003 (0.0 to 0.011).
- **Absolute paired share, E1-J-sym:** long and deep 0.379 (0.281 to 0.500); medium and deep 0.316
  (0.186 to 0.478); long and shallow 0.263; medium and shallow 0.159; short and deep 0.083; short
  and shallow 0.030.
- **Found share barely moves** (long and deep 0.503 to 0.515). Donor megawatts found in the long and
  deep cell rise from 0.441 to 0.481, and in the medium and deep cell from 0.399 to 0.434. Most of
  the gain is therefore pairing of donors that the event table already held.
- **Decoded transfers.** E1-J-sym decodes 1230 transfers with 283 correct ends, against 788 and 220
  for E1-J. The 63 additional correct ends come with 442 additional decoded transfers.
- **Placebo.** On a rewired graph E1-J-sym pairs 0.057 (long and deep) and 0.064 (medium and deep),
  with 730 decoded transfers and 39 correct ends. The gain depends on the real graph.
- **Robustness.** On 20 long-heavy worlds the long and deep difference is +0.043 (0.019 to 0.072);
  in short-heavy worlds it is +0.018 (0.0 to 0.054), whose interval includes zero, and in
  constant-MW square worlds +0.083 (0.010 to 0.193). The rest is in `injected_by_cell.json`.

## Real record

- **Decoded transfers** (2024-01 to 2026-06, 36.0 Primary-years): 8 for E1-J, 12 for E1-J-sym, 0 for
  E1-J-sym on the stored rewired graph (median 2 over 100 rewired graphs, which keep 15.7% of the
  real links), or 0.22 and 0.33 per Primary-year for E1-J and E1-J-sym.
- **Verified D5 to D16 transfer (December 2025):** kept in the right direction, with the same late
  starts and ends as E1-J (D5 start 24 hours late, D16 start 12.5 hours late, both ends 24 hours
  late), so the symmetric search adds nothing there.
- **Logged shifts longer than 1 day (77 in the scored years):** 9 of 77 have an event start within 2
  hours (E1-J: 8) and 19 within 24 hours (E1-J: 18). NGED's event labels are an incomplete dump, so
  every count of real events here is a lower bound.
- **Event-free time:** neither method calls a transfer in the 27.6 Primary-years more than 14 days
  from any logged or verified event.

## Pre-registered predictions

- Held: long and shallow +0.03 (got +0.076); medium and shallow +0.02 (+0.028); short cells no worse
  than -0.03; rewired graph at a third or fewer of the real record's transfers; D5 to D16 kept.
- Did not hold: long and deep +0.10 (+0.094); medium and deep +0.10 (+0.071); donor megawatts found
  0.54 or more (0.481).

## Method

- **Code:**
  `ml-experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/`,
  importing the folders of experiments 653, 660, 665, and 670. Unchanged from experiment 670: the
  panel, the residual, the worlds, the cells, and the found and paired definitions.
- **Thresholds,** fixed in the plan: rise at least 4 noise units; donor fall at least 2 noise units;
  donor start from 42 half-hours before to 6 half-hours after the rise; share 0.3 to 2.0; loose
  conservation cost at most 2.5. No tuning was done.
- **Reviews.** A fresh Sonnet code review ran before the first scored run. It found that the gate
  only printed, that the stage 1 branch letters were swapped against the plan, that the verdict rule
  was not encoded, and that the conservation cost used the found donor's step where experiment 670
  uses the anchor event's step. All four were fixed before the scored run.
- **Compute:** CPU only, about 7 seconds per world on one core, under 10 minutes in all.
- **Not used:** the final-test window from 2026-07-01 is never read, and no NGED generator is named.

## What the next experiment should answer

Whether a joint reconstruction of both Primaries' residuals by an edge flow, with a fixed cost per
change, pairs more megawatts than E1-J-sym without adding as many transfers that match no injected
end (part 2).

## Files

- `figure_paired_by_cell.svg`: the figure above.
- Private tables:
  `data/ml-experiments/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/`.
