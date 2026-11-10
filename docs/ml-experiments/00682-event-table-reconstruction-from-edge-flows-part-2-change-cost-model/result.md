---
id: "00682"
slug: event-table-reconstruction-from-edge-flows-part-2-change-cost-model
title: >-
  Does choosing edge flows by a fixed cost per change pair more injected load-transfer megawatts
  than the searches of experiments 670 and 680, without adding more false transfers?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private score tables (parquet and JSON) and figures under
  data/ml-experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/
  only. (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: a
  reconstruction by edge flows has more than one defensible form, so its costs, grid, and tuning
  rule are fixed below before any score is read. (5) Callers: this folder's scripts, which import
  the folders of experiments 653, 660, 665, 670, and 680; nothing imports them. CPU only, 2 cores.
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
  R-hyb (0.35). Medium and deep: at least 0.35, and at least 0.03 above the better hybrid
  (about 0.33). Donor megawatts found in the long and deep cell at least 0.54. Long and shallow:
  E1-J + 0.05. Medium and shallow: E1-J + 0.03. Short cells: no worse than E1-J - 0.03. E1-F
  minus E1-J-sym at least +0.05 paired in both deep cells. E1-F minus E1-F all pairs: a
  false-transfer share at least 0.10 lower in both deep cells, at a paired share no more than
  0.03 lower. E1-F rewired pairs 0.05 or less in both deep cells and calls a third or fewer of
  E1-F's real-record transfers. E1-F no edges has a found share within 0.03 of E1-F's in the deep
  cells. On the real record the verified D5 to D16 transfer is one flow with D5 as the donor, both
  starts within 6 hours of the verified onsets (D5 2025-12-29 01:00, D16 about 15 hours later),
  the end within 12 hours of 2026-01-13 11:30, 1.2 to 2.2 MW at each end, and a share from 0.8
  to 2.0. The author of the spec put the chance that E1-F beats E1-J by +0.10 in both deep cells
  at about 35%, and the chance that it beats the better hybrid by +0.10 at about 15%.
decision_rule: >-
  Fixed before any score is read. Worlds, residual, cells, and the found and paired definitions
  are those of experiments 670 and 680. Intervals resample whole worlds. A cell with fewer than 20
  events or 10 edge-transfer ends is too small to read. Tuning. The change costs c_edge (2, 4, 6,
  8, 12) and delta (0.5, 1, 2, 4, 8) are chosen on 60 validation worlds whose events start from
  2023-07-01 to 2023-12-31 (seeds 20000 to 20059, readings after 2023-12-31 hidden). The chosen
  pair maximises the pooled paired share of the two deep cells among the pairs whose
  false-transfer share (one minus correct ends over decoded transfers) is no higher than E1-J's on
  the same worlds, with ties broken towards larger delta. If no pair meets the constraint, the
  pair with the highest paired share is used and the page says so. Stop rules, applied to the
  point estimate on the scored worlds. (a) Close the edge-flow line at 15 Primaries if E1-F misses
  E1-J + 0.10 paired in both deep cells; (b) close it as well if E1-F rewired pairs more than
  half as much as E1-F in a deep cell, or E1-F all pairs matches E1-F on the false-transfer
  share (the topology is not identified). (c) If E1-F passes in at least one deep cell, a learnt
  noise model (ceiling 1 GPU-hour) may be planned, and must add +0.03 paired in that cell. (d) If
  E1-F passes in one deep cell only, use E1-F for that cell's lengths and E1-J-sym for the rest.
  Verdict. `supported` if both deep differences from E1-J reach +0.10 and the topology checks
  hold. `partly supported` if one reaches +0.10, or both reach +0.05. `small gain` if both deep
  intervals exclude zero but neither reaches +0.05. `not supported` otherwise. The porting
  threshold is zero, because the result is a research table. Gate. The experiment reads E1-J and
  E1-J-sym from experiments 670 and 680 and recomputes E1 from the same code, so the gate is that
  E1-F no edges reproduces experiment 670's E1 found share per cell within 0.005. A cross-injector
  check tunes on the constant-MW square worlds and scores the default worlds, and the reverse; the
  paired-share drop must be 0.05 or less.
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
  verified D5 and D16 transfer, the 8 transfers E1-J calls, and logged shifts matched.
code: ml-experiments/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/
production_status: not-for-production
adopted_in: none
---

**This is the plan of part 2: a reconstruction of the event table by edge flows, chosen by a fixed
cost per change.** Experiments 672 and 676 closed the neural-network graph-learning approach at 15
Primaries. This experiment is a different model: it uses no neural network, and it explains pairs of
steps by one flow on an edge.

## Question

A *Primary substation* steps electricity down to a local distribution network. A *load transfer*
moves demand from one Primary (the *donor*) to a neighbour (the *recipient*). [Experiment
680](../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md)'s
searches accept a pair when the two steps roughly conserve megawatts. They add many pairs that match
no injected transfer. This experiment asks whether deciding every pair jointly, by the change cost
saved against the misfit of one common flow, accepts fewer false pairs for the same true ones.

## Model (E1-F)

- **Steps.** E1's steps of 2 noise units or more, plus, for each, the opposite step it proposes at
  each connected Primary (so either end lifts the other).
- **Cost of a state.** A step that stands alone costs `c_edge + delta`. An edge flow costs `c_edge`
  however many ends it serves, plus 1 for each recipient beyond the first. The misfit of an edge
  flow is half the squared distance, in noise units, of the donor step and the summed recipient step
  from the best common amplitude and share (donor `-a`, recipients `+rho * a`), plus half the
  squared distance of `log rho` from a normal prior (mean `log 0.8`, standard deviation 0.5). A
  proposed step that is not merged costs the evidence the data hold for it.
- **Search.** Greedy: accept the pair (or one donor with up to three recipients) with the largest
  saving, remove its steps, repeat until no saving is positive. Amplitudes are closed form and are
  not shrunk.
- **Arms.** E1-F (real edges); E1-F rewired (the draw of 30 that keeps the fewest real links); E1-F
  no edges; E1-F all pairs (every pair of Primaries may connect). References: E1-J and E1-J-sym
  (experiments 670 and 680), and F-hyb, R-hyb, and 676-G.

## Plan

- **Scored worlds:** the 40 default worlds of experiments 670 and 680. Robustness: 20 worlds each of
  the long-heavy, short-heavy, and constant-MW square configurations.
- **Not in this model:** ramps, a separate routine-step cost (E1's routine steps are kept as
  candidates), a lag longer than 36 half-hours, and the optional learnt noise model (part 3 of the
  brief, run only under stop rule (c)). The model in the spec that is not built here is the
  segment-boundary search with end moves; E1's own segment boundaries are used instead.
- **Real record:** the checks of the plan above, the 8 transfers E1-J calls (kept, dropped, or
  reassigned), the new transfers, the counts on the rewired and all-pairs graphs, the 77 logged
  shifts longer than 1 day, and the D3 and D22 case of October 2024, the D10 episode of December
  2025, and the D12 step of 8 December 2025 as reports only. NGED's event labels are an incomplete
  dump, so every count of real events is a lower bound.
- **Standing practices:** cleaned data, the D15 and D18 cut, and the final-test window exclusion are
  inherited from experiment 653's panel. No XGBoost model is fitted. No NGED generator is named.
- **Compute:** about 4 seconds per validation world for the whole 25-pair grid, and about 10 seconds
  per scored world for four arms, so under 30 minutes on 2 cores. GPU: none.
- **Reviews:** one fresh Sonnet code review before the first scored run, one Sonnet review of the
  results, one Sonnet review of the write-up and prose. No Opus review.
