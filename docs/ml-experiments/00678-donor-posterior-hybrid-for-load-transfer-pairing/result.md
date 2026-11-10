---
id: "00678"
slug: donor-posterior-hybrid-for-load-transfer-pairing
title: >-
  Does feeding a probe's donor alarms into the recipient search of experiment 670 pair more injected
  load-transfer megawatts than the event table alone?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: finished
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private injected-world scores and figures under
  data/ml-experiments/00678-donor-posterior-hybrid-for-load-transfer-pairing/ only. (2) Serving
  path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible
  for turning an alarm into a donor event, so the construction is fixed below before any score is
  read. (5) Callers: this folder's scripts, which import the folders of experiments 653, 660, 665,
  670, and 675; nothing imports them. CPU only; the probe scores of experiment 675 are reused, so
  no GPU time is spent.
hypothesis: >-
  Experiment 675 found that the donor was not found for 0.87 (long and deep) and 0.84 (medium and
  deep) of the recipient megawatts that E1-J found and left unpaired, and that a logistic
  regression on the node inputs (F) finds 0.63 of the long and deep donor megawatts against E1-J's
  0.44 at the same false-donor rate. A donor alarm from such a probe, refined to a start time and
  an end time and handed to the recipient search of experiment 670 as an extra donor step, pairs
  more transfer megawatts than E1-J does. It reopens the graph line only at the level of a donor
  posterior feeding the existing search; the edge-state network of the original plan is not built
  here.
prediction: >-
  Written before any score is read; each is a mean over 3 seeds where the arm has seeds, and a seed
  more than 0.05 below its bar fails the prediction. "Paired" is the MW-weighted share of injected
  edge-transfer ends paired correctly, as in experiment 670. P1: the logistic hybrid (F-hyb) pairs
  at least 0.10 more than E1-J in the long and deep and in the medium and deep cells (E1-J: 0.29
  and 0.25). P2: the random-init-encoder hybrid (R-hyb) does the same. P3: F-hyb pairs at least
  0.05 more than E1-J in the long and shallow cell and at least 0.03 more in the medium and
  shallow cell. P4: the false transfers of a hybrid (decoded transfers matching no injected end,
  per Primary-year) are at most 1.5 times E1-J's. P5: the rewired-graph placebo pairs at most a
  third of what the same hybrid pairs on the real graph in the two deep cells. The no-edge
  placebo pairs 0 by construction and is not scored.
decision_rule: >-
  Fixed before any score is read. Worlds: experiment 675's 40 scored worlds (experiment 660's
  default generator, seeds 900 to 939, scored years 2024-01-01 to 2026-06-30); the probes are
  experiment 675's, trained on worlds from seed 100000 and never retrained, with their thresholds
  at E1-J's false-donor rate (0.625 per Primary-year). Residual panel: experiment 327's arm C, as
  in experiments 670 and 675. The final-test window from 2026-07-01 is never read. Donor
  construction: for each alarm of a probe (the peak hour of each merged run of hours above the
  threshold), the donor start is the half-hour in [alarm - 24 hours, alarm + 12 hours] (the window
  in which experiment 675 credits an alarm) with the
  most negative step size of experiment 670's `step_size`, and the end is the first half-hour at
  least 2 days after the start where the 1-day running median of the residual returns to within
  half of the step from its level before the start, or none. An alarm whose start lies within 24 half-hours of an
  E1 downward event of 4 noise units or more at the same Primary, or of a donor already added
  there, adds nothing. An open-ended donor with 180 or more days of record left is permanent, as
  in experiment 653. Each added donor is a downward event of the measured step size, then experiment 670's
  `search_recipients` runs unchanged on E1's events plus the added donors. Hybrids: F-hyb, R-hyb,
  P-M-hyb, and P-C-hyb (3 seeds each except F), and E1-J itself as the reference. The adoption
  rule: if a hybrid's mean paired share is 0.60 or more in both the long and deep and the medium
  and deep cells, the page says so and recommends adopting it as the event table. Verdict.
  `supported` if P1 holds for F-hyb or R-hyb (either one is enough) in both deep cells and P4
  holds for that hybrid; `partly supported` if P1 holds in one deep cell only, or holds for a
  hybrid for which P4 fails; `not supported` otherwise. The other predictions are reported as held
  or failed and do not enter the verdict. A deep cell with fewer than 20 events or 10 edge-transfer
  ends is unreadable and is handled as in experiment 675. If no hybrid reaches +0.10 in either
  deep cell, the graph line is closed at 15 Primaries and the edge-state network is not built. If
  a hybrid passes but pairs under 0.60, the edge-state network (a joint donor and recipient model
  with a recipient lag) is a candidate for a separate experiment. The porting threshold is zero,
  because the result is a research table that changes no production code.
benchmark: >-
  E1-J on the same 40 worlds, re-run here from experiment 670's code and the stored arm-C
  residuals, with world-resampled 95% intervals
row_set_id: >-
  experiment 660's injected worlds (default configuration, scored seeds 900 to 939, scored years
  2024-01-01 to 2026-06-30) with experiment 653's real graph of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted paired share, found share, and donor-found share per cell, with 95% intervals from
  resampling whole worlds; decoded transfers matching no injected end per Primary-year; start and
  end timing of the added donors. No forecast is scored, so the four load-shift scenarios, tail
  calibration, and the curtailment-cost metric do not apply.
code: ml-experiments/00678-donor-posterior-hybrid-for-load-transfer-pairing/
verdict: not supported
best_arm: >-
  The hybrid with donor alarms from the no-encoder logistic regression (F-hyb) in the long and deep
  cell: 0.36 of transfer megawatts paired (95% interval 0.26 to 0.48), against 0.29 for E1-J
headline: >-
  Adding a probe's donor alarms to E1-J's recipient search pairs 0.08 (F-hyb, interval -0.00 to
  +0.17) and 0.06 (R-hyb, interval +0.01 to +0.12) more of the long and deep transfer megawatts than
  E1-J, and 0.08 and 0.05 more in the medium and deep cell, against a predicted +0.10. No hybrid
  reaches the 0.60 adoption bar. The rewired-graph placebo pairs 0.04 and 0.06.
lessons: >-
  The donor alarms of a probe turn into a small, consistent gain in pairing (+0.03 to +0.08 in the
  deep cells) and none in the short cells, where the construction cannot give a donor an end. The
  gain is largest for the no-encoder logistic regression and smallest for the causal arm, so a
  neural encoder adds no pairing over the node inputs. Pairing stays far below the donor-found
  share of the probes (0.36 against 0.63 for F alone in the long and deep cell), because a donor
  must also get a start within 12 hours and an end that overlaps the truth. The false transfers
  per Primary-year rise from 0.38 (E1-J) to 0.51 (F-hyb) and 0.54 (R-hyb, mean of 3 seeds).
take_home_message: >-
  A donor probe feeding E1-J's recipient search pairs a little more than E1-J (0.36 against 0.29 of
  long and deep megawatts), not the +0.10 that was predicted, and far below the 0.60 that would
  have made it the event table. The stop rule closes the graph line at 15 Primaries: the donor
  alarm helps, but a neural encoder adds nothing over a logistic regression, and a rewired graph
  pairs almost nothing, so the connection graph does the pairing and the encoder does not.
advice:
  - >-
    Do not build an edge-state network on 15 Primaries for pairing, because every hybrid with a
    donor probe stays under +0.10 and 0.40 paired in the deep cells (paired share).
  - >-
    If a donor probe is wanted, use the logistic regression on the node inputs, because the
    neural encoders pair no more than it does (F-hyb 0.36, R-hyb 0.35, P-M-hyb 0.34, P-C-hyb 0.32
    in the long and deep cell).
  - >-
    Give an added donor an end that can be shorter than 2 days before using the hybrid on short
    events, because the short cells are unchanged by construction (paired share).
production_status: not-for-production
adopted_in: none
---

**Feeding the donor alarms of a probe into E1-J's recipient search pairs a little more transfer
megawatts, but less than the predicted +0.10, so the graph line is closed at 15 Primaries.**
Experiment 678 continues the graph line that [experiment
675](../00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/result.md)
reopened at the maintainer's request after [experiment
672](../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md) closed it, with
experiment 660's generator unchanged. The hybrid with the logistic regression's alarms (F-hyb)
pairs 0.36 of the long and deep megawatts against 0.29 for E1-J (difference +0.08, 95% interval
-0.00 to +0.17), and 0.32 against 0.25 in the medium and deep cell (+0.08, +0.03 to +0.13). The
verdict is *not supported*: no hybrid reaches +0.10 in either deep cell.

## Terms

- **F-hyb, R-hyb, P-M-hyb, P-C-hyb.** E1's event table plus one added downward donor event per
  alarm of the probe named, searched for recipients by experiment 670's search. The probes are
  experiment 675's: a logistic regression on the node inputs (F), a randomly initialised encoder
  (R), and encoders pretrained by masked reconstruction (P-M) and by causal prediction (P-C).
- **Rewired placebo.** The same hybrid on a degree-preserving rewired graph, with the draw out of
  30 that keeps the fewest real links.
- **Cell, paired, found, E1-J.** As in [experiment
  672](../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md#terms). *Paired* is
  the MW-weighted share of injected edge-transfer ends whose decoded event has the right partner.

## Figures

**The figure shows that every hybrid pairs more than E1-J in the long and deep and the medium and
deep cells, by 0.03 to 0.08, and that the logistic regression's alarms give the largest gain.** The
gain is below the +0.10 bar in every cell.

![Paired share of each hybrid minus E1-J, per cell](figure_paired_minus_e1j.svg)

## Results

- **Paired share per cell, MW-weighted, with 95% intervals from resampling the 40 worlds** (hybrids
  are means over 3 seeds, except F-hyb, which has one fit):

    | Cell | E1-J | F-hyb | R-hyb | P-M-hyb | P-C-hyb | F-hyb rewired |
    |---|---|---|---|---|---|---|
    | Long, deep | 0.29 (0.19 to 0.41) | 0.36 (0.26 to 0.48) | 0.35 (0.25 to 0.47) | 0.34 (0.24 to 0.46) | 0.32 (0.23 to 0.44) | 0.04 |
    | Medium, deep | 0.25 (0.13 to 0.38) | 0.32 (0.20 to 0.46) | 0.30 (0.18 to 0.44) | 0.27 (0.16 to 0.42) | 0.27 (0.16 to 0.42) | 0.06 |
    | Long, shallow | 0.19 (0.14 to 0.24) | 0.20 (0.15 to 0.25) | 0.21 (0.16 to 0.26) | 0.21 (0.16 to 0.27) | 0.20 (0.15 to 0.25) | 0.03 |
    | Medium, shallow | 0.13 (0.09 to 0.18) | 0.16 (0.10 to 0.22) | 0.16 (0.11 to 0.22) | 0.16 (0.11 to 0.23) | 0.16 (0.11 to 0.22) | 0.01 |
    | Short, deep | 0.08 | 0.08 | 0.08 | 0.08 | 0.08 | 0.00 |
    | Short, shallow | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.00 |

- **Differences from E1-J, with 95% intervals.** Long and deep: F-hyb +0.08 (-0.00 to +0.17), R-hyb
  +0.06 (+0.01 to +0.12), P-M-hyb +0.05 (+0.01 to +0.10), P-C-hyb +0.03 (-0.01 to +0.08). Medium
  and deep: +0.08 (+0.03 to +0.13), +0.05 (+0.01 to +0.11), +0.03 (-0.00 to +0.08), and +0.03
  (-0.00 to +0.08). Long and shallow: +0.01, +0.02, +0.03, and +0.01. Medium and shallow: +0.03 in
  every hybrid.
- **Short cells.** The hybrids equal E1-J exactly, because the added donor's end is searched from 2
  days after its start, so a donor shorter than 2 days gets a late end or none and cannot match.
- **Seeds.** R-hyb's three seeds pair 0.37, 0.36, and 0.31 in the long and deep cell, and 0.31,
  0.30, and 0.29 in the medium and deep cell, so every seed is below the +0.10 bar.
- **Placebo.** On the rewired graph F-hyb pairs 0.04 and 0.06 in the two deep cells and R-hyb 0.05
  and 0.06, so the real connection graph supplies almost all of the pairing (E1-J rewired: 0.04
  and 0.06). The empty graph pairs 0 by construction and was not scored.
- **False transfers.** Decoded transfers that match no injected end per Primary-year: E1-J 0.38;
  F-hyb 0.51; R-hyb 0.54 (0.41 to 0.62 over seeds); P-M-hyb 0.58; P-C-hyb 0.47.

## Predictions

- **P1 (F-hyb or R-hyb at least +0.10 in both deep cells): failed.** F-hyb +0.08 and +0.08, R-hyb
  +0.06 and +0.05.
- **P2 (R-hyb at least +0.10): failed**, as above.
- **P3 (F-hyb at least +0.05 in the long and shallow cell and +0.03 in the medium and shallow
  cell): failed** (+0.01 and +0.03; the second is +0.027, short of +0.03).
- **P4 (false transfers at most 1.5 times E1-J's, 0.57): held** for F-hyb (0.51), R-hyb (0.54),
  and P-C-hyb (0.47), and failed for P-M-hyb (0.58).
- **P5 (the rewired placebo pairs at most a third of its hybrid in the deep cells): held**
  (0.04 against 0.36, and 0.06 against 0.32).
- **Adoption bar (0.60 in both deep cells): not reached.** The best hybrid pairs 0.36 and 0.32.
- **Stop rule fires.** No hybrid reaches +0.10 in either deep cell, so the graph line is closed at
  15 Primaries and the edge-state network is not built.

## Checks

- **Code review.** A fresh Sonnet reviewer found no use of injected truth in the donor
  construction. Before any score was read, the plan and code were changed on its findings: the
  start-search window now matches the window in which experiment 675 credits an alarm (24 hours
  before to 12 hours after), an alarm near any E1 downward event of 4 noise units or more is
  skipped whether or not E1 paired it, a donor counts as permanent at 180 or more days of record
  left (experiment 653's rule), and donors added by earlier alarms count for the near-duplicate
  rule.
- **Known biases.** The added donor's end is late by up to about 12 hours and is searched from 2
  days after the start. Donor-found and false-transfer counts include added donors under 4 noise
  units, which the search ignores but the scoring can match. The real-graph hybrid and its
  rewired placebo have different added-donor lists, because E1's pairing differs on the two
  graphs. A post hoc rule for short events was not tried.
- **Audit.** AUDIT_PLACEHOLDER
- **Caveats.** F-hyb is one fit, and the intervals resample worlds only, not fit variance. NGED's
  logs may miss load shifts, so matched logged shifts on the real record are lower bounds, and the
  injected start hours come from NGED's logged onsets up to 2026-07, as in experiments 670, 672,
  and 675. The probes are experiment 675's, trained on 200 worlds.

## What to try next

- The pre-registered stop rule closes the graph line at 15 Primaries. A donor probe is worth
  testing again only with a donor end that can be shorter than 2 days, and on the fleet-scale
  adjacency of about 1,161 Primaries; both are for the roadmap to decide.

## The plan as committed

The plan below was committed before any score was read, and was changed once afterwards, on the
code review above and before any score was read.

**This is the plan, committed before any score is read.** It continues the graph line that
[experiment 675](../00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/result.md)
reopened, and which [experiment 672](../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md)
had closed under its stop rule. Experiment 675 found that masked pretraining did not help, and that
neither did causal pretraining as designed (the causal arm finds 0.26 and 0.20 fewer donor
megawatts than the random-init encoder in the two deep cells). The graph Discuss session therefore
asked for the cheapest test of the finding that matters: a donor probe finds more long and deep
donor megawatts than E1-J, so does that gain turn into more pairs?

## The question

Experiment 675's step 0a showed that the donor, not the link to the recipient, is where E1-J loses
the megawatts it finds at the recipient. The no-encoder logistic regression (F) found 0.63 of long
and deep donor megawatts against E1-J's 0.44 (one fit, no interval for E1-J), and the random-init
encoder (R) found 0.59 (seeds 0.46 to 0.70). The question here is whether adding those donors to
E1-J's search lifts the paired share. F is the non-neural baseline that any neural donor head has
to beat, so it is run beside R.

## Method

- **Probe scores.** Experiment 675's stored probe scores for the 40 scored worlds: F (1 fit) and R,
  P-M, and P-C (3 seeds each). No probe is retrained, and no GPU time is used.
- **From an alarm to a donor event.** Each alarm becomes a downward donor event whose start and end
  are measured on the arm-C residual by the rule in the decision rule above, with the size from
  experiment 670's `step_size`. An alarm near an existing unpaired E1 donor adds nothing.
- **Recipient search.** Experiment 670's `search_recipients`, unchanged, on E1's events plus the
  added donors, on the real graph. Placebos: the degree-preserving rewired graph of experiment 670
  (`rewire_low_overlap`, the draw keeping the fewest real links, with the share kept reported) and
  an empty graph.
- **Scoring.** Experiment 653's `match` and experiment 665's cell statistics, as in experiment 670,
  with world-resampled intervals. Decoded transfers matching no injected end are counted per
  Primary-year.
- **Reviews and running.** A fresh Sonnet code review before the first run (checking that no
  injected truth enters the donor construction and that the probe scores are the stored ones), a
  results audit, a write-up review, and a prose review. CPU work runs under `systemd-run --user
  --scope -p MemoryMax=16G -p OOMPolicy=continue`.
- **Inherited caveats.** NGED's logs may miss load shifts, so matched logged shifts on the real
  record are lower bounds, and the injected start hours come from NGED's logged onsets up to
  2026-07, as in experiments 670, 672, and 675.
