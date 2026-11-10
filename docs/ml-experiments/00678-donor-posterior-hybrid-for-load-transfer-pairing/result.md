---
id: "00678"
slug: donor-posterior-hybrid-for-load-transfer-pairing
title: >-
  Does feeding a probe's donor alarms into the recipient search of experiment 670 pair more injected
  load-transfer megawatts than the event table alone?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
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
  threshold), the donor start is the half-hour in [alarm - 30 hours, alarm + 6 hours] with the
  most negative step size of experiment 670's `step_size`, and the end is the first half-hour at
  least 2 days after the start where the 1-day running median of the residual returns to within
  half of the step from its level before the start, or none. An alarm within 24 half-hours of an
  existing E1 donor event (downward, 4 noise units or more, unpaired) at the same Primary adds
  nothing. Each added donor is a downward event of the measured step size, then experiment 670's
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
---

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
