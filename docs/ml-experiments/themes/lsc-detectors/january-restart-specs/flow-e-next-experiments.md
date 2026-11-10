# Spec: FLOW-E next experiments

**This is a copy of the answer that an Opus sub-agent gave the Discuss session on 2026-11-09,
trimmed of conversation.** Every number in it is the sub-agent's own analysis of the stored outputs
of [experiment
651](../../../00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1/result.md)
and is unaudited. Scores are CRPS in points of capacity, on event-free rows at 24 to 120 hours,
in the development months 2025-10 and 2025-12 only. No row on or after 2026-07-01 was read. None of
the three proposals has run.

## Take-home

- **FLOW-E's only measured cost comes almost entirely from one probable unlabelled event.** I
  recomputed FLOW-E minus G0 at the six MW Primaries as +0.022 (the page gives +0.021). Five
  Primary-days at D10, 8 to 12 December 2025, carry 62% of that excess. Dropping D10's rows for 1 to
  16 December 2025 (5% of the six MW rows) leaves +0.003, and the two seeds then disagree in sign,
  as they do for every other arm.
- **In those days D10 runs 15 to 21 points of capacity above G0's median for about ten days, and the
  forecasts of its electrical neighbours are off too.** D12 runs 25 to 34 points above from 8
  December, D15 13 to 60 points above, and D11 about 5 points below from 3 to 7 December. NGED's
  labels mark every one of those D10 rows event-free. FLOW-E lowers D10's median by about 0.7 points
  there, which is what a normal-running-arrangement (NRA) forecast should do if this was a transfer.
  I cannot verify that it was a transfer.
- **The same episode also inflates the class 2 recipe's "excess over A1".** G0 minus A1 at the six
  MW Primaries is +0.141 on all rows and +0.078 without D10's rows for 1 to 16 December. D10 itself
  goes from +0.45 to −0.16. A1 follows load shifts most (0.119 on bank M), so A1 scores best on an
  unlabelled shift. The remaining excess sits mostly at D16 (+0.29 in October, +0.32 in December).
- **FLOW-E's follow gain needs a metered mirror.** On bank X (no mirror) FLOW-E follows 0.059,
  exactly G0's figure. Its whole gain on banks M and S (0.038 against G0's 0.059 and C's 0.044 and
  0.046) comes from reading the neighbour that received the load.
- **So FLOW-E has not yet been shown to cost anything on ordinary days, and it holds follow.** The
  cheapest informative step is the confirmation run that experiment 651's gate cancelled. That run
  needs a rewired-graph placebo and the probable unlabelled event handled the same way for every
  arm.

## Brief facts checked

| Claim in the brief | Check |
|---|---|
| Antisymmetric MW flow on each of 29 electrical edges; per-node correction = minus net outflow over capacity, times a learnt scale starting at zero | Correct (`e00651_nbmodel.flow_correction`). The flow neural network reads only each end's **two current gap values** (day lag and week lag), plus the log capacity ratio, and no gap history as the design proposed. The correction is hidden on 50% of training rows. |
| Auxiliary loss on the known synthetic flow, weight 0.1 | **Not quite.** The loss is on the per-node lag correction, not the edge flow. It is (correction + injected lag shift)² / 0.2², and it applies on every training row with a present lag: target zero on unshifted rows, on all rows whether or not messages are hidden. |
| Two months, two seeds, fine-tuning only; pretraining from experiment 591's chain | Correct. The graph layer exists only in step 3. |
| Follow on bank M 0.038 against 0.044; six MW CRPS 2.479 against 2.458 (+0.021, interval +0.010 to +0.032); q99 pinball 0.196 against 0.188 | Correct (day-block 95% interval). The FLOW-E minus C leave-one-Primary-out contrast never flips: +0.017 to +0.028. |
| The six MW excess over A1 (0.13) is the same for G0 | Correct as stated, but about 45% of the excess is the D10 episode above. |
| Seed noise exceeds every contrast except FLOW-E's | Correct on all rows. Without the D10 episode it exceeds FLOW-E's as well. |

**Training data of experiment 651** (from `e00418_prep`, `e00448_run.pretrain_rows`, `e00574_fit`,
`e00397_prep`, and the logs of experiment 591):

- **Step 1, pretraining:** estimates of past weather (ERA5 means and CAMS shortwave) in pseudo-runs
  every 2 days from 2019-09-01, at leads of 0.5 to 351 h. Five Primaries start later after
  early-calibration cuts (D11 2019-09-17, D10 2019-12-01, D5 2020-02-01, D14 2020-07-01, D16
  2021-02-01). Each fold has 7.0 to 7.9 million rows, ending at the start of the fold's month minus
  the cutoff age.
- **Step 2:** a fresh mixer fitted on the ECMWF ENS history from 2024-04-01 to the cutoff, with a
  3-day embargo.
- **Step 3, the step in which FLOW-E was trained:** a daily fine-tune refitted on days 0 and 14 of
  each fold, on ENS rows with E3 rows and experiment 581's lag-only synthetic shifts.
- **Not used:** the 2021-03 to 2024-03 ENS backfill from Source Cooperative.

## Ranked proposals

### 1. FLOW-C: confirm FLOW-E on all five development months with a rewired placebo (run first)

- **Question.** Does FLOW-E keep a lower follow than G0 and the Candidate (C) across five months,
  and does the real electricity-network wiring beat a rewired graph?
- **Architecture changes.** None to FLOW-E. Add FLOW-R, which is FLOW-E on a degree-preserving
  rewired graph, using experiment 651's GAT-R rewiring. Store the per-node correction (`flow_scale ×
  share`) per row, so that it can be scored as a detector.
- **Training data.** The same as experiment 651, so that G0 and C stay matched. The full 2019-onward
  ERA5 and CAMS history is already in step 1. This experiment adds none of it to the flow layer,
  because that would confound the confirmation: proposal 2 tests the full history.
- **Arms.** G0, FLOW-E, and FLOW-R, fitted on 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06 with
  seeds 0 to 2. The stored G0 and FLOW-E fits for 2025-10 and 2025-12 at seeds 0 and 1 are reused. C
  comes from the stored seed 0 to 2 forecasts of experiments 612 and 613. Banks M, S, and X.
- **Unlabelled-event screen, fixed before any score.** One Primary-day mask for every arm, built
  from observed power and existing detectors only and never from any arm's residual. An example is
  K1's G4 gate or the refit-offset neighbour-count detector (RO-NC) firing on 3 or more consecutive
  days. The unscreened score is primary and the screened score is reported beside it. Report D10 for
  1 to 16 December 2025 as its own descriptive row.
- **Predictions that could fail.**
    - Pooled over five months, FLOW-E minus G0 follow on bank M is at most −0.015, with the
      month-block 95% interval excluding zero.
    - FLOW-E minus C on bank M is at most −0.005, in at least 4 of the 5 months.
    - On bank X, FLOW-E is within 0.005 of G0. This is a known-answer check: with no mirror there is
      nothing to read.
    - FLOW-R follows at least 0.010 more than FLOW-E on bank M.
    - Six MW event-free CRPS: the upper 95% bound of FLOW-E minus G0 is below +0.044 on the screened
      rows, and the point estimate is at most +0.01.
    - The correction scores an area under the curve (AUC) of at least 0.75 on injected lag-days of
      bank M against K1's G4.
- **Stop rule.** Stop the FLOW line if either of these holds:
    - the FLOW-E minus G0 follow interval on bank M includes zero;
    - FLOW-R is within 0.005 of FLOW-E. In that case write "the mirror helps, the wiring does not",
      and proposals 2 and 3 do not run.

    Report a CRPS failure on the screened rows as "not non-inferior". Proposal 3 may still run on
    the correction alone in that case.
- **GPU-hours.** Experiment 651 cost about 0.125 GPU-h per arm-month-seed including three banks.
  That gives 37 new arm-month-seeds, about 4.6 GPU-h, plus about 0.5 GPU-h for C on banks S and X.
  **Ceiling 6 GPU-h**, one GPU process at a time. Time one FLOW-R cutoff first. Outputs about 8 GB,
  so delete the S and X bank parquet once they are scored.
- **Dependencies.** Experiment 651's code (arm, grouped bank, scorer, gate) and the experiment 612
  and 613 stored forecasts. The GPU is idle as of this writing.
- **Improves:** the NRA forecast (follow) directly. The prevailing forecast gains only through the
  stored correction, which proposal 3 uses.

### 2. FLOW-P: train the flow layer on the full 2019-onward history, with gap histories as inputs (only if proposal 1 passes)

- **Question.** Does a flow layer trained from step 1 onward, on the full 2019-onward ERA5 and CAMS
  history with mirrored synthetic shifts, and reading several days of gap history, follow less than
  FLOW-E on banks M and S without costing anything on ordinary days?
- **Architecture changes.**
    - The flow layer and its auxiliary loss start in step 1 and continue through steps 2 and 3.
    - The flow neural network reads each end's gap at the day lag and the week lag for the last 7
      days (14 values per end, instead of 2), a MW-or-MVA flag per end, and the log capacity ratio.
    - Inject experiment 581's mirrored synthetic shifts into the pretraining rows at 5% of each
      Primary's time, as step 3 already does.
- **Training data.** The full 2019-onward ERA5 and CAMS history in step 1, about 7 million rows per
  fold, against the fine-tuning window alone in experiment 651. Step 2 stays on ENS from 2024-04 so
  that the arms stay matched. The ENS backfill from 2021 is a separate question. Two caveats:
    - In pretraining the gap signal comes from a twin forecast driven by ERA5, which is less noisy
      than the ENS-driven twin used at forecast time. Scoring stays on ENS rows.
    - Unlabelled real shifts in the 2019 to 2024 targets teach every arm to follow. G0-P below
      controls for that.
- **Arms.** FLOW-E (from proposal 1); FLOW-P (pretrained, two current gaps as inputs); FLOW-PH
  (pretrained, 7-day gap histories); and G0-P (the same pretraining bank of mirrored shifts, no flow
  layer). G0-P tests whether the synthetic bank in pretraining does the work by itself. Five months,
  seeds 0 and 1, the same screen as proposal 1.
- **Predictions that could fail.**
    - FLOW-PH follows at most 0.030 on bank M and at most 0.032 on bank S (FLOW-E: 0.038 and 0.038).
    - FLOW-PH minus FLOW-E on bank M is at most −0.005, with the month-block interval excluding
      zero.
    - G0-P stays within 0.005 of G0 on bank M.
    - Six MW event-free CRPS of FLOW-PH minus G0 has an upper 95% bound below +0.044 on the screened
      rows.
- **Stop rule.**
    - If FLOW-PH does not beat FLOW-E on bank M by 0.005, keep FLOW-E as it is and stop changing its
      inputs.
    - If G0-P matches FLOW-PH, the gain comes from the synthetic bank, not from the flow layer, and
      the flow layer is dropped.
- **GPU-hours.** Steps 1 and 2 for 3 arms × 5 months × 2 seeds come to about 30 units at about 0.22
  GPU-h with the grouped bank, which is about 6.5 GPU-h. **Ceiling 8 GPU-h.** Time one step-1 fit
  first and stop if it runs over 15 GPU-minutes.
- **Dependencies.** Proposal 1 passes. A new grouped loader for the pretraining rows is the main
  code to write, and the leak test of experiment 651 is extended to it.
- **Improves:** the NRA forecast.

### 3. FLOW-H: the FLOW-E correction as evidence for the prevailing forecast (CPU first; only if proposal 1 keeps a working correction)

- **Question.** Does the neighbour-mirror evidence that FLOW-E's correction carries improve the
  prevailing-conditions forecast of experiment 652, scored by the state of the shift at the origin?
- **Architecture changes.**
    - Add the stored per-node correction for the last 7 days, and a "mirror present" summary (the
      correction's absolute value over the own gap), as inputs to 652's R-haz head.
    - Add the same two inputs to a B-haz variant whose survival curve is fitted separately for
      shifts with a mirror and shifts without one (no neural network).
    - Experiment 652's head is a few thousand weights and is fitted on the CPU.
- **Training data.** Experiment 652's banks (experiment 650's extended generator, whose donor and
  receivers are conserved across real electrical neighbours, with no neighbour in 15% of shifts) and
  K1 on all 5 development months. FLOW-E inference runs on those banks, on the CPU as K1's did in
  experiment 650. The full history is not needed, because the head reads K1 and FLOW-E outputs only.
- **Arms.** R-haz, R-haz+FLOW, B-haz, B-haz-mirror, R0-168, a placebo (R-haz+FLOW with the
  correction shuffled among the same Primary's origins), and K1 as context. 2 seeds.
- **Scoring.** As experiment 652: event-free, far-from-end, middle, near-end, and just-ended origin
  strata, by lead day 1 to 9, scaled and unscaled, with the one veto (event-free cost against K1 at
  most 0.03).
- **Predictions that could fail.**
    - With a shift in force (strata pooled), R-haz+FLOW minus R-haz scaled CRPS is at most −0.10,
      with the month-block interval excluding zero, on shifts that have a mirror.
    - On shifts with no neighbour, the same contrast is within ±0.05. The prediction holds only if
      that placebo-by-construction holds too.
    - On event-free origins, R-haz+FLOW is no worse than R-haz by more than 0.02.

    My prior that the first prediction holds is low, at about 25%. A shift's end reaches both ends'
    lags at the same delay of at least 24 h, so the mirror cannot reduce the near-end cost. Its only
    route to a gain is a better estimate of the depth, and fewer false departures.
- **Stop rule.** If the in-force gain is above −0.05 or the placebo moves the score by as much,
  record that the electricity-network mirror does not help the prevailing forecast and close the
  FLOW line for the prevailing forecast.
- **GPU-hours.** 0 planned. If FLOW-E inference on CPU is too slow, at most 1 GPU-h.
- **Dependencies.** Proposal 1, with its stored correction for all five months. Experiment 652's
  code and banks.
- **Improves:** the prevailing forecast only.

### 0. The option of doing nothing more on FLOW-E

The reasons for stopping here:

- The follow gain is small. On bank M it is −0.006 against C and −0.021 against G0, and on bank S it
  is −0.008 against C. FLOW-E does nothing without a metered mirror (bank X equals G0).
- Eight of the 14 scored Primaries report MVA, which cannot show the direction of a flow.
- Real transfers can split across neighbours or go to Primaries off the roster.
- Two verified real episodes is too few to judge on.
- At 16 Primaries and 29 edges the electricity-network graph has little room, and fleet-scale
  adjacency (about 1,161 Primaries) has not been supplied.

The reasons against stopping:

- FLOW-E is the only arm of experiment 651 that held follow.
- Its measured ordinary-day cost now looks like a scoring artefact of one probable unlabelled event.
- FLOW-E is the "conservation along electrical edges" framing the roadmap already favours over
  learned message passing.
- Proposal 1 costs at most 6 GPU-h and settles the question.

**My recommendation: run proposal 1 only.** Run proposals 2 and 3 only if proposal 1 passes.

## Not a FLOW-E proposal, but changed by these analyses

The "0.13-point excess over A1" that experiment 651 pointed at the class 2 recipe (E3 rows, D5
offset, 50% signal dropout) is about 45% the D10 episode, and the rest is mostly D16. Re-score that
excess with proposal 1's screen on all five months before planning any recipe ablation. A recipe
ablation that is planned against the unscreened number could chase a scoring artefact.
