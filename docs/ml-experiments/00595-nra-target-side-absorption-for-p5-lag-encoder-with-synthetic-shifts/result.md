---
id: "00595"
slug: nra-target-side-absorption-for-p5-lag-encoder-with-synthetic-shifts
title: >-
  When a load-shift sits in the training power itself, how much of its depth does the fine-tuned
  demand network absorb into its median forecast if it reads power lags through the P5 recipe
  fine-tuned on synthetic lag-only shifts (C1), or through a shared lag encoder (A1)?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: refits of two arms on event-carrying power, forecasts and score tables in the
  private data folder only. One design is defensible (experiment 587's method and harness,
  unchanged), so the plan fixes only the arms, the quantity, and the decision rule. Spans code of
  experiments 574, 584, 587, and 591, which this experiment wraps without editing. Serving path
  and degradation rule: not touched (this repository holds none). Compute: under the 1 GPU-h
  ceiling (experiment 587's P5 run, 13 cases, took about 17 minutes).
hypothesis: >-
  The fine-tuned network of C1 (P5 plus lag-only synthetic shifts) and of A1 (shared lag encoder
  plus lag id) absorbs a share of an injected load-shift's depth into its median forecast, 7 days
  after the load-shift ends, that is no lower than plain P5's 0.219, because neither arm changes
  the targets that the refit sees.
prediction: >-
  Written before any score. P1: C1's total share (a shift in the training power and in the inputs,
  q50, day-14 refit, 7 days after the end, mean over the 12 injected designs) is at least 0.15. P1
  fails if it is 0.10 or below. P2: A1's total share is at least 0.15 as well. P3: if both are 0.10
  or below, the lag-following leak that the release bar calls L4 is no concern for C1 and A1.
decision_rule: >-
  All shares are descriptive, and the verdict reads P1 and P3 from the means. Intervals resample
  designs (both Primaries of one transfer in one block) with 2000 draws, and each printed
  half-width is the minimum detectable effect (MDE). Counted tests: 6 (C1 minus P5 and A1 minus P5,
  each for the first 4 days, day 4 to the end, and the 7 days after the end, total share at q50,
  day-14 refit, paired by design). A difference counts only if its Bonferroni interval excludes
  zero. C1 minus A1, the refit-only shares, the real D5/D16 pair, and the tail shares are
  descriptive. The experiment adopts nothing.
benchmark: >-
  Plain P5 (experiment 574) at 0.219 (0.173 to 0.264), M0 and D5, all from the stored forecasts of
  experiment 587. The corrected-target refit is the no-event world N, as in 587.
row_set_id: experiment_470_intersection_v1
metric: >-
  Share of the event's depth that the median forecast moves toward the event (sign times the
  difference of medians over depth) by phase (before, first 4 days, day 4 to the end, 7 days after
  the end), for the total effect and for the refit alone, at q50 and at q05, q95, q01, and q99.
---

**Plan only; no score has been read.** Experiment 587 measured the share of an injected load-shift
that a refit moves into the median forecast, for plain P5 (0.219, 0.173 to 0.264, 7 days after the
end). The release candidates are not plain P5: C1 adds lag-only synthetic shifts to the fine-tuning
inputs ([experiment 584](../00584-nra-candidate-pack-x6/result.md)) and A1 shares one lag encoder
across the lags ([experiment 591](../00591-shared-lag-encoder-auxiliary-shift-depth-loss/result.md)).
The question is whether those two arms leak a shift in the training power as much as plain P5 does.
If C1 behaves like P5, it misses the proposed bar L4 (total share at most 0.10 of the depth).

- **Arms.** C1 and A1, each fitted with the code of its own experiment (584, 591), unchanged except
  that the power stack is the injected world. Reference arms M0, D5, and P5 are read from experiment
  587's stored forecasts (same designs, folds, seeds, refit days, and scoring rows), so nothing is
  refitted for them.
- **Worlds and designs.** Experiment 587's 12 injected designs and the verified D5/D16 pair; for
  each, the event world E (power plus the shift, in the targets and the lag inputs) and the no-event
  world N. Total share = sign x (q[E] - q[N]) / depth; refit share uses the E-fitted weights on N's
  inputs. The two Primaries of a design are a donor and a recipient, which carry opposite shifts.
- **Schedule.** Coarse refits on day 0 and day 14 of each fold month; seeds 0 and 1; five
  development months; July 2026 unused. Weather from the run issued at or before the forecast origin
  minus 9 h; lag inputs at least 24 h old. Cleaned power; D15 and D18 are outside the 12 designs.
- **Evidence read before planning.** Experiments 574, 580, 584, 587, 590, 591, and 592 and the L4
  paragraph of the release-candidate review. No page merged since the brief changes the design.
  Experiment 594 (placebos) is in flight and does not touch this method.
- **Reports.** By phase, MDE and the Bonferroni count as numbers, the real D5/D16 pair descriptively,
  tail shares, and plots of the median against the no-event forecast for four designs.
- **Compute.** One GPU process at a time, `MemoryMax=8G`, `OMP_NUM_THREADS=1`; about 0.3 GPU-h for
  both arms. Every fit is saved as a `.pt` file under the experiment's data folder.
