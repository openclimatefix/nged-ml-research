---
id: "00574"
slug: part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms
title: >-
  Which add-ons (multi-variable ensemble spread, thermal-lag and solar-gain columns, an expected
  signal from the ERA5/CAMS encoders, a three-step schedule) improve the modular lag-encoder network
  with the D5 output offset, and what does the combined final candidate score?
theme: lagged-power-and-error-correction
also_relevant_to:
  - differentiable-physics-long-history-and-drift
  - retraining-and-training-history
status: planned
kind: other
size: >-
  complex. Stored: fitted weights, forecasts, and scores in the private data folder only (final
  candidate weights kept). More than one defensible design: how the thermal-lag columns reach the
  lag encoder's expected embeddings, and how the spread columns join the lag network's mixer.
  Callers: the code copies and imports experiments 559, 560, 565, 568, 569, and 572 (more than two
  callers, found by search). Serving path and degradation rule: not touched (this repository holds
  none). Cost: ceiling 6 GPU-h (the manager's limit).
hypothesis: >-
  Four add-ons to the modular lag-encoder network with the D5 offset (P0) each lower CRPS: the
  multi-variable ensemble spread columns in the mixer (P1), thermal-lag and solar-gain columns
  through the weather encoder (P2), an expected signal for the lag encoder taken from encoders
  trained on ERA5 and CAMS only (P3), and a three-step schedule with a 3-day embargo (P4). The
  factors that meet the keep rule, plus P2 by the maintainer's policy, form a final candidate (P5)
  that stays within 0.04 of A-pool-24 at 24-120 h at the eight MVA Primaries.
prediction: >-
  Written before any fit. P0 beats E2 by at least 0.15 at 24-120 h at the eight MVA Primaries
  (fails under 0.08). P1 gains at least 0.03 at 24-120 h at one group, concentrated in the top
  spread tercile (fails within 0.02). P2 minus its placebo on cold winter nights at the six MW
  Primaries is at most -0.05 and no other slice is hurt (fails within the MDE). P3 lowers the
  real-event absorbed share against P0 by at least 0.005 with CRPS within the MDE (fails if the
  share rises). P4-E lowers the injected shift by at least 1.0 point of capacity and the absorbed
  share to at most 0.017, at a CRPS cost of at most 0.05 (fails over 0.10 or unchanged). P5 is
  within 0.04 of A-pool-24 at 24-120 h at the eight MVA Primaries, beats E2 by at least 0.15, has a
  lower absorbed share than P0, and no Primary is worse than L-1d by more than 1.0 point (fails if
  more than 0.08 behind A-pool-24 or any Primary is worse than L-1d by more than 2.0). P5 with lags
  aged one extra day keeps at least 60% of its lag gain (fails under 40%). Each leave-one-out
  removal costs at least the factor's slice MDE (fails otherwise). P6 (P5 plus W-T) adds at least
  0.05 on cold winter nights (fails within the MDE).
decision_rule: >-
  Keep rule of addendum 14 (part C of the slices brief): a factor is kept if it improves all rows
  at one group by at least that contrast's MDE with a Bonferroni interval excluding zero, or
  improves one pre-registered slice (cold winter nights, Christmas week, high-spread days) by at
  least that slice's MDE, and does not actively hurt another slice (worse than the control by more
  than the larger of 0.03 and the slice's MDE with the interval excluding zero on the harmful
  side, or the harmful sign in 4 of 5 folds). The leak (6.0 points) is a flag, not a gate. The
  threshold is the MDE because each factor adds a network part or a data input. Bonferroni count
  42: the four add-one contrasts x 2 groups x 3 lead bands (24) plus P5 against P0, E2, and
  A-pool-24 x 2 groups x 3 lead bands (18). A separate family of 2: P2 against its placebo on cold
  winter nights at each group. 5-block month bootstrap (near the minimum to maximum of the five
  folds); five folds of one sign give a sign-test p of 0.0625.
benchmark: >-
  P0 (experiment 572's v3+D5) for the add-one arms; E2 (experiment 565's D5), A-pool-24 (experiment
  558), L-1d (experiment 511), the climatology, and the manual heuristic as references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity in three lead bands at the eight MVA Primaries and the six
  MW Primaries (D15 and D18 report-only); leak: real-event absorbed share of depth (primary) and
  the injected 7-day shift (stress, flag at 6.0).
---

**Planned. Written before any fit.** The experiment follows the brief
`part_d_recipe_brief.md` (addendum 23) and the common rules `post_reset_common.md`.

## Plan

- **Base P0 is experiment 572's v3+D5, stored.** Its pre-training (ERA5 and CAMS rows, capped 24 h
  lags with block dropout, D5 monthly knots learnt) is
  [experiment 572's](../00572-two-offset-part-1-lags-versus-the-slow-output-offset/result.md)
  `pretrained_V3D5`, built on [experiment
  560's](../00560-modular-lag-encoder-capped-lags-and-block-dropout/result.md) v3 lag encoder and
  [experiment 565's](../00565-drift-embedding-variants-six-designs/result.md) D5 offset. The
  fine-tune is [experiment 559's](../00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty/result.md)
  one-step schedule (fresh mixer, frozen weather and datetime encoders, 90-day half-life). P0 needs
  no new fit; the add-one arms are fitted against its stored forecasts. E2 is experiment 565's D5.
- **Arms.** P1 adds the 7 spread columns of [experiment
  568](../00568-attention-mixer-modular-network-with-lead-and-spread-token/result.md) (T4) to the
  fine-tuning mixer. P2 passes [experiment 569's](../00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md)
  B1 columns through the weather encoder in pre-training and fine-tuning, with a shuffled-day
  placebo P2-plac of the same width. P3 gives the lag encoder's expected embeddings from a frozen
  copy of the ERA5/CAMS-only weather and datetime encoders. P4 is the three-step schedule (a fresh
  mixer on the whole ensemble history up to the cutoff minus 3.5 h without recency weighting, then
  the daily fine-tune), and P4-E adds the 3-day embargo of [experiment
  556](../00556-lag-free-finetune-absorption-of-load-shift-events-and-unlearn-time/result.md). P5,
  P6, leave-one-out arms, P7, and the daily harness follow the brief.
- **Design choices (stated before the code).** The B1 columns at a lag's own time are built from
  the same ERA5/CAMS (pre-training) or ensemble (fine-tuning) source as the target-time columns, so
  the weather encoder reads the same column set at both times. The spread block joins only the
  fine-tuning mixer, as in experiment 568's T4, because pre-training has no ensemble spread.
- **Timing asserts**, row by row with one printed example per arm and lead band: weather from the
  run issued at or before the origin minus 9 h; every lag at or before the origin minus 3.5 h (24 h
  for the base); the end time of each slow summary and offset knot is the refit cut.
- **Cost.** Experiment 572 measured 0.13 GPU-h for both seeds of one arm's fine-tuning, so the
  six-hour ceiling is far above the planned work. The first new arm's real cost is measured and the
  projection reported to the manager before the others start.
- **Reviews.** One fresh Sonnet review of the plan and code before the first fit; one Sonnet
  review of the write-up against the printed report.
