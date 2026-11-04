---
id: "00545"
slug: modular-lag-free-demand-network-weather-and-datetime-encoders
title: >-
  Does splitting the lag-free demand network into a weather encoder, a datetime encoder, and a mixer,
  with or without auxiliary forecast heads, improve the day-ahead to five-day-ahead forecast?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted network weights, window forecasts, and stress-run forecasts under this
  experiment's private data folder only; no earlier experiment's files are written. Serving path:
  not touched (this repository holds none). Degradation rule: not touched. Designs: more than one
  defensible (which columns each encoder reads, where the auxiliary heads attach, how the fine-tuning
  mixer starts), so the plan fixes each before any score is read. Callers: this folder's scripts
  import experiments 397, 428, 448, 510, and 511. One Sonnet code review before the first fit and
  one Sonnet review of the write-up.
hypothesis: >-
  A lag-free demand network built from a weather encoder, a datetime encoder, and a mixer, where the
  mixer is replaced at fine-tuning by one that also reads ensemble spread and lead time, forecasts
  as well as the single multilayer network L-1d (M0). Adding an auxiliary forecast head after each
  encoder (M1) forces each encoder to carry information about the power distribution on its own,
  and so improves the forecast at 24 to 120 hours.
prediction: >-
  Written before any score is read. P1 (M0): M0 is within 0.05 points of CRPS of L-1d in every
  24-120 h band at the eight. P2 (M1): M1 beats L-1d by at least 0.05 points of CRPS in each of the
  24-48 h, 48-72 h, and 72-120 h bands at the eight. P2 may fail: experiments 524 and 528 suggest
  architecture changes are null. Stop rule: if M0 is more than 0.05 points worse than L-1d in any of
  those three bands, M1 is not run.
decision_rule: >-
  M1 is worth porting only if its paired contrast against L-1d has a Bonferroni interval excluding
  zero in the predicted direction and a point estimate of at most -0.05 in each of the three bands,
  with the injected-shift leak no more than 0.5 points of capacity above L-1d's 4.49. The 0.05
  threshold is set by the cost of porting a new network class and a new pretraining recipe, and sits
  below the minimum detectable effect (0.10 to 0.13 at the eight in these bands), so a pass reads
  "interval excludes zero" and never "at least 0.05".
benchmark: >-
  L-1d (experiment 511, stored) for every counted contrast; R-DW (experiment 539), A-pool (experiment
  425), and the rebuilt K2-S1 (experiment 541, when its predictions are stored) as report-only
  references.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521, at the eight
  MVA-only Primaries and the six signed-MW Primaries, with D15 and D18 report-only.
production_status: not-for-production
adopted_in: none
---

Planned. The plan is the front matter above and the section below; the write-up replaces this page
when the scores exist.

## Design fixed before any fit

- **Rows, folds, refits:** experiment 511's L-1d recipe. Five test months (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), a daily refit with a 90-day recency half-life, two seeds, ERA5 and CAMS
  pretraining restarted for each fine-tune. July 2026 is not scored and not tuned on.
- **Inputs:** the 37 lag-free columns of experiment 428. The weather encoder reads the 7
  ensemble-mean weather columns and the Primary embedding. The datetime encoder reads the 21 solar
  geometry, calendar, and holiday columns and the Primary embedding. The pretraining mixer reads
  both encoder outputs and the embedding. The fine-tuning mixer also reads the 7 ensemble-spread
  columns and the 2 lead columns, and starts from fresh weights (seed 12345).
- **Auxiliary heads (M1):** a small mixture head on each encoder output plus the embedding. Each
  head's negative log-likelihood has weight 0.25 beside the main mixture's, in pretraining and in
  fine-tuning. Early stopping scores the main mixture alone.
- **Standing rule on weather:** every weather input for a target comes from the run issued at or
  before the forecast origin; the fit script asserts it for every test origin and every training
  origin and prints one example row.
- **M2 (one pooled network over Primaries, bulk supply points, and grid supply points):** run only
  if the check shows clean series and weather rows for them; see the write-up.
- **Statistics:** month-block bootstrap with 5 blocks (near min-to-max), per-fold signs, the minimum
  detectable effect beside every contrast, and a Bonferroni count of 2 counted contrasts × 7 bands ×
  2 sets per arm.
